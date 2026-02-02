# plotting/best_model_selector.py
import numpy as np
import pandas as pd

_TIEBREAKS_DEFAULT = ('wrmse','ks','mae','ensemble')



import numpy as np
import pandas as pd

# round floats to a stable signature so keys match across CSV parsing noise
def _rf(x):
    return float(f"{float(x):.12g}")

def _key_from_ind(ind):
    # ind layout: [0..4]=discrete indices, [5..14]=continuous
    return (
        int(ind[0]), int(ind[1]), int(ind[2]), int(ind[3]), int(ind[4]),
        _rf(ind[5]), _rf(ind[6]), _rf(ind[7]), _rf(ind[8]), _rf(ind[9]),
        _rf(ind[10]), _rf(ind[11]), _rf(ind[12]), _rf(ind[13]), _rf(ind[14]),
    )

def _key_from_row(row: pd.Series):
    return (
        int(row['comp_idx']), int(row['imf_idx']), int(row['sn1a_idx']),
        int(row['sy_idx']), int(row['sn1ar_idx']),
        _rf(row['sigma_2']), _rf(row['t_1']), _rf(row['t_2']),
        _rf(row['infall_1']), _rf(row['infall_2']),
        _rf(row['sfe']), _rf(row['delta_sfe']),
        _rf(row['imf_upper']), _rf(row['mgal']), _rf(row['nb']),
    )

def truncate_results_to_history(GalGA, df: pd.DataFrame) -> pd.DataFrame:
    # build keyset from the currently-loaded GalGA.results (these are the walker_history reconstructions in plot-only)
    res = getattr(GalGA, 'results', [])
    if not isinstance(df, pd.DataFrame) or len(df) == 0 or len(res) == 0:
        return df
    ks = {_key_from_ind(r[:15]) for r in res}
    mask = df.apply(lambda row: _key_from_row(row) in ks, axis=1)
    out = df.loc[mask].reset_index(drop=True)
    return out





def _stable_best_row_index(results_df: pd.DataFrame,
                           primary: str = 'fitness',
                           tiebreaks=_TIEBREAKS_DEFAULT) -> int:
    if results_df is None or results_df.empty:
        raise ValueError("results_df is empty")

    df = results_df.copy()
    cols = [c for c in (primary,)+tuple(tiebreaks) if c in df.columns]
    if not cols:
        # Fall back to original order deterministically
        return int(df.index[0])

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Stable sort, NaNs -> worst
    df['__ord__'] = np.arange(len(df))
    idx = df.sort_values(cols+['__ord__'], na_position='last', kind='mergesort').index[0]
    return int(idx)

def _map_row_to_model_idx(GalGA, row: pd.Series) -> int:
    """
    IMPROVED VERSION: Uses ALL 15 parameters for robust matching.
    
    This is necessary because the CSV is sorted by fitness but GalGA arrays
    are in evaluation order. We need to find which evaluation produced this
    set of parameters.
    """
    # Preferred: explicit column (if it exists)
    if 'model_idx' in row and pd.notna(row['model_idx']):
        mi = int(row['model_idx'])
        n = len(getattr(GalGA, 'results', []))
        if mi < 0 or mi >= n:
            raise ValueError(f"model_idx {mi} out of bounds for results length {n}")
        return mi

    # ========== IMPROVED: Use ALL 15 parameters for matching ==========
    # Extract all parameters from the row
    try:
        row_params = (
            int(row['comp_idx']), int(row['imf_idx']), int(row['sn1a_idx']),
            int(row['sy_idx']), int(row['sn1ar_idx']),
            float(row['sigma_2']), float(row['t_1']), float(row['t_2']),
            float(row['infall_1']), float(row['infall_2']),
            float(row['sfe']), float(row['delta_sfe']),
            float(row['imf_upper']), float(row['mgal']), float(row['nb'])
        )
    except KeyError as e:
        raise ValueError(f"Missing required parameter column: {e}")

    # 1) Try exact match on ALL parameters
    for i, r in enumerate(GalGA.results):
        # Check discrete parameters (indices 0-4)
        if (int(r[0]) != row_params[0] or int(r[1]) != row_params[1] or 
            int(r[2]) != row_params[2] or int(r[3]) != row_params[3] or 
            int(r[4]) != row_params[4]):
            continue
        
        # Check continuous parameters (indices 5-14) with exact match
        if (float(r[5]) == row_params[5] and float(r[6]) == row_params[6] and
            float(r[7]) == row_params[7] and float(r[8]) == row_params[8] and
            float(r[9]) == row_params[9] and float(r[10]) == row_params[10] and
            float(r[11]) == row_params[11] and float(r[12]) == row_params[12] and
            float(r[13]) == row_params[13] and float(r[14]) == row_params[14]):
            return i

    # 2) Epsilon-based matching with ALL parameters
    # Use relative tolerance for better floating-point comparison
    rtol = 1e-9  # Relative tolerance
    atol = 1e-12  # Absolute tolerance
    
    best_match = None
    best_distance = float('inf')
    
    for i, r in enumerate(GalGA.results):
        # First check discrete parameters - must match exactly
        if (int(r[0]) != row_params[0] or int(r[1]) != row_params[1] or 
            int(r[2]) != row_params[2] or int(r[3]) != row_params[3] or 
            int(r[4]) != row_params[4]):
            continue
        
        # Calculate normalized distance for continuous parameters
        # Use relative error to handle different scales
        distance = 0.0
        for j in range(5, 15):
            expected = row_params[j]
            actual = float(r[j])
            
            # Relative error with absolute fallback for near-zero values
            if abs(expected) > atol:
                rel_err = abs(actual - expected) / abs(expected)
            else:
                rel_err = abs(actual - expected)
            
            distance += rel_err
        
        # Keep track of closest match
        if distance < best_distance:
            best_distance = distance
            best_match = i
    
    # Accept match if distance is small enough
    # With 10 continuous parameters, total relative error should be < 1e-8
    if best_match is not None and best_distance < 1e-7:
        return best_match
    
    # 3) If still no match, provide detailed error message
    print("\n" + "="*80)
    print("ERROR: Could not find matching model in GalGA.results")
    print("="*80)
    print(f"Looking for parameters:")
    print(f"  Discrete: comp={row_params[0]}, imf={row_params[1]}, sn1a={row_params[2]}, sy={row_params[3]}, sn1ar={row_params[4]}")
    print(f"  Continuous: sigma_2={row_params[5]:.6f}, t_1={row_params[6]:.6f}, t_2={row_params[7]:.6f}")
    print(f"              infall_1={row_params[8]:.6f}, infall_2={row_params[9]:.6f}")
    print(f"              sfe={row_params[10]:.8f}, delta_sfe={row_params[11]:.6f}")
    print(f"              imf_upper={row_params[12]:.2f}, mgal={row_params[13]:.4e}, nb={row_params[14]:.4e}")
    
    if best_match is not None:
        print(f"\nClosest match was index {best_match} with distance {best_distance:.2e}")
        r = GalGA.results[best_match]
        print(f"  Its parameters:")
        print(f"  Discrete: comp={int(r[0])}, imf={int(r[1])}, sn1a={int(r[2])}, sy={int(r[3])}, sn1ar={int(r[4])}")
        print(f"  Continuous: sigma_2={float(r[5]):.6f}, t_1={float(r[6]):.6f}, t_2={float(r[7]):.6f}")
        print(f"              infall_1={float(r[8]):.6f}, infall_2={float(r[9]):.6f}")
        print(f"              sfe={float(r[10]):.8f}, delta_sfe={float(r[11]):.6f}")
        print(f"              imf_upper={float(r[12]):.2f}, mgal={float(r[13]):.4e}, nb={float(r[14]):.4e}")
    
    print(f"\nTotal models in GalGA.results: {len(GalGA.results)}")
    print("="*80 + "\n")
    
    raise ValueError("Cannot map best row to GalGA.results - parameter mismatch too large")


import numpy as np
import pandas as pd

def assert_galga_alignment(GalGA) -> bool:
    n = len(getattr(GalGA, 'results', []))
    issues = []
    for name in ('mdf_data', 'age_data', 'alpha_data'):
        seq = getattr(GalGA, name, None)
        if seq is None or len(seq) != n:
            issues.append(f"{name} length {len(seq) if seq is not None else 'None'} != results length {n}")
    if issues:
        print("[warn][alignment]", "; ".join(issues))
        return False
    return True

def _soft_map_row_to_model_idx(GalGA, row, rtol=1e-8, atol=1e-6):
    """Try to find the row's model in GalGA.results without assuming equal lengths."""
    if not hasattr(GalGA, 'results') or not GalGA.results:
        return None

    names_disc = ['comp_idx','imf_idx','sn1a_idx','sy_idx','sn1ar_idx']
    names_cont = ['sigma_2','t_1','t_2','infall_1','infall_2','sfe','delta_sfe','imf_upper','mgal','nb']

    # pre-extract row values (None if missing)
    row_disc = []
    for n in names_disc:
        row_disc.append(None if n not in row.index or pd.isna(row[n]) else int(row[n]))
    row_cont = []
    for n in names_cont:
        row_cont.append(None if n not in row.index or pd.isna(row[n]) else float(row[n]))

    for i, r in enumerate(GalGA.results):
        ok = True
        # compare discrete
        for j, rv in enumerate(row_disc):
            if rv is None:
                continue
            if int(r[j]) != rv:
                ok = False; break
        if not ok:
            continue
        # compare continuous (r[5:15] vs row_cont)
        for k, rv in enumerate(row_cont):
            if rv is None:
                continue
            if not np.isclose(float(r[5+k]), rv, rtol=rtol, atol=atol):
                ok = False; break
        if ok:
            return i
    return None

def get_best_model_index(GalGA, results_df: pd.DataFrame,
                         primary: str = 'fitness',
                         tiebreaks=_TIEBREAKS_DEFAULT) -> tuple[int, int]:
    """Return (idx_in_GalGA.results or -1 if not found, best_row_index_in_results_df)."""
    # soft check only; don't explode on mismatch
    _ = assert_galga_alignment(GalGA)

    row_idx = _stable_best_row_index(results_df, primary, tiebreaks)
    row = results_df.loc[row_idx]

    # Use improved mapping function
    try:
        model_idx = _map_row_to_model_idx(GalGA, row)
    except ValueError as e:
        print(f"Warning: {e}")
        model_idx = -1
    
    return model_idx, row_idx




def get_best_from_truncated_df(results_df: pd.DataFrame,
                               primary: str = 'fitness',
                               tiebreaks=_TIEBREAKS_DEFAULT) -> tuple[int, int]:
    """
    Return (history_curve_index, best_row_index_in_results_df_trunc).
    Assumes results_df has '_hist_idx' created by truncation step.
    """
    row_idx = _stable_best_row_index(results_df, primary, tiebreaks)
    row = results_df.loc[row_idx]
    if "_hist_idx" not in row.index:
        raise RuntimeError("Truncated results_df missing _hist_idx; mapping step not applied.")
    hist_idx = int(row["_hist_idx"])
    return hist_idx, row_idx
