# standalone_plots/ensembles.py
import numpy as np
import pandas as pd






# --------- generic helpers ---------
def _weighted_quantile(x, w, qs):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    out = []
    for q in qs:
        idx = np.searchsorted(cdf, q)
        if idx <= 0:
            out.append(x[0])
        elif idx >= len(x):
            out.append(x[-1])
        else:
            out.append(x[idx])
    return np.array(out)




def _extract_xy_pair_by_bases(row, bases):
    """Try bases like 'mdf', 'MDF', 'mdf_track' → look for base_x/base_y or packed [x,y]."""
    idx = row.index

    # 1) look for exact base_x/base_y
    for base in bases:
        xk, yk = f"{base}_x", f"{base}_y"
        if xk in idx and yk in idx:
            return np.asarray(row[xk], float), np.asarray(row[yk], float)

    # 2) look for packed single column: list/tuple of [x, y] or dict {'x','y'}
    for base in bases:
        if base in idx:
            v = row[base]
            # [x, y] or (x, y)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                return np.asarray(v[0], float), np.asarray(v[1], float)
            # dict-like
            if isinstance(v, dict):
                if "x" in v and "y" in v:
                    return np.asarray(v["x"], float), np.asarray(v["y"], float)

    # 3) scan for any unique *_x/*_y pair containing the base token (case-insensitive)
    lower = {c.lower(): c for c in idx}
    for base in bases:
        token = base.lower()
        x_cands = [orig for low, orig in lower.items() if low.endswith("_x") and token in low]
        y_cands = [orig for low, orig in lower.items() if low.endswith("_y") and token in low]
        # choose pair with matching prefixes before the _x/_y
        pairs = []
        for xk in x_cands:
            prefix = xk[:-2]
            yk = prefix + "_y"
            if yk in idx:
                pairs.append((xk, yk))
        if len(pairs) == 1:
            xk, yk = pairs[0]
            return np.asarray(row[xk], float), np.asarray(row[yk], float)

    raise KeyError(f"Could not resolve XY pair for bases={bases}")









def _extract_mdf_xy(row):
    # try common variants
    return _extract_xy_pair_by_bases(row, bases=("mdf", "MDF", "mdf_track", "mdf_curve"))










# --------- ensemble builders ---------
def build_mdf_ensemble(df: pd.DataFrame, weights, feh_grid):
    feh_grid = np.asarray(feh_grid, float)
    weights = np.asarray(weights, float)
    n = len(df)
    y_stack = np.zeros((n, feh_grid.size), dtype=float)

    for i, (_, row) in enumerate(df.iterrows()):
        x, y = _extract_mdf_xy(row)
        y_interp = np.interp(feh_grid, x, y, left=np.nan, right=np.nan)
        y_stack[i] = y_interp

    mask = np.isfinite(y_stack)
    median = np.zeros_like(feh_grid)
    lo16 = np.zeros_like(feh_grid)
    hi84 = np.zeros_like(feh_grid)

    for j in range(feh_grid.size):
        vals = y_stack[:, j]
        m = mask[:, j]
        if not np.any(m):
            median[j] = np.nan; lo16[j] = np.nan; hi84[j] = np.nan
        else:
            v = vals[m]; w = weights[m]
            lo16[j], median[j], hi84[j] = _weighted_quantile(v, w, [0.16, 0.5, 0.84])

    return median, lo16, hi84













def _extract_age_xy(row):
    # Expect: row["age_x"] in years (monotonic but direction unknown), row["age_y"] in [Fe/H]
    x = np.asarray(row["age_x"], float)
    y = np.asarray(row["age_y"], float)

    # Convert to "Age since start" in Gyr, and sort ascending for interp
    # Age(t) = (t_end - t) / 1e9
    t_end = x[-1]  # last sample is final time in years in your data layout
    age_gyr = (t_end - x) / 1e9

    order = np.argsort(age_gyr)          # enforce ascending for np.interp
    return age_gyr[order], y[order]


def build_age_feh_ensemble(df: pd.DataFrame, weights, age_grid_gyr):
    age_grid_gyr = np.asarray(age_grid_gyr, float)
    w_all = np.asarray(weights, float)

    # mask non-finite or non-positive weights (doesn't change quantiles except for degenerate cases)
    good_w = np.isfinite(w_all) & (w_all > 0)
    if not np.any(good_w):
        raise ValueError("All posterior weights are non-positive or NaN.")
    w_all = w_all.copy()
    w_all[~good_w] = 0.0

    y_stack = np.full((len(df), age_grid_gyr.size), np.nan, dtype=float)

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            x_age, y_feh = _extract_age_xy(row)   # already Gyr ascending
            y_interp = np.interp(age_grid_gyr, x_age, y_feh, left=np.nan, right=np.nan)
            y_stack[i] = y_interp
        except Exception:
            # leave this row as NaNs
            continue

    median = np.full_like(age_grid_gyr, np.nan, dtype=float)
    lo16  = np.full_like(age_grid_gyr, np.nan, dtype=float)
    hi84  = np.full_like(age_grid_gyr, np.nan, dtype=float)

    mask = np.isfinite(y_stack)
    for j in range(age_grid_gyr.size):
        m = mask[:, j]
        if not np.any(m):
            continue
        v = y_stack[m, j]
        w = w_all[m]
        if np.sum(w) <= 0:
            continue
        lo16[j], median[j], hi84[j] = _weighted_quantile(v, w, [0.16, 0.5, 0.84])

    return median, lo16, hi84





def _extract_alpha_xy(row, k):
    """
    Prefer 'alpha_tracks' packed as list/tuple length-4 of [x,y] entries.
    Fallback: per-element columns like 'alpha_si_x', 'alpha_si_y', etc.
    Order (k): 0..3 → Si, Ca, Mg, Ti (adjust if your order differs).
    """
    idx = row.index
    # packed alpha_tracks
    if "alpha_tracks" in idx and row["alpha_tracks"] is not None:
        tracks = row["alpha_tracks"]
        ax, ay = tracks[k]
        return np.asarray(ax, float), np.asarray(ay, float)

    # fallback by names
    names = [("si",), ("ca",), ("mg",), ("ti",)]
    tag = names[k][0]
    bases = (f"alpha_{tag}", f"{tag}_alpha", f"{tag}_fe")
    return _extract_xy_pair_by_bases(row, bases=bases)

def build_alpha_ensemble(df: pd.DataFrame, weights, feh_grid, alpha_index):
    feh_grid = np.asarray(feh_grid, float)
    weights = np.asarray(weights, float)
    n = len(df)
    y_stack = np.zeros((n, feh_grid.size), dtype=float)

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            ax, ay = _extract_alpha_xy(row, alpha_index)
            y_interp = np.interp(feh_grid, ax, ay, left=np.nan, right=np.nan)
        except Exception:
            y_interp = np.full_like(feh_grid, np.nan)
        y_stack[i] = y_interp

    mask = np.isfinite(y_stack)
    median = np.zeros_like(feh_grid)
    lo16 = np.zeros_like(feh_grid)
    hi84 = np.zeros_like(feh_grid)

    for j in range(feh_grid.size):
        vals = y_stack[:, j]
        m = mask[:, j]
        if not np.any(m):
            median[j] = np.nan; lo16[j] = np.nan; hi84[j] = np.nan
        else:
            v = vals[m]; w = weights[m]
            lo16[j], median[j], hi84[j] = _weighted_quantile(v, w, [0.16, 0.5, 0.84])

    return median, lo16, hi84






