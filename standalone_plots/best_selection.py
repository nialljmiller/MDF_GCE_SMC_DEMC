# standalone_plots/best_selection.py
import numpy as np
import pandas as pd

_TIEBREAKS_DEFAULT = ("wrmse", "ks", "mae", "ensemble")

def _rf(x):
    return float(f"{float(x):.12g}")

def stable_best_index(df: pd.DataFrame,
                      primary: str = "loss",
                      tiebreaks=_TIEBREAKS_DEFAULT) -> int:
    if df is None or df.empty:
        raise ValueError("results_df is empty")

    tmp = df.copy()
    cols = [c for c in (primary,) + tuple(tiebreaks) if c in tmp.columns]
    if not cols:
        return int(tmp.index[0])

    for c in cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    tmp["__ord__"] = np.arange(len(tmp))
    idx = tmp.sort_values(cols + ["__ord__"],
                          na_position="last",
                          kind="mergesort").index[0]
    return int(idx)





import numpy as np
import pandas as pd

# columns that define the parameter vector
PARAM_COLUMNS = [
    "sigma_2","t_1","t_2","infall_1","infall_2",
    "sfe","delta_sfe","imf_upper","mgal","nb"
]

def map_best_index(df: pd.DataFrame,
                      primary: str = "loss",
                      weight_col: str = "posterior_w",
                      bins: int = 50) -> int:
    cols = [c for c in PARAM_COLUMNS if c in df.columns]
    if weight_col in df.columns:
        w = df[weight_col].to_numpy(dtype=float)
        map_vec = []
        for c in cols:
            x = df[c].to_numpy(dtype=float)
            hist, edges = np.histogram(x, bins=bins, weights=w)
            k = np.argmax(hist)
            x_lo, x_hi = edges[k], edges[k+1]
            map_vec.append(0.5*(x_lo + x_hi))
        map_vec = np.array(map_vec, float)

        X = df[cols].to_numpy(dtype=float)
        q25 = np.nanpercentile(X, 25, axis=0)
        q75 = np.nanpercentile(X, 75, axis=0)
        scale = (q75 - q25)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0

        Z = (X - map_vec) / scale
        d2 = np.sum(Z*Z, axis=1)
        idx = int(np.argmin(d2))
        return int(df.index[idx])

    tmp = df.copy()
    tmp["__ord__"] = np.arange(len(tmp))
    cols_sort = [c for c in [primary] if c in tmp.columns] + ["__ord__"]
    idx = tmp.sort_values(cols_sort, kind="mergesort").index[0]
    return int(idx)
