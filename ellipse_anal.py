#!/usr/bin/env python3
"""
Standalone script to select folders, group their CSV files by generation number,
compute the semimajor axis of the HPD ellipse for a specific parameter combo,
and plot it versus the number of generations with increased density from multiple folders.
"""

import os
import re
import glob
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from collections import defaultdict

# ---- UncertaintyAnalysis class (minimal version for this script) ----
class UncertaintyAnalysis:
    def __init__(self, results_file, output_path='SMC_DEMC/'):
        self.results_file = results_file
        self.output_path = output_path
        self.df = pd.read_csv(results_file)
        self.fitness_col = 'fitness' if 'fitness' in self.df.columns else 'wrmse'
        self.continuous_params = [
            'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2', 
            'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb'
        ]
        self.continuous_params = [p for p in self.continuous_params if p in self.df.columns]
        self.df_sorted = self.df.sort_values(self.fitness_col, ascending=True)

# ---- Helper functions ----
def _discover_result_folders(base_dir):
    """Return [(folder_name, folder_path, [csvs...])] for folders that contain simulation_results*.csv"""
    out = []
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if not os.path.isdir(p) or name.startswith('.'):
            continue
        csvs = sorted(glob.glob(os.path.join(p, "simulation_results*.csv")))
        if csvs:
            out.append((name, p, csvs))
    return out

def _parse_suffix_from_name(fname):
    # prefer _gen_<N>.csv, else _<N>.csv, else None
    m = re.search(r'_gen_(\d+)\.csv$', fname)
    if m: return int(m.group(1))
    m = re.search(r'_(\d+)\.csv$', fname)
    if m: return int(m.group(1))
    return None

def _union_param_ranges(analyzers, params):
    """Union [lo,hi] across analyzers; fallback to data min/max."""
    union = {}
    for p in params:
        los, his = [], []
        for a in analyzers:
            if p in a.df.columns:
                v = a.df[p].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if v.size:
                    los.append(float(np.min(v)))
                    his.append(float(np.max(v)))
        if los:
            union[p] = (min(los), max(his))
    return union

def _select_by_cutoff_or_percentile(a, cutoff, fallback_percentile=100, weight_power=1.0):
    """Return (df_selected, weights) using cutoff if present; else top percentile."""
    col = a.fitness_col
    df = a.df_sorted
    if isinstance(cutoff, (int, float)):
        sel = df[df[col] <= cutoff]
    else:
        sel = None
    if sel is None or len(sel) == 0:
        n_top = max(1, int(len(df) * fallback_percentile / 100))
        sel = df.head(n_top)
    fit = np.asarray(sel[col].values, float)
    eps = np.nanmin(fit) * 1e-3 if np.isfinite(np.nanmin(fit)) else 1e-12
    w = 1.0 / np.power(fit + eps, weight_power)
    w = w / np.sum(w)
    return sel, w

def _combined_top_selection(analyzers, params, percentile=100, weight_power=1.0):
    frames, w_all = [], []
    for a in analyzers:
        # Use None for cutoff to fallback to percentile
        t, w = _select_by_cutoff_or_percentile(a, cutoff=None, fallback_percentile=percentile, weight_power=weight_power)
        if not t.empty:
            keep_cols = [p for p in params if p in t.columns]
            frames.append(t[keep_cols].copy())
            w_all.append(w)
    if not frames:
        return pd.DataFrame(columns=params), np.array([])
    W = np.concatenate(w_all)
    return pd.concat(frames, axis=0, ignore_index=True), W

def _kde_2d(x, y, weights, Xg, Yg):
    x = np.asarray(x, float); y = np.asarray(y, float)
    weights = np.asarray(weights, float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights)
    x = x[good]; y = y[good]; weights = weights[good]
    if len(x) == 0: return np.zeros_like(Xg)
    xy = np.vstack([x, y])
    try:
        kde = gaussian_kde(xy, weights=weights)
    except Exception:
        import numpy.random as rng
        rng.seed(1337)
        N = min(max(1000, 5 * len(x)), 5000)
        p = weights / np.sum(weights) if np.sum(weights) > 0 else None
        idx = rng.choice(np.arange(len(x)), size=N, replace=True, p=p)
        kde = gaussian_kde(np.vstack([x[idx], y[idx]]))
    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
    return Z

def _hpd_threshold(Z, p, dx, dy):
    z = Z.ravel()
    if not np.isfinite(z).any() or z.max() <= 0:
        return None
    order = np.argsort(z)[::-1]
    z_sorted = z[order]
    csum = np.cumsum(z_sorted) * dx * dy
    total = z.sum() * dx * dy
    if total <= 0: return None
    idx = np.searchsorted(csum, p * total)
    idx = np.clip(idx, 0, z_sorted.size - 1)
    return float(z_sorted[idx])

def _ellipse_from_hpd(Xg, Yg, Z, p=0.68):
    dx = float(Xg[0, 1] - Xg[0, 0]); dy = float(Yg[1, 0] - Yg[0, 0])
    thr = _hpd_threshold(Z, p, dx, dy)
    if thr is None: return None
    mask = Z >= thr
    xs = Xg[mask]; ys = Yg[mask]
    if xs.size < 8: return None
    mx, my = float(xs.mean()), float(ys.mean())
    Xc = np.vstack([xs - mx, ys - my])
    C = (Xc @ Xc.T) / xs.size
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]; evecs = evecs[:, order]
    a = 2.0 * np.sqrt(max(evals[0], 0.0))
    b = 2.0 * np.sqrt(max(evals[1], 0.0))
    theta = float(np.arctan2(evecs[1, 0], evecs[0, 0]))
    return {"thr": thr, "center": (mx, my), "a": float(a), "b": float(b), "theta": theta, "evecs": evecs}

# ---- Main logic ----
if __name__ == "__main__":
    current_dir = os.getcwd()
    found = _discover_result_folders(current_dir)
    if not found:
        print(f"No result folders found under {current_dir}")
        sys.exit(0)

    print(f"\nFound {len(found)} candidate folders:")
    for idx, (name, path, csvs) in enumerate(found, start=1):
        primary = csvs[0] if csvs else None  # Just show first for listing
        print(f"  [{idx:>2}] {name:30s}  CSVs: {len(csvs)}")

    sel = input("\nEnter indices to use (e.g., 1 2 3 or 1,2,3). Press Enter for ALL: ").strip()

    # default-to-all semantics
    if not sel or sel.lower() in {"all", "a", "*"}:
        idxs = list(range(1, len(found) + 1))
        print(f"Selected ALL {len(found)} folders.")
    else:
        # parse indices
        tokens = re.split(r'[,\s]+', sel)
        try:
            idxs = sorted(set(int(t) for t in tokens if t))
        except ValueError:
            print("Invalid input. Use numbers separated by space/comma or 'all'.")
            sys.exit(1)

    # validate & build selection
    chosen = []
    invalid = []
    for i in idxs:
        if 1 <= i <= len(found):
            chosen.append(found[i-1])
        else:
            invalid.append(i)

    if invalid:
        print(f"Ignoring out-of-range indices: {invalid}")

    if not chosen:
        print("No valid indices chosen. Exiting.")
        sys.exit(0)

    # Now, collect ALL CSVs from selected folders and group by gen
    groups = defaultdict(list)  # gen -> list of (csv_path, analyzer? wait, collect paths first)
    for name, path, csvs in chosen:
        for csv in csvs:
            gen = _parse_suffix_from_name(os.path.basename(csv))
            if gen is not None:
                groups[gen].append(csv)

    if not groups:
        print("No CSVs with parsable generation numbers in selected folders.")
        sys.exit(0)

    # Parameters for ellipse (change as needed)
    param1 = 'sigma_2'
    param2 = 't_2'
    params = [param1, param2]
    p_hpd = 0.68
    grid_n = 240
    percentile = 100
    weight_power = 1.0

    # Compute semimajor axis for each group
    gen_numbers = sorted(groups.keys())
    semimajors = []
    for gen in gen_numbers:
        csv_list = groups[gen]
        if not csv_list:
            continue
        if len(csv_list) > 1:
            analyzers = []
            for csv_path in csv_list:
                out_root = os.path.dirname(csv_path) + os.sep  # Minimal output path
                a = UncertaintyAnalysis(results_file=csv_path, output_path=out_root)
                analyzers.append(a)

            # Get combined data
            T, w_comb = _combined_top_selection(analyzers, params, percentile=percentile, weight_power=weight_power)

            if T.empty or len(T) < 8:
                print(f"Skipping gen {gen}: insufficient combined data points ({len(T)}).")
                continue

            # Union ranges
            param_ranges = _union_param_ranges(analyzers, params)

            # Grid
            lo_x, hi_x = param_ranges.get(param1, (np.nanmin(T[param1]), np.nanmax(T[param1])))
            lo_y, hi_y = param_ranges.get(param2, (np.nanmin(T[param2]), np.nanmax(T[param2])))
            if not (np.isfinite(lo_x) and np.isfinite(hi_x) and hi_x > lo_x and
                    np.isfinite(lo_y) and np.isfinite(hi_y) and hi_y > lo_y):
                print(f"Skipping gen {gen}: invalid parameter ranges.")
                continue
            xg = np.linspace(lo_x, hi_x, grid_n)
            yg = np.linspace(lo_y, hi_y, grid_n)
            Xg, Yg = np.meshgrid(xg, yg)

            # Combined KDE
            xC = T[param1].to_numpy(float)
            yC = T[param2].to_numpy(float)
            goodC = np.isfinite(xC) & np.isfinite(yC) & np.isfinite(w_comb)
            xC = xC[goodC]; yC = yC[goodC]; wC = w_comb[goodC]
            if xC.size < 8:
                print(f"Skipping gen {gen}: insufficient finite points after filtering.")
                continue
            Zc = _kde_2d(xC, yC, wC, Xg, Yg)
            Zc = np.nan_to_num(Zc, nan=0.0, posinf=0.0, neginf=0.0)

            # Ellipse
            el = _ellipse_from_hpd(Xg, Yg, Zc, p=p_hpd)
            if el is None:
                print(f"Skipping gen {gen}: could not compute ellipse.")
                continue
            semimajors.append(el['a'])
            print(f"Gen {gen} ({len(csv_list)} CSVs): semimajor axis = {el['a']:.4f}")

    if not semimajors:
        print("No valid ellipses computed.")
        sys.exit(0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gen_numbers, semimajors, marker='o', linestyle='-', color='b')
    ax.set_xlabel('Number of Generations')
    ax.set_ylabel('Semimajor Axis of Ellipse')
    ax.set_title(f'Ellipse Semimajor Axis vs Generations ({param1} vs {param2})')
    ax.grid(True)

    save_path = os.path.join(current_dir, "analysis", "semimajor_vs_generations.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {save_path}")