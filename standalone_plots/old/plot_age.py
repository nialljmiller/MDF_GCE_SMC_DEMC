import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .best_selection import *
from .ensembles import build_age_feh_ensemble
from .obs import load_observed_amr  # consistent with MDF design
from .ensembles import _extract_mdf_xy

from matplotlib import gridspec
from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors


# --- helpers ---
def _binned_xy(x, y, bins):
    m = np.isfinite(x) & np.isfinite(y)
    means, _, _ = binned_statistic(x[m], y[m], statistic="mean", bins=bins)
    stds,  _, _ = binned_statistic(x[m], y[m], statistic="std",  bins=bins)
    cnts,  _, _ = binned_statistic(x[m], y[m], statistic="count", bins=bins)
    ctrs = 0.5 * (bins[:-1] + bins[1:])
    sem = stds / np.sqrt(np.maximum(cnts, 1))
    ok = (cnts > 0) & np.isfinite(means)
    return ctrs[ok], means[ok], stds[ok], sem[ok]

def _smoothed_hist(vals, bins, sigma_bins=1.2):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    counts, edges = np.histogram(v, bins=bins)
    counts = counts.astype(float)
    counts_s = gaussian_filter1d(counts, sigma=sigma_bins, mode="nearest")
    if counts_s.max() > 0:
        counts_s /= counts_s.max()
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts_s

def _interp_clean(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    keep = np.ones_like(xs, dtype=bool)
    keep[1:] = (np.diff(xs) > 1e-12)
    return xs[keep], ys[keep]

def _weighted_quantile(values, weights, q):
    # q in [0..1], vectorized
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    s = np.argsort(v)
    v, w = v[s], w[s]
    cw = np.cumsum(w)
    cw /= cw[-1] if cw[-1] != 0 else 1.0
    return np.interp(q, cw, v)

# --------------------------------------------------------------------------------
# Detailed best-model plot with residuals + sideways histogram
# --------------------------------------------------------------------------------
def plot_age(
    df: pd.DataFrame,
    output_dir: str,
    n_bins=12,
    feh_bins=None,
    age_limit_gyr=16.0,
    top_overlay=99999999999999,
    loss_metric = "loss"
):
    os.makedirs(output_dir, exist_ok=True)

    best_idx = stable_best_index(df, primary=loss_metric)
    best_row = df.loc[best_idx]
    Fe_H, age_Joyce, age_Bensby = load_observed_amr()

    if feh_bins is None:
        feh_bins = np.linspace(-2.0, 1.0, 28)

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[4, 1],
                           height_ratios=[3, 1],
                           wspace=0.0, hspace=0.0,
                           left=0.07, right=0.97, top=0.96, bottom=0.08)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_res  = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_side = fig.add_subplot(gs[0, 1], sharey=ax_main)

    # faint background overlays (top by posterior_w)
    order = np.argsort(df["posterior_w"].to_numpy())[::-1][:min(top_overlay, len(df))]
    for i in order:
        row = df.iloc[i]
        x = np.asarray(row["age_x"], float)
        y = np.asarray(row["age_y"], float)
        x = (x[-1] - x) / 1e9
        ax_main.plot(x, y, alpha=0.07, lw=0.8, color="0.6", zorder=1)

    # best track
    bx = np.asarray(best_row["age_x"], float)
    by = np.asarray(best_row["age_y"], float)
    bx = (bx[-1] - bx) / 1e9
    ax_main.plot(bx, by, color="crimson", lw=2.5, zorder=5, label="Best model")

    # observations (raw)
    mJ = np.isfinite(age_Joyce) & np.isfinite(Fe_H)
    mB = np.isfinite(age_Bensby) & np.isfinite(Fe_H)
    ax_main.scatter(age_Joyce[mJ], Fe_H[mJ], marker="*", s=55, color="tab:red", alpha=0.8, zorder=6, label="Joyce (raw)")
    ax_main.scatter(age_Bensby[mB], Fe_H[mB], marker="^", s=55, color="tab:blue", alpha=0.8, zorder=6, label="Bensby (raw)")

    # binned obs
    age_bins = np.linspace(0.0, age_limit_gyr, n_bins + 1)
    Jc, Jm, Js, _ = _binned_xy(age_Joyce, Fe_H, age_bins)
    Bc, Bm, Bs, _ = _binned_xy(age_Bensby, Fe_H, age_bins)
    ax_main.plot(Jc, Jm, color="tab:red", lw=2.0, zorder=7, label="Joyce (binned)")
    ax_main.errorbar(Jc, Jm, yerr=Js, color="tab:red", alpha=0.35, lw=1.0, capsize=3, zorder=6)
    ax_main.plot(Bc, Bm, color="tab:blue", lw=2.0, zorder=7, label="Bensby (binned)")
    ax_main.errorbar(Bc, Bm, yerr=Bs, color="tab:blue", alpha=0.35, lw=1.0, capsize=3, zorder=6)

    # residuals vs best model
    xs, ys = _interp_clean(bx, by)
    f_best = interp1d(xs, ys, kind="linear", bounds_error=False, fill_value=np.nan)
    rJ = f_best(age_Joyce[mJ]) - Fe_H[mJ]
    rB = f_best(age_Bensby[mB]) - Fe_H[mB]
    vJ = np.isfinite(rJ)
    vB = np.isfinite(rB)
    ax_res.scatter(age_Joyce[mJ][vJ], rJ[vJ], marker="*", s=40, color="tab:red", alpha=0.9, label="Joyce residuals")
    ax_res.scatter(age_Bensby[mB][vB], rB[vB], marker="^", s=40, color="tab:blue", alpha=0.9, label="Bensby residuals")
    ax_res.axhline(0.0, ls="--", lw=1.0, color="black", alpha=0.8)
    res_all = np.concatenate([rJ[vJ], rB[vB]]) if (vJ.any() or vB.any()) else np.array([0.0])
    s = np.nanstd(res_all)
    ax_res.set_ylim(-max(0.5, 3*s), +max(0.5, 3*s))

    # sideways Fe/H histogram (data)
    centers_obs, norm_counts = _smoothed_hist(Fe_H[np.isfinite(Fe_H)], feh_bins, sigma_bins=1.2)
    #ax_side.fill_betweenx(centers_obs, 0, norm_counts, facecolor="none", hatch="///", edgecolor="0.2", linewidth=0.0, alpha=1.0, label="Observed [Fe/H]")
    ax_side.plot(norm_counts, centers_obs, lw=2, color="0.2")

    # best-model MDF (same row), using helper that already exists in this module
    mx, my = _extract_mdf_xy(best_row)
    mx = np.asarray(mx, float); my = np.asarray(my, float)
    # re-bin onto feh_bins for overlay
    counts, edges = np.histogram(mx, bins=feh_bins, weights=my)
    counts = gaussian_filter1d(counts.astype(float), 1.2, mode="nearest")
    if counts.max() > 0:
        counts /= counts.max()
    ctr = 0.5 * (edges[:-1] + edges[1:])
    #ax_side.fill_betweenx(ctr, 0, counts, color="crimson", alpha=0.18, label="Best MDF")
    ax_side.plot(counts, ctr, color="crimson", lw=2, ls="--")

    # cosmetics
    ax_main.set_xlim(0.0, age_limit_gyr)
    ax_main.set_ylim(-2.0, 1.0)
    ax_main.set_ylabel("[Fe/H]")
    ax_main.tick_params(axis="x", labelbottom=False)
    ax_main.legend(loc="upper left", fontsize=10, frameon=True)

    ax_res.set_xlabel("Age (Gyr)")
    ax_res.set_ylabel("Model − Obs [Fe/H]")
    ax_res.legend(loc="upper left", fontsize=10, frameon=True)

    ax_side.set_xlabel("Normalized counts")
    ax_side.set_xlim(0, 1.15)
    ax_side.set_ylim(ax_main.get_ylim())
    ax_side.yaxis.set_label_position("right")
    ax_side.yaxis.tick_right()
    ax_side.tick_params(axis="y", labelright=True, labelleft=False, length=3)
    ax_side.legend(loc="lower right", fontsize=9, frameon=True)

    out = os.path.join(output_dir, "Age_Metallicity_all_detailed.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# --------------------------------------------------------------------------------
# Posterior version with residuals + sideways MDF posterior band
# --------------------------------------------------------------------------------
def plot_age_posterior(
    df: pd.DataFrame,
    output_dir: str,
    n_bins=12,
    feh_bins=None,
    age_limit_gyr=16.0,
    n_grid=3000,
    top_for_mdf=256,
    loss_metric="loss"
):
    os.makedirs(output_dir, exist_ok=True)

    best_idx = stable_best_index(df, primary=loss_metric)
    best_row = df.loc[best_idx]
    Fe_H, age_Joyce, age_Bensby = load_observed_amr()
    if feh_bins is None:
        feh_bins = np.linspace(-2.0, 1.0, 28)

    # age posterior band (uses your existing ensemble builder)
    age_grid_gyr = np.linspace(0.0, age_limit_gyr, n_grid)
    w = df["posterior_w"].to_numpy()
    median, lo16, hi84 = build_age_feh_ensemble(df, w, age_grid_gyr)

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[4, 1],
                           height_ratios=[3, 1],
                           wspace=0.0, hspace=0.0,
                           left=0.07, right=0.97, top=0.96, bottom=0.08)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_res  = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_side = fig.add_subplot(gs[0, 1], sharey=ax_main)

    # Posterior credible band + median
    ax_main.fill_between(age_grid_gyr, lo16, hi84, color="Blue", alpha=0.25, label="68% posterior")
    ax_main.plot(age_grid_gyr, median, color="Blue", lw=2.0, label="Posterior median")

    # Best model overlay
    bx = np.asarray(best_row["age_x"], float)
    by = np.asarray(best_row["age_y"], float)
    bx = (bx[-1] - bx) / 1e9
    ax_main.plot(bx, by, color="crimson", lw=2.5, label="Best model")

    # observations (raw) + binned
    mJ = np.isfinite(age_Joyce) & np.isfinite(Fe_H)
    mB = np.isfinite(age_Bensby) & np.isfinite(Fe_H)
    ax_main.scatter(age_Joyce[mJ], Fe_H[mJ], marker="*", s=55, color="tab:red",   alpha=0.8, label="Joyce (raw)")
    ax_main.scatter(age_Bensby[mB], Fe_H[mB], marker="^", s=55, color="tab:blue", alpha=0.8, label="Bensby (raw)")

    age_bins = np.linspace(0.0, age_limit_gyr, n_bins + 1)
    Jc, Jm, Js, _ = _binned_xy(age_Joyce, Fe_H, age_bins)
    Bc, Bm, Bs, _ = _binned_xy(age_Bensby, Fe_H, age_bins)
    ax_main.plot(Jc, Jm, color="tab:red", lw=2.0, label="Joyce (binned)")
    ax_main.errorbar(Jc, Jm, yerr=Js, color="tab:red", alpha=0.35, lw=1.0, capsize=3)
    ax_main.plot(Bc, Bm, color="tab:blue", lw=2.0, label="Bensby (binned)")
    ax_main.errorbar(Bc, Bm, yerr=Bs, color="tab:blue", alpha=0.35, lw=1.0, capsize=3)

    # Residuals vs posterior median
    xs, ys = _interp_clean(age_grid_gyr, median)
    f_med = interp1d(xs, ys, kind="linear", bounds_error=False, fill_value=np.nan)

    rJ = f_med(age_Joyce[mJ]) - Fe_H[mJ]
    rB = f_med(age_Bensby[mB]) - Fe_H[mB]
    vJ = np.isfinite(rJ)
    vB = np.isfinite(rB)
    ax_res.scatter(age_Joyce[mJ][vJ], rJ[vJ], marker="*", s=40, color="tab:red", alpha=0.9, label="Joyce residuals")
    ax_res.scatter(age_Bensby[mB][vB], rB[vB], marker="^", s=40, color="tab:blue", alpha=0.9, label="Bensby residuals")
    ax_res.axhline(0.0, ls="--", lw=1.0, color="black", alpha=0.8)
    res_all = np.concatenate([rJ[vJ], rB[vB]]) if (vJ.any() or vB.any()) else np.array([0.0])
    s = np.nanstd(res_all)
    ax_res.set_ylim(-max(0.5, 3*s), +max(0.5, 3*s))

    # Sideways MDF posterior from df rows (weighted quantiles across common grid)
    # 1) build common Fe/H grid
    feh_grid = 0.5 * (feh_bins[:-1] + feh_bins[1:])

    # 2) pick top rows (by posterior_w)
    pick = np.argsort(w)[::-1][:min(top_for_mdf, len(df))]
    curves = []
    weights = []
    for i in pick:
        row = df.iloc[i]
        mx, my = _extract_mdf_xy(row)
        mx = np.asarray(mx, float); my = np.asarray(my, float)
        if not (np.isfinite(mx).any() and np.isfinite(my).any()):
            continue
        # normalize like the side histogram
        if np.max(my) > 0:
            my = my / np.max(my)
        curves.append(np.interp(feh_grid, mx, my, left=0.0, right=0.0))
        weights.append(df.iloc[i]["posterior_w"])
    if len(curves):
        M = np.vstack(curves)
        w_use = np.asarray(weights, float)
        # compute weighted quantiles per bin
        q16 = []; q50 = []; q84 = []
        for j in range(M.shape[1]):
            col = M[:, j]
            q16.append(_weighted_quantile(col, w_use, 0.16))
            q50.append(_weighted_quantile(col, w_use, 0.50))
            q84.append(_weighted_quantile(col, w_use, 0.84))
        q16 = np.asarray(q16); q50 = np.asarray(q50); q84 = np.asarray(q84)

        ax_side.fill_betweenx(feh_grid, q16, q84, color="Blue", alpha=0.18, label="MDF 68% posterior")
        ax_side.plot(q50, feh_grid, color="Blue", lw=2, ls="--", label="MDF median")

    # Obs Fe/H histogram
    centers_obs, norm_counts = _smoothed_hist(Fe_H[np.isfinite(Fe_H)], feh_bins, sigma_bins=1.2)
    #ax_side.fill_betweenx(centers_obs, 0, norm_counts, facecolor="none", hatch="///", edgecolor="0.2", linewidth=0.0, alpha=1.0, label="Observed [Fe/H]")
    ax_side.plot(norm_counts, centers_obs, lw=2, color="0.2")

    # cosmetics
    ax_main.set_xlim(0.0, age_limit_gyr)
    ax_main.set_ylim(-2.0, 1.0)
    ax_main.set_ylabel("[Fe/H]")
    ax_main.tick_params(axis="x", labelbottom=False)
    ax_main.legend(loc="best", fontsize=10, frameon=True)

    ax_res.set_xlabel("Age (Gyr)")
    ax_res.set_ylabel("Model − Data")
    ax_res.legend(loc="upper left", fontsize=10, frameon=True)

    ax_side.set_xlabel("Normalized counts")
    ax_side.set_xlim(0, 1.15)
    ax_side.set_ylim(ax_main.get_ylim())
    ax_side.yaxis.set_label_position("right")
    ax_side.yaxis.tick_right()
    ax_side.tick_params(axis="y", labelright=True, labelleft=False, length=3)
    ax_side.legend(loc="lower right", fontsize=9, frameon=True)

    out = os.path.join(output_dir, "Age_Metallicity_posterior_detailed.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")






def plot_age_posterior(
    df: pd.DataFrame,
    output_dir: str,
    n_bins=12,          # unused now but kept for API compatibility
    feh_bins=None,
    age_limit_gyr=16.0,
    n_grid=3000,        # used as number of age bins (x direction)
    top_for_mdf=256,    # unused now but kept
    loss_metric="loss",
    smooth_sigma: float = 1.2,
    cmap: str = "Blues",
    posterior_gamma: float = 0.7,
):
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Best row + observations
    # ------------------------------------------------------------------
    best_idx = stable_best_index(df, primary=loss_metric)
    best_row = df.loc[best_idx]

    Fe_H, age_Joyce, age_Bensby = load_observed_amr()

    if feh_bins is None:
        feh_bins = np.linspace(-2.0, 1.0, 28)

    feh_min = float(feh_bins[0])
    feh_max = float(feh_bins[-1])

    # ------------------------------------------------------------------
    # Build age–[Fe/H] ensemble grid: x = Age (Gyr), y = [Fe/H]
    # ------------------------------------------------------------------
    n_grid_x = int(n_grid)
    n_grid_y = 100

    age_bins = np.linspace(0.0, age_limit_gyr, n_grid_x + 1)
    age_grid = 0.5 * (age_bins[:-1] + age_bins[1:])

    weights = df["posterior_w"].to_numpy(dtype=float)
    n_models = len(df)
    feh_stack = np.full((n_models, age_grid.size), np.nan, dtype=float)

    for i, (_, row) in enumerate(df.iterrows()):
        ax = np.asarray(row["age_x"], float)
        ay = np.asarray(row["age_y"], float)
        if ax.size < 2 or ay.size < 2:
            continue

        # convert to look-back age in Gyr (0 → now, max → oldest)
        ax_gyr = (ax[-1] - ax) / 1.0e9

        # clean + sort
        m = np.isfinite(ax_gyr) & np.isfinite(ay)
        if not np.any(m):
            continue
        ax_gyr = ax_gyr[m]
        ay = ay[m]
        order = np.argsort(ax_gyr)
        ax_gyr = ax_gyr[order]
        ay = ay[order]

        # interpolate to common age grid
        feh_stack[i] = np.interp(age_grid, ax_gyr, ay, left=np.nan, right=np.nan)


    # ------------------------------------------------------------------
    # Posterior median AMR track (per Age bin)
    # ------------------------------------------------------------------
    good_w = np.isfinite(weights) & (weights > 0)
    median_track = np.nanmedian(feh_stack[good_w], axis=0)

    # ------------------------------------------------------------------
    # Build 2D normalized density p([Fe/H] | Age)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Build 2D normalized density p([Fe/H] | Age)
    # ------------------------------------------------------------------
    feh_bins_2d = np.linspace(feh_min, feh_max, n_grid_y + 1)
    H = np.zeros((n_grid_y, n_grid_x))

    good_w = np.isfinite(weights) & (weights > 0)

    for j in range(n_grid_x):
        y_slice = feh_stack[:, j]
        mask = np.isfinite(y_slice) & good_w
        if not np.any(mask):
            continue

        w_slice = weights[mask]
        y_slice = y_slice[mask]

        hist, _ = np.histogram(
            y_slice, bins=feh_bins_2d, weights=w_slice, density=False
        )

        s = hist.sum()
        if s > 0:
            hist /= s  # normalize this Age slice
        H[:, j] = hist

    # ------------------------------------------------------------------
    # Smooth + saturation control (same style as MDF posterior)
    # ------------------------------------------------------------------
    H_smooth = gaussian_filter(H, sigma=smooth_sigma)

    hmax = np.nanmax(H_smooth)
    if hmax > 0:
        H_smooth = H_smooth / hmax

    positive = H_smooth[H_smooth > 0]
    if positive.size > 0:
        vmax = np.nanpercentile(positive, 99.0)  # tweak 95–99 if needed
    else:
        vmax = 1.0

    norm = mcolors.PowerNorm(
        gamma=posterior_gamma,
        vmin=0.0,
        vmax=vmax,
    )

    # ------------------------------------------------------------------
    # Figure: main panel + residuals (same structure as MDF posterior)
    # ------------------------------------------------------------------
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    # 2D age–[Fe/H] posterior
    ax_main.pcolormesh(
        age_bins,
        feh_bins_2d,
        H_smooth,
        cmap=cmap,
        norm=norm,
        shading="auto",
        rasterized=True,
        alpha=1.0,
        zorder=1,
    )

    # ------------------------------------------------------------------
    # Overplots: best model track + observational points
    # ------------------------------------------------------------------
    bx = np.asarray(best_row["age_x"], float)
    by = np.asarray(best_row["age_y"], float)
    bx_gyr = (bx[-1] - bx) / 1.0e9

    ax_main.plot(
        age_grid,
        median_track,
        color="crimson",
        lw=1.8,
        alpha=0.95,
        label="Best model",
        zorder=3,
    )


    mJ = np.isfinite(age_Joyce) & np.isfinite(Fe_H)
    mB = np.isfinite(age_Bensby) & np.isfinite(Fe_H)

    ax_main.scatter(
        age_Joyce[mJ],
        Fe_H[mJ],
        marker="*",
        s=45,
        color="tab:red",
        alpha=0.9,
        label="Joyce",
        zorder=4,
    )
    ax_main.scatter(
        age_Bensby[mB],
        Fe_H[mB],
        marker="^",
        s=45,
        color="tab:blue",
        alpha=0.9,
        label="Bensby",
        zorder=4,
    )

    ax_main.set_xlim(0.0, age_limit_gyr)
    ax_main.set_ylim(feh_min, feh_max)
    ax_main.set_ylabel("[Fe/H]")
    ax_main.tick_params(axis="x", labelbottom=False)
    ax_main.legend(loc="best", fontsize=9)

    # ------------------------------------------------------------------
    # Residuals: best model − data
    # ------------------------------------------------------------------
    def _interp_nan(x_new, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 2:
            return np.full_like(np.asarray(x_new, float), np.nan, float)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        x_new = np.asarray(x_new, float)
        return np.interp(x_new, x, y, left=np.nan, right=np.nan)

    y_best_J = _interp_nan(age_Joyce[mJ], bx_gyr, by)
    y_best_B = _interp_nan(age_Bensby[mB], bx_gyr, by)

    rJ = y_best_J - Fe_H[mJ]
    rB = y_best_B - Fe_H[mB]

    vJ = np.isfinite(rJ)
    vB = np.isfinite(rB)

    y_med_J = _interp_nan(age_Joyce[mJ], age_grid, median_track)
    y_med_B = _interp_nan(age_Bensby[mB], age_grid, median_track)

    rJ = y_med_J - Fe_H[mJ]
    rB = y_med_B - Fe_H[mB]




    ax_res.axhline(0.0, color="0.3", lw=1.0, ls="--", zorder=0)
    ax_res.scatter(
        age_Joyce[mJ][vJ],
        rJ[vJ],
        marker="*",
        s=35,
        color="tab:red",
        alpha=0.9,
        label="Joyce residuals",
    )
    ax_res.scatter(
        age_Bensby[mB][vB],
        rB[vB],
        marker="^",
        s=35,
        color="tab:blue",
        alpha=0.9,
        label="Bensby residuals",
    )

    res_all = np.concatenate([rJ[vJ], rB[vB]]) if (vJ.any() or vB.any()) else np.array([0.0])
    finite_res = res_all[np.isfinite(res_all)]
    if finite_res.size > 0:
        p2, p98 = np.nanpercentile(finite_res, [2, 98])
        span = (p98 - p2) if np.isfinite(p98 - p2) else 1.0
        ax_res.set_ylim(min(-0.1, p2 - 0.2 * span), max(0.1, p98 + 0.2 * span))
    else:
        ax_res.set_ylim(-0.5, 0.5)

    ax_res.set_xlim(0.0, age_limit_gyr)
    ax_res.set_xlabel("Age (Gyr)")
    ax_res.set_ylabel("Model − Data")
    ax_res.legend(loc="upper left", fontsize=9)

    out = os.path.join(output_dir, "Age_Metallicity_posterior_2D.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
