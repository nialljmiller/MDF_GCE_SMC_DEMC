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



def compute_posterior_amr_median(
    df: pd.DataFrame,
    feh_bins=None,
    age_limit_gyr: float = 16.0,
    n_grid: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the posterior median AMR track:
    returns (age_grid [Gyr], median [Fe/H](age_grid)).
    """

    if feh_bins is None:
        feh_bins = np.linspace(-2.0, 1.0, 28)

    feh_min = float(feh_bins[0])
    feh_max = float(feh_bins[-1])

    # Age grid (look-back age, Gyr)
    n_grid_x = int(n_grid)
    n_grid_y = 100  # unused here but mirrors plot_age_posterior

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

        m = np.isfinite(ax_gyr) & np.isfinite(ay)
        if not np.any(m):
            continue
        ax_gyr = ax_gyr[m]
        ay = ay[m]
        order = np.argsort(ax_gyr)
        ax_gyr = ax_gyr[order]
        ay = ay[order]

        feh_stack[i] = np.interp(age_grid, ax_gyr, ay, left=np.nan, right=np.nan)

    good_w = np.isfinite(weights) & (weights > 0)
    median_track = np.nanmedian(feh_stack[good_w], axis=0)

    return age_grid, median_track




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






