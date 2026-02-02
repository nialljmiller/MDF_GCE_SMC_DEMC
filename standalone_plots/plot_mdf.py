# standalone_plots/plot_mdf.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from .best_selection import *
from .ensembles import build_mdf_ensemble
from .obs import load_observed_mdf
from .ensembles import _extract_mdf_xy
from scipy.ndimage import gaussian_filter 
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline


# --- drop-in replacement for plot_mdf ---
def plot_mdf(
    df: pd.DataFrame,
    output_dir,
    feh_min: float = -1.5,
    feh_max: float = 0.8,
    n_grid: int = 2000,
    obs_mdf_path: str | None = None,
    top_overlay=99999999999999,
    loss_metric = "loss"):
    """
    Spaghetti + best + observed points (legacy style) with residual panel.
    Keeps current data flow (stable_best_index, _extract_mdf_xy, load_observed_mdf).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Best row by loss (with tiebreaks preserved)
    best_idx = stable_best_index(df, primary=loss_metric)
    best_row = df.loc[best_idx]

    # Observed MDF (already sorted + normalized to max=1 by loader)
    x_obs, y_obs = load_observed_mdf(obs_mdf_path)

    # Helper: linear interp that returns NaN outside support
    def _interp_nan(x_new, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < 2:
            return np.full_like(np.asarray(x_new, float), np.nan, float)
        order = np.argsort(x)
        x, y = x[order], y[order]
        # np.interp can return NaN via left/right if given np.nan
        x_new = np.asarray(x_new, float)
        y_new = np.interp(x_new, x, y, left=np.nan, right=np.nan)
        return y_new

    # Figure: main + residuals
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    # --- Background spaghetti (posterior-weight sorted) ---
    order = np.argsort(df["posterior_w"].to_numpy())[::-1][:min(top_overlay, len(df))]
    alpha = max(0.02, min(0.6, 8.0 / max(1, top_overlay)))
    for i in order:
        row = df.iloc[i]
        x_i, y_i = _extract_mdf_xy(row)
        ax_main.plot(x_i, y_i, alpha=0.05 * (alpha / 0.08), lw=0.7, color="0.6", zorder=1)

    # --- Best model curve ---
    bx, by = _extract_mdf_xy(best_row)
    ax_main.plot(bx, by, color="crimson", lw=1.8, label="Model", zorder=3)

    # --- Observed points ---
    ax_main.plot(x_obs, y_obs, "x", color="k", ms=4.5, mew=0.9, label="Data", zorder=4)

    # Styling (main)
    ax_main.set_xlim(feh_min, feh_max)
    ax_main.set_ylabel("Normalized number")
    ax_main.legend(loc="upper left", fontsize=9, handlelength=1.6)
    ax_main.xaxis.set_ticks_position("top")
    ax_main.xaxis.set_label_position("top")
    ax_main.set_xlabel("[Fe/H]")
    ax_main.tick_params(axis="x", bottom=False)

    # --- Residuals: best(model on observed abscissa) − observed ---
    y_best_on_obs = _interp_nan(x_obs, bx, by)
    res = y_best_on_obs - y_obs

    ax_res.axhline(0.0, ls="--", lw=1.0, color="0.2", alpha=0.8, zorder=1)
    ax_res.plot(x_obs, res, ".", ms=3.5, color="0.1", zorder=2)

    # Robust y-limits
    finite = np.isfinite(res)
    if finite.any():
        s = np.nanstd(res[finite])
        if s > 0:
            lo, hi = np.nanpercentile(res[finite], [2, 98])
            pad = max(2.5 * s, 0.1 * (hi - lo))
            ax_res.set_ylim(min(lo, -2.5 * s) - pad * 0.0, max(hi, 2.5 * s) + pad * 0.0)

    ax_res.set_xlabel("[Fe/H]")
    ax_res.set_ylabel("Model − Data")


    ax_res.set_xlim(feh_min,feh_max)
    ax_main.set_xlim(feh_min,feh_max)

    out = os.path.join(output_dir, "MDF_multiple_results.png")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")
    return fig


# --- drop-in replacement for plot_mdf_posterior ---
def plot_mdf_posterior(
    df: pd.DataFrame,
    output_dir,
    feh_min: float = -1.5,
    feh_max: float = 0.8,
    n_grid: int = 2000,
    obs_mdf_path: str | None = None,
    loss_metric = "loss",
):
    """
    Posterior band + best + observed points with residual band below.
    - Residuals panel shows (posterior band − observed MDF interpolated to grid).
    - Also overlays best-model residual line for reference.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Best row
    best_idx = stable_best_index(df, primary=loss_metric)
    best_row = df.loc[best_idx]

    # Observed MDF (sorted/normalized)
    x_obs, y_obs = load_observed_mdf(obs_mdf_path)

    # Common grid and ensemble
    feh_grid = np.linspace(feh_min, feh_max, n_grid)
    median, lo16, hi84 = build_mdf_ensemble(df, df["posterior_w"].to_numpy(), feh_grid)

    # Helper: interp to grid returning NaN outside support
    def _interp_nan(x_new, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < 2:
            return np.full_like(np.asarray(x_new, float), np.nan, float)
        order = np.argsort(x)
        x, y = x[order], y[order]
        x_new = np.asarray(x_new, float)
        y_new = np.interp(x_new, x, y, left=np.nan, right=np.nan)
        return y_new

    # Observed MDF mapped to the grid (for residual band)
    y_obs_on_grid = _interp_nan(feh_grid, x_obs, y_obs)

    # Figure: main + residuals
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    # --- Posterior band + median ---
    ax_main.fill_between(feh_grid, lo16, hi84, alpha=0.30, label="68% posterior", zorder=1)
    ax_main.plot(feh_grid, median, color="Blue", lw=1.6, label="Posterior median", zorder=2)

    # --- Best model (for reference) ---
    bx, by = _extract_mdf_xy(best_row)
    ax_main.plot(bx, by, color="crimson", lw=1.2, alpha=0.8, label="Best model", zorder=3)

    # --- Observed points ---
    ax_main.plot(x_obs, y_obs, "x", color="k", ms=4.5, mew=0.9, label="Data", zorder=4)

    # Styling (main)
    ax_main.set_xlim(feh_min, feh_max)
    ax_main.set_ylabel("Normalized number")
    ax_main.legend(loc="upper left", fontsize=9, handlelength=1.6)
    ax_main.xaxis.set_ticks_position("top")
    ax_main.xaxis.set_label_position("top")
    ax_main.set_xlabel("[Fe/H]")
    ax_main.tick_params(axis="x", bottom=False)

    # --- Residual posterior band: (posterior − observed_on_grid) ---
    r_med = median - y_obs_on_grid
    r_lo  = lo16  - y_obs_on_grid
    r_hi  = hi84  - y_obs_on_grid

    ax_res.fill_between(feh_grid, r_lo, r_hi, alpha=0.25, color="0.3", zorder=1, label=None)
    ax_res.plot(feh_grid, r_med, lw=1.2, color="0.05", zorder=2)

    # Also overlay best-model residuals evaluated at observed points
    y_best_on_obs = _interp_nan(x_obs, bx, by)
    res_best = y_best_on_obs - y_obs
    ax_res.plot(x_obs, res_best, ".", ms=3.2, color="tab:blue", alpha=0.9, zorder=3)

    ax_res.axhline(0.0, color="0.3", lw=1.0, ls="--", zorder=0)

    # Robust y-limits
    resid_all = np.concatenate([
        r_med[np.isfinite(r_med)],
        res_best[np.isfinite(res_best)]
    ]) if np.isfinite(r_med).any() or np.isfinite(res_best).any() else np.array([])
    if resid_all.size:
        p2, p98 = np.nanpercentile(resid_all, [2, 98])
        span = (p98 - p2) if np.isfinite(p98 - p2) else 1.0
        ax_res.set_ylim(p2 - 0.10 * span, p98 + 0.10 * span)

    ax_res.set_ylabel("Model − Data")
    ax_res.set_xlabel("[Fe/H]")


    ax_res.set_xlim(feh_min,feh_max)
    ax_main.set_xlim(feh_min,feh_max)

    out = os.path.join(output_dir, "MDF_posterior.png")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")
    return fig








# ---- extend best model curve using posterior median, not raw obs ----
def _extend_with_obs(
    bx, by,
    x_obs, y_obs,          # kept for API compatibility, not used anymore
    feh_min, feh_max,
    feh_grid,              # grid used for the posterior
    post_median,           # median(y | x) on feh_grid
    pad_bins: int = 3,     # how many bins to use for the blend on each side
    smooth_sigma_1d: float = 2.0
):
    """
    Build an extended best-model MDF curve:

    - Inside the well-sampled region we follow the original best model.
    - Outside it, we follow the (smoothed) posterior median.
    - Over 'pad_bins' points near each edge we *blend* best ↔ median
      so the curve doesn't kink.

    All output is on feh_grid, which spans [feh_min, feh_max].
    """

    bx = np.asarray(bx, float)
    by = np.asarray(by, float)
    feh_grid = np.asarray(feh_grid, float)
    post_median = np.asarray(post_median, float)

    # smooth the posterior median so the tails are gentle
    post_med_smooth = gaussian_filter1d(post_median, sigma=smooth_sigma_1d, mode="nearest")

    # put the best model onto the same grid
    best_on_grid = np.interp(feh_grid, bx, by, left=np.nan, right=np.nan)

    # where do we actually have best-model values?
    finite = np.isfinite(best_on_grid)
    idx = np.where(finite)[0]

    # if for some reason best model isn't defined, just return the median
    if idx.size == 0:
        return feh_grid, post_med_smooth

    idx_min = idx[0]
    idx_max = idx[-1]

    # this is the "3 points before" behaviour:
    # we start blending pad_bins bins in from each edge,
    # so the extreme tails are *pure* posterior median.
    left_blend_start  = idx_min
    left_blend_end    = min(idx_min + pad_bins, idx_max)

    right_blend_end   = idx_max
    right_blend_start = max(idx_max - pad_bins, left_blend_end)

    y_out = post_med_smooth.copy()  # default: follow median everywhere

    # ----- left blend: median → best -----
    if left_blend_end >= left_blend_start:
        for j in range(left_blend_start, left_blend_end + 1):
            # t=0 => pure median, t→1 => pure best
            t = (j - left_blend_start + 1) / (left_blend_end - left_blend_start + 2)
            y_out[j] = (1.0 - t) * post_med_smooth[j] + t * best_on_grid[j]

    # ----- central region: pure best model -----
    central_start = left_blend_end + 1
    central_end   = right_blend_start - 1
    if central_end >= central_start:
        y_out[central_start:central_end + 1] = best_on_grid[central_start:central_end + 1]

    # ----- right blend: best → median -----
    if right_blend_end >= right_blend_start:
        for j in range(right_blend_start, right_blend_end + 1):
            # t=0 => pure best, t→1 => pure median
            t = (j - right_blend_start + 1) / (right_blend_end - right_blend_start + 2)
            y_out[j] = (1.0 - t) * best_on_grid[j] + t * post_med_smooth[j]

    # clip tiny negatives if any
    y_out[y_out < 0.0] = 0.0

    return feh_grid, y_out












# --- v2: Posterior 2D density plot ---
def plot_mdf_posterior(
    df: pd.DataFrame,
    output_dir,
    feh_min: float = -1.5,
    feh_max: float = 0.8,
    n_grid_x: int = 200,           # Bins for [Fe/H] axis
    n_grid_y: int = 100,           # Bins for Normalized Number axis
    obs_mdf_path: str | None = None,
    loss_metric="loss",
    smooth_sigma: float = 1.2,     # Smoothing strength
    cmap: str = "Blues",            # Colormap
    posterior_gamma = 0.7
):
    """
    Posterior 2D density + best + observed points with residual band below.
    - Main panel shows full 2D weighted posterior density, normalized per [Fe/H] slice.
    - Residuals panel shows (best-model residual line).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Best row
    best_idx = stable_best_index(df, primary=loss_metric)
    best_row = df.loc[best_idx]

    # Observed MDF (sorted/normalized)
    x_obs, y_obs = load_observed_mdf(obs_mdf_path)

    # --- 1. Build Ensemble Data ---
    # This logic is extracted from build_mdf_ensemble
    feh_grid_bins = np.linspace(feh_min, feh_max, n_grid_x + 1)
    feh_grid = 0.5 * (feh_grid_bins[:-1] + feh_grid_bins[1:]) # bin centers
    
    weights = df["posterior_w"].to_numpy(dtype=float)
    n_models = len(df)
    y_stack = np.full((n_models, feh_grid.size), np.nan, dtype=float)

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            x, y = _extract_mdf_xy(row)
            y_interp = np.interp(feh_grid, x, y, left=np.nan, right=np.nan)
            y_stack[i] = y_interp
        except Exception:
            continue  # Leave as np.nan

    # --- 2. Create 2D Normalized Density Histogram ---
    # We need a 2D histogram p(y | x)
    # We loop over each x-bin ([Fe/H] slice) and create a
    # 1D weighted, normalized histogram of the y-values in that slice.

    # Bins for the y-axis (Normalized Number)
    # Find robust max, ignoring outliers
    y_max_robust = np.nanpercentile(y_stack, 99) * 1.05
    y_bins = np.linspace(0.0, max(y_max_robust, 1.05), n_grid_y + 1)

    # H will store the 2D density, shape (n_grid_y, n_grid_x)
    H = np.zeros((n_grid_y, n_grid_x))
    
    good_w = np.isfinite(weights) & (weights > 0)
    
    for j in range(n_grid_x):
        y_slice = y_stack[:, j]
        mask = np.isfinite(y_slice) & good_w
        
        if not np.any(mask):
            continue
            
        w_slice = weights[mask]
        y_slice = y_slice[mask]

        hist, _ = np.histogram(
            y_slice, bins=y_bins, weights=w_slice, density=False
        )
        
        s = hist.sum()
        if s > 0:
            hist /= s  # Normalize this [Fe/H] slice to sum to 1
            
        H[:, j] = hist

    # --- 3. Smooth and Plot ---
    # Smooth the 2D density map
    H_smooth = gaussian_filter(H, sigma=smooth_sigma)



    # ---- saturation control ----
    # Normalize to [0, 1]
    hmax = np.nanmax(H_smooth)
    if hmax > 0:
        H_smooth = H_smooth / hmax

    # Clip the very highest values so the ridge doesn't dominate
    positive = H_smooth[H_smooth > 0]
    if positive.size > 0:
        vmax = np.nanpercentile(positive, 99.0)  # tweak 95–99 to taste
    else:
        vmax = 1.0

    norm = mcolors.PowerNorm(
        gamma=posterior_gamma,  # e.g. 0.5
        vmin=0.0,
        vmax=vmax
    )


    # Setup Figure
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    ax_main.pcolormesh(
        feh_grid_bins, y_bins, H_smooth,
        cmap=cmap,
        norm=norm,              # <--- this was missing
        shading="auto",
        rasterized=True,
        alpha=1.0,
        zorder=1,
    )


    # --- 2b. Posterior *median* MDF as a function of [Fe/H] ---
    def _weighted_median(values, w):
        values = np.asarray(values, float)
        w = np.asarray(w, float)
        order = np.argsort(values)
        v = values[order]
        ww = w[order]
        cdf = np.cumsum(ww)
        cdf /= cdf[-1]
        return np.interp(0.5, cdf, v)

    post_median = np.full(feh_grid.size, np.nan, dtype=float)
    for j in range(n_grid_x):
        y_slice = y_stack[:, j]
        mask = np.isfinite(y_slice) & good_w
        if np.any(mask):
            post_median[j] = _weighted_median(y_slice[mask], weights[mask])

    # fill any gaps linearly, then smooth
    mask_med = np.isfinite(post_median)
    post_median = np.interp(
        feh_grid,
        feh_grid[mask_med],
        post_median[mask_med]
    )


    bx, by = _extract_mdf_xy(best_row)
    bx, by = _extend_with_obs(
        bx, by,
        x_obs, y_obs,
        feh_min, feh_max,
        feh_grid=feh_grid,
        post_median=post_median,
        pad_bins=5,
        smooth_sigma_1d=2.0,
    )






    ax_main.plot(bx, by, color="crimson", lw=1.2, alpha=0.8, label="Best model", zorder=3)

    # Observed points
    ax_main.plot(x_obs, y_obs, "x", color="k", ms=4.5, mew=0.9, label="Data", zorder=4)

    # Styling (main)
    ax_main.set_xlim(feh_min, feh_max)
    ax_main.set_ylim(y_bins[0], y_bins[-1])
    ax_main.set_ylabel("Normalized number")
    ax_main.legend(loc="upper left", fontsize=9, handlelength=1.6)
    ax_main.xaxis.set_ticks_position("top")
    ax_main.xaxis.set_label_position("top")
    ax_main.set_xlabel("[Fe/H]")
    ax_main.tick_params(axis="x", bottom=False)

    # --- 5. Residual Plot ---
    # Helper: interp to grid returning NaN outside support
    def _interp_nan(x_new, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < 2:
            return np.full_like(np.asarray(x_new, float), np.nan, float)
        order = np.argsort(x); x, y = x[order], y[order]
        x_new = np.asarray(x_new, float)
        y_new = np.interp(x_new, x, y, left=np.nan, right=np.nan)
        return y_new

    # Best-model residuals evaluated at observed points
    y_best_on_obs = _interp_nan(x_obs, bx, by)
    res_best = y_best_on_obs - y_obs
    
    ax_res.axhline(0.0, color="0.3", lw=1.0, ls="--", zorder=0)
    ax_res.plot(x_obs, res_best, ".", ms=3.2, color="crimson", alpha=0.9, zorder=3)

    # Robust y-limits for residuals
    finite_res = res_best[np.isfinite(res_best)]
    if finite_res.size > 0:
        p2, p98 = np.nanpercentile(finite_res, [2, 98])
        span = (p98 - p2) if np.isfinite(p98 - p2) else 1.0
        ax_res.set_ylim(min(-0.05, p2 - 0.1 * span), max(0.05, p98 + 0.1 * span))
    else:
        ax_res.set_ylim(-0.2, 0.2) # Default if no data

    # --- RMS text ---
    if finite_res.size > 0:
        rms = np.sqrt(np.mean(finite_res**2))
        ax_res.text(
            0.02, 0.95,
            rf"RMS = {rms:.3f}",
            transform=ax_res.transAxes,
            ha="left", va="top"
        )
    # --- end RMS text ---

    ax_res.set_ylabel("Model − Data")
    ax_res.set_xlabel("[Fe/H]")
    ax_res.set_xlim(feh_min, feh_max)
    ax_main.set_xlim(feh_min, feh_max)


    out = os.path.join(output_dir, "MDF_posterior_2D.png")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")
    return fig