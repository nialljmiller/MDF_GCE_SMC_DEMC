import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .best_selection import *
from .ensembles import build_alpha_ensemble
from .obs import load_observed_alpha
from .ensembles import _extract_mdf_xy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
from .plot_age import compute_posterior_amr_median

from scipy.ndimage import gaussian_filter1d

def smooth_alpha_track_time_ordered(x_data, y_data, sigma=3):
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    x = np.asarray(x_data)[mask]
    y = np.asarray(y_data)[mask]
    return (
        gaussian_filter1d(x, sigma=sigma, mode='nearest'),
        gaussian_filter1d(y, sigma=sigma, mode='nearest'),
    )


def plot_alpha_age_posterior(
    df: pd.DataFrame,
    output_dir,
    age_limit_gyr: float = 16.0,
    n_grid=400,
    loss_metric="loss",
    smooth_sigma: float = 1.0,
    cmap: str = "Blues",
    posterior_gamma: float = 0.6,
):
    """
    IDENTICAL layout to plot_alpha_posterior, but x-axis is Age (Gyr).
    Fe/H     → Age transform built from posterior AMR median track.
    Data     → transformed using SAME mapping.
    Posterior → interpolated onto uniform AGE grid.
    """

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------
    # Build Age_grid and posterior median AMR track
    # -----------------------------------------------------
    age_grid, feh_med = compute_posterior_amr_median(
        df,
        feh_bins=None,
        age_limit_gyr=age_limit_gyr,
        n_grid=n_grid
    )

    # Build FeH → Age interpolator
    m = np.isfinite(feh_med) & np.isfinite(age_grid)
    feh_sorted = feh_med[m]
    age_sorted = age_grid[m]
    order = np.argsort(feh_sorted)
    feh_sorted = feh_sorted[order]
    age_sorted = age_sorted[order]

    feh_to_age = interp1d(
        feh_sorted,
        age_sorted,
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )

    # -----------------------------------------------------
    # Load α–Fe/H data and transform FeH → Age
    # -----------------------------------------------------
    Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe = load_observed_alpha()
    element_names = ["Mg", "Si", "Ca", "Ti"]
    obs_data = []
    for (obs_x, obs_y) in [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]:
        obs_age = feh_to_age(obs_x)
        obs_data.append((obs_age, obs_y))

    # -----------------------------------------------------
    # Shared AGE grid
    # -----------------------------------------------------
    age_min = 0.0
    age_max = age_limit_gyr
    age_grid_uniform = np.linspace(age_min, age_max, n_grid)
    xlim = (age_min, age_max)
    ylim = (-0.6, 0.8)

    xbins = np.linspace(xlim[0], xlim[1], 40)
    ybins = np.linspace(ylim[0], ylim[1], 40)
    age_bins_2d = np.linspace(xlim[0], xlim[1], age_grid_uniform.size + 1)

    weights = df["posterior_w"].to_numpy(float)
    good_w = np.isfinite(weights) & (weights > 0)

    # -----------------------------------------------------
    # FIGURE LAYOUT (identical to original)
    # -----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, elem in enumerate(element_names):
        ax = axes[i]

        df_el = df.copy(deep=False)
        df_el["Fe_H_x"] = df_el["alpha_tracks"].apply(
            lambda tr: np.asarray(tr[i][0], float)
        )
        df_el[f"{elem}_Fe_y"] = df_el["alpha_tracks"].apply(
            lambda tr: np.asarray(tr[i][1], float)
        )

        n_models = len(df_el)
        n_x = age_grid_uniform.size

        # -----------------------------------------
        # Convert each model track: Fe/H→Age and interpolate
        # -----------------------------------------
        y_stack = np.full((n_models, n_x), np.nan, float)
        for k, (_, row_el) in enumerate(df_el.iterrows()):

            feh_x = row_el["Fe_H_x"]
            alpha_y = row_el[f"{elem}_Fe_y"]

            age_x = feh_to_age(feh_x)
            mask = np.isfinite(age_x) & np.isfinite(alpha_y)
            if not np.any(mask):
                continue

            age_x = age_x[mask]
            alpha_y = alpha_y[mask]
            order = np.argsort(age_x)
            age_x = age_x[order]
            alpha_y = alpha_y[order]

            if age_x.size < 2:
                continue

            y_interp = np.interp(
                age_grid_uniform, age_x, alpha_y,
                left=np.nan, right=np.nan
            )
            y_stack[k] = y_interp

        # Posterior median α(Age)
        median_track = np.nanmedian(y_stack[good_w], axis=0)

        # ----------------------------------------------------------
        # Build 2D posterior p(alpha | Age)
        # ----------------------------------------------------------
        H = np.zeros((len(ybins) - 1, n_x))
        for j in range(n_x):
            y_slice = y_stack[:, j]
            mask = good_w & np.isfinite(y_slice)
            if not np.any(mask):
                continue

            hist, _ = np.histogram(
                y_slice[mask],
                bins=ybins,
                weights=weights[mask],
                density=False,
            )
            s = hist.sum()
            if s > 0:
                hist = hist / s
            H[:, j] = hist

        H_smooth = gaussian_filter(H, sigma=smooth_sigma)
        hmax = np.nanmax(H_smooth)
        if hmax > 0:
            H_smooth /= hmax

        positive = H_smooth[H_smooth > 0]
        vmax = np.nanpercentile(positive, 99.0) if positive.size > 0 else 1.0

        norm = mcolors.PowerNorm(
            gamma=posterior_gamma,
            vmin=0.0,
            vmax=vmax,
        )

        # ----------------------------------------------------------
        # Main panel — IDENTICAL STYLE
        # ----------------------------------------------------------
        ax.pcolormesh(
            age_bins_2d,
            ybins,
            H_smooth,
            cmap=cmap,
            norm=norm,
            shading="auto",
            rasterized=True,
            zorder=1,
        )

        ax.plot(
            age_grid_uniform,
            median_track,
            color="crimson",
            lw=2.2,
            alpha=0.95,
            zorder=3,
            label="Best model",
        )

        # Transformed observational data
        obs_age, obs_alpha = obs_data[i]
        ax.scatter(
            obs_age,
            obs_alpha,
            color="k",
            s=12,
            alpha=0.8,
            edgecolor="none",
            zorder=4,
            label="Data" if i == 0 else None,
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Age (Gyr)")
        ax.set_ylabel(f"[{elem}/Fe]")

        # ----------------------------------------------------------
        # Marginal panels — EXACT COPIES OF YOUR VERSION
        # ----------------------------------------------------------
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="16%", pad=0.04, sharex=ax)
        ax_right = divider.append_axes("right", size="16%", pad=0.04, sharey=ax)

        m_obs = np.isfinite(obs_age) & np.isfinite(obs_alpha)
        ax_top.hist(obs_age[m_obs], bins=xbins, density=True,
                    histtype="step", lw=1.4, color="black")
        ax_right.hist(obs_alpha[m_obs], bins=ybins, density=True,
                      histtype="step", lw=1.4, color="black",
                      orientation="horizontal")

        m_med = np.isfinite(age_grid_uniform) & np.isfinite(median_track)
        ax_top.hist(age_grid_uniform[m_med], bins=xbins, density=True,
                    histtype="step", lw=1.4, color="red")
        ax_right.hist(median_track[m_med], bins=ybins, density=True,
                      histtype="step", lw=1.4, color="red",
                      orientation="horizontal")

        for axm in (ax_top, ax_right):
            axm.tick_params(left=False, bottom=False,
                            labelleft=False, labelbottom=False)
            for s in axm.spines.values():
                s.set_visible(False)

        if i == 0:
            ax.legend(loc="lower left", fontsize=9, frameon=True)

    out = os.path.join(output_dir, "Four_Panel_Alpha_Age_Posterior.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")




def plot_alpha_posterior(
    df: pd.DataFrame,
    output_dir,
    n_grid=400,
    loss_metric="loss",
    smooth_sigma: float = 1.0,
    cmap: str = "Blues",
    posterior_gamma: float = 0.6,
):
    os.makedirs(output_dir, exist_ok=True)

    best_idx = stable_best_index(df, primary=loss_metric)
    best_row = df.loc[best_idx]
    Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe = load_observed_alpha()

    element_names = ["Mg", "Si", "Ca", "Ti"]
    obs_data = [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]

    # Shared Fe/H grid
    feh_grid = np.linspace(-2.5, 0.6, n_grid)
    xlim = (feh_grid.min(), feh_grid.max())
    ylim = (-0.6, 0.8)

    # Bins for marginals and 2D density
    xbins = np.linspace(xlim[0], xlim[1], 40)
    ybins = np.linspace(ylim[0], ylim[1], 40)  # edges for pcolormesh too
    feh_grid_bins = np.linspace(xlim[0], xlim[1], feh_grid.size + 1)

    weights = df["posterior_w"].to_numpy(dtype=float)
    good_w = np.isfinite(weights) & (weights > 0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, elem in enumerate(element_names):
        ax = axes[i]

        df_el = df.copy(deep=False)
        df_el["Fe_H_x"] = df_el["alpha_tracks"].apply(
            lambda tr: np.asarray(tr[i][0], float)
        )
        df_el[f"{elem}_Fe_y"] = df_el["alpha_tracks"].apply(
            lambda tr: np.asarray(tr[i][1], float)
        )

        n_models = len(df_el)
        n_x = feh_grid.size

        # Interpolate all models to grid
        y_stack = np.full((n_models, n_x), np.nan, dtype=float)
        for k, (_, row_el) in enumerate(df_el.iterrows()):
            x = row_el["Fe_H_x"]
            y = row_el[f"{elem}_Fe_y"]
            y_interp = np.interp(feh_grid, x, y, left=np.nan, right=np.nan)
            y_stack[k] = y_interp

        # ----- posterior median track -----
        median_track = np.nanmedian(y_stack[good_w], axis=0)

        # Build posterior density
        H = np.zeros((len(ybins) - 1, n_x))
        for j in range(n_x):
            y_slice = y_stack[:, j]
            mask = np.isfinite(y_slice) & good_w
            if not np.any(mask):
                continue

            hist, _ = np.histogram(
                y_slice[mask],
                bins=ybins,
                weights=weights[mask],
                density=False,
            )
            s = hist.sum()
            if s > 0:
                hist = hist / s
            H[:, j] = hist

        H_smooth = gaussian_filter(H, sigma=smooth_sigma)
        hmax = np.nanmax(H_smooth)
        if hmax > 0:
            H_smooth = H_smooth / hmax

        positive = H_smooth[H_smooth > 0]
        vmax = np.nanpercentile(positive, 99.0) if positive.size > 0 else 1.0

        norm = mcolors.PowerNorm(
            gamma=posterior_gamma,
            vmin=0.0,
            vmax=vmax,
        )

        ax.pcolormesh(
            feh_grid_bins,
            ybins,
            H_smooth,
            cmap=cmap,
            norm=norm,
            shading="auto",
            rasterized=True,
            zorder=1,
        )

        # plot posterior median track
        ax.plot(
            feh_grid,
            median_track,
            color="crimson",
            lw=2.2,
            alpha=0.95,
            zorder=3,
            label="Best model",
        )

        # Observed points
        obs_x, obs_y = obs_data[i]
        ax.scatter(
            obs_x,
            obs_y,
            color="k",
            s=12,
            alpha=0.8,
            edgecolor="none",
            zorder=4,
            label="Data" if i == 0 else None,
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)



        # ----- marginal histograms -----
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="16%", pad=0.04, sharex=ax)
        ax_right = divider.append_axes("right", size="16%", pad=0.04, sharey=ax)

        m_obs = np.isfinite(obs_x) & np.isfinite(obs_y)
        ax_top.hist(obs_x[m_obs], bins=xbins, density=True,
                    histtype="step", lw=1.4, color="black")
        ax_right.hist(obs_y[m_obs], bins=ybins, density=True,
                      histtype="step", lw=1.4, color="black",
                      orientation="horizontal")

        # posterior-median histograms (red)
        m_med = np.isfinite(feh_grid) & np.isfinite(median_track)
        ax_top.hist(feh_grid[m_med], bins=xbins, density=True,
                    histtype="step", lw=1.4, color="red")
        ax_right.hist(median_track[m_med], bins=ybins, density=True,
                      histtype="step", lw=1.4, color="red",
                      orientation="horizontal")

        for axm in (ax_top, ax_right):
            axm.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for s in axm.spines.values():
                s.set_visible(False)

        if i == 0:
            ax.legend(loc="lower left", fontsize=9, frameon=True)


    out = os.path.join(output_dir, "Four_Panel_Alpha_Posterior.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")





















def plot_alpha_age_posterior(
    df: pd.DataFrame,
    output_dir,
    age_limit_gyr: float = 16.0,
    n_grid=400,
    loss_metric="loss",
    smooth_sigma: float = 1.0,
    cmap: str = "Blues",
    posterior_gamma: float = 0.6,
):
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------
    # Best model
    # ------------------------
    best_idx = stable_best_index(df, primary=loss_metric)
    best_row = df.loc[best_idx]

    # ------------------------
    # Build AMR → Fe/H→Age map
    # ------------------------
    age_grid, feh_med = compute_posterior_amr_median(
        df,
        feh_bins=None,
        age_limit_gyr=age_limit_gyr,
        n_grid=n_grid
    )

    mask = np.isfinite(feh_med) & np.isfinite(age_grid)
    feh_sorted = feh_med[mask]
    age_sorted = age_grid[mask]
    order = np.argsort(feh_sorted)
    feh_sorted = feh_sorted[order]
    age_sorted = age_sorted[order]

    feh_to_age = interp1d(
        feh_sorted,
        age_sorted,
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )

    # ------------------------
    # Load & transform observed α–Fe/H data
    # ------------------------
    Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe = load_observed_alpha()
    element_names = ["Mg", "Si", "Ca", "Ti"]
    obs_data = []
    for (obs_x, obs_y) in [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]:
        obs_age = feh_to_age(obs_x)
        obs_data.append((obs_age, obs_y))

    # ------------------------
    # Shared AGE grid
    # ------------------------
    age_min = 0.0
    age_max = age_limit_gyr
    age_grid_uniform = np.linspace(age_min, age_max, n_grid)

    xlim = (age_min, age_max)
    ylim = (-0.6, 0.8)

    xbins = np.linspace(xlim[0], xlim[1], 40)
    ybins = np.linspace(ylim[0], ylim[1], 40)
    age_bins_2d = np.linspace(xlim[0], xlim[1], age_grid_uniform.size + 1)

    weights = df["posterior_w"].to_numpy(float)
    good_w = np.isfinite(weights) & (weights > 0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, elem in enumerate(element_names):
        ax = axes[i]

        # extract α–Fe/H tracks
        df_el = df.copy(deep=False)
        df_el["Fe_H_x"] = df_el["alpha_tracks"].apply(
            lambda tr: np.asarray(tr[i][0], float)
        )
        df_el[f"{elem}_Fe_y"] = df_el["alpha_tracks"].apply(
            lambda tr: np.asarray(tr[i][1], float)
        )

        n_models = len(df_el)
        n_x = age_grid_uniform.size

        # ------------------------
        # Build 2D posterior grid
        # ------------------------
        y_stack = np.full((n_models, n_x), np.nan, float)

        for k, (_, row_el) in enumerate(df_el.iterrows()):
            feh_x = row_el["Fe_H_x"]
            alpha_y = row_el[f"{elem}_Fe_y"]

            age_x = feh_to_age(feh_x)
            mask = np.isfinite(age_x) & np.isfinite(alpha_y)
            if not np.any(mask):
                continue
            age_x = age_x[mask]
            alpha_y = alpha_y[mask]
            order = np.argsort(age_x)
            age_x = age_x[order]
            alpha_y = alpha_y[order]

            if age_x.size < 2:
                continue

            y_interp = np.interp(
                age_grid_uniform, age_x, alpha_y, left=np.nan, right=np.nan
            )
            y_stack[k] = y_interp

        # 2D posterior KDE map
        H = np.zeros((len(ybins) - 1, n_x))
        for j in range(n_x):
            col = y_stack[:, j]
            m = good_w & np.isfinite(col)
            if not np.any(m):
                continue
            hist, _ = np.histogram(
                col[m],
                bins=ybins,
                weights=weights[m],
                density=False,
            )
            s = hist.sum()
            if s > 0:
                hist /= s
            H[:, j] = hist

        H_smooth = gaussian_filter(H, sigma=smooth_sigma)
        hmax = np.nanmax(H_smooth)
        if hmax > 0:
            H_smooth /= hmax
        positive = H_smooth[H_smooth > 0]
        vmax = np.nanpercentile(positive, 99.0) if positive.size > 0 else 1.0

        norm = mcolors.PowerNorm(
            gamma=posterior_gamma,
            vmin=0.0,
            vmax=vmax
        )

        ax.pcolormesh(
            age_bins_2d, ybins, H_smooth,
            cmap=cmap, norm=norm,
            shading="auto",
            rasterized=True,
            zorder=1
        )

        # ------------------------
        # BEST MODEL red line (not posterior median)
        # ------------------------
        btracks = best_row["alpha_tracks"]
        best_feh = np.asarray(btracks[i][0], float)
        best_alpha = np.asarray(btracks[i][1], float)

        best_age = feh_to_age(best_feh)

        # --- NEW: smooth raw parametric Age–Alpha curve ---
        age_s, alpha_s = smooth_alpha_track_time_ordered(best_age, best_alpha, sigma=3)

        ax.plot(
            age_s,
            alpha_s,
            color="crimson",
            lw=2.2,
            alpha=0.95,
            zorder=3,
            label="Best model",
        )

        # observed transformed data
        obs_age, obs_alpha = obs_data[i]
        ax.scatter(
            obs_age, obs_alpha,
            color="k",
            s=12,
            alpha=0.8,
            edgecolor="none",
            zorder=4,
            label="Data" if i == 0 else None,
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Age (Gyr)")
        ax.set_ylabel(f"[{elem}/Fe]")

        # ------------------------
        # Marginals (best model)
        # ------------------------
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="16%", pad=0.04, sharex=ax)
        ax_right = divider.append_axes("right", size="16%", pad=0.04, sharey=ax)

        # Data
        m_obs = np.isfinite(obs_age) & np.isfinite(obs_alpha)
        ax_top.hist(obs_age[m_obs], bins=xbins, density=True, histtype="step",
                    lw=1.4, color="black")
        ax_right.hist(obs_alpha[m_obs], bins=ybins, density=True, histtype="step",
                      lw=1.4, color="black", orientation="horizontal")

        # Best model
        m_best = np.isfinite(best_age) & np.isfinite(best_alpha)
        if np.any(m_best):
            ax_top.hist(best_age[m_best], bins=xbins, density=True,
                        histtype="step", lw=1.4, color="red")
            ax_right.hist(best_alpha[m_best], bins=ybins, density=True,
                          histtype="step", lw=1.4, color="red",
                          orientation="horizontal")

        for axm in (ax_top, ax_right):
            axm.tick_params(left=False, bottom=False,
                            labelleft=False, labelbottom=False)
            for s in axm.spines.values():
                s.set_visible(False)

        if i == 0:
            ax.legend(loc="lower left", fontsize=9, frameon=True)

    out = os.path.join(output_dir, "Four_Panel_Alpha_Age_Posterior.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")








def plot_alpha_posterior(
    df: pd.DataFrame,
    output_dir,
    n_grid=400,
    loss_metric="loss",
    smooth_sigma: float = 1.0,
    cmap: str = "Blues",
    posterior_gamma: float = 0.6,
):
    os.makedirs(output_dir, exist_ok=True)

    best_idx = stable_best_index(df, primary=loss_metric)
    best_row = df.loc[best_idx]

    Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe = load_observed_alpha()
    element_names = ["Mg", "Si", "Ca", "Ti"]
    obs_data = [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]

    feh_grid = np.linspace(-2.5, 0.6, n_grid)
    xlim = (feh_grid.min(), feh_grid.max())
    ylim = (-0.6, 0.8)

    xbins = np.linspace(xlim[0], xlim[1], 40)
    ybins = np.linspace(ylim[0], ylim[1], 40)
    feh_grid_bins = np.linspace(xlim[0], xlim[1], feh_grid.size + 1)

    weights = df["posterior_w"].to_numpy(float)
    good_w = np.isfinite(weights) & (weights > 0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, elem in enumerate(element_names):
        ax = axes[i]

        df_el = df.copy(deep=False)
        df_el["Fe_H_x"] = df_el["alpha_tracks"].apply(
            lambda tr: np.asarray(tr[i][0], float)
        )
        df_el[f"{elem}_Fe_y"] = df_el["alpha_tracks"].apply(
            lambda tr: np.asarray(tr[i][1], float)
        )

        n_models = len(df_el)
        n_x = feh_grid.size

        # Build 2D posterior
        y_stack = np.full((n_models, n_x), np.nan, float)
        for k, (_, row_el) in enumerate(df_el.iterrows()):
            x = row_el["Fe_H_x"]
            y = row_el[f"{elem}_Fe_y"]
            y_interp = np.interp(feh_grid, x, y, left=np.nan, right=np.nan)
            y_stack[k] = y_interp

        H = np.zeros((len(ybins) - 1, n_x))
        for j in range(n_x):
            col = y_stack[:, j]
            m = good_w & np.isfinite(col)
            if not np.any(m):
                continue
            hist, _ = np.histogram(
                col[m],
                bins=ybins,
                weights=weights[m],
                density=False
            )
            s = hist.sum()
            if s > 0:
                hist /= s
            H[:, j] = hist

        H_smooth = gaussian_filter(H, sigma=smooth_sigma)
        hmax = np.nanmax(H_smooth)
        if hmax > 0:
            H_smooth /= hmax
        positive = H_smooth[H_smooth > 0]
        vmax = np.nanpercentile(positive, 99.0) if positive.size > 0 else 1.0

        norm = mcolors.PowerNorm(
            gamma=posterior_gamma,
            vmin=0.0,
            vmax=vmax,
        )

        ax.pcolormesh(
            feh_grid_bins, ybins, H_smooth,
            cmap=cmap, norm=norm,
            shading="auto",
            rasterized=True,
            zorder=1,
        )

        # ------------------------
        # BEST MODEL red line
        # ------------------------
        btracks = best_row["alpha_tracks"]
        bx = np.asarray(btracks[i][0], float)
        by = np.asarray(btracks[i][1], float)
        # sort by FeH
        m = np.isfinite(bx) & np.isfinite(by)
        bx2 = bx[m]
        by2 = by[m]

        order = np.argsort(bx2)
        bx2 = bx2[order]
        by2 = by2[order]

        bx = np.asarray(btracks[i][0], float)
        by = np.asarray(btracks[i][1], float)

        # --- NEW: smooth the parametric best-model curve ---
        bx_s, by_s = smooth_alpha_track_time_ordered(bx, by, sigma=3)

        ax.plot(
            bx_s,
            by_s,
            color="crimson",
            lw=2.2,
            alpha=0.95,
            zorder=3,
            label="Best model",
        )


        # observed data
        obs_x, obs_y = obs_data[i]
        ax.scatter(
            obs_x, obs_y,
            color="k",
            s=12,
            alpha=0.8,
            edgecolor="none",
            zorder=4,
            label="Data" if i == 0 else None,
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        # ------------------------
        # Marginals (best model)
        # ------------------------
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="16%", pad=0.04, sharex=ax)
        ax_right = divider.append_axes("right", size="16%", pad=0.04, sharey=ax)

        # Data
        m_obs = np.isfinite(obs_x) & np.isfinite(obs_y)
        ax_top.hist(obs_x[m_obs], bins=xbins, density=True,
                    histtype="step", lw=1.4, color="black")
        ax_right.hist(obs_y[m_obs], bins=ybins, density=True,
                      histtype="step", lw=1.4, color="black",
                      orientation="horizontal")

        # Best model
        m_best = np.isfinite(bx) & np.isfinite(by)
        ax_top.hist(bx[m_best], bins=xbins, density=True,
                    histtype="step", lw=1.4, color="red")
        ax_right.hist(by[m_best], bins=ybins, density=True,
                      histtype="step", lw=1.4, color="red",
                      orientation="horizontal")

        for axm in (ax_top, ax_right):
            axm.tick_params(left=False, bottom=False,
                            labelleft=False, labelbottom=False)
            for s in axm.spines.values():
                s.set_visible(False)

        if i == 0:
            ax.legend(loc="lower left", fontsize=9, frameon=True)

    out = os.path.join(output_dir, "Four_Panel_Alpha_Posterior.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")
