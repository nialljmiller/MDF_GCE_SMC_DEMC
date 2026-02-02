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


def plot_alpha(df: pd.DataFrame, output_dir, n_grid=200, top_overlay=99999999999999, loss_metric = "loss"):
    os.makedirs(output_dir, exist_ok=True)
    best_idx = stable_best_index(df, primary=loss_metric)
    best_row = df.loc[best_idx]
    Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe = load_observed_alpha()

    element_names = ["Mg", "Si", "Ca", "Ti"]
    obs_data = [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]

    # Global limits + binning so histograms match across panels
    xlim = (-2.0, 0.7)
    ylim = (-0.6, 0.7)
    xbins = np.linspace(xlim[0], xlim[1], 40)
    ybins = np.linspace(ylim[0], ylim[1], 40)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    # Draw panels
    for i, elem in enumerate(element_names):
        ax = axes[i]

        # many grey curves, highest posterior first (same as MDF style)
        order = np.argsort(df["posterior_w"].to_numpy())[::-1][:min(top_overlay, len(df))]
        for j in order:
            row = df.iloc[j]
            tracks = row["alpha_tracks"]  # [[FeH, MgFe],[FeH, SiFe],...]
            x = np.asarray(tracks[i][0], float)
            y = np.asarray(tracks[i][1], float)
            ax.plot(x, y, alpha=0.05, lw=0.7, color="gray", zorder=1)

        # best in red
        btracks = best_row["alpha_tracks"]
        bx = np.asarray(btracks[i][0], float)
        by = np.asarray(btracks[i][1], float)
        ax.plot(bx, by, color="red", lw=2.0, label="best model", zorder=3)

        # observed points
        obs_x, obs_y = obs_data[i]
        ax.scatter(obs_x, obs_y, color="k", s=12, alpha=0.8, edgecolor="none", zorder=2)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if i == 0 or i == 2:
            ax.set_ylabel(r"[$\alpha$/Fe]")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', labelleft=False)
        if i == 2 or i == 3:
            ax.set_xlabel("[Fe/H]")
        else:
            ax.tick_params(axis='x', labelbottom=False)

        ax.text(0.05, 0.95, elem, transform=ax.transAxes, ha='left', va='top', fontsize=25, weight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))


        # ---------- Marginal histograms (top and right) ----------
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="16%", pad=0.04, sharex=ax)
        ax_right = divider.append_axes("right", size="16%", pad=0.04, sharey=ax)

        # Data histograms
        m_obs = np.isfinite(obs_x) & np.isfinite(obs_y)
        ax_top.hist(obs_x[m_obs], bins=xbins, density=True, histtype="step", lw=1.4, color="black")
        ax_right.hist(obs_y[m_obs], bins=ybins, density=True, histtype="step",
                      lw=1.4, color="black", orientation="horizontal")

        # Model histograms (best track)
        m_best = np.isfinite(bx) & np.isfinite(by)
        if np.any(m_best):
            ax_top.hist(bx[m_best], bins=xbins, density=True, histtype="step", lw=1.4, color="red")
            ax_right.hist(by[m_best], bins=ybins, density=True, histtype="step",
                          lw=1.4, color="red", orientation="horizontal")

        # Clean look for the marginals
        for axm in (ax_top, ax_right):
            axm.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for s in axm.spines.values():
                s.set_visible(False)

        # Minimal legend (only once)
        #if i == 0:
        #    ax.legend(loc="upper left", fontsize=9, frameon=True)

    out = os.path.join(output_dir, "Four_Panel_Alpha.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")





def plot_alpha_posterior(df: pd.DataFrame, output_dir, n_grid=2000, loss_metric = "loss"):
    os.makedirs(output_dir, exist_ok=True)
    best_idx = stable_best_index(df, primary=loss_metric)
    best_row = df.loc[best_idx]
    Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe = load_observed_alpha()

    element_names = ["Mg", "Si", "Ca", "Ti"]
    obs_data = [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]

    # Use a shared grid for posterior bands
    feh_grid = np.linspace(-2.5, 0.6, n_grid)

    # Global limits + binning for consistent histograms
    xlim = (feh_grid.min(), feh_grid.max())
    ylim = (-0.6, 0.8)
    xbins = np.linspace(xlim[0], xlim[1], 40)
    ybins = np.linspace(ylim[0], ylim[1], 40)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, elem in enumerate(element_names):
        ax = axes[i]

        # Prepare per-element arrays
        df_el = df.copy(deep=False)
        df_el["Fe_H_x"] = df_el["alpha_tracks"].apply(lambda tr: np.asarray(tr[i][0], float))
        df_el[f"{elem}_Fe_y"] = df_el["alpha_tracks"].apply(lambda tr: np.asarray(tr[i][1], float))

        # Posterior band
        median, lo16, hi84 = build_alpha_ensemble(
            df_el,
            df_el["posterior_w"].to_numpy(),
            feh_grid,
            i
        )
        ax.fill_between(feh_grid, lo16, hi84, alpha=0.30, label="68% posterior", zorder=2)
        ax.plot(feh_grid, median, lw=1.5, label="posterior median", zorder=3)

        # Best in red
        btracks = best_row["alpha_tracks"]
        bx = np.asarray(btracks[i][0], float)
        by = np.asarray(btracks[i][1], float)
        #ax.plot(bx, by, color="red", lw=2.0, label="best model", zorder=4)

        # Observed points
        obs_x, obs_y = obs_data[i]
        ax.scatter(obs_x, obs_y, color="k", s=12, alpha=0.8, edgecolor="none", zorder=5)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if i == 0 or i == 2:
            ax.set_ylabel(r"[$\alpha$/Fe]")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', labelleft=False)
        if i == 2 or i == 3:
            ax.set_xlabel("[Fe/H]")
        else:
            ax.tick_params(axis='x', labelbottom=False)

        ax.text(0.05, 0.95, elem, transform=ax.transAxes, ha='left', va='top', fontsize=25, weight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))



        # ---------- Marginal histograms (top and right) ----------
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="16%", pad=0.04, sharex=ax)
        ax_right = divider.append_axes("right", size="16%", pad=0.04, sharey=ax)

        # Data histograms
        m_obs = np.isfinite(obs_x) & np.isfinite(obs_y)
        ax_top.hist(obs_x[m_obs], bins=xbins, density=True, histtype="step", lw=1.4, color="black")
        ax_right.hist(obs_y[m_obs], bins=ybins, density=True, histtype="step",
                      lw=1.4, color="black", orientation="horizontal")

        # Model histograms from POSTERIOR MEDIAN curve (not bounds)
        m_med = np.isfinite(feh_grid) & np.isfinite(median)
        if np.any(m_med):
            ax_top.hist(feh_grid[m_med], bins=xbins, density=True, histtype="step", lw=1.4, color="red")
            ax_right.hist(median[m_med], bins=ybins, density=True, histtype="step",
                          lw=1.4, color="red", orientation="horizontal")

        # Clean look for the marginals
        for axm in (ax_top, ax_right):
            axm.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for s in axm.spines.values():
                s.set_visible(False)

        #if i == 0:
        #    ax.legend(loc="upper left", fontsize=9, frameon=True)

    out = os.path.join(output_dir, "Four_Panel_Alpha_Posterior.png")
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

        # Per-element tracks for all models
        df_el = df.copy(deep=False)
        df_el["Fe_H_x"] = df_el["alpha_tracks"].apply(
            lambda tr: np.asarray(tr[i][0], float)
        )
        df_el[f"{elem}_Fe_y"] = df_el["alpha_tracks"].apply(
            lambda tr: np.asarray(tr[i][1], float)
        )

        n_models = len(df_el)
        n_x = feh_grid.size

        # Interpolate all models onto shared Fe/H grid
        y_stack = np.full((n_models, n_x), np.nan, dtype=float)
        for k, (_, row_el) in enumerate(df_el.iterrows()):
            x = row_el["Fe_H_x"]
            y = row_el[f"{elem}_Fe_y"]
            y_interp = np.interp(feh_grid, x, y, left=np.nan, right=np.nan)
            y_stack[k] = y_interp

        # Build 2D density p(y | x) slice-by-slice
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

        # Smooth and normalize
        H_smooth = gaussian_filter(H, sigma=smooth_sigma)
        hmax = np.nanmax(H_smooth)
        if hmax > 0:
            H_smooth = H_smooth / hmax

        positive = H_smooth[H_smooth > 0]
        if positive.size > 0:
            vmax = np.nanpercentile(positive, 99.0)
        else:
            vmax = 1.0

        norm = mcolors.PowerNorm(
            gamma=posterior_gamma,
            vmin=0.0,
            vmax=vmax,
        )

        # Background posterior density
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

        # Best model track (for this element)
        btracks = best_row["alpha_tracks"]
        bx = np.asarray(btracks[i][0], float)
        by = np.asarray(btracks[i][1], float)
        ax.plot(bx, by, color="crimson", lw=2.0, alpha=0.9, zorder=3, label="Best model")

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

        if i in (0, 2):
            ax.set_ylabel(r"[$\alpha$/Fe]")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

        if i in (2, 3):
            ax.set_xlabel("[Fe/H]")
        else:
            ax.tick_params(axis="x", labelbottom=False)

        ax.text(
            0.05,
            0.95,
            elem,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=25,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

        # ---------- Marginal histograms (data vs best model) ----------
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="16%", pad=0.04, sharex=ax)
        ax_right = divider.append_axes("right", size="16%", pad=0.04, sharey=ax)

        m_obs = np.isfinite(obs_x) & np.isfinite(obs_y)
        ax_top.hist(
            obs_x[m_obs],
            bins=xbins,
            density=True,
            histtype="step",
            lw=1.4,
            color="black",
        )
        ax_right.hist(
            obs_y[m_obs],
            bins=ybins,
            density=True,
            histtype="step",
            lw=1.4,
            color="black",
            orientation="horizontal",
        )

        m_best = np.isfinite(bx) & np.isfinite(by)
        ax_top.hist(
            bx[m_best],
            bins=xbins,
            density=True,
            histtype="step",
            lw=1.4,
            color="red",
        )
        ax_right.hist(
            by[m_best],
            bins=ybins,
            density=True,
            histtype="step",
            lw=1.4,
            color="red",
            orientation="horizontal",
        )

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
