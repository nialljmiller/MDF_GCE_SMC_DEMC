import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde  # only used for your smoothing helper if needed
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plotting.style import *
use_paper_style()





def smooth_alpha_track_time_ordered(x_data, y_data, sigma=3):
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    x = np.asarray(x_data)[mask]
    y = np.asarray(y_data)[mask]
    return gaussian_filter1d(x, sigma=sigma, mode='nearest'), gaussian_filter1d(y, sigma=sigma, mode='nearest')




def plot_age_feh_detailed(GalGA, Fe_H, age_Joyce, age_Bensby, results_df=None, save_path=None, n_bins=12, feh_bins=None, age_limit_gyr=14.2):

    if save_path is None: save_path = GalGA.output_path + 'Age_Metallicity_all.png'
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    Fe_H = np.asarray(Fe_H, float)
    age_Joyce = np.asarray(age_Joyce, float)
    age_Bensby = np.asarray(age_Bensby, float)

    r0 = results_df.iloc[0]
    best_params = (float(r0['sigma_2']), float(r0['t_2']), float(r0['infall_2']))

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[3, 1], wspace=0.0, hspace=0.0, left=0.07, right=0.97, top=0.96, bottom=0.08)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_res  = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_side = fig.add_subplot(gs[0, 1], sharey=ax_main)

    best_age_gyr, best_feh = None, None
    alpha_all = 0.15
    for (t, feh), res in zip(GalGA.age_data, GalGA.results):
        params = (float(res[5]), float(res[7]), float(res[9]))
        t = np.asarray(t, float); feh = np.asarray(feh, float)
        age_gyr = (t[-1] - t) / 1e9
        if params == best_params:
            best_age_gyr, best_feh = age_gyr.copy(), feh.copy()
            ax_main.plot(age_gyr, feh, 'r-', lw=2.5, zorder=5, label='Best model')
        else:
            ax_main.plot(age_gyr, feh, '-', color='gray', lw=0.7, alpha=alpha_all, zorder=1)

    mJ = np.isfinite(age_Joyce) & np.isfinite(Fe_H)
    mB = np.isfinite(age_Bensby) & np.isfinite(Fe_H)
    ax_main.scatter(age_Joyce[mJ], Fe_H[mJ], marker='*', s=55, color='blue', alpha=0.7, zorder=6, label='Joyce (raw)')
    ax_main.scatter(age_Bensby[mB], Fe_H[mB], marker='^', s=55, color='orange', alpha=0.7, zorder=6, label='Bensby (raw)')

    def binned(x, y, bins):
        m = np.isfinite(x) & np.isfinite(y)
        means, _, _ = binned_statistic(x[m], y[m], statistic='mean', bins=bins)
        stds,  _, _ = binned_statistic(x[m], y[m], statistic='std',  bins=bins)
        centers = 0.5 * (bins[:-1] + bins[1:])
        return centers, means, stds

    age_bins = np.linspace(0, age_limit_gyr, n_bins + 1)
    cJ, mJm, mJs = binned(age_Joyce, Fe_H, age_bins)
    cB, mBm, mBs = binned(age_Bensby, Fe_H, age_bins)
    ax_main.plot(cJ, mJm, color='blue', lw=2.0, zorder=7, label='Joyce (binned)')
    ax_main.errorbar(cJ, mJm, yerr=mJs, color='blue', alpha=0.3, lw=1.0, capsize=3, zorder=6)
    ax_main.plot(cB, mBm, color='orange', lw=2.0, zorder=7, label='Bensby (binned)')
    ax_main.errorbar(cB, mBm, yerr=mBs, color='orange', alpha=0.3, lw=1.0, capsize=3, zorder=6)

    residuals = []
    idx = np.argsort(best_age_gyr)
    xs, ys = best_age_gyr[idx], best_feh[idx]
    f_best = interp1d(xs, ys, kind='linear', bounds_error=False, fill_value=np.nan)

    rJ = f_best(age_Joyce[mJ]) - Fe_H[mJ]
    rB = f_best(age_Bensby[mB]) - Fe_H[mB]
    vJ = np.isfinite(rJ); vB = np.isfinite(rB)
    ax_res.scatter(age_Joyce[mJ][vJ], rJ[vJ], marker='*', s=40, color='blue', alpha=0.8, label='Joyce residuals')
    ax_res.scatter(age_Bensby[mB][vB], rB[vB], marker='^', s=40, color='orange', alpha=0.8, label='Bensby residuals')
    residuals = np.concatenate([rJ[vJ], rB[vB]]) if (vJ.any() or vB.any()) else np.array([])

    ax_res.axhline(0.0, ls='--', lw=1.0, color='black', alpha=0.7)
    if residuals.size:
        s = np.nanstd(residuals)
        ax_res.set_ylim(-3*s, 3*s)

    if feh_bins is None: feh_bins = np.linspace(-2.0, 1.0, 28)

    def smooth_hist(vals, bins, sigma=1.2):
        v = np.asarray(vals, float); v = v[np.isfinite(v)]
        c, e = np.histogram(v, bins=bins)
        c = gaussian_filter1d(c.astype(float), sigma, mode='nearest')
        c = c / c.max() if c.max() > 0 else c
        ctr = 0.5 * (e[:-1] + e[1:])
        return ctr, c

    obs_ctr, obs_norm = smooth_hist(Fe_H[np.isfinite(Fe_H)], feh_bins, 1.2)
    ax_side.fill_betweenx(obs_ctr, 0, obs_norm, facecolor='none', hatch='///', edgecolor='blue', linewidth=3, alpha=1.0, label='Observed Fe/H')
    ax_side.fill_betweenx(obs_ctr, 0, obs_norm, facecolor='none', hatch='\\\\\\', edgecolor='orange', linewidth=4, alpha=1.0)
    ax_side.plot(obs_norm, obs_ctr, lw=2, color ='blue')
    ax_side.plot(obs_norm, obs_ctr, lw=1, color ='orange')

    for (mx, my), res in zip(GalGA.mdf_data, GalGA.results):
        params = (float(res[5]), float(res[7]), float(res[9]))
        if params == best_params:
            mx = np.asarray(mx, float); my = np.asarray(my, float)
            counts, edges = np.histogram(mx, bins=feh_bins, weights=my)
            counts = gaussian_filter1d(counts.astype(float), 1.2, mode='nearest')
            counts = counts / counts.max() if counts.max() > 0 else counts
            ctr = 0.5 * (edges[:-1] + edges[1:])
            ax_side.fill_betweenx(ctr, 0, counts, color='red', alpha=0.20, label='Best model MDF')
            ax_side.plot(counts, ctr, color='red', lw=2, ls='--')
            break

    ax_main.set_xlim(0, age_limit_gyr); ax_main.set_ylim(-2.0, 1.0); ax_main.set_ylabel('[Fe/H]', fontsize=14)
    ax_main.tick_params(axis='x', labelbottom=False)
    ax_main.legend(loc='upper left', fontsize=10, frameon=True)

    ax_res.set_xlabel('Age (Gyr)', fontsize=14); ax_res.set_ylabel('Model − Obs [Fe/H]', fontsize=12)
    ax_res.set_xlim(0, age_limit_gyr); ax_res.legend(loc='upper left', fontsize=10, frameon=True)

    ax_side.set_xlabel('Normalized counts', fontsize=12); ax_side.set_xlim(0, 1.15); ax_side.set_ylim(ax_main.get_ylim())
    ax_side.yaxis.set_label_position('right'); ax_side.yaxis.tick_right()
    ax_side.tick_params(axis='y', labelright=True, labelleft=False, length=3)
    ax_side.legend(loc='lower right', fontsize=9, frameon=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {save_path}")
    return fig



def plot_mdf_curves(GalGA, feh, normalized_count, results_df=None, save_path=None):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    if save_path is None:
        save_path = os.path.join(GalGA.output_path, "MDF_multiple_results.png")

    # ---- choose best by parameter match, not index ----
    r0 = results_df.iloc[0]
    best_params = (float(r0["sigma_2"]), float(r0["t_2"]), float(r0["infall_2"]))

    tol = 1e-10
    best_idx = None
    for i, res in enumerate(GalGA.results):
        if (abs(float(res[5]) - best_params[0]) < tol and
            abs(float(res[7]) - best_params[1]) < tol and
            abs(float(res[9]) - best_params[2]) < tol):
            best_idx = i
            break
    if best_idx is None:
        best_idx = 0  # minimal fallback

    # ---- figure: main + residuals ----
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    # gray background curves
    n_all = len(GalGA.mdf_data)
    alpha = max(0.02, min(0.6, 8.0 / max(1, n_all)))
    for i, (x, y) in enumerate(GalGA.mdf_data):
        if i == best_idx:
            continue
        ax_main.plot(x, y, color="0.75", alpha=0.15 * alpha, lw=0.8, zorder=1)

    # best curve evaluated on the OBSERVATIONAL feh grid
    bx, by = GalGA.mdf_data[best_idx]
    f_best = interp1d(np.asarray(bx, float), np.asarray(by, float),
                      kind="linear", bounds_error=False, fill_value=0.0)
    y_best_on_feh = f_best(np.asarray(feh, float))
    m = np.isfinite(y_best_on_feh)
    if m.any():
        y_best_on_feh = y_best_on_feh / y_best_on_feh[m].max()

    # draw best + data
    ax_main.plot(feh, y_best_on_feh, color="crimson", lw=1.8, label="Model", zorder=3)
    ax_main.plot(feh, normalized_count, "x", color="k", ms=4.5, mew=0.9, label="Data", zorder=4)

    ax_main.set_xlim(-2, 1)
    ax_main.set_ylabel("Normalized number")
    ax_main.legend(loc="upper left", fontsize=9, handlelength=1.6)
    ax_main.xaxis.set_ticks_position("top")
    ax_main.xaxis.set_label_position("top")
    ax_main.set_xlabel("[Fe/H]")
    ax_main.tick_params(axis="x", bottom=False)

    # residuals on the same feh grid
    resids = y_best_on_feh - np.asarray(normalized_count, float)
    ax_res.axhline(0.0, ls="--", lw=1.0, color="black", alpha=0.7)
    ax_res.plot(feh, resids, ".", ms=3.5)
    s = np.nanstd(resids)
    if np.isfinite(s) and s > 0:
        ax_res.set_ylim(-3*s, 3*s)
    ax_res.set_xlabel("[Fe/H]")
    ax_res.set_ylabel("Model − data")

    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)





def plot_four_panel_alpha(GalGA, Fe_H, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe, results_df=None, save_path=None):
    import os, numpy as np, matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if save_path is None: save_path = GalGA.output_path + 'Four_Panel_Alpha.png'

    element_names = ['Mg', 'Si', 'Ca', 'Ti']
    observational_data = [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]

    bm = results_df.iloc[0]
    best_params = (float(bm['sigma_2']), float(bm['t_2']), float(bm['infall_2']))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=False, sharey=False)
    fig.subplots_adjust(hspace=0.1, wspace=0.1, left=0.07, right=0.94, top=0.97, bottom=0.08)

    xlim = (-2.0, 1.0)
    ylim = (-0.8, 0.8)
    xbins = np.linspace(xlim[0], xlim[1], 36)
    ybins = np.linspace(ylim[0], ylim[1], 36)

    def get_best_track(idx):
        for alpha_arrs, res in zip(GalGA.alpha_data, GalGA.results):
            params = (float(res[5]), float(res[7]), float(res[9]))
            if params == best_params:
                x = np.asarray(alpha_arrs[idx][0]); y = np.asarray(alpha_arrs[idx][1])
                x, y = smooth_alpha_track_time_ordered(x, y, sigma=3)
                return x, y
        return np.array([]), np.array([])

    Fe_H = np.asarray(Fe_H, float)

    for idx, (element, obs_data) in enumerate(zip(element_names, observational_data)):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        for alpha_arrs, res in zip(GalGA.alpha_data, GalGA.results):
            x = np.asarray(alpha_arrs[idx][0]); y = np.asarray(alpha_arrs[idx][1])
            x, y = smooth_alpha_track_time_ordered(x, y, sigma=3)
            params = (float(res[5]), float(res[7]), float(res[9]))
            if params == best_params: ax.plot(x, y, color="red", lw=2.5, zorder=3)
            else: ax.plot(x, y, color='gray', alpha=0.03, lw=1.0, zorder=1)

        obs_y = np.asarray(obs_data, float)
        mask = np.isfinite(Fe_H) & np.isfinite(obs_y)
        ax.scatter(Fe_H[mask], obs_y[mask], c='k', s=16, zorder=2, edgecolor='none')

        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        if col == 0: ax.set_ylabel(r"[$\alpha$/Fe]")
        else: ax.set_ylabel(""); ax.tick_params(axis='y', labelleft=False)
        if row == 1: ax.set_xlabel("[Fe/H]")
        else: ax.tick_params(axis='x', labelbottom=False)

        ax.text(0.05, 0.95, element, transform=ax.transAxes, ha='left', va='top', fontsize=25, weight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="16%", pad=0.04, sharex=ax)
        ax_right = divider.append_axes("right", size="16%", pad=0.04, sharey=ax)

        ax_top.hist(Fe_H[mask], bins=xbins, density=True, histtype='step', lw=1.5, color='black')
        x_best, y_best = get_best_track(idx)
        ax_top.hist(x_best[np.isfinite(x_best)], bins=xbins, density=True, histtype='step', lw=1.5, color='red')

        ax_right.hist(obs_y[mask], bins=ybins, density=True, histtype='step', lw=1.5, color='black', orientation='horizontal')
        ax_right.hist(y_best[np.isfinite(y_best)], bins=ybins, density=True, histtype='step', lw=1.5, color='red', orientation='horizontal')

        for axm in (ax_top, ax_right):
            axm.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for s in axm.spines.values(): s.set_visible(False)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Four-panel alpha plot with marginal histograms saved to {save_path}")
    return fig
