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





# Smoother (kept from your original)
def smooth_alpha_track_time_ordered(x_data, y_data, sigma=3):
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    x = np.asarray(x_data)[mask]
    y = np.asarray(y_data)[mask]
    if len(x) < 10:
        return x_data, y_data
    return gaussian_filter1d(x, sigma=sigma, mode='nearest'), gaussian_filter1d(y, sigma=sigma, mode='nearest')








def plot_age_feh_detailed(
    GalGA,
    Fe_H,
    age_Joyce,
    age_Bensby,
    results_df=None,
    save_path=None,
    n_bins=12,
    feh_bins=None,
    age_limit_gyr=14.2
):
    """
    One-shot Age–[Fe/H] figure with:
      1) all model attempts (grey),
      2) best attempt (red),
      3) raw Joyce/Bensby data,
      4) binned Joyce/Bensby curves with error bars,
      5) residuals panel (model - observations),
      6) sideways histogram (Fe/H distributions) on the right axis.

    Assumptions:
      - GalGA.age_data is an iterable of (time_array, feh_array) per model.
        time_array is in seconds; we convert to "stellar age" in Gyr as (t_final - t)/1e9.
      - GalGA.results is aligned with age_data; each result has indices:
          [ ..., sigma_2 -> 5, ..., t_2 -> 7, ..., infall_2 -> 9, ... ]
      - Optionally GalGA.mdf_data is iterable of (feh_values, mdf_weights) per model.
      - Fe_H is paired with both age_Joyce and age_Bensby (same-length vectors per dataset masks).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from scipy.interpolate import interp1d, UnivariateSpline
    from scipy.stats import binned_statistic


    if save_path is None:
        save_path = GalGA.output_path + 'Age_Metallicity_multiple_results_h.png'
    

    # ---- basic checks ----
    if not hasattr(GalGA, 'age_data') or len(GalGA.age_data) == 0:
        print("No age_data available on GalGA; nothing to plot.")
        return None
    if not hasattr(GalGA, 'results') or len(GalGA.results) == 0:
        print("No results on GalGA; nothing to plot.")
        return None

    # ---- I/O paths ----
    if save_path is None:
        save_path = os.path.join(getattr(GalGA, 'output_path', ''), 'Age_Metallicity_all.png')
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # ---- sanitize arrays ----
    Fe_H = np.asarray(Fe_H, dtype=float)
    age_Joyce = np.asarray(age_Joyce, dtype=float)
    age_Bensby = np.asarray(age_Bensby, dtype=float)

    # ---- find "best" model params ----
    def _best_params_from_results_df(df):
        # expects columns 'sigma_2', 't_2', 'infall_2'
        # if your df names differ, change here.
        row0 = df.iloc[0]
        return (float(row0['sigma_2']), float(row0['t_2']), float(row0['infall_2']))

    if (results_df is not None) and (not results_df.empty):
        best_params = _best_params_from_results_df(results_df)
    else:
        r0 = GalGA.results[0]
        best_params = (float(r0[5]), float(r0[7]), float(r0[9]))

    # ---- figure layout: main + residuals + side histogram ----
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[4, 1],
        height_ratios=[3, 1],
        wspace=0.0,
        hspace=0.0,
        left=0.07, right=0.97, top=0.96, bottom=0.08
    )

    # AFTER
    ax_main = fig.add_subplot(gs[0, 0])
    ax_res  = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_side = fig.add_subplot(gs[0, 1], sharey=ax_main)

    # ---- plot all models; stash the best for later interpolation ----
    best_age_gyr, best_feh = None, None
    n_models = len(GalGA.age_data)
    # nicer alpha across many curves
    alpha_all = max(0.02, min(0.6, 8.0 / max(1, n_models)))

    for age_data, res in zip(GalGA.age_data, GalGA.results):
        params = (float(res[5]), float(res[7]), float(res[9]))
        x_time, y_feh = age_data
        x_time = np.asarray(x_time, dtype=float)
        y_feh = np.asarray(y_feh, dtype=float)

        # Convert to "age since formation" in Gyr
        age_gyr = (x_time[-1] / 1e9) - (x_time / 1e9)

        if all(abs(p - b) < 1e-12 for p, b in zip(params, best_params)):
            best_age_gyr = np.array(age_gyr, copy=True)
            best_feh = np.array(y_feh, copy=True)
            ax_main.plot(age_gyr, y_feh, color='red', lw=2.5, zorder=5, label='Best model')
        else:
            ax_main.plot(age_gyr, y_feh, color='gray', lw=0.7, alpha=alpha_all, zorder=1)

    # ---- overlay raw observational points ----
    # Using the same Fe_H vector for both datasets, masked by their valid ages.
    mask_J = np.isfinite(age_Joyce) & np.isfinite(Fe_H)
    mask_B = np.isfinite(age_Bensby) & np.isfinite(Fe_H)

    ax_main.scatter(age_Joyce[mask_J], Fe_H[mask_J], marker='*', s=55,
                    color='blue', alpha=0.7, zorder=6, label='Joyce et al. (raw)')
    ax_main.scatter(age_Bensby[mask_B], Fe_H[mask_B], marker='^', s=55,
                    color='orange', alpha=0.7, zorder=6, label='Bensby et al. (raw)')

    # ---- binned curves with errors for Joyce and Bensby ----
    def _binned(age, feh, bins):
        m = np.isfinite(age) & np.isfinite(feh)
        if np.count_nonzero(m) < 3:
            return None
        means, _, _ = binned_statistic(age[m], feh[m], statistic='mean', bins=bins)
        stds,  _, _ = binned_statistic(age[m], feh[m], statistic='std',  bins=bins)
        cnts,  _, _ = binned_statistic(age[m], feh[m], statistic='count', bins=bins)
        ctrs = 0.5 * (bins[:-1] + bins[1:])
        sem = stds / np.sqrt(np.maximum(cnts, 1))
        valid = (cnts > 0) & np.isfinite(means)
        return ctrs[valid], means[valid], stds[valid], sem[valid]

    age_bins = np.linspace(0, age_limit_gyr, n_bins + 1)
    J = _binned(age_Joyce, Fe_H, age_bins)
    B = _binned(age_Bensby, Fe_H, age_bins)

    if J is not None:
        xc, ym, ys, ysem = J
        ax_main.plot(xc, ym, color='blue', lw=2.5, zorder=7, label='Joyce (binned)')
        ax_main.errorbar(xc, ym, yerr=ys, color='blue', alpha=0.3, lw=1.0, capsize=3, zorder=6)
    if B is not None:
        xc, ym, ys, ysem = B
        ax_main.plot(xc, ym, color='orange', lw=2.5, zorder=7, label='Bensby (binned)')
        ax_main.errorbar(xc, ym, yerr=ys, color='orange', alpha=0.3, lw=1.0, capsize=3, zorder=6)

    # ---- residuals for the best model (model - obs) using interpolation ----
    def _interp_clean(x, y):
        """Sort x, drop duplicates, return monotonic arrays for interpolation."""
        idx = np.argsort(x)
        xs, ys = x[idx], y[idx]
        # unique in x
        keep = np.ones_like(xs, dtype=bool)
        keep[1:] = (np.diff(xs) > 1e-12)
        return xs[keep], ys[keep]

    residuals_all = []

    if best_age_gyr is not None and best_feh is not None and len(best_age_gyr) > 1:
        try:
            xs, ys = _interp_clean(best_age_gyr, best_feh)
            f_best = interp1d(xs, ys, kind='linear', bounds_error=False, fill_value=np.nan)

            # Joyce residuals
            if np.count_nonzero(mask_J) > 0:
                ageJ = age_Joyce[mask_J]
                fehJ = Fe_H[mask_J]
                # within model domain
                rng = (ageJ >= np.nanmin(xs)) & (ageJ <= np.nanmax(xs))
                if np.count_nonzero(rng) > 0:
                    mj = f_best(ageJ[rng])
                    rj = mj - fehJ[rng]
                    v = np.isfinite(rj)
                    if np.count_nonzero(v) > 0:
                        ax_res.scatter(ageJ[rng][v], rj[v], marker='*', s=40,
                                       color='blue', alpha=0.8, label='Joyce residuals')
                        residuals_all.append(rj[v])

            # Bensby residuals
            if np.count_nonzero(mask_B) > 0:
                ageB = age_Bensby[mask_B]
                fehB = Fe_H[mask_B]
                rng = (ageB >= np.nanmin(xs)) & (ageB <= np.nanmax(xs))
                if np.count_nonzero(rng) > 0:
                    mb = f_best(ageB[rng])
                    rb = mb - fehB[rng]
                    v = np.isfinite(rb)
                    if np.count_nonzero(v) > 0:
                        ax_res.scatter(ageB[rng][v], rb[v], marker='^', s=40,
                                       color='orange', alpha=0.8, label='Bensby residuals')
                        residuals_all.append(rb[v])
        except Exception as e:
            print(f"Residuals skipped: {e}")

    # zero line + autoscale for residuals
    ax_res.axhline(0.0, ls='--', lw=1.0, color='black', alpha=0.7)
    if len(residuals_all) > 0:
        res = np.concatenate(residuals_all)
        if np.size(res) > 0:
            s = np.nanstd(res)
            ylim = max(0.5, 3.0 * s)
            ax_res.set_ylim(-ylim, +ylim)

    # ---- sideways histogram (Fe/H distributions) ----
    # Build Fe/H bins if not provided
    if feh_bins is None:
        # cover main plot y-limits: [-2, 1] by default
        feh_bins = np.linspace(-2.0, 1.0, 28)  # ~0.11 dex bins

    def _norm_counts(vals, bins):
        if vals is None or np.count_nonzero(np.isfinite(vals)) == 0:
            return None
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        c, edges = np.histogram(v, bins=bins)
        c = c.astype(float)
        if c.max() > 0:
            c /= c.max()
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, c

    def _smoothed_hist(vals, bins, sigma_bins=1.2):
        """Histogram → Gaussian-smoothed counts → normalized to max=1."""
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        counts, edges = np.histogram(v, bins=bins)
        counts = counts.astype(float)
        if counts.max() <= 0:
            return None
        counts_s = gaussian_filter1d(counts, sigma=sigma_bins, mode='nearest')
        if counts_s.max() > 0:
            counts_s /= counts_s.max()
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, counts_s

    # Observed Fe/H distribution (Joyce and Bensby are identical → plot once)
    obs_mask = (np.isfinite(Fe_H)) & (mask_J | mask_B)
    obs_hist = _smoothed_hist(Fe_H[obs_mask], feh_bins, sigma_bins=1.2)

    ax_side.cla()  # refresh the side axis
    if obs_hist is not None:
        yC, nC = obs_hist

        ax_side.fill_betweenx(
            yC, 0, nC,
            facecolor='none', hatch='///', edgecolor='blue', linewidth=0, alpha=1.0,
            label='Observed Fe/H'
        )

        ax_side.fill_betweenx(
            yC, 0, nC,
            facecolor='none', hatch='\\\\\\', edgecolor='orange', linewidth=0, alpha=1.0
        )

        ax_side.plot(nC, yC, color='green', lw=2)

    # Best-model MDF (if available) — smooth to match obs treatment
    if hasattr(GalGA, 'mdf_data') and len(GalGA.mdf_data) == len(GalGA.results):
        for (mdf_x, mdf_y), res in zip(GalGA.mdf_data, GalGA.results):
            params = (float(res[5]), float(res[7]), float(res[9]))
            if all(abs(p - b) < 1e-12 for p, b in zip(params, best_params)):
                mdf_x = np.asarray(mdf_x, float)
                mdf_y = np.asarray(mdf_y, float)
                ok = np.isfinite(mdf_x) & np.isfinite(mdf_y) & (mdf_y > 0)
                if np.count_nonzero(ok) > 1:
                    # Bin on same grid then smooth
                    counts, edges = np.histogram(mdf_x[ok], bins=feh_bins, weights=mdf_y[ok])
                    counts = counts.astype(float)
                    counts_s = gaussian_filter1d(counts, sigma=1.2, mode='nearest')
                    if counts_s.max() > 0:
                        counts_s /= counts_s.max()
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    ax_side.fill_betweenx(centers, 0, counts_s, color='red', alpha=0.20, label='Best model MDF')
                    ax_side.plot(counts_s, centers, color='red', lw=2, ls='--')
                break

    # Legend on the side panel
    handles_side, labels_side = ax_side.get_legend_handles_labels()
    if handles_side:
        ax_side.legend(loc='lower right', fontsize=9, frameon=True)


    # ---- cosmetics ----
    # Main
    ax_main.set_xlim(0, age_limit_gyr)
    ax_main.set_ylim(-2.0, 1.0)
    ax_main.set_ylabel('[Fe/H]', fontsize=14)
    ax_main.tick_params(axis='x', labelbottom=False)
    leg = ax_main.legend(loc='upper left', fontsize=10, frameon=True)
    leg.get_frame().set_alpha(0.9)

    # Residuals
    ax_res.set_xlabel('Age (Gyr)', fontsize=14)
    ax_res.set_ylabel('Model − Obs [Fe/H]', fontsize=12)
    ax_res.set_xlim(0, age_limit_gyr)
    ax_res.legend(loc='upper left', fontsize=10, frameon=True)

    # Side histogram axis
    ax_side.set_xlabel('Normalized counts', fontsize=12)
    ax_side.set_xlim(0, 1.15)
    ax_side.set_ylim(ax_main.get_ylim())  # ensure identical y-range
    ax_side.yaxis.set_label_position('right')
    ax_side.yaxis.tick_right()
    ax_side.tick_params(axis='y', labelright=True, labelleft=False, length=3)

    # tidy spines/ticks
    ax_side.grid(False)
    # combine legends
    handles_side, labels_side = ax_side.get_legend_handles_labels()
    if handles_side:
        ax_side.legend(loc='lower right', fontsize=9, frameon=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")
    return fig






def plot_mdf_curves(GalGA, feh, normalized_count, results_df=None, save_path=None):
    """
    All MDF curves (grey), best model (red), data points (black x), + residuals panel.
    """
    if save_path is None:
        save_path = os.path.join(GalGA.output_path, "MDF_multiple_results.png")

    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        best_params = (bm["sigma_2"], bm["t_2"], bm["infall_2"])
    else:
        r = GalGA.results[0]
        best_params = (r[5], r[7], r[9])

    # Plot all MDFs, highlight best
    best_x = best_y = None
    alpha = max(0.02, min(0.6, 8.0 / max(1, len(GalGA.results))))
    for (x, y), label, res in zip(GalGA.mdf_data, GalGA.labels, GalGA.results):
        is_best = all(abs(p - b) < 1e-5 for p, b in zip((res[5], res[7], res[9]), best_params))
        if is_best:
            best_x, best_y = np.asarray(x), np.asarray(y)
        else:
            ax_main.plot(x, y, color="0.75", alpha=0.15 * alpha, lw=0.8, zorder=1)

    if best_x is not None:
        ax_main.plot(best_x, best_y, color="crimson", lw=1.8, label="Model", zorder=3)

    # Data (MDF histogram points)
    ax_main.plot(feh, normalized_count, "x", color="k", ms=4.5, mew=0.9, label="Data", zorder=4)

    ax_main.set_xlim(-2, 1)
    ax_main.set_ylabel("Normalized number")
    ax_main.legend(loc="upper left", fontsize=9, handlelength=1.6)
    ax_main.xaxis.set_ticks_position("top")
    ax_main.xaxis.set_label_position("top")
    ax_main.set_xlabel("[Fe/H]")
    ax_main.tick_params(axis="x", bottom=False)

    

    # Residuals (model - data) on a shared Fe/H grid (simple linear interp of best model)
    if best_x is not None:
        f = interp1d(best_x, best_y, kind="linear", bounds_error=False, fill_value=np.nan)
        y_model_on_data = f(feh)
        resids = y_model_on_data - normalized_count
        ax_res.axhline(0.0, color="0.3", lw=1)
        ax_res.plot(feh, resids, "-", color="0.1", lw=1)
        ax_res.set_ylabel("Model − Data")
        ax_res.set_xlabel("[Fe/H]")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return fig










def plot_four_panel_alpha(GalGA, Fe_H, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe, results_df=None, save_path=None):
    """
    Four-panel [alpha/Fe] vs [Fe/H] with marginal histograms (top: [Fe/H], right: [alpha/Fe]).
    - Observations: black points; histograms filled.
    - Best model: red track; histograms as step outlines (no fill).
    - Other models: faint gray tracks.
    """
    if save_path is None:
        save_path = GalGA.output_path + 'Four_Panel_Alpha.png'

    element_names = ['Mg', 'Si', 'Ca', 'Ti']
    observational_data = [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]

    # pick "best" params (from results_df first row if provided, else first in GalGA.results)
    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        best_params = (float(bm['sigma_2']), float(bm['t_2']), float(bm['infall_2']))
    else:
        r = GalGA.results[0]
        best_params = (float(r[5]), float(r[7]), float(r[9]))

    # figure + 2x2 grid
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), sharex=False, sharey=False)
    # keep panels close; we'll fine-tune with tiny pads for the marginal axes below
    fig.subplots_adjust(hspace=0.1, wspace=0.1, left=0.07, right=0.94, top=0.97, bottom=0.08)

    xlim = (-2.0, 1.0)
    ylim = (-0.8, 0.8)
    xbins = np.linspace(xlim[0], xlim[1], 36)
    ybins = np.linspace(ylim[0], ylim[1], 36)

    # helper: fetch best-model track arrays for a given element index
    def get_best_track(idx):
        for alpha_arrs, _, res in zip(GalGA.alpha_data, GalGA.labels, GalGA.results):
            params = (float(res[5]), float(res[7]), float(res[9]))
            if all(abs(p - b) < 1e-5 for p, b in zip(params, best_params)) and idx < len(alpha_arrs):
                x = np.asarray(alpha_arrs[idx][0])
                y = np.asarray(alpha_arrs[idx][1])
                # your smoother (kept as in your original)
                x, y = smooth_alpha_track_time_ordered(x, y, sigma=3)
                return x, y
        return None, None

    for idx, (element, obs_data) in enumerate(zip(element_names, observational_data)):
        row, col = divmod(idx, 2)
        ax_main = axes[row, col]

        # draw model curves: best in red, others light gray
        for alpha_arrs, _, res in zip(GalGA.alpha_data, GalGA.labels, GalGA.results):
            if idx >= len(alpha_arrs):
                continue
            x_curve = np.asarray(alpha_arrs[idx][0])
            y_curve = np.asarray(alpha_arrs[idx][1])
            x_curve, y_curve = smooth_alpha_track_time_ordered(x_curve, y_curve, sigma=3)

            params = (float(res[5]), float(res[7]), float(res[9]))
            if all(abs(p - b) < 1e-5 for p, b in zip(params, best_params)):
                ax_main.plot(x_curve, y_curve, color="red", lw=2.5, zorder=3)
            else:
                ax_main.plot(x_curve, y_curve, color='gray', alpha=0.03, lw=1.0, zorder=1)

        # observations: clean and scatter
        obs_y = np.where((obs_data >= ylim[0]) & (obs_data <= ylim[1]), obs_data, np.nan)
        mask = np.isfinite(Fe_H) & np.isfinite(obs_y)
        if np.count_nonzero(mask) > 5:
            ax_main.scatter(Fe_H[mask], obs_y[mask], c='k', s=16, zorder=2, edgecolor='none')

        # axes limits/labels
        ax_main.set_xlim(*xlim)
        ax_main.set_ylim(*ylim)
        if col == 0:
            ax_main.set_ylabel(r"[$\alpha$/Fe]")
        else:
            # no duplicate y label on right column
            ax_main.set_ylabel("")
            ax_main.tick_params(axis='y', labelleft=False)

        if row == 1:
            ax_main.set_xlabel("[Fe/H]")
        else:
            ax_main.tick_params(axis='x', labelbottom=False)

        # element tag (boxed to avoid overlapping points)
        ax_main.text(0.05, 0.95, element, transform=ax_main.transAxes,
                     ha='left', va='top', fontsize=25, weight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

        # ----- marginal histograms (keep panels tight) -----
        divider = make_axes_locatable(ax_main)
        ax_top   = divider.append_axes("top",   size="16%", pad=0.04, sharex=ax_main)
        ax_right = divider.append_axes("right", size="16%", pad=0.04, sharey=ax_main)

        # TOP: Fe/H histogram (obs filled, model step)
        if np.count_nonzero(mask) > 5:
            ax_top.hist(Fe_H[mask],  bins=xbins, density=True, histtype='step', lw=1.5, color='black')

        x_best, y_best = get_best_track(idx)
        if x_best is not None:
            ax_top.hist(x_best[np.isfinite(x_best)], bins=xbins, density=True,
                        histtype='step', lw=1.5, color='red')

        # RIGHT: alpha histogram (obs filled, model step) – horizontal
        if np.count_nonzero(mask) > 5:
            ax_right.hist(obs_y[mask], bins=ybins, density=True,
                          histtype='step', lw=1.5, color='black', orientation='horizontal')

        if y_best is not None:
            ax_right.hist(y_best[np.isfinite(y_best)], bins=ybins, density=True,
                          histtype='step', lw=1.5, color='red', orientation='horizontal')

        # clean up marginal axes (no labels/ticks, invisible spines)
        for axm in (ax_top, ax_right):
            axm.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for s in axm.spines.values():
                s.set_visible(False)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Four-panel alpha plot with marginal histograms saved to {save_path}")



