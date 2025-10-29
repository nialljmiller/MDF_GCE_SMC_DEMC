import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
import corner

# Import posterior utilities
from posterior_plotting_package.posterior_utils import *

from posterior_plotting_package.posterior_utils_density import * #plot_density_posterior_simple, plot_density_posterior_simple_vertical

from plotting.style import *
use_paper_style()

def smooth_alpha_track_time_ordered(x_data, y_data, sigma=3):
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    x = np.asarray(x_data)[mask]
    y = np.asarray(y_data)[mask]
    if len(x) < 10:
        return x_data, y_data
    return gaussian_filter1d(x, sigma=sigma, mode='nearest'), gaussian_filter1d(y, sigma=sigma, mode='nearest')


def post_plot_age_feh_detailed(
    GalGA,
    Fe_H,
    age_Joyce,
    age_Bensby,
    results_df=None,
    save_path=None,
    n_bins=12,
    feh_bins=None,
    age_limit_gyr=14.2,
    use_posterior=True,
    percentile=10
):

    if percentile == -1:
        percentile = choose_cutoff_lognorm_mixture(results_df['fitness'].values, bins=100, kde_points=1024, em_max_iter=200, tol=1e-6)

    if save_path is None:
        save_path = GalGA.output_path + 'Age_Metallicity_posterior.png'

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    
    # Sanitize arrays
    Fe_H = np.asarray(Fe_H, dtype=float)
    age_Joyce = np.asarray(age_Joyce, dtype=float)
    age_Bensby = np.asarray(age_Bensby, dtype=float)
    
    # Figure layout: main + residuals + side histogram
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[4, 1],
        height_ratios=[3, 1],
        wspace=0.0,
        hspace=0.0,
        left=0.07, right=0.97, top=0.96, bottom=0.08
    )
    
    ax_main = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_side = fig.add_subplot(gs[0, 1], sharey=ax_main)
    

    #top_df, weights = get_weighted_posterior_samples(results_df, fitness_col='fitness', percentile=percentile)
    top_df, weights = posterior_resample(results_df, weight_col='posterior_w', fitness_col='fitness', percentile=percentile, resampling='systematic')

    # Compute age-[Fe/H] ensemble
    ensemble = compute_age_feh_ensemble(GalGA, top_df, weights, 
                                       age_range=(0, age_limit_gyr), 
                                       n_bins=200)
    
    age_common = ensemble['x']
    median_feh = ensemble['median']
    lower_feh = ensemble['lower']
    upper_feh = ensemble['upper']
    
    # Plot 1σ uncertainty band with density shading
    plot_density_posterior_simple(ax_main, age_common, median_feh, 
                                 lower_feh, upper_feh, 
                                 color='crimson', n_levels=20, 
                                 zorder=4, label='1σ posterior')
    
    # Store for residuals
    best_age_gyr = age_common
    best_feh = median_feh
    
    # ---- Overlay raw observational points ----
    mask_J = np.isfinite(age_Joyce) & np.isfinite(Fe_H)
    mask_B = np.isfinite(age_Bensby) & np.isfinite(Fe_H)
    
    ax_main.scatter(age_Joyce[mask_J], Fe_H[mask_J], marker='*', s=55,
                    color='blue', alpha=0.7, zorder=6, label='Joyce et al. (raw)')
    ax_main.scatter(age_Bensby[mask_B], Fe_H[mask_B], marker='^', s=55,
                    color='orange', alpha=0.7, zorder=6, label='Bensby et al. (raw)')
    
    # ---- Binned curves with errors for Joyce and Bensby ----
    def _binned(age, feh, bins):
        m = np.isfinite(age) & np.isfinite(feh)
        if np.count_nonzero(m) < 3:
            return None
        means, _, _ = binned_statistic(age[m], feh[m], statistic='mean', bins=bins)
        stds, _, _ = binned_statistic(age[m], feh[m], statistic='std', bins=bins)
        cnts, _, _ = binned_statistic(age[m], feh[m], statistic='count', bins=bins)
        ctrs = 0.5 * (bins[:-1] + bins[1:])
        sem = stds / np.sqrt(np.maximum(cnts, 1))
        valid = (cnts > 0) & np.isfinite(means)
        return ctrs[valid], means[valid], stds[valid], sem[valid]
    
    age_bins = np.linspace(0, age_limit_gyr, n_bins + 1)
    J = _binned(age_Joyce, Fe_H, age_bins)
    B = _binned(age_Bensby, Fe_H, age_bins)
    
    xc, ym, ys, ysem = J
    ax_main.plot(xc, ym, color='blue', lw=2.5, zorder=7, label='Joyce (binned)')
    ax_main.errorbar(xc, ym, yerr=ys, color='blue', alpha=0.3, lw=1.0, capsize=3, zorder=6)

    xc, ym, ys, ysem = B
    ax_main.plot(xc, ym, color='orange', lw=2.5, zorder=7, label='Bensby (binned)')
    ax_main.errorbar(xc, ym, yerr=ys, color='orange', alpha=0.3, lw=1.0, capsize=3, zorder=6)
    
    # ---- Residuals for the median/best model ----
    def _interp_clean(x, y):
        idx = np.argsort(x)
        xs, ys = x[idx], y[idx]
        keep = np.ones_like(xs, dtype=bool)
        keep[1:] = (np.diff(xs) > 1e-12)
        return xs[keep], ys[keep]
    
    residuals_all = []
    
    xs, ys = _interp_clean(best_age_gyr, best_feh)
    f_best = interp1d(xs, ys, kind='linear', bounds_error=False, fill_value=np.nan)
    
    # Joyce residuals
    ageJ = age_Joyce[mask_J]
    fehJ = Fe_H[mask_J]
    rng = (ageJ >= np.nanmin(xs)) & (ageJ <= np.nanmax(xs))
    mj = f_best(ageJ[rng])
    rj = mj - fehJ[rng]
    v = np.isfinite(rj)
    ax_res.scatter(ageJ[rng][v], rj[v], marker='*', s=40,
                   color='blue', alpha=0.8, label='Joyce residuals')
    residuals_all.append(rj[v])
            
    # Bensby residuals
    ageB = age_Bensby[mask_B]
    fehB = Fe_H[mask_B]
    rng = (ageB >= np.nanmin(xs)) & (ageB <= np.nanmax(xs))
    mb = f_best(ageB[rng])
    rb = mb - fehB[rng]
    v = np.isfinite(rb)
    ax_res.scatter(ageB[rng][v], rb[v], marker='^', s=40,
                   color='orange', alpha=0.8, label='Bensby residuals')
    residuals_all.append(rb[v])


    # Zero line + autoscale for residuals
    ax_res.axhline(0.0, ls='--', lw=1.0, color='black', alpha=0.7)
    res = np.concatenate(residuals_all)
    s = np.nanstd(res)
    ylim = max(0.5, 3.0 * s)
    ax_res.set_ylim(-ylim, +ylim)
    
    # ---- Sideways histogram (Fe/H distributions) ----
    feh_bins = np.linspace(-2.0, 1.0, 28)
    
    def _smoothed_hist(vals, bins, sigma_bins=1.2):
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

    
    # Observed Fe/H distribution
    obs_mask = (np.isfinite(Fe_H)) & (mask_J | mask_B)
    obs_hist = _smoothed_hist(Fe_H[obs_mask], feh_bins, sigma_bins=1.2)
    
    ax_side.cla()
    yC, nC = obs_hist
    ax_side.fill_betweenx(yC, 0, nC, facecolor='none', hatch='///', 
                          edgecolor='blue', linewidth=0, alpha=1.0,
                          label='Observed Fe/H')
    ax_side.fill_betweenx(yC, 0, nC, facecolor='none', hatch='\\\\\\', 
                          edgecolor='orange', linewidth=0, alpha=1.0)
    ax_side.plot(nC, yC, color='green', lw=2)

    # Compute MDF ensemble
    top_df, weights = get_weighted_posterior_samples(results_df, 
                                                     fitness_col='fitness', 
                                                     percentile=percentile)

    mdf_ensemble = compute_mdf_ensemble(GalGA, top_df, weights, 
                                       feh_range=(-2.0, 1.0), n_bins=50)
    # Bin to match feh_bins for histogram
    mdf_hist = np.interp(0.5 * (feh_bins[:-1] + feh_bins[1:]),
                        mdf_ensemble['x'], mdf_ensemble['median'],
                        left=0, right=0)

    mdf_hist /= mdf_hist.max()

    centers = 0.5 * (feh_bins[:-1] + feh_bins[1:])

    ax_side.fill_betweenx(centers, 0, mdf_hist, color='crimson', 
                         alpha=0.20, label='Median model MDF')

    ax_side.plot(mdf_hist, centers, color='crimson', lw=2, ls='--')
    
    # Legend on the side panel
    handles_side, labels_side = ax_side.get_legend_handles_labels()
    ax_side.legend(loc='lower right', fontsize=9, frameon=True)
    
    # ---- Cosmetics ----
    ax_main.set_xlim(0, age_limit_gyr)
    ax_main.set_ylim(-2.0, 1.0)
    ax_main.set_ylabel('[Fe/H]', fontsize=14)
    ax_main.tick_params(axis='x', labelbottom=False)
    leg = ax_main.legend(loc='upper left', fontsize=10, frameon=True)
    leg.get_frame().set_alpha(0.9)
    
    ax_res.set_xlabel('Age (Gyr)', fontsize=14)
    ax_res.set_ylabel('Model − Obs [Fe/H]', fontsize=12)
    ax_res.set_xlim(0, age_limit_gyr)
    ax_res.legend(loc='upper left', fontsize=10, frameon=True)
    
    ax_side.set_xlabel('Normalized counts', fontsize=12)
    ax_side.set_xlim(0, 1.15)
    ax_side.set_ylim(ax_main.get_ylim())
    ax_side.yaxis.set_label_position('right')
    ax_side.yaxis.tick_right()
    ax_side.tick_params(axis='y', labelright=True, labelleft=False, length=3)
    ax_side.grid(False)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")
    return fig




def post_plot_mdf_curves(GalGA, feh, normalized_count, results_df=None, save_path=None, use_posterior=True, percentile=10):

    if percentile == -1:
        percentile = choose_cutoff_lognorm_mixture(results_df['fitness'].values, bins=100, kde_points=1024, em_max_iter=200, tol=1e-6)

    if save_path is None:
        save_path = os.path.join(GalGA.output_path, "MDF_posterior.png")
    
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )
    
    # POSTERIOR MODE
    print(f"Computing MDF posterior from top {percentile}% of models...")
    
    top_df, weights = get_weighted_posterior_samples(results_df, 
                                                     fitness_col='fitness', 
                                                     percentile=percentile)
    
    mdf_ensemble = compute_mdf_ensemble(GalGA, top_df, weights, 
                                       feh_range=(-2.0, 1.0), n_bins=100)
    
    feh_common = mdf_ensemble['x']
    median_mdf = mdf_ensemble['median']
    lower_mdf = mdf_ensemble['lower']
    upper_mdf = mdf_ensemble['upper']
    
    # Plot 1σ uncertainty band with density shading
    plot_density_posterior_simple(ax_main, feh_common, median_mdf,
                                 lower_mdf, upper_mdf,
                                 color='crimson', n_levels=20,
                                 zorder=2, label='1σ posterior')
    
    best_x, best_y = feh_common, median_mdf
    
    # Data (MDF histogram points)
    ax_main.plot(feh, normalized_count, "x", color="k", ms=4.5, mew=0.9, label="Data", zorder=4)
    
    ax_main.set_xlim(-2, 1)
    ax_main.set_ylabel("Normalized number")
    ax_main.legend(loc="upper left", fontsize=9, handlelength=1.6)
    ax_main.xaxis.set_ticks_position("top")
    ax_main.xaxis.set_label_position("top")
    ax_main.set_xlabel("[Fe/H]")
    ax_main.tick_params(axis="x", bottom=False)
    
    # Residuals
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
    print(f"Saved: {save_path}")
    return fig









def post_plot_four_panel_alpha(GalGA, Fe_H, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe, results_df=None, save_path=None, use_posterior=True, percentile=10):

    if percentile == -1:
        percentile = choose_cutoff_lognorm_mixture(results_df['fitness'].values, bins=100, kde_points=1024, em_max_iter=200, tol=1e-6)

    if save_path is None:
        save_path = GalGA.output_path + 'Four_Panel_Alpha_Posterior.png'
    
    element_names = ['Mg', 'Si', 'Ca', 'Ti']
    observational_data = [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]
    
    # Figure + 2x2 grid
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), sharex=False, sharey=False)
    fig.subplots_adjust(hspace=0.1, wspace=0.1, left=0.07, right=0.94, top=0.97, bottom=0.08)
    
    xlim = (-2.0, 1.0)
    ylim = (-0.8, 0.8)
    xbins = np.linspace(xlim[0], xlim[1], 36)
    ybins = np.linspace(ylim[0], ylim[1], 36)
    
    for idx, (element, obs_data) in enumerate(zip(element_names, observational_data)):
        row, col = divmod(idx, 2)
        ax_main = axes[row, col]
        
        top_df, weights = get_weighted_posterior_samples(results_df, 
                                                         fitness_col='fitness', 
                                                         percentile=percentile)
        
        alpha_ensemble = compute_alpha_ensemble(GalGA, top_df, weights, 
                                               element_idx=idx,
                                               feh_range=xlim, n_bins=150)
        
        feh_common = alpha_ensemble['x']
        median_alpha = alpha_ensemble['median']
        lower_alpha = alpha_ensemble['lower']
        upper_alpha = alpha_ensemble['upper']
        
        # Plot 1σ uncertainty band with density shading
        plot_density_posterior_simple(ax_main, feh_common, median_alpha,
                                     lower_alpha, upper_alpha,
                                     color='crimson', n_levels=20,
                                     zorder=2, label='1σ posterior')
        
        best_x, best_y = feh_common, median_alpha
        
        
        # Observations
        obs_y = np.where((obs_data >= ylim[0]) & (obs_data <= ylim[1]), obs_data, np.nan)
        mask = np.isfinite(Fe_H) & np.isfinite(obs_y)
        if np.count_nonzero(mask) > 5:
            ax_main.scatter(Fe_H[mask], obs_y[mask], c='k', s=16, zorder=2, edgecolor='none')
        
        # Axes limits/labels
        ax_main.set_xlim(*xlim)
        ax_main.set_ylim(*ylim)
        if col == 0:
            ax_main.set_ylabel(r"[$\alpha$/Fe]")
        else:
            ax_main.set_ylabel("")
            ax_main.tick_params(axis='y', labelleft=False)
        
        if row == 1:
            ax_main.set_xlabel("[Fe/H]")
        else:
            ax_main.tick_params(axis='x', labelbottom=False)
        
        # Element tag
        ax_main.text(0.05, 0.95, element, transform=ax_main.transAxes,
                     ha='left', va='top', fontsize=25, weight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        
        # Marginal histograms
        divider = make_axes_locatable(ax_main)
        ax_top = divider.append_axes("top", size="16%", pad=0.04, sharex=ax_main)
        ax_right = divider.append_axes("right", size="16%", pad=0.04, sharey=ax_main)
        
        # TOP: Fe/H histogram
        ax_top.hist(Fe_H[mask], bins=xbins, density=True, histtype='step', lw=1.5, color='black')
        
        ax_top.hist(best_x[np.isfinite(best_x)], bins=xbins, density=True,
                        histtype='step', lw=1.5, color='crimson')
        
        ax_right.hist(obs_y[mask], bins=ybins, density=True,
                          histtype='step', lw=1.5, color='black', orientation='horizontal')
        
        ax_right.hist(best_y[np.isfinite(best_y)], bins=ybins, density=True,
                          histtype='step', lw=1.5, color='crimson', orientation='horizontal')
        
        # Clean up marginal axes
        for axm in (ax_top, ax_right):
            axm.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for s in axm.spines.values():
                s.set_visible(False)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Four-panel alpha plot with posterior saved to {save_path}")


def post_plot_corner(GalGA, results_df=None, save_path=None, use_posterior=True, percentile=None, nsamples=999, metric_val = 'fitness'):

    if save_path is None:
        save_path = GalGA.output_path

    df = results_df.sort_values(metric_val).reset_index(drop=True)
    loss = df[metric_val].values
    weights, _, _ = compute_weights(loss)

    if metric_val == 'physics_penalty':
        weights = 1/weights


    params = [
        "sigma_2",
        "t_1",
        "t_2",
        "infall_1",
        "infall_2",
        "sfe",
        "delta_sfe",
        "imf_upper",
        "mgal",
        "nb",
    ]
    data = df[params].to_numpy()

    post_csv = save_path + "/posteriors.csv"
    df[params].to_csv(post_csv, index=False)
    weights_csv = save_path + "/posterior_weights.csv"
    df_weights = df.assign(weight=weights)
    df_weights.to_csv(weights_csv, index=False)

    _save_corner(df[params], df[metric_val], save_path + "/" + metric_val + "_corner.png")


def _save_corner(samples, weights, out_path):

    percentile = choose_cutoff_lognorm_mixture(weights, bins=100, kde_points=1024, em_max_iter=200, tol=1e-6)


    data = samples.to_numpy()
    labels = [c.replace("_", " ") for c in samples.columns]
    title_fmt = ".3g"

    fig = corner.corner(
        data,
        labels=labels,
        weights=weights,
        quantiles=[0.16, 0.5, 0.84],  # Add quantiles for MCMC-like credibility intervals
        show_titles=True,  # Enable built-in titles for quantiles
        title_fmt=title_fmt,
        bins=40,
        smooth=0.8,
        levels=[percentile]#[1 - np.exp(-0.5 * r**2) for r in [2]],  # 1σ and 2σ contours
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



