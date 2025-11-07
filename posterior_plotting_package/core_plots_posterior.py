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
from scipy.ndimage import gaussian_filter   

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



def post_plot_mdf_curves(
    GalGA,
    feh,
    normalized_count,
    results_df=None,
    save_path=None,
    use_posterior=True,
    percentile=10,
):
    if percentile == -1:
        percentile = choose_cutoff_lognorm_mixture(
            results_df["fitness"].values,
            bins=100, kde_points=1024, em_max_iter=200, tol=1e-6
        )

    if save_path is None:
        save_path = os.path.join(GalGA.output_path, "MDF_posterior.png")

    # --- observed MDF (sorted, finite) ---
    feh = np.asarray(feh, float)
    normalized_count = np.asarray(normalized_count, float)
    m = np.isfinite(feh) & np.isfinite(normalized_count)
    idx = np.argsort(feh[m])
    x_data = feh[m][idx]
    y_data = normalized_count[m][idx]

    # --- posterior draws (systematic resampling) ---
    draw_df, draw_w = posterior_resample(
        results_df,
        weight_col="posterior_w",
        fitness_col="fitness",
        percentile=percentile,
        resampling="systematic",
    )

    # --- common x grid tied to data support ---
    n_bins = x_data.size * 2
    x_lo, x_hi = float(x_data.min()), float(x_data.max())


    print(draw_w, draw_df, results_df)
    # compute posterior ensemble from actual draws (no parametric shape assumptions)
    ens = compute_mdf_ensemble(
        GalGA,
        draw_df,
        draw_w,
        feh_range=(x_lo, x_hi),
        n_bins=n_bins
    )
    print(ens)

    x_common = ens["x"]
    y_med = ens["median"]
    y_lo = ens["lower"]
    y_hi = ens["upper"]

    # put DATA onto the same grid (linear interp, no smoothing)
    f_data_on_common = interp1d(
        x_data, y_data, kind="linear", bounds_error=False, fill_value=np.nan
    )
    y_data_common = f_data_on_common(x_common)

    # --- figure: main + residuals ---
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    # posterior band (credible region from real draws)
    plot_density_posterior_simple(
        ax_main, x_common, y_med, y_lo, y_hi,
        color="crimson", n_levels=20, zorder=2, label="1σ posterior"
    )

    # observed points
    ax_main.plot(
        x_data, y_data, "x", color="k", ms=4.5, mew=0.9, label="Data", zorder=4
    )

    ax_main.set_xlim(x_lo, x_hi)
    ax_main.set_ylabel("Normalized number")
    ax_main.legend(loc="upper left", fontsize=9, handlelength=1.6)
    ax_main.xaxis.set_ticks_position("top")
    ax_main.xaxis.set_label_position("top")
    ax_main.set_xlabel("[Fe/H]")
    ax_main.tick_params(axis="x", bottom=False)

    # residual posterior (model − data), same draws, same grid
    r_med = y_med - y_data_common
    r_lo  = y_lo  - y_data_common
    r_hi  = y_hi  - y_data_common

    plot_density_posterior_simple(
        ax_res, x_common, r_med, r_lo, r_hi,
        color="0.1", n_levels=12, zorder=2, label=None
    )
    ax_res.axhline(0.0, color="0.3", lw=1)
    ax_res.set_ylabel("Model − Data")
    ax_res.set_xlabel("[Fe/H]")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return fig


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
    percentile=10,
):
    # percentile via mixture cutoff (if requested)
    if percentile == -1:
        percentile = choose_cutoff_lognorm_mixture(
            results_df["fitness"].values,
            bins=100, kde_points=1024, em_max_iter=200, tol=1e-6
        )

    if save_path is None:
        save_path = os.path.join(GalGA.output_path, "Age_Metallicity_posterior.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # sanitize obs arrays
    Fe_H = np.asarray(Fe_H, float)
    age_Joyce = np.asarray(age_Joyce, float)
    age_Bensby = np.asarray(age_Bensby, float)

    # figure: main + residuals + side histogram
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[4, 1],
        height_ratios=[3, 1],
        wspace=0.0, hspace=0.0,
        left=0.07, right=0.97, top=0.96, bottom=0.08
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_res  = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_side = fig.add_subplot(gs[0, 1], sharey=ax_main)

    # posterior draws from actual samples (systematic resampling)
    top_df, weights = posterior_resample(
        results_df,
        weight_col="posterior_w",
        fitness_col="fitness",
        percentile=percentile,
        resampling="systematic",
    )

    # age–[Fe/H] ensemble on a common age grid
    ens = compute_age_feh_ensemble(
        GalGA, top_df, weights,
        age_range=(0.0, age_limit_gyr),
        n_bins=200
    )
    age_common = ens["x"]
    feh_med    = ens["median"]
    feh_lo     = ens["lower"]
    feh_hi     = ens["upper"]

    # posterior credible band (sample-based; no analytic shape assumption)
    plot_density_posterior_simple(
        ax_main, age_common, feh_med, feh_lo, feh_hi,
        color="crimson", n_levels=20, zorder=4, label="1σ posterior"
    )

    # observations (raw)
    mask_J = np.isfinite(age_Joyce) & np.isfinite(Fe_H)
    mask_B = np.isfinite(age_Bensby) & np.isfinite(Fe_H)
    ax_main.scatter(age_Joyce[mask_J], Fe_H[mask_J], marker="*", s=55,
                    color="blue", alpha=0.7, zorder=6, label="Joyce et al. (raw)")
    ax_main.scatter(age_Bensby[mask_B], Fe_H[mask_B], marker="^", s=55,
                    color="orange", alpha=0.7, zorder=6, label="Bensby et al. (raw)")

    # binned curves + error bars for each dataset
    def _binned(age, feh, bins):
        m = np.isfinite(age) & np.isfinite(feh)
        means, _, _ = binned_statistic(age[m], feh[m], statistic="mean", bins=bins)
        stds,  _, _ = binned_statistic(age[m], feh[m], statistic="std",  bins=bins)
        cnts,  _, _ = binned_statistic(age[m], feh[m], statistic="count", bins=bins)
        ctrs = 0.5 * (bins[:-1] + bins[1:])
        sem = stds / np.sqrt(np.maximum(cnts, 1))
        valid = (cnts > 0) & np.isfinite(means)
        return ctrs[valid], means[valid], stds[valid], sem[valid]

    age_bins = np.linspace(0.0, age_limit_gyr, n_bins + 1)
    Jc, Jm, Js, _ = _binned(age_Joyce, Fe_H, age_bins)
    Bc, Bm, Bs, _ = _binned(age_Bensby, Fe_H, age_bins)

    ax_main.plot(Jc, Jm, color="blue", lw=2.5, zorder=7, label="Joyce (binned)")
    ax_main.errorbar(Jc, Jm, yerr=Js, color="blue", alpha=0.3, lw=1.0, capsize=3, zorder=6)

    ax_main.plot(Bc, Bm, color="orange", lw=2.5, zorder=7, label="Bensby (binned)")
    ax_main.errorbar(Bc, Bm, yerr=Bs, color="orange", alpha=0.3, lw=1.0, capsize=3, zorder=6)

    # residuals vs posterior median
    def _interp_clean(x, y):
        idx = np.argsort(x)
        xs, ys = x[idx], y[idx]
        keep = np.ones_like(xs, dtype=bool)
        keep[1:] = (np.diff(xs) > 1e-12)
        return xs[keep], ys[keep]

    xs, ys = _interp_clean(age_common, feh_med)
    f_med = interp1d(xs, ys, kind="linear", bounds_error=False, fill_value=np.nan)

    res_all = []

    ageJ, fehJ = age_Joyce[mask_J], Fe_H[mask_J]
    rng = (ageJ >= np.nanmin(xs)) & (ageJ <= np.nanmax(xs))
    rj  = f_med(ageJ[rng]) - fehJ[rng]
    v   = np.isfinite(rj)
    ax_res.scatter(ageJ[rng][v], rj[v], marker="*", s=40, color="blue", alpha=0.8, label="Joyce residuals")
    res_all.append(rj[v])

    ageB, fehB = age_Bensby[mask_B], Fe_H[mask_B]
    rng = (ageB >= np.nanmin(xs)) & (ageB <= np.nanmax(xs))
    rb  = f_med(ageB[rng]) - fehB[rng]
    v   = np.isfinite(rb)
    ax_res.scatter(ageB[rng][v], rb[v], marker="^", s=40, color="orange", alpha=0.8, label="Bensby residuals")
    res_all.append(rb[v])

    ax_res.axhline(0.0, ls="--", lw=1.0, color="black", alpha=0.7)
    res = np.concatenate(res_all) if len(res_all) else np.array([0.0])
    s = np.nanstd(res)
    ax_res.set_ylim(-max(0.5, 3.0 * s), +max(0.5, 3.0 * s))

    # side panel: Fe/H distributions + MDF posterior from the SAME draws
    if feh_bins is None:
        feh_bins = np.linspace(-2.0, 1.0, 28)

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

    obs_mask = (np.isfinite(Fe_H)) & (mask_J | mask_B)
    centers_obs, norm_counts = _smoothed_hist(Fe_H[obs_mask], feh_bins, sigma_bins=1.2)

    ax_side.fill_betweenx(centers_obs, 0, norm_counts, facecolor="none", hatch="///",
                          edgecolor="blue", linewidth=0, alpha=1.0, label="Observed Fe/H")
    ax_side.plot(norm_counts, centers_obs, color="green", lw=2)

    mdf_ens = compute_mdf_ensemble(
        GalGA, top_df, weights,
        feh_range=(-2.0, 1.0), n_bins=50
    )
    centers = 0.5 * (feh_bins[:-1] + feh_bins[1:])
    med = np.interp(centers, mdf_ens["x"], mdf_ens["median"], left=0, right=0)
    lo  = np.interp(centers, mdf_ens["x"], mdf_ens["lower"],  left=0, right=0)
    hi  = np.interp(centers, mdf_ens["x"], mdf_ens["upper"],  left=0, right=0)

    s = med.max() if np.isfinite(med).any() else 1.0
    if s > 0:
        med /= s; lo /= s; hi /= s

    ax_side.fill_betweenx(centers, lo, hi, color="crimson", alpha=0.18, label="MDF 1σ posterior")
    ax_side.plot(med, centers, color="crimson", lw=2, ls="--", label="Median MDF")

    # cosmetics
    ax_main.set_ylabel("[Fe/H]")
    ax_main.set_xlabel("")  # x on top panel suppressed; residuals carry x label
    ax_main.legend(loc="best", fontsize=10)
    ax_res.set_xlabel("Age (Gyr)")
    ax_res.set_ylabel("Model − Data")

    ax_side.legend(loc="lower right", fontsize=9, frameon=True)
    ax_side.set_xlabel("Normalized counts", fontsize=12)
    ax_side.set_xlim(0, 1.15)
    ax_side.set_ylim(ax_main.get_ylim())
    ax_side.yaxis.set_label_position("right")
    ax_side.yaxis.tick_right()
    ax_side.tick_params(axis="y", labelright=True, labelleft=False, length=3)
    ax_side.grid(False)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return fig








def post_plot_four_panel_alpha(
    GalGA,
    Fe_H,
    Mg_Fe,
    Si_Fe,
    Ca_Fe,
    Ti_Fe,
    results_df=None,
    save_path=None,
    use_posterior=True,
    percentile=10,
):
    if percentile == -1:
        percentile = choose_cutoff_lognorm_mixture(
            results_df["fitness"].values, bins=100, kde_points=1024, em_max_iter=200, tol=1e-6
        )

    if save_path is None:
        save_path = os.path.join(GalGA.output_path, "Four_Panel_Alpha_Posterior.png")

    element_names = ["Mg", "Si", "Ca", "Ti"]
    observational_data = [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]




    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), sharex=False, sharey=False)
    fig.subplots_adjust(hspace=0.1, wspace=0.1, left=0.07, right=0.94, top=0.97, bottom=0.08)

    xlim = (-2.0, 1.0)
    ylim = (-0.8, 0.8)
    xbins = np.linspace(xlim[0], xlim[1], 36)
    ybins = np.linspace(ylim[0], ylim[1], 36)

    Fe_H = np.asarray(Fe_H, float)

    for idx, (element, obs_data) in enumerate(zip(element_names, observational_data)):
        row, col = divmod(idx, 2)
        ax_main = axes[row, col]

        # real posterior draws via systematic resampling
        top_df, weights = posterior_resample(
            results_df,
            weight_col="posterior_w",
            fitness_col="fitness",
            percentile=percentile,
            resampling="systematic",
        )

        # α–Fe ensemble on common Fe/H grid
        ens = compute_alpha_ensemble(
            GalGA,
            top_df,
            weights,
            element_idx=idx,
            feh_range=xlim,
            n_bins=150,
        )
        feh_common   = ens["x"]
        median_alpha = ens["median"]
        lower_alpha  = ens["lower"]
        upper_alpha  = ens["upper"]

        # posterior band (sample-based; no parametric shape)
        plot_density_posterior_simple(
            ax_main, feh_common, median_alpha, lower_alpha, upper_alpha,
            color="crimson", n_levels=20, zorder=2, label="1σ posterior"
        )






        # observations
        obs_data = np.asarray(obs_data, float)
        obs_y = np.where((obs_data >= ylim[0]) & (obs_data <= ylim[1]), obs_data, np.nan)
        mask = np.isfinite(Fe_H) & np.isfinite(obs_y)

        # --- NEW: posterior point samples (match N to observations for fair hist compare)
        Nobs = int(np.count_nonzero(mask))
        Nx = max(200, Nobs)   # allow a small floor so the red hist isn’t too jaggy
        post_x, post_y = sample_posterior_points(GalGA, top_df, weights, idx, Nx, feh_range=xlim)

        # main scatter: keep your observed points
        if Nobs > 5:
            ax_main.scatter(Fe_H[mask], obs_y[mask], c='k', s=16, zorder=3, edgecolor='none')




        # axes setup
        ax_main.set_xlim(*xlim)
        ax_main.set_ylim(*ylim)
        if col == 0:
            ax_main.set_ylabel(r"[$\alpha$/Fe]")
        else:
            ax_main.set_ylabel("")
            ax_main.tick_params(axis="y", labelleft=False)
        if row == 1:
            ax_main.set_xlabel("[Fe/H]")
        else:
            ax_main.tick_params(axis="x", labelbottom=False)

        # element tag
        ax_main.text(
            0.05, 0.95, element, transform=ax_main.transAxes,
            ha="left", va="top", fontsize=25, weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
        )

        # marginals (top/right)
        divider = make_axes_locatable(ax_main)
        ax_top   = divider.append_axes("top",   size="16%", pad=0.04, sharex=ax_main)
        ax_right = divider.append_axes("right", size="16%", pad=0.04, sharey=ax_main)

        feh_grid = feh_common
        med = median_alpha
        lo  = lower_alpha
        hi  = upper_alpha

        # normalize curves for marginals (scale-only)
        med_n = med / np.nanmax(np.abs(med))
        lo_n  = lo  / np.nanmax(np.abs(lo))
        hi_n  = hi  / np.nanmax(np.abs(hi))



        # TOP: Fe/H hist (observed vs posterior)
        ax_top.hist(Fe_H[mask], bins=xbins, density=True, histtype="step", lw=1.5, color="black", label="Observed")
        ax_top.hist(post_x[np.isfinite(post_x)], bins=xbins, density=True, histtype="step", lw=1.5, color="crimson", label="Posterior")

        # RIGHT: [α/Fe] hist (observed vs posterior)
        ax_right.hist(obs_y[mask], bins=ybins, density=True, histtype="step", lw=1.5, color="black", orientation="horizontal")
        ax_right.hist(post_y[np.isfinite(post_y)], bins=ybins, density=True, histtype="step", lw=1.5, color="crimson", orientation="horizontal")


        # tidy marginals
        for axm in (ax_top, ax_right):
            axm.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for s in axm.spines.values():
                s.set_visible(False)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Four-panel alpha plot with posterior saved to {save_path}")
























# ---------------- public entry ----------------
def post_plot_corner(GalGA, results_df=None, save_path=None, use_posterior=True,
                     percentile=None, nsamples=999, metric_val='fitness'):
    if save_path is None:
        save_path = GalGA.output_path
    os.makedirs(f"{save_path}/corner/", exist_ok=True)

    df = results_df.sort_values(metric_val).reset_index(drop=True)
    loss = df[metric_val].to_numpy()
    weights, _, _ = compute_weights(loss)

    params = ["sigma_2","t_1","t_2","infall_1","infall_2","sfe","delta_sfe","imf_upper","mgal","nb"]
    S = df[params]



    hdi_mass=0.25
    post_crop=0.4
    bins1d=30
    bins2d=40
    smooth=1.
    smooth1d=0.9
    cmap="ocean"
    cmin=50
    point_size=4
    point_alpha=0.5
    title_fs=18
    dpi=300



    '''
    # plots
    _save_corner_weighted(S, loss,    f"{save_path}/corner/{metric_val}_loss.png",
                        hdi_mass=hdi_mass, post_crop=post_crop,
                        bins1d=bins1d, bins2d=bins2d, smooth=smooth, smooth1d=smooth1d,
                        cmap=cmap, cmin=cmin, point_size=point_size, point_alpha=point_alpha,
                        title_fs=title_fs, dpi=dpi)
                          
    _save_corner_weighted(S, weights, f"{save_path}/corner/{metric_val}_weight.png",
                        hdi_mass=hdi_mass, post_crop=post_crop,
                        bins1d=bins1d, bins2d=bins2d, smooth=smooth, smooth1d=smooth1d,
                        cmap=cmap, cmin=cmin, point_size=point_size, point_alpha=point_alpha,
                        title_fs=title_fs, dpi=dpi)


    # plots
    _save_corner_weighted_cd(S, loss,    f"{save_path}/corner/{metric_val}_loss_cd.png",
                        hdi_mass=hdi_mass, post_crop=post_crop,
                        bins1d=bins1d, bins2d=bins2d, smooth=smooth, smooth1d=smooth1d,
                        cmap=cmap, cmin=cmin, point_size=point_size, point_alpha=point_alpha,
                        title_fs=title_fs, dpi=dpi)
    '''                   
    
    _save_corner_weighted_cd(S, weights, f"{save_path}/corner/{metric_val}_weight_cd.png",
                        hdi_mass=hdi_mass, post_crop=post_crop,
                        bins1d=bins1d, bins2d=bins2d, smooth=smooth, smooth1d=smooth1d,
                        cmap=cmap, cmin=cmin, point_size=point_size, point_alpha=point_alpha,
                        title_fs=title_fs, dpi=dpi)



    # summaries (compact loop)
    for name, w in [("weighted", weights), ("unweighted", np.ones_like(weights)), ("loss", loss)]:
        rows = []
        for col in S.columns:
            x_map, lo, hi = _weighted_mode_and_hdi(S[col].to_numpy(), w, bins=80, mass=0.68)
            rows.append(dict(param=col, MAP=x_map, HPD_lo=lo, HPD_hi=hi))
        pd.DataFrame(rows).to_csv(f"{save_path}/corner/{metric_val}_summary_{name}.csv", index=False)





# ---------------- core stats helpers ----------------
def _weighted_mode_and_hdi(x, w, bins=80, mass=0.68):
    x = np.asarray(x, float); w = np.asarray(w, float)
    dens, edges = np.histogram(x, bins=bins, weights=w, density=True)
    widths = np.diff(edges)
    probs  = dens * widths
    centers = 0.5*(edges[:-1] + edges[1:])
    k_map = int(np.argmax(dens))
    x_map = centers[k_map]
    order = np.argsort(dens)[::-1]
    acc = 0.0; chosen = []
    for k in order:
        chosen.append(k); acc += probs[k]
        if acc >= mass: break
    lo, hi = edges[min(chosen)], edges[max(chosen)+1]
    return float(x_map), float(lo), float(hi)

def _weighted_hdi_1d(x, w, mass=0.68, bins=200, pad=0.02):
    x = np.asarray(x, float); w = np.asarray(w, float)
    H, edges = np.histogram(x, bins=bins, weights=w, density=True)
    dx = np.diff(edges); p = H*dx
    order = np.argsort(H)[::-1]
    acc = 0.0; kept = np.zeros_like(H, bool)
    for k in order:
        kept[k] = True; acc += p[k]
        if acc >= mass: break
    lo = edges[np.argmax(kept)]
    hi = edges[np.where(kept)[0][-1] + 1]
    pad_abs = pad*(hi - lo)
    return float(lo - pad_abs), float(hi + pad_abs)

# ---------------- plotting ----------------
def _save_corner_weighted_cd(samples, weights, out_path, *, hdi_mass=0.25, post_crop=0.80,
                          bins1d=80, bins2d=50, smooth=0.8, smooth1d=0.9,
                          cmap="inferno", cmin=1, point_size=4, point_alpha=0.35,
                          title_fs=18, dpi=300):
    data   = samples.to_numpy(float)
    labels = [c.replace("_", " ") + "\n" for c in samples.columns]
    K = data.shape[1]

    ranges = [_weighted_hdi_1d(data[:, i], weights, bins=max(200, 2*bins1d))
              for i in range(K)]

    h2 = dict(cmap=cmap, bins=bins2d)
    if cmin is not None: h2["cmin"] = cmin

    fig = corner.corner(
        data,
        labels=labels,
        weights=weights,
        #range=ranges,
        bins=bins1d,
        smooth=smooth,
        smooth1d=smooth1d,
        plot_datapoints=True,
        plot_density=False,
        plot_contours=False,
        fill_contours=False,
        scatter_kwargs=dict(s=point_size, alpha=point_alpha, linewidths=0),
        hist2d_kwargs=h2,
        show_titles=False,
        verbose=False,
    )

    axes = np.array(fig.axes).reshape(K, K)
    for i in range(K):
        xi = data[:, i]; ax = axes[i, i]
        m, lo, hi = _weighted_mode_and_hdi(xi, weights, bins=max(80, bins1d), mass=hdi_mass)
        ax.axvline(m,  ls="-",  lw=1.6)
        ax.axvline(lo, ls="--", lw=1.0)
        ax.axvline(hi, ls="--", lw=1.0)
        ax.set_title(f"{labels[i].strip()}\nMAP={m:.3g}\nHPD[{int(hdi_mass*100)}%]: {lo:.3g}–{hi:.3g}", fontsize=title_fs)


        for j in range(i):
            ax = axes[i, j]
            xi, xj = data[:, j], data[:, i]  # note: corner flips i,j
            H, xed, yed = np.histogram2d(xi, xj, bins=bins2d,
                                        #range=[ranges[j], ranges[i]],
                                        weights=weights, density=True)
            # normalize
            H /= (np.sum(H) * np.diff(xed)[0] * np.diff(yed)[0] + 1e-12)
            H = gaussian_filter(H, sigma=smooth)
            
            peak = np.max(H)
            H = np.where(H > post_crop * peak, H, np.nan)

            X, Y = np.meshgrid(xed, yed)
            ax.pcolormesh(X, Y, H.T, cmap=cmap, shading='auto', alpha=0.8, zorder=0)        





    # --- LITERATURE OVERLAY: ADD FINAL VALUES FROM EXCEL ---
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load your excel
    excel_path = "data/Previous_GCE_results.xlsx"  # change if needed
    df_lit = pd.read_excel(excel_path, sheet_name=0, header=None)

    # Map corner params to excel row names
    param_to_rowname = {
        't_1': 'Timescale1_final',
        't_2': 'Timescale2_final',
        'infall_1': 'Infall1_time_final',
        'infall_2': 'Infall2_time_final',
        'sfe': 'SFE_final',
        'delta_sfe': 'SFE_infall2_final',
        'imf_upper': 'IMF_mmax_final',
        'mgal': 'BulgeMass_final',
        'nb': 'SNIa_perMsun_final'
    }


    study_cols = [str(s).strip() for s in df_lit.iloc[0, 2:8].tolist() if pd.notna(s)]
    colors = plt.cm.tab10.colors[:len(study_cols)]

    # Map param → row index in excel
    row_indices = {}
    for param, rowname in param_to_rowname.items():
        matches = df_lit[df_lit.iloc[:, 0].astype(str).str.contains(rowname, na=False)]
        if not matches.empty:
            row_indices[param] = matches.index[0]

    # Pre-collect values per study
    lit_values = {study: {} for study in study_cols}
    for col_idx, study in zip(range(2, 8), study_cols):
        for param, row_idx in row_indices.items():
            val = df_lit.iloc[row_idx, col_idx]
            try:
                lit_values[study][param] = float(val)
            except (ValueError, TypeError):
                pass  # skip non-numeric

    # Legend handles
    legend_handles = []
    legend_labels = []





    for i, param in enumerate(samples.columns):
        if param not in row_indices:
            continue
        row_idx = row_indices[param]
        ax = axes[i, i]

        for col_idx, study in zip(range(2, 8), study_cols):
            val = df_lit.iloc[row_idx, col_idx]
            try:
                val = float(val)
            except (ValueError, TypeError):
                continue  # skip non-numeric

            # Only plot if in range
            if not (ranges[i][0] <= val <= ranges[i][1]):
                continue

            color = colors[col_idx - 2]
            handle = ax.scatter(val, 0.5* ax.get_ylim()[1], c=[color], s=120, marker='*', edgecolors='k', linewidth=0.5, zorder=10)
            if study not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(study)




    # Loop over lower-triangle cells (i = row, j = col, i > j)
    for i in range(K):  # y-param
        param_y = samples.columns[i]
        if param_y not in row_indices:
            continue
        range_y = ranges[i]

        for j in range(i):  # x-param
            param_x = samples.columns[j]
            if param_x not in row_indices:
                continue
            range_x = ranges[j]
            ax = axes[i, j]

            for idx, study in enumerate(study_cols):
                color = colors[idx]
                vx = lit_values[study].get(param_x)
                vy = lit_values[study].get(param_y)

                plotted = False

                # Case 1: both values → star
                if vx is not None and vy is not None:
                    if range_x[0] <= vx <= range_x[1] and range_y[0] <= vy <= range_y[1]:
                        h = ax.scatter(vx, vy, c=[color], s=140, marker='*', 
                                       edgecolors='black', linewidth=0.8, zorder=10)
                        plotted = True

                # Case 2: only x → vertical line
                elif vx is not None:# and range_x[0] <= vx <= range_x[1]:
                    ax.axvline(vx, color=color, linewidth=2.5, alpha=0.8, zorder=5)
                    plotted = True


                # Case 3: only y → horizontal line
                elif vy is not None:# and range_y[0] <= vy <= range_y[1]:
                    ax.axhline(vy, color=color, linewidth=2.5, alpha=0.8, zorder=5)
                    plotted = True

                # Add to legend (once per study)
                if plotted and study not in legend_labels:
                    # use star as handle
                    h_star = plt.Line2D([], [], color=color, marker='*', markersize=10, 
                                        linestyle='None', markeredgecolor='black', markeredgewidth=0.8)
                    legend_handles.append(h_star)
                    legend_labels.append(study)

    # Legend in top-right blank
    if legend_handles:
        leg_ax = fig.add_axes([0.80, 0.80, 0.16, 0.16])
        leg_ax.set_facecolor('white')
        leg_ax.patch.set_alpha(0.95)
        leg_ax.axis('off')
        leg_ax.legend(handles=legend_handles, labels=legend_labels,
                      loc='center', fontsize=9.5, frameon=True, edgecolor='k')
    # --- END 2D OVERLAY ---









    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



# ---------------- plotting ----------------
def _save_corner_weighted(samples, weights, out_path, *, hdi_mass=0.25, post_crop=0.80,
                          bins1d=80, bins2d=50, smooth=0.8, smooth1d=0.9,
                          cmap="inferno", cmin=1, point_size=4, point_alpha=0.35,
                          title_fs=18, dpi=300):
    data   = samples.to_numpy(float)
    labels = [c.replace("_", " ") + "\n" for c in samples.columns]
    K = data.shape[1]



    ranges = [_weighted_hdi_1d(data[:, i], weights, bins=max(200, 2*bins1d))
              for i in range(K)]

    h2 = dict(cmap=cmap, bins=bins2d)
    if cmin is not None: h2["cmin"] = cmin

    fig = corner.corner(
        data,
        labels=labels,
        weights=weights,
        #range=ranges,
        bins=bins1d,
        smooth=smooth,
        smooth1d=smooth1d,
        plot_datapoints=True,
        plot_density=True,
        plot_contours=False,
        fill_contours=False,
        scatter_kwargs=dict(s=point_size, alpha=point_alpha, linewidths=0),
        hist2d_kwargs=h2,
        show_titles=False,
        verbose=False,
    )

    axes = np.array(fig.axes).reshape(K, K)
    for i in range(K):
        xi = data[:, i]; ax = axes[i, i]
        m, lo, hi = _weighted_mode_and_hdi(xi, weights, bins=max(80, bins1d), mass=hdi_mass)
        ax.axvline(m,  ls="-",  lw=1.6)
        ax.axvline(lo, ls="--", lw=1.0)
        ax.axvline(hi, ls="--", lw=1.0)
        ax.set_title(f"{labels[i].strip()}\nMAP={m:.3g}\nHPD[{int(hdi_mass*100)}%]: {lo:.3g}–{hi:.3g}", fontsize=title_fs)





    # --- LITERATURE OVERLAY: ADD FINAL VALUES FROM EXCEL ---
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load your excel
    excel_path = "data/Previous_GCE_results.xlsx"  # change if needed
    df_lit = pd.read_excel(excel_path, sheet_name=0, header=None)

    # Map corner params to excel row names
    param_to_rowname = {
        't_1': 'Timescale1_final',
        't_2': 'Timescale2_final',
        'infall_1': 'Infall1_time_final',
        'infall_2': 'Infall2_time_final',
        'sfe': 'SFE_final',
        'delta_sfe': 'SFE_infall2_final',
        'imf_upper': 'IMF_mmax_final',
        'mgal': 'BulgeMass_final',
        'nb': 'SNIa_perMsun_final'
    }


    study_cols = [str(s).strip() for s in df_lit.iloc[0, 2:8].tolist() if pd.notna(s)]
    colors = plt.cm.tab10.colors[:len(study_cols)]

    # Map param → row index in excel
    row_indices = {}
    for param, rowname in param_to_rowname.items():
        matches = df_lit[df_lit.iloc[:, 0].astype(str).str.contains(rowname, na=False)]
        if not matches.empty:
            row_indices[param] = matches.index[0]

    # Pre-collect values per study
    lit_values = {study: {} for study in study_cols}
    for col_idx, study in zip(range(2, 8), study_cols):
        for param, row_idx in row_indices.items():
            val = df_lit.iloc[row_idx, col_idx]
            try:
                lit_values[study][param] = float(val)
            except (ValueError, TypeError):
                pass  # skip non-numeric

    # Legend handles
    legend_handles = []
    legend_labels = []





    for i, param in enumerate(samples.columns):
        if param not in row_indices:
            continue
        row_idx = row_indices[param]
        ax = axes[i, i]

        for col_idx, study in zip(range(2, 8), study_cols):
            val = df_lit.iloc[row_idx, col_idx]
            try:
                val = float(val)
            except (ValueError, TypeError):
                continue  # skip non-numeric

            # Only plot if in range
            if not (ranges[i][0] <= val <= ranges[i][1]):
                continue

            color = colors[col_idx - 2]
            handle = ax.scatter(val, 0.5* ax.get_ylim()[1], c=[color], s=120, marker='*', edgecolors='k', linewidth=0.5, zorder=10)
            if study not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(study)




    # Loop over lower-triangle cells (i = row, j = col, i > j)
    for i in range(K):  # y-param
        param_y = samples.columns[i]
        if param_y not in row_indices:
            continue
        range_y = ranges[i]

        for j in range(i):  # x-param
            param_x = samples.columns[j]
            if param_x not in row_indices:
                continue
            range_x = ranges[j]
            ax = axes[i, j]

            for idx, study in enumerate(study_cols):
                color = colors[idx]
                vx = lit_values[study].get(param_x)
                vy = lit_values[study].get(param_y)

                plotted = False

                # Case 1: both values → star
                if vx is not None and vy is not None:
                    if range_x[0] <= vx <= range_x[1] and range_y[0] <= vy <= range_y[1]:
                        h = ax.scatter(vx, vy, c=[color], s=140, marker='*', 
                                       edgecolors='black', linewidth=0.8, zorder=10)
                        plotted = True

                # Case 2: only x → vertical line
                elif vx is not None:# and range_x[0] <= vx <= range_x[1]:
                    ax.axvline(vx, color=color, linewidth=2.5, alpha=0.8, zorder=5)
                    plotted = True


                # Case 3: only y → horizontal line
                elif vy is not None:# and range_y[0] <= vy <= range_y[1]:
                    ax.axhline(vy, color=color, linewidth=2.5, alpha=0.8, zorder=5)
                    plotted = True

                # Add to legend (once per study)
                if plotted and study not in legend_labels:
                    # use star as handle
                    h_star = plt.Line2D([], [], color=color, marker='*', markersize=10, 
                                        linestyle='None', markeredgecolor='black', markeredgewidth=0.8)
                    legend_handles.append(h_star)
                    legend_labels.append(study)

    # Legend in top-right blank
    if legend_handles:
        leg_ax = fig.add_axes([0.80, 0.80, 0.16, 0.16])
        leg_ax.set_facecolor('white')
        leg_ax.patch.set_alpha(0.95)
        leg_ax.axis('off')
        leg_ax.legend(handles=legend_handles, labels=legend_labels,
                      loc='center', fontsize=9.5, frameon=True, edgecolor='k')
    # --- END 2D OVERLAY ---



    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

