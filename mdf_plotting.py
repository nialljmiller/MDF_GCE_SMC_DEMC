# mdf_plotting.py — unified plotting, clean & explicit, no error handling
# Assumes all required inputs/files exist and are valid.
# Organization:
#   0) Imports & style
#   1) Small utilities
#   2) Core plotters (MDF, AMR/age–[Fe/H], 4× alpha, alpha histos, SFR/mass)
#   3) Orchestrator: generate_all_plots(...)

from __future__ import annotations

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

# External plot modules you already split out
from plotting.loss_plot import *              # loss & walker plots, 2D/3D scatter, etc.
from plotting.plot_amr import *
from plotting.omni_plot import *              # omni info figure (if you use it)
from plotting.core_plots import *
from plotting.phys_plot import *


from posterior_plotting_package.core_plots_posterior_v2 import plot_age_feh_detailed, plot_mdf_curves, plot_four_panel_alpha
from posterior_plotting_package.phys_plot_posterior_v2 import plot_real_infall_physics


from plotting.style import *
use_paper_style()

# =========================
# 1) UTILITIES
# =========================

def ensure_dirs(output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "analysis"), exist_ok=True)

def extract_metrics(results_file: str):
    """
    Returns:
      sigma_2_vals, t_1_vals, t_2_vals, infall_1_vals, infall_2_vals,
      sfe_vals, delta_sfe_vals, imf_upper_vals, mgal_vals, nb_vals,
      metrics_dict, df
    """
    df = pd.read_csv(results_file)
    cols = df.columns

    # Required columns (names taken from your original codebase)
    sigma_2_vals     = df["sigma_2"].values
    t_1_vals         = df["t_1"].values
    t_2_vals         = df["t_2"].values
    infall_1_vals    = df["infall_1"].values
    infall_2_vals    = df["infall_2"].values
    sfe_vals         = df["sfe"].values
    delta_sfe_vals   = df["delta_sfe"].values
    imf_upper_vals   = df["imf_upper"].values
    mgal_vals        = df["m_gal"].values if "m_gal" in cols else df["mgal"].values
    nb_vals          = df["n_bulge"].values if "n_bulge" in cols else df["nb"].values

    # Collect any loss / metric-looking columns
    metric_cols = [c for c in cols if c.lower() not in {
        "sigma_2","t_1","t_2","infall_1","infall_2","sfe","delta_sfe",
        "imf_upper","m_gal","mgal","n_bulge","nb"
    }]
    metrics_dict = {c: df[c].values for c in metric_cols}

    return (sigma_2_vals, t_1_vals, t_2_vals, infall_1_vals, infall_2_vals,
            sfe_vals, delta_sfe_vals, imf_upper_vals, mgal_vals, nb_vals,
            metrics_dict, df)







def generate_all_plots(GalGA, feh, normalized_count, results_file=None):
    """Generate the MDF, AMR, alpha fits, and the posterior corner plot."""

    # ----------------------------
    # Resolve inputs / paths
    # ----------------------------
    if results_file is None:
        results_file = os.path.join(GalGA.output_path, "simulation_results.csv")

    # Load observational alpha/age data (prefer local 'data/', else '../data/')
    try:
        f = open("data/Bensby_Data.tsv")
    except FileNotFoundError:
        f = open("../data/Bensby_Data.tsv")

    from plotting.posterior_analysis import run_posterior_report  # local import to avoid hard dependency

    # ----------------------------
    # Parse observational table
    # ----------------------------
    lines = f.readlines()
    header = lines[0].split()

    # Column indices (computed once)
    Fe_H_ind       = header.index('[Fe/H]')
    Si_Fe_ind      = header.index('[Si/Fe]')
    Ca_Fe_ind      = header.index('[Ca/Fe]')
    Mg_Fe_ind      = header.index('[Mg/Fe]')
    Ti_Fe_ind      = header.index('[Ti/Fe]')
    age_Joyce_ind  = header.index('Joyce_age')
    age_Bensby_ind = header.index('Bensby')

    Fe_H, age_Joyce, age_Bensby = [], [], []
    Si_Fe, Ca_Fe, Mg_Fe, Ti_Fe  = [], [], [], []

    for line in lines[1:]:
        toks = line.split()
        age_Joyce.append(float(toks[age_Joyce_ind]))
        age_Bensby.append(float(toks[age_Bensby_ind]))
        Fe_H.append(float(toks[Fe_H_ind]))
        Si_Fe.append(float(toks[Si_Fe_ind]))
        Ca_Fe.append(float(toks[Ca_Fe_ind]))
        Mg_Fe.append(float(toks[Mg_Fe_ind]))
        Ti_Fe.append(float(toks[Ti_Fe_ind]))

    f.close()

    # Numpy arrays for element series (keep ages as lists, matching original usage)
    Fe_H  = np.array(Fe_H)
    Si_Fe = np.array(Si_Fe)
    Ca_Fe = np.array(Ca_Fe)
    Mg_Fe = np.array(Mg_Fe)
    Ti_Fe = np.array(Ti_Fe)

    # ----------------------------
    # Ensure output folders exist
    # ----------------------------
    ensure_dirs(GalGA.output_path)

    # Results dataframe (retain existing error handling/prints)
    try:
        sigma_2_vals, t_1_vals, t_2_vals, infall_1_vals, infall_2_vals, sfe_vals, delta_sfe_vals, imf_upper_vals, mgal_vals, nb_vals, metrics_dict, df = extract_metrics(results_file)
        #df = pd.read_csv(results_file)
    except FileNotFoundError:
        print(f"Results file {results_file} not found; continuing without a dataframe.")
        df = pd.DataFrame()
    except Exception as exc:
        print(f"Unable to load {results_file}: {exc}")
        df = pd.DataFrame()




    ####################
    ####################
    print("data loaded")
    ####################
    ####################



    # Core plots (fast)
    plot_age_feh_detailed(GalGA, Fe_H, age_Joyce, age_Bensby, results_df=df, use_posterior=True, percentile=-1)

    plot_mdf_curves(GalGA, feh, normalized_count, results_df=df, use_posterior=True, percentile=-1)

    plot_four_panel_alpha(GalGA, Fe_H, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe, results_df=df, use_posterior=True, percentile=-1)

    plot_real_infall_physics(GalGA, results_df=df, use_posterior=True, max_models=20, percentile=-1)



    # ----------------------------
    # Core plots
    # ----------------------------
    print("Generating MDF fit plot...")
    plot_mdf_curves(GalGA, feh, normalized_count, df if not df.empty else None)

    print("Generating four-panel alpha comparison...")
    plot_four_panel_alpha(GalGA, Fe_H, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe, df if not df.empty else None)

    print("Generating age-metallicity relation plots...")
    plot_age_feh_detailed(GalGA, Fe_H, age_Joyce, age_Bensby, results_df=df if not df.empty else None, n_bins=10)

    plot_age_metallicity_curves(GalGA, Fe_H, age_Joyce, age_Bensby, df if not df.empty else None)

    plt.close('all')


    generate_physics_plots(GalGA, results_file=results_file)



    # ----------------------------
    # Posterior analysis
    # ----------------------------
    posterior_args = argparse.Namespace(
        results=os.path.abspath(results_file),
        history=None,
        pcard="bulge_pcard.txt",
        output=None,
        params=None,
        nsamples=5000,
        temperature=None,
        seed=42,
    )
    summary = run_posterior_report(posterior_args)
    posterior_dir = summary.get("output_dir",
                                os.path.join(os.path.dirname(results_file),
                                             "analysis", "posterior"))
    ess = summary.get("effective_sample_size")
    ess_text = f"{ess:.1f}" if isinstance(ess, (int, float)) else "n/a"
    print(f"Posterior analysis complete. Outputs written to {posterior_dir}")
    print(f"Posterior draws: {summary.get('posterior_draws')} (ESS={ess_text})")

    print("All plotting complete! Generated MDF, AMR, alpha, and posterior diagnostics.")
    print(f"Loaded {len(Fe_H)} observational data points for individual alpha elements")



    # ----------------------------
    # Binned loss / marginals / gradients
    # ----------------------------
    analysis_dir = os.path.join(GalGA.output_path, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    key_pairs = [
        ('t_2', 'infall_2'),
        ('sigma_2', 't_2'),
        ('sigma_2', 'infall_2'),
    ]

    # 1D marginals
    for p in {'t_2', 'infall_2', 'sigma_2'}:
        if p in df.columns and 'fitness' in df.columns:
            try:
                plot_marginal_loss(
                    df, p, losscol='fitness', bins=50, agg='median',
                    save_path=os.path.join(analysis_dir, f'marginal_{p}.png')
                )
            except Exception as e:
                print(f"[marginal {p}] skipped: {e}")

    # 2D binned surfaces + Δ-loss + gradient fields
    for xcol, ycol in key_pairs:
        if all(c in df.columns for c in [xcol, ycol, 'fitness']):
            try:
                out_base = os.path.join(analysis_dir, f"binned_fitness_{xcol}_{ycol}")
                Z, xedges, yedges, N = plot_binned_loss(
                    GalGA, df, xcol=xcol, ycol=ycol, losscol='fitness',
                    bins=(50, 50), agg='median', min_per_bin=1, smooth_sigma=1.0,
                    cmap='rainbow', save_path=out_base + ".png"
                )
                plot_delta_and_gradient(
                    xcol, ycol, Z, xedges, yedges,
                    save_prefix=out_base, quiver_step=3
                )
            except Exception as e:
                print(f"[binned {xcol} vs {ycol}] skipped: {e}")
        else:
            missing = [c for c in [xcol, ycol, 'fitness'] if c not in df.columns]
            print(f"[binned {xcol} vs {ycol}] missing columns: {missing}")

    # ----------------------------
    # Parameter-space exploration (2D/3D)
    # (kept as-is, including external variables referenced)
    # ----------------------------
    # ========== INFALL PARAMETERS ==========
    metric_name = 'fitness'
    metric_vals = metrics_dict[metric_name]

    plot_2d_scatter(GalGA, t_2_vals, infall_2_vals, metric_vals, metric_name + '_t2_infall2',
                    xlabel='t_2 (Gyr)', ylabel='infall_2 (Gyr)')
    plot_2d_scatter(GalGA, sigma_2_vals, infall_2_vals, metric_vals, metric_name + '_sigma2_infall2',
                    xlabel='sigma_2', ylabel='infall_2 (Gyr)')
    plot_2d_scatter(GalGA, sigma_2_vals, t_2_vals, metric_vals, metric_name + '_sigma2_t2',
                    xlabel='sigma_2', ylabel='t_2 (Gyr)')

    # First infall episode
    plot_2d_scatter(GalGA, t_1_vals, infall_1_vals, metric_vals, metric_name + '_t1_infall1',
                    xlabel='t_1 (Gyr)', ylabel='infall_1 (Gyr)')
    plot_2d_scatter(GalGA, t_1_vals, infall_2_vals, metric_vals, metric_name + '_t1_infall2',
                    xlabel='t_1 (Gyr)', ylabel='infall_2 (Gyr)')

    # Cross-infall comparisons
    plot_2d_scatter(GalGA, t_1_vals, t_2_vals, metric_vals, metric_name + '_t1_t2',
                    xlabel='t_1 (Gyr)', ylabel='t_2 (Gyr)')
    plot_2d_scatter(GalGA, infall_1_vals, infall_2_vals, metric_vals, metric_name + '_infall1_infall2',
                    xlabel='infall_1 (Gyr)', ylabel='infall_2 (Gyr)')

    # ========== STAR FORMATION EFFICIENCY ==========
    plot_2d_scatter(GalGA, sfe_vals, delta_sfe_vals, metric_vals, metric_name + '_sfe_deltasfe',
                    xlabel='SFE', ylabel='Delta SFE')
    plot_2d_scatter(GalGA, sfe_vals, t_2_vals, metric_vals, metric_name + '_sfe_t2',
                    xlabel='SFE', ylabel='t_2 (Gyr)')
    plot_2d_scatter(GalGA, sfe_vals, sigma_2_vals, metric_vals, metric_name + '_sfe_sigma2',
                    xlabel='SFE', ylabel='sigma_2')
    plot_2d_scatter(GalGA, delta_sfe_vals, t_2_vals, metric_vals, metric_name + '_deltasfe_t2',
                    xlabel='Delta SFE', ylabel='t_2 (Gyr)')
    plot_2d_scatter(GalGA, delta_sfe_vals, infall_2_vals, metric_vals, metric_name + '_deltasfe_infall2',
                    xlabel='Delta SFE', ylabel='infall_2 (Gyr)')

    # ========== GALAXY MASS RELATIONS ==========
    plot_2d_scatter(GalGA, mgal_vals, sfe_vals, metric_vals, metric_name + '_mgal_sfe',
                    xlabel='M_gal (M_sun)', ylabel='SFE')
    plot_2d_scatter(GalGA, mgal_vals, sigma_2_vals, metric_vals, metric_name + '_mgal_sigma2',
                    xlabel='M_gal (M_sun)', ylabel='sigma_2')
    plot_2d_scatter(GalGA, mgal_vals, t_2_vals, metric_vals, metric_name + '_mgal_t2',
                    xlabel='M_gal (M_sun)', ylabel='t_2 (Gyr)')
    plot_2d_scatter(GalGA, mgal_vals, infall_2_vals, metric_vals, metric_name + '_mgal_infall2',
                    xlabel='M_gal (M_sun)', ylabel='infall_2 (Gyr)')

    # ========== IMF AND STELLAR PARAMETERS ==========
    plot_2d_scatter(GalGA, imf_upper_vals, sfe_vals, metric_vals, metric_name + '_imf_sfe',
                    xlabel='IMF Upper (M_sun)', ylabel='SFE')
    plot_2d_scatter(GalGA, imf_upper_vals, t_2_vals, metric_vals, metric_name + '_imf_t2',
                    xlabel='IMF Upper (M_sun)', ylabel='t_2 (Gyr)')
    plot_2d_scatter(GalGA, imf_upper_vals, mgal_vals, metric_vals, metric_name + '_imf_mgal',
                    xlabel='IMF Upper (M_sun)', ylabel='M_gal (M_sun)')
    plot_2d_scatter(GalGA, nb_vals, imf_upper_vals, metric_vals, metric_name + '_nb_imf',
                    xlabel='SN1a per Solar Mass', ylabel='IMF Upper (M_sun)')

    # ========== SN1A PARAMETERS ==========
    plot_2d_scatter(GalGA, nb_vals, sfe_vals, metric_vals, metric_name + '_nb_sfe',
                    xlabel='SN1a per Solar Mass', ylabel='SFE')
    plot_2d_scatter(GalGA, nb_vals, t_2_vals, metric_vals, metric_name + '_nb_t2',
                    xlabel='SN1a per Solar Mass', ylabel='t_2 (Gyr)')
    plot_2d_scatter(GalGA, nb_vals, mgal_vals, metric_vals, metric_name + '_nb_mgal',
                    xlabel='SN1a per Solar Mass', ylabel='M_gal (M_sun)')
    plot_2d_scatter(GalGA, nb_vals, sigma_2_vals, metric_vals, metric_name + '_nb_sigma2',
                    xlabel='SN1a per Solar Mass', ylabel='sigma_2')

    # ========== INFALL-FOCUSED 3D PLOTS ==========
    plot_3d_scatter(GalGA, sigma_2_vals, t_2_vals, infall_2_vals, metric_vals,
                    metric_name + '_infall2_complete',
                    xlabel='sigma_2', ylabel='t_2 (Gyr)', zlabel='infall_2 (Gyr)')
    plot_3d_scatter(GalGA, t_1_vals, t_2_vals, infall_2_vals, metric_vals,
                    metric_name + '_timing_comparison',
                    xlabel='t_1 (Gyr)', ylabel='t_2 (Gyr)', zlabel='infall_2 (Gyr)')
    plot_3d_scatter(GalGA, infall_1_vals, infall_2_vals, sigma_2_vals, metric_vals,
                    metric_name + '_infall_timescales',
                    xlabel='infall_1 (Gyr)', ylabel='infall_2 (Gyr)', zlabel='sigma_2')

    # ========== SFE-FOCUSED 3D PLOTS ==========
    plot_3d_scatter(GalGA, sfe_vals, delta_sfe_vals, infall_2_vals, metric_vals,
                    metric_name + '_sfe_evolution',
                    xlabel='SFE', ylabel='Delta SFE', zlabel='infall_2 (Gyr)')
    plot_3d_scatter(GalGA, sfe_vals, t_1_vals, infall_2_vals, metric_vals,
                    metric_name + '_sfe_timing',
                    xlabel='SFE', ylabel='t_1 (Gyr)', zlabel='infall_2 (Gyr)')
    plot_3d_scatter(GalGA, sfe_vals, t_2_vals, sigma_2_vals, metric_vals,
                    metric_name + '_sfe_infall2_params',
                    xlabel='SFE', ylabel='t_2 (Gyr)', zlabel='sigma_2')
    plot_3d_scatter(GalGA, delta_sfe_vals, t_2_vals, infall_2_vals, metric_vals,
                    metric_name + '_deltasfe_timing',
                    xlabel='Delta SFE', ylabel='t_2 (Gyr)', zlabel='infall_2 (Gyr)')

    # ========== GALAXY MASS-FOCUSED 3D PLOTS ==========
    plot_3d_scatter(GalGA, mgal_vals, sfe_vals, infall_2_vals, metric_vals,
                    metric_name + '_mgal_sfe_infall',
                    xlabel='M_gal (M_sun)', ylabel='SFE', zlabel='infall_2 (Gyr)')
    plot_3d_scatter(GalGA, mgal_vals, t_2_vals, sigma_2_vals, metric_vals,
                    metric_name + '_mgal_infall2_params',
                    xlabel='M_gal (M_sun)', ylabel='t_2 (Gyr)', zlabel='sigma_2')
    plot_3d_scatter(GalGA, mgal_vals, sfe_vals, delta_sfe_vals, metric_vals,
                    metric_name + '_mgal_sfe_evolution',
                    xlabel='M_gal (M_sun)', ylabel='SFE', zlabel='Delta SFE')

    # ========== STELLAR/IMF-FOCUSED 3D PLOTS ==========
    plot_3d_scatter(GalGA, imf_upper_vals, sfe_vals, infall_2_vals, metric_vals,
                    metric_name + '_imf_sfe_infall',
                    xlabel='IMF Upper (M_sun)', ylabel='SFE', zlabel='infall_2 (Gyr)')
    plot_3d_scatter(GalGA, nb_vals, imf_upper_vals, infall_2_vals, metric_vals,
                    metric_name + '_stellar_params_infall',
                    xlabel='SN1a per Solar Mass', ylabel='IMF Upper (M_sun)', zlabel='infall_2 (Gyr)')
    plot_3d_scatter(GalGA, nb_vals, sfe_vals, t_2_vals, metric_vals,
                    metric_name + '_sn1a_sfe_timing',
                    xlabel='SN1a per Solar Mass', ylabel='SFE', zlabel='t_2 (Gyr)')

    # ========== CROSS-PARAMETER EXPLORATION ==========
    plot_3d_scatter(GalGA, sigma_2_vals, sfe_vals, mgal_vals, metric_vals,
                    metric_name + '_sigma_sfe_mgal',
                    xlabel='sigma_2', ylabel='SFE', zlabel='M_gal (M_sun)')
    plot_3d_scatter(GalGA, t_1_vals, sfe_vals, delta_sfe_vals, metric_vals,
                    metric_name + '_t1_sfe_evolution',
                    xlabel='t_1 (Gyr)', ylabel='SFE', zlabel='Delta SFE')
    plot_3d_scatter(GalGA, infall_1_vals, infall_2_vals, sfe_vals, metric_vals,
                    metric_name + '_infall_timescales_sfe',
                    xlabel='infall_1 (Gyr)', ylabel='infall_2 (Gyr)', zlabel='SFE')

    # ----------------------------
    # Walker evolution / loss history
    # ----------------------------
    try:
        print("Generating walker evolution plots...")
        param_names   = ["sigma_2", "t_2", "infall_2", "sfe", "delta_sfe"]
        param_indices = [5, 7, 9, 10, 11]
        plot_walker_history(GalGA, GalGA.walker_history, param_names, param_indices)

        print("Generating walker loss history plots...")
        for metric in ['ks', 'huber', 'cosine', 'log_cosh', 'fitness', 'age_meta_fitness', 'physics_penalty']:
            plot_walker_loss_history(GalGA, GalGA.walker_history, results_file, loss_metric=metric)
            plot_multiple_success_thresholds(GalGA, GalGA.walker_history,
                                             results_csv=results_file,
                                             thresholds=[0.01, 0.1, 0.001],
                                             loss_metric=metric)
    except:
        pass

    # ----------------------------
    # Omni figures
    # ----------------------------
    print("Generating dashboard figure...")
    plot_omni_info_figure(
        GalGA, Fe_H, age_Joyce, age_Bensby, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe,
        feh, normalized_count, df
    )
    plot_omni_figure(
        GalGA, Fe_H, age_Joyce, age_Bensby, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe,
        feh, normalized_count, df
    )

    #plot_omni_figure_enhanced(GalGA, Fe_H, age_Joyce, age_Bensby, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe,feh, normalized_count, df)

    #plot_omni_figure_ultimate(GalGA, Fe_H, age_Joyce, age_Bensby, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe,feh, normalized_count, df)

    print("Omni info figure generated!")


    # ----------------------------
    # Wrap-up
    # ----------------------------
    plt.close('all')  # (optional) belt-and-suspenders at the end of an iteration
    print("All plotting complete! Check the output directory for results.")
    print("Generated parameter space exploration plots:")
    # print(f"- {len(metrics_dict)} metrics × 24 2D plots = {len(metrics_dict) * 24} 2D scatter plots")
    # print(f"- {len(metrics_dict)} metrics × 16 3D plots = {len(metrics_dict) * 16} 3D scatter plots")
    print("- Plus walker evolution, loss history, PCA analysis, and correlation matrix plots")
    print(f"Loaded {len(Fe_H)} observational data points for individual alpha elements")
