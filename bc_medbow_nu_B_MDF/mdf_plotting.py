
# Clean plotting façade for the MDF_GCE_SMC_DEMC project.
import matplotlib
matplotlib.use("Agg")

from plotting import (
    use_paper_style,
    plot_mdf_family, plot_mdf_single,
    plot_alpha_histograms,
    plot_sfr_history, plot_mass_evolution,
    plot_amr,
    corner_from_df, plot_loss_vs_gen, plot_walker_paths,
)

__all__ = [
    "use_paper_style",
    "plot_mdf_family", "plot_mdf_single",
    "plot_alpha_histograms",
    "plot_sfr_history", "plot_mass_evolution",
    "plot_amr",
    "corner_from_df", "plot_loss_vs_gen", "plot_walker_paths",
    "generate_all_plots",
]

import os, glob, re
import pandas as pd

import omni_plot as omni
import ga_unified_plots as gup
import loss_plot as lplot
import analysis_plot as anplot
import phys_plot as pplot
import age_meta as amplot


def _infer_generation(results_csv_path):
    m = re.search(r"simulation_results_gen_(\d+)\.csv$", str(results_csv_path))
    return int(m.group(1)) if m else None

def _scan_loss_curve(output_path, loss_col):
    files = sorted(glob.glob(os.path.join(output_path, "simulation_results_gen_*.csv")))
    generations = []
    best_loss = []
    for f in files:
        m = re.search(r"(\d+)", os.path.basename(f))
        if not m:
            continue
        gen = int(m.group(1))
        df = pd.read_csv(f)
        generations.append(gen)
        best_loss.append(df[loss_col].min())
    return generations, best_loss


def generate_all_plots(GalGA, feh, normalized_count, results_csv_path):
    """
    Plot everything for this generation. No guards, no try/except.
    If inputs are missing, let it fail loudly.
    """
    use_paper_style()
    out = GalGA.output_path

    results_df = pd.read_csv(results_csv_path)
    gen = _infer_generation(results_csv_path)

    # --- Core MDF family ---
    fname_mdf = f"MDF_family_gen_{gen}.png" if gen is not None else "MDF_family.png"
    plot_mdf_family(GalGA, feh, normalized_count, results_df=results_df,
                    save_path=os.path.join(out, fname_mdf))

    # --- Loss vs generation (min loss per generation) ---
    generations, best_loss = _scan_loss_curve(out, GalGA.loss_metric)
    plot_loss_vs_gen(generations, best_loss,
                     save_path=os.path.join(out, "loss_vs_generation.png"),
                     ylabel=f"Min {GalGA.loss_metric}")

    # --- Corner (key parameters) ---
    corner_cols = ["sigma_2", "t_2", "infall_2"]
    corner_from_df(results_df, corner_cols, out_png=os.path.join(out, f"corner_gen_{gen or 'latest'}.png"))

    # --- Physics diagnostics ---
    pplot.plot_real_infall_physics(GalGA, results_df, save_path=os.path.join(out, "infall_physics.png"))
    pplot.plot_omega_diagnostics(GalGA, results_df, save_path=os.path.join(out, "omega_diagnostics.png"))

    # --- AMR ---
    amplot.plot_age_metallicity_curves(GalGA, feh, GalGA.age_Joyce, GalGA.age_Bensby,
                                       results_df, os.path.join(out, f"AMR_curves_gen_{gen or 'latest'}.png"))
    amplot.plot_age_feh_detailed(GalGA, feh, GalGA.age_Joyce, GalGA.age_Bensby,
                                 results_df, os.path.join(out, f"AMR_detailed_gen_{gen or 'latest'}.png"))

    # --- α-element summary ---
    plot_alpha_histograms(GalGA.alpha_obs, GalGA.alpha_models,
                          save_path=os.path.join(out, f"alpha_histograms_gen_{gen or 'latest'}.png"))

    # --- Omni figure ---
    omni.plot_omni_figure_ultimate(GalGA, feh, GalGA.age_Joyce, GalGA.age_Bensby,
                                   feh, normalized_count, results_df,
                                   os.path.join(out, f"omni_ultimate_gen_{gen or 'latest'}.png"))

    # --- Loss diagnostics ---
    lplot.plot_walker_loss_history(GalGA, GalGA.walker_history, results_csv_path, GalGA.loss_metric)
    lplot.plot_multiple_loss_metrics_evolution(GalGA, GalGA.walker_history, results_csv_path,
                                               [GalGA.loss_metric], os.path.join(out, "loss_metrics_evolution.png"))
    lplot.plot_loss_convergence_analysis(GalGA, GalGA.walker_history, results_csv_path,
                                         GalGA.loss_metric, os.path.join(out, "loss_convergence.png"))

    # --- PCA / correlations ---
    anplot.plot_pca_degeneracy_analysis(GalGA, results_csv_path, os.path.join(out, "pca_degeneracy.png"))
    anplot.plot_parameter_correlation_matrix(GalGA, results_csv_path, os.path.join(out, "param_correlations.png"))
