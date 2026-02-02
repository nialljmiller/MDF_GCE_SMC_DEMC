import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

from .best_selection import *
from .ensembles import (
    build_mdf_ensemble,
    build_age_feh_ensemble,
    build_alpha_ensemble,
    _extract_mdf_xy,
    _weighted_quantile)
from .obs import load_observed_amr, load_observed_alpha, load_observed_mdf

import ast
import re

_PCARD_CAT_KEYS = {
    "comp_array":            "comp_idx",
    "imf_array":             "imf_idx",
    "sn1a_assumptions":      "sn1a_idx",
    "stellar_yield_assumptions": "sy_idx",
    "sn1a_rates":            "sn1ar_idx",
}
_PCARD_HEADER_KEYS = ["sn1a_header", "iniab_header"]  # optional string prefixes

def _parse_pcard_categories(pcard_path: str) -> tuple[dict[str, list], dict[str, str]]:
    """
    Read bulge_pcard.txt and return:
      (categories, headers)
    categories: { 'comp_array': [...], 'imf_array': [...], ... }
    headers:    { 'sn1a_header': '...', 'iniab_header': '...' }  (optional)
    """
    txt = open(pcard_path, "r", encoding="utf-8").read()
    cats: dict[str, list] = {}
    heads: dict[str, str] = {}

    # parse arrays like: key: [ ... ]
    for key in _PCARD_CAT_KEYS.keys():
        m = re.search(rf'^\s*{re.escape(key)}\s*:\s*(\[.*?\])\s*$', txt, flags=re.MULTILINE|re.DOTALL)
        if m:
            try:
                cats[key] = list(ast.literal_eval(m.group(1)))
            except Exception:
                pass

    # parse simple string headers if present, e.g. sn1a_header: "path/prefix_"
    for key in _PCARD_HEADER_KEYS:
        m = re.search(rf'^\s*{re.escape(key)}\s*:\s*(.+)$', txt, flags=re.MULTILINE)
        if m:
            val = m.group(1).strip().strip('\'"')
            heads[key] = val
    return cats, heads

def _resolve_categorical_labels(best_row: pd.Series,
                                cats: dict[str, list],
                                heads: dict[str, str]) -> list[tuple[str, str]]:
    """
    Use best_row indices to pick names from cats. Returns list of (label, value_text).
    If headers (e.g., iniab/sn1a header) exist, they’re prepended for clarity.
    """
    out = []
    for arr_key, idx_col in _PCARD_CAT_KEYS.items():
        if idx_col not in best_row or arr_key not in cats:
            continue
        idx = int(best_row[idx_col])
        arr = cats[arr_key]
        if 0 <= idx < len(arr):
            name = str(arr[idx])
        else:
            name = f"idx={idx} (out of range)"
        # decorate a couple that commonly use a header prefix
        if arr_key == "comp_array" and "iniab_header" in heads:
            name = f"{heads['iniab_header']}{name}"
        if arr_key in ("sn1a_assumptions",) and "sn1a_header" in heads:
            name = f"{heads['sn1a_header']}{name}"
        # pretty label
        pretty = {
            "comp_array": "composition",
            "imf_array": "IMF",
            "sn1a_assumptions": "SNIa table",
            "stellar_yield_assumptions": "yields",
            "sn1a_rates": "SNIa rate",
        }[arr_key]
        out.append((pretty, name))
    return out

# helper anywhere above plot_omni_posterior
def _summarize_params(df, w, keys):
    out = []
    for k in keys:
        if k in df.columns:
            v = np.asarray(df[k], float)
            q16, q50, q84 = _weighted_quantile(v, w, [0.16, 0.5, 0.84])
            out.append((k, q50, q50 - q16, q84 - q50))
    return out



def plot_omni(df: pd.DataFrame, output_dir: str, obs_mdf_path: str | None = None,top_overlay=99999999999999, loss_metric='loss'):
    """
    Combined 'Omni' dashboard plot showing MDF, AMR, and Alpha-element diagnostics
    together with best-fit parameters and fit metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Full combined results table (with posterior weights, model data, etc.)
    output_dir : str
        Output directory for the figure.
    obs_mdf_path : str, optional
        Path to observed MDF file, passed to load_observed_mdf().
    """

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Best-fit model row
    # -------------------------------------------------------------------------
    best_idx = stable_best_index(df, primary=loss_metric)
    best_row = df.loc[best_idx]


    # --- locate and parse the pcard used for this run ---
    pcard_hints = [
        os.getcwd(),
        output_dir,
        os.path.dirname(output_dir),
    ]

    # if run_name encodes a folder, try that too
    pcard_path = output_dir.split('/')[0] + '/' + 'bulge_pcard.txt'
    cats, heads = _parse_pcard_categories(pcard_path)

    N = top_overlay

    # -------------------------------------------------------------------------
    # Observational data
    # -------------------------------------------------------------------------
    Fe_H, age_Joyce, age_Bensby = load_observed_amr()
    Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe = load_observed_alpha()
    x_obs, y_obs = (np.array([]), np.array([]))
    x_obs_mdf, y_obs_mdf = load_observed_mdf(obs_mdf_path)  # normalized + sorted

    # -------------------------------------------------------------------------
    # Figure layout
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.35,
                  left=0.05, right=0.97, top=0.95, bottom=0.06)



    # -------------------------------------------------------------------------
    # PANEL 1: Parameter summary (top-left)
    # -------------------------------------------------------------------------

    # ================================
    # PANEL 1: Parameters (2 columns)
    # ================================
    ax_params = fig.add_subplot(gs[0, :2])
    ax_params.axis("off")

    num_keys = ["sigma_2","t_1","t_2","infall_1","infall_2",
                "sfe","delta_sfe","imf_upper","mgal","nb"]
    pairs = [(k, f"{best_row[k]:.4g}") for k in num_keys]
    mid = math.ceil(len(pairs)/2)
    left, right = pairs[:mid], pairs[mid:]

    lines = ["BEST-FIT MODEL PARAMETERS", "="*28]
    for i in range(mid):
        k1,v1 = left[i]
        if i < len(right):
            k2,v2 = right[i]
            lines.append(f"{k1:<10}: {v1:>10}    {k2:<10}: {v2:>10}")
        else:
            lines.append(f"{k1:<10}: {v1:>10}")

    # categorical choices (one per line; show basename only)
    lines.append("")
    lines.append("CATEGORICAL CHOICES")
    lines.append("-"*21)
    for label, value in _resolve_categorical_labels(best_row, cats, heads):
        val = os.path.basename(str(value))
        lines.append(f"{label:<12}: {val}")

    ax_params.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax_params.transAxes, va="top",
        fontfamily="monospace", fontsize=11, linespacing=1.05,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9)
    )

    # ================================
    # PANEL 2: Fit metrics (simple list)
    # ================================
    ax_metrics = fig.add_subplot(gs[0, 2:])
    ax_metrics.axis("off")

    metric_keys = ["fitness","wrmse","mae","huber","cosine","ks","ensemble"]
    mlines = ["FIT QUALITY METRICS", "="*20]
    for k in metric_keys:
        if k in best_row:
            mlines.append(f"{k:<10}: {best_row[k]:.5f}")

    ax_metrics.text(
        0.02, 0.98, "\n".join(mlines),
        transform=ax_metrics.transAxes, va="top",
        fontfamily="monospace", fontsize=11, linespacing=1.05,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9)
    )


    # -------------------------------------------------------------------------
    # PANEL 3: MDF
    # -------------------------------------------------------------------------
    ax_mdf = fig.add_subplot(gs[1, :2])
    order = np.argsort(df["posterior_w"].to_numpy())[::-1][:N]
    for i in order:
        row = df.iloc[i]
        x, y = _extract_mdf_xy(row)
        ax_mdf.plot(x, y, alpha=0.05, lw=0.7, color="gray")
    bx, by = _extract_mdf_xy(best_row)
    ax_mdf.plot(bx, by, color="red", lw=2.0, label="best model")
    ax_mdf.plot(x_obs_mdf, y_obs_mdf, "x", color="k", ms=4.5, mew=0.9, label="Observed MDF")
    ax_mdf.set_xlabel("[Fe/H]")
    ax_mdf.set_ylabel("MDF (normalized)")
    ax_mdf.set_xlim(-2, 1)

    ax_mdf.legend()
    ax_mdf.grid(alpha=0.3)

    # -------------------------------------------------------------------------
    # PANEL 4: Age–Metallicity Relation
    # -------------------------------------------------------------------------
    ax_age = fig.add_subplot(gs[1, 2:])
    order = np.argsort(df["posterior_w"].to_numpy())[::-1][:N]
    for i in order:
        row = df.iloc[i]
        x = np.asarray(row["age_x"], float)
        x = (x[-1] - x) / 1e9
        y = np.asarray(row["age_y"], float)
        ax_age.plot(x, y, alpha=0.05, lw=0.7, color="gray")
    bx, by = np.asarray(best_row["age_x"], float), np.asarray(best_row["age_y"], float)
    bx = (bx[-1] - bx) / 1e9
    ax_age.plot(bx, by, color="red", lw=2.0, label="best model")
    ax_age.scatter(age_Bensby, Fe_H, color="tab:blue", marker="^", s=35, label="Bensby+14")
    ax_age.scatter(age_Joyce, Fe_H, color="tab:red", marker="*", s=55, label="Joyce+23")
    ax_age.set_xlabel("Age [Gyr]")
    ax_age.set_ylabel("[Fe/H]")
    ax_age.set_xlim(0, 15)
    ax_age.legend()
    ax_age.grid(alpha=0.3)

    # -------------------------------------------------------------------------
    # PANELS 5–8: Alpha Elements (2×2 grid)
    # -------------------------------------------------------------------------
    element_names = ["Mg", "Si", "Ca", "Ti"]
    obs_data = [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]
    btracks = best_row["alpha_tracks"]

    for i, elem in enumerate(element_names):
        row_idx = 2 + i // 2
        col_idx = (i % 2) * 2
        ax = fig.add_subplot(gs[row_idx, col_idx:col_idx + 2])

        # spaghetti
        for j in order:
            row = df.iloc[j]
            tracks = row["alpha_tracks"]
            x = np.asarray(tracks[i][0], float)
            y = np.asarray(tracks[i][1], float)
            ax.plot(x, y, alpha=0.05, lw=0.7, color="gray")

        # best
        bx = np.asarray(btracks[i][0], float)
        by = np.asarray(btracks[i][1], float)
        ax.plot(bx, by, color="red", lw=2.0, label="best model")

        # observed
        obs_x, obs_y = obs_data[i]
        ax.scatter(obs_x, obs_y, color="k", s=10, alpha=0.7)

        ax.set_xlim(-2.5, 0.6)
        ax.set_ylim(-0.2, 0.6)
        ax.set_xlabel("[Fe/H]")
        ax.set_ylabel(f"[{elem}/Fe]")
        ax.text(0.05, 0.9, elem, transform=ax.transAxes,
                fontsize=14, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.grid(alpha=0.3)

    # -------------------------------------------------------------------------
    # Final formatting + save
    # -------------------------------------------------------------------------
    fig.savefig(os.path.join(output_dir, "Omni_Info_Figure.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)
    print('saved posterior omni fig')








def plot_omni_posterior(df: pd.DataFrame, output_dir: str, obs_mdf_path: str | None = None, loss_metric='loss'):
    """
    Combined 'Omni' posterior-band dashboard showing MDF, AMR, and Alpha-element
    posterior ensembles together with best-fit parameters and metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Full combined results table (with posterior weights, model data, etc.)
    output_dir : str
        Output directory for the figure.
    obs_mdf_path : str, optional
        Path to observed MDF file, passed to load_observed_mdf().
    """

    os.makedirs(output_dir, exist_ok=True)
    best_idx = stable_best_index(df, primary=loss_metric)
    best_row = df.loc[best_idx]

    # Observations
    Fe_H, age_Joyce, age_Bensby = load_observed_amr()
    Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe = load_observed_alpha()
    x_obs_mdf, y_obs_mdf = load_observed_mdf(obs_mdf_path)

    # --- locate and parse the pcard used for this run ---
    pcard_hints = [
        os.getcwd(),
        output_dir,
        os.path.dirname(output_dir),
    ]

    # if run_name encodes a folder, try that too
    pcard_path = output_dir.split('/')[0] + '/' + 'bulge_pcard.txt'
    cats, heads = _parse_pcard_categories(pcard_path)


    # Posterior weights
    w = df["posterior_w"].to_numpy()

    # Figure layout
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.35,
                  left=0.05, right=0.97, top=0.95, bottom=0.06)


    # ================================
    # PANEL 1: Parameters (2 columns)  [REPLACE YOUR EXISTING PANEL 1 BLOCK WITH THIS]
    # ================================
    ax_params = fig.add_subplot(gs[0, :2])
    ax_params.axis("off")

    num_keys = ["sigma_2","t_1","t_2","infall_1","infall_2",
                "sfe","delta_sfe","imf_upper","mgal","nb"]

    param_stats = _summarize_params(df, w, num_keys)
    # format: name: median  -d16/+d84
    pairs = [(k, f"{m:.4g}  -{dm:.3g}/+{dp:.3g}") for (k, m, dm, dp) in param_stats]

    mid = math.ceil(len(pairs) / 2)
    left, right = pairs[:mid], pairs[mid:]

    lines = ["BEST-FIT MODEL PARAMETERS (posterior)", "="*36]
    for i in range(mid):
        k1, v1 = left[i]
        if i < len(right):
            k2, v2 = right[i]
            lines.append(f"{k1:<10}: {v1:>18}    {k2:<10}: {v2:>18}")
        else:
            lines.append(f"{k1:<10}: {v1:>18}")

    # categorical choices (basename only), keep your existing resolver
    lines.append("")
    lines.append("CATEGORICAL CHOICES")
    lines.append("-"*21)
    for label, value in _resolve_categorical_labels(best_row, cats, heads):
        val = os.path.basename(str(value))
        lines.append(f"{label:<12}: {val}")

    ax_params.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax_params.transAxes, va="top",
        fontfamily="monospace", fontsize=11, linespacing=1.05,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9)
    )


    # ================================
    # PANEL 2: Fit metrics (simple list)
    # ================================
    ax_metrics = fig.add_subplot(gs[0, 2:])
    ax_metrics.axis("off")

    metric_keys = ["fitness","wrmse","mae","huber","cosine","ks","ensemble"]
    mlines = ["FIT QUALITY METRICS", "="*20]
    for k in metric_keys:
        if k in best_row:
            mlines.append(f"{k:<10}: {best_row[k]:.5f}")

    ax_metrics.text(
        0.02, 0.98, "\n".join(mlines),
        transform=ax_metrics.transAxes, va="top",
        fontfamily="monospace", fontsize=11, linespacing=1.05,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9)
    )



    # ==========================================================
    # PANEL 3: MDF posterior
    # ==========================================================
    ax_mdf = fig.add_subplot(gs[1, :2])
    feh_grid = np.linspace(-2.5, 0.8, 200)
    median, lo16, hi84 = build_mdf_ensemble(df, w, feh_grid)
    ax_mdf.fill_between(feh_grid, lo16, hi84, alpha=0.3, label="68% posterior")
    ax_mdf.plot(feh_grid, median, label="posterior median")
    bx, by = _extract_mdf_xy(best_row)
    ax_mdf.plot(bx, by, color="red", lw=2.0, label="best model")
    ax_mdf.plot(x_obs_mdf, y_obs_mdf, "x", color="k", ms=4.5, mew=0.9, label="Observed MDF")
    ax_mdf.set_xlabel("[Fe/H]")
    ax_mdf.set_ylabel("MDF (normalized)")
    ax_mdf.legend()
    ax_mdf.grid(alpha=0.3)

    # ==========================================================
    # PANEL 4: Age–Metallicity posterior
    # ==========================================================
    ax_age = fig.add_subplot(gs[1, 2:])
    age_grid_gyr = np.linspace(0.0, 15.0, 300)
    median, lo16, hi84 = build_age_feh_ensemble(df, w, age_grid_gyr)
    ax_age.fill_between(age_grid_gyr, lo16, hi84, alpha=0.3, label="68% posterior")
    ax_age.plot(age_grid_gyr, median, label="posterior median")

    bx = np.asarray(best_row["age_x"], float)
    by = np.asarray(best_row["age_y"], float)
    bx = (bx[-1] - bx) / 1e9
    ax_age.plot(bx, by, color="red", lw=2.0, label="best model")

    ax_age.scatter(age_Bensby, Fe_H, color="tab:blue", marker="^", s=35, label="Bensby+14")
    ax_age.scatter(age_Joyce, Fe_H, color="tab:red", marker="*", s=55, label="Joyce+23")
    ax_age.set_xlabel("Age [Gyr]")
    ax_age.set_ylabel("[Fe/H]")
    ax_age.set_xlim(0, 15)
    ax_age.legend()
    ax_age.grid(alpha=0.3)

    # ==========================================================
    # PANELS 5–8: Alpha-element posteriors
    # ==========================================================
    element_names = ["Mg", "Si", "Ca", "Ti"]
    obs_data = [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]
    btracks = best_row["alpha_tracks"]

    for i, elem in enumerate(element_names):
        row_idx = 2 + i // 2
        col_idx = (i % 2) * 2
        ax = fig.add_subplot(gs[row_idx, col_idx:col_idx + 2])

        feh_grid = np.linspace(-2.5, 0.6, 200)
        df_el = df.copy(deep=False)
        df_el["Fe_H_x"] = df_el["alpha_tracks"].apply(lambda tr: np.asarray(tr[i][0], float))
        df_el[f"{elem}_Fe_y"] = df_el["alpha_tracks"].apply(lambda tr: np.asarray(tr[i][1], float))
        median, lo16, hi84 = build_alpha_ensemble(df_el, w, feh_grid, i)

        ax.fill_between(feh_grid, lo16, hi84, alpha=0.3, label="68% posterior")
        ax.plot(feh_grid, median, label="posterior median")

        bx = np.asarray(btracks[i][0], float)
        by = np.asarray(btracks[i][1], float)
        ax.plot(bx, by, color="red", lw=2.0, label="best model")

        obs_x, obs_y = obs_data[i]
        ax.scatter(obs_x, obs_y, color="k", s=10, alpha=0.7)

        ax.set_xlim(-2.5, 0.6)
        ax.set_ylim(-0.2, 0.6)
        ax.set_xlabel("[Fe/H]")
        ax.set_ylabel(f"[{elem}/Fe]")
        ax.text(0.05, 0.9, elem, transform=ax.transAxes,
                fontsize=14, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.grid(alpha=0.3)

    # ==========================================================
    # Final formatting + save
    # ==========================================================
    fig.savefig(os.path.join(output_dir, "Omni_Info_Figure_Posterior.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)
    print('saved posterior omni fig')