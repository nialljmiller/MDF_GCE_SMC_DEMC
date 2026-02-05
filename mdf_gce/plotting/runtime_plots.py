#!/usr/bin/env python3
"""
Runtime plotting for MDF_GCE_SMC_DEMC.

These functions are designed to work with LIVE data during GA training,
directly accessing evaluation_results rather than loading from files.

Key difference from paper_plots.py:
- paper_plots.py: Post-hoc plotting from saved CSV/NPZ files
- runtime_plots.py: Real-time plotting from evaluation_results in memory
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import binned_statistic
from typing import Dict, List, Optional, Tuple, Any

# Try to import style
try:
    from mdf_gce.plotting.style import (
        use_paper_style, 
        COLOR_BEST, 
        COLOR_OBS,
        COLOR_JOYCE,
        COLOR_BENSBY
    )
    use_paper_style()
except ImportError:
    COLOR_BEST = 'red'
    COLOR_OBS = 'black'
    COLOR_JOYCE = 'red'
    COLOR_BENSBY = 'blue'


def smooth_track(x_data, y_data, sigma=3):
    """Smooth a track using Gaussian filter."""
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    x = np.asarray(x_data)[mask]
    y = np.asarray(y_data)[mask]
    if len(x) < 10:
        return x, y
    return gaussian_filter1d(x, sigma=sigma, mode='nearest'), gaussian_filter1d(y, sigma=sigma, mode='nearest')


def build_curve_dicts(evaluation_results: List[Dict]) -> Tuple[Dict, Dict, Dict]:
    """
    Build curve data dictionaries from evaluation results.
    
    Returns
    -------
    mdf_data : dict
        Maps index to (feh_bins, counts) for MDF
    alpha_data : dict  
        Maps index to dict of element tracks {element: (x, y)}
    age_data : dict
        Maps index to (time_years, metallicity) for AMR
    """
    mdf_data = {}
    alpha_data = {}
    age_data = {}
    
    for i, r in enumerate(evaluation_results):
        if 'mdf_x' in r and 'mdf_y' in r:
            mdf_data[i] = (np.asarray(r['mdf_x'], dtype=float), 
                          np.asarray(r['mdf_y'], dtype=float))
        
        if 'alpha_data' in r and r['alpha_data']:
            # Convert to list format: [(Mg_x, Mg_y), (Si_x, Si_y), ...]
            ad = r['alpha_data']
            elem_order = ['[Mg/Fe]', '[Si/Fe]', '[Ca/Fe]', '[Ti/Fe]']
            tracks = []
            for el in elem_order:
                if el in ad:
                    tracks.append((np.asarray(ad[el][0], dtype=float),
                                  np.asarray(ad[el][1], dtype=float)))
                else:
                    tracks.append((np.array([]), np.array([])))
            alpha_data[i] = tracks
        
        if 'age_x' in r and 'age_y' in r:
            age_data[i] = (np.asarray(r['age_x'], dtype=float),
                          np.asarray(r['age_y'], dtype=float))
    
    return mdf_data, alpha_data, age_data


def get_best_model_idx(evaluation_results: List[Dict], fitness_col: str = 'fitness') -> int:
    """Find index of best (lowest fitness) model."""
    best_idx = 0
    best_fitness = float('inf')
    
    for i, r in enumerate(evaluation_results):
        fit = r.get(fitness_col, r.get('fitness', float('inf')))
        if fit < best_fitness:
            best_fitness = fit
            best_idx = i
    
    return best_idx


def plot_mdf_runtime(
    evaluation_results: List[Dict],
    obs_feh: np.ndarray,
    obs_mdf: np.ndarray,
    output_path: str,
    gen: Optional[int] = None,
) -> str:
    """
    Plot MDF with all models and best model highlighted.
    
    Parameters
    ----------
    evaluation_results : list
        Live results from GA
    obs_feh : array
        Observed [Fe/H] bins
    obs_mdf : array
        Observed MDF counts (normalized)
    output_path : str
        Output directory
    gen : int, optional
        Generation number for filename
        
    Returns
    -------
    save_path : str
    """
    mdf_data, _, _ = build_curve_dicts(evaluation_results)
    best_idx = get_best_model_idx(evaluation_results)
    
    os.makedirs(output_path, exist_ok=True)
    
    fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=(9, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    
    # Normalize observations
    obs_feh = np.asarray(obs_feh, dtype=float)
    obs_mdf = np.asarray(obs_mdf, dtype=float)
    if obs_mdf.max() > 0:
        obs_mdf = obs_mdf / obs_mdf.max()
    
    # Background curves (faint)
    n_all = len(mdf_data)
    alpha = max(0.02, min(0.6, 8.0 / max(1, n_all)))
    
    for i, (x, y) in mdf_data.items():
        if i == best_idx:
            continue
        if len(x) > 0 and np.nanmax(y) > 0:
            y_norm = y / np.nanmax(y)
            ax_main.plot(x, y_norm, color="0.75", alpha=0.15 * alpha, lw=0.8, zorder=1)
    
    # Best model
    if best_idx in mdf_data:
        x_best, y_best = mdf_data[best_idx]
        if len(x_best) > 0:
            order = np.argsort(x_best)
            x_best = x_best[order]
            y_best = y_best[order]
            if np.nanmax(y_best) > 0:
                y_best = y_best / np.nanmax(y_best)
            ax_main.plot(x_best, y_best, color=COLOR_BEST, lw=2.0, label="Best model", zorder=3)
    
    # Observations
    ax_main.plot(obs_feh, obs_mdf, "x", color=COLOR_OBS, ms=5, mew=1.0, label="Data", zorder=4)
    
    ax_main.set_xlim(-2.5, 1.0)
    ax_main.set_ylabel("Normalized number", fontsize=12)
    ax_main.legend(loc="upper left", fontsize=10)
    ax_main.xaxis.set_ticks_position("top")
    ax_main.xaxis.set_label_position("top")
    ax_main.set_xlabel("[Fe/H]", fontsize=12)
    ax_main.tick_params(axis="x", bottom=False)
    
    # Residuals
    if best_idx in mdf_data:
        x_best, y_best = mdf_data[best_idx]
        if len(x_best) > 0:
            order = np.argsort(x_best)
            f_best = interp1d(x_best[order], y_best[order] / np.nanmax(y_best[order]), 
                             kind="linear", bounds_error=False, fill_value=0.0)
            y_model = f_best(obs_feh)
            res = y_model - obs_mdf
            
            ax_res.axhline(0.0, ls="--", lw=1.0, color="gray")
            ax_res.plot(obs_feh, res, "o", ms=4, color=COLOR_BEST)
            
            s = np.nanstd(res)
            if s > 0:
                ax_res.set_ylim(-3*s, 3*s)
    
    ax_res.set_xlabel("[Fe/H]", fontsize=12)
    ax_res.set_ylabel("Model − Data", fontsize=11)
    ax_res.set_xlim(-2.5, 1.0)
    
    suffix = f"_gen{gen}" if gen is not None else ""
    save_path = os.path.join(output_path, f'MDF_runtime{suffix}.png')
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    
    print(f"Saved: {save_path}")
    return save_path


def plot_amr_runtime(
    evaluation_results: List[Dict],
    obs_feh: np.ndarray,
    obs_age_joyce: np.ndarray,
    obs_age_bensby: np.ndarray,
    output_path: str,
    gen: Optional[int] = None,
    age_limit_gyr: float = 14.0,
) -> str:
    """
    Plot age-metallicity relation with all models and best highlighted.
    
    Parameters
    ----------
    evaluation_results : list
        Live results from GA
    obs_feh : array
        Observed [Fe/H]
    obs_age_joyce : array
        Ages from Joyce+23
    obs_age_bensby : array
        Ages from Bensby+17
    output_path : str
        Output directory
    gen : int, optional
        Generation number
    age_limit_gyr : float
        Maximum age for x-axis
        
    Returns
    -------
    save_path : str
    """
    _, _, age_data = build_curve_dicts(evaluation_results)
    mdf_data, _, _ = build_curve_dicts(evaluation_results)
    best_idx = get_best_model_idx(evaluation_results)
    
    os.makedirs(output_path, exist_ok=True)
    
    obs_feh = np.asarray(obs_feh, dtype=float)
    obs_age_joyce = np.asarray(obs_age_joyce, dtype=float)
    obs_age_bensby = np.asarray(obs_age_bensby, dtype=float)
    
    # Figure layout
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[3, 1],
                          wspace=0.02, hspace=0.05,
                          left=0.08, right=0.95, top=0.95, bottom=0.08)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_side = fig.add_subplot(gs[0, 1], sharey=ax_main)
    
    best_age_gyr, best_feh = None, None
    
    # Plot all model tracks
    n_all = len(age_data)
    alpha = max(0.02, min(0.5, 5.0 / max(1, n_all)))
    
    for i, (t, feh) in age_data.items():
        if len(t) < 2:
            continue
        
        t = np.asarray(t, dtype=float)
        feh = np.asarray(feh, dtype=float)
        
        # Convert simulation time (years) to stellar age (Gyr)
        # age_gyr = time since star formation = (t_final - t) / 1e9
        t_final = t[-1] if len(t) > 0 else 0
        age_gyr = (t_final - t) / 1e9
        
        if i == best_idx:
            best_age_gyr = age_gyr.copy()
            best_feh = feh.copy()
            ax_main.plot(age_gyr, feh, color=COLOR_BEST, lw=2.5, zorder=5, label='Best model')
        else:
            ax_main.plot(age_gyr, feh, '-', color='gray', lw=0.7, alpha=alpha, zorder=1)
    
    # Observational data
    mJ = np.isfinite(obs_age_joyce) & np.isfinite(obs_feh)
    mB = np.isfinite(obs_age_bensby) & np.isfinite(obs_feh)
    
    ax_main.scatter(obs_age_joyce[mJ], obs_feh[mJ], marker='*', s=60, 
                   color='red', alpha=0.7, zorder=6, label='Joyce+23')
    ax_main.scatter(obs_age_bensby[mB], obs_feh[mB], marker='^', s=50,
                   color='blue', alpha=0.7, zorder=6, label='Bensby+17')
    
    # Binned observational curves
    def binned(x, y, bins):
        m = np.isfinite(x) & np.isfinite(y)
        if not np.any(m):
            return np.array([]), np.array([]), np.array([])
        means, _, _ = binned_statistic(x[m], y[m], statistic='mean', bins=bins)
        stds, _, _ = binned_statistic(x[m], y[m], statistic='std', bins=bins)
        centers = 0.5 * (bins[:-1] + bins[1:])
        valid = np.isfinite(means)
        return centers[valid], means[valid], stds[valid]
    
    age_bins = np.linspace(0, age_limit_gyr, 13)
    cJ, mJm, mJs = binned(obs_age_joyce, obs_feh, age_bins)
    cB, mBm, mBs = binned(obs_age_bensby, obs_feh, age_bins)
    
    if len(cJ) > 0:
        ax_main.plot(cJ, mJm, color='red', lw=2.0, zorder=7)
        ax_main.errorbar(cJ, mJm, yerr=mJs, color='red', alpha=0.3, lw=1.0, capsize=3, zorder=6)
    if len(cB) > 0:
        ax_main.plot(cB, mBm, color='blue', lw=2.0, zorder=7)
        ax_main.errorbar(cB, mBm, yerr=mBs, color='blue', alpha=0.3, lw=1.0, capsize=3, zorder=6)
    
    # Residuals
    if best_age_gyr is not None and best_feh is not None:
        idx = np.argsort(best_age_gyr)
        xs, ys = best_age_gyr[idx], best_feh[idx]
        f_best = interp1d(xs, ys, kind='linear', bounds_error=False, fill_value=np.nan)
        
        rJ = f_best(obs_age_joyce[mJ]) - obs_feh[mJ]
        rB = f_best(obs_age_bensby[mB]) - obs_feh[mB]
        vJ = np.isfinite(rJ)
        vB = np.isfinite(rB)
        
        ax_res.scatter(obs_age_joyce[mJ][vJ], rJ[vJ], marker='*', s=40, 
                      color='red', alpha=0.8, label='Joyce+23')
        ax_res.scatter(obs_age_bensby[mB][vB], rB[vB], marker='^', s=35,
                      color='blue', alpha=0.8, label='Bensby+17')
        
        ax_res.axhline(0.0, ls='--', lw=1.0, color='gray')
        
        residuals = np.concatenate([rJ[vJ], rB[vB]]) if (vJ.any() or vB.any()) else np.array([0])
        if residuals.size > 0:
            s = np.nanstd(residuals)
            ax_res.set_ylim(-max(0.5, 3*s), max(0.5, 3*s))
    
    # Side panel: [Fe/H] histogram
    feh_bins = np.linspace(-2.0, 1.0, 30)
    
    def smooth_hist(vals, bins, sigma=1.2):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return np.zeros(len(bins)-1), 0.5*(bins[:-1]+bins[1:])
        c, e = np.histogram(v, bins=bins)
        c = gaussian_filter1d(c.astype(float), sigma, mode='nearest')
        c = c / c.max() if c.max() > 0 else c
        ctr = 0.5 * (e[:-1] + e[1:])
        return c, ctr
    
    obs_norm, obs_ctr = smooth_hist(obs_feh[np.isfinite(obs_feh)], feh_bins)
    ax_side.fill_betweenx(obs_ctr, 0, obs_norm, color='gray', alpha=0.3, label='Observed')
    ax_side.plot(obs_norm, obs_ctr, lw=1.5, color='black')
    
    # Best model MDF on side panel
    if best_idx in mdf_data:
        mx, my = mdf_data[best_idx]
        if len(mx) > 0:
            counts, edges = np.histogram(mx, bins=feh_bins, weights=my)
            counts = gaussian_filter1d(counts.astype(float), 1.2, mode='nearest')
            if counts.max() > 0:
                counts = counts / counts.max()
            ctr = 0.5 * (edges[:-1] + edges[1:])
            ax_side.fill_betweenx(ctr, 0, counts, color=COLOR_BEST, alpha=0.2, label='Model MDF')
            ax_side.plot(counts, ctr, color=COLOR_BEST, lw=2, ls='--')
    
    # Formatting
    ax_main.set_xlim(0, age_limit_gyr)
    ax_main.set_ylim(-2.0, 1.0)
    ax_main.set_ylabel('[Fe/H]', fontsize=13)
    ax_main.tick_params(axis='x', labelbottom=False)
    ax_main.legend(loc='upper left', fontsize=10, frameon=True)
    
    ax_res.set_xlabel('Age (Gyr)', fontsize=13)
    ax_res.set_ylabel('Model − Obs [Fe/H]', fontsize=11)
    ax_res.set_xlim(0, age_limit_gyr)
    ax_res.legend(loc='upper left', fontsize=9)
    
    ax_side.set_xlabel('Normalized counts', fontsize=11)
    ax_side.set_xlim(0, 1.15)
    ax_side.yaxis.set_label_position('right')
    ax_side.yaxis.tick_right()
    ax_side.tick_params(axis='y', labelright=True, labelleft=False)
    ax_side.legend(loc='lower right', fontsize=9)
    
    suffix = f"_gen{gen}" if gen is not None else ""
    save_path = os.path.join(output_path, f'AMR_runtime{suffix}.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {save_path}")
    return save_path


def plot_alpha_runtime(
    evaluation_results: List[Dict],
    obs_feh: np.ndarray,
    obs_alpha: Dict[str, np.ndarray],
    output_path: str,
    gen: Optional[int] = None,
) -> str:
    """
    Plot four-panel alpha element abundances.
    
    Parameters
    ----------
    evaluation_results : list
        Live results from GA
    obs_feh : array
        Observed [Fe/H]
    obs_alpha : dict
        Dict with keys 'Mg', 'Si', 'Ca', 'Ti' containing observed [X/Fe]
    output_path : str
        Output directory
    gen : int, optional
        Generation number
        
    Returns
    -------
    save_path : str
    """
    _, alpha_data, _ = build_curve_dicts(evaluation_results)
    best_idx = get_best_model_idx(evaluation_results)
    
    os.makedirs(output_path, exist_ok=True)
    
    element_names = ['Mg', 'Si', 'Ca', 'Ti']
    obs_feh = np.asarray(obs_feh, dtype=float)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.subplots_adjust(hspace=0.12, wspace=0.12, left=0.08, right=0.95, top=0.95, bottom=0.08)
    
    xlim = (-2.5, 0.8)
    ylim = (-0.6, 0.8)
    
    n_all = len(alpha_data)
    alpha_bg = max(0.02, min(0.3, 3.0 / max(1, n_all)))
    
    for idx, element in enumerate(element_names):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        
        # Plot all tracks
        for i, tracks in alpha_data.items():
            if idx < len(tracks):
                x, y = tracks[idx]
                if len(x) < 2:
                    continue
                
                x, y = smooth_track(x, y, sigma=3)
                
                if i == best_idx:
                    ax.plot(x, y, color=COLOR_BEST, lw=2.5, zorder=3, label='Best model')
                else:
                    ax.plot(x, y, color='gray', alpha=alpha_bg, lw=0.8, zorder=1)
        
        # Observational data
        if element in obs_alpha:
            obs_y = np.asarray(obs_alpha[element], dtype=float)
            mask = np.isfinite(obs_feh) & np.isfinite(obs_y)
            ax.scatter(obs_feh[mask], obs_y[mask], c='black', s=15, 
                      zorder=2, edgecolor='none', alpha=0.6, label='Data')
        
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        
        if col == 0:
            ax.set_ylabel(r'[$\alpha$/Fe]', fontsize=12)
        else:
            ax.tick_params(axis='y', labelleft=False)
        
        if row == 1:
            ax.set_xlabel('[Fe/H]', fontsize=12)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        
        # Element label
        ax.text(0.05, 0.95, element, transform=ax.transAxes, ha='left', va='top',
               fontsize=20, weight='bold',
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        
        if idx == 0:
            ax.legend(loc='lower left', fontsize=10)
    
    suffix = f"_gen{gen}" if gen is not None else ""
    save_path = os.path.join(output_path, f'Alpha_runtime{suffix}.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {save_path}")
    return save_path


def generate_runtime_plots(
    ga_instance,
    gen: Optional[int] = None,
    plot_mdf: bool = True,
    plot_amr: bool = True,
    plot_alpha: bool = True,
) -> Dict[str, str]:
    """
    Generate all runtime plots from a GA instance.
    
    Parameters
    ----------
    ga_instance : GalacticEvolutionGA
        Live GA instance with evaluation_results
    gen : int, optional
        Generation number
    plot_mdf : bool
        Whether to plot MDF
    plot_amr : bool
        Whether to plot AMR
    plot_alpha : bool
        Whether to plot alpha elements
        
    Returns
    -------
    paths : dict
        Dict of plot names to saved file paths
    """
    if not hasattr(ga_instance, 'evaluation_results') or not ga_instance.evaluation_results:
        print("No evaluation results to plot")
        return {}
    
    plot_dir = os.path.join(ga_instance.output_path, 'plots')
    if gen is not None:
        plot_dir = os.path.join(plot_dir, f'gen{gen}')
    os.makedirs(plot_dir, exist_ok=True)
    
    paths = {}
    
    # Helper to check if obs_age_data exists and is non-empty
    def has_obs_age_data():
        if not hasattr(ga_instance, 'obs_age_data'):
            return False
        obs = ga_instance.obs_age_data
        if obs is None:
            return False
        # Handle DataFrame
        if hasattr(obs, 'empty'):
            return not obs.empty
        # Handle dict
        if isinstance(obs, dict):
            return len(obs) > 0
        return bool(obs)
    
    # MDF plot
    if plot_mdf and hasattr(ga_instance, 'feh') and hasattr(ga_instance, 'normalized_count'):
        try:
            paths['mdf'] = plot_mdf_runtime(
                ga_instance.evaluation_results,
                ga_instance.feh,
                ga_instance.normalized_count,
                plot_dir,
                gen=gen
            )
        except Exception as e:
            print(f"MDF plot error: {e}")
    
    # AMR plot
    if plot_amr and has_obs_age_data():
        obs = ga_instance.obs_age_data
        try:
            # Handle DataFrame vs dict
            if hasattr(obs, 'columns'):
                # It's a DataFrame
                feh_key = '[Fe/H]' if '[Fe/H]' in obs.columns else 'feh' if 'feh' in obs.columns else None
                joyce_key = 'Joyce_age' if 'Joyce_age' in obs.columns else 'joyce' if 'joyce' in obs.columns else None
                bensby_key = 'Bensby' if 'Bensby' in obs.columns else 'bensby' if 'bensby' in obs.columns else None
                
                if feh_key:
                    paths['amr'] = plot_amr_runtime(
                        ga_instance.evaluation_results,
                        obs[feh_key].values if feh_key else np.array([]),
                        obs[joyce_key].values if joyce_key else np.array([]),
                        obs[bensby_key].values if bensby_key else np.array([]),
                        plot_dir,
                        gen=gen
                    )
            else:
                # It's a dict
                feh_key = '[Fe/H]' if '[Fe/H]' in obs else 'feh'
                joyce_key = 'Joyce_age' if 'Joyce_age' in obs else 'joyce'
                bensby_key = 'Bensby' if 'Bensby' in obs else 'bensby'
                
                if feh_key in obs:
                    paths['amr'] = plot_amr_runtime(
                        ga_instance.evaluation_results,
                        obs[feh_key],
                        obs.get(joyce_key, np.array([])),
                        obs.get(bensby_key, np.array([])),
                        plot_dir,
                        gen=gen
                    )
        except Exception as e:
            print(f"AMR plot error: {e}")
            import traceback
            traceback.print_exc()
    
    # Alpha plot
    if plot_alpha and has_obs_age_data():
        obs = ga_instance.obs_age_data
        
        # Handle DataFrame vs dict
        if hasattr(obs, 'columns'):
            cols = obs.columns
            feh_key = '[Fe/H]' if '[Fe/H]' in cols else 'feh' if 'feh' in cols else None
        else:
            feh_key = '[Fe/H]' if '[Fe/H]' in obs else 'feh' if 'feh' in obs else None
        
        # Build alpha dict
        alpha_keys = {
            'Mg': ['[Mg/Fe]', 'Mg_Fe', 'Mg'],
            'Si': ['[Si/Fe]', 'Si_Fe', 'Si'],
            'Ca': ['[Ca/Fe]', 'Ca_Fe', 'Ca'],
            'Ti': ['[Ti/Fe]', 'Ti_Fe', 'Ti'],
        }
        
        obs_alpha = {}
        for elem, keys in alpha_keys.items():
            for k in keys:
                if hasattr(obs, 'columns'):
                    if k in obs.columns:
                        obs_alpha[elem] = obs[k].values
                        break
                else:
                    if k in obs:
                        obs_alpha[elem] = obs[k]
                        break
        
        if obs_alpha and feh_key:
            try:
                feh_vals = obs[feh_key].values if hasattr(obs, 'columns') else obs[feh_key]
                paths['alpha'] = plot_alpha_runtime(
                    ga_instance.evaluation_results,
                    feh_vals,
                    obs_alpha,
                    plot_dir,
                    gen=gen
                )
            except Exception as e:
                print(f"Alpha plot error: {e}")
    
    return paths
