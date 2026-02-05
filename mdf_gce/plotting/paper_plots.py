#!/usr/bin/env python3
"""
Publication-quality plotting module for MDF_GCE_SMC_DEMC.

Produces the four main figure types from the paper:
1. MDF posterior with residuals (Figure 7)
2. AMR posterior with residuals (Figure 9)
3. Four-panel alpha element plots (Figure 8)
4. Parameter corner plot (Figure 3)

All plots use weighted posterior distributions from the GA results.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import interp1d

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
    print("Warning: corner package not installed. Corner plots disabled.")

from .style import use_paper_style, PLOT_COLORS

# Apply paper style on import
use_paper_style()


# =============================================================================
# CONSTANTS
# =============================================================================

# Parameter labels for plots (LaTeX)
PARAM_LABELS = {
    'sigma_2': r'$\sigma_2$',
    't_1': r'$t_1$ [Gyr]',
    't_2': r'$t_2$ [Gyr]',
    'infall_1': r'$\tau_1$ [Gyr]',
    'infall_2': r'$\tau_2$ [Gyr]',
    'sfe': r'SFE [Gyr$^{-1}$]',
    'delta_sfe': r'$\Delta$SFE [Gyr$^{-1}$]',
    'imf_upper': r'$M_{\rm max}$ [$M_\odot$]',
    'mgal': r'$M_{\rm gal}$ [$M_\odot$]',
    'nb': r'$N_{\rm Ia}/M_\odot$',
}

CONTINUOUS_PARAMS = [
    'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2',
    'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb'
]

# Colors matching paper style
COLOR_POSTERIOR = "Blues"      # Colormap for 2D posterior density
COLOR_BEST = "#d62728"         # Red for best/MAP model
COLOR_MEDIAN = "#1f77b4"       # Blue for median
COLOR_OBS = "black"            # Black for observations
COLOR_JOYCE = "#d62728"        # Red stars for Joyce data
COLOR_BENSBY = "#1f77b4"       # Blue triangles for Bensby data


# =============================================================================
# POSTERIOR WEIGHTING UTILITIES
# =============================================================================

def compute_posterior_weights(
    loss: np.ndarray,
    temperature: Optional[float] = None,
    mode: str = 'exp',
    floor: float = 1e-12,
) -> Tuple[np.ndarray, float, float]:
    """
    Convert loss values to normalized posterior weights.
    
    Parameters
    ----------
    loss : array-like
        Loss/fitness values (lower is better)
    temperature : float, optional
        Temperature for exponential weighting. If None, auto-tune.
    mode : str
        'exp' for exponential, 'inv' for inverse weighting
    floor : float
        Minimum temperature to prevent overflow
        
    Returns
    -------
    weights : ndarray
        Normalized weights summing to 1
    temperature : float
        Temperature used
    ess : float
        Effective sample size
    """
    loss = np.asarray(loss, dtype=float)
    
    # Remove invalid entries
    valid = np.isfinite(loss)
    if not np.any(valid):
        n = len(loss)
        return np.ones(n) / n, 1.0, 1.0
    
    loss_valid = loss[valid]
    
    # Auto-tune temperature if not provided
    if temperature is None:
        # Use MAD (median absolute deviation) as robust scale
        median = np.median(loss_valid)
        mad = np.median(np.abs(loss_valid - median))
        temperature = max(mad, floor) if mad > 0 else max(np.std(loss_valid), floor)
    
    temperature = max(temperature, floor)
    
    # Compute weights
    if mode == 'exp':
        # Shift to prevent overflow
        loss_shifted = loss - np.nanmin(loss)
        log_weights = -loss_shifted / temperature
        log_weights = np.where(valid, log_weights, -np.inf)
        # Normalize in log space
        max_log = np.max(log_weights[valid])
        weights = np.exp(log_weights - max_log)
    else:  # inverse
        weights = np.where(valid, 1.0 / (loss + floor), 0.0)
    
    # Normalize
    total = np.sum(weights)
    if total > 0:
        weights = weights / total
    else:
        weights = np.ones_like(weights) / len(weights)
    
    # Effective sample size
    ess = 1.0 / (np.sum(weights**2) + 1e-12)
    
    return weights, temperature, ess


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: List[float],
) -> np.ndarray:
    """Compute weighted quantiles."""
    values = np.asarray(values)
    weights = np.asarray(weights)
    
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.full(len(quantiles), np.nan)
    
    v = values[valid]
    w = weights[valid]
    w = w / w.sum()
    
    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]
    
    cumsum = np.cumsum(w_sorted)
    
    results = []
    for q in quantiles:
        idx = np.searchsorted(cumsum, q)
        idx = min(idx, len(v_sorted) - 1)
        results.append(v_sorted[idx])
    
    return np.array(results)


# =============================================================================
# DATA EXTRACTION UTILITIES
# =============================================================================

def extract_mdf_xy(row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Extract MDF x, y arrays from a results row."""
    # Try different column name conventions
    for x_col in ['mdf_x', 'MDF_x', 'feh_grid']:
        if x_col in row.index:
            x = row[x_col]
            break
    else:
        return np.array([]), np.array([])
    
    for y_col in ['mdf_y', 'MDF_y', 'mdf_values']:
        if y_col in row.index:
            y = row[y_col]
            break
    else:
        return np.array([]), np.array([])
    
    # Handle string representations
    if isinstance(x, str):
        try:
            import ast
            x = np.array(ast.literal_eval(x))
        except:
            return np.array([]), np.array([])
    if isinstance(y, str):
        try:
            import ast
            y = np.array(ast.literal_eval(y))
        except:
            return np.array([]), np.array([])
    
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def extract_age_feh(row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract age-metallicity relation from a results row.
    
    Converts simulation time (years from universe start) to lookback age (Gyr):
    age_gyr = (t_final - t) / 1e9
    
    This gives: oldest stars (formed early) have largest age values.
    
    Returns
    -------
    age_gyr, feh : arrays
    """
    # Find age/time column
    x = None
    for x_col in ['age_x', 'age_array', 'ages', 'time']:
        if x_col in row.index:
            x = row[x_col]
            break
    
    if x is None:
        return np.array([]), np.array([])
    
    # Find metallicity column  
    y = None
    for y_col in ['age_y', 'feh_array', 'metallicity', '[Fe/H]']:
        if y_col in row.index:
            y = row[y_col]
            break
    
    if y is None:
        return np.array([]), np.array([])
    
    # Parse string representations if needed
    if isinstance(x, str):
        try:
            import ast
            x = np.array(ast.literal_eval(x))
        except:
            return np.array([]), np.array([])
    if isinstance(y, str):
        try:
            import ast
            y = np.array(ast.literal_eval(y))
        except:
            return np.array([]), np.array([])
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if len(x) == 0 or len(y) == 0:
        return np.array([]), np.array([])
    
    # Ensure same length
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    
    # Convert simulation time (years from Big Bang) to stellar age (Gyr)
    # If max value > 1e6, assume it's in years (typical ~14e9 years)
    if np.nanmax(np.abs(x)) > 1e6:
        t_final = x[-1] if len(x) > 0 else 0
        # age_gyr = how long ago this timestep occurred
        x = (t_final - x) / 1e9
    
    return x, y


def extract_alpha_track(row: pd.Series, element: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract alpha element track from a results row."""
    col_map = {
        'Mg': ['alpha_mg', 'Mg_Fe', '[Mg/Fe]'],
        'Si': ['alpha_si', 'Si_Fe', '[Si/Fe]'],
        'Ca': ['alpha_ca', 'Ca_Fe', '[Ca/Fe]'],
        'Ti': ['alpha_ti', 'Ti_Fe', '[Ti/Fe]'],
    }
    
    # Try to get from alpha_tracks or alpha_data
    if 'alpha_tracks' in row.index:
        tracks = row['alpha_tracks']
        if isinstance(tracks, str):
            try:
                import ast
                tracks = ast.literal_eval(tracks)
            except:
                return np.array([]), np.array([])
        
        elem_idx = {'Mg': 0, 'Si': 1, 'Ca': 2, 'Ti': 3}.get(element, -1)
        if elem_idx >= 0 and elem_idx < len(tracks):
            x, y = tracks[elem_idx]
            return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    
    # Try individual columns
    for y_col in col_map.get(element, []):
        if y_col in row.index:
            y = row[y_col]
            # x is typically [Fe/H]
            if 'feh_alpha' in row.index:
                x = row['feh_alpha']
            elif 'alpha_feh' in row.index:
                x = row['alpha_feh']
            else:
                continue
            
            if isinstance(x, str):
                try:
                    import ast
                    x = np.array(ast.literal_eval(x))
                except:
                    continue
            if isinstance(y, str):
                try:
                    import ast
                    y = np.array(ast.literal_eval(y))
                except:
                    continue
            
            return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    
    return np.array([]), np.array([])


# =============================================================================
# ENSEMBLE BUILDING
# =============================================================================

def build_mdf_ensemble(
    df: pd.DataFrame,
    weights: np.ndarray,
    feh_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build weighted MDF ensemble statistics.
    
    Returns
    -------
    median, lo16, hi84 : arrays
        Median and 16/84 percentile bounds on feh_grid
    """
    n_models = len(df)
    n_grid = len(feh_grid)
    y_stack = np.full((n_models, n_grid), np.nan)
    
    for i, (_, row) in enumerate(df.iterrows()):
        x, y = extract_mdf_xy(row)
        if len(x) < 2:
            continue
        
        # Interpolate to common grid
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < 2:
            continue
        
        y_interp = np.interp(feh_grid, x[valid], y[valid], left=np.nan, right=np.nan)
        
        # Normalize
        if np.nansum(y_interp) > 0:
            y_interp = y_interp / np.nanmax(y_interp)
        
        y_stack[i] = y_interp
    
    # Compute weighted quantiles at each grid point
    median = np.zeros(n_grid)
    lo16 = np.zeros(n_grid)
    hi84 = np.zeros(n_grid)
    
    for j in range(n_grid):
        col = y_stack[:, j]
        valid = np.isfinite(col)
        if np.sum(valid) < 3:
            median[j] = lo16[j] = hi84[j] = np.nan
            continue
        
        q = weighted_quantile(col, weights, [0.16, 0.5, 0.84])
        lo16[j], median[j], hi84[j] = q
    
    return median, lo16, hi84


def build_amr_ensemble(
    df: pd.DataFrame,
    weights: np.ndarray,
    age_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build weighted age-metallicity ensemble statistics.
    
    Returns
    -------
    median, lo16, hi84 : arrays
        Median and 16/84 percentile bounds of [Fe/H] on age_grid
    """
    n_models = len(df)
    n_grid = len(age_grid)
    y_stack = np.full((n_models, n_grid), np.nan)
    
    for i, (_, row) in enumerate(df.iterrows()):
        x, y = extract_age_feh(row)
        if len(x) < 2:
            continue
        
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < 2:
            continue
        
        # Sort by age
        order = np.argsort(x[valid])
        x_sorted = x[valid][order]
        y_sorted = y[valid][order]
        
        y_interp = np.interp(age_grid, x_sorted, y_sorted, left=np.nan, right=np.nan)
        y_stack[i] = y_interp
    
    median = np.zeros(n_grid)
    lo16 = np.zeros(n_grid)
    hi84 = np.zeros(n_grid)
    
    for j in range(n_grid):
        col = y_stack[:, j]
        valid = np.isfinite(col)
        if np.sum(valid) < 3:
            median[j] = lo16[j] = hi84[j] = np.nan
            continue
        
        q = weighted_quantile(col, weights, [0.16, 0.5, 0.84])
        lo16[j], median[j], hi84[j] = q
    
    return median, lo16, hi84


def build_alpha_ensemble(
    df: pd.DataFrame,
    weights: np.ndarray,
    feh_grid: np.ndarray,
    element: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build weighted alpha element ensemble statistics.
    
    Returns
    -------
    median, lo16, hi84 : arrays
        Median and 16/84 percentile bounds of [X/Fe] on feh_grid
    """
    n_models = len(df)
    n_grid = len(feh_grid)
    y_stack = np.full((n_models, n_grid), np.nan)
    
    for i, (_, row) in enumerate(df.iterrows()):
        x, y = extract_alpha_track(row, element)
        if len(x) < 2:
            continue
        
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < 2:
            continue
        
        # Sort by [Fe/H]
        order = np.argsort(x[valid])
        x_sorted = x[valid][order]
        y_sorted = y[valid][order]
        
        y_interp = np.interp(feh_grid, x_sorted, y_sorted, left=np.nan, right=np.nan)
        y_stack[i] = y_interp
    
    median = np.zeros(n_grid)
    lo16 = np.zeros(n_grid)
    hi84 = np.zeros(n_grid)
    
    for j in range(n_grid):
        col = y_stack[:, j]
        valid = np.isfinite(col)
        if np.sum(valid) < 3:
            median[j] = lo16[j] = hi84[j] = np.nan
            continue
        
        q = weighted_quantile(col, weights, [0.16, 0.5, 0.84])
        lo16[j], median[j], hi84[j] = q
    
    return median, lo16, hi84


# =============================================================================
# 2D DENSITY UTILITIES
# =============================================================================

def build_2d_density(
    df: pd.DataFrame,
    weights: np.ndarray,
    x_grid: np.ndarray,
    y_bins: np.ndarray,
    extract_fn,
    smooth_sigma: float = 1.2,
) -> np.ndarray:
    """
    Build 2D weighted density histogram for posterior visualization.
    
    Parameters
    ----------
    df : DataFrame
        Results dataframe
    weights : array
        Posterior weights
    x_grid : array
        Grid for x-axis (e.g., [Fe/H] or Age)
    y_bins : array
        Bin edges for y-axis
    extract_fn : callable
        Function to extract (x, y) from a row
    smooth_sigma : float
        Gaussian smoothing sigma
        
    Returns
    -------
    H : 2D array
        Density histogram, shape (len(y_bins)-1, len(x_grid))
    """
    n_x = len(x_grid)
    n_y = len(y_bins) - 1
    H = np.zeros((n_y, n_x))
    
    # Interpolate all models to common x_grid
    n_models = len(df)
    y_stack = np.full((n_models, n_x), np.nan)
    
    for i, (_, row) in enumerate(df.iterrows()):
        x, y = extract_fn(row)
        if len(x) < 2:
            continue
        
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < 2:
            continue
        
        order = np.argsort(x[valid])
        y_interp = np.interp(x_grid, x[valid][order], y[valid][order], 
                            left=np.nan, right=np.nan)
        y_stack[i] = y_interp
    
    # Build 2D histogram
    good_w = np.isfinite(weights) & (weights > 0)
    
    for j in range(n_x):
        y_slice = y_stack[:, j]
        mask = np.isfinite(y_slice) & good_w
        
        if not np.any(mask):
            continue
        
        w_slice = weights[mask]
        y_slice = y_slice[mask]
        
        hist, _ = np.histogram(y_slice, bins=y_bins, weights=w_slice, density=False)
        
        # Normalize this slice
        s = hist.sum()
        if s > 0:
            hist = hist / s
        
        H[:, j] = hist
    
    # Smooth
    if smooth_sigma > 0:
        H = gaussian_filter(H, sigma=smooth_sigma)
    
    # Normalize to [0, 1]
    if H.max() > 0:
        H = H / H.max()
    
    return H


# =============================================================================
# MAIN PLOTTING FUNCTIONS
# =============================================================================

def plot_mdf_posterior(
    df: pd.DataFrame,
    obs_feh: np.ndarray,
    obs_mdf: np.ndarray,
    output_path: str,
    loss_col: str = 'fitness',
    feh_min: float = -1.5,
    feh_max: float = 0.8,
    n_grid_x: int = 200,
    n_grid_y: int = 100,
    smooth_sigma: float = 1.2,
    cmap: str = "Blues",
    posterior_gamma: float = 0.7,
) -> str:
    """
    Create MDF posterior plot matching Figure 7 of the paper.
    
    Parameters
    ----------
    df : DataFrame
        Results with MDF data columns
    obs_feh, obs_mdf : arrays
        Observed MDF
    output_path : str
        Output directory
    loss_col : str
        Column name for loss/fitness values
        
    Returns
    -------
    save_path : str
        Path to saved figure
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Compute weights
    if loss_col not in df.columns:
        loss_col = 'fitness' if 'fitness' in df.columns else 'loss'
    
    loss = df[loss_col].values
    weights, temp, ess = compute_posterior_weights(loss)
    
    # Find best model
    best_idx = np.argmin(loss)
    best_row = df.iloc[best_idx]
    best_x, best_y = extract_mdf_xy(best_row)
    
    # Normalize observed MDF
    obs_mdf_norm = obs_mdf / obs_mdf.max() if obs_mdf.max() > 0 else obs_mdf
    
    # Build grids
    feh_grid = np.linspace(feh_min, feh_max, n_grid_x)
    y_max = max(1.05, np.nanmax(obs_mdf_norm) * 1.1)
    y_bins = np.linspace(0, y_max, n_grid_y + 1)
    
    # Build 2D density
    H = build_2d_density(df, weights, feh_grid, y_bins, extract_mdf_xy, smooth_sigma)
    
    # Create figure with residual panel
    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.05)
    
    ax_main = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_main)
    
    # Main panel: 2D posterior density
    y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
    X, Y = np.meshgrid(feh_grid, y_centers)
    
    # Power norm for better visualization
    norm = mcolors.PowerNorm(gamma=posterior_gamma, vmin=0, vmax=1)
    
    im = ax_main.pcolormesh(X, Y, H, cmap=cmap, norm=norm, shading='auto', rasterized=True)
    
    # Best model
    if len(best_x) > 0:
        best_y_norm = best_y / best_y.max() if best_y.max() > 0 else best_y
        ax_main.plot(best_x, best_y_norm, color=COLOR_BEST, lw=2, label='Best model')
    
    # Observations
    ax_main.plot(obs_feh, obs_mdf_norm, 'x', color=COLOR_OBS, ms=5, mew=1.2, label='Data')
    
    ax_main.set_xlim(feh_min, feh_max)
    ax_main.set_ylim(0, y_max)
    ax_main.set_ylabel('Normalized number', fontsize=14)
    ax_main.legend(loc='upper left', fontsize=11)
    ax_main.tick_params(labelbottom=False)
    
    # Add [Fe/H] label at top
    ax_main.xaxis.set_label_position('top')
    ax_main.set_xlabel('[Fe/H]', fontsize=14)
    ax_main.xaxis.tick_top()
    ax_main.xaxis.set_tick_params(labeltop=True)
    
    # Residual panel
    if len(best_x) > 0:
        # Interpolate best model to observation points
        best_interp = np.interp(obs_feh, best_x, best_y_norm, left=np.nan, right=np.nan)
        residuals = best_interp - obs_mdf_norm
        rms = np.sqrt(np.nanmean(residuals**2))
        
        ax_res.axhline(0, color='gray', ls='--', lw=1)
        ax_res.plot(obs_feh, residuals, 'o', color=COLOR_BEST, ms=4)
        ax_res.text(0.02, 0.85, f'RMS = {rms:.3f}', transform=ax_res.transAxes, 
                   fontsize=11, va='top')
    
    ax_res.set_xlabel('[Fe/H]', fontsize=14)
    ax_res.set_ylabel('Model − Data', fontsize=12)
    ax_res.set_xlim(feh_min, feh_max)
    
    save_path = os.path.join(output_path, 'MDF_posterior_2D.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {save_path}")
    return save_path


def plot_amr_posterior(
    df: pd.DataFrame,
    obs_ages_joyce: np.ndarray,
    obs_ages_bensby: np.ndarray,
    obs_feh: np.ndarray,
    output_path: str,
    loss_col: str = 'fitness',
    age_max: float = 14.0,
    n_grid_x: int = 200,
    n_grid_y: int = 100,
    smooth_sigma: float = 1.2,
    cmap: str = "Blues",
    posterior_gamma: float = 0.7,
) -> str:
    """
    Create AMR posterior plot matching Figure 9 of the paper.
    
    Parameters
    ----------
    df : DataFrame
        Results with age-metallicity data
    obs_ages_joyce, obs_ages_bensby : arrays
        Observed ages from Joyce+23 and Bensby+17
    obs_feh : array
        Observed [Fe/H] values
    output_path : str
        Output directory
        
    Returns
    -------
    save_path : str
        Path to saved figure
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Compute weights
    if loss_col not in df.columns:
        loss_col = 'fitness' if 'fitness' in df.columns else 'loss'
    
    loss = df[loss_col].values
    weights, temp, ess = compute_posterior_weights(loss)
    
    # Find best model
    best_idx = np.argmin(loss)
    best_row = df.iloc[best_idx]
    best_age, best_feh = extract_age_feh(best_row)
    
    # Build grids
    age_grid = np.linspace(0, age_max, n_grid_x)
    feh_bins = np.linspace(-2.0, 1.0, n_grid_y + 1)
    
    # Build 2D density
    H = build_2d_density(df, weights, age_grid, feh_bins, extract_age_feh, smooth_sigma)
    
    # Create figure with residual panel and side MDF panel
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[4, 1], 
                  hspace=0.05, wspace=0.05)
    
    ax_main = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_side = fig.add_subplot(gs[0, 1], sharey=ax_main)
    
    # Main panel: 2D posterior density
    feh_centers = 0.5 * (feh_bins[:-1] + feh_bins[1:])
    X, Y = np.meshgrid(age_grid, feh_centers)
    
    norm = mcolors.PowerNorm(gamma=posterior_gamma, vmin=0, vmax=1)
    im = ax_main.pcolormesh(X, Y, H, cmap=cmap, norm=norm, shading='auto', rasterized=True)
    
    # Best model
    if len(best_age) > 0:
        ax_main.plot(best_age, best_feh, color=COLOR_BEST, lw=2, label='Best model')
    
    # Observations
    valid_joyce = np.isfinite(obs_ages_joyce) & np.isfinite(obs_feh)
    valid_bensby = np.isfinite(obs_ages_bensby) & np.isfinite(obs_feh)
    
    ax_main.scatter(obs_ages_joyce[valid_joyce], obs_feh[valid_joyce], 
                   marker='*', s=80, color=COLOR_JOYCE, label='Joyce+23', zorder=5)
    ax_main.scatter(obs_ages_bensby[valid_bensby], obs_feh[valid_bensby],
                   marker='^', s=50, color=COLOR_BENSBY, label='Bensby+17', zorder=5)
    
    ax_main.set_xlim(0, age_max)
    ax_main.set_ylim(-2.0, 1.0)
    ax_main.set_ylabel('[Fe/H]', fontsize=14)
    ax_main.legend(loc='upper left', fontsize=10)
    ax_main.tick_params(labelbottom=False)
    
    # Residual panel
    if len(best_age) > 0:
        # Interpolate best model to Joyce ages
        f_interp = interp1d(best_age, best_feh, bounds_error=False, fill_value=np.nan)
        
        res_joyce = f_interp(obs_ages_joyce[valid_joyce]) - obs_feh[valid_joyce]
        res_bensby = f_interp(obs_ages_bensby[valid_bensby]) - obs_feh[valid_bensby]
        
        ax_res.axhline(0, color='gray', ls='--', lw=1)
        ax_res.scatter(obs_ages_joyce[valid_joyce], res_joyce, 
                      marker='*', s=40, color=COLOR_JOYCE, alpha=0.7, label='Joyce+23')
        ax_res.scatter(obs_ages_bensby[valid_bensby], res_bensby,
                      marker='^', s=30, color=COLOR_BENSBY, alpha=0.7, label='Bensby+17')
        ax_res.legend(loc='upper left', fontsize=9)
    
    ax_res.set_xlabel('Age (Gyr)', fontsize=14)
    ax_res.set_ylabel('Model − Obs [Fe/H]', fontsize=12)
    ax_res.set_xlim(0, age_max)
    
    # Side panel: MDF histogram
    # Plot observed MDF vertically
    ax_side.set_xlabel('Normalized counts', fontsize=12)
    ax_side.yaxis.set_label_position('right')
    ax_side.yaxis.tick_right()
    ax_side.tick_params(axis='y', labelright=True, labelleft=False)
    
    # Histogram of observed [Fe/H]
    hist, edges = np.histogram(obs_feh[np.isfinite(obs_feh)], bins=30, density=True)
    hist_norm = hist / hist.max() if hist.max() > 0 else hist
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax_side.fill_betweenx(centers, 0, hist_norm, color='gray', alpha=0.3)
    ax_side.plot(hist_norm, centers, color='black', lw=1.5)
    ax_side.set_xlim(0, 1.15)
    
    save_path = os.path.join(output_path, 'AMR_posterior.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {save_path}")
    return save_path


def plot_alpha_posterior(
    df: pd.DataFrame,
    obs_feh: np.ndarray,
    obs_alpha: Dict[str, np.ndarray],
    output_path: str,
    loss_col: str = 'fitness',
    feh_min: float = -2.5,
    feh_max: float = 0.6,
    n_grid: int = 200,
    smooth_sigma: float = 1.0,
    cmap: str = "Blues",
    posterior_gamma: float = 0.6,
) -> str:
    """
    Create four-panel alpha element posterior plot matching Figure 8.
    
    Parameters
    ----------
    df : DataFrame
        Results with alpha track data
    obs_feh : array
        Observed [Fe/H] values
    obs_alpha : dict
        Dict with keys 'Mg', 'Si', 'Ca', 'Ti' containing observed [X/Fe]
    output_path : str
        Output directory
        
    Returns
    -------
    save_path : str
        Path to saved figure
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Compute weights
    if loss_col not in df.columns:
        loss_col = 'fitness' if 'fitness' in df.columns else 'loss'
    
    loss = df[loss_col].values
    weights, temp, ess = compute_posterior_weights(loss)
    
    # Find best model
    best_idx = np.argmin(loss)
    best_row = df.iloc[best_idx]
    
    elements = ['Mg', 'Si', 'Ca', 'Ti']
    
    # Build grids
    feh_grid = np.linspace(feh_min, feh_max, n_grid)
    alpha_bins = np.linspace(-0.6, 0.8, 50)
    
    # Marginal histogram bins
    feh_hist_bins = np.linspace(feh_min, feh_max, 40)
    alpha_hist_bins = np.linspace(-0.6, 0.8, 40)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    
    for i, elem in enumerate(elements):
        ax = axes[i]
        
        # Build 2D density for this element
        def extract_elem(row):
            return extract_alpha_track(row, elem)
        
        H = build_2d_density(df, weights, feh_grid, alpha_bins, extract_elem, smooth_sigma)
        
        # Plot density
        alpha_centers = 0.5 * (alpha_bins[:-1] + alpha_bins[1:])
        X, Y = np.meshgrid(feh_grid, alpha_centers)
        
        norm = mcolors.PowerNorm(gamma=posterior_gamma, vmin=0, vmax=1)
        ax.pcolormesh(X, Y, H, cmap=cmap, norm=norm, shading='auto', rasterized=True)
        
        # Best model track
        best_x, best_y = extract_alpha_track(best_row, elem)
        if len(best_x) > 0:
            # Smooth the track
            valid = np.isfinite(best_x) & np.isfinite(best_y)
            if np.sum(valid) > 5:
                best_y_smooth = gaussian_filter1d(best_y[valid], sigma=3)
                ax.plot(best_x[valid], best_y_smooth, color=COLOR_BEST, lw=2.5, label='Best model')
        
        # Observations
        if elem in obs_alpha:
            obs_y = obs_alpha[elem]
            valid = np.isfinite(obs_feh) & np.isfinite(obs_y)
            ax.scatter(obs_feh[valid], obs_y[valid], s=15, color='black', 
                      alpha=0.5, label='Observed', zorder=3)
        
        ax.set_xlim(feh_min, feh_max)
        ax.set_ylim(-0.6, 0.8)
        ax.set_xlabel('[Fe/H]', fontsize=12)
        ax.set_ylabel(f'[{elem}/Fe]', fontsize=12)
        
        # Element label in corner
        ax.text(0.95, 0.95, elem, transform=ax.transAxes, fontsize=16, 
               fontweight='bold', va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if i == 0:
            ax.legend(loc='lower left', fontsize=9)
    
    fig.subplots_adjust(hspace=0.15, wspace=0.15)
    
    save_path = os.path.join(output_path, 'Four_Panel_Alpha_Posterior.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {save_path}")
    return save_path


def plot_corner_posterior(
    df: pd.DataFrame,
    output_path: str,
    loss_col: str = 'fitness',
    params: Optional[List[str]] = None,
    bins: int = 40,
    smooth: float = 0.9,
    hdi_mass: float = 0.68,
) -> str:
    """
    Create parameter corner plot matching Figure 3.
    
    Parameters
    ----------
    df : DataFrame
        Results dataframe
    output_path : str
        Output directory
    loss_col : str
        Column for loss values
    params : list, optional
        Parameters to include (default: all continuous)
    bins : int
        Number of histogram bins
    smooth : float
        Smoothing parameter
    hdi_mass : float
        Mass for HDI calculation
        
    Returns
    -------
    save_path : str
        Path to saved figure
    """
    if not HAS_CORNER:
        print("Corner package not available. Skipping corner plot.")
        return ""
    
    os.makedirs(output_path, exist_ok=True)
    
    # Determine parameters
    if params is None:
        params = [p for p in CONTINUOUS_PARAMS if p in df.columns]
    else:
        params = [p for p in params if p in df.columns]
    
    if len(params) < 2:
        print("Not enough parameters for corner plot")
        return ""
    
    # Compute weights
    if loss_col not in df.columns:
        loss_col = 'fitness' if 'fitness' in df.columns else 'loss'
    
    loss = df[loss_col].values
    weights, temp, ess = compute_posterior_weights(loss)
    
    # Extract data
    data = df[params].values
    labels = [PARAM_LABELS.get(p, p) for p in params]
    
    # Find MAP
    best_idx = np.argmin(loss)
    map_values = data[best_idx]
    
    # Create corner plot
    fig = corner.corner(
        data,
        labels=labels,
        weights=weights,
        quantiles=None,
        show_titles=False,
        bins=bins,
        smooth=smooth,
        smooth1d=smooth,
        plot_datapoints=True,
        plot_density=True,
        plot_contours=True,
        fill_contours=True,
        levels=[1 - np.exp(-0.5 * r**2) for r in [1, 2]],
        color='#1f77b4',
    )
    
    # Add MAP marker and HDI annotations
    ndim = len(params)
    axes = np.array(fig.axes).reshape((ndim, ndim))
    
    for i in range(ndim):
        ax = axes[i, i]
        
        # Compute MAP and HDI for this parameter
        xi = data[:, i]
        valid = np.isfinite(xi)
        
        # Weighted histogram for HDI
        hist, edges = np.histogram(xi[valid], bins=bins, weights=weights[valid], density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        
        # Find mode (MAP for this marginal)
        mode_idx = np.argmax(hist)
        mode = centers[mode_idx]
        
        # HDI
        q = weighted_quantile(xi, weights, [0.5 - hdi_mass/2, 0.5, 0.5 + hdi_mass/2])
        lo, med, hi = q
        
        # Mark on histogram
        ax.axvline(map_values[i], color='black', ls='-', lw=1.5)
        ax.axvline(lo, color='black', ls='--', lw=1)
        ax.axvline(hi, color='black', ls='--', lw=1)
        
        # Title with MAP and HDI
        title = f"{labels[i]}\nMAP={map_values[i]:.3g}\n+{hi-med:.3g}−{med-lo:.3g}"
        ax.set_title(title, fontsize=10)
    
    # Mark MAP in 2D panels
    for i in range(ndim):
        for j in range(i):
            ax = axes[i, j]
            ax.axvline(map_values[j], color='black', ls='-', lw=0.8, alpha=0.5)
            ax.axhline(map_values[i], color='black', ls='-', lw=0.8, alpha=0.5)
            ax.plot(map_values[j], map_values[i], 'x', color='black', ms=8, mew=2)
    
    save_path = os.path.join(output_path, 'posterior_corner.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {save_path}")
    return save_path


# =============================================================================
# HIGH-LEVEL INTERFACE
# =============================================================================

def generate_all_paper_plots(
    results_file: str,
    output_path: str,
    obs_mdf_file: Optional[str] = None,
    obs_age_file: Optional[str] = None,
    loss_col: str = 'fitness',
) -> Dict[str, str]:
    """
    Generate all paper-quality plots from results file.
    
    Parameters
    ----------
    results_file : str
        Path to simulation_results.csv or similar
    output_path : str
        Output directory for plots
    obs_mdf_file : str, optional
        Path to observed MDF file
    obs_age_file : str, optional
        Path to observed age data file
    loss_col : str
        Column name for loss/fitness values
        
    Returns
    -------
    paths : dict
        Dictionary mapping plot name to file path
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Load results
    print(f"Loading results from {results_file}")
    df = pd.read_csv(results_file)
    
    # Check for required data columns
    has_mdf = any(c in df.columns for c in ['mdf_x', 'MDF_x'])
    has_amr = any(c in df.columns for c in ['age_x', 'ages'])
    has_alpha = 'alpha_tracks' in df.columns
    
    paths = {}
    
    # Load observational data
    obs_feh = None
    obs_mdf = None
    obs_ages_joyce = None
    obs_ages_bensby = None
    obs_alpha = {}
    
    if obs_mdf_file and os.path.exists(obs_mdf_file):
        data = np.loadtxt(obs_mdf_file, usecols=(0, 1))
        obs_feh = data[:, 0]
        obs_mdf = data[:, 1]
    
    if obs_age_file and os.path.exists(obs_age_file):
        # Try to load Bensby-style TSV
        try:
            age_df = pd.read_csv(obs_age_file, sep='\t')
            if '[Fe/H]' in age_df.columns:
                obs_feh = age_df['[Fe/H]'].values
            if 'Joyce_age' in age_df.columns:
                obs_ages_joyce = age_df['Joyce_age'].values
            if 'Bensby' in age_df.columns:
                obs_ages_bensby = age_df['Bensby'].values
            
            # Alpha elements
            for elem, col in [('Mg', '[Mg/Fe]'), ('Si', '[Si/Fe]'), 
                             ('Ca', '[Ca/Fe]'), ('Ti', '[Ti/Fe]')]:
                if col in age_df.columns:
                    obs_alpha[elem] = age_df[col].values
        except Exception as e:
            print(f"Warning: Could not load age data: {e}")
    
    # Generate plots
    if has_mdf and obs_feh is not None and obs_mdf is not None:
        print("Generating MDF posterior plot...")
        paths['mdf'] = plot_mdf_posterior(df, obs_feh, obs_mdf, output_path, loss_col)
    else:
        print("Skipping MDF plot (missing data)")
    
    if has_amr and obs_feh is not None:
        if obs_ages_joyce is None:
            obs_ages_joyce = np.full_like(obs_feh, np.nan)
        if obs_ages_bensby is None:
            obs_ages_bensby = np.full_like(obs_feh, np.nan)
        
        print("Generating AMR posterior plot...")
        paths['amr'] = plot_amr_posterior(df, obs_ages_joyce, obs_ages_bensby, 
                                          obs_feh, output_path, loss_col)
    else:
        print("Skipping AMR plot (missing data)")
    
    if has_alpha and obs_alpha and obs_feh is not None:
        print("Generating alpha posterior plot...")
        paths['alpha'] = plot_alpha_posterior(df, obs_feh, obs_alpha, output_path, loss_col)
    else:
        print("Skipping alpha plot (missing data)")
    
    print("Generating corner plot...")
    paths['corner'] = plot_corner_posterior(df, output_path, loss_col)
    
    print(f"\nGenerated {len(paths)} plots in {output_path}")
    return paths


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate paper-quality plots')
    parser.add_argument('results', help='Path to results CSV file')
    parser.add_argument('-o', '--output', default='.', help='Output directory')
    parser.add_argument('--obs-mdf', help='Observed MDF file')
    parser.add_argument('--obs-age', help='Observed age data file')
    parser.add_argument('--loss-col', default='fitness', help='Loss column name')
    
    args = parser.parse_args()
    
    generate_all_paper_plots(
        args.results,
        args.output,
        args.obs_mdf,
        args.obs_age,
        args.loss_col,
    )
