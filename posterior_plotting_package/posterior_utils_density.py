"""
Density-based posterior visualization utilities.

This module extends posterior_utils.py with density-based shading functions
that visualize posterior uncertainty with gradient transparency based on
probability density.

Authors: N Miller
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.stats import gaussian_kde


def compute_density_at_percentiles(y_samples, weights, x_common, n_levels=20, percentiles=[16, 84]):
    """
    Compute probability density at each point for density-based shading.
    
    This function computes the density of the posterior distribution at each
    x-coordinate, which can be used to create gradient shading where regions
    with higher model agreement are darker.
    
    Parameters
    ----------
    y_samples : np.ndarray
        Array of shape (n_samples, n_points) with y values from different models
    weights : np.ndarray
        Weights for each sample (length n_samples)
    x_common : np.ndarray
        Common x-coordinates (length n_points)
    n_levels : int
        Number of density levels for shading (default 20)
    percentiles : list
        Percentile bounds for the uncertainty band (default [16, 84] for 1σ)
    
    Returns
    -------
    density_info : dict
        Dictionary with:
        - 'x': x-coordinates
        - 'y_levels': list of (lower, upper) y-bounds for each density level
        - 'alphas': list of alpha values for each level (darker = higher density)
        - 'median': median y values
        - 'lower': lower percentile bound
        - 'upper': upper percentile bound
    """
    from posterior_utils import weighted_quantile
    
    n_samples, n_points = y_samples.shape
    
    # Compute median and percentile bounds
    median = np.full(n_points, np.nan)
    lower = np.full(n_points, np.nan)
    upper = np.full(n_points, np.nan)
    
    for i in range(n_points):
        y_at_point = y_samples[:, i]
        valid = np.isfinite(y_at_point)
        
        if np.sum(valid) == 0:
            continue
        
        y_valid = y_at_point[valid]
        w_valid = weights[valid]
        w_valid = w_valid / np.sum(w_valid)
        
        pcts = weighted_quantile(y_valid, np.array([percentiles[0], 50, percentiles[1]]) / 100.0, w_valid)
        lower[i], median[i], upper[i] = pcts
    
    # Compute density levels
    # Strategy: divide the uncertainty band into n_levels horizontal slices
    # and compute the probability mass in each slice
    y_levels = []
    alphas = []
    
    # Create levels from median outward
    for level in range(n_levels):
        # Fraction of distance from median to edge
        frac = (level + 1) / n_levels
        
        # Compute y-bounds for this level
        y_lower_level = median - frac * (median - lower)
        y_upper_level = median + frac * (upper - median)
        
        y_levels.append((y_lower_level, y_upper_level))
        
        # Alpha decreases as we move away from median
        # Use quadratic falloff for smooth gradient
        alpha = (1.0 - frac**2) * 0.4  # Max alpha = 0.4
        alphas.append(alpha)
    
    # Reverse so we plot from outside in (darker on top)
    y_levels = list(reversed(y_levels))
    alphas = list(reversed(alphas))
    
    return {
        'x': x_common,
        'y_levels': y_levels,
        'alphas': alphas,
        'median': median,
        'lower': lower,
        'upper': upper
    }


def plot_density_posterior_band(ax, x, y_samples, weights, color='crimson', 
                                n_levels=20, percentiles=[16, 84], zorder=2,
                                label='1σ posterior'):
    """
    Plot posterior uncertainty band with density-based gradient shading.
    
    This creates a visually appealing uncertainty band where the opacity
    reflects the probability density - darker where more models agree,
    lighter in the tails.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x : np.ndarray
        X-coordinates
    y_samples : np.ndarray
        Array of shape (n_samples, n_points) with y values from different models
    weights : np.ndarray
        Weights for each sample
    color : str
        Base color for the band (default 'crimson')
    n_levels : int
        Number of density levels (more = smoother gradient, default 20)
    percentiles : list
        Percentile bounds (default [16, 84] for 1σ)
    zorder : int
        Plotting order (default 2)
    label : str
        Label for legend (default '1σ posterior')
    
    Returns
    -------
    median_line : matplotlib.lines.Line2D
        The median line artist
    """
    # Compute density information
    density_info = compute_density_at_percentiles(y_samples, weights, x, 
                                                  n_levels=n_levels, 
                                                  percentiles=percentiles)
    
    # Plot density levels from outside in
    for i, ((y_lower, y_upper), alpha) in enumerate(zip(density_info['y_levels'], 
                                                         density_info['alphas'])):
        # Only add label to outermost level
        level_label = label if i == 0 else None
        
        ax.fill_between(x, y_lower, y_upper, 
                       color=color, alpha=alpha, 
                       zorder=zorder, label=level_label,
                       linewidth=0, edgecolor='none')
    
    # Plot median line on top
    median_line = ax.plot(x, density_info['median'], 
                         color=color, lw=2.5, zorder=zorder+3,
                         label='Median model')[0]
    
    return median_line


def plot_density_posterior_band_vertical(ax, y, x_samples, weights, color='crimson',
                                        n_levels=20, percentiles=[16, 84], zorder=2,
                                        label='1σ posterior', orientation='vertical'):
    """
    Plot posterior uncertainty band with density-based shading (vertical orientation).
    
    This is the vertical version for plots where x and y are swapped (e.g., MDF on side panel).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    y : np.ndarray
        Y-coordinates (vertical axis)
    x_samples : np.ndarray
        Array of shape (n_samples, n_points) with x values from different models
    weights : np.ndarray
        Weights for each sample
    color : str
        Base color for the band
    n_levels : int
        Number of density levels
    percentiles : list
        Percentile bounds
    zorder : int
        Plotting order
    label : str
        Label for legend
    orientation : str
        'vertical' or 'horizontal' (default 'vertical')
    
    Returns
    -------
    median_line : matplotlib.lines.Line2D
        The median line artist
    """
    # Compute density information (treating x as y)
    density_info = compute_density_at_percentiles(x_samples, weights, y,
                                                  n_levels=n_levels,
                                                  percentiles=percentiles)
    
    # Plot density levels from outside in
    for i, ((x_lower, x_upper), alpha) in enumerate(zip(density_info['y_levels'],
                                                         density_info['alphas'])):
        level_label = label if i == 0 else None
        
        ax.fill_betweenx(y, x_lower, x_upper,
                        color=color, alpha=alpha,
                        zorder=zorder, label=level_label,
                        linewidth=0, edgecolor='none')
    
    # Plot median line
    median_line = ax.plot(density_info['median'], y,
                         color=color, lw=2.5, zorder=zorder+3,
                         label='Median model')[0]
    
    return median_line


def plot_density_posterior_simple(ax, x, median, lower, upper, color='crimson',
                                  n_levels=20, zorder=2, label='1σ posterior'):
    """
    Plot density-based posterior band from pre-computed percentiles.
    
    This is a simpler version that works with pre-computed median/lower/upper
    arrays instead of the full sample ensemble.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x : np.ndarray
        X-coordinates
    median : np.ndarray
        Median y values
    lower : np.ndarray
        Lower percentile bound
    upper : np.ndarray
        Upper percentile bound
    color : str
        Base color
    n_levels : int
        Number of density levels
    zorder : int
        Plotting order
    label : str
        Label for legend
    
    Returns
    -------
    median_line : matplotlib.lines.Line2D
        The median line artist
    """
    # Create density levels from median outward
    for level in range(n_levels):
        frac = (level + 1) / n_levels
        
        # Interpolate between median and bounds
        y_lower_level = median - frac * (median - lower)
        y_upper_level = median + frac * (upper - median)
        
        # Alpha decreases quadratically from median
        alpha = (1.0 - frac**2) * 0.4
        
        # Only label outermost level
        level_label = label if level == n_levels - 1 else None
        
        ax.fill_between(x, y_lower_level, y_upper_level,
                       color=color, alpha=alpha, zorder=zorder,
                       label=level_label, linewidth=0, edgecolor='none')
    
    # Plot median line on top
    median_line = ax.plot(x, median, color=color, lw=2.5, zorder=zorder+3,
                         label='Median model')[0]
    
    return median_line


def plot_density_posterior_simple_vertical(ax, y, median, lower, upper, color='crimson',
                                          n_levels=20, zorder=2, label='1σ posterior'):
    """
    Plot density-based posterior band (vertical) from pre-computed percentiles.
    
    Vertical version of plot_density_posterior_simple.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    y : np.ndarray
        Y-coordinates (vertical axis)
    median : np.ndarray
        Median x values
    lower : np.ndarray
        Lower percentile bound
    upper : np.ndarray
        Upper percentile bound
    color : str
        Base color
    n_levels : int
        Number of density levels
    zorder : int
        Plotting order
    label : str
        Label for legend
    
    Returns
    -------
    median_line : matplotlib.lines.Line2D
        The median line artist
    """
    # Create density levels from median outward
    for level in range(n_levels):
        frac = (level + 1) / n_levels
        
        # Interpolate between median and bounds
        x_lower_level = median - frac * (median - lower)
        x_upper_level = median + frac * (upper - median)
        
        # Alpha decreases quadratically from median
        alpha = (1.0 - frac**2) * 0.4
        
        # Only label outermost level
        level_label = label if level == n_levels - 1 else None
        
        ax.fill_betweenx(y, x_lower_level, x_upper_level,
                        color=color, alpha=alpha, zorder=zorder,
                        label=level_label, linewidth=0, edgecolor='none')
    
    # Plot median line on top
    median_line = ax.plot(median, y, color=color, lw=2.5, zorder=zorder+3,
                         label='Median model')[0]
    
    return median_line
