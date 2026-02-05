"""
Matplotlib style configuration for MDF_GCE_SMC_DEMC.

Provides consistent, publication-quality plot styling across all visualizations.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/batch use
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# COLOR PALETTES
# =============================================================================

# Primary color palette for multi-run comparisons
PLOT_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

# Sequential colormap for density/posterior plots
POSTERIOR_CMAP = "Greys"
DENSITY_CMAP = "viridis"

# Specific colors for key plot elements
COLOR_BEST_MODEL = "#d62728"      # red
COLOR_MEDIAN = "#1f77b4"          # blue
COLOR_OBSERVATION = "#2ca02c"     # green
COLOR_UNCERTAINTY = "#1f77b4"     # blue with alpha
COLOR_SECONDARY = "#ff7f0e"       # orange

# Aliases for convenience
COLOR_BEST = COLOR_BEST_MODEL
COLOR_OBS = COLOR_OBSERVATION
COLOR_JOYCE = 'red'               # Joyce+23 data
COLOR_BENSBY = 'blue'             # Bensby+17 data

# Alpha values
ALPHA_BAND = 0.25
ALPHA_SCATTER = 0.6
ALPHA_SECONDARY = 0.3


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

def use_paper_style():
    """
    Apply publication-quality matplotlib styling.
    
    Call this at the start of any script that generates figures.
    """
    plt.rcParams.update({
        # Figure settings
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.figsize": (8, 6),
        "figure.facecolor": "white",
        
        # Font settings
        "font.family": "serif",
        "font.size": 12,
        "mathtext.fontset": "dejavuserif",
        
        # Axes settings
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "axes.axisbelow": True,
        
        # Tick settings
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        
        # Legend settings
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "lightgray",
        
        # Line settings
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        
        # Patch settings (for histograms, etc.)
        "patch.edgecolor": "black",
        "patch.linewidth": 0.5,
        
        # Save settings
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def use_minimal_style():
    """
    Apply minimal styling for quick diagnostic plots.
    """
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "lines.linewidth": 1.0,
    })


def use_presentation_style():
    """
    Apply larger fonts suitable for presentations.
    """
    use_paper_style()
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "lines.linewidth": 2.0,
    })


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_figure(fig, path: str, close: bool = True):
    """
    Save figure with proper directory creation.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    path : str
        Output path
    close : bool
        If True, close the figure after saving
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    if close:
        plt.close(fig)


def get_colors(n: int, palette: list = None) -> list:
    """
    Get n colors from a palette, cycling if necessary.
    
    Parameters
    ----------
    n : int
        Number of colors needed
    palette : list, optional
        Color palette. Default: PLOT_COLORS
        
    Returns
    -------
    list
        List of n colors
    """
    if palette is None:
        palette = PLOT_COLORS
    
    return [palette[i % len(palette)] for i in range(n)]


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """
    Create a truncated colormap.
    
    Parameters
    ----------
    cmap : str or Colormap
        Original colormap
    minval, maxval : float
        Range to use from original colormap
    n : int
        Number of colors in new colormap
        
    Returns
    -------
    LinearSegmentedColormap
        Truncated colormap
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


def add_colorbar(fig, ax, mappable, label: str = "", orientation: str = "vertical"):
    """
    Add a colorbar to an axes with consistent styling.
    
    Parameters
    ----------
    fig : Figure
        Parent figure
    ax : Axes
        Parent axes
    mappable : ScalarMappable
        The image/contour to make colorbar for
    label : str
        Colorbar label
    orientation : str
        'vertical' or 'horizontal'
        
    Returns
    -------
    Colorbar
        The created colorbar
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    divider = make_axes_locatable(ax)
    if orientation == "vertical":
        cax = divider.append_axes("right", size="5%", pad=0.1)
    else:
        cax = divider.append_axes("bottom", size="5%", pad=0.3)
    
    cbar = fig.colorbar(mappable, cax=cax, orientation=orientation)
    if label:
        cbar.set_label(label)
    
    return cbar


def set_axis_limits_with_padding(ax, xdata, ydata, pad_frac: float = 0.05):
    """
    Set axis limits with fractional padding.
    
    Parameters
    ----------
    ax : Axes
        Axes to modify
    xdata, ydata : array-like
        Data to determine limits from
    pad_frac : float
        Fractional padding to add
    """
    xmin, xmax = np.nanmin(xdata), np.nanmax(xdata)
    ymin, ymax = np.nanmin(ydata), np.nanmax(ydata)
    
    xpad = (xmax - xmin) * pad_frac
    ypad = (ymax - ymin) * pad_frac
    
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)


# =============================================================================
# INITIALIZE DEFAULT STYLE
# =============================================================================

# Apply paper style by default when module is imported
use_paper_style()
