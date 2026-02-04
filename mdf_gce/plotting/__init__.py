"""
Plotting modules for MDF_GCE_SMC_DEMC.

Provides unified visualization tools for:
- MDF comparisons
- Age-metallicity relations
- Alpha element abundances
- Corner plots and posteriors
- Physics evolution (SFR, infall, gas mass)
- Diagnostic plots (loss convergence, walker tracks)
"""

from .style import use_paper_style, PLOT_COLORS

# Paper-quality plotting functions
from .paper_plots import (
    plot_mdf_posterior,
    plot_amr_posterior,
    plot_alpha_posterior,
    plot_corner_posterior,
    generate_all_paper_plots,
    compute_posterior_weights,
    weighted_quantile,
    build_mdf_ensemble,
    build_amr_ensemble,
    build_alpha_ensemble,
    build_2d_density,
)

__all__ = [
    # Style
    'use_paper_style',
    'PLOT_COLORS',
    # Main plotting functions
    'plot_mdf_posterior',
    'plot_amr_posterior',
    'plot_alpha_posterior',
    'plot_corner_posterior',
    'generate_all_paper_plots',
    # Utilities
    'compute_posterior_weights',
    'weighted_quantile',
    'build_mdf_ensemble',
    'build_amr_ensemble',
    'build_alpha_ensemble',
    'build_2d_density',
]
