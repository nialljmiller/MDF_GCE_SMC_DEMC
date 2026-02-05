"""
Plotting modules for MDF_GCE_SMC_DEMC.

Provides unified visualization tools for:
- MDF comparisons
- Age-metallicity relations
- Alpha element abundances
- Corner plots and posteriors
- Physics evolution (SFR, infall, gas mass)
- Diagnostic plots (loss convergence, walker tracks)

Two modes:
1. Runtime plotting (runtime_plots.py): Uses live data during GA training
2. Post-hoc plotting (paper_plots.py): Loads from saved CSV/NPZ files
"""

from .style import use_paper_style, PLOT_COLORS

# Paper-quality plotting functions (post-hoc from files)
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

# Runtime plotting functions (live data during training)
from .runtime_plots import (
    plot_mdf_runtime,
    plot_amr_runtime,
    plot_alpha_runtime,
    generate_runtime_plots,
    build_curve_dicts,
    get_best_model_idx,
)

__all__ = [
    # Style
    'use_paper_style',
    'PLOT_COLORS',
    # Paper-quality plotting (post-hoc)
    'plot_mdf_posterior',
    'plot_amr_posterior',
    'plot_alpha_posterior',
    'plot_corner_posterior',
    'generate_all_paper_plots',
    # Runtime plotting (live)
    'plot_mdf_runtime',
    'plot_amr_runtime',
    'plot_alpha_runtime',
    'generate_runtime_plots',
    'build_curve_dicts',
    'get_best_model_idx',
    # Utilities
    'compute_posterior_weights',
    'weighted_quantile',
    'build_mdf_ensemble',
    'build_amr_ensemble',
    'build_alpha_ensemble',
    'build_2d_density',
]
