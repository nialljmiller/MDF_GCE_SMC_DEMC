"""
Centralized constants for MDF_GCE_SMC_DEMC.

This module is the SINGLE SOURCE OF TRUTH for:
- Parameter names and column mappings
- Display labels (short, full, LaTeX)
- Categorical vs continuous parameter classification
- Index mappings for the GA individual representation
- Default values and color palettes
"""

from typing import Dict, List, Tuple

# =============================================================================
# PARAMETER COLUMNS - Order matters! This is the GA individual structure.
# =============================================================================

# Categorical parameter columns (indices 0-4 in individual)
CATEGORICAL_PARAMS: List[str] = [
    "comp_idx",      # Composition array index
    "imf_idx",       # IMF array index
    "sn1a_idx",      # SNe Ia yield table index
    "sy_idx",        # Stellar yield table index
    "sn1ar_idx",     # SNe Ia rate model index
]

# Continuous parameter columns (indices 5-14 in individual)
CONTINUOUS_PARAMS: List[str] = [
    "sigma_2",       # Second/first infall mass ratio
    "t_1",           # First infall onset time [Gyr]
    "t_2",           # Second infall onset time [Gyr]
    "infall_1",      # First infall timescale [Gyr]
    "infall_2",      # Second infall timescale [Gyr]
    "sfe",           # Star formation efficiency [Gyr^-1]
    "delta_sfe",     # Change in SFE at second infall
    "imf_upper",     # IMF upper mass limit [Msun]
    "mgal",          # Initial gas mass [Msun]
    "nb",            # SNe Ia per solar mass [Msun^-1]
]

# All parameter columns in order
PARAM_COLUMNS: List[str] = CATEGORICAL_PARAMS + CONTINUOUS_PARAMS

# Alternative column names found in some CSV files (for compatibility)
PARAM_COLUMN_ALIASES: Dict[str, str] = {
    "tmax_1": "t_1",
    "tmax_2": "t_2",
    "infall_timescale_1": "infall_1",
    "infall_timescale_2": "infall_2",
    "sfe_val": "sfe",
    "delta_sfe_val": "delta_sfe",
    "imf_upper_limits": "imf_upper",
    "mgal_values": "mgal",
    "nb_array": "nb",
}

# =============================================================================
# INDEX MAPPINGS - For GA individual <-> parameter name conversion
# =============================================================================

INDEX_TO_PARAM_MAP: Dict[int, str] = {
    0: "comp_array",
    1: "imf_array",
    2: "sn1a_assumptions",
    3: "stellar_yield_assumptions",
    4: "sn1a_rates",
    5: "sigma_2",
    6: "tmax_1",
    7: "tmax_2",
    8: "infall_timescale_1",
    9: "infall_timescale_2",
    10: "sfe",
    11: "delta_sfe",
    12: "imf_upper_limits",
    13: "mgal_values",
    14: "nb_array",
}

PARAM_TO_INDEX_MAP: Dict[str, int] = {v: k for k, v in INDEX_TO_PARAM_MAP.items()}

# Indices for slicing
CATEGORICAL_INDICES: List[int] = [0, 1, 2, 3, 4]
CONTINUOUS_INDICES: List[int] = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# =============================================================================
# PARAMETER LABELS - For plotting and display
# =============================================================================

# Short labels for compact displays (corner plots, tables)
PARAM_LABELS_SHORT: Dict[str, str] = {
    "sigma_2":   r"$\sigma_2$",
    "t_1":       r"$t_1$ [Gyr]",
    "t_2":       r"$t_2$ [Gyr]",
    "infall_1":  r"$\tau_1$ [Gyr]",
    "infall_2":  r"$\tau_2$ [Gyr]",
    "sfe":       r"SFE [Gyr$^{-1}$]",
    "delta_sfe": r"$\Delta$SFE [Gyr$^{-1}$]",
    "imf_upper": r"$M_{\max}$ [$M_\odot$]",
    "mgal":      r"$M_{\mathrm{gal}}$ [$M_\odot$]",
    "nb":        r"$N_{\rm Ia}/M_\odot$",
    # Categorical
    "comp_idx":  "Comp",
    "imf_idx":   "IMF",
    "sn1a_idx":  "SNIa",
    "sy_idx":    "Yields",
    "sn1ar_idx": "Rate",
}

# Full labels with complete descriptions
PARAM_LABELS_FULL: Dict[str, str] = {
    "sigma_2":   r"Second/first infall mass ratio $\sigma_2$",
    "t_1":       r"First infall time $t_1$ (Gyr since Big Bang)",
    "t_2":       r"Second infall time $t_2$ (Gyr since Big Bang)",
    "infall_1":  r"First infall timescale $\tau_1$ (Gyr)",
    "infall_2":  r"Second infall timescale $\tau_2$ (Gyr)",
    "sfe":       r"Star formation efficiency SFE (Gyr$^{-1}$)",
    "delta_sfe": r"Change in SFE $\Delta$SFE (Gyr$^{-1}$)",
    "imf_upper": r"IMF upper mass $M_{\max}$ ($M_\odot$)",
    "mgal":      r"Initial bulge gas mass $M_{\mathrm{gal}}$ ($M_\odot$)",
    "nb":        r"SNe Ia per formed mass $N_{\rm Ia}/M_\odot$",
}

# Unicode labels for terminal/console output
PARAM_LABELS_UNICODE: Dict[str, str] = {
    "sigma_2":   "σ₂",
    "t_1":       "t₁",
    "t_2":       "t₂",
    "infall_1":  "τ₁",
    "infall_2":  "τ₂",
    "sfe":       "SFE",
    "delta_sfe": "ΔSFE",
    "imf_upper": "IMF_max",
    "mgal":      "M_gal",
    "nb":        "N_Ia",
}

# Default label set (use short for most plots)
PARAM_LABELS = PARAM_LABELS_SHORT


def get_param_label(name: str, style: str = "short") -> str:
    """
    Get display label for a parameter.
    
    Parameters
    ----------
    name : str
        Internal parameter name (e.g., 'sigma_2')
    style : str
        Label style: 'short', 'full', or 'unicode'
    
    Returns
    -------
    str
        Human-readable label
    """
    label_dicts = {
        "short": PARAM_LABELS_SHORT,
        "full": PARAM_LABELS_FULL,
        "unicode": PARAM_LABELS_UNICODE,
    }
    labels = label_dicts.get(style, PARAM_LABELS_SHORT)
    return labels.get(name, name.replace("_", " ").title())


# =============================================================================
# PARAMETER UNITS
# =============================================================================

PARAM_UNITS: Dict[str, str] = {
    "sigma_2":   "",           # dimensionless
    "t_1":       "Gyr",
    "t_2":       "Gyr",
    "infall_1":  "Gyr",
    "infall_2":  "Gyr",
    "sfe":       "Gyr⁻¹",
    "delta_sfe": "Gyr⁻¹",
    "imf_upper": "M☉",
    "mgal":      "M☉",
    "nb":        "M☉⁻¹",
}

# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

# Parameters that should use log scale on plots
LOG_SCALE_PARAMS: Dict[str, bool] = {
    "sigma_2":   False,
    "t_1":       True,
    "t_2":       False,
    "infall_1":  False,
    "infall_2":  True,
    "sfe":       False,
    "delta_sfe": False,
    "imf_upper": False,
    "mgal":      True,
    "nb":        True,
}

# Color palettes for multi-run comparisons
INK_COLORS: List[str] = [
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

# Categorical parameter display mappings
CATEGORICAL_DISPLAY_NAMES: Dict[str, Dict[int, str]] = {
    "imf_idx": {
        0: "Salpeter",
        1: "Chabrier",
        2: "Kroupa",
    },
    # Add more as needed based on your comp_array, sn1a_assumptions, etc.
}

# =============================================================================
# LOSS METRICS
# =============================================================================

LOSS_COLUMNS: List[str] = [
    "fitness",
    "loss",
    "wrmse",
    "mae",
    "mdf_loss",
    "age_loss",
]

# Available loss metrics for age-metallicity fitting
AGE_LOSS_METRICS: List[str] = [
    "mae",
    "rmse",
    "rms",
    "weighted_mae",
    "weighted_rmse",
    "huber_loss",
    "log_likelihood",
    "aic",
    "bic",
    "correlation",
    "spearman_correlation",
]

# =============================================================================
# FILE PATTERNS
# =============================================================================

RESULTS_CSV_PATTERN: str = "simulation_results*.csv"
WALKER_HISTORY_FILE: str = "walker_history.npz"
CHECKPOINT_FILE: str = "ga_checkpoint.pkl"
POSTERIOR_SAMPLES_FILE: str = "posterior_samples.csv"
CHAINS_FILE: str = "chains.csv"

# =============================================================================
# DEFAULT VALUES
# =============================================================================

DEFAULT_OUTPUT_PATH: str = "SMC_DEMC/"
DEFAULT_PCARD_FILE: str = "bulge_pcard.txt"
DEFAULT_POPULATION_SIZE: int = 96
DEFAULT_NUM_GENERATIONS: int = 100
