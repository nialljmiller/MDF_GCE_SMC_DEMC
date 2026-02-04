"""
MDF_GCE_SMC_DEMC: Metallicity Distribution Function fitting via Genetic Algorithm
with Differential Evolution Markov Chain Monte Carlo refinement for Galactic
Chemical Evolution modeling.

Main components:
- core: GA engine, SMC-DEMC sampler, loss functions, constraints
- analysis: Posterior analysis, uncertainty quantification
- plotting: Visualization tools
- io: Results saving and loading
"""

__version__ = "1.0.0"
__author__ = "N. Miller"

from .config import parse_inlist, load_config
from .constants import (
    PARAM_COLUMNS,
    PARAM_LABELS,
    PARAM_LABELS_SHORT,
    PARAM_LABELS_FULL,
    CATEGORICAL_PARAMS,
    CONTINUOUS_PARAMS,
    INDEX_TO_PARAM_MAP,
    LOG_SCALE_PARAMS,
)
from .utils import (
    ensure_dirs,
    find_latest_csv,
    find_highest_gen_file,
    find_result_folders,
    load_results_df,
)

# Lazy import of GA to avoid circular dependencies
def GalacticEvolutionGA(*args, **kwargs):
    """Create a GalacticEvolutionGA instance."""
    from .core.ga import GalacticEvolutionGA as _GA
    return _GA(*args, **kwargs)

def run_ga_from_config(*args, **kwargs):
    """Run GA from configuration dictionary."""
    from .core.ga import run_ga_from_config as _run
    return _run(*args, **kwargs)

# I/O functions (lazy import)
def ResultsLoader(*args, **kwargs):
    """Create a ResultsLoader instance for loading GA results."""
    from .io import ResultsLoader as _RL
    return _RL(*args, **kwargs)

def load_complete_results(*args, **kwargs):
    """Load complete GA results including curves."""
    from .io import load_complete_results as _load
    return _load(*args, **kwargs)

def save_complete_results(*args, **kwargs):
    """Save complete GA results including curves."""
    from .io import save_complete_results as _save
    return _save(*args, **kwargs)
