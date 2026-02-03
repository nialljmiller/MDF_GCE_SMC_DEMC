"""
MDF_GCE_SMC_DEMC: Metallicity Distribution Function fitting via Genetic Algorithm
with Differential Evolution Markov Chain Monte Carlo refinement for Galactic
Chemical Evolution modeling.

Main components:
- core: GA engine, SMC-DEMC sampler, loss functions, constraints
- analysis: Posterior analysis, uncertainty quantification
- plotting: Visualization tools
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
