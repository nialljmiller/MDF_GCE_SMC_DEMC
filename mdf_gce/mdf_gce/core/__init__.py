"""
Core optimization modules for MDF_GCE_SMC_DEMC.

Components:
- ga: GalacticEvolutionGA class - main genetic algorithm engine
- smc_demc: Sequential Monte Carlo with DE-MC moves
- loss: All loss/fitness functions
- constraints: Physical constraint penalties
- exploration: Voronoi-based sparse region exploration
"""

from .loss import (
    compute_mdf_loss,
    compute_ensemble_loss,
    compute_wrmse,
    compute_age_metallicity_loss,
    compute_combined_loss,
    calculate_all_metrics,
)
from .smc_demc import (
    Bound,
    run_smc_demc,
    de_mh_move,
    effective_sample_size,
    systematic_resample,
)
from .constraints import (
    apply_physics_penalty,
    compute_total_penalty,
)
from .exploration import (
    voronoi_explore_dearths,
    identify_sparse_regions_voronoi,
)

# Lazy imports for GA (requires DEAP)
def __getattr__(name):
    if name == 'GalacticEvolutionGA':
        from .ga import GalacticEvolutionGA
        return GalacticEvolutionGA
    elif name == 'CheckpointManager':
        from .ga import CheckpointManager
        return CheckpointManager
    elif name == 'run_ga_from_config':
        from .ga import run_ga_from_config
        return run_ga_from_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Loss
    'compute_mdf_loss',
    'compute_ensemble_loss',
    'compute_wrmse',
    'compute_age_metallicity_loss',
    'compute_combined_loss',
    'calculate_all_metrics',
    # SMC-DEMC
    'Bound',
    'run_smc_demc',
    'de_mh_move',
    'effective_sample_size',
    'systematic_resample',
    # GA (lazy)
    'GalacticEvolutionGA',
    'CheckpointManager',
    'run_ga_from_config',
    # Constraints
    'apply_physics_penalty',
    'compute_total_penalty',
    # Exploration
    'voronoi_explore_dearths',
    'identify_sparse_regions_voronoi',
]
