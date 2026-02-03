#!/usr/bin/env python3
"""
Galactic Chemical Evolution Genetic Algorithm Engine.

This module contains the GalacticEvolutionGA class that implements
a hybrid GA + DE-MC optimization for fitting GCE models to observations.
"""

import gc
import os
import random
import time
from multiprocessing import Pool, cpu_count, get_context
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from deap import base, creator, tools

# Import from new package structure
from mdf_gce.constants import (
    PARAM_COLUMNS,
    PARAM_LABELS_SHORT,
    CATEGORICAL_PARAMS,
    CONTINUOUS_PARAMS,
)
from mdf_gce.utils import ensure_dirs, alloc_cores
from mdf_gce.core.smc_demc import Bound, run_smc_demc, de_mh_move
from mdf_gce.core.loss import (
    compute_ensemble_loss,
    compute_wrmse,
    compute_age_metallicity_loss,
)
from mdf_gce.core.constraints import apply_physics_penalty
from mdf_gce.core.exploration import voronoi_explore_dearths

# JINAPyCEE import - expected in parent directory or installed
try:
    from JINAPyCEE import omega_plus
except ImportError:
    omega_plus = None
    print("Warning: JINAPyCEE not found. Model evaluation will fail.")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log_uniform(min_val: float, max_val: float) -> float:
    """Sample uniformly in log space."""
    log_min = np.log10(min_val)
    log_max = np.log10(max_val)
    return 10 ** random.uniform(log_min, log_max)


def should_use_log(min_val: float, max_val: float, threshold: float = 2.0) -> bool:
    """Check if parameter spans more than threshold orders of magnitude."""
    if min_val <= 0 or max_val <= 0:
        return False
    return np.log10(max_val / min_val) >= threshold


def find_nearest(array: np.ndarray, value: float) -> Tuple[int, float]:
    """Find index and value of nearest element in array."""
    idx = np.abs(array - value).argmin()
    return idx, array[idx]


def _summarize(name: str, x: Any) -> str:
    """Return multi-line string summarizing object x."""
    if isinstance(x, (str, bytes)):
        return f"{name}: {repr(x)}"
    try:
        n = len(x)
        if hasattr(x, 'dtype'):
            return f"{name}: len={n}, dtype={x.dtype}, range=[{np.min(x):.4g}, {np.max(x):.4g}]"
        return f"{name}: len={n}, values={list(x)[:5]}{'...' if n > 5 else ''}"
    except TypeError:
        return f"{name}: {repr(x)}"


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    """Checkpoint manager for GA runs."""
    
    def __init__(self, save_path: str = 'SMC_DEMC/'):
        self.filename = os.path.join(save_path, 'ga_checkpoint.pkl')
    
    def save(self, generation: int, population: List, ga_instance: Any) -> None:
        """Save checkpoint."""
        import pickle
        data = {
            'generation': generation,
            'population': population,
            'ga_state': {k: v for k, v in ga_instance.__dict__.items() 
                        if not k.startswith('_') and not callable(v)},
        }
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        with open(self.filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Checkpoint saved at generation {generation}")
    
    def load(self) -> Optional[Dict]:
        """Load checkpoint if exists."""
        import pickle
        if not os.path.exists(self.filename):
            return None
        with open(self.filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded checkpoint from generation {data['generation']}")
        return data
    
    def clear(self) -> None:
        """Clear checkpoint file."""
        if os.path.exists(self.filename):
            print(f"Checkpoint file {self.filename} preserved.")


# =============================================================================
# MAIN GA CLASS
# =============================================================================

class GalacticEvolutionGA:
    """
    Genetic Algorithm for Galactic Chemical Evolution model optimization.
    
    Implements a hybrid GA + DE-MC approach with:
    - Tournament selection
    - Fitness-weighted crossover
    - Adaptive Gaussian mutation
    - Optional Voronoi-based sparse region exploration
    - DE-MC moves during each generation
    - Optional SMC-DEMC posterior refinement
    """
    
    def __init__(
        self,
        output_path: str,
        sn1a_header: str,
        iniab_header: str,
        sigma_2_list: List[float],
        tmax_1_list: List[float],
        tmax_2_list: List[float],
        infall_timescale_1_list: List[float],
        infall_timescale_2_list: List[float],
        comp_array: List[str],
        imf_array: List[str],
        sfe_array: List[float],
        delta_sfe_array: List[float],
        imf_upper_limits: List[float],
        sn1a_assumptions: List[str],
        stellar_yield_assumptions: List[str],
        mgal_values: List[float],
        nb_array: List[float],
        sn1a_rates: List[str],
        timesteps: int,
        A1: float,
        A2: float,
        feh: np.ndarray,
        normalized_count: np.ndarray,
        obs_age_data: Optional[Dict] = None,
        loss_metric: str = 'ensemble',
        obs_age_data_loss_metric: str = 'None',
        obs_age_data_target: str = 'joyce',
        mdf_vs_age_weight: float = 1.0,
        fancy_mutation: str = 'gaussian',
        shrink_range: bool = False,
        tournament_size: int = 3,
        lambda_diversity: float = 0.01,
        threshold: float = -1,
        cxpb: float = 0.5,
        mutpb: float = 0.5,
        gaussian_sigma_scale: float = 0.01,
        crossover_noise_fraction: float = 0.05,
        perturbation_strength: float = 0.1,
        physical_constraints_freq: int = 10,
        exploration_steps: int = 0,
        PP: bool = False,
        demc_hybrid: bool = True,
        demc_fraction: float = 0.5,
        demc_moves_per_gen: int = 1,
        demc_gamma: Optional[float] = None,
        demc_rng_seed: Optional[int] = None,
        demc_workers: Optional[int] = None,
        plot_mode: str = "full",
    ):
        """Initialize the GA with all parameters."""
        
        # Output path
        self.output_path = output_path.rstrip('/') + '/'
        ensure_dirs(self.output_path, ['plots', 'checkpoints'])
        
        # Model configuration headers
        self.sn1a_header = sn1a_header
        self.iniab_header = iniab_header
        
        # Continuous parameter ranges (stored as [min, max])
        self.sigma_2_list = sigma_2_list
        self.tmax_1_list = tmax_1_list
        self.tmax_2_list = tmax_2_list
        self.infall_timescale_1_list = infall_timescale_1_list
        self.infall_timescale_2_list = infall_timescale_2_list
        self.sfe_array = sfe_array
        self.delta_sfe_array = delta_sfe_array
        self.imf_upper_limits = imf_upper_limits
        self.mgal_values = mgal_values
        self.nb_array = nb_array
        
        # Categorical parameter arrays
        self.comp_array = comp_array
        self.imf_array = imf_array
        self.sn1a_assumptions = sn1a_assumptions
        self.stellar_yield_assumptions = stellar_yield_assumptions
        self.sn1a_rates = sn1a_rates
        
        # Simulation settings
        self.timesteps = timesteps
        self.A1 = A1
        self.A2 = A2
        
        # Observational data
        self.feh = feh
        self.normalized_count = normalized_count
        self.obs_age_data = obs_age_data
        
        # Loss function settings
        self.loss_metric = loss_metric
        self.obs_age_data_loss_metric = obs_age_data_loss_metric
        self.obs_age_data_target = obs_age_data_target
        self.mdf_vs_age_weight = mdf_vs_age_weight
        
        # GA settings
        self.fancy_mutation = fancy_mutation
        self.shrink_range = shrink_range
        self.tournament_size = tournament_size
        self.lambda_diversity = lambda_diversity
        self.threshold = threshold
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.gaussian_sigma_scale = gaussian_sigma_scale
        self.crossover_noise_fraction = crossover_noise_fraction
        self.perturbation_strength = perturbation_strength
        self.physical_constraints_freq = physical_constraints_freq
        self.exploration_steps = exploration_steps
        
        # Parallelization
        self.PP = PP
        
        # DE-MC settings
        self.demc_hybrid = demc_hybrid
        self.demc_fraction = demc_fraction
        self.demc_moves_per_gen = demc_moves_per_gen
        self.demc_gamma = demc_gamma
        self.demc_rng_seed = demc_rng_seed
        self.demc_workers = demc_workers
        
        # Plotting
        self.plot_mode = plot_mode
        
        # Internal state
        self.results = []
        self.evaluation_results = []
        self.walker_history = {}
        self.gen = 0
        self.num_generations = 0
        self.physics_timer = 0
        
        # Parameter indices
        self.categorical_indices = [0, 1, 2, 3, 4]
        self.continuous_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        
        # Index to parameter name mapping
        self.index_to_param_map = {
            0: 'comp_array',
            1: 'imf_array',
            2: 'sn1a_assumptions',
            3: 'stellar_yield_assumptions',
            4: 'sn1a_rates',
            5: 'sigma_2',
            6: 'tmax_1',
            7: 'tmax_2',
            8: 'infall_timescale_1',
            9: 'infall_timescale_2',
            10: 'sfe',
            11: 'delta_sfe',
            12: 'imf_upper_limits',
            13: 'mgal_values',
            14: 'nb_array'
        }
        
        # Compute bounds
        self._compute_bounds()
        
        # Print configuration
        self._print_config()
    
    def _compute_bounds(self) -> None:
        """Compute parameter bounds for continuous parameters."""
        self.sigma_2_min, self.sigma_2_max = min(self.sigma_2_list), max(self.sigma_2_list)
        self.t_1_min, self.t_1_max = min(self.tmax_1_list), max(self.tmax_1_list)
        self.t_2_min, self.t_2_max = min(self.tmax_2_list), max(self.tmax_2_list)
        self.infall_1_min, self.infall_1_max = min(self.infall_timescale_1_list), max(self.infall_timescale_1_list)
        self.infall_2_min, self.infall_2_max = min(self.infall_timescale_2_list), max(self.infall_timescale_2_list)
        self.sfe_min, self.sfe_max = min(self.sfe_array), max(self.sfe_array)
        self.delta_sfe_min, self.delta_sfe_max = min(self.delta_sfe_array), max(self.delta_sfe_array)
        self.imf_upper_min, self.imf_upper_max = min(self.imf_upper_limits), max(self.imf_upper_limits)
        self.mgal_min, self.mgal_max = min(self.mgal_values), max(self.mgal_values)
        self.nb_min, self.nb_max = min(self.nb_array), max(self.nb_array)
    
    def _print_config(self) -> None:
        """Print configuration summary."""
        print("\n" + "=" * 70)
        print("GALACTIC EVOLUTION GA CONFIG")
        print("=" * 70)
        
        print("\nOUTPUT")
        print(f"  output_path: {self.output_path}")
        
        print("\nCATEGORICAL PARAMETERS")
        print(f"  comp_array: {len(self.comp_array)} options")
        print(f"  imf_array: {len(self.imf_array)} options")
        print(f"  sn1a_assumptions: {len(self.sn1a_assumptions)} options")
        print(f"  stellar_yield_assumptions: {len(self.stellar_yield_assumptions)} options")
        print(f"  sn1a_rates: {len(self.sn1a_rates)} options")
        
        print("\nCONTINUOUS PARAMETERS")
        print(f"  sigma_2: [{self.sigma_2_min:.4g}, {self.sigma_2_max:.4g}]")
        print(f"  t_1 (Gyr): [{self.t_1_min:.4g}, {self.t_1_max:.4g}]")
        print(f"  t_2 (Gyr): [{self.t_2_min:.4g}, {self.t_2_max:.4g}]")
        print(f"  infall_1 (Gyr): [{self.infall_1_min:.4g}, {self.infall_1_max:.4g}]")
        print(f"  infall_2 (Gyr): [{self.infall_2_min:.4g}, {self.infall_2_max:.4g}]")
        print(f"  sfe: [{self.sfe_min:.4g}, {self.sfe_max:.4g}]")
        print(f"  delta_sfe: [{self.delta_sfe_min:.4g}, {self.delta_sfe_max:.4g}]")
        print(f"  imf_upper: [{self.imf_upper_min:.4g}, {self.imf_upper_max:.4g}]")
        print(f"  mgal: [{self.mgal_min:.4g}, {self.mgal_max:.4g}]")
        print(f"  nb: [{self.nb_min:.4g}, {self.nb_max:.4g}]")
        
        print("\nOBSERVATIONAL DATA")
        print(f"  MDF bins: {len(self.feh)}")
        print(f"  [Fe/H] range: [{self.feh.min():.2f}, {self.feh.max():.2f}]")
        
        print("\nGA SETTINGS")
        print(f"  loss_metric: {self.loss_metric}")
        print(f"  mutation: {self.fancy_mutation}")
        print(f"  tournament_size: {self.tournament_size}")
        print(f"  cxpb: {self.cxpb}, mutpb: {self.mutpb}")
        print(f"  demc_hybrid: {self.demc_hybrid}")
        print(f"  demc_fraction: {self.demc_fraction}")
        
        n_cat = (len(self.comp_array) * len(self.imf_array) * 
                 len(self.sn1a_assumptions) * len(self.stellar_yield_assumptions) *
                 len(self.sn1a_rates))
        print(f"\nPARAMETER SPACE: {n_cat:,} categorical × 10 continuous dims")
        print("=" * 70 + "\n")
    
    def get_param_bounds(self, index: int) -> Tuple[float, float]:
        """Get bounds for parameter at given index."""
        bounds_map = {
            5: (self.sigma_2_min, self.sigma_2_max),
            6: (self.t_1_min, self.t_1_max),
            7: (self.t_2_min, self.t_2_max),
            8: (self.infall_1_min, self.infall_1_max),
            9: (self.infall_2_min, self.infall_2_max),
            10: (self.sfe_min, self.sfe_max),
            11: (self.delta_sfe_min, self.delta_sfe_max),
            12: (self.imf_upper_min, self.imf_upper_max),
            13: (self.mgal_min, self.mgal_max),
            14: (self.nb_min, self.nb_max),
        }
        return bounds_map.get(index, (0.0, 1.0))
    
    def _reflect_at_bounds(self, value: float, lo: float, hi: float) -> float:
        """Reflect value at boundaries."""
        if lo >= hi:
            return lo
        while value < lo or value > hi:
            if value < lo:
                value = lo + (lo - value)
            if value > hi:
                value = hi - (value - hi)
        return value
    
    # =========================================================================
    # DEAP TOOLBOX INITIALIZATION
    # =========================================================================
    
    def init_GenAl(self, population_size: int) -> Tuple[List, base.Toolbox]:
        """
        Initialize DEAP toolbox and create initial population.
        
        Returns:
            (population, toolbox) tuple
        """
        # Create fitness and individual types
        if not hasattr(creator, 'FitnessMin'):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        
        # Categorical attribute generators
        toolbox.register("comp_attr", lambda: random.randint(0, len(self.comp_array) - 1))
        toolbox.register("imf_attr", lambda: random.randint(0, len(self.imf_array) - 1))
        toolbox.register("sn1a_attr", lambda: random.randint(0, len(self.sn1a_assumptions) - 1))
        toolbox.register("sy_attr", lambda: random.randint(0, len(self.stellar_yield_assumptions) - 1))
        toolbox.register("sn1a_rate_attr", lambda: random.randint(0, len(self.sn1a_rates) - 1))
        
        # Continuous attribute generators (use log scale where appropriate)
        toolbox.register("sigma_2_attr", log_uniform, self.sigma_2_min, self.sigma_2_max)
        
        if should_use_log(self.t_1_min, self.t_1_max):
            toolbox.register("t_1_attr", log_uniform, self.t_1_min, self.t_1_max)
        else:
            toolbox.register("t_1_attr", random.uniform, self.t_1_min, self.t_1_max)
        
        if should_use_log(self.t_2_min, self.t_2_max):
            toolbox.register("t_2_attr", log_uniform, self.t_2_min, self.t_2_max)
        else:
            toolbox.register("t_2_attr", random.uniform, self.t_2_min, self.t_2_max)
        
        if should_use_log(self.infall_1_min, self.infall_1_max):
            toolbox.register("infall_1_attr", log_uniform, self.infall_1_min, self.infall_1_max)
        else:
            toolbox.register("infall_1_attr", random.uniform, self.infall_1_min, self.infall_1_max)
        
        if should_use_log(self.infall_2_min, self.infall_2_max):
            toolbox.register("infall_2_attr", log_uniform, self.infall_2_min, self.infall_2_max)
        else:
            toolbox.register("infall_2_attr", random.uniform, self.infall_2_min, self.infall_2_max)
        
        toolbox.register("sfe_attr", random.uniform, self.sfe_min, self.sfe_max)
        toolbox.register("delta_sfe_attr", random.uniform, self.delta_sfe_min, self.delta_sfe_max)
        toolbox.register("imf_upper_attr", random.uniform, self.imf_upper_min, self.imf_upper_max)
        
        if should_use_log(self.mgal_min, self.mgal_max):
            toolbox.register("mgal_attr", log_uniform, self.mgal_min, self.mgal_max)
        else:
            toolbox.register("mgal_attr", random.uniform, self.mgal_min, self.mgal_max)
        
        if should_use_log(self.nb_min, self.nb_max):
            toolbox.register("nb_attr", log_uniform, self.nb_min, self.nb_max)
        else:
            toolbox.register("nb_attr", random.uniform, self.nb_min, self.nb_max)
        
        # Individual: combine all 15 attributes
        toolbox.register(
            "individual", 
            tools.initCycle, 
            creator.Individual,
            (toolbox.comp_attr, toolbox.imf_attr, toolbox.sn1a_attr,
             toolbox.sy_attr, toolbox.sn1a_rate_attr,
             toolbox.sigma_2_attr, toolbox.t_1_attr, toolbox.t_2_attr,
             toolbox.infall_1_attr, toolbox.infall_2_attr,
             toolbox.sfe_attr, toolbox.delta_sfe_attr, toolbox.imf_upper_attr,
             toolbox.mgal_attr, toolbox.nb_attr),
            n=1
        )
        
        # Population
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Genetic operators
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", self.crossover, max_bias=0.55)
        
        if self.fancy_mutation.lower() == 'gaussian':
            toolbox.register("mutate", lambda ind: self.gaussian_mutate(ind, self.gaussian_sigma_scale))
        else:
            toolbox.register("mutate", self.uniform_mutate)
        
        toolbox.register("select", self.selTournament, tournsize=self.tournament_size)
        
        # DEMC workers
        if self.demc_workers is None:
            self.demc_workers = population_size
        
        # Create population
        population = toolbox.population(n=population_size)
        
        print(f"Initialized population: {population_size} individuals")
        print(f"DE-MC workers: {self.demc_workers}")
        
        return population, toolbox
    
    # =========================================================================
    # GENETIC OPERATORS
    # =========================================================================
    
    def selTournament(self, individuals: List, k: int = None, tournsize: int = 3) -> List:
        """Tournament selection."""
        if k is None:
            k = len(individuals)
        selected = []
        for _ in range(k):
            aspirants = random.sample(individuals, min(tournsize, len(individuals)))
            selected.append(min(aspirants, key=lambda x: x.fitness.values[0] if x.fitness.valid else float('inf')))
        return selected
    
    def get_fitness_scale(self, individual) -> float:
        """Calculate fitness-based scaling factor for mutation."""
        if not individual.fitness.valid:
            return 1.0
        fitness = individual.fitness.values[0]
        # Higher fitness (worse) -> larger mutations
        if fitness < 0.1:
            return 0.5
        elif fitness < 1.0:
            return 0.7
        elif fitness < 10.0:
            return 1.0
        else:
            return 1.5
    
    def crossover(self, ind1: List, ind2: List, max_bias: float = 0.55) -> Tuple[List, List]:
        """
        Fitness-weighted crossover.
        
        Categorical parameters: inherit from fitter parent with some probability.
        Continuous parameters: weighted average with noise.
        """
        # Get fitness values
        f1 = ind1.fitness.values[0] if ind1.fitness.valid else float('inf')
        f2 = ind2.fitness.values[0] if ind2.fitness.valid else float('inf')
        
        # Compute weights (lower fitness is better)
        if f1 + f2 > 0:
            w1 = f2 / (f1 + f2)  # Higher weight to fitter parent
            w2 = f1 / (f1 + f2)
        else:
            w1 = w2 = 0.5
        
        # Limit bias
        w1 = min(max_bias, max(1 - max_bias, w1))
        w2 = 1 - w1
        
        child1 = creator.Individual(ind1[:])
        child2 = creator.Individual(ind2[:])
        
        # Categorical parameters: probabilistic inheritance
        for i in self.categorical_indices:
            if random.random() < w1:
                child1[i] = ind1[i]
                child2[i] = ind2[i]
            else:
                child1[i] = ind2[i]
                child2[i] = ind1[i]
        
        # Continuous parameters: weighted blend with noise
        for i in self.continuous_indices:
            noise_scale = self.crossover_noise_fraction * abs(ind1[i] - ind2[i])
            
            child1[i] = w1 * ind1[i] + w2 * ind2[i] + random.gauss(0, noise_scale + 1e-10)
            child2[i] = w2 * ind1[i] + w1 * ind2[i] + random.gauss(0, noise_scale + 1e-10)
            
            # Ensure bounds
            lo, hi = self.get_param_bounds(i)
            child1[i] = self._reflect_at_bounds(child1[i], lo, hi)
            child2[i] = self._reflect_at_bounds(child2[i], lo, hi)
        
        return child1, child2
    
    def gaussian_mutate(self, individual: List, base_sigma_scale: float = 0.01, indpb: float = 0.3):
        """
        Adaptive Gaussian mutation.
        
        Mutation strength adapts based on generation progress and fitness.
        """
        current_values = individual[:]
        fitness_scale = self.get_fitness_scale(individual)
        
        for i in range(len(individual)):
            if random.random() < indpb:
                if i in self.categorical_indices:
                    # Small chance to flip categorical
                    if random.random() < 0.1:
                        param_name = self.index_to_param_map[i]
                        arr = getattr(self, param_name, [0])
                        individual[i] = random.randint(0, max(0, len(arr) - 1))
                else:
                    lo, hi = self.get_param_bounds(i)
                    range_size = hi - lo
                    
                    # Adaptive step size
                    if hasattr(self, 'gen') and hasattr(self, 'num_generations') and self.num_generations > 0:
                        progress = self.gen / self.num_generations
                        scale = base_sigma_scale * (1 - 0.5 * progress)
                    else:
                        scale = base_sigma_scale
                    
                    scale = max(0.3 * base_sigma_scale, scale)
                    step_mult = 0.5 + 0.5 * random.random()
                    sigma = range_size * scale * step_mult * fitness_scale
                    
                    new_value = individual[i] + random.gauss(0, sigma)
                    individual[i] = self._reflect_at_bounds(new_value, lo, hi)
        
        return individual,
    
    def uniform_mutate(self, individual: List, indpb: float = 0.3):
        """Uniform mutation."""
        for i in range(len(individual)):
            if random.random() < indpb:
                if i in self.categorical_indices:
                    if random.random() < 0.1:
                        param_name = self.index_to_param_map[i]
                        arr = getattr(self, param_name, [0])
                        individual[i] = random.randint(0, max(0, len(arr) - 1))
                else:
                    lo, hi = self.get_param_bounds(i)
                    individual[i] = random.uniform(lo, hi)
        return individual,
    
    def prevent_duplicates(self, population: List, toolbox) -> List:
        """Remove duplicate individuals by adding small perturbations."""
        seen = set()
        unique = []
        
        for ind in population:
            key = tuple(round(x, 6) if isinstance(x, float) else x for x in ind)
            if key in seen:
                # Perturb
                new_ind = toolbox.clone(ind)
                for i in self.continuous_indices:
                    lo, hi = self.get_param_bounds(i)
                    new_ind[i] += random.gauss(0, (hi - lo) * 0.001)
                    new_ind[i] = self._reflect_at_bounds(new_ind[i], lo, hi)
                if hasattr(new_ind.fitness, 'values'):
                    del new_ind.fitness.values
                unique.append(new_ind)
            else:
                seen.add(key)
                unique.append(ind)
        
        return unique
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    
    def evaluate(self, individual: List) -> Tuple[Tuple[float], Dict]:
        """
        Evaluate an individual by running omega_plus GCE model.
        
        Returns:
            ((fitness,), result_dict) tuple
        """
        if omega_plus is None:
            raise RuntimeError("JINAPyCEE omega_plus not available")
        
        # Repair individual (enforce constraints)
        self._repair_individual(individual)
        
        # Extract parameters
        comp_idx = int(np.clip(int(individual[0]), 0, len(self.comp_array) - 1))
        imf_idx = int(np.clip(int(individual[1]), 0, len(self.imf_array) - 1))
        sn1a_idx = int(np.clip(int(individual[2]), 0, len(self.sn1a_assumptions) - 1))
        sy_idx = int(np.clip(int(individual[3]), 0, len(self.stellar_yield_assumptions) - 1))
        sn1ar_idx = int(np.clip(int(individual[4]), 0, len(self.sn1a_rates) - 1))
        
        comp = self.comp_array[comp_idx]
        imfval = self.imf_array[imf_idx]
        sn1a = self.sn1a_assumptions[sn1a_idx]
        sy = self.stellar_yield_assumptions[sy_idx]
        sn1ar = self.sn1a_rates[sn1ar_idx]
        
        sigma_2 = individual[5]
        t_1 = individual[6]
        t_2 = individual[7]
        infall_1 = individual[8]
        infall_2 = individual[9]
        sfe_val = individual[10]
        delta_sfe = individual[11]
        imf_upper = individual[12]
        mgal = individual[13]
        nb = individual[14]
        
        # Run model with retries for mass constraint
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                kwargs = {
                    'special_timesteps': self.timesteps,
                    'twoinfall_sigmas': [1300, sigma_2],
                    'galradius': 1800,
                    'exp_infall': [[-1, t_1*1e9, infall_1*1e9], [-1, t_2*1e9, infall_2*1e9]],
                    'substeps': [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
                    'tolerance': 1e-5,
                    'tauup': [0.1*infall_1*1e9, 0.1*infall_2*1e9],
                    'mgal': mgal,
                    'iniZ': 0.0,
                    'mass_loading': 0.0,
                    'table': self.sn1a_header + sy,
                    'sfe': sfe_val,
                    'delta_sfe': delta_sfe,
                    't_star': 1.0e9,
                    'imf_type': imfval,
                    'sn1a_table': self.sn1a_header + sn1a,
                    'imf_yields_range': [1, imf_upper],
                    'iniabu_table': self.iniab_header + comp,
                    'nb_1a_per_m': nb,
                    'sn1a_rate': sn1ar,
                }
                
                GCE_model = omega_plus.omega_plus(**kwargs)
                
                # Check mass constraint
                m_gas = sum(GCE_model.inner.ymgal[-1])
                m_locked = sum(GCE_model.inner.history.m_locked)
                tot_mass = m_gas + m_locked
                
                if 5e9 < tot_mass < 3e10:
                    break  # Valid model
                
                if attempt < max_retries:
                    # Perturb and retry
                    self._perturb_for_mass(individual, attempt + 1)
                    sigma_2 = individual[5]
                    t_1 = individual[6]
                    t_2 = individual[7]
                    infall_1 = individual[8]
                    infall_2 = individual[9]
                    sfe_val = individual[10]
                    mgal = individual[13]
                    
            except Exception as e:
                if attempt == max_retries:
                    # Return high penalty
                    result = {
                        'individual': list(individual),
                        'fitness': 1000.0,
                        'error': str(e),
                    }
                    return (1000.0,), result
                self._perturb_for_mass(individual, attempt + 1)
        
        # Extract MDF
        try:
            MDF_x, MDF_y = GCE_model.inner.plot_mdf(
                axis_mdf='[Fe/H]', 
                sigma_gauss=0.1, 
                norm=True, 
                return_x_y=True
            )
        except Exception:
            return (1000.0,), {'individual': list(individual), 'fitness': 1000.0}
        
        # Interpolate to observation grid
        model_mdf = np.interp(self.feh, MDF_x, MDF_y, left=0.0, right=0.0)
        model_mdf = model_mdf / (model_mdf.sum() + 1e-10)
        
        # Compute loss
        obs_sigma = np.sqrt(self.normalized_count + 0.01)
        fitness = compute_ensemble_loss(self.normalized_count, model_mdf, obs_sigma)
        
        # Extract additional data for storage
        elements = ['[Si/Fe]', '[Ca/Fe]', '[Mg/Fe]', '[Ti/Fe]']
        alpha_data = {}
        for el in elements:
            try:
                x, y = GCE_model.inner.plot_spectro(
                    xaxis='[Fe/H]', yaxis=el, return_x_y=True
                )
                alpha_data[el] = (np.array(x), np.array(y))
            except:
                pass
        
        # Age-metallicity
        try:
            age_x = np.array(GCE_model.inner.history.age)
            age_y = np.array(GCE_model.inner.history.metallicity)
        except:
            age_x, age_y = np.array([]), np.array([])
        
        # Build result dictionary
        result = {
            'individual': list(individual),
            'fitness': float(fitness),
            'mdf_x': MDF_x,
            'mdf_y': MDF_y,
            'alpha_data': alpha_data,
            'age_x': age_x,
            'age_y': age_y,
            'total_mass': tot_mass if 'tot_mass' in dir() else None,
        }
        
        return (float(fitness),), result
    
    def _repair_individual(self, individual: List) -> None:
        """Enforce physical constraints on individual."""
        # t_2 >= t_1 + 0.5
        if individual[7] < individual[6] + 0.5:
            individual[7] = individual[6] + 0.5
        
        # Clip categorical indices
        individual[0] = int(np.clip(int(individual[0]), 0, len(self.comp_array) - 1))
        individual[1] = int(np.clip(int(individual[1]), 0, len(self.imf_array) - 1))
        individual[2] = int(np.clip(int(individual[2]), 0, len(self.sn1a_assumptions) - 1))
        individual[3] = int(np.clip(int(individual[3]), 0, len(self.stellar_yield_assumptions) - 1))
        individual[4] = int(np.clip(int(individual[4]), 0, len(self.sn1a_rates) - 1))
        
        # Clip continuous to bounds
        for i in self.continuous_indices:
            lo, hi = self.get_param_bounds(i)
            individual[i] = max(lo, min(hi, individual[i]))
    
    def _perturb_for_mass(self, individual: List, attempt: int) -> None:
        """Perturb individual to try to satisfy mass constraint."""
        scale = 0.1 * attempt
        for i in [5, 10, 13]:  # sigma_2, sfe, mgal
            lo, hi = self.get_param_bounds(i)
            individual[i] += random.gauss(0, (hi - lo) * scale)
            individual[i] = self._reflect_at_bounds(individual[i], lo, hi)
    
    def _record_evaluation_result(self, result: Dict) -> None:
        """Record evaluation result."""
        self.evaluation_results.append(result)
    
    # =========================================================================
    # MAIN GA LOOP
    # =========================================================================
    
    def GenAl(
        self,
        population_size: int,
        num_generations: int,
        population: List,
        toolbox: base.Toolbox,
        checkpoint_manager: Optional[CheckpointManager] = None,
        start_gen: int = 0,
        output_interval: Optional[int] = None,
    ) -> None:
        """
        Main GA loop with DE-MC hybrid moves.
        
        After GA generations, optionally runs SMC-DEMC refinement.
        """
        import multiprocessing as mp
        
        total_start = time.time()
        self.num_generations = num_generations
        
        num_cores = alloc_cores()
        
        print("\nGA CONFIGURATION:")
        print(f"├─ Generations: {num_generations}")
        print(f"├─ Population Size: {population_size}")
        print(f"└─ CPU cores: {num_cores}")
        print("=" * 60)
        
        # Run GA
        if self.PP:
            mp.set_start_method("spawn", force=True)
            ctx = get_context("spawn")
            with ctx.Pool(processes=num_cores) as pool:
                toolbox.register("map", pool.map)
                self._run_genetic_algorithm(
                    population, toolbox, num_generations,
                    start_gen=start_gen,
                    checkpoint_manager=checkpoint_manager,
                    output_interval=output_interval,
                )
        else:
            self._run_genetic_algorithm(
                population, toolbox, num_generations,
                start_gen=start_gen,
                checkpoint_manager=checkpoint_manager,
                output_interval=output_interval,
            )
        
        elapsed = time.time() - total_start
        print(f"\nGA completed in {elapsed:.1f}s")
        
        # Export results
        self.export_ga_samples()
        self.save_results()
        
        gc.collect()
    
    def _run_genetic_algorithm(
        self,
        population: List,
        toolbox: base.Toolbox,
        num_generations: int,
        start_gen: int = 0,
        checkpoint_manager: Optional[CheckpointManager] = None,
        output_interval: Optional[int] = None,
    ) -> None:
        """Internal GA loop."""
        
        if not hasattr(self, 'walker_history') or start_gen == 0:
            self.walker_history = {i: [] for i in range(len(population))}
        
        # Elitism
        elitism_k = max(1, len(population) // 16)
        
        for gen in range(start_gen, num_generations):
            print(f"\n{'='*20} Generation {gen}/{num_generations} {'='*20}")
            self.gen = gen
            
            # Step 1: Evaluate invalid individuals
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            if invalid_ind:
                if self.PP:
                    results = toolbox.map(toolbox.evaluate, invalid_ind)
                else:
                    results = [toolbox.evaluate(ind) for ind in invalid_ind]
                
                for ind, (fit, result) in zip(invalid_ind, results):
                    ind.fitness.values = fit
                    self._record_evaluation_result(result)
            
            # Step 2: Select elites
            elites = tools.selBest(population, elitism_k)
            elites = [toolbox.clone(e) for e in elites]
            
            # Print best fitness
            best = min(population, key=lambda x: x.fitness.values[0])
            print(f"Best fitness: {best.fitness.values[0]:.6f}")
            
            # Step 3: Select parents
            mating_pool = toolbox.select(population)
            mating_pool = list(map(toolbox.clone, mating_pool))
            needed_children = len(population) - elitism_k
            breed_pool = mating_pool[:needed_children]
            
            # Step 4: Crossover and mutation
            offspring = list(map(toolbox.clone, breed_pool))
            
            # Crossover (pairwise)
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    toolbox.mate(c1, c2)
                    if hasattr(c1.fitness, 'values'):
                        del c1.fitness.values
                    if hasattr(c2.fitness, 'values'):
                        del c2.fitness.values
            
            # Mutation
            for m in offspring:
                if random.random() < self.mutpb:
                    toolbox.mutate(m)
                    if hasattr(m.fitness, 'values'):
                        del m.fitness.values
            
            # De-duplicate
            offspring = self.prevent_duplicates(offspring, toolbox)
            
            # Ensure correct size
            if len(offspring) > needed_children:
                offspring = offspring[:needed_children]
            elif len(offspring) < needed_children:
                fillers = tools.selBest(population, needed_children - len(offspring))
                offspring += [toolbox.clone(f) for f in fillers]
            
            # Step 5: Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                if self.PP:
                    results = toolbox.map(toolbox.evaluate, invalid_ind)
                else:
                    results = [toolbox.evaluate(ind) for ind in invalid_ind]
                
                for ind, (fit, result) in zip(invalid_ind, results):
                    ind.fitness.values = fit
                    self._record_evaluation_result(result)
            
            # Step 6: Record history
            for idx, ind in enumerate(population):
                self.walker_history[idx].append(list(ind))
            
            # Step 7: Replace population
            population[:] = elites + offspring
            
            # Step 7b: DE-MC moves
            if self.demc_hybrid:
                self.apply_demc_hybrid_moves(population, toolbox)
            
            # Step 8: Voronoi exploration (early generations)
            if gen < 32 and self.exploration_steps > 0:
                try:
                    moved = voronoi_explore_dearths(self, population, exploration_fraction=0.1)
                    if moved > 0:
                        print(f"  Voronoi exploration: moved {moved} individuals")
                except Exception as e:
                    print(f"  Voronoi exploration failed: {e}")
            
            # Step 9: Checkpoint
            if checkpoint_manager:
                checkpoint_manager.save(gen, population, self)
            
            # Step 10: Periodic output
            if output_interval and (gen % output_interval == 0 or gen == num_generations - 1):
                self.save_partial_results(gen)
            
            gc.collect()
    
    def apply_demc_hybrid_moves(self, population: List, toolbox) -> None:
        """Apply DE-MC moves to a fraction of the population."""
        n_walkers = len(population)
        n_update = max(1, int(n_walkers * self.demc_fraction))
        
        # Select worst performers for DE-MC
        ranked = sorted(range(n_walkers), 
                       key=lambda i: population[i].fitness.values[0] if population[i].fitness.valid else float('inf'),
                       reverse=True)
        update_indices = ranked[:n_update]
        
        # DE-MC parameters
        d = len(self.continuous_indices)
        gamma = self.demc_gamma if self.demc_gamma else 2.38 / np.sqrt(2 * d)
        
        # Big jump every 6th generation
        if self.gen % 6 == 0:
            gamma = 1.0
        
        accepted = 0
        for idx in update_indices:
            # Select two different random walkers
            others = [i for i in range(n_walkers) if i != idx]
            if len(others) < 2:
                continue
            r1, r2 = random.sample(others, 2)
            
            # Propose new position
            current = population[idx]
            proposal = toolbox.clone(current)
            
            for i in self.continuous_indices:
                diff = population[r1][i] - population[r2][i]
                jitter = random.gauss(0, 1e-6)
                proposal[i] = current[i] + gamma * diff + jitter
                
                lo, hi = self.get_param_bounds(i)
                proposal[i] = self._reflect_at_bounds(proposal[i], lo, hi)
            
            # Evaluate proposal
            if hasattr(proposal.fitness, 'values'):
                del proposal.fitness.values
            
            fit, result = toolbox.evaluate(proposal)
            proposal.fitness.values = fit
            self._record_evaluation_result(result)
            
            # Metropolis-Hastings acceptance
            current_loss = current.fitness.values[0]
            proposal_loss = proposal.fitness.values[0]
            
            log_alpha = -(proposal_loss - current_loss)  # Assuming loss is negative log-likelihood
            
            if np.log(random.random()) < log_alpha:
                population[idx][:] = proposal[:]
                population[idx].fitness.values = proposal.fitness.values
                accepted += 1
        
        if n_update > 0:
            print(f"  DE-MC: {accepted}/{n_update} accepted ({100*accepted/n_update:.1f}%)")
    
    def update_operator_rates(self, population: List, gen: int, num_gens: int) -> None:
        """Optionally adjust crossover/mutation rates based on progress."""
        # Could implement adaptive rates here
        pass
    
    # =========================================================================
    # RESULTS SAVING
    # =========================================================================
    
    def save_results(self) -> None:
        """Save final results to CSV."""
        if not self.evaluation_results:
            print("No results to save")
            return
        
        # Build dataframe from evaluation results
        rows = []
        for r in self.evaluation_results:
            if 'individual' not in r:
                continue
            ind = r['individual']
            row = {
                'comp_idx': int(ind[0]),
                'imf_idx': int(ind[1]),
                'sn1a_idx': int(ind[2]),
                'sy_idx': int(ind[3]),
                'sn1ar_idx': int(ind[4]),
                'sigma_2': ind[5],
                't_1': ind[6],
                't_2': ind[7],
                'infall_1': ind[8],
                'infall_2': ind[9],
                'sfe': ind[10],
                'delta_sfe': ind[11],
                'imf_upper': ind[12],
                'mgal': ind[13],
                'nb': ind[14],
                'fitness': r.get('fitness', float('inf')),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('fitness', ascending=True)
        df = df.drop_duplicates(subset=PARAM_COLUMNS[:15], keep='first')
        
        output_file = os.path.join(self.output_path, 'simulation_results.csv')
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} results to {output_file}")
        
        # Also save best model info
        if len(df) > 0:
            best = df.iloc[0]
            print(f"\nBest model (fitness={best['fitness']:.6f}):")
            for col in PARAM_COLUMNS[:15]:
                if col in best:
                    print(f"  {col}: {best[col]}")
    
    def save_partial_results(self, gen: int) -> None:
        """Save intermediate results."""
        if not self.evaluation_results:
            return
        
        rows = []
        for r in self.evaluation_results:
            if 'individual' not in r:
                continue
            ind = r['individual']
            row = {
                'comp_idx': int(ind[0]),
                'imf_idx': int(ind[1]),
                'sn1a_idx': int(ind[2]),
                'sy_idx': int(ind[3]),
                'sn1ar_idx': int(ind[4]),
                'sigma_2': ind[5],
                't_1': ind[6],
                't_2': ind[7],
                'infall_1': ind[8],
                'infall_2': ind[9],
                'sfe': ind[10],
                'delta_sfe': ind[11],
                'imf_upper': ind[12],
                'mgal': ind[13],
                'nb': ind[14],
                'fitness': r.get('fitness', float('inf')),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('fitness', ascending=True)
        
        output_file = os.path.join(self.output_path, f'simulation_results_gen{gen}.csv')
        df.to_csv(output_file, index=False)
        print(f"Saved intermediate results to {output_file}")
    
    def export_ga_samples(self) -> None:
        """Export all evaluated individuals as GA samples."""
        if not self.evaluation_results:
            return
        
        rows = []
        for r in self.evaluation_results:
            if 'individual' not in r:
                continue
            ind = r['individual']
            row = {
                'comp_idx': int(ind[0]),
                'imf_idx': int(ind[1]),
                'sn1a_idx': int(ind[2]),
                'sy_idx': int(ind[3]),
                'sn1ar_idx': int(ind[4]),
                'sigma_2': ind[5],
                't_1': ind[6],
                't_2': ind[7],
                'infall_1': ind[8],
                'infall_2': ind[9],
                'sfe': ind[10],
                'delta_sfe': ind[11],
                'imf_upper': ind[12],
                'mgal': ind[13],
                'nb': ind[14],
                'fitness': r.get('fitness', float('inf')),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        output_file = os.path.join(self.output_path, 'ga_population_samples.csv')
        df.to_csv(output_file, index=False)
        print(f"Exported {len(df)} GA samples to {output_file}")
    
    def save_walker_history(self) -> None:
        """Save walker history to NPZ file."""
        if not self.walker_history:
            return
        
        # Also gather MDF/alpha data from evaluation results
        mdf_data = {}
        alpha_data = {}
        age_data = {}
        
        for i, r in enumerate(self.evaluation_results):
            if 'mdf_x' in r and 'mdf_y' in r:
                mdf_data[i] = (r['mdf_x'], r['mdf_y'])
            if 'alpha_data' in r:
                alpha_data[i] = r['alpha_data']
            if 'age_x' in r and 'age_y' in r:
                age_data[i] = (r['age_x'], r['age_y'])
        
        output_file = os.path.join(self.output_path, 'walker_history.npz')
        np.savez(
            output_file,
            walker_history=self.walker_history,
            mdf_data=mdf_data,
            alpha_data=alpha_data,
            age_data=age_data,
        )
        print(f"Saved walker history to {output_file}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_ga_from_config(config: Dict, checkpoint_manager: Optional[CheckpointManager] = None) -> GalacticEvolutionGA:
    """
    Create and run GA from configuration dictionary.
    
    Parameters
    ----------
    config : dict
        Configuration from parse_inlist or load_config
    checkpoint_manager : CheckpointManager, optional
        Checkpoint manager for resuming
        
    Returns
    -------
    GalacticEvolutionGA
        The GA instance after running
    """
    # Load observational data
    feh, count = np.loadtxt(config['obs_file'], usecols=(0, 1), unpack=True)
    normalized_count = count / max(count.max(), 1.0)
    
    # Create GA instance
    ga = GalacticEvolutionGA(
        output_path=config.get('output_path', 'SMC_DEMC/'),
        sn1a_header=config['sn1a_header'],
        iniab_header=config['iniab_header'],
        sigma_2_list=config['sigma_2_list'],
        tmax_1_list=config['tmax_1_list'],
        tmax_2_list=config['tmax_2_list'],
        infall_timescale_1_list=config['infall_timescale_1_list'],
        infall_timescale_2_list=config['infall_timescale_2_list'],
        comp_array=config['comp_array'],
        imf_array=config['imf_array'],
        sfe_array=config['sfe_array'],
        delta_sfe_array=config['delta_sfe_array'],
        imf_upper_limits=config['imf_upper_limits'],
        sn1a_assumptions=config['sn1a_assumptions'],
        stellar_yield_assumptions=config['stellar_yield_assumptions'],
        mgal_values=config['mgal_values'],
        nb_array=config['nb_array'],
        sn1a_rates=config['sn1a_rates'],
        timesteps=config.get('timesteps', 1000),
        A1=config.get('A1', 1.0),
        A2=config.get('A2', 1.0),
        feh=feh,
        normalized_count=normalized_count,
        obs_age_data=config.get('obs_age_data'),
        loss_metric=config.get('loss_metric', 'ensemble'),
        obs_age_data_loss_metric=config.get('obs_age_data_loss_metric', 'None'),
        obs_age_data_target=config.get('obs_age_data_target', 'joyce'),
        mdf_vs_age_weight=config.get('mdf_vs_age_weight', 1.0),
        fancy_mutation=config.get('fancy_mutation', 'gaussian'),
        shrink_range=config.get('shrink_range', False),
        tournament_size=config.get('tournament_size', 3),
        threshold=config.get('selection_threshold', -1),
        cxpb=config.get('crossover_probability', 0.5),
        mutpb=config.get('mutation_probability', 0.5),
        gaussian_sigma_scale=config.get('gaussian_sigma_scale', 0.01),
        crossover_noise_fraction=config.get('crossover_noise_fraction', 0.05),
        perturbation_strength=config.get('perturbation_strength', 0.1),
        physical_constraints_freq=config.get('physical_constraints_freq', 10),
        exploration_steps=config.get('exploration_steps', 0),
        PP=config.get('PP', True),
        demc_hybrid=config.get('demc_hybrid', True),
        demc_fraction=config.get('demc_fraction', 0.5),
        demc_moves_per_gen=config.get('demc_moves_per_gen', 1),
        plot_mode=config.get('plot_mode', 'full'),
    )
    
    # Initialize
    popsize = config.get('popsize', 96)
    generations = config.get('generations', 256)
    output_interval = config.get('output_interval', 16)
    
    population, toolbox = ga.init_GenAl(population_size=popsize)
    
    # Check for checkpoint
    start_gen = 0
    if checkpoint_manager:
        cp_data = checkpoint_manager.load()
        if cp_data:
            start_gen = cp_data.get('generation', 0) + 1
            # Could restore population here
    
    # Run
    ga.GenAl(
        population_size=popsize,
        num_generations=generations,
        population=population,
        toolbox=toolbox,
        checkpoint_manager=checkpoint_manager,
        start_gen=start_gen,
        output_interval=output_interval,
    )
    
    return ga
