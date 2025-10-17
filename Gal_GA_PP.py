#!/usr/bin/env python3.8
################################
# Author: N Miller
################################
#import imp
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import sys
#testing jesus
#from sklearn import preprocessing
sys.path.append('../')

import gc
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
from matplotlib import cm
from matplotlib.lines import *
from matplotlib.patches import *
from JINAPyCEE import omega_plus
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, cpu_count

from deap import base, creator, tools
import random
import pandas as pd
import os
import mdf_plotting
import corner
from smc_demc import Bound, run_smc_demc, de_mh_move
from loss import *
from physical_constraints import apply_physics_penalty
from explore_dearth import voronoi_explore_dearths
import ast
from age_meta import age_meta_loss, test_age_meta_loss_function



def alloc_cores():
    import os
    try:
        # exact count in current cpuset (best under Slurm)
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # fallback to Slurm env or Python's view
        return int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))





# Function to find the index of the nearest value in an array
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


# Function to model inflow rates
def two_inflow_fn(t, exp_inflow):
    if t < exp_inflow[1][1]:
        return exp_inflow[0][0] * np.exp(-t / exp_inflow[0][2])
    else:
        return (exp_inflow[0][0] * np.exp(-t / exp_inflow[0][2]) +
                exp_inflow[1][0] * np.exp(-(t - exp_inflow[1][1]) / exp_inflow[1][2]))


# Function to parse the inlist file into a dictionary
def parse_inlist(file_path):
    """Parse an inlist file and return a dictionary of parameters."""
    params = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            lowered = value.lower().strip("'\"")
            if lowered in {'true', 'false'}:
                parsed_value = lowered == 'true'
            else:
                try:
                    parsed_value = ast.literal_eval(value)
                except Exception:
                    parsed_value = value

            params[key] = parsed_value

    return params



def print_population(GA, population, generation):
    """Helper function to print population details."""
    print(f"\nFitness:")
    for i, individual in enumerate(population):
        print(f"Individual {i}: {individual}, Fitness: {individual.fitness.values if individual.fitness.valid else 'Not evaluated'}")
    print(f"---------------\n")




def log_uniform(min_val, max_val):
    """Sample uniformly in log space"""
    log_min = np.log10(min_val)
    log_max = np.log10(max_val)
    return 10**random.uniform(log_min, log_max)

def should_use_log(min_val, max_val, threshold=2.0):
    """Check if parameter spans more than threshold orders of magnitude"""
    if min_val <= 0 or max_val <= 0:
        return False
    return False#np.log10(max_val / min_val) >= threshold


# ---------- helpers for printing ----------
def _is_numeric_seq(x):
    try:
        arr = _np.asarray(x)
        return arr.dtype.kind in "iufo"
    except Exception:
        return False

def _minmax(x):
    arr = _np.asarray(x)
    return _np.nanmin(arr), _np.nanmax(arr)

def _dtype(x):
    try:
        return str(_np.asarray(x).dtype)
    except Exception:
        return type(x).__name__

def _preview(x, max_items=12):
    # robust preview for sequences / arrays; for scalars just repr
    try:
        seq = list(x)
    except TypeError:
        return repr(x)
    n = len(seq)
    if n <= max_items:
        return repr(seq)
    head = ", ".join(repr(v) for v in seq[:max_items//2])
    tail = ", ".join(repr(v) for v in seq[-max_items//2:])
    return f"[{head}, ..., {tail}]  (showing {max_items} of {n})"

def _summarize(name, x):
    # returns multi-line string summarizing object x
    if isinstance(x, (str, bytes)):
        return f"{name}: type={type(x).__name__}, value={repr(x)}"
    # sequences/arrays
    try:
        n = len(x)  # will fail for scalars
        numeric = _is_numeric_seq(x)
        dt = _dtype(x)
        if numeric and n > 0:
            lo, hi = _minmax(x)
            return f"{name}: len={n}, dtype={dt}, min={lo}, max={hi}, values={_preview(x)}"
        else:
            return f"{name}: len={n}, dtype={dt}, values={_preview(x)}"
    except Exception:
        # scalar
        return f"{name}: type={type(x).__name__}, value={repr(x)}"



class GalacticEvolutionGA:

    def __init__(self, output_path, sn1a_header, iniab_header, sigma_2_list, tmax_1_list, tmax_2_list, infall_timescale_1_list, 
                infall_timescale_2_list, comp_array, imf_array, sfe_array, delta_sfe_array, imf_upper_limits, sn1a_assumptions,
                stellar_yield_assumptions, mgal_values, nb_array, sn1a_rates, timesteps,A1, A2, feh, normalized_count, obs_age_data,
                loss_metric='huber', obs_age_data_loss_metric = 'None', obs_age_data_target = 'joyce', mdf_vs_age_weight = 1, fancy_mutation = 'gaussian', 
                shrink_range = False, tournament_size = 3, lambda_diversity = 0.01, threshold = -1, cxpb=0.5, mutpb=0.5,
                gaussian_sigma_scale=0.01, crossover_noise_fraction=0.05, perturbation_strength=0.1, physical_constraints_freq = 10, exploration_steps=0, PP = False,
                demc_hybrid=True, demc_fraction=0.5, demc_moves_per_gen=1, demc_gamma=None, demc_rng_seed=None, demc_workers=None, plot_mode="full"):

        # Initialize parameters as instance variables
        self.output_path = output_path
        self.sn1a_header = sn1a_header
        self.iniab_header = iniab_header
        self.sigma_2_list = sigma_2_list
        self.tmax_1_list = tmax_1_list
        self.tmax_2_list = tmax_2_list
        self.infall_timescale_1_list = infall_timescale_1_list
        self.infall_timescale_2_list = infall_timescale_2_list        
        self.comp_array = comp_array
        self.imf_array = imf_array
        self.sfe_array = sfe_array
        self.delta_sfe_array = delta_sfe_array  # Change in SFE at second infall
        self.imf_upper_limits = imf_upper_limits
        self.sn1a_assumptions = sn1a_assumptions
        self.stellar_yield_assumptions = stellar_yield_assumptions
        self.mgal_values = mgal_values
        self.nb_array = nb_array
        self.sn1a_rates = sn1a_rates
        self.timesteps = timesteps
        self.A1 = A1
        self.A2 = A2        
        self.feh = feh
        self.normalized_count = normalized_count
        self.obs_age_data = obs_age_data
        self.placeholder_sigma_array = np.zeros(len(normalized_count)) + 1  # Assume all sigmas are 1
        self.fancy_mutation = fancy_mutation
        self.PP = PP
        self.quant_individuals = False
        self.model_count = 0
        self.mdf_data = []
        self.age_data = []
        self.alpha_data = []
        self.results = []
        self.labels = []
        self.MDFs = []
        self.alpha_data = []
        self.model_numbers = []
        self.metric_header = [
            'comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx',
            'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2',
            'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb',
            'ks', 'ensemble', 'wrmse', 'mae', 'mape', 'huber',
            'cosine', 'log_cosh', 'fitness', 'age_meta_fitness', 'physics_penalty'
        ]
        self.sample_records = []
        self.ga_samples_path = None
        self.shrink_range = shrink_range
        # Min and max values for sigma_2, t_2, and infall_2
        self.sigma_2_min, self.sigma_2_max = min(sigma_2_list), max(sigma_2_list)
        self.t_2_min, self.t_2_max = min(tmax_2_list), max(tmax_2_list)
        self.infall_2_min, self.infall_2_max = min(infall_timescale_2_list), max(infall_timescale_2_list)

        self.loss_metric = loss_metric
        self.obs_age_data_loss_metric = obs_age_data_loss_metric
        self.obs_age_data_target = obs_age_data_target
        self.mdf_vs_age_weight = mdf_vs_age_weight

        self.cxpb=cxpb
        self.mutpb=mutpb
        self.gaussian_sigma_scale = gaussian_sigma_scale
        self.crossover_noise_fraction = crossover_noise_fraction
        self.perturbation_strength = perturbation_strength
        self.exploration_steps = exploration_steps
        self.best_amr_loss = 0.10
        self.plot_mode = str(plot_mode)

        # Differential Evolution MCMC hybrid configuration
        self.demc_hybrid = bool(demc_hybrid)
        self.demc_fraction = float(np.clip(demc_fraction, 0.0, 1.0))
        self.demc_moves_per_gen = max(1, int(demc_moves_per_gen))
        self.demc_gamma = demc_gamma
        self.demc_workers = None if demc_workers in (None, 0) else int(demc_workers)


        self.demc_rng = np.random.default_rng(demc_rng_seed)

        
        # Calculate parameter space dimensions for reporting
        categorical_params = len(comp_array) * len(imf_array) * len(sn1a_assumptions) * len(stellar_yield_assumptions) * len(sn1a_rates)
        continuous_param_ranges = [
            (max(sigma_2_list) - min(sigma_2_list)),
            (max(tmax_1_list) - min(tmax_1_list)), 
            (max(tmax_2_list) - min(tmax_2_list)),
            (max(infall_timescale_1_list) - min(infall_timescale_1_list)),
            (max(infall_timescale_2_list) - min(infall_timescale_2_list)),
            (max(sfe_array) - min(sfe_array)),
            (max(delta_sfe_array) - min(delta_sfe_array)),
            (max(imf_upper_limits) - min(imf_upper_limits)),
            (max(mgal_values) - min(mgal_values)),
            (max(nb_array) - min(nb_array))
        ]
        
        observational_constraints = len(feh)


        # Define available loss metrics
        self.loss_functions = {
            'wrmse': compute_wrmse,
            'mae': compute_mae,
            'mape': compute_mape,
            'huber': compute_huber,
            'cosine': compute_cosine_similarity,
            'ks': compute_ks_distance,
            'ensemble': compute_ensemble_metric,
            'log_cosh': compute_log_cosh
        }

        # Select the loss function based on user input
        if loss_metric not in self.loss_functions:
            raise ValueError(f"Invalid loss metric. Available options are: {list(self.loss_functions.keys())}")
        
        self.selected_loss_function = self.loss_functions[loss_metric]

        self.all_gene_values_successful = []
        self.all_gene_values_unsuccessful = []
        self.all_losses_successful = []
        self.all_losses_unsuccessful = []
        self.gene_bounds = []
        self.global_min_loss = None
        self.global_max_loss = None
        
        self.threshold = threshold
        self.tournament_size = tournament_size
        self.backup_tournament_size = 0
        self.lambda_diversity = lambda_diversity #A higher value places more emphasis on diversity.

        self.physics_timer = 0
        self.physical_constraints_freq = physical_constraints_freq
        # Define which indices are categorical vs continuous
        self.categorical_indices = [0, 1, 2, 3, 4]  # comp, imf, sn1a, stellar_yield, sn1a_rate
        self.continuous_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # sigma_2, t_1, t_2, etc.
        
        # Map from index to parameter name (for getting bounds dynamically)
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



        # ---------- CONFIG DUMP ----------
        print("\n==================== GALACTIC EVOLUTION GA CONFIG ====================")
        print("FILES / HEADERS")
        print(_summarize("output_path", output_path))
        print(_summarize("sn1a_header", sn1a_header))
        print(_summarize("iniab_header", iniab_header))
        print()

        print("PARAMETER GRIDS (CATEGORICAL / DISCRETE)")
        print(_summarize("comp_array", comp_array))
        print(_summarize("imf_array", imf_array))
        print(_summarize("sn1a_assumptions", sn1a_assumptions))
        print(_summarize("stellar_yield_assumptions", stellar_yield_assumptions))
        print(_summarize("sn1a_rates", sn1a_rates))
        print()

        print("PARAMETER RANGES (CONTINUOUS / NUMERIC LISTS)")
        print(_summarize("sigma_2_list ", sigma_2_list))
        print(_summarize("tmax_1_list (Gyr)", tmax_1_list))
        print(_summarize("tmax_2_list (Gyr)", tmax_2_list))
        print(_summarize("infall_timescale_1_list (Gyr)", infall_timescale_1_list))
        print(_summarize("infall_timescale_2_list (Gyr)", infall_timescale_2_list))
        print(_summarize("sfe_array", sfe_array))
        print(_summarize("delta_sfe_array", delta_sfe_array))
        print(_summarize("imf_upper_limits (Msun)", imf_upper_limits))
        print(_summarize("mgal_values (Msun)", mgal_values))
        print(_summarize("nb_array (per Msun)", nb_array))
        print()

        print("OBSERVATIONAL DATA")
        print(_summarize("[Fe/H] grid (feh)", feh))
        print(_summarize("normalized_count MDF", normalized_count))
        print(_summarize("obs_age_data", obs_age_data))
        print()

        print("MODEL INTEGRATION / RESOLUTION")
        print(_summarize("timesteps", timesteps))
        print(_summarize("A1 (infall amplitude 1)", A1))
        print(_summarize("A2 (infall amplitude 2)", A2))
        print()

        print("LOSS / TARGETS")
        print(f"available_loss_metrics: {list(self.loss_functions.keys())}")
        print(f"selected_loss_metric: {loss_metric}")
        print(f"obs_age_data_loss_metric: {obs_age_data_loss_metric}")
        print(f"obs_age_data_target: {obs_age_data_target}")
        print(f"mdf_vs_age_weight: {mdf_vs_age_weight}")
        print()

        print("GA SETTINGS")
        print(f"selection: tournament (size={tournament_size})")
        print(f"crossover: cxpb={cxpb}, noise_fraction={crossover_noise_fraction}")
        print(f"mutation:  mutpb={mutpb}, gaussian_sigma_scale={gaussian_sigma_scale}, perturbation_strength={perturbation_strength}")
        print(f"fancy_mutation: {fancy_mutation}")
        print(f"lambda_diversity: {lambda_diversity}")
        print(f"threshold (early-stop / acceptance): {threshold}")
        print(f"physics_constraints_freq: every {physical_constraints_freq} evals")
        print(f"exploration_steps (pre-random walks): {exploration_steps}")
        print(f"parallel_processing (PP): {bool(PP)}")
        print(f"plot_mode: {self.plot_mode}")
        print(f"shrink_range: {bool(shrink_range)}")
        print()

        print("DEMC HYBRID")
        print(f"demc_hybrid: {bool(self.demc_hybrid)}")
        print(f"demc_fraction: {self.demc_fraction}")
        print(f"demc_moves_per_gen: {self.demc_moves_per_gen}")
        print(f"demc_gamma: {self.demc_gamma}")
        print(f"demc_rng_seed: {demc_rng_seed}")
        print(f"demc_workers: {self.demc_workers}")
        print()

        print("DERIVED / SANITY CHECKS")
        print(f"t2_range (Gyr): [{self.t_2_min}, {self.t_2_max}]")
        print(f"sigma2_range : [{self.sigma_2_min}, {self.sigma_2_max}]")
        print(f"infall2_timescale_range (Gyr): [{self.infall_2_min}, {self.infall_2_max}]")
        try:
            total_cont_dims = 10
            print(f"parameter_space: {categorical_params:,} categorical × {total_cont_dims} continuous dims")
        except Exception:
            pass
        print(f"observational_points (MDF bins): {observational_constraints}")
        try:
            vol_est = f"~{categorical_params:.0e}"
            print(f"expected_discrete_combinations: {vol_est}")
        except Exception:
            pass
        print("======================================================================\n")



    def init_GenAl(self, population_size):
        # DEAP framework setup for Genetic Algorithm (GA)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Toolbox to define how individuals (solutions) are created and evolve
        toolbox = base.Toolbox()

        # Register attribute generators for all parameters
        # Truly discrete choices (categorical parameters)
        toolbox.register("comp_attr", lambda: random.randint(0, len(self.comp_array) - 1))
        toolbox.register("imf_attr", lambda: random.randint(0, len(self.imf_array) - 1))
        toolbox.register("sn1a_attr", lambda: random.randint(0, len(self.sn1a_assumptions) - 1))
        toolbox.register("sy_attr", lambda: random.randint(0, len(self.stellar_yield_assumptions) - 1))
        toolbox.register("sn1a_rate_attr", lambda: random.randint(0, len(self.sn1a_rates) - 1))
                

        #toolbox.register("sigma_2_attr", random.uniform, min(self.sigma_2_list), max(self.sigma_2_list))
        toolbox.register("sigma_2_attr", log_uniform, min(self.sigma_2_list), max(self.sigma_2_list))

        # t_1
        if should_use_log(min(self.tmax_1_list), max(self.tmax_1_list)):
            toolbox.register("t_1_attr", log_uniform, min(self.tmax_1_list), max(self.tmax_1_list))
        else:
            toolbox.register("t_1_attr", random.uniform, min(self.tmax_1_list), max(self.tmax_1_list))

        # t_2
        if should_use_log(min(self.tmax_2_list), max(self.tmax_2_list)):
            toolbox.register("t_2_attr", log_uniform, min(self.tmax_2_list), max(self.tmax_2_list))
        else:
            toolbox.register("t_2_attr", random.uniform, min(self.tmax_2_list), max(self.tmax_2_list))

        # infall_1
        if should_use_log(min(self.infall_timescale_1_list), max(self.infall_timescale_1_list)):
            toolbox.register("infall_1_attr", log_uniform, min(self.infall_timescale_1_list), max(self.infall_timescale_1_list))
        else:
            toolbox.register("infall_1_attr", random.uniform, min(self.infall_timescale_1_list), max(self.infall_timescale_1_list))

        # infall_2
        if should_use_log(min(self.infall_timescale_2_list), max(self.infall_timescale_2_list)):
            toolbox.register("infall_2_attr", log_uniform, min(self.infall_timescale_2_list), max(self.infall_timescale_2_list))
        else:
            toolbox.register("infall_2_attr", random.uniform, min(self.infall_timescale_2_list), max(self.infall_timescale_2_list))

        # sfe
        toolbox.register("sfe_attr", log_uniform, min(self.sfe_array), max(self.sfe_array))

        # delta_sfe
        if should_use_log(min(self.delta_sfe_array), max(self.delta_sfe_array)):
            toolbox.register("delta_sfe_attr", log_uniform, min(self.delta_sfe_array), max(self.delta_sfe_array))
        else:
            toolbox.register("delta_sfe_attr", random.uniform, min(self.delta_sfe_array), max(self.delta_sfe_array))

        # imf_upper
        if should_use_log(min(self.imf_upper_limits), max(self.imf_upper_limits)):
            toolbox.register("imf_upper_attr", log_uniform, min(self.imf_upper_limits), max(self.imf_upper_limits))
        else:
            toolbox.register("imf_upper_attr", random.uniform, min(self.imf_upper_limits), max(self.imf_upper_limits))

        # mgal
        if should_use_log(min(self.mgal_values), max(self.mgal_values)):
            toolbox.register("mgal_attr", log_uniform, min(self.mgal_values), max(self.mgal_values))
        else:
            toolbox.register("mgal_attr", random.uniform, min(self.mgal_values), max(self.mgal_values))

        # nb
        if should_use_log(min(self.nb_array), max(self.nb_array)):
            toolbox.register("nb_attr", log_uniform, min(self.nb_array), max(self.nb_array))
        else:
            toolbox.register("nb_attr", random.uniform, min(self.nb_array), max(self.nb_array))

        # Create an individual by combining all attributes
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.comp_attr, toolbox.imf_attr, toolbox.sn1a_attr, 
                          toolbox.sy_attr, toolbox.sn1a_rate_attr,
                          toolbox.sigma_2_attr, toolbox.t_1_attr, toolbox.t_2_attr, 
                          toolbox.infall_1_attr, toolbox.infall_2_attr,
                          toolbox.sfe_attr, toolbox.delta_sfe_attr, toolbox.imf_upper_attr, 
                          toolbox.mgal_attr, toolbox.nb_attr), n=1)

        # Create a population by repeating individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register the evaluation function
        toolbox.register("evaluate", self.evaluate)

        # Register genetic operations
        toolbox.register("mate", self.crossover, max_bias=0.55)

        # Define different mutation functions based on fancy_mutation parameter
        if self.fancy_mutation.lower() == 'uniform':
            def mutate_with_population(individual):
                return self.uniform_mutate(individual)
            
        elif self.fancy_mutation.lower() == 'gaussian':
            def mutate_with_population(individual):
                return self.gaussian_mutate(individual, base_sigma_scale=self.gaussian_sigma_scale)
                

        toolbox.register("mutate", mutate_with_population)
        
        toolbox.register("select", self.selTournament, tournsize=self.tournament_size)#, lambda_diversity=self.lambda_diversity)


        if self.demc_workers is None:
            self.demc_workers = population_size

        print("Walkers")
        print(f"GA individuals: {population_size}")
        print(f"demc_workers: {self.demc_workers}")
        print("")


        #test that supid ass age meta stuff. 
        test_age_meta_loss_function(self)

        # Create the initial population
        population = toolbox.population(n=population_size)
        return population, toolbox


    def crossover(self, ind1, ind2, max_bias=0.75):
        """Crossover that favors the fitter parent up to max_bias%"""
        
        # Determine which parent is fitter (lower fitness = better since we minimize)
        if ind1.fitness.valid and ind2.fitness.valid:
            fit1 = ind1.fitness.values[0]
            fit2 = ind2.fitness.values[0]
            
            # Calculate fitness difference and weight toward better parent
            total_fitness = fit1 + fit2
            if total_fitness > 0:
                # Weight inversely proportional to fitness (lower fit = higher weight)
                weight1 = fit2 / total_fitness  # Better parent gets higher weight
                weight2 = fit1 / total_fitness
                
                # Cap the bias at max_bias
                if weight1 > max_bias:
                    weight1 = max_bias
                    weight2 = 1 - max_bias
                elif weight2 > max_bias:
                    weight2 = max_bias
                    weight1 = 1 - max_bias
            else:
                # Fallback if fitness calculation fails
                weight1 = weight2 = 0.5
        else:
            # If fitness not available, use equal weighting
            weight1 = weight2 = 0.5
        
        # Create copies of parents
        ind1_copy = ind1[:]
        ind2_copy = ind2[:]
        
        # Handle categorical parameters with fitness-weighted selection
        categorical_indices = [0, 1, 2, 3, 4]

        for i in categorical_indices:
            # child 1
            ind1_copy[i] = ind1[i] if random.random() < weight1 else ind2[i]
            # child 2
            ind2_copy[i] = ind1[i] if random.random() < weight1 else ind2[i]

        # Handle continuous parameters with fitness-weighted blending
        continuous_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        for i in continuous_indices:
            # Fitness-weighted average with small noise
            avg_val = weight1 * ind1[i] + weight2 * ind2[i]
            noise_scale = abs(ind1[i] - ind2[i]) * self.crossover_noise_fraction
            
            ind1_copy[i] = avg_val + random.gauss(0, noise_scale)
            ind2_copy[i] = avg_val + random.gauss(0, noise_scale)
            
            # Use reflection instead:
            min_bound, max_bound = self.get_param_bounds(i)
            ind1_copy[i] = self._reflect_at_bounds(ind1_copy[i], min_bound, max_bound)
            ind2_copy[i] = self._reflect_at_bounds(ind2_copy[i], min_bound, max_bound)

        return ind1_copy, ind2_copy



    def selTournament(self, individuals, tournsize=3):
        """
        Tournament selection that heavily prioritizes fitness, with occasional diversity.
        """
        selected = []
        
        while len(selected) < len(individuals):
            tournament = random.sample(individuals, tournsize)
            winner = min(tournament, key=lambda ind: ind.fitness.values[0])
            selected.append(winner)
        
        return selected



    def prevent_duplicates(self, offspring, toolbox, max_attempts=5):
        """Replace duplicate individuals with controlled perturbations"""
        unique_keys = set()
        distinct_offspring = []

        for ind in offspring:
            key = tuple(round(x, 6) if isinstance(x, float) else x for x in ind)

            if key in unique_keys:
                new_ind = toolbox.clone(ind)
                attempt = 0

                while attempt < max_attempts:
                    # Use a single, varied perturbation instead of multiple mutations
                    self.controlled_perturbation(new_ind, strength=self.perturbation_strength * (1 + random.random()))
                    del new_ind.fitness.values

                    new_key = tuple(round(x, 6) if isinstance(x, float) else x for x in new_ind)
                    if new_key not in unique_keys:
                        key = new_key
                        break
                    attempt += 1

                distinct_offspring.append(new_ind)
                unique_keys.add(key)
            else:
                unique_keys.add(key)
                distinct_offspring.append(ind)

        return distinct_offspring



    def controlled_perturbation(self, individual, strength=None):
        """Apply a controlled perturbation with varied step sizes using reflection at boundaries, scaled by fitness"""
        if strength is None:
            strength = self.perturbation_strength
        
        # Get fitness-based scaling factor
        fitness_scale = self.get_fitness_scale(individual)
        
        # Apply fitness scaling to strength
        scaled_strength = strength * fitness_scale
        
        for i in range(len(individual)):
            if i in self.categorical_indices:
                # Small chance to change categorical parameters (also scaled by fitness)
                if random.random() < (0.05 * fitness_scale):
                    param_name = self.index_to_param_map[i]
                    num_categories = len(getattr(self, param_name))
                    individual[i] = random.randint(0, num_categories - 1)
            else:
                # Varied continuous perturbations with reflection
                min_bound, max_bound = self.get_param_bounds(i)
                range_size = max_bound - min_bound
                
                # Random step size between 0.5% and 10% of range, scaled by strength and fitness
                step_fraction = (1.0 * random.random()) * scaled_strength
                sigma = range_size * step_fraction
                
                # Apply perturbation
                new_value = individual[i] + random.gauss(0, sigma)
                
                # Reflect at boundaries to preserve perturbation magnitude
                new_value = self._reflect_at_bounds(new_value, min_bound, max_bound)
                individual[i] = new_value



    def get_fitness_scale(self, individual):
        """Calculate fitness-based scaling factor with activation function"""
        if not individual.fitness.valid:
            return 1.0  # Default scaling if no fitness available
        
        fitness = individual.fitness.values[0]
        
        # Activation function: linear below 0.1, higher scaling above
        if fitness < 0.1:
            # Linear scaling below 0.1
            scale = fitness
        else:
            # Higher scaling above 0.1: 0.1 + (fitness - 0.1)^1.5
            scale = 0.1 + (fitness - 0.1) ** 1.5
        
        # Ensure minimum scaling to prevent zero perturbation
        return max(scale, 0.01)





    def _reflect_at_bounds(self, value, min_bound, max_bound):
        """Reflect value at boundaries to preserve perturbation magnitude"""
        range_size = max_bound - min_bound
        
        if value < min_bound:
            # Reflect below lower bound
            excess = min_bound - value
            # Handle multiple reflections for large perturbations
            excess = excess % (2 * range_size)
            if excess <= range_size:
                return min_bound + excess
            else:
                return max_bound - (excess - range_size)
        
        elif value > max_bound:
            # Reflect above upper bound  
            excess = value - max_bound
            # Handle multiple reflections for large perturbations
            excess = excess % (2 * range_size)
            if excess <= range_size:
                return max_bound - excess
            else:
                return min_bound + (excess - range_size)
        
        else:
            # Within bounds, no reflection needed
            return value



    def update_operator_rates(self, population, generation, num_generations):
        """
        Deterministic 3-phase schedule (seed-agnostic):
          Phase A (0 → gA): brief exploration
          Phase B (gA → gB): focus
          Phase C (gB → end): exploit
        Voronoi dearth exploration: early only, deterministic (or disabled if not supported).
        """

        # --- schedule (fractions of total gens) ---
        explore_frac = getattr(self, "explore_frac", self.exploration_steps)   # first 20% of gens
        focus_frac   = getattr(self, "focus_frac",   0.70)   # until 70%; last 30% exploit

        gA = int(explore_frac * num_generations)
        gB = int(focus_frac   * num_generations)

        # --- operator levels (fixed numbers) ---
        mut_hi, mut_md, mut_lo = self.mutpb, 0.25, 0.15
        cx_lo,  cx_md,  cx_hi  = 0.50, 0.65, self.cxpb

        # keep tournament size small and constant (prevents lock-in to wrong basin)
        self.tournament_size = getattr(self, "tournament_size", 2)

        # --- choose phase by generation index (no feedback from population) ---
        if generation < gA:
            # Phase A: exploration
            self.mutpb = mut_hi
            self.cxpb  = cx_lo
            vor_frac   = 0.4
        elif generation < gB:
            # Phase B: focus
            self.mutpb = mut_md
            self.cxpb  = cx_md
            vor_frac   = 0.0
        else:
            # Phase C: exploit
            self.mutpb = mut_lo
            self.cxpb  = cx_hi
            vor_frac   = 0.0

        if generation >= gB:
            self.tournament_size = 4


        # --- early Voronoi only, without randomness ---
        if vor_frac > 0.0:
                voronoi_explore_dearths(
                    self, population,
                    exploration_fraction=vor_frac,
                )

        if generation % 1 == 0 or generation in (0, gA, gB, num_generations-1):
            print(f"Gen {generation:>4}/{num_generations} "
                  f"mutpb={self.mutpb:.2f}  cxpb={self.cxpb:.2f}  Voronoi={vor_frac:.2f}")



    def get_param_bounds(self, index):
        """
        Returns the min and max bounds for the given continuous parameter index.
        """
        if index == 5:
            return self.sigma_2_min, self.sigma_2_max
        elif index == 6:
            return min(self.tmax_1_list), max(self.tmax_1_list)
        elif index == 7:
            return self.t_2_min, self.t_2_max
        elif index == 8:
            return min(self.infall_timescale_1_list), max(self.infall_timescale_1_list)
        elif index == 9:
            return self.infall_2_min, self.infall_2_max
        elif index == 10:
            return min(self.sfe_array), max(self.sfe_array)
        elif index == 11:
            return min(self.delta_sfe_array), max(self.delta_sfe_array)            
        elif index == 12:
            return min(self.imf_upper_limits), max(self.imf_upper_limits)
        elif index == 13:
            return min(self.mgal_values), max(self.mgal_values)
        elif index == 14:
            return min(self.nb_array), max(self.nb_array)
        else:
            raise IndexError(f"No bounds defined for parameter index {index}")



    


    def uniform_mutate(self, individual, indpb=1.0):
        """
        Uniform mutation that replaces values with uniform random values 
        within parameter bounds.
        """
        for i in range(len(individual)):
            if random.random() < indpb:
                # Handle categorical parameters
                if i in self.categorical_indices:
                    param_name = self.index_to_param_map[i]
                    num_categories = len(getattr(self, param_name))
                    individual[i] = random.randint(0, num_categories - 1)
                # Handle continuous parameters
                else:
                    min_bound, max_bound = self.get_param_bounds(i)
                    individual[i] = random.uniform(min_bound, max_bound)
        
        return individual,




    def gaussian_mutate(self, individual, indpb=1.0, base_sigma_scale=None):
        """Mutation with anti-oscillation and varied step sizes"""

        if base_sigma_scale is None:
            base_sigma_scale = self.gaussian_sigma_scale
        
        # Store previous values if available (you'd need to track this)
        if hasattr(individual, 'prev_values'):
            prev_values = individual.prev_values
        else:
            prev_values = None
        
        current_values = individual[:]
        
        fitness_scale = self.get_fitness_scale(individual)

        for i in range(len(individual)):
            if random.random() < indpb:
                if i in self.categorical_indices:
                    if random.random() < 0.1:
                        param_name = self.index_to_param_map[i]
                        num_categories = len(getattr(self, param_name))
                        individual[i] = random.randint(0, num_categories - 1)
                else:
                    min_bound, max_bound = self.get_param_bounds(i)
                    range_size = max_bound - min_bound
                    
                    # Adaptive step size based on generation progress
                    if hasattr(self, 'gen') and hasattr(self, 'num_generations'):
                        progress = self.gen / self.num_generations
                        # Start larger, get smaller, but maintain some diversity
                        base_scale = base_sigma_scale * (1 - 0.5 * progress)
                    else:
                        base_scale = base_sigma_scale
                    
                    
                    step_multiplier = 0.5 + 0.5 * random.random()
                    base_scale = max(0.3 * base_sigma_scale, base_scale)

                    sigma = range_size * base_scale * step_multiplier * fitness_scale
                    
                    # Anti-oscillation: if we're moving back toward previous value, 
                    # sometimes force movement in same direction
                    new_value = individual[i] + random.gauss(0, sigma)
                    


                    # Apply bounds
                    new_value = self._reflect_at_bounds(new_value, min_bound, max_bound)
                    individual[i] = new_value
        
        # Store current values as previous for next mutation
        individual.prev_values = current_values[:]
        
        return individual,







    def GenAl(
        self,
        population_size,
        num_generations,
        population,
        toolbox,
        checkpoint_manager=None,
        start_gen=0,
        output_interval=None,
    ):
        """
        GA main loop (unchanged behavior) followed by the integrated SMC-DEMC
        refinement stage. After finishing the GA generations we run a tempered
        Sequential Monte Carlo with Differential-Evolution Metropolis moves to
        turn the final ensemble into posterior-quality samples.
        """
        import gc, time
        import multiprocessing as mp
        from multiprocessing import Pool, cpu_count

        total_eval_time = 0
        total_eval_steps = 0
        total_start_time = time.time()
        self.num_generations = num_generations

        mp.set_start_method("spawn", force=True)   # do once at program entry
        num_cores = alloc_cores()




        print('GA CONFIGURATION:')
        print(f'├─ Generations: {num_generations}')
        print(f'├─ Population Size: {population_size}')
        print(f"└─ Number of CPU cores: {num_cores}")
        print('═' * 80)
        print()

        # --- run the GA exactly as before ---
        if self.PP:

            ctx = get_context("spawn")
            with ctx.Pool(processes=num_cores) as pool:
                toolbox.register("map", pool.map)      # DEAP actually goes parallel
                self._run_genetic_algorithm(
                    population,
                    toolbox,
                    num_generations,
                    requantize=lambda ind: ind,   # no requantization by default
                    start_gen=start_gen,
                    checkpoint_manager=checkpoint_manager,
                    output_interval=output_interval,
                )
        else:
            self._run_genetic_algorithm(
                population,
                toolbox,
                num_generations,
                requantize=lambda ind: ind,
                start_gen=start_gen,
                checkpoint_manager=checkpoint_manager,
                output_interval=output_interval,
            )

        total_time = time.time() - total_start_time
        if total_eval_steps > 0:
            eff_avg_eval_time = total_time / total_eval_steps
            overall_avg_eval_time = total_eval_time / total_eval_steps
            print(f"Overall average evaluation time per individual: {overall_avg_eval_time:.4f} s")
            print(f"Effective overall average evaluation time per individual: {eff_avg_eval_time:.4f} s")
        else:
            print("No evaluations were performed.")

        gc.collect()

        self.export_ga_samples()

        # --- NEW: SMC+DE-MCMC refinement stage ---
        print("\n[smc-demc] Starting Sequential Monte Carlo refinement using the final GA ensemble...")
        smc_products = self.run_smc_demc_stage(
            population=population,
            toolbox=toolbox,
            ess_trigger=0.60,         # resample when ESS/N < 0.60
            moves_per_stage=3,        # DE–MH steps per stage
            big_step_every=6,         # γ≈1 sweep every k stages
            nsamples=200_000,         # rows sampled after burn-in thin/resample
            burn_frac=0.20            # discard first 20% stages
        )
        self.smc_demc_products = smc_products
        print("[smc-demc] Finished. Refinement artefacts written under:", self.output_path)

    def evaluate(self, individual):
        # Extract parameters from the individual
        # Categorical parameters (indices)
        comp_idx = int(individual[0])
        imf_idx = int(individual[1])
        sn1a_idx = int(individual[2])
        sy_idx = int(individual[3])
        sn1ar_idx = int(individual[4])
        
        # Continuous parameters
        sigma_2 = individual[5]
        t_1 = individual[6]
        t_2 = individual[7]
        infall_1 = individual[8]
        infall_2 = individual[9]
        sfe_val = individual[10]
        delta_sfe_val = individual[11]
        imf_upper = individual[12]
        mgal = individual[13]
        nb = individual[14]
        
        # Look up the actual values for categorical parameters
        comp = self.comp_array[comp_idx]
        imf_val = self.imf_array[imf_idx]
        sn1a = self.sn1a_assumptions[sn1a_idx]
        sy = self.stellar_yield_assumptions[sy_idx]
        sn1ar = self.sn1a_rates[sn1ar_idx]
        
        A1 = self.A1
        A2 = self.A2
        sn1a_header = self.sn1a_header
        iniab_header = self.iniab_header


        # --- deterministic, repair-first dt_in builder ---
        N_total = self.timesteps
        T_total = 13.0e9  # yr

        t1 = t_1 * 1e9;   t2 = t_2 * 1e9
        tau1 = max(infall_1 * 1e9, 1e4)  # avoid zeros
        tau2 = max(infall_2 * 1e9, 1e4)

        # windows: [t_i - 0.2 τ_i, t_i + 3 τ_i], clamped to domain
        w1_lo, w1_hi = max(0.0, t1 - 0.2*tau1), min(T_total, t1 + 3.0*tau1)
        w2_lo, w2_hi = max(0.0, t2 - 0.2*tau2), min(T_total, t2 + 3.0*tau2)

        # merge if they overlap
        if w1_hi > w2_lo:
            w2_lo, w2_hi = min(w1_lo, w2_lo), max(w1_hi, w2_hi)
            w1_lo, w1_hi = 0.0, 0.0  # first window empty

        # segment caps
        dt_min      = 2.0e6          # 2 Myr
        cap_hi1     = min(0.1*tau1, 3.0e7)
        cap_hi2     = min(0.1*tau2, 3.0e7)
        cap_mid     = 1.5e8
        cap_tail    = 2.5e8

        # segments: [0,w1_lo],[w1_lo,w1_hi],[w1_hi,w2_lo],[w2_lo,w2_hi],[w2_hi,T_total]
        segs = []
        if w1_hi > w1_lo:
            segs.append((0.0,    w1_lo, 'mid',  0.05))
            segs.append((w1_lo,  w1_hi, 'hi1',  None))
        else:
            segs.append((0.0,    w2_lo, 'mid',  0.10))
        segs.append((w1_hi, w2_lo, 'mid', 0.15))
        if w2_hi > w2_lo:
            segs.append((w2_lo,  w2_hi, 'hi2',  None))
        segs.append((w2_hi, T_total,'tail',0.30))

        # choose per-segment caps
        def cap_for(kind):
            return {'hi1':cap_hi1, 'hi2':cap_hi2, 'mid':cap_mid, 'tail':cap_tail}[kind]

        # initial integer allocation that WILL sum to N_total
        # rule: ≥30 steps in each hi window (if present); distribute the rest by duration
        N = []
        duration_total = sum(max(0.0, t1 - t0) for (t0,t1,_,_) in segs)
        baseline = N_total

        # reserve hi-window minima first
        hi_min = 0
        for (t0,t1,kind,_) in segs:
            dur = max(0.0, t1 - t0)
            if kind in ('hi1','hi2') and dur > 0:
                N.append(30); hi_min += 30; baseline -= 30
            else:
                N.append(0)

        # distribute the remaining steps proportionally by duration (rounded)
        if baseline < 0: baseline = 0
        durs = [max(0.0, t1 - t0) for (t0,t1,_,_) in segs]
        weights = [(d/duration_total if duration_total>0 else 0.0) for d in durs]
        extra = [int(round(baseline*w)) for w in weights]
        # fix rounding drift to hit exactly N_total
        drift = (hi_min + sum(extra)) - N_total
        # adjust extras by subtracting/adding 1 where it hurts least
        idxs = sorted(range(len(extra)), key=lambda i: durs[i], reverse=(drift>0))
        for i in idxs:
            if drift == 0: break
            if drift > 0 and extra[i] > 0:
                extra[i] -= 1; drift -= 1
            elif drift < 0:
                extra[i] += 1; drift += 1

        # final per-segment counts
        for i in range(len(N)):
            N[i] += extra[i]
            # ensure at least 1 if segment has duration
            if durs[i] > 0 and N[i] < 1: N[i] = 1

        # now build dt for each segment with cap/floor, then renormalize each segment to its exact duration
        parts = []
        for (t0,t1,kind,_), n in zip(segs, N):
            dur = max(0.0, t1 - t0)
            if dur == 0.0 or n == 0:
                continue
            cap = cap_for(kind)
            raw = np.full(n, max(dt_min, min(cap, dur/max(1,n))), float)
            # sum(raw) may not equal dur; rescale uniformly to match exactly
            scale = dur / raw.sum()
            raw *= scale
            # after rescale, enforce dt_min by borrowing uniformly from larger bins
            below = raw < dt_min
            if below.any():
                deficit = dt_min*below.sum() - raw[below].sum()
                raw[below] = dt_min
                # take evenly from non-below bins
                nb = (~below).sum()
                if nb > 0:
                    raw[~below] -= deficit/nb
                # if any went sub-min due to borrow, clip and renormalize again
                raw = np.clip(raw, dt_min, None)
                raw *= dur / raw.sum()
            parts.append(raw)

        custom_dt_in = np.concatenate(parts)

        # final rounding repair to hit N_total exactly (rare off-by-one)
        if len(custom_dt_in) > N_total:
            # merge the last few tiny bins
            extra = len(custom_dt_in) - N_total
            custom_dt_in[-(extra+1)] += custom_dt_in[-extra:].sum()
            custom_dt_in = custom_dt_in[:N_total]
        elif len(custom_dt_in) < N_total:
            pad = np.full(N_total-len(custom_dt_in), custom_dt_in[-1], float)
            custom_dt_in = np.concatenate([custom_dt_in, pad])

        # final guarantee: exact sum & no micro last-bin
        custom_dt_in *= (T_total / custom_dt_in.sum())
        if custom_dt_in[-1] < dt_min and len(custom_dt_in) > 1:
            custom_dt_in[-2] += custom_dt_in[-1] - dt_min
            custom_dt_in[-1]  = dt_min


        kwargs = {
            'special_timesteps': len(custom_dt_in),
            'dt_in': custom_dt_in,
            'tend': float(custom_dt_in.sum()),
            'twoinfall_sigmas': [1300, sigma_2],
            'galradius': 1800,
            'exp_infall': [[A1, t_1*1e9, infall_1*1e9],
                           [A2, t_2*1e9, infall_2*1e9]],
            'substeps': [2,4,8,12,16,24,32,48,64,96,128,192,256],
            'tolerance': 1e-5,
            'tauup': [0.1*infall_1*1e9, 0.1*infall_2*1e9],
            'mgal': mgal, 'iniZ': 0.0, 'mass_loading': 0.0,
            'table': sn1a_header + sy,
            'sfe': sfe_val, 'delta_sfe': delta_sfe_val,
            'imf_type': imf_val,
            'sn1a_table': sn1a_header + sn1a,
            'imf_yields_range': [1, imf_upper],
            'iniabu_table': iniab_header + comp,
            'nb_1a_per_m': nb, 'sn1a_rate': sn1ar
        }


        # Run GCE model and compute MDF
        GCE_model = omega_plus.omega_plus(**kwargs)
        MDF_x_data, MDF_y_data = GCE_model.inner.plot_mdf(axis_mdf='[Fe/H]', sigma_gauss=0.1, norm=True, return_x_y=True)
        MDF_x_data = np.array(MDF_x_data)
        MDF_y_data = np.array(MDF_y_data)


        elements = ['[Si/Fe]','[Ca/Fe]','[Mg/Fe]','[Ti/Fe]']
        alpha_arrs = []
        for el in elements:
            alpha_x_data, alpha_y_data = GCE_model.inner.plot_spectro(xaxis='[Fe/H]', yaxis=el, return_x_y=True)
            alpha_arrs.append([np.array(alpha_x_data), np.array(alpha_y_data)])


        age_x_data, age_y_data=GCE_model.inner.plot_spectro(xaxis='age', yaxis='[Fe/H]', return_x_y=True)
        age_x_data = np.array(age_x_data)
        age_y_data = np.array(age_y_data)

        # Evaluate the spline at the same [Fe/H] grid as your data
        cs_MDF = CubicSpline(MDF_x_data, MDF_y_data)
        #fmin, fmax = MDF_x_data.min(), MDF_x_data.max()
        #feh_clamped = np.clip(self.feh, fmin, fmax)
        #theory_count_array = cs_MDF(feh_clamped)



        # Sort (safety), clamp, interpolate without overshoot
        order = np.argsort(MDF_x_data)
        x = np.asarray(MDF_x_data)[order]
        y = np.clip(np.asarray(MDF_y_data)[order], 0, None)    # no negatives

        interp = PchipInterpolator(x, y, extrapolate=False)

        fmin, fmax = x[0], x[-1]
        feh_clamped = np.clip(self.feh, fmin, fmax)

        theory = interp(feh_clamped)
        theory = np.clip(theory, 0, None)

        # match the observational normalization convention (your data are count/max)
        m = theory.max()
        theory_count_array = theory / m #if m > 0 else theory




        # Compare with the observed distribution
        ks, ensemble, wrmse, mae, mape, huber, cos_similarity, log_cosh = calculate_all_metrics(self, theory_count_array)

        penalty_factor = 1.0
        obs_age_loss_value = 1.0
        primary_loss_value = 1.0


        # Use selected loss
        primary_loss_value = self.selected_loss_function(self,theory_count_array)

        if self.obs_age_data_loss_metric is not None:
            obs_age_loss_value = age_meta_loss(self, age_x_data, age_y_data, self.obs_age_data, self.obs_age_data_loss_metric, dataset=self.obs_age_data_target)
            primary_loss_value = (obs_age_loss_value * (1.0 - self.mdf_vs_age_weight)) + (primary_loss_value * self.mdf_vs_age_weight)


        if self.physical_constraints_freq > 0:
            # Apply physics penalty
            if self.physics_timer < self.physical_constraints_freq:
                self.physics_timer = self.physics_timer + 1

            else:

                self.physics_timer = 0
                penalty_factor = apply_physics_penalty(
                    primary_loss_value, 
                    MDF_x_data, MDF_y_data, 
                    alpha_arrs, 
                    age_x_data, age_y_data
                )
                primary_loss_value = primary_loss_value * penalty_factor


        # Return the result with a detailed label
        label = (f'comp: {comp}, imf: {imf_val}, sn1a: {sn1a}, sy: {sy}, sn1ar: {sn1ar}, '
                 f'sigma2: {sigma_2:.3f}, t1: {t_1:.3f}, t2: {t_2:.3f}, '
                 f'infall1: {infall_1:.3f}, infall2: {infall_2:.3f}, '
                 f'sfe: {sfe_val:.5f}, delta_sfe: {delta_sfe_val:.3f}, imf_upper: {imf_upper:.1f}, '
                 f'mgal: {mgal:.2e}, nb: {nb:.2e}')
                 
        # Create metrics list for results storage.  Include the final
        # fitness value (after physics penalty) so it can be tracked
        # alongside the other loss metrics.
        metrics = [
            comp_idx, imf_idx, sn1a_idx, sy_idx, sn1ar_idx,
            sigma_2, t_1, t_2, infall_1, infall_2,
            sfe_val, delta_sfe_val, imf_upper, mgal, nb,
            ks, ensemble, wrmse, mae, mape, huber,
            cos_similarity, log_cosh, primary_loss_value, obs_age_loss_value, penalty_factor
        ]

        result = {
            'label': label,
            'MDF_x_data': MDF_x_data,
            'MDF_y_data': MDF_y_data,
            'age_x_data': age_x_data,
            'age_y_data': age_y_data,
            'alpha_arrs': alpha_arrs,
            'metrics': metrics,
            'fitness': primary_loss_value,
            'cs_MDF': cs_MDF,
            'model_number': self.model_count
        }

        return (primary_loss_value,), result

    def _record_evaluation_result(self, result):
        """Store side-effects from a successful model evaluation."""
        if result is None:
            return

        self.labels.append(result.get('label'))
        self.mdf_data.append([result.get('MDF_x_data'), result.get('MDF_y_data')])
        self.alpha_data.append(result.get('alpha_arrs'))
        self.age_data.append([result.get('age_x_data'), result.get('age_y_data')])

        metrics = result.get('metrics')
        if metrics is not None:
            self.results.append(metrics)
            eval_index = self.model_count
            if len(metrics) == len(self.metric_header):
                sample = {name: value for name, value in zip(self.metric_header, metrics)}
                sample['generation'] = getattr(self, 'gen', -1)
                sample['evaluation'] = eval_index
                if 'fitness' in sample:
                    sample['loss'] = sample['fitness']
                self.sample_records.append(sample)
        else:
            self.results.append([None] * len(self.metric_header))

        self.MDFs.append(result.get('cs_MDF'))
        self.model_numbers.append(result.get('model_number'))
        self.model_count += 1

    def run_smc_demc_stage(self,
                           population,
                           toolbox,
                           ess_trigger=0.60,
                           moves_per_stage=3,
                           big_step_every=6,
                           nsamples=200_000,
                           burn_frac=0.20):
        """
        Tempered SMC with Differential-Evolution Metropolis moves, executed as the
        refinement phase of the GA pipeline.

        Produces CSV artefacts:
          - {output_path}/chains.csv
          - {output_path}/smc_demc_samples.csv (and legacy posterior_samples.csv)
        Returns a dictionary with the written artefacts and in-memory draws.
        """
        import os
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(42)

        categorical_names = [
            'comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx'
        ]
        param_names = [
            'sigma_2', 'tmax_1', 'tmax_2', 'infall_timescale_1', 'infall_timescale_2',
            'sfe', 'delta_sfe', 'imf_upper_limits', 'mgal_values', 'nb_array'
        ]

        start_idx = 5
        end_idx = 15

        X0 = np.vstack([
            np.array(ind[start_idx:end_idx], dtype=float)
            for ind in population
        ])

        categorical0 = np.vstack([
            np.array(ind[:start_idx], dtype=int)
            for ind in population
        ])

        bounds = [
            Bound(*self.get_param_bounds(start_idx + j))
            for j in range(X0.shape[1])
        ]

        base_template = toolbox.clone(population[0])

        def loss_from_vector(theta, cat_vals=None):
            ind = toolbox.clone(base_template)
            if cat_vals is None:
                cat_vals = categorical0[0]
            for idx, val in enumerate(cat_vals):
                ind[idx] = int(val)
            ind[start_idx:end_idx] = list(theta)
            if hasattr(ind.fitness, 'values'):
                del ind.fitness.values
            fit, _ = toolbox.evaluate(ind)
            return float(fit[0])





        import multiprocessing as mp

        Ncores = alloc_cores()
        self.demc_workers = min(Ncores, X0.shape[0])   # not more threads than walkers

        if mp.current_process().name != "MainProcess":
            raise RuntimeError("DEMC must run only in the coordinator process")

        ensemble, chains_df = run_smc_demc(
            X0,
            loss_from_vector,
            bounds,
            metadata0=categorical0,
            ess_trigger=ess_trigger,
            moves_per_stage=moves_per_stage,
            rng=rng,
            gamma_schedule=(None, 1.0),
            big_step_every=big_step_every,
            max_workers=self.demc_workers,          # <- uses all available cores
        )


        self.refined_population = ensemble.copy()

        os.makedirs(self.output_path, exist_ok=True)

        meta_cols = [f"m{j}" for j in range(categorical0.shape[1])]
        param_cols = [f"p{j}" for j in range(ensemble.shape[1])]
        rename_map = {
            **{old: new for old, new in zip(meta_cols, categorical_names)},
            **{old: new for old, new in zip(param_cols, param_names)},
        }
        chains_df.rename(columns=rename_map, inplace=True)

        max_stage = int(chains_df["stage"].max()) if len(chains_df) else 0
        burn_cut = int(np.floor(max_stage * burn_frac))
        kept = chains_df[chains_df["stage"] >= burn_cut]

        if len(kept) > 0:
            per_pid = max(1, nsamples // max(1, kept["pid"].nunique()))
            sample_parts = []
            for pid, group in kept.groupby("pid"):
                take = min(per_pid, len(group))
                sample_parts.append(
                    group.sample(n=take, replace=(take > len(group)), random_state=0)
                )
            samples = pd.concat(sample_parts, axis=0, ignore_index=True)
            samples = samples[param_names].reset_index(drop=True)
        else:
            samples = pd.DataFrame(columns=param_names)

        chains_path = os.path.join(self.output_path, "chains.csv")
        chains_df.to_csv(chains_path, index=False)
        samples_path = os.path.join(self.output_path, "smc_demc_samples.csv")

        samples.to_csv(samples_path, index=False)
        legacy_samples_path = os.path.join(self.output_path, "posterior_samples.csv")
        if legacy_samples_path != samples_path:
            samples.to_csv(legacy_samples_path, index=False)

        corner_path = None
        if not samples.empty:
            corner_labels = [
                "σ₂",
                "t₁",
                "t₂",
                "τ₁",
                "τ₂",
                "SFE",
                "ΔSFE",
                "IMF₍max₎",
                "Mgal",
                "N₍B₎",
            ]
            try:
                fig = corner.corner(
                    samples.to_numpy(),
                    labels=corner_labels,
                    show_titles=True,
                    title_fmt=".3f",
                    quantiles=[0.16, 0.50, 0.84],
                    color="black",
                    plot_datapoints=False,
                    fill_contours=True,
                    hist_kwargs={"histtype": "stepfilled", "alpha": 0.35, "edgecolor": "black"},
                    contour_kwargs={"linewidths": 1.0},
                    contourf_kwargs={"cmap": "Greys"},   # grayscale fills, darker in the center
                    label_kwargs={"fontsize": 12},
                    title_kwargs={"fontsize": 11},
                )
            except Exception as exc:
                print(f"[smc-demc] corner plot failed: {exc}")
            else:
                corner_path = os.path.join(self.output_path, "smc_demc_posterior_corner.png")
                fig.savefig(corner_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"[smc-demc] wrote {corner_path}")

        print(f"[smc-demc] wrote {chains_path}")
        print(f"[smc-demc] wrote {samples_path}")
        if legacy_samples_path != samples_path:
            print(f"[smc-demc] wrote {legacy_samples_path}")

        return {
            "ensemble": ensemble,
            "chains": chains_df,
            "chains_path": chains_path,
            "samples": samples,
            "samples_path": samples_path,
            "legacy_samples_path": legacy_samples_path,
            "corner_path": corner_path,
        }


    def export_ga_samples(self):
        """Persist the GA evaluation history as a sampling-friendly CSV."""
        if not self.sample_records:
            return None

        df = pd.DataFrame(self.sample_records)
        if 'loss' not in df.columns and 'fitness' in df.columns:
            df['loss'] = df['fitness']

        base_cols = [col for col in ('generation', 'evaluation') if col in df.columns]
        other_cols = [c for c in df.columns if c not in base_cols]
        df = df[base_cols + other_cols]

        os.makedirs(self.output_path, exist_ok=True)
        path = os.path.join(self.output_path, 'ga_population_samples.csv')
        df.to_csv(path, index=False)
        self.ga_samples_path = path
        print(f"[ga-sampler] wrote {path} ({len(df)} rows)")
        return path


    def apply_demc_hybrid_moves(self, population, toolbox):
        """Apply Differential Evolution MCMC moves to a subset of the population."""
        if not self.demc_hybrid or len(population) < 3:
            return

        move_count = int(np.ceil(self.demc_fraction * len(population)))
        move_count = max(3, min(len(population), move_count))
        if move_count <= 0:
            return

        indices = self.demc_rng.choice(len(population), size=move_count, replace=False)

        start_idx = 5
        end_idx = 15

        X = np.vstack([
            np.array(population[i][start_idx:end_idx], dtype=float)
            for i in indices
        ])

        metadata = np.vstack([
            np.array(population[i][:start_idx], dtype=int)
            for i in indices
        ])

        bounds = [
            Bound(*self.get_param_bounds(start_idx + j))
            for j in range(X.shape[1])
        ]

        base_template = toolbox.clone(population[0])
        eval_cache = {}

        def cache_key(cat_vals, theta):
            cat_tuple = tuple(int(v) for v in np.asarray(cat_vals).ravel())
            theta_bytes = np.asarray(theta, dtype=np.float64).tobytes()
            return cat_tuple, theta_bytes

        def loss_from_vector(theta, cat_vals):
            key = cache_key(cat_vals, theta)
            if key not in eval_cache:
                ind = toolbox.clone(base_template)
                for idx, val in enumerate(cat_vals):
                    ind[idx] = int(val)
                ind[start_idx:end_idx] = list(theta)
                if hasattr(ind.fitness, 'values'):
                    del ind.fitness.values
                eval_cache[key] = toolbox.evaluate(ind)
            fit, _ = eval_cache[key]
            return float(fit[0])

        def loglike(theta, meta=None):
            return -loss_from_vector(theta, meta)

        X_new, accepted = de_mh_move(
            X,
            loglike,
            bounds,
            metadata=metadata,
            steps=self.demc_moves_per_gen,
            gamma=self.demc_gamma,
            rng=self.demc_rng,
            max_workers=self.demc_workers,
        )

        accepted_indices = np.where(accepted)[0]
        if accepted_indices.size == 0:
            return

        for local_idx in accepted_indices:
            pop_idx = indices[local_idx]
            theta = X_new[local_idx]
            cat_vals = metadata[local_idx]
            key = cache_key(cat_vals, theta)
            fit, result = eval_cache.get(key, (None, None))
            if fit is None:
                ind_tmp = toolbox.clone(base_template)
                for idx, val in enumerate(cat_vals):
                    ind_tmp[idx] = int(val)
                ind_tmp[start_idx:end_idx] = list(theta)
                if hasattr(ind_tmp.fitness, 'values'):
                    del ind_tmp.fitness.values
                fit, result = toolbox.evaluate(ind_tmp)

            individual = population[pop_idx]
            for idx, val in enumerate(cat_vals):
                individual[idx] = int(val)
            individual[start_idx:end_idx] = list(theta)
            individual.fitness.values = fit

            self._record_evaluation_result(result)
            self.walker_history.setdefault(pop_idx, []).append(list(individual))

        print(f"[demc-hybrid] updated {accepted_indices.size} walkers via DE-MC proposals")


    def _run_genetic_algorithm(
        self,
        population,
        toolbox,
        num_generations,
        requantize,
        start_gen=0,
        checkpoint_manager=None,
        output_interval=None,
    ):
        """
        GA main loop with small elitism:
          - evaluate invalid
          - pick k elites (protected)
          - select/mate/mutate the remaining (len(pop)-k)
          - optional quantize + de-dup
          - evaluate invalid
          - replace population = elites ⊕ children
          - (optional) update operator rates, checkpoint, partial results
        """
        if not hasattr(self, 'walker_history') or start_gen == 0:
            self.walker_history = {i: [] for i in range(len(population))}

        # small, fixed elitism unless user set self.elitism_k
        base_k = max(1, len(population) // 16)  # ~6%
        elitism_k = max(1, int(getattr(self, 'elitism_k', base_k)))

        for gen in range(start_gen, num_generations):
            print(f"-- =================== --")
            print(f"-- Generation {gen}/{num_generations} --")
            self.gen = gen

            # ---------- Step 1: evaluate invalid in current population ----------
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            if invalid_ind:
                if self.PP:
                    fitnesses_and_results = toolbox.map(toolbox.evaluate, invalid_ind)
                else:
                    fitnesses_and_results = [toolbox.evaluate(ind) for ind in invalid_ind]

                for (ind, (fit, result)) in zip(invalid_ind, fitnesses_and_results):
                    ind.fitness.values = fit
                    self._record_evaluation_result(result)

            gc.collect()

            # ---------- Step 2: pick elites (PROTECTED) ----------
            elites = tools.selBest(population, elitism_k)
            elites = [toolbox.clone(e) for e in elites]
            for e in elites:
                # force re-eval flag off; we keep the elite genomes intact and their fitness as-is
                # (no del e.fitness.values)
                pass

            # ---------- Step 3: select parents for breeding (rest of pop) ----------
            # full selection for pressure; then take the needed count for children
            mating_pool = toolbox.select(population)
            mating_pool = list(map(toolbox.clone, mating_pool))
            # keep only the number we need to refill to full size
            needed_children = len(population) - elitism_k
            breed_pool = mating_pool[:max(needed_children, 0)]

            # Targeted improvement for very poor parents - no poors allowed
            if len(population) > 0:
                best_walker = tools.selBest(population, 1)[0]
                best_clone = toolbox.clone(best_walker)
                for p in breed_pool:
                    if p.fitness.valid and p.fitness.values[0] > 100.0:
                        # two nudges + biased mate with best
                        toolbox.mutate(p); toolbox.mutate(p)
                        child, _ = toolbox.mate(p, best_clone)
                        p[:] = child
                        if hasattr(p.fitness, 'values'):
                            del p.fitness.values

            # ---------- Step 4: crossover then mutation (children only; elites protected) ----------
            offspring = list(map(toolbox.clone, breed_pool))

            # crossover (pairwise)
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    toolbox.mate(c1, c2)
                    if hasattr(c1.fitness, 'values'): del c1.fitness.values
                    if hasattr(c2.fitness, 'values'): del c2.fitness.values

            # mutation
            for m in offspring:
                if random.random() < self.mutpb:
                    toolbox.mutate(m)
                    if hasattr(m.fitness, 'values'): del m.fitness.values

            # optional quantization (children only)
            if self.quant_individuals:
                offspring = [requantize(ind) for ind in offspring]

            # de-duplicate CHILDREN ONLY; elites remain unchanged
            offspring = self.prevent_duplicates(offspring, toolbox)

            # ensure we have exactly the needed number of children
            if len(offspring) > needed_children:
                offspring = offspring[:needed_children]
            elif len(offspring) < needed_children:
                # pad with best clones if de-dup trimmed too far
                fillers = tools.selBest(population, needed_children - len(offspring))
                offspring += [toolbox.clone(f) for f in fillers]

            # ---------- Step 5: evaluate invalid children ----------
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                if self.PP:
                    fitnesses_and_results = toolbox.map(toolbox.evaluate, invalid_ind)
                else:
                    fitnesses_and_results = [toolbox.evaluate(ind) for ind in invalid_ind]

                for (ind, (fit, result)) in zip(invalid_ind, fitnesses_and_results):
                    ind.fitness.values = fit
                    self._record_evaluation_result(result)

            # ---------- Step 6: record history before replacement ----------
            for idx, ind in enumerate(population):
                self.walker_history[idx].append(list(ind))

            # ---------- Step 7: replace population = elites ⊕ offspring ----------
            new_population = elites + offspring
            population[:] = new_population  # size preserved

            # ---------- Step 7b: optional DE-MCMC refinement on the living population ----------
            self.apply_demc_hybrid_moves(population, toolbox)

            # ---------- Step 8: (optional) adaptive operator rates ----------
            self.update_operator_rates(population, gen, num_generations)

            # ---------- Step 9: checkpoint + periodic results ----------
            if checkpoint_manager:
                checkpoint_manager.save(gen, population, self)
            else:
                print('checkpoint_manager missing')
                exit()

            if output_interval and ((gen) % output_interval == 0 or gen == num_generations - 1):
                self.save_partial_results(gen)

            gc.collect()


    def save_partial_results(self, generation):
        """Save results and generate plots for the current generation."""
        df = pd.DataFrame(self.results, columns=self.metric_header)
        df['loss'] = df[self.loss_metric]
        df.sort_values('loss', inplace=True)
        df.reset_index(drop=True, inplace=True)

        os.makedirs(self.output_path, exist_ok=True)
        results_file = self.output_path + f"simulation_results_gen_{generation}.csv"
        df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")

        mdf_plotting.generate_all_plots(self, self.feh, self.normalized_count, results_file)







