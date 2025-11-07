#!/usr/bin/env python3.8
################################
# Author: N Miller, M Joyce
################################

import sys
sys.stdout.reconfigure(line_buffering=True)

import matplotlib.pyplot as plt
import warnings
import numpy as np
import sys
import argparse
from scipy.interpolate import CubicSpline
from deap import base, creator, tools
import random
import Gal_GA_PP as Gal_GA
import pandas as pd
import os

for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")

os.environ.setdefault("MPLBACKEND", "Agg")

import checkpoint
import plotting.mdf_plotting as mdf_plotting
from multiprocessing import cpu_count
import numpy as _np, random as _random, os as _os
import os, shutil

from types import SimpleNamespace



def get_latest_csv(path):
    """
    Find the latest results CSV in the folder:
    - Prefer 'simulation_results.csv' if exists
    - Else, find the highest 'simulation_results_gen_XX.csv'
    Returns full path or None if no matching files.
    """
    files = [f for f in os.listdir(path) if f.startswith('simulation_results') and f.endswith('.csv')]
    if not files:
        return None

    if 'simulation_results.csv' in files:
        return os.path.join(path, 'simulation_results.csv')

    # Parse gen numbers
    gens = []
    for f in files:
        if '_gen_' in f:
            try:
                gen_str = f.split('_gen_')[1].split('.csv')[0]
                gen = int(gen_str)
                gens.append((gen, f))
            except ValueError:
                pass

    if not gens:
        return None

    max_gen, max_f = max(gens)
    return os.path.join(path, max_f)





def load_bensby_data(file_path='data/Bensby_Data.tsv'):
    obs_age_data = pd.read_csv(file_path, sep='\t')
    print(f"Loaded Bensby data with shape: {obs_age_data.shape}")
    print(f"Columns available: {list(obs_age_data.columns)}")
    return obs_age_data




def save_walker_history():
    if not hasattr(GalGA, 'walker_history'):
        return
    
    np.savez_compressed(
        os.path.join(output_path, 'walker_history.npz'),
        walker_ids=np.array(list(GalGA.walker_history.keys()), dtype=np.int32),
        histories=np.array([np.array(h) for h in GalGA.walker_history.values()], dtype=object),
        mdf_data=np.array(GalGA.mdf_data, dtype=object),
        alpha_data=np.array(GalGA.alpha_data, dtype=object),
        age_data=np.array(getattr(GalGA, 'age_data', []), dtype=object)
    )
    
    print("Walker history saved")

def load_walker_history():
    history_path = os.path.join(output_path, 'walker_history.npz')
    
    data = np.load(history_path, allow_pickle=True)
    walker_ids = data['walker_ids']
    histories = data['histories']
    
    walker_history = {}
    for i, walker_id in enumerate(walker_ids):
        walker_history[int(walker_id)] = histories[i]
    
    print("Walker history loaded")
    return walker_history



















def run_ga(cp_manager):
    global GalGA
    import numpy as _np
    
    GalGA = Gal_GA.GalacticEvolutionGA(
        output_path=output_path,
        iniab_header=iniab_header,
        sn1a_header=sn1a_header,
        sigma_2_list=sigma_2_list,
        tmax_1_list=tmax_1_list,
        tmax_2_list=tmax_2_list,
        infall_timescale_1_list=infall_timescale_1_list,
        infall_timescale_2_list=infall_timescale_2_list,
        comp_array=comp_array,
        imf_array=imf_array,
        sfe_array=sfe_array,
        delta_sfe_array=delta_sfe_array,
        imf_upper_limits=imf_upper_limits,
        sn1a_assumptions=sn1a_assumptions,
        stellar_yield_assumptions=stellar_yield_assumptions,
        mgal_values=mgal_values,
        nb_array=nb_array,
        sn1a_rates=sn1a_rates,
        timesteps=timesteps,
        A1=A1,
        A2=A2,
        feh=feh,
        normalized_count=normalized_count,
        obs_age_data=obs_age_data,
        loss_metric=loss_metric,
        obs_age_data_loss_metric=obs_age_data_loss_metric,
        obs_age_data_target=obs_age_data_target,
        mdf_vs_age_weight=mdf_vs_age_weight,
        fancy_mutation=fancy_mutation,
        shrink_range=shrink_range,
        gaussian_sigma_scale=gaussian_sigma_scale,
        crossover_noise_fraction=crossover_noise_fraction,
        perturbation_strength=perturbation_strength,
        tournament_size=tournament_size,
        threshold=selection_threshold,
        cxpb=crossover_probability,
        mutpb=mutation_probability,
        physical_constraints_freq=physical_constraints_freq,
        exploration_steps=exploration_steps,
        PP=True,
        demc_hybrid=True,
        demc_fraction=demc_fraction,
        demc_moves_per_gen=1,
        demc_gamma=None,
        demc_rng_seed=None
    )
    
    init_population, toolbox = GalGA.init_GenAl(population_size=popsize)
    
    def _invalidate(ind):
        if getattr(ind.fitness, "valid", False):
            del ind.fitness.values
    
    def _tiny_jitter(ind, frac=1e-3):
        for gi in range(5, len(ind)):
            x = float(ind[gi])
            span = max(abs(x), 1.0) * frac
            ind[gi] = x + _np.random.normal(0.0, span)
    
    cp_data = cp_manager.load()
    start_gen = 0
    population = None
    num_generations = generations
    
    if cp_data:
        cp_gen = int(cp_data.get("generation", -1))
        ga_state = dict(cp_data.get("ga_state", {}))
        full_pop = list(cp_data.get("population", []) or [])
        
        #GalGA.__dict__.update(ga_state)
        GalGA.checkpoint_population = full_pop[:]
        
        def _fit(ind):
            if getattr(ind.fitness, "valid", False) and hasattr(ind.fitness, "values"):
                return float(ind.fitness.values[0])
            return float("inf")
        
        if len(full_pop) >= popsize:
            ranked = sorted(full_pop, key=_fit)
            print(f'reducing from {len(full_pop)} to {popsize}')
            population = ranked[:popsize]
        else:
            ranked = sorted(full_pop, key=_fit)
            population = ranked[:]
            if population:
                seed = population[0]
                while len(population) < popsize:
                    clone = toolbox.clone(seed)
                    for gi in range(5, len(clone)):
                        xv = float(clone[gi])
                        span = max(abs(xv), 1.0) * 1e-4
                        clone[gi] = xv + _np.random.normal(0.0, span)
                    if getattr(clone.fitness, "valid", False):
                        del clone.fitness.values
                    population.append(clone)
            else:
                population = init_population
        
        GalGA.walker_history = {i: [] for i in range(len(population))}
        
        start_gen = cp_gen + 1
        if start_gen >= num_generations:
            num_generations = start_gen + 1
            print(f"Extending generations to {num_generations} to ensure ≥1 generation runs after resume.")
        
    else:
        # Seed from prior results CSV in current output_path (pcard still governs everything)
        csv_path = os.path.join(output_path, "simulation_results.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)

            # Prefer 'loss' if present, else 'fitness', else keep existing order
            if 'loss' in df.columns:
                df = df.sort_values('loss')
            elif 'fitness' in df.columns:
                df = df.sort_values('fitness')

            # Gene columns (0..14) in your results schema
            # comp_idx, imf_idx, sn1a_idx, sy_idx, sn1ar_idx,
            # sigma_2, t_1, t_2, infall_1, infall_2, sfe, delta_sfe, imf_upper, mgal, nb
            cols = [
                'comp_idx','imf_idx','sn1a_idx','sy_idx','sn1ar_idx',
                'sigma_2','t_1','t_2','infall_1','infall_2',
                'sfe','delta_sfe','imf_upper','mgal','nb'
            ]  # matches the front of your col_names list (see below)

            rows = df[cols].head(popsize).to_numpy()

            population = []
            template = init_population[0]
            for r in rows:
                ind = toolbox.clone(template)
                for gi, val in enumerate(r):
                    ind[gi] = int(val) if gi < 5 else float(val)  # first 5 categorical, rest float
                if getattr(ind.fitness, "valid", False):
                    del ind.fitness.values
                population.append(ind)

            # pad if needed: tiny jitter (your existing idiom)
            while len(population) < popsize:
                clone = toolbox.clone(population[0])
                for gi in range(5, len(clone)):
                    xv = float(clone[gi])
                    span = max(abs(xv), 1.0) * 1e-4
                    clone[gi] = xv + _np.random.normal(0.0, span)
                if getattr(clone.fitness, "valid", False):
                    del clone.fitness.values
                population.append(clone)

            GalGA.walker_history = {i: [] for i in range(len(population))}
            start_gen = 0
        else:
            population = init_population
            GalGA.walker_history = {i: [] for i in range(len(population))}
            start_gen = 0

    
    GalGA.GenAl(
        population_size=popsize,
        num_generations=num_generations,
        population=population,
        toolbox=toolbox,
        checkpoint_manager=cp_manager,
        start_gen=start_gen,
        output_interval=output_interval,
    )
    
    smc_products = getattr(GalGA, "smc_demc_products", None)
    if smc_products:
        print("SMC-DEMC refinement summary:")
        print(f"  Ensemble shape: {smc_products['ensemble'].shape}")
        print(f"  Chains log: {smc_products['chains_path']}")
        print(f"  Samples: {smc_products['samples_path']}")
        legacy_path = smc_products.get('legacy_samples_path')
        if legacy_path and legacy_path != smc_products['samples_path']:
            print(f"  Legacy samples mirror: {legacy_path}")
    
    col_names = [
        'comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx',
        'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2',
        'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb',
        'ks', 'ensemble', 'wrmse', 'mae', 'mape', 'huber',
        'cosine', 'log_cosh', 'EMD', 'fitness', 'age_meta_fitness', 'physics_penalty'
    ]
    
    results_df = pd.DataFrame(GalGA.results, columns=col_names) if GalGA.results else pd.DataFrame(columns=col_names)
    if 'loss' not in results_df.columns and not results_df.empty:
        results_df['loss'] = results_df[loss_metric]
        results_df.sort_values('loss', inplace=True)
        results_df.reset_index(drop=True, inplace=True)
    
    results_file = os.path.join(output_path, 'simulation_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")
    
    if not results_df.empty:
        best_model = results_df.iloc[0]
        print("Best model from results dataframe:")
        print(best_model)
    
    return results_file














def _plot_only(entry, bins=60):
    import os
    import numpy as np
    import pandas as pd
    from scipy.interpolate import CubicSpline
    
    import plotting.mdf_plotting as mdf_plotting
    

    if os.path.isdir(entry):
        outdir = os.path.abspath(entry)
        results_csv = get_latest_csv(outdir)
    else:
        results_csv = os.path.abspath(entry)
        outdir = os.path.dirname(results_csv) or "."


    
    pcard = os.path.join(outdir, "bulge_pcard.txt")
    npz_path = os.path.join(outdir, "walker_history.npz")
    
    params = Gal_GA.parse_inlist(pcard)
    obs_file = params["obs_file"]
    
    feh, count = np.loadtxt(obs_file, usecols=(0, 1), unpack=True)
    normalized_count = count / max(count.max(), 1.0)
    
    obs_age_data = load_bensby_data('data/Bensby_Data.tsv')
    
    GalGA = Gal_GA.GalacticEvolutionGA(
        output_path=outdir,
        iniab_header=params["iniab_header"],
        sn1a_header=params["sn1a_header"],
        sigma_2_list=params["sigma_2_list"],
        tmax_1_list=params["tmax_1_list"],
        tmax_2_list=params["tmax_2_list"],
        infall_timescale_1_list=params["infall_timescale_1_list"],
        infall_timescale_2_list=params["infall_timescale_2_list"],
        comp_array=params["comp_array"],
        imf_array=params["imf_array"],
        sfe_array=params["sfe_array"],
        delta_sfe_array=params["delta_sfe_array"],
        imf_upper_limits=params["imf_upper_limits"],
        sn1a_assumptions=params["sn1a_assumptions"],
        stellar_yield_assumptions=params["stellar_yield_assumptions"],
        mgal_values=params["mgal_values"],
        nb_array=params["nb_array"],
        sn1a_rates=params["sn1a_rates"],
        timesteps=params["timesteps"],
        A1=params["A1"],
        A2=params["A2"],
        feh=feh,
        normalized_count=normalized_count,
        obs_age_data=obs_age_data,
        loss_metric=params["loss_metric"],
        obs_age_data_loss_metric=params["obs_age_data_loss_metric"],
        obs_age_data_target=params["obs_age_data_target"],
        mdf_vs_age_weight=params["mdf_vs_age_weight"],
        fancy_mutation=params["fancy_mutation"],
        shrink_range=params["shrink_range"],
        gaussian_sigma_scale=params.get("gaussian_sigma_scale", 0.01),
        crossover_noise_fraction=params.get("crossover_noise_fraction", 0.05),
        perturbation_strength=params.get("perturbation_strength", 0.1),
        tournament_size=params["tournament_size"],
        threshold=params["selection_threshold"],
        cxpb=params["crossover_probability"],
        mutpb=params["mutation_probability"],
        physical_constraints_freq=params["physical_constraints_freq"],
        exploration_steps=params["exploration_steps"],
        PP=False,
        demc_hybrid=False,
        plot_mode="plot-only",
    )
    
    df = pd.read_csv(results_csv)
    cols = [c for c in GalGA.metric_header if c in df.columns]
    GalGA.results = df[cols].values.tolist()
    
    if "loss" not in df.columns and "fitness" in df.columns:
        df["loss"] = df["fitness"]
        df.to_csv(results_csv, index=False)
    
    GalGA.walker_history = {}
    GalGA.mdf_data = []
    GalGA.alpha_data = []
    GalGA.age_data = []
    GalGA.MDFs = []
    




    data = np.load(npz_path, allow_pickle=True)

    
    walker_ids = data.get("walker_ids")
    histories = data.get("histories")
    GalGA.walker_history = {int(wid): list(hist) for wid, hist in zip(walker_ids, histories)}

    
    mdf_arr = data.get("mdf_data")
    GalGA.mdf_data = [ (np.asarray(x), np.asarray(y)) for (x, y) in mdf_arr ]
    for x, y in GalGA.mdf_data:
        if len(x) >= 4 and len(x) == len(y):
            GalGA.MDFs.append(CubicSpline(x, np.clip(y, 0, None)))


    raw_alpha = data.get("alpha_data")
    clean_alpha = []
    for model_alpha in raw_alpha:
        elems = []
        for pair in model_alpha:
            # Expect pair like [feh_vec, alpha_vec]
            x = np.asarray(pair[0], dtype=np.float64).ravel()
            y = np.asarray(pair[1], dtype=np.float64).ravel()
            m = np.isfinite(x) & np.isfinite(y)
            elems.append((x[m], y[m]))
        clean_alpha.append(elems)
    GalGA.alpha_data = clean_alpha


    age_arr = data.get("age_data")
    GalGA.age_data = [ (np.asarray(a[0]), np.asarray(a[1])) for a in age_arr ]


    
    mdf_plotting.generate_all_plots(GalGA, feh, normalized_count, results_file=results_csv)
    print(f"[plot-only] plots written to: {outdir}")




















if len(sys.argv) > 1:
    arg_path = sys.argv[1] + '/'
    if os.path.isdir(arg_path):
        pcard_to_be_parsed = os.path.join(arg_path, 'bulge_pcard.txt')
    else:
        pcard_to_be_parsed = arg_path
else:
    pcard_to_be_parsed = 'bulge_pcard.txt'

plot_tokens = {"true", "1", "plot", "--plot", "-p"}
plot_only = False
if len(sys.argv) > 2:
    token = str(sys.argv[2]).lower()
    plot_only = token in plot_tokens

if plot_only:
    target = arg_path if arg_path is not None else "."
    _plot_only(target, bins=69)
    print("done, i shall now die :)")
    sys.exit(0)

else: 
    params = Gal_GA.parse_inlist(pcard_to_be_parsed)

    output_path = params['output_path']
    os.makedirs(output_path, exist_ok=True)

    dest_pcard = os.path.join(output_path, 'bulge_pcard.txt')
    src_pcard = os.path.abspath(pcard_to_be_parsed)
    dst_pcard = os.path.abspath(dest_pcard)
    if src_pcard != dst_pcard:
        shutil.copy2(src_pcard, dest_pcard)

    obs_file = params['obs_file']
    iniab_header = params['iniab_header']
    sn1a_header = params['sn1a_header']
    sigma_2_list = params['sigma_2_list']
    tmax_1_list = params['tmax_1_list']
    tmax_2_list = params['tmax_2_list']
    infall_timescale_1_list = params['infall_timescale_1_list']
    infall_timescale_2_list = params['infall_timescale_2_list']
    comp_array = params['comp_array']
    sfe_array = params['sfe_array']
    imf_array = params['imf_array']
    imf_upper_limits = params['imf_upper_limits']
    sn1a_assumptions = params['sn1a_assumptions']
    stellar_yield_assumptions = params['stellar_yield_assumptions']
    mgal_values = params['mgal_values']
    nb_array = params['nb_array']
    sn1a_rates = params['sn1a_rates']
    timesteps = params['timesteps']
    A2 = params['A2']
    A1 = params['A1']
    physical_constraints_freq = params['physical_constraints_freq']
    delta_sfe_array = params['delta_sfe_array']
    exploration_steps = params['exploration_steps']
    popsize = params['popsize']

    if popsize < 0:
        popsize = int(cpu_count() * (popsize * -1))

    generations = params['generations']
    crossover_probability = params['crossover_probability']
    mutation_probability = params['mutation_probability']
    tournament_size = params['tournament_size']
    selection_threshold = params['selection_threshold']

    demc_fraction = params.get('demc_fraction', 0.4)
    obs_age_data_loss_metric = params['obs_age_data_loss_metric']
    obs_age_data_target = params['obs_age_data_target']
    mdf_vs_age_weight = params['mdf_vs_age_weight']
    rand_seed = params['seed']

    if rand_seed > 0:
        _random.seed(rand_seed)
        _np.random.seed(rand_seed)
        _os.environ['PYTHONHASHSEED'] = str(rand_seed)

    output_interval = params.get('output_interval')
    loss_metric = params['loss_metric']
    fancy_mutation = params['fancy_mutation']
    shrink_range = params['shrink_range']

    gaussian_sigma_scale = params.get('gaussian_sigma_scale', 0.01)
    crossover_noise_fraction = params.get('crossover_noise_fraction', 0.05)
    perturbation_strength = params.get('perturbation_strength', 0.1)

    feh, count = np.loadtxt(obs_file, usecols=(0, 1), unpack=True)
    normalized_count = count / count.max()

    obs_age_data = load_bensby_data('data/Bensby_Data.tsv')

    GalGA = None
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'loss'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'analysis'), exist_ok=True)
    
    results_file = checkpoint.run_with_checkpoint(run_ga, output_path)
    save_walker_history()

