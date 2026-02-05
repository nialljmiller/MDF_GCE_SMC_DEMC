#!/usr/bin/env python3
"""
Main entry point for running the MDF GCE Genetic Algorithm.

Usage:
    python run_ga.py [pcard_path] [options]
    
Examples:
    python run_ga.py                           # Use default bulge_pcard.txt
    python run_ga.py path/to/pcard.txt         # Use custom pcard
    python run_ga.py SMC_DEMC/ --plot-only     # Generate plots from results
"""

import argparse
import os
import sys
import numpy as np
from multiprocessing import cpu_count

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdf_gce.config import parse_inlist
from mdf_gce.utils import ensure_dirs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run MDF GCE Genetic Algorithm optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ga.py                       Run with default pcard
  python run_ga.py my_pcard.txt          Run with custom pcard
  python run_ga.py results/ --plot-only  Generate plots from existing results
        """
    )
    
    parser.add_argument(
        'pcard',
        nargs='?',
        default='bulge_pcard.txt',
        help='Path to parameter card file or output directory (default: bulge_pcard.txt)'
    )
    
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='Skip optimization, only generate plots from existing results'
    )
    
    parser.add_argument(
        '--plot-mode',
        choices=['full', 'minimal', 'posterior_minimal', 'none'],
        default='full',
        help='Plotting mode (default: full)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if available'
    )
    
    parser.add_argument(
        '--no-smc',
        action='store_true',
        help='Skip SMC-DEMC refinement stage'
    )
    
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        default=True,
        help='Enable parallel evaluation (default: True)'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel evaluation'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def load_observational_data(obs_file: str, bensby_file: str = None):
    """Load observational MDF and age data."""
    # Load MDF
    feh, count = np.loadtxt(obs_file, usecols=(0, 1), unpack=True)
    normalized_count = count / max(count.max(), 1.0)
    
    # Load Bensby age data if available
    obs_age_data = None
    if bensby_file and os.path.exists(bensby_file):
        try:
            from mdf_gce.utils import load_bensby_data
            obs_age_data = load_bensby_data(bensby_file)
        except Exception as e:
            print(f"Warning: Could not load Bensby data: {e}")
    
    return feh, normalized_count, obs_age_data


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine if pcard is a file or directory
    if os.path.isdir(args.pcard):
        # Plot-only mode from results directory
        if args.plot_only:
            run_plot_only(args.pcard, args.plot_mode)
            return
        else:
            # Look for pcard in directory
            pcard_path = os.path.join(args.pcard, 'bulge_pcard.txt')
            if not os.path.exists(pcard_path):
                print(f"Error: No bulge_pcard.txt found in {args.pcard}")
                sys.exit(1)
    else:
        pcard_path = args.pcard
    
    if not os.path.exists(pcard_path):
        print(f"Error: Parameter card not found: {pcard_path}")
        sys.exit(1)
    
    print(f"Loading configuration from: {pcard_path}")
    
    # Parse configuration
    config = parse_inlist(pcard_path)
    
    # Validate
    popsize = config.get('popsize', 96)

    if popsize < 0:
        popsize = int(cpu_count() * (popsize * -1))

    generations = config.get('generations', 256)
    
    if popsize < 10:
        print("Warning: population_size < 10 may cause poor convergence")
    if generations < 10:
        print("Warning: num_generations < 10 may cause poor convergence")
    
    # Set output path
    output_path = config.get('output_path', 'SMC_DEMC/')
    if not output_path.endswith('/'):
        output_path += '/'
    
    print(f"Output directory: {output_path}")
    ensure_dirs(output_path, ['plots', 'checkpoints'])
    
    # Check for JINAPyCEE
    try:
        from JINAPyCEE import omega_plus
        print("JINAPyCEE omega_plus loaded successfully")
    except ImportError:
        print("ERROR: JINAPyCEE not found!")
        print("Please ensure JINAPyCEE is installed or in your PYTHONPATH")
        print("Expected location: ../JINAPyCEE/ or installed via pip")
        sys.exit(1)
    
    # Load observational data
    obs_file = config.get('obs_file', 'data/MDF_Bulge_Composite.txt')
    if not os.path.isabs(obs_file):
        # Try relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        obs_file_candidates = [
            obs_file,
            os.path.join(parent_dir, obs_file),
            os.path.join(parent_dir, 'data', os.path.basename(obs_file)),
        ]
        for candidate in obs_file_candidates:
            if os.path.exists(candidate):
                obs_file = candidate
                break
    
    if not os.path.exists(obs_file):
        print(f"ERROR: Observational data file not found: {obs_file}")
        sys.exit(1)
    
    print(f"Loading observational data from: {obs_file}")
    feh, normalized_count, obs_age_data = load_observational_data(
        obs_file,
        config.get('bensby_file', 'data/Bensby_Data.tsv')
    )
    
    # Import GA
    from mdf_gce.core.ga import GalacticEvolutionGA, CheckpointManager
    
    # Create GA instance
    print("\nInitializing Galactic Evolution GA...")
    
    ga = GalacticEvolutionGA(
        output_path=output_path,
        sn1a_header=config.get('sn1a_header', 'yield_tables/sn1a/'),
        iniab_header=config.get('iniab_header', 'yield_tables/iniabu/'),
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
        obs_age_data=obs_age_data,
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
        PP=not args.no_parallel,
        demc_hybrid=config.get('demc_hybrid', True),
        demc_fraction=config.get('demc_fraction', 0.5),
        demc_moves_per_gen=config.get('demc_moves_per_gen', 1),
        plot_mode=args.plot_mode,
    )
    
    # Initialize population
    population, toolbox = ga.init_GenAl(population_size=popsize)
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(save_path=output_path)
    
    # Check for resume
    start_gen = 0
    if args.resume:
        cp_data = checkpoint_manager.load()
        if cp_data:
            start_gen = cp_data.get('generation', 0) + 1
            print(f"Resuming from generation {start_gen}")
    
    # Copy pcard to output
    import shutil
    pcard_dest = os.path.join(output_path, 'bulge_pcard.txt')
    if os.path.abspath(pcard_path) != os.path.abspath(pcard_dest):
        shutil.copy2(pcard_path, pcard_dest)
        print(f"Copied pcard to {pcard_dest}")
    
    # Run GA
    print("\n" + "=" * 60)
    print("STARTING GENETIC ALGORITHM OPTIMIZATION")
    print("=" * 60 + "\n")
    
    ga.GenAl(
        population_size=popsize,
        num_generations=generations,
        population=population,
        toolbox=toolbox,
        checkpoint_manager=checkpoint_manager,
        start_gen=start_gen,
        output_interval=config.get('output_interval', 16),
    )
    
    # Save walker history
    ga.save_walker_history()
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")


def run_plot_only(target_path: str, plot_mode: str) -> None:
    """Generate plots from existing results."""
    print(f"Plot-only mode: {target_path}")
    print(f"Plot mode: {plot_mode}")
    
    # Find results CSV
    import glob
    csv_files = glob.glob(os.path.join(target_path, 'simulation_results*.csv'))
    
    if not csv_files:
        print(f"Error: No simulation_results*.csv found in {target_path}")
        return
    
    # Use most recent
    results_csv = max(csv_files, key=os.path.getmtime)
    print(f"Using results: {results_csv}")
    
    # Run analysis
    from mdf_gce.analysis import UncertaintyAnalysis
    
    ua = UncertaintyAnalysis(results_csv, output_path=target_path)
    
    if plot_mode in ['full', 'minimal']:
        print("Generating corner plot...")
        ua.plot_corner()
        
        print("Generating marginal plots...")
        ua.plot_marginals()
    
    print("Generating uncertainty report...")
    report = ua.generate_report(save=True)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
