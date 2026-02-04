#!/usr/bin/env python3
"""
Regenerate paper-quality plots from existing GA results.

Usage:
    python -m mdf_gce.plotting.regenerate_plots /path/to/results
    
    # Or with options:
    python -m mdf_gce.plotting.regenerate_plots /path/to/results \
        --obs-mdf data/mdf_APOGEE.txt \
        --obs-age data/Bensby_ages.tsv \
        --output plots/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate paper plots from GA results'
    )
    parser.add_argument(
        'results_dir',
        help='Directory containing GA results (CSV + NPZ files)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output directory for plots (default: results_dir/plots)'
    )
    parser.add_argument(
        '--obs-mdf',
        default=None,
        help='Path to observed MDF file'
    )
    parser.add_argument(
        '--obs-age',
        default=None,
        help='Path to observed age data file (TSV with Fe/H, Joyce_age, Bensby, alpha columns)'
    )
    parser.add_argument(
        '--loss-col',
        default='fitness',
        help='Column name for loss/fitness values'
    )
    parser.add_argument(
        '--plots',
        default='all',
        choices=['all', 'mdf', 'amr', 'alpha', 'corner'],
        help='Which plots to generate'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output) if args.output else results_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import modules
    try:
        from mdf_gce.io import ResultsLoader
        from mdf_gce.plotting.paper_plots import (
            plot_mdf_posterior,
            plot_amr_posterior, 
            plot_alpha_posterior,
            plot_corner_posterior,
            compute_posterior_weights,
        )
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure mdf_gce package is installed")
        sys.exit(1)
    
    # Load results
    print(f"Loading results from {results_dir}...")
    loader = ResultsLoader(str(results_dir))
    
    try:
        df = loader.load(include_curves=True)
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} models")
    
    # Add posterior weights
    df = loader.add_posterior_weights(df, args.loss_col)
    
    # Load observational data
    obs = loader.get_observational_data()
    
    # Override with command line args
    if args.obs_mdf and os.path.exists(args.obs_mdf):
        print(f"Loading MDF data from {args.obs_mdf}")
        data = np.loadtxt(args.obs_mdf, usecols=(0, 1))
        obs['feh'] = data[:, 0]
        obs['mdf'] = data[:, 1]
    
    if args.obs_age and os.path.exists(args.obs_age):
        print(f"Loading age data from {args.obs_age}")
        age_df = pd.read_csv(args.obs_age, sep='\t')
        if '[Fe/H]' in age_df.columns and 'feh' not in obs:
            obs['feh'] = age_df['[Fe/H]'].values
        if 'Joyce_age' in age_df.columns:
            obs['age_joyce'] = age_df['Joyce_age'].values
        if 'Bensby' in age_df.columns:
            obs['age_bensby'] = age_df['Bensby'].values
        for elem, col in [('mg_fe', '[Mg/Fe]'), ('si_fe', '[Si/Fe]'),
                         ('ca_fe', '[Ca/Fe]'), ('ti_fe', '[Ti/Fe]')]:
            if col in age_df.columns:
                obs[elem] = age_df[col].values
    
    # Check what data we have
    has_feh = 'feh' in obs
    has_mdf = 'mdf' in obs
    has_ages = 'age_joyce' in obs or 'age_bensby' in obs
    has_alpha = any(k in obs for k in ['mg_fe', 'si_fe', 'ca_fe', 'ti_fe'])
    has_mdf_curves = 'mdf_x' in df.columns
    has_age_curves = 'age_x' in df.columns
    has_alpha_curves = 'alpha_tracks' in df.columns
    
    if args.verbose:
        print(f"\nObservational data available:")
        print(f"  [Fe/H]: {has_feh}")
        print(f"  MDF: {has_mdf}")
        print(f"  Ages: {has_ages}")
        print(f"  Alpha: {has_alpha}")
        print(f"\nModel curve data available:")
        print(f"  MDF curves: {has_mdf_curves}")
        print(f"  Age curves: {has_age_curves}")
        print(f"  Alpha curves: {has_alpha_curves}")
    
    generated = []
    
    # Generate requested plots
    plots_to_make = ['mdf', 'amr', 'alpha', 'corner'] if args.plots == 'all' else [args.plots]
    
    if 'mdf' in plots_to_make:
        if has_feh and has_mdf and has_mdf_curves:
            print("\nGenerating MDF posterior plot...")
            try:
                path = plot_mdf_posterior(
                    df, obs['feh'], obs['mdf'],
                    str(output_dir), args.loss_col
                )
                generated.append(('MDF', path))
            except Exception as e:
                print(f"Error generating MDF plot: {e}")
        else:
            print("Skipping MDF plot (missing data)")
    
    if 'amr' in plots_to_make:
        if has_feh and has_ages and has_age_curves:
            print("\nGenerating AMR posterior plot...")
            try:
                age_joyce = obs.get('age_joyce', np.full_like(obs['feh'], np.nan))
                age_bensby = obs.get('age_bensby', np.full_like(obs['feh'], np.nan))
                path = plot_amr_posterior(
                    df, age_joyce, age_bensby, obs['feh'],
                    str(output_dir), args.loss_col
                )
                generated.append(('AMR', path))
            except Exception as e:
                print(f"Error generating AMR plot: {e}")
        else:
            print("Skipping AMR plot (missing data)")
    
    if 'alpha' in plots_to_make:
        if has_feh and has_alpha and has_alpha_curves:
            print("\nGenerating alpha posterior plot...")
            try:
                alpha_obs = {
                    'Mg': obs.get('mg_fe', np.array([])),
                    'Si': obs.get('si_fe', np.array([])),
                    'Ca': obs.get('ca_fe', np.array([])),
                    'Ti': obs.get('ti_fe', np.array([])),
                }
                path = plot_alpha_posterior(
                    df, obs['feh'], alpha_obs,
                    str(output_dir), args.loss_col
                )
                generated.append(('Alpha', path))
            except Exception as e:
                print(f"Error generating alpha plot: {e}")
        else:
            print("Skipping alpha plot (missing data)")
    
    if 'corner' in plots_to_make:
        print("\nGenerating corner plot...")
        try:
            path = plot_corner_posterior(df, str(output_dir), args.loss_col)
            generated.append(('Corner', path))
        except Exception as e:
            print(f"Error generating corner plot: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Generated {len(generated)} plots:")
    for name, path in generated:
        print(f"  {name}: {path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
