#!/usr/bin/env python3
"""
Standalone example demonstrating posterior_utils.py functionality.

This example shows how the posterior statistics utilities work
without requiring the full GalGA/JINAPyCEE environment.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

from posterior_utils import (
    get_weighted_posterior_samples,
    weighted_quantile,
    interpolate_to_common_grid,
    compute_percentile_bands
)


def create_mock_results(n_models=1000):
    """Create mock results dataframe for demonstration"""
    
    np.random.seed(42)
    
    # Generate mock parameters with some correlation
    sigma_2 = np.random.uniform(100, 500, n_models)
    t_2 = np.random.uniform(6, 10, n_models)
    infall_2 = np.random.uniform(1, 5, n_models)
    
    # Generate fitness with some dependence on parameters
    # Lower fitness = better fit
    fitness = (
        0.1 * (sigma_2 - 300)**2 / 100**2 +
        0.1 * (t_2 - 8)**2 / 2**2 +
        0.1 * (infall_2 - 3)**2 / 2**2 +
        np.random.exponential(0.5, n_models)
    )
    
    # Create dataframe
    results_df = pd.DataFrame({
        'sigma_2': sigma_2,
        't_2': t_2,
        'infall_2': infall_2,
        'fitness': fitness,
        'comp_idx': np.random.randint(0, 3, n_models),
        'imf_idx': np.random.randint(0, 2, n_models),
        'sn1a_idx': np.random.randint(0, 2, n_models),
        'sy_idx': np.random.randint(0, 2, n_models),
        'sn1ar_idx': np.random.randint(0, 2, n_models)
    })
    
    # Sort by fitness
    results_df = results_df.sort_values('fitness', ascending=True)
    
    return results_df


def example_weighted_sampling():
    """Demonstrate weighted posterior sampling"""
    
    print("=" * 70)
    print("EXAMPLE 1: Weighted Posterior Sampling")
    print("=" * 70)
    print()
    
    # Create mock results
    results_df = create_mock_results(n_models=1000)
    
    print(f"Total models: {len(results_df)}")
    print(f"Best fitness: {results_df['fitness'].iloc[0]:.4f}")
    print(f"Median fitness: {results_df['fitness'].median():.4f}")
    print(f"Worst fitness: {results_df['fitness'].iloc[-1]:.4f}")
    print()
    
    # Extract top 10% with weights
    top_df, weights = get_weighted_posterior_samples(results_df, 
                                                     fitness_col='fitness',
                                                     percentile=10)
    
    print(f"Top 10% models: {len(top_df)}")
    print(f"Weight range: [{weights.min():.6f}, {weights.max():.6f}]")
    print(f"Sum of weights: {weights.sum():.6f} (should be 1.0)")
    print()
    
    # Compute weighted statistics for a parameter
    param = 'sigma_2'
    values = top_df[param].values
    
    # Unweighted statistics
    unweighted_median = np.median(values)
    unweighted_std = np.std(values)
    
    # Weighted statistics
    weighted_median = weighted_quantile(values, [0.50], weights)[0]
    weighted_16, weighted_84 = weighted_quantile(values, [0.16, 0.84], weights)
    
    print(f"Parameter: {param}")
    print(f"  Unweighted median: {unweighted_median:.2f}")
    print(f"  Weighted median:   {weighted_median:.2f}")
    print(f"  Unweighted std:    {unweighted_std:.2f}")
    print(f"  Weighted 1σ range: [{weighted_16:.2f}, {weighted_84:.2f}]")
    print()


def example_interpolation():
    """Demonstrate interpolation to common grid"""
    
    print("=" * 70)
    print("EXAMPLE 2: Interpolation to Common Grid")
    print("=" * 70)
    print()
    
    # Create mock model curves with different grids
    n_models = 5
    x_arrays = []
    y_arrays = []
    
    for i in range(n_models):
        # Each model has a different grid
        n_points = np.random.randint(20, 50)
        x = np.sort(np.random.uniform(0, 10, n_points))
        
        # Each model has a different functional form
        y = np.sin(x + i * 0.5) + 0.1 * i
        
        x_arrays.append(x)
        y_arrays.append(y)
    
    print(f"Number of models: {n_models}")
    print(f"Grid sizes: {[len(x) for x in x_arrays]}")
    print()
    
    # Define common grid
    x_common = np.linspace(0, 10, 100)
    
    # Interpolate all models to common grid
    y_interp = interpolate_to_common_grid(x_arrays, y_arrays, x_common)
    
    print(f"Common grid size: {len(x_common)}")
    print(f"Interpolated array shape: {y_interp.shape}")
    print(f"  (n_models={y_interp.shape[0]}, n_points={y_interp.shape[1]})")
    print()


def example_percentile_bands():
    """Demonstrate percentile band computation"""
    
    print("=" * 70)
    print("EXAMPLE 3: Percentile Band Computation")
    print("=" * 70)
    print()
    
    # Create mock ensemble of curves
    n_models = 100
    n_points = 50
    x_common = np.linspace(0, 10, n_points)
    
    # Generate ensemble with some spread
    y_samples = np.zeros((n_models, n_points))
    for i in range(n_models):
        amplitude = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(-0.5, 0.5)
        noise = np.random.normal(0, 0.1, n_points)
        y_samples[i, :] = amplitude * np.sin(x_common + phase) + noise
    
    # Create mock weights (inverse fitness)
    fitness = np.random.exponential(1.0, n_models)
    weights = 1.0 / (fitness + 0.1)
    weights = weights / weights.sum()
    
    print(f"Ensemble size: {n_models} models")
    print(f"Number of points: {n_points}")
    print()
    
    # Compute percentile bands
    bands = compute_percentile_bands(y_samples, weights, percentiles=[16, 50, 84])
    
    print("Computed bands:")
    print(f"  Lower (16th percentile): shape={bands['lower'].shape}")
    print(f"  Median (50th percentile): shape={bands['median'].shape}")
    print(f"  Upper (84th percentile): shape={bands['upper'].shape}")
    print()
    
    # Compute band width statistics
    band_width = bands['upper'] - bands['lower']
    print(f"Band width statistics:")
    print(f"  Mean width: {np.nanmean(band_width):.4f}")
    print(f"  Min width:  {np.nanmin(band_width):.4f}")
    print(f"  Max width:  {np.nanmax(band_width):.4f}")
    print()


def example_full_workflow():
    """Demonstrate full workflow from results to posterior bands"""
    
    print("=" * 70)
    print("EXAMPLE 4: Full Workflow")
    print("=" * 70)
    print()
    
    # Step 1: Create mock results
    print("Step 1: Generate mock results...")
    results_df = create_mock_results(n_models=500)
    print(f"  ✓ Generated {len(results_df)} models")
    print()
    
    # Step 2: Extract top percentile with weights
    print("Step 2: Extract top 10% with fitness weights...")
    top_df, weights = get_weighted_posterior_samples(results_df, 
                                                     fitness_col='fitness',
                                                     percentile=10)
    print(f"  ✓ Selected {len(top_df)} models")
    print()
    
    # Step 3: Generate mock model predictions
    print("Step 3: Generate mock model predictions...")
    x_common = np.linspace(0, 14, 100)  # Age in Gyr
    y_arrays = []
    x_arrays = []
    
    for idx, row in top_df.iterrows():
        # Each model predicts [Fe/H] vs age
        # Use parameters to create variation
        sigma_2 = row['sigma_2']
        t_2 = row['t_2']
        
        x = np.linspace(0, 14, 50)
        y = -1.5 + 0.1 * x + 0.001 * (sigma_2 - 300) - 0.05 * (t_2 - 8)
        y += np.random.normal(0, 0.05, len(x))
        
        x_arrays.append(x)
        y_arrays.append(y)
    
    print(f"  ✓ Generated predictions for {len(y_arrays)} models")
    print()
    
    # Step 4: Interpolate to common grid
    print("Step 4: Interpolate to common grid...")
    y_interp = interpolate_to_common_grid(x_arrays, y_arrays, x_common)
    print(f"  ✓ Interpolated to grid with {len(x_common)} points")
    print()
    
    # Step 5: Compute percentile bands
    print("Step 5: Compute percentile bands...")
    bands = compute_percentile_bands(y_interp, weights, percentiles=[16, 50, 84])
    print(f"  ✓ Computed median and 1σ bands")
    print()
    
    # Step 6: Summary statistics
    print("Step 6: Summary statistics...")
    median_feh = bands['median']
    lower_feh = bands['lower']
    upper_feh = bands['upper']
    
    print(f"  Median [Fe/H] at age=0: {median_feh[0]:.3f}")
    print(f"  Median [Fe/H] at age=14: {median_feh[-1]:.3f}")
    print(f"  Mean 1σ band width: {np.nanmean(upper_feh - lower_feh):.3f}")
    print()
    
    print("✓ Full workflow completed successfully!")
    print()


def main():
    """Run all examples"""
    
    print()
    print("=" * 70)
    print("POSTERIOR UTILITIES - STANDALONE EXAMPLES")
    print("=" * 70)
    print()
    print("These examples demonstrate the core functionality of posterior_utils.py")
    print("without requiring the full GalGA/JINAPyCEE environment.")
    print()
    
    # Run examples
    example_weighted_sampling()
    example_interpolation()
    example_percentile_bands()
    example_full_workflow()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Review the code in posterior_utils.py")
    print("  2. Integrate into your analysis workflow")
    print("  3. See POSTERIOR_PLOTTING_README.md for full documentation")
    print()


if __name__ == '__main__':
    main()

