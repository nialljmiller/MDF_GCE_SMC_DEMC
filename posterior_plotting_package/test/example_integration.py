#!/usr/bin/env python3
"""
Example integration script showing how to use posterior plotting functions.

This script demonstrates:
1. Loading results from MCMC/GA sampling
2. Generating plots with posterior uncertainty bands
3. Comparing posterior mode vs. legacy single-best mode
"""

import pandas as pd
import numpy as np

# Import the new posterior plotting functions
from core_plots_posterior import (
    plot_age_feh_detailed,
    plot_mdf_curves,
    plot_four_panel_alpha
)
from phys_plot_posterior import plot_real_infall_physics


def main():
    """Main function demonstrating posterior plotting workflow"""
    
    # ========================================================================
    # STEP 1: Load your GalGA object and results
    # ========================================================================
    
    # This is your existing code - replace with your actual loading logic
    # GalGA = load_your_galga_object()  # Your existing function
    # results_df = pd.read_csv('SMC_DEMC/simulation_results.csv')
    
    # For this example, we'll assume these are already loaded
    print("=" * 70)
    print("EXAMPLE: Posterior Uncertainty Plotting Integration")
    print("=" * 70)
    print()
    print("This script demonstrates how to integrate posterior plotting")
    print("into your existing analysis workflow.")
    print()
    
    # ========================================================================
    # STEP 2: Prepare results dataframe
    # ========================================================================
    
    print("STEP 1: Load and prepare results")
    print("-" * 70)
    
    # Load results (your existing code)
    # results_df = pd.read_csv('SMC_DEMC/simulation_results.csv')
    
    # Ensure results are sorted by fitness (lower is better)
    # results_df = results_df.sort_values('fitness', ascending=True)
    
    print("✓ Results loaded and sorted by fitness")
    print(f"  Total models: [your_count]")
    print(f"  Best fitness: [your_best_fitness]")
    print()
    
    # ========================================================================
    # STEP 3: Load observational data
    # ========================================================================
    
    print("STEP 2: Load observational data")
    print("-" * 70)
    
    # Your existing data loading code
    # Fe_H, age_Joyce, age_Bensby = load_age_metallicity_data()
    # feh_mdf, count_mdf = load_mdf_data()
    # Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe = load_alpha_data()
    
    print("✓ Observational data loaded")
    print()
    
    # ========================================================================
    # STEP 4: Generate core plots with posterior uncertainty
    # ========================================================================
    
    print("STEP 3: Generate core plots with posterior uncertainty")
    print("-" * 70)
    
    # Age-Metallicity Relation
    print("Generating Age-[Fe/H] plot with posterior bands...")
    # plot_age_feh_detailed(
    #     GalGA, Fe_H, age_Joyce, age_Bensby,
    #     results_df=results_df,
    #     save_path='output/age_feh_posterior.png',
    #     use_posterior=True,  # Enable posterior mode
    #     percentile=10        # Top 10% of models
    # )
    print("✓ Age-[Fe/H] plot saved to: output/age_feh_posterior.png")
    
    # MDF
    print("Generating MDF plot with posterior bands...")
    # plot_mdf_curves(
    #     GalGA, feh_mdf, count_mdf,
    #     results_df=results_df,
    #     save_path='output/mdf_posterior.png',
    #     use_posterior=True,
    #     percentile=10
    # )
    print("✓ MDF plot saved to: output/mdf_posterior.png")
    
    # Alpha Elements
    print("Generating Alpha elements plot with posterior bands...")
    # plot_four_panel_alpha(
    #     GalGA, Fe_H, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe,
    #     results_df=results_df,
    #     save_path='output/alpha_posterior.png',
    #     use_posterior=True,
    #     percentile=10
    # )
    print("✓ Alpha elements plot saved to: output/alpha_posterior.png")
    print()
    
    # ========================================================================
    # STEP 5: Generate physics plots (slower)
    # ========================================================================
    
    print("STEP 4: Generate physics plots with posterior uncertainty")
    print("-" * 70)
    print("WARNING: Physics plots require reconstructing omega_plus models")
    print("         This may take 5-10 minutes for 20 models")
    print()
    
    # Physics plots
    print("Generating physics plot with posterior bands...")
    # plot_real_infall_physics(
    #     GalGA,
    #     results_df=results_df,
    #     save_path='output/physics_posterior.png',
    #     use_posterior=True,
    #     percentile=10,
    #     max_models=20  # Limit to 20 reconstructions for efficiency
    # )
    print("✓ Physics plot saved to: output/physics_posterior.png")
    print()
    
    # ========================================================================
    # STEP 6: Optional - Generate legacy plots for comparison
    # ========================================================================
    
    print("STEP 5 (Optional): Generate legacy single-best plots for comparison")
    print("-" * 70)
    
    # Generate legacy plots with use_posterior=False
    print("Generating legacy Age-[Fe/H] plot (single best model)...")
    # plot_age_feh_detailed(
    #     GalGA, Fe_H, age_Joyce, age_Bensby,
    #     results_df=results_df,
    #     save_path='output/age_feh_legacy.png',
    #     use_posterior=False  # Disable posterior mode
    # )
    print("✓ Legacy Age-[Fe/H] plot saved to: output/age_feh_legacy.png")
    print()
    
    # ========================================================================
    # STEP 7: Sensitivity analysis (optional)
    # ========================================================================
    
    print("STEP 6 (Optional): Sensitivity analysis - different percentiles")
    print("-" * 70)
    
    for pct in [5, 10, 20]:
        print(f"Generating Age-[Fe/H] plot with top {pct}% of models...")
        # plot_age_feh_detailed(
        #     GalGA, Fe_H, age_Joyce, age_Bensby,
        #     results_df=results_df,
        #     save_path=f'output/age_feh_posterior_p{pct}.png',
        #     use_posterior=True,
        #     percentile=pct
        # )
        print(f"✓ Saved to: output/age_feh_posterior_p{pct}.png")
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Generated plots:")
    print("  ✓ Age-[Fe/H] with posterior bands")
    print("  ✓ MDF with posterior bands")
    print("  ✓ Alpha elements with posterior bands")
    print("  ✓ Physics (infall, SFR, etc.) with posterior bands")
    print("  ✓ Legacy single-best plots for comparison")
    print("  ✓ Sensitivity analysis (different percentiles)")
    print()
    print("Next steps:")
    print("  1. Examine plots and assess uncertainty bands")
    print("  2. Adjust percentile parameter if bands are too wide/narrow")
    print("  3. Compare posterior vs. legacy plots")
    print("  4. Use posterior plots in your paper/presentation")
    print()
    print("For questions, see POSTERIOR_PLOTTING_README.md")
    print("=" * 70)


def example_custom_configuration():
    """
    Example showing custom configuration options.
    """
    
    print("\nEXAMPLE: Custom Configuration")
    print("-" * 70)
    
    # Custom age range and binning
    # plot_age_feh_detailed(
    #     GalGA, Fe_H, age_Joyce, age_Bensby,
    #     results_df=results_df,
    #     save_path='output/age_feh_custom.png',
    #     n_bins=20,           # More bins for observations
    #     age_limit_gyr=12.0,  # Shorter age range
    #     use_posterior=True,
    #     percentile=15        # Top 15% instead of 10%
    # )
    
    # Custom MDF range
    # plot_mdf_curves(
    #     GalGA, feh_mdf, count_mdf,
    #     results_df=results_df,
    #     save_path='output/mdf_custom.png',
    #     use_posterior=True,
    #     percentile=10
    # )
    
    # Physics with fewer model reconstructions (faster)
    # plot_real_infall_physics(
    #     GalGA,
    #     results_df=results_df,
    #     save_path='output/physics_fast.png',
    #     use_posterior=True,
    #     percentile=5,   # Fewer models
    #     max_models=10   # Fewer reconstructions
    # )
    
    print("✓ Custom configuration examples shown")


def example_troubleshooting():
    """
    Example showing common troubleshooting scenarios.
    """
    
    print("\nEXAMPLE: Troubleshooting")
    print("-" * 70)
    
    # Check if results_df has required columns
    # required_cols = ['sigma_2', 't_2', 'infall_2', 'fitness']
    # missing = [col for col in required_cols if col not in results_df.columns]
    # if missing:
    #     print(f"ERROR: Missing required columns: {missing}")
    #     return
    
    # Check if results_df is sorted
    # if not results_df['fitness'].is_monotonic_increasing:
    #     print("WARNING: results_df not sorted by fitness, sorting now...")
    #     results_df = results_df.sort_values('fitness', ascending=True)
    
    # Check for NaN values
    # if results_df[required_cols].isnull().any().any():
    #     print("WARNING: NaN values detected in results_df")
    #     print(results_df[required_cols].isnull().sum())
    
    # Try with different percentiles if bands are too wide
    # if bands_too_wide:
    #     print("Trying with smaller percentile (5% instead of 10%)...")
    #     plot_age_feh_detailed(..., percentile=5)
    
    # Try with more models if bands are too narrow
    # if bands_too_narrow:
    #     print("Trying with larger percentile (20% instead of 10%)...")
    #     plot_age_feh_detailed(..., percentile=20)
    
    print("✓ Troubleshooting examples shown")


if __name__ == '__main__':
    print()
    print("=" * 70)
    print("POSTERIOR PLOTTING - EXAMPLE INTEGRATION SCRIPT")
    print("=" * 70)
    print()
    print("This script demonstrates how to integrate posterior plotting")
    print("into your existing GCE analysis workflow.")
    print()
    print("To use this script:")
    print("  1. Uncomment the actual function calls")
    print("  2. Replace placeholder variables with your actual data")
    print("  3. Run the script to generate plots")
    print()
    print("For full documentation, see POSTERIOR_PLOTTING_README.md")
    print("=" * 70)
    print()
    
    # Run main example
    main()
    
    # Additional examples
    example_custom_configuration()
    example_troubleshooting()
    
    print()
    print("=" * 70)
    print("Example script completed successfully!")
    print("=" * 70)
    print()

