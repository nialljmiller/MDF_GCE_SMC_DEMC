# Posterior Uncertainty Visualization for GCE Models

## Overview

This package provides enhanced plotting functions that display **posterior distributions** (median + 1σ uncertainty bands) instead of single "best" models for galactic chemical evolution (GCE) analysis. This approach properly represents the uncertainty and degeneracy inherent in MCMC/GA sampling methods.

## Files Provided

### Core Modules

1. **`posterior_utils.py`** - Core utilities for posterior statistics
   - `get_weighted_posterior_samples()` - Extract top percentile with fitness weights
   - `weighted_quantile()` - Compute weighted percentiles
   - `compute_mdf_ensemble()` - MDF posterior with uncertainty bands
   - `compute_age_feh_ensemble()` - Age-[Fe/H] posterior
   - `compute_alpha_ensemble()` - Alpha element posteriors
   - `interpolate_to_common_grid()` - Interpolation utilities

2. **`core_plots_posterior.py`** - Modified core plotting functions
   - `plot_age_feh_detailed()` - Age-metallicity with posterior bands
   - `plot_mdf_curves()` - MDF with posterior bands
   - `plot_four_panel_alpha()` - Alpha elements with posterior bands

3. **`phys_plot_posterior.py`** - Modified physics plotting functions
   - `plot_real_infall_physics()` - Infall rates, SFR, gas flows with posterior bands
   - `compute_physics_ensemble()` - Reconstruct multiple models for physics posterior

### Documentation

4. **`posterior_viz_design.md`** - Detailed design document
5. **`analysis_notes.md`** - Code structure analysis
6. **`POSTERIOR_PLOTTING_README.md`** - This file

## Key Concepts

### Why Posterior Distributions?

Your MCMC/GA sampling approach explores parameter space and generates a **distribution of models**, not a single "best" model. The posterior distribution captures:

1. **Parameter uncertainty** - How well-constrained each parameter is
2. **Degeneracies** - Which parameter combinations produce similar fits
3. **Model uncertainty** - Range of predictions consistent with data

Plotting only the "best" model ignores this rich information and can be misleading, especially in highly degenerate, high-dimensional problems like GCE modeling.

### Fitness-Weighted Sampling

The approach uses **inverse-fitness weighting** to emphasize better-fitting models:

```python
# Lower fitness = better fit → higher weight
weights = 1.0 / (fitness + epsilon)
weights = weights / sum(weights)  # Normalize
```

### Percentile Selection

By default, we use the **top 10%** of models by fitness. This balances:
- Including enough models to capture uncertainty
- Excluding poor fits that would artificially widen bands

### 1σ Uncertainty Bands

We compute **16th, 50th (median), 84th percentiles** at each point:
- **Median** (50th percentile) - Central tendency
- **16th-84th percentile range** - Approximately 1σ (68% credible interval)

## Usage

### Basic Usage

Replace your existing plotting calls with the new functions:

```python
# Import the new modules
from core_plots_posterior import (
    plot_age_feh_detailed,
    plot_mdf_curves,
    plot_four_panel_alpha
)
from phys_plot_posterior import plot_real_infall_physics

# Enable posterior mode (default)
plot_age_feh_detailed(
    GalGA, Fe_H, age_Joyce, age_Bensby,
    results_df=results_df,
    use_posterior=True,  # Enable posterior mode
    percentile=10        # Top 10% of models
)

plot_mdf_curves(
    GalGA, feh, normalized_count,
    results_df=results_df,
    use_posterior=True,
    percentile=10
)

plot_four_panel_alpha(
    GalGA, Fe_H, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe,
    results_df=results_df,
    use_posterior=True,
    percentile=10
)

plot_real_infall_physics(
    GalGA,
    results_df=results_df,
    use_posterior=True,
    percentile=10,
    max_models=20  # Limit reconstructions for computational efficiency
)
```

### Legacy Mode (Single Best Model)

To revert to the original behavior:

```python
plot_age_feh_detailed(
    GalGA, Fe_H, age_Joyce, age_Bensby,
    results_df=results_df,
    use_posterior=False  # Disable posterior mode
)
```

### Configuration Options

#### Core Plots

```python
plot_age_feh_detailed(
    GalGA, Fe_H, age_Joyce, age_Bensby,
    results_df=results_df,
    save_path='custom_path.png',  # Custom output path
    n_bins=12,                     # Number of bins for observations
    age_limit_gyr=14.2,            # Maximum age
    use_posterior=True,            # Enable posterior mode
    percentile=10                  # Top X% of models
)
```

#### Physics Plots

```python
plot_real_infall_physics(
    GalGA,
    results_df=results_df,
    save_path='physics_posterior.png',
    use_posterior=True,
    percentile=10,
    max_models=20  # Limit to 20 model reconstructions
)
```

**Note**: Physics plots require reconstructing omega_plus models, which is computationally expensive. The `max_models` parameter limits the number of reconstructions. For `percentile=10` with 1000 total models, this would normally require 100 reconstructions, but `max_models=20` subsamples to 20.

## Installation

### Requirements

```bash
pip install numpy pandas scipy matplotlib seaborn
```

### Integration into Existing Codebase

1. **Copy files to your project**:
   ```bash
   cp posterior_utils.py /path/to/your/project/
   cp core_plots_posterior.py /path/to/your/project/
   cp phys_plot_posterior.py /path/to/your/project/
   ```

2. **Update imports in your main scripts**:
   ```python
   # Replace old imports
   # from core_plots import plot_age_feh_detailed
   
   # With new imports
   from core_plots_posterior import plot_age_feh_detailed
   ```

3. **Ensure `plotting.style` module is available**:
   The code assumes you have a `plotting/style.py` module with `use_paper_style()`. If not, comment out these lines:
   ```python
   # from plotting.style import *
   # use_paper_style()
   ```

## Interpretation Guide

### Reading Uncertainty Bands

- **Narrow bands** → Well-constrained predictions, low model uncertainty
- **Wide bands** → Poorly constrained, high degeneracy
- **Bands that don't include observations** → Systematic model bias

### Example Interpretations

1. **Age-Metallicity Relation**:
   - Narrow bands at young ages → Recent evolution well-constrained
   - Wide bands at old ages → Early evolution uncertain (fewer constraints)

2. **MDF**:
   - Narrow bands at peak → Peak location well-constrained
   - Wide bands in tails → Tail behavior uncertain (sensitive to parameters)

3. **Alpha Elements**:
   - Narrow bands at low [Fe/H] → Early enrichment well-constrained
   - Wide bands at high [Fe/H] → Late enrichment uncertain

4. **Physics (SFR, Infall)**:
   - Narrow bands → Physical process well-constrained by observables
   - Wide bands → Multiple physical scenarios fit data equally well

## Performance Considerations

### Core Plots (Fast)

Core plots (MDF, Age-[Fe/H], Alpha) are **fast** because they use pre-computed model outputs stored in `GalGA.mdf_data`, `GalGA.age_data`, `GalGA.alpha_data`.

**Typical runtime**: ~1-5 seconds per plot

### Physics Plots (Slow)

Physics plots require **reconstructing omega_plus models**, which is computationally expensive.

**Typical runtime**: 
- Single model: ~10-30 seconds
- 20 models: ~5-10 minutes
- 100 models: ~30-60 minutes

**Optimization strategies**:
1. Use `max_models=20` (default) to limit reconstructions
2. Use `percentile=5` to reduce sample size
3. Run physics plots separately or overnight
4. Consider caching reconstructed models

### Caching Strategy (Advanced)

For repeated analysis, consider caching:

```python
import pickle

# After first run, save ensemble
ensemble = compute_physics_ensemble(GalGA, top_df, weights, max_models=20)
with open('physics_ensemble_cache.pkl', 'wb') as f:
    pickle.dump(ensemble, f)

# On subsequent runs, load from cache
with open('physics_ensemble_cache.pkl', 'rb') as f:
    ensemble = pickle.load(f)
```

## Troubleshooting

### Issue: "Could not compute ensemble"

**Cause**: Not enough models in `results_df` or parameter matching failed

**Solution**:
- Check that `results_df` has columns: `sigma_2`, `t_2`, `infall_2`, `fitness`
- Ensure `GalGA.results` is aligned with `GalGA.age_data`, `GalGA.mdf_data`, etc.
- Try increasing `percentile` (e.g., from 10 to 20)

### Issue: "Model reconstruction failed"

**Cause**: Invalid parameters or missing yield tables

**Solution**:
- Check that all parameter indices are valid (e.g., `comp_idx < len(GalGA.comp_array)`)
- Ensure yield tables and initial abundance tables are accessible
- Check for NaN or inf values in parameters

### Issue: Bands are too wide

**Cause**: High parameter degeneracy or too many poor fits included

**Solution**:
- Reduce `percentile` (e.g., from 10 to 5) to focus on best fits
- Check for bimodal distributions (multiple solution islands)
- Consider tighter priors or additional constraints

### Issue: Bands are too narrow

**Cause**: Not enough models or overconfident weights

**Solution**:
- Increase `percentile` (e.g., from 10 to 20)
- Check fitness distribution (are all models similar?)
- Consider using uniform weights: `weights = np.ones(len(top_df)) / len(top_df)`

### Issue: Physics plots are too slow

**Solution**:
- Reduce `max_models` (e.g., from 20 to 10)
- Reduce `percentile` (e.g., from 10 to 5)
- Use parallel processing (see Advanced Usage)

## Advanced Usage

### Custom Percentile Levels

For 2σ (95%) bands instead of 1σ (68%):

```python
# In posterior_utils.py, modify compute_percentile_bands()
bands = compute_percentile_bands(samples, weights, percentiles=[2.5, 50, 97.5])
```

### Parallel Model Reconstruction

For physics plots, parallelize model reconstruction:

```python
from multiprocessing import Pool

def reconstruct_wrapper(args):
    GalGA, row = args
    return reconstruct_model_from_row(GalGA, row)

with Pool(processes=4) as pool:
    models = pool.map(reconstruct_wrapper, [(GalGA, row) for _, row in top_df.iterrows()])
```

### Custom Weighting Schemes

Try different weighting:

```python
# Uniform weights (no fitness weighting)
weights = np.ones(len(top_df)) / len(top_df)

# Exponential weights
weights = np.exp(-fitness / temperature)
weights = weights / np.sum(weights)

# Threshold weights (top N only)
weights = (fitness < threshold).astype(float)
weights = weights / np.sum(weights)
```

## Scientific Best Practices

### Reporting Results

When using posterior plots in publications:

1. **State the method**: "We show the median model with 1σ posterior uncertainty bands computed from the top 10% of MCMC samples, weighted by inverse fitness."

2. **Justify percentile choice**: "We use the top 10% to balance capturing uncertainty while excluding poor fits."

3. **Discuss wide bands**: "Wide uncertainty bands in [region] indicate [parameter degeneracy / lack of constraints / model limitations]."

4. **Compare to best model**: "The median model is consistent with the single best-fit model, but uncertainty bands reveal [additional insight]."

### Validation

Before publishing, validate the approach:

1. **Check convergence**: Ensure MCMC/GA has converged (use `uncertainty_analysis.py`)
2. **Test sensitivity**: Try different percentiles (5%, 10%, 20%) - results should be qualitatively similar
3. **Visual inspection**: Ensure bands look reasonable (not too wide/narrow)
4. **Compare to corner plots**: Posterior bands should be consistent with parameter distributions

### Limitations

Be aware of limitations:

1. **Assumes fitness is meaningful**: If fitness metric is flawed, weights are meaningless
2. **Assumes sampling is adequate**: If parameter space is poorly sampled, posterior is unreliable
3. **Interpolation artifacts**: Bands are interpolated to common grids, which can smooth features
4. **Computational cost**: Physics plots require many model reconstructions

## Examples

### Example 1: Standard Workflow

```python
import pandas as pd
from core_plots_posterior import plot_age_feh_detailed, plot_mdf_curves, plot_four_panel_alpha
from phys_plot_posterior import plot_real_infall_physics

# Load results
results_df = pd.read_csv('SMC_DEMC/simulation_results.csv')
results_df = results_df.sort_values('fitness', ascending=True)  # Lower is better

# Load observational data (your existing code)
Fe_H, age_Joyce, age_Bensby = load_age_data()
feh_mdf, count_mdf = load_mdf_data()
Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe = load_alpha_data()

# Generate plots with posterior uncertainty
plot_age_feh_detailed(GalGA, Fe_H, age_Joyce, age_Bensby, results_df=results_df,
                     use_posterior=True, percentile=10)

plot_mdf_curves(GalGA, feh_mdf, count_mdf, results_df=results_df,
               use_posterior=True, percentile=10)

plot_four_panel_alpha(GalGA, Fe_H, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe, results_df=results_df,
                     use_posterior=True, percentile=10)

plot_real_infall_physics(GalGA, results_df=results_df,
                        use_posterior=True, percentile=10, max_models=20)
```

### Example 2: Comparing Different Percentiles

```python
# Generate plots for different percentiles to assess sensitivity
for pct in [5, 10, 20]:
    plot_age_feh_detailed(GalGA, Fe_H, age_Joyce, age_Bensby, results_df=results_df,
                         use_posterior=True, percentile=pct,
                         save_path=f'age_feh_posterior_p{pct}.png')
```

### Example 3: Legacy Comparison

```python
# Generate both posterior and legacy plots for comparison
plot_age_feh_detailed(GalGA, Fe_H, age_Joyce, age_Bensby, results_df=results_df,
                     use_posterior=True, save_path='age_feh_posterior.png')

plot_age_feh_detailed(GalGA, Fe_H, age_Joyce, age_Bensby, results_df=results_df,
                     use_posterior=False, save_path='age_feh_best_only.png')
```

## Citation

If you use this posterior visualization approach in your research, please cite:

- Your paper describing the GCE model and MCMC/GA method
- This README as supplementary material

## Contact

For questions or issues:
- Open an issue on the GitHub repository
- Contact: [Your contact information]

## Changelog

### Version 1.0 (2025-01-XX)
- Initial release
- Core plots: MDF, Age-[Fe/H], Alpha elements
- Physics plots: Infall, SFR, gas flows, masses
- Fitness-weighted posterior statistics
- Backward compatibility with legacy single-best mode

## Future Enhancements

Potential future improvements:

1. **Parallel processing** for physics plots
2. **Caching** of reconstructed models
3. **Interactive plots** with adjustable percentiles
4. **Additional diagnostics** (e.g., convergence checks)
5. **Alternative weighting schemes** (e.g., Bayesian evidence)
6. **2D posterior contours** for parameter pairs
7. **Time-dependent uncertainty** visualization

## Acknowledgments

This posterior visualization framework was developed to properly represent the uncertainty and degeneracy in high-dimensional GCE parameter inference problems. It builds on the existing `uncertainty_analysis.py` module and extends it to all core plotting functions.

