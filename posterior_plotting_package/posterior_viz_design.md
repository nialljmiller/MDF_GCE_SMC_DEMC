# Posterior Visualization Design Document

## Overview

This document outlines the strategy for modifying `core_plots.py` and `phys_plot.py` to display posterior distributions (median + 1σ bands) instead of single "best" models.

## Design Principles

1. **Fitness-Weighted Sampling**: Use inverse-fitness weights to emphasize better-fitting models
2. **Top Percentile Selection**: Focus on top 10% of models by fitness (configurable)
3. **Weighted Quantiles**: Compute 16th, 50th (median), 84th percentiles for 1σ bands
4. **Common Grid Interpolation**: Interpolate all models to common grids for consistent comparison
5. **Backward Compatibility**: Maintain ability to plot single best model if desired

## Architecture

### New Module: `posterior_utils.py`

Create a utility module with reusable functions for posterior statistics:

```python
# posterior_utils.py - Posterior statistics utilities

def get_weighted_posterior_samples(df, fitness_col='fitness', percentile=10)
def weighted_quantile(values, quantiles, sample_weight)
def compute_model_ensemble(GalGA, top_df, weights, data_type='mdf')
def interpolate_to_common_grid(x_arrays, y_arrays, x_common)
def compute_percentile_bands(y_samples_on_grid, weights, percentiles=[16, 50, 84])
```

### Modified Plotting Functions

Each plotting function will follow this pattern:

```python
def plot_X_with_posterior(GalGA, obs_data, results_df=None, 
                          use_posterior=True, percentile=10, 
                          save_path=None):
    """
    Parameters:
    -----------
    use_posterior : bool
        If True, plot median + 1σ bands from top percentile
        If False, plot single best model (legacy behavior)
    percentile : float
        Top X% of models to include in posterior (default 10)
    """
    
    if use_posterior:
        # Get weighted samples
        top_df, weights = get_weighted_posterior_samples(results_df, percentile=percentile)
        
        # Compute ensemble predictions
        ensemble = compute_model_ensemble(GalGA, top_df, weights, data_type='...')
        
        # Extract median and bands
        median, lower, upper = ensemble['median'], ensemble['lower'], ensemble['upper']
        
        # Plot with uncertainty
        ax.plot(x, median, color='red', lw=2.5, label='Median model')
        ax.fill_between(x, lower, upper, color='red', alpha=0.3, label='1σ uncertainty')
    else:
        # Legacy: plot single best model
        ...
```

## Data Type-Specific Strategies

### 1. MDF (Metallicity Distribution Function)

**Challenge**: Different models may have slightly different [Fe/H] grids

**Solution**:
```python
def compute_mdf_ensemble(GalGA, top_df, weights):
    # Define common [Fe/H] grid
    feh_common = np.linspace(-2.0, 1.0, 100)
    
    # Extract MDF for each model in top_df
    mdf_samples = []
    for idx, row in top_df.iterrows():
        # Find matching model in GalGA.mdf_data
        model_mdf_x, model_mdf_y = find_model_mdf(GalGA, row)
        # Interpolate to common grid
        mdf_interp = np.interp(feh_common, model_mdf_x, model_mdf_y, 
                               left=0, right=0)
        mdf_samples.append(mdf_interp)
    
    # Compute weighted percentiles at each [Fe/H]
    mdf_samples = np.array(mdf_samples)  # shape: (n_models, n_feh)
    
    median = np.zeros(len(feh_common))
    lower = np.zeros(len(feh_common))
    upper = np.zeros(len(feh_common))
    
    for i in range(len(feh_common)):
        pcts = weighted_quantile(mdf_samples[:, i], [0.16, 0.50, 0.84], weights)
        lower[i], median[i], upper[i] = pcts
    
    return {'x': feh_common, 'median': median, 'lower': lower, 'upper': upper}
```

### 2. Age-Metallicity Relation

**Challenge**: Time evolution with varying timesteps across models

**Solution**:
```python
def compute_age_feh_ensemble(GalGA, top_df, weights):
    # Define common age grid (0 to 14 Gyr)
    age_common = np.linspace(0, 14.0, 200)
    
    # Extract age-[Fe/H] tracks for each model
    age_feh_samples = []
    for idx, row in top_df.iterrows():
        # Find matching model in GalGA.age_data
        time_array, feh_array = find_model_age_data(GalGA, row)
        
        # Convert time to age (t_final - t) / 1e9
        age_gyr = (time_array[-1] - time_array) / 1e9
        
        # Interpolate to common age grid
        # Handle potential non-monotonicity by sorting
        sort_idx = np.argsort(age_gyr)
        age_sorted = age_gyr[sort_idx]
        feh_sorted = feh_array[sort_idx]
        
        feh_interp = np.interp(age_common, age_sorted, feh_sorted,
                               left=np.nan, right=np.nan)
        age_feh_samples.append(feh_interp)
    
    # Compute weighted percentiles at each age
    age_feh_samples = np.array(age_feh_samples)
    
    median = np.zeros(len(age_common))
    lower = np.zeros(len(age_common))
    upper = np.zeros(len(age_common))
    
    for i in range(len(age_common)):
        valid = np.isfinite(age_feh_samples[:, i])
        if np.sum(valid) > 0:
            pcts = weighted_quantile(age_feh_samples[valid, i], 
                                    [0.16, 0.50, 0.84], 
                                    weights[valid] / np.sum(weights[valid]))
            lower[i], median[i], upper[i] = pcts
        else:
            lower[i] = median[i] = upper[i] = np.nan
    
    return {'x': age_common, 'median': median, 'lower': lower, 'upper': upper}
```

### 3. Alpha Elements ([α/Fe] vs [Fe/H])

**Challenge**: 2D tracks in [Fe/H]-[α/Fe] space, need to handle as parametric curves

**Solution**:
```python
def compute_alpha_ensemble(GalGA, top_df, weights, element_idx):
    # Define common [Fe/H] grid
    feh_common = np.linspace(-2.0, 1.0, 150)
    
    # Extract alpha tracks for each model
    alpha_samples = []
    for idx, row in top_df.iterrows():
        # Find matching model in GalGA.alpha_data
        alpha_arrs = find_model_alpha_data(GalGA, row)
        
        if element_idx < len(alpha_arrs):
            feh_model, alpha_model = alpha_arrs[element_idx]
            
            # Apply smoothing (as in original code)
            feh_smooth, alpha_smooth = smooth_alpha_track_time_ordered(
                feh_model, alpha_model, sigma=3)
            
            # Interpolate to common [Fe/H] grid
            # Handle non-monotonicity: use parametric approach
            # or bin-based approach
            alpha_interp = interpolate_alpha_track(feh_smooth, alpha_smooth, 
                                                   feh_common)
            alpha_samples.append(alpha_interp)
    
    # Compute weighted percentiles at each [Fe/H]
    alpha_samples = np.array(alpha_samples)
    
    median = np.zeros(len(feh_common))
    lower = np.zeros(len(feh_common))
    upper = np.zeros(len(feh_common))
    
    for i in range(len(feh_common)):
        valid = np.isfinite(alpha_samples[:, i])
        if np.sum(valid) > 0:
            pcts = weighted_quantile(alpha_samples[valid, i], 
                                    [0.16, 0.50, 0.84], 
                                    weights[valid] / np.sum(weights[valid]))
            lower[i], median[i], upper[i] = pcts
        else:
            lower[i] = median[i] = upper[i] = np.nan
    
    return {'x': feh_common, 'median': median, 'lower': lower, 'upper': upper}
```

### 4. Physical Quantities (SFR, Infall, etc.)

**Challenge**: Requires reconstructing omega_plus models (computationally expensive)

**Solution**: Two-tier approach

**Option A: Full Reconstruction (Accurate but Slow)**
```python
def compute_physics_ensemble(GalGA, top_df, weights, max_models=20):
    """Reconstruct top N models and extract physics"""
    
    # Limit to top N models for computational efficiency
    if len(top_df) > max_models:
        # Resample with replacement according to weights
        indices = np.random.choice(len(top_df), size=max_models, 
                                   replace=True, p=weights)
        top_df_subset = top_df.iloc[indices]
        weights_subset = np.ones(max_models) / max_models
    else:
        top_df_subset = top_df
        weights_subset = weights
    
    # Reconstruct each model
    physics_samples = {
        'sfr': [], 'inflow': [], 'outflow': [], 
        'gas_mass': [], 'stellar_mass': [], 'metallicity': []
    }
    
    for idx, row in top_df_subset.iterrows():
        GCE_model = reconstruct_model_from_params(GalGA, row)
        
        # Extract physical quantities
        ages = np.array(GCE_model.inner.history.age) / 1e9
        physics_samples['sfr'].append((ages[:-1], GCE_model.inner.history.sfr_abs[:-1]))
        # ... extract other quantities
    
    # Interpolate to common age grid and compute percentiles
    age_common = np.linspace(0, 14.0, 200)
    results = {}
    
    for key in physics_samples.keys():
        samples_interp = []
        for ages, values in physics_samples[key]:
            values_interp = np.interp(age_common, ages, values,
                                     left=np.nan, right=np.nan)
            samples_interp.append(values_interp)
        
        samples_interp = np.array(samples_interp)
        median, lower, upper = compute_percentile_bands(samples_interp, 
                                                        weights_subset)
        results[key] = {'x': age_common, 'median': median, 
                       'lower': lower, 'upper': upper}
    
    return results
```

**Option B: Parameter-Based Approximation (Fast but Approximate)**
```python
def compute_physics_ensemble_approx(GalGA, top_df, weights):
    """Use parameter distributions to estimate physics uncertainty"""
    
    # Compute parameter percentiles
    param_pcts = {}
    for param in ['sigma_2', 't_1', 't_2', 'infall_1', 'infall_2', 
                  'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb']:
        pcts = weighted_quantile(top_df[param].values, 
                                [0.16, 0.50, 0.84], weights)
        param_pcts[param] = {'lower': pcts[0], 'median': pcts[1], 
                            'upper': pcts[2]}
    
    # Reconstruct 3 models: median, lower bound, upper bound
    # Use median for most params, vary key params for bounds
    models = {
        'median': reconstruct_from_param_dict(GalGA, 
                    {k: v['median'] for k, v in param_pcts.items()}),
        'lower': reconstruct_from_param_dict(GalGA, 
                    {k: v['lower'] for k, v in param_pcts.items()}),
        'upper': reconstruct_from_param_dict(GalGA, 
                    {k: v['upper'] for k, v in param_pcts.items()})
    }
    
    # Extract physics from each
    results = extract_physics_from_models(models)
    
    return results
```

**Recommendation**: Use Option A with `max_models=20-50` for accuracy while maintaining reasonable computation time.

## Visualization Enhancements

### Color Scheme
- **Median line**: Bold red/crimson (lw=2.5)
- **1σ band**: Semi-transparent red (alpha=0.3)
- **Observations**: Black points/markers (unchanged)
- **All models (optional)**: Very light gray (alpha=0.02-0.05) in background

### Legend Updates
```python
ax.plot(x, median, color='crimson', lw=2.5, label='Median model')
ax.fill_between(x, lower, upper, color='crimson', alpha=0.3, 
                label='1σ posterior')
```

### Residuals Panel
For plots with residuals (MDF, Age-[Fe/H]):
- Compute residuals for median model
- Show uncertainty band in residuals panel
- Propagate observation uncertainties if available

## Implementation Priority

### Phase 1: Core Plots (High Priority)
1. `plot_mdf_curves()` - Simplest case, 1D histogram
2. `plot_age_feh_detailed()` - Age-metallicity with residuals
3. `plot_four_panel_alpha()` - Alpha elements

### Phase 2: Physics Plots (Medium Priority)
4. `plot_real_infall_physics()` - Infall rates, SFR, etc.
5. `plot_omega_diagnostics()` - Additional diagnostics

### Phase 3: Integration (Low Priority)
6. Update wrapper functions that call these plots
7. Add command-line flags for posterior mode
8. Update documentation

## Testing Strategy

1. **Synthetic Test**: Create mock results_df with known distribution
2. **Visual Inspection**: Ensure bands look reasonable (not too wide/narrow)
3. **Consistency Check**: Median should pass through best model region
4. **Edge Cases**: Test with small sample sizes, degenerate parameters
5. **Performance**: Time reconstruction of N models

## Configuration Options

Add to function signatures:
```python
posterior_config = {
    'use_posterior': True,           # Enable posterior mode
    'percentile': 10,                # Top X% of models
    'max_models_physics': 30,        # Max models for expensive reconstructions
    'show_all_models': False,        # Show gray background of all models
    'credible_interval': 68,         # 68% (1σ) or 95% (2σ)
}
```

## Documentation Requirements

1. **Docstring Updates**: Explain new parameters and behavior
2. **Example Usage**: Show how to enable/disable posterior mode
3. **Interpretation Guide**: How to read uncertainty bands
4. **Performance Notes**: Warn about computational cost for physics plots

## Backward Compatibility

Ensure existing code continues to work:
```python
# Old behavior (default)
plot_mdf_curves(GalGA, feh, count, results_df)

# New behavior (opt-in)
plot_mdf_curves(GalGA, feh, count, results_df, use_posterior=True)
```

## Next Steps

1. Implement `posterior_utils.py` with core functions
2. Modify `plot_mdf_curves()` as proof-of-concept
3. Test with actual data
4. Iterate on design based on results
5. Apply to remaining plotting functions

