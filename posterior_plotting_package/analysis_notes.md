# Code Analysis: Posterior Uncertainty Visualization

## Current State

### Data Flow
1. **Results Storage**: MCMC/GA samples stored in `simulation_results.csv` or `smc_demc_samples.csv`
2. **Current Plotting**: `core_plots.py` and `phys_plot.py` plot single "best" model (first row of sorted results)
3. **Posterior Analysis**: `uncertainty_analysis.py` performs posterior quantification with KDE and corner plots

### Key Functions in core_plots.py
- `plot_age_feh_detailed()`: Age-metallicity relation with best model (red line) + all models (grey)
- `plot_mdf_curves()`: MDF with best model + residuals
- `plot_four_panel_alpha()`: 4-panel alpha elements with best model tracks

### Key Functions in phys_plot.py
- `reconstruct_best_model()`: Rebuilds omega_plus model from best parameters
- `plot_real_infall_physics()`: Plots infall rates, SFR, gas flows, etc. for best model

### Key Functions in uncertainty_analysis.py
- `plot_marginalized_posteriors_kde_weighted()`: 1D posteriors with fitness-weighted KDE
- `plot_corner_2d_kde()`: Corner plots with 2D KDE contours
- Provides weighted quantiles (16th, 50th, 84th percentiles)

## Problem Statement

The current approach plots a single "best" model, but the method is designed to sample parameter space and construct posterior distributions. The goal is to:

1. **Show median/mean** instead of single best value
2. **Show 1-sigma uncertainty bands** around the median/mean
3. Handle **high dimensionality** and **degeneracy** in parameter space

## Proposed Solution Strategy

### Phase 1: Extract Posterior Statistics
From `uncertainty_analysis.py`, we can leverage:
- Fitness-weighted samples (top 10% by fitness)
- Weighted quantiles (16th, 50th, 84th percentiles)
- KDE for smooth posterior distributions

### Phase 2: Modify core_plots.py
For each plot type, we need to:

1. **MDF Plot**:
   - Compute MDF for each model in top N%
   - Calculate median MDF at each [Fe/H] bin
   - Calculate 16th/84th percentile bands
   - Plot median line + shaded uncertainty region

2. **Age-Metallicity Plot**:
   - Compute age-[Fe/H] tracks for top N% models
   - Interpolate to common age grid
   - Calculate median and percentile bands at each age
   - Plot median track + uncertainty band

3. **Alpha Elements Plot**:
   - Similar approach for [α/Fe] vs [Fe/H] tracks
   - Compute median and percentile bands
   - Plot for each element (Mg, Si, Ca, Ti)

### Phase 3: Modify phys_plot.py
For physics plots:

1. **Reconstruct Multiple Models**:
   - Instead of single best model, reconstruct top N% models
   - Extract physical quantities (SFR, infall, etc.) from each

2. **Compute Posterior Bands**:
   - Calculate median and percentiles for each physical quantity
   - Handle time-series data with interpolation to common grid

3. **Plot with Uncertainty**:
   - Replace single line with median + shaded bands

## Implementation Details

### Weighted Sampling Strategy
```python
# From uncertainty_analysis.py approach:
def get_weighted_posterior_samples(df, fitness_col='fitness', percentile=10):
    """Get top percentile with fitness weights"""
    n_top = max(1, int(len(df) * percentile / 100))
    top = df.head(n_top)
    
    # Inverse fitness weights (lower is better)
    fit = top[fitness_col].values
    eps = np.min(fit) * 0.001
    w_raw = 1.0 / (fit + eps)
    w = w_raw / np.sum(w_raw)
    
    return top, w
```

### Percentile Band Calculation
```python
def compute_percentile_bands(x_grid, y_samples, weights, percentiles=[16, 50, 84]):
    """Compute weighted percentile bands on common grid"""
    bands = []
    for x_val in x_grid:
        # Get y values at this x from all samples
        y_at_x = [interpolate_sample(sample, x_val) for sample in y_samples]
        # Compute weighted percentiles
        pcts = weighted_quantile(y_at_x, percentiles, weights)
        bands.append(pcts)
    return np.array(bands)
```

## Key Challenges

1. **Interpolation**: Different models may have different time grids or [Fe/H] grids
2. **Degeneracy**: High parameter degeneracy may lead to wide uncertainty bands
3. **Performance**: Reconstructing many models (especially omega_plus) is computationally expensive
4. **Visualization**: Balancing clarity with information density

## Next Steps

1. Create helper functions for posterior statistics extraction
2. Modify each plotting function in core_plots.py
3. Modify physics plotting in phys_plot.py
4. Test with actual data to ensure reasonable uncertainty bands
5. Document the approach for reproducibility

