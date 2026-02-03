# Analysis Guide

Step-by-step guide for analyzing MDF_GCE_SMC_DEMC results, generating posterior plots, and troubleshooting common issues.

## Table of Contents

- [Quick Start](#quick-start)
- [Running Posterior Analysis](#running-posterior-analysis)
- [Interpreting Corner Plots](#interpreting-corner-plots)
- [Combining Multiple Runs](#combining-multiple-runs)
- [Physics Reconstruction](#physics-reconstruction)
- [Custom Analysis](#custom-analysis)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

After a successful run, you'll have results in your output directory (default: `SMC_DEMC/`).

```bash
# View best model parameters
head -2 SMC_DEMC/simulation_results.csv | column -t -s,

# Quick corner plot
python standalone_plots/plot_corner.py SMC_DEMC/simulation_results.csv

# Full posterior analysis
python posterior_analysis_code/posterior_analysis_plots.py
```

---

## Running Posterior Analysis

### Interactive Mode

The main analysis script provides interactive folder selection:

```bash
python posterior_analysis_code/posterior_analysis_plots.py
```

Output:
```
Folders with simulation results:
1: SMC_DEMC
2: old_run_backup
3: bc_combined_MDF
Enter the numbers of folders to analyze (comma-separated): 1
```

### Direct Analysis

For scripted/automated analysis:

```python
from posterior_analysis_code.uncertainty_analysis import UncertaintyAnalysis

# Load results
ua = UncertaintyAnalysis('SMC_DEMC/simulation_results.csv')
ua.bulge_pcard_path = 'bulge_pcard.txt'

# Generate corner plot (top 10% by loss, fitness-weighted)
ua.plot_posterior_corner(
    percentile=10,      # Use top 10% of models
    weight_power=1.0,   # Exponential weighting power
    save_path='my_corner.png'
)

# Get statistics
top_df, weights = ua._select_top_and_weights(percentile=10)
print(f"Using {len(top_df)} models")
print(f"Effective sample size: {1/np.sum(weights**2):.1f}")
```

### Command-Line Analysis

```bash
python plotting/posterior_analysis.py \
    --results SMC_DEMC/simulation_results.csv \
    --pcard bulge_pcard.txt \
    --nsamples 5000 \
    --output SMC_DEMC/analysis/
```

Options:
- `--results`: Path to simulation_results.csv
- `--history`: Path to walker_history.npz (optional)
- `--pcard`: Parameter card file
- `--nsamples`: Number of posterior samples to draw
- `--temperature`: Manual temperature for weighting (auto-tuned if omitted)
- `--params`: Subset of parameters to include

---

## Interpreting Corner Plots

### Reading the Plot

Corner plots show:

1. **Diagonal panels**: 1D marginalized posteriors
   - Peak = most probable value
   - Width = uncertainty
   - Titles show median and ±1σ quantiles

2. **Off-diagonal panels**: 2D joint distributions
   - Contours show 1σ, 2σ, 3σ credible regions
   - Elongated contours indicate correlation
   - Diagonal orientation = positive correlation
   - Anti-diagonal = negative correlation

### Key Parameter Correlations

Common physical correlations to look for:

| Parameters | Expected Correlation | Physical Reason |
|------------|---------------------|-----------------|
| `t_1` vs `infall_1` | Negative | Earlier onset with shorter duration |
| `sfe` vs `mgal` | Negative | Higher SFE needs less gas for same enrichment |
| `sigma_2` vs `t_2` | Variable | Depends on MDF shape |
| `nb` vs `sfe` | Positive | SNe Ia rate and SFR coupled |

### Assessing Convergence

Good convergence indicators:
- Smooth, unimodal posteriors
- Contours are elliptical (not banana-shaped)
- Parameter ranges don't hit prior bounds

Poor convergence warnings:
- Multimodal posteriors (multiple peaks)
- Parameters piled at prior edges
- Very irregular contour shapes

---

## Combining Multiple Runs

When you have multiple independent runs (different initial conditions, different yields, etc.):

### Automatic Combination

```bash
python posterior_analysis_code/combine_posterior.py
```

This:
1. Scans current directory for result folders
2. Prompts for selection
3. Merges `simulation_results.csv` files
4. Combines `walker_history.npz` files
5. Outputs to `bc_combined_MDF/`

### Manual Combination

```python
import pandas as pd
import numpy as np

# Combine CSVs
df1 = pd.read_csv('run1/simulation_results.csv')
df2 = pd.read_csv('run2/simulation_results.csv')
combined = pd.concat([df1, df2], ignore_index=True)
combined = combined.sort_values('loss').reset_index(drop=True)
combined.to_csv('combined/simulation_results.csv', index=False)

# Remove duplicates if needed
combined = combined.drop_duplicates(subset=[
    'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2',
    'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb'
])
```

### Combined Analysis

```bash
python posterior_analysis_code/combined_analysis.py
```

Features:
- Generates comparison plots across runs
- Exports best models per folder
- Creates merged posteriors

---

## Physics Reconstruction

Reconstruct physical evolution from posterior samples:

### Full Physics Plot

```python
from posterior_plotting_package.test.phys_plot_posterior import (
    plot_real_infall_physics,
    compute_physics_ensemble
)

# With uncertainty bands from posterior
plot_real_infall_physics(
    GalGA,
    results_df=results_df,
    save_path='physics_with_posterior.png',
    use_posterior=True,
    percentile=10,
    max_models=50  # Number of models to average
)
```

### Individual Quantities

```python
from posterior_plotting_package.core.model_reconstruction import (
    reconstruct_model_from_params,
    extract_physics_from_model
)

# Reconstruct a single model
params = df.iloc[0].to_dict()
model = reconstruct_model_from_params(GalGA, params)
physics = extract_physics_from_model(model)

# Access individual histories
ages = physics['ages']           # Time array [yr]
sfr = physics['sfr_rates']       # Star formation rate
inflow = physics['inflow_rates'] # Gas infall rate
gas_mass = physics['gas_mass']   # Gas mass evolution
metallicity = physics['feh']     # [Fe/H] vs time
```

### Ensemble Statistics

```python
ensemble = compute_physics_ensemble(GalGA, top_df, weights, max_models=50)

# Median and bands for each quantity
sfr_median = ensemble['sfr']['median']
sfr_lower = ensemble['sfr']['lower']    # 16th percentile
sfr_upper = ensemble['sfr']['upper']    # 84th percentile
```

---

## Custom Analysis

### Extracting Parameter Constraints

```python
import numpy as np
from scipy.stats import gaussian_kde

# Load posterior samples
samples = pd.read_csv('SMC_DEMC/posterior_samples.csv')

# Get credible intervals for a parameter
param = 'sigma_2'
values = samples[param].values
median = np.median(values)
lower, upper = np.percentile(values, [16, 84])
print(f"{param} = {median:.3f} +{upper-median:.3f} -{median-lower:.3f}")

# Full distribution
kde = gaussian_kde(values)
x = np.linspace(values.min(), values.max(), 200)
pdf = kde(x)
```

### Custom Corner Plot

```python
import corner

# Select parameters to plot
params = ['sigma_2', 't_2', 'sfe', 'mgal']
samples = pd.read_csv('SMC_DEMC/posterior_samples.csv')[params]

# Custom labels
labels = [r'$\sigma_2$', r'$t_2$ [Gyr]', r'SFE [Gyr$^{-1}$]', r'$M_{\rm gal}$ [$M_\odot$]']

fig = corner.corner(
    samples.values,
    labels=labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt='.2f'
)
fig.savefig('custom_corner.png', dpi=300)
```

### Cross-Matching with Walker History

```python
# Load walker history
data = np.load('SMC_DEMC/walker_history.npz', allow_pickle=True)
results = pd.read_csv('SMC_DEMC/simulation_results.csv')

# Get MDF for best model
best_params = results.iloc[0]
# ... match to walker history by parameters
```

---

## Troubleshooting

### Common Issues

#### "No simulation_results*.csv found"

**Cause**: Run didn't complete or output path is wrong.

**Solution**:
1. Check `output_path` in bulge_pcard.txt
2. Look for checkpoint files to resume
3. Check SLURM logs for errors

#### Corner plot shows parameters at prior bounds

**Cause**: Prior range too narrow for the data.

**Solution**:
1. Expand parameter ranges in bulge_pcard.txt
2. Check if best-fit is physically reasonable
3. May indicate model inadequacy

#### Very wide posteriors (poor constraints)

**Cause**: Insufficient data, degenerate parameters, or not enough generations.

**Solution**:
1. Run more generations
2. Increase population size
3. Check for parameter degeneracies in corner plot
4. Consider fixing poorly-constrained parameters

#### "ValueError: No continuous parameters found"

**Cause**: Column names don't match expected names.

**Solution**:
```python
# Check available columns
df = pd.read_csv('simulation_results.csv')
print(df.columns.tolist())

# Map to standard names if needed
rename_map = {
    'old_name': 'sigma_2',
    # ...
}
df = df.rename(columns=rename_map)
```

#### Memory error during analysis

**Cause**: Too many models or large walker history.

**Solution**:
1. Use `percentile` parameter to limit models:
   ```python
   ua.plot_posterior_corner(percentile=5)  # Top 5% only
   ```
2. Process in chunks
3. Increase system memory

#### SMC-DEMC didn't run

**Cause**: `run_dmc = False` in Gal_GA_PP.py

**Solution**:
1. Check the flag in your version
2. Run SMC-DEMC separately:
   ```bash
   sbatch smc_demc_sbatch.sh
   ```

### Log Files

Check these logs for debugging:

| File | Contents |
|------|----------|
| `slurm_output.log` | Standard output from SLURM |
| `slurm_error.log` | Errors and warnings |
| `MDF_SMC_DEMC_runtime.log` | Detailed runtime log |

### Getting Help

1. Check existing documentation in `docs/`
2. Review docstrings in source code
3. Open an issue on the repository

---

## Best Practices

1. **Always check convergence** before trusting posteriors
2. **Save intermediate results** using checkpoint files
3. **Document your runs** with parameter cards and notes
4. **Combine multiple runs** for better coverage
5. **Use meaningful output paths** to organize experiments
6. **Version control** your parameter cards
