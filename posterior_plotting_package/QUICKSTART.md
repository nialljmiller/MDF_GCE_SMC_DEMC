# Quick Start Guide: Posterior Uncertainty Plotting

## TL;DR

Replace your single "best" model plots with posterior distributions (median + 1σ bands) in 3 steps:

### Step 1: Copy Files

```bash
cp posterior_utils.py /path/to/your/project/
cp core_plots_posterior.py /path/to/your/project/
cp phys_plot_posterior.py /path/to/your/project/
```

### Step 2: Update Imports

```python
# OLD
from core_plots import plot_age_feh_detailed, plot_mdf_curves, plot_four_panel_alpha
from phys_plot import plot_real_infall_physics

# NEW
from core_plots_posterior import plot_age_feh_detailed, plot_mdf_curves, plot_four_panel_alpha
from phys_plot_posterior import plot_real_infall_physics
```

### Step 3: Add `use_posterior=True`

```python
# Core plots (fast)
plot_age_feh_detailed(GalGA, Fe_H, age_Joyce, age_Bensby, 
                     results_df=results_df, use_posterior=True)

plot_mdf_curves(GalGA, feh, count, 
               results_df=results_df, use_posterior=True)

plot_four_panel_alpha(GalGA, Fe_H, Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe,
                     results_df=results_df, use_posterior=True)

# Physics plots (slow, ~5-10 min)
plot_real_infall_physics(GalGA, results_df=results_df, 
                        use_posterior=True, max_models=20)
```

## What You Get

### Before (Single Best Model)
- Single red line = "best" model
- Gray lines = all other models (hard to interpret)
- No uncertainty quantification

### After (Posterior Distribution)
- **Bold line** = median model (50th percentile)
- **Shaded band** = 1σ uncertainty (16th-84th percentile)
- Properly represents model uncertainty and degeneracy

## Key Parameters

- `use_posterior=True` - Enable posterior mode (default: True)
- `percentile=10` - Use top 10% of models (default: 10)
- `max_models=20` - For physics plots only, limit reconstructions (default: 20)

## Troubleshooting

### "Could not compute ensemble"
→ Check that `results_df` has columns: `sigma_2`, `t_2`, `infall_2`, `fitness`

### Bands too wide
→ Reduce `percentile` from 10 to 5

### Bands too narrow
→ Increase `percentile` from 10 to 20

### Physics plots too slow
→ Reduce `max_models` from 20 to 10

## Legacy Mode

To revert to original single-best behavior:

```python
plot_age_feh_detailed(GalGA, Fe_H, age_Joyce, age_Bensby,
                     results_df=results_df, use_posterior=False)
```

## Full Documentation

See `POSTERIOR_PLOTTING_README.md` for complete documentation.

