# Output File Format Reference

Detailed documentation of all output files produced by the MDF_GCE_SMC_DEMC pipeline.

## Table of Contents

- [Directory Structure](#directory-structure)
- [CSV Files](#csv-files)
- [NPZ Files](#npz-files)
- [Plot Files](#plot-files)
- [Checkpoint Files](#checkpoint-files)

---

## Directory Structure

A typical run produces the following structure:

```
output_path/                          # Specified in bulge_pcard.txt
├── simulation_results.csv            # Final merged results
├── simulation_results_gen_10.csv     # Per-generation snapshots
├── simulation_results_gen_20.csv
├── ...
├── ga_population_samples.csv         # Full GA evaluation history
├── walker_history.npz                # Walker trajectories
├── chains.csv                        # SMC-DEMC chain log
├── smc_demc_samples.csv             # Posterior samples
├── posterior_samples.csv            # Legacy alias for above
├── posteriors.csv                   # Fitness-weighted posterior
├── smc_demc_posterior_corner.png    # Corner plot
├── mdf_fit.png                      # MDF comparison
├── age_feh.png                      # Age-metallicity plot
├── physics_*.png                    # Physics diagnostic plots
└── ga_checkpoint.pkl                # Checkpoint for resumption
```

---

## CSV Files

### `simulation_results.csv`

**Description**: Master table of all evaluated models with their parameters and loss values.

**Columns**:

| Column | Type | Description |
|--------|------|-------------|
| `comp_idx` | int | Composition array index (0-5) |
| `imf_idx` | int | IMF array index |
| `sn1a_idx` | int | SNe Ia yield table index |
| `sy_idx` | int | Stellar yield table index |
| `sn1ar_idx` | int | SNe Ia rate model index |
| `sigma_2` | float | Second/first infall mass ratio |
| `t_1` | float | First infall onset time [Gyr] |
| `t_2` | float | Second infall onset time [Gyr] |
| `infall_1` | float | First infall timescale [Gyr] |
| `infall_2` | float | Second infall timescale [Gyr] |
| `sfe` | float | Star formation efficiency [Gyr⁻¹] |
| `delta_sfe` | float | SFE change at second infall |
| `imf_upper` | float | IMF upper mass limit [M☉] |
| `mgal` | float | Initial gas mass [M☉] |
| `nb` | float | SNe Ia per solar mass [M☉⁻¹] |
| `loss` | float | Total loss value (lower is better) |
| `mdf_loss` | float | MDF component of loss (if tracked) |
| `age_loss` | float | Age-[Fe/H] component (if tracked) |

**Notes**:
- Sorted by `loss` in ascending order (best models first)
- May contain duplicate parameter sets if re-evaluated
- Use `drop_duplicates()` for unique models

**Example Usage**:
```python
import pandas as pd
df = pd.read_csv('SMC_DEMC/simulation_results.csv')
best = df.iloc[0]  # Best model
top_10pct = df.head(int(len(df) * 0.1))  # Top 10%
```

---

### `simulation_results_gen_N.csv`

**Description**: Snapshot of results after generation N.

**Format**: Same as `simulation_results.csv`

**Notes**:
- Useful for tracking convergence
- Generated every `output_interval` generations
- Final generation also produces `simulation_results.csv`

---

### `ga_population_samples.csv`

**Description**: Every individual evaluated during the GA, including duplicates and rejected mutations.

**Additional Columns**:

| Column | Type | Description |
|--------|------|-------------|
| `generation` | int | GA generation number |
| `evaluation` | int | Evaluation counter |

**Notes**:
- Larger than `simulation_results.csv` (includes all evaluations)
- Useful for analyzing GA exploration behavior
- Can be processed as an ensemble sampler output

---

### `chains.csv`

**Description**: Full SMC-DEMC chain log recording every stage transition.

**Columns**:

| Column | Type | Description |
|--------|------|-------------|
| `stage` | int | SMC stage number (0 = initial) |
| `pid` | int | Particle ID (0 to N-1) |
| `accepted` | bool | Whether DE-MH move was accepted |
| `comp_idx` | int | Composition index |
| `imf_idx` | int | IMF index |
| `sn1a_idx` | int | SNe Ia yield index |
| `sy_idx` | int | Stellar yield index |
| `sn1ar_idx` | int | SNe Ia rate index |
| `tmax_1` | float | First infall time [Gyr] |
| `tmax_2` | float | Second infall time [Gyr] |
| `infall_timescale_1` | float | First timescale [Gyr] |
| `infall_timescale_2` | float | Second timescale [Gyr] |
| `sfe` | float | SFE [Gyr⁻¹] |
| `delta_sfe` | float | ΔSFE |
| `imf_upper_limits` | float | IMF upper [M☉] |
| `mgal_values` | float | Gas mass [M☉] |
| `nb_array` | float | N_Ia [M☉⁻¹] |

**Notes**:
- Records full MCMC trajectory
- `accepted=True` indicates successful Metropolis-Hastings step
- Stage 0 is the initial GA ensemble

---

### `smc_demc_samples.csv` / `posterior_samples.csv`

**Description**: Resampled posterior draws with burn-in removed.

**Columns**: Continuous parameters only:

| Column | Type | Units |
|--------|------|-------|
| `sigma_2` | float | dimensionless |
| `tmax_1` | float | Gyr |
| `tmax_2` | float | Gyr |
| `infall_timescale_1` | float | Gyr |
| `infall_timescale_2` | float | Gyr |
| `sfe` | float | Gyr⁻¹ |
| `delta_sfe` | float | Gyr⁻¹ |
| `imf_upper_limits` | float | M☉ |
| `mgal_values` | float | M☉ |
| `nb_array` | float | M☉⁻¹ |

**Notes**:
- `posterior_samples.csv` is a legacy alias pointing to the same data
- Default: 200,000 samples after 20% burn-in
- Ready for direct use with `corner.corner()`

---

### `posteriors.csv`

**Description**: Fitness-weighted posterior constructed from GA results.

**Columns**: Same as posterior_samples.csv plus:

| Column | Type | Description |
|--------|------|-------------|
| `weight` | float | Importance weight |

**Notes**:
- Alternative posterior construction method
- Uses exponential weighting: w ∝ exp(-loss/T)
- Temperature T auto-tuned for ~30% ESS

---

## NPZ Files

### `walker_history.npz`

**Description**: Complete walker trajectories with associated simulation data.

**Arrays**:

| Key | Shape | Description |
|-----|-------|-------------|
| `walker_ids` | (N,) | Walker ID numbers |
| `histories` | (N,) object | List of parameter vectors per walker |
| `mdf_data` | (M,) object | MDF curves: [(feh_bins, counts), ...] |
| `alpha_data` | (M,) object | [α/Fe] vs [Fe/H] data |
| `age_data` | (M,) object | Age-[Fe/H] relations: [(time, feh), ...] |

**Loading**:
```python
import numpy as np
data = np.load('walker_history.npz', allow_pickle=True)
walker_ids = data['walker_ids']
histories = data['histories']
mdf_data = data['mdf_data']
```

**Notes**:
- `histories[i]` is a list of 15-element parameter vectors for walker i
- MDF/alpha/age arrays are appended sequentially, not aligned with walkers
- Use `make_history.py` to cross-match with loss values

---

### `history_with_loss.npz`

**Description**: Walker history cross-matched with loss values from results CSV.

**Additional Arrays**:

| Key | Shape | Description |
|-----|-------|-------------|
| `losses` | (N,) | Loss values aligned with walker_ids |
| `inds` | (N,) object | Parameter vectors aligned with losses |

**Notes**:
- Generated by `make_history.py` or `make_history_gpt.py`
- NaN losses indicate walkers not found in results CSV

---

## Plot Files

### `smc_demc_posterior_corner.png`

**Description**: Corner plot of SMC-DEMC posterior samples.

**Features**:
- 1D marginal histograms on diagonal
- 2D contour plots off-diagonal
- 16th, 50th, 84th percentile quantiles
- Grayscale color scheme

---

### `mdf_fit.png`

**Description**: Comparison of model MDF vs observed MDF.

**Contents**:
- Observed MDF with error bars
- Best-fit model MDF
- Posterior uncertainty band (if using posterior mode)

---

### `age_feh.png`

**Description**: Age-metallicity relation comparison.

**Contents**:
- Joyce et al. and/or Bensby et al. data points
- Model age-[Fe/H] tracks
- Uncertainty bands from posterior

---

### `physics_*.png`

**Description**: Physical evolution diagnostics.

**Variants**:
- `physics_infall.png`: Infall rate vs time
- `physics_sfr.png`: Star formation rate vs time
- `physics_gas.png`: Gas mass evolution
- `physics_stellar.png`: Stellar mass buildup
- `physics_metals.png`: Metallicity evolution

---

## Checkpoint Files

### `ga_checkpoint.pkl`

**Description**: Pickle file for resuming interrupted runs.

**Contents**:
- Current population
- Generation number
- Random state
- Walker history
- Results accumulated so far

**Usage**:
```python
import pickle
with open('ga_checkpoint.pkl', 'rb') as f:
    checkpoint = pickle.load(f)
population = checkpoint['population']
start_gen = checkpoint['generation']
```

**Notes**:
- Automatically loaded on restart if present
- Deleted after successful completion (optional)
