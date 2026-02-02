# MDF_GCE_SMC_DEMC

Metallicity Distribution Function fitting via Genetic Algorithm with Differential Evolution Markov Chain Monte Carlo refinement for Galactic Chemical Evolution modeling.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Files](#output-files)
- [Analysis Tools](#analysis-tools)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Citation](#citation)

## Features

- **Hybrid GA + DE-MC optimization**: Genetic algorithm drives global exploration while DE-MC moves are injected into each generation for local refinement
- **SMC-DEMC posterior refinement**: Tempered Sequential Monte Carlo with DE-MH moves produces calibrated posterior samples
- **Configurable parameter bounds**: All physical parameters controlled via `bulge_pcard.txt`
- **Multi-objective loss**: Configurable weighting between MDF fitting and age-metallicity relation
- **Built-in posterior analysis**: Uncertainty quantification, corner plots, and physics reconstruction
- **SLURM batch job support**: Ready-to-use scripts for HPC clusters
- **Comprehensive diagnostics**: Per-generation snapshots, walker histories, and convergence tracking

## Installation

### Requirements

- Python 3.8+
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- Pandas ≥ 1.3
- Matplotlib ≥ 3.4
- corner ≥ 2.2
- DEAP ≥ 1.3
- seaborn ≥ 0.11
- scikit-learn ≥ 0.24

### Install

```bash
git clone https://github.com/[user]/MDF_GCE_SMC_DEMC.git
cd MDF_GCE_SMC_DEMC
pip install -r requirements.txt
```

### NuPyCEE Dependency

The chemical evolution engine (OMEGA+) is included as a submodule or local copy. Ensure `NuPyCEE/` is present in the project root.

## Quick Start

```bash
# Run with default configuration
python MDF_SMC_DEMC_Launcher.py

# Run with custom output directory (edit bulge_pcard.txt first)
# output_path: 'my_run/'

# Minimal plotting (faster)
python MDF_SMC_DEMC_Launcher.py --plot-mode posterior_minimal

# Full diagnostic suite
python MDF_SMC_DEMC_Launcher.py --plot-mode full
```

The launcher executes the full pipeline: GA exploration → SMC-DEMC refinement → posterior analysis.

## Configuration

### Parameter Card (`bulge_pcard.txt`)

The parameter card controls all aspects of the optimization. Key sections:

#### Infall Parameters

| Parameter | Description | Units | Typical Range |
|-----------|-------------|-------|---------------|
| `sigma_2_list` | Mass ratio of second to first infall | dimensionless | [0.1, 5.0] |
| `tmax_1_list` | First infall onset time | Gyr since Big Bang | [0.005, 0.1] |
| `tmax_2_list` | Second infall onset time | Gyr since Big Bang | [0.1, 10.0] |
| `infall_timescale_1_list` | First infall duration | Gyr | [0.001, 0.1] |
| `infall_timescale_2_list` | Second infall duration | Gyr | [0.1, 10.0] |

#### Star Formation Parameters

| Parameter | Description | Units | Typical Range |
|-----------|-------------|-------|---------------|
| `sfe_array` | Star formation efficiency | Gyr⁻¹ | [1.0, 20.0] |
| `delta_sfe_array` | Change in SFE at second infall | Gyr⁻¹ | [0.01, 0.85] |

#### Stellar Physics Parameters

| Parameter | Description | Units | Typical Range |
|-----------|-------------|-------|---------------|
| `imf_array` | Initial mass function | categorical | ['salpeter', 'chabrier', 'kroupa'] |
| `imf_upper_limits` | IMF upper mass cutoff | M☉ | [60, 130] |
| `mgal_values` | Initial bulge gas mass | M☉ | [1e9, 1e11] |
| `nb_array` | SNe Ia per solar mass formed | M☉⁻¹ | [0.5e-3, 1.5e-3] |

#### GA Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `population_size` | Number of individuals per generation | 96 |
| `num_generations` | Total GA generations | 100 |
| `mutation_probability` | Per-gene mutation chance | 0.1 |
| `crossover_probability` | Crossover rate | 0.7 |
| `timesteps` | OMEGA+ time resolution | 100 |

#### Loss Configuration

| Parameter | Description |
|-----------|-------------|
| `mdf_vs_age_weight` | Weight for MDF vs age-[Fe/H] loss (1.0 = MDF only) |
| `obs_age_data_loss_metric` | Loss metric: 'mae', 'rmse', 'rms', etc. |
| `obs_age_data_target` | Age data source: 'joyce' or 'bensby' |

### Example Configuration

```yaml
# Two-infall model for Galactic bulge
sigma_2_list: [0.1, 5.0]
tmax_1_list: [0.005, 0.1]
tmax_2_list: [0.1, 10.0]
sfe_array: [1.0, 20.0]
mdf_vs_age_weight: 0.8

# GA settings
population_size: 96
num_generations: 100
```

## Usage

### Local Execution

```bash
python MDF_SMC_DEMC_Launcher.py
```

### SLURM Batch Execution

Several SLURM scripts are provided:

```bash
# Standard 96-core run
sbatch submit_mdf_96core.sh

# Large-memory run
sbatch submit_mdf_bigcore.sh

# Posterior-only from existing GA
sbatch smc_demc_sbatch.sh

# Sweep over multiple configurations
./launch_many.sh
```

### Plot Modes

| Mode | Description |
|------|-------------|
| `posterior_minimal` | MDF fits, alpha comparison, physics plots, posterior summary |
| `full` | Complete GA diagnostic suite including 3D scatter plots |

## Output Files

All outputs are written to the directory specified by `output_path` in the pcard (default: `SMC_DEMC/`).

### Core Results

| File | Description |
|------|-------------|
| `simulation_results.csv` | All evaluated models with loss values |
| `simulation_results_gen_N.csv` | Per-generation snapshots |
| `ga_population_samples.csv` | Every GA evaluation for ensemble analysis |

### Walker Tracking

| File | Description |
|------|-------------|
| `walker_history.npz` | Full walker trajectories with MDF/alpha/age data |
| `history_with_loss.npz` | Walker history cross-matched with loss values |

### Posterior Products

| File | Description |
|------|-------------|
| `chains.csv` | Full SMC-DEMC chain log (stage, particle, accepted, params) |
| `smc_demc_samples.csv` | Resampled posterior draws (burn-in removed) |
| `posterior_samples.csv` | Legacy alias for posterior draws |
| `posteriors.csv` | Fitness-weighted posterior from GA results |

### Plots

| File | Description |
|------|-------------|
| `smc_demc_posterior_corner.png` | Parameter corner plot from SMC-DEMC |
| `mdf_fit.png` | MDF model vs observations |
| `age_feh.png` | Age-metallicity relation |
| `physics_*.png` | Infall, SFR, gas mass evolution |

### CSV Column Reference

The `simulation_results.csv` contains:

| Column | Description |
|--------|-------------|
| `comp_idx`, `imf_idx`, `sn1a_idx`, `sy_idx`, `sn1ar_idx` | Categorical parameter indices |
| `sigma_2` | Second/first infall mass ratio |
| `t_1`, `t_2` | Infall onset times (Gyr) |
| `infall_1`, `infall_2` | Infall timescales (Gyr) |
| `sfe`, `delta_sfe` | Star formation efficiency parameters |
| `imf_upper` | IMF upper mass limit (M☉) |
| `mgal` | Initial gas mass (M☉) |
| `nb` | SNe Ia per solar mass |
| `loss` | Total loss value (lower is better) |

## Analysis Tools

### Posterior Analysis

```bash
# Interactive folder selection
python posterior_analysis_code/posterior_analysis_plots.py

# Direct analysis
python plotting/posterior_analysis.py \
    --results SMC_DEMC/simulation_results.csv \
    --pcard bulge_pcard.txt \
    --nsamples 5000
```

### Combine Multiple Runs

```bash
python posterior_analysis_code/combine_posterior.py
```

Produces combined `simulation_results.csv` and `walker_history.npz` in `bc_combined_MDF/`.

### Corner Plots

```bash
python standalone_plots/plot_corner.py SMC_DEMC/simulation_results.csv
```

### Physics Reconstruction

```bash
python posterior_plotting_package/test/phys_plot_posterior.py
```

Reconstructs infall rates, SFR, gas mass, and metallicity evolution with uncertainty bands.

## API Reference

### `GalacticEvolutionGA`

Main GA controller class in `Gal_GA.py` / `Gal_GA_PP.py`.

```python
from Gal_GA import GalacticEvolutionGA

ga = GalacticEvolutionGA(
    output_path='results/',
    feh=observed_feh,
    normalized_count=observed_mdf,
    # ... parameter bounds from pcard
)

population, toolbox = ga.init_GenAl(population_size=96)
ga.GenAl(
    population_size=96,
    num_generations=100,
    population=population,
    toolbox=toolbox
)
```

#### Key Methods

| Method | Description |
|--------|-------------|
| `init_GenAl(population_size)` | Initialize population and DEAP toolbox |
| `GenAl(...)` | Run GA optimization |
| `run_smc_demc_stage(...)` | Execute SMC-DEMC refinement |
| `export_ga_samples()` | Save GA history to CSV |

### `run_smc_demc`

SMC-DEMC sampler in `smc_demc.py`.

```python
from smc_demc import run_smc_demc

ensemble, chains_df = run_smc_demc(
    X0,                    # Initial ensemble (N × d)
    loss_fn,               # Loss function: theta → float
    bounds,                # Parameter bounds
    ess_trigger=0.60,      # Resample threshold
    moves_per_stage=3,     # DE-MH steps per stage
)
```

#### Returns

| Key | Description |
|-----|-------------|
| `ensemble` | Final particle positions (N × d array) |
| `chains_df` | DataFrame with stage/particle/accepted/params |

### `UncertaintyAnalysis`

Posterior analysis class in `posterior_analysis_code/uncertainty_analysis.py`.

```python
from uncertainty_analysis import UncertaintyAnalysis

ua = UncertaintyAnalysis('results/simulation_results.csv')
ua.bulge_pcard_path = 'bulge_pcard.txt'

# Generate corner plot
ua.plot_posterior_corner(percentile=10, weight_power=1.0)

# Get weighted statistics
top_df, weights = ua._select_top_and_weights(percentile=10)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow NumPy docstring format for all functions
4. Add tests for new functionality
5. Submit a pull request

## Citation

If you use this code, please cite:

> Miller & Joyce et al. (2024), *Galactic Bulge Chemical Evolution via Hybrid Genetic Algorithm and Sequential Monte Carlo*, in preparation.

## License

[License information here]
