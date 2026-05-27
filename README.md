# MDF_GCE_SMC_DEMC

**Metallicity Distribution Function fitting via Genetic Algorithm with SMC-DEMC refinement for Galactic Chemical Evolution modeling.**

This package implements a hybrid optimization approach combining:
- **Genetic Algorithm (GA)** for global parameter space exploration
- **Differential Evolution Markov Chain Monte Carlo (DE-MC)** moves during GA generations
- **Sequential Monte Carlo with DE-MC (SMC-DEMC)** for posterior refinement

## Installation

```bash
# Clone the repository
git clone https://github.com/user/MDF_GCE_SMC_DEMC.git
cd MDF_GCE_SMC_DEMC

# Install in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Quick Start

### Running the GA

```bash
# Run with default configuration
python scripts/run_ga.py

# Run with custom parameter card
python scripts/run_ga.py path/to/bulge_pcard.txt

# Generate plots from existing results (no optimization)
python scripts/run_ga.py SMC_DEMC/ --plot-only
```

### Python API

```python
from mdf_gce.config import load_config
from mdf_gce.analysis import UncertaintyAnalysis

# Load configuration
config = load_config('bulge_pcard.txt')

# Analyze results
ua = UncertaintyAnalysis('SMC_DEMC/simulation_results.csv')
report = ua.generate_report()
ua.plot_corner(percentile=10)
ua.plot_marginals()
```

## Package Structure

```
MDF_GCE_SMC_DEMC/
├── mdf_gce/                      # Main package
│   ├── __init__.py
│   ├── config.py                 # Configuration parsing
│   ├── constants.py              # Parameter names, labels, defaults
│   ├── utils.py                  # File I/O, array utilities
│   │
│   ├── core/                     # Core optimization
│   │   ├── __init__.py
│   │   ├── smc_demc.py           # SMC-DEMC sampler
│   │   ├── loss.py               # Loss functions (MDF, AMR, combined)
│   │   ├── constraints.py        # Physical constraint penalties
│   │   └── exploration.py        # Voronoi sparse region exploration
│   │
│   ├── analysis/                 # Post-run analysis
│   │   ├── __init__.py
│   │   ├── uncertainty.py        # UncertaintyAnalysis class
│   │   └── posterior.py          # Weights, resampling, quantiles
│   │
│   └── plotting/                 # Visualization
│       ├── __init__.py
│       └── style.py              # Matplotlib configuration
│
├── scripts/                      # CLI entry points
│   └── run_ga.py                 # Main GA runner
│
├── docs/                         # Documentation
│   ├── PARAMETERS.md             # Parameter reference
│   ├── OUTPUT_FORMAT.md          # Output file specifications
│   └── ANALYSIS_GUIDE.md         # Analysis workflow guide
│
├── data/                         # Observational data
├── slurm/                        # Batch job scripts
├── tests/                        # Unit tests
│
├── pyproject.toml                # Package metadata
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Configuration

The GA is configured via a parameter card file (`bulge_pcard.txt`). Key sections:

### Parameter Ranges (Continuous)

| Parameter | Description | Units | Typical Range |
|-----------|-------------|-------|---------------|
| `sigma_2_list` | Second/first infall mass ratio | - | [0.1, 5.0] |
| `tmax_1_list` | First infall onset time | Gyr | [0.005, 0.1] |
| `tmax_2_list` | Second infall onset time | Gyr | [0.1, 10.0] |
| `infall_timescale_1_list` | First infall e-folding time | Gyr | [0.001, 0.1] |
| `infall_timescale_2_list` | Second infall e-folding time | Gyr | [0.1, 10.0] |
| `sfe_array` | Star formation efficiency | Gyr⁻¹ | [1.0, 20.0] |
| `delta_sfe_array` | Change in SFE at second infall | - | [0.01, 0.85] |
| `imf_upper_limits` | IMF upper mass cutoff | M☉ | [60, 130] |
| `mgal_values` | Initial gas mass | M☉ | [1e9, 1e11] |
| `nb_array` | SNe Ia per solar mass | M☉⁻¹ | [5e-4, 1.5e-3] |

### GA Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `popsize` | Population size | 96 |
| `generations` | Number of generations | 256 |
| `crossover_probability` | Crossover rate | 0.5 |
| `mutation_probability` | Mutation rate | 0.5 |
| `demc_fraction` | Fraction for DE-MC moves | 0.69 |
| `output_interval` | Generations between saves | 16 |

## Output Files

Results are written to the `output_path` directory (default: `SMC_DEMC/`):

| File | Description |
|------|-------------|
| `simulation_results.csv` | All models sorted by loss |
| `walker_history.npz` | Full walker trajectories + MDF/alpha/age data |
| `ga_population_samples.csv` | Every evaluated individual |
| `chains.csv` | SMC-DEMC stage transitions |
| `smc_demc_samples.csv` | Final posterior samples |
| `posterior_samples.csv` | Legacy alias for posterior |
| `bulge_pcard.txt` | Copy of input configuration |

## Analysis

### Uncertainty Quantification

```python
from mdf_gce.analysis import UncertaintyAnalysis

ua = UncertaintyAnalysis('SMC_DEMC/simulation_results.csv')

# Get summary statistics
stats = ua.get_summary_statistics(percentile=10)
for param, s in stats.items():
    print(f"{param}: {s['median']:.3f} +{s['q84']-s['median']:.3f} -{s['median']-s['q16']:.3f}")

# Generate plots
ua.plot_corner(params=['sigma_2', 't_2', 'infall_2', 'sfe'])
ua.plot_marginals()
```

### Posterior Weighting

```python
from mdf_gce.analysis.posterior import compute_weights, weighted_quantile

# Compute importance weights from loss values
loss = df['fitness'].values
weights, temperature, ess = compute_weights(loss)

print(f"Temperature: {temperature:.3f}")
print(f"Effective Sample Size: {ess:.1f}")

# Compute weighted quantiles
q16, q50, q84 = weighted_quantile(df['sigma_2'].values, [0.16, 0.5, 0.84], weights)
```

## Algorithm Details

### Hybrid GA + DE-MC

During each GA generation:
1. Evaluate fitness of new individuals
2. Select elite individuals (top ~6%)
3. Apply crossover and mutation to create offspring
4. **Apply DE-MC moves** to a fraction of the population
5. Replace population with elites + offspring

The DE-MC moves use the ter Braak (2006) scheme:
```
θ_proposed = θ_current + γ × (θ_r1 - θ_r2) + ε
```
where γ = 2.38/√(2d) and ε is small Gaussian noise.

### SMC-DEMC Refinement

After GA convergence, the final population is refined via tempered SMC:
1. Start at β=0 (prior)
2. Adaptively increase β toward 1 (posterior)
3. At each temperature stage:
   - Resample if ESS drops below threshold
   - Apply DE-MC moves with tempered likelihood
4. Final ensemble represents posterior samples

### Loss Function

The default ensemble loss combines three terms:
```
L_ensemble = 0.7 × L_WRMSE + 0.2 × (1 - cos_sim) + 0.1 × L_Huber
```

- **WRMSE**: Weighted root mean square error (shape sensitivity)
- **Cosine similarity**: Pattern matching independent of normalization
- **Huber loss**: Robust to outlier bins

## References

- ter Braak, C.J.F. (2006). A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution. *Statistics and Computing*, 16, 239-249.
- Goldberg, D.E. (1989). *Genetic Algorithms in Search, Optimization and Machine Learning*. Addison-Wesley.
- Del Moral, P., Doucet, A., & Jasra, A. (2006). Sequential Monte Carlo samplers. *JRSS B*, 68(3), 411-436.

## License

MIT License - see LICENSE file for details.

## Authors

N. Miller

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Self-contained code archive

This repository is intended as a self-contained archival version of the code used for the accepted ApJ manuscript.

The main analysis and optimization framework developed for this work is contained in `mdf_gce/`. The repository also includes the OMEGA+/NuPyCEE/JINAPyCEE code and yield-table files required to run the Galactic Chemical Evolution calculations without requiring users to reconstruct the working environment from external repositories.

The included NuPyCEE/JINAPyCEE components are vendored here for reproducibility of the paper calculations. Their original licences and acknowledgements are preserved in the relevant directories.
