# MDF_GCE_SMC_DEMC

This repository runs the metallicity distribution function (MDF) galactic chemical evolution pipeline
with a hybrid search: a genetic algorithm drives the global exploration while Differential Evolution
Markov Chain Monte Carlo (DE-MC) moves are injected directly into each generation.  The final ensemble
is then polished with a Sequential Monte Carlo DE-MC (SMC-DEMC) refinement stage.  The legacy GA
utilities have been updated so that all tooling, paths, and batch scripts reference the new
SMC-DEMC-aware workflow.

## How the GA and DE-MC interact

- During the GA loop the population is still bred with crossover and mutation, but after each
  replacement step a configurable fraction of walkers undergoes one or more DE-MC sweeps.  These
  sweeps call the exact same loss function as the GA, so accepted proposals are genuine MCMC moves and
  their evaluations contribute to the stored diagnostics.
- The behaviour is controlled by the `GalacticEvolutionGA` constructor parameters:
  - `demc_hybrid` (default `True`) toggles the intra-GA DE-MC stage.
  - `demc_fraction` sets the fraction of walkers to update per generation.
  - `demc_moves_per_gen`, `demc_gamma`, and `demc_rng_seed` mirror the sampler knobs exposed in
    `smc_demc.py`.
- After the GA finishes, `run_smc_demc_stage` performs the tempered SMC-DEMC refinement that produces
  posterior chains and resampled draws as CSV files alongside the legacy `posterior_samples.csv`
  alias.  The GA run itself now exports `ga_population_samples.csv`, which captures every evaluated
  individual so the evolutionary search can be post-processed like an ensemble sampler.

## Running locally

```bash
python MDF_SMC_DEMC_Launcher.py
```

The launcher executes the full pipeline. A GA exploration is followed automatically by the
SMC-DEMC refinement stage, which consumes the final GA ensemble and produces calibrated posterior
draws. Results, diagnostics, the GA sampling history, and SMC-DEMC chains are written under the
`output_path` specified in the card (defaults to `SMC_DEMC/`) using plain CSV artefacts.

By default the launcher injects ``--plot-mode posterior_minimal`` so that only the MDF fits,
four-panel alpha comparison, required physics plots, and the posterior summary are generated.
Override the flag (e.g. ``python MDF_SMC_DEMC_Launcher.py --plot-mode full``) to restore the
complete GA diagnostic suite when needed.

## Batch execution

Updated SLURM batch scripts are provided in the repository:

- `submit_mdf_96core.sh` and `submit_mdf_bigcore.sh` launch the full pipeline on the shared cluster.
- `smc_demc_sbatch.sh` runs only the posterior sampling stage starting from an existing GA solution.
- `launch_many.sh` sweeps over multiple in-list combinations while using the new launcher.

All scripts now point to `/project/galacticbulge/MDF_GCE_SMC_DEMC` and call `MDF_SMC_DEMC_Launcher.py`.

## Posterior analysis tools

Utilities such as `posterior_analysis.py`, `loss_plot.py`, `analysis_plot.py`, and
`phys_plot.py` read the consolidated posterior outputs produced by the SMC-DEMC stage.  Default
paths and filenames have been updated so they read from `SMC_DEMC/` by default.  The
`GalacticEvolutionGA` class now delegates posterior sampling to `smc_demc.run_smc_demc`, ensuring a
single implementation of the SMC-DEMC algorithm is used across the codebase.

