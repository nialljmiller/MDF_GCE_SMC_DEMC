# MDF_GCE_SMC_DEMC

This repository runs the metallicity distribution function (MDF) galactic chemical evolution pipeline
with a genetic algorithm search followed by a Sequential Monte Carlo differential evolution MCMC
(SMC-DEMC) posterior refinement stage.  The legacy GA utilities have been updated so that all tooling,
paths, and batch scripts reference the new SMC-DEMC workflow.

## Running locally

```bash
python MDF_SMC_DEMC_Launcher.py
```

The launcher executes the full pipeline (GA optimisation plus the SMC-DEMC posterior sampler) using
the configuration in `bulge_pcard.txt`.  Results, diagnostics, and posterior chains are written under
the `output_path` specified in that card (defaults to `SMC_DEMC/`).

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
