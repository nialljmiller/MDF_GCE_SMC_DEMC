#!/bin/bash
#SBATCH --job-name=mdf_smc_demc
#SBATCH --output=logs/mdf_smc_demc_%j.out
#SBATCH --error=logs/mdf_smc_demc_%j.err
#SBATCH --account=galacticbulge
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=96:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"

cd /project/galacticbulge/MDF_GCE_SMC_DEMC || exit 1
source ~/python_projects/venv/bin/activate
mkdir -p logs
python -u MDF_SMC_DEMC_Launcher.py

echo "Finished at: $(date)"

