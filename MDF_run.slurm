#!/bin/bash

#SBATCH --account=phys4840
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --output=/home/nmille39/MDF_GCE_GA/slurm_output.log
#SBATCH --error=/home/nmille39/MDF_GCE_GA/slurm_error.log

# Load modules if needed (optional)
# module load python/3.x.x  # Uncomment if using a module system

# Activate virtual environment
source ~/python_projects/venv/bin/activate

# Change to project directory
cd /home/nmille39/MDF_GCE_GA

# Run the Python script and log output
python MDF_GA.py | tee MDF_GA_runtime.log
