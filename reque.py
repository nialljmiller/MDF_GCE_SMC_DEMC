#!/usr/bin/env python3

import os
import subprocess
import time

# Root project dir
PROJECT_DIR = "/project/galacticbulge/MDF_GCE_SMC_DEMC"

# The DEMC run dirs you want to keep alive
RUN_DIRS = [f"bc_medbow_Z{i}_MDF" for i in range(8)]  # Z0..Z7

# Check interval in seconds
CHECK_INTERVAL = 60


def get_active_job_names():
    user = os.environ["USER"]
    cmd = ["squeue", "-u", user, "-h", "-o", "%j"]
    out = subprocess.check_output(cmd, text=True)
    lines = out.splitlines()
    return [line.strip() for line in lines if line.strip()]


def main():
    os.chdir(PROJECT_DIR)

    while True:
        active_names = get_active_job_names()
        print("Active job names:", active_names)

        for run_dir in RUN_DIRS:
            # Look for this run_dir substring in any job name, e.g.
            # "GA_SMC_DEMC-bc_medbow_Z3_MDF"
            has_job = False
            for name in active_names:
                if run_dir in name:
                    has_job = True
                    break

            if not has_job:
                print(f"[RESUBMIT] No active job for {run_dir}, submitting new job.")
                subprocess.call(["python", "submit_demc.py", run_dir + "/"])

        print(f"Sleeping {CHECK_INTERVAL} seconds")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()

