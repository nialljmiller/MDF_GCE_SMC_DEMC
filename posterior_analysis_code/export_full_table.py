#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd

def build_full_table(run_dir: str):
    run_dir = os.path.abspath(run_dir)

    ga_path = os.path.join(run_dir, "ga_population_samples.csv")
    npz_path = os.path.join(run_dir, "walker_history.npz")

    if not os.path.exists(ga_path):
        raise FileNotFoundError(f"{ga_path} not found")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} not found")

    df = pd.read_csv(ga_path)
    if "evaluation" not in df.columns:
        raise RuntimeError("ga_population_samples.csv has no 'evaluation' column")

    data = np.load(npz_path, allow_pickle=True)
    mdf_data = data["mdf_data"]
    alpha_data = data["alpha_data"]
    age_data = data.get("age_data", [])

    eval_idx = df["evaluation"].astype(int).to_numpy()

    df["mdf_x"] = [mdf_data[i][0] for i in eval_idx]
    df["mdf_y"] = [mdf_data[i][1] for i in eval_idx]
    df["alpha_tracks"] = [alpha_data[i] for i in eval_idx]

    if len(age_data) > 0:
        df["age_x"] = [age_data[i][0] for i in eval_idx]
        df["age_y"] = [age_data[i][1] for i in eval_idx]

    out_pkl = os.path.join(run_dir, "full_evaluation_table.pkl")
    out_csv = os.path.join(run_dir, "full_evaluation_table.csv")
    df.to_pickle(out_pkl)
    df.to_csv(out_csv, index=False)

    print(f"[full-table] wrote {out_pkl} and {out_csv} (rows={len(df)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="bc_*_MDF directory")
    args = ap.parse_args()
    build_full_table(args.run_dir)
