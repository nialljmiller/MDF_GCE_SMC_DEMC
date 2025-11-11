#!/usr/bin/env python3
"""
Tool to combine multiple GA/DEMC result catalogues into a single consolidated
``simulation_results.csv`` as if all evaluations happened in one big run.

This script searches for folders containing any ``simulation_results*.csv`` (similar
to the uncertainty analysis script), lets the user select a subset interactively,
concatenates their DataFrames (deduplicating by parameter tuples if needed),
and writes the combined catalogue to ``bc_combined_MDF/simulation_results.csv``.

Optionally, if ``walker_history.npz`` files are present in the selected folders,
it attempts to merge them too (appending histories with walker ID offsets to
avoid collisions).

For ongoing runs, it automatically selects the latest ``simulation_results_gen_XX.csv``
if the final ``simulation_results.csv`` is not yet available.

Usage:
    python combine_results.py [root_dir]

If ``root_dir`` is omitted, it defaults to the current directory.
"""

import argparse
import os
import sys
#os.chdir("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import pickle

def _build_full_table_from_checkpoint(run_name, folder_path):
    """
    Rebuild a 'full_evaluation_table' for a single run from ga_checkpoint.pkl.

    Columns:
      - everything in sample_records (params, metrics, loss, generation, evaluation)
      - MDF_x, MDF_y
      - age_x, age_y
      - alpha_tracks (object: list of [x,y] pairs)
      - label, model_number
      - run_name
    """
    ckpt_path = os.path.join(folder_path, "ga_checkpoint.pkl")
    if not os.path.exists(ckpt_path):
        print(f"No ga_checkpoint.pkl in {run_name}; skipping full-table reconstruction")
        return None

    with open(ckpt_path, "rb") as f:
        payload = pickle.load(f)

    # In your code, ga_state is saved as ga_state.__dict__, i.e. already a dict
    ga_state = payload.get("ga_state", None)
    if ga_state is None:
        print(f"Checkpoint in {run_name} has no 'ga_state'; skipping")
        return None
    if not isinstance(ga_state, dict):
        # ultra-defensive, but just in case you ever change checkpoint format
        ga_dict = getattr(ga_state, "__dict__", None)
        if ga_dict is None:
            print(f"Checkpoint ga_state in {run_name} is not a dict and has no __dict__; skipping")
            return None
        ga_state = ga_dict

    # --- pull stuff out of the state dict ---
    sample_records = ga_state.get("sample_records", None)
    if not sample_records:
        print(f"[full-table] sample_records is empty for {run_name}; nothing to export")
        return None

    mdf_data   = ga_state.get("mdf_data",   [])
    alpha_data = ga_state.get("alpha_data", [])
    age_data   = ga_state.get("age_data",   [])
    labels     = ga_state.get("labels",     [])
    model_nums = ga_state.get("model_numbers", [])

    df = pd.DataFrame(sample_records)

    if "evaluation" not in df.columns:
        print(f"[full-table] no 'evaluation' column in sample_records for {run_name}; cannot link tracks")
        return None

    eval_idx = df["evaluation"].to_numpy(dtype=np.int64)
    max_eval = int(eval_idx.max())

    # Sanity check (not fatal, just noisy if something got truncated)
    if len(mdf_data) <= max_eval or len(alpha_data) <= max_eval or len(age_data) <= max_eval:
        print(f"[full-table] WARNING: track arrays shorter than max evaluation index in {run_name}")
        print(f"  len(mdf_data)={len(mdf_data)}, len(alpha_data)={len(alpha_data)}, len(age_data)={len(age_data)}, max_eval={max_eval}")

    def safe_get(arr, idx, default=None):
        return arr[idx] if 0 <= idx < len(arr) else default

    mdf_x_list, mdf_y_list   = [], []
    age_x_list, age_y_list   = [], []
    alpha_list               = []
    label_list               = []
    modelnum_list            = []

    for e in eval_idx:
        ei = int(e)

        # MDF
        mdf_entry = safe_get(mdf_data, ei, [None, None])
        if mdf_entry is None:
            mdf_entry = [None, None]
        mdf_x, mdf_y = mdf_entry
        mdf_x_list.append(np.asarray(mdf_x) if mdf_x is not None else None)
        mdf_y_list.append(np.asarray(mdf_y) if mdf_y is not None else None)

        # Age–[Fe/H]
        age_entry = safe_get(age_data, ei, [None, None])
        if age_entry is None:
            age_entry = [None, None]
        age_x, age_y = age_entry
        age_x_list.append(np.asarray(age_x) if age_x is not None else None)
        age_y_list.append(np.asarray(age_y) if age_y is not None else None)

        # Alpha tracks
        alpha_entry = safe_get(alpha_data, ei, None)
        if alpha_entry is not None:
            tracks = []
            for pair in alpha_entry:
                if pair is None or len(pair) != 2:
                    tracks.append([None, None])
                else:
                    ax, ay = pair
                    tracks.append([
                        np.asarray(ax) if ax is not None else None,
                        np.asarray(ay) if ay is not None else None,
                    ])
            alpha_list.append(tracks)
        else:
            alpha_list.append(None)

        # Labels, model numbers (optional)
        label_list.append(safe_get(labels, ei, None))
        modelnum_list.append(safe_get(model_nums, ei, None))

    df["MDF_x"]        = mdf_x_list
    df["MDF_y"]        = mdf_y_list
    df["age_x"]        = age_x_list
    df["age_y"]        = age_y_list
    df["alpha_tracks"] = alpha_list
    df["label"]        = label_list
    df["model_number"] = modelnum_list
    df["run_name"]     = run_name

    return df

def combine_full_tables_from_checkpoints(chosen, output_dir):
    """
    Use ga_checkpoint.pkl in each selected folder to build one big full table.
    Writes:
      - combined_full_evaluation_table.pkl
      - combined_full_evaluation_table.csv
    """
    frames = []
    for name, path in chosen:   # <-- FIXED: only (name, path)
        print(f"Rebuilding full table from checkpoint in {name} ...")
        df = _build_full_table_from_checkpoint(name, path)
        if df is None or df.empty:
            print(f"  -> no usable records in {name}")
            continue
        frames.append(df)

    if not frames:
        print("No full evaluation tables reconstructed from checkpoints; nothing to combine.")
        return

    combined = pd.concat(frames, axis=0, ignore_index=True)

    # If 'loss' exists, sort by it for convenience
    if "loss" in combined.columns:
        combined = combined.sort_values("loss", ascending=True, kind="mergesort").reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    pkl_out  = os.path.join(output_dir, "combined_full_evaluation_table.pkl")
    csv_out  = os.path.join(output_dir, "combined_full_evaluation_table.csv")

    combined.to_pickle(pkl_out)
    combined.to_csv(csv_out, index=False)

    print(f"Combined full evaluation table written to:\n  {pkl_out}\n  {csv_out}")



def find_result_folders(root_dir: str = '.') -> List[Tuple[str, str]]:
    """
    Recursively find folders containing any 'simulation_results*.csv'.
    Returns list of (folder_name, full_path) tuples.
    """
    candidates = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(f.startswith('simulation_results') and f.endswith('.csv') for f in filenames):
            rel_name = os.path.relpath(dirpath, root_dir)
            if rel_name == '.':
                rel_name = os.path.basename(os.path.abspath(root_dir))
            candidates.append((rel_name, dirpath))
    candidates.sort()
    return candidates

def select_folders(candidates: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Interactively select folders from the list.
    Supports comma-separated indices and ranges (e.g., '1,3-5,7').
    """
    if not candidates:
        print("No folders with simulation_results*.csv found.")
        sys.exit(0)

    print("\nAvailable folders:")
    for i, (name, path) in enumerate(candidates):
        print(f"[{i}] {name} ({path})")

    selection = input("\nEnter comma-separated indices or ranges (e.g., 0,2-4,6): ").strip()
    if not selection:
        print("No selection made. Exiting.")
        sys.exit(0)

    chosen_idx = set()
    for part in selection.split(','):
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                chosen_idx.update(range(start, end + 1))
            except ValueError:
                print(f"Invalid range: {part}")
                continue
        else:
            try:
                chosen_idx.add(int(part))
            except ValueError:
                print(f"Invalid index: {part}")
                continue

    chosen = [candidates[i] for i in sorted(chosen_idx) if 0 <= i < len(candidates)]
    if not chosen:
        print("No valid selections. Exiting.")
        sys.exit(0)

    print("\nSelected folders:")
    for name, path in chosen:
        print(f"- {name} ({path})")

    return chosen


def get_latest_csv(path: str) -> Optional[str]:
    """
    Find the latest results CSV in the folder:
    - Prefer 'simulation_results.csv' if exists
    - Else, find the highest 'simulation_results_gen_XX.csv'
    Returns full path or None if no matching files.
    """
    files = [f for f in os.listdir(path) if f.startswith('simulation_results') and f.endswith('.csv')]
    if not files:
        return None

    if 'simulation_results.csv' in files:
        return os.path.join(path, 'simulation_results.csv')

    # Parse gen numbers
    gens = []
    for f in files:
        if '_gen_' in f:
            try:
                gen_str = f.split('_gen_')[1].split('.csv')[0]
                gen = int(gen_str)
                gens.append((gen, f))
            except ValueError:
                pass

    if not gens:
        return None

    max_gen, max_f = max(gens)
    return os.path.join(path, max_f)

def combine_csvs(selected: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Load and concatenate the latest results CSV from each selected folder.
    Deduplicate by parameter columns if duplicates exist.
    """
    dfs = []
    param_cols = [
        'comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx',
        'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2',
        'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb'
    ]  # Adjust if your CSVs have different params

    for name, path in selected:
        csv_path = get_latest_csv(path)
        if not csv_path:
            print(f"No results CSV found in {name}")
            continue
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} rows from {name} using {os.path.basename(csv_path)}")
            dfs.append(df)
        except Exception as exc:
            print(f"Failed to load {csv_path}: {exc}")
            continue

    if not dfs:
        raise ValueError("No valid CSVs loaded")

    combined = pd.concat(dfs, ignore_index=True)

    # Deduplicate based on parameters (keep first occurrence)
    #dup_mask = combined.duplicated(subset=param_cols, keep='first')
    #if dup_mask.any():
    #    print(f"Removing {dup_mask.sum()} duplicate parameter sets")
    #    combined = combined[~dup_mask]

    # Sort by fitness/loss if present (assuming lower is better)
    if 'fitness' in combined.columns:
        combined.sort_values('fitness', ascending=True, inplace=True)
    elif 'loss' in combined.columns:
        combined.sort_values('loss', ascending=True, inplace=True)

    combined.reset_index(drop=True, inplace=True)
    print(f"Combined total: {len(combined)} unique models")

    return combined



def combine_histories(selected: List[Tuple[str, str]]) -> tuple[dict, dict] | None:
    """
    Attempt to combine walker_history.npz files if present.

    Returns:
        (all_hist, run_offsets) or None

        all_hist: dict with 'histories', 'walker_ids', and optional 'mdf_data', etc.
        run_offsets: mapping {run_name: offset_in_combined_mdf_data}

    Walker IDs are offset to avoid collisions. mdf/alpha/age are simply concatenated.
    """
    all_hist = {
        'histories': [],
        'walker_ids': [],
        'mdf_data': [],
        'alpha_data': [],
        'age_data': [],
    }
    id_offset = 0
    run_offsets: dict[str, int] = {}

    for name, path in selected:
        npz_path = os.path.join(path, 'walker_history.npz')
        if not os.path.exists(npz_path):
            print(f"No walker_history.npz in {name}; skipping")
            continue

        try:
            data = np.load(npz_path, allow_pickle=True)
            print(f"Loaded history from {name}: {len(data['walker_ids'])} walkers")

            # --- histories + walker IDs ---
            all_hist['histories'].extend(list(data['histories']))

            walker_ids = data['walker_ids'] + id_offset
            all_hist['walker_ids'].extend(walker_ids)
            id_offset += len(walker_ids) + 1  # +1 for safety

            # --- tracks: record offset BEFORE appending ---
            # We only define an offset if mdf_data exists for this run.
            if 'mdf_data' in data.files:
                offset = len(all_hist['mdf_data'])
                run_offsets[name] = offset

                all_hist['mdf_data'].extend(list(data['mdf_data']))

                if 'alpha_data' in data.files:
                    all_hist['alpha_data'].extend(list(data['alpha_data']))
                if 'age_data' in data.files:
                    all_hist['age_data'].extend(list(data['age_data']))
            else:
                print(f"Warning: {name} has walker_history.npz but no mdf_data; "
                      f"cannot map tracks for this run.")

        except Exception as exc:
            print(f"Failed to load {npz_path}: {exc}")
            continue

    if not all_hist['histories']:
        print("No histories combined")
        return None

    print(f"Combined histories: {len(all_hist['histories'])} walkers total")
    print(f"Combined MDF entries: {len(all_hist['mdf_data'])}")
    return all_hist, run_offsets


def combine_ga_samples(
    selected: List[Tuple[str, str]],
    run_offsets: dict[str, int],
) -> pd.DataFrame | None:
    """
    Combine per-run ga_population_samples.csv into a single catalogue and
    attach a global track_index that points into the combined mdf/alpha/age arrays.

    For each row:
        global_track_index = run_offsets[run_name] + evaluation

    Only runs that have both walker_history.mdf_data and ga_population_samples.csv
    will get a defined track_index.
    """
    dfs = []

    for name, path in selected:
        samples_path = os.path.join(path, "ga_population_samples.csv")
        if not os.path.exists(samples_path):
            print(f"No ga_population_samples.csv in {name}; skipping for GA samples")
            continue

        if name not in run_offsets:
            # We have GA samples but no recorded tracks -> we cannot map safely
            print(f"Warning: have GA samples in {name} but no run_offset "
                  f"(no mdf_data when combining histories). Skipping track_index.")
            continue

        try:
            df = pd.read_csv(samples_path)
        except Exception as exc:
            print(f"Failed to load {samples_path}: {exc}")
            continue

        if "evaluation" not in df.columns:
            print(f"Warning: ga_population_samples.csv in {name} has no 'evaluation' "
                  f"column; cannot define track_index. Skipping.")
            continue

        offset = run_offsets[name]
        df = df.copy()
        df["run_name"] = name
        df["track_index"] = offset + df["evaluation"].astype(int)

        print(f"Loaded {len(df)} GA-sample rows from {name}; "
              f"offset={offset} → track_index in "
              f"[{df['track_index'].min()}, {df['track_index'].max()}]")
        dfs.append(df)

    if not dfs:
        print("No GA sample tables combined")
        return None

    combined = pd.concat(dfs, ignore_index=True)

    # Optional: sort by loss if present
    if "loss" in combined.columns:
        combined.sort_values("loss", ascending=True, inplace=True)
    elif "fitness" in combined.columns:
        combined.sort_values("fitness", ascending=True, inplace=True)

    combined.reset_index(drop=True, inplace=True)
    print(f"Combined GA samples: {len(combined)} rows total")
    return combined

def main():
    parser = argparse.ArgumentParser(description="Combine multiple GA/DEMC result folders")
    parser.add_argument("root_dir", nargs='?', default='.', help="Root directory to search (default: current)")
    args = parser.parse_args()

    candidates = find_result_folders(args.root_dir)
    selected = select_folders(candidates)

    output_dir = os.path.abspath('bc_combined_MDF')
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 1) Combine the legacy simulation_results*.csv ----------
    combined_df = combine_csvs(selected)
    csv_out = os.path.join(output_dir, 'simulation_results.csv')
    combined_df.to_csv(csv_out, index=False)
    print(f"\nCombined results written to: {csv_out}")

    # ---------- 2) Combine walker histories + track arrays ----------
    hist_result = combine_histories(selected)
    run_offsets: dict[str, int] = {}
    if hist_result is not None:
        histories, run_offsets = hist_result

        npz_out = os.path.join(output_dir, 'walker_history.npz')
        np.savez_compressed(
            npz_out,
            walker_ids=np.array(histories['walker_ids']),
            histories=np.array(histories['histories'], dtype=object),
            mdf_data=np.array(histories['mdf_data'], dtype=object),
            alpha_data=np.array(histories['alpha_data'], dtype=object),
            age_data=np.array(histories['age_data'], dtype=object),
        )
        print(f"Combined walker history written to: {npz_out}")
    else:
        print("Skipping combined walker_history.npz (no histories)")

    # ---------- 3) Combine GA samples with global track indices ----------
    if run_offsets:
        ga_combined = combine_ga_samples(selected, run_offsets)
        if ga_combined is not None:
            ga_out = os.path.join(output_dir, "ga_population_samples_combined.csv")
            ga_combined.to_csv(ga_out, index=False)
            print(f"Combined GA samples written to: {ga_out}")
    else:
        print("No run_offsets defined; cannot build GA+track mapping.")

    # ---------- 4) Build giant fully-linked table from ga_checkpoint.pkl ----------
    combine_full_tables_from_checkpoints(selected, output_dir)

    print("\nDone! You can now run analysis/plotting on the combined output.")

if __name__ == "__main__":
    main()