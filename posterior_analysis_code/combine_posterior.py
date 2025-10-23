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
    dup_mask = combined.duplicated(subset=param_cols, keep='first')
    if dup_mask.any():
        print(f"Removing {dup_mask.sum()} duplicate parameter sets")
        combined = combined[~dup_mask]

    # Sort by fitness/loss if present (assuming lower is better)
    if 'fitness' in combined.columns:
        combined.sort_values('fitness', ascending=True, inplace=True)
    elif 'loss' in combined.columns:
        combined.sort_values('loss', ascending=True, inplace=True)

    combined.reset_index(drop=True, inplace=True)
    print(f"Combined total: {len(combined)} unique models")

    return combined

def combine_histories(selected: List[Tuple[str, str]]) -> dict | None:
    """
    Attempt to combine walker_history.npz files if present.
    Returns dict with 'histories', 'walker_ids', and optional 'mdf_data', etc.
    Walker IDs are offset to avoid collisions.
    """
    all_hist = {'histories': [], 'walker_ids': [], 'mdf_data': [], 'alpha_data': [], 'age_data': []}
    id_offset = 0

    for name, path in selected:
        npz_path = os.path.join(path, 'walker_history.npz')
        if not os.path.exists(npz_path):
            print(f"No walker_history.npz in {name}; skipping")
            continue

        try:
            data = np.load(npz_path, allow_pickle=True)
            print(f"Loaded history from {name}: {len(data['walker_ids'])} walkers")

            # Append histories
            all_hist['histories'].extend([h for h in data['histories']])

            # Offset and append walker IDs
            walker_ids = data['walker_ids'] + id_offset
            all_hist['walker_ids'].extend(walker_ids)
            id_offset += len(walker_ids) + 1  # +1 for safety

            # Append optional arrays if present
            for key in ['mdf_data', 'alpha_data', 'age_data']:
                if key in data.files:
                    all_hist[key].extend([item for item in data[key]])

        except Exception as exc:
            print(f"Failed to load {npz_path}: {exc}")
            continue

    if not all_hist['histories']:
        print("No histories combined")
        return None

    print(f"Combined histories: {len(all_hist['histories'])} walkers total")
    return all_hist

def main():
    parser = argparse.ArgumentParser(description="Combine multiple GA/DEMC result folders")
    parser.add_argument("root_dir", nargs='?', default='.', help="Root directory to search (default: current)")
    args = parser.parse_args()

    candidates = find_result_folders(args.root_dir)
    selected = select_folders(candidates)

    output_dir = os.path.abspath('bc_combined_MDF')
    os.makedirs(output_dir, exist_ok=True)

    # Combine CSVs
    combined_df = combine_csvs(selected)
    csv_out = os.path.join(output_dir, 'simulation_results.csv')
    combined_df.to_csv(csv_out, index=False)
    print(f"\nCombined results written to: {csv_out}")

    # Combine histories if possible
    histories = combine_histories(selected)
    if histories:
        npz_out = os.path.join(output_dir, 'walker_history.npz')
        np.savez_compressed(
            npz_out,
            walker_ids=np.array(histories['walker_ids']),
            histories=np.array(histories['histories'], dtype=object),
            mdf_data=np.array(histories['mdf_data'], dtype=object),
            alpha_data=np.array(histories['alpha_data'], dtype=object),
            age_data=np.array(histories['age_data'], dtype=object)
        )
        print(f"Combined walker history written to: {npz_out}")

    print("\nDone! You can now run analysis/plotting on the combined output.")

if __name__ == "__main__":
    main()