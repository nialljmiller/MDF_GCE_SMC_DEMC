#!/usr/bin/env python3
"""
ga_plotting.py — compute-free plotting runner for any results folder
Usage:
  python ga_plotting.py --results bc_combined_MDF/simulation_results.csv [--bins 60]
"""
import os
import argparse
import numpy as np
from types import SimpleNamespace
import mdf_plotting


def _build_obs_mdf(bins=60):
    try:
        f = open("data/Bensby_Data.tsv")
    except FileNotFoundError:
        f = open("../data/Bensby_Data.tsv")
    lines = f.readlines(); f.close()
    hdr = lines[0].split()
    feh_idx = hdr.index("[Fe/H]")
    feh_vals = np.array([float(l.split()[feh_idx]) for l in lines[1:]], float)
    lo = np.nanmin(feh_vals) - 0.05
    hi = np.nanmax(feh_vals) + 0.05
    edges = np.linspace(lo, hi, bins + 1)
    counts, edges = np.histogram(feh_vals, bins=edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    norm = counts.max() if counts.max() > 0 else 1.0
    return centers, counts / norm

def main(results_csv):

    if not os.path.isfile(results_csv):
        raise SystemExit(f"Missing results file: {results_csv}")

    output_dir = os.path.dirname(results_csv)
    os.makedirs(output_dir, exist_ok=True)

    feh, normalized_count = _build_obs_mdf(bins=args.bins)
    GalGA = SimpleNamespace(output_path=output_dir)

    mdf_plotting.generate_all_plots(GalGA, feh, normalized_count, results_file=results_csv)
    print(f"[plot] wrote plots to: {output_dir}")

