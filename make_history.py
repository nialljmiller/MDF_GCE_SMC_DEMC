#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

import Gal_GA_PP as Gal_GA  # parse_inlist, GalacticEvolutionGA, etc.

# ---------- helpers (side-effect free) ----------

def get_latest_csv(path: str) -> str | None:
    files = [f for f in os.listdir(path) if f.startswith('simulation_results') and f.endswith('.csv')]
    if not files:
        return None
    if 'simulation_results.csv' in files:
        return os.path.join(path, 'simulation_results.csv')
    gens = []
    for f in files:
        if '_gen_' in f:
            try:
                gen = int(f.split('_gen_')[1].split('.csv')[0])
                gens.append((gen, f))
            except ValueError:
                pass
    if not gens:
        return None
    return os.path.join(path, max(gens)[1])

def load_bensby_data(file_path='data/Bensby_Data.tsv'):
    df = pd.read_csv(file_path, sep='\t')
    print(f"Loaded Bensby data with shape: {df.shape}")
    print(f"Columns available: {list(df.columns)}")
    return df

# --- replace your write_history(...) with this ---

def _as_object_array(seq):
    # Defensive: handle None, ndarray, list, tuple; always return 1-D object array
    if seq is None:
        return np.empty(0, dtype=object)
    if isinstance(seq, np.ndarray) and seq.dtype == object and seq.ndim == 1:
        return seq
    try:
        n = len(seq)
    except TypeError:
        # not iterable -> single item
        out = np.empty(1, dtype=object)
        out[0] = seq
        return out
    out = np.empty(n, dtype=object)
    for i, v in enumerate(seq):
        out[i] = v
    return out

def write_history(output_path, gal, inds=None, df_sel=None, metric_col="fitness"):
    path = os.path.join(output_path, "walker_history.npz")

    walker_ids = np.array(sorted(gal.walker_history.keys()), dtype=np.int32)

    # histories: always a 1-D object array of per-walker sequences (each sequence is a list/array of frames)
    histories_list = []
    for k in walker_ids:
        frames = gal.walker_history[k]
        # ensure each frame is a plain float array
        frames = [np.asarray(f, dtype=float) for f in frames]
        histories_list.append(frames)
    histories = _as_object_array(histories_list)

    if inds is None:
        inds = np.array([gal.walker_history[k][-1] for k in walker_ids], dtype=float)
    else:
        inds = np.asarray(inds, dtype=float)

    losses = None
    if df_sel is not None and metric_col in df_sel.columns:
        losses = np.asarray(df_sel[metric_col].to_numpy(), dtype=float)

    # These three are ragged by design; force 1-D object arrays to avoid broadcasting
    mdf_obj   = _as_object_array(getattr(gal, "mdf_data", []))
    alpha_obj = _as_object_array(getattr(gal, "alpha_data", []))
    age_obj   = _as_object_array(getattr(gal, "age_data", []))

    np.savez_compressed(
        path,
        walker_ids=walker_ids,
        histories=histories,
        mdf_data=mdf_obj,
        alpha_data=alpha_obj,
        age_data=age_obj,
        inds=inds,
        losses=losses,
    )
    print(f"[write] {path}  (walkers={len(walker_ids)}  mdf={len(mdf_obj)}  alpha={len(alpha_obj)}  age={len(age_obj)})")


def _row_to_individual(row):
    disc = [int(row['comp_idx']), int(row['imf_idx']), int(row['sn1a_idx']), int(row['sy_idx']), int(row['sn1ar_idx'])]
    cont = [float(row['sigma_2']), float(row['t_1']), float(row['t_2']), float(row['infall_1']), float(row['infall_2']),
            float(row['sfe']), float(row['delta_sfe']), float(row['imf_upper']), float(row['mgal']), float(row['nb'])]
    return disc + cont

def _as_tuple(ind):
    # exact-value tuple; CSV -> float is deterministic so equality is fine here
    return tuple(float(x) for x in ind)

# --- and harden _load_existing_npz(...) like this ---

def _load_existing_npz(path):
    npz = np.load(path, allow_pickle=True)
    walker_ids = npz["walker_ids"]
    histories  = npz["histories"]

    # tolerate both list-like and object arrays
    def _to_list(x):
        if isinstance(x, np.ndarray) and x.dtype == object:
            return list(x)
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    mdf_data   = _to_list(npz["mdf_data"])   if "mdf_data"   in npz else []
    alpha_data = _to_list(npz["alpha_data"]) if "alpha_data" in npz else []
    age_data   = _to_list(npz["age_data"])   if "age_data"   in npz else []

    return walker_ids, histories, mdf_data, alpha_data, age_data

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entry", help="Either a results CSV or a directory containing simulation_results*.csv")
    ap.add_argument("--percentile", type=float, default=10.0)
    args = ap.parse_args()

    entry = os.path.abspath(args.entry)
    if os.path.isdir(entry):
        results_csv = get_latest_csv(entry)
        if results_csv is None:
            raise FileNotFoundError(f"No simulation_results*.csv found under {entry}")
        outdir = entry
    else:
        results_csv = entry
        outdir = os.path.dirname(results_csv) or "."

    pcard = os.path.join(outdir, "bulge_pcard.txt")

    # Build GalGA in plot-only mode from pcard + data
    params = Gal_GA.parse_inlist(pcard)
    feh, count = np.loadtxt(params["obs_file"], usecols=(0, 1), unpack=True)
    normalized_count = count / max(count.max(), 1.0)
    obs_age_data = load_bensby_data('data/Bensby_Data.tsv')

    GalGA = Gal_GA.GalacticEvolutionGA(
        output_path=outdir,
        iniab_header=params["iniab_header"],
        sn1a_header=params["sn1a_header"],
        sigma_2_list=params["sigma_2_list"],
        tmax_1_list=params["tmax_1_list"],
        tmax_2_list=params["tmax_2_list"],
        infall_timescale_1_list=params["infall_timescale_1_list"],
        infall_timescale_2_list=params["infall_timescale_2_list"],
        comp_array=params["comp_array"],
        imf_array=params["imf_array"],
        sfe_array=params["sfe_array"],
        delta_sfe_array=params["delta_sfe_array"],
        imf_upper_limits=params["imf_upper_limits"],
        sn1a_assumptions=params["sn1a_assumptions"],
        stellar_yield_assumptions=params["stellar_yield_assumptions"],
        mgal_values=params["mgal_values"],
        nb_array=params["nb_array"],
        sn1a_rates=params["sn1a_rates"],
        timesteps=params["timesteps"],
        A1=params["A1"], A2=params["A2"],
        feh=feh, normalized_count=normalized_count, obs_age_data=obs_age_data,
        loss_metric=params["loss_metric"],
        obs_age_data_loss_metric=params["obs_age_data_loss_metric"],
        obs_age_data_target=params["obs_age_data_target"],
        mdf_vs_age_weight=params["mdf_vs_age_weight"],
        fancy_mutation=params["fancy_mutation"],
        shrink_range=params["shrink_range"],
        gaussian_sigma_scale=params.get("gaussian_sigma_scale", 0.01),
        crossover_noise_fraction=params.get("crossover_noise_fraction", 0.05),
        perturbation_strength=params.get("perturbation_strength", 0.1),
        tournament_size=params["tournament_size"],
        threshold=params["selection_threshold"],
        cxpb=params["crossover_probability"],
        mutpb=params["mutation_probability"],
        physical_constraints_freq=params["physical_constraints_freq"],
        exploration_steps=params["exploration_steps"],
        PP=False, demc_hybrid=False, plot_mode="plot-only",
    )

    # --- selection from results CSV ---
    df = pd.read_csv(results_csv)
    metric = params["loss_metric"]
    df = df.sort_values(metric, ascending=True).reset_index(drop=True)
    k = int(np.ceil(len(df) * (args.percentile / 100.0)))
    k = max(k, 1)
    sel = df.iloc[:k]

    # --- existing history (if any) ---
    npz_path = os.path.join(outdir, "walker_history.npz")
    existing = os.path.exists(npz_path)

    existing_walker_history = {}
    existing_mdf = []
    existing_alpha = []
    existing_age = []
    next_wid = 0
    seen_inds = set()

    if existing:
        walker_ids, histories, mdf_data, alpha_data, age_data = _load_existing_npz(npz_path)
        for wid, hist in zip(walker_ids.tolist(), list(histories)):
            existing_walker_history[int(wid)] = [np.asarray(frame, dtype=float) for frame in list(hist)]
            for frame in hist:
                seen_inds.add(_as_tuple(frame))
        existing_mdf = list(mdf_data)
        existing_alpha = list(alpha_data)
        existing_age = list(age_data)
        next_wid = int(max(walker_ids)) + 1
        print(f"[load] found existing history with {len(walker_ids)} walkers, {len(existing_mdf)} MDF entries")
    else:
        print("[load] no existing history; starting fresh")

    # Seed current runtime with existing so writes are merged
    GalGA.walker_history = {k: [np.asarray(vv, dtype=float) for vv in v] for k, v in existing_walker_history.items()}
    GalGA.mdf_data   = list(existing_mdf)
    GalGA.alpha_data = list(existing_alpha)
    GalGA.age_data   = list(existing_age)

    # --- build candidate list + check what needs computing ---
    candidates = []
    for _, row in sel.iterrows():
        ind = _row_to_individual(row)
        candidates.append((_as_tuple(ind), ind, float(np.asarray(row[metric]).ravel()[0])))

    todo = [(ind_t, ind, old) for (ind_t, ind, old) in candidates if ind_t not in seen_inds]
    skip = [(ind_t, ind, old) for (ind_t, ind, old) in candidates if ind_t in seen_inds]

    print(f"[plan] selected={len(candidates)}  already_have={len(skip)}  to_compute={len(todo)}")

    # --- evaluate only missing ones ---
    def _eval(i, ind, old_loss):
        print(f"[{i+1:>5d}/{len(todo)}] start  σ2={ind[5]:.6g}  t2={ind[7]:.6g}  τ2={ind[9]:.6g}  old={old_loss:.6g}")
        loss, result = GalGA.evaluate(ind)
        new_loss = float(np.asarray(loss).ravel()[0])
        print(f"[{i+1:>5d}/{len(todo)}] done   σ2={ind[5]:.6g}  t2={ind[7]:.6g}  τ2={ind[9]:.6g}  old={old_loss:.6g}  new={new_loss:.6g}")
        return (ind, result)

    new_results = []
    if len(todo) > 0:
        with ThreadPoolExecutor(max_workers=(os.cpu_count() or 1)) as ex:
            for (ind_t, ind, old), (out_ind, result) in zip(todo, ex.map(lambda p: _eval(*p), [(i, ind, old) for i, (_, ind, old) in enumerate(todo)])):
                new_results.append((out_ind, result))

    # --- record results and append histories ---
    # 1) push curves/metrics for each new evaluation
    for ind, result in new_results:
        GalGA._record_evaluation_result(result)

    # 2) append to walker_history
    #    - if the exact ind already exists: do nothing (we're not recomputing)
    #    - if it's new: create a new walker id and add a one-deep history [ind]
    for ind, _ in new_results:
        GalGA.walker_history[int(next_wid)] = [np.asarray(ind, dtype=float)]
        next_wid += 1

    # --- assemble aligned inds for convenience (existing last frames + new ones) ---
    walker_ids_sorted = np.array(sorted(GalGA.walker_history.keys()), dtype=np.int32)
    inds_aligned = np.array([np.asarray(GalGA.walker_history[k][-1], dtype=float) for k in walker_ids_sorted], dtype=float)

    # write merged NPZ
    write_history(outdir, GalGA, inds=inds_aligned, df_sel=sel, metric_col=metric)

if __name__ == "__main__":
    main()
