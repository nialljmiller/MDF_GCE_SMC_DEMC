#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import Gal_GA_PP as Gal_GA  # parse_inlist, GalacticEvolutionGA, etc.

# ================================================================
# 0) Utilities
# ================================================================
GENE_COLS_INT   = ['comp_idx','imf_idx','sn1a_idx','sy_idx','sn1ar_idx']
GENE_COLS_FLOAT = ['sigma_2','t_1','t_2','infall_1','infall_2',
                   'sfe_val','delta_sfe_val','imf_upper','mgal','nb']
GENE_COLS_ALL   = GENE_COLS_INT + GENE_COLS_FLOAT


def _as_object_array(seq):
    if seq is None:
        return np.empty(0, dtype=object)
    if isinstance(seq, np.ndarray) and seq.dtype == object and seq.ndim == 1:
        return seq
    try:
        n = len(seq)
    except TypeError:
        out = np.empty(1, dtype=object)
        out[0] = seq
        return out
    out = np.empty(n, dtype=object)
    for i, v in enumerate(seq):
        out[i] = v
    return out

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

def _tuple_from_genes(ind):
    # 15-long tuple for exact matching (csv -> float is deterministic)
    return tuple([int(ind[0]), int(ind[1]), int(ind[2]), int(ind[3]), int(ind[4])] +
                 [float(ind[i]) for i in range(5, 15)])

def _tuple_from_dfrow(row):
    return tuple([int(row[c]) for c in GENE_COLS_INT] +
                 [float(row[c]) for c in GENE_COLS_FLOAT])

def _load_existing_npz(path):
    npz = np.load(path, allow_pickle=True)
    def _to_list(x):
        if isinstance(x, np.ndarray) and x.dtype == object: return list(x)
        if isinstance(x, (list, tuple)): return list(x)
        return [x]
    walker_ids = np.array(npz["walker_ids"], dtype=np.int32)
    histories  = _to_list(npz["histories"])
    mdf_data   = _to_list(npz["mdf_data"])   if "mdf_data"   in npz else []
    alpha_data = _to_list(npz["alpha_data"]) if "alpha_data" in npz else []
    age_data   = _to_list(npz["age_data"])   if "age_data"   in npz else []
    return walker_ids, histories, mdf_data, alpha_data, age_data

def load_bensby_data(file_path='data/Bensby_Data.tsv'):
    df = pd.read_csv(file_path, sep='\t')
    return df

# ================================================================
# 0) CLI
# ================================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Attach loss to walker histories (loss-aware tracks).")
    ap.add_argument("entry",
                    help="Path to a results CSV OR a directory containing simulation_results*.csv")
    ap.add_argument("--percentile", type=float, default=10.0,
                    help="Only crossmatch/recompute for top P%% rows by loss from the results CSV.")
    ap.add_argument("--history", default=None,
                    help="Path to walker_history.npz. Default: <outdir>/walker_history.npz")
    ap.add_argument("--out", default=None,
                    help="Output NPZ path. Default: <outdir>/history_with_loss.npz")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 1)//2),
                    help="Max workers for optional recompute.")
    return ap.parse_args()

# ================================================================
# main
# ================================================================

def main():
    args = parse_args()

    # ---------- resolve input/output ----------
    entry = os.path.abspath(args.entry)
    if os.path.isdir(entry):
        results_csv = get_latest_csv(entry)
        if results_csv is None:
            raise FileNotFoundError(f"No simulation_results*.csv found under {entry}")
        outdir = entry
    else:
        results_csv = entry
        outdir = os.path.dirname(results_csv) or "."

    hist_path = args.history or os.path.join(outdir, "walker_history.npz")
    out_npz   = args.out or os.path.join(outdir, "history_with_loss.npz")
    out_csv   = os.path.join(outdir, "history_with_loss_long.csv")

    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"walker history not found: {hist_path}")

    # ---------- 0) read user inputs ----------
    pct = float(args.percentile)
    if not (0 < pct <= 100):
        raise ValueError("--percentile must be in (0, 100].")

    # ---------- load CSV + select top percentile by loss ----------
    df_all = pd.read_csv(results_csv)
    if 'loss' not in df_all.columns and 'fitness' in df_all.columns:
        df_all['loss'] = df_all['fitness']
    df_all = df_all.sort_values('loss', ascending=True).reset_index(drop=True)

    loss_cut = df_all['loss'].quantile(args.percentile / 100.0)   # <= this is the threshold
    # Keep a LUT for exact matches (some will match, many won't — that’s fine)
    lut_all = {_tuple_from_dfrow(r): float(r['loss']) for _, r in df_all.iterrows()}

    top_keys = set(_tuple_from_dfrow(r) for _, r in df_top.iterrows())

    # ---------- 1) load histories (walkers + tracks we already have) ----------
    walker_ids, histories_raw, mdf_data, alpha_data, age_data = _load_existing_npz(hist_path)

    # Normalize to numpy float arrays
    histories = []
    for seq in histories_raw:
        frames = [np.asarray(f, dtype=float) for f in list(seq)]
        histories.append(frames)

    # ---------- 2–3) match histories to results; make aligned loss histories ----------
    loss_histories = []
    missing_frames = []  # (wid, gen, genes)
    for wid, frames in zip(walker_ids.tolist(), histories):
        loss_seq = []
        for gen, genes in enumerate(frames):
            key = _tuple_from_genes(genes)
            if key in lut_all:                      # exact match found in CSV
                loss_seq.append(float(lut_all[key]))
            else:
                loss_seq.append(np.nan)
                missing_frames.append((wid, gen, genes))
        loss_histories.append(np.asarray(loss_seq, float))

    print(f"[match] walkers={len(walker_ids)}  missing_losses={int(sum(np.isnan(s).sum() for s in loss_histories))}  "
          f"loss_cut@{args.percentile}%={loss_cut:.6g}")


    # ---------- 5) recompute losses for eligible missing frames (top percentile only) ----------
    if len(missing_frames) > 0 and args.threads > 0:
        # Build GalGA in plot-only mode (no GA run), so evaluate() is available.
        pcard = os.path.join(outdir, "bulge_pcard.txt")
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

        def _eval_one(item):
            wid, gen, key, genes = item
            # genes is a 15-long vector; evaluate expects a DEAP "individual" shape
            loss, _ = GalGA.evaluate(list(genes))
            return wid, gen, float(np.asarray(loss).ravel()[0])

        # fan out
        results = {}
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            futs = [ex.submit(_eval_one, it) for it in missing_frames]
            for f in as_completed(futs):
                wid, gen, loss_val = f.result()
                results[(wid, gen)] = loss_val

        # inject recomputed losses
        for i, wid in enumerate(walker_ids.tolist()):
            for gen in range(len(loss_histories[i])):
                if np.isnan(loss_histories[i][gen]):
                    v = results.get((wid, gen))
                    if v is not None:
                        loss_histories[i][gen] = v

        # stats
        n_after = sum(np.isnan(s).sum() for s in loss_histories)
        print(f"[recompute] filled={int(n_missing - n_after)}  still_missing={int(n_after)}")
    else:
        print("[recompute] none needed or threads=0")

    # ---------- 6) save single cross-matched file (loss-aware histories) ----------
    # object arrays so shapes can differ per walker
    histories_obj = _as_object_array([np.asarray(frames, dtype=float) for frames in histories])
    losses_obj    = _as_object_array([np.asarray(seq,    dtype=float) for seq in loss_histories])

    np.savez_compressed(
        out_npz,
        walker_ids=walker_ids,
        histories=histories_obj,        # per-walker [gen, 15]
        loss_histories=losses_obj,       # per-walker [gen]
        mdf_data=_as_object_array(mdf_data),
        alpha_data=_as_object_array(alpha_data),
        age_data=_as_object_array(age_data),
    )
    print(f"[write] {out_npz}  (walkers={len(walker_ids)} "
          f"with_loss={sum(len(x) for x in loss_histories)})")

    # also emit a long-form CSV for quick plotting/debug
    long_rows = []
    for wid, frames, lseq in zip(walker_ids.tolist(), histories, loss_histories):
        for gen, (genes, lv) in enumerate(zip(frames, lseq)):
            row = {'walker_id': wid, 'generation': gen, 'loss': lv}
            for j, name in enumerate(GENE_COLS_ALL):
                row[name] = float(genes[j])
            long_rows.append(row)
    pd.DataFrame(long_rows).to_csv(out_csv, index=False)
    print(f"[write] {out_csv}")

if __name__ == "__main__":
    main()
