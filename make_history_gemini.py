#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import Gal_GA_PP as Gal_GA  # parse_inlist, GalacticEvolutionGA, etc.

# ---------- helpers (side-effect free) ----------

def get_latest_csv(path: str) -> str | None:
    """Find the latest results CSV, preferring 'simulation_results.csv'."""
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
    """Loads the Bensby observational data."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        print(f"Loaded Bensby data with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Warning: Bensby data file not found at {file_path}. Proceeding without it.")
        return None

def _as_object_array(seq):
    """Safely convert a sequence to a 1D numpy object array."""
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

def _row_to_individual(row):
    """Convert a DataFrame row to the 15-parameter individual list."""
    disc = [int(row['comp_idx']), int(row['imf_idx']), int(row['sn1a_idx']), int(row['sy_idx']), int(row['sn1ar_idx'])]
    cont = [float(row['sigma_2']), float(row['t_1']), float(row['t_2']), float(row['infall_1']), float(row['infall_2']),
            float(row['sfe']), float(row['delta_sfe']), float(row['imf_upper']), float(row['mgal']), float(row['nb'])]
    return disc + cont

def _as_tuple(ind):
    """Convert an individual list/array to a hashable, rounded tuple."""
    # Round to 9 decimal places to avoid floating point mismatches
    # between CSV (text) and NPZ (binary) representations.
    return tuple(round(float(x), 9) for x in ind)

def _load_existing_npz(path):
    """
    Load the existing walker history NPZ file.
    Loads 'inds' (final parameters) for each history.
    Returns a set of ALL parameter tuples ever recorded in any history.
    """
    if not os.path.exists(path):
        print("[load] No existing history file found. Starting fresh.")
        return {}, {}, {}, {}, np.empty((0, 15), dtype=float), set() # Return empty set

    print(f"[load] Loading existing history from: {path}")
    npz = np.load(path, allow_pickle=True)
    
    walker_ids = npz.get("walker_ids", np.array([]))
    histories = npz.get("histories", np.array([], dtype=object))
    
    # Load 'inds' or reconstruct it from the last frame of each history
    if "inds" in npz:
        inds = npz["inds"]
    else:
        print("[load] 'inds' array not found. Reconstructing from history end-frames.")
        inds_list = []
        param_len = 15  # Default length of an individual
        
        # Handle case where histories might be empty or malformed
        if len(histories) > 0 and histories.dtype == object:
            # Check if histories is an array of lists/arrays
            if len(histories[0]) > 0:
                 param_len = len(histories[0][-1])
        
        for hist in histories:
            if len(hist) > 0:
                inds_list.append(np.asarray(hist[-1], dtype=float))
            else:
                inds_list.append(np.full(param_len, np.nan))
        inds = np.array(inds_list, dtype=float)

    # Load data blobs (these are not aligned with walkers)
    def _to_list(arr):
        # Handle 0-d object arrays which can't be list()ed
        if isinstance(arr, np.ndarray) and arr.ndim == 0:
            arr = arr.item() # extract the list
        
        if isinstance(arr, (list, np.ndarray)):
            return list(arr)
        return [arr] if arr is not None else []


    mdf_data = _to_list(npz.get("mdf_data", []))
    alpha_data = _to_list(npz.get("alpha_data", []))
    age_data = _to_list(npz.get("age_data", []))

    # Create the walker_id -> history map
    walker_history_map = {
        int(wid): [np.asarray(frame, dtype=float) for frame in list(hist)]
        for wid, hist in zip(walker_ids, histories)
    }
    
    # --- NEW: Build a set of ALL frames ---
    all_frames_set = set()
    for hist in histories:
        for frame in hist:
            all_frames_set.add(_as_tuple(frame))
    # --- END NEW ---

    print(f"[load] Found {len(walker_ids)} existing walkers and {len(all_frames_set)} unique parameter sets in history.")
    return walker_history_map, mdf_data, alpha_data, age_data, inds, all_frames_set

def write_matched_history(output_path, gal, walker_ids, histories, inds, losses):
    """
    Saves the new, matched NPZ file.
    - walker_ids, histories, inds, and losses are all aligned.
    - mdf_data, alpha_data, and age_data are the appended blobs of all data.
    """
    path = os.path.join(output_path, "walker_history_matched.npz")
    
    # Get unaligned data blobs from the GalGA instance
    mdf_obj   = _as_object_array(getattr(gal, "mdf_data", []))
    alpha_obj = _as_object_array(getattr(gal, "alpha_data", []))
    age_obj   = _as_object_array(getattr(gal, "age_data", []))

    # Convert histories (list of lists) to a 1D object array for saving
    histories_obj = _as_object_array(histories)
    
    # Sanity check alignment
    if not (len(walker_ids) == len(histories_obj) == len(inds) == len(losses)):
        print(f"[write] FATAL ERROR: Aligned array lengths do not match!")
        print(f"  walker_ids: {len(walker_ids)}")
        print(f"  histories:  {len(histories_obj)}")
        print(f"  inds:       {len(inds)}")
        print(f"  losses:     {len(losses)}")
        return

    np.savez_compressed(
        path,
        walker_ids=walker_ids,      # Aligned array
        histories=histories_obj,    # Aligned array
        inds=inds,                  # Aligned array
        losses=losses,              # Aligned array
        
        # Unaligned data blobs (as before)
        mdf_data=mdf_obj,
        alpha_data=alpha_obj,
        age_data=age_obj,
    )
    
    nan_count = np.count_nonzero(np.isnan(losses))
    print(f"[write] Successfully saved matched history file to: {path}")
    print(f"[write]   Total walkers: {len(walker_ids)}")
    print(f"[write]   Data entries (MDF/Alpha/Age): {len(mdf_obj)}/{len(alpha_obj)}/{len(age_obj)}")
    if nan_count > 0:
        print(f"[write] WARNING: {nan_count} walkers had no matching loss value in the results CSV.")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Cross-match GA results with walker histories and re-run missing walkers.")
    ap.add_argument("entry", help="Path to a simulation_results*.csv file or a directory containing one.")
    ap.add_argument("--percentile", type=float, default=10.0, help="Percentile of best walkers to include (e.g., 10.0 for top 10%%).")
    ap.add_argument("-t", "--threads", type=int, default=None, help="Number of threads to use for re-evaluation. Defaults to all available cores.")
    args = ap.parse_args()

    max_workers = args.threads if args.threads else (os.cpu_count() or 1)
    print(f"Using max {max_workers} worker threads.")

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
    if not os.path.exists(pcard):
        raise FileNotFoundError(f"Could not find 'bulge_pcard.txt' in {outdir}")

    # --- 1. Load Config and Build GalGA instance ---
    print(f"Loading parameters from: {pcard}")
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

    # --- 2. Load and Filter Results CSV ---
    print(f"Loading results from: {results_csv}")
    df_full = pd.read_csv(results_csv)
    loss_col = params.get("loss_metric", "fitness")
    if loss_col not in df_full.columns:
        if "fitness" in df_full.columns:
            loss_col = "fitness"
        else:
            raise KeyError(f"Loss metric '{loss_col}' not found in {results_csv}")
    print(f"Using loss metric: '{loss_col}'")

    df_full = df_full.sort_values(loss_col, ascending=True).reset_index(drop=True)
    
    # Get all results within the percentile cutoff
    cutoff_loss = df_full[loss_col].quantile(args.percentile / 100.0)
    df_top_k = df_full[df_full[loss_col] <= cutoff_loss].copy()
    print(f"Selected top {args.percentile}%% walkers ({len(df_top_k)} individuals) with loss <= {cutoff_loss:.6f}")

    # Create a lookup map for (ind_tuple) -> loss
    all_losses_map = {
        _as_tuple(_row_to_individual(row)): float(row[loss_col])
        for _, row in df_full.iterrows()
    }

    # --- 3. Load Existing History File ---
    npz_path = os.path.join(outdir, "walker_history.npz")
    existing_walker_history, existing_mdf, existing_alpha, existing_age, existing_inds, all_frames_set = _load_existing_npz(npz_path)

    # Create a set of final parameters from the existing history for quick lookup
    # This check is no longer the main one, but keep it for now.
    seen_final_inds = set(_as_tuple(ind) for ind in existing_inds if not np.isnan(ind).any())
    
    # Populate GalGA instance with existing data
    GalGA.walker_history = existing_walker_history
    GalGA.mdf_data   = list(existing_mdf)
    GalGA.alpha_data = list(existing_alpha)
    GalGA.age_data   = list(existing_age)
    next_wid = max(existing_walker_history.keys()) + 1 if existing_walker_history else 0

    # --- 4. Identify Missing Walkers ---
    candidates_to_run = []
    for _, row in df_top_k.iterrows():
        ind = _row_to_individual(row)
        ind_t = _as_tuple(ind)
        # *** THIS IS THE MAIN LOGIC FIX ***
        # Check against the set of ALL frames, not just the final ones
        if ind_t not in all_frames_set:
            candidates_to_run.append((ind, float(row[loss_col])))

    print(f"[plan] Found {len(df_top_k)} walkers in percentile.")
    print(f"[plan]   {len(df_top_k) - len(candidates_to_run)} are already in '{npz_path}' history.")
    print(f"[plan]   {len(candidates_to_run)} need to be re-evaluated.")

    # --- 5. Re-calculate Missing Walkers ---
    def _eval(i, ind, old_loss):
        """Wrapper for parallel evaluation."""
        loss, result = GalGA.evaluate(ind)
        new_loss = float(np.asarray(loss).ravel()[0])
        return (ind, result, old_loss, new_loss)

    if candidates_to_run:
        print(f"--- Starting re-evaluation of {len(candidates_to_run)} walkers ---")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_eval, i, ind, old_loss): (ind, old_loss)
                for i, (ind, old_loss) in enumerate(candidates_to_run)
            }
            
            for future in tqdm(as_completed(futures), total=len(candidates_to_run), desc="Evaluating walkers"):
                ind, result, old_loss, new_loss = future.result()
                
                # Record the data (MDF, alpha, age) to the GalGA instance's lists
                GalGA._record_evaluation_result(result)
                
                # Add a new, single-frame history for this walker
                GalGA.walker_history[int(next_wid)] = [np.asarray(ind, dtype=float)]
                next_wid += 1
                
                tqdm.write(f"  [eval] WID {next_wid-1}: old_loss={old_loss:.6f}, new_loss={new_loss:.6f} (diff: {new_loss-old_loss:+.6f})")
        print("--- Re-evaluation complete ---")

    # --- 6. Save Matched Data to New File ---
    print("Assembling final aligned arrays...")
    
    # We only want to save walkers that are in our top percentile list
    final_walkers_map = {} # ind_tuple -> (wid, history)
    
    # Map all *final positions* from the history to their wid and history
    for wid, hist in GalGA.walker_history.items():
        if len(hist) > 0:
            final_ind_t = _as_tuple(hist[-1])
            # We map the *final* position tuple to the walker data
            final_walkers_map[final_ind_t] = (wid, hist)
        else:
            print(f"Warning: Walker ID {wid} has an empty history.")


    # Prepare final aligned lists
    final_walker_ids = []
    final_histories_list = []
    final_inds_list = []
    final_losses_list = []

    # Iterate through the official top-k dataframe to ensure order and completeness
    for _, row in df_top_k.iterrows():
        ind = _row_to_individual(row)
        ind_t = _as_tuple(ind)
        loss = float(row[loss_col])
        
        # Find this walker in our history map
        # We must check *all* frames, not just the final one.
        
        found_wid = -1
        found_hist = None
        
        # This is slow, but necessary for cross-matching
        # We must find which walker ID corresponds to this top-k individual
        if ind_t in final_walkers_map:
            # Easy case: this walker is the *final* position of a known walker
            wid, hist = final_walkers_map[ind_t]
            found_wid = wid
            found_hist = hist
        else:
            # Hard case: this walker is from the CSV but not a *final* position
            # We must search all frames of all histories
            for wid, hist in GalGA.walker_history.items():
                for frame in hist:
                    if _as_tuple(frame) == ind_t:
                        found_wid = wid
                        found_hist = hist
                        break
                if found_wid != -1:
                    break

        if found_wid != -1:
            final_walker_ids.append(found_wid)
            final_histories_list.append([np.asarray(f, dtype=float) for f in found_hist])
            final_inds_list.append(np.asarray(found_hist[-1], dtype=float)) # Save the final position
            final_losses_list.append(loss) # Save the loss from the CSV
        else:
            # This should only happen if a re-evaluated walker fails to be added,
            # which would be a bug.
            print(f"Warning: Walker {ind_t} from top-k was not found in final history map. Skipping.")

    # Convert to numpy arrays
    final_walker_ids_np = np.array(final_walker_ids, dtype=np.int32)
    final_inds_np = np.array(final_inds_list, dtype=float)
    final_losses_np = np.array(final_losses_list, dtype=float)

    # Write the new matched file
    write_matched_history(
        outdir,
        GalGA,  # Pass for its data blobs (mdf, alpha, age)
        final_walker_ids_np,
        final_histories_list,
        final_inds_np,
        final_losses_np
    )
    
    print("--- Process complete ---")

if __name__ == "__main__":
    main()