"""
Utility functions for MDF_GCE_SMC_DEMC.

This module is the SINGLE SOURCE OF TRUTH for:
- Directory creation and management
- File discovery (CSV, NPZ, checkpoints)
- DataFrame loading and standardization
- Array conversions and helpers
"""

import os
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

from .constants import (
    PARAM_COLUMN_ALIASES,
    RESULTS_CSV_PATTERN,
    LOSS_COLUMNS,
    CONTINUOUS_PARAMS,
    CATEGORICAL_PARAMS,
)

# =============================================================================
# DIRECTORY MANAGEMENT
# =============================================================================

def ensure_dirs(output_path: str, subdirs: Optional[List[str]] = None) -> None:
    """
    Create output directory and standard subdirectories.
    
    Parameters
    ----------
    output_path : str
        Base output directory
    subdirs : list of str, optional
        Additional subdirectories to create. Default creates 'analysis' and 'uncertainty'.
    """
    os.makedirs(output_path, exist_ok=True)
    
    if subdirs is None:
        subdirs = ["analysis", "uncertainty"]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(output_path, subdir), exist_ok=True)


def ensure_output_dirs(base_path: str) -> None:
    """Alias for ensure_dirs for backward compatibility."""
    ensure_dirs(base_path)


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def find_latest_csv(path: str, pattern: str = "simulation_results") -> Optional[str]:
    """
    Find the most recent simulation results CSV in a directory.
    
    Prefers 'simulation_results.csv' if it exists, otherwise finds the
    highest generation number from files like 'simulation_results_gen_N.csv'.
    
    Parameters
    ----------
    path : str
        Directory to search
    pattern : str
        Base filename pattern to match
        
    Returns
    -------
    str or None
        Full path to the CSV, or None if not found
    """
    if not os.path.isdir(path):
        return None
    
    files = [f for f in os.listdir(path) 
             if f.startswith(pattern) and f.endswith('.csv')]
    
    if not files:
        return None
    
    # Prefer the merged final file
    if f"{pattern}.csv" in files:
        return os.path.join(path, f"{pattern}.csv")
    
    # Otherwise find highest generation
    gens = []
    for f in files:
        match = re.search(r'_gen_(\d+)\.csv$', f)
        if match:
            gens.append((int(match.group(1)), f))
    
    if not gens:
        # Return first match if no generation pattern found
        return os.path.join(path, files[0])
    
    # Return highest generation
    _, best_file = max(gens, key=lambda x: x[0])
    return os.path.join(path, best_file)


def find_highest_gen_file(folder: str) -> Optional[str]:
    """Alias for find_latest_csv for backward compatibility."""
    return find_latest_csv(folder)


def get_latest_csv(path: str) -> Optional[str]:
    """Alias for find_latest_csv for backward compatibility."""
    return find_latest_csv(path)


def find_result_folders(root_dir: str = ".") -> List[Tuple[str, str, List[str]]]:
    """
    Recursively find all folders containing simulation results.
    
    Parameters
    ----------
    root_dir : str
        Root directory to search
        
    Returns
    -------
    list of tuples
        Each tuple contains (folder_name, folder_path, list_of_csv_files)
    """
    candidates = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        csvs = [f for f in filenames 
                if f.startswith("simulation_results") and f.endswith(".csv")]
        if csvs:
            name = os.path.basename(dirpath) or dirpath
            candidates.append((name, dirpath, csvs))
    
    return sorted(candidates, key=lambda x: x[0])


def choose_primary_csv(csv_files: List[str], folder: str = "") -> str:
    """
    Choose the primary CSV from a list of result files.
    
    Prefers 'simulation_results.csv', then highest generation number.
    
    Parameters
    ----------
    csv_files : list of str
        List of CSV filenames
    folder : str
        Folder path to prepend
        
    Returns
    -------
    str
        Path to the chosen CSV
    """
    if "simulation_results.csv" in csv_files:
        return os.path.join(folder, "simulation_results.csv") if folder else "simulation_results.csv"
    
    # Find highest generation
    gens = []
    for f in csv_files:
        match = re.search(r'_gen_(\d+)\.csv$', f)
        if match:
            gens.append((int(match.group(1)), f))
    
    if gens:
        _, best = max(gens, key=lambda x: x[0])
        return os.path.join(folder, best) if folder else best
    
    # Fallback to first file
    return os.path.join(folder, csv_files[0]) if folder else csv_files[0]


# =============================================================================
# DATAFRAME LOADING AND STANDARDIZATION
# =============================================================================

def load_results_df(
    path: str,
    standardize_columns: bool = True,
    sort_by_loss: bool = True,
) -> pd.DataFrame:
    """
    Load simulation results CSV with standardized column names.
    
    Parameters
    ----------
    path : str
        Path to CSV file or directory containing results
    standardize_columns : bool
        If True, rename columns using PARAM_COLUMN_ALIASES
    sort_by_loss : bool
        If True, sort by loss column (ascending)
        
    Returns
    -------
    pd.DataFrame
        Loaded and optionally processed results
    """
    # Handle directory input
    if os.path.isdir(path):
        csv_path = find_latest_csv(path)
        if csv_path is None:
            raise FileNotFoundError(f"No simulation_results CSV found in {path}")
        path = csv_path
    
    df = pd.read_csv(path)
    
    # Standardize column names
    if standardize_columns:
        df = standardize_df_columns(df)
    
    # Sort by loss
    if sort_by_loss:
        loss_col = get_loss_column(df)
        if loss_col:
            df = df.sort_values(loss_col, ascending=True).reset_index(drop=True)
    
    return df


def standardize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column names using aliases.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names
    """
    rename_map = {}
    for old_name, new_name in PARAM_COLUMN_ALIASES.items():
        if old_name in df.columns and new_name not in df.columns:
            rename_map[old_name] = new_name
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    return df


def get_loss_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find the loss/fitness column in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    str or None
        Name of the loss column, or None if not found
    """
    for col in LOSS_COLUMNS:
        if col in df.columns:
            return col
    return None


def filter_valid_rows(
    df: pd.DataFrame,
    params: Optional[List[str]] = None,
    require_nonzero: bool = True,
) -> pd.DataFrame:
    """
    Filter DataFrame to rows with valid (non-NaN, optionally non-zero) parameter values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    params : list of str, optional
        Parameters to check. Default is CONTINUOUS_PARAMS.
    require_nonzero : bool
        If True, also filter out rows with zero values
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if params is None:
        params = CONTINUOUS_PARAMS
    
    cols = [c for c in params if c in df.columns]
    if not cols:
        return df
    
    mask = df[cols].notna().all(axis=1)
    if require_nonzero:
        mask &= (df[cols] != 0).all(axis=1)
    
    return df[mask].copy()


# =============================================================================
# ARRAY UTILITIES
# =============================================================================

def as_object_array(seq: Any) -> np.ndarray:
    """
    Safely convert a sequence to a 1D numpy object array.
    
    Handles None, existing object arrays, lists, and tuples.
    
    Parameters
    ----------
    seq : any
        Input sequence or None
        
    Returns
    -------
    np.ndarray
        1D object array
    """
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


def row_to_individual(row: pd.Series) -> List:
    """
    Convert a DataFrame row to GA individual format (15-element list).
    
    Parameters
    ----------
    row : pd.Series
        Row from simulation_results DataFrame
        
    Returns
    -------
    list
        15-element list matching GA individual structure
    """
    discrete = [
        int(row.get("comp_idx", 0)),
        int(row.get("imf_idx", 0)),
        int(row.get("sn1a_idx", 0)),
        int(row.get("sy_idx", 0)),
        int(row.get("sn1ar_idx", 0)),
    ]
    continuous = [
        float(row.get("sigma_2", 0)),
        float(row.get("t_1", 0)),
        float(row.get("t_2", 0)),
        float(row.get("infall_1", 0)),
        float(row.get("infall_2", 0)),
        float(row.get("sfe", 0)),
        float(row.get("delta_sfe", 0)),
        float(row.get("imf_upper", 0)),
        float(row.get("mgal", 0)),
        float(row.get("nb", 0)),
    ]
    return discrete + continuous


def individual_to_tuple(ind: List) -> Tuple:
    """
    Convert GA individual to hashable tuple for matching.
    
    Parameters
    ----------
    ind : list
        15-element GA individual
        
    Returns
    -------
    tuple
        Hashable tuple representation
    """
    return tuple(
        [int(ind[0]), int(ind[1]), int(ind[2]), int(ind[3]), int(ind[4])] +
        [round(float(ind[i]), 9) for i in range(5, 15)]
    )


# =============================================================================
# NPZ FILE HANDLING
# =============================================================================

def load_walker_history(path: str) -> Dict[str, Any]:
    """
    Load walker history NPZ file.
    
    Parameters
    ----------
    path : str
        Path to walker_history.npz
        
    Returns
    -------
    dict
        Dictionary with keys: walker_ids, histories, mdf_data, alpha_data, age_data
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Walker history not found: {path}")
    
    npz = np.load(path, allow_pickle=True)
    
    def to_list(x):
        if isinstance(x, np.ndarray) and x.dtype == object:
            return list(x)
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]
    
    return {
        "walker_ids": npz["walker_ids"],
        "histories": to_list(npz["histories"]),
        "mdf_data": to_list(npz.get("mdf_data", [])),
        "alpha_data": to_list(npz.get("alpha_data", [])),
        "age_data": to_list(npz.get("age_data", [])),
    }


def save_walker_history(
    path: str,
    walker_ids: np.ndarray,
    histories: List,
    mdf_data: Optional[List] = None,
    alpha_data: Optional[List] = None,
    age_data: Optional[List] = None,
) -> None:
    """
    Save walker history to NPZ file.
    
    Parameters
    ----------
    path : str
        Output path
    walker_ids : np.ndarray
        Walker ID array
    histories : list
        List of walker histories
    mdf_data, alpha_data, age_data : list, optional
        Associated simulation data
    """
    np.savez_compressed(
        path,
        walker_ids=np.array(walker_ids, dtype=np.int32),
        histories=as_object_array(histories),
        mdf_data=as_object_array(mdf_data or []),
        alpha_data=as_object_array(alpha_data or []),
        age_data=as_object_array(age_data or []),
    )


# =============================================================================
# OBSERVATIONAL DATA
# =============================================================================

def load_bensby_data(file_path: str = "data/Bensby_Data.tsv") -> Optional[pd.DataFrame]:
    """
    Load Bensby et al. observational data.
    
    Parameters
    ----------
    file_path : str
        Path to TSV file
        
    Returns
    -------
    pd.DataFrame or None
        Loaded data, or None if file not found
    """
    # Try multiple paths
    paths_to_try = [
        file_path,
        os.path.join("..", file_path),
        os.path.join(os.path.dirname(__file__), "..", "data", "Bensby_Data.tsv"),
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t')
            return df
    
    return None


# =============================================================================
# MISCELLANEOUS
# =============================================================================

def find_nearest(array: np.ndarray, value: float) -> Tuple[int, float]:
    """
    Find the index and value of the nearest element in an array.
    
    Parameters
    ----------
    array : np.ndarray
        Array to search
    value : float
        Target value
        
    Returns
    -------
    tuple
        (index, nearest_value)
    """
    idx = np.abs(array - value).argmin()
    return idx, array[idx]


def alloc_cores() -> int:
    """
    Determine the number of CPU cores available for parallel processing.
    
    Respects SLURM allocation if running in a job.
    
    Returns
    -------
    int
        Number of available cores
    """
    try:
        # Exact count in current cpuset (best under Slurm)
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # Fallback to Slurm env or Python's view
        return int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
