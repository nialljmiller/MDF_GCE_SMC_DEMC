#!/usr/bin/env python3
"""
Enhanced matching utilities for posterior weights to walker tracks.

This module provides a unified, fast matching system that ensures self-consistent
plots across all visualization functions by matching posterior weights to walker
tracks on ALL parameters.

Authors: N Miller
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List


# Parameter bookkeeping (from posterior_utils.py)
_RESULT_PARAM_INDICES = {
    'comp_idx': 0,
    'imf_idx': 1,
    'sn1a_idx': 2,
    'sy_idx': 3,
    'sn1ar_idx': 4,
    'sigma_2': 5,
    't_1': 6,
    't_2': 7,
    'infall_1': 8,
    'infall_2': 9,
    'sfe': 10,
    'delta_sfe': 11,
    'imf_upper': 12,
    'mgal': 13,
    'nb': 14,
}

_DISCRETE_PARAM_COLS = {'comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx'}

_MATCH_PARAM_COLS = [
    'comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx',
    'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2',
    'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb',
]


def _normalize_param_value(name: str, value: float, ndigits: int = 12):
    """Normalize a parameter value for matching."""
    if name in _DISCRETE_PARAM_COLS:
        return int(round(float(value)))
    return float(np.round(float(value), ndigits))


def _normalize_param_series(name: str, series: pd.Series, ndigits: int = 12):
    """Normalize a parameter series for matching."""
    series = pd.to_numeric(series, errors='coerce')
    if name in _DISCRETE_PARAM_COLS:
        return series.round().astype('Int64')
    return series.round(ndigits)


def build_parameter_lookup(GalGA) -> Dict[tuple, int]:
    """
    Build a fast lookup dictionary mapping parameter tuples to model indices.
    
    This creates a comprehensive mapping from all parameter combinations to their
    corresponding model indices in GalGA.results, enabling O(1) matching.
    
    Parameters
    ----------
    GalGA : object
        Container with 'results' attribute containing model parameters
        
    Returns
    -------
    lookup : dict
        Dictionary mapping normalized parameter tuples to model indices
        Key: tuple of (comp_idx, imf_idx, ..., nb)
        Value: int model index in GalGA.results
    """
    # Check cache first
    if hasattr(GalGA, '_param_lookup_cache'):
        return GalGA._param_lookup_cache
    
    lookup = {}
    results = getattr(GalGA, 'results', None) or []
    
    if not results:
        GalGA._param_lookup_cache = lookup
        return lookup
    
    # Build lookup from results
    for idx, result in enumerate(results):
        result = np.asarray(result, dtype=float)
        if result.size < len(_MATCH_PARAM_COLS):
            continue
        
        try:
            key = tuple(
                _normalize_param_value(name, result[_RESULT_PARAM_INDICES[name]])
                for name in _MATCH_PARAM_COLS
            )
            # Only store first occurrence (best match)
            if key not in lookup:
                lookup[key] = idx
        except (IndexError, KeyError, ValueError):
            continue
    
    # Cache for future use
    GalGA._param_lookup_cache = lookup
    return lookup


def match_row_to_model(GalGA, row: pd.Series, param_lookup: Optional[Dict] = None) -> Optional[int]:
    """
    Match a single row to a model index using parameter-based lookup.
    
    Parameters
    ----------
    GalGA : object
        Container with results
    row : pd.Series
        Row containing parameter values
    param_lookup : dict, optional
        Pre-built parameter lookup dictionary (will be built if not provided)
        
    Returns
    -------
    model_idx : int or None
        Matched model index, or None if no match found
    """
    if param_lookup is None:
        param_lookup = build_parameter_lookup(GalGA)
    
    if not param_lookup:
        return None
    
    # Try to extract all parameters from row
    try:
        key = tuple(
            _normalize_param_value(name, row[name])
            for name in _MATCH_PARAM_COLS
            if name in row.index and pd.notna(row[name])
        )
        
        # If we don't have all parameters, can't match
        if len(key) != len(_MATCH_PARAM_COLS):
            return None
        
        return param_lookup.get(key)
    
    except (KeyError, ValueError, TypeError):
        return None


def match_dataframe_to_models(
    GalGA,
    df: pd.DataFrame,
    inplace: bool = False,
    column: str = "model_idx",
    drop_missing: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Match all rows in a dataframe to model indices using fast parameter lookup.
    
    This is the PRIMARY matching function that should be used for all plotting.
    It ensures consistent matching across all visualization functions.
    
    Parameters
    ----------
    GalGA : object
        Container with results
    df : pd.DataFrame
        Dataframe containing parameter values
    inplace : bool, optional
        If True, modify df in place; otherwise return a copy
    column : str, optional
        Name of column to store matched indices (default: "model_idx")
    drop_missing : bool, optional
        If True, drop rows that couldn't be matched
    verbose : bool, optional
        If True, print matching statistics
        
    Returns
    -------
    df_matched : pd.DataFrame
        Dataframe with model_idx column added/updated
    """
    if df is None or df.empty:
        return df if inplace else pd.DataFrame()
    
    out = df if inplace else df.copy()
    
    # Build lookup once for all rows
    param_lookup = build_parameter_lookup(GalGA)
    
    if not param_lookup:
        if verbose:
            print("[WARNING] No parameter lookup available - cannot match rows")
        return out
    
    # Match each row
    matched_indices = []
    n_matched = 0
    
    for idx, row in out.iterrows():
        model_idx = match_row_to_model(GalGA, row, param_lookup)
        matched_indices.append(model_idx)
        if model_idx is not None:
            n_matched += 1
    
    out[column] = matched_indices
    
    if verbose:
        n_total = len(out)
        pct = 100.0 * n_matched / n_total if n_total > 0 else 0.0
        print(f"[MATCHING] Matched {n_matched}/{n_total} rows ({pct:.1f}%)")
    
    # Drop unmatched rows if requested
    if drop_missing:
        mask = pd.notna(out[column])
        n_dropped = (~mask).sum()
        out = out.loc[mask].copy()
        out.reset_index(drop=True, inplace=True)
        out[column] = out[column].astype(int)
        if verbose and n_dropped > 0:
            print(f"[MATCHING] Dropped {n_dropped} unmatched rows")
    
    return out


def get_matched_posterior_samples(
    GalGA,
    results_df: pd.DataFrame,
    fitness_col: str = 'fitness',
    percentile: float = 10,
    verbose: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """
    Get posterior samples with matched model indices and weights.
    
    This is the UNIFIED function for getting posterior samples that should be
    used by all plotting functions. It ensures:
    1. Consistent percentile selection
    2. Consistent weight calculation
    3. Consistent matching to model indices
    
    Parameters
    ----------
    GalGA : object
        Container with results
    results_df : pd.DataFrame
        Results dataframe with parameters and fitness
    fitness_col : str, optional
        Column name for fitness metric (lower is better)
    percentile : float, optional
        Top X% of models to include
    verbose : bool, optional
        If True, print statistics
        
    Returns
    -------
    matched_df : pd.DataFrame or None
        Dataframe with matched model indices
    weights : np.ndarray or None
        Normalized weights (sum to 1)
    """
    if results_df is None or results_df.empty:
        if verbose:
            print("[WARNING] Empty results dataframe")
        return None, None
    
    # Sort by fitness (lower is better)
    df_sorted = results_df.sort_values(fitness_col, ascending=True).copy()
    
    # Select top percentile
    n_top = max(1, int(len(df_sorted) * percentile / 100))
    top_df = df_sorted.head(n_top).reset_index(drop=True)
    
    if verbose:
        print(f"[POSTERIOR] Selected top {n_top} models ({percentile}% of {len(results_df)})")
        print(f"[POSTERIOR] Fitness range: {top_df[fitness_col].min():.6f} to {top_df[fitness_col].max():.6f}")
    
    # Compute inverse-fitness weights
    fit = top_df[fitness_col].values
    eps = np.min(fit) * 0.001 if np.min(fit) > 0 else 1e-10
    w_raw = 1.0 / (fit + eps)
    weights = w_raw / np.sum(w_raw)
    
    # Match to model indices
    matched_df = match_dataframe_to_models(
        GalGA,
        top_df,
        inplace=False,
        column="model_idx",
        drop_missing=True,
        verbose=verbose
    )
    
    if matched_df is None or matched_df.empty:
        if verbose:
            print("[WARNING] No models could be matched")
        return None, None
    
    # Update weights to match only successfully matched rows
    if len(matched_df) < len(top_df):
        # Some rows were dropped - need to renormalize weights
        matched_indices = matched_df.index.values
        weights = weights[matched_indices]
        weights = weights / np.sum(weights)
        if verbose:
            print(f"[POSTERIOR] After matching: {len(matched_df)} models with renormalized weights")
    
    return matched_df, weights


def get_best_model_matched(
    GalGA,
    results_df: pd.DataFrame,
    fitness_col: str = 'fitness',
    verbose: bool = True
) -> Tuple[Optional[int], Optional[pd.Series]]:
    """
    Get the single best model with matched model index.
    
    This is the UNIFIED function for getting the best model that should be
    used by all single-model plotting functions.
    
    Parameters
    ----------
    GalGA : object
        Container with results
    results_df : pd.DataFrame
        Results dataframe with parameters and fitness
    fitness_col : str, optional
        Column name for fitness metric (lower is better)
    verbose : bool, optional
        If True, print statistics
        
    Returns
    -------
    model_idx : int or None
        Matched model index in GalGA.results
    best_row : pd.Series or None
        Best model row from results_df
    """
    if results_df is None or results_df.empty:
        if verbose:
            print("[WARNING] Empty results dataframe")
        return None, None
    
    # Find best by fitness
    best_row_idx = results_df[fitness_col].idxmin()
    best_row = results_df.loc[best_row_idx]
    
    if verbose:
        print(f"[BEST MODEL] Row index: {best_row_idx}, Fitness: {best_row[fitness_col]:.6f}")
    
    # Match to model index
    param_lookup = build_parameter_lookup(GalGA)
    model_idx = match_row_to_model(GalGA, best_row, param_lookup)
    
    if model_idx is None:
        if verbose:
            print("[WARNING] Best model could not be matched to GalGA.results")
        return None, best_row
    
    if verbose:
        print(f"[BEST MODEL] Matched to model index: {model_idx}")
    
    return model_idx, best_row


def validate_matching(GalGA, results_df: pd.DataFrame, verbose: bool = True) -> Dict:
    """
    Validate matching quality and return diagnostic statistics.
    
    Parameters
    ----------
    GalGA : object
        Container with results
    results_df : pd.DataFrame
        Results dataframe
    verbose : bool, optional
        If True, print detailed diagnostics
        
    Returns
    -------
    stats : dict
        Dictionary containing matching statistics
    """
    stats = {
        'n_results': len(getattr(GalGA, 'results', [])),
        'n_df_rows': len(results_df) if results_df is not None else 0,
        'n_matched': 0,
        'match_rate': 0.0,
        'n_unique_matches': 0,
    }
    
    if results_df is None or results_df.empty:
        return stats
    
    # Build lookup
    param_lookup = build_parameter_lookup(GalGA)
    stats['n_lookup_entries'] = len(param_lookup)
    
    # Match all rows
    matched_df = match_dataframe_to_models(
        GalGA, results_df, inplace=False, verbose=False
    )
    
    if matched_df is not None and not matched_df.empty:
        matched_mask = pd.notna(matched_df['model_idx'])
        stats['n_matched'] = matched_mask.sum()
        stats['match_rate'] = 100.0 * stats['n_matched'] / len(matched_df)
        
        if stats['n_matched'] > 0:
            stats['n_unique_matches'] = matched_df.loc[matched_mask, 'model_idx'].nunique()
    
    if verbose:
        print("\n" + "="*60)
        print("MATCHING VALIDATION REPORT")
        print("="*60)
        print(f"GalGA.results entries:     {stats['n_results']}")
        print(f"Results dataframe rows:    {stats['n_df_rows']}")
        print(f"Parameter lookup entries:  {stats.get('n_lookup_entries', 0)}")
        print(f"Successfully matched:      {stats['n_matched']} ({stats['match_rate']:.1f}%)")
        print(f"Unique model indices:      {stats['n_unique_matches']}")
        print("="*60 + "\n")
    
    return stats
