#!/usr/bin/env python3
"""
Posterior statistics utilities for GCE model ensemble analysis.

This module provides functions to extract posterior distributions from
MCMC/GA samples and compute weighted statistics (median, percentiles)
for visualization with uncertainty bands.

Authors: N Miller
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def get_weighted_posterior_samples(results_df, fitness_col='fitness', percentile=10):
    """
    Extract top percentile of models with inverse-fitness weights.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with model parameters and fitness values
    fitness_col : str
        Column name containing fitness metric (lower is better)
    percentile : float
        Top X% of models to include (default 10)
    
    Returns
    -------
    top_df : pd.DataFrame
        Top percentile of models sorted by fitness
    weights : np.ndarray
        Normalized weights (sum to 1), with lower fitness → higher weight
    """
    if results_df is None or results_df.empty:
        return None, None
    
    # Sort by fitness (lower is better)
    df_sorted = results_df.sort_values(fitness_col, ascending=True).copy()
    
    # Select top percentile
    n_top = max(1, int(len(df_sorted) * percentile / 100))
    top_df = df_sorted.head(n_top).reset_index(drop=True)
    
    # Compute inverse-fitness weights
    fit = top_df[fitness_col].values
    eps = np.min(fit) * 0.001 if np.min(fit) > 0 else 1e-10
    w_raw = 1.0 / (fit + eps)
    weights = w_raw / np.sum(w_raw)
    
    return top_df, weights



def find_model_by_params(GalGA, params_row, param_cols=['sigma_2', 't_2', 'infall_2'], tol=1e-5):
    """
    Find model index in GalGA.results that matches parameter row.
    
    Parameters
    ----------
    GalGA : object
        GalGA object with results attribute
    params_row : pd.Series or dict
        Row from results_df with parameter values
    param_cols : list
        Parameter columns to match (default: sigma_2, t_2, infall_2)
    tol : float
        Tolerance for floating-point comparison
    
    Returns
    -------
    idx : int or None
        Index in GalGA.results that matches, or None if not found
    """
    if not hasattr(GalGA, 'results') or len(GalGA.results) == 0:
        return None
    
    # Parameter indices in GalGA.results array
    # Based on code analysis: [comp_idx, imf_idx, sn1a_idx, sy_idx, sn1ar_idx,
    #                           sigma_2, t_1, t_2, infall_1, infall_2, ...]
    param_indices = {'sigma_2': 5, 't_2': 7, 'infall_2': 9}
    
    target_vals = [float(params_row[col]) for col in param_cols]
    
    for idx, result in enumerate(GalGA.results):
        result_vals = [float(result[param_indices[col]]) for col in param_cols]
        if all(abs(t - r) < tol for t, r in zip(target_vals, result_vals)):
            return idx
    
    return None


def interpolate_to_common_grid(x_arrays, y_arrays, x_common, method='linear'):
    """
    Interpolate multiple (x, y) curves to a common x grid.
    
    Parameters
    ----------
    x_arrays : list of np.ndarray
        List of x-coordinate arrays (may have different lengths/grids)
    y_arrays : list of np.ndarray
        List of corresponding y-coordinate arrays
    x_common : np.ndarray
        Common x grid for interpolation
    method : str
        Interpolation method ('linear', 'cubic', etc.)
    
    Returns
    -------
    y_interp : np.ndarray
        Array of shape (n_curves, len(x_common)) with interpolated y values
        NaN where extrapolation would be required
    """
    n_curves = len(x_arrays)
    n_points = len(x_common)
    y_interp = np.full((n_curves, n_points), np.nan)
    
    for i, (x, y) in enumerate(zip(x_arrays, y_arrays)):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Remove NaN and sort by x
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            continue
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Sort by x
        sort_idx = np.argsort(x_clean)
        x_sorted = x_clean[sort_idx]
        y_sorted = y_clean[sort_idx]
        
        # Remove duplicate x values (keep first)
        unique_mask = np.ones(len(x_sorted), dtype=bool)
        unique_mask[1:] = np.diff(x_sorted) > 1e-12
        x_unique = x_sorted[unique_mask]
        y_unique = y_sorted[unique_mask]
        
        if len(x_unique) < 2:
            continue
        
        # Interpolate
        try:
            f = interp1d(x_unique, y_unique, kind=method, 
                        bounds_error=False, fill_value=np.nan)
            y_interp[i, :] = f(x_common)
        except Exception as e:
            print(f"Warning: Interpolation failed for curve {i}: {e}")
            continue
    
    return y_interp


def compute_percentile_bands(y_samples, weights, percentiles=[16, 50, 84]):
    """
    Compute weighted percentile bands across samples.
    
    Parameters
    ----------
    y_samples : np.ndarray
        Array of shape (n_samples, n_points) with y values from different models
    weights : np.ndarray
        Weights for each sample (length n_samples)
    percentiles : list
        Percentile levels (default [16, 50, 84] for median ± 1σ)
    
    Returns
    -------
    bands : dict
        Dictionary with keys 'lower', 'median', 'upper' (or custom percentile names)
        Each value is np.ndarray of length n_points
    """
    n_samples, n_points = y_samples.shape
    n_percentiles = len(percentiles)
    
    bands_array = np.full((n_percentiles, n_points), np.nan)
    
    for i in range(n_points):
        y_at_point = y_samples[:, i]
        
        # Filter out NaN values
        valid = np.isfinite(y_at_point)
        if np.sum(valid) == 0:
            continue
        
        y_valid = y_at_point[valid]
        w_valid = weights[valid]
        w_valid = w_valid / np.sum(w_valid)  # Renormalize
        
        # Compute weighted percentiles
        pcts = weighted_quantile(y_valid, np.array(percentiles) / 100.0, w_valid)
        bands_array[:, i] = pcts
    
    # Package into dictionary
    if len(percentiles) == 3 and percentiles == [16, 50, 84]:
        bands = {
            'lower': bands_array[0, :],
            'median': bands_array[1, :],
            'upper': bands_array[2, :]
        }
    else:
        bands = {f'p{p}': bands_array[i, :] for i, p in enumerate(percentiles)}
    
    return bands


def smooth_alpha_track_time_ordered(x_data, y_data, sigma=3):
    """
    Smooth alpha element tracks (from core_plots.py).
    
    Parameters
    ----------
    x_data : array-like
        X coordinates (e.g., [Fe/H])
    y_data : array-like
        Y coordinates (e.g., [α/Fe])
    sigma : float
        Gaussian smoothing kernel width
    
    Returns
    -------
    x_smooth, y_smooth : np.ndarray
        Smoothed coordinates
    """
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    x = np.asarray(x_data)[mask]
    y = np.asarray(y_data)[mask]
    
    if len(x) < 10:
        return x_data, y_data
    
    x_smooth = gaussian_filter1d(x, sigma=sigma, mode='nearest')
    y_smooth = gaussian_filter1d(y, sigma=sigma, mode='nearest')
    
    return x_smooth, y_smooth


def compute_mdf_ensemble(GalGA, top_df, weights, feh_range=(-2.0, 1.0), n_bins=100):
    """
    Compute MDF ensemble with median and percentile bands.
    
    Parameters
    ----------
    GalGA : object
        GalGA object with mdf_data and results
    top_df : pd.DataFrame
        Top percentile models
    weights : np.ndarray
        Model weights
    feh_range : tuple
        [Fe/H] range for common grid
    n_bins : int
        Number of bins in common grid
    
    Returns
    -------
    ensemble : dict
        Dictionary with 'x', 'median', 'lower', 'upper' arrays
    """
    if not hasattr(GalGA, 'mdf_data') or len(GalGA.mdf_data) == 0:
        return None
    
    # Define common [Fe/H] grid
    feh_common = np.linspace(feh_range[0], feh_range[1], n_bins)
    
    # Collect MDF samples
    mdf_x_arrays = []
    mdf_y_arrays = []
    
    for idx, row in top_df.iterrows():
        model_idx = find_model_by_params(GalGA, row)
        if model_idx is None:
            continue
        
        if model_idx < len(GalGA.mdf_data):
            mdf_x, mdf_y = GalGA.mdf_data[model_idx]
            mdf_x_arrays.append(np.asarray(mdf_x, dtype=float))
            mdf_y_arrays.append(np.asarray(mdf_y, dtype=float))
    
    if len(mdf_x_arrays) == 0:
        return None
    
    # Interpolate to common grid
    mdf_samples = interpolate_to_common_grid(mdf_x_arrays, mdf_y_arrays, 
                                             feh_common, method='linear')
    
    # Compute percentile bands
    bands = compute_percentile_bands(mdf_samples, weights[:len(mdf_samples)])
    
    ensemble = {
        'x': feh_common,
        'median': bands['median'],
        'lower': bands['lower'],
        'upper': bands['upper']
    }
    
    return ensemble


def compute_age_feh_ensemble(GalGA, top_df, weights, age_range=(0, 14.0), n_bins=200):
    """
    Compute Age-[Fe/H] ensemble with median and percentile bands.
    
    Parameters
    ----------
    GalGA : object
        GalGA object with age_data and results
    top_df : pd.DataFrame
        Top percentile models
    weights : np.ndarray
        Model weights
    age_range : tuple
        Age range in Gyr for common grid
    n_bins : int
        Number of bins in common grid
    
    Returns
    -------
    ensemble : dict
        Dictionary with 'x', 'median', 'lower', 'upper' arrays
    """
    if not hasattr(GalGA, 'age_data') or len(GalGA.age_data) == 0:
        return None
    
    # Define common age grid
    age_common = np.linspace(age_range[0], age_range[1], n_bins)
    
    # Collect age-[Fe/H] samples
    age_arrays = []
    feh_arrays = []
    
    for idx, row in top_df.iterrows():
        model_idx = find_model_by_params(GalGA, row)
        if model_idx is None:
            continue
        
        if model_idx < len(GalGA.age_data):
            time_array, feh_array = GalGA.age_data[model_idx]
            time_array = np.asarray(time_array, dtype=float)
            feh_array = np.asarray(feh_array, dtype=float)
            
            # Convert time to age (t_final - t) / 1e9
            age_gyr = (time_array[-1] - time_array) / 1e9
            
            age_arrays.append(age_gyr)
            feh_arrays.append(feh_array)
    
    if len(age_arrays) == 0:
        return None
    
    # Interpolate to common grid
    feh_samples = interpolate_to_common_grid(age_arrays, feh_arrays, 
                                             age_common, method='linear')
    
    # Compute percentile bands
    bands = compute_percentile_bands(feh_samples, weights[:len(feh_samples)])
    
    ensemble = {
        'x': age_common,
        'median': bands['median'],
        'lower': bands['lower'],
        'upper': bands['upper']
    }
    
    return ensemble


def compute_alpha_ensemble(GalGA, top_df, weights, element_idx, 
                          feh_range=(-2.0, 1.0), n_bins=150, smooth_sigma=3):
    """
    Compute [α/Fe] vs [Fe/H] ensemble for a specific element.
    
    Parameters
    ----------
    GalGA : object
        GalGA object with alpha_data and results
    top_df : pd.DataFrame
        Top percentile models
    weights : np.ndarray
        Model weights
    element_idx : int
        Element index (0=Mg, 1=Si, 2=Ca, 3=Ti)
    feh_range : tuple
        [Fe/H] range for common grid
    n_bins : int
        Number of bins in common grid
    smooth_sigma : float
        Gaussian smoothing kernel width
    
    Returns
    -------
    ensemble : dict
        Dictionary with 'x', 'median', 'lower', 'upper' arrays
    """
    if not hasattr(GalGA, 'alpha_data') or len(GalGA.alpha_data) == 0:
        return None
    
    # Define common [Fe/H] grid
    feh_common = np.linspace(feh_range[0], feh_range[1], n_bins)
    
    # Collect alpha element samples
    feh_arrays = []
    alpha_arrays = []
    
    for idx, row in top_df.iterrows():
        model_idx = find_model_by_params(GalGA, row)
        if model_idx is None:
            continue
        
        if model_idx < len(GalGA.alpha_data):
            alpha_arrs = GalGA.alpha_data[model_idx]
            
            if element_idx < len(alpha_arrs):
                feh_model, alpha_model = alpha_arrs[element_idx]
                
                # Apply smoothing (as in original code)
                feh_smooth, alpha_smooth = smooth_alpha_track_time_ordered(
                    feh_model, alpha_model, sigma=smooth_sigma)
                
                feh_arrays.append(feh_smooth)
                alpha_arrays.append(alpha_smooth)
    
    if len(feh_arrays) == 0:
        return None
    
    # Interpolate to common grid
    alpha_samples = interpolate_to_common_grid(feh_arrays, alpha_arrays, 
                                               feh_common, method='linear')
    
    # Compute percentile bands
    bands = compute_percentile_bands(alpha_samples, weights[:len(alpha_samples)])
    
    ensemble = {
        'x': feh_common,
        'median': bands['median'],
        'lower': bands['lower'],
        'upper': bands['upper']
    }
    
    return ensemble





# ----------------------------------------------------------------------------
# Weighting utilities
# ----------------------------------------------------------------------------

def _effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    s = np.sum(w)
    if s <= 0.0:
        return 0.0
    w = w / s
    return float((w.sum() ** 2) / (np.sum(np.square(w)) + 1e-300))


def _auto_temperature(residuals: np.ndarray) -> float:
    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad > 0:
        return float(mad)
    std = np.std(residuals)
    if std > 0:
        return float(std)
    return 1.0


def compute_weights(loss, temperature, floor,):
    """Turn a loss array into normalized weights.

    Parameters
    ----------
    loss:
        Iterable of fitness/loss values (lower is better).
    temperature:
        Optional temperature for the exponential weighting.  If ``None`` a
        robust scale (MAD) is used.
    floor:
        Minimum allowable temperature.

    Returns
    -------
    weights, temperature_used, ess
    """

    arr = np.asarray(loss, dtype=float)
    if arr.ndim != 1:
        raise ValueError("loss must be 1-D")
    finite = np.isfinite(arr)
    if np.count_nonzero(finite) < 3:
        raise ValueError("Not enough finite loss values to build a posterior")

    arr = arr.copy()
    arr[~finite] = np.nanmax(arr[finite])

    resid = arr - np.nanmin(arr)
    T = float(temperature) if temperature and temperature > 0 else _auto_temperature(resid)
    T = max(float(T), floor)

    weights = np.exp(-resid / T)
    weights[~finite] = 0.0
    s = np.sum(weights)
    if s <= 0:
        weights = np.ones_like(arr)
        s = np.sum(weights)
    weights /= s

    ess = _effective_sample_size(weights)
    return weights, T, ess




def _systematic_resample(weights, n):
    # low-variance resampling (a.k.a. systematic)
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    cdf = np.cumsum(weights)
    u0 = np.random.rand() / n
    u = u0 + (np.arange(n) / n)
    return np.searchsorted(cdf, u, side='right')



def posterior_resample(results_df, *,
                       weight_col=None,          # e.g. 'posterior_w' or 'sample_count'
                       fitness_col='fitness',    # fallback
                       percentile=None,          # optional fallback guard
                       n_draws=512,              # controls plot MC noise
                       resampling='systematic'): # or 'multinomial'

    if results_df is None or results_df.empty:
        raise ValueError("Empty results_df")

    df = results_df.copy()

    # --- 1) choose weights from the actual sampling, if present ---
    if weight_col and (weight_col in df.columns):
        w = np.asarray(df[weight_col], float)
        w = np.clip(w, 0, np.inf)
        if not np.isfinite(w).any() or w.sum() <= 0:
            raise ValueError(f"Non-positive weights in {weight_col}")
        w = w / w.sum()
    elif 'sample_count' in df.columns:
        # frequency of visits from DEMC/GA
        w = np.asarray(df['sample_count'], float)
        w = w / w.sum()
    else:
        # --- 2) defensible fallback: fitness-based, optionally with a cut ---
        df = df.sort_values(fitness_col, ascending=True)
        if percentile is not None:
            n_top = max(1, int(len(df) * (percentile / 100.0)))
            df = df.head(n_top).reset_index(drop=True)
        fit = np.asarray(df[fitness_col], float)
        eps = (fit.min() * 1e-3) if fit.min() > 0 else 1e-10
        w = 1.0 / (fit + eps)
        w = w / w.sum()

    # --- 3) draw a fixed number of posterior samples ---
    idx = (_systematic_resample(w, n_draws)
           if resampling == 'systematic'
           else np.random.choice(len(df), size=n_draws, replace=True, p=w))

    # compress duplicates so heavy reconstructions happen once
    uniq, counts = np.unique(idx, return_counts=True)
    df_unique = df.iloc[uniq].reset_index(drop=True)
    weights_unique = counts.astype(float)
    weights_unique /= weights_unique.sum()
    return df_unique, weights_unique


# Supporting functions
def compute_weights(loss):
    arr = np.asarray(loss, dtype=float)
    resid = arr - np.min(arr)
    T = np.median(np.abs(resid - np.median(resid))) or 1.0
    weights = np.exp(-resid / T)
    weights /= np.sum(weights)
    return weights, T, 0.0

def weighted_quantile(values, quantiles, sample_weight):
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    sample_weight = sample_weight / np.sum(sample_weight)
    sorter = np.argsort(values)
    values_sorted = values[sorter]
    weights_sorted = sample_weight[sorter]
    cdf = np.cumsum(weights_sorted)
    quantiles_out = np.interp(quantiles, cdf, values_sorted)
    return quantiles_out


def get_posterior_samples_and_weights(results_df, metric_val='fitness'):
    """
    Unified function to get full samples and exponential weights (as used in corner plot).
    
    Sorts by loss, computes exponential weights on full set (no cutoff).
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe
    metric_val : str
        Loss column (default 'fitness')
    
    Returns
    -------
    df : pd.DataFrame
        Full sorted dataframe
    weights : np.ndarray
        Normalized exponential weights
    """
    if results_df is None or results_df.empty:
        return None, None
    
    df = results_df.sort_values(metric_val).reset_index(drop=True)
    loss = df[metric_val].values
    weights, _, _ = compute_weights(loss)
    return df, weights






def choose_cutoff_lognorm_mixture(in_weights, bins=100, kde_points=1024, em_max_iter=200, tol=1e-6, force_k2=False):
    """
    Simple, reviewer-proof cutoff:
      - Work in y = log(loss).
      - Fit K=1 and K=2 Gaussian mixtures in y by EM; pick K by BIC (unless force_k2=True).
      - If K=2: cutoff = equal-responsibility boundary where pi1*N1(y)=pi2*N2(y).
      - If K=1: no hard cut (use all models).
    Writes two plots and a small audit file; returns cutoff & realized keep fraction.
    """
    import os, numpy as np, matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde, norm
    from scipy.special import logsumexp

    # ---------- data ----------
    L = np.asarray(in_weights, float)
    L = L[np.isfinite(L)]
    if L.size == 0:
        raise RuntimeError("No finite losses/fitness values.")
    eps = 1e-12
    y = np.log(L + eps)
    N = y.size

    # ---------- helper: EM for 1D K-component Gaussian mixture ----------
    def em_gmm_1d(y, K, iters=200, tol=1e-6):
        # init by quantiles
        qs = np.linspace(0.2, 0.8, K)
        mu = np.quantile(y, qs) if K > 1 else np.array([float(np.mean(y))])
        s0 = float(np.std(y))
        s0 = s0 if s0 > 1e-6 else 0.1
        sig = np.full(K, s0, float)
        pi = np.full(K, 1.0 / K, float)

        c_norm = -0.5*np.log(2*np.pi)

        def logpdf(y, mu, sig):
            return c_norm - np.log(sig) - 0.5*((y - mu)/sig)**2

        prev_ll = -np.inf
        for _ in range(iters):
            # E-step: responsibilities (log-space)
            log_comp = np.stack([np.log(pi[k]) + logpdf(y, mu[k], sig[k] + 1e-12) for k in range(K)], axis=1)
            log_den = logsumexp(log_comp, axis=1, keepdims=True)
            R = np.exp(log_comp - log_den)  # N x K
            Nk = R.sum(axis=0) + 1e-12

            # M-step
            mu_new = (R * y[:, None]).sum(axis=0) / Nk
            sig_new = np.sqrt((R * (y[:, None] - mu_new[None, :])**2).sum(axis=0) / Nk)
            sig_new = np.maximum(sig_new, 1e-6)
            pi_new = Nk / N

            # log-likelihood
            ll = float(np.sum(log_den))
            if abs(ll - prev_ll) < tol:
                mu, sig, pi = mu_new, sig_new, pi_new
                prev_ll = ll
                break
            mu, sig, pi, prev_ll = mu_new, sig_new, pi_new, ll

        # BIC: p = (K-1) + K (means) + K (stds) = 2K - 1
        bic = -2.0*prev_ll + (2*K - 1)*np.log(N)
        # order by mean
        order = np.argsort(mu)
        return pi[order], mu[order], sig[order], prev_ll, bic

    # ---------- fit K=1 and K=2 ----------
    pi1, mu1, sg1, ll1, bic1 = em_gmm_1d(y, 1, em_max_iter, tol)
    pi2, mu2, sg2, ll2, bic2 = em_gmm_1d(y, 2, em_max_iter, tol)
    choose_K2 = force_k2 or (bic2 < bic1)

    # ---------- cutoff (if K=2), else None ----------
    loss_cutoff = None
    chosen_K = 2 if choose_K2 else 1
    if choose_K2:
        # components already ordered: comp0 is the elite (lower mu)
        pi = pi2; mu = mu2; sig = sg2

        # Solve pi0*N0(y) = pi1*N1(y) analytically
        A = 0.5*(1.0/sig[1]**2 - 1.0/sig[0]**2)
        B = (mu[0]/sig[0]**2 - mu[1]/sig[1]**2)
        D = 0.5*(mu[1]**2/sig[1]**2 - mu[0]**2/sig[0]**2)
        const = np.log((pi[1]/sig[1])/(pi[0]/sig[0]))
        C = D - const

        if abs(A) < 1e-12:
            y_cut = -C / (B + 1e-12)  # equal-variance fallback
        else:
            disc = max(B*B - 4*A*C, 0.0)
            roots = np.sort(( -B + np.array([-1.0, 1.0])*np.sqrt(disc) ) / (2*A))
            # prefer a root between the two means; otherwise, nearest to their midpoint
            mid = 0.5*(mu[0] + mu[1])
            if (mu[0] <= roots[0] <= mu[1]) or (mu[0] <= roots[1] <= mu[1]):
                y_cut = roots[0] if (mu[0] <= roots[0] <= mu[1]) else roots[1]
            else:
                y_cut = roots[np.argmin(np.abs(roots - mid))]

        loss_cutoff = float(np.exp(y_cut))
        frac = float(np.mean(L <= loss_cutoff))
    else:
        # no hard selection
        frac = 1.0

    pct = 100.0 * frac
    
    return pct



def sample_posterior_points(GalGA, top_df, weights, element_idx, n_points, feh_range=(-2,1)):
    """
    Draw ~n_points (Fe/H, alpha) from the posterior by:
    1) sampling a track index by weights,
    2) sampling a random point along that track within feh_range (uniform in Fe/H support).
    """
    rng = np.random.default_rng()
    # pre-extract tracks once
    tracks = []
    for _, row in top_df.iterrows():
        model_idx = find_model_by_params(GalGA, row)
        if model_idx is None or model_idx >= len(GalGA.alpha_data): 
            tracks.append(None); continue
        feh, a = GalGA.alpha_data[model_idx][element_idx]
        if feh is None or a is None: tracks.append(None); continue
        x, y = smooth_alpha_track_time_ordered(np.asarray(feh, float), np.asarray(a, float), sigma=3)
        m = np.isfinite(x) & np.isfinite(y) & (x >= feh_range[0]) & (x <= feh_range[1])
        if m.sum() >= 5:
            # monotonic-in-x subsampling to avoid weird backtracks
            order = np.argsort(x[m])
            tracks.append((x[m][order], y[m][order]))
        else:
            tracks.append(None)

    # normalize weights over valid tracks only
    valid = np.array([t is not None for t in tracks])
    if valid.sum() == 0:
        return np.empty((0,)), np.empty((0,))
    w = np.array(weights, float)
    w[~valid] = 0.0
    w = w / (w.sum() + 1e-300)

    # draw indices and then uniform points in each chosen track's Fe/H domain
    idx = rng.choice(len(tracks), size=n_points, replace=True, p=w)
    xs = np.empty(n_points); ys = np.empty(n_points)
    for i, j in enumerate(idx):
        x, y = tracks[j]
        # pick a random Fe/H bin, then optionally jitter inside the bin via linear interp
        k = rng.integers(0, len(x)-1)
        x0, x1 = x[k], x[k+1]
        t = rng.random()
        xs[i] = (1-t)*x0 + t*x1
        # local linear interpolation
        y0, y1 = y[k], y[k+1]
        ys[i] = (1-t)*y0 + t*y1
    return xs, ys
