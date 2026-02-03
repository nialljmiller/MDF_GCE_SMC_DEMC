"""
Loss functions for MDF_GCE_SMC_DEMC.

This module contains all loss/fitness functions for:
- MDF (Metallicity Distribution Function) fitting
- Age-metallicity relation fitting
- Combined loss computation
- Various distance metrics (WRMSE, MAE, KS, Huber, etc.)
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any

# Small epsilon to avoid division by zero
EPS = 1e-12


# =============================================================================
# MDF LOSS FUNCTIONS
# =============================================================================

def compute_ks_distance(observed: np.ndarray, predicted: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov distance between two distributions.
    
    Parameters
    ----------
    observed : np.ndarray
        Observed distribution (histogram counts)
    predicted : np.ndarray
        Predicted distribution (histogram counts)
        
    Returns
    -------
    float
        KS distance (max absolute difference of CDFs)
    """
    model_cdf = np.cumsum(predicted)
    model_cdf /= model_cdf[-1] + EPS
    
    data_cdf = np.cumsum(observed)
    data_cdf /= data_cdf[-1] + EPS
    
    return float(np.max(np.abs(model_cdf - data_cdf)))


def compute_wrmse(
    observed: np.ndarray,
    predicted: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    Weighted Root Mean Square Error.
    
    Parameters
    ----------
    observed : np.ndarray
        Observed values
    predicted : np.ndarray
        Predicted values
    sigma : np.ndarray
        Uncertainties (weights = 1/sigma)
        
    Returns
    -------
    float
        WRMSE value
    """
    return float(np.sqrt(np.mean(((predicted - observed) / (sigma + EPS)) ** 2)))


def compute_mae(observed: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Error.
    
    Parameters
    ----------
    observed : np.ndarray
        Observed values
    predicted : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        MAE value
    """
    return float(np.mean(np.abs(predicted - observed)))


def compute_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """
    Root Mean Square Error.
    
    Parameters
    ----------
    observed : np.ndarray
        Observed values
    predicted : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        RMSE value
    """
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def compute_mape(observed: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    
    Parameters
    ----------
    observed : np.ndarray
        Observed values
    predicted : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        MAPE value
    """
    return float(np.mean(np.abs((predicted - observed) / (observed + EPS))))


def compute_huber_loss(
    observed: np.ndarray,
    predicted: np.ndarray,
    delta: float = 1.0,
) -> float:
    """
    Huber loss (robust to outliers).
    
    Parameters
    ----------
    observed : np.ndarray
        Observed values
    predicted : np.ndarray
        Predicted values
    delta : float
        Threshold for switching between quadratic and linear loss
        
    Returns
    -------
    float
        Huber loss value
    """
    error = predicted - observed
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return float(np.where(is_small_error, squared_loss, linear_loss).mean())


def compute_cosine_similarity(observed: np.ndarray, predicted: np.ndarray) -> float:
    """
    Cosine similarity between distributions.
    
    Parameters
    ----------
    observed : np.ndarray
        Observed values
    predicted : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        Cosine similarity (0-1, higher is more similar)
    """
    dot = np.dot(observed, predicted)
    norm_obs = np.linalg.norm(observed)
    norm_pred = np.linalg.norm(predicted)
    
    if norm_obs < EPS or norm_pred < EPS:
        return 0.0
    
    return float(dot / (norm_obs * norm_pred))


def compute_log_cosh_loss(observed: np.ndarray, predicted: np.ndarray) -> float:
    """
    Log-cosh loss (smooth approximation to MAE).
    
    Parameters
    ----------
    observed : np.ndarray
        Observed values
    predicted : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        Log-cosh loss value
    """
    error = predicted - observed
    return float(np.mean(np.log(np.cosh(error + EPS))))


def compute_emd(observed: np.ndarray, predicted: np.ndarray) -> float:
    """
    Earth Mover's Distance (1D Wasserstein distance).
    
    Parameters
    ----------
    observed : np.ndarray
        Observed distribution
    predicted : np.ndarray
        Predicted distribution
        
    Returns
    -------
    float
        EMD value
    """
    # Normalize to probability distributions
    obs_norm = observed / (np.sum(observed) + EPS)
    pred_norm = predicted / (np.sum(predicted) + EPS)
    
    # Compute CDFs
    obs_cdf = np.cumsum(obs_norm)
    pred_cdf = np.cumsum(pred_norm)
    
    # EMD is integral of |CDF difference|
    return float(np.sum(np.abs(obs_cdf - pred_cdf)))


def compute_ensemble_loss(
    observed: np.ndarray,
    predicted: np.ndarray,
    sigma: np.ndarray,
    weights: Tuple[float, float, float] = (0.7, 0.2, 0.1),
) -> float:
    """
    Ensemble loss combining WRMSE, cosine similarity, and Huber loss.
    
    This is the primary fitness function used in the paper:
    L_ensemble = 0.7 * WRMSE + 0.2 * (1 - cosine) + 0.1 * Huber
    
    Parameters
    ----------
    observed : np.ndarray
        Observed MDF
    predicted : np.ndarray
        Predicted MDF
    sigma : np.ndarray
        Uncertainties
    weights : tuple of float
        Weights for (WRMSE, cosine_loss, Huber)
        
    Returns
    -------
    float
        Ensemble loss value
    """
    w_wrmse, w_cosine, w_huber = weights
    
    wrmse = compute_wrmse(observed, predicted, sigma)
    cosine_loss = 1.0 - compute_cosine_similarity(observed, predicted)
    huber = compute_huber_loss(observed, predicted, delta=1.0)
    
    return float(w_wrmse * wrmse + w_cosine * cosine_loss + w_huber * huber)


# =============================================================================
# ALL METRICS COMPUTATION
# =============================================================================

def calculate_all_mdf_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    sigma: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate all MDF loss metrics at once.
    
    Parameters
    ----------
    observed : np.ndarray
        Observed MDF
    predicted : np.ndarray
        Predicted MDF
    sigma : np.ndarray, optional
        Uncertainties. If None, uses uniform weights.
        
    Returns
    -------
    dict
        Dictionary mapping metric names to values
    """
    if sigma is None:
        sigma = np.ones_like(observed)
    
    return {
        "ks": compute_ks_distance(observed, predicted),
        "wrmse": compute_wrmse(observed, predicted, sigma),
        "mae": compute_mae(observed, predicted),
        "rmse": compute_rmse(observed, predicted),
        "mape": compute_mape(observed, predicted),
        "huber": compute_huber_loss(observed, predicted),
        "cosine": compute_cosine_similarity(observed, predicted),
        "log_cosh": compute_log_cosh_loss(observed, predicted),
        "emd": compute_emd(observed, predicted),
        "ensemble": compute_ensemble_loss(observed, predicted, sigma),
    }


# =============================================================================
# AGE-METALLICITY LOSS FUNCTIONS
# =============================================================================

def compute_age_metallicity_loss(
    model_ages: np.ndarray,
    model_feh: np.ndarray,
    obs_ages: np.ndarray,
    obs_feh: np.ndarray,
    obs_errors: Optional[np.ndarray] = None,
    metric: str = "rmse",
    n_bins: int = 10,
) -> float:
    """
    Compute loss between model and observed age-metallicity relation.
    
    Uses binned comparison to handle different sampling densities.
    
    Parameters
    ----------
    model_ages : np.ndarray
        Model ages (in Gyr)
    model_feh : np.ndarray
        Model [Fe/H] values
    obs_ages : np.ndarray
        Observed stellar ages (in Gyr)
    obs_feh : np.ndarray
        Observed [Fe/H] values
    obs_errors : np.ndarray, optional
        Observational uncertainties on [Fe/H]
    metric : str
        Loss metric: 'mae', 'rmse', 'weighted_mae', 'weighted_rmse'
    n_bins : int
        Number of age bins
        
    Returns
    -------
    float
        Loss value
    """
    # Clean data
    valid_obs = np.isfinite(obs_ages) & np.isfinite(obs_feh)
    clean_ages = obs_ages[valid_obs]
    clean_feh = obs_feh[valid_obs]
    
    if obs_errors is not None:
        clean_errors = obs_errors[valid_obs]
    else:
        clean_errors = np.ones_like(clean_feh) * 0.1
    
    if len(clean_ages) < 5:
        return 1000.0
    
    # Create age bins
    age_min, age_max = clean_ages.min(), clean_ages.max()
    age_bins = np.linspace(age_min, age_max, n_bins + 1)
    bin_centers = 0.5 * (age_bins[:-1] + age_bins[1:])
    
    # Compute binned statistics
    bin_means = np.zeros(n_bins)
    bin_stds = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        mask = (clean_ages >= age_bins[i]) & (clean_ages < age_bins[i + 1])
        if i == n_bins - 1:
            mask = (clean_ages >= age_bins[i]) & (clean_ages <= age_bins[i + 1])
        
        if mask.sum() >= 3:
            bin_means[i] = np.mean(clean_feh[mask])
            bin_stds[i] = np.std(clean_feh[mask])
            bin_counts[i] = mask.sum()
    
    # Valid bins have at least 3 points
    valid_bins = bin_counts >= 3
    if valid_bins.sum() < 3:
        return 1000.0
    
    valid_centers = bin_centers[valid_bins]
    valid_means = bin_means[valid_bins]
    valid_stds = bin_stds[valid_bins]
    valid_stds = np.maximum(valid_stds, 0.05)  # Minimum uncertainty
    
    # Interpolate model to bin centers
    model_age_gyr = np.asarray(model_ages)
    model_feh_arr = np.asarray(model_feh)
    
    # Remove duplicates for interpolation
    unique_mask = np.concatenate([[True], np.diff(model_age_gyr) != 0])
    model_age_unique = model_age_gyr[unique_mask]
    model_feh_unique = model_feh_arr[unique_mask]
    
    model_interp = np.interp(valid_centers, model_age_unique, model_feh_unique)
    
    # Compute loss
    if metric == "mae":
        return float(np.mean(np.abs(model_interp - valid_means)))
    elif metric in ("rmse", "rms"):
        return float(np.sqrt(np.mean((model_interp - valid_means) ** 2)))
    elif metric == "weighted_mae":
        weights = 1.0 / valid_stds
        return float(np.average(np.abs(model_interp - valid_means), weights=weights))
    elif metric == "weighted_rmse":
        weights = 1.0 / valid_stds
        return float(np.sqrt(np.average((model_interp - valid_means) ** 2, weights=weights)))
    else:
        # Default to RMSE
        return float(np.sqrt(np.mean((model_interp - valid_means) ** 2)))


# =============================================================================
# COMBINED LOSS
# =============================================================================

def compute_combined_loss(
    mdf_loss: float,
    age_loss: float,
    mdf_weight: float = 0.8,
) -> float:
    """
    Combine MDF and age-metallicity losses.
    
    Parameters
    ----------
    mdf_loss : float
        MDF loss value
    age_loss : float
        Age-metallicity loss value
    mdf_weight : float
        Weight for MDF loss (age_weight = 1 - mdf_weight)
        
    Returns
    -------
    float
        Combined loss
    """
    mdf_loss = max(mdf_loss, 1e-6)
    age_loss = max(age_loss, 1e-6)
    
    return mdf_weight * mdf_loss + (1.0 - mdf_weight) * age_loss


# =============================================================================
# LEGACY INTERFACE (for backward compatibility)
# =============================================================================

def compute_mdf_loss(GA_class, theory_count_array: np.ndarray) -> float:
    """
    Compute MDF loss using GA class attributes.
    
    Legacy interface for backward compatibility.
    
    Parameters
    ----------
    GA_class : GalacticEvolutionGA
        GA class instance with normalized_count and placeholder_sigma_array
    theory_count_array : np.ndarray
        Model MDF
        
    Returns
    -------
    float
        Ensemble loss value
    """
    return compute_ensemble_loss(
        GA_class.normalized_count,
        theory_count_array,
        GA_class.placeholder_sigma_array,
    )


def calculate_all_metrics(
    model_ages: np.ndarray,
    model_feh: np.ndarray,
    obs_ages: np.ndarray,
    obs_feh: np.ndarray,
    obs_errors: Optional[np.ndarray],
    dataset_name: str,
) -> Dict[str, float]:
    """
    Calculate all age-metallicity metrics.
    
    Legacy interface for backward compatibility.
    """
    metrics = {}
    
    for metric in ["mae", "rmse", "weighted_mae", "weighted_rmse"]:
        try:
            metrics[metric] = compute_age_metallicity_loss(
                model_ages, model_feh,
                obs_ages, obs_feh, obs_errors,
                metric=metric,
            )
        except Exception:
            metrics[metric] = 1000.0
    
    return metrics
