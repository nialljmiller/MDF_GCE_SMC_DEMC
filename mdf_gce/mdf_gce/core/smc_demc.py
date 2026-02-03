"""
Sequential Monte Carlo with Differential Evolution Markov Chain Monte Carlo.

This module implements:
- DE-MC moves (ter Braak scheme)
- Tempered SMC-DEMC for posterior sampling
- Utility functions for resampling and ESS computation

The sampler can be used:
1. Inside the GA run for hybrid DE-MC/GA exploration
2. As post-GA refinement to convert the final ensemble into posterior samples
"""

import os
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Bound:
    """Parameter bounds for continuous parameters."""
    lo: float
    hi: float
    
    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError(f"Lower bound ({self.lo}) > upper bound ({self.hi})")
    
    @property
    def range(self) -> float:
        return self.hi - self.lo


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def reflect_to_bounds(x: np.ndarray, bounds: List[Bound]) -> np.ndarray:
    """
    Reflect parameter values to stay within bounds.
    
    Uses periodic boundary reflection to preserve perturbation magnitude.
    
    Parameters
    ----------
    x : np.ndarray
        Parameter vector
    bounds : list of Bound
        Bounds for each parameter
        
    Returns
    -------
    np.ndarray
        Reflected parameter vector
    """
    y = x.copy()
    for j, b in enumerate(bounds):
        lo, hi = b.lo, b.hi
        L = hi - lo
        if L <= 0:
            continue
        # Reflect using modular arithmetic
        t = (y[j] - lo) % (2 * L)
        y[j] = lo + (t if t <= L else 2 * L - t)
    return y


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Compute effective sample size from importance weights.
    
    ESS = (sum w)^2 / sum(w^2)
    
    Parameters
    ----------
    weights : np.ndarray
        Importance weights (need not be normalized)
        
    Returns
    -------
    float
        Effective sample size
    """
    s = weights.sum()
    return s * s / np.dot(weights, weights)


def systematic_resample(
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Systematic resampling from weighted particles.
    
    Parameters
    ----------
    weights : np.ndarray
        Normalized importance weights
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    np.ndarray
        Indices of resampled particles
    """
    N = len(weights)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    idx = np.searchsorted(cumsum, positions, side='right')
    return idx


def choose_next_beta(
    loss: np.ndarray,
    beta_prev: float,
    target_ess_frac: float = 0.6,
) -> float:
    """
    Choose next inverse temperature to maintain target ESS.
    
    Uses binary search to find the largest beta increment that keeps
    ESS above target_ess_frac * N.
    
    Parameters
    ----------
    loss : np.ndarray
        Current loss values for all particles
    beta_prev : float
        Current inverse temperature
    target_ess_frac : float
        Target ESS as fraction of N
        
    Returns
    -------
    float
        Next inverse temperature (in [beta_prev, 1])
    """
    N = len(loss)
    lo, hi = 1e-6, max(1e-6, 1.0 - beta_prev)
    target = target_ess_frac * N
    
    # Binary search for optimal increment
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        w = np.exp(-mid * loss)
        ess = effective_sample_size(w)
        if ess < target:
            hi = mid
        else:
            lo = mid
    
    return min(1.0, beta_prev + lo)


# =============================================================================
# DE-MC MOVES
# =============================================================================

def de_mh_move(
    X: np.ndarray,
    loglike: Callable[[np.ndarray, object], float],
    bounds: List[Bound],
    metadata: Optional[np.ndarray] = None,
    steps: int = 2,
    gamma: Optional[float] = None,
    jitter: float = 1e-9,
    rng: Optional[np.random.Generator] = None,
    max_workers: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Differential Evolution Metropolis-Hastings moves on an ensemble.
    
    Implements the ter Braak DE-MC scheme: each walker proposes a new
    position using the scaled difference of two peers plus Gaussian jitter.
    
    Parameters
    ----------
    X : np.ndarray
        Current ensemble positions, shape (N, d)
    loglike : callable
        Log-likelihood function: (theta, metadata) -> float
    bounds : list of Bound
        Parameter bounds for reflection
    metadata : np.ndarray, optional
        Ancillary data for each particle
    steps : int
        Number of DE-MC sweeps
    gamma : float, optional
        Scaling factor. Default: 2.38 / sqrt(2*d)
    jitter : float
        Gaussian noise scale for proposals
    rng : np.random.Generator, optional
        Random number generator
    max_workers : int, optional
        Number of parallel workers
        
    Returns
    -------
    X_new : np.ndarray
        Updated ensemble positions
    accepted : np.ndarray
        Boolean mask of accepted moves
    """
    if rng is None:
        rng = np.random.default_rng()
    
    N, d = X.shape
    if gamma is None:
        gamma = 2.38 / np.sqrt(2 * d)  # ter Braak default
    
    accepted = np.zeros(N, dtype=bool)
    
    # Setup parallel evaluation
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = max(1, int(max_workers))
    use_threads = max_workers > 1
    pool = ThreadPool(processes=max_workers) if use_threads else None
    batch_size = max_workers if use_threads else 1
    
    if metadata is not None:
        meta_array = np.asarray(metadata, dtype=object)
    else:
        meta_array = None
    
    def _loglike(idx: int, theta: np.ndarray) -> float:
        if meta_array is None:
            return loglike(theta, None)
        return loglike(theta, meta_array[idx])
    
    def _eval_proposal(args):
        idx, theta = args
        return _loglike(idx, theta)
    
    # Initial log-likelihoods
    L = np.array([_loglike(i, X[i]) for i in range(N)], dtype=float)
    
    try:
        for _ in range(steps):
            order = rng.permutation(N)
            for start in range(0, N, batch_size):
                batch = order[start:start + batch_size]
                proposals = []
                eval_args = []
                
                for i in batch:
                    # Pick two distinct other indices
                    js = list(range(N))
                    js.remove(i)
                    r1, r2 = rng.choice(js, size=2, replace=False)
                    
                    # DE proposal
                    prop = X[i] + gamma * (X[r1] - X[r2]) + rng.normal(scale=jitter, size=d)
                    prop = reflect_to_bounds(prop, bounds)
                    
                    proposals.append((i, prop))
                    eval_args.append((i, prop.copy()))
                
                # Evaluate proposals
                if pool is not None:
                    L_news = pool.map(_eval_proposal, eval_args)
                else:
                    L_news = [_eval_proposal(arg) for arg in eval_args]
                
                # Metropolis accept/reject
                for (i, prop), L_new in zip(proposals, L_news):
                    if np.log(rng.random()) < (L_new - L[i]):
                        X[i] = prop
                        L[i] = L_new
                        accepted[i] = True
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    
    return X, accepted


# =============================================================================
# SMC-DEMC SAMPLER
# =============================================================================

def run_smc_demc(
    X0: np.ndarray,
    loss_fn: Callable[[np.ndarray, object], float],
    bounds: List[Bound],
    metadata0: Optional[np.ndarray] = None,
    ess_trigger: float = 0.6,
    moves_per_stage: int = 3,
    rng: Optional[np.random.Generator] = None,
    gamma_schedule: Tuple[Optional[float], float] = (None, 1.0),
    big_step_every: int = 6,
    max_workers: Optional[int] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Tempered Sequential Monte Carlo with DE-MC mutation moves.
    
    Anneals from a prior (beta=0) to the posterior (beta=1) using
    adaptive temperature selection and DE-MC moves at each stage.
    
    This function is used:
    1. Inside the GA for hybrid DE-MC exploration
    2. As the dedicated posterior refinement stage after GA convergence
    
    Parameters
    ----------
    X0 : np.ndarray
        Initial ensemble from GA, shape (N, d)
    loss_fn : callable
        Loss function: (theta, metadata) -> scalar loss (lower is better)
    bounds : list of Bound
        Parameter bounds
    metadata0 : np.ndarray, optional
        Ancillary data for each particle (e.g., categorical indices)
    ess_trigger : float
        Resample when ESS/N < ess_trigger
    moves_per_stage : int
        Number of DE-MC moves per temperature stage
    rng : np.random.Generator, optional
        Random number generator
    gamma_schedule : tuple
        (default_gamma, big_gamma) - use big_gamma every big_step_every stages
    big_step_every : int
        Frequency of big gamma steps
    max_workers : int, optional
        Parallel workers for evaluation
        
    Returns
    -------
    ensemble : np.ndarray
        Final ensemble positions, shape (N, d)
    chains_df : pd.DataFrame
        Full chain log with columns: stage, pid, accepted, [metadata], [params]
    """
    if rng is None:
        rng = np.random.default_rng()
    
    N, d = X0.shape
    
    # Handle metadata
    if metadata0 is not None:
        metadata = np.asarray(metadata0, dtype=object)
        if metadata.ndim == 1:
            metadata = metadata[:, None]
        if metadata.shape[0] != N:
            raise ValueError("metadata0 must have the same length as the ensemble")
    else:
        metadata = None
    
    # Convert loss to log-likelihood: loglike = -loss
    def loglike(theta, meta=None):
        return -float(loss_fn(theta, meta))
    
    # Initialize state
    X = X0.copy()
    beta = 0.0
    stage = 0
    chains = []
    weights = np.ones(N) / N
    
    # Helper to recompute weights for a beta jump
    def reweight(beta_prev, beta_new):
        nonlocal weights
        delta = beta_new - beta_prev
        loss = np.array([
            loss_fn(X[i], None if metadata is None else metadata[i])
            for i in range(N)
        ], dtype=float)
        u = np.exp(-delta * loss)
        w = weights * u
        w /= w.sum()
        return w, loss
    
    # Anneal to beta=1
    while beta < 1.0:
        # Choose next beta by ESS control
        loss_now = np.array([
            loss_fn(X[i], None if metadata is None else metadata[i])
            for i in range(N)
        ], dtype=float)
        
        beta_next = choose_next_beta(loss_now, beta, target_ess_frac=ess_trigger)
        beta_next = max(beta_next, min(1.0, beta + 1e-3))
        
        # Reweight
        weights, loss_now = reweight(beta, beta_next)
        beta = beta_next
        
        # Resample if needed
        ess = effective_sample_size(weights)
        if ess < ess_trigger * N:
            idx = systematic_resample(weights, rng)
            X = X[idx]
            if metadata is not None:
                metadata = metadata[idx]
            weights = np.ones(N) / N
        
        # Select gamma for this stage
        default_gamma, big_gamma = gamma_schedule
        gamma = big_gamma if (stage % big_step_every == 0 and stage > 0) else default_gamma
        
        # DE-MH moves with tempered likelihood
        def beta_loglike(theta, meta=None):
            return beta * loglike(theta, meta)
        
        X, acc = de_mh_move(
            X,
            beta_loglike,
            bounds,
            metadata=metadata,
            steps=moves_per_stage,
            gamma=gamma,
            rng=rng,
            max_workers=max_workers,
        )
        
        # Log chains
        for pid in range(N):
            row = [stage, pid, bool(acc[pid])]
            if metadata is not None:
                row.extend(np.asarray(metadata[pid]).tolist())
            row.extend(X[pid].tolist())
            chains.append(row)
        
        acc_rate = float(acc.mean()) if acc.size else 0.0
        print(f"[smc-demc] stage={stage:02d} beta={beta:.3f} ess={ess:.1f}/{N} accept={acc_rate:.2f}")
        
        stage += 1
        if beta >= 1.0 - 1e-12:
            break
    
    # Build DataFrame
    meta_cols = []
    if metadata is not None:
        meta_cols = [f"m{j}" for j in range(metadata.shape[1])]
    
    chains_df = pd.DataFrame(
        chains,
        columns=["stage", "pid", "accepted"] + meta_cols + [f"p{j}" for j in range(X.shape[1])],
    )
    
    return X.copy(), chains_df
