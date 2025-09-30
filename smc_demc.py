"""Differential Evolution MCMC utilities shared by the GA and SMC refinement stages."""

# smc_demc.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, List, Tuple

@dataclass
class Bound:
    lo: float
    hi: float

def reflect_to_bounds(x: np.ndarray, bounds: List[Bound]) -> np.ndarray:
    y = x.copy()
    for j,(lo,hi) in enumerate([(b.lo,b.hi) for b in bounds]):
        L = hi - lo
        if L <= 0: continue
        t = (y[j] - lo) % (2*L)
        y[j] = lo + (t if t <= L else 2*L - t)
    return y

def systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    N = len(w)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(w)
    idx = np.searchsorted(cumsum, positions, side='right')
    return idx

def effective_sample_size(w: np.ndarray) -> float:
    s = w.sum()
    return s*s / np.dot(w, w)

def choose_next_beta(loss: np.ndarray, beta_prev: float, target_ess_frac: float=0.6) -> float:
    """
    Pick the next beta so that ESS(new) ≈ target_ess_frac * N.
    Binary search on delta_beta in [0, 2] but clip at 1-beta_prev.
    """
    N = len(loss)
    lo, hi = 1e-6, max(1e-6, 1.0 - beta_prev)
    target = target_ess_frac * N
    for _ in range(30):
        mid = 0.5*(lo+hi)
        w = np.exp(-(mid)*loss)
        ess = effective_sample_size(w)
        if ess < target:
            hi = mid
        else:
            lo = mid
    return min(1.0, beta_prev + lo)

def de_mh_move(X: np.ndarray,
               loglike: Callable[[np.ndarray, object], float],
               bounds: List[Bound],
               metadata: np.ndarray = None,
               steps: int = 2,
               gamma: float = None,
               jitter: float = 1e-9,
               rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray]:
    """Run Differential-Evolution Metropolis proposals on an ensemble.

    The routine mirrors the original ter Braak DE-MC scheme: each walker proposes a new
    position using the scaled difference of two peers plus optional Gaussian jitter.  The
    supplied ``loglike`` is evaluated on every proposal, so callers can cache those results
    when they want to re-use the same evaluations later (as the hybrid GA now does).

    Returns ``(X_new, accepted)`` with a boolean mask signalling which walkers moved.
    """
    if rng is None: rng = np.random.default_rng()
    N, d = X.shape
    if gamma is None:
        gamma = 2.38 / np.sqrt(2*d)  # ter Braak default

    accepted = np.zeros(N, dtype=bool)

    if metadata is not None:
        meta_array = np.asarray(metadata, dtype=object)
    else:
        meta_array = None

    def _loglike(idx: int, theta: np.ndarray) -> float:
        if meta_array is None:
            return loglike(theta, None)
        return loglike(theta, meta_array[idx])

    L = np.array([_loglike(i, X[i]) for i in range(N)], dtype=float)

    for _ in range(steps):
        order = rng.permutation(N)
        for i in order:
            # pick two distinct other indices
            js = list(range(N)); js.remove(i)
            r1, r2 = rng.choice(js, size=2, replace=False)
            prop = X[i] + gamma*(X[r1]-X[r2]) + rng.normal(scale=jitter, size=d)
            prop = reflect_to_bounds(prop, bounds)
            L_new = _loglike(i, prop)
            # Metropolis accept on *loglike*
            if np.log(rng.random()) < (L_new - L[i]):
                X[i] = prop
                L[i] = L_new
                accepted[i] = True
    return X, accepted

def run_smc_demc(
    X0: np.ndarray,                         # initial ensemble (from your GA)
    loss_fn: Callable[[np.ndarray, object], float],  # returns scalar loss
    bounds: List[Bound],
    metadata0: np.ndarray = None,           # ancillary data carried with each particle
    ess_trigger: float = 0.6,               # resample when ESS/N < 0.6
    moves_per_stage: int = 3,
    rng: np.random.Generator = None,
    gamma_schedule: Tuple[float,float] = (None, 1.0),  # (default_gamma, occasional_big)
    big_step_every: int = 6,                # every k stages use gamma≈1
):
    """Tempered SMC with DE-MC mutation moves.

    The function is used in two places:

    1. Inside the GA run where it now shares its DE-MC move logic via ``de_mh_move``.
    2. As the dedicated SMC-DEMC posterior stage executed after the GA converges.

    It returns a refined ensemble and a Pandas DataFrame describing every stage/particle
    transition, which downstream tooling can persist as CSV artefacts.
    """
    if rng is None: rng = np.random.default_rng()

    N, d = X0.shape

    if metadata0 is not None:
        metadata = np.asarray(metadata0, dtype=object)
        if metadata.ndim == 1:
            metadata = metadata[:, None]
        if metadata.shape[0] != N:
            raise ValueError("metadata0 must have the same length as the ensemble")
    else:
        metadata = None

    # Work in *log-likelihood*; we have loss L, so loglike = -L
    def loglike(theta, meta=None):
        return -float(loss_fn(theta, meta))

    # state
    X = X0.copy()
    beta = 0.0
    stage = 0
    chains = []          # [(stage, particle_id, accepted, *params)]
    weights = np.ones(N) / N

    # helper to recompute weights for a beta jump
    def reweight(beta_prev, beta_new):
        nonlocal weights
        delta = beta_new - beta_prev
        # importance weights proportional to exp(-delta * loss)
        loss = np.array([loss_fn(X[i], None if metadata is None else metadata[i])
                         for i in range(N)], dtype=float)
        u = np.exp(-delta*loss)
        w = weights * u
        w /= w.sum()
        return w, loss

    # anneal to beta=1
    while beta < 1.0:
        # choose next beta by ESS control
        loss_now = np.array([loss_fn(X[i], None if metadata is None else metadata[i])
                             for i in range(N)], dtype=float)
        beta_next = choose_next_beta(loss_now, beta, target_ess_frac=ess_trigger)
        beta_next = max(beta_next, min(1.0, beta + 1e-3))
        # reweight
        weights, loss_now = reweight(beta, beta_next)
        beta = beta_next

        # resample if needed
        ess = effective_sample_size(weights)
        if ess < ess_trigger * N:
            idx = systematic_resample(weights, rng)
            X = X[idx]
            if metadata is not None:
                metadata = metadata[idx]
            weights = np.ones(N)/N

        # DE-MH move steps (posterior-invariant at current beta since accept uses loglike = -loss)
        default_gamma, big_gamma = gamma_schedule
        gamma = (big_gamma if (stage % big_step_every == 0 and stage > 0) else default_gamma)

        # scale loglike by current beta: accept ratio uses beta * loglike
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
        )

        # log chains
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

    meta_cols = []
    if metadata is not None:
        meta_cols = [f"m{j}" for j in range(metadata.shape[1])]
    chains_df = pd.DataFrame(
        chains,
        columns=["stage", "pid", "accepted"] + meta_cols + [f"p{j}" for j in range(X.shape[1])],
    )
    return X.copy(), chains_df

