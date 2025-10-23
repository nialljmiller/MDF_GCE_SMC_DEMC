```python
#!/usr/bin/env python3
"""
Unified Posterior Analysis Toolkit for GA Outputs.

This script combines uncertainty quantification and posterior analysis for galactic chemical evolution
genetic algorithm results. It can handle single or multiple folders, producing consolidated reports
including posteriors.csv, corner plots, MDF/AMR fits, and combined uncertainty visualizations.

For single folder: Produces standard posterior deliverables (posteriors.csv, corner.png, fit_mdf.png, etc.).
For multiple folders: Produces combined corner plots, marginals, and covariant uncertainties.

Usage:
- Run interactively; it will prompt for folder(s) to analyze (comma-separated for multiple).
- Outputs written to <folder>/analysis/ or <current>/analysis/ for multi-folder.

Dependencies: numpy, pandas, matplotlib, corner, scipy
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import corner
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'corner'. Install with `pip install corner`."
    ) from exc

from scipy.stats import gaussian_kde
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import glob
from scipy import stats

from plotting.style import *
use_paper_style()


# ----------------------------------------------------------------------------

def ensure_output_dirs(base_path):
    """Create necessary output directories under base_path/uncertainty"""
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, 'uncertainty'), exist_ok=True)

# ----------------------------------------------------------------------------

class UncertaintyAnalysis:
    """
    Comprehensive uncertainty quantification for galactic chemical evolution
    genetic algorithm results.
    """
    
    def __init__(self, results_file, output_path='SMC_DEMC/'):
        """
        Initialize uncertainty analysis.
        
        Parameters:
        -----------
        results_file : str
            Path to CSV file containing GA results
        output_path : str
            Base path for output files
        """
        self.results_file = results_file
        self.output_path = output_path
        ensure_output_dirs(output_path)
        
        # Load data
        self.df = pd.read_csv(results_file)
        self.fitness_col = 'fitness' if 'fitness' in self.df.columns else 'wrmse'
        
        # Define parameter sets
        self.continuous_params = [
            'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2', 
            'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb'
        ]
        self.categorical_params = [
            'comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx'
        ]
        
        # Filter to available parameters
        self.continuous_params = [p for p in self.continuous_params if p in self.df.columns]
        self.categorical_params = [p for p in self.categorical_params if p in self.df.columns]
        
        # Sort by fitness (lower is better)
        self.df_sorted = self.df.sort_values(self.fitness_col, ascending=True)
        
        print(f"Loaded {len(self.df)} models from {results_file}")
        print(f"Best fitness: {self.df_sorted[self.fitness_col].iloc[0]:.6f}")
        print(f"Available continuous parameters: {self.continuous_params}")

    # ---- helpers -----------------------------------------------------------
    def _select_top_and_weights(self, percentile=10, weight_power=1.0):
        """Return top subset and normalized inverse-fitness weights."""
        df = self.df_sorted.copy()
        n_top = max(1, int(len(df) * percentile / 100))
        top = df.head(n_top)
        fit = np.asarray(top[self.fitness_col].values, dtype=float)
        eps = np.min(fit) * 0.001
        w = 1.0 / np.power(fit + eps, weight_power)
        w = w / np.sum(w)
        return top, w

    def _wmean(self, x, w):
        x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float)
        return float(np.sum(w*x) / np.sum(w))

    def _wvar(self, x, w):
        m = self._wmean(x, w)
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        return float(np.sum(w*(x-m)**2) / np.sum(w))

    def _wcov(self, x, y, w):
        mx = self._wmean(x, w); my = self._wmean(y, w)
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        return float(np.sum(w*(x-mx)*(y-my)) / np.sum(w))

    def _wpercentile(self, x, w, p):
        x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float)
        w = w / np.sum(w)
        idx = np.argsort(x)
        x = x[idx]; w = w[idx]
        cum = np.cumsum(w)
        i = np.searchsorted(cum, p/100.0)
        if i == 0:
            return x[0]
        elif i >= len(x):
            return x[-1]
        frac = (p/100.0 - cum[i-1]) / (cum[i] - cum[i-1])
        return x[i-1] + frac * (x[i] - x[i-1])

    def _wcredint(self, x, w, p_hpd=68.3):
        """Weighted credible interval (centered on median)."""
        lo = self._wpercentile(x, w, 50 - p_hpd/2)
        hi = self._wpercentile(x, w, 50 + p_hpd/2)
        return lo, hi

    def _wkde(self, x, w, bw=0.05):
        """Weighted KDE."""
        x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float)
        w = w / np.sum(w)
        kde = gaussian_kde(x, bw_method=bw, weights=w)
        return kde

    def _wkde_pdf(self, x, w, bw=0.05, n_grid=200):
        kde = self._wkde(x, w, bw=bw)
        grid = np.linspace(np.min(x), np.max(x), n_grid)
        pdf = kde.evaluate(grid)
        return grid, pdf

    def _wkde_resample(self, x, w, n_samples=1000, bw=0.05):
        kde = self._wkde(x, w, bw=bw)
        return kde.resample(n_samples).flatten()

    def _wkde_cdf(self, x, w, bw=0.05, n_grid=200):
        grid, pdf = self._wkde_pdf(x, w, bw=bw, n_grid=n_grid)
        cdf = np.cumsum(pdf)
        cdf /= cdf[-1]
        return interp1d(grid, cdf, bounds_error=False, fill_value=(0,1))

    def _wkde_ppf(self, x, w, bw=0.05, n_grid=200):
        grid, pdf = self._wkde_pdf(x, w, bw=bw, n_grid=n_grid)
        cdf = np.cumsum(pdf)
        cdf /= cdf[-1]
        return interp1d(cdf, grid, bounds_error=False, fill_value=(grid[0], grid[-1]))

    def _wkde_percentile(self, x, w, p, bw=0.05, n_grid=200):
        ppf = self._wkde_ppf(x, w, bw=bw, n_grid=n_grid)
        return ppf(p/100.0)

    def _wkde_hpd(self, x, w, p_hpd=68.3, bw=0.05, n_grid=200):
        """Weighted KDE HPD interval (minimum-width)."""
        grid, pdf = self._wkde_pdf(x, w, bw=bw, n_grid=n_grid)
        cdf = np.cumsum(pdf)
        cdf /= cdf[-1]
        # find shortest interval containing p_hpd mass
        level = np.percentile(pdf, 100 - p_hpd)
        idx = np.where(pdf >= level)[0]
        if len(idx) == 0:
            return np.min(x), np.max(x)
        starts = idx[np.diff(idx, prepend=-1) > 1]
        ends = idx[np.diff(idx, append=len(pdf)) > 1]
        widths = grid[ends] - grid[starts]
        i = np.argmin(widths)
        return grid[starts[i]], grid[ends[i]]

    def _wkde_mode(self, x, w, bw=0.05, n_grid=200):
        grid, pdf = self._wkde_pdf(x, w, bw=bw, n_grid=n_grid)
        return grid[np.argmax(pdf)]

    def _wkde_modes(self, x, w, bw=0.05, n_grid=200, threshold=0.1):
        grid, pdf = self._wkde_pdf(x, w, bw=bw, n_grid=n_grid)
        peaks = np.where((pdf[1:-1] > pdf[:-2]) & (pdf[1:-1] > pdf[2:]))[0] + 1
        thresh = threshold * np.max(pdf)
        peaks = peaks[pdf[peaks] >= thresh]
        return grid[peaks]

    def _wresample(self, x, w, n_samples=1000):
        w = np.asarray(w, dtype=float) / np.sum(w)
        return np.random.choice(x, size=n_samples, p=w, replace=True)

    def _wbootstrap(self, x, w, statistic=np.mean, n_resamples=1000, ci=68.3):
        resamples = [statistic(self._wresample(x, w)) for _ in range(n_resamples)]
        lo = np.percentile(resamples, 50 - ci/2)
        hi = np.percentile(resamples, 50 + ci/2)
        return np.mean(resamples), lo, hi

    def _wspearmanr(self, x, y, w):
        """Weighted Spearman rank correlation."""
        # rank the data (with average ties)
        rx = stats.rankdata(x)
        ry = stats.rankdata(y)
        # center
        mx = self._wmean(rx, w)
        my = self._wmean(ry, w)
        rx = rx - mx; ry = ry - my
        # covariance
        cov = self._wcov(rx, ry, w)
        # std devs
        sx = np.sqrt(self._wvar(rx, w))
        sy = np.sqrt(self._wvar(ry, w))
        return cov / (sx * sy + 1e-300)

    def _wpearsonr(self, x, y, w):
        return self._wcov(x, y, w) / (np.sqrt(self._wvar(x, w)) * np.sqrt(self._wvar(y, w)) + 1e-300)

    def _wassoc(self, x, y, w, metric='spearman'):
        if metric == 'spearman':
            return self._wspearmanr(x, y, w)
        elif metric == 'pearson':
            return self._wpearsonr(x, y, w)
        else:
            raise ValueError(f"Unknown association metric: {metric}")

    # ---- marginal summaries ------------------------------------------------
    def marginal_summary(self, param, percentile=10, weight_power=1.0, p_hpd=68.3):
        """Compute weighted marginal statistics for a parameter."""
        top, w = self._select_top_and_weights(percentile, weight_power)
        x = np.asarray(top[param].values, dtype=float)
        
        stats = {
            'param': param,
            'mean': self._wmean(x, w),
            'var': self._wvar(x, w),
            'std': np.sqrt(self._wvar(x, w)),
            'median': self._wpercentile(x, w, 50),
            'mode': self._wkde_mode(x, w),
            'p16': self._wpercentile(x, w, 50 - p_hpd/2),
            'p84': self._wpercentile(x, w, 50 + p_hpd/2),
            'p2.5': self._wpercentile(x, w, 2.5),
            'p97.5': self._wpercentile(x, w, 97.5),
            'hpd_lo': self._wkde_hpd(x, w, p_hpd)[0],
            'hpd_hi': self._wkde_hpd(x, w, p_hpd)[1],
            'skew': stats.skew(x, bias=False),
            'kurtosis': stats.kurtosis(x, bias=False),
            'n_modes': len(self._wkde_modes(x, w)),
        }
        
        return stats

    def all_marginal_summaries(self, percentile=10, weight_power=1.0, p_hpd=68.3):
        summaries = []
        for p in self.continuous_params:
            summaries.append(self.marginal_summary(p, percentile, weight_power, p_hpd))
        return pd.DataFrame(summaries)

    # ---- pairwise associations ---------------------------------------------
    def pairwise_associations(self, params=None, percentile=10, weight_power=1.0, metric='spearman'):
        if params is None:
            params = self.continuous_params
        n = len(params)
        assoc = np.zeros((n,n))
        top, w = self._select_top_and_weights(percentile, weight_power)
        
        for i in range(n):
            x = np.asarray(top[params[i]].values, dtype=float)
            for j in range(i+1, n):
                y = np.asarray(top[params[j]].values, dtype=float)
                r = self._wassoc(x, y, w, metric=metric)
                assoc[i,j] = r
                assoc[j,i] = r
        
        return pd.DataFrame(assoc, index=params, columns=params)

    # ---- plotting helpers --------------------------------------------------
    def plot_marginal_pdf(self, param, percentile=10, weight_power=1.0, bw=0.05, 
                         ax=None, color='blue', label=None, alpha=0.5):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,4))
        
        top, w = self._select_top_and_weights(percentile, weight_power)
        x = np.asarray(top[param].values, dtype=float)
        grid, pdf = self._wkde_pdf(x, w, bw=bw)
        
        ax.fill_between(grid, 0, pdf, color=color, alpha=alpha, label=label)
        ax.plot(grid, pdf, color=color, alpha=0.8)
        
        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        if label:
            ax.legend()
        
        return ax

    def plot_corner_with_marginals(self, params=None, percentile=10, weight_power=1.0,
                                  bins=40, assoc_metric='spearman', alpha_gamma=0.9,
                                  color='blue', save_path=None):
        if params is None:
            params = self.continuous_params
        n = len(params)
        
        fig, axes = plt.subplots(n, n, figsize=(2*n, 2*n))
        top, w = self._select_top_and_weights(percentile, weight_power)
        
        for i in range(n):
            x = np.asarray(top[params[i]].values, dtype=float)
            
            # diagonal: marginal PDF
            ax = axes[i,i]
            self.plot_marginal_pdf(params[i], ax=ax, color=color, alpha=alpha_gamma)
            ax.set_yticks([])
            
            for j in range(i+1, n):
                y = np.asarray(top[params[j]].values, dtype=float)
                
                # lower: scatter/hexbin
                ax = axes[j,i]
                ax.hexbin(x, y, gridsize=bins, C=w, reduce_C_function=np.sum,
                         cmap='Blues', mincnt=1e-10)
                
                # upper: association
                axu = axes[i,j]
                r = self._wassoc(x, y, w, metric=assoc_metric)
                axu.text(0.5, 0.5, f"{r:.2f}", ha='center', va='center', fontsize=12)
                axu.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        return fig

# ---- multi-run helpers -------------------------------------------------
def _choose_common_params(analyzers):
    """Find intersection of continuous parameters across analyzers."""
    if not analyzers:
        return []
    params = set(analyzers[0].continuous_params)
    for a in analyzers[1:]:
        params &= set(a.continuous_params)
    return sorted(list(params))

def plot_corner_with_marginals_multi(analyzers, params=None, percentile=100, weight_power=1.0,
                                    bins=40, assoc_metric='spearman', alpha_gamma=0.9,
                                    ink_colors=None, legend_labels=None, save_path=None):
    if params is None:
        params = _choose_common_params(analyzers)
    n = len(params)
    n_runs = len(analyzers)
    
    if ink_colors is None:
        ink_colors = plt.cm.viridis(np.linspace(0,1,n_runs))
    if legend_labels is None:
        legend_labels = [f"Run {i+1}" for i in range(n_runs)]
    
    fig = plt.figure(figsize=(2*n, 2*n))
    gs = GridSpec(n, n, fig)
    
    for i in range(n):
        for j in range(i+1, n):
            # lower triangle: scatters
            ax = fig.add_subplot(gs[j,i])
            for k, a in enumerate(analyzers):
                top, w = a._select_top_and_weights(percentile, weight_power)
                x = np.asarray(top[params[i]].values, dtype=float)
                y = np.asarray(top[params[j]].values, dtype=float)
                ax.scatter(x, y, s=10*w/np.max(w), c=ink_colors[k], alpha=alpha_gamma,
                          label=legend_labels[k] if i==0 and j==1 else None)
            if j == n-1:
                ax.set_xlabel(params[i])
            if i == 0:
                ax.set_ylabel(params[j])
        
        # diagonal: marginals
        ax = fig.add_subplot(gs[i,i])
        for k, a in enumerate(analyzers):
            a.plot_marginal_pdf(params[i], ax=ax, color=ink_colors[k], 
                               alpha=alpha_gamma, label=legend_labels[k])
    
    if legend_labels:
        fig.legend(loc='upper right')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    return fig

def compute_and_plot_combined_covariant_uncertainties(analyzers, params=None, percentile=100,
                                                     weight_power=1.0, p_hpd=68.3, grid_n=240,
                                                     alpha_gamma=0.9, ink_colors=None,
                                                     save_dir=None):
    if params is None:
        params = _choose_common_params(analyzers)
    n = len(params)
    n_runs = len(analyzers)
    
    if ink_colors is None:
        ink_colors = plt.cm.viridis(np.linspace(0,1,n_runs))
    
    fig, axes = plt.subplots(n, n, figsize=(2*n, 2*n))
    
    for i in range(n):
        for j in range(i+1, n):
            ax = axes[j,i]
            for k, a in enumerate(analyzers):
                top, w = a._select_top_and_weights(percentile, weight_power)
                x = np.asarray(top[params[i]].values, dtype=float)
                y = np.asarray(top[params[j]].values, dtype=float)
                
                # 2D KDE contour
                kde = gaussian_kde(np.vstack([x,y]), weights=w)
                xi, yi = np.mgrid[np.min(x):np.max(x):grid_n*1j, np.min(y):np.max(y):grid_n*1j]
                zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
                ax.contour(xi, yi, zi.reshape(xi.shape), colors=ink_colors[k], alpha=alpha_gamma)
    
    if save_dir:
        save_path = os.path.join(save_dir, 'combined_covariant_uncertainties.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    
    return fig

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


def compute_weights(
    loss: Sequence[float],
    temperature: float | None = None,
    floor: float = 1e-12,
) -> Tuple[np.ndarray, float, float]:
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


# ----------------------------------------------------------------------------
# NPZ loader utilities
# ----------------------------------------------------------------------------


def _load_history(npz_path: Path) -> Dict[str, List]:
    data = np.load(npz_path, allow_pickle=True)
    out: Dict[str, List] = {}
    for key in ("histories", "mdf_data", "alpha_data", "age_data"):
        if key in data.files:
            out[key] = list(data[key])
        else:
            out[key] = []
    out["walker_ids"] = list(data["walker_ids"]) if "walker_ids" in data.files else []
    return out


# ----------------------------------------------------------------------------


def _posterior_draws(
    df: pd.DataFrame,
    weights: np.ndarray,
    params: List[str],
    n_draws: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if len(df) != len(weights):
        raise ValueError("DataFrame and weights must have the same length")
    draws = df[params].sample(n=n_draws, replace=True, weights=weights, random_state=rng)
    return draws.reset_index(drop=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_corner(df: pd.DataFrame, save_path: Path) -> None:
    fig = corner.corner(df, quantiles=[0.16, 0.5, 0.84], show_titles=True)
    fig.savefig(save_path)
    plt.close(fig)


def _reorder_by_fitness(
    df_sorted: pd.DataFrame,
    curves: List,
    fitness_col: str = "fitness",
) -> List:
    if not curves:
        return []
    if "walker_id" not in df_sorted.columns:
        return curves  # no reordering possible
    order = df_sorted["walker_id"].values.astype(int)
    idx_map = {wid: i for i, wid in enumerate(order) if not np.isnan(wid)}
    reordered = [None] * len(curves)
    for i, curve in enumerate(curves):
        if not isinstance(curve, dict) or "walker_id" not in curve:
            continue
        wid = curve["walker_id"]
        new_idx = idx_map.get(wid)
        if new_idx is not None:
            reordered[new_idx] = curve
    # fill any gaps with originals (unlikely)
    for i, c in enumerate(reordered):
        if c is None and i < len(curves):
            reordered[i] = curves[i]
    return reordered


def _plot_mdf(
    mdf_curves: List[Dict[str, np.ndarray]],
    weights: np.ndarray,
    obs_file: Path,
    save_path: Path,
) -> None:
    if not mdf_curves:
        print("[posterior] no MDF curves found; skipping fit_mdf.png")
        return
    obs_feh, obs_count = np.loadtxt(obs_file, usecols=(0, 1), unpack=True)
    obs_norm = obs_count / obs_count.max()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(obs_feh, obs_norm, "k--", label="Observed")

    # weighted quantiles
    n = len(mdf_curves)
    if len(weights) != n:
        weights = np.ones(n) / n

    feh_grid = np.linspace(-2.5, 1.0, 200)
    mdf_interp = np.zeros((n, len(feh_grid)))
    for i, curve in enumerate(mdf_curves):
        if not isinstance(curve, dict) or "feh" not in curve or "norm_count" not in curve:
            continue
        interp = np.interp(feh_grid, curve["feh"], curve["norm_count"], left=0, right=0)
        mdf_interp[i] = interp

    q16 = np.percentile(mdf_interp, 16, axis=0, weights=weights, method='inverted_cdf')
    q50 = np.percentile(mdf_interp, 50, axis=0, weights=weights, method='inverted_cdf')
    q84 = np.percentile(mdf_interp, 84, axis=0, weights=weights, method='inverted_cdf')

    ax.fill_between(feh_grid, q16, q84, color="blue", alpha=0.3, label="68% credible")
    ax.plot(feh_grid, q50, "b-", label="Median")

    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("Normalized Count")
    ax.set_xlim(-2.5, 1.0)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _plot_amr(
    age_curves: List[Dict[str, np.ndarray]],
    weights: np.ndarray,
    obs_df: pd.DataFrame,
    dataset: str,
    save_path: Path,
) -> None:
    if not age_curves:
        print("[posterior] no AMR curves found; skipping fit_amr.png")
        return
    if obs_df.empty:
        print("[posterior] no observational age data; skipping fit_amr.png")
        return

    age_col = "Joyce_age" if dataset.lower() == "joyce" else "Bensby"
    if age_col not in obs_df.columns:
        raise ValueError(f"Age column '{age_col}' not in observational data")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        obs_df[age_col],
        obs_df["[Fe/H]"],
        s=10,
        alpha=0.5,
        color="gray",
        label="Observed",
    )

    n = len(age_curves)
    if len(weights) != n:
        weights = np.ones(n) / n

    age_grid = np.linspace(0, 14, 200)
    feh_interp = np.zeros((n, len(age_grid)))
    for i, curve in enumerate(age_curves):
        if not isinstance(curve, dict) or "age_gyr" not in curve or "feh" not in curve:
            continue
        interp = np.interp(age_grid, curve["age_gyr"], curve["feh"], left=np.nan, right=np.nan)
        feh_interp[i] = interp

    mask = np.isfinite(feh_interp).all(axis=0)
    q16 = np.percentile(feh_interp[:, mask], 16, axis=0, weights=weights, method='inverted_cdf')
    q50 = np.percentile(feh_interp[:, mask], 50, axis=0, weights=weights, method='inverted_cdf')
    q84 = np.percentile(feh_interp[:, mask], 84, axis=0, weights=weights, method='inverted_cdf')

    ax.fill_between(age_grid[mask], q16, q84, color="blue", alpha=0.3, label="68% credible")
    ax.plot(age_grid[mask], q50, "b-", label="Median")

    ax.set_xlabel("Age (Gyr)")
    ax.set_ylabel("[Fe/H]")
    ax.set_xlim(0, 14)
    ax.set_ylim(-2, 1)
    ax.legend(frameon=False)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _plot_alpha(
    alpha_curves: List[Dict[str, np.ndarray]],
    weights: np.ndarray,
    obs_df: pd.DataFrame,
    save_path: Path,
) -> None:
    if not alpha_curves:
        print("[posterior] no alpha curves found; skipping fit_alpha.png")
        return
    if obs_df.empty:
        print("[posterior] no observational data; skipping fit_alpha.png")
        return

    elements = ["[Mg/Fe]", "[Si/Fe]", "[Ca/Fe]", "[Ti/Fe]"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    n = len(alpha_curves)
    if len(weights) != n:
        weights = np.ones(n) / n

    feh_grid = np.linspace(-2, 1, 200)

    for idx, elem in enumerate(elements):
        ax = axes[idx]
        if elem not in obs_df.columns:
            continue
        ax.scatter(
            obs_df["[Fe/H]"],
            obs_df[elem],
            s=10,
            alpha=0.5,
            color="gray",
            label="Observed",
        )

        alpha_interp = np.zeros((n, len(feh_grid)))
        for i, curve in enumerate(alpha_curves):
            if not isinstance(curve, dict) or "feh" not in curve or elem not in curve:
                continue
            interp = np.interp(
                feh_grid, curve["feh"], curve[elem], left=np.nan, right=np.nan
            )
            alpha_interp[i] = interp

        mask = np.isfinite(alpha_interp).all(axis=0)
        q16 = np.percentile(alpha_interp[:, mask], 16, axis=0, weights=weights, method='inverted_cdf')
        q50 = np.percentile(alpha_interp[:, mask], 50, axis=0, weights=weights, method='inverted_cdf')
        q84 = np.percentile(alpha_interp[:, mask], 84, axis=0, weights=weights, method='inverted_cdf')

        ax.fill_between(feh_grid[mask], q16, q84, color="blue", alpha=0.3, label="68% credible")
        ax.plot(feh_grid[mask], q50, "b-", label="Median")

        ax.set_xlabel("[Fe/H]")
        ax.set_ylabel(elem)
        if idx == 0:
            ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _plot_walker_paths(
    histories: List[List[List[float]]],
    param_names: List[str],
    save_path: Path,
) -> None:
    if not histories:
        print("[posterior] no walker histories found; skipping walker_paths.png")
        return

    n_walkers = len(histories)
    n_params = len(param_names)
    n_steps = max(len(h) for h in histories) if histories else 0

    if n_steps == 0:
        return

    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2 * n_params), sharex=True)
    axes = np.atleast_1d(axes)

    for i, ax in enumerate(axes):
        for w in range(n_walkers):
            path = [step[i] for step in histories[w] if len(step) > i]
            ax.plot(path, alpha=0.3)

        ax.set_ylabel(param_names[i])

    axes[-1].set_xlabel("Generation")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def run_single_posterior_report(
    results_file: str,
    history_file: str = None,
    pcard_file: str = "bulge_pcard.txt",
    output_dir: str = None,
    params: List[str] = None,
    nsamples: int = 5000,
    temperature: float = None,
    seed: int = 42,
) -> Dict:
    results_path = Path(results_file).expanduser().resolve()
    if not results_path.is_file():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    df = pd.read_csv(results_path)
    fitness_col = "fitness" if "fitness" in df.columns else "wrmse"
    if fitness_col not in df.columns:
        raise ValueError(f"No fitness column found in {results_file}")

    df_sorted = df.sort_values(fitness_col, ascending=True).reset_index(drop=True)
    loss = df_sorted[fitness_col].values

    weights, temperature, ess = compute_weights(loss, temperature=temperature)

    if params is None:
        params = [c for c in df_sorted.columns if c not in {fitness_col, "walker_id"}]

    if "walker_id" in df_sorted.columns:
        order = df_sorted["walker_id"].values.astype(int)
    else:
        order = np.arange(len(df_sorted))

    rng = np.random.default_rng(seed)
    posterior_draws = _posterior_draws(df_sorted, weights, params, nsamples, rng)

    base_dir = Path(output_dir).expanduser().resolve() if output_dir else results_path.parent / "posterior"
    _ensure_dir(base_dir)

    post_csv = base_dir / "posteriors.csv"
    posterior_draws.to_csv(post_csv, index=False)

    weights_csv = base_dir / "posterior_weights.csv"
    df_weights = df_sorted.assign(weight=weights)
    df_weights.to_csv(weights_csv, index=False)

    _save_corner(posterior_draws, base_dir / "corner.png")

    history_path = Path(history_file).expanduser().resolve() if history_file else results_path.parent / "walker_history.npz"
    history = _load_history(history_path) if history_path.is_file() else {"histories": [], "mdf_data": [], "alpha_data": [], "age_data": []}

    def _reorder_curves(key: str) -> List:
        curves = history.get(key, [])
        if not curves:
            return []
        if len(curves) != len(order):
            return list(curves)
        return [curves[i] for i in order]

    mdf_curves = _reorder_curves("mdf_data")
    alpha_curves = _reorder_curves("alpha_data")
    age_curves = _reorder_curves("age_data")

    pcard_path = Path(pcard_file).expanduser().resolve()
    pcard = {}  # Placeholder; implement parse_inlist if needed
    base_root = pcard_path.parent
    obs_file = Path(pcard.get("obs_file", "data/statistically_rigorous_mdf.dat"))
    if not obs_file.is_absolute():
        obs_file = base_root / obs_file

    _plot_mdf(mdf_curves, weights, obs_file, base_dir / "fit_mdf.png")

    obs_age_path = Path(pcard.get("obs_age_data", "data/Bensby_Data.tsv"))
    if not obs_age_path.is_absolute():
        obs_age_path = base_root / obs_age_path
    obs_age_df = pd.read_csv(obs_age_path, sep="\t") if obs_age_path.is_file() else pd.DataFrame()
    obs_dataset = pcard.get("obs_age_data_target", "joyce")

    if not obs_age_df.empty:
        _plot_amr(age_curves, weights, obs_age_df, obs_dataset, base_dir / "fit_amr.png")
        _plot_alpha(alpha_curves, weights, obs_age_df, base_dir / "fit_alpha.png")

    _plot_walker_paths(history.get("histories", []), df_sorted.columns.tolist(), base_dir / "walker_paths.png")

    summary = {
        "results_file": str(results_path),
        "history_file": str(history_path) if history_path.is_file() else None,
        "pcard": str(pcard_path) if pcard_path.is_file() else None,
        "temperature": temperature,
        "effective_sample_size": ess,
        "n_models": len(df_sorted),
        "posterior_draws": int(nsamples),
        "parameters": params,
        "output_dir": str(base_dir),
    }

    with open(base_dir / "posterior_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


def run_multi_folder_analysis(folders: List[str]):
    analyzers = []
    labels = []
    ink_colors = ['#F0B800', '#004C40', '#0099A1', '#C20016', '#E8DCD8', '#97BAAB', '#1E6E6C', '#99724B', '#59454E']
    
    for folder in folders:
        results_file = find_highest_gen_file(folder)
        if not results_file or not os.path.isfile(results_file):
            print(f"Skipping {folder}: No valid simulation_results CSV found")
            continue
        
        a = UncertaintyAnalysis(results_file, output_path=folder)
        analyzers.append(a)
        labels.append(os.path.basename(folder) + f" ({a.df_sorted[a.fitness_col].iloc[0]:.4f})")

    if not analyzers:
        print("No valid folders found")
        return

    params = _choose_common_params(analyzers)
    overlay_dir = os.path.join(os.getcwd(), "combined_analysis")
    os.makedirs(overlay_dir, exist_ok=True)

    fig = plot_corner_with_marginals_multi(
        analyzers,
        params=params,
        percentile=100,
        weight_power=1.0,
        bins=40,
        assoc_metric='spearman',
        alpha_gamma=0.9,
        ink_colors=ink_colors[:len(analyzers)],
        legend_labels=labels,
        save_path=os.path.join(overlay_dir, "corner_with_marginals_combined.png")
    )

    _ = compute_and_plot_combined_covariant_uncertainties(
        analyzers,
        params=params,
        percentile=100,
        weight_power=1.0,
        p_hpd=68.3,
        grid_n=240,
        alpha_gamma=0.9,
        ink_colors=ink_colors[:len(analyzers)],
        save_dir=overlay_dir
    )

    print(f"Combined analysis outputs written to: {overlay_dir}")

def find_folders_with_results(root_dir='.'):
    folders = []
    for dirpath, _, filenames in os.walk(root_dir):
        if any(f.startswith('simulation_results') and f.endswith('.csv') for f in filenames):
            folders.append(dirpath)
    return sorted(folders)

def find_highest_gen_file(folder):
    files = glob.glob(os.path.join(folder, 'simulation_results_gen_*.csv'))
    if not files:
        files = glob.glob(os.path.join(folder, 'simulation_results.csv'))
    if not files:
        return None
    gen_numbers = []
    for f in files:
        match = re.search(r'simulation_results_gen_(\d+)\.csv', f)
        if match:
            gen_numbers.append(int(match.group(1)))
    if gen_numbers:
        highest_gen = max(gen_numbers)
        return os.path.join(folder, f'simulation_results_gen_{highest_gen}.csv')
    return files[0]  # Fallback to simulation_results.csv if no gen files


def main():
    folders = find_folders_with_results()
    if not folders:
        print("No folders containing simulation_results*.csv found. Exiting.")
        return

    print("Folders with simulation results:")
    for i, folder in enumerate(folders, 1):
        print(f"{i}: {folder}")
    
    user_input = input("Enter the numbers of folders to analyze (comma-separated, e.g., 1,2,3): ").strip()
    selected_indices = [int(i.strip()) - 1 for i in user_input.split(',') if i.strip().isdigit()]
    selected_folders = [folders[i] for i in selected_indices if 0 <= i < len(folders)]

    if not selected_folders:
        print("No valid folders selected. Exiting.")
        return

    if len(selected_folders) == 1:
        folder = selected_folders[0]
        results_file = find_highest_gen_file(folder)
        if not results_file:
            print(f"No valid simulation_results CSV found in {folder}. Exiting.")
            return
        history_file = os.path.join(folder, 'walker_history.npz')
        summary = run_single_posterior_report(results_file, history_file=history_file)
        print(json.dumps(summary, indent=2))
    else:
        run_multi_folder_analysis(selected_folders)

if __name__ == "__main__":
    main()
```