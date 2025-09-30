#!/usr/bin/env python3
"""Posterior analysis toolkit for GA outputs.

This script takes the GA result catalogue (``simulation_results.csv``) plus the
companion ``walker_history.npz`` snapshot and produces a consolidated posterior
report.  The following deliverables are written to ``<output>/posterior/`` by
default:

* ``posteriors.csv`` – posterior draws resampled according to the
  loss-derived weights.
* ``posterior_weights.csv`` – the original catalogue with the normalized
  weights for transparency.
* ``corner.png`` – weighted corner plot for all continuous parameters.
* ``fit_mdf.png`` – MDF fit with 16/50/84%-credible bands.
* ``fit_amr.png`` – AMR fit against the chosen observational catalogue.
* ``fit_alpha.png`` – [α/Fe] trends with posterior credible bands.
* ``walker_paths.png`` – parameter trajectories for every GA walker.
* ``posterior_summary.json`` – metadata (temperature, ESS, etc.).

The goal is to keep the analysis focused on a small, well-defined set of plots
that the paper repeatedly relies upon.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import corner
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency 'corner'. Install with `pip install corner`."
    ) from exc

from Gal_GA_PP import parse_inlist


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
# Weighted summary helpers
# ----------------------------------------------------------------------------


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: Sequence[float]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return np.full(len(qs), np.nan)
    v = v[mask]
    w = w[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return np.interp(qs, cdf, v)


def _grid_summary(
    curves: Sequence[Tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
    grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.full((len(curves), grid.size), np.nan, dtype=float)
    for i, (x, y) in enumerate(curves):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        if mask.sum() < 2:
            continue
        matrix[i] = np.interp(grid, x_arr[mask], y_arr[mask], left=np.nan, right=np.nan)

    qs = (0.16, 0.5, 0.84)
    lo = np.empty(grid.size)
    med = np.empty(grid.size)
    hi = np.empty(grid.size)
    for j in range(grid.size):
        column = matrix[:, j]
        if np.all(np.isnan(column)):
            lo[j] = med[j] = hi[j] = np.nan
            continue
        q16, q50, q84 = _weighted_quantile(column, weights, qs)
        lo[j], med[j], hi[j] = q16, q50, q84
    return lo, med, hi


# ----------------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_corner(samples: pd.DataFrame, out_path: Path) -> None:
    fig = corner.corner(
        samples.to_numpy(),
        labels=[c.replace("_", " ") for c in samples.columns],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3g",
        bins=40,
        smooth=0.9,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_mdf(
    curves: Sequence[Tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
    obs_file: Path,
    out_path: Path,
) -> None:
    if not curves or not obs_file.is_file():
        return

    obs = np.loadtxt(obs_file, usecols=(0, 1))
    obs_x = obs[:, 0]
    obs_y = obs[:, 1]
    if obs_y.max() > 0:
        obs_y = obs_y / obs_y.max()

    valid_curves: List[Tuple[np.ndarray, np.ndarray]] = []
    valid_weights: List[float] = []
    for idx, (x, y) in enumerate(curves):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        if mask.sum() < 4:
            continue
        valid_curves.append((x_arr[mask], y_arr[mask]))
        if idx < len(weights):
            valid_weights.append(float(weights[idx]))
        else:
            valid_weights.append(0.0)

    if not valid_curves:
        return

    valid_weights = np.asarray(valid_weights, dtype=float)
    if valid_weights.sum() <= 0:
        valid_weights = np.ones(len(valid_curves), dtype=float) / len(valid_curves)
    else:
        valid_weights /= valid_weights.sum()

    grid = np.linspace(
        min(float(np.min(x)) for x, _ in valid_curves),
        max(float(np.max(x)) for x, _ in valid_curves),
        256,
    )
    lo, med, hi = _grid_summary(valid_curves, valid_weights, grid)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(obs_x, obs_y, color="k", lw=1.6, label="Observed MDF")
    ax.plot(grid, med, color="C1", lw=2.0, label="Posterior median")
    ax.fill_between(grid, lo, hi, color="C1", alpha=0.3, label="16–84% band")
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("Normalized counts")
    ax.set_title("Metallicity Distribution Function")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _convert_model_age(age_series: np.ndarray) -> np.ndarray:
    arr = np.asarray(age_series, dtype=float)
    if arr.size == 0:
        return arr
    if np.nanmax(arr) > 100.0:
        return (arr[-1] / 1e9) - arr / 1e9
    return arr


def _plot_amr(
    age_curves: Sequence[Tuple[np.ndarray, np.ndarray]],
    weights: np.ndarray,
    obs_age: pd.DataFrame,
    obs_dataset: str,
    out_path: Path,
) -> None:
    if not age_curves:
        return

    dataset = obs_dataset.lower()
    if dataset not in {"joyce", "bensby"}:
        dataset = "joyce"

    obs_age_col = "Joyce_age" if dataset == "joyce" else "Bensby"
    if obs_age_col not in obs_age.columns or "[Fe/H]" not in obs_age.columns:
        return
    obs_mask = np.isfinite(obs_age[obs_age_col]) & np.isfinite(obs_age["[Fe/H]"])
    obs_age_vals = obs_age.loc[obs_mask, obs_age_col]
    obs_feh_vals = obs_age.loc[obs_mask, "[Fe/H]"]

    curves_gyr: List[Tuple[np.ndarray, np.ndarray]] = []
    weights_used: List[float] = []
    for idx, (ages, feh) in enumerate(age_curves):
        ages_arr = _convert_model_age(np.asarray(ages))
        feh_arr = np.asarray(feh, dtype=float)
        mask = np.isfinite(ages_arr) & np.isfinite(feh_arr)
        if mask.sum() < 4:
            continue
        curves_gyr.append((ages_arr[mask], feh_arr[mask]))
        if idx < len(weights):
            weights_used.append(float(weights[idx]))
        else:
            weights_used.append(0.0)

    if not curves_gyr:
        return

    weights_used = np.asarray(weights_used, dtype=float)
    if weights_used.sum() <= 0:
        weights_used = np.ones(len(curves_gyr), dtype=float) / len(curves_gyr)
    else:
        weights_used /= weights_used.sum()

    grid = np.linspace(0, max(max(a) for a, _ in curves_gyr), 256)
    lo, med, hi = _grid_summary(curves_gyr, weights_used, grid)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(obs_age_vals, obs_feh_vals, s=18, color="k", alpha=0.6, label=f"{obs_dataset.title()} data")
    ax.plot(grid, med, color="C2", lw=2.0, label="Posterior median")
    ax.fill_between(grid, lo, hi, color="C2", alpha=0.3, label="16–84% band")
    ax.set_xlabel("Age (Gyr)")
    ax.set_ylabel("[Fe/H]")
    ax.set_title("Age–Metallicity Relation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


ALPHA_LABELS = ["[Si/Fe]", "[Ca/Fe]", "[Mg/Fe]", "[Ti/Fe]"]


def _plot_alpha(
    alpha_curves: Sequence[Sequence[Tuple[np.ndarray, np.ndarray]]],
    weights: np.ndarray,
    obs_age: pd.DataFrame,
    out_path: Path,
) -> None:
    if not alpha_curves:
        return

    cols = 2
    rows = int(math.ceil(len(ALPHA_LABELS) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), sharex=True, sharey=True)
    axes = axes.flatten()

    if "[Fe/H]" not in obs_age.columns:
        return
    feh_obs = obs_age["[Fe/H]"]

    for idx, label in enumerate(ALPHA_LABELS):
        ax = axes[idx]
        obs_col = label
        if obs_col not in obs_age.columns:
            continue
        obs_mask = np.isfinite(obs_age[obs_col]) & np.isfinite(feh_obs)
        ax.scatter(
            feh_obs[obs_mask],
            obs_age.loc[obs_mask, obs_col],
            s=16,
            color="k",
            alpha=0.5,
            label="Observations",
        )

        curves_this: List[Tuple[np.ndarray, np.ndarray]] = []
        weights_this: List[float] = []
        for idx_curve, alpha_arrs in enumerate(alpha_curves):
            if idx >= len(alpha_arrs):
                continue
            x, y = alpha_arrs[idx]
            x_arr = np.asarray(x, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            if mask.sum() < 4:
                continue
            curves_this.append((x_arr[mask], y_arr[mask]))
            if idx_curve < len(weights):
                weights_this.append(float(weights[idx_curve]))
            else:
                weights_this.append(0.0)

        if curves_this:
            weights_this = np.asarray(weights_this, dtype=float)
            if weights_this.sum() <= 0:
                weights_this = np.ones(len(curves_this), dtype=float) / len(curves_this)
            else:
                weights_this /= weights_this.sum()
            grid = np.linspace(min(min(c[0]) for c in curves_this), max(max(c[0]) for c in curves_this), 256)
            lo, med, hi = _grid_summary(curves_this, weights_this, grid)
            ax.plot(grid, med, color="C3", lw=2.0, label="Posterior median")
            ax.fill_between(grid, lo, hi, color="C3", alpha=0.3, label="16–84% band")

        ax.set_title(label)
        ax.set_xlabel("[Fe/H]")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    for ax in axes[len(ALPHA_LABELS) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_walker_paths(
    histories: Sequence[np.ndarray],
    param_names: Sequence[str],
    out_path: Path,
) -> None:
    if not histories:
        return

    array_histories = []
    for hist in histories:
        arr = np.asarray(hist, dtype=float)
        if arr.ndim == 2:
            array_histories.append(arr)
    if not array_histories:
        return

    genome_len = array_histories[0].shape[1]
    if any(h.shape[1] != genome_len for h in array_histories):
        return

    cont_indices = list(range(5, genome_len))  # continuous parameters after categorical genes
    plot_names = [param_names[i] if i < len(param_names) else f"p{i}" for i in cont_indices]

    cols = 3
    rows = int(math.ceil(len(cont_indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 2.5 * rows), sharex=True)
    axes = axes.flatten()

    for ax, idx, name in zip(axes, cont_indices, plot_names):
        for hist in array_histories:
            if idx >= hist.shape[1]:
                continue
            y = hist[:, idx]
            x = np.arange(y.size)
            ax.plot(x, y, alpha=0.25, lw=0.8)
        ax.set_ylabel(name.replace("_", " "))
        ax.grid(alpha=0.2)
    for ax in axes:
        ax.set_xlabel("Generation")

    fig.suptitle("Walker trajectories")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------


def _posterior_draws(
    df: pd.DataFrame,
    weights: np.ndarray,
    params: Sequence[str],
    nsamples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if len(df) == 0:
        raise ValueError("Empty catalogue")
    if nsamples <= 0:
        raise ValueError("nsamples must be >0")
    p = np.asarray(weights, dtype=float)
    p /= p.sum()
    idx = rng.choice(len(df), size=nsamples, replace=True, p=p)
    return df.iloc[idx][list(params)].reset_index(drop=True)


def _default_params(df: pd.DataFrame) -> List[str]:
    candidates = [
        "sigma_2",
        "t_1",
        "t_2",
        "infall_1",
        "infall_2",
        "sfe",
        "delta_sfe",
        "imf_upper",
        "mgal",
        "nb",
    ]
    return [c for c in candidates if c in df.columns]


def run_posterior_report(args: argparse.Namespace) -> Mapping[str, float]:
    results_path = Path(args.results).expanduser().resolve()
    if not results_path.is_file():
        raise SystemExit(f"Results file not found: {results_path}")

    df = pd.read_csv(results_path)
    if "fitness" not in df.columns:
        raise SystemExit("Expected a 'fitness' column in results CSV")

    df_sorted_idx = df.sort_values("fitness", ascending=True).reset_index()
    order = df_sorted_idx["index"].to_numpy()
    df_sorted = df_sorted_idx.drop(columns="index")
    weights, temperature, ess = compute_weights(df_sorted["fitness"].to_numpy(), args.temperature)

    params = _default_params(df_sorted)
    if args.params:
        params = [p for p in args.params if p in df_sorted.columns]
        if not params:
            raise SystemExit("None of the requested parameters are present in the catalogue")

    rng = np.random.default_rng(args.seed)
    posterior_draws = _posterior_draws(df_sorted, weights, params, args.nsamples, rng)

    base_dir = Path(args.output).expanduser().resolve() if args.output else results_path.parent / "analysis" / "posterior"
    _ensure_dir(base_dir)

    post_csv = base_dir / "posteriors.csv"
    posterior_draws.to_csv(post_csv, index=False)

    weights_csv = base_dir / "posterior_weights.csv"
    df_weights = df_sorted.assign(weight=weights)
    df_weights.to_csv(weights_csv, index=False)

    _save_corner(posterior_draws, base_dir / "corner.png")

    history_path = Path(args.history).expanduser().resolve() if args.history else results_path.parent / "walker_history.npz"
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

    pcard_path = Path(args.pcard).expanduser().resolve() if args.pcard else Path("bulge_pcard.txt").resolve()
    pcard = parse_inlist(str(pcard_path)) if pcard_path.is_file() else {}
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
        "posterior_draws": int(args.nsamples),
        "parameters": params,
    }

    with open(base_dir / "posterior_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Posterior analysis and focused plotting for GA outputs.")
    parser.add_argument("--results", default="GA/simulation_results.csv", help="Path to the GA results CSV.")
    parser.add_argument("--history", default="GA/walker_history.npz", help="Path to walker history NPZ (optional).")
    parser.add_argument("--pcard", default="bulge_pcard.txt", help="Path to the pcard/inlist file.")
    parser.add_argument("--output", help="Directory for the posterior outputs (default: <results>/analysis/posterior)")
    parser.add_argument("--params", nargs="*", help="Subset of parameters to include in posteriors.csv")
    parser.add_argument("--nsamples", type=int, default=5000, help="Number of posterior draws to resample (default 5000)")
    parser.add_argument("--temperature", type=float, default=None, help="Manual temperature for exponential weighting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for resampling")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_posterior_report(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()

