from __future__ import annotations

"""Utilities for loading shared plotting data.

This module centralises the logic for loading simulation results and
observational comparison data so that every plot operates on a consistent
set of inputs.  Historically each plotting script loaded the CSV and picked
its own notion of the "best" metric which easily drifted apart.  By routing
everything through this helper we guarantee that all plots share the same
DataFrame ordering and loss column selection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "ObservationalData",
    "load_observational_data",
    "load_results_dataframe",
    "select_metric_column",
]

# Preference order for potential loss columns.  The first one that exists is
# used as the controlling column for ranking models.
_PREFERRED_METRICS: Tuple[str, ...] = (
    "fitness",
    "confidence",
    "wrmse",
    "mae",
    "loss",
    "total_loss",
    "mape",
    "huber",
    "ks",
    "ensemble",
)


@dataclass
class ObservationalData:
    """Container for the observational datasets used by the plots."""

    fe_h: np.ndarray
    age_joyce: np.ndarray
    age_bensby: np.ndarray
    mg_fe: np.ndarray
    si_fe: np.ndarray
    ca_fe: np.ndarray
    ti_fe: np.ndarray

    def as_alpha_tuple(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.mg_fe, self.si_fe, self.ca_fe, self.ti_fe


def select_metric_column(df: pd.DataFrame, preferred: Iterable[str] | None = None) -> str:
    """Return the loss/metric column that should be used for ranking.

    Parameters
    ----------
    df:
        DataFrame containing the simulation results.
    preferred:
        Optional explicit priority order.  When ``None`` the module level
        :data:`_PREFERRED_METRICS` is used.

    Returns
    -------
    str
        Name of the column to use.  A :class:`ValueError` is raised when no
        suitable column is available.
    """

    order = tuple(preferred or _PREFERRED_METRICS)
    for candidate in order:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "None of the preferred metric columns were found in the results CSV."
    )


def load_results_dataframe(
    results_file: str,
    preferred_metrics: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load the simulation results and standardise the ranking column.

    The returned DataFrame is filtered to rows with finite metric values and
    sorted in ascending order of the selected metric so that ``iloc[0]`` always
    corresponds to the best model according to that metric.  A copy of the
    original column is preserved; when the controlling column is not named
    ``"fitness"`` an alias is created to avoid downstream KeyError issues.
    """

    df = pd.read_csv(results_file)
    metric = select_metric_column(df, preferred_metrics)

    # Always keep a numeric version of the metric for sorting/comparison.
    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df = df.replace({np.inf: np.nan, -np.inf: np.nan})
    mask = np.isfinite(df[metric])
    df = df.loc[mask].sort_values(metric, ascending=True).reset_index(drop=True)

    return df, metric


def load_observational_data(path_hint: str | None = None) -> ObservationalData:
    """Load the Bensby et al. observational comparison dataset.

    Parameters
    ----------
    path_hint:
        Optional base directory.  When omitted the function tries ``data`` and
        ``../data`` relative to the current working directory.
    """

    possible_roots = []
    if path_hint:
        possible_roots.append(Path(path_hint))
    possible_roots.extend((Path("data"), Path("../data")))

    data_path = None
    for root in possible_roots:
        candidate = root / "Bensby_Data.tsv"
        if candidate.exists():
            data_path = candidate
            break
    if data_path is None:
        raise FileNotFoundError("Unable to locate Bensby_Data.tsv in the data directories.")

    with data_path.open() as fh:
        lines = fh.readlines()

    header = lines[0].split()
    idx = {name: header.index(name) for name in header}

    fe_h, age_joyce, age_bensby = [], [], []
    si_fe, ca_fe, mg_fe, ti_fe = [], [], [], []

    for line in lines[1:]:
        toks = line.split()
        fe_h.append(float(toks[idx["[Fe/H]"]]))
        mg_fe.append(float(toks[idx["[Mg/Fe]"]]))
        si_fe.append(float(toks[idx["[Si/Fe]"]]))
        ca_fe.append(float(toks[idx["[Ca/Fe]"]]))
        ti_fe.append(float(toks[idx["[Ti/Fe]"]]))
        age_joyce.append(float(toks[idx["Joyce_age"]]))
        age_bensby.append(float(toks[idx["Bensby"]]))

    return ObservationalData(
        fe_h=np.asarray(fe_h, dtype=float),
        age_joyce=np.asarray(age_joyce, dtype=float),
        age_bensby=np.asarray(age_bensby, dtype=float),
        mg_fe=np.asarray(mg_fe, dtype=float),
        si_fe=np.asarray(si_fe, dtype=float),
        ca_fe=np.asarray(ca_fe, dtype=float),
        ti_fe=np.asarray(ti_fe, dtype=float),
    )
