# standalone_plots/obs.py
import numpy as np
from pathlib import Path

# -----------------------------
# MDF (two columns: [Fe/H], counts)
# -----------------------------
def load_observed_mdf(path: str):
    arr = np.loadtxt(path, usecols=(0, 1))
    x = arr[:, 0].astype(float)
    y = arr[:, 1].astype(float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    order = np.argsort(x)
    x, y = x[order], y[order]
    ymax = y.max()
    if ymax > 0:
        y = y / ymax  # legacy behavior
    return x, y


# -----------------------------
# Bensby dataset loader (TSV)
# Returns a dict of numpy arrays
# -----------------------------
def load_observational_data(path_hint: str | None = None):
    """
    Load Bensby comparison data from Bensby_Data.tsv and return dict of arrays:
      fe_h, age_joyce, age_bensby, mg_fe, si_fe, ca_fe, ti_fe
    Search order: path_hint/, data/, ../data/
    """
    candidates = []
    if path_hint:
        candidates.append(Path(path_hint))
    candidates.extend([Path("data"), Path("../data")])

    data_path = None
    for root in candidates:
        p = root / "Bensby_Data.tsv"
        if p.exists():
            data_path = p
            break
    if data_path is None:
        raise FileNotFoundError("Bensby_Data.tsv not found in path_hint/, data/, or ../data/")

    # Read TSV with whitespace splitting (no pandas dependency)
    with data_path.open() as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    header = lines[0].split()
    idx = {name: header.index(name) for name in header}

    # Expected column names (must exist in header)
    required = ["[Fe/H]", "[Mg/Fe]", "[Si/Fe]", "[Ca/Fe]", "[Ti/Fe]", "Joyce_age", "Bensby"]
    for col in required:
        if col not in idx:
            raise KeyError(f"Required column '{col}' not found in {data_path}")

    fe_h, age_joyce, age_bensby = [], [], []
    mg_fe, si_fe, ca_fe, ti_fe = [], [], [], []

    for line in lines[1:]:
        toks = line.split()
        fe_h.append(float(toks[idx["[Fe/H]"]]))
        mg_fe.append(float(toks[idx["[Mg/Fe]"]]))
        si_fe.append(float(toks[idx["[Si/Fe]"]]))
        ca_fe.append(float(toks[idx["[Ca/Fe]"]]))
        ti_fe.append(float(toks[idx["[Ti/Fe]"]]))
        age_joyce.append(float(toks[idx["Joyce_age"]]))
        age_bensby.append(float(toks[idx["Bensby"]]))

    return {
        "fe_h":      np.asarray(fe_h, dtype=float),
        "age_joyce": np.asarray(age_joyce, dtype=float),
        "age_bensby":np.asarray(age_bensby, dtype=float),
        "mg_fe":     np.asarray(mg_fe, dtype=float),
        "si_fe":     np.asarray(si_fe, dtype=float),
        "ca_fe":     np.asarray(ca_fe, dtype=float),
        "ti_fe":     np.asarray(ti_fe, dtype=float),
    }


# -----------------------------------------------------------
# AGE–METALLICITY RELATION (Joyce & Bensby)
# -----------------------------------------------------------
def load_observed_amr(path_hint: str | None = None):
    """
    Return (Fe_H_array, age_Joyce_array, age_Bensby_array)
    """
    d = load_observational_data(path_hint)
    return d["fe_h"], d["age_joyce"], d["age_bensby"]


# -----------------------------------------------------------
# ALPHA-ELEMENT TRENDS (Mg, Si, Ca, Ti)
# -----------------------------------------------------------
def load_observed_alpha(path_hint: str | None = None):
    """
    Return tuple of 4 pairs:
      ( (Fe_H, Mg/Fe), (Fe_H, Si/Fe), (Fe_H, Ca/Fe), (Fe_H, Ti/Fe) )
    """
    d = load_observational_data(path_hint)
    feh = d["fe_h"]
    return ( (feh, d["mg_fe"]),
             (feh, d["si_fe"]),
             (feh, d["ca_fe"]),
             (feh, d["ti_fe"]) )
