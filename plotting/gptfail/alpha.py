
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from .style import save

def plot_alpha_histograms(obs_dict, model_dict, save_path, bins=25):
    elts = list(obs_dict.keys())
    n = len(elts)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(6*ncols, 4*nrows))
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.3, hspace=0.4)

    for idx, elt in enumerate(elts):
        ax = fig.add_subplot(gs[idx])
        ax.hist(obs_dict[elt], bins=bins, histtype="stepfilled", alpha=0.3, color="C0", label="Obs")
        Ys = [np.asarray(arr[idx], float) for arr in model_dict.values()]
        alpha_mod = np.nanmean(np.vstack(Ys), axis=0)
        ax.hist(alpha_mod, bins=bins, histtype="step", lw=2, color="C1", label="Model")
        ax.set_title(f"{elt} Distribution")
        ax.set_xlabel(elt); ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize="small")

    save(fig, save_path)
    return fig



def plot_alpha_tracks(alpha_arrs, save_path, elements=("Mg","Si","Ca","Ti")):
    """Plot model [α/Fe] vs [Fe/H] tracks for a single model (2×2 panels)."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, name, arr in zip(axes.ravel(), elements, alpha_arrs):
        x, y = np.asarray(arr[0], float), np.asarray(arr[1], float)
        ax.plot(x, y, lw=2)
        ax.set_xlabel("[Fe/H]")
        ax.set_ylabel(f"[{name}/Fe]")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, save_path)
    return fig
