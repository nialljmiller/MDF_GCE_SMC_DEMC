
import numpy as np
import matplotlib.pyplot as plt
from .style import save

def _best_params_from_results(results_df, GalGA):
    if results_df is not None and len(results_df) > 0:
        bm = results_df.iloc[0]
        return (bm["sigma_2"], bm["t_2"], bm["infall_2"])
    r = GalGA.results[0]
    return (r[5], r[7], r[9])

def plot_mdf_family(GalGA, feh, normalized_count, results_df=None, save_path=None):
    if save_path is None:
        save_path = GalGA.output_path + "MDF_multiple_results.png"

    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(9, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )

    best_params = _best_params_from_results(results_df, GalGA)
    best_x, best_y = None, None
    alpha = 10 / max(1, len(GalGA.results))

    for (x, y), label, res in zip(GalGA.mdf_data, GalGA.labels, GalGA.results):
        params = (res[5], res[7], res[9])
        if all(abs(p - b) < 1e-5 for p, b in zip(params, best_params)):
            best_x = np.array(x)
            best_y = np.array(y)
            ax_main.plot(x, y, color="C3", linewidth=2.5, zorder=10, label="Best Model")
        else:
            ax_main.plot(x, y, linewidth=1, color="gray", alpha=alpha)

    ax_main.plot(feh, normalized_count, "x", ms=8, color="k", zorder=11, label="Observations")
    ax_main.set_ylabel("Normalized Number Density")
    ax_main.set_xlim(-2, 1)
    ax_main.legend(frameon=False, loc="upper left")
    ax_main.tick_params(axis="x", labelbottom=False)

    if best_x is not None and best_y is not None:
        model_min, model_max = np.min(best_x), np.max(best_x)
        mask = (feh >= model_min) & (feh <= model_max)
        feh_f = feh[mask]; obs_f = normalized_count[mask]
        model_f = np.interp(feh_f, best_x, best_y)
        residuals = model_f - obs_f
        ax_res.plot(feh_f, residuals, "rx", ms=6, alpha=0.8)
        ax_res.axhline(0, color="k", ls="--", alpha=0.5)
        if residuals.size:
            s = np.std(residuals)
            ax_res.set_ylim(-3*s, 3*s)
            ax_res.text(0.02, 0.9, f"RMS = {np.sqrt(np.mean(residuals**2)):.3f}",
                        transform=ax_res.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))

    ax_res.set_xlabel("[Fe/H]")
    ax_res.set_ylabel("Model - Obs")
    ax_res.set_xlim(-2, 1)

    save(fig, save_path)
    return fig

def plot_mdf_single(feh, obs_y, model_x, model_y, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(model_x, model_y, lw=2, label="Model")
    ax.plot(feh, obs_y, "x", ms=6, label="Obs", color="k")
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("Normalized Number Density")
    ax.set_xlim(-2, 1)
    ax.legend(frameon=False)
    save(fig, save_path)
    return fig
