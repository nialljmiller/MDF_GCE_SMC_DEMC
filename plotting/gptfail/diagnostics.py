
import numpy as np
import matplotlib.pyplot as plt
import corner
from .style import save

def corner_from_df(df, columns, out_png, weights=None, bins=40, smooth=0.9):
    data = df[columns].to_numpy()
    fig = corner.corner(
        data,
        labels=[c.replace("_", " ") for c in columns],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3g",
        bins=bins,
        smooth=smooth,
        weights=weights
    )
    save(fig, out_png)
    return fig

def plot_loss_vs_gen(gens, loss, save_path, ylabel="Loss"):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(gens, loss, lw=1.75)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    save(fig, save_path)
    return fig

def plot_walker_paths(walker_history, param_index, param_name, save_path):
    # walker_history: dict[int -> array_like of individuals per generation]
    # Each history entry is an array of shape (n_gen, n_params).
    fig, ax = plt.subplots(figsize=(8,5))
    for wid, hist in walker_history.items():
        H = np.asarray(hist)
        ax.plot(np.arange(len(H)), H[:, param_index], alpha=0.6)
    ax.set_xlabel("Generation")
    ax.set_ylabel(param_name)
    save(fig, save_path)
    return fig
