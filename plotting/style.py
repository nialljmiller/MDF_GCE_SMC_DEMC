
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def use_paper_style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 18,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 1.5,
    })

# Always-on default style
use_paper_style()

def save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
