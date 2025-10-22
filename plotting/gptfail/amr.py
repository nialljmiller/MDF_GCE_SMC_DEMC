
import numpy as np
import matplotlib.pyplot as plt
from .style import save

def plot_amr(model_age_yr, model_feh, obs_age_yr, obs_feh, obs_feh_err, save_path):
    # Age-metallicity relation: model curve + observed points with errors.
    age_gyr_m = np.array(model_age_yr) / 1e9
    age_gyr_o = np.array(obs_age_yr) / 1e9

    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(age_gyr_m, model_feh, lw=2, label="Model")
    ax.errorbar(age_gyr_o, obs_feh, yerr=obs_feh_err, fmt="o", ms=3, lw=1, alpha=0.8, label="Obs")
    ax.set_xlabel("Age (Gyr)")
    ax.set_ylabel("[Fe/H]")
    ax.legend(frameon=False)
    save(fig, save_path)
    return fig
