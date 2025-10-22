
import numpy as np
import matplotlib.pyplot as plt
from .style import save

def plot_sfr_history(bulge_dict, save_path):
    fig, ax = plt.subplots(figsize=(6,5))
    for label, model in bulge_dict.items():
        age_gyr = np.array(model.inner.history.age) / 1e9
        sfr = np.array(model.inner.history.sfr_abs)
        ax.plot(age_gyr, sfr, label=label)
    ax.set_xlabel("Age (Gyr)")
    ax.set_ylabel(r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]")
    ax.legend(frameon=False, fontsize="small")
    save(fig, save_path)
    return fig

def plot_mass_evolution(bulge_dict, save_path):
    fig, ax = plt.subplots(figsize=(6,5))
    for label, model in bulge_dict.items():
        age_gyr = np.array(model.inner.history.age) / 1e9
        m_locked = np.array(getattr(model.inner.history, "m_locked", []))
        m_gas = np.array(getattr(model.inner.history, "m_gas_exp", getattr(model.inner.history, "m_gas", [])))
        ax.plot(age_gyr, m_locked + m_gas, label=label)
    ax.set_xlabel("Age (Gyr)")
    ax.set_ylabel(r"Bulge Mass [$M_\odot$]")
    ax.axhline(2e10, ls="--", color="k", label=r"2\times10^{10} $M_\odot$")
    ax.legend(frameon=False, fontsize="small")
    save(fig, save_path)
    return fig
