"""
Enhanced Omni Plot with Detailed Infall Physics Information

This module extends the basic omni plot to include comprehensive information
about the two-infall mechanism physics, including:
- Infall rate evolution over time
- Star formation history
- Gas mass evolution
- Stellar mass buildup
- Infall mass contributions and ratios
- Gas fraction evolution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

from plotting.style import *
use_paper_style()






def extract_infall_physics(omega_model, best_params):
    """
    Extract detailed infall physics from the omega model.

    Parameters
    ----------
    omega_model : omega object
        The best-fit omega model
    best_params : tuple
        (sigma_2, t_2, infall_2) for the second infall

    Returns
    -------
    physics_data : dict
        Dictionary containing all physics arrays
    """
    physics_data = {}

    # Time array (convert from years to Gyr)
    if hasattr(omega_model, 'history') and hasattr(omega_model.history, 'age'):
        time_yr = np.array(omega_model.history.age)
        physics_data['time_gyr'] = time_yr / 1e9
    elif hasattr(omega_model, 'inner') and hasattr(omega_model.inner, 'history'):
        time_yr = np.array(omega_model.inner.history.age)
        physics_data['time_gyr'] = time_yr / 1e9
    else:
        # Fallback: create time array
        physics_data['time_gyr'] = np.linspace(0, 13, 1000)

    # Extract infall parameters
    sigma_2, t_2, infall_2 = best_params

    # Try to get infall parameters from model
    if hasattr(omega_model, 'inner'):
        inner = omega_model.inner

        # First infall parameters (from omega initialization)
        if hasattr(inner, 'in_out_control'):
            physics_data['has_infall_data'] = True
        else:
            physics_data['has_infall_data'] = False

        # Get SFH
        if hasattr(inner.history, 'sfr'):
            physics_data['sfr'] = np.array(inner.history.sfr)

        # Get gas mass
        if hasattr(inner.history, 'm_gas'):
            physics_data['m_gas'] = np.array(inner.history.m_gas)

        # Get stellar mass (locked mass)
        if hasattr(inner.history, 'm_locked'):
            physics_data['m_locked'] = np.array(inner.history.m_locked)

        # Get infall rate if available
        if hasattr(inner.history, 'infall_rate'):
            physics_data['infall_rate'] = np.array(inner.history.infall_rate)

    # Store parameters for reconstruction
    physics_data['sigma_2'] = sigma_2
    physics_data['t_2'] = t_2
    physics_data['infall_2'] = infall_2

    return physics_data


def plot_infall_rates(ax, physics_data):
    """Plot infall rate vs time for both episodes."""
    time = physics_data['time_gyr']

    if 'infall_rate' in physics_data:
        # Use actual infall rate from model
        infall_rate = physics_data['infall_rate']
        ax.plot(time, infall_rate, 'k-', lw=1.5, label='Total infall')
    else:
        # Reconstruct from parameters (if available)
        # This would require access to the infall function parameters
        ax.text(0.5, 0.5, 'Infall rate\ndata unavailable',
               ha='center', va='center', transform=ax.transAxes, color='0.5')

    ax.set_xlabel('Time (Gyr)')
    ax.set_ylabel(r'Infall rate (M$_\odot$ yr$^{-1}$)')
    ax.set_xlim(0, 13)
    ax.legend(loc='best', fontsize=8)
    ax.set_title('Infall Rate History', fontsize=10, pad=8)

    # Add text annotations for infall episodes
    if 't_2' in physics_data:
        t2 = physics_data['t_2']
        ax.axvline(t2, color='crimson', ls='--', lw=1, alpha=0.5, label=f'2nd infall (t={t2:.2f} Gyr)')


def plot_sfh(ax, physics_data):
    """Plot star formation history."""
    time = physics_data['time_gyr']

    if 'sfr' in physics_data:
        sfr = physics_data['sfr']
        ax.plot(time, sfr, 'b-', lw=1.5)

        # Find and mark peak SFR
        peak_idx = np.argmax(sfr)
        peak_time = time[peak_idx]
        peak_sfr = sfr[peak_idx]
        ax.plot(peak_time, peak_sfr, 'r*', ms=10, label=f'Peak at {peak_time:.2f} Gyr')

        # Calculate and display integrated SFR
        total_stars = np.trapz(sfr, time * 1e9)  # Convert Gyr to yr
        ax.text(0.97, 0.97, f'Total: {total_stars:.2e} M$_\\odot$',
               transform=ax.transAxes, ha='right', va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'SFH data\nunavailable',
               ha='center', va='center', transform=ax.transAxes, color='0.5')

    ax.set_xlabel('Time (Gyr)')
    ax.set_ylabel(r'SFR (M$_\odot$ yr$^{-1}$)')
    ax.set_xlim(0, 13)
    ax.legend(loc='best', fontsize=8)
    ax.set_title('Star Formation History', fontsize=10, pad=8)


def plot_gas_evolution(ax, physics_data):
    """Plot gas mass evolution over time."""
    time = physics_data['time_gyr']

    if 'm_gas' in physics_data:
        m_gas = physics_data['m_gas']
        ax.plot(time, m_gas, 'g-', lw=1.5, label='Gas mass')

        # Calculate gas fraction at end
        if 'm_locked' in physics_data:
            m_locked = physics_data['m_locked']
            final_gas_frac = m_gas[-1] / (m_gas[-1] + m_locked[-1])
            ax.text(0.97, 0.97, f'Final gas fraction: {final_gas_frac:.1%}',
                   transform=ax.transAxes, ha='right', va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'Gas mass\ndata unavailable',
               ha='center', va='center', transform=ax.transAxes, color='0.5')

    ax.set_xlabel('Time (Gyr)')
    ax.set_ylabel(r'Gas mass (M$_\odot$)')
    ax.set_xlim(0, 13)
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=8)
    ax.set_title('Gas Mass Evolution', fontsize=10, pad=8)


def plot_stellar_mass(ax, physics_data):
    """Plot stellar mass buildup over time."""
    time = physics_data['time_gyr']

    if 'm_locked' in physics_data:
        m_locked = physics_data['m_locked']
        ax.plot(time, m_locked, 'orange', lw=1.5, label='Stellar mass')

        # Mark final stellar mass
        final_mass = m_locked[-1]
        ax.text(0.97, 0.03, f'Final: {final_mass:.2e} M$_\\odot$',
               transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='bisque', alpha=0.5))

        # Calculate mean stellar age (mass-weighted)
        if 'sfr' in physics_data:
            sfr = physics_data['sfr']
            # Approximate mean age
            stellar_ages = time[-1] - time
            mean_age = np.trapz(sfr * stellar_ages, time) / np.trapz(sfr, time)
            ax.text(0.03, 0.97, f'Mean age: {mean_age:.2f} Gyr',
                   transform=ax.transAxes, ha='left', va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='bisque', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'Stellar mass\ndata unavailable',
               ha='center', va='center', transform=ax.transAxes, color='0.5')

    ax.set_xlabel('Time (Gyr)')
    ax.set_ylabel(r'Stellar mass (M$_\odot$)')
    ax.set_xlim(0, 13)
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=8)
    ax.set_title('Stellar Mass Buildup', fontsize=10, pad=8)



def extract_comprehensive_physics(omega_model, params_dict):
    """Extract all available physics data from omega model."""
    physics = {}
    
    # Get the inner model
    if hasattr(omega_model, 'inner'):
        inner = omega_model.inner
    else:
        inner = omega_model
    
    # Time array
    if hasattr(inner, 'history') and hasattr(inner.history, 'age'):
        physics['time_yr'] = np.array(inner.history.age)
        physics['time_gyr'] = physics['time_yr'] / 1e9
    else:
        physics['time_gyr'] = np.linspace(0, 13, 1000)
        physics['time_yr'] = physics['time_gyr'] * 1e9
    
    # Star formation history
    if hasattr(inner, 'history') and hasattr(inner.history, 'sfr'):
        physics['sfr'] = np.array(inner.history.sfr)
    
    # Gas mass
    if hasattr(inner, 'history') and hasattr(inner.history, 'm_gas'):
        physics['m_gas'] = np.array(inner.history.m_gas)
    
    # Stellar mass
    if hasattr(inner, 'history') and hasattr(inner.history, 'm_locked'):
        physics['m_locked'] = np.array(inner.history.m_locked)
    
    # Infall rate
    if hasattr(inner, 'history') and hasattr(inner.history, 'infall_rate'):
        physics['infall_rate'] = np.array(inner.history.infall_rate)
    
    # Outflow rate
    if hasattr(inner, 'history') and hasattr(inner.history, 'outflow_rate'):
        physics['outflow_rate'] = np.array(inner.history.outflow_rate)
    
    # Store parameters
    physics['params'] = params_dict
    
    return physics


def reconstruct_two_infall_rate(time_gyr, params_dict):
    """
    Reconstruct the two-infall rate from parameters.
    
    Assumes exponential infall: I(t) = A * exp(-t/tau)
    """
    # First infall
    t1 = params_dict.get('t_1', 0.0)
    tau1 = params_dict.get('infall_1', 0.1)
    sigma1 = params_dict.get('sigma_1', 1.0)
    
    # Second infall
    t2 = params_dict.get('t_2', 1.0)
    tau2 = params_dict.get('infall_2', 1.0)
    sigma2 = params_dict.get('sigma_2', 1.0)
    
    # Normalization (these would need to be extracted or estimated)
    A1 = 1.0  # Placeholder
    A2 = sigma2  # Use sigma as proxy for normalization
    
    # Calculate infall rates
    infall_1 = np.where(time_gyr >= t1, A1 * np.exp(-(time_gyr - t1) / tau1), 0)
    infall_2 = np.where(time_gyr >= t2, A2 * np.exp(-(time_gyr - t2) / tau2), 0)
    
    return infall_1, infall_2


def plot_infall_rates_detailed(ax, physics, params_dict):
    """Plot separate infall rates for each episode."""
    time = physics['time_gyr']
    
    if 'infall_rate' in physics:
        # Try to decompose total infall into components
        infall_1, infall_2 = reconstruct_two_infall_rate(time, params_dict)
        
        # Normalize to match total if available
        total_infall = physics['infall_rate']
        
        ax.plot(time, total_infall, 'k-', lw=2, label='Total infall', zorder=3)
        ax.fill_between(time, 0, total_infall, color='0.8', alpha=0.3)
        
        # Mark infall episodes
        t2 = params_dict.get('t_2', 1.0)
        ax.axvline(t2, color='crimson', ls='--', lw=1.5, alpha=0.7, 
                  label=f'2nd infall start ({t2:.2f} Gyr)')
    else:
        # Reconstruct from parameters
        infall_1, infall_2 = reconstruct_two_infall_rate(time, params_dict)
        ax.plot(time, infall_1, 'b-', lw=1.5, label='1st infall')
        ax.plot(time, infall_2, 'r-', lw=1.5, label='2nd infall')
        ax.plot(time, infall_1 + infall_2, 'k--', lw=2, label='Total')
    
    ax.set_xlabel('Time (Gyr)', fontsize=10)
    ax.set_ylabel(r'Infall rate (M$_\odot$ yr$^{-1}$)', fontsize=10)
    ax.set_xlim(0, 13)
    ax.legend(loc='best', fontsize=8)
    ax.set_title('Infall Rate Components', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)


def plot_cumulative_infall(ax, physics, params_dict):
    """Plot cumulative mass from each infall episode."""
    time = physics['time_gyr']
    
    if 'infall_rate' in physics:
        infall_rate = physics['infall_rate']
        # Integrate to get cumulative mass
        dt = np.diff(time, prepend=0) * 1e9  # Convert to years
        cumulative_total = np.cumsum(infall_rate * dt)
        
        ax.plot(time, cumulative_total, 'k-', lw=2, label='Total accreted')
        ax.fill_between(time, 0, cumulative_total, color='0.8', alpha=0.3)
        
        # Try to estimate contributions from each episode
        t2 = params_dict.get('t_2', 1.0)
        idx_t2 = np.argmin(np.abs(time - t2))
        mass_from_1 = cumulative_total[idx_t2] if idx_t2 < len(cumulative_total) else 0
        mass_from_2 = cumulative_total[-1] - mass_from_1
        
        # Add text box with masses
        textstr = f'1st infall: {mass_from_1:.2e} M$_\\odot$\n'
        textstr += f'2nd infall: {mass_from_2:.2e} M$_\\odot$\n'
        textstr += f'Ratio M2/M1: {mass_from_2/mass_from_1:.2f}' if mass_from_1 > 0 else 'Ratio: N/A'
        
        ax.text(0.97, 0.50, textstr, transform=ax.transAxes,
               fontsize=8, va='center', ha='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.set_xlabel('Time (Gyr)', fontsize=10)
    ax.set_ylabel(r'Cumulative mass (M$_\odot$)', fontsize=10)
    ax.set_xlim(0, 13)
    ax.set_yscale('log')
    ax.set_title('Cumulative Infall Mass', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)


def plot_sfh_detailed(ax, physics):
    """Plot detailed star formation history with annotations."""
    time = physics['time_gyr']
    
    if 'sfr' in physics:
        sfr = physics['sfr']
        ax.plot(time, sfr, 'b-', lw=2)
        ax.fill_between(time, 0, sfr, color='blue', alpha=0.2)
        
        # Mark peak
        peak_idx = np.argmax(sfr)
        ax.plot(time[peak_idx], sfr[peak_idx], 'r*', ms=12, 
               label=f'Peak: {time[peak_idx]:.2f} Gyr')
        
        # Calculate total stellar mass formed
        dt = np.diff(time, prepend=0) * 1e9
        total_formed = np.sum(sfr * dt)
        
        ax.text(0.03, 0.97, f'Total formed:\n{total_formed:.2e} M$_\\odot$',
               transform=ax.transAxes, fontsize=8, va='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax.set_xlabel('Time (Gyr)', fontsize=10)
    ax.set_ylabel(r'SFR (M$_\odot$ yr$^{-1}$)', fontsize=10)
    ax.set_xlim(0, 13)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Star Formation History', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)


def plot_mass_evolution(ax, physics):
    """Plot gas and stellar mass evolution."""
    time = physics['time_gyr']
    
    if 'm_gas' in physics and 'm_locked' in physics:
        m_gas = physics['m_gas']
        m_stellar = physics['m_locked']
        
        ax.plot(time, m_gas, 'g-', lw=2, label='Gas')
        ax.plot(time, m_stellar, 'orange', lw=2, label='Stars')
        ax.plot(time, m_gas + m_stellar, 'k--', lw=1.5, label='Total', alpha=0.7)
        
        # Gas fraction evolution
        ax2 = ax.twinx()
        gas_frac = m_gas / (m_gas + m_stellar)
        ax2.plot(time, gas_frac, 'r:', lw=1.5, alpha=0.6, label='Gas fraction')
        ax2.set_ylabel('Gas fraction', fontsize=9, color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim(0, 1)
    
    ax.set_xlabel('Time (Gyr)', fontsize=10)
    ax.set_ylabel(r'Mass (M$_\odot$)', fontsize=10)
    ax.set_xlim(0, 13)
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('Mass Evolution', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)


def plot_sfe_evolution(ax, physics, params_dict):
    """Plot star formation efficiency evolution."""
    time = physics['time_gyr']
    
    if 'sfr' in physics and 'm_gas' in physics:
        sfr = physics['sfr']
        m_gas = physics['m_gas']
        
        # SFE = SFR / M_gas (units: yr^-1)
        sfe = np.where(m_gas > 0, sfr / m_gas, 0)
        sfe_gyr = sfe * 1e9  # Convert to Gyr^-1
        
        ax.plot(time, sfe_gyr, 'purple', lw=2)
        
        # Mark change at second infall
        t2 = params_dict.get('t_2', 1.0)
        delta_sfe = params_dict.get('delta_sfe', 1.0)
        ax.axvline(t2, color='crimson', ls='--', alpha=0.5)
        
        # Add parameter values
        sfe_param = params_dict.get('sfe', None)
        if sfe_param:
            textstr = f'SFE param: {sfe_param:.1f} Gyr$^{{-1}}$\n'
            textstr += f'Δ SFE: {delta_sfe:.2f}'
            ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
                   fontsize=8, va='top', ha='right',
                   bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
    
    ax.set_xlabel('Time (Gyr)', fontsize=10)
    ax.set_ylabel(r'SFE (Gyr$^{-1}$)', fontsize=10)
    ax.set_xlim(0, 13)
    ax.set_title('Star Formation Efficiency', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)


def plot_gas_depletion(ax, physics):
    """Plot gas depletion timescale evolution."""
    time = physics['time_gyr']
    
    if 'sfr' in physics and 'm_gas' in physics:
        sfr = physics['sfr']
        m_gas = physics['m_gas']
        
        # Depletion time = M_gas / SFR (in Gyr)
        t_dep = np.where(sfr > 0, (m_gas / sfr) / 1e9, np.nan)
        
        # Clip extreme values for plotting
        t_dep_clipped = np.clip(t_dep, 0, 20)
        
        ax.plot(time, t_dep_clipped, 'teal', lw=2)
        ax.fill_between(time, 0, t_dep_clipped, color='teal', alpha=0.2)
        
        # Add reference lines
        ax.axhline(1, color='k', ls=':', alpha=0.5, label='1 Gyr')
        ax.axhline(2, color='k', ls=':', alpha=0.5, label='2 Gyr')
    
    ax.set_xlabel('Time (Gyr)', fontsize=10)
    ax.set_ylabel(r'$\tau_{dep}$ (Gyr)', fontsize=10)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.set_title('Gas Depletion Timescale', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def plot_mass_budget(ax, physics):
    """Plot final mass budget as pie chart."""
    if 'm_gas' in physics and 'm_locked' in physics:
        final_gas = physics['m_gas'][-1]
        final_stars = physics['m_locked'][-1]
        
        # Estimate outflows (if available)
        if 'outflow_rate' in physics:
            dt = np.diff(physics['time_gyr'], prepend=0) * 1e9
            total_outflow = np.sum(physics['outflow_rate'] * dt)
        else:
            # Estimate from mass conservation
            if 'infall_rate' in physics:
                dt = np.diff(physics['time_gyr'], prepend=0) * 1e9
                total_infall = np.sum(physics['infall_rate'] * dt)
                total_outflow = total_infall - final_gas - final_stars
                total_outflow = max(0, total_outflow)  # Can't be negative
            else:
                total_outflow = 0
        
        # Create pie chart
        sizes = [final_stars, final_gas, total_outflow]
        labels = [f'Stars\n{final_stars:.2e} M$_\\odot$',
                 f'Gas\n{final_gas:.2e} M$_\\odot$',
                 f'Outflows\n{total_outflow:.2e} M$_\\odot$']
        colors = ['orange', 'green', 'red']
        explode = (0.05, 0, 0)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
              autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title('Final Mass Budget', fontsize=11, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Mass budget\ndata unavailable',
               ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')


def plot_key_metrics(ax, physics, params_dict):
    """Display key physics metrics as text."""
    ax.axis('off')
    
    metrics_text = "KEY PHYSICS METRICS\n" + "="*30 + "\n\n"
    
    # Infall parameters
    metrics_text += "INFALL PARAMETERS:\n"
    t1 = params_dict.get('t_1', 0.0)
    tau1 = params_dict.get('infall_1', 0.1)
    t2 = params_dict.get('t_2', 1.0)
    tau2 = params_dict.get('infall_2', 1.0)
    
    metrics_text += f"  1st infall: t={t1:.3f} Gyr, τ={tau1:.3f} Gyr\n"
    metrics_text += f"  2nd infall: t={t2:.3f} Gyr, τ={tau2:.3f} Gyr\n"
    metrics_text += f"  Delay: {t2-t1:.3f} Gyr\n"
    metrics_text += f"  Timescale ratio: {tau2/tau1:.2f}\n\n"
    
    # Star formation
    if 'sfr' in physics:
        sfr = physics['sfr']
        time = physics['time_gyr']
        peak_idx = np.argmax(sfr)
        metrics_text += "STAR FORMATION:\n"
        metrics_text += f"  Peak SFR: {sfr[peak_idx]:.2e} M☉/yr\n"
        metrics_text += f"  Peak time: {time[peak_idx]:.3f} Gyr\n"
        
        dt = np.diff(time, prepend=0) * 1e9
        total_formed = np.sum(sfr * dt)
        metrics_text += f"  Total formed: {total_formed:.2e} M☉\n\n"
    
    # Final state
    if 'm_gas' in physics and 'm_locked' in physics:
        final_gas = physics['m_gas'][-1]
        final_stars = physics['m_locked'][-1]
        gas_frac = final_gas / (final_gas + final_stars)
        
        metrics_text += "FINAL STATE:\n"
        metrics_text += f"  Stellar mass: {final_stars:.2e} M☉\n"
        metrics_text += f"  Gas mass: {final_gas:.2e} M☉\n"
        metrics_text += f"  Gas fraction: {gas_frac:.1%}\n\n"
    
    # Mean stellar age
    if 'sfr' in physics:
        stellar_ages = time[-1] - time
        mean_age = np.trapz(sfr * stellar_ages, time) / np.trapz(sfr, time)
        metrics_text += f"STELLAR POPULATION:\n"
        metrics_text += f"  Mean age: {mean_age:.2f} Gyr\n"
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
           fontsize=9, va='top', ha='left', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))






















def plot_omni_info_figure(
    GalGA,
    Fe_H,
    age_Joyce,
    age_Bensby,
    Mg_Fe,
    Si_Fe,
    Ca_Fe,
    Ti_Fe,
    feh_mdf,
    normalized_count_mdf,
    results_df=None,
    save_path=None,
    metric_col: str = 'fitness',
):
    """
    Create a dashboard showing the best-fit model parameters and performance
    across all key observational diagnostics.

    Parameters:
    -----------
    GalGA : Galactic Evolution GA object
    Fe_H, age_Joyce, age_Bensby : observational age-metallicity data
    Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe : observational alpha element data
    feh_mdf, normalized_count_mdf : observational MDF data
    results_df : DataFrame with model results
    save_path : output file path
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import CubicSpline, interp1d
    from scipy.stats import gaussian_kde, binned_statistic
    from matplotlib.gridspec import GridSpec
    import os

    if save_path is None:
        save_path = GalGA.output_path + 'Omni_Info_Figure.png'

    # Ensure we have the required data
    if not hasattr(GalGA, 'age_data') or len(GalGA.age_data) == 0:
        print("No age data available for plotting")
        return None

    if not hasattr(GalGA, 'mdf_data') or len(GalGA.mdf_data) == 0:
        print("No MDF data available for plotting")
        return None

    if not hasattr(GalGA, 'alpha_data') or len(GalGA.alpha_data) == 0:
        print("No alpha data available for plotting")
        return None

    # Determine best model parameters
    if results_df is not None and not results_df.empty:
        best_params, best_idx = _best_param_tuple(results_df, metric_col=metric_col)
        best_row = results_df.loc[best_idx]
    else:
        r = GalGA.results[0]
        best_params = (r[5], r[7], r[9])
        # Create a mock row for parameter display
        col_names = [
            'comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx',
            'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2',
            'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb',
            'ks', 'ensemble', 'wrmse', 'mae', 'mape', 'huber',
            'cosine', 'log_cosh', 'fitness'
        ]
        best_row = dict(zip(col_names, r))





    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.3,
                  left=0.05, right=0.98, top=0.95, bottom=0.05)

    # =====================================================
    # PANEL 1: MODEL PARAMETERS (Top Left)
    # =====================================================
    ax_params = fig.add_subplot(gs[0, :2])
    ax_params.axis('off')

    # Create parameter text
    param_text = "BEST-FIT MODEL PARAMETERS\n" + "="*35 + "\n"
    param_text += f"σ₂ (second infall radio): {best_row['sigma_2']:.1f} \n"
    param_text += f"t₁ (first infall time): {best_row['t_1']:.3f} Gyr\n"
    param_text += f"t₂ (second infall time): {best_row['t_2']:.3f} Gyr\n"
    param_text += f"τ₁ (first infall timescale): {best_row['infall_1']:.3f} Gyr\n"
    param_text += f"τ₂ (second infall timescale): {best_row['infall_2']:.3f} Gyr\n"
    param_text += f"SFE (star formation efficiency): {best_row['sfe']:.5f}\n"
    param_text += f"ΔSFE (SFE change at t₂): {best_row['delta_sfe']:.3f}\n"
    param_text += f"IMF upper limit: {best_row['imf_upper']:.1f} M☉\n"
    param_text += f"Galaxy mass: {best_row['mgal']:.2e} M☉\n"
    param_text += f"SN Ia rate: {best_row['nb']:.2e} per M☉\n"

    ax_params.text(0.05, 0.95, param_text, transform=ax_params.transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    # =====================================================
    # PANEL 2: FIT QUALITY METRICS (Top Middle)
    # =====================================================
    ax_metrics = fig.add_subplot(gs[0, 2:4])
    ax_metrics.axis('off')

    # Create metrics text
    metrics_text = "FIT QUALITY METRICS\n" + "="*25 + "\n"
    metrics_text += f"Primary Loss (Fitness): {best_row['fitness']:.4f}\n"
    metrics_text += f"WRMSE: {best_row['wrmse']:.4f}\n"
    metrics_text += f"MAE: {best_row['mae']:.4f}\n"
    metrics_text += f"Huber Loss: {best_row['huber']:.4f}\n"
    metrics_text += f"Cosine Similarity: {best_row['cosine']:.4f}\n"
    metrics_text += f"KS Distance: {best_row['ks']:.4f}\n"
    metrics_text += f"Ensemble Metric: {best_row['ensemble']:.4f}\n"

    ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))

    # =====================================================
    # PANEL 3: MODEL SUMMARY (Top Right)
    # =====================================================
    ax_summary = fig.add_subplot(gs[0, 4:])
    ax_summary.axis('off')

    # Create model summary
    summary_text = "MODEL INTERPRETATION\n" + "="*25 + "\n"

    # Interpret the parameters
    if best_row['t_2'] < 2.0:
        infall_interp = "Early second infall"
    elif best_row['t_2'] < 8.0:
        infall_interp = "Mid-age second infall"
    else:
        infall_interp = "Late second infall"

    if best_row['delta_sfe'] > 0:
        sfe_interp = "SFE increases at second infall"
    elif best_row['delta_sfe'] < -0.01:
        sfe_interp = "SFE decreases at second infall"
    else:
        sfe_interp = "SFE unchanged at second infall"

    summary_text += f"• {infall_interp}\n"
    summary_text += f"• {sfe_interp}\n"
    summary_text += f"• First infall: τ = {best_row['infall_1']:.2f} Gyr\n"
    summary_text += f"• Second infall: τ = {best_row['infall_2']:.2f} Gyr\n"

    #add the list of catagorical params here

    if best_row['infall_2'] < best_row['infall_1']:
        summary_text += "• Faster second infall\n"
    else:
        summary_text += "• Slower second infall\n"

    summary_text += f"• Total models evaluated: {len(GalGA.results)}\n"

    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    # =====================================================
    # PANEL 4: METALLICITY DISTRIBUTION FUNCTION
    # =====================================================
    ax_mdf = fig.add_subplot(gs[1, :3])

    # Find best MDF model
    best_mdf_x = None
    best_mdf_y = None
    for mdf_data, res in zip(GalGA.mdf_data, GalGA.results):
        params = (res[5], res[7], res[9])
        is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
        if is_best:
            best_mdf_x, best_mdf_y = mdf_data
            break

    if best_mdf_x is not None:
        ax_mdf.plot(best_mdf_x, best_mdf_y, 'r-', linewidth=3, label='Best Model', zorder=3)
    ax_mdf.plot(feh_mdf, normalized_count_mdf, 'ko', markersize=6, label='Observed', zorder=2)

    ax_mdf.set_xlabel('[Fe/H]', fontsize=14)
    ax_mdf.set_ylabel('Normalized Number Density', fontsize=14)
    ax_mdf.set_xlim(-2, 1)
    ax_mdf.legend(fontsize=12)
    ax_mdf.grid(True, alpha=0.3)

    # =====================================================
    # PANEL 5: AGE-METALLICITY RELATION
    # =====================================================
    ax_age = fig.add_subplot(gs[1, 3:])

    # Find best age-metallicity model
    best_age_x = None
    best_age_y = None
    for age_data, res in zip(GalGA.age_data, GalGA.results):
        params = (res[5], res[7], res[9])
        is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
        if is_best:
            x_age_raw, y_feh = age_data
            best_age_x = (x_age_raw[-1] / 1e9) - np.array(x_age_raw) / 1e9
            best_age_y = np.array(y_feh)
            break

    # Plot observational data
    ax_age.scatter(age_Joyce, Fe_H, marker='*', s=40, color='blue', alpha=0.6, label='Joyce et al.')
    ax_age.scatter(age_Bensby, Fe_H, marker='^', s=40, color='orange', alpha=0.6, label='Bensby et al.')

    # Plot best model
    if best_age_x is not None:
        ax_age.plot(best_age_x, best_age_y, 'r-', linewidth=3, label='Best Model', zorder=3)

    ax_age.set_xlabel('Age (Gyr)', fontsize=14)
    ax_age.set_ylabel('[Fe/H]', fontsize=14)
    ax_age.set_xlim(0, 14)
    ax_age.set_ylim(-2, 1)
    ax_age.legend(fontsize=11)
    ax_age.grid(True, alpha=0.3)

    # =====================================================
    # PANEL 6-9: ALPHA ELEMENT ABUNDANCES (2x2 grid)
    # =====================================================
    alpha_elements = ['Mg', 'Si', 'Ca', 'Ti']
    alpha_obs_data = [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]

    for idx, (element, obs_data) in enumerate(zip(alpha_elements, alpha_obs_data)):
        row = 2 + idx // 2
        col = (idx % 2) * 3
        ax_alpha = fig.add_subplot(gs[row, col:col+3])

        # Find best alpha model for this element
        best_alpha_x = None
        best_alpha_y = None
        for alpha_arrs, res in zip(GalGA.alpha_data, GalGA.results):
            params = (res[5], res[7], res[9])
            is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
            if is_best and idx < len(alpha_arrs):
                best_alpha_x, best_alpha_y = alpha_arrs[idx]
                break

        # Clean observational data
        obs_clean = np.where((obs_data >= -2.0) & (obs_data <= 2.0), obs_data, np.nan)
        mask = np.isfinite(Fe_H) & np.isfinite(obs_clean)

        # Plot observational data
        if np.sum(mask) > 10:
            ax_alpha.scatter(Fe_H[mask], obs_clean[mask], s=20, alpha=0.6,
                           color='gray', label='Observed', zorder=1)

        # Plot best model
        if best_alpha_x is not None:
            ax_alpha.plot(best_alpha_x, best_alpha_y, 'r-', linewidth=3,
                         label='Best Model', zorder=3)

        ax_alpha.set_xlabel('[Fe/H]', fontsize=12)
        ax_alpha.set_ylabel(f'[{element}/Fe]', fontsize=12)
        ax_alpha.set_xlim(-2, 1)
        ax_alpha.set_ylim(-0.6, 0.8)
        ax_alpha.legend(fontsize=10, loc='upper right')
        ax_alpha.grid(True, alpha=0.3)

        # Add element label
        ax_alpha.text(0.05, 0.9, element, transform=ax_alpha.transAxes,
                     fontsize=16, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # =====================================================
    # FINAL TOUCHES
    # =====================================================

    # Add a subtle background color to distinguish sections
    fig.patch.set_facecolor('white')

    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"dashboard saved to {save_path}")
    print(f"Best-fit parameters:")
    print(f"  σ₂ = {best_row['sigma_2']:.1f}")
    print(f"  t₂ = {best_row['t_2']:.3f} Gyr")
    print(f"  τ₂ = {best_row['infall_2']:.3f} Gyr")
    print(f"  SFE = {best_row['sfe']:.5f}")
    print(f"  Fitness = {best_row['fitness']:.4f}")

    return fig


def _best_index_by_params(results_df, metric_col: str = 'fitness'):
    if results_df is None or results_df.empty:
        return 0
    preferred = metric_col if metric_col in results_df.columns else None
    if preferred is None and 'fitness' in results_df.columns:
        preferred = 'fitness'
    if preferred is None and 'confidence' in results_df.columns:
        preferred = 'confidence'
    if preferred is None:
        return int(results_df.index[0])
    series = pd.to_numeric(results_df[preferred], errors='coerce')
    return int(series.idxmin())


def _best_param_tuple(results_df, metric_col: str = 'fitness'):
    i = _best_index_by_params(results_df, metric_col=metric_col)
    r = results_df.loc[i]
    return (float(r['sigma_2']), float(r['t_2']), float(r['infall_2'])), i





def plot_omni_figure(
    GalGA,
    Fe_H,
    age_Joyce,
    age_Bensby,
    Mg_Fe,
    Si_Fe,
    Ca_Fe,
    Ti_Fe,
    feh_mdf,
    normalized_count_mdf,
    results_df=None,
    save_path=None,
    metric_col: str = 'fitness',
):
    """
    ApJ-clean figure: MDF (top-left), AMR (top-right), 4×alpha panels (bottom).
    Minimal legends/labels. Tight spacing. Same IO pattern as your code.
    Returns the Matplotlib Figure.
    """
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import os



    if save_path is None:
        save_path = os.path.join(getattr(GalGA, "output_path", ""), " Omni_Info_Figure_ApJ.png")

    best_params, best_idx = _best_param_tuple(results_df, metric_col=metric_col)


    # ------ Figure layout (tight, no wasted whitespace) ------
    fig = plt.figure(figsize=(15, 8))  # ApJ 2-col width
    gs = GridSpec(
        2, 8, figure=fig,
        left=0.065, right=0.995, bottom=0.10, top=0.965,
        wspace=0.16, hspace=0.2  # small gap between rows, as requested
    )

    # Top row
    ax_mdf = fig.add_subplot(gs[0, 0:4])
    ax_amr = fig.add_subplot(gs[0, 4:8])

    # ------ MDF ------
    best_x = best_y = None
    for (x, y), res in zip(GalGA.mdf_data, GalGA.results):
        is_best = all(abs(p - b) < 1e-5 for p, b in zip((res[5], res[7], res[9]), best_params))
        if is_best:
            best_x, best_y = np.asarray(x), np.asarray(y)
        else:
            ax_mdf.plot(x, y, color="0.75", alpha=0.001, lw=0.8, zorder=1)

    if best_x is not None:
        ax_mdf.plot(best_x, best_y, color="crimson", lw=1.8, label="Model", zorder=3)

    ax_mdf.plot(feh_mdf, normalized_count_mdf, "x", color="k", ms=4.5, mew=0.9, label="Data", zorder=4)

    ax_mdf.set_xlim(-2, 1)
    ax_mdf.set_ylabel("Normalized number")

    # x-axis at top only
    ax_mdf.xaxis.set_ticks_position("top")
    ax_mdf.xaxis.set_label_position("top")
    ax_mdf.set_xlabel("[Fe/H]")
    ax_mdf.tick_params(axis="x", bottom=False)

    ax_mdf.legend(loc="upper left", fontsize=9, handlelength=1.6)

    # ------ AMR (y-axis on right) ------
    best_age_x = best_age_y = None
    for (t_arr, feh_arr), res in zip(GalGA.age_data, GalGA.results):
        is_best = all(abs(p - b) < 1e-5 for p, b in zip((res[5], res[7], res[9]), best_params))
        if is_best:
            t = np.asarray(t_arr, float)  # years
            age = (t[-1] - t) / 1e9       # Age (Gyr), increasing to the right
            best_age_x, best_age_y = age, np.asarray(feh_arr, float)
        else:
            t = np.asarray(t_arr, float)  # years
            age = (t[-1] - t) / 1e9       # Age (Gyr), increasing to the right
            age_x, age_y = age, np.asarray(feh_arr, float)
            ax_amr.plot(age_x, age_y, color="0.75", alpha=0.001, lw=0.8, zorder=1)

    ax_amr.scatter(age_Joyce, Fe_H, s=10, facecolor="none", edgecolor="0.35", lw=0.7, label="Joyce")
    ax_amr.scatter(age_Bensby, Fe_H, s=10, marker="^", facecolor="none", edgecolor="0.55", lw=0.7, label="Bensby")
    if best_age_x is not None:
        ax_amr.plot(best_age_x, best_age_y, color="crimson", lw=1.8, label="Model", zorder=3)

    ax_amr.set_xlim(0, 14)
    ax_amr.set_ylim(-2, 1)

    # x-axis at top only
    ax_amr.xaxis.set_ticks_position("top")
    ax_amr.xaxis.set_label_position("top")
    ax_amr.set_xlabel("Age (Gyr)")
    ax_amr.tick_params(axis="x", bottom=False)

    # y-axis on right
    ax_amr.yaxis.tick_right()
    ax_amr.yaxis.set_label_position("right")
    ax_amr.set_ylabel("[Fe/H]")

    ax_amr.legend(loc="lower left", fontsize=9, ncol=3, columnspacing=0.9, handlelength=1.6)

    # ------ Alpha row ------
    alpha_elems = ["Mg", "Si", "Ca", "Ti"]
    alpha_obs   = [Mg_Fe, Si_Fe, Ca_Fe, Ti_Fe]
    axes_alpha  = [fig.add_subplot(gs[1, 2*i:2*i+2]) for i in range(4)]

    # Fetch best alpha arrays once
    best_alpha = None
    for alpha_arrs, res in zip(GalGA.alpha_data, GalGA.results):
        is_best = all(abs(p - b) < 1e-5 for p, b in zip((res[5], res[7], res[9]), best_params))
        if is_best:
            best_alpha = alpha_arrs
            break

    xlim = (-2, 1)
    ylim = (-0.6, 0.8)

    for i, (elt, obs, ax) in enumerate(zip(alpha_elems, alpha_obs, axes_alpha)):
        # Observations
        obs_clean = np.where((obs > -2.5) & (obs < 2.5), obs, np.nan)
        mask = np.isfinite(Fe_H) & np.isfinite(obs_clean)
        if np.count_nonzero(mask) > 5:
            ax.scatter(Fe_H[mask], obs_clean[mask], s=10, color="0.35", alpha=0.9, edgecolor="none", label="Data")

        # Model
        if best_alpha is not None and i < len(best_alpha):
            mx, my = best_alpha[i]
            ax.plot(mx, my, color="crimson", lw=1.6, label="Model")

        # Limits, labels
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("[Fe/H]")

        # Only leftmost has y-label
        if i == 0:
            ax.set_ylabel("[α/Fe]")
        else:
            ax.set_ylabel("")

        # Element tag
        ax.text(0.03, 0.95, elt, transform=ax.transAxes, ha="left", va="top", fontsize=17)

        # Middle two: hide y-numbering (keep ticks for alignment)
        if i in (1, 2):
            ax.set_yticklabels([])

        # Rightmost: y-axis on right (no y-label)
        if i == 3:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

    # Single small legend for the alpha set inside the last panel
    h, l = axes_alpha[-1].get_legend_handles_labels()
    if h:
        axes_alpha[-1].legend(loc="lower right", fontsize=9, handlelength=1.6)

    # ------ Save & return ------
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    return fig









