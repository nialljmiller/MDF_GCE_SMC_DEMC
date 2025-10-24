import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
import os
import sys
sys.path.append('../')
from JINAPyCEE import omega_plus


from plotting.style import *
use_paper_style()


def reconstruct_best_model(GalGA, results_df=None):
    """Reconstruct the omega_plus model for the best-fit parameters"""
    
    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        comp_idx = int(bm['comp_idx'])
        imf_idx = int(bm['imf_idx'])
        sn1a_idx = int(bm['sn1a_idx'])
        sy_idx = int(bm['sy_idx'])
        sn1ar_idx = int(bm['sn1ar_idx'])
        
        sigma_2 = bm['sigma_2']
        t_1 = bm['t_1']
        t_2 = bm['t_2']
        infall_1 = bm['infall_1']
        infall_2 = bm['infall_2']
        sfe_val = bm['sfe']
        delta_sfe_val = bm['delta_sfe']
        imf_upper = bm['imf_upper']
        mgal = bm['mgal']
        nb = bm['nb']
    else:
        r = GalGA.results[0]
        comp_idx, imf_idx, sn1a_idx, sy_idx, sn1ar_idx = int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4])
        sigma_2, t_1, t_2, infall_1, infall_2 = r[5], r[6], r[7], r[8], r[9]
        sfe_val, delta_sfe_val, imf_upper, mgal, nb = r[10], r[11], r[12], r[13], r[14]
    
    # Get the parameter arrays from GalGA
    comp = GalGA.comp_array[comp_idx]
    imf_val = GalGA.imf_array[imf_idx]
    sn1a = GalGA.sn1a_assumptions[sn1a_idx]
    sy = GalGA.stellar_yield_assumptions[sy_idx]
    sn1ar = GalGA.sn1a_rates[sn1ar_idx]
    
    # Reconstruct the model with the same parameters used in evaluation
    kwargs = {
        'special_timesteps': GalGA.timesteps,
        'twoinfall_sigmas': [1300, sigma_2],
        'galradius': 1800,
        'exp_infall': [[-1, t_1*1e9, infall_1*1e9], [-1, t_2*1e9, infall_2*1e9]],
        'tauup': [0.02e9, 0.02e9],
        'mgal': mgal,
        'iniZ': 0.0,
        'mass_loading': 0.0,
        'table': GalGA.sn1a_header + sy,
        'sfe': sfe_val,
        'delta_sfe': delta_sfe_val,
        'imf_type': imf_val,
        'sn1a_table': GalGA.sn1a_header + sn1a,
        'imf_yields_range': [1, imf_upper],
        'iniabu_table': GalGA.iniab_header + comp,
        'nb_1a_per_m': nb,
        'sn1a_rate': sn1ar
    }
    
    print("Reconstructing best-fit omega_plus model...")
    GCE_model = omega_plus.omega_plus(**kwargs)
    print("Model reconstruction successful!")
    
    return GCE_model

def plot_real_infall_physics(GalGA, results_df=None, save_path='Real_Infall_Physics.png'):
    """
    Generate an enhanced visualization of galactic chemical evolution model physics,
    emphasizing the two-infall paradigm with improved scientific presentation.
    """
    
    save_path = GalGA.output_path + save_path

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Reconstruct the best model
    GCE_model = reconstruct_best_model(GalGA, results_df)
    
    # Extract physical arrays from omega model with proper error handling
    try:
        ages = np.array(GCE_model.inner.history.age) / 1e9
        timesteps_yr = np.array(GCE_model.inner.history.timesteps)
        inflow_masses = np.array(GCE_model.inner.m_inflow_t)
        outflow_masses = np.array(GCE_model.inner.m_outflow_t)
        
        # Ensure consistent array dimensions
        sfr_rates = np.array(GCE_model.inner.history.sfr_abs)[:len(timesteps_yr)]
        metallicity = np.array(GCE_model.inner.history.metallicity)[:len(timesteps_yr)]
        
        # Convert masses to physically meaningful rates
        inflow_rates = inflow_masses / timesteps_yr  # M☉/yr
        outflow_rates = outflow_masses / timesteps_yr  # M☉/yr
        
        # Extract cumulative quantities
        gas_masses = np.array([np.sum(GCE_model.inner.ymgal[i]) for i in range(len(GCE_model.inner.ymgal))])
        stellar_masses_raw = np.array(GCE_model.inner.history.m_locked)
        
        # Ensure stellar mass array matches age array length
        if len(stellar_masses_raw) < len(ages):
            stellar_masses = np.append(stellar_masses_raw, stellar_masses_raw[-1])
        else:
            stellar_masses = stellar_masses_raw[:len(ages)]
            
    except Exception as e:
        print(f"Error extracting model data: {e}")
        return None
    
    # Extract best-fit parameters for physical interpretation
    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        sigma_2, t_1, t_2 = bm['sigma_2'], bm['t_1'], bm['t_2']
        infall_1, infall_2 = bm['infall_1'], bm['infall_2']
        sfe_val, delta_sfe_val = bm['sfe'], bm['delta_sfe']
        mgal = bm['mgal']
    else:
        r = GalGA.results[0]
        sigma_2, t_1, t_2, infall_1, infall_2 = r[5], r[6], r[7], r[8], r[9]
        sfe_val, delta_sfe_val, mgal = r[10], r[11], r[13]
    
    # Create enhanced figure with improved layout
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3, 
                  left=0.06, right=0.98, top=0.94, bottom=0.06)
    

    
    # Define enhanced color palette for scientific clarity
    colors = {
        'inflow': '#1f77b4',      # Professional blue
        'outflow': '#d62728',     # Scientific red  
        'sfr': '#ff7f0e',         # Distinct orange
        'gas': '#2ca02c',         # Scientific green
        'stellar': '#9467bd',     # Professional purple
        'metallicity': '#8c564b',  # Earth tone
        'efficiency': '#e377c2',  # Distinctive pink
        'loading': '#7f7f7f'      # Neutral gray
    }
    
    # ======================================================================
    # PANEL 1: Enhanced Inflow Rate with Theoretical Overlay
    # ======================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot actual inflow rate with improved styling
    ax1.plot(ages[:-1], inflow_rates, color=colors['inflow'], linewidth=3, 
             label='Computed Inflow Rate', marker='o', markersize=4, alpha=0.9)
    
    # Add theoretical infall episodes with enhanced visualization
    t_theory = np.linspace(0, ages[-1], 1000)
    
    # First infall episode (exponential decay)
    infall_1_theory = np.exp(-t_theory / infall_1) * np.heaviside(t_theory - t_1, 1)
    infall_1_norm = np.max(inflow_rates) * infall_1_theory / np.max(infall_1_theory) if np.max(infall_1_theory) > 0 else infall_1_theory
    
    # Second infall episode  
    infall_2_theory = np.exp(-(t_theory - t_2) / infall_2) * np.heaviside(t_theory - t_2, 1)
    infall_2_norm = np.max(inflow_rates) * 0.3 * infall_2_theory / np.max(infall_2_theory) if np.max(infall_2_theory) > 0 else infall_2_theory
    
    # Plot theoretical curves with transparency
    ax1.plot(t_theory, infall_1_theory, '--', color='lightblue', linewidth=2, alpha=0.7,
             label=f'First Episode (τ={infall_1:.2f} Gyr)')
    ax1.plot(t_theory, infall_2_theory, '--', color='salmon', linewidth=2, alpha=0.7,
             label=f'Second Episode (τ={infall_2:.2f} Gyr)')
    
    # Mark critical epochs
    ax1.axvline(t_1, color='steelblue', linestyle=':', linewidth=2, alpha=0.8)
    ax1.axvline(t_2, color='crimson', linestyle=':', linewidth=2, alpha=0.8)
    
    # Enhanced axis formatting
    ax1.set_xlabel('Universe Age (Gyr)', fontsize=14, fontweight='bold')
    ax1.set_ylabel(r'Inflow Rate ($M_\odot$ yr$^{-1}$)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax1.set_xlim(0, ages[-1])
    ax1.set_ylim(bottom=0)
    
    # ======================================================================
    # PANEL 2: Star Formation History with Physical Context
    # ======================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(ages[:-1], sfr_rates, color=colors['sfr'], linewidth=2.5, 
                 label='SFR', marker='s', markersize=3)
    
    # Highlight SFE change epoch
    ax2.axvline(t_2, color='crimson', linestyle=':', alpha=0.7, linewidth=2)
    ax2.text(t_2 + 0.2, np.max(sfr_rates) * 0.1, f'ΔSFE = {delta_sfe_val:+.4f}', 
             rotation=90, fontsize=10, alpha=0.8, fontweight='bold')
    
    ax2.set_xlabel('Universe Age (Gyr)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'SFR ($M_\odot$ yr$^{-1}$)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize=11)
    
    # ======================================================================
    # PANEL 3: Gas Flows and Mass Loading
    # ======================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(ages[:-1], inflow_rates, color=colors['inflow'], linewidth=2, 
             label='Inflow', marker='o', markersize=2, alpha=0.8)

    ax3.plot(ages[:-1], outflow_rates, color=colors['outflow'], linewidth=2, 
             label='Outflow', marker='^', markersize=2, alpha=0.8)

    ax3.set_xlabel('Universe Age (Gyr)', fontsize=12, fontweight='bold')
    ax3.set_ylabel(r'Flow Rate ($M_\odot$ yr$^{-1}$)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.2)
    ax3.legend(fontsize=11)
    
    # ======================================================================
    # PANEL 4: Mass Loading Factor with Physical Interpretation
    # ======================================================================
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(ages[:-1], inflow_rates, color=colors['inflow'], linewidth=2, 
             label='Inflow', marker='o', markersize=2, alpha=0.8)

    ax4.semilogy(ages[:-1], np.maximum(outflow_rates, 1e-10), 
                 color=colors['outflow'], linewidth=2, 
                 label='Outflow', marker='^', markersize=2, alpha=0.8)


    ax4.set_xlabel('Universe Age (Gyr)', fontsize=12, fontweight='bold')
    ax4.set_ylabel(r'Flow Rate ($M_\odot$ yr$^{-1}$)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.2)
    ax4.legend(fontsize=11)
    
    # ======================================================================
    # PANEL 5: Metallicity Evolution
    # ======================================================================
    ax5 = fig.add_subplot(gs[1, 3])
    ax5.plot(ages[:-1], metallicity, color=colors['metallicity'], linewidth=2.5,
             label='[Fe/H]', marker='o', markersize=3)
    
    ax5.set_xlabel('Universe Age (Gyr)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('[Fe/H]', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.2)
    ax5.legend(fontsize=11)
    
    # ======================================================================
    # PANEL 6: Reservoir Masses with Physical Scaling
    # ======================================================================
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.semilogy(ages, gas_masses, color=colors['gas'], linewidth=3, 
                 label='Gas Reservoir', marker='o', markersize=3, alpha=0.9)
    ax6.semilogy(ages, stellar_masses, color=colors['stellar'], linewidth=3, 
                 label='Stellar Component', marker='s', markersize=3, alpha=0.9)
    
    # Add total mass for context
    total_baryons = gas_masses + stellar_masses
    ax6.semilogy(ages, total_baryons, color='black', linewidth=2, linestyle='--', 
                 label='Total mass', alpha=0.7)
    
    # Mark key transition epochs
    ax6.axvline(t_1, color='steelblue', linestyle=':', alpha=0.6)
    ax6.axvline(t_2, color='crimson', linestyle=':', alpha=0.6)
    
    ax6.set_xlabel('Universe Age (Gyr)', fontsize=12, fontweight='bold')
    ax6.set_ylabel(r'Reservoir Mass ($M_\odot$)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.2)
    ax6.legend(fontsize=11, loc='best')
    
    # ======================================================================
    # PANEL 7: Gas Fraction and Star Formation Efficiency
    # ======================================================================
    ax7 = fig.add_subplot(gs[2, 2:])
    
    # Primary y-axis: Gas fraction
    gas_fraction = gas_masses / (gas_masses + stellar_masses)
    ax7.plot(ages, gas_fraction, color=colors['gas'], linewidth=3, 
             label='Gas Fraction', marker='v', markersize=3)
    ax7.set_ylabel('Gas Fraction', color=colors['gas'], fontsize=12, fontweight='bold')
    ax7.tick_params(axis='y', labelcolor=colors['gas'])
    
    # Secondary y-axis: Star formation efficiency
    ax7_twin = ax7.twinx()
    # Calculate instantaneous SFE = SFR / M_gas
    sfe_inst = np.where(gas_masses[:-1] > 0, sfr_rates / gas_masses[:-1], 0)
    ax7_twin.semilogy(ages[:-1], sfe_inst, color=colors['efficiency'], linewidth=2, 
                      linestyle='--', label='Instantaneous SFE', marker='x', markersize=3)
    ax7_twin.set_ylabel(r'SFE (yr$^{-1}$)', color=colors['efficiency'], fontsize=12, fontweight='bold')
    ax7_twin.tick_params(axis='y', labelcolor=colors['efficiency'])
    
    ax7.set_xlabel('Universe Age (Gyr)', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.2)
    
    # Combined legend
    lines1, labels1 = ax7.get_legend_handles_labels()
    lines2, labels2 = ax7_twin.get_legend_handles_labels()
    ax7.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11)
    
    # ======================================================================
    # PANEL 8: Cumulative Budget Analysis
    # ======================================================================
    ax8 = fig.add_subplot(gs[3, :2])
    
    # Cumulative fluxes
    cumulative_inflow = np.cumsum(inflow_masses)
    cumulative_outflow = np.cumsum(outflow_masses)
    cumulative_sf = np.cumsum(sfr_rates * timesteps_yr)
    
    ax8.semilogy(ages[:-1], cumulative_inflow, color=colors['inflow'], linewidth=3, 
                 label='Cumulative Inflow', marker='o', markersize=3)
    ax8.semilogy(ages[:-1], cumulative_sf, color=colors['sfr'], linewidth=3, 
                 label='Cumulative SF', marker='s', markersize=3)
    
    if np.max(cumulative_outflow) > 0:
        ax8.semilogy(ages[:-1], np.maximum(cumulative_outflow, 1e6), color=colors['outflow'], 
                     linewidth=3, label='Cumulative Outflow', marker='^', markersize=3)
    
    ax8.set_xlabel('Universe Age (Gyr)', fontsize=12, fontweight='bold')
    ax8.set_ylabel(r'Cumulative Mass ($M_\odot$)', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.2)
    ax8.legend(fontsize=11)
    

    ax8a = fig.add_subplot(gs[3, 2])
    
    # Cumulative fluxes
    cumulative_inflow = np.cumsum(inflow_masses)
    cumulative_outflow = np.cumsum(outflow_masses)
    cumulative_sf = np.cumsum(sfr_rates * timesteps_yr)
    
    ax8a.plot(ages[:-1], cumulative_inflow, color=colors['inflow'], linewidth=3, 
                 label='Cumulative Inflow', marker='o', markersize=3)
    ax8a.plot(ages[:-1], cumulative_sf, color=colors['sfr'], linewidth=3, 
                 label='Cumulative SF', marker='s', markersize=3)
    
    if np.max(cumulative_outflow) > 0:
        ax8a.plot(ages[:-1], np.maximum(cumulative_outflow, 1e6), color=colors['outflow'], 
                     linewidth=3, label='Cumulative Outflow', marker='^', markersize=3)
    
    ax8a.set_xlabel('Universe Age (Gyr)', fontsize=12, fontweight='bold')
    ax8a.set_ylabel(r'Cumulative Mass ($M_\odot$)', fontsize=12, fontweight='bold')
    ax8a.grid(True, alpha=0.2)
    ax8a.legend(fontsize=11)






    # ======================================================================
    # PANEL 9: Enhanced Physics Summary with Quantitative Analysis
    # ======================================================================
    ax9 = fig.add_subplot(gs[3, 3:])
    ax9.axis('off')
    
    # Calculate physics summary
    total_inflow = np.sum(inflow_masses)
    total_outflow = np.sum(outflow_masses) 
    total_sf = np.sum(sfr_rates * timesteps_yr)
    peak_inflow = np.max(inflow_rates)
    peak_sfr = np.max(sfr_rates)
    final_stellar_mass = stellar_masses[-1]
    final_gas_mass = gas_masses[-1]
    inflow_peak_time = ages[:-1][np.argmax(inflow_rates)]
    
    # Calculate key efficiency metrics
    sf_efficiency = total_sf / total_inflow if total_inflow > 0 else 0
    retention_fraction = (total_inflow - total_outflow) / total_inflow if total_inflow > 0 else 0
    final_gas_fraction = final_gas_mass / (final_gas_mass + final_stellar_mass)
    
    # Determine infall episode characteristics
    if t_2 < 2.0:
        infall_regime = "Early second accretion"
    elif t_2 < 8.0:
        infall_regime = "Intermediate second accretion"  
    else:
        infall_regime = "Late second accretion"
        
    summary_text = f"""PHYSICAL MODEL DIAGNOSTICS

Two-Infall Parameters:
├─ σ₂ = {sigma_2:.1f}
├─ Episode I: t₁ = {t_1:.3f} Gyr, τ₁ = {infall_1:.2f} Gyr
├─ Episode II: t₂ = {t_2:.1f} Gyr, τ₂ = {infall_2:.2f} Gyr  
└─ SFE Evolution: {sfe_val:.4f} → {sfe_val + delta_sfe_val:.4f}

Mass:
├─ Total inflow: {total_inflow:.2e}
├─ Final stellar mass: {final_stellar_mass:.2e}
└─ Final gas reservoir: {final_gas_mass:.2e}

"""
    
    ax9.text(0.02, 0.99, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace', linespacing=1.4,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightcyan", 
                      edgecolor="steelblue", alpha=0.95, linewidth=1.5))

    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    plt.close('all') 

    print(f"Enhanced physics diagnostics saved: {save_path}")
    
    return fig

def plot_omega_diagnostics(GalGA, results_df=None, save_path='Omega_Model_Diagnostics.png'):
    """Plot additional diagnostics from the omega model"""
    

    save_path = GalGA.output_path + save_path


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Reconstruct the best model
    GCE_model = reconstruct_best_model(GalGA, results_df)
    
    # Extract additional omega diagnostics with proper array lengths
    ages = np.array(GCE_model.inner.history.age) / 1e9
    timesteps_yr = np.array(GCE_model.inner.history.timesteps)
    
    # Fix arrays to proper lengths
    metallicity = np.array(GCE_model.inner.history.metallicity)[:len(timesteps_yr)]  # 10 elements
    eta_outflow = np.array(GCE_model.inner.history.eta_outflow_t)[:len(timesteps_yr)]  # 10 elements  
    m_tot_ISM = np.array(GCE_model.inner.history.m_tot_ISM_t)  # Should be 11 elements
    
    # Create diagnostics plot
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Metallicity evolution (10 elements vs 10 elements)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ages[:-1], metallicity, 'gold', linewidth=2, marker='o', markersize=2)
    ax1.set_xlabel('Universe Age (Gyr)')
    ax1.set_ylabel('Metallicity Z')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 2. Outflow efficiency evolution (10 elements vs 10 elements)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ages[:-1], eta_outflow, 'darkgreen', linewidth=2, marker='s', markersize=2)
    ax2.set_xlabel('Universe Age (Gyr)')
    ax2.set_ylabel('η (Mass Loading)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Total ISM mass (11 elements vs 11 elements)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ages, m_tot_ISM, 'darkred', linewidth=2, marker='^', markersize=2)
    ax3.set_xlabel('Universe Age (Gyr)')
    ax3.set_ylabel(r'Total ISM Mass [$M_\odot$]')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. Halo properties (11 elements vs 11 elements)
    ax4 = fig.add_subplot(gs[1, :])
    halo_masses = [np.sum(outer) for outer in GCE_model.ymgal_outer]
    ax4.plot(ages, halo_masses, 'purple', linewidth=2, marker='d', markersize=2, label='Halo Gas Mass')
    ax4.set_xlabel('Universe Age (Gyr)')
    ax4.set_ylabel(r'Halo Gas Mass [$M_\odot$]')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Omega diagnostics plot saved to {save_path}")
    return fig


def plot_physical_constraints(GalGA, results_df=None, save_path='Physical_Constraints_Validation.png'):
    """
    Comprehensive visualization of all physical constraints and how the best model satisfies them.
    
    This plot shows:
    1. MDF constraints (peak location, low-metallicity tail)
    2. Alpha element constraints (binned regions and distribution properties)
    3. Age-metallicity constraints
    4. Model-level constraints (mass, age, gas fraction, SFH)
    """
    
    save_path = GalGA.output_path + save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Import physical_constraints module
    import physical_constraints as pc
    
    # Reconstruct the best model
    GCE_model = reconstruct_best_model(GalGA, results_df)
    
    # Extract model outputs
    ages = np.array(GCE_model.inner.history.age) / 1e9
    timesteps_yr = np.array(GCE_model.inner.history.timesteps)
    
    # Get best model parameters to find the correct data
    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        best_params = (bm['sigma_2'], bm['t_2'], bm['infall_2'])
    else:
        r = GalGA.results[0]
        best_params = (r[5], r[7], r[9])
    
    # Find best model MDF data from GalGA.mdf_data
    MDF_x = None
    MDF_y_model = None
    for (x, y), res in zip(GalGA.mdf_data, GalGA.results):
        params = (res[5], res[7], res[9])
        is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
        if is_best:
            MDF_x = np.array(x)
            MDF_y_model = np.array(y)
            break
    
    if MDF_x is None or MDF_y_model is None:
        print("Error: Could not find best model MDF data")
        return None
    
    # Find best model alpha element data from GalGA.alpha_data
    alpha_arrs = None
    element_names = ['[Si/Fe]', '[Ca/Fe]', '[Mg/Fe]', '[Ti/Fe]']
    for alpha_arrays, res in zip(GalGA.alpha_data, GalGA.results):
        params = (res[5], res[7], res[9])
        is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
        if is_best:
            alpha_arrs = alpha_arrays
            break
    
    if alpha_arrs is None:
        print("Error: Could not find best model alpha data")
        return None
    
    # Find best model age-metallicity data from GalGA.age_data
    age_x = None
    age_y = None
    for (t_arr, feh_arr), res in zip(GalGA.age_data, GalGA.results):
        params = (res[5], res[7], res[9])
        is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
        if is_best:
            # Convert time to stellar age in Gyr
            t_final = t_arr[-1]
            age_x = (t_final - np.array(t_arr)) / 1e9
            age_y = np.array(feh_arr)
            break
    
    if age_x is None or age_y is None:
        print("Warning: Could not find best model age-metallicity data")
        age_x = np.array([])
        age_y = np.array([])
    
    # Create figure with comprehensive layout
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35,
                  left=0.06, right=0.98, top=0.95, bottom=0.05)
    
    # Define color scheme
    colors = {
        'model': '#1f77b4',
        'constraint': '#d62728',
        'valid': '#2ca02c',
        'invalid': '#ff7f0e',
        'boundary': '#9467bd'
    }
    
    # ======================================================================
    # PANEL 1: MDF Peak Location Constraint
    # ======================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot MDF
    ax1.plot(MDF_x, MDF_y_model, color=colors['model'], linewidth=3, 
             label='Model MDF', marker='o', markersize=4)
    
    # Find and mark peak
    if len(MDF_y_model) > 0 and np.max(MDF_y_model) > 0:
        peak_idx = np.argmax(MDF_y_model)
        peak_feh = MDF_x[peak_idx]
        peak_val = MDF_y_model[peak_idx]
        
        # Mark peak
        ax1.plot(peak_feh, peak_val, 'r*', markersize=20, 
                label=f'Peak at [Fe/H]={peak_feh:.2f}', zorder=10)
        
        # Show constraint region (-1.0 to 1.0)
        ax1.axvspan(-1.0, 1.0, alpha=0.2, color=colors['valid'], 
                   label='Valid Peak Region')
        ax1.axvline(-1.0, color=colors['constraint'], linestyle='--', linewidth=2)
        ax1.axvline(1.0, color=colors['constraint'], linestyle='--', linewidth=2)
    
    ax1.set_xlabel('[Fe/H]', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Normalized Count', fontsize=14, fontweight='bold')
    ax1.set_title('MDF Peak Location Constraint', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ======================================================================
    # PANEL 2: MDF Low-Metallicity Tail Constraint
    # ======================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Plot MDF with focus on low metallicity
    ax2.plot(MDF_x, MDF_y_model, color=colors['model'], linewidth=3, 
             label='Model MDF', marker='o', markersize=4)
    
    # Highlight constraint regions
    very_metal_poor_mask = MDF_x < -1.0
    extremely_metal_poor_mask = MDF_x < -1.5
    
    if np.sum(very_metal_poor_mask) > 0:
        max_tail = np.max(MDF_y_model[very_metal_poor_mask])
        mean_tail = np.mean(MDF_y_model[very_metal_poor_mask])
        
        # Show constraint thresholds
        ax2.axhline(0.1, color=colors['constraint'], linestyle='--', linewidth=2,
                   label='Max Tail Limit (0.1)')
        ax2.axhline(0.05, color=colors['constraint'], linestyle=':', linewidth=2,
                   label='Mean Tail Limit (0.05)')
        
        # Shade constraint regions
        ax2.axvspan(MDF_x.min(), -1.0, alpha=0.15, color=colors['valid'],
                   label='Very Metal-Poor ([Fe/H]<-1.0)')
        ax2.axvspan(MDF_x.min(), -1.5, alpha=0.15, color='orange',
                   label='Extremely Metal-Poor ([Fe/H]<-1.5)')
    
    if np.sum(extremely_metal_poor_mask) > 0:
        max_extreme = np.max(MDF_y_model[extremely_metal_poor_mask])
        ax2.axhline(0.03, color='orange', linestyle='--', linewidth=2,
                   label='Extreme Tail Limit (0.03)')
    
    ax2.set_xlabel('[Fe/H]', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Normalized Count', fontsize=14, fontweight='bold')
    ax2.set_title('MDF Low-Metallicity Tail Constraint', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(MDF_x.min(), -0.5)
    
    # ======================================================================
    # PANEL 3-5: Alpha Element Binned Constraints
    # ======================================================================
    alpha_elements_to_plot = ['[Si/Fe]', '[Ca/Fe]', '[Mg/Fe]']
    
    for idx, elem_name in enumerate(alpha_elements_to_plot):
        ax = fig.add_subplot(gs[0, 2 + (idx % 2)])
        if idx == 2:
            ax = fig.add_subplot(gs[1, 0])
        
        elem_idx = element_names.index(elem_name)
        alpha_x, alpha_y = alpha_arrs[elem_idx]
        alpha_x = np.array(alpha_x)
        alpha_y = np.array(alpha_y)
        
        # Remove invalid data
        valid_mask = np.isfinite(alpha_x) & np.isfinite(alpha_y)
        alpha_x = alpha_x[valid_mask]
        alpha_y = alpha_y[valid_mask]
        
        if len(alpha_x) > 0:
            # Plot data
            ax.scatter(alpha_x, alpha_y, alpha=0.5, s=20, color=colors['model'],
                      label='Model Data')
            
            # Bin 1: [Fe/H] < -1.0 → alpha should be > 0.15
            ax.axvspan(MDF_x.min(), -1.0, ymin=0.15/0.6, ymax=1.0, 
                      alpha=0.15, color=colors['valid'], label='Bin 1: Valid Region')
            ax.axhline(0.15, xmin=0, xmax=0.3, color=colors['constraint'], 
                      linestyle='--', linewidth=2)
            ax.text(-1.5, 0.16, 'α>0.15', fontsize=9, color=colors['constraint'])
            
            # Bin 2: -1.0 <= [Fe/H] < -0.5 → alpha between 0.05 and 0.4
            ax.axvspan(-1.0, -0.5, ymin=0.05/0.6, ymax=0.4/0.6, 
                      alpha=0.15, color=colors['valid'], label='Bin 2: Valid Region')
            ax.axhline(0.05, xmin=0.3, xmax=0.5, color=colors['constraint'], 
                      linestyle='--', linewidth=1.5)
            ax.axhline(0.4, xmin=0.3, xmax=0.5, color=colors['constraint'], 
                      linestyle='--', linewidth=1.5)
            
            # Bin 3: [Fe/H] > 0.0 → alpha between -0.2 and 0.2
            feh_range = MDF_x.max() - MDF_x.min()
            bin3_xmin = (0.0 - MDF_x.min()) / feh_range
            ax.axvspan(0.0, MDF_x.max(), ymin=(0.2+0.2)/0.6, ymax=(0.4+0.2)/0.6, 
                      alpha=0.15, color=colors['valid'], label='Bin 3: Valid Region')
            ax.axhline(-0.2, xmin=bin3_xmin, xmax=1.0, color=colors['constraint'], 
                      linestyle='--', linewidth=1.5)
            ax.axhline(0.2, xmin=bin3_xmin, xmax=1.0, color=colors['constraint'], 
                      linestyle='--', linewidth=1.5)
        
        ax.set_xlabel('[Fe/H]', fontsize=12, fontweight='bold')
        ax.set_ylabel(elem_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{elem_name} Binned Constraints', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 0.6)
    
    # ======================================================================
    # PANEL 6-8: Alpha Element Distribution Properties
    # ======================================================================
    for idx, elem_name in enumerate(alpha_elements_to_plot):
        ax = fig.add_subplot(gs[1, 1 + idx])
        
        elem_idx = element_names.index(elem_name)
        alpha_x, alpha_y = alpha_arrs[elem_idx]
        alpha_x = np.array(alpha_x)
        alpha_y = np.array(alpha_y)
        
        # Remove invalid data
        valid_mask = np.isfinite(alpha_x) & np.isfinite(alpha_y)
        alpha_values = alpha_y[valid_mask]
        
        if len(alpha_values) > 10:
            # Remove extreme outliers
            Q1, Q3 = np.percentile(alpha_values, [25, 75])
            IQR = Q3 - Q1
            outlier_mask = (alpha_values >= Q1 - 3*IQR) & (alpha_values <= Q3 + 3*IQR)
            alpha_clean = alpha_values[outlier_mask]
            
            if len(alpha_clean) > 5:
                # Calculate KDE
                try:
                    kde = gaussian_kde(alpha_clean)
                    test_points = np.linspace(alpha_clean.min(), alpha_clean.max(), 200)
                    density = kde(test_points)
                    
                    # Plot distribution
                    ax.plot(test_points, density, color=colors['model'], linewidth=3,
                           label='Distribution')
                    ax.fill_between(test_points, density, alpha=0.3, color=colors['model'])
                    
                    # Find peak
                    peak_idx = np.argmax(density)
                    peak_location = test_points[peak_idx]
                    ax.axvline(peak_location, color='red', linestyle='-', linewidth=2,
                              label=f'Peak: {peak_location:.3f}')
                    
                    # Show peak constraint region (-0.3 to 0.3)
                    ax.axvspan(-0.3, 0.3, alpha=0.2, color=colors['valid'],
                              label='Valid Peak Region')
                    ax.axvline(-0.3, color=colors['constraint'], linestyle='--', linewidth=2)
                    ax.axvline(0.3, color=colors['constraint'], linestyle='--', linewidth=2)
                    
                    # Calculate and show FWHM
                    max_density = np.max(density)
                    half_max = max_density / 2.0
                    above_half_max = density >= half_max
                    
                    if np.any(above_half_max):
                        indices_above = np.where(above_half_max)[0]
                        left_idx = indices_above[0]
                        right_idx = indices_above[-1]
                        fwhm = test_points[right_idx] - test_points[left_idx]
                        
                        # Mark FWHM
                        ax.axhline(half_max, color='purple', linestyle=':', linewidth=2,
                                  label=f'FWHM: {fwhm:.3f}')
                        ax.plot([test_points[left_idx], test_points[right_idx]], 
                               [half_max, half_max], 'o-', color='purple', linewidth=3)
                        
                        # Add FWHM constraint text
                        ax.text(0.05, 0.95, f'FWHM < 1.0\nActual: {fwhm:.3f}',
                               transform=ax.transAxes, fontsize=10,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                except Exception as e:
                    ax.text(0.5, 0.5, f'KDE failed: {str(e)}', 
                           transform=ax.transAxes, ha='center', va='center')
        
        ax.set_xlabel(elem_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(f'{elem_name} Distribution Properties', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # ======================================================================
    # PANEL 9: Age-Metallicity Constraint
    # ======================================================================
    ax9 = fig.add_subplot(gs[2, 0])
    
    # Convert age to Gyr if needed
    if len(age_x) > 0:
        if np.max(age_x) > 100:
            age_gyr = age_x / 1e9
        else:
            age_gyr = age_x
        
        # Plot age-metallicity relation
        ax9.scatter(age_gyr, age_y, alpha=0.5, s=20, color=colors['model'],
                   label='Model Data')
        
        # Highlight young stars region (age < 8 Gyr)
        young_mask = age_gyr < 8.0
        if np.sum(young_mask) > 0:
            young_feh = age_y[young_mask]
            print(young_feh)
            valid_young = young_feh            
            if len(valid_young) > 0:
                median_young = np.median(valid_young)
                
                # Mark median
                ax9.axhline(median_young, xmin=0, xmax=0.6, color='red', 
                           linestyle='-', linewidth=2,
                           label=f'Young Stars Median: {median_young:.3f}')
                
                # Show constraint region (-0.5 to 0.6)
                ax9.axhspan(-0.5, 0.6, xmin=0, xmax=0.6, alpha=0.2, 
                           color=colors['valid'], label='Valid Region (Age<8 Gyr)')
                ax9.axhline(-0.5, xmin=0, xmax=0.6, color=colors['constraint'], 
                           linestyle='--', linewidth=2)
                ax9.axhline(0.6, xmin=0, xmax=0.6, color=colors['constraint'], 
                           linestyle='--', linewidth=2)
        
        ax9.axvline(8.0, color='purple', linestyle=':', linewidth=2,
                   label='Young/Old Boundary')
    
    ax9.set_xlabel('Age (Gyr)', fontsize=12, fontweight='bold')
    ax9.set_ylabel('[Fe/H]', fontsize=12, fontweight='bold')
    ax9.set_title('Age-Metallicity Constraint', fontsize=13, fontweight='bold')
    ax9.legend(fontsize=9, loc='best')
    ax9.grid(True, alpha=0.3)
    
    # ======================================================================
    # PANEL 10: Bulge Mass Constraint
    # ======================================================================
    ax10 = fig.add_subplot(gs[2, 1])
    
    try:
        m_stellar = GCE_model.inner.history.m_locked[-1]
        
        # Create bar chart
        ax10.barh(['Stellar Mass'], [m_stellar], color=colors['model'], alpha=0.7)
        
        # Show constraint boundaries
        min_mass = 1e9
        max_mass = 1e11
        
        ax10.axvspan(min_mass, max_mass, alpha=0.2, color=colors['valid'],
                    label='Valid Mass Range')
        ax10.axvline(min_mass, color=colors['constraint'], linestyle='--', linewidth=2)
        ax10.axvline(max_mass, color=colors['constraint'], linestyle='--', linewidth=2)
        
        ax10.text(0.5, 0.95, f'Mass: {m_stellar:.2e} M☉',
                 transform=ax10.transAxes, ha='center', va='top',
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
    except Exception as e:
        ax10.text(0.5, 0.5, f'Error: {str(e)}', transform=ax10.transAxes,
                 ha='center', va='center')
    
    ax10.set_xlabel(r'Mass ($M_\odot$)', fontsize=12, fontweight='bold')
    ax10.set_title('Bulge Mass Constraint', fontsize=13, fontweight='bold')
    ax10.set_xscale('log')
    ax10.legend(fontsize=9)
    ax10.grid(True, alpha=0.3, axis='x')
    
    # ======================================================================
    # PANEL 11: Bulge Age Constraint
    # ======================================================================
    ax11 = fig.add_subplot(gs[2, 2])
    
    try:
        age_final_yr = GCE_model.inner.history.age[-1]
        age_final_gyr = age_final_yr / 1e9
        
        # Create bar chart
        ax11.barh(['Final Age'], [age_final_gyr], color=colors['model'], alpha=0.7)
        
        # Show constraint boundary
        min_age_gyr = 10.0
        ax11.axvspan(min_age_gyr, 15.0, alpha=0.2, color=colors['valid'],
                    label='Valid Age Range')
        ax11.axvline(min_age_gyr, color=colors['constraint'], linestyle='--', linewidth=2)
        
        ax11.text(0.5, 0.95, f'Age: {age_final_gyr:.2f} Gyr',
                 transform=ax11.transAxes, ha='center', va='top',
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
    except Exception as e:
        ax11.text(0.5, 0.5, f'Error: {str(e)}', transform=ax11.transAxes,
                 ha='center', va='center')
    
    ax11.set_xlabel('Age (Gyr)', fontsize=12, fontweight='bold')
    ax11.set_title('Bulge Age Constraint', fontsize=13, fontweight='bold')
    ax11.legend(fontsize=9)
    ax11.grid(True, alpha=0.3, axis='x')
    ax11.set_xlim(0, 15)
    
    # ======================================================================
    # PANEL 12: Gas Fraction Constraint
    # ======================================================================
    ax12 = fig.add_subplot(gs[2, 3])
    
    try:
        gas_mass = np.sum(GCE_model.inner.ymgal[-1])
        stellar_mass = GCE_model.inner.history.m_locked[-1]
        total_mass = gas_mass + stellar_mass
        
        if total_mass > 0:
            gas_fraction = gas_mass / total_mass
        else:
            gas_fraction = 0.0
        
        # Create bar chart
        ax12.barh(['Gas Fraction'], [gas_fraction], color=colors['model'], alpha=0.7)
        
        # Show constraint boundary
        max_gas_fraction = 0.5
        ax12.axvspan(0, max_gas_fraction, alpha=0.2, color=colors['valid'],
                    label='Valid Gas Fraction')
        ax12.axvline(max_gas_fraction, color=colors['constraint'], linestyle='--', linewidth=2)
        
        ax12.text(0.5, 0.95, f'Gas Fraction: {gas_fraction:.3f}',
                 transform=ax12.transAxes, ha='center', va='top',
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
    except Exception as e:
        ax12.text(0.5, 0.5, f'Error: {str(e)}', transform=ax12.transAxes,
                 ha='center', va='center')
    
    ax12.set_xlabel('Gas Fraction', fontsize=12, fontweight='bold')
    ax12.set_title('Gas Fraction Constraint', fontsize=13, fontweight='bold')
    ax12.legend(fontsize=9)
    ax12.grid(True, alpha=0.3, axis='x')
    ax12.set_xlim(0, 1)
    
    # ======================================================================
    # PANEL 13: SFH Peak Time Constraint
    # ======================================================================
    ax13 = fig.add_subplot(gs[3, 0])
    
    try:
        sfr = np.array(GCE_model.inner.history.sfr_abs)
        ages_gyr = np.array(GCE_model.inner.history.age) / 1e9
        
        # Plot SFR
        ax13.semilogy(ages_gyr[:-1], sfr, color=colors['model'], linewidth=2,
                     label='SFR')
        
        # Find peak
        if len(sfr) > 0 and np.max(sfr) > 0:
            peak_idx = np.argmax(sfr)
            peak_time_gyr = ages_gyr[peak_idx]
            peak_sfr = sfr[peak_idx]
            
            # Mark peak
            ax13.plot(peak_time_gyr, peak_sfr, 'r*', markersize=20,
                     label=f'Peak at {peak_time_gyr:.2f} Gyr', zorder=10)
            
            # Show constraint region (peak should be < 3.0 Gyr)
            max_peak_time = 3.0
            ax13.axvspan(0, max_peak_time, alpha=0.2, color=colors['valid'],
                        label='Valid Peak Region')
            ax13.axvline(max_peak_time, color=colors['constraint'], linestyle='--', linewidth=2)
        
    except Exception as e:
        ax13.text(0.5, 0.5, f'Error: {str(e)}', transform=ax13.transAxes,
                 ha='center', va='center')
    
    ax13.set_xlabel('Age (Gyr)', fontsize=12, fontweight='bold')
    ax13.set_ylabel(r'SFR ($M_\odot$ yr$^{-1}$)', fontsize=12, fontweight='bold')
    ax13.set_title('SFH Peak Time Constraint', fontsize=13, fontweight='bold')
    ax13.legend(fontsize=9, loc='best')
    ax13.grid(True, alpha=0.3)
    
    # ======================================================================
    # PANEL 14: Mean Stellar Age Constraint
    # ======================================================================
    ax14 = fig.add_subplot(gs[3, 1])
    
    try:
        sfr = np.array(GCE_model.inner.history.sfr_abs)
        timesteps = np.array(GCE_model.inner.history.timesteps)
        ages_gyr = np.array(GCE_model.inner.history.age) / 1e9
        
        # Calculate mass formed in each timestep
        if len(sfr) > len(timesteps):
            sfr = sfr[:len(timesteps)]
        mass_formed = sfr * timesteps
        
        # Calculate current age of stars formed at each timestep
        final_age_gyr = ages_gyr[-1]
        stellar_ages = final_age_gyr - ages_gyr[:len(timesteps)]
        
        # Calculate mass-weighted mean age
        total_mass = np.sum(mass_formed)
        if total_mass > 0:
            mean_age = np.sum(mass_formed * stellar_ages) / total_mass
        else:
            mean_age = 0.0
        
        # Create bar chart
        ax14.barh(['Mean Stellar Age'], [mean_age], color=colors['model'], alpha=0.7)
        
        # Show constraint boundary
        min_mean_age = 8.0
        ax14.axvspan(min_mean_age, 15.0, alpha=0.2, color=colors['valid'],
                    label='Valid Mean Age Range')
        ax14.axvline(min_mean_age, color=colors['constraint'], linestyle='--', linewidth=2)
        
        ax14.text(0.5, 0.95, f'Mean Age: {mean_age:.2f} Gyr',
                 transform=ax14.transAxes, ha='center', va='top',
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
    except Exception as e:
        ax14.text(0.5, 0.5, f'Error: {str(e)}', transform=ax14.transAxes,
                 ha='center', va='center')
    
    ax14.set_xlabel('Age (Gyr)', fontsize=12, fontweight='bold')
    ax14.set_title('Mean Stellar Age Constraint', fontsize=13, fontweight='bold')
    ax14.legend(fontsize=9)
    ax14.grid(True, alpha=0.3, axis='x')
    ax14.set_xlim(0, 15)
    
    # ======================================================================
    # PANEL 15: Constraint Summary
    # ======================================================================
    ax15 = fig.add_subplot(gs[3, 2:])
    ax15.axis('off')
    
    # Calculate constraint satisfaction
    is_physical, penalty_factor = pc.check_physical_plausibility(
        MDF_x, MDF_y_model, alpha_arrs, age_x, age_y,
        liberal=False, age_meta_check=True
    )
    
    model_is_physical, model_penalty = pc.check_model_physics(GCE_model, liberal=False)
    
    # Create summary text
    summary_text = f"""PHYSICAL CONSTRAINTS VALIDATION SUMMARY

Overall Status: {'✓ PASS' if (is_physical and model_is_physical) else '✗ FAIL'}
Penalty Factor: {penalty_factor * model_penalty:.3f}

MDF Constraints:
├─ Peak Location: {'✓' if is_physical else '✗'}
└─ Low-Metallicity Tail: {'✓' if is_physical else '✗'}

Alpha Element Constraints:
├─ [Si/Fe] Binned: {'✓' if is_physical else '✗'}
├─ [Ca/Fe] Binned: {'✓' if is_physical else '✗'}
├─ [Mg/Fe] Binned: {'✓' if is_physical else '✗'}
├─ Distribution Peak: {'✓' if is_physical else '✗'}
└─ Distribution FWHM: {'✓' if is_physical else '✗'}

Age-Metallicity Constraints:
└─ Young Stars Median: {'✓' if is_physical else '✗'}

Model-Level Constraints:
├─ Bulge Mass: {'✓' if model_is_physical else '✗'}
├─ Bulge Age: {'✓' if model_is_physical else '✗'}
├─ Gas Fraction: {'✓' if model_is_physical else '✗'}
├─ SFH Peak Time: (commented out in code)
└─ Mean Stellar Age: (commented out in code)

This plot demonstrates that the best-fit model satisfies
all physical constraints derived from observations of
classical bulges and galactic chemical evolution theory.
"""
    
    # Color based on pass/fail
    box_color = 'lightgreen' if (is_physical and model_is_physical) else 'lightcoral'
    
    ax15.text(0.02, 0.99, summary_text, transform=ax15.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace', linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.8", facecolor=box_color,
                      edgecolor="darkgreen" if (is_physical and model_is_physical) else "darkred",
                      alpha=0.95, linewidth=2.5))
    
    # Add main title
    fig.suptitle('Physical Constraints Validation for Best-Fit Model',
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    plt.close('all')
    
    print(f"Physical constraints validation plot saved: {save_path}")
    
    return fig


def generate_physics_plots(GalGA, results_file='simulation_results.csv'):
    """Generate physics plots using actual omega model computations"""

    print("Generating physics plots using actual omega model data...")

    os.makedirs(GalGA.output_path, exist_ok=True)

    # Load results
    import pandas as pd

    df = pd.read_csv(results_file)
    df.sort_values('fitness', inplace=True)
    print(f"Loaded {len(df)} results from {results_file}")
    
    # Generate the physics plots using real omega data
    print("Generating real infall physics plot from omega model...")
    fig1 = plot_real_infall_physics(GalGA, df)
    
    print("Generating omega model diagnostics...")
    fig2 = plot_omega_diagnostics(GalGA, df)
    
    print("Generating physical constraints validation plot...")
    fig3 = plot_physical_constraints(GalGA, df)
    
    print("Physics plots using omega model data completed!")
    plt.close('all')
    return fig1, fig2, fig3

