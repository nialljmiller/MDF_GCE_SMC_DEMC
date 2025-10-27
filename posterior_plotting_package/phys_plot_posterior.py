import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
sys.path.append('../')
from JINAPyCEE import omega_plus

# from plotting.style import *
# use_paper_style()

# Import posterior utilities
from posterior_plotting_package.posterior_utils import get_weighted_posterior_samples, posterior_resample
from posterior_plotting_package.posterior_utils_density import plot_density_posterior_simple


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
    
    print("Reconstructing omega_plus model...")
    GCE_model = omega_plus.omega_plus(**kwargs)
    print("Model reconstruction successful!")
    
    return GCE_model


def reconstruct_model_from_row(GalGA, row):
    """Reconstruct omega_plus model from a single parameter row"""
    
    comp_idx = int(row['comp_idx'])
    imf_idx = int(row['imf_idx'])
    sn1a_idx = int(row['sn1a_idx'])
    sy_idx = int(row['sy_idx'])
    sn1ar_idx = int(row['sn1ar_idx'])
    
    sigma_2 = row['sigma_2']
    t_1 = row['t_1']
    t_2 = row['t_2']
    infall_1 = row['infall_1']
    infall_2 = row['infall_2']
    sfe_val = row['sfe']
    delta_sfe_val = row['delta_sfe']
    imf_upper = row['imf_upper']
    mgal = row['mgal']
    nb = row['nb']
    
    # Get the parameter arrays from GalGA
    comp = GalGA.comp_array[comp_idx]
    imf_val = GalGA.imf_array[imf_idx]
    sn1a = GalGA.sn1a_assumptions[sn1a_idx]
    sy = GalGA.stellar_yield_assumptions[sy_idx]
    sn1ar = GalGA.sn1a_rates[sn1ar_idx]
    
    # Reconstruct the model
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
    
    GCE_model = omega_plus.omega_plus(**kwargs)
    return GCE_model


def extract_physics_from_model(GCE_model):
    """Extract physical quantities from omega_plus model"""
    
    ages = np.array(GCE_model.inner.history.age) / 1e9
    timesteps_yr = np.array(GCE_model.inner.history.timesteps)
    inflow_masses = np.array(GCE_model.inner.m_inflow_t)
    outflow_masses = np.array(GCE_model.inner.m_outflow_t)
    
    sfr_rates = np.array(GCE_model.inner.history.sfr_abs)[:len(timesteps_yr)]
    metallicity = np.array(GCE_model.inner.history.metallicity)[:len(timesteps_yr)]
    
    inflow_rates = inflow_masses / timesteps_yr
    outflow_rates = outflow_masses / timesteps_yr
    
    gas_masses = np.array([np.sum(GCE_model.inner.ymgal[i]) for i in range(len(GCE_model.inner.ymgal))])
    stellar_masses_raw = np.array(GCE_model.inner.history.m_locked)
    
    if len(stellar_masses_raw) < len(ages):
        stellar_masses = np.append(stellar_masses_raw, stellar_masses_raw[-1])
    else:
        stellar_masses = stellar_masses_raw[:len(ages)]
    
    return {
        'ages': ages,
        'timesteps_yr': timesteps_yr,
        'inflow_rates': inflow_rates,
        'outflow_rates': outflow_rates,
        'sfr_rates': sfr_rates,
        'metallicity': metallicity,
        'gas_masses': gas_masses,
        'stellar_masses': stellar_masses
    }


def compute_physics_ensemble(GalGA, top_df, weights, max_models=2000000000):
    """
    Compute physics ensemble by reconstructing multiple models.
    
    Parameters
    ----------
    GalGA : object
        GalGA object
    top_df : pd.DataFrame
        Top percentile models
    weights : np.ndarray
        Model weights
    max_models : int
        Maximum number of models to reconstruct (for computational efficiency)
    
    Returns
    -------
    ensemble : dict
        Dictionary with median and percentile bands for each physical quantity
    """
    
    # Limit to top N models for computational efficiency
    if len(top_df) > max_models:
        print(f"Subsampling {max_models} models from top {len(top_df)} for computational efficiency...")
        # Resample with replacement according to weights
        indices = np.random.choice(len(top_df), size=max_models, replace=True, p=weights)
        top_df_subset = top_df.iloc[indices].reset_index(drop=True)
        weights_subset = np.ones(max_models) / max_models
    else:
        top_df_subset = top_df
        weights_subset = weights
    
    # Reconstruct each model and extract physics
    physics_samples = {
        'sfr': [], 'inflow': [], 'outflow': [], 
        'gas_mass': [], 'stellar_mass': [], 'metallicity': []
    }
    
    age_arrays = []
    
    for idx, row in top_df_subset.iterrows():
        try:
            print(f"  Reconstructing model {idx+1}/{len(top_df_subset)}...")
            GCE_model = reconstruct_model_from_row(GalGA, row)
            phys = extract_physics_from_model(GCE_model)
            
            age_arrays.append(phys['ages'][:-1])
            physics_samples['sfr'].append(phys['sfr_rates'])
            physics_samples['inflow'].append(phys['inflow_rates'])
            physics_samples['outflow'].append(phys['outflow_rates'])
            physics_samples['gas_mass'].append(phys['gas_masses'][:-1])
            physics_samples['stellar_mass'].append(phys['stellar_masses'][:-1])
            physics_samples['metallicity'].append(phys['metallicity'])
            
        except Exception as e:
            print(f"  Warning: Failed to reconstruct model {idx}: {e}")
            continue
    
    if len(age_arrays) == 0:
        print("Error: No models successfully reconstructed")
        return None
    
    # Define common age grid
    age_common = np.linspace(0, 14.0, 200)
    
    # Interpolate each quantity to common grid and compute percentiles
    from scipy.interpolate import interp1d
    from posterior_plotting_package.posterior_utils import weighted_quantile
    
    ensemble = {}
    
    for key in ['sfr', 'inflow', 'outflow', 'gas_mass', 'stellar_mass', 'metallicity']:
        samples_interp = []
        
        for ages, values in zip(age_arrays, physics_samples[key]):
            # Interpolate to common grid
            if len(ages) > 1 and len(values) > 1:
                f = interp1d(ages, values, kind='linear', bounds_error=False, fill_value=np.nan)
                values_interp = f(age_common)
                samples_interp.append(values_interp)
        
        if len(samples_interp) == 0:
            continue
        
        samples_interp = np.array(samples_interp)
        
        # Compute weighted percentiles at each age
        median = np.zeros(len(age_common))
        lower = np.zeros(len(age_common))
        upper = np.zeros(len(age_common))
        
        for i in range(len(age_common)):
            valid = np.isfinite(samples_interp[:, i])
            if np.sum(valid) > 0:
                w_valid = weights_subset[:len(samples_interp)][valid]
                w_valid = w_valid / np.sum(w_valid)
                pcts = weighted_quantile(samples_interp[valid, i], [0.16, 0.50, 0.84], w_valid)
                lower[i], median[i], upper[i] = pcts
            else:
                lower[i] = median[i] = upper[i] = np.nan
        
        ensemble[key] = {
            'x': age_common,
            'median': median,
            'lower': lower,
            'upper': upper
        }
    
    return ensemble






def choose_cutoff_lognorm_mixture(df_sorted, bins=100, kde_points=1024, em_max_iter=200, tol=1e-6, force_k2=False):
    """
    Simple, reviewer-proof cutoff:
      - Work in y = log(loss).
      - Fit K=1 and K=2 Gaussian mixtures in y by EM; pick K by BIC (unless force_k2=True).
      - If K=2: cutoff = equal-responsibility boundary where pi1*N1(y)=pi2*N2(y).
      - If K=1: no hard cut (use all models).
    Writes two plots and a small audit file; returns cutoff & realized keep fraction.
    """
    import os, numpy as np, matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde, norm
    from scipy.special import logsumexp

    # ---------- data ----------
    L = np.asarray(df_sorted['fitness'].values, float)
    L = L[np.isfinite(L)]
    if L.size == 0:
        raise RuntimeError("No finite losses/fitness values.")
    eps = 1e-12
    y = np.log(L + eps)
    N = y.size

    # ---------- helper: EM for 1D K-component Gaussian mixture ----------
    def em_gmm_1d(y, K, iters=200, tol=1e-6):
        # init by quantiles
        qs = np.linspace(0.2, 0.8, K)
        mu = np.quantile(y, qs) if K > 1 else np.array([float(np.mean(y))])
        s0 = float(np.std(y))
        s0 = s0 if s0 > 1e-6 else 0.1
        sig = np.full(K, s0, float)
        pi = np.full(K, 1.0 / K, float)

        c_norm = -0.5*np.log(2*np.pi)

        def logpdf(y, mu, sig):
            return c_norm - np.log(sig) - 0.5*((y - mu)/sig)**2

        prev_ll = -np.inf
        for _ in range(iters):
            # E-step: responsibilities (log-space)
            log_comp = np.stack([np.log(pi[k]) + logpdf(y, mu[k], sig[k] + 1e-12) for k in range(K)], axis=1)
            log_den = logsumexp(log_comp, axis=1, keepdims=True)
            R = np.exp(log_comp - log_den)  # N x K
            Nk = R.sum(axis=0) + 1e-12

            # M-step
            mu_new = (R * y[:, None]).sum(axis=0) / Nk
            sig_new = np.sqrt((R * (y[:, None] - mu_new[None, :])**2).sum(axis=0) / Nk)
            sig_new = np.maximum(sig_new, 1e-6)
            pi_new = Nk / N

            # log-likelihood
            ll = float(np.sum(log_den))
            if abs(ll - prev_ll) < tol:
                mu, sig, pi = mu_new, sig_new, pi_new
                prev_ll = ll
                break
            mu, sig, pi, prev_ll = mu_new, sig_new, pi_new, ll

        # BIC: p = (K-1) + K (means) + K (stds) = 2K - 1
        bic = -2.0*prev_ll + (2*K - 1)*np.log(N)
        # order by mean
        order = np.argsort(mu)
        return pi[order], mu[order], sig[order], prev_ll, bic

    # ---------- fit K=1 and K=2 ----------
    pi1, mu1, sg1, ll1, bic1 = em_gmm_1d(y, 1, em_max_iter, tol)
    pi2, mu2, sg2, ll2, bic2 = em_gmm_1d(y, 2, em_max_iter, tol)
    choose_K2 = force_k2 or (bic2 < bic1)

    # ---------- cutoff (if K=2), else None ----------
    loss_cutoff = None
    chosen_K = 2 if choose_K2 else 1
    if choose_K2:
        # components already ordered: comp0 is the elite (lower mu)
        pi = pi2; mu = mu2; sig = sg2

        # Solve pi0*N0(y) = pi1*N1(y) analytically
        A = 0.5*(1.0/sig[1]**2 - 1.0/sig[0]**2)
        B = (mu[0]/sig[0]**2 - mu[1]/sig[1]**2)
        D = 0.5*(mu[1]**2/sig[1]**2 - mu[0]**2/sig[0]**2)
        const = np.log((pi[1]/sig[1])/(pi[0]/sig[0]))
        C = D - const

        if abs(A) < 1e-12:
            y_cut = -C / (B + 1e-12)  # equal-variance fallback
        else:
            disc = max(B*B - 4*A*C, 0.0)
            roots = np.sort(( -B + np.array([-1.0, 1.0])*np.sqrt(disc) ) / (2*A))
            # prefer a root between the two means; otherwise, nearest to their midpoint
            mid = 0.5*(mu[0] + mu[1])
            if (mu[0] <= roots[0] <= mu[1]) or (mu[0] <= roots[1] <= mu[1]):
                y_cut = roots[0] if (mu[0] <= roots[0] <= mu[1]) else roots[1]
            else:
                y_cut = roots[np.argmin(np.abs(roots - mid))]

        loss_cutoff = float(np.exp(y_cut))
        frac = float(np.mean(L <= loss_cutoff))
    else:
        # no hard selection
        frac = 1.0

    pct = 100.0 * frac
    
    return pct




def plot_real_infall_physics(GalGA, results_df=None, save_path='Real_Infall_Physics_Posterior.png',
                             use_posterior=True, percentile=10, max_models=20):
    """
    Generate physics visualization with posterior uncertainty bands.
    
    Parameters
    ----------
    GalGA : object
        GalGA object
    results_df : pd.DataFrame, optional
        Results dataframe
    save_path : str
        Output file path
    use_posterior : bool
        If True, plot median + 1σ bands from top percentile
    percentile : float
        Top X% of models to include in posterior
    max_models : int
        Maximum number of models to reconstruct for ensemble
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """


    if percentile == -1:
        percentile = choose_cutoff_lognorm_mixture(results_df, bins=100, kde_points=1024, em_max_iter=200, tol=1e-6)

    
    save_path = GalGA.output_path + save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Define color palette
    colors = {
        'inflow': '#1f77b4',
        'outflow': '#d62728',
        'sfr': '#ff7f0e',
        'gas': '#2ca02c',
        'stellar': '#9467bd',
        'metallicity': '#8c564b',
        'efficiency': '#e377c2',
        'loading': '#7f7f7f'
    }
    
    # Extract best-fit parameters for annotations
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
    
    # Compute physics ensemble or use single best model
    if use_posterior and results_df is not None and not results_df.empty:
        print(f"Computing physics posterior from top {percentile}% of models...")

        draws_df, draw_w = posterior_resample(
            results_df,
            weight_col='posterior_w',     # if you have it; else remove so it falls back
            fitness_col='fitness',
            percentile=percentile,        # optional guard; you can set None if you want "use all"
            n_draws=max_models,
            resampling='systematic'
        )

        ensemble = compute_physics_ensemble(GalGA, draws_df, draw_w, max_models=max_models)
        
        if ensemble is not None:
            # Extract median and bands
            ages = ensemble['sfr']['x']
            sfr_median = ensemble['sfr']['median']
            sfr_lower = ensemble['sfr']['lower']
            sfr_upper = ensemble['sfr']['upper']
            
            inflow_median = ensemble['inflow']['median']
            inflow_lower = ensemble['inflow']['lower']
            inflow_upper = ensemble['inflow']['upper']
            
            outflow_median = ensemble['outflow']['median']
            outflow_lower = ensemble['outflow']['lower']
            outflow_upper = ensemble['outflow']['upper']
            
            gas_median = ensemble['gas_mass']['median']
            gas_lower = ensemble['gas_mass']['lower']
            gas_upper = ensemble['gas_mass']['upper']
            
            stellar_median = ensemble['stellar_mass']['median']
            stellar_lower = ensemble['stellar_mass']['lower']
            stellar_upper = ensemble['stellar_mass']['upper']
            
            metal_median = ensemble['metallicity']['median']
            metal_lower = ensemble['metallicity']['lower']
            metal_upper = ensemble['metallicity']['upper']
        else:
            print("Warning: Could not compute physics ensemble, falling back to best model")
            use_posterior = False

    
    # Fallback to single best model
    if not use_posterior:
        GCE_model = reconstruct_best_model(GalGA, results_df)
        phys = extract_physics_from_model(GCE_model)
        
        ages = phys['ages'][:-1]
        sfr_median = phys['sfr_rates']
        inflow_median = phys['inflow_rates']
        outflow_median = phys['outflow_rates']
        gas_median = phys['gas_masses'][:-1]
        stellar_median = phys['stellar_masses'][:-1]
        metal_median = phys['metallicity']
        
        # No uncertainty bands in legacy mode
        sfr_lower = sfr_upper = None
        inflow_lower = inflow_upper = None
        outflow_lower = outflow_upper = None
        gas_lower = gas_upper = None
        stellar_lower = stellar_upper = None
        metal_lower = metal_upper = None
    
    # Create figure
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3, 
                  left=0.06, right=0.98, top=0.94, bottom=0.06)
    
    # ======================================================================
    # PANEL 1: Inflow Rate with Theoretical Overlay
    # ======================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    if use_posterior and inflow_lower is not None:
        # Plot with density shading
        plot_density_posterior_simple(ax1, ages, inflow_median,
                                     inflow_lower, inflow_upper,
                                     color=colors['inflow'], n_levels=20,
                                     zorder=2, label='1σ posterior')
    else:
        # Legacy mode: just plot median
        ax1.plot(ages, inflow_median, color=colors['inflow'], linewidth=3,
                label='Median Inflow Rate', marker='o', markersize=4, alpha=0.9)
    
    # Theoretical infall episodes
    t_theory = np.linspace(0, ages[-1], 1000)
    infall_1_theory = np.exp(-t_theory / infall_1) * np.heaviside(t_theory - t_1, 1)
    infall_2_theory = np.exp(-(t_theory - t_2) / infall_2) * np.heaviside(t_theory - t_2, 1)
    
    if np.max(infall_1_theory) > 0:
        infall_1_norm = np.max(inflow_median) * infall_1_theory / np.max(infall_1_theory)
        ax1.plot(t_theory, infall_1_norm, '--', color='lightblue', linewidth=2, alpha=0.7,
                 label=f'First Episode (τ={infall_1:.2f} Gyr)')
    
    if np.max(infall_2_theory) > 0:
        infall_2_norm = np.max(inflow_median) * 0.3 * infall_2_theory / np.max(infall_2_theory)
        ax1.plot(t_theory, infall_2_norm, '--', color='salmon', linewidth=2, alpha=0.7,
                 label=f'Second Episode (τ={infall_2:.2f} Gyr)')
    
    ax1.axvline(t_1, color='steelblue', linestyle=':', linewidth=2, alpha=0.8)
    ax1.axvline(t_2, color='crimson', linestyle=':', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Universe Age (Gyr)', fontsize=14, fontweight='bold')
    ax1.set_ylabel(r'Inflow Rate ($M_\odot$ yr$^{-1}$)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax1.set_xlim(0, ages[-1])
    ax1.set_ylim(bottom=0)
    
    # ======================================================================
    # PANEL 2: Star Formation History
    # ======================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    if use_posterior and sfr_lower is not None:
        # Plot with density shading (handle log scale)
        plot_density_posterior_simple(ax2, ages, sfr_median,
                                     np.maximum(sfr_lower, 1e-10), sfr_upper,
                                     color=colors['sfr'], n_levels=20,
                                     zorder=2, label='1σ posterior')
        ax2.set_yscale('log')
    else:
        # Legacy mode
        ax2.semilogy(ages, sfr_median, color=colors['sfr'], linewidth=2.5,
                    label='Median SFR', marker='s', markersize=3)
    
    ax2.axvline(t_2, color='crimson', linestyle=':', alpha=0.7, linewidth=2)
    ax2.text(t_2 + 0.2, np.max(sfr_median) * 0.1, f'ΔSFE = {delta_sfe_val:+.4f}', 
             rotation=90, fontsize=10, alpha=0.8, fontweight='bold')
    
    ax2.set_xlabel('Universe Age (Gyr)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'SFR ($M_\odot$ yr$^{-1}$)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize=11)
    
    # ======================================================================
    # PANEL 3: Gas Flows
    # ======================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    
    if use_posterior:
        if inflow_lower is not None:
            plot_density_posterior_simple(ax3, ages, inflow_median,
                                         inflow_lower, inflow_upper,
                                         color=colors['inflow'], n_levels=15,
                                         zorder=2, label='Median Inflow')
        else:
            ax3.plot(ages, inflow_median, color=colors['inflow'], linewidth=2,
                    label='Median Inflow', marker='o', markersize=2, alpha=0.8)
        
        if outflow_lower is not None:
            plot_density_posterior_simple(ax3, ages, outflow_median,
                                         outflow_lower, outflow_upper,
                                         color=colors['outflow'], n_levels=15,
                                         zorder=2, label='Median Outflow')
        else:
            ax3.plot(ages, outflow_median, color=colors['outflow'], linewidth=2,
                    label='Median Outflow', marker='^', markersize=2, alpha=0.8)
    else:
        ax3.plot(ages, inflow_median, color=colors['inflow'], linewidth=2,
                label='Median Inflow', marker='o', markersize=2, alpha=0.8)
        ax3.plot(ages, outflow_median, color=colors['outflow'], linewidth=2,
                label='Median Outflow', marker='^', markersize=2, alpha=0.8)
    
    ax3.set_xlabel('Universe Age (Gyr)', fontsize=12, fontweight='bold')
    ax3.set_ylabel(r'Flow Rate ($M_\odot$ yr$^{-1}$)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.2)
    ax3.legend(fontsize=11)
    
    # ======================================================================
    # PANEL 4: Metallicity Evolution
    # ======================================================================
    ax4 = fig.add_subplot(gs[1, 2])
    
    if use_posterior and metal_lower is not None:
        plot_density_posterior_simple(ax4, ages, metal_median,
                                     metal_lower, metal_upper,
                                     color=colors['metallicity'], n_levels=20,
                                     zorder=2, label='1σ posterior')
    else:
        ax4.plot(ages, metal_median, color=colors['metallicity'], linewidth=2.5,
                label='Median [Fe/H]', marker='o', markersize=3)
    
    ax4.set_xlabel('Universe Age (Gyr)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('[Fe/H]', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.2)
    ax4.legend(fontsize=11)
    
    # ======================================================================
    # PANEL 5: Reservoir Masses
    # ======================================================================
    ax5 = fig.add_subplot(gs[2, :2])
    
    if use_posterior:
        if gas_lower is not None:
            plot_density_posterior_simple(ax5, ages, gas_median,
                                         np.maximum(gas_lower, 1e6), gas_upper,
                                         color=colors['gas'], n_levels=15,
                                         zorder=2, label='Median Gas')
            ax5.set_yscale('log')
        else:
            ax5.semilogy(ages, gas_median, color=colors['gas'], linewidth=3,
                        label='Median Gas', marker='o', markersize=3, alpha=0.9)
        
        if stellar_lower is not None:
            plot_density_posterior_simple(ax5, ages, stellar_median,
                                         np.maximum(stellar_lower, 1e6), stellar_upper,
                                         color=colors['stellar'], n_levels=15,
                                         zorder=2, label='Median Stellar')
            if gas_lower is None:  # Only set if not already set
                ax5.set_yscale('log')
        else:
            ax5.semilogy(ages, stellar_median, color=colors['stellar'], linewidth=3,
                        label='Median Stellar', marker='s', markersize=3, alpha=0.9)
    else:
        ax5.semilogy(ages, gas_median, color=colors['gas'], linewidth=3,
                    label='Median Gas', marker='o', markersize=3, alpha=0.9)
        ax5.semilogy(ages, stellar_median, color=colors['stellar'], linewidth=3,
                    label='Median Stellar', marker='s', markersize=3, alpha=0.9)
    
    total_median = gas_median + stellar_median
    ax5.semilogy(ages, total_median, color='black', linewidth=2, linestyle='--', 
                 label='Total mass', alpha=0.7)
    
    ax5.axvline(t_1, color='steelblue', linestyle=':', alpha=0.6)
    ax5.axvline(t_2, color='crimson', linestyle=':', alpha=0.6)
    
    ax5.set_xlabel('Universe Age (Gyr)', fontsize=12, fontweight='bold')
    ax5.set_ylabel(r'Reservoir Mass ($M_\odot$)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.2)
    ax5.legend(fontsize=11, loc='best')
    
    # ======================================================================
    # PANEL 6: Summary Text
    # ======================================================================
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.axis('off')
    
    mode_text = "POSTERIOR MODE" if use_posterior else "BEST MODEL MODE"
    summary_text = f"""{mode_text}

Two-Infall Parameters:
├─ σ₂ = {sigma_2:.1f}
├─ Episode I: t₁ = {t_1:.3f} Gyr, τ₁ = {infall_1:.2f} Gyr
├─ Episode II: t₂ = {t_2:.1f} Gyr, τ₂ = {infall_2:.2f} Gyr  
└─ SFE Evolution: {sfe_val:.4f} → {sfe_val + delta_sfe_val:.4f}

Final Masses (median):
├─ Stellar: {stellar_median[-1]:.2e} M☉
└─ Gas: {gas_median[-1]:.2e} M☉
"""
    
    if use_posterior:
        summary_text += f"\nPosterior computed from top {percentile}% of models"
    
    ax6.text(0.02, 0.99, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace', linespacing=1.4,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightcyan", 
                      edgecolor="steelblue", alpha=0.95, linewidth=1.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    plt.close('all')
    
    print(f"Physics plot with posterior saved: {save_path}")
    
    return fig


if __name__ == '__main__':
    print("Physics plots module with posterior uncertainty bands loaded.")

