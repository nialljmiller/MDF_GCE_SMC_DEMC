"""
Physical constraints for Galactic Chemical Evolution models.

This module provides penalty functions that enforce physically plausible
parameter combinations and model outputs.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any


def apply_physics_penalty(
    model_outputs: Dict[str, Any],
    params: Dict[str, float],
    penalty_scale: float = 1.0,
) -> float:
    """
    Apply physics-based penalty to model fitness.
    
    Penalizes unphysical parameter combinations and model outputs.
    
    Parameters
    ----------
    model_outputs : dict
        Dictionary containing model outputs:
        - 'gas_mass_final': Final gas mass
        - 'stellar_mass_final': Final stellar mass
        - 'sfr_history': Star formation rate history
        - 'metallicity_final': Final metallicity
    params : dict
        Dictionary of model parameters
    penalty_scale : float
        Overall scaling factor for penalties
        
    Returns
    -------
    float
        Penalty value (0 = no penalty, higher = worse)
    """
    penalty = 0.0
    
    # Check temporal ordering: t2 should be > t1
    t1 = params.get('t_1', params.get('tmax_1', 0))
    t2 = params.get('t_2', params.get('tmax_2', 0))
    if t2 < t1 + 0.5:
        penalty += 10.0 * (t1 + 0.5 - t2)
    
    # Check infall timescales are positive
    tau1 = params.get('infall_1', params.get('infall_timescale_1', 0.1))
    tau2 = params.get('infall_2', params.get('infall_timescale_2', 0.1))
    if tau1 <= 0:
        penalty += 100.0
    if tau2 <= 0:
        penalty += 100.0
    
    # Check SFE is positive
    sfe = params.get('sfe', 1.0)
    if sfe <= 0:
        penalty += 100.0
    
    # Check mass ratio is physical
    sigma_2 = params.get('sigma_2', 1.0)
    if sigma_2 < 0:
        penalty += 100.0 * abs(sigma_2)
    
    # Model output checks (if available)
    if model_outputs:
        # Final gas mass shouldn't be negative
        gas_final = model_outputs.get('gas_mass_final', None)
        if gas_final is not None and gas_final < 0:
            penalty += 50.0 * abs(gas_final)
        
        # Final stellar mass should be positive
        stellar_final = model_outputs.get('stellar_mass_final', None)
        if stellar_final is not None and stellar_final <= 0:
            penalty += 100.0
        
        # Metallicity should be in reasonable range
        z_final = model_outputs.get('metallicity_final', None)
        if z_final is not None:
            if z_final < -3.0 or z_final > 1.0:
                penalty += 10.0 * max(0, -3.0 - z_final, z_final - 1.0)
    
    return penalty * penalty_scale


def check_mdf_constraints(
    mdf_x: np.ndarray,
    mdf_y: np.ndarray,
    obs_mdf_x: np.ndarray,
    obs_mdf_y: np.ndarray,
) -> Dict[str, float]:
    """
    Check MDF-specific physical constraints.
    
    Parameters
    ----------
    mdf_x : np.ndarray
        Model [Fe/H] bins
    mdf_y : np.ndarray
        Model MDF values
    obs_mdf_x : np.ndarray
        Observed [Fe/H] bins
    obs_mdf_y : np.ndarray
        Observed MDF values
        
    Returns
    -------
    dict
        Dictionary of constraint violations and penalties
    """
    constraints = {}
    
    # MDF should be normalized
    model_sum = np.sum(mdf_y)
    if not np.isclose(model_sum, 1.0, rtol=0.1):
        constraints['normalization_penalty'] = abs(model_sum - 1.0) * 10
    
    # MDF should have a peak in reasonable range
    peak_idx = np.argmax(mdf_y)
    peak_feh = mdf_x[peak_idx]
    
    # Bulge MDF typically peaks around -0.2 to +0.3
    if peak_feh < -1.0 or peak_feh > 0.5:
        constraints['peak_location_penalty'] = abs(peak_feh + 0.1) * 5
    
    # Low-metallicity tail shouldn't be too strong
    low_met_mask = mdf_x < -1.5
    if np.any(low_met_mask):
        low_met_fraction = np.sum(mdf_y[low_met_mask])
        if low_met_fraction > 0.1:
            constraints['low_met_tail_penalty'] = (low_met_fraction - 0.1) * 20
    
    return constraints


def check_alpha_constraints(
    alpha_feh: np.ndarray,
    alpha_values: np.ndarray,
    element_name: str = "[Mg/Fe]",
) -> Dict[str, float]:
    """
    Check alpha element abundance constraints.
    
    Parameters
    ----------
    alpha_feh : np.ndarray
        [Fe/H] values
    alpha_values : np.ndarray
        Alpha element abundances
    element_name : str
        Name of the alpha element
        
    Returns
    -------
    dict
        Dictionary of constraint violations
    """
    constraints = {}
    
    # Alpha elements should show knee around -1 to 0
    # High alpha at low metallicity, declining toward solar
    
    low_met_mask = alpha_feh < -1.0
    high_met_mask = alpha_feh > -0.3
    
    if np.any(low_met_mask) and np.any(high_met_mask):
        low_met_alpha = np.median(alpha_values[low_met_mask])
        high_met_alpha = np.median(alpha_values[high_met_mask])
        
        # Should have declining alpha with increasing metallicity
        if high_met_alpha > low_met_alpha:
            constraints['alpha_trend_penalty'] = (high_met_alpha - low_met_alpha) * 10
    
    # Alpha values should be in reasonable range
    if np.any(alpha_values < -0.5) or np.any(alpha_values > 0.8):
        out_of_range = np.sum(alpha_values < -0.5) + np.sum(alpha_values > 0.8)
        constraints['alpha_range_penalty'] = out_of_range * 5
    
    return constraints


def check_age_metallicity_constraints(
    ages: np.ndarray,
    metallicities: np.ndarray,
) -> Dict[str, float]:
    """
    Check age-metallicity relation constraints.
    
    Parameters
    ----------
    ages : np.ndarray
        Stellar ages (Gyr)
    metallicities : np.ndarray
        [Fe/H] values
        
    Returns
    -------
    dict
        Dictionary of constraint violations
    """
    constraints = {}
    
    # General trend: metallicity should increase with decreasing age
    # (younger stars are more metal-rich)
    
    if len(ages) > 10:
        # Compute correlation
        valid = np.isfinite(ages) & np.isfinite(metallicities)
        if np.sum(valid) > 10:
            corr = np.corrcoef(ages[valid], metallicities[valid])[0, 1]
            
            # Expect negative correlation (older = lower metallicity)
            if corr > 0.3:
                constraints['amr_trend_penalty'] = corr * 10
    
    # Oldest stars shouldn't be super metal-rich
    old_mask = ages > 10.0  # > 10 Gyr
    if np.any(old_mask):
        old_metallicity = np.median(metallicities[old_mask])
        if old_metallicity > 0.0:
            constraints['old_star_metallicity_penalty'] = old_metallicity * 20
    
    return constraints


def compute_total_penalty(
    model_outputs: Dict[str, Any],
    params: Dict[str, float],
    mdf_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    obs_mdf_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    alpha_data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    amr_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> float:
    """
    Compute total physics penalty from all constraint checks.
    
    Parameters
    ----------
    model_outputs : dict
        Model output dictionary
    params : dict
        Model parameters
    mdf_data : tuple, optional
        (mdf_x, mdf_y) for model
    obs_mdf_data : tuple, optional
        (obs_x, obs_y) for observations
    alpha_data : dict, optional
        {element: (feh, alpha)} for alpha elements
    amr_data : tuple, optional
        (ages, metallicities) for AMR
        
    Returns
    -------
    float
        Total penalty value
    """
    total = 0.0
    
    # Basic parameter penalties
    total += apply_physics_penalty(model_outputs, params)
    
    # MDF constraints
    if mdf_data is not None and obs_mdf_data is not None:
        mdf_constraints = check_mdf_constraints(
            mdf_data[0], mdf_data[1],
            obs_mdf_data[0], obs_mdf_data[1]
        )
        total += sum(mdf_constraints.values())
    
    # Alpha constraints
    if alpha_data is not None:
        for element, (feh, alpha) in alpha_data.items():
            alpha_constraints = check_alpha_constraints(feh, alpha, element)
            total += sum(alpha_constraints.values())
    
    # AMR constraints
    if amr_data is not None:
        amr_constraints = check_age_metallicity_constraints(
            amr_data[0], amr_data[1]
        )
        total += sum(amr_constraints.values())
    
    return total
