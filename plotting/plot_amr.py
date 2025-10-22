
from plotting.style import *
use_paper_style()





def plot_age_metallicity_curves(GalGA, Fe_H, age_Joyce, age_Bensby, results_df=None, save_path=None):

    """
    Plot all age-metallicity model curves, highlight the best model, overlay data, and show residuals.
    Similar to plot_mdf_curves but for age-metallicity relation.
    
    This function creates a two-panel plot:
    - Top: Age vs [Fe/H] with all models (gray) + best model (red) + observations
    - Bottom: Age vs Residuals (Model - Observations)
    """
    if save_path is None:
        save_path = GalGA.output_path + 'Age_Metallicity_multiple_results.png'
    
    import numpy as np
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    import os
    
    # Check if we have age data
    if not hasattr(GalGA, 'age_data') or len(GalGA.age_data) == 0:
        print("No age data available for plotting")
        return None
    
    # Ensure all inputs are numpy arrays to avoid indexing issues
    Fe_H = np.asarray(Fe_H, dtype=float)
    age_Joyce = np.asarray(age_Joyce, dtype=float)
    age_Bensby = np.asarray(age_Bensby, dtype=float)
    
    # Create figure with subplots - main plot and residuals
    fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=(12, 10), 
                                          gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    
    # Determine best model parameters
    if results_df is not None and not results_df.empty:
        bm = results_df.iloc[0]
        best_params = (bm['sigma_2'], bm['t_2'], bm['infall_2'])
    else:
        r = GalGA.results[0]
        best_params = (r[5], r[7], r[9])
    
    best_flag = False
    best_age_x = None
    best_age_y = None
    
    alpha = 10/len(GalGA.results)
    # Plot all model curves on main panel
    for (age_data, label, res) in zip(GalGA.age_data, GalGA.labels, GalGA.results):
        params = (res[5], res[7], res[9])
        is_best = all(abs(p - b) < 1e-5 for p, b in zip(params, best_params))
        
        # Extract and transform age data
        x_age_raw, y_feh = age_data
        x_age_gyr = (x_age_raw[-1] / 1e9) - np.array(x_age_raw) / 1e9
        
        if is_best:
            best_age_x = np.array(x_age_gyr)
            best_age_y = np.array(y_feh)
            if not best_flag:
                ax_main.plot(x_age_gyr, y_feh, color='red', linewidth=2.5,
                           label="Best", zorder=3)
                best_flag = True
            else:
                ax_main.plot(x_age_gyr, y_feh, color='red', linewidth=2.5, zorder=3)
        else:

            ax_main.plot(x_age_gyr, y_feh, color='gray', alpha=alpha, linewidth=0.5, zorder=1)
    
    # Plot observational data on main panel
    ax_main.scatter(age_Joyce, Fe_H, marker='*', s=60, color='blue', 
                   alpha=0.7, label='Joyce et al.', zorder=2)
    ax_main.scatter(age_Bensby, Fe_H, marker='^', s=60, color='orange', 
                   alpha=0.7, label='Bensby et al.', zorder=2)
    
    # Calculate and plot residuals
    residuals_calculated = False
    
    if best_age_x is not None and best_age_y is not None:
        # Create interpolation function for the best model
        try:
            # Sort model data by age for proper interpolation
            sort_idx = np.argsort(best_age_x)
            sorted_age_x = best_age_x[sort_idx]
            sorted_age_y = best_age_y[sort_idx]
            
            # Remove any duplicate age values that could cause interpolation issues
            unique_mask = np.concatenate(([True], np.diff(sorted_age_x) > 1e-10))
            unique_age_x = sorted_age_x[unique_mask]
            unique_age_y = sorted_age_y[unique_mask]
            
            if len(unique_age_x) > 1:
                interp_func = interp1d(unique_age_x, unique_age_y, kind='linear', 
                                     bounds_error=False, fill_value=np.nan)
                
                # Model age range for filtering observations
                model_age_min, model_age_max = np.min(unique_age_x), np.max(unique_age_x)
                
                # For Joyce data
                joyce_mask = np.isfinite(age_Joyce) & np.isfinite(Fe_H)
                if np.sum(joyce_mask) > 0:
                    joyce_age_filtered = age_Joyce[joyce_mask]
                    joyce_feh_filtered = Fe_H[joyce_mask]
                    
                    # Only use Joyce data within model age range
                    joyce_in_range = ((joyce_age_filtered >= model_age_min) & 
                                     (joyce_age_filtered <= model_age_max))
                    
                    if np.sum(joyce_in_range) > 0:
                        joyce_ages_valid = joyce_age_filtered[joyce_in_range]
                        joyce_feh_valid = joyce_feh_filtered[joyce_in_range]
                        
                        # Interpolate model to Joyce ages
                        model_interp_joyce = interp_func(joyce_ages_valid)
                        
                        # Calculate residuals (model - observations)
                        residuals_joyce = model_interp_joyce - joyce_feh_valid
                        
                        # Plot residuals for valid points
                        valid_joyce_res = np.isfinite(residuals_joyce)
                        if np.sum(valid_joyce_res) > 0:
                            ax_res.scatter(joyce_ages_valid[valid_joyce_res], 
                                         residuals_joyce[valid_joyce_res], 
                                         marker='*', s=40, color='blue', alpha=0.7, 
                                         label='Joyce residuals')
                            residuals_calculated = True
                
                # For Bensby data
                bensby_mask = np.isfinite(age_Bensby) & np.isfinite(Fe_H)
                if np.sum(bensby_mask) > 0:
                    bensby_age_filtered = age_Bensby[bensby_mask]
                    bensby_feh_filtered = Fe_H[bensby_mask]
                    
                    # Only use Bensby data within model age range
                    bensby_in_range = ((bensby_age_filtered >= model_age_min) & 
                                      (bensby_age_filtered <= model_age_max))
                    
                    if np.sum(bensby_in_range) > 0:
                        bensby_ages_valid = bensby_age_filtered[bensby_in_range]
                        bensby_feh_valid = bensby_feh_filtered[bensby_in_range]
                        
                        # Interpolate model to Bensby ages
                        model_interp_bensby = interp_func(bensby_ages_valid)
                        
                        # Calculate residuals (model - observations)
                        residuals_bensby = model_interp_bensby - bensby_feh_valid
                        
                        # Plot residuals for valid points
                        valid_bensby_res = np.isfinite(residuals_bensby)
                        if np.sum(valid_bensby_res) > 0:
                            ax_res.scatter(bensby_ages_valid[valid_bensby_res], 
                                         residuals_bensby[valid_bensby_res], 
                                         marker='^', s=40, color='orange', alpha=0.7, 
                                         label='Bensby residuals')
                            residuals_calculated = True
                
                # Calculate and display RMS residuals
                rms_text = ""
                all_residuals = []
                
                if 'residuals_joyce' in locals():
                    valid_joyce_residuals = residuals_joyce[np.isfinite(residuals_joyce)]
                    if len(valid_joyce_residuals) > 0:
                        rms_joyce = np.sqrt(np.mean(valid_joyce_residuals**2))
                        rms_text += f'Joyce RMS = {rms_joyce:.3f}\n'
                        all_residuals.extend(valid_joyce_residuals)
                
                if 'residuals_bensby' in locals():
                    valid_bensby_residuals = residuals_bensby[np.isfinite(residuals_bensby)]
                    if len(valid_bensby_residuals) > 0:
                        rms_bensby = np.sqrt(np.mean(valid_bensby_residuals**2))
                        rms_text += f'Bensby RMS = {rms_bensby:.3f}'
                        all_residuals.extend(valid_bensby_residuals)
                
                if rms_text:
                    ax_res.text(0.02, 0.95, rms_text.strip(), 
                               transform=ax_res.transAxes, fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                               verticalalignment='top')
                
                # Set reasonable y-limits for residuals
                if len(all_residuals) > 0:
                    res_std = np.std(all_residuals)
                    res_range = max(3*res_std, 0.5)  # Ensure minimum visible range
                    ax_res.set_ylim(-res_range, res_range)
                
        except Exception as e:
            print(f"Warning: Could not calculate residuals - {e}")
            residuals_calculated = False
    
    # Add zero line to residuals regardless of whether we calculated residuals
    ax_res.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    
    # Format main plot
    ax_main.set_ylabel('[Fe/H]', fontsize=14)
    ax_main.set_xlim(0, 14.2)
    ax_main.set_ylim(-2, 1)
    
    # Create legend with multi-line label positioned appropriately
    legend = ax_main.legend(loc='upper left', bbox_to_anchor=(0., 1.), frameon=True, 
                          fontsize=9, facecolor='white', edgecolor='gray')
    legend.get_frame().set_alpha(0.9)
    
    ax_main.tick_params(axis='x', labelbottom=False)  # Remove x-axis labels from main plot
    
    # Format residuals plot
    ax_res.set_xlabel('Age (Gyr)', fontsize=14)
    ax_res.set_ylabel('Model - Obs [Fe/H]', fontsize=12)
    ax_res.set_xlim(0, 14.2)

    plt.tight_layout()
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Age-metallicity curves with residuals saved to {save_path}")
    return fig



