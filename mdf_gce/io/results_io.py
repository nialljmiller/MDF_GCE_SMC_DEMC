#!/usr/bin/env python3
"""
Results I/O for MDF_GCE_SMC_DEMC.

Provides comprehensive saving and loading of GA results including:
- Parameter values and fitness
- MDF curves (mdf_x, mdf_y)
- Age-metallicity curves (age_x, age_y)
- Alpha element tracks ([Mg/Fe], [Si/Fe], [Ca/Fe], [Ti/Fe])

LINKAGE GUARANTEE:
------------------
The save_complete_results() function produces two files that are FULLY LINKED:
- {prefix}results.csv: Contains model_id column + all parameters
- {prefix}curves.npz: Contains model_ids array + all curve data

The model_id in the CSV matches the model_ids array in the NPZ.
To get curves for CSV row with model_id=X:
1. Find index i where curves['model_ids'][i] == X
2. Access curves['mdf_x'][i], curves['mdf_y'][i], etc.

The load_complete_results() function handles this automatically, returning
a DataFrame with curve columns (mdf_x, mdf_y, age_x, age_y, alpha_tracks).

Storage formats:
1. CSV + NPZ: Compatible with basic tools, CSV for parameters, NPZ for curves
2. HDF5: Single file with everything (requires h5py, optional)
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path


# =============================================================================
# CSV + NPZ FORMAT (Default)
# =============================================================================

def save_complete_results(
    evaluation_results: List[Dict],
    output_path: str,
    prefix: str = '',
    save_curves: bool = True,
) -> Tuple[str, str]:
    """
    Save complete GA results to CSV + NPZ files with guaranteed linkage.
    
    CRITICAL: The model_id column in the CSV matches the model_ids array in the NPZ.
    This allows you to find the curves for any parameter set.
    
    Parameters
    ----------
    evaluation_results : list of dict
        Results from GA evaluations, each containing:
        - individual: parameter values (15 elements)
        - fitness: loss value
        - mdf_x, mdf_y: MDF curve arrays
        - age_x, age_y: age-metallicity curve arrays  
        - alpha_data: dict with keys '[Mg/Fe]', '[Si/Fe]', '[Ca/Fe]', '[Ti/Fe]'
    output_path : str
        Output directory
    prefix : str
        Prefix for output files (e.g., 'gen100_' produces gen100_results.csv)
    save_curves : bool
        Whether to save curve data (MDF, AMR, alpha)
        
    Returns
    -------
    csv_path, npz_path : str, str
        Paths to saved files (linked by model_id)
        
    Example
    -------
    >>> save_complete_results(ga.evaluation_results, 'output/', prefix='final_')
    # Produces: final_results.csv + final_curves.npz
    # To load and access curves:
    >>> df = load_complete_results('output/', prefix='final_')
    >>> df.loc[df['model_id'] == 0, 'mdf_x']  # Get MDF x-values for model 0
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Build parameter dataframe
    rows = []
    for i, r in enumerate(evaluation_results):
        if 'individual' not in r:
            continue
            
        ind = r['individual']
        row = {
            'model_id': i,  # Reference to curve data
            'comp_idx': int(ind[0]),
            'imf_idx': int(ind[1]),
            'sn1a_idx': int(ind[2]),
            'sy_idx': int(ind[3]),
            'sn1ar_idx': int(ind[4]),
            'sigma_2': ind[5],
            't_1': ind[6],
            't_2': ind[7],
            'infall_1': ind[8],
            'infall_2': ind[9],
            'sfe': ind[10],
            'delta_sfe': ind[11],
            'imf_upper': ind[12],
            'mgal': ind[13],
            'nb': ind[14],
            'fitness': r.get('fitness', float('inf')),
            'total_mass': r.get('total_mass', np.nan),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_path, f'{prefix}results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} results to {csv_path}")
    
    # Save curves to NPZ
    npz_path = ''
    if save_curves:
        curves = {
            'model_ids': [],
            'mdf_x': [],
            'mdf_y': [],
            'age_x': [],
            'age_y': [],
            'alpha_Mg_x': [],
            'alpha_Mg_y': [],
            'alpha_Si_x': [],
            'alpha_Si_y': [],
            'alpha_Ca_x': [],
            'alpha_Ca_y': [],
            'alpha_Ti_x': [],
            'alpha_Ti_y': [],
        }
        
        for i, r in enumerate(evaluation_results):
            if 'individual' not in r:
                continue
                
            curves['model_ids'].append(i)
            
            # MDF
            if 'mdf_x' in r and 'mdf_y' in r:
                curves['mdf_x'].append(np.asarray(r['mdf_x'], dtype=float))
                curves['mdf_y'].append(np.asarray(r['mdf_y'], dtype=float))
            else:
                curves['mdf_x'].append(np.array([]))
                curves['mdf_y'].append(np.array([]))
            
            # Age-metallicity
            if 'age_x' in r and 'age_y' in r:
                curves['age_x'].append(np.asarray(r['age_x'], dtype=float))
                curves['age_y'].append(np.asarray(r['age_y'], dtype=float))
            else:
                curves['age_x'].append(np.array([]))
                curves['age_y'].append(np.array([]))
            
            # Alpha elements
            alpha_data = r.get('alpha_data', {})
            for elem, col_names in [
                ('Mg', ['[Mg/Fe]', 'Mg_Fe', 'mg_fe']),
                ('Si', ['[Si/Fe]', 'Si_Fe', 'si_fe']),
                ('Ca', ['[Ca/Fe]', 'Ca_Fe', 'ca_fe']),
                ('Ti', ['[Ti/Fe]', 'Ti_Fe', 'ti_fe']),
            ]:
                x_arr, y_arr = np.array([]), np.array([])
                for col in col_names:
                    if col in alpha_data:
                        x_arr, y_arr = alpha_data[col]
                        break
                curves[f'alpha_{elem}_x'].append(np.asarray(x_arr, dtype=float))
                curves[f'alpha_{elem}_y'].append(np.asarray(y_arr, dtype=float))
        
        # Convert to object arrays to handle varying lengths
        save_dict = {}
        save_dict['model_ids'] = np.array(curves['model_ids'])
        for key in curves:
            if key != 'model_ids':
                save_dict[key] = np.array(curves[key], dtype=object)
        
        npz_path = os.path.join(output_path, f'{prefix}curves.npz')
        np.savez_compressed(npz_path, **save_dict)
        print(f"Saved curve data to {npz_path}")
    
    return csv_path, npz_path


def load_complete_results(
    output_path: str,
    prefix: str = '',
    load_curves: bool = True,
) -> pd.DataFrame:
    """
    Load complete results including curves into a single DataFrame.
    
    Parameters
    ----------
    output_path : str
        Directory containing results
    prefix : str
        Prefix for files
    load_curves : bool
        Whether to load curve data
        
    Returns
    -------
    df : DataFrame
        Results with curve columns:
        - mdf_x, mdf_y: MDF arrays
        - age_x, age_y: age-metallicity arrays
        - alpha_tracks: list of (x, y) for [Mg, Si, Ca, Ti]
    """
    # Load CSV
    csv_path = os.path.join(output_path, f'{prefix}results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Try alternative names
        for alt in ['simulation_results.csv', 'ga_population_samples.csv']:
            alt_path = os.path.join(output_path, alt)
            if os.path.exists(alt_path):
                df = pd.read_csv(alt_path)
                break
        else:
            raise FileNotFoundError(f"No results CSV found in {output_path}")
    
    if not load_curves:
        return df
    
    # Load curves
    npz_path = os.path.join(output_path, f'{prefix}curves.npz')
    if not os.path.exists(npz_path):
        # Try alternative names
        alt_paths = [
            os.path.join(output_path, 'walker_history.npz'),
            os.path.join(output_path, 'curves.npz'),
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                npz_path = alt
                break
        else:
            print(f"Warning: No curve data found, returning parameters only")
            return df
    
    data = np.load(npz_path, allow_pickle=True)
    
    # Handle different storage formats
    if 'model_ids' in data:
        # New format with separate curve arrays
        model_ids = data['model_ids']
        
        # Create mapping from model_id to curve data
        id_to_idx = {mid: i for i, mid in enumerate(model_ids)}
        
        def get_curve(key, model_id):
            if model_id in id_to_idx:
                idx = id_to_idx[model_id]
                return data[key][idx]
            return np.array([])
        
        # Add curve columns
        if 'model_id' in df.columns:
            df['mdf_x'] = df['model_id'].apply(lambda x: get_curve('mdf_x', x))
            df['mdf_y'] = df['model_id'].apply(lambda x: get_curve('mdf_y', x))
            df['age_x'] = df['model_id'].apply(lambda x: get_curve('age_x', x))
            df['age_y'] = df['model_id'].apply(lambda x: get_curve('age_y', x))
            
            # Alpha tracks as list of 4 tuples
            def get_alpha_tracks(model_id):
                return [
                    (get_curve('alpha_Mg_x', model_id), get_curve('alpha_Mg_y', model_id)),
                    (get_curve('alpha_Si_x', model_id), get_curve('alpha_Si_y', model_id)),
                    (get_curve('alpha_Ca_x', model_id), get_curve('alpha_Ca_y', model_id)),
                    (get_curve('alpha_Ti_x', model_id), get_curve('alpha_Ti_y', model_id)),
                ]
            df['alpha_tracks'] = df['model_id'].apply(get_alpha_tracks)
        else:
            print("Warning: CSV does not have model_id column - cannot link to curves.")
            print("  Use {prefix}results.csv (not simulation_results.csv) for curve linkage.")
    
    elif 'mdf_data' in data:
        # Old format with dict-based storage
        mdf_data = data['mdf_data'].item()
        age_data = data.get('age_data', {}).item() if 'age_data' in data else {}
        alpha_data = data.get('alpha_data', {}).item() if 'alpha_data' in data else {}
        
        # Match by index
        n = len(df)
        mdf_x_list = []
        mdf_y_list = []
        age_x_list = []
        age_y_list = []
        alpha_list = []
        
        for i in range(n):
            if i in mdf_data:
                mdf_x_list.append(mdf_data[i][0])
                mdf_y_list.append(mdf_data[i][1])
            else:
                mdf_x_list.append(np.array([]))
                mdf_y_list.append(np.array([]))
            
            if i in age_data:
                age_x_list.append(age_data[i][0])
                age_y_list.append(age_data[i][1])
            else:
                age_x_list.append(np.array([]))
                age_y_list.append(np.array([]))
            
            if i in alpha_data:
                ad = alpha_data[i]
                tracks = []
                for elem in ['[Mg/Fe]', '[Si/Fe]', '[Ca/Fe]', '[Ti/Fe]']:
                    if elem in ad:
                        tracks.append(ad[elem])
                    else:
                        tracks.append((np.array([]), np.array([])))
                alpha_list.append(tracks)
            else:
                alpha_list.append([(np.array([]), np.array([]))] * 4)
        
        df['mdf_x'] = mdf_x_list
        df['mdf_y'] = mdf_y_list
        df['age_x'] = age_x_list
        df['age_y'] = age_y_list
        df['alpha_tracks'] = alpha_list
    
    return df


# =============================================================================
# HDF5 FORMAT (Optional, requires h5py)
# =============================================================================

def save_curves_to_hdf5(
    evaluation_results: List[Dict],
    output_path: str,
    filename: str = 'results.h5',
) -> str:
    """
    Save complete results to HDF5 format.
    
    Requires h5py. Provides efficient storage and fast loading
    for large result sets.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for HDF5 storage. Install with: pip install h5py")
    
    os.makedirs(output_path, exist_ok=True)
    h5_path = os.path.join(output_path, filename)
    
    with h5py.File(h5_path, 'w') as f:
        # Metadata
        f.attrs['n_models'] = len(evaluation_results)
        
        # Parameters group
        params = f.create_group('parameters')
        
        # Extract all parameter arrays
        n = len(evaluation_results)
        param_names = [
            'comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx',
            'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2',
            'sfe', 'delta_sfe', 'imf_upper', 'mgal', 'nb'
        ]
        
        param_arrays = {name: np.zeros(n) for name in param_names}
        fitness = np.zeros(n)
        
        for i, r in enumerate(evaluation_results):
            if 'individual' in r:
                ind = r['individual']
                for j, name in enumerate(param_names):
                    param_arrays[name][i] = ind[j]
            fitness[i] = r.get('fitness', np.inf)
        
        for name, arr in param_arrays.items():
            params.create_dataset(name, data=arr)
        params.create_dataset('fitness', data=fitness)
        
        # Curves group - using variable length datasets
        curves = f.create_group('curves')
        
        # MDF
        mdf_grp = curves.create_group('mdf')
        for i, r in enumerate(evaluation_results):
            if 'mdf_x' in r and 'mdf_y' in r:
                g = mdf_grp.create_group(str(i))
                g.create_dataset('x', data=np.asarray(r['mdf_x']))
                g.create_dataset('y', data=np.asarray(r['mdf_y']))
        
        # Age-metallicity
        age_grp = curves.create_group('age')
        for i, r in enumerate(evaluation_results):
            if 'age_x' in r and 'age_y' in r:
                g = age_grp.create_group(str(i))
                g.create_dataset('x', data=np.asarray(r['age_x']))
                g.create_dataset('y', data=np.asarray(r['age_y']))
        
        # Alpha elements
        alpha_grp = curves.create_group('alpha')
        for elem in ['Mg', 'Si', 'Ca', 'Ti']:
            elem_grp = alpha_grp.create_group(elem)
            for i, r in enumerate(evaluation_results):
                alpha_data = r.get('alpha_data', {})
                for col_name in [f'[{elem}/Fe]', f'{elem}_Fe']:
                    if col_name in alpha_data:
                        x, y = alpha_data[col_name]
                        g = elem_grp.create_group(str(i))
                        g.create_dataset('x', data=np.asarray(x))
                        g.create_dataset('y', data=np.asarray(y))
                        break
    
    print(f"Saved complete results to {h5_path}")
    return h5_path


def load_curves_from_hdf5(h5_path: str) -> pd.DataFrame:
    """Load complete results from HDF5 format."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required. Install with: pip install h5py")
    
    with h5py.File(h5_path, 'r') as f:
        n = f.attrs['n_models']
        
        # Load parameters
        params = f['parameters']
        param_names = list(params.keys())
        
        data = {name: params[name][:] for name in param_names}
        df = pd.DataFrame(data)
        
        # Load curves
        curves = f['curves']
        
        # MDF
        mdf_x_list = []
        mdf_y_list = []
        for i in range(n):
            if str(i) in curves['mdf']:
                g = curves['mdf'][str(i)]
                mdf_x_list.append(g['x'][:])
                mdf_y_list.append(g['y'][:])
            else:
                mdf_x_list.append(np.array([]))
                mdf_y_list.append(np.array([]))
        
        df['mdf_x'] = mdf_x_list
        df['mdf_y'] = mdf_y_list
        
        # Age
        age_x_list = []
        age_y_list = []
        for i in range(n):
            if str(i) in curves['age']:
                g = curves['age'][str(i)]
                age_x_list.append(g['x'][:])
                age_y_list.append(g['y'][:])
            else:
                age_x_list.append(np.array([]))
                age_y_list.append(np.array([]))
        
        df['age_x'] = age_x_list
        df['age_y'] = age_y_list
        
        # Alpha
        alpha_list = []
        for i in range(n):
            tracks = []
            for elem in ['Mg', 'Si', 'Ca', 'Ti']:
                if str(i) in curves['alpha'][elem]:
                    g = curves['alpha'][elem][str(i)]
                    tracks.append((g['x'][:], g['y'][:]))
                else:
                    tracks.append((np.array([]), np.array([])))
            alpha_list.append(tracks)
        
        df['alpha_tracks'] = alpha_list
    
    return df


# =============================================================================
# CONVENIENCE LOADER CLASS
# =============================================================================

class ResultsLoader:
    """
    Convenience class for loading and processing GA results.
    
    Handles both new (CSV+NPZ) and legacy (walker_history.npz) formats.
    
    Example
    -------
    >>> loader = ResultsLoader('/path/to/results')
    >>> df = loader.load()
    >>> df_with_weights = loader.add_posterior_weights(df)
    """
    
    def __init__(self, results_dir: str):
        """
        Initialize loader.
        
        Parameters
        ----------
        results_dir : str
            Directory containing results files
        """
        self.results_dir = Path(results_dir)
        self._detect_format()
    
    def _detect_format(self):
        """Detect which format the results are in."""
        self.csv_path = None
        self.npz_path = None
        self.h5_path = None
        
        # Check for different file types
        for f in self.results_dir.glob('*.csv'):
            if 'results' in f.name or 'simulation' in f.name:
                self.csv_path = f
                break
        
        for f in self.results_dir.glob('*.npz'):
            if 'curves' in f.name or 'walker' in f.name:
                self.npz_path = f
                break
        
        for f in self.results_dir.glob('*.h5'):
            self.h5_path = f
            break
    
    def load(self, include_curves: bool = True) -> pd.DataFrame:
        """
        Load results into DataFrame.
        
        Parameters
        ----------
        include_curves : bool
            Whether to load curve data
            
        Returns
        -------
        df : DataFrame
            Results with optional curve columns
        """
        if self.h5_path:
            return load_curves_from_hdf5(str(self.h5_path))
        
        return load_complete_results(
            str(self.results_dir),
            load_curves=include_curves,
        )
    
    def add_posterior_weights(
        self,
        df: pd.DataFrame,
        fitness_col: str = 'fitness',
        temperature: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Add posterior weights to DataFrame.
        
        Parameters
        ----------
        df : DataFrame
            Results dataframe
        fitness_col : str
            Column containing fitness/loss values
        temperature : float, optional
            Temperature for weighting (auto-determined if None)
            
        Returns
        -------
        df : DataFrame
            DataFrame with 'posterior_w' column added
        """
        loss = df[fitness_col].values
        
        # Remove invalid
        valid = np.isfinite(loss)
        
        # Auto-determine temperature
        if temperature is None:
            loss_valid = loss[valid]
            median = np.median(loss_valid)
            mad = np.median(np.abs(loss_valid - median))
            temperature = max(mad, 0.01)
        
        # Compute weights
        loss_shifted = loss - np.nanmin(loss)
        log_weights = np.where(valid, -loss_shifted / temperature, -np.inf)
        log_weights -= np.max(log_weights[valid])
        weights = np.exp(log_weights)
        weights /= np.sum(weights)
        
        df = df.copy()
        df['posterior_w'] = weights
        
        # Compute ESS
        ess = 1.0 / (np.sum(weights**2) + 1e-12)
        print(f"Posterior weights: ESS = {ess:.1f} / {len(df)} models")
        
        return df
    
    def get_best_model(self, df: pd.DataFrame, fitness_col: str = 'fitness') -> pd.Series:
        """Get row with best (lowest) fitness."""
        return df.loc[df[fitness_col].idxmin()]
    
    def get_observational_data(self) -> Dict[str, Any]:
        """
        Try to load observational data from common locations.
        
        Returns dict with keys: feh, mdf, age_joyce, age_bensby, 
        mg_fe, si_fe, ca_fe, ti_fe
        """
        obs = {}
        
        # Try various locations
        search_paths = [
            self.results_dir,
            self.results_dir / 'data',
            self.results_dir.parent / 'data',
            Path('data'),
        ]
        
        for path in search_paths:
            # MDF
            for mdf_name in ['mdf_APOGEE.txt', 'MDF.txt', 'observed_mdf.txt']:
                mdf_file = path / mdf_name
                if mdf_file.exists():
                    data = np.loadtxt(mdf_file, usecols=(0, 1))
                    obs['feh'] = data[:, 0]
                    obs['mdf'] = data[:, 1]
                    break
            
            # Age data
            for age_name in ['Bensby_ages.tsv', 'ages.tsv', 'observed_ages.tsv']:
                age_file = path / age_name
                if age_file.exists():
                    try:
                        age_df = pd.read_csv(age_file, sep='\t')
                        if 'feh' not in obs and '[Fe/H]' in age_df.columns:
                            obs['feh'] = age_df['[Fe/H]'].values
                        if 'Joyce_age' in age_df.columns:
                            obs['age_joyce'] = age_df['Joyce_age'].values
                        if 'Bensby' in age_df.columns:
                            obs['age_bensby'] = age_df['Bensby'].values
                        for elem, col in [('mg_fe', '[Mg/Fe]'), ('si_fe', '[Si/Fe]'),
                                         ('ca_fe', '[Ca/Fe]'), ('ti_fe', '[Ti/Fe]')]:
                            if col in age_df.columns:
                                obs[elem] = age_df[col].values
                    except:
                        pass
                    break
        
        return obs
