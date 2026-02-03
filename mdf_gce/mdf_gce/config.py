"""
Configuration parsing for MDF_GCE_SMC_DEMC.

This module handles reading and validating the parameter card (inlist) file
that configures GA runs.
"""

import ast
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .constants import DEFAULT_PCARD_FILE, DEFAULT_OUTPUT_PATH


# =============================================================================
# INLIST PARSING
# =============================================================================

def parse_inlist(file_path: str) -> Dict[str, Any]:
    """
    Parse an inlist/pcard file and return a dictionary of parameters.
    
    The file format is YAML-like with key: value pairs.
    Lines starting with # are comments.
    
    Parameters
    ----------
    file_path : str
        Path to the parameter card file
        
    Returns
    -------
    dict
        Dictionary of parsed parameters
        
    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    ValueError
        If a line cannot be parsed
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parameter card not found: {file_path}")
    
    params = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse key: value
            if ':' not in line:
                continue  # Skip malformed lines
            
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Handle boolean values
            lowered = value.lower().strip("'\"")
            if lowered in {'true', 'false'}:
                parsed_value = lowered == 'true'
            else:
                # Try to parse as Python literal
                try:
                    parsed_value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    # Keep as string
                    parsed_value = value.strip("'\"")
            
            params[key] = parsed_value
    
    return params


def load_config(
    pcard_path: str = DEFAULT_PCARD_FILE,
    copy_to_output: bool = True,
) -> Dict[str, Any]:
    """
    Load configuration from pcard file and optionally copy to output directory.
    
    Parameters
    ----------
    pcard_path : str
        Path to parameter card file
    copy_to_output : bool
        If True, copy pcard to output directory
        
    Returns
    -------
    dict
        Parsed configuration
    """
    params = parse_inlist(pcard_path)
    
    # Ensure output path exists
    output_path = params.get('output_path', DEFAULT_OUTPUT_PATH)
    os.makedirs(output_path, exist_ok=True)
    
    # Copy pcard to output
    if copy_to_output:
        dest_pcard = os.path.join(output_path, 'bulge_pcard.txt')
        src_pcard = os.path.abspath(pcard_path)
        dst_pcard = os.path.abspath(dest_pcard)
        if src_pcard != dst_pcard:
            shutil.copy2(src_pcard, dest_pcard)
    
    return params


# =============================================================================
# PARAMETER RANGE EXTRACTION
# =============================================================================

def parse_pcard_ranges(pcard_path: str) -> Dict[str, Tuple[float, float]]:
    """
    Extract parameter ranges from pcard file.
    
    Looks for entries like:
        sigma_2_list: [0.1, 5.0]
        tmax_1_list: [0.005, 0.1]
    
    Parameters
    ----------
    pcard_path : str
        Path to parameter card file
        
    Returns
    -------
    dict
        Mapping from parameter name to (low, high) bounds
    """
    # Mapping from pcard keys to standard column names
    key_to_param = {
        'sigma_2_list': 'sigma_2',
        'tmax_1_list': 't_1',
        'tmax_2_list': 't_2',
        'infall_timescale_1_list': 'infall_1',
        'infall_timescale_2_list': 'infall_2',
        'sfe_array': 'sfe',
        'delta_sfe_array': 'delta_sfe',
        'imf_upper_limits': 'imf_upper',
        'mgal_values': 'mgal',
        'nb_array': 'nb',
    }
    
    ranges = {}
    
    if not os.path.exists(pcard_path):
        return ranges
    
    with open(pcard_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for pcard_key, param_name in key_to_param.items():
        # Match pattern: key: [val1, val2]
        pattern = rf'^\s*{re.escape(pcard_key)}\s*:\s*\[([^\]]+)\]'
        match = re.search(pattern, content, flags=re.MULTILINE)
        
        if not match:
            continue
        
        try:
            values = [float(x.strip()) for x in match.group(1).split(',')]
            if len(values) == 2:
                lo, hi = float(values[0]), float(values[1])
                if lo < hi:
                    ranges[param_name] = (lo, hi)
        except (ValueError, IndexError):
            continue
    
    return ranges


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class GAConfig:
    """
    Configuration container for GA runs.
    
    Provides structured access to all configuration parameters with defaults
    and validation.
    """
    # Output configuration
    output_path: str = DEFAULT_OUTPUT_PATH
    
    # Observational data
    obs_file: str = ""
    iniab_header: str = ""
    sn1a_header: str = ""
    obs_age_data_target: str = "joyce"
    
    # Parameter ranges (continuous)
    sigma_2_list: List[float] = field(default_factory=lambda: [0.1, 5.0])
    tmax_1_list: List[float] = field(default_factory=lambda: [0.005, 0.1])
    tmax_2_list: List[float] = field(default_factory=lambda: [0.1, 10.0])
    infall_timescale_1_list: List[float] = field(default_factory=lambda: [0.001, 0.1])
    infall_timescale_2_list: List[float] = field(default_factory=lambda: [0.1, 10.0])
    sfe_array: List[float] = field(default_factory=lambda: [1.0, 20.0])
    delta_sfe_array: List[float] = field(default_factory=lambda: [0.01, 0.85])
    imf_upper_limits: List[float] = field(default_factory=lambda: [60, 130])
    mgal_values: List[float] = field(default_factory=lambda: [1e9, 1e11])
    nb_array: List[float] = field(default_factory=lambda: [0.5e-3, 1.5e-3])
    
    # Parameter arrays (categorical)
    comp_array: List[str] = field(default_factory=list)
    imf_array: List[str] = field(default_factory=lambda: ['salpeter', 'chabrier', 'kroupa'])
    sn1a_assumptions: List[str] = field(default_factory=list)
    stellar_yield_assumptions: List[str] = field(default_factory=list)
    sn1a_rates: List[str] = field(default_factory=lambda: ['power_law'])
    
    # GA configuration
    population_size: int = 96
    num_generations: int = 100
    mutation_probability: float = 0.1
    crossover_probability: float = 0.7
    gaussian_sigma_scale: float = 0.02
    crossover_noise_fraction: float = 0.001
    perturbation_strength: float = 0.2
    physical_constraints_freq: int = 1
    
    # Loss configuration
    mdf_vs_age_weight: float = 1.0
    obs_age_data_loss_metric: str = "rms"
    
    # Simulation configuration
    timesteps: int = 100
    
    @classmethod
    def from_pcard(cls, pcard_path: str) -> "GAConfig":
        """
        Create GAConfig from a pcard file.
        
        Parameters
        ----------
        pcard_path : str
            Path to parameter card file
            
        Returns
        -------
        GAConfig
            Configured instance
        """
        params = parse_inlist(pcard_path)
        
        # Filter to known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in params.items() if k in known_fields}
        
        return cls(**filtered)
    
    def get_param_bounds(self, param_name: str) -> Tuple[float, float]:
        """
        Get bounds for a continuous parameter.
        
        Parameters
        ----------
        param_name : str
            Parameter name (e.g., 'sigma_2', 'sfe')
            
        Returns
        -------
        tuple
            (low, high) bounds
        """
        mapping = {
            'sigma_2': self.sigma_2_list,
            't_1': self.tmax_1_list,
            'tmax_1': self.tmax_1_list,
            't_2': self.tmax_2_list,
            'tmax_2': self.tmax_2_list,
            'infall_1': self.infall_timescale_1_list,
            'infall_timescale_1': self.infall_timescale_1_list,
            'infall_2': self.infall_timescale_2_list,
            'infall_timescale_2': self.infall_timescale_2_list,
            'sfe': self.sfe_array,
            'delta_sfe': self.delta_sfe_array,
            'imf_upper': self.imf_upper_limits,
            'imf_upper_limits': self.imf_upper_limits,
            'mgal': self.mgal_values,
            'mgal_values': self.mgal_values,
            'nb': self.nb_array,
            'nb_array': self.nb_array,
        }
        
        bounds = mapping.get(param_name)
        if bounds is None:
            raise KeyError(f"Unknown parameter: {param_name}")
        
        return (float(bounds[0]), float(bounds[1]))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config(params: Dict[str, Any]) -> List[str]:
    """
    Validate configuration parameters.
    
    Parameters
    ----------
    params : dict
        Configuration dictionary
        
    Returns
    -------
    list of str
        List of warning/error messages (empty if valid)
    """
    warnings = []
    
    # Check required fields
    required = ['obs_file', 'output_path']
    for field in required:
        if field not in params:
            warnings.append(f"Missing required field: {field}")
    
    # Check parameter ranges are valid (low < high)
    range_params = [
        'sigma_2_list', 'tmax_1_list', 'tmax_2_list',
        'infall_timescale_1_list', 'infall_timescale_2_list',
        'sfe_array', 'delta_sfe_array', 'imf_upper_limits',
        'mgal_values', 'nb_array',
    ]
    
    for param in range_params:
        if param in params:
            val = params[param]
            if isinstance(val, (list, tuple)) and len(val) == 2:
                if val[0] >= val[1]:
                    warnings.append(f"Invalid range for {param}: {val} (low >= high)")
    
    # Check GA parameters
    if params.get('population_size', 1) < 10:
        warnings.append("population_size < 10 may cause poor convergence")
    
    if params.get('num_generations', 1) < 10:
        warnings.append("num_generations < 10 may cause poor convergence")
    
    # Check weight is in valid range
    weight = params.get('mdf_vs_age_weight', 1.0)
    if not 0.0 <= weight <= 1.0:
        warnings.append(f"mdf_vs_age_weight should be in [0, 1], got {weight}")
    
    return warnings
