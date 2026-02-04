"""
Data I/O module for MDF_GCE_SMC_DEMC.

Handles saving and loading of GA results including all curve data
needed for posterior analysis and plotting.
"""

from .results_io import (
    save_complete_results,
    load_complete_results,
    save_curves_to_hdf5,
    load_curves_from_hdf5,
    ResultsLoader,
)

__all__ = [
    'save_complete_results',
    'load_complete_results',
    'save_curves_to_hdf5',
    'load_curves_from_hdf5',
    'ResultsLoader',
]
