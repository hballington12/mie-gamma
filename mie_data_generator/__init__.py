"""
Mie Scattering Data Generation Pipeline

Generates training datasets for neural networks by running systematic
Mie scattering calculations with our Fortran implementation.
"""

from .generator import generate_dataset, generate_test_dataset
from .parameter_sampling import sample_parameters
from .mie_wrapper import run_mie_calculation
from .data_storage import save_dataset, load_dataset
from .validation import validate_dataset
from .dataset_utils import combine_datasets, generate_incremental_dataset, dataset_summary, generate_overview_plot

__version__ = "0.1.0"