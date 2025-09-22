"""Configuration settings for Mie data generation pipeline."""

import numpy as np

DEFAULT_CONFIG = {
    'n_samples': 100_000,
    'batch_size': 1000,
    'n_workers': 8,  # CPU cores for parallel processing
    'size_param_range': (0.1, 100.0),
    'n_real_range': (1.3, 1.8),
    'n_imag_range': (1e-6, 1e-1),
    'wavelength': 0.532,  # μm (fixed)
    'n_angles': 181,  # 0° to 180° in 1° steps
    'output_format': 'hdf5',  # or 'npz'
    'compress': True,
    'log_s11': True,  # Store log(S11) for NN training
    'random_seed': 42,  # For reproducible parameter sampling
}

def get_config(**overrides):
    """Get configuration with optional overrides."""
    config = DEFAULT_CONFIG.copy()
    config.update(overrides)
    return config