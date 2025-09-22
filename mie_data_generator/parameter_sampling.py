"""Parameter sampling for Mie scattering calculations."""

import numpy as np
from .config import get_config


def sample_parameters(n_samples, config=None):
    """
    Sample parameters for Mie scattering calculations.
    
    Args:
        n_samples: Number of parameter sets to generate
        config: Configuration dictionary (uses defaults if None)
        
    Returns:
        dict with keys: 'size_param', 'n_real', 'n_imag', 'wavelength', 'radius'
    """
    if config is None:
        config = get_config()
    
    # Set random seed for reproducibility
    np.random.seed(config['random_seed'])
    
    # Log-uniform sampling for size parameter
    size_param_min, size_param_max = config['size_param_range']
    log_size_min = np.log10(size_param_min)
    log_size_max = np.log10(size_param_max)
    size_params = 10 ** np.random.uniform(log_size_min, log_size_max, n_samples)
    
    # Uniform sampling for real refractive index
    n_real_min, n_real_max = config['n_real_range']
    n_reals = np.random.uniform(n_real_min, n_real_max, n_samples)
    
    # Log-uniform sampling for imaginary refractive index
    n_imag_min, n_imag_max = config['n_imag_range']
    log_nimag_min = np.log10(n_imag_min)
    log_nimag_max = np.log10(n_imag_max)
    n_imags = 10 ** np.random.uniform(log_nimag_min, log_nimag_max, n_samples)
    
    # Fixed wavelength
    wavelengths = np.full(n_samples, config['wavelength'])
    
    # Calculate radius from size parameter: x = 2πr/λ → r = xλ/(2π)
    radii = size_params * wavelengths / (2 * np.pi)
    
    return {
        'size_param': size_params,
        'n_real': n_reals,
        'n_imag': n_imags,
        'wavelength': wavelengths,
        'radius': radii
    }


def validate_parameters(params):
    """Validate that parameters are in expected ranges."""
    config = get_config()
    
    # Check ranges
    assert np.all(params['size_param'] >= config['size_param_range'][0])
    assert np.all(params['size_param'] <= config['size_param_range'][1])
    assert np.all(params['n_real'] >= config['n_real_range'][0])
    assert np.all(params['n_real'] <= config['n_real_range'][1])
    assert np.all(params['n_imag'] >= config['n_imag_range'][0])
    assert np.all(params['n_imag'] <= config['n_imag_range'][1])
    
    # Check for NaN/inf
    for key, values in params.items():
        assert np.all(np.isfinite(values)), f"Non-finite values in {key}"
    
    print(f"✓ Parameter validation passed for {len(params['size_param'])} samples")


if __name__ == "__main__":
    # Test parameter sampling
    params = sample_parameters(1000)
    validate_parameters(params)
    
    print("Size parameter range:", params['size_param'].min(), "to", params['size_param'].max())
    print("n_real range:", params['n_real'].min(), "to", params['n_real'].max()) 
    print("n_imag range:", params['n_imag'].min(), "to", params['n_imag'].max())
    print("Radius range:", params['radius'].min(), "to", params['radius'].max(), "μm")