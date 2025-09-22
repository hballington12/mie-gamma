"""Data storage utilities for Mie scattering datasets."""

import numpy as np
import h5py
from pathlib import Path
from .config import get_config


def save_dataset_hdf5(filename, parameters, angles, s11_data, config, metadata=None):
    """
    Save dataset to HDF5 format with compression.
    
    Args:
        filename: Output filename
        parameters: Dict with parameter arrays (size_param, n_real, n_imag, etc.)
        angles: Array of scattering angles  
        s11_data: Array of S11 values [n_samples, n_angles]
        config: Configuration dict
        metadata: Optional metadata dict
    """
    filename = Path(filename)
    
    with h5py.File(filename, 'w') as f:
        # Store configuration
        config_group = f.create_group('config')
        for key, value in config.items():
            config_group.attrs[key] = value
            
        # Store metadata
        if metadata:
            meta_group = f.create_group('metadata')
            for key, value in metadata.items():
                meta_group.attrs[key] = value
        
        # Store shared arrays
        f.create_dataset('angles', data=angles, compression='gzip', compression_opts=9)
        
        # Store input parameters
        params_group = f.create_group('parameters')
        for key, values in parameters.items():
            params_group.create_dataset(key, data=values, compression='gzip', compression_opts=9)
        
        # Store S11 data (potentially in log space)
        compression_opts = {'compression': 'gzip', 'compression_opts': 9}
        if config.get('log_s11', True):
            # Store in log space for neural network training
            log_s11 = np.log10(s11_data)
            f.create_dataset('log_s11', data=log_s11, **compression_opts)
            f.create_dataset('s11', data=s11_data, **compression_opts)
        else:
            f.create_dataset('s11', data=s11_data, **compression_opts)
    
    print(f"✓ Dataset saved to {filename}")
    print(f"  File size: {filename.stat().st_size / 1024**2:.1f} MB")


def save_dataset_npz(filename, parameters, angles, s11_data, config, metadata=None):
    """
    Save dataset to compressed NPZ format.
    
    Args:
        filename: Output filename
        parameters: Dict with parameter arrays
        angles: Array of scattering angles
        s11_data: Array of S11 values [n_samples, n_angles]
        config: Configuration dict
        metadata: Optional metadata dict
    """
    filename = Path(filename)
    
    # Prepare data for saving
    save_dict = {
        'angles': angles,
        'config': config,
        **parameters
    }
    
    if metadata:
        save_dict['metadata'] = metadata
    
    # Store S11 data
    if config.get('log_s11', True):
        save_dict['log_s11'] = np.log10(s11_data)
        save_dict['s11'] = s11_data
    else:
        save_dict['s11'] = s11_data
    
    np.savez_compressed(filename, **save_dict)
    
    print(f"✓ Dataset saved to {filename}")
    print(f"  File size: {filename.stat().st_size / 1024**2:.1f} MB")


def save_dataset(filename, parameters, angles, s11_data, config=None, metadata=None):
    """
    Save dataset in the configured format.
    
    Args:
        filename: Output filename (extension determines format)
        parameters: Dict with parameter arrays
        angles: Array of scattering angles
        s11_data: Array of S11 values [n_samples, n_angles]
        config: Configuration dict (uses defaults if None)
        metadata: Optional metadata dict
    """
    if config is None:
        config = get_config()
    
    filename = Path(filename)
    
    if filename.suffix.lower() in ['.h5', '.hdf5']:
        save_dataset_hdf5(filename, parameters, angles, s11_data, config, metadata)
    elif filename.suffix.lower() in ['.npz']:
        save_dataset_npz(filename, parameters, angles, s11_data, config, metadata)
    else:
        # Default to HDF5
        if not filename.suffix:
            filename = filename.with_suffix('.h5')
        save_dataset_hdf5(filename, parameters, angles, s11_data, config, metadata)


def load_dataset_hdf5(filename):
    """Load dataset from HDF5 file."""
    filename = Path(filename)
    
    with h5py.File(filename, 'r') as f:
        # Load configuration
        config = dict(f['config'].attrs)
        
        # Load metadata if present
        metadata = dict(f['metadata'].attrs) if 'metadata' in f else {}
        
        # Load shared arrays
        angles = f['angles'][:]
        
        # Load parameters
        parameters = {}
        for key in f['parameters'].keys():
            parameters[key] = f['parameters'][key][:]
        
        # Load S11 data
        if 'log_s11' in f:
            s11_data = f['s11'][:]
            log_s11_data = f['log_s11'][:]
        else:
            s11_data = f['s11'][:]
            log_s11_data = None
    
    return {
        'parameters': parameters,
        'angles': angles,
        's11': s11_data,
        'log_s11': log_s11_data,
        'config': config,
        'metadata': metadata
    }


def load_dataset_npz(filename):
    """Load dataset from NPZ file."""
    filename = Path(filename)
    
    data = np.load(filename, allow_pickle=True)
    
    # Extract components
    config = data['config'].item() if 'config' in data else {}
    metadata = data['metadata'].item() if 'metadata' in data else {}
    angles = data['angles']
    
    # Extract parameters
    param_keys = ['size_param', 'n_real', 'n_imag', 'wavelength', 'radius']
    parameters = {key: data[key] for key in param_keys if key in data}
    
    # Extract S11 data
    s11_data = data['s11'] if 's11' in data else None
    log_s11_data = data['log_s11'] if 'log_s11' in data else None
    
    return {
        'parameters': parameters,
        'angles': angles,
        's11': s11_data,
        'log_s11': log_s11_data,
        'config': config,
        'metadata': metadata
    }


def load_dataset(filename):
    """Load dataset, automatically detecting format."""
    filename = Path(filename)
    
    if filename.suffix.lower() in ['.h5', '.hdf5']:
        return load_dataset_hdf5(filename)
    elif filename.suffix.lower() in ['.npz']:
        return load_dataset_npz(filename)
    else:
        raise ValueError(f"Unsupported file format: {filename.suffix}")


def dataset_info(filename):
    """Print summary information about a dataset."""
    data = load_dataset(filename)
    
    print(f"Dataset: {filename}")
    print(f"Samples: {len(data['parameters']['size_param'])}")
    print(f"Angles: {len(data['angles'])} ({data['angles'].min():.1f}° to {data['angles'].max():.1f}°)")
    
    params = data['parameters']
    print(f"Size parameter: {params['size_param'].min():.3f} to {params['size_param'].max():.1f}")
    print(f"n_real: {params['n_real'].min():.3f} to {params['n_real'].max():.3f}")
    print(f"n_imag: {params['n_imag'].min():.2e} to {params['n_imag'].max():.2e}")
    
    if data['s11'] is not None:
        s11 = data['s11']
        print(f"S11 range: {s11.min():.2e} to {s11.max():.2e}")
    
    print(f"Config: {data['config']}")


if __name__ == "__main__":
    # Test data storage
    from .parameter_sampling import sample_parameters
    
    print("Testing data storage...")
    
    # Generate small test dataset
    n_samples = 10
    params = sample_parameters(n_samples)
    angles = np.linspace(0, 180, 181)
    s11_data = np.random.lognormal(0, 1, (n_samples, len(angles)))  # Fake data
    
    config = get_config(n_samples=n_samples)
    metadata = {'test': True, 'generator_version': '0.1.0'}
    
    # Test HDF5 saving
    save_dataset('test_dataset.h5', params, angles, s11_data, config, metadata)
    
    # Test loading
    loaded = load_dataset('test_dataset.h5')
    print(f"✓ Loaded {len(loaded['parameters']['size_param'])} samples")
    
    # Test info
    dataset_info('test_dataset.h5')
    
    # Cleanup
    Path('test_dataset.h5').unlink()
    print("✓ Data storage test completed")