"""Utilities for combining and managing multiple datasets."""

import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
from .data_storage import load_dataset, save_dataset
from .validation import validate_dataset


def combine_datasets(input_files, output_file, validate_result=True):
    """
    Combine multiple HDF5 datasets into a single dataset.
    
    Args:
        input_files: List of input dataset filenames
        output_file: Output filename for combined dataset
        validate_result: Whether to validate the combined dataset
        
    Returns:
        Path to combined dataset file
    """
    print(f"Combining {len(input_files)} datasets into {output_file}")
    
    all_parameters = None
    all_s11_data = []
    all_log_s11_data = []
    angles = None
    combined_config = None
    combined_metadata = {'combined_from': [], 'combination_time': datetime.now().isoformat()}
    
    total_samples = 0
    
    for i, input_file in enumerate(input_files):
        print(f"  Loading dataset {i+1}/{len(input_files)}: {input_file}")
        
        data = load_dataset(input_file)
        n_samples = len(data['parameters']['size_param'])
        total_samples += n_samples
        
        # Initialize or validate angles array
        if angles is None:
            angles = data['angles']
        else:
            if not np.allclose(angles, data['angles']):
                raise ValueError(f"Angle grids don't match between datasets")
        
        # Initialize or combine parameters
        if all_parameters is None:
            all_parameters = {key: [values] for key, values in data['parameters'].items()}
            combined_config = data['config'].copy()
        else:
            for key, values in data['parameters'].items():
                all_parameters[key].append(values)
        
        # Collect S11 data
        all_s11_data.append(data['s11'])
        if data['log_s11'] is not None:
            all_log_s11_data.append(data['log_s11'])
        
        # Combine metadata
        combined_metadata['combined_from'].append({
            'filename': str(input_file),
            'samples': n_samples,
            'generation_time': data['metadata'].get('generation_time', 'unknown')
        })
    
    print(f"  Total samples: {total_samples}")
    
    # Concatenate all arrays
    print("  Concatenating arrays...")
    final_parameters = {}
    for key, value_list in all_parameters.items():
        final_parameters[key] = np.concatenate(value_list)
    
    final_s11 = np.vstack(all_s11_data)
    final_log_s11 = np.vstack(all_log_s11_data) if all_log_s11_data else None
    
    # Update config
    combined_config['n_samples'] = total_samples
    combined_metadata['total_samples'] = total_samples
    combined_metadata['n_input_files'] = len(input_files)
    
    # Save combined dataset
    print("  Saving combined dataset...")
    output_path = Path(output_file)
    
    with h5py.File(output_path, 'w') as f:
        # Store configuration
        config_group = f.create_group('config')
        for key, value in combined_config.items():
            config_group.attrs[key] = value
            
        # Store metadata
        meta_group = f.create_group('metadata')
        for key, value in combined_metadata.items():
            if isinstance(value, list):
                # Convert list to string for HDF5 storage
                meta_group.attrs[key] = str(value)
            else:
                meta_group.attrs[key] = value
        
        # Store arrays
        compression_opts = {'compression': 'gzip', 'compression_opts': 9}
        f.create_dataset('angles', data=angles, **compression_opts)
        
        params_group = f.create_group('parameters')
        for key, values in final_parameters.items():
            params_group.create_dataset(key, data=values, **compression_opts)
        
        f.create_dataset('s11', data=final_s11, **compression_opts)
        if final_log_s11 is not None:
            f.create_dataset('log_s11', data=final_log_s11, **compression_opts)
    
    print(f"✓ Combined dataset saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024**2:.1f} MB")
    
    # Validate if requested
    if validate_result:
        print("  Validating combined dataset...")
        validate_dataset(output_path)
    
    return output_path


def append_to_dataset(base_file, new_file, output_file=None):
    """
    Append a new dataset to an existing base dataset.
    
    Args:
        base_file: Existing dataset file
        new_file: New dataset to append
        output_file: Output filename (if None, overwrites base_file)
        
    Returns:
        Path to updated dataset
    """
    if output_file is None:
        output_file = base_file
        
    return combine_datasets([base_file, new_file], output_file)


def generate_incremental_dataset(base_file, additional_samples, **kwargs):
    """
    Generate additional samples and append to existing dataset.
    
    Args:
        base_file: Existing dataset file (or None for new dataset)
        additional_samples: Number of new samples to generate
        **kwargs: Additional arguments for generate_dataset
        
    Returns:
        Path to updated dataset
    """
    from .generator import generate_dataset
    from .config import get_config
    
    # Generate new samples
    temp_file = f"temp_increment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    
    config = get_config(n_samples=additional_samples, **kwargs)
    
    print(f"Generating {additional_samples} additional samples...")
    new_dataset = generate_dataset(temp_file, config)
    
    # Combine with existing dataset if it exists
    if base_file and Path(base_file).exists():
        print(f"Appending to existing dataset: {base_file}")
        combined_file = append_to_dataset(base_file, new_dataset, base_file)
        
        # Clean up temporary file
        Path(temp_file).unlink()
        
        return combined_file
    else:
        # First dataset - just rename the temp file
        final_file = base_file or "mie_dataset.h5"
        Path(temp_file).rename(final_file)
        return Path(final_file)


def dataset_summary(filename):
    """Print a summary of dataset contents and metadata."""
    data = load_dataset(filename)
    
    print(f"\n📊 Dataset Summary: {filename}")
    print("=" * 50)
    
    # Basic info
    n_samples = len(data['parameters']['size_param'])
    n_angles = len(data['angles'])
    
    print(f"Samples: {n_samples:,}")
    print(f"Angles: {n_angles:,} ({data['angles'].min():.1f}° to {data['angles'].max():.1f}°)")
    
    # Parameter ranges
    params = data['parameters']
    print(f"\nParameter Ranges:")
    print(f"  Size parameter: {params['size_param'].min():.3f} to {params['size_param'].max():.1f}")
    print(f"  n_real: {params['n_real'].min():.3f} to {params['n_real'].max():.3f}")
    print(f"  n_imag: {params['n_imag'].min():.2e} to {params['n_imag'].max():.2e}")
    print(f"  Radius: {params['radius'].min():.3f} to {params['radius'].max():.1f} μm")
    
    # S11 data info
    if data['s11'] is not None:
        s11 = data['s11']
        print(f"\nS11 Data:")
        print(f"  Range: {s11.min():.2e} to {s11.max():.2e}")
        print(f"  Shape: {s11.shape}")
        print(f"  Storage: {'log_s11' if data['log_s11'] is not None else 's11 only'}")
    
    # File info
    file_path = Path(filename)
    file_size_mb = file_path.stat().st_size / 1024**2
    print(f"\nFile Info:")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Efficiency: {file_size_mb/n_samples*1000:.1f} KB/sample")
    
    # Metadata
    if 'metadata' in data and data['metadata']:
        print(f"\nMetadata:")
        for key, value in data['metadata'].items():
            if key not in ['combined_from']:  # Skip verbose keys
                print(f"  {key}: {value}")


def generate_overview_plot(filename, output_file=None):
    """
    Generate overview plot for an existing dataset.
    
    Args:
        filename: Input dataset filename
        output_file: Output plot filename (auto-generated if None)
        
    Returns:
        Path to generated plot file
    """
    from .validation import plot_dataset_overview
    
    input_path = Path(filename)
    if output_file is None:
        output_path = input_path.with_suffix('.png')
    else:
        output_path = Path(output_file)
    
    plot_dataset_overview(input_path, output_path)
    print(f"✓ Overview plot saved to {output_path}")
    
    return output_path


if __name__ == "__main__":
    print("Dataset utilities ready for combining and managing datasets")
    print("\nExample usage:")
    print("  combine_datasets(['day1.h5', 'day2.h5'], 'combined.h5')")
    print("  generate_incremental_dataset('existing.h5', 5000)")
    print("  dataset_summary('my_dataset.h5')")