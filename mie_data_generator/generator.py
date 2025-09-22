"""Main data generation pipeline for Mie scattering datasets."""

import numpy as np
import multiprocessing as mp
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm

from .config import get_config
from .parameter_sampling import sample_parameters
from .mie_wrapper import run_mie_calculation
from .data_storage import save_dataset
from .validation import validate_dataset, plot_dataset_overview


def generate_single_sample(args):
    """
    Generate a single Mie scattering sample.
    
    Args:
        args: Tuple of (sample_index, wavelength, n_real, n_imag, radius, executable_path)
        
    Returns:
        Tuple of (sample_index, angles, s11_values) or (sample_index, None, error_msg)
    """
    sample_idx, wavelength, n_real, n_imag, radius, executable_path = args
    
    try:
        angles, s11_values = run_mie_calculation(wavelength, n_real, n_imag, radius, executable_path)
        return sample_idx, angles, s11_values
    except Exception as e:
        return sample_idx, None, str(e)


def generate_batch(parameters, batch_indices, executable_path, n_workers=1):
    """
    Generate a batch of Mie scattering calculations.
    
    Args:
        parameters: Dict with parameter arrays
        batch_indices: Indices of samples to process in this batch
        executable_path: Path to Mie executable
        n_workers: Number of parallel workers
        
    Returns:
        Tuple of (angles, s11_batch, failed_indices)
    """
    # Prepare arguments for parallel processing
    args_list = []
    for i in batch_indices:
        args = (
            i,
            parameters['wavelength'][i],
            parameters['n_real'][i], 
            parameters['n_imag'][i],
            parameters['radius'][i],
            executable_path
        )
        args_list.append(args)
    
    # Run calculations in parallel
    if n_workers == 1:
        # Serial processing for debugging
        results = [generate_single_sample(args) for args in args_list]
    else:
        # Parallel processing
        with mp.Pool(n_workers) as pool:
            results = pool.map(generate_single_sample, args_list)
    
    # Process results
    angles = None
    s11_batch = []
    failed_indices = []
    
    for sample_idx, result_angles, result_s11 in results:
        if result_angles is None:
            # Failed calculation
            failed_indices.append(sample_idx)
            print(f"Warning: Sample {sample_idx} failed: {result_s11}")
        else:
            # Successful calculation
            if angles is None:
                angles = result_angles  # First successful sample sets angle grid
            s11_batch.append(result_s11)
    
    if len(s11_batch) == 0:
        raise RuntimeError("All calculations in batch failed")
    
    s11_batch = np.array(s11_batch)
    
    return angles, s11_batch, failed_indices


def generate_dataset(output_file, config=None, executable_path="./spher_f_mono", 
                    validate=True, plot_overview=True):
    """
    Generate a complete Mie scattering dataset.
    
    Args:
        output_file: Output filename for dataset
        config: Configuration dict (uses defaults if None)
        executable_path: Path to Mie scattering executable
        validate: Whether to validate the generated dataset
        plot_overview: Whether to create overview plots
        
    Returns:
        Path to generated dataset file
    """
    if config is None:
        config = get_config()
    
    print(f"Generating Mie scattering dataset:")
    print(f"  Samples: {config['n_samples']:,}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Workers: {config['n_workers']}")
    print(f"  Output: {output_file}")
    
    start_time = time.time()
    
    # Generate parameters
    print("\n1. Sampling parameters...")
    parameters = sample_parameters(config['n_samples'], config)
    print(f"✓ Generated {config['n_samples']:,} parameter sets")
    
    # Initialize storage
    angles = None
    all_s11_data = []
    all_failed_indices = []
    
    # Process in batches
    print(f"\n2. Running Mie calculations...")
    n_batches = int(np.ceil(config['n_samples'] / config['batch_size']))
    
    with tqdm(total=config['n_samples'], desc="Samples") as pbar:
        for batch_idx in range(n_batches):
            start_idx = batch_idx * config['batch_size']
            end_idx = min(start_idx + config['batch_size'], config['n_samples'])
            batch_indices = range(start_idx, end_idx)
            
            try:
                batch_angles, batch_s11, batch_failed = generate_batch(
                    parameters, batch_indices, executable_path, config['n_workers']
                )
                
                if angles is None:
                    angles = batch_angles
                    print(f"  Angle grid: {len(angles)} points ({angles.min():.1f}° to {angles.max():.1f}°)")
                
                all_s11_data.append(batch_s11)
                all_failed_indices.extend(batch_failed)
                
                # Update progress
                successful_in_batch = len(batch_indices) - len(batch_failed)
                pbar.update(successful_in_batch)
                
            except Exception as e:
                print(f"Batch {batch_idx} failed completely: {e}")
                all_failed_indices.extend(batch_indices)
                continue
    
    # Combine all successful calculations
    if len(all_s11_data) == 0:
        raise RuntimeError("No successful calculations generated")
    
    s11_data = np.vstack(all_s11_data)
    n_successful = s11_data.shape[0]
    n_failed = len(all_failed_indices)
    
    print(f"\n3. Processing results...")
    print(f"  Successful: {n_successful:,} samples")
    print(f"  Failed: {n_failed:,} samples")
    print(f"  Success rate: {n_successful/(n_successful+n_failed):.1%}")
    
    # Remove failed samples from parameters
    if n_failed > 0:
        successful_indices = [i for i in range(config['n_samples']) if i not in all_failed_indices]
        for key in parameters:
            parameters[key] = parameters[key][successful_indices]
    
    # Create metadata
    metadata = {
        'generation_time': datetime.now().isoformat(),
        'generator_version': '0.1.0',
        'n_samples_requested': config['n_samples'],
        'n_samples_successful': n_successful,
        'n_samples_failed': n_failed,
        'success_rate': n_successful/(n_successful+n_failed),
        'executable_path': str(executable_path),
        'generation_time_seconds': time.time() - start_time
    }
    
    # Save dataset
    print(f"\n4. Saving dataset...")
    output_path = Path(output_file)
    save_dataset(output_path, parameters, angles, s11_data, config, metadata)
    
    # Validation
    if validate:
        print(f"\n5. Validating dataset...")
        validate_dataset(output_path)
    
    # Create overview plots
    if plot_overview:
        print(f"\n6. Creating overview plots...")
        plot_path = output_path.with_suffix('.png')
        plot_dataset_overview(output_path, plot_path)
    
    total_time = time.time() - start_time
    print(f"\n✓ Dataset generation completed in {total_time:.1f} seconds")
    print(f"  Rate: {n_successful/total_time:.1f} samples/second")
    
    return output_path


def generate_test_dataset(n_samples=100, output_file="test_mie_dataset.h5"):
    """Generate a small test dataset for validation."""
    
    config = get_config(
        n_samples=n_samples,
        batch_size=min(50, n_samples),
        n_workers=2  # Conservative for testing
    )
    
    return generate_dataset(output_file, config, validate=True, plot_overview=True)


if __name__ == "__main__":
    print("Testing dataset generation...")
    
    # Generate small test dataset
    test_file = generate_test_dataset(n_samples=10, output_file="test_tiny.h5")
    print(f"✓ Test dataset generated: {test_file}")
    
    # Cleanup
    test_file.unlink()
    print("✓ Test completed")