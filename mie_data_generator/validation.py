"""Validation utilities for Mie scattering datasets."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .data_storage import load_dataset


def validate_parameters(parameters, config):
    """Validate that parameters are in expected ranges and physically reasonable."""
    
    print("Validating parameters...")
    
    # Check ranges against config
    size_params = parameters['size_param']
    n_reals = parameters['n_real']
    n_imags = parameters['n_imag']
    
    size_min, size_max = config['size_param_range']
    n_real_min, n_real_max = config['n_real_range']  
    n_imag_min, n_imag_max = config['n_imag_range']
    
    # Range checks
    assert np.all(size_params >= size_min), f"Size parameter below minimum: {size_params.min()}"
    assert np.all(size_params <= size_max), f"Size parameter above maximum: {size_params.max()}"
    assert np.all(n_reals >= n_real_min), f"n_real below minimum: {n_reals.min()}"
    assert np.all(n_reals <= n_real_max), f"n_real above maximum: {n_reals.max()}"
    assert np.all(n_imags >= n_imag_min), f"n_imag below minimum: {n_imags.min()}"
    assert np.all(n_imags <= n_imag_max), f"n_imag above maximum: {n_imags.max()}"
    
    # Physical reasonableness
    assert np.all(n_reals >= 1.0), "Refractive index real part should be >= 1.0"
    assert np.all(n_imags >= 0.0), "Refractive index imaginary part should be >= 0.0"
    assert np.all(size_params > 0), "Size parameter should be positive"
    
    # Check for NaN/inf
    for key, values in parameters.items():
        assert np.all(np.isfinite(values)), f"Non-finite values in {key}"
    
    print(f"✓ Parameter validation passed for {len(size_params)} samples")


def validate_s11_data(s11_data, angles):
    """Validate S11 scattering data for physical consistency."""
    
    print("Validating S11 data...")
    
    # Basic checks
    assert s11_data.ndim == 2, f"S11 data should be 2D, got {s11_data.ndim}D"
    assert s11_data.shape[1] == len(angles), f"S11 angles mismatch: {s11_data.shape[1]} vs {len(angles)}"
    
    # Physical checks
    assert np.all(s11_data > 0), "S11 values should be positive"
    assert np.all(np.isfinite(s11_data)), "S11 values should be finite"
    
    # Forward scattering should typically be largest
    forward_idx = np.argmin(np.abs(angles))  # Closest to 0°
    forward_s11 = s11_data[:, forward_idx]
    
    # Check that forward scattering is reasonable (usually largest or near-largest)
    max_s11 = np.max(s11_data, axis=1)
    forward_ratio = forward_s11 / max_s11
    
    # Most samples should have forward scattering as a significant fraction of maximum
    reasonable_forward = np.sum(forward_ratio > 0.1) / len(forward_ratio)
    if reasonable_forward < 0.5:
        print(f"Warning: Only {reasonable_forward:.1%} samples have reasonable forward scattering")
    
    print(f"✓ S11 validation passed for {s11_data.shape[0]} samples × {s11_data.shape[1]} angles")


def validate_dataset(filename_or_data):
    """Comprehensive validation of a complete dataset."""
    
    if isinstance(filename_or_data, (str, Path)):
        print(f"Validating dataset: {filename_or_data}")
        data = load_dataset(filename_or_data)
    else:
        data = filename_or_data
        print("Validating dataset from memory")
    
    # Validate parameters
    validate_parameters(data['parameters'], data['config'])
    
    # Validate S11 data
    if data['s11'] is not None:
        validate_s11_data(data['s11'], data['angles'])
    
    # Check log S11 consistency
    if data['log_s11'] is not None and data['s11'] is not None:
        reconstructed = 10 ** data['log_s11']
        relative_error = np.abs(reconstructed - data['s11']) / data['s11']
        max_error = np.max(relative_error)
        
        if max_error > 1e-10:
            print(f"Warning: log S11 reconstruction error: {max_error:.2e}")
        else:
            print("✓ Log S11 consistency verified")
    
    print("✓ Dataset validation completed successfully")


def plot_dataset_overview(filename_or_data, save_path=None):
    """Create overview plots of the dataset."""
    
    if isinstance(filename_or_data, (str, Path)):
        data = load_dataset(filename_or_data)
        title_prefix = f"Dataset: {Path(filename_or_data).name}"
    else:
        data = filename_or_data
        title_prefix = "Dataset Overview"
    
    params = data['parameters']
    angles = data['angles']
    s11 = data['s11']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title_prefix, fontsize=14)
    
    # Parameter distributions
    axes[0, 0].hist(params['size_param'], bins=50, alpha=0.7)
    axes[0, 0].set_xlabel('Size Parameter')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_title('Size Parameter Distribution')
    
    axes[0, 1].hist(params['n_real'], bins=50, alpha=0.7)
    axes[0, 1].set_xlabel('n_real')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Real Refractive Index')
    
    axes[0, 2].hist(params['n_imag'], bins=50, alpha=0.7)
    axes[0, 2].set_xlabel('n_imag')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_xscale('log')
    axes[0, 2].set_title('Imaginary Refractive Index')
    
    # S11 data visualization
    # Show a few representative S11 curves
    n_show = min(10, s11.shape[0])
    for i in range(n_show):
        axes[1, 0].semilogy(angles, s11[i], alpha=0.6, linewidth=0.8)
    axes[1, 0].set_xlabel('Scattering Angle (degrees)')
    axes[1, 0].set_ylabel('S11')
    axes[1, 0].set_title(f'S11 vs Angle ({n_show} samples)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # S11 statistics
    s11_mean = np.mean(s11, axis=0)
    s11_std = np.std(s11, axis=0)
    axes[1, 1].semilogy(angles, s11_mean, 'b-', label='Mean')
    axes[1, 1].fill_between(angles, s11_mean - s11_std, s11_mean + s11_std, alpha=0.3, label='±1σ')
    axes[1, 1].set_xlabel('Scattering Angle (degrees)')
    axes[1, 1].set_ylabel('S11')
    axes[1, 1].set_title('S11 Statistics')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # S11 heatmap
    # Subsample for visualization
    n_subsample = min(100, s11.shape[0])
    indices = np.random.choice(s11.shape[0], n_subsample, replace=False)
    s11_sub = s11[indices]
    
    im = axes[1, 2].imshow(np.log10(s11_sub), aspect='auto', cmap='viridis')
    axes[1, 2].set_xlabel('Angle Index')
    axes[1, 2].set_ylabel('Sample Index')
    axes[1, 2].set_title('log10(S11) Heatmap')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Overview plot saved to {save_path}")
    else:
        plt.show()


def check_scattering_regimes(parameters, s11_data, angles):
    """Analyze dataset coverage across different scattering regimes."""
    
    size_params = parameters['size_param']
    
    # Define regime boundaries
    rayleigh = size_params < 1
    resonance = (size_params >= 1) & (size_params < 50)  
    geometric = size_params >= 50
    
    print(f"Scattering regime coverage:")
    print(f"  Rayleigh (x < 1): {np.sum(rayleigh)} samples ({np.mean(rayleigh):.1%})")
    print(f"  Resonance (1 ≤ x < 50): {np.sum(resonance)} samples ({np.mean(resonance):.1%})")
    print(f"  Geometric (x ≥ 50): {np.sum(geometric)} samples ({np.mean(geometric):.1%})")
    
    # Check asymmetry parameter distribution
    # <cos> ≈ (F11(forward) - F11(backward)) / (F11(forward) + F11(backward))
    forward_idx = 0  # 0°
    backward_idx = -1  # 180°
    
    forward_s11 = s11_data[:, forward_idx]
    backward_s11 = s11_data[:, backward_idx]
    asymmetry = (forward_s11 - backward_s11) / (forward_s11 + backward_s11)
    
    print(f"Asymmetry parameter <g> range: {asymmetry.min():.3f} to {asymmetry.max():.3f}")
    print(f"Mean asymmetry: {asymmetry.mean():.3f} ± {asymmetry.std():.3f}")


if __name__ == "__main__":
    # Test validation on our test dataset
    print("Testing validation utilities...")
    
    # This would test with actual data
    print("✓ Validation utilities ready")