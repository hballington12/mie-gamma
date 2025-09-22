"""Python wrapper for the Fortran Mie scattering code."""

import subprocess
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path


def run_mie_calculation(wavelength, n_real, n_imag, aa, bb, executable_path="../spher_f_mono"):
    """
    Run a single Mie scattering calculation with gamma distribution.
    
    Args:
        wavelength: Wavelength in μm
        n_real: Real part of refractive index
        n_imag: Imaginary part of refractive index  
        aa: Effective radius in μm
        bb: Effective variance
        executable_path: Path to the compiled Fortran executable
        
    Returns:
        tuple: (angles, s11_values, scattering_params) where angles in degrees, s11_values are linear scale
    """
    # Validate inputs
    if not all(np.isfinite([wavelength, n_real, n_imag, aa, bb])):
        raise ValueError(f"Non-finite input values: λ={wavelength}, n_r={n_real}, n_i={n_imag}, aa={aa}, bb={bb}")
    
    if wavelength <= 0 or aa <= 0 or bb <= 0 or n_real < 1.0 or n_imag < 0:
        raise ValueError(f"Invalid physical parameters: λ={wavelength}, n_r={n_real}, n_i={n_imag}, aa={aa}, bb={bb}")
    # Create temporary directory for this calculation
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        executable_full_path = Path(executable_path).resolve()
        
        # Copy executable to temp directory
        temp_executable = temp_path / "spher_f_mono"
        shutil.copy2(executable_full_path, temp_executable)
        
        # Run calculation with command line arguments
        # Format numbers to avoid scientific notation issues in old Fortran
        wavelength_str = f"{wavelength:.8f}"
        n_real_str = f"{n_real:.8f}"
        
        # For very small imaginary parts, use fixed notation if possible
        if n_imag < 1e-6:
            n_imag_str = f"{n_imag:.10f}"  # Use many decimal places for tiny numbers
        elif n_imag < 1e-3:
            n_imag_str = f"{n_imag:.8f}"
        else:
            n_imag_str = f"{n_imag:.6f}"
            
        aa_str = f"{aa:.8f}"
        bb_str = f"{bb:.8f}"
        
        cmd = [str(temp_executable), wavelength_str, n_real_str, n_imag_str, aa_str, bb_str]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode != 0:
                param_str = f"λ={wavelength_str}, n_r={n_real_str}, n_i={n_imag_str}, aa={aa_str}, bb={bb_str}"
                raise RuntimeError(f"Mie calculation failed with params ({param_str}): {result.stderr}")
                
            # Parse the output file
            output_file = temp_path / "spher.print"
            if not output_file.exists():
                raise FileNotFoundError("Output file spher.print not found")
                
            angles, s11_values, scattering_params = parse_mie_output(output_file)
            return angles, s11_values, scattering_params
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Mie calculation timed out")
        except Exception as e:
            raise RuntimeError(f"Error running Mie calculation: {e}")


def parse_mie_output(output_file):
    """
    Parse the spher.print output file to extract scattering data and parameters.
    
    Args:
        output_file: Path to spher.print file
        
    Returns:
        tuple: (angles, s11_values, scattering_params) where scattering_params is a dict
    """
    angles = []
    s11_values = []
    scattering_params = {}
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    # Parse scattering parameters from the header
    for line in lines:
        line = line.strip()
        
        # Parse <COS> (asymmetry parameter)
        if line.startswith('<COS>'):
            parts = line.split('=')
            if len(parts) >= 2:
                scattering_params['asymmetry'] = float(parts[1].strip().replace('D', 'E'))
        
        # Parse CEXT, CSCA, ALBEDO
        if 'CEXT=' in line and 'CSCA=' in line and 'ALBEDO' in line:
            parts = line.split()
            for part in parts:
                if part.startswith('CEXT='):
                    scattering_params['cext'] = float(part.split('=')[1].replace('D', 'E'))
                elif part.startswith('CSCA='):
                    scattering_params['csca'] = float(part.split('=')[1].replace('D', 'E'))
                elif 'ALBEDO' in part and '=' in part:
                    scattering_params['albedo'] = float(part.split('=')[1].strip().replace('D', 'E'))
    
    # Calculate absorption cross section
    if 'cext' in scattering_params and 'csca' in scattering_params:
        scattering_params['cabs'] = scattering_params['cext'] - scattering_params['csca']
    
    # Parse the scattering matrix data from the end of the file
    # Format: angle F11 F33 F12 F34
    # We need to find where the angle data starts (after expansion coefficients)
    
    in_scattering_data = False
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        parts = line.split()
        if len(parts) >= 5:
            try:
                # Try to parse as: angle F11 F33 F12 F34
                angle = float(parts[0])
                f11 = float(parts[1])
                
                # Check if this looks like angle data (0-180°)
                # The expansion coefficient section has integer indices, not decimal angles
                if 0 <= angle <= 180 and '.' in parts[0]:
                    angles.append(angle)
                    s11_values.append(f11)
                    in_scattering_data = True
                    
            except (ValueError, IndexError):
                continue
    
    if len(angles) == 0:
        raise ValueError("No scattering data found in output file")
    
    return np.array(angles), np.array(s11_values), scattering_params


def calculate_size_parameter(aa, wavelength):
    """Calculate size parameter x = 2πaa/λ based on effective radius."""
    return 2 * np.pi * aa / wavelength


def aa_from_size_parameter(size_param, wavelength):
    """Calculate effective radius from size parameter: aa = xλ/(2π)."""
    return size_param * wavelength / (2 * np.pi)


if __name__ == "__main__":
    # Test the wrapper with a simple calculation
    print("Testing Mie wrapper with gamma distribution...")
    
    # Test parameters
    wavelength = 0.532  # μm
    n_real = 1.3
    n_imag = 0.0
    aa = 5.0  # effective radius μm
    bb = 0.1  # effective variance
    
    try:
        angles, s11, params = run_mie_calculation(wavelength, n_real, n_imag, aa, bb)
        
        print(f"✓ Calculation completed successfully")
        print(f"  Size parameter: {calculate_size_parameter(aa, wavelength):.3f}")
        print(f"  Number of angles: {len(angles)}")
        print(f"  Angle range: {angles.min():.1f}° to {angles.max():.1f}°")
        print(f"  S11 range: {s11.min():.2e} to {s11.max():.2e}")
        print(f"  Forward scattering S11(0°): {s11[0]:.2e}")
        print(f"  Scattering parameters: {params}")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")