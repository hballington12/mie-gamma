#!/usr/bin/env python3
"""
Generate comprehensive Mie scattering dataset using paramiter-generated parameter combinations.
This script processes the CSV output from paramiter and runs Mie calculations for all combinations.
"""

import sys
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import subprocess
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import logging

# Add the mie_data_generator to path
sys.path.append('../mie_data_generator')
from mie_wrapper import run_mie_calculation, parse_mie_output


def setup_logging(log_file="dataset_generation.log"):
    """Setup logging configuration."""
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def parse_full_spher_output(output_file):
    """
    Enhanced version of parse_mie_output that extracts ALL scattering elements and metadata.
    
    Returns:
        dict: Complete scattering data including all matrix elements
    """
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    # Initialize data structures
    metadata = {}
    angles = []
    scattering_matrix = {
        'F11': [], 'F33': [], 'F12': [], 'F34': []
    }
    
    # Parse header metadata
    for line in lines:
        line = line.strip()
        
        # Extract wavelength from "Wavelength= VALUE (um)"
        if line.startswith('Wavelength='):
            parts = line.split()
            metadata['wavelength'] = float(parts[1])
        
        # Extract refractive index from "Refractive index= REAL +i* IMAG"
        elif line.startswith('Refractive index='):
            parts = line.split()
            metadata['n_real'] = float(parts[2])
            metadata['n_imag'] = float(parts[4])
        
        # Extract effective radius from "Effective radius (aa)= VALUE (um)"
        elif line.startswith('Effective radius (aa)='):
            parts = line.split()
            metadata['aa'] = float(parts[3])
        
        # Extract effective variance from "Effective variance (bb)= VALUE"
        elif line.startswith('Effective variance (bb)='):
            parts = line.split('=')
            metadata['bb'] = float(parts[1].strip().replace('D', 'E'))
        
        # Extract from compact format: LAM = MRR= MRI= (for compatibility)
        elif 'LAM =' in line and 'MRR=' in line and 'MRI=' in line:
            parts = line.replace('=', ' ').split()
            for i, part in enumerate(parts):
                if part == 'LAM' and i+1 < len(parts):
                    metadata['wavelength'] = float(parts[i+1])
                elif part == 'MRR' and i+1 < len(parts):
                    metadata['n_real'] = float(parts[i+1])
                elif part == 'MRI' and i+1 < len(parts):
                    metadata['n_imag'] = float(parts[i+1])
        
        # Extract size parameter
        if 'SIZE PARAMETER' in line and '=' in line:
            parts = line.split('=')
            if len(parts) >= 2:
                metadata['size_parameter'] = float(parts[1].strip().replace('D', 'E'))
        
        # Extract asymmetry parameter
        if line.startswith('<COS>'):
            parts = line.split('=')
            if len(parts) >= 2:
                metadata['asymmetry'] = float(parts[1].strip().replace('D', 'E'))
        
        # Extract cross sections and albedo
        if 'CEXT=' in line and 'CSCA=' in line and 'ALBEDO' in line:
            parts = line.split()
            for part in parts:
                if part.startswith('CEXT='):
                    metadata['cext'] = float(part.split('=')[1].replace('D', 'E'))
                elif part.startswith('CSCA='):
                    metadata['csca'] = float(part.split('=')[1].replace('D', 'E'))
                elif 'ALBEDO' in part and '=' in part:
                    metadata['albedo'] = float(part.split('=')[1].strip().replace('D', 'E'))
    
    # Calculate derived quantities
    if 'cext' in metadata and 'csca' in metadata:
        metadata['cabs'] = metadata['cext'] - metadata['csca']
    
    # Parse scattering matrix data
    # Format: angle F11 F33 F12 F34
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) >= 5:
            try:
                angle = float(parts[0])
                # Check if this looks like angle data (0-180°)
                if 0 <= angle <= 180 and '.' in parts[0]:
                    angles.append(angle)
                    scattering_matrix['F11'].append(float(parts[1]))
                    scattering_matrix['F33'].append(float(parts[2]))
                    scattering_matrix['F12'].append(float(parts[3]))
                    scattering_matrix['F34'].append(float(parts[4]))
                    
            except (ValueError, IndexError):
                continue
    
    # Convert to numpy arrays
    angles = np.array(angles)
    for key in scattering_matrix:
        scattering_matrix[key] = np.array(scattering_matrix[key])
    
    if len(angles) == 0:
        raise ValueError("No scattering data found in output file")
    
    return {
        'metadata': metadata,
        'angles': angles,
        'scattering_matrix': scattering_matrix
    }


def process_single_calculation(row_data, executable_path="../spher_f_mono"):
    """
    Process a single parameter combination.
    
    Args:
        row_data: Dict with keys 'row', 'wavelength', 'n_real', 'n_imag', 'aa', 'bb'
        executable_path: Path to Mie scattering executable
        
    Returns:
        dict: Complete calculation results or error info
    """
    try:
        # Extract parameters
        row_num = row_data['row']
        wavelength = float(row_data['wavelength'])
        n_real = float(row_data['n_real'])
        n_imag = float(row_data['n_imag'])
        aa = float(row_data['aa'])
        bb = float(row_data['bb'])
        
        # Run calculation with enhanced parsing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            executable_full_path = Path(executable_path).resolve()
            
            # Copy executable
            temp_executable = temp_path / "spher_f_mono"
            shutil.copy2(executable_full_path, temp_executable)
            
            # Format parameters
            cmd = [
                str(temp_executable),
                f"{wavelength:.8f}",
                f"{n_real:.8f}", 
                f"{n_imag:.10f}",
                f"{aa:.8f}",
                f"{bb:.8f}"
            ]
            
            # Run calculation
            result = subprocess.run(
                cmd, cwd=temp_dir, capture_output=True, text=True, timeout=60
            )
            
            if result.returncode != 0:
                return {
                    'row': row_num,
                    'success': False,
                    'error': f"Calculation failed: {result.stderr}"
                }
            
            # Parse output
            output_file = temp_path / "spher.print"
            if not output_file.exists():
                return {
                    'row': row_num,
                    'success': False,
                    'error': "Output file not found"
                }
            
            full_data = parse_full_spher_output(output_file)
            full_data['row'] = row_num
            full_data['success'] = True
            
            return full_data
            
    except Exception as e:
        return {
            'row': row_data.get('row', -1),
            'success': False,
            'error': str(e)
        }


def save_dataset_chunk(results, chunk_num, output_dir):
    """Save a chunk of results in compressed format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Separate successful and failed calculations
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    if successful:
        # Organize data for efficient storage
        chunk_data = {
            'metadata': {
                'chunk_number': chunk_num,
                'n_successful': len(successful),
                'n_failed': len(failed),
                'timestamp': time.time()
            },
            'parameters': {},
            'scattering_data': {},
            'angles': successful[0]['angles']  # Same for all calculations
        }
        
        # Collect parameters and scattering data
        param_keys = ['row', 'wavelength', 'n_real', 'n_imag', 'aa', 'bb', 
                     'size_parameter', 'cext', 'csca', 'cabs', 'albedo', 'asymmetry']
        
        for key in param_keys:
            chunk_data['parameters'][key] = []
        
        matrix_keys = ['F11', 'F33', 'F12', 'F34']
        for key in matrix_keys:
            chunk_data['scattering_data'][key] = []
        
        # Fill data arrays
        for result in successful:
            metadata = result['metadata']
            scattering = result['scattering_matrix']
            
            chunk_data['parameters']['row'].append(result['row'])
            for key in param_keys[1:]:  # Skip 'row'
                chunk_data['parameters'][key].append(metadata.get(key, np.nan))
            
            for key in matrix_keys:
                chunk_data['scattering_data'][key].append(scattering[key])
        
        # Convert to numpy arrays
        for key in chunk_data['parameters']:
            chunk_data['parameters'][key] = np.array(chunk_data['parameters'][key])
        
        for key in chunk_data['scattering_data']:
            chunk_data['scattering_data'][key] = np.array(chunk_data['scattering_data'][key])
        
        # Save as compressed numpy archive
        chunk_file = output_dir / f"mie_dataset_chunk_{chunk_num:04d}.npz"
        np.savez_compressed(chunk_file, **chunk_data)
        
        # Save failed calculations log
        if failed:
            failed_file = output_dir / f"failed_chunk_{chunk_num:04d}.json"
            with open(failed_file, 'w') as f:
                json.dump(failed, f, indent=2)
        
        return len(successful), len(failed)
    
    return 0, len(failed)


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive Mie scattering dataset')
    parser.add_argument('csv_file', help='CSV file with parameter combinations from paramiter')
    parser.add_argument('--output-dir', default='mie_dataset', help='Output directory for dataset chunks')
    parser.add_argument('--chunk-size', type=int, default=100, help='Number of calculations per chunk')
    parser.add_argument('--max-workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--start-row', type=int, default=1, help='Starting row number (1-indexed)')
    parser.add_argument('--max-rows', type=int, help='Maximum number of rows to process')
    parser.add_argument('--executable', default='../spher_f_mono', help='Path to Mie executable')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(Path(args.output_dir) / "generation.log")
    
    logger.info(f"Starting dataset generation from {args.csv_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Chunk size: {args.chunk_size}, Workers: {args.max_workers}")
    
    # Read parameter combinations
    df = pd.read_csv(args.csv_file)
    logger.info(f"Loaded {len(df)} parameter combinations")
    
    # Apply row limits
    start_idx = args.start_row - 1  # Convert to 0-indexed
    end_idx = len(df)
    if args.max_rows:
        end_idx = min(start_idx + args.max_rows, len(df))
    
    df_subset = df.iloc[start_idx:end_idx]
    logger.info(f"Processing rows {args.start_row} to {start_idx + len(df_subset)}")
    
    # Process in chunks
    total_successful = 0
    total_failed = 0
    start_time = time.time()
    
    for chunk_start in range(0, len(df_subset), args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, len(df_subset))
        chunk_df = df_subset.iloc[chunk_start:chunk_end]
        chunk_num = chunk_start // args.chunk_size
        
        logger.info(f"Processing chunk {chunk_num}: rows {chunk_start} to {chunk_end-1}")
        
        # Convert chunk to list of dicts
        chunk_data = chunk_df.to_dict('records')
        
        # Process chunk in parallel
        chunk_results = []
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all calculations
            futures = {
                executor.submit(process_single_calculation, row_data, args.executable): i 
                for i, row_data in enumerate(chunk_data)
            }
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    chunk_results.append(result)
                except Exception as e:
                    logger.error(f"Calculation failed with exception: {e}")
                    chunk_results.append({
                        'row': -1, 'success': False, 'error': str(e)
                    })
        
        # Save chunk results
        n_success, n_failed = save_dataset_chunk(chunk_results, chunk_num, args.output_dir)
        total_successful += n_success
        total_failed += n_failed
        
        logger.info(f"Chunk {chunk_num} completed: {n_success} successful, {n_failed} failed")
        
        # Progress update
        elapsed = time.time() - start_time
        completed_rows = chunk_end
        total_rows = len(df_subset)
        
        if completed_rows > 0:
            rate = completed_rows / elapsed
            eta = (total_rows - completed_rows) / rate if rate > 0 else 0
            logger.info(f"Progress: {completed_rows}/{total_rows} ({100*completed_rows/total_rows:.1f}%), "
                       f"Rate: {rate:.1f} calcs/sec, ETA: {eta/3600:.1f}h")
    
    # Final summary
    elapsed = time.time() - start_time
    logger.info(f"Dataset generation completed!")
    logger.info(f"Total successful: {total_successful}")
    logger.info(f"Total failed: {total_failed}")
    logger.info(f"Total time: {elapsed/3600:.2f} hours")
    logger.info(f"Average rate: {total_successful/elapsed:.2f} successful calcs/sec")


if __name__ == "__main__":
    main()