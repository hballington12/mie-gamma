#!/usr/bin/env python3
"""
Interactive plotting tool for Mie scattering dataset.
Loads dataset chunks and allows user to plot specific aa/bb combinations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Suppress display output
from pathlib import Path
import glob
import sys

def load_dataset_info(dataset_dir="test_dataset"):
    """
    Load all dataset chunks and build index of available parameter combinations.
    
    Returns:
        dict: Available combinations with their chunk locations
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"❌ Dataset directory '{dataset_dir}' not found")
        return None
    
    chunk_files = list(dataset_path.glob("mie_dataset_chunk_*.npz"))
    if not chunk_files:
        print(f"❌ No dataset chunks found in '{dataset_dir}'")
        return None
    
    print(f"🔍 Loading dataset info from {len(chunk_files)} chunks...")
    
    available_combinations = {}
    
    for chunk_file in sorted(chunk_files):
        try:
            data = np.load(chunk_file, allow_pickle=True)
            params = data['parameters'].item()
            
            aa_values = params['aa']
            bb_values = params['bb']
            
            for i, (aa, bb) in enumerate(zip(aa_values, bb_values)):
                key = (round(float(aa), 2), round(float(bb), 2))
                available_combinations[key] = {
                    'chunk_file': chunk_file,
                    'index': i,
                    'aa': float(aa),
                    'bb': float(bb)
                }
                
        except Exception as e:
            print(f"⚠️  Error loading {chunk_file}: {e}")
            continue
    
    print(f"✅ Found {len(available_combinations)} parameter combinations")
    return available_combinations

def get_user_input(available_combinations):
    """Get user input for which combinations to plot."""
    
    # Show available ranges
    aa_values = sorted(set(combo[0] for combo in available_combinations.keys()))
    bb_values = sorted(set(combo[1] for combo in available_combinations.keys()))
    
    print(f"\n📊 Available parameter ranges:")
    print(f"   aa: {aa_values[0]} to {aa_values[-1]} ({len(aa_values)} values)")
    print(f"   bb: {bb_values[0]} to {bb_values[-1]} ({len(bb_values)} values)")
    
    # Ask for multiple combinations
    print(f"\n🎯 Would you like to plot multiple combinations on the same graph? (y/n): ", end="")
    multi_plot = input().strip().lower() in ['y', 'yes']
    
    selected_combinations = []
    
    if multi_plot:
        print(f"\n📝 Enter aa,bb combinations (one per line, empty line to finish):")
        print(f"   Example: 5.0,0.1")
        
        while True:
            user_input = input("   aa,bb: ").strip()
            if not user_input:
                break
                
            try:
                parts = user_input.split(',')
                if len(parts) != 2:
                    print("   ❌ Please enter aa,bb format (e.g., 5.0,0.1)")
                    continue
                    
                aa = float(parts[0].strip())
                bb = float(parts[1].strip())
                key = (round(aa, 2), round(bb, 2))
                
                if key in available_combinations:
                    selected_combinations.append(available_combinations[key])
                    print(f"   ✅ Added aa={aa}, bb={bb}")
                else:
                    print(f"   ❌ Combination aa={aa}, bb={bb} not found in dataset")
                    
            except ValueError:
                print("   ❌ Invalid input. Please enter numbers (e.g., 5.0,0.1)")
    else:
        # Single combination
        while True:
            print(f"\n📝 Enter aa,bb combination to plot:")
            user_input = input("   aa,bb: ").strip()
            
            try:
                parts = user_input.split(',')
                if len(parts) != 2:
                    print("   ❌ Please enter aa,bb format (e.g., 5.0,0.1)")
                    continue
                    
                aa = float(parts[0].strip())
                bb = float(parts[1].strip())
                key = (round(aa, 2), round(bb, 2))
                
                if key in available_combinations:
                    selected_combinations.append(available_combinations[key])
                    break
                else:
                    print(f"   ❌ Combination aa={aa}, bb={bb} not found in dataset")
                    print(f"   💡 Try values like: {list(available_combinations.keys())[:5]}...")
                    
            except ValueError:
                print("   ❌ Invalid input. Please enter numbers (e.g., 5.0,0.1)")
    
    return selected_combinations

def load_scattering_data(combination):
    """Load scattering data for a specific combination."""
    try:
        data = np.load(combination['chunk_file'], allow_pickle=True)
        angles = data['angles']
        scattering_data = data['scattering_data'].item()
        
        idx = combination['index']
        return {
            'angles': angles,
            'F11': scattering_data['F11'][idx],
            'F33': scattering_data['F33'][idx], 
            'F12': scattering_data['F12'][idx],
            'F34': scattering_data['F34'][idx],
            'aa': combination['aa'],
            'bb': combination['bb']
        }
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def create_plot(scattering_datasets, plot_type='F11'):
    """Create and save scattering plot."""
    
    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scattering_datasets)))
    
    for i, data in enumerate(scattering_datasets):
        if data is None:
            continue
            
        angles = data['angles']
        values = data[plot_type]
        aa = data['aa']
        bb = data['bb']
        
        label = f"aa={aa}, bb={bb}"
        plt.semilogy(angles, values, color=colors[i], linewidth=1.5, label=label, alpha=0.8)
    
    plt.xlabel('Scattering Angle (degrees)', fontsize=12)
    plt.ylabel(f'{plot_type} Scattering Element', fontsize=12)
    plt.title(f'{plot_type} Scattering Pattern - Gamma Distribution', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 180)
    
    # Generate filename
    if len(scattering_datasets) == 1:
        data = scattering_datasets[0]
        filename = f"{plot_type}_aa{data['aa']}_bb{data['bb']}.png"
    else:
        filename = f"{plot_type}_multi_combinations.png"
    
    filepath = plots_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plot saved: {filepath}")
    return filepath

def main():
    """Main interactive plotting function."""
    
    print("🐅 Mie Scattering Dataset Plotter")
    print("=" * 40)
    
    # Check for dataset directories
    possible_dirs = ["test_dataset", "mie_dataset", "full_mie_dataset"]
    dataset_dir = None
    
    for dirname in possible_dirs:
        if Path(dirname).exists():
            dataset_dir = dirname
            break
    
    if not dataset_dir:
        print("❌ No dataset directory found. Expected one of:")
        for dirname in possible_dirs:
            print(f"   - {dirname}")
        sys.exit(1)
    
    # Load dataset info
    available_combinations = load_dataset_info(dataset_dir)
    if not available_combinations:
        sys.exit(1)
    
    # Get user input
    selected_combinations = get_user_input(available_combinations)
    
    if not selected_combinations:
        print("❌ No valid combinations selected")
        sys.exit(1)
    
    print(f"\n🔄 Loading scattering data for {len(selected_combinations)} combination(s)...")
    
    # Load scattering data
    scattering_datasets = []
    for combo in selected_combinations:
        data = load_scattering_data(combo)
        if data:
            scattering_datasets.append(data)
            print(f"   ✅ Loaded aa={combo['aa']}, bb={combo['bb']}")
        else:
            print(f"   ❌ Failed to load aa={combo['aa']}, bb={combo['bb']}")
    
    if not scattering_datasets:
        print("❌ No data could be loaded")
        sys.exit(1)
    
    # Ask which scattering element to plot
    print(f"\n🎨 Which scattering element would you like to plot?")
    print(f"   1. F11 (phase function)")
    print(f"   2. F33")
    print(f"   3. F12") 
    print(f"   4. F34")
    
    while True:
        choice = input("   Choice (1-4): ").strip()
        if choice == '1':
            plot_type = 'F11'
            break
        elif choice == '2':
            plot_type = 'F33'
            break
        elif choice == '3':
            plot_type = 'F12'
            break
        elif choice == '4':
            plot_type = 'F34'
            break
        else:
            print("   ❌ Please enter 1, 2, 3, or 4")
    
    # Create and save plot
    print(f"\n🎨 Creating {plot_type} plot...")
    filepath = create_plot(scattering_datasets, plot_type)
    
    print(f"\n✅ Plotting complete!")
    print(f"📁 Plot saved to: {filepath}")

if __name__ == "__main__":
    main()