# Gamma Distribution Mie Scattering Dataset Generator

## Compilation

Compile the Fortran source code using the makefile:

```bash
make
```

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generate dataset
```bash
python3 generate.py mie_gamma_params_combinations.csv --output-dir mie_dataset --chunk-size 100 --max-workers 4
```

### Plot results
```bash
python3 plot_dataset.py
```

## Files

- `src/` - Fortran source code for Mie scattering calculations
- `Makefile` - Compilation instructions
- `mie_data_generator/` - Python wrapper for Fortran executable
- `mie_gamma_params.toml` - Parameter ranges (aa: 1-100, bb: 0.01-1.0)
- `mie_gamma_params_combinations.csv` - 10,000 parameter combinations 
- `generate.py` - Main dataset generation script
- `plot_dataset.py` - Interactive plotting tool