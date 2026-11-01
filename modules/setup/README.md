# Setup Directory

This directory contains the environment setup script and dependency requirements for the Abstract Constraint Transformer (ACT) framework.

## Files Overview

### Main Setup Script
- **`setup.sh`**: Automated setup script that creates the ACT conda environment
  - Creates the `act-py312` environment
  - Installs all Python dependencies from requirement files
  - Configures Gurobi optimizer

### Python Requirements
- **`main_requirements.txt`**: Dependencies for ACT environment (`act-py312`)
  - PyTorch with CUDA support
  - ONNX tools and runtime
  - Gurobi Python interface
  - NumPy and other scientific computing libraries

## Usage

### Quick Setup
```
source modules/setup/setup.sh
```

## Troubleshooting

### Environment Creation Failures
If conda environment fails to create:
```
conda clean --all
conda update conda
```

### Gurobi License Issues
- Academic users: https://www.gurobi.com/academia/
- Ensure license is activated in each conda environment
- Check license with: `python -c "import gurobipy; print('Gurobi OK')"`

## Environment Specifications

### act-py312 (Python 3.12)
Primary ACT framework with:
- Hybrid Zonotope verification
- Specification refinement BaB
- Full ONNX/PyTorch model support
