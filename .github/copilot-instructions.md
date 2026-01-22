# Copilot Instructions for Abstract Constraint Transformer (ACT)

## Project Overview
ACT is a unified neural network verification framework with a modern three-tier architecture: **Front-End** (data/model/spec processing), **Back-End** (verification core), and **Pipeline** (testing/integration). The framework supports PyTorch-native verification with automatic Torch→ACT conversion and spec-free verification.

## Architecture Essentials

### Three-Tier Architecture
1. **Front-End** (`act/front_end/`) - User-facing components for data processing
2. **Back-End** (`act/back_end/`) - Core verification engine with Torch-native analysis
3. **Pipeline** (`act/pipeline/`) - Testing framework and Torch→ACT integration

### Core Components

#### Front-End (`act/front_end/`)
- **Loaders** (`loaders/`) - `DatasetLoader`, `ModelLoader`, `SpecLoader` for MNIST/CIFAR/VNNLIB
- **Specifications** (`specs.py`) - `InputSpec`/`OutputSpec` with `InKind`/`OutKind` enums
- **Wrapper Layers** (`verifiable_model.py`) - PyTorch modules for verification: `InputLayer`, `InputSpecLayer`, `OutputSpecLayer`
- **Model Synthesis** (`model_synthesis.py`) - Advanced model generation and optimization
- **Device Management** (`util/device_manager.py`) - GPU-first CUDA device handling
- **Preprocessors** - Image (`preprocessor_image.py`) and text (`preprocessor_text.py`) processing

#### Back-End (`act/back_end/`)
- **Core Engine** (`core.py`) - `Net`, `Layer`, `Bounds`, `Con`, `ConSet` data structures
- **Verification** (`verifier.py`) - Spec-free verification: `verify_once()`, `verify_bab()`
- **Layer Schema** (`layer_schema.py`) - Layer type definitions and validation rules
- **Solvers** (`solver/`) - `GurobiSolver`, `TorchLPSolver` for MILP/LP optimization
- **Transfer Functions** (`transfer_funs/`) - MLP, CNN, RNN, Transformer analysis
- **Branch-and-Bound** (`bab.py`) - BaB refinement with counterexample validation

#### Pipeline (`act/pipeline/`)
- **Torch2ACT Converter** (`torch2act.py`) - Automatic PyTorch→ACT Net conversion
- **Testing Framework** - Mock generation, correctness validation, regression testing
- **Integration Bridge** (`integration.py`) - Front-end integration for real verification
- **Configuration** (`config.py`) - YAML-based test scenario management

### Key Data Structures
- **Verification Results**: `VerifyStatus.{CERTIFIED, FALSIFIED, UNKNOWN, TIMEOUT, VERIFIER_ERROR, MODEL_INFER_FAILURE}`
- **Specifications**: `InKind.{BOX, L_INF, LIN_POLY}`, `OutKind.{SAFETY, ASSERT}`
- **Core ACT Types**: `Layer` (id, kind, params, meta, vars), `Net` (layers, graph)
- **Bounds**: Box constraints with `lb`/`ub` tensors for variable ranges

## Development Workflows

### Environment Setup
```bash
cd setup/
source setup.sh main  # Creates conda env 'act-main'
conda activate act-main
```

### Running Verification
```bash
python act/wrapper_exts/ext_runner.py \
  --model_path act/wrapper_exts/models/vnnmodels/MNIST/small_relu_mnist_cnn_model_1.onnx \
  --dataset mnist --spec_type local_lp \
  --start 0 --end 1 --epsilon 0.03 --norm inf \
  --mean 0.1307 --std 0.3081
```

### Testing
- **Pipeline tests**: `python act/pipeline/run_tests.py` for comprehensive validation
- **Integration tests**: Built into the pipeline framework for end-to-end validation

## Configuration System

### Dataset Defaults
Config files in `modules/configs/` provide verifier-specific defaults:
```ini
[MNIST] # eran_defaults.ini
mean = [0.1307]
std = [0.3081]
spec_type = "local_lp"
```

### Verifier Selection
- **Command loading**: CLI parameters defined across multiple `options.py` files
- **Backend routing**: Factory pattern in `main.py` based on verifier selection
- **External integrations**: αβ-CROWN and ERAN wrappers in `wrapper_exts/`

## Critical Conventions

### Path Handling
- **ALWAYS use `act/util/path_config.py`**: For any file path operations, use the centralized path utilities:
  - `get_project_root()` - Project root directory
  - `get_pipeline_log_dir()` - Pipeline log directory
  - `get_data_dir()` - Data directory
  - `get_model_dir()` - Model directory
  - Never hardcode paths - always use path_config.py functions
- **Project root**: Always use project root as working directory
- **Model paths**: Relative to project root (`act/wrapper_exts/models/vnnmodels/...`)
- **Import structure**: Hierarchical imports following `act/front_end`, `act/back_end`, `act/pipeline`

### Device and Dtype Management
- **ALWAYS use `act/util/device_manager.py`**: For device and dtype operations:
  - `DeviceManager.get_device()` - Get CUDA device if available, else CPU
  - `DeviceManager.get_dtype()` - Get default dtype (float32)
  - `DeviceManager.to_device(tensor)` - Move tensor to managed device
  - Never hardcode `torch.device('cuda')` or `torch.float32` - use DeviceManager

### Memory Patterns
- **Auto-cleanup**: Torch cache clearing in memory-intensive operations
- **Progress tracking**: Memory monitoring throughout verification pipeline
- **Batch processing**: Single-sample verification with result aggregation

### Error Handling
- **Graceful degradation**: Unknown results instead of crashes
- **Verification stats**: Comprehensive tracking in `clean_prediction_stats`
- **Timeout support**: Built into BaB refinement with configurable limits

## Integration Points

### External Tools
- **ERAN**: Subprocess execution with parameter translation
- **αβ-CROWN**: YAML config generation and temp file management  
- **Gurobi**: License required in `modules/gurobi/gurobi.lic` for MILP optimization

### VNNLIB Compatibility
Full SMT-LIB format support with:
- Variable extraction (`X_i`, `Y_j`, `X_hat_k`)
- Constraint parsing (linear combinations, bounds)
- Local vs. global property detection

## Python Coding Standards

When writing code for this project, follow these Python best practices:

### Code Quality
- Use clear, descriptive variable and function names
- Always add type hints and docstrings for functions/classes
- Use dataclasses for simple data containers
- Apply abstract base classes (ABC) for extensible interfaces
- Use factory patterns for object creation when appropriate
- Handle errors with try/except and custom exceptions
- Good practices for raising ValueError, TypeError, etc.
- Manage resources with context managers (`with` statements)
- Use logging (not print) for diagnostics
- Write modular, testable code (single responsibility principle)
- Avoid global state and side effects
- Prefer list/dict comprehensions over loops when clear
- Use f-strings for formatting
- Follow PEP8 style and PEP257 docstrings
- Add comments for non-obvious logic
- If you have modifications, make sure to remove legacy code and also remove backward compatibility to make cleaner code

### Testing Requirements
- Focus on integration tests for critical verification workflows
- Mock external verifier calls (ERAN, αβ-CROWN) when needed
- Test both typical and edge cases, including error handling
- Keep test logic isolated and independent
- Use the pipeline testing framework for comprehensive validation

### Debugging Practices
- Use structured logging with appropriate levels
- Add assertions to check invariants
- Write clear, actionable error messages
- Profile memory usage in verification operations
- Add memory cleanup in long-running verification tasks

When implementing new features, follow the BaseVerifier plugin pattern and ensure compatibility with the unified CLI interface.