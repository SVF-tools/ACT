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
- **Loaders** - `torchvision_loader/`, `vnnlib_loader/`, and `creator_registry.py` for benchmark discovery
- **Specifications** (`specs.py`) - `InputSpec`/`OutputSpec` with `InKind`/`OutKind` enums
- **Wrapper Layers** (`verifiable_model.py`) - PyTorch modules for verification: `InputLayer`, `InputSpecLayer`, `OutputSpecLayer`
- **Model Synthesis** (`model_synthesis.py`) - Advanced model generation and optimization
- **Device Management** (`util/device_manager.py`) - GPU-first CUDA device handling

#### Back-End (`act/back_end/`)
- **Core Engine** (`core.py`) - `Net`, `Layer`, `Bounds`, `Con`, `ConSet` data structures
- **Verification** (`verifier.py`) - Spec-free verification: `verify_once()`, `verify_lp_batched()`, `verify_bab()`, `verify_bab_batched()`
- **Layer Schema** (`layer_schema.py`) - Layer type definitions and validation rules
- **Solvers** (`solver/`) - `GurobiSolver`, `TorchLPSolver`, `DualSolver`, `HybridZSolver`
- **Transfer Functions** - `interval_tf/`, `hybridz_tf/`, `dual_tf/` dirs
- **Branch-and-Bound** (`bab/`) - BaB refinement with counterexample validation

#### Pipeline (`act/pipeline/`)
- **Torch2ACT Converter** (`torch2act.py`) - Automatic PyTorch→ACT Net conversion
- **Testing Framework** - Trace-based fuzzer, correctness validation, regression testing
- **Integration Bridge** (`verification/`) - Front-end integration for real verification

### Key Data Structures
- **Verification Results**: `VerifyStatus.{CERTIFIED, FALSIFIED, UNKNOWN, TIMEOUT, VERIFIER_ERROR, MODEL_INFER_FAILURE}`
- **Specifications**: `InKind.{BOX, LINF_BALL, LIN_POLY}`, `OutKind.{LINEAR_LE, TOP1_ROBUST, MARGIN_ROBUST, RANGE, UNSAFE_LINEAR}`
- **Core ACT Types**: `Layer` (id, kind, params, meta, vars), `Net` (layers, graph)
- **Bounds**: Box constraints with `lb`/`ub` tensors for variable ranges

## Development Workflows

### Environment Setup
```bash
# Core environment only
conda env create -f environment.yml
conda activate act-py312
```

### Testing
- **Pipeline tests**: `python -m act.pipeline --verify vnnlib` for comprehensive validation
- **Verifier validation**: `python -m act.pipeline --verify netfactory --validate-soundness` for end-to-end correctness

## Configuration System

### Verifier Selection
- **Command loading**: CLI parameters defined across multiple `options.py` files
- **Entry points**: `python -m act.{front_end,back_end,pipeline}`
- **Backend routing**: Modular CLI architecture based on verifier selection

## Critical Conventions

### Path Handling
- **ALWAYS use `act/util/path_config.py`**: For any file path operations, use the centralized path utilities:
  - `get_project_root()` - Project root directory
  - `get_pipeline_log_dir()` - Pipeline log directory
  - `get_data_dir()` - Data directory
  - `get_model_dir()` - Model directory
  - Never hardcode paths - always use path_config.py functions
- **Project root**: Always use project root as working directory
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
- **Gurobi**: License required in `modules/gurobi/gurobi.lic` for MILP optimization

### VNNLIB Compatibility
The maintained support matrix is [`vnncomp/README.md`](../vnncomp/README.md).
Update its operator, model, and property rows in the same PR whenever ONNX
conversion, layer/transfer semantics, VNNLIB parsing or synthesis, counterexample
handling, or the competition runner changes. Include parameter/shape limits and
dated verification evidence; distinguish conversion, verification, and witness
export. Do not infer support from catalogue membership or handler registration.
ACT accepts the supported VNNLIB 2.0 tensor-declaration syntax; unsupported
expressions must be reported explicitly rather than silently omitted.

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
