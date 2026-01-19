# ACT Directory

This directory contains the core verification framework for the Abstract Constraint Transformer (ACT) system. It implements a modern three-tier architecture: Front-End (data/model/spec processing), Back-End (verification core), and Pipeline (testing/integration) with PyTorch-native verification capabilities.

## Recent Development Updates

### Unified CLI Architecture (November 2025)
- **Shared CLI Utilities**: Created `act/util/cli_utils.py` with `add_device_args()` and `initialize_from_args()` for consistent device/dtype handling
- **Comprehensive CLI Coverage**: All ACT modules now have dedicated CLIs with unified interface
  - `act.front_end` - Main front-end CLI and loader-specific CLIs
  - `act.pipeline` - Pipeline testing and integration CLI
  - `act.back_end` - Back-end verification and network generation CLI (NEW)
- **Architecture Separation**: Clean separation between ACT native components (use cli_utils) and external verifiers (use local get_parser())
- **Back-End CLI**: New comprehensive CLI supporting network generation, verification, inspection, and serialization testing
- **Deprecated Code Removal**: Removed `act/main.py` in favor of modern modular CLI architecture

### Debugging and Performance Framework (October 2025)
- **PerformanceOptions**: Global debugging flags with `debug_tf`, `validate_constraints`, and configurable logging
- **Transfer Function Logging**: Detailed layer-by-layer analysis logging to `act/pipeline/log/act_debug_tf.log`
- **Constraint Validation**: Targeted validation framework that checks only referenced variables
- **ConSet Improvements**: Added `__iter__` and `__len__` methods for Pythonic container usage
- **Path Management**: Centralized logging to `act/pipeline/log/` using `path_config.py`

### Code Quality Improvements
- **Batch Dimension Fix**: Fixed `affine_bounds()` with proper batch dimension handling and squeeze operations
- **Cleaner Syntax**: Updated all code to use ConSet wrappers (`for con in cons` instead of `.S.values()`)
- **Guarded File I/O**: All debug file operations protected by feature flags
- **Architecture Cleanup**: Removed legacy loaders and raw_processors, consolidated into spec creator system

## Directory Structure

```
act/
├── __init__.py                     # Package initialization
│
├── front_end/                      # Front-End: User-facing data processing
│   ├── cli.py                      # Main front-end CLI with unified device/dtype args
│   ├── torchvision_loader/         # TorchVision integration
│   │   ├── cli.py                  # TorchVision-specific CLI
│   │   ├── create_specs.py         # TorchVisionSpecCreator for dataset-model pairs
│   │   ├── data_model_loader.py    # TorchVision dataset and model loading
│   │   └── data_model_mapping.py   # Dataset-model compatibility mappings
│   ├── vnnlib_loader/              # VNNLIB integration
│   │   ├── cli.py                  # VNNLIB-specific CLI
│   │   └── create_specs.py         # VNNLibSpecCreator for VNNLIB specs
│   ├── specs.py                    # InputSpec/OutputSpec with InKind/OutKind enums
│   ├── spec_creator_base.py        # Base spec creator interface
│   ├── creator_registry.py         # Spec creator registration and discovery
│   ├── verifiable_model.py         # PyTorch verification wrapper modules
│   ├── model_synthesis.py          # Model synthesis using spec creators
│   └── README.md                   # Front-end documentation
│
├── back_end/                       # Back-End: Core verification engine
│   ├── cli.py                      # Back-end CLI (generate, verify, info, test)
│   ├── __main__.py                 # Entry point for python -m act.back_end
│   ├── core.py                     # Net, Layer, Bounds, Con, ConSet data structures
│   ├── verifier.py                 # Spec-free verification: verify_once(), verify_bab()
│   ├── layer_schema.py             # Layer type definitions and validation rules
│   ├── layer_util.py               # Layer validation and creation utilities
│   ├── bab.py                      # Branch-and-bound refinement with CE validation
│   ├── utils.py                    # Backend utilities (affine_bounds, validate_constraints)
│   ├── analyze.py                  # Network analysis and bounds propagation
│   ├── cons_exportor.py            # Constraint export to solvers
│   ├── transfer_functions.py       # Transfer function interface and dispatch
│   ├── net_factory.py              # Generator-driven network factory for examples
│   ├── solver/                     # MILP/LP optimization solvers
│   │   ├── solver_base.py          # Base solver interface
│   │   ├── solver_gurobi.py        # Gurobi MILP solver integration
│   │   └── solver_torch.py         # PyTorch-based LP solver
│   ├── interval_tf/                # Interval-based transfer functions
│   │   ├── interval_tf.py          # Interval TF implementation
│   │   ├── tf_mlp.py               # MLP layer interval analysis
│   │   ├── tf_cnn.py               # CNN layer interval analysis
│   │   ├── tf_rnn.py               # RNN layer interval analysis
│   │   └── tf_transformer.py       # Transformer interval analysis
│   ├── hybridz_tf/                 # HybridZ zonotope transfer functions
│   │   ├── hybridz_tf.py           # HybridZ TF implementation
│   │   ├── tf_mlp.py               # MLP layer zonotope analysis
│   │   ├── tf_cnn.py               # CNN layer zonotope analysis
│   │   ├── tf_rnn.py               # RNN layer zonotope analysis
│   │   └── tf_transformer.py       # Transformer zonotope analysis
│   ├── serialization/              # Net serialization and deserialization
│   │   ├── serialization.py        # NetSerializer with tensor encoding
│   │   └── test_serialization.py   # Serialization correctness tests
│   ├── examples/                   # Example networks and configurations
│   │   ├── config_gen_act_net.yaml # Generator configuration for random nets
│   │   ├── nets/                   # Generated ACT Net JSON files
│   │   └── README.md               # Examples documentation
│   └── README.md                   # Back-end documentation
│
├── pipeline/                       # Pipeline: Testing framework and integration
│   ├── cli.py                      # Pipeline CLI with unified device/dtype args
│   ├── verification/               # Verification utilities submodule
│   │   ├── __init__.py             # Verification module initialization
│   │   ├── torch2act.py            # Automatic PyTorch→ACT Net conversion
│   │   ├── act2torch.py            # ACT Net→PyTorch conversion utilities
│   │   ├── validate_verifier.py    # Verifier correctness validation with concrete tests
│   │   ├── model_factory.py        # ACT Net factory for test networks
│   │   ├── utils.py                # Shared utilities and performance profiling
│   │   └── llm_probe.py            # LLM-based verification probing and analysis
│   ├── fuzzing/                    # Fuzzing utilities
│   ├── configs/                    # Configuration files
│   │   ├── mock_inputs.yaml        # Mock data generation templates
│   │   ├── test_scenarios.yaml     # Complete test scenario definitions
│   │   └── solver_settings.yaml    # Solver configuration options
│   ├── examples/                   # Example usage and quick tests
│   ├── log/                        # Test execution logs (includes act_debug_tf.log)
│   └── README.md                   # Pipeline documentation
│
├── util/                           # Shared Utilities
│   ├── cli_utils.py                # Shared CLI utilities (add_device_args, initialize_from_args)
│   ├── device_manager.py           # GPU-first CUDA device handling
│   ├── path_config.py              # Project path configuration and management
│   ├── options.py                  # PerformanceOptions and debugging configuration
│   ├── stats.py                    # Statistics and performance tracking
│   └── model_inference.py          # Model inference utilities
│
└── wrapper_exts/                   # External verifier integrations
    ├── ext_config.py               # External verifier configuration
    ├── ext_runner.py               # External verifier CLI runner (ERAN, αβ-CROWN)
    ├── abcrown/                    # αβ-CROWN integration module
    │   ├── abcrown_verifier.py     # αβ-CROWN wrapper and interface
    │   └── abcrown_runner.py       # αβ-CROWN backend execution script
    └── eran/                       # ERAN integration module
        └── eran_verifier.py        # ERAN wrapper and interface
```

## Module Documentation

### **Command-Line Interfaces**
All ACT modules now have unified CLI architecture with consistent device/dtype handling:

#### **Front-End CLIs**
- **`front_end/cli.py`**: Main front-end CLI
  - Commands: `--list`, `--create`, `--validate`
  - Unified device/dtype arguments via `cli_utils`
  - Usage: `python -m act.front_end [options]`

- **`front_end/torchvision_loader/cli.py`**: TorchVision-specific CLI
  - Dataset and model listing, spec creation
  - Usage: `python -m act.front_end.torchvision_loader [options]`

- **`front_end/vnnlib_loader/cli.py`**: VNNLIB-specific CLI
  - VNNLIB file parsing and validation
  - Usage: `python -m act.front_end.vnnlib_loader [options]`

#### **Pipeline CLI**
- **`pipeline/cli.py`**: Pipeline testing and integration CLI
  - Commands: Testing, validation, regression, reporting
  - Unified device/dtype arguments via `cli_utils`
  - Usage: `python -m act.pipeline [options]`

#### **Back-End CLI**
- **`back_end/cli.py`**: Comprehensive back-end verification CLI
  - Commands:
    - `--generate`: Generate example networks from YAML config
    - `--list-examples`: List all available example networks
    - `--info`: Display network structure and details
    - `--verify`: Run verification (single-shot or branch-and-bound)
    - `--test-serialization`: Test save/load round-trip
  - Unified device/dtype arguments via `cli_utils`
  - Usage: `python -m act.back_end [options]`
  - Examples:
    ```bash
    # Generate all example networks
    python -m act.back_end --generate --device cpu --dtype float64
    
    # List available networks
    python -m act.back_end --list-examples
    
    # Show network info
    python -m act.back_end --info --network mnist_robust_easy.json --verbose
    
    # Run verification
    python -m act.back_end --verify --network mnist_robust_easy.json --device cpu
    
    # Test serialization
    python -m act.back_end --test-serialization --device cpu
    ```

#### **External Verifier CLI**
- **`wrapper_exts/ext_runner.py`**: External verifier CLI (ERAN, αβ-CROWN)
  - Uses local `get_parser()` for external verifier compatibility
  - Separate from ACT native CLI architecture
  - Usage: `python act/wrapper_exts/ext_runner.py [options]`

#### **Shared CLI Utilities**
- **`util/cli_utils.py`**: Shared CLI infrastructure for all ACT native modules
  - `add_device_args(parser)`: Adds `--device {cpu,cuda,gpu}` and `--dtype {float32,float64}`
  - `initialize_from_args(args)`: Calls `device_manager.initialize_device()` with parsed args
  - Used by: front_end, pipeline, back_end (not external verifiers)

### **`front_end/` - User-Facing Data Processing**
- **Spec Creator System**: Unified framework for creating specifications from various sources
  - **`TorchVisionSpecCreator`**: Creates specs from TorchVision datasets and models
  - **`VNNLibSpecCreator`**: Creates specs from VNNLIB files
  - **`BaseSpecCreator`**: Abstract interface for spec creators

- **`specs.py`**: Specification data structures and enums
  - `InputSpec`/`OutputSpec` classes with `InKind`/`OutKind` type safety
  - Support for BOX, L_INF, LIN_POLY input constraints and SAFETY, ASSERT output properties

- **`verifiable_model.py`**: PyTorch verification wrapper modules
  - `InputLayer`: Declares symbolic input blocks for verification
  - `InputSpecLayer`: Wraps ACT InputSpec as nn.Module for seamless integration
  - `OutputSpecLayer`: Wraps ACT OutputSpec as nn.Module for property specification

- **`model_synthesis.py`**: Model synthesis using spec creators
  - Unified synthesis pipeline using spec creator system
  - Automatic wrapped model generation from dataset-model pairs

- **Preprocessors**: Modular preprocessing pipeline
  - **`preprocessor_image.py`**: Image normalization, augmentation, and format conversion
  - **`preprocessor_text.py`**: Text preprocessing utilities
  - **`preprocessor_base.py`**: Base preprocessor interface and common functionality

### **`back_end/` - Core Verification Engine**
- **`core.py`**: Fundamental ACT data structures
  - `Net`: Network representation with layers and graph connectivity
  - `Layer`: Individual layer with params, metadata, and variable mappings
  - `Bounds`: Box constraints with lb/ub tensors for variable ranges
  - `Con`/`ConSet`: Constraint representation with Pythonic iteration support (`__iter__`, `__len__`)

- **`verifier.py`**: Spec-free verification engine
  - `verify_once()`: Single-shot verification using embedded ACT constraints
  - `verify_bab()`: Branch-and-bound refinement with counterexample validation
  - Integrated constraint validation with targeted variable checking
  - No external input specs required - all constraints extracted from ACT Net

- **`utils.py`**: Backend utility functions
  - `affine_bounds()`: Affine transformation with proper batch dimension handling
  - `validate_constraints()`: Targeted constraint validation (only checks referenced variables)
  - Debug logging support with guarded file operations

- **`transfer_functions.py`**: Transfer function interface and dispatch
  - Abstract `TransferFunction` base class with `supports_layer()` and `apply()` methods
  - Global transfer function registry with mode selection (interval/hybridz)
  - `dispatch_tf()`: Main entry point with optional debug logging
  - Configurable constraint logging (up to 50 constraints by default via `PerformanceOptions`)

- **`net_factory.py`**: YAML-driven network factory
  - Generates example ACT networks from YAML configurations
  - Automatic parameter generation for INPUT_SPEC and ASSERT layers
  - Comprehensive ASSERT specification guide (TOP1_ROBUST, MARGIN_ROBUST, LINEAR_LE, RANGE)
  - Proper tensor serialization using NetSerializer

- **`layer_schema.py`**: Layer type definitions and validation rules
  - Comprehensive schema definitions for all supported layer types
  - Parameter validation and metadata requirements

- **`bab.py`**: Branch-and-bound refinement implementation
  - BaB tree management with priority queues
  - Counterexample validation and refinement strategies
  - Configurable depth limits and timeout handling

- **`solver/`**: MILP/LP optimization backend
  - **`solver_gurobi.py`**: Gurobi MILP solver integration with license management
  - **`solver_torch.py`**: PyTorch-based LP solver for lightweight optimization
  - **`solver_base.py`**: Unified solver interface and status handling

- **Transfer Function Implementations**: Two precision/performance modes
  - **`interval_tf/`**: Fast interval-based bounds propagation
    - `IntervalTF`: Main implementation with layer-specific modules
    - Separate modules for MLP, CNN, RNN, and Transformer layers
  - **`hybridz_tf/`**: High-precision zonotope-based analysis
    - `HybridzTF`: Enhanced precision with zonotope domains
    - Separate modules for MLP, CNN, RNN, and Transformer layers

- **`serialization/`**: Net persistence and loading
  - **`serialization.py`**: `NetSerializer` with proper tensor encoding/decoding
  - **`test_serialization.py`**: Serialization correctness validation

- **`examples/`**: Example networks and test cases
  - **`config_gen_act_net.yaml`**: Generator configuration for example networks
  - **`nets/`**: Generated ACT Net JSON files (MNIST, CIFAR, control, reachability)
  - Networks include embedded INPUT_SPEC and ASSERT layers for spec-free verification

### **`pipeline/` - Testing Framework and Integration**
- **`torch2act.py`**: Automatic PyTorch→ACT Net conversion
  - Seamless conversion from PyTorch nn.Module to ACT Net representation
  - Preserves all verification constraints and model semantics
  - Support for complex wrapper layer patterns
  - Debug logging support for conversion process

- **`validate_verifier.py`**: Comprehensive verifier validation framework
  - Tests 12 networks (MNIST, CIFAR, control, reachability) with 2 solvers (Gurobi, TorchLP)
  - Concrete counterexample generation and validation
  - Formal verification result checking (SAT/UNSAT/CERTIFIED)
  - Detailed test reporting with pass/fail/inconclusive status

- **`model_factory.py`**: ACT Net factory for test networks
  - Pre-loads networks from `act/back_end/examples/nets/`
  - PyTorch model generation from ACT Nets
  - Integration with VerifiableModel wrapper layers

- **Testing Framework**: Comprehensive validation and regression testing
  - **`correctness.py`**: Verifier correctness validation with property-based testing
  - **`regression.py`**: Baseline capture and performance regression detection
  - **`integration.py`**: Front-end integration bridge for real ACT component testing

- **`config.py`**: YAML-based test scenario management
  - Configuration loading and validation
  - Test scenario composition and parameter management

- **Performance and Reporting**:
  - **`utils.py`**: Performance profiling, memory tracking, and optimization utilities
  - **`reporting.py`**: Results analysis and comprehensive report generation
  - **`run_tests.py`**: Command-line testing interface with parallel execution

- **`log/`**: Centralized logging directory
  - **`act_debug_tf.log`**: Transfer function debug output (layer-by-layer analysis, bounds, constraints)
  - Test execution logs and validation results

### **`util/` - Shared Utilities**

- **`cli_utils.py`**: Shared CLI utilities for unified device/dtype handling
  - `add_device_args(parser)`: Adds `--device` and `--dtype` arguments to ArgumentParser
  - `initialize_from_args(args)`: Initializes device_manager with parsed arguments
  - Used by all ACT native modules (front_end, pipeline, back_end)
  - Ensures consistent CLI interface across the entire ACT framework

- **`device_manager.py`**: GPU-first CUDA device handling
  - Automatic device detection and management
  - GPU memory optimization and fallback strategies
  - Global PyTorch device and dtype configuration
  - `initialize_device(device_str, dtype_str)`: Main initialization function

- **`path_config.py`**: Project path configuration and management
  - `get_project_root()`, `get_data_root()`, `get_config_root()`
  - `get_pipeline_log_dir()`: Returns absolute path to `act/pipeline/log/`
  - `ensure_gurobi_license()`: Automatic Gurobi license detection
  - Centralized path management for all ACT modules

- **`options.py`**: Performance configuration (PerformanceOptions only)
  - **`PerformanceOptions`**: Global debugging and performance flags
    - `debug_tf`: Enable/disable transfer function debug logging (default: True)
    - `validate_constraints`: Enable/disable constraint validation (default: True)
    - `debug_output_file`: Path to debug log (default: `act/pipeline/log/act_debug_tf.log`)
    - `debug_tf_max_constraints`: Max constraints to log per layer (default: 50)
    - Methods: `enable_debug_tf()`, `disable_all()`, `set_debug_output_file()`
  - Note: CLI argument parsing moved to individual module CLIs and cli_utils

- **`stats.py`**: Statistics and performance tracking
  - Verification result logging and analysis
  - Performance metrics collection and reporting

- **`model_inference.py`**: Model inference utilities
  - Helper functions for model execution and testing

### **`wrapper_exts/` - External Verifier Integrations**

#### **`abcrown/` - αβ-CROWN Integration**
- **`abcrown_verifier.py`**: αβ-CROWN wrapper and interface
  - Translates ACT parameters to αβ-CROWN format
  - Manages conda environment isolation for αβ-CROWN execution
  - Handles subprocess communication and result parsing
  - Provides error handling and comprehensive logging

- **`abcrown_runner.py`**: αβ-CROWN backend execution script
  - Contains code adapted from the open-source αβ-CROWN project with enhancements for ACT framework integration
  - Executes within isolated `act-abcrown` conda environment
  - Direct interface to αβ-CROWN complete verification engine
  - Independent execution without ACT path dependencies

#### **`eran/` - ERAN Integration**
- **`eran_verifier.py`**: ERAN wrapper and interface
  - Integration with ERAN abstract interpretation methods
  - Support for DeepPoly, DeepZono, and other ERAN domains
  - Parameter translation for ERAN backend compatibility


## Architecture Benefits

The three-tier modular architecture provides several key advantages:

### **Three-Tier Design with Unified CLI**
- **Front-End Separation**: User-facing data processing isolated from core verification logic
- **Back-End Focus**: Pure verification engine with PyTorch-native analysis and optimization
- **Pipeline Integration**: Comprehensive testing framework and Torch→ACT conversion bridge
- **Clean Boundaries**: Clear interfaces between data processing, verification, and testing
- **Unified CLI**: Consistent device/dtype handling across all ACT native modules via `cli_utils`

### **Modern CLI Architecture**
- **Modular CLIs**: Each major component (front_end, pipeline, back_end) has dedicated CLI
- **Shared Utilities**: `cli_utils.py` provides consistent `--device` and `--dtype` arguments
- **Clean Separation**: ACT native components use `cli_utils`, external verifiers use local `get_parser()`
- **Entry Points**: All CLIs executable via `python -m act.<module>` pattern
- **Command Discovery**: `--help` shows available commands and options for each module

### **Modern Verification Features**
- **Spec-Free Verification**: All constraints embedded in PyTorch models via wrapper layers
- **PyTorch-Native**: Verification engine operates directly on PyTorch tensors for performance
- **Automatic Conversion**: Seamless PyTorch→ACT Net conversion preserving all semantics
- **GPU-First**: Optimized CUDA device management with automatic fallback strategies
- **Debug Infrastructure**: Comprehensive debugging with transfer function logging and constraint validation

### **Code Quality and Maintainability**
- **Pythonic Containers**: ConSet with `__iter__` and `__len__` for natural iteration
- **Type Safety**: Proper type hints and validation throughout codebase
- **Guarded Operations**: Debug file I/O protected by feature flags to prevent performance impact
- **Centralized Logging**: All debug output to `act/pipeline/log/` with configurable detail levels
- **Batch Handling**: Proper tensor dimension management with assertions and squeeze operations

### **Modular Design**
- **Clear Separation**: Front-end, back-end, and pipeline modules have distinct responsibilities
- **Independent Development**: Modules can be developed, tested, and maintained separately
- **Easy Extension**: Add new verifiers by creating new modules in `wrapper_exts/`
- **Reusable Components**: Shared utilities and interfaces enable code reuse
- **Transfer Function Modes**: Pluggable TF implementations (interval vs. hybridz) via global registry

### **Testing and Validation**
- **Comprehensive Testing**: Pipeline framework provides correctness, regression, and performance testing
- **Validation Framework**: `validate_verifier.py` tests 12 networks across 2 solvers (24 test cases)
- **Concrete Counterexamples**: Real input generation to validate formal verification results
- **Integration Testing**: Real ACT component testing with front-end bridge
- **Continuous Validation**: Baseline capture and regression detection for quality assurance
- **Constraint Validation**: Targeted validation checks only variables referenced in constraints

### **Configuration Management**
- **Centralized Defaults**: Configuration files in `../modules/configs/` provide optimal parameters
- **Device Management**: Intelligent GPU/CPU device selection via unified CLI arguments
- **Memory Optimization**: Automatic memory tracking and optimization strategies
- **Environment Isolation**: Different verifiers can use separate conda environments
- **Parameter Management**: Unified command-line interface with type validation across all modules
- **Path Configuration**: Centralized path management via `path_config.py`
- **Consistent Interface**: All ACT native CLIs accept `--device {cpu,cuda,gpu}` and `--dtype {float32,float64}`

### **Debugging and Development**
- **PerformanceOptions**: Global flags for enabling/disabling debug features
- **Transfer Function Logging**: Layer-by-layer analysis with bounds, parameters, and constraints
- **Configurable Detail**: Control constraint logging depth (default: 50 per layer)
- **Targeted Validation**: Efficient constraint validation focusing on referenced variables
- **Guarded I/O**: All debug operations protected to minimize production overhead

### **Integration Flexibility**
- **Unified Interface**: Single entry point (`main.py`) for all verification tasks
- **Backend Abstraction**: Consistent API regardless of underlying verification method
- **Parameter Translation**: Automatic conversion between ACT and backend-specific formats
- **Result Standardization**: Uniform output format across all verification backends

### **Performance Optimization**
- **Memory Management**: Comprehensive memory tracking and optimization throughout pipeline
- **Utility Reuse**: Common operations centralized to eliminate code duplication
- **Efficient Imports**: Modular structure reduces import overhead and circular dependencies
- **GPU Acceleration**: PyTorch-native verification leverages GPU computation where beneficial
- **Configurable Logging**: Disable debug features in production for optimal performance

## Usage Examples

### Front-End CLI Examples
```bash
# List available TorchVision datasets and models
python -m act.front_end.torchvision_loader --list --device cpu

# List available VNNLIB files
python -m act.front_end.vnnlib_loader --list --device cpu

# Create specifications from TorchVision
python -m act.front_end --create --dataset mnist --model resnet18 --device cuda
```

### Pipeline CLI Examples
```bash
# Run pipeline tests
python -m act.pipeline --test --device cpu --dtype float32

# Run validation
python -m act.pipeline --validate --device cuda
```

### Back-End CLI Examples
```bash
# Generate all example networks from YAML
python -m act.back_end --generate --device cpu --dtype float64

# List all available example networks
python -m act.back_end --list-examples

# Display network structure (basic)
python -m act.back_end --info --network mnist_robust_easy.json --device cpu

# Display detailed network information
python -m act.back_end --info --network mnist_robust_easy.json --verbose --device cpu --dtype float64

# Run single-shot verification
python -m act.back_end --verify --network mnist_robust_easy.json --device cpu --dtype float64

# Run branch-and-bound verification
python -m act.back_end --verify --network mnist_robust_hard.json --bab --device cpu

# Test serialization round-trip
python -m act.back_end --test-serialization --device cpu --dtype float64
```

### External Verifier Examples
```bash
# Run ERAN verification
python act/wrapper_exts/ext_runner.py --verifier eran --model model.onnx --dataset mnist

# Run αβ-CROWN verification
python act/wrapper_exts/ext_runner.py --verifier abcrown --model model.onnx --dataset cifar10
```

## Examples and Network Generation
Example ACT networks are stored as JSON under `act/back_end/examples/nets/`.
These files are generated from the generator configuration
`act/back_end/examples/config_gen_act_net.yaml` using the network factory.
The test suite and serializer load networks from the `examples/nets` directory.
When authoring new example networks prefer the generator configuration and the
factory rather than hand-editing the JSON files.

### Using the Back-End CLI for Network Generation
The back-end CLI provides comprehensive tools for working with ACT networks:

1. **Generate Networks**: `--generate` creates networks defined in `config_gen_act_net.yaml`
2. **List Networks**: `--list-examples` shows all available networks organized by category
3. **Inspect Networks**: `--info` displays structure, use `--verbose` for detailed layer information
4. **Verify Networks**: `--verify` runs verification with optional `--bab` for branch-and-bound
5. **Test Serialization**: `--test-serialization` validates save/load functionality

All commands support unified `--device {cpu,cuda,gpu}` and `--dtype {float32,float64}` arguments.
