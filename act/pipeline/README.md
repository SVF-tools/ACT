# ACT Pipeline Module

Testing and integration framework for the Abstract Constraint Transformer (ACT). This module provides automatic PyTorch→ACT conversion, whitebox fuzzing, and comprehensive verifier validation.

## Overview

The ACT Pipeline bridges the front-end data processing and back-end verification core. It includes tools for:
- **PyTorch ↔ ACT Conversion**: Seamlessly convert between PyTorch `nn.Module` and ACT `Net` representations.
- **Inference-based Fuzzing**: Rapidly find counterexamples using gradient-guided mutations and coverage tracking.
- **Verifier Validation**: Rigorous soundness and numerical correctness checks for verification backends.
- **Benchmark Management**: Automated downloading and listing of VNNLIB benchmarks and TorchVision data-model pairs.

## Architecture

```
act/pipeline/
├── verification/         # Verification utilities submodule
│   ├── torch2act.py      # Automatic PyTorch→ACT conversion
│   ├── act2torch.py      # ACT→PyTorch conversion utilities
│   ├── validate_verifier.py # Verifier correctness validation
│   ├── model_factory.py  # ACT Net factory for test networks
│   ├── utils.py          # Shared utilities and profiling
│   ├── llm_probe.py      # LLM-based probing
│   └── per_neuron_bounds.py # Per-neuron activation checking
├── fuzzing/              # Whitebox fuzzing framework
│   ├── actfuzzer.py      # Main fuzzing engine
│   └── ...
└── log/                  # Centralized execution logs
```

## Command-Line Interface

The pipeline is accessible via `python -m act.pipeline`. It provides commands for benchmark management, fuzzing, and verification.

### Benchmark Management

```bash
# List available VNNLIB categories
python -m act.pipeline --list

# Search for specific benchmarks
python -m act.pipeline --search acas

# Get detailed information about a category
python -m act.pipeline --info acasxu_2023

# Download a VNNLIB category
python -m act.pipeline --download acasxu_2023

# List downloaded data-model pairs
python -m act.pipeline --list-downloaded
```

### Whitebox Fuzzing

Run ACTFuzzer on VNNLIB or TorchVision targets:

```bash
# Fuzz a VNNLIB benchmark
python -m act.pipeline --fuzz --category acasxu_2023 --iterations 5000

# Fuzz a TorchVision dataset
python -m act.pipeline --fuzz --creator torchvision --dataset MNIST
```

### Verifier Validation

Ensure verifier soundness and numerical precision:

```bash
# Run unified two-level validation on NetFactory nets (Level 1 + Level 2)
python -m act.pipeline --verify netfactory --validate-soundness --device cpu --dtype float64

# Run with specific solver and TF mode
python -m act.pipeline --verify netfactory --validate-soundness --tf-modes interval --solvers torchlp
python -m act.pipeline --verify netfactory --validate-soundness --solvers dual

# Run on a VNNLIB benchmark with validation enabled
python -m act.pipeline --verify vnnlib --category acasxu_2023 --validate-soundness
```

### Conversion Tests

```bash
# Run PyTorch→ACT conversion tests
python -m act.pipeline --verify torch2act

# Run ACT→PyTorch conversion tests
python -m act.pipeline --verify act2torch
```

## Key Components

### Torch2ACT Converter (`verification/torch2act.py`)
Automatically converts PyTorch models to ACT's intermediate representation. It preserves verification constraints embedded in `VerifiableModel` wrappers and ensures weight equivalence.

### Verifier Validator (`verification/validate_verifier.py`)
Implements multi-level validation:
1. **Level 1 (Soundness)**: Verifies that the verifier does not report CERTIFIED when concrete counterexamples exist.
2. **Level 2 (Numerical Precision)**: Checks that abstract bounds correctly overapproximate concrete activation values across all layers.

### ACTFuzzer (`fuzzing/actfuzzer.py`)
A fast, GPU-accelerated fuzzer that uses:
- **Gradient Mutations**: FGSM/PGD-style perturbations.
- **Coverage Tracking**: DeepXplore-style neuron coverage.
- **Property Checking**: Automated detection of `OutputSpec` violations.

## Logging and Diagnostics

All execution logs, including detailed transfer function analysis, are stored in `act/pipeline/log/`.
- `act_debug_tf.log`: Layer-by-layer transfer function analysis (enabled via `PerformanceOptions`).
- `fuzzing_results/`: Summary and counterexamples from fuzzing runs.

## License

ACT is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

