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
├── gui/                  # ACT-side support for ACT-GUI
│   ├── act_gui.py        # Shared sink channel, record base, options, and run control
│   ├── verifier_gui.py   # Propagation records, ReLU describers, verify_once options
│   ├── bab_gui.py        # BaB wave and terminal search records
│   ├── fuzzer_gui.py     # ACTFuzzer options, lifecycle records, and FuzzControl
│   └── provenance_gui.py # Input provenance queries and offline SafeNLP reconstruction
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
- **Property Checking**: `OutputSpec` violations through `act/util/counterexample.py`, the
  same check verification uses.

### GUI support (`gui/`)
All ACT code that exists for ACT-GUI lives here. Verification (including BaB) and fuzzing
share one interface in `gui.act_gui`:

- **Channel**: `observe(sink)` registers a sink for a block, `emit(record)` delivers to every
  sink, and `observed_phase(name)` brackets a function with `PhaseEvent` start/end/error.
  A sink that raises is logged and detached. Producers check `enabled` before building records.
- **Records**: every record derives from `Record` and carries `t_ns` (`perf_counter_ns`).
- **Options**: `Option` (kind, admissible values, live flag, `validate`), `OptionDefault`
  (an option with its YAML default), and `validate_changes(options, changes)`.
- **Control**: `RunControl(options)`: `request_stop()`, `pause()` / `resume()`, and
  `update(**changes)` for live options, validated at once and applied by the engine at its
  own safe points.

`gui.verifier_gui` adds the `verify_once` records: committed layer bounds (`LayerBounds`),
ReLU descriptions (`Relaxation`, requested with the sink attribute `wants_relaxation = True`
and built by `DESCRIBERS`, one per domain), and dual backward/specification records
(`DualBackward`, `SpecBound`), all keyed by `RecordKey`. `verification_options()` lists the
tunable `verify_once` settings (`dual_iters`, `dual_refresh_forward`, `dual_config.<field>`,
`hybridz_tolerance`) with their backend.yaml defaults.
These records reference live tensors; sinks copy values they need to retain.

Interval and HybridZ report each layer's committed output (stage `out`); dual forward
reports activation inputs as `pre` and other layers as `out`; dual backward records
(alpha `[B, M, n]`, margins and slack `[B, M]`) use pass `backward`.

`gui.bab_gui` adds `BaBWave` and `BaBTerminal` records for the batched search loop.
Wave records contain solved nodes, parent/child IDs, split branches, bounds, and frontier
changes; terminal records cover root presolve, counterexamples, and ordinary completion.

`gui.fuzzer_gui` covers the whole `ACTFuzzer` lifecycle on the same interface:

- **Options**: `fuzzing_options()` lists every tunable `FuzzingConfig` field
  (`FUZZING_OPTIONS` in `fuzzing/actfuzzer.py`) with its pipeline.yaml default.
- **Records** (to the same sinks): `FuzzStart` (resolved config, normalized strategy
  weights, batch size, device); `FuzzStep` per fuzzed batch (seeds, mutated inputs,
  strategy, the lanes ACT's property checker accepted with their `Counterexample`
  objects, coverage, corpus size); `FuzzConfigChange` per applied or rejected update;
  `FuzzEnd` (report and stop reason).
- **Control**: `ACTFuzzer(..., control=FuzzControl())`, the `RunControl` over
  `FUZZING_OPTIONS`; requests apply between batches (paused time is excluded from the
  timeout) through `ACTFuzzer.update_config`, which scripts can call directly.

The GUI modules each have a self-check entrypoint: `python -m act.pipeline.gui.act_gui`
checks the shared channel; `python -m act.pipeline.gui.verifier_gui` checks propagation
records; `python -m act.pipeline.gui.fuzzer_gui` checks fuzzing records and live control;
and `python -m act.pipeline.gui.bab_gui` checks BaB wave and terminal records.

`gui.provenance_gui` keeps NLP input provenance in one module: `input_provenance`
reads the source sentence and perturbations recorded for a property, while
`nearest_inputs` ranks known sentences near an input-space point. It checks the
property's recorded hash and never decodes a counterexample into text.
The same module rebuilds the SafeNLP sidecar offline from the published safeNLP
boxes and ANTONIO data and generator. Run
`python -m act.pipeline.gui.provenance_gui --help` for the build inputs, or
`python -m act.pipeline.gui.provenance_gui --self-check` to check both the reader
and reconstruction helpers. Running the module without arguments also runs both
self-checks. Encoder and dataset packages are loaded only when rebuilding data.

## Logging and Diagnostics

All execution logs, including detailed transfer function analysis, are stored in `act/pipeline/log/`.
- `act_debug_tf.log`: Layer-by-layer transfer function analysis (enabled via `PerformanceOptions`).
- `fuzzing_results/`: Summary and counterexamples from fuzzing runs.

## License

ACT is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

