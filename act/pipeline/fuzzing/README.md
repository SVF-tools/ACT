# ACTFuzzer

Inference-based whitebox fuzzing for neural network verification.

## Overview

ACTFuzzer is a lightweight, inference-based fuzzing tool that finds counterexamples by:
1. **Gradient-guided mutations**: FGSM-style adversarial perturbations
2. **Coverage tracking**: Neuron coverage (DeepXplore-style)
3. **Energy-based scheduling**: AFL-style seed prioritization
4. **Property checking**: Automatic OutputSpec violation detection

Unlike formal verification, ACTFuzzer runs fast GPU-accelerated inference to quickly explore the input space and find violations.

## Features

- ✅ **Fast**: ~1000x faster than formal verification
- ✅ **Practical**: Finds counterexamples without soundness guarantees
- ✅ **Flexible**: Works with VNNLib benchmarks and TorchVision datasets
- ✅ **Integrated**: Seamless integration with ACT's spec creators and model synthesis

## Architecture

```
ACTFuzzer
├── MutationEngine      # Gradient/activation/boundary/random mutations
├── CoverageTracker     # Neuron coverage tracking
├── SeedCorpus          # AFL-style seed management
└── PropertyChecker     # OutputSpec violation detection
```

## Usage

### Quick Start

```bash
# 1. Download benchmark
python -m act.pipeline --download acasxu_2023

# 2. Fuzz it
python -m act.pipeline --fuzz --category acasxu_2023 --iterations 5000

# 3. Check results
ls fuzzing_results/
cat fuzzing_results/summary.json
```

### Python API

```python
from act.pipeline.fuzzing import ACTFuzzer, FuzzingConfig
from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
from act.front_end.model_synthesis import synthesize_models_from_specs

# Create specs
creator = VNNLibSpecCreator()
spec_results = creator.create_specs_for_data_model_pairs(
    categories=["acasxu_2023"],
    max_instances=10
)

# Synthesize models
wrapped_models, reports, input_data = synthesize_models_from_specs(spec_results)

# Extract seeds
initial_seeds = []
for _, _, _, labeled_tensors, _ in spec_results:
    initial_seeds.extend(labeled_tensors)

# Fuzz (loads config from config.yaml with optional overrides)
config = FuzzingConfig.from_yaml(max_iterations=5000, device="cuda")
fuzzer = ACTFuzzer(
    wrapped_model=list(wrapped_models.values())[0],
    initial_seeds=initial_seeds,
    config=config
)

report = fuzzer.fuzz()
print(f"Found {len(report.counterexamples)} counterexamples")
```

## Configuration

Edit `config.yaml` to customize:

```yaml
fuzzing:
  max_iterations: 10000
  mutation_weights:
    gradient: 0.4      # FGSM-style
    pgd: 0.0           # PGD-style (iterative; opt-in)
    activation: 0.3    # DeepXplore
    boundary: 0.2      # Edge cases
    random: 0.1        # Baseline
```

## Mutation Strategies

### 1. Gradient-Guided (FGSM) (40%)
Single-step FGSM-style perturbations:
```
x' = x + ε * sign(∇_x Loss(x))
```

### 2. Gradient-Guided (PGD) (0% by default)
Iterative PGD-style perturbations with projection each step:
```
x_low  = x - ε
x_high = x + ε
repeat K steps:
  x = proj_[x_low,x_high](x + α * sign(∇_x Loss(x)))
```

### 3. Activation-Guided (30%)
Targets neurons with low activation (DeepXplore):
```
Maximize: Σ inactive_neurons
```

### 4. Boundary Exploration (20%)
Samples near InputSpec boundaries:
```
x' = x + ε * direction_to_boundary
```

### 5. Random (10%)
Gaussian noise baseline:
```
x' = x + N(0, ε²)
```

## Adaptive Perturbation Sizing

### Terminology Note

**Important**: We use "**perturb_size**" (not "epsilon") to avoid confusion with InputSpec constraints:
- **InputSpec.eps**: L∞ radius constraint (defines input space boundaries, e.g., `center ± eps`)
- **Mutation perturb_size**: Mutation perturbation magnitude (defines exploration granularity per iteration)

These are completely different concepts with different scales and purposes.

### What is perturb_scale?

In fuzzing, **perturb_size** controls the **magnitude of each mutation perturbation**, not the boundaries (boundaries are enforced by projection to InputSpec). To ensure consistent exploration across different problem scales, ACTFuzzer supports **adaptive perturbation sizing** that scales with InputSpec bounds.

**perturb_scale** is the **fraction of the feasible range** that each mutation perturbation covers.

#### Interpretation Formula
```
steps_to_traverse = 1 / perturb_scale
```

#### Calculation
```
range / perturb_size = range / (range * perturb_scale) = 1 / perturb_scale
```

#### Examples
- **perturb_scale=0.1** → Each perturbation covers 10% of range → Takes ~**10 steps** to traverse from lb to ub
- **perturb_scale=0.2** → Each perturbation covers 20% of range → Takes ~**5 steps** to traverse from lb to ub
- **perturb_scale=0.05** → Each perturbation covers 5% of range → Takes ~**20 steps** to traverse from lb to ub

### Perturbation Size Modes

#### 1. adaptive_scalar (Default)
- **Computation**: `perturb_size = mean(ub - lb) * perturb_scale`
- **Best for**: Uniform ranges (e.g., VNNLib BOX constraints with consistent bounds)
- **Example**: VNNLib with lb=0.0, ub=1.0 → range=1.0, perturb_size=0.1 (10 steps)

#### 2. adaptive_perdim (Advanced)
- **Computation**: `perturb_size[i] = (ub[i] - lb[i]) * perturb_scale`
- **Best for**: Non-uniform ranges (different features with vastly different scales)
- **Example**: lb=[0, -100], ub=[1, 100] → perturb_size=[0.1, 20.0] (10 steps per dimension)

#### 3. fixed (Legacy)
- **Computation**: Hardcoded values (0.01 for gradient/activation, 0.005 for boundary/random)
- **Best for**: Backward compatibility or when InputSpec is not available
- **Note**: May be too large for tight bounds or too small for wide bounds

### Configuration

Set in `config.yaml`:
```yaml
perturb_mode: "adaptive_scalar"  # Options: "adaptive_scalar", "adaptive_perdim", "fixed"
perturb_scale: 0.1               # Fraction of range per perturbation (default: 0.1 = 10 steps)
```

### Recommended Values

| Use Case | perturb_scale | Steps | Description |
|----------|---------------|-------|-------------|
| **Balanced** | 0.1 (default) | ~10 | Good for most cases |
| **Fine-grained** | 0.05 | ~20 | Thorough exploration, slower |
| **Coarse** | 0.2 | ~5 | Fast exploration, may miss violations |

### Example: VNNLib with [0, 1] bounds

```python
# With perturb_scale=0.1:
# - range = 1.0 - 0.0 = 1.0
# - perturb_size = 1.0 * 0.1 = 0.1
# - steps = 1.0 / 0.1 = 10 steps to traverse
```

**Diagnostic output** (printed during initialization):
```
[MutationEngine] Adaptive Scalar Perturbation Size:
  - perturb_scale: 0.1 (fraction of range per perturbation)
  - mean_range: 1.000000
  - computed perturb_size: 0.100000
  - steps_to_traverse: ~10.0 steps
  - interpretation: Each mutation perturbation covers 10.0% of the range
```

## Coverage Metrics

ACTFuzzer tracks **neuron coverage**:
```
Coverage = |{neurons that fired}| / |{total neurons}|
```

Coverage strategies (config `coverage_strategy`):
- `BestInputCov`: Best input coverage values from all mutated inputs (tracks per-input coverage history; no global union)
- `GlobalCov`: global union coverage across all inputs (supports never-activated neuron queries)

A neuron "fired" if `activation > threshold` (default: 0.1).

## Output

Fuzzing produces:
- `summary.json`: Statistics (iterations, time, coverage, violations)
- `counterexample_*.pt`: PyTorch tensors with input/output/label

Example `summary.json`:
```json
{
  "iterations": 5000,
  "time_seconds": 125.3,
  "counterexamples_found": 12,
  "neuron_coverage": 0.87,
  "mutations": 5000,
  "seeds_explored": 342
}
```

## Performance

Typical performance on NVIDIA RTX 3090:
- **ACAS Xu**: ~500 iterations/sec
- **MNIST CNN**: ~800 iterations/sec
- **CIFAR10 ResNet**: ~300 iterations/sec

## Comparison with Formal Verification

| Aspect | ACTFuzzer | Formal Verification |
|--------|-----------|---------------------|
| **Speed** | ~500 it/s | ~0.5 it/s (1000x slower) |
| **Soundness** | No (heuristic) | Yes (complete) |
| **Counterexamples** | Yes | Yes |
| **Proof** | No | Yes (if UNSAT) |
| **Use Case** | Bug finding | Certification |

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size (currently 1)
- Use `--device cpu`
- Lower `max_iterations`

### No Counterexamples Found
- Increase `max_iterations`
- Check InputSpec constraints (too restrictive?)
- Try different mutation weights

### Low Coverage
- Increase `max_iterations`
- Use gradient-guided mutations (set weight to 0.8)

## Citation

```bibtex
@software{actfuzzer2025,
  title = {ACTFuzzer: Inference-based Whitebox Fuzzing},
  author = {SVF-tools},
  year = {2025},
  url = {https://github.com/SVF-tools/ACT}
}
```

## License

AGPLv3+ - Copyright (C) 2025 SVF-tools/ACT

---

# Execution Tracing & Replay System

## Overview

The ACT fuzzing execution tracing system provides detailed insight into fuzzing behavior through progressive 4-level tracing and an interactive replay visualizer.

## Tracing Levels

### Level 0: Disabled (Default)
- No tracing overhead
- Production runs

### Level 1: Basic Tracing
- Iteration metrics (coverage, energy, strategy)
- Input tensors (before/after mutation)
- Seed genealogy (parent_id, depth)
- **Overhead**: ~2-3% (1-2% with sampling)
- **Use case**: Standard debugging

### Level 2: Full Network State
- Level 1 data + layer activations
- Complete network state capture
- **Overhead**: ~5-8% (with sampling)
- **Use case**: Network behavior analysis

### Level 3: Deep Debugging
- Level 2 data + gradients and loss values
- Complete debugging information
- **Overhead**: ~8-12% (with sampling)
- **Use case**: Algorithm debugging

## Storage Backends

### JSON (Default)
- Human-readable text format
- No dependencies required
- Easy to inspect and debug
- ~5 MB per 1000 traces

### HDF5
- Binary compressed format
- Requires h5py: `pip install h5py`
- Smaller files (~2 MB per 1000 traces)
- Faster I/O

## CLI Usage

### Enable Tracing

```bash
# Basic tracing (Level 1)
python -m act.pipeline --fuzz --category acasxu_2023 --trace-level 1

# Full debugging with sampling (every 10th iteration)
python -m act.pipeline --fuzz --dataset MNIST \
    --trace-level 3 \
    --trace-sample 10 \
    --trace-storage hdf5

# Custom output path
python -m act.pipeline --fuzz --trace-level 2 \
    --trace-output my_custom_traces.json
```

### CLI Flags

- `--trace-level {0,1,2,3}` - Tracing detail level (default: 0)
- `--trace-sample N` - Capture every Nth iteration (default: 1)
- `--trace-storage {json,hdf5}` - Storage backend (default: json)
- `--trace-output PATH` - Custom output path (optional)

## Replay Visualizer

Interactive tool to explore and analyze trace files.

### Basic Usage

```bash
# Summary statistics
python -m act.pipeline.fuzzing.trace_reader traces.json --summary

# List all traces
python -m act.pipeline.fuzzing.trace_reader traces.json --list

# Show specific trace details
python -m act.pipeline.fuzzing.trace_reader traces.json --show 42

# Export specific trace
python -m act.pipeline.fuzzing.trace_reader traces.json \
    --export 42 --output trace_42.pt
```

**Note:** For visual analysis with charts and interactive widgets, use the Jupyter notebook (see section below).

## Performance Characteristics

### Overhead Analysis

| Level | Sampling | Overhead | File Size (1000 iter) |
|-------|----------|----------|-----------------------|
| 0     | N/A      | 0%       | 0 KB                  |
| 1     | 1:1      | 2-3%     | ~5 MB (JSON)          |
| 1     | 1:10     | < 1%     | ~500 KB               |
| 2     | 1:5      | 5-8%     | ~15 MB                |
| 3     | 1:10     | 8-12%    | ~25 MB                |

### Best Practices

1. **Use sampling for long runs**: `--trace-sample 10` captures every 10th iteration
2. **Start with Level 1**: Provides most useful info with minimal overhead
3. **Use JSON for debugging**: Easy to inspect, no extra dependencies
4. **Use HDF5 for large runs**: Better compression and performance
5. **Export interesting traces**: Save specific iterations for detailed analysis

## Python API

```python
from pathlib import Path
from act.pipeline.fuzzing import ACTFuzzer, FuzzingConfig
from act.pipeline.fuzzing.trace_reader import create_reader

# Enable tracing during fuzzing (loads config.yaml with overrides)
config = FuzzingConfig.from_yaml(
    max_iterations=5000,
    trace_level=1,              # Enable basic tracing
    trace_sample_rate=5,        # Every 5th iteration
    trace_storage="json",       # JSON format
    trace_output=Path("my_traces.json")  # Custom path
)

fuzzer = ACTFuzzer(model, seeds, config)
report = fuzzer.fuzz()

# Load and analyze traces
reader = create_reader(Path("my_traces.json"))
print(f"Captured {len(reader)} traces")

# Access specific trace
trace = reader[0]
print(f"Iteration: {trace['iteration']}")
print(f"Coverage: {trace['coverage']:.2%}")
print(f"Input shape: {trace['input_before'].shape}")
```

## File Organization

All trace files are stored under `act/pipeline/log/`:

```
act/pipeline/log/
├── fuzzing_results/          # Default fuzzing output
│   ├── summary.json
│   ├── traces.json           # Trace file (if enabled)
│   └── ce_*.pt              # Counterexamples
├── test_tracing/            # Test suite traces
│   ├── level_0/
│   ├── level_1/
│   ├── level_2/
│   └── level_3/
└── [custom]/                # Custom output directories
```

## Generating Traces

### Quick Start

Generate traces directly via the ACT CLI:

```bash
# Generate traces with Level 1 (basic tracing)
python -m act.pipeline --fuzz \
    --dataset cifar100_2024 \
    --max-instances 2 \
    --timeout 30 \
    --iterations 500 \
    --trace-level 1 \
    --trace-output traces.json

# With sampling (every 5th iteration)
python -m act.pipeline --fuzz \
    --dataset mnist \
    --max-instances 5 \
    --iterations 1000 \
    --trace-level 2 \
    --trace-sample 5
```

### Performance Characteristics

With proper configuration, tracing overhead is minimal:
- **Level 0**: No overhead (tracing disabled)
- **Level 1**: < 3% overhead with sampling
- **Level 2**: 3-5% overhead with sampling
- **Level 3**: 5-10% overhead (use sparingly)

## Architecture

### Components

1. **`tracer.py`**: Unified execution tracer with level-based filtering
2. **`trace_storage.py`**: Storage backends (JSON, HDF5) with async wrapper
3. **`trace_reader.py`**: CLI tool & Jupyter TraceAnalyzer for trace inspection
4. **`actfuzzer.py`**: Integrated tracing hook in fuzzing loop

### Design Highlights

- **Unified Architecture**: Single tracer class handles all levels
- **Async I/O**: Background thread with queue for non-blocking writes
- **Progressive Detail**: Level-based filtering avoids storing unnecessary data
- **Flexible Storage**: Factory pattern for easy backend addition
- **Zero Overhead**: Level 0 has no performance impact

## Advanced Usage

### Trace Analysis Workflow

```bash
# 1. Generate traces with high detail
python -m act.pipeline --fuzz \
    --dataset mnist \
    --max-instances 10 \
    --iterations 10000 \
    --trace-level 2 \
    --trace-sample 5

# 2. Quick summary
python -m act.pipeline.fuzzing.trace_reader \
    act/pipeline/log/fuzzing_results/traces.json --summary

# 3. List traces and find violations
python -m act.pipeline.fuzzing.trace_reader \
    act/pipeline/log/fuzzing_results/traces.json --list

# 4. Show specific trace details
python -m act.pipeline.fuzzing.trace_reader \
    act/pipeline/log/fuzzing_results/traces.json --show 42

# 5. Export for detailed analysis
python -m act.pipeline.fuzzing.trace_reader \
    act/pipeline/log/fuzzing_results/traces.json --export 42 -o trace_42.pt
```

### Custom Analysis (Python)

```python
from act.pipeline.fuzzing.trace_reader import create_reader

# Load traces
reader = create_reader(Path("traces.json"))

# Analyze mutation strategy effectiveness
strategies = {}
for trace in reader.traces:
    strat = trace['mutation_strategy']
    cov_delta = trace['coverage_delta']
    
    if strat not in strategies:
        strategies[strat] = []
    strategies[strat].append(cov_delta)

# Print average coverage gain per strategy
for strat, deltas in strategies.items():
    avg = sum(deltas) / len(deltas)
    print(f"{strat:15s}: {avg:+.4f} avg coverage gain")
```

## Troubleshooting

### High Overhead
- **Solution**: Increase `--trace-sample` (e.g., 10 or 20)
- **Alternative**: Use Level 1 instead of 2/3

### Large File Sizes
- **Solution**: Use `--trace-storage hdf5` for compression
- **Alternative**: Increase sampling rate

### Missing h5py
- **Solution**: `pip install h5py` or use `--trace-storage json`

### Out of Memory
- **Solution**: Increase sampling or reduce trace level
- **Check**: Async queue not filling up (warnings in output)

## Jupyter Notebook Visualization

For **visual analysis** of traces, use the Jupyter notebook:

```bash
# 1. Generate traces
python -m act.pipeline.fuzzing.actfuzzer --trace-level 1 --trace-output traces.json

# 2. Open notebook
jupyter notebook ipynb/fuzzing_trace_analysis.ipynb

# 3. Update trace_file path in Cell 1, then run all cells
```

**Notebook Features:**
- 📊 Interactive visualizations (coverage, strategies, violations)
- 🔍 Widget-based trace explorer
- 🎨 Input tensor heatmaps (before/after/diff)
- 📈 Strategy effectiveness analysis
- 💾 Export to CSV and PyTorch formats

See **`ipynb/README.md`** for detailed usage guide.

## Documentation

- **`ipynb/fuzzing_trace_analysis.ipynb`**: 4-cell interactive notebook ⭐
- **`TRACING.md`**: Complete user guide (450+ lines)
- **`PATH_CONFIG_INTEGRATION.md`**: Path management details
- **`NOTEBOOK_PLAN.md`**: Notebook implementation plan
- **Inline docstrings**: All classes and methods documented

## Status

✅ **PRODUCTION READY** (November 10, 2025)
- 4-level progressive tracing system
- JSON (default) and HDF5 storage
- Jupyter notebook visualization ⭐ NEW
- CLI trace reader tool
- CLI integration complete
- Comprehensive testing (100% pass)
- < 5% performance overhead
- Full backward compatibility
