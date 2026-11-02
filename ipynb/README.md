# Jupyter Notebooks

This directory contains Jupyter notebooks for demonstrating and visualizing ACT's capabilities.

## Available Notebooks

### `torchvision_visualization.ipynb`

**Purpose**: Demonstrates ACT's TorchVision loader with custom perturbation visualization.

**Contents**:
1. **TorchVision MNIST Visualization**: Load MNIST dataset with ACT's TorchVision loader, create input specifications, and visualize perturbed images with model predictions
2. **Creating Custom Verification Bounds**: Tutorial on creating L∞ perturbation bounds for images

**Key Features**:
- Interactive visualization of MNIST input perturbations
- Side-by-side comparison of original and perturbed images
- Model inference on perturbed inputs with color-coded predictions (green=correct, red=incorrect)
- Flexible custom specification creation
- Educational examples for understanding verification concepts

**Usage**:
```bash
# Open in Jupyter
jupyter notebook ipynb/torchvision_visualization.ipynb

# Or use VS Code's notebook interface
code ipynb/torchvision_visualization.ipynb
```

### `vnnlib_visualization.ipynb`

**Purpose**: Demonstrates ACT's VNNLib loader with ACAS Xu network visualization.

**Contents**:
1. **VNNLib ACAS Xu Visualization**: Load ACAS Xu networks from VNNLib benchmarks, visualize input bounds, and test network behavior on sample points
2. **Understanding VNNLib Specifications**: Tutorial on SMT-LIB format constraints and standardized benchmarks

**Key Features**:
- ACAS Xu input specification visualization as bar charts
- Network testing on boundary points (lower/center/upper)
- Collision avoidance action interpretation
- SMT-LIB constraint parsing explanation
- VNN-COMP benchmark workflow demonstration

**Usage**:
```bash
# Open in Jupyter
jupyter notebook ipynb/vnnlib_visualization.ipynb

# Or use VS Code's notebook interface
code ipynb/vnnlib_visualization.ipynb
```

### `vnnlib_verifier.ipynb`

**Purpose**: Demonstrates the end-to-end verification workflow using ACT's native verifiers on VNNLIB benchmarks.

**Contents**:
1. **Environment Setup**: Loading ACT modules and checking Gurobi license.
2. **Benchmark Loading**: Using `VNNLibSpecCreator` to load models and properties.
3. **Model Synthesis**: Creating `VerifiableModel` instances.
4. **Formal Verification**: Running `verify_once` with different transfer function modes (Interval, HybridZ) and solvers (Gurobi, TorchLP).
5. **Result Analysis**: Interpreting `VerifyStatus` and exploring counterexamples.

**Key Features**:
- Step-by-step formal verification tutorial.
- Comparison between different verification precisions.
- Automated benchmark property parsing.
- Integration with Gurobi for MILP-based verification.

**Usage**:
```bash
# Open in Jupyter
jupyter notebook ipynb/vnnlib_verifier.ipynb
```

### `vnnlib_fuzzer.ipynb` ⭐ NEW

**Purpose**: End-to-end demonstration of ACTFuzzer on CIFAR-100 VNNLib benchmarks with integrated trace analysis.

**Contents**:
1. **VNNLib Benchmark Loading**: Load CIFAR-100 VNNLib instances with model and specification parsing
2. **Multi-Instance Fuzzing**: Run fuzzing on multiple instances with configurable timeout and mutation strategies
3. **Counterexample Visualization**: Side-by-side comparison of original/perturbation/perturbed images
4. **Trace Analysis**: Integrated trace visualization with coverage, strategy effectiveness, and violations
5. **Interactive Trace Explorer**: Widget-based trace browser for detailed inspection

**Key Features**:
- 🎯 **Complete workflow**: Load → Fuzz → Analyze → Visualize in one notebook
- 📊 **Automatic trace capture**: Fuzzing traces saved and analyzed with rich visualizations
- 🔍 **Interactive widgets**: Dropdown explorer for browsing traces across multiple instances
- 🎨 **Input heatmaps**: Before/after/diff visualizations for mutated inputs
- 📈 **Strategy analysis**: Box plots and pie charts showing mutation effectiveness
- 💾 **Export capabilities**: Trace export to PyTorch files, CSV summaries
- ⚡ **Performance tracking**: Iterations/sec, coverage metrics, counterexample detection

**Usage**:
```bash
# Open in Jupyter
jupyter notebook ipynb/vnnlib_fuzzer.ipynb

# Or use VS Code's notebook interface
code ipynb/vnnlib_fuzzer.ipynb
```

**Requirements**:
- `matplotlib`, `pandas` (visualization and analysis)
- `ipywidgets>=8.0.0` (interactive explorer)
- `torch`, `torchvision` (model inference)

**Configuration**:
- Modify `timeout_seconds` in Cell 4 to adjust fuzzing budget
- Change `trace_level` (0=disabled, 1=basic, 2=full) for tracing detail
- Adjust `trace_sample_rate` (1=every iteration, 10=every 10th) to control file size
- Select `trace_storage` ("json" or "hdf5") based on performance needs

### `tensor_bab.ipynb`

**Purpose**: Demonstrates K-batched ("tensor") branch-and-bound (`verify_bab_batched`) on three VNN-COMP instances: ACAS Xu (one CERTIFIED, one FALSIFIED) and a CIFAR-100 ResNet (CERTIFIED).
Also compares CLIMB off/on on CIFAR-100 idx_8420 with the same 100 s budget.

**Contents**:
1. **Setup and helpers**: ONNX + VNNLIB to an ACT net, then BaB with a per-iteration trace.
2. **Certified case (prop_3 / net 2_4)**: one iteration table showing K, pruning, branching, pool size and seconds.
3. **Falsified case (prop_2 / net 1_2)**: a counterexample checked on the concrete network and against its input box.
4. **CIFAR-100 ResNet (needs a CUDA GPU)**: ReLU-neuron splitting with the alpha-eta dual, automatic K and a 100 s budget; skipped on CPU.
5. **CLIMB comparison (CUDA)**: idx_8420 is UNKNOWN with CLIMB off and CERTIFIED with it on; only `climb_enabled` changes.
6. **Summary**: one row per initial case (verdict, ground truth, iterations, sub-problems, max K, seconds).

**Key Features**:
- Plain dual bounds on up to 1024 sub-boxes per iteration, splitting the widest input dimension (ACAS Xu).
- The ACAS Xu cases run in seconds on GPU or CPU; fetch missing files with `python -m act.front_end --download acasxu_2023` (and `cifar100_2024`).

**Usage**:
```bash
jupyter notebook ipynb/tensor_bab.ipynb
```

## Running Notebooks

### Prerequisites
Ensure you have the `act-py312` (or your ACT environment) conda environment activated with:
- `ipykernel`
- `matplotlib`
- `numpy`
- `torch`
- `torchvision`

### Environment Setup
```bash
# Activate ACT environment
conda activate act-py312

# Install Jupyter if needed
conda install jupyter ipykernel matplotlib -y

# Register kernel
python -m ipykernel install --user --name act-py312 --display-name "Python (ACT)"
```

## Adding New Notebooks

When creating new notebooks for ACT:

1. **Place in this directory**: Keep all notebooks organized in `ipynb/` at the project root
2. **Document purpose**: Add a section to this README describing the notebook
3. **Use relative imports**: Import ACT modules with proper path handling:
   ```python
   import sys
   import os
   act_root = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
   if act_root not in sys.path:
       sys.path.insert(0, act_root)
   ```
4. **Test with clean kernel**: Restart kernel and run all cells to ensure reproducibility

## Notebook Categories

Future notebooks might include:
- **Tutorials**: Step-by-step guides for using ACT features
- **Benchmarks**: Performance analysis and comparison visualizations
- **Examples**: Real-world verification case studies
- **Debugging**: Diagnostic tools for troubleshooting verification issues
