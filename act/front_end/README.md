# ACT Front-End: Specification Creators & Unified CLI

The ACT front-end provides specification creators for generating verification tasks from different data sources, with a unified CLI interface featuring automatic detection.

## Direct instances and compact BERT queries

```bash
python -m act.back_end --verify --onnx model.onnx --vnnlib property.vnnlib \
  --save-net verified.json --solver torchlp --device cpu --dtype float64
python -m act.back_end --verify --query-index queries.csv --query-id 0 \
  --save-net verified.json --solver torchlp --device cpu --dtype float64
python -m act.front_end --synthesis --creator bert \
  --checkpoint-dir buffet/models/model_sst_1 --position-mode sweep --conversion-route b \
  --device cpu --dtype float64
```

The query CSV needs `dataset,split,depth,example_id,position,p,eps`; optional
`checkpoint_sha256` must match the checkpoint and its manifest. Query IDs count
data rows from zero. SST/Yelp raw data and the matching compact checkpoint must
already be present under ACT's data root; `get_bert_artifact_root()` retains the
`data/buffet` checkpoint and manifest directory. For Yelp, use
`calibration` or `evaluation` with the fixed split of raw `test.csv` rows.
Front-end synthesis selects the compact model when `--checkpoint-dir` is set;
without it, the embedding demo remains the default. Relative
checkpoint paths, including direct `load_compact_bert` calls, are resolved under ACT's data root; absolute paths
are used as given.
`prefix` perturbs positions starting at index zero, so compact BERT's `[CLS]`
is included in `perturbed_words`. `sweep` and indexed queries instead select
non-special WordPieces that are not `##` continuations; query positions index
the complete sequence including `[CLS]`. Both routes preserve this convention.
`--save-net` writes the Net actually verified; multiple VNNLIB disjuncts use
`verified.d0.json`, `verified.d1.json`, and so on.

## Quick Start

```bash
# ============================================================================
# LISTING - Browse available datasets and categories
# ============================================================================

# List all data sources (40 TorchVision + 34 VNNLIB)
python -m act.front_end --list

# List only TorchVision datasets
python -m act.front_end --list --creator torchvision

# List only VNNLIB categories
python -m act.front_end --list --creator vnnlib

# Show available creators with details
python -m act.front_end --list-creators

# ============================================================================
# SEARCHING - Find specific datasets or categories
# ============================================================================

# Search across both creators (auto-detects)
python -m act.front_end --search mnist         # Finds: MNIST, FashionMNIST, KMNIST, EMNIST
python -m act.front_end --search cifar         # Finds: CIFAR10, CIFAR100, cifar100_2024
python -m act.front_end --search imagenet      # Finds: ImageNet (TorchVision)
python -m act.front_end --search yolo          # Finds: yolo_2023 (VNNLIB)
python -m act.front_end --search transformer   # Finds: vit_2023, safenlp_2024 (VNNLIB)
python -m act.front_end --search acas          # Finds: acasxu_2023 (VNNLIB)

# Search with creator filter
python -m act.front_end --search mnist --creator torchvision
python -m act.front_end --search cifar --creator vnnlib

# ============================================================================
# INFO - Get detailed information about datasets/categories
# ============================================================================

# Auto-detect and show info (TorchVision datasets)
python -m act.front_end --info MNIST           # Dataset info + recommended models
python -m act.front_end --info CIFAR10         # Shows: resnet18, mobilenet_v2, etc.
python -m act.front_end --info ImageNet        # Large-scale dataset info
python -m act.front_end --info FashionMNIST    # Fashion items dataset

# Auto-detect and show info (VNNLIB categories)
python -m act.front_end --info acasxu_2023     # ACAS Xu collision avoidance
python -m act.front_end --info vit_2023        # Vision Transformer verification
python -m act.front_end --info yolo_2023       # YOLO object detection
python -m act.front_end --info cifar100_2024   # CIFAR100 VNNLIB benchmark

# Explicit creator override (when name could be ambiguous)
python -m act.front_end --info MNIST --creator torchvision
python -m act.front_end --info cifar100_2024 --creator vnnlib

# ============================================================================
# DOWNLOAD - Download datasets, models, and benchmarks
# ============================================================================

# Auto-detect and download (TorchVision - downloads dataset + ALL recommended models)
python -m act.front_end --download MNIST              # → MNIST + simple_cnn, lenet5, resnet18, etc.
python -m act.front_end --download CIFAR10            # → CIFAR10 + resnet18, mobilenet_v2, etc.
python -m act.front_end --download FashionMNIST       # → FashionMNIST + models
python -m act.front_end --download ImageNet           # → ImageNet (large!)

# Auto-detect and download (VNNLIB - downloads ONNX models + VNNLIB properties)
python -m act.front_end --download acasxu_2023        # → 45 ONNX models + properties
python -m act.front_end --download vit_2023           # → Vision Transformer benchmarks
python -m act.front_end --download yolo_2023          # → YOLO verification benchmarks
python -m act.front_end --download cifar100_2024      # → CIFAR100 VNNLIB benchmarks

# Force specific creator (if name could match multiple)
python -m act.front_end --download MNIST --creator torchvision
python -m act.front_end --download cifar100_2024 --creator vnnlib

# ============================================================================
# DOWNLOAD MANAGEMENT - Track what's been downloaded
# ============================================================================

# List all downloaded items (grouped by creator)
python -m act.front_end --list-downloads

# List only TorchVision downloads
python -m act.front_end --list-downloads --creator torchvision

# List only VNNLIB downloads
python -m act.front_end --list-downloads --creator vnnlib

# ============================================================================
# MODEL SYNTHESIS & INFERENCE - Creator-specific workflows
# ============================================================================

# Run model synthesis (defaults to TorchVision)
python -m act.front_end --synthesis

# Run synthesis for specific creator
python -m act.front_end --synthesis --creator torchvision   # PyTorch models with specs
python -m act.front_end --synthesis --creator vnnlib        # ONNX models with VNNLIB specs

# Run inference on synthesized models (defaults to TorchVision)
python -m act.front_end --inference

# Run inference for specific creator
python -m act.front_end --inference --creator torchvision   # Validates PyTorch models
python -m act.front_end --inference --creator vnnlib        # Validates ONNX→PyTorch models

# ============================================================================
# ADVANCED WORKFLOWS
# ============================================================================

# Download multiple categories sequentially
python -m act.front_end --download MNIST && \
python -m act.front_end --download CIFAR10 && \
python -m act.front_end --download acasxu_2023

# Search and download pipeline
python -m act.front_end --search acas          # Find available ACAS benchmarks
python -m act.front_end --info acasxu_2023     # Check details
python -m act.front_end --download acasxu_2023 # Download it

# Complete TorchVision workflow
python -m act.front_end --download MNIST       # Download dataset + models
python -m act.front_end --synthesis            # Generate wrapped models
python -m act.front_end --inference            # Validate correctness

# Check what's available vs what's downloaded
python -m act.front_end --list                 # Show all available
python -m act.front_end --list-downloads       # Show what's downloaded
```

## Spec Creators Overview

| Creator | Data Source | Models | Specs | Documentation |
|---------|-------------|--------|-------|---------------|
| **TorchVision** | 40 PyTorch datasets | 63 models | ε-perturbations | [torchvision_loader/README.md](torchvision_loader/README.md) |
| **VNNLIB** | 34 VNN-COMP categories | ONNX models | VNNLIB files | [vnnlib_loader/README.md](vnnlib_loader/README.md) |
| **BERT** | SST/Yelp text | Compact checkpoint | Selected WordPiece perturbation | [Direct queries](#direct-instances-and-compact-bert-queries) |

All creators implement `BaseSpecCreator` and generate:
```python
List[Tuple[data_source, model_name, pytorch_model, input_tensors, spec_pairs]]
```

## Unified CLI Features

### Auto-Detection
The CLI automatically determines whether a name refers to:
- **TorchVision dataset** (e.g., MNIST, CIFAR10, ImageNet)
- **VNNLIB category** (e.g., acasxu_2023, cifar100_2024, vggnet16_2022)

### Smart Downloads
```bash
# TorchVision: Downloads dataset + ALL recommended models
python -m act.front_end --download MNIST
# ✓ Downloads: MNIST dataset + simple_cnn, lenet5, resnet18, efficientnet_b0

# VNNLIB: Downloads category with ONNX + VNNLIB files
python -m act.front_end --download acasxu_2023
# ✓ Downloads: 45 ONNX models + 100s of VNNLIB properties
```

### Explicit Creator Override
```bash
# Force specific creator (if ambiguous or needed)
python -m act.front_end --download mnist --creator vnnlib
python -m act.front_end --list --creator torchvision
```

## Domain-Specific CLIs

### TorchVision CLI (`torchvision_loader/cli.py`)
```bash
# TorchVision-specific features
python -m act.front_end.torchvision_loader --models-for CIFAR10
python -m act.front_end.torchvision_loader --datasets-for resnet18
python -m act.front_end.torchvision_loader --validate MNIST resnet18
python -m act.front_end.torchvision_loader --preprocessing-summary

# Download specific dataset-model pair (not all models)
python -m act.front_end.torchvision_loader --download MNIST simple_cnn
```

### VNNLIB CLI (`vnnlib_loader/cli.py`)
```bash
# VNNLIB-specific features
python -m act.front_end.vnnlib_loader --list
python -m act.front_end.vnnlib_loader --info acasxu_2023
python -m act.front_end.vnnlib_loader --download cifar100_2024 --max-instances 10
```

## Programmatic Usage

### TorchVision Creator
```python
from act.front_end.torchvision_loader.create_specs import TorchVisionSpecCreator

creator = TorchVisionSpecCreator()
results = creator.create_specs_for_data_model_pairs(
    datasets=["MNIST", "CIFAR10"],
    models=["simple_cnn", "resnet18"],
    num_samples=3,
    spec_type="local_lp",
    epsilon=0.03,
    p_norm=float("inf")
)
```

### VNNLIB Creator
```python
from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator

creator = VNNLibSpecCreator()
results = creator.create_specs_for_data_model_pairs(
    categories=["acasxu_2023", "cifar100_2024"],
    max_instances=10
)
```

### Creator Registry (Auto-Detection)
```python
from act.front_end.creator_registry import detect_creator, get_creator

# Auto-detect
creator_name, normalized = detect_creator("MNIST")
# Returns: ('torchvision', 'MNIST')

# Get creator instance
creator = get_creator('torchvision')  # or 'vnnlib'
```

## Architecture

```
front_end/
├── __main__.py                  # 🆕 Entry point: python -m act.front_end
├── creator_registry.py          # 🆕 Factory + auto-detection
├── spec_creator_base.py         # Base interface
├── specs.py                     # InputSpec/OutputSpec
├── verifiable_model.py          # Wrapper layers live here
├── model_synthesis.py           # Wrap models with specs
│
├── torchvision_loader/          # TorchVision Creator
│   ├── __main__.py              # Entry point: python -m act.front_end.torchvision_loader
│   ├── README.md
│   ├── cli.py                   # Domain-specific CLI
│   ├── create_specs.py
│   ├── data_model_mapping.py    # 40 datasets, 63 models
│   └── data_model_loader.py
│
└── vnnlib_loader/               # VNNLIB Creator  
    ├── __main__.py              # 🆕 Entry point: python -m act.front_end.vnnlib_loader
    ├── README.md                # 🆕
    ├── cli.py                   # 🆕 Domain-specific CLI
    ├── create_specs.py
    ├── category_mapping.py      # 🆕 34 VNN-COMP categories
    ├── data_model_loader.py
    ├── vnnlib_parser.py
    └── onnx_converter.py
```

## Integration with ACT Pipeline

1. **Spec Creation** → 2. **Model Synthesis** → 3. **Torch→ACT** → 4. **Verification**

```python
# 1. Create specs (either creator)
from act.front_end.torchvision_loader.create_specs import TorchVisionSpecCreator
creator = TorchVisionSpecCreator()
results = creator.create_specs_for_data_model_pairs(...)

# 2. Synthesize wrapped models
from act.front_end.model_synthesis import synthesize_models_from_specs
wrapped_models = synthesize_models_from_specs(results)

# 3. Convert to ACT Net
from act.pipeline.verification.torch2act import TorchToACT
# wrapped_model is one of the values in wrapped_models dict
net = TorchToACT(wrapped_model).run()

# 4. Verify
from act.back_end.verifier import verify_once
result = verify_once(net, solver=solver)
```

## See Also

- **TorchVision**: [torchvision_loader/README.md](torchvision_loader/README.md) - 40 datasets, 63 models
- **VNNLIB**: [vnnlib_loader/README.md](vnnlib_loader/README.md) - 34 VNN-COMP categories
- **Data**: [../data/torchvision/README.md](../../data/torchvision/README.md), [../data/vnnlib/README.md](../../data/vnnlib/README.md)
- **Pipeline**: [../pipeline/README.md](../pipeline/README.md)

---

# 🧩 Spec-Free, Input-Free Torch→ACT + Verification — Two-File Design

This document specifies a compact, production-ready pattern to convert **wrapped PyTorch models** to ACT
and run verification with **no external inputs or specs** passed at runtime.

**Exactly two files to implement**:
- `../pipeline/torch2act.py` — Torch→ACT converter (reads embedded specs from wrapper; no input_shape needed)
- `verifier.py` — Single-shot (**verify_once**) and Branch-and-Bound (**verify_bab**) verification using only the ACT `Net`

> All input/output specifications are embedded in the wrapper via `InputSpecLayer` and `OutputSpecLayer`.
> The converter and verifier read those layers directly; **no external spec or input tensors** are required.

---

## ✅ Wrapper Contract

Your wrapped model is an `nn.Module` with **four named children**:

```
input_layer (InputLayer) → input_spec (InputSpecLayer) → model (any nn.Module) → output_spec (OutputSpecLayer)
```

- `InputLayer(shape=(1,...), center=?)` — declares the input variable block (symbolic).
- `InputSpecLayer(spec=InputSpec(...))` — input constraints (BOX, LINF_BALL as BOX, or LIN_POLY) directly on input space.
- `[optional nn.Flatten]` — reshaping only.
- `Model` — learned layers (e.g., `nn.Linear`, `nn.ReLU`).
- `OutputSpecLayer(spec=OutputSpec(...))` — final property (`ASSERT`) over outputs.

**Note**: Preprocessing (normalization, channel conversion, etc.) should be handled by data loader (e.g., `torchvision.transforms.Compose`) before wrapping the model.

---

## 📦 File 1: `../pipeline/torch2act.py` (Converter)

### Responsibilities
- Convert the wrapper into an ACT `Net` of `Layer` objects.
- Put **numeric tensors** (weights, bounds) in `Layer.params` and **JSON-able** flags/shapes in `Layer.meta`.
- Enforce a verification-ready wrapper via **hard assertions**:
  - exactly one `InputLayer` (no `input_shape` arg needed),
  - at least one `InputSpecLayer`,
  - last module is `OutputSpecLayer` (ACT last layer is `ASSERT`).

### Module → ACT Layer Mapping
| Torch module | ACT kind | Notes |
|--------------|----------|-------|
| `InputLayer` | `INPUT` | allocates initial variable block; `params['shape']`[, `center`] |
| `InputSpecLayer` | `INPUT_SPEC` | **constraint-only**, `out_vars == in_vars` |
| `nn.Flatten` | `FLATTEN` | reshape only |
| `nn.Linear` | `DENSE` | `params['W']`, `params['b']` |
| `nn.ReLU` | `RELU` | same-width block |
| `OutputSpecLayer` | `ASSERT` | **constraint-only**, `out_vars == in_vars` |

### Public API
```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch, torch.nn as nn

@dataclass
class Layer:
    id: int
    kind: str
    params: Dict[str, torch.Tensor]
    meta: Dict[str, Any]
    in_vars: List[int]
    out_vars: List[int]
    cache: Dict[str, torch.Tensor] = field(default_factory=dict)
    def is_validation(self) -> bool: return self.kind == "ASSERT"

@dataclass
class Net:
    layers: List[Layer]
    preds: Dict[int, List[int]]
    succs: Dict[int, List[int]]
    by_id: Dict[int, Layer] = field(init=False)
    def __post_init__(self): self.by_id = {L.id: L for L in self.layers}
    def last_validation(self): ...
    def assert_last_is_validation(self): ...

class TorchToACT:
    def __init__(self, wrapped: nn.Module): ...
    def run(self) -> Net: ...
```

---

## 📁 File 2: `verifier.py` (Verification)

This module is **spec-free, input-free**. All constraints are extracted from the ACT `Net`.
The public entry points **do not accept** input shapes, var ids, or external spec objects.

### Public API
```python
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any
import numpy as np
import torch
from act.back_end.solver.solver_base import Solver

class VerifyStatus(Enum):
    CERTIFIED = "certified"           # Property proven safe
    FALSIFIED = "falsified"           # Property violated (counterexample found)
    UNKNOWN = "unknown"               # Inconclusive result
    TIMEOUT = "timeout"               # Time limit exceeded
    VERIFIER_ERROR = "verifier_error" # Verifier error
    MODEL_INFER_FAILURE = "model_infer_failure"  # Model inference failed

@dataclass
class VerifyResult:
    status: VerifyStatus
    counterexample: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@torch.no_grad()
def verify_once(net, solver: Solver, timelimit: Optional[float]=None) -> VerifyResult: ...

def verify_bab(net, solver: Solver,
               max_depth: int=20, max_nodes: int=2000, time_budget_s: float=300.0) -> VerifyResult: ...
```

### How it works
- Extract from `net`:
  - `entry_id` from `INPUT` layer,
  - `input_ids` from `INPUT.out_vars`,
  - `output_ids` from `ASSERT.in_vars`,
  - list of `INPUT_SPEC` layers,
  - final `ASSERT` layer.
- Build **seed box** from `INPUT_SPEC` layers (`BOX` or `LINF_BALL`; `LIN_POLY` alone requires a seed policy → error).
- Call `analyze(net, entry_id, seed)` → `(before, after, globalC)`.
- Add all input specs to `globalC` (`BOX`/`LINF_BALL` as boxes; `LIN_POLY` tagged as inequalities).
- `export_to_solver(globalC, solver, ...)` and `materialise_input_poly(...)` to push linear rows.
- Add **negated** ASSERT to the solver (LINEAR_LE, TOP1_ROBUST, MARGIN_ROBUST, RANGE, UNSAFE_LINEAR policy).
- Set a linear objective (max violation for robust kinds if requested).
- Solve and interpret:
  - `INFEASIBLE` → `CERTIFIED`
  - `FEASIBLE/OPTIMAL` + solution → `COUNTEREXAMPLE`
  - else → `UNKNOWN`

### Branch-and-Bound
- Root node uses the seed box.
- Each node calls the same solve path with `node.box`.
- If SAT → obtain `x_ce` and numerically check against `ASSERT` (TRUE_CE/FALSE_CE).
- If INFEASIBLE → node is certified and pruned.
- Otherwise branch on widest box dimension and continue.

---

## ⚠️ Edge Cases & Policies

- **Multiple INPUT_SPEC layers** supported — all added to constraints. First BOX/LINF_BALL is used as the seed.
- **LIN_POLY-only** inputs: require a seed box or raise `ValueError` (unchanged policy).
- **RANGE** negation is disjunctive; the default encodes a one-sided violation (≥ ub + ε). If needed, add a second pass or branching for the ≤ lb − ε side.
- **Unsupported Torch modules** should raise `NotImplementedError` in `../pipeline/torch2act.py`.

---

## 🧪 Minimal Example

```python
# Convert (spec-free)
from act.pipeline.verification.torch2act import TorchToACT
net = TorchToACT(wrapped).run()
```

---

## ✅ Checklist

- [ ] Implement `../pipeline/torch2act.py` with strong assertions and mapping table.
- [ ] Use `verifier.py` from this repo (spec-free, input-free).
- [ ] Ensure `INPUT_SPEC` and `ASSERT` are constraint-only (`out_vars == in_vars`).
- [ ] Keep numerics in `params` (Torch tensors) and flags/shapes in `meta`.
- [ ] Decide RANGE negation policy (one-sided or both via branching).

---

## 📜 License

Add your project license here.
