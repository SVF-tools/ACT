# Batch-Native Refactoring Plan

> **Base Commit**: `bf42741f93a3fb9e37addeac8c83ec945b8effe5` (Merge PR #29 - finetune)
> **Branch**: `refactor/batch-native-v2`
> **Location**: `/data1/guanqin/newACT/stat/ACT-batch-refactor/`

---

## Overview

This plan transforms ACT from single-sample processing to batch-native processing across all layers:
- **Front-End**: Batch-aware specs, unified SpecLoader API
- **Back-End**: Batched verification, transfer functions, solvers
- **Pipeline**: Batched fuzzing with property checking

---

## Current State (bf42741)

### `act/front_end/specs.py` (51 lines)
```python
# Minimal specs - no batch support
@dataclass
class InputSpec:
    kind: str
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    center: Optional[torch.Tensor] = None
    eps: Optional[float] = None  # scalar only
    A: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None

@dataclass
class OutputSpec:
    kind: str
    c: Optional[torch.Tensor] = None
    d: Optional[float] = None
    y_true: Optional[int] = None  # single int only
    margin: float = 0.0
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict)
```

### Missing Components
- No `SpecLoader` unified API
- No batch properties (`batch_size`, `device`, `dtype`)
- No validation (`__post_init__`)
- No batched fuzzing

---

## PR Structure

```
PR 1: specs.py (foundation)
  │
  ├──▶ PR 2: SpecLoader (depends on PR 1)
  │         │
  │         └──▶ PR 4: Fuzzing (depends on PR 1 + PR 2)
  │
  └──▶ PR 3: Backend (parallel, depends on PR 1)
              │
              └──▶ PR 5: Integration (depends on all)
```

---

## PR 1: Batch-Native Specs

**Branch**: `feat/batch-specs`
**Base**: `bf42741`

### Goal
Make `InputSpec` and `OutputSpec` batch-aware with validation and utility methods.

### Files to Modify

#### `act/front_end/specs.py` (~200 lines, was 51)

**BEFORE** (51 lines):
```python
@dataclass
class InputSpec:
    kind: str
    lb: Optional[torch.Tensor] = None
    # ... no batch support
```

**AFTER** (~200 lines):
```python
@dataclass
class InputSpec:
    """
    Input specification for verification (batched, where B=1 is single sample).
    
    All tensors have shape [B, ...] where B is batch size.
    """
    kind: str
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    center: Optional[torch.Tensor] = None
    eps: Optional[Union[float, torch.Tensor]] = None  # scalar or [B] tensor
    A: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Validate required fields based on kind."""
        if self.kind == InKind.BOX:
            assert self.lb is not None and self.ub is not None, "BOX requires lb, ub"
        elif self.kind == InKind.LINF_BALL:
            assert self.center is not None and self.eps is not None, "LINF_BALL requires center, eps"
        elif self.kind == InKind.LIN_POLY:
            assert self.A is not None and self.b is not None, "LIN_POLY requires A, b"
    
    def _get_tensor(self) -> torch.Tensor:
        """Get the primary tensor for this spec kind (for batch_size, device, dtype)."""
        if self.kind == InKind.BOX:
            return self.lb
        elif self.kind == InKind.LINF_BALL:
            return self.center
        elif self.kind == InKind.LIN_POLY:
            return self.A
        raise ValueError(f"Unknown kind: {self.kind}")
    
    @property
    def batch_size(self) -> int:
        return self._get_tensor().shape[0]
    
    @property
    def device(self) -> torch.device:
        return self._get_tensor().device
    
    @property
    def dtype(self) -> torch.dtype:
        return self._get_tensor().dtype
    
    def get_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (lb, ub) tensors. For LINF_BALL: lb=center-eps, ub=center+eps."""
        if self.kind == InKind.BOX:
            return self.lb, self.ub
        elif self.kind == InKind.LINF_BALL:
            eps = self.eps.view(-1, *([1]*(self.center.dim()-1))) if isinstance(self.eps, torch.Tensor) else self.eps
            return self.center - eps, self.center + eps
        raise NotImplementedError(f"get_bounds not supported for {self.kind}")
    
    def to(self, device: Union[str, torch.device]) -> 'InputSpec':
        """Move spec to device."""
        def mv(t): return t.to(device) if isinstance(t, torch.Tensor) else t
        return InputSpec(self.kind, mv(self.lb), mv(self.ub), mv(self.center), mv(self.eps), mv(self.A), mv(self.b))
    
    def __len__(self) -> int:
        return self.batch_size
    
    def __repr__(self) -> str:
        shape = list(self._get_tensor().shape)
        eps_s = f", eps={self.eps}" if self.eps is not None and not isinstance(self.eps, torch.Tensor) else ""
        return f"InputSpec({self.kind}, B={self.batch_size}, shape={shape}{eps_s})"


@dataclass
class OutputSpec:
    """
    Output specification for verification (batched, where B=1 is single sample).
    """
    kind: str
    c: Optional[torch.Tensor] = None
    d: Optional[Union[float, torch.Tensor]] = None
    y_true: Optional[Union[int, torch.Tensor]] = None  # int or [B] tensor
    margin: Union[float, torch.Tensor] = 0.0
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate required fields based on kind."""
        if self.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            assert self.y_true is not None, f"{self.kind} requires y_true"
        elif self.kind == OutKind.LINEAR_LE:
            assert self.c is not None and self.d is not None, "LINEAR_LE requires c, d"
        elif self.kind == OutKind.RANGE:
            assert self.lb is not None or self.ub is not None, "RANGE requires lb or ub"
    
    def _get_tensor(self) -> torch.Tensor:
        """Get the primary tensor for this spec kind."""
        if self.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            if isinstance(self.y_true, torch.Tensor):
                return self.y_true
            raise ValueError("Cannot get tensor from scalar y_true. Use tensor labels for batched specs.")
        elif self.kind == OutKind.LINEAR_LE:
            return self.c
        elif self.kind == OutKind.RANGE:
            return self.lb if self.lb is not None else self.ub
        raise ValueError(f"Unknown kind: {self.kind}")
    
    @property
    def batch_size(self) -> int:
        """Get batch size. Returns 1 for scalar y_true."""
        if self.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            if isinstance(self.y_true, torch.Tensor):
                return self.y_true.shape[0]
            return 1  # Scalar label = single sample
        return self._get_tensor().shape[0]
    
    @property
    def device(self) -> torch.device:
        """Get device. Returns CPU for scalar y_true."""
        if self.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            if isinstance(self.y_true, torch.Tensor):
                return self.y_true.device
            return torch.device('cpu')
        return self._get_tensor().device
    
    def to(self, device: Union[str, torch.device]) -> 'OutputSpec':
        """Move spec to device."""
        def mv(t): return t.to(device) if isinstance(t, torch.Tensor) else t
        return OutputSpec(self.kind, mv(self.c), mv(self.d), mv(self.y_true), mv(self.margin), mv(self.lb), mv(self.ub), self.meta)
    
    def __len__(self) -> int:
        return self.batch_size
    
    def __repr__(self) -> str:
        if self.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            if isinstance(self.y_true, torch.Tensor):
                return f"OutputSpec({self.kind}, B={self.batch_size}, labels={len(self.y_true.unique())})"
            return f"OutputSpec({self.kind}, y_true={self.y_true})"
        return f"OutputSpec({self.kind}, B={self.batch_size})"
```

#### `act/front_end/__init__.py`

Add new exports:
```python
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind

__all__ = [
    'InputSpec', 'OutputSpec', 'InKind', 'OutKind',
    # ... existing exports
]
```

#### `act/front_end/verifiable_model.py` (~750 lines, was 729)

**Key Changes:**

1. **VerifiableModel.forward()** - Now returns batch tensors:
```python
# OLD: Returns scalar booleans
return {
    'output': x,
    'input_satisfied': True,  # scalar
    'output_satisfied': True,  # scalar
}

# NEW: Returns [B] boolean tensors
return {
    'output': x,
    'input_satisfied': input_sat,    # [B] tensor
    'output_satisfied': output_sat,  # [B] tensor
    'all_satisfied': all_sat,        # [B] tensor (NEW)
    'summary': f"Batch: {n}/{B} satisfied",  # NEW
}
```

2. **VerifiableModel.get_satisfaction_rate()** - NEW method:
```python
def get_satisfaction_rate(self, x: torch.Tensor) -> Dict[str, float]:
    """Returns {'input': %, 'output': %, 'all': %}"""
```

3. **InputLayer.__init__()** - Remove batch=1 restriction:
```python
# OLD (line 226-227):
if shape[0] != 1:
    raise ValueError(f"Verification wrapper assumes batch=1...")

# NEW: No restriction, supports any batch size
# Note: Batch dimension is now flexible - supports batch=1 and batch>1
```

4. **InputSpecLayer.forward()** - Returns [B] tensor:
```python
# OLD: Returns (x, bool, str)
return (x, satisfied, explanation)  # satisfied is scalar

# NEW: Returns (x, Tensor[B], str)
satisfied = ((x >= lb) & (x <= ub)).flatten(1).all(dim=1)  # [B]
return (x, satisfied, explanation)
```

5. **OutputSpecLayer.forward()** - Returns [B] tensor:
```python
# OLD: Returns (y, bool, str)
# NEW: Returns (y, Tensor[B], str)
satisfied = y_flat.argmax(dim=1) == y_true  # [B]
```

#### `act/front_end/model_synthesis.py` (~490 lines, was 376)

**New Functions:**

```python
def _get_spec_batch_size(spec: Union[InputSpec, OutputSpec]) -> int:
    """Safely extract batch size from a spec."""

def _validate_and_cap_batch(input_spec, output_spec, max_batch_size) -> Tuple:
    """Validate batch sizes match and optionally cap."""

def _wrap_model(pytorch_model, input_spec, output_spec, ...) -> Tuple[VerifiableModel, WrapReport]:
    """Core helper: wrap a model with specs."""

def _make_input_spec(images, eps, input_kind) -> InputSpec:
    """Create InputSpec from images and epsilon."""

def synthesize_model(pytorch_model, images, labels, eps, ...) -> Tuple[VerifiableModel, WrapReport]:
    """Synthesize VerifiableModel from batched tensors."""

def synthesize_models_grouped(spec_results, max_batch_size) -> Tuple[Dict, Dict]:
    """Synthesize models from spec creator results."""
```

**WrapReport Updated:**
```python
@dataclass
class WrapReport:
    input_shape: Tuple[int, ...]
    in_spec_kind: str
    out_spec_kind: str
    data_source: str
    model_name: str
    batch_size: int = 1      # NEW
    unique_labels: int = 1   # NEW
```

### Test Commands
```bash
# Verify imports work
python -c "
from act.front_end import InputSpec, OutputSpec, InKind, OutKind
import torch

# Test InputSpec
lb = torch.zeros(4, 1, 28, 28)
ub = torch.ones(4, 1, 28, 28)
spec = InputSpec(kind=InKind.BOX, lb=lb, ub=ub)
print(f'InputSpec: B={spec.batch_size}, device={spec.device}')

# Test OutputSpec
labels = torch.tensor([0, 1, 2, 3])
out_spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=labels)
print(f'OutputSpec: B={out_spec.batch_size}')

# Test validation
try:
    bad_spec = OutputSpec(kind=OutKind.TOP1_ROBUST)  # Missing y_true
except AssertionError as e:
    print(f'Validation works: {e}')
"
```

### Commit Message
```
feat(specs): make InputSpec/OutputSpec batch-native

- Add __post_init__ validation for required fields per kind
- Add _get_tensor() for polymorphic tensor access
- Add batch_size, device, dtype properties
- Add get_bounds() for unified lb/ub access
- Add .to(device) for device transfer
- Support tensor eps and y_true for batched operations
```

---

## PR 2: SpecLoader Unified API

**Branch**: `feat/spec-loader`
**Base**: `feat/batch-specs` (after PR 1 merged)

### Goal
Create a single entry point for all data loading: TorchVision, VNNLib, or raw tensors.

### Files to Create/Modify

#### `act/front_end/spec_loader.py` (NEW - ~300 lines)

```python
"""
SpecLoader - Unified entry point for loading data, models, and creating specs.

Usage:
    # TorchVision (needs eps)
    loader = SpecLoader.from_torchvision("MNIST", 32, eps=0.1, model_name="resnet18")
    
    # VNNLib (no eps - uses lb/ub from file)
    loader = SpecLoader.from_vnnlib("cifar100_2024", 32, onnx_model="...")
    
    # Direct tensors
    loader = SpecLoader.from_tensors(images, labels, eps=0.1, model=model)
    
    # Common interface
    images, labels, model = loader.data
    input_spec, output_spec = loader.get_specs()
    wrapped, report = loader.synthesize()
"""

from __future__ import annotations
from typing import Optional, Union, List, Tuple, Dict, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn

from act.front_end.specs import InKind, OutKind, InputSpec, OutputSpec


@dataclass
class SpecLoader:
    """Unified entry point for loading data, models, and creating specs."""
    
    images: torch.Tensor          # [B, C, H, W]
    labels: torch.Tensor          # [B]
    eps: Union[float, torch.Tensor] = 0.0
    model: Optional[nn.Module] = None
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    # For VNNLib: store bounds directly (no eps needed)
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    
    @property
    def batch_size(self) -> int:
        return self.images.shape[0]
    
    @property
    def B(self) -> int:
        return self.batch_size
    
    @property
    def has_bounds(self) -> bool:
        """True if lb/ub are set (VNNLib source)."""
        return self.lb is not None and self.ub is not None
    
    @property
    def data(self) -> Tuple[torch.Tensor, torch.Tensor, nn.Module]:
        """Quick access: (images, labels, model) tuple."""
        if self.model is None:
            raise ValueError("No model loaded. Use model_name parameter.")
        return self.images, self.labels, self.model
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def from_torchvision(
        cls,
        dataset_name: str,
        num_samples: int = 32,
        eps: float = 0.0,
        model_name: Optional[str] = None,
        split: str = "test",
        verbose: bool = False,
    ) -> 'SpecLoader':
        """Load from TorchVision dataset."""
        from act.front_end.torchvision_loader import load_dataset_model_pair
        
        result = load_dataset_model_pair(
            dataset_name=dataset_name,
            model_name=model_name,
            split=split,
            batch_size=num_samples,
            shuffle=False,
            verbose=verbose,
        )
        
        images, labels = next(iter(result['dataloader']))
        
        return cls(
            images=images,
            labels=labels,
            eps=eps,
            model=result.get('model'),
            source=f"torchvision:{dataset_name}",
            metadata={
                'split': split,
                'model_name': model_name,
                'num_classes': result.get('num_classes', 10),
                'dataset_name': dataset_name,
            },
        )
    
    @classmethod
    def from_vnnlib(
        cls,
        category: str,
        num_samples: int = 32,
        onnx_model: Optional[str] = None,
        verbose: bool = False,
    ) -> 'SpecLoader':
        """Load from VNNLib benchmark (no eps needed - bounds from file)."""
        from act.front_end.vnnlib_loader import load_vnnlib_category
        
        result = load_vnnlib_category(
            category=category,
            num_samples=num_samples,
            onnx_model=onnx_model,
            verbose=verbose,
        )
        
        return cls(
            images=result['images'],
            labels=result['labels'],
            eps=0.0,  # Not used for VNNLib
            model=result['model'],
            source=f"vnnlib:{category}",
            lb=result['lb'],
            ub=result['ub'],
            metadata={
                'category': category,
                'onnx_model': onnx_model,
            },
        )
    
    @classmethod
    def from_tensors(
        cls,
        images: torch.Tensor,
        labels: torch.Tensor,
        eps: float = 0.0,
        model: Optional[nn.Module] = None,
        lb: Optional[torch.Tensor] = None,
        ub: Optional[torch.Tensor] = None,
    ) -> 'SpecLoader':
        """Create from raw tensors."""
        return cls(
            images=images,
            labels=labels,
            eps=eps,
            model=model,
            source="tensors",
            lb=lb,
            ub=ub,
        )
    
    # =========================================================================
    # Spec Creation
    # =========================================================================
    
    def get_specs(
        self,
        input_kind: Optional[str] = None,
        output_kind: str = OutKind.TOP1_ROBUST,
    ) -> Tuple[InputSpec, OutputSpec]:
        """
        Create input/output specs.
        
        Args:
            input_kind: Override input spec kind. Auto-detects if None:
                       - VNNLib (has_bounds=True) -> BOX
                       - TorchVision (eps > 0) -> LINF_BALL
            output_kind: Output spec kind (default: TOP1_ROBUST)
        """
        # Auto-detect input kind
        if input_kind is None:
            if self.has_bounds:
                input_kind = InKind.BOX
            else:
                input_kind = InKind.LINF_BALL
        
        # Create InputSpec
        if input_kind == InKind.BOX:
            if self.has_bounds:
                input_spec = InputSpec(kind=InKind.BOX, lb=self.lb, ub=self.ub)
            else:
                # Create BOX from eps
                input_spec = InputSpec(
                    kind=InKind.BOX,
                    lb=self.images - self.eps,
                    ub=self.images + self.eps,
                )
        elif input_kind == InKind.LINF_BALL:
            input_spec = InputSpec(
                kind=InKind.LINF_BALL,
                center=self.images,
                eps=self.eps,
            )
        else:
            raise ValueError(f"Unsupported input_kind: {input_kind}")
        
        # Create OutputSpec
        output_spec = OutputSpec(kind=output_kind, y_true=self.labels)
        
        return input_spec, output_spec
    
    def synthesize(
        self,
        output_kind: str = OutKind.TOP1_ROBUST,
    ) -> Tuple['VerifiableModel', 'WrapReport']:
        """Create wrapped VerifiableModel ready for verification."""
        from act.front_end.model_synthesis import synthesize_model
        
        input_spec, output_spec = self.get_specs(output_kind=output_kind)
        
        return synthesize_model(
            model=self.model,
            input_spec=input_spec,
            output_spec=output_spec,
            data_source=self.source,
            model_name=self.metadata.get('model_name', 'unknown'),
        )
    
    # =========================================================================
    # Slicing & Batching
    # =========================================================================
    
    def __getitem__(self, idx) -> 'SpecLoader':
        """Slice the loader to get a subset of samples."""
        return SpecLoader(
            images=self.images[idx],
            labels=self.labels[idx],
            eps=self.eps[idx] if isinstance(self.eps, torch.Tensor) else self.eps,
            model=self.model,
            source=self.source,
            metadata=self.metadata,
            lb=self.lb[idx] if self.lb is not None else None,
            ub=self.ub[idx] if self.ub is not None else None,
        )
    
    def __len__(self) -> int:
        return self.batch_size
    
    def __repr__(self) -> str:
        model_str = type(self.model).__name__ if self.model else "None"
        if self.has_bounds:
            return f"SpecLoader({self.source}, B={self.B}, bounds=lb/ub, model={model_str})"
        eps_str = f"{self.eps:.4f}" if isinstance(self.eps, float) else "[B]"
        return f"SpecLoader({self.source}, B={self.B}, eps={eps_str}, model={model_str})"
```

#### `act/front_end/__init__.py`

Update exports:
```python
from act.front_end.spec_loader import SpecLoader
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind

__all__ = [
    'SpecLoader',  # PRIMARY API
    'InputSpec', 'OutputSpec', 'InKind', 'OutKind',
    # ... existing exports
]
```

#### `act/front_end/torchvision_loader/data_model_loader.py`

Add `verbose` parameter to `load_dataset_model_pair()`:
```python
def load_dataset_model_pair(
    dataset_name: str,
    model_name: Optional[str] = None,
    split: str = "test",
    batch_size: int = 32,
    shuffle: bool = False,
    verbose: bool = True,  # ADD THIS
) -> Dict[str, Any]:
    """..."""
    if verbose:
        print(f"Loading {dataset_name}...")
    # ... rest of function
```

#### `act/front_end/vnnlib_loader/data_model_loader.py`

Add `load_vnnlib_category()` function:
```python
def load_vnnlib_category(
    category: str,
    num_samples: int = 32,
    onnx_model: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Load images, labels, model, and bounds from VNNLib category.
    
    Returns:
        dict with keys: images, labels, model, lb, ub
    """
    # Implementation details...
```

### Test Commands
```bash
# Test TorchVision loading
python -c "
from act.front_end import SpecLoader

loader = SpecLoader.from_torchvision('MNIST', 4, eps=0.1, model_name='resnet18', verbose=False)
print(loader)
print(f'has_bounds: {loader.has_bounds}')

images, labels, model = loader.data
print(f'images: {images.shape}, labels: {labels.shape}')

wrapped, report = loader.synthesize()
print(f'Wrapped: {type(wrapped).__name__}, B={report.batch_size}')
"

# Test slicing
python -c "
from act.front_end import SpecLoader

loader = SpecLoader.from_torchvision('MNIST', 8, eps=0.1, verbose=False)
sliced = loader[:4]
print(f'Original: {loader.B}, Sliced: {sliced.B}')
"
```

### Commit Message
```
feat(front-end): add SpecLoader as unified data loading API

- Add SpecLoader.from_torchvision() for PyTorch datasets
- Add SpecLoader.from_vnnlib() for VNN-COMP benchmarks
- Add SpecLoader.from_tensors() for direct tensor input
- Add has_bounds property to detect VNNLib vs TorchVision sources
- Add get_specs() with auto-detection of input kind
- Add synthesize() for one-step VerifiableModel creation
- Add slicing support for batch subsetting
- Add verbose=False option to reduce console noise
```

---

## PR 3: Backend Batch Infrastructure

**Branch**: `feat/batch-backend`
**Base**: `feat/batch-specs` (after PR 1 merged)

### Goal
Add batch-native verification support to backend.

### Files to Create/Modify

#### `act/back_end/batch_utils.py` (NEW)

```python
"""Batch tensor utilities for verification."""

import torch
from typing import Tuple, List

def split_batch(tensor: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
    """Split tensor into chunks of batch_size."""
    return list(tensor.split(batch_size, dim=0))

def merge_results(results: List[torch.Tensor]) -> torch.Tensor:
    """Merge list of result tensors."""
    return torch.cat(results, dim=0)

def batched_bounds_check(
    x: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> torch.Tensor:
    """Check if x is within bounds. Returns [B] bool tensor."""
    in_lb = (x >= lb).all(dim=tuple(range(1, x.dim())))
    in_ub = (x <= ub).all(dim=tuple(range(1, x.dim())))
    return in_lb & in_ub
```

#### `act/back_end/batch_verifier.py` (NEW)

```python
"""Batch-native verification entry points."""

import torch
from typing import List, Optional
from dataclasses import dataclass

from act.back_end.core import Net, Bounds
from act.back_end.verifier import verify_once, VerifyResult, VerifyStatus

@dataclass
class BatchVerifyResult:
    """Results for batch verification."""
    results: List[VerifyResult]
    
    @property
    def statuses(self) -> List[VerifyStatus]:
        return [r.status for r in self.results]
    
    @property
    def certified_count(self) -> int:
        return sum(1 for s in self.statuses if s == VerifyStatus.CERTIFIED)
    
    @property
    def falsified_count(self) -> int:
        return sum(1 for s in self.statuses if s == VerifyStatus.FALSIFIED)

def verify_batch(
    nets: List[Net],
    solver,
    parallel: bool = False,
) -> BatchVerifyResult:
    """Verify a batch of networks."""
    results = []
    for net in nets:
        result = verify_once(net, solver)
        results.append(result)
    return BatchVerifyResult(results=results)
```

#### `act/back_end/solver/solver_torch.py`

Update to handle batched constraints efficiently:
```python
# Add batched LP solving support
def solve_batched(self, problems: List[LPProblem]) -> List[LPResult]:
    """Solve multiple LP problems efficiently."""
    # Batch matrix operations where possible
    pass
```

### Test Commands
```bash
python -c "
from act.back_end.batch_utils import split_batch, batched_bounds_check
import torch

x = torch.randn(8, 3, 32, 32)
chunks = split_batch(x, 4)
print(f'Split 8 into {len(chunks)} chunks of {chunks[0].shape[0]}')

lb = torch.zeros(8, 3, 32, 32)
ub = torch.ones(8, 3, 32, 32)
in_bounds = batched_bounds_check(x.clamp(0, 1), lb, ub)
print(f'In bounds: {in_bounds}')
"
```

### Commit Message
```
feat(back-end): add batch verification infrastructure

- Add batch_utils.py for batched tensor operations
- Add batch_verifier.py with BatchVerifyResult
- Update solver_torch.py for batched LP solving
- Improve GPU memory efficiency with batched processing
```

---

## PR 4: Batched Fuzzing

**Branch**: `feat/batch-fuzzing`
**Base**: `feat/spec-loader` (after PR 2 merged)

### Goal
Add batched fuzzing with property checking for all output spec kinds.

### Files to Create/Modify

#### `act/pipeline/fuzzing/actfuzzer.py`

Add `BatchedACTFuzzer` class:
```python
class BatchedACTFuzzer:
    """GPU-accelerated batched fuzzer."""
    
    def __init__(
        self,
        model: nn.Module,
        input_spec: InputSpec,
        output_spec: OutputSpec,
        batch_size: int = 64,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.batch_size = batch_size
        self.device = device
        self.checker = PropertyChecker(output_spec)
    
    def fuzz(self, num_iterations: int = 100) -> FuzzResult:
        """Run batched fuzzing."""
        lb, ub = self.input_spec.get_bounds()
        
        for i in range(num_iterations):
            # Generate batch of candidates
            candidates = self._generate_batch(lb, ub)
            
            # Forward pass
            outputs = self.model(candidates)
            
            # Check properties
            violations = self.checker.check_batch(outputs)
            
            if violations.any():
                return FuzzResult(
                    found_violation=True,
                    counterexample=candidates[violations][0],
                )
        
        return FuzzResult(found_violation=False)
```

#### `act/pipeline/fuzzing/checker.py`

Add `PropertyChecker` class:
```python
class PropertyChecker:
    """Check output properties for batched outputs."""
    
    def __init__(self, spec: OutputSpec):
        self.spec = spec
    
    def check_batch(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Check if outputs violate the property.
        
        Returns:
            [B] bool tensor, True where property is violated
        """
        if self.spec.kind == OutKind.TOP1_ROBUST:
            return self._check_top1(outputs)
        elif self.spec.kind == OutKind.MARGIN_ROBUST:
            return self._check_margin(outputs)
        elif self.spec.kind == OutKind.LINEAR_LE:
            return self._check_linear(outputs)
        elif self.spec.kind == OutKind.RANGE:
            return self._check_range(outputs)
        raise ValueError(f"Unknown kind: {self.spec.kind}")
    
    def _check_top1(self, outputs: torch.Tensor) -> torch.Tensor:
        """TOP1_ROBUST: predicted class != y_true."""
        preds = outputs.argmax(dim=-1)
        y_true = self.spec.y_true
        if isinstance(y_true, int):
            return preds != y_true
        return preds != y_true
    
    def _check_margin(self, outputs: torch.Tensor) -> torch.Tensor:
        """MARGIN_ROBUST: margin to runner-up < threshold."""
        # ... implementation
    
    def _check_linear(self, outputs: torch.Tensor) -> torch.Tensor:
        """LINEAR_LE: c @ y > d."""
        return (outputs @ self.spec.c.T).squeeze(-1) > self.spec.d
    
    def _check_range(self, outputs: torch.Tensor) -> torch.Tensor:
        """RANGE: outputs outside [lb, ub]."""
        violations = torch.zeros(outputs.shape[0], dtype=torch.bool, device=outputs.device)
        if self.spec.lb is not None:
            violations |= (outputs < self.spec.lb).any(dim=-1)
        if self.spec.ub is not None:
            violations |= (outputs > self.spec.ub).any(dim=-1)
        return violations
```

#### `act/pipeline/fuzzing/test_batched_fuzzing.py` (NEW)

```python
"""Tests for batched fuzzing."""

import pytest
import torch
from act.front_end import OutputSpec, OutKind
from act.pipeline.fuzzing.checker import PropertyChecker

class TestPropertyCheckerTOP1:
    def test_all_correct_predictions(self):
        spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=torch.tensor([0, 1, 2]))
        checker = PropertyChecker(spec)
        
        outputs = torch.tensor([
            [10.0, 1.0, 1.0],  # pred=0, correct
            [1.0, 10.0, 1.0],  # pred=1, correct
            [1.0, 1.0, 10.0],  # pred=2, correct
        ])
        
        violations = checker.check_batch(outputs)
        assert violations.tolist() == [False, False, False]
    
    def test_all_wrong_predictions(self):
        spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=torch.tensor([0, 1, 2]))
        checker = PropertyChecker(spec)
        
        outputs = torch.tensor([
            [1.0, 10.0, 1.0],  # pred=1, wrong (expected 0)
            [10.0, 1.0, 1.0],  # pred=0, wrong (expected 1)
            [1.0, 10.0, 1.0],  # pred=1, wrong (expected 2)
        ])
        
        violations = checker.check_batch(outputs)
        assert violations.tolist() == [True, True, True]

# ... more tests for each OutKind
```

### Test Commands
```bash
python -m pytest act/pipeline/fuzzing/test_batched_fuzzing.py -v
```

### Commit Message
```
feat(fuzzing): add batched fuzzing for improved throughput

- Add BatchedACTFuzzer for GPU-accelerated fuzzing
- Add PropertyChecker for batched output validation
- Support TOP1_ROBUST, MARGIN_ROBUST, LINEAR_LE, RANGE
- Add test_batched_fuzzing.py with comprehensive tests
```

---

## PR 5: Integration & Cleanup

**Branch**: `feat/batch-integration`
**Base**: Main (after PR 1-4 merged)

### Goal
Final integration, cleanup, and documentation updates.

### Files to Modify
- Update all READMEs
- Regenerate example nets (if needed)
- Update notebooks with new API examples
- Remove any deprecated code

### Commit Message
```
chore: integrate batch-native changes and update docs

- Update README documentation with new SpecLoader API
- Update jupyter notebook examples
- Remove deprecated single-sample code paths
```

---

## Execution Checklist

### PR 1: Specs
- [ ] Modify `act/front_end/specs.py`
- [ ] Update `act/front_end/__init__.py` exports
- [ ] Run test commands
- [ ] Create PR, get review, merge

### PR 2: SpecLoader
- [ ] Create `act/front_end/spec_loader.py`
- [ ] Update `act/front_end/__init__.py` exports
- [ ] Add `verbose` param to torchvision loader
- [ ] Add `load_vnnlib_category()` to vnnlib loader
- [ ] Run test commands
- [ ] Create PR, get review, merge

### PR 3: Backend
- [ ] Create `act/back_end/batch_utils.py`
- [ ] Create `act/back_end/batch_verifier.py`
- [ ] Update `act/back_end/solver/solver_torch.py`
- [ ] Run test commands
- [ ] Create PR, get review, merge

### PR 4: Fuzzing
- [ ] Update `act/pipeline/fuzzing/actfuzzer.py`
- [ ] Update `act/pipeline/fuzzing/checker.py`
- [ ] Create `act/pipeline/fuzzing/test_batched_fuzzing.py`
- [ ] Run pytest
- [ ] Create PR, get review, merge

### PR 5: Integration
- [ ] Update READMEs
- [ ] Update notebooks
- [ ] Final testing
- [ ] Create PR, get review, merge

---

## Commands Reference

```bash
# Setup
cd /data1/guanqin/newACT/stat/ACT-batch-refactor
source ~/guanqin/miniconda3/etc/profile.d/conda.sh
conda activate act-py312

# Create PR branch
git checkout -b feat/batch-specs

# After changes, commit and push
git add -A
git commit -m "feat(specs): make InputSpec/OutputSpec batch-native"
git push -u origin feat/batch-specs

# Create PR via GitHub CLI
gh pr create --title "feat(specs): batch-native InputSpec/OutputSpec" --body "..."
```
