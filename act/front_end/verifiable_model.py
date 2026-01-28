#===- act/front_end/verifiable_model.py - PyTorch Wrapper Layers -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   PyTorch wrapper layers for spec-free verification. Provides nn.Module
#   components that embed specifications directly into models, enabling
#   constraint checking during inference and seamless ACT conversion.
#
# Key Features:
#   - Spec-free: Constraints embedded in model architecture, not external
#   - PyTorch-native: Full nn.Module compatibility for training/inference
#   - Bidirectional: Converts to/from ACT format via torch2act/act2torch
#   - Automatic verification: Returns constraint satisfaction status
#   - Rich metadata: Tracks input shapes, dtypes, devices for verification
#
# Core Wrapper Layers:
#
#   InputLayer:
#     Declares symbolic input with metadata (shape, dtype, device).
#     No-op at inference, converted to INPUT layer in ACT.
#
#   InputSpecLayer:
#     Input constraint checking (BOX, L_INF, LIN_POLY).
#     Returns (x, satisfied, explanation) tuple during inference.
#     Converted to INPUT_SPEC layer in ACT.
#
#   OutputSpecLayer:
#     Output constraint checking (SAFETY, TOP1_ROBUST, MARGIN, etc.).
#     Returns (x, satisfied, explanation) tuple during inference.
#     Converted to ASSERT layer in ACT.
#
# Verification Workflow:
#   1. Build model with wrapper layers:
#      model = nn.Module(
#          InputLayer(shape=(1, 28, 28)),
#          InputSpecLayer(InputSpec(kind=InKind.L_INF, eps=0.03)),
#          nn.Flatten(),
#          nn.Linear(784, 128),
#          nn.ReLU(),
#          nn.Linear(128, 10),
#          OutputSpecLayer(OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=5))
#      )
#
#   2. Wrap with VerifiableModel (from act2torch.py):
#      verifiable = VerifiableModel(*model)
#
#   3. Run with automatic constraint checking:
#      results = verifiable(input_tensor)
#      # Returns: {'output', 'input_satisfied', 'output_satisfied', ...}
#
#   4. Convert to ACT for formal verification:
#      from act.pipeline.torch2act import TorchToACT
#      act_net = TorchToACT(verifiable).run()
#
# Specification Support:
#   Input constraints (InKind):
#     - BOX: Interval bounds [lb, ub]
#     - L_INF: ε-ball around center
#     - LIN_POLY: Linear polyhedron Ax ≤ b
#
#   Output constraints (OutKind):
#     - SAFETY: Linear constraints cx ≤ d
#     - TOP1_ROBUST: Classification robustness (true label stays top-1)
#     - MARGIN: Classification margin > threshold
#     - LOCAL_ROBUST: Local robustness verification
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union

# Import ACT components
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.spec_creator_base import LabeledInputTensor
from act.back_end.layer_schema import LayerKind, REGISTRY
from act.back_end.layer_util import create_layer


def prod(seq: Tuple[int, ...]) -> int:
    """Helper function to compute product of shape dimensions."""
    p = 1
    for s in seq:
        p *= s
    return p


class VerifiableModel(nn.Sequential):
    """
    Module wrapper that provides spec-free verification.
    
    Automatically collects constraint checking results from InputSpecLayer
    and OutputSpecLayer, returning a dict with both model output and
    constraint satisfaction status.
    
    Strict Mode:
        Controlled via VerifiableModel.set_strict_mode(True/False).
        When enabled, raises ValueError on input/output constraint violations.
        Default: False (graceful violation reporting).
    
    Args:
        *args: Layers to include in the sequential model
    
    Returns:
        Dict with keys:
        - 'output': Model output tensor
        - 'input_satisfied': True if input constraints satisfied
        - 'input_explanation': Human-readable input constraint result
        - 'output_satisfied': True if output constraints satisfied
        - 'output_explanation': Human-readable output constraint result
    
    Raises:
        ValueError: If strict mode enabled and input/output constraints are violated
    """
    
    # Class-level strict mode setting (shared across all instances)
    _strict_mode: bool = False
    
    def __init__(self, *args):
        super().__init__(*args)
    
    @classmethod
    def set_strict_mode(cls, enabled: bool) -> None:
        """
        When enabled, forward() raises ValueError on input/output violations.
        """
        cls._strict_mode = enabled
    
    @classmethod
    def get_strict_mode(cls) -> bool:
        return cls._strict_mode
    
    def forward(self, x):
        """
        Forward pass with automatic constraint checking.
        
        Intercepts tuple returns from InputSpecLayer/OutputSpecLayer
        and collects verification results.
        """
        input_satisfied = True
        input_explanation = "No INPUT_SPEC layer"
        output_satisfied = True
        output_explanation = "No OUTPUT_SPEC layer"
        
        # Process through all layers
        for i, module in enumerate(self):
            result = module(x)
            
            # Check if layer returned constraint checking tuple
            if isinstance(result, tuple) and len(result) == 3:
                x, satisfied, explanation = result
                
                # Identify if this is input or output spec layer
                # Input spec layers typically appear early (first few layers)
                # Output spec layers typically appear at the end
                if i < len(self) // 2:  # First half = likely INPUT_SPEC
                    input_satisfied = satisfied
                    input_explanation = explanation
                else:  # Second half = likely OUTPUT_SPEC
                    output_satisfied = satisfied
                    output_explanation = explanation
            else:
                # Regular layer, just pass through
                x = result
        
        # Strict mode: raise on constraint violations
        if self._strict_mode:
            if not input_satisfied:
                print(f"[STRICT MODE] {input_explanation}")
                raise ValueError(
                    f"Input constraint violated in strict mode: {input_explanation}"
                )
            if not output_satisfied:
                print(f"[STRICT MODE] {output_explanation}")
                raise ValueError(
                    f"Output constraint violated in strict mode: {output_explanation}"
                )
        
        # Return comprehensive verification result
        return {
            'output': x,
            'input_satisfied': input_satisfied,
            'input_explanation': input_explanation,
            'output_satisfied': output_satisfied,
            'output_explanation': output_explanation
        }


class InputLayer(nn.Module):
    """
    Declares the symbolic input block with rich metadata. No-op at inference.
    
    Stores complete labeled input sample (tensor + label) for self-contained models.
    Supports comprehensive metadata tracking for verification including data type,
    layout, dataset information.
    
    NOTE: dtype is REQUIRED for verification soundness (different dtypes have
    different precision/range affecting bound propagation).
    """
    def __init__(
        self,
        labeled_input: "LabeledInputTensor",  # Complete input sample with label
        shape: Tuple[int, ...],
        dtype: torch.dtype,  # REQUIRED: Critical for verification soundness
        desc: str = "input",
        # Tier 1: Essential metadata (strongly recommended)
        layout: Optional[str] = None,
        dataset_name: Optional[str] = None,
        # Tier 2: Important metadata
        num_classes: Optional[int] = None,
        value_range: Optional[Tuple[float, float]] = None,
        scale_hint: Optional[str] = None,
        distribution: Optional[str] = None,  # "uniform", "normal", "normalized", "unknown", or custom
        # Tier 3: Optional metadata
        sample_id: Optional[Union[int, str]] = None,
        domain: Optional[str] = None,
        channels: Optional[int] = None,
    ):
        super().__init__()
        if shape[0] != 1:
            raise ValueError(f"Verification wrapper assumes batch=1, got batch size {shape[0]}")
        
        # Core attributes (dtype now required)
        self.shape = tuple(shape)
        self.dtype = dtype  # REQUIRED
        self.desc = desc
        
        # Tier 1: Essential metadata
        self.layout = layout
        self.dataset_name = dataset_name
        
        # Tier 2: Important metadata
        self.num_classes = num_classes
        self.value_range = tuple(value_range) if value_range else None
        self.scale_hint = scale_hint
        self.distribution = distribution
        
        # Tier 3: Optional metadata
        self.sample_id = sample_id
        self.domain = domain
        self.channels = channels
        
        # Store labeled input as PyTorch buffers
        # Assumes device/dtype already initialized by caller (e.g., via initialize_device)
        self.register_buffer("_input_tensor", labeled_input.tensor)
        
        # Store label: convert to tensor if needed
        # Note: Labels use int64 regardless of float dtype initialization
        label_value = labeled_input.label
        if isinstance(label_value, int):
            label_tensor = torch.tensor([label_value])
        elif isinstance(label_value, (list, tuple)):
            label_tensor = torch.tensor(label_value)
        elif isinstance(label_value, torch.Tensor):
            label_tensor = label_value.reshape(-1)
        else:
            raise TypeError(f"Unsupported label type: {type(label_value)}")
        self.register_buffer("_label_tensor", label_tensor)
        
        self._validate_schema()
    
    @property
    def labeled_input(self) -> "LabeledInputTensor":
        """Get the complete labeled input (tensor + label pair)."""
        from act.front_end.spec_creator_base import LabeledInputTensor
        # Return original label format (int if single element, list otherwise)
        label = self._label_tensor.item() if self._label_tensor.numel() == 1 else self._label_tensor.tolist()
        return LabeledInputTensor(tensor=self._input_tensor, label=label)
    
    @property
    def input_tensor(self) -> torch.Tensor:
        """Get input tensor (convenience accessor)."""
        return self._input_tensor
    
    @property
    def label(self) -> Union[int, List[int]]:
        """Get label (convenience accessor)."""
        return self._label_tensor.item() if self._label_tensor.numel() == 1 else self._label_tensor.tolist()

    def _validate_schema(self):
        """Validate parameters against INPUT layer schema"""
        schema = REGISTRY[LayerKind.INPUT.value]
        
        # Collect params (optional: labeled_input as unified object)
        params = {
            "labeled_input": self.labeled_input,
        }
        
        # Collect meta (everything else - dtype now REQUIRED)
        meta = {
            "shape": self.shape,
            "dtype": str(self.dtype)  # REQUIRED - must always be present
        }
        
        # Add non-default desc
        if self.desc != "input":
            meta["desc"] = self.desc
        
        # Add Tier 1-3 metadata (only if not None)
        if self.layout is not None:
            meta["layout"] = self.layout
        if self.dataset_name is not None:
            meta["dataset_name"] = self.dataset_name
        if self.num_classes is not None:
            meta["num_classes"] = self.num_classes
        if self.value_range is not None:
            meta["value_range"] = self.value_range
        if self.scale_hint is not None:
            meta["scale_hint"] = self.scale_hint
        if self.distribution is not None:
            meta["distribution"] = self.distribution
        if self.sample_id is not None:
            meta["sample_id"] = self.sample_id
        if self.domain is not None:
            meta["domain"] = self.domain
        if self.channels is not None:
            meta["channels"] = self.channels
        
        # Check required/optional params and meta
        for key in schema["params_required"]:
            if key not in params:
                raise ValueError(f"InputLayer missing required param: {key}")
        for key in params:
            if key not in schema["params_required"] + schema["params_optional"]:
                raise ValueError(f"InputLayer has unknown param: {key}")
        for key in schema["meta_required"]:
            if key not in meta:
                raise ValueError(f"InputLayer missing required meta: {key}")
        for key in meta:
            if key not in schema["meta_required"] + schema["meta_optional"]:
                raise ValueError(f"InputLayer has unknown meta: {key}")

    def to_act_layers(self, layer_id_start: int, in_vars: List[int]) -> Tuple[List, List[int]]:
        """Convert to ACT Layer(s) and return (layers, out_vars)"""
        N = prod(self.shape[1:])
        out_vars = list(range(len(in_vars), len(in_vars) + N))
        
        # Collect params (optional: labeled_input for ACT serialization)
        params = {
            "labeled_input": self.labeled_input,
        }
        
        # Collect meta (dtype is REQUIRED, always present)
        meta = {
            "shape": self.shape,
            "dtype": str(self.dtype)  # REQUIRED
        }
        
        if self.desc != "input":
            meta["desc"] = self.desc
        
        # Add all optional metadata fields (only if not None)
        if self.layout is not None:
            meta["layout"] = self.layout
        if self.dataset_name is not None:
            meta["dataset_name"] = self.dataset_name
        if self.num_classes is not None:
            meta["num_classes"] = self.num_classes
        if self.value_range is not None:
            meta["value_range"] = self.value_range
        if self.scale_hint is not None:
            meta["scale_hint"] = self.scale_hint
        if self.distribution is not None:
            meta["distribution"] = self.distribution
        if self.sample_id is not None:
            meta["sample_id"] = self.sample_id
        if self.domain is not None:
            meta["domain"] = self.domain
        if self.channels is not None:
            meta["channels"] = self.channels
        
        layer = create_layer(
            id=layer_id_start,
            kind=LayerKind.INPUT.value,
            params=params,
            meta=meta,
            in_vars=in_vars,
            out_vars=out_vars
        )
        return [layer], out_vars

    def get_metadata_summary(self) -> Dict[str, Any]:
        """Return a summary of all metadata for debugging/logging"""
        return {
            "shape": self.shape,
            "desc": self.desc,
            "dtype": str(self.dtype),  # Always present (required)
            "layout": self.layout,
            "dataset_name": self.dataset_name,
            "num_classes": self.num_classes,
            "value_range": self.value_range,
            "scale_hint": self.scale_hint,
            "distribution": self.distribution,
            "label": self.label,
            "sample_id": self.sample_id,
            "domain": self.domain,
            "channels": self.channels,
            "input_tensor_shape": tuple(self.input_tensor.shape),
        }

    def __repr__(self) -> str:
        """Enhanced string representation with key metadata"""
        meta_str = f"shape={self.shape}"
        if self.dataset_name:
            meta_str += f", dataset={self.dataset_name}"
        meta_str += f", label={self.label}"
        if self.layout:
            meta_str += f", layout={self.layout}"
        return f"InputLayer({meta_str})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class InputSpecLayer(nn.Module):
    """
    Wraps ACT's InputSpec AND is an nn.Module. Returns constraint checking tuple.
    
    Args:
        spec: InputSpec object with constraint information
    
    Returns:
        Tuple of (tensor, satisfied, explanation) for use with VerifiableModel.
    """
    def __init__(self, spec: Optional[InputSpec] = None, **kwargs):
        super().__init__()
        self.spec = spec or InputSpec(**kwargs)
        self.kind = self.spec.kind
        self.eps = float(self.spec.eps) if self.spec.eps is not None else None

        # Register tensor fields as buffers so .to(device) works
        for name in ("lb", "ub", "center", "A", "b"):
            val = getattr(self.spec, name, None)
            if isinstance(val, torch.Tensor):
                self.register_buffer(name, val)
            else:
                setattr(self, name, None)
        self._validate_schema()

    def _validate_schema(self):
        """Validate parameters against INPUT_SPEC layer schema"""
        schema = REGISTRY[LayerKind.INPUT_SPEC.value]
        params = {}
        for name in ("lb", "ub", "center", "A", "b"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        meta = {"kind": self.kind}
        if self.eps is not None:
            meta["eps"] = self.eps
        
        # Check schema compliance
        for key in schema["meta_required"]:
            if key not in meta:
                raise ValueError(f"InputSpecLayer missing required meta: {key}")
        for key in meta:
            if key not in schema["meta_required"] + schema["meta_optional"]:
                raise ValueError(f"InputSpecLayer has unknown meta: {key}")

    def to_act_layers(self, layer_id_start: int, in_vars: List[int]) -> Tuple[List, List[int]]:
        """Convert to ACT Layer(s) - INPUT_SPEC doesn't create new vars"""
        params = {}
        for name in ("lb", "ub", "center", "A", "b"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        meta = {"kind": self.kind}
        if self.eps is not None:
            meta["eps"] = self.eps
        
        layer = create_layer(
            id=layer_id_start,
            kind=LayerKind.INPUT_SPEC.value,
            params=params,
            meta=meta,
            in_vars=in_vars,
            out_vars=in_vars  # INPUT_SPEC doesn't change variables
        )
        return [layer], in_vars

    def forward(self, x: torch.Tensor):
        """
        Forward pass with constraint checking.
        
        Returns:
            Tuple of (tensor, satisfied, explanation)
        """
        # If no spec, pass through without checking
        if self.spec is None:
            return (x, True, "✅ INPUT: No constraints")
        
        # Check constraints based on kind
        if self.kind == InKind.BOX:
            # Box constraint: lb <= x <= ub
            if self.lb is None or self.ub is None:
                return (x, True, "⚠️ INPUT BOX: Missing lb/ub")
            
            # Reshape bounds to match input shape (handles flat -> image reshape)
            lb = self.lb.reshape(x.shape)
            ub = self.ub.reshape(x.shape)
            
            lb_satisfied = (x >= lb).all()
            ub_satisfied = (x <= ub).all()
            satisfied = bool(lb_satisfied and ub_satisfied)
            
            if satisfied:
                margin_lb = (x - lb).min().item()
                margin_ub = (ub - x).min().item()
                margin = min(margin_lb, margin_ub)
                explanation = f"✅ INPUT BOX: lb≤x≤ub (margin={margin:.4f})"
            else:
                lb_viol = (x < lb).sum().item()
                ub_viol = (x > ub).sum().item()
                explanation = f"❌ INPUT BOX: {lb_viol} lb violations, {ub_viol} ub violations"
            
            return (x, satisfied, explanation)
        
        elif self.kind == InKind.LINF_BALL:
            # L∞-ball constraint: ||x - center||∞ <= eps
            if self.center is None or self.eps is None:
                return (x, True, "⚠️ INPUT L∞: Missing center/eps")
            
            # Center has batch dimension matching x (both are (1, C, H, W))
            linf_dist = (x - self.center).abs().max().item()
            satisfied = linf_dist <= self.eps
            
            if satisfied:
                explanation = f"✅ INPUT L∞: ||x-c||∞={linf_dist:.4f}≤ε={self.eps:.4f}"
            else:
                explanation = f"❌ INPUT L∞: ||x-c||∞={linf_dist:.4f}>ε={self.eps:.4f}"
            
            return (x, satisfied, explanation)
        
        elif self.kind == InKind.LIN_POLY:
            # Linear polytope: Ax <= b
            if self.A is None or self.b is None:
                return (x, True, "⚠️ INPUT LIN_POLY: Missing A/b")
            
            x_flat = x.reshape(-1)
            residuals = self.A @ x_flat - self.b  # Should be <= 0
            max_violation = residuals.max().item()
            satisfied = max_violation <= 0
            
            if satisfied:
                margin = -max_violation  # How much slack we have
                explanation = f"✅ INPUT LIN_POLY: Ax≤b (margin={margin:.4f})"
            else:
                num_violations = (residuals > 0).sum().item()
                explanation = f"❌ INPUT LIN_POLY: {num_violations} constraints violated (max={max_violation:.4f})"
            
            return (x, satisfied, explanation)
        
        else:
            return (x, True, f"⚠️ INPUT: Unknown kind {self.kind}")


class OutputSpecLayer(nn.Module):
    """
    Wraps ACT's OutputSpec AND is an nn.Module. Returns constraint checking tuple.
    
    Args:
        spec: OutputSpec object with constraint information
    
    Returns:
        Tuple of (tensor, satisfied, explanation) for use with VerifiableModel.
    """
    def __init__(self, spec: Optional[OutputSpec] = None, **kwargs):
        super().__init__()
        self.spec = spec or OutputSpec(**kwargs)
        self.kind = self.spec.kind
        self.y_true = self.spec.y_true
        self.margin = float(self.spec.margin)
        self.d = None if self.spec.d is None else float(self.spec.d)
        self.meta = dict(self.spec.meta)

        for name in ("c", "lb", "ub"):
            val = getattr(self.spec, name, None)
            if isinstance(val, torch.Tensor):
                self.register_buffer(name, val)
            else:
                setattr(self, name, None)
        self._validate_schema()

    def _validate_schema(self):
        """Validate parameters against ASSERT layer schema"""
        schema = REGISTRY[LayerKind.ASSERT.value]
        params = {}
        for name in ("c", "lb", "ub"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        meta = {"kind": self.kind}
        if self.y_true is not None:
            meta["y_true"] = self.y_true
        if self.margin is not None:
            meta["margin"] = self.margin
        if self.d is not None:
            meta["d"] = self.d
        
        # Check schema compliance
        for key in schema["meta_required"]:
            if key not in meta:
                raise ValueError(f"OutputSpecLayer missing required meta: {key}")

    def to_act_layers(self, layer_id_start: int, in_vars: List[int]) -> Tuple[List, List[int]]:
        """Convert to ACT Layer(s) - ASSERT doesn't create new vars"""
        params = {}
        for name in ("c", "lb", "ub"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        meta = {"kind": self.kind}
        if self.y_true is not None:
            meta["y_true"] = self.y_true
        if self.margin is not None:
            meta["margin"] = self.margin
        if self.d is not None:
            meta["d"] = self.d
        
        layer = create_layer(
            id=layer_id_start,
            kind=LayerKind.ASSERT.value,
            params=params,
            meta=meta,
            in_vars=in_vars,
            out_vars=in_vars  # ASSERT doesn't change variables
        )
        return [layer], in_vars

    def forward(self, y: torch.Tensor):
        """
        Forward pass with constraint checking.
        
        Returns:
            Tuple of (tensor, satisfied, explanation)
        """
        # If no spec, pass through without checking
        if self.spec is None:
            return (y, True, "✅ OUTPUT: No constraints")
        
        # Check constraints based on kind
        if self.kind == OutKind.TOP1_ROBUST:
            # Top-1 robustness: y_true class has highest score
            if self.y_true is None:
                return (y, True, "⚠️ OUTPUT TOP1: Missing y_true")
            
            y_flat = y.reshape(-1)
            pred_class = y_flat.argmax().item()
            y_true_score = y_flat[self.y_true].item()
            max_other_score = y_flat[[i for i in range(len(y_flat)) if i != self.y_true]].max().item()
            margin = y_true_score - max_other_score
            
            satisfied = pred_class == self.y_true
            
            if satisfied:
                explanation = f"✅ OUTPUT TOP1: Class {self.y_true} wins (margin={margin:.4f})"
            else:
                explanation = f"❌ OUTPUT TOP1: Class {pred_class} wins, expected {self.y_true} (margin={margin:.4f})"
            
            return (y, satisfied, explanation)
        
        elif self.kind == OutKind.MARGIN_ROBUST:
            # Margin robustness: y_true class score exceeds others by margin
            if self.y_true is None or self.margin is None:
                return (y, True, "⚠️ OUTPUT MARGIN: Missing y_true/margin")
            
            y_flat = y.reshape(-1)
            y_true_score = y_flat[self.y_true].item()
            max_other_score = y_flat[[i for i in range(len(y_flat)) if i != self.y_true]].max().item()
            actual_margin = y_true_score - max_other_score
            
            satisfied = actual_margin >= self.margin
            
            if satisfied:
                explanation = f"✅ OUTPUT MARGIN: margin={actual_margin:.4f}≥{self.margin:.4f}"
            else:
                explanation = f"❌ OUTPUT MARGIN: margin={actual_margin:.4f}<{self.margin:.4f}"
            
            return (y, satisfied, explanation)
        
        elif self.kind == OutKind.LINEAR_LE:
            # Linear inequality: c^T y <= d
            if self.c is None or self.d is None:
                return (y, True, "⚠️ OUTPUT LINEAR_LE: Missing c/d")
            
            y_flat = y.reshape(-1)
            # Ensure dtype consistency for dot product
            c_typed = self.c.to(dtype=y_flat.dtype, device=y_flat.device)
            lhs = (c_typed @ y_flat).item()
            satisfied = lhs <= self.d
            
            if satisfied:
                margin = self.d - lhs
                explanation = f"✅ OUTPUT LINEAR_LE: c^T·y={lhs:.4f}≤d={self.d:.4f} (margin={margin:.4f})"
            else:
                violation = lhs - self.d
                explanation = f"❌ OUTPUT LINEAR_LE: c^T·y={lhs:.4f}>d={self.d:.4f} (violation={violation:.4f})"
            
            return (y, satisfied, explanation)
        
        elif self.kind == OutKind.RANGE:
            # Range constraint: lb <= y <= ub
            if self.lb is None or self.ub is None:
                return (y, True, "⚠️ OUTPUT RANGE: Missing lb/ub")
            
            lb_satisfied = (y >= self.lb).all()
            ub_satisfied = (y <= self.ub).all()
            satisfied = bool(lb_satisfied and ub_satisfied)
            
            if satisfied:
                margin_lb = (y - self.lb).min().item()
                margin_ub = (self.ub - y).min().item()
                margin = min(margin_lb, margin_ub)
                explanation = f"✅ OUTPUT RANGE: lb≤y≤ub (margin={margin:.4f})"
            else:
                lb_viol = (y < self.lb).sum().item()
                ub_viol = (y > self.ub).sum().item()
                explanation = f"❌ OUTPUT RANGE: {lb_viol} lb violations, {ub_viol} ub violations"
            
            return (y, satisfied, explanation)
        
        else:
            return (y, True, f"⚠️ OUTPUT: Unknown kind {self.kind}")