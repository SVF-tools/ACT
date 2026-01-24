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
#   - Batch-native: All operations support arbitrary batch sizes
#   - Spec-free: Constraints embedded in model architecture, not external
#   - PyTorch-native: Full nn.Module compatibility for training/inference
#   - Bidirectional: Converts to/from ACT format via torch2act/act2torch
#   - Automatic verification: Returns per-sample constraint satisfaction status
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
#     Returns (x, satisfied[B], explanation) tuple during inference.
#     Converted to INPUT_SPEC layer in ACT.
#
#   OutputSpecLayer:
#     Output constraint checking (SAFETY, TOP1_ROBUST, MARGIN, etc.).
#     Returns (x, satisfied[B], explanation) tuple during inference.
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
#   2. Wrap with VerifiableModel:
#      verifiable = VerifiableModel(*model)
#
#   3. Run with automatic constraint checking:
#      results = verifiable(input_tensor)
#      # Returns: {'output', 'input_satisfied'[B], 'output_satisfied'[B], ...}
#
#   4. Convert to ACT for formal verification:
#      from act.pipeline.torch2act import TorchToACT
#      act_net = TorchToACT(verifiable).run()
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
    Module wrapper that provides batch-native spec-free verification.
    
    Automatically collects constraint checking results from InputSpecLayer
    and OutputSpecLayer, returning a dict with both model output and
    per-sample constraint satisfaction status.
    
    Strict Mode:
        Controlled via VerifiableModel.set_strict_mode(True/False).
        When enabled, raises ValueError if ANY sample violates constraints.
        Default: False (graceful violation reporting).
    
    Args:
        *args: Layers to include in the sequential model
    
    Returns:
        Dict with keys:
        - 'output': Model output tensor [B, ...]
        - 'input_satisfied': Boolean tensor [B] for input constraints
        - 'output_satisfied': Boolean tensor [B] for output constraints
        - 'all_satisfied': Boolean tensor [B] (input AND output)
        - 'input_explanation': Human-readable input constraint summary
        - 'output_explanation': Human-readable output constraint summary
        - 'summary': Batch satisfaction summary string
    
    Raises:
        ValueError: If strict mode enabled and any constraint is violated
    """
    
    # Class-level strict mode setting (shared across all instances)
    _strict_mode: bool = False
    
    def __init__(self, *args):
        super().__init__(*args)
    
    @classmethod
    def set_strict_mode(cls, enabled: bool) -> None:
        """When enabled, forward() raises ValueError on any constraint violation."""
        cls._strict_mode = enabled
    
    @classmethod
    def get_strict_mode(cls) -> bool:
        return cls._strict_mode
    
    def forward(self, x):
        """
        Forward pass with batched constraint checking.
        
        Returns dict with output tensor and per-sample satisfaction status.
        """
        B, device = x.shape[0], x.device
        input_sat = output_sat = None
        input_exp = output_exp = ""
        
        for module in self._modules.values():
            result = module(x)
            
            if isinstance(module, InputSpecLayer):
                x, input_sat, input_exp = result
            elif isinstance(module, OutputSpecLayer):
                x, output_sat, output_exp = result
            else:
                x = result
        
        # Default to all-satisfied if no spec layers
        ones = lambda: torch.ones(B, dtype=torch.bool, device=device)
        if input_sat is None:
            input_sat, input_exp = ones(), "No INPUT_SPEC layer"
        if output_sat is None:
            output_sat, output_exp = ones(), "No OUTPUT_SPEC layer"
        
        all_sat = input_sat & output_sat
        
        if self._strict_mode and not all_sat.all():
            msg = input_exp if not input_sat.all() else output_exp
            raise ValueError(f"Constraint violated in strict mode: {msg}")
        
        return {
            'output': x,
            'input_satisfied': input_sat,
            'output_satisfied': output_sat,
            'all_satisfied': all_sat,
            'input_explanation': input_exp,
            'output_explanation': output_exp,
            'summary': f"Batch: {all_sat.sum().item()}/{B} satisfied",
        }
    
    def get_satisfaction_rate(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Compute satisfaction rates for input, output, and combined constraints.
        
        Args:
            x: Input tensor of shape [B, ...] where B is batch size
            
        Returns:
            Dict with percentage satisfaction rates:
            - 'input': Percentage of samples satisfying input constraints
            - 'output': Percentage of samples satisfying output constraints
            - 'all': Percentage of samples satisfying both constraints
        """
        with torch.no_grad():
            result = self.forward(x)
        
        B = x.shape[0]
        input_rate = 100.0 * result['input_satisfied'].sum().item() / B
        output_rate = 100.0 * result['output_satisfied'].sum().item() / B
        all_rate = 100.0 * result['all_satisfied'].sum().item() / B
        
        return {
            'input': input_rate,
            'output': output_rate,
            'all': all_rate,
        }


class InputLayer(nn.Module):
    """
    Declares the symbolic input block with rich metadata. No-op at inference.
    
    Stores complete labeled input sample (tensor + label) for self-contained models.
    Supports comprehensive metadata tracking for verification including data type,
    layout, dataset information.
    
    NOTE: dtype is REQUIRED for verification soundness (different dtypes have
    different precision/range affecting bound propagation).
    
    Batch Support:
        Batch dimension is flexible - supports both batch=1 (formal verification)
        and batch>1 (batched inference/fuzzing). Shape[0] is the batch size.
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
        # Note: Batch dimension is now flexible - supports both batch=1 (formal verification)
        # and batch>1 (batched inference/fuzzing). Shape[0] is the batch size.
        
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
        # INPUT layer creates new variables for each element in input
        n_vars = prod(self.shape[1:])  # Exclude batch dimension
        out_vars = list(range(layer_id_start, layer_id_start + n_vars))
        
        # Collect params
        params = {
            "labeled_input": self.labeled_input,
        }
        
        # Collect meta (everything with values)
        meta = {
            "shape": self.shape,
            "dtype": str(self.dtype),  # Always present (required)
        }
        if self.desc != "input":
            meta["desc"] = self.desc
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
            in_vars=[],  # INPUT has no predecessors
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
    
    Batch-native: Returns per-sample satisfaction as [B] boolean tensor.
    
    Args:
        spec: InputSpec object with constraint information
    
    Returns:
        Tuple of (tensor, satisfied[B], explanation) for use with VerifiableModel.
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Forward pass with batched constraint checking.
        
        Args:
            x: Input tensor of shape [B, ...] where B is batch size
        
        Returns:
            Tuple of (tensor, satisfied, explanation) where:
            - tensor: Pass-through input tensor [B, ...]
            - satisfied: Boolean tensor [B] indicating per-sample satisfaction
            - explanation: Human-readable summary string
        """
        B = x.shape[0]
        device = x.device
        
        # If no spec, pass through without checking
        if self.spec is None:
            return (x, torch.ones(B, dtype=torch.bool, device=device), "INPUT: No constraints")
        
        # Check constraints based on kind
        if self.kind == InKind.BOX:
            # Box constraint: lb <= x <= ub
            if self.lb is None or self.ub is None:
                return (x, torch.ones(B, dtype=torch.bool, device=device), "INPUT BOX: Missing lb/ub")
            
            # Expand bounds to match input shape [B, ...]
            lb = self.lb.expand_as(x)
            ub = self.ub.expand_as(x)
            
            # Per-sample satisfaction: all elements must satisfy bounds
            satisfied = ((x >= lb) & (x <= ub)).flatten(1).all(dim=1)  # [B]
            n = satisfied.sum().item()
            explanation = f"BOX: {n}/{B} satisfied"
            return (x, satisfied, explanation)
        
        elif self.kind == InKind.LINF_BALL:
            # L∞-ball constraint: ||x - center||∞ <= eps
            if self.center is None or self.eps is None:
                return (x, torch.ones(B, dtype=torch.bool, device=device), "INPUT LINF: Missing center/eps")
            
            # Expand center to match input shape
            center = self.center.expand_as(x)
            
            # Per-sample L∞ distance
            linf_dist = (x - center).abs().flatten(1).max(dim=1).values  # [B]
            satisfied = linf_dist <= self.eps  # [B]
            n = satisfied.sum().item()
            max_dist = linf_dist.max().item()
            explanation = f"LINF: {n}/{B} satisfied, max_dist={max_dist:.4f}"
            return (x, satisfied, explanation)
        
        elif self.kind == InKind.LIN_POLY:
            # Linear polytope: Ax <= b
            if self.A is None or self.b is None:
                return (x, torch.ones(B, dtype=torch.bool, device=device), "INPUT LIN_POLY: Missing A/b")
            
            x_flat = x.flatten(1)  # [B, D]
            
            # Handle both batched A [B, M, D] and single A [M, D]
            if self.A.dim() == 2:
                # Single A matrix: broadcast across batch
                residuals = x_flat @ self.A.T - self.b  # [B, M]
            else:
                # Batched A matrices
                residuals = torch.bmm(self.A, x_flat.unsqueeze(-1)).squeeze(-1) - self.b  # [B, M]
            
            satisfied = residuals.max(dim=1).values <= 0  # [B]
            n = satisfied.sum().item()
            explanation = f"LIN_POLY: {n}/{B} satisfied"
            return (x, satisfied, explanation)
        
        else:
            return (x, torch.ones(B, dtype=torch.bool, device=device), f"INPUT: Unknown kind {self.kind}")


class OutputSpecLayer(nn.Module):
    """
    Wraps ACT's OutputSpec AND is an nn.Module. Returns constraint checking tuple.
    
    Batch-native: Returns per-sample satisfaction as [B] boolean tensor.
    
    Args:
        spec: OutputSpec object with constraint information
    
    Returns:
        Tuple of (tensor, satisfied[B], explanation) for use with VerifiableModel.
    """
    def __init__(self, spec: Optional[OutputSpec] = None, **kwargs):
        super().__init__()
        self.spec = spec or OutputSpec(**kwargs)
        self.kind = self.spec.kind
        self.y_true = self.spec.y_true
        self.margin = float(self.spec.margin) if self.spec.margin is not None else 0.0
        self.d = None if self.spec.d is None else float(self.spec.d)
        self.meta = dict(self.spec.meta)

        for name in ("c", "lb", "ub"):
            val = getattr(self.spec, name, None)
            if isinstance(val, torch.Tensor):
                self.register_buffer(name, val)
            else:
                setattr(self, name, None)
        
        # Register y_true as buffer if it's a tensor
        if isinstance(self.y_true, torch.Tensor):
            self.register_buffer("_y_true_tensor", self.y_true)
        
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

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Forward pass with batched constraint checking.
        
        Args:
            y: Output tensor of shape [B, ...] where B is batch size
        
        Returns:
            Tuple of (tensor, satisfied, explanation) where:
            - tensor: Pass-through output tensor [B, ...]
            - satisfied: Boolean tensor [B] indicating per-sample satisfaction
            - explanation: Human-readable summary string
        """
        B = y.shape[0]
        device = y.device
        y_flat = y.view(B, -1)  # [B, C]
        
        # If no spec, pass through without checking
        if self.spec is None:
            return (y, torch.ones(B, dtype=torch.bool, device=device), "OUTPUT: No constraints")
        
        # Check constraints based on kind
        if self.kind == OutKind.TOP1_ROBUST:
            # Top-1 robustness: y_true class has highest score
            if self.y_true is None:
                return (y, torch.ones(B, dtype=torch.bool, device=device), "OUTPUT TOP1: Missing y_true")
            
            # Handle y_true as int (single label) or Tensor[B] (batched labels)
            if isinstance(self.y_true, torch.Tensor):
                y_true = self.y_true.to(device)
            else:
                y_true = torch.tensor([self.y_true], device=device, dtype=torch.long).expand(B)
            
            satisfied = y_flat.argmax(dim=1) == y_true  # [B]
            n = satisfied.sum().item()
            explanation = f"TOP1: {n}/{B} correct"
            return (y, satisfied, explanation)
        
        elif self.kind == OutKind.MARGIN_ROBUST:
            # Margin robustness: y_true class score exceeds others by margin
            if self.y_true is None:
                return (y, torch.ones(B, dtype=torch.bool, device=device), "OUTPUT MARGIN: Missing y_true")
            
            # Handle y_true as int or Tensor[B]
            if isinstance(self.y_true, torch.Tensor):
                y_true = self.y_true.to(device)
            else:
                y_true = torch.tensor([self.y_true], device=device, dtype=torch.long).expand(B)
            
            C = y_flat.shape[1]
            # Create mask for "other" classes
            mask = torch.ones(B, C, dtype=torch.bool, device=device)
            mask.scatter_(1, y_true.unsqueeze(1), False)
            
            # Get scores
            y_true_scores = y_flat.gather(1, y_true.unsqueeze(1)).squeeze(1)  # [B]
            max_other = y_flat.masked_fill(~mask, float('-inf')).max(dim=1).values  # [B]
            margins = y_true_scores - max_other  # [B]
            
            # Handle margin as float or Tensor
            margin_threshold = self.margin if not isinstance(self.margin, torch.Tensor) else self.margin
            satisfied = margins >= margin_threshold  # [B]
            n = satisfied.sum().item()
            min_margin = margins.min().item()
            explanation = f"MARGIN: {n}/{B} satisfied, min={min_margin:.4f}"
            return (y, satisfied, explanation)
        
        elif self.kind == OutKind.LINEAR_LE:
            # Linear inequality: c^T y <= d
            if self.c is None or self.d is None:
                return (y, torch.ones(B, dtype=torch.bool, device=device), "OUTPUT LINEAR_LE: Missing c/d")
            
            c = self.c.to(dtype=y_flat.dtype, device=device)
            
            # Handle c as [C] (single) or [B, C] (batched)
            if c.dim() == 1:
                lhs = (y_flat * c).sum(dim=1)  # [B]
            else:
                lhs = (y_flat * c).sum(dim=1)  # [B]
            
            # Handle d as float or Tensor
            d = self.d if not isinstance(self.d, torch.Tensor) else self.d
            satisfied = lhs <= d  # [B]
            n = satisfied.sum().item()
            explanation = f"LINEAR_LE: {n}/{B} satisfied"
            return (y, satisfied, explanation)
        
        elif self.kind == OutKind.RANGE:
            # Range constraint: lb <= y <= ub
            satisfied = torch.ones(B, dtype=torch.bool, device=device)
            
            if self.lb is not None:
                lb = self.lb.expand_as(y_flat) if self.lb.dim() > 0 else self.lb
                satisfied &= (y_flat >= lb).all(dim=1)
            if self.ub is not None:
                ub = self.ub.expand_as(y_flat) if self.ub.dim() > 0 else self.ub
                satisfied &= (y_flat <= ub).all(dim=1)
            
            n = satisfied.sum().item()
            explanation = f"RANGE: {n}/{B} satisfied"
            return (y, satisfied, explanation)
        
        else:
            return (y, torch.ones(B, dtype=torch.bool, device=device), f"OUTPUT: Unknown kind {self.kind}")
