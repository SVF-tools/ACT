#===- act/back_end/transfer_functions.py - Transfer Function Interface --===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Transfer Function Interface. Defines the abstract interface for transfer
#   function implementations in the ACT verification framework. Transfer
#   functions compute bounds and constraints.
#
#===---------------------------------------------------------------------===#

"""
for different layer types during the analysis phase.

The interface supports multiple implementations:
- IntervalTF: Interval-based bounds propagation  
- HybridzTF: HybridZ zonotope-based analysis with enhanced precision
"""

#===- act/back_end/transfer_functions.py - Transfer Function Interface --===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Transfer Function Interface. Defines the abstract interface for transfer
#   function implementations in the ACT verification framework. Transfer
#   functions compute bounds and constraints.
#
#===---------------------------------------------------------------------===#

"""
Transfer function dispatch interface used by the backend analysis.

This module defines a small abstract interface for transfer function
implementations and a global registry to select between implementations
(e.g. IntervalTF, HybridzTF).
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from act.back_end.core import Bounds, Fact, Layer, Net
from act.util.options import PerformanceOptions


# ============================================================================
# Shape Validation Utilities
# ============================================================================

# Layer kinds that require spatial shape validation
_CNN_LAYER_KINDS = frozenset({
    'CONV1D', 'CONV2D', 'CONV3D', 'CONVTRANSPOSE2D',
    'MAXPOOL1D', 'MAXPOOL2D', 'MAXPOOL3D',
    'AVGPOOL1D', 'AVGPOOL2D', 'AVGPOOL3D',
    'PAD', 'UPSAMPLE', 'FLATTEN'
})


def _validate_cnn_input_shape(L: Layer, input_bounds: Bounds) -> None:
    """Validate that input bounds match expected shape for CNN layers.

    For CNN layers, the input_shape metadata must be present and the
    flattened bounds length must match the product of input_shape dimensions.

    Shape conventions:
      - 1D: (N, C, W) where N=batch, C=channels, W=width
      - 2D: (N, C, H, W) where H=height, W=width
      - 3D: (N, C, D, H, W) where D=depth

    Raises:
        AssertionError: If shape validation fails
    """
    if L.kind not in _CNN_LAYER_KINDS:
        return

    # Bounds must be 1D (flattened)
    assert input_bounds.lb.dim() == 1, (
        f"Layer {L.id} ({L.kind}): bounds must be 1D (flattened), "
        f"got lb.shape={input_bounds.lb.shape}"
    )

    # Check input_shape metadata exists for CNN layers (except FLATTEN which is flexible)
    if L.kind != 'FLATTEN' and 'input_shape' not in L.meta:
        raise AssertionError(
            f"Layer {L.id} ({L.kind}): missing 'input_shape' in metadata. "
            f"CNN layers require input_shape for shape validation."
        )

    # Validate bounds length matches expected shape
    if 'input_shape' in L.meta:
        expected_shape = tuple(L.meta['input_shape'])
        expected_numel = 1
        for dim in expected_shape:
            expected_numel *= int(dim)

        actual_numel = input_bounds.lb.numel()
        assert actual_numel == expected_numel, (
            f"Layer {L.id} ({L.kind}): bounds size mismatch. "
            f"Expected {expected_numel} elements from input_shape={expected_shape}, "
            f"got {actual_numel} elements."
        )

        # Validate shape format based on layer type
        ndim = len(expected_shape)
        if L.kind in ('CONV1D', 'MAXPOOL1D', 'AVGPOOL1D'):
            assert ndim == 3, (
                f"Layer {L.id} ({L.kind}): expected 3D input_shape (N,C,W), "
                f"got {ndim}D shape {expected_shape}"
            )
        elif L.kind in ('CONV2D', 'CONVTRANSPOSE2D', 'MAXPOOL2D', 'AVGPOOL2D', 'PAD', 'UPSAMPLE'):
            assert ndim == 4, (
                f"Layer {L.id} ({L.kind}): expected 4D input_shape (N,C,H,W), "
                f"got {ndim}D shape {expected_shape}"
            )
        elif L.kind in ('CONV3D', 'MAXPOOL3D', 'AVGPOOL3D'):
            assert ndim == 5, (
                f"Layer {L.id} ({L.kind}): expected 5D input_shape (N,C,D,H,W), "
                f"got {ndim}D shape {expected_shape}"
            )


def _validate_output_shape(L: Layer, output_bounds: Bounds) -> None:
    """Validate that output bounds match expected shape.

    Raises:
        AssertionError: If shape validation fails
    """
    # Bounds must be 1D (flattened)
    assert output_bounds.lb.dim() == 1, (
        f"Layer {L.id} ({L.kind}): output bounds must be 1D (flattened), "
        f"got lb.shape={output_bounds.lb.shape}"
    )

    # Check output matches out_vars count
    assert output_bounds.lb.numel() == len(L.out_vars), (
        f"Layer {L.id} ({L.kind}): output bounds size mismatch. "
        f"Expected {len(L.out_vars)} elements (out_vars), "
        f"got {output_bounds.lb.numel()} elements."
    )

    # Validate output_shape metadata if present
    if 'output_shape' in L.meta:
        expected_shape = tuple(L.meta['output_shape'])
        expected_numel = 1
        for dim in expected_shape:
            expected_numel *= int(dim)

        actual_numel = output_bounds.lb.numel()
        assert actual_numel == expected_numel, (
            f"Layer {L.id} ({L.kind}): output bounds size mismatch. "
            f"Expected {expected_numel} elements from output_shape={expected_shape}, "
            f"got {actual_numel} elements."
        )


class TransferFunction(ABC):
    """Abstract base class for transfer function implementations.
    
    Transfer functions compute output bounds and constraints for network layers
    during the analysis phase. Different implementations provide different
    precision/performance tradeoffs.
    """
    
    @abstractmethod
    def supports_layer(self, layer_kind: str) -> bool:
        """Check if this transfer function implementation supports the given layer kind.
        
        Args:
            layer_kind: Layer type (e.g., "DENSE", "RELU", "CONV2D")
            
        Returns:
            True if this implementation can handle the layer kind
        """
        pass
    
    @abstractmethod
    def apply(self, L: Layer, input_bounds: Bounds, net: Net, 
              before: Dict[int, Fact], after: Dict[int, Fact]) -> Fact:
        """Apply transfer function to compute output bounds and constraints.
        
        Args:
            L: Layer to process
            input_bounds: Input bounds for this layer  
            net: Complete network structure
            before: Pre-processing facts for all layers
            after: Post-processing facts for all layers
            
        Returns:
            Fact containing output bounds and generated constraints
        """
        pass
    
    @property
    @abstractmethod 
    def name(self) -> str:
        """Implementation name for debugging and logging."""
        pass


# Global transfer function management
_current_tf: TransferFunction = None


def set_transfer_function(tf_impl: TransferFunction) -> None:
    """Set the global transfer function implementation."""
    global _current_tf
    _current_tf = tf_impl


def get_transfer_function() -> TransferFunction:
    """Get the current transfer function implementation."""
    if _current_tf is None:
        raise RuntimeError("No transfer function implementation set. Call set_transfer_function() first.")
    return _current_tf


def set_transfer_function_mode(mode: str = "interval") -> None:
    """Set transfer function implementation by mode name.
    
    Args:
        mode: "interval" for IntervalTF, "hybridz" for HybridzTF, "dual" for DualTF
    """
    if mode == "interval":
        from act.back_end.interval_tf import IntervalTF
        set_transfer_function(IntervalTF())
    elif mode == "hybridz":
        from act.back_end.hybridz_tf import HybridzTF
        set_transfer_function(HybridzTF())
    elif mode == "dual":
        from act.back_end.dual_tf import DualTF
        set_transfer_function(DualTF())
    else:
        raise ValueError(f"Unknown transfer function mode: {mode}. Use 'interval', 'hybridz', or 'dual'.")



@torch.no_grad()
def dispatch_tf(L: Layer, before: Dict[int, Fact], after: Dict[int, Fact], net: Net) -> Fact:
    """Dispatch to current transfer function implementation.
    
    This is the main entry point called by analyze() for each layer.
    Performs shape validation before and after transfer function execution.
    Optionally logs detailed debug information to file when debug_tf is enabled.
    """
    tf_impl = get_transfer_function()
    input_bounds = before[L.id].bounds

    # Pre-TF shape validation (Problem 1: standardize shapes with assertions)
    _validate_cnn_input_shape(L, input_bounds)

    result = tf_impl.apply(L, input_bounds, net, before, after)

    # Post-TF shape validation
    _validate_output_shape(L, result.bounds)
    
    # Debug logging to file (GUARDED)
    if PerformanceOptions.debug_tf:
        with open(PerformanceOptions.debug_output_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Layer {L.id} ({L.kind})\n")
            f.write(f"{'='*80}\n")
            
            # Input bounds info (single Bounds object)
            lb_min, lb_max = input_bounds.lb.min().item(), input_bounds.lb.max().item()
            ub_min, ub_max = input_bounds.ub.min().item(), input_bounds.ub.max().item()
            f.write(f"Input bounds: shape={input_bounds.lb.shape}, "
                   f"lb_range=[{lb_min:.4f}, {lb_max:.4f}], "
                   f"ub_range=[{ub_min:.4f}, {ub_max:.4f}]\n")
            
            # Output bounds info (single Bounds object)
            out_bounds = result.bounds
            lb_min, lb_max = out_bounds.lb.min().item(), out_bounds.lb.max().item()
            ub_min, ub_max = out_bounds.ub.min().item(), out_bounds.ub.max().item()
            f.write(f"Output bounds: shape={out_bounds.lb.shape}, "
                   f"lb_range=[{lb_min:.4f}, {lb_max:.4f}], "
                   f"ub_range=[{ub_min:.4f}, {ub_max:.4f}]\n")
            
            # Parameter info
            if L.kind == 'DENSE' and 'W' in L.params:
                W = L.params['W']
                b = L.params['b']
                f.write(f"Parameters: W.shape={W.shape}, b.shape={b.shape}\n")
            elif L.kind == 'CONV2D' and 'weight' in L.params:
                weight = L.params['weight']
                f.write(f"Parameters: weight.shape={weight.shape}\n")
            
            # Constraint info
            cons = result.cons
            f.write(f"Constraints generated: {len(cons)}\n")
            max_to_show = PerformanceOptions.debug_tf_max_constraints
            for i, con in enumerate(list(cons)[:max_to_show]):
                if con.kind == 'LIN_POLY':
                    f.write(f"  Con {i}: LIN_POLY, A.shape={con.A.shape}, b.shape={con.b.shape}, var_ids={con.var_ids}\n")
                else:
                    f.write(f"  Con {i}: {con.kind}, var_ids={con.var_ids}\n")
            if len(cons) > max_to_show:
                f.write(f"  ... and {len(cons) - max_to_show} more constraints\n")
    
    return result
