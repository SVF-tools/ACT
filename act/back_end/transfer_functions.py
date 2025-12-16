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

import torch
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
from typing import Dict, List
from act.back_end.core import Bounds, Fact, Layer, Net
from act.util.options import PerformanceOptions


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
        mode: "interval" for IntervalTF, "hybridz" for HybridzTF
    """
    if mode == "interval":
        from act.back_end.interval_tf import IntervalTF
        set_transfer_function(IntervalTF())
    elif mode == "hybridz":
        from act.back_end.hybridz_tf import HybridzTF
        set_transfer_function(HybridzTF())
    else:
        raise ValueError(f"Unknown transfer function mode: {mode}. Use 'interval' or 'hybridz'.")



@torch.no_grad()
def dispatch_tf(L: Layer, before: Dict[int, Fact], after: Dict[int, Fact], net: Net) -> Fact:
    """Dispatch to current transfer function implementation.
    
    This is the main entry point called by analyze() for each layer.
    Optionally logs detailed debug information to file when debug_tf is enabled.
    """
    tf_impl = get_transfer_function()
    input_bounds = before[L.id].bounds
    result = tf_impl.apply(L, input_bounds, net, before, after)
    
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
