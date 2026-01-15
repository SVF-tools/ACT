#===- act/back_end/dual_tf/dual_tf.py - Dual Transfer Function Class ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   DualTF class implementing Wong & Kolter style backward pass for
#   Lagrangian dual bound computation. Follows the TransferFunction pattern
#   used by IntervalTF.
#
# Algorithm:
#   1. Run forward interval analysis (reuse IntervalTF)
#   2. Initialize v = -c (negated objective)
#   3. Backward through layers: v_{i-1} = backward_i(v_i)
#   4. Accumulate contributions: bound = sum h_i(v)
#
#===---------------------------------------------------------------------===#

import torch
from typing import Dict, Optional, Tuple

from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.layer_schema import LayerKind

from .tf_mlp import (
    dual_relu_backward,
    dual_dense_backward,
    dual_bias_backward,
    dual_scale_backward,
    dual_bn_backward,
    dual_identity_backward,
    dual_conv2d_backward,
)


class DualTF:
    """
    Dual transfer function for Lagrangian bound computation.
    
    Implements Wong & Kolter style backward pass to compute certified
    lower bounds on linear objectives c^T @ output.
    
    Unlike IntervalTF (forward), DualTF operates backward from output to input.
    It requires pre-computed bounds from forward analysis (e.g., IntervalTF).
    
    Usage:
        # 1. Run forward interval analysis first
        from act.back_end.analyze import analyze
        before, after, _ = analyze(net, entry_id, entry_fact)
        bounds_dict = {lid: f.bounds for lid, f in after.items()}
        
        # 2. Compute dual bound
        dual_tf = DualTF()
        lower_bound = dual_tf.compute_bound(net, bounds_dict, c)
    """
    
    # Layer kind to backward function mapping
    _BACKWARD_REGISTRY = {
        LayerKind.DENSE.value: "_backward_dense",
        LayerKind.RELU.value: "_backward_relu",
        LayerKind.CONV2D.value: "_backward_conv2d",
        "BIAS": "_backward_bias",
        "SCALE": "_backward_scale",
        "BN": "_backward_bn",
        "LRELU": "_backward_relu",  # TODO: implement proper leaky ReLU
        
        # Identity-like layers
        LayerKind.INPUT.value: "_backward_identity",
        LayerKind.INPUT_SPEC.value: "_backward_identity",
        LayerKind.ASSERT.value: "_backward_identity",
        "FLATTEN": "_backward_identity",
        "RESHAPE": "_backward_identity",
        "TRANSPOSE": "_backward_identity",
        "SQUEEZE": "_backward_identity",
        "UNSQUEEZE": "_backward_identity",
        
        # Placeholder for future implementation
        "SIGMOID": "_backward_identity",  # TODO: implement tangent relaxation
        "TANH": "_backward_identity",      # TODO: implement tangent relaxation
    }
    
    @property
    def name(self) -> str:
        return "DualTF"
    
    def supports_layer(self, layer_kind: str) -> bool:
        """Check if this dual TF supports the given layer kind."""
        return layer_kind.upper() in self._BACKWARD_REGISTRY
    
    @torch.no_grad()
    def compute_bound(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        c: torch.Tensor,
        return_sce: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute certified lower bound on c^T @ output.
        
        Args:
            net: ACT network
            bounds_dict: Pre-computed bounds from forward analysis
            c: Objective vector [num_outputs]
            return_sce: If True, also return spurious counterexample
            
        Returns:
            If return_sce=False: Scalar lower bound on c^T @ output
            If return_sce=True: Tuple of (lower_bound, sce)
                - sce: spurious counterexample input that minimizes c^T @ output
        """
        # Store bounds for backward pass
        self._bounds_dict = bounds_dict
        
        # Get network layers
        layers = list(net.layers)
        
        # Initialize v with negated objective
        nu = -c.clone()
        
        # Accumulate dual objective
        objective = torch.tensor(0.0, dtype=c.dtype, device=c.device)
        
        # Backward through layers in reverse order
        for layer in reversed(layers):
            k = layer.kind.upper()
            
            # Skip input layers (handled separately)
            if k in [LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value]:
                continue
            
            # Get backward handler
            handler_name = self._BACKWARD_REGISTRY.get(k)
            if handler_name is None:
                # Unknown layer - pass through with warning
                import warnings
                warnings.warn(f"DualTF: Unknown layer kind '{k}', using identity")
                handler_name = "_backward_identity"
            
            handler = getattr(self, handler_name)
            nu, contribution = handler(layer, nu)
            objective = objective + contribution
        
        # Add input contribution and optionally get sce
        input_contrib, sce = self._compute_input_contribution(net, nu, return_sce=True)
        objective = objective + input_contrib
        
        if return_sce:
            return objective, sce
        return objective
    
    @torch.no_grad()
    def compute_robust_bound(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        y_true: int,
        num_classes: int
    ) -> Tuple[torch.Tensor, bool]:
        """
        Compute robust classification bound.
        
        Verifies: output[y_true] > output[j] for all j != y_true
        
        Args:
            net: ACT network
            bounds_dict: Pre-computed bounds
            y_true: True class label
            num_classes: Total number of classes
            
        Returns:
            Tuple of (min_margin, is_certified)
        """
        # Get device/dtype from bounds
        sample_bounds = next(iter(bounds_dict.values()))
        device, dtype = sample_bounds.lb.device, sample_bounds.lb.dtype
        
        margins = []
        for j in range(num_classes):
            if j == y_true:
                continue
            
            # Objective: output[y_true] - output[j]
            c = torch.zeros(num_classes, dtype=dtype, device=device)
            c[y_true] = 1.0
            c[j] = -1.0
            
            margin = self.compute_bound(net, bounds_dict, c)
            margins.append(margin)
        
        margins = torch.stack(margins)
        min_margin = margins.min()
        is_certified = (min_margin > 0).item()
        
        return min_margin, is_certified
    
    # =========================================================================
    # Backward Handlers
    # =========================================================================
    
    def _backward_dense(self, layer: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense layer backward."""
        W = layer.params["W"]
        b = layer.params.get("b", None)
        return dual_dense_backward(nu, W, b)
    
    def _backward_relu(self, layer: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ReLU layer backward."""
        bounds = self._bounds_dict.get(layer.id)
        if bounds is None:
            # Fallback: identity
            return dual_identity_backward(nu)
        return dual_relu_backward(nu, bounds)
    
    def _backward_bias(self, layer: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bias layer backward."""
        c = layer.params["c"]
        return dual_bias_backward(nu, c)
    
    def _backward_scale(self, layer: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale layer backward."""
        a = layer.params["a"]
        return dual_scale_backward(nu, a)
    
    def _backward_bn(self, layer: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """BatchNorm layer backward."""
        A = layer.params["A"]
        c = layer.params["c"]
        return dual_bn_backward(nu, A, c)
    
    def _backward_conv2d(self, layer: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Conv2D layer backward."""
        weight = layer.params["weight"]
        bias = layer.params.get("bias", None)
        stride = layer.meta.get("stride", 1)
        padding = layer.meta.get("padding", 0)
        input_shape = layer.meta.get("input_shape")
        output_shape = layer.meta.get("output_shape")
        
        # Normalize stride/padding to int
        if isinstance(stride, (list, tuple)):
            stride = stride[0]
        if isinstance(padding, (list, tuple)):
            padding = padding[0]
        
        return dual_conv2d_backward(
            nu, weight, bias,
            stride=stride,
            padding=padding,
            input_shape=input_shape,
            output_shape=output_shape
        )
    
    def _backward_identity(self, layer: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Identity-like layer backward."""
        return dual_identity_backward(nu)
    
    # =========================================================================
    # Input Contribution
    # =========================================================================
    
    def _compute_input_contribution(
        self, 
        net: Net, 
        nu: torch.Tensor,
        return_sce: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute input layer's contribution to dual objective.
        
        For box constraints: contribution = lb^T @ [v]- + ub^T @ [v]+
        
        Also generates spurious counterexample (sce) using greedy strategy:
        - When coefficient > 0: pick lower bound (minimizes contribution)
        - When coefficient <= 0: pick upper bound (minimizes contribution)
        
        Args:
            net: Network
            nu: Final coefficient vector at input layer
            return_sce: If True, generate spurious counterexample
            
        Returns:
            Tuple of (contribution, sce) where sce is None if return_sce=False
        """
        # Find input layer
        input_layer = None
        for layer in net.layers:
            if layer.kind == LayerKind.INPUT_SPEC.value:
                input_layer = layer
                break
            elif layer.kind == LayerKind.INPUT.value:
                input_layer = layer
        
        if input_layer is None:
            return torch.tensor(0.0, dtype=nu.dtype, device=nu.device), None
        
        # Get input bounds
        input_bounds = self._bounds_dict.get(input_layer.id)
        if input_bounds is None:
            if "lb" in input_layer.params and "ub" in input_layer.params:
                lb = input_layer.params["lb"]
                ub = input_layer.params["ub"]
            else:
                return torch.tensor(0.0, dtype=nu.dtype, device=nu.device), None
        else:
            lb, ub = input_bounds.lb, input_bounds.ub
        
        # Store original shape for sce
        original_shape = lb.shape
        
        # Flatten and align sizes
        lb_flat = lb.flatten()
        ub_flat = ub.flatten()
        nu_flat = nu.flatten()
        
        min_size = min(len(lb_flat), len(nu_flat))
        lb_flat = lb_flat[:min_size]
        ub_flat = ub_flat[:min_size]
        nu_flat = nu_flat[:min_size]
        
        # Box contribution: lb^T @ [v]- + ub^T @ [v]+
        nu_pos = nu_flat.clamp(min=0)
        nu_neg = nu_flat.clamp(max=0)
        
        contribution = (lb_flat @ nu_neg) + (ub_flat @ nu_pos)
        
        # Generate spurious counterexample (sce)
        # Greedy strategy: pick boundary that minimizes c^T @ x
        # Since we're minimizing, when coeff > 0, pick lb; when coeff <= 0, pick ub
        sce = None
        if return_sce:
            sce = torch.where(nu_flat > 0, lb_flat, ub_flat)
            # Reshape to original input shape
            if sce.numel() == lb.numel():
                sce = sce.view(original_shape)
        
        return contribution, sce


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_dual_bound(
    net: Net,
    bounds_dict: Dict[int, Bounds],
    c: torch.Tensor,
    return_sce: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute certified lower bound on c^T @ output.
    
    Convenience function that creates DualTF instance.
    
    Args:
        net: ACT network
        bounds_dict: Pre-computed bounds from forward analysis (IntervalTF)
        c: Objective vector [num_outputs]
        return_sce: If True, also return spurious counterexample
        
    Returns:
        If return_sce=False: Scalar lower bound
        If return_sce=True: Tuple of (lower_bound, sce)
    """
    dual_tf = DualTF()
    return dual_tf.compute_bound(net, bounds_dict, c, return_sce=return_sce)


def compute_robust_loss_bound(
    net: Net,
    bounds_dict: Dict[int, Bounds],
    y_true: int,
    num_classes: int
) -> Tuple[torch.Tensor, bool]:
    """
    Compute robust classification bound.
    
    Convenience function for classification robustness verification.
    
    Args:
        net: ACT network
        bounds_dict: Pre-computed bounds
        y_true: True class label
        num_classes: Number of classes
        
    Returns:
        Tuple of (min_margin, is_certified)
    """
    dual_tf = DualTF()
    return dual_tf.compute_robust_bound(net, bounds_dict, y_true, num_classes)
