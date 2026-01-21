#===- act/back_end/dual_tf/dual_tf.py - Dual Transfer Function Class ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   DualTF class implementing Wong & Kolter backward pass for dual bounds.
#   Algorithm: v=-c, backward through layers, accumulate contributions.
#
#===---------------------------------------------------------------------===#


import torch
from typing import Dict, Optional, Tuple
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.layer_schema import LayerKind
from act.back_end.transfer_functions import TransferFunction
from .tf_mlp import (dual_relu_backward, dual_dense_backward, dual_bias_backward,
                     dual_scale_backward, dual_bn_backward, dual_identity_backward)
from .tf_cnn import dual_conv2d_backward
from .tf_smooth import dual_sigmoid_backward, dual_tanh_backward
from .tf_forward import compute_forward_bounds


class DualTF(TransferFunction):
    """Dual TF for Lagrangian bounds. Backward pass: v=-c, propagate, accumulate."""
    
    _BACKWARD_REGISTRY = {
        # Core layers
        LayerKind.DENSE.value: "_backward_dense",
        LayerKind.RELU.value: "_backward_relu",
        LayerKind.CONV2D.value: "_backward_conv2d",
        "BIAS": "_backward_bias", "SCALE": "_backward_scale", "BN": "_backward_bn",
        "LRELU": "_backward_relu",  # TODO: proper leaky ReLU
        # Identity-like
        LayerKind.INPUT.value: "_backward_identity",
        LayerKind.INPUT_SPEC.value: "_backward_identity",
        LayerKind.ASSERT.value: "_backward_identity",
        "FLATTEN": "_backward_identity", "RESHAPE": "_backward_identity",
        "TRANSPOSE": "_backward_identity", "SQUEEZE": "_backward_identity",
        "UNSQUEEZE": "_backward_identity",
        # Smooth activations (S-shaped with tangent relaxation)
        "SIGMOID": "_backward_sigmoid", "TANH": "_backward_tanh",
        # Multi-input operations (residual connections)
        "ADD": "_backward_add",
    }
    
    def __init__(self):
        """Initialize DualTF with empty cache for forward bounds."""
        self._forward_bounds_cache: Dict[int, Bounds] = {}
        self._cache_net_id: Optional[int] = None  # Track which net the cache is for
    
    @property
    def name(self) -> str: return "DualTF"
    
    def supports_layer(self, layer_kind: str) -> bool:
        """Check if this transfer function supports the given layer kind."""
        return layer_kind.upper() in self._BACKWARD_REGISTRY
    
    # -------- TransferFunction Interface (for analyze()) --------
    def apply(self, L: Layer, input_bounds: Bounds, net: Net,
              before: Dict[int, Fact], after: Dict[int, Fact]) -> Fact:
        """
        Apply forward bounds for layer L.
        
        On first call, computes forward bounds for ALL layers using compute_forward_bounds()
        and caches them. Subsequent calls return cached bounds for the requested layer.
        """
        # Check if we need to recompute (new net or empty cache)
        net_id = id(net)
        if self._cache_net_id != net_id or not self._forward_bounds_cache:
            # Find input bounds from entry layer
            input_lb, input_ub = None, None
            for layer in net.layers:
                if layer.kind.upper() in [LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, "INPUT", "INPUT_SPEC"]:
                    if layer.id in before:
                        input_lb = before[layer.id].bounds.lb
                        input_ub = before[layer.id].bounds.ub
                        break
                    elif "lb" in layer.params and "ub" in layer.params:
                        input_lb = layer.params["lb"]
                        input_ub = layer.params["ub"]
                        break
            
            if input_lb is None or input_ub is None:
                # Fallback: use input_bounds directly
                input_lb, input_ub = input_bounds.lb, input_bounds.ub
            
            # Compute all forward bounds at once
            # Use post_activation=True for validation (concrete activations are POST-activation)
            self._forward_bounds_cache = compute_forward_bounds(net, input_lb, input_ub, post_activation=True)
            self._cache_net_id = net_id
        
        # Return cached bounds for this layer
        if L.id in self._forward_bounds_cache:
            bounds = self._forward_bounds_cache[L.id]
            return Fact(bounds=bounds, cons=ConSet())
        else:
            # Fallback for layers not in cache (shouldn't happen normally)
            return Fact(bounds=input_bounds, cons=ConSet())
    
    def clear_cache(self):
        """Clear the forward bounds cache."""
        self._forward_bounds_cache.clear()
        self._cache_net_id = None
    
    @torch.no_grad()
    def compute_bound(self, net: Net, bounds_dict: Dict[int, Bounds], c: torch.Tensor,
                      return_sce: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute certified lower bound on c^T @ output."""
        assert c.dim() == 1, f"c must be 1D, got shape {c.shape}"
        assert len(bounds_dict) > 0, "bounds_dict cannot be empty"
        
        self._bounds_dict = bounds_dict
        nu = c.clone()  # Start with c (not -c) for lower bound computation
        obj = torch.tensor(0.0, dtype=c.dtype, device=c.device)
        
        for layer in reversed(list(net.layers)):
            k = layer.kind.upper()
            if k in [LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value]: continue
            
            handler_name = self._BACKWARD_REGISTRY.get(k)
            if handler_name is None:
                import warnings; warnings.warn(f"DualTF: unknown layer '{k}', using identity")
                handler_name = "_backward_identity"
            
            nu, contrib = getattr(self, handler_name)(layer, nu)
            obj = obj + contrib
        
        input_contrib, sce = self._input_contribution(net, nu, return_sce=True)
        obj = obj + input_contrib
        return (obj, sce) if return_sce else obj
    
    @torch.no_grad()
    def compute_robust_bound(self, net: Net, bounds_dict: Dict[int, Bounds],
                             y_true: int, num_classes: int) -> Tuple[torch.Tensor, bool]:
        """Compute min margin: output[y_true] - output[j] for all j != y_true."""
        sample = next(iter(bounds_dict.values()))
        device, dtype = sample.lb.device, sample.lb.dtype
        
        margins = []
        for j in range(num_classes):
            if j == y_true: continue
            c = torch.zeros(num_classes, dtype=dtype, device=device)
            c[y_true], c[j] = 1.0, -1.0
            margins.append(self.compute_bound(net, bounds_dict, c))
        
        margins = torch.stack(margins)
        return margins.min(), (margins.min() > 0).item()
    
    # -------- Backward Handlers --------
    def _backward_dense(self, L: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return dual_dense_backward(nu, L.params["W"], L.params.get("b"))
    
    def _backward_relu(self, L: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bounds = self._bounds_dict.get(L.id)
        return dual_identity_backward(nu) if bounds is None else dual_relu_backward(nu, bounds)
    
    def _backward_bias(self, L: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return dual_bias_backward(nu, L.params["c"])
    
    def _backward_scale(self, L: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return dual_scale_backward(nu, L.params["a"])
    
    def _backward_bn(self, L: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return dual_bn_backward(nu, L.params["A"], L.params["c"])
    
    def _backward_conv2d(self, L: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stride, padding = L.meta.get("stride", 1), L.meta.get("padding", 0)
        if isinstance(stride, (list, tuple)): stride = stride[0]
        if isinstance(padding, (list, tuple)): padding = padding[0]
        return dual_conv2d_backward(nu, L.params["weight"], L.params.get("bias"),
                                    stride=stride, padding=padding,
                                    input_shape=L.meta.get("input_shape"),
                                    output_shape=L.meta.get("output_shape"))
    
    def _backward_identity(self, L: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return dual_identity_backward(nu)
    
    def _backward_sigmoid(self, L: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bounds = self._bounds_dict.get(L.id)
        return dual_identity_backward(nu) if bounds is None else dual_sigmoid_backward(nu, bounds)
    
    def _backward_tanh(self, L: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bounds = self._bounds_dict.get(L.id)
        return dual_identity_backward(nu) if bounds is None else dual_tanh_backward(nu, bounds)
    
    def _backward_add(self, L: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ADD backward: z = x + y (+ bias)
        
        For z = x + y, the gradient passes through unchanged to both inputs:
        ∂L/∂x = ∂L/∂z * ∂z/∂x = nu * 1 = nu
        ∂L/∂y = ∂L/∂z * ∂z/∂y = nu * 1 = nu
        
        The contribution to the objective comes from the bias term (if present).
        
        Note: In networks with skip connections (ResNet), the proper handling requires
        graph-aware gradient accumulation. This implementation passes nu through as-is,
        which is correct for the ADD operation itself. The sequential traversal in
        compute_bound will need enhancement for full skip connection support.
        """
        contrib = torch.tensor(0.0, dtype=nu.dtype, device=nu.device)
        
        # If ADD has a bias, it contributes to the objective: nu^T @ bias
        if "bias" in L.params and L.params["bias"] is not None:
            b = L.params["bias"].flatten()
            n = min(nu.flatten().numel(), b.numel())
            contrib = (nu.flatten()[:n] * b[:n]).sum()
        
        return nu, contrib
    
    # -------- Input Contribution --------
    def _input_contribution(self, net: Net, nu: torch.Tensor,
                            return_sce: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input contrib: lb^T@[v]+ + ub^T@[v]- (minimization over input box)."""
        # Find input layer
        input_layer = None
        for layer in net.layers:
            if layer.kind == LayerKind.INPUT_SPEC.value:
                input_layer = layer; break
            elif layer.kind == LayerKind.INPUT.value:
                input_layer = layer
        
        if input_layer is None:
            return torch.tensor(0.0, dtype=nu.dtype, device=nu.device), None
        
        # Get bounds
        bounds = self._bounds_dict.get(input_layer.id)
        if bounds is None:
            if "lb" in input_layer.params and "ub" in input_layer.params:
                lb, ub = input_layer.params["lb"], input_layer.params["ub"]
            else:
                return torch.tensor(0.0, dtype=nu.dtype, device=nu.device), None
        else:
            lb, ub = bounds.lb, bounds.ub
        
        orig_shape = lb.shape
        l, u, v = lb.flatten(), ub.flatten(), nu.flatten()
        n = min(l.numel(), v.numel())
        l, u, v = l[:n], u[:n], v[:n]
        
        assert (l <= u).all(), f"Invalid input bounds: l > u at {(l > u).nonzero().flatten().tolist()[:5]}"
        
        # Contribution: minimize v^T @ x s.t. l <= x <= u
        # If v_i > 0: x_i = l_i (use lower bound)
        # If v_i < 0: x_i = u_i (use upper bound)
        contrib = (l @ v.clamp(min=0)) + (u @ v.clamp(max=0))
        
        # SCE: pick lb when v>0, ub otherwise (the minimizing assignment)
        sce = None
        if return_sce:
            sce = torch.where(v > 0, l, u)
            if sce.numel() == lb.numel(): sce = sce.view(orig_shape)
        
        return contrib, sce


# -------- Convenience Functions --------
def compute_dual_bound(net: Net, bounds_dict: Dict[int, Bounds], c: torch.Tensor,
                       return_sce: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute certified lower bound on c^T @ output."""
    return DualTF().compute_bound(net, bounds_dict, c, return_sce=return_sce)

def compute_robust_loss_bound(net: Net, bounds_dict: Dict[int, Bounds],
                              y_true: int, num_classes: int) -> Tuple[torch.Tensor, bool]:
    """Compute robust classification bound (min margin, is_certified)."""
    return DualTF().compute_robust_bound(net, bounds_dict, y_true, num_classes)
