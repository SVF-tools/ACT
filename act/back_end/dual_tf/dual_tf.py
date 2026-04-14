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
from .tf_mlp import (
    dual_relu_backward,
    dual_dense_backward,
    dual_bias_backward,
    dual_scale_backward,
    dual_bn_backward,
    dual_identity_backward,
)
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
        LayerKind.BIAS.value: "_backward_bias",
        LayerKind.SCALE.value: "_backward_scale",
        LayerKind.BN.value: "_backward_bn",
        LayerKind.LRELU.value: "_backward_relu",  # TODO: proper leaky ReLU
        # Identity-like
        LayerKind.INPUT.value: "_backward_identity",
        LayerKind.INPUT_SPEC.value: "_backward_identity",
        LayerKind.ASSERT.value: "_backward_identity",
        LayerKind.FLATTEN.value: "_backward_identity",
        LayerKind.RESHAPE.value: "_backward_identity",
        LayerKind.TRANSPOSE.value: "_backward_identity",
        LayerKind.SQUEEZE.value: "_backward_identity",
        LayerKind.UNSQUEEZE.value: "_backward_identity",
        # Smooth activations (S-shaped with tangent relaxation)
        LayerKind.SIGMOID.value: "_backward_sigmoid",
        LayerKind.TANH.value: "_backward_tanh",
        # Multi-input operations (residual connections)
        LayerKind.ADD.value: "_backward_add",
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
    def apply(
        self,
        L: Layer,
        input_bounds: Bounds,
        net: Net,
        before: Dict[int, Fact],
        after: Dict[int, Fact],
    ) -> Fact:
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
                if layer.kind.upper() in [
                    LayerKind.INPUT.value,
                    LayerKind.INPUT_SPEC.value,
                    "INPUT",
                    "INPUT_SPEC",
                ]:
                    if layer.id in before:
                        input_lb = before[layer.id].bounds.lb
                        input_ub = before[layer.id].bounds.ub
                        break
                    elif "lb" in layer.params and "ub" in layer.params:
                        input_lb = layer.params["lb"]
                        input_ub = layer.params["ub"]
                        break
            
            if input_lb is None or input_ub is None:
                input_lb, input_ub = input_bounds.lb, input_bounds.ub
            
            if input_lb.dim() == 1:
                input_lb = input_lb.unsqueeze(0)
                input_ub = input_ub.unsqueeze(0)

            self._forward_bounds_cache = compute_forward_bounds(
                net, input_lb, input_ub, post_activation=True
            )
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
    def compute_bound(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        c: torch.Tensor,
        return_sce: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute certified lower bound on c^T @ output."""
        assert len(bounds_dict) > 0, "bounds_dict cannot be empty"
        sample = next(iter(bounds_dict.values()))
        B = sample.lb.shape[0]
        
        self._bounds_dict = bounds_dict
        nu = c.unsqueeze(0).expand(B, -1).clone()
        obj = torch.zeros(B, dtype=c.dtype, device=c.device)
        
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
        
        margins = torch.stack(margins)  # [num_classes-1, B]
        min_margins = margins.min(dim=0).values  # [B]
        return min_margins, (min_margins > 0)
    
    # -------- Backward Handlers --------
    def _backward_dense(self, L: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return dual_dense_backward(nu, L.params["weight"], L.params.get("bias"))
    
    def _backward_relu(self, L: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bounds = self._bounds_dict.get(L.id)
        return (
            dual_identity_backward(nu)
            if bounds is None
            else dual_relu_backward(nu, bounds)
        )

    def _backward_bias(
        self, L: Layer, nu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return dual_bias_backward(nu, L.params["c"])
    
    def _backward_scale(self, L: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return dual_scale_backward(nu, L.params["a"])
    
    def _backward_bn(self, L: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return dual_bn_backward(nu, L.params["A"], L.params["c"])
    
    def _backward_conv2d(self, L: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stride, padding = L.params.get("stride", 1), L.params.get("padding", 0)
        if isinstance(stride, (list, tuple)):
            stride = stride[0]
        if isinstance(padding, (list, tuple)):
            padding = padding[0]
        return dual_conv2d_backward(
            nu,
            L.params["weight"],
            L.params.get("bias"),
            stride=stride,
            padding=padding,
            input_shape=L.params.get("input_shape"),
            output_shape=L.params.get("output_shape"),
        )

    def _backward_identity(
        self, L: Layer, nu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return dual_identity_backward(nu)
    
    def _backward_sigmoid(self, L: Layer, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bounds = self._bounds_dict.get(L.id)
        return (
            dual_identity_backward(nu)
            if bounds is None
            else dual_sigmoid_backward(nu, bounds)
        )

    def _backward_tanh(
        self, L: Layer, nu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bounds = self._bounds_dict.get(L.id)
        return (
            dual_identity_backward(nu)
            if bounds is None
            else dual_tanh_backward(nu, bounds)
        )

    def _backward_add(
        self, L: Layer, nu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        contrib = torch.zeros(nu.shape[0], dtype=nu.dtype, device=nu.device)

        if "bias" in L.params and L.params["bias"] is not None:
            b = L.params["bias"].flatten()
            n = min(nu.shape[-1], b.numel())
            contrib = (nu[..., :n] * b[:n]).sum(dim=-1)
        
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
            return torch.zeros(nu.shape[0], dtype=nu.dtype, device=nu.device), None

        bounds = self._bounds_dict.get(input_layer.id)
        if bounds is None:
            if "lb" in input_layer.params and "ub" in input_layer.params:
                lb, ub = input_layer.params["lb"], input_layer.params["ub"]
            else:
                return torch.zeros(nu.shape[0], dtype=nu.dtype, device=nu.device), None
        else:
            lb, ub = bounds.lb, bounds.ub
        
        l, u, v = lb, ub, nu
        n = min(v.shape[-1], l.shape[-1])
        l, u, v = l[..., :n], u[..., :n], v[..., :n]

        contrib = (l * v.clamp(min=0)).sum(dim=-1) + (u * v.clamp(max=0)).sum(dim=-1)

        sce = None
        if return_sce:
            sce = torch.where(v > 0, l, u)
        
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
