#===- act/back_end/dual_tf/tf_forward.py - Forward Bounds ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Forward bound propagation for DualTF using linear coefficient tracking.
#   Tracks linear coefficients: output = A @ input + bias
#   Bounds: lb = A @ x0 + bias - |A| @ eps, ub = A @ x0 + bias + |A| @ eps
#   
#   Much tighter than interval propagation for deeper networks.
#   For activation layers, returns PRE-activation bounds (needed by dual backward).
#
#===---------------------------------------------------------------------===#

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from act.back_end.core import Bounds, Net, Layer
from act.back_end.layer_schema import LayerKind

# ============================================================================
# Main Entry Point
# ============================================================================

@torch.no_grad()
def compute_forward_bounds(net: Net, input_lb: torch.Tensor, input_ub: torch.Tensor,
                           post_activation: bool = False) -> Dict[int, Bounds]:
    """
    Compute forward bounds using linear coefficient tracking.
    
    Args:
        net: ACT network
        input_lb, input_ub: Input bounds
        post_activation: If True, return POST-activation bounds (for validation).
                        If False, return PRE-activation bounds (for dual backward).
    """
    bounds_dict: Dict[int, Bounds] = {}
    lb_in, ub_in = input_lb.flatten(), input_ub.flatten()
    input_dim = lb_in.numel()
    device, dtype = lb_in.device, lb_in.dtype
    
    # State: output = A @ input + bias
    x0 = (lb_in + ub_in) / 2  # Center
    eps = (ub_in - lb_in) / 2  # Half-width
    A = torch.eye(input_dim, device=device, dtype=dtype)
    bias = torch.zeros(input_dim, device=device, dtype=dtype)
    lb, ub = lb_in.clone(), ub_in.clone()
    
    for layer in net.layers:
        lid, kind = layer.id, layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        
        # Input layers
        if kind in [LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, "INPUT", "INPUT_SPEC"]:
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            continue
        
        # Dispatch
        if kind in [LayerKind.RELU.value, "RELU"]:
            if not post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())  # PRE-activation (for dual backward)
            A, bias, lb, ub = _fwd_relu(A, bias, x0, eps, lb, ub)
            if post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())  # POST-activation (for validation)
                # Reset state after ReLU in post_activation mode for sound interval propagation
                A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            
        elif kind in [LayerKind.DENSE.value, "DENSE"]:
            A, bias, lb, ub = _fwd_dense(layer, A, bias, x0, eps)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            
        elif kind in [LayerKind.CONV2D.value, "CONV2D"]:
            lb, ub = _fwd_conv2d(layer, lb, ub)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            
        elif kind == "BIAS":
            A, bias, lb, ub = _fwd_bias(layer, A, bias, x0, eps)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            
        elif kind == "SCALE":
            A, bias, lb, ub = _fwd_scale(layer, A, bias, x0, eps)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            
        elif kind == "BN":
            A, bias, lb, ub = _fwd_bn(layer, A, bias, x0, eps)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            
        elif kind in ["FLATTEN", "RESHAPE"]:
            lb, ub = lb.flatten(), ub.flatten()
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            
        elif kind in [LayerKind.SIGMOID.value, "SIGMOID"]:
            if not post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())  # PRE-activation (for dual backward)
            lb, ub = torch.sigmoid(lb), torch.sigmoid(ub)
            if post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())  # POST-activation (for validation)
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            
        elif kind in [LayerKind.TANH.value, "TANH"]:
            if not post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())  # PRE-activation (for dual backward)
            lb, ub = torch.tanh(lb), torch.tanh(ub)
            if post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())  # POST-activation (for validation)
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            
        elif kind in ["LRELU", "LEAKY_RELU"]:
            if not post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())  # PRE-activation (for dual backward)
            alpha = float(layer.meta.get("alpha", 0.01))
            A, bias, lb, ub = _fwd_lrelu(A, bias, x0, eps, lb, ub, alpha)
            if post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())  # POST-activation (for validation)
            
        elif kind in ["MAXPOOL2D"]:
            lb, ub = _fwd_maxpool2d(layer, lb, ub)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            
        elif kind in ["AVGPOOL2D"]:
            lb, ub = _fwd_avgpool2d(layer, lb, ub)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            
        elif kind in [LayerKind.ASSERT.value, "ASSERT", "TRANSPOSE", "SQUEEZE", "UNSQUEEZE"]:
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            
        elif kind == "ADD":
            # ADD layer: z = x + y (+ bias if present)
            # Get bounds from predecessor layers via x_src and y_src
            x_src = layer.meta.get("x_src")
            y_src = layer.meta.get("y_src")
            
            if x_src is not None and y_src is not None and x_src in bounds_dict and y_src in bounds_dict:
                lb_x, ub_x = bounds_dict[x_src].lb.flatten(), bounds_dict[x_src].ub.flatten()
                lb_y, ub_y = bounds_dict[y_src].lb.flatten(), bounds_dict[y_src].ub.flatten()
                
                # Handle shape mismatch (broadcasting)
                if lb_x.numel() != lb_y.numel():
                    min_size = min(lb_x.numel(), lb_y.numel())
                    lb_x, ub_x = lb_x[:min_size], ub_x[:min_size]
                    lb_y, ub_y = lb_y[:min_size], ub_y[:min_size]
                
                lb = lb_x + lb_y
                ub = ub_x + ub_y
                
                # Add bias if present
                if "bias" in layer.params and layer.params["bias"] is not None:
                    b = layer.params["bias"].flatten()
                    if b.numel() != lb.numel():
                        b = b[:lb.numel()] if b.numel() > lb.numel() else b.repeat((lb.numel() + b.numel() - 1) // b.numel())[:lb.numel()]
                    lb = lb + b
                    ub = ub + b
            # else: keep current lb, ub as fallback
            
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            
        else:
            import warnings
            warnings.warn(f"forward_bounds: Unknown layer '{kind}', passing through")
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
    
    return bounds_dict

# ============================================================================
# Layer Handlers
# ============================================================================

def _fwd_dense(layer: Layer, A: torch.Tensor, bias: torch.Tensor, 
               x0: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense: new = W @ (A @ x + bias) + b = (W @ A) @ x + (W @ bias + b)"""
    W = layer.params["W"]
    b = layer.params.get("b")
    
    n_in = W.shape[1]
    if A.shape[0] != n_in:
        if A.shape[0] < n_in:
            pad = n_in - A.shape[0]
            A = torch.cat([A, torch.zeros(pad, A.shape[1], dtype=A.dtype, device=A.device)], dim=0)
            bias = torch.cat([bias, torch.zeros(pad, dtype=bias.dtype, device=bias.device)])
        else:
            A, bias = A[:n_in, :], bias[:n_in]
    
    A_new = W @ A
    bias_new = W @ bias + b if b is not None else W @ bias
    center = A_new @ x0 + bias_new
    radius = A_new.abs() @ eps
    return A_new, bias_new, center - radius, center + radius

def _fwd_relu(A: torch.Tensor, bias: torch.Tensor, x0: torch.Tensor, eps: torch.Tensor,
              lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ReLU forward bounds using CROWN-style upper envelope, interval-style lower bound.
    
    Upper bound: uses linear relaxation y <= (u/(u-l)) * (x - l) for crossing neurons
    Lower bound: uses interval propagation lb_out = max(0, lb_in) for soundness
    
    This ensures sound bounds (lb_out <= true <= ub_out) at the cost of some tightness
    in the lower bound. This is the standard approach for forward propagation.
    """
    device, dtype = lb.device, lb.dtype
    on, off, amb = lb >= 0, ub <= 0, ~((lb >= 0) | (ub <= 0))
    
    # Upper bound linear relaxation slope
    d_ub = torch.where(on, torch.ones_like(lb), torch.zeros_like(lb))
    offset_ub = torch.zeros_like(lb)
    
    if amb.any():
        denom = (ub - lb).clamp(min=1e-12)
        slope = ub / denom
        d_ub = torch.where(amb, slope, d_ub)
        offset_ub = torch.where(amb, -slope * lb, offset_ub)
    
    # Compute upper bound using linear relaxation
    A_ub = d_ub.unsqueeze(1) * A
    bias_ub = d_ub * bias + offset_ub
    center_ub = A_ub @ x0 + bias_ub
    radius_ub = A_ub.abs() @ eps
    ub_out = center_ub + radius_ub
    
    # Lower bound: use interval propagation (sound but looser)
    # lb_out = max(0, lb_in), ub_out from linear relaxation
    lb_out = lb.clamp(min=0)
    
    # For tracking: use upper bound coefficients (will be used for next layer)
    # But the lb_out is computed via interval for soundness
    return A_ub, bias_ub, lb_out, ub_out

def _fwd_bias(layer: Layer, A: torch.Tensor, bias: torch.Tensor,
              x0: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bias: y = x + c"""
    c = _align(layer.params["c"], bias.numel())
    bias_new = bias + c
    center = A @ x0 + bias_new
    radius = A.abs() @ eps
    return A, bias_new, center - radius, center + radius

def _fwd_scale(layer: Layer, A: torch.Tensor, bias: torch.Tensor,
               x0: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale: y = a * x"""
    a = _align(layer.params["a"], A.shape[0])
    A_new = a.unsqueeze(1) * A
    bias_new = a * bias
    center = A_new @ x0 + bias_new
    radius = A_new.abs() @ eps
    return A_new, bias_new, center - radius, center + radius

def _fwd_bn(layer: Layer, A: torch.Tensor, bias: torch.Tensor,
            x0: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """BatchNorm: y = A_bn * x + c"""
    A_bn = _align(layer.params["A"], A.shape[0])
    c = _align(layer.params["c"], bias.numel())
    A_new = A_bn.unsqueeze(1) * A
    bias_new = A_bn * bias + c
    center = A_new @ x0 + bias_new
    radius = A_new.abs() @ eps
    return A_new, bias_new, center - radius, center + radius

def _fwd_lrelu(A: torch.Tensor, bias: torch.Tensor, x0: torch.Tensor, eps: torch.Tensor,
               lb: torch.Tensor, ub: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Leaky ReLU: y = x if x >= 0, y = α*x if x < 0
    
    Upper bound relaxation (line through (l, αl) and (u, u)):
    - Active (lb >= 0): d = 1, offset = 0
    - Inactive (ub <= 0): d = α, offset = 0
    - Crossing: d = (u - αl) / (u - l), offset = αl - d*l
    """
    on, off, amb = lb >= 0, ub <= 0, ~((lb >= 0) | (ub <= 0))
    
    # Active: slope = 1, Inactive: slope = α
    d = torch.where(on, torch.ones_like(lb), torch.full_like(lb, alpha))
    offset = torch.zeros_like(lb)
    
    if amb.any():
        # Line through (l, αl) and (u, u): slope = (u - αl) / (u - l)
        denom = (ub - lb).clamp(min=1e-12)
        slope = (ub - alpha * lb) / denom
        # offset = αl - slope * l = l * (α - slope)
        off_val = alpha * lb - slope * lb
        d = torch.where(amb, slope, d)
        offset = torch.where(amb, off_val, offset)
    
    A_new = d.unsqueeze(1) * A
    bias_new = d * bias + offset
    center = A_new @ x0 + bias_new
    radius = A_new.abs() @ eps
    
    # Leaky ReLU output bounds (can be negative if α > 0)
    lb_out = center - radius
    ub_out = center + radius
    return A_new, bias_new, lb_out, ub_out

def _fwd_conv2d(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Conv2D interval based"""
    weight, bias = layer.params["weight"], layer.params.get("bias")
    stride = layer.meta.get("stride", 1)
    padding = layer.meta.get("padding", 0)
    dilation = layer.meta.get("dilation", 1)
    groups = layer.meta.get("groups", 1)
    
    if isinstance(stride, (list, tuple)): stride = stride[0]
    if isinstance(padding, (list, tuple)): padding = padding[0]
    if isinstance(dilation, (list, tuple)): dilation = dilation[0]
    
    out_c, in_c_per_g, kH, kW = weight.shape
    in_c = in_c_per_g * groups
    spatial = lb.numel() // in_c
    in_h = in_w = int(spatial ** 0.5)
    
    try:
        lb_4d, ub_4d = lb.view(1, in_c, in_h, in_w), ub.view(1, in_c, in_h, in_w)
    except RuntimeError:
        return lb, ub
    
    W_pos, W_neg = weight.clamp(min=0), weight.clamp(max=0)
    lb_out = F.conv2d(lb_4d, W_pos, None, stride, padding, dilation, groups) + \
             F.conv2d(ub_4d, W_neg, None, stride, padding, dilation, groups)
    ub_out = F.conv2d(ub_4d, W_pos, None, stride, padding, dilation, groups) + \
             F.conv2d(lb_4d, W_neg, None, stride, padding, dilation, groups)
    
    if bias is not None:
        lb_out, ub_out = lb_out + bias.view(1, -1, 1, 1), ub_out + bias.view(1, -1, 1, 1)
    return lb_out.flatten(), ub_out.flatten()

def _fwd_maxpool2d(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """MaxPool2D (interval based)"""
    kernel_size = layer.meta.get("kernel_size", 2)
    stride = layer.meta.get("stride", kernel_size)
    padding = layer.meta.get("padding", 0)
    dilation = layer.meta.get("dilation", 1)
    input_shape = layer.meta.get("input_shape")
    if input_shape is None: return lb, ub
    
    b, c, h, w = input_shape
    lb_out = F.max_pool2d(lb.view(b, c, h, w), kernel_size, stride, padding, dilation)
    ub_out = F.max_pool2d(ub.view(b, c, h, w), kernel_size, stride, padding, dilation)
    return lb_out.flatten(), ub_out.flatten()

def _fwd_avgpool2d(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """AvgPool2D (interval based)"""
    kernel_size = layer.meta.get("kernel_size", 2)
    stride = layer.meta.get("stride", kernel_size)
    padding = layer.meta.get("padding", 0)
    input_shape = layer.meta.get("input_shape")
    if input_shape is None: return lb, ub
    
    b, c, h, w = input_shape
    lb_out = F.avg_pool2d(lb.view(b, c, h, w), kernel_size, stride, padding)
    ub_out = F.avg_pool2d(ub.view(b, c, h, w), kernel_size, stride, padding)
    return lb_out.flatten(), ub_out.flatten()

# ============================================================================
# Helpers
# ============================================================================

def _align(a: torch.Tensor, n: int) -> torch.Tensor:
    """Align tensor to size n."""
    a = a.flatten()
    if a.numel() == n: return a
    elif a.numel() > n: return a[:n]
    else: return a.repeat((n + a.numel() - 1) // a.numel())[:n]

def _reset_state(lb: torch.Tensor, ub: torch.Tensor, device, dtype):
    """Reset linear tracking state after non-linear layers."""
    curr_dim = lb.numel()
    A = torch.eye(curr_dim, device=device, dtype=dtype)
    bias = torch.zeros(curr_dim, device=device, dtype=dtype)
    x0 = (lb + ub) / 2
    eps = (ub - lb) / 2
    return A, bias, x0, eps
