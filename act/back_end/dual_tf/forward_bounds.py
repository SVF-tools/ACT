#===- act/back_end/dual_tf/forward_bounds.py - Forward Interval Bounds ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Self-contained forward interval bound propagation for use with DualTF.
#   Handles size mismatches gracefully and works with ACT wrapped models.
#
#===---------------------------------------------------------------------===#

import torch
from typing import Dict, Tuple, Optional
from act.back_end.core import Bounds, Net, Layer
from act.back_end.layer_schema import LayerKind


@torch.no_grad()
def compute_forward_bounds(
    net: Net,
    input_lb: torch.Tensor,
    input_ub: torch.Tensor
) -> Dict[int, Bounds]:
    """
    Compute forward interval bounds for all layers.
    
    Args:
        net: ACT network
        input_lb: Input lower bounds (flattened)
        input_ub: Input upper bounds (flattened)
        
    Returns:
        Dict[layer_id -> Bounds] for all layers
    """
    bounds_dict: Dict[int, Bounds] = {}
    
    # Current bounds flowing through the network
    lb = input_lb.flatten()
    ub = input_ub.flatten()
    
    for layer in net.layers:
        lid = layer.id
        kind = layer.kind
        
        # Store input bounds for INPUT and INPUT_SPEC layers
        if kind in [LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value]:
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            continue
        
        # Dispatch to appropriate handler
        if kind == LayerKind.DENSE.value:
            lb, ub = _forward_dense(layer, lb, ub)
        elif kind == LayerKind.RELU.value:
            # Store pre-ReLU bounds (needed for dual backward)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            lb, ub = _forward_relu(lb, ub)
            continue  # Already stored bounds
        elif kind == "SCALE":
            lb, ub = _forward_scale(layer, lb, ub)
        elif kind == "BIAS":
            lb, ub = _forward_bias(layer, lb, ub)
        elif kind in ["FLATTEN", "RESHAPE"]:
            lb, ub = lb.flatten(), ub.flatten()
        elif kind == LayerKind.CONV2D.value:
            lb, ub = _forward_conv2d(layer, lb, ub)
        elif kind == LayerKind.ASSERT.value:
            pass  # Just pass through
        elif kind in [LayerKind.SIGMOID.value, LayerKind.TANH.value, 
                      LayerKind.SILU.value, "LRELU"]:
            # For smooth activations, just pass bounds (will be refined later)
            lb, ub = _forward_smooth_activation(kind, layer, lb, ub)
        else:
            # Unknown layer - pass through with warning
            import warnings
            warnings.warn(f"forward_bounds: Unknown layer kind '{kind}', passing through")
        
        bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
    
    return bounds_dict


def _forward_dense(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward bounds through Dense layer."""
    W = layer.params["W"]
    b = layer.params.get("b", None)
    
    # Handle size mismatch
    expected_in = W.shape[1]
    actual_in = lb.numel()
    
    if actual_in != expected_in:
        if actual_in < expected_in:
            # Pad input with zeros (conservative)
            pad_lb = torch.zeros(expected_in - actual_in, dtype=lb.dtype, device=lb.device)
            pad_ub = torch.zeros(expected_in - actual_in, dtype=ub.dtype, device=ub.device)
            lb = torch.cat([lb, pad_lb])
            ub = torch.cat([ub, pad_ub])
        else:
            # Truncate input
            lb = lb[:expected_in]
            ub = ub[:expected_in]
    
    W_pos = W.clamp(min=0)
    W_neg = W.clamp(max=0)
    
    lb_out = W_pos @ lb + W_neg @ ub
    ub_out = W_pos @ ub + W_neg @ lb
    
    if b is not None:
        lb_out = lb_out + b
        ub_out = ub_out + b
    
    return lb_out, ub_out


def _forward_relu(lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward bounds through ReLU."""
    lb_out = lb.clamp(min=0)
    ub_out = ub.clamp(min=0)
    return lb_out, ub_out


def _forward_scale(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward bounds through Scale layer (y = a * x)."""
    a = layer.params["a"]
    
    # Handle size mismatch
    if a.numel() != lb.numel():
        if a.numel() > lb.numel():
            a = a[:lb.numel()]
        else:
            repeats = (lb.numel() + a.numel() - 1) // a.numel()
            a = a.repeat(repeats)[:lb.numel()]
    
    # For scaling: if a >= 0, lb_out = a*lb, ub_out = a*ub
    #              if a < 0,  lb_out = a*ub, ub_out = a*lb
    lb_out = torch.where(a >= 0, a * lb, a * ub)
    ub_out = torch.where(a >= 0, a * ub, a * lb)
    
    return lb_out, ub_out


def _forward_bias(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward bounds through Bias layer (y = x + c)."""
    c = layer.params["c"]
    
    # Handle size mismatch
    if c.numel() != lb.numel():
        if c.numel() > lb.numel():
            c = c[:lb.numel()]
        else:
            repeats = (lb.numel() + c.numel() - 1) // c.numel()
            c = c.repeat(repeats)[:lb.numel()]
    
    return lb + c, ub + c

def _forward_smooth_activation(
    kind: str, 
    layer: Layer, 
    lb: torch.Tensor, 
    ub: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward bounds through smooth activations (conservative)."""
    if kind == LayerKind.SIGMOID.value:
        # sigmoid is monotonic: [sigmoid(lb), sigmoid(ub)]
        return torch.sigmoid(lb), torch.sigmoid(ub)
    elif kind == LayerKind.TANH.value:
        # tanh is monotonic: [tanh(lb), tanh(ub)]
        return torch.tanh(lb), torch.tanh(ub)
    elif kind == LayerKind.SILU.value:
        # SiLU (x * sigmoid(x)) - use min/max of endpoints and inflection
        silu = lambda x: x * torch.sigmoid(x)
        lb_out = torch.minimum(silu(lb), silu(ub))
        ub_out = torch.maximum(silu(lb), silu(ub))
        # Check if 0 is in interval (inflection point)
        mask = (lb < 0) & (ub > 0)
        lb_out[mask] = torch.minimum(lb_out[mask], silu(torch.zeros_like(lb[mask])))
        return lb_out, ub_out
    elif kind == "LRELU":
        alpha = float(layer.meta.get("alpha", 0.01))
        # Leaky ReLU: x if x >= 0, alpha*x if x < 0
        lb_out = torch.where(lb >= 0, lb, alpha * lb)
        ub_out = torch.where(ub >= 0, ub, alpha * ub)
        # Handle crossing zero
        mask = (lb < 0) & (ub > 0)
        lb_out[mask] = torch.minimum(alpha * lb[mask], torch.zeros_like(lb[mask]))
        return lb_out, ub_out
    else:
        return lb, ub


# =============================================================================
# CNN Layers (Placeholders)
# =============================================================================

def _forward_conv2d(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward bounds through Conv2d using matrix form."""
    weight = layer.params["weight"]
    bias = layer.params.get("bias", None)
    
    stride = layer.meta.get("stride", 1)
    padding = layer.meta.get("padding", 0)
    dilation = layer.meta.get("dilation", 1)
    groups = layer.meta.get("groups", 1)
    
    # Normalize to tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # Get weight dimensions
    out_channels, in_channels_per_group, kernel_h, kernel_w = weight.shape
    in_channels = in_channels_per_group * groups
    
    # Infer spatial dimensions from input size
    actual_input_size = lb.numel()
    spatial_size = actual_input_size // in_channels
    in_h = in_w = int(spatial_size ** 0.5)
    
    # Compute output dimensions
    out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    
    # Reshape input bounds to 4D
    try:
        lb_4d = lb.view(1, in_channels, in_h, in_w)
        ub_4d = ub.view(1, in_channels, in_h, in_w)
    except RuntimeError:
        # If reshape fails, just pass through
        return lb, ub
    
    # Use PyTorch conv for forward pass with interval arithmetic
    import torch.nn.functional as F
    
    # Split weights into positive and negative parts
    W_pos = weight.clamp(min=0)
    W_neg = weight.clamp(max=0)
    
    # Compute bounds
    lb_out = F.conv2d(lb_4d, W_pos, None, stride, padding, dilation, groups) + \
             F.conv2d(ub_4d, W_neg, None, stride, padding, dilation, groups)
    ub_out = F.conv2d(ub_4d, W_pos, None, stride, padding, dilation, groups) + \
             F.conv2d(lb_4d, W_neg, None, stride, padding, dilation, groups)
    
    if bias is not None:
        lb_out = lb_out + bias.view(1, -1, 1, 1)
        ub_out = ub_out + bias.view(1, -1, 1, 1)
    
    return lb_out.flatten(), ub_out.flatten()


def _forward_maxpool2d(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward bounds through MaxPool2d (placeholder)."""
    raise NotImplementedError("_forward_maxpool2d")


def _forward_avgpool2d(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward bounds through AvgPool2d (placeholder)."""
    raise NotImplementedError("_forward_avgpool2d")


# =============================================================================
# RNN Layers (Placeholders)
# =============================================================================

def _forward_lstm(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward bounds through LSTM (placeholder)."""
    raise NotImplementedError("_forward_lstm")


def _forward_gru(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward bounds through GRU (placeholder)."""
    raise NotImplementedError("_forward_gru")


# =============================================================================
# Transformer Layers (Placeholders)
# =============================================================================

def _forward_attention(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward bounds through Multi-head Attention (placeholder)."""
    raise NotImplementedError("_forward_attention")


def _forward_layernorm(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward bounds through LayerNorm (placeholder)."""
    raise NotImplementedError("_forward_layernorm")
