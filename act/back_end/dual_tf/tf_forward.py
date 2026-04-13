# ===- act/back_end/dual_tf/tf_forward.py - Forward Bounds ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Forward bound propagation for DualTF using linear coefficient tracking.
#   Tracks linear coefficients: output = A @ input + bias
#   Bounds: lb = A @ x0 + bias - |A| @ eps, ub = A @ x0 + bias + |A| @ eps
#
#   Much tighter than interval propagation for deeper networks.
#   For activation layers, returns PRE-activation bounds (needed by dual backward).
#
# ===---------------------------------------------------------------------===#

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from act.back_end.core import Bounds, Net, Layer
from act.back_end.layer_schema import LayerKind

# ============================================================================
# Main Entry Point
# ============================================================================


@torch.no_grad()
def compute_forward_bounds(
    net: Net,
    input_lb: torch.Tensor,
    input_ub: torch.Tensor,
    post_activation: bool = False,
) -> Dict[int, Bounds]:
    """
    Compute forward bounds using linear coefficient tracking.

    Args:
        net: ACT network
        input_lb, input_ub: Input bounds
        post_activation: If True, return POST-activation bounds (for validation).
                        If False, return PRE-activation bounds (for dual backward).
    """
    bounds_dict: Dict[int, Bounds] = {}
    assert input_lb.dim() == 2, f"input_lb must be [B, n], got shape {input_lb.shape}"
    if input_lb.shape[0] > 1 and not post_activation:
        raise ValueError(
            "Batch mode (B>1) requires post_activation=True. "
            "With post_activation=False, RELU slopes make A per-instance "
            "which breaks the shared-A invariant."
        )
    lb_in = input_lb  # [B, n]
    ub_in = input_ub  # [B, n]
    B = lb_in.shape[0]
    input_dim = lb_in.shape[1]
    device, dtype = lb_in.device, lb_in.dtype

    x0 = (lb_in + ub_in) / 2  # [B, input_dim]
    eps = (ub_in - lb_in) / 2  # [B, input_dim]
    A = torch.eye(input_dim, device=device, dtype=dtype)  # [n, n] shared
    bias = torch.zeros(input_dim, device=device, dtype=dtype)  # [n] shared
    lb, ub = lb_in.clone(), ub_in.clone()  # [B, n]

    for layer in net.layers:
        lid, kind = (
            layer.id,
            layer.kind.upper() if isinstance(layer.kind, str) else layer.kind,
        )

        # Input layers
        if kind in [
            LayerKind.INPUT.value,
            LayerKind.INPUT_SPEC.value,
            "INPUT",
            "INPUT_SPEC",
        ]:
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            continue

        # Dispatch
        if kind in [LayerKind.RELU.value, "RELU"]:
            if not post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            if post_activation or B > 1:
                lb, ub = _fwd_relu_bounds(A, bias, x0, eps, lb, ub)
                if post_activation:
                    bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
                A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            else:
                A, bias, lb, ub = _fwd_relu_tracked(A, bias, x0, eps, lb, ub)

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
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())

        elif kind in [LayerKind.SIGMOID.value, "SIGMOID"]:
            if not post_activation:
                bounds_dict[lid] = Bounds(
                    lb.clone(), ub.clone()
                )  # PRE-activation (for dual backward)
            lb, ub = torch.sigmoid(lb), torch.sigmoid(ub)
            if post_activation:
                bounds_dict[lid] = Bounds(
                    lb.clone(), ub.clone()
                )  # POST-activation (for validation)
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)

        elif kind in [LayerKind.TANH.value, "TANH"]:
            if not post_activation:
                bounds_dict[lid] = Bounds(
                    lb.clone(), ub.clone()
                )  # PRE-activation (for dual backward)
            lb, ub = torch.tanh(lb), torch.tanh(ub)
            if post_activation:
                bounds_dict[lid] = Bounds(
                    lb.clone(), ub.clone()
                )  # POST-activation (for validation)
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)

        elif kind in ["LRELU", "LEAKY_RELU"]:
            if not post_activation:
                bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            alpha = float(layer.params.get("alpha", 0.01))
            if post_activation or B > 1:
                lb, ub = _fwd_lrelu_bounds(A, bias, x0, eps, lb, ub, alpha)
                if post_activation:
                    bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
                A, bias, x0, eps = _reset_state(lb, ub, device, dtype)
            else:
                A, bias, lb, ub = _fwd_lrelu_tracked(A, bias, x0, eps, lb, ub, alpha)

        elif kind in ["MAXPOOL2D"]:
            lb, ub = _fwd_maxpool2d(layer, lb, ub)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)

        elif kind in ["AVGPOOL2D"]:
            lb, ub = _fwd_avgpool2d(layer, lb, ub)
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())
            A, bias, x0, eps = _reset_state(lb, ub, device, dtype)

        elif kind in [
            LayerKind.ASSERT.value,
            "ASSERT",
            "TRANSPOSE",
            "SQUEEZE",
            "UNSQUEEZE",
        ]:
            bounds_dict[lid] = Bounds(lb.clone(), ub.clone())

        elif kind == "ADD":
            # ADD layer: z = x + y (+ bias if present)
            # Get bounds from predecessor layers via x_src and y_src
            x_src = layer.params.get("x_src")
            y_src = layer.params.get("y_src")

            if (
                x_src is not None
                and y_src is not None
                and x_src in bounds_dict
                and y_src in bounds_dict
            ):
                lb_x, ub_x = bounds_dict[x_src].lb, bounds_dict[x_src].ub
                lb_y, ub_y = bounds_dict[y_src].lb, bounds_dict[y_src].ub

                if lb_x.shape[-1] != lb_y.shape[-1]:
                    min_size = min(lb_x.shape[-1], lb_y.shape[-1])
                    lb_x, ub_x = lb_x[..., :min_size], ub_x[..., :min_size]
                    lb_y, ub_y = lb_y[..., :min_size], ub_y[..., :min_size]

                lb = lb_x + lb_y
                ub = ub_x + ub_y

                # Add bias if present
                if "bias" in layer.params and layer.params["bias"] is not None:
                    b = layer.params["bias"].flatten()
                    n_out = lb.shape[-1]
                    if b.numel() != n_out:
                        b = (
                            b[:n_out]
                            if b.numel() > n_out
                            else b.repeat((n_out + b.numel() - 1) // b.numel())[:n_out]
                        )
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


def _fwd_dense(
    layer: Layer,
    A: torch.Tensor,
    bias: torch.Tensor,
    x0: torch.Tensor,
    eps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense: new = W @ (A @ x + bias) + b = (W @ A) @ x + (W @ bias + b)"""
    W = layer.params["weight"]
    b = layer.params.get("bias")

    n_in = W.shape[1]
    if A.shape[0] != n_in:
        if A.shape[0] < n_in:
            pad = n_in - A.shape[0]
            A = torch.cat(
                [A, torch.zeros(pad, A.shape[1], dtype=A.dtype, device=A.device)], dim=0
            )
            bias = torch.cat(
                [bias, torch.zeros(pad, dtype=bias.dtype, device=bias.device)]
            )
        else:
            A, bias = A[:n_in, :], bias[:n_in]

    A_new = W @ A
    bias_new = W @ bias + b if b is not None else W @ bias
    center = x0 @ A_new.T + bias_new
    radius = eps @ A_new.abs().T
    return A_new, bias_new, center - radius, center + radius


def _fwd_relu_bounds(
    A: torch.Tensor,
    bias: torch.Tensor,
    x0: torch.Tensor,
    eps: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ReLU forward bounds using diagonal decomposition (batched).
    diag(d) @ A @ x0 = d * (x0 @ A.T) — avoids materializing [B, n, input_dim].
    """
    on, off, amb = lb >= 0, ub <= 0, ~((lb >= 0) | (ub <= 0))

    d_ub = torch.where(on, torch.ones_like(lb), torch.zeros_like(lb))
    offset_ub = torch.zeros_like(lb)

    if amb.any():
        denom = (ub - lb).clamp(min=1e-12)
        slope = ub / denom
        d_ub = torch.where(amb, slope, d_ub)
        offset_ub = torch.where(amb, -slope * lb, offset_ub)

    Ax0 = x0 @ A.T
    Aeps = eps @ A.abs().T
    center_ub = d_ub * (Ax0 + bias) + offset_ub
    radius_ub = d_ub * Aeps
    ub_out = center_ub + radius_ub
    lb_out = lb.clamp(min=0)

    return lb_out, ub_out


def _fwd_relu_tracked(A, bias, x0, eps, lb, ub):
    """B=1 RELU with A tracking for post_activation=False. Squeezes to 1D internally."""
    lb_s, ub_s = lb.squeeze(0), ub.squeeze(0)
    on, off, amb = lb_s >= 0, ub_s <= 0, ~((lb_s >= 0) | (ub_s <= 0))
    d_ub = torch.where(on, torch.ones_like(lb_s), torch.zeros_like(lb_s))
    offset_ub = torch.zeros_like(lb_s)
    if amb.any():
        denom = (ub_s - lb_s).clamp(min=1e-12)
        slope = ub_s / denom
        d_ub = torch.where(amb, slope, d_ub)
        offset_ub = torch.where(amb, -slope * lb_s, offset_ub)
    A_ub = d_ub.unsqueeze(1) * A
    bias_ub = d_ub * bias + offset_ub
    center_ub = x0 @ A_ub.T + bias_ub
    radius_ub = eps @ A_ub.abs().T
    return A_ub, bias_ub, lb.clamp(min=0), center_ub + radius_ub


def _fwd_lrelu_tracked(A, bias, x0, eps, lb, ub, alpha):
    """B=1 LRELU with A tracking for post_activation=False."""
    lb_s, ub_s = lb.squeeze(0), ub.squeeze(0)
    on, off, amb = lb_s >= 0, ub_s <= 0, ~((lb_s >= 0) | (ub_s <= 0))
    d = torch.where(on, torch.ones_like(lb_s), torch.full_like(lb_s, alpha))
    offset = torch.zeros_like(lb_s)
    if amb.any():
        denom = (ub_s - lb_s).clamp(min=1e-12)
        slope = (ub_s - alpha * lb_s) / denom
        d = torch.where(amb, slope, d)
        offset = torch.where(amb, alpha * lb_s - slope * lb_s, offset)
    A_new = d.unsqueeze(1) * A
    bias_new = d * bias + offset
    center = x0 @ A_new.T + bias_new
    radius = eps @ A_new.abs().T
    lb_out = torch.where(on, lb, alpha * lb)
    return A_new, bias_new, lb_out, center + radius


def _fwd_bias(
    layer: Layer,
    A: torch.Tensor,
    bias: torch.Tensor,
    x0: torch.Tensor,
    eps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bias: y = x + c"""
    c = _align(layer.params["c"], bias.numel())
    bias_new = bias + c
    center = x0 @ A.T + bias_new
    radius = eps @ A.abs().T
    return A, bias_new, center - radius, center + radius


def _fwd_scale(
    layer: Layer,
    A: torch.Tensor,
    bias: torch.Tensor,
    x0: torch.Tensor,
    eps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale: y = a * x"""
    a = _align(layer.params["a"], A.shape[0])
    A_new = a.unsqueeze(1) * A
    bias_new = a * bias
    center = x0 @ A_new.T + bias_new
    radius = eps @ A_new.abs().T
    return A_new, bias_new, center - radius, center + radius


def _fwd_bn(
    layer: Layer,
    A: torch.Tensor,
    bias: torch.Tensor,
    x0: torch.Tensor,
    eps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """BatchNorm: y = A_bn * x + c"""
    A_bn = _align(layer.params["A"], A.shape[0])
    c = _align(layer.params["c"], bias.numel())
    A_new = A_bn.unsqueeze(1) * A
    bias_new = A_bn * bias + c
    center = x0 @ A_new.T + bias_new
    radius = eps @ A_new.abs().T
    return A_new, bias_new, center - radius, center + radius


def _fwd_lrelu_bounds(
    A: torch.Tensor,
    bias: torch.Tensor,
    x0: torch.Tensor,
    eps: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Leaky ReLU forward bounds using diagonal decomposition (batched)."""
    on, off, amb = lb >= 0, ub <= 0, ~((lb >= 0) | (ub <= 0))

    d = torch.where(on, torch.ones_like(lb), torch.full_like(lb, alpha))
    offset = torch.zeros_like(lb)

    if amb.any():
        denom = (ub - lb).clamp(min=1e-12)
        slope = (ub - alpha * lb) / denom
        off_val = alpha * lb - slope * lb
        d = torch.where(amb, slope, d)
        offset = torch.where(amb, off_val, offset)

    Ax0 = x0 @ A.T
    Aeps = eps @ A.abs().T
    center = d * (Ax0 + bias) + offset
    radius = d.abs() * Aeps
    ub_out = center + radius
    lb_out = torch.where(on, lb, alpha * lb)
    return lb_out, ub_out


def _fwd_conv2d(
    layer: Layer, lb: torch.Tensor, ub: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Conv2D interval based"""
    weight, bias = layer.params["weight"], layer.params.get("bias")
    stride = layer.params.get("stride", 1)
    padding = layer.params.get("padding", 0)
    dilation = layer.params.get("dilation", 1)
    groups = layer.params.get("groups", 1)

    if isinstance(stride, (list, tuple)):
        stride = stride[0]
    if isinstance(padding, (list, tuple)):
        padding = padding[0]
    if isinstance(dilation, (list, tuple)):
        dilation = dilation[0]

    out_c, in_c_per_g, kH, kW = weight.shape
    in_c = in_c_per_g * groups
    B = lb.shape[0]
    spatial = lb.shape[1] // in_c
    in_h = in_w = int(spatial**0.5)

    try:
        lb_4d, ub_4d = lb.view(B, in_c, in_h, in_w), ub.view(B, in_c, in_h, in_w)
    except RuntimeError:
        return lb, ub

    W_pos, W_neg = weight.clamp(min=0), weight.clamp(max=0)
    lb_out = F.conv2d(lb_4d, W_pos, None, stride, padding, dilation, groups) + F.conv2d(
        ub_4d, W_neg, None, stride, padding, dilation, groups
    )
    ub_out = F.conv2d(ub_4d, W_pos, None, stride, padding, dilation, groups) + F.conv2d(
        lb_4d, W_neg, None, stride, padding, dilation, groups
    )

    if bias is not None:
        lb_out, ub_out = (
            lb_out + bias.view(1, -1, 1, 1),
            ub_out + bias.view(1, -1, 1, 1),
        )
    return lb_out.flatten(start_dim=1), ub_out.flatten(start_dim=1)


def _fwd_maxpool2d(
    layer: Layer, lb: torch.Tensor, ub: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MaxPool2D (interval based)"""
    kernel_size = layer.params.get("kernel_size", 2)
    stride = layer.params.get("stride", kernel_size)
    padding = layer.params.get("padding", 0)
    dilation = layer.params.get("dilation", 1)
    input_shape = layer.params.get("input_shape")
    if input_shape is None:
        return lb, ub

    _, c, h, w = input_shape
    B = lb.shape[0]
    lb_out = F.max_pool2d(lb.view(B, c, h, w), kernel_size, stride, padding, dilation)
    ub_out = F.max_pool2d(ub.view(B, c, h, w), kernel_size, stride, padding, dilation)
    return lb_out.flatten(start_dim=1), ub_out.flatten(start_dim=1)


def _fwd_avgpool2d(
    layer: Layer, lb: torch.Tensor, ub: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """AvgPool2D (interval based)"""
    kernel_size = layer.params.get("kernel_size", 2)
    stride = layer.params.get("stride", kernel_size)
    padding = layer.params.get("padding", 0)
    input_shape = layer.params.get("input_shape")
    if input_shape is None:
        return lb, ub

    _, c, h, w = input_shape
    B = lb.shape[0]
    lb_out = F.avg_pool2d(lb.view(B, c, h, w), kernel_size, stride, padding)
    ub_out = F.avg_pool2d(ub.view(B, c, h, w), kernel_size, stride, padding)
    return lb_out.flatten(start_dim=1), ub_out.flatten(start_dim=1)


# ============================================================================
# Helpers
# ============================================================================


def _align(a: torch.Tensor, n: int) -> torch.Tensor:
    """Align tensor to size n."""
    a = a.flatten()
    if a.numel() == n:
        return a
    elif a.numel() > n:
        return a[:n]
    else:
        return a.repeat((n + a.numel() - 1) // a.numel())[:n]


def _reset_state(lb: torch.Tensor, ub: torch.Tensor, device, dtype):
    """Reset linear tracking state after non-linear layers."""
    curr_dim = lb.shape[1]
    A = torch.eye(curr_dim, device=device, dtype=dtype)
    bias = torch.zeros(curr_dim, device=device, dtype=dtype)
    x0 = (lb + ub) / 2
    eps = (ub - lb) / 2
    return A, bias, x0, eps
