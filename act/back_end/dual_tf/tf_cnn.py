#===- act/back_end/dual_tf/tf_cnn.py - CNN Dual Transfer Functions ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# Batch-aware Conv2D backward. nu: [B, *out_shape] -> v_out: [B, *in_flat], contrib: [B].
#===---------------------------------------------------------------------===#

# Note: Gradient enablement for dual backward helpers is governed by the
# caller's torch.set_grad_enabled() context (see DualSolver.evaluate_spec).
# @torch.no_grad() decorators on these helpers were removed to allow
# gradient flow during robust training; verify_once / verify_bab paths
# remain under no_grad via their own outer guards.

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List, cast
from act.back_end.core import Bounds, Layer

from .tf_forward import (
    LinearBound, Frame,
    _fwd_conv2d, _fwd_conv2d_interval, _fwd_maxpool2d, _fwd_avgpool2d,
    _reset_forward_box, _concretize,
)


# Forward registry handlers. Each returns (stored, out, lin, frame).


# ---- CONV2D ----
def forward_conv2d(
    L: Layer,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Conv2D forward bounds with dual-track fallback.

    Source: tf_forward.py lines 380-391 (pre-refactor monolithic CONV2D branch).

    Tries linear-relaxation conv via ``_fwd_conv2d``; when it returns ``None``
    (kernel/stride shape unsupported), falls back to the interval conv
    ``_fwd_conv2d_interval`` and resets the dual-track state at the new
    concrete box via ``_reset_forward_box``. ``stored`` equals ``out``
    (CONV2D has no activation split).
    """
    assert len(parent_boxes) == 1, f"CONV2D expects 1 predecessor, got {len(parent_boxes)}"
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    frame = parent_frames[0]
    x_L, x_U = frame

    new_lin = _fwd_conv2d(L, parent_lin)
    if new_lin is None:
        lb, ub = _fwd_conv2d_interval(L, parent_box.lb, parent_box.ub)
        out = Bounds(lb, ub)
        stored = out
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
    else:
        lin = new_lin
        lb, ub = _concretize(lin, x_L, x_U)
        out = Bounds(lb, ub)
        stored = out
    return stored, out, lin, frame


_CONV_CHANNEL_CHUNK_SIZE = 32


def backward_conv2d(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                    preds: List[int], M: int = 1, alpha=None
                    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    stride = L.params.get("stride", 1)
    padding = L.params.get("padding", 0)
    if isinstance(stride, (list, tuple)): stride = stride[0]
    if isinstance(padding, (list, tuple)): padding = padding[0]
    nu_out, contrib = dual_conv2d_backward(
        nu, L.params["weight"], L.params.get("bias"),
        stride=stride, padding=padding,
        input_shape=L.params.get("input_shape"),
        output_shape=L.params.get("output_shape"),
    )
    assert len(preds) == 1, f"CONV2D expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_conv2d_backward(
    nu: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
    stride: int = 1, padding: int = 0,
    input_shape: Optional[tuple] = None, output_shape: Optional[tuple] = None,
    channel_chunk_size: int = _CONV_CHANNEL_CHUNK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Conv2D backward via F.conv_transpose2d (naturally batched).

    Channel-folding: processes out_channels in chunks of
    ``channel_chunk_size`` to bound peak memory. Sum over chunks is
    bit-identical to unchunked conv_transpose2d (PyTorch's reduction is
    sequential pairwise on contiguous memory). Peak intermediate memory
    drops by ``channel_chunk_size / out_channels`` vs the unchunked path.
    """
    assert weight.dim() == 4, f"weight must be 4D [oC,iC,kH,kW], got {weight.shape}"
    assert nu.dim() >= 2, f"nu must be batched (>=2D), got {nu.shape}"
    B = nu.shape[0]
    oC, iC, kH, kW = weight.shape

    v_flat = nu.flatten(start_dim=1)                              # [B, n_out]
    n = v_flat.shape[-1]

    if output_shape is not None:
        if len(output_shape) == 4:   _, oC2, oH, oW = output_shape
        elif len(output_shape) == 3: oC2, oH, oW = output_shape
        else:                        oC2, oH, oW = oC, 1, 1
        assert oC2 == oC, f"output_shape channels {oC2} != weight oC {oC}"
    else:
        spatial = n // oC if oC > 0 else n
        side = int(spatial ** 0.5) if spatial > 0 else 1
        oH = oW = side

    expected = oC * oH * oW
    if n == expected:
        v_4d = v_flat.view(B, oC, oH, oW)
    elif n > expected:
        v_4d = v_flat[:, :expected].contiguous().view(B, oC, oH, oW)
    else:
        v_pad = torch.zeros(B, expected, dtype=v_flat.dtype, device=v_flat.device)
        v_pad[:, :n] = v_flat
        v_4d = v_pad.view(B, oC, oH, oW)

    if isinstance(stride, (list, tuple)): stride = stride[0]
    if isinstance(padding, (list, tuple)): padding = padding[0]

    # Derive output_padding so conv_transpose2d exactly recovers the original
    # input spatial size (stride > 1 loses an increment of up to stride-1).
    output_padding = 0
    if input_shape is not None:
        shape = list(input_shape)
        if len(shape) >= 4:
            iH, iW = shape[-2], shape[-1]
        elif len(shape) == 3:
            iH, iW = shape[-2], shape[-1]
        else:
            iH = iW = None
        if iH is not None:
            computed_h = (v_4d.shape[-2] - 1) * stride - 2 * padding + kH
            op_h = iH - computed_h
            if op_h > 0:
                output_padding = op_h

    if channel_chunk_size >= oC:
        v_out_4d: torch.Tensor = F.conv_transpose2d(v_4d, weight, None,
                                                    stride=stride, padding=padding,
                                                    output_padding=output_padding)
    else:
        first_chunk_end = min(channel_chunk_size, oC)
        v_out_4d = F.conv_transpose2d(v_4d[:, :first_chunk_end, :, :],
                                      weight[:first_chunk_end, :, :, :], None,
                                      stride=stride, padding=padding,
                                      output_padding=output_padding)
        for c_start in range(channel_chunk_size, oC, channel_chunk_size):
            c_end = min(c_start + channel_chunk_size, oC)
            v_out_4d = v_out_4d + F.conv_transpose2d(
                v_4d[:, c_start:c_end, :, :],
                weight[c_start:c_end, :, :, :], None,
                stride=stride, padding=padding,
                output_padding=output_padding,
            )
    v_out = v_out_4d.flatten(start_dim=1)

    if bias is not None:
        per_ch = v_4d.sum(dim=(-1, -2))
        contrib = -(per_ch @ bias.flatten())
    else:
        contrib = torch.zeros(B, dtype=nu.dtype, device=nu.device)
    return v_out, contrib


# ---- MAXPOOL2D ----
def forward_maxpool2d(
    L: Layer,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """MaxPool2D forward bounds (interval-only; dual-track resets).

    Source: tf_forward.py lines 448-452 (pre-refactor monolithic MAXPOOL2D
    branch).

    MaxPool has no linear relaxation here, so we compute interval bounds
    via ``_fwd_maxpool2d`` and reset the dual-track state at the resulting
    concrete box. ``stored`` equals ``out``.
    """
    assert len(parent_boxes) == 1, f"MAXPOOL2D expects 1 predecessor, got {len(parent_boxes)}"
    parent_box = parent_boxes[0]
    lb, ub = _fwd_maxpool2d(L, parent_box.lb, parent_box.ub)
    out = Bounds(lb, ub)
    stored = out
    lin, frame = _reset_forward_box(lb, ub, device, dtype)
    return stored, out, lin, frame


def backward_maxpool2d(L, nu, bounds_dict, preds, M: int = 1, alpha=None):
    """MaxPool2D backward — conservative constant-bound (sound but loose).

    MaxPool is non-linear: ``y = max(x_window)``. For sound dual lower bound
    on ``c @ output``, we use the dual decomposition ``nu @ y = nu_pos @ y +
    nu_neg @ y >= nu_pos @ LB(y) + nu_neg @ UB(y)``, where:

      - LB(y) = max over window of lb_in = bounds_dict[L].lb  (constant, sound: max(x) >= LB(max(x)) = max(lb))
      - UB(y) = max over window of ub_in = bounds_dict[L].ub  (constant, sound: max(x) <= UB(max(x)) = max(ub))

    Both bounds are CONSTANT in x → ``nu_in = 0`` (no propagation), full
    contribution to obj via ``contrib = nu_pos @ lb_out + nu_neg @ ub_out``.

    This is the LOOSEST sound approach. Tighter alternatives:
      - Linear-in-x at argmax_LB (one-hot): tighter LB but needs cached argmax
        indices in forward (future enhancement).

    Sound because:
      LB(y) is sound: max(x_window) >= max_{k in window} lb[k] = bounds.lb[output_position]
      UB(y) is sound: max(x_window) <= max_{k in window} ub[k] = bounds.ub[output_position]
      Both are computed by forward_maxpool2d via F.max_pool2d on lb_in / ub_in.

    Lazy M-broadcast: bounds_dict[L.id] is at [B, *shape]; nu is at
    [B*M, *shape]. We broadcast the [B, 1, n] bounds against [B, M, n] nu.
    """
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_maxpool2d: layer {L.id} missing bounds in bounds_dict")

    BM = nu.shape[0]
    assert BM % M == 0, f"backward_maxpool2d: nu batch {BM} not divisible by M={M}"
    B_actual = BM // M

    v_flat = nu.flatten(start_dim=1)
    lb_out_flat = bounds.lb.flatten(start_dim=1)
    ub_out_flat = bounds.ub.flatten(start_dim=1)
    n = min(v_flat.shape[-1], lb_out_flat.shape[-1])
    if v_flat.shape[-1] != lb_out_flat.shape[-1]:
        v_flat = v_flat[..., :n]
        lb_out_flat = lb_out_flat[..., :n]
        ub_out_flat = ub_out_flat[..., :n]

    v = v_flat.view(B_actual, M, n)
    lb_b = lb_out_flat.unsqueeze(1)
    ub_b = ub_out_flat.unsqueeze(1)

    nu_pos = v.clamp(min=0)
    nu_neg = v.clamp(max=0)
    contrib_BMn = nu_pos * lb_b + nu_neg * ub_b
    contrib = contrib_BMn.sum(dim=-1).view(BM)

    input_shape = L.params.get("input_shape")
    if not isinstance(input_shape, (list, tuple)):
        raise ValueError(f"backward_maxpool2d: layer {L.id} missing/invalid 'input_shape' param")
    shape = list(input_shape)
    if len(shape) == 4:
        _, c_in, iH, iW = shape
    else:
        c_in, iH, iW = shape[-3:]
    n_in = int(c_in) * int(iH) * int(iW)
    v_out = torch.zeros(BM, n_in, dtype=nu.dtype, device=nu.device)

    assert len(preds) == 1, f"MAXPOOL2D expects 1 predecessor, got {len(preds)}"
    return [v_out], contrib


# ---- AVGPOOL2D ----
def forward_avgpool2d(
    L: Layer,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """AvgPool2D forward bounds (interval-only; dual-track resets).

    Source: tf_forward.py lines 454-458 (pre-refactor monolithic AVGPOOL2D
    branch).

    Analogous to :func:`forward_maxpool2d` using ``_fwd_avgpool2d``.
    ``stored`` equals ``out``.
    """
    assert len(parent_boxes) == 1, f"AVGPOOL2D expects 1 predecessor, got {len(parent_boxes)}"
    parent_box = parent_boxes[0]
    lb, ub = _fwd_avgpool2d(L, parent_box.lb, parent_box.ub)
    out = Bounds(lb, ub)
    stored = out
    lin, frame = _reset_forward_box(lb, ub, device, dtype)
    return stored, out, lin, frame


def backward_avgpool2d(L, nu, bounds_dict, preds, M: int = 1, alpha=None):
    """AvgPool2D backward — exact linear transpose.

    AvgPool is mathematically EXACT linear: ``y = (1/k²) · conv(x, ones_k)``,
    so the backward is mathematically exact (no relaxation):
    ``nu_in = (1/k²) · conv_transpose(nu_out, ones_k)``, per channel.
    contrib = 0 (no bias, no nonlinear gap).
    """
    kernel_size = L.params.get("kernel_size", 2)
    stride = L.params.get("stride", kernel_size)
    padding = L.params.get("padding", 0)
    input_shape = L.params.get("input_shape")
    if isinstance(kernel_size, (list, tuple)): kernel_size = int(kernel_size[0])
    if isinstance(stride, (list, tuple)): stride = int(stride[0])
    if isinstance(padding, (list, tuple)): padding = int(padding[0])
    kernel_size = int(kernel_size)
    stride = int(stride)
    padding = int(padding)

    if input_shape is None:
        raise ValueError(f"backward_avgpool2d: layer {L.id} missing 'input_shape' param")
    shape = list(input_shape)
    if len(shape) == 4:
        _, c, iH, iW = shape
    else:
        c, iH, iW = shape[-3:]
    c = int(c); iH = int(iH); iW = int(iW)

    oH = (iH + 2 * padding - kernel_size) // stride + 1
    oW = (iW + 2 * padding - kernel_size) // stride + 1

    BM = nu.shape[0]
    v_flat = nu.flatten(start_dim=1)
    n = v_flat.shape[-1]
    expected = c * oH * oW
    if n == expected:
        v_4d = v_flat.view(BM, c, oH, oW)
    elif n > expected:
        v_4d = v_flat[:, :expected].contiguous().view(BM, c, oH, oW)
    else:
        v_pad = torch.zeros(BM, expected, dtype=v_flat.dtype, device=v_flat.device)
        v_pad[:, :n] = v_flat
        v_4d = v_pad.view(BM, c, oH, oW)

    avg_weight = torch.full((c, 1, kernel_size, kernel_size),
                            1.0 / (kernel_size * kernel_size),
                            dtype=nu.dtype, device=nu.device)

    computed_h = (oH - 1) * stride - 2 * padding + kernel_size
    output_padding = max(0, iH - computed_h)

    v_out_4d = F.conv_transpose2d(v_4d, avg_weight, None,
                                  stride=stride, padding=padding,
                                  output_padding=output_padding,
                                  groups=c)
    v_out = v_out_4d.flatten(start_dim=1)
    contrib = torch.zeros(BM, dtype=nu.dtype, device=nu.device)

    assert len(preds) == 1, f"AVGPOOL2D expects 1 predecessor, got {len(preds)}"
    return [v_out], contrib


def _upsample_kwargs(L: Layer, mode: str, in_shape: Tuple[int, ...]) -> Dict[str, Any]:
    n_spatial = max(1, len(in_shape) - 2)
    size = L.params.get("size")
    scale = L.params.get("scale_factor")
    if isinstance(size, (list, tuple)) and len(size) > n_spatial:
        size = tuple(int(s) for s in size[-n_spatial:])
    if isinstance(scale, (list, tuple)) and len(scale) > n_spatial:
        scale = tuple(float(s) for s in scale[-n_spatial:])
    return {
        "size": size,
        "scale_factor": scale,
        "mode": mode,
        "align_corners": bool(L.params.get("align_corners", False)) if "linear" in mode else None,
    }


def forward_upsample(
    L: Layer, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """UPSAMPLE forward: interpolation is monotone (nearest replicates, linear is a
    positive-weight average), so interpolating lb/ub separately is the exact interval
    image; the dual frame resets over it (upsample grows n, so keep the track O(n))."""
    parent = parent_boxes[0]
    B = parent.lb.shape[0]
    in_shape = tuple(int(d) for d in cast(Tuple[int, ...], L.params["input_shape"]))
    mode = str(L.params.get("mode", "nearest"))
    kw = _upsample_kwargs(L, mode, in_shape)
    lb = F.interpolate(parent.lb.reshape(B, *in_shape[1:]), **kw).reshape(B, -1)
    ub = F.interpolate(parent.ub.reshape(B, *in_shape[1:]), **kw).reshape(B, -1)
    out = Bounds(lb, ub)
    lin, frame = _reset_forward_box(lb, ub, device, dtype)
    return out, out, lin, frame


def backward_upsample(L: Layer, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                      preds: List[int], M: int = 1, alpha=None
                      ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """UPSAMPLE backward: interpolation is a linear map, so its exact adjoint is the
    vjp of F.interpolate (autograd of a linear op computes its transpose). Sound for
    every mode; contrib zero."""
    assert len(preds) == 1, f"UPSAMPLE expects 1 predecessor, got {len(preds)}"
    in_shape = tuple(int(d) for d in cast(Tuple[int, ...], L.params["input_shape"]))
    mode = str(L.params.get("mode", "nearest"))
    kw = _upsample_kwargs(L, mode, in_shape)
    BM = nu.shape[0]
    x = torch.zeros(BM, *in_shape[1:], device=nu.device, dtype=nu.dtype, requires_grad=True)
    with torch.enable_grad():
        y = F.interpolate(x, **kw)
        (grad,) = torch.autograd.grad(y, x, grad_outputs=nu.reshape(y.shape))
    contrib = torch.zeros(BM, dtype=nu.dtype, device=nu.device)
    return [grad.reshape(BM, -1)], contrib
