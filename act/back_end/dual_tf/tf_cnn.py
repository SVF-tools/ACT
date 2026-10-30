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
from act.back_end.core import Bounds, Layer, ParamValue
from act.back_end.layer_schema import LayerKind
from act.back_end.interval_tf.tf_cnn import _conv1d_to_linear_matrix
from act.back_end.utils import avgpool2d_denominators, avgpool2d_output_hw, pair_2d
from .tf_mlp import forward_dense, backward_dense

from .tf_forward import (
    LinearBound, Frame,
    _fwd_conv2d, _fwd_conv2d_interval, _fwd_maxpool2d, _fwd_maxpool2d_lin,
    _fwd_avgpool2d, _reset_forward_box, _concretize, _intersect_boxes,
)


# Forward registry handlers. Each returns (stored, out, lin, frame).


def _conv1d_as_dense(layer: Layer, reference: torch.Tensor) -> Layer:
    """Use the existing Conv1d affine-map owner and dense frame machinery."""
    weight = layer.params["weight"]
    assert isinstance(weight, torch.Tensor)
    options: dict[str, int] = {}
    for name, default in (("stride", 1), ("padding", 0), ("dilation", 1), ("groups", 1)):
        value = layer.params.get(name, default)
        if isinstance(value, (tuple, list)):
            value, = value
        assert isinstance(value, int), f"CONV1D {name} must be an integer"
        options[name] = value
    output_shape = layer.get_tuple("output_shape")
    matrix = _conv1d_to_linear_matrix(
        weight.to(reference), layer.get_tuple("input_shape"), output_shape, **options,
    )
    params: dict[str, ParamValue] = {
        "weight": matrix, "in_features": matrix.shape[1], "out_features": matrix.shape[0],
    }
    bias = layer.params.get("bias")
    assert bias is None or isinstance(bias, torch.Tensor)
    if bias is not None:
        params["bias"] = bias.to(reference).repeat_interleave(output_shape[-1])
    return Layer(layer.id, LayerKind.DENSE.value, params, layer.in_vars, layer.out_vars)


def forward_conv1d(
    L: Layer, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Propagate Conv1d through its exact affine map, retaining valid frames."""
    return forward_dense(
        _conv1d_as_dense(L, parent_boxes[0].lb), parent_boxes, parent_lins,
        parent_frames, preds, post_activation, device, dtype,
    )


def backward_conv1d(
    L: Layer, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
    preds: List[int], M: int = 1, alpha=None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Exact Conv1d adjoint, with the solver's negative affine-bias convention."""
    return backward_dense(_conv1d_as_dense(L, nu), nu, bounds_dict, preds, M, alpha)


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
    concrete box via ``_reset_forward_box``. On the symbolic path the
    concretized box is intersected with the centre-radius interval image
    (mirroring :func:`~act.back_end.dual_tf.tf_mlp.forward_dense`): the raw
    concretization's lb/ub are two independently rounded conv accumulations,
    so at collapsed box widths they cross by a few ulps of the PARTIAL-sum
    magnitude (vgg16 late convs: partials 1e5-1e7, results O(1-100)), which
    the backward degenerate check rejects. ``_intersect_boxes`` both tightens
    and degrades such crossings to the structurally ordered interval box.
    ``stored`` equals ``out`` (CONV2D has no activation split).
    """
    assert len(parent_boxes) == 1, f"CONV2D expects 1 predecessor, got {len(parent_boxes)}"
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    frame = parent_frames[0]

    new_lin = _fwd_conv2d(L, parent_lin)
    if new_lin is None:
        lb, ub = _fwd_conv2d_interval(L, parent_box.lb, parent_box.ub)
        out = Bounds(lb, ub)
        stored = out
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
    else:
        lin = new_lin
        lin_lb, lin_ub = _concretize(lin, frame)
        int_lb, int_ub = _fwd_conv2d_interval(L, parent_box.lb, parent_box.ub)
        lb, ub = _intersect_boxes(lin_lb, lin_ub, int_lb, int_ub)
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
    input_shape: Optional[tuple[int, ...]] = None, output_shape: Optional[tuple[int, ...]] = None,
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
            iH = shape[-2]
        elif len(shape) == 3:
            iH = shape[-2]
        else:
            iH = None
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
    """MaxPool2D forward bounds (DeepPoly dual-track when the parent is symbolic).

    When the parent carries an explicit linear bound, ``_fwd_maxpool2d_lin``
    gathers each window's argmax-lb row (exact upper bound where that input
    dominates, constant interval max elsewhere) and the frame survives the
    pool. The argmax indices and dominance mask are cached on ``L.cache`` for
    the dual backward pass. On the lazy-identity path (parent ``A`` is None)
    the previous behaviour is kept: interval bounds via ``_fwd_maxpool2d``
    and a dual-track reset at the concrete box. ``stored`` equals ``out``.
    """
    assert len(parent_boxes) == 1, f"MAXPOOL2D expects 1 predecessor, got {len(parent_boxes)}"
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    result = _fwd_maxpool2d_lin(L, parent_lin, parent_box.lb, parent_box.ub, parent_frame)
    if result is None:
        lb, ub = _fwd_maxpool2d(L, parent_box.lb, parent_box.ub)
        out = Bounds(lb, ub)
        stored = out
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
        return stored, out, lin, frame
    lin, idx_flat, dominant, box_lb, box_ub = result
    # Exact comparisons that tolerate NaN bounds, which degrade to UNKNOWN downstream.
    assert torch.isclose(
        parent_box.lb.flatten(start_dim=1).gather(1, idx_flat), box_lb,
        rtol=0.0, atol=0.0, equal_nan=True,
    ).all(), f"MAXPOOL2D layer {L.id}: argmax index does not select the pooled lower bound"
    assert torch.isclose(
        parent_box.ub.flatten(start_dim=1).gather(1, idx_flat)[dominant], box_ub[dominant],
        rtol=0.0, atol=0.0, equal_nan=True,
    ).all(), f"MAXPOOL2D layer {L.id}: a dominant window's max upper bound is not at its argmax"

    lin_lb, lin_ub = _concretize(lin, parent_frame)
    lb, ub = _intersect_boxes(lin_lb, lin_ub, box_lb, box_ub)
    L.cache["maxpool_argmax_flat"] = idx_flat
    L.cache["maxpool_dominant"] = dominant
    # The mask is only valid for the box it was computed on. Cache the pooled
    # output box actually stored so the backward pass can reject a stale mask
    # left by a later forward on a different box (root_bounds_reuse modes run
    # backward with refresh_forward=False against earlier bounds_dicts).
    L.cache["maxpool_lb"] = lb.clone()
    L.cache["maxpool_ub"] = ub.clone()
    out = Bounds(lb, ub)
    stored = out
    return stored, out, lin, parent_frame


def _cache_box_matches(cached: Any, current: torch.Tensor) -> bool:
    """True iff the forward-cached pooled box bit-matches the caller's bounds.

    ``bounds_dict`` entries are ``stored.copy()`` clones, so identity checks
    cannot work; ``torch.equal`` on the ``[B, n_out]`` box is cheap. Shape and
    dtype are guarded first because ``torch.equal`` raises across devices and
    dtypes on some builds.
    """
    if not isinstance(cached, torch.Tensor):
        return False
    if cached.dtype != current.dtype or cached.shape != current.shape:
        return False
    return bool(torch.equal(cached.to(device=current.device), current))


def backward_maxpool2d(L, nu, bounds_dict, preds, M: int = 1, alpha=None):
    """MaxPool2D backward — one-hot pass-through at the argmax-lb input.

    Contract (shared by every dual backward handler, cf. the ReLU kernel):
    returns ``(pred_nus, contrib)`` with ``nu @ y >= pred_nu @ x + contrib``
    pointwise on the box, so the full backward pass yields a LOWER bound on
    ``c @ output``.

    Uses the cache written by :func:`forward_maxpool2d` via
    ``_fwd_maxpool2d_lin``: ``maxpool_argmax_flat`` (long ``[B, n_out]``, flat
    index into the pool input of each window's argmax-lb entry ``i*``) and
    ``maxpool_dominant`` (bool ``[B, n_out]``, true where ``lb_{i*} >=
    max_{j != i*} ub_j``). Per window and envelope side:

      - dominant: ``y = x_{i*}`` exactly, so ``nu`` (both signs) passes
        through one-hot to ``i*``; zero contribution.
      - non-dominant, ``nu >= 0`` (lower envelope): ``y >= x_{i*}`` pointwise
        and ``nu >= 0`` give ``nu*y >= nu*x_{i*}`` — pass through to ``i*``,
        zero contribution (matches the forward lower relaxation, which is the
        argmax-lb input's own linear bound).
      - non-dominant, ``nu < 0`` (upper envelope): the constant bound
        ``y <= ub_out`` gives ``nu*y >= nu*ub_out`` — contribution only.

    Each window feeds ``v_out`` OR ``contrib`` per envelope side, never both.
    Overlapping windows may target the same ``i*``; ``scatter_add_``
    accumulates them.

    Fallback (cache absent — forward ran the lazy-identity/interval path —
    cached shapes do not match ``[B, n_out]``, or the cached pooled box
    ``maxpool_lb``/``maxpool_ub`` is not bit-equal to ``bounds_dict[L.id]``,
    i.e. the mask is stale from a forward on a DIFFERENT box under
    root_bounds_reuse): the constant relaxation ``nu @ y = nu_pos @ y +
    nu_neg @ y >= nu_pos @ LB(y) + nu_neg @ UB(y)`` with ``contrib =
    nu_pos @ lb_out + nu_neg @ ub_out`` and ``v_out = 0``.

    Lazy M-broadcast: bounds_dict[L.id] and the cache are at [B, *shape]; nu
    is at [B*M, *shape]. We broadcast the [B, 1, n] tensors against [B, M, n].
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

    idx_flat = L.cache.get("maxpool_argmax_flat")
    dominant = L.cache.get("maxpool_dominant")
    cached_lb = L.cache.get("maxpool_lb")
    cached_ub = L.cache.get("maxpool_ub")
    cache_ok = (
        isinstance(idx_flat, torch.Tensor)
        and isinstance(dominant, torch.Tensor)
        and idx_flat.shape == (B_actual, n)
        and dominant.shape == (B_actual, n)
        and _cache_box_matches(cached_lb, bounds.lb.flatten(start_dim=1))
        and _cache_box_matches(cached_ub, bounds.ub.flatten(start_dim=1))
    )
    if cache_ok:
        dom = dominant.to(device=nu.device).unsqueeze(1)              # [B, 1, n]
        idx = idx_flat.to(device=nu.device).unsqueeze(1).expand(B_actual, M, n)
        pass_nu = torch.where(dom, v, v.clamp(min=0))                 # [B, M, n]
        v_out.view(B_actual, M, n_in).scatter_add_(2, idx, pass_nu)
        contrib_BMn = torch.where(dom, torch.zeros_like(v), v.clamp(max=0)) * ub_b
        contrib = contrib_BMn.sum(dim=-1).view(BM)
    else:
        # A missing or stale cache is legitimate (lazy forward, root_bounds_reuse): no assert.
        nu_pos = v.clamp(min=0)
        nu_neg = v.clamp(max=0)
        contrib_BMn = nu_pos * lb_b + nu_neg * ub_b
        contrib = contrib_BMn.sum(dim=-1).view(BM)

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

    AvgPool is linear. Scale each output coefficient by its window divisor,
    then apply the transpose of the unit-kernel sum, per channel.
    contrib = 0 (no bias, no nonlinear gap).
    """
    kernel_size = L.params.get("kernel_size", 2)
    stride = L.params.get("stride", kernel_size)
    padding = L.params.get("padding", 0)
    ceil_mode = bool(L.params.get("ceil_mode", False))
    count_include_pad = bool(L.params.get("count_include_pad", True))
    divisor_override = L.params.get("divisor_override")
    input_shape = L.params.get("input_shape")
    kernel_size = pair_2d(kernel_size)
    stride = pair_2d(stride)
    padding = pair_2d(padding)

    if input_shape is None:
        raise ValueError(f"backward_avgpool2d: layer {L.id} missing 'input_shape' param")
    shape = list(input_shape)
    if len(shape) == 4:
        _, c, iH, iW = shape
    else:
        c, iH, iW = shape[-3:]
    c = int(c); iH = int(iH); iW = int(iW)

    oH, oW = avgpool2d_output_hw(
        (iH, iW), kernel_size, stride, padding, ceil_mode
    )

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

    denominators = avgpool2d_denominators(
        (iH, iW),
        (oH, oW),
        kernel_size,
        stride,
        padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        device=nu.device,
        dtype=nu.dtype,
    )
    v_4d = v_4d / denominators.view(1, 1, oH, oW)
    avg_weight = torch.ones(
        (c, 1, kernel_size[0], kernel_size[1]),
        dtype=nu.dtype,
        device=nu.device,
    )

    computed_h = (oH - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
    computed_w = (oW - 1) * stride[1] - 2 * padding[1] + kernel_size[1]
    output_padding = (
        max(0, iH - computed_h),
        max(0, iW - computed_w),
    )
    if output_padding[0] >= stride[0] or output_padding[1] >= stride[1]:
        raise ValueError("avgpool2d transpose output padding is inconsistent")

    v_out_4d = F.conv_transpose2d(
        v_4d,
        avg_weight,
        None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=c,
    )
    v_out_4d = v_out_4d[..., :iH, :iW]
    if v_out_4d.shape[-2:] != (iH, iW):
        v_out_4d = F.pad(
            v_out_4d,
            (0, iW - v_out_4d.shape[-1], 0, iH - v_out_4d.shape[-2]),
        )
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
