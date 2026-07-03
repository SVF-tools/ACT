#===- act/back_end/dual_tf/tf_mlp.py - MLP Dual Transfer Functions ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# Batch-aware MLP backward kernels for linear-relaxation dual bound computation.
#
# Kernel convention (STRICT, batch-first):
#   nu      : Tensor[B, *layer_shape]   # dual variable, batch-first
#   v_out   : Tensor[B, *next_shape]
#   contrib : Tensor[B]                 # per-instance scalar
#
# ReLU uses FIXED upper-bound slope for crossing neurons.
#===---------------------------------------------------------------------===#

# Note: Gradient enablement for dual backward helpers is governed by the
# caller's torch.set_grad_enabled() context (see DualSolver.evaluate_spec).
# @torch.no_grad() decorators on these helpers were removed to allow
# gradient flow during robust training; verify_once / verify_bab paths
# remain under no_grad via their own outer guards.

import torch
from typing import Tuple, Optional, Dict, Any, List
from act.back_end.core import Bounds

from .tf_forward import (
    LinearBound, Frame,
    _fwd_dense, _fwd_relu, _fwd_bias, _fwd_scale, _fwd_bn, _fwd_lrelu,
    _concretize, _box_dense, _box_bias, _box_scale, _box_bn, _box_relu,
    _box_lrelu, _intersect_boxes, _reset_forward_box,
)

_DEGENERATE_TOL_ABS = 1e-3
_DEGENERATE_TOL_REL = 1e-3


def _repair_degenerate_interval(l: torch.Tensor, u: torch.Tensor, where: str) -> torch.Tensor:
    """Repair float32-rounding interval inversions (l slightly above u).

    A deep float32 forward can invert a near-stable neuron's pre-activation
    bounds by ~1e-6 - a degenerate (empty) interval that the harden/refine
    paths already repair via ``ub = max(ub, lb)``. This applies the same clamp
    at the backward consumption point so the kernel never sees l > u. An
    inversion beyond the float tolerance is a genuine soundness bug and raises.
    """
    gap = l - u
    tol = _DEGENERATE_TOL_ABS + _DEGENERATE_TOL_REL * torch.maximum(l.abs(), u.abs())
    if bool((gap > tol).any()):
        raise ValueError(
            f"{where}: lb exceeds ub by {float(gap.max()):.3e} (beyond float tolerance)"
        )
    return torch.maximum(u, l)


# Forward handlers: (L, parent_boxes, parent_lins, parent_frames, preds,
#   post_activation, device, dtype) -> (stored, out, lin, frame).
#   `parent_*` are parallel lists indexed by `preds`; unary handlers read [0].
#
# Backward handlers: (L, nu, bounds_dict, preds, M=1) -> (pred_nus, contrib).
#   Each pred_nus[i] is the ν routed to predecessor preds[i]. Unary layers
#   (DENSE, RELU, BIAS, SCALE, BN) return [nu_out]. backward_identity handles
#   both 0-pred (pure INPUT) and 1-pred (FLATTEN/RESHAPE/…) cases.


# ---- HELPERS ----
def _align(a: torch.Tensor, n: int) -> torch.Tensor:
    """Align 1-D parameter tensor to size n by truncating or tiling."""
    if a.numel() == n: return a.flatten()
    elif a.numel() > n: return a.flatten()[:n]
    else: return a.flatten().repeat((n + a.numel() - 1) // a.numel())[:n]


# ---- IDENTITY ----
def forward_identity(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Pass-through handler for INPUT / INPUT_SPEC / ASSERT / TRANSPOSE / SQUEEZE / UNSQUEEZE.

    Source: tf_forward.py lines 357-358 (INPUT family) and 460-461
    (ASSERT / TRANSPOSE / SQUEEZE / UNSQUEEZE family). Both branches are
    `pass`, so stored/out/lin/frame are whatever the predecessor produced.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    return parent_box, parent_box, parent_lin, parent_frame


def backward_identity(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                      preds: List[int], M: int = 1, alpha=None
                      ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    nu_out, contrib = dual_identity_backward(nu)
    # 0 preds (pure INPUT) -> []; 1 pred (FLATTEN/RESHAPE/…) -> [nu_out].
    return [nu_out] * len(preds), contrib


def dual_identity_backward(nu: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten/Reshape/Transpose backward: v_out = nu, contrib = zeros[B]."""
    B = nu.shape[0]
    contrib = torch.zeros(B, dtype=nu.dtype, device=nu.device)
    return nu, contrib


# ---- MEAN ----
def _mean_axes(L: Any) -> Tuple[Tuple[int, ...], Tuple[int, ...], bool, int]:
    """Resolve ``(in_shape, dims, keepdim, N)`` for a MEAN layer."""
    in_shape = tuple(int(d) for d in L.params["input_shape"])
    dims = tuple(int(d) for d in L.params["dim"])
    keepdim = bool(L.params.get("keepdim", 0))
    n = 1
    for d in dims:
        n *= in_shape[d]
    return in_shape, dims, keepdim, n


def forward_mean(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward box for MEAN. The average is monotone-linear, so the per-bound
    mean is the exact interval image; the dual frame resets over it."""
    parent = parent_boxes[0]
    B = parent.lb.shape[0]
    in_shape, dims, keepdim, _ = _mean_axes(L)
    batch_dims = [d + 1 for d in dims]
    lb = parent.lb.reshape(B, *in_shape).mean(dim=batch_dims, keepdim=keepdim).reshape(B, -1)
    ub = parent.ub.reshape(B, *in_shape).mean(dim=batch_dims, keepdim=keepdim).reshape(B, -1)
    lin, frame = _reset_forward_box(lb, ub, device, dtype)
    out = Bounds(lb, ub)
    return out, out, lin, frame


def backward_mean(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                  preds: List[int], M: int = 1, alpha=None
                  ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """MEAN backward: each input shares 1/N of its output's ν (transpose of the
    averaging), broadcast back over the reduced axes. Exact, contrib zero."""
    assert len(preds) == 1, f"MEAN expects 1 predecessor, got {len(preds)}"
    in_shape, dims, _, n = _mean_axes(L)
    BM = nu.shape[0]
    broadcast_shape = [s if i not in dims else 1 for i, s in enumerate(in_shape)]
    nu_in = (nu.reshape(BM, *broadcast_shape) / n).expand(BM, *in_shape).reshape(BM, -1)
    contrib = torch.zeros(BM, dtype=nu.dtype, device=nu.device)
    return [nu_in], contrib


def _reduce_sum_axes(L: Any):
    raw = L.params.get("input_shape")
    in_shape = tuple(int(s) for s in raw) if raw else None
    axes = L.params.get("axes")
    keepdim = bool(L.params.get("keepdims", 0))
    return in_shape, list(axes) if axes else None, keepdim


def forward_reduce_sum(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward box for REDUCE_SUM: the sum is monotone-linear, so summing the
    per-element lb/ub is the exact interval image; the dual frame resets over it."""
    parent = parent_boxes[0]
    B = parent.lb.shape[0]
    in_shape, axes, keepdim = _reduce_sum_axes(L)
    lb_in, ub_in = parent.lb, parent.ub
    if in_shape:
        lb_in = lb_in.reshape(B, *in_shape)
        ub_in = ub_in.reshape(B, *in_shape)
    dims = tuple(a + 1 for a in axes) if axes else tuple(range(1, lb_in.dim()))
    lb = lb_in.sum(dim=dims, keepdim=keepdim).reshape(B, -1)
    ub = ub_in.sum(dim=dims, keepdim=keepdim).reshape(B, -1)
    lin, frame = _reset_forward_box(lb, ub, device, dtype)
    out = Bounds(lb, ub)
    return out, out, lin, frame


def backward_reduce_sum(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                        preds: List[int], M: int = 1, alpha=None
                        ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """REDUCE_SUM backward: the transpose of a sum broadcasts each output's ν
    back to every input that fed it (no 1/N scaling, unlike MEAN). Exact, contrib zero."""
    assert len(preds) == 1, f"REDUCE_SUM expects 1 predecessor, got {len(preds)}"
    BM = nu.shape[0]
    in_shape, axes, _ = _reduce_sum_axes(L)
    if in_shape is None:
        n_in = bounds_dict[preds[0]].lb.flatten(start_dim=1).shape[-1]
        nu_in = nu.reshape(BM, -1)[:, :1].expand(BM, n_in)
        return [nu_in], torch.zeros(BM, dtype=nu.dtype, device=nu.device)
    dims0 = axes if axes else list(range(len(in_shape)))
    broadcast_shape = [s if i not in dims0 else 1 for i, s in enumerate(in_shape)]
    nu_in = nu.reshape(BM, *broadcast_shape).expand(BM, *in_shape).reshape(BM, -1)
    contrib = torch.zeros(BM, dtype=nu.dtype, device=nu.device)
    return [nu_in], contrib


# ---- RESHAPE ----
def forward_reshape(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for FLATTEN / RESHAPE.

    Source: tf_forward.py lines 420-422. Reshapes predecessor box lb/ub to
    ``[B, -1]`` and keeps lin/frame unchanged; downstream dense layers
    rematch the output-feature axis via ``_match_lin_input_dim``.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    B = parent_box.lb.shape[0]
    out = Bounds(parent_box.lb.reshape(B, -1), parent_box.ub.reshape(B, -1))
    stored = out
    return stored, out, parent_lin, parent_frame


# ---- DENSE ----
def forward_dense(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for DENSE (linear-relaxation dual-track + interval intersection).

    Source: tf_forward.py lines 371-378. Composes lin via ``_fwd_dense``,
    concretizes against the predecessor frame, intersects with the interval
    box update, and returns ``stored == out`` (no pre/post distinction for
    an affine layer). Frame passes through unchanged.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    prev_lb, prev_ub = parent_box.lb, parent_box.ub
    new_lin = _fwd_dense(L, parent_lin)
    int_lb, int_ub = _box_dense(L, prev_lb, prev_ub)
    if new_lin is None:
        lb, ub = int_lb, int_ub
        out = Bounds(lb, ub)
        stored = out
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
        return stored, out, lin, frame
    lin = new_lin
    lin_lb, lin_ub = _concretize(lin, x_L, x_U)
    lb, ub = _intersect_boxes(lin_lb, lin_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out
    return stored, out, lin, parent_frame


def backward_dense(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                   preds: List[int], M: int = 1, alpha=None
                   ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    nu_out, contrib = dual_dense_backward(nu, L.params["weight"], L.params.get("bias"))
    assert len(preds) == 1, f"DENSE expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_dense_backward(nu: torch.Tensor, W: torch.Tensor,
                        b: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched dense backward: v_out = nu @ W.

    nu : [B, out], W : [out, in] -> v_out: [B, in], contrib: [B].
    """
    assert W.dim() == 2, f"W must be 2D, got {W.shape}"
    assert nu.dim() >= 2, f"nu must be batched (>=2D), got {nu.shape}"
    B = nu.shape[0]
    nu_flat = nu.flatten(start_dim=1)
    assert nu_flat.shape[-1] == W.shape[0], \
        f"nu last dim {nu_flat.shape[-1]} != W.shape[0] {W.shape[0]}"
    v_out = nu_flat @ W
    if b is not None:
        contrib = -(nu_flat @ b.flatten())
    else:
        contrib = torch.zeros(B, dtype=nu.dtype, device=nu.device)
    return v_out, contrib


# ---- RELU / LRELU ----
def get_relu_masks(l: torch.Tensor, u: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Element-wise masks (on, off, amb); shape-preserving."""
    on, off = l >= 0, u <= 0
    return on, off, ~(on | off)


def forward_relu(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for RELU (linear relaxation + interval intersection).

    Source: tf_forward.py lines 360-369. When ``post_activation`` is True,
    ``stored == out`` (post-ReLU box) and lin/frame are reset to identity
    over the new concrete box via ``_reset_forward_box``. Otherwise
    ``stored`` is the pre-activation box and lin/frame pass through.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    new_lin = _fwd_relu(parent_lin, pre_lb, pre_ub)
    int_lb, int_ub = _box_relu(pre_lb, pre_ub)
    if new_lin is None:
        lb, ub = int_lb, int_ub
        out = Bounds(lb, ub)
        stored = out if post_activation else Bounds(pre_lb, pre_ub)
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
        return stored, out, lin, frame
    lin = new_lin
    lin_lb, lin_ub = _concretize(lin, x_L, x_U)
    lb, ub = _intersect_boxes(lin_lb, lin_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    frame = parent_frame
    if post_activation:
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
    return stored, out, lin, frame


def forward_lrelu(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for LRELU / LEAKY_RELU (triangle linear relaxation).

    Source: tf_forward.py lines 436-446. Reads ``alpha`` from
    ``L.params.get("alpha", 0.01)``. ``post_activation`` handling mirrors
    :func:`forward_relu`: when True ``stored == out`` and lin/frame are
    reset to identity on the new box; otherwise ``stored`` is the
    pre-activation box and lin/frame pass through.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    alpha = L.params.get("alpha", 0.01)
    lin = _fwd_lrelu(parent_lin, pre_lb, pre_ub, alpha)
    int_lb, int_ub = _box_lrelu(pre_lb, pre_ub, alpha)
    if lin is None:
        out = Bounds(int_lb, int_ub)
        stored = out if post_activation else Bounds(pre_lb, pre_ub)
        lin, frame = _reset_forward_box(int_lb, int_ub, device, dtype)
        return stored, out, lin, frame
    lin_lb, lin_ub = _concretize(lin, x_L, x_U)
    lb, ub = _intersect_boxes(lin_lb, lin_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    frame = parent_frame
    if post_activation:
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
    return stored, out, lin, frame


def backward_relu(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                  preds: List[int], M: int = 1, alpha=None
                  ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_relu: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_relu_backward(nu, bounds, M, alpha=alpha)
    assert len(preds) == 1, f"RELU expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_relu_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1,
                       alpha: Optional[torch.Tensor] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward kernel for ReLU with optional Lagrange-relaxed lower slope.

    When alpha is None: use chord slope u/(u-l) on both lower and upper
    envelopes (fixed-slope dual relaxation).
    When alpha is provided shape [B, M, n] (per-spec optimizable lower slope):
        v >= 0 (LB env): slope = alpha[b, m, i]   (α ∈ [0, 1], optimizable)
        v <  0 (UB env): slope = u/(u-l)          (chord, unchanged)

    Crossing intercept: -slope * l where slope = chord, applied only where v < 0.
    Preserves lazy-M-broadcast: bounds [B, n] broadcast against nu [B, M, n].
    """
    BM = nu.shape[0]
    assert BM % M == 0, f"dual_relu_backward: nu batch {BM} not divisible by M={M}"
    B = BM // M

    v_flat = nu.flatten(start_dim=1)
    l_B = bounds.lb.flatten(start_dim=1)
    u_B = bounds.ub.flatten(start_dim=1)
    n = min(v_flat.shape[-1], l_B.shape[-1])
    if v_flat.shape[-1] != l_B.shape[-1]:
        v_flat = v_flat[..., :n]
        l_B = l_B[..., :n]
        u_B = u_B[..., :n]
        if alpha is not None and alpha.shape[-1] != n:
            alpha = alpha[..., :n]
    u_B = _repair_degenerate_interval(l_B, u_B, "dual_relu_backward")

    v = v_flat.view(B, M, n)
    l = l_B.unsqueeze(1)
    u = u_B.unsqueeze(1)

    on, off, amb = get_relu_masks(l, u)
    if amb.any():
        denom = (u - l).clamp(min=1e-12)
        slope = u / denom

        if alpha is not None:
            assert alpha.dim() >= 2, f"alpha must be batched, got {alpha.shape}"
            if alpha.dim() == 2:
                alpha_bmn = alpha.unsqueeze(1).expand(B, M, n)
            else:
                alpha_bmn = alpha
            d_lower = alpha_bmn
            d_upper_bmn = slope.expand(B, M, n)
            d = torch.where(v >= 0, d_lower, d_upper_bmn)
            d = torch.where(on.expand_as(d), torch.ones_like(d), d)
            d = torch.where(off.expand_as(d), torch.zeros_like(d), d)
        else:
            d = torch.zeros_like(v)
            d = torch.where(on, torch.ones_like(d), d)
            d = torch.where(amb, slope.expand_as(v), d)

        intercept = -slope * l
        contrib_per = torch.where(amb, v.clamp(max=0) * intercept,
                                  torch.zeros_like(v))
        contrib = contrib_per.sum(dim=-1).view(BM)
    else:
        d = torch.zeros_like(v)
        d = torch.where(on, torch.ones_like(d), d)
        contrib = torch.zeros(BM, dtype=nu.dtype, device=nu.device)

    v_out = d * v
    return v_out.view(BM, n), contrib


# ---- SIGN ----
def forward_sign(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for SIGN (step function y = sign(x) in {-1, 0, 1}).

    sign is monotone non-decreasing, so the sound interval box is exactly
    [sign(lb), sign(ub)]. The step has zero slope everywhere, so no linear
    plane is carried: the frame resets to identity over the new box (like the
    RELU no-linearisation branch).
    """
    pre = parent_boxes[0]
    lb = torch.sign(pre.lb)
    ub = torch.sign(pre.ub)
    out = Bounds(lb, ub)
    lin, frame = _reset_forward_box(lb, ub, device, dtype)
    return out, out, lin, frame


def backward_sign(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                  preds: List[int], M: int = 1, alpha=None
                  ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Backward for SIGN: a step relaxed by two ZERO-slope constant planes.

    Sound relaxation over [l, u]: sign(l) <= sign(x) <= sign(u) (constants,
    since sign is monotone). Both planes have slope 0, so no ν flows to the
    input (nu_in = 0) and each output contributes its constant plane, selected
    by the sign of ν (>=0 -> lower plane sign(l), <0 -> upper plane sign(u)).
    The stored box already holds sign(l)/sign(u). Fully batched over B and M.
    """
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_sign: layer {L.id} missing bounds in bounds_dict")
    assert len(preds) == 1, f"SIGN expects 1 predecessor, got {len(preds)}"
    BM = nu.shape[0]
    assert BM % M == 0, f"backward_sign: nu batch {BM} not divisible by M={M}"
    B = BM // M
    v_flat = nu.flatten(start_dim=1)
    lo_B = bounds.lb.flatten(start_dim=1)
    hi_B = bounds.ub.flatten(start_dim=1)
    n = min(v_flat.shape[-1], lo_B.shape[-1])
    v = v_flat[..., :n].view(B, M, n)
    lo = lo_B[..., :n].unsqueeze(1)
    hi = hi_B[..., :n].unsqueeze(1)
    const = torch.where(v >= 0, lo.expand_as(v), hi.expand_as(v))
    contrib = (v * const).sum(dim=-1).reshape(BM)
    nu_in = torch.zeros(BM, n, dtype=nu.dtype, device=nu.device)
    return [nu_in], contrib


# ---- BIAS ----
def forward_bias(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for BIAS (``y = x + c``).

    Source: tf_forward.py lines 393-400. Composes via ``_fwd_bias``,
    concretizes, intersects with interval box update; ``stored == out``.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    prev_lb, prev_ub = parent_box.lb, parent_box.ub
    lin = _fwd_bias(L, parent_lin)
    lin_lb, lin_ub = _concretize(lin, x_L, x_U)
    int_lb, int_ub = _box_bias(L, prev_lb, prev_ub)
    lb, ub = _intersect_boxes(lin_lb, lin_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out
    return stored, out, lin, parent_frame


def backward_bias(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                  preds: List[int], M: int = 1, alpha=None
                  ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    nu_out, contrib = dual_bias_backward(nu, L.params["c"])
    assert len(preds) == 1, f"BIAS expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_bias_backward(nu: torch.Tensor, c: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """y = x + c ; v_out = nu, contrib = -(v * c_flat).sum(dim=-1)."""
    v = nu.flatten(start_dim=1)
    c_flat = _align(c, v.shape[-1])
    contrib = -(v * c_flat).sum(dim=-1)                          # [B]
    return nu, contrib


# ---- SCALE ----
def forward_scale(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for SCALE (``y = a * x``, element-wise).

    Source: tf_forward.py lines 402-409. Composes via ``_fwd_scale``,
    concretizes, intersects with interval box update; ``stored == out``.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    prev_lb, prev_ub = parent_box.lb, parent_box.ub
    lin = _fwd_scale(L, parent_lin)
    int_lb, int_ub = _box_scale(L, prev_lb, prev_ub)
    if lin is None:
        out = Bounds(int_lb, int_ub)
        lin, frame = _reset_forward_box(int_lb, int_ub, device, dtype)
        return out, out, lin, frame
    lin_lb, lin_ub = _concretize(lin, x_L, x_U)
    lb, ub = _intersect_boxes(lin_lb, lin_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out
    return stored, out, lin, parent_frame


def backward_scale(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                   preds: List[int], M: int = 1, alpha=None
                   ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    nu_out, contrib = dual_scale_backward(nu, L.params["a"])
    assert len(preds) == 1, f"SCALE expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_scale_backward(nu: torch.Tensor, a: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """y = a * x ; v_out = a * nu, contrib = 0."""
    B = nu.shape[0]
    flat = nu.flatten(start_dim=1)
    a_aligned = _align(a, flat.shape[-1])
    out = (a_aligned * flat).view(nu.shape)
    contrib = torch.zeros(B, dtype=nu.dtype, device=nu.device)
    return out, contrib


# ---- BN ----
def forward_bn(
    L: Any,
    parent_boxes: List[Bounds],
    parent_lins: List[LinearBound],
    parent_frames: List[Frame],
    preds: List[int],
    post_activation: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward handler for BN (``y = A * x + c``, element-wise).

    Source: tf_forward.py lines 411-418. Composes via ``_fwd_bn``,
    concretizes, intersects with interval box update; ``stored == out``.
    """
    parent_box = parent_boxes[0]
    parent_lin = parent_lins[0]
    parent_frame = parent_frames[0]
    x_L, x_U = parent_frame
    prev_lb, prev_ub = parent_box.lb, parent_box.ub
    lin = _fwd_bn(L, parent_lin)
    int_lb, int_ub = _box_bn(L, prev_lb, prev_ub)
    if lin is None:
        out = Bounds(int_lb, int_ub)
        lin, frame = _reset_forward_box(int_lb, int_ub, device, dtype)
        return out, out, lin, frame
    lin_lb, lin_ub = _concretize(lin, x_L, x_U)
    lb, ub = _intersect_boxes(lin_lb, lin_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out
    return stored, out, lin, parent_frame


def backward_bn(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                preds: List[int], M: int = 1, alpha=None
                ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    nu_out, contrib = dual_bn_backward(nu, L.params["A"], L.params["c"])
    assert len(preds) == 1, f"BN expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_bn_backward(nu: torch.Tensor, A: torch.Tensor, c: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """y = A*x + c ; v_out = A*nu, contrib = -(v * c_flat).sum(dim=-1)."""
    B = nu.shape[0]
    v = nu.flatten(start_dim=1)
    A_aligned = _align(A, v.shape[-1])
    c_aligned = _align(c, v.shape[-1])
    out = (A_aligned * v).view(nu.shape)
    contrib = -(v * c_aligned).sum(dim=-1)                       # [B]
    return out, contrib
