#===- act/back_end/dual_tf/tf_smooth.py - Smooth Activation Dual TF -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# Batch-aware smooth (S-shaped) activation backward.
# nu: [B, *shape] -> v_out: [B, *shape], contrib: [B].
#===---------------------------------------------------------------------===#

# Note: Gradient enablement for dual backward helpers is governed by the
# caller's torch.set_grad_enabled() context (see DualSolver.evaluate_spec).
# @torch.no_grad() decorators on these helpers were removed to allow
# gradient flow during robust training; verify_once / verify_bab paths
# remain under no_grad via their own outer guards.

import math

import torch
from typing import Tuple, Callable, Dict, Any, List, Final
from act.back_end.core import Bounds
from act.back_end.interval_tf.tf_mlp import _sin_interval, _cos_interval, _quantize_params_flat, _quantize_qdq_value
from .tf_forward import (
    LinearBound, Frame, _reset_forward_box, _concretize,
    _fwd_elementwise_planes, _intersect_boxes,
)


# ---- Shared primitives ----

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)

def dsigmoid(x: torch.Tensor) -> torch.Tensor:
    # Avoid cancellation in 1-sigmoid(x) on the positive saturated side.
    return torch.sigmoid(x) * torch.sigmoid(-x)

def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)

def dtanh(x: torch.Tensor) -> torch.Tensor:
    return 1 - torch.tanh(x) ** 2

def erf(x: torch.Tensor) -> torch.Tensor:
    return torch.erf(x)

def derf(x: torch.Tensor) -> torch.Tensor:
    return (2.0 / math.sqrt(math.pi)) * torch.exp(-x * x)


def _dual_constant_box_backward(nu: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, M: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    BM = nu.shape[0]
    assert BM % M == 0, f"constant-box backward: nu batch {BM} not divisible by M={M}"
    B = BM // M
    v_flat = nu.flatten(start_dim=1)
    l_B = lower.flatten(start_dim=1) if lower.dim() >= 2 else lower.flatten().unsqueeze(0).expand(B, -1)
    u_B = upper.flatten(start_dim=1) if upper.dim() >= 2 else upper.flatten().unsqueeze(0).expand(B, -1)
    n = min(v_flat.shape[-1], l_B.shape[-1])
    if v_flat.shape[-1] != l_B.shape[-1]:
        v_flat = v_flat[..., :n]
        l_B = l_B[..., :n]
        u_B = u_B[..., :n]
    l = l_B.unsqueeze(1)
    u = u_B.unsqueeze(1)
    v = v_flat.view(B, M, n)
    b = torch.where(v >= 0, l, u)
    v_out = torch.zeros_like(v)
    contrib = (v * b).sum(dim=-1).view(BM)
    return v_out.view(BM, n), contrib


_SMOOTH_TANGENT_BISECTIONS: Final[int] = 40


def compute_smooth_relaxation(
    l: torch.Tensor, u: torch.Tensor,
    func: Callable[[torch.Tensor], torch.Tensor],
    dfunc: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enclosing planes for increasing, convex-then-concave sigmoid/tanh/erf.

    A one-sided interval uses a midpoint tangent and a chord on the appropriate
    curvature sides. Across zero, endpoint-anchored tangents are bracketed;
    reading the larger derivative at each bracket keeps the plane enclosing.
    Works element-wise on any broadcastable shape (including batched [B, n]).
    """
    from .tf_mlp import _repair_degenerate_interval
    u = _repair_degenerate_interval(l, u, "smooth_relaxation")
    f_l, f_u = func(l), func(u)
    width = u - l
    # Every positive width needs its actual secant. A singleton instead has
    # a zero numerator and a constant chord; only that denominator is replaced.
    k_chord = (f_u - f_l) / torch.where(width > 0, width, torch.ones_like(width))
    b_chord = f_l - k_chord * l
    midpoint = (l + u) * 0.5
    k_tangent = dfunc(midpoint)
    b_tangent = func(midpoint) - k_tangent * midpoint
    convex, concave = u <= 0, l >= 0
    crossing = ~(convex | concave)
    k_lower = torch.where(convex, k_tangent, k_chord)
    b_lower = torch.where(convex, b_tangent, b_chord)
    k_upper = torch.where(concave, k_tangent, k_chord)
    b_upper = torch.where(concave, b_tangent, b_chord)

    if bool(crossing.any()):
        l_cross = torch.where(crossing, l, -1.0)
        u_cross = torch.where(crossing, u, 1.0)
        f_left, f_right = func(l_cross), func(u_cross)
        # The upper plane is anchored at l; f' decreases on its positive
        # tangency bracket, so its left endpoint supplies a safe slope.
        lo, hi = torch.zeros_like(u_cross), u_cross
        for _ in range(_SMOOTH_TANGENT_BISECTIONS):
            mid = (lo + hi) * 0.5
            residual = dfunc(mid) * (mid - l_cross) - (func(mid) - f_left)
            lo, hi = torch.where(residual >= 0, mid, lo), torch.where(residual >= 0, hi, mid)
        chord_upper = dfunc(u_cross) * (u_cross - l_cross) >= f_right - f_left
        k_hi = torch.where(chord_upper, k_chord, dfunc(lo))

        # The lower plane is anchored at u; f' increases on the negative
        # bracket, so use its right endpoint. Non-crossing entries are ignored.
        lo, hi = l_cross, torch.zeros_like(l_cross)
        for _ in range(_SMOOTH_TANGENT_BISECTIONS):
            mid = (lo + hi) * 0.5
            residual = f_right - func(mid) - dfunc(mid) * (u_cross - mid)
            lo, hi = torch.where(residual >= 0, mid, lo), torch.where(residual >= 0, hi, mid)
        chord_lower = f_right - f_left <= dfunc(l_cross) * (u_cross - l_cross)
        k_lo = torch.where(chord_lower, k_chord, dfunc(hi))
        k_lower = torch.where(crossing, k_lo, k_lower)
        b_lower = torch.where(crossing, f_u - k_lo * u, b_lower)
        k_upper = torch.where(crossing, k_hi, k_upper)
        b_upper = torch.where(crossing, f_l - k_hi * l, b_upper)

    return k_lower, b_lower, k_upper, b_upper


def dual_smooth_backward(
    nu: torch.Tensor, bounds: Bounds,
    func: Callable[[torch.Tensor], torch.Tensor],
    dfunc: Callable[[torch.Tensor], torch.Tensor],
    M: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched smooth activation backward (sigmoid/tanh) with lazy M-broadcast.

    Same broadcast pattern as :func:`dual_relu_backward`: relaxation
    coefficients ``(k_lower, b_lower, k_upper, b_upper)`` depend on the
    bounds only (spec-agnostic) and are computed once at ``[B, 1, n]``,
    then broadcast against ``nu`` viewed at ``[B, M, n]``.

    Args:
        nu: dual variable, shape ``[B*M, *shape]``.
        bounds: layer bounds, shape ``[B, *shape]``. NOT M-expanded.
        func/dfunc: activation function and its derivative.
        M: spec-row multiplicity (default 1).
    """
    BM = nu.shape[0]
    assert BM % M == 0, f"dual_smooth_backward: nu batch {BM} not divisible by M={M}"
    B = BM // M

    v_flat = nu.flatten(start_dim=1)                              # [BM, n]
    l_B = bounds.lb.flatten(start_dim=1) if bounds.lb.dim() >= 2 \
          else bounds.lb.flatten().unsqueeze(0).expand(B, -1)
    u_B = bounds.ub.flatten(start_dim=1) if bounds.ub.dim() >= 2 \
          else bounds.ub.flatten().unsqueeze(0).expand(B, -1)
    n = min(v_flat.shape[-1], l_B.shape[-1])
    if v_flat.shape[-1] != l_B.shape[-1]:
        v_flat = v_flat[..., :n]
        l_B = l_B[..., :n]
        u_B = u_B[..., :n]

    l = l_B.unsqueeze(1)                                          # [B, 1, n]
    u = u_B.unsqueeze(1)                                          # [B, 1, n]
    k_lower, b_lower, k_upper, b_upper = compute_smooth_relaxation(l, u, func, dfunc)

    v = v_flat.view(B, M, n)                                      # [B, M, n] view
    v_pos = v >= 0                                                # [B, M, n]
    k = torch.where(v_pos, k_lower, k_upper)                      # broadcast → [B, M, n]
    b = torch.where(v_pos, b_lower, b_upper)

    v_out = v * k                                                 # [B, M, n]
    contrib = (v * b).sum(dim=-1).view(BM)                        # [BM]
    return v_out.view(BM, n), contrib


def _forward_smooth_planes(
    parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], post_activation: bool,
    int_lb: torch.Tensor, int_ub: torch.Tensor,
    planes: Callable[[torch.Tensor, torch.Tensor],
                     Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Compose valid scalar envelopes while retaining the caller's grad context."""
    parent, relation = parent_boxes[0], parent_lins[0]
    lin = None
    if relation.A_lb is not None and relation.A_ub is not None:
        lin = _fwd_elementwise_planes(relation, planes(parent.lb, parent.ub))
    lb, ub = int_lb, int_ub
    if lin is not None:
        linear_lb, linear_ub = _concretize(lin, parent_frames[0])
        lb, ub = _intersect_boxes(linear_lb, linear_ub, int_lb, int_ub)
    out = Bounds(lb, ub)
    stored = out if post_activation else Bounds(parent.lb, parent.ub)
    if lin is not None and not post_activation:
        return stored, out, lin, parent_frames[0]
    lin, frame = _reset_forward_box(lb, ub, lb.device, lb.dtype)
    return stored, out, lin, frame


# ---- SIGMOID ----

@torch.no_grad()
def forward_sigmoid(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Propagate sigmoid bounds and compose its planes on an explicit frame."""
    parent_box = parent_boxes[0]
    return _forward_smooth_planes(
        parent_boxes, parent_lins, parent_frames, post_activation,
        torch.sigmoid(parent_box.lb), torch.sigmoid(parent_box.ub),
        lambda l, u: compute_smooth_relaxation(l, u, sigmoid, dsigmoid),
    )


def backward_sigmoid(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                     preds: List[int], M: int = 1, alpha=None
                     ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_sigmoid: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_sigmoid_backward(nu, bounds, M)
    assert len(preds) == 1, f"SIGMOID expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_sigmoid_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    return dual_smooth_backward(nu, bounds, sigmoid, dsigmoid, M)


# ---- TANH ----

@torch.no_grad()
def forward_tanh(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Propagate tanh bounds and compose its planes on an explicit frame."""
    parent_box = parent_boxes[0]
    return _forward_smooth_planes(
        parent_boxes, parent_lins, parent_frames, post_activation,
        torch.tanh(parent_box.lb), torch.tanh(parent_box.ub),
        lambda l, u: compute_smooth_relaxation(l, u, tanh, dtanh),
    )


def backward_tanh(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                  preds: List[int], M: int = 1, alpha=None
                  ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_tanh: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_tanh_backward(nu, bounds, M)
    assert len(preds) == 1, f"TANH expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_tanh_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    return dual_smooth_backward(nu, bounds, tanh, dtanh, M)


# ---- ERF ----

@torch.no_grad()
def forward_erf(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward pass for ERF activation, mirroring TANH."""
    parent_box = parent_boxes[0]
    return _forward_smooth_planes(
        parent_boxes, parent_lins, parent_frames, post_activation,
        torch.erf(parent_box.lb), torch.erf(parent_box.ub),
        lambda l, u: compute_smooth_relaxation(l, u, erf, derf),
    )


def backward_erf(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                 preds: List[int], M: int = 1, alpha=None
                 ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_erf: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_erf_backward(nu, bounds, M)
    assert len(preds) == 1, f"ERF expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_erf_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    return dual_smooth_backward(nu, bounds, erf, derf, M)


# ---- SQRT ----

@torch.no_grad()
def forward_sqrt(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    lo_e = torch.clamp(pre_lb, min=0.0)
    hi_e = torch.clamp(pre_ub, min=0.0)
    out = Bounds(torch.sqrt(lo_e), torch.sqrt(hi_e))
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_sqrt(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                  preds: List[int], M: int = 1, alpha=None
                  ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_sqrt: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_sqrt_backward(nu, bounds, M)
    assert len(preds) == 1, f"SQRT expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_sqrt_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    lo_e = torch.clamp(bounds.lb, min=0.0)
    hi_e = torch.clamp(bounds.ub, min=0.0)
    return _dual_constant_box_backward(nu, torch.sqrt(lo_e), torch.sqrt(hi_e), M)


@torch.no_grad()
def forward_square(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Exact interval image of x squared, stored as an output box."""
    lower, upper = parent_boxes[0].lb, parent_boxes[0].ub
    lb = torch.where(
        (lower <= 0) & (upper >= 0), torch.zeros_like(lower),
        torch.minimum(lower.square(), upper.square()),
    )
    ub = torch.maximum(lower.square(), upper.square())
    out = Bounds(lb, ub)
    lin, frame = _reset_forward_box(lb, ub, device, dtype)
    return out, out, lin, frame


def backward_square(
    L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
    preds: List[int], M: int = 1, alpha=None,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Absorb square's exact output-box support without an unsound derivative."""
    assert len(preds) == 1, f"SQUARE expects 1 predecessor, got {len(preds)}"
    out = bounds_dict[L.id]
    nu_in, contribution = _dual_constant_box_backward(nu, out.lb, out.ub, M)
    return [nu_in], contribution


# ---- SIN ----

@torch.no_grad()
def forward_sin(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    out = _sin_interval(pre_lb, pre_ub)
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_sin(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                 preds: List[int], M: int = 1, alpha=None
                 ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_sin: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_sin_backward(nu, bounds, M)
    assert len(preds) == 1, f"SIN expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_sin_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    box = _sin_interval(bounds.lb, bounds.ub)
    return _dual_constant_box_backward(nu, box.lb, box.ub, M)


# ---- COS ----

@torch.no_grad()
def forward_cos(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    out = _cos_interval(pre_lb, pre_ub)
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_cos(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                 preds: List[int], M: int = 1, alpha=None
                 ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_cos: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_cos_backward(nu, bounds, M)
    assert len(preds) == 1, f"COS expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_cos_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    box = _cos_interval(bounds.lb, bounds.ub)
    return _dual_constant_box_backward(nu, box.lb, box.ub, M)


# ---- QUANTIZE / QDQ real-valued map ----

@torch.no_grad()
def forward_quantize(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    n = pre_lb.numel()
    scale, zp, qmin, qmax = _quantize_params_flat(L, n, pre_lb.device, pre_lb.dtype)
    lo = pre_lb.reshape(-1)
    hi = torch.maximum(pre_ub.reshape(-1), lo)
    z_lo = _quantize_qdq_value(lo, scale, zp, qmin, qmax)
    z_hi = _quantize_qdq_value(hi, scale, zp, qmin, qmax)
    out = Bounds(torch.minimum(z_lo, z_hi).reshape_as(pre_lb), torch.maximum(z_lo, z_hi).reshape_as(pre_ub))
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_quantize(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                      preds: List[int], M: int = 1, alpha=None
                      ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_quantize: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_quantize_backward(L, nu, bounds, M)
    assert len(preds) == 1, f"QUANTIZE expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_quantize_backward(L: Any, nu: torch.Tensor, bounds: Bounds, M: int = 1):
    n = bounds.lb.numel()
    scale, zp, qmin, qmax = _quantize_params_flat(L, n, bounds.lb.device, bounds.lb.dtype)
    lo = bounds.lb.reshape(-1)
    hi = torch.maximum(bounds.ub.reshape(-1), lo)
    z_lo = _quantize_qdq_value(lo, scale, zp, qmin, qmax).reshape_as(bounds.lb)
    z_hi = _quantize_qdq_value(hi, scale, zp, qmin, qmax).reshape_as(bounds.ub)
    return _dual_constant_box_backward(nu, torch.minimum(z_lo, z_hi), torch.maximum(z_lo, z_hi), M)


if __name__ == "__main__":
    from act.util.device_manager import initialize_device, get_default_device, get_default_dtype

    for dtype_name in ("float32", "float64"):
        initialize_device("cpu", dtype_name)
        device, dtype = get_default_device(), get_default_dtype()
        lower = torch.tensor(
            [-3., 1., -1., -5., -.1, 0., -5e-13, 1e-13, -2e-13],
            device=device, dtype=dtype,
        )
        upper = torch.tensor(
            [-1., 3., 2., .1, 5., 0., -1e-13, 5e-13, 5e-13],
            device=device, dtype=dtype,
        )
        fractions = torch.linspace(0., 1., 9, device=device, dtype=dtype)
        points = lower[:, None] + (upper - lower)[:, None] * fractions
        tolerance = 64 * torch.finfo(dtype).eps
        for function, derivative in ((sigmoid, dsigmoid), (tanh, dtanh), (erf, derf)):
            k_lo, b_lo, k_hi, b_hi = compute_smooth_relaxation(lower, upper, function, derivative)
            concrete = function(points)
            scale = torch.maximum(function(lower).abs(), function(upper).abs()).clamp_min(torch.finfo(dtype).tiny)
            assert ((k_lo[:, None] * points + b_lo[:, None]) <= concrete + tolerance * scale[:, None]).all()
            assert ((k_hi[:, None] * points + b_hi[:, None]) >= concrete - tolerance * scale[:, None]).all()
