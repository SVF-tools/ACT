#===- act/back_end/dual_tf/tf_transformer.py - Transformer Dual TFs -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Dual/CROWN transformer transfer functions: real forward/backward handlers
#   for GELU, SOFTMAX, LAYERNORM and the bilinear attention cores ATT_SCORES
#   (Q Kᵀ) and ATT_MIX (probs · V), all feeding the general dual backward solver.
#
#   GELU, SOFTMAX and LAYERNORM have real dual forward/backward handlers. GELU
#   is an elementwise curvature-aware relaxation of the exact erf activation.
#   SOFTMAX and LAYERNORM are single-input but vector-coupled: their backward
#   kernels re-derive sound per-output linear planes locally from the layer's
#   input box and route the dual variable through the transpose of those planes
#   (softmax = exp / row-sum reciprocal; layernorm = mean-subtract / variance /
#   rsqrt / scale-shift). All three use a fixed slope (no per-neuron alpha is
#   allocated for them yet) and feed the general dual backward solver.
#
#   The bilinear attention cores ATT_SCORES (Q Kᵀ) and ATT_MIX (probs · V) have
#   real dual forward/backward handlers: the backward kernel re-derives the local
#   McCormick planes from the per-input boxes in bounds_dict, fuses the two valid
#   planes by an optimizable slope α ∈ [0, 1] (a convex combination, hence sound
#   for any α), and sign-splits the dual variable across the two inputs as
#   DISTINCT tensors. MHA projection roles and joins are exact affine maps;
#   MASK_ADD remains a stub.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import math
from typing import Callable

import torch

from act.back_end.core import Bounds, Layer
from act.back_end.interval_tf.tf_transformer import softmax_rowsize, _layernorm_box, _gelu_box
from act.back_end.utils import four_corner_envelope
from act.back_end.interval_tf.tf_attention import (
    _GELU_INFLECTION,
    _GELU_MIN_X,
    _GELU_MIN_Y,
    LinearBounds,
    rule_based_alpha,
)


def forward_mha(L, parent_boxes, parent_lins, parent_frames, preds,
                post_activation, device, dtype):
    """MASK_ADD forward remains unsupported."""
    raise NotImplementedError("forward for MASK_ADD not implemented in dual_tf")


def backward_mha(L, nu, bounds_dict, preds, M: int = 1, alpha=None):
    """MASK_ADD backward remains unsupported."""
    raise NotImplementedError("backward for MASK_ADD not implemented in dual_tf")


def _mha_projection(
    layer: Layer, input_width: int, reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select token coordinates and compact projection rows for an MHA role."""
    weight = layer.params["weight"]
    assert isinstance(weight, torch.Tensor)
    weight = weight.to(reference)
    hidden = layer.get_int("hidden_size", weight.shape[1])
    shape = layer.get_tuple("input_shape")
    length = int(shape[-2]) if len(shape) >= 3 else input_width // hidden
    if weight.shape[1] != hidden or length * hidden != input_width:
        raise ValueError(f"MHA_SPLIT {layer.id}: input/projection shape mismatch")
    bias = layer.params.get("bias")
    assert bias is None or isinstance(bias, torch.Tensor)
    bias = weight.new_zeros(weight.shape[0]) if bias is None else bias.to(reference).flatten()
    features = torch.arange(input_width, device=reference.device)
    role = layer.params.get("role", "all")
    if role in ("query", "key"):
        position = layer.get_int("position", 0)
        if not 0 <= position < length:
            raise ValueError(f"MHA_SPLIT {layer.id}: position {position} outside {length}")
        features = features.reshape(length, hidden)[position]
    elif role == "value":
        feature = layer.get_int("feature", 0)
        if not 0 <= feature < weight.shape[0]:
            raise ValueError(f"MHA_SPLIT {layer.id}: feature {feature} outside projection")
        weight, bias = weight[feature:feature + 1], bias[feature:feature + 1]
    return features, weight, bias


def forward_mha_split(L, parent_boxes, parent_lins, parent_frames, preds,
                      post_activation, device, dtype):
    """Project selected token/feature rows without expanding a block weight."""
    from .tf_forward import LinearBound, _concretize, _fwd_rowwise_planes, _intersect_boxes, _reset_forward_box

    assert len(preds) == 1, f"MHA_SPLIT expects 1 predecessor, got {len(preds)}"
    lower, upper = parent_boxes[0].lb.flatten(1), parent_boxes[0].ub.flatten(1)
    features, weight, bias = _mha_projection(L, lower.shape[1], lower)
    batch, hidden = lower.shape[0], weight.shape[1]
    lower, upper = (x.index_select(1, features).reshape(batch, -1, hidden) for x in (lower, upper))
    middle = ((lower + upper) * .5) @ weight.T + bias
    radius = ((upper - lower) * .5) @ weight.abs().T
    lb, ub = (middle - radius).flatten(1), (middle + radius).flatten(1)
    parent = parent_lins[0]
    if parent.A_lb is not None and parent.A_ub is not None:
        selected = LinearBound(
            parent.A_lb.index_select(1, features), parent.b_lb.index_select(1, features),
            parent.A_ub.index_select(1, features), parent.b_ub.index_select(1, features),
        )
        groups = batch * lower.shape[1]
        coefficients = weight.unsqueeze(0).expand(groups, -1, -1)
        intercepts = bias.unsqueeze(0).expand(groups, -1)
        lin = _fwd_rowwise_planes(
            selected, hidden, (coefficients, intercepts, coefficients, intercepts)
        )
        assert lin is not None
        frame = parent_frames[0]
        lin_lb, lin_ub = _concretize(lin, frame)
        lb, ub = _intersect_boxes(lin_lb, lin_ub, lb, ub)
    else:
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
    out = Bounds(lb, ub)
    return out, out, lin, frame


def backward_mha_split(L, nu, bounds_dict, preds, M: int = 1, alpha=None):
    """Apply the exact adjoint of the selected projection and its bias."""
    assert len(preds) == 1, f"MHA_SPLIT expects 1 predecessor, got {len(preds)}"
    input_width = bounds_dict[preds[0]].lb.flatten(1).shape[1]
    features, weight, bias = _mha_projection(L, input_width, nu)
    values = nu.reshape(nu.shape[0], -1, weight.shape[0])
    selected = (values @ weight).flatten(1)
    result = nu.new_zeros(nu.shape[0], input_width)
    result.index_copy_(1, features, selected)
    return [result], (values * bias).sum(dim=(1, 2))


# ---------------------------------------------------------------------------
# Bilinear attention cores: ATT_SCORES (Q Kᵀ) and ATT_MIX (probs · V)
# ---------------------------------------------------------------------------
#
# Both layers compute one scalar output  out = scale * Σ_d x_d · y_d (+ mask),
# the McCormick-relaxed product of two interval inputs. The dual forward box is
# the four-corner envelope; the dual backward kernel re-derives the LOCAL
# McCormick planes from the per-input boxes in ``bounds_dict`` and sign-splits
# ν across them, fusing the two valid planes by a convex weight ``w ∈ [0, 1]``.
# Because a convex combination of two valid lower (resp. upper) bounds is itself
# a valid lower (resp. upper) bound, the fused plane is sound for ANY weight,
# so the attention slope α may be optimized freely in [0, 1]. The SAME rule init
# (``rule_based_alpha`` on the local plane-difference corners) seeds both the
# allocator warm start and the ``alpha=None`` fallback.


def _attention_input_boxes(L, bounds_dict):
    """Per-input interval boxes ``(x_l, x_u, y_l, y_u, scale, mask)`` of a core.

    ATT_SCORES reads its query/key boxes from ``q_src``/``k_src`` and scales by
    ``1/dk``. ATT_MIX reads its value box from ``v_src`` and bounds the softmax
    weights by their definitional range ``[0, 1]`` (a sound, structure-free box:
    attention probabilities always lie in the unit interval). Boxes are flattened
    to ``[B, n]`` so the bilinear sum runs over the contraction axis.
    """
    k = L.kind.upper() if isinstance(L.kind, str) else L.kind
    if k == "ATT_SCORES":
        q = bounds_dict[int(L.params["q_src"])]
        kk = bounds_dict[int(L.params["k_src"])]
        scale = 1.0 / float(L.params["dk"])
        return (q.lb.flatten(start_dim=1), q.ub.flatten(start_dim=1),
                kk.lb.flatten(start_dim=1), kk.ub.flatten(start_dim=1),
                scale, L.params.get("mask"))
    if k == "ATT_MIX":
        w = bounds_dict[int(L.params["w_src"])]
        v = bounds_dict[int(L.params["v_src"])]
        w_l = torch.zeros_like(w.lb.flatten(start_dim=1))
        w_u = torch.ones_like(w.lb.flatten(start_dim=1))
        return (w_l, w_u,
                v.lb.flatten(start_dim=1), v.ub.flatten(start_dim=1),
                1.0, None)
    raise ValueError(f"_attention_input_boxes: unsupported attention kind {k!r}")


def _bilinear_diff_corners(
    x_l: torch.Tensor, x_u: torch.Tensor,
    y_l: torch.Tensor, y_u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Min/max over the box of the two plane differences, summed over terms.

    The McCormick relaxation of ``Σ_d x_d y_d`` carries two valid lower planes
    (``z=False``/``z=True`` corners) and two valid upper planes. The fusion acts
    on their difference: ``l_min``/``l_max`` bracket ``Σ (plane2_lo - plane1_lo)``
    and ``u_min``/``u_max`` bracket ``Σ (plane1_hi - plane2_hi)``. Each difference
    is linear in ``(x, y)`` with sign-definite coefficients, so the extrema are
    read off the box corners. The last axis is the contraction dimension;
    arbitrary broadcastable leading axes identify independent scalar outputs.
    """
    # Lower-plane difference: (y_u - y_l) x + (x_u - x_l) y + (x_l y_l - x_u y_u).
    cx_l = y_u - y_l
    cy_l = x_u - x_l
    const_l = x_l * y_l - x_u * y_u
    # Preserve K-major summation at the rule-slope threshold, where a rounding
    # difference can select a different plane. Only the reduction terms need
    # contiguous storage; the broadcast operand bounds remain views.
    l_min = (cx_l.clamp(min=0) * x_l + cx_l.clamp(max=0) * x_u
             + cy_l.clamp(min=0) * y_l + cy_l.clamp(max=0) * y_u
             + const_l).contiguous().sum(dim=-1, keepdim=True)
    l_max = (cx_l.clamp(min=0) * x_u + cx_l.clamp(max=0) * x_l
             + cy_l.clamp(min=0) * y_u + cy_l.clamp(max=0) * y_l
             + const_l).contiguous().sum(dim=-1, keepdim=True)
    # Upper-plane difference: (y_u - y_l) x + (x_l - x_u) y + (x_u y_l - x_l y_u).
    cx_u = y_u - y_l
    cy_u = x_l - x_u
    const_u = x_u * y_l - x_l * y_u
    u_min = (cx_u.clamp(min=0) * x_l + cx_u.clamp(max=0) * x_u
             + cy_u.clamp(min=0) * y_l + cy_u.clamp(max=0) * y_u
             + const_u).contiguous().sum(dim=-1, keepdim=True)
    u_max = (cx_u.clamp(min=0) * x_u + cx_u.clamp(max=0) * x_l
             + cy_u.clamp(min=0) * y_u + cy_u.clamp(max=0) * y_l
             + const_u).contiguous().sum(dim=-1, keepdim=True)
    return l_min, l_max, u_min, u_max


def _fusion_weights(pos, cross, omega):
    """Convex fusion weight ``w = pos + cross·ω`` clamped into ``[0, 1]``.

    ``ω`` is clamped to ``[0, 1]`` BEFORE forming ``w`` so a stray slope (an
    externally supplied optimized α outside the unit interval) can never push
    ``w`` past the two valid planes and produce an unsound extrapolated plane;
    ``pos``/``cross`` are disjoint 0/1 masks, so ``w`` stays a convex weight.
    """
    return pos + cross * omega.clamp(0.0, 1.0)


def _fused_mccormick_planes(x_l, x_u, y_l, y_u, w_l, w_u):
    """Per-term fused McCormick planes ``(a_x, a_y, c)`` for lower and upper.

    Each plane is the convex ``w``-combination of the two valid McCormick planes
    of ``x·y`` (the ``z=False``/``z=True`` corners), so it is itself a valid
    lower (resp. upper) plane for any ``w ∈ [0, 1]``. Shared verbatim by the
    scalar attention backward and the batched MATMUL backward.
    """
    a_x_lo = (1 - w_l) * y_l + w_l * y_u
    a_y_lo = (1 - w_l) * x_l + w_l * x_u
    c_lo = (1 - w_l) * (-x_l * y_l) + w_l * (-x_u * y_u)
    a_x_hi = (1 - w_u) * y_u + w_u * y_l
    a_y_hi = (1 - w_u) * x_l + w_u * x_u
    c_hi = (1 - w_u) * (-x_l * y_u) + w_u * (-x_u * y_l)
    return a_x_lo, a_y_lo, c_lo, a_x_hi, a_y_hi, c_hi


def attention_rule_alpha(
    x_l: torch.Tensor, x_u: torch.Tensor,
    y_l: torch.Tensor, y_u: torch.Tensor, k_thresh: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Rule-based fusion slopes ``{omega_l, omega_u}`` for a bilinear core.

    Mirrors the forward :func:`rule_based_alpha` warm start but on the LOCAL
    per-input box corners, so the allocator and the ``alpha=None`` kernel path
    agree on the same init. Each slope is ``[..., 1]`` over the broadcast
    output axes, including ``[B, 1]`` for a scalar attention core.
    """
    l_min, l_max, u_min, u_max = _bilinear_diff_corners(x_l, x_u, y_l, y_u)
    lower_cross = ((l_min < 0) & (l_max > 0)).to(x_l.dtype)
    upper_cross = ((u_min < 0) & (u_max > 0)).to(x_l.dtype)
    omega_l = rule_based_alpha(lower_cross, l_max, l_min, k_thresh)
    omega_u = rule_based_alpha(upper_cross, u_max, u_min, k_thresh)
    return {"omega_l": omega_l, "omega_u": omega_u}


def _bilinear_planes(x_l, x_u, y_l, y_u, alpha, k_thresh):
    """Use one convex fusion rule for attention, MATMUL and their frames."""
    l_min, l_max, u_min, u_max = _bilinear_diff_corners(x_l, x_u, y_l, y_u)
    lower_pos = ((l_min > 0) & (l_max > 0)).to(x_l.dtype)
    lower_cross = ((l_min < 0) & (l_max > 0)).to(x_l.dtype)
    upper_pos = ((u_min > 0) & (u_max > 0)).to(x_l.dtype)
    upper_cross = ((u_min < 0) & (u_max > 0)).to(x_l.dtype)
    if alpha is None:
        omega_l = rule_based_alpha(lower_cross, l_max, l_min, k_thresh)
        omega_u = rule_based_alpha(upper_cross, u_max, u_min, k_thresh)
    else:
        omega_l = alpha["omega_l"].to(x_l).reshape(l_min.shape)
        omega_u = alpha["omega_u"].to(x_l).reshape(u_min.shape)
    return _fused_mccormick_planes(
        x_l, x_u, y_l, y_u,
        _fusion_weights(lower_pos, lower_cross, omega_l),
        _fusion_weights(upper_pos, upper_cross, omega_u),
    )


def _dual_bilinear_backward(
    nu: torch.Tensor,
    x_l: torch.Tensor, x_u: torch.Tensor,
    y_l: torch.Tensor, y_u: torch.Tensor,
    scale: float, mask, alpha, k_thresh: float, M: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sign-split ν across the fused McCormick planes of ``scale·Σ x_d y_d``.

    Returns ``(nu_x, nu_y, contrib)`` where ``nu_x``/``nu_y`` are DISTINCT
    tensors (one per bilinear input, never aliased — the solver adds both routes
    when the two inputs trace to a shared predecessor) and ``contrib`` is the
    once-counted McCormick constant of the selected plane, shape ``[B*M]``.

    For the scalar output, ``ν >= 0`` selects the fused lower plane and ``ν < 0``
    the fused upper plane, so ``ν · out`` stays a valid lower bound. The fusion
    weight ``w = pos·1 + cross·omega`` lies in ``[0, 1]`` on every entry, hence
    the fused plane is a convex combination of two valid planes and is sound for
    any ``omega`` (rule init when ``alpha`` is ``None``, else the optimized α).
    """
    BM = nu.shape[0]
    if BM % M != 0:
        raise ValueError(f"_dual_bilinear_backward: nu batch {BM} not divisible by M={M}")
    B = BM // M
    nu_flat = nu.flatten(start_dim=1)
    if nu_flat.shape[-1] != 1:
        raise ValueError(
            f"_dual_bilinear_backward: expected scalar output, got width {nu_flat.shape[-1]}"
        )
    n = x_l.shape[-1]

    a_x_lo, a_y_lo, c_lo, a_x_hi, a_y_hi, c_hi = _bilinear_planes(
        x_l, x_u, y_l, y_u, alpha, k_thresh)

    v = nu_flat.reshape(B, M, 1)
    pos = v >= 0
    a_x = torch.where(pos, a_x_lo.unsqueeze(1), a_x_hi.unsqueeze(1))
    a_y = torch.where(pos, a_y_lo.unsqueeze(1), a_y_hi.unsqueeze(1))
    c_sel = torch.where(pos, c_lo.unsqueeze(1), c_hi.unsqueeze(1))

    v_scaled = v * scale
    nu_x = (v_scaled * a_x).reshape(BM, n)
    nu_y = (v_scaled * a_y).reshape(BM, n)
    contrib = (v_scaled * c_sel).sum(dim=-1).reshape(BM)

    if isinstance(mask, torch.Tensor):
        # Mask is an additive constant on the score; ν · mask is counted once.
        contrib = contrib + (v * mask.to(nu).reshape(-1, 1, 1)).reshape(BM)
    return nu_x, nu_y, contrib


def forward_attention(L, parent_boxes, parent_lins, parent_frames, preds,
                      post_activation, device, dtype):
    """Dual forward interval box for a bilinear attention core.

    ATT_SCORES bounds ``scale·Σ_d Q_d K_d (+ mask)`` and ATT_MIX bounds
    ``Σ_s W_s V_s`` by the four-corner McCormick envelope of each product term,
    summed over the contraction axis. Compatible explicit operand relations
    retain their shared symbolic frame; other operands use the interval box.
    """
    from act.back_end.core import Bounds
    from .tf_forward import LinearBound, _concretize, _intersect_boxes, _reset_forward_box, _shared_explicit_frame

    k = L.kind.upper() if isinstance(L.kind, str) else L.kind
    x_box, y_box = parent_boxes[0], parent_boxes[1]
    x_l = x_box.lb.flatten(start_dim=1)
    x_u = x_box.ub.flatten(start_dim=1)
    y_l = y_box.lb.flatten(start_dim=1)
    y_u = y_box.ub.flatten(start_dim=1)
    if k == "ATT_SCORES":
        scale = 1.0 / float(L.params["dk"])
        mask = L.params.get("mask")
    elif k == "ATT_MIX":
        scale = 1.0
        mask = None
    else:
        raise NotImplementedError(f"forward_attention: unsupported kind {k!r}")

    lo, hi = four_corner_envelope(x_l, x_u, y_l, y_u, sum_axis=-1, keepdim=True, scale=scale)
    if isinstance(mask, torch.Tensor):
        m = mask.to(lo).reshape(-1, 1)
        lo = lo + m
        hi = hi + m
    if _shared_explicit_frame(parent_lins, parent_frames, [x_l.shape[1], y_l.shape[1]]):
        lin = _fwd_matmul_lin(
            parent_lins[0], parent_lins[1], x_l, x_u, y_l, y_u,
            1, 1, x_l.shape[1], 1, float(L.params.get("k_thresh", 1.0)),
        )
        assert lin.A_lb is not None and lin.A_ub is not None
        offset = mask.to(lo).reshape(-1, 1) if isinstance(mask, torch.Tensor) else 0.0
        lin = LinearBound(
            lin.A_lb * scale, lin.b_lb * scale + offset,
            lin.A_ub * scale, lin.b_ub * scale + offset,
        )
        lin_lb, lin_ub = _concretize(lin, parent_frames[0])
        lb, ub = _intersect_boxes(lin_lb, lin_ub, lo, hi)
        out = Bounds(lb, ub)
        return out, out, lin, parent_frames[0]
    out = Bounds(lo, hi)
    lin, frame = _reset_forward_box(lo, hi, device, dtype)
    return out, out, lin, frame


def backward_attention(L, nu, bounds_dict, preds, M: int = 1, alpha=None):
    """Bilinear attention dual backward for ATT_SCORES and ATT_MIX.

    Routes ν through the fused McCormick planes re-derived from the per-input
    boxes (queries/keys for ATT_SCORES; softmax weights in ``[0, 1]`` and values
    for ATT_MIX). Returns one ν per input as DISTINCT tensors in predecessor
    order, plus the once-counted McCormick/mask constant. ``alpha`` is the
    per-layer ``{omega_l, omega_u}`` slope pytree (``None`` uses the rule init).
    """
    if len(preds) != 2:
        raise ValueError(
            f"backward_attention: layer {L.id} expects 2 predecessors, got {len(preds)}")
    x_l, x_u, y_l, y_u, scale, mask = _attention_input_boxes(L, bounds_dict)
    k_thresh = float(L.params.get("k_thresh", 1.0))
    nu_x, nu_y, contrib = _dual_bilinear_backward(
        nu, x_l, x_u, y_l, y_u, scale, mask, alpha, k_thresh, M)
    # preds == [x_src, y_src] by the converter contract; return ν positionally
    # as DISTINCT tensors. When x_src == y_src (Q and K share one embedding) the
    # solver adds both routes into the same accumulator, which is exactly right.
    is_scores = (L.kind.upper() if isinstance(L.kind, str) else L.kind) == "ATT_SCORES"
    x_src = int(L.params["q_src"]) if is_scores else int(L.params["w_src"])
    y_src = int(L.params["k_src"]) if is_scores else int(L.params["v_src"])
    if [int(preds[0]), int(preds[1])] != [x_src, y_src]:
        raise ValueError(
            f"backward_attention: preds {list(preds)} != [x_src={x_src}, y_src={y_src}]")
    return [nu_x, nu_y], contrib


# ---------------------------------------------------------------------------
# General batched bilinear MATMUL (var x var):  Z[g,i,j] = sum_k X[g,i,k] Y[g,k,j]
# ---------------------------------------------------------------------------
#
# The ONNX import emits a generic MATMUL for the attention Q Kᵀ and probs · V
# products. Each output element is the SAME scalar bilinear dot the attention
# cores relax, so the per-term McCormick planes and the convex α-fusion are
# reused verbatim; only the bookkeeping generalizes from one scalar to a matrix.
# X-rows and Y-columns are shared across outputs, so the backward scatters each
# output's ν onto its row/column and ADDS the routes (the exact adjoint of the
# chosen planes — sound for any fusion weight, here the rule init).


def _matmul_shapes(x_shape, y_shape):
    """Factor batched matmul shapes into ``(G, I, K, J)`` with shared batch G."""
    x_shape = tuple(int(d) for d in x_shape)
    y_shape = tuple(int(d) for d in y_shape)
    if len(x_shape) < 2 or len(y_shape) < 2:
        raise ValueError(f"MATMUL needs >=2D operands, got {x_shape} @ {y_shape}")
    I, K = x_shape[-2], x_shape[-1]
    if y_shape[-2] != K:
        raise ValueError(f"MATMUL contraction mismatch: {x_shape} @ {y_shape}")
    J = y_shape[-1]
    G = math.prod(x_shape[:-2])
    Gy = math.prod(y_shape[:-2])
    if G != Gy:
        raise ValueError(f"MATMUL batch mismatch: {x_shape} @ {y_shape}")
    return G, I, K, J


def _matmul_mccormick_box(x_l, x_u, y_l, y_u, G, I, K, J):
    """Four-corner McCormick interval box of ``Z`` flattened to ``[B, G*I*J]``."""
    B = x_l.shape[0]
    xl = x_l.reshape(B, G, I, K).unsqueeze(-1)
    xu = x_u.reshape(B, G, I, K).unsqueeze(-1)
    yl = y_l.reshape(B, G, K, J).unsqueeze(-3)
    yu = y_u.reshape(B, G, K, J).unsqueeze(-3)
    lo, hi = four_corner_envelope(xl, xu, yl, yu, sum_axis=-2)
    return lo.reshape(B, -1), hi.reshape(B, -1)


def _matmul_term_boxes(x_l, x_u, y_l, y_u, G, I, K, J):
    """Expose broadcast views [B,G,I,1,K] and [B,G,1,J,K], without expansion."""
    B = x_l.shape[0]
    xl = x_l.reshape(B, G, I, K).unsqueeze(3)
    xu = x_u.reshape(B, G, I, K).unsqueeze(3)
    yl = y_l.reshape(B, G, K, J).transpose(-1, -2).unsqueeze(2)
    yu = y_u.reshape(B, G, K, J).transpose(-1, -2).unsqueeze(2)
    return xl, xu, yl, yu


def matmul_rule_alpha(x_l, x_u, y_l, y_u, G, I, K, J, k_thresh=1.0):
    """Allocate lane-major omega leaves [B,G*I*J], without a spec-row axis."""
    boxes = _matmul_term_boxes(x_l, x_u, y_l, y_u, G, I, K, J)
    tree = attention_rule_alpha(*boxes, k_thresh)
    return {key: value.reshape(x_l.shape[0], G * I * J) for key, value in tree.items()}


def _matmul_bilinear_backward(
    nu, x_l, x_u, y_l, y_u, G, I, K, J, M, k_thresh=1.0, alpha=None,
):
    """Route each signed objective through valid per-output McCormick planes."""
    B = x_l.shape[0]
    if nu.shape[0] != B * M:
        raise ValueError(f"_matmul_bilinear_backward: nu batch {nu.shape[0]} != B*M={B*M}")
    boxes = _matmul_term_boxes(x_l, x_u, y_l, y_u, G, I, K, J)
    a_x_lo, a_y_lo, c_lo, a_x_hi, a_y_hi, c_hi = _bilinear_planes(*boxes, alpha, k_thresh)

    v = nu.reshape(B, M, G, I, J)
    pos = v >= 0
    v_pos = torch.where(pos, v, 0.)
    v_neg = torch.where(pos, 0., v)
    # Contract directly over output rows/columns; never build [B,M,G,I,J,K].
    nu_x = (
        torch.einsum("bmgij,bgijk->bmgik", v_pos, a_x_lo)
        + torch.einsum("bmgij,bgijk->bmgik", v_neg, a_x_hi)
    ).reshape(B * M, G * I * K)
    nu_y = (
        torch.einsum("bmgij,bgijk->bmgkj", v_pos, a_y_lo)
        + torch.einsum("bmgij,bgijk->bmgkj", v_neg, a_y_hi)
    ).reshape(B * M, G * K * J)
    contrib = (
        torch.einsum("bmgij,bgij->bm", v_pos, c_lo.sum(dim=-1))
        + torch.einsum("bmgij,bgij->bm", v_neg, c_hi.sum(dim=-1))
    ).reshape(B * M)
    return nu_x, nu_y, contrib


def _matmul_frame_term(a, first_a, second_a, first_b, second_b, specs):
    """Compose signed contraction coefficients with the selected affine tracks."""
    positive, negative = a.clamp(min=0), a.clamp(max=0)
    A = torch.einsum(specs[0], positive, first_a) + torch.einsum(specs[0], negative, second_a)
    b = torch.einsum(specs[1], positive, first_b) + torch.einsum(specs[1], negative, second_b)
    return A, b


def _fwd_matmul_lin(lin_x, lin_y, x_l, x_u, y_l, y_u, G, I, K, J, k_thresh):
    """Compose bilinear planes over a shared explicit frame, not its box alone."""
    from .tf_forward import LinearBound

    B, symbols = x_l.shape[0], lin_x.A_lb.shape[-1]
    boxes = _matmul_term_boxes(x_l, x_u, y_l, y_u, G, I, K, J)
    ax_l, ay_l, c_l, ax_u, ay_u, c_u = _bilinear_planes(*boxes, None, k_thresh)
    xl_a, xu_a = (a.reshape(B, G, I, K, symbols) for a in (lin_x.A_lb, lin_x.A_ub))
    xl_b, xu_b = (b.reshape(B, G, I, K) for b in (lin_x.b_lb, lin_x.b_ub))
    yl_a, yu_a = (a.reshape(B, G, K, J, symbols) for a in (lin_y.A_lb, lin_y.A_ub))
    yl_b, yu_b = (b.reshape(B, G, K, J) for b in (lin_y.b_lb, lin_y.b_ub))
    x_specs = ("bgijk,bgiks->bgijs", "bgijk,bgik->bgij")
    y_specs = ("bgijk,bgkjs->bgijs", "bgijk,bgkj->bgij")
    xlo_a, xlo_b = _matmul_frame_term(ax_l, xl_a, xu_a, xl_b, xu_b, x_specs)
    xhi_a, xhi_b = _matmul_frame_term(ax_u, xu_a, xl_a, xu_b, xl_b, x_specs)
    ylo_a, ylo_b = _matmul_frame_term(ay_l, yl_a, yu_a, yl_b, yu_b, y_specs)
    yhi_a, yhi_b = _matmul_frame_term(ay_u, yu_a, yl_a, yu_b, yl_b, y_specs)
    return LinearBound(
        (xlo_a + ylo_a).reshape(B, G * I * J, symbols),
        (xlo_b + ylo_b + c_l.sum(-1)).reshape(B, G * I * J),
        (xhi_a + yhi_a).reshape(B, G * I * J, symbols),
        (xhi_b + yhi_b + c_u.sum(-1)).reshape(B, G * I * J),
    )


def forward_matmul(L, parent_boxes, parent_lins, parent_frames, preds,
                   post_activation, device, dtype):
    """Bound MATMUL and retain compatible explicit operand relations."""
    from act.back_end.core import Bounds
    from .tf_forward import _concretize, _intersect_boxes, _reset_forward_box, _shared_explicit_frame

    G, I, K, J = _matmul_shapes(L.params["x_shape"], L.params["y_shape"])
    x_box, y_box = parent_boxes[0], parent_boxes[1]
    lo, hi = _matmul_mccormick_box(
        x_box.lb, x_box.ub, y_box.lb, y_box.ub, G, I, K, J)
    if _shared_explicit_frame(parent_lins, parent_frames, [G * I * K, G * K * J]):
        lin = _fwd_matmul_lin(
            parent_lins[0], parent_lins[1], x_box.lb, x_box.ub, y_box.lb, y_box.ub,
            G, I, K, J, float(L.params.get("k_thresh", 1.0)),
        )
        lin_lb, lin_ub = _concretize(lin, parent_frames[0])
        lb, ub = _intersect_boxes(lin_lb, lin_ub, lo, hi)
        out = Bounds(lb, ub)
        return out, out, lin, parent_frames[0]
    out = Bounds(lo, hi)
    lin, frame = _reset_forward_box(lo, hi, device, dtype)
    return out, out, lin, frame


def backward_matmul(L, nu, bounds_dict, preds, M: int = 1, alpha=None):
    """Bilinear dual backward for a batched MATMUL; one ν per operand."""
    if len(preds) != 2:
        raise ValueError(
            f"backward_matmul: layer {L.id} expects 2 predecessors, got {len(preds)}")
    G, I, K, J = _matmul_shapes(L.params["x_shape"], L.params["y_shape"])
    x_box = bounds_dict[int(preds[0])]
    y_box = bounds_dict[int(preds[1])]
    nu_x, nu_y, contrib = _matmul_bilinear_backward(
        nu, x_box.lb, x_box.ub, y_box.lb, y_box.ub, G, I, K, J, M,
        float(L.params.get("k_thresh", 1.0)), alpha)
    return [nu_x, nu_y], contrib


# ---------------------------------------------------------------------------
# Element-wise bilinear MUL (var x var):  Z[n] = X[n] * Y[n]
# ---------------------------------------------------------------------------
# Element-wise multiply is a batched MATMUL with one scalar product per output:
# G = N (batch over the N flattened elements), I = K = J = 1. Reusing the exact
# four-corner McCormick box and the fused-plane bilinear backward guarantees the
# same soundness as MATMUL, fully tensorised (no per-element loop) across the
# subproblem batch B and spec multiplicity M. Broadcasting operands (unequal
# element counts) are not reframed this way and fail loudly.


def _mul_flatten(x_box, y_box):
    B = x_box.lb.shape[0]
    xl = x_box.lb.reshape(B, -1); xu = x_box.ub.reshape(B, -1)
    yl = y_box.lb.reshape(B, -1); yu = y_box.ub.reshape(B, -1)
    n = xl.shape[-1]
    if yl.shape[-1] != n:
        raise NotImplementedError(
            f"dual MUL: broadcasting operands (x has {n}, y has {yl.shape[-1]} "
            f"elements) is not supported; only element-wise same-shape MUL.")
    return xl, xu, yl, yu, n


def forward_mul(L, parent_boxes, parent_lins, parent_frames, preds,
                post_activation, device, dtype):
    """Dual forward interval box for an element-wise bilinear MUL (var x var)."""
    from act.back_end.core import Bounds
    from .tf_forward import _reset_forward_box

    if len(parent_boxes) < 2:
        raise NotImplementedError(
            f"dual MUL layer {L.id}: needs both operands as predecessor edges but only "
            f"{len(parent_boxes)} present (preds={preds}). The other operand references a "
            f"non-adjacent variable block (params x_vars/y_vars) that torch2act did not wire "
            f"as a graph edge — this also breaks the interval path's get_predecessor_bounds "
            f"index and must be fixed in the converter, not the TF.")
    xl, xu, yl, yu, n = _mul_flatten(parent_boxes[0], parent_boxes[1])
    lo, hi = _matmul_mccormick_box(xl, xu, yl, yu, n, 1, 1, 1)
    out = Bounds(lo, hi)
    lin, frame = _reset_forward_box(lo, hi, device, dtype)
    return out, out, lin, frame


def backward_mul(L, nu, bounds_dict, preds, M: int = 1, alpha=None):
    """Element-wise bilinear dual backward for MUL; one ν per operand."""
    if len(preds) != 2:
        raise ValueError(
            f"backward_mul: layer {L.id} expects 2 predecessors, got {len(preds)}")
    xl, xu, yl, yu, n = _mul_flatten(bounds_dict[int(preds[0])], bounds_dict[int(preds[1])])
    nu_x, nu_y, contrib = _matmul_bilinear_backward(nu, xl, xu, yl, yu, n, 1, 1, 1, M)
    return [nu_x, nu_y], contrib


_LocalPlanes = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _local_vector_planes(
    l: torch.Tensor, u: torch.Tensor,
    build: "Callable[[LinearBounds], LinearBounds]",
) -> _LocalPlanes:
    """Sound per-output linear planes of a vector op over a local input box.

    Seeds an identity :class:`LinearBounds` frame whose unit L-infinity ball maps
    exactly onto the box ``[l, u]`` (each coordinate gets its own axis scaled by
    its half-width), runs the validated interval relaxation ``build`` to obtain
    the op's lower/upper envelopes in that frame, then rebases the envelopes onto
    the raw input coordinates by the exact change of variables
    ``x_k = c_k + r_k t_k``. Soundness is inherited from the interval relaxation;
    the rebase is linear so it preserves the enclosure.

    Args:
        l: Per-group lower box, shape ``[G, n]`` (groups fold batch x rows).
        u: Per-group upper box, shape ``[G, n]``.
        build: Maps the seed frame to the relaxed output frame (softmax / layer
            norm over the last axis).

    Returns:
        ``(A_lo, c_lo, A_hi, c_hi)`` with ``A`` shape ``[G, n_out, n_in]`` and
        ``c`` shape ``[G, n_out]`` such that ``A_lo x + c_lo <= f(x) <= A_hi x
        + c_hi`` for every ``x`` in the box.
    """
    G, n = l.shape
    center = 0.5 * (l + u)
    radius = 0.5 * (u - l)
    eye = torch.eye(n, device=l.device, dtype=l.dtype)
    seed_w = (radius.unsqueeze(-1) * eye).unsqueeze(1)
    seed_b = center.unsqueeze(1)
    seed = LinearBounds(
        seed_w, seed_w.clone(), seed_b, seed_b.clone(),
        p=math.inf, eps=1.0, perturbed_words=1,
    )
    out = build(seed)
    lw, uw = out.lw[:, 0], out.uw[:, 0]
    lb, ub = out.lb[:, 0], out.ub[:, 0]
    rden = torch.where(radius > 0, radius, torch.ones_like(radius)).unsqueeze(1)
    a_lo = lw.transpose(1, 2) / rden
    a_hi = uw.transpose(1, 2) / rden
    c_lo = lb - (a_lo * center.unsqueeze(1)).sum(dim=-1)
    c_hi = ub - (a_hi * center.unsqueeze(1)).sum(dim=-1)
    return a_lo, c_lo, a_hi, c_hi


def _softmax_interval_box(l: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable exact coordinate extrema, using softmax's signed monotonicity."""
    diagonal = torch.eye(l.shape[-1], dtype=torch.bool, device=l.device)
    lower_differences = (u.unsqueeze(-2) - l.unsqueeze(-1)).masked_fill(diagonal, 0.)
    upper_differences = (l.unsqueeze(-2) - u.unsqueeze(-1)).masked_fill(diagonal, 0.)
    point = (l == u).all(-1, keepdim=True)
    value = l.softmax(-1)
    return (
        torch.where(point, value, torch.exp(-torch.logsumexp(lower_differences, dim=-1))),
        torch.where(point, value, torch.exp(-torch.logsumexp(upper_differences, dim=-1))),
    )


def _softmax_local_planes(l: torch.Tensor, u: torch.Tensor) -> _LocalPlanes:
    """Replace nonfinite exponential/reciprocal planes by valid box planes."""
    a_lo, c_lo, a_hi, c_hi = _local_vector_planes(l, u, lambda bound: bound.softmax())
    point = (l == u).all(-1, keepdim=True)
    bad_lo = point | ~(torch.isfinite(a_lo).all(-1) & torch.isfinite(c_lo))
    bad_hi = point | ~(torch.isfinite(a_hi).all(-1) & torch.isfinite(c_hi))
    if bool(bad_lo.any()) or bool(bad_hi.any()):
        lower, upper = _softmax_interval_box(l, u)
        a_lo = torch.where(bad_lo.unsqueeze(-1), torch.zeros_like(a_lo), a_lo)
        c_lo = torch.where(bad_lo, lower, c_lo)
        a_hi = torch.where(bad_hi.unsqueeze(-1), torch.zeros_like(a_hi), a_hi)
        c_hi = torch.where(bad_hi, upper, c_hi)
    return a_lo, c_lo, a_hi, c_hi


def _dual_vector_backward(
    nu: torch.Tensor, l: torch.Tensor, u: torch.Tensor, row: int,
    planes_fn: Callable[[torch.Tensor, torch.Tensor], _LocalPlanes], M: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route the dual variable through the transpose of local relaxation planes.

    The op acts independently over each contiguous ``row``-sized block of the
    feature axis, so blocks fold into the group axis. Each output coordinate
    routes by the sign of its dual entry -- nonnegative entries take the lower
    plane and negative entries the upper plane -- which keeps ``nu . f(x)`` a
    valid lower bound. ``nu`` is sample-major ``[B*M, n]`` while the box stays at
    ``[B, n]``, so the planes are built once per sample and broadcast over ``M``.
    """
    BM = nu.shape[0]
    if BM % M != 0:
        raise ValueError(f"_dual_vector_backward: nu batch {BM} not divisible by M={M}")
    B = BM // M
    n = nu.flatten(start_dim=1).shape[-1]
    if n % row != 0:
        raise ValueError(f"_dual_vector_backward: feature {n} not divisible by row {row}")
    n_rows = n // row

    l_g = l.flatten(start_dim=1)[:, :n].reshape(B * n_rows, row)
    u_g = u.flatten(start_dim=1)[:, :n].reshape(B * n_rows, row)
    a_lo, c_lo, a_hi, c_hi = planes_fn(l_g, u_g)
    a_lo = a_lo.reshape(B, n_rows, row, row)
    a_hi = a_hi.reshape(B, n_rows, row, row)
    c_lo = c_lo.reshape(B, n_rows, row)
    c_hi = c_hi.reshape(B, n_rows, row)

    v = nu.flatten(start_dim=1).reshape(B, M, n_rows, row)
    pos = v >= 0
    a_sel = torch.where(
        pos.unsqueeze(-1), a_lo.unsqueeze(1), a_hi.unsqueeze(1),
    )
    c_sel = torch.where(pos, c_lo.unsqueeze(1), c_hi.unsqueeze(1))
    nu_in = torch.einsum("bmrj,bmrjk->bmrk", v, a_sel)
    contrib = (v * c_sel).sum(dim=(-1, -2)).reshape(BM)
    return nu_in.reshape(BM, n), contrib


def _layernorm_builder(
    L, n: int, device: torch.device, dtype: torch.dtype,
) -> "Callable[[LinearBounds], LinearBounds]":
    """Bind a layer's gamma/beta/variant into a LayerNorm relaxation closure."""
    variant = L.params.get("variant", L.params.get("layer_norm", "standard"))
    gamma = L.params["gamma"].to(device=device, dtype=dtype).flatten()
    beta = L.params["beta"].to(device=device, dtype=dtype).flatten()
    if gamma.numel() != n and n % gamma.numel() == 0:
        repeat = n // gamma.numel()
        gamma = gamma.repeat(repeat)
        beta = beta.repeat(repeat)
    eps = float(L.params.get("eps", 1e-5))
    return lambda bound: bound.layer_norm(gamma, beta, variant=variant, eps=eps)


def forward_softmax(L, parent_boxes, parent_lins, parent_frames, preds,
                    post_activation, device, dtype):
    """Dual forward box for row-wise softmax via the local relaxation envelope.

    The output box is the concretization of the same per-output planes used by
    :func:`backward_softmax`, intersected with the exact ``[0, 1]`` simplex range
    of softmax, so the forward and backward relaxations stay consistent. The
    pre-activation box is stored for the backward pass when bounds are kept
    pre-activation.
    """
    from act.back_end.core import Bounds
    from .tf_forward import _concretize, _fwd_rowwise_planes, _intersect_boxes, _reset_forward_box

    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    row = softmax_rowsize(L, parent_box)
    if row is None:
        raise NotImplementedError("Dual SOFTMAX requires contiguous last-axis rows")
    planes, l_g, u_g = _vector_local_planes(pre_lb, pre_ub, row, _softmax_local_planes)
    out_lb, out_ub = _vector_planes_box(planes, l_g, u_g, pre_lb.shape[0])
    out_lb = out_lb.clamp(0.0, 1.0)
    out_ub = out_ub.clamp(0.0, 1.0)
    box_lb, box_ub = _softmax_interval_box(l_g, u_g)
    out_lb, out_ub = _intersect_boxes(
        out_lb, out_ub, box_lb.reshape_as(out_lb), box_ub.reshape_as(out_ub))
    lin = _fwd_rowwise_planes(parent_lins[0], row, planes)
    if lin is not None:
        lin_lb, lin_ub = _concretize(lin, parent_frames[0])
        out_lb, out_ub = _intersect_boxes(lin_lb, lin_ub, out_lb, out_ub)
    out = Bounds(out_lb, out_ub)
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    if lin is not None and not post_activation:
        return stored, out, lin, parent_frames[0]
    lin, frame = _reset_forward_box(out_lb, out_ub, device, dtype)
    return stored, out, lin, frame


def backward_softmax(L, nu, bounds_dict, preds, M: int = 1, alpha=None):
    """Softmax dual backward as the transpose of the exp / row-sum decomposition.

    Single-input, vector-coupled: the local input box yields sound per-output
    planes of softmax (``exp`` then divide by the row sum), through which the
    dual variable is routed by sign. ``alpha`` is accepted for registry-contract
    parity but unused -- softmax uses the fixed-slope relaxation for now.
    """
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_softmax: layer {L.id} missing bounds in bounds_dict")
    if len(preds) != 1:
        raise ValueError(f"SOFTMAX expects 1 predecessor, got {len(preds)}")
    row = softmax_rowsize(L, bounds)
    if row is None:
        raise NotImplementedError("Dual SOFTMAX requires contiguous last-axis rows")
    nu_in, contrib = _dual_vector_backward(
        nu, bounds.lb, bounds.ub, row, _softmax_local_planes, M)
    return [nu_in], contrib


def forward_layernorm(L, parent_boxes, parent_lins, parent_frames, preds,
                      post_activation, device, dtype):
    """Dual forward box for LayerNorm via the local relaxation envelope.

    The output box concretizes the same per-output planes used by
    :func:`backward_layernorm` (mean-subtract, variance, rsqrt, scale-shift), so
    forward and backward relaxations agree. The pre-activation box is stored for
    the backward pass when bounds are kept pre-activation.
    """
    from act.back_end.core import Bounds
    from .tf_forward import _concretize, _fwd_rowwise_planes, _intersect_boxes, _reset_forward_box

    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    n = pre_lb.flatten(1).shape[-1]
    gamma_n = L.params["gamma"].numel()
    row = n if gamma_n == 0 or n % gamma_n else gamma_n
    builder = _layernorm_builder(L, row, pre_lb.device, pre_lb.dtype)
    planes, l_g, u_g = _vector_local_planes(
        pre_lb, pre_ub, row, lambda l, u: _local_vector_planes(l, u, builder),
    )
    out_lb, out_ub = _vector_planes_box(planes, l_g, u_g, pre_lb.shape[0])
    interval = _layernorm_box(L, parent_box)
    out_lb, out_ub = _intersect_boxes(out_lb, out_ub, interval.lb, interval.ub)
    lin = _fwd_rowwise_planes(parent_lins[0], row, planes)
    if lin is not None:
        lin_lb, lin_ub = _concretize(lin, parent_frames[0])
        out_lb, out_ub = _intersect_boxes(lin_lb, lin_ub, out_lb, out_ub)
    out = Bounds(out_lb, out_ub)
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    if lin is not None and not post_activation:
        return stored, out, lin, parent_frames[0]
    lin, frame = _reset_forward_box(out_lb, out_ub, device, dtype)
    return stored, out, lin, frame


def backward_layernorm(L, nu, bounds_dict, preds, M: int = 1, alpha=None):
    """LayerNorm dual backward as the transpose of the normalization chain.

    Single-input, vector-coupled: the local input box yields sound per-output
    planes of ``gamma * (x - mean) / sqrt(var + eps) + beta``, through which the
    dual variable is routed by sign. ``alpha`` is accepted for registry-contract
    parity but unused -- layernorm uses the fixed-slope relaxation for now.
    """
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_layernorm: layer {L.id} missing bounds in bounds_dict")
    if len(preds) != 1:
        raise ValueError(f"LAYERNORM expects 1 predecessor, got {len(preds)}")
    n = bounds.lb.flatten(start_dim=1).shape[-1]
    gamma_n = L.params["gamma"].numel()
    row = n if gamma_n == 0 or n % gamma_n else gamma_n
    builder = _layernorm_builder(L, row, bounds.lb.device, bounds.lb.dtype)
    nu_in, contrib = _dual_vector_backward(
        nu, bounds.lb, bounds.ub, row, lambda l, u: _local_vector_planes(l, u, builder), M)
    return [nu_in], contrib


def _vector_local_planes(
    pre_lb: torch.Tensor, pre_ub: torch.Tensor, row: int,
    planes_fn: Callable[[torch.Tensor, torch.Tensor], _LocalPlanes],
) -> tuple[_LocalPlanes, torch.Tensor, torch.Tensor]:
    """Build row-local planes once for both box and symbolic propagation."""
    B = pre_lb.shape[0]
    n = pre_lb.flatten(start_dim=1).shape[-1]
    if row < 1 or n % row:
        raise ValueError(f"Vector row width {row} does not divide feature width {n}")
    n_rows = n // row
    l_g = pre_lb.flatten(start_dim=1).reshape(B * n_rows, row)
    u_g = pre_ub.flatten(start_dim=1).reshape(B * n_rows, row)
    return planes_fn(l_g, u_g), l_g, u_g


def _vector_planes_box(
    planes: _LocalPlanes, l_g: torch.Tensor, u_g: torch.Tensor, batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concretize local planes over the row boxes they were constructed for."""
    a_lo, c_lo, a_hi, c_hi = planes
    out_lo = (a_lo.clamp(min=0) * l_g.unsqueeze(1)
              + a_lo.clamp(max=0) * u_g.unsqueeze(1)).sum(dim=-1) + c_lo
    out_hi = (a_hi.clamp(min=0) * u_g.unsqueeze(1)
              + a_hi.clamp(max=0) * l_g.unsqueeze(1)).sum(dim=-1) + c_hi
    return out_lo.reshape(batch, -1), out_hi.reshape(batch, -1)


# GELU soundness constants (_GELU_MIN_X/_GELU_MIN_Y/_GELU_INFLECTION) are
# defined once in interval_tf.tf_attention and imported above so the sound
# floor cannot drift between the two relaxation implementations.
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _gelu_value(x: torch.Tensor) -> torch.Tensor:
    """Exact erf GELU ``x * Phi(x)`` matching ``torch.nn.functional.gelu``."""
    return torch.nn.functional.gelu(x)


def _gelu_deriv(x: torch.Tensor) -> torch.Tensor:
    """Derivative ``Phi(x) + x * phi(x)`` of the exact erf GELU."""
    return 0.5 * (1.0 + torch.erf(x * _INV_SQRT_2)) + \
        x * _INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def _gelu_relaxation(
    l: torch.Tensor, u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sound lower/upper planes ``(k_lo, b_lo, k_hi, b_hi)`` of ``k*x + b``.

    Curvature-aware envelope cloned from
    :meth:`interval_tf.tf_attention.LinearBounds.gelu`. erf GELU has
    ``g''(x) = phi(x) * (2 - x^2)``, so it is convex on ``[-sqrt2, sqrt2]`` and
    concave outside. On a single-curvature interval the midpoint tangent and the
    endpoint secant are exact one-sided envelopes -- tangent below and secant
    above when convex, roles swapped when concave. An interval straddling an
    inflection point mixes curvature and falls back to the min-aware box (a
    zero-slope envelope with the global minimum ``_GELU_MIN_Y`` floor when
    enclosed). ``_GELU_MIN_Y`` is a sound floor for the erf form.
    """
    from .tf_mlp import _repair_degenerate_interval
    u = _repair_degenerate_interval(l, u, "gelu_relaxation")

    g_l, g_u = _gelu_value(l), _gelu_value(u)
    m = (l + u) * 0.5
    g_m, dg_m = _gelu_value(m), _gelu_deriv(m)
    width = u - l
    secant = (g_u - g_l) / torch.where(width > 0, width, torch.ones_like(width))

    convex = (l >= -_GELU_INFLECTION) & (u <= _GELU_INFLECTION)
    concave = (u <= -_GELU_INFLECTION) | (l >= _GELU_INFLECTION)
    contains_min = (l <= _GELU_MIN_X) & (u >= _GELU_MIN_X)
    box_lo = torch.where(
        contains_min, torch.full_like(g_l, _GELU_MIN_Y),
        torch.minimum(g_l, g_u))
    box_hi = torch.maximum(g_l, g_u)

    zero = torch.zeros_like(l)
    k_lo = torch.where(convex, dg_m, torch.where(concave, secant, zero))
    xl0 = torch.where(concave, l, m)
    yl0 = torch.where(convex, g_m, torch.where(concave, g_l, box_lo))
    k_hi = torch.where(convex, secant, torch.where(concave, dg_m, zero))
    xu0 = torch.where(convex, l, m)
    yu0 = torch.where(convex, g_l, torch.where(concave, g_m, box_hi))

    b_lo = yl0 - k_lo * xl0
    b_hi = yu0 - k_hi * xu0
    return k_lo, b_lo, k_hi, b_hi


def forward_gelu(L, parent_boxes, parent_lins, parent_frames, preds,
                 post_activation, device, dtype):
    """Compose erf GELU planes; use the shared interval image for tanh GELU."""
    from .tf_smooth import _forward_smooth_planes
    from .tf_forward import _reset_forward_box

    parent_box = parent_boxes[0]
    out = _gelu_box(L, parent_box)
    if L.params.get("approximate", "none") == "tanh":
        lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
        return out if post_activation else parent_box, out, lin, frame
    return _forward_smooth_planes(
        parent_boxes, parent_lins, parent_frames, post_activation,
        out.lb, out.ub, _gelu_relaxation,
    )


def _dual_gelu_backward(
    nu: torch.Tensor, bounds, M: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched GELU backward with lazy M-broadcast (cf. ``dual_smooth_backward``).

    Relaxation coefficients depend only on the per-neuron box, computed once at
    ``[B, 1, n]`` and broadcast against ``nu`` viewed at ``[B, M, n]``. ``nu`` is
    routed by sign so ``nu^T g(x)`` stays a valid lower bound: positive ``nu``
    uses the lower plane, negative ``nu`` the upper plane.
    """
    BM = nu.shape[0]
    if BM % M != 0:
        raise ValueError(f"_dual_gelu_backward: nu batch {BM} not divisible by M={M}")
    B = BM // M

    v_flat = nu.flatten(start_dim=1)
    l_B = bounds.lb.flatten(start_dim=1) if bounds.lb.dim() >= 2 \
        else bounds.lb.flatten().unsqueeze(0).expand(B, -1)
    u_B = bounds.ub.flatten(start_dim=1) if bounds.ub.dim() >= 2 \
        else bounds.ub.flatten().unsqueeze(0).expand(B, -1)
    n = min(v_flat.shape[-1], l_B.shape[-1])
    if v_flat.shape[-1] != l_B.shape[-1]:
        v_flat = v_flat[..., :n]
        l_B = l_B[..., :n]
        u_B = u_B[..., :n]

    l = l_B.unsqueeze(1)
    u = u_B.unsqueeze(1)
    k_lo, b_lo, k_hi, b_hi = _gelu_relaxation(l, u)

    v = v_flat.view(B, M, n)
    v_pos = v >= 0
    k = torch.where(v_pos, k_lo, k_hi)
    b = torch.where(v_pos, b_lo, b_hi)

    v_out = v * k
    contrib = (v * b).sum(dim=-1).view(BM)
    return v_out.view(BM, n), contrib


def backward_gelu(L, nu, bounds_dict, preds, M: int = 1, alpha=None):
    """GELU dual backward via the curvature-aware linear relaxation.

    Elementwise single-input kernel mirroring
    :func:`dual_tf.tf_smooth.backward_sigmoid`: the per-neuron box from
    ``bounds_dict`` yields the sound planes of :func:`_gelu_relaxation`. ``alpha``
    is accepted for registry-contract parity but unused -- GELU uses the fixed
    slope, as no per-neuron slope is allocated for it.
    """
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_gelu: layer {L.id} missing bounds in bounds_dict")
    if len(preds) != 1:
        raise ValueError(f"GELU expects 1 predecessor, got {len(preds)}")
    if L.params.get("approximate", "none") == "tanh":
        from .tf_smooth import _dual_constant_box_backward
        out = _gelu_box(L, bounds)
        nu_out, contrib = _dual_constant_box_backward(nu, out.lb, out.ub, M)
        return [nu_out], contrib
    nu_out, contrib = _dual_gelu_backward(nu, bounds, M)
    return [nu_out], contrib
