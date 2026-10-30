#===- act/back_end/bab/bound_refresh.py - Split-conditioned bounds --------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
"""One split refresh traversal, with optional literal-slot dependencies.

Bounds remain ordinary Bounds objects. Supports are sufficient (not minimal):
each bit names a packed literal slot, never a spec row. An endpoint equal to
its split-free reference needs no bits, because that reference holds on every
retained subset. Unknown provenance must not be read as an empty set.
"""
from dataclasses import dataclass, replace
from typing import Protocol

import torch

from act.back_end.core import Bounds, Layer, Net, topological_sort
from act.back_end.dual_tf.tf_forward import (
    Frame, LinearBound, _entry_lin_frame, _fwd_relu,
    _box_bias, _box_bn, _box_dense, _box_scale, _fwd_conv2d_interval,
    _align, _intersect_boxes, _lin_bound_lane_chunk, _sum_interval_bounds, input_support,
)
from act.back_end.layer_schema import LayerKind
from act.config.config import DualConfig

# Each output endpoint reads only the same endpoint of the same-index operand.
_ENDPOINT_PRESERVING = frozenset({
    LayerKind.ADD.value, LayerKind.BIAS.value, LayerKind.FLATTEN.value,
    LayerKind.RELU.value, LayerKind.RESHAPE.value,
})


class _LiteralDecoder(Protocol):
    """The existing LiteralCodec surface needed by the numerical traversal."""

    def decode_tensor(
        self, literals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode packed literal slots into layer, neuron and sign tensors."""
        ...


@dataclass(frozen=True)
class BoundSupport:
    """Lower/upper dependencies [K,N,S] and provenance validity [K,N]."""

    lower: torch.Tensor
    upper: torch.Tensor
    known: torch.Tensor


@dataclass(frozen=True)
class RefreshResult:
    """Numerical bounds plus an optional, spec-independent support sidecar."""

    bounds: dict[int, Bounds]
    supports: dict[int, BoundSupport] | None
    linear_refined: bool = False


def _lane_phase(split_signs: dict[int, torch.Tensor], lid: int) -> torch.Tensor | None:
    """Return one layer's [K,N] phases after checking spec invariance."""
    phase = split_signs.get(lid)
    if phase is None or phase.dim() != 3:
        return phase
    if not torch.equal(phase, phase[:, :1].expand_as(phase)):
        raise ValueError("split phases must be spec invariant")
    return phase[:, 0]


def _phase_bits(
    phase: torch.Tensor, lid: int, layers: torch.Tensor, neurons: torch.Tensor,
    signs: torch.Tensor, active: torch.Tensor,
) -> torch.Tensor:
    """Name the slot behind each phase; an unnamed phase needs its whole row."""
    bits = torch.zeros(*phase.shape, active.shape[1], dtype=torch.bool, device=phase.device)
    own = torch.zeros_like(phase, dtype=torch.bool)
    lanes, slots = torch.where(active & (layers == lid))
    indices = neurons[lanes, slots]
    matching = signs[lanes, slots] == phase[lanes, indices]
    own[lanes[matching], indices[matching]] = True
    bits[lanes[matching], indices[matching], slots[matching]] = True
    return bits | (((phase != 0) & ~own)[:, :, None] & active[:, None, :])


def _root_valid_supports(
    net: Net, base: dict[int, Bounds], split_signs: dict[int, torch.Tensor],
    literals: torch.Tensor, codec: _LiteralDecoder,
) -> dict[int, BoundSupport]:
    """Supports of the split-free fallback: only a ReLU's own phase clamp.

    Every interval, including those of additional nonlinear operators, is the
    root reference, so no residual depends on any other literal.
    """
    layers, neurons, signs = codec.decode_tensor(literals.masked_fill(literals == 0, 1))
    active = literals != 0
    supports = {}
    for lid, box in base.items():
        lb, ub = box.lb.flatten(1), box.ub.flatten(1)
        lower = torch.zeros(*lb.shape, literals.shape[1], dtype=torch.bool, device=lb.device)
        upper = lower.clone()
        known = torch.ones_like(lb, dtype=torch.bool)
        phase = _lane_phase(split_signs, lid) if net.by_id[lid].kind == LayerKind.RELU.value else None
        if phase is not None:
            bits = _phase_bits(phase, lid, layers, neurons, signs, active)
            lower = bits & ((phase > 0) & (lb < 0))[:, :, None]
            upper = bits & ((phase < 0) & (ub > 0))[:, :, None]
            known &= ~(phase.ne(0).any(1) & ~active.any(1))[:, None]
        supports[lid] = BoundSupport(lower, upper, known)
    return supports


def _inherited_support(
    layer: Layer, preds: list[int], post_support: dict[int, BoundSupport],
    shape: tuple[int, int], width: int, device: torch.device,
) -> BoundSupport:
    """Follow the operands each output endpoint used, per endpoint when known."""
    k, n = shape
    operands = [post_support[pid] for pid in preds]
    if layer.kind in _ENDPOINT_PRESERVING and all(ps.known.shape == shape for ps in operands):
        lower = torch.zeros(k, n, width, dtype=torch.bool, device=device)
        upper = lower.clone()
        known = torch.ones(shape, dtype=torch.bool, device=device)
        for operand in operands:
            lower |= operand.lower
            upper |= operand.upper
            known &= operand.known
        return BoundSupport(lower, upper, known)
    weight = layer.params.get("weight") if layer.kind == LayerKind.DENSE.value else None
    lower = torch.zeros(k, n, width, dtype=torch.bool, device=device)
    upper = lower.clone()
    known = torch.ones(shape, dtype=torch.bool, device=device)
    # Bound the temporary union by slots rather than materializing
    # [K,N_parent,N_output,S] adjacency or broadcasting across specs.
    if (isinstance(weight, torch.Tensor) and weight.dim() == 2 and weight.shape[0] == n
            and operands[0].known.shape[1] == weight.shape[1]):
        parent = operands[0]
        positive = (weight > 0).T.to(device=device, dtype=torch.float32)
        negative = (weight < 0).T.to(device=device, dtype=torch.float32)

        def reaches(mask: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
            return (mask.transpose(1, 2).to(selector) @ selector).transpose(1, 2) > 0

        for start in range(0, width, 64):
            end = start + 64
            low, high = parent.lower[..., start:end], parent.upper[..., start:end]
            lower[..., start:end] = reaches(low, positive) | reaches(high, negative)
            upper[..., start:end] = reaches(high, positive) | reaches(low, negative)
        known = ((~parent.known).to(positive) @ (positive + negative)) == 0
        return BoundSupport(lower, upper, known)
    for start in range(0, width, 64):
        end = start + 64
        inherited = torch.zeros(k, min(end, width) - start, dtype=torch.bool, device=device)
        for operand in operands:
            inherited |= (operand.lower[..., start:end] | operand.upper[..., start:end]).any(1)
        lower[..., start:end] = inherited[:, None, :]
    for operand in operands:
        known &= operand.known.all(1, keepdim=True)
    return BoundSupport(lower, lower.clone(), known)


def _halfspace_lower(
    a: torch.Tensor, b: torch.Tensor, frame: Frame,
    h: torch.Tensor, g: torch.Tensor,
) -> torch.Tensor:
    """Lower-bound a*x+b subject to h*x+g>=0 by a nonnegative dual.

    Sorting coefficient breakpoints finds the one-halfspace box optimum's
    multiplier in O(D log D), without a [D,D] identity or candidate tensor.
    The final certificate uses existing input_support, so root p-balls remain
    available too. Empty halfspaces simply use lambda=0, never an inf proof.
    """
    lanes, rows, symbols = a.shape
    chunk = _lin_bound_lane_chunk(lanes, rows, symbols, a, 8, -1, "split_halfspace")
    if chunk is None:
        return torch.full_like(b, -torch.inf)
    if chunk < lanes:
        parts = []
        for start in range(0, lanes, chunk):
            end = min(start + chunk, lanes)
            geometry = frame.geometry
            if geometry is not None and geometry.center.shape[0] != 1:
                geometry = replace(
                    geometry, center=geometry.center[start:end],
                    eps=geometry.eps[start:end], mask=geometry.mask[start:end],
                )
            local_frame = Frame(
                Bounds(frame.box.lb[start:end], frame.box.ub[start:end]),
                geometry, frame.columns,
            )
            parts.append(_halfspace_lower(a[start:end], b[start:end], local_frame, h[start:end], g[start:end]))
        return torch.cat(parts)
    lo, hi = frame.box.lb[:, None], frame.box.ub[:, None]
    h = h[:, None]
    corner = torch.where(a >= 0, lo, hi)
    deficit = -((h * corner).sum(-1) + g[:, None])
    switch = (a * h > 0) | ((a == 0) & (h > 0))
    ratio = torch.where(switch, a / torch.where(h == 0, 1, h), torch.inf)
    order = ratio.argsort(-1)
    jumps = (h.abs() * (hi - lo)).expand_as(a).masked_fill(~switch, 0)
    reached = jumps.gather(-1, order).cumsum(-1) >= deficit[:, :, None]
    first = reached.to(torch.int64).argmax(-1, keepdim=True)
    multiplier = ratio.gather(-1, order).gather(-1, first).squeeze(-1)
    multiplier = torch.where(
        (deficit > 0) & reached.any(-1) & torch.isfinite(multiplier),
        multiplier.clamp(min=0), 0,
    )
    coefficients = a - multiplier[:, :, None] * h
    value = input_support(coefficients, frame)[0] + b - multiplier * g[:, None]
    scale = torch.maximum(lo.abs(), hi.abs())
    magnitude = (a.abs() * scale).sum(-1) + b.abs() + multiplier * (
        (h.abs() * scale).sum(-1) + g[:, None].abs()
    )
    # Conservative arithmetic band, evaluated without introducing a second
    # domain support implementation. Nonfinite candidates never tighten.
    value = value - 16 * torch.finfo(a.dtype).eps * magnitude.clamp(min=1)
    return torch.where(torch.isfinite(value), value, -torch.inf)


@torch.no_grad()
def refresh_bounds(
    net: Net,
    base: dict[int, Bounds],
    split_signs: dict[int, torch.Tensor],
    *,
    literals: torch.Tensor | None = None,
    codec: _LiteralDecoder | None = None,
    linear: bool = False,
    support_refinement: bool = True,
    rows_cap: int = 64,
    forward_lin_max_perturbed: int | None = None,
) -> RefreshResult:
    """Refresh from a split-free reference, intersecting only sound root bounds.

    Unsupported handlers fall back to the entire split-free reference with
    unknown provenance. The optional sidecar is constructed during this same
    numerical traversal, without M.
    """
    if (literals is None) != (codec is None):
        raise ValueError("tracked refresh requires both literals and codec")
    if rows_cap < 1:
        raise ValueError("linear refresh rows_cap must be positive")
    from act.back_end.dual_tf.dual_tf import DualTF

    affine_chain = {
        LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, LayerKind.ASSERT.value,
        LayerKind.DENSE.value, LayerKind.RELU.value, LayerKind.FLATTEN.value,
        LayerKind.RESHAPE.value, LayerKind.BIAS.value, LayerKind.SCALE.value,
        LayerKind.BN.value,
    }
    linear = linear and all(
        layer.kind in affine_chain and len(net.preds.get(layer.id, [])) <= 1
        for layer in net.layers
    )
    lin: LinearBound | None = None
    frame: Frame | None = None
    lin_support: torch.Tensor | None = None
    # Each h*x+g>=0 is a necessary split condition in the CURRENT frame.
    halfspaces: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    linear_refined = False
    supports: dict[int, BoundSupport] | None = {} if literals is not None else None
    out = dict(base)
    values: dict[int, Bounds] = {}
    post_support: dict[int, BoundSupport] = {}
    layers = neurons = signs = active = None
    if literals is not None:
        assert codec is not None
        layers, neurons, signs = codec.decode_tensor(literals.masked_fill(literals == 0, 1))
        active = literals != 0
    for lid in topological_sort(net):
        layer = net.by_id[lid]
        kind = layer.kind
        if kind == LayerKind.ASSERT.value:
            continue
        root = base[lid]
        root_lb, root_ub = root.lb.flatten(1), root.ub.flatten(1)
        k, n = root_lb.shape
        preds = net.preds.get(lid, [])
        if kind in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value):
            lb, ub = root_lb, root_ub
            preds = []
            if linear and lin is None:
                geometry = net.input_geometry
                lin, frame = _entry_lin_frame(
                    lb, ub,
                    DualConfig().forward_lin_max_perturbed if forward_lin_max_perturbed is None else forward_lin_max_perturbed,
                    lb.device, lb.dtype, geometry.to(lb) if geometry is not None else None,
                )
                if literals is not None:
                    lin_support = torch.zeros_like(literals, dtype=torch.bool)
        elif linear and kind != LayerKind.RELU.value:
            assert lin is not None and frame is not None
            parent_frame = frame
            stored, _, lin, frame = DualTF._FORWARD_REGISTRY[kind](
                layer, [values[preds[0]]], [lin], [frame], preds,
                False, root_lb.device, root_lb.dtype,
            )
            lb, ub = stored.lb, stored.ub
            if frame is not parent_frame:
                # Intermediate reset coordinates are not input coordinates.
                halfspaces.clear()
        elif kind == LayerKind.DENSE.value:
            parent = values[preds[0]]
            lb, ub = _box_dense(layer, parent.lb, parent.ub)
        elif kind == LayerKind.CONV2D.value:
            parent = values[preds[0]]
            lb, ub = _fwd_conv2d_interval(layer, parent.lb, parent.ub)
            lb, ub = lb.flatten(1), ub.flatten(1)
        elif kind == LayerKind.ADD.value:
            # Use operand multiplicity, not unique predecessor identities.
            summed = _sum_interval_bounds([values[pid] for pid in preds])
            lb, ub = summed.lb, summed.ub
            if layer.params.get("bias") is not None:
                bias = _align(layer.params["bias"].flatten(), n)
                lb, ub = lb + bias, ub + bias
        elif kind in (LayerKind.FLATTEN.value, LayerKind.RESHAPE.value, LayerKind.RELU.value):
            parent = values[preds[0]]
            lb, ub = parent.lb, parent.ub
        elif kind in (LayerKind.BIAS.value, LayerKind.SCALE.value, LayerKind.BN.value):
            parent = values[preds[0]]
            handler = {
                LayerKind.BIAS.value: _box_bias,
                LayerKind.SCALE.value: _box_scale,
                LayerKind.BN.value: _box_bn,
            }[kind]
            lb, ub = handler(layer, parent.lb, parent.ub)
        else:
            # Some stored boxes are PRE-activation, so treating an unsupported
            # layer's root entry as its output would be unsound downstream.
            # Fall back for the complete traversal, not just this operator.
            if literals is None:
                return RefreshResult(dict(base), None)
            assert codec is not None
            return RefreshResult(dict(base), _root_valid_supports(net, base, split_signs, literals, codec))

        lower = upper = known = None
        if supports is not None:
            assert literals is not None
            if literals.shape[0] != k:
                raise ValueError("literal slots must match refresh lanes")
            inherited = _inherited_support(
                layer, preds, post_support, (k, n), literals.shape[1], lb.device,
            )
            lower, upper, known = inherited.lower, inherited.upper, inherited.known
            if linear:
                assert lin_support is not None
                lower |= lin_support[:, None, :]
                upper |= lin_support[:, None, :]
            # An endpoint no stronger than the root is the split-free root
            # endpoint after intersection, so it needs no inherited bits.
            lower &= (lb > root_lb).unsqueeze(-1)
            upper &= (ub < root_ub).unsqueeze(-1)
        lb, ub = _intersect_boxes(lb, ub, root_lb, root_ub)
        phase = _lane_phase(split_signs, lid) if kind == LayerKind.RELU.value else None
        if phase is not None:
            if phase.shape != lb.shape:
                raise ValueError("split phases must match [K,N] preactivation bounds")
            if known is not None:
                assert active is not None
                known &= ~(phase.ne(0).any(1) & ~active.any(1))[:, None]
        if linear:
            assert lin is not None and frame is not None
            # An implicit identity above the existing forward cap must not
            # become a huge dense identity just to encode phase halfspaces.
            if lin.A_lb is None or lin.A_ub is None:
                linear = False
            else:
                if phase is not None:
                    selected = torch.where(phase.ne(0).any(0))[0]
                    for first in range(0, selected.numel(), rows_cap):
                        neurons_chunk = selected[first:first + rows_cap]
                        sign = phase.index_select(1, neurons_chunk)
                        h = torch.where(
                            (sign > 0)[:, :, None],
                            lin.A_ub.index_select(1, neurons_chunk),
                            -lin.A_lb.index_select(1, neurons_chunk),
                        ) * sign.ne(0)[:, :, None]
                        g = torch.where(
                            sign > 0, lin.b_ub.index_select(1, neurons_chunk),
                            -lin.b_lb.index_select(1, neurons_chunk),
                        ) * sign.ne(0)
                        dependencies = None
                        if literals is not None:
                            assert lin_support is not None and active is not None
                            assert layers is not None and neurons is not None and signs is not None
                            own = (
                                active[:, None, :] & (layers[:, None, :] == lid)
                                & (neurons[:, None, :] == neurons_chunk[None, :, None])
                                & (signs[:, None, :] == sign[:, :, None])
                            )
                            own |= (sign.ne(0) & ~own.any(2))[:, :, None] & active[:, None, :]
                            dependencies = lin_support[:, None, :] | own
                        # Unbind produces device views, not scalar GPU indices:
                        # no per-neuron .item() synchronization during extraction.
                        for index, (normal, offset) in enumerate(zip(h.unbind(1), g.unbind(1))):
                            halfspaces.append((
                                normal, offset,
                                dependencies[:, index] if dependencies is not None else None,
                            ))
                # Query in bounded row blocks. Halfspace candidates are
                # independent valid lower/upper bounds, so intersecting them
                # is sound; this is not a joint-polytope LP certificate.
                for start in range(0, n, rows_cap):
                    end = min(start + rows_cap, n)
                    for h, g, dependency in halfspaces:
                        new_lb = _halfspace_lower(lin.A_lb[:, start:end], lin.b_lb[:, start:end], frame, h, g)
                        new_ub = -_halfspace_lower(-lin.A_ub[:, start:end], -lin.b_ub[:, start:end], frame, h, g)
                        compatible = (
                            torch.maximum(new_lb, lb[:, start:end])
                            <= torch.minimum(new_ub, ub[:, start:end])
                        )
                        if supports is not None:
                            assert lower is not None and upper is not None and dependency is not None
                            assert lin_support is not None
                            dep = (dependency | lin_support)[:, None, :]
                            if support_refinement:
                                lower[:, start:end] = torch.where(
                                    ((new_lb > lb[:, start:end]) & compatible)[:, :, None], dep, lower[:, start:end]
                                ) | (dep & (new_lb == lb[:, start:end])[:, :, None])
                                upper[:, start:end] = torch.where(
                                    ((new_ub < ub[:, start:end]) & compatible)[:, :, None], dep, upper[:, start:end]
                                ) | (dep & (new_ub == ub[:, start:end])[:, :, None])
                            else:
                                lower[:, start:end] |= dep
                                upper[:, start:end] |= dep
                        lb[:, start:end], ub[:, start:end] = _intersect_boxes(
                            new_lb, new_ub, lb[:, start:end], ub[:, start:end]
                        )
                        linear_refined = True
        if kind == LayerKind.RELU.value:
            if phase is not None:
                if supports is not None:
                    assert active is not None and layers is not None and neurons is not None and signs is not None
                    assert lower is not None and upper is not None
                    # A clamped endpoint rests only on its own literal; an
                    # already stable endpoint does not need the literal at all.
                    bits = _phase_bits(phase, lid, layers, neurons, signs, active)
                    lower = torch.where(((phase > 0) & (lb < 0))[:, :, None], bits, lower)
                    upper = torch.where(((phase < 0) & (ub > 0))[:, :, None], bits, upper)
                    # An inverted box's repaired upper endpoint is its lower one.
                    empty = ((phase > 0) & (ub < 0)) | ((phase < 0) & (lb > 0))
                    upper = torch.where(empty[:, :, None], lower | upper, upper)
                lb = torch.where(phase > 0, lb.clamp(min=0), lb)
                ub = torch.where(phase < 0, ub.clamp(max=0), ub)
                # A contradictory split denotes an empty region. Keep a valid
                # box representation without using that as an empty-core proof.
                ub = torch.maximum(lb, ub)
        out[lid] = Bounds(lb.reshape_as(root.lb), ub.reshape_as(root.ub))
        if supports is not None:
            assert lower is not None and upper is not None and known is not None
            supports[lid] = post_support[lid] = BoundSupport(lower, upper, known)
            if kind == LayerKind.RELU.value:
                # relu(z) >= 0 is global; the post-activation upper endpoint
                # still rests on the preactivation endpoint that produced it.
                post_support[lid] = BoundSupport(lower & (lb > 0)[:, :, None], upper, known)
        values[lid] = Bounds(lb.clamp(min=0), ub.clamp(min=0)) if kind == LayerKind.RELU.value else Bounds(lb, ub)
        if linear and kind == LayerKind.RELU.value:
            assert lin is not None
            # Only this local symbolic replay uses conditioned phase slopes;
            # ordinary DualSolver optimization and its alpha state are untouched.
            lin = _fwd_relu(lin, lb, ub)
            if lin is None:
                linear = False
            if supports is not None:
                assert lower is not None and upper is not None and lin_support is not None
                lin_support |= (lower | upper).any(1)
    return RefreshResult(out, supports, linear_refined)
