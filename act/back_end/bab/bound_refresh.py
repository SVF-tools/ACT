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
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, cast

import torch

from act.back_end.bab.climb import (
    BoundSupport, _inherited_support, _lane_phase, _phase_bits,
    _root_valid_supports,
)
from act.back_end.core import Bounds, Net, topological_sort
from act.back_end.dual_tf.tf_forward import (
    Frame, LinearBound, _entry_lin_frame, _fwd_relu, _relu_relaxation,
    _box_bias, _box_bn, _box_dense, _box_scale, _fwd_conv2d_interval,
    _align, _intersect_boxes, _lin_bound_lane_chunk, _sum_interval_bounds, input_support,
)
from act.back_end.layer_schema import LayerKind
from act.config.config import DualConfig

if TYPE_CHECKING:
    from act.back_end.bab.climb import LiteralCodec

# Iterations and step decay of the joint multiplier ascent.
_JOINT_HALFSPACE_STEPS = 30
_JOINT_HALFSPACE_DECAY = 0.9
# Conditions per joint solve. Tiles bound the [K,R,C] multipliers; a condition
# outside a tile only stops being combined with it.
_JOINT_HALFSPACE_TILE = 64
_JOINT_HALFSPACE_ROWS = 1024
# Coefficient entries per backward pass; larger waves run in lane chunks.
_BACKWARD_MAX_ELEMENTS = 1 << 26

@dataclass(frozen=True)
class RefreshResult:
    """Numerical bounds plus an optional, spec-independent support sidecar."""

    bounds: dict[int, Bounds]
    supports: dict[int, BoundSupport] | None
    linear_refined: bool = False


def _frame_lanes(frame: Frame, start: int, end: int) -> Frame:
    """Restrict a frame to lanes [start, end), keeping any shared root ball."""
    geometry = frame.geometry
    if geometry is not None and geometry.center.shape[0] != 1:
        geometry = replace(
            geometry, center=geometry.center[start:end],
            eps=geometry.eps[start:end], mask=geometry.mask[start:end],
        )
    return Frame(Bounds(frame.box.lb[start:end], frame.box.ub[start:end]), geometry, frame.columns)


def _chunk_lanes(
    frame: Frame, chunk: int,
    run: Callable[[int, int, Frame], torch.Tensor | tuple[torch.Tensor, ...]],
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Run a lane-wise calculation in chunks and concatenate its unchanged result."""
    parts = [
        run(start, min(start + chunk, frame.box.lb.shape[0]),
            _frame_lanes(frame, start, min(start + chunk, frame.box.lb.shape[0])))
        for start in range(0, frame.box.lb.shape[0], chunk)
    ]
    if isinstance(parts[0], tuple):
        return tuple(torch.cat(values) for values in zip(*parts))
    return torch.cat(parts)


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
        return cast(torch.Tensor, _chunk_lanes(
            frame, chunk,
            lambda start, end, local: _halfspace_lower(
                a[start:end], b[start:end], local, h[start:end], g[start:end],
            ),
        ))
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


def _joint_dual_value(
    a: torch.Tensor, b: torch.Tensor, frame: Frame,
    h: torch.Tensor, g: torch.Tensor, multipliers: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one weak-duality bound of a*x+b over all h*x+g>=0 jointly.

    Any nonnegative multipliers are sound. The domain minimum comes from
    input_support, under the same arithmetic band as one halfspace.
    """
    value = input_support(a - multipliers @ h, frame)[0] + b - (multipliers * g[:, None]).sum(-1)
    scale = torch.maximum(frame.box.lb.abs(), frame.box.ub.abs())[:, None]
    magnitude = (a.abs() * scale).sum(-1) + b.abs() + (multipliers * (
        (h.abs() * scale).sum(-1) + g.abs()
    )[:, None]).sum(-1)
    value = value - 16 * torch.finfo(a.dtype).eps * magnitude.clamp(min=1)
    return torch.where(torch.isfinite(value), value, -torch.inf)


def _ascend_joint_dual(
    a: torch.Tensor, b: torch.Tensor, frame: Frame,
    h: torch.Tensor, g: torch.Tensor, active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Projected supergradient ascent over [K,R,C] multipliers; keep the best.

    Box corners steer and rank the ascent on every domain, and the chosen
    iterate is then valued on the true domain. Normalizing each row's step by its own coefficient
    scale lets one iterate reach multipliers far from one.
    """
    mask = active[:, None].to(a)
    multipliers = a.new_zeros(*a.shape[:2], h.shape[1])
    tiny = torch.finfo(a.dtype).tiny
    witness = multipliers
    ranked = torch.full_like(b, -torch.inf)
    step = a.abs().amax(-1, keepdim=True) / h.abs().amax((1, 2), keepdim=True).clamp(min=tiny)
    for _ in range(_JOINT_HALFSPACE_STEPS):
        coefficients = a - multipliers @ h
        corner = torch.where(coefficients >= 0, frame.box.lb[:, None], frame.box.ub[:, None])
        value = (coefficients * corner).sum(-1) + b - (multipliers * g[:, None]).sum(-1)
        improved = value > ranked
        witness = torch.where(improved[:, :, None], multipliers, witness)
        ranked = torch.where(improved, value, ranked)
        supergradient = -(corner @ h.transpose(1, 2) + g[:, None]) * mask
        norm = supergradient.abs().amax(-1, keepdim=True)
        multipliers = (multipliers + step * supergradient / norm.clamp(min=tiny)).clamp(min=0)
        step = step * _JOINT_HALFSPACE_DECAY
    # Only this evaluation is a bound; the box values above only rank iterates.
    return _joint_dual_value(a, b, frame, h, g, witness), witness


def _backward_row_bounds(
    ops: list[tuple[torch.Tensor, ...]], rows: torch.Tensor, frame: Frame,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound rows of the last dense output through every lane's ReLU relaxations.

    Forward linear bounds keep one relaxation per neuron; pushing each row back
    picks the lower or upper line by that row's own sign, which is what makes a
    deep chain's later layers tight. Any relaxation valid on the lane's region is
    sound, and each pass is cut to bounded lane chunks.
    """
    lanes = frame.box.lb.shape[0]
    count = rows.numel()
    width = ops[-1][0].shape[0]
    largest = max(op[0].shape[-1] for op in ops if len(op) == 2)
    chunk = max(1, _BACKWARD_MAX_ELEMENTS // (2 * count * max(width, largest)))
    if chunk < lanes:
        return cast(tuple[torch.Tensor, torch.Tensor], _chunk_lanes(
            frame, chunk,
            lambda start, end, local: _backward_row_bounds(
                [op if len(op) == 2 else tuple(t[start:end] for t in op) for op in ops],
                rows, local,
            ),
        ))
    dtype = frame.box.lb
    coefficients = dtype.new_zeros(lanes, 2 * count, width)
    picked = torch.arange(count, device=rows.device)
    coefficients[:, picked, rows] = 1.0
    coefficients[:, picked + count, rows] = -1.0
    constant = dtype.new_zeros(lanes, 2 * count)
    magnitude = constant.clone()
    for op in reversed(ops):
        if len(op) == 2:  # (weight, bias); three tensors are a lane-wise ReLU relaxation.
            weight, bias = op
            step = coefficients @ bias
            constant, magnitude = constant + step, magnitude + (coefficients.abs() @ bias.abs())
            coefficients = coefficients @ weight
        else:
            lower_slope, upper_slope, upper_bias = op
            positive, negative = coefficients.clamp(min=0), coefficients.clamp(max=0)
            step = (negative * upper_bias[:, None]).sum(-1)
            constant, magnitude = constant + step, magnitude + step.abs()
            coefficients = positive * lower_slope[:, None] + negative * upper_slope[:, None]
    scale = torch.maximum(frame.box.lb.abs(), frame.box.ub.abs())[:, None]
    magnitude = magnitude + (coefficients.abs() * scale).sum(-1)
    value = input_support(coefficients, frame)[0] + constant
    value = value - 16 * torch.finfo(value.dtype).eps * magnitude.clamp(min=1)
    value = torch.where(torch.isfinite(value), value, -torch.inf)
    return value[:, :count], -value[:, count:]


def _witness_support(witness: torch.Tensor, dependencies: torch.Tensor | None) -> torch.Tensor | None:
    """Union the [K,C,S] slots behind each row's nonzero [K,R,C] multipliers."""
    if dependencies is None:
        return None
    return (witness > 0).to(witness) @ dependencies.to(witness) > 0


def _joint_halfspace_lower(
    a: torch.Tensor, b: torch.Tensor, frame: Frame,
    h: torch.Tensor, g: torch.Tensor, active: torch.Tensor,
    warm_active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower-bound a*x+b on the active halfspaces, returning value and witness.

    ``warm_active`` names a source core's larger constraint set. Its witness is
    only a hint: projected onto ``active``, then revalued, so a deleted
    condition is neither assumed nor reused.
    """
    lanes, rows, symbols = a.shape
    chunk = _lin_bound_lane_chunk(lanes, rows, symbols + h.shape[1], a, 8, -1, "joint_halfspace")
    if chunk is None:
        return torch.full_like(b, -torch.inf), a.new_zeros(lanes, rows, h.shape[1])
    if chunk < lanes:
        return cast(tuple[torch.Tensor, torch.Tensor], _chunk_lanes(
            frame, chunk,
            lambda start, end, local: _joint_halfspace_lower(
                a[start:end], b[start:end], local, h[start:end], g[start:end],
                active[start:end],
                None if warm_active is None else warm_active[start:end],
            ),
        ))
    best, witness = _ascend_joint_dual(a, b, frame, h, g, active)
    if warm_active is not None:
        hint = _ascend_joint_dual(a, b, frame, h, g, warm_active)[1] * active[:, None].to(a)
        value = _joint_dual_value(a, b, frame, h, g, hint)
        witness = torch.where((value > best)[:, :, None], hint, witness)
        best = torch.maximum(best, value)
    return best, witness


@torch.no_grad()
def refresh_bounds(
    net: Net,
    base: dict[int, Bounds],
    split_signs: dict[int, torch.Tensor],
    *,
    literals: torch.Tensor | None = None,
    codec: LiteralCodec | None = None,
    linear: bool = False,
    support_refinement: bool = True,
    rows_cap: int = 64,
    forward_lin_max_perturbed: int | None = None,
    warm_signs: dict[int, torch.Tensor] | None = None,
) -> RefreshResult:
    """Refresh from a split-free reference, intersecting only sound root bounds.

    Unsupported handlers fall back to the entire split-free reference with
    unknown provenance. The optional sidecar is constructed during this same
    numerical traversal, without M. ``warm_signs`` are a source core's phases,
    which may only warm-start the joint multipliers of a retained subset.
    """
    if (literals is None) != (codec is None):
        raise ValueError("tracked refresh requires both literals and codec")
    if rows_cap < 1:
        raise ValueError("linear refresh rows_cap must be positive")
    from act.back_end.dual_tf.dual_tf import DualTF

    consumers = Counter(pid for operands in net.preds.values() for pid in operands)
    # A sole-consumer ReLU queries the same symbolic rows again, with its own
    # conditions added, so its parent need not query them first.
    relu_parents = {
        net.preds[layer.id][0] for layer in net.layers
        if layer.kind == LayerKind.RELU.value and consumers[net.preds[layer.id][0]] == 1
    }
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
    backward_kinds = {
        LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, LayerKind.ASSERT.value,
        LayerKind.DENSE.value, LayerKind.RELU.value, LayerKind.FLATTEN.value, LayerKind.RESHAPE.value,
    }
    backward = linear and all(layer.kind in backward_kinds for layer in net.layers)
    backward_ops: list[tuple[torch.Tensor, ...]] = []
    relaxed = False
    input_frame: Frame | None = None
    lin: LinearBound | None = None
    frame: Frame | None = None
    lin_support: torch.Tensor | None = None
    # Each active h*x+g>=0 is a necessary split condition in the CURRENT frame.
    # An inactive row is a deleted source condition, kept only as a witness hint.
    halfspaces: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]] = []
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
                input_frame = Frame(Bounds(lb, ub), geometry.to(lb) if geometry is not None else None)
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
            if linear or backward:
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
        # An implicit identity above the existing forward cap must not become a
        # huge dense identity just to encode phase halfspaces. Backward passes
        # read only layer weights, so they continue on such wide inputs.
        if linear and (lin is None or lin.A_lb is None or lin.A_ub is None):
            linear = False
        if linear or backward:
            if linear and phase is not None:
                warm = _lane_phase(warm_signs, lid) if warm_signs is not None else None
                condition = phase if warm is None else torch.where(phase != 0, phase, warm)
                selected = torch.where(condition.ne(0).any(0))[0]
                for first in range(0, selected.numel(), rows_cap):
                    neurons_chunk = selected[first:first + rows_cap]
                    sign = condition.index_select(1, neurons_chunk)
                    active_rows = phase.index_select(1, neurons_chunk).ne(0)
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
                        own |= (active_rows & ~own.any(2))[:, :, None] & active[:, None, :]
                        dependencies = (lin_support[:, None, :] | own) & active_rows[:, :, None]
                    # Unbind produces device views, not scalar GPU indices:
                    # no per-neuron .item() synchronization during extraction.
                    for index, (normal, offset) in enumerate(zip(h.unbind(1), g.unbind(1))):
                        halfspaces.append((
                            normal, offset,
                            dependencies[:, index] if dependencies is not None else None,
                            active_rows[:, index],
                        ))
            # Only a still-open neuron's relaxation can gain from a tighter
            # box; a stable or clamped row keeps its line.
            if kind == LayerKind.RELU.value:
                open_rows = (lb < 0) & (ub > 0)
                if phase is not None:
                    open_rows &= phase == 0
                query = torch.where(open_rows.any(0))[0]
            elif lid in relu_parents or not (halfspaces or relaxed):
                query = lb.new_empty(0, dtype=torch.long)
            else:
                query = torch.arange(n, device=lb.device)
            joint_tiles = []
            if query.numel() and len(halfspaces) > 1 and not (backward and relaxed):
                for first in range(0, len(halfspaces), _JOINT_HALFSPACE_TILE):
                    tile = halfspaces[first:first + _JOINT_HALFSPACE_TILE]
                    normals = torch.stack([item[0] for item in tile], 1)
                    tile_active = torch.stack([item[3] for item in tile], 1)
                    warm_active = normals.abs().sum(-1) > 0
                    if tile_active.any():
                        joint_tiles.append((
                            normals, torch.stack([item[1] for item in tile], 1),
                            torch.stack([item[2] for item in tile], 1) if supports is not None else None,
                            tile_active,
                            None if torch.equal(warm_active, tile_active) else warm_active,
                        ))

            def tighten(
                rows: torch.Tensor, new_lb: torch.Tensor, new_ub: torch.Tensor,
                lower_dep: torch.Tensor | None, upper_dep: torch.Tensor | None,
            ) -> None:
                """Intersect one candidate, crediting its own witness on improvement.

                Without support refinement a same-layer condition is not
                charged at all: as in the source's conservative supports,
                deleting one stays a proposal the mandatory replay decides.
                """
                nonlocal lower, upper, linear_refined
                linear_refined = True
                old_lb, old_ub = lb[:, rows], ub[:, rows]
                if supports is not None:
                    assert lower is not None and upper is not None and lin_support is not None
                    earlier = lin_support[:, None, :]
                    if support_refinement:
                        assert lower_dep is not None and upper_dep is not None
                        compatible = torch.maximum(new_lb, old_lb) <= torch.minimum(new_ub, old_ub)
                        lower_dep, upper_dep = lower_dep | earlier, upper_dep | earlier
                        lower[:, rows] = torch.where(
                            ((new_lb > old_lb) & compatible)[:, :, None], lower_dep, lower[:, rows]
                        ) | (lower_dep & (new_lb == old_lb)[:, :, None])
                        upper[:, rows] = torch.where(
                            ((new_ub < old_ub) & compatible)[:, :, None], upper_dep, upper[:, rows]
                        ) | (upper_dep & (new_ub == old_ub)[:, :, None])
                    else:
                        lower[:, rows] |= earlier
                        upper[:, rows] |= earlier
                lb[:, rows], ub[:, rows] = _intersect_boxes(new_lb, new_ub, old_lb, old_ub)

            if backward and kind == LayerKind.DENSE.value:
                weight = layer.params["weight"].to(lb)
                bias = layer.params.get("bias")
                backward_ops.append((
                    weight, bias.to(lb).flatten() if isinstance(bias, torch.Tensor) else lb.new_zeros(weight.shape[0]),
                ))
            # One joint ascent can serve many rows; its multipliers are bounded per tile.
            rows_step = rows_cap if relaxed or not joint_tiles else max(rows_cap, _JOINT_HALFSPACE_ROWS)
            for first in range(0, query.numel(), rows_step):
                rows = query[first:first + rows_step]
                if backward and relaxed and input_frame is not None:
                    back_lb, back_ub = _backward_row_bounds(backward_ops, rows, input_frame)
                    earlier_rows = None
                    if lin_support is not None:
                        earlier_rows = lin_support[:, None, :].expand(-1, rows.numel(), -1)
                    tighten(rows, back_lb, back_ub, earlier_rows, earlier_rows)
                # Past the first ReLU a backward pass already uses each row's own
                # lines, so further dual multipliers rarely repay their ascent.
                if not linear or (backward and relaxed):
                    continue
                a_lb, b_lb = lin.A_lb.index_select(1, rows), lin.b_lb.index_select(1, rows)
                a_ub, b_ub = -lin.A_ub.index_select(1, rows), -lin.b_ub.index_select(1, rows)
                if len(halfspaces) == 1:
                    h, g, dependency, condition_active = halfspaces[0]
                    mask = condition_active.to(h)
                    h, g = h * mask[:, None], g * mask
                    tighten(
                        rows,
                        _halfspace_lower(a_lb, b_lb, frame, h, g),
                        -_halfspace_lower(a_ub, b_ub, frame, h, g),
                        None if dependency is None else dependency[:, None, :],
                        None if dependency is None else dependency[:, None, :],
                    )
                for h, g, dependency_rows, tile_active, hint in joint_tiles:
                    new_lb, lower_witness = _joint_halfspace_lower(a_lb, b_lb, frame, h, g, tile_active, hint)
                    new_ub, upper_witness = _joint_halfspace_lower(a_ub, b_ub, frame, h, g, tile_active, hint)
                    tighten(
                        rows, new_lb, -new_ub,
                        _witness_support(lower_witness, dependency_rows),
                        _witness_support(upper_witness, dependency_rows),
                    )
        if kind == LayerKind.RELU.value:
            if phase is not None:
                if supports is not None:
                    assert active is not None and layers is not None and neurons is not None and signs is not None
                    assert lower is not None and upper is not None
                    # A clamped endpoint rests only on its own literal, even where
                    # same-layer halfspaces already crossed zero: those conditions
                    # are not charged, so the literal must carry that dependency.
                    # A neuron stable at the root does not need the literal at all.
                    bits = _phase_bits(phase, lid, layers, neurons, signs, active)
                    lower = torch.where(((phase > 0) & (root_lb < 0))[:, :, None], bits, lower)
                    upper = torch.where(((phase < 0) & (root_ub > 0))[:, :, None], bits, upper)
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
        if (linear or backward) and kind == LayerKind.RELU.value and supports is not None:
            # A replay recomputes later bounds from retained literals alone, so a
            # later endpoint may rest on any earlier decision, tightened here or not.
            assert lower is not None and upper is not None and lin_support is not None
            assert active is not None and layers is not None
            lin_support |= (lower | upper).any(1) | (active & (layers == lid))
        if backward and kind == LayerKind.RELU.value:
            backward_ops.append(_relu_relaxation(lb, ub))
            relaxed = True
        if linear and kind == LayerKind.RELU.value:
            assert lin is not None
            # Only this local symbolic replay uses conditioned phase slopes;
            # ordinary DualSolver optimization and its alpha state are untouched.
            lin = _fwd_relu(lin, lb, ub)
            if lin is None:
                linear = False
    return RefreshResult(out, supports, linear_refined)
