#===- act/back_end/solver/solver_dual.py - Dual Bounds Solver ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# DualSolver: linear-relaxation dual certified lower-bound solver.
# STRICT batched API ([B, *shape] only). Raises ValueError on 1-D input.
# Mirrors HZSolver precedent in solver_hz.py.
#===---------------------------------------------------------------------===#
# pyright: reportMissingImports=false, reportImportCycles=false
# justification: torch C-extension stubs are absent in CI; DualSolver and verifier share result utilities during type analysis

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeAlias, Union, cast

import torch
from act.back_end.core import Bounds, Layer, Net, get_topo_order
from act.back_end.layer_schema import LayerKind
from act.back_end.dual_tf.tf_forward import (
    Frame, ForwardFrame, ForwardPrefixCache, _intersect_boxes, _reset_forward_box,
    forward_frame_row_lower_bounds, input_support,
)
from act.config.config import DualConfig
from act.back_end.solver.solver_base import (
    BatchBoundResult,
    BatchLPSolution,
    BoundResult,
    SolveStatus,
    Solver,
    SolverCaps,
)
from act.front_end.specs import OutputSpec, OutKind, lp_norm
from act.util.device_manager import get_default_device, get_default_dtype
from act.util.stats import SpecBatchResult

_monotonic = time.monotonic


@dataclass(frozen=True)
class MixedAlpha:
    """Compact ReLU slopes for a wave containing shared and per-row lanes.

    Row maps preserve logical lane order without expanding shared payloads
    along the spec axis in the frontier. Only a numerical consumer expands.
    """

    shared: torch.Tensor
    per_row: torch.Tensor
    shared_rows: torch.Tensor
    per_row_rows: torch.Tensor

    def materialize(self, rows: int, *, shared: bool = False) -> torch.Tensor:
        """Restore logical lane order at a solver or replay boundary."""
        n = self.shared.shape[0] + self.per_row.shape[0]
        width = self.shared.shape[-1]
        if self.per_row.shape[1] not in (1, rows):
            raise ValueError("mixed alpha spec rows do not match the consumer")
        out = self.shared.new_empty((n, width) if shared else (n, rows, width))
        shared_values = self.shared if shared else self.shared.unsqueeze(1).expand(-1, rows, -1)
        row_values = self.per_row.mean(1) if shared else self.per_row.expand(-1, rows, -1)
        out.index_copy_(0, self.shared_rows.to(out.device), shared_values)
        out.index_copy_(0, self.per_row_rows.to(out.device), row_values.to(out))
        return out


AlphaTree: TypeAlias = Union[
    torch.Tensor, Dict[Any, "AlphaTree"], List["AlphaTree"],
    Tuple["AlphaTree", ...], MixedAlpha, None,
]
AlphaState: TypeAlias = Dict[int, AlphaTree]


def _source_phase_strip(
    nu: torch.Tensor, bounds: Bounds, split_signs: torch.Tensor,
    phase_slope_signs: torch.Tensor, m_specs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Relax a dropped ReLU literal with its source slope, without its phase.

    Slope 1 (active) or 0 (inactive) is a global lower line. The matching upper
    line ``z + max(0, -l)`` or ``max(0, u)`` holds on the candidate interval
    ``[l, u]``, so it never assumes the deleted condition. Returns the ``nu``
    the ordinary kernel must still relax, plus the strip's predecessor ``nu``
    and objective contribution.
    """
    bm = nu.shape[0]
    values = nu.flatten(1).view(bm // m_specs, m_specs, -1)
    current = split_signs.reshape(bm // m_specs, -1, values.shape[-1]).expand_as(values)
    source = phase_slope_signs.reshape(bm // m_specs, -1, values.shape[-1]).expand_as(values)
    current, source = current.to(values.device), source.to(values.device)
    dropped = source.ne(0) & current.eq(0)
    active = source > 0
    lower = bounds.lb.flatten(1).unsqueeze(1).to(values)
    upper = bounds.ub.flatten(1).unsqueeze(1).to(values)
    intercept = torch.where(active, (-lower).clamp(min=0), upper.clamp(min=0))
    contribution = (values.clamp(max=0) * intercept).masked_fill(~dropped, 0)
    predecessor = (values * active.to(values)).masked_fill(~dropped, 0)
    return (
        values.masked_fill(dropped, 0).view(bm, -1), predecessor.view(bm, -1),
        contribution.sum(-1).view(bm),
    )


@dataclass(frozen=True)
class DualResult(BoundResult):
    """Result of ``compute_certified_bound``. Fields depend on caller flags."""

    margins: torch.Tensor
    sce: Optional[Any] = None
    alpha_state: Optional[AlphaState] = None
    eta_state: Optional[Dict[int, torch.Tensor]] = None
    nu_per_layer: Optional[Dict[int, torch.Tensor]] = None
    dual_margins: Optional[torch.Tensor] = None
    forward_margins: Optional[torch.Tensor] = None

    def row_lower_bounds(self) -> torch.Tensor:
        """Return the certified lower bounds."""
        return self.margins

    def row_witness_inputs(self) -> Optional[torch.Tensor]:
        """Return the optional dual witness inputs."""
        return cast(Optional[torch.Tensor], self.sce)


@dataclass(frozen=True)
class DualBatchResult(BatchBoundResult):
    """Decoded result of :meth:`DualSolver.solve_spec_batch` for one K-lane batch."""

    solution: BatchLPSolution
    margins: torch.Tensor
    lower_bounds: torch.Tensor
    bounds_dict: Optional[Dict[int, Bounds]] = None
    nu_per_layer: Optional[Dict[int, torch.Tensor]] = None
    alpha_state: Optional[AlphaState] = None
    eta_state: Optional[Dict[int, torch.Tensor]] = None
    witness_input: Optional[torch.Tensor] = None
    row_slack: Optional[torch.Tensor] = None
    c_rows: Optional[torch.Tensor] = None
    thresholds: Optional[torch.Tensor] = None
    m_specs: int = 0
    reference_bounds: Optional[Dict[int, Bounds]] = None
    base_reference_bounds: Optional[Dict[int, Bounds]] = None
    dual_row_slack: Optional[torch.Tensor] = None
    forward_only_certified: Optional[torch.Tensor] = None

    def lane_solution(self) -> BatchLPSolution:
        """Return the decoded lane solution."""
        return self.solution

    def lane_lower_bounds(self) -> torch.Tensor:
        """Return the decoded lower bounds."""
        return self.lower_bounds

    def spec_row_slack(self) -> Optional[torch.Tensor]:
        """Return per-row slack when output rows were decoded."""
        return self.row_slack

    def lane_witness_inputs(self) -> Optional[torch.Tensor]:
        """Return lane witness inputs when the relaxation produced them."""
        return self.witness_input

    def intermediate_bounds(self) -> Optional[Dict[int, Bounds]]:
        """Return intermediate bounds requested for branching."""
        return self.bounds_dict

    def neuron_sensitivities(self) -> Optional[Dict[int, torch.Tensor]]:
        """Return dual sensitivities requested for neuron branching."""
        return self.nu_per_layer


def expand_bounds_dict(bounds_dict: Dict[int, Bounds], M: int) -> Dict[int, Bounds]:
    """Expand each batched Bounds entry from [B, *shape] to [B*M, *shape].

    The dual solver threads ``M`` through ``compute_certified_bound`` and
    broadcasts inside the activation handlers (lazy M-broadcast), avoiding the
    M× memory blowup; this explicit expansion is for callers that need a
    materialized M-expanded bounds dict.

    repeat_interleave aligns with row b*M+j sharing sample b's bounds. All
    entries must already be batched (lb.dim() >= 2). M=1 returns the dict
    unchanged.
    """
    if M <= 0:
        raise ValueError(f"expand_bounds_dict: M must be positive, got {M}")
    if M == 1:
        return dict(bounds_dict)
    out: Dict[int, Bounds] = {}
    for lid, bounds in bounds_dict.items():
        if bounds.lb.dim() < 2:
            raise ValueError(
                f"expand_bounds_dict: layer {lid} bounds must be batched "
                f"[B, *shape], got dim={bounds.lb.dim()} shape={tuple(bounds.lb.shape)}"
            )
        out[lid] = Bounds(
            lb=bounds.lb.repeat_interleave(M, dim=0),
            ub=bounds.ub.repeat_interleave(M, dim=0),
        )
    return out


def _alpha_tree_leaves(
    tree: Union[AlphaTree, Dict[int, torch.Tensor]], *, logical: bool = False,
) -> Iterator[torch.Tensor]:
    """Yield the tensor leaves of a dual-alpha pytree.

    A per-layer alpha is a pytree whose shape is interpreted by the matching
    backward kernel: RELU is a single tensor leaf; future per-kind allocators
    may nest leaves in lists/dicts. ``None`` marks a fixed-slope kind with no
    alpha. The optimizer, projection, and keep-best clone all walk these leaves
    so they cannot drift out of agreement on the pytree shape. ``logical``
    counts one ReLU leaf for mixed storage, excluding its lane-map tensors.
    """
    if tree is None:
        return
    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, MixedAlpha):
        yield tree.shared
        if not logical:
            yield tree.per_row
            yield tree.shared_rows
            yield tree.per_row_rows
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _alpha_tree_leaves(value, logical=logical)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _alpha_tree_leaves(value, logical=logical)
    else:
        raise TypeError(f"unsupported alpha pytree node: {type(tree)!r}")


def _alpha_tree_map(
    tree: AlphaTree, transform: Callable[[torch.Tensor], torch.Tensor]
) -> AlphaTree:
    """Transform tensor leaves without changing their per-kernel structure."""
    if tree is None:
        return None
    if isinstance(tree, torch.Tensor):
        return transform(tree)
    if isinstance(tree, MixedAlpha):
        shared, per_row = transform(tree.shared), transform(tree.per_row)
        return MixedAlpha(shared, per_row, tree.shared_rows.to(shared.device),
                          tree.per_row_rows.to(per_row.device))
    if isinstance(tree, dict):
        return {key: _alpha_tree_map(value, transform) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_alpha_tree_map(value, transform) for value in tree)
    raise TypeError(f"unsupported alpha pytree node: {type(tree)!r}")


def _clone_alpha_tree(tree: AlphaTree) -> AlphaTree:
    """Detach-clone evaluated optimizer state without retaining its graph."""
    return _alpha_tree_map(tree, lambda leaf: leaf.detach().clone())


def _alpha_tree_gather_lanes(tree: AlphaTree, indices: torch.Tensor) -> AlphaTree:
    """Gather axis-zero lanes, including repeated indices, into owned storage."""
    if isinstance(tree, MixedAlpha):
        # A dense lane-to-payload lookup is O(K), never O(K*M*N).
        n = tree.shared.shape[0] + tree.per_row.shape[0]
        lookup = torch.empty(n, device=tree.shared.device, dtype=torch.long)
        lookup[tree.shared_rows.to(lookup.device)] = torch.arange(
            tree.shared.shape[0], device=lookup.device)
        lookup[tree.per_row_rows.to(lookup.device)] = -1 - torch.arange(
            tree.per_row.shape[0], device=lookup.device)
        selected = lookup.index_select(0, indices.to(lookup.device))
        shared_rows = torch.where(selected >= 0)[0]
        per_row_rows = torch.where(selected < 0)[0]
        shared = tree.shared.index_select(0, selected[shared_rows])
        per_row = tree.per_row.index_select(0, (-1 - selected[per_row_rows]).to(tree.per_row.device))
        if not per_row_rows.numel():
            return shared
        if not shared_rows.numel():
            return per_row
        return MixedAlpha(shared, per_row, shared_rows, per_row_rows)
    if isinstance(tree, dict):
        return {key: _alpha_tree_gather_lanes(value, indices) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_alpha_tree_gather_lanes(value, indices) for value in tree)
    return _alpha_tree_map(tree, lambda leaf: leaf.index_select(0, indices.to(leaf.device)))


def _alpha_tree_repeat_lanes(tree: AlphaTree, repeats: int) -> AlphaTree:
    """Repeat each lane consecutively, matching repeat_interleave bound rows."""
    if repeats < 1:
        raise ValueError(f"alpha lane repeats must be positive, got {repeats}")
    if isinstance(tree, MixedAlpha):
        count = tree.shared.shape[0] + tree.per_row.shape[0]
        return _alpha_tree_gather_lanes(
            tree, torch.arange(count, device=tree.shared.device).repeat_interleave(repeats))
    if isinstance(tree, dict):
        return {key: _alpha_tree_repeat_lanes(value, repeats) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_alpha_tree_repeat_lanes(value, repeats) for value in tree)
    return _alpha_tree_map(tree, lambda leaf: leaf.repeat_interleave(repeats, dim=0))


def _alpha_tree_zero_lanes(tree: AlphaTree, lanes: int) -> AlphaTree:
    """Preserve payload shapes while zero-initializing missing warm-start lanes."""
    if isinstance(tree, MixedAlpha):
        return tree.shared.new_zeros((lanes, tree.shared.shape[-1]))
    if isinstance(tree, dict):
        return {key: _alpha_tree_zero_lanes(value, lanes) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_alpha_tree_zero_lanes(value, lanes) for value in tree)
    return _alpha_tree_map(tree, lambda leaf: leaf.new_zeros((lanes, *leaf.shape[1:])))


def _alpha_tree_concat_lanes(
    left: AlphaTree, right: AlphaTree, n_left: int, n_right: int
) -> AlphaTree:
    """Concatenate lane state, filling absent subtrees from the present layout."""
    if left is None and right is None:
        return None
    if left is None:
        left = _alpha_tree_zero_lanes(right, n_left)
    if right is None:
        right = _alpha_tree_zero_lanes(left, n_right)
    if isinstance(left, (torch.Tensor, MixedAlpha)) and isinstance(right, (torch.Tensor, MixedAlpha)):
        mixed = isinstance(left, MixedAlpha) or isinstance(right, MixedAlpha)
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            mixed = (
                left.ndim != right.ndim and {left.ndim, right.ndim} == {2, 3}
            ) or (
                left.ndim == right.ndim == 3 and left.shape[1] != right.shape[1]
                and 1 in (left.shape[1], right.shape[1])
            )
        if mixed:
            parts = []
            for value, count in ((left, n_left), (right, n_right)):
                if isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[1] == 1:
                    value = value[:, 0]
                if isinstance(value, MixedAlpha):
                    parts.append(value)
                elif value.ndim == 2:
                    parts.append(MixedAlpha(
                        value, value.new_empty((0, 1, value.shape[-1])),
                        torch.arange(count, device=value.device),
                        torch.empty(0, device=value.device, dtype=torch.long)))
                elif value.ndim == 3:
                    parts.append(MixedAlpha(
                        value.new_empty((0, value.shape[-1])), value,
                        torch.empty(0, device=value.device, dtype=torch.long),
                        torch.arange(count, device=value.device)))
                else:
                    raise ValueError("mixed alpha requires ReLU lane/spec layouts")
            a, b = parts
            m = max(a.per_row.shape[1], b.per_row.shape[1])
            if any(part.per_row.shape[1] not in (1, m) for part in parts):
                raise ValueError("mixed alpha spec-row counts must agree")
            return MixedAlpha(
                torch.cat((a.shared, b.shared.to(a.shared))),
                torch.cat((a.per_row.expand(-1, m, -1), b.per_row.to(a.per_row).expand(-1, m, -1))),
                torch.cat((a.shared_rows, b.shared_rows.to(a.shared_rows) + n_left)),
                torch.cat((a.per_row_rows, b.per_row_rows.to(a.per_row_rows) + n_left)))
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape[1:] != right.shape[1:]:
            raise ValueError("alpha leaf trailing shapes must agree when concatenating lanes")
        return torch.cat((left, right.to(device=left.device, dtype=left.dtype)), dim=0)
    if isinstance(left, dict) and isinstance(right, dict):
        keys = list(left) + [key for key in right if key not in left]
        return {
            key: _alpha_tree_concat_lanes(left.get(key), right.get(key), n_left, n_right)
            for key in keys
        }
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        if len(left) != len(right):
            raise ValueError("alpha sequence lengths must agree when concatenating lanes")
        values = [
            _alpha_tree_concat_lanes(a, b, n_left, n_right)
            for a, b in zip(left, right, strict=True)
        ]
        return tuple(values) if isinstance(left, tuple) else values
    raise TypeError(f"alpha pytree structure mismatch: {type(left)!r} vs {type(right)!r}")


def _alpha_spec_row_count(state: Optional[AlphaState]) -> int:
    """Infer spec rows from ReLU tensors; shared/attention payloads do not fix M."""
    counts = {
        int(tree.shape[1]) for tree in (state or {}).values()
        if isinstance(tree, torch.Tensor) and tree.dim() >= 3 and tree.shape[1] != 1
    }
    counts.update(tree.per_row.shape[1] for tree in (state or {}).values()
                  if isinstance(tree, MixedAlpha) and tree.per_row.shape[1] != 1)
    if len(counts) > 1:
        raise ValueError(f"inconsistent alpha spec-row counts: {sorted(counts)}")
    return next(iter(counts), 1)


def _alpha_tree_select_spec_rows(
    state: Optional[AlphaState], keep_rows: torch.Tensor
) -> Optional[AlphaState]:
    """Select nonshared ReLU spec rows; keep per-kind attention payloads shared."""
    if state is None:
        return None
    return {
        lid: (
            MixedAlpha(tree.shared, tree.per_row.index_select(1, keep_rows.to(tree.per_row.device))
                       if tree.per_row.shape[1] != 1 else tree.per_row,
                       tree.shared_rows, tree.per_row_rows)
            if isinstance(tree, MixedAlpha) else
            tree.index_select(1, keep_rows.to(tree.device))
            if isinstance(tree, torch.Tensor) and tree.dim() >= 3 and tree.shape[1] != 1
            else _alpha_tree_map(tree, lambda leaf: leaf.clone())
        )
        for lid, tree in state.items()
    }


def unproven_spec_rows(
    slack: torch.Tensor,
    out_kind: str = OutKind.LINEAR_LE,
    reference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reject nonfinite proofs and respect the unsafe set's boundary.

    Closed unsafe sets require strictly positive slack. ``reference`` applies
    the shared rounding band; omit it only for already adjusted replay slack.
    LINEAR_LE/RANGE keep their legal equality, without a permissive tolerance.
    """
    if out_kind in OutputSpec.CLOSED_KINDS:
        band = 0.0 if reference is None else certification_tolerance(reference)
        return ~(torch.isfinite(slack) & (slack > band))
    return (slack < 0) | ~torch.isfinite(slack)


def certification_tolerance(reference: torch.Tensor) -> torch.Tensor:
    """Keep strict escape proofs outside the dtype-scaled rounding band."""
    eps = max(100.0 * torch.finfo(reference.dtype).eps, 1e-11)
    return eps * reference.abs().clamp(min=1.0)


def escaping_spec_rows(slack: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """UNSAFE_LINEAR requires at least one finite, strictly escaping row."""
    return ~unproven_spec_rows(slack, OutKind.UNSAFE_LINEAR, reference)


class DualSolver(Solver):
    """Dual (linear-relaxation) certified bounds solver. Strict [B, *shape] API."""

    _PRE_ACTIVATION_BOUNDS_KINDS = {
        LayerKind.RELU.value, LayerKind.LRELU.value, "LEAKY_RELU",
        LayerKind.SIGMOID.value, LayerKind.TANH.value, LayerKind.ERF.value,
        LayerKind.SQRT.value, LayerKind.SIN.value, LayerKind.COS.value,
        LayerKind.QUANTIZE.value, LayerKind.SOFTMAX.value,
        LayerKind.LAYERNORM.value, LayerKind.GELU.value,
    }
    _AFFINE_CONTRIB_KINDS = {
        LayerKind.DENSE.value,
        LayerKind.CONV1D.value,
        LayerKind.CONV2D.value,
        LayerKind.BIAS.value,
        LayerKind.BN.value,
        LayerKind.ADD.value,
    }

    def __init__(self, n_iters: int = 0):
        # DualTF is a backward-handler registry (not a TransferFunction);
        # instantiate it internally so DualSolver is self-contained and callers
        # need no knowledge of it.
        from act.back_end.dual_tf.dual_tf import DualTF
        self.tf = DualTF()
        self.n_iters = n_iters
        self._last_bounds: Optional[Bounds] = None
        self.last_forward_bounds: Optional[Dict[int, Bounds]] = None
        self.last_alpha_iterations = 0
        self.last_alpha_stop_reason = "fixed"

    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=True, supports_csp=False, supports_dual=True)

    def _bilinear_operand_bounds(
        self, net: Net, bounds_dict: Dict[int, Bounds], source_ids: List[int]
    ) -> Dict[int, Bounds]:
        """View activation outputs as boxes without changing stored pre-activation bounds."""
        view = dict(bounds_dict)
        for lid in set(source_ids):
            layer = net.by_id[lid]
            kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if kind not in self._PRE_ACTIVATION_BOUNDS_KINDS:
                continue
            pre = bounds_dict[lid]
            parent = Bounds(pre.lb.flatten(start_dim=1), pre.ub.flatten(start_dim=1))
            lin, frame = _reset_forward_box(
                parent.lb, parent.ub, parent.lb.device, parent.lb.dtype
            )
            handler = self.tf._FORWARD_REGISTRY[kind]
            _, out, _, _ = handler(
                layer, [parent], [lin], [frame], net.preds[lid], True,
                parent.lb.device, parent.lb.dtype,
            )
            view[lid] = out
        return view

    def solve_spec_batch(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        out_spec: OutputSpec,
        *,
        optimize: bool,
        dual_config: DualConfig,
        input_shape: tuple[int, ...],
        keep_rows: Optional[torch.Tensor] = None,
        split_signs: Optional[Dict[int, torch.Tensor]] = None,
        eta: Optional[Dict[int, torch.Tensor]] = None,
        incremental_alphas: Optional[AlphaState] = None,
        incremental_etas: Optional[Dict[int, torch.Tensor]] = None,
        optimize_alpha: bool = True,
        refresh_forward: bool = True,
        return_nu: bool = False,
        reuse_bounds_for_branching: bool = False,
        forward_frame: Optional[ForwardFrame] = None,
    ) -> DualBatchResult:
        """Solve and decode one prepared K-lane output-spec batch.

        Bound construction and refinement are caller policy. This method owns
        output-row encoding, dual optimization, lane status/witness decoding,
        and the optional converged-state nu pass used by neuron branching.
        """
        sample_bounds = next(iter(bounds_dict.values()))
        device = sample_bounds.lb.device
        dtype = sample_bounds.lb.dtype
        if sample_bounds.lb.dim() < 2:
            raise ValueError(
                "DualSolver.solve_spec_batch: bounds_dict entries must be batched "
                f"[K, *shape]; got dim={sample_bounds.lb.dim()}"
            )
        k_actual = int(sample_bounds.lb.shape[0])

        assert_layers = [
            layer
            for layer in net.layers
            if (layer.kind.upper() if isinstance(layer.kind, str) else layer.kind)
            == LayerKind.ASSERT.value
        ]
        if not assert_layers:
            raise ValueError("DualSolver.solve_spec_batch: net has no ASSERT layer")
        assert_layer = assert_layers[0]
        assert_preds = net.preds.get(assert_layer.id, [])
        if len(assert_preds) != 1:
            raise ValueError(
                f"ASSERT layer {assert_layer.id} must have exactly 1 predecessor, "
                f"got {len(assert_preds)}"
            )
        output_bounds = bounds_dict[assert_preds[0]]
        n_out = int(output_bounds.lb.flatten(start_dim=1).shape[-1])
        encoded_spec = out_spec.encode_linear(
            B=k_actual, n_out=n_out, device=device, dtype=dtype
        )
        m_specs = int(encoded_spec["M"])

        if out_spec.kind == OutKind.UNSAFE_LINEAR:
            c_rows = cast(torch.Tensor, encoded_spec["C"]).contiguous()
            thresholds = cast(torch.Tensor, encoded_spec["thresholds"]).contiguous()
        else:
            c_rows = -cast(torch.Tensor, encoded_spec["C"]).contiguous()
            thresholds = -cast(torch.Tensor, encoded_spec["thresholds"]).contiguous()
            if keep_rows is not None:
                idx = keep_rows.to(device=device, dtype=torch.long)
                c_rows = (
                    c_rows.view(k_actual, m_specs, n_out)
                    .index_select(1, idx)
                    .reshape(k_actual * int(idx.numel()), n_out)
                    .contiguous()
                )
                thresholds = thresholds.index_select(1, idx).contiguous()
                m_specs = int(idx.numel())
        active_mask = torch.ones(
            k_actual, m_specs, dtype=torch.bool, device=device
        )

        def lanes_certified(row_bounds: torch.Tensor) -> torch.Tensor:
            """Use the same row rule for final decoding and early stopping."""
            values = row_bounds.reshape(k_actual, m_specs)
            slack = values - thresholds
            proven = ~unproven_spec_rows(
                slack, out_spec.kind, torch.maximum(values.abs(), thresholds.abs())
            )
            if out_spec.kind == OutKind.UNSAFE_LINEAR:
                return (proven & active_mask).any(dim=-1)
            return ~(~proven & active_mask).any(dim=-1)

        compute_certified_bound = cast(Any, self.compute_certified_bound)
        if optimize:
            dual_result = compute_certified_bound(
                net,
                bounds_dict,
                c_rows,
                M=m_specs,
                optimize=True,
                optimize_alpha=optimize_alpha,
                refresh_forward=refresh_forward,
                forward_lin_max_perturbed=dual_config.forward_lin_max_perturbed,
                n_iters=dual_config.n_iters,
                lr_alpha=dual_config.lr_alpha,
                lr_beta=dual_config.lr_beta,
                lr_decay=dual_config.lr_decay,
                eta=eta,
                incremental_alphas=incremental_alphas,
                incremental_etas=incremental_etas,
                split_signs=split_signs,
                return_optimized=True,
                return_sce=True,
                per_class_alpha=dual_config.per_class_alpha,
                forward_frame=forward_frame,
                dual_config=dual_config,
                stop_criterion=(
                    lambda values: bool(lanes_certified(values).all())
                ) if dual_config.stop_when_verified else None,
                **({"return_nu_per_layer": True} if return_nu else {}),
            )
        else:
            dual_result = compute_certified_bound(
                net,
                bounds_dict,
                c_rows,
                M=m_specs,
                return_sce=True,
                forward_frame=forward_frame,
                **({"return_nu_per_layer": True} if return_nu else {}),
            )

        margins = dual_result.margins.view(k_actual, m_specs)
        sce = cast(Optional[torch.Tensor], dual_result.sce)
        slack = margins - thresholds
        certified = lanes_certified(margins)
        assert dual_result.dual_margins is not None
        dual_slack = dual_result.dual_margins.view(k_actual, m_specs) - thresholds
        forward_only_certified = certified & ~lanes_certified(dual_result.dual_margins)
        if out_spec.kind == OutKind.UNSAFE_LINEAR:
            candidate_rows = torch.zeros(
                k_actual, dtype=torch.long, device=device
            )
        else:
            violations = unproven_spec_rows(
                slack, out_spec.kind, torch.maximum(margins.abs(), thresholds.abs())
            ) & active_mask
            candidate_rows = torch.where(
                violations.any(dim=1),
                violations.to(torch.int64).argmax(dim=1),
                torch.zeros(k_actual, dtype=torch.long, device=device),
            )

        statuses = tuple(
            SolveStatus.UNSAT if bool(is_certified.item()) else SolveStatus.SAT
            for is_certified in certified
        )
        nvars = (
            max((max(layer.out_vars) for layer in net.layers if layer.out_vars), default=-1)
            + 1
        )
        x_candidate = torch.zeros(k_actual, nvars, device=device, dtype=dtype)
        input_layers = [
            layer
            for layer in net.layers
            if (layer.kind.upper() if isinstance(layer.kind, str) else layer.kind)
            == LayerKind.INPUT.value
        ]
        if len(input_layers) != 1:
            raise ValueError(
                f"Expected exactly one INPUT layer, found {len(input_layers)}."
            )
        input_ids_list = list(input_layers[0].out_vars)
        input_ids = torch.tensor(input_ids_list, device=device, dtype=torch.long)
        if sce is not None:
            sce_flat = sce.flatten(start_dim=1).to(device=device)
            row_offsets = (
                torch.arange(k_actual, device=device) * m_specs
                + candidate_rows.to(device=device)
            )
            chosen_sce = sce_flat.index_select(0, row_offsets)
            x_candidate[:, input_ids] = chosen_sce.to(device=device, dtype=dtype)
        else:
            statuses = tuple(
                SolveStatus.UNSAT if status == SolveStatus.UNSAT else SolveStatus.UNKNOWN
                for status in statuses
            )
        solution = BatchLPSolution(
            statuses=statuses,
            x=x_candidate,
            max_viol=-(
                slack.max(dim=1).values if out_spec.kind == OutKind.UNSAFE_LINEAR
                else slack.min(dim=1).values
            ).detach(),
        )

        branch_bounds: Optional[Dict[int, Bounds]] = None
        branch_nu: Optional[Dict[int, torch.Tensor]] = None
        if return_nu and reuse_bounds_for_branching:
            branch_bounds = bounds_dict
            branch_nu = dual_result.nu_per_layer
            if branch_nu is None:
                nu_pass = self.compute_certified_bound(
                    net,
                    bounds_dict,
                    c_rows,
                    M=m_specs,
                    alpha=dual_result.alpha_state,
                    eta=dual_result.eta_state if split_signs is not None else None,
                    split_signs=split_signs,
                    return_nu_per_layer=True,
                )
                branch_nu = nu_pass.nu_per_layer
        elif return_nu:
            branch_bounds, branch_nu = self.recompute_bounds_and_nu(
                net,
                bounds_dict,
                c_rows,
                m_specs,
                alpha_state=dual_result.alpha_state,
                eta_state=dual_result.eta_state if split_signs is not None else None,
                split_signs=split_signs,
                per_class_alpha=dual_config.per_class_alpha,
            )

        witness_input = (
            x_candidate[:, input_ids].reshape(k_actual, *input_shape).detach()
            if sce is not None
            else None
        )
        lower_bounds = -solution.max_viol
        return DualBatchResult(
            solution=solution,
            margins=margins,
            lower_bounds=lower_bounds,
            bounds_dict=branch_bounds,
            nu_per_layer=branch_nu,
            alpha_state=dual_result.alpha_state,
            eta_state=dual_result.eta_state,
            witness_input=witness_input,
            row_slack=slack.detach(),
            c_rows=c_rows.detach(),
            thresholds=thresholds.detach(),
            m_specs=m_specs,
            dual_row_slack=dual_slack.detach(),
            forward_only_certified=forward_only_certified.detach(),
            reference_bounds=bounds_dict,
        )

    def compute_certified_bound(
        self, net: Net, bounds_dict: Dict[int, Bounds],
        c: torch.Tensor, M: int = 1,
        return_sce: bool = False,
        enable_grad: bool = False,
        alpha: Optional[AlphaState] = None,
        eta: Optional[Dict[int, torch.Tensor]] = None,
        split_signs: Optional[Union[Dict[int, torch.Tensor], List[Dict[int, torch.Tensor]]]] = None,
        optimize: bool = False,
        n_iters: int = 50,
        lr_alpha: float = 0.1,
        lr_beta: float = 0.1,
        lr_decay: float = 0.98,
        incremental_alphas: Optional[AlphaState] = None,
        incremental_etas: Optional[Dict[int, torch.Tensor]] = None,
        return_optimized: bool = False,
        per_class_alpha: bool = True,
        return_nu_per_layer: bool = False,
        optimize_alpha: bool = True,
        refresh_forward: bool = True,
        start_lid: Optional[int] = None,
        local_phase_clamp: bool = False,
        forward_lin_max_perturbed: Optional[int] = None,
        forward_frame: Optional[ForwardFrame] = None,
        dual_config: Optional[DualConfig] = None,
        stop_criterion: Optional[Callable[[torch.Tensor], bool]] = None,
        phase_slope_signs: Optional[Dict[int, torch.Tensor]] = None,
    ) -> DualResult:
        """Batched certified lower bound on c^T @ output (DAG-aware).

        Implements ``Solver.compute_certified_bound``; see base for the
        full contract. DualSolver realises this via reverse-topological
        backward propagation of a per-layer accumulator:
          nu_accum[lid] = sum over all successors s of ν routed by s's handler to lid.

        Each handler returns per-pred νs; the outer loop distributes them to preds.

        Unknown layer kind raises ValueError (no silent identity fallback for soundness).

        Lazy M-broadcast: ``c`` has shape ``[B*M, num_classes]`` packed
        sample-major (row ``b*M+j`` = sample b's j-th spec row), but
        ``bounds_dict`` entries stay at ``[B, *shape]``. Activation handlers
        (RELU/SIGMOID/TANH) view nu as ``[B, M, n]`` and broadcast bounds
        ``[B, 1, n]`` against it — mathematically equivalent to the legacy
        M-expanded path, with M× lower bounds memory.

        Caveats:
            Uses GLOBAL intermediate bounds — bounds_dict stays at [B, *shape] across
            BaB lanes within the same sample (lazy-M-broadcast design). Per-lane
            intermediate-bound tightening (a stricter dual variant) is OUT OF SCOPE
            for this solver; bounds may be looser than a per-lane refinement would
            give, but the dual lower bound remains SOUND.

        When ``optimize=True``, runs joint Adam optimization over ReLU α/η via
        ``_optimize_alpha_eta``. If ``return_optimized=True``, also returns the optimized
        α state for incremental-starting. When ``optimize=False``, executes a single-pass
        backward; with ``alpha=None`` it uses the default fixed-slope relaxation.

        eta: Per-layer η multiplier — Lagrange (KKT) multiplier for branch-split
            constraints on activation pre-activations. Keyed by activation layer id
            (RELU, LRELU, SIGMOID, TANH, GELU). η ≥ 0 invariant (enforced by clamp).
        split_signs: Per-layer split direction. {-1: inactive, +1: active, 0: unsplit}.
            Same key set as eta.
        local_phase_clamp: For fixed-parameter replay, shallow-copy and clamp only
            the current ReLU handler's bounds. Reference bounds remain immutable;
            optimization is rejected because it requires globally hardened bounds.
        phase_slope_signs: For fixed-parameter replay of a reduced core, the
            source literals. A literal absent from ``split_signs`` keeps its
            source slope on a line valid over the candidate interval; its
            multiplier and phase clamp stay removed.
        η is applied to the TRUE pre-activation variable (immediately AFTER the
        activation handler in the reverse-topological backward loop):
        nu_pre = slope · nu_post − η · sign, so the multiplier acts on the
        affine pre-activation unscaled. Applying it before the handler would
        scale η by the relaxation slope, forcing the effective multiplier to 0
        on inactive-split (slope = 0) neurons and discarding the z ≤ 0
        constraint's input-region information. Sound for any η ≥ 0.

        forward_lin_max_perturbed: ``None`` resolves to the DualConfig default
            at call time rather than being frozen at import time.
        forward_frame: a valid output relation for this domain. Combined
            margins may use it; ``dual_margins`` always keeps backward evidence.
        dual_config: optional stopping policy; direct optimizer arguments
            still determine iterations and learning rates.
        """
        if alpha is not None:
            alpha = {
                lid: tree.materialize(M) if isinstance(tree, MixedAlpha) else tree
                for lid, tree in alpha.items()
            }
        if forward_lin_max_perturbed is None:
            forward_lin_max_perturbed = DualConfig().forward_lin_max_perturbed
        if isinstance(split_signs, list):
            # KFSB path: accept K split hypotheses and return stacked margins
            # [K, N, ...], evaluating each hypothesis through the single-hypothesis
            # backward pass (no leading-K vectorization).
            if not split_signs:
                raise ValueError("split_signs list cannot be empty")
            if optimize or phase_slope_signs is not None:
                raise ValueError(
                    "list-form split_signs requires optimize=False and no phase_slope_signs"
                )
            if return_optimized:
                raise ValueError("return_optimized requires optimize=True and single split_signs")
            stacked_split_signs = self._stack_split_sign_hypotheses(split_signs)
            normalized_split_signs = self._unstack_split_sign_hypotheses(stacked_split_signs)
            margins = []
            dual_margins = []
            sce_values = []
            for hypo in normalized_split_signs:
                result = self.compute_certified_bound(
                    net,
                    bounds_dict,
                    c,
                    M=M,
                    return_sce=return_sce,
                    enable_grad=enable_grad,
                    alpha=alpha,
                    eta=eta,
                    split_signs=hypo,
                    optimize=False,
                    n_iters=n_iters,
                    lr_alpha=lr_alpha,
                    lr_beta=lr_beta,
                    lr_decay=lr_decay,
                    incremental_alphas=incremental_alphas,
                    incremental_etas=incremental_etas,
                    return_optimized=False,
                    per_class_alpha=per_class_alpha,
                    return_nu_per_layer=False,
                    local_phase_clamp=local_phase_clamp,
                    forward_lin_max_perturbed=forward_lin_max_perturbed,
                    forward_frame=forward_frame,
                )
                margins.append(result.margins)
                assert result.dual_margins is not None
                dual_margins.append(result.dual_margins)
                if return_sce:
                    sce_values.append(result.sce)
            stacked_sce = None
            if return_sce and sce_values and all(sce is not None for sce in sce_values):
                stacked_sce = torch.stack(cast(List[torch.Tensor], sce_values), dim=0)
            return DualResult(
                margins=torch.stack(margins, dim=0), sce=stacked_sce,
                dual_margins=torch.stack(dual_margins, dim=0),
            )

        if optimize and (local_phase_clamp or phase_slope_signs is not None):
            raise ValueError(
                "local_phase_clamp and phase_slope_signs are only valid for fixed-parameter replay"
            )
        if not local_phase_clamp:
            bounds_dict = self._harden_split_bounds(bounds_dict, split_signs)

        if optimize:
            result = self._optimize_alpha_eta(
                net,
                bounds_dict,
                c,
                M=M,
                n_iters=n_iters,
                lr_alpha=lr_alpha,
                lr_beta=lr_beta,
                lr_decay=lr_decay,
                incremental_alphas=incremental_alphas,
                incremental_etas=incremental_etas,
                split_signs=split_signs,
                return_sce=return_sce,
                per_class_alpha=per_class_alpha,
                optimize_alpha=optimize_alpha,
                refresh_forward=refresh_forward,
                start_lid=start_lid,
                forward_lin_max_perturbed=forward_lin_max_perturbed,
                forward_frame=forward_frame,
                dual_config=dual_config,
                stop_criterion=stop_criterion,
            )
            if return_optimized:
                return result
            return DualResult(
                margins=result.margins, sce=result.sce,
                dual_margins=result.dual_margins,
                forward_margins=result.forward_margins,
            )

        if c.dim() != 2:
            raise ValueError(
                f"c must be 2-D [B*M, num_classes], got shape {tuple(c.shape)}. "
                "Use c.unsqueeze(0) for single instance.")
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")
        if c.shape[0] % M != 0:
            raise ValueError(
                f"c batch dim {c.shape[0]} not divisible by M={M}; "
                f"expected c.shape[0] == B*M for some integer B"
            )
        with torch.set_grad_enabled(enable_grad):
            assert len(bounds_dict) > 0, "bounds_dict cannot be empty"
            device, dtype = get_default_device(), get_default_dtype()
            if c.dtype != dtype or c.device != device:
                c = c.to(device=device, dtype=dtype)
            B = c.shape[0]

            if start_lid is not None:
                # Interior start: c is a linear functional on layer
                # ``start_lid``'s output; only its ancestors are visited by
                # the backward loop (non-ancestors never enter nu_accum).
                # Used by refine_intermediate_bounds.
                output_lid = start_lid
            else:
                assert_layer = None
                for layer in net.layers:
                    k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
                    if k == LayerKind.ASSERT.value:
                        assert_layer = layer
                        break
                if assert_layer is None:
                    raise ValueError("DualSolver.compute_certified_bound: net has no ASSERT layer")

                assert_preds = net.preds.get(assert_layer.id, [])
                if len(assert_preds) != 1:
                    raise ValueError(
                        f"DualSolver.compute_certified_bound: ASSERT layer {assert_layer.id} must have "
                        f"exactly 1 predecessor, got {len(assert_preds)}"
                    )

                output_lid = assert_preds[0]
            nu_accum: Dict[int, torch.Tensor] = {output_lid: c.clone()}
            nu_snapshot: Dict[int, torch.Tensor] = {}
            obj = torch.zeros(B, dtype=c.dtype, device=c.device)

            topo_order = get_topo_order(net, reverse=True)
            registry = self.tf._BACKWARD_REGISTRY

            for lid in topo_order:
                layer = net.by_id[lid]
                k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind

                if k in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, LayerKind.ASSERT.value):
                    continue

                if lid not in nu_accum:
                    continue

                nu_here = nu_accum.pop(lid)
                handler = registry.get(k)
                if handler is None:
                    raise ValueError(
                        f"DualSolver.compute_certified_bound: unknown layer kind '{k}' at layer {lid}; "
                        f"soundness requires explicit backward handler. "
                        f"Supported kinds: {sorted(registry.keys())}"
                    )

                if return_nu_per_layer and k == LayerKind.RELU.value:
                    nu_snapshot[lid] = nu_here.detach().clone()

                preds = list(net.preds.get(lid, []))
                handler_bounds = bounds_dict
                if (
                    local_phase_clamp
                    and k == LayerKind.RELU.value
                    and split_signs is not None
                    and lid in split_signs
                ):
                    handler_bounds = self._local_phase_bounds(
                        bounds_dict, lid, split_signs[lid]
                    )
                if k in (LayerKind.MUL.value, LayerKind.MATMUL.value):
                    handler_bounds = self._bilinear_operand_bounds(
                        net, handler_bounds, preds
                    )
                elif k in (LayerKind.ATT_SCORES.value, LayerKind.ATT_MIX.value):
                    source_keys = (
                        ("q_src", "k_src")
                        if k == LayerKind.ATT_SCORES.value else ("w_src", "v_src")
                    )
                    handler_bounds = self._bilinear_operand_bounds(
                        net, handler_bounds,
                        [cast(int, layer.params[key]) for key in source_keys],
                    )
                strip = None
                if (
                    phase_slope_signs is not None
                    and split_signs is not None
                    and k == LayerKind.RELU.value
                    and lid in phase_slope_signs
                ):
                    nu_here, *strip = _source_phase_strip(
                        nu_here, bounds_dict[lid], split_signs[lid],
                        phase_slope_signs[lid], M,
                    )
                if alpha is None:
                    pred_nus, contrib = handler(layer, nu_here, handler_bounds, preds, M)
                else:
                    pred_nus, contrib = handler(
                        layer, nu_here, handler_bounds, preds, M, alpha=alpha.get(lid)
                    )
                if strip is not None:
                    pred_nus = [pred_nus[0] + strip[0], *pred_nus[1:]]
                    contrib = contrib + strip[1]

                if eta is not None and split_signs is not None and lid in eta:
                    # Split Lagrangian on the TRUE pre-activation variable:
                    # nu_pre = D nu_post - eta * sign.
                    # Applying it BEFORE the handler would scale eta by the
                    # relaxation slope D - forcing beta = 0 on inactive-split
                    # (D = 0) neurons, which silently discards the z <= 0
                    # constraint's restriction of the input region (the only
                    # channel carrying it: the y = 0 relaxation says nothing
                    # about x). Sound for any eta >= 0: on the child region
                    # sign * z >= 0, so -eta * sign * z <= 0 only lowers the
                    # minimized Lagrangian below the true child minimum.
                    eta_l = eta[lid].to(device=nu_here.device, dtype=nu_here.dtype)
                    signs_l = split_signs[lid].to(device=nu_here.device, dtype=nu_here.dtype)
                    if eta_l.dim() == 3:
                        if eta_l.shape[1] == 1:
                            eta_l = eta_l.expand(-1, M, -1)
                        eta_l = eta_l.reshape(-1, eta_l.shape[-1])
                    if signs_l.dim() == 3:
                        if signs_l.shape[1] == 1:
                            signs_l = signs_l.expand(-1, M, -1)
                        signs_l = signs_l.reshape(-1, signs_l.shape[-1])
                    pn = pred_nus[0]
                    n_clip = min(pn.shape[-1], eta_l.shape[-1])
                    pn = pn.clone()
                    pn[..., :n_clip] = (
                        pn[..., :n_clip]
                        - eta_l[..., :n_clip] * signs_l[..., :n_clip]
                    )
                    pred_nus = [pn, *pred_nus[1:]]

                if len(pred_nus) != len(preds):
                    raise ValueError(
                        f"handler {k} at layer {lid} returned {len(pred_nus)} pred_nus, "
                        f"expected {len(preds)}"
                    )
                if contrib.shape != (B,):
                    raise ValueError(
                        f"handler {k} at layer {lid} contrib shape {tuple(contrib.shape)}, "
                        f"expected ({B},)"
                    )

                if k in self._AFFINE_CONTRIB_KINDS:
                    contrib = -contrib

                obj = obj + contrib
                for pred_id, pred_nu in zip(preds, pred_nus):
                    if pred_id in nu_accum:
                        nu_accum[pred_id] = nu_accum[pred_id] + pred_nu
                    else:
                        nu_accum[pred_id] = pred_nu.clone()

            input_lid = self._find_input_layer_id(net)
            nu_final = nu_accum.get(input_lid) if input_lid is not None else None
            sce = None
            if input_lid is not None and nu_final is not None:
                input_contrib, sce = self._input_contribution_from_nu(
                    net, input_lid, nu_final, bounds_dict, M=M,
                    return_sce=return_sce, enable_grad=enable_grad,
                )
                obj = obj + input_contrib
            forward_margins = None
            if start_lid is None:
                # Activation entries can hold PRE-activation bounds; only
                # known affine outputs may use the stored box directly.
                if net.by_id[output_lid].kind in self._AFFINE_CONTRIB_KINDS:
                    output_box = bounds_dict[output_lid]
                    rows = c.reshape(B // M, M, -1)
                    forward_margins = (
                        (rows.clamp(min=0) * output_box.lb.flatten(1).unsqueeze(1)).sum(-1)
                        + (rows.clamp(max=0) * output_box.ub.flatten(1).unsqueeze(1)).sum(-1)
                    ).reshape(B)
                if forward_frame is not None:
                    if forward_frame.lid != output_lid:
                        raise ValueError("Forward frame does not describe the objective layer")
                    frame_bound = forward_frame_row_lower_bounds(forward_frame, c, M)
                    forward_margins = (
                        frame_bound if forward_margins is None
                        else torch.maximum(forward_margins, frame_bound)
                    )
            return DualResult(
                margins=obj if forward_margins is None else torch.maximum(obj, forward_margins),
                sce=sce if return_sce else None,
                nu_per_layer=nu_snapshot if return_nu_per_layer else None,
                dual_margins=obj,
                forward_margins=forward_margins,
            )

    def _stack_split_sign_hypotheses(
        self,
        split_signs: List[Dict[int, torch.Tensor]],
    ) -> Dict[int, torch.Tensor]:
        layer_ids = sorted({lid for hypo in split_signs for lid in hypo})
        stacked: Dict[int, torch.Tensor] = {}
        for lid in layer_ids:
            template = next(hypo[lid] for hypo in split_signs if lid in hypo)
            entries = [hypo.get(lid, torch.zeros_like(template)) for hypo in split_signs]
            stacked[lid] = torch.stack(entries, dim=0)
        return stacked

    def _unstack_split_sign_hypotheses(
        self,
        stacked_split_signs: Dict[int, torch.Tensor],
    ) -> List[Dict[int, torch.Tensor]]:
        first = next(iter(stacked_split_signs.values()))
        hypotheses: List[Dict[int, torch.Tensor]] = []
        for idx in range(first.shape[0]):
            hypotheses.append({lid: signs[idx] for lid, signs in stacked_split_signs.items()})
        return hypotheses

    def _init_alpha(
        self,
        net: Net,
        layer: Layer,
        bounds_dict: Dict[int, Bounds],
        B: int,
        M: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        per_class_alpha: bool,
        optimize_alpha: bool,
        incremental_alphas: Optional[AlphaState],
    ) -> AlphaTree:
        """Per-kind dual-alpha allocation returning a pytree of leaves, or None.

        RELU allocates an optimizable lower-envelope slope ``[B, M, n]`` (or
        ``[B, n]`` when ``per_class_alpha`` is off), warm-started from
        ``incremental_alphas`` when present. SOFTMAX and LAYERNORM are
        fixed-slope and allocate no alpha; attention cores and MATMUL allocate
        fusion slopes. Each kind owns the shape of its own pytree, which its
        backward kernel reads back via
        ``alpha.get(lid)``.
        """
        k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        if k in (LayerKind.ATT_SCORES.value, LayerKind.ATT_MIX.value, LayerKind.MATMUL.value):
            return self._init_bilinear_alpha(
                net, layer, bounds_dict, device, dtype,
                optimize_alpha=optimize_alpha,
                incremental_alphas=incremental_alphas,
            )
        if k != LayerKind.RELU.value:
            return None
        b = bounds_dict.get(layer.id)
        assert b is not None, f"_init_alpha: no bounds for RELU layer {layer.id}"
        if incremental_alphas is not None and layer.id in incremental_alphas:
            previous = incremental_alphas[layer.id]
            if isinstance(previous, MixedAlpha):
                previous = previous.materialize(M, shared=not per_class_alpha)
            assert isinstance(previous, torch.Tensor)
            if not per_class_alpha and previous.ndim == 3:
                previous = previous.mean(dim=1)
            elif per_class_alpha and previous.ndim == 2:
                previous = previous.unsqueeze(1).expand(B, M, -1)
            alpha_init = (
                previous
                .detach()
                .clone()
                .to(device=device, dtype=dtype)
                .clamp(0.0, 1.0)
            )
        else:
            lb_flat = b.lb.to(device=device, dtype=dtype).flatten(start_dim=1)
            ub_flat = b.ub.to(device=device, dtype=dtype).flatten(start_dim=1)
            n_neurons = lb_flat.shape[-1]
            denom = (ub_flat - lb_flat).clamp(min=1e-12)
            alpha_init_bn = (ub_flat / denom).clamp(0.0, 1.0).detach()
            if per_class_alpha:
                alpha_init = (
                    alpha_init_bn.unsqueeze(1)
                    .expand(B, M, n_neurons)
                    .contiguous()
                )
            else:
                alpha_init = alpha_init_bn.contiguous()
        return torch.nn.Parameter(alpha_init) if optimize_alpha else alpha_init.detach()

    def _init_bilinear_alpha(
        self,
        net: Net,
        layer: Layer,
        bounds_dict: Dict[int, Bounds],
        device: torch.device,
        dtype: torch.dtype,
        *,
        optimize_alpha: bool,
        incremental_alphas: Optional[AlphaState],
    ) -> AlphaTree:
        """Allocate lane-major fusion slopes for attention or MATMUL.

        The pytree is ``{omega_l, omega_u}`` with ``[B, outputs]`` slopes,
        warm-started at the rule init derived from the same local input boxes the
        backward kernel reads (so allocator, kernel, keep-best clone and the
        ``[0, 1]`` projection all agree on the shape). Returns ``None`` when the
        predecessor boxes are missing so the kernel falls back to its rule slope.
        """
        from act.back_end.dual_tf.tf_transformer import (
            attention_rule_alpha, _attention_input_boxes, matmul_rule_alpha, _matmul_shapes,
        )
        kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        if kind == LayerKind.MATMUL.value:
            source_ids = net.preds[layer.id]
        else:
            src_keys = ("q_src", "k_src") if kind == "ATT_SCORES" else ("w_src", "v_src")
            source_ids = [cast(int, layer.params[key]) for key in src_keys]
        if any(lid not in bounds_dict for lid in source_ids):
            # Missing predecessor bounds leave the kernel at its fixed rule.
            return None
        operands = self._bilinear_operand_bounds(net, bounds_dict, source_ids)
        if kind == LayerKind.MATMUL.value:
            x_box, y_box = (operands[lid] for lid in source_ids)
            x_l, x_u, y_l, y_u = (box.flatten(1) for box in (x_box.lb, x_box.ub, y_box.lb, y_box.ub))
        else:
            x_l, x_u, y_l, y_u, _scale, _mask = _attention_input_boxes(layer, operands)
        x_l = x_l.to(device=device, dtype=dtype)
        x_u = x_u.to(device=device, dtype=dtype)
        y_l = y_l.to(device=device, dtype=dtype)
        y_u = y_u.to(device=device, dtype=dtype)
        if incremental_alphas is not None and layer.id in incremental_alphas:
            prior = cast(Dict[str, torch.Tensor], cast(object, incremental_alphas[layer.id]))
            tree = {
                "omega_l": prior["omega_l"].detach().clone().to(device=device, dtype=dtype).clamp(0.0, 1.0),
                "omega_u": prior["omega_u"].detach().clone().to(device=device, dtype=dtype).clamp(0.0, 1.0),
            }
        else:
            k_thresh = float(cast(float, layer.params.get("k_thresh", 1.0)))
            if kind == LayerKind.MATMUL.value:
                shape = _matmul_shapes(layer.params["x_shape"], layer.params["y_shape"])
                tree = matmul_rule_alpha(x_l, x_u, y_l, y_u, *shape, k_thresh)
            else:
                tree = attention_rule_alpha(x_l, x_u, y_l, y_u, k_thresh)
        if optimize_alpha:
            return {key: torch.nn.Parameter(val.detach().clone()) for key, val in tree.items()}
        return {key: val.detach() for key, val in tree.items()}

    def _optimize_alpha_eta(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        c: torch.Tensor,
        M: int = 1,
        n_iters: int = 50,
        lr_alpha: float = 0.1,
        lr_beta: float = 0.1,
        lr_decay: float = 0.98,
        incremental_alphas: Optional[AlphaState] = None,
        incremental_etas: Optional[Dict[int, torch.Tensor]] = None,
        split_signs: Optional[Dict[int, torch.Tensor]] = None,
        return_sce: bool = False,
        per_class_alpha: bool = True,
        optimize_alpha: bool = True,
        refresh_forward: bool = True,
        start_lid: Optional[int] = None,
        forward_lin_max_perturbed: Optional[int] = None,
        forward_frame: Optional[ForwardFrame] = None,
        dual_config: Optional[DualConfig] = None,
        stop_criterion: Optional[Callable[[torch.Tensor], bool]] = None,
    ) -> DualResult:
        """Joint α/η optimization: iterative dual lower-bound refinement.

        Each ReLU gets a learnable lower-envelope slope α constrained to [0, 1].
        Each split layer with a nonzero split sign gets a learnable η multiplier
        constrained to η ≥ 0.

        Keep independent best backward/forward bounds. Snapshots are taken
        before Adam updates, and shared alpha states are replayed before they
        become CLIMB evidence. Stop checks run after a completed evaluation.
        """
        policy = dual_config if dual_config is not None else DualConfig()
        self.last_alpha_iterations = 0
        self.last_alpha_stop_reason = "fixed"
        if forward_lin_max_perturbed is None:
            forward_lin_max_perturbed = DualConfig().forward_lin_max_perturbed
        if c.dim() != 2:
            raise ValueError(
                f"c must be 2-D [B*M, n_out], got shape {tuple(c.shape)}"
            )
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")
        BM = c.shape[0]
        if BM % M != 0:
            raise ValueError(
                f"c batch dim {BM} not divisible by M={M}; expected B*M rows"
            )
        B = BM // M
        device, dtype = c.device, c.dtype

        input_lid = self._find_input_layer_id(net)
        if input_lid is None:
            raise ValueError("DualSolver._optimize_alpha_eta: net has no INPUT/INPUT_SPEC layer")
        by_id = getattr(net, "by_id", {layer.id: layer for layer in net.layers})
        input_layer = by_id[input_lid]
        input_bounds = bounds_dict.get(input_lid)
        if input_bounds is None:
            if "lb" not in input_layer.params or "ub" not in input_layer.params:
                raise ValueError(
                        f"DualSolver._optimize_alpha_eta: input layer {input_lid} has no bounds"
                )
            input_lb = cast(torch.Tensor, input_layer.params["lb"])
            input_ub = cast(torch.Tensor, input_layer.params["ub"])
        else:
            input_lb = input_bounds.lb
            input_ub = input_bounds.ub
        input_lb = input_lb.to(device=device, dtype=dtype)
        input_ub = input_ub.to(device=device, dtype=dtype)

        ancestor_lids: Optional[set[int]] = None
        if start_lid is not None:
            # Interior-start objectives only depend on ancestor layers; alpha
            # parameters outside that cone receive no gradient and would crash
            # the optimizer.
            ancestor_lids = {start_lid}
            stack = [start_lid]
            while stack:
                for p in net.preds.get(stack.pop(), []):
                    if p not in ancestor_lids:
                        ancestor_lids.add(p)
                        stack.append(p)

        alphas: AlphaState = {}
        for layer in net.layers:
            if ancestor_lids is not None and layer.id not in ancestor_lids:
                continue
            tree = self._init_alpha(
                net, layer, bounds_dict, B, M, device, dtype,
                per_class_alpha=per_class_alpha,
                optimize_alpha=optimize_alpha,
                incremental_alphas=incremental_alphas,
            )
            if tree is not None:
                alphas[layer.id] = tree

        etas: Dict[int, torch.nn.Parameter] = {}
        if split_signs is not None:
            for lid, signs in split_signs.items():
                signs_init = signs.detach().to(device=device, dtype=dtype)
                if not (signs_init != 0).any():
                    continue
                if incremental_etas is not None and lid in incremental_etas:
                    eta_init = (
                        incremental_etas[lid]
                        .detach()
                        .clone()
                        .to(device=device, dtype=dtype)
                        .clamp(min=0)
                    )
                    if eta_init.shape != signs_init.shape:
                        eta_init = torch.zeros_like(signs_init)
                else:
                    eta_init = torch.zeros_like(signs_init)
                etas[lid] = torch.nn.Parameter(eta_init)

        param_groups: List[Dict[str, object]] = []
        alpha_params = [
            leaf
            for tree in alphas.values()
            for leaf in _alpha_tree_leaves(tree)
            if isinstance(leaf, torch.nn.Parameter)
        ]
        eta_params = list(etas.values())
        if alpha_params:
            param_groups.append({"params": alpha_params, "lr": lr_alpha})
        if eta_params:
            param_groups.append({"params": eta_params, "lr": lr_beta})
        if not param_groups or n_iters <= 0:
            result = self.compute_certified_bound(
                net, bounds_dict, c, M=M, return_sce=return_sce,
                alpha=alphas if alphas else None,
                eta=cast(Dict[int, torch.Tensor], etas) if etas else None,
                split_signs=split_signs, start_lid=start_lid,
                forward_frame=forward_frame,
            )
            self.last_alpha_stop_reason = "n_iters" if n_iters <= 0 else "fixed"
            return DualResult(
                margins=result.margins.detach(), sce=result.sce,
                alpha_state={lid: _clone_alpha_tree(tree) for lid, tree in alphas.items()},
                eta_state={lid: value.detach().clone() for lid, value in etas.items()},
                dual_margins=result.dual_margins,
                forward_margins=result.forward_margins,
            )
        optimizer = torch.optim.Adam(param_groups)
        scheduler = (
            torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
            if lr_decay < 1.0
            else None
        )
        best_bounds = torch.full((BM,), float("-inf"), device=device, dtype=dtype)
        best_forward: Optional[torch.Tensor] = None
        best_sce: Optional[torch.Tensor] = None
        best_alpha_state: AlphaState = {
            lid: _clone_alpha_tree(tree) for lid, tree in alphas.items()
        }
        best_eta_state: Dict[int, torch.Tensor] = {
            lid: e.detach().clone() for lid, e in etas.items()
        }

        from act.back_end.dual_tf.tf_forward import compute_forward_bounds

        forward_alphas = {
            lid: value[:, 0, :] if value.dim() == 3 else value
            for lid, value in alphas.items()
            if isinstance(value, torch.Tensor)
        }
        started = _monotonic() if policy.max_time > 0 else 0.0
        prefix = ForwardPrefixCache(
            net, input_lb, input_ub, alpha_relu_ids=set(forward_alphas),
            forward_lin_max_perturbed=forward_lin_max_perturbed,
        ) if refresh_forward else None
        flat_iterations = 0
        self.last_alpha_stop_reason = "n_iters"
        with torch.enable_grad():
            for _ in range(n_iters):
                optimizer.zero_grad()
                eta_tensors = cast(Dict[int, torch.Tensor], etas)
                if refresh_forward:
                    fresh_bounds, current_frame, _stats = compute_forward_bounds(
                        net,
                        input_lb,
                        input_ub,
                        post_activation=False,
                        alphas=forward_alphas,
                        forward_lin_max_perturbed=forward_lin_max_perturbed,
                        prefix_cache=prefix,
                    )
                else:
                    # Fixed intermediate bounds (root-reuse mode): alpha/eta
                    # only enter the backward pass; sound for any valid
                    # bounds_dict.
                    fresh_bounds = bounds_dict
                    current_frame = forward_frame
                result = self.compute_certified_bound(
                    net,
                    fresh_bounds,
                    c,
                    M=M,
                    return_sce=return_sce,
                    enable_grad=True,
                    alpha=alphas,
                    eta=eta_tensors,
                    split_signs=split_signs,
                    start_lid=start_lid,
                    forward_frame=current_frame if start_lid is None else None,
                )
                assert result.dual_margins is not None
                bound_bm = result.dual_margins
                sce = result.sce

                with torch.no_grad():
                    if result.forward_margins is not None:
                        values = result.forward_margins.detach()
                        best_forward = values if best_forward is None else torch.maximum(best_forward, values)
                    detached = bound_bm.detach()
                    improved = torch.isfinite(detached) & (detached > best_bounds)
                    gained = torch.isfinite(detached) & (detached > best_bounds + policy.stagnation_tol)
                    flat_iterations = 0 if bool(gained.any()) else flat_iterations + 1
                    best_bounds = torch.where(improved, detached, best_bounds)
                    rows_improved = improved.reshape(B, M)
                    if bool(improved.any()):
                        for lid, tree in alphas.items():
                            evaluated = _clone_alpha_tree(tree)
                            for old, new in zip(
                                _alpha_tree_leaves(best_alpha_state[lid]),
                                _alpha_tree_leaves(evaluated),
                            ):
                                if new.dim() and new.shape[0] == B:
                                    per_row = (
                                        per_class_alpha and net.by_id[lid].kind == LayerKind.RELU.value
                                        and new.dim() == 3 and new.shape[1] == M
                                    )
                                    mask = rows_improved if per_row else rows_improved.any(dim=1)
                                    new.copy_(torch.where(
                                        mask.reshape(*mask.shape, *((1,) * (new.dim() - mask.dim()))),
                                        new, old,
                                    ))
                            best_alpha_state[lid] = evaluated
                        for lid, value in etas.items():
                            per_row = value.dim() == 3 and value.shape[:2] == (B, M)
                            mask = rows_improved if per_row else rows_improved.any(dim=1)
                            best_eta_state[lid] = torch.where(
                                mask.reshape(*mask.shape, *((1,) * (value.dim() - mask.dim()))),
                                value.detach(), best_eta_state[lid],
                            ).clone()
                    if return_sce and sce is not None:
                        if best_sce is None:
                            best_sce = sce.detach().clone()
                        else:
                            best_sce[improved] = sce[improved].detach()
                    self.last_alpha_iterations += 1
                    combined = best_bounds if best_forward is None else torch.maximum(best_bounds, best_forward)
                    if stop_criterion is not None and stop_criterion(combined):
                        self.last_alpha_stop_reason = "verified"
                        break
                    if policy.stagnation_patience > 0 and flat_iterations >= policy.stagnation_patience:
                        self.last_alpha_stop_reason = "stagnation"
                        break
                    if policy.max_time > 0 and _monotonic() - started >= policy.max_time:
                        self.last_alpha_stop_reason = "time"
                        break
                    if not bound_bm.requires_grad:
                        self.last_alpha_stop_reason = "fixed"
                        break
                    if not bool(torch.isfinite(bound_bm).all()):
                        self.last_alpha_stop_reason = "nonfinite"
                        break
                (-bound_bm.sum()).backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                with torch.no_grad():
                    for tree in alphas.values():
                        for leaf in _alpha_tree_leaves(tree):
                            leaf.clamp_(0.0, 1.0)
                    for e in etas.values():
                        e.clamp_(min=0)

        return DualResult(
            margins=best_bounds if best_forward is None else torch.maximum(best_bounds, best_forward),
            sce=best_sce, alpha_state=best_alpha_state, eta_state=best_eta_state,
            dual_margins=best_bounds, forward_margins=best_forward,
        )

    def _harden_split_bounds(
        self,
        bounds_dict: Dict[int, Bounds],
        split_signs: Optional[Dict[int, torch.Tensor]],
    ) -> Dict[int, Bounds]:
        """Fix split ReLU phases in the relaxation itself.

        sign=+1 asserts pre-activation >= 0 (lb clamped to 0: the relaxation
        collapses to the exact identity); sign=-1 asserts <= 0 (ub clamped to
        0: exact zero). Applied alongside the eta Lagrangian term, the split
        tightens the bound through both channels. Sound: clamping encodes
        exactly the branch's split assumption.
        """
        if not split_signs or isinstance(split_signs, list):
            return bounds_dict
        out = dict(bounds_dict)
        for lid, signs in split_signs.items():
            b = out.get(lid)
            if b is None:
                continue
            s = signs[:, 0, :] if signs.dim() == 3 else signs
            if not bool((s != 0).any().item()):
                continue
            lb = b.lb.flatten(start_dim=1).clone()
            ub = b.ub.flatten(start_dim=1).clone()
            n = min(lb.shape[-1], s.shape[-1])
            s_n = s[..., :n].to(device=lb.device)
            lb[..., :n] = torch.where(s_n > 0, lb[..., :n].clamp(min=0.0), lb[..., :n])
            ub[..., :n] = torch.where(s_n < 0, ub[..., :n].clamp(max=0.0), ub[..., :n])
            # An infeasible branch (split contradicts a stable phase) yields a
            # degenerate [x, x] interval; the branch represents an empty input
            # region, so any bound for it is sound.
            ub[..., :n] = torch.maximum(ub[..., :n], lb[..., :n])
            out[lid] = Bounds(lb.view_as(b.lb), ub.view_as(b.ub))
        return out

    def _local_phase_bounds(
        self,
        bounds_dict: Dict[int, Bounds],
        lid: int,
        signs: torch.Tensor,
    ) -> Dict[int, Bounds]:
        """Clamp one ReLU entry for its handler without mutating reference bounds."""
        bounds = bounds_dict.get(lid)
        if bounds is None:
            return bounds_dict
        phase = signs[:, 0, :] if signs.dim() == 3 else signs
        if not bool((phase != 0).any().item()):
            return bounds_dict
        lb = bounds.lb.flatten(start_dim=1).clone()
        ub = bounds.ub.flatten(start_dim=1).clone()
        n = min(lb.shape[-1], phase.shape[-1])
        phase = phase[..., :n].to(device=lb.device)
        lb[..., :n] = torch.where(
            phase > 0, lb[..., :n].clamp(min=0.0), lb[..., :n]
        )
        ub[..., :n] = torch.where(
            phase < 0, ub[..., :n].clamp(max=0.0), ub[..., :n]
        )
        ub[..., :n] = torch.maximum(ub[..., :n], lb[..., :n])
        local = dict(bounds_dict)
        local[lid] = Bounds(lb.view_as(bounds.lb), ub.view_as(bounds.ub))
        return local

    def refine_intermediate_bounds(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        mode: str = "auto",
        blowup_ratio: float = 10.0,
        max_rows_per_call: int = 4096,
        optimize_iters: int = 20,
        max_tensor_mib: float = 0.0,
    ) -> Dict[int, Bounds]:
        """Metric-driven backward refinement of selected pre-activation bounds.

        Forward-mode concretization loses correlation at wide fan-in affine
        layers; a backward pass from the affected layer to the input keeps it.
        Selection is architecture-agnostic: an activation layer qualifies if it
        has unstable neurons and (mode="auto") its mean pre-activation width
        exceeds ``blowup_ratio`` x the median width of all activation layers;
        mode="all" refines every unstable activation layer. Refined bounds are
        intersected with the forward bounds (both are valid over-approximations,
        so the intersection is sound). Layers are processed in topological
        order so later refinements consume earlier ones.
        """
        if mode == "none":
            return bounds_dict
        if mode not in ("auto", "all", "tail"):
            raise ValueError(
                f"intermediate_refine mode must be none|auto|all|tail, got {mode!r}"
            )

        stats = []
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k != LayerKind.RELU.value or layer.id not in bounds_dict:
                continue
            b = bounds_dict[layer.id]
            lb, ub = b.lb.flatten(start_dim=1), b.ub.flatten(start_dim=1)
            unstable = int(((lb < 0) & (ub > 0)).sum().item())
            stats.append((layer.id, unstable, float((ub - lb).mean().item())))
        if not stats:
            return bounds_dict

        median_width = sorted(s[2] for s in stats)[len(stats) // 2]
        threshold = max(median_width, 1e-9) * blowup_ratio
        if mode == "tail":
            unstable_lids = [lid for lid, unstable, _ in stats if unstable > 0]
            selected = unstable_lids[-2:]
        else:
            selected = [
                lid for lid, unstable, width in stats
                if unstable > 0 and (mode == "all" or width > threshold)
            ]
        if not selected:
            return bounds_dict

        max_rows_per_call, _ = self._refinement_chunks(
            bounds_dict, max_rows_per_call, 1, max_tensor_mib, optimize_iters)
        if max_rows_per_call == 0:
            return bounds_dict
        out = dict(bounds_dict)
        for lid in selected:
            preds = net.preds.get(lid, [])
            if len(preds) != 1:
                continue
            pred_lid = preds[0]
            b = out[lid]
            lb0 = b.lb.flatten(start_dim=1)
            ub0 = b.ub.flatten(start_dim=1)
            if lb0.shape[0] != 1:
                continue
            n = lb0.shape[-1]
            device, dtype = lb0.device, lb0.dtype
            # Only unstable neurons need refinement: stable phases make the
            # relaxation exact regardless of bound width, so querying them
            # would spend backward rows for zero tightening.
            amb_idx = torch.where((lb0[0] < 0) & (ub0[0] > 0))[0]
            n_amb = int(amb_idx.numel())
            if n_amb == 0:
                continue
            lb_new = torch.empty(n_amb, device=device, dtype=dtype)
            ub_new = torch.empty(n_amb, device=device, dtype=dtype)
            for s in range(0, n_amb, max_rows_per_call):
                e = min(s + max_rows_per_call, n_amb)
                eye = torch.zeros(e - s, n, device=device, dtype=dtype)
                eye[torch.arange(e - s), amb_idx[s:e]] = 1.0
                rows = torch.cat([eye, -eye], dim=0)
                res = self.compute_certified_bound(
                    net, out, rows.contiguous(), M=int(rows.shape[0]),
                    start_lid=pred_lid,
                    optimize=optimize_iters > 0,
                    n_iters=optimize_iters,
                    lr_alpha=0.25,
                    lr_decay=0.98,
                    per_class_alpha=True,
                    refresh_forward=False,
                )
                lb_new[s:e] = res.margins[: e - s]
                ub_new[s:e] = -res.margins[e - s:]
            lb_ref = lb0[0].clone()
            ub_ref = ub0[0].clone()
            lb_ref[amb_idx], ub_ref[amb_idx] = _intersect_boxes(lb_ref[amb_idx], ub_ref[amb_idx], lb_new, ub_new)
            ub_ref = torch.maximum(ub_ref, lb_ref)
            refined = Bounds(
                lb_ref.view_as(b.lb[0]).unsqueeze(0).clone(),
                ub_ref.view_as(b.ub[0]).unsqueeze(0).clone(),
            )
            out[lid] = refined
            if pred_lid in out and out[pred_lid].lb.shape == refined.lb.shape:
                out[pred_lid] = refined
        return out

    def refine_intermediate_bounds_batched(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        split_signs: Optional[Dict[int, torch.Tensor]] = None,
        mode: str = "tail",
        rows_cap: int = 64,
        optimize_iters: int = 0,
        lane_chunk: int = 32,
        max_tensor_mib: float = 0.0,
    ) -> Dict[int, Bounds]:
        """K-lane per-subproblem sparse refinement of pre-activation bounds.

        Batched counterpart of ``refine_intermediate_bounds`` for the BaB loop:
        ``bounds_dict`` entries are ``[K, *shape]`` (one lane per subproblem).
        Split constraints are applied by hardening the bounds FIRST (sign=+1
        clamps lb to 0, sign=-1 clamps ub to 0); the backward pass then sees
        the hardened relaxation slopes, which propagates each lane's splits
        relationally to downstream layers - the tightening that the interval
        refresh cannot provide. ``split_signs`` is NOT forwarded to
        ``compute_certified_bound`` (its eta machinery is shaped for the final
        spec's M, not the refine rows); hardening alone carries the split.

        Rows are the per-neuron one-hot +/- queries for the UNION of unstable
        neurons across lanes, capped at ``rows_cap`` by descending interval
        width. Each refined bound is intersected per lane with the existing
        bound (both are valid over-approximations: sound). Layers are visited
        in topological order so later refinements consume earlier ones.
        """
        if mode == "none":
            return bounds_dict
        if mode not in ("tail", "all"):
            raise ValueError(
                f"per_subproblem_refine mode must be none|tail|all, got {mode!r}"
            )

        out = self._harden_split_bounds(bounds_dict, split_signs)
        rows_cap, lane_chunk = self._refinement_chunks(
            out, rows_cap, lane_chunk, max_tensor_mib, optimize_iters)
        if rows_cap == 0:
            return out

        stats: List[tuple[int, int]] = []
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k != LayerKind.RELU.value or layer.id not in out:
                continue
            b = out[layer.id]
            lb, ub = b.lb.flatten(start_dim=1), b.ub.flatten(start_dim=1)
            n_unstable = int(((lb < 0) & (ub > 0)).any(dim=0).sum().item())
            stats.append((layer.id, n_unstable))
        unstable_lids = [lid for lid, n_unstable in stats if n_unstable > 0]
        if not unstable_lids:
            return out
        selected = unstable_lids[-2:] if mode == "tail" else unstable_lids

        for lid in selected:
            preds = net.preds.get(lid, [])
            if len(preds) != 1:
                continue
            pred_lid = preds[0]
            b = out[lid]
            lb0 = b.lb.flatten(start_dim=1)
            ub0 = b.ub.flatten(start_dim=1)
            k_lanes = lb0.shape[0]
            n = lb0.shape[-1]
            device, dtype = lb0.device, lb0.dtype
            amb_union = ((lb0 < 0) & (ub0 > 0)).any(dim=0)
            amb_idx = torch.where(amb_union)[0]
            n_amb = int(amb_idx.numel())
            if n_amb == 0:
                continue
            if n_amb > rows_cap:
                width = (ub0 - lb0).amax(dim=0)[amb_idx]
                amb_idx = amb_idx[torch.topk(width, k=rows_cap).indices]
                n_amb = rows_cap
            eye = torch.zeros(n_amb, n, device=device, dtype=dtype)
            eye[torch.arange(n_amb), amb_idx] = 1.0
            rows = torch.cat([eye, -eye], dim=0)
            m_rows = int(rows.shape[0])
            margins = torch.empty(k_lanes, m_rows, device=device, dtype=dtype)
            for k0 in range(0, k_lanes, lane_chunk):
                k1 = min(k0 + lane_chunk, k_lanes)
                sub = {
                    l: Bounds(bb.lb[k0:k1], bb.ub[k0:k1]) for l, bb in out.items()
                }
                c = rows.repeat(k1 - k0, 1).contiguous()
                res = self.compute_certified_bound(
                    net, sub, c, M=m_rows,
                    start_lid=pred_lid,
                    optimize=optimize_iters > 0,
                    n_iters=optimize_iters,
                    lr_alpha=0.25,
                    lr_decay=0.98,
                    per_class_alpha=True,
                    refresh_forward=False,
                )
                margins[k0:k1] = res.margins.view(k1 - k0, m_rows)
            lb_new = margins[:, :n_amb]
            ub_new = -margins[:, n_amb:]
            lb_ref = lb0.clone()
            ub_ref = ub0.clone()
            lb_ref[:, amb_idx], ub_ref[:, amb_idx] = _intersect_boxes(lb_ref[:, amb_idx], ub_ref[:, amb_idx], lb_new, ub_new)
            ub_ref = torch.maximum(ub_ref, lb_ref)
            refined = Bounds(
                lb_ref.view_as(b.lb).clone(),
                ub_ref.view_as(b.ub).clone(),
            )
            out[lid] = refined
            if pred_lid in out and out[pred_lid].lb.shape == refined.lb.shape:
                out[pred_lid] = refined
        return out

    @staticmethod
    def _refinement_chunks(
        bounds_dict: Dict[int, Bounds], rows: int, lanes: int,
        max_tensor_mib: float, optimize_iters: int,
    ) -> tuple[int, int]:
        """Limit existing refinement chunks before allocating query tensors.

        Both signs and optimizer state are included in the conservative
        per-query estimate. If even one query cannot fit, retain the already
        valid bounds instead of skipping a necessary verification solve.
        """
        if not math.isfinite(max_tensor_mib) or max_tensor_mib < 0:
            raise ValueError("max_tensor_mib must be finite and non-negative")
        if max_tensor_mib == 0:
            return rows, lanes
        per_query = sum(
            box.lb[0].numel() * box.lb.element_size()
            for box in bounds_dict.values()
        ) * (16 if optimize_iters > 0 else 4)
        capacity = int(max_tensor_mib * 1024**2) // max(1, per_query)
        if capacity == 0:
            return 0, 0
        rows = min(rows, capacity)
        return rows, min(lanes, max(1, capacity // rows))

    def recompute_bounds_and_nu(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        c: torch.Tensor,
        M: int,
        alpha_state: Optional[AlphaState] = None,
        eta_state: Optional[Dict[int, torch.Tensor]] = None,
        split_signs: Optional[Dict[int, torch.Tensor]] = None,
        per_class_alpha: bool = True,
    ) -> Tuple[Dict[int, Bounds], Optional[Dict[int, torch.Tensor]]]:
        """Forward bounds and per-RELU ν at the converged (α, η), from one pass.

        BaBSR/FSB scoring pairs each RELU's slope/intercept (from interval bounds)
        with its backward multiplier ν. Both MUST come from the same forward pass or
        the heuristic mixes an un-optimized interval with an optimized multiplier.
        Returns ``(fresh_bounds, nu_per_layer)``; soundness is unaffected (the
        certified bound is produced separately — this output is heuristic-only).
        """
        from act.back_end.dual_tf.tf_forward import compute_forward_bounds

        device, dtype = c.device, c.dtype
        input_lid = self._find_input_layer_id(net)
        assert input_lid is not None, "recompute_bounds_and_nu: net has no INPUT layer"
        input_bounds = bounds_dict.get(input_lid)
        assert input_bounds is not None, (
            f"recompute_bounds_and_nu: no bounds for INPUT layer {input_lid}"
        )
        input_lb = input_bounds.lb.to(device=device, dtype=dtype)
        input_ub = input_bounds.ub.to(device=device, dtype=dtype)

        with torch.no_grad():
            forward_alphas = (
                {lid: (a[:, 0, :] if a.dim() == 3 else a)
                 for lid, a in alpha_state.items() if isinstance(a, torch.Tensor)}
                if alpha_state
                else None
            )
            fresh_bounds, forward_frame, _stats = compute_forward_bounds(
                net, input_lb, input_ub, post_activation=False, alphas=forward_alphas,
            )
            result = self.compute_certified_bound(
                net,
                fresh_bounds,
                c,
                M=M,
                enable_grad=False,
                alpha=alpha_state if alpha_state else None,
                eta=eta_state if eta_state else None,
                split_signs=split_signs,
                return_nu_per_layer=True,
                forward_frame=forward_frame,
            )
        return fresh_bounds, result.nu_per_layer

    def evaluate_spec(
        self, net: Net,
        out_spec: OutputSpec,
        bounds_dict: Optional[Dict[int, Bounds]] = None,
        num_classes: Optional[int] = None,
        chunk_size: Optional[int] = None,
        enable_grad: bool = False,
        collect_bounds: bool = False,
    ) -> SpecBatchResult:
        """Dual bound evaluation for any OutputSpec — self-contained entry point.

        Refactor note: ``bounds_dict`` is optional. When omitted (the typical
        case), the solver gathers the net's INPUT_SPEC seed bounds and computes
        per-layer pre-activation forward bounds internally via
        ``compute_forward_bounds(post_activation=False)``. Callers who already
        have a bounds_dict (e.g. BaB refinement loops) may pass it explicitly to
        skip the recomputation. When ``collect_bounds`` is true, the solver
        stores the same dict on ``last_forward_bounds`` so post-verification
        soundness checks validate exactly the bounds used by the dual certificate.

        Strategy: dispatch on ``out_spec.kind`` into two branches that share
        ``compute_certified_bound`` but use opposite sign conventions and
        opposite row aggregators.

        - ALL-rows kinds (LINEAR_LE, TOP1_ROBUST, MARGIN_ROBUST, RANGE):
          ``encode_linear`` emits (C, thresholds) in UB-cert form.
          Pass ``-C`` / ``-thresholds`` to
          ``compute_certified_bound`` and compare using the unsafe-set boundary:
          nonnegative slack for LINEAR_LE/RANGE, strictly escaping slack for
          TOP1/MARGIN. Certified iff every active row passes (``.all()``).
        - EXISTS-row kind (UNSAFE_LINEAR): the unsafe polytope is
          ``P = {y : c_i^T y <= d_i for ALL i}``. SAFE iff for all reachable
          y, some row i satisfies ``c_i^T y > d_i`` (escape). Sound
          strengthening via quantifier swap (mirrors ``verifier.py:574-580``):
          certify SAFE iff there exists a row i with ``LB_dual(c_i^T y) > d_i``.
          ``encode_linear`` emits UNSAFE_LINEAR in LB-cert form, so pass ``+C``
          / ``+thresholds`` directly (no sign flip). Certified iff any row
          escapes (``.any()``).

        Raises:
            ValueError: if net lacks ASSERT layer, ASSERT has != 1 predecessor,
                or (when bounds_dict is supplied) the output layer's bounds are
                missing / unbatched.
        """
        forward_frame: Optional[ForwardFrame] = None
        if bounds_dict is None:
            from act.back_end.dual_tf.tf_forward import compute_forward_bounds
            from act.back_end.verifier import (
                gather_input_spec_layers,
                seed_from_input_specs,
            )
            spec_layers = gather_input_spec_layers(net)
            seed_bounds = seed_from_input_specs(spec_layers)
            bounds_dict, forward_frame, _stats = compute_forward_bounds(
                net, seed_bounds.lb, seed_bounds.ub, post_activation=False,
            )

        if collect_bounds:
            self.last_forward_bounds = bounds_dict

        sample = next(iter(bounds_dict.values()))
        device = sample.lb.device
        dtype = sample.lb.dtype
        # Soundness tolerance for the certify comparison: a dual bound whose
        # slack is within float rounding of the threshold must yield UNKNOWN,
        # not CERTIFIED (a tolerance-free `slack < 0` lets a boundary-case
        # margin that rounds slightly positive falsely certify — observed as
        # concrete_ce=FOUND + verifier=CERTIFIED on netfactory CNNs).
        # 100 ulp = the same arithmetic-noise-floor convention as the
        # per-neuron 'auto' bounds tolerance (act/pipeline/cli.py
        # _per_neuron_config): pairwise-reduction drift of the largest
        # layers is ~log2(n)*eps. The 1e-11 floor additionally covers the
        # ACCUMULATED rounding of the dual bound computation itself over
        # deep conv chains in float64 (~1e-12): netfactory random draws can
        # land a true margin within that band of zero, and a 100-ulp-only
        # f64 tolerance (2.2e-14) was observed to false-CERTIFY such a draw
        # in CI while a concrete counterexample existed. float32 is
        # unaffected (100 ulp = 1.2e-5 > 1e-11).
        if sample.lb.dim() < 2:
            raise ValueError(
                "DualSolver.evaluate_spec: bounds_dict entries must be batched "
                f"[B, *shape]; got dim={sample.lb.dim()}"
            )
        B = sample.lb.shape[0]

        assert_layer = None
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k == LayerKind.ASSERT.value:
                assert_layer = layer
                break
        if assert_layer is None:
            raise ValueError("DualSolver.evaluate_spec: net has no ASSERT layer")
        assert_preds = net.preds.get(assert_layer.id, [])
        if len(assert_preds) != 1:
            raise ValueError(
                f"ASSERT layer must have exactly 1 predecessor, got {len(assert_preds)}"
            )
        output_lid = assert_preds[0]
        if output_lid not in bounds_dict:
            raise ValueError(
                f"DualSolver.evaluate_spec: bounds_dict missing output layer "
                f"{output_lid} (ASSERT predecessor); run forward analysis first."
            )
        out_bounds = bounds_dict[output_lid]
        if out_bounds.lb.dim() < 2:
            raise ValueError(
                f"DualSolver.evaluate_spec: output layer {output_lid} bounds "
                f"must be batched; got dim={out_bounds.lb.dim()}"
            )
        n_out = int(out_bounds.lb.flatten(start_dim=1).shape[-1])

        if out_spec.kind == OutKind.UNSAFE_LINEAR:
            # EXISTS-row branch. encode_linear emits LB-cert form for
            # UNSAFE_LINEAR (specs.py:179-201) — pass +C / +thresholds
            # directly. Certified iff any row escapes the unsafe polytope.
            # Slack semantics is ASYMMETRIC vs ALL-rows kinds below:
            # here ``slack > 0`` means the row certifies; ``min_slack`` is
            # NOT a meaningful summary (use ``slack.max(dim=-1)`` instead).
            fe_params = out_spec.encode_linear(B=B, n_out=n_out, device=device, dtype=dtype)
            C = fe_params["C"].contiguous()
            thresholds = fe_params["thresholds"].contiguous()
            N = int(fe_params["M"])
            active_mask = torch.ones(B, N, dtype=torch.bool, device=device)

            with torch.set_grad_enabled(enable_grad):
                if chunk_size is None or N <= chunk_size:
                    result = self.compute_certified_bound(
                        net, bounds_dict, C, M=N, enable_grad=enable_grad,
                        forward_frame=forward_frame,
                    )
                    margins_flat = result.margins
                else:
                    margins_flat = self._chunked_eval(
                        net, bounds_dict, C, B, N, n_out, chunk_size, enable_grad,
                        forward_frame=forward_frame,
                    )
                margins = margins_flat.view(B, N)
                slack = margins - thresholds
                certified = (escaping_spec_rows(
                    slack, torch.maximum(margins.abs(), thresholds.abs())
                ) & active_mask).any(dim=-1)

            return SpecBatchResult(
                margins=margins,
                slack=slack,
                active_mask=active_mask,
                certified=certified,
            )

        fe_params = out_spec.encode_linear(B=B, n_out=n_out, device=device, dtype=dtype)
        C_neg = -fe_params["C"].contiguous()
        thresholds_neg = -fe_params["thresholds"].contiguous()
        M = int(fe_params["M"])
        active_mask = torch.ones(B, M, dtype=torch.bool, device=device)

        with torch.set_grad_enabled(enable_grad):
            if chunk_size is None or M <= chunk_size:
                result = self.compute_certified_bound(
                    net, bounds_dict, C_neg, M=M, enable_grad=enable_grad,
                    forward_frame=forward_frame,
                )
                margins_flat = result.margins
            else:
                margins_flat = self._chunked_eval(
                    net, bounds_dict, C_neg, B, M, n_out, chunk_size, enable_grad,
                    forward_frame=forward_frame,
                )

            margins = margins_flat.view(B, M)
            slack = margins - thresholds_neg
            violations = unproven_spec_rows(
                slack, out_spec.kind,
                torch.maximum(margins.abs(), thresholds_neg.abs()),
            ) & active_mask
            certified = ~violations.any(dim=-1)

        return SpecBatchResult(
            margins=margins,
            slack=slack,
            active_mask=active_mask,
            certified=certified,
        )

    def _chunked_eval(
        self, net: Net, bounds_dict: Dict[int, Bounds],
        C_neg: torch.Tensor, B: int, M: int, n_out: int,
        chunk_size: int, enable_grad: bool,
        forward_frame: Optional[ForwardFrame] = None,
    ) -> torch.Tensor:
        """Evaluate sign-flipped C in chunks along the M dimension.

        For large M (e.g. CIFAR-100 K=100), trades time for memory by
        processing chunk_size specs per sample at a time.

        Chunked evaluation invariant: slicing the leading B*M axis at arbitrary
        chunk_size is bit-identical to unchunked evaluation, because each
        (sample, spec) row is fully independent in the dual backward pass — no
        cross-row computation exists within a chunk or across chunk boundaries.
        """
        C_view = C_neg.view(B, M, n_out)
        chunks: List[torch.Tensor] = []
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            m_chunk = end - start
            # Slice specs [start:end] for all B samples — independent rows, invariant-safe.
            C_chunk = C_view[:, start:end, :].reshape(B * m_chunk, n_out).contiguous()
            result = self.compute_certified_bound(
                net, bounds_dict, C_chunk, M=m_chunk, enable_grad=enable_grad,
                forward_frame=forward_frame,
            )
            chunks.append(result.margins.view(B, m_chunk))
        return torch.cat(chunks, dim=1).reshape(B * M)

    def compute_robust_bound(
        self, net: Net, bounds_dict: Dict[int, Bounds],
        y_true: Union[int, torch.Tensor], num_classes: int,
        margin: float = 0.0,
        return_full: bool = False,
        enable_grad: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], SpecBatchResult]:
        """Dual certified robust bound for classification (top-1 or margin).

        Unified via evaluate_spec(). Retained as a first-class API for robust
        training loops and existing verification callers.

        Args:
            net: the ACT Net with an ASSERT layer.
            bounds_dict: layer bounds from forward analysis.
            y_true: [B] true class labels, or scalar for uniform label.
            num_classes: K (output dim of network's ASSERT predecessor).
            margin: if > 0 use MARGIN_ROBUST semantics (require y_t - y_j >= margin);
                    else use TOP1_ROBUST (require y_t - y_j >= 0).
            return_full: if True, return the full SpecBatchResult (has per-class
                         [B, K] margins useful for training losses). If False,
                         return legacy tuple (min_slack: Tensor[B], certified: Tensor[B] bool).
            enable_grad: if True, allow gradients to flow through the computation
                         (for robust training). Default False (inference/verification).

        Returns:
            SpecBatchResult if return_full else (Tensor[B], Tensor[B] bool).
        """
        sample = next(iter(bounds_dict.values()))
        device = sample.lb.device
        if isinstance(y_true, int):
            B = sample.lb.shape[0] if sample.lb.dim() >= 2 else 1
            y_true_t = torch.full((B,), y_true, dtype=torch.long, device=device)
        else:
            y_true_t = y_true.to(device=device, dtype=torch.long)

        kind = OutKind.MARGIN_ROBUST if margin > 0 else OutKind.TOP1_ROBUST
        out_spec = OutputSpec(
            kind=kind,
            y_true=y_true_t,
            margin=(
                torch.as_tensor([margin], device=device, dtype=sample.lb.dtype)
                if margin > 0
                else None
            ),
        )
        result = self.evaluate_spec(
            net, out_spec,
            bounds_dict=bounds_dict,
            num_classes=num_classes,
            enable_grad=enable_grad,
        )
        if return_full:
            return result
        return result.min_slack, result.certified

    def _find_input_layer_id(self, net: Net) -> Optional[int]:
        """Return the INPUT_SPEC layer id if present, else INPUT's id, else None."""
        input_spec_id = None
        input_id = None
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k == LayerKind.INPUT_SPEC.value:
                input_spec_id = layer.id
            elif k == LayerKind.INPUT.value:
                input_id = layer.id
        return input_spec_id if input_spec_id is not None else input_id

    def _input_contribution_from_nu(self, net: Net, input_lid: int,
                                    nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                                    M: int = 1,
                                    return_sce: bool = False,
                                    enable_grad: bool = False
                                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Bound the same linear form over the root balls and the current box.

        The returned candidate minimizes a root ball, not necessarily its
        intersection with a child box. Concrete input/spec checks, not this
        relaxation, decide whether a candidate is a counterexample.
        """
        with torch.set_grad_enabled(enable_grad):
            BM = nu.shape[0]
            if M <= 0 or BM % M:
                raise ValueError(f"Input coefficient batch {BM} is incompatible with M={M}")
            B = BM // M
            input_layer = net.by_id[input_lid]
            bounds = bounds_dict.get(input_lid)
            if bounds is None:
                lb, ub = input_layer.get_tensor("lb"), input_layer.get_tensor("ub")
                if lb is None or ub is None:
                    raise ValueError(f"Input layer {input_lid} has no bounds")
            else:
                lb, ub = bounds.lb, bounds.ub
            box = Bounds(
                lb.flatten(1) if lb.dim() >= 2 else lb.unsqueeze(0),
                ub.flatten(1) if ub.dim() >= 2 else ub.unsqueeze(0),
            )
            coefficients = nu.reshape(B, M, -1)
            geometry = net.input_geometry
            if geometry is not None:
                geometry = geometry.to(nu)
            if geometry is None or geometry.p == float("inf"):
                # Preserve the existing backward box reduction order.
                lower = (
                    (coefficients.clamp(min=0) * box.lb.unsqueeze(1)).sum(-1)
                    + (coefficients.clamp(max=0) * box.ub.unsqueeze(1)).sum(-1)
                )
            else:
                lower, _ = input_support(coefficients, Frame(box, geometry))
            if not return_sce:
                return lower.reshape(BM), None

            if geometry is None or geometry.p == float("inf"):
                candidate = torch.where(
                    coefficients > 0, box.lb.unsqueeze(1), box.ub.unsqueeze(1)
                )
            else:
                coeff = coefficients.reshape(B, M, -1, geometry.embed_dim)
                if geometry.q == float("inf"):
                    index = coeff.abs().argmax(dim=-1, keepdim=True)
                    direction = torch.zeros_like(coeff).scatter(
                        -1, index, -coeff.gather(-1, index).sign()
                    )
                else:
                    norm = lp_norm(coeff, geometry.q, keepdim=True)
                    unit = coeff.abs() / torch.where(norm == 0, 1.0, norm)
                    direction = -coeff.sign() * unit.pow(geometry.q - 1)
                center = geometry.center.to(nu).reshape(geometry.center.shape[0], 1, -1, geometry.embed_dim)
                radii = geometry.radii.to(nu).reshape(geometry.center.shape[0], 1, -1, 1)
                candidate = center + radii * direction
            tail = lb.shape[1:] if lb.dim() >= 2 else lb.shape
            return lower.reshape(BM), candidate.reshape(BM, *tail)
