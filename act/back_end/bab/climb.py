# ===- act/back_end/bab/climb.py - CLIMB Certificate Reuse ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Query-local CLIMB certificate replay, core learning, and phase propagation.
#   ClimbSession owns one verification run's CLIMB state and BaB-loop hooks.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

import bisect
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeAlias

import torch

from act.back_end.bab.branching.bounding import TopKBounding
from act.back_end.bab.node import (
    SubproblemBatch,
    _layer_neuron_count,
    slice_bounds_dict,
)
from act.back_end.core import Bounds, Net
from act.back_end.solver.solver_base import SolveStatus
from act.back_end.solver.solver_dual import DualBatchResult
from act.config.config import (
    BaBConfig,
    CLIMB_SOLVER_TIER,
    ConfigError,
    NEURON_BRANCHING_METHODS,
    NO_REFINEMENT_MODE,
    TOP_K_BOUNDINGS,
)
from act.front_end.specs import OutKind
from act.util.device_manager import get_default_device, get_default_dtype


# ---------------------------------------------------------------------------
# CLIMB certificate reuse
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiteralCodec:
    """Immutable DIMACS-style codec for the network's ReLU phase literals."""

    layer_ids: Tuple[int, ...]
    widths: Tuple[int, ...]
    offsets: Tuple[int, ...]

    @classmethod
    def from_net(cls, net: Net) -> LiteralCodec:
        layers = sorted(
            (layer for layer in net.layers if layer.kind == "RELU"),
            key=lambda layer: layer.id,
        )
        layer_ids = tuple(layer.id for layer in layers)
        widths = tuple(_layer_neuron_count(layer) for layer in layers)
        offsets: List[int] = []
        total = 0
        for width in widths:
            offsets.append(total)
            total += width
        return cls(layer_ids, widths, tuple(offsets))

    def __post_init__(self) -> None:
        if len(self.layer_ids) != len(self.widths) or len(self.widths) != len(self.offsets):
            raise ValueError("literal codec tables must have equal lengths")
        if tuple(sorted(self.layer_ids)) != self.layer_ids:
            raise ValueError("literal codec layer ids must be increasing")
        expected = 0
        for width, offset in zip(self.widths, self.offsets):
            if width < 0 or offset != expected:
                raise ValueError("literal codec offsets must be contiguous")
            expected += width

    def width(self, layer_id: int) -> int:
        index = bisect.bisect_left(self.layer_ids, layer_id)
        if index == len(self.layer_ids) or self.layer_ids[index] != layer_id:
            raise KeyError(layer_id)
        return self.widths[index]

    def encode(self, layer_id: int, neuron: int, sign: int) -> int:
        index = bisect.bisect_left(self.layer_ids, layer_id)
        if index == len(self.layer_ids) or self.layer_ids[index] != layer_id:
            raise KeyError(layer_id)
        if neuron < 0 or neuron >= self.widths[index]:
            raise IndexError(neuron)
        if sign not in (-1, 1):
            raise ValueError("literal sign must be -1 or 1")
        return sign * (self.offsets[index] + neuron + 1)

    def decode(self, lit: int) -> tuple[int, int, int]:
        if lit == 0:
            raise ValueError("zero is literal padding")
        variable = abs(lit) - 1
        index = bisect.bisect_right(self.offsets, variable) - 1
        if index < 0 or variable >= self.offsets[index] + self.widths[index]:
            raise IndexError(variable)
        return self.layer_ids[index], variable - self.offsets[index], 1 if lit > 0 else -1

    def encode_tensor(
        self, layer_ids: torch.Tensor, neurons: torch.Tensor, signs: torch.Tensor
    ) -> torch.Tensor:
        table_ids = torch.tensor(self.layer_ids, dtype=torch.long, device=layer_ids.device)
        table_offsets = torch.tensor(self.offsets, dtype=torch.long, device=layer_ids.device)
        indices = torch.searchsorted(table_ids, layer_ids.to(torch.long))
        return signs.to(torch.long) * (
            table_offsets.index_select(0, indices) + neurons.to(torch.long) + 1
        )

    def decode_tensor(
        self, literals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        variables = literals.abs() - 1
        ends = torch.tensor(
            tuple(offset + width for offset, width in zip(self.offsets, self.widths)),
            dtype=torch.long,
            device=literals.device,
        )
        offsets = torch.tensor(self.offsets, dtype=torch.long, device=literals.device)
        layer_ids = torch.tensor(self.layer_ids, dtype=torch.long, device=literals.device)
        indices = torch.searchsorted(ends, variables.clamp(min=0), right=True)
        flat_indices = indices.reshape(-1)
        return (
            layer_ids.index_select(0, flat_indices).reshape_as(literals),
            variables - offsets.index_select(0, flat_indices).reshape_as(literals),
            literals.sign(),
        )


Core: TypeAlias = frozenset[int]


def _core_has_no_complements(core: Core) -> bool:
    return len(core) == len({abs(literal) for literal in core})


def _counters_non_decreasing(
    before: tuple[int, int, int], after: tuple[int, int, int]
) -> bool:
    return all(current >= previous for previous, current in zip(before, after))


def _retained_literals_are_subset(
    original: torch.Tensor, retained: torch.Tensor
) -> bool:
    return original.shape == retained.shape and bool(
        ((retained == 0) | (retained == original)).all()
    )


def _only_fills_unassigned(initial: torch.Tensor, final: torch.Tensor) -> bool:
    return initial.shape == final.shape and bool(
        ((initial == 0) | (final == initial)).all()
    )


@dataclass
class ClimbMetrics:
    enabled: bool = False
    generated_children: int = 0
    main_bound_row_passes: int = 0
    main_bound_calls: int = 0
    peak_frontier_rows: int = 0
    prebound_discharged: int = 0
    presplit_discharged: int = 0
    recheck_row_passes: int = 0
    recheck_calls: int = 0
    recheck_time_s: float = 0.0
    coarsen_time_s: float = 0.0
    propagate_time_s: float = 0.0
    cores_inserted: int = 0
    core_literals_original: int = 0
    core_literals_retained: int = 0
    library: Optional[CoreLibrary] = None

    def observe_frontier(self, pending_plus_active_rows: int) -> None:
        self.peak_frontier_rows = max(self.peak_frontier_rows, pending_plus_active_rows)

    def metadata(self) -> Dict[str, int | float]:
        if not self.enabled:
            return {}
        return {
            "climb_generated_children": self.generated_children,
            "climb_main_bound_row_passes": self.main_bound_row_passes,
            "climb_main_bound_calls": self.main_bound_calls,
            "climb_peak_frontier_rows": self.peak_frontier_rows,
            "climb_prebound_discharged": self.prebound_discharged,
            "climb_presplit_discharged": self.presplit_discharged,
            "climb_recheck_row_passes": self.recheck_row_passes,
            "climb_recheck_calls": self.recheck_calls,
            "climb_recheck_time_s": self.recheck_time_s,
            "climb_coarsen_time_s": self.coarsen_time_s,
            "climb_propagate_time_s": self.propagate_time_s,
            "climb_cores_inserted": self.cores_inserted,
            "climb_cores_subsumed": self.library.subsumed if self.library is not None else 0,
            "climb_cores_resolved": self.library.resolved if self.library is not None else 0,
            "climb_cores_evicted": self.library.evicted if self.library is not None else 0,
            "climb_core_literals_original": self.core_literals_original,
            "climb_core_literals_retained": self.core_literals_retained,
        }


def pack_split_matrix(
    split_signs: Optional[Dict[int, torch.Tensor]],
    n_rows: int,
    codec: LiteralCodec,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Pack split-state dictionaries into zero-padded signed literal rows."""
    target_device = (
        next(iter(split_signs.values())).device
        if split_signs
        else (device if device is not None else get_default_device())
    )
    row_parts: List[torch.Tensor] = []
    literal_parts: List[torch.Tensor] = []
    count_parts: List[torch.Tensor] = []
    if split_signs:
        for layer_id in sorted(split_signs):
            values = split_signs[layer_id]
            if values.shape[0] != n_rows:
                raise ValueError("split_signs leading dimension mismatch")
            flattened = values.reshape(n_rows, values.shape[1], -1)
            first = flattened[:, :1, :]
            if not torch.equal(flattened, first.expand_as(flattened)):
                raise ValueError("CLIMB requires spec-invariant split signs")
            width = flattened.shape[-1]
            if codec.width(layer_id) != width:
                raise ValueError("split_signs width does not match literal codec")
            lane_values = first[:, 0, :].to(target_device)
            positions = torch.nonzero(lane_values, as_tuple=False)
            counts = torch.bincount(positions[:, 0], minlength=n_rows)
            count_parts.append(counts)
            row_parts.append(positions[:, 0])
            literal_parts.append(
                codec.encode_tensor(
                    torch.full_like(positions[:, 0], layer_id),
                    positions[:, 1],
                    lane_values[positions[:, 0], positions[:, 1]].sign(),
                )
            )
    if not count_parts:
        return torch.zeros((n_rows, 0), dtype=torch.long, device=target_device)
    total_counts = torch.stack(count_parts).sum(dim=0)
    packed = torch.zeros(
        (n_rows, int(total_counts.max())), dtype=torch.long, device=target_device
    )
    layer_bases = torch.zeros(n_rows, dtype=torch.long, device=target_device)
    for rows, literals, counts in zip(row_parts, literal_parts, count_parts):
        starts = counts.cumsum(0) - counts
        local_slots = torch.arange(rows.numel(), device=target_device) - torch.repeat_interleave(
            starts, counts
        )
        packed[rows, layer_bases[rows] + local_slots] = literals
        layer_bases += counts
    return packed


class CoreLibrary:
    """Query-local certified phase cores with subsumption and resolution."""

    def __init__(self, max_cores: int) -> None:
        if max_cores < 1:
            raise ValueError("max_cores must be positive")
        self.max_cores = max_cores
        self.subsumed = 0
        self.resolved = 0
        self.evicted = 0
        self._cores: List[Core] = []

    @property
    def core_count(self) -> int:
        return len(self._cores)

    @property
    def cores(self) -> Tuple[Core, ...]:
        return tuple(self._cores)

    def _admit(self, literals: Core) -> int:
        """Insert unless subsumed; drop strict supersets. Returns removed count."""
        assert _core_has_no_complements(literals), (
            "CLIMB invariant violated: admitted core contains complementary literals "
            f"(core_size={len(literals)})"
        )
        before = len(self._cores)
        self._cores = [core for core in self._cores if not literals < core]
        self._cores.append(literals)
        return before + 1 - len(self._cores)

    def _is_subsumed(self, literals: Core) -> bool:
        return any(core <= literals for core in self._cores)

    def insert(self, packed: torch.Tensor) -> int:
        counters_before = (self.subsumed, self.resolved, self.evicted)
        inserted = 0
        for lane in range(packed.shape[0]):
            literals = frozenset(packed[lane, packed[lane] != 0].tolist())
            if self._is_subsumed(literals):
                self.subsumed += 1
                continue
            self.subsumed += self._admit(literals)
            inserted += 1

        while (resolvent := self._find_resolvent()) is not None:
            self._admit(resolvent)
            self.resolved += 1

        self._cores.sort(key=len)
        self.evicted += max(0, len(self._cores) - self.max_cores)
        self._cores = self._cores[: self.max_cores]
        counters_after = (self.subsumed, self.resolved, self.evicted)
        assert _counters_non_decreasing(counters_before, counters_after), (
            "CLIMB invariant violated: core-library counters decreased "
            f"(before={counters_before}, after={counters_after})"
        )
        return inserted

    def _find_resolvent(self) -> Optional[Core]:
        """First pair differing only by one opposite pivot sign, if any."""
        snapshot = list(self._cores)
        for left_index, left in enumerate(snapshot):
            for right in snapshot[left_index + 1 :]:
                difference = left ^ right
                if len(left) != len(right) or len(difference) != 2 or sum(difference) != 0:
                    continue
                resolvent = left & right
                if not self._is_subsumed(resolvent):
                    return resolvent
        return None

    def pack(self, device: torch.device) -> torch.Tensor:
        width = max((len(core) for core in self._cores), default=0)
        packed = torch.zeros((len(self._cores), width), dtype=torch.long, device=device)
        for row, core in enumerate(self._cores):
            values = sorted(core, key=lambda literal: (abs(literal), literal))
            packed[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        return packed


def apply_literal_assignments(
    batch: SubproblemBatch,
    assignments: torch.Tensor,
    active_variables: torch.Tensor,
    codec: LiteralCodec,
) -> SubproblemBatch:
    """Decode compact inferred phases back to a lossless subproblem batch."""
    signs = {key: value.clone() for key, value in (batch.split_signs or {}).items()}
    specs = 1
    if signs:
        specs = next(iter(signs.values())).shape[1]
    elif batch.incremental_alpha:
        specs = next(iter(batch.incremental_alpha.values())).shape[1]
    elif batch.incremental_eta:
        specs = next(iter(batch.incremental_eta.values())).shape[1]
    if active_variables.numel() == 0:
        batch.split_signs = signs or None
        return batch
    positive_literals = active_variables + 1
    layer_ids, neurons, _ = codec.decode_tensor(positive_literals)
    for layer_id, width in zip(codec.layer_ids, codec.widths):
        columns = torch.where(layer_ids == layer_id)[0]
        if columns.numel() == 0:
            continue
        if layer_id not in signs:
            signs[layer_id] = torch.zeros(
                batch.batch_size,
                specs,
                width,
                dtype=batch.lb.dtype,
                device=batch.lb.device,
            )
        layer_values = assignments.index_select(1, columns)
        lane_indices, local_columns = torch.nonzero(layer_values, as_tuple=True)
        if lane_indices.numel() == 0:
            continue
        layer_neurons = neurons.index_select(0, columns).index_select(0, local_columns)
        values = layer_values[lane_indices, local_columns].to(signs[layer_id])
        signs[layer_id][lane_indices, :, layer_neurons] = values.unsqueeze(1)
    batch.split_signs = signs or None
    return batch


@torch.no_grad()
def propagate(
    batches: Sequence[SubproblemBatch],
    library: CoreLibrary,
    codec: LiteralCodec,
    *,
    core_chunk: int,
) -> Tuple[SubproblemBatch, ...]:
    """Apply certified cores jointly to pending and active groups to fixpoint.

    ``core_chunk`` caps the cores per matmul chunk; it bounds memory only, the
    fixpoint is independent of it.
    """
    if core_chunk < 1:
        raise ValueError("core_chunk must be positive")
    if not batches:
        return ()
    group_sizes = [batch.batch_size for batch in batches]
    packed_batches = [
        pack_split_matrix(
            batch.split_signs,
            batch.batch_size,
            codec,
            device=batch.lb.device,
        )
        for batch in batches
    ]
    if library.core_count == 0:
        return tuple(batches)
    device = batches[0].lb.device
    packed_cores = library.pack(device)
    batch_values = torch.cat(
        [packed[packed != 0] for packed in packed_batches]
    )
    core_values = packed_cores[packed_cores != 0]
    all_values = torch.cat((batch_values, core_values))
    active_variables, inverse = torch.unique(
        all_values.abs() - 1, sorted=True, return_inverse=True
    )
    assignments = torch.zeros(
        (sum(group_sizes), active_variables.numel()), dtype=torch.int8, device=device
    )
    offset = 0
    inverse_offset = 0
    for packed, size in zip(packed_batches, group_sizes):
        lanes, slots = torch.nonzero(packed, as_tuple=True)
        count = lanes.numel()
        assignments[offset + lanes, inverse[inverse_offset : inverse_offset + count]] = packed[
            lanes, slots
        ].sign().to(torch.int8)
        inverse_offset += count
        offset += size
    initial_assignments = assignments.clone()
    core_dense = torch.zeros(
        (library.core_count, active_variables.numel()), dtype=torch.int8, device=device
    )
    core_rows, core_slots = torch.nonzero(packed_cores, as_tuple=True)
    core_dense[core_rows, inverse[inverse_offset:]] = packed_cores[
        core_rows, core_slots
    ].sign().to(torch.int8)
    active = torch.ones(assignments.shape[0], dtype=torch.bool, device=assignments.device)
    compute_dtype = get_default_dtype()
    core_abs = core_dense.abs()
    lengths = core_abs.sum(dim=1).to(compute_dtype)
    while bool(active.any().item()):
        active_rows = torch.where(active)[0]
        work = assignments[active]
        work_f = work.to(compute_dtype)
        work_abs_f = work.abs().to(compute_dtype)
        # One snapshot per round: every chunk reads the same `work`, so the
        # fixpoint is independent of chunk size (memory is O(N * chunk * W)).
        discharge_local = torch.zeros(work.shape[0], dtype=torch.bool, device=work.device)
        positive = torch.zeros_like(work, dtype=torch.bool)
        negative = torch.zeros_like(positive)
        for start in range(0, core_dense.shape[0], core_chunk):
            chunk = core_dense[start : start + core_chunk]
            chunk_lengths = lengths[start : start + core_chunk]
            product = work_f @ chunk.to(compute_dtype).T
            overlap = work_abs_f @ core_abs[start : start + core_chunk].to(compute_dtype).T
            discharge_local |= (product == chunk_lengths.unsqueeze(0)).any(dim=1)
            unit = (product == overlap) & (overlap == chunk_lengths.unsqueeze(0) - 1)
            lane_local, core_local = torch.nonzero(unit, as_tuple=True)
            if lane_local.numel() != 0:
                missing = (chunk.index_select(0, core_local) != 0) & (
                    work.index_select(0, lane_local) == 0
                )
                valid = missing.sum(dim=1) == 1
                lane_local = lane_local[valid]
                core_local = core_local[valid]
                columns = missing[valid].to(torch.int64).argmax(dim=1)
                proposed = -chunk[core_local, columns]
                positive[lane_local[proposed > 0], columns[proposed > 0]] = True
                negative[lane_local[proposed < 0], columns[proposed < 0]] = True
        if bool(discharge_local.any().item()):
            active[active_rows[discharge_local]] = False

        survivor_local = ~discharge_local
        if not bool(survivor_local.any().item()):
            continue
        survivor_rows = active_rows[survivor_local]
        positive = positive[survivor_local]
        negative = negative[survivor_local]
        conflict = (positive & negative).any(dim=1)
        if bool(conflict.any().item()):
            active[survivor_rows[conflict]] = False
        writable = ~conflict
        changed = False
        if bool(writable.any().item()):
            rows = survivor_rows[writable]
            proposals = positive[writable].to(torch.int8) - negative[writable].to(torch.int8)
            write = (assignments[rows] == 0) & (proposals != 0)
            if bool(write.any().item()):
                assignments[rows] = torch.where(write, proposals, assignments[rows])
                changed = True
        if not changed and not bool(discharge_local.any().item()) and not bool(conflict.any().item()):
            break

    assert _only_fills_unassigned(initial_assignments, assignments), (
        "CLIMB invariant violated: propagation changed an existing split literal "
        f"(rows={assignments.shape[0]}, variables={assignments.shape[1]})"
    )
    result_batches: List[SubproblemBatch] = []
    offset = 0
    for batch, size in zip(batches, group_sizes):
        group_active = active[offset : offset + size]
        keep = torch.where(group_active)[0]
        restricted = batch.select(keep)
        restricted_assignments = assignments[offset : offset + size].index_select(0, keep)
        propagated = apply_literal_assignments(
            restricted, restricted_assignments, active_variables, codec
        )
        assert propagated.batch_size <= size, (
            "CLIMB invariant violated: propagation increased a group batch size "
            f"(input_rows={size}, output_rows={propagated.batch_size})"
        )
        result_batches.append(propagated)
        offset += size
    return tuple(result_batches)


def is_certified(slack: torch.Tensor, out_kind: str) -> torch.Tensor:
    """Per-lane certification mask over ``[N, M]`` slack rows.

    Byte-consistent with the ``DualSolver.solve_spec_batch`` lane statuses: an
    ``UNSAFE_LINEAR`` lane certifies when any finite row is strictly positive;
    every other kind certifies only when all rows are finite and non-negative.
    """
    if out_kind == OutKind.UNSAFE_LINEAR:
        return (torch.isfinite(slack) & (slack > 0)).any(dim=1)
    return torch.isfinite(slack).all(dim=1) & (slack >= 0).all(dim=1)


@torch.no_grad()
def certificate_replay(
    *,
    net: Net,
    batch: SubproblemBatch,
    reference_bounds: Dict[int, Bounds],
    c_rows: torch.Tensor,
    thresholds: torch.Tensor,
    m_specs: int,
    out_kind: str,
    literals: Optional[torch.Tensor] = None,
    codec: Optional[LiteralCodec] = None,
    with_costs: bool = True,
) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Replay a frozen alpha/eta certificate directly through ``DualSolver``.

    Returns ``(slack, costs)`` — for ``UNSAFE_LINEAR`` both are gathered on the
    pinned strict row (``slack [N, 1]``, ``costs [N, 1, W]``); otherwise
    ``slack`` is ``[N, M]`` and ``costs`` is ``[N, M, W]``. ``costs`` is
    ``None`` when ``with_costs=False``. Returns ``None`` when the batch lacks
    replayable state or the solver emits no usable per-layer nu.
    """
    if with_costs and (literals is None or codec is None):
        raise ValueError("with_costs=True requires packed literals")
    if (
        batch.incremental_alpha is None
        or batch.incremental_eta is None
        or batch.split_signs is None
        or m_specs < 1
    ):
        return None
    from act.back_end.solver.solver_dual import DualSolver

    eta = {
        layer_id: value.detach().clone().clamp(min=0)
        for layer_id, value in batch.incremental_eta.items()
    }
    result = DualSolver().compute_certified_bound(
        net,
        reference_bounds,
        c_rows,
        M=m_specs,
        alpha=batch.incremental_alpha,
        eta=eta,
        split_signs=batch.split_signs,
        optimize=False,
        return_nu_per_layer=True,
        local_phase_clamp=True,
    )
    n_lanes = batch.batch_size
    slack = result.margins.reshape(n_lanes, m_specs)
    slack = slack - thresholds.reshape(n_lanes, m_specs).to(slack)
    pinned_rows: Optional[torch.Tensor] = None
    if out_kind == OutKind.UNSAFE_LINEAR:
        passing = torch.isfinite(slack) & (slack > 0)
        pinned_rows = passing.to(torch.int64).argmax(dim=1)
    assert result.nu_per_layer is not None, (
        "certificate_replay: solver returned no nu_per_layer despite return_nu_per_layer=True"
    )
    nu: Dict[int, torch.Tensor] = {}
    for layer_id, value in result.nu_per_layer.items():
        if value.shape[0] == n_lanes * m_specs:
            nu[layer_id] = value.reshape(n_lanes, m_specs, -1)
        elif value.shape[0] == n_lanes:
            nu[layer_id] = value.reshape(n_lanes, -1, value.shape[-1])
        else:
            return None
    costs: Optional[torch.Tensor] = None
    if with_costs:
        assert literals is not None
        assert codec is not None
        costs = _literal_costs(
            literals, codec, slack, eta, nu, reference_bounds, pinned_rows
        )
    if pinned_rows is not None:
        slack = slack.gather(1, pinned_rows[:, None])
    return slack, costs


@torch.no_grad()
def _literal_costs(
    literals: torch.Tensor,
    codec: LiteralCodec,
    slack: torch.Tensor,
    eta: Dict[int, torch.Tensor],
    nu: Dict[int, torch.Tensor],
    bounds: Dict[int, Bounds],
    pinned_rows: Optional[torch.Tensor],
) -> torch.Tensor:
    """Full ReLU Eq. 12 costs ``(eta + relu(-nu)) * m`` in ``slack.dtype``, ``[N, M, W]``."""
    n_lanes, width = literals.shape
    m_specs = slack.shape[1]
    costs = torch.zeros(
        (n_lanes, m_specs, width),
        dtype=slack.dtype,
        device=literals.device,
    )
    mask = literals != 0
    if bool(mask.any()):
        layer_ids, neurons, signs = codec.decode_tensor(literals.masked_fill(~mask, 1))
        for layer_id in codec.layer_ids:
            lane, slot = torch.where(mask & (layer_ids == layer_id))
            if lane.numel() == 0:
                continue
            if layer_id not in eta or layer_id not in nu or layer_id not in bounds:
                costs[lane, :, slot] = torch.inf
                continue
            neuron = neurons[lane, slot]
            lane_eta = eta[layer_id][lane, :, neuron].clamp(min=0)
            lane_nu = nu[layer_id][lane, :, neuron]
            layer_bounds = bounds[layer_id]
            lower = layer_bounds.lb.flatten(start_dim=1)[lane, neuron]
            upper = layer_bounds.ub.flatten(start_dim=1)[lane, neuron]
            magnitude = torch.where(
                signs[lane, slot] > 0,
                (-lower).clamp(min=0),
                upper.clamp(min=0),
            )
            costs[lane, :, slot] = (
                lane_eta + (-lane_nu).clamp(min=0)
            ) * magnitude.unsqueeze(1)
    if pinned_rows is not None:
        rows = pinned_rows.to(device=costs.device, dtype=torch.long)
        costs = costs.gather(1, rows[:, None, None].expand(-1, 1, width))
    return costs


@torch.no_grad()
def coarsen(
    literals: torch.Tensor,
    costs: torch.Tensor,
    slack: torch.Tensor,
    *,
    theta: float,
    delta_abs: float,
    delta_rel: float,
    recheck_k: int,
    recheck: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Alg. 2 vector-budget deletion plus at most ``recheck_k`` one-step rechecks.

    Per row ``m`` the budget is ``theta * slack_m - delta_m`` with
    ``delta_m = delta_abs + delta_rel * |slack_m|``; rows with
    ``slack_m <= delta_m`` never fund a deletion. The budget arithmetic runs
    in the dtype of ``costs``/``slack``. Literals are ordered by
    ``(max_m cost_m / budget_m, column, slot)`` and the longest feasible prefix
    is deleted. A recheck retries the next literal in that same order for the
    first ``recheck_k`` lanes; ``recheck`` returns the ``BoolTensor[K]``
    certification mask that confirms each candidate deletion.
    """
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must be in [0, 1]")
    if not (delta_abs >= 0.0 and delta_rel >= 0.0):
        raise ValueError("delta_abs and delta_rel must be non-negative")
    if recheck_k < 0:
        raise ValueError("recheck_k must be non-negative")
    n_lanes, width = literals.shape
    if costs.shape[0] != n_lanes or costs.shape[2] != width:
        raise ValueError("cost tensor shape must be [N, M, W]")
    if slack.shape != costs.shape[:2]:
        raise ValueError("slack tensor shape must be [N, M]")
    retained = literals != 0
    next_in_cost_order: Dict[int, int] = {}
    for lane in range(n_lanes):
        valid_slots = torch.where(literals[lane] != 0)[0]
        if valid_slots.numel() == 0:
            continue
        lane_costs = costs[lane].index_select(1, valid_slots)
        lane_slack = slack[lane]
        if not bool(torch.isfinite(lane_costs).all().item()) or not bool(
            torch.isfinite(lane_slack).all().item()
        ):
            continue
        delta = delta_abs + delta_rel * lane_slack.abs()
        if bool((lane_slack <= delta).any().item()):
            continue
        budget = theta * lane_slack - delta
        denominator = budget.clamp(min=torch.finfo(budget.dtype).tiny)
        ratios = (lane_costs / denominator[:, None]).amax(dim=0)
        ordered = sorted(
            range(valid_slots.numel()),
            key=lambda pos: (
                float(ratios[pos].item()),
                int(literals[lane, valid_slots[pos]].abs().item()) - 1,
                int(valid_slots[pos].item()),
            ),
        )
        running = torch.zeros_like(lane_slack)
        for position in ordered:
            candidate = running + lane_costs[:, position]
            slot = int(valid_slots[position].item())
            if not bool((candidate <= budget).all().item()):
                next_in_cost_order[lane] = slot
                break
            retained[lane, slot] = False
            running = candidate

    candidates: List[int] = []
    next_slots: List[int] = []
    if recheck_k > 0:
        for lane, slot in next_in_cost_order.items():
            candidates.append(lane)
            next_slots.append(slot)
            if len(candidates) == recheck_k:
                break
    if candidates:
        rows = torch.tensor(candidates, dtype=torch.long, device=literals.device)
        candidate_mask = retained.index_select(0, rows).clone()
        for local, slot in enumerate(next_slots):
            candidate_mask[local, slot] = False
        candidate = literals.index_select(0, rows).masked_fill(~candidate_mask, 0)
        passing = recheck(candidate, rows)
        for local, passed in enumerate(passing.tolist()):
            if passed:
                retained[candidates[local], next_slots[local]] = False

    return literals.masked_fill(~retained, 0)


def validate_climb_config(config: BaBConfig) -> None:
    """Fail before solving when CLIMB's soundness prerequisites are absent."""
    if not config.climb_enabled:
        return
    failures: List[str] = []
    if config.solver_tier != CLIMB_SOLVER_TIER:
        failures.append(f"solver_tier must be {CLIMB_SOLVER_TIER!r}")
    if config.bounding not in TOP_K_BOUNDINGS:
        failures.append(f"bounding must be one of {TOP_K_BOUNDINGS}")
    if config.branching_method not in NEURON_BRANCHING_METHODS:
        failures.append(f"branching_method must be one of {NEURON_BRANCHING_METHODS}")
    if config.root_bounds_reuse != "none":
        failures.append("root_bounds_reuse must be 'none'")
    if config.intermediate_refine != NO_REFINEMENT_MODE:
        failures.append(f"intermediate_refine must be {NO_REFINEMENT_MODE!r}")
    if config.per_subproblem_refine != NO_REFINEMENT_MODE:
        failures.append(f"per_subproblem_refine must be {NO_REFINEMENT_MODE!r}")
    if failures:
        raise ConfigError("CLIMB configuration error: " + "; ".join(failures))


@dataclass(frozen=True)
class _ReplayContext:
    batch: SubproblemBatch
    bounds: Dict[int, Bounds]
    c_lanes: torch.Tensor
    thresholds: torch.Tensor
    m_specs: int


class ClimbSession:
    """Own the query-local CLIMB library, metrics, and BaB-loop hooks."""

    def __init__(self, config: BaBConfig, net: Net, out_kind: str) -> None:
        self._net = net
        self._out_kind = out_kind
        self._theta = config.climb_theta
        self._delta_abs = config.climb_delta_abs
        self._delta_rel = config.climb_delta_rel
        self._recheck_k = config.climb_recheck_k
        self._propagate_core_chunk = config.climb_propagate_core_chunk
        self.codec = LiteralCodec.from_net(net)
        self.library = CoreLibrary(config.climb_max_cores)
        self.metrics = ClimbMetrics(enabled=True, library=self.library)

    @classmethod
    def from_config(
        cls, config: BaBConfig, net: Net, out_kind: str
    ) -> Optional[ClimbSession]:
        validate_climb_config(config)
        return cls(config, net, out_kind) if config.climb_enabled else None

    def before_pop(self, pool: TopKBounding) -> None:
        if self.library.core_count > 0:
            # H1: eager discharge of the whole pending pool before any lane is
            # bounded. view_all/replace_all never score, cool, or probe.
            pending_before = pool.view_all()
            propagate_started = time.perf_counter()
            (pending_after,) = propagate(
                [pending_before],
                self.library,
                self.codec,
                core_chunk=self._propagate_core_chunk,
            )
            self.metrics.propagate_time_s += time.perf_counter() - propagate_started
            self.metrics.prebound_discharged += (
                pending_before.batch_size - pending_after.batch_size
            )
            pool.replace_all(pending_after)
            if pool.empty:
                return
        self.metrics.observe_frontier(len(pool))

    def after_pop(self, frontier_rows: int) -> None:
        self.metrics.observe_frontier(frontier_rows)

    def after_main_bound(self, k_actual: int) -> None:
        self.metrics.main_bound_calls += 1
        self.metrics.main_bound_row_passes += k_actual

    def learn(
        self,
        batch: SubproblemBatch,
        statuses: Sequence[str],
        dual_solve_result: DualBatchResult,
        k_actual: int,
    ) -> None:
        # H2: replay -> coarsen -> insert. Certified lanes only; replay reads
        # the immutable forward snapshot, the theta budget coarsens the
        # validated vector, and the survivor is inserted as a core.
        unsat_idx = torch.tensor(
            [i for i, status in enumerate(statuses) if status == SolveStatus.UNSAT],
            device=batch.lb.device,
            dtype=torch.long,
        )
        if (
            int(unsat_idx.numel()) == 0
            or dual_solve_result.c_rows is None
            or dual_solve_result.thresholds is None
            or dual_solve_result.reference_bounds is None
            or dual_solve_result.m_specs <= 0
        ):
            return
        replay_batch = batch.select(unsat_idx)
        replay_m_specs = dual_solve_result.m_specs
        # The replay reference is the exact unhardened forward snapshot the
        # main solve consumed, sliced by certified lane; it is never recomputed
        # so Eq. 12 magnitudes and the local clamps read the same immutable bounds.
        replay_bounds = slice_bounds_dict(
            dual_solve_result.reference_bounds, unsat_idx
        )
        replay_c_lanes = dual_solve_result.c_rows.reshape(
            k_actual, replay_m_specs, -1
        ).index_select(0, unsat_idx.to(dual_solve_result.c_rows.device))
        replay_c = replay_c_lanes.reshape(-1, replay_c_lanes.shape[-1])
        replay_thresholds = dual_solve_result.thresholds.index_select(
            0, unsat_idx.to(dual_solve_result.thresholds.device)
        )
        packed = pack_split_matrix(
            replay_batch.split_signs,
            replay_batch.batch_size,
            self.codec,
            device=replay_batch.lb.device,
        )
        replay = certificate_replay(
            net=self._net,
            batch=replay_batch,
            reference_bounds=replay_bounds,
            c_rows=replay_c,
            thresholds=replay_thresholds,
            m_specs=replay_m_specs,
            out_kind=self._out_kind,
            literals=packed,
            codec=self.codec,
        )
        if replay is not None:
            slack, costs = replay
            valid = is_certified(slack, self._out_kind)
        else:
            slack = costs = valid = None
        if costs is None or valid is None or not bool(valid.any().item()):
            return
        budget_slack = slack
        assert budget_slack is not None
        costs[~valid] = torch.inf
        context = _ReplayContext(
            replay_batch, replay_bounds, replay_c_lanes, replay_thresholds, replay_m_specs
        )
        coarsen_started = time.perf_counter()
        retained = coarsen(
            packed,
            costs,
            budget_slack,
            theta=self._theta,
            delta_abs=self._delta_abs,
            delta_rel=self._delta_rel,
            recheck_k=self._recheck_k,
            recheck=lambda candidate, rows: self._instrumented_recheck(
                context, candidate, rows
            ),
        )
        self.metrics.coarsen_time_s += time.perf_counter() - coarsen_started
        original_literals = int((packed != 0).sum())
        retained_literals = int((retained != 0).sum())
        assert _retained_literals_are_subset(packed, retained), (
            "CLIMB invariant violated: coarsening added or changed a literal "
            f"(rows={packed.shape[0]}, width={packed.shape[1]}, "
            f"original_literals={original_literals}, retained_literals={retained_literals})"
        )
        self.metrics.core_literals_original += original_literals
        self.metrics.core_literals_retained += retained_literals
        assert (
            self.metrics.core_literals_retained
            <= self.metrics.core_literals_original
        ), (
            "CLIMB invariant violated: retained literal accounting exceeds original "
            f"(original_total={self.metrics.core_literals_original}, "
            f"retained_total={self.metrics.core_literals_retained})"
        )
        valid_rows = torch.where(valid)[0]
        valid_cores = retained.index_select(0, valid_rows)
        self.metrics.cores_inserted += self.library.insert(valid_cores)

    def _recheck(
        self,
        context: _ReplayContext,
        candidate: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        candidate_batch = context.batch.select(rows)
        # The solver indexes split_signs[lid] for every incremental_eta layer, so a
        # layer whose literals were all coarsened away stays as an all-zero block.
        if candidate_batch.split_signs is not None:
            candidate_batch.split_signs = {
                lid: torch.zeros_like(signs)
                for lid, signs in candidate_batch.split_signs.items()
            }
        candidate_values = candidate[candidate != 0]
        active_variables = torch.unique(candidate_values.abs() - 1, sorted=True)
        dense = torch.zeros(
            (candidate.shape[0], active_variables.numel()),
            dtype=torch.int8,
            device=candidate.device,
        )
        lanes, slots = torch.nonzero(candidate, as_tuple=True)
        columns = torch.searchsorted(active_variables, candidate[lanes, slots].abs() - 1)
        dense[lanes, columns] = candidate[lanes, slots].sign().to(torch.int8)
        apply_literal_assignments(candidate_batch, dense, active_variables, self.codec)
        eta_layers = set(candidate_batch.incremental_eta or {})
        sign_layers = set(candidate_batch.split_signs or {})
        missing_layers = sorted(eta_layers - sign_layers)
        assert not missing_layers, (
            "CLIMB invariant violated: recheck split_signs missing incremental_eta layer "
            f"(KeyError: {missing_layers[0]}, eta_layers={len(eta_layers)}, "
            f"split_sign_layers={len(sign_layers)})"
        )
        row_bounds = slice_bounds_dict(context.bounds, rows)
        row_c_lanes = context.c_lanes.index_select(
            0, rows.to(context.c_lanes.device)
        )
        row_c = row_c_lanes.reshape(-1, row_c_lanes.shape[-1])
        row_thresholds = context.thresholds.index_select(
            0, rows.to(context.thresholds.device)
        )
        replayed = certificate_replay(
            net=self._net,
            batch=candidate_batch,
            reference_bounds=row_bounds,
            c_rows=row_c,
            thresholds=row_thresholds,
            m_specs=context.m_specs,
            out_kind=self._out_kind,
            with_costs=False,
        )
        if replayed is None:
            return torch.zeros(
                int(rows.numel()), dtype=torch.bool, device=rows.device
            )
        replayed_slack, _ = replayed
        return is_certified(replayed_slack, self._out_kind)

    def _instrumented_recheck(
        self,
        context: _ReplayContext,
        candidate: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        recheck_started = time.perf_counter()
        passing = self._recheck(context, candidate, rows)
        self.metrics.recheck_time_s += time.perf_counter() - recheck_started
        self.metrics.recheck_calls += 1
        self.metrics.recheck_row_passes += int(rows.numel())
        return passing

    def before_split(
        self, pool: TopKBounding, unresolved: SubproblemBatch
    ) -> SubproblemBatch:
        if self.library.core_count == 0:
            return unresolved
        # H3: one joint fixpoint over (pending, active); pending removals were
        # never bounded (pre-bound), active removals are parents spared a split
        # (pre-split).
        self.metrics.observe_frontier(len(pool) + unresolved.batch_size)
        propagation_groups: List[SubproblemBatch] = []
        has_pool = len(pool) > 0
        if has_pool:
            propagation_groups.append(pool.view_all())
        propagation_groups.append(unresolved)
        presplit_before = unresolved.batch_size
        propagate_started = time.perf_counter()
        propagation = propagate(
            propagation_groups,
            self.library,
            self.codec,
            core_chunk=self._propagate_core_chunk,
        )
        self.metrics.propagate_time_s += time.perf_counter() - propagate_started
        if has_pool:
            pool.replace_all(propagation[0])
            self.metrics.prebound_discharged += (
                propagation_groups[0].batch_size - propagation[0].batch_size
            )
        unresolved = propagation[-1]
        self.metrics.presplit_discharged += presplit_before - unresolved.batch_size
        self.metrics.observe_frontier(len(pool) + unresolved.batch_size)
        return unresolved

    def reject_input_axis_split(self) -> None:
        raise ValueError(
            "CLIMB requires ReLU phase literals only: the "
            "brancher fell back to an input-axis split, which "
            "would break the shared-box premise of core reuse"
        )

    def after_children(self, generated: int, frontier_rows: int) -> None:
        self.metrics.generated_children += generated
        self.metrics.observe_frontier(frontier_rows)

    def metadata(self) -> Dict[str, int | float]:
        return self.metrics.metadata()
