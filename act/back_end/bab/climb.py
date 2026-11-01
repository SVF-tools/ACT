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
from typing import TYPE_CHECKING, Any, Callable, Collection, Dict, List, Optional, Sequence, Tuple

import torch

from act.back_end.bab.bound_refresh import BoundSupport
from act.back_end.bab.branching.bounding import TopKBounding
from act.back_end.bab.branching.branching import BranchingScores
from act.back_end.bab.node import (
    SubproblemBatch,
    _encode_literal_tensor,
    _relu_literal_layout,
    _state_spec_row_count,
    slice_bounds_dict,
)
from act.back_end.core import Bounds, Net
from act.back_end.solver.solver_base import SolveStatus
from act.back_end.solver.solver_dual import (
    DualBatchResult, MixedAlpha, certification_tolerance, unproven_spec_rows,
)
from act.config.config import (
    BaBConfig,
    CLIMB_SOLVER_TIER,
    ConfigError,
    DualConfig,
    NEURON_BRANCHING_METHODS,
    NO_REFINEMENT_MODE,
    TOP_K_BOUNDINGS,
)
from act.front_end.specs import OutKind, OutputSpec
from act.util.device_manager import get_default_device, get_default_dtype

if TYPE_CHECKING:
    from act.back_end.solver.terminal_lp import TerminalResult


# ---------------------------------------------------------------------------
# CLIMB certificate reuse
# ---------------------------------------------------------------------------

# Partners tried per row and pivot; only exact duplicate rows ever reach this.
_MERGE_PROBE_ROWS = 64
_MERGE_GROUP_FANOUT = 64
_SPLIT_REFRESH_MODES = ("split_refresh", "split_refresh_linear")


@dataclass(frozen=True)
class LiteralCodec:
    """Immutable DIMACS-style codec for the network's ReLU phase literals."""

    layer_ids: Tuple[int, ...]
    widths: Tuple[int, ...]
    offsets: Tuple[int, ...]

    @classmethod
    def from_net(cls, net: Net) -> LiteralCodec:
        return cls(*_relu_literal_layout(net))

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
        return _encode_literal_tensor(self.layer_ids, self.offsets, layer_ids, neurons, signs)

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


def _canonical_rows(rows: torch.Tensor) -> torch.Tensor:
    """Sort exact signed literals by variable, with unique values and zero padding."""
    keys = rows.abs().masked_fill(rows == 0, torch.iinfo(torch.long).max)
    rows = rows.gather(1, keys.argsort(dim=1, stable=True))
    if rows.shape[1] > 1:
        repeated = (rows[:, 1:].abs() == rows[:, :-1].abs()) & (rows[:, 1:] != 0)
        if bool((repeated & (rows[:, 1:] != rows[:, :-1])).any()):
            raise ValueError("core contains complementary literals")
        rows[:, 1:] = rows[:, 1:].masked_fill(repeated, 0)
        rows = rows.gather(1, rows.abs().masked_fill(rows == 0, torch.iinfo(torch.long).max).argsort(dim=1, stable=True))
    width = int((rows != 0).sum(1).max()) if rows.shape[0] else 0
    return rows[:, :width]


def _subset_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Exact pairwise subset test; callers bound both row dimensions."""
    if right.shape[1] == 0:
        return (left == 0).all(1)[:, None].expand(-1, right.shape[0])
    keys = right.abs().masked_fill(right == 0, torch.iinfo(torch.long).max)
    indices = torch.searchsorted(
        keys[None].expand(left.shape[0], -1, -1).contiguous(),
        left.abs()[:, None].expand(-1, right.shape[0], -1).contiguous(),
    ).clamp(max=right.shape[1] - 1)
    matched = right[None].expand(left.shape[0], -1, -1).gather(2, indices)
    return ((left[:, None] == 0) | (left[:, None] == matched)).all(2)


def _first_unique_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Distinct rows with the position of each row's first occurrence."""
    if rows.shape[1] == 0:  # Only the empty core, however many pairs derived it.
        return rows[:1], torch.zeros(1, dtype=torch.long, device=rows.device)
    unique, inverse = torch.unique(rows, dim=0, return_inverse=True)
    first = torch.full((unique.shape[0],), rows.shape[0], dtype=torch.long, device=rows.device)
    first.scatter_reduce_(0, inverse, torch.arange(rows.shape[0], device=rows.device), "amin")
    return unique, first


def _canonical_ranked(
    rows: torch.Tensor, ranks: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Carry decision chronology as metadata beside canonical literal storage."""
    canonical = _canonical_rows(rows)
    if ranks is None:
        return canonical, None
    ranked = torch.zeros_like(canonical)
    if canonical.shape[1]:
        keys = canonical.abs().masked_fill(canonical == 0, torch.iinfo(torch.long).max)
        positions = torch.searchsorted(keys.contiguous(), rows.abs().contiguous()).clamp(max=canonical.shape[1] - 1)
        ranked.scatter_reduce_(1, positions, ranks.masked_fill(rows == 0, 0), reduce="amax")
    return canonical, ranked


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


# Frontier counters a failed wave must not keep.
_FRONTIER_COUNTERS = (
    "merged_pairs", "prebound_discharged", "presplit_discharged",
    "unit_writes", "merge_calls", "merge_rows", "merge_literals",
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
    coarsen_replay_fallbacks: int = 0
    retired_unknown_lanes: int = 0
    propagate_time_s: float = 0.0
    cores_inserted: int = 0
    merged_pairs: int = 0
    complement_calls: int = 0
    complement_row_passes: int = 0
    complement_time_s: float = 0.0
    steered_candidates: int = 0
    core_literals_original: int = 0
    core_literals_retained: int = 0
    terminal_literals_original: int = 0
    terminal_literals_retained: int = 0
    # Executed work: never rolled back, since the time was really spent.
    core_sources: int = 0
    cost_deleted_literals: int = 0
    optional_rechecks: int = 0
    mandatory_admissions: int = 0
    terminal_admissions: int = 0
    forward_core_rows: int = 0
    propagation_rows_scanned: int = 0
    propagation_rows_skipped: int = 0
    # Committed frontier changes: restored with the frontier on OOM rollback.
    unit_writes: int = 0
    merge_calls: int = 0
    merge_rows: int = 0
    merge_literals: int = 0
    merge_proved_rows: int = 0
    merge_open_rows: int = 0
    merge_stopped: bool = False
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
            "climb_coarsen_replay_fallbacks": self.coarsen_replay_fallbacks,
            "climb_retired_unknown_lanes": self.retired_unknown_lanes,
            "climb_propagate_time_s": self.propagate_time_s,
            "climb_cores_inserted": self.cores_inserted,
            "climb_merged_pairs": self.merged_pairs,
            "climb_complement_calls": self.complement_calls,
            "climb_complement_row_passes": self.complement_row_passes,
            "climb_complement_time_s": self.complement_time_s,
            "climb_steered_candidates": self.steered_candidates,
            "climb_cores_subsumed": self.library.subsumed if self.library is not None else 0,
            "climb_cores_resolved": self.library.resolved if self.library is not None else 0,
            "climb_cores_covering_resolved": self.library.covering_resolved if self.library is not None else 0,
            "climb_cores_evicted": self.library.evicted if self.library is not None else 0,
            "climb_core_literals_original": self.core_literals_original,
            "climb_core_literals_retained": self.core_literals_retained,
            "climb_terminal_literals_original": self.terminal_literals_original,
            "climb_terminal_literals_retained": self.terminal_literals_retained,
            "climb_core_sources": self.core_sources,
            "climb_cost_deleted_literals": self.cost_deleted_literals,
            "climb_optional_rechecks": self.optional_rechecks,
            "climb_mandatory_admissions": self.mandatory_admissions,
            "climb_terminal_admissions": self.terminal_admissions,
            "climb_forward_core_rows": self.forward_core_rows,
            "climb_propagation_rows_scanned": self.propagation_rows_scanned,
            "climb_propagation_rows_skipped": self.propagation_rows_skipped,
            "climb_unit_writes": self.unit_writes,
            "climb_merge_calls": self.merge_calls,
            "climb_merge_rows": self.merge_rows,
            "climb_merge_proved_rows": self.merge_proved_rows,
            "climb_merge_open_rows": self.merge_open_rows,
            "climb_merge_stopped": self.merge_stopped,
            "climb_merge_literals": self.merge_literals,
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
    if n_rows == 0:
        return torch.empty((0, 0), dtype=torch.long, device=target_device)
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
    """One packed tensor of exact certified cores; comparisons use bounded tiles."""

    def __init__(self, max_cores: int, *, track_complements: bool = False) -> None:
        if max_cores < 1:
            raise ValueError("max_cores must be positive")
        self.max_cores = max_cores
        self.subsumed = 0
        self.resolved = 0
        self.evicted = 0
        self.covering_resolved = 0
        # Bumped whenever the stored antichain changes, so a row is only rescanned
        # when a core could differ from the set it reached a fixed point against.
        self.revision = 0
        self._cores = torch.empty((0, 0), dtype=torch.long)
        self._ranks = torch.empty((0, 0), dtype=torch.long) if track_complements else None
        self._attempted = torch.empty(0, dtype=torch.bool)
        self.chunk = 128

    @property
    def core_count(self) -> int:
        return self._cores.shape[0]

    def _admit_new(
        self, rows: torch.Tensor, ranks: Optional[torch.Tensor], *, attempted: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store rows no stored core subsumes, dropping supersets.

        Returns the mask of stored rows kept and an all-true mask of the rows added.
        """
        width = max(self._cores.shape[1], rows.shape[1])
        stored = torch.nn.functional.pad(self._cores, (0, width - self._cores.shape[1]))
        rows = torch.nn.functional.pad(rows, (0, width - rows.shape[1]))
        keep_new = torch.ones(rows.shape[0], dtype=torch.bool, device=rows.device)
        keep_old = torch.ones(stored.shape[0], dtype=torch.bool, device=rows.device)
        lengths = rows.ne(0).sum(1)
        for start in range(0, rows.shape[0], self.chunk):
            tile = rows[start:start + self.chunk]
            for other in range(0, stored.shape[0], self.chunk):
                keep_old[other:other + self.chunk] &= ~_subset_rows(tile, stored[other:other + self.chunk]).any(0)
            for other in range(0, rows.shape[0], self.chunk):
                # Rows are distinct, so a subset among them is strictly shorter.
                keep_new[start:start + self.chunk] &= ~(
                    _subset_rows(rows[other:other + self.chunk], tile)
                    & (lengths[other:other + self.chunk, None] < lengths[None, start:start + self.chunk])
                ).any(0)
        self._cores = torch.cat((stored[keep_old], rows[keep_new]))
        if self._ranks is not None:
            assert ranks is not None
            self._ranks = torch.cat((
                torch.nn.functional.pad(self._ranks, (0, width - self._ranks.shape[1]))[keep_old],
                torch.nn.functional.pad(ranks, (0, width - ranks.shape[1]))[keep_new],
            ))
            self._attempted = torch.cat((
                self._attempted[keep_old],
                torch.full((int(keep_new.sum()),), attempted, dtype=torch.bool, device=rows.device),
            ))
        return keep_old, torch.ones(int(keep_new.sum()), dtype=torch.bool, device=rows.device)

    def insert(self, packed: torch.Tensor, *, suppress_complements: bool = False) -> int:
        """Admit a whole batch, preserving chronological insertion counters."""
        if packed.shape[0] == 0:
            return 0
        counters_before = (self.subsumed, self.resolved, self.evicted)
        cores_before = self._cores
        ranks = (torch.arange(1, packed.shape[1] + 1, device=packed.device)[None].expand_as(packed)
                 if self._ranks is not None else None)
        packed, ranks = _canonical_ranked(packed, ranks)
        old_count = self.core_count
        width = max(self._cores.shape[1], packed.shape[1])
        rows = torch.cat((
            torch.nn.functional.pad(self._cores.to(packed.device), (0, width - self._cores.shape[1])),
            torch.nn.functional.pad(packed, (0, width - packed.shape[1])),
        ))
        count = rows.shape[0]
        ids = torch.arange(count, device=packed.device)
        lengths = rows.ne(0).sum(1)
        keep = torch.ones(count, dtype=torch.bool, device=packed.device)
        admitted = torch.ones_like(keep)
        chunk = min(self.chunk, max(1, int((1048576 / max(1, width)) ** .5)))
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            for other in range(0, count, chunk):
                end = min(other + chunk, count)
                if stop <= old_count and end <= old_count:
                    continue  # The existing library is already an antichain.
                subset = _subset_rows(rows[other:end], rows[start:stop])
                earlier = ids[other:end, None] < ids[None, start:stop]
                shorter = lengths[other:end, None] < lengths[None, start:stop]
                admitted[start:stop] &= ~(subset & earlier).any(0)
                keep[start:stop] &= ~(subset & (shorter | earlier)).any(0)
        inserted = int(admitted[old_count:].sum())
        # Every skipped incoming row or removed stored row increments the same
        # subsumption counter as sequential admission, including supersets that
        # were admitted earlier in this batch and removed by a later row.
        self.subsumed += count - int(keep.sum())
        self._cores = rows[keep]
        if self._ranks is not None:
            assert ranks is not None
            self._ranks = torch.cat((
                torch.nn.functional.pad(self._ranks.to(packed.device), (0, width - self._ranks.shape[1])),
                torch.nn.functional.pad(ranks, (0, width - ranks.shape[1])),
            ))[keep]
            self._attempted = torch.cat((
                self._attempted.to(packed.device),
                torch.full((packed.shape[0],), suppress_complements, dtype=torch.bool, device=packed.device),
            ))[keep]

        # Only pairs touching a new core can resolve: the rest were already
        # resolved, or could not be, against each other.
        pending = torch.zeros(self.core_count, dtype=torch.bool, device=packed.device)
        pending[int(keep[:old_count].sum()):] = True
        while (resolvents := self._find_resolvents(pending)) is not None:
            rows, row_ranks, owners, covering = resolvents
            revisit = torch.zeros_like(pending)
            revisit[owners] = True
            kept, fresh = self._admit_new(rows, row_ranks, attempted=suppress_complements)
            pending = torch.cat((revisit[kept], fresh))
            self.resolved += fresh.numel()
            self.covering_resolved += covering

        order = (self._cores != 0).sum(1).argsort(stable=True)
        self.evicted += max(0, self.core_count - self.max_cores)
        self._cores = _canonical_rows(self._cores[order[:self.max_cores]])
        if self._ranks is not None:
            self._ranks = self._ranks[order[:self.max_cores], :self._cores.shape[1]]
            self._attempted = self._attempted[order[:self.max_cores]]
        counters_after = (self.subsumed, self.resolved, self.evicted)
        assert _counters_non_decreasing(counters_before, counters_after), (
            "CLIMB invariant violated: core-library counters decreased "
            f"(before={counters_before}, after={counters_after})"
        )
        if not torch.equal(cores_before.to(self._cores.device), self._cores):
            self.revision += 1
        return inserted

    def _find_resolvents(
        self, pending: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, int]]:
        """One covering resolvent per pending core against the library, if any.

        ``C1 | {l}`` and ``C2 | {-l}`` with ``C1 <= C2`` make every assignment of
        ``C2`` unsafe, so ``C2`` is a core, subsuming the longer side. Equal sides
        are the plain case. Over signed variables, ``(overlap + dot) / 2`` counts
        shared literals and ``(overlap - dot) / 2`` opposite ones, so one short
        literal is missing exactly when both counts single it out.

        Taking all resolvents of a core at once explodes: most are subsumed by the
        next round's. Each core therefore yields its first unsubsumed one per round
        and is revisited only if it survives. Returns the resolvents, their ranks,
        the pending rows that produced one, and how many were covering; None at the
        fixed point.
        """
        rows = self._cores
        new_rows = torch.where(pending)[0]
        if not new_rows.numel() or not rows.numel():
            return None
        values = rows[rows != 0]
        variables, columns = torch.unique(values.abs() - 1, sorted=True, return_inverse=True)
        dense = torch.zeros((rows.shape[0], variables.numel()), dtype=torch.float32, device=rows.device)
        lanes, slots = torch.nonzero(rows, as_tuple=True)
        dense[lanes, columns] = rows[lanes, slots].sign().to(dense)
        lengths = dense.abs().sum(1)
        found: List[torch.Tensor] = []
        for start in range(0, new_rows.numel(), self.chunk):
            selected = new_rows[start:start + self.chunk]
            for other in range(0, rows.shape[0], self.chunk):
                indices = torch.arange(other, min(other + self.chunk, rows.shape[0]), device=rows.device)
                dot = dense[selected] @ dense[indices].T
                overlap = dense[selected].abs() @ dense[indices].abs().T
                opposite = (overlap - dot) == 2
                shared = (overlap + dot) / 2
                # Either side may be the short one; the long side loses the pivot.
                as_short = torch.nonzero(opposite & (lengths[selected, None] - shared == 1))
                as_long = torch.nonzero(opposite & (lengths[indices][None] - shared == 1))
                # Columns: owner, partner, short row, long row.
                if as_short.numel():
                    owner, partner = selected[as_short[:, 0]], indices[as_short[:, 1]]
                    found.append(torch.stack((owner, partner, owner, partner), 1))
                if as_long.numel():
                    owner, partner = selected[as_long[:, 0]], indices[as_long[:, 1]]
                    found.append(torch.stack((owner, partner, partner, owner), 1))
        if not found:
            return None
        pairs = torch.cat(found)
        shorts, longs = pairs[:, 2], pairs[:, 3]
        pivots = (dense[shorts] * dense[longs] < 0).to(torch.int64).argmax(1)
        mask = rows[longs].abs() == (variables[pivots] + 1)[:, None]
        resolved, ranks = _canonical_ranked(
            rows[longs].masked_fill(mask, 0),
            self._ranks[longs].masked_fill(mask, 0) if self._ranks is not None else None,
        )
        width = max(rows.shape[1], resolved.shape[1])
        padded = torch.nn.functional.pad(resolved, (0, width - resolved.shape[1]))
        stored = torch.nn.functional.pad(rows, (0, width - rows.shape[1]))
        unsubsumed = torch.ones(resolved.shape[0], dtype=torch.bool, device=rows.device)
        for start in range(0, padded.shape[0], self.chunk):
            for other in range(0, stored.shape[0], self.chunk):
                unsubsumed[start:start + self.chunk] &= ~_subset_rows(
                    stored[other:other + self.chunk], padded[start:start + self.chunk],
                ).any(0)
        if not unsubsumed.any():
            return None
        # Library order decides each owner's first candidate, as a sequential scan would.
        candidates = torch.where(unsubsumed)[0]
        order = candidates[torch.argsort(pairs[candidates, 1], stable=True)]
        owners, inverse = torch.unique(pairs[order, 0], return_inverse=True)
        first = torch.full((owners.numel(),), order.numel(), dtype=torch.long, device=rows.device)
        first.scatter_reduce_(0, inverse, torch.arange(order.numel(), device=rows.device), "amin")
        chosen = order[first]
        unique, inverse_rows = _first_unique_rows(padded[chosen])
        covering = (lengths[shorts[chosen]] < lengths[longs[chosen]])[inverse_rows]
        kept_ranks = ranks[chosen][inverse_rows] if ranks is not None else None
        return unique, kept_ranks, owners, int(covering.sum())

    def take_complements(self, limit: int, device: torch.device) -> torch.Tensor:
        """Flip each new core's deepest actual decision once; never recurse."""
        if self._ranks is None:
            return torch.empty((0, 0), dtype=torch.long, device=device)
        rows = torch.where(~self._attempted & (self._cores != 0).any(1))[0][:max(0, limit)]
        self._attempted[rows] = True
        candidates = self._cores[rows].clone()
        if rows.numel():
            pivot = self._ranks[rows].argmax(1)
            candidates[torch.arange(rows.numel(), device=rows.device), pivot] *= -1
        return candidates.to(device)

    def pack(self, device: torch.device) -> torch.Tensor:
        return self._cores.to(device)


def apply_literal_assignments(
    batch: SubproblemBatch,
    assignments: torch.Tensor,
    active_variables: torch.Tensor,
    codec: LiteralCodec,
) -> SubproblemBatch:
    """Decode compact inferred phases back to a lossless subproblem batch."""
    signs = {key: value.clone() for key, value in (batch.split_signs or {}).items()}
    specs = _state_spec_row_count(batch)
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


def _replace_literal_assignments(
    batch: SubproblemBatch, literals: torch.Tensor, codec: LiteralCodec,
) -> None:
    """Rebuild only the supplied conditions; implied phases are not free facts."""
    batch.split_signs = {lid: torch.zeros_like(signs)
                         for lid, signs in (batch.split_signs or {}).items()}
    variables = torch.unique(literals[literals != 0].abs() - 1, sorted=True)
    dense = torch.zeros((batch.batch_size, variables.numel()), dtype=torch.int8, device=literals.device)
    lanes, slots = torch.nonzero(literals, as_tuple=True)
    columns = torch.searchsorted(variables, literals[lanes, slots].abs() - 1)
    dense[lanes, columns] = literals[lanes, slots].sign().to(torch.int8)
    apply_literal_assignments(batch, dense, variables, codec)


def _literal_keys(rows: torch.Tensor) -> List[bytes]:
    """Exact canonical identities of signed literal rows, independent of layout."""
    return [row[row != 0].tobytes() for row in _canonical_rows(rows).cpu().numpy()]


def _row_literal_counts(batch: SubproblemBatch) -> torch.Tensor:
    """Count each row's assigned phases without packing their literals."""
    counts = torch.zeros(batch.batch_size, dtype=torch.long, device=batch.lb.device)
    for values in (batch.split_signs or {}).values():
        counts += values.reshape(batch.batch_size, values.shape[1], -1)[:, 0].ne(0).sum(1).to(counts.device)
    return counts


def _merge_hash_weights(count: int, device: torch.device) -> torch.Tensor:
    """Fixed odd weights: hashes only propose pairs, exact comparison decides."""
    generator = torch.Generator().manual_seed(0)
    return (torch.randint(0, 1 << 62, (count,), generator=generator, device="cpu") | 1).to(device)


def _complementary_pairs(
    dense: torch.Tensor, variables: torch.Tensor, eligible: torch.Tensor,
    fresh: torch.Tensor, undo: Collection[bytes],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose disjoint rows equal except one opposed column, lexicographically.

    A pair rebuilding an already bounded unresolved literal set would only undo
    its split, unless propagation has since given either side a new condition.
    """
    rows, columns = torch.nonzero(dense * eligible[:, None].to(dense), as_tuple=True)
    empty = rows[:0]
    if not rows.numel():
        return empty, empty, empty
    signs = dense[rows, columns].to(torch.long)
    weights = _merge_hash_weights(dense.shape[1], dense.device)
    hashes = torch.zeros(dense.shape[0], dtype=torch.long, device=dense.device)
    hashes.index_add_(0, rows, signs * weights[columns])
    groups = torch.unique(
        torch.stack((columns, hashes[rows] - signs * weights[columns]), 1),
        dim=0, return_inverse=True,
    )[1]
    negative = torch.where(signs < 0)[0]
    negative = negative[groups[negative].argsort(stable=True)]
    positive = torch.where(signs > 0)[0]
    sizes = torch.bincount(groups[negative], minlength=int(groups.max()) + 1)
    starts = sizes.cumsum(0) - sizes
    fanout = sizes[groups[positive]].clamp(max=_MERGE_GROUP_FANOUT)
    total = int(fanout.sum())
    offsets = torch.arange(total, device=dense.device) - torch.repeat_interleave(
        fanout.cumsum(0) - fanout, fanout,
    )
    partners = rows[negative[torch.repeat_interleave(starts[groups[positive]], fanout) + offsets]]
    anchors = torch.repeat_interleave(rows[positive], fanout)
    pivots = torch.repeat_interleave(columns[positive], fanout)
    left, right = torch.minimum(anchors, partners), torch.maximum(anchors, partners)
    # Exact rows, not their hash, establish the projection.
    same = dense[left] == dense[right]
    same[torch.arange(total, device=dense.device), pivots] = True
    keep = same.all(1)
    left, right, pivots = left[keep], right[keep], pivots[keep]
    for key in (pivots, right, left):
        order = key.argsort(stable=True)
        left, right, pivots = left[order], right[order], pivots[order]
    if undo and left.numel():
        projected = dense[left].to(torch.long) * (variables + 1)
        projected[torch.arange(left.numel(), device=dense.device), pivots] = 0
        rebuilds = torch.tensor([key in undo for key in _literal_keys(projected)], device=dense.device)
        keep = ~rebuilds | fresh[left] | fresh[right]
        left, right, pivots = left[keep], right[keep], pivots[keep]
    # Iterating locally minimal edges equals scanning sorted edges greedily.
    chosen = []
    ranks = torch.arange(left.numel(), device=dense.device)
    while ranks.numel():
        first = torch.full((dense.shape[0],), left.numel(), device=dense.device)
        first.scatter_reduce_(0, left[ranks], ranks, "amin")
        first.scatter_reduce_(0, right[ranks], ranks, "amin")
        minimal = ranks[(first[left[ranks]] == ranks) & (first[right[ranks]] == ranks)]
        chosen.append(minimal)
        used = torch.zeros(dense.shape[0], dtype=torch.bool, device=dense.device)
        used[left[minimal]] = used[right[minimal]] = True
        ranks = ranks[~(used[left[ranks]] | used[right[ranks]])]
    if not chosen:
        return empty, empty, empty
    selected = torch.cat(chosen).sort().values
    return left[selected], right[selected], pivots[selected]


@dataclass(frozen=True)
class _FrontierMerge:
    """Rows surviving a merge fixed point; ``merged`` rows need a fresh bound.

    ``literals`` holds only surviving merged rows, in row order.
    """

    alive: torch.Tensor
    merged: torch.Tensor
    literals: torch.Tensor
    depths: torch.Tensor
    lower_bounds: Optional[torch.Tensor]
    parent_margins: Optional[torch.Tensor]
    decisions: Optional[torch.Tensor]
    node_ids: Optional[torch.Tensor]
    parent_ids: Optional[torch.Tensor]
    pairs: int


def _merge_frontier(
    packed: torch.Tensor, eligible: torch.Tensor, fresh: torch.Tensor,
    depths: torch.Tensor, decisions: Optional[torch.Tensor],
    node_ids: Optional[torch.Tensor], parent_ids: Optional[torch.Tensor],
    undo: Collection[bytes],
    lower_bounds: Optional[torch.Tensor] = None, parent_margins: Optional[torch.Tensor] = None,
) -> _FrontierMerge:
    """Replace complementary rows by their union until no pair remains.

    Eligible rows share one input box, so a union's region is exactly its
    retained conditions. Only decisions in both constituents remain branch
    history; a decision of one side would not describe the union. The weaker
    side's bound only ranks the union for its mandatory fresh bound; it is no
    proof, and an infinite rank would erase every other row's normalized rank.
    """
    count = packed.shape[0]
    device = packed.device
    values = packed[packed != 0]
    variables, columns = torch.unique(values.abs() - 1, sorted=True, return_inverse=True)
    dense = torch.zeros((count, variables.numel()), dtype=torch.int8, device=device)
    dense[torch.nonzero(packed, as_tuple=True)[0], columns] = values.sign().to(torch.int8)
    alive = torch.ones(count, dtype=torch.bool, device=device)
    merged = torch.zeros_like(alive)
    fresh, depths = fresh.clone(), depths.clone()
    decisions = None if decisions is None else decisions.clone()
    node_ids = None if node_ids is None else node_ids.clone()
    parent_ids = None if parent_ids is None else parent_ids.clone()
    lower_bounds = None if lower_bounds is None else lower_bounds.clone()
    parent_margins = None if parent_margins is None else parent_margins.clone()
    pairs = 0
    while True:
        left, right, pivots = _complementary_pairs(dense, variables, eligible & alive, fresh, undo)
        if not left.numel():
            break
        pairs += left.numel()
        dense[left, pivots] = 0
        alive[right] = False
        merged[left] = True
        fresh[left] |= fresh[right]
        depths[left] = (torch.minimum(depths[left], depths[right]) - 1).clamp(min=0)
        for rank in (lower_bounds, parent_margins):
            if rank is not None:
                rank[left] = torch.minimum(rank[left], rank[right])
        if decisions is not None:
            shared = (decisions[left][:, :, None] == decisions[right][:, None, :]).any(2)
            decisions[left] = decisions[left].masked_fill(~shared, 0)
        if node_ids is not None and parent_ids is not None:
            revived = (parent_ids[left] == parent_ids[right]) & (parent_ids[left] >= 0)
            node_ids[left] = torch.where(revived, parent_ids[left], node_ids[left])
            parent_ids[left] = -1
    return _FrontierMerge(
        alive, merged, dense[alive & merged].to(torch.long) * (variables + 1), depths,
        lower_bounds, parent_margins, decisions, node_ids, parent_ids, pairs,
    )


@torch.no_grad()
def propagate(
    batches: Sequence[SubproblemBatch],
    library: CoreLibrary,
    codec: LiteralCodec,
    *,
    core_chunk: int,
) -> Tuple[Tuple[SubproblemBatch, ...], Tuple[torch.Tensor, ...]]:
    """Apply certified cores jointly to pending and active groups to fixpoint.

    ``core_chunk`` caps the cores per matmul chunk; it bounds memory only, the
    fixpoint is independent of it. Return the propagated groups and their
    original-row selectors so external bound/witness arrays stay aligned.
    """
    if core_chunk < 1:
        raise ValueError("core_chunk must be positive")
    if not batches:
        return (), ()
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
        return tuple(batches), tuple(
            torch.arange(batch.batch_size, device=batch.lb.device) for batch in batches
        )
    device = batches[0].lb.device
    packed_batches = [packed.to(device) for packed in packed_batches]
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
    # Snapshot actual decisions before adding inferred phases. Runtime roots
    # already carry ordered slots; this also covers manually constructed batches.
    for batch, packed in zip(batches, packed_batches):
        if batch.decision_literals is None:
            batch.decision_literals = packed.to(batch.lb.device)
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
    kept_rows: List[torch.Tensor] = []
    offset = 0
    for batch, size in zip(batches, group_sizes):
        group_active = active[offset : offset + size]
        keep = torch.where(group_active)[0]
        kept_rows.append(keep.to(batch.lb.device))
        restricted = batch.select(keep)
        restricted_assignments = assignments[offset : offset + size].index_select(0, keep)
        propagated = apply_literal_assignments(
            restricted, restricted_assignments.to(batch.lb.device),
            active_variables.to(batch.lb.device), codec
        )
        assert propagated.batch_size <= size, (
            "CLIMB invariant violated: propagation increased a group batch size "
            f"(input_rows={size}, output_rows={propagated.batch_size})"
        )
        result_batches.append(propagated)
        offset += size
    return tuple(result_batches), tuple(kept_rows)


def is_certified(slack: torch.Tensor, out_kind: str) -> torch.Tensor:
    """Per-lane certification mask over ``[N, M]`` slack rows.

    Replay has already subtracted the shared band for closed unsafe sets.
    """
    proven = ~unproven_spec_rows(slack, out_kind)
    return proven.any(dim=1) if out_kind == OutKind.UNSAFE_LINEAR else proven.all(dim=1)


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
    base_reference_bounds: Optional[Dict[int, Bounds]] = None,
    refresh_mode: str = "none",
    lin_support_refinement: bool = True,
    refresh_rows_cap: int = 64,
    forward_lin_max_perturbed: Optional[int] = None,
    phase_slope_signs: Optional[Dict[int, torch.Tensor]] = None,
) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Replay a frozen alpha/eta certificate directly through ``DualSolver``.

    With ``phase_slope_signs`` the batch's ``split_signs`` are a reduced core and
    those signs its source literals; see ``DualSolver.compute_certified_bound``.

    Returns ``(slack, costs)`` — for ``UNSAFE_LINEAR`` both are gathered on the
    pinned strict row (``slack [N, 1]``, ``costs [N, 1, W]``); otherwise
    ``slack`` is ``[N, M]`` and ``costs`` is ``[N, M, W]``. ``costs`` is
    ``None`` when ``with_costs=False``. Returns ``None`` when the batch lacks
    replayable state or the solver emits no usable per-layer nu.
    """
    if with_costs and (literals is None or codec is None):
        raise ValueError("with_costs=True requires packed literals")
    split_refresh = refresh_mode in _SPLIT_REFRESH_MODES
    if split_refresh and base_reference_bounds is None:
        raise ValueError("split-dependent replay requires base_reference_bounds")
    if (
        batch.incremental_alpha is None
        or (batch.warm_state_present is not None
            and not bool(batch.warm_state_present.all()))
        or batch.incremental_eta is None
        or batch.split_signs is None
        or m_specs < 1
    ):
        return None
    from act.back_end.dual_tf.tf_mlp import DegenerateIntervalError
    from act.back_end.solver.solver_dual import DualSolver

    n_lanes = batch.batch_size
    if c_rows.dim() != 2 or c_rows.shape[0] != n_lanes * m_specs or thresholds.shape != (n_lanes, m_specs):
        raise ValueError("replay objectives must match lane/spec dimensions")
    supports = None
    if split_refresh:
        from act.back_end.bab.bound_refresh import refresh_bounds
        assert base_reference_bounds is not None
        refreshed = refresh_bounds(
            net, base_reference_bounds, batch.split_signs,
            literals=literals if with_costs else None,
            codec=codec if with_costs else None,
            linear=refresh_mode == "split_refresh_linear",
            support_refinement=lin_support_refinement,
            rows_cap=refresh_rows_cap,
            forward_lin_max_perturbed=forward_lin_max_perturbed,
            warm_signs=phase_slope_signs,
        )
        reference_bounds, supports = refreshed.bounds, refreshed.supports
    eta = {
        layer_id: value.detach().clone().clamp(min=0)
        for layer_id, value in batch.incremental_eta.items()
    }
    try:
        result = DualSolver().compute_certified_bound(
            net,
            reference_bounds,
            c_rows,
            M=m_specs,
            alpha=batch.incremental_alpha,
            eta=eta,
            split_signs=batch.split_signs,
            optimize=False,
            return_nu_per_layer=with_costs,
            local_phase_clamp=True,
            phase_slope_signs=phase_slope_signs,
        )
    except DegenerateIntervalError:
        # The retained literals' refreshed box is inverted, so this replay proves nothing.
        return None
    assert result.dual_margins is not None
    margins = result.dual_margins.reshape(n_lanes, m_specs)
    threshold_rows = thresholds.reshape(n_lanes, m_specs).to(margins)
    slack = margins - threshold_rows
    pinned_rows: Optional[torch.Tensor] = None
    if out_kind in OutputSpec.CLOSED_KINDS:
        # The budget must remain a backward certificate, including its strict
        # rounding band. Combined forward margins cannot fund literal deletion.
        slack = slack - certification_tolerance(
            torch.maximum(margins.abs(), threshold_rows.abs())
        )
    if out_kind == OutKind.UNSAFE_LINEAR:
        passing = ~unproven_spec_rows(slack, out_kind)
        pinned_rows = passing.to(torch.int64).argmax(dim=1)
    costs: Optional[torch.Tensor] = None
    if with_costs:
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
        assert literals is not None
        assert codec is not None
        if supports is None:
            costs = _literal_costs(literals, codec, slack, eta, nu, reference_bounds, pinned_rows)
        else:
            assert base_reference_bounds is not None
            costs = _supported_literal_costs(
                literals, codec, slack, eta, nu, base_reference_bounds, reference_bounds,
                batch.split_signs, supports, pinned_rows,
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
    """Estimate Eq. 12 costs ``(eta + relu(-nu)) * m``, shaped ``[N, M, W]``.

    Retained alpha can change the relaxation after literal deletion. These
    costs select candidates; only replay certifies a changed learned core.
    """
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
def _supported_literal_costs(
    literals: torch.Tensor,
    codec: LiteralCodec,
    slack: torch.Tensor,
    eta: Dict[int, torch.Tensor],
    nu: Dict[int, torch.Tensor],
    base_bounds: Dict[int, Bounds],
    reference_bounds: Dict[int, Bounds],
    split_signs: Dict[int, torch.Tensor],
    supports: Dict[int, BoundSupport],
    pinned_rows: Optional[torch.Tensor],
) -> torch.Tensor:
    """Charge ``eta*m + sum(eps_i)`` for split-refreshed certificates.

    ``m`` and each ``eps_i`` are measured over the split-free reference, which
    holds on every retained subset. ``eps_i`` is the largest violation there of
    the upper line frozen on the phase-clamped refreshed interval, weighted by
    the negative part of its coefficient (lower lines are global). It is charged
    to the endpoint support that line used. A support needed but unknown makes
    the whole row undeletable; these costs only propose a row for replay.
    """
    zero_nu = {layer_id: torch.zeros_like(value) for layer_id, value in eta.items()}
    costs = _literal_costs(literals, codec, slack, eta, zero_nu, base_bounds, None)
    unknown = torch.zeros(literals.shape[0], dtype=torch.bool, device=costs.device)
    for layer_id, coefficient in nu.items():
        weight = (-coefficient).clamp(min=0).to(costs)
        support = supports.get(layer_id)
        if support is None or layer_id not in reference_bounds or layer_id not in base_bounds:
            unknown |= weight.flatten(1).gt(0).any(1)
            continue
        lower = reference_bounds[layer_id].lb.flatten(1).to(costs)
        upper = reference_bounds[layer_id].ub.flatten(1).to(costs)
        if layer_id in split_signs:
            phase = split_signs[layer_id][:, 0].to(costs)
            lower = torch.where(phase > 0, lower.clamp(min=0), lower)
            upper = torch.where(phase < 0, upper.clamp(max=0), upper)
            upper = torch.maximum(lower, upper)
        lower0 = base_bounds[layer_id].lb.flatten(1).to(costs)
        upper0 = base_bounds[layer_id].ub.flatten(1).to(costs)
        span = (upper - lower).clamp(min=torch.finfo(costs.dtype).tiny)
        chord = torch.maximum(upper * (lower - lower0), -lower * (upper0 - upper)) / span
        unstable = (lower < 0) & (upper > 0)
        known = support.known[:, None]
        for residual, mask in (
            (torch.where(unstable, chord.clamp(min=0), 0), support.lower | support.upper),
            (torch.where(lower >= 0, (-lower0).clamp(min=0), 0), support.lower),
            (torch.where(upper <= 0, upper0.clamp(min=0), 0), support.upper),
        ):
            charge = weight * residual[:, None]
            unknown |= ((charge > 0) & ~known).flatten(1).any(1)
            used = charge.ne(0).flatten(0, 1).any(0)
            if bool(used.any()):
                costs += charge[..., used].masked_fill(~known[..., used], 0) @ mask[:, used].to(costs)
    costs = torch.nan_to_num(costs, nan=torch.inf, posinf=torch.inf)
    costs[unknown] = torch.inf
    if pinned_rows is not None:
        rows = pinned_rows.to(device=costs.device, dtype=torch.long)
        costs = costs.gather(1, rows[:, None, None].expand(-1, 1, costs.shape[2]))
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
    recheck_k: int = 0,
    recheck: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Choose budget deletions plus at most ``recheck_k`` one-step rechecks.

    The caller must replay every changed result before core admission,
    including rows changed only by the cost-budget heuristic.

    Per row ``m`` the budget is ``theta * slack_m - delta_m`` with
    ``delta_m = delta_abs + delta_rel * |slack_m|``; rows with
    ``slack_m <= delta_m`` never fund a deletion. The budget arithmetic runs
    in the dtype of ``costs``/``slack``. Literals are ordered by
    ``(max_m cost_m / budget_m, column, slot)`` and the longest feasible prefix
    is deleted; exactly-zero costs are never blocked by a negative budget. At
    ``(theta, recheck_k) == (0, 0)`` a tensor mask selects exactly those zeros,
    with no sort. A recheck retries the next literal in that same order for the
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
    if theta == 0.0 and recheck_k == 0:
        # Rows lacking a finite certificate or a safe slack keep every literal.
        delta = delta_abs + delta_rel * slack.abs()
        eligible = (
            torch.isfinite(slack).all(1)
            & torch.isfinite(costs).all(2).all(1)
            & (slack > delta).all(1)
        )
        retained &= ~(eligible[:, None] & (costs == 0).all(1))
        return literals.masked_fill(~retained, 0)
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
            if not bool((candidate <= budget.clamp(min=0)).all().item()):
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
        assert recheck is not None
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
    base_bounds: Optional[Dict[int, Bounds]] = None
    checked_candidates: Optional[torch.Tensor] = None
    checked_passing: Optional[torch.Tensor] = None


class ClimbSession:
    """Own the query-local CLIMB library, metrics, and BaB-loop hooks."""

    def __init__(
        self, config: BaBConfig, net: Net, out_kind: str,
        dual_config: Optional[DualConfig] = None,
    ) -> None:
        self._net = net
        self._out_kind = out_kind
        self._coarsening_enabled = config.climb_coarsening_enabled
        self._propagation_enabled = config.climb_propagation_enabled
        self._presplit_propagation_enabled = config.climb_presplit_propagation_enabled
        self._complement_bounding = config.climb_complement_bounding
        self._steer = config.climb_steer
        self._merge_enabled = config.climb_merge_enabled
        self._merge_min_proof_rate = config.climb_merge_min_proof_rate
        self._merge_pending = torch.empty(0, dtype=torch.long)
        # Every core ever considered, independent of later subsumption/eviction.
        self._seen_cores: set[tuple[int, ...]] = set()
        self._merge_armed = False
        self._bounded_unresolved: set[bytes] = set()
        self._fresh_rows: set[bytes] = set()
        self._frontier_undo: Optional[Tuple[set[bytes], set[bytes], set[bytes]]] = None
        self._theta = config.climb_theta
        self._delta_abs = config.climb_delta_abs
        self._delta_rel = config.climb_delta_rel
        self._recheck_k = config.climb_recheck_k
        self._terminal_cores = config.terminal_lp_enabled and config.climb_terminal_cores
        self._terminal_lane_bases: Dict[tuple[bytes, bytes], Any] = {}
        self._propagate_core_chunk = config.climb_propagate_core_chunk
        self._refresh_mode = config.root_bounds_reuse
        self._lin_support_refinement = config.climb_lin_support_refinement
        self._refresh_rows_cap = config.per_subproblem_refine_rows_cap
        self._forward_lin_max_perturbed = (dual_config or DualConfig()).forward_lin_max_perturbed
        from act.back_end.verifier import gather_input_spec_layers, seed_from_input_specs
        root = seed_from_input_specs(gather_input_spec_layers(net))
        self._root_bounds = Bounds(root.lb.clone(), root.ub.clone())
        self.codec = LiteralCodec.from_net(net)
        self.library = CoreLibrary(config.climb_max_cores, track_complements=self._complement_bounding)
        self.metrics = ClimbMetrics(enabled=True, library=self.library)

    @classmethod
    def from_config(
        cls, config: BaBConfig, net: Net, out_kind: str,
        dual_config: Optional[DualConfig] = None,
    ) -> Optional[ClimbSession]:
        validate_climb_config(config)
        return cls(config, net, out_kind, dual_config) if config.climb_enabled else None

    def before_pop(self, pool: TopKBounding) -> None:
        if not self._propagation_enabled:
            self.metrics.observe_frontier(len(pool))
            return
        if self.library.core_count > 0:
            # H1: eager discharge of the pending pool before any lane is bounded.
            # Only rows stale against this library are scanned; the whole library
            # applies to them, as an old core may fire on a newly written phase.
            # The pool's score, cooling and probe state is not involved.
            stale = pool.propagation_batch(self.library.revision)
            self.metrics.propagation_rows_skipped += len(pool) - (stale[0].numel() if stale else 0)
            if stale is not None:
                selected, pending_before = stale
                propagate_started = time.perf_counter()
                (pending_after,), (kept,) = propagate(
                    [pending_before], self.library, self.codec,
                    core_chunk=self._propagate_core_chunk,
                )
                self.metrics.propagate_time_s += time.perf_counter() - propagate_started
                self.metrics.propagation_rows_scanned += selected.numel()
                self.metrics.prebound_discharged += (
                    pending_before.batch_size - pending_after.batch_size
                )
                pool.apply_propagation(selected, kept, pending_after, self.library.revision)
                self._note_propagation(pending_before, kept, pending_after)
            if pool.empty:
                return
        if not self._presplit_propagation_enabled:
            self._merge(pool)  # The deferred pre-split merge runs on this propagation.
        self.metrics.observe_frontier(len(pool))

    def _on_root_domain(self, batch: SubproblemBatch) -> torch.Tensor:
        """Rows whose input box is the query's: only these join core lifting."""
        lower = self._root_bounds.lb.flatten(1).to(batch.lb)
        upper = self._root_bounds.ub.flatten(1).to(batch.ub)
        return (batch.lb == lower).all(1) & (batch.ub == upper).all(1)

    def _source_literals(self, batch: SubproblemBatch) -> torch.Tensor:
        """Pack a row's complete conditions S, each to be proved, not assumed.

        Propagation-written phases precede actual decisions, so insertion order
        still ranks the deepest decision last for complement probing.
        """
        packed = pack_split_matrix(
            batch.split_signs, batch.batch_size, self.codec, device=batch.lb.device,
        )
        decisions = batch.decision_literals
        if decisions is None or packed.shape[1] == 0:
            return packed
        decided = (packed[:, :, None] == decisions[:, None, :]).any(2)
        applied = (decisions[:, :, None] == packed[:, None, :]).any(2)
        rows = torch.cat((packed.masked_fill(decided, 0), decisions.masked_fill(~applied, 0)), 1)
        rows = rows.gather(1, (rows == 0).to(torch.int64).argsort(dim=1, stable=True))
        return rows[:, :packed.shape[1]]

    def _insert_cores(self, rows: torch.Tensor, *, suppress_complements: bool = False) -> int:
        """Insert cores; any core not considered before arms one frontier merge."""
        resolved = self.library.resolved
        self._consider_cores(rows)
        inserted = self.library.insert(rows, suppress_complements=suppress_complements)
        if self.library.resolved != resolved:
            self._consider_cores(self.library.pack(rows.device))
        return inserted

    def _consider_cores(self, rows: torch.Tensor) -> None:
        if not self._merge_enabled:
            return
        for row in _canonical_rows(rows).tolist():
            core = tuple(literal for literal in row if literal)
            if core not in self._seen_cores:
                self._seen_cores.add(core)
                self._merge_armed = True

    def _remember_unresolved(self, rows: SubproblemBatch) -> None:
        """Record bounded unresolved sets; a merge must not simply rebuild them."""
        root_rows = rows.select(torch.where(self._on_root_domain(rows))[0])
        keys = set(_literal_keys(pack_split_matrix(
            root_rows.split_signs, root_rows.batch_size, self.codec, device=rows.lb.device,
        )))
        if self._frontier_undo is not None:
            self._frontier_undo[0].update(keys - self._bounded_unresolved)
            self._frontier_undo[2].update(keys & self._fresh_rows)
        self._bounded_unresolved |= keys
        self._fresh_rows -= keys

    def _note_propagation(
        self, before: SubproblemBatch, kept: torch.Tensor, after: SubproblemBatch,
    ) -> None:
        """Rows given a new condition may merge past their earlier undo guard."""
        if not after.batch_size:
            return
        writes = _row_literal_counts(after) - _row_literal_counts(before).index_select(0, kept)
        self.metrics.unit_writes += int(writes.sum())
        if not self._merge_enabled:
            return
        rows = after.select(torch.where(writes > 0)[0])
        keys = set(_literal_keys(pack_split_matrix(
            rows.split_signs, rows.batch_size, self.codec, device=rows.lb.device,
        )))
        if self._frontier_undo is not None:
            self._frontier_undo[1].update(keys - self._fresh_rows)
        self._fresh_rows |= keys

    def _merge(
        self, pool: TopKBounding, active: Optional[SubproblemBatch] = None,
    ) -> Tuple[Optional[SubproblemBatch], torch.Tensor]:
        """Merge root-domain complementary rows to a fixed point on an unseen core.

        Merged rows return to pending, so they are bounded again before any
        split; untouched rows keep their state. Nothing is committed
        before ``replace_all``, so a failure leaves the trigger armed.
        """
        keep_active = torch.arange(
            active.batch_size if active is not None else 0,
            device=active.lb.device if active is not None else self._root_bounds.lb.device,
        )
        if not self._merge_armed:
            return active, keep_active
        resident = pool.checkpoint()
        parts = [part for part in (resident, active) if part is not None and part.batch_size]
        if not parts:
            self._merge_armed = False
            return active, keep_active
        n_pending = resident.batch_size if resident is not None else 0
        joint = parts[0].drop_warm_state()
        if len(parts) == 2:
            joint = joint.concat(parts[1].drop_warm_state())
        device = joint.lb.device
        packed = pack_split_matrix(joint.split_signs, joint.batch_size, self.codec, device=device)
        fresh = torch.tensor(
            [key in self._fresh_rows for key in _literal_keys(packed)] if self._fresh_rows
            else [False] * joint.batch_size,
            dtype=torch.bool, device=device,
        )
        plan = _merge_frontier(
            packed, self._on_root_domain(joint), fresh, joint.depths, joint.decision_literals,
            joint.node_id, joint.parent_id, self._bounded_unresolved,
            joint.lower_bound, joint.parent_margins,
        )
        if not plan.pairs:
            self._merge_armed = False
            return active, keep_active
        rows = torch.where(plan.alive & plan.merged)[0]
        # A union starts its mandatory fresh bound from one constituent's optimizer
        # state: any such state is a valid start, and a cold lane would chill its
        # whole wave. Multipliers of conditions the union dropped no longer apply.
        offset = 0
        merged_parts = []
        for part in parts:
            owned = rows[(rows >= offset) & (rows < offset + part.batch_size)] - offset
            offset += part.batch_size
            if owned.numel():
                merged_parts.append(part.select(owned))
        merged = merged_parts[0] if len(merged_parts) == 1 else merged_parts[0].concat(merged_parts[1])
        _replace_literal_assignments(merged, plan.literals, self.codec)
        if merged.incremental_eta is not None:
            signs = merged.split_signs or {}
            merged.incremental_eta = {
                lid: eta * signs[lid].ne(0).to(eta) if lid in signs else torch.zeros_like(eta)
                for lid, eta in merged.incremental_eta.items()
            }
        merged.depths = plan.depths[rows]
        merged.decision_literals = plan.decisions[rows] if plan.decisions is not None else None
        merged.node_id = plan.node_ids[rows] if plan.node_ids is not None else None
        merged.parent_id = plan.parent_ids[rows] if plan.parent_ids is not None else None
        merged.propagation_revision = None  # New conditions, not yet at a fixed point.
        merged.lower_bound = plan.lower_bounds[rows] if plan.lower_bounds is not None else None
        merged.parent_margins = plan.parent_margins[rows] if plan.parent_margins is not None else None
        untouched = plan.alive & ~plan.merged
        pending_rows = torch.where(untouched[:n_pending])[0]
        assert resident is not None or not pending_rows.numel()
        pool.replace_all(
            resident.select(pending_rows).concat(merged)
            if resident is not None and pending_rows.numel() else merged
        )
        if active is not None:
            keep_active = torch.where(untouched[n_pending:])[0].to(active.lb.device)
            active = active.select(keep_active)
        self._merge_armed = False
        if merged.node_id is not None:
            self._merge_pending = torch.cat((self._merge_pending.to(merged.node_id), merged.node_id))
        self.metrics.merged_pairs += plan.pairs
        self.metrics.merge_calls += 1
        self.metrics.merge_rows += rows.numel()
        self.metrics.merge_literals += int(_row_literal_counts(merged).sum())
        return active, keep_active

    def _settle_merged_rows(self, batch: SubproblemBatch, statuses: Sequence[str]) -> None:
        """Stop merging once bounded unions rarely prove, as they then only split again."""
        if not self._merge_pending.numel() or batch.node_id is None:
            return
        settled = torch.isin(batch.node_id, self._merge_pending.to(batch.node_id))
        if not bool(settled.any()):
            return
        proved = torch.tensor([status == SolveStatus.UNSAT for status in statuses], device=settled.device)
        self.metrics.merge_proved_rows += int((settled & proved).sum())
        self.metrics.merge_open_rows += int((settled & ~proved).sum())
        self._merge_pending = self._merge_pending[~torch.isin(self._merge_pending, batch.node_id)]
        total = self.metrics.merge_proved_rows + self.metrics.merge_open_rows
        if total >= _MERGE_PROBE_ROWS and self.metrics.merge_proved_rows < self._merge_min_proof_rate * total:
            self._merge_enabled = False
            self._merge_armed = False
            self.metrics.merge_stopped = True

    def checkpoint_frontier(self) -> Tuple[bool, Tuple[int, ...]]:
        """Open an undo journal for this wave's uncommitted frontier bookkeeping."""
        self._frontier_undo = (set(), set(), set())
        return self._merge_armed, tuple(getattr(self.metrics, name) for name in _FRONTIER_COUNTERS)

    def restore_frontier(self, state: Tuple[bool, Tuple[int, ...]]) -> None:
        """Roll back frontier bookkeeping, retaining independently learned proofs.

        A merge undone here is owed again; a core seen since the checkpoint
        stays armed, because the set of considered cores is never rolled back.
        """
        armed, counters = state
        for name, value in zip(_FRONTIER_COUNTERS, counters):
            setattr(self.metrics, name, value)
        self._merge_armed |= armed
        assert self._frontier_undo is not None
        bounded, fresh, cleared = self._frontier_undo
        self._bounded_unresolved -= bounded
        self._fresh_rows -= fresh
        self._fresh_rows |= cleared
        self._frontier_undo = None

    def after_pop(self, frontier_rows: int) -> None:
        self.metrics.observe_frontier(frontier_rows)

    def after_main_bound(self, k_actual: int) -> None:
        self.metrics.main_bound_calls += 1
        self.metrics.main_bound_row_passes += k_actual

    def retire_lanes_without_relu_candidate(self, retired: int) -> None:
        """Record unresolved lanes that cannot use CLIMB's neuron-only splits."""
        self.metrics.retired_unknown_lanes += retired

    def learn(
        self,
        batch: SubproblemBatch,
        statuses: Sequence[str],
        dual_solve_result: DualBatchResult,
        k_actual: int,
        *,
        allow_complements: bool = True,
    ) -> None:
        # Replay uses the immutable root-domain reference, optionally refined
        # once at the root. Budget coarsening proposes a smaller row; a final
        # replay must certify it before admission, or restore the original.
        if k_actual != batch.batch_size or len(statuses) != k_actual:
            raise ValueError("CLIMB admission lane count mismatch")
        self._settle_merged_rows(batch, statuses)
        if self._refresh_mode in _SPLIT_REFRESH_MODES and dual_solve_result.base_reference_bounds is None:
            raise ValueError("split-dependent learning requires base_reference_bounds")
        unsat_idx = torch.tensor(
            [i for i, status in enumerate(statuses) if status == SolveStatus.UNSAT],
            device=batch.lb.device,
            dtype=torch.long,
        )
        # A proof learned on a smaller input box cannot become a global
        # root-domain core, even when that lane still carries ReLU decisions.
        unsat_idx = unsat_idx[self._on_root_domain(batch).index_select(0, unsat_idx)]
        if dual_solve_result.forward_only_certified is not None:
            forward_only = dual_solve_result.forward_only_certified.to(unsat_idx.device)
            forward_rows = unsat_idx[forward_only.index_select(0, unsat_idx)]
            unsat_idx = unsat_idx[~forward_only.index_select(0, unsat_idx)]
            if forward_rows.numel():
                # No dual multipliers can price a deletion, but the forward proof
                # already covers this row's whole region: its complete conditions.
                self.metrics.forward_core_rows += forward_rows.numel()
                self.metrics.cores_inserted += self._insert_cores(
                    self._source_literals(batch.select(forward_rows)),
                    suppress_complements=not allow_complements,
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
        # Slice both snapshots with the same lane selector. Split-dependent
        # replay regenerates its conditioned bounds from the split-free one.
        replay_bounds = slice_bounds_dict(
            dual_solve_result.reference_bounds, unsat_idx
        )
        replay_base = (
            slice_bounds_dict(dual_solve_result.base_reference_bounds, unsat_idx)
            if dual_solve_result.base_reference_bounds is not None else None
        )
        replay_c_lanes = dual_solve_result.c_rows.reshape(
            k_actual, replay_m_specs, -1
        ).index_select(0, unsat_idx.to(dual_solve_result.c_rows.device))
        replay_c = replay_c_lanes.reshape(-1, replay_c_lanes.shape[-1])
        replay_thresholds = dual_solve_result.thresholds.index_select(
            0, unsat_idx.to(dual_solve_result.thresholds.device)
        )
        packed = self._source_literals(replay_batch)
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
            with_costs=self._coarsening_enabled,
            base_reference_bounds=replay_base,
            refresh_mode=self._refresh_mode,
            lin_support_refinement=self._lin_support_refinement,
            refresh_rows_cap=self._refresh_rows_cap,
            forward_lin_max_perturbed=self._forward_lin_max_perturbed,
        )
        if replay is not None:
            slack, costs = replay
            valid = is_certified(slack, self._out_kind)
        else:
            slack = costs = valid = None
        if valid is None or not bool(valid.any().item()):
            return
        self.metrics.core_sources += int(valid.sum())
        budget_slack = slack
        assert budget_slack is not None
        context = _ReplayContext(
            replay_batch, replay_bounds, replay_c_lanes, replay_thresholds, replay_m_specs,
            replay_base, packed.clone(), valid.clone(),
        )
        coarsen_started = time.perf_counter()
        if self._coarsening_enabled:
            assert costs is not None
            costs[~valid] = torch.inf
            retained = coarsen(
                packed,
                costs,
                budget_slack,
                theta=self._theta,
                delta_abs=self._delta_abs,
                delta_rel=self._delta_rel,
                recheck_k=self._recheck_k,
                recheck=lambda candidate, rows: self._optional_recheck(context, candidate, rows),
            )
        else:
            retained = packed
        changed = torch.where(valid & (retained != packed).any(dim=1))[0]
        if changed.numel():
            assert context.checked_candidates is not None and context.checked_passing is not None
            fresh = (retained.index_select(0, changed) != context.checked_candidates.index_select(0, changed)).any(1)
            pending = changed[fresh]
            self.metrics.cost_deleted_literals += int(
                ((packed != 0) & (retained == 0)).index_select(0, changed).sum()
            )
            if pending.numel():
                self.metrics.mandatory_admissions += pending.numel()
                self._instrumented_recheck(context, retained.index_select(0, pending), pending)
            certified = context.checked_passing.index_select(0, changed)
            failed = changed[~certified]
            retained[failed] = packed[failed]
            self.metrics.coarsen_replay_fallbacks += int(failed.numel())
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
        self.metrics.cores_inserted += self._insert_cores(
            valid_cores, suppress_complements=not allow_complements,
        )

    def bound_complements(
        self, solve: Callable[[SubproblemBatch, float], DualBatchResult],
        remaining: Callable[[], float], node_budget: int,
    ) -> int:
        """Probe new cores on the original input domain using the shared bounder."""
        if not self._complement_bounding or node_budget <= 0 or remaining() <= 0:
            return 0
        literals = self.library.take_complements(node_budget, self._root_bounds.lb.device)
        count = literals.shape[0]
        if not count:
            return 0
        candidate = SubproblemBatch.from_bounds(Bounds(
            self._root_bounds.lb.expand(count, *self._root_bounds.lb.shape[1:]),
            self._root_bounds.ub.expand(count, *self._root_bounds.ub.shape[1:]),
        ))
        candidate.decision_literals = literals
        candidate.depths = (literals != 0).sum(1)
        _replace_literal_assignments(candidate, literals, self.codec)
        cap = remaining()
        if cap <= 0:
            return 0
        started = time.perf_counter()
        self.metrics.complement_calls += 1
        self.metrics.complement_row_passes += count
        try:
            result = solve(candidate, cap)
        except TimeoutError:
            return count
        finally:
            self.metrics.complement_time_s += time.perf_counter() - started
        if remaining() > 0:
            self.learn(candidate, result.lane_solution().statuses, result, count, allow_complements=False)
        return count

    def learn_terminal(self, batch: SubproblemBatch, *, deadline: float) -> None:
        """Admit LP cores only after fresh root-domain replay of their own literals.

        The caller's proof used lane bounds, possibly on a smaller box, so it is
        no evidence for a global core. The complete conditions S are proved
        first; failure publishes nothing. Checked duals then propose a subset
        within theta's slack budget, which needs its own LP certificate, and
        otherwise the admitted complete row stands.
        """
        rows = torch.where(self._on_root_domain(batch))[0]
        if not self._terminal_cores or not rows.numel() or time.monotonic() >= deadline:
            return
        from act.back_end.solver.terminal_lp import solve_terminal_batch

        def replay(candidate: torch.Tensor, *, prefer_affine: bool) -> TerminalResult:
            """Rebuild the LP without conditioned bounds."""
            count = candidate.shape[0]
            proof_batch = SubproblemBatch.from_bounds(Bounds(
                self._root_bounds.lb.expand(count, *self._root_bounds.lb.shape[1:]),
                self._root_bounds.ub.expand(count, *self._root_bounds.ub.shape[1:]),
            ))
            _replace_literal_assignments(proof_batch, candidate, self.codec)
            started = time.perf_counter()
            result = solve_terminal_batch(
                self._net,
                Bounds(
                    proof_batch.lb.reshape(count, *self._root_bounds.lb.shape[1:]),
                    proof_batch.ub.reshape(count, *self._root_bounds.ub.shape[1:]),
                ),
                proof_batch.split_signs, deadline=deadline, prefer_affine=prefer_affine,
                lane_bases=self._terminal_lane_bases,
            )
            self.metrics.recheck_calls += 1
            self.metrics.recheck_row_passes += count
            self.metrics.recheck_time_s += time.perf_counter() - started
            return result

        literals = self._source_literals(batch.select(rows))
        source = replay(literals, prefer_affine=False)
        admitted = [lane for lane in range(literals.shape[0]) if source.has_lp_certificate(lane)]
        if not admitted:
            return
        self.metrics.terminal_admissions += len(admitted)
        retained = literals[admitted].clone()
        self.metrics.terminal_literals_original += int((retained != 0).sum())
        started = time.perf_counter()
        if self._coarsening_enabled:
            proposal = coarsen(
                retained, *self._terminal_costs(source, admitted, retained),
                theta=self._theta, delta_abs=self._delta_abs, delta_rel=self._delta_rel,
            )
            changed = torch.where((proposal != retained).any(1))[0]
            if changed.numel():
                self.metrics.cost_deleted_literals += int(
                    ((retained != 0) & (proposal == 0)).index_select(0, changed).sum()
                )
                self.metrics.mandatory_admissions += changed.numel()
                reduced = replay(proposal[changed], prefer_affine=True)
                passing = torch.tensor(
                    [reduced.has_lp_certificate(lane) for lane in range(changed.numel())],
                    dtype=torch.bool, device=changed.device,
                )
                accepted = changed[passing]
                retained[accepted] = proposal[accepted]
                self.metrics.coarsen_replay_fallbacks += int((~passing).sum())
        self.metrics.coarsen_time_s += time.perf_counter() - started
        self.metrics.terminal_literals_retained += int((retained != 0).sum())
        self.metrics.cores_inserted += self._insert_cores(retained)

    def _terminal_costs(
        self, result: TerminalResult, lanes: Sequence[int], literals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align checked LP charges with slots; an unattributable lane stays full.

        Float64 throughout: narrowing could round a tiny positive charge to an
        exact zero, which the deletion rule treats as free.
        """
        from act.back_end.solver.terminal_lp import terminal_literal_costs

        rows = max(len(result.certificates[lane]) for lane in lanes)
        costs = torch.full((*literals.shape[:1], rows, literals.shape[1]), torch.inf,
                           dtype=torch.float64, device=literals.device)
        slack = torch.ones(literals.shape[0], rows, dtype=torch.float64, device=literals.device)
        for index, lane in enumerate(lanes):
            charged = terminal_literal_costs(result, lane)
            if charged is None:
                continue
            keys, charges, slacks = charged
            slots = {literal: slot for slot, literal in enumerate(literals[index].tolist()) if literal}
            columns = [slots.get(self.codec.encode(*key), -1) for key in keys]
            if sorted(column for column in columns if column >= 0) != sorted(slots.values()):
                continue
            # A single infeasibility proof covers every spec row at once.
            costs[index, :] = 0
            values = torch.as_tensor(charges, device=costs.device)
            positions = torch.tensor(columns, dtype=torch.long, device=costs.device)
            costs[index, :, positions[positions >= 0]] = values[:, positions >= 0]
            slack[index] = torch.as_tensor(slacks, device=slack.device)
        return costs, slack

    def steer_scores(self, scores: BranchingScores) -> BranchingScores:
        """Bias finite branching candidates by core frequency, never add facts."""
        if not self._steer or self.library.core_count == 0:
            return scores
        for attr in ("per_layer", "intercept_per_layer", "babsr_per_layer"):
            values = getattr(scores, attr)
            if not values:
                continue
            biased = {}
            for lid, value in values.items():
                literals = self.library.pack(value.device)
                variables = literals[literals != 0].abs() - 1
                layers, neurons, _ = self.codec.decode_tensor(variables + 1)
                counts = torch.bincount(neurons[layers == lid], minlength=value.shape[1]).to(value)
                finite = torch.isfinite(value)
                scale = value.masked_fill(~finite, 0).abs().amax(1, keepdim=True).clamp(min=1)
                bias = counts[:value.shape[1]][None] / max(1, self.library.core_count)
                biased[lid] = torch.where(finite, value + scale * bias, value)
                self.metrics.steered_candidates += int((finite & (bias > 0)).sum())
            setattr(scores, attr, biased)
        return scores

    def _recheck(
        self,
        context: _ReplayContext,
        candidate: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        candidate_batch = context.batch.select(rows)
        source_signs = candidate_batch.split_signs
        if self._refresh_mode in _SPLIT_REFRESH_MODES and candidate_batch.incremental_alpha is not None:
            # A neuron the source interval made stable used slope 1 or 0, not its
            # dormant alpha. Those slopes remain global lower lines afterwards.
            source_bounds = slice_bounds_dict(context.bounds, rows)
            slopes = dict(candidate_batch.incremental_alpha)
            for lid in self.codec.layer_ids:
                alpha = slopes.get(lid)
                if isinstance(alpha, MixedAlpha):
                    alpha = alpha.materialize(context.m_specs)
                if isinstance(alpha, torch.Tensor) and lid in source_bounds:
                    lower = source_bounds[lid].lb.flatten(1).to(alpha)
                    upper = source_bounds[lid].ub.flatten(1).to(alpha)
                    if alpha.dim() == 3:
                        lower, upper = lower[:, None], upper[:, None]
                    slopes[lid] = torch.where(
                        lower >= 0, torch.ones_like(alpha),
                        torch.where(upper <= 0, torch.zeros_like(alpha), alpha),
                    )
            candidate_batch.incremental_alpha = slopes
        # The solver indexes split_signs[lid] for every incremental_eta layer, so a
        # layer whose literals were all coarsened away stays as an all-zero block.
        _replace_literal_assignments(candidate_batch, candidate, self.codec)
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
            base_reference_bounds=(
                slice_bounds_dict(context.base_bounds, rows)
                if context.base_bounds is not None else None
            ),
            refresh_mode=self._refresh_mode,
            lin_support_refinement=self._lin_support_refinement,
            refresh_rows_cap=self._refresh_rows_cap,
            forward_lin_max_perturbed=self._forward_lin_max_perturbed,
            phase_slope_signs=source_signs,
        )
        if replayed is None:
            return torch.zeros(
                int(rows.numel()), dtype=torch.bool, device=rows.device
            )
        replayed_slack, _ = replayed
        return is_certified(replayed_slack, self._out_kind)

    def _optional_recheck(
        self, context: _ReplayContext, candidate: torch.Tensor, rows: torch.Tensor,
    ) -> torch.Tensor:
        """A coarsening's extra one-literal attempt, beyond its cost budget."""
        self.metrics.optional_rechecks += rows.numel()
        return self._instrumented_recheck(context, candidate, rows)

    def _instrumented_recheck(
        self,
        context: _ReplayContext,
        candidate: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        recheck_started = time.perf_counter()
        passing = self._recheck(context, candidate, rows)
        if context.checked_candidates is not None and context.checked_passing is not None:
            context.checked_candidates[rows] = candidate
            context.checked_passing[rows] = passing
        self.metrics.recheck_time_s += time.perf_counter() - recheck_started
        self.metrics.recheck_calls += 1
        self.metrics.recheck_row_passes += int(rows.numel())
        return passing

    def before_split(
        self, pool: TopKBounding, unresolved: SubproblemBatch
    ) -> Tuple[SubproblemBatch, torch.Tensor]:
        original_rows = torch.arange(unresolved.batch_size, device=unresolved.lb.device)
        if not self._propagation_enabled:
            self.metrics.observe_frontier(len(pool) + unresolved.batch_size)
            return unresolved, original_rows
        if self._merge_enabled:
            self._remember_unresolved(unresolved)
        if not self._presplit_propagation_enabled:
            self.metrics.observe_frontier(len(pool) + unresolved.batch_size)
            return unresolved, original_rows
        if self.library.core_count == 0:
            return unresolved, original_rows
        # H3: one joint fixpoint over (pending, active); pending removals were
        # never bounded (pre-bound), active removals are parents spared a split
        # (pre-split).
        self.metrics.observe_frontier(len(pool) + unresolved.batch_size)
        propagation_groups: List[SubproblemBatch] = []
        revision = self.library.revision
        stale = pool.propagation_batch(revision)
        self.metrics.propagation_rows_skipped += len(pool) - (stale[0].numel() if stale else 0)
        if stale is not None:
            propagation_groups.append(stale[1])
        propagation_groups.append(unresolved)
        self.metrics.propagation_rows_scanned += sum(group.batch_size for group in propagation_groups)
        presplit_before = unresolved.batch_size
        propagate_started = time.perf_counter()
        propagation, kept_rows = propagate(
            propagation_groups,
            self.library,
            self.codec,
            core_chunk=self._propagate_core_chunk,
        )
        self.metrics.propagate_time_s += time.perf_counter() - propagate_started
        if stale is not None:
            pool.apply_propagation(stale[0], kept_rows[0], propagation[0], revision)
            self.metrics.prebound_discharged += (
                propagation_groups[0].batch_size - propagation[0].batch_size
            )
            self._note_propagation(propagation_groups[0], kept_rows[0], propagation[0])
        self._note_propagation(unresolved, kept_rows[-1], propagation[-1])
        unresolved = propagation[-1]
        self.metrics.presplit_discharged += presplit_before - unresolved.batch_size
        original_rows = original_rows.index_select(0, kept_rows[-1])
        merged_active, kept = self._merge(pool, unresolved)
        assert merged_active is not None
        self.metrics.observe_frontier(len(pool) + merged_active.batch_size)
        return merged_active, original_rows.index_select(0, kept)

    def after_children(self, generated: int, frontier_rows: int) -> None:
        self.metrics.generated_children += generated
        self.metrics.observe_frontier(frontier_rows)

    def metadata(self) -> Dict[str, int | float]:
        return {
            **self.metrics.metadata(),
            "climb_merge_enabled": self._merge_enabled,
            "climb_complement_bounding": self._complement_bounding,
            "climb_steer": self._steer,
            "climb_propagation_enabled": self._propagation_enabled,
            "climb_presplit_propagation_enabled": self._presplit_propagation_enabled,
            # Host-side merge memory, outside the planner's tensor accounting.
            "climb_merge_memo_rows": len(self._seen_cores) + len(self._bounded_unresolved) + len(self._fresh_rows),
        }


if __name__ == "__main__":
    from dataclasses import replace
    from tempfile import TemporaryDirectory
    from typing import Any
    from unittest.mock import patch

    from types import SimpleNamespace

    import numpy as np

    from act.back_end.bab.bound_refresh import (
        _backward_row_bounds, _inherited_support, _joint_halfspace_lower,
    )
    from act.back_end.bab.branching.bounding import DiverseTopKBounding
    from act.back_end.dual_tf.tf_forward import (
        _entry_lin_frame, _identical_forward_lanes, compute_forward_bounds,
        forward_frame_row_lower_bounds,
    )
    from act.back_end.bab.violation import (
        _forward_for_violation_check, clear_violation_check_module_cache,
    )
    from act.back_end.core import Layer
    from act.back_end.net_factory import LAYER_TESTING_SPECS, NetFactory
    from act.back_end.solver.milp_escalate import solve_milp
    from act.back_end.solver.solver_dual import DualSolver
    from act.back_end.solver import terminal_lp as terminal_module
    from act.back_end.solver.terminal_lp import solve_terminal_batch, terminal_literal_costs
    from act.back_end.verifier import gather_input_spec_layers, seed_from_input_specs
    from act.config.config import BackendConfig, DualConfig
    from act.front_end.specs import InputGeometry, OutputSpec
    from act.pipeline.verification.act2torch import ACTHardsigmoid, ACTHardswish
    from act.util.device_manager import initialize_device
    from act.util.stats import VerifyStatus

    initialize_device("cpu", "float64")

    def _alias_regression_net(alias: bool, unsafe: bool, *, identity: bool = False) -> Net:
        """Encode relu(x)-x, or a supported identity alias with repeated operands."""
        value = torch.ones(1, 1, device=get_default_device(), dtype=get_default_dtype())
        post = 1 if alias else 2
        right = post if identity else 0
        spec = OutputSpec(
            OutKind.UNSAFE_LINEAR if unsafe else OutKind.LINEAR_LE,
            c=-value if unsafe else value.flatten(),
            d=value.flatten() * -.5 if unsafe else value.new_tensor(.25),
        )
        layers = [
            Layer(0, "INPUT", {"shape": (1, 1), "dtype": str(value.dtype)}, [], [0]),
            Layer(1, "INPUT_SPEC", {"kind": "BOX", "lb": -value, "ub": value}, [0], [0]),
            Layer(2, "DENSE", {
                "weight": value, "bias": value.new_zeros(1),
                "in_features": 1, "out_features": 1,
            }, [0], [1]),
            Layer(3, "FLATTEN" if identity else "RELU", {}, [1], [post]),
            Layer(4, "SUB", {"x_vars": [post], "y_vars": [right]}, [post, right], [post + 1]),
            Layer(5, "ASSERT", spec.encode_linear(1, 1, value.device, value.dtype), [post + 1], []),
        ]
        return Net(
            layers,
            {0: [], 1: [0], 2: [1], 3: [2], 4: [3, 3 if identity else 1], 5: [4]},
            {0: [1], 1: [2] if identity else [2, 4], 2: [3],
             3: [4, 4] if identity else [4], 4: [5], 5: []},
        )

    # B3: proof-variable reuse must not erase the concrete x=-1, y=1 witness,
    # certify either property, or admit an empty global terminal core.
    for alias in (False, True):
        for unsafe in (False, True):
            clear_violation_check_module_cache()
            alias_net = _alias_regression_net(alias, unsafe)
            alias_box = Bounds(
                alias_net.layers[1].params["lb"], alias_net.layers[1].params["ub"],
            )
            concrete = _forward_for_violation_check(alias_net, alias_box.lb)
            assert alias_box.lb.item() == -1. and concrete.item() == 1.
            assert torch.equal(concrete, torch.relu(alias_box.lb) - alias_box.lb)
            for affine in (False, True):
                result = solve_terminal_batch(alias_net, alias_box, prefer_affine=affine)
                assert result.statuses == (
                    VerifyStatus.UNKNOWN if alias else VerifyStatus.FALSIFIED,
                ), (alias, unsafe, affine, result)
                assert not result.has_lp_certificate(0) and result.certificates == ((),)
                if alias:
                    assert result.reasons == ("unsupported_variable_alias:3",)
            alias_session = ClimbSession(
                BaBConfig(terminal_lp_enabled=True), alias_net,
                OutKind.UNSAFE_LINEAR if unsafe else OutKind.LINEAR_LE,
            )
            alias_lane = SubproblemBatch.from_bounds(alias_box)
            alias_session.learn_terminal(alias_lane, deadline=time.monotonic() + 30.)
            assert alias_session.library.core_count == 0
            assert alias_session.library.pack(alias_box.lb.device).shape[0] == 0
            if unsafe:
                exact = solve_milp(
                    alias_net, BaBConfig(milp_escalation=True),
                    deadline=time.monotonic() + 30.,
                )
                reason = exact.metadata["milp_escalation"]["reason"]
                assert "certificate_kind" not in exact.metadata
                if alias:
                    assert exact.status == VerifyStatus.UNKNOWN, exact
                    assert reason == "unsupported: unsupported_variable_alias:3"
                elif reason == "unavailable" or reason.startswith("license_unavailable"):
                    print(f"  SKIP  B3 real MILP control: {reason}")
                else:
                    assert exact.status == VerifyStatus.FALSIFIED, exact
                    assert exact.counterexample is not None
                    assert (_forward_for_violation_check(alias_net, exact.counterexample) >= .5).all()

    # INPUT_SPEC and FLATTEN may preserve variable IDs; SUB(x, x) is supported.
    clear_violation_check_module_cache()
    identity_net = _alias_regression_net(True, False, identity=True)
    identity_box = Bounds(
        identity_net.layers[1].params["lb"], identity_net.layers[1].params["ub"],
    )
    assert _forward_for_violation_check(identity_net, identity_box.lb).item() == 0.
    for affine in (False, True):
        identity_result = solve_terminal_batch(identity_net, identity_box, prefer_affine=affine)
        assert identity_result.statuses == (VerifyStatus.CERTIFIED,), identity_result
        assert identity_result.has_lp_certificate(0)
    clear_violation_check_module_cache()

    with TemporaryDirectory(prefix="act-climb-admission-") as directory:
        factory = NetFactory(BackendConfig.from_yaml().generation.net_factory,
                             output_dir=directory, write_manifest=False)
        net = factory.create_network("admission", LAYER_TESTING_SPECS["layer_testing_climb_admission"]())
        upper = torch.ones(1, 1, device=get_default_device(), dtype=get_default_dtype())
        lower = -upper
        terminal_spec = LAYER_TESTING_SPECS["layer_testing_climb_admission"]()
        terminal_spec["layers"][-1]["params"].update(c=[[1.], [-1.]], d=[.25, 2.])
        terminal_net = factory.create_network("terminal_admission", terminal_spec)
        terminal_config = BaBConfig(
            terminal_lp_enabled=True, climb_terminal_cores=True,
            climb_coarsening_enabled=True, climb_recheck_k=1,
        )
        terminal_session = ClimbSession(terminal_config, terminal_net, OutKind.LINEAR_LE)
        terminal_lane = SubproblemBatch.from_bounds(Bounds(lower, upper))
        terminal_lid = terminal_session.codec.layer_ids[0]
        terminal_lane.split_signs = {terminal_lid: -upper.reshape(1, 1, 1).expand(1, 2, 1)}
        terminal_deadline = time.monotonic() + 60
        with patch(
            "act.back_end.solver.terminal_lp.solve_terminal_batch",
            wraps=solve_terminal_batch,
        ) as terminal_spy:
            terminal_session.learn_terminal(terminal_lane, deadline=terminal_deadline)
        # A positive charge within no budget costs no second LP.
        assert terminal_spy.call_count == 1
        assert all("reference_bounds" not in call.kwargs for call in terminal_spy.call_args_list)
        assert torch.equal(terminal_session.library.pack(upper.device), torch.tensor([[-1]], device=upper.device))
        assert terminal_session.metrics.terminal_admissions == 1
        assert terminal_session.metrics.terminal_literals_original == 1
        assert terminal_session.metrics.terminal_literals_retained == 1
        for call in terminal_spy.call_args_list:
            assert torch.equal(call.args[1].lb, lower) and torch.equal(call.args[1].ub, upper)
        # The LP's own row duals price the phase: u0 * |Pi| for y <= d under z = 0.
        priced = solve_terminal_batch(terminal_net, Bounds(lower, upper), terminal_lane.split_signs,
                                      deadline=terminal_deadline, prefer_affine=False)
        priced_costs = terminal_literal_costs(priced, 0)
        assert priced_costs is not None and priced._export is not None
        keys, charges, margins = priced_costs
        assert keys == ((terminal_lid, 0, -1),) and charges.shape == (2, 1)
        assert charges[0, 0] == 1. and charges[1, 0] == 0. and margins.tolist() == [.25, 2.]
        reduced_tier = solve_terminal_batch(terminal_net, Bounds(lower, upper), terminal_lane.split_signs,
                                            deadline=terminal_deadline)
        assert reduced_tier.tiers == ("affine",) and terminal_literal_costs(reduced_tier, 0) is None
        # A phase row in an uneliminated LP must be attributable, or nothing is proposed.
        assert terminal_literal_costs(replace(priced, _export=None), 0) is None
        misread = replace(priced, _export=replace(priced._export, _phase_rows=((replace(
            priced._export._phase_rows[0][0], sign=1,
        ),),)))
        assert terminal_literal_costs(misread, 0) is None
        for implied, child_box in ((True, False), (False, True)):
            excluded = terminal_lane.select(torch.tensor([0], device=upper.device))
            if implied:
                # A propagation-written phase is still part of S, and needed here.
                excluded.decision_literals = torch.empty((1, 0), dtype=torch.long, device=upper.device)
            else:
                excluded.ub.zero_()
            learner = ClimbSession(terminal_config, terminal_net, OutKind.LINEAR_LE)
            with patch(
                "act.back_end.solver.terminal_lp.solve_terminal_batch",
                wraps=solve_terminal_batch,
            ) as excluded_spy:
                learner.learn_terminal(excluded, deadline=terminal_deadline)
            assert learner.library.core_count == int(implied)
            assert excluded_spy.call_count == (0 if child_box else 1)
        def terminal_spec_net(name: str, kind: str, c: List[List[float]], d: Sequence[float]) -> Net:
            """Vary only the output property of the one-neuron admission net."""
            spec = LAYER_TESTING_SPECS["layer_testing_climb_admission"]()
            spec["layers"][-1]["params"].update(kind=kind, c=c, d=list(d))
            return factory.create_network(name, spec)

        for theta, recheck_k, d, core in (
            (0., 0, (1.5,), [[-1]]),      # default never spends a positive charge
            (1., 0, (1.5,), [[]]),        # budget pays 1 from slack 1.5
            (1., 3, (1.5,), [[]]),        # K does not alter terminal admission
            (0., 3, (1.5,), [[-1]]),
            (1., 0, (.25,), [[-1]]),      # no slack would pay it, and no LP would pass
        ):
            budget_net = terminal_spec_net(f"terminal_budget_{theta}_{recheck_k}_{d[0]}", "LINEAR_LE", [[1.]], d)
            budgeted = ClimbSession(replace(terminal_config, climb_theta=theta, climb_recheck_k=recheck_k),
                                    budget_net, OutKind.LINEAR_LE)
            budgeted.learn_terminal(terminal_lane, deadline=terminal_deadline)
            assert budgeted.library.pack(upper.device).tolist() == core, (theta, recheck_k, d)
        # Default terminal admission: y >= -2 under z = 0 has Pi > 0, so the
        # literal is free, yet its removal still pays for a fresh LP.
        free_net = terminal_spec_net("terminal_free", "LINEAR_LE", [[-1.]], (2.,))
        for coarsening, calls, core in ((True, 2, [[]]), (False, 1, [[-1]])):
            free = ClimbSession(replace(terminal_config, climb_recheck_k=0, climb_coarsening_enabled=coarsening),
                                free_net, OutKind.LINEAR_LE)
            with patch("act.back_end.solver.terminal_lp.solve_terminal_batch",
                       wraps=solve_terminal_batch) as free_spy:
                free.learn_terminal(terminal_lane, deadline=terminal_deadline)
            assert free_spy.call_count == calls and free.library.pack(upper.device).tolist() == core
        # A failed or expired changed admission retains the admitted full row.
        real_terminal = solve_terminal_batch
        solves: List[int] = []

        def fail_second(*args: Any, **kwargs: Any) -> Any:
            solves.append(1)
            result = real_terminal(*args, **kwargs)
            if len(solves) == 2:
                return replace(result, statuses=(VerifyStatus.TIMEOUT,), certificates=((),))
            return result

        expired = ClimbSession(replace(terminal_config, climb_recheck_k=0), free_net, OutKind.LINEAR_LE)
        with patch("act.back_end.solver.terminal_lp.solve_terminal_batch", side_effect=fail_second):
            expired.learn_terminal(terminal_lane, deadline=terminal_deadline)
        assert expired.library.pack(upper.device).tolist() == [[-1]]
        assert expired.metrics.coarsen_replay_fallbacks == 1
        unpriced = ClimbSession(replace(terminal_config, climb_recheck_k=0), free_net, OutKind.LINEAR_LE)
        with patch("act.back_end.solver.terminal_lp.terminal_literal_costs", return_value=None):
            unpriced.learn_terminal(terminal_lane, deadline=terminal_deadline)
        assert unpriced.library.pack(upper.device).tolist() == [[-1]]
        # Farkas: the closed unsafe set is empty only under the inactive phase.
        unsafe_terminal = terminal_spec_net("terminal_unsafe", "UNSAFE_LINEAR", [[-1.]], (-.5,))
        farkas = solve_terminal_batch(unsafe_terminal, Bounds(lower, upper), terminal_lane.split_signs,
                                      deadline=terminal_deadline, prefer_affine=False)
        assert farkas.certificate_kinds == ("phase1_residual",)
        farkas_priced = terminal_literal_costs(farkas, 0)
        assert farkas_priced is not None
        farkas_keys, farkas_costs, farkas_slack = farkas_priced
        assert farkas_keys == ((terminal_lid, 0, -1),) and farkas_slack[0] > 0
        assert farkas_costs.shape == (1, 1) and farkas_costs[0, 0] > 0
        unsafe_session = ClimbSession(replace(terminal_config, climb_recheck_k=0), unsafe_terminal, OutKind.UNSAFE_LINEAR)
        unsafe_session.learn_terminal(terminal_lane, deadline=terminal_deadline)
        assert unsafe_session.library.pack(upper.device).tolist() == [[-1]]
        checked = solve_terminal_batch(
            terminal_net, Bounds(lower, upper), terminal_lane.split_signs,
            deadline=terminal_deadline,
        )
        assert checked.has_lp_certificate(0)
        for forged in (
            replace(checked, certificates=((),)),
            replace(checked, certificates=(tuple(
                replace(certificate, kind="milp_status_infeasible")
                for certificate in checked.certificates[0]
            ),)),
        ):
            assert not forged.has_lp_certificate(0)
            rejected = ClimbSession(terminal_config, terminal_net, OutKind.LINEAR_LE)
            with patch("act.back_end.solver.terminal_lp.solve_terminal_batch", return_value=forged):
                rejected.learn_terminal(terminal_lane, deadline=terminal_deadline)
            assert rejected.library.core_count == 0
        for changes, calls in (
            ({"climb_terminal_cores": False}, 0),
            ({"terminal_lp_enabled": False}, 0),
            ({"climb_coarsening_enabled": False}, 1),
            ({"climb_recheck_k": 9}, 1),
        ):
            gated = ClimbSession(replace(terminal_config, **changes), terminal_net, OutKind.LINEAR_LE)
            with patch("act.back_end.solver.terminal_lp.solve_terminal_batch", wraps=solve_terminal_batch) as gated_spy:
                gated.learn_terminal(terminal_lane, deadline=terminal_deadline)
            assert gated_spy.call_count == calls
        with patch("act.back_end.solver.terminal_lp.solve_terminal_batch") as expired_spy:
            terminal_session.learn_terminal(terminal_lane, deadline=time.monotonic() - 1)
        expired_spy.assert_not_called()
        empty_session = ClimbSession(terminal_config, net, OutKind.LINEAR_LE)
        empty_lane = SubproblemBatch.from_bounds(Bounds(lower, upper))
        empty_session.learn_terminal(empty_lane, deadline=terminal_deadline)
        assert empty_session.library.pack(upper.device).shape == (1, 0)
        bounds, frame, _ = compute_forward_bounds(net, lower, upper)
        for tile in (1, 2, 128):
            library = CoreLibrary(8)
            library.chunk = tile
            assert library.insert(torch.tensor([[2, 1, 1], [1, -2, 0]])) == 2
            assert library.resolved == 1
            assert torch.equal(library.pack(upper.device), torch.tensor([[1]], device=upper.device))
            assert library.insert(torch.tensor([[1, 3], [1, 0]])) == 0
            assert library.subsumed == 2
            assert library.insert(torch.tensor([[-1]])) == 1
            assert library.resolved == 2 and library.pack(upper.device).shape == (1, 0)
            try:
                library.insert(torch.tensor([[1, -1]]))
            except ValueError:
                pass
            else:
                raise AssertionError("complementary literals admitted")
            evict = CoreLibrary(1)
            evict.insert(torch.tensor([[1, 2], [3, 0]]))
            assert evict.evicted == 1 and torch.equal(evict.pack(upper.device), torch.tensor([[3]], device=upper.device))
            chronological = CoreLibrary(8)
            chronological.chunk = tile
            assert chronological.insert(torch.tensor([[1, 2], [1, 0], [1, 3]])) == 2
            assert chronological.subsumed == 2
            assert torch.equal(chronological.pack(upper.device), torch.tensor([[1]], device=upper.device))
        merge_config = BaBConfig(climb_enabled=True, climb_merge_enabled=True, climb_propagate_core_chunk=1)
        empty_active = SubproblemBatch(
            lb=lower[:0], ub=upper[:0],
            depths=torch.empty(0, dtype=torch.long, device=upper.device),
        )
        empty_active.split_signs = {2: upper.new_empty(0, 2, 1)}
        empty_library = CoreLibrary(8)
        empty_library.insert(torch.tensor([[-1]]))
        survivor = SubproblemBatch.from_bounds(Bounds(lower, upper))
        (pending_after, active_after), (_, active_rows) = propagate(
            (survivor, empty_active), empty_library, LiteralCodec.from_net(net),
            core_chunk=1,
        )
        assert pending_after.batch_size == 1
        assert active_after.batch_size == active_rows.numel() == 0
        def frontier_rows(literals: List[List[int]], **fields: Any) -> SubproblemBatch:
            """Root-box rows over one six-neuron layer, one spec row, no solve state."""
            count = len(literals)
            signs = upper.new_zeros(count, 1, 6)
            for row, literal_row in enumerate(literals):
                for literal in literal_row:
                    signs[row, 0, abs(literal) - 1] = 1. if literal > 0 else -1.
            state: Dict[str, Any] = {
                "depths": torch.full((count,), 3, dtype=torch.long),
                "node_id": torch.arange(1, count + 1),
                "parent_id": torch.arange(100, 100 + count),
                "decision_literals": torch.zeros(count, 1, dtype=torch.long),
                **fields,
            }
            return SubproblemBatch(lb=lower.expand(count, -1).clone(), ub=upper.expand(count, -1).clone(),
                                   split_signs={2: signs}, **state)

        def merge_fixture(config: BaBConfig = merge_config) -> ClimbSession:
            session = ClimbSession(config, net, OutKind.LINEAR_LE)
            session.codec = LiteralCodec((2,), (6,), (0,))
            return session

        def pool_of(rows: SubproblemBatch) -> TopKBounding:
            frontier = TopKBounding()
            frontier.push(rows)
            return frontier

        def row_sets(frontier: TopKBounding, codec: LiteralCodec) -> set[tuple[int, ...]]:
            rows = frontier.view_all()
            packed = pack_split_matrix(rows.split_signs, rows.batch_size, codec, device=rows.lb.device)
            return {tuple(row[row != 0].tolist()) for row in _canonical_rows(packed)}

        # The paper's four-row frontier: propagation, then two merge rounds.
        paper = merge_fixture()
        paper_rows = frontier_rows([[-1, 6], [1, -2], [1, 2, -3, -6], [1, 2, -3, 6]])
        paper_pool = pool_of(paper_rows)
        paper._insert_cores(torch.tensor([[-6], [3]]))
        no_active = paper_rows.select(torch.empty(0, dtype=torch.long))
        kept_after, kept_rows = paper.before_split(paper_pool, no_active)
        assert kept_after.batch_size == kept_rows.numel() == 0
        assert row_sets(paper_pool, paper.codec) == {(-3, 6)}
        assert paper.metrics.merged_pairs == 2 and paper.metrics.prebound_discharged == 1
        revived = paper_pool.pop()
        assert revived.warm_state_present is not None and not bool(revived.warm_state_present.any())
        assert revived.incremental_alpha is revived.incremental_eta is None
        assert revived.lower_bound is not None and bool(torch.isfinite(revived.lower_bound).all())
        assert revived.depths.item() == 1, "each merge removes one level"
        assert torch.equal(revived.lb, lower) and torch.equal(revived.ub, upper)
        # A cross-ancestor union keeps only the decisions both sides made.
        for left, right, parents, node, parent, decided in (
            ([1, 2], [-1, 2], [7, 7], 7, -1, [[0, 2]]),   # siblings revive their parent
            ([1, 2], [-1, 2], [7, 8], 1, -1, [[0, 2]]),   # different ancestors merge too
        ):
            pair = merge_fixture()
            pending = frontier_rows(
                [left, right], parent_id=torch.tensor(parents), depths=torch.tensor([3, 5]),
                decision_literals=torch.tensor([left, right]),
            )
            pair_pool = pool_of(pending)
            active = pair_pool.pop()
            # Waves without an unseen core never gather the frontier to merge.
            with patch.object(pair_pool, "checkpoint", side_effect=AssertionError("unneeded gather")):
                pair.before_split(pair_pool, active)
            assert pair._bounded_unresolved
            pair._insert_cores(torch.tensor([[-3, -2]]))  # This core writes nothing.
            active_after, selected = pair.before_split(pair_pool, active)
            assert active_after.batch_size == selected.numel() == 0 and len(pair_pool) == 1
            merged_row = pair_pool.pop()
            assert merged_row.node_id is not None and merged_row.parent_id is not None
            assert merged_row.node_id.item() == node and merged_row.parent_id.item() == parent
            assert merged_row.depths.item() == 2
            assert merged_row.decision_literals is not None
            assert merged_row.decision_literals.tolist() == decided
            assert merged_row.split_signs is not None
            assert merged_row.split_signs[2].flatten().tolist() == [0., 1., 0., 0., 0., 0.]
            # Re-offering a considered core arms nothing.
            pair._insert_cores(torch.tensor([[-3, -2]]))
            assert not pair._merge_armed
        # Different boxes, remaining literals, or a hash collision never merge.
        for case in ("box", "rest", "collision", "same_sign"):
            rows = frontier_rows({
                "box": [[1, 2], [-1, 2]], "rest": [[1, 2], [-1, 3]],
                "collision": [[1, 2], [-1, 3]], "same_sign": [[1, 2], [1, 2]],
            }[case])
            if case == "box":
                rows.lb[1] += .1
            refused = merge_fixture()
            refused_pool = pool_of(rows)
            refused._insert_cores(torch.tensor([[-4, -5]]))
            weights = (lambda count, device: torch.zeros(count, dtype=torch.long, device=device)
                       ) if case == "collision" else _merge_hash_weights
            with patch(f"{__name__}._merge_hash_weights", side_effect=weights):
                refused.before_split(refused_pool, rows.select(torch.empty(0, dtype=torch.long)))
            assert refused.metrics.merged_pairs == 0 and not refused._merge_armed
            assert len(refused_pool) == 2
        # Merging may not undo a split bounded unresolved, until propagation adds a condition.
        for writes in (False, True):
            guarded = merge_fixture()
            guarded._remember_unresolved(frontier_rows([[2, -4] if writes else [2]]))
            guarded_rows = frontier_rows([[1, 2], [-1, 2]])
            guarded_pool = pool_of(guarded_rows)
            guarded._insert_cores(torch.tensor([[4]]) if writes else torch.tensor([[-3, -2]]))
            guarded.before_split(guarded_pool, guarded_rows.select(torch.empty(0, dtype=torch.long)))
            assert guarded.metrics.merged_pairs == int(writes)
            assert row_sets(guarded_pool, guarded.codec) == ({(2, -4)} if writes else {(1, 2), (-1, 2)})
        # Consideration is permanent across subsumption and eviction.
        permanent = merge_fixture(replace(merge_config, climb_max_cores=1))
        permanent._insert_cores(torch.tensor([[5, 6]]))
        assert permanent._merge_armed
        permanent._merge_armed = False
        permanent._insert_cores(torch.tensor([[5]]))      # unseen though it subsumes
        assert permanent._merge_armed
        permanent._merge_armed = False
        permanent._insert_cores(torch.tensor([[3]]))      # unseen, and evicts a shorter tie
        permanent._merge_armed = False
        permanent._insert_cores(torch.tensor([[5, 6]]))   # subsumed/evicted, still seen
        permanent._insert_cores(torch.tensor([[5]]))
        permanent._insert_cores(torch.tensor([[3]]))
        assert not permanent._merge_armed
        resolvent = merge_fixture()
        resolvent._insert_cores(torch.tensor([[5, 6], [5, -6]]))
        assert (5,) in resolvent._seen_cores
        # An allocation failure leaves frontier, trigger and counts untouched.
        failing_merge = merge_fixture()
        failing_rows = frontier_rows([[1, 2], [-1, 2]])
        failing_pool = pool_of(failing_rows)
        failing_merge._insert_cores(torch.tensor([[-3, -2]]))
        no_rows = failing_rows.select(torch.empty(0, dtype=torch.long))
        with patch.object(failing_pool, "replace_all", side_effect=torch.cuda.OutOfMemoryError("merge")):
            try:
                failing_merge._merge(failing_pool, no_rows)
            except torch.cuda.OutOfMemoryError:
                pass
            else:
                raise AssertionError("injected failure did not fire")
        assert failing_merge._merge_armed and failing_merge.metrics.merged_pairs == 0
        assert row_sets(failing_pool, failing_merge.codec) == {(1, 2), (-1, 2)}
        failing_merge._merge(failing_pool, no_rows)
        assert row_sets(failing_pool, failing_merge.codec) == {(2,)} and not failing_merge._merge_armed
        # A wave rollback re-owes its merge and forgets its uncommitted memo,
        # while a core first seen after the checkpoint stays armed.
        rolled = merge_fixture()
        rolled_rows = frontier_rows([[1, 2], [-1, 2]])
        rolled_pool = pool_of(rolled_rows)
        rolled._insert_cores(torch.tensor([[-3, -2]]))
        frontier_state = rolled_pool.checkpoint()
        wave_state = rolled.checkpoint_frontier()
        rolled.before_split(rolled_pool, frontier_rows([[3]]))
        assert rolled.metrics.merged_pairs == 1 and not rolled._merge_armed
        assert rolled._bounded_unresolved
        rolled_pool.restore(frontier_state)
        rolled.restore_frontier(wave_state)
        assert rolled._merge_armed and rolled.metrics.merged_pairs == 0
        assert not rolled._bounded_unresolved and row_sets(rolled_pool, rolled.codec) == {(1, 2), (-1, 2)}
        # The total gate stops merge; the pre-split gate only defers it to pre-bound.
        for total, presplit, merged_at_split, merged_at_pop in (
            (False, True, 0, 0), (False, False, 0, 0), (True, True, 1, 1), (True, False, 0, 1),
        ):
            gated_merge = merge_fixture(replace(
                merge_config, climb_propagation_enabled=total, climb_presplit_propagation_enabled=presplit,
            ))
            gated_rows = frontier_rows([[1, 2], [-1, 2]])
            gated_pool = pool_of(gated_rows)
            gated_merge._insert_cores(torch.tensor([[-3, -2]]))
            gated_merge.before_split(gated_pool, gated_rows.select(torch.empty(0, dtype=torch.long)))
            assert gated_merge.metrics.merged_pairs == merged_at_split, (total, presplit)
            gated_merge.before_pop(gated_pool)
            assert gated_merge.metrics.merged_pairs == merged_at_pop, (total, presplit)
        # Pending and active rows pair up; every surviving row keeps a warm state.
        warm_rows = frontier_rows([[1, 2], [4]])
        warm_rows.incremental_alpha = {2: upper.new_ones(2, 6)}
        warm_rows.incremental_eta = {2: upper.new_zeros(2, 1, 6)}
        warm_rows.incremental_eta[2][0, 0, :2] = upper.new_tensor([.5, .25])
        warm_rows.warm_state_present = torch.ones(2, dtype=torch.bool)
        warm_pool = pool_of(warm_rows)
        mixed_merge = merge_fixture()
        mixed_merge._insert_cores(torch.tensor([[-3, -2]]))
        mixed_active, mixed_kept = mixed_merge.before_split(warm_pool, frontier_rows([[-1, 2], [5]]))
        assert mixed_active.batch_size == 1 and mixed_kept.tolist() == [1]
        assert mixed_merge.metrics.merged_pairs == 1 and len(warm_pool) == 2
        warm_view = warm_pool.view_all()
        assert warm_view.warm_state_present is not None
        # A union starts warm, without the multiplier of the condition it dropped.
        assert warm_view.warm_state_present.tolist() == [True, True]
        assert warm_view.incremental_eta[2][1, 0, :2].tolist() == [0., .25]
        disabled = ClimbSession(BaBConfig(climb_propagation_enabled=False), net, OutKind.LINEAR_LE)
        disabled.library.insert(torch.tensor([[1]]))
        pool = TopKBounding()
        with patch.object(pool, "view_all", side_effect=AssertionError("disabled propagation gathered")):
            disabled.before_pop(pool)
        ordered = CoreLibrary(8, track_complements=True)
        ordered.insert(torch.tensor([[2, -1]]))
        assert torch.equal(ordered.take_complements(1, upper.device), torch.tensor([[1, 2]], device=upper.device))
        ordered.insert(torch.tensor([[2, -1]]))
        assert ordered.take_complements(1, upper.device).shape[0] == 0
        ordered.insert(torch.tensor([[2, 1]]), suppress_complements=True)
        assert ordered.take_complements(8, upper.device).shape[0] == 0
        assert torch.equal(ordered.pack(upper.device), torch.tensor([[2]], device=upper.device))
        derived = CoreLibrary(8, track_complements=True)
        derived.insert(torch.tensor([[2, 1, 3], [2, -1, 3]]))
        assert torch.equal(derived.take_complements(1, upper.device), torch.tensor([[2, -3]], device=upper.device))
        try:
            SubproblemBatch.from_bounds(Bounds(lower, upper)).concat(
                SubproblemBatch(lb=lower, ub=upper, depths=torch.zeros(1, dtype=torch.long),
                                decision_literals=torch.empty((1, 0), dtype=torch.long)),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("mixed decision provenance silently lost")
        for total in (False, True):
            for presplit_gate in (False, True):
                gates = ClimbSession(BaBConfig(
                    climb_propagation_enabled=total,
                    climb_presplit_propagation_enabled=presplit_gate,
                ), net, OutKind.LINEAR_LE)
                gates.library.insert(torch.tensor([[-1]]))
                gated = SubproblemBatch.from_bounds(Bounds(lower, upper))
                gated.split_signs = {2: -upper[:, None]}
                pending = TopKBounding()
                pending.push(gated)
                gates.before_pop(pending)
                assert len(pending) == int(not total)
                pending = TopKBounding()
                pending.push(gated)
                remaining_batch, remaining_rows = gates.before_split(pending, gated)
                assert remaining_batch.batch_size == remaining_rows.numel() == int(not (total and presplit_gate))
                assert len(pending) == int(not (total and presplit_gate))
        steering = ClimbSession(BaBConfig(climb_steer=True), net, OutKind.LINEAR_LE)
        steering.codec = LiteralCodec((2,), (3,), (0,))
        untouched = BranchingScores(per_layer={2: upper.new_tensor([[.1, .2, -torch.inf]])})
        assert steering.steer_scores(untouched) is untouched
        steering.library.insert(torch.tensor([[1]]))
        assert untouched.per_layer is not None
        steered_tables = steering.steer_scores(untouched).per_layer
        assert steered_tables is not None
        steered = steered_tables[2]
        assert steered.argmax(1).item() == 0 and torch.isneginf(steered[0, 2])
        assert steering.metrics.steered_candidates == 1
        from unittest.mock import Mock

        probing = ClimbSession(BaBConfig(climb_complement_bounding=True), net, OutKind.LINEAR_LE)
        solve_complement = Mock(side_effect=TimeoutError("complement budget spent"))
        assert probing.bound_complements(solve_complement, lambda: 1., 1) == 0
        probing.library.insert(torch.tensor([[-1]]))
        assert probing.bound_complements(solve_complement, lambda: 0., 1) == 0
        assert probing.bound_complements(solve_complement, lambda: 1., 0) == 0
        assert solve_complement.call_count == 0
        assert probing.bound_complements(solve_complement, lambda: 1., 1) == 1
        assert probing.bound_complements(solve_complement, lambda: 1., 1) == 0
        assert solve_complement.call_count == 1 and probing.metrics.complement_row_passes == 1
        from types import SimpleNamespace
        from typing import cast

        for proved_rows, stops in ((0, True), (32, False)):
            gated = ClimbSession(BaBConfig(climb_merge_enabled=True), net, OutKind.LINEAR_LE)
            gated._merge_pending = torch.arange(_MERGE_PROBE_ROWS)
            gated._settle_merged_rows(
                cast(SubproblemBatch, SimpleNamespace(node_id=torch.arange(_MERGE_PROBE_ROWS))),
                [SolveStatus.UNSAT] * proved_rows + ["open"] * (_MERGE_PROBE_ROWS - proved_rows),
            )
            assert gated.metrics.merge_stopped == stops and gated._merge_enabled != stops
            assert gated.metrics.merge_proved_rows == proved_rows and not gated._merge_pending.numel()
        failing_probe = ClimbSession(BaBConfig(climb_complement_bounding=True), net, OutKind.LINEAR_LE)
        failing_probe.library.insert(torch.tensor([[-1]]))
        try:
            failing_probe.bound_complements(Mock(side_effect=RuntimeError("solver bug")), lambda: 1., 1)
        except RuntimeError:
            pass
        else:
            raise AssertionError("complement probing swallowed a solver failure")
        attempted_batch, cap = solve_complement.call_args.args
        assert cap == 1. and torch.equal(attempted_batch.lb, lower)
        assert torch.equal(attempted_batch.decision_literals, torch.tensor([[1]]))
        assert disabled.bound_complements(solve_complement, lambda: 1., 1) == 0
        from act.back_end.bab.bound_refresh import refresh_bounds

        for precision in ("float32", "float64"):
            initialize_device("cpu", precision)
            # B2, M2: opposite lanes share decision-slot layout but never bits.
            refresh_spec = LAYER_TESTING_SPECS["layer_testing_climb_admission"]()
            refresh_spec["layers"].insert(3, {
                "kind": "ADD", "params": {
                    "bias": torch.tensor([.125], dtype=get_default_dtype()),
                    "input_shape": [1, 1], "output_shape": [1, 1],
                },
                "preds": [2, 2],
            })
            refresh_net = factory.create_network(f"refresh_{precision}", refresh_spec)
            refresh_box = torch.ones(2, 1, dtype=get_default_dtype())
            base, _, _ = compute_forward_bounds(refresh_net, -refresh_box, refresh_box)
            phases = {2: torch.tensor([[[1.], [1.]], [[-1.], [-1.]]], dtype=get_default_dtype())}
            codec = LiteralCodec.from_net(refresh_net)
            packed = pack_split_matrix(phases, 2, codec, device=refresh_box.device)
            tracked = refresh_bounds(refresh_net, base, phases, literals=packed, codec=codec)
            plain = refresh_bounds(refresh_net, base, phases)
            assert plain.supports is None and tracked.supports is not None
            assert not refresh_bounds(refresh_net, base, phases, linear=True).linear_refined
            assert tracked.supports[2].lower.shape == (2, 1, 1)
            assert bool(tracked.supports[2].known.all())
            assert not bool(tracked.supports[0].lower.any())
            missing_decisions = refresh_bounds(
                refresh_net, base, phases, literals=packed[:, :0], codec=codec
            )
            assert missing_decisions.supports is not None
            assert not bool(missing_decisions.supports[2].known.any())
            for lid in plain.bounds:
                assert torch.equal(plain.bounds[lid].lb, tracked.bounds[lid].lb)
                assert torch.equal(plain.bounds[lid].ub, tracked.bounds[lid].ub)
            assert torch.equal(tracked.bounds[3].ub[1], refresh_box[1] * .125)
            released = refresh_bounds(refresh_net, base, {2: torch.zeros_like(phases[2])})
            assert bool((released.bounds[3].ub[1] > tracked.bounds[3].ub[1]).all())
            assert bool((released.bounds[2].lb[0] < tracked.bounds[2].lb[0]).all())
            inconsistent = phases[2].clone()
            inconsistent[:, 1] *= -1
            try:
                refresh_bounds(refresh_net, base, {2: inconsistent})
            except ValueError:
                pass
            else:
                raise AssertionError("spec-dependent phases were accepted")
            unsupported_spec = LAYER_TESTING_SPECS["layer_testing_climb_admission"]()
            unsupported_spec["layers"].insert(3, {"kind": "SIGMOID", "params": {}})
            unsupported = factory.create_network(f"unsupported_{precision}", unsupported_spec)
            unknown_base, _, _ = compute_forward_bounds(unsupported, -refresh_box, refresh_box)
            unknown = refresh_bounds(unsupported, unknown_base, phases, literals=packed, codec=codec)
            # The split-free fallback's only local fact is each ReLU's own phase,
            # so an additional operator's root-valid relaxation charges nothing.
            assert unknown.supports is not None
            assert unknown.bounds[2] is unknown_base[2]
            assert all(bool(s.known.all()) for s in unknown.supports.values())
            assert unknown.supports[2].lower[0].tolist() == [[True]]
            assert unknown.supports[2].upper[1].tolist() == [[True]]
            assert not bool(unknown.supports[0].lower.any())
            # Without the row's literals that local fact has unknown provenance.
            nameless = refresh_bounds(unsupported, unknown_base, phases, literals=packed[:, :0], codec=codec)
            assert nameless.supports is not None and not bool(nameless.supports[2].known.any())
            unknown_batch = SubproblemBatch.from_bounds(Bounds(-refresh_box, refresh_box), depth=1)
            unknown_batch.split_signs = {2: -torch.ones(2, 1, 1, dtype=refresh_box.dtype)}
            unknown_batch.incremental_alpha = {2: torch.ones(2, 1, 1, dtype=refresh_box.dtype)}
            assert unknown_batch.warm_state_present is not None
            unknown_batch.warm_state_present.fill_(True)
            unknown_batch.incremental_eta = {2: torch.zeros(2, 1, 1, dtype=refresh_box.dtype)}
            unknown_literals = pack_split_matrix(unknown_batch.split_signs, 2, codec, device=refresh_box.device)
            unknown_replay = certificate_replay(
                net=unsupported, batch=unknown_batch, reference_bounds=unknown_base,
                base_reference_bounds=unknown_base, c_rows=refresh_box, thresholds=refresh_box * 0,
                m_specs=1, out_kind=OutKind.LINEAR_LE, literals=unknown_literals, codec=codec,
                refresh_mode="split_refresh",
            )
            assert unknown_replay is not None and unknown_replay[1] is not None
            assert bool(torch.isfinite(unknown_replay[1]).all())
            # Only an inverted box means "no proof"; any other solver failure stays visible.
            from act.back_end.dual_tf.tf_mlp import DegenerateIntervalError
            from act.back_end.solver.solver_dual import DualSolver
            replay_args = dict(
                net=unsupported, batch=unknown_batch, reference_bounds=unknown_base,
                base_reference_bounds=unknown_base, c_rows=refresh_box, thresholds=refresh_box * 0,
                m_specs=1, out_kind=OutKind.LINEAR_LE, literals=unknown_literals, codec=codec,
                refresh_mode="split_refresh",
            )
            with patch.object(DualSolver, "compute_certified_bound", side_effect=DegenerateIntervalError("inverted")):
                assert certificate_replay(**replay_args) is None
            for failure in (RuntimeError("solver bug"), ValueError("shape bug"), torch.cuda.OutOfMemoryError("full")):
                with patch.object(DualSolver, "compute_certified_bound", side_effect=failure):
                    try:
                        certificate_replay(**replay_args)
                    except type(failure):
                        pass
                    else:
                        raise AssertionError(f"replay swallowed {failure!r}")
            assert nameless.supports is not None
            for layer_id, support in nameless.supports.items():
                nameless.supports[layer_id] = BoundSupport(
                    torch.zeros(*support.known.shape, 1, dtype=torch.bool), support.upper[..., :0].new_zeros(
                        *support.known.shape, 1), support.known,
                )
            unknown_costs = _supported_literal_costs(
                unknown_literals, codec, unknown_replay[0], unknown_batch.incremental_eta,
                {2: -torch.ones(2, 1, 1, dtype=refresh_box.dtype)}, unknown_base, unknown_base,
                unknown_batch.split_signs, nameless.supports, None,
            )
            assert bool(torch.isinf(unknown_costs).all()), "unknown support cannot fund deletion"
            token_net = factory.create_network(
                f"token_{precision}", LAYER_TESTING_SPECS["layer_testing_token_wise_dense"]()
            )
            token_lb = torch.arange(12, dtype=get_default_dtype()).reshape(2, 2, 3) / 10
            token_base, _, _ = compute_forward_bounds(token_net, token_lb, token_lb + .1)
            token_refresh = refresh_bounds(token_net, token_base, {})
            weight, bias = token_net.by_id[2].params["weight"], token_net.by_id[2].params["bias"]
            exact_lo = (token_lb @ weight.clamp(min=0).T
                        + (token_lb + .1) @ weight.clamp(max=0).T + bias).flatten(1)
            assert token_refresh.bounds[2].lb.shape == (2, 4)
            assert torch.allclose(token_refresh.bounds[2].lb, exact_lo)
        for precision in ("float32", "float64"):
            initialize_device("cpu", precision)
            dtype = get_default_dtype()
            # Default extraction deletes exactly the eligible zero-cost slots.
            def default_mask(
                rows: List[List[int]], cost: List[List[List[float]]], margin: List[List[float]],
            ) -> List[List[int]]:
                return coarsen(
                    torch.tensor(rows), torch.tensor(cost, dtype=dtype), torch.tensor(margin, dtype=dtype),
                    theta=0., delta_abs=1e-9, delta_rel=1e-7,
                ).tolist()

            tiny = torch.finfo(dtype).tiny
            assert default_mask([[1, 2]], [[[0., 0.]]], [[1.]]) == [[0, 0]]
            assert default_mask([[1, 2]], [[[0., 1.]]], [[1.]]) == [[0, 2]]
            assert default_mask([[1, 2]], [[[0., tiny]]], [[1.]]) == [[0, 2]], "near zero is not zero"
            assert default_mask([[1, 2], [3, 0], [0, 0]], [[[1., 0.]], [[0., 0.]], [[0., 0.]]],
                                [[1.], [1.], [1.]]) == [[1, 0], [0, 0], [0, 0]]
            for bad_cost, bad_slack in (
                (torch.nan, 1.), (torch.inf, 1.), (0., torch.nan), (0., torch.inf), (0., 0.), (0., 1e-10),
            ):
                assert default_mask([[1, 2]], [[[0., bad_cost]]], [[bad_slack]]) == [[1, 2]], (bad_cost, bad_slack)
            # Every spec row must agree, on a two-lane two-row tensor.
            assert default_mask([[1, 2], [3, 4]], [[[0., 0.], [0., 1.]], [[1., 0.], [0., 0.]]],
                                [[1., 1.], [1., 1.]]) == [[0, 2], [3, 0]]
            assert default_mask([[1, 2]], [[[0., 0.], [0., 0.]]], [[1., 1e-10]]) == [[1, 2]]
            # A negative budget cannot block exact zeros when theta buys nothing else.
            assert coarsen(torch.tensor([[1, 2]]), torch.tensor([[[0., 1.]]], dtype=dtype),
                           torch.tensor([[1e-8]], dtype=dtype), theta=.01, delta_abs=1e-9, delta_rel=0.,
                           ).tolist() == [[0, 2]]
            with patch(f"{__name__}.sorted", create=True, side_effect=AssertionError("default path sorted")):
                assert default_mask([[1]], [[[0.]]], [[1.]]) == [[0]]

            # The four-row frontier does not depend on the bound dtype.
            typed_rows = frontier_rows([[-1, 6], [1, -2], [1, 2, -3, -6], [1, 2, -3, 6]])
            assert typed_rows.split_signs is not None
            typed_rows = replace(typed_rows, lb=typed_rows.lb.to(dtype), ub=typed_rows.ub.to(dtype),
                                 split_signs={2: typed_rows.split_signs[2].to(dtype)})
            typed_session = merge_fixture()
            typed_pool = pool_of(typed_rows)
            typed_session._insert_cores(torch.tensor([[-6], [3]]))
            typed_session.before_split(typed_pool, typed_rows.select(torch.empty(0, dtype=torch.long)))
            assert row_sets(typed_pool, typed_session.codec) == {(-3, 6)}

            # Dense supports follow endpoint direction, drop zero weights, and a
            # post-ReLU lower endpoint of zero is global.
            weights = torch.tensor([[1., -1., 0.], [0., 0., 1.]], dtype=dtype)
            direction_spec = LAYER_TESTING_SPECS["layer_testing_climb_reuse"]()
            direction_spec["layers"] = direction_spec["layers"][:2] + [
                {"kind": "DENSE", "params": {"weight": torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=dtype),
                                             "bias": torch.zeros(3, dtype=dtype),
                                             "in_features": 2, "out_features": 3}},
                {"kind": "RELU", "params": {}},
                {"kind": "DENSE", "params": {"weight": weights, "bias": torch.zeros(2, dtype=dtype),
                                             "in_features": 3, "out_features": 2}},
                {"kind": "ASSERT", "params": {"kind": "LINEAR_LE", "c": [1., 1.], "d": 9.}},
            ]
            direction_net = factory.create_network(f"direction_{precision}", direction_spec)
            direction_box = torch.ones(1, 2, dtype=dtype)
            direction_base, _, _ = compute_forward_bounds(direction_net, -direction_box, direction_box)
            direction_codec = LiteralCodec.from_net(direction_net)
            direction_lid = direction_codec.layer_ids[0]
            direction_signs = {direction_lid: torch.tensor([[[-1., -1., 1.]]], dtype=dtype)}
            direction_literals = pack_split_matrix(direction_signs, 1, direction_codec, device=direction_box.device)
            direction = refresh_bounds(
                direction_net, direction_base, direction_signs,
                literals=direction_literals, codec=direction_codec,
            ).supports
            assert direction is not None
            output_lid = direction_lid + 1
            assert direction[output_lid].lower[0].tolist() == [[False, True, False], [False, False, False]]
            assert direction[output_lid].upper[0].tolist() == [[True, False, False], [False, False, False]]
            assert bool(direction[output_lid].known.all())
            # An unknown operand taints only outputs that actually read it.
            tainted = {direction_lid: BoundSupport(
                direction[direction_lid].lower, direction[direction_lid].upper,
                torch.tensor([[True, False, True]]),
            )}
            assert _inherited_support(
                direction_net.by_id[output_lid], [direction_lid], tainted, (1, 2), 3, direction_box.device,
            ).known.tolist() == [[False, True]]

            # The paper's running example, f = 2 z5 - z6 over [-1, 1]^2.
            toy_spec = LAYER_TESTING_SPECS["layer_testing_climb_reuse"]()
            toy_layers = (
                ([[1., 1.], [1., -1.]], [0., 0.]),
                ([[1., -1.], [1., -1.]], [-1., 1.]),
                ([[1., 1.], [1., 1.]], [-.25, -.5]),
            )
            toy_spec["layers"] = toy_spec["layers"][:2]
            for weight, bias in toy_layers:
                toy_spec["layers"] += [
                    {"kind": "DENSE", "params": {
                        "weight": torch.tensor(weight, dtype=dtype), "bias": torch.tensor(bias, dtype=dtype),
                        "in_features": 2, "out_features": 2,
                    }},
                    {"kind": "RELU", "params": {}},
                ]
            toy_spec["layers"].append({"kind": "ASSERT", "params": {"kind": "LINEAR_LE", "c": [-2., 1.], "d": .25}})
            toy = factory.create_network(f"paper_toy_{precision}", toy_spec)
            toy_codec = LiteralCodec.from_net(toy)
            toy_ids = toy_codec.layer_ids
            toy_box = torch.ones(1, 2, dtype=dtype)
            toy_base, _, _ = compute_forward_bounds(toy, -toy_box, toy_box)
            assert toy_base[toy_ids[2]].lb.flatten().tolist() == [-.25, -.5]
            assert toy_base[toy_ids[2]].ub.flatten().tolist() == [3.75, 3.5]
            assert toy_base[toy_ids[1]].lb.flatten().tolist() == [-3., -1.]

            def toy_cost(
                literals: List[int], eta_values: Dict[int, List[float]], nu_values: Dict[int, List[float]],
                reference: Dict[int, Bounds], supports: Dict[int, BoundSupport],
            ) -> torch.Tensor:
                packed_row = torch.tensor([literals])
                signs = {
                    lid: torch.zeros(1, 1, 2, dtype=dtype) for lid in toy_ids
                }
                for literal in literals:
                    lid, neuron, sign = toy_codec.decode(literal)
                    signs[lid][0, 0, neuron] = float(sign)
                return _supported_literal_costs(
                    packed_row, toy_codec, torch.ones(1, 1, dtype=dtype),
                    {lid: torch.tensor([[eta_values.get(lid, [0., 0.])]], dtype=dtype) for lid in toy_ids},
                    {lid: torch.tensor([[nu_values.get(lid, [0., 0.])]], dtype=dtype) for lid in toy_ids},
                    toy_base, reference, signs, supports, None,
                )[0, 0]

            def toy_supports(width: int, lower: Dict[tuple[int, int], int] | None = None,
                             upper: Dict[tuple[int, int], int] | None = None) -> Dict[int, BoundSupport]:
                masks = {}
                for lid in toy_ids:
                    low = torch.zeros(1, 2, width, dtype=torch.bool)
                    high = low.clone()
                    for (layer, neuron), slot in (lower or {}).items():
                        if layer == lid:
                            low[0, neuron, slot] = True
                    for (layer, neuron), slot in (upper or {}).items():
                        if layer == lid:
                            high[0, neuron, slot] = True
                    masks[lid] = BoundSupport(low, high, torch.ones(1, 2, dtype=torch.bool))
                return masks

            first, second, third = toy_ids
            # C1 = z1- & z6-: removing z6- invalidates z6 = 0, costing 7/2; z1- is free.
            assert toy_cost([-1, -6], {}, {third: [0., -1.]}, toy_base,
                            toy_supports(2, upper={(third, 1): 1})).tolist() == [0., 3.5]
            # C2 = z1+ & z2+ & z3+: eta 2 on z3+ (m = 3) and the z6 = zhat6 line
            # supported only by z3+ (eps = 1/2) give 13/2; the others are free.
            conditioned_toy = dict(toy_base)
            conditioned_toy[second] = Bounds(torch.tensor([[0., -1.]], dtype=dtype), toy_base[second].ub)
            conditioned_toy[third] = Bounds(torch.tensor([[-.25, 1.5]], dtype=dtype), toy_base[third].ub)
            assert toy_cost(
                [1, 2, 3], {second: [2., 0.]}, {third: [2., -1.], second: [1., 1.], first: [0., 0.]},
                conditioned_toy, toy_supports(3, lower={(third, 1): 2, (second, 0): 2}),
            ).tolist() == [0., 0., 6.5]
            # P1 = z1- & z6+: both multipliers zero, yet z6+ costs 1/2.
            assert toy_cost([-1, 6], {}, {third: [1., -1.]}, toy_base,
                            toy_supports(2, lower={(third, 1): 1})).tolist() == [0., .5]
            # A chord frozen on a narrower interval charges what supports it; the
            # same dependency on an unchanged interval has zero residual.
            narrowed = dict(toy_base)
            narrowed[third] = Bounds(toy_base[third].lb, torch.tensor([[3.75, 1.5]], dtype=dtype))
            chord_support = toy_supports(1, upper={(third, 1): 0})
            assert toy_cost([1], {}, {third: [0., -1.]}, narrowed, chord_support).tolist() == [
                .5 * (3.5 - 1.5) / 2.]
            assert toy_cost([1], {}, {third: [0., -1.]}, toy_base, chord_support).tolist() == [0.]
            # Positive coefficients use global lower lines, so they are never charged.
            assert toy_cost([1], {}, {third: [0., 4.]}, narrowed, chord_support).tolist() == [0.]
            # UNSAFE_LINEAR charges only the row that proved it.
            pinned_costs = _supported_literal_costs(
                torch.tensor([[-6]]), toy_codec, torch.ones(1, 2, dtype=dtype),
                {lid: torch.zeros(1, 2, 2, dtype=dtype) for lid in toy_ids},
                {third: torch.tensor([[[0., 0.], [0., -1.]]], dtype=dtype)},
                toy_base, toy_base,
                {**{lid: torch.zeros(1, 2, 2, dtype=dtype) for lid in toy_ids},
                 third: torch.tensor([[[0., -1.], [0., -1.]]], dtype=dtype)},
                toy_supports(1, upper={(third, 1): 0}), torch.tensor([1]),
            )
            assert pinned_costs.shape == (1, 1, 1) and pinned_costs.item() == 3.5

            # Real replay of C1: the z6- literal survives, the z1- literal is proved free,
            # and the bound itself is re-established with a fresh pass.
            c1_signs = {lid: torch.zeros(1, 1, 2, dtype=dtype) for lid in toy_ids}
            c1_signs[first][0, 0, 0] = -1.
            c1_signs[third][0, 0, 1] = -1.
            c1 = SubproblemBatch.from_bounds(Bounds(-toy_box, toy_box), depth=2)
            c1.split_signs = c1_signs
            c1.decision_literals = torch.tensor([[-1, -6]])
            c1.incremental_alpha = {lid: torch.zeros(1, 1, 2, dtype=dtype) for lid in toy_ids}
            c1.incremental_eta = {lid: torch.zeros(1, 1, 2, dtype=dtype) for lid in toy_ids}
            assert c1.warm_state_present is not None
            c1.warm_state_present.fill_(True)
            toy_spec_out = OutputSpec(OutKind.LINEAR_LE, c=torch.tensor([-2., 1.], dtype=dtype),
                                      d=torch.tensor(.25, dtype=dtype))
            c1_reference = refresh_bounds(toy, toy_base, c1_signs).bounds
            c1_result = DualSolver().solve_spec_batch(
                toy, c1_reference, toy_spec_out, optimize=True, dual_config=DualConfig(n_iters=0),
                input_shape=(2,), split_signs=c1_signs, incremental_alphas=c1.incremental_alpha,
                refresh_forward=False,
            )
            assert c1_result.solution.statuses == (SolveStatus.UNSAT,)
            c1_result = replace(c1_result, base_reference_bounds=toy_base)
            c1.incremental_alpha, c1.incremental_eta = c1_result.alpha_state, c1_result.eta_state
            c1_session = ClimbSession(BaBConfig(root_bounds_reuse="split_refresh"), toy, OutKind.LINEAR_LE)
            c1_session.learn(c1, c1_result.solution.statuses, c1_result, 1)
            assert c1_session.library.pack(toy_box.device).tolist() == [[-6]]
            assert c1_session.metrics.core_literals_original == 2
            assert c1_session.metrics.core_literals_retained == 1
            assert c1_session.metrics.coarsen_replay_fallbacks == 0
            # Terminal admission reaches the same proper subset through LP duals.
            terminal_c1 = ClimbSession(BaBConfig(terminal_lp_enabled=True), toy, OutKind.LINEAR_LE)
            terminal_c1.learn_terminal(c1, deadline=time.monotonic() + 60)
            terminal_core = terminal_c1.library.pack(toy_box.device).tolist()
            assert terminal_core == [[-6]], terminal_core
            assert terminal_c1.metrics.terminal_admissions == terminal_c1.metrics.mandatory_admissions == 1
            assert terminal_c1.metrics.cost_deleted_literals == 1
            assert terminal_c1.metrics.terminal_literals_original == 2
            assert terminal_c1.metrics.terminal_literals_retained == 1
        initialize_device("cpu", "float64")
        for alpha in (1., 0.):
            batch = SubproblemBatch.from_bounds(Bounds(lower, upper), depth=1)
            batch.split_signs = {2: -torch.ones(1, 1, 1, device=upper.device, dtype=upper.dtype)}
            result = DualSolver().solve_spec_batch(
                net, bounds, OutputSpec(OutKind.LINEAR_LE, c=-upper[0], d=upper.new_tensor(.25)),
                optimize=True, dual_config=DualConfig(n_iters=0), input_shape=(1,),
                split_signs=batch.split_signs, incremental_alphas={2: torch.full_like(batch.split_signs[2], alpha)},
                refresh_forward=False, forward_frame=frame,
            )
            assert result.solution.statuses == (SolveStatus.UNSAT,)
            batch.incremental_alpha, batch.incremental_eta = result.alpha_state, result.eta_state
            assert batch.warm_state_present is not None
            batch.warm_state_present.fill_(True)
            for coarsening in (True, False):
                for propagation in (True, False):
                    config = BaBConfig(
                        climb_enabled=True, solver_tier="dual_alpha_eta",
                        branching_method="babsr", bounding="depth_bound_blend",
                        climb_theta=.5, climb_recheck_k=0,
                        climb_coarsening_enabled=coarsening,
                        climb_propagation_enabled=propagation,
                    )
                    session = ClimbSession(config, net, OutKind.LINEAR_LE)
                    with (
                        patch.object(
                            DualSolver, "compute_certified_bound", autospec=True,
                            side_effect=DualSolver.compute_certified_bound,
                        ) as solve_spy,
                        patch(f"{__name__}._literal_costs", wraps=_literal_costs) as costs_spy,
                    ):
                        session.learn(batch, result.solution.statuses, result, 1)
                    # Literal -1 keeps its slope 0 whatever the dormant alpha, so y >= 0 survives.
                    expected = torch.tensor([[-1]] if not coarsening else [[]], dtype=torch.long)
                    assert torch.equal(session.library.pack(upper.device), expected.to(upper.device))
                    assert solve_spy.call_count == 1 + int(coarsening)
                    assert costs_spy.call_count == int(coarsening)
                    assert solve_spy.call_args_list[0].kwargs["return_nu_per_layer"] == coarsening
                    assert session.metadata()["climb_recheck_calls"] == int(coarsening)
                    assert session.metadata()["climb_coarsen_replay_fallbacks"] == 0
                    assert session.metrics.core_sources == 1
                    assert session.metrics.cost_deleted_literals == int(coarsening)
                    assert session.metrics.mandatory_admissions == int(coarsening)
                    assert session.metrics.optional_rechecks == 0

                    pool = TopKBounding()
                    pool.push(batch)
                    session.before_pop(pool)
                    assert len(pool) == int(not propagation)
                    pool = TopKBounding()
                    pool.push(batch)
                    unresolved, kept = session.before_split(pool, batch)
                    assert len(pool) == unresolved.batch_size == kept.numel() == int(not propagation)
                    if not propagation:
                        assert unresolved is batch
                        assert session.metrics.prebound_discharged == 0
                        assert session.metrics.presplit_discharged == 0
                        assert session.metrics.propagate_time_s == 0.0
                        assert session.metrics.propagation_rows_scanned == 0
                        assert session.metrics.unit_writes == 0

            # Disabling coarsening must still reject a failed first replay:
            # alpha=1 only certifies this reference under the inactive literal.
            if alpha == 1:
                invalid = batch.select(torch.zeros(1, device=upper.device, dtype=torch.long))
                invalid.split_signs = {2: torch.zeros_like(batch.split_signs[2])}
                invalid.decision_literals = torch.empty((1, 0), dtype=torch.long, device=upper.device)
                session = ClimbSession(config, net, OutKind.LINEAR_LE)
                with patch.object(
                    DualSolver, "compute_certified_bound", autospec=True,
                    side_effect=DualSolver.compute_certified_bound,
                ) as solve_spy:
                    session.learn(invalid, result.solution.statuses, result, 1)
                assert solve_spy.call_count == 1
                assert session.library.core_count == 0

        # A stale conditioned reference would certify deletion of the only
        # inactive literal. Rebuild from the mandatory split-free base instead.
        replay_batch = SubproblemBatch.from_bounds(Bounds(lower, upper), depth=1)
        replay_batch.split_signs = {2: -torch.ones_like(upper[:, None])}
        replay_batch.incremental_alpha = {2: torch.ones_like(upper[:, None])}
        replay_batch.incremental_eta = {2: torch.zeros_like(upper[:, None])}
        conditioned = refresh_bounds(net, bounds, replay_batch.split_signs).bounds
        replay_result = DualSolver().solve_spec_batch(
            net, conditioned, OutputSpec(OutKind.LINEAR_LE, c=-upper[0], d=upper.new_tensor(.25)),
            optimize=True, dual_config=DualConfig(n_iters=0), input_shape=(1,),
            split_signs=replay_batch.split_signs,
            incremental_alphas=replay_batch.incremental_alpha, refresh_forward=False,
        )
        replay_result = replace(replay_result, base_reference_bounds=bounds)
        replay_batch.incremental_alpha, replay_batch.incremental_eta = replay_result.alpha_state, replay_result.eta_state
        assert replay_batch.warm_state_present is not None
        replay_batch.warm_state_present.fill_(True)
        implied_only = replay_batch.select(torch.tensor([0], device=upper.device))
        implied_only.decision_literals = torch.empty((1, 0), dtype=torch.long, device=upper.device)
        implied_session = ClimbSession(BaBConfig(
            root_bounds_reuse="split_refresh", climb_coarsening_enabled=False,
        ), net, OutKind.LINEAR_LE)
        implied_session.learn(implied_only, replay_result.solution.statuses, replay_result, 1)
        # Propagation-written phases are source conditions of the proved region.
        assert implied_session.library.pack(upper.device).tolist() == [[-1]]
        objective_rows, threshold_rows = replay_result.c_rows, replay_result.thresholds
        assert objective_rows is not None and threshold_rows is not None

        def replay_test(
            candidate: SubproblemBatch,
            base_reference: Optional[Dict[int, Bounds]],
            thresholds: torch.Tensor,
            spec_count: int,
        ) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
            """Exercise replay's real proof path with explicit typed inputs."""
            assert objective_rows is not None
            return certificate_replay(
                net=net, batch=candidate, reference_bounds=conditioned,
                base_reference_bounds=base_reference, c_rows=objective_rows,
                thresholds=thresholds, m_specs=spec_count,
                out_kind=OutKind.LINEAR_LE, with_costs=False, refresh_mode="split_refresh",
            )

        full = replay_test(replay_batch, bounds, threshold_rows, 1)
        assert full is not None and bool(is_certified(full[0], OutKind.LINEAR_LE).all())
        deleted = replay_batch.select(torch.tensor([0], device=upper.device))
        deleted.split_signs = {2: torch.zeros_like(upper[:, None])}
        empty = replay_test(deleted, bounds, threshold_rows, 1)
        assert empty is not None and not bool(is_certified(empty[0], OutKind.LINEAR_LE).any())
        split_config = BaBConfig(
            climb_enabled=True, solver_tier="dual_alpha_eta",
            branching_method="babsr", bounding="depth_bound_blend",
            root_bounds_reuse="split_refresh", climb_theta=.5, climb_recheck_k=1,
        )
        validate_climb_config(split_config)
        split_session = ClimbSession(split_config, net, OutKind.LINEAR_LE)
        # The heuristic may be arbitrarily optimistic; admission cannot be: y <= .25
        # needs the phase, and slope 0 with its upper line max(0, u) cannot replace it.
        needed = replace(replay_batch, incremental_alpha={2: torch.ones_like(upper[:, None])})
        needed_result = DualSolver().solve_spec_batch(
            net, conditioned, OutputSpec(OutKind.LINEAR_LE, c=upper[0], d=upper.new_tensor(.25)),
            optimize=True, dual_config=DualConfig(n_iters=0), input_shape=(1,),
            split_signs=needed.split_signs, incremental_alphas=needed.incremental_alpha,
            refresh_forward=False,
        )
        assert needed_result.solution.statuses == (SolveStatus.UNSAT,)
        needed_result = replace(needed_result, base_reference_bounds=bounds)
        needed.incremental_alpha, needed.incremental_eta = needed_result.alpha_state, needed_result.eta_state
        with patch(f"{__name__}.coarsen", side_effect=lambda packed, *args, **kwargs: torch.zeros_like(packed)):
            split_session.learn(needed, needed_result.solution.statuses, needed_result, 1)
        assert torch.equal(split_session.library.pack(upper.device), torch.tensor([[-1]], device=upper.device))
        assert split_session.metrics.coarsen_replay_fallbacks == 1
        cheap = replace(replay_batch, incremental_alpha={2: torch.zeros_like(upper[:, None])})
        cheap_session = ClimbSession(split_config, net, OutKind.LINEAR_LE)
        with patch.object(DualSolver, "compute_certified_bound", autospec=True,
                          side_effect=DualSolver.compute_certified_bound) as replay_spy:
            cheap_session.learn(cheap, replay_result.solution.statuses, replay_result, 1)
        assert cheap_session.library.pack(upper.device).shape == (1, 0)
        assert replay_spy.call_count == 2, "one changed candidate must not be replayed twice"
        # A forward-only proof has no multipliers to replay: it keeps its complete row.
        forward_learned = ClimbSession(split_config, net, OutKind.LINEAR_LE)
        with patch.object(DualSolver, "compute_certified_bound", side_effect=AssertionError("must not replay")):
            forward_learned.learn(
                replay_batch, replay_result.solution.statuses,
                replace(replay_result, forward_only_certified=torch.ones(1, dtype=torch.bool)), 1,
            )
        assert forward_learned.metrics.forward_core_rows == 1
        assert forward_learned.library.core_count == 1
        # The same forward proof on a smaller input box is not a root-domain core.
        excluded = ClimbSession(split_config, net, OutKind.LINEAR_LE)
        with patch.object(DualSolver, "compute_certified_bound", side_effect=AssertionError("must not replay")):
            excluded.learn(
                replace(replay_batch, lb=replay_batch.lb / 2), replay_result.solution.statuses, replay_result, 1,
            )
        assert excluded.library.core_count == 0
        failed = ClimbSession(split_config, net, OutKind.LINEAR_LE)
        from act.back_end.dual_tf.tf_mlp import DegenerateIntervalError
        with patch.object(DualSolver, "compute_certified_bound", side_effect=DegenerateIntervalError("inverted")):
            failed.learn(replay_batch, replay_result.solution.statuses, replay_result, 1)
        assert failed.library.core_count == 0
        forward_batch = replace(replay_batch, split_signs={2: torch.zeros_like(upper[:, None])})
        forward_result = DualSolver().solve_spec_batch(
            net, bounds, OutputSpec(OutKind.LINEAR_LE, c=-upper[0], d=upper.new_tensor(.25)),
            optimize=True, dual_config=DualConfig(n_iters=0), input_shape=(1,),
            split_signs=forward_batch.split_signs,
            incremental_alphas={2: torch.ones_like(upper[:, None])},
            refresh_forward=False, forward_frame=frame,
        )
        assert forward_result.forward_only_certified is not None
        assert bool(forward_result.forward_only_certified.all())
        failed.learn(forward_batch, forward_result.solution.statuses,
                     replace(forward_result, base_reference_bounds=bounds), 1)
        assert failed.library.core_count == 1 and failed.metrics.forward_core_rows == 1
        for bad_base, bad_thresholds, bad_count in (
            (None, threshold_rows, 1),
            (bounds, upper.expand(2, 1), 1),
            (bounds, threshold_rows, 2),
        ):
            try:
                replay_test(replay_batch, bad_base, bad_thresholds, bad_count)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid replay provenance/objectives accepted")
        try:
            failed.learn(replay_batch, replay_result.solution.statuses,
                         replace(replay_result, base_reference_bounds=None), 1)
        except ValueError:
            pass
        else:
            raise AssertionError("missing base was silently replaced by conditioned bounds")

        # Independent phase enumeration/LP oracle for the linear path. The
        # numerical engine remains ACT's existing forward handlers; SciPy is
        # used only here to solve explicit two-variable phase polytopes.
        from itertools import product
        from scipy.optimize import linprog
        from act.back_end.bab.bound_refresh import _halfspace_lower
        from act.back_end.dual_tf.tf_forward import Frame

        for precision in ("float32", "float64"):
            initialize_device("cpu", precision)
            w1 = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], dtype=get_default_dtype())
            b1 = torch.tensor([0., .2, 0.], dtype=get_default_dtype())
            w2 = torch.tensor([[1., -1., 0.], [0., 0., 1.]], dtype=get_default_dtype())
            b2 = torch.tensor([.3, .1], dtype=get_default_dtype())
            linear_spec = LAYER_TESTING_SPECS["layer_testing_climb_reuse"]()
            linear_spec["layers"] = linear_spec["layers"][:2] + [
                {"kind": "DENSE", "params": {"weight": w1, "bias": b1, "in_features": 2, "out_features": 3}},
                {"kind": "RELU", "params": {}},
                {"kind": "DENSE", "params": {"weight": w2, "bias": b2, "in_features": 3, "out_features": 2}},
                {"kind": "RELU", "params": {}},
                {"kind": "ASSERT", "params": {"kind": "LINEAR_LE", "c": [1., 1.], "d": 2.}},
            ]
            linear_net = factory.create_network(f"linear_{precision}", linear_spec)
            lin_box = torch.ones(2, 2, dtype=get_default_dtype())
            lin_base, _, _ = compute_forward_bounds(linear_net, -lin_box, lin_box)
            lin_codec = LiteralCodec.from_net(linear_net)
            lin_phases = {3: torch.tensor(
                [[[1., 0., -1.], [1., 0., -1.]], [[-1., 0., 1.], [-1., 0., 1.]]],
                dtype=get_default_dtype(),
            )}
            lin_packed = pack_split_matrix(lin_phases, 2, lin_codec, device=lin_box.device)
            linear_results = []
            for refined in (True, False):
                r = refresh_bounds(
                    linear_net, lin_base, lin_phases, literals=lin_packed, codec=lin_codec,
                    linear=True, support_refinement=refined, rows_cap=1,
                )
                assert r.linear_refined and r.supports is not None
                assert r.bounds[3].lb[0, 1] > .19, "same-layer halfspace must imply an active phase"
                assert r.bounds[5].lb[0, 0] > .09
                if refined:
                    assert r.supports[3].lower[0, 1].tolist() == [True, False]
                else:
                    assert not r.supports[3].lower[0, 1].any()  # Left for replay to decide.
                linear_results.append(r)
            for lid in lin_base:
                assert torch.equal(linear_results[0].bounds[lid].lb, linear_results[1].bounds[lid].lb)
                assert torch.equal(linear_results[0].bounds[lid].ub, linear_results[1].bounds[lid].ub)
            dropped = {3: lin_phases[3].clone()}
            dropped[3][:, :, 0] = 0
            enlarged = refresh_bounds(linear_net, lin_base, dropped, linear=True)
            assert enlarged.bounds[3].lb[0, 1] < 0
            assert enlarged.bounds[5].lb[0, 0] < linear_results[0].bounds[5].lb[0, 0]
            wide_rows = refresh_bounds(linear_net, lin_base, lin_phases, linear=True, rows_cap=64)
            assert torch.equal(wide_rows.bounds[5].lb, linear_results[0].bounds[5].lb)
            # Unsupported conv/DAG never advertises linear refinement.
            conv_spec = LAYER_TESTING_SPECS["layer_testing_climb_admission"]()
            conv_spec["layers"][0]["params"]["shape"] = [1, 1, 2, 2]
            conv_spec["layers"].insert(2, {"kind": "CONV2D", "params": {
                "weight": torch.ones(1, 1, 1, 1, dtype=get_default_dtype()),
                "bias": torch.zeros(1, dtype=get_default_dtype()),
                "in_channels": 1, "out_channels": 1, "kernel_size": 1,
                "stride": 1, "padding": 0, "dilation": 1, "groups": 1,
                "input_shape": [1, 1, 2, 2], "output_shape": [1, 1, 2, 2],
            }})
            conv_spec["layers"][-1]["params"].update(c=[1., 1., 1., 1.], d=5.)
            conv_net = factory.create_network(f"conv_fallback_{precision}", conv_spec)
            conv_box = torch.ones(1, 1, 2, 2, dtype=get_default_dtype())
            conv_base, _, _ = compute_forward_bounds(conv_net, -conv_box, conv_box)
            assert not refresh_bounds(conv_net, conv_base, {3: torch.ones(1, 2, 4)}, linear=True).linear_refined
            feasible = infeasible = 0
            oracle_base = {lid: Bounds(box.lb[:1], box.ub[:1]) for lid, box in lin_base.items()}
            for assignments in product((-1., 1.), repeat=5):
                first = w1.new_tensor(assignments[:3])
                second = w1.new_tensor(assignments[3:])
                a2 = w2 @ (first.clamp(min=0)[:, None] * w1)
                c2 = w2 @ (first.clamp(min=0) * b1) + b2
                a_ub = torch.cat([-first[:, None] * w1, -second[:, None] * a2]).double().numpy()
                b_ub = torch.cat([first * b1, second * c2]).double().numpy()
                feasibility = linprog([0., 0.], A_ub=a_ub, b_ub=b_ub, bounds=[(-1., 1.)] * 2, method="highs")
                enum_phases = {3: first.reshape(1, 1, 3), 5: second.reshape(1, 1, 2)}
                with patch(
                    "act.back_end.bab.bound_refresh._backward_row_bounds", wraps=_backward_row_bounds,
                ) as second_layer_spy:
                    observed = refresh_bounds(linear_net, oracle_base, enum_phases, linear=True)
                assert second_layer_spy.call_count <= 1
                assert all(bool(torch.isfinite(box.lb).all() & torch.isfinite(box.ub).all()) for box in observed.bounds.values())
                if feasibility.status == 2:
                    infeasible += 1
                    continue
                assert feasibility.success
                feasible += 1
                for lid, a, intercept in ((3, w1, b1), (5, a2, c2)):
                    for j in range(a.shape[0]):
                        minimum = linprog(a[j].double().numpy(), A_ub=a_ub, b_ub=b_ub, bounds=[(-1., 1.)] * 2, method="highs")
                        maximum = linprog(-a[j].double().numpy(), A_ub=a_ub, b_ub=b_ub, bounds=[(-1., 1.)] * 2, method="highs")
                        assert minimum.success and maximum.success
                        tolerance = 2e-6 if precision == "float32" else 1e-10
                        assert observed.bounds[lid].lb[0, j] <= minimum.fun + float(intercept[j]) + tolerance
                        assert observed.bounds[lid].ub[0, j] >= -maximum.fun + float(intercept[j]) - tolerance
            assert feasible > 0 and infeasible > 0
            objectives = w1.new_tensor([
                [[.4, .2, -1.], [1., 1., 1.], [-1., 2., -.5]],
                [[1., -.3, .7], [-.5, 1., 2.], [1., -.5, -2.]],
            ])
            halfspace = w1.new_tensor([[1., -2., .5], [-1., .5, 2.]])
            offset = w1.new_tensor([-.3, .1])
            cube = torch.ones_like(halfspace)
            estimates = _halfspace_lower(
                objectives, w1.new_zeros(2, 3), Frame(Bounds(-cube, cube)),
                halfspace, offset,
            )
            with patch("act.back_end.bab.bound_refresh._lin_bound_lane_chunk", return_value=1):
                chunked_estimates = _halfspace_lower(
                    objectives, w1.new_zeros(2, 3), Frame(Bounds(-cube, cube)),
                    halfspace, offset,
                )
            assert torch.equal(estimates, chunked_estimates)
            with patch("act.back_end.bab.bound_refresh._lin_bound_lane_chunk", return_value=None):
                no_memory_bound = _halfspace_lower(
                    objectives, w1.new_zeros(2, 3), Frame(Bounds(-cube, cube)),
                    halfspace, offset,
                )
            assert bool(torch.isneginf(no_memory_bound).all())
            for lane in range(2):
                for row in range(3):
                    optimum = linprog(
                        objectives[lane, row].double().numpy(),
                        A_ub=-halfspace[lane:lane + 1].double().numpy(),
                        b_ub=offset[lane:lane + 1].double().numpy(),
                        bounds=[(-1., 1.)] * 3, method="highs",
                    )
                    assert optimum.success
                    error = optimum.fun - estimates[lane, row].item()
                    assert -1e-10 <= error <= (1e-4 if precision == "float32" else 1e-10)
            # Empty halfspaces cannot inject infinities or an empty root core.
            empty_bound = _halfspace_lower(
                w1[:1].unsqueeze(0), b1[:1].unsqueeze(0),
                Frame(Bounds(-lin_box[:1], lin_box[:1])),
                torch.zeros_like(lin_box[:1]), lin_box.new_tensor([-1.]),
            )
            assert bool(torch.isfinite(empty_bound).all()) and empty_bound.item() <= -1

        # Joint halfspaces, terminal operators, block LPs, shared forward lanes
        # and revision-aware propagation, each with its failure counterexample.
        for precision in ("float32", "float64"):
            initialize_device("cpu", precision)
            kind_dtype = get_default_dtype()
            tolerance = 1e-3 if precision == "float32" else 1e-10

            def values(*data: Any) -> torch.Tensor:
                return torch.tensor(*data, dtype=kind_dtype)

            # x+y over x>=0, y>=0 on the square needs both multipliers at once.
            square = Frame(Bounds(-torch.ones(1, 2, dtype=kind_dtype), torch.ones(1, 2, dtype=kind_dtype)))
            target, shift = values([[[1., 1.]]]), values([[0.]])
            axes, zero_offset = torch.eye(2, dtype=kind_dtype)[None], values([[0., 0.]])
            both = torch.ones(1, 2, dtype=torch.bool)
            joint_value, joint_witness = _joint_halfspace_lower(target, shift, square, axes, zero_offset, both)
            assert abs(joint_value.item()) <= tolerance and (joint_witness > 0).all()
            single_value = _halfspace_lower(target, shift, square, axes[:, 0], zero_offset[:, 0])
            assert single_value.item() <= -1
            # Deleting or flipping one condition reopens the true -1 minimum.
            for constraint, active in (
                (axes, values([[1., 0.]]).bool()),
                (values([[[1., 0.], [0., -1.]]]), both),
            ):
                assert _joint_halfspace_lower(target, shift, square, constraint, zero_offset, active)[0].item() <= -1
            warm_value, warm_witness = _joint_halfspace_lower(
                target, shift, square, axes, zero_offset, values([[1., 0.]]).bool(), both,
            )
            assert warm_value.item() <= -1 and warm_witness[0, 0, 1] == 0
            with patch("act.back_end.bab.bound_refresh._lin_bound_lane_chunk", return_value=None):
                assert torch.isneginf(_joint_halfspace_lower(target, shift, square, axes, zero_offset, both)[0]).all()

            # Independent LP oracle: K=2 lanes, M=2 rows, with lane chunking.
            generator = torch.Generator().manual_seed(7)
            oracle_a = torch.randn(2, 2, 4, generator=generator, dtype=kind_dtype)
            oracle_h = torch.randn(2, 5, 4, generator=generator, dtype=kind_dtype)
            oracle_g = torch.rand(2, 5, generator=generator, dtype=kind_dtype)
            oracle_frame = Frame(Bounds(-torch.ones(2, 4, dtype=kind_dtype), torch.ones(2, 4, dtype=kind_dtype)))
            oracle_active = torch.ones(2, 5, dtype=torch.bool)
            oracle_value = _joint_halfspace_lower(
                oracle_a, torch.zeros(2, 2, dtype=kind_dtype), oracle_frame, oracle_h, oracle_g, oracle_active,
            )[0]
            with patch("act.back_end.bab.bound_refresh._lin_bound_lane_chunk", return_value=1):
                chunked_value = _joint_halfspace_lower(
                    oracle_a, torch.zeros(2, 2, dtype=kind_dtype), oracle_frame, oracle_h, oracle_g, oracle_active,
                )[0]
            assert torch.equal(oracle_value, chunked_value)
            for lane in range(2):
                for row in range(2):
                    oracle_lp = linprog(
                        oracle_a[lane, row].double().numpy(), A_ub=-oracle_h[lane].double().numpy(),
                        b_ub=oracle_g[lane].double().numpy(), bounds=[(-1., 1.)] * 4, method="highs",
                    )
                    assert oracle_lp.success and oracle_value[lane, row].item() <= oracle_lp.fun + tolerance

            # A finite-p ball is the evaluated domain, not its enclosing box.
            ball = InputGeometry(
                torch.zeros(1, 1, 2, dtype=kind_dtype), torch.ones(1, 1, 1, dtype=kind_dtype) * .5,
                torch.ones(1, 1, dtype=torch.bool), 2.,
            )
            ball_frame = Frame(Bounds(-torch.ones(1, 2, dtype=kind_dtype) * .5, torch.ones(1, 2, dtype=kind_dtype) * .5), ball)
            ball_value = _joint_halfspace_lower(values([[[1., -1.]]]), shift, ball_frame, axes, zero_offset, both)[0]
            assert ball_value.item() <= -.5 + tolerance  # The true minimum at (0, .5).

            # A third neuron x is bounded by x+y>=0 and x-y>=0 only jointly.
            joint_w = values([[1., 1.], [1., -1.], [1., 0.]])
            joint_input = torch.ones(1, 2, dtype=kind_dtype)
            joint_net = Net([
                Layer(0, "INPUT", {"shape": (1, 2), "dtype": str(kind_dtype)}, [], [0, 1]),
                Layer(1, "INPUT_SPEC", {"kind": "BOX", "lb": -joint_input, "ub": joint_input}, [0, 1], [0, 1]),
                Layer(2, "DENSE", {
                    "weight": joint_w, "bias": torch.zeros(3, dtype=kind_dtype),
                    "in_features": 2, "out_features": 3,
                }, [0, 1], [2, 3, 4]),
                Layer(3, "RELU", {}, [2, 3, 4], [5, 6, 7]),
                Layer(4, "ASSERT", OutputSpec(
                    OutKind.LINEAR_LE, c=values([1., 0., 0.]), d=values(10.),
                ).encode_linear(1, 3, joint_w.device, kind_dtype), [5, 6, 7], []),
            ], {0: [], 1: [0], 2: [1], 3: [2], 4: [3]}, {0: [1], 1: [2], 2: [3], 3: [4], 4: []})
            joint_base, _, _ = compute_forward_bounds(joint_net, -joint_input, joint_input)
            joint_codec = LiteralCodec.from_net(joint_net)
            joint_signs = {3: values([[[1., 1., 0.]], [[1., 0., 0.]]])}
            lane_base = {lid: Bounds(box.lb.expand(2, *box.lb.shape[1:]), box.ub.expand(2, *box.ub.shape[1:]))
                         for lid, box in joint_base.items()}
            lane_literals = torch.tensor([[1, 2], [1, 0]])
            joint_results = [
                refresh_bounds(
                    joint_net, lane_base, joint_signs, literals=lane_literals, codec=joint_codec,
                    linear=True, rows_cap=rows_cap,
                )
                for rows_cap in (1, 64)
            ]
            for refreshed in joint_results:
                assert refreshed.linear_refined
                # One condition alone gives -1; the supergradient nears the LP's 0.
                assert -.1 <= refreshed.bounds[3].lb[0, 2].item() <= tolerance
                assert refreshed.bounds[3].lb[1, 2].item() <= -1 + tolerance
                assert refreshed.supports is not None
                assert refreshed.supports[3].lower[0, 2].tolist() == [True, True]
                # An endpoint left at the split-free reference rests on nothing.
                assert refreshed.supports[3].lower[1, 2].tolist() == [False, False]
            # A warm source witness must not smuggle in the deleted condition.
            warm_reduced = refresh_bounds(
                joint_net, joint_base, {3: values([[[1., 0., 0.]]])}, linear=True,
                warm_signs={3: values([[[1., 1., 0.]]])},
            )
            assert warm_reduced.bounds[3].lb[0, 2].item() <= -1 + tolerance

            # Executable terminal matrix: each operator proves and refutes through
            # its own equality rows; an incomplete encoding proves nothing.
            def operator_net(kind: str, params: Dict[str, Any], width: int, outputs: int,
                             row: List[float], threshold: float) -> Tuple[Net, Bounds]:
                inputs, results = list(range(width)), list(range(width, width + outputs))
                unit_box = torch.ones(1, width, dtype=kind_dtype)
                spec = OutputSpec(OutKind.UNSAFE_LINEAR, c=values([row]), d=values([threshold]))
                return Net([
                    Layer(0, "INPUT", {"shape": (1, width), "dtype": str(kind_dtype)}, [], inputs),
                    Layer(1, "INPUT_SPEC", {"kind": "BOX", "lb": -unit_box, "ub": unit_box}, inputs, inputs),
                    Layer(2, kind, params, inputs, results),
                    Layer(3, "ASSERT", spec.encode_linear(1, outputs, unit_box.device, kind_dtype), results, []),
                ], {0: [], 1: [0], 2: [1], 3: [2]}, {0: [1], 1: [2], 2: [3], 3: []}), Bounds(-unit_box, unit_box)

            def concat_net(row: List[float], threshold: float, axis: int) -> Tuple[Net, Bounds]:
                unit_box = torch.ones(1, 2, dtype=kind_dtype)
                spec = OutputSpec(OutKind.UNSAFE_LINEAR, c=values([row]), d=values([threshold]))
                return Net([
                    Layer(0, "INPUT", {"shape": (1, 2), "dtype": str(kind_dtype)}, [], [0, 1]),
                    Layer(1, "INPUT_SPEC", {"kind": "BOX", "lb": -unit_box, "ub": unit_box}, [0, 1], [0, 1]),
                    Layer(2, "DENSE", {
                        "weight": torch.eye(2, dtype=kind_dtype) * 2, "bias": torch.zeros(2, dtype=kind_dtype),
                        "in_features": 2, "out_features": 2,
                    }, [0, 1], [2, 3]),
                    Layer(3, "CONCAT", {"concat_dim": axis}, [2, 3, 0, 1], [4, 5, 6, 7]),
                    Layer(4, "ASSERT", spec.encode_linear(1, 4, unit_box.device, kind_dtype), [4, 5, 6, 7], []),
                ], {0: [], 1: [0], 2: [1], 3: [2, 1], 4: [3]}, {0: [1], 1: [2, 3], 2: [3], 3: [4], 4: []}), Bounds(-unit_box, unit_box)

            bn = {"A": values([2.]), "c": values([1.])}
            constant = {"value": values([1., 2.]), "output_shape": (1, 2)}
            total = {"axes": [1], "keepdims": 1, "input_shape": (1, 2), "output_shape": (1, 1)}
            average = {"dim": [1], "keepdim": 1, "input_shape": (1, 2), "output_shape": (1, 1)}
            spread_shape = {"shape": [1, 3], "input_shape": [1, 1], "output_shape": [1, 3]}
            terminal_matrix = [
                (operator_net("BN", bn, 1, 1, [1.], -2.), VerifyStatus.CERTIFIED),
                (operator_net("BN", bn, 1, 1, [1.], 0.), VerifyStatus.FALSIFIED),
                (operator_net("CONSTANT", constant, 2, 2, [1., 1.], 2.5), VerifyStatus.CERTIFIED),
                (operator_net("CONSTANT", constant, 2, 2, [1., 1.], 3.), VerifyStatus.FALSIFIED),
                (operator_net("REDUCE_SUM", total, 2, 1, [1.], -2.5), VerifyStatus.CERTIFIED),
                (operator_net("REDUCE_SUM", total, 2, 1, [1.], -1.5), VerifyStatus.FALSIFIED),
                (operator_net("MEAN", average, 2, 1, [1.], -1.5), VerifyStatus.CERTIFIED),
                (operator_net("MEAN", average, 2, 1, [1.], -.5), VerifyStatus.FALSIFIED),
                (operator_net("EXPAND", spread_shape, 1, 3, [1., 1., -1.], -1.5), VerifyStatus.CERTIFIED),
                (operator_net("EXPAND", spread_shape, 1, 3, [1., 1., -1.], -.5), VerifyStatus.FALSIFIED),
                (concat_net([1., 0., -1., 0.], -1.5, 1), VerifyStatus.CERTIFIED),
                (concat_net([1., 0., -1., 0.], -.5, 1), VerifyStatus.FALSIFIED),
                # Axis order is unknown, no shape names the batch axis, malformed
                # affine data, and an operator the source could not execute either.
                (concat_net([1., 0., -1., 0.], -1.5, -1), VerifyStatus.UNKNOWN),
                (operator_net("REDUCE_SUM", {"axes": [0], "keepdims": 0}, 2, 1, [1.], -2.5), VerifyStatus.UNKNOWN),
                (operator_net("BN", {"A": values([2., 2.]), "c": values([1.])}, 1, 1, [1.], 0.), VerifyStatus.UNKNOWN),
                (operator_net("SQUEEZE", {"dims": [1]}, 2, 2, [1., 1.], -3.), VerifyStatus.UNKNOWN),
            ]
            for (terminal_case, terminal_box_case), expected_status in terminal_matrix:
                clear_violation_check_module_cache()
                for affine in (False, True):
                    verdict = solve_terminal_batch(terminal_case, terminal_box_case, prefer_affine=affine)
                    assert verdict.statuses == (expected_status,), (terminal_case.layers[2].kind, verdict)
                    assert verdict.has_lp_certificate(0) == (expected_status == VerifyStatus.CERTIFIED)
            clear_violation_check_module_cache()

            # Block LP: lanes keep their own verdict and proof, with fewer solves.
            def phase_net(kind: str, bias: float = 0.) -> Tuple[Net, Bounds]:
                """relu(x+y+bias), relu(x-y): property unreachable under every phase."""
                unit_box = torch.ones(1, 2, dtype=kind_dtype)
                spec = (
                    OutputSpec(OutKind.UNSAFE_LINEAR, c=values([[1., 1.]]), d=values([-.5]))
                    if kind == OutKind.UNSAFE_LINEAR else OutputSpec(OutKind.LINEAR_LE, c=values([1., 1.]), d=values(9.))
                )
                return Net([
                    Layer(0, "INPUT", {"shape": (1, 2), "dtype": str(kind_dtype)}, [], [0, 1]),
                    Layer(1, "INPUT_SPEC", {"kind": "BOX", "lb": -unit_box, "ub": unit_box}, [0, 1], [0, 1]),
                    Layer(2, "DENSE", {
                        "weight": values([[1., 1.], [1., -1.]]), "bias": values([bias, 0.]),
                        "in_features": 2, "out_features": 2,
                    }, [0, 1], [2, 3]),
                    Layer(3, "RELU", {}, [2, 3], [4, 5]),
                    Layer(4, "ASSERT", spec.encode_linear(1, 2, unit_box.device, kind_dtype), [4, 5], []),
                ], {0: [], 1: [0], 2: [1], 3: [2], 4: [3]}, {0: [1], 1: [2], 2: [3], 3: [4], 4: []}), Bounds(-unit_box, unit_box)

            real_solve = terminal_module._solve
            all_phases = values([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
            for block_kind, solves_without_block in ((OutKind.UNSAFE_LINEAR, 1), (OutKind.LINEAR_LE, 1)):
                block_net, block_box = phase_net(block_kind)
                for lanes in (1, 2, 33):
                    block_signs = {3: all_phases[torch.arange(lanes) % 4]}
                    wave = Bounds(block_box.lb.expand(lanes, -1), block_box.ub.expand(lanes, -1))
                    with patch.object(terminal_module, "_solve", wraps=real_solve) as solve_spy:
                        batched = solve_terminal_batch(block_net, wave, block_signs)
                    with patch.object(terminal_module, "_BLOCK_MAX_LANES", 1), patch.object(
                        terminal_module, "_solve", wraps=real_solve,
                    ) as lane_spy:
                        per_lane = solve_terminal_batch(block_net, wave, block_signs)
                    assert batched.statuses == per_lane.statuses == (VerifyStatus.CERTIFIED,) * lanes
                    assert all(batched.has_lp_certificate(lane) for lane in range(lanes))
                    assert lane_spy.call_count == lanes * solves_without_block
                    if lanes == 33:
                        assert solve_spy.call_count == 1 + solves_without_block  # 32 + the last lane.
                    elif lanes == 2:
                        assert solve_spy.call_count == 1
                with patch.object(terminal_module, "_BLOCK_MAX_NONZEROS", 1), patch.object(
                    terminal_module, "_solve", wraps=real_solve,
                ) as capped_spy:
                    solve_terminal_batch(block_net, Bounds(block_box.lb.expand(2, -1), block_box.ub.expand(2, -1)), {3: all_phases[:2]})
                assert capped_spy.call_count == 2 * solves_without_block
            # An infeasible lane sets its elastic column, so its primal is never
            # exact. Its residual proof still holds, and the other lane is untouched.
            empty_net, empty_box = phase_net(OutKind.LINEAR_LE, bias=-3.)
            empty_signs = {3: values([[1., 1.], [-1., 1.]])}
            empty_wave = Bounds(empty_box.lb.expand(2, -1), empty_box.ub.expand(2, -1))
            block_outcomes: List[Dict[tuple[int, int], Any]] = []
            real_blocks = terminal_module._solve_blocks

            def recorded_blocks(*args: Any) -> Dict[tuple[int, int], Any]:
                block_outcomes.append(real_blocks(*args))
                return block_outcomes[-1]

            with patch.object(terminal_module, "_solve_blocks", side_effect=recorded_blocks):
                mixed = solve_terminal_batch(empty_net, empty_wave, empty_signs)
            assert not block_outcomes[0][0, 0].exact and block_outcomes[0][1, 0].exact
            assert mixed.statuses == (VerifyStatus.CERTIFIED, VerifyStatus.CERTIFIED)
            assert all(mixed.has_lp_certificate(lane) for lane in range(2))
            # Corrupted block duals never certify. A failed block only costs the
            # shared attempt, as each lane then proves itself.
            def corrupted(problem: Any, objective: np.ndarray, deadline: Optional[float]) -> Any:
                outcome = real_solve(problem, objective, deadline)
                if problem.le.shape[1] > 3:
                    outcome.ineqlin.marginals = np.abs(outcome.ineqlin.marginals) + 1.
                return outcome

            def block_failure(problem: Any, objective: np.ndarray, deadline: Optional[float]) -> Any:
                if problem.le.shape[1] > 3:
                    raise TimeoutError("block")
                return real_solve(problem, objective, deadline)

            ordinary_net, ordinary_box = phase_net(OutKind.LINEAR_LE)
            ordinary_wave = Bounds(ordinary_box.lb.expand(2, -1), ordinary_box.ub.expand(2, -1))
            with patch.object(terminal_module, "_solve", side_effect=corrupted):
                forged = solve_terminal_batch(ordinary_net, ordinary_wave, {3: all_phases[:2]})
            assert forged.statuses == (VerifyStatus.UNKNOWN, VerifyStatus.UNKNOWN)
            assert forged.certificates == ((), ())
            with patch.object(terminal_module, "_solve", side_effect=block_failure) as fault_spy:
                recovered = solve_terminal_batch(ordinary_net, ordinary_wave, {3: all_phases[:2]})
            assert fault_spy.call_count == 3
            assert recovered.statuses == (VerifyStatus.CERTIFIED, VerifyStatus.CERTIFIED)
            assert all(recovered.has_lp_certificate(lane) for lane in range(2))

            # Identical lanes run once; results and frames equal the batched pass.
            shared_net = factory.create_network(
                f"shared_{precision}", LAYER_TESTING_SPECS["layer_testing_climb_reuse"](),
            )
            shared_root = seed_from_input_specs(gather_input_spec_layers(shared_net))
            shared_lb = shared_root.lb.expand(3, *shared_root.lb.shape[1:])
            shared_ub = shared_root.ub.expand(3, *shared_root.ub.shape[1:])
            with patch("act.back_end.dual_tf.tf_forward._entry_lin_frame", wraps=_entry_lin_frame) as entry_spy:
                shared_bounds, shared_frame, _ = compute_forward_bounds(shared_net, shared_lb, shared_ub)
            assert [call.args[0].shape[0] for call in entry_spy.call_args_list] == [1]
            with patch("act.back_end.dual_tf.tf_forward._identical_forward_lanes", return_value=False):
                full_bounds, full_frame, _ = compute_forward_bounds(shared_net, shared_lb, shared_ub)
            assert shared_frame is not None and full_frame is not None
            assert all(torch.equal(shared_bounds[lid].lb, full_bounds[lid].lb)
                       and torch.equal(shared_bounds[lid].ub, full_bounds[lid].ub) for lid in full_bounds)
            frame_rows = torch.ones(6, shared_frame.lin.b_lb.shape[1], dtype=kind_dtype)
            assert torch.equal(
                forward_frame_row_lower_bounds(shared_frame, frame_rows, 2),
                forward_frame_row_lower_bounds(full_frame, frame_rows, 2),
            )
            shared_bounds[next(iter(shared_bounds))].lb[0] += 1
            assert not torch.equal(*shared_bounds[next(iter(shared_bounds))].lb[:2])  # No lane alias.
            child_ub = shared_ub.clone()
            child_ub[1] = (shared_lb[1] + shared_ub[1]) / 2
            with patch("act.back_end.dual_tf.tf_forward._entry_lin_frame", wraps=_entry_lin_frame) as entry_spy:
                child_bounds, _, _ = compute_forward_bounds(shared_net, shared_lb, child_ub)
                compute_forward_bounds(shared_net, shared_lb, shared_ub, alphas={})
            assert [call.args[0].shape[0] for call in entry_spy.call_args_list] == [3, 3]
            assert all(box.lb.shape[0] == 3 for box in child_bounds.values())
            # Equal boxes with different balls are different domains.
            for changed in ("center", "eps", "mask"):
                lanes_ball = {
                    "center": torch.zeros(2, 1, 2, dtype=kind_dtype), "eps": torch.ones(2, 1, 1, dtype=kind_dtype),
                    "mask": torch.ones(2, 1, dtype=torch.bool),
                }
                lanes_ball[changed] = lanes_ball[changed].clone()
                lanes_ball[changed][1] = False if changed == "mask" else 1 if changed == "center" else 2
                stub = SimpleNamespace(layers=[], input_geometry=InputGeometry(p=2., **lanes_ball))
                same_box = torch.zeros(2, 2, dtype=kind_dtype)
                assert not _identical_forward_lanes(stub, same_box, same_box)
            stub = SimpleNamespace(layers=[], input_geometry=None)
            assert _identical_forward_lanes(stub, same_box, same_box)

            # Covering resolution keeps exactly the unsafe assignments of its inputs.
            def covered(rows: torch.Tensor, variables: int) -> set[tuple[int, ...]]:
                cores = [{int(v) for v in row if v} for row in rows.tolist()]
                unsafe = set()
                for bits in range(1 << variables):
                    world = {(v + 1) if bits >> v & 1 else -(v + 1) for v in range(variables)}
                    if any(core <= world for core in cores):
                        unsafe.add(tuple(sorted(world)))
                return unsafe

            covering = CoreLibrary(64)
            covering.insert(torch.tensor([[1, 2, 0], [-2, 1, 3]]))
            assert covering.pack(torch.device("cpu")).tolist() == [[1, 2], [1, 3]]
            assert covering.resolved == covering.covering_resolved == 1
            plain = CoreLibrary(64)
            plain.insert(torch.tensor([[1, 2], [1, -2]]))
            assert plain.pack(torch.device("cpu")).tolist() == [[1]]
            assert plain.resolved == 1 and plain.covering_resolved == 0
            # Equal variables without an opposite literal, or two opposite ones, stay.
            for stuck in ([[1, 2], [1, 3]], [[1, 2], [-1, -2]]):
                held = CoreLibrary(64)
                held.insert(torch.tensor(stuck))
                assert held.resolved == 0 and held.core_count == 2
            resolution_rng = torch.Generator().manual_seed(11)
            for _ in range(40):
                count = int(torch.randint(2, 9, (1,), generator=resolution_rng))
                signs_table = torch.randint(-1, 2, (count, 5), generator=resolution_rng)
                raw = signs_table * torch.arange(1, 6)
                raw = raw[raw.ne(0).any(1)]
                if not raw.shape[0]:
                    continue
                resolved_library = CoreLibrary(4096)
                resolved_library.insert(raw[:raw.shape[0] // 2 + 1])
                resolved_library.insert(raw[raw.shape[0] // 2 + 1:])
                stored = resolved_library.pack(torch.device("cpu"))
                assert covered(stored, 5) == covered(raw, 5)
                # The stored antichain is at the fixed point: nothing more resolves.
                assert resolved_library._find_resolvents(
                    torch.ones(stored.shape[0], dtype=torch.bool, device=stored.device)
                ) is None

            # Without refinement a same-layer condition is a proposal, never charged.
            uncharged = refresh_bounds(
                joint_net, lane_base, joint_signs, literals=lane_literals, codec=joint_codec,
                linear=True, support_refinement=False,
            )
            assert uncharged.supports[3].lower[0, 2].tolist() == [False, False]
            assert torch.equal(uncharged.bounds[3].lb, joint_results[1].bounds[3].lb)
            # A later layer's rows tighten through each lane's own relaxations.
            with patch(
                "act.back_end.bab.bound_refresh._backward_row_bounds", wraps=_backward_row_bounds,
            ) as backward_spy:
                refresh_bounds(joint_net, lane_base, joint_signs, linear=True)
            assert backward_spy.call_count == 0  # No ReLU precedes this first layer.
            # Neuron 0 is clamped in both lanes, and the parent dense layer never queries.
            with patch(
                "act.back_end.bab.bound_refresh._joint_halfspace_lower", wraps=_joint_halfspace_lower,
            ) as joint_spy:
                refresh_bounds(joint_net, lane_base, joint_signs, linear=True)
            assert joint_spy.call_count == 2 and all(call.args[0].shape[1] == 2 for call in joint_spy.call_args_list)

            # Concrete hard activations match their exact function on any device.
            hard_input = torch.linspace(-5, 5, 41, dtype=kind_dtype)
            exact = ((hard_input + 3) / 6).clamp(0, 1)
            assert torch.equal(ACTHardsigmoid()(hard_input), exact)
            assert torch.equal(ACTHardswish()(hard_input), hard_input * exact)
            assert isinstance(ACTHardsigmoid(), torch.nn.Hardsigmoid)
            assert isinstance(ACTHardswish(), torch.nn.Hardswish)

            # Revision-aware propagation matches a full scan and skips unchanged rows.
            def revision_session() -> ClimbSession:
                return ClimbSession(BaBConfig(
                    climb_enabled=True, solver_tier="dual_alpha_eta", branching_method="babsr",
                    bounding="depth_bound_blend", climb_max_cores=3,
                ), joint_net, OutKind.LINEAR_LE)

            def revision_rows(literals: List[List[int]]) -> SubproblemBatch:
                rows = SubproblemBatch.from_bounds(Bounds(
                    -joint_input.expand(len(literals), -1), joint_input.expand(len(literals), -1),
                ))
                rows.split_signs = {3: torch.zeros(len(literals), 1, 3, dtype=kind_dtype)}
                _replace_literal_assignments(rows, torch.tensor(literals), joint_codec)
                rows.decision_literals = torch.tensor(literals)
                rows.incremental_alpha = {3: torch.zeros(len(literals), 1, 3, dtype=kind_dtype)}
                rows.incremental_eta = {3: torch.zeros(len(literals), 1, 3, dtype=kind_dtype)}
                rows.warm_state_present = torch.ones(len(literals), dtype=torch.bool)
                return rows

            for pool_kind in (TopKBounding, DiverseTopKBounding):
                incremental, full_scan = pool_kind(), pool_kind()
                incremental_session, full_session = revision_session(), revision_session()
                events = (
                    ("push", [[1, 0], [-1, 0]]), ("core", [[1, 2]]), ("pop", None), ("pop", None),
                    # An already-covered core keeps the revision and the rows checked.
                    ("core", [[1, 2]]), ("pop", None),
                    # A newly written phase activates an old core through a new one.
                    ("core", [[-2, 3]]), ("pop", None), ("push", [[-1, 2]]), ("pop", None),
                    # Eviction and resolution change the stored antichain.
                    ("core", [[-1, -2]]), ("pop", None),
                )
                for name, event in events:
                    if name == "push":
                        incremental.push(revision_rows(event))
                        full_scan.push(revision_rows(event))
                    elif name == "core":
                        incremental_session._insert_cores(torch.tensor(event))
                        full_session._insert_cores(torch.tensor(event))
                    else:
                        resident = full_scan._resident_batch()
                        if resident is not None:
                            full_scan._assign(replace(resident, propagation_revision=None))
                        incremental_session.before_pop(incremental)
                        full_session.before_pop(full_scan)
                        left, right = incremental._resident_batch(), full_scan._resident_batch()
                        assert (left is None) == (right is None)
                        if left is not None and right is not None:
                            assert left.batch_size == right.batch_size
                            assert torch.equal(left.decision_literals, right.decision_literals)
                            assert all(torch.equal(left.split_signs[lid], right.split_signs[lid]) for lid in right.split_signs)
                            assert left.incremental_alpha is not None and left.warm_state_present.all()
                assert incremental_session.metrics.propagation_rows_skipped > 0
                assert incremental_session.metrics.propagation_rows_scanned < full_session.metrics.propagation_rows_scanned
                assert incremental_session.metrics.unit_writes == full_session.metrics.unit_writes
                assert incremental_session.metrics.prebound_discharged == full_session.metrics.prebound_discharged

            # Chain: 1 forces -2, whose new core then forces -3 from the old row.
            chain_session, chain_pool = revision_session(), TopKBounding()
            chain_pool.push(revision_rows([[1, 0]]))
            chain_session._insert_cores(torch.tensor([[1, 2]]))
            first_revision = chain_session.library.revision
            chain_session.before_pop(chain_pool)
            chain_session._insert_cores(torch.tensor([[1, 2]]))
            assert chain_session.library.revision == first_revision
            with patch(f"{__name__}.propagate", side_effect=AssertionError("unchanged rows rescanned")):
                chain_session.before_pop(chain_pool)
            assert chain_session.metrics.propagation_rows_skipped == 1
            # A failed publication marks nothing, keeps every row, and rescans.
            chain_session._insert_cores(torch.tensor([[-2, 3]]))
            frontier_before = chain_pool.checkpoint()
            with patch.object(TopKBounding, "_assign", side_effect=torch.cuda.OutOfMemoryError("publish")):
                try:
                    chain_session.before_pop(chain_pool)
                except torch.cuda.OutOfMemoryError:
                    pass
                else:
                    raise AssertionError("publication fault was not raised")
            assert chain_pool.checkpoint().propagation_revision.tolist() == [first_revision]
            assert chain_pool.checkpoint().split_signs[3] is frontier_before.split_signs[3]
            chain_session.before_pop(chain_pool)
            assert chain_pool.view_all().split_signs[3][0, 0].tolist() == [1., -1., -1.]
            assert chain_pool.view_all().propagation_revision.tolist() == [chain_session.library.revision]
            # A merged row has new conditions, so it is stale despite its parents.
            merged_session, merged_pool = revision_session(), TopKBounding()
            merged_session._merge_enabled = True
            merged_pool.push(revision_rows([[1, 2], [1, -2]]))
            merged_session.library._cores = torch.empty((0, 0), dtype=torch.long)
            merged_session._consider_cores(torch.tensor([[3]]))
            merged_pool._assign(replace(merged_pool._resident_batch(), propagation_revision=torch.zeros(2, dtype=torch.long)))
            frontier_state = merged_session.checkpoint_frontier()
            merged_session._merge(merged_pool)
            assert merged_session.metrics.merge_calls == merged_session.metrics.merge_rows == 1
            assert merged_session.metrics.merge_literals == 1
            assert merged_pool.view_all().propagation_revision is None
            merged_session.restore_frontier(frontier_state)
            assert merged_session.metrics.merge_calls == merged_session.metrics.merge_rows == 0
        initialize_device("cpu", "float64")
