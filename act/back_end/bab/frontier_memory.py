# ===- act/back_end/bab/frontier_memory.py - Frontier Memory Policy ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
"""One CUDA budget for frontier storage, transfer, concatenation and waves."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from traceback import clear_frames
from typing import TYPE_CHECKING, Iterable, Iterator

import torch

if TYPE_CHECKING:
    from act.back_end.bab.branching.bounding import BoundingStrategy
    from act.back_end.bab.node import SubproblemBatch

PER_CLASS_ALPHA_MAX_BYTES = 1024 * 1024**2
WARM_STATE_MAX_BYTES_PER_LANE = 32 * 1024**2


def allow_per_class_alpha(estimated_bytes: int, *, enabled: bool) -> bool:
    """Keep the legacy policy unless the CUDA planner explicitly opts in."""
    return not enabled or estimated_bytes <= PER_CLASS_ALPHA_MAX_BYTES


def allow_warm_state(estimated_bytes: int, lanes: int, *, enabled: bool) -> bool:
    """Bound retained optimizer state without changing a domain or its proof."""
    return not enabled or estimated_bytes <= max(1, lanes) * WARM_STATE_MAX_BYTES_PER_LANE


@dataclass(frozen=True)
class MemoryBudget:
    """A snapshot in bytes; outside allocations are never treated as free."""

    limit: int
    outside: int
    frontier: int
    incoming: int
    inflight: int
    temporary: int

    @property
    def available(self) -> int:
        """Additional wave bytes that fit after all already-held storage."""
        return max(0, self.limit - self.outside - self.frontier
                   - self.incoming - self.inflight - self.temporary)

    @property
    def fits(self) -> bool:
        """Whether held storage and the next temporary allocation fit."""
        return (self.outside + self.frontier + self.incoming
                + self.inflight + self.temporary <= self.limit)


class FrontierMemoryPlanner:
    """Manage storage pressure while the BaB loop owns its single effective K.

    CPU and safety=0 never inspect allocator state or move tensor payloads.
    Host offload retains complete domains; only a last-resort eviction loses
    coverage, and its count must veto an empty-frontier certificate.
    """

    def __init__(
        self, device: torch.device, safety: float, offload: bool,
        estimated_lane_bytes: int,
    ) -> None:
        self.device = device
        self.active = device.type == "cuda" and safety > 0
        self.safety = safety
        self.offload = offload
        self.lane_bytes = max(1, estimated_lane_bytes)
        self._observed_lane_bytes = 0
        self.peak_allocated_bytes = 0
        self.peak_reserved_bytes = 0
        self.offloaded = False
        self.offload_count = 0
        self.warm_drop_count = 0
        self.evicted_nodes = 0
        self.oom_retries = 0
        self.oom_failures = 0
        self.wave_shrinks = 0
        self.alpha_waves = {"shared": 0, "per_row": 0}
        self.wave_oom = False

    @contextmanager
    def wave(self) -> Iterator[None]:
        """End the failed traceback's lifetime before the caller restores domains."""
        self.wave_oom = False
        try:
            yield
        except torch.cuda.OutOfMemoryError as error:
            if error.__traceback__ is not None:
                clear_frames(error.__traceback__)
                error.__traceback__ = None
            self.wave_oom = True

    def metadata(self) -> dict[str, int | bool]:
        """Report independent loss and mitigation counters."""
        self._record_peak()
        return {
            "frontier_memory_active": self.active,
            "frontier_offload_count": self.offload_count,
            "frontier_warm_drop_count": self.warm_drop_count,
            "frontier_memory_evicted_nodes": self.evicted_nodes,
            "any_dropped_frontier_memory": self.evicted_nodes > 0,
            "oom_retries": self.oom_retries,
            "oom_failures": self.oom_failures,
            "memory_wave_shrinks": self.wave_shrinks,
            "shared_alpha_waves": self.alpha_waves["shared"],
            "per_row_alpha_waves": self.alpha_waves["per_row"],
            "peak_cuda_allocated_bytes": self.peak_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_reserved_bytes,
        }

    def snapshot(
        self, pool: BoundingStrategy, incoming: SubproblemBatch | None = None,
        inflight: Iterable[torch.Tensor] = (), temporary: int = 0,
    ) -> MemoryBudget:
        """Count unique resident storage, including aliases across owners."""
        from act.back_end.bab.node import tensor_storage_bytes

        frontier = tuple(pool.resident_tensors())
        arriving = () if incoming is None else tuple(incoming.tensors())
        held = tuple(inflight)
        fbytes = tensor_storage_bytes(frontier, self.device)
        fibytes = tensor_storage_bytes(chain(frontier, arriving), self.device)
        owned = tensor_storage_bytes(chain(frontier, arriving, held), self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        reclaimable = free + torch.cuda.memory_reserved(self.device)
        limit = int(min(total * self.safety, reclaimable * .9))
        return MemoryBudget(
            limit, max(0, allocated - owned), fbytes, fibytes - fbytes,
            owned - fibytes, temporary,
        )

    def _host_available(self) -> int:
        """Leave half of currently available host RAM for unrelated work."""
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") * .5)

    def _relieve(self, pool: BoundingStrategy) -> None:
        """Offload complete storage if affordable, otherwise discard warm state."""
        if (self.offload and not self.offloaded and len(pool)
                and sum(t.numel() * t.element_size() for t in pool.resident_tensors())
                <= self._host_available()):
            pool.to("cpu")
            self.offloaded = True
            self.offload_count += 1
        else:
            before = pool.storage_bytes()
            pool.drop_warm_state()
            self.warm_drop_count += int(pool.storage_bytes() < before)

    def pop_temporary(self, pool: BoundingStrategy, device: str | None = None) -> int:
        """Bound selected/remaining gathers while the original pool stays live.

        Both gather outputs together contain the full logical payload. A second
        payload covers padding/selection scratch; ranking uses linear indices.
        Diverse soft selection additionally holds a dense pairwise distance map.
        """
        target = torch.device(device) if device is not None else self.device
        tensors = tuple(t for t in pool.resident_tensors() if t.device.type == target.type)
        if not tensors:
            return 0
        payload = sum(t.numel() * t.element_size() for t in tensors)
        temporary = 2 * payload + 64 * len(pool)
        if getattr(pool, "diversity_mode", None) == "soft":
            temporary += 4 * len(pool)**2 * tensors[0].element_size()
        return temporary

    def plan_wave(self, pool: BoundingStrategy, requested: int) -> int:
        """Relieve storage, shrink K, then evict only if one lane cannot fit."""
        if not self.active:
            return requested

        def wave_bytes() -> int:
            # Mixed alpha groups can occupy only a subset of the lanes.
            # A frontier-wide average undercounts a pop selecting those lanes.
            return self.lane_bytes + sum(
                (t.numel() // t.shape[0]) * t.element_size()
                for t in pool.resident_tensors()
                if t.device.type != self.device.type and t.shape[0]
            )

        budget = self.snapshot(pool, temporary=self.pop_temporary(pool))
        if budget.available < requested * wave_bytes():
            self._relieve(pool)
            budget = self.snapshot(pool, temporary=self.pop_temporary(pool))
        chosen = min(requested, budget.available // wave_bytes())
        if self.offloaded and chosen < requested:
            before = pool.storage_bytes()
            pool.drop_warm_state()
            self.warm_drop_count += int(pool.storage_bytes() < before)
            budget = self.snapshot(pool, temporary=self.pop_temporary(pool))
            chosen = min(requested, budget.available // wave_bytes())
        # A host frontier consumes no device budget; evicting it cannot help.
        while chosen < 1 and len(pool) and budget.frontier:
            dropped = pool.evict_to(len(pool) // 2)
            self.evicted_nodes += dropped
            if not dropped:
                break
            budget = self.snapshot(pool, temporary=self.pop_temporary(pool))
            chosen = min(requested, budget.available // wave_bytes())
        if self.offloaded:
            # CPU pop is transactional too. Reducing K cannot remove its
            # remaining-frontier copy, so evict only after warm state is cold.
            if (pool.storage_bytes() + self.pop_temporary(pool, "cpu")
                    > self._host_available()):
                before = pool.storage_bytes()
                pool.drop_warm_state()
                self.warm_drop_count += int(pool.storage_bytes() < before)
                while len(pool) and (
                    pool.storage_bytes() + self.pop_temporary(pool, "cpu")
                    > self._host_available()
                ):
                    self.evicted_nodes += pool.evict_to(len(pool) // 2)
                budget = self.snapshot(pool, temporary=self.pop_temporary(pool))
                chosen = min(requested, budget.available // wave_bytes())
        if chosen < requested:
            self.wave_shrinks += 1
        return max(0, int(chosen))

    def prepare_push(
        self, pool: BoundingStrategy, incoming: SubproblemBatch,
        inflight: Iterable[torch.Tensor] = (),
        *, allow_evict: bool = False,
    ) -> SubproblemBatch:
        """Reserve the actual pool concat peak before committing incoming rows."""
        if not self.active:
            return incoming
        from act.back_end.bab.node import tensor_storage_bytes
        from act.back_end.solver.solver_dual import _alpha_tree_leaves

        held = tuple(inflight)
        warm_bytes = tensor_storage_bytes(chain(
            _alpha_tree_leaves(incoming.incremental_alpha),
            _alpha_tree_leaves(incoming.incremental_eta),
        ))
        if not allow_warm_state(warm_bytes, incoming.batch_size, enabled=True):
            incoming = incoming.drop_warm_state()
            self.warm_drop_count += 1
        extra = pool.projected_push_bytes(incoming)
        budget = self.snapshot(pool, incoming, held, extra)
        if not budget.fits:
            self._relieve(pool)
            incoming = incoming.drop_warm_state()
            self.warm_drop_count += 1
        if self.offloaded:
            # Transport and CPU concatenation coexist with the source payload.
            host_peak = pool.storage_bytes() + tensor_storage_bytes(incoming.tensors())
            host_peak += pool.projected_push_bytes(incoming)
            if host_peak > self._host_available():
                pool.drop_warm_state()
                incoming = incoming.drop_warm_state()
                self.warm_drop_count += 1
                if not allow_evict:
                    raise torch.cuda.OutOfMemoryError("frontier host concatenation budget")
                while len(pool) and (
                    pool.storage_bytes() + tensor_storage_bytes(incoming.tensors())
                    + pool.projected_push_bytes(incoming) > self._host_available()
                ):
                    self.evicted_nodes += pool.evict_to(len(pool) // 2)
            if (pool.storage_bytes() + tensor_storage_bytes(incoming.tensors())
                    + pool.projected_push_bytes(incoming) > self._host_available()):
                raise torch.cuda.OutOfMemoryError("frontier host transport budget")
            return incoming.to("cpu")
        while len(pool):
            budget = self.snapshot(
                pool, incoming, held, pool.projected_push_bytes(incoming),
            )
            if budget.fits:
                break
            if not allow_evict:
                raise torch.cuda.OutOfMemoryError("frontier concatenation budget")
            self.evicted_nodes += pool.evict_to(len(pool) // 2)
        if not self.snapshot(
            pool, incoming, held, pool.projected_push_bytes(incoming),
        ).fits:
            raise torch.cuda.OutOfMemoryError("incoming frontier exceeds memory budget")
        return incoming

    def begin_bound(self) -> int:
        """Measure extra wave allocation, excluding persistent/root storage."""
        if not self.active:
            return 0
        self._record_peak()
        baseline = torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        return baseline

    def observe_wave(self, peak_bytes: int, lanes: int) -> None:
        """Replace the startup estimate with observed incremental bytes per lane."""
        if self.active:
            self._record_peak()
            measured = (max(0, peak_bytes) + max(1, lanes) - 1) // max(1, lanes)
            self._observed_lane_bytes = max(self._observed_lane_bytes, measured)
            self.lane_bytes = max(1, self._observed_lane_bytes)

    def _record_peak(self) -> None:
        """Preserve run peaks in metadata before resetting per-wave counters."""
        if self.active:
            self.peak_allocated_bytes = max(
                self.peak_allocated_bytes, torch.cuda.max_memory_allocated(self.device),
            )
            self.peak_reserved_bytes = max(
                self.peak_reserved_bytes, torch.cuda.max_memory_reserved(self.device),
            )
