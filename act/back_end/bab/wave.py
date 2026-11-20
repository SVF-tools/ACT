# ===- act/back_end/bab/wave.py - BaB Wave Results ----------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#

"""Bind wave results to source lanes and align evidence for branching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from act.back_end.bab.node import SubproblemBatch, slice_branching_state
from act.back_end.core import Bounds
from act.back_end.solver.solver_base import BatchLPSolution
from act.back_end.solver.solver_dual import DualBatchResult


@dataclass(frozen=True)
class _BranchInputs:
    """A processed sub-batch and branching evidence in the same lane order."""

    batch: SubproblemBatch
    bounds: Optional[Dict[int, Bounds]] = None
    sensitivities: Optional[Dict[int, torch.Tensor]] = None
    witness_input: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class _DualWaveEvidence:
    """Keep dual-specific evidence and its packed-row layout together."""

    result: DualBatchResult

    def for_branching(
        self, batch: SubproblemBatch, source_rows: torch.Tensor,
        source_size: int, *, include_witness: bool,
    ) -> _BranchInputs:
        bounds, sensitivities = slice_branching_state(
            self.result.intermediate_bounds(), self.result.neuron_sensitivities(),
            source_rows, source_size,
        )
        witness = self.result.lane_witness_inputs() if include_witness else None
        if witness is not None:
            witness = witness.index_select(0, source_rows.to(witness.device))
        return _BranchInputs(batch, bounds, sensitivities, witness)


@dataclass(frozen=True)
class _BoundWave:
    """Bind a lane solution to its source batch, retaining backend evidence.

    LP modeling stays in analyze/export/solve. Dual tiers share the same
    evidence type. Objects are shared and released at the existing cleanup
    points; the source batch's lane order must remain unchanged.
    """

    batch: SubproblemBatch
    solution: BatchLPSolution
    dual: Optional[_DualWaveEvidence] = None

    def __post_init__(self) -> None:
        if len(self.solution.statuses) != self.batch.batch_size:
            raise ValueError("bound result does not match the wave's lane count")

    @classmethod
    def from_lp(cls, batch: SubproblemBatch, result: BatchLPSolution) -> "_BoundWave":
        return cls(batch, result)

    @classmethod
    def from_dual(cls, batch: SubproblemBatch, result: DualBatchResult) -> "_BoundWave":
        return cls(batch, result.lane_solution(), _DualWaveEvidence(result))

    def for_branching(
        self, batch: SubproblemBatch, source_rows: torch.Tensor,
        *, include_witness: bool = False,
    ) -> _BranchInputs:
        # Preserve lower bounds and CLIMB changes on the caller's selected batch.
        if source_rows.ndim != 1 or source_rows.numel() != batch.batch_size:
            raise ValueError("branching rows do not match the selected batch")
        if self.dual is None:
            return _BranchInputs(batch)
        return self.dual.for_branching(
            batch, source_rows, self.batch.batch_size,
            include_witness=include_witness,
        )
