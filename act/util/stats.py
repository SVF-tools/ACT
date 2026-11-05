#===- util.stats.py ----ACT Statistics -------------------------------#
#
#                 ACT: Abstract Constraints Transformer
#
# Copyright (C) <2025->  ACT Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Purpose:
#   Utilities for the Abstract Constraints Transformer (ACT), including memory usage tracking, verification result
#   summaries, and performance reporting across the verification framework.
#
#===----------------------------------------------------------------------===#

import torch
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional



# =============================================================================
# Verification Status and Result Types
# =============================================================================

class VerifyStatus(Enum):
    """Verification result status codes.
    
    Terminology Note:
        These differ from solver-level SAT/UNSAT terminology:
        - SolveStatus.SAT (constraint satisfied) -> VerifyStatus.FALSIFIED (counterexample found)
        - SolveStatus.UNSAT (constraint unsatisfied) -> VerifyStatus.CERTIFIED (property holds)
    
    Attributes:
        CERTIFIED: Property proven safe - no counterexample exists
        FALSIFIED: Property violated - valid counterexample found  
        UNKNOWN: Inconclusive result (e.g., approximation too coarse)
        TIMEOUT: Time limit exceeded before conclusion
        VERIFIER_ERROR: Verification failed due to an error in the verifier
        MODEL_INFER_FAILURE: Model inference failed on clean input (pre-verification check)
    """
    CERTIFIED = "certified"
    FALSIFIED = "falsified"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    VERIFIER_ERROR = "verifier_error"
    MODEL_INFER_FAILURE = "model_infer_failure"


@dataclass
class VerifyResult:
    """Verification result with optional counterexample.
    
    Attributes:
        status: The verification outcome (VerifyStatus enum)
        counterexample: Input tensor that violates the property (only if FALSIFIED)
        metadata: Solver/verification metadata (timing, nodes explored, etc.)
    """
    status: VerifyStatus
    counterexample: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass(frozen=True)
class SpecBatchResult:
    """Batched per-spec verification result — solver-agnostic.

    Vectorized counterpart of ``VerifyResult``: carries ``[B, M]`` margins
    and slack tensors plus a ``[B]`` certification bool. Produced by any
    solver path that evaluates B samples against M specs in one pass
    (currently DualSolver; interval/hz paths could adopt this contract for
    consistency). Use ``to_verify_results()`` to lower into the per-sample
    ``List[VerifyResult]`` API.

    Attributes:
        margins:     [B, M] - margin per (sample, spec) cell (interpretation
                              depends on the solver's certification form;
                              see ``source`` metadata in to_verify_results).
        slack:       [B, M] - margin - threshold, with `slack >= 0` denoting
                              certified cells.
        active_mask: [B, M] bool - which cells participated in certification.
        certified:   [B] bool - True iff every active cell has `slack >= 0`.
    """
    margins: torch.Tensor
    slack: torch.Tensor
    active_mask: torch.Tensor
    certified: torch.Tensor
    witness: Optional[torch.Tensor] = None  # [B, M, n_in] dual-minimizing inputs, when requested

    def __post_init__(self) -> None:
        assert self.margins.dim() == 2, f"margins must be [B, M], got {self.margins.shape}"
        assert self.slack.shape == self.margins.shape
        assert self.active_mask.shape == self.margins.shape
        assert self.active_mask.dtype == torch.bool
        assert self.certified.shape == (self.margins.shape[0],)
        assert self.certified.dtype == torch.bool

    @property
    def min_slack(self) -> torch.Tensor:
        """[B] - min slack over active cells, +inf for all-inactive samples."""
        inf_fill = torch.full_like(self.slack, float("inf"))
        masked = torch.where(self.active_mask, self.slack, inf_fill)
        return masked.min(dim=-1).values

    def to_verify_results(
        self, *, source: str = "dual_bound",
        counterexamples: Optional[torch.Tensor] = None,
        falsified: Optional[torch.Tensor] = None,
        counterexample_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List["VerifyResult"]:
        """Assemble results from certification and already-validated evidence.

        Callers own candidate generation and concrete validation. Certification
        takes precedence; absent evidence leaves an unresolved lane UNKNOWN.
        """
        results: List[VerifyResult] = []
        min_slack_vals = self.min_slack
        B = self.margins.shape[0]
        certified = self.certified.tolist()
        violations = falsified.tolist() if falsified is not None else [False] * B
        witnesses = counterexamples.detach().cpu() if counterexamples is not None else None
        for b in range(B):
            metadata = {
                "margins": self.margins[b].detach().cpu().tolist(),
                "slack": self.slack[b].detach().cpu().tolist(),
                "min_slack": float(min_slack_vals[b].item()),
                "source": source,
            }
            counterexample = None
            if certified[b]:
                status = VerifyStatus.CERTIFIED
            elif violations[b]:
                if witnesses is None:
                    raise ValueError("FALSIFIED requires a concrete counterexample")
                status = VerifyStatus.FALSIFIED
                counterexample = witnesses[b].clone()
                if counterexample_metadata is not None:
                    metadata.update(counterexample_metadata[b])
            else:
                status = VerifyStatus.UNKNOWN
            results.append(VerifyResult(status, counterexample, metadata))
        return results
