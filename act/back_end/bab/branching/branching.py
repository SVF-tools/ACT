# ===- act/back_end/bab/branching/branching.py - Branching ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Branching strategies for Branch-and-Bound.
#
#   Contains the abstract base class ``BranchingStrategy`` and the
#   ``RandomBranching`` baseline implementation.
#
#   A branching strategy decides *which dimension (or neuron) to split*
#   for each subproblem in a batch.  Two paradigms are supported:
#
#     1. Input splitting  — bisect the input domain along a dimension.
#     2. Neuron splitting — fix an unstable ReLU's activation phase.
#
#   All tensor shapes follow the (N, D) batch convention so that a
#   future batch-parallel BaB engine can evaluate scores for many
#   subproblems in one forward pass.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from act.back_end.bab.node import SubproblemBatch
from act.back_end.core import Net


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BranchingStrategy(ABC):
    """Abstract branching strategy for Branch-and-Bound.

    Lifecycle (called by the BaB engine per iteration)::

        scores     = strategy.compute_scores(batch, net, unstable_mask)
        split_dims = strategy.select(scores)
        left, right = split_subproblems(batch, split_dims)

    Subclass contract
    ~~~~~~~~~~~~~~~~~
    * ``compute_scores`` **must** return ``(N, D)`` float tensor.
    * Dimensions that must not be split should receive score ``-inf``
      (or ``0`` when ``select`` uses ``argmax``).
    * ``select`` defaults to row-wise ``argmax``; override for
      stochastic or top-k selection.
    """

    @abstractmethod
    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score every candidate split dimension.

        Args:
            batch:          Current subproblems ``(N, D)``.
            net:            ACT network (for layer structure / neuron info).
            unstable_mask:  ``(D,)`` or ``(N, D)`` bool tensor marking
                            neurons eligible for splitting.
                            ``None`` ⇒ all dimensions are candidates
                            (input-split mode).

        Returns:
            ``(N, D)`` float tensor — higher score = better split.
        """
        ...

    def select(self, scores: torch.Tensor) -> torch.Tensor:
        """Pick one split dimension per subproblem.

        Default implementation: deterministic ``argmax`` per row.
        Override for stochastic or multi-split selection.

        Args:
            scores: ``(N, D)`` branching scores.

        Returns:
            ``(N,)`` long tensor of selected dimension indices.
        """
        return scores.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------


class RandomBranching(BranchingStrategy):
    """Uniform-random branching over eligible dimensions.

    Supports both paradigms:

    * **Input split** (``unstable_mask is None``):
      Random scores weighted by domain width — wider dimensions are more
      likely to be chosen, and zero-width dimensions are excluded.

    * **Neuron split** (``unstable_mask`` provided):
      Uniform-random scores masked to unstable neurons.  Stable neurons
      receive score 0 and are never selected.
    """

    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N, D = batch.batch_size, batch.input_dim
        device = batch.lb.device

        scores = torch.rand(N, D, device=device)

        if unstable_mask is not None:
            # Neuron-split mode: zero-out stable neurons
            mask = unstable_mask.float()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(N, -1)  # (D,) → (N, D)
            scores = scores * mask
        else:
            # Input-split mode: weight by width (zero-width → score 0)
            widths = batch.widths()  # (N, D)
            scores = scores * (widths > 0).float()

        return scores
