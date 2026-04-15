# ===- act/back_end/bab/node.py - Subproblem Representation ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Tensor-native representation of BaB subproblems.
#
#   SubproblemBatch is the primary data structure — every field is a tensor
#   with leading batch dimension (N, …) so that branching, bounding, and
#   (future) batched solving operate in pure tensor arithmetic.
#
#   BabNode is retained for backward compatibility with existing callers.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from act.back_end.core import Bounds


# ---------------------------------------------------------------------------
# Tensor-native batch representation (primary)
# ---------------------------------------------------------------------------


@dataclass
class SubproblemBatch:
    """Batched BaB subproblems for tensor-driven processing.

    Shape convention
    ~~~~~~~~~~~~~~~~
    * ``lb``, ``ub``:  ``(N, D)``  — input-space bounds per subproblem.
    * ``depths``:       ``(N,)``    — tree depth of each subproblem.

    All operations on this class are designed for batch-parallel execution.
    Future batch-solving will pass an entire ``SubproblemBatch`` to the
    solver backend in one call.
    """

    lb: torch.Tensor  # (N, D)  lower bounds
    ub: torch.Tensor  # (N, D)  upper bounds
    depths: torch.Tensor  # (N,)    tree depth

    # -- properties ---------------------------------------------------------

    @property
    def batch_size(self) -> int:
        """Number of subproblems in this batch."""
        return self.lb.shape[0]

    @property
    def input_dim(self) -> int:
        """Dimensionality of the input space."""
        return self.lb.shape[-1]

    def __len__(self) -> int:
        return self.batch_size

    # -- constructors -------------------------------------------------------

    @staticmethod
    def from_bounds(bounds: Bounds, depth: int = 0) -> SubproblemBatch:
        """Wrap a single ``Bounds`` into a batch of size 1."""
        lb = bounds.lb.detach().unsqueeze(0)  # (1, D)
        ub = bounds.ub.detach().unsqueeze(0)  # (1, D)
        depths = torch.tensor([depth], dtype=torch.long)
        return SubproblemBatch(lb=lb, ub=ub, depths=depths)

    # -- conversions --------------------------------------------------------

    def to_bounds_list(self) -> list[Bounds]:
        """Convert back to a list of ``Bounds`` for solver compatibility."""
        return [Bounds(self.lb[i], self.ub[i]) for i in range(self.batch_size)]

    # -- geometry -----------------------------------------------------------

    def widths(self) -> torch.Tensor:
        """Per-dimension widths: ``(N, D)``."""
        return self.ub - self.lb

    def total_width(self) -> torch.Tensor:
        """Sum of widths per subproblem: ``(N,)``."""
        return self.widths().sum(dim=-1)


# ---------------------------------------------------------------------------
# Batch splitting (tensor-native)
# ---------------------------------------------------------------------------


def split_subproblems(
    batch: SubproblemBatch,
    split_dims: torch.Tensor,
) -> tuple[SubproblemBatch, SubproblemBatch]:
    """Bisect each subproblem along the chosen input dimension.

    This is a pure tensor operation — no Python loops over the batch.

    Args:
        batch:      ``(N, D)`` subproblems.
        split_dims: ``(N,)`` long tensor — dimension to bisect per subproblem.

    Returns:
        ``(left, right)`` — two ``SubproblemBatch`` of the same shape,
        where ``left.ub[i, d] == right.lb[i, d] == midpoint``.
    """
    mid = (batch.lb + batch.ub) / 2  # (N, D)
    split_vals = mid.gather(1, split_dims.unsqueeze(1))  # (N, 1)

    # Left child: upper bound clamped at midpoint
    left_ub = batch.ub.clone()
    left_ub.scatter_(1, split_dims.unsqueeze(1), split_vals)

    # Right child: lower bound raised to midpoint
    right_lb = batch.lb.clone()
    right_lb.scatter_(1, split_dims.unsqueeze(1), split_vals)

    new_depths = batch.depths + 1

    left = SubproblemBatch(
        lb=batch.lb.clone(),
        ub=left_ub,
        depths=new_depths.clone(),
    )
    right = SubproblemBatch(
        lb=right_lb,
        ub=batch.ub.clone(),
        depths=new_depths.clone(),
    )
    return left, right


# ---------------------------------------------------------------------------
# Legacy compat
# ---------------------------------------------------------------------------


@dataclass
class BabNode:
    """Legacy single-node representation (backward compatibility).

    New code should prefer :class:`SubproblemBatch`.
    """

    box: Bounds
    depth: int
    score: float
    candidate_ce: Optional[np.ndarray] = None

    def __lt__(self, other: BabNode) -> bool:  # max-heap by score
        return self.score > other.score

    def to_batch(self) -> SubproblemBatch:
        """Upgrade to tensor batch of size 1."""
        return SubproblemBatch.from_bounds(self.box, depth=self.depth)
