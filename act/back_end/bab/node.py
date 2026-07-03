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
#   BabNode is a thin single-subproblem wrapper that carries one priority
#   score and one optional candidate counterexample. ``BabNode.to_batch``
#   converts a single node into a ``SubproblemBatch`` of size 1 so that
#   call sites holding a single node can dispatch through the batched
#   code path.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

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

    # -- incremental-start fields ---------------------------------------------------
    incremental_alpha: Optional[Dict[int, torch.Tensor]] = None  # layer_id → [N, M, n]
    incremental_eta: Optional[Dict[int, torch.Tensor]] = None  # layer_id → [N, M, n]
    split_signs: Optional[Dict[int, torch.Tensor]] = None  # layer_id → [N, M, n] in {-1, 0, +1}
    parent_margins: Optional[torch.Tensor] = None  # [N]
    # [N] certified lower bound (min slack) from the parent solve; the node-selection
    # signal consumed by TopKBounding. Distinct from parent_margins (FSB's baseline).
    lower_bound: Optional[torch.Tensor] = None
    node_id: Optional[torch.Tensor] = None  # [N] long; logical identity
    parent_id: Optional[torch.Tensor] = None  # [N] long; parent's node_id (-1 for root)

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
        """Wrap ``Bounds[B, *shape]`` into ``SubproblemBatch[B, D]``.

        Multi-sample inputs (``B > 1`` — e.g., wrapped models carrying several
        specs at once) become ``B`` independent subproblems so each gets its
        own BaB exploration tree. Single-sample (``B == 1``) is the legacy
        path. Unbatched ``Bounds`` (``lb.dim() < 2``) is treated as ``B = 1``.
        """
        lb_raw = bounds.lb.detach()
        ub_raw = bounds.ub.detach()
        b = lb_raw.shape[0] if lb_raw.dim() >= 2 else 1
        lb = lb_raw.reshape(b, -1)
        ub = ub_raw.reshape(b, -1)
        depths = torch.full((b,), depth, dtype=torch.long, device=lb.device)
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


def _gather_optional_dict(
    d: Optional[Dict[int, torch.Tensor]],
    idx: torch.Tensor,
) -> Optional[Dict[int, torch.Tensor]]:
    if d is None:
        return None
    return {k: t.index_select(0, idx.to(t.device)) for k, t in d.items()}


def split_input(
    batch: SubproblemBatch,
    split_dims: torch.Tensor,
) -> tuple[SubproblemBatch, torch.Tensor]:
    n = batch.batch_size
    device = batch.lb.device
    mid = (batch.lb + batch.ub) / 2
    split_vals = mid.gather(1, split_dims.unsqueeze(1))
    parent_index = torch.arange(n, device=device).repeat(2)

    child_lb = batch.lb.index_select(0, parent_index)
    child_ub = batch.ub.index_select(0, parent_index)
    dims2 = split_dims.unsqueeze(1)
    child_ub[:n].scatter_(1, dims2, split_vals)
    child_lb[n:].scatter_(1, dims2, split_vals)
    child_depths = batch.depths.index_select(0, parent_index) + 1

    children = SubproblemBatch(
        lb=child_lb,
        ub=child_ub,
        depths=child_depths,
        incremental_alpha=_gather_optional_dict(batch.incremental_alpha, parent_index),
        incremental_eta=_gather_optional_dict(batch.incremental_eta, parent_index),
        split_signs=_gather_optional_dict(batch.split_signs, parent_index),
        parent_margins=(
            batch.parent_margins.index_select(0, parent_index.to(batch.parent_margins.device))
            if batch.parent_margins is not None
            else None
        ),
        lower_bound=(
            batch.lower_bound.index_select(0, parent_index.to(batch.lower_bound.device))
            if batch.lower_bound is not None
            else None
        ),
    )
    return children, parent_index


def rederive_embedding_block_eps(
    lb: torch.Tensor,
    ub: torch.Tensor,
    input_shape: tuple[int, ...],
    perturbed_positions: Optional[torch.Tensor],
    p_norm: float,
) -> torch.Tensor:
    """Return child per-token Lp radii for split embedding boxes.

    BaB input splits partition the enclosing embedding box.  For finite-p
    LP_EMBEDDING children, the dual A2 term must use a radius for each token
    block that contains the split child box around its midpoint.  The radius is
    therefore the primal-p norm of the child half-width vector in that block;
    non-perturbed token blocks keep radius zero.
    """
    if len(input_shape) < 2:
        raise ValueError(f"embedding input_shape must include token and embedding axes, got {input_shape}")
    n = int(lb.shape[0])
    token_count = int(input_shape[-2])
    embed_dim = int(input_shape[-1])
    flat_dim = token_count * embed_dim
    if int(lb.shape[-1]) < flat_dim or int(ub.shape[-1]) < flat_dim:
        raise ValueError(
            f"input bounds have dim {lb.shape[-1]}/{ub.shape[-1]}, expected at least {flat_dim}"
        )
    half = ((ub[:, :flat_dim] - lb[:, :flat_dim]) * 0.5).reshape(n, token_count, embed_dim)

    if p_norm == float("inf"):
        radii = half.abs().amax(dim=-1)
    elif p_norm == 1.0:
        radii = half.abs().sum(dim=-1)
    elif p_norm == 2.0:
        radii = torch.linalg.vector_norm(half, ord=2, dim=-1)
    else:
        radii = torch.linalg.vector_norm(half, ord=p_norm, dim=-1)

    if perturbed_positions is None:
        return radii
    pos = perturbed_positions.to(device=lb.device) if isinstance(perturbed_positions, torch.Tensor) else torch.as_tensor(perturbed_positions, device=lb.device)
    if pos.dtype == torch.bool:
        if tuple(pos.shape) == (token_count,):
            mask = pos.reshape(1, token_count).expand(n, -1)
        elif tuple(pos.shape) == (n, token_count):
            mask = pos
        elif pos.numel() == token_count:
            mask = pos.reshape(1, token_count).expand(n, -1)
        else:
            mask = pos.reshape(-1, token_count)
            if mask.shape[0] == 1:
                mask = mask.expand(n, -1)
    else:
        mask = torch.zeros((n, token_count), device=lb.device, dtype=torch.bool)
        if pos.numel() > 0:
            mask.index_fill_(1, pos.to(dtype=torch.long).flatten(), True)
    return torch.where(mask.to(dtype=torch.bool), radii, torch.zeros_like(radii))


def split_input_nary(
    batch: SubproblemBatch,
    cut_dim: torch.Tensor,
    k: int,
) -> tuple[SubproblemBatch, torch.Tensor]:
    if k < 2:
        raise ValueError(f"fanout must be >= 2, got {k}")
    n = batch.batch_size
    device = batch.lb.device
    parent_index = torch.arange(n, device=device).repeat(k)
    section = torch.arange(k, device=device).repeat_interleave(n)

    child_lb = batch.lb.index_select(0, parent_index)
    child_ub = batch.ub.index_select(0, parent_index)
    cut_c = cut_dim.to(device=device, dtype=torch.long).index_select(0, parent_index)
    cut_c2 = cut_c.unsqueeze(1)
    lb_at = child_lb.gather(1, cut_c2)
    ub_at = child_ub.gather(1, cut_c2)
    seg = (ub_at - lb_at) / k
    section_f = section.to(dtype=child_lb.dtype).unsqueeze(1)
    new_lb = torch.where(section.unsqueeze(1) == 0, lb_at, lb_at + section_f * seg)
    new_ub = torch.where(
        section.unsqueeze(1) == k - 1,
        ub_at,
        lb_at + (section_f + 1) * seg,
    )
    child_lb.scatter_(1, cut_c2, new_lb)
    child_ub.scatter_(1, cut_c2, new_ub)

    depth_inc = math.ceil(math.log2(k))
    child_depths = batch.depths.index_select(0, parent_index) + depth_inc

    children = SubproblemBatch(
        lb=child_lb,
        ub=child_ub,
        depths=child_depths,
        incremental_alpha=_gather_optional_dict(batch.incremental_alpha, parent_index),
        incremental_eta=_gather_optional_dict(batch.incremental_eta, parent_index),
        split_signs=_gather_optional_dict(batch.split_signs, parent_index),
        parent_margins=(
            batch.parent_margins.index_select(0, parent_index.to(batch.parent_margins.device))
            if batch.parent_margins is not None
            else None
        ),
        lower_bound=(
            batch.lower_bound.index_select(0, parent_index.to(batch.lower_bound.device))
            if batch.lower_bound is not None
            else None
        ),
    )
    return children, parent_index


def concat_children(a: SubproblemBatch, b: SubproblemBatch) -> SubproblemBatch:
    def _concat_optional_dicts(
        da: Optional[Dict[int, torch.Tensor]],
        db: Optional[Dict[int, torch.Tensor]],
    ) -> Optional[Dict[int, torch.Tensor]]:
        if da is None or db is None:
            assert da is None and db is None, "child dict fields must both be present or absent"
            return None
        assert set(da) == set(db), "child dict fields must have equal keys"
        return {key: torch.cat([da[key], db[key]], dim=0) for key in sorted(da)}

    def _concat_optional_tensor(
        ta: Optional[torch.Tensor],
        tb: Optional[torch.Tensor],
        name: str,
    ) -> Optional[torch.Tensor]:
        if ta is None or tb is None:
            assert ta is None and tb is None, f"{name} must be present in both children or neither"
            return None
        return torch.cat([ta, tb], dim=0)

    return SubproblemBatch(
        lb=torch.cat([a.lb, b.lb], dim=0),
        ub=torch.cat([a.ub, b.ub], dim=0),
        depths=torch.cat([a.depths, b.depths], dim=0),
        incremental_alpha=_concat_optional_dicts(a.incremental_alpha, b.incremental_alpha),
        incremental_eta=_concat_optional_dicts(a.incremental_eta, b.incremental_eta),
        split_signs=_concat_optional_dicts(a.split_signs, b.split_signs),
        parent_margins=_concat_optional_tensor(a.parent_margins, b.parent_margins, "parent_margins"),
        lower_bound=_concat_optional_tensor(a.lower_bound, b.lower_bound, "lower_bound"),
    )


def split_subproblems(
    batch: SubproblemBatch,
    split_dims: torch.Tensor,
) -> tuple[SubproblemBatch, SubproblemBatch]:
    """Bisect each subproblem along the chosen input dimension.

    This is a pure tensor operation — no Python loops over the batch. Incremental-state
    fields are deep-copied into both children so they do not alias the parent or
    each other.

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

    def _clone_dict_tensors(
        tensors: Optional[Dict[int, torch.Tensor]],
    ) -> Optional[Dict[int, torch.Tensor]]:
        return (
            {key: tensor.clone() for key, tensor in tensors.items()}
            if tensors is not None
            else None
        )

    left_incremental_alpha = _clone_dict_tensors(batch.incremental_alpha)
    left_incremental_eta = _clone_dict_tensors(batch.incremental_eta)
    left_split_signs = _clone_dict_tensors(batch.split_signs)
    left_parent_margins = (
        batch.parent_margins.clone() if batch.parent_margins is not None else None
    )

    right_incremental_alpha = _clone_dict_tensors(batch.incremental_alpha)
    right_incremental_eta = _clone_dict_tensors(batch.incremental_eta)
    right_split_signs = _clone_dict_tensors(batch.split_signs)
    right_parent_margins = (
        batch.parent_margins.clone() if batch.parent_margins is not None else None
    )

    left = SubproblemBatch(
        lb=batch.lb.clone(),
        ub=left_ub,
        depths=new_depths.clone(),
        incremental_alpha=left_incremental_alpha,
        incremental_eta=left_incremental_eta,
        split_signs=left_split_signs,
        parent_margins=left_parent_margins,
        lower_bound=(batch.lower_bound.clone() if batch.lower_bound is not None else None),
    )
    right = SubproblemBatch(
        lb=right_lb,
        ub=batch.ub.clone(),
        depths=new_depths.clone(),
        incremental_alpha=right_incremental_alpha,
        incremental_eta=right_incremental_eta,
        split_signs=right_split_signs,
        parent_margins=right_parent_margins,
        lower_bound=(batch.lower_bound.clone() if batch.lower_bound is not None else None),
    )
    return left, right


def split_neuron_subproblems(
    batch: SubproblemBatch,
    *,
    layer_id: int,
    neuron_idx: int,
    n_neurons: int,
    n_specs: int,
) -> tuple[SubproblemBatch, SubproblemBatch]:
    if neuron_idx < 0 or neuron_idx >= n_neurons:
        raise IndexError(f"neuron_idx {neuron_idx} out of range for n_neurons={n_neurons}")

    def _clone_dict_tensors(
        tensors: Optional[Dict[int, torch.Tensor]],
    ) -> Optional[Dict[int, torch.Tensor]]:
        return {key: tensor.clone() for key, tensor in tensors.items()} if tensors is not None else None

    def _clone_signs(sign: float) -> Dict[int, torch.Tensor]:
        signs = _clone_dict_tensors(batch.split_signs) or {}
        if layer_id not in signs:
            signs[layer_id] = torch.zeros(
                (batch.batch_size, n_specs, n_neurons),
                dtype=batch.lb.dtype,
                device=batch.lb.device,
            )
        signs[layer_id][:, :, neuron_idx] = sign
        return signs

    new_depths = batch.depths + 1
    on = SubproblemBatch(
        lb=batch.lb.clone(),
        ub=batch.ub.clone(),
        depths=new_depths.clone(),
        incremental_alpha=_clone_dict_tensors(batch.incremental_alpha),
        incremental_eta=_clone_dict_tensors(batch.incremental_eta),
        split_signs=_clone_signs(+1.0),
        parent_margins=batch.parent_margins.clone() if batch.parent_margins is not None else None,
        lower_bound=batch.lower_bound.clone() if batch.lower_bound is not None else None,
    )
    off = SubproblemBatch(
        lb=batch.lb.clone(),
        ub=batch.ub.clone(),
        depths=new_depths.clone(),
        incremental_alpha=_clone_dict_tensors(batch.incremental_alpha),
        incremental_eta=_clone_dict_tensors(batch.incremental_eta),
        split_signs=_clone_signs(-1.0),
        parent_margins=batch.parent_margins.clone() if batch.parent_margins is not None else None,
        lower_bound=batch.lower_bound.clone() if batch.lower_bound is not None else None,
    )
    return on, off


# ---------------------------------------------------------------------------
# Single-subproblem wrapper
# ---------------------------------------------------------------------------


@dataclass
class BabNode:
    """Single-subproblem record: one bounds box, one priority score, one
    optional candidate counterexample.

    Prefer :class:`SubproblemBatch` for batch-parallel processing; use this
    record when a call site only ever holds one subproblem at a time and
    wants priority-queue ordering. ``to_batch`` lifts a single node into a
    ``SubproblemBatch`` of size 1 for dispatch through the batched path.
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
