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
#   ``SubproblemBatch.select`` and ``SubproblemBatch.concat`` are the lane
#   gather / stack primitives shared by the BaB loop and the branchers;
#   child construction for input and neuron splits lives here.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Dict, Iterable, Iterator, Optional, cast

import torch

from act.back_end.core import Bounds, Layer, Net
from act.back_end.solver.solver_dual import (
    AlphaState,
    AlphaTree,
    _alpha_spec_row_count,
    _alpha_tree_concat_lanes,
    _alpha_tree_gather_lanes,
    _alpha_tree_leaves,
    _alpha_tree_map,
)


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
    incremental_alpha: Optional[AlphaState] = None  # layer_id → kernel alpha tree
    incremental_eta: Optional[Dict[int, torch.Tensor]] = None  # layer_id → [N, M, n]
    split_signs: Optional[Dict[int, torch.Tensor]] = None  # layer_id → [N, M, n] in {-1, 0, +1}
    parent_margins: Optional[torch.Tensor] = None  # [N]
    # [N] certified lower bound (min slack) from the parent solve; the node-selection
    # signal consumed by TopKBounding. Distinct from parent_margins (FSB's baseline).
    lower_bound: Optional[torch.Tensor] = None
    node_id: Optional[torch.Tensor] = None  # [N] long; logical identity
    parent_id: Optional[torch.Tensor] = None  # [N] long; parent's node_id (-1 for root)
    decision_literals: Optional[torch.Tensor] = None  # [N, S] ordered decisions, never inferred phases
    warm_state_present: Optional[torch.Tensor] = None  # [N] bool; padded alpha is not a warm start

    def __post_init__(self) -> None:
        """Record whether each lane owns real, complete optimizer state."""
        if self.warm_state_present is None:
            self.warm_state_present = torch.full(
                (self.batch_size,),
                next(_alpha_tree_leaves(self.incremental_alpha), None) is not None,
                dtype=torch.bool, device=self.lb.device,
            )

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

        Each leading row becomes a subproblem lane. The BaB entry point starts
        from one root sample; descendant boxes may be batched here.
        Unbatched ``Bounds`` (``lb.dim() < 2``) is treated as ``B = 1``.
        """
        lb_raw = bounds.lb.detach()
        ub_raw = bounds.ub.detach()
        b = lb_raw.shape[0] if lb_raw.dim() >= 2 else 1
        lb = lb_raw.reshape(b, -1)
        ub = ub_raw.reshape(b, -1)
        depths = torch.full((b,), depth, dtype=torch.long, device=lb.device)
        return SubproblemBatch(lb=lb, ub=ub, depths=depths)

    # -- lane selection / concatenation -------------------------------------

    def select(self, indices: torch.Tensor) -> SubproblemBatch:
        """Return the lanes at ``indices`` with every per-lane field preserved."""
        return SubproblemBatch(
            lb=self.lb.index_select(0, indices.to(self.lb.device)),
            ub=self.ub.index_select(0, indices.to(self.ub.device)),
            depths=self.depths.index_select(0, indices.to(self.depths.device)),
            incremental_alpha=cast(
                Optional[AlphaState],
                _alpha_tree_gather_lanes(self.incremental_alpha, indices),
            ),
            incremental_eta=_gather_optional_dict(self.incremental_eta, indices),
            split_signs=_gather_optional_dict(self.split_signs, indices),
            parent_margins=_gather_optional_tensor(self.parent_margins, indices),
            lower_bound=_gather_optional_tensor(self.lower_bound, indices),
            node_id=_gather_optional_tensor(self.node_id, indices),
            parent_id=_gather_optional_tensor(self.parent_id, indices),
            decision_literals=_gather_optional_tensor(self.decision_literals, indices),
            warm_state_present=_gather_optional_tensor(self.warm_state_present, indices),
        )

    def concat(self, other: SubproblemBatch) -> SubproblemBatch:
        """Stack ``other``'s lanes after this batch's lanes.

        ``other`` is moved to this batch's device (optional per-lane tensors
        also to its dtype). The per-layer state dicts may disagree on their key
        sets: a layer touched only by one side's splits exists there and not in
        the other. Zero padding is exact for ``split_signs`` and
        ``incremental_eta`` (0 = unconstrained / no multiplier), so it
        preserves each side's feasible region. Missing ``incremental_alpha``
        subtrees are zero-initialized through the shared alpha-tree helper, but
        their lanes are marked cold and must not use those placeholders as proof.
        """
        if other.lb.device != self.lb.device:
            other = other.to(self.lb.device)
        n_self, n_other = self.batch_size, other.batch_size
        alpha = _alpha_tree_concat_lanes(
            self.incremental_alpha, other.incremental_alpha, n_self, n_other
        )
        leaf_count = sum(1 for _ in _alpha_tree_leaves(alpha, logical=True))
        presence = []
        for batch in (self, other):
            assert batch.warm_state_present is not None
            complete = leaf_count > 0 and leaf_count == sum(
                1 for _ in _alpha_tree_leaves(batch.incremental_alpha, logical=True)
            )
            presence.append(batch.warm_state_present & complete)
        return SubproblemBatch(
            lb=torch.cat([self.lb, other.lb.to(self.lb.device)], dim=0),
            ub=torch.cat([self.ub, other.ub.to(self.ub.device)], dim=0),
            depths=torch.cat([self.depths, other.depths.to(self.depths.device)], dim=0),
            incremental_alpha=cast(Optional[AlphaState], alpha),
            incremental_eta=_concat_padded_dict(
                self.incremental_eta, other.incremental_eta, n_self, n_other
            ),
            split_signs=_concat_padded_dict(self.split_signs, other.split_signs, n_self, n_other),
            parent_margins=_concat_optional_tensor(
                self.parent_margins, other.parent_margins, n_self, n_other
            ),
            lower_bound=_concat_optional_tensor(
                self.lower_bound, other.lower_bound, n_self, n_other
            ),
            node_id=_concat_optional_tensor(self.node_id, other.node_id),
            parent_id=_concat_optional_tensor(self.parent_id, other.parent_id),
            decision_literals=_concat_decisions(self.decision_literals, other.decision_literals),
            warm_state_present=torch.cat((presence[0], presence[1].to(self.lb.device))),
        )

    def tensors(self) -> Iterator[torch.Tensor]:
        """Yield resident tensors without gathering or allocating their payload."""
        for field in fields(self):
            yield from _alpha_tree_leaves(getattr(self, field.name))

    def to(self, device: torch.device | str) -> SubproblemBatch:
        """Return all state on ``device``, leaving this batch intact on failure."""
        return replace(self, **{
            field.name: _alpha_tree_map(
                getattr(self, field.name), lambda tensor: tensor.to(device=device)
            )
            for field in fields(self)
        })

    def drop_warm_state(self) -> SubproblemBatch:
        """Discard optimizer state only; retain geometry, phases and provenance."""
        return replace(
            self, incremental_alpha=None, incremental_eta=None,
            warm_state_present=torch.zeros(
                self.batch_size, dtype=torch.bool, device=self.lb.device
            ),
        )

    # -- geometry -----------------------------------------------------------

    def widths(self) -> torch.Tensor:
        """Per-dimension widths: ``(N, D)``."""
        return self.ub - self.lb


# ---------------------------------------------------------------------------
# Lane gather / bounds slicing / concat helpers
# ---------------------------------------------------------------------------

def tensor_storage_bytes(
    tensors: Iterable[torch.Tensor], device: torch.device | str | None = None
) -> int:
    """Count unique actual storages, including the full backing of tensor views."""
    target = None if device is None else torch.device(device)
    seen: set[tuple[torch.device, int]] = set()
    total = 0
    for tensor in tensors:
        if target is not None and (
            tensor.device.type != target.type
            or (target.index is not None and tensor.device.index != target.index)
        ):
            continue
        storage = tensor.untyped_storage()
        key = (tensor.device, storage._cdata)
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def slice_bounds_dict(
    bounds_dict: Dict[int, Bounds], rows: torch.Tensor
) -> Dict[int, Bounds]:
    """Select lanes from a batched bounds dictionary without mutating it."""
    return {
        layer_id: Bounds(
            bounds.lb.index_select(0, rows.to(bounds.lb.device)),
            bounds.ub.index_select(0, rows.to(bounds.ub.device)),
        )
        for layer_id, bounds in bounds_dict.items()
    }


def slice_branching_state(
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
    lane_idx: torch.Tensor,
    k_actual: int,
) -> tuple[Optional[Dict[int, Bounds]], Optional[Dict[int, torch.Tensor]]]:
    # ν/bounds are computed over the full k_actual wave; the brancher runs on the
    # sub-batch actually being split. Bounds are [k_actual, *]; ν is [k_actual*M, n]
    # packed sample-major (row b*M+j), so ν rows expand per selected lane.
    bd_out = (
        slice_bounds_dict(bounds_dict, lane_idx) if bounds_dict is not None else None
    )
    nu_out: Optional[Dict[int, torch.Tensor]] = None
    if nu_per_layer is not None:
        nu_out = {}
        for lid, tensor in nu_per_layer.items():
            total = int(tensor.shape[0])
            if k_actual > 0 and total != k_actual and total % k_actual == 0:
                m = total // k_actual
                rows = (
                    lane_idx.to(tensor.device).unsqueeze(1) * m
                    + torch.arange(m, device=tensor.device)
                ).reshape(-1)
            else:
                rows = lane_idx.to(tensor.device)
            nu_out[lid] = tensor.index_select(0, rows)
    return bd_out, nu_out


def _layer_neuron_count(layer: Layer) -> int:
    """Width of a layer's contiguous output-variable block."""
    span = int(layer.out_vars[-1] - layer.out_vars[0] + 1)
    assert span == len(layer.out_vars), (
        f"layer {layer.id} has non-contiguous out_vars: span {span} != "
        f"{len(layer.out_vars)} declared variables"
    )
    return span


def _relu_literal_layout(net: Net) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """One network-wide numbering layout for decisions and the CLIMB codec."""
    layers = sorted((layer for layer in net.layers if layer.kind == "RELU"), key=lambda layer: layer.id)
    widths = tuple(_layer_neuron_count(layer) for layer in layers)
    offsets: list[int] = []
    total = 0
    for width in widths:
        offsets.append(total)
        total += width
    return tuple(layer.id for layer in layers), widths, tuple(offsets)


def _encode_literal_tensor(
    table_ids: tuple[int, ...], table_offsets: tuple[int, ...],
    layer_ids: torch.Tensor, neurons: torch.Tensor, signs: torch.Tensor,
) -> torch.Tensor:
    """Encode signed neuron decisions using the shared codec layout."""
    ids = torch.tensor(table_ids, dtype=torch.long, device=layer_ids.device)
    offsets = torch.tensor(table_offsets, dtype=torch.long, device=layer_ids.device)
    indices = torch.searchsorted(ids, layer_ids.to(torch.long))
    return signs.to(torch.long) * (offsets.index_select(0, indices) + neurons.to(torch.long) + 1)


def _gather_optional_dict(
    d: Optional[Dict[int, torch.Tensor]],
    idx: torch.Tensor,
) -> Optional[Dict[int, torch.Tensor]]:
    """Gather rows ``idx`` from every tensor of an optional per-layer dict."""
    return cast(
        Optional[Dict[int, torch.Tensor]],
        _alpha_tree_gather_lanes(cast(AlphaTree, d), idx),
    )


def _gather_optional_tensor(
    t: Optional[torch.Tensor],
    idx: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Gather rows ``idx`` from an optional per-lane tensor."""
    return None if t is None else t.index_select(0, idx.to(t.device))


def _concat_optional_tensor(
    x: Optional[torch.Tensor],
    y: Optional[torch.Tensor],
    n_x: int = 0,
    n_y: int = 0,
) -> Optional[torch.Tensor]:
    """Stack optional rows, filling absent scores but refusing absent identities."""
    if x is None and y is None:
        return None
    if x is None:
        if not n_x:
            raise ValueError("cannot concatenate known and missing node identity")
        assert y is not None
        x = y.new_zeros((n_x, *y.shape[1:]))
    if y is None:
        if not n_y:
            raise ValueError("cannot concatenate known and missing node identity")
        y = x.new_zeros((n_y, *x.shape[1:]))
    return torch.cat([x, y.to(device=x.device, dtype=x.dtype)], dim=0)


def _concat_decisions(
    x: Optional[torch.Tensor], y: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Pad ordered literal slots without inventing missing provenance."""
    if (x is None) != (y is None):
        raise ValueError("cannot concatenate known and missing decision provenance")
    if x is None or y is None:
        return None
    width = max(x.shape[1], y.shape[1])
    return torch.cat((torch.nn.functional.pad(x, (0, width - x.shape[1])),
                      torch.nn.functional.pad(y.to(x), (0, width - y.shape[1]))))


def _concat_padded_dict(
    x: Optional[Dict[int, torch.Tensor]],
    y: Optional[Dict[int, torch.Tensor]],
    n_x: int,
    n_y: int,
) -> Optional[Dict[int, torch.Tensor]]:
    """Stack multipliers/phases, broadcasting their shared spec axis as needed."""
    if x is not None and y is not None:
        x, y = dict(x), dict(y)
        for lid in x.keys() & y.keys():
            left, right = x[lid], y[lid]
            if left.ndim == right.ndim == 3 and left.shape[1] != right.shape[1]:
                rows = max(left.shape[1], right.shape[1])
                if left.shape[1] not in (1, rows) or right.shape[1] not in (1, rows):
                    raise ValueError("multiplier/phase spec-row counts must agree")
                x[lid] = left.expand(-1, rows, -1)
                y[lid] = right.expand(-1, rows, -1)
    return cast(
        Optional[Dict[int, torch.Tensor]],
        _alpha_tree_concat_lanes(cast(AlphaTree, x), cast(AlphaTree, y), n_x, n_y),
    )


def _state_spec_row_count(batch: SubproblemBatch) -> int:
    """Infer active spec rows without treating shared attention leaves as specs."""
    counts = {_alpha_spec_row_count(batch.incremental_alpha) or 1}
    for state in (batch.incremental_eta, batch.split_signs):
        if state:
            counts.update(int(tensor.shape[1]) for tensor in state.values())
    nonshared = counts - {1}
    if len(nonshared) > 1:
        raise ValueError(f"incompatible incremental-state spec counts: {sorted(counts)}")
    return next(iter(nonshared), 1)


# ---------------------------------------------------------------------------
# Batch splitting (tensor-native)
# ---------------------------------------------------------------------------


def _assert_splittable(batch: SubproblemBatch, dims2: torch.Tensor) -> None:
    """Refuse a zero-width cut: its children are copies of the parent."""
    widths_at = batch.widths().gather(1, dims2)
    if not bool((widths_at > 0).all().item()):
        raise ValueError(
            f"split dimension has zero width (min {float(widths_at.min().item()):g}); "
            "bisecting it returns two children identical to the parent"
        )


def _child_lanes(
    batch: SubproblemBatch,
    parent_index: torch.Tensor,
    depth_inc: int,
) -> SubproblemBatch:
    """Gather child lanes while leaving provenance for the BaB engine to assign."""
    return SubproblemBatch(
        lb=batch.lb.index_select(0, parent_index.to(batch.lb.device)),
        ub=batch.ub.index_select(0, parent_index.to(batch.ub.device)),
        depths=(
            batch.depths.index_select(0, parent_index.to(batch.depths.device))
            + depth_inc
        ),
        incremental_alpha=cast(
            Optional[AlphaState],
            _alpha_tree_gather_lanes(batch.incremental_alpha, parent_index),
        ),
        incremental_eta=_gather_optional_dict(batch.incremental_eta, parent_index),
        split_signs=_gather_optional_dict(batch.split_signs, parent_index),
        parent_margins=_gather_optional_tensor(batch.parent_margins, parent_index),
        lower_bound=_gather_optional_tensor(batch.lower_bound, parent_index),
        decision_literals=_gather_optional_tensor(batch.decision_literals, parent_index),
        warm_state_present=_gather_optional_tensor(batch.warm_state_present, parent_index),
    )


def split_input(
    batch: SubproblemBatch,
    split_dims: torch.Tensor,
) -> tuple[SubproblemBatch, torch.Tensor]:
    n = batch.batch_size
    device = batch.lb.device
    dims2 = split_dims.unsqueeze(1)
    _assert_splittable(batch, dims2)
    mid = (batch.lb + batch.ub) / 2
    split_vals = mid.gather(1, dims2)
    parent_index = torch.arange(n, device=device).repeat(2)

    children = _child_lanes(batch, parent_index, 1)
    children.ub[:n].scatter_(1, dims2, split_vals)
    children.lb[n:].scatter_(1, dims2, split_vals)
    return children, parent_index


def _selected_parent_sign_conflicts(
    batch: SubproblemBatch,
    top_layers: torch.Tensor,
    top_neurons: torch.Tensor,
    k: int,
) -> int:
    """Count selected neurons that already carry a parent split literal."""
    if not batch.split_signs:
        return 0
    conflicts = 0
    for bit in range(k):
        for layer_id_value in torch.unique(top_layers[:, bit]).tolist():
            layer_id = int(layer_id_value)
            parent_signs = batch.split_signs.get(layer_id)
            if parent_signs is None:
                continue
            lanes = torch.where(top_layers[:, bit] == layer_id_value)[0]
            neurons = top_neurons[lanes, bit].to(
                device=parent_signs.device, dtype=torch.long
            )
            selected = parent_signs[
                lanes.to(parent_signs.device), :, neurons
            ]
            conflicts += int((selected != 0).any(dim=1).sum().item())
    return conflicts


def split_neurons(
    batch: SubproblemBatch,
    net: Net,
    top_layers: torch.Tensor,
    top_neurons: torch.Tensor,
    k: int,
    *,
    first_sign: float = -1.0,
) -> tuple[SubproblemBatch, torch.Tensor]:
    """Build every sign-combination child for each lane's selected neurons."""
    n_lanes = batch.batch_size
    n_children = 2**k
    device = batch.lb.device
    assert (
        conflicts := _selected_parent_sign_conflicts(
            batch, top_layers, top_neurons, k
        )
    ) == 0, (
        "NEURON SPLIT invariant violated: selected neuron already split in parent "
        f"(lanes={n_lanes}, k={k}, selections={n_lanes * k}, "
        f"conflicts={conflicts})"
    )
    parent_index = torch.arange(n_lanes, device=device).repeat(n_children)
    children = _child_lanes(batch, parent_index, k)
    expected_children = n_lanes * n_children
    assert (
        children.batch_size == parent_index.numel() == expected_children
    ), (
        "NEURON SPLIT invariant violated: child/provenance count mismatch "
        f"(lanes={n_lanes}, k={k}, expected={expected_children}, "
        f"children={children.batch_size}, parent_indices={parent_index.numel()})"
    )

    m_specs = _state_spec_row_count(batch)

    signs = children.split_signs or {}
    for layer_id_value in torch.unique(top_layers).tolist():
        layer_id = int(layer_id_value)
        n_neurons = _layer_neuron_count(net.by_id[layer_id])
        if layer_id not in signs:
            signs[layer_id] = torch.zeros(
                n_children * n_lanes,
                m_specs,
                n_neurons,
                device=device,
                dtype=batch.lb.dtype,
            )
        else:
            signs[layer_id] = signs[layer_id].clone()
        for bit in range(k):
            lane_selection = torch.where(top_layers[:, bit] == layer_id_value)[0]
            if lane_selection.numel() == 0:
                continue
            neuron_selection = top_neurons[lane_selection, bit].to(
                device=device, dtype=torch.long
            )
            for child in range(n_children):
                sign_value = (
                    -first_sign if (child >> bit) & 1 else first_sign
                )
                rows = child * n_lanes + lane_selection
                # Paired advanced indices write rows[i] to neuron_selection[i].
                signs[layer_id][rows, :, neuron_selection] = sign_value

    children.split_signs = signs
    if children.decision_literals is not None:
        layer_ids, _, offsets = _relu_literal_layout(net)
        selected = _encode_literal_tensor(
            layer_ids, offsets,
            top_layers.reshape(-1), top_neurons.reshape(-1),
            torch.ones(n_lanes * k, dtype=torch.long, device=device),
        ).reshape(n_lanes, k)
        bits = (torch.arange(n_children, device=device)[:, None]
                >> torch.arange(k, device=device)[None]) & 1
        polarity = torch.where(bits != 0, -first_sign, first_sign).to(torch.long)
        added = (selected[None] * polarity[:, None]).reshape(children.batch_size, k)
        children.decision_literals = torch.cat((children.decision_literals, added), dim=1)
    assert children.batch_size == n_children * n_lanes, (
        f"split_neurons: {children.batch_size} children for {n_lanes} lanes x {n_children} sign combinations"
    )
    assert parent_index.numel() == children.batch_size, (
        f"split_neurons: parent_index has {parent_index.numel()} entries for {children.batch_size} children"
    )
    return children, parent_index


def split_input_nary(
    batch: SubproblemBatch,
    cut_dim: torch.Tensor,
    k: int,
) -> tuple[SubproblemBatch, torch.Tensor]:
    if k < 2:
        raise ValueError(f"fanout must be >= 2, got {k}")
    n = batch.batch_size
    device = batch.lb.device
    _assert_splittable(
        batch, cut_dim.to(device=device, dtype=torch.long).reshape(-1).unsqueeze(1)
    )
    parent_index = torch.arange(n, device=device).repeat(k)
    section = torch.arange(k, device=device).repeat_interleave(n)

    children = _child_lanes(batch, parent_index, math.ceil(math.log2(k)))
    cut_c = cut_dim.to(device=device, dtype=torch.long).index_select(0, parent_index)
    cut_c2 = cut_c.unsqueeze(1)
    lb_at = children.lb.gather(1, cut_c2)
    ub_at = children.ub.gather(1, cut_c2)
    seg = (ub_at - lb_at) / k
    section_f = section.to(dtype=children.lb.dtype).unsqueeze(1)
    new_lb = torch.where(section.unsqueeze(1) == 0, lb_at, lb_at + section_f * seg)
    new_ub = torch.where(
        section.unsqueeze(1) == k - 1,
        ub_at,
        lb_at + (section_f + 1) * seg,
    )
    children.lb.scatter_(1, cut_c2, new_lb)
    children.ub.scatter_(1, cut_c2, new_ub)
    return children, parent_index
