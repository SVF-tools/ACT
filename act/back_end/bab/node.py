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

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch

from act.back_end.core import Bounds, Layer, Net, ParamValue
from act.back_end.layer_schema import LayerKind
from act.front_end.specs import InKind


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

    # -- lane selection / concatenation -------------------------------------

    def select(self, indices: torch.Tensor) -> SubproblemBatch:
        """Return the lanes at ``indices`` with every per-lane field preserved."""
        return SubproblemBatch(
            lb=self.lb.index_select(0, indices.to(self.lb.device)),
            ub=self.ub.index_select(0, indices.to(self.ub.device)),
            depths=self.depths.index_select(0, indices.to(self.depths.device)),
            incremental_alpha=_gather_optional_dict(self.incremental_alpha, indices),
            incremental_eta=_gather_optional_dict(self.incremental_eta, indices),
            split_signs=_gather_optional_dict(self.split_signs, indices),
            parent_margins=_gather_optional_tensor(self.parent_margins, indices),
            lower_bound=_gather_optional_tensor(self.lower_bound, indices),
            node_id=_gather_optional_tensor(self.node_id, indices),
            parent_id=_gather_optional_tensor(self.parent_id, indices),
        )

    def concat(self, other: SubproblemBatch) -> SubproblemBatch:
        """Stack ``other``'s lanes after this batch's lanes.

        ``other`` is moved to this batch's device (optional per-lane tensors
        also to its dtype). The per-layer state dicts may disagree on their key
        sets: a layer touched only by one side's splits exists there and not in
        the other. Zero padding is exact for ``split_signs`` and
        ``incremental_eta`` (0 = unconstrained / no multiplier), so it
        preserves each side's feasible region. ``incremental_alpha`` is only
        ever concatenated between children of the same parent batch, whose key
        sets agree.
        """
        n_self, n_other = self.batch_size, other.batch_size
        return SubproblemBatch(
            lb=torch.cat([self.lb, other.lb.to(self.lb.device)], dim=0),
            ub=torch.cat([self.ub, other.ub.to(self.ub.device)], dim=0),
            depths=torch.cat([self.depths, other.depths.to(self.depths.device)], dim=0),
            incremental_alpha=_concat_padded_dict(
                self.incremental_alpha, other.incremental_alpha, n_self, n_other
            ),
            incremental_eta=_concat_padded_dict(
                self.incremental_eta, other.incremental_eta, n_self, n_other
            ),
            split_signs=_concat_padded_dict(self.split_signs, other.split_signs, n_self, n_other),
            parent_margins=_concat_optional_tensor(self.parent_margins, other.parent_margins),
            lower_bound=_concat_optional_tensor(self.lower_bound, other.lower_bound),
            node_id=_concat_optional_tensor(self.node_id, other.node_id),
            parent_id=_concat_optional_tensor(self.parent_id, other.parent_id),
        )

    # -- geometry -----------------------------------------------------------

    def widths(self) -> torch.Tensor:
        """Per-dimension widths: ``(N, D)``."""
        return self.ub - self.lb


# ---------------------------------------------------------------------------
# Lane gather / bounds slicing / concat helpers
# ---------------------------------------------------------------------------


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


def _gather_optional_dict(
    d: Optional[Dict[int, torch.Tensor]],
    idx: torch.Tensor,
) -> Optional[Dict[int, torch.Tensor]]:
    """Gather rows ``idx`` from every tensor of an optional per-layer dict."""
    if d is None:
        return None
    return {k: t.index_select(0, idx.to(t.device)) for k, t in d.items()}


def _gather_optional_tensor(
    t: Optional[torch.Tensor],
    idx: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Gather rows ``idx`` from an optional per-lane tensor."""
    return None if t is None else t.index_select(0, idx.to(t.device))


def _concat_optional_tensor(
    x: Optional[torch.Tensor],
    y: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Stack two optional per-lane tensors; ``None`` unless both are present."""
    if x is None or y is None:
        return None
    return torch.cat([x, y.to(device=x.device, dtype=x.dtype)], dim=0)


def _concat_padded_dict(
    x: Optional[Dict[int, torch.Tensor]],
    y: Optional[Dict[int, torch.Tensor]],
    n_x: int,
    n_y: int,
) -> Optional[Dict[int, torch.Tensor]]:
    """Stack two per-layer dicts, zero-filling a layer block missing on one side."""
    if not x and not y:
        return None
    x = x or {}
    y = y or {}
    out: Dict[int, torch.Tensor] = {}
    for lid in sorted(set(x) | set(y)):
        present = x.get(lid)
        if present is None:
            present = y[lid]
        trailing = present.shape[1:]
        left = x.get(lid)
        if left is None:
            left = torch.zeros((n_x, *trailing), dtype=present.dtype, device=present.device)
        right = y.get(lid)
        if right is None:
            right = torch.zeros((n_y, *trailing), dtype=present.dtype, device=present.device)
        out[lid] = torch.cat([left, right.to(device=left.device)], dim=0)
    return out


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
        incremental_alpha=_gather_optional_dict(
            batch.incremental_alpha, parent_index
        ),
        incremental_eta=_gather_optional_dict(batch.incremental_eta, parent_index),
        split_signs=_gather_optional_dict(batch.split_signs, parent_index),
        parent_margins=_gather_optional_tensor(batch.parent_margins, parent_index),
        lower_bound=_gather_optional_tensor(batch.lower_bound, parent_index),
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

    m_specs = 1
    if batch.incremental_alpha:
        m_specs = int(next(iter(batch.incremental_alpha.values())).shape[1])
    elif batch.split_signs:
        m_specs = int(next(iter(batch.split_signs.values())).shape[1])

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
    assert children.batch_size == n_children * n_lanes, (
        f"split_neurons: {children.batch_size} children for {n_lanes} lanes x {n_children} sign combinations"
    )
    assert parent_index.numel() == children.batch_size, (
        f"split_neurons: parent_index has {parent_index.numel()} entries for {children.batch_size} children"
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
    from act.front_end.specs import normalize_position_mask

    mask = normalize_position_mask(
        perturbed_positions, token_count, batch_shape=(n,), device=lb.device,
    )
    return torch.where(mask, radii, torch.zeros_like(radii))


def _finite_embedding_spec(net: Net) -> Optional[Layer]:
    for layer in net.layers:
        if (
            layer.kind == LayerKind.INPUT_SPEC.value
            and layer.params.get("kind") == InKind.LP_EMBEDDING
        ):
            p_norm = layer.params.get("p_norm", float("inf"))
            if isinstance(p_norm, torch.Tensor):
                p_value = float(p_norm.reshape(-1)[0].item())
            elif isinstance(p_norm, (int, float, bool)):
                p_value = float(p_norm)
            else:
                continue
            if p_value != float("inf"):
                return layer
    return None


def _install_embedding_child_block_eps(
    net: Net,
    batched_bounds: Bounds,
    batch: SubproblemBatch,
) -> list[tuple[Layer, ParamValue]]:
    """Install finite-p embedding radii derived from a split child box."""
    spec = _finite_embedding_spec(net)
    if (
        spec is None
        or batch.depths.numel() == 0
        or int(batch.depths.max().item()) == 0
    ):
        return []
    p_raw = spec.params.get("p_norm", float("inf"))
    if isinstance(p_raw, torch.Tensor):
        p_norm = float(p_raw.reshape(-1)[0].item())
    elif isinstance(p_raw, (int, float, bool)):
        p_norm = float(p_raw)
    else:
        return []
    input_shape = tuple(batched_bounds.lb.shape[1:])
    positions_raw = spec.params.get("perturbed_positions")
    positions = positions_raw if isinstance(positions_raw, torch.Tensor) else None
    block_eps = rederive_embedding_block_eps(
        batched_bounds.lb.flatten(start_dim=1),
        batched_bounds.ub.flatten(start_dim=1),
        input_shape,
        positions,
        p_norm,
    )
    old_values: list[tuple[Layer, ParamValue]] = []
    for layer in net.layers:
        kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        if kind in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value):
            old_values.append((layer, layer.params.get("bab_block_eps", None)))
            layer.params["bab_block_eps"] = block_eps
    return old_values


def _restore_embedding_child_block_eps(
    updates: list[tuple[Layer, ParamValue]],
) -> None:
    """Restore input-layer embedding radii after a child solve."""
    for layer, old in updates:
        if old is None:
            layer.params.pop("bab_block_eps", None)
        else:
            layer.params["bab_block_eps"] = old


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
