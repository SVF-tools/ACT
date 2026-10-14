# ===- act/back_end/bab/branching/multi_split.py - Joint Multi-Neuron Split -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Joint multi-neuron splitting (verdict-boundary) for Branch-and-Bound.
#
#   Implements the 2^k sign-combination fan-out described in "Mining Verdict
#   Boundaries for Neural Network Verification" (Jiawei Ren, Guanqin Zhang,
#   Zhenya Zhang, Yulei Sui, FM 2026).  Each lane's top-k BaBSR-scored neurons
#   are split together into all 2^k sign combinations in a single wave; the
#   joint gain is super-additive versus k greedy single splits.
#
#   Entry points:
#     ``presplit_root``                 — choose root neurons for pre-splitting.
#     ``gain_tested_decision``          — choose a measured-gain neuron split.
#     ``groups_to_tensors``             — normalize controller neuron groups.
#     ``enumerate_unstable_candidates`` — list splittable neurons with scores.
#     ``_collect_neuron_candidates``    — per-lane BaBSR scores (internal).
#     ``_multi_split_from_decision``    — top-level entry: score, select, split.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

from act.config.config import BaBConfig, DualConfig
from act.back_end.bab.branching.branching import SplitDecision
from act.back_end.bab.node import (
    SubproblemBatch,
    _gather_optional_dict,
    _layer_neuron_count,
    split_neurons,
)
from act.back_end.core import Bounds, Layer, Net

if TYPE_CHECKING:
    from act.back_end.solver.solver_dual import DualBatchResult


# ---------------------------------------------------------------------------
# Joint multi-neuron splitting (verdict-boundary)
#
#   "Mining Verdict Boundaries for Neural Network Verification"
#   Jiawei Ren, Guanqin Zhang, Zhenya Zhang, Yulei Sui
#   FM 2026
# ---------------------------------------------------------------------------


def presplit_root(
    root: SubproblemBatch,
    net: Net,
    bounds_dict: Dict[int, Bounds],
    nu_per_layer: Dict[int, torch.Tensor],
    k: int,
) -> Optional[SubproblemBatch]:
    """Materialize the root descendants from one best layer's top-k neurons."""
    best: Optional[tuple[int, torch.Tensor, torch.Tensor]] = None
    for layer_id, nu in nu_per_layer.items():
        bounds = bounds_dict.get(layer_id)
        if bounds is None:
            continue
        lb = bounds.lb.flatten(start_dim=1)[0]
        ub = bounds.ub.flatten(start_dim=1)[0]
        n = min(lb.shape[-1], nu.shape[-1])
        lb, ub = lb[:n], ub[:n]
        ambiguous = (lb < 0) & (ub > 0)
        if not bool(ambiguous.any().item()):
            continue
        area = (-lb * ub / (ub - lb).clamp(min=1e-12)).clamp(min=0)
        score = area * nu.reshape(-1, nu.shape[-1])[:, :n].abs().sum(dim=0)
        score = torch.where(ambiguous, score, torch.zeros_like(score))
        if best is None or float(score.max()) > float(best[1].max()):
            best = (layer_id, score, lb)
    if best is None:
        return None
    layer_id, score, _ = best
    k = min(k, int((score > 0).sum().item()))
    if k < 1:
        return None
    top_idx = torch.topk(score, k=k).indices
    assert score.shape[-1] == _layer_neuron_count(net.by_id[layer_id])
    children, _ = split_neurons(
        root,
        net,
        torch.full(
            (root.batch_size, k),
            layer_id,
            dtype=torch.long,
            device=root.lb.device,
        ),
        top_idx.unsqueeze(0),
        k,
    )
    return children


def _collect_neuron_candidates(
    branch_batch: SubproblemBatch,
    bounds_dict: Dict[int, Bounds],
    nu_per_layer: Dict[int, torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per-lane BaBSR scores (area x |nu|) over all splittable neurons.

    Returns ``(scores, layer_ids, neuron_ids)``, each ``[K, C]`` over the
    concatenated candidate axis; stable or already-split neurons score -inf.
    """
    kb = branch_batch.batch_size
    device = branch_batch.lb.device
    cand_layers: List[torch.Tensor] = []
    cand_neurons: List[torch.Tensor] = []
    cand_scores: List[torch.Tensor] = []
    for lid, nut in nu_per_layer.items():
        b = bounds_dict.get(lid)
        if b is None:
            continue
        lb = b.lb.flatten(start_dim=1)
        ub = b.ub.flatten(start_dim=1)
        if lb.shape[0] != kb:
            continue
        n = min(lb.shape[-1], nut.shape[-1])
        amb = (lb[:, :n] < 0) & (ub[:, :n] > 0)
        already = branch_batch.split_signs.get(lid) if branch_batch.split_signs else None
        if already is not None:
            amb &= already[:, 0, :n].to(device) == 0
        area = (-lb[:, :n] * ub[:, :n] / (ub[:, :n] - lb[:, :n]).clamp(min=1e-12)).clamp(min=0)
        nv = nut.reshape(kb, -1, nut.shape[-1])[:, :, :n].abs().sum(dim=1)
        sc = torch.where(amb, area * nv, torch.full_like(area, float("-inf")))
        cand_scores.append(sc)
        cand_layers.append(torch.full((kb, n), lid, device=device, dtype=torch.long))
        cand_neurons.append(
            torch.arange(n, device=device, dtype=torch.long).expand(kb, n)
        )
    if not cand_scores:
        return None
    return (
        torch.cat(cand_scores, dim=1),
        torch.cat(cand_layers, dim=1),
        torch.cat(cand_neurons, dim=1),
    )


def enumerate_unstable_candidates(
    branch_batch: SubproblemBatch,
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if bounds_dict is None or nu_per_layer is None:
        return []
    cand = _collect_neuron_candidates(branch_batch, bounds_dict, nu_per_layer)
    if cand is None:
        return []
    scores, layers, neurons = cand
    finite_mask = torch.isfinite(scores)
    total = int(finite_mask.sum().item())
    if total == 0 or (limit is not None and total > int(limit)):
        return []
    out: List[Dict[str, Any]] = []
    for lane in range(scores.shape[0]):
        flat_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for col in torch.where(finite_mask[lane])[0].tolist():
            lid = int(layers[lane, col].item())
            nidx = int(neurons[lane, col].item())
            b = bounds_dict.get(lid)
            if b is not None and lid not in flat_cache:
                flat_cache[lid] = (b.lb.flatten(start_dim=1), b.ub.flatten(start_dim=1))
            lb = float(flat_cache[lid][0][lane, nidx].item()) if lid in flat_cache else 0.0
            ub = float(flat_cache[lid][1][lane, nidx].item()) if lid in flat_cache else 0.0
            denom = ub - lb
            area = float(max(0.0, (-lb * ub) / denom)) if denom > 1e-12 else 0.0
            score = float(scores[lane, col].item())
            out.append({
                "lane": lane, "layer_id": lid, "neuron_idx": nidx,
                "score": score, "lb": lb, "ub": ub,
                "nu": (score / area) if area > 1e-12 else None, "area": area,
            })
    return out


def groups_to_tensors(
    groups: Dict[int, Any], batch: SubproblemBatch
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    # Duplicate pairs are sound but waste 2^k rows, so preserve order and dedupe.
    batch_size = batch.batch_size
    if len(groups) != batch_size:
        return None, None, 0
    deduped: Dict[int, List[Tuple[int, int]]] = {}
    for lane in range(batch_size):
        seen: Dict[Tuple[int, int], None] = {}
        for layer_id, neuron_index in groups.get(lane, []):
            seen.setdefault((int(layer_id), int(neuron_index)), None)
        deduped[lane] = list(seen)
    k_eff = min((len(entries) for entries in deduped.values()), default=0)
    if k_eff < 1:
        return None, None, 0
    device = batch.lb.device
    top_layers = torch.zeros(
        batch_size, k_eff, dtype=torch.long, device=device
    )
    top_neurons = torch.zeros(
        batch_size, k_eff, dtype=torch.long, device=device
    )
    for lane in range(batch_size):
        for bit, (layer_id, neuron_index) in enumerate(deduped[lane][:k_eff]):
            top_layers[lane, bit] = layer_id
            top_neurons[lane, bit] = neuron_index
    return top_layers, top_neurons, k_eff


def gain_tested_decision(
    branch_batch: SubproblemBatch,
    net: Net,
    assert_layer: Layer,
    config: BaBConfig,
    dual_config: DualConfig,
    keep_rows: Optional[torch.Tensor],
    root_bounds_dict: Optional[Dict[int, Bounds]],
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
    input_shape: tuple[int, ...],
    solve_dual: Callable[..., DualBatchResult],
    n_candidates: int = 3,
) -> Optional[SplitDecision]:
    """Pick each lane's split by measured child bounds, not by score proxy."""
    if bounds_dict is None or nu_per_layer is None:
        return None
    batch_size = branch_batch.batch_size
    device = branch_batch.lb.device

    candidates = _collect_neuron_candidates(
        branch_batch, bounds_dict, nu_per_layer
    )
    if candidates is None:
        return None
    all_scores, all_layers, all_neurons = candidates
    n_selected = min(n_candidates, all_scores.shape[1])
    top = torch.topk(all_scores, k=n_selected, dim=1).indices
    top_layers = all_layers.gather(1, top)
    top_neurons = all_neurons.gather(1, top)

    # Keep this probe builder separate: rows are lane-major, depth is unchanged,
    # and M comes from alpha (else 1) because solve_dual reads the probe depth.
    repeated_indices = torch.arange(batch_size, device=device).repeat_interleave(
        2 * n_selected
    )
    m_specs = 1
    if branch_batch.incremental_alpha:
        m_specs = int(
            next(iter(branch_batch.incremental_alpha.values())).shape[1]
        )
    signs = _gather_optional_dict(
        branch_batch.split_signs, repeated_indices
    ) or {}
    for layer_id_value in torch.unique(top_layers).tolist():
        layer_id = int(layer_id_value)
        n_neurons = _layer_neuron_count(net.by_id[layer_id])
        if layer_id not in signs:
            signs[layer_id] = torch.zeros(
                2 * n_selected * batch_size,
                m_specs,
                n_neurons,
                device=device,
                dtype=branch_batch.lb.dtype,
            )
        else:
            signs[layer_id] = signs[layer_id].clone()
        for lane in range(batch_size):
            for candidate in range(n_selected):
                if int(top_layers[lane, candidate]) != layer_id:
                    continue
                row = lane * 2 * n_selected + 2 * candidate
                neuron = int(top_neurons[lane, candidate])
                signs[layer_id][row, :, neuron] = 1.0
                signs[layer_id][row + 1, :, neuron] = -1.0

    probe = SubproblemBatch(
        lb=branch_batch.lb.index_select(0, repeated_indices),
        ub=branch_batch.ub.index_select(0, repeated_indices),
        depths=branch_batch.depths.index_select(0, repeated_indices),
        incremental_alpha=_gather_optional_dict(
            branch_batch.incremental_alpha, repeated_indices
        ),
        incremental_eta=_gather_optional_dict(
            branch_batch.incremental_eta, repeated_indices
        ),
        split_signs=signs,
    )
    n_probe = probe.batch_size
    probe_bounds = Bounds(
        probe.lb.reshape(n_probe, *input_shape) if input_shape else probe.lb,
        probe.ub.reshape(n_probe, *input_shape) if input_shape else probe.ub,
    )
    result = solve_dual(
        net=net,
        assert_layer=assert_layer,
        batched_bounds=probe_bounds,
        k_actual=n_probe,
        batch=probe,
        config=config,
        dual_config=dual_config,
        optimize=False,
        keep_rows=keep_rows,
        root_bounds_dict=root_bounds_dict,
    )
    child_lbs = (-result.solution.max_viol).view(
        batch_size, n_selected, 2
    )
    pair_gain = child_lbs.min(dim=2).values
    best_candidate = pair_gain.argmax(dim=1)
    lane_indices = torch.arange(batch_size, device=device)
    return SplitDecision(
        kind="neuron",
        layer_id=top_layers[lane_indices, best_candidate],
        neuron_idx=top_neurons[lane_indices, best_candidate],
    )


def _multi_split_from_decision(
    batch: SubproblemBatch,
    net: Net,
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
    k_levels: int,
) -> Optional[Tuple[SubproblemBatch, torch.Tensor]]:
    """Joint top-k neuron split: each lane emits all 2^k sign combinations.

    Verdict-boundary multi-neuron splitting — "Mining Verdict Boundaries for
    Neural Network Verification", Jiawei Ren, Guanqin Zhang, Zhenya Zhang,
    Yulei Sui, FM 2026. Candidates are scored by the BaBSR heuristic
    (``_collect_neuron_candidates``).

    The 2^k children exactly partition each lane's region (every selected
    neuron is constrained to >=0 or <=0 in both directions across the
    combination set), so replacing the lane by its children is sound. Joint
    splits are super-additive in bound gain versus greedy single splits.

    Lanes are grouped by how many finite candidates they actually have, so a
    single candidate-starved lane no longer collapses ``k_eff`` for the whole
    wave: lanes with >=2 candidates take the joint split, lanes with exactly one
    take a plain 2-child split, and the two groups are concatenated. Returns
    None only when some lane has no splittable neuron at all, since neuron
    branching cannot cover such a lane and the caller must fall back to the
    decision path (which can still split an input axis).
    """
    if bounds_dict is None or nu_per_layer is None:
        return None
    cand = _collect_neuron_candidates(batch, bounds_dict, nu_per_layer)
    if cand is None:
        return None
    all_scores, all_layers, all_neurons = cand
    finite_per_lane = torch.isfinite(all_scores).sum(dim=1)
    if int(finite_per_lane.min().item()) < 1:
        return None
    k_lane = finite_per_lane.clamp(max=k_levels)
    joint_mask = k_lane >= 2
    if not bool(joint_mask.any().item()):
        return None

    def _group(lanes: torch.Tensor, k: int) -> Tuple[SubproblemBatch, torch.Tensor]:
        sub = batch.select(lanes)
        scores = all_scores.index_select(0, lanes)
        top = torch.topk(scores, k=k, dim=1).indices
        children, local_parent = split_neurons(
            sub,
            net,
            all_layers.index_select(0, lanes).gather(1, top),
            all_neurons.index_select(0, lanes).gather(1, top),
            k,
        )
        return children, lanes.to(local_parent.device).index_select(0, local_parent)

    k_eff = min(k_levels, int(finite_per_lane[joint_mask].min().item()))
    lanes_joint = torch.where(joint_mask)[0]
    lanes_single = torch.where(~joint_mask)[0]
    joint_children, joint_parent = _group(lanes_joint, k_eff)
    if lanes_single.numel() == 0:
        return joint_children, joint_parent
    single_children, single_parent = _group(lanes_single, 1)
    return (
        joint_children.concat(single_children),
        torch.cat([joint_parent, single_parent]),
    )
