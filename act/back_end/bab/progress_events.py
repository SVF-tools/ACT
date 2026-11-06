"""Read-only snapshots of committed BaB waves for optional observers."""

from __future__ import annotations

import math
from typing import Any, Callable

import torch

from act.back_end.bab.node import SubproblemBatch
from act.back_end.core import Bounds
from act.back_end.solver.solver_base import BatchLPSolution, SolveStatus


class BaBProgressEmitter:
    def __init__(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self.callback = callback
        self.seq = 0
        self.children: list[dict[str, Any]] = []
        self.outcomes: dict[int, str] = {}
        self.evicted_node_ids: list[int] = []

    def begin_wave(self) -> None:
        # Also discards records from a wave rolled back after an OOM.
        self.children.clear()
        self.outcomes.clear()
        self.evicted_node_ids.clear()

    def mark_outcome(self, batch: SubproblemBatch, outcome: str) -> None:
        if batch.node_id is not None:
            self.outcomes.update({int(node_id): outcome for node_id in batch.node_id.tolist()})

    def record_eviction(self, before: list[int], after: list[int]) -> None:
        kept = set(after)
        self.evicted_node_ids.extend(node_id for node_id in before if node_id not in kept)

    def _emit(self, event: dict[str, Any]) -> None:
        self.seq += 1
        self.callback({"seq": self.seq, **event})

    def finish_terminal(
        self, *, wave: int, status: str, reason: str,
        frontier_size: int, node_id: int | None = None,
        parent_id: int | None = None, depth: int | None = None,
    ) -> None:
        self._emit({
            "kind": "terminal", "wave": wave, "status": status,
            "reason": reason, "node_id": node_id,
            "parent_id": parent_id, "depth": depth,
            "frontier_size": frontier_size,
        })

    def record_children(
        self, parents: SubproblemBatch, children: SubproblemBatch,
        parent_index: torch.Tensor,
    ) -> None:
        for child_row, parent_row in enumerate(parent_index.tolist()):
            changed = torch.where(
                (children.lb[child_row] != parents.lb[parent_row])
                | (children.ub[child_row] != parents.ub[parent_row])
            )[0].tolist()
            if changed:
                branch: dict[str, Any] = {
                    "kind": "input_axis",
                    "intervals": [
                        {"axis": axis, "lb": float(children.lb[child_row, axis]),
                         "ub": float(children.ub[child_row, axis])}
                        for axis in changed
                    ],
                }
            else:
                phases = []
                for layer_id, signs in (children.split_signs or {}).items():
                    old_signs = (parents.split_signs or {}).get(layer_id)
                    old = old_signs[parent_row] if old_signs is not None else torch.zeros_like(signs[child_row])
                    for spec_row, neuron in (signs[child_row] != old).nonzero().tolist():
                        phases.append({
                            "layer_id": int(layer_id), "spec_row": int(spec_row),
                            "neuron": int(neuron),
                            "sign": int(signs[child_row, spec_row, neuron]),
                        })
                branch = {"kind": "neuron", "phases": phases} if phases else {"kind": "unknown"}
            self.children.append({
                "node_id": int(children.node_id[child_row]) if children.node_id is not None else None,
                "parent_id": int(children.parent_id[child_row]) if children.parent_id is not None else None,
                "depth": int(children.depths[child_row]),
                "branch": branch,
            })

    def finish_wave(
        self, wave: int, batch: SubproblemBatch, solution: BatchLPSolution,
        node_lower_bound: torch.Tensor, layer_bounds: dict[int, Bounds] | None,
        solver_tier: str, frontier_size: int, frontier_node_ids: list[int],
    ) -> None:
        nodes = []
        branched = {child["parent_id"] for child in self.children}
        for row, status in enumerate(solution.statuses):
            per_layer = None if layer_bounds is None else {
                int(layer_id): {
                    "lb": bounds.lb[row].detach().cpu().reshape(-1).tolist(),
                    "ub": bounds.ub[row].detach().cpu().reshape(-1).tolist(),
                }
                for layer_id, bounds in layer_bounds.items()
            }
            layer_summary = None if layer_bounds is None else {
                int(layer_id): {
                    "lb_min": _finite_or_none(bounds.lb[row].min()) if bounds.lb[row].numel() else None,
                    "ub_max": _finite_or_none(bounds.ub[row].max()) if bounds.ub[row].numel() else None,
                    "neuron_count": bounds.lb[row].numel(),
                }
                for layer_id, bounds in layer_bounds.items()
            }
            node_id = int(batch.node_id[row]) if batch.node_id is not None else None
            nodes.append({
                "node_id": node_id,
                "parent_id": int(batch.parent_id[row]) if batch.parent_id is not None else None,
                "depth": int(batch.depths[row]),
                "solve_status": status,
                "resolved_by_main_solver": status == SolveStatus.UNSAT,
                "outcome": self.outcomes.get(
                    node_id,
                    "proved_main" if status == SolveStatus.UNSAT else
                    "branched" if node_id in branched else "unresolved",
                ),
                "bound": _finite_or_none(node_lower_bound[row]),
                "bound_kind": (
                    "certified_node_lower_bound" if solver_tier != "lp"
                    else "negative_max_constraint_violation"
                ),
                "layer_bounds": per_layer,
                "layer_bound_summary": layer_summary,
            })
        frontier_ids = set(frontier_node_ids)
        children = [
            {**child, "frontier_state": (
                "pending" if child["node_id"] in frontier_ids else "evicted"
            )}
            for child in self.children
        ]
        self._emit({
            "kind": "wave", "wave": wave, "nodes": nodes,
            "children": children, "frontier_size": frontier_size,
            "evicted_node_ids": list(self.evicted_node_ids),
        })


def _finite_or_none(value: torch.Tensor) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None
