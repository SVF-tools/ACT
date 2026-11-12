"""BaB search records for the shared GUI observation channel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List

import torch

from act.back_end.bab.node import SubproblemBatch
from act.back_end.core import Bounds
from act.back_end.solver.solver_base import BatchLPSolution, SolveStatus
from act.pipeline.gui import act_gui as _gui
from act.pipeline.gui.act_gui import Record


@dataclass(frozen=True)
class BaBWave(Record):
    """One committed search wave with its solved nodes and new children."""

    seq: int
    wave: int
    nodes: tuple[dict[str, Any], ...]
    children: tuple[dict[str, Any], ...]
    frontier_size: int
    evicted_node_ids: tuple[int, ...]


@dataclass(frozen=True)
class BaBTerminal(Record):
    """Final search verdict, including exits before a wave can finish."""

    seq: int
    wave: int
    status: str
    reason: str
    frontier_size: int
    node_id: int | None = None
    parent_id: int | None = None
    depth: int | None = None


class BaBProgressEmitter:
    def __init__(self, callback: Callable[[dict[str, Any]], None] | None = None) -> None:
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
        event = {"seq": self.seq, **event}
        if _gui.enabled:
            if event["kind"] == "wave":
                _gui.emit(BaBWave(
                    event["seq"], event["wave"], tuple(event["nodes"]),
                    tuple(event["children"]), event["frontier_size"],
                    tuple(event["evicted_node_ids"]),
                ))
            else:
                _gui.emit(BaBTerminal(
                    event["seq"], event["wave"], event["status"], event["reason"],
                    event["frontier_size"], event["node_id"],
                    event["parent_id"], event["depth"],
                ))
        if self.callback is not None:
            self.callback(event)

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


__all__ = ["BaBWave", "BaBTerminal", "BaBProgressEmitter"]


# Self-check: python -m act.pipeline.gui.bab_gui
def _check_tree(waves: list[BaBWave]) -> None:
    """Every observed child must link to a node solved in its birth wave."""
    known: dict[int, tuple[int, int]] = {0: (-1, 0)}
    for wave in waves:
        current = {node["node_id"]: node for node in wave.nodes}
        for node in wave.nodes:
            assert node["node_id"] in known, (wave.wave, node)
            assert (node["parent_id"], node["depth"]) == known[node["node_id"]], (wave.wave, node)
        for child in wave.children:
            parent = current.get(child["parent_id"])
            assert parent is not None, (wave.wave, child)
            assert child["depth"] == parent["depth"] + 1, (wave.wave, child)
            assert child["node_id"] not in known, (wave.wave, child)
            known[child["node_id"]] = (child["parent_id"], child["depth"])
    print(f"BaB tree links: {len(known)} node IDs checked")

def _check_bab(output_dir: str) -> None:
    """Exercise BaB search records through the shared observer and legacy callback."""
    from dataclasses import replace
    import json

    from act.back_end.bab.bab import verify_bab_batched
    from act.back_end.net_factory import LAYER_TESTING_SPECS, NetFactory
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.config.config import BackendConfig, PipelineConfig

    torch.manual_seed(0)
    factory = NetFactory(
        BackendConfig.from_yaml().generation.net_factory,
        output_dir=output_dir, base_seed=0, write_manifest=False,
    )
    net = factory.create_network(
        "gui_bab_deep", LAYER_TESTING_SPECS["layer_testing_bab_deep"](),
    )
    pipeline = PipelineConfig.from_yaml()
    config = replace(
        pipeline.bab, solver_tier="dual", bounding="random",
        provenance_enabled=False, max_nodes=2, max_depth=12,
    )
    records: List[object] = []
    torch.manual_seed(123)
    with _gui.observe(records.append):
        result = verify_bab_batched(
            net, TorchLPSolver, config, max_batch_size=2,
            time_budget_s=10, dual_config=pipeline.dual,
        )
    waves = [record for record in records if isinstance(record, BaBWave)]
    terminals = [record for record in records if isinstance(record, BaBTerminal)]
    print(f"BaB GUI sink: result={result.status.name}, reason={terminals[0].reason}, "
          f"waves={len(waves)}, terminals={len(terminals)}")
    assert waves and len(terminals) == 1
    if result.status.name == "UNKNOWN":
        assert terminals[0].reason == result.metadata["reason"]
    assert all(node["node_id"] is not None for wave in waves for node in wave.nodes)
    assert [record.seq for record in [*waves, *terminals]] == list(range(1, len(waves) + 2))
    assert all(node["layer_bounds"] for wave in waves for node in wave.nodes)
    _check_tree(waves)

    # The callback remains usable by an application that already consumes dicts.
    callback_events: List[dict[str, Any]] = []
    torch.manual_seed(123)
    callback_result = verify_bab_batched(
        net, TorchLPSolver, config, max_batch_size=2,
        time_budget_s=10, dual_config=pipeline.dual,
        on_progress=callback_events.append,
    )
    assert callback_result.status == result.status
    assert [event["seq"] for event in callback_events] == list(range(1, len(callback_events) + 1))
    assert any(event["kind"] == "wave" for event in callback_events)
    json.dumps(callback_events, allow_nan=False)

    # Main-solver UNKNOWN does not say whether a node branched or hit max_depth.
    capped = replace(config, solver_tier="lp", max_nodes=20, max_depth=2, frontier_cap=2)
    records = []
    torch.manual_seed(123)
    with _gui.observe(records.append):
        capped_result = verify_bab_batched(
            net, TorchLPSolver, capped, max_batch_size=2,
            time_budget_s=10, dual_config=pipeline.dual,
        )
    capped_waves = [record for record in records if isinstance(record, BaBWave)]
    capped_terminals = [record for record in records if isinstance(record, BaBTerminal)]
    if capped_result.status.name == "UNKNOWN":
        assert len(capped_terminals) == 1
        assert capped_terminals[0].reason == capped_result.metadata["reason"]
    outcomes = {node["outcome"] for wave in capped_waves for node in wave.nodes}
    evicted = [node_id for wave in capped_waves for node_id in wave.evicted_node_ids]
    print(f"BaB capped sink: reason={capped_terminals[0].reason}, "
          f"outcomes={sorted(outcomes)}, evicted_ids={evicted}")
    assert {"branched", "retired_max_depth"} <= outcomes and evicted
    assert any(child["frontier_state"] == "evicted" for wave in capped_waves for child in wave.children)

    # A root proof has no wave, and a concrete counterexample exits mid-wave.
    records = []
    with _gui.observe(records.append):
        root_result = verify_bab_batched(
            net, TorchLPSolver, replace(config, solver_tier="dual_alpha_eta"),
            max_batch_size=2, time_budget_s=10, dual_config=pipeline.dual,
        )
    root_ends = [record for record in records if isinstance(record, BaBTerminal)]
    print(f"BaB root: result={root_result.status.name}, terminal_waves={[r.wave for r in root_ends]}")
    if root_result.metadata.get("resolved_by") == "root_presolve":
        assert len(root_ends) == 1 and root_ends[0].wave == 0

    spec = LAYER_TESTING_SPECS["layer_testing_bab_deep"]()
    spec["layers"][-1]["params"]["d"] = -100.0
    counterexample_net = factory.create_network("gui_bab_counterexample", spec)
    records = []
    with _gui.observe(records.append):
        ce_result = verify_bab_batched(
            counterexample_net, TorchLPSolver, config,
            max_batch_size=2, time_budget_s=10, dual_config=pipeline.dual,
        )
    ce_ends = [record for record in records if isinstance(record, BaBTerminal)]
    print(f"BaB counterexample: result={ce_result.status.name}, "
          f"terminal_nodes={[r.node_id for r in ce_ends]}")
    assert ce_result.status.name == "FALSIFIED" and len(ce_ends) == 1
    assert ce_ends[0].node_id is not None


def _self_check() -> None:
    import tempfile
    from act.util.device_manager import initialize_device

    initialize_device("cpu", "float64")
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        _check_bab(tmp)
    print("BaB GUI self-check passed")


if __name__ == "__main__":
    from act.pipeline.gui import bab_gui

    bab_gui._self_check()
