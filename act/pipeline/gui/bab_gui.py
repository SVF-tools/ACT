"""BaB search records for the shared GUI observation channel."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List

import torch

from act.back_end.solver.solver_base import SolveStatus
from act.pipeline.gui import act_gui as _gui
from act.pipeline.gui.act_gui import Option, OptionDefault, Record, validate_changes

if TYPE_CHECKING:
    from act.config.config import BaBConfig, DualConfig
    # act.back_end.bab imports this module (bab.py), so importing it here would make
    # ``import act.pipeline.gui.bab_gui`` circular; these names are annotations only.
    from act.back_end.bab.node import SubproblemBatch
    from act.back_end.core import Bounds
    from act.back_end.solver.solver_base import BatchLPSolution


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


# -- BaB settings -------------------------------------------------------------

BAB_OPTION_GROUPS: tuple[str, ...] = ("bounding", "search", "dual", "advanced")
# BaBConfig fields without a setting: the LLM probe needs an external model endpoint
# and API key, and ``method`` selects a BERT text-verification preset.
_UNTUNED_BAB_FIELDS: tuple[str, ...] = ("llm_probe_", "method")


@dataclass(frozen=True)
class BaBOptionDefault(OptionDefault):
    """A verify_bab_batched setting, its configured default, and its settings group."""

    group: str


def _bab_option_table() -> tuple[tuple[str, Option], ...]:
    """(group, Option) per setting. Names are PipelineConfig.from_yaml overrides
    (``bab_<BaBConfig field>``, ``dual_<DualConfig field>``) plus ``max_batch_size``."""
    from act.config.config import (
        NEURON_BRANCHING_METHODS, NO_REFINEMENT_MODE, VALID_BOUNDINGS, VALID_ROOT_BOUNDS_REUSE,
        VALID_SOLVER_TIERS,
    )

    def opt(group: str, name: str, kind: str, description: str, **limits: Any) -> tuple[str, Option]:
        return group, Option(name, kind, False, description, **limits)

    count, positive, unit = {"minimum": 0}, {"minimum": 1}, {"minimum": 0.0, "maximum": 1.0}
    rate = {"minimum": 0.0, "exclusive_minimum": True}
    return (
        opt("bounding", "bab_solver_tier", "choice",
            "how each wave is bounded: lp = interval bounds with the LP solver; dual = dual bounds with "
            "fixed ReLU slopes; dual_alpha = Adam-optimized slopes; dual_alpha_eta = optimized slopes and "
            "split multipliers", choices=VALID_SOLVER_TIERS),
        opt("search", "bab_max_depth", "int",
            "maximum tree depth; unresolved subproblems at it are dropped, which rules out CERTIFIED", **positive),
        opt("search", "bab_max_nodes", "int",
            "subproblems bounded in total; any left unproven make the verdict UNKNOWN", **positive),
        opt("search", "max_batch_size", "int",
            "subproblems bounded together per wave; 0 sizes the batch from free GPU memory", **count),
        opt("search", "bab_branching_method", "choice",
            "what to split: random and width split an input axis; the others split an unstable ReLU "
            "and need solver_tier dual_alpha or dual_alpha_eta",
            choices=("random", "width", *NEURON_BRANCHING_METHODS)),
        opt("search", "bab_bounding", "choice", "which pending subproblems the next wave expands",
            choices=VALID_BOUNDINGS),
        opt("search", "bab_bounding_depth_weight", "float", "depth_bound_blend: weight on normalized depth",
            minimum=0.0),
        opt("search", "bab_bounding_bound_weight", "float", "depth_bound_blend: weight on bound urgency",
            minimum=0.0),
        opt("search", "bab_top_k", "int",
            "subproblems popped per wave, 0 = unbounded; only for the order-ranked pools "
            "(depth_bound_blend, greedy, annealed, diverse_split_signs)", **count),
        opt("search", "bab_sa_cooling_rate", "float", "annealed: temperature decay per wave",
            maximum=1.0, **rate),
        opt("search", "bab_mcts_exploration", "float", "mcts: UCB1 exploration constant", minimum=0.0),
        opt("search", "bab_mcts_lambda", "float", "mcts: reward weight of depth against bound rank", **unit),
        opt("search", "bab_mcts_virtual_loss", "float", "mcts: virtual loss for batched selection",
            minimum=0.0),
        opt("search", "bab_frontier_cap", "int",
            "pending subproblems kept; above it the lowest-priority ones are evicted, which rules out "
            "CERTIFIED; 0 = no cap", **count),
        opt("search", "bab_input_split_fanout", "int", "pieces per input-axis split", minimum=2),
        opt("search", "bab_multi_split_levels", "int",
            "neurons split jointly per branching step (2^k children); needs a dual_alpha tier and neuron "
            "branching", **positive),
        opt("search", "bab_ramp_split_levels", "int", "joint splits per step while a convolutional frontier "
            "is narrow", **positive),
        opt("search", "bab_ramp_split_lanes", "int", "branching lanes below which ramp_split_levels applies; "
            "0 = off", **count),
        opt("search", "bab_presplit_levels", "int",
            "pre-split the root's top-k unstable neurons into 2^k sign combinations (dual tiers)", **count),
        opt("dual", "dual_n_iters", "int", "Adam iterations of the alpha/eta optimization", **count),
        opt("dual", "dual_lr_alpha", "float", "Adam learning rate for the ReLU slopes alpha", **rate),
        opt("dual", "dual_lr_beta", "float", "Adam learning rate for the split multipliers eta", **rate),
        opt("dual", "dual_lr_decay", "float", "learning-rate decay per iteration", maximum=1.0, **rate),
        opt("dual", "dual_stagnation_patience", "int",
            "stop after this many non-improving iterations; 0 disables", **count),
        opt("dual", "dual_stagnation_tol", "float", "improvement that resets the stagnation counter",
            minimum=0.0),
        opt("dual", "dual_stop_when_verified", "bool", "stop optimizing once every lane is certified"),
        opt("dual", "dual_max_time", "float", "optimization time limit in seconds; 0 disables", minimum=0.0),
        opt("dual", "dual_per_class_alpha", "bool", "separate slopes alpha per output class"),
        opt("dual", "dual_incremental_start_enabled", "bool", "start children from the parent's alpha/eta"),
        opt("dual", "dual_forward_lin_max_perturbed", "int",
            "perturbed input dims kept as linear forward bounds; above it interval-only", **count),
        opt("dual", "bab_child_n_iters", "int", "alpha/eta iterations on all-child batches; 0 = dual n_iters",
            **count),
        opt("dual", "bab_eta_only_children", "bool", "children keep the parent's alpha and optimize only eta"),
        opt("dual", "bab_root_bounds_reuse", "choice",
            "descendants reuse the root's forward bounds; none re-propagates per node",
            choices=VALID_ROOT_BOUNDS_REUSE),
        opt("dual", "bab_intermediate_refine", "choice",
            "backward refinement of the root's intermediate bounds: auto refines unusually wide layers",
            choices=(NO_REFINEMENT_MODE, "auto", "all")),
        opt("dual", "bab_intermediate_refine_ratio", "float",
            "auto refinement: refine layers wider than this multiple of the median width", **rate),
        opt("dual", "bab_intermediate_refine_max_tensor_mib", "float",
            "extra refinement tensor budget in MiB; 0 keeps the existing chunks", minimum=0.0),
        opt("dual", "bab_per_subproblem_refine", "choice",
            "per-subproblem refinement of intermediate bounds; needs root_bounds_reuse",
            choices=(NO_REFINEMENT_MODE, "tail", "all")),
        opt("dual", "bab_per_subproblem_refine_iters", "int",
            "Adam iterations of per-subproblem refinement; 0 = one fixed-slope backward pass", **count),
        opt("dual", "bab_per_subproblem_refine_rows_cap", "int",
            "refined neurons per layer and batch, widest first", **positive),
        opt("dual", "bab_climb_lin_support_refinement", "bool",
            "split_refresh_linear: keep only the sufficient winning halfspace supports"),
        opt("dual", "bab_presolve_time_fraction", "float",
            "root work window as a fraction of the BaB time budget; 0 = the remaining budget", **unit),
        opt("advanced", "bab_frontier_mem_safety", "float",
            "CUDA memory fraction for the frontier planner; 0 keeps the unplanned path", **unit),
        opt("advanced", "bab_frontier_offload", "bool", "allow host frontier storage when the planner is active"),
        opt("advanced", "bab_terminal_lp_enabled", "bool",
            "solve candidate-less lanes with independently checked terminal LP evidence"),
        opt("advanced", "bab_milp_escalation", "bool",
            "try a bounded exact ReLU MILP after unresolved root tiers, before branching"),
        opt("advanced", "bab_milp_max_variables", "int", "MILP variables, ReLU binaries included", **positive),
        opt("advanced", "bab_milp_max_constraints", "int", "MILP linear constraints", **positive),
        opt("advanced", "bab_milp_timeout", "float", "MILP seconds, also capped by the remaining budget", **rate),
        opt("advanced", "bab_climb_enabled", "bool", "CLIMB: replay query-local certificates and propagate cores"),
        opt("advanced", "bab_climb_terminal_cores", "bool", "CLIMB: publish root-box terminal LP cores"),
        opt("advanced", "bab_climb_coarsening_enabled", "bool",
            "CLIMB: coarsen literal budgets before admitting replay-certified cores"),
        opt("advanced", "bab_climb_propagation_enabled", "bool",
            "CLIMB: apply learned cores to pending and active subproblems"),
        opt("advanced", "bab_climb_merge_enabled", "bool", "CLIMB: merge complementary root-domain rows"),
        opt("advanced", "bab_climb_merge_min_proof_rate", "float", "CLIMB: proof rate that enables merging",
            **unit),
        opt("advanced", "bab_climb_complement_bounding", "bool",
            "CLIMB: bound each new core's deepest-decision complement on the root domain"),
        opt("advanced", "bab_climb_steer", "bool", "CLIMB: bias branching scores by core frequency"),
        opt("advanced", "bab_climb_presplit_propagation_enabled", "bool",
            "CLIMB: propagate and merge before splitting"),
        opt("advanced", "bab_climb_theta", "float", "CLIMB: slack fraction for the deletion budget", **unit),
        opt("advanced", "bab_climb_delta_abs", "float", "CLIMB: absolute deletion-budget margin", minimum=0.0),
        opt("advanced", "bab_climb_delta_rel", "float", "CLIMB: relative deletion-budget margin", minimum=0.0),
        opt("advanced", "bab_climb_recheck_k", "int", "CLIMB: dual lanes given an extra replay after coarsening",
            **count),
        opt("advanced", "bab_climb_max_cores", "int", "CLIMB: retained cores", **positive),
        opt("advanced", "bab_climb_propagate_core_chunk", "int", "CLIMB: cores per propagation chunk", **positive),
        opt("advanced", "bab_provenance_enabled", "bool",
            "track node and parent ids in the order-ranked pools; random bounding rejects it"),
        opt("advanced", "bab_auto_batch_safety", "float",
            "GPU memory fraction the automatic batch sizer may target", maximum=1.0, **rate),
        opt("advanced", "bab_auto_batch_cap", "int", "largest automatic batch (also the CPU fallback)",
            **positive),
        opt("advanced", "bab_auto_batch_floor", "int", "smallest automatic batch", **positive),
    )


def bab_options(config_path: str | Path | None = None) -> tuple[BaBOptionDefault, ...]:
    """Every tunable verify_bab_batched setting with its default: the pipeline YAML's
    ``verification.bab`` / ``verification.dual`` entry, else the dataclass default, and
    backend.yaml's ``bab_max_batch_size`` (``auto`` as 0)."""
    from act.config.config import BackendConfig, PipelineConfig

    pc = PipelineConfig.from_yaml(config_path)
    batch = BackendConfig.from_yaml().bab_max_batch_size

    def default(name: str) -> Any:
        if name == "max_batch_size":
            return 0 if batch == "auto" else batch
        section, field = name.split("_", 1)
        return getattr(pc.bab if section == "bab" else pc.dual, field)

    return tuple(BaBOptionDefault(option, default(option.name), group) for group, option in _bab_option_table())


def bab_run_config(
    changes: Mapping[str, Any], config_path: str | Path | None = None,
) -> tuple[BaBConfig, DualConfig, int | str]:
    """(BaBConfig, DualConfig, max_batch_size) for verify_bab_batched with ``changes``
    (bab_options names) over the defaults. ValueError (ConfigError included) for an
    unknown or invalid setting or a combination BaBConfig or DualConfig rejects."""
    from act.config.config import BackendConfig, PipelineConfig

    checked = validate_changes([option for _, option in _bab_option_table()], changes)
    batch = checked.pop("max_batch_size", None)
    pc = PipelineConfig.from_yaml(config_path, **checked)
    if batch is None:
        batch = BackendConfig.from_yaml().bab_max_batch_size
    return pc.bab, pc.dual, "auto" if batch == 0 else batch


__all__ = ["BaBWave", "BaBTerminal", "BaBProgressEmitter", "BaBOptionDefault", "BAB_OPTION_GROUPS",
           "bab_options", "bab_run_config"]


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


def _check_options() -> None:
    """bab_options covers every tuned BaBConfig/DualConfig field once, and bab_run_config applies them."""
    from dataclasses import fields

    from act.config.config import BaBConfig, DualConfig, PipelineConfig

    options = bab_options()
    names = [item.option.name for item in options]
    tuned = {f"bab_{f.name}" for f in fields(BaBConfig) if not f.name.startswith(_UNTUNED_BAB_FIELDS)}
    expected = tuned | {f"dual_{f.name}" for f in fields(DualConfig)} | {"max_batch_size"}
    assert len(names) == len(set(names)) and set(names) == expected, sorted(set(names) ^ expected)
    for item in options:
        assert item.group in BAB_OPTION_GROUPS, item
        item.option.validate(item.default)
    bab, dual, _ = bab_run_config({})
    pipeline = PipelineConfig.from_yaml()
    assert (bab, dual) == (pipeline.bab, pipeline.dual)
    bab, dual, batch = bab_run_config({"bab_solver_tier": "lp", "bab_max_nodes": 5, "dual_n_iters": 3,
                                       "max_batch_size": 0})
    assert (bab.solver_tier, bab.max_nodes, dual.n_iters, batch) == ("lp", 5, 3, "auto")
    for invalid in ({"bab_unknown": 1}, {"bab_max_nodes": 0}, {"bab_bounding": "random", "bab_top_k": 5}):
        try:
            bab_run_config(invalid)
        except ValueError:
            continue
        raise AssertionError(f"bab_run_config accepted {invalid}")
    print(f"BaB settings: {len(options)} options")


def _self_check() -> None:
    import tempfile
    from act.util.device_manager import initialize_device

    initialize_device("cpu", "float64")
    torch.set_num_threads(1)
    _check_options()
    with tempfile.TemporaryDirectory() as tmp:
        _check_bab(tmp)
    print("BaB GUI self-check passed")


if __name__ == "__main__":
    from act.pipeline.gui import bab_gui

    bab_gui._self_check()
