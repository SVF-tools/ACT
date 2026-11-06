"""Inspect BaB progress on a reproducible, generated network.

Run: .venv/bin/python -m act.back_end.bab.test_bab_progress
"""

from __future__ import annotations

from dataclasses import replace
import json
from tempfile import TemporaryDirectory

import torch

from act.back_end.bab.bab import verify_bab_batched
from act.back_end.net_factory import LAYER_TESTING_SPECS, NetFactory
from act.back_end.solver.solver_torchlp import TorchLPSolver
from act.config.config import BackendConfig, PipelineConfig
from act.util.device_manager import initialize_device


def announce_case(title: str) -> None:
    print(f"\n\033[1;31m{'=' * 16} {title} {'=' * 16}\033[0m", flush=True)


def make_network(output_dir: str):
    """Generate the same small ReLU network for every solver tier."""
    torch.manual_seed(0)
    factory = NetFactory(
        BackendConfig.from_yaml().generation.net_factory,
        output_dir=output_dir, base_seed=0, write_manifest=False,
    )
    return factory.create_network(
        "self_contained_deep", LAYER_TESTING_SPECS["layer_testing_bab_deep"](),
    )


def check_tree(events: list[dict]) -> None:
    """Report whether observed child IDs connect to their parent nodes."""
    known: dict[int, tuple[int, int]] = {0: (-1, 0)}
    issues: list[str] = []
    for event in events:
        if event["kind"] != "wave":
            continue
        current = {node["node_id"]: node for node in event["nodes"]}
        for node in event["nodes"]:
            if node["node_id"] not in known:
                issues.append(f"wave {event['wave']}: node {node['node_id']} has no birth record")
            elif (node["parent_id"], node["depth"]) != known[node["node_id"]]:
                issues.append(f"wave {event['wave']}: node {node['node_id']} has inconsistent parent/depth")
        for child in event["children"]:
            parent = current.get(child["parent_id"])
            if parent is None:
                issues.append(f"wave {event['wave']}: child {child['node_id']} has no parent in this wave")
            elif child["depth"] != parent["depth"] + 1:
                issues.append(f"wave {event['wave']}: child {child['node_id']} has inconsistent depth")
            if child["node_id"] in known:
                issues.append(f"wave {event['wave']}: child ID {child['node_id']} was already used")
            known[child["node_id"]] = (child["parent_id"], child["depth"])
    print(f"tree check: {len(known)} node IDs, {len(issues)} link issue(s)")
    for issue in issues:
        print(f"  {issue}")


def check_random_progress_ids(output_dir: str) -> None:
    """The GUI's random pool must expose IDs when progress is requested."""
    announce_case("RANDOM: progress node IDs")
    net = make_network(output_dir)
    pipeline = PipelineConfig.from_yaml()
    config = replace(
        pipeline.bab, solver_tier="lp", bounding="random",
        provenance_enabled=False, max_nodes=8, max_depth=3,
    )
    events: list[dict] = []
    torch.manual_seed(123)
    result = verify_bab_batched(
        net, TorchLPSolver, config, max_batch_size=4,
        time_budget_s=10, dual_config=pipeline.dual, on_progress=events.append,
    )
    missing = sum(
        node["node_id"] is None or node["parent_id"] is None
        for event in events if event["kind"] == "wave" for node in event["nodes"]
    ) + sum(
        child["node_id"] is None or child["parent_id"] is None
        for event in events if event["kind"] == "wave" for child in event["children"]
    )
    waves = [event for event in events if event["kind"] == "wave"]
    print(f"random progress: status={result.status.name}, "
          f"waves={len(waves)}, records_without_ids={missing}")
    assert waves, "the generated network should enter at least one BaB wave"
    assert missing == 0, "on_progress should provide IDs with the default random pool"
    check_tree(events)


def check_node_outcomes(output_dir: str) -> None:
    """A solved node may branch, retire at depth, or lose children to the cap."""
    announce_case("CAPPED: max_depth=2, frontier_cap=2")
    net = make_network(output_dir)
    pipeline = PipelineConfig.from_yaml()
    config = replace(
        pipeline.bab, solver_tier="lp", bounding="random",
        provenance_enabled=False, max_nodes=20, max_depth=2, frontier_cap=2,
    )
    events: list[dict] = []
    torch.manual_seed(123)
    verify_bab_batched(
        net, TorchLPSolver, config, max_batch_size=2,
        time_budget_s=10, dual_config=pipeline.dual, on_progress=events.append,
    )
    status_outcomes = [
        (node["solve_status"], node["outcome"])
        for event in events if event["kind"] == "wave" for node in event["nodes"]
    ]
    outcomes = [outcome for _, outcome in status_outcomes]
    children = [child for event in events if event["kind"] == "wave" for child in event["children"]]
    evicted = [node_id for event in events if event["kind"] == "wave" for node_id in event["evicted_node_ids"]]
    print(f"capped progress: status -> outcome={sorted(set(status_outcomes))}, "
          f"children_pending={sum(c['frontier_state'] == 'pending' for c in children)}, "
          f"children_evicted={sum(c['frontier_state'] == 'evicted' for c in children)}, "
          f"evicted_ids={evicted}")
    assert events and "branched" in outcomes and "retired_max_depth" in outcomes
    assert any(child["frontier_state"] == "evicted" for child in children)


def check_dual_stream(output_dir: str) -> None:
    """ACT hands complete, ordered dual snapshots to its observer."""
    announce_case("DUAL EVENTS: seq, terminal, full bounds")
    net = make_network(output_dir)
    pipeline = PipelineConfig.from_yaml()
    config = replace(
        pipeline.bab, solver_tier="dual", bounding="depth_bound_blend",
        provenance_enabled=True, max_nodes=2, max_depth=12,
    )
    events: list[dict] = []
    torch.manual_seed(123)
    result = verify_bab_batched(
        net, TorchLPSolver, config, max_batch_size=2,
        time_budget_s=10, dual_config=pipeline.dual, on_progress=events.append,
    )
    waves = [event for event in events if event["kind"] == "wave"]
    sizes = [len(json.dumps(event, allow_nan=False)) for event in events]
    print(f"dual stream: result={result.status.name}, "
          f"events={[(e['seq'], e['kind'], e['wave']) for e in events]}, "
          f"bytes={sizes}, summary_layers={len(waves[0]['nodes'][0]['layer_bound_summary']) if waves else 0}")
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert waves and events[-1]["kind"] == "terminal"
    assert all(node["layer_bounds"] for event in waves for node in event["nodes"])


def check_dual_counterexample(output_dir: str) -> None:
    """A dual counterexample must emit a terminal event before its early return."""
    announce_case("DUAL COUNTEREXAMPLE: early terminal event")
    torch.manual_seed(0)
    factory = NetFactory(
        BackendConfig.from_yaml().generation.net_factory,
        output_dir=output_dir, base_seed=0, write_manifest=False,
    )
    spec = LAYER_TESTING_SPECS["layer_testing_bab_deep"]()
    spec["layers"][-1]["params"]["d"] = -100.0
    net = factory.create_network("self_contained_counterexample", spec)
    pipeline = PipelineConfig.from_yaml()
    config = replace(
        pipeline.bab, solver_tier="dual", bounding="depth_bound_blend",
        provenance_enabled=True, max_nodes=4,
    )
    events: list[dict] = []
    result = verify_bab_batched(
        net, TorchLPSolver, config, max_batch_size=2,
        time_budget_s=10, dual_config=pipeline.dual, on_progress=events.append,
    )
    print(f"dual counterexample: result={result.status.name}, "
          f"events={[(e['seq'], e['kind'], e['wave'], e.get('node_id')) for e in events]}")
    assert result.status.name == "FALSIFIED"
    assert events[-1]["kind"] == "terminal" and events[-1]["node_id"] is not None


def run_case(solver_tier: str, output_dir: str) -> None:
    announce_case(f"{solver_tier.upper()}: max_depth=12")
    net = make_network(output_dir)
    pipeline = PipelineConfig.from_yaml()
    config = replace(
        pipeline.bab, solver_tier=solver_tier,
        bounding="depth_bound_blend", provenance_enabled=True,
        max_nodes=200, max_depth=12,
    )
    events: list[dict] = []

    def show_wave(event: dict) -> None:
        events.append(event)
        if event["kind"] == "terminal":
            print(f"\n{solver_tier} terminal seq={event['seq']} "
                  f"status={event['status']} reason={event['reason']}")
            return
        print(f"\n{solver_tier} wave {event['wave']}: "
              f"nodes={len(event['nodes'])}, frontier={event['frontier_size']}")
        for node in event["nodes"]:
            layers = node["layer_bounds"] or {}
            layer_ranges = {
                layer_id: (round(min(bounds["lb"]), 5), round(max(bounds["ub"]), 5))
                for layer_id, bounds in layers.items() if bounds["lb"] and bounds["ub"]
            }
            print(f"  node {node['node_id']} <- {node['parent_id']} "
                  f"depth={node['depth']} status={node['solve_status']} "
                  f"outcome={node['outcome']} "
                  f"bound={node['bound']:.6g} layer_ranges={layer_ranges}")
        for child in event["children"]:
            print(f"  child {child['node_id']} <- {child['parent_id']} "
                  f"depth={child['depth']} frontier={child['frontier_state']} "
                  f"branch={child['branch']}")
        if event["evicted_node_ids"]:
            print(f"  evicted IDs: {event['evicted_node_ids']}")

    torch.manual_seed(123)
    result = verify_bab_batched(
        net, TorchLPSolver, config, max_batch_size=8,
        time_budget_s=10, dual_config=pipeline.dual, on_progress=show_wave,
    )
    print(f"\n{solver_tier}: {result.status.name}, "
          f"iterations={result.metadata['iterations']}, "
          f"resolved_by={result.metadata.get('resolved_by')}")
    print(f"observed waves={sum(e['kind'] == 'wave' for e in events)}, "
          f"reported iterations={result.metadata['iterations']}")
    if result.metadata.get("resolved_by") == "root_presolve":
        assert len(events) == 1 and events[0]["kind"] == "terminal" and events[0]["wave"] == 0
    first_wave = next((event for event in events if event["kind"] == "wave"), None)
    if first_wave is not None:
        print(f"first wave event: {len(json.dumps(first_wave, allow_nan=False))} bytes")
    check_tree(events)


def main() -> None:
    initialize_device("cpu", "float64")
    torch.set_num_threads(1)
    with TemporaryDirectory(prefix="act-bab-progress-") as output_dir:
        check_random_progress_ids(output_dir)
        check_node_outcomes(output_dir)
        check_dual_stream(output_dir)
        check_dual_counterexample(output_dir)
        run_case("dual", output_dir)
        run_case("dual_alpha_eta", output_dir)


if __name__ == "__main__":
    main()
