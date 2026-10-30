#!/usr/bin/env python3
"""
ACT Back-End Command-Line Interface.

Provides CLI tools for core verification operations:
- Network verification (single-shot and branch-and-bound)
- Network factory (generate example networks from YAML)
- Network serialization (save/load ACT Net structures)
- Analysis and constraint inspection

Exit status (--verify):
  0  every lane received a verdict (CERTIFIED / FALSIFIED / UNKNOWN / TIMEOUT)
  1  error, no verdict, or a network the selected solver cannot handle
  2  invalid configuration (ConfigError), reported as "❌ <network>: <message>"

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

import argparse
from dataclasses import fields, replace
import datetime
import glob
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union, cast, get_args, get_origin, get_type_hints

from act.config.config import ConfigError, GurobiConfig, TorchLPConfig, VALID_BERT_METHODS, VALID_BOUNDINGS, VALID_ROOT_BOUNDS_REUSE, VALID_SOLVER_TIERS, _VALID_SOLVERS
from act.back_end.layer_schema import LayerKind
from act.front_end.specs import OutKind
from act.util.cli_utils import add_device_args, initialize_from_args
from act.util.format_utils import rule


_TF_MODES: tuple[str, ...] = ("interval", "hybridz")
_SOLVERS: tuple[str, ...] = tuple(sorted(_VALID_SOLVERS))
logger = logging.getLogger(__name__)


def _strip_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is Union:
        args = [arg for arg in get_args(tp) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _parse_config_value(tp: Any):
    base = _strip_optional(tp)
    origin = get_origin(base)
    if base in (int, float, str):
        return base
    if origin is Union and get_args(base) == (int, str):
        def _parse_batch_size(raw: str) -> Union[int, str]:
            return "auto" if raw == "auto" else int(raw)
        return _parse_batch_size
    if origin in (list, List):
        def _parse_list(raw: str) -> list[Any]:
            if raw.strip().startswith("["):
                value = json.loads(raw)
                if not isinstance(value, list):
                    raise argparse.ArgumentTypeError("expected a JSON list")
                return value
            return [item for item in raw.split(",") if item]
        return _parse_list
    return str


def _add_dataclass_config_args(parser: argparse.ArgumentParser) -> None:
    """Expose all BackendConfig dataclass fields without hand-maintained drift."""
    from act.config.config import BackendConfig, BaBConfig, DualConfig, GenerationConfig, GurobiConfig, HybridZConfig, TorchLPConfig

    existing_options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    # Dests _collect_backend_overrides reads. A key aliased to a differently
    # spelled explicit flag (bab_max_nodes -> --bab-max-subproblems) is absent,
    # so a generated flag for it would set a dest nothing ever reads.
    read_dests = {attr for _, attr, *_ in _BACKEND_OVERRIDE_SPEC}

    def add_group(cls: type[Any], title: str, flag_prefix: str, dest_prefix: str, skip: set[str]) -> None:
        group = parser.add_argument_group(title)
        type_hints = get_type_hints(cls)
        for fld in fields(cls):
            if fld.name in skip:
                continue
            flag = f"--{flag_prefix}{fld.name.replace('_', '-')}"
            dest = f"{dest_prefix}{fld.name}"
            if dest not in read_dests:
                continue
            field_type = type_hints.get(fld.name, fld.type)
            base = _strip_optional(field_type)
            help_text = f"Override config field {cls.__name__}.{fld.name} (default: from config.yaml)"
            if flag in existing_options:
                no_flag = f"--no-{flag[2:]}"
                if base is bool and no_flag not in existing_options:
                    group.add_argument(
                        no_flag,
                        action="store_false",
                        default=None,
                        dest=dest,
                        help=help_text,
                    )
                    existing_options.add(no_flag)
                continue
            if base is bool:
                group.add_argument(
                    flag,
                    action=argparse.BooleanOptionalAction,
                    default=None,
                    dest=dest,
                    help=help_text,
                )
                existing_options.add(flag)
                existing_options.add(f"--no-{flag[2:]}")
            else:
                group.add_argument(
                    flag,
                    type=_parse_config_value(field_type),
                    default=None,
                    dest=dest,
                    help=help_text,
                )
                existing_options.add(flag)

    add_group(
        BackendConfig,
        "Backend Config Overrides (generated)",
        "",
        "",
        {"bab", "generation", "hybridz", "gurobi", "torchlp", "dual"},
    )
    add_group(BaBConfig, "BaB Config Overrides (generated)", "bab-", "bab_", set())
    add_group(DualConfig, "Dual Config Overrides (generated)", "dual-", "dual_", set())
    add_group(GenerationConfig, "Generation Config Overrides (generated)", "gen-", "gen_", {"net_factory"})
    add_group(HybridZConfig, "HybridZ Config Overrides (generated)", "hz-", "hybridz_", set())
    add_group(GurobiConfig, "Gurobi Config Overrides (generated)", "gurobi-", "gurobi_", set())
    add_group(TorchLPConfig, "TorchLP Config Overrides (generated)", "torchlp-", "torchlp_", set())


# Backend YAML sub-section (== the BackendConfig nested-dataclass field name) ->
# the flat prefix its fields carry in the CLI/override namespace. Single source of
# truth, shared by the override-key derivation below AND the config-parity check
# (act/config/check_parity.py). Adding a field to an existing sub-config needs no
# change here; only a brand-new sub-config does.
_BACKEND_SUBCONFIG_PREFIX: dict[str, str] = {
    "bab": "bab_",
    "generation": "gen_",
    "hybridz": "hybridz_",
    "gurobi": "gurobi_",
    "torchlp": "torchlp_",
    "dual": "dual_",
}


def _backend_override_keys_from_dataclasses() -> set[str]:
    from act.config.config import BackendConfig

    hints = get_type_hints(BackendConfig)
    keys: set[str] = set()
    for fld in fields(BackendConfig):
        prefix = _BACKEND_SUBCONFIG_PREFIX.get(fld.name)
        if prefix is None:
            keys.add(fld.name)
        else:
            sub_type = _strip_optional(hints.get(fld.name, fld.type))
            keys.update(f"{prefix}{sub.name}" for sub in fields(sub_type))
    return keys


class _SkipUnsupported(NamedTuple):
    """Tagged-union result for nets the active TF cannot verify.

    Replaces an earlier ``"SKIP_UNSUPPORTED: ..."`` string-sentinel encoding
    that risked false-negative skips if any unrelated ``load_net_from_file``
    exception happened to start with the same prefix. ``isinstance(err,
    _SkipUnsupported)`` is unambiguous.
    """
    tf_name: str
    kinds: tuple[str, ...]


def _make_solver(
    solver_name: str,
    torchlp_config: Optional[TorchLPConfig] = None,
    gurobi_config: Optional[GurobiConfig] = None,
):
    """LP-cascade solver factory (gurobi / torchlp / auto). Dual is routed
    separately via ``is_dual_solver_active`` since it implements
    ``compute_certified_bound``, not ``solve_batch``.
    """
    from act.back_end.solver.solver_torchlp import TorchLPSolver

    if solver_name == "gurobi":
        from act.back_end.solver.solver_gurobi import GurobiSolver

        return GurobiSolver(config=gurobi_config)
    if solver_name == "torchlp":
        return TorchLPSolver(config=torchlp_config)
    # "auto": Gurobi when gurobipy is importable, otherwise TorchLP
    from act.back_end.solver.solver_gurobi import GurobiSolver, is_gurobi_available

    if is_gurobi_available():
        return GurobiSolver(config=gurobi_config)
    return TorchLPSolver(config=torchlp_config)


def explicit_bab_fields(args: argparse.Namespace) -> set[str]:
    """BaB fields the user pinned on the command line.

    Every BaB override flag parses with ``default=None``, so a non-None value is
    proof the user typed it. A named preset must yield on exactly those fields.
    """
    from act.config.config import BaBConfig

    bab_fields = {fld.name for fld in fields(BaBConfig)}
    destinations = {
        key[4:]: attr
        for key, attr, *_ in _BACKEND_OVERRIDE_SPEC
        if key.startswith("bab_") and key[4:] in bab_fields
    }
    return {
        field_name
        for field_name, destination in destinations.items()
        if getattr(args, destination, None) is not None
    }


def _resolve_bab_config(
    backend_cfg: Any,
    pinned_fields: set[str],
    *,
    is_dual: bool,
) -> Any:
    """Apply the selected BaB preset below explicit CLI overrides."""
    preset_name = backend_cfg.bab_preset or ("dual" if is_dual else None)
    if preset_name is None:
        return backend_cfg.bab
    preset_values = {
        field_name: value
        for field_name, value in backend_cfg.bab_preset_values(preset_name).items()
        if field_name not in pinned_fields
    }
    return replace(backend_cfg.bab, **preset_values) if preset_values else backend_cfg.bab


def _verify_net(
    net: Any,
    backend_cfg,
    explicit_bab_fields: Optional[set[str]] = None,
    *,
    deadline: Optional[float] = None,
) -> tuple[list[Any], Optional[_SkipUnsupported], Optional[int]]:
    """[BATCHED-API] Verify one already-constructed Net via the refinement cascade.

    Named BaB presets yield to fields pinned by CLI flags. All stages share
    one deadline; tensor propagation is not preemptible, but its elapsed time
    is deducted before launching any refinement.

    Returns ``(results, err, n_layers)`` where ``err`` is one of:
      * ``None`` on success
      * ``_SkipUnsupported(tf_name, kinds)`` when the active TF cannot handle
        the net (unsupported layer kinds and/or unsupported ASSERT spec).
        Reported distinctly from verifier bugs, but still a nonzero outcome
        because no lane received a verdict.

    Raises ``ConfigError`` when the resolved configuration is invalid (BaB
    preset application, CLIMB prerequisites) so the caller can exit with 2.

    Tier 1 — interval (verify_once): runs while budget remains; certifies or
              falsifies via pure-tensor bounds propagation.
    Tier 2 — LP-batched (verify_lp_batched): runs on UNKNOWN lanes when
              backend_cfg.lp_enabled is True AND active TF propagates LP
              constraints. Skipped under DualTF (see soundness note below).
    Optional MILP (solve_milp): runs after root LP and before BaB if root
              results remain UNKNOWN and bab.milp_escalation is enabled.
              Skipped under HybridZ; unsupported exports remain UNKNOWN.
              Its budget is capped by milp_timeout and the same shared
              deadline; BaB receives only the remaining time.
    Tier 3 — BaB (verify_bab_batched): runs on remaining UNKNOWN lanes when
              backend_cfg.bab_enabled is True AND active TF propagates LP
              constraints. bab_max_batch_size=1 disables K-batching.
    """
    from act.back_end.bab.violation import clear_violation_check_module_cache
    from act.back_end.transfer_functions import (
        ensure_active_tf,
        is_dual_solver_active,
        is_hybridz_solver_active,
    )
    from act.back_end.verifier import (
        gather_input_spec_layers, seed_from_input_specs, verify_once, verify_lp_batched,
    )
    from act.util.stats import VerifyResult, VerifyStatus

    stop_at = time.monotonic() + backend_cfg.timeout if deadline is None else deadline
    clear_violation_check_module_cache()

    n_layers = len(net.layers)

    active_tf = ensure_active_tf("interval")
    is_dual = is_dual_solver_active()
    is_hybridz = is_hybridz_solver_active()

    # Pre-filter helper: DualTF is the registry holder for dual backward
    # kernels; under --solver dual the kind-support check must go through
    # DualTF, not active_tf (which is still IntervalTF for forward bounds).
    # Under non-dual modes the active TF is the authority.
    if is_dual:
        from act.back_end.dual_tf.dual_tf import DualTF
        kind_authority = DualTF()
        authority_name = "DualSolver"
    else:
        kind_authority = active_tf
        authority_name = active_tf.name

    unsupported_kinds = sorted(
        {L.kind for L in net.layers if not kind_authority.supports_layer(L.kind)}
    )
    # OutputSpec owns the supported property kinds. DualSolver evaluates
    # UNSAFE_LINEAR by proving strict escape from at least one unsafe row.
    blocking = tuple(unsupported_kinds)
    if blocking:
        return [], _SkipUnsupported(tf_name=authority_name, kinds=blocking), n_layers

    remaining = stop_at - time.monotonic()
    if remaining <= 0.0:
        batch_size = seed_from_input_specs(gather_input_spec_layers(net)).lb.shape[0]
        return [VerifyResult(VerifyStatus.TIMEOUT) for _ in range(batch_size)], None, n_layers
    hz_timeout = remaining
    hz_tolerance = None
    if is_hybridz:
        hz_timeout = min(backend_cfg.hybridz.timeout or remaining, remaining)
        hz_tolerance = backend_cfg.hybridz.tolerance
    results: List[Any] = list(
        verify_once(
            net=net,
            timelimit=hz_timeout,
            hybridz_tolerance=hz_tolerance,
        )
    )

    any_unknown = any(r.status == VerifyStatus.UNKNOWN for r in results)

    # Dual does not produce LP-feed constraints; HybridZ UNKNOWN is the
    # final result of the selected pure solver and must not be rescued.
    remaining = stop_at - time.monotonic()
    if any_unknown and backend_cfg.lp_enabled and not (is_dual or is_hybridz) and remaining > 0.0:
        try:
            lp_results = verify_lp_batched(
                net,
            solver_factory=lambda: _make_solver(
                backend_cfg.solver, backend_cfg.torchlp, backend_cfg.gurobi
            ),
                timelimit=remaining,
            )
            results = [
                lp_results[i] if results[i].status == VerifyStatus.UNKNOWN else results[i]
                for i in range(len(results))
            ]
            any_unknown = any(r.status == VerifyStatus.UNKNOWN for r in results)
        except NotImplementedError as e:
            # cons_exportor.export_to_batch_problem lacks an LP encoding
            # for one of this net's layer kinds (AVGPOOL2D / MAXPOOL2D /
            # GELU / explicit-reject tags like max:/min:/div:/clip:).
            # Graceful degradation: keep the Tier-1 UNKNOWN result rather
            # than fail the net — LP is a refinement, its absence does
            # not invalidate prior tiers. Reraise unrelated NIE so real
            # missing-implementation bugs still surface as ERROR.
            if "export_to_batch_problem" not in str(e):
                raise

    bab_cfg = None
    milp_metadata: dict[int, dict[str, Any]] = {}
    if any_unknown and not is_hybridz and (
        backend_cfg.bab.milp_escalation
        or (backend_cfg.bab_enabled and time.monotonic() < stop_at)
    ):
        bab_cfg = _resolve_bab_config(
            backend_cfg, explicit_bab_fields or set(), is_dual=is_dual,
        )
        if bab_cfg.milp_escalation:
            # Optional import: default commands do not load either numerical
            # escalation owner or gurobipy through this gate.
            from act.back_end.solver.milp_escalate import solve_milp

            attempt = solve_milp(net, bab_cfg, deadline=stop_at)
            for i, result in enumerate(results):
                if result.status == VerifyStatus.UNKNOWN:
                    milp_metadata[i] = attempt.metadata["milp_escalation"]
                    if len(results) == 1:
                        results[i] = attempt
            any_unknown = any(r.status == VerifyStatus.UNKNOWN for r in results)

    if any_unknown and backend_cfg.bab_enabled and not is_hybridz and time.monotonic() < stop_at:
        # verify_bab_batched operates on a single-instance (B=1) net and
        # returns one VerifyResult. For multi-sample nets we slice per-lane
        # and dispatch one BaB call per still-UNKNOWN sample.
        from act.back_end.bab.bab import verify_bab_batched as _vbb
        from act.back_end.verifier import slice_net_to_sample

        try:
            for i, result in enumerate(results):
                if result.status != VerifyStatus.UNKNOWN:
                    continue
                lane_net = slice_net_to_sample(net, i)
                remaining = stop_at - time.monotonic()
                if remaining <= 0.0:
                    break
                results[i] = _vbb(
                    lane_net,
                    solver_factory=lambda: _make_solver(
                        backend_cfg.solver, backend_cfg.torchlp, backend_cfg.gurobi
                    ),
                    config=bab_cfg,
                    max_batch_size=backend_cfg.bab_max_batch_size,
                    time_budget_s=remaining,
                    dual_config=backend_cfg.dual,
                )
        except NotImplementedError as e:
            # Same graceful-degradation contract as the Tier-2 LP gate:
            # BaB also runs through cons_exportor (via setup_and_solve_batch
            # in bab.py) and hits the same "unsupported tag" failure on
            # AVGPOOL2D / MAXPOOL2D / GELU / etc. Keep Tier-1 results;
            # reraise unrelated NIE so genuine bugs surface as ERROR.
            if "export_to_batch_problem" not in str(e):
                raise

    if time.monotonic() >= stop_at:
        results = [
            VerifyResult(VerifyStatus.TIMEOUT, metadata=dict(result.metadata))
            if result.status == VerifyStatus.UNKNOWN else result
            for result in results
        ]
    for i, metadata in milp_metadata.items():
        results[i].metadata["milp_escalation"] = metadata
    return results, None, n_layers


def _verify_one_net(
    net_path: str,
    backend_cfg,
    explicit_bab_fields: Optional[set[str]] = None,
) -> tuple[list[Any], Optional[_SkipUnsupported], Optional[int]]:
    """Load one serialized Net and verify it through the shared cascade."""
    from act.back_end.serialization.serialization import load_net_from_file

    net = load_net_from_file(net_path, target_device=backend_cfg.device)
    return _verify_net(net, backend_cfg, explicit_bab_fields)


def _report_config_error(label: str, err: ConfigError) -> int:
    """Print *err* in the CLI error format and return exit status 2."""
    print(f"❌ {label}: {err}")
    return 2


def run_vnnlib_verification(args, backend_cfg):
    """Verify every synthesized ONNX/VNNLIB disjunct through the shared cascade."""
    from act.back_end.serialization.serialization import save_net_to_file
    from act.front_end.vnnlib_loader.vnncomp_instance import load_vnncomp_instance
    from act.pipeline.verification.instance import DisjunctResults, convert_model
    from act.util.stats import VerifyResult, VerifyStatus

    deadline = time.monotonic() + backend_cfg.timeout
    instance = load_vnncomp_instance(args.onnx, args.vnnlib, merge_relus=True)
    if instance.merged_relus:
        print(f"Merged {instance.merged_relus} split-ReLU neurons")

    aggregate = DisjunctResults(len(instance.models))
    exit_status = 0
    pinned_bab_fields = explicit_bab_fields(args)

    for index, budget in aggregate.pending(deadline):
        assert budget is not None
        model_deadline = min(deadline, time.monotonic() + budget)
        net = convert_model(instance.models[index])
        remaining = model_deadline - time.monotonic()
        if remaining <= 0.0:
            aggregate.record((VerifyResult(VerifyStatus.TIMEOUT),))
            continue
        save_path = aggregate.save_path(args.save_net, index)
        if save_path is not None:
            save_net_to_file(net, save_path)
        try:
            results, err, n_layers = _verify_net(
                net, backend_cfg, pinned_bab_fields, deadline=model_deadline,
            )
        except ConfigError as e:
            return _report_config_error(f"disjunct {index}", e)

        print(f"Disjunct {index}:")
        exit_status = max(exit_status, _report_verification_results(
            f"disjunct {index}", results, err, n_layers, backend_cfg,
        ))
        aggregate.record(results)
        if aggregate.counterexample is not None:
            break

    print(f"VNNLIB verdict: {aggregate.verdict.upper()}")
    return exit_status


def _report_verification_results(
    label: str,
    results: list[Any],
    err: Optional[_SkipUnsupported],
    n_layers: Optional[int],
    backend_cfg: Any,
) -> int:
    """Report shared Net-verification results and return the CLI exit status."""
    from act.util.stats import VerifyStatus

    if err is not None:
        print(f"⏭️  {label}: {err.tf_name} cannot handle: {','.join(err.kinds)}")
        return 1
    assert results, f"{label}: no verdict"
    assert n_layers is not None
    print(f"Loaded {n_layers}-layer net; solver={backend_cfg.solver}")
    valid_outcomes = (
        VerifyStatus.CERTIFIED,
        VerifyStatus.FALSIFIED,
        VerifyStatus.UNKNOWN,
        VerifyStatus.TIMEOUT,
    )
    multi = len(results) > 1
    for index, result in enumerate(results):
        prefix = f"Sample {index}: " if multi else f"Lane {index}: "
        print(f"{prefix}{result.status}")
        if backend_cfg.verbose and result.metadata:
            for key, value in result.metadata.items():
                print(f"  {key}: {value}")
    return 0 if all(result.status in valid_outcomes for result in results) else 1


def run_verification(args, backend_cfg):
    """Run verification on a network using *backend_cfg*.

    Returns 0 when every lane received a verdict, 1 on error / no verdict /
    unsupported network, and 2 on invalid configuration.
    """
    pinned_bab_fields = explicit_bab_fields(args)
    try:
        results, err, n_layers = _verify_one_net(
            args.network, backend_cfg, pinned_bab_fields
        )
    except ConfigError as e:
        return _report_config_error(args.network, e)
    return _report_verification_results(args.network, results, err, n_layers, backend_cfg)


def run_query_verification(args, backend_cfg):
    """Build one indexed compact-BERT query and verify its converted Net."""
    from act.front_end.bert_loader.query_index import (
        QueryIndexError,
        build_query_model,
        read_query,
    )
    from act.pipeline.verification.instance import convert_model

    label = f"{args.query_index} query {args.query_id}"
    try:
        query = read_query(args.query_index, args.query_id)
        net = convert_model(build_query_model(query), args.save_net)
        results, err, n_layers = _verify_net(
            net, backend_cfg, explicit_bab_fields(args)
        )
    except ConfigError as error:
        return _report_config_error(label, error)
    except (QueryIndexError, FileNotFoundError, ValueError) as error:
        print(f"❌ {label}: {error}")
        return 1
    return _report_verification_results(label, results, err, n_layers, backend_cfg)


def run_network_factory(args, backend_cfg):
    """Generate example networks using TF-aware NetFactory."""
    print(f"\n{rule()}")
    print(f"ACT NETWORK FACTORY")
    print(f"{rule()}\n")

    from act.back_end.net_factory import NetFactory

    gen = backend_cfg.generation

    if gen.tf_targets:
        print(f"TF targets: {gen.tf_targets} (mode: {gen.registry_mode})")
    print(f"Output: {gen.output_dir}")
    print(f"Instances: {gen.num_instances}, Seed: {gen.base_seed}")
    print()

    factory = NetFactory(
        config=gen.net_factory,
        output_dir=gen.output_dir,
        base_seed=gen.base_seed,
        num_instances=gen.num_instances,
        name_prefix=gen.name_prefix,
        tf_targets=gen.tf_targets,
        registry_mode=gen.registry_mode,
        write_manifest=gen.write_manifest,
    )
    factory.generate()
    print(f"\n{rule()}")
    print(f"✓ Network generation complete")
    print(f"{rule()}\n")

    return 0


def run_network_info(args):
    """Display information about a network."""
    print(f"\n{rule()}")
    print(f"NETWORK INFORMATION")
    print(f"{rule()}\n")

    from act.back_end.serialization.serialization import load_net_from_file

    print(f"Loading network from: {args.network}\n")
    net = load_net_from_file(args.network)

    # Basic info
    print(f"Network: {Path(args.network).stem}")
    print(f"Total layers: {len(net.layers)}")
    print(f"Predecessors: {sum(len(p) for p in net.preds.values())} edges")
    print(f"Successors: {sum(len(s) for s in net.succs.values())} edges")

    # Layer breakdown by kind
    layer_kinds = {}
    for layer in net.layers:
        kind = layer.kind
        layer_kinds[kind] = layer_kinds.get(kind, 0) + 1

    print(f"\nLayer breakdown:")
    for kind, count in sorted(layer_kinds.items()):
        print(f"  {kind:20s}: {count}")

    # Detailed layer info if verbose
    if args.verbose:
        print(f"\n{rule()}")
        print(f"DETAILED LAYER INFORMATION")
        print(f"{rule()}\n")

        for layer in net.layers:
            print(f"Layer {layer.id}: {layer.kind}")
            print(f"  In vars: {layer.in_vars}")
            print(f"  Out vars: {layer.out_vars}")
            if layer.params:
                print(f"  Params: {layer.params}")

            # Show predecessors
            preds = net.preds.get(layer.id, [])
            if preds:
                print(f"  Predecessors: {preds}")

            # Show successors
            succs = net.succs.get(layer.id, [])
            if succs:
                print(f"  Successors: {succs}")
            print()

    print(f"{rule()}\n")
    return 0


def run_serialization_test(args):
    """Test network serialization (save/load round-trip)."""
    print(f"\n{rule()}")
    print(f"SERIALIZATION TEST")
    print(f"{rule()}\n")

    from act.back_end.serialization.test_serialization import main as test_main

    print("Running serialization tests...\n")
    result = test_main()

    assert result == 0, "serialization tests failed"
    print(f"\n{rule()}")
    print("✓ All serialization tests passed")
    print(f"{rule()}\n")

    return 0


def list_examples(args):
    """List available example networks."""
    print(f"\n{rule()}")
    print(f"AVAILABLE EXAMPLE NETWORKS")
    print(f"{rule()}\n")

    from act.pipeline.verification.model_factory import ModelFactory

    factory = ModelFactory()
    names = factory.list_networks()
    print(f"Total networks: {len(names)}\n")

    # Group by category (inferred from filename)
    categories: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for name in names:
        info = factory.get_network_info(name)
        nl = name.lower()
        if "mnist" in nl:
            cat = "MNIST Classification"
        elif "cifar" in nl:
            cat = "CIFAR Classification"
        elif "control" in nl:
            cat = "Control Systems"
        elif "reachability" in nl:
            cat = "Reachability Analysis"
        else:
            cat = "Generated"
        categories.setdefault(cat, []).append((name, info))

    for cat, nets in sorted(categories.items()):
        print(f"{cat} ({len(nets)} networks):")
        print(rule(70, "-"))
        for name, info in sorted(nets):
            shape = info.get("input_shape", "?")
            layers = info.get("num_layers", "?")
            print(f"  {name:40s}  shape={shape}  layers={layers}")
        print()

    print(f"{rule()}")
    print("To generate networks: python -m act.back_end --generate")
    print(f"{rule()}\n")

    return 0


def _bench_default_path(kind: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("act", "pipeline", "log", f"bench_{kind}_{ts}.json")


def _write_bench_result(out_path: str, result: object) -> None:
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Wrote {out_path}")


def _run_bench_cnn(out_path: str) -> int:
    import torch
    from act.back_end.serialization.serialization import load_net_from_file
    from act.back_end.analyze import analyze
    from act.back_end.core import Fact, ConSet
    from act.back_end.verifier import find_entry_layer_id, gather_input_spec_layers, seed_from_input_specs

    nets = sorted(
        p for p in glob.glob("act/back_end/examples/nets/cnn2d_plain_*.json")
        if "_meta" not in p
    )
    if not nets:
        print("No CNN example nets found at act/back_end/examples/nets/cnn2d_plain_*.json")
        return 1

    results: Dict[str, Any] = {}
    for path in nets:
        net = load_net_from_file(path)
        entry = find_entry_layer_id(net)
        seed = seed_from_input_specs(gather_input_spec_layers(net))
        fact = Fact(bounds=seed, cons=ConSet())
        for _ in range(2):
            analyze(net, entry, fact)
        times: List[float] = []
        for _ in range(5):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            analyze(net, entry, fact)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        results[path] = {
            "mean": statistics.mean(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
            "all": times,
        }
        print(f"  {path}: mean={results[path]['mean']:.4f}s")

    _write_bench_result(out_path, results)
    return 0


def _run_bench_hybridz(out_path: str) -> int:
    import torch
    from act.back_end.core import Net, Layer, Bounds, Fact, ConSet
    from act.back_end.layer_schema import LayerKind
    from act.back_end.analyze import analyze
    from act.back_end.transfer_functions import set_transfer_function_mode
    from act.front_end.specs import OutputSpec

    def _build_net(B: int = 1, n_in: int = 8, n_hid: int = 16, n_out: int = 8) -> Net:
        layers: List[Any] = []
        next_id = 0
        next_var = 0

        def alloc_vars(n: int) -> List[int]:
            nonlocal next_var
            vs = list(range(next_var, next_var + n))
            next_var += n
            return vs

        in_v = alloc_vars(n_in)
        layers.append(Layer(id=next_id, kind=LayerKind.INPUT.value,
            params={"shape": (B, n_in), "dtype": "torch.float32"},
            in_vars=[], out_vars=in_v))
        next_id += 1
        layers.append(Layer(id=next_id, kind=LayerKind.INPUT_SPEC.value,
            params={"kind": "BOX",
                    "lb": torch.full((B, n_in), -1.0),
                    "ub": torch.full((B, n_in),  1.0)},
            in_vars=in_v, out_vars=in_v))
        next_id += 1
        h1_v = alloc_vars(n_hid)
        W1 = torch.randn(n_hid, n_in)
        b1 = torch.zeros(n_hid)
        layers.append(Layer(id=next_id, kind=LayerKind.DENSE.value,
            params={"weight": W1, "in_features": n_in, "out_features": n_hid,
                    "weight_pos": W1.clamp(min=0), "weight_neg": W1.clamp(max=0),
                    "bias": b1, "input_shape": (n_in,)},
            in_vars=in_v, out_vars=h1_v))
        next_id += 1
        layers.append(Layer(id=next_id, kind=LayerKind.RELU.value,
            params={"input_shape": (n_hid,)},
            in_vars=h1_v, out_vars=h1_v))
        next_id += 1
        out_v = alloc_vars(n_out)
        W2 = torch.randn(n_out, n_hid)
        b2 = torch.zeros(n_out)
        layers.append(Layer(id=next_id, kind=LayerKind.DENSE.value,
            params={"weight": W2, "in_features": n_hid, "out_features": n_out,
                    "weight_pos": W2.clamp(min=0), "weight_neg": W2.clamp(max=0),
                    "bias": b2, "input_shape": (n_hid,)},
            in_vars=h1_v, out_vars=out_v))
        next_id += 1
        assert_params = OutputSpec(
            kind="LINEAR_LE",
            c=torch.zeros(n_out),
            d=torch.tensor(1.0),
        ).encode_linear(B=B, n_out=n_out, device=torch.device("cpu"), dtype=torch.float32)
        layers.append(Layer(id=next_id, kind=LayerKind.ASSERT.value,
            params=assert_params, in_vars=out_v, out_vars=out_v))
        preds = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]}
        succs = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []}
        return Net(layers=layers, preds=preds, succs=succs)

    torch.manual_seed(42)
    net = _build_net()
    set_transfer_function_mode("hybridz")
    entry_id = next(l.id for l in net.layers if l.kind == LayerKind.INPUT.value)
    spec_layer = next(l for l in net.layers if l.kind == LayerKind.INPUT_SPEC.value)
    import torch as _torch
    lb_t = cast(_torch.Tensor, spec_layer.params["lb"])
    ub_t = cast(_torch.Tensor, spec_layer.params["ub"])
    seed = Bounds(lb_t.clone(), ub_t.clone())
    fact = Fact(bounds=seed, cons=ConSet())
    for _ in range(2):
        analyze(net, entry_id, fact)
    times: List[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        analyze(net, entry_id, fact)
        times.append(time.perf_counter() - t0)
    result = {
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
    }
    print(f"  hybridz synthetic 4-layer MLP: mean={result['mean']:.4f}s")
    _write_bench_result(out_path, result)
    return 0


def run_bench(args) -> int:
    """Run timing benchmarks for CNN and/or HybridZ analyze() code paths."""
    kind = args.bench
    bench_out = getattr(args, "bench_out", None)

    print(f"\n{rule()}")
    print(f"ACT BENCH: {kind.upper()}")
    print(f"{rule()}\n")

    if kind in ("cnn", "all"):
        out_path = bench_out if (bench_out and kind == "cnn") else _bench_default_path("cnn")
        print(f"--- CNN benchmark ---")
        rc = _run_bench_cnn(out_path)
        if rc != 0:
            return rc

    if kind in ("hybridz", "all"):
        out_path = bench_out if (bench_out and kind == "hybridz") else _bench_default_path("hybridz")
        print(f"\n--- HybridZ benchmark ---")
        rc = _run_bench_hybridz(out_path)
        if rc != 0:
            return rc

    print(f"\n{rule()}")
    print(f"Bench complete")
    print(f"{rule()}\n")
    return 0


def run_diff_nets(args) -> int:
    """Load two ACT Net JSON files and print a unified-diff-style layer comparison."""
    from act.back_end.serialization.serialization import load_net_from_file

    path_a, path_b = args.diff_nets

    net_a = load_net_from_file(path_a)

    net_b = load_net_from_file(path_b)

    print(f"\n{rule()}")
    print(f"NET DIFF")
    print(f"  A: {path_a}")
    print(f"  B: {path_b}")
    print(f"{rule()}\n")

    la, lb = len(net_a.layers), len(net_b.layers)
    marker = "  " if la == lb else "!"
    print(f"{marker} Layer count: A={la}  B={lb}")

    n_common = min(la, lb)
    for i in range(n_common):
        lyr_a = net_a.layers[i]
        lyr_b = net_b.layers[i]
        diffs: List[str] = []
        if lyr_a.kind != lyr_b.kind:
            diffs.append(f"kind: {lyr_a.kind!r} -> {lyr_b.kind!r}")
        if len(lyr_a.in_vars) != len(lyr_b.in_vars):
            diffs.append(f"in_vars: {len(lyr_a.in_vars)} -> {len(lyr_b.in_vars)}")
        if len(lyr_a.out_vars) != len(lyr_b.out_vars):
            diffs.append(f"out_vars: {len(lyr_a.out_vars)} -> {len(lyr_b.out_vars)}")
        keys_a = set(lyr_a.params.keys())
        keys_b = set(lyr_b.params.keys())
        if keys_a != keys_b:
            only_a = sorted(keys_a - keys_b)
            only_b = sorted(keys_b - keys_a)
            if only_a:
                diffs.append(f"params only in A: {only_a}")
            if only_b:
                diffs.append(f"params only in B: {only_b}")
        if diffs:
            print(f"! Layer {i:2d} ({lyr_a.kind:20s}): " + "; ".join(diffs))
        else:
            print(f"  Layer {i:2d} ({lyr_a.kind:20s}): identical")

    if la != lb:
        extra_net = net_a if la > lb else net_b
        extra_side = "A" if la > lb else "B"
        for i in range(n_common, max(la, lb)):
            lyr = extra_net.layers[i]
            print(f"+ Layer {i:2d} ({lyr.kind:20s}): only in {extra_side}")

    print(f"\n{rule()}\n")
    return 0


def main():
    """Main CLI entry point for ACT Back-End."""
    parser = argparse.ArgumentParser(
        description="ACT Back-End: Core Verification Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ============================================================================
  # NETWORK FACTORY - Generate example networks
  # ============================================================================
  
  # Generate all example networks from default config
  python -m act.back_end --generate
  
  # Generate with custom config
  python -m act.back_end --generate --config my_config.yaml --output ./networks
  
  # List available example networks
  python -m act.back_end --list-examples
  
  # ============================================================================
  # VERIFICATION - Run verification on networks
  # ============================================================================
  
  # Single-shot verification
  python -m act.back_end --verify --network act/back_end/examples/nets/mnist_robust_easy.json
  
  # Branch-and-bound verification
  python -m act.back_end --verify --network mnist_robust_hard.json --bab
  
  # Custom BaB parameters
  python -m act.back_end --verify --network control_strict.json \\
    --bab --bab-max-depth 10 --bab-max-subproblems 1000 --timeout 300
  
  # Use specific solver
  python -m act.back_end --verify --network cifar_margin_tight.json \\
    --solver gurobi --timeout 60
  
  # ============================================================================
  # NETWORK INSPECTION - Analyze network structure
  # ============================================================================
  
  # Show network information
  python -m act.back_end --info --network mnist_robust_easy.json
  
  # Detailed layer information
  python -m act.back_end --info --network control_balanced.json --verbose
  
  # ============================================================================
  # TESTING - Run internal tests
  # ============================================================================
  
  # Test serialization (save/load round-trip)
  python -m act.back_end --test-serialization
  
  # ============================================================================
  # BENCHMARKING - Time analyze() on example nets
  # ============================================================================
  
  # Benchmark CNN analyze() on all cnn2d_plain_* example nets
  python -m act.back_end --bench cnn
  
  # Benchmark HybridZ analyze() on a synthetic MLP
  python -m act.back_end --bench hybridz
  
  # Run both benchmarks and write JSON output
  python -m act.back_end --bench all
  python -m act.back_end --bench cnn --bench-out /tmp/my_cnn_timing.json
  
  # ============================================================================
  # NET DIFF - Compare two network JSON files
  # ============================================================================
  
  # Compare layer count, kinds, variable widths, and param keys
  python -m act.back_end --diff-nets act/back_end/examples/nets/net_a.json \\
                                      act/back_end/examples/nets/net_b.json
  
  # ============================================================================
  # DEVICE CONFIGURATION
  # ============================================================================
  
  # Use CPU with float32
  python -m act.back_end --verify --network mnist.json --device cpu --dtype float32
  
  # Use GPU with float64
  python -m act.back_end --verify --network cifar.json --device cuda --dtype float64

Exit status (--verify):
  0  every lane received a verdict (CERTIFIED / FALSIFIED / UNKNOWN / TIMEOUT)
  1  error, no verdict, or a network the selected solver cannot handle
  2  invalid configuration (ConfigError)
        """,
    )

    # Command groups
    cmd_group = parser.add_mutually_exclusive_group(required=True)

    cmd_group.add_argument(
        "--generate",
        "-g",
        action="store_true",
        help="Generate example networks from YAML configuration",
    )
    cmd_group.add_argument(
        "--verify", "-v", action="store_true", help="Run verification on a network"
    )
    cmd_group.add_argument(
        "--info", "-i", action="store_true", help="Display network information"
    )
    cmd_group.add_argument(
        "--list-examples",
        "-l",
        action="store_true",
        dest="list_examples",
        help="List available example networks",
    )
    cmd_group.add_argument(
        "--test-serialization",
        action="store_true",
        dest="test_serialization",
        help="Run serialization tests",
    )
    cmd_group.add_argument(
        "--bench",
        type=str,
        choices=["cnn", "hybridz", "all"],
        metavar="{cnn,hybridz,all}",
        dest="bench",
        help="Run analyze() timing benchmarks: cnn nets, hybridz synthetic MLP, or all",
    )
    cmd_group.add_argument(
        "--diff-nets",
        nargs=2,
        metavar=("NET_A", "NET_B"),
        dest="diff_nets",
        help="Load two ACT Net JSON files and print a layer-level diff summary",
    )
    # Bench options
    bench_group = parser.add_argument_group("Bench Options")
    bench_group.add_argument(
        "--bench-out",
        type=str,
        default=None,
        dest="bench_out",
        help=(
            "Output JSON path for bench results "
            "(default: act/pipeline/log/bench_<kind>_<timestamp>.json)"
        ),
    )

    # Network factory options
    factory_group = parser.add_argument_group("Network Factory Options")
    factory_group.add_argument(
        "--config", "-c", type=str, help="Path to YAML configuration file"
    )
    factory_group.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory for generated networks (default: act/back_end/examples/nets)",
    )
    factory_group.add_argument(
        "--num", type=int, help="Number of networks to generate (generate mode)"
    )
    factory_group.add_argument(
        "--base-seed",
        type=int,
        dest="base_seed",
        help="Base seed for reproducible generation",
    )
    factory_group.add_argument(
        "--name-prefix",
        type=str,
        dest="name_prefix",
        help="Filename prefix for generated networks",
    )
    factory_group.add_argument(
        "--tf-targets",
        type=str,
        nargs="+",
        dest="tf_targets",
        choices=_TF_MODES,
        help="Target TFs for layer filtering (generate mode)",
    )
    factory_group.add_argument(
        "--registry-mode",
        type=str,
        dest="registry_mode",
        choices=["intersection", "union"],
        default="intersection",
        help="How to combine TF layer sets: 'intersection' (default) or 'union'",
    )

    # Verification options
    verify_group = parser.add_argument_group("Verification Options")
    verify_group.add_argument(
        "--network", "-n", type=str, help="Path to network file (JSON format)"
    )
    verify_group.add_argument(
        "--onnx",
        type=str,
        help="ONNX network paired with --vnnlib for direct verification",
    )
    verify_group.add_argument(
        "--vnnlib",
        type=str,
        help="VNNLIB property paired with --onnx for direct verification",
    )
    verify_group.add_argument(
        "--query-index",
        type=str,
        dest="query_index",
        help="CSV index of checkpoint-backed SST/Yelp verification queries",
    )
    verify_group.add_argument(
        "--query-id",
        type=int,
        dest="query_id",
        help="Zero-based data-row id paired with --query-index",
    )
    verify_group.add_argument(
        "--save-net",
        type=str,
        dest="save_net",
        help="Save the exact converted Net(s) verified from a direct input source",
    )
    verify_group.add_argument(
        "--solver",
        "-s",
        type=str,
        choices=_SOLVERS,
        default=None,
        dest="solver",
        help=(
            "Solver backend:\n"
            "  'gurobi'  — commercial MILP/LP (license required).  LP cascade.\n"
            "  'torchlp' — PyTorch-tensor LP (Adam + penalty + box projection,\n"
            "              GPU-capable).  LP cascade.\n"
            "  'hybridz' — Hybrid Zonotope propagation with a standalone open-source\n"
            "              MILP verdict; automatically selects HybridzTF.\n"
            "  'dual'    — DualSolver, linear-relaxation dual certified bounds via\n"
            "              backward propagation.  No LP cascade (DualSolver is\n"
            "              its own verification pipeline).\n"
            "  'auto'    — try gurobi, fall back to torchlp.\n"
            "Default: from config.yaml / $ACT_SOLVER / 'auto'."
        ),
    )
    verify_group.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=None,
        help="Verification timeout in seconds (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--tf-mode",
        type=str,
        choices=_TF_MODES,
        default=None,
        dest="tf_mode",
        help=(
            "Forward-bounds transfer function: 'interval' or 'hybridz'.  Selects "
            "the abstract interpretation used during analyze() to seed bounds "
            "for the LP cascade.  Default: configured default (typically "
            "'interval').  The standalone hybridz solver selects HybridzTF "
            "automatically; dual does not use this option."
        ),
    )
    verify_group.add_argument(
        "--method",
        type=str,
        choices=[name.replace("_", "-") for name in VALID_BERT_METHODS],
        default=None,
        help="SST/Yelp method: planar, rule, alpha, ibp, or discrete.",
    )
    verify_group.add_argument(
        "--p",
        type=float,
        default=None,
        help="Text embedding perturbation norm metadata (2 or inf).",
    )
    verify_group.add_argument(
        "--perturbed-words",
        type=int,
        choices=[1, 2],
        default=None,
        dest="perturbed_words",
        help="Number of SST/Yelp token positions perturbed together.",
    )
    verify_group.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Initial SST/Yelp verification radius.",
    )
    verify_group.add_argument(
        "--max-eps",
        type=float,
        default=None,
        dest="max_eps",
        help="Maximum radius for SST/Yelp certified-radius search.",
    )
    verify_group.add_argument(
        "--num-verify-iters",
        type=int,
        default=None,
        dest="num_verify_iters",
        help="Binary-search iterations for SST/Yelp certified radius.",
    )
    verify_group.add_argument(
        "--k",
        type=int,
        default=None,
        help="Rule threshold for rule-slope attention alpha.",
    )
    verify_group.add_argument(
        "--alpha-opt-steps",
        type=int,
        default=None,
        dest="alpha_opt_steps",
        help="Optimization steps for optimized-alpha refinement.",
    )

    # BaB mode: --bab enables, --no-bab disables, absent = from config.yaml
    bab_toggle = verify_group.add_mutually_exclusive_group()
    bab_toggle.add_argument(
        "--bab",
        action="store_true",
        default=None,
        dest="bab",
        help="Enable branch-and-bound verification",
    )
    bab_toggle.add_argument(
        "--no-bab",
        action="store_false",
        dest="bab",
        help="Disable branch-and-bound (single-shot)",
    )

    # BaB algorithm parameters
    verify_group.add_argument(
        "--bab-max-depth",
        type=int,
        default=None,
        dest="bab_max_depth",
        help="Maximum BaB tree depth (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-max-subproblems",
        type=int,
        default=None,
        dest="bab_max_subproblems",
        help="Maximum number of BaB subproblems (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-solver-tier",
        type=str,
        default=None,
        choices=VALID_SOLVER_TIERS,
        dest="bab_solver_tier",
        help="Solver tier for BaB bound computation (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-root-bounds-reuse",
        type=str,
        default=None,
        choices=VALID_ROOT_BOUNDS_REUSE,
        dest="bab_root_bounds_reuse",
        help=(
            "Descendant reuse of the root box's forward bounds (dual tiers): "
            "none = every node re-propagates; "
            "plain = children get the root dict untouched; "
            "split_refresh = root dict hardened by the split-derived interval "
            "refresh and --bab-per-subproblem-refine; "
            "split_refresh_linear = affine/ReLU halfspace refresh with interval fallback "
            "(default: from config.yaml)"
        ),
    )
    verify_group.add_argument(
        "--bab-branching",
        type=str,
        default=None,
        dest="bab_branching",
        help=(
            "Branching strategy: which neuron or input axis to split. "
            "Neuron branching (babsr/fsb/gain) requires "
            "--bab-solver-tier dual_alpha or dual_alpha_eta. Note that "
            "--bab-multi-split-levels is orthogonal to this choice "
            "(default: from config.yaml)"
        ),
    )
    verify_group.add_argument(
        "--bab-bounding",
        type=str,
        default=None,
        choices=VALID_BOUNDINGS,
        dest="bab_bounding",
        help=(
            "Pool selection: which pending subproblems the next wave expands. "
            "depth_bound_blend = 0.5*norm(depth) + 0.5*bound-urgency blend; "
            "greedy = best-first on |lb| (Oliva-Greedy, ECOOP 2025); "
            "annealed = Gumbel noise with temp = sa_cooling_rate**step "
            "(Oliva-SA, ECOOP 2025); "
            "diverse_split_signs = top-k then split-sign diversity repulsion; "
            "random = uniform sampling; "
            "mcts = N/Q side tables over the BaB tree. "
            "The first four honour --bab-top-k; random and mcts reject it "
            "(default: from config.yaml)"
        ),
    )
    verify_group.add_argument(
        "--bab-sa-cooling-rate",
        type=float,
        default=None,
        dest="bab_sa_cooling_rate",
        help="Cooling rate for --bab-bounding annealed (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-frontier-cap",
        type=int,
        default=None,
        dest="bab_frontier_cap",
        help="Maximum pending BaB frontier leaves to retain; 0 disables eviction (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-llm-probe-enabled",
        action="store_true",
        default=None,
        dest="bab_llm_probe_enabled",
        help="Enable the LLM-probe BaB controller (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-llm-probe-backend",
        type=str,
        default=None,
        choices=["mock", "openrouter", "openai", "glm", "minimax", "claude_cli"],
        dest="bab_llm_probe_backend",
        help="LLM-probe backend (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-llm-probe-model",
        type=str,
        default=None,
        dest="bab_llm_probe_model",
        help="LLM-probe model name (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-llm-probe-base-url",
        type=str,
        default=None,
        dest="bab_llm_probe_base_url",
        help="LLM-probe OpenAI-compatible base URL (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-llm-probe-cadence",
        type=int,
        default=None,
        dest="bab_llm_probe_cadence",
        help="LLM-probe consult cadence in waves (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-llm-probe-api-key-env",
        type=str,
        default=None,
        dest="bab_llm_probe_api_key_env",
        help="LLM-probe environment variable name holding the API key (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-llm-probe-temperature",
        type=float,
        default=None,
        dest="bab_llm_probe_temperature",
        help="LLM-probe sampling temperature (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-llm-probe-timeout",
        type=float,
        default=None,
        dest="bab_llm_probe_timeout",
        help="LLM-probe per-call wall-clock timeout in seconds; a slower response raises and "
             "falls back to baseline for that wave (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-llm-probe-max-candidates",
        type=int,
        default=None,
        dest="bab_llm_probe_max_candidates",
        help="LLM-probe maximum unstable neuron candidates per domain (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-llm-probe-max-candidates-total",
        type=int,
        default=None,
        dest="bab_llm_probe_max_candidates_total",
        help="LLM-probe total candidate budget across all domains; exceeding falls back to FSB (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-llm-probe-neuron-topk",
        type=int,
        default=None,
        dest="bab_llm_probe_neuron_topk",
        help="If >0, truncate the neuron-selection candidate set to the top-K by score before "
             "sending to the LLM (0=full set up to max-candidates-total) (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-llm-probe-history",
        type=int,
        default=None,
        dest="bab_llm_probe_history",
        help="LLM-probe number of past wave outcomes to include in context (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-llm-probe-max-failures",
        type=int,
        default=None,
        dest="bab_llm_probe_max_failures",
        help="LLM-probe consecutive failure threshold before circuit-breaker disables the controller (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-llm-probe-decisions",
        type=str,
        default=None,
        dest="bab_llm_probe_decisions",
        help="LLM-probe comma-separated decision types to enable; tokens: split,frontier,refine,neuron (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-llm-probe-log",
        action="store_true",
        default=None,
        dest="bab_llm_probe_log",
        help="Enable per-wave LLM-probe decision logging (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-multi-split-levels",
        type=int,
        default=None,
        dest="bab_multi_split_levels",
        help=(
            "Neurons split jointly per branching step; each lane fans out into "
            "all 2^k sign combinations (verdict-boundary joint splitting). "
            "1 = single split. The k neurons are chosen by the BaBSR heuristic "
            "(area x |nu|) regardless of --bab-branching. Requires "
            "--bab-solver-tier dual_alpha or dual_alpha_eta "
            "(default: from config.yaml)"
        ),
    )
    verify_group.add_argument(
        "--bab-top-k",
        type=int,
        default=None,
        dest="bab_top_k",
        help=(
            "Cap on subproblems popped per BaB wave, independent of the batch "
            "size; 0 = unbounded. Honoured by --bab-bounding "
            "depth_bound_blend/greedy/annealed/diverse_split_signs; rejected "
            "for random/mcts (default: from config.yaml)"
        ),
    )
    verify_group.add_argument(
        "--bab-input-split-fanout",
        type=int,
        default=None,
        dest="bab_input_split_fanout",
        help="Uniform fanout for input splits (default: from config.yaml)",
    )

    # Backend config file
    verify_group.add_argument(
        "--backend-config",
        type=str,
        default=None,
        dest="backend_config",
        help="Path to backend YAML config (default: act/config/backend.yaml)",
    )

    # Common options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Add standard device/dtype arguments (shared across all ACT CLIs)
    add_device_args(parser)

    # Add missing config-backed flags from dataclass fields. Explicit legacy
    # flags above win on spelling/dest behavior; generated flags fill gaps.
    _add_dataclass_config_args(parser)

    # Detect user-provided flags BEFORE parsing so env vars / config.yaml
    # can serve as fallbacks without overriding explicit CLI flags.
    argv = sys.argv[1:]
    _user_set = lambda flag: any(  # noqa: E731
        a == flag or a.startswith(flag + "=") for a in argv
    )

    args = parser.parse_args()

    # Validate arguments based on command
    if args.info and not args.network:
        parser.error("--network is required for --info")
    if not args.verify and (
        args.onnx is not None
        or args.vnnlib is not None
        or args.save_net is not None
        or args.query_index is not None
        or args.query_id is not None
    ):
        parser.error("dynamic verification sources and --save-net require --verify")
    if args.verify:
        direct_source = args.onnx is not None or args.vnnlib is not None
        query_source = args.query_index is not None or args.query_id is not None
        source_count = int(bool(args.network)) + int(direct_source) + int(query_source)
        if source_count > 1:
            parser.error(
                "choose exactly one verification source: --network, paired "
                "--onnx/--vnnlib, or paired --query-index/--query-id"
            )
        if (args.onnx is None) != (args.vnnlib is None):
            parser.error("--onnx and --vnnlib must be provided together")
        if (args.query_index is None) != (args.query_id is None):
            parser.error("--query-index and --query-id must be provided together")
        if args.save_net and not (direct_source or query_source):
            parser.error(
                "--save-net requires paired --onnx/--vnnlib or --query-index/--query-id"
            )
        if source_count == 0:
            parser.error(
                "--network, paired --onnx/--vnnlib, or paired "
                "--query-index/--query-id is required for --verify"
            )
        if direct_source:
            for label, value in (("--onnx", args.onnx), ("--vnnlib", args.vnnlib)):
                if not Path(value).is_file():
                    parser.error(f"{label} must name an existing file: {value}")
        if query_source and not Path(args.query_index).is_file():
            parser.error(f"--query-index must name an existing file: {args.query_index}")
        if (direct_source or query_source) and args.timeout is not None and args.timeout <= 0:
            parser.error("--timeout must be greater than zero")

    if args.diff_nets:
        return run_diff_nets(args)

    # ── Build BackendConfig ──────────────────────────────────────────────
    # Load YAML, then apply env vars and CLI flags. BaB presets are resolved at
    # verification time so explicit BaB flags remain the highest precedence.
    from act.config.config import BackendConfig

    try:
        backend_cfg = BackendConfig.from_yaml(
            config_path=args.backend_config,
            **_collect_backend_overrides(args, _user_set),
        )
    except ConfigError as e:
        return _report_config_error(args.network or args.query_index or "Error", e)

    # Initialize device manager from the resolved config
    import argparse as _ap

    initialize_from_args(
        _ap.Namespace(device=backend_cfg.device, dtype=backend_cfg.dtype)
    )

    tf_mode = args.tf_mode
    if backend_cfg.solver == "hybridz":
        if tf_mode is not None and tf_mode != "hybridz":
            logger.warning(
                "--solver hybridz requires the hybridz transformer; overriding "
                "--tf-mode %s",
                tf_mode,
            )
        tf_mode = "hybridz"

    if tf_mode is not None:
        from act.back_end.analyze import initialize_tf_mode

        initialize_tf_mode(tf_mode, backend_cfg.hybridz)

    # Set the solver-mode global so verify_once / _verify_one_net can dispatch
    # dual ↔ LP-cascade without consulting the TF mode (refactor decoupled
    # dual from the --tf-mode axis). Always set, so a previous process state
    # cannot leak across invocations.
    from act.back_end.transfer_functions import set_solver_mode
    set_solver_mode(backend_cfg.solver)

    # Execute command
    try:
        if args.generate:
            return run_network_factory(args, backend_cfg)
        elif args.verify:
            if args.onnx is not None:
                return run_vnnlib_verification(args, backend_cfg)
            if args.query_index is not None:
                return run_query_verification(args, backend_cfg)
            return run_verification(args, backend_cfg)
        elif args.info:
            return run_network_info(args)
        elif args.list_examples:
            return list_examples(args)
        elif args.test_serialization:
            return run_serialization_test(args)
        elif args.bench:
            return run_bench(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130


# (override_key, args_attr, env_var, env_cast, cli_check)
# cli_check="user_set" for flags with non-None defaults (--device/--dtype/--registry-mode)
_BACKEND_ALIAS_OVERRIDE_SPEC: list[tuple[str, str, Optional[str], Any, str]] = [
    ("solver",               "solver",              "ACT_SOLVER",     None, "not_none"),
    ("device",               "device",              "ACT_DEVICE",     None, "user_set"),
    ("dtype",                "dtype",               "ACT_DTYPE",      None, "user_set"),
    ("verbose",              "verbose",             None,             None, "user_set"),
    ("timeout",              "timeout",             None,             None, "not_none"),
    ("method",               "method",              None,             None, "not_none"),
    ("p",                    "p",                   None,             None, "not_none"),
    ("perturbed_words",      "perturbed_words",     None,             None, "not_none"),
    ("eps",                  "eps",                 None,             None, "not_none"),
    ("max_eps",              "max_eps",             None,             None, "not_none"),
    ("num_verify_iters",     "num_verify_iters",    None,             None, "not_none"),
    ("k",                    "k",                   None,             None, "not_none"),
    ("alpha_opt_steps",      "alpha_opt_steps",     None,             None, "not_none"),
    ("bab_enabled",          "bab",                 None,             None, "not_none"),
    ("bab_max_depth",        "bab_max_depth",       None,             None, "not_none"),
    ("bab_max_nodes",        "bab_max_subproblems", None,             None, "not_none"),
    ("bab_branching_method", "bab_branching",       None,             None, "not_none"),
    ("bab_sa_cooling_rate",  "bab_sa_cooling_rate", None,             None, "not_none"),
    ("bab_frontier_cap",     "bab_frontier_cap",    None,             None, "not_none"),
    ("bab_input_split_fanout", "bab_input_split_fanout", None,          None, "not_none"),
    ("bab_llm_probe_enabled", "bab_llm_probe_enabled",  None,             None, "user_set"),
    ("bab_llm_probe_backend", "bab_llm_probe_backend",  None,             None, "not_none"),
    ("bab_llm_probe_model",   "bab_llm_probe_model",    None,             None, "not_none"),
    ("bab_llm_probe_base_url", "bab_llm_probe_base_url", None,            None, "not_none"),
    ("bab_llm_probe_cadence", "bab_llm_probe_cadence",  None,             int,  "not_none"),
    ("bab_llm_probe_api_key_env",        "bab_llm_probe_api_key_env",        None, None,  "not_none"),
    ("bab_llm_probe_temperature",        "bab_llm_probe_temperature",        None, float, "not_none"),
    ("bab_llm_probe_timeout",            "bab_llm_probe_timeout",            None, float, "not_none"),
    ("bab_llm_probe_max_candidates",     "bab_llm_probe_max_candidates",     None, int,   "not_none"),
    ("bab_llm_probe_max_candidates_total", "bab_llm_probe_max_candidates_total", None, int, "not_none"),
    ("bab_llm_probe_neuron_topk",         "bab_llm_probe_neuron_topk",         None, int,   "not_none"),
    ("bab_llm_probe_history",            "bab_llm_probe_history",            None, int,   "not_none"),
    ("bab_llm_probe_max_failures",       "bab_llm_probe_max_failures",       None, int,   "not_none"),
    ("bab_llm_probe_decisions",          "bab_llm_probe_decisions",          None, None,  "not_none"),
    ("bab_llm_probe_log",                "bab_llm_probe_log",                None, None,  "user_set"),
    ("bab_multi_split_levels",           "bab_multi_split_levels",           None, int,   "not_none"),
    ("bab_top_k",                        "bab_top_k",                        None, int,   "not_none"),
    ("gen_output_dir",       "output",              "ACT_GEN_OUTPUT", None, "not_none"),
    ("gen_num_instances",    "num",                 "ACT_GEN_NUM",    int,  "not_none"),
    ("gen_base_seed",        "base_seed",           "ACT_GEN_SEED",   int,  "not_none"),
    ("gen_name_prefix",      "name_prefix",         None,             None, "not_none"),
    ("gen_tf_targets",       "tf_targets",          None,             None, "not_none"),
    ("gen_registry_mode",    "registry_mode",       None,             None, "user_set"),
]


def _build_backend_override_spec() -> list[tuple[str, str, Optional[str], Any, str]]:
    spec = list(_BACKEND_ALIAS_OVERRIDE_SPEC)
    aliased_keys = {key for key, *_ in spec}
    for key in sorted(_backend_override_keys_from_dataclasses() - aliased_keys):
        spec.append((key, key, None, None, "not_none"))
    return spec


_BACKEND_OVERRIDE_SPEC = _build_backend_override_spec()


def _collect_backend_overrides(args: Any, _user_set: Any) -> dict[str, Any]:
    """Build overrides dict from CLI flags + env vars.

    Precedence before BaB preset resolution: CLI > env > backend.yaml.
    """
    overrides: dict[str, Any] = {}
    for key, attr, env, env_cast, check in _BACKEND_OVERRIDE_SPEC:
        cli_val = getattr(args, attr, None)
        if check == "user_set":
            attr_flag = attr.replace('_', '-')
            cli_provided = _user_set(f"--{attr_flag}") or _user_set(f"--no-{attr_flag}")
        else:
            cli_provided = cli_val is not None
        if cli_provided:
            overrides[key] = cli_val
        elif env is not None and os.environ.get(env):
            overrides[key] = env_cast(os.environ[env]) if env_cast else os.environ[env]

    return overrides


def _run_cli_cascade_smoke() -> int:
    """Light integration smoke: build a tiny net, run the 3-tier cascade, verify dispatch."""
    import torch
    from act.back_end.core import Layer, Net
    from act.back_end.layer_schema import LayerKind
    from act.front_end.specs import OutputSpec
    from act.config.config import BackendConfig
    from act.util.stats import VerifyStatus

    def _check(label: str, fn) -> None:
        fn()
        print(f"  PASS  {label}")

    def _build_tiny_net(B: int = 1, n_in: int = 4, n_out: int = 2) -> Net:
        layers: List[Any] = []
        nv = 0

        def alloc(n: int) -> List[int]:
            nonlocal nv
            vs = list(range(nv, nv + n))
            nv += n
            return vs

        in_v = alloc(n_in)
        layers.append(Layer(id=0, kind=LayerKind.INPUT.value,
            params={"shape": (B, n_in), "dtype": "torch.float32"},
            in_vars=[], out_vars=in_v))
        layers.append(Layer(id=1, kind=LayerKind.INPUT_SPEC.value,
            params={"kind": "BOX",
                    "lb": torch.full((B, n_in), -1.0),
                    "ub": torch.full((B, n_in),  1.0)},
            in_vars=in_v, out_vars=in_v))
        out_v = alloc(n_out)
        W = torch.eye(n_out, n_in)
        b = torch.zeros(n_out)
        layers.append(Layer(id=2, kind=LayerKind.DENSE.value,
            params={"weight": W, "in_features": n_in, "out_features": n_out,
                    "weight_pos": W.clamp(min=0), "weight_neg": W.clamp(max=0),
                    "bias": b, "input_shape": (n_in,)},
            in_vars=in_v, out_vars=out_v))
        assert_params = OutputSpec(
            kind="LINEAR_LE",
            c=torch.zeros(n_out),
            d=torch.tensor(1.0),
        ).encode_linear(B=B, n_out=n_out, device=torch.device("cpu"), dtype=torch.float32)
        layers.append(Layer(id=3, kind=LayerKind.ASSERT.value,
            params=assert_params, in_vars=out_v, out_vars=out_v))
        preds = {0: [], 1: [0], 2: [1], 3: [2]}
        succs = {0: [1], 1: [2], 2: [3], 3: []}
        return Net(layers=layers, preds=preds, succs=succs)

    def _t_interval_only():
        from act.back_end.verifier import verify_once
        net = _build_tiny_net()
        results = list(verify_once(net=net))
        assert len(results) == 1
        assert results[0].status in (VerifyStatus.CERTIFIED, VerifyStatus.UNKNOWN, VerifyStatus.FALSIFIED)

    def _t_cascade_default_config():
        cfg = BackendConfig(solver="torchlp", lp_enabled=True, bab_enabled=False)
        net = _build_tiny_net()
        from act.back_end.verifier import verify_once, verify_lp_batched
        results = list(verify_once(net=net))
        assert len(results) == 1
        if results[0].status == VerifyStatus.UNKNOWN and cfg.lp_enabled:
            lp_results = verify_lp_batched(
                net,
        solver_factory=lambda: _make_solver(cfg.solver, cfg.torchlp, cfg.gurobi),
                timelimit=cfg.timeout,
            )
            assert len(lp_results) == 1
            assert lp_results[0].status in (
                VerifyStatus.CERTIFIED, VerifyStatus.FALSIFIED, VerifyStatus.UNKNOWN
            )

    def _t_lp_disabled_skips_tier2():
        cfg = BackendConfig(solver="torchlp", lp_enabled=False, bab_enabled=False)
        assert not cfg.lp_enabled

    def _t_bab_max_batch_size_default():
        cfg = BackendConfig()
        assert cfg.bab_max_batch_size == 8

    def _t_frontier_config() -> None:
        """Generated CLI preserves auto and rejects invalid memory budgets."""
        from act.config.config import BaBConfig

        parser = argparse.ArgumentParser()
        _add_dataclass_config_args(parser)
        for raw, expected in (("auto", "auto"), ("3", 3)):
            args = parser.parse_args(["--bab-max-batch-size", raw])
            assert BackendConfig(bab_max_batch_size=args.bab_max_batch_size).bab_max_batch_size == expected
        flags = [
            "--bab-frontier-mem-safety", "0.4", "--no-bab-frontier-offload",
            "--bab-intermediate-refine-max-tensor-mib", "32",
        ]
        args = parser.parse_args(flags)
        cfg = BackendConfig.from_yaml(**_collect_backend_overrides(args, lambda flag: flag in flags))
        assert cfg.bab.frontier_mem_safety == 0.4
        assert not cfg.bab.frontier_offload
        assert cfg.bab.intermediate_refine_max_tensor_mib == 32
        for value in (0, -1, True, 1.5, "invalid", "2"):
            try:
                BackendConfig.from_yaml(bab_max_batch_size=value)
            except ConfigError:
                pass
            else:
                raise AssertionError(f"accepted invalid K: {value!r}")
        for field in ("frontier_mem_safety", "intermediate_refine_max_tensor_mib"):
            for value in (-1.0, float("nan"), float("inf")):
                try:
                    if field == "frontier_mem_safety":
                        BaBConfig(frontier_mem_safety=value)
                    else:
                        BaBConfig(intermediate_refine_max_tensor_mib=value)
                except ConfigError:
                    pass
                else:
                    raise AssertionError(f"accepted invalid {field}: {value}")
        try:
            BackendConfig(solver="gurobi", lp_enabled=False, bab_max_batch_size="auto")
        except ConfigError:
            pass
        else:
            raise AssertionError("Gurobi accepted unbounded batch sizing")

    def _t_dual_unsafe_linear() -> None:
        """The real cascade accepts EXISTS specs without certifying a tie."""
        from act.back_end.transfer_functions import set_solver_mode
        from act.util.device_manager import get_default_dtype

        # Initialize the managed dtype before constructing any fixture tensor.
        get_default_dtype()
        cfg = BackendConfig(solver="dual", lp_enabled=False, bab_enabled=False)
        set_solver_mode("dual")
        try:
            for threshold in (-2.0, 0.0):
                net = _build_tiny_net()
                if threshold == 0.0:
                    net.layers[2].params["weight"].zero_()
                    net.layers[2].params["weight_pos"].zero_()
                    net.layers[2].params["weight_neg"].zero_()
                weight = net.layers[2].params["weight"]
                net.layers[-1].params = OutputSpec(
                    kind=OutKind.UNSAFE_LINEAR,
                    c=weight.new_tensor([[1.0, 0.0]]),
                    d=weight.new_tensor([threshold]),
                ).encode_linear(B=1, n_out=2, device=weight.device, dtype=weight.dtype)
                results, error, _ = _verify_net(net, cfg)
                assert error is None and len(results) == 1
                assert (results[0].status == VerifyStatus.CERTIFIED) == (threshold < 0)
        finally:
            set_solver_mode("torchlp")

    def _t_cascade_shares_deadline(
        interval_seconds: float = 3.0,
        lp_seconds: float = 4.0,
        expected_limits: tuple[float, ...] = (7.0, 3.0, 2.0),
    ) -> None:
        """Every refinement consumes the same budget, including earlier lanes."""
        from unittest.mock import patch
        from act.back_end.transfer_functions import set_solver_mode
        from act.util.stats import VerifyResult

        cfg = BackendConfig(solver="torchlp", timeout=10, lp_enabled=True, bab_enabled=True)
        net = _build_tiny_net(B=2)
        set_solver_mode("torchlp")
        clock = [0.0]
        limits: list[float] = []

        def interval(*args, **kwargs) -> list[VerifyResult]:
            clock[0] += interval_seconds
            return [VerifyResult(VerifyStatus.UNKNOWN) for _ in range(2)]

        def lp(*args, timelimit: float, **kwargs) -> list[VerifyResult]:
            limits.append(timelimit)
            clock[0] += lp_seconds
            return [VerifyResult(VerifyStatus.UNKNOWN) for _ in range(2)]

        def bab(*args, time_budget_s: float, **kwargs) -> VerifyResult:
            limits.append(time_budget_s)
            clock[0] += 1.0
            return VerifyResult(VerifyStatus.UNKNOWN)

        with (
            patch.object(time, "monotonic", side_effect=lambda: clock[0]),
            patch("act.back_end.verifier.verify_once", side_effect=interval),
            patch("act.back_end.verifier.verify_lp_batched", side_effect=lp),
            patch("act.back_end.bab.bab.verify_bab_batched", side_effect=bab),
        ):
            results, err, _ = _verify_net(net, cfg)
        assert err is None and len(results) == 2
        assert tuple(limits) == expected_limits, limits
        expected_status = VerifyStatus.TIMEOUT if clock[0] >= cfg.timeout else VerifyStatus.UNKNOWN
        assert all(result.status == expected_status for result in results)

    def _milp_net(threshold: float = -0.1) -> Net:
        """Identical unstable ReLUs cancel exactly, but not in their relaxation."""
        from act.util.device_manager import get_default_device, get_default_dtype

        def tensor(value: Any) -> torch.Tensor:
            """Construct fixture tensors using the managed device and dtype."""
            return torch.tensor(value, device=get_default_device(), dtype=get_default_dtype())
        layers = [
            Layer(0, "INPUT", {"shape": (1, 1), "dtype": str(get_default_dtype())}, [], [0]),
            Layer(1, "INPUT_SPEC", {"kind": "BOX", "lb": tensor([[-1.0]]), "ub": tensor([[1.0]])}, [0], [0]),
            Layer(2, "DENSE", {"weight": tensor([[1.0], [1.0]]), "bias": tensor([0.0, 0.0]),
                               "in_features": 1, "out_features": 2}, [0], [1, 2]),
            Layer(3, "RELU", {}, [1, 2], [3, 4]),
            Layer(4, "DENSE", {"weight": tensor([[1.0, -1.0]]), "bias": tensor([0.0]),
                               "in_features": 2, "out_features": 1}, [3, 4], [5]),
            Layer(5, "ASSERT", OutputSpec(
                kind=OutKind.UNSAFE_LINEAR, c=tensor([[1.0]]), d=tensor([threshold]),
            ).encode_linear(B=1, n_out=1, device=get_default_device(), dtype=get_default_dtype()), [5], [5]),
        ]
        return Net(layers, {i: [i - 1] if i else [] for i in range(6)},
                   {i: [i + 1] if i < 5 else [] for i in range(6)})

    def _t_milp_config_and_default() -> None:
        """Defaults and invalid values remain enforced through the public config."""
        from unittest.mock import patch
        from act.config.config import BaBConfig
        from act.util.stats import VerifyResult

        cfg = BackendConfig(solver="torchlp", lp_enabled=False, bab_enabled=False)
        assert not cfg.bab.milp_escalation and not cfg.bab.terminal_lp_enabled
        assert cfg.bab.climb_terminal_cores
        assert (cfg.bab.milp_max_variables, cfg.bab.milp_max_constraints, cfg.bab.milp_timeout) == (2000, 2000, 60.0)
        assert BackendConfig.from_yaml().bab.milp_escalation is False
        parser = argparse.ArgumentParser()
        _add_dataclass_config_args(parser)
        assert parser.parse_args([]).bab_milp_escalation is None
        assert parser.parse_args(["--bab-milp-escalation"]).bab_milp_escalation is True
        assert parser.parse_args(["--no-bab-milp-escalation"]).bab_milp_escalation is False
        for key, values in (
            ("milp_timeout", (float("nan"), float("inf"), -1.0, 0.0)),
            ("milp_max_variables", (True, 1.5, 0)),
            ("milp_max_constraints", (False, 2.5, -1)),
        ):
            for value in values:
                invalid: dict[str, Any] = {key: value}
                try:
                    BaBConfig(**invalid)
                except ConfigError:
                    pass
                else:
                    raise AssertionError((key, value))
        import builtins
        original_import = builtins.__import__

        def checked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            """Reject numerical-tier imports while escalation is disabled."""
            assert name not in {"act.back_end.solver.milp_escalate", "act.back_end.solver.terminal_lp"}
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=checked_import), patch(
            "act.back_end.verifier.verify_once", return_value=[VerifyResult(VerifyStatus.UNKNOWN)],
        ), patch(
            f"{__name__}._resolve_bab_config",
            side_effect=AssertionError("Disabled refinement stages must not resolve presets"),
        ):
            result, error, _ = _verify_net(_milp_net(), cfg)
        assert error is None and result[0].status == VerifyStatus.UNKNOWN

    def _t_milp_boundaries() -> None:
        """Failure paths neither certify nor accept an unchecked incumbent."""
        from types import SimpleNamespace
        from unittest.mock import patch
        from act.back_end.solver import milp_escalate as milp
        from act.config.config import BaBConfig

        cfg = BaBConfig(milp_escalation=True)
        net = _milp_net()

        def reason(expected: str, **changes) -> None:
            result = milp.solve_milp(net, replace(cfg, **changes), deadline=time.monotonic() + 30.0)
            assert result.status == VerifyStatus.UNKNOWN, result
            assert result.metadata["milp_escalation"]["reason"].startswith(expected), result
            assert "certificate_kind" not in result.metadata

        reason("variable_limit", milp_max_variables=1)
        reason("variable_limit", milp_max_variables=6)
        reason("constraint_limit", milp_max_constraints=1)
        expired = milp.solve_milp(net, cfg, deadline=time.monotonic() - 1.0)
        assert expired.metadata["milp_escalation"]["reason"] == "deadline"
        for deadline in (float("nan"), float("inf")):
            assert milp.solve_milp(net, cfg, deadline=deadline).status == VerifyStatus.UNKNOWN
        for unsupported, expected in (
            (_build_tiny_net(B=2), "unsupported_batch"),
            (_build_tiny_net(), "unsupported_output"),
        ):
            result = milp.solve_milp(unsupported, cfg, deadline=time.monotonic() + 30.0)
            assert result.status == VerifyStatus.UNKNOWN and result.metadata["milp_escalation"]["reason"] == expected
        net.layers[3].kind = "TANH"
        reason("unsupported")
        net.layers[3].kind = "RELU"
        for status in ("infeasible_or_unbounded", "time_limit", "solver_status_12"):
            with patch.object(milp, "_optimize", return_value=(status, None)):
                reason(status)
        for value in (0.0, 2.0, float("nan")):
            candidate = net.layers[1].params["lb"].new_tensor([[value]])
            with patch.object(milp, "_optimize", return_value=("optimal", candidate)):
                reason("invalid_incumbent")
        with patch.object(milp, "import_gurobi", return_value=(False, None, None)):
            reason("unavailable")

        class LicenseError(Exception):
            """A deterministic license failure at the actual environment boundary."""
            errno = 10009

        def unavailable_env(**kwargs: Any) -> Any:
            """Fail at environment creation without contacting a license service."""
            raise LicenseError("missing")

        fake_gp = SimpleNamespace(GurobiError=LicenseError, Env=unavailable_env)
        fake_grb = SimpleNamespace(Error=SimpleNamespace(NO_LICENSE=10009))
        with patch.object(milp, "import_gurobi", return_value=(True, fake_gp, fake_grb)):
            reason("license_unavailable")

    def _t_milp_real() -> None:
        """Real tiny exact SAT/UNSAT solves, explicitly reporting absent licenses."""
        from unittest.mock import patch
        from act.back_end.solver.milp_escalate import solve_milp
        from act.config.config import BaBConfig

        cfg = BaBConfig(milp_escalation=True)

        def solve(net: Net) -> Any:
            """Exercise real Gurobi while rejecting whole sparse-block densification."""
            with patch.object(torch.Tensor, "to_dense", side_effect=AssertionError("MILP matrices must stay sparse")):
                return solve_milp(net, cfg, deadline=time.monotonic() + 30.0)

        for threshold, expected in ((-0.1, VerifyStatus.CERTIFIED), (0.0, VerifyStatus.FALSIFIED)):
            result = solve(_milp_net(threshold))
            metadata = result.metadata["milp_escalation"]
            if metadata["reason"] == "unavailable" or metadata["reason"].startswith("license_unavailable"):
                print(f"  SKIP  real tiny MILP: {metadata['reason']}")
                return
            assert result.status == expected, result
            assert metadata["binaries"] == 2
            if expected == VerifyStatus.CERTIFIED:
                assert result.metadata["certificate_kind"] == "milp_status_infeasible"
            else:
                assert result.counterexample is not None
        stable = _milp_net(0.0)
        stable.layers[1].params["lb"].fill_(1.0)
        stable.layers[1].params["ub"].fill_(2.0)
        result = solve(stable)
        assert result.status == VerifyStatus.FALSIFIED and result.metadata["milp_escalation"]["binaries"] == 0

    def _t_milp_cascade() -> None:
        """Escalation consumes the common deadline and retains reasons after BaB."""
        from unittest.mock import patch
        from act.back_end.solver import milp_escalate
        from act.config.config import BaBConfig
        from act.util.stats import VerifyResult

        for duration in (1.0, 11.0):
            clock = [0.0]
            cfg = BackendConfig(solver="torchlp", timeout=10.0, lp_enabled=False, bab_enabled=True,
                                bab=BaBConfig(milp_escalation=True))

            def escalation(net: Net, config: BaBConfig, *, deadline: float) -> VerifyResult:
                """Consume root budget without making a terminal claim."""
                assert deadline == 10.0 and config.milp_escalation
                clock[0] += 3.0
                return VerifyResult(VerifyStatus.UNKNOWN, metadata={"milp_escalation": {"reason": "license_unavailable"}})

            def bab(*args: Any, time_budget_s: float, **kwargs: Any) -> VerifyResult:
                """Assert the remaining budget and simulate a subsequent result."""
                assert time_budget_s == 7.0
                clock[0] += duration
                return VerifyResult(VerifyStatus.UNKNOWN, metadata={"bab": True})

            with (
                patch.object(time, "monotonic", side_effect=lambda: clock[0]),
                patch("act.back_end.verifier.verify_once", return_value=[VerifyResult(VerifyStatus.UNKNOWN)]),
                patch.object(milp_escalate, "solve_milp", side_effect=escalation),
                patch("act.back_end.bab.bab.verify_bab_batched", side_effect=bab),
            ):
                results, error, _ = _verify_net(_milp_net(), cfg)
            assert error is None
            assert results[0].status == (VerifyStatus.UNKNOWN if duration == 1.0 else VerifyStatus.TIMEOUT)
            assert results[0].metadata == {"bab": True, "milp_escalation": {"reason": "license_unavailable"}}

    print("cli.py cascade smoke tests")
    _check("tier-1 interval verify_once returns VerifyStatus", _t_interval_only)
    _check("cascade with lp_enabled=True dispatches tier-2 on UNKNOWN", _t_cascade_default_config)
    _check("lp_enabled=False skips tier-2", _t_lp_disabled_skips_tier2)
    _check("default bab_max_batch_size=8", _t_bab_max_batch_size_default)
    _check("frontier CLI budgets and automatic K", _t_frontier_config)
    _check("dual cascade accepts unsafe-linear escape and rejects ties", _t_dual_unsafe_linear)
    _check("cascade shares one deadline across stages and lanes", _t_cascade_shares_deadline)
    _check("expired LP budget skips BaB", lambda: _t_cascade_shares_deadline(3.0, 8.0, (7.0,)))
    _check("expired propagation budget skips refinements", lambda: _t_cascade_shares_deadline(11.0, 0.0, ()))
    _check("MILP defaults, invalid config and default-off imports", _t_milp_config_and_default)
    _check("MILP unsupported, limits, deadlines, license and incumbent boundaries", _t_milp_boundaries)
    _check("real tiny exact MILP outcomes (or explicit license skip)", _t_milp_real)
    _check("MILP reason survives BaB and shared deadline timeout", _t_milp_cascade)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        sys.exit(_run_cli_cascade_smoke())
    sys.exit(main())
