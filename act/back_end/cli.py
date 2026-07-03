#!/usr/bin/env python3
"""
ACT Back-End Command-Line Interface.

Provides CLI tools for core verification operations:
- Network verification (single-shot and branch-and-bound)
- Network factory (generate example networks from YAML)
- Network serialization (save/load ACT Net structures)
- Analysis and constraint inspection

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

import argparse
import datetime
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union, cast

from act.back_end.config import VALID_BERT_METHODS, _VALID_SOLVERS
from act.back_end.layer_schema import LayerKind
from act.front_end.specs import OutKind
from act.util.cli_utils import add_device_args, initialize_from_args


_TF_MODES: tuple[str, ...] = ("interval", "hybridz")
_SOLVERS: tuple[str, ...] = tuple(sorted(_VALID_SOLVERS))


class _SkipUnsupported(NamedTuple):
    """Tagged-union result for nets the active TF cannot verify.

    Replaces an earlier ``"SKIP_UNSUPPORTED: ..."`` string-sentinel encoding
    that risked false-negative skips if any unrelated ``load_net_from_file``
    exception happened to start with the same prefix. ``isinstance(err,
    _SkipUnsupported)`` is unambiguous.
    """
    tf_name: str
    kinds: tuple[str, ...]


def _make_solver(solver_name: str):
    """LP-cascade solver factory (gurobi / torchlp / auto). Dual is routed
    separately via ``is_dual_solver_active`` since it implements
    ``compute_certified_bound``, not ``solve_batch``.
    """
    from act.back_end.solver.solver_torchlp import TorchLPSolver

    if solver_name == "gurobi":
        from act.back_end.solver.solver_gurobi import GurobiSolver

        return GurobiSolver()
    if solver_name == "torchlp":
        return TorchLPSolver()
    # "auto": try Gurobi, fall back to TorchLP
    try:
        from act.back_end.solver.solver_gurobi import GurobiSolver

        return GurobiSolver()
    except Exception:
        return TorchLPSolver()


def _verify_one_net(
    net_path: str, backend_cfg
) -> tuple[list[Any], Optional[Union[_SkipUnsupported, str]], Optional[int]]:
    """[BATCHED-API] Verify *net_path* via 3-tier cascade.

    Returns ``(results, err, n_layers)`` where ``err`` is one of:
      * ``None`` on success
      * ``_SkipUnsupported(tf_name, kinds)`` when the active TF cannot handle
        the net (unsupported layer kinds and/or unsupported ASSERT spec).
        Treated as a clean skip by callers, NOT as a verifier bug.
      * ``str`` for any other exception (genuine error).

    Tier 1 — interval (verify_once): always runs; certifies or falsifies via
              pure-tensor bounds propagation.
    Tier 2 — LP-batched (verify_lp_batched): runs on UNKNOWN lanes when
              backend_cfg.lp_enabled is True AND active TF propagates LP
              constraints. Skipped under DualTF (see soundness note below).
    Tier 3 — BaB (verify_bab_batched): runs on remaining UNKNOWN lanes when
              backend_cfg.bab_enabled is True AND active TF propagates LP
              constraints. bab_max_batch_size=1 disables K-batching.
    """
    from act.back_end.bab.bab import clear_violation_check_module_cache
    from act.back_end.serialization.serialization import load_net_from_file
    from act.back_end.transfer_functions import ensure_active_tf, is_dual_solver_active
    from act.back_end.verifier import verify_once, verify_lp_batched
    from act.util.stats import VerifyStatus

    clear_violation_check_module_cache()

    try:
        net = load_net_from_file(net_path, target_device=backend_cfg.device)
        n_layers = len(net.layers)

        active_tf = ensure_active_tf("interval")
        is_dual = is_dual_solver_active()

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
        # DualSolver rejects UNSAFE_LINEAR (EXISTS quantifier not representable
        # by sound dual lower bounds — see solver_dual.evaluate_spec). Detect
        # this distinct skip reason in parallel with the layer-kind check so
        # users see ALL skip reasons in one pass, not sequentially.
        unsupported_specs: List[str] = []
        if is_dual:
            for L in net.layers:
                if (
                    L.kind == LayerKind.ASSERT.value
                    and L.params.get("kind") == OutKind.UNSAFE_LINEAR
                ):
                    unsupported_specs.append(f"{LayerKind.ASSERT.value}:{OutKind.UNSAFE_LINEAR}")
                    break

        blocking = tuple(unsupported_kinds + unsupported_specs)
        if blocking:
            return [], _SkipUnsupported(tf_name=authority_name, kinds=blocking), n_layers

        results: List[Any] = list(verify_once(net=net))

        any_unknown = any(r.status == VerifyStatus.UNKNOWN for r in results)

        # SOUNDNESS-CRITICAL: under --solver dual, verify_once already ran
        # DualSolver.evaluate_spec (linear-relaxation dual backward) — Tier-2 LP and Tier-3 BaB
        # would build under-constrained LPs (DualSolver does not produce LP-feed
        # ConSet entries; the forward analyze() pipeline is bypassed) and emit
        # spurious FALSIFIED. Do not remove this gate without first switching
        # back to an LP-feeding forward TF (interval / hybridz).
        if any_unknown and backend_cfg.lp_enabled and not is_dual:
            try:
                lp_results = verify_lp_batched(
                    net,
                    solver_factory=lambda: _make_solver(backend_cfg.solver),
                    timelimit=backend_cfg.timeout,
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

        if any_unknown and backend_cfg.bab_enabled:
            # verify_bab_batched operates on a single-instance (B=1) net and
            # returns one VerifyResult. For multi-sample nets we slice per-lane
            # and dispatch one BaB call per still-UNKNOWN sample.
            from act.back_end.bab.bab import verify_bab_batched as _vbb
            from act.back_end.verifier import slice_net_to_sample

            bab_cfg = backend_cfg.bab
            if is_dual:
                # Dual-tier BaB: optimized alpha/eta bounds with gain-tested
                # joint multi-neuron (verdict-boundary) branching.
                import dataclasses

                bab_cfg = dataclasses.replace(
                    bab_cfg,
                    solver_tier="dual_alpha_eta",
                    branching_method="gain",
                    reuse_root_bounds=True,
                    intermediate_refine="all",
                    multi_split_levels=4,
                )

            try:
                results = [
                    _vbb(
                        slice_net_to_sample(net, i),
                        solver_factory=lambda: _make_solver(backend_cfg.solver),
                        config=bab_cfg,
                        max_batch_size=backend_cfg.bab_max_batch_size,
                        time_budget_s=backend_cfg.timeout,
                    )
                    if results[i].status == VerifyStatus.UNKNOWN
                    else results[i]
                    for i in range(len(results))
                ]
            except NotImplementedError as e:
                # Same graceful-degradation contract as the Tier-2 LP gate:
                # BaB also runs through cons_exportor (via setup_and_solve_batch
                # in bab.py) and hits the same "unsupported tag" failure on
                # AVGPOOL2D / MAXPOOL2D / GELU / etc. Keep Tier-1 results;
                # reraise unrelated NIE so genuine bugs surface as ERROR.
                if "export_to_batch_problem" not in str(e):
                    raise

        return results, None, n_layers
    except Exception as e:  # noqa: BLE001 — surface per-net error, keep iterating
        return [], str(e), None


def run_verification(args, backend_cfg):
    """Run verification on a network using *backend_cfg*."""
    from act.util.stats import VerifyStatus

    results, err, n_layers = _verify_one_net(args.network, backend_cfg)
    if err is not None:
        if isinstance(err, _SkipUnsupported):
            print(
                f"⏭️  {args.network}: {err.tf_name} cannot handle: "
                f"{','.join(err.kinds)}"
            )
            return 0
        print(f"❌ {args.network}: {err}")
        return 1
    print(f"Loaded {n_layers}-layer net; solver={backend_cfg.solver}")

    valid_outcomes = (
        VerifyStatus.CERTIFIED,
        VerifyStatus.FALSIFIED,
        VerifyStatus.UNKNOWN,
        VerifyStatus.TIMEOUT,
    )
    multi = len(results) > 1
    for i, result in enumerate(results):
        prefix = f"Sample {i}: " if multi else f"Lane {i}: "
        print(f"{prefix}{result.status}")
        if backend_cfg.verbose and result.metadata:
            for k, v in result.metadata.items():
                print(f"  {k}: {v}")
    return 0 if all(r.status in valid_outcomes for r in results) else 1


def run_network_factory(args, backend_cfg):
    """Generate example networks using TF-aware NetFactory."""
    print(f"\n{'=' * 80}")
    print(f"ACT NETWORK FACTORY")
    print(f"{'=' * 80}\n")

    from act.back_end.net_factory import NetFactory

    gen = backend_cfg.generation

    if gen.tf_targets:
        print(f"TF targets: {gen.tf_targets} (mode: {gen.registry_mode})")
    print(f"Config: {gen.gen_config_path}")
    print(f"Output: {gen.output_dir}")
    print(f"Instances: {gen.num_instances}, Seed: {gen.base_seed}")
    print()

    try:
        factory = NetFactory(
            gen_config_path=gen.gen_config_path,
            output_dir=gen.output_dir,
            base_seed=gen.base_seed,
            num_instances=gen.num_instances,
            name_prefix=gen.name_prefix,
            tf_targets=gen.tf_targets,
            registry_mode=gen.registry_mode,
            write_manifest=gen.write_manifest,
        )
        factory.generate()
        print(f"\n{'=' * 80}")
        print(f"✓ Network generation complete")
        print(f"{'=' * 80}\n")

        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if backend_cfg.verbose:
            import traceback

            traceback.print_exc()
        return 1


def run_network_info(args):
    """Display information about a network."""
    print(f"\n{'=' * 80}")
    print(f"NETWORK INFORMATION")
    print(f"{'=' * 80}\n")

    from act.back_end.serialization.serialization import load_net_from_file
    from act.back_end.layer_schema import LayerKind

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
        print(f"\n{'=' * 80}")
        print(f"DETAILED LAYER INFORMATION")
        print(f"{'=' * 80}\n")

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

    print(f"{'=' * 80}\n")
    return 0


def run_serialization_test(args):
    """Test network serialization (save/load round-trip)."""
    print(f"\n{'=' * 80}")
    print(f"SERIALIZATION TEST")
    print(f"{'=' * 80}\n")

    from act.back_end.serialization.test_serialization import main as test_main

    print("Running serialization tests...\n")
    result = test_main()

    print(f"\n{'=' * 80}")
    if result == 0:
        print("✓ All serialization tests passed")
    else:
        print("❌ Some serialization tests failed")
    print(f"{'=' * 80}\n")

    return result


def list_examples(args):
    """List available example networks."""
    print(f"\n{'=' * 80}")
    print(f"AVAILABLE EXAMPLE NETWORKS")
    print(f"{'=' * 80}\n")

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
        print("-" * 70)
        for name, info in sorted(nets):
            shape = info.get("input_shape", "?")
            layers = info.get("num_layers", "?")
            print(f"  {name:40s}  shape={shape}  layers={layers}")
        print()

    print(f"{'=' * 80}")
    print("To generate networks: python -m act.back_end --generate")
    print(f"{'=' * 80}\n")

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

    print(f"\n{'=' * 80}")
    print(f"ACT BENCH: {kind.upper()}")
    print(f"{'=' * 80}\n")

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

    print(f"\n{'=' * 80}")
    print(f"Bench complete")
    print(f"{'=' * 80}\n")
    return 0


def run_diff_nets(args) -> int:
    """Load two ACT Net JSON files and print a unified-diff-style layer comparison."""
    from act.back_end.serialization.serialization import load_net_from_file

    path_a, path_b = args.diff_nets

    try:
        net_a = load_net_from_file(path_a)
    except Exception as e:
        print(f"Error loading {path_a}: {e}")
        return 1

    try:
        net_b = load_net_from_file(path_b)
    except Exception as e:
        print(f"Error loading {path_b}: {e}")
        return 1

    print(f"\n{'=' * 80}")
    print(f"NET DIFF")
    print(f"  A: {path_a}")
    print(f"  B: {path_b}")
    print(f"{'=' * 80}\n")

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

    print(f"\n{'=' * 80}\n")
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
        "--solver",
        "-s",
        type=str,
        choices=_SOLVERS,
        default=None,
        dest="solver",
        help=(
            "Solver backend.  Three alternative families:\n"
            "  'gurobi'  — commercial MILP/LP (license required).  LP cascade.\n"
            "  'torchlp' — PyTorch-tensor LP (Adam + penalty + box projection,\n"
            "              GPU-capable).  LP cascade.\n"
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
            "'interval').  For dual certified bounds, use --solver dual instead "
            "(dual is a solver, not a TF — see --solver help)."
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
        "--bab-branching",
        type=str,
        default=None,
        dest="bab_branching",
        help="Branching strategy (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-bounding",
        type=str,
        default=None,
        dest="bab_bounding",
        help="Bounding strategy (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-bounding-order",
        type=str,
        default=None,
        choices=["depth_lb", "greedy", "sa"],
        dest="bab_bounding_order",
        help="TopKBounding order policy (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-sa-cooling-rate",
        type=float,
        default=None,
        dest="bab_sa_cooling_rate",
        help="Cooling rate for --bab-bounding-order sa (default: from config.yaml)",
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
        default=False,
        dest="bab_llm_probe_enabled",
        help="Enable the LLM-probe BaB controller (default: from config.yaml)",
    )
    verify_group.add_argument(
        "--bab-llm-probe-backend",
        type=str,
        default=None,
        choices=["mock", "openrouter", "openai", "glm", "minimax"],
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
        "--multi-split-levels",
        type=int,
        default=None,
        dest="bab_multi_split_levels",
        help="Simultaneous neuron splits per branching step for gain branching (default: from config.yaml)",
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
        help="Path to backend YAML config (default: act/back_end/config.yaml)",
    )

    # Common options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Add standard device/dtype arguments (shared across all ACT CLIs)
    add_device_args(parser)

    # Detect user-provided flags BEFORE parsing so env vars / config.yaml
    # can serve as fallbacks without overriding explicit CLI flags.
    argv = sys.argv[1:]
    _user_set = lambda flag: any(  # noqa: E731
        a == flag or a.startswith(flag + "=") for a in argv
    )

    args = parser.parse_args()

    # Validate arguments based on command
    if args.verify or args.info:
        if not args.network:
            parser.error("--network is required for --verify and --info")

    if args.diff_nets:
        return run_diff_nets(args)

    # ── Build BackendConfig ──────────────────────────────────────────────
    # Load YAML as baseline, then overlay env vars and CLI flags on top.
    # Precedence: CLI flag > env var > config.yaml > dataclass default
    from act.back_end.config import BackendConfig

    backend_cfg = BackendConfig.from_yaml(
        config_path=args.backend_config,
        **_collect_backend_overrides(args, _user_set),
    )

    # Initialize device manager from the resolved config
    import argparse as _ap

    initialize_from_args(
        _ap.Namespace(device=backend_cfg.device, dtype=backend_cfg.dtype)
    )

    if args.tf_mode is not None:
        from act.back_end.analyze import initialize_tf_mode

        initialize_tf_mode(args.tf_mode)

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
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


# (override_key, args_attr, env_var, env_cast, cli_check)
# cli_check="user_set" for flags with non-None defaults (--device/--dtype/--registry-mode)
_BACKEND_OVERRIDE_SPEC: list[tuple[str, str, Optional[str], Any, str]] = [
    ("solver",               "solver",              "ACT_SOLVER",     None, "not_none"),
    ("device",               "device",              "ACT_DEVICE",     None, "user_set"),
    ("dtype",                "dtype",               "ACT_DTYPE",      None, "user_set"),
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
    ("bab_bounding_method",  "bab_bounding",        None,             None, "not_none"),
    ("bab_bounding_order",   "bab_bounding_order",  None,             None, "not_none"),
    ("bab_sa_cooling_rate",  "bab_sa_cooling_rate", None,             None, "not_none"),
    ("bab_frontier_cap",     "bab_frontier_cap",    None,             None, "not_none"),
    ("bab_input_split_fanout", "bab_input_split_fanout", None,          None, "not_none"),
    ("bab_llm_probe_enabled", "bab_llm_probe_enabled",  None,             None, "user_set"),
    ("bab_llm_probe_backend", "bab_llm_probe_backend",  None,             None, "not_none"),
    ("bab_llm_probe_model",   "bab_llm_probe_model",    None,             None, "not_none"),
    ("bab_llm_probe_base_url", "bab_llm_probe_base_url", None,            None, "not_none"),
    ("bab_llm_probe_cadence", "bab_llm_probe_cadence",  None,             int,  "not_none"),
    ("bab_multi_split_levels",           "bab_multi_split_levels",           None, int,   "not_none"),
    ("gen_gen_config_path",  "config",              None,             None, "not_none"),
    ("gen_output_dir",       "output",              "ACT_GEN_OUTPUT", None, "not_none"),
    ("gen_num_instances",    "num",                 "ACT_GEN_NUM",    int,  "not_none"),
    ("gen_base_seed",        "base_seed",           "ACT_GEN_SEED",   int,  "not_none"),
    ("gen_name_prefix",      "name_prefix",         None,             None, "not_none"),
    ("gen_tf_targets",       "tf_targets",          None,             None, "not_none"),
    ("gen_registry_mode",    "registry_mode",       None,             None, "user_set"),
]


def _collect_backend_overrides(args: Any, _user_set: Any) -> dict[str, Any]:
    """Build overrides dict from CLI flags + env vars (precedence: CLI > env > yaml)."""
    overrides: dict[str, Any] = {}
    for key, attr, env, cast, check in _BACKEND_OVERRIDE_SPEC:
        cli_val = getattr(args, attr, None)
        if check == "user_set":
            cli_provided = _user_set(f"--{attr.replace('_', '-')}")
        else:
            cli_provided = cli_val is not None
        if cli_provided:
            overrides[key] = cli_val
        elif env is not None and os.environ.get(env):
            overrides[key] = cast(os.environ[env]) if cast else os.environ[env]

    if args.verbose:
        overrides["verbose"] = True

    return overrides


def _run_cli_cascade_smoke() -> int:
    """Light integration smoke: build a tiny net, run the 3-tier cascade, verify dispatch."""
    import torch
    from act.back_end.core import Layer, Net, Bounds, Fact, ConSet
    from act.back_end.layer_schema import LayerKind
    from act.front_end.specs import OutputSpec
    from act.back_end.config import BackendConfig
    from act.util.stats import VerifyStatus

    passed = 0
    failed = 0

    def _check(label: str, fn) -> None:
        nonlocal passed, failed
        try:
            fn()
            print(f"  PASS  {label}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {label}: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

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
                solver_factory=lambda: _make_solver(cfg.solver),
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

    print("cli.py cascade smoke tests")
    _check("tier-1 interval verify_once returns VerifyStatus", _t_interval_only)
    _check("cascade with lp_enabled=True dispatches tier-2 on UNKNOWN", _t_cascade_default_config)
    _check("lp_enabled=False skips tier-2", _t_lp_disabled_skips_tier2)
    _check("default bab_max_batch_size=8", _t_bab_max_batch_size_default)

    print(f"\n{passed}/{passed + failed} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        sys.exit(_run_cli_cascade_smoke())
    sys.exit(main())
