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
import os
import sys
from pathlib import Path
from typing import Optional

from act.util.cli_utils import add_device_args, initialize_from_args


def _make_solver(solver_name: str):
    """Instantiate a solver backend by name."""
    from act.back_end.solver.solver_interval import TorchLPSolver

    if solver_name == "gurobi":
        from act.back_end.solver.solver_gurobi import GurobiSolver

        return GurobiSolver()
    if solver_name == "torch":
        return TorchLPSolver()
    # "auto": try Gurobi, fall back to TorchLP
    try:
        from act.back_end.solver.solver_gurobi import GurobiSolver

        return GurobiSolver()
    except Exception:
        return TorchLPSolver()


def run_verification(args, backend_cfg):
    """Run verification on a network using *backend_cfg*."""
    print(f"\n{'=' * 80}")
    print(f"ACT BACK-END VERIFICATION")
    print(f"{'=' * 80}\n")

    from act.back_end.serialization.serialization import load_net_from_file
    from act.back_end.verifier import verify_once
    from act.back_end.bab import verify_bab
    from act.util.stats import VerifyStatus

    print(f"Loading network from: {args.network}")
    net = load_net_from_file(args.network)
    print(f"Loaded network with {len(net.layers)} layers")

    solver = _make_solver(backend_cfg.solver)
    print(f"Solver: {backend_cfg.solver}")

    if backend_cfg.bab_enabled:
        bab = backend_cfg.bab
        print(f"\nRunning Branch-and-Bound verification...")
        print(f"  Max depth: {bab.max_depth}")
        print(f"  Max subproblems: {bab.max_nodes}")
        print(f"  Branching: {bab.branching_method}")
        print(f"  Bounding: {bab.bounding_method}")
        print(f"  Timeout: {backend_cfg.timeout}s\n")

        result = verify_bab(
            net=net,
            solver=solver,
            config=bab,
            time_budget_s=backend_cfg.timeout,
        )
    else:
        print(f"\nRunning single-shot verification...")
        print(f"  Timeout: {backend_cfg.timeout}s\n")

        result = verify_once(net=net, solver=solver, timelimit=backend_cfg.timeout)

    print(f"\n{'=' * 80}")
    print(f"VERIFICATION RESULT: {result.status}")
    print(f"{'=' * 80}")

    if "time" in result.metadata:
        print(f"Time: {result.metadata['time']:.3f}s")

    if result.counterexample is not None:
        print(f"Counterexample found:")
        print(f"  Shape: {result.counterexample.shape}")
        if backend_cfg.verbose:
            print(f"  Values: {result.counterexample}")

    if backend_cfg.verbose and result.metadata:
        print(f"\nVerification metadata:")
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")

    print(f"\n{'=' * 80}\n")

    return 0 if result.status == VerifyStatus.CERTIFIED else 1


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
    categories: dict = {}
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
        choices=["interval", "hybridz", "dual"],
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
        choices=["auto", "gurobi", "torch"],
        default=None,
        help="Solver backend (default: from config.yaml / $ACT_SOLVER / 'auto')",
    )
    verify_group.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=None,
        help="Verification timeout in seconds (default: from config.yaml)",
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


def _collect_backend_overrides(args, _user_set) -> dict:
    """Build overrides dict from CLI flags + env vars.

    Only includes keys the user explicitly provided (CLI flag) or that are
    set in the environment.  Everything else falls through to config.yaml.

    Prefix conventions: ``bab_<field>`` → BaBConfig, ``gen_<field>`` → GenerationConfig.
    """
    overrides: dict = {}

    # ── Runtime selectors: CLI > env > config.yaml ──
    if args.solver is not None:
        overrides["solver"] = args.solver
    elif os.environ.get("ACT_SOLVER"):
        overrides["solver"] = os.environ["ACT_SOLVER"]

    if _user_set("--device"):
        overrides["device"] = args.device
    elif os.environ.get("ACT_DEVICE"):
        overrides["device"] = os.environ["ACT_DEVICE"]

    if _user_set("--dtype"):
        overrides["dtype"] = args.dtype
    elif os.environ.get("ACT_DTYPE"):
        overrides["dtype"] = os.environ["ACT_DTYPE"]

    if args.verbose:
        overrides["verbose"] = True

    # ── Verification ──
    if args.timeout is not None:
        overrides["timeout"] = args.timeout

    # bab enabled: --bab / --no-bab (None = defer to config.yaml)
    if args.bab is not None:
        overrides["bab_enabled"] = args.bab

    if args.bab_max_depth is not None:
        overrides["bab_max_depth"] = args.bab_max_depth
    if args.bab_max_subproblems is not None:
        overrides["bab_max_nodes"] = args.bab_max_subproblems
    if args.bab_branching is not None:
        overrides["bab_branching_method"] = args.bab_branching
    if args.bab_bounding is not None:
        overrides["bab_bounding_method"] = args.bab_bounding

    # ── Generation: CLI > env > config.yaml ──
    config_flag = getattr(args, "config", None)
    if config_flag is not None:
        overrides["gen_gen_config_path"] = config_flag

    output_flag = getattr(args, "output", None)
    if output_flag is not None:
        overrides["gen_output_dir"] = output_flag
    elif os.environ.get("ACT_GEN_OUTPUT"):
        overrides["gen_output_dir"] = os.environ["ACT_GEN_OUTPUT"]

    num_flag = getattr(args, "num", None)
    if num_flag is not None:
        overrides["gen_num_instances"] = num_flag
    elif os.environ.get("ACT_GEN_NUM"):
        overrides["gen_num_instances"] = int(os.environ["ACT_GEN_NUM"])

    seed_flag = getattr(args, "base_seed", None)
    if seed_flag is not None:
        overrides["gen_base_seed"] = seed_flag
    elif os.environ.get("ACT_GEN_SEED"):
        overrides["gen_base_seed"] = int(os.environ["ACT_GEN_SEED"])

    prefix_flag = getattr(args, "name_prefix", None)
    if prefix_flag is not None:
        overrides["gen_name_prefix"] = prefix_flag

    tf_flag = getattr(args, "tf_targets", None)
    if tf_flag is not None:
        overrides["gen_tf_targets"] = tf_flag

    reg_flag = getattr(args, "registry_mode", None)
    if reg_flag is not None and _user_set("--registry-mode"):
        overrides["gen_registry_mode"] = reg_flag

    return overrides


if __name__ == "__main__":
    sys.exit(main())
