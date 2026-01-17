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
import sys
from pathlib import Path
from typing import Optional

from act.util.cli_utils import add_device_args, initialize_from_args


def run_verification(args):
    """Run verification on a network."""
    print(f"\n{'='*80}")
    print(f"ACT BACK-END VERIFICATION")
    print(f"{'='*80}\n")
    
    # Import here to avoid circular dependencies
    from act.back_end.core import Net
    from act.back_end.serialization.serialization import load_net_from_file
    from act.back_end.verifier import verify_once
    from act.back_end.bab import verify_bab
    from act.back_end.solver.solver_gurobi import GurobiSolver
    from act.back_end.solver.solver_torch import TorchLPSolver
    
    # Load network
    print(f"Loading network from: {args.network}")
    net, metadata = load_net_from_file(args.network)
    print(f"✓ Loaded network with {len(net.layers)} layers")
    
    # Select solver
    if args.solver == 'gurobi':
        print(f"Using solver: Gurobi")
        solver = GurobiSolver()
    elif args.solver == 'torch':
        print(f"Using solver: TorchLP")
        solver = TorchLPSolver()
    else:  # auto
        try:
            from act.back_end.solver.solver_gurobi import GurobiSolver
            print(f"Using solver: Gurobi (auto-detected)")
            solver = GurobiSolver()
        except:
            print(f"Using solver: TorchLP (Gurobi not available)")
            solver = TorchLPSolver()
    
    # Run verification
    if args.bab:
        print(f"\nRunning Branch-and-Bound verification...")
        print(f"  Max depth: {args.bab_max_depth}")
        print(f"  Max subproblems: {args.bab_max_subproblems}")
        print(f"  Timeout: {args.timeout}s\n")
        
        result = verify_bab(
            net=net,
            solver=solver,
            timelimit=args.timeout,
            max_depth=args.bab_max_depth,
            max_subproblems=args.bab_max_subproblems,
            verbose=args.verbose
        )
    else:
        print(f"\nRunning single-shot verification...")
        print(f"  Timeout: {args.timeout}s\n")
        
        result = verify_once(
            net=net,
            solver=solver,
            timelimit=args.timeout
        )
    
    # Display results
    print(f"\n{'='*80}")
    print(f"VERIFICATION RESULT: {result.status}")
    print(f"{'='*80}")
    
    if 'time' in result.stats:
        print(f"Time: {result.stats['time']:.3f}s")
    
    if result.counterexample is not None:
        print(f"Counterexample found:")
        print(f"  Shape: {result.counterexample.shape}")
        if args.verbose:
            print(f"  Values: {result.counterexample}")
    
    if args.verbose and result.stats:
        print(f"\nVerification stats:")
        for key, value in result.stats.items():
            print(f"  {key}: {value}")
    
    print(f"\n{'='*80}\n")
    
    return 0 if result.status == "CERTIFIED" else 1


def run_network_factory(args):
    """Generate example networks from YAML configuration."""
    print(f"\n{'='*80}")
    print(f"ACT NETWORK FACTORY")
    print(f"{'='*80}\n")
    
    from act.back_end.net_factory import NetFactory
    
    config_file = args.config if args.config else "act/back_end/examples/examples_config.yaml"
    
    print(f"Configuration: {config_file}")
    print(f"Output directory: act/back_end/examples/nets (fixed)\n")
    
    if args.output:
        print(f"⚠️  Note: Custom output directory not supported by NetFactory")
        print(f"   Networks will be saved to: act/back_end/examples/nets\n")
    
    try:
        factory = NetFactory(config_file)
        factory.generate_all()
        
        print(f"\n{'='*80}")
        print(f"✓ Network generation complete")
        print(f"{'='*80}\n")
        
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_network_info(args):
    """Display information about a network."""
    print(f"\n{'='*80}")
    print(f"NETWORK INFORMATION")
    print(f"{'='*80}\n")
    
    from act.back_end.serialization.serialization import load_net_from_file
    from act.back_end.layer_schema import LayerKind
    
    print(f"Loading network from: {args.network}\n")
    net, metadata = load_net_from_file(args.network)
    
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
        print(f"\n{'='*80}")
        print(f"DETAILED LAYER INFORMATION")
        print(f"{'='*80}\n")
        
        for layer in net.layers:
            print(f"Layer {layer.id}: {layer.kind}")
            print(f"  In vars: {layer.in_vars}")
            print(f"  Out vars: {layer.out_vars}")
            if layer.params:
                print(f"  Params: {list(layer.params.keys())}")
            if layer.meta:
                print(f"  Meta: {layer.meta}")
            
            # Show predecessors
            preds = net.preds.get(layer.id, [])
            if preds:
                print(f"  Predecessors: {preds}")
            
            # Show successors
            succs = net.succs.get(layer.id, [])
            if succs:
                print(f"  Successors: {succs}")
            print()
    
    print(f"{'='*80}\n")
    return 0


def run_serialization_test(args):
    """Test network serialization (save/load round-trip)."""
    print(f"\n{'='*80}")
    print(f"SERIALIZATION TEST")
    print(f"{'='*80}\n")
    
    from act.back_end.serialization.test_serialization import main as test_main
    
    print("Running serialization tests...\n")
    result = test_main()
    
    print(f"\n{'='*80}")
    if result == 0:
        print("✓ All serialization tests passed")
    else:
        print("❌ Some serialization tests failed")
    print(f"{'='*80}\n")
    
    return result


def list_examples(args):
    """List available example networks."""
    print(f"\n{'='*80}")
    print(f"AVAILABLE EXAMPLE NETWORKS")
    print(f"{'='*80}\n")
    
    import yaml
    from pathlib import Path
    
    config_file = "act/back_end/examples/examples_config.yaml"
    
    if not Path(config_file).exists():
        print(f"❌ Configuration file not found: {config_file}")
        return 1
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    networks = config.get('networks', {})
    
    print(f"Total networks: {len(networks)}\n")
    
    # Group by category
    categories = {}
    for net_name, net_config in networks.items():
        desc = net_config.get('description', 'No description')
        # Try to infer category from name
        if 'mnist' in net_name.lower():
            cat = 'MNIST Classification'
        elif 'cifar' in net_name.lower():
            cat = 'CIFAR Classification'
        elif 'control' in net_name.lower():
            cat = 'Control Systems'
        elif 'reachability' in net_name.lower():
            cat = 'Reachability Analysis'
        else:
            cat = 'Other'
        
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((net_name, desc))
    
    for category, nets in sorted(categories.items()):
        print(f"{category} ({len(nets)} networks):")
        print("-" * 80)
        for net_name, desc in sorted(nets):
            print(f"  {net_name:30s} - {desc}")
        print()
    
    print(f"{'='*80}")
    print("To generate networks: python -m act.back_end --generate")
    print(f"{'='*80}\n")
    
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
  # CONFIGNET - Sample and materialize networks
  # ============================================================================

  # Sample instance specs only
  python -m act.back_end --confignet sample --num 3

  # Generate and materialize into examples_config.yaml
  python -m act.back_end --confignet generate --num 2 --device cpu --dtype float64

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
        """
    )
    
    # Command groups
    cmd_group = parser.add_mutually_exclusive_group(required=True)
    
    cmd_group.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Generate example networks from YAML configuration"
    )
    cmd_group.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Run verification on a network"
    )
    cmd_group.add_argument(
        "--info", "-i",
        action="store_true",
        help="Display network information"
    )
    cmd_group.add_argument(
        "--list-examples", "-l",
        action="store_true",
        dest="list_examples",
        help="List available example networks"
    )
    cmd_group.add_argument(
        "--confignet",
        action="store_true",
        help="Run Confignet subcommands (sample/generate)"
    )
    cmd_group.add_argument(
        "--test-serialization",
        action="store_true",
        dest="test_serialization",
        help="Run serialization tests"
    )
    
    # Network factory options
    factory_group = parser.add_argument_group("Network Factory Options")
    factory_group.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file (default: act/back_end/examples/examples_config.yaml)"
    )
    factory_group.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for generated networks (default: act/back_end/examples/nets)"
    )
    
    # Verification options
    verify_group = parser.add_argument_group("Verification Options")
    verify_group.add_argument(
        "--network", "-n",
        type=str,
        help="Path to network file (JSON format)"
    )
    verify_group.add_argument(
        "--solver", "-s",
        type=str,
        choices=['auto', 'gurobi', 'torch'],
        default='auto',
        help="Solver backend (default: auto - try Gurobi first, fallback to Torch)"
    )
    verify_group.add_argument(
        "--bab",
        action="store_true",
        help="Use branch-and-bound refinement (instead of single-shot)"
    )
    verify_group.add_argument(
        "--bab-max-depth",
        type=int,
        default=5,
        dest="bab_max_depth",
        help="Maximum BaB tree depth (default: 5)"
    )
    verify_group.add_argument(
        "--bab-max-subproblems",
        type=int,
        default=100,
        dest="bab_max_subproblems",
        help="Maximum number of BaB subproblems (default: 100)"
    )
    verify_group.add_argument(
        "--timeout", "-t",
        type=float,
        default=300.0,
        help="Solver timeout in seconds (default: 300)"
    )
    
    # Common options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Add standard device/dtype arguments (shared across all ACT CLIs)
    add_device_args(parser)
    
    args, unknown = parser.parse_known_args()
    
    if args.confignet:
        from act.back_end import confignet
        return confignet.main(unknown)

    if unknown:
        parser.error(f"Unrecognized arguments: {' '.join(unknown)}")

    # Initialize device manager from CLI arguments
    initialize_from_args(args)
    
    # Validate arguments based on command
    if args.verify or args.info:
        if not args.network:
            parser.error("--network is required for --verify and --info")
    
    # Execute command
    try:
        if args.generate:
            return run_network_factory(args)
        elif args.verify:
            return run_verification(args)
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


if __name__ == "__main__":
    sys.exit(main())
