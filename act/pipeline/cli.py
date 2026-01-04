#!/usr/bin/env python3
"""
ACT Pipeline Command-Line Interface.

Provides fuzzing capabilities for neural network verification with support for:
- VNNLib verification benchmarks (default)
- TorchVision datasets (alternative)

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

import argparse
from pathlib import Path
from typing import List, Optional
import sys

from act.util.cli_utils import add_device_args, initialize_from_args
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
from act.front_end.vnnlib_loader import data_model_loader as vnnlib_loader
from act.front_end.vnnlib_loader import category_mapping as vnnlib_mapping
from act.front_end.torchvision_loader.create_specs import TorchVisionSpecCreator
from act.front_end.torchvision_loader import data_model_loader as tv_loader
from act.front_end.torchvision_loader import data_model_mapping as tv_mapping
from act.front_end.model_synthesis import synthesize_models_from_specs
from act.pipeline.fuzzing.actfuzzer import ACTFuzzer, FuzzingConfig, FuzzingReport


def print_header():
    """Print simple header."""
    print(f"\n{'='*80}")
    print(f"ACT: Abstract Constraint Transformer")
    print(f"Inference-based whitebox fuzzing for neural network verification")
    print(f"{'='*80}\n")


# ============================================================================
# Data-Model Pair Management Commands
# ============================================================================

def cmd_list_available(creator: str):
    """List available datasets/categories."""
    print(f"\n{'='*80}")
    print(f"AVAILABLE DATA-MODEL PAIRS ({creator.upper()})")
    print(f"{'='*80}\n")
    
    if creator == 'vnnlib':
        categories = vnnlib_mapping.list_categories()
        print(f"VNNLIB Categories ({len(categories)}):")
        print('-' * 80)
        for cat_name in sorted(categories):
            info = vnnlib_mapping.get_category_info(cat_name)
            print(f"  {cat_name:30s} ({info['type']}) - {info['description']}")
            print(f"    └─ Models: {info['models']}, Properties: {info['properties']}")
    
    elif creator == 'torchvision':
        datasets = sorted(tv_mapping.DATASET_MODEL_MAPPING.keys())
        print(f"TorchVision Datasets ({len(datasets)}):")
        print('-' * 80)
        for ds_name in datasets:
            info = tv_mapping.DATASET_MODEL_MAPPING[ds_name]
            models = info.get('models', [])
            print(f"  {ds_name:30s} [{info.get('category', 'N/A')}]")
            if models:
                print(f"    └─ Models: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
    
    print(f"\n{'='*80}\n")


def cmd_search(query: str, creator: str):
    """Search for datasets/categories."""
    print(f"\n{'='*80}")
    print(f"SEARCH RESULTS: '{query}' ({creator.upper()})")
    print(f"{'='*80}\n")
    
    if creator == 'vnnlib':
        matches = vnnlib_mapping.search_categories(query)
        if matches:
            print(f"Found {len(matches)} VNNLIB categories:")
            print('-' * 80)
            for cat_name in sorted(matches):
                info = vnnlib_mapping.get_category_info(cat_name)
                print(f"  {cat_name:30s} ({info['type']}) - {info['description']}")
        else:
            print(f"No VNNLIB categories found for '{query}'")
    
    elif creator == 'torchvision':
        matches = tv_mapping.search_datasets(query)
        if matches:
            print(f"Found {len(matches)} TorchVision datasets:")
            print('-' * 80)
            for ds_name in sorted(matches):
                info = tv_mapping.DATASET_MODEL_MAPPING[ds_name]
                print(f"  {ds_name:30s} [{info.get('category', 'N/A')}]")
        else:
            print(f"No TorchVision datasets found for '{query}'")
    
    print(f"\n{'='*80}\n")


def cmd_info(name: str, creator: str):
    """Show detailed information about dataset/category."""
    print(f"\n{'='*80}")
    print(f"INFO: {name} ({creator.upper()})")
    print(f"{'='*80}\n")
    
    if creator == 'vnnlib':
        try:
            info = vnnlib_mapping.get_category_info(name)
            print(f"Category: {name}")
            print(f"Type: {info['type']}")
            print(f"Year: {info['year']}")
            print(f"Description: {info['description']}")
            print(f"\nModel Information:")
            print(f"  • Models: {info['models']}")
            print(f"  • Properties: {info['properties']}")
            print(f"  • Input Dim: {info['input_dim']}")
            print(f"  • Output Dim: {info['output_dim']}")
            
            # Check if downloaded
            downloaded = vnnlib_loader.list_downloaded_pairs()
            matching = [p for p in downloaded if p['category'] == name]
            if matching:
                print(f"\n✓ Downloaded: {len(matching)} instances")
            else:
                print(f"\n⚠ Not downloaded (use --download {name})")
        except ValueError as e:
            print(f"Error: {e}")
    
    elif creator == 'torchvision':
        try:
            info = tv_mapping.get_dataset_info(name)
            print(f"Dataset: {name}")
            print(f"Category: {info.get('category', 'N/A')}")
            print(f"Input Size: {info.get('input_size', 'N/A')}")
            print(f"Classes: {info.get('num_classes', 'N/A')}")
            
            models = info.get('models', [])
            if models:
                print(f"\nRecommended Models ({len(models)}):")
                for model in models:
                    print(f"  • {model}")
            
            # Check if downloaded
            downloaded = tv_loader.list_downloaded_pairs()
            matching = [p for p in downloaded if p['dataset'] == name]
            if matching:
                print(f"\n✓ Downloaded: {len(matching)} model pairs")
            else:
                print(f"\n⚠ Not downloaded (use --download {name} --creator torchvision)")
        except ValueError as e:
            print(f"Error: {e}")
    
    print(f"\n{'='*80}\n")


def cmd_download(name: str, creator: str):
    """Download dataset/category."""
    print(f"\n{'='*80}")
    print(f"DOWNLOADING: {name} ({creator.upper()})")
    print(f"{'='*80}\n")
    
    if creator == 'vnnlib':
        try:
            result = vnnlib_loader.download_vnnlib_category(name)
            
            if result['status'] == 'success':
                print(f"✓ Successfully downloaded: {name}")
                print(f"  Location: {result['category_path']}")
                print(f"  Instances: {result['num_instances']}")
            else:
                print(f"✗ Download failed: {result['message']}")
                print(f"\nNote: VNNLIB benchmarks must be downloaded manually from VNN-COMP.")
                print(f"Expected location: data/vnnlib/{name}/")
                print(f"\nManual steps:")
                print(f"  1. Visit: https://github.com/ChristopherBrix/vnncomp_benchmarks")
                print(f"  2. Download '{name}' benchmark")
                print(f"  3. Extract to: data/vnnlib/{name}/")
                print(f"  4. Ensure structure:")
                print(f"     - onnx/         (ONNX model files)")
                print(f"     - vnnlib/       (VNNLIB property files)")
                print(f"     - instances.csv (benchmark instances)")
        except Exception as e:
            print(f"✗ Download error: {e}")
    
    elif creator == 'torchvision':
        try:
            info = tv_mapping.get_dataset_info(name)
            models = info.get('models', [])
            
            if not models:
                print(f"⚠ No models available for {name}")
                return
            
            print(f"Downloading {name} with {len(models)} models...\n")
            
            success_count = 0
            for model in models:
                result = tv_loader.download_dataset_model_pair(name, model)
                if result['status'] == 'success':
                    print(f"✓ {name} + {model}")
                    success_count += 1
                else:
                    print(f"✗ {name} + {model} - {result['message']}")
            
            print(f"\n{'='*80}")
            print(f"Downloaded {success_count}/{len(models)} model pairs")
            print(f"{'='*80}")
        except Exception as e:
            print(f"✗ Download error: {e}")
    
    print()


def cmd_list_downloaded(creator: str):
    """List downloaded data-model pairs."""
    print(f"\n{'='*80}")
    print(f"DOWNLOADED DATA-MODEL PAIRS ({creator.upper()})")
    print(f"{'='*80}\n")
    
    if creator == 'vnnlib':
        downloaded = vnnlib_loader.list_downloaded_pairs()
        if downloaded:
            # Group by category
            categories = {}
            for item in downloaded:
                cat = item['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(item)
            
            print(f"VNNLIB Downloads ({len(downloaded)} instances):")
            print('-' * 80)
            for cat in sorted(categories.keys()):
                instances = categories[cat]
                print(f"  {cat:30s} ({len(instances)} instances)")
                if len(instances) <= 5:
                    for inst in instances:
                        print(f"    └─ {inst['instance_id']}: {inst['onnx_model']} + {inst['vnnlib_spec']}")
        else:
            print("No VNNLIB downloads found")
            print("Use --download <category> to download benchmarks")
    
    elif creator == 'torchvision':
        downloaded = tv_loader.list_downloaded_pairs()
        if downloaded:
            # Group by dataset
            datasets = {}
            for item in downloaded:
                ds = item['dataset']
                if ds not in datasets:
                    datasets[ds] = []
                datasets[ds].append(item['model'])
            
            print(f"TorchVision Downloads ({len(downloaded)} pairs):")
            print('-' * 80)
            for ds in sorted(datasets.keys()):
                models = datasets[ds]
                print(f"  {ds:30s} ({len(models)} models)")
                for model in sorted(models):
                    print(f"    └─ {model}")
        else:
            print("No TorchVision downloads found")
            print("Use --download <dataset> --creator torchvision to download data-model pairs")
    
    print(f"\n{'='*80}\n")


# ============================================================================
# Fuzzing Commands
# ============================================================================

def cmd_fuzz(args):
    """Run ACTFuzzer."""
    print_header()
    
    # Determine creator
    creator = args.creator
    print(f"📦 Using spec creator: {creator.upper()}")
    if args.strict_mode:
        print(f"⚠️  Strict mode enabled: Errors will be raised on constraint violations")
    print()
    
    # Load configuration
    config = FuzzingConfig(
        max_iterations=args.iterations,
        timeout_seconds=args.timeout,
        device=args.device,
        save_counterexamples=not args.no_save,
        output_dir=Path(args.output),
        report_interval=args.report_interval,
        mutation_weights={
            "gradient": 0.4,
            "pgd": 0.0,
            "activation": 0.3,
            "boundary": 0.2,
            "random": 0.1
        },
        # Tracing configuration
        trace_level=args.trace_level,
        trace_sample_rate=args.trace_sample,
        trace_storage=args.trace_storage,
        trace_output=Path(args.trace_output) if args.trace_output else None
    )
    
    # Create spec creator and load data-model pairs
    print(f"{'='*80}")
    print(f"STEP 1: Loading Data-Model Pairs")
    print(f"{'='*80}\n")
    
    spec_results = []
    initial_seeds = []
    
    try:
        if creator == 'vnnlib':
            spec_creator = VNNLibSpecCreator()
            
            if args.category:
                # Specific category
                categories = [args.category]
            else:
                # Use all downloaded categories
                downloaded = vnnlib_loader.list_downloaded_pairs()
                if not downloaded:
                    print("❌ No VNNLIB categories downloaded!")
                    print("Use: python -m act.pipeline --download <category>")
                    return
                categories = list(set(p['category'] for p in downloaded))
            
            print(f"Loading {len(categories)} VNNLIB category(ies):")
            for cat in categories:
                print(f"  • {cat}")
            print()
            
            spec_results = spec_creator.create_specs_for_data_model_pairs(
                categories=categories,
                max_instances=args.max_instances
            )
        
        elif creator == 'torchvision':
            spec_creator = TorchVisionSpecCreator()
            
            if args.dataset:
                # Specific dataset
                datasets = [args.dataset]
            else:
                # Use all downloaded datasets
                downloaded = tv_loader.list_downloaded_pairs()
                if not downloaded:
                    print("❌ No TorchVision datasets downloaded!")
                    print("Use: python -m act.pipeline --download <dataset> --creator torchvision")
                    return
                datasets = list(set(p['dataset'] for p in downloaded))
            
            print(f"Loading {len(datasets)} TorchVision dataset(s):")
            for ds in datasets:
                print(f"  • {ds}")
            print()
            
            # Get models for each dataset
            if args.model:
                # Specific model for all datasets
                model_names = [args.model]
            else:
                # Use first available model for each dataset
                downloaded = tv_loader.list_downloaded_pairs()
                model_names = []
                for ds in datasets:
                    ds_models = [p['model'] for p in downloaded if p['dataset'] == ds]
                    if ds_models:
                        model_names.append(ds_models[0])
            
            if not model_names:
                print("❌ No models found for selected datasets!")
                return
            
            spec_results = spec_creator.create_specs_for_data_model_pairs(
                dataset_names=datasets,
                model_names=model_names,
                num_samples=args.num_samples
            )
    
    except Exception as e:
        print(f"❌ Error loading data-model pairs: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not spec_results:
        print("❌ No spec results generated!")
        return
    
    print(f"✓ Generated {len(spec_results)} spec result(s)\n")
    
    # Synthesize models
    print(f"{'='*80}")
    print(f"STEP 2: Model Synthesis")
    print(f"{'='*80}\n")
    
    # Set strict mode for all VerifiableModel instances
    from act.front_end.verifiable_model import VerifiableModel
    VerifiableModel.set_strict_mode(args.strict_mode)
    
    try:
        wrapped_models, reports, input_data = synthesize_models_from_specs(spec_results)
    except Exception as e:
        print(f"❌ Model synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not wrapped_models:
        print("❌ No models synthesized!")
        return
    
    print(f"✓ Synthesized {len(wrapped_models)} wrapped model(s)\n")
    
    # Extract initial seeds
    print(f"{'='*80}")
    print(f"STEP 3: Seed Extraction")
    print(f"{'='*80}\n")
    
    for data_source, model_name, pytorch_model, labeled_tensors, spec_pairs in spec_results:
        initial_seeds.extend(labeled_tensors)
    
    if not initial_seeds:
        print("❌ No initial seeds extracted!")
        return
    
    print(f"✓ Extracted {len(initial_seeds)} initial seeds\n")
    
    # Run fuzzing on first model
    print(f"{'='*80}")
    print(f"STEP 4: Fuzzing")
    print(f"{'='*80}\n")
    
    model_id = list(wrapped_models.keys())[0]
    wrapped_model = wrapped_models[model_id]
    
    print(f"Fuzzing model: {model_id}\n")
    
    try:
        fuzzer = ACTFuzzer(
            wrapped_model=wrapped_model,
            initial_seeds=initial_seeds,
            config=config
        )
        
        report = fuzzer.fuzz()
        
        # Print final results
        print(f"\n{'='*80}")
        print(f"FUZZING COMPLETE")
        print(f"{'='*80}")
        print(f"Iterations: {report.total_iterations}")
        print(f"Time: {report.total_time:.1f}s")
        print(f"Counterexamples: {len(report.counterexamples)}")
        print(f"Coverage: {report.neuron_coverage:.2%}")
        print(f"Seeds explored: {report.seeds_explored}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"❌ Fuzzing failed: {e}")
        import traceback
        traceback.print_exc()
        return


# ============================================================================
# Verification Commands
# ============================================================================

def cmd_list_verifications():
    """List available verification tests."""
    print(f"\n{'='*80}")
    print(f"AVAILABLE VERIFICATION TESTS")
    print(f"{'='*80}\n")
    
    tests = [
        ("act2torch", "ACT→PyTorch conversion validation (model_factory)"),
        ("torch2act", "PyTorch→ACT conversion validation (torch2act)"),
        ("validate_verifier", "Verifier correctness validation with concrete tests"),
        ("all", "Run all verification tests"),
    ]
    
    for name, description in tests:
        print(f"  {name:25s} - {description}")
    
    print(f"\n{'='*80}\n")


def cmd_verify(target: str, args):
    """Run verification tests from the verification submodule."""
    print_header()
    
    # Import verification test modules
    from act.pipeline.verification import model_factory, torch2act
    
    tests_to_run = []
    if target == 'all':
        tests_to_run = ['act2torch', 'torch2act']
    else:
        tests_to_run = [target]
    
    results = {}
    
    for test_name in tests_to_run:
        print(f"\n{'='*80}")
        if test_name == 'act2torch':
            print(f"VERIFICATION TEST: ACT→PyTorch Conversion")
            print(f"{'='*80}\n")
            try:
                model_factory.main()
                results[test_name] = 'PASSED'
            except Exception as e:
                print(f"\n❌ Test failed: {e}")
                import traceback
                traceback.print_exc()
                results[test_name] = 'FAILED'
        
        elif test_name == 'torch2act':
            print(f"VERIFICATION TEST: PyTorch→ACT Conversion")
            print(f"{'='*80}\n")
            try:
                torch2act.main()
                results[test_name] = 'PASSED'
            except Exception as e:
                print(f"\n❌ Test failed: {e}")
                import traceback
                traceback.print_exc()
                results[test_name] = 'FAILED'
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"VERIFICATION TEST SUMMARY")
    print(f"{'='*80}")
    for test_name, result in results.items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"  {status} {test_name:25s} {result}")
    print(f"{'='*80}\n")
    
    # Exit with error if any test failed
    if any(r == 'FAILED' for r in results.values()):
        sys.exit(1)


def cmd_validate_verifier(args):
    """Run verifier validation with specified mode."""
    import torch
    from act.pipeline.verification.validate_verifier import VerificationValidator
    
    print_header()
    
    # Convert dtype string to torch dtype
    dtype = torch.float64 if args.dtype == 'float64' else torch.float32
    
    # Create validator
    validator = VerificationValidator(device=args.device, dtype=dtype)
    
    # Parse networks if specified
    networks = args.networks.split(',') if args.networks else None
    
    # Run validation based on mode
    try:
        if args.mode == 'counterexample':
            summary = validator.validate_counterexamples(
                networks=networks,
                solvers=args.solvers
            )
            # Exit 1 if failures or errors, unless --ignore-errors is set
            exit_code = 0 if args.ignore_errors else (
                1 if (summary['failed'] > 0 or summary.get('errors', 0) > 0) else 0
            )
        elif args.mode == 'bounds':
            summary = validator.validate_bounds(
                networks=networks,
                tf_modes=args.tf_modes,
                num_samples=args.samples
            )
            # Exit 1 if failures or errors, unless --ignore-errors is set
            exit_code = 0 if args.ignore_errors else (
                1 if (summary['failed'] > 0 or summary.get('errors', 0) > 0) else 0
            )
        else:  # comprehensive
            combined = validator.validate_comprehensive(
                networks=networks,
                solvers=args.solvers,
                tf_modes=args.tf_modes,
                num_samples=args.samples
            )
            # Exit 1 if any failures or errors, unless --ignore-errors is set
            exit_code = 0 if args.ignore_errors else (
                1 if combined['overall_status'] in ('FAILED', 'ERROR') else 0
            )
        
        sys.exit(exit_code)
    
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m act.pipeline",
        description="ACT Pipeline: Inference-based whitebox fuzzing for neural networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available VNNLIB categories
  python -m act.pipeline --list
  
  # Search for benchmarks
  python -m act.pipeline --search acas
  
  # Get detailed information
  python -m act.pipeline --info acasxu_2023
  
  # Download data-model pairs
  python -m act.pipeline --download acasxu_2023
  
  # List downloaded pairs
  python -m act.pipeline --list-downloaded
  
  # Fuzz VNNLIB benchmark
  python -m act.pipeline --fuzz --category acasxu_2023 --iterations 5000
  
  # Fuzz TorchVision dataset
  python -m act.pipeline --fuzz --creator torchvision --dataset MNIST
  
  # Run verification tests
  python -m act.pipeline --verify act2torch --device cpu
  python -m act.pipeline --verify torch2act --device cpu
  python -m act.pipeline --verify all --device cpu
  
  # Run verifier validation (comprehensive by default)
  python -m act.pipeline --validate-verifier --device cpu --dtype float64
  python -m act.pipeline --validate-verifier --mode counterexample
  python -m act.pipeline --validate-verifier --mode bounds --samples 20
        """
    )
    
    # Command selection (mutually exclusive)
    cmd_group = parser.add_mutually_exclusive_group(required=True)
    cmd_group.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets/categories"
    )
    cmd_group.add_argument(
        "--search", "-s",
        type=str,
        metavar="QUERY",
        help="Search for datasets/categories"
    )
    cmd_group.add_argument(
        "--info", "-i",
        type=str,
        metavar="NAME",
        help="Show detailed information"
    )
    cmd_group.add_argument(
        "--download", "-d",
        type=str,
        metavar="NAME",
        help="Download dataset/category"
    )
    cmd_group.add_argument(
        "--list-downloaded",
        action="store_true",
        help="List downloaded data-model pairs"
    )
    cmd_group.add_argument(
        "--fuzz", "-f",
        action="store_true",
        help="Run ACTFuzzer"
    )
    cmd_group.add_argument(
        "--verify",
        type=str,
        metavar="TARGET",
        choices=['act2torch', 'torch2act', 'all'],
        help="Run verification tests: act2torch, torch2act, or all"
    )
    cmd_group.add_argument(
        "--validate-verifier",
        action="store_true",
        help="Run verifier validation (counterexample and bounds checking)"
    )
    cmd_group.add_argument(
        "--list-verifications",
        action="store_true",
        help="List available verification tests"
    )
    
    # Creator selection
    parser.add_argument(
        "--creator", "-c",
        type=str,
        choices=['vnnlib', 'torchvision'],
        default='vnnlib',
        help="Spec creator (default: vnnlib)"
    )
    
    # VNNLIB-specific options
    vnnlib_group = parser.add_argument_group('VNNLIB Options')
    vnnlib_group.add_argument(
        "--category",
        type=str,
        help="VNNLIB category to fuzz (e.g., acasxu_2023)"
    )
    vnnlib_group.add_argument(
        "--max-instances",
        type=int,
        default=10,
        help="Max VNNLIB instances to load (default: 10)"
    )
    
    # TorchVision-specific options
    tv_group = parser.add_argument_group('TorchVision Options')
    tv_group.add_argument(
        "--dataset",
        type=str,
        help="TorchVision dataset to fuzz (e.g., MNIST)"
    )
    tv_group.add_argument(
        "--model",
        type=str,
        help="TorchVision model to fuzz (e.g., simple_cnn)"
    )
    tv_group.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to load (default: 10)"
    )
    
    # Fuzzing configuration
    fuzz_group = parser.add_argument_group('Fuzzing Options')
    fuzz_group.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Max fuzzing iterations (default: 10000)"
    )
    fuzz_group.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Timeout in seconds (default: 3600)"
    )
    fuzz_group.add_argument(
        "--output",
        type=str,
        default="fuzzing_results",
        help="Output directory (default: fuzzing_results)"
    )
    fuzz_group.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save counterexamples to disk"
    )
    fuzz_group.add_argument(
        "--report-interval",
        type=int,
        default=100,
        help="Report progress every N iterations (default: 100)"
    )
    fuzz_group.add_argument(
        "--strict-mode",
        action="store_true",
        help="Enable strict mode: raise errors on input/output constraint violations (default: False)"
    )
    
    # Tracing options
    trace_group = parser.add_argument_group('Execution Tracing Options')
    trace_group.add_argument(
        "--trace-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Tracing detail level: 0=disabled (default), 1=basic (iteration metrics + inputs), "
             "2=full (+ layer activations), 3=debug (+ gradients and loss)"
    )
    trace_group.add_argument(
        "--trace-sample",
        type=int,
        default=1,
        metavar="N",
        help="Capture every Nth iteration (default: 1 = all iterations). "
             "Use higher values to reduce overhead (e.g., 10 = every 10th iteration)"
    )
    trace_group.add_argument(
        "--trace-storage",
        type=str,
        choices=['hdf5', 'json'],
        default='json',
        help="Storage backend: json=text/readable (default), hdf5=binary/compressed"
    )
    trace_group.add_argument(
        "--trace-output",
        type=str,
        help="Custom trace output path (default: <output-dir>/traces.{hdf5|json})"
    )
    
    # Validation options
    validation_group = parser.add_argument_group('Validation Options')
    validation_group.add_argument(
        "--mode",
        type=str,
        choices=['counterexample', 'bounds', 'comprehensive'],
        default='comprehensive',
        help="Validation mode (default: comprehensive)"
    )
    validation_group.add_argument(
        "--networks",
        type=str,
        help="Comma-separated list of networks to validate (default: all)"
    )
    validation_group.add_argument(
        "--solvers",
        nargs='+',
        default=['gurobi', 'torchlp'],
        help="Solvers for Level 1 validation (default: gurobi torchlp)"
    )
    validation_group.add_argument(
        "--tf-modes",
        nargs='+',
        default=['interval'],
        help="Transfer function modes for Level 3 (default: interval)"
    )
    validation_group.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples for Level 3 validation (default: 10)"
    )
    validation_group.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Always exit 0 (ignore failures and errors for CI)"
    )
    
    # Add standard device/dtype arguments (shared across all ACT CLIs)
    add_device_args(parser)
    
    args = parser.parse_args()
    
    # Initialize device manager from CLI arguments
    initialize_from_args(args)
    
    # Handle --dataset as alias for --category (for VNNLIB)
    # This provides a more intuitive interface: python -m act.pipeline --fuzz --dataset cifar100_2024
    if args.creator == 'vnnlib' and args.dataset and not args.category:
        args.category = args.dataset
    
    # Execute command
    try:
        if args.list:
            cmd_list_available(args.creator)
        elif args.search:
            cmd_search(args.search, args.creator)
        elif args.info:
            cmd_info(args.info, args.creator)
        elif args.download:
            cmd_download(args.download, args.creator)
        elif args.list_downloaded:
            cmd_list_downloaded(args.creator)
        elif args.fuzz:
            cmd_fuzz(args)
        elif args.verify:
            cmd_verify(args.verify, args)
        elif args.validate_verifier:
            cmd_validate_verifier(args)
        elif args.list_verifications:
            cmd_list_verifications()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
