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
from contextlib import contextmanager
from copy import deepcopy
import logging
from pathlib import Path
from typing import Any, cast
import sys
import torch

from act.util.cli_utils import add_device_args, initialize_from_args
from act.back_end.config import VALID_SOLVER_TIERS

logger = logging.getLogger(__name__)
from act.front_end.specs import OutputSpec
from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
from act.front_end.vnnlib_loader import data_model_loader as vnnlib_loader
from act.front_end.vnnlib_loader import category_mapping as vnnlib_mapping
from act.front_end.torchvision_loader.create_specs import TorchVisionSpecCreator
from act.front_end.torchvision_loader import data_model_loader as tv_loader
from act.front_end.torchvision_loader import data_model_mapping as tv_mapping
from act.front_end.model_synthesis import synthesize_models_from_specs
from act.pipeline.fuzzing.actfuzzer import ACTFuzzer, FuzzingConfig
from act.pipeline.verification.per_neuron_bounds import PerNeuronCheckConfig


_FUZZ_MUTATION_WEIGHT_KEYS = frozenset(
    {"gradient", "pgd", "activation", "boundary", "random"}
)


_FUZZ_OVERRIDE_SPEC: list[tuple[str, str, str, Any]] = [
    ("max_iterations", "--iterations", "iterations", int),
    ("timeout_seconds", "--timeout", "timeout", float),
    ("coverage_strategy", "--coverage-strategy", "coverage_strategy", str),
    ("activation_threshold", "--activation-threshold", "activation_threshold", float),
    ("mutation_weights", "--mutation-weights", "mutation_weights", dict),
    ("perturb_mode", "--perturb-mode", "perturb_mode", str),
    ("perturb_scale", "--perturb-scale", "perturb_scale", float),
    ("seed_selection_strategy", "--seed-selection", "seed_selection_strategy", str),
    ("save_counterexamples", "--no-save", "save_counterexamples", bool),
    ("output_dir", "--output", "output", Path),
    ("report_interval", "--report-interval", "report_interval", int),
    ("verbose", "--fuzz-verbose", "fuzz_verbose", int),
    ("trace_level", "--trace-level", "trace_level", int),
    ("trace_sample_rate", "--trace-sample", "trace_sample", int),
    ("trace_storage", "--trace-storage", "trace_storage", str),
    ("trace_output", "--trace-output", "trace_output", Path),
    ("stop_on_first_violation", "--stop-on-first-violation", "stop_on_first_violation", bool),
]


def _parse_mutation_weights(raw: str) -> dict[str, float]:
    """Parse a complete mutation-weight override from key=value CSV text."""
    weights: dict[str, float] = {}
    try:
        for item in raw.split(","):
            key, value = item.split("=", 1)
            key = key.strip()
            weights[key] = float(value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated key=value pairs, e.g. "
            "gradient=0,pgd=0.4,activation=0.3,boundary=0.2,random=0.1"
        ) from exc

    missing = _FUZZ_MUTATION_WEIGHT_KEYS - weights.keys()
    extra = weights.keys() - _FUZZ_MUTATION_WEIGHT_KEYS
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unknown: {', '.join(sorted(extra))}")
        raise argparse.ArgumentTypeError(
            "mutation weights must provide exactly "
            f"{', '.join(sorted(_FUZZ_MUTATION_WEIGHT_KEYS))} ({'; '.join(details)})"
        )

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise argparse.ArgumentTypeError(
            f"mutation weights must sum to 1.0, got {total:g}"
        )
    return weights


def _add_fuzz_config_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Fuzzing Config Overrides")
    group.add_argument(
        "--coverage-strategy",
        type=str,
        default=None,
        help="Coverage strategy (default: from config.yaml)",
    )
    group.add_argument(
        "--activation-threshold",
        type=float,
        default=None,
        help="Neuron activation threshold for coverage (default: from config.yaml)",
    )
    group.add_argument(
        "--mutation-weights",
        type=str,
        default=None,
        metavar="K=V,...",
        help="Full mutation weights override, e.g. gradient=0,pgd=0.4,activation=0.3,boundary=0.2,random=0.1",
    )
    group.add_argument(
        "--perturb-mode",
        type=str,
        default=None,
        help="Perturbation sizing mode (default: from config.yaml)",
    )
    group.add_argument(
        "--perturb-scale",
        type=float,
        default=None,
        help="Fraction of feasible range per mutation (default: from config.yaml)",
    )
    group.add_argument(
        "--seed-selection",
        type=str,
        default=None,
        help="Seed selection strategy (default: from config.yaml)",
    )
    group.add_argument(
        "--fuzz-verbose",
        type=int,
        default=None,
        help="Fuzzing verbosity: 0=silent, 1=progress violations, 2=each violation (default: from config.yaml)",
    )
    group.add_argument(
        "--stop-on-first-violation",
        action="store_true",
        default=None,
        help="Stop after the first counterexample (default: from config.yaml/FuzzingConfig default)",
    )


def _collect_fuzzing_overrides(args: Any) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, _flag, attr, cast_fn in _FUZZ_OVERRIDE_SPEC:
        value = getattr(args, attr, None)
        if value is None:
            continue
        if key == "save_counterexamples":
            value = False
        elif key == "mutation_weights":
            value = _parse_mutation_weights(value)
        elif cast_fn is Path and not isinstance(value, Path):
            value = Path(value)
        elif cast_fn is int and not isinstance(value, int):
            value = int(str(value))
        elif cast_fn is float and not isinstance(value, float):
            value = float(str(value))
        overrides[key] = value
    return overrides



def print_header():
    """Print simple header."""
    print(f"\n{'=' * 80}")
    print(f"ACT: Abstract Constraint Transformer")
    print(f"Inference-based whitebox fuzzing for neural network verification")
    print(f"{'=' * 80}\n")


# ============================================================================
# Data-Model Pair Management Commands
# ============================================================================


def cmd_list_available(creator: str):
    """List available datasets/categories."""
    print(f"\n{'=' * 80}")
    print(f"AVAILABLE DATA-MODEL PAIRS ({creator.upper()})")
    print(f"{'=' * 80}\n")

    if creator == "vnnlib":
        categories = vnnlib_mapping.list_categories()
        print(f"VNNLIB Categories ({len(categories)}):")
        print("-" * 80)
        for cat_name in sorted(categories):
            info = vnnlib_mapping.get_category_info(cat_name)
            print(f"  {cat_name:30s} ({info['type']}) - {info['description']}")
            print(f"    └─ Models: {info['models']}, Properties: {info['properties']}")

    elif creator == "torchvision":
        datasets = sorted(tv_mapping.DATASET_MODEL_MAPPING.keys())
        print(f"TorchVision Datasets ({len(datasets)}):")
        print("-" * 80)
        for ds_name in datasets:
            info = tv_mapping.DATASET_MODEL_MAPPING[ds_name]
            models = info.get("models", [])
            print(f"  {ds_name:30s} [{info.get('category', 'N/A')}]")
            if models:
                print(
                    f"    └─ Models: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}"
                )

    print(f"\n{'=' * 80}\n")


def cmd_search(query: str, creator: str):
    """Search for datasets/categories."""
    print(f"\n{'=' * 80}")
    print(f"SEARCH RESULTS: '{query}' ({creator.upper()})")
    print(f"{'=' * 80}\n")

    if creator == "vnnlib":
        matches = vnnlib_mapping.search_categories(query)
        if matches:
            print(f"Found {len(matches)} VNNLIB categories:")
            print("-" * 80)
            for cat_name in sorted(matches):
                info = vnnlib_mapping.get_category_info(cat_name)
                print(f"  {cat_name:30s} ({info['type']}) - {info['description']}")
        else:
            print(f"No VNNLIB categories found for '{query}'")

    elif creator == "torchvision":
        matches = tv_mapping.search_datasets(query)
        if matches:
            print(f"Found {len(matches)} TorchVision datasets:")
            print("-" * 80)
            for ds_name in sorted(matches):
                info = tv_mapping.DATASET_MODEL_MAPPING[ds_name]
                print(f"  {ds_name:30s} [{info.get('category', 'N/A')}]")
        else:
            print(f"No TorchVision datasets found for '{query}'")

    print(f"\n{'=' * 80}\n")


def cmd_info(name: str, creator: str):
    """Show detailed information about dataset/category."""
    print(f"\n{'=' * 80}")
    print(f"INFO: {name} ({creator.upper()})")
    print(f"{'=' * 80}\n")

    if creator == "vnnlib":
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
            matching = [p for p in downloaded if p["category"] == name]
            if matching:
                print(f"\n✓ Downloaded: {len(matching)} instances")
            else:
                print(f"\n⚠ Not downloaded (use --download {name})")
        except ValueError as e:
            print(f"Error: {e}")

    elif creator == "torchvision":
        try:
            info = tv_mapping.get_dataset_info(name)
            print(f"Dataset: {name}")
            print(f"Category: {info.get('category', 'N/A')}")
            print(f"Input Size: {info.get('input_size', 'N/A')}")
            print(f"Classes: {info.get('num_classes', 'N/A')}")

            models = info.get("models", [])
            if models:
                print(f"\nRecommended Models ({len(models)}):")
                for model in models:
                    print(f"  • {model}")

            # Check if downloaded
            downloaded = tv_loader.list_downloaded_pairs()
            matching = [p for p in downloaded if p["dataset"] == name]
            if matching:
                print(f"\n✓ Downloaded: {len(matching)} model pairs")
            else:
                print(
                    f"\n⚠ Not downloaded (use --download {name} --creator torchvision)"
                )
        except ValueError as e:
            print(f"Error: {e}")

    print(f"\n{'=' * 80}\n")


def cmd_download(name: str, creator: str):
    """Download dataset/category."""
    print(f"\n{'=' * 80}")
    print(f"DOWNLOADING: {name} ({creator.upper()})")
    print(f"{'=' * 80}\n")

    if creator == "vnnlib":
        try:
            result = vnnlib_loader.download_vnnlib_category(name)

            if result["status"] == "success":
                print(f"✓ Successfully downloaded: {name}")
                print(f"  Location: {result['category_path']}")
                print(f"  Instances: {result['num_instances']}")
            else:
                print(f"✗ Download failed: {result['message']}")
                print(
                    f"\nNote: VNNLIB benchmarks must be downloaded manually from VNN-COMP."
                )
                print(f"Expected location: data/vnnlib/{name}/")
                print(f"\nManual steps:")
                print(
                    f"  1. Visit: https://github.com/ChristopherBrix/vnncomp_benchmarks"
                )
                print(f"  2. Download '{name}' benchmark")
                print(f"  3. Extract to: data/vnnlib/{name}/")
                print(f"  4. Ensure structure:")
                print(f"     - onnx/         (ONNX model files)")
                print(f"     - vnnlib/       (VNNLIB property files)")
                print(f"     - instances.csv (benchmark instances)")
        except Exception as e:
            print(f"✗ Download error: {e}")

    elif creator == "torchvision":
        try:
            info = tv_mapping.get_dataset_info(name)
            models = info.get("models", [])

            if not models:
                print(f"⚠ No models available for {name}")
                return

            print(f"Downloading {name} with {len(models)} models...\n")

            success_count = 0
            for model in models:
                result = tv_loader.download_dataset_model_pair(name, model)
                if result["status"] == "success":
                    print(f"✓ {name} + {model}")
                    success_count += 1
                else:
                    print(f"✗ {name} + {model} - {result['message']}")

            print(f"\n{'=' * 80}")
            print(f"Downloaded {success_count}/{len(models)} model pairs")
            print(f"{'=' * 80}")
        except Exception as e:
            print(f"✗ Download error: {e}")

    print()


def cmd_list_downloaded(creator: str):
    """List downloaded data-model pairs."""
    print(f"\n{'=' * 80}")
    print(f"DOWNLOADED DATA-MODEL PAIRS ({creator.upper()})")
    print(f"{'=' * 80}\n")

    if creator == "vnnlib":
        downloaded = vnnlib_loader.list_downloaded_pairs()
        if downloaded:
            # Group by category
            categories = {}
            for item in downloaded:
                cat = item["category"]
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(item)

            print(f"VNNLIB Downloads ({len(downloaded)} instances):")
            print("-" * 80)
            for cat in sorted(categories.keys()):
                instances = categories[cat]
                print(f"  {cat:30s} ({len(instances)} instances)")
                if len(instances) <= 5:
                    for inst in instances:
                        print(
                            f"    └─ {inst['instance_id']}: {inst['onnx_model']} + {inst['vnnlib_spec']}"
                        )
        else:
            print("No VNNLIB downloads found")
            print("Use --download <category> to download benchmarks")

    elif creator == "torchvision":
        downloaded = tv_loader.list_downloaded_pairs()
        if downloaded:
            # Group by dataset
            datasets = {}
            for item in downloaded:
                ds = item["dataset"]
                if ds not in datasets:
                    datasets[ds] = []
                datasets[ds].append(item["model"])

            print(f"TorchVision Downloads ({len(downloaded)} pairs):")
            print("-" * 80)
            for ds in sorted(datasets.keys()):
                models = datasets[ds]
                print(f"  {ds:30s} ({len(models)} models)")
                for model in sorted(models):
                    print(f"    └─ {model}")
        else:
            print("No TorchVision downloads found")
            print(
                "Use --download <dataset> --creator torchvision to download data-model pairs"
            )

    print(f"\n{'=' * 80}\n")


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

    # Load configuration from YAML with CLI overrides. Unset CLI options remain
    # None sentinels and therefore do not clobber config.yaml values.
    overrides = _collect_fuzzing_overrides(args)
    config = FuzzingConfig.from_yaml(**overrides)

    # Create spec creator and load data-model pairs
    print(f"{'=' * 80}")
    print(f"STEP 1: Loading Data-Model Pairs")
    print(f"{'=' * 80}\n")

    spec_results = []
    initial_seeds = []

    try:
        if creator == "vnnlib":
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
                categories = list(set(p["category"] for p in downloaded))

            print(f"Loading {len(categories)} VNNLIB category(ies):")
            for cat in categories:
                print(f"  • {cat}")
            print()

            spec_results = spec_creator.create_specs_for_data_model_pairs(
                categories=categories, max_instances=args.max_instances
            )

        elif creator == "torchvision":
            spec_creator = TorchVisionSpecCreator()

            if args.dataset:
                # Specific dataset
                datasets = [args.dataset]
            else:
                # Use all downloaded datasets
                downloaded = tv_loader.list_downloaded_pairs()
                if not downloaded:
                    print("❌ No TorchVision datasets downloaded!")
                    print(
                        "Use: python -m act.pipeline --download <dataset> --creator torchvision"
                    )
                    return
                datasets = list(set(p["dataset"] for p in downloaded))

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
                    ds_models = [p["model"] for p in downloaded if p["dataset"] == ds]
                    if ds_models:
                        model_names.append(ds_models[0])

            if not model_names:
                print("❌ No models found for selected datasets!")
                return

            spec_results = spec_creator.create_specs_for_data_model_pairs(
                dataset_names=datasets,
                model_names=model_names,
                num_samples=args.num_samples,
            )

        elif creator == "bert":
            from act.front_end.bert_loader.create_specs import BertSpecCreator

            spec_creator = BertSpecCreator()
            datasets = [args.dataset] if args.dataset else ["sst"]

            print(f"Loading {len(datasets)} bert dataset(s):")
            for ds in datasets:
                print(f"  • {ds}")
            print()

            spec_results = spec_creator.create_specs_for_data_model_pairs(
                dataset_names=datasets,
                num_samples=args.num_samples,
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
    print(f"{'=' * 80}")
    print(f"STEP 2: Model Synthesis")
    print(f"{'=' * 80}\n")

    # Set strict mode for all VerifiableModel instances
    from act.front_end.verifiable_model import VerifiableModel

    VerifiableModel.set_strict_mode(args.strict_mode)

    try:
        wrapped_models = synthesize_models_from_specs(cast(Any, spec_results))
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
    print(f"{'=' * 80}")
    print(f"STEP 3: Seed Extraction")
    print(f"{'=' * 80}\n")

    # Single model only; mixing seeds across spec_results breaks SeedCorpus(torch.cat).
    _, _, _, labeled_tensors, _ = spec_results[0]
    initial_seeds.extend(labeled_tensors)

    if not initial_seeds:
        print("❌ No initial seeds extracted!")
        return

    print(f"✓ Extracted {len(initial_seeds)} initial seeds\n")

    # Run fuzzing on first model
    print(f"{'=' * 80}")
    print(f"STEP 4: Fuzzing")
    print(f"{'=' * 80}\n")

    model_id = list(wrapped_models.keys())[0]
    wrapped_model = wrapped_models[model_id]

    print(f"Fuzzing model: {model_id}\n")

    try:
        fuzzer = ACTFuzzer(
            wrapped_model=wrapped_model, initial_seeds=initial_seeds, config=config
        )

        report = fuzzer.fuzz()

        # Print final results
        print(f"\n{'=' * 80}")
        print(f"FUZZING COMPLETE")
        print(f"{'=' * 80}")
        print(f"Iterations: {report.total_iterations}")
        print(f"Time: {report.total_time:.1f}s")
        print(f"Counterexamples: {len(report.counterexamples)}")
        print(f"Coverage: {report.neuron_coverage:.2%}")
        print(f"Seeds explored: {report.seeds_explored}")
        print(f"{'=' * 80}\n")

        if report.counterexamples and config.save_counterexamples:
            import os
            import torch as _torch
            from act.front_end.vnnlib_loader.vnnlib_parser import write_vnncomp_result

            os.makedirs(config.output_dir, exist_ok=True)
            ce0 = report.counterexamples[0]
            x = cast(Any, getattr(ce0, "input", ce0))
            with _torch.no_grad():
                y_raw = wrapped_model(x)
            y = cast(Any, y_raw["output"] if isinstance(y_raw, dict) else y_raw)
            fname = "_".join(map(str, model_id)) if isinstance(model_id, tuple) else str(model_id)
            write_vnncomp_result(
                os.path.join(config.output_dir, f"{fname}_result.txt"),
                "sat", x=x, y=y,
                in_decl=("X", "float32", tuple(x.shape)),
                out_decl=("Y", "float32", tuple(y.shape)),
            )
            print(f"✓ counterexample witness written for {fname}")

    except Exception as e:
        print(f"❌ Fuzzing failed: {e}")
        import traceback

        traceback.print_exc()
        return


# ============================================================================
# Verification Commands
# ============================================================================


def _effective_tf_modes(solver: str, requested_modes) -> list[str]:
    modes = list(requested_modes or ["interval"])
    if solver != "hybridz":
        return modes
    if requested_modes is not None and modes != ["hybridz"]:
        logger.warning(
            "--solvers hybridz requires the hybridz transformer; overriding "
            "--tf-modes %s",
            " ".join(modes),
        )
    return ["hybridz"]


def _build_validator(args):
    from act.pipeline.verification.validate_verifier import VerificationValidator

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    return VerificationValidator(device=args.device, dtype=dtype)


def _per_neuron_config(args) -> PerNeuronCheckConfig:
    """Resolve the per-neuron check config; 'auto' tolerance = 100 ulp of --dtype.

    100 ulp is the arithmetic noise floor between the abstract and concrete
    computation paths (pairwise-reduction drift of the largest layers is
    ~log2(n)*eps ≈ 18 ulp; float32 auto ≈ 1.2e-5 reproduces the historically
    validated value from commit 0af6397). Pass '0,0' for strict zero.
    """
    if args.bounds_tolerance.strip().lower() == "auto":
        dtype = torch.float64 if args.dtype == "float64" else torch.float32
        floor = 100.0 * torch.finfo(dtype).eps
        tol_abs = tol_rel = floor
    else:
        parts = [float(x) for x in args.bounds_tolerance.split(",")]
        tol_abs = parts[0]
        tol_rel = parts[1] if len(parts) > 1 else 0.0
    return PerNeuronCheckConfig(
        topk=int(args.per_neuron_topk),
        tol_abs=tol_abs,
        tol_rel=tol_rel,
    )


def _verify_and_validate_cell(
    *,
    tag: str,
    model,
    net,
    args,
    validator,
    solver: str,
    tf_mode: str,
    per_neuron_config,
    batch_size=None,
    cell_label=None,
) -> None:
    """Shared verify_once + optional soundness-validation tail for the drivers.

    ``cell_label`` overrides the printed status line (netfactory sweep cells);
    ``tag`` is always the identity passed to ``validator.validate``.
    """
    from act.back_end.verifier import verify_once

    verify_kwargs = (
        {"timelimit": getattr(args, "timeout", None)}
        if solver == "hybridz"
        else {}
    )
    if args.validate_soundness:
        results, facts = verify_once(net, collect_facts=True, **verify_kwargs)
    else:
        results = verify_once(net, **verify_kwargs)
        facts = None
    statuses = [r.status.name for r in results]
    print(f"  {cell_label if cell_label is not None else tag}: {statuses}")
    if args.validate_soundness:
        assert validator is not None
        validator.validate(
            tag,
            model,
            net,
            results,
            solver=solver,
            tf_mode=tf_mode,
            facts=facts,
            num_samples=args.samples,
            per_neuron_config=per_neuron_config,
            batch_size=batch_size,
        )


def _run_vnnlib_verify(args) -> bool:
    """Drive ``verify_once`` over a VNNLIB benchmark end-to-end.

    Bridges the front-end load → ACT-Net path that ``act.back_end --verify
    --network`` does not provide: ``VNNLibSpecCreator`` →
    ``synthesize_models_from_specs`` → ``TorchToACT`` → ``verify_once``.

    Single-mode per invocation, matching the ``act.back_end --verify`` CLI
    contract. Multi-mode sweeps are the caller's job. Dual ignores
    ``--tf-modes``; the standalone HybridZ solver selects HybridzTF.
    """
    from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
    from act.front_end.model_synthesis import synthesize_models_from_specs
    from act.pipeline.verification.torch2act import TorchToACT
    from act.back_end.transfer_functions import (
        set_solver_mode,
        set_transfer_function_mode,
    )

    if not args.category:
        raise ValueError("--verify vnnlib requires --category (e.g. --category acasxu_2023)")

    solver = (args.solvers or ["torchlp"])[0]
    tf_mode = _effective_tf_modes(solver, args.tf_modes)[0]

    set_solver_mode(solver)
    if solver != "dual":
        set_transfer_function_mode(tf_mode)
    label = solver if solver == "dual" else f"{tf_mode}/{solver}"
    print(f"[vnnlib] category={args.category} max_instances={args.max_instances} mode={label}")

    spec_results = VNNLibSpecCreator().create_specs_for_data_model_pairs(
        categories=[args.category], max_instances=args.max_instances,
    )
    if not spec_results:
        raise RuntimeError(f"VNNLibSpecCreator produced no spec_results for category={args.category!r}")

    if getattr(args, "merge_split_relus", False):
        from act.front_end.model_synthesis import merge_split_relus

        merged_results = []
        for sr in spec_results:
            merged_model, n_merged = merge_split_relus(sr[2])
            if n_merged:
                print(f"[merge] fused {n_merged} split-ReLU neurons in {sr[1]}")
                sr = tuple(merged_model if i == 2 else v for i, v in enumerate(sr))
            merged_results.append(sr)
        spec_results = merged_results

    wrapped = synthesize_models_from_specs(spec_results)
    if not wrapped:
        raise RuntimeError("synthesize_models_from_specs produced no VerifiableModels")

    per_neuron_config = _per_neuron_config(args)
    validator = _build_validator(args) if args.validate_soundness else None
    for mid, vm in wrapped.items():
        tag = "/".join(str(p) for p in mid)
        net = TorchToACT(vm).run()
        if getattr(args, "bab", False):
            if args.validate_soundness:
                print("⚠️  --validate-soundness is not yet supported with --bab; skipping validation")
            status = _run_bab_on_net(net, args)
            label = f"BaB[{args.bab_solver_tier}]"
            print(f"  {tag}: {label} → {status}")
        else:
            _verify_and_validate_cell(
                tag=tag,
                model=vm,
                net=net,
                args=args,
                validator=validator,
                solver=solver,
                tf_mode=tf_mode,
                per_neuron_config=per_neuron_config,
            )

    if args.validate_soundness:
        assert validator is not None
        return validator.overall_failed(args.ignore_errors)
    return False


@contextmanager
def _sliced_net_view(net, sample_idx: int, batch_size: int):
    """Yield a per-sample view of ``net`` with spec/assert/input layers sliced.

    On exit, original layer params/out_vars are restored. Safer than inline
    try/finally because mutation surface is encapsulated.
    """
    from act.back_end.verifier import (
        find_entry_layer_id,
        gather_input_spec_layers,
        get_assert_layer,
    )

    assert_layer = get_assert_layer(net)
    spec_layers = gather_input_spec_layers(net)
    input_layer = net.by_id[find_entry_layer_id(net)]
    full_input_ids = list(input_layer.out_vars)
    input_dim = len(full_input_ids) // batch_size
    if len(full_input_ids) != input_dim * batch_size:
        raise RuntimeError(
            f"InputLayer.out_vars ({len(full_input_ids)}) not divisible by B={batch_size}"
        )

    orig_assert_params = deepcopy(assert_layer.params)
    orig_spec_params = [deepcopy(spec_layer.params) for spec_layer in spec_layers]
    orig_input_outvars = list(input_layer.out_vars)
    try:
        for key in OutputSpec.SLICEABLE_PARAM_KEYS:
            val = orig_assert_params.get(key)
            if (
                val is not None
                and hasattr(val, "dim")
                and val.dim() >= 1
                and val.shape[0] == batch_size
            ):
                assert_layer.params[key] = val[sample_idx : sample_idx + 1]

        for spec_layer, sp_orig in zip(spec_layers, orig_spec_params):
            for sp_key, sp_val in sp_orig.items():
                if (
                    hasattr(sp_val, "dim")
                    and sp_val.dim() >= 1
                    and sp_val.shape[0] == batch_size
                ):
                    spec_layer.params[sp_key] = sp_val[sample_idx : sample_idx + 1]

        input_layer.out_vars = full_input_ids[
            sample_idx * input_dim : (sample_idx + 1) * input_dim
        ]
        yield net
    finally:
        assert_layer.params = orig_assert_params
        for spec_layer, sp_orig in zip(spec_layers, orig_spec_params):
            spec_layer.params = sp_orig
        input_layer.out_vars = orig_input_outvars


def _run_bab_on_net(net, args, bab_first_sample_only: bool = False):
    """Verify an ACT Net via verify_bab_batched.

    For single-sample wrappers (B=1) returns one status string.
    For multi-sample wrappers (B>1, e.g. TorchVision), the behavior depends
    on ``bab_first_sample_only``:
      - True  → only sample 0 is verified (one local-robustness instance —
                the BaB-natural unit), returning a single status string.
      - False → all B samples are verified via per-sample iteration,
                returning a list of status strings.
    """
    from act.back_end.bab.bab import verify_bab_batched
    from act.back_end.config import BaBConfig
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.back_end.verifier import (
        gather_input_spec_layers,
        seed_from_input_specs,
    )

    config = BaBConfig(
        solver_tier=args.bab_solver_tier,
        max_depth=args.bab_max_depth,
        max_nodes=args.bab_max_nodes,
        branching_method=getattr(args, "bab_branching_method", "random"),
        bounding_method=getattr(args, "bab_bounding_method", "random"),
        bounding_order=getattr(args, "bab_bounding_order", "depth_lb"),
        sa_cooling_rate=getattr(args, "bab_sa_cooling_rate", 0.99),
        frontier_cap=getattr(args, "bab_frontier_cap", 0),
        input_split_fanout=getattr(args, "bab_input_split_fanout", 2),
        per_class_alpha=(
            str(getattr(args, "bab_per_class_alpha", "true")).lower() == "true"
        ),
        incremental_start_enabled=not getattr(args, "bab_no_incremental_start", False),
        provenance_enabled=getattr(args, "bab_provenance", False),
    )
    budget = float(getattr(args, "timeout", 60.0) or 60.0)

    spec_layers = gather_input_spec_layers(net)
    seed_bounds = seed_from_input_specs(spec_layers)
    B = seed_bounds.lb.shape[0] if seed_bounds.lb.dim() >= 2 else 1

    if B <= 1:
        result = verify_bab_batched(
            net=net,
            solver_factory=TorchLPSolver,
            config=config,
            max_batch_size=None,
            time_budget_s=budget,
        )
        return result.status.name

    sample_range = range(1) if bab_first_sample_only else range(B)

    statuses = []
    for sample_idx in sample_range:
        with _sliced_net_view(net, sample_idx, B) as sliced_net:
            result = verify_bab_batched(
                net=sliced_net,
                solver_factory=TorchLPSolver,
                config=config,
                max_batch_size=None,
                time_budget_s=budget,
            )
            statuses.append(result.status.name)
    return statuses[0] if bab_first_sample_only and statuses else statuses


def _run_torchvision_verify(args) -> bool:
    """Drive ``verify_once`` over a TorchVision dataset-model pair end-to-end.

    Bridges the front-end load → ACT-Net path for TorchVision the same way
    ``_run_vnnlib_verify`` does for VNNLIB benchmarks:
    ``TorchVisionSpecCreator`` → ``synthesize_models_from_specs`` →
    ``TorchToACT`` → ``verify_once``.  Single-mode per invocation, matching
    the ``act.back_end --verify`` CLI contract.

    All three solvers (interval+torchlp, hybridz+torchlp, dual) are
    supported on TorchVision smoke (MNIST + simple_cnn at 224×224). The
    dual track auto-falls back to interval-only at layers whose input
    dim exceeds ``_DENSE_LIN_BOUND_MAX_DIM`` (see ``tf_forward.py``) to
    avoid materializing the dense linear-bound matrix at high dims.
    """
    from act.front_end.torchvision_loader.create_specs import TorchVisionSpecCreator
    from act.front_end.model_synthesis import synthesize_models_from_specs
    from act.pipeline.verification.torch2act import TorchToACT
    from act.back_end.transfer_functions import (
        set_solver_mode,
        set_transfer_function_mode,
    )

    if not args.dataset:
        raise ValueError("--verify torchvision requires --dataset (e.g. --dataset MNIST)")

    solver = (args.solvers or ["torchlp"])[0]
    tf_mode = _effective_tf_modes(solver, args.tf_modes)[0]

    set_solver_mode(solver)
    if solver != "dual":
        set_transfer_function_mode(tf_mode)
    label = solver if solver == "dual" else f"{tf_mode}/{solver}"
    model_label = args.model or "<all>"
    print(
        f"[torchvision] dataset={args.dataset} model={model_label} "
        f"num_samples={args.num_samples} mode={label}"
    )

    spec_results = TorchVisionSpecCreator().create_specs_for_data_model_pairs(
        dataset_names=[args.dataset],
        model_names=[args.model] if args.model else None,
        num_samples=args.num_samples,
    )
    if not spec_results:
        raise RuntimeError(
            f"TorchVisionSpecCreator produced no spec_results for "
            f"dataset={args.dataset!r}, model={args.model!r}"
        )

    wrapped = synthesize_models_from_specs(spec_results)
    if not wrapped:
        raise RuntimeError("synthesize_models_from_specs produced no VerifiableModels")

    if getattr(args, "bab", False):
        if args.validate_soundness:
            print("⚠️  --validate-soundness is not yet supported with --bab; skipping validation")
        local_robust = [
            (mid, vm) for mid, vm in wrapped.items() if "LINF_BALL" in tuple(str(p) for p in mid)
        ]
        if not local_robust:
            local_robust = list(wrapped.items())
        mid, vm = local_robust[0]
        tag = "/".join(str(p) for p in mid)
        net = TorchToACT(vm).run()
        status = _run_bab_on_net(net, args, bab_first_sample_only=True)
        label = f"BaB[{args.bab_solver_tier}]"
        print(f"  {tag} (sample 0 / local-robustness): {label} → {status}")
        return False

    per_neuron_config = _per_neuron_config(args)
    validator = _build_validator(args) if args.validate_soundness else None
    for mid, vm in wrapped.items():
        tag = "/".join(str(p) for p in mid)
        net = TorchToACT(vm).run()
        _verify_and_validate_cell(
            tag=tag,
            model=vm,
            net=net,
            args=args,
            validator=validator,
            solver=solver,
            tf_mode=tf_mode,
            per_neuron_config=per_neuron_config,
        )

    if args.validate_soundness:
        assert validator is not None
        return validator.overall_failed(args.ignore_errors)
    return False


def _run_netfactory_verify(args) -> bool:
    """Run verify_once over ModelFactory networks, optionally with validation."""
    from act.back_end.solver.solver_gurobi import is_gurobi_available
    from act.back_end.transfer_functions import set_solver_mode, set_transfer_function_mode

    validator = _build_validator(args)
    networks = args.networks.split(",") if args.networks else validator.factory.list_networks()
    solvers = list(args.solvers or ["torchlp"])
    if "gurobi" in solvers and not is_gurobi_available():
        logger.warning("Skipping gurobi solver: gurobipy is not available.")
        solvers = [s for s in solvers if s != "gurobi"]
    batch_sizes = _resolve_batch_sizes(getattr(args, "batch_sizes", None))
    per_neuron_config = _per_neuron_config(args)
    errors_seen = False

    for name in networks:
        for solver in solvers:
            for tf_mode in _effective_tf_modes(solver, args.tf_modes):
                for batch_size in batch_sizes:
                    try:
                        set_solver_mode(solver)
                        if solver != "dual":
                            set_transfer_function_mode(tf_mode)
                        act_net = validator.factory.get_act_net(name)
                        act_net = validator._batchify_net(act_net, batch_size)
                        reason = validator.skip_reason(act_net, solver, tf_mode)
                        if reason:
                            validator.record_skip(name, solver, tf_mode, batch_size, reason)
                            continue

                        # Reconstruct the model from the SAME (batchified) net being
                        # verified: create_model(name) returns the single-lane model
                        # whose OutputSpecLayer carries only y_true[0], so the CE
                        # probe's per-sample satisfied flags would test lane 0's
                        # class on every lane -- misattributing a CE to certified
                        # lanes (false [soundness] FAILED).
                        from act.pipeline.verification.act2torch import ACTToTorch
                        model = ACTToTorch(act_net).run()
                        label = solver if solver == "dual" else f"{tf_mode}/{solver}"
                        _verify_and_validate_cell(
                            tag=name,
                            model=model,
                            net=act_net,
                            args=args,
                            validator=validator,
                            solver=solver,
                            tf_mode=tf_mode,
                            per_neuron_config=per_neuron_config,
                            batch_size=batch_size,
                            cell_label=f"{name} B={batch_size} mode={label}",
                        )
                    except Exception as e:
                        errors_seen = True
                        logger.error(
                            "Validation failed for %s/%s/%s/B=%s: %s",
                            name,
                            solver,
                            tf_mode,
                            batch_size,
                            e,
                        )
                        import traceback

                        traceback.print_exc()
                        validator.record_error(
                            name, solver, tf_mode, batch_size, f"Outer exception: {str(e)}"
                        )

    return (
        validator.overall_failed(args.ignore_errors)
        if args.validate_soundness
        else (False if args.ignore_errors else errors_seen)
    )


def cmd_verify(target: str, args):
    """Run verification tests from the verification submodule."""
    print_header()

    from act.pipeline.verification import model_factory, torch2act

    tests_to_run = []
    if target == "all":
        tests_to_run = ["act2torch", "torch2act", "netfactory"]
    else:
        tests_to_run = [target]

    results = {}

    for test_name in tests_to_run:
        print(f"\n{'=' * 80}")
        if test_name == "act2torch":
            print(f"VERIFICATION TEST: ACT→PyTorch Conversion")
            print(f"{'=' * 80}\n")
            try:
                model_factory.main()
                results[test_name] = "PASSED"
            except Exception as e:
                print(f"\n❌ Test failed: {e}")
                import traceback

                traceback.print_exc()
                results[test_name] = "FAILED"

        elif test_name == "torch2act":
            print(f"VERIFICATION TEST: PyTorch→ACT Conversion")
            print(f"{'=' * 80}\n")
            try:
                torch2act.main()
                results[test_name] = "PASSED"
            except Exception as e:
                print(f"\n❌ Test failed: {e}")
                import traceback

                traceback.print_exc()
                results[test_name] = "FAILED"

        elif test_name == "netfactory":
            print(f"VERIFICATION TEST: ModelFactory → verify_once")
            print(f"{'=' * 80}\n")
            try:
                validation_failed = _run_netfactory_verify(args)
                results[test_name] = "FAILED" if validation_failed else "PASSED"
            except Exception as e:
                print(f"\n❌ Test failed: {e}")
                import traceback

                traceback.print_exc()
                results[test_name] = "FAILED"

        elif test_name == "vnnlib":
            print(f"VERIFICATION TEST: VNNLIB → VerifiableModel → verify_once")
            print(f"{'=' * 80}\n")
            try:
                soundness_failed = _run_vnnlib_verify(args)
                results[test_name] = "FAILED" if soundness_failed else "PASSED"
            except Exception as e:
                print(f"\n❌ Test failed: {e}")
                import traceback

                traceback.print_exc()
                results[test_name] = "FAILED"

        elif test_name == "torchvision":
            print(f"VERIFICATION TEST: TorchVision → VerifiableModel → verify_once")
            print(f"{'=' * 80}\n")
            try:
                soundness_failed = _run_torchvision_verify(args)
                results[test_name] = "FAILED" if soundness_failed else "PASSED"
            except Exception as e:
                print(f"\n❌ Test failed: {e}")
                import traceback

                traceback.print_exc()
                results[test_name] = "FAILED"

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"VERIFICATION TEST SUMMARY")
    print(f"{'=' * 80}")
    for test_name, result in results.items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"  {status} {test_name:25s} {result}")
    print(f"{'=' * 80}\n")

    # Exit with error if any test failed
    if any(r == "FAILED" for r in results.values()):
        sys.exit(1)


def _resolve_batch_sizes(cli_value):
    """CLI flag > YAML ``validate.batch_sizes`` > built-in default ``[None]``.

    The ``[None]`` fallback means "validate each network at its native
    batch size from JSON only" (no batchification).
    """
    if cli_value:
        return cli_value
    try:
        import yaml
        from act.util.path_config import get_project_root
        cfg_path = (
            Path(get_project_root())
            / "act/back_end/examples/config_gen_act_net.yaml"
        )
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            yaml_val = (cfg.get("validate") or {}).get("batch_sizes")
            if yaml_val:
                return yaml_val
    except Exception as e:
        # Intentional: optional YAML override; missing/malformed files fall through to default [None].
        logger.debug("suppressed: %s", e)
    return [None]


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
  python -m act.pipeline --verify netfactory --device cpu
  python -m act.pipeline --verify all --device cpu

  # Run verifier on a VNNLIB benchmark end-to-end (load → ACT → verify_once).
  # Single (tf, solver) per invocation; matrix sweeps by repeated calls.
  python -m act.pipeline --verify vnnlib --category acasxu_2023 --max-instances 3 --tf-modes interval --solvers torchlp
  python -m act.pipeline --verify vnnlib --category acasxu_2023 --max-instances 3 --tf-modes hybridz --solvers torchlp
  python -m act.pipeline --verify vnnlib --category acasxu_2023 --max-instances 3                          --solvers hybridz
  python -m act.pipeline --verify vnnlib --category acasxu_2023 --max-instances 3                          --solvers dual

  # Run verifier on a TorchVision dataset-model pair end-to-end.
  python -m act.pipeline --verify torchvision --dataset MNIST --model simple_cnn --num-samples 2 --tf-modes interval --solvers torchlp
  python -m act.pipeline --verify torchvision --dataset MNIST --model simple_cnn --num-samples 2 --tf-modes hybridz  --solvers torchlp
  python -m act.pipeline --verify torchvision --dataset MNIST --model simple_cnn --num-samples 2                     --solvers hybridz
  python -m act.pipeline --verify torchvision --dataset MNIST --model simple_cnn --num-samples 2                     --solvers dual

  # Run unified two-level verifier validation after verification.
  python -m act.pipeline --verify netfactory --solvers torchlp --tf-modes interval --validate-soundness
  python -m act.pipeline --verify vnnlib --category acasxu_2023 --max-instances 3 --validate-soundness
  python -m act.pipeline --verify torchvision --dataset MNIST --model simple_cnn --num-samples 2 --validate-soundness
        """,
    )

    # Command selection (mutually exclusive)
    cmd_group = parser.add_mutually_exclusive_group(required=True)
    cmd_group.add_argument(
        "--list", "-l", action="store_true", help="List available datasets/categories"
    )
    cmd_group.add_argument(
        "--search",
        "-s",
        type=str,
        metavar="QUERY",
        help="Search for datasets/categories",
    )
    cmd_group.add_argument(
        "--info", "-i", type=str, metavar="NAME", help="Show detailed information"
    )
    cmd_group.add_argument(
        "--download", "-d", type=str, metavar="NAME", help="Download dataset/category"
    )
    cmd_group.add_argument(
        "--list-downloaded",
        action="store_true",
        help="List downloaded data-model pairs",
    )
    cmd_group.add_argument("--fuzz", "-f", action="store_true", help="Run ACTFuzzer")
    cmd_group.add_argument(
        "--verify",
        type=str,
        metavar="TARGET",
        choices=["act2torch", "torch2act", "netfactory", "vnnlib", "torchvision", "all"],
        help="Run verification tests: act2torch, torch2act, netfactory, vnnlib, torchvision, "
        "or all. The 'netfactory' target runs generated ACT example nets; "
        "the 'vnnlib' target runs the verifier on a VNNLIB benchmark "
        "end-to-end (requires --category); 'torchvision' does the same for a "
        "TorchVision dataset-model pair (requires --dataset, optionally --model). "
        "Both read the FIRST element of --tf-modes / --solvers (single mode per "
        "invocation; matrix sweeps by repeated calls).",
    )


    # Creator selection
    parser.add_argument(
        "--creator",
        "-c",
        type=str,
        choices=["vnnlib", "torchvision", "bert"],
        default="vnnlib",
        help="Spec creator (default: vnnlib)",
    )

    # VNNLIB-specific options
    vnnlib_group = parser.add_argument_group("VNNLIB Options")
    vnnlib_group.add_argument(
        "--category", type=str, help="VNNLIB category to fuzz (e.g., acasxu_2023)"
    )
    vnnlib_group.add_argument(
        "--max-instances",
        type=int,
        default=10,
        help="Max VNNLIB instances to load (default: 10)",
    )

    # TorchVision-specific options
    tv_group = parser.add_argument_group("TorchVision Options")
    tv_group.add_argument(
        "--dataset", type=str, help="TorchVision dataset to fuzz (e.g., MNIST)"
    )
    tv_group.add_argument(
        "--model", type=str, help="TorchVision model to fuzz (e.g., simple_cnn)"
    )
    tv_group.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to load (default: 10)",
    )

    bab_group = parser.add_argument_group("Branch-and-Bound Options (--verify {vnnlib,torchvision})")
    bab_group.add_argument(
        "--bab",
        action="store_true",
        help="Run BaB (verify_bab_batched) instead of single-shot verify_once",
    )
    bab_group.add_argument(
        "--bab-solver-tier",
        type=str,
        default="dual_alpha_eta",
        choices=list(VALID_SOLVER_TIERS),
        help=(
            "BaB solver tier when --bab is set (default: dual_alpha_eta). "
            "'lp' uses the existing LP/MILP backend; 'dual' uses DualSolver "
            "single-pass; 'dual_alpha' adds Lagrange-relaxed lower-slope "
            "optimization; 'dual_alpha_eta' adds joint slope + split-constraint "
            "KKT multipliers."
        ),
    )
    bab_group.add_argument(
        "--bab-max-depth",
        type=int,
        default=8,
        help="Maximum BaB tree depth (default: 8)",
    )
    bab_group.add_argument(
        "--bab-max-nodes",
        type=int,
        default=100,
        help="Maximum BaB nodes to expand (default: 100)",
    )
    bab_group.add_argument(
        "--bab-branching-method",
        type=str,
        default="random",
        choices=["random", "babsr", "fsb", "gain", "width"],
        help="BaB branching strategy when --bab is set (default: random)",
    )
    bab_group.add_argument(
        "--bab-bounding-method",
        type=str,
        default="random",
        choices=["random", "topk"],
        help=(
            "Pool selection when subproblems exceed the batch size: 'random' or "
            "'topk' (keep the top-k by depth + lower-bound). Default: random."
        ),
    )
    bab_group.add_argument(
        "--bab-bounding-order",
        type=str,
        default="depth_lb",
        choices=["depth_lb", "greedy", "sa"],
        help="TopKBounding order policy (default: depth_lb)",
    )
    bab_group.add_argument(
        "--bab-sa-cooling-rate",
        type=float,
        default=0.99,
        help="Cooling rate for --bab-bounding-order sa (default: 0.99)",
    )
    bab_group.add_argument(
        "--bab-per-class-alpha",
        type=str,
        default="true",
        choices=["true", "false"],
        help=(
            "Per-spec α tensor (True; tighter bounds, M× memory) vs shared α "
            "across specs (False; looser, 1× memory). Default: true."
        ),
    )
    bab_group.add_argument(
        "--bab-no-incremental-start",
        action="store_true",
        help="Disable parent→child α/η incremental-start propagation (debugging / ablation).",
    )
    bab_group.add_argument(
        "--bab-frontier-cap",
        type=int,
        default=0,
        help="Maximum pending BaB frontier leaves to retain; 0 disables eviction (default: 0)",
    )
    bab_group.add_argument(
        "--bab-input-split-fanout",
        type=int,
        default=2,
        help="Uniform fanout for input splits; 2 preserves binary splitting (default: 2)",
    )
    bab_group.add_argument(
        "--bab-provenance",
        action="store_true",
        help="Enable node_id/parent_id provenance sidecar (requires --bab-bounding-method topk).",
    )

    # Fuzzing configuration
    fuzz_group = parser.add_argument_group("Fuzzing Options")
    fuzz_group.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Max fuzzing iterations (default: from config.yaml)",
    )
    fuzz_group.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "Time budget in seconds for fuzzing, BaB, or standalone HybridZ "
            "verification (default: component configuration)"
        ),
    )
    fuzz_group.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: from config.yaml)",
    )
    fuzz_group.add_argument(
        "--no-save",
        action="store_false",
        default=None,
        dest="save_counterexamples",
        help="Don't save counterexamples to disk",
    )
    fuzz_group.add_argument(
        "--report-interval",
        type=int,
        default=None,
        help="Report progress every N iterations (default: from config.yaml)",
    )
    fuzz_group.add_argument(
        "--strict-mode",
        action="store_true",
        help="Enable strict mode: raise errors on input/output constraint violations (default: False)",
    )

    # Tracing options
    trace_group = parser.add_argument_group("Execution Tracing Options")
    trace_group.add_argument(
        "--trace-level",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        help="Tracing detail level: 0=disabled (default), 1=basic (iteration metrics + inputs), "
        "2=full (+ layer activations), 3=debug (+ gradients and loss)",
    )
    trace_group.add_argument(
        "--trace-sample",
        type=int,
        default=None,
        metavar="N",
        help="Capture every Nth iteration (default: 1 = all iterations). "
        "Use higher values to reduce overhead (e.g., 10 = every 10th iteration)",
    )
    trace_group.add_argument(
        "--trace-storage",
        type=str,
        choices=["hdf5", "json"],
        default=None,
        help="Storage backend: json=text/readable, hdf5=binary/compressed (default: from config.yaml)",
    )
    trace_group.add_argument(
        "--trace-output",
        type=str,
        help="Custom trace output path (default: <output-dir>/traces.{hdf5|json})",
    )

    # Validation options
    validation_group = parser.add_argument_group("Validation Options")
    validation_group.add_argument(
        "--validate-soundness",
        action="store_true",
        help="After --verify {netfactory,vnnlib,torchvision}, run unified two-level soundness validation: Level 1 counterexample cross-check (all solvers) + Level 2 per-neuron bounds check (analyze facts for interval/hybridz; dual forward bounds for --solvers dual)",
    )
    validation_group.add_argument(
        "--networks",
        type=str,
        help="Comma-separated list of networks to validate (default: all)",
    )
    validation_group.add_argument(
        "--solvers",
        nargs="+",
        default=["gurobi", "torchlp"],
        help=(
            "Verification solvers: gurobi, torchlp, hybridz, or dual "
            "(default: gurobi torchlp)"
        ),
    )
    validation_group.add_argument(
        "--tf-modes",
        nargs="+",
        default=None,
        help=(
            "Transfer function modes for bounds propagation: interval or "
            "hybridz (default: interval); standalone hybridz selects HybridzTF "
            "and dual ignores this option"
        ),
    )
    validation_group.add_argument(
        "--input-samples",
        type=int,
        default=10,
        dest="samples",
        help="Number of input samples for Level 2 bounds validation (default: 10)",
    )
    validation_group.add_argument(
        "--per-neuron-topk",
        type=int,
        default=10,
        metavar="K",
        help="Number of worst per-neuron violations to report (default: 10). "
        "The bounds check itself is zero-tolerance by default — any deviation "
        "outside [lb, ub] is flagged as unsound (see --bounds-tolerance).",
    )
    validation_group.add_argument(
        "--bounds-tolerance",
        type=str,
        default="auto",
        metavar="ABS[,REL]|auto",
        help="FP-noise floor for the per-neuron bounds check: violation iff "
        "gap > ABS + REL*max(|lb|,|ub|). Default 'auto' = 100 ulp of --dtype "
        "(~1.2e-5 float32, ~2.2e-14 float64) — the arithmetic noise floor "
        "between abstract and concrete kernels, far below any genuine "
        "unsoundness. Pass '0,0' for strict zero tolerance.",
    )
    validation_group.add_argument(
        "--batch-sizes",
        type=lambda s: [
            (None if (b.strip() == "" or b.strip().lower() == "none") else int(b))
            for b in s.split(",")
        ],
        default=None,
        metavar="B1,B2,...",
        help="Batch sizes to validate at, e.g. '1,4'. Use 'none' for the "
        "network's native batch (from JSON). When omitted, falls back to "
        "the ``validate.batch_sizes`` list in config_gen_act_net.yaml, "
        "then to ``[None]`` (native only).",
    )
    validation_group.add_argument(
        "--ignore-errors",
        action="store_true",
        help="Always exit 0 (ignore failures and errors for CI)",
    )
    validation_group.add_argument(
        "--merge-split-relus",
        action="store_true",
        dest="merge_split_relus",
        help="Collapse provably-affine DENSE->ReLU->DENSE sandwiches (ReluSplitter "
             "inverse) on loaded models before verification",
    )

    # Add standard device/dtype arguments (shared across all ACT CLIs)
    add_device_args(parser)

    _add_fuzz_config_args(parser)

    args = parser.parse_args()

    # Initialize device manager from CLI arguments
    initialize_from_args(args)

    # Handle --dataset as alias for --category (for VNNLIB)
    # This provides a more intuitive interface: python -m act.pipeline --fuzz --dataset cifar100_2024
    if args.creator == "vnnlib" and args.dataset and not args.category:
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
