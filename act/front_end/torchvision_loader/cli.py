#!/usr/bin/env python3
"""
Command-Line Interface for TorchVision Dataset-Model Mapping.

Provides CLI tools for exploring datasets, downloading dataset-model pairs,
validating compatibility, and running comprehensive tests.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

import argparse
from typing import Optional

from act.util.cli_utils import add_device_args, initialize_from_args
from act.util.format_utils import rule

from act.front_end.torchvision_loader.data_model_mapping import (
    DATASET_MODEL_MAPPING,
    get_dataset_info,
    list_datasets_by_category,
    get_all_categories,
    get_preprocessing_transforms,
    validate_dataset_model_compatibility,
    find_dataset_name,
    find_model_name,
)

from act.front_end.torchvision_loader.data_model_loader import (
    download_dataset_model_pair,
    load_dataset_model_pair,
    list_downloaded_pairs,
    model_inference_with_dataset,
    _format_size,
)


def print_mapping_summary(category: Optional[str] = None):
    """
    Print a summary of dataset-to-model mappings.
    
    Args:
        category: If provided, only show datasets in this category
    """
    print(rule(100))
    print("TORCHVISION DATASET → MODEL MAPPING SUMMARY")
    print(rule(100))
    
    categories_to_show = [category] if category else get_all_categories()
    
    for cat in categories_to_show:
        datasets = list_datasets_by_category(cat)
        if not datasets:
            continue
            
        print(f"\n{rule(100)}")
        print(f"{cat.upper()} ({len(datasets)} datasets)")
        print(rule(100))
        
        for dataset in sorted(datasets):
            info = DATASET_MODEL_MAPPING[dataset]
            print(f"\n{dataset}:")
            print(f"  Models ({len(info['models'])}): {', '.join(info['models'][:5])}"
                  f"{' ...' if len(info['models']) > 5 else ''}")
            print(f"  Input Size: {info['input_size']}")
            if info['num_classes']:
                print(f"  Classes: {info['num_classes']}")
            print(f"  Notes: {info['notes']}")


def print_preprocessing_summary():
    """
    Print an aggregated summary of preprocessing requirements across all datasets.
    
    Groups datasets by their preprocessing needs:
    - Grayscale→RGB conversion
    - Resize requirements
    - Ready without preprocessing
    """
    print(rule(100))
    print("PREPROCESSING REQUIREMENTS SUMMARY")
    print(rule(100))
    
    needs_grayscale_rgb = []
    needs_resize = []
    ready_to_use = []
    
    # Check ALL datasets
    for dataset_name in sorted(DATASET_MODEL_MAPPING.keys()):
        preprocessing = get_preprocessing_transforms(dataset_name)
        
        if not preprocessing:
            ready_to_use.append(dataset_name)
            continue
            
        if preprocessing.get("grayscale_to_rgb", False):
            needs_grayscale_rgb.append(dataset_name)
        
        if "resize_to" in preprocessing:
            needs_resize.append(dataset_name)
        
        if not preprocessing.get("grayscale_to_rgb") and "resize_to" not in preprocessing:
            ready_to_use.append(dataset_name)
    
    print(f"\nDatasets requiring grayscale→RGB conversion: {len(needs_grayscale_rgb)}")
    if needs_grayscale_rgb:
        print(f"  {', '.join(needs_grayscale_rgb)}")
    
    print(f"\nDatasets requiring resize to 224×224: {len(needs_resize)}")
    if needs_resize:
        print(f"  {', '.join(needs_resize)}")
    
    print(f"\nDatasets ready without preprocessing: {len(ready_to_use)}")
    if ready_to_use:
        print(f"  {', '.join(ready_to_use)}")
    else:
        print(f"  (None - all require some preprocessing)")
    
    print(rule(100))


def print_dataset_detail(dataset_name: str):
    """
    Print detailed information about a specific dataset (case-insensitive).
    
    Args:
        dataset_name: Name of the dataset (case-insensitive)
    """
    actual_name = find_dataset_name(dataset_name)
    info = get_dataset_info(actual_name)
    
    print(rule(100))
    print(f"DATASET: {actual_name}")
    print(rule(100))
    print(f"Category: {info['category']}")
    print(f"Input Size: {info['input_size']}")
    if info['num_classes']:
        print(f"Number of Classes: {info['num_classes']}")
    
    # Show preprocessing requirements
    if "preprocessing" in info:
        print(f"\nPreprocessing Required:")
        preprocessing = info["preprocessing"]
        if preprocessing.get("grayscale_to_rgb", False):
            print(f"  ✓ Grayscale to RGB conversion (1 channel → 3 channels)")
        if "resize_to" in preprocessing:
            target = preprocessing["resize_to"]
            print(f"  ✓ Resize to {target[0]}×{target[1]}")
        if "normalize" in preprocessing:
            norm = preprocessing["normalize"]
            print(f"  ✓ Normalize - Mean: {norm['mean'][:3]}, Std: {norm['std'][:3]}")
    
    print(f"\nRecommended Models ({len(info['models'])}):")
    for i, model in enumerate(info['models'], 1):
        print(f"  {i:2d}. {model}")
    print(f"\nNotes:")
    print(f"  {info['notes']}")
    print(rule(100))


def main():
    """
    Main CLI entry point for TorchVision-specific operations.
    
    Note: For common operations like --list, --search, --download, and --info,
    use the unified CLI: python -m act.front_end
    
    This CLI provides TorchVision-specific functionality like preprocessing
    details, compatibility validation, and comprehensive testing.
    """
    parser = argparse.ArgumentParser(
        description="TorchVision-Specific Dataset/Model Operations (use 'python -m act.front_end' for common operations)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TorchVision-Specific Commands:
  --category CATEGORY       Show datasets in a specific category
  --dataset DATASET         Show detailed dataset information
  --summary                 Show complete mapping summary with all datasets
  --preprocessing-summary   Show preprocessing requirements across datasets
  --show-preprocessing DS   Show specific preprocessing for a dataset
  --validate DS MODEL       Validate dataset-model compatibility
  --models-for DATASET      Show recommended models for a dataset
  --datasets-for MODEL      Show compatible datasets for a model
  --load-torchvision DS M   Load downloaded dataset-model pair
  --inference DS MODEL      Test inference on a dataset-model pair

For Common Operations (list, search, download, info):
  Use the unified CLI: python -m act.front_end
  Examples:
    python -m act.front_end --list
    python -m act.front_end --search mnist
    python -m act.front_end --download MNIST
    python -m act.front_end --info CIFAR10
        """
    )
    
    # Domain-specific commands (keep in TorchVision CLI)
    parser.add_argument(
        "--category", "-c",
        type=str,
        help="Show datasets in specific category (classification, detection, segmentation, video, optical_flow)"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        help="Show detailed information for a specific dataset"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print complete mapping summary with all datasets and categories"
    )
    # Preprocessing analysis commands
    parser.add_argument(
        "--show-preprocessing",
        dest="show_preprocessing",
        type=str,
        metavar="DATASET",
        help="Show preprocessing requirements for a specific dataset"
    )
    parser.add_argument(
        "--preprocessing-summary",
        action="store_true",
        dest="preprocessing_summary",
        help="Show aggregated preprocessing requirements across all datasets"
    )
    # Compatibility and validation commands
    parser.add_argument(
        "--validate",
        nargs=2,
        metavar=("DATASET", "MODEL"),
        help="Validate dataset-model compatibility (e.g., --validate MNIST resnet18)"
    )
    # Model/dataset relationship queries
    parser.add_argument(
        "--models-for",
        type=str,
        metavar="DATASET",
        dest="models_for",
        help="Show all recommended models for a specific dataset"
    )
    parser.add_argument(
        "--datasets-for",
        type=str,
        metavar="MODEL",
        dest="datasets_for",
        help="Show all datasets that can run inference with a specific model"
    )
    # Download/load commands (TorchVision-specific; for the cross-creator CLI use ``python -m act.front_end``).
    parser.add_argument(
        "--download",
        nargs=2,
        metavar=("DATASET", "MODEL"),
        help="Download a dataset-model pair (for basic usage, prefer: python -m act.front_end --download DATASET)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "both"],
        help="Dataset split to download (default: test)"
    )
    parser.add_argument(
        "--list-downloads",
        action="store_true",
        dest="list_downloads",
        help="List downloaded pairs (for simple listing, prefer: python -m act.front_end --list-downloads)"
    )
    # Loading and inference commands
    parser.add_argument(
        "--load-torchvision",
        nargs=2,
        metavar=("DATASET", "MODEL"),
        dest="load_torchvision",
        help="Load a downloaded dataset-model pair (e.g., --load-torchvision MNIST simple_cnn)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        dest="batch_size",
        help="Batch size for DataLoader when loading (default: 1)"
    )
    parser.add_argument(
        "--inference",
        nargs=2,
        metavar=("DATASET", "MODEL"),
        help="Test inference for a specific dataset-model pair (e.g., --inference MNIST resnet18)"
    )
    parser.add_argument(
        "--inference-split",
        type=str,
        default="test",
        choices=["train", "test"],
        dest="inference_split",
        help="Dataset split for inference testing (default: test)"
    )
    
    # Add standard device/dtype arguments
    add_device_args(parser)
    
    args = parser.parse_args()
    
    # Initialize device manager from CLI arguments
    initialize_from_args(args)
    
    # Handle commands
    if args.category:
        try:
            datasets = list_datasets_by_category(args.category)
            print(f"\nDatasets in category '{args.category}':")
            for name in sorted(datasets):
                print(f"  • {name}")
        except ValueError as e:
            print(f"Error: {e}")
            raise
    
    elif args.dataset:
        try:
            print_dataset_detail(args.dataset)
        except ValueError as e:
            print(f"Error: {e}")
            raise
    
    elif args.summary:
        print_mapping_summary()
    
    elif args.preprocessing_summary:
        print_preprocessing_summary()
    
    elif args.download:
        # Download dataset-model pair
        dataset_name, model_name = args.download
        result = download_dataset_model_pair(
            dataset_name=dataset_name,
            model_name=model_name,
            split=args.split
        )
        
        assert result['status'] == 'success', f"Download failed: {result['message']}"
        # Success message already printed by download function
    
    elif args.list_downloads:
        # List all downloaded pairs
        downloaded = list_downloaded_pairs()
        
        if not downloaded:
            print("\nNo downloaded dataset-model pairs found.")
            print("Use --download DATASET MODEL to download a pair.")
        else:
            print(f"\n{rule()}")
            print(f"DOWNLOADED DATASET-MODEL PAIRS ({len(downloaded)})")
            print(f"{rule()}")
            
            # Calculate total size
            total_size_bytes = sum(item.get('size_bytes', 0) for item in downloaded)
            total_size_formatted = _format_size(total_size_bytes)
            
            for item in downloaded:
                print(f"\n{item['dataset']} + {item['model']}")
                print(f"  Category: {item['category']}")
                print(f"  Classes: {item['num_classes']}")
                print(f"  Splits: {', '.join(item['splits_downloaded'])}")
                print(f"  Size: {item.get('size_formatted', 'Unknown')}")
                print(f"  Location: {item['paths']['raw_data']}")
                if item['preprocessing_required']:
                    print(f"  Preprocessing: {', '.join(item['preprocessing_steps'])}")
            
            print(f"\n{rule()}")
            print(f"Total Size: {total_size_formatted}")
            print(f"{rule()}")
    
    elif args.load_torchvision:
        # Load a downloaded dataset-model pair
        dataset_name = args.load_torchvision[0]
        model_name = args.load_torchvision[1]
        split = "test"  # Default split
        batch_size = args.batch_size
        
        print(f"\n{rule()}")
        print(f"LOADING DATASET-MODEL PAIR")
        print(f"{rule()}")
        print(f"Dataset: {dataset_name}")
        print(f"Model: {model_name}")
        print(f"Split: {split}")
        print(f"Batch Size: {batch_size}")
        print(f"{rule()}\n")
        
        try:
            # Load the dataset and model
            result = load_dataset_model_pair(
                dataset_name=dataset_name,
                model_name=model_name,
                split=split,
                batch_size=batch_size,
                auto_download=True
            )
            
            # Display success information
            print(f"✓ Successfully loaded dataset and model!\n")
            
            # Dataset info
            print(f"Dataset Information:")
            print(f"  • Total samples: {len(result['dataset']):,}")
            print(f"  • Split: {split}")
            print(f"  • Category: {result['metadata']['category']}")
            print(f"  • Classes: {result['metadata']['num_classes']}")
            
            # Model info
            print(f"\nModel Information:")
            print(f"  • Architecture: {model_name}")
            total_params = sum(p.numel() for p in result['model'].parameters())
            print(f"  • Total parameters: {total_params:,}")
            
            # DataLoader info
            print(f"\nDataLoader Information:")
            print(f"  • Batch size: {batch_size}")
            print(f"  • Number of batches: {len(result['dataloader'])}")
            
            # Preprocessing info
            if result['metadata']['preprocessing_required']:
                print(f"\nPreprocessing Applied:")
                for step in result['metadata']['preprocessing_steps']:
                    print(f"  • {step}")
            
            print(f"\n{rule()}")
            print(f"✓ Load completed successfully!")
            print(f"{rule()}\n")
            
        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            print(f"\nThe dataset-model pair has not been downloaded yet.")
            print(f"Use: python -m act.front_end.torchvision_loader.cli --download {dataset_name} {model_name}")
            raise
        except ValueError as e:
            print(f"✗ Error: {e}")
            raise
    
    elif args.models_for:
        # Show all recommended models for a dataset (case-insensitive)
        try:
            dataset_name = find_dataset_name(args.models_for)
            info = get_dataset_info(dataset_name)
            models = info['models']
            
            print(f"\n{rule(100)}")
            print(f"RECOMMENDED MODELS FOR: {dataset_name}")
            print(f"{rule(100)}")
            print(f"Category: {info['category']}")
            print(f"Input Size: {info['input_size']}")
            if info['num_classes']:
                print(f"Number of Classes: {info['num_classes']}")
            
            # Check preprocessing requirements
            preprocessing = get_preprocessing_transforms(dataset_name)
            if preprocessing:
                print(f"\nPreprocessing Required:")
                if preprocessing.get("grayscale_to_rgb", False):
                    print(f"  • Grayscale → RGB conversion")
                if "resize_to" in preprocessing:
                    target = preprocessing["resize_to"]
                    print(f"  • Resize to {target[0]}×{target[1]}")
                if "normalize" in preprocessing:
                    print(f"  • Normalize (dataset-specific mean/std)")
            else:
                print(f"\nNo preprocessing required")
            
            print(f"\nRecommended Models ({len(models)}):")
            
            # Check which models are available in torchvision
            import torchvision
            available_models = []
            custom_models = []
            
            for model in models:
                if hasattr(torchvision.models, model):
                    available_models.append(model)
                else:
                    custom_models.append(model)
            
            if available_models:
                print(f"\n  Standard TorchVision Models ({len(available_models)}):")
                for i, model in enumerate(available_models, 1):
                    print(f"    {i:2d}. {model}")
            
            if custom_models:
                print(f"\n  Custom/External Models ({len(custom_models)}):")
                for i, model in enumerate(custom_models, 1):
                    print(f"    {i:2d}. {model}")
            
            print(f"\nNotes: {info['notes']}")
            print(f"{rule(100)}")
            
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Use --list to see all available datasets")
            raise
    
    elif args.datasets_for:
        # Show all datasets that can run inference with a model (case-insensitive)
        import torchvision
        
        model_name = find_model_name(args.datasets_for)
        
        # Check if model exists in torchvision
        if not hasattr(torchvision.models, model_name):
            print(f"\nNote: '{model_name}' not found in torchvision.models (might be custom)")
        
        # Find all datasets that include this model (case-insensitive)
        matching_datasets = []
        for dataset_name, info in DATASET_MODEL_MAPPING.items():
            # Case-insensitive comparison
            if any(model.lower() == model_name.lower() for model in info['models']):
                matching_datasets.append({
                    'name': dataset_name,
                    'category': info['category'],
                    'input_size': info['input_size'],
                    'num_classes': info['num_classes'],
                    'preprocessing': get_preprocessing_transforms(dataset_name)
                })
        
        print(f"\n{rule(100)}")
        print(f"DATASETS COMPATIBLE WITH: {model_name}")
        print(f"{rule(100)}")
        
        if not matching_datasets:
            print(f"\nNo datasets found that recommend '{model_name}'")
            print(f"\nNote: This doesn't mean the model can't be used with datasets,")
            print(f"      just that it's not in the recommended models list.")
        else:
            print(f"\nFound {len(matching_datasets)} dataset(s):\n")
            
            # Group by category
            by_category = {}
            for ds in matching_datasets:
                cat = ds['category']
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(ds)
            
            # Print by category
            for category in sorted(by_category.keys()):
                datasets = by_category[category]
                print(f"\n{category.upper()} ({len(datasets)} dataset{'s' if len(datasets) > 1 else ''}):")
                print(rule(100, "-"))
                
                for ds in sorted(datasets, key=lambda x: x['name']):
                    print(f"\n  {ds['name']}")
                    print(f"    Input Size: {ds['input_size']}")
                    if ds['num_classes']:
                        print(f"    Classes: {ds['num_classes']}")
                    
                    # Show preprocessing if needed
                    if ds['preprocessing']:
                        prep_steps = []
                        if ds['preprocessing'].get('grayscale_to_rgb'):
                            prep_steps.append("Grayscale→RGB")
                        if 'resize_to' in ds['preprocessing']:
                            target = ds['preprocessing']['resize_to']
                            prep_steps.append(f"Resize→{target[0]}×{target[1]}")
                        if 'normalize' in ds['preprocessing']:
                            prep_steps.append("Normalize")
                        print(f"    Preprocessing: {', '.join(prep_steps)}")
                    else:
                        print(f"    Preprocessing: None required")
        
        print(f"\n{rule(100)}")
    
    elif args.inference:
        # Test inference for a specific dataset-model pair
        dataset_name, model_name = args.inference
        split = args.inference_split
        
        print(f"\n{rule()}")
        print(f"INFERENCE TEST: {dataset_name} + {model_name}")
        print(f"{rule()}")
        print(f"Split: {split}")
        print(f"{rule()}\n")
        
        try:
            result = model_inference_with_dataset(
                dataset_name=dataset_name,
                model_name=model_name,
                split=split,
                verbose=True
            )
            
            print(f"\n{rule()}")
            print(f"INFERENCE TEST RESULT")
            print(f"{rule()}")
            print(f"Status: {result['status'].upper()}")
            
            if result['status'] == 'success':
                print(f"✓ Inference successful!")
                print(f"  Output shape: {result['output_shape']}")
                print(f"  Dataset samples: {result['num_samples']}")
            else:
                print(f"✗ Inference failed!")
                print(f"  Error: {result.get('error', 'Unknown error')}")
                if 'num_samples' in result:
                    print(f"  Dataset samples: {result['num_samples']}")
            
            print(f"{rule()}")
            assert result['status'] == 'success', f"Inference failed: {result.get('error')}"
            
        except FileNotFoundError as e:
            print(f"\n✗ Error: Dataset-model pair not found locally")
            print(f"   {e}")
            print(f"\nDownload first with:")
            print(f"   python -m act.front_end.torchvision_loader.cli --download {dataset_name} {model_name}")
            raise
        except ValueError as e:
            print(f"\n✗ Error: {e}")
            raise
    
    elif args.validate:
        user_dataset, user_model = args.validate
        try:
            dataset_name = find_dataset_name(user_dataset)
            model_name = find_model_name(user_model)
            
            result = validate_dataset_model_compatibility(dataset_name, model_name)
            print(f"\n{rule(100)}")
            print(f"COMPATIBILITY CHECK: {dataset_name} + {model_name}")
            print(f"{rule(100)}")
            print(f"Compatible: {'✓ YES' if result['compatible'] else '✗ NO'}")
            
            if result['issues']:
                print(f"\nIssues:")
                for issue in result['issues']:
                    print(f"  ⚠ {issue}")
            
            if result['preprocessing_required']:
                print(f"\nPreprocessing Required:")
                for step in result['preprocessing_steps']:
                    print(f"  • {step}")
            else:
                print(f"\nNo preprocessing required - direct compatibility!")
            
            print(f"{rule(100)}")
        except ValueError as e:
            print(f"Error: {e}")
            raise
    
    elif args.show_preprocessing:
        try:
            dataset_name = find_dataset_name(args.show_preprocessing)
            
            preprocessing = get_preprocessing_transforms(dataset_name)
            info = get_dataset_info(dataset_name)
            print(f"\n{rule(100)}")
            print(f"PREPROCESSING FOR: {dataset_name}")
            print(f"{rule(100)}")
            print(f"Input Size: {info['input_size']}")
            
            if preprocessing:
                print(f"\nRequired Transformations:")
                if preprocessing.get("grayscale_to_rgb", False):
                    print(f"  1. Grayscale → RGB (repeat channels)")
                if "resize_to" in preprocessing:
                    print(f"  2. Resize to {preprocessing['resize_to']}")
                if "normalize" in preprocessing:
                    norm = preprocessing["normalize"]
                    print(f"  3. Normalize:")
                    print(f"     Mean: {norm['mean']}")
                    print(f"     Std:  {norm['std']}")
                
                print(f"\nPyTorch Code Example:")
                print(f"  from act.front_end.torchvision_loader.data_model_mapping import create_preprocessing_pipeline")
                print(f"  transform = create_preprocessing_pipeline('{dataset_name}')")
                print(f"  dataset = torchvision.datasets.{dataset_name}(")
                print(f"      root='./data', transform=transform, download=True)")
            else:
                print(f"\nNo special preprocessing required!")
            print(f"{rule(100)}")
        except ValueError as e:
            print(f"Error: {e}")
            raise
    
    else:
        # Default: show quick examples
        print("\n" + rule(100))
        print("TORCHVISION DOMAIN-SPECIFIC CLI")
        print(rule(100))
        print("\nFor common operations, use the unified CLI:")
        print("  python -m act.front_end --list              # List all datasets/categories")
        print("  python -m act.front_end --search mnist      # Search across all")
        print("  python -m act.front_end --info CIFAR10      # Show dataset info")
        print("  python -m act.front_end --download MNIST    # Download dataset + models")
        
        print("\n" + rule(100, "-"))
        print("TorchVision-Specific Commands (use --help for full list):")
        print("  python -m act.front_end.torchvision_loader --category classification")
        print("  python -m act.front_end.torchvision_loader --dataset CIFAR10")
        print("  python -m act.front_end.torchvision_loader --summary")
        print("  python -m act.front_end.torchvision_loader --preprocessing-summary")
        print("  python -m act.front_end.torchvision_loader --show-preprocessing MNIST")
        print("  python -m act.front_end.torchvision_loader --validate MNIST resnet18")
        print("  python -m act.front_end.torchvision_loader --models-for CIFAR10")
        print("  python -m act.front_end.torchvision_loader --datasets-for resnet18")
        print("  python -m act.front_end.torchvision_loader --inference MNIST resnet18")
        print(rule(100))


if __name__ == "__main__":
    main()
