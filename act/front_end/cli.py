#!/usr/bin/env python3
"""
Unified Command-Line Interface for ACT Front-End.

Provides a unified CLI that supports both TorchVision datasets/models and VNNLIB
verification benchmarks with automatic creator detection.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

import argparse
from pathlib import Path
from typing import Any, Optional, cast

from act.front_end.creator_registry import detect_creator, list_creators, get_creator
from act.util.cli_utils import add_device_args, initialize_from_args

# Import domain-specific CLIs for delegation
from act.front_end.torchvision_loader import data_model_mapping as tv_mapping
from act.front_end.torchvision_loader import data_model_loader as tv_loader
from act.front_end.vnnlib_loader import category_mapping as vnnlib_mapping
from act.front_end.bert_loader import data_loader as bert_loader
from act.back_end.config import VALID_BERT_METHODS


def print_unified_list(creator: Optional[str] = None):
    """
    Print unified list of all available datasets/categories.
    
    Args:
        creator: If provided, only show items from this creator
    """
    print(f"\n{'='*100}")
    print(f"ACT FRONT-END UNIFIED CATALOG")
    print(f"{'='*100}")
    
    if creator is None or creator == 'torchvision':
        # List TorchVision datasets
        datasets = sorted(tv_mapping.DATASET_MODEL_MAPPING.keys())
        print(f"\nTorchVision Datasets ({len(datasets)}):")
        print('-' * 100)
        for ds_name in datasets:
            info = tv_mapping.DATASET_MODEL_MAPPING[ds_name]
            category = info.get('category', 'N/A')
            num_classes = info.get('num_classes', 'N/A')
            print(f"  {ds_name:30s} [{category:15s}] - {num_classes} classes")
    
    if creator is None or creator == 'vnnlib':
        # List VNNLIB categories
        categories = vnnlib_mapping.list_categories()
        print(f"\nVNNLIB Categories ({len(categories)}):")
        print('-' * 100)
        for cat_name in sorted(categories):
            info = vnnlib_mapping.get_category_info(cat_name)
            print(f"  {cat_name:30s} ({info['type']}) - {info['description']}")
    
    if creator is None or creator == 'bert':
        datasets = bert_loader.list_bert_datasets()
        print(f"\nBERT Datasets ({len(datasets)}):")
        print('-' * 100)
        for ds_name in datasets:
            print(f"  {ds_name:30s} [sentiment] - {bert_loader.BERT_DATASETS[ds_name]}")
    
    print(f"\n{'='*100}\n")


def print_unified_search(query: str, creator: Optional[str] = None):
    """
    Search across both TorchVision and VNNLIB.
    
    Args:
        query: Search query string
        creator: If provided, only search this creator
    """
    print(f"\n{'='*100}")
    print(f"SEARCH RESULTS: '{query}'")
    print(f"{'='*100}")
    
    found_any = False
    
    if creator is None or creator == 'torchvision':
        tv_matches = tv_mapping.search_datasets(query)
        if tv_matches:
            found_any = True
            print(f"\nTorchVision Datasets ({len(tv_matches)}):")
            print('-' * 100)
            for ds_name in sorted(tv_matches):
                info = tv_mapping.DATASET_MODEL_MAPPING[ds_name]
                category = info.get('category', 'N/A')
                print(f"  {ds_name:30s} [{category}]")
    
    if creator is None or creator == 'vnnlib':
        vnnlib_matches = vnnlib_mapping.search_categories(query)
        if vnnlib_matches:
            found_any = True
            print(f"\nVNNLIB Categories ({len(vnnlib_matches)}):")
            print('-' * 100)
            for cat_name in sorted(vnnlib_matches):
                info = vnnlib_mapping.get_category_info(cat_name)
                print(f"  {cat_name:30s} ({info['type']}) - {info['description']}")
    
    if creator is None or creator == 'bert':
        bert_matches = [
            name for name in bert_loader.list_bert_datasets()
            if query.lower() in name.lower()
        ]
        if bert_matches:
            found_any = True
            print(f"\nBERT Datasets ({len(bert_matches)}):")
            print('-' * 100)
            for ds_name in sorted(bert_matches):
                print(f"  {ds_name:30s} [sentiment] - {bert_loader.BERT_DATASETS[ds_name]}")
    
    if not found_any:
        print(f"\nNo results found for '{query}'")
    
    print(f"\n{'='*100}\n")


def print_unified_info(name: str, explicit_creator: Optional[str] = None):
    """
    Show detailed information about a dataset/category with auto-detection.
    
    Args:
        name: Name of dataset or category
        explicit_creator: Override auto-detection
    """
    try:
        creator_name, normalized_name = detect_creator(name, explicit_creator)
        
        print(f"\n{'='*100}")
        print(f"DETECTED CREATOR: {creator_name.upper()}")
        print(f"{'='*100}")
        
        if creator_name == 'torchvision':
            info = tv_mapping.get_dataset_info(normalized_name)
            print(f"\nDataset: {normalized_name}")
            print(f"Category: {info.get('category', 'N/A')}")
            print(f"Input Size: {info.get('input_size', 'N/A')}")
            print(f"Classes: {info.get('num_classes', 'N/A')}")
            print(f"Notes: {info.get('notes', 'N/A')}")
            
            # Show recommended models
            models = info.get('models', [])
            if models:
                print(f"\nRecommended Models ({len(models)}):")
                for model in models:
                    print(f"  • {model}")
            
        elif creator_name == 'vnnlib':
            info = vnnlib_mapping.get_category_info(normalized_name)
            print(f"\nCategory: {normalized_name}")
            print(f"Type: {info['type']}")
            print(f"Year: {info['year']}")
            print(f"Description: {info['description']}")
            print(f"\nModel Information:")
            print(f"  • Models: {info['models']}")
            print(f"  • Properties: {info['properties']}")
            print(f"  • Input Dim: {info['input_dim']}")
            print(f"  • Output Dim: {info['output_dim']}")
        elif creator_name == 'bert':
            print(f"\nDataset: {normalized_name}")
            print("Type: binary sentiment text")
            print("Models: embedding_classifier")
            print("Input: clean embeddings [B, L, D]")
            print("Specs: LP_EMBEDDING + MARGIN_ROBUST")
        
        print(f"\n{'='*100}\n")
        
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nTip: Use --search to find available names or --creator to override detection")


def handle_unified_download(name: str, explicit_creator: Optional[str] = None):
    """
    Download a dataset/category with auto-detection.
    
    Args:
        name: Name of dataset or category
        explicit_creator: Override auto-detection
    """
    try:
        creator_name, normalized_name = detect_creator(name, explicit_creator)
        
        print(f"\n{'='*100}")
        print(f"DOWNLOADING: {normalized_name} (creator: {creator_name})")
        print(f"{'='*100}\n")
        
        if creator_name == 'torchvision':
            # For TorchVision, download dataset + all recommended models
            info = tv_mapping.get_dataset_info(normalized_name)
            models = info.get('models', [])
            
            if not models:
                print(f"⚠️  No models available for {normalized_name}")
                return
            
            print(f"Downloading dataset '{normalized_name}' with {len(models)} models...\n")
            
            success_count = 0
            for model in models:
                try:
                    result = tv_loader.download_dataset_model_pair(normalized_name, model)
                    if result['status'] == 'success':
                        print(f"✓ {normalized_name} + {model}")
                        success_count += 1
                    else:
                        print(f"✗ {normalized_name} + {model} - {result['message']}")
                except Exception as e:
                    print(f"✗ {normalized_name} + {model} - Error: {e}")
            
            print(f"\n{'='*100}")
            print(f"Downloaded {success_count}/{len(models)} model pairs")
            print(f"{'='*100}\n")
            
        elif creator_name == 'vnnlib':
            # Import VNNLIB loader
            from act.front_end.vnnlib_loader import data_model_loader as vnnlib_loader
            
            print(f"Downloading VNNLIB category '{normalized_name}'...")
            print(f"This will download:")
            print(f"  • ONNX model files")
            print(f"  • VNNLIB property files")
            print(f"  • instances.csv mapping\n")
            
            try:
                result = vnnlib_loader.download_vnnlib_category(normalized_name)
                
                if result['status'] == 'success':
                    print(f"\n{'='*100}")
                    print(f"✓ Successfully downloaded category: {normalized_name}")
                    print(f"  Location: {result['category_path']}")
                    print(f"  Instances: {result['num_instances']}")
                    print(f"{'='*100}\n")
                else:
                    print(f"\n{'='*100}")
                    print(f"✗ Download failed: {result['message']}")
                    print(f"\nNote: VNNLIB benchmarks must be downloaded manually from VNN-COMP.")
                    print(f"Expected location: data/vnnlib/{normalized_name}/")
                    print(f"\nManual download steps:")
                    print(f"  1. Visit: https://github.com/ChristopherBrix/vnncomp_benchmarks")
                    print(f"  2. Download the '{normalized_name}' benchmark")
                    print(f"  3. Extract to: data/vnnlib/{normalized_name}/")
                    print(f"  4. Ensure structure includes:")
                    print(f"     - onnx/         (ONNX model files)")
                    print(f"     - vnnlib/       (VNNLIB property files)")
                    print(f"     - instances.csv (benchmark instances)")
                    print(f"{'='*100}\n")
                    
            except Exception as e:
                print(f"\n{'='*100}")
                print(f"✗ Download error: {e}")
                print(f"\nNote: VNNLIB benchmarks must be downloaded manually from VNN-COMP.")
                print(f"Expected location: data/vnnlib/{normalized_name}/")
                print(f"{'='*100}\n")
        elif creator_name == 'bert':
            print("BERT datasets are file-based.")
            print(f"Expected raw files under data/{normalized_name}/")
            print("SST accepts test/dev .txt and train-nodes.tsv; fixture mode is deterministic.")
        
    except ValueError as e:
        print(f"Error: {e}")


def print_list_downloads(creator: Optional[str] = None):
    """
    List all downloaded datasets/categories.
    
    Args:
        creator: If provided, only show downloads from this creator
    """
    print(f"\n{'='*100}")
    print(f"DOWNLOADED ITEMS")
    print(f"{'='*100}")
    
    if creator is None or creator == 'torchvision':
        tv_downloads = tv_loader.list_downloaded_pairs()
        if tv_downloads:
            print(f"\nTorchVision Downloads ({len(tv_downloads)}):")
            print('-' * 100)
            for item in sorted(tv_downloads, key=lambda x: (x['dataset'], x['model'])):
                print(f"  {item['dataset']:30s} + {item['model']}")
        else:
            print(f"\nNo TorchVision downloads found")
    
    if creator is None or creator == 'vnnlib':
        from act.front_end.vnnlib_loader import data_model_loader as vnnlib_loader
        
        vnnlib_downloads = vnnlib_loader.list_downloaded_pairs()
        if vnnlib_downloads:
            print(f"\nVNNLIB Downloads ({len(vnnlib_downloads)} instances):")
            print('-' * 100)
            
            # Group by category
            categories = {}
            for item in vnnlib_downloads:
                cat = item['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(item)
            
            for cat in sorted(categories.keys()):
                instances = categories[cat]
                print(f"  {cat:30s} ({len(instances)} instances)")
        else:
            print(f"\nNo VNNLIB downloads found")
    
    if creator is None or creator == 'bert':
        print(f"\nBERT Downloads:")
        print('-' * 100)
        print("  File-based loader; place SST/Yelp raw files under data/sst or data/yelp")
    
    print(f"\n{'='*100}\n")


def print_creators():
    """Print information about all available creators."""
    creators = list_creators()

    print(f"\n{'='*100}")
    print(f"AVAILABLE CREATORS")
    print(f"{'='*100}")

    for creator_name in sorted(creators):
        creator = get_creator(creator_name)
        print(f"\n{creator_name.upper()}")
        print('-' * 100)
        print(f"  Class:       {type(creator).__name__}")

        if creator_name == 'torchvision':
            datasets = list(tv_mapping.DATASET_MODEL_MAPPING.keys())
            print(f"  Description: TorchVision datasets and models")
            print(f"  Module:      act.front_end.torchvision_loader")
            print(f"  Items:       {len(datasets)} datasets")
            print(f"  Kinds:       classification, regression")
        elif creator_name == 'vnnlib':
            categories = vnnlib_mapping.list_categories()
            print(f"  Description: VNNLIB verification benchmarks")
            print(f"  Module:      act.front_end.vnnlib_loader")
            print(f"  Items:       {len(categories)} VNN-COMP categories")
            print(f"  Kinds:       robustness, safety, reachability")
        elif creator_name == 'bert':
            datasets = bert_loader.list_bert_datasets()
            print(f"  Description: BERT datasets for embedding-space verification")
            print(f"  Module:      act.front_end.bert_loader")
            print(f"  Items:       {len(datasets)} dataset names/aliases")
            print(f"  Kinds:       binary sentiment robustness")

    print(f"\n{'='*100}\n")


def print_creator_info(name: str) -> None:
    try:
        creator = get_creator(name)
        creators = list_creators()
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Available creators: {list_creators()}")
        return

    print(f"\n{'='*100}")
    print(f"CREATOR: {name.upper()}")
    print(f"{'='*100}")
    print(f"  Class:        {type(creator).__name__}")
    print(f"  Module:       {type(creator).__module__}")
    print(f"  All creators: {creators}")

    if hasattr(creator, 'config') and creator.config:
        config_keys = sorted(creator.config.keys())
        print(f"  Config keys:  {config_keys}")

    if name == 'torchvision':
        datasets = sorted(tv_mapping.DATASET_MODEL_MAPPING.keys())
        sample = datasets[:5]
        print(f"  Datasets ({len(datasets)}):  {sample} ...")
        print(f"  Kinds:        classification, regression")
        print(f"  Spec types:   BOX, LINF_BALL (epsilon perturbations)")
    elif name == 'vnnlib':
        categories = sorted(vnnlib_mapping.list_categories())
        sample = categories[:5]
        print(f"  Categories ({len(categories)}): {sample} ...")
        print(f"  Kinds:        robustness, safety, reachability")
        print(f"  Spec types:   VNNLIB SMT-LIB format")
    elif name == 'bert':
        datasets = sorted(bert_loader.list_bert_datasets())
        print(f"  Datasets ({len(datasets)}):  {datasets}")
        print(f"  Kinds:        binary sentiment robustness")
        print(f"  Spec types:   LP_EMBEDDING, MARGIN_ROBUST")

    print(f"\n{'='*100}\n")


def main():
    """Main unified CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ACT Front-End Unified CLI - Auto-detects TorchVision and VNNLIB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ============================================================================
  # LISTING - Browse available datasets and categories
  # ============================================================================
  
  # List all available items (40 TorchVision + 34 VNNLIB)
  python -m act.front_end --list
  
  # List only TorchVision datasets
  python -m act.front_end --list --creator torchvision
  
  # List only VNNLIB categories
  python -m act.front_end --list --creator vnnlib
  
  # Show available creators with details (name, class, item count, supported kinds)
  python -m act.front_end --list-creators
  
  # Show detailed metadata for a specific creator
  python -m act.front_end --info-creator vnnlib
  python -m act.front_end --info-creator torchvision
  
  # ============================================================================
  # SEARCHING - Find specific datasets or categories
  # ============================================================================
  
  # Search across both creators (auto-detects)
  python -m act.front_end --search mnist         # Finds: MNIST, FashionMNIST, KMNIST, EMNIST
  python -m act.front_end --search cifar         # Finds: CIFAR10, CIFAR100, cifar100_2024
  python -m act.front_end --search imagenet      # Finds: ImageNet (TorchVision)
  python -m act.front_end --search yolo          # Finds: yolo_2023 (VNNLIB)
  python -m act.front_end --search transformer   # Finds: vit_2023, safenlp_2024 (VNNLIB)
  python -m act.front_end --search acas          # Finds: acasxu_2023 (VNNLIB)
  python -m act.front_end --search resnet        # Finds: resnet* (VNNLIB categories)
  
  # Search with creator filter
  python -m act.front_end --search mnist --creator torchvision
  python -m act.front_end --search cifar --creator vnnlib
  
  # ============================================================================
  # INFO - Get detailed information about datasets/categories
  # ============================================================================
  
  # Auto-detect and show info (TorchVision datasets)
  python -m act.front_end --info MNIST           # Dataset info + recommended models
  python -m act.front_end --info CIFAR10         # Shows: resnet18, mobilenet_v2, etc.
  python -m act.front_end --info ImageNet        # Large-scale dataset info
  python -m act.front_end --info FashionMNIST    # Fashion items dataset
  
  # Auto-detect and show info (VNNLIB categories)
  python -m act.front_end --info acasxu_2023     # ACAS Xu collision avoidance
  python -m act.front_end --info vit_2023        # Vision Transformer verification
  python -m act.front_end --info yolo_2023       # YOLO object detection
  python -m act.front_end --info cifar100_2024   # CIFAR100 VNNLIB benchmark
  python -m act.front_end --info vggnet16_2022   # VGG-16 verification
  
  # Explicit creator override (when name could be ambiguous)
  python -m act.front_end --info MNIST --creator torchvision
  python -m act.front_end --info CIFAR10 --creator torchvision
  python -m act.front_end --info cifar100_2024 --creator vnnlib
  
  # ============================================================================
  # DOWNLOAD - Download datasets, models, and benchmarks
  # ============================================================================
  
  # Auto-detect and download (TorchVision - downloads dataset + ALL recommended models)
  python -m act.front_end --download MNIST              # Downloads MNIST + simple_cnn, lenet5, resnet18, etc.
  python -m act.front_end --download CIFAR10            # Downloads CIFAR10 + resnet18, mobilenet_v2, etc.
  python -m act.front_end --download FashionMNIST       # Downloads FashionMNIST + models
  python -m act.front_end --download ImageNet           # Downloads ImageNet (large!)
  python -m act.front_end --download SVHN               # Downloads Street View House Numbers
  
  # Auto-detect and download (VNNLIB - downloads ONNX models + VNNLIB properties)
  python -m act.front_end --download acasxu_2023        # Downloads 45 ONNX models + properties
  python -m act.front_end --download vit_2023           # Vision Transformer benchmarks
  python -m act.front_end --download yolo_2023          # YOLO verification benchmarks
  python -m act.front_end --download cifar100_2024      # CIFAR100 VNNLIB benchmarks
  python -m act.front_end --download safenlp_2024       # NLP verification benchmarks
  python -m act.front_end --download vggnet16_2022      # VGG-16 benchmarks
  
  # Force specific creator (if name could match multiple)
  python -m act.front_end --download MNIST --creator torchvision
  python -m act.front_end --download CIFAR10 --creator torchvision
  python -m act.front_end --download cifar100_2024 --creator vnnlib
  
  # ============================================================================
  # DOWNLOAD MANAGEMENT - Track what's been downloaded
  # ============================================================================
  
  # List all downloaded items (grouped by creator)
  python -m act.front_end --list-downloads
  
  # List only TorchVision downloads
  python -m act.front_end --list-downloads --creator torchvision
  
  # List only VNNLIB downloads
  python -m act.front_end --list-downloads --creator vnnlib
  
  # ============================================================================
  # MODEL SYNTHESIS & INFERENCE - Creator-specific workflows
  # ============================================================================
  
  # Run model synthesis (defaults to TorchVision)
  python -m act.front_end --synthesis
  
  # Run synthesis for specific creator
  python -m act.front_end --synthesis --creator torchvision   # PyTorch models with specs
  python -m act.front_end --synthesis --creator vnnlib        # ONNX models with VNNLIB specs
  
  # Run inference on synthesized models (defaults to TorchVision)
  python -m act.front_end --inference
  
  # Run inference for specific creator
  python -m act.front_end --inference --creator torchvision   # Validates PyTorch models
  python -m act.front_end --inference --creator vnnlib        # Validates ONNX→PyTorch models
  
  # ============================================================================
  # ADVANCED WORKFLOWS
  # ============================================================================
  
  # Download multiple categories sequentially
  python -m act.front_end --download MNIST && \\
  python -m act.front_end --download CIFAR10 && \\
  python -m act.front_end --download acasxu_2023
  
  # Search and download pipeline
  python -m act.front_end --search acas          # Find available ACAS benchmarks
  python -m act.front_end --info acasxu_2023     # Check details
  python -m act.front_end --download acasxu_2023 # Download it
  
  # Complete TorchVision workflow
  python -m act.front_end --download MNIST       # Download dataset + models
  python -m act.front_end --synthesis            # Generate wrapped models
  python -m act.front_end --inference            # Validate correctness
  
  # Check what's available vs what's downloaded
  python -m act.front_end --list                 # Show all available
  python -m act.front_end --list-downloads       # Show what's downloaded
        """
    )
    
    # Common commands
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available datasets/categories from all creators"
    )
    parser.add_argument(
        "--search", "-s",
        type=str,
        help="Search for datasets/categories by name across all creators"
    )
    parser.add_argument(
        "--info", "-i",
        type=str,
        metavar="NAME",
        help="Show detailed information (auto-detects creator)"
    )
    parser.add_argument(
        "--download", "-d",
        type=str,
        metavar="NAME",
        help="Download dataset/category (auto-detects creator)"
    )
    parser.add_argument(
        "--list-downloads",
        action="store_true",
        dest="list_downloads",
        help="List all downloaded items from all creators"
    )
    
    # Creator management
    parser.add_argument(
        "--creator", "-c",
        type=str,
        choices=['torchvision', 'vnnlib', 'bert'],
        help="Override auto-detection and use specific creator"
    )
    parser.add_argument(
        "--list-creators",
        action="store_true",
        dest="list_creators",
        help="Show information about all available creators"
    )
    parser.add_argument(
        "--info-creator",
        type=str,
        metavar="NAME",
        dest="info_creator",
        help="Show detailed metadata for a specific creator (e.g. vnnlib, torchvision)"
    )
    
    # Model synthesis and inference
    parser.add_argument(
        "--synthesis",
        action="store_true",
        help="Run model synthesis to generate verification-ready models (defaults to TorchVision, use --creator to specify)"
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run inference on synthesized models to validate correctness (defaults to TorchVision, use --creator to specify)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=[name.replace("_", "-") for name in VALID_BERT_METHODS],
        default=None,
        help="Text verifier method selector for SST/Yelp workflows.",
    )
    parser.add_argument("--p", type=float, default=None, help="Text embedding perturbation norm.")
    parser.add_argument(
        "--perturbed-words",
        type=int,
        choices=[1, 2],
        default=None,
        dest="perturbed_words",
        help="Number of text token positions perturbed together.",
    )
    parser.add_argument("--eps", type=float, default=None, help="Text verification radius.")
    parser.add_argument("--max-eps", type=float, default=None, dest="max_eps", help="Text certified-radius upper bound.")
    parser.add_argument(
        "--num-verify-iters",
        type=int,
        default=None,
        dest="num_verify_iters",
        help="Certified-radius binary-search iterations.",
    )
    parser.add_argument("--k", type=int, default=None, help="Rule-slope alpha rule threshold.")
    parser.add_argument(
        "--alpha-opt-steps",
        type=int,
        default=None,
        dest="alpha_opt_steps",
        help="Optimized-alpha optimization steps.",
    )
    
    # Add standard device/dtype arguments
    add_device_args(parser)
    
    args = parser.parse_args()
    
    # Initialize device manager from CLI arguments
    initialize_from_args(args)
    
    # Handle commands
    if args.list:
        print_unified_list(creator=args.creator)
    
    elif args.search:
        print_unified_search(args.search, creator=args.creator)
    
    elif args.info:
        print_unified_info(args.info, explicit_creator=args.creator)
    
    elif args.download:
        handle_unified_download(args.download, explicit_creator=args.creator)
    
    elif args.list_downloads:
        print_list_downloads(creator=args.creator)
    
    elif args.list_creators:
        print_creators()

    elif args.info_creator:
        print_creator_info(args.info_creator)

    elif args.synthesis:
        creator_name = args.creator if args.creator else 'torchvision'
        print(f"\n{'='*100}")
        print(f"MODEL SYNTHESIS - {creator_name.upper()}")
        print(f"{'='*100}\n")
        
        try:
            from act.front_end.model_synthesis import model_synthesis
            from act.util.model_inference import model_inference
            
            wrapped_models = model_synthesis(creator=creator_name)
            print(f"\n✓ Successfully synthesized {len(wrapped_models)} models")
            
            # Automatically run inference after synthesis
            print(f"\n{'='*100}")
            print(f"MODEL INFERENCE - {creator_name.upper()}")
            print(f"{'='*100}\n")
            
            # model_inference extracts input from InputLayer 
            successful_models = model_inference(cast(Any, wrapped_models))
            print(f"\n✓ Successfully ran inference on {len(successful_models)}/{len(wrapped_models)} models")
            print(f"  Models are ready for verification")
        except Exception as e:
            print(f"\n✗ Synthesis/Inference failed: {e}")
    
    elif args.inference:
        creator_name = args.creator if args.creator else 'torchvision'
        print(f"\n{'='*100}")
        print(f"MODEL INFERENCE - {creator_name.upper()}")
        print(f"{'='*100}\n")
        
        try:
            # Get downloaded pairs for the creator
            downloaded_pairs = []
            if creator_name == 'torchvision':
                downloaded_pairs = tv_loader.list_downloaded_pairs()
            elif creator_name == 'vnnlib':
                from act.front_end.vnnlib_loader import data_model_loader as vnnlib_loader
                downloaded_pairs = vnnlib_loader.list_downloaded_pairs()
            
            if not downloaded_pairs:
                print(f"⚠️  No downloaded models found for creator '{creator_name}'")
                print(f"   Use --download to download datasets/categories first")
                print(f"   Example: python -m act.front_end --download MNIST --creator {creator_name}")
            else:
                print(f"Found {len(downloaded_pairs)} downloaded pair(s) for {creator_name}")
                print(f"Running inference on downloaded models...\n")
                
                results = []
                
                # Run inference on each downloaded pair
                if creator_name == 'torchvision':
                    for pair_info in downloaded_pairs:
                        dataset_name = pair_info['dataset']
                        model_name = pair_info['model']
                        
                        # Test the pair
                        result = tv_loader.model_inference_with_dataset(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            split='test',
                            verbose=False
                        )
                        results.append(result)
                        
                        # Print result
                        if result['status'] == 'success':
                            print(f"✓ {dataset_name:20s} + {model_name:20s} - Output: {result['output_shape']}")
                        else:
                            print(f"✗ {dataset_name:20s} + {model_name:20s} - Error: {result.get('error', 'Unknown')[:50]}")
                
                elif creator_name == 'vnnlib':
                    from act.front_end.vnnlib_loader import data_model_loader as vnnlib_loader
                    
                    for pair_info in downloaded_pairs:
                        category = pair_info['category']
                        onnx_model = pair_info['onnx_model']
                        vnnlib_spec = pair_info['vnnlib_spec']
                        
                        # Test the pair
                        result = vnnlib_loader.model_inference_with_vnnlib(
                            category=category,
                            onnx_model=onnx_model,
                            vnnlib_spec=vnnlib_spec,
                            verbose=False
                        )
                        results.append(result)
                        
                        # Print result
                        if result['status'] == 'success':
                            print(f"✓ {category:20s} + {result['model']:30s} - Output: {result['output_shape']}")
                        else:
                            print(f"✗ {category:20s} + {result['model']:30s} - Error: {result.get('error', 'Unknown')[:50]}")
                
                # Summary
                successful = sum(1 for r in results if r['status'] == 'success')
                failed = len(results) - successful
                
                print(f"\n{'='*100}")
                print(f"INFERENCE SUMMARY")
                print(f"{'='*100}")
                print(f"✓ Successful: {successful}/{len(results)}")
                if failed > 0:
                    print(f"✗ Failed: {failed}/{len(results)}")
                print(f"{'='*100}\n")
                
        except Exception as e:
            import traceback
            print(f"\n✗ Inference failed: {e}")
            print(f"\nTraceback:")
            traceback.print_exc()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
