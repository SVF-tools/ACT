#!/usr/bin/env python3
"""
Command-Line Interface for VNNLIB Category Management.

Provides CLI tools for exploring VNN-COMP categories, downloading benchmarks,
parsing VNNLIB files, and managing ONNX models.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

import argparse
from typing import Optional

from act.util.cli_utils import add_device_args, initialize_from_args
from act.util.format_utils import rule

from act.front_end.vnnlib_loader.category_mapping import (
    CATEGORY_MAPPING,
    get_category_info,
    list_categories,
    list_categories_by_type,
    get_all_types,
    search_categories,
    find_category_name,
    get_summary_statistics,
)
from act.front_end.vnnlib_loader.data_model_loader import (
    download_vnnlib_category,
    list_downloaded_pairs,
)


def print_category_list(category_type: Optional[str] = None):
    """
    Print list of available VNNLIB categories.
    
    Args:
        category_type: If provided, only show categories of this type
    """
    if category_type:
        categories = list_categories_by_type(category_type)
        print(f"\n{rule(100)}")
        print(f"VNNLIB CATEGORIES - TYPE: {category_type}")
        print(f"{rule(100)}")
    else:
        categories = list_categories()
        print(f"\n{rule(100)}")
        print(f"ALL VNNLIB CATEGORIES ({len(categories)})")
        print(f"{rule(100)}")
    
    if not categories:
        print(f"No categories found for type: {category_type}")
        return
    
    # Group by type for display
    by_type = {}
    for cat_name in categories:
        info = CATEGORY_MAPPING[cat_name]
        cat_type = info['type']
        if cat_type not in by_type:
            by_type[cat_type] = []
        by_type[cat_type].append((cat_name, info))
    
    # Print by type
    for cat_type in sorted(by_type.keys()):
        items = by_type[cat_type]
        print(f"\n{cat_type.upper()} ({len(items)} categories)")
        print(rule(100, "-"))
        
        for cat_name, info in sorted(items, key=lambda x: x[0]):
            print(f"  {cat_name:30s} ({info['year']}) - {info['description']}")


def print_category_detail(category_name: str):
    """
    Print detailed information about a specific category.
    
    Args:
        category_name: Name of the category (case-insensitive)
    """
    try:
        actual_name = find_category_name(category_name)
        info = get_category_info(actual_name)
        
        print(f"\n{rule(100)}")
        print(f"CATEGORY: {actual_name}")
        print(f"{rule(100)}")
        print(f"Type: {info['type']}")
        print(f"Year: {info['year']}")
        print(f"Description: {info['description']}")
        
        print(f"\nModel Information:")
        print(f"  • Models: {info['models']}")
        print(f"  • Properties: {info['properties']}")
        
        print(f"\nDimensions:")
        print(f"  • Input: {info['input_dim']}")
        print(f"  • Output: {info['output_dim']}")
        
        print(f"{rule(100)}\n")
        
    except ValueError as e:
        print(f"Error: {e}")
        raise


def print_summary_statistics():
    """Print summary statistics about VNNLIB categories."""
    stats = get_summary_statistics()
    
    print(f"\n{rule(100)}")
    print(f"VNNLIB CATEGORIES SUMMARY")
    print(f"{rule(100)}")
    print(f"Total Categories: {stats['total_categories']}")
    print(f"Category Types: {stats['total_types']}")
    print(f"Year Range: {stats['oldest_year']} - {stats['newest_year']}")
    
    print(f"\nCategories by Type:")
    for cat_type, count in sorted(stats['categories_by_type'].items()):
        print(f"  • {cat_type:30s}: {count:2d} categories")
    
    print(f"{rule(100)}\n")


def main():
    """Main CLI entry point for VNNLIB category management."""
    parser = argparse.ArgumentParser(
        description="VNNLIB Category Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # List and search commands
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available VNNLIB categories"
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        help="Show categories of specific type (e.g., image_classification, control, collision_avoidance)"
    )
    parser.add_argument(
        "--info", "-i",
        type=str,
        metavar="CATEGORY",
        help="Show detailed information about a specific category"
    )
    parser.add_argument(
        "--search", "-s",
        type=str,
        help="Search for categories by name"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary statistics"
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        dest="list_types",
        help="List all available category types"
    )
    
    # Download and management commands
    parser.add_argument(
        "--download",
        type=str,
        metavar="CATEGORY",
        help="Download a VNNLIB category (ONNX + VNNLIB files)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist (use with --download)"
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        metavar="N",
        dest="max_instances",
        help="Maximum number of instances to download (use with --download)"
    )
    parser.add_argument(
        "--list-downloads",
        action="store_true",
        dest="list_downloads",
        help="List all downloaded categories"
    )
    # Add standard device/dtype arguments
    add_device_args(parser)
    
    args = parser.parse_args()
    
    # Initialize device manager from CLI arguments
    initialize_from_args(args)
    
    # Handle commands
    if args.list:
        print_category_list()
    
    elif args.type:
        try:
            print_category_list(category_type=args.type)
        except ValueError as e:
            print(f"Error: {e}")
            print(f"\nAvailable types: {', '.join(get_all_types())}")
            raise
    
    elif args.info:
        print_category_detail(args.info)
    
    elif args.search:
        matches = search_categories(args.search)
        if matches:
            print(f"\nCategories matching '{args.search}':")
            for name in sorted(matches):
                info = CATEGORY_MAPPING[name]
                print(f"  • {name:30s} ({info['type']}) - {info['description']}")
        else:
            print(f"No categories found matching '{args.search}'")
    
    elif args.summary:
        print_summary_statistics()
    
    elif args.list_types:
        types = get_all_types()
        print(f"\nAvailable Category Types ({len(types)}):")
        for cat_type in sorted(types):
            # Count categories of this type
            count = len(list_categories_by_type(cat_type))
            print(f"  • {cat_type:30s} ({count} categories)")
    
    elif args.download:
        # Download VNNLIB category
        category = args.download
        
        try:
            # Validate category exists
            actual_name = find_category_name(category)
            info = get_category_info(actual_name)
            
            print(f"\n{rule(100)}")
            print(f"DOWNLOADING VNNLIB CATEGORY: {actual_name}")
            print(f"{rule(100)}")
            print(f"Type: {info['type']}")
            print(f"Description: {info['description']}")
            print(f"Year: {info['year']}")
            
            if args.max_instances:
                print(f"Max instances: {args.max_instances}")
            if args.force:
                print(f"Force re-download: Yes")
            
            print(f"\nDownloading from VNN-COMP GitHub repository...")
            print(f"{rule(100)}\n")
            
            # Download the category
            result = download_vnnlib_category(
                category=actual_name,
                force_redownload=args.force
            )
            
            if result['status'] != 'success':
                print(f"\n{rule(100)}")
                print(f"⚠️  DOWNLOAD FAILED")
                print(f"{rule(100)}")
                print(f"Error: {result.get('message', 'Unknown error')}")
                print(f"{rule(100)}\n")
            assert result['status'] == 'success', f"Download failed: {result.get('message')}"
            
            print(f"\n{rule(100)}")
            print(f"✓ DOWNLOAD SUCCESSFUL")
            print(f"{rule(100)}")
            print(f"Category: {actual_name}")
            print(f"Path: {result['category_path']}")
            print(f"Instances: {result['num_instances']}")
            if 'num_onnx_models' in result:
                print(f"ONNX models: {result['num_onnx_models']}")
            if 'num_vnnlib_specs' in result:
                print(f"VNNLIB specs: {result['num_vnnlib_specs']}")
            print(f"{rule(100)}\n")
                
        except ValueError as e:
            print(f"\n{rule(100)}")
            print(f"⚠️  INVALID CATEGORY")
            print(f"{rule(100)}")
            print(f"Error: {e}")
            print(f"\nUse --list to see available categories")
            print(f"{rule(100)}\n")
            raise
    
    elif args.list_downloads:
        # List downloaded categories
        downloaded = list_downloaded_pairs()
        
        if not downloaded:
            print(f"\nNo downloaded VNNLIB categories found.")
            print(f"Use --download <category> to download a benchmark.")
            return
        
        # Group by category
        by_category = {}
        for pair in downloaded:
            cat = pair['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(pair)
        
        print(f"\n{rule(100)}")
        print(f"DOWNLOADED VNNLIB CATEGORIES ({len(by_category)})")
        print(f"{rule(100)}")
        
        for category in sorted(by_category.keys()):
            instances = by_category[category]
            info = CATEGORY_MAPPING.get(category, {})
            
            print(f"\n{category} ({len(instances)} instances)")
            if info:
                print(f"  Type: {info.get('type', 'N/A')}")
                print(f"  Description: {info.get('description', 'N/A')}")
            print(f"  Path: {instances[0]['category_path']}")
        
        print(f"\n{rule(100)}")
        print(f"Total instances: {len(downloaded)}")
        print(f"{rule(100)}\n")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
