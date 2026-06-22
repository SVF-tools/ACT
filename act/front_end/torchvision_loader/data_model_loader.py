#!/usr/bin/env python3
"""
Dataset-Model Loader for TorchVision.

Provides functionality to download, load, and manage dataset-model pairs
from TorchVision, including automatic downloading, preprocessing, and
model instantiation.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

import json
from pathlib import Path
from typing import Any, List, Optional

import torch

# Import path configuration
from act.util.path_config import get_torchvision_data_root

# Import from data_model_mapping
from act.front_end.torchvision_loader.data_model_mapping import (
    get_dataset_info,
    validate_dataset_model_compatibility,
    create_preprocessing_pipeline,
    find_dataset_name,
    find_model_name
)

# Import custom model definitions
from act.front_end.torchvision_loader.model_definitions import _get_custom_model_definition


def _resolve_external_dataset(
    dataset_info: dict[str, Any],
    raw_dir: Path,
    train: bool,
    transform: Any = None,
    download: bool = False,
):
    """Load a non-TorchVision dataset declared via the mapping's "download" key.

    The archive's "image_root" must follow the ImageFolder layout (one
    sub-directory per class). When "index_file" and "split_file" are present,
    the dataset's official train/test split is applied via Subset.
    """
    from torch.utils.data import Subset
    from torchvision.datasets import ImageFolder
    from torchvision.datasets.utils import download_and_extract_archive

    cfg = dataset_info["download"]
    raw_dir = Path(raw_dir)
    image_root = raw_dir / cfg["image_root"]

    if not image_root.exists():
        if not download:
            raise FileNotFoundError(
                f"Archive dataset not found at {image_root}; download it first."
            )

        download_and_extract_archive(
            url=cfg["url"], download_root=str(raw_dir), md5=cfg.get("md5"),
        )

    dataset = ImageFolder(str(image_root), transform=transform)

    if "index_file" in cfg and "split_file" in cfg:
        dataset = Subset(
            dataset, _official_split_indices(dataset, raw_dir, cfg, train),
        )

    return dataset


def _official_split_indices(
    dataset: Any, raw_dir: Path, cfg: dict[str, Any], train: bool,
) -> list[int]:
    """Deterministic official-split filter from index/split text files.

    Both files are whitespace-separated: index_file maps image_id to a path
    relative to image_root's parent; split_file maps image_id to 1 (train)
    or 0 (test).
    """
    id_to_rel: dict[int, str] = {}

    for line in (raw_dir / cfg["index_file"]).read_text().splitlines():
        img_id, rel_path = line.split()

        id_to_rel[int(img_id)] = rel_path

    wanted = set()

    for line in (raw_dir / cfg["split_file"]).read_text().splitlines():
        img_id, is_train = line.split()

        if bool(int(is_train)) == train and int(img_id) in id_to_rel:
            wanted.add(id_to_rel[int(img_id)])

    root = Path(dataset.root)

    return [
        idx for idx, (path, _) in enumerate(dataset.samples)
        if Path(path).relative_to(root).as_posix() in wanted
    ]


def _download_dataset(
    root_dir: str,
    dataset_name: str,
    dataset_info: dict[str, Any],
    split: str,
) -> tuple[list[str], list[str]]:
    """
    Downloads `split` for a given `dataset_name`. Its `dataset_info` is retrieved from
    `get_dataset_info`.

    Returns a list of splits downloaded, plus any errors encountered in the download process.
    """
    raw_dir = Path(root_dir) / dataset_name / "raw"
    # We determine an "external" (non-TorchVision) dataset through the `download`
    # key, which also contains information on where the raw images are, how to split
    # training/testing sets after extraction.
    is_external = "download" in dataset_info
    
    downloaded_splits = []
    errors = []

    if is_external:
        # Archive datasets ship train+test in one archive: download
        # once, then both splits are available regardless of `split`.
        print(f"  • Downloading archive from {dataset_info['download']['url']} ...")
        try:
            for split_name in ('test', 'train'):
                ds = _resolve_external_dataset(
                    dataset_info, raw_dir,
                    train=(split_name == 'train'),
                    download=True,
                )
                downloaded_splits.append(split_name)
                print(f"    ✓ {split_name.capitalize()} split: {len(ds)} samples")
        except Exception as e:
            error = f"External archive download failed: {e}"
            errors.append(error)
            print("    ⚠ " + error)
    else:
        import torchvision.datasets
        dataset_class = getattr(torchvision.datasets, dataset_name, None)

        if dataset_class is None:
            error = f"Dataset {dataset_name} not found in torchvision.datasets"
            errors.append(error)
            print("    ⚠ " + error)
            
            return [], errors

        if split in ['test', 'both']:
            print(f"  • Downloading test split...")
            try:
                test_dataset = dataset_class(
                    root=str(raw_dir),
                    train=False,
                    download=True
                )
                downloaded_splits.append('test')
                print(f"    ✓ Test split: {len(test_dataset)} samples")
            except Exception as e:
                error = f"Test split failed: {e}"
                errors.append(error)
                print("    ⚠ " + error)

        if split in ['train', 'both']:
            print(f"  • Downloading train split...")
            try:
                train_dataset = dataset_class(
                    root=str(raw_dir),
                    train=True,
                    download=True
                )
                downloaded_splits.append('train')
                print(f"    ✓ Train split: {len(train_dataset)} samples")
            except Exception as e:
                error = f"Train split failed: {e}"
                errors.append(error)
                print("    ⚠ " + error)
    
    return downloaded_splits, errors


def download_dataset_model_pair(
    dataset_name: str,
    model_name: str,
    root_dir: Optional[str] = None,
    split: str = "test"
) -> dict:
    """
    Download a dataset-model pair to the specified directory.
    Always replaces existing files.
    
    Validates the dataset-model pair by running inference test before downloading.
    Raises AssertionError if the pair fails validation (not testable, incompatible, or inference fails).
    
    Args:
        dataset_name: Name of the torchvision dataset (case-insensitive)
        model_name: Name of the model (case-insensitive)
        root_dir: Root directory for downloads (default: from path_config.get_torchvision_data_root())
        split: Dataset split to download ('train', 'test', or 'both')
        
    Returns:
        Dictionary with download information:
        - dataset_path: Path to downloaded dataset
        - model_path: Path to saved model architecture
        - info_path: Path to metadata JSON file
        - status: 'success' or 'error'
        - message: Status message
        
    Raises:
        AssertionError: If the dataset-model pair fails validation tests
        
    Example:
        >>> result = download_dataset_model_pair("MNIST", "simple_cnn", split="test")
        >>> print(result['dataset_path'])
        '/path/to/ACT/data/torchvision/MNIST/raw/'
    """
    import shutil
    import traceback
    
    # Use path from config if not specified
    if root_dir is None:
        root_dir = get_torchvision_data_root()
    
    try:
        # Normalize names (case-insensitive)
        dataset_name = find_dataset_name(dataset_name)
        model_name = find_model_name(model_name)
        
        # VALIDATION STEP: Test the dataset-model pair before downloading
        print(f"\n{'='*80}")
        print(f"PRE-DOWNLOAD VALIDATION: {dataset_name} + {model_name}")
        print(f"{'='*80}")
        
        from act.front_end.torchvision_loader.cli import _test_single_dataset_model
        import torchvision.models
        
        # Get dataset info to check if it's a classification dataset
        dataset_info = get_dataset_info(dataset_name)
        is_classification = dataset_info['category'] == 'classification'
        
        # Check if model is standard or custom
        is_standard_model = hasattr(torchvision.models, model_name)
        model_type = "standard (torchvision.models)" if is_standard_model else "custom (model_definitions.py)"
        print(f"Model Type: {model_type}")
        
        # Run validation test (with inference for classification datasets)
        model_cache = {}
        is_testable, is_compatible, inference_result = _test_single_dataset_model(
            dataset_name=dataset_name,
            model_name=model_name,
            run_inference=is_classification,  # Only test inference for classification
            model_cache=model_cache
        )
        
        print(f"Validation Results:")
        print(f"  • Model Loadable: {is_testable}")
        print(f"  • Dataset-Model Compatible: {is_compatible}")
        if is_classification and inference_result is not None:
            print(f"  • Inference Test: {'✓ PASSED' if inference_result else '✗ FAILED'}")
        
        # Testability covers standard torchvision.models and any custom
        # models registered in model_definitions.get_model().
        if not is_testable:
            error_msg = (
                f"❌ VALIDATION FAILED: Model '{model_name}' cannot be loaded.\n"
                f"   • Standard models must exist in torchvision.models\n"
                f"   • Custom models must be defined in model_definitions.py with get_model() support"
            )
            print(f"\n{error_msg}")
            assert False, error_msg
        
        # Check compatibility
        if not is_compatible:
            error_msg = (
                f"❌ VALIDATION FAILED: Dataset '{dataset_name}' is incompatible with model '{model_name}'.\n"
                f"   The pair failed compatibility validation checks."
            )
            print(f"\n{error_msg}")
            assert False, error_msg
        
        # Check inference result for classification datasets
        if is_classification:
            print(f"  • Inference test: {inference_result}")
            
            if inference_result is False:
                error_msg = (
                    f"❌ VALIDATION FAILED: Inference test failed for {dataset_name} + {model_name}.\n"
                    f"   The model could not process preprocessed data from this dataset."
                )
                print(f"\n{error_msg}")
                assert False, error_msg
            elif inference_result is True:
                print(f"\n✓ Validation passed! Proceeding with download...")
        else:
            print(f"  • Inference test: N/A (non-classification dataset)")
            print(f"\n✓ Validation passed! Proceeding with download...")
        
        print(f"{'='*80}\n")
        
        # Validate compatibility
        validation = validate_dataset_model_compatibility(dataset_name, model_name)
        if not validation['compatible']:
            return {
                'status': 'error',
                'message': f"Incompatible pair: {', '.join(validation['issues'])}"
            }
        
        # Get dataset info
        dataset_info = get_dataset_info(dataset_name)
        
        # Create directory structure
        dataset_dir = Path(root_dir) / dataset_name
        raw_dir = dataset_dir / "raw"
        models_dir = dataset_dir / "models"
        
        # Check if dataset already exists (for incremental download)
        dataset_exists = dataset_dir.exists() and raw_dir.exists()
        model_file_path = models_dir / f"{model_name}.py"
        model_already_exists = model_file_path.exists()
        
        if dataset_exists:
            if model_already_exists:
                print(f"\n⚠️  Model '{model_name}' already exists for dataset '{dataset_name}'")
                print(f"   Skipping download (pair already complete)")
                return {
                    'status': 'success',
                    'message': f"Model {model_name} already exists for {dataset_name}",
                    'path': str(dataset_dir)
                }
            else:
                print(f"\n✓ Dataset '{dataset_name}' already downloaded - adding model '{model_name}'")
                print(f"  Location: {raw_dir}")
                
                # Load existing metadata to check splits
                info_path = dataset_dir / "info.json"
                if info_path.exists():
                    with open(info_path, 'r') as f:
                        existing_metadata = json.load(f)
                    downloaded_splits = existing_metadata.get('splits_downloaded', [])
                    print(f"  Existing splits: {', '.join(downloaded_splits)}")
                else:
                    downloaded_splits = []
                
                # Create models directory if it doesn't exist
                models_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Fresh download - create directories
            raw_dir.mkdir(parents=True, exist_ok=True)
            models_dir.mkdir(parents=True, exist_ok=True)
            downloaded_splits = []
        
        print(f"\n{'='*80}")
        print(f"{'ADDING MODEL' if dataset_exists else 'DOWNLOADING'}: {dataset_name} + {model_name}")
        print(f"{'='*80}")
        print(f"Target: {dataset_dir}")
        if not dataset_exists:
            print(f"Split: {split}")
        
        # Download dataset (only if not already present)
        if not dataset_exists:
            print(f"\n[1/3] Downloading dataset...")

            downloaded_splits, errors = _download_dataset(root_dir, dataset_name, dataset_info, split)

            if not downloaded_splits:
                return {
                    'status': 'error',
                    'message': f"Failed to download any splits for {dataset_name}, with errors: {"\n".join(errors)}"
                }
        else:
            print(f"\n[1/3] Dataset already present, skipping download...")
        
        # Save model architecture
        print(f"\n[2/3] Saving model architecture...")
        model_path = models_dir / f"{model_name}.py"
        
        import torchvision.models
        if hasattr(torchvision.models, model_name):
            # Standard TorchVision model
            model_code = f"""
# Standard TorchVision Model: {model_name}
# Can be loaded with: torchvision.models.{model_name}()

import torch
import torchvision.models as models

# Load model architecture with pre-trained weights
model = models.{model_name}(weights="DEFAULT")

# Modify final layer for {dataset_info['num_classes']} classes if needed
# Example for ResNet:
# if hasattr(model, 'fc'):
#     in_features = model.fc.in_features
#     model.fc = torch.nn.Linear(in_features, {dataset_info['num_classes']})

print(f"Model: {model_name}")
print(f"Parameters: {{sum(p.numel() for p in model.parameters()):,}}")
"""
        else:
            # Custom model - get definition
            model_code = _get_custom_model_definition(model_name, dataset_info['num_classes'])
        
        with open(model_path, 'w') as f:
            f.write(model_code)
        print(f"  ✓ Model architecture saved: {model_path}")
        
        # Save/update metadata
        print(f"\n[3/3] Saving metadata...")
        info_path = dataset_dir / "info.json"
        
        # Load existing metadata if it exists
        if info_path.exists():
            try:
                with open(info_path, 'r') as f:
                    existing_metadata = json.load(f)
                
                # Check if this is old format (single model) or new format (list of models)
                if 'models' not in existing_metadata:
                    # Old format - convert to new format
                    old_model = existing_metadata.get('model')
                    models_list = [old_model] if old_model else []
                else:
                    models_list = existing_metadata.get('models', [])
                
                # Add new model if not already in list
                if model_name not in models_list:
                    models_list.append(model_name)
                
                # Preserve existing splits if incremental
                if dataset_exists:
                    downloaded_splits = existing_metadata.get('splits_downloaded', downloaded_splits)
            except Exception as e:
                print(f"  ⚠ Could not read existing metadata: {e}")
                models_list = [model_name]
        else:
            models_list = [model_name]
        
        # Create updated metadata with list of models
        metadata = {
            'dataset': dataset_name,
            'models': models_list,  # Changed from single 'model' to list 'models'
            'category': dataset_info['category'],
            'input_size': dataset_info['input_size'],
            'num_classes': dataset_info['num_classes'],
            'splits_downloaded': downloaded_splits,
            'preprocessing_required': validation['preprocessing_required'],
            'preprocessing_steps': validation['preprocessing_steps'],
            'notes': dataset_info['notes'],
            'paths': {
                'raw_data': str(raw_dir),
                'models_directory': str(models_dir),
                'metadata': str(info_path)
            }
        }
        
        with open(info_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Metadata saved: {info_path}")
        print(f"  ✓ Models registered: {', '.join(models_list)}")
        
        # Calculate directory size
        total_size_bytes = _get_directory_size(dataset_dir)
        total_size_formatted = _format_size(total_size_bytes)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"✓ DOWNLOAD COMPLETE")
        print(f"{'='*80}")
        print(f"Dataset: {raw_dir}")
        print(f"Model: {model_path}")
        print(f"Info: {info_path}")
        print(f"Size: {total_size_formatted} ({total_size_bytes:,} bytes)")
        
        if validation['preprocessing_required']:
            print(f"\n⚠ Preprocessing Required:")
            for step in validation['preprocessing_steps']:
                print(f"  • {step}")
        
        return {
            'status': 'success',
            'dataset_path': str(raw_dir),
            'model_path': str(model_path),
            'info_path': str(info_path),
            'message': 'Download completed successfully',
            'metadata': metadata,
            'size_bytes': total_size_bytes,
            'size_formatted': total_size_formatted
        }
        
    except AssertionError:
        # Re-raise AssertionError from validation without catching
        raise
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Download failed: {str(e)}",
            'traceback': traceback.format_exc()
        }


def _get_directory_size(path) -> int:
    """
    Calculate total size of a directory in bytes.
    
    Args:
        path: Path to directory
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    try:
        for item in Path(path).rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
    except Exception as e:
        pass  # Silently ignore permission errors
    return total_size


def _format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB", "234 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def list_downloaded_pairs(root_dir: Optional[str] = None) -> List[dict]:
    """
    List all downloaded dataset-model pairs with size information.

    Reads ``info.json`` per dataset; each file stores a list of models
    sharing that dataset.

    Args:
        root_dir: Root directory for downloads (default: from path_config.get_torchvision_data_root())
        
    Returns:
        List of dictionaries with information about each downloaded pair,
        including 'size_bytes' and 'size_formatted' fields
    """
    # Use path from config if not specified
    if root_dir is None:
        root_dir = get_torchvision_data_root()
    
    root_path = Path(root_dir)
    if not root_path.exists():
        return []
    
    downloaded = []
    
    for dataset_dir in root_path.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        info_path = dataset_dir / "info.json"
        if not info_path.exists():
            continue
        
        try:
            with open(info_path, 'r') as f:
                metadata = json.load(f)
            
            # Calculate directory size (once per dataset)
            size_bytes = _get_directory_size(dataset_dir)
            size_formatted = _format_size(size_bytes)
            
            # info.json may store either a list under 'models' or a single
            # 'model' string; normalise both into models_list.
            if 'models' in metadata:
                models_list = metadata['models']
            else:
                model = metadata.get('model')
                models_list = [model] if model else []
            
            # Create one entry per model
            for model_name in models_list:
                pair_metadata = {
                    'dataset': metadata['dataset'],
                    'model': model_name,
                    'category': metadata.get('category', 'unknown'),
                    'input_size': metadata.get('input_size', []),
                    'num_classes': metadata.get('num_classes', 0),
                    'splits_downloaded': metadata.get('splits_downloaded', []),
                    'preprocessing_required': metadata.get('preprocessing_required', True),
                    'preprocessing_steps': metadata.get('preprocessing_steps', []),
                    'notes': metadata.get('notes', ''),
                    'paths': {
                        'raw_data': metadata['paths'].get('raw_data', ''),
                        'model_architecture': str(dataset_dir / "models" / f"{model_name}.py"),
                        'metadata': str(info_path)
                    },
                    'size_bytes': size_bytes,
                    'size_formatted': size_formatted
                }
                downloaded.append(pair_metadata)
                
        except Exception as e:
            print(f"Warning: Could not read {info_path}: {e}")
    
    return downloaded


def load_dataset_model_pair(
    dataset_name: str,
    model_name: str,
    root_dir: Optional[str] = None,
    split: str = "test",
    batch_size: int = 1,
    shuffle: bool = False,
    auto_download: bool = True
) -> dict:
    """
    Load a previously downloaded dataset-model pair.
    If not found and auto_download=True, downloads it first.
    
    Args:
        dataset_name: Name of the dataset (case-insensitive)
        model_name: Name of the model (case-insensitive)
        root_dir: Root directory where datasets are stored (default: from path_config.get_torchvision_data_root())
        split: Which split to load ('train' or 'test')
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        auto_download: If True and pair not found, automatically download it
        
    Returns:
        Dictionary containing:
        - dataset: torch.utils.data.Dataset object
        - dataloader: torch.utils.data.DataLoader object
        - model: torch.nn.Module object
        - metadata: Dictionary with dataset/model information
        - preprocessing: Transform pipeline used
        
    Example:
        >>> result = load_dataset_model_pair("MNIST", "simple_cnn")
        >>> model = result['model']
        >>> dataloader = result['dataloader']
        >>> for images, labels in dataloader:
        ...     outputs = model(images)
    """
    import importlib.util
    
    # Use path from config if not specified
    if root_dir is None:
        root_dir = get_torchvision_data_root()
    
    # Normalize names (case-insensitive)
    dataset_name = find_dataset_name(dataset_name)
    model_name = find_model_name(model_name)
    
    # Check if pair exists
    dataset_dir = Path(root_dir) / dataset_name
    info_path = dataset_dir / "info.json"
    
    # Auto-download if not found
    if not dataset_dir.exists() or not info_path.exists():
        if auto_download:
            print(f"\n{'='*80}")
            print(f"Dataset-model pair not found locally. Downloading...")
            print(f"{'='*80}\n")
            
            # Determine which split to download
            download_split = split if split in ['train', 'test'] else 'both'
            
            # Download the pair
            download_result = download_dataset_model_pair(
                dataset_name=dataset_name,
                model_name=model_name,
                root_dir=root_dir,
                split=download_split
            )
            
            if download_result['status'] != 'success':
                raise RuntimeError(
                    f"Failed to download dataset-model pair: {download_result['message']}"
                )
            
            print(f"\n{'='*80}")
            print(f"Download completed. Proceeding to load...")
            print(f"{'='*80}\n")
        else:
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_dir}\n"
                f"Use --download {dataset_name} {model_name} to download first, "
                f"or set auto_download=True."
            )
    
    if not info_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {info_path}\n"
            f"The dataset directory may be incomplete."
        )
    
    # Load metadata
    with open(info_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"LOADING: {dataset_name} + {model_name}")
    print(f"{'='*80}")
    
    # Check if requested split was downloaded
    if split not in metadata['splits_downloaded']:
        available = ', '.join(metadata['splits_downloaded'])
        raise ValueError(
            f"Split '{split}' not available. Downloaded splits: {available}\n"
            f"Use --download {dataset_name} {model_name} --split {split} to download."
        )
    
    # Load dataset with preprocessing
    print(f"[1/3] Loading dataset ({split} split)...")
    raw_dir = dataset_dir / "raw"
    
    # Create preprocessing pipeline (uniform for all models)
    preprocessing = create_preprocessing_pipeline(dataset_name)
    
    # Load dataset
    dataset_info = get_dataset_info(dataset_name)
    is_train = (split == "train")

    try:
        if "download" in dataset_info:
            dataset = _resolve_external_dataset(
                dataset_info, raw_dir,
                train=is_train,
                transform=preprocessing,
                download=False,  # Already downloaded
            )
        else:
            import torchvision.datasets
            dataset_class = getattr(torchvision.datasets, dataset_name)
            dataset = dataset_class(
                root=str(raw_dir),
                train=is_train,
                transform=preprocessing,
                download=False  # Already downloaded
            )
        print(f"  ✓ Loaded {len(dataset)} samples")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    
    # Load model
    print(f"[2/3] Loading model architecture...")
    model_path = dataset_dir / "models" / f"{model_name}.py"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Import torchvision.models explicitly to avoid circular import issues
    import torchvision.models
    
    # Load model based on type
    if hasattr(torchvision.models, model_name):
        # Standard TorchVision model
        model_fn = getattr(torchvision.models, model_name)
        model = model_fn(weights="DEFAULT")
        print(f"  ✓ Loaded {model_name} with pre-trained weights")
        
        # Adjust final layer for number of classes if needed
        num_classes = metadata['num_classes']
        if hasattr(model, 'fc'):
            in_features = model.fc.in_features
            if model.fc.out_features != num_classes:
                model.fc = torch.nn.Linear(in_features, num_classes)
                print(f"  ✓ Adjusted final layer: {in_features} → {num_classes} classes")
        elif hasattr(model, 'classifier'):
            classifier = model.classifier
            named_kids = list(classifier.named_children())
            if named_kids:
                last_name, last_child = named_kids[-1]
                in_features = last_child.in_features
                if last_child.out_features != num_classes:
                    setattr(classifier, last_name, torch.nn.Linear(in_features, num_classes))
                    print(f"  ✓ Adjusted classifier: {in_features} → {num_classes} classes")
            else:
                in_features = classifier.in_features
                if classifier.out_features != num_classes:
                    model.classifier = torch.nn.Linear(in_features, num_classes)
                    print(f"  ✓ Adjusted classifier: {in_features} → {num_classes} classes")
        
        print(f"  ✓ Loaded {model_name} from torchvision.models")
    else:
        # Custom model - need to execute the .py file
        spec = importlib.util.spec_from_file_location("custom_model", model_path)
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        model = custom_module.model
        print(f"  ✓ Loaded custom model from {model_path.name}")
    
    model.eval()  # Set to evaluation mode by default
    
    # Print summary
    print(f"[3/3] Summary...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Dataset: {len(dataset)} samples ({split} split)")
    print(f"  Model: {total_params:,} parameters ({trainable_params:,} trainable)")
    print(f"  Batch size: {batch_size}")
    print(f"  Preprocessing: {'Yes' if metadata['preprocessing_required'] else 'No'}")
    
    print(f"\n{'='*80}")
    print(f"✓ LOADED SUCCESSFULLY")
    print(f"{'='*80}")
    
    return {
        'dataset': dataset,
        'dataloader': dataloader,
        'model': model,
        'metadata': metadata,
        'preprocessing': preprocessing,
        'num_samples': len(dataset),
        'num_classes': metadata['num_classes'],
        'split': split
    }


def model_inference_with_dataset(
    dataset_name: str,
    model_name: str,
    split: str = "test",
    root_dir: Optional[str] = None,
    verbose: bool = True
) -> dict:
    """
    Test inference for a single dataset-model pair.
    
    Args:
        dataset_name: Name of the dataset (case-insensitive)
        model_name: Name of the model (case-insensitive)
        split: Dataset split to test ('train' or 'test')
        root_dir: Root directory where datasets are stored
        verbose: Whether to print detailed progress information
        
    Returns:
        Dictionary with test results:
        - status: 'success', 'failed', or 'error'
        - output_shape: Tuple of output shape (if successful)
        - error: Error message (if failed/error)
        - num_samples: Number of samples in dataset
        - dataset: Dataset name
        - model: Model name
        - split: Split tested
        
    Example:
        >>> result = model_inference_with_dataset("MNIST", "resnet18")
        >>> print(result['status'])
        'success'
    """
    from act.util.model_inference import infer_single_model
    
    # Normalize names
    dataset_name = find_dataset_name(dataset_name)
    model_name = find_model_name(model_name)
    
    combo_id = f"m:{model_name}|x:{dataset_name}|split:{split}"
    
    try:
        # Load the pair
        if verbose:
            print(f"Testing: {dataset_name} + {model_name} ({split} split)")
        
        result = load_dataset_model_pair(
            dataset_name=dataset_name,
            model_name=model_name,
            split=split,
            batch_size=1,
            shuffle=False,
            auto_download=True,
            root_dir=root_dir
        )
        
        model = result['model']
        dataloader = result['dataloader']
        num_samples = result['num_samples']
        
        # Get first sample to test
        input_tensor = None
        try:
            for batch in dataloader:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    # Standard (image, label) tuple
                    input_tensor, labels = batch[0], batch[1]
                elif isinstance(batch, (tuple, list)) and len(batch) == 1:
                    input_tensor = batch[0]
                else:
                    input_tensor = batch
                break  # Only need first batch
        except Exception as e:
            error_msg = f'Batch loading error: {str(e)[:50]}'
            if verbose:
                print(f"  ✗ FAILED - {error_msg}")
            return {
                'dataset': dataset_name,
                'model': model_name,
                'split': split,
                'status': 'failed',
                'error': error_msg,
                'num_samples': num_samples
            }
        
        if input_tensor is None:
            error_msg = 'Empty dataloader'
            if verbose:
                print(f"  ✗ FAILED - {error_msg}")
            return {
                'dataset': dataset_name,
                'model': model_name,
                'split': split,
                'status': 'failed',
                'error': error_msg,
                'num_samples': num_samples
            }
        
        # Verify we have a tensor
        if not isinstance(input_tensor, torch.Tensor):
            error_msg = f'Input type {type(input_tensor).__name__}, expected torch.Tensor'
            if verbose:
                print(f"  ✗ FAILED - {error_msg}")
            return {
                'dataset': dataset_name,
                'model': model_name,
                'split': split,
                'status': 'failed',
                'error': error_msg,
                'num_samples': num_samples
            }
        
        # Run inference
        success, output, error_msg = infer_single_model(combo_id, model, input_tensor)
        
        if success:
            output_shape = tuple(output.shape) if output is not None else None
            if verbose:
                print(f"  ✓ SUCCESS - Output shape: {output_shape}")
            return {
                'dataset': dataset_name,
                'model': model_name,
                'split': split,
                'status': 'success',
                'output_shape': output_shape,
                'num_samples': num_samples
            }
        else:
            if verbose:
                print(f"  ✗ FAILED - {error_msg}")
            return {
                'dataset': dataset_name,
                'model': model_name,
                'split': split,
                'status': 'failed',
                'error': error_msg,
                'num_samples': num_samples
            }
            
    except Exception as e:
        error_msg = str(e)[:100]
        if verbose:
            print(f"  ✗ ERROR - {error_msg}")
        return {
            'dataset': dataset_name,
            'model': model_name,
            'split': split,
            'status': 'error',
            'error': error_msg
        }


def main():
    """
    Main function to test all downloaded dataset-model pairs with inference.
    
    Loads all pairs from data/torchvision/, performs inference on each sample,
    and reports success/failure statistics.
    """
    print("="*80)
    print("DATASET-MODEL PAIR INFERENCE TESTING")
    print("="*80)
    
    # Get all downloaded pairs
    downloaded_pairs = list_downloaded_pairs()
    
    if not downloaded_pairs:
        print("\n⚠️  No downloaded dataset-model pairs found.")
        print("   Use --download to download some pairs first:")
        print("   python -m act.front_end.torchvision_loader.cli --download MNIST resnet18")
        return
    
    print(f"\n📊 Found {len(downloaded_pairs)} downloaded pairs")
    print(f"{'='*80}\n")
    
    # Track statistics
    results = []
    
    # Test each pair
    for pair_info in downloaded_pairs:
        dataset_name = pair_info['dataset']
        model_name = pair_info['model']
        splits = pair_info['splits_downloaded']
        
        print(f"Testing: {dataset_name} + {model_name}")
        print(f"  Splits: {', '.join(splits)}")
        
        for split in splits:
            # Test the pair
            result = model_inference_with_dataset(
                dataset_name=dataset_name,
                model_name=model_name,
                split=split,
                verbose=True
            )
            results.append(result)
        
        print()  # Blank line between pairs
    
    # Calculate statistics
    total_pairs = len(results)
    successful_pairs = sum(1 for r in results if r['status'] == 'success')
    failed_pairs = total_pairs - successful_pairs
    
    # Print summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total pairs tested: {total_pairs}")
    print(f"Successful: {successful_pairs}")
    print(f"Failed: {failed_pairs}")
    
    if total_pairs > 0:
        success_rate = (successful_pairs / total_pairs) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    # Show successful pairs
    if successful_pairs > 0:
        print(f"\n✓ Successful Pairs ({successful_pairs}):")
        for r in results:
            if r['status'] == 'success':
                print(f"   • {r['dataset']} + {r['model']} ({r['split']}): {r['output_shape']}")
    
    # Show failed pairs with reasons
    if failed_pairs > 0:
        print(f"\n✗ Failed Pairs ({failed_pairs}):")
        failure_reasons = {}
        for r in results:
            if r['status'] in ('failed', 'error'):
                pair_key = f"{r['dataset']} + {r['model']}"
                if pair_key not in failure_reasons:
                    failure_reasons[pair_key] = r.get('error', 'Unknown error')[:80]
        
        for pair_key, reason in failure_reasons.items():
            print(f"   • {pair_key}: {reason}")
    
    print("="*80)


if __name__ == "__main__":
    main()
