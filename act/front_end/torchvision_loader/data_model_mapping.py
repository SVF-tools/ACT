#!/usr/bin/env python3
"""
Dataset-Model Mapping for TorchVision Datasets.

This module provides comprehensive mappings between torchvision datasets
and appropriate PyTorch models, including recommended architectures,
input sizes, number of classes, and usage notes.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from typing import Dict, Any, List, Optional


# Comprehensive torchvision dataset to model mapping
DATASET_MODEL_MAPPING: Dict[str, Dict[str, Any]] = {
    # ========== MNIST Family (28x28 grayscale) ==========
    "MNIST": {
        "models": ["simple_cnn", "lenet5", "resnet18", "efficientnet_b0"],
        "input_size": (1, 28, 28),
        "num_classes": 10,
        "category": "classification",
        "preprocessing": {
            "grayscale_to_rgb": True,  # Repeat channels: (1,28,28) → (3,28,28)
            "resize_to": (224, 224),    # Then resize to model input size
            "normalize": {"mean": [0.1307]*3, "std": [0.3081]*3}
        },
        "notes": "Small resolution grayscale. Use simple_cnn/lenet5 for native 1-channel, or convert to RGB for pre-trained models."
    },
    "FashionMNIST": {
        "models": ["simple_cnn", "lenet5", "resnet18", "efficientnet_b0"],
        "input_size": (1, 28, 28),
        "num_classes": 10,
        "category": "classification",
        "preprocessing": {
            "grayscale_to_rgb": True,
            "resize_to": (224, 224),
            "normalize": {"mean": [0.5]*3, "std": [0.5]*3}
        },
        "notes": "Clothing items. Same preprocessing as MNIST. Simple architectures recommended for grayscale."
    },
    "KMNIST": {
        "models": ["simple_cnn", "lenet5", "resnet18"],
        "input_size": (1, 28, 28),
        "num_classes": 10,
        "category": "classification",
        "preprocessing": {
            "grayscale_to_rgb": True,
            "resize_to": (224, 224),
            "normalize": {"mean": [0.5]*3, "std": [0.5]*3}
        },
        "notes": "Japanese Kuzushiji characters. MNIST-like preprocessing required for pre-trained models."
    },
    "EMNIST": {
        "models": ["simple_cnn", "resnet18"],
        "input_size": (1, 28, 28),
        "num_classes": 62,  # varies by split (balanced: 47, byclass: 62, etc.)
        "category": "classification",
        "preprocessing": {
            "grayscale_to_rgb": True,
            "resize_to": (224, 224),
            "normalize": {"mean": [0.5]*3, "std": [0.5]*3}
        },
        "notes": "Extended MNIST with letters/digits. Multiple splits available. Convert grayscale for ResNet."
    },
    "QMNIST": {
        "models": ["simple_cnn", "lenet5", "resnet18"],
        "input_size": (1, 28, 28),
        "num_classes": 10,
        "category": "classification",
        "preprocessing": {
            "grayscale_to_rgb": True,
            "resize_to": (224, 224),
            "normalize": {"mean": [0.1307]*3, "std": [0.3081]*3}
        },
        "notes": "MNIST reconstruction. Same preprocessing as original MNIST."
    },
    
    # ========== CIFAR Family (32x32 RGB) ==========
    "CIFAR10": {
        "models": ["resnet18", "resnet34", "resnet50", "vgg16", "mobilenet_v2", "efficientnet_b0"],
        "input_size": (3, 32, 32),
        "num_classes": 10,
        "category": "classification",
        "preprocessing": {
            "resize_to": (224, 224),  # Upsample to standard size
            "normalize": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]}
        },
        "notes": "Low resolution RGB. Resize to 224x224 for pre-trained models or train from scratch at 32x32."
    },
    "CIFAR100": {
        "models": ["resnet18", "resnet34", "resnet50", "vgg16", "mobilenet_v2", "efficientnet_b0"],
        "input_size": (3, 32, 32),
        "num_classes": 100,
        "category": "classification",
        "preprocessing": {
            "resize_to": (224, 224),
            "normalize": {"mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761]}
        },
        "notes": "100 fine-grained classes (20 superclasses). Same preprocessing as CIFAR10."
    },
    "STL10": {
        "models": ["resnet18", "resnet34", "resnet50", "mobilenet_v2", "efficientnet_b0"],
        "input_size": (3, 96, 96),
        "num_classes": 10,
        "category": "classification",
        "preprocessing": {
            "resize_to": (224, 224),  # Upsample from 96x96
            "normalize": {"mean": [0.4467, 0.4398, 0.4066], "std": [0.2603, 0.2566, 0.2713]}
        },
        "notes": "Higher resolution than CIFAR. Good for semi-supervised learning. Resize to 224x224."
    },
    "SVHN": {
        "models": ["resnet18", "resnet34", "vgg16", "mobilenet_v2", "efficientnet_b0"],
        "input_size": (3, 32, 32),
        "num_classes": 10,
        "category": "classification",
        "preprocessing": {
            "resize_to": (224, 224),
            "normalize": {"mean": [0.4377, 0.4438, 0.4728], "std": [0.1980, 0.2010, 0.1970]}
        },
        "notes": "Street View House Numbers. Digit recognition in natural images. Resize required."
    },
    
    # ========== Large-scale datasets (224x224 RGB standard) ==========
    "ImageNet": {
        "models": [
            # ResNet family
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
            # VGG family
            "vgg11", "vgg13", "vgg16", "vgg19",
            # EfficientNet family
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
            "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
            # ConvNeXt family
            "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
            # Vision Transformers
            "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32",
            # Swin Transformers
            "swin_t", "swin_s", "swin_b",
            # MobileNet family
            "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
            # DenseNet family
            "densenet121", "densenet161", "densenet169", "densenet201"
        ],
        "input_size": (3, 224, 224),
        "num_classes": 1000,
        "category": "classification",
        "preprocessing": {
            "resize_to": (224, 224),  # Already standard size
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        },
        "notes": "Standard benchmark. All models have pre-trained weights with ImageNet normalization. Use for transfer learning."
    },
    "Places365": {
        "models": ["resnet50", "resnet152", "vgg16", "densenet161"],
        "input_size": (3, 224, 224),
        "num_classes": 365,
        "category": "classification",
        "preprocessing": {
            "resize_to": (224, 224),
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        },
        "notes": "Scene recognition. Use ImageNet pre-trained weights then fine-tune with Places365 data."
    },
    
    # ========== Fine-grained classification (224x224 RGB) ==========
    "Caltech101": {
        "models": ["resnet50", "efficientnet_b0", "vit_b_16", "convnext_tiny"],
        "input_size": (3, 224, 224),
        "num_classes": 101,
        "category": "classification",
        "notes": "Object recognition, transfer learning from ImageNet strongly recommended"
    },
    "Caltech256": {
        "models": ["resnet50", "resnet101", "efficientnet_b0", "vit_b_16"],
        "input_size": (3, 224, 224),
        "num_classes": 257,
        "category": "classification",
        "notes": "Extended Caltech101, more challenging with more classes"
    },
    "Flowers102": {
        "models": ["resnet50", "efficientnet_b0", "vit_b_16", "convnext_tiny"],
        "input_size": (3, 224, 224),
        "num_classes": 102,
        "category": "classification",
        "notes": "Fine-grained flower species classification, 102 categories"
    },
    "Food101": {
        "models": ["resnet50", "resnet101", "efficientnet_b1", "vit_b_16"],
        "input_size": (3, 224, 224),
        "num_classes": 101,
        "category": "classification",
        "notes": "Food category classification, challenging with high intra-class variance"
    },
    "OxfordIIITPet": {
        "models": ["resnet50", "efficientnet_b0", "mobilenet_v2", "vit_b_16"],
        "input_size": (3, 224, 224),
        "num_classes": 37,
        "category": "classification",
        "notes": "Pet breed classification, 37 breeds (cats and dogs)"
    },
    "StanfordCars": {
        "models": ["resnet50", "resnet101", "efficientnet_b3", "vit_b_16", "convnext_small"],
        "input_size": (3, 224, 224),
        "num_classes": 196,
        "category": "classification",
        "notes": "Fine-grained car model recognition, requires attention to subtle details"
    },
    "FGVCAircraft": {
        "models": ["resnet50", "resnet101", "efficientnet_b3", "vit_b_16"],
        "input_size": (3, 224, 224),
        "num_classes": 100,
        "category": "classification",
        "notes": "Fine-grained aircraft variant recognition, 100 aircraft models"
    },
    "EuroSAT": {
        "models": ["resnet18", "resnet50", "efficientnet_b0", "vit_b_16"],
        "input_size": (3, 64, 64),
        "num_classes": 10,
        "category": "classification",
        "notes": "Satellite image classification, land use/cover classification"
    },
    "SUN397": {
        "models": ["resnet50", "resnet101", "vgg16", "densenet161"],
        "input_size": (3, 224, 224),
        "num_classes": 397,
        "category": "classification",
        "notes": "Scene understanding, 397 scene categories"
    },
    "Country211": {
        "models": ["resnet50", "efficientnet_b0", "vit_b_16"],
        "input_size": (3, 224, 224),
        "num_classes": 211,
        "category": "classification",
        "notes": "Geographic location recognition from images"
    },
    
    # ========== Object Detection ==========
    "CocoDetection": {
        "models": [
            "fasterrcnn_resnet50_fpn",
            "fasterrcnn_resnet50_fpn_v2",
            "fasterrcnn_mobilenet_v3_large_fpn",
            "fasterrcnn_mobilenet_v3_large_320_fpn",
            "fcos_resnet50_fpn",
            "retinanet_resnet50_fpn",
            "retinanet_resnet50_fpn_v2",
            "ssd300_vgg16",
            "ssdlite320_mobilenet_v3_large"
        ],
        "input_size": "variable",
        "num_classes": 80,
        "category": "detection",
        "notes": "COCO object detection, 80 object categories. Models maintain aspect ratio."
    },
    "VOCDetection": {
        "models": [
            "fasterrcnn_resnet50_fpn",
            "fasterrcnn_mobilenet_v3_large_fpn",
            "retinanet_resnet50_fpn",
            "ssd300_vgg16"
        ],
        "input_size": "variable",
        "num_classes": 20,
        "category": "detection",
        "notes": "PASCAL VOC detection, 20 object categories (person, vehicles, animals, household)"
    },
    "WIDERFace": {
        "models": ["fasterrcnn_resnet50_fpn", "retinanet_resnet50_fpn"],
        "input_size": "variable",
        "num_classes": 1,
        "category": "detection",
        "notes": "Face detection benchmark with diverse scales and occlusions"
    },
    
    # ========== Semantic Segmentation ==========
    "VOCSegmentation": {
        "models": [
            "fcn_resnet50",
            "fcn_resnet101",
            "deeplabv3_resnet50",
            "deeplabv3_resnet101",
            "deeplabv3_mobilenet_v3_large",
            "lraspp_mobilenet_v3_large"
        ],
        "input_size": "variable",
        "num_classes": 21,
        "category": "segmentation",
        "notes": "PASCAL VOC semantic segmentation, 20 object classes + background"
    },
    "Cityscapes": {
        "models": [
            "fcn_resnet50",
            "fcn_resnet101",
            "deeplabv3_resnet50",
            "deeplabv3_resnet101",
            "deeplabv3_mobilenet_v3_large"
        ],
        "input_size": (3, 1024, 2048),
        "num_classes": 19,
        "category": "segmentation",
        "notes": "Urban street scene segmentation for autonomous driving"
    },
    "SBDataset": {
        "models": ["fcn_resnet50", "deeplabv3_resnet50"],
        "input_size": "variable",
        "num_classes": 20,
        "category": "segmentation",
        "notes": "Semantic Boundaries Dataset, augments PASCAL VOC"
    },
    
    # ========== Face/Person datasets ==========
    "CelebA": {
        "models": ["resnet34", "resnet50", "mobilenet_v2", "efficientnet_b0"],
        "input_size": (3, 224, 224),
        "num_classes": 40,  # for attribute classification
        "category": "classification",
        "notes": "Celebrity faces with 40 binary attributes. Also used for face detection/recognition."
    },
    "LFWPeople": {
        "models": ["resnet34", "resnet50", "mobilenet_v2", "efficientnet_b0"],
        "input_size": (3, 224, 224),
        "num_classes": 5749,
        "category": "classification",
        "notes": "Labeled Faces in the Wild for face verification and recognition"
    },
    "LFWPairs": {
        "models": ["resnet34", "resnet50", "mobilenet_v2"],
        "input_size": (3, 224, 224),
        "num_classes": 2,  # same/different person
        "category": "classification",
        "notes": "Face verification pairs (same/different person)"
    },
    
    # ========== Video datasets ==========
    "Kinetics": {
        "models": [
            "r3d_18",
            "mc3_18",
            "r2plus1d_18",
            "s3d",
            "mvit_v1_b",
            "mvit_v2_s",
            "swin3d_t",
            "swin3d_s",
            "swin3d_b"
        ],
        "input_size": "T x H x W (video clips)",
        "num_classes": 400,  # Kinetics-400
        "category": "video",
        "notes": "Action recognition in videos, large-scale benchmark (400/600/700 classes)"
    },
    "HMDB51": {
        "models": ["r3d_18", "mc3_18", "r2plus1d_18"],
        "input_size": "T x H x W",
        "num_classes": 51,
        "category": "video",
        "notes": "Human motion recognition, 51 action categories"
    },
    "UCF101": {
        "models": ["r3d_18", "mc3_18", "r2plus1d_18", "s3d"],
        "input_size": "T x H x W",
        "num_classes": 101,
        "category": "video",
        "notes": "Action recognition, 101 action categories from YouTube"
    },
    
    # ========== Optical Flow ==========
    "FlyingChairs": {
        "models": ["raft_large", "raft_small"],
        "input_size": "variable",
        "num_classes": None,
        "category": "optical_flow",
        "notes": "Optical flow estimation, synthetic training data with ground truth"
    },
    "FlyingThings3D": {
        "models": ["raft_large", "raft_small"],
        "input_size": "variable",
        "num_classes": None,
        "category": "optical_flow",
        "notes": "Optical flow estimation, synthetic 3D scenes with complex motion"
    },
    "Sintel": {
        "models": ["raft_large", "raft_small"],
        "input_size": "variable",
        "num_classes": None,
        "category": "optical_flow",
        "notes": "Optical flow benchmark from animated movie, realistic motion blur"
    },
    "KittiFlow": {
        "models": ["raft_large", "raft_small"],
        "input_size": "variable",
        "num_classes": None,
        "category": "optical_flow",
        "notes": "Optical flow for autonomous driving, real-world scenarios"
    },
    
    # ========== Other specialized datasets ==========
    "Omniglot": {
        "models": ["simple_cnn", "resnet18"],
        "input_size": (1, 105, 105),
        "num_classes": 1623,
        "category": "classification",
        "preprocessing": {
            "grayscale_to_rgb": True,  # For resnet18
            "resize_to": (224, 224),
            "normalize": {"mean": [0.5]*3, "std": [0.5]*3}
        },
        "notes": "Few-shot learning benchmark. Handwritten characters from 50 alphabets. Convert grayscale for ResNet."
    },
    "PCAM": {
        "models": ["resnet18", "resnet50", "efficientnet_b0"],
        "input_size": (3, 96, 96),
        "num_classes": 2,
        "category": "classification",
        "notes": "Patch Camelyon, histopathologic cancer detection in tissue images"
    },
    "INaturalist": {
        "models": ["resnet50", "resnet101", "efficientnet_b3", "vit_b_16"],
        "input_size": (3, 224, 224),
        "num_classes": 8142,  # varies by year
        "category": "classification",
        "notes": "Species classification, highly imbalanced with long tail"
    },

    # ========== Datasets not in TorchVision ==========
    # Non-TorchVision datasets are configured declaratively via the "download"
    # key: an archive URL whose extracted "image_root" follows the ImageFolder
    # layout (one sub-directory per class). Optional "index_file"/"split_file"
    # (image_id -> relative path / image_id -> is_train) select the dataset's
    # official train/test split. No custom Dataset class is required.
    "CUB200": {
        "models": ["resnet18"],
        "input_size": (3, 224, 224),
        "num_classes": 200,
        "category": "classification",
        "preprocessing": {
            "resize_to": (224, 224)
        },
        "notes": "Caltech-UCSD Birds 200-2011; archive-distributed ImageFolder layout.",
        "download": {
            "url": "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz",
            "md5": "97eceeb196236b17998738112f37df78",
            "image_root": "CUB_200_2011/images",
            "index_file": "CUB_200_2011/images.txt",
            "split_file": "CUB_200_2011/train_test_split.txt"
        }
    }
}


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get complete information about a dataset including recommended models.
    
    Args:
        dataset_name: Name of the torchvision dataset
        
    Returns:
        Dictionary with keys: models, input_size, num_classes, category, notes
        
    Raises:
        ValueError: If dataset name is not found
    """
    if dataset_name not in DATASET_MODEL_MAPPING:
        available = list(DATASET_MODEL_MAPPING.keys())
        raise ValueError(
            f"Unknown dataset: {dataset_name}.\n"
            f"Available datasets: {', '.join(available)}"
        )
    
    return DATASET_MODEL_MAPPING[dataset_name].copy()


def list_datasets_by_category(category: str) -> List[str]:
    """
    List all datasets in a specific category.
    
    Args:
        category: One of 'classification', 'detection', 'segmentation', 'video', 'optical_flow'
        
    Returns:
        List of dataset names in the specified category
    """
    valid_categories = {'classification', 'detection', 'segmentation', 'video', 'optical_flow'}
    if category not in valid_categories:
        raise ValueError(f"Invalid category: {category}. Must be one of {valid_categories}")
    
    return [
        name for name, info in DATASET_MODEL_MAPPING.items()
        if info["category"] == category
    ]


def get_all_categories() -> List[str]:
    """
    Get list of all dataset categories.
    
    Returns:
        List of category names
    """
    categories = set(info["category"] for info in DATASET_MODEL_MAPPING.values())
    return sorted(categories)


def search_datasets(query: str) -> List[str]:
    """
    Search for datasets by name (case-insensitive substring match).
    
    Args:
        query: Search string
        
    Returns:
        List of matching dataset names
    """
    query_lower = query.lower()
    return [
        name for name in DATASET_MODEL_MAPPING.keys()
        if query_lower in name.lower()
    ]


def get_preprocessing_transforms(dataset_name: str):
    """
    Get the required preprocessing transforms for a dataset.
    
    Args:
        dataset_name: Name of the torchvision dataset
        
    Returns:
        Dictionary with preprocessing requirements or None if no special preprocessing needed
        
    Example:
        >>> get_preprocessing_transforms("MNIST")
        {
            'grayscale_to_rgb': True,
            'resize_to': (224, 224),
            'normalize': {'mean': [0.1307, 0.1307, 0.1307], 'std': [0.3081, 0.3081, 0.3081]}
        }
    """
    info = get_dataset_info(dataset_name)
    return info.get("preprocessing", None)


def create_preprocessing_pipeline(dataset_name: str):
    """
    Create a torchvision transforms pipeline for a dataset.
    
    Args:
        dataset_name: Name of the torchvision dataset
        
    Returns:
        torchvision.transforms.Compose object ready to use
        
    Example:
        >>> transform = create_preprocessing_pipeline("MNIST")
        >>> dataset = torchvision.datasets.MNIST(root='./data', transform=transform)
    """
    try:
        import torchvision.transforms as transforms
    except ImportError:
        raise ImportError("torchvision is required. Install with: pip install torchvision")
    
    info = get_dataset_info(dataset_name)
    preprocessing = info.get("preprocessing", {})
    
    transform_list = []
    
    # Handle resizing FIRST (works on PIL Images)
    if "resize_to" in preprocessing:
        resize_size = preprocessing["resize_to"]
        transform_list.append(transforms.Resize(resize_size))
    
    # Convert to tensor (PIL Image → Tensor)
    transform_list.append(transforms.ToTensor())
    
    # Handle grayscale to RGB conversion AFTER ToTensor (works on tensors)
    if preprocessing.get("grayscale_to_rgb", False):
        # Lambda to repeat grayscale channel to RGB
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))
    
    # Handle normalization (works on tensors)
    if "normalize" in preprocessing:
        norm_params = preprocessing["normalize"]
        mean = norm_params.get("mean", [0.5, 0.5, 0.5])
        std = norm_params.get("std", [0.5, 0.5, 0.5])
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def validate_dataset_model_compatibility(dataset_name: str, model_name: str) -> dict:
    """
    Validate if a dataset-model pair is compatible and return alignment info.
    
    Args:
        dataset_name: Name of the torchvision dataset
        model_name: Name of the PyTorch model
        
    Returns:
        Dictionary with:
        - compatible: bool
        - issues: list of compatibility issues
        - preprocessing_required: bool
        - preprocessing_steps: list of required preprocessing steps
        
    Example:
        >>> result = validate_dataset_model_compatibility("MNIST", "resnet18")
        >>> print(result['preprocessing_required'])  # True
        >>> print(result['preprocessing_steps'])  # ['grayscale_to_rgb', 'resize']
    """
    info = get_dataset_info(dataset_name)
    
    result = {
        "compatible": False,
        "issues": [],
        "preprocessing_required": False,
        "preprocessing_steps": []
    }
    
    # Check if model is in recommended list
    if model_name not in info["models"]:
        result["issues"].append(f"{model_name} not in recommended models for {dataset_name}")
        result["issues"].append(f"Recommended: {', '.join(info['models'][:5])}")
    
    # Check preprocessing requirements
    preprocessing = info.get("preprocessing", {})
    
    if preprocessing.get("grayscale_to_rgb", False):
        result["preprocessing_required"] = True
        result["preprocessing_steps"].append("grayscale_to_rgb (1 channel → 3 channels)")
    
    if "resize_to" in preprocessing:
        result["preprocessing_required"] = True
        target_size = preprocessing["resize_to"]
        current_size = info["input_size"]
        if isinstance(current_size, tuple):
            result["preprocessing_steps"].append(
                f"resize ({current_size[1]}x{current_size[2]} → {target_size[0]}x{target_size[1]})"
            )
    
    if "normalize" in preprocessing:
        result["preprocessing_required"] = True
        result["preprocessing_steps"].append("normalize (mean/std adjustment)")
    
    # If preprocessing handles all issues, mark as compatible
    if not result["issues"] or model_name in info["models"]:
        result["compatible"] = True
    
    return result


def find_dataset_name(user_input: str) -> str:
    """
    Find the correct dataset name using case-insensitive matching.
    
    Args:
        user_input: User-provided dataset name (any case)
        
    Returns:
        The correctly-cased dataset name from DATASET_MODEL_MAPPING
        
    Raises:
        ValueError: If dataset not found
    """
    for dataset_name in DATASET_MODEL_MAPPING.keys():
        if dataset_name.lower() == user_input.lower():
            return dataset_name
    raise ValueError(f"Dataset '{user_input}' not found. Use --list to see all available datasets.")


def find_model_name(user_input: str) -> str:
    """
    Find the correct model name using case-insensitive matching.
    Checks torchvision.models first, then all models in dataset mappings.
    
    Args:
        user_input: User-provided model name (any case)
        
    Returns:
        The correctly-cased model name
    """
    import torchvision
    import torchvision.models
    
    # Check exact match in torchvision first
    if hasattr(torchvision.models, user_input):
        return user_input
    
    # Try case-insensitive match in torchvision.models
    for attr in dir(torchvision.models):
        if attr.lower() == user_input.lower() and not attr.startswith('_'):
            if callable(getattr(torchvision.models, attr)):
                return attr
    
    # Check all models in dataset mappings (case-insensitive)
    all_models = set()
    for info in DATASET_MODEL_MAPPING.values():
        all_models.update(info['models'])
    
    for model in all_models:
        if model.lower() == user_input.lower():
            return model
    
    # Return user input as-is if not found
    return user_input
