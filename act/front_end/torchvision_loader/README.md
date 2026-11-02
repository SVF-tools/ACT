# TorchVision Dataset-Model Mapping

A comprehensive framework for downloading, managing, and loading TorchVision datasets with compatible PyTorch models for the ACT verification framework.

## Features

- **40+ Datasets**: Comprehensive mapping of TorchVision datasets across multiple categories
- **189 Dataset-Model Pairs**: Compatible dataset-model combinations
- **63 Unique Models**: Support for CNNs, ResNets, EfficientNets, ViTs, and more
- **Auto-Download**: Automatically download missing datasets when loading
- **Preprocessing**: Automatic preprocessing pipelines (resize, normalize, grayscale→RGB)
- **Path Management**: Centralized configuration via `act.util.path_config`
- **Modular Architecture**: Separated concerns (mapping, loading, CLI, models)

## Dataset Categories

- **Classification**: MNIST, CIFAR10/100, ImageNet, fine-grained datasets (27 datasets)
- **Detection**: COCO, PASCAL VOC, WIDERFace (3 datasets)
- **Segmentation**: Cityscapes, VOC, SBDataset (3 datasets)
- **Video**: Kinetics, HMDB51, UCF101 (3 datasets)
- **Optical Flow**: FlyingChairs, Sintel, KITTI (4 datasets)

## Module Structure

```
act/front_end/torchvision_loader/
├── __init__.py              # Package marker; import from the submodules
├── __main__.py              # Entry point for python -m execution
├── cli.py                   # Command-line interface and test functions
├── data_model_mapping.py    # Core dataset-model mapping and API
├── data_model_loader.py     # Download and load functionality
├── model_definitions.py     # Custom model architectures (SimpleCNN, LeNet5)
└── README.md                # This file
```

### File Descriptions

**`data_model_mapping.py` (684 lines)**
- Core dataset-model mapping dictionary (`DATASET_MODEL_MAPPING`)
- Dataset information retrieval (`get_dataset_info`, `list_datasets_by_category`)
- Compatibility validation (`validate_dataset_model_compatibility`)
- Preprocessing pipeline creation (`create_preprocessing_pipeline`)
- Search and query functions
- Case-insensitive name resolution

**`data_model_loader.py`**
- Download functionality with compatibility checks (`download_dataset_model_pair`)
- Load functionality with auto-download (`load_dataset_model_pair`)
- Downloaded pairs management (`list_downloaded_pairs`)
- Directory size calculation and formatting
- Metadata management

**`cli.py`**
- Command-line interface implementation
- Detailed output formatting and reporting
- All CLI commands (--summary, --download, --load, etc.)

**`model_definitions.py`**
- Custom model architectures (SimpleCNN, LeNet5)
- Model definition code generation
- Support for 1-channel grayscale input

**`__init__.py`**
- Package-level exports
- Unified import interface
- Version information

**`__main__.py`**
- Entry point for `python -m act.front_end.torchvision_loader`
- Delegates to CLI main function

## Installation

```bash
# Ensure you're in the ACT project root
cd /path/to/ACT

# Activate the ACT environment
conda activate act-py312

# All dependencies should already be installed via setup.sh
```

## Command-Line Usage

### Module Execution

The CLI can be executed as follows:

```bash
python -m act.front_end.torchvision_loader [OPTIONS]
```

Note: For common operations (list, search, download, info), prefer the unified CLI: `python -m act.front_end`.

### 1. Dataset Discovery

```bash
# Browse datasets by category (classification, detection, segmentation, video, optical_flow)
python -m act.front_end.torchvision_loader --category classification

# Show detailed information for a specific dataset
python -m act.front_end.torchvision_loader --dataset MNIST

# Show all recommended models for a specific dataset
python -m act.front_end.torchvision_loader --models-for MNIST

# Show all datasets compatible with a specific model
python -m act.front_end.torchvision_loader --datasets-for resnet18
```

### 2. Compatibility and Preprocessing

```bash
# Validate dataset-model compatibility
python -m act.front_end.torchvision_loader --validate MNIST resnet18

# Show preprocessing requirements for a specific dataset
python -m act.front_end.torchvision_loader --show-preprocessing MNIST

# Show aggregated preprocessing requirements across all datasets
python -m act.front_end.torchvision_loader --preprocessing-summary
```

### 3. Download Dataset-Model Pairs

**Basic Download (Test Split)**
```bash
# Download test split only (default)
python -m act.front_end.torchvision_loader --download MNIST resnet18
python -m act.front_end.torchvision_loader --download CIFAR10 resnet18
python -m act.front_end.torchvision_loader --download FashionMNIST resnet18

# Custom models registered in model_definitions.py are supported
python -m act.front_end.torchvision_loader --download MNIST simple_cnn
```

**Download Specific Split**
```bash
# Download train split only
python -m act.front_end.torchvision_loader --download MNIST resnet18 --split train

# Download test split only
python -m act.front_end.torchvision_loader --download MNIST resnet18 --split test

# Download both splits
python -m act.front_end.torchvision_loader --download MNIST resnet18 --split both
```

**Standard TorchVision Models**
```bash
# ResNet family
python -m act.front_end.torchvision_loader --download CIFAR10 resnet18
python -m act.front_end.torchvision_loader --download CIFAR10 resnet34
python -m act.front_end.torchvision_loader --download CIFAR10 resnet50
python -m act.front_end.torchvision_loader --download CIFAR100 resnet18
python -m act.front_end.torchvision_loader --download STL10 resnet18

# EfficientNet family
python -m act.front_end.torchvision_loader --download CIFAR10 efficientnet_b0
python -m act.front_end.torchvision_loader --download SVHN efficientnet_b0
python -m act.front_end.torchvision_loader --download Flowers102 efficientnet_b0

# VGG family
python -m act.front_end.torchvision_loader --download CIFAR10 vgg16
python -m act.front_end.torchvision_loader --download SVHN vgg16

# MobileNet family
python -m act.front_end.torchvision_loader --download CIFAR10 mobilenet_v2
python -m act.front_end.torchvision_loader --download STL10 mobilenet_v2
```

### 4. List Downloaded Pairs

```bash
# List all downloaded dataset-model pairs with sizes
python -m act.front_end.torchvision_loader --list-downloads
```

**Example Output:**
```
================================================================================
DOWNLOADED DATASET-MODEL PAIRS (3)
================================================================================

MNIST + resnet18
  Category: classification
  Classes: 10
  Splits: test
  Size: 63.5 MB
  Location: /path/to/ACT/data/torchvision/MNIST/raw
  Preprocessing: grayscale_to_rgb (1 channel → 3 channels), resize (28x28 → 224x224), normalize

FashionMNIST + resnet18
  Category: classification
  Classes: 10
  Splits: test
  Size: 81.9 MB
  Location: /path/to/ACT/data/torchvision/FashionMNIST/raw
  Preprocessing: grayscale_to_rgb (1 channel → 3 channels), resize (28x28 → 224x224), normalize

CIFAR10 + resnet18
  Category: classification
  Classes: 10
  Splits: test
  Size: 340.2 MB
  Location: /path/to/ACT/data/torchvision/CIFAR10/raw
  Preprocessing: resize (32x32 → 224x224), normalize

================================================================================
Total Size: 485.6 MB
================================================================================
```

### 5. Load Dataset-Model Pairs

```bash
# Load with default settings (auto-download if not found)
python -m act.front_end.torchvision_loader --load-torchvision MNIST resnet18

# Load with custom batch size
python -m act.front_end.torchvision_loader --load-torchvision CIFAR10 resnet18 --batch-size 64
```

### 6. Validation and Preprocessing

```bash
# Validate dataset-model compatibility
python -m act.front_end.torchvision_loader --validate MNIST resnet18
python -m act.front_end.torchvision_loader --validate CIFAR10 efficientnet_b0

# Show preprocessing requirements
python -m act.front_end.torchvision_loader --show-preprocessing MNIST
python -m act.front_end.torchvision_loader --show-preprocessing CIFAR10

# Show preprocessing summary for all datasets
python -m act.front_end.torchvision_loader --preprocessing-summary
```

## Quick Start Examples

### Example 1: Download and Load MNIST with ResNet18

```bash
# Download MNIST with ResNet18
python -m act.front_end.torchvision_loader --download MNIST resnet18 --split test

# Load it for use
python -m act.front_end.torchvision_loader --load-torchvision MNIST resnet18
```

### Example 2: Explore Compatible Models

```bash
# Find models for CIFAR10
python -m act.front_end.torchvision_loader --models-for CIFAR10

# Find datasets for resnet50
python -m act.front_end.torchvision_loader --datasets-for resnet50
```

## Recommended Dataset-Model Pairs

Common compatible pairs include:

### MNIST Family (28×28 grayscale)

```bash
# MNIST with all compatible models
python -m act.front_end.torchvision_loader --download MNIST simple_cnn
python -m act.front_end.torchvision_loader --download MNIST lenet5
python -m act.front_end.torchvision_loader --download MNIST resnet18
python -m act.front_end.torchvision_loader --download MNIST efficientnet_b0

# FashionMNIST with all compatible models
python -m act.front_end.torchvision_loader --download FashionMNIST simple_cnn
python -m act.front_end.torchvision_loader --download FashionMNIST lenet5
python -m act.front_end.torchvision_loader --download FashionMNIST resnet18
python -m act.front_end.torchvision_loader --download FashionMNIST efficientnet_b0

# KMNIST with all compatible models
python -m act.front_end.torchvision_loader --download KMNIST simple_cnn
python -m act.front_end.torchvision_loader --download KMNIST lenet5
python -m act.front_end.torchvision_loader --download KMNIST resnet18

# QMNIST with all compatible models
python -m act.front_end.torchvision_loader --download QMNIST simple_cnn
python -m act.front_end.torchvision_loader --download QMNIST lenet5
python -m act.front_end.torchvision_loader --download QMNIST resnet18

# EMNIST with all compatible models
python -m act.front_end.torchvision_loader --download EMNIST simple_cnn
python -m act.front_end.torchvision_loader --download EMNIST resnet18
```

### CIFAR Family (32×32 RGB)

```bash
# CIFAR10 with all compatible models
python -m act.front_end.torchvision_loader --download CIFAR10 resnet18
python -m act.front_end.torchvision_loader --download CIFAR10 resnet34
python -m act.front_end.torchvision_loader --download CIFAR10 resnet50
python -m act.front_end.torchvision_loader --download CIFAR10 vgg16
python -m act.front_end.torchvision_loader --download CIFAR10 mobilenet_v2
python -m act.front_end.torchvision_loader --download CIFAR10 efficientnet_b0

# CIFAR100 with all compatible models
python -m act.front_end.torchvision_loader --download CIFAR100 resnet18
python -m act.front_end.torchvision_loader --download CIFAR100 resnet34
python -m act.front_end.torchvision_loader --download CIFAR100 resnet50
python -m act.front_end.torchvision_loader --download CIFAR100 vgg16
python -m act.front_end.torchvision_loader --download CIFAR100 mobilenet_v2
python -m act.front_end.torchvision_loader --download CIFAR100 efficientnet_b0

# STL10 with all compatible models
python -m act.front_end.torchvision_loader --download STL10 resnet18
python -m act.front_end.torchvision_loader --download STL10 resnet34
python -m act.front_end.torchvision_loader --download STL10 resnet50
python -m act.front_end.torchvision_loader --download STL10 mobilenet_v2
python -m act.front_end.torchvision_loader --download STL10 efficientnet_b0

# SVHN with all compatible models
python -m act.front_end.torchvision_loader --download SVHN resnet18
python -m act.front_end.torchvision_loader --download SVHN resnet34
python -m act.front_end.torchvision_loader --download SVHN vgg16
python -m act.front_end.torchvision_loader --download SVHN mobilenet_v2
python -m act.front_end.torchvision_loader --download SVHN efficientnet_b0
```

### Fine-Grained Classification (224×224 RGB)

```bash
# Flowers102 with compatible models
python -m act.front_end.torchvision_loader --download Flowers102 resnet50
python -m act.front_end.torchvision_loader --download Flowers102 efficientnet_b0
python -m act.front_end.torchvision_loader --download Flowers102 vit_b_16
python -m act.front_end.torchvision_loader --download Flowers102 convnext_tiny

# Food101 with compatible models
python -m act.front_end.torchvision_loader --download Food101 resnet50
python -m act.front_end.torchvision_loader --download Food101 resnet101
python -m act.front_end.torchvision_loader --download Food101 efficientnet_b1
python -m act.front_end.torchvision_loader --download Food101 vit_b_16

# OxfordIIITPet with compatible models
python -m act.front_end.torchvision_loader --download OxfordIIITPet resnet50
python -m act.front_end.torchvision_loader --download OxfordIIITPet efficientnet_b0
python -m act.front_end.torchvision_loader --download OxfordIIITPet mobilenet_v2
python -m act.front_end.torchvision_loader --download OxfordIIITPet vit_b_16

# StanfordCars with compatible models
python -m act.front_end.torchvision_loader --download StanfordCars resnet50
python -m act.front_end.torchvision_loader --download StanfordCars resnet101
python -m act.front_end.torchvision_loader --download StanfordCars efficientnet_b3
python -m act.front_end.torchvision_loader --download StanfordCars vit_b_16
python -m act.front_end.torchvision_loader --download StanfordCars convnext_small

# Caltech101 with compatible models
python -m act.front_end.torchvision_loader --download Caltech101 resnet50
python -m act.front_end.torchvision_loader --download Caltech101 efficientnet_b0
python -m act.front_end.torchvision_loader --download Caltech101 vit_b_16
python -m act.front_end.torchvision_loader --download Caltech101 convnext_tiny

# Caltech256 with compatible models
python -m act.front_end.torchvision_loader --download Caltech256 resnet50
python -m act.front_end.torchvision_loader --download Caltech256 resnet101
python -m act.front_end.torchvision_loader --download Caltech256 efficientnet_b0
python -m act.front_end.torchvision_loader --download Caltech256 vit_b_16

# FGVCAircraft with compatible models
python -m act.front_end.torchvision_loader --download FGVCAircraft resnet50
python -m act.front_end.torchvision_loader --download FGVCAircraft resnet101
python -m act.front_end.torchvision_loader --download FGVCAircraft efficientnet_b3
python -m act.front_end.torchvision_loader --download FGVCAircraft vit_b_16

# SUN397 with compatible models
python -m act.front_end.torchvision_loader --download SUN397 resnet50
python -m act.front_end.torchvision_loader --download SUN397 resnet101
python -m act.front_end.torchvision_loader --download SUN397 vgg16
python -m act.front_end.torchvision_loader --download SUN397 densenet161

# Country211 with compatible models
python -m act.front_end.torchvision_loader --download Country211 resnet50
python -m act.front_end.torchvision_loader --download Country211 efficientnet_b0
python -m act.front_end.torchvision_loader --download Country211 vit_b_16
```

### Other Specialized Datasets

```bash
# Omniglot (few-shot learning)
python -m act.front_end.torchvision_loader --download Omniglot simple_cnn
python -m act.front_end.torchvision_loader --download Omniglot resnet18

# PCAM (medical imaging)
python -m act.front_end.torchvision_loader --download PCAM resnet18
python -m act.front_end.torchvision_loader --download PCAM resnet50
python -m act.front_end.torchvision_loader --download PCAM efficientnet_b0

# EuroSAT (satellite imagery)
python -m act.front_end.torchvision_loader --download EuroSAT resnet18
python -m act.front_end.torchvision_loader --download EuroSAT resnet50
python -m act.front_end.torchvision_loader --download EuroSAT efficientnet_b0
python -m act.front_end.torchvision_loader --download EuroSAT vit_b_16

# CelebA (face attributes)
python -m act.front_end.torchvision_loader --download CelebA resnet34
python -m act.front_end.torchvision_loader --download CelebA resnet50
python -m act.front_end.torchvision_loader --download CelebA mobilenet_v2
python -m act.front_end.torchvision_loader --download CelebA efficientnet_b0

# LFWPeople (face recognition)
python -m act.front_end.torchvision_loader --download LFWPeople resnet34
python -m act.front_end.torchvision_loader --download LFWPeople resnet50
python -m act.front_end.torchvision_loader --download LFWPeople mobilenet_v2
python -m act.front_end.torchvision_loader --download LFWPeople efficientnet_b0

# INaturalist (species classification)
python -m act.front_end.torchvision_loader --download INaturalist resnet50
python -m act.front_end.torchvision_loader --download INaturalist resnet101
python -m act.front_end.torchvision_loader --download INaturalist efficientnet_b3
python -m act.front_end.torchvision_loader --download INaturalist vit_b_16
```

## Python API Usage

### Imports

```python
from act.front_end.torchvision_loader.data_model_loader import (
    load_dataset_model_pair,
    download_dataset_model_pair
)
from act.front_end.torchvision_loader.data_model_mapping import (
    get_dataset_info,
    validate_dataset_model_compatibility,
    DATASET_MODEL_MAPPING
)
```

### Loading Datasets Programmatically

```python
from act.front_end.torchvision_loader.data_model_loader import load_dataset_model_pair

# Load a dataset-model pair (auto-downloads if not found)
result = load_dataset_model_pair(
    dataset_name='MNIST',
    model_name='resnet18',  # Must be in torchvision.models
    split='test',
    batch_size=32,
    auto_download=True  # Automatically download if not found
)

# Access components
dataset = result['dataset']
dataloader = result['dataloader']
model = result['model']
metadata = result['metadata']
preprocessing = result['preprocessing']

# Use in training/inference
for images, labels in dataloader:
    outputs = model(images)
    # ... your code here
```

### Downloading Programmatically

```python
from act.front_end.torchvision_loader.data_model_loader import download_dataset_model_pair

# Download a dataset-model pair
try:
    result = download_dataset_model_pair(
        dataset_name='CIFAR10',
        model_name='resnet18',  # Must exist in torchvision.models
        split='both'  # Download both train and test
    )
    
    if result['status'] == 'success':
        print(f"Downloaded to: {result['dataset_path']}")
        print(f"Size: {result['size_formatted']}")
```

### Testing Single Dataset-Model Pair

```python

# Test a specific pair before downloading
    dataset_name='MNIST',
    model_name='resnet18',
    run_inference=True,  # Run inference validation
    model_cache={}       # Cache loaded models
)

print(f"Testable (in torchvision.models): {is_testable}")
print(f"Compatible: {is_compatible}")
print(f"Inference passed: {inference_result}")
```

### Querying Dataset Information

```python
from act.front_end.torchvision_loader.data_model_mapping import (
    get_dataset_info,
    list_datasets_by_category,
    validate_dataset_model_compatibility
)

# Get dataset information
info = get_dataset_info('MNIST')
print(f"Models: {info['models']}")
print(f"Input size: {info['input_size']}")
print(f"Classes: {info['num_classes']}")

# List all classification datasets
classification_datasets = list_datasets_by_category('classification')
print(f"Found {len(classification_datasets)} classification datasets")

# Validate compatibility
validation = validate_dataset_model_compatibility('MNIST', 'resnet18')
print(f"Compatible: {validation['compatible']}")
print(f"Preprocessing required: {validation['preprocessing_required']}")
```

## Directory Structure

Downloaded pairs are stored in: `data/torchvision/<dataset>/`

```
data/torchvision/
├── MNIST/
│   ├── raw/                    # Downloaded dataset files
│   ├── models/
│   │   └── simple_cnn.py      # Model architecture definition
│   └── info.json              # Metadata (splits, preprocessing, etc.)
├── CIFAR10/
│   ├── raw/
│   ├── models/
│   │   └── resnet18.py
│   └── info.json
└── ...
```

## Configuration

The download directory is configured in `act/util/path_config.py`:

```python
from act.util.path_config import get_torchvision_data_root

# Default: /path/to/ACT/data/torchvision
root_dir = get_torchvision_data_root()
```

## Preprocessing Details

### Automatic Preprocessing

The framework automatically handles:

1. **Grayscale → RGB**: MNIST-family datasets (1 channel → 3 channels)
2. **Resize**: Upsampling/downsampling to model input size (typically 224×224)
3. **Normalization**: Dataset-specific mean/std normalization

### Dataset-Specific Preprocessing

- **MNIST**: Grayscale → RGB, resize to 224×224, normalize (mean=0.1307, std=0.3081)
- **CIFAR10**: Resize to 224×224, normalize (mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
- **ImageNet**: Resize to 224×224, normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Tips and Best Practices

1. **Start with test split**: Use `--split test` for faster downloads and testing
2. **Check compatibility**: Use `--validate` before downloading large datasets
3. **Monitor disk space**: Use `--list-downloads` to track total size
4. **Use auto-download**: `load_dataset_model_pair()` with `auto_download=True` for seamless workflow

## Troubleshooting

### Validation Failed: Model Cannot Be Loaded
```bash
# Error: Model not in torchvision.models or model_definitions.py
AssertionError: Model 'unknown_model' cannot be loaded.

# Solution: Use supported models
python -m act.front_end.torchvision_loader --download MNIST resnet18
python -m act.front_end.torchvision_loader --download FashionMNIST simple_cnn
```

### Dataset Not Found
```bash
# Check if dataset exists
python -m act.front_end.torchvision_loader --summary | grep -i mnist

# Download it first
python -m act.front_end.torchvision_loader --download MNIST resnet18
```

### Model Not Compatible
```bash
# Validate before downloading
python -m act.front_end.torchvision_loader --validate MNIST resnet18

# Check recommended models
python -m act.front_end.torchvision_loader --models-for MNIST

# Check compatible datasets for a model
python -m act.front_end.torchvision_loader --datasets-for resnet18
```

### Disk Space Issues
```bash
# Check total size
python -m act.front_end.torchvision_loader --list-downloads

# Remove old downloads manually
rm -rf data/torchvision/MNIST
```

## Complete Command Reference

| Command | Description |
|---------|-------------|
| `--category`, `-c` | Show datasets in specific category |
| `--dataset`, `-d` | Show detailed information for a dataset |
| `--summary` | Print complete mapping summary |
| `--category`, `-c` | Show datasets in specific category |
| `--dataset`, `-d` | Show detailed information for a dataset |
| `--download DATASET MODEL` | Download dataset-model pair (with validation) |
| `--split {train,test,both}` | Choose split to download (default: test) |
| `--list-downloads` | List all downloaded pairs with sizes |
| `--load-torchvision DATASET MODEL` | Load downloaded pair for use |
| `--batch-size N` | DataLoader batch size (default: 1) |
| `--validate DATASET MODEL` | Validate dataset-model compatibility |
| `--show-preprocessing DATASET` | Show preprocessing requirements |
| `--preprocessing-summary` | Show preprocessing summary for all |
| `--models-for DATASET` | Show all compatible models for dataset |
| `--datasets-for MODEL` | Show all datasets compatible with model |

## Statistics

- **Total Datasets**: 40
- **Classification Datasets**: 27 (testable with inference)
- **Detection/Segmentation/Video/Flow**: 13 (custom models only)
- **Total Models**: 63 unique architectures
- **Total Pairs**: 189 (including custom model combinations)
- **Categories**: 5 (classification, detection, segmentation, video, optical_flow)
- **Custom Models**: 2 (SimpleCNN, LeNet5 - rejected by validation)
- **Standard TorchVision Models**: 61 (ResNet, VGG, EfficientNet, ViT, etc.)

## License

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+

## See Also

- **VNNLIB Creator**: `../vnnlib_loader/README.md`
- **Unified CLI**: `../README.md`
- **Data Organization**: `../../../data/torchvision/README.md`
- **Pipeline Testing**: `../../pipeline/README.md`
- **PyTorch Models**: https://pytorch.org/vision/stable/models.html
