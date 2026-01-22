# VNNLIB Category-Instance Mapping

A comprehensive framework for downloading, managing, and loading VNNLIB benchmark instances from VNN-COMP competitions for the ACT verification framework.

## Features

- **26 Categories**: Complete coverage of VNN-COMP 2022-2025 benchmarks
- **Standardized Instances**: Pre-validated ONNX models + VNNLIB property files
- **Multiple Domains**: Image classification, control systems, object detection, transformers, and more
- **Auto-Download**: Automatically download VNNLIB categories with ONNX models and properties
- **VNNLIB Parsing**: Full SMT-LIB format support for input/output constraints
- **ONNX Conversion**: Automatic ONNX → PyTorch conversion with operator support
- **Path Management**: Centralized configuration via `act.util.path_config`
- **Modular Architecture**: Separated concerns (mapping, loading, parsing, CLI)

## Category Overview

- **Image Classification**: CIFAR-100, TinyImageNet, VGG-16, traffic signs (5 categories)
- **Collision Avoidance**: ACAS Xu, aerospace systems (3 categories)
- **Object Detection**: YOLO, traffic sign detection (2 categories)
- **Control Systems**: State-space controllers, reachability analysis (3 categories)
- **Advanced Architectures**: Vision Transformers, NLP transformers (2 categories)
- **Specialized Domains**: Power systems, malware detection, 3D reconstruction (11 categories)

## Module Structure

```
act/front_end/vnnlib/
├── __init__.py              # Package exports and imports
├── __main__.py              # Entry point for python -m execution
├── cli.py                   # Command-line interface
├── category_mapping.py      # Core category mapping and API (26 categories)
├── create_specs.py          # VNNLibSpecCreator implementation
├── data_model_loader.py     # Download and load VNNLIB instances
├── vnnlib_parser.py         # Parse VNNLIB files (SMT-LIB format)
├── onnx_converter.py        # Convert ONNX → PyTorch
└── README.md                # This file
```

### File Descriptions

**`category_mapping.py` (494 lines)**
- Core category mapping dictionary (`CATEGORY_MAPPING`)
- 26 VNN-COMP categories with metadata (type, description, models, properties, dimensions, year)
- Category information retrieval (`get_category_info`, `list_categories`)
- Search and query functions (`search_categories`, `find_category_name`)
- Category type filtering (`list_categories_by_type`, `get_all_types`)
- Summary statistics (`get_summary_statistics`)
- Case-insensitive name resolution

**`create_specs.py`**
- `VNNLibSpecCreator` class implementation
- Generates InputSpec (BOX constraints) and OutputSpec (LINEAR_LE constraints)
- Processes category-instance pairs
- Integrates ONNX conversion and VNNLIB parsing
- Batch processing for multiple categories

**`data_model_loader.py`**
- Download functionality (`download_category`)
- Load functionality with auto-download (`load_category_instance`)
- Downloaded categories management (`list_downloaded_categories`)
- Instance file parsing (ONNX-VNNLIB mappings)
- Directory size calculation and formatting

**`vnnlib_parser.py`**
- VNNLIB format parser (SMT-LIB syntax)
- Input constraint extraction (BOX bounds)
- Output property parsing (LINEAR_LE constraints)
- Variable extraction (`X_i`, `Y_j`, `X_hat_k`)
- Constraint coefficient calculation

**`onnx_converter.py`**
- ONNX → PyTorch model conversion
- Operator support (Conv, ReLU, MaxPool, Gemm, BatchNorm, etc.)
- Automatic weight loading
- Shape inference and validation

**`cli.py`**
- Command-line interface implementation
- Category listing and search
- Download and load commands
- Instance validation
- VNNLIB parsing and inspection

**`__init__.py`**
- Package-level exports
- Unified import interface
- Version information

**`__main__.py`**
- Entry point for `python -m act.front_end.vnnlib_loader`
- Delegates to CLI main function

## Installation

```bash
# Ensure you're in the ACT project root
cd /path/to/ACT

# Activate the ACT environment
conda activate act-main

# All dependencies should already be installed via setup.sh
```

## VNNLIB Categories

The framework supports 26 categories from VNN-COMP 2022-2025, organized by application domain:

### Image Classification (5 categories)

| Category                              | Input Size    | Classes | Year | Description                                   |
|---------------------------------------|---------------|---------|------|-----------------------------------------------|
| **cifar100_2024**                     | 3×32×32       | 100     | 2024 | CIFAR-100 image classification with ResNet/VGG |
| **tinyimagenet_2024**                 | 3×64×64       | 200     | 2024 | TinyImageNet classification benchmarks        |
| **vggnet16_2022**                     | 3×224×224     | 1000    | 2022 | VGG-16 ImageNet classification               |
| **cersyve**                           | Varies        | Varies  | 2024 | Certified robustness benchmarks              |
| **traffic_signs_recognition_2023**    | 3×32×32       | 43      | 2023 | Traffic sign CNN classification              |

### Collision Avoidance & Safety (3 categories)

| Category                              | Input Dim | Output Dim | Year | Description                                      |
|---------------------------------------|-----------|------------|------|--------------------------------------------------|
| **acasxu_2023**                       | 5         | 5          | 2023 | ACAS Xu aircraft collision avoidance (45 networks) |
| **collins_aerospace_benchmark**       | Varies    | Varies     | 2024 | Industrial aerospace safety systems              |
| **collins_rul_cnn_2022**              | Varies    | 1          | 2022 | Remaining Useful Life prediction CNNs            |

### Object Detection (2 categories)

| Category                | Input Size    | Year | Description                             |
|-------------------------|---------------|------|-----------------------------------------|
| **yolo_2023**           | 3×416×416     | 2023 | YOLOv3/YOLOv5 object detection         |
| **cctsdb_yolo_2023**    | 3×416×416     | 2023 | Chinese traffic sign detection (YOLO)  |

### Control Systems (3 categories)

| Category          | Year | Description                                  |
|-------------------|------|----------------------------------------------|
| **lsnc_relu**     | 2024 | Learning-enabled state-space controllers     |
| **nn4sys**        | 2022 | Neural network system control and modeling   |
| **cora_2024**     | 2024 | Reachability analysis benchmarks             |

### Advanced Architectures (2 categories)

| Category              | Architecture        | Year | Description                          |
|-----------------------|---------------------|------|--------------------------------------|
| **vit_2023**          | Vision Transformer  | 2023 | ViT robustness verification          |
| **safenlp_2024**      | Transformer         | 2024 | NLP transformer safety properties    |

### Specialized Domains (11 categories)

| Category                      | Domain             | Year | Description                           |
|-------------------------------|--------------------|----- |---------------------------------------|
| **cgan_2023**                 | Generative         | 2023 | Conditional GAN verification          |
| **ml4acopf_2024**             | Power Systems      | 2024 | Power grid optimization networks      |
| **malbeware**                 | Security           | 2024 | Malware detection neural networks     |
| **metaroom_2023**             | 3D Vision          | 2023 | 3D scene reconstruction networks      |
| **dist_shift_2023**           | Robustness         | 2023 | Distribution shift robustness         |
| **tllverifybench_2023**       | Transfer Learning  | 2023 | Transfer learning verification        |
| **linearizenn_2024**          | Algorithm          | 2024 | Neural network linearization          |
| **relusplitter**              | Algorithm          | 2024 | ReLUSplitter algorithm benchmarks     |
| **sat_relu**                  | Algorithm          | 2024 | SAT-based verification                |
| **soundnessbench**            | Testing            | 2024 | Verifier soundness testing            |
| **test**                      | Testing            | 2024 | Test cases for verification tools     |

### Category Types

Categories are grouped by the following types:

| Type                        | Count | Examples                                              |
|-----------------------------|-------|-------------------------------------------------------|
| **image_classification**    | 5     | cifar100_2024, vggnet16_2022, tinyimagenet_2024       |
| **collision_avoidance**     | 1     | acasxu_2023                                           |
| **object_detection**        | 2     | yolo_2023, cctsdb_yolo_2023                           |
| **control**                 | 3     | lsnc_relu, nn4sys, cora_2024                          |
| **vision_transformer**      | 1     | vit_2023                                              |
| **nlp**                     | 1     | safenlp_2024                                          |
| **generative**              | 1     | cgan_2023                                             |
| **power_systems**           | 1     | ml4acopf_2024                                         |
| **malware_detection**       | 1     | malbeware                                             |
| **3d_reconstruction**       | 1     | metaroom_2023                                         |
| **distribution_shift**      | 1     | dist_shift_2023                                       |
| **transfer_learning**       | 1     | tllverifybench_2023                                   |
| **linearization**           | 1     | linearizenn_2024                                      |
| **verification_algorithm**  | 3     | relusplitter, sat_relu, soundnessbench                |
| **aerospace**               | 2     | collins_aerospace_benchmark, collins_rul_cnn_2022     |
| **robustness**              | 1     | cersyve                                               |
| **test**                    | 1     | test                                                  |

## Command-Line Usage

### Module Execution

The CLI can be executed in two ways:

```bash
# Method 1: Via package main (recommended)
python -m act.front_end.vnnlib_loader [OPTIONS]

# Method 2: Direct CLI module execution
python -m act.front_end.vnnlib_loader [OPTIONS]
```

Both methods are equivalent and provide the same functionality.

### 1. List Available Categories

```bash
# List all categories
python -m act.front_end.vnnlib_loader --list

# List categories by type
python -m act.front_end.vnnlib_loader --type image_classification
python -m act.front_end.vnnlib_loader --type control
python -m act.front_end.vnnlib_loader --type object_detection
python -m act.front_end.vnnlib_loader --type collision_avoidance

# Search for categories
python -m act.front_end.vnnlib_loader --search yolo
python -m act.front_end.vnnlib_loader --search acas
python -m act.front_end.vnnlib_loader --search transformer
```

### 2. Get Category Details

```bash
# Show detailed information about a specific category
python -m act.front_end.vnnlib_loader --info acasxu_2023
python -m act.front_end.vnnlib_loader --info cifar100_2024
python -m act.front_end.vnnlib_loader --info yolo_2023

# Show summary statistics
python -m act.front_end.vnnlib_loader --summary
```

### 3. Download Categories

```bash
# Download a complete category (ONNX models + VNNLIB properties)
python -m act.front_end.vnnlib_loader --download acasxu_2023
python -m act.front_end.vnnlib_loader --download cifar100_2024
python -m act.front_end.vnnlib_loader --download yolo_2023

# Download with verification
python -m act.front_end.vnnlib_loader --download test --verify
```

**Download Structure:**
```
data/vnnlib/
└── acasxu_2023/
    ├── onnx/              # ONNX model files
    ├── vnnlib/            # VNNLIB property files
    └── instances.csv      # Instance mappings (onnx, vnnlib, timeout)
```

### 4. List Downloaded Categories

```bash
# List all downloaded categories with details
python -m act.front_end.vnnlib_loader --list-downloads
```

**Example Output:**
```
================================================================================
DOWNLOADED VNNLIB CATEGORIES (3)
================================================================================

acasxu_2023
  Type: collision_avoidance
  Models: 45 neural networks
  Properties: Collision avoidance safety
  Input: 5-dimensional (state vector)
  Output: 5-dimensional (advisory vector)
  Year: 2023
  Size: 2.5 MB
  Instances: 10
  Location: /path/to/ACT/data/vnnlib/acasxu_2023

cifar100_2024
  Type: image_classification
  Models: ResNet, VGG variants
  Properties: Robustness under L-inf perturbations
  Input: 3×32×32 images
  Output: 100 classes
  Year: 2024
  Size: 150.3 MB
  Instances: 100
  Location: /path/to/ACT/data/vnnlib/cifar100_2024

yolo_2023
  Type: object_detection
  Models: YOLOv3, YOLOv5
  Properties: Detection robustness
  Input: 3×416×416 images
  Output: Bounding boxes + classes
  Year: 2023
  Size: 450.7 MB
  Instances: 50
  Location: /path/to/ACT/data/vnnlib/yolo_2023

================================================================================
Total Size: 603.5 MB
Total Instances: 160
================================================================================
```

### 5. Load Category Instances

```bash
# Load a specific instance from a category
python -m act.front_end.vnnlib_loader --load acasxu_2023 0

# Load with auto-download
python -m act.front_end.vnnlib_loader --load cifar100_2024 5 --auto-download
```

### 6. Parse and Validate VNNLIB Files

```bash
# Parse a VNNLIB file
python -m act.front_end.vnnlib_loader --parse path/to/property.vnnlib

# Validate VNNLIB syntax
python -m act.front_end.vnnlib_loader --validate path/to/property.vnnlib

# Show instance details
python -m act.front_end.vnnlib_loader --instance-info acasxu_2023 0
```

### 7. Unified CLI (Auto-Detection)

The unified CLI automatically detects VNNLIB categories:

```bash
# Auto-detects and downloads VNNLIB category
python -m act.front_end --download acasxu_2023

# Lists both TorchVision datasets and VNNLIB categories
python -m act.front_end --list

# Search across both creators
python -m act.front_end --search yolo

# Get info (auto-detects creator)
python -m act.front_end --info acasxu_2023
```

## Quick Start Examples

### Example 1: Download and Use ACAS Xu

```bash
# Download ACAS Xu category
python -m act.front_end.vnnlib_loader --download acasxu_2023

# List instances
python -m act.front_end.vnnlib_loader --instance-info acasxu_2023

# Load a specific instance
python -m act.front_end.vnnlib_loader --load acasxu_2023 0
```

### Example 2: Explore Categories by Type

```bash
# Find image classification categories
python -m act.front_end.vnnlib_loader --type image_classification

# Find control system categories
python -m act.front_end.vnnlib_loader --type control
```

### Example 3: Parse VNNLIB Properties

```bash
# Download and parse a category
python -m act.front_end.vnnlib_loader --download test
python -m act.front_end.vnnlib_loader --parse data/vnnlib/test/vnnlib/prop_1.vnnlib
```

## Recommended Categories by Use Case

### For Image Classification Verification

```bash
# CIFAR-100 with modern architectures
python -m act.front_end.vnnlib_loader --download cifar100_2024

# TinyImageNet benchmarks
python -m act.front_end.vnnlib_loader --download tinyimagenet_2024

# VGG-16 networks
python -m act.front_end.vnnlib_loader --download vggnet16_2022

# Traffic sign recognition
python -m act.front_end.vnnlib_loader --download traffic_signs_recognition_2023
```

### For Safety-Critical Systems

```bash
# Aircraft collision avoidance (ACAS Xu)
python -m act.front_end.vnnlib_loader --download acasxu_2023

# Aerospace industrial benchmarks
python -m act.front_end.vnnlib_loader --download collins_aerospace_benchmark

# Remaining Useful Life prediction
python -m act.front_end.vnnlib_loader --download collins_rul_cnn_2022
```

### For Object Detection

```bash
# YOLO detection networks
python -m act.front_end.vnnlib_loader --download yolo_2023

# Traffic sign detection (Chinese)
python -m act.front_end.vnnlib_loader --download cctsdb_yolo_2023
```

### For Control Systems

```bash
# State-space controllers
python -m act.front_end.vnnlib_loader --download lsnc_relu

# System control benchmarks
python -m act.front_end.vnnlib_loader --download nn4sys

# Reachability analysis
python -m act.front_end.vnnlib_loader --download cora_2024
```

### For Advanced Architectures

```bash
# Vision Transformers
python -m act.front_end.vnnlib_loader --download vit_2023

# NLP Transformers
python -m act.front_end.vnnlib_loader --download safenlp_2024

# Conditional GANs
python -m act.front_end.vnnlib_loader --download cgan_2023
```

### For Verification Algorithm Research

```bash
# Test cases for verifiers
python -m act.front_end.vnnlib_loader --download test

# ReLUSplitter algorithm benchmarks
python -m act.front_end.vnnlib_loader --download relusplitter

# SAT-based verification
python -m act.front_end.vnnlib_loader --download sat_relu

# Verifier soundness testing
python -m act.front_end.vnnlib_loader --download soundnessbench
```

## Programmatic Usage

### Loading Category Instances

```python
from act.front_end.vnnlib_loader import load_category_instance

# Load a specific instance (auto-downloads if not found)
result = load_category_instance(
    category_name='acasxu_2023',
    instance_id=0,
    auto_download=True
)

# Access components
onnx_model = result['onnx_model']        # ONNX model object
pytorch_model = result['pytorch_model']  # Converted PyTorch model
vnnlib_path = result['vnnlib_path']      # Path to VNNLIB file
input_spec = result['input_spec']        # Parsed input constraints
output_spec = result['output_spec']      # Parsed output properties
metadata = result['metadata']            # Instance metadata

# Use the PyTorch model
import torch
input_tensor = torch.randn(1, 5)  # ACAS Xu input
output = pytorch_model(input_tensor)
```

### Downloading Categories

```python
from act.front_end.vnnlib_loader import download_category

# Download a complete category
result = download_category(
    category_name='cifar100_2024',
    verify=True  # Verify downloaded files
)

if result['status'] == 'success':
    print(f"Downloaded to: {result['category_path']}")
    print(f"ONNX models: {result['onnx_count']}")
    print(f"VNNLIB files: {result['vnnlib_count']}")
    print(f"Instances: {result['instance_count']}")
    print(f"Size: {result['size_formatted']}")
```

### Creating Verification Specs

```python
from act.front_end.vnnlib_loader import VNNLibSpecCreator

# Create spec creator
creator = VNNLibSpecCreator()

# Generate specs for specific categories
results = creator.create_specs_for_data_model_pairs(
    categories=["acasxu_2023", "test"],
    max_instances=5  # Limit instances per category
)

# Process results
for category, instance_id, pytorch_model, input_tensors, spec_pairs in results:
    print(f"\n{category}/{instance_id}:")
    print(f"  Model: {pytorch_model}")
    print(f"  Input shape: {input_tensors[0].shape}")
    print(f"  Spec pairs: {len(spec_pairs)}")
    
    for i, (input_spec, output_spec) in enumerate(spec_pairs):
        print(f"  Spec {i}: {input_spec.in_kind} → {output_spec.out_kind}")
        # Use specs for verification
```

### Querying Categories

```python
from act.front_end.vnnlib_loader import (
    get_category_info,
    list_categories,
    search_categories,
    find_category_name,
    list_categories_by_type
)

# Get detailed category information
info = get_category_info('acasxu_2023')
print(f"Type: {info['type']}")
print(f"Models: {info['models']}")
print(f"Input dim: {info['input_dim']}")
print(f"Output dim: {info['output_dim']}")

# List all categories
all_categories = list_categories()
print(f"Total categories: {len(all_categories)}")

# Search categories
yolo_categories = search_categories('yolo')
print(f"YOLO categories: {yolo_categories}")

# Find exact match (case-insensitive)
category_name = find_category_name('ACASXU')  # Returns 'acasxu_2023'

# List by type
control_categories = list_categories_by_type('control')
print(f"Control categories: {control_categories}")
```

### Parsing VNNLIB Files

```python
from act.front_end.vnnlib_loader import parse_vnnlib

# Parse a VNNLIB file
result = parse_vnnlib('data/vnnlib/acasxu_2023/vnnlib/prop_1.vnnlib')

# Access parsed constraints
input_bounds = result['input_bounds']      # List of (var, lb, ub)
output_constraints = result['output_constraints']  # List of linear constraints
variables = result['variables']            # All declared variables

# Convert to ACT specs
input_spec = result['input_spec']    # InputSpec with BOX constraints
output_spec = result['output_spec']  # OutputSpec with LINEAR_LE constraints

print(f"Input variables: {len(input_bounds)}")
print(f"Output constraints: {len(output_constraints)}")
```

### Converting ONNX Models

```python
from act.front_end.vnnlib_loader import convert_onnx_to_pytorch

# Convert ONNX model to PyTorch
pytorch_model = convert_onnx_to_pytorch(
    onnx_path='data/vnnlib/acasxu_2023/onnx/model.onnx'
)

# Use the converted model
import torch
input_tensor = torch.randn(1, 5)
output = pytorch_model(input_tensor)
print(f"Output shape: {output.shape}")
```

## File Structure

Downloaded VNNLIB benchmarks are organized in `data/vnnlib/`:

```
data/vnnlib/
└── acasxu_2023/                    # Category directory
    ├── onnx/                       # ONNX model files
    │   ├── ACASXU_run2a_1_1_batch_2000.onnx
    │   ├── ACASXU_run2a_1_2_batch_2000.onnx
    │   ├── ACASXU_run2a_1_3_batch_2000.onnx
    │   └── ...                     # 45 ONNX models
    ├── vnnlib/                     # VNNLIB property files
    │   ├── prop_1.vnnlib
    │   ├── prop_2.vnnlib
    │   ├── prop_3.vnnlib
    │   └── ...                     # Multiple properties
    └── instances.csv               # Instance mappings
```

### Instance File Format

The `instances.csv` file maps ONNX models to VNNLIB properties:

```csv
onnx,vnnlib,timeout
onnx/ACASXU_run2a_1_1_batch_2000.onnx,vnnlib/prop_1.vnnlib,300
onnx/ACASXU_run2a_1_2_batch_2000.onnx,vnnlib/prop_2.vnnlib,300
onnx/ACASXU_run2a_1_3_batch_2000.onnx,vnnlib/prop_1.vnnlib,300
```

Each row defines a verification instance:
- **onnx**: Path to ONNX model file (relative to category directory)
- **vnnlib**: Path to VNNLIB property file (relative to category directory)
- **timeout**: Verification timeout in seconds

## VNNLIB Format

VNNLIB files define verification properties using SMT-LIB 2.0 syntax:

### Input Constraints (BOX)

Input constraints define box bounds for each input variable:

```lisp
; Declare input variables
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const X_2 Real)

; Assert box bounds [lb, ub]
(assert (>= X_0 0.6))
(assert (<= X_0 0.6798577687))
(assert (>= X_1 -0.5))
(assert (<= X_1 -0.4528301887))
(assert (>= X_2 -0.5))
(assert (<= X_2 0.5))
```

**Parsing Result:**
- Variable `X_0`: bounds [0.6, 0.6798577687]
- Variable `X_1`: bounds [-0.5, -0.4528301887]
- Variable `X_2`: bounds [-0.5, 0.5]

**ACT Representation:**
```python
InputSpec(
    in_kind=InKind.BOX,
    lb=torch.tensor([0.6, -0.5, -0.5]),
    ub=torch.tensor([0.6798577687, -0.4528301887, 0.5])
)
```

### Output Constraints (LINEAR_LE)

Output constraints define safety properties as disjunctive linear inequalities:

```lisp
; Declare output variables
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)
(declare-const Y_3 Real)
(declare-const Y_4 Real)

; Safety property: Y_0 >= Y_i for all i != 0
(assert (or
    (and (>= Y_0 Y_1)
         (>= Y_0 Y_2)
         (>= Y_0 Y_3)
         (>= Y_0 Y_4))
))
```

**Interpretation:**
- Verify that output Y_0 (advisory 0) is the maximum
- Equivalent to: Y_1 - Y_0 ≤ 0, Y_2 - Y_0 ≤ 0, Y_3 - Y_0 ≤ 0, Y_4 - Y_0 ≤ 0

**ACT Representation:**
```python
OutputSpec(
    out_kind=OutKind.SAFETY,
    # A·y ≤ b where A = [[1, -1, 0, 0, 0], [1, 0, -1, 0, 0], ...]
    A=torch.tensor([
        [1, -1, 0, 0, 0],   # Y_1 - Y_0 ≤ 0
        [1, 0, -1, 0, 0],   # Y_2 - Y_0 ≤ 0
        [1, 0, 0, -1, 0],   # Y_3 - Y_0 ≤ 0
        [1, 0, 0, 0, -1]    # Y_4 - Y_0 ≤ 0
    ]),
    b=torch.zeros(4)
)
```

### Complex Properties

VNNLIB supports complex disjunctive properties:

```lisp
; Property: Either Y_0 >= Y_1 OR Y_2 >= Y_3
(assert (or
    (and (>= Y_0 Y_1))
    (and (>= Y_2 Y_3))
))
```

Each disjunct becomes a separate verification query (split into multiple specs).

### Variable Naming Conventions

- **Input variables**: `X_0`, `X_1`, ..., `X_n` (network inputs)
- **Output variables**: `Y_0`, `Y_1`, ..., `Y_m` (network outputs)
- **Intermediate variables**: `X_hat_0`, `X_hat_1`, ... (optional, for intermediate layers)

## ONNX Model Support

### Supported Operators

The ONNX → PyTorch converter supports:

- **Linear**: `Gemm`, `MatMul`, `Add`
- **Convolution**: `Conv`, `ConvTranspose`
- **Pooling**: `MaxPool`, `AveragePool`, `GlobalAveragePool`
- **Activation**: `Relu`, `Sigmoid`, `Tanh`, `LeakyRelu`
- **Normalization**: `BatchNormalization`, `InstanceNormalization`
- **Reshape**: `Reshape`, `Flatten`, `Squeeze`, `Unsqueeze`
- **Other**: `Concat`, `Transpose`, `Slice`

### Conversion Example

```python
from act.front_end.vnnlib_loader import convert_onnx_to_pytorch

# Convert ONNX to PyTorch
pytorch_model = convert_onnx_to_pytorch(
    onnx_path='data/vnnlib/acasxu_2023/onnx/model.onnx'
)

# Model is ready to use
import torch
x = torch.randn(1, 5)  # ACAS Xu input
y = pytorch_model(x)    # Forward pass
print(f"Output: {y}")   # 5-dimensional advisory vector
```

## Integration with ACT Pipeline

The VNNLIB creator integrates seamlessly with ACT's verification pipeline:

### Complete Verification Workflow

```python
from act.front_end.vnnlib_loader import VNNLibSpecCreator
from act.front_end.model_synthesis import model_synthesis
from act.pipeline.verification.torch2act import torch_to_act_net
from act.back_end.verifier import verify_once

# 1. Create specs from VNNLIB benchmarks
creator = VNNLibSpecCreator()
results = creator.create_specs_for_data_model_pairs(
    categories=["acasxu_2023"],
    max_instances=5
)

# 2. Synthesize wrapped models (InputSpecLayer + Model + OutputSpecLayer)
wrapped_models, input_data = model_synthesis(spec_results=results)

# 3. Convert to ACT Net representation
for model_id, wrapped_model in wrapped_models.items():
    # Convert PyTorch → ACT
    act_net = torch_to_act_net(
        model=wrapped_model,
        input_sample=input_data[model_id][0]
    )
    
    # 4. Verify with ACT backend
    result = verify_once(
        net=act_net,
        solver='gurobi',
        timeout=300
    )
    
    print(f"{model_id}: {result.status}")
    if result.status == VerifyStatus.CERTIFIED:
        print(f"  ✓ Property verified!")
    elif result.status == VerifyStatus.FALSIFIED:
        print(f"  ✗ Counterexample found: {result.counterexample}")
```

### Step-by-Step Integration

#### Step 1: Spec Creation

```python
from act.front_end.vnnlib_loader import VNNLibSpecCreator

creator = VNNLibSpecCreator()
results = creator.create_specs_for_data_model_pairs(
    categories=["test"],
    max_instances=1
)

# Results: List[(category, instance_id, pytorch_model, input_tensors, spec_pairs)]
for category, instance_id, pytorch_model, input_tensors, spec_pairs in results:
    print(f"Category: {category}")
    print(f"Instance: {instance_id}")
    print(f"Model: {pytorch_model}")
    print(f"Inputs: {[t.shape for t in input_tensors]}")
    print(f"Specs: {len(spec_pairs)} pairs")
```

#### Step 2: Model Synthesis

```python
from act.front_end.model_synthesis import model_synthesis

# Wrap models with spec layers
wrapped_models, input_data = model_synthesis(spec_results=results)

# Each wrapped model: InputSpecLayer → PyTorchModel → OutputSpecLayer
for model_id, wrapped_model in wrapped_models.items():
    print(f"Model ID: {model_id}")
    print(f"Layers: {list(wrapped_model.children())}")
    print(f"Input data: {input_data[model_id][0].shape}")
```

#### Step 3: Torch → ACT Conversion

```python
from act.pipeline.verification.torch2act import torch_to_act_net

# Convert each wrapped model to ACT representation
for model_id, wrapped_model in wrapped_models.items():
    act_net = torch_to_act_net(
        model=wrapped_model,
        input_sample=input_data[model_id][0]
    )
    
    print(f"ACT Net: {len(act_net.layers)} layers")
    print(f"Input vars: {act_net.layers[0].vars}")
    print(f"Output vars: {act_net.layers[-1].vars}")
```

#### Step 4: Verification

```python
from act.back_end.verifier import verify_once
from act.back_end.bab import verify_bab
from act.back_end import VerifyStatus, VerifyResult

# Single-shot verification
result = verify_once(
    net=act_net,
    solver='gurobi'  # or 'torch_lp'
)

if result.status == VerifyStatus.CERTIFIED:
    print("✓ Property verified (no counterexamples exist)")
elif result.status == VerifyStatus.FALSIFIED:
    print(f"✗ Counterexample found: {result.counterexample}")
elif result.status == VerifyStatus.UNKNOWN:
    # Refine with branch-and-bound
    result = verify_bab(
        net=act_net,
        solver='gurobi',
        max_depth=10,
        timeout=300
    )
```

### Batch Processing

```python
from act.front_end.vnnlib_loader import VNNLibSpecCreator
from act.front_end.model_synthesis import model_synthesis
from act.pipeline.verification.torch2act import torch_to_act_net
from act.back_end.verifier import verify_once
from act.back_end import VerifyStatus

# Process multiple categories
categories = ["acasxu_2023", "test", "sat_relu"]

for category in categories:
    print(f"\n{'='*80}")
    print(f"Processing category: {category}")
    print('='*80)
    
    # Create specs
    creator = VNNLibSpecCreator()
    results = creator.create_specs_for_data_model_pairs(
        categories=[category],
        max_instances=10
    )
    
    # Synthesize and verify
    wrapped_models, input_data = model_synthesis(spec_results=results)
    
    certified = 0
    falsified = 0
    unknown = 0
    
    for model_id, wrapped_model in wrapped_models.items():
        try:
            act_net = torch_to_act_net(wrapped_model, input_data[model_id][0])
            result = verify_once(act_net, solver='gurobi', timeout=60)
            
            if result.status == VerifyStatus.CERTIFIED:
                certified += 1
            elif result.status == VerifyStatus.FALSIFIED:
                falsified += 1
            else:
                unknown += 1
        except Exception as e:
            print(f"  Error on {model_id}: {e}")
            unknown += 1
    
    print(f"\nResults for {category}:")
    print(f"  Certified: {certified}")
    print(f"  Falsified: {falsified}")
    print(f"  Unknown: {unknown}")
```

## Differences from TorchVision Creator

| Aspect | TorchVision Creator | VNNLIB Creator |
|--------|---------------------|----------------|
| **Data Source** | PyTorch torchvision datasets | VNN-COMP benchmark repository |
| **Model Format** | PyTorch native (.pth, torchvision.models) | ONNX (.onnx) → PyTorch conversion |
| **Model Origin** | Pre-trained or custom PyTorch models | Competition benchmark networks |
| **Spec Source** | Generated from ε-perturbations | Parsed from VNNLIB files |
| **Input Constraints** | L-inf/L-p balls, linear polytopes | BOX constraints from SMT-LIB |
| **Output Constraints** | Safety properties (max class) | LINEAR_LE from VNNLIB properties |
| **Download Unit** | Dataset + individual models | Category (all ONNX + VNNLIB files) |
| **Instance Count** | User-defined (dataset samples) | Fixed (instances.csv mappings) |
| **Use Case** | Custom verification workflows | Standard benchmark evaluation |
| **Flexibility** | High (any dataset-model pair) | Medium (predefined instances) |
| **Standardization** | Low (custom specs) | High (VNN-COMP format) |
| **Domains** | General vision tasks | Specialized verification domains |

### When to Use Each Creator

**Use TorchVision Creator when:**
- Verifying custom models on standard datasets
- Exploring different ε-perturbation values
- Working with PyTorch-native models
- Flexible spec generation needed
- Rapid prototyping and experimentation

**Use VNNLIB Creator when:**
- Reproducing VNN-COMP results
- Comparing with other verifiers
- Standard benchmark evaluation
- Working with specialized domains (control, aerospace)
- Need standardized specifications

## Data Organization

See `data/vnnlib/README.md` for detailed information about:
- Directory structure
- Category statistics
- Instance file format
- Download instructions
- File sizes and storage requirements

## Performance Tips

### Memory Management

```python
import torch
import gc

# Clear cache between verifications
for model_id in wrapped_models:
    # Verify model
    result = verify_once(...)
    
    # Clean up
    torch.cuda.empty_cache()  # If using GPU
    gc.collect()
```

### Parallel Processing

```python
from multiprocessing import Pool

def verify_instance(args):
    category, instance_id = args
    # Load, convert, verify
    result = ...
    return result

# Process multiple instances in parallel
with Pool(4) as pool:
    results = pool.map(verify_instance, instance_list)
```

### Timeout Management

```python
# Set appropriate timeouts per category
timeouts = {
    'test': 60,           # Fast test cases
    'acasxu_2023': 300,   # Medium difficulty
    'yolo_2023': 3600     # Complex models
}

result = verify_once(
    act_net,
    solver='gurobi',
    timeout=timeouts.get(category, 300)
)
```

## Troubleshooting

### ONNX Conversion Issues

```python
# Check ONNX model before conversion
import onnx
model = onnx.load('path/to/model.onnx')
onnx.checker.check_model(model)

# View operators
for node in model.graph.node:
    print(f"Op: {node.op_type}")
```

### VNNLIB Parsing Issues

```python
# Debug VNNLIB parsing
from act.front_end.vnnlib_loader import parse_vnnlib

try:
    result = parse_vnnlib('path/to/property.vnnlib')
    print(f"Input vars: {len(result['input_bounds'])}")
    print(f"Output constraints: {len(result['output_constraints'])}")
except Exception as e:
    print(f"Parse error: {e}")
    # Check VNNLIB file syntax
```

### Instance Loading Issues

```python
# Verify instance file exists
import os
instance_csv = 'data/vnnlib/category/instances.csv'
if not os.path.exists(instance_csv):
    print("Run: python -m act.front_end.vnnlib_loader --download category")
```

## See Also

- **TorchVision Creator**: `act/front_end/torchvision/README.md`
- **Unified CLI**: `act/front_end/README.md`
- **Data Organization**: `data/vnnlib/README.md`
- **Pipeline Testing**: `act/pipeline/README.md`
- **VNN-COMP Benchmarks**: https://github.com/VNN-COMP/vnncomp2025_benchmarks
- **VNNLIB Format**: https://www.vnnlib.org/

## Citation

If you use the VNNLIB categories in your research, please cite VNN-COMP:

```bibtex
@misc{vnncomp2025,
  title={VNN-COMP 2025 Benchmarks},
  author={VNN-COMP Organizers},
  year={2025},
  howpublished={\url{https://github.com/VNN-COMP/vnncomp2025_benchmarks}}
}
```
