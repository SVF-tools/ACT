# NetFactory Package

This package contains the modular NetFactory implementation used by the CLI.

## Architecture

The NetFactory package modules:

```
net_factory/
├── __init__.py          # Package exports
├── factory.py           # NetFactory + ConfigSampler (with internal utilities)
└── layer_builder.py     # Layer construction functions (with internal utilities)
```

## Module Responsibilities

### `factory.py` - NetFactory + ConfigSampler

**ConfigSampler** - Generic YAML-based sampling:
- `sample_family()`: Sample architecture family and parameters using YAML rules
- `sample_input_spec()`: Sample input specifications (BOX, LINF_BALL)
- `sample_output_spec()`: Sample output specifications (TOP1_ROBUST, MARGIN_ROBUST, LINEAR_LE, RANGE)
- `_sample_value()`: Rule-based value sampling (choice, range, weighted, repeat, probability, const)
- `_sample_dict()`: Recursive dict sampling

**NetFactory** - Main orchestration:
- Load YAML configuration and manage output paths
- Orchestrate sampling, building, and serialization
- Generate weight tensors and handle layer variables
- Create and save Net objects

Key methods:
- `__init__()`: Initialize factory with config and paths
- `generate()`: Main entry point for network generation
- `create_network()`: Create Net from specification
- `save_network()`: Serialize Net to JSON

### `layer_builder.py` - Layer Construction

Layer-by-layer construction functions:
- `build_mlp_layers()`: Build MLP layer sequences (plain, block, residual variants)
- `build_cnn_layers()`: Build CNN layer sequences (plain, stage variants)
- `append_conv2d()`: Add CONV2D layer
- `append_pool2d()`: Add pooling layer
- `append_dense()`: Add DENSE layer
- `append_act()`: Add activation layer
- `append_add()`: Add ADD layer (for residual connections)

### Internal Utility Functions

Each module contains its own internal utility functions (prefixed with `_`):

**factory.py**:
- `_stable_u32_from_bytes()`: Extract stable u32 from bytes
- `_derive_seed()`: Deterministic seed generation
- `_choose()`: Random choice with error handling

**layer_builder.py**:
- `_activation_kind()`: Map activation names to layer kinds
- `_infer_conv2d_output_hw()`: Compute Conv2D output shape
- `_infer_pool2d_output_hw()`: Compute Pool2D output shape
- `_ensure_batch1()`: Validate batch dimension
- `_prod()`: Product of shape dimensions
- `_as_block_param()`: Extract per-block parameters

This design follows the principle: "Keep utilities with their usage site for better locality."

## Benefits

1. **Reduced Complexity**: Each module has a single, clear responsibility
2. **Improved Maintainability**: Easier to locate and modify specific functionality
3. **Better Testability**: Modules can be tested independently
4. **Enhanced Readability**: Smaller files are easier to understand
5. **Easier Extension**: Adding new layer types or sampling strategies is straightforward


## Usage

```python
from act.back_end.net_factory import NetFactory

# Create factory with default config
factory = NetFactory()

# Generate networks
factory.generate()

# Or with custom config
factory = NetFactory(
    gen_config_path="custom_config.yaml",
    output_dir="output/nets",
    num_instances=10,
    base_seed=42
)
names = factory.generate()
```

## Configuration

All sampling behavior is controlled by `config_gen_act_net.yaml`:
- Model families (MLP, CNN)
- Architecture parameters (depth, width, channels)
- Input specifications (BOX, LINF_BALL)
- Output specifications (TOP1_ROBUST, MARGIN_ROBUST, LINEAR_LE, RANGE)
- Sampling probabilities

See `act/back_end/examples/config_gen_act_net.yaml` for details.

## Testing

Recommended checks:
- `python -m act.back_end --generate --config act/back_end/examples/config_gen_act_net.yaml --num 1`
- `python -m act.back_end --test-serialization --device cpu --dtype float32`

## Validation Strategy (Three-Layer Design)

NetFactory leverages ACT's existing validation infrastructure instead of implementing redundant checks. The validation responsibility is clearly separated across three stages:

### 1. Generation Stage (Automatic - Structural Validation)

**What**: Layer/Graph structure and field correctness
**When**: Automatically triggered during `Layer.__post_init__()` and `Net.__post_init__()`
**Where**: `act/back_end/layer_util.py` + `layer_schema.py`

Enforced checks:
- **`validate_layer()`**: Strict params/meta validation against REGISTRY
  - Required fields present
  - No unknown fields (with suggestions)
  - Correct types (e.g., LabeledInputTensor for INPUT)
- **`validate_graph()`**: Graph structure validation
  - Layer ID uniqueness
  - Variable ID validity (non-negative integers)
- **`validate_wrapper_graph()`**: Wrapper structure enforcement
  - INPUT → (model layers) → INPUT_SPEC → ... → ASSERT
  - Exactly one INPUT, at least one INPUT_SPEC, ASSERT at end

**Result**: If NetFactory generates an invalid network, construction will fail immediately with a clear error message.

### 2. JSON Stage (Optional - Format Validation)

**What**: JSON structure integrity
**When**: Optional manual/CI checks after generation
**Where**: `act/back_end/serialization/serialization.py`

```python
from act.back_end.serialization import validate_json_schema

# Optional: validate JSON structure before loading
with open("generated_net.json") as f:
    net_dict = json.load(f)
errors = validate_json_schema(net_dict)
if errors:
    print("JSON validation errors:", errors)
```

Checks:
- Top-level fields (`format_version`, `act_net`)
- Layer fields (`id`, `kind`, `params`, `meta`, `in_vars`, `out_vars`)
- Basic structure completeness

**Use case**: CI pipelines, batch validation of generated files

### 3. Verification Stage (Automatic - Constraint Validation)

**What**: Constraint set consistency (bounds, linear constraints)
**When**: Before solving/verification
**Where**: `act/back_end/utils.py` (called by verifier)

Checks:
- Variable bounds consistency
- Linear constraint dimensions
- NaN/Inf detection

**Note**: Not relevant to NetFactory (generation phase only)

### Why This Design?

1. **No Redundancy**: Reuses authoritative validation logic written by ACT core team
2. **Clear Separation**: Generation → Structure | JSON → Format | Verification → Constraints
3. **Automatic Safety**: Invalid networks cannot be constructed
4. **Minimal Overhead**: NetFactory focuses on "sample + build + save"

The `schema` section in `config_gen_act_net.yaml` serves as **documentation** to explain network classes and constraints, not as executable validation logic.

## Schema Contract (Documentation)

### Overview

The configuration includes a **schema contract** that documents the structure and constraints of generated networks. This serves as a formal specification and reference guide.

### Schema Structure

The schema is defined in `config_gen_act_net.yaml` under the `schema` section:

```yaml
schema:
  common_fields:        # Required for all networks
    - family            # mlp, cnn2d
    - variant           # plain, block, residual, stage
    - input_shape       # Tensor shape
    - num_classes       # Output classes
    - num_layers        # Total layers
    - num_params        # Parameter count

  size_tiers:          # Parameter count thresholds
    tiny: ≤ 50k
    small: ≤ 200k
    medium: ≤ 1M

  classes:             # Type-specific constraints
    mlp_plain: ...     # Plain MLP constraints
    mlp_block: ...     # Block MLP constraints
    mlp_residual: ...  # Residual MLP constraints
    cnn2d_plain: ...   # Plain CNN constraints
    resnet: ...        # ResNet constraints
```

### Network Classes

The schema defines 5 network classes:

| Class | Family | Variant | Description |
|-------|--------|---------|-------------|
| `mlp_plain` | mlp | plain | Feedforward MLP |
| `mlp_block` | mlp | block | Block-structured MLP |
| `mlp_residual` | mlp | residual | Residual MLP with skip connections |
| `cnn2d_plain` | cnn2d | plain | Sequential CNN |
| `resnet` | cnn2d | stage | ResNet-like staged CNN |

### Size Tiers

Networks are classified by parameter count:

- **Tiny**: ≤ 50,000 parameters
- **Small**: ≤ 200,000 parameters
- **Medium**: ≤ 1,000,000 parameters

### Purpose

The schema serves as:
- **Documentation**: Formal specification of network structure
- **Contract**: Agreement on required fields and valid ranges
- **Reference**: Guide for understanding generated networks

See `schema` section in `config_gen_act_net.yaml` for complete specifications including:
- Required fields for each class
- Constraint definitions
- Field extraction rules

## Future Extensions

The modular structure makes it easy to add:
- New layer types (add to `layer_builder.py`)
- New sampling rules (extend `ConfigSampler._sample_value()` in `factory.py`)
- New model families (add to YAML config and `layer_builder.py`)
- New network classes (add to `schema.classes` in YAML as documentation)
- Additional utility functions (add to relevant module with `_` prefix)
