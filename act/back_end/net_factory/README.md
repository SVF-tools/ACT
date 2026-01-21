# NetFactory Package

This package contains the modular NetFactory implementation used by the CLI.

## Architecture

The NetFactory package modules:

```
net_factory/
├── __init__.py          # Package exports
├── factory.py           # Main NetFactory class (with internal utilities)
├── sampler.py           # Configuration sampling logic (with internal utilities)
└── layer_builder.py     # Layer construction functions (with internal utilities)
```

## Module Responsibilities

### `factory.py` - NetFactory Class

Core responsibilities:
- Load YAML configuration
- Orchestrate sampling, building, and serialization
- Generate weight tensors
- Handle layer variables
- Create and save Net objects

Key methods:
- `__init__()`: Initialize factory with config and paths
- `generate()`: Main entry point for network generation
- `create_network()`: Create Net from specification
- `save_network()`: Serialize Net to JSON

### `sampler.py` - ConfigSampler Class

Handles all configuration sampling logic:
- `sample_family()`: Choose MLP or CNN architecture
- `sample_mlp()`: Sample MLP configuration
- `sample_cnn2d()`: Sample CNN configuration
- `sample_input_spec()`: Sample input specifications (BOX, LINF_BALL)
- `sample_output_spec()`: Sample output specifications (TOP1_ROBUST, MARGIN_ROBUST, etc.)

All sampling uses the YAML config from `config_gen_act_net.yaml`.

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
- `_derive_seed()`: Deterministic seed generation
- `_choose()`: Random choice with error handling

**sampler.py**:
- `_choose()`: Random choice with error handling
- `_randint_inclusive()`: Sample from range

**layer_builder.py**:
- `_activation_kind()`: Map activation names to layer kinds
- `_infer_conv2d_output_hw()`: Compute Conv2D output shape
- `_infer_pool2d_output_hw()`: Compute Pool2D output shape
- `_ensure_batch1()`: Validate batch dimension
- `_prod()`: Product of shape dimensions
- `_as_block_param()`: Extract per-block parameters

This design follows the principle: "If a utility is used only once, keep it with its usage site."

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

## Future Extensions

The modular structure makes it easy to add:
- New layer types (add to `layer_builder.py`)
- New sampling strategies (extend `ConfigSampler`)
- New model families (add methods to `sampler.py` and `layer_builder.py`)
- Additional utility functions (add to `utils.py`)
