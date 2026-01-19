# ACT Pipeline Module# ACT Pipeline Testing Framework



PyTorch model generation, ACT conversion, and testing utilities for the Abstract Constraint Transformer (ACT) framework.A comprehensive testing framework for validating the Abstract Constraint Transformer (ACT) abstraction verifier with configurable mock testing, property-based validation, and regression testing capabilities.



## Overview## Overview



The ACT Pipeline module provides tools for:The ACT Pipeline Testing Framework provides a robust testing infrastructure to validate the correctness, performance, and reliability of the ACT abstraction verifier. It supports multiple testing methodologies including mock input generation, real dataset integration, regression testing, and performance profiling.

- **PyTorch Model Generation**: Create PyTorch models from YAML configurations with exact weight loading

- **PyTorch ↔ ACT Conversion**: Convert between PyTorch nn.Module and ACT Net representations## Architecture

- **Regression Testing**: Track performance baselines and detect degradations

- **Performance Analysis**: Profile execution time, memory usage, and generate reports### Core Components

- **Utility Functions**: Memory management, logging, parallel execution

```

## Architectureact/pipeline/

├── __init__.py           # Main entry points and convenience functions

### Module Structure├── config.py             # Configuration management and validation

├── mock_factory.py       # Mock input generation from YAML configs

```├── correctness.py        # Verifier correctness and property validation

act/pipeline/├── regression.py         # Baseline capture and regression testing

├── __init__.py           # Module exports and initialization├── integration.py        # Front-end integration bridge

├── model_factory.py      # PyTorch model creation from YAML configs├── reporting.py          # Results analysis and report generation  

├── torch2act.py          # PyTorch→ACT Net converter├── utils.py              # Shared utilities and performance profiling

├── data_classes.py       # Shared data structures (ValidationResult, PerformanceResult, VerifyResult)├── run_tests.py          # Command-line interface

├── regression.py         # Baseline management and regression testing├── torch2act.py          # Torch→ACT converter 

├── reporting.py          # Report generation and visualization├── modules/configs/      # Configuration files (moved under modules)

├── utils.py              # Performance profiling and utilities│   ├── mock_inputs.yaml  # Mock data generation templates

├── llm_probe.py          # LLM integration utilities│   ├── test_scenarios.yaml # Complete test scenario definitions

└── README.md             # This file│   ├── solver_settings.yaml # Solver configuration options

```│   └── baselines.json    # Performance baseline storage

└── examples/             # Usage examples and quick tests

### Core Files    ├── quick_test.py     # Simple validation examples

    ├── custom_inputs.py  # Custom input generation examples

After cleanup, only these production-ready files remain:    └── ci_setup.py       # CI/CD integration examples

- **model_factory.py** - Core PyTorch model generation from YAML```

- **torch2act.py** - PyTorch→ACT Net conversion utilities

- **data_classes.py** - Shared type definitions## Design Principles

- **regression.py** - Regression testing (optional dependencies)

- **reporting.py** - Report generation (optional: matplotlib, seaborn)### 1. **Modular Testing Architecture**

- **utils.py** - Performance profiling and utilities- **Separation of Concerns**: Each module handles a specific aspect (mocking, validation, regression, etc.)

- **llm_probe.py** - LLM integration utilities- **Pluggable Components**: Easy to extend with new test types and validation methods

- **Independent Testing**: Each test type can run independently or as part of a suite

## Core Features

### 2. **Configuration-Driven Testing**

### 1. PyTorch Model Factory- **YAML Configurations**: Human-readable test specifications without code changes

- **Flexible Mock Generation**: Generate diverse test inputs from configuration templates

Create PyTorch models from `config_gen_act_net.yaml` with exact weight equivalence to ACT Nets:- **Scenario Composition**: Combine different components into complete test scenarios



```python### 3. **Comprehensive Validation Coverage**

from act.pipeline import ModelFactory- **Correctness Testing**: Validate verifier results against expected outcomes

- **Property-Based Testing**: Test fundamental properties like soundness and completeness

# Initialize factory- **Performance Testing**: Monitor execution time, memory usage, and resource consumption

factory = ModelFactory()- **Regression Testing**: Track changes and detect performance/correctness regressions



# Create model with exact weights from ACT Net### 4. **Real-World Integration**

model = factory.create_model("mnist_mlp_small", load_weights=True)- **Front-End Bridge**: Integration with ACT's actual front-end loaders and specifications

- **Dataset Support**: Testing with real MNIST, CIFAR, and custom datasets

# Create model with random weights (for testing)- **Model Support**: Testing with various neural network architectures

model = factory.create_model("mnist_cnn_small", load_weights=False)

```## Key Features



**Supported Layer Types** (30+):### Mock Input Generation (`mock_factory.py`)

- **Linear**: DENSE (fully connected)```python

- **Convolutional**: CONV1D, CONV2D, CONV3D# Generate diverse test inputs from YAML configuration

- **Pooling**: MAXPOOL1D/2D/3D, AVGPOOL1D/2D/3Dfactory = MockInputFactory()

- **Activations**: RELU, SIGMOID, TANH, SOFTMAX, LEAKY_RELU, ELU, SELU, GELUdata, labels = factory.generate_sample_data("mnist_small")

- **Normalization**: BATCHNORM1D/2D/3D, LAYERNORM, INSTANCENORM1D/2D/3D, GROUPNORMinput_spec = factory.generate_input_spec("robust_l_inf")

- **Recurrent**: RNN, LSTM, GRUmodel = factory.generate_model("simple_relu")

- **Regularization**: DROPOUT, DROPOUT2D, DROPOUT3D```

- **Reshaping**: FLATTEN, RESHAPE, TRANSPOSE, PERMUTE

**Capabilities:**

**Weight Transfer**:- **Sample Data**: Images, tensors with configurable distributions (uniform, normal, gaussian noise)

- Loads ACT Net JSON from `act/back_end/examples/nets/`- **Input Specifications**: L∞/L2 perturbations, box constraints, custom bounds

- Copies exact weight and bias tensors to PyTorch layers- **Output Specifications**: Classification robustness, margin constraints, custom properties

- Ensures numerical equivalence (tested with diff = 0.00e+00)- **Neural Networks**: Various architectures (linear, ReLU, CNN, custom)

- Handles missing biases gracefully (zero initialization)

### Correctness Validation (`correctness.py`)

### 2. PyTorch → ACT Conversion```python

# Validate verifier correctness with comprehensive testing

Convert PyTorch models to ACT Net representation:validator = AbstractionVerifierValidator()

result = validator.validate_correctness(test_cases)

```python```

from act.pipeline import TorchToACT

**Validation Types:**

# Initialize converter- **Basic Correctness**: Expected SAT/UNSAT results match actual outcomes

converter = TorchToACT()- **Property Testing**: Soundness (no false negatives) and completeness validation

- **Performance Testing**: Execution time, memory usage, resource consumption

# Convert PyTorch model to ACT Net- **BaB Refinement**: Branch-and-bound refinement effectiveness testing

act_net = converter.convert(pytorch_model, input_shape=(1, 784))

### Regression Testing (`regression.py`)

# Use ACT Net for verification```python

from act.back_end.verifier import verify_once# Capture baselines and detect regressions

result = verify_once(act_net, bounds, output_cons)baseline_mgr = BaselineManager()

```baseline_mgr.capture_baseline("mnist_cnn_v1", validation_results, performance_results)

regression_result = baseline_mgr.compare_to_baseline("mnist_cnn_v1", current_results)

### 3. Data Classes```



The module provides shared data structures for testing and verification:**Features:**

- **Baseline Capture**: Store performance and correctness metrics as baselines

```python- **Trend Analysis**: Track metrics over time and detect degradation patterns

from act.pipeline import ValidationResult, PerformanceResult, VerifyResult- **Regression Detection**: Automated detection of performance/correctness regressions

- **Threshold Configuration**: Configurable thresholds for regression sensitivity

# ValidationResult: Stores validation test results

val_result = ValidationResult(### Integration Testing (`integration.py`)

    success=True,```python

    total_tests=100,# Test with real ACT front-end components

    passed_tests=95,bridge = ACTFrontendBridge()

    failed_tests=5,test_case = IntegrationTestCase(

    execution_time=45.2,    dataset_name="mnist",

    memory_usage_mb=512.0,    model_path="models/mnist_cnn.onnx", 

    results=[...]    spec_type="local_lp",

)    epsilon=0.1

)

# PerformanceResult: Stores performance metricsresult = bridge.run_test(test_case)

perf_result = PerformanceResult(```

    test_id="test_001",

    execution_time=1.23,**Integration Points:**

### Testing Components

The pipeline provides comprehensive testing infrastructure:

- **Spec Creators**: TorchVision and VNNLIB spec generation
- **Model Synthesis**: Wrapped model generation from specs
- **Verification Tests**: Correctness validation and regression testing

)- **Device Management**: CPU/GPU testing with proper device handling



# VerifyResult: Verification outcome enum### Performance Profiling (`utils.py`)

print(VerifyResult.UNSAT)  # Property verified safe```python

print(VerifyResult.SAT)    # Counterexample found# Comprehensive performance monitoring

```profiler = PerformanceProfiler()

profiler.start()

**VerifyResult Values**:# ... run verification ...

- `SAT` - Property violated (counterexample found)metrics = profiler.stop()  # execution_time, peak_memory_mb, cpu_usage_percent

- `UNSAT` - Property holds (verified safe)```

- `CLEAN_FAILURE` - Failed with clean error

- `UNKNOWN` - Could not determine**Monitoring:**

- `TIMEOUT` - Verification timed out- **Execution Time**: Precise timing of verification operations

- `ERROR` - Unexpected error- **Memory Usage**: Peak memory consumption tracking

- **CPU/GPU Usage**: Resource utilization monitoring

### 4. Regression Testing- **Parallel Execution**: Multi-threaded test execution with resource tracking



Track performance baselines and detect regressions:### Report Generation (`reporting.py`)

```python

```python# Generate comprehensive test reports

from act.pipeline import BaselineManager, RegressionTestergenerator = ReportGenerator()

generator.generate_full_report(validation_results, performance_results, regression_results)

# Capture baseline```

baseline_mgr = BaselineManager()

baseline = baseline_mgr.capture_baseline(**Report Types:**

    name="v1.0",- **HTML Reports**: Interactive dashboards with plots and metrics

    validation_results=val_results,- **JSON Reports**: Machine-readable results for CI/CD integration

    performance_results=perf_results,- **Performance Analysis**: Bottleneck identification and optimization suggestions

    model_path="models/mnist_cnn.onnx"- **Trend Visualization**: Performance trends and regression analysis

)

## Usage Examples

# Compare to baseline

tester = RegressionTester()### 1. Quick Validation (3 lines)

regression_result = tester.compare_to_baseline(```python

    baseline_name="v1.0",from act.pipeline import validate_abstraction_verifier

    current_results=(new_val_results, new_perf_results)result = validate_abstraction_verifier("modules/configs/my_tests.yaml")

)print(f"Status: {'✅ PASSED' if result.success else '❌ FAILED'}")

```

if regression_result.has_regression:

    print(f"⚠️ Regression detected: {regression_result.summary}")### 2. Ultra-Simple Validation (1 line)

``````python

from act.pipeline import quick_validate

**Regression Detection**:success = quick_validate()  # Uses sensible defaults

- Accuracy drops (default: >5%)```

- Execution time increases (default: >20%)

- Memory usage increases (default: >15%)### 3. Custom Mock Testing

- Timeout rate increases (default: >10%)```python

- Customizable thresholdsfrom act.pipeline import MockInputFactory, AbstractionVerifierValidator



### 5. Report Generation# Generate custom test inputs

factory = MockInputFactory()

Generate comprehensive test reports with visualization:test_data = factory.generate_from_config("modules/configs/custom_mocks.yaml")



```python# Validate with custom inputs

from act.pipeline import ReportGenerator, ReportConfigvalidator = AbstractionVerifierValidator()

results = validator.run_validation_suite(test_data)

# Configure report```

config = ReportConfig(

    output_dir="reports/",### 4. Regression Testing

    include_plots=True,```python

    plot_format="png"from act.pipeline import BaselineManager, RegressionTester

)

# Capture new baseline

# Generate reportbaseline_mgr = BaselineManager()

generator = ReportGenerator(config=config)baseline_mgr.capture_baseline("v2.1", validation_results, performance_results)

generator.generate_full_report(

    validation_results=val_results,# Compare against previous baseline

    performance_results=perf_results,regression_tester = RegressionTester()

    regression_results=reg_results,regression_result = regression_tester.compare_baselines("v2.0", "v2.1")

    output_path="reports/test_report.html"```

)

```### 5. Command-Line Usage

```bash

**Report Features**:# Quick validation

- Test summary with pass/fail ratespython run_tests.py --quick

- Performance metrics (time, memory, CPU usage)

- Regression analysis with trend plots# Full test suite with reporting

- Interactive HTML dashboardspython run_tests.py --comprehensive --report results.html

- JSON export for CI/CD integration

# CI mode (fast, essential tests)

### 6. Performance Profilingpython run_tests.py --ci --output ci_results.json



Monitor execution time, memory, and resource usage:# Custom configuration

python run_tests.py --config my_tests.yaml --mock-config my_mocks.yaml

```python```

from act.pipeline import PerformanceProfiler

## Configuration System

# Profile verification

profiler = PerformanceProfiler()### Test Scenarios (`modules/configs/test_scenarios.yaml`)

profiler.start()```yaml

scenarios:

# ... run verification ...  quick_smoke_test:

    sample_data: "mnist_small"

metrics = profiler.stop()    input_spec: "robust_l_inf_small" 

print(f"Time: {metrics.execution_time:.2f}s")    output_spec: "classification"

print(f"Memory: {metrics.peak_memory_mb:.1f} MB")    model: "simple_relu"

print(f"CPU: {metrics.cpu_percent:.1f}%")    expected_result: "UNSAT"

```    timeout: 30

```

### 7. Utility Functions

### Mock Inputs (`modules/configs/mock_inputs.yaml`)

Memory management, logging, and parallel execution:```yaml

sample_data:

```python  mnist_small:

from act.pipeline import (    type: "image"

    print_memory_usage,    shape: [1, 28, 28]

    clear_torch_cache,    distribution: "uniform"

    setup_logging,    range: [0, 1]

    ProgressTracker,    batch_size: 10

    ParallelExecutor    num_classes: 10

)

input_specs:

# Setup logging  robust_l_inf_small:

setup_logging(level="INFO")    spec_type: "LOCAL_LP"

    norm_type: "inf"

# Memory management    epsilon: 0.1

print_memory_usage()```

clear_torch_cache()

### Solver Settings (`modules/configs/solver_settings.yaml`)

# Progress tracking```yaml

tracker = ProgressTracker(total=100)solvers:

for i in range(100):  torch_lp:

    # ... do work ...    enabled: true

    tracker.update(1)    timeout: 300

    memory_limit: "8GB"

# Parallel execution  

executor = ParallelExecutor(num_workers=4)  gurobi:

results = executor.run_parallel(tasks)    enabled: true

```    timeout: 600

    threads: 4

## Configuration Files```



### config_gen_act_net.yaml## Testing Workflow



Defines network architectures for model generation:### 1. **Development Testing**

```python

```yaml# During development - quick feedback

networks:from act.pipeline import quick_validate

  mnist_mlp_small:assert quick_validate(), "Basic functionality broken"

    architecture_type: mlp```

    input_shape: [784]

    layers:### 2. **Feature Testing**

      - kind: DENSE```python

        meta:# When adding new features - comprehensive validation

          in_features: 784result = validate_abstraction_verifier("modules/configs/feature_tests.yaml")

          out_features: 100assert result.validations.correctness.success, "Correctness regression detected"

          bias_enabled: true  # Generate bias tensors```

      - kind: RELU

      - kind: DENSE### 3. **Release Testing**

        meta:```bash

          in_features: 100# Before releases - full test suite with baseline comparison

          out_features: 10python run_tests.py --comprehensive --regression --report release_report.html

          bias_enabled: true```

```

### 4. **CI/CD Integration**

**Location**: `act/back_end/examples/config_gen_act_net.yaml````bash

# In CI pipelines - fast, reliable tests

## Testingpython run_tests.py --ci --timeout 120 --output ci_results.json

```

### Model Factory Tests

## Extension Points

Test PyTorch model creation and weight transfer:

### Adding New Test Types

```bash```python

cd /Users/z3310488/Documents/workspace/ACTclass CustomValidator(BaseValidator):

python act/pipeline/model_factory.py    def validate(self, test_case: TestCase) -> ValidationResult:

```        # Custom validation logic

        return ValidationResult(...)

**Expected output**:

```# Register with framework

Testing ModelFactory with all 4 networks from config_gen_act_net.yaml...validator.register_custom_validator("my_test", CustomValidator())

```

Testing network: mnist_mlp_small

✓ Model created successfully### Custom Mock Generators

✓ PyTorch model output matches ACT Net (diff = 0.00e+00)```python

class CustomGenerator(BaseGenerator):

Testing network: mnist_cnn_small    def generate(self, config: Dict[str, Any]) -> Any:

✓ Model created successfully        # Custom generation logic

✓ PyTorch model output matches ACT Net (diff = 0.00e+00)        return generated_data



Testing network: adversarial_simple# Register with factory

✓ Model created successfullyfactory.register_generator("custom_type", CustomGenerator())

✓ PyTorch model output matches ACT Net (diff = 0.00e+00)```



Testing network: custom_mlp_small### Custom Report Formats

✓ Model created successfully```python

✓ PyTorch model output matches ACT Net (diff = 0.00e+00)class CustomReportGenerator:

    def generate(self, results: List[ValidationResult]) -> str:

All 4 networks tested successfully!        # Custom reporting logic

```        return report_content



### Import Tests# Use with reporting system

generator.add_format("custom", CustomReportGenerator())

Verify module imports correctly:```



```bash## Error Handling and Debugging

python -c "from act.pipeline import ModelFactory, TorchToACT, ValidationResult, VerifyResult; print('✓ Import successful!')"

```### Comprehensive Error Reporting

- **Detailed Error Messages**: Clear error descriptions with context

## Optional Dependencies- **Stack Trace Capture**: Full debugging information for failures

- **Resource Monitoring**: Track resource usage during failures

Some features require additional packages:- **Graceful Degradation**: Continue testing even when individual tests fail



### Core Features (Always Available)### Debugging Support

- `torch` - PyTorch (required)```python

- `numpy` - Numerical operations (required)# Enable debug logging

import logging

### Optional Featureslogging.getLogger('act.pipeline').setLevel(logging.DEBUG)

- `matplotlib` - Plotting and visualization (for reporting)

- `seaborn` - Enhanced visualizations (for reporting)# Memory debugging

from act.pipeline.verification.utils import print_memory_usage

Install optional dependencies:print_memory_usage("Before verification")

```bash```

pip install matplotlib seaborn

```## Performance Considerations



**Graceful Degradation**: If optional dependencies are missing, the module will still import successfully but optional features will be disabled.### Parallel Execution

- **Multi-threaded Testing**: Parallel execution of independent tests

## Integration with ACT- **Resource Management**: Intelligent resource allocation and cleanup

- **Memory Optimization**: Efficient memory usage with automatic cleanup

### Front-End Integration

### Scalability Features

The pipeline integrates with ACT's spec creators for efficient testing:

- **Batch Processing**: Efficient handling of large test suites
- **Incremental Testing**: Only run tests affected by changes
- **Resource Limits**: Configurable memory and time limits

```python
from act.front_end.torchvision_loader.create_specs import TorchVisionSpecCreator
from act.pipeline import ModelFactory

# Create specs and load data-model pairs
creator = TorchVisionSpecCreator(config_name="torchvision_classification")
spec_results = creator.create_specs_for_data_model_pairs(num_samples=10)
```

## Verification Validation

The ACT Pipeline provides a comprehensive verification validation framework (`validate_verifier.py`) to ensure the soundness and correctness of the ACT verification backend. The validation framework implements two levels of validation:

### Level 1: Counterexample/Soundness Validation

Validates that the verifier doesn't claim CERTIFIED when concrete counterexamples exist. This is a critical soundness check.

**Validation Strategy:**
1. Generate strategic test cases (center, boundary, random) within input specification
2. Run concrete execution to find property violations
3. If counterexample found, run formal verification
4. Cross-validate: verifier MUST report FALSIFIED or UNKNOWN (not CERTIFIED)

**Validation Matrix:**
```
┌─────────────────────────┬────────────────────────────────────┬──────────────┐
│ Concrete Counterexample │ Verifier Result                    │ Validation   │
├─────────────────────────┼────────────────────────────────────┼──────────────┤
│ FOUND                   │ CERTIFIED                          │ ❌ FAILED    │
│                         │ (Soundness Bug - false negative)   │              │
├─────────────────────────┼────────────────────────────────────┼──────────────┤
│ FOUND                   │ FALSIFIED                          │ ✅ PASSED    │
│                         │ (Correct - verifier found issue)   │              │
├─────────────────────────┼────────────────────────────────────┼──────────────┤
│ FOUND                   │ UNKNOWN                            │ ⚠️ ACCEPTABLE│
│                         │ (Incomplete but sound)             │              │
├─────────────────────────┼────────────────────────────────────┼──────────────┤
│ NOT FOUND               │ Any Result                         │ ❓ INCONC.   │
│                         │ (Cannot validate - no ground truth)│              │
└─────────────────────────┴────────────────────────────────────┴──────────────┘
```

### Level 3: Bounds/Numerical Validation

Validates that abstract interpretation correctly overapproximates concrete activation values. This ensures transfer function soundness.

**Validation Strategy:**
1. Sample concrete inputs from input specification
2. Run concrete forward pass through PyTorch model → get concrete activations
3. Run abstract analysis through ACT → get abstract bounds for each layer
4. Check: `concrete_value ∈ [lb, ub]` for all layers and neurons

**Validation Matrix:**
```
┌──────────────────────┬────────────────────────┬──────────────┐
│ Concrete Values      │ Abstract Bounds        │ Validation   │
├──────────────────────┼────────────────────────┼──────────────┤
│ value ∈ [lb, ub]     │ All layers/neurons     │ ✅ PASSED    │
│ (Sound bounds)       │                        │              │
├──────────────────────┼────────────────────────┼──────────────┤
│ value ∉ [lb, ub]     │ Any layer/neuron       │ ❌ FAILED    │
│ (Unsound bounds)     │ (Transfer function bug)│              │
└──────────────────────┴────────────────────────┴──────────────┘
```

### Usage

**Via CLI (Recommended):**

```bash
# Run comprehensive validation (both levels, default)
python -m act.pipeline --validate-verifier --device cpu --dtype float64

# Run only counterexample validation (Level 1)
python -m act.pipeline --validate-verifier --mode counterexample

# Run only bounds validation (Level 3)
python -m act.pipeline --validate-verifier --mode bounds --samples 20

# Test specific networks
python -m act.pipeline --validate-verifier --networks mnist_mlp_small,mnist_cnn_small

# Test with specific solvers
python -m act.pipeline --validate-verifier --mode counterexample --solvers gurobi torchlp

# Test with different transfer function modes
python -m act.pipeline --validate-verifier --mode bounds --tf-modes interval hybridz
```

**Direct Python API:**

```python
from act.pipeline.verification.validate_verifier import VerificationValidator
import torch

# Create validator with specific device/dtype
validator = VerificationValidator(device='cpu', dtype=torch.float64)

# Run Level 1: Counterexample validation
summary_l1 = validator.validate_counterexamples(
    networks=['mnist_mlp_small', 'mnist_cnn_small'],
    solvers=['gurobi', 'torchlp']
)

# Run Level 3: Bounds validation
summary_l3 = validator.validate_bounds(
    networks=['mnist_mlp_small'],
    tf_modes=['interval'],
    num_samples=10
)

# Run comprehensive validation (both levels)
combined = validator.validate_comprehensive(
    networks=None,  # None = all networks
    solvers=['gurobi'],
    tf_modes=['interval'],
    num_samples=10
)

# Check results
if combined['overall_status'] == 'PASSED':
    print("✅ All validation passed!")
else:
    print("❌ Validation failed - check detailed results")
```

**CI/CD Integration:**

The validation framework is integrated into the GitHub Actions workflow for continuous testing:

```yaml
# Float64 test suite
- name: Run Verifier Validation (float64)
  run: python -m act.pipeline --validate-verifier --device cpu --dtype float64

# Float32 test suite  
- name: Run Verifier Validation (float32)
  run: python -m act.pipeline --validate-verifier --device cpu --dtype float32
```

**Exit Codes:**
- `0`: All validations passed (no soundness bugs)
- `1`: Soundness bugs detected (Level 1) or unsound bounds (Level 3)

**Test Networks:**

The validator uses test networks from `act/back_end/examples/nets/`:
- MLP networks: `mnist_mlp_small`, `mnist_mlp_medium`
- CNN networks: `mnist_cnn_small`, `cifar_cnn_small`
- RNN networks: (if available)
- Transformer networks: (if available)

## Integration with ACT Framework

The pipeline seamlessly integrates with ACT's core components:

- **Back-End Integration**: Direct use of `act.back_end` verification components
- **Front-End Bridge**: Integration with `act.front_end` spec creators and synthesis
- **Device Management**: Proper CUDA/CPU device handling
- **Configuration Compatibility**: Works with existing ACT configuration systems

Example integration:

```python
from act.front_end.torchvision_loader.create_specs import TorchVisionSpecCreator
from act.front_end.model_synthesis import synthesize_models_from_specs

# Create specs from TorchVision datasets
creator = TorchVisionSpecCreator(config_name="torchvision_classification")
spec_results = creator.create_specs_for_data_model_pairs(num_samples=10)

# Synthesize wrapped models
wrapped_models, reports, input_data = synthesize_models_from_specs(spec_results)
```

This design provides a robust, extensible testing framework that ensures the reliability and performance of the ACT abstraction verifier while being easy to use and extend.

### Back-End Integration

Convert to ACT and verify:

```python
from act.pipeline import TorchToACT
from act.back_end.verifier import verify_once

# Convert to ACT Net
converter = TorchToACT()
act_net = converter.convert(pytorch_model, input_shape=(1, 784))

# Verify
result = verify_once(act_net, bounds, output_cons)
print(f"Result: {result}")  # VerifyResult.SAT or VerifyResult.UNSAT
```

## Design Principles

### 1. Modularity
- Each file handles a specific responsibility
- Clear separation between core and optional features
- Easy to extend with new layer types or features

### 2. Numerical Equivalence
- PyTorch models are exactly equivalent to ACT Nets
- Weight transfer ensures bit-perfect copying
- Validated with maximum difference of 0.00e+00

### 3. Optional Dependencies
- Core functionality (ModelFactory, TorchToACT) has minimal dependencies
- Advanced features (regression, reporting) gracefully degrade if dependencies missing
- Clear error messages when optional features unavailable

### 4. Clean Architecture
- Removed obsolete testing framework files
- Only production-ready code remains
- Comprehensive documentation and testing

## Migration Guide

### From Old Testing Framework

The old testing framework (mock_factory, integration, correctness, config, run_tests) has been removed. Use the new simplified approach:

**Old approach** (removed):
```python
from act.pipeline import MockInputFactory, PipelineValidator
factory = MockInputFactory()
validator = PipelineValidator()
```

**New approach**:
```python
from act.pipeline import ModelFactory
factory = ModelFactory()
model = factory.create_model("mnist_mlp_small", load_weights=True)
```

## Strict Mode

**Strict mode** causes `VerifiableModel` to raise `ValueError` immediately on constraint violations instead of returning status gracefully. Useful for debugging and fail-fast testing.

**CLI Usage:**
```bash
python act/pipeline/cli.py --strict-mode [options...]
```

**Programmatic Usage:**
```python
from act.front_end.verifiable_model import VerifiableModel

# Enable globally (affects all instances)
VerifiableModel.set_strict_mode(True)
model = VerifiableModel(*layers)

try:
    result = model(input)  # Raises ValueError on violation
except ValueError as e:
    print(f"Violation: {e}")
```

**Behavior:**
- **Non-Strict** (default): Returns `{'input_satisfied': False, ...}`
- **Strict**: Prints explanation, raises `ValueError`

## Future Enhancements

- [ ] Add support for more layer types (attention, transformers)
- [ ] ACT → PyTorch conversion (inverse direction)
- [ ] Automatic test generation from VNNLIB properties
- [ ] Enhanced regression testing with statistical analysis
- [ ] Integration with CI/CD pipelines (GitHub Actions, Jenkins)

## License

ACT: Abstract Constraint Transformer  
Copyright (C) 2025– ACT Team

Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).  
Distributed without any warranty; see <http://www.gnu.org/licenses/>.
