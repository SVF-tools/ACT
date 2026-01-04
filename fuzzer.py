# Setup and imports
import sys, os
act_root = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
sys.path.insert(0, act_root) if act_root not in sys.path else None

import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path

# Force reload all fuzzing modules to pick up latest changes
import importlib
modules_to_reload = [
    'act.pipeline.fuzzing.mutations',
    'act.pipeline.fuzzing.coverage',
    'act.pipeline.fuzzing.corpus',
    'act.pipeline.fuzzing.checker',
    'act.pipeline.fuzzing.actfuzzer',
    'act.front_end.model_synthesis',
    'act.front_end.vnnlib_loader.create_specs',
    'act.front_end.vnnlib_loader.vnnlib_parser',
    'act.front_end.vnnlib_loader.onnx_converter',
    'act.front_end.vnnlib_loader.data_model_loader',
]
for mod_name in modules_to_reload:
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])

from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
from act.front_end.model_synthesis import synthesize_models_from_specs
from act.pipeline.fuzzing.actfuzzer import ACTFuzzer, FuzzingConfig

# CIFAR-100 class names (100 fine-grained classes)
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

print("✓ Setup complete (modules reloaded with batch dimension fixes)")



print("="*80)
print("LOADING CIFAR-100 VNNLIB BENCHMARK")
print("="*80)

# Load VNNLib instances
creator = VNNLibSpecCreator(config_name="vnnlib_default")
spec_results = creator.create_specs_for_data_model_pairs(
    categories=["cifar100_2024"],
    max_instances=20,
    validate_shapes=True
)
print(f"✓ Loaded {len(spec_results)} instances\n")

# Randomly select 3 instances for fuzzing
import random
num_seeds = 3
random_indices = random.sample(range(len(spec_results)), min(num_seeds, len(spec_results)))
print(f"🎲 Randomly selected {len(random_indices)} instances: {[i+1 for i in random_indices]}\n")

selected_instances = [spec_results[i] for i in random_indices]

# Synthesize wrapped models for all selected instances
# Note: synthesize_models_from_specs returns (wrapped_models, reports) tuple
print("Synthesizing wrapped models for selected instances...")
wrapped_models, reports = synthesize_models_from_specs(selected_instances)
print(f"✓ Models wrapped for {len(wrapped_models)} instances\n")

# Load fuzzing config from YAML
config_path = "./act/pipeline/fuzzing/config.yaml"
with open(config_path) as f:
    yaml_data = yaml.safe_load(f)
    yaml_config = yaml_data['fuzzing']

# Override with 1-minute budget per instance + ENABLE TRACING
config = FuzzingConfig(
    max_iterations=yaml_config['max_iterations'],
    timeout_seconds=60.0,  # 1 minute per instance
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_counterexamples=yaml_config['save_counterexamples'],
    output_dir=Path("fuzzing_results_vnnlib"),
    report_interval=yaml_config['report_interval'],
    mutation_weights=yaml_config['mutation_weights'],
    # ============================================================================
    # 🔍 TRACING ENABLED: Capture fuzzing execution traces for analysis
    # ============================================================================
    trace_level=1,           # 1=default (strategies, coverage), 2=full (+ input snapshots)
    trace_sample_rate=10,     # Capture every iteration (1=all, 10=every 10th, etc.)
    trace_storage="json",    # Storage format: "json" or "hdf5"
    trace_output=None        # Auto-generate path in output_dir/traces.json
)

print(f"Fuzzing config: timeout={config.timeout_seconds}s per instance, device={config.device}")
print(f"Mutation weights: {config.mutation_weights}")
print(f"🔍 Tracing: ENABLED (level={config.trace_level}, storage={config.trace_storage})\n")

print("="*80)
print("STARTING FUZZING - ONE MODEL PER INSTANCE")
print("="*80)

# Fuzz each wrapped model independently
all_reports = []
all_fuzzers = []  # Store all fuzzers to access their corpus seeds
instance_info = []
trace_files = []  # Store trace file paths for each instance

for idx, (combo_id, wrapped_model) in enumerate(wrapped_models.items()):
    # Extract labeled input from model's InputLayer
    input_layer = wrapped_model[0]  # First layer is InputLayer
    labeled_input = input_layer.labeled_input  # LabeledInputTensor
    
    # Extract input spec from InputSpecLayer (second layer)
    input_spec_layer = wrapped_model[1]  # Second layer is InputSpecLayer
    input_spec = input_spec_layer.spec  # Access .spec attribute
    epsilon = float((input_spec.ub - input_spec.lb).max())
    
    # Extract output spec from OutputSpecLayer (last layer)
    output_spec_layer = wrapped_model[-1]  # Last layer is OutputSpecLayer
    output_spec = output_spec_layer.spec  # Access .spec attribute
    
    print(f"\n[{idx+1}/{len(wrapped_models)}] Instance: {combo_id}")
    print(f"   True label: {labeled_input.label} ({CIFAR100_CLASSES[labeled_input.label]})")
    print(f"   Epsilon: {epsilon:.6f}")
    
    # Store instance info for visualization
    instance_info.append({
        'index': idx,
        'input_tensor': labeled_input.tensor,
        'true_label': labeled_input.label,
        'epsilon': epsilon,
        'input_spec': input_spec,
        'combo_id': combo_id
    })
    
    # Run fuzzing with this model's stored input as seed
    fuzzer = ACTFuzzer(
        wrapped_model=wrapped_model,
        initial_seeds=[labeled_input],
        config=config
    )
    report = fuzzer.fuzz()
    all_reports.append(report)
    all_fuzzers.append(fuzzer)  # Store fuzzer for this instance
    
    # Store trace file path if tracing is enabled (FIXED: use fuzzer.tracer.output_path)
    if hasattr(fuzzer, 'tracer') and fuzzer.tracer is not None:
        trace_file_path = fuzzer.tracer.output_path
        trace_files.append(trace_file_path)
        print(f"   📊 Trace saved: {trace_file_path}")
    
    print(f"   ✓ Iterations: {report.total_iterations}, "
          f"Time: {report.total_time:.2f}s, "
          f"Counterexamples: {len(report.counterexamples)}")

# Aggregate results
print("\n" + "="*80)
print("FUZZING COMPLETE - ALL INSTANCES")
print("="*80)
total_iterations = sum(r.total_iterations for r in all_reports)
total_time = sum(r.total_time for r in all_reports)
total_counterexamples = sum(len(r.counterexamples) for r in all_reports)
avg_coverage = sum(r.neuron_coverage for r in all_reports) / len(all_reports)

print(f"Total iterations: {total_iterations} ({total_iterations / total_time:.1f} it/s)")
print(f"Total time: {total_time:.2f}s")
print(f"Total counterexamples: {total_counterexamples}")
print(f"Average coverage: {avg_coverage:.2%}")
print("="*80)

# Combine all counterexamples and create a unified report for visualization
report = type('CombinedReport', (), {
    'total_iterations': total_iterations,
    'total_time': total_time,
    'counterexamples': [ce for r in all_reports for ce in r.counterexamples],
    'neuron_coverage': avg_coverage,
    'seeds_explored': sum(r.seeds_explored for r in all_reports)
})()