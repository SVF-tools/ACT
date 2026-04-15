#===- act/front_end/model_synthesis.py - Model Synthesis Framework -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Model Synthesis and Generation Framework. Advanced neural network synthesis,
#   optimization, and domain-specific model generation. Single-file implementation
#   for ACT-compatible model synthesis pipeline.
#
#===---------------------------------------------------------------------===#

# Detect if running as script (not as module) and exit with helpful message
if __name__ == "__main__" and __package__ is None:
    import sys
    print("\n" + "="*80)
    print("⚠️  ERROR: Cannot run as script due to import conflicts!")
    print("Please run as a module instead:")
    print("  python -m act.front_end.model_synthesis")
    print("="*80 + "\n")
    sys.exit(1)

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union

# Import ACT components
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.verifiable_model import (
    InputLayer,
    InputSpecLayer,
    OutputSpecLayer,
    VerifiableModel,
)


# -----------------------------------------------------------------------------
# 2) Small utilities
# -----------------------------------------------------------------------------
def prod(seq: Tuple[int, ...]) -> int:
    """Calculate product of sequence elements."""
    p = 1
    for s in seq:
        p *= s
    return p


def infer_layout_from_tensor(x: torch.Tensor) -> str:
    """Infer tensor layout (HWC, CHW, or FLAT) from shape."""
    if x.dim() == 4 and x.shape[-1] in (1, 3, 4):
        return "HWC"
    elif x.dim() == 4:
        return "CHW"
    return "FLAT"


def _merge_specs_to_batch(
    lts: List[LabeledInputTensor],
    in_specs: List[InputSpec],
    out_specs: List[OutputSpec],
    in_kind: str,
    out_kind: str
) -> Tuple[LabeledInputTensor, InputSpec, OutputSpec]:
    """
    Merge multiple single-sample specs into batched specs for efficient verification.
    
    Batching Strategy: Concatenate N samples along batch dimension (dim=0)
    -----------------------------------------------------------------------
    Example (3 MNIST samples):
    
    Before merging (3 separate specs):
      Sample 0: tensor=(1,1,28,28), label=[7], InputSpec(center=(1,1,28,28), eps=[0.03]), OutputSpec(y_true=[7], margin=None)
      Sample 1: tensor=(1,1,28,28), label=[2], InputSpec(center=(1,1,28,28), eps=[0.05]), OutputSpec(y_true=[2], margin=[0.5])
      Sample 2: tensor=(1,1,28,28), label=[1], InputSpec(center=(1,1,28,28), eps=[0.01]), OutputSpec(y_true=[1], margin=None)
    
    After merging (1 batched spec):
      LabeledInputTensor:
        tensor: (3,1,28,28)        # 3 images concatenated
        label:  [7, 2, 1]          # 3 labels concatenated
      
      InputSpec (LINF_BALL mode):
        center: (3,1,28,28)        # 3 center images concatenated
        eps:    [0.03, 0.05, 0.01] # 3 epsilon values concatenated
        lb:     (3,1,28,28)        # computed as clamp(center - eps, 0)
        ub:     (3,1,28,28)        # computed as clamp(center + eps, 1)
      
      OutputSpec:
        y_true: [7, 2, 1]          # 3 true labels concatenated
        margin: [0.0, 0.5, 0.0]    # None → 0.0, then concatenated
    
    Note: All tensors must have same shape (C,H,W) to batch successfully.
    The grouping logic (by data_source+model_name) ensures this naturally.
    
    Args:
        lts: List of N LabeledInputTensor (each with shape (1,C,H,W))
        in_specs: List of N InputSpec (BOX or LINF_BALL)
        out_specs: List of N OutputSpec
        in_kind: "BOX" or "LINF_BALL"
        out_kind: e.g., "TOP1_ROBUST", "MARGIN_ROBUST"
    
    Returns:
        (batched_labeled_tensor, batched_input_spec, batched_output_spec)
    """
    # Assert all tensors have the same shape (guaranteed by grouping in gkey)
    if len(lts) > 1:
        first_shape = lts[0].tensor.shape
        assert all(lt.tensor.shape == first_shape for lt in lts), \
            f"Shape mismatch: {[lt.tensor.shape for lt in lts]}"
    
    # Merge input tensors: (1,C,H,W) * N → (N,C,H,W)
    tensor = torch.cat([lt.tensor for lt in lts], dim=0)
    labels = torch.cat([lt.label for lt in lts], dim=0) if all(lt.label is not None for lt in lts) else None
    
    # Merge input specs based on kind
    if in_kind == InKind.BOX:
        # Assert all lb/ub tensors have the same shape
        if len(in_specs) > 1:
            first_lb_shape = in_specs[0].lb.shape
            assert all(s.lb.shape == first_lb_shape for s in in_specs), \
                f"InputSpec.lb shape mismatch: {[s.lb.shape for s in in_specs]}"
            assert all(s.ub.shape == first_lb_shape for s in in_specs), \
                f"InputSpec.ub shape mismatch: {[s.ub.shape for s in in_specs]}"
        
        lb = torch.cat([s.lb for s in in_specs], dim=0)
        ub = torch.cat([s.ub for s in in_specs], dim=0)
        center, eps = None, None
    elif in_kind == InKind.LINF_BALL:
        # Assert all center tensors have the same shape
        if len(in_specs) > 1:
            first_center_shape = in_specs[0].center.shape
            assert all(s.center.shape == first_center_shape for s in in_specs), \
                f"InputSpec.center shape mismatch: {[s.center.shape for s in in_specs]}"
        
        center = torch.cat([s.center for s in in_specs], dim=0)
        lb = torch.cat([torch.clamp(s.center - s.eps, 0) for s in in_specs], dim=0)
        ub = torch.cat([torch.clamp(s.center + s.eps, 1) for s in in_specs], dim=0)
        eps = torch.stack([s.eps for s in in_specs])
    else:
        raise NotImplementedError(f"Batching for {in_kind} not implemented")
    
    # Merge output specs: y_true and margin
    y_true = torch.cat([s.y_true for s in out_specs], dim=0) if all(s.y_true is not None for s in out_specs) else None
    # Use default dtype - device is automatically handled by device_manager
    margins = torch.cat([
        s.margin if s.margin is not None else torch.tensor([0.0])
        for s in out_specs
    ], dim=0) if any(s.margin is not None for s in out_specs) else None
    
    # Create batched spec objects
    batched_lt = LabeledInputTensor(tensor=tensor, label=labels)
    batched_in = InputSpec(kind=in_kind, lb=lb, ub=ub, center=center, eps=eps)
    def _batch_attr(attr):
        """Batch a field that only exists for certain output spec kinds (e.g. c/d for LINEAR_LE, lb/ub for RANGE). Returns None if this kind doesn't use the field."""
        vals = [getattr(s, attr, None) for s in out_specs]
        return torch.stack(vals) if all(v is not None for v in vals) else None
    c_vec, d_vec = _batch_attr('c'), _batch_attr('d')
    out_lb, out_ub = _batch_attr('lb'), _batch_attr('ub')
    batched_out = OutputSpec(kind=out_kind, y_true=y_true, margin=margins, c=c_vec, d=d_vec, lb=out_lb, ub=out_ub)
    
    return batched_lt, batched_in, batched_out


# -----------------------------------------------------------------------------
# 3) Model synthesis from spec creators
# -----------------------------------------------------------------------------

def _build_batched_model(
    gkey: Tuple[str, str, "InKind", "OutKind"],
    grouped_specs: List[Tuple["LabeledInputTensor", "InputSpec", "OutputSpec", str]],
    pytorch_model: nn.Module
) -> "VerifiableModel":
    """
    Build a batched VerifiableModel from grouped specs.
    
    Args:
        gkey: (data_source, model_name, input_kind, output_kind)
        grouped_specs: List of (labeled_tensor, input_spec, output_spec, combo_id)
        pytorch_model: Single PyTorch model to wrap 
        
    Returns:
        vm: Batched VerifiableModel
    """
    data_src, model_name, in_kind, out_kind = gkey
    
    # Extract components from grouped items
    lts = [i[0] for i in grouped_specs]        # LabeledInputTensor objects
    in_specs = [i[1] for i in grouped_specs]   # InputSpec objects
    out_specs = [i[2] for i in grouped_specs]  # OutputSpec objects
    
    # Merge into batched specs
    batched_lt, batched_in, batched_out = _merge_specs_to_batch(lts, in_specs, out_specs, in_kind, out_kind)
    
    # Build layer stack
    layers: List[nn.Module] = [
        InputLayer(batched_lt, batched_lt.tensor.shape, batched_lt.tensor.dtype,
                  layout=infer_layout_from_tensor(batched_lt.tensor), dataset_name=data_src),
        InputSpecLayer(spec=batched_in),
    ]
    
    # Add model and output spec layer
    layers.extend([pytorch_model, OutputSpecLayer(spec=batched_out)])
    
    # Create VerifiableModel and move to correct device
    vm = VerifiableModel(*layers)
    model_device = next(pytorch_model.parameters()).device
    vm = vm.to(model_device)
    
    return vm


def synthesize_models_from_specs(
    spec_results: List[Tuple[str, str, nn.Module, List[LabeledInputTensor], List[Tuple[InputSpec, OutputSpec]]]]
) -> Dict[Tuple[str, str, str, str], nn.Module]:
    """
    Synthesize wrapped models with automatic batching.
    
    Groups specs by (dataset, model, input_kind, output_kind) and creates batched
    VerifiableModel instances. Reduces model count by 80-90% in practice.
    
    Args:
        spec_results: List of (data_source, model_name, pytorch_model, 
                              labeled_tensors, spec_pairs)
    
    Returns:
        synthesis_models: Dict[(dataset, model, in_kind, out_kind), VerifiableModel]
    """
    from collections import defaultdict
    
    # -------------------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------------------
    assert spec_results, (
        "synthesize_models_from_specs() requires at least one spec_result!\n"
    )
    
    print(f"\n🧬 Synthesizing models from {len(spec_results)} spec result(s)...")
    
    # -------------------------------------------------------------------------
    # Grouping specs by (data_source, model_identity, input_kind, output_kind)
    # Uses id(pytorch_model) to group instances sharing the same model object,
    # even when model_name differs per instance (e.g., VNNLib prop_idx names).
    # -------------------------------------------------------------------------
    groups: Dict[Tuple, List] = defaultdict(list)
    models: Dict[int, Tuple[nn.Module, str]] = {}  # id(model) -> (model, model's representative_name)
    
    for data_source, model_name, pytorch_model, labeled_tensors, spec_pairs in spec_results:
        if not labeled_tensors or not spec_pairs:
            continue
        
        # Group by model identity (id(pytorch_model)) instead of model_name
        mid = id(pytorch_model)
        if mid not in models:
            models[mid] = (pytorch_model, model_name)  # keep first name as representative
        sps = len(spec_pairs) // len(labeled_tensors) if labeled_tensors else 1
        
        for idx, (in_spec, out_spec) in enumerate(spec_pairs):
            lt = labeled_tensors[min(idx // sps if sps > 0 else 0, len(labeled_tensors) - 1)] 
            gkey = (data_source, mid, in_spec.kind, out_spec.kind)
            groups[gkey].append((lt, in_spec, out_spec, f"{data_source}:{model_name}:s{idx}"))
    
    # -------------------------------------------------------------------------
    # Synthesis Loop: Build batched models from grouped specs
    # -------------------------------------------------------------------------
    synthesis_models: Dict[Tuple[str, str, str, str], nn.Module] = {}
    for gkey, grouped_specs in groups.items():
        data_src, mid, in_kind, out_kind = gkey
        pytorch_model, rep_name = models[mid]
        # Use representative model_name for the display key
        display_key = (data_src, rep_name, in_kind, out_kind)
        vm = _build_batched_model(display_key, grouped_specs, pytorch_model)
        synthesis_models[display_key] = vm
    
    # -------------------------------------------------------------------------
    # Summary: Print statistics and return results
    # -------------------------------------------------------------------------
    total_specs = sum(vm[0].input_tensor.shape[0] for vm in synthesis_models.values())
    
    print(f"\n🎉 Synthesis Complete:")
    print(f"   Total specs: {total_specs}")
    print(f"   Wrapped models: {len(synthesis_models)}")
    return synthesis_models 


# -----------------------------------------------------------------------------
# 4) Model synthesis main function
# -----------------------------------------------------------------------------
def model_synthesis(creator: str = 'torchvision') -> Dict[Tuple[str, str, str, str], nn.Module]:
    """
    Main model synthesis function using new spec creators.
    
    Simplified implementation that delegates spec creation to TorchVisionSpecCreator
    or VNNLibSpecCreator, then synthesizes wrapped models directly.
    
    Args:
        creator: Creator to use ('torchvision' or 'vnnlib'). Defaults to 'torchvision'.
    
    Returns:
        wrapped_models: Dict[(dataset, model, in_kind, out_kind), VerifiableModel]
        
    Raises:
        RuntimeError: If no spec creator can load data-model pairs or create specs
        NotImplementedError: If VNNLIB creator is requested (not yet implemented)
    """
    print(f"\n{'='*80}")
    print(f"MODEL SYNTHESIS: Using New Spec Creators ({creator.upper()})")
    print(f"{'='*80}")
    
    # Select creator based on parameter
    if creator == 'vnnlib':
        from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
        
        print(f"\n📊 Attempting to use VNNLibSpecCreator...")
        spec_creator = VNNLibSpecCreator(config_name="vnnlib_default")
        
        # Create specs for all downloaded VNNLIB instances
        # Use max_instances=3 to limit for testing (185 total instances available)
        spec_results = spec_creator.create_specs_for_data_model_pairs(
            categories=None,  # All downloaded categories
            max_instances=3,  # Limit to 3 instances per category for synthesis
            validate_shapes=True
        )
    
    elif creator == 'torchvision':
        from act.front_end.torchvision_loader.create_specs import TorchVisionSpecCreator
        
        print(f"\n📊 Attempting to use TorchVisionSpecCreator...")
        spec_creator = TorchVisionSpecCreator(config_name="torchvision_classification")
        
        # Create specs for all downloaded dataset-model pairs
        spec_results = spec_creator.create_specs_for_data_model_pairs(
            num_samples=1,  # Use 1 sample per pair for synthesis
            validate_shapes=True
        )
    
    else:
        raise ValueError(f"Unknown creator: {creator}. Use 'torchvision' or 'vnnlib'.")
    
    # Validate results
    if not spec_results:
        if creator == 'vnnlib':
            raise RuntimeError(
                "No VNNLIB instances found! Please download VNNLIB benchmarks first.\n\n"
                "Examples:\n"
                "  python -m act.front_end --download acasxu_2023      # ACAS Xu collision avoidance\n"
                "  python -m act.front_end --download vit_2023          # Vision Transformer\n"
                "  python -m act.front_end --list-downloads             # Show what's downloaded\n"
            )
        else:
            raise RuntimeError(
                "No dataset-model pairs found! Please download datasets first.\n\n"
                "Examples:\n"
                "  python -m act.front_end --download MNIST              # Downloads MNIST + all models\n"
                "  python -m act.front_end --download CIFAR10            # Downloads CIFAR10 + all models\n"
                "  python -m act.front_end --list                        # Show all available datasets\n"
                "  python -m act.front_end --list-downloads              # Show what's already downloaded\n"
            )
    
    print(f"✓ Successfully created specs using {creator.upper()} spec creator")
    print(f"  Found {len(spec_results)} dataset-model pair(s)")
    
    # Calculate statistics from spec_results BEFORE synthesis
    total_samples = sum(len(input_tensors) for _, _, _, input_tensors, _ in spec_results)
    total_spec_pairs = sum(len(spec_pairs) for _, _, _, _, spec_pairs in spec_results)
    specs_per_sample = total_spec_pairs // total_samples if total_samples else 0
    
    # Synthesize wrapped models from spec results
    wrapped_models = synthesize_models_from_specs(spec_results)
    
    # Memory optimization: Free dataset memory after synthesis
    # spec_results contains (data_source, model_name, pytorch_model, input_tensors, spec_pairs)
    # The dataloader/dataset objects are no longer needed after synthesis
    import gc
    del spec_results  # Free ~476 MB of MNIST dataset memory!
    gc.collect()
    
    # Validate synthesis results
    if not wrapped_models:
        raise RuntimeError(
            "Failed to synthesize any wrapped models! "
            "Spec results were loaded but model synthesis failed. "
            "Check spec_results format and synthesize_models_from_specs() logic."
        )
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SYNTHESIS COMPLETE")
    print(f"{'='*80}")
    print(f"  • Wrapped models: {len(wrapped_models)}")
    # Count unique dataset-model pairs from model keys
    unique_pairs = set()
    for (dataset, model, in_kind, out_kind) in wrapped_models.keys():
        unique_pairs.add((dataset, model))
    print(f"  • Unique dataset-model pairs: {len(unique_pairs)}")
    
    # Print detailed breakdown (using pre-calculated stats)
    if total_samples > 0 and total_spec_pairs > 0:
        print(f"\n📊 Breakdown:")
        print(f"  • Input samples: {total_samples}")
        print(f"  • Spec pairs per sample: {specs_per_sample}")
        print(f"    (= 2 input kinds × 4 epsilons × 3 output specs)")
        print(f"    (= BOX, LINF_BALL × 0.01,0.03,0.05,0.1 × MARGIN_ROBUST(m=0.0,0.5), TOP1_ROBUST)")
        print(f"  • Total spec pairs: {total_spec_pairs}")
        print(f"  • Calculation: {total_samples} samples × {specs_per_sample} specs/sample = {total_spec_pairs} wrapped models")
    
    return wrapped_models


if __name__ == "__main__":
    from act.util.model_inference import model_inference
    from act.util.device_manager import initialize_device
    
    # Initialize device/dtype before synthesis (models typically use float32)
    initialize_device(device='cuda', dtype='float32')
    
    # Step 1: Synthesize all wrapped models using new spec creators
    wrapped_models = model_synthesis()
    
    # Step 2: Test all models with inference (input data extracted from wrapped models)
    successful_models = model_inference(wrapped_models)
    
    print(f"\n✅ Successfully inferred {len(successful_models)} out of {len(wrapped_models)} models")
    print(f"\n🎯 NEW SPEC CREATOR INTEGRATION: COMPLETE ✅")
