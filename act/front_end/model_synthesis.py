# pyright: reportCallIssue=false, reportArgumentType=false, reportOptionalMemberAccess=false, reportMissingTypeArgument=false, reportIndexIssue=false, reportAttributeAccessIssue=false, reportConstantRedefinition=false
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
    from act.util.format_utils import rule
    print("\n" + rule())
    print("⚠️  ERROR: Cannot run as script due to import conflicts!")
    print("Please run as a module instead:")
    print("  python -m act.front_end.model_synthesis")
    print(rule() + "\n")
    sys.exit(1)

import copy
import torch
import torch.fx as fx
import torch.nn as nn
from typing import Dict, Any, List, Literal, Tuple, Optional

# Import ACT components
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.verifiable_model import (
    InputLayer,
    InputSpecLayer,
    OutputSpecLayer,
    VerifiableModel,
)
from act.util.format_utils import rule


# -----------------------------------------------------------------------------
# 2) Small utilities
# -----------------------------------------------------------------------------
def infer_layout_from_tensor(x: torch.Tensor) -> str:
    """Infer tensor layout (HWC, CHW, or FLAT) from shape."""
    if x.dim() == 3:
        return "EMBEDDING"
    if x.dim() == 4 and x.shape[-1] in (1, 3, 4):
        return "HWC"
    elif x.dim() == 4:
        return "CHW"
    return "FLAT"


def _is_fixed_model_input(labeled: LabeledInputTensor, spec: InputSpec) -> bool:
    """Identify one fixed model tensor carried inside one spec lane."""
    seed = spec.lb if isinstance(spec.lb, torch.Tensor) else spec.center
    return (
        isinstance(seed, torch.Tensor)
        and labeled.tensor.ndim >= 2 and labeled.tensor.shape[0] != 1
        and tuple(seed.shape[1:]) == tuple(labeled.tensor.shape)
    )


def _merge_specs_to_batch(
    lts: List[LabeledInputTensor],
    in_specs: List[InputSpec],
    out_specs: List[OutputSpec],
    in_kind: InKind,
    out_kind: OutKind,
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
    if len(lts) != 1 and any(
        _is_fixed_model_input(labeled, spec) for labeled, spec in zip(lts, in_specs)
    ):
        raise ValueError(
            "Fixed-shape model input cannot be merged into independent verifier lanes"
        )

    # Assert all tensors have the same shape (guaranteed by grouping in gkey)
    if len(lts) > 1:
        first_shape = lts[0].tensor.shape
        assert all(lt.tensor.shape == first_shape for lt in lts), \
            f"Shape mismatch: {[lt.tensor.shape for lt in lts]}"
    
    # Merge input tensors: (1,C,H,W) * N → (N,C,H,W)
    tensor = torch.cat([lt.tensor for lt in lts], dim=0)
    labels = torch.cat([lt.label for lt in lts], dim=0) if all(lt.label is not None for lt in lts) else None
    
    # Merge input specs based on kind
    p_norm: torch.Tensor | None = None
    perturbed_positions: torch.Tensor | None = None
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
        first_center = in_specs[0].center
        assert first_center is not None
        if len(in_specs) > 1:
            first_center_shape = first_center.shape
            assert all(s.center is not None and s.center.shape == first_center_shape for s in in_specs), \
                f"InputSpec.center shape mismatch: {[s.center.shape if s.center is not None else None for s in in_specs]}"
        
        centers: List[torch.Tensor] = []
        lbs: List[torch.Tensor] = []
        ubs: List[torch.Tensor] = []
        eps_values: List[torch.Tensor] = []
        for spec in in_specs:
            assert spec.center is not None and spec.eps is not None
            eps_tensor = spec.eps if isinstance(spec.eps, torch.Tensor) else torch.tensor([float(spec.eps)])
            centers.append(spec.center)
            lbs.append(torch.clamp(spec.center - eps_tensor, 0))
            ubs.append(torch.clamp(spec.center + eps_tensor, 1))
            eps_values.append(eps_tensor.reshape(-1)[0])
        center = torch.cat(centers, dim=0)
        lb = torch.cat(lbs, dim=0)
        ub = torch.cat(ubs, dim=0)
        eps = torch.stack(eps_values)
        eps = eps.reshape(center.shape[0], *([1] * (center.ndim - 1)))
    elif in_kind == InKind.LP_EMBEDDING:
        first_center = in_specs[0].center
        assert first_center is not None
        if len(in_specs) > 1:
            first_center_shape = first_center.shape
            assert all(s.center is not None and s.center.shape == first_center_shape for s in in_specs), \
                f"InputSpec.center shape mismatch: {[s.center.shape if s.center is not None else None for s in in_specs]}"

        centers = []
        lbs = []
        ubs = []
        eps_values = []
        position_masks = []
        p_norm_values = []
        for spec in in_specs:
            assert spec.center is not None and spec.eps is not None
            eps_tensor = spec.eps if isinstance(spec.eps, torch.Tensor) else torch.tensor([float(spec.eps)])
            spec_lb, spec_ub = spec.materialize_box_seed()
            centers.append(spec.center)
            lbs.append(spec_lb)
            ubs.append(spec_ub)
            eps_values.append(eps_tensor.reshape(-1)[0])
            position_masks.append((spec_lb != spec_ub).any(dim=-1))
            p_norm = spec.p_norm if isinstance(spec.p_norm, torch.Tensor) else torch.tensor([float(spec.p_norm)])
            p_norm_values.append(p_norm.reshape(-1)[0])

        center = torch.cat(centers, dim=0)
        lb = torch.cat(lbs, dim=0)
        ub = torch.cat(ubs, dim=0)
        eps = torch.stack(eps_values).reshape(center.shape[0], *([1] * (center.ndim - 1)))
        p_norm = torch.stack(p_norm_values)
        perturbed_positions = torch.cat(position_masks, dim=0)
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
    if in_kind == InKind.LP_EMBEDDING:
        assert p_norm is not None and perturbed_positions is not None
        batched_in = InputSpec(
            kind=in_kind,
            lb=lb,
            ub=ub,
            center=center,
            eps=eps,
            p_norm=p_norm,
            perturbed_positions=perturbed_positions,
        )
    else:
        batched_in = InputSpec(kind=in_kind, lb=lb, ub=ub, center=center, eps=eps)
    def _batch_attr(attr):
        """Batch a field that only exists for certain output spec kinds (e.g. c/d for LINEAR_LE, lb/ub for RANGE). Returns None if this kind doesn't use the field."""
        vals = [getattr(s, attr, None) for s in out_specs]
        return torch.stack(vals) if all(v is not None for v in vals) else None
    if out_kind in (OutKind.UNSAFE_LINEAR, OutKind.LINEAR_LE):
        c_vec = out_specs[0].c
        if c_vec is not None:
            c_ref = c_vec.detach().cpu().reshape(-1).numpy().tobytes()
            assert all(
                s.c is not None
                and s.c.shape == c_vec.shape
                and s.c.detach().cpu().reshape(-1).numpy().tobytes() == c_ref
                for s in out_specs
            ), f"{out_kind.name}: grouped lanes must share one c matrix"
        d_all = [s.d for s in out_specs]
        if (
            len(d_all) > 1
            and all(t is not None for t in d_all)
            and all(t.shape == d_all[0].shape for t in d_all)
            and not all(torch.equal(t, d_all[0]) for t in d_all)
        ):
            d_vec = torch.stack(d_all)
        else:
            d_vec = d_all[0]
    else:
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
    
    vm = VerifiableModel(
        input_layer=InputLayer(
            batched_lt, batched_lt.tensor.shape, batched_lt.tensor.dtype,
            layout=infer_layout_from_tensor(batched_lt.tensor), dataset_name=data_src,
        ),
        input_spec=InputSpecLayer(spec=batched_in),
        model=pytorch_model,
        output_spec=OutputSpecLayer(spec=batched_out),
    )
    # Parameterless ONNX-converted models (e.g. some VNN-COMP graphs that inline
    # constants) have an empty .parameters() iterator — fall back to CPU.
    try:
        model_device = next(pytorch_model.parameters()).device
    except StopIteration:
        model_device = torch.device('cpu')
    vm = vm.to(model_device)
    
    return vm


def synthesize_models_and_seeds_from_specs(
    spec_results: List[Tuple[str, str, nn.Module, List[LabeledInputTensor], List[Tuple[InputSpec, OutputSpec]]]],
    *,
    cd_group: Literal["value", "shape"] = "value",
) -> Tuple[
    Dict[Tuple[str, str, str, str], nn.Module],
    Dict[Tuple[str, str, str, str], List[LabeledInputTensor]],
]:
    """
    Synthesize wrapped models with automatic batching.
    
    Groups specs by (dataset, model, input_kind, output_kind) and creates batched
    VerifiableModel instances. Reduces model count by 80-90% in practice.
    
    Args:
        spec_results: List of (data_source, model_name, pytorch_model, 
                              labeled_tensors, spec_pairs)
    
    Returns:
        A pair containing synthesized models and their spec-row-aligned seeds,
        both keyed by (dataset, model, in_kind, out_kind).
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
    
    for result_index, (data_source, model_name, pytorch_model, labeled_tensors, spec_pairs) in enumerate(spec_results):
        if not labeled_tensors or not spec_pairs:
            continue
        
        # Group by model identity (id(pytorch_model)) instead of model_name
        mid = id(pytorch_model)
        if mid not in models:
            models[mid] = (pytorch_model, model_name)  # keep first name as representative
        sps = len(spec_pairs) // len(labeled_tensors) if labeled_tensors else 1
        
        for idx, (in_spec, out_spec) in enumerate(spec_pairs):
            lt = labeled_tensors[min(idx // sps if sps > 0 else 0, len(labeled_tensors) - 1)]
            cd_sig: Any = None
            if out_spec.kind in (OutKind.UNSAFE_LINEAR, OutKind.LINEAR_LE) and out_spec.c is not None:
                merge_d = (
                    cd_group == "shape" and out_spec.kind is OutKind.UNSAFE_LINEAR
                )
                if out_spec.d is None:
                    d_sig: Any = None
                elif merge_d:
                    d_sig = tuple(out_spec.d.shape)
                else:
                    d_sig = out_spec.d.detach().cpu().reshape(-1).numpy().tobytes()
                cd_sig = (
                    tuple(out_spec.c.shape),
                    out_spec.c.detach().cpu().reshape(-1).numpy().tobytes(),
                    d_sig,
                )
            gkey = (data_source, mid, in_spec.kind, out_spec.kind, cd_sig)
            if _is_fixed_model_input(lt, in_spec):
                gkey += (result_index, idx)
            groups[gkey].append((lt, in_spec, out_spec, f"{data_source}:{model_name}:s{idx}"))
    
    # -------------------------------------------------------------------------
    # Synthesis Loop: Build batched models from grouped specs
    # -------------------------------------------------------------------------
    synthesis_models: Dict[Tuple[str, str, str, str], nn.Module] = {}
    synthesis_seeds: Dict[
        Tuple[str, str, str, str], List[LabeledInputTensor]
    ] = {}
    disjunct_counter: Dict[Tuple[str, str, str, str], int] = defaultdict(int)
    for gkey, grouped_specs in groups.items():
        data_src, mid, in_kind, out_kind, _cd_sig = gkey[:5]
        pytorch_model, rep_name = models[mid]
        # Use representative model_name for the display key; if a single
        # (data_src, model_name) expands into multiple UNSAFE_LINEAR disjuncts
        # (e.g. ACAS Xu prop_10), suffix the disjunct index so display keys stay unique.
        base_key = (data_src, rep_name, in_kind, out_kind)
        idx = disjunct_counter[base_key]
        disjunct_counter[base_key] = idx + 1
        if idx == 0:
            display_key = base_key
        else:
            display_key = (data_src, f"{rep_name}#d{idx}", in_kind, out_kind)
        vm = _build_batched_model(display_key, grouped_specs, pytorch_model)
        synthesis_models[display_key] = vm
        synthesis_seeds[display_key] = [item[0] for item in grouped_specs]
    
    # -------------------------------------------------------------------------
    # Summary: Print statistics and return results
    # -------------------------------------------------------------------------
    total_specs = sum(int(vm.input_spec.spec.lb.shape[0]) for vm in synthesis_models.values())
    
    print(f"\n🎉 Synthesis Complete:")
    print(f"   Total specs: {total_specs}")
    print(f"   Wrapped models: {len(synthesis_models)}")
    return synthesis_models, synthesis_seeds


def synthesize_models_from_specs(
    spec_results: List[Tuple[str, str, nn.Module, List[LabeledInputTensor], List[Tuple[InputSpec, OutputSpec]]]],
    *,
    cd_group: Literal["value", "shape"] = "value",
) -> Dict[Tuple[str, str, str, str], nn.Module]:
    """Synthesize wrapped models with automatic batching."""
    synthesis_models, _ = synthesize_models_and_seeds_from_specs(
        spec_results, cd_group=cd_group
    )
    return synthesis_models


# -----------------------------------------------------------------------------
# 4) Model synthesis main function
# -----------------------------------------------------------------------------
def model_synthesis(
    creator: str = 'torchvision',
    spec_overrides: Optional[Dict[str, Any]] = None,
    text_verification_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[Tuple[str, str, str, str], nn.Module]:
    """
    Main model synthesis function using new spec creators.
    
    Simplified implementation that delegates spec creation to TorchVisionSpecCreator
    or VNNLibSpecCreator, then synthesizes wrapped models directly.
    
    Args:
        creator: Creator to use ('torchvision', 'vnnlib', or 'bert'). Defaults to 'torchvision'.
        spec_overrides: Runtime spec configuration overrides.
        text_verification_overrides: Runtime text verification overrides for BERT.
    
    Returns:
        wrapped_models: Dict[(dataset, model, in_kind, out_kind), VerifiableModel]
        
    Raises:
        RuntimeError: If no spec creator can load data-model pairs or create specs
        NotImplementedError: If VNNLIB creator is requested (not yet implemented)
    """
    print(f"\n{rule()}")
    print(f"MODEL SYNTHESIS: Using New Spec Creators ({creator.upper()})")
    print(f"{rule()}")
    
    # Select creator based on parameter
    if creator == 'vnnlib':
        from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
        
        print(f"\n📊 Attempting to use VNNLibSpecCreator...")
        spec_creator = VNNLibSpecCreator(
            config_name="vnnlib_default",
            config_dict=spec_overrides,
        )
        
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
        spec_creator = TorchVisionSpecCreator(
            config_name="torchvision_classification",
            config_dict=spec_overrides,
        )
        
        # Create specs for all downloaded dataset-model pairs
        spec_results = spec_creator.create_specs_for_data_model_pairs(
            num_samples=1,  # Use 1 sample per pair for synthesis
            validate_shapes=True
        )
    
    elif creator == 'bert':
        from act.front_end.bert_loader.create_specs import BertSpecCreator
        from act.config.config import FrontEndConfig

        print(f"\n📊 Attempting to use BertSpecCreator...")
        front_end_config = FrontEndConfig.from_yaml(**(text_verification_overrides or {}))
        text_config = front_end_config.text_verification
        creator_overrides: Dict[str, Any] = {}
        if spec_overrides:
            creator_overrides.update(spec_overrides)
        creator_overrides.update(text_config)
        spec_creator = BertSpecCreator(config_dict=creator_overrides)
        spec_results = spec_creator.create_specs_for_data_model_pairs(
            num_samples=1,
            validate_shapes=True,
            epsilon=float(text_config['eps']),
            p_norm=float(text_config['p']),
            perturbed_words=int(text_config['perturbed_words']),
        )

    else:
        raise ValueError(f"Unknown creator: {creator}. Use 'torchvision', 'vnnlib', or 'bert'.")
    
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
    print(f"\n{rule()}")
    print(f"SYNTHESIS COMPLETE")
    print(f"{rule()}")
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


def _smoke_compact_bert() -> None:
    """Check nonconstant compact routes, indexed queries and CLI Net round trips."""
    import hashlib
    from importlib import import_module
    import json
    from pathlib import Path
    import sys
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    from transformers import BertConfig, BertForSequenceClassification

    from act.back_end.serialization import load_net_from_file, save_net_to_file
    from act.front_end.bert_loader.compact_bert import (
        CompactBertRouteBModel, NoVarLayerNorm, embedding_sum,
        load_compact_bert, tokenize_act_tokens,
    )
    from act.front_end.bert_loader.create_specs import BertSpecCreator
    from act.util import path_config
    from act.util.device_manager import get_current_settings, initialize_device
    from act.util.stats import VerifyStatus

    # These clients import synthesis themselves; load them only for this smoke.
    backend_cli = import_module("act.config.backend_cli")
    frontend_cli = import_module("act.config.frontend_cli")
    query_index = import_module("act.front_end.bert_loader.query_index")
    verify_once = import_module("act.back_end.verifier").verify_once
    ACTToTorch = import_module("act.pipeline.verification.act2torch").ACTToTorch
    TorchToACT = import_module("act.pipeline.verification.torch2act").TorchToACT

    initialize_device(device="cpu", dtype="float64")
    device, dtype = get_current_settings()
    torch.manual_seed(42)
    with TemporaryDirectory(prefix="act-compact-smoke-") as directory:
        root = Path(directory)
        with patch.object(path_config, "get_project_root", return_value=str(root)):
            model_dir = Path(path_config.get_bert_artifact_root()) / "models" / "model_sst_1"
            checkpoint = model_dir / "ckpt-7"
            checkpoint.mkdir(parents=True)
            config = BertConfig(
                vocab_size=10, hidden_size=8, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=16,
                max_position_embeddings=32, type_vocab_size=2,
                hidden_act="relu", num_labels=2,
            )
            source = BertForSequenceClassification(config).to(device=device, dtype=dtype)
            source.bert.embeddings.LayerNorm = NoVarLayerNorm(config.hidden_size).to(
                device=device, dtype=dtype
            )
            for block in source.bert.encoder.layer:
                block.attention.output.LayerNorm = NoVarLayerNorm(config.hidden_size).to(
                    device=device, dtype=dtype
                )
                block.output.LayerNorm = NoVarLayerNorm(config.hidden_size).to(
                    device=device, dtype=dtype
                )
            with torch.no_grad():
                source.classifier.bias.copy_(source.classifier.bias.new_tensor([-1.0, 1.0]))
            source.eval()
            (model_dir / "checkpoint").write_text("7\n", encoding="utf-8")
            (checkpoint / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
            (checkpoint / "vocab.txt").write_text(
                "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\nplay\n##ing\n!\n",
                encoding="utf-8",
            )
            weights = checkpoint / "pytorch_model.bin"
            torch.save(source.state_dict(), weights)
            raw_sst = Path(path_config.get_data_root()) / "sst"
            raw_sst.mkdir()
            (raw_sst / "test.txt").write_text(
                "(4 (4 hello) (4 playing) (4 world) (4 hello) (4 world) (4 hello))\n",
                encoding="utf-8",
            )
            checksum = hashlib.sha256(weights.read_bytes()).hexdigest()
            relative_weights = weights.relative_to(path_config.get_data_root()).as_posix()
            (Path(path_config.get_bert_artifact_root()) / "SHA256SUMS").write_text(
                f"{checksum}  {relative_weights}\n", encoding="utf-8"
            )
            index = root / "queries.csv"
            index.write_text(
                "dataset,split,depth,example_id,position,p,eps,checkpoint_sha256\n"
                f"sst,test,1,0,1,inf,0.001,{checksum}\n"
                "sst,test,1,0,3,1,0.001,\n",
                encoding="utf-8",
            )
            relative_model = model_dir.relative_to(path_config.get_data_root())
            loaded = load_compact_bert(relative_model)
            tokens = tokenize_act_tokens(
                ["hello", "playing", "world", "hello", "world", "hello"], loaded.tokenizer
            )
            assert tokens.wordpiece_tokens == [
                "[CLS]", "hello", "play", "##ing", "world", "hello", "world", "hello", "[SEP]",
            ]
            seed = embedding_sum(loaded, tokens.input_ids, tokens.token_type_ids)
            changed = seed.clone()
            changed[:, 0, 0] += 0.25
            route_b = CompactBertRouteBModel(loaded.model, seed.shape[1])
            with torch.no_grad():
                expected = loaded.model(seed)
                changed_expected = loaded.model(changed)
                assert (changed_expected - expected).abs().max() > 1e-7
                torch.testing.assert_close(
                    source(input_ids=tokens.input_ids.unsqueeze(0)).logits, expected,
                    rtol=1e-9, atol=1e-10,
                )
                for inputs in (seed, changed):
                    torch.testing.assert_close(route_b(inputs), loaded.model(inputs), rtol=1e-9, atol=1e-10)

            creator = BertSpecCreator(config_dict={"checkpoint_dir": str(relative_model)})
            with patch("act.front_end.bert_loader.create_specs.BertVocabulary") as vocabulary:
                for route in ("a", "b"):
                    results = creator.create_specs_for_data_model_pairs(conversion_route=route)
                    assert results[0][1] == "compact_bert"
                    assert results[0][4][0][0].perturbed_positions.tolist() == [0]
                    (wrapped,) = synthesize_models_from_specs(results).values()
                    net = TorchToACT(wrapped).run()
                    path = root / f"route-{route}.json"
                    save_net_to_file(net, str(path))
                    for candidate in (net, load_net_from_file(str(path))):
                        restored = ACTToTorch(candidate).run().eval()
                        for inputs in (seed, changed):
                            torch.testing.assert_close(
                                restored(inputs)["output"], loaded.model(inputs),
                                rtol=1e-8, atol=1e-9,
                            )
                vocabulary.assert_not_called()

            query = query_index.read_query(index, 0)
            assert query_index.verify_checkpoint_sha256(query) == weights
            with patch.object(
                BertSpecCreator, "_single_spec_pair", wraps=BertSpecCreator._single_spec_pair
            ) as single:
                wrapped_query = query_index.build_query_model(query)
                assert single.call_count == 1
            mask = wrapped_query.input_spec.spec.perturbed_positions
            assert mask.shape == seed.shape[:2]
            assert mask[0].nonzero().flatten().tolist() == [1]
            try:
                query_index.build_query_model(query_index.read_query(index, 1))
            except query_index.QueryIndexError:
                pass
            else:
                raise AssertionError("WordPiece continuation accepted as a query position")
            partition = query_index.yelp_partition_indices(10)
            assert len(partition["calibration"]) == 2
            assert sorted(partition["calibration"] + partition["evaluation"]) == list(range(10))
            assert not set(partition["calibration"]) & set(partition["evaluation"])

            with patch.object(sys, "argv", [
                "act.front_end", "--synthesis", "--creator", "bert",
                "--checkpoint-dir", str(relative_model),
                "--device", "cpu", "--dtype", "float64",
            ]):
                assert frontend_cli.main() in (None, 0)
            saved = root / "verified.json"
            for source_args in (
                ["--query-index", str(index), "--query-id", "0", "--save-net", str(saved)],
                ["--network", str(saved)],
            ):
                with patch.object(sys, "argv", [
                    "act.back_end", "--verify", *source_args, "--solver", "torchlp",
                    "--no-bab", "--timeout", "30", "--device", "cpu", "--dtype", "float64",
                ]):
                    assert backend_cli.main() == 0
            restored_net = load_net_from_file(str(saved))
            restored_query = ACTToTorch(restored_net).run().eval()
            for inputs in (seed, changed):
                torch.testing.assert_close(
                    restored_query(inputs)["output"], loaded.model(inputs), rtol=1e-8, atol=1e-9,
                )
            assert all(result.status == VerifyStatus.CERTIFIED for result in verify_once(restored_net))
    print("Compact BERT A/B/ACT/reload and query CLI smoke passed")


if __name__ == "__main__":
    import sys

    if "--compact-smoke" in sys.argv:
        _smoke_compact_bert()
        sys.exit(0)

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


# Merge split-ReLUs: invert the ReluSplitter benchmark transformation.
#
# ReluSplitter rewrites a linear pre-activation  z = W_orig x + b_orig  into a
# DENSE -> ReLU -> DENSE "sandwich" that computes the SAME affine map by
# exploiting  a = ReLU(a) - ReLU(-a): each base row (w,b) is emitted as an
# anti-parallel pair (+(w,b), -(w,b)) in the first DENSE, and the second DENSE
# recombines the pair with opposite-sign weights so its output is again the
# linear z. The spurious (always-unstable) ReLUs only loosen the dual
# relaxation, leaving pct>=0.4 instances "unknown". This pass detects any
# DENSE -> ReLU -> DENSE sandwich that is PROVABLY a global affine map and
# collapses it back to a single DENSE -- exactly semantics-preserving. It is a
# strict no-op unless global affinity is certified, so genuine ReLU layers
# (e.g. ACAS Xu) are left untouched.
#
# Exact soundness certificate (over R^d): group the first DENSE's rows by their
# AUGMENTED direction (w_i | b_i); rows n in a group satisfy a_n = t_n * a_rep.
# With ReLU(t*a)=t*ReLU(a) (t>0) and ReLU(t*a)=|t|*(ReLU(a)-a) (t<0), output k
# receives  R_g[k]*ReLU(a_rep) + L_g[k]*a_rep, where R_g[k]=sum_n W2[k,n]|t_n|.
# The sandwich is affine  <=>  R_g[k]==0 for every group g and output k. The
# collapsed map is  M[k]=sum_g L_g[k]*w_rep_g,  m[k]=b2[k]+sum_g L_g[k]*b_rep_g
# with  L_g[k] = -sum_{n: t_n<0} W2[k,n]|t_n|  (closed form, exact).



# grouping tolerance on |cos| between augmented rows (exact copies give |cos|==1)
_MERGE_RTOL = 1e-6
# affinity certificate threshold: R is exactly 0 for a true split, O(0.1..1) for
# a genuine ReLU layer, so this cleanly separates them while tolerating ULP.
_MERGE_AFFINE_TOL = 1e-8


def _iter_dense_relu_dense(gm: fx.GraphModule):
    """Yield (l1_node, relu_node, l2_node) for sole-consumer Linear->ReLU->Linear chains.

    Only fires when the ReLU output feeds nothing but the next Linear (and the
    first Linear feeds nothing but the ReLU), so merging cannot affect any other
    consumer of the intermediate activations.
    """
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op != "call_module" or not isinstance(modules.get(node.target), nn.Linear):
            continue
        if len(node.users) != 1:
            continue
        relu = next(iter(node.users))
        if relu.op != "call_module" or not isinstance(modules.get(relu.target), nn.ReLU):
            continue
        if len(relu.users) != 1 or len(relu.args) != 1 or relu.args[0] is not node:
            continue
        l2 = next(iter(relu.users))
        if l2.op != "call_module" or not isinstance(modules.get(l2.target), nn.Linear):
            continue
        if not l2.args or l2.args[0] is not relu:
            continue
        yield node, relu, l2


def _certify_affine_collapse(l1: nn.Linear, l2: nn.Linear):
    """Return (M, m, n_merged) in float64 if l1->ReLU->l2 is a global affine map, else None.

    n_merged = number of rows removed = sum over augmented-direction groups of
    (group_size - 1). Computation is in float64; the caller casts to model dtype.
    """
    dev = l1.weight.device
    W1 = l1.weight.detach().double()
    out1, in1 = W1.shape
    b1 = l1.bias.detach().double() if l1.bias is not None else torch.zeros(out1, dtype=torch.float64, device=dev)
    W2 = l2.weight.detach().double()
    out2 = W2.shape[0]
    b2 = l2.bias.detach().double() if l2.bias is not None else torch.zeros(out2, dtype=torch.float64, device=dev)

    aug = torch.cat([W1, b1.unsqueeze(1)], dim=1)
    norms = aug.norm(dim=1)
    unit = aug / norms.clamp_min(1e-30).unsqueeze(1)

    visited = [False] * out1
    groups: list[list[tuple[int, float]]] = []
    for i in range(out1):
        if visited[i]:
            continue
        visited[i] = True
        grp = [(i, 1.0)]
        if norms[i] > 1e-30:
            ui = unit[i]
            for j in range(i + 1, out1):
                if visited[j] or norms[j] <= 1e-30:
                    continue
                if abs(abs(float(ui @ unit[j])) - 1.0) <= _MERGE_RTOL:
                    t = float(aug[j] @ aug[i] / (aug[i] @ aug[i]))
                    grp.append((j, t))
                    visited[j] = True
        groups.append(grp)

    thr = _MERGE_AFFINE_TOL * max(1.0, float(W2.abs().max()) if W2.numel() else 1.0)
    M = torch.zeros(out2, in1, dtype=torch.float64, device=dev)
    m = b2.clone()
    n_merged = 0
    for grp in groups:
        idx = [n for n, _ in grp]
        ts = torch.tensor([t for _, t in grp], dtype=torch.float64, device=dev)
        cols = W2[:, idx]
        relu_coeff = (cols * ts.abs().unsqueeze(0)).sum(dim=1)
        if float(relu_coeff.abs().max()) > thr:
            return None
        neg = ts < 0
        if bool(neg.any()):
            lin_coeff = -(cols[:, neg] * ts[neg].abs().unsqueeze(0)).sum(dim=1)
            rep = grp[0][0]
            M += torch.outer(lin_coeff, W1[rep])
            m += lin_coeff * b1[rep]
        n_merged += len(grp) - 1

    return None if n_merged == 0 else (M, m, n_merged)


def _splice_affine(gm: fx.GraphModule, l1_node, relu_node, l2_node, M, m) -> None:
    """Replace the sandwich with a single Linear: reuse l1's node (rewrite its
    weights), rewire l2's consumers to l1, and erase the dead ReLU + l2 nodes."""
    l1 = dict(gm.named_modules())[l1_node.target]
    dtype, dev = l1.weight.dtype, l1.weight.device
    l1.weight = nn.Parameter(M.to(dtype=dtype, device=dev), requires_grad=False)
    l1.bias = nn.Parameter(m.to(dtype=dtype, device=dev), requires_grad=False)
    l1.out_features, l1.in_features = int(M.shape[0]), int(M.shape[1])
    for consumer in list(l2_node.users):
        consumer.replace_input_with(l2_node, l1_node)
    gm.graph.erase_node(l2_node)
    gm.graph.erase_node(relu_node)
    gm.graph.lint()
    gm.recompile()


def merge_split_relus(model: nn.Module):
    """Collapse every provably-affine DENSE->ReLU->DENSE sandwich into one DENSE.

    Returns (merged_model, n_merged). The input ``model`` is never mutated (work
    happens on a deepcopy); when nothing is merged the ORIGINAL object is
    returned so callers can keep using it unchanged.
    """
    if not isinstance(model, fx.GraphModule):
        return model, 0
    gm = copy.deepcopy(model)
    total = 0
    while True:
        for l1_node, relu_node, l2_node in _iter_dense_relu_dense(gm):
            mods = dict(gm.named_modules())
            certified = _certify_affine_collapse(mods[l1_node.target], mods[l2_node.target])
            if certified is None:
                continue
            M, m, n = certified
            _splice_affine(gm, l1_node, relu_node, l2_node, M, m)
            total += n
            break
        else:
            break
    return (gm, total) if total else (model, 0)
