#===- act/front_end/model_inference.py - Model Inference Framework -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Model inference and testing framework. Functions for testing
#   synthesized models, analyzing failures, and providing user-friendly
#   explanations for architecture mismatches in the ACT verification pipeline.
#
#===---------------------------------------------------------------------===#


from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

# ------------------- Model Inference Function --------------------------------
# Helper function for single model inference
# -----------------------------------------------------------------------------
def infer_single_model(combo_id: Union[str, Tuple], model: nn.Module, input_tensor: torch.Tensor) -> Tuple[bool, Optional[torch.Tensor], Optional[str]]:
    """
    Test a single wrapped model with input tensor.
    
    Args:
        combo_id: Model identifier (str or tuple of (dataset, model, in_kind, out_kind))
        model: Synthesized wrapped model to test (may return tensor or dict)
        input_tensor: Input tensor for model inference
        
    Returns:
        Tuple of (success, output, error_msg):
            - success: True if inference succeeded, False otherwise
            - output: Model output tensor if successful, None otherwise
            - error_msg: Error message if failed, None otherwise
    """
    try:
        with torch.no_grad():
            output = model(input_tensor)
            # Extract tensor if model returns dict (VerifiableModel)
            if isinstance(output, dict):
                if 'output' not in output:
                    raise KeyError(
                        f"Model {combo_id!r} returned a dictionary without an 'output' entry"
                    )
                output = output['output']
            return True, output, None
    except Exception as exc:
        raise RuntimeError(f"Model inference failed for {combo_id!r}: {exc}") from exc


# Main model inference function
# -----------------------------------------------------------------------------
def model_inference(models: Dict[Union[str, Tuple], nn.Module]) -> Dict[Union[str, Tuple], nn.Module]:
    """
    Test all wrapped models with their stored inputs and provide execution statistics.
    
    Args:
        models: Dict[combo_id, nn.Module] - Synthesized wrapped models to test
                combo_id can be str or tuple of (dataset, model, in_kind, out_kind)
        
    Returns:
        Dict[combo_id, nn.Module] - Successfully inferred models only
    """
    print(f"\n🔧 Testing {len(models)} models...")
    
    # Handle case where no models were generated
    if not models:
        raise ValueError(
            "No models to test; check spec generation and synthesis configuration"
        )
    
    success_count = 0
    failure_count = 0
    correct_predictions = 0
    failure_summary = {}  # Track unique failure types
    successful_models = {}  # Track successfully inferred models
    
    for combo_id, model in models.items():
        # Extract input and label from InputLayer (named child on VerifiableModel)
        input_layer = model.input_layer
        if not hasattr(input_layer, 'input_tensor'):
            raise AttributeError(
                f"Model {combo_id!r} InputLayer is missing input_tensor"
            )
        
        test_input = input_layer.input_tensor
        test_label = input_layer.label
        
        # Move input to same device as model
        # Parameter-free models (e.g. pure-constant ONNX nets) keep the input's device.
        first_param = next(model.parameters(), None)
        model_device = first_param.device if first_param is not None else test_input.device
        test_input = test_input.to(model_device)
        
        # Parse combo_id: handle tuple keys (dataset, model, in_kind, out_kind)
        if isinstance(combo_id, tuple):
            # New tuple format: (dataset, model, in_kind, out_kind)
            dataset = combo_id[0] if len(combo_id) > 0 else 'unknown'
            model_name = combo_id[1] if len(combo_id) > 1 else 'unknown'
        elif '|' in combo_id:
            # Old string format: m:model_name|x:dataset_name
            dataset = combo_id.split('|')[1].split(':')[1]
            model_name = combo_id.split('|')[0].split(':')[1]
        else:
            # String format: DATASET:MODEL:IN_KIND:OUT_KIND
            parts = combo_id.split(':')
            dataset = parts[0] if len(parts) > 0 else 'unknown'
            model_name = parts[1] if len(parts) > 1 else 'unknown'
        
        success, output, error_msg = infer_single_model(combo_id, model, test_input)
        
        if success:
            # Validate prediction against ground truth label tensor (N,), output is (N, num_classes).
            # Unlabelled or non-classification outputs (e.g. VNNLIB safety nets) have no accuracy.
            is_correct = False
            if test_label is not None and test_label.numel() > 0 and output.dim() >= 2:
                pred_classes = output.argmax(dim=1)  # (N,)
                is_correct = (pred_classes == test_label).all().item()
            
            success_count += 1
            if is_correct:
                correct_predictions += 1
            successful_models[combo_id] = model
        else:
            failure_count += 1
            # Track unique failure patterns
            pattern = f"{model_name.split('_')[0]} + {dataset}"  # e.g., "mnist + cifar10"
            if pattern not in failure_summary:
                failure_summary[pattern] = {'count': 0, 'error': error_msg}
            failure_summary[pattern]['count'] += 1
    
    success_rate = (success_count / len(models)) * 100 if len(models) > 0 else 0
    accuracy = (correct_predictions / success_count * 100) if success_count > 0 else 0
    
    print(f"\n📊 Overall: {success_count}/{len(models)} successful ({success_rate:.1f}%)")
    print(f"🎯 Accuracy: {correct_predictions}/{success_count} correct predictions ({accuracy:.1f}%)")
    
    # Show concise failure analysis
    if failure_summary:
        print(f"\n❌ Failure patterns:")
        for pattern, info in failure_summary.items():
            print(f"   • {pattern}: {info['count']} failures (architecture mismatch)")
        print(f"   💡 Tip: Use domain-matched combinations (mnist+mnist, cifar+cifar) for 100% success")
    
    return successful_models
