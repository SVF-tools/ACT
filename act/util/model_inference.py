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
from typing import Dict, Any, Optional, List, Tuple, Union

# ------------------- Model Inference Function --------------------------------
# Helper function for single model inference
# -----------------------------------------------------------------------------
def infer_single_model(combo_id: str, model: nn.Module, input_tensor: torch.Tensor) -> Tuple[bool, Optional[torch.Tensor], Optional[str]]:
    """
    Test a single wrapped model with input tensor.
    
    Args:
        combo_id: Model identifier (e.g., "m:mnist_mlp_small|x:mnist")
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
                output = output['output']
            return True, output, None
    except Exception as e:
        return False, None, str(e)[:100]


# Main model inference function
# -----------------------------------------------------------------------------
def model_inference(models: Dict[str, nn.Module]) -> Dict[str, nn.Module]:
    """
    Test all wrapped models with their stored inputs and provide execution statistics.
    
    Args:
        models: Dict[combo_id, nn.Module] - Synthesized wrapped models to test
        
    Returns:
        Dict[combo_id, nn.Module] - Successfully inferred models only
    """
    print(f"\n🔧 Testing {len(models)} models...")
    
    # Handle case where no models were generated
    if not models:
        print("⚠️  No models to test - check spec generation and synthesis configuration")
        return {}
    
    success_count = 0
    failure_count = 0
    correct_predictions = 0
    failure_summary = {}  # Track unique failure types
    successful_models = {}  # Track successfully inferred models
    
    for combo_id, model in models.items():
        # Extract input and label from VerifiableModel
        if not hasattr(model, 'labeled_input') or model.labeled_input is None:
            print(f"⚠️  Model {combo_id} missing labeled_input in VerifiableModel")
            failure_count += 1
            continue
        
        test_input = model.labeled_input.tensor
        test_label = model.labeled_input.label  # Extract ground truth label
        dataset = combo_id.split('|')[1].split(':')[1]  # Extract from x:dataset_name
        model_name = combo_id.split('|')[0].split(':')[1]
        success, output, error_msg = infer_single_model(combo_id, model, test_input)
        
        if success:
            # Validate prediction against ground truth label
            pred_class = output.argmax().item() if output.dim() > 0 else output.item()
            is_correct = (pred_class == test_label)
            
            success_count += 1
            if is_correct:
                correct_predictions += 1
            successful_models[combo_id] = model  # Store successful model
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
        
        # Optional: Add detailed explanation for first failure (can be enabled if needed)
        # if "--verbose" in sys.argv:
        #     first_pattern = list(failure_summary.keys())[0]
        #     model_name, dataset_name = first_pattern.split(" + ")
        #     print(f"\n🔍 DETAILED ANALYSIS (example):")
        #     print(explain_architecture_mismatch(model_name, dataset_name, list(failure_summary.values())[0]['error']))
    
    return successful_models
        
# -----------------------------------------------------------------------------
# Helper functions for user-friendly error explanations
# -----------------------------------------------------------------------------
def extract_shape_info(error_msg: str) -> dict:
    """Extract shape information from error messages for detailed explanations."""
    import re
    
    info = {"input_features": None, "expected_features": None, "input_shape": None}
    
    # Pattern for "mat1 and mat2 shapes cannot be multiplied (1x180 and 245x10)"
    mat_pattern = r"mat1 and mat2 shapes cannot be multiplied \(1x(\d+) and (\d+)x\d+\)"
    match = re.search(mat_pattern, error_msg)
    if match:
        info["input_features"] = int(match.group(1))
        info["expected_features"] = int(match.group(2))
    
    # Pattern for input shape errors
    shape_pattern = r"input\[([^\]]+)\]"
    match = re.search(shape_pattern, error_msg)
    if match:
        info["input_shape"] = match.group(1)
    
    return info


def get_model_architecture_info(model_domain: str) -> dict:
    """Get detailed architecture information for different model domains."""
    arch_info = {
        "mnist": {
            "input_size": "28×28 pixels",
            "channels": "1 (grayscale)",
            "total_pixels": "784 features",
            "architecture": "CNN optimized for handwritten digits",
            "typical_features_before_fc": "196 (after conv/pool layers)"
        },
        "cifar10": {
            "input_size": "32×32 pixels", 
            "channels": "3 (RGB)",
            "total_pixels": "3072 features",
            "architecture": "CNN optimized for natural images",
            "typical_features_before_fc": "245 (after conv/pool layers)"
        },
        "unknown": {
            "input_size": "unknown",
            "channels": "unknown",
            "total_pixels": "unknown",
            "architecture": "unknown architecture",
            "typical_features_before_fc": "unknown"
        }
    }
    return arch_info.get(model_domain, arch_info["unknown"])


def get_domain_info(domain: str) -> str:
    """Get descriptive information about a domain."""
    domain_info = {
        "mnist": "28×28 grayscale handwritten digits",
        "cifar10": "32×32 RGB natural images",
        "unknown": "unknown image format"
    }
    return domain_info.get(domain, "unknown format")


def explain_architecture_mismatch(model_name: str, dataset_name: str, error_msg: str) -> str:
    """Provide concise explanations for architecture mismatches."""
    
    # Extract domains
    model_domain = "mnist" if "mnist" in model_name.lower() else "cifar10" if "cifar10" in model_name.lower() else "unknown"
    data_domain = "mnist" if "mnist" in dataset_name.lower() else "cifar10" if "cifar10" in dataset_name.lower() else "unknown"
    
    # Extract key error info
    import re
    shape_match = re.search(r"mat1 and mat2 shapes cannot be multiplied \(1x(\d+) and (\d+)x\d+\)", error_msg)
    
    if shape_match:
        actual_features = shape_match.group(1)
        expected_features = shape_match.group(2)
        explanation = f"""
🔍 MISMATCH: {model_name} + {dataset_name}
   Model expects {expected_features} features, got {actual_features}
   Cause: {model_domain.upper()} model designed for {get_domain_info(model_domain)}
          {data_domain.upper()} data provides {get_domain_info(data_domain)}
   Fix: Use {model_domain}+{model_domain} combinations for compatibility"""
    else:
        explanation = f"""
🔍 MISMATCH: {model_name} + {dataset_name}
   Architecture incompatibility between {model_domain.upper()} model and {data_domain.upper()} data
   Fix: Use domain-matched combinations"""
    
    return explanation.strip()

