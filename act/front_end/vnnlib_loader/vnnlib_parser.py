#===- act/front_end/vnnlib/vnnlib_parser.py - VNNLIB Parser ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Parse VNNLIB SMT-LIB format files to extract input tensors and constraints.
#   Converts VNNLIB specifications to InputSpec and OutputSpec objects.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import logging
import torch
import re

from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind

if TYPE_CHECKING:
    from act.front_end.spec_creator_base import LabeledInputTensor

logger = logging.getLogger(__name__)


class VNNLibParseError(Exception):
    """Exception raised when VNNLIB parsing fails."""
    pass


def parse_vnnlib_to_tensors(
    vnnlib_path: Path,
    input_shape: Optional[Tuple[int, ...]] = None
) -> Tuple[torch.Tensor, Dict[str, any]]:
    """
    Parse a VNNLIB file to extract input tensor and metadata.
    
    The input tensor represents the center of the constrained input region.
    For box constraints with bounds [lb, ub], the center is (lb + ub) / 2.
    
    Args:
        vnnlib_path: Path to .vnnlib file
        input_shape: Expected input shape INCLUDING batch dimension (e.g., (1, 3, 32, 32))
                    If None, will be inferred or use flat shape
        
    Returns:
        Tuple of (input_tensor, metadata_dict) where:
        - input_tensor: torch.Tensor with batch dimension (e.g., shape (1, 3, 32, 32))
        - metadata_dict: Contains 'input_bounds', 'num_outputs', 'property_type'
        
    Raises:
        VNNLibParseError: If parsing fails
    """
    if not vnnlib_path.exists():
        raise VNNLibParseError(f"VNNLIB file not found: {vnnlib_path}")
    
    try:
        with open(vnnlib_path, 'r') as f:
            content = f.read()
        
        # Extract variable declarations to determine shapes
        num_inputs = _extract_num_inputs(content)
        num_outputs = _extract_num_outputs(content)
        
        # Extract input bounds (X_i constraints)
        input_bounds = _extract_input_bounds(content, num_inputs)
        
        # Create input tensor from bounds center
        input_values = []
        for i in range(num_inputs):
            if i in input_bounds:
                lb, ub = input_bounds[i]
                center = (lb + ub) / 2.0
            else:
                # Default to 0 if no constraint
                center = 0.0
            input_values.append(center)
        
        input_tensor = torch.tensor(input_values)
        
        # Reshape if shape provided (shape now includes batch dimension)
        if input_shape is not None:
            expected_numel = 1
            for dim in input_shape:
                expected_numel *= dim
            if input_tensor.numel() != expected_numel:
                raise VNNLibParseError(
                    f"Input size mismatch: got {input_tensor.numel()} "
                    f"values but expected {expected_numel} from shape {input_shape}"
                )
            # Reshape directly - input_shape already includes batch dimension
            input_tensor = input_tensor.view(*input_shape)
        
        # Infer property type
        property_type = _infer_property_type(content, num_outputs)
        
        metadata = {
            'input_bounds': input_bounds,
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'property_type': property_type,
            'vnnlib_path': str(vnnlib_path)
        }
        
        logger.info(
            f"Parsed VNNLIB: {num_inputs} inputs, {num_outputs} outputs, "
            f"type={property_type}"
        )
        
        return input_tensor, metadata
        
    except Exception as e:
        raise VNNLibParseError(f"Failed to parse {vnnlib_path}: {str(e)}")


def parse_vnnlib_to_specs(
    vnnlib_path: Path,
    labeled_tensor: Optional['LabeledInputTensor'] = None
) -> Tuple[InputSpec, OutputSpec]:
    """
    Parse VNNLIB file to create InputSpec and OutputSpec objects.
    
    Args:
        vnnlib_path: Path to .vnnlib file
        labeled_tensor: LabeledInputTensor containing input tensor and ground truth label.
                       If provided, tensor.shape is used for reshaping bounds and
                       label is used to promote RANGE to TOP1_ROBUST for classification.
        
    Returns:
        Tuple of (InputSpec, OutputSpec)
        
    Raises:
        VNNLibParseError: If parsing fails
    """
    try:
        with open(vnnlib_path, 'r') as f:
            content = f.read()
        
        num_inputs = _extract_num_inputs(content)
        num_outputs = _extract_num_outputs(content)
        input_bounds = _extract_input_bounds(content, num_inputs)
        
        # Create InputSpec (BOX constraints)
        lb_values = []
        ub_values = []
        for i in range(num_inputs):
            if i in input_bounds:
                lb, ub = input_bounds[i]
            else:
                lb, ub = float('-inf'), float('inf')
            lb_values.append(lb)
            ub_values.append(ub)
        
        lb_tensor = torch.tensor(lb_values)
        ub_tensor = torch.tensor(ub_values)
        
        # Extract input_shape and true_label from labeled_tensor if provided
        input_shape = labeled_tensor.tensor.shape if labeled_tensor is not None else None
        true_label = labeled_tensor.label if labeled_tensor is not None else None
        
        if input_shape is not None:
            # Reshape bounds to match input shape (which now includes batch dimension)
            lb_tensor = lb_tensor.view(*input_shape)
            ub_tensor = ub_tensor.view(*input_shape)
        
        input_spec = InputSpec(
            kind=InKind.BOX,
            lb=lb_tensor,
            ub=ub_tensor
        )
        
        # Create OutputSpec (LINEAR_LE constraints)
        output_constraints = _extract_output_constraints(content, num_outputs)
        
        if output_constraints:
            # Use first constraint as representative
            c, d = output_constraints[0]
            output_spec = OutputSpec(
                kind=OutKind.LINEAR_LE,
                c=torch.tensor(c),
                d=torch.tensor(float(d)),
            )
        else:
            # If true_label is provided, promote to TOP1_ROBUST
            # for classification robustness properties. Otherwise, use RANGE.
            if true_label is not None:
                # true_label is already a tensor with correct device from labeled_tensor
                output_spec = OutputSpec(
                    kind=OutKind.TOP1_ROBUST,
                    y_true=true_label.clone() if isinstance(true_label, torch.Tensor) else torch.tensor([int(true_label)], dtype=torch.int64),
                )
            else:
                output_spec = OutputSpec(
                    kind=OutKind.RANGE,
                    lb=torch.tensor([float('-inf')] * num_outputs),
                    ub=torch.tensor([float('inf')] * num_outputs)
                )
        
        logger.info(f"Created specs from VNNLIB: {input_spec.kind}, {output_spec.kind}")
        
        return input_spec, output_spec
        
    except Exception as e:
        raise VNNLibParseError(f"Failed to create specs from {vnnlib_path}: {str(e)}")


def _extract_num_inputs(content: str) -> int:
    """
    Extract number of input variables from VNNLIB content.
    
    Looks for patterns like:
    - (declare-const X_0 Real)
    - (declare-const X_1 Real)
    """
    x_vars = set()
    # Match X_<number>
    pattern = r'X_(\d+)'
    matches = re.findall(pattern, content)
    for match in matches:
        x_vars.add(int(match))
    
    if not x_vars:
        raise VNNLibParseError("No input variables (X_i) found")
    
    # Number of inputs is max index + 1 (assuming 0-indexed)
    return max(x_vars) + 1


def _extract_num_outputs(content: str) -> int:
    """
    Extract number of output variables from VNNLIB content.
    
    Looks for patterns like:
    - (declare-const Y_0 Real)
    - (declare-const Y_1 Real)
    """
    y_vars = set()
    # Match Y_<number>
    pattern = r'Y_(\d+)'
    matches = re.findall(pattern, content)
    for match in matches:
        y_vars.add(int(match))
    
    if not y_vars:
        logger.warning("No output variables (Y_i) found in VNNLIB")
        return 0
    
    return max(y_vars) + 1


def _extract_input_bounds(content: str, num_inputs: int) -> Dict[int, Tuple[float, float]]:
    """
    Extract lower and upper bounds for each input variable.
    
    Parses constraints like:
    - (assert (>= X_0 0.5))  -> lower bound
    - (assert (<= X_0 1.5))  -> upper bound
    
    Returns:
        Dict mapping input index to (lower_bound, upper_bound)
    """
    bounds = {}
    
    # Initialize with infinity bounds
    for i in range(num_inputs):
        bounds[i] = [float('-inf'), float('inf')]
    
    # Match lower bounds: (>= X_i value) or (>= value X_i)
    # Pattern 1: (>= X_i value) - groups[0]=idx, groups[1]=value
    # Pattern 2: (>= value X_i) - groups[0]=value, groups[1]=idx
    lb_pattern_1 = r'\(>=\s+X_(\d+)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)'
    lb_pattern_2 = r'\(>=\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+X_(\d+)\)'
    
    # Pattern 1: X_i comes first
    for match in re.finditer(lb_pattern_1, content):
        idx = int(match.group(1))  # Index
        lb = float(match.group(2))  # Value
        if idx < num_inputs:
            bounds[idx][0] = max(bounds[idx][0], lb)
    
    # Pattern 2: Value comes first
    for match in re.finditer(lb_pattern_2, content):
        lb = float(match.group(1))  # Value
        idx = int(match.group(2))  # Index
        if idx < num_inputs:
            bounds[idx][0] = max(bounds[idx][0], lb)
    
    # Match upper bounds: (<= X_i value) or (<= value X_i)
    # Pattern 1: (<= X_i value) - groups[0]=idx, groups[1]=value
    # Pattern 2: (<= value X_i) - groups[0]=value, groups[1]=idx
    ub_pattern_1 = r'\(<=\s+X_(\d+)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)'
    ub_pattern_2 = r'\(<=\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+X_(\d+)\)'
    
    # Pattern 1: X_i comes first
    for match in re.finditer(ub_pattern_1, content):
        idx = int(match.group(1))  # Index
        ub = float(match.group(2))  # Value
        if idx < num_inputs:
            bounds[idx][1] = min(bounds[idx][1], ub)
    
    # Pattern 2: Value comes first
    for match in re.finditer(ub_pattern_2, content):
        ub = float(match.group(1))  # Value
        idx = int(match.group(2))  # Index
        if idx < num_inputs:
            bounds[idx][1] = min(bounds[idx][1], ub)
    
    # Convert to tuples and filter infinite bounds
    result = {}
    for i, (lb, ub) in bounds.items():
        if lb != float('-inf') or ub != float('inf'):
            result[i] = (lb, ub)
    
    return result


def _extract_output_constraints(content: str, num_outputs: int) -> List[Tuple[List[float], float]]:
    """
    Extract output constraints (linear combinations of Y_i).
    
    Returns list of (coefficients, bias) tuples representing c^T * y <= d.
    """
    constraints = []
    
    # This is a simplified parser for common patterns
    # Full VNNLIB can have complex nested assertions
    
    # Match patterns like: (<= (+ (* a0 Y_0) (* a1 Y_1) ...) d)
    # This would require more sophisticated parsing for general VNNLIB
    
    # For now, return empty list (would need full SMT-LIB parser for complete support)
    logger.debug("Output constraint extraction not fully implemented (requires full SMT-LIB parser)")
    
    return constraints


def _infer_property_type(content: str, num_outputs: int) -> str:
    """
    Infer the property type from VNNLIB content.
    
    Returns:
        One of: 'classification', 'safety', 'unknown'
    """
    content_lower = content.lower()
    
    # Classification properties often involve comparisons between outputs
    if 'y_' in content_lower and num_outputs > 1:
        # Check for patterns like Y_i - Y_j > 0 (classification margin)
        if re.search(r'y_\d+\s*[-]\s*y_\d+', content_lower):
            return 'classification'
    
    # Safety properties typically have output range constraints
    if num_outputs == 1 or 'range' in content_lower:
        return 'safety'
    
    return 'unknown'


def validate_vnnlib_file(vnnlib_path: Path) -> bool:
    """
    Validate that a VNNLIB file is parseable.
    
    Args:
        vnnlib_path: Path to .vnnlib file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        parse_vnnlib_to_tensors(vnnlib_path)
        return True
    except VNNLibParseError as e:
        logger.error(f"VNNLIB validation failed: {e}")
        return False


def list_vnnlib_variables(vnnlib_path: Path) -> Dict[str, int]:
    """
    List all variables declared in a VNNLIB file.
    
    Args:
        vnnlib_path: Path to .vnnlib file
        
    Returns:
        Dict with 'num_inputs' and 'num_outputs'
    """
    try:
        with open(vnnlib_path, 'r') as f:
            content = f.read()
        
        return {
            'num_inputs': _extract_num_inputs(content),
            'num_outputs': _extract_num_outputs(content)
        }
    except Exception as e:
        logger.error(f"Failed to list variables: {e}")
        return {'num_inputs': 0, 'num_outputs': 0}


def extract_label_from_vnnlib(vnnlib_path: Path) -> Optional[int]:
    """
    Extract ground truth label from VNNLIB file comment.
    
    Many VNNLIB files (e.g., CIFAR-100) include ground truth labels in comments:
    ; CIFAR100 property with label: 14.
    
    Args:
        vnnlib_path: Path to .vnnlib file
        
    Returns:
        Ground truth label as integer, or None if not found
        
    Example:
        >>> label = extract_label_from_vnnlib(Path("spec.vnnlib"))
        >>> print(label)
        14
    """
    try:
        with open(vnnlib_path, 'r') as f:
            # Read first few lines (label is typically in first comment)
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                
                # Match patterns like: ; CIFAR100 property with label: 14.
                match = re.search(r'label:\s*(\d+)', line, re.IGNORECASE)
                if match:
                    return int(match.group(1))
        
        return None
    except Exception as e:
        logger.debug(f"Failed to extract label from {vnnlib_path}: {e}")
        return None
