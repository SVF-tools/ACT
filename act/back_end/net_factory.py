#===- act/back_end/net_factory.py - YAML-Driven Network Factory ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   YAML-driven network factory for generating ACT example networks with
#   automatic parameter generation for INPUT_SPEC and ASSERT layers.
#
#===---------------------------------------------------------------------===#
#
# ASSERT Layer Specification Guide
# =================================
#
# ASSERT layers define verification properties that the network output must satisfy.
# They are used for spec-free verification where constraints are embedded directly
# in the model, enabling single-call checking:
#
#     results = model(input)  # Returns dict with satisfaction status
#
# Four ASSERT kinds are supported, each with distinct verification semantics:
#
# 1. TOP1_ROBUST (Classification Robustness)
#    -----------------------------------------------
#    Purpose: Verify that the true class has the highest score
#    Verification: argmax(y) == y_true
#    
#    Required meta:
#    - y_true: Index of the ground truth class (int)
#    
#    Use cases:
#    - Adversarial robustness: Ensure predictions remain correct under perturbations
#    - Safety-critical classification: Verify correct class prediction
#    - MNIST/CIFAR robustness benchmarks
#    
#    Expected outcome:
#    - PASS: True class has highest logit/probability
#    - FAIL: Different class has higher score (misclassification)
#    
#    Example:
#    meta:
#      kind: "TOP1_ROBUST"
#      y_true: 7  # Verify output predicts class 7
#
# 2. MARGIN_ROBUST (Classification with Safety Margin)
#    -----------------------------------------------
#    Purpose: Verify true class exceeds others by a safety margin
#    Verification: y[y_true] - max(y[i≠y_true]) >= margin
#    
#    Required meta:
#    - y_true: Index of the ground truth class (int)
#    - margin: Minimum required separation from other classes (float)
#    
#    Use cases:
#    - High-confidence verification: Ensure robust predictions with buffer
#    - Safety margins for critical applications
#    - Confidence-based filtering
#    
#    Expected outcome:
#    - PASS: True class exceeds others by at least margin
#    - FAIL: Margin too small (weak confidence) or misclassification
#    
#    Example:
#    meta:
#      kind: "MARGIN_ROBUST"
#      y_true: 3
#      margin: 0.5  # Require 0.5 separation from other classes
#
# 3. LINEAR_LE (Linear Inequality Constraint)
#    -----------------------------------------------
#    Purpose: Verify linear combination of outputs satisfies inequality
#    Verification: c^T · y <= d
#    
#    Required params:
#    - c: Coefficient vector (list/tensor, shape matches output)
#    - d: Threshold scalar (float)
#    
#    Use cases:
#    - Control systems: Verify output stays within operational limits
#    - Resource constraints: Total output bounded (e.g., sum of activations)
#    - Custom safety properties: Linear combination constraints
#    - Reachability analysis: Verify state space boundaries
#    
#    Expected outcome:
#    - PASS: c^T · y <= d (constraint satisfied)
#    - FAIL: c^T · y > d (constraint violated)
#    
#    Example (verify sum of outputs ≤ 5.0):
#    params:
#      c: [1.0, 1.0, 1.0, 1.0, 1.0]  # Sum all 5 outputs
#    meta:
#      kind: "LINEAR_LE"
#      d: 5.0  # Upper bound
#
# 4. RANGE (Box Constraint on Outputs)
#    -----------------------------------------------
#    Purpose: Verify all outputs lie within specified bounds
#    Verification: lb <= y <= ub (element-wise)
#    
#    Required params:
#    - lb: Lower bound vector (list/tensor, shape matches output)
#    - ub: Upper bound vector (list/tensor, shape matches output)
#    
#    Use cases:
#    - Output range safety: Ensure values stay within physical limits
#    - Control systems: Verify actuator outputs within safe range
#    - Regression verification: Output predictions within expected bounds
#    - Reachability: Verify state remains in safe region
#    
#    Expected outcome:
#    - PASS: All elements satisfy lb <= y[i] <= ub (safe region)
#    - FAIL: One or more elements outside bounds (unsafe region)
#    
#    Example (verify regression output in [0, 10]):
#    params:
#      lb: [0.0, 0.0, 0.0]  # 3 outputs, all >= 0
#      ub: [10.0, 10.0, 10.0]  # All <= 10
#    meta:
#      kind: "RANGE"
#
# Notes:
# - All params specified as lists in YAML are automatically converted to tensors
# - TOP1_ROBUST and MARGIN_ROBUST are classification-specific (discrete classes)
# - LINEAR_LE and RANGE are general (work with any output shape)
# - Verification happens automatically in OutputSpecLayer.forward()
# - Results returned in dict: {output, output_satisfied, output_explanation}
#
#===---------------------------------------------------------------------===#

import json
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from act.back_end.core import Layer, Net
from act.back_end.serialization.serialization import NetSerializer
from act.front_end.specs import InKind, OutKind
from act.util.device_manager import get_default_dtype


class NetFactory:
    """Concise factory that reads config and generates models in nets folder."""
    
    def __init__(
        self,
        config_path: str = "act/back_end/examples/examples_config.yaml",
        output_dir: Optional[str] = None,
    ):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        if output_dir is None:
            output_dir = (self.config or {}).get("generator", {}).get(
                "output_dir",
                "act/back_end/examples/nets",
            )
        self.output_dir = Path(str(output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_weight_tensor(self, kind: str, meta: Dict[str, Any]) -> torch.Tensor:
        """Generate minimal weight tensors that satisfy schema requirements."""
        if kind == "DENSE":
            in_features = meta.get("in_features", 10)
            out_features = meta.get("out_features", 10)
            # Create minimal weight tensor W
            return torch.randn(out_features, in_features) * 0.1
        elif kind in ["CONV2D", "CONV1D", "CONV3D"]:
            in_channels = meta.get("in_channels", 1)
            out_channels = meta.get("out_channels", 1)
            kernel_size = meta.get("kernel_size", 3)
            if isinstance(kernel_size, int):
                if kind == "CONV1D":
                    weight_shape = (out_channels, in_channels, kernel_size)
                elif kind == "CONV2D":
                    weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
                else:  # CONV3D
                    weight_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
            else:
                # kernel_size is a tuple/list
                if kind == "CONV1D":
                    weight_shape = (out_channels, in_channels, kernel_size[0])
                elif kind == "CONV2D":
                    weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
                else:  # CONV3D
                    weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])
            return torch.randn(*weight_shape) * 0.1
        return None
    
    def _generate_input_spec_params(self, params: Dict[str, Any], meta: Dict[str, Any], input_shape: Optional[List[int]]) -> None:
        """Generate INPUT_SPEC params based on kind and meta values."""
        if not input_shape:
            raise ValueError("Cannot generate INPUT_SPEC params: input shape is required but not provided")
        
        spec_kind = meta.get("kind")
        
        # Compare with enum class variables (these are strings, not Enum objects)
        if spec_kind == InKind.BOX:
            # Generate lb/ub from meta values
            lb_val = meta.get("lb_val", 0.0)
            ub_val = meta.get("ub_val", 1.0)
            params["lb"] = torch.full(input_shape, lb_val)
            params["ub"] = torch.full(input_shape, ub_val)
        
        elif spec_kind == InKind.LINF_BALL:
            # Generate center + lb/ub from center_val and eps
            eps = meta.get("eps")
            if eps is None:
                raise ValueError("LINF_BALL requires 'eps' in meta")
            
            center_val = meta.get("center_val", 0.5)  # Default to 0.5 for normalized inputs
            params["center"] = torch.full(input_shape, center_val)
            params["lb"] = params["center"] - eps
            params["ub"] = params["center"] + eps
        
        # LIN_POLY: skip (too complex, user must provide A and b matrices)
    
    def _generate_assert_params(self, params: Dict[str, Any], meta: Dict[str, Any], output_shape: Optional[List[int]]) -> None:
        """Generate ASSERT (OutputSpec) params based on kind and meta values.
        
        Supports four ASSERT kinds: TOP1_ROBUST, MARGIN_ROBUST, LINEAR_LE, and RANGE.
        See file header for detailed documentation of each kind.
        """
        if not output_shape:
            raise ValueError("Cannot generate ASSERT params: output shape is required but not provided")
        
        assert_kind = meta.get("kind")
        
        # Compare with OutKind class variables (these are strings, not Enum objects)
        if assert_kind == OutKind.TOP1_ROBUST:
            # No params to generate (y_true already in meta)
            # Just validate y_true is present
            if "y_true" not in meta:
                raise ValueError("TOP1_ROBUST requires 'y_true' in meta")
        
        elif assert_kind == OutKind.MARGIN_ROBUST:
            # No params to generate (y_true and margin already in meta)
            # Just validate they are present
            if "y_true" not in meta:
                raise ValueError("MARGIN_ROBUST requires 'y_true' in meta")
            if "margin" not in meta:
                raise ValueError("MARGIN_ROBUST requires 'margin' in meta")
        
        elif assert_kind == OutKind.LINEAR_LE:
            # Convert c from list to tensor if present
            if "c" in params and isinstance(params["c"], list):
                params["c"] = torch.tensor(params["c"], dtype=torch.float32)
            
            # Validate d is present in meta
            if "d" not in meta:
                raise ValueError("LINEAR_LE requires 'd' in meta")
        
        elif assert_kind == OutKind.RANGE:
            # Convert lb/ub from lists to tensors if present
            if "lb" in params and isinstance(params["lb"], list):
                params["lb"] = torch.tensor(params["lb"], dtype=torch.float32)
            if "ub" in params and isinstance(params["ub"], list):
                params["ub"] = torch.tensor(params["ub"], dtype=torch.float32)
            
            # Validate both are present
            if "lb" not in params or "ub" not in params:
                raise ValueError("RANGE requires both 'lb' and 'ub' in params")

    def _generate_layer_variables(self, kind: str,
                                  layer_index: int,
                                  var_counter: int,
                                  meta: Dict[str, Any],
                                  layers: List[Layer]) -> Tuple[List[int], List[int], int]:
        """generate vars based on layer shape"""
        if kind == "INPUT":
            shape = meta.get("shape", [])
            if shape:
                out_num_vars = torch.Size(shape).numel()
            else:
                out_num_vars = 1
            out_vars = list(range(var_counter, var_counter + out_num_vars))
            var_counter += out_num_vars
            return [], out_vars, var_counter

        elif kind == "DENSE":
            in_features = meta.get("in_features", 1)
            out_features = meta.get("out_features", 1)
            if layers and layer_index > 0:
                prev_out_vars = layers[layer_index - 1].out_vars
                if len(prev_out_vars) != in_features:
                    raise ValueError(f"DENSE layer expects {in_features} inputs but got {len(prev_out_vars)}")
                in_vars = prev_out_vars
            else:
                in_vars = []
            out_vars = list(range(var_counter, var_counter + out_features))
            var_counter += out_features
            return in_vars, out_vars, var_counter

        elif kind in ["RELU", "SIGMOID", "TANH"]:
            if layers and layer_index > 0:
                in_vars = layers[layer_index - 1].out_vars
                # Allocate new variable IDs for activation output
                out_vars = list(range(var_counter, var_counter + len(in_vars)))
                var_counter += len(in_vars)
                return in_vars, out_vars, var_counter
            else:
                raise ValueError(f"Activation layer '{kind}' cannot be the first layer in network")

        elif kind.startswith("CONV"):
            if layers and layer_index > 0:
                in_vars = layers[layer_index - 1].out_vars
            else:
                raise ValueError(f"Convolutional layer '{kind}' cannot be the first layer in network")

            output_shape = meta.get("output_shape")
            if output_shape:
                out_num_vars = torch.Size(output_shape).numel()
            else:
                raise ValueError(
                    f"Convolutional layer '{kind}' requires 'output_shape' in meta for variable generation")

            out_vars = list(range(var_counter, var_counter + out_num_vars))
            var_counter += out_num_vars
            return in_vars, out_vars, var_counter
        
        elif kind in ["MAXPOOL2D", "AVGPOOL2D", "ADAPTIVEAVGPOOL2D"]:
            if layers and layer_index > 0:
                in_vars = layers[layer_index - 1].out_vars
            else:
                raise ValueError(f"Pooling layer '{kind}' cannot be the first layer in network")

            output_shape = meta.get("output_shape")
            if output_shape:
                out_num_vars = torch.Size(output_shape).numel()
            else:
                raise ValueError(
                    f"Pooling layer '{kind}' requires 'output_shape' in meta for variable generation")

            out_vars = list(range(var_counter, var_counter + out_num_vars))
            var_counter += out_num_vars
            return in_vars, out_vars, var_counter

        elif kind == "FLATTEN":
            if layers and layer_index > 0:
                in_vars = layers[layer_index - 1].out_vars
                out_vars = list(range(var_counter, var_counter + len(in_vars)))
                var_counter += len(in_vars)
                return in_vars, out_vars, var_counter
            else:
                raise ValueError(f"Flatten layer cannot be the first layer in network")


        elif kind in ["INPUT_SPEC", "ASSERT"]:
            if layers and layer_index > 0:
                prev_vars = layers[layer_index - 1].out_vars
                return prev_vars, prev_vars.copy(), var_counter
            else:
                raise ValueError(f"Layer '{kind}' cannot be the first layer in network")

        else:
            supported_types = ["INPUT", "DENSE", "RELU", "SIGMOID", "TANH",
                "CONV1D", "CONV2D", "CONV3D", "MAXPOOL2D", "AVGPOOL2D", "ADAPTIVEAVGPOOL2D",
                "FLATTEN", "INPUT_SPEC","ASSERT",]
            raise NotImplementedError(
                f"Layer type '{kind}' is not supported for variable generation. "
                f"Supported types: {supported_types}. "
                f"Please implement _generate_layer_variables logic for '{kind}' layer."
            )

    def create_network(self, name: str, spec: Dict[str, Any]) -> Net:
        """Create network from YAML spec."""
        # Get current device_manager dtype for INPUT layer consistency
        current_dtype = str(get_default_dtype())
        
        layers = []
        var_counter = 0  # init in/out var counter

        for i, layer_spec in enumerate(spec['layers']):
            # Copy params and add required weight tensors if needed
            params = layer_spec.get('params', {}).copy()
            meta = layer_spec.get('meta', {}).copy()  # Use copy() to avoid modifying YAML
            kind = layer_spec['kind']

            # Simple sequential variable assignment
            in_vars, out_vars, var_counter = self._generate_layer_variables(kind, i, var_counter, meta, layers)
            
            # Update INPUT layer dtype to match current device_manager
            if kind == "INPUT" and 'dtype' in meta:
                meta['dtype'] = current_dtype
            
            # Get input shape for INPUT_SPEC generation
            input_shape = None
            if i > 0 and layers[i-1].kind == "INPUT":
                input_shape = layers[i-1].meta.get("shape")
            
            # Get output shape for ASSERT generation (from last non-wrapper layer)
            output_shape = None
            if i > 0:
                # Look at previous layer's meta for out_features (DENSE) or output shape
                for j in range(i-1, -1, -1):
                    prev_layer = layers[j]
                    if prev_layer.kind == "DENSE":
                        out_features = prev_layer.meta.get("out_features")
                        if out_features:
                            output_shape = [1, out_features]
                            break
                    elif prev_layer.kind in ["CONV2D", "CONV1D", "CONV3D"]:
                        # For conv layers, would need to compute output shape
                        # For now, skip as we're using flatten + dense
                        pass
            
            # === AUTO-GENERATION DISPATCH ===
            if kind == "INPUT_SPEC":
                self._generate_input_spec_params(params, meta, input_shape)
            elif kind == "ASSERT":
                self._generate_assert_params(params, meta, output_shape)
            elif kind == "DENSE" and "W" not in params:
                weight = self.generate_weight_tensor(kind, meta)
                if weight is not None:
                    params["W"] = weight
                # Generate bias if enabled
                if meta.get("bias_enabled", False):
                    out_features = meta.get("out_features", 10)
                    params["b"] = torch.zeros(out_features)
            elif kind.startswith("CONV") and "weight" not in params:
                weight = self.generate_weight_tensor(kind, meta)
                if weight is not None:
                    params["weight"] = weight
                # Generate bias if needed (CONV layers typically have bias by default)
                # For now, we don't add bias to CONV layers unless specified
            
            # Create layer (validation happens automatically in __post_init__)
            layer = Layer(
                id=i,
                kind=kind,
                params=params,
                meta=meta,
                in_vars=in_vars,
                out_vars=out_vars
            )
            
            layers.append(layer)
        
        # Create graph structure for Net
        preds = {i: [i-1] if i > 0 else [] for i in range(len(layers))}
        succs = {i: [i+1] if i < len(layers)-1 else [] for i in range(len(layers))}
        
        net = Net(layers=layers, preds=preds, succs=succs)
        net.meta = {
            'name': name,
            'description': spec.get('description', ''),
            'architecture_type': spec.get('architecture_type', ''),
            'input_shape': spec.get('input_shape', [])
        }
        return net
    
    def save_network(self, net: Net, name: str) -> None:
        """Save network using proper ACT serialization with tensor encoding."""
        output_path = self.output_dir / f"{name}.json"
        
        # Use NetSerializer to properly handle tensors
        net_dict = NetSerializer.serialize_net(net, metadata={'generated_by': 'NetFactory'})
        
        with open(output_path, 'w') as f:
            json.dump(net_dict, f, indent=2)
        print(f"Saved: {output_path}")
    
    def generate_all(self) -> None:
        """Generate all networks from config."""
        networks = self.config['networks']
        print(f"Generating {len(networks)} networks...")
        
        for name, spec in networks.items():
            net = self.create_network(name, spec)
            self.save_network(net, name)
        
        print(f"All networks generated in {self.output_dir}")


if __name__ == "__main__":
    factory = NetFactory()
    factory.generate_all()
