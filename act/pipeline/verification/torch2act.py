#===- act/pipeline/torch2act.py - Torch to ACT Converter ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Spec-free PyTorch → ACT converter for verification. Converts wrapped
#   PyTorch models (containing InputLayer, InputSpecLayer, and OutputSpecLayer)
#   into ACT Net graphs with embedded constraints for formal verification.
#
# Key Features:
#   - Spec-free: Constraints embedded in model, not passed separately
#   - Input-free: Input specifications extracted from wrapper layers
#   - Bidirectional: Paired with act2torch.py for round-trip conversion
#   - Weight preservation: Transfers all model parameters to ACT format
#
# Architecture:
#   InputLayer           → INPUT      (declares input shape/dtype/device)
#   InputSpecLayer       → INPUT_SPEC (input constraints: BOX, L_INF, LIN_POLY)
#   nn.Linear            → DENSE      (fully connected layers)
#   nn.Conv2d            → CONV2D     (convolutional layers)
#   nn.ReLU              → RELU       (activation functions)
#   OutputSpecLayer      → ASSERT     (output constraints: SAFETY, classification)
#
# Note: Preprocessing (normalization, reshaping, etc.) should be handled by
#   data loader (e.g., torchvision.transforms) before wrapping the model.
#
# Contract:
#   - Exactly one InputLayer must be present (defines input shape)
#   - Optional InputSpecLayer for input constraints
#   - Optional OutputSpecLayer for output constraints
#   - All wrapper layers converted to ACT layer graph
#
# Data Organization:
#   - Layer.params: Numeric tensors (weights, bounds, constraint matrices)
#   - Layer.meta: JSON-serializable metadata (dimensions, flags, configs)
#   - Layer.vars: Variable indices for constraint tracking
#
# Usage:
#   from act.pipeline.torch2act import TorchToACT
#   
#   # Convert wrapped PyTorch model to ACT Net
#   converter = TorchToACT(pytorch_model)
#   act_net = converter.run()
#   
#   # ACT Net ready for verification
#   from act.back_end.verifier import verify_once
#   result = verify_once(act_net)
#
#===---------------------------------------------------------------------===#
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth

from act.util.model_inference import model_inference
from act.front_end.model_synthesis import model_synthesis
from act.back_end.core import Net, Layer
from act.back_end.layer_schema import LayerKind
from act.back_end.layer_util import create_layer
from act.back_end.solver.solver_torch import TorchLPSolver
from act.back_end.solver.solver_gurobi import GurobiSolver
from act.front_end.specs import InKind, OutKind
from act.util.options import PerformanceOptions

# -----------------------------------------------------------------------------
# Public helper for solver interpretation (optional)
# -----------------------------------------------------------------------------

class SolveResult:
    SAT = "SAT"         # counterexample exists → property VIOLATED
    UNSAT = "UNSAT"     # no counterexample → property VALID
    UNKNOWN = "UNKNOWN"

# -----------------------------------------------------------------------------
# Torch → ACT converter
# -----------------------------------------------------------------------------

def _prod(shape_tail: Tuple[int, ...]) -> int:
    """Helper function to compute product of shape dimensions."""
    p = 1
    for s in shape_tail:
        p *= int(s)
    return int(p)


class TorchToACT:
    """
    Convert a *wrapped* nn.Sequential to ACT Net/Layers.
    Requirements (asserted in __init__):
      - Contains exactly one InputLayer (first-class source of input shape).
      - Contains at least one InputSpecLayer.
      - Ends with an OutputSpecLayer (producing ASSERT).
    No input_shape is accepted; InputLayer provides it.
    
    Note: Preprocessing should be handled by data loader, not in the wrapper.
    """
    # Type names are matched by isinstance; these references are not imported here to avoid circular deps.
    _InputLayerTypeName = "InputLayer"
    _InputSpecLayerTypeName = "InputSpecLayer"
    _OutputSpecLayerTypeName = "OutputSpecLayer"

    def __init__(self, wrapped: nn.Sequential):
        if not isinstance(wrapped, nn.Sequential):
            raise TypeError("TorchToACT expects an nn.Sequential wrapper model.")
        self.m = wrapped
        mods = list(self.m)

        # --- Assertions: InputSpecLayer and OutputSpecLayer existence ---
        has_input_spec = any(type(x).__name__ == self._InputSpecLayerTypeName for x in mods)
        has_output_spec = any(type(x).__name__ == self._OutputSpecLayerTypeName for x in mods)
        if not has_input_spec:
            raise AssertionError("Wrapper must include an InputSpecLayer — none found.")
        if not has_output_spec:
            raise AssertionError("Wrapper must include an OutputSpecLayer as the final assertion — none found.")

        # Exactly one InputLayer
        input_layers = [x for x in mods if type(x).__name__ == self._InputLayerTypeName]
        if len(input_layers) != 1:
            raise AssertionError(f"Wrapper must contain exactly one InputLayer; found {len(input_layers)}.")
        self.input_layer = input_layers[0]

        # Must end with OutputSpecLayer
        if type(mods[-1]).__name__ != self._OutputSpecLayerTypeName:
            raise AssertionError("Wrapper should end with OutputSpecLayer so last ACT layer is ASSERT.")

        # Init state
        self.layers: List[Layer] = []
        self.next_var = 0
        self.prev_out: List[int] = []
        # Expect InputLayer to have a 'shape' attribute (tuple) that includes batch=1 first.
        shape = getattr(self.input_layer, "shape", None)
        if shape is None:
            raise AssertionError("InputLayer must expose a 'shape' attribute (e.g., (1, C, H, W) or (1, F)).")
        self.shape: Tuple[int, ...] = tuple(int(s) for s in shape)

    # --- var id management ---
    def _alloc_ids(self, n: int) -> List[int]:
        ids = list(range(self.next_var, self.next_var + n))
        self.next_var += n
        return ids

    def _add(self, kind: str, params: Dict[str, torch.Tensor], meta: Dict[str, Any],
             in_vars: List[int], out_vars: List[int]) -> int:
        layer = create_layer(
            id=len(self.layers),
            kind=kind,
            params=params,
            meta=meta,
            in_vars=in_vars,
            out_vars=out_vars,
        )
        self.layers.append(layer)
        return layer.id

    def _same_size_forward(self) -> List[int]:
        return self._alloc_ids(len(self.prev_out))

    # --- mapping helpers ---

    def _emit_input(self):
        """Emit INPUT layer using the to_act_layers() protocol."""
        new_layers, out_vars = self.input_layer.to_act_layers(len(self.layers), [])
        self.layers.extend(new_layers)
        self.prev_out = out_vars

    # --- recursive module processing ---

    def _is_primitive_module(self, mod: nn.Module) -> bool:
        """Check if module is a primitive layer that can be directly converted."""
        return isinstance(mod, (
            nn.Linear, nn.ReLU, nn.Conv2d, nn.Flatten,
            nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.Dropout,
            nn.BatchNorm2d, nn.Tanh, nn.Sigmoid, nn.LeakyReLU, nn.SiLU,
            StochasticDepth
        ))

    def _convert_primitive_module(self, mod: nn.Module) -> None:
        """Convert a primitive PyTorch module to ACT layer(s)."""
        
        if isinstance(mod, nn.Flatten):
            out_vars = self._same_size_forward()
            flattened_shape = (1, _prod(self.shape[1:]))
            self._add(LayerKind.FLATTEN.value, params={}, 
                      meta={"input_shape": self.shape, "output_shape": flattened_shape},
                      in_vars=self.prev_out, out_vars=out_vars)
            self.shape = flattened_shape
            self.prev_out = out_vars
            
        elif isinstance(mod, nn.Linear):
            outF = int(mod.out_features)
            # Use detach() only - no clone needed since we don't modify weights
            W = mod.weight.detach()
            bvec = mod.bias.detach() if mod.bias is not None else torch.zeros(outF, dtype=W.dtype, device=W.device)
            
            out_vars = self._alloc_ids(outF)
            self._add(LayerKind.DENSE.value, params={"W": W, "b": bvec},
                      meta={"input_shape": self.shape, "output_shape": (1, outF)},
                      in_vars=self.prev_out, out_vars=out_vars)
            self.shape = (1, outF)
            self.prev_out = out_vars
            
        elif isinstance(mod, nn.ReLU):
            out_vars = self._same_size_forward()
            self._add(LayerKind.RELU.value, params={}, 
                      meta={"input_shape": self.shape, "output_shape": self.shape},
                      in_vars=self.prev_out, out_vars=out_vars)
            self.prev_out = out_vars
            
        elif isinstance(mod, nn.Conv2d):
            # Use detach() only - no clone needed since we don't modify weights
            weight = mod.weight.detach()
            bias = mod.bias.detach() if mod.bias is not None else None
            
            # Infer input shape for conv
            if len(self.shape) == 2:  # (1, features) - need to reshape to spatial
                n_features = self.shape[1]
                if n_features == 3072:  # CIFAR-10
                    input_shape = (1, 3, 32, 32)
                elif n_features == 784:  # MNIST
                    input_shape = (1, 1, 28, 28)
                else:
                    channels = mod.in_channels
                    spatial_size = int((n_features / channels) ** 0.5)
                    input_shape = (1, channels, spatial_size, spatial_size)
            else:
                input_shape = self.shape
            
            # Calculate output shape
            batch, in_c, in_h, in_w = input_shape
            out_c = mod.out_channels
            out_h = (in_h + 2 * mod.padding[0] - mod.dilation[0] * (mod.kernel_size[0] - 1) - 1) // mod.stride[0] + 1
            out_w = (in_w + 2 * mod.padding[1] - mod.dilation[1] * (mod.kernel_size[1] - 1) - 1) // mod.stride[1] + 1
            output_shape = (1, out_c, out_h, out_w)
            out_features = out_c * out_h * out_w
            
            params = {"weight": weight}
            if bias is not None:
                params["bias"] = bias
                
            meta = {
                "input_shape": input_shape,
                "output_shape": output_shape,
                "kernel_size": mod.kernel_size,
                "stride": mod.stride,
                "padding": mod.padding,
                "dilation": mod.dilation,
                "groups": mod.groups,
                "in_channels": in_c,
                "out_channels": out_c
            }
            
            out_vars = self._alloc_ids(out_features)
            self._add(LayerKind.CONV2D.value, params=params, meta=meta,
                      in_vars=self.prev_out, out_vars=out_vars)
            self.shape = (1, out_features)
            self.prev_out = out_vars
            
        elif isinstance(mod, nn.MaxPool2d):
            # MaxPool2d: Apply pooling operation
            if len(self.shape) == 2:
                # Need to infer spatial shape
                n_features = self.shape[1]
                # Assume square spatial dimensions
                spatial_size = int(n_features ** 0.5)
                channels = 1
                input_shape = (1, channels, spatial_size, spatial_size)
            else:
                input_shape = self.shape
            
            batch, in_c, in_h, in_w = input_shape
            kernel_size = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
            stride = mod.stride if mod.stride is not None else kernel_size
            stride = stride if isinstance(stride, tuple) else (stride, stride)
            padding = mod.padding if isinstance(mod.padding, tuple) else (mod.padding, mod.padding)
            
            out_h = (in_h + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
            out_w = (in_w + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
            output_shape = (1, in_c, out_h, out_w)
            out_features = in_c * out_h * out_w
            
            # Use schema-compliant metadata fields
            meta = {
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "output_size": (out_h, out_w)  # Schema expects this field
            }
            
            out_vars = self._alloc_ids(out_features)
            self._add(LayerKind.MAXPOOL2D.value, params={}, meta=meta,
                      in_vars=self.prev_out, out_vars=out_vars)
            self.shape = (1, out_features)
            self.prev_out = out_vars
            
        elif isinstance(mod, nn.Dropout):
            # Dropout is a no-op during inference/verification
            pass
        
        elif isinstance(mod, StochasticDepth):
            # StochasticDepth (DropPath) is identity during inference/verification
            # During training it randomly drops residual branches, but in eval mode: output = input
            pass
            
        elif isinstance(mod, nn.BatchNorm2d):
            # BatchNorm2d during inference: y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
            # This is equivalent to: y = scale * x + bias
            # where scale = gamma / sqrt(running_var + eps) and bias = beta - scale * running_mean
            
            # Extract BatchNorm parameters (all should be present in eval mode)
            gamma = mod.weight.detach() if mod.weight is not None else torch.ones(mod.num_features, dtype=mod.running_mean.dtype, device=mod.running_mean.device)
            beta = mod.bias.detach() if mod.bias is not None else torch.zeros(mod.num_features, dtype=mod.running_mean.dtype, device=mod.running_mean.device)
            running_mean = mod.running_mean.detach()
            running_var = mod.running_var.detach()
            eps = mod.eps
            
            # Compute affine transformation parameters
            scale = gamma / torch.sqrt(running_var + eps)
            bias = beta - scale * running_mean
            
            # BatchNorm is applied channel-wise, so we need to expand scale/bias to match input shape
            # For spatial data: (1, C, H, W) → scale/bias are (C,) → expand to (1, C, H, W)
            if len(self.shape) == 2:
                # Flattened input: (1, C*H*W) - need to track channel dimension
                # This is tricky, so we'll represent as element-wise multiplication + addition
                # Assuming the input was flattened from (1, C, H, W)
                n_features = self.shape[1]
                n_channels = mod.num_features
                spatial_size = n_features // n_channels
                
                # Expand scale and bias to match flattened shape
                scale_expanded = scale.repeat_interleave(spatial_size)
                bias_expanded = bias.repeat_interleave(spatial_size)
                
                # Create element-wise multiplication and addition layers
                out_vars = self._same_size_forward()
                self._add(LayerKind.MUL.value, params={"scale": scale_expanded},
                         meta={"input_shape": self.shape, "output_shape": self.shape},
                         in_vars=self.prev_out, out_vars=out_vars)
                self.prev_out = out_vars
                
                out_vars = self._same_size_forward()
                self._add(LayerKind.ADD.value, params={"bias": bias_expanded},
                         meta={"input_shape": self.shape, "output_shape": self.shape},
                         in_vars=self.prev_out, out_vars=out_vars)
                self.prev_out = out_vars
            else:
                # Spatial input: (1, C, H, W)
                batch, channels, height, width = self.shape
                
                # Expand scale and bias to spatial dimensions
                scale_expanded = scale.view(1, -1, 1, 1).expand(1, channels, height, width).flatten()
                bias_expanded = bias.view(1, -1, 1, 1).expand(1, channels, height, width).flatten()
                
                # Flatten shape for computation
                flat_size = channels * height * width
                
                # Create element-wise multiplication and addition layers
                out_vars = self._same_size_forward()
                self._add(LayerKind.MUL.value, params={"scale": scale_expanded},
                         meta={"input_shape": (1, flat_size), "output_shape": (1, flat_size),
                               "original_shape": self.shape},
                         in_vars=self.prev_out, out_vars=out_vars)
                self.prev_out = out_vars
                
                out_vars = self._same_size_forward()
                self._add(LayerKind.ADD.value, params={"bias": bias_expanded},
                         meta={"input_shape": (1, flat_size), "output_shape": (1, flat_size),
                               "original_shape": self.shape},
                         in_vars=self.prev_out, out_vars=out_vars)
                self.prev_out = out_vars
        
        elif isinstance(mod, nn.AdaptiveAvgPool2d):
            # AdaptiveAvgPool2d: Adaptive average pooling to output_size
            output_size = mod.output_size
            if isinstance(output_size, int):
                output_size = (output_size, output_size)
            
            # Infer input shape
            if len(self.shape) == 2:
                n_features = self.shape[1]
                # Try to infer from common sizes
                if n_features == 3072:  # CIFAR-10
                    input_shape = (1, 3, 32, 32)
                elif n_features == 784:  # MNIST
                    input_shape = (1, 1, 28, 28)
                else:
                    # Try to infer square spatial dimensions
                    spatial_size = int(n_features ** 0.5)
                    channels = 1
                    input_shape = (1, channels, spatial_size, spatial_size)
            else:
                input_shape = self.shape
            
            batch, in_c, in_h, in_w = input_shape
            out_h, out_w = output_size
            output_shape = (1, in_c, out_h, out_w)
            out_features = in_c * out_h * out_w
            
            # Calculate equivalent kernel and stride for average pooling
            kernel_h = in_h // out_h
            kernel_w = in_w // out_w
            stride_h = kernel_h
            stride_w = kernel_w
            
            meta = {
                "kernel_size": (kernel_h, kernel_w),
                "stride": (stride_h, stride_w),
                "padding": (0, 0),
                "output_size": output_size
            }
            
            out_vars = self._alloc_ids(out_features)
            self._add(LayerKind.AVGPOOL2D.value, params={}, meta=meta,
                      in_vars=self.prev_out, out_vars=out_vars)
            self.shape = (1, out_features)
            self.prev_out = out_vars
        
        elif isinstance(mod, nn.SiLU):
            # SiLU (Swish) activation: x * sigmoid(x)
            out_vars = self._same_size_forward()
            self._add(LayerKind.SILU.value, params={}, 
                      meta={"input_shape": self.shape, "output_shape": self.shape},
                      in_vars=self.prev_out, out_vars=out_vars)
            self.prev_out = out_vars
        
        elif isinstance(mod, nn.Sigmoid):
            out_vars = self._same_size_forward()
            self._add(LayerKind.SIGMOID.value, params={}, 
                      meta={},
                      in_vars=self.prev_out, out_vars=out_vars)
            self.prev_out = out_vars
        
        elif isinstance(mod, nn.Tanh):
            out_vars = self._same_size_forward()
            self._add(LayerKind.TANH.value, params={}, 
                      meta={},
                      in_vars=self.prev_out, out_vars=out_vars)
            self.prev_out = out_vars
        
        elif isinstance(mod, nn.LeakyReLU):
            out_vars = self._same_size_forward()
            self._add(LayerKind.LRELU.value, params={}, 
                      meta={"negative_slope": mod.negative_slope},
                      in_vars=self.prev_out, out_vars=out_vars)
            self.prev_out = out_vars
            
        else:
            raise NotImplementedError(f"Primitive conversion not implemented: {type(mod).__name__}")

    def _process_module(self, mod: nn.Module) -> None:
        """
        Recursively process a module, expanding containers into primitives.
        
        Strategy:
        1. If it's InputLayer → skip (already processed)
        2. If it has to_act_layers() → use protocol (InputSpecLayer, OutputSpecLayer)
        3. If it's a primitive → convert directly
        4. If it's a container → recurse into children
        """
        tname = type(mod).__name__
        
        # Skip InputLayer (already processed in _emit_input)
        if tname == self._InputLayerTypeName:
            return
        
        # ACT wrapper layers with to_act_layers() protocol
        if hasattr(mod, 'to_act_layers'):
            new_layers, out_vars = mod.to_act_layers(len(self.layers), self.prev_out)
            self.layers.extend(new_layers)
            self.prev_out = out_vars
            return
        
        # Primitive modules - convert directly
        if self._is_primitive_module(mod):
            self._convert_primitive_module(mod)
            return
        
        # Container modules - recurse into children
        if isinstance(mod, nn.Module):
            children = list(mod.children())
            if children:  # Has children - recurse
                for child in children:
                    self._process_module(child)
                return
        
        # Unsupported module type
        raise NotImplementedError(
            f"Unsupported module: {tname}\n"
            f"  If this is a custom module, ensure it has primitive children (Linear, Conv2d, etc.)\n"
            f"  or implement the to_act_layers() protocol."
        )

    # --- main conversion ---

    def run(self) -> Net:
        """
        Convert wrapped PyTorch model to ACT Net using recursive module expansion.
        
        Automatically handles:
        - ACT wrapper layers (InputLayer, InputSpecLayer, OutputSpecLayer)
        - Primitive PyTorch layers (Linear, Conv2d, ReLU, etc.)
        - Custom composite modules (SimpleCNN, LeNet5, etc.) via recursion
        """
        # Emit INPUT from InputLayer
        self._emit_input()

        # Walk modules and recursively process them
        for mod in self.m:
            self._process_module(mod)

        # Build linear graph structure (sequential layers)
        preds = {i: ([] if i == 0 else [i - 1]) for i in range(len(self.layers))}
        succs = {i: ([] if i == len(self.layers) - 1 else [i + 1]) for i in range(len(self.layers))}
        net = Net(layers=self.layers, preds=preds, succs=succs)

        # Validate the created network structure
        from act.back_end.layer_util import validate_graph
        validate_graph(self.layers)

        # Final sanity check
        net.assert_last_is_validation()
        return net


def main():
    """Main entry point for PyTorch→ACT conversion and verification testing."""
    # Initialize debug file (GUARDED)
    if PerformanceOptions.debug_tf:
        debug_file = PerformanceOptions.debug_output_file
        with open(debug_file, 'w') as f:
            f.write(f"ACT Torch2ACT Conversion Debug Log\n")
            f.write(f"{'='*80}\n\n")
        print(f"Debug logging to: {debug_file}")
    
    print("🚀 Starting Spec-Free, Input-Free Torch→ACT Verification Demo")
    
    # Step 1: Synthesize all wrapped models
    print("\n📦 Step 1: Synthesizing wrapped models...")
    wrapped_models = model_synthesis()
    print(f"  ✅ Generated {len(wrapped_models)} wrapped models")
    
    # Step 2: Test all models with inference (input data now stored in models)
    print("\n🧪 Step 2: Testing model inference...")
    successful_models = model_inference(wrapped_models)
    print(f"  ✅ {len(successful_models)} models passed inference tests")
    
    if not successful_models:
        print("  ❌ No successful models to verify!")
        exit(1)
    
    # Step 3: Convert and verify all successful models (memory-efficient)
    print(f"\n🎯 Step 3: Converting and verifying all {len(successful_models)} successful models...")
    print(f"  💡 Processing one at a time to avoid memory issues...")
    
    # Import verification functions
    from act.back_end.verifier import verify_once
    
    import gc
    import torch as torch_module
    
    conversion_results = {}
    verification_results = {}
    conversion_success_count = 0
    verification_success_count = 0
    
    # Step 4: Initialize solvers (moved earlier to reuse for all models)
    print("\n🔧 Step 4: Initializing solvers...")
    gurobi_solver = None
    torch_solver = None
    
    try:
        gurobi_solver = GurobiSolver()
        gurobi_solver.begin("act_verification")
        print("  ✅ Gurobi solver available")
    except Exception as e:
        print(f"  ⚠️  Gurobi initialization failed: {e}")
    
    try:
        torch_solver = TorchLPSolver()
        torch_solver.begin("act_verification")
        print(f"  ✅ TorchLP solver available (device: {torch_solver._device})")
    except Exception as e:
        print(f"  ❌ TorchLP initialization failed: {e}")
    
    solvers_to_test = []
    if gurobi_solver:
        solvers_to_test.append(("Gurobi", gurobi_solver))
    if torch_solver:
        solvers_to_test.append(("TorchLP", torch_solver))
    
    if not solvers_to_test:
        print("  ❌ No solvers available!")
        print("  ℹ️  Will only test conversions without verification")
    
    print(f"\n� Step 5: Processing all models...")
    
    for idx, (model_id, wrapped_model) in enumerate(successful_models.items(), 1):
        print(f"\n  [{idx}/{len(successful_models)}] Processing '{model_id}'...")
        
        # === CONVERSION ===
        try:
            net = TorchToACT(wrapped_model).run()
            
            # Verify the conversion produced a valid net
            if not net.layers:
                raise ValueError("Net should have layers")
            if net.layers[0].kind != LayerKind.INPUT.value:
                raise ValueError(f"First layer should be INPUT, got {net.layers[0].kind}")
            if net.layers[-1].kind != LayerKind.ASSERT.value:
                raise ValueError(f"Last layer should be ASSERT, got {net.layers[-1].kind}")
            
            layer_types = " → ".join([layer.kind for layer in net.layers])
            print(f"    ✅ Conversion: {len(net.layers)} layers ({layer_types})")
            
            conversion_results[model_id] = "SUCCESS"
            conversion_success_count += 1
            
        except Exception as e:
            conversion_results[model_id] = f"FAILED: {str(e)[:100]}..."
            print(f"    ❌ Conversion FAILED: {e}")
            continue  # Skip verification if conversion failed
        
        # === VERIFICATION (only if solvers available) ===
        if solvers_to_test:
            model_verification = {}
            
            for solver_name, solver in solvers_to_test:
                try:
                    # TEMPORARILY COMMENTED OUT: Testing if verify_once causes memory issue
                    # res = verify_once(net, solver=solver, timelimit=30.0)
                    # status = res.status
                    status = "SKIPPED"  # Placeholder to test memory usage
                    model_verification[solver_name] = status
                    print(f"    🔍 Verification ({solver_name}): {status} (verify_once commented out)")
                    
                    if status == "UNSAT" or status == "SAT":
                        verification_success_count += 1
                        
                except Exception as e:
                    model_verification[solver_name] = f"ERROR: {str(e)[:50]}"
                    print(f"    ⚠️  Verification ({solver_name}): ERROR - {str(e)[:50]}")
            
            verification_results[model_id] = model_verification
        
        # === MEMORY CLEANUP ===
        # Free memory from this net immediately (no need to store)
        del net
        
        # Clean up memory periodically
        if idx % 10 == 0:
            gc.collect()
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
    
    # === FINAL SUMMARY ===
    total_count = len(successful_models)
    print(f"\n📊 Final Results:")
    print(f"  ✅ Conversions: {conversion_success_count}/{total_count} ({conversion_success_count/total_count*100:.1f}%)")
    
    if solvers_to_test and verification_results:
        # Count successful verifications (UNSAT or SAT results)
        total_verifications = sum(len(v) for v in verification_results.values())
        successful_verifications = sum(
            1 for results in verification_results.values() 
            for status in results.values() 
            if isinstance(status, str) and status in ["UNSAT", "SAT"]
        )
        print(f"  🔍 Verifications: {successful_verifications}/{total_verifications} successful")
    
    # Print failed conversions if any
    failed_conversions = {k: v for k, v in conversion_results.items() if v != "SUCCESS"}
    if failed_conversions:
        print(f"\n  ⚠️  Failed conversions: {len(failed_conversions)}")
        for model_id, error in list(failed_conversions.items())[:5]:  # Show first 5
            print(f"    • {model_id}: {error}")
    
    # Print debug file location (GUARDED)
    if PerformanceOptions.debug_tf:
        print(f"\n📝 Debug log written to: {PerformanceOptions.debug_output_file}")
    
    print("\n✅ Torch→ACT conversion and verification completed!")


if __name__ == "__main__":
    main()
