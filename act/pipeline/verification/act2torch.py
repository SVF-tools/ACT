#!/usr/bin/env python3
#===- act/pipeline/act2torch.py - ACT to Torch Converter ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   ACT → PyTorch converter with spec-free verification support. Converts
#   ACT Net graphs (abstract constraint representations) into executable
#   PyTorch models with embedded constraint checking capabilities.
#
# Key Features:
#   - Bidirectional: Inverse of torch2act.py for round-trip conversion
#   - Weight preservation: Transfers all ACT parameters to PyTorch layers
#   - VerifiableModel: Returns wrapped model with automatic constraint checking
#   - Spec reconstruction: Rebuilds InputSpecLayer/OutputSpecLayer from ACT
#   - Comprehensive coverage: Supports 50+ layer types (MLP, CNN, RNN, etc.)
#   - DAG Support: VerifiableGraphModel for multi-input layers (ADD, CONCAT, etc.)
#
# Architecture:
#   INPUT      → (skipped)           (no-op, shape already defined)
#   INPUT_SPEC → InputSpecLayer      (input constraint checking)
#   DENSE      → nn.Linear           (fully connected layers)
#   CONV2D     → nn.Conv2d           (convolutional layers)
#   RELU       → nn.ReLU             (activation functions)
#   ADD        → torch.add           (multi-input binary operation)
#   ASSERT     → OutputSpecLayer     (output constraint checking)
#
# VerifiableModel (Sequential):
#   Wraps nn.Module to provide automatic constraint verification for linear chains.
#   Returns dict with:
#   - 'output': Model predictions
#   - 'input_satisfied': Input constraint satisfaction status
#   - 'input_explanation': Human-readable input constraint result
#   - 'output_satisfied': Output constraint satisfaction status
#   - 'output_explanation': Human-readable output constraint result
#
# VerifiableGraphModel (DAG):
#   NEW: Supports DAG structures with multi-input layers via explicit preds.
#   Executes layers in topological order, caching intermediate outputs.
#   Strict mode: Raises on unsupported layers (no silent skipping).
#
# Usage:
#   from act.pipeline.act2torch import ACTToTorch
#
#   # Convert ACT Net to verifiable PyTorch model
#   converter = ACTToTorch(act_net, use_graph_model=True, strict=True)
#   model = converter.run()
#
#   # Run inference with automatic constraint checking
#   results = model(input_tensor)
#   print(f"Output: {results['output']}")
#   print(f"Input OK: {results['input_satisfied']}")
#   print(f"Output OK: {results['output_satisfied']}")
#
# Design:
#   - Mirrors TorchToACT for symmetrical bidirectional conversion
#   - Weight transfer: Copies all parameters from ACT Layer.params to PyTorch
#   - Spec preservation: Reconstructs InputSpec/OutputSpec from ACT metadata
#   - Layer factory: Creates appropriate PyTorch layers from ACT layer kinds
#
#===---------------------------------------------------------------------===#
#
# Supported layer types are defined in:
#   - act/back_end/layer_schema.py (REGISTRY) for schema validation
#   - _create_torch_layer() method below for PyTorch conversion logic
#
#===---------------------------------------------------------------------===#

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import logging

from act.back_end.core import Net, Layer
from act.util.device_manager import get_default_dtype, get_default_device

logger = logging.getLogger(__name__)


class ACTToTorch:
    """
    Convert ACT Net to PyTorch nn.Module.

    This class provides the inverse transformation of TorchToACT, enabling
    bidirectional conversion between verification representations (ACT) and
    executable models (PyTorch).

    Usage (Sequential mode - backward compatible):
        converter = ACTToTorch(act_net)
        model = converter.run()  # Returns VerifiableModel (nn.Sequential)

    Usage (DAG mode - supports multi-input layers):
        converter = ACTToTorch(act_net, use_graph_model=True, strict=True)
        model = converter.run()  # Returns VerifiableGraphModel (DAG)

    Args:
        act_net: ACT Net object containing layers with architecture and weights
        use_graph_model: If True, return VerifiableGraphModel (DAG support).
                         If False, return VerifiableModel (Sequential, legacy).
        strict: If True, raise ValueError on unsupported layers.
                If False, log warning and skip (legacy behavior).

    Returns:
        PyTorch nn.Module model ready for inference
    """

    def __init__(self, act_net: Net, use_graph_model: bool = False, strict: bool = False):
        """
        Initialize converter with ACT Net.

        Args:
            act_net: ACT Net object (contains architecture + weights)
            use_graph_model: Use DAG-capable VerifiableGraphModel (default: False)
            strict: Raise on unsupported layers instead of skipping (default: False)

        Raises:
            TypeError: If act_net is not a Net instance
        """
        if not isinstance(act_net, Net):
            raise TypeError(f"ACTToTorch expects a Net object, got {type(act_net)}")
        self.act_net = act_net
        self.use_graph_model = use_graph_model
        self.strict = strict
    
    def run(self) -> nn.Module:
        """
        Convert ACT Net to PyTorch nn.Module.

        Iterates through ACT layers, creates corresponding PyTorch layers,
        transfers weights, and assembles into VerifiableModel or VerifiableGraphModel.

        Returns:
            VerifiableModel (Sequential) or VerifiableGraphModel (DAG) with embedded constraint checking

        Raises:
            ValueError: If no valid PyTorch layers can be created, or if strict mode
                       encounters unsupported layers
        """
        if self.use_graph_model:
            return self._run_graph_model()
        else:
            return self._run_sequential_model()

    def _run_sequential_model(self) -> nn.Module:
        """
        Legacy Sequential model conversion (backward compatible).

        Returns VerifiableModel (nn.Sequential subclass).
        """
        torch_layers = []
        has_input_spec = False
        has_output_spec = False

        # Get target dtype/device once for all tensor conversions
        target_dtype = get_default_dtype()
        target_device = get_default_device()

        for i, act_layer in enumerate(self.act_net.layers):
            kind = act_layer.kind
            meta = act_layer.meta

            # Handle wrapper layers specially
            if kind == 'INPUT':
                continue  # Skip INPUT layer (no-op)

            if kind == 'INPUT_SPEC':
                # Create InputSpecLayer for constraint checking
                from act.front_end.verifiable_model import InputSpecLayer
                from act.front_end.specs import InputSpec, InKind

                # Build InputSpec from ACT layer
                kind_str = meta['kind']
                spec_kind = getattr(InKind, kind_str)  # Convert string to enum
                spec_dict = {'kind': spec_kind}
                if 'eps' in meta:
                    spec_dict['eps'] = meta['eps']

                # Convert parameter tensors to device_manager dtype for consistency
                for param_key in ['lb', 'ub', 'center', 'A', 'b']:
                    if param_key in act_layer.params:
                        tensor = act_layer.params[param_key]
                        spec_dict[param_key] = tensor.to(dtype=target_dtype, device=target_device)

                spec = InputSpec(**spec_dict)
                # InputSpecLayer now always returns tuples
                torch_layers.append(InputSpecLayer(spec))
                has_input_spec = True
                continue

            elif kind == 'ASSERT':
                # Create OutputSpecLayer for constraint checking
                from act.front_end.verifiable_model import OutputSpecLayer
                from act.front_end.specs import OutputSpec, OutKind

                # Build OutputSpec from ACT layer
                kind_str = meta['kind']
                spec_kind = getattr(OutKind, kind_str)  # Convert string to enum
                spec_dict = {'kind': spec_kind}
                if 'y_true' in meta:
                    spec_dict['y_true'] = meta['y_true']
                if 'margin' in meta:
                    spec_dict['margin'] = meta['margin']
                if 'd' in meta:
                    spec_dict['d'] = meta['d']

                # Convert parameter tensors to device_manager dtype for consistency
                for param_key in ['c', 'lb', 'ub']:
                    if param_key in act_layer.params:
                        tensor = act_layer.params[param_key]
                        spec_dict[param_key] = tensor.to(dtype=target_dtype, device=target_device)

                spec = OutputSpec(**spec_dict)
                # OutputSpecLayer now always returns tuples
                torch_layers.append(OutputSpecLayer(spec))
                has_output_spec = True
                continue

            # Build PyTorch layer from ACT layer (includes weight transfer)
            torch_layer = self._create_torch_layer(kind, meta, act_layer)

            if torch_layer is not None:
                torch_layers.append(torch_layer)

        if not torch_layers:
            raise ValueError("No valid PyTorch layers found in ACT Net")

        # Return VerifiableModel for automatic constraint checking
        from act.front_end.verifiable_model import VerifiableModel
        model = VerifiableModel(*torch_layers)
        model.eval()  # Set to evaluation mode by default

        logger.info(f"Created VerifiableModel with {len(torch_layers)} layers "
                   f"(INPUT_SPEC={has_input_spec}, OUTPUT_SPEC={has_output_spec})")

        return model

    def _run_graph_model(self) -> nn.Module:
        """
        DAG-capable VerifiableGraphModel conversion.

        Supports multi-input layers (ADD, CONCAT, etc.) via explicit preds.
        Strict mode: Raises ValueError on unsupported layers.

        Returns:
            VerifiableGraphModel instance
        """
        from act.front_end.verifiable_model import VerifiableGraphModel

        # Build layer modules dictionary
        layer_modules = {}  # layer_id -> (nn.Module or callable, is_spec_layer, kind)
        input_spec_layer_id = None
        output_spec_layer_id = None

        # Get target dtype/device once for all tensor conversions
        target_dtype = get_default_dtype()
        target_device = get_default_device()

        for act_layer in self.act_net.layers:
            kind = act_layer.kind
            meta = act_layer.meta
            layer_id = act_layer.id

            # Handle wrapper layers specially
            if kind == 'INPUT':
                # Store INPUT layer info but don't create PyTorch layer
                layer_modules[layer_id] = (None, False, kind)
                continue

            if kind == 'INPUT_SPEC':
                # Create InputSpecLayer for constraint checking
                from act.front_end.verifiable_model import InputSpecLayer
                from act.front_end.specs import InputSpec, InKind

                # Build InputSpec from ACT layer
                kind_str = meta['kind']
                spec_kind = getattr(InKind, kind_str)
                spec_dict = {'kind': spec_kind}
                if 'eps' in meta:
                    spec_dict['eps'] = meta['eps']

                # Convert parameter tensors to device_manager dtype for consistency
                for param_key in ['lb', 'ub', 'center', 'A', 'b']:
                    if param_key in act_layer.params:
                        tensor = act_layer.params[param_key]
                        spec_dict[param_key] = tensor.to(dtype=target_dtype, device=target_device)

                spec = InputSpec(**spec_dict)
                layer_modules[layer_id] = (InputSpecLayer(spec), True, kind)
                input_spec_layer_id = layer_id
                continue

            elif kind == 'ASSERT':
                # Create OutputSpecLayer for constraint checking
                from act.front_end.verifiable_model import OutputSpecLayer
                from act.front_end.specs import OutputSpec, OutKind

                # Build OutputSpec from ACT layer
                kind_str = meta['kind']
                spec_kind = getattr(OutKind, kind_str)
                spec_dict = {'kind': spec_kind}
                if 'y_true' in meta:
                    spec_dict['y_true'] = meta['y_true']
                if 'margin' in meta:
                    spec_dict['margin'] = meta['margin']
                if 'd' in meta:
                    spec_dict['d'] = meta['d']

                # Convert parameter tensors to device_manager dtype for consistency
                for param_key in ['c', 'lb', 'ub']:
                    if param_key in act_layer.params:
                        tensor = act_layer.params[param_key]
                        spec_dict[param_key] = tensor.to(dtype=target_dtype, device=target_device)

                spec = OutputSpec(**spec_dict)
                layer_modules[layer_id] = (OutputSpecLayer(spec), True, kind)
                output_spec_layer_id = layer_id
                continue

            # Build PyTorch layer from ACT layer (includes weight transfer)
            torch_layer = self._create_torch_layer(kind, meta, act_layer)

            if torch_layer is None:
                if self.strict:
                    raise ValueError(
                        f"Strict mode: Layer kind '{kind}' not implemented in _create_torch_layer() at layer {layer_id}. "
                        f"Add implementation for this layer type or disable strict mode."
                    )
                else:
                    logger.warning(f"Unsupported layer kind '{kind}' at layer {layer_id} - skipping")
                    continue

            layer_modules[layer_id] = (torch_layer, False, kind)

        if not layer_modules:
            raise ValueError("No valid PyTorch layers found in ACT Net")

        # Create VerifiableGraphModel
        model = VerifiableGraphModel(
            net=self.act_net,
            layer_modules=layer_modules,
            input_spec_layer_id=input_spec_layer_id,
            output_spec_layer_id=output_spec_layer_id
        )
        model.eval()  # Set to evaluation mode by default

        logger.info(f"Created VerifiableGraphModel with {len(layer_modules)} layers "
                   f"(INPUT_SPEC={input_spec_layer_id is not None}, "
                   f"OUTPUT_SPEC={output_spec_layer_id is not None})")

        return model
    
    def _transfer_weights(self, torch_layer: nn.Module, act_layer: Layer, 
                         weight_key: str = "W", bias_key: str = "b") -> None:
        """
        Transfer weights and biases from ACT layer to PyTorch layer.
        
        Args:
            torch_layer: PyTorch layer with weight/bias parameters
            act_layer: ACT layer containing parameter tensors
            weight_key: Key for weight parameter in act_layer.params ("W" or "weight")
            bias_key: Key for bias parameter in act_layer.params ("b" or "bias")
        """
        with torch.no_grad():
            # Transfer weights
            if weight_key in act_layer.params:
                torch_layer.weight.copy_(act_layer.params[weight_key])
            
            # Transfer bias (or zero it if not present in ACT layer)
            if hasattr(torch_layer, 'bias') and torch_layer.bias is not None:
                if bias_key in act_layer.params:
                    torch_layer.bias.copy_(act_layer.params[bias_key])
                else:
                    torch_layer.bias.zero_()
    
    def _create_torch_layer(self, kind: str, meta: Dict[str, Any], 
                           act_layer: Optional[Layer] = None) -> Optional[nn.Module]:
        """
        Create PyTorch layer from ACT layer kind and metadata.
        
        Args:
            kind: Layer kind string (DENSE, CONV2D, RELU, etc.)
            meta: Layer metadata dictionary
            act_layer: Optional ACT Layer to load weights from
            
        Returns:
            PyTorch nn.Module or None if layer should be skipped
        
        Raises:
            ValueError: If required metadata is missing for layer type
        """
        # Dense/Linear layers
        if kind == "DENSE":
            in_features = meta.get("in_features")
            out_features = meta.get("out_features")
            bias_enabled = meta.get("bias_enabled", True)
            
            if in_features is None:
                raise ValueError("DENSE layer requires 'in_features' in meta")
            if out_features is None:
                raise ValueError("DENSE layer requires 'out_features' in meta")
            
            layer = nn.Linear(in_features, out_features, bias=bias_enabled)
            
            # Transfer weights and bias from ACT layer
            if act_layer is not None:
                self._transfer_weights(layer, act_layer, weight_key="W", bias_key="b")
            
            return layer

        # Affine transformations (BIAS, SCALE, BN)
        elif kind == "BIAS":
            # Bias addition: y = x + c
            if act_layer is None or "c" not in act_layer.params:
                raise ValueError("BIAS layer requires 'c' in params")
            c = act_layer.params["c"]

            class BiasModule(nn.Module):
                def __init__(self, bias):
                    super().__init__()
                    self.register_buffer('bias', bias)
                def forward(self, x):
                    return x + self.bias
            return BiasModule(c)

        elif kind == "SCALE":
            # Element-wise scaling: y = x * a
            if act_layer is None or "a" not in act_layer.params:
                raise ValueError("SCALE layer requires 'a' in params")
            a = act_layer.params["a"]

            class ScaleModule(nn.Module):
                def __init__(self, scale):
                    super().__init__()
                    self.register_buffer('scale', scale)
                def forward(self, x):
                    return x * self.scale
            return ScaleModule(a)

        elif kind == "BN":
            # Batch normalization affine transform: y = A*x + c (frozen)
            if act_layer is None or "A" not in act_layer.params or "c" not in act_layer.params:
                raise ValueError("BN layer requires 'A' and 'c' in params")
            A = act_layer.params["A"]
            c = act_layer.params["c"]

            class BNAffineModule(nn.Module):
                def __init__(self, A, c):
                    super().__init__()
                    self.register_buffer('A', A)
                    self.register_buffer('c', c)
                def forward(self, x):
                    return self.A * x + self.c
            return BNAffineModule(A, c)

        # Convolutional layers
        elif kind == "CONV2D":
            in_channels = meta.get("in_channels")
            out_channels = meta.get("out_channels")
            kernel_size = meta.get("kernel_size", 3)
            stride = meta.get("stride", 1)
            padding = meta.get("padding", 0)
            dilation = meta.get("dilation", 1)
            groups = meta.get("groups", 1)
            
            if in_channels is None:
                raise ValueError("CONV2D layer requires 'in_channels' in meta")
            if out_channels is None:
                raise ValueError("CONV2D layer requires 'out_channels' in meta")
            
            layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups
            )
            
            # Transfer weights and bias from ACT layer
            if act_layer is not None:
                self._transfer_weights(layer, act_layer, weight_key="weight", bias_key="bias")
            
            return layer
        
        elif kind == "CONV1D":
            in_channels = meta.get("in_channels")
            out_channels = meta.get("out_channels")
            kernel_size = meta.get("kernel_size", 3)
            stride = meta.get("stride", 1)
            padding = meta.get("padding", 0)
            
            if in_channels is None:
                raise ValueError("CONV1D layer requires 'in_channels' in meta")
            if out_channels is None:
                raise ValueError("CONV1D layer requires 'out_channels' in meta")
            
            layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
            
            # Transfer weights and bias from ACT layer
            if act_layer is not None:
                self._transfer_weights(layer, act_layer, weight_key="weight", bias_key="bias")
            
            return layer
        
        elif kind == "CONV3D":
            in_channels = meta.get("in_channels")
            out_channels = meta.get("out_channels")
            kernel_size = meta.get("kernel_size", 3)
            stride = meta.get("stride", 1)
            padding = meta.get("padding", 0)
            
            if in_channels is None:
                raise ValueError("CONV3D layer requires 'in_channels' in meta")
            if out_channels is None:
                raise ValueError("CONV3D layer requires 'out_channels' in meta")
            
            layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

            # Transfer weights and bias from ACT layer
            if act_layer is not None:
                self._transfer_weights(layer, act_layer, weight_key="weight", bias_key="bias")

            return layer

        elif kind == "CONVTRANSPOSE2D":
            in_channels = meta.get("in_channels")
            out_channels = meta.get("out_channels")
            kernel_size = meta.get("kernel_size", 3)
            stride = meta.get("stride", 1)
            padding = meta.get("padding", 0)
            output_padding = meta.get("output_padding", 0)
            groups = meta.get("groups", 1)
            dilation = meta.get("dilation", 1)

            if in_channels is None:
                raise ValueError("CONVTRANSPOSE2D layer requires 'in_channels' in meta")
            if out_channels is None:
                raise ValueError("CONVTRANSPOSE2D layer requires 'out_channels' in meta")

            layer = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                dilation=dilation
            )

            # Transfer weights and bias from ACT layer
            if act_layer is not None:
                self._transfer_weights(layer, act_layer, weight_key="weight", bias_key="bias")

            return layer

        # Pooling layers
        elif kind == "MAXPOOL2D":
            kernel_size = meta.get("kernel_size")
            stride = meta.get("stride")
            padding = meta.get("padding", 0)
            
            if kernel_size is None:
                raise ValueError("MAXPOOL2D layer requires 'kernel_size' in meta")
            
            return nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        
        elif kind == "MAXPOOL1D":
            kernel_size = meta.get("kernel_size")
            stride = meta.get("stride")
            padding = meta.get("padding", 0)
            
            if kernel_size is None:
                raise ValueError("MAXPOOL1D layer requires 'kernel_size' in meta")
            
            return nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        
        elif kind == "MAXPOOL3D":
            kernel_size = meta.get("kernel_size")
            stride = meta.get("stride")
            padding = meta.get("padding", 0)
            
            if kernel_size is None:
                raise ValueError("MAXPOOL3D layer requires 'kernel_size' in meta")
            
            return nn.MaxPool3d(kernel_size, stride=stride, padding=padding)
        
        elif kind == "AVGPOOL1D":
            kernel_size = meta.get("kernel_size")
            stride = meta.get("stride")
            padding = meta.get("padding", 0)

            if kernel_size is None:
                raise ValueError("AVGPOOL1D layer requires 'kernel_size' in meta")

            return nn.AvgPool1d(kernel_size, stride=stride, padding=padding)

        elif kind == "AVGPOOL2D":
            kernel_size = meta.get("kernel_size")
            stride = meta.get("stride")
            padding = meta.get("padding", 0)

            if kernel_size is None:
                raise ValueError("AVGPOOL2D layer requires 'kernel_size' in meta")

            return nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

        elif kind == "AVGPOOL3D":
            kernel_size = meta.get("kernel_size")
            stride = meta.get("stride")
            padding = meta.get("padding", 0)

            if kernel_size is None:
                raise ValueError("AVGPOOL3D layer requires 'kernel_size' in meta")

            return nn.AvgPool3d(kernel_size, stride=stride, padding=padding)

        elif kind == "ADAPTIVEAVGPOOL2D":
            output_size = meta.get("output_size", 1)
            return nn.AdaptiveAvgPool2d(output_size)
        
        # Activation functions
        elif kind == "RELU":
            return nn.ReLU()
        
        elif kind == "LRELU":
            negative_slope = meta.get("negative_slope", 0.01)
            return nn.LeakyReLU(negative_slope)
        
        elif kind == "PRELU":
            return nn.PReLU()
        
        elif kind == "SIGMOID":
            return nn.Sigmoid()
        
        elif kind == "TANH":
            return nn.Tanh()
        
        elif kind == "SOFTPLUS":
            return nn.Softplus()
        
        elif kind == "SILU":
            return nn.SiLU()
        
        elif kind == "GELU":
            approximate = meta.get("approximate", "none")
            return nn.GELU(approximate=approximate)
        
        elif kind == "RELU6":
            return nn.ReLU6()
        
        elif kind == "HARDTANH":
            min_val = meta.get("min_val", -1.0)
            max_val = meta.get("max_val", 1.0)
            return nn.Hardtanh(min_val, max_val)
        
        elif kind == "HARDSIGMOID":
            return nn.Hardsigmoid()
        
        elif kind == "HARDSWISH":
            return nn.Hardswish()
        
        elif kind == "SOFTSIGN":
            return nn.Softsign()
        
        elif kind == "MISH":
            return nn.Mish()
        
        # Tensor operations
        elif kind == "FLATTEN":
            start_dim = meta.get("start_dim", 1)
            end_dim = meta.get("end_dim", -1)
            return nn.Flatten(start_dim, end_dim)

        elif kind == "PAD":
            # Padding operation using functional API
            # Compatible with pad/pads/padding field names
            padding = meta.get("pad")
            if padding is None:
                padding = meta.get("pads")
            if padding is None:
                padding = meta.get("padding")
            if padding is None:
                raise ValueError("PAD requires 'pad', 'pads', or 'padding' in meta")
            mode = meta.get("mode", "constant")
            value = meta.get("value", 0.0)
            # padding format: (left, right, top, bottom, front, back, ...)
            class PadModule(nn.Module):
                def __init__(self, padding, mode, value):
                    super().__init__()
                    self.padding = padding
                    self.mode = mode
                    self.value = value
                def forward(self, x):
                    return torch.nn.functional.pad(x, self.padding, mode=self.mode, value=self.value)
            return PadModule(padding, mode, value)

        elif kind == "UPSAMPLE":
            # Upsampling operation
            scale_factor = meta.get("scale_factor")
            size = meta.get("size")
            mode = meta.get("mode", "nearest")

            if scale_factor is None and size is None:
                raise ValueError("UPSAMPLE requires either 'scale_factor' or 'size' in meta")

            return nn.Upsample(scale_factor=scale_factor, size=size, mode=mode)

        elif kind == "DROPOUT":
            p = meta.get("p", 0.5)
            return nn.Dropout(p)
        
        elif kind == "BATCHNORM2D":
            num_features = meta.get("num_features")
            if num_features is None:
                raise ValueError("BATCHNORM2D requires 'num_features' in meta")
            return nn.BatchNorm2d(num_features)
        
        elif kind == "BATCHNORM1D":
            num_features = meta.get("num_features")
            if num_features is None:
                raise ValueError("BATCHNORM1D requires 'num_features' in meta")
            return nn.BatchNorm1d(num_features)
        
        elif kind == "LAYERNORM":
            normalized_shape = meta.get("normalized_shape")
            if normalized_shape is None:
                raise ValueError("LAYERNORM requires 'normalized_shape' in meta")
            return nn.LayerNorm(normalized_shape)
        
        # Embedding and sequence layers
        elif kind == "EMBEDDING":
            num_embeddings = meta.get("num_embeddings")
            embedding_dim = meta.get("embedding_dim")
            
            if num_embeddings is None:
                raise ValueError("EMBEDDING requires 'num_embeddings' in meta")
            if embedding_dim is None:
                raise ValueError("EMBEDDING requires 'embedding_dim' in meta")
            
            return nn.Embedding(num_embeddings, embedding_dim)
        
        elif kind == "RNN":
            input_size = meta.get("input_size")
            hidden_size = meta.get("hidden_size")
            num_layers = meta.get("num_layers", 1)
            bidirectional = meta.get("bidirectional", False)
            batch_first = meta.get("batch_first", False)
            
            if input_size is None:
                raise ValueError("RNN requires 'input_size' in meta")
            if hidden_size is None:
                raise ValueError("RNN requires 'hidden_size' in meta")
            
            return nn.RNN(input_size, hidden_size, num_layers, 
                         batch_first=batch_first, bidirectional=bidirectional)
        
        elif kind == "LSTM":
            input_size = meta.get("input_size")
            hidden_size = meta.get("hidden_size")
            num_layers = meta.get("num_layers", 1)
            bidirectional = meta.get("bidirectional", False)
            batch_first = meta.get("batch_first", False)
            
            if input_size is None:
                raise ValueError("LSTM requires 'input_size' in meta")
            if hidden_size is None:
                raise ValueError("LSTM requires 'hidden_size' in meta")
            
            return nn.LSTM(input_size, hidden_size, num_layers,
                          batch_first=batch_first, bidirectional=bidirectional)
        
        elif kind == "GRU":
            input_size = meta.get("input_size")
            hidden_size = meta.get("hidden_size")
            num_layers = meta.get("num_layers", 1)
            bidirectional = meta.get("bidirectional", False)
            batch_first = meta.get("batch_first", False)
            
            if input_size is None:
                raise ValueError("GRU requires 'input_size' in meta")
            if hidden_size is None:
                raise ValueError("GRU requires 'hidden_size' in meta")
            
            return nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=batch_first, bidirectional=bidirectional)
        
        elif kind == "SOFTMAX":
            axis = meta.get("axis", -1)
            return nn.Softmax(dim=axis)

        # Extended activation functions (Phase 2)
        elif kind == "SIGMOID":
            return nn.Sigmoid()

        elif kind == "TANH":
            return nn.Tanh()

        elif kind == "LRELU":
            negative_slope = meta.get("negative_slope", 0.01)
            return nn.LeakyReLU(negative_slope)

        elif kind == "RELU6":
            return nn.ReLU6()

        elif kind == "HARDTANH":
            min_val = meta.get("min_val", -1.0)
            max_val = meta.get("max_val", 1.0)
            return nn.Hardtanh(min_val, max_val)

        elif kind == "SOFTPLUS":
            return nn.Softplus()

        elif kind == "SILU":
            return nn.SiLU()

        elif kind == "HARDSIGMOID":
            return nn.Hardsigmoid()

        elif kind == "HARDSWISH":
            return nn.Hardswish()

        elif kind == "MISH":
            return nn.Mish()

        elif kind == "SOFTSIGN":
            return nn.Softsign()

        # Simple unary operators (Phase 2)
        # NOTE: Wrapped in nn.Module for Sequential compatibility
        elif kind == "ABS":
            class AbsModule(nn.Module):
                def forward(self, x):
                    return torch.abs(x)
            return AbsModule()

        elif kind == "CLIP":
            # Get min/max from meta
            min_val = meta.get("min", -1.0)
            max_val = meta.get("max", 1.0)
            class ClipModule(nn.Module):
                def __init__(self, min_val, max_val):
                    super().__init__()
                    self.min_val = min_val
                    self.max_val = max_val
                def forward(self, x):
                    return torch.clamp(x, min=self.min_val, max=self.max_val)
            return ClipModule(min_val, max_val)

        elif kind == "SQUARE":
            class SquareModule(nn.Module):
                def forward(self, x):
                    return torch.square(x)
            return SquareModule()

        elif kind == "POWER":
            exponent = meta.get("exponent", 2.0)
            class PowerModule(nn.Module):
                def __init__(self, exponent):
                    super().__init__()
                    self.exponent = exponent
                def forward(self, x):
                    return torch.pow(x, self.exponent)
            return PowerModule(exponent)

        # Multi-input operations (DAG support - Phase 3)
        elif kind == "ADD":
            # ADD is a functional operation (torch.add), not an nn.Module
            # For VerifiableGraphModel, we return a function that performs addition
            # NOTE: Inputs are provided by VerifiableGraphModel.forward() via preds
            # Supports 2+ inputs via reduce
            def add_op(*inputs):
                if len(inputs) == 2:
                    return torch.add(inputs[0], inputs[1])
                else:
                    # Reduce: sum all inputs
                    result = inputs[0]
                    for inp in inputs[1:]:
                        result = torch.add(result, inp)
                    return result
            return add_op

        elif kind == "SUB":
            return lambda x, y: torch.sub(x, y)

        elif kind == "MUL":
            return lambda x, y: torch.mul(x, y)

        elif kind == "DIV":
            return lambda x, y: torch.div(x, y)

        elif kind == "POW":
            # Binary power: x ** y (different from unary POWER with fixed exponent)
            return lambda x, y: torch.pow(x, y)

        elif kind == "MAX":
            # Element-wise maximum (supports 2+ inputs via reduce)
            # VerifiableGraphModel will provide inputs as tuple if >2 preds
            def max_op(*inputs):
                result = inputs[0]
                for inp in inputs[1:]:
                    result = torch.maximum(result, inp)
                return result
            return max_op

        elif kind == "MIN":
            # Element-wise minimum (supports 2+ inputs via reduce)
            def min_op(*inputs):
                result = inputs[0]
                for inp in inputs[1:]:
                    result = torch.minimum(result, inp)
                return result
            return min_op

        elif kind == "MATMUL":
            # Matrix multiplication: A @ B
            def matmul_op(x, y):
                # Minimal shape validation: inner dimensions must match
                if x.shape[-1] != y.shape[-2]:
                    raise ValueError(
                        f"MATMUL shape mismatch: x.shape={x.shape}, y.shape={y.shape}. "
                        f"Inner dimensions must match (x.shape[-1]={x.shape[-1]} != y.shape[-2]={y.shape[-2]})"
                    )
                return torch.matmul(x, y)
            return matmul_op

        elif kind == "CONCAT":
            # Concatenate tensors along specified dimension
            concat_dim = meta.get("concat_dim", 0)
            if concat_dim is None:
                raise ValueError("CONCAT requires 'concat_dim' in meta")
            # VerifiableGraphModel will provide inputs as tuple from preds
            def concat_op(*inputs):
                # Minimal shape validation: all inputs must have same rank
                if len(inputs) < 2:
                    raise ValueError(f"CONCAT requires at least 2 inputs, got {len(inputs)}")
                ranks = [len(inp.shape) for inp in inputs]
                if len(set(ranks)) > 1:
                    shapes = [tuple(inp.shape) for inp in inputs]
                    raise ValueError(
                        f"CONCAT shape mismatch: all inputs must have same rank. Got shapes: {shapes}"
                    )
                return torch.cat(inputs, dim=concat_dim)
            return concat_op

        # Shape transformation operations (Phase 3)
        elif kind == "RESHAPE":
            target_shape = meta.get("target_shape")
            if target_shape is None:
                raise ValueError("RESHAPE requires 'target_shape' in meta")
            class ReshapeModule(nn.Module):
                def __init__(self, target_shape):
                    super().__init__()
                    self.target_shape = tuple(target_shape)
                def forward(self, x):
                    return torch.reshape(x, self.target_shape)
            return ReshapeModule(target_shape)

        elif kind == "TRANSPOSE":
            perm = meta.get("perm")
            if perm is None:
                raise ValueError("TRANSPOSE requires 'perm' in meta")
            class TransposeModule(nn.Module):
                def __init__(self, perm):
                    super().__init__()
                    self.perm = tuple(perm)
                def forward(self, x):
                    return torch.permute(x, self.perm)
            return TransposeModule(perm)

        elif kind == "SQUEEZE":
            # Compatible with both dim (single int) and dims (list)
            dims = meta.get("dims")
            if dims is None:
                dims = meta.get("dim")

            class SqueezeModule(nn.Module):
                def __init__(self, dims):
                    super().__init__()
                    self.dims = dims
                def forward(self, x):
                    if self.dims is None:
                        # Squeeze all singleton dimensions
                        return torch.squeeze(x)
                    elif isinstance(self.dims, int):
                        # Single dimension
                        return torch.squeeze(x, dim=self.dims)
                    else:
                        # Multiple dimensions: squeeze in descending order to avoid index shifting
                        result = x
                        for dim in sorted(self.dims, reverse=True):
                            result = torch.squeeze(result, dim=dim)
                        return result
            return SqueezeModule(dims)

        elif kind == "UNSQUEEZE":
            # Compatible with both dim (single int) and dims (list)
            dims = meta.get("dims")
            if dims is None:
                dims = meta.get("dim")
            if dims is None:
                raise ValueError("UNSQUEEZE requires 'dims' or 'dim' in meta")

            class UnsqueezeModule(nn.Module):
                def __init__(self, dims):
                    super().__init__()
                    self.dims = dims
                def forward(self, x):
                    if isinstance(self.dims, int):
                        return torch.unsqueeze(x, dim=self.dims)
                    else:
                        # Multiple dimensions: apply unsqueeze iteratively
                        result = x
                        for dim in sorted(self.dims):
                            result = torch.unsqueeze(result, dim=dim)
                        return result
            return UnsqueezeModule(dims)

        elif kind == "TILE":
            repeats = meta.get("repeats")
            if repeats is None:
                raise ValueError("TILE requires 'repeats' in meta")
            class TileModule(nn.Module):
                def __init__(self, repeats):
                    super().__init__()
                    self.repeats = tuple(repeats)
                def forward(self, x):
                    return torch.tile(x, self.repeats)
            return TileModule(repeats)

        elif kind == "EXPAND":
            shape = meta.get("shape")
            if shape is None:
                raise ValueError("EXPAND requires 'shape' in meta")
            # Note: expand returns a view, not a copy
            class ExpandModule(nn.Module):
                def __init__(self, shape):
                    super().__init__()
                    self.shape = shape
                def forward(self, x):
                    return x.expand(*self.shape)
            return ExpandModule(shape)

        elif kind == "SLICE":
            starts = meta.get("starts")
            ends = meta.get("ends")
            axes = meta.get("axes")
            steps = meta.get("steps")

            if starts is None or ends is None:
                raise ValueError("SLICE requires 'starts' and 'ends' in meta")

            class SliceModule(nn.Module):
                def __init__(self, starts, ends, axes, steps):
                    super().__init__()
                    self.starts = starts
                    self.ends = ends
                    self.axes = axes
                    self.steps = steps
                def forward(self, x):
                    # Default: slice all dimensions if axes not specified
                    if self.axes is None:
                        # Apply slices to first len(starts) dimensions
                        slices = [slice(s, e, step if self.steps else None)
                                 for s, e, step in zip(self.starts, self.ends, self.steps or [None]*len(self.starts))]
                    else:
                        # Apply slices only to specified axes
                        slices = [slice(None)] * x.ndim
                        for axis, start, end in zip(self.axes, self.starts, self.ends):
                            step = self.steps[self.axes.index(axis)] if self.steps else None
                            slices[axis] = slice(start, end, step)
                    return x[tuple(slices)]
            return SliceModule(starts, ends, axes, steps)

        elif kind == "GATHER":
            # GATHER semantics: Uses torch.index_select (not torch.gather)
            # Rationale: interval_tf/tf_mlp.py:tf_gather uses torch.index_select with 1D indices,
            # which matches numpy.take and TensorFlow's gather with axis parameter.
            # This is simpler than torch.gather which requires indices with same rank as input.
            # Compatible with axis/dim field names
            axis = meta.get("axis", meta.get("dim", 0))
            # Compatible with indices/index field names
            indices = None
            if act_layer is not None:
                indices = meta.get("indices")
                if indices is None:
                    indices = act_layer.params.get("indices")
                if indices is None:
                    indices = act_layer.params.get("index")
            else:
                indices = meta.get("indices")

            if indices is None:
                raise ValueError("GATHER requires 'indices' or 'index' in params or meta")

            class GatherModule(nn.Module):
                def __init__(self, axis, indices):
                    super().__init__()
                    self.axis = axis
                    # Store indices as buffer if it's already a tensor
                    if isinstance(indices, torch.Tensor):
                        self.register_buffer('indices', indices)
                    else:
                        self.indices = indices
                def forward(self, x):
                    idx = self.indices
                    if not isinstance(idx, torch.Tensor):
                        idx = torch.tensor(idx, dtype=torch.long, device=x.device)
                    elif idx.device != x.device:
                        idx = idx.to(device=x.device)
                    return torch.index_select(x, dim=self.axis, index=idx)
            return GatherModule(axis, indices)

        elif kind == "INDEX_SELECT":
            # Compatible with dim field name
            dim = meta.get("dim")
            if dim is None:
                raise ValueError("INDEX_SELECT requires 'dim' in meta")

            # Compatible with indices/index field names
            indices = None
            if act_layer is not None:
                indices = meta.get("indices")
                if indices is None:
                    indices = act_layer.params.get("indices")
                if indices is None:
                    indices = act_layer.params.get("index")
            else:
                indices = meta.get("indices")

            if indices is None:
                raise ValueError("INDEX_SELECT requires 'indices' or 'index' in params or meta")

            class IndexSelectModule(nn.Module):
                def __init__(self, dim, indices):
                    super().__init__()
                    self.dim = dim
                    # Store indices as buffer if it's already a tensor
                    if isinstance(indices, torch.Tensor):
                        self.register_buffer('indices', indices)
                    else:
                        self.indices = indices
                def forward(self, x):
                    idx = self.indices
                    if not isinstance(idx, torch.Tensor):
                        idx = torch.tensor(idx, dtype=torch.long, device=x.device)
                    elif idx.device != x.device:
                        idx = idx.to(device=x.device)
                    return torch.index_select(x, dim=self.dim, index=idx)
            return IndexSelectModule(dim, indices)

        # Skip or warn about unsupported layers
        else:
            if self.strict:
                # In strict mode, don't return None - raise immediately
                # Note: act_layer may be None, so we can't always access layer_id here
                layer_info = f" at layer {act_layer.id}" if act_layer is not None else ""
                raise ValueError(
                    f"Strict mode: Layer kind '{kind}' not implemented in _create_torch_layer(){layer_info}. "
                    f"Add implementation for this layer type or disable strict mode."
                )
            logger.warning(f"Unsupported layer kind '{kind}' - skipping")
            return None


# ============================================================================
# SELF-TEST: Minimal Residual Network (Phase 0+1 Validation)
# ============================================================================

def _self_test_minimal_residual():
    """
    Minimal residual network test for DAG conversion validation.

    Network structure:
        INPUT → INPUT_SPEC → DENSE (branch1)
                            ↘
                              ADD → ASSERT
                            ↗
                           DENSE (branch2)

    Tests:
        1. DAG conversion with preds-based multi-input
        2. Strict mode enforcement
        3. Spec layer constraint checking
        4. Forward pass execution
    """
    import torch
    from act.back_end.core import Net
    from act.back_end.layer_util import create_layer
    from act.front_end.specs import InKind, OutKind

    print("\n" + "="*80)
    print("SELF-TEST: Minimal Residual Network (DAG Conversion)")
    print("="*80)

    # Create minimal residual network
    layers = []

    # Layer 0: INPUT
    layers.append(create_layer(
        id=0,
        kind="INPUT",
        params={},
        meta={"shape": [1, 4], "dtype": "torch.float64"},
        in_vars=[],
        out_vars=[0, 1, 2, 3]
    ))

    # Layer 1: INPUT_SPEC
    layers.append(create_layer(
        id=1,
        kind="INPUT_SPEC",
        params={"center": torch.zeros(4, dtype=torch.float64)},
        meta={"kind": "LINF_BALL", "eps": 0.1},
        in_vars=[0, 1, 2, 3],
        out_vars=[0, 1, 2, 3]
    ))

    # Layer 2: DENSE (branch 1 - direct path)
    W1 = torch.eye(4, dtype=torch.float64)
    b1 = torch.zeros(4, dtype=torch.float64)
    layers.append(create_layer(
        id=2,
        kind="DENSE",
        params={"W": W1, "b": b1},
        meta={"in_features": 4, "out_features": 4, "bias_enabled": True},
        in_vars=[0, 1, 2, 3],
        out_vars=[4, 5, 6, 7]
    ))

    # Layer 3: RELU (activation on branch 1)
    layers.append(create_layer(
        id=3,
        kind="RELU",
        params={},
        meta={},
        in_vars=[4, 5, 6, 7],
        out_vars=[8, 9, 10, 11]
    ))

    # Layer 4: DENSE (branch 2 - residual path from INPUT_SPEC)
    W2 = 0.5 * torch.eye(4, dtype=torch.float64)
    b2 = torch.zeros(4, dtype=torch.float64)
    layers.append(create_layer(
        id=4,
        kind="DENSE",
        params={"W": W2, "b": b2},
        meta={"in_features": 4, "out_features": 4, "bias_enabled": True, "preds_indices": [1]},
        in_vars=[0, 1, 2, 3],  # From INPUT_SPEC
        out_vars=[12, 13, 14, 15]
    ))

    # Layer 5: ADD (merge two branches)
    layers.append(create_layer(
        id=5,
        kind="ADD",
        params={},
        meta={"x_vars": [8, 9, 10, 11], "y_vars": [12, 13, 14, 15], "preds_indices": [3, 4]},
        in_vars=[8, 9, 10, 11, 12, 13, 14, 15],
        out_vars=[16, 17, 18, 19]
    ))

    # Layer 6: ASSERT
    layers.append(create_layer(
        id=6,
        kind="ASSERT",
        params={},
        meta={"kind": "TOP1_ROBUST", "y_true": 0},
        in_vars=[16, 17, 18, 19],
        out_vars=[16, 17, 18, 19]
    ))

    # Build preds and succs dictionaries for DAG structure
    preds = {
        0: [],          # INPUT: no predecessors
        1: [0],         # INPUT_SPEC: from INPUT
        2: [1],         # DENSE (branch1): from INPUT_SPEC
        3: [2],         # RELU: from DENSE
        4: [1],         # DENSE (branch2): from INPUT_SPEC (fork point!)
        5: [3, 4],      # ADD: from RELU and DENSE (branch2)
        6: [5]          # ASSERT: from ADD
    }

    succs = {
        0: [1],         # INPUT → INPUT_SPEC
        1: [2, 4],      # INPUT_SPEC → DENSE (branch1) and DENSE (branch2)
        2: [3],         # DENSE → RELU
        3: [5],         # RELU → ADD
        4: [5],         # DENSE (branch2) → ADD
        5: [6],         # ADD → ASSERT
        6: []           # ASSERT: no successors
    }

    # Create Net
    net = Net(layers=layers, preds=preds, succs=succs)
    print(f"✅ Created ACT Net with {len(layers)} layers")
    print(f"   Structure: INPUT → INPUT_SPEC → DENSE → RELU (branch1)")
    print(f"                                    ↘               ↗")
    print(f"                                      ADD (merge)")
    print(f"                                    ↗               ↘")
    print(f"                              DENSE (branch2)     ASSERT")

    # Test 1: Sequential model (should fail on ADD)
    print("\n--- Test 1: Sequential Model (should skip ADD) ---")
    try:
        converter_seq = ACTToTorch(net, use_graph_model=False, strict=False)
        model_seq = converter_seq.run()
        print(f"✅ Sequential model created (ADD skipped as expected)")
    except Exception as e:
        print(f"❌ Sequential model failed: {e}")

    # Test 2: DAG model (should succeed)
    print("\n--- Test 2: DAG Model (should support ADD) ---")
    try:
        converter_dag = ACTToTorch(net, use_graph_model=True, strict=True)
        model_dag = converter_dag.run()
        print(f"✅ DAG model created successfully")

        # Test forward pass
        input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
        print(f"\n   Input: {input_tensor.tolist()}")

        result = model_dag(input_tensor)
        print(f"   Output: {result['output'].tolist()}")
        print(f"   Input satisfied: {result['input_satisfied']} - {result['input_explanation']}")
        print(f"   Output satisfied: {result['output_satisfied']} - {result['output_explanation']}")

        # Verify computation
        # branch1: DENSE(input) -> RELU -> (1,2,3,4) after ReLU
        # branch2: DENSE(input) with W=0.5*I -> (0.5, 1, 1.5, 2)
        # ADD: (1,2,3,4) + (0.5,1,1.5,2) = (1.5, 3, 4.5, 6)
        expected = torch.tensor([[1.5, 3.0, 4.5, 6.0]], dtype=torch.float64)
        actual = result['output']
        if torch.allclose(actual, expected, atol=1e-6):
            print(f"   ✅ Output matches expected: {expected.tolist()}")
        else:
            print(f"   ❌ Output mismatch! Expected: {expected.tolist()}, Got: {actual.tolist()}")

    except Exception as e:
        print(f"❌ DAG model failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Strict mode with unsupported layer (should fail)
    print("\n--- Test 3: Strict Mode with Unsupported Layer ---")
    try:
        # Create an unsupported layer manually (bypass create_layer schema validation)
        # Insert ABS between ADD and ASSERT (middle of network)
        # ABS is in layer schema but NOT in OPERATOR_MAPPING (minimal set)
        from act.back_end.core import Layer
        unsupported_layer = Layer(
            id=7,
            kind="ABS",  # Not in OPERATOR_MAPPING (minimal set), but in layer schema
            params={},
            meta={},
            in_vars=[16, 17, 18, 19],
            out_vars=[20, 21, 22, 23]
        )

        # Rebuild layers with SOFTMAX inserted between ADD and ASSERT
        layers_with_unsupported = layers[:6].copy()  # INPUT, INPUT_SPEC, DENSE, RELU, DENSE, ADD
        layers_with_unsupported.append(unsupported_layer)  # SOFTMAX

        # Re-create ASSERT to follow SOFTMAX
        layers_with_unsupported.append(create_layer(
            id=8,
            kind="ASSERT",
            params={},
            meta={"kind": "TOP1_ROBUST", "y_true": 0},
            in_vars=[20, 21, 22, 23],
            out_vars=[20, 21, 22, 23]
        ))

        # Rebuild preds/succs
        preds_unsupported = {0: [], 1: [0], 2: [1], 3: [2], 4: [1], 5: [3, 4], 7: [5], 8: [7]}
        succs_unsupported = {0: [1], 1: [2, 4], 2: [3], 3: [5], 4: [5], 5: [7], 7: [8], 8: []}

        net_unsupported = Net(layers=layers_with_unsupported, preds=preds_unsupported, succs=succs_unsupported)

        converter_strict = ACTToTorch(net_unsupported, use_graph_model=True, strict=True)
        model_strict = converter_strict.run()
        print(f"❌ Strict mode should have raised ValueError for SOFTMAX (not in minimal OPERATOR_MAPPING)")
    except ValueError as e:
        if "Strict mode" in str(e):
            print(f"✅ Strict mode correctly raised: {e}")
        else:
            print(f"❌ Unexpected ValueError: {e}")
    except Exception as e:
        print(f"❌ Unexpected exception: {e}")

    print("\n" + "="*80)
    print("SELF-TEST COMPLETE")
    print("="*80 + "\n")


# ============================================================================
# Main Entry Point (Self-Test)
# ============================================================================

if __name__ == "__main__":
    """Run minimal residual network self-test when executed directly."""
    _self_test_minimal_residual()
