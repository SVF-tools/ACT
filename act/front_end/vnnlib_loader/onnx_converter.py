#===- act/front_end/vnnlib/onnx_converter.py - ONNX to PyTorch -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Convert ONNX models to PyTorch nn.Module for unified verification interface.
#   Supports model validation and shape inference.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_QDQ_REGISTERED = False


def _register_qdq_converters() -> None:
    """Register ACT-local onnx2torch converters for ONNX QuantizeLinear /
    DequantizeLinear (opset 10/13). Idempotent; imports onnx2torch lazily so
    non-VNNLIB workflows never require the onnx stack."""
    global _QDQ_REGISTERED
    if _QDQ_REGISTERED:
        return

    from typing import cast

    from onnx2torch.node_converters.registry import add_converter
    from onnx2torch.onnx_graph import OnnxGraph
    from onnx2torch.onnx_node import OnnxNode
    from onnx2torch.utils.common import (
        OnnxToTorchModule,
        OperationConverterResult,
        get_const_value,
        onnx_mapping_from_node,
    )

    def _reshape_axis_param(param: torch.Tensor, x: torch.Tensor, axis) -> torch.Tensor:
        if param.numel() == 1 or axis is None or param.dim() != 1 or x.dim() == 0:
            return param
        ax = axis + x.dim() if axis < 0 else axis
        shape = [1] * x.dim()
        shape[ax] = int(param.numel())
        return param.reshape(shape)

    def _qrange_from_zero_point(zero_point: torch.Tensor):
        if zero_point.dtype == torch.int8:
            return -128, 127, "int8"
        if zero_point.dtype == torch.uint8:
            return 0, 255, "uint8"
        if zero_point.dtype == torch.int32:
            return -(2**31), 2**31 - 1, "int32"
        raise NotImplementedError(f"QuantizeLinear zero_point dtype {zero_point.dtype} is not supported")

    class OnnxQuantizeLinear(nn.Module, OnnxToTorchModule):
        def __init__(self, scale, zero_point, axis=None):
            super().__init__()
            self.register_buffer("scale", scale.detach().clone().to(dtype=torch.float32))
            self.register_buffer("zero_point", zero_point.detach().clone())
            self.axis = axis
            self.qmin, self.qmax, self.dtype_name = _qrange_from_zero_point(cast(torch.Tensor, self.zero_point))

        def forward(self, x, scale=None, zero_point=None):
            s = _reshape_axis_param(cast(torch.Tensor, self.scale).to(device=x.device, dtype=x.dtype), x, self.axis)
            zp = _reshape_axis_param(cast(torch.Tensor, self.zero_point).to(device=x.device, dtype=x.dtype), x, self.axis)
            return s * torch.clamp(torch.round(x / s), min=float(self.qmin) - zp, max=float(self.qmax) - zp)

    class OnnxDequantizeLinear(nn.Module, OnnxToTorchModule):
        def __init__(self, scale, zero_point, axis=None):
            super().__init__()
            self.register_buffer("scale", scale.detach().clone().to(dtype=torch.float32))
            self.register_buffer("zero_point", zero_point.detach().clone())
            self.axis = axis
            self.qmin, self.qmax, self.dtype_name = _qrange_from_zero_point(cast(torch.Tensor, self.zero_point))

        def forward(self, q, scale=None, zero_point=None):
            dtype = torch.float32 if not q.is_floating_point() else q.dtype
            qf = q.to(dtype=dtype)
            s = _reshape_axis_param(cast(torch.Tensor, self.scale).to(device=q.device, dtype=dtype), qf, self.axis)
            zp = _reshape_axis_param(cast(torch.Tensor, self.zero_point).to(device=q.device, dtype=dtype), qf, self.axis)
            return s * (qf - zp)

    def _axis(node: "OnnxNode"):
        raw = node.attributes.get("axis")
        return None if raw is None else int(raw)

    def _const_tensor(name: str, graph: "OnnxGraph") -> torch.Tensor:
        value = get_const_value(name, graph)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        return value

    @add_converter(operation_type="QuantizeLinear", version=10)
    @add_converter(operation_type="QuantizeLinear", version=13)
    def _q(node: "OnnxNode", graph: "OnnxGraph") -> "OperationConverterResult":
        return OperationConverterResult(
            torch_module=OnnxQuantizeLinear(_const_tensor(node.input_values[1], graph),
                                            _const_tensor(node.input_values[2], graph), _axis(node)),
            onnx_mapping=onnx_mapping_from_node(node=node),
        )

    @add_converter(operation_type="DequantizeLinear", version=10)
    @add_converter(operation_type="DequantizeLinear", version=13)
    def _dq(node: "OnnxNode", graph: "OnnxGraph") -> "OperationConverterResult":
        return OperationConverterResult(
            torch_module=OnnxDequantizeLinear(_const_tensor(node.input_values[1], graph),
                                              _const_tensor(node.input_values[2], graph), _axis(node)),
            onnx_mapping=onnx_mapping_from_node(node=node),
        )

    _QDQ_REGISTERED = True


class ONNXConversionError(Exception):
    """Exception raised when ONNX conversion fails."""
    pass


def _fold_dequantize_initializers(onnx_model):
    """Fold DequantizeLinear nodes whose data input is an initializer.

    onnx2torch Conv/Gemm converters expect weights/biases as initializers, not
    computed node outputs. This rewrite is exact: y = scale * (q - zero_point),
    including per-axis scale/zero_point via the ONNX axis attribute.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    init_map = {init.name: numpy_helper.to_array(init) for init in onnx_model.graph.initializer}
    keep_nodes = []
    folded = 0
    for node in onnx_model.graph.node:
        if node.op_type != 'DequantizeLinear' or len(node.input) < 3 or node.input[0] not in init_map:
            keep_nodes.append(node)
            continue
        q = init_map[node.input[0]]
        scale = init_map.get(node.input[1])
        zp = init_map.get(node.input[2])
        if scale is None or zp is None:
            keep_nodes.append(node)
            continue
        axis = None
        for attr in node.attribute:
            if attr.name == 'axis':
                axis = int(onnx.helper.get_attribute_value(attr))
                break
        qf = q.astype(np.float32)
        sf = scale.astype(np.float32)
        zpf = zp.astype(np.float32)
        if axis is not None and sf.ndim == 1 and sf.size != 1 and qf.ndim > 0:
            ax = axis + qf.ndim if axis < 0 else axis
            shape = [1] * qf.ndim
            shape[ax] = sf.size
            sf = sf.reshape(shape)
            zpf = zpf.reshape(shape)
        value = (sf * (qf - zpf)).astype(np.float32)
        onnx_model.graph.initializer.append(numpy_helper.from_array(value, name=node.output[0]))
        folded += 1
    if folded:
        del onnx_model.graph.node[:]
        onnx_model.graph.node.extend(keep_nodes)
        logger.info(f"Folded {folded} constant DequantizeLinear node(s) into initializers")
    return onnx_model


def _preprocess_onnx_for_onnx2torch(onnx_model):
    """Workarounds for onnx2torch quirks. Called on both main and retry paths.

    1. Symbolic batch dim (vit_2023): set first ``dim_value=0`` → 1. Must
       ``ClearField('dim_param')`` first since ``dim_value`` / ``dim_param``
       are a protobuf oneof. Only normalise the first dim; leave variable
       spatial dims alone so we don't mask real shape-concreteness errors.
    2. Empty Clip max input (cctsdb_yolo_2023): trim trailing empty input
       slots so onnx2torch's clip.py doesn't see them as present.
    """
    for inp in onnx_model.graph.input:
        dims = list(inp.type.tensor_type.shape.dim)
        if dims and dims[0].dim_value == 0:
            dims[0].ClearField('dim_param')
            dims[0].dim_value = 1
    for node in onnx_model.graph.node:
        if node.op_type == 'Clip':
            while len(node.input) > 1 and not node.input[-1]:
                del node.input[-1]
    onnx_model = _fold_dequantize_initializers(onnx_model)
    return onnx_model


def convert_onnx_to_pytorch(
    onnx_path: Path,
    simplify: bool = True
) -> nn.Module:
    """
    Convert ONNX model to PyTorch nn.Module.
    
    Args:
        onnx_path: Path to .onnx file
        simplify: Whether to simplify ONNX model before conversion
        
    Returns:
        PyTorch nn.Module equivalent to ONNX model
        
    Raises:
        ONNXConversionError: If conversion fails
    """
    if not onnx_path.exists():
        raise ONNXConversionError(f"ONNX file not found: {onnx_path}")
    
    try:
        # Import here to avoid requiring onnx for non-VNNLIB workflows
        import onnx
        _register_qdq_converters()
        from onnx2torch import convert
        
        # Load ONNX model
        logger.info(f"Loading ONNX model from {onnx_path}")
        onnx_model = onnx.load(str(onnx_path))
        
        # Upgrade old opsets for onnx2torch compatibility (e.g. ACAS Xu ships opset 8)
        try:
            from onnx import version_converter
            current_opset = max((op.version for op in onnx_model.opset_import if not op.domain or op.domain == 'ai.onnx'), default=0)
            if 0 < current_opset < 13:
                logger.info(f"Upgrading ONNX opset {current_opset} → 13")
                onnx_model = version_converter.convert_version(onnx_model, 13)
        except Exception as e:
            logger.warning(f"Opset upgrade failed ({e}), proceeding with original opset")
        
        onnx_model = _preprocess_onnx_for_onnx2torch(onnx_model)

        # Optionally simplify
        if simplify:
            try:
                import onnxsim
                logger.info("Simplifying ONNX model")
                onnx_model, check = onnxsim.simplify(onnx_model)
                if not check:
                    logger.warning("ONNX simplification check failed, using original model")
            except ImportError:
                logger.warning("onnxsim not available, skipping simplification")
            except Exception as e:
                logger.warning(f"ONNX simplification failed: {e}, using original model")

        # Propagate types/shapes through the graph after any earlier rewrites.
        # VNN-COMP ONNX files can leave intermediate values with ValueType.UNKNOWN
        # (nn4sys) or lose annotations during onnxsim simplification; without this
        # step onnx2torch raises "Got unexpected input value type".
        try:
            from onnx import shape_inference
            onnx_model = shape_inference.infer_shapes(onnx_model)
        except Exception as e:
            logger.warning(f"ONNX shape inference failed ({e}); proceeding without it")

        # Convert to PyTorch. Simplification occasionally leaves intermediate
        # values with ValueType.UNKNOWN (nn4sys); only retry-without-simplify
        # for that specific upstream error so unrelated conversion bugs
        # (unsupported ops, shape errors) still surface normally.
        logger.info("Converting ONNX to PyTorch")
        try:
            pytorch_model = convert(onnx_model)
        except Exception as convert_err:
            if simplify and "ValueType.UNKNOWN" in str(convert_err):
                logger.warning(
                    f"onnx2torch failed on simplified graph ({convert_err}); "
                    f"retrying with simplify=False"
                )
                raw_model = onnx.load(str(onnx_path))
                # Apply the same preprocessing here too -- without it, the
                # fallback can hit the very issues the main path's
                # _preprocess_onnx_for_onnx2torch was added to fix.
                raw_model = _preprocess_onnx_for_onnx2torch(raw_model)
                try:
                    from onnx import shape_inference
                    raw_model = shape_inference.infer_shapes(raw_model)
                except Exception as e:
                    # Intentional: shape inference is best-effort; converter handles missing shapes downstream.
                    logger.debug("suppressed: %s", e)
                pytorch_model = convert(raw_model)
            else:
                raise
        pytorch_model.eval()
        
        # Convert model to match device_manager settings
        try:
            from act.util.device_manager import get_default_device, get_default_dtype
            target_device = get_default_device()
            target_dtype = get_default_dtype()
            
            # Move model to target device and dtype
            pytorch_model = pytorch_model.to(dtype=target_dtype, device=target_device)
            logger.info(f"Converted model to device={target_device}, dtype={target_dtype}")
        except Exception as e:
            logger.warning(f"Could not apply device_manager settings: {e}")
        
        logger.info(f"Successfully converted ONNX model: {onnx_path.name}")
        return pytorch_model
        
    except ImportError as e:
        raise ONNXConversionError(
            f"Missing dependency for ONNX conversion: {e}\n"
            "Install with: pip install onnx onnx2torch onnx-simplifier"
        )
    except Exception as e:
        raise ONNXConversionError(f"Failed to convert {onnx_path}: {str(e)}")


def get_onnx_input_shape(onnx_path: Path) -> Tuple[int, ...]:
    """
    Extract input shape from ONNX model.
    
    Args:
        onnx_path: Path to .onnx file
        
    Returns:
        Input shape tuple WITH batch=1 (normalized to (1, C, H, W) format)
        
    Raises:
        ONNXConversionError: If shape extraction fails
    """
    try:
        import onnx
        
        onnx_model = onnx.load(str(onnx_path))
        graph = onnx_model.graph
        
        if not graph.input:
            raise ONNXConversionError("ONNX model has no inputs")
        
        # Get first input tensor
        input_tensor = graph.input[0]
        shape = _extract_shape_from_tensor(input_tensor)
        
        # Handle batch dimension - keep original, but normalize dynamic batch
        if not shape:
            raise ONNXConversionError("Failed to extract valid shape from ONNX model")
        
        if shape[0] == -1:
            # Dynamic batch: normalize to 1 for verification (requires concrete shape)
            shape = (1,) + tuple(shape[1:])
            logger.info(f"Normalized dynamic batch to 1: {shape}")
        else:
            # Keep original batch dimension (whether 1, 32, etc.)
            logger.info(f"Extracted input shape: {shape}")
            if shape[0] != 1:
                logger.warning(
                    f"ONNX model has batch size {shape[0]}, but verification "
                    f"assumes batch=1. Results may be incorrect."
                )
        
        return tuple(shape)
        
    except ImportError:
        raise ONNXConversionError("onnx library not installed")
    except Exception as e:
        raise ONNXConversionError(f"Failed to extract shape from {onnx_path}: {str(e)}")


def get_onnx_output_shape(onnx_path: Path) -> Tuple[int, ...]:
    """
    Extract output shape from ONNX model.
    
    Args:
        onnx_path: Path to .onnx file
        
    Returns:
        Output shape tuple WITH batch=1 (normalized to (1, num_classes) format)
        
    Raises:
        ONNXConversionError: If shape extraction fails
    """
    try:
        import onnx
        
        onnx_model = onnx.load(str(onnx_path))
        graph = onnx_model.graph
        
        if not graph.output:
            raise ONNXConversionError("ONNX model has no outputs")
        
        # Get first output tensor
        output_tensor = graph.output[0]
        shape = _extract_shape_from_tensor(output_tensor)
        
        # Handle batch dimension - keep original, but normalize dynamic batch
        if not shape:
            raise ONNXConversionError("Failed to extract valid shape from ONNX model")
        
        if shape[0] == -1:
            # Dynamic batch: normalize to 1 for verification (requires concrete shape)
            shape = (1,) + tuple(shape[1:])
            logger.info(f"Normalized dynamic batch to 1: {shape}")
        else:
            # Keep original batch dimension
            logger.info(f"Extracted output shape: {shape}")
            if shape[0] != 1:
                logger.warning(
                    f"ONNX model has output batch size {shape[0]}, but verification "
                    f"assumes batch=1. Results may be incorrect."
                )
        
        return tuple(shape)
        
    except ImportError:
        raise ONNXConversionError("onnx library not installed")
    except Exception as e:
        raise ONNXConversionError(f"Failed to extract output shape from {onnx_path}: {str(e)}")


def _extract_shape_from_tensor(tensor) -> list:
    """
    Extract shape from ONNX tensor proto.
    
    Args:
        tensor: ONNX tensor (ValueInfoProto)
        
    Returns:
        List of dimension sizes (-1 for dynamic dimensions)
    """
    shape = []
    
    if hasattr(tensor, 'type') and hasattr(tensor.type, 'tensor_type'):
        tensor_type = tensor.type.tensor_type
        if hasattr(tensor_type, 'shape'):
            for dim in tensor_type.shape.dim:
                if hasattr(dim, 'dim_value'):
                    shape.append(dim.dim_value if dim.dim_value > 0 else -1)
                elif hasattr(dim, 'dim_param'):
                    # Dynamic dimension
                    shape.append(-1)
    
    return shape


def test_onnx_conversion(
    onnx_path: Path,
    input_shape: Optional[Tuple[int, ...]] = None,
    batch_size: int = 1
) -> bool:
    """
    Test ONNX to PyTorch conversion with dummy input.
    
    Args:
        onnx_path: Path to .onnx file
        input_shape: Input shape (inferred from model if not provided)
        batch_size: Batch size for test input
        
    Returns:
        True if conversion successful and model runs, False otherwise
    """
    try:
        # Convert model
        pytorch_model = convert_onnx_to_pytorch(onnx_path)
        
        # Get input shape if not provided
        if input_shape is None:
            input_shape = get_onnx_input_shape(onnx_path)
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape)
        
        # Run forward pass
        with torch.no_grad():
            output = pytorch_model(dummy_input)
        
        logger.info(
            f"ONNX conversion test passed: "
            f"input {dummy_input.shape} -> output {output.shape}"
        )
        return True
        
    except Exception as e:
        logger.error(f"ONNX conversion test failed: {e}")
        return False


def get_onnx_metadata(onnx_path: Path) -> dict:
    """
    Extract metadata from ONNX model.
    
    Args:
        onnx_path: Path to .onnx file
        
    Returns:
        Dict with model metadata (producer, version, shapes, etc.)
    """
    try:
        import onnx
        
        onnx_model = onnx.load(str(onnx_path))
        
        metadata = {
            'producer_name': onnx_model.producer_name,
            'producer_version': onnx_model.producer_version,
            'ir_version': onnx_model.ir_version,
            'opset_version': None,
            'input_shapes': [],
            'output_shapes': []
        }
        
        # Get opset version
        if onnx_model.opset_import:
            metadata['opset_version'] = onnx_model.opset_import[0].version
        
        # Get input/output shapes
        graph = onnx_model.graph
        
        for inp in graph.input:
            shape = _extract_shape_from_tensor(inp)
            metadata['input_shapes'].append({
                'name': inp.name,
                'shape': shape
            })
        
        for out in graph.output:
            shape = _extract_shape_from_tensor(out)
            metadata['output_shapes'].append({
                'name': out.name,
                'shape': shape
            })
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to extract ONNX metadata: {e}")
        return {}


def validate_onnx_file(onnx_path: Path) -> bool:
    """
    Validate that an ONNX file is well-formed.
    
    Args:
        onnx_path: Path to .onnx file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        import onnx
        
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info(f"ONNX model validated: {onnx_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        return False


def convert_and_save_pytorch(
    onnx_path: Path,
    output_path: Optional[Path] = None,
    simplify: bool = True
) -> Path:
    """
    Convert ONNX model to PyTorch and save as .pt file.
    
    Args:
        onnx_path: Path to .onnx file
        output_path: Path for .pt file (defaults to same dir as ONNX)
        simplify: Whether to simplify ONNX before conversion
        
    Returns:
        Path to saved .pt file
        
    Raises:
        ONNXConversionError: If conversion or saving fails
    """
    try:
        # Convert to PyTorch
        pytorch_model = convert_onnx_to_pytorch(onnx_path, simplify=simplify)
        
        # Determine output path
        if output_path is None:
            output_path = onnx_path.with_suffix('.pt')
        
        # Save model
        torch.save(pytorch_model.state_dict(), output_path)
        logger.info(f"Saved PyTorch model to {output_path}")
        
        return output_path
        
    except Exception as e:
        raise ONNXConversionError(f"Failed to convert and save: {str(e)}")
