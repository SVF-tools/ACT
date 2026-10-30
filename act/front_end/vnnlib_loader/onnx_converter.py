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
from typing import TYPE_CHECKING, Tuple
import logging
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from onnx import ModelProto

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
            # Registered at the file's own dtype so the model-wide .to(dtype=...)
            # reaches it; zero_point stays integral, which .to() leaves alone.
            self.register_buffer("scale", scale.detach().clone())
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
            self.register_buffer("scale", scale.detach().clone())
            self.register_buffer("zero_point", zero_point.detach().clone())
            self.axis = axis
            self.qmin, self.qmax, self.dtype_name = _qrange_from_zero_point(cast(torch.Tensor, self.zero_point))

        def forward(self, q, scale=None, zero_point=None):
            # ONNX DequantizeLinear yields the scale's type; for an integral q
            # that is the only thing naming the working float precision.
            scale_buffer = cast(torch.Tensor, self.scale)
            dtype = scale_buffer.dtype if not q.is_floating_point() else q.dtype
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
        qf = q.astype(np.float64)
        sf = scale.astype(np.float64)
        zpf = zp.astype(np.float64)
        if axis is not None and sf.ndim == 1 and sf.size != 1 and qf.ndim > 0:
            ax = axis + qf.ndim if axis < 0 else axis
            shape = [1] * qf.ndim
            shape[ax] = sf.size
            sf = sf.reshape(shape)
            zpf = zpf.reshape(shape)
        # Computed wide, then stored back at the scale's own width: a wider
        # initializer would not match the types the rest of the graph declares.
        value = (sf * (qf - zpf)).astype(scale.dtype)
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


def _ensure_unique_node_names(onnx_model: ModelProto) -> ModelProto:
    """Give graph nodes nonempty unique names without changing value edges."""
    reserved = {node.name for node in onnx_model.graph.node if node.name}
    seen: set[str] = set()
    for index, node in enumerate(onnx_model.graph.node):
        if node.name and node.name not in seen:
            seen.add(node.name)
            continue
        base = f"act_{node.op_type}_{index}"
        name = base
        suffix = 1
        while name in reserved:
            name = f"{base}_{suffix}"
            suffix += 1
        node.name = name
        reserved.add(name)
        seen.add(name)
    return onnx_model


def _prepare_onnx_for_conversion(onnx_model: ModelProto) -> ModelProto:
    """Apply the same opset and graph preparation on main and raw retry."""
    try:
        from onnx import version_converter

        current_opset = max(
            (op.version for op in onnx_model.opset_import if not op.domain or op.domain == "ai.onnx"),
            default=0,
        )
        if 0 < current_opset < 13:
            logger.info("Upgrading ONNX opset %s → 13", current_opset)
            onnx_model = version_converter.convert_version(onnx_model, 13)
    except Exception as exc:
        logger.warning("Opset upgrade failed (%s), proceeding with original opset", exc)
    return _ensure_unique_node_names(_preprocess_onnx_for_onnx2torch(onnx_model))


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
        
        onnx_model = _prepare_onnx_for_conversion(onnx_model)

        # Optionally simplify
        if simplify:
            try:
                import onnxsim
                logger.info("Simplifying ONNX model")
                onnx_model, check = onnxsim.simplify(onnx_model)
                if not check:
                    logger.warning("ONNX simplification check failed, using original model")
                onnx_model = _ensure_unique_node_names(onnx_model)
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
                raw_model = _prepare_onnx_for_conversion(raw_model)
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
            "Install with: pip install onnx onnx2torch onnxsim"
        )
    except Exception as e:
        raise ONNXConversionError(f"Failed to convert {onnx_path}: {str(e)}")


def get_onnx_input_shape(onnx_path: Path) -> Tuple[int, ...]:
    """
    Extract input shape from ONNX model.
    
    Args:
        onnx_path: Path to .onnx file
        
    Returns:
        Declared input shape; only a dynamic leading dimension is normalized
        to 1. Fixed leading extents remain model structure for the selected
        supported conversion operators.
        
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
            # A fixed leading extent remains part of one model input.
            logger.info("Extracted fixed ONNX input shape: %s", shape)
        
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
        Declared output shape; only a dynamic leading dimension is normalized
        to 1. A fixed leading extent does not imply multiple verifier lanes.
        
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
            logger.info("Extracted fixed ONNX output shape: %s", shape)
        
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


def get_onnx_metadata(onnx_path: Path) -> dict:
    """
    Extract metadata from ONNX model.
    
    Args:
        onnx_path: Path to .onnx file
        
    Returns:
        Dict with model metadata (producer, version, shapes, etc.)
    """
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


def validate_onnx_file(onnx_path: Path) -> bool:
    """
    Validate that an ONNX file is well-formed.
    
    Args:
        onnx_path: Path to .onnx file
        
    Returns:
        True if the model passes the ONNX checker, False if it is malformed.
        Load errors (missing file, unreadable protobuf) are raised.
    """
    import onnx
    
    onnx_model = onnx.load(str(onnx_path))
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        logger.warning(f"ONNX validation failed for {onnx_path.name}: {e}")
        return False
    logger.info(f"ONNX model validated: {onnx_path.name}")
    return True


