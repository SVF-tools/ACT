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


class ONNXConversionError(Exception):
    """Exception raised when ONNX conversion fails."""
    pass


_O2T_SLICE_PATCHED = False


def _patch_onnx2torch_slice() -> None:
    """Idempotently monkey-patch onnx2torch._get_slices to return Python ints.

    Newer OnnxSlice opset builds slice() literals at runtime from numpy
    arrays, which fx.symbolic_trace cannot serialize. Done once per process.
    """
    global _O2T_SLICE_PATCHED
    if _O2T_SLICE_PATCHED:
        return
    try:
        import onnx2torch.node_converters.slice as _slice_mod
        import numpy as _np
    except ImportError:
        return
    _orig = _slice_mod._get_slices

    def _patched(starts, ends, axes, steps):
        flip_dims, pos_slices, neg_slices = _orig(starts, ends, axes, steps)

        def _norm_slice(s):
            if not isinstance(s, slice):
                return s

            def _to_int(x):
                if x is None:
                    return None
                if isinstance(x, _np.integer):
                    return int(x)
                return x
            return slice(_to_int(s.start), _to_int(s.stop), _to_int(s.step))

        flip_dims = [int(d) if isinstance(d, _np.integer) else d for d in flip_dims]
        pos_slices = [_norm_slice(s) for s in pos_slices]
        neg_slices = [_norm_slice(s) if not (s is Ellipsis) else s
                      for s in neg_slices]
        return flip_dims, pos_slices, neg_slices

    _slice_mod._get_slices = _patched
    _O2T_SLICE_PATCHED = True


def _sanitize_numpy_in_modules(model: nn.Module) -> None:
    """Cast numpy scalars / slice objects in module attrs to Python natives.

    Required for fx symbolic_trace compat with onnx2torch (OnnxSliceV9
    stores numpy.int64 in slice() literals, which fx.create_arg rejects).
    """
    import numpy as _np

    def _conv(v):
        if isinstance(v, _np.integer):
            return int(v)
        if isinstance(v, _np.floating):
            return float(v)
        if isinstance(v, _np.bool_):
            return bool(v)
        if isinstance(v, _np.ndarray):
            try:
                return torch.from_numpy(v.copy())
            except Exception:
                return v.tolist()
        if isinstance(v, slice):
            def _norm(x):
                if x is None:
                    return None
                if isinstance(x, _np.integer):
                    return int(x)
                return x
            return slice(_norm(v.start), _norm(v.stop), _norm(v.step))
        if isinstance(v, tuple):
            return tuple(_conv(x) for x in v)
        if isinstance(v, list):
            return [_conv(x) for x in v]
        if isinstance(v, dict):
            return {k: _conv(x) for k, x in v.items()}
        return v

    _SKIP = (
        "_parameters", "_buffers", "_modules",
        "_forward_hooks", "_forward_pre_hooks",
        "_backward_hooks", "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
        "_non_persistent_buffers_set",
    )
    for module in model.modules():
        for attr_name in list(vars(module).keys()):
            if attr_name in _SKIP:
                continue
            try:
                val = getattr(module, attr_name)
            except Exception:
                continue
            new_val = _conv(val)
            if new_val is not val:
                try:
                    setattr(module, attr_name, new_val)
                except (AttributeError, TypeError):
                    pass


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
        _patch_onnx2torch_slice()
        try:
            pytorch_model = convert(onnx_model)
        except Exception as convert_err:
            if simplify and "ValueType.UNKNOWN" in str(convert_err):
                logger.warning(
                    f"onnx2torch failed on simplified graph ({convert_err}); "
                    f"retrying with simplify=False"
                )
                raw_model = onnx.load(str(onnx_path))
                try:
                    from onnx import shape_inference
                    raw_model = shape_inference.infer_shapes(raw_model)
                except Exception:
                    pass
                pytorch_model = convert(raw_model)
            else:
                raise
        _sanitize_numpy_in_modules(pytorch_model)
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
