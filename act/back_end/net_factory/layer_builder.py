#===- act/back_end/net_factory/layer_builder.py - Layer Building Logic ---===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Layer construction helpers for NetFactory.
#   - Build layer specs in-place (list of dicts).
#   - Expects already-sampled configs from ConfigSampler (in factory.py).
#   - No randomness except the caller-provided rng for CNN variants.
#   - Does not create Net/Layer objects or serialize output.
#   - All utility functions are now internal to this module.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple


# ============================================================================
# Internal Utility Functions
# ============================================================================


def _prod(shape: Tuple[int, ...]) -> int:
    """Compute product of shape dimensions."""
    p = 1
    for s in shape:
        p *= int(s)
    return p


def _ensure_batch1(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Ensure shape has batch=1 as first dimension."""
    if len(shape) < 2:
        raise ValueError(f"input_shape must include batch dim, got {shape}")
    if int(shape[0]) != 1:
        raise ValueError(f"Generator assumes batch=1, got {shape}")
    return tuple(int(x) for x in shape)


def _activation_kind(name: str) -> str:
    """Map activation name to layer kind."""
    name = (name or "relu").lower()
    mapping = {
        "relu": "RELU",
        "tanh": "TANH",
        "sigmoid": "SIGMOID",
        "lrelu": "LRELU",
        "relu6": "RELU6",
        "silu": "SILU",
        "gelu": "GELU",
        # A-group: unary operations
        "abs": "ABS",
        "square": "SQUARE",
        "power": "POWER",
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation '{name}'")
    return mapping[name]


def _infer_conv2d_output_hw(
    H: int,
    W: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> Tuple[int, int]:
    """Compute Conv2D output spatial dimensions."""
    def out_dim(x: int) -> int:
        return int((x + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1)
    return out_dim(H), out_dim(W)


def _infer_pool2d_output_hw(
    H: int,
    W: int,
    kernel: int,
    stride: int,
    padding: int = 0,
) -> Tuple[int, int]:
    """Compute Pool2D output spatial dimensions."""
    def out_dim(x: int) -> int:
        return int((x + 2 * padding - (kernel - 1) - 1) // stride + 1)
    return out_dim(H), out_dim(W)


def _infer_conv1d_output_w(
    W: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> int:
    """Compute Conv1D output width."""
    return int((W + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1)


def _infer_conv3d_output_dhw(
    D: int, H: int, W: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> Tuple[int, int, int]:
    """Compute Conv3D output spatial dimensions."""
    def out_dim(x: int) -> int:
        return int((x + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1)
    return out_dim(D), out_dim(H), out_dim(W)


def _infer_pool1d_output_w(
    W: int,
    kernel: int,
    stride: int,
    padding: int = 0,
) -> int:
    """Compute Pool1D output width."""
    return int((W + 2 * padding - (kernel - 1) - 1) // stride + 1)


def _infer_pool3d_output_dhw(
    D: int, H: int, W: int,
    kernel: int,
    stride: int,
    padding: int = 0,
) -> Tuple[int, int, int]:
    """Compute Pool3D output spatial dimensions."""
    def out_dim(x: int) -> int:
        return int((x + 2 * padding - (kernel - 1) - 1) // stride + 1)
    return out_dim(D), out_dim(H), out_dim(W)


def _as_block_param(v: Any, i: int, n_blocks: int, name: str) -> int:
    """Extract per-block parameter from int or tuple."""
    if isinstance(v, int):
        return int(v)
    t = tuple(int(x) for x in v)
    if len(t) == 1:
        return int(t[0])
    if len(t) == n_blocks:
        return int(t[i])
    raise ValueError(f"{name} must be int or tuple of len 1 or len {n_blocks}, got len={len(t)}")


# ============================================================================
# Layer Construction Functions
# ============================================================================


def _validate_conv_params(
    ndim: int,
    spatial_dims: Tuple[int, ...],
    kernel: int,
    stride: int,
    padding: int,
    dilation: int = 1,
    in_ch: int = None,
    groups: int = 1,
) -> None:
    """Validate convolution parameters with defensive assertions (Problem 2).

    Args:
        ndim: Number of spatial dimensions (1, 2, or 3)
        spatial_dims: Spatial dimensions (W,) or (H, W) or (D, H, W)
        kernel: Kernel size
        stride: Stride value
        padding: Padding value
        dilation: Dilation value (default=1)
        in_ch: Input channels (for groups validation)
        groups: Number of groups (default=1)

    Raises:
        ValueError: If parameters are invalid
    """
    if kernel <= 0:
        raise ValueError(f"kernel must be positive, got {kernel}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if padding < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")
    if dilation <= 0:
        raise ValueError(f"dilation must be positive, got {dilation}")
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")

    # Validate channel/groups divisibility
    if in_ch is not None and groups > 1:
        if in_ch % groups != 0:
            raise ValueError(
                f"in_channels ({in_ch}) must be divisible by groups ({groups})"
            )

    # Validate spatial dimensions are positive
    dim_names = ['W'] if ndim == 1 else (['H', 'W'] if ndim == 2 else ['D', 'H', 'W'])
    for name, dim in zip(dim_names, spatial_dims):
        if dim <= 0:
            raise ValueError(f"spatial dimension {name} must be positive, got {dim}")

    # Calculate effective kernel size with dilation: kernel_eff = dilation * (kernel - 1) + 1
    effective_kernel = dilation * (kernel - 1) + 1

    # Validate effective kernel doesn't exceed spatial dimension (after padding)
    for name, dim in zip(dim_names, spatial_dims):
        effective_dim = dim + 2 * padding
        if effective_kernel > effective_dim:
            raise ValueError(
                f"effective kernel size ({effective_kernel} = dilation({dilation}) * "
                f"(kernel({kernel}) - 1) + 1) exceeds effective spatial dimension "
                f"{name} ({dim} + 2*{padding} = {effective_dim})"
            )


def _validate_pool_params(
    ndim: int,
    spatial_dims: Tuple[int, ...],
    kernel: int,
    stride: int,
    padding: int,
) -> None:
    """Validate pooling parameters with defensive assertions.

    Args:
        ndim: Number of spatial dimensions (1, 2, or 3)
        spatial_dims: Spatial dimensions
        kernel: Kernel size
        stride: Stride value
        padding: Padding value

    Raises:
        ValueError: If parameters are invalid
    """
    if kernel <= 0:
        raise ValueError(f"pool kernel must be positive, got {kernel}")
    if stride <= 0:
        raise ValueError(f"pool stride must be positive, got {stride}")
    if padding < 0:
        raise ValueError(f"pool padding must be non-negative, got {padding}")

    dim_names = ['W'] if ndim == 1 else (['H', 'W'] if ndim == 2 else ['D', 'H', 'W'])
    for name, dim in zip(dim_names, spatial_dims):
        if dim <= 0:
            raise ValueError(f"spatial dimension {name} must be positive, got {dim}")
        effective_dim = dim + 2 * padding
        if kernel > effective_dim:
            raise ValueError(
                f"pool kernel size ({kernel}) exceeds effective spatial dimension "
                f"{name} ({dim} + 2*{padding} = {effective_dim})"
            )


def append_conv_nd(
    layers: List[Dict[str, Any]],
    *,
    ndim: int,
    in_ch: int,
    out_ch: int,
    spatial_dims: Tuple[int, ...],
    kernel: int,
    stride: int,
    padding: int,
    dilation: int = 1,
    groups: int = 1,
) -> Tuple[int, ...]:
    """Unified N-dimensional convolution layer builder (Problem 2 & 6).

    Handles CONV1D, CONV2D, and CONV3D layers with a single function.

    Args:
        layers: Layer specification list to append to
        ndim: Number of spatial dimensions (1, 2, or 3)
        in_ch: Input channels
        out_ch: Output channels
        spatial_dims: Input spatial dimensions as tuple (W,) or (H, W) or (D, H, W)
        kernel: Kernel size (uniform for all dimensions)
        stride: Stride value (uniform for all dimensions)
        padding: Padding value (uniform for all dimensions)
        dilation: Dilation value (default=1)
        groups: Number of groups for grouped convolution (default=1)

    Returns:
        Output spatial dimensions as tuple

    Raises:
        ValueError: If ndim not in {1, 2, 3} or invalid parameters
    """
    if ndim not in (1, 2, 3):
        raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")
    if len(spatial_dims) != ndim:
        raise ValueError(f"spatial_dims length ({len(spatial_dims)}) must match ndim ({ndim})")

    # Defensive assertions for convolution parameters (includes dilation and groups)
    _validate_conv_params(
        ndim, spatial_dims, kernel, stride, padding,
        dilation=dilation, in_ch=in_ch, groups=groups
    )

    # Additional validation: out_channels must be divisible by groups
    if groups > 1 and out_ch % groups != 0:
        raise ValueError(
            f"out_channels ({out_ch}) must be divisible by groups ({groups})"
        )

    # Compute output dimensions
    def out_dim(x: int) -> int:
        return int((x + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1)

    output_spatial = tuple(out_dim(d) for d in spatial_dims)

    # Validate output dimensions are positive
    for i, out_d in enumerate(output_spatial):
        if out_d <= 0:
            dim_name = ['W', 'H', 'D'][i] if ndim > 1 else 'W'
            raise ValueError(
                f"Invalid CONV{ndim}D output {dim_name}={out_d} "
                f"(input={spatial_dims[i]}, kernel={kernel}, stride={stride}, padding={padding})"
            )

    # Build layer kind and shapes
    kind = f"CONV{ndim}D"
    input_shape = [1, int(in_ch)] + [int(d) for d in spatial_dims]
    output_shape = [1, int(out_ch)] + [int(d) for d in output_spatial]

    layers.append({
        "kind": kind,
        "params": {},
        "meta": {
            "in_channels": int(in_ch),
            "out_channels": int(out_ch),
            "kernel_size": int(kernel),
            "stride": int(stride),
            "padding": int(padding),
            "dilation": int(dilation),
            "groups": int(groups),
            "input_shape": input_shape,
            "output_shape": output_shape,
        },
    })
    return output_spatial


def append_pool_nd(
    layers: List[Dict[str, Any]],
    *,
    ndim: int,
    kind: str,
    in_ch: int,
    spatial_dims: Tuple[int, ...],
    kernel: int,
    stride: int,
    padding: int = 0,
    dilation: int = 1,
) -> Tuple[int, ...]:
    """Unified N-dimensional pooling layer builder (Problem 6).

    Handles MAXPOOL1D/2D/3D and AVGPOOL1D/2D/3D layers with a single function.

    Args:
        layers: Layer specification list to append to
        ndim: Number of spatial dimensions (1, 2, or 3)
        kind: Pool type (e.g., "MAXPOOL2D", "AVGPOOL2D")
        in_ch: Input channels
        spatial_dims: Input spatial dimensions as tuple (W,) or (H, W) or (D, H, W)
        kernel: Kernel size
        stride: Stride value
        padding: Padding value (default=0)
        dilation: Dilation value (default=1, only for maxpool)

    Returns:
        Output spatial dimensions as tuple

    Raises:
        ValueError: If ndim not in {1, 2, 3} or invalid parameters
    """
    if ndim not in (1, 2, 3):
        raise ValueError(f"ndim must be 1, 2, or 3, got {ndim}")
    if len(spatial_dims) != ndim:
        raise ValueError(f"spatial_dims length ({len(spatial_dims)}) must match ndim ({ndim})")

    # Defensive assertions for pooling parameters
    _validate_pool_params(ndim, spatial_dims, kernel, stride, padding)

    # Compute output dimensions (pooling formula)
    def out_dim(x: int) -> int:
        return int((x + 2 * padding - (kernel - 1) - 1) // stride + 1)

    output_spatial = tuple(out_dim(d) for d in spatial_dims)

    # Validate output dimensions are positive
    for i, out_d in enumerate(output_spatial):
        if out_d <= 0:
            dim_name = ['W', 'H', 'D'][i] if ndim > 1 else 'W'
            raise ValueError(
                f"Invalid {kind} output {dim_name}={out_d} "
                f"(input={spatial_dims[i]}, kernel={kernel}, stride={stride}, padding={padding})"
            )

    # Build shapes
    input_shape = [1, int(in_ch)] + [int(d) for d in spatial_dims]
    output_shape = [1, int(in_ch)] + [int(d) for d in output_spatial]

    layers.append({
        "kind": kind,
        "params": {},
        "meta": {
            "kernel_size": int(kernel),
            "stride": int(stride),
            "padding": int(padding),
            "dilation": int(dilation),
            "input_shape": input_shape,
            "output_shape": output_shape,
        },
    })
    return output_spatial


# ============================================================================
# Convenience Wrappers (backward-compatible API)
# ============================================================================


def append_conv2d(
    layers: List[Dict[str, Any]],
    *,
    in_ch: int,
    out_ch: int,
    H: int,
    W: int,
    kernel: int,
    stride: int,
    padding: int,
) -> Tuple[int, int]:
    """Append CONV2D layer and return output H, W."""
    result = append_conv_nd(
        layers, ndim=2, in_ch=in_ch, out_ch=out_ch,
        spatial_dims=(H, W), kernel=kernel, stride=stride, padding=padding
    )
    return result[0], result[1]


def append_pool2d(
    layers: List[Dict[str, Any]],
    *,
    kind: str,
    in_ch: int,
    H: int,
    W: int,
    kernel: int,
    stride: int,
    padding: int = 0,
) -> Tuple[int, int]:
    """Append 2D pooling layer and return output H, W."""
    result = append_pool_nd(
        layers, ndim=2, kind=kind, in_ch=in_ch,
        spatial_dims=(H, W), kernel=kernel, stride=stride, padding=padding
    )
    return result[0], result[1]


def append_conv1d(
    layers: List[Dict[str, Any]],
    *,
    in_ch: int,
    out_ch: int,
    W: int,
    kernel: int,
    stride: int,
    padding: int,
) -> int:
    """Append CONV1D layer and return output W."""
    result = append_conv_nd(
        layers, ndim=1, in_ch=in_ch, out_ch=out_ch,
        spatial_dims=(W,), kernel=kernel, stride=stride, padding=padding
    )
    return result[0]


def append_conv3d(
    layers: List[Dict[str, Any]],
    *,
    in_ch: int,
    out_ch: int,
    D: int, H: int, W: int,
    kernel: int,
    stride: int,
    padding: int,
) -> Tuple[int, int, int]:
    """Append CONV3D layer and return output D, H, W."""
    result = append_conv_nd(
        layers, ndim=3, in_ch=in_ch, out_ch=out_ch,
        spatial_dims=(D, H, W), kernel=kernel, stride=stride, padding=padding
    )
    return result[0], result[1], result[2]


def append_pool1d(
    layers: List[Dict[str, Any]],
    *,
    kind: str,
    in_ch: int,
    W: int,
    kernel: int,
    stride: int,
    padding: int = 0,
) -> int:
    """Append 1D pooling layer and return output W."""
    result = append_pool_nd(
        layers, ndim=1, kind=kind, in_ch=in_ch,
        spatial_dims=(W,), kernel=kernel, stride=stride, padding=padding
    )
    return result[0]


def append_pool3d(
    layers: List[Dict[str, Any]],
    *,
    kind: str,
    in_ch: int,
    D: int, H: int, W: int,
    kernel: int,
    stride: int,
    padding: int = 0,
) -> Tuple[int, int, int]:
    """Append 3D pooling layer and return output D, H, W."""
    result = append_pool_nd(
        layers, ndim=3, kind=kind, in_ch=in_ch,
        spatial_dims=(D, H, W), kernel=kernel, stride=stride, padding=padding
    )
    return result[0], result[1], result[2]


def append_pad(
    layers: List[Dict[str, Any]],
    *,
    in_ch: int,
    H: int, W: int,
    padding: Tuple[int, int, int, int],
    mode: str = "constant",
    value: float = 0.0,
) -> Tuple[int, int]:
    """Append PAD layer and return output H, W.

    Args:
        padding: (left, right, top, bottom)
    """
    left, right, top, bottom = padding
    input_shape = [1, int(in_ch), int(H), int(W)]
    out_H = H + top + bottom
    out_W = W + left + right
    output_shape = [1, int(in_ch), int(out_H), int(out_W)]
    layers.append({
        "kind": "PAD",
        "params": {},
        "meta": {
            "pad": list(padding),
            "mode": mode,
            "value": float(value),
            "input_shape": input_shape,
            "output_shape": output_shape,
        },
    })
    return out_H, out_W


def append_upsample(
    layers: List[Dict[str, Any]],
    *,
    in_ch: int,
    H: int, W: int,
    scale_factor: float,
    mode: str = "nearest",
) -> Tuple[int, int]:
    """Append UPSAMPLE layer and return output H, W."""
    input_shape = [1, int(in_ch), int(H), int(W)]
    out_H = int(H * scale_factor)
    out_W = int(W * scale_factor)
    output_shape = [1, int(in_ch), int(out_H), int(out_W)]
    layers.append({
        "kind": "UPSAMPLE",
        "params": {},
        "meta": {
            "scale_factor": float(scale_factor),
            "mode": mode,
            "input_shape": input_shape,
            "output_shape": output_shape,
        },
    })
    return out_H, out_W


def append_dense(
    layers: List[Dict[str, Any]],
    *,
    in_features: int,
    out_features: int,
    use_bias: bool,
) -> None:
    """Append DENSE layer."""
    layers.append({
        "kind": "DENSE",
        "params": {},
        "meta": {
            "in_features": int(in_features),
            "out_features": int(out_features),
            "bias_enabled": bool(use_bias),
        },
    })


def append_bias(
    layers: List[Dict[str, Any]],
    *,
    input_shape: List[int] = None,
    output_shape: List[int] = None,
    preds_indices: List[int] = None,
) -> None:
    """Append BIAS layer (element-wise bias addition).

    Args:
        layers: Layer specification list
        input_shape: Input tensor shape (copied from predecessor for DAG consistency)
        output_shape: Output tensor shape (same as input for element-wise ops)
        preds_indices: Predecessor layer indices for DAG tracking

    Params will be auto-filled by factory.create_network based on previous layer output size.
    """
    meta = {}
    if input_shape is not None:
        meta["input_shape"] = list(input_shape)
    if output_shape is not None:
        meta["output_shape"] = list(output_shape)
    elif input_shape is not None:
        meta["output_shape"] = list(input_shape)  # Same shape for element-wise op
    if preds_indices is not None:
        meta["preds_indices"] = list(preds_indices)

    layers.append({
        "kind": "BIAS",
        "params": {},  # Will be filled with {"c": tensor} by factory
        "meta": meta,
    })


def append_scale(
    layers: List[Dict[str, Any]],
    *,
    input_shape: List[int] = None,
    output_shape: List[int] = None,
    preds_indices: List[int] = None,
) -> None:
    """Append SCALE layer (element-wise scaling).

    Args:
        layers: Layer specification list
        input_shape: Input tensor shape (copied from predecessor for DAG consistency)
        output_shape: Output tensor shape (same as input for element-wise ops)
        preds_indices: Predecessor layer indices for DAG tracking

    Params will be auto-filled by factory.create_network based on previous layer output size.
    """
    meta = {}
    if input_shape is not None:
        meta["input_shape"] = list(input_shape)
    if output_shape is not None:
        meta["output_shape"] = list(output_shape)
    elif input_shape is not None:
        meta["output_shape"] = list(input_shape)  # Same shape for element-wise op
    if preds_indices is not None:
        meta["preds_indices"] = list(preds_indices)

    layers.append({
        "kind": "SCALE",
        "params": {},  # Will be filled with {"a": tensor} by factory
        "meta": meta,
    })


def append_bn(
    layers: List[Dict[str, Any]],
    *,
    input_shape: List[int] = None,
    output_shape: List[int] = None,
    preds_indices: List[int] = None,
) -> None:
    """Append BN (Batch Normalization) layer.

    Args:
        layers: Layer specification list
        input_shape: Input tensor shape (copied from predecessor for DAG consistency)
        output_shape: Output tensor shape (same as input for BN)
        preds_indices: Predecessor layer indices for DAG tracking

    Params will be auto-filled by factory.create_network based on previous layer output size.
    """
    meta = {}
    if input_shape is not None:
        meta["input_shape"] = list(input_shape)
    if output_shape is not None:
        meta["output_shape"] = list(output_shape)
    elif input_shape is not None:
        meta["output_shape"] = list(input_shape)  # Same shape for BN
    if preds_indices is not None:
        meta["preds_indices"] = list(preds_indices)

    layers.append({
        "kind": "BN",
        "params": {},  # Will be filled with {"A": tensor, "c": tensor} by factory
        "meta": meta,
    })


def append_act(layers: List[Dict[str, Any]], act_kind: str, *, act_params: Dict[str, Any] = None) -> None:
    """Append activation layer with optional parameters."""
    meta = {}
    if act_params:
        # Map activation parameters to meta fields
        if act_kind == "LRELU" and "lrelu_alpha" in act_params:
            meta["negative_slope"] = float(act_params["lrelu_alpha"])
        elif act_kind == "POWER":
            if "power_exponent" in act_params:
                meta["exponent"] = float(act_params["power_exponent"])
    layers.append({"kind": act_kind, "params": {}, "meta": meta})


def append_add(layers: List[Dict[str, Any]], *, skip_idx: int, main_idx: int) -> None:
    """Append ADD layer for residual connections."""
    layers.append({
        "kind": "ADD",
        "params": {},
        "meta": {},
        "inputs": {"x": skip_idx, "y": main_idx},
        "preds": [skip_idx, main_idx],
    })


def append_binary_op(
    layers: List[Dict[str, Any]],
    *,
    op_kind: str,
    x_idx: int,
    y_idx: int,
) -> None:
    """Append binary operation layer (SUB/MUL/DIV/POW/MAX/MIN).

    Args:
        layers: Layer specification list
        op_kind: Operation type (SUB, MUL, DIV, POW, MAX, MIN)
        x_idx: Index of first input layer
        y_idx: Index of second input layer

    Note: Both inputs must have the same shape for element-wise operations.
    The x_vars and y_vars will be populated by factory during variable generation.
    """
    layers.append({
        "kind": op_kind,
        "params": {},
        "meta": {},  # x_vars and y_vars will be set by factory._generate_layer_variables
        "inputs": {"x": x_idx, "y": y_idx},
        "preds": [x_idx, y_idx],
    })


def append_reshape(
    layers: List[Dict[str, Any]],
    *,
    target_shape: Tuple[int, ...],
) -> None:
    """Append RESHAPE layer.

    Note: Caller must ensure target_shape has same number of elements as input.
    """
    layers.append({
        "kind": "RESHAPE",
        "params": {},
        "meta": {"target_shape": list(target_shape)},
    })


def append_transpose(
    layers: List[Dict[str, Any]],
    *,
    perm: Tuple[int, ...],
) -> None:
    """Append TRANSPOSE layer."""
    layers.append({
        "kind": "TRANSPOSE",
        "params": {},
        "meta": {"perm": list(perm)},
    })


def append_squeeze(
    layers: List[Dict[str, Any]],
    *,
    dims: Tuple[int, ...],
) -> None:
    """Append SQUEEZE layer."""
    layers.append({
        "kind": "SQUEEZE",
        "params": {},
        "meta": {"dims": list(dims)},
    })


def append_unsqueeze(
    layers: List[Dict[str, Any]],
    *,
    dims: Tuple[int, ...],
) -> None:
    """Append UNSQUEEZE layer."""
    layers.append({
        "kind": "UNSQUEEZE",
        "params": {},
        "meta": {"dims": list(dims)},
    })


def append_slice(
    layers: List[Dict[str, Any]],
    *,
    starts: List[int],
    ends: List[int],
    axes: List[int],
    input_shape: List[int],
    output_shape: List[int],
) -> None:
    """Append SLICE layer."""
    layers.append({
        "kind": "SLICE",
        "params": {},
        "meta": {
            "starts": starts,
            "ends": ends,
            "axes": axes,
            "input_shape": input_shape,
            "output_shape": output_shape,
        },
    })


def append_gather(
    layers: List[Dict[str, Any]],
    *,
    indices: List[int],
    axis: int,
    input_shape: List[int],
    output_shape: List[int],
) -> None:
    """Append GATHER layer."""
    layers.append({
        "kind": "GATHER",
        "params": {},
        "meta": {
            "indices": indices,
            "axis": axis,
            "input_shape": input_shape,
            "output_shape": output_shape,
        },
    })


def append_concat(
    layers: List[Dict[str, Any]],
    *,
    input_indices: List[int],
    concat_dim: int = 0,
) -> None:
    """Append CONCAT layer.

    Args:
        input_indices: List of layer indices to concatenate
        concat_dim: Dimension along which to concatenate
    """
    layers.append({
        "kind": "CONCAT",
        "params": {},
        "meta": {"concat_dim": concat_dim},
        "preds": input_indices,
    })


def append_matmul(
    layers: List[Dict[str, Any]],
    *,
    x_idx: int,
    y_idx: int,
) -> None:
    """Append MATMUL layer for matrix multiplication.

    Args:
        x_idx: Index of first input layer (matrix A)
        y_idx: Index of second input layer (matrix B)

    Note: Caller must ensure shapes are compatible for matrix multiplication.
    The x_vars and y_vars will be populated by factory during variable generation.
    """
    layers.append({
        "kind": "MATMUL",
        "params": {},
        "meta": {},  # x_vars and y_vars will be set by factory._generate_layer_variables
        "inputs": {"x": x_idx, "y": y_idx},
        "preds": [x_idx, y_idx],
    })


def build_mlp_layers(layers: List[Dict[str, Any]], *, cfg: Dict[str, Any]) -> None:
    """Build MLP layer sequence based on config."""
    shape = _ensure_batch1(tuple(cfg["input_shape"]))
    in_features = int(shape[1]) if len(shape) == 2 else _prod(shape[1:])

    if len(shape) > 2:
        layers.append({"kind": "FLATTEN", "params": {}, "meta": {"start_dim": 1}})

    act_kind = _activation_kind(cfg["activation"])
    use_bias = bool(cfg["use_bias"])
    variant = cfg["variant"]

    if variant == "plain":
        for h in cfg["hidden_sizes"]:
            append_dense(layers, in_features=in_features, out_features=int(h), use_bias=use_bias)
            append_act(layers, act_kind, act_params=cfg)
            in_features = int(h)
    elif variant == "block":
        width = int(cfg["block_width"])
        append_dense(layers, in_features=in_features, out_features=width, use_bias=use_bias)
        append_act(layers, act_kind, act_params=cfg)
        in_features = int(width)

        for _ in range(int(cfg["num_blocks"])):
            append_dense(layers, in_features=in_features, out_features=in_features, use_bias=use_bias)
            append_act(layers, act_kind, act_params=cfg)
            append_dense(layers, in_features=in_features, out_features=in_features, use_bias=use_bias)
            if cfg["post_block_activation"]:
                append_act(layers, act_kind, act_params=cfg)
    elif variant == "residual":
        width = int(cfg["residual_width"])
        if in_features != width:
            append_dense(layers, in_features=in_features, out_features=width, use_bias=use_bias)
            append_act(layers, act_kind, act_params=cfg)
            in_features = int(width)

        for _ in range(int(cfg["num_residual_blocks"])):
            skip_idx = len(layers) - 1
            append_dense(layers, in_features=in_features, out_features=in_features, use_bias=use_bias)
            append_act(layers, act_kind, act_params=cfg)
            append_dense(layers, in_features=in_features, out_features=in_features, use_bias=use_bias)
            main_idx = len(layers) - 1
            append_add(layers, skip_idx=skip_idx, main_idx=main_idx)
            append_act(layers, act_kind, act_params=cfg)
    else:
        raise ValueError(f"Unsupported MLP variant '{variant}'")

    append_dense(layers, in_features=in_features, out_features=int(cfg["num_classes"]), use_bias=True)


def build_cnn_layers(
    layers: List[Dict[str, Any]],
    *,
    cfg: Dict[str, Any],
    rng: random.Random,
) -> None:
    """Build CNN layer sequence based on config."""
    shape = _ensure_batch1(tuple(cfg["input_shape"]))
    if len(shape) != 4:
        raise ValueError(f"CNN2D expects input_shape=(1,C,H,W), got {shape}")
    _, in_ch, H, W = shape
    in_ch = int(in_ch)
    act_kind = _activation_kind(cfg["activation"])

    if cfg["variant"] == "plain":
        n_blocks = len(cfg["conv_channels"])
        for i, out_ch in enumerate(cfg["conv_channels"]):
            k = _as_block_param(cfg["kernel_sizes"], i, n_blocks, "kernel_sizes")
            s = _as_block_param(cfg["strides"], i, n_blocks, "strides")
            p = _as_block_param(cfg["paddings"], i, n_blocks, "paddings")

            H, W = append_conv2d(
                layers, in_ch=in_ch, out_ch=int(out_ch), H=H, W=W,
                kernel=int(k), stride=int(s), padding=int(p),
            )
            append_act(layers, act_kind, act_params=cfg)
            in_ch = int(out_ch)

            # Support for different pooling types
            use_pooling = cfg.get("use_pooling", cfg.get("use_maxpool", False))
            if use_pooling:
                pool_kind = cfg.get("pool_kind", "maxpool")
                if pool_kind == "maxpool":
                    pool_type = "MAXPOOL2D"
                elif pool_kind == "avgpool":
                    pool_type = "AVGPOOL2D"
                else:
                    pool_type = None

                if pool_type:
                    pool_kernel = int(cfg.get("pool_kernel", cfg.get("maxpool_kernel", 2)))
                    pool_stride = int(cfg.get("pool_stride", cfg.get("maxpool_stride", 2)))
                    H, W = append_pool2d(
                        layers, kind=pool_type, in_ch=in_ch, H=H, W=W,
                        kernel=pool_kernel,
                        stride=pool_stride,
                        padding=0,
                    )

        layers.append({"kind": "FLATTEN", "params": {}, "meta": {"start_dim": 1}})
        feat = int(in_ch * H * W)
        append_dense(layers, in_features=int(feat), out_features=int(cfg["fc_hidden"]), use_bias=True)
        append_act(layers, act_kind, act_params=cfg)
        append_dense(layers, in_features=int(cfg["fc_hidden"]), out_features=int(cfg["num_classes"]), use_bias=True)
        return

    elif cfg["variant"] == "residual":
        # Residual variant: Conv blocks with skip connections (ResNet-style)
        ch = int(cfg["residual_channels"])

        # Initial conv to match residual_channels
        H, W = append_conv2d(layers, in_ch=in_ch, out_ch=ch, H=H, W=W, kernel=3, stride=1, padding=1)
        append_act(layers, act_kind, act_params=cfg)

        # Residual blocks
        for _ in range(int(cfg["num_residual_blocks"])):
            skip_idx = len(layers) - 1

            # Conv -> Act -> Conv
            H, W = append_conv2d(layers, in_ch=ch, out_ch=ch, H=H, W=W, kernel=3, stride=1, padding=1)
            append_act(layers, act_kind, act_params=cfg)
            H, W = append_conv2d(layers, in_ch=ch, out_ch=ch, H=H, W=W, kernel=3, stride=1, padding=1)
            main_idx = len(layers) - 1

            # Skip connection (ADD)
            append_add(layers, skip_idx=skip_idx, main_idx=main_idx)
            append_act(layers, act_kind, act_params=cfg)

        # Head: global pooling -> flatten -> FC
        while H > 1 or W > 1:
            H, W = append_pool2d(layers, kind="AVGPOOL2D", in_ch=ch, H=H, W=W, kernel=2, stride=2, padding=0)
            if H <= 0 or W <= 0:
                raise ValueError("Invalid spatial dims after head pooling")

        layers.append({"kind": "FLATTEN", "params": {}, "meta": {"start_dim": 1}})
        feat = int(ch * H * W)
        append_dense(layers, in_features=int(feat), out_features=int(cfg["num_classes"]), use_bias=True)
        return

    # Stage variant (default for cnn2d)
    ch = int(cfg["base_channels"])
    H, W = append_conv2d(layers, in_ch=in_ch, out_ch=ch, H=H, W=W, kernel=3, stride=1, padding=1)
    append_act(layers, act_kind, act_params=cfg)

    for stage in range(int(cfg["stages"])):
        if stage > 0:
            next_ch = min(64, ch * int(cfg["channel_mult"]))
            if cfg["downsample"] == "stride2_conv":
                H, W = append_conv2d(layers, in_ch=ch, out_ch=next_ch, H=H, W=W, kernel=3, stride=2, padding=1)
                append_act(layers, act_kind, act_params=cfg)
                ch = next_ch
            else:
                pool_kind = "MAXPOOL2D" if cfg["downsample"] == "maxpool" else "AVGPOOL2D"
                H, W = append_pool2d(layers, kind=pool_kind, in_ch=ch, H=H, W=W, kernel=2, stride=2, padding=0)
                if next_ch != ch:
                    H, W = append_conv2d(layers, in_ch=ch, out_ch=next_ch, H=H, W=W, kernel=1, stride=1, padding=0)
                    append_act(layers, act_kind, act_params=cfg)
                    ch = next_ch

        for _ in range(int(cfg["blocks_per_stage"])):
            make_double_conv = (rng.random() < float(cfg["double_conv_p"]))
            if make_double_conv:
                H, W = append_conv2d(layers, in_ch=ch, out_ch=ch, H=H, W=W, kernel=3, stride=1, padding=1)
                append_act(layers, act_kind, act_params=cfg)
                H, W = append_conv2d(layers, in_ch=ch, out_ch=ch, H=H, W=W, kernel=3, stride=1, padding=1)
                append_act(layers, act_kind, act_params=cfg)
            else:
                H, W = append_conv2d(layers, in_ch=ch, out_ch=ch, H=H, W=W, kernel=3, stride=1, padding=1)
                append_act(layers, act_kind, act_params=cfg)

    while cfg["head_pool_to_1x1"] and (H > 1 or W > 1):
        H, W = append_pool2d(layers, kind="AVGPOOL2D", in_ch=ch, H=H, W=W, kernel=2, stride=2, padding=0)
        if H <= 0 or W <= 0:
            raise ValueError("Invalid spatial dims after head pooling")

    layers.append({"kind": "FLATTEN", "params": {}, "meta": {"start_dim": 1}})
    feat = int(ch * H * W)
    append_dense(layers, in_features=int(feat), out_features=int(cfg["num_classes"]), use_bias=True)
