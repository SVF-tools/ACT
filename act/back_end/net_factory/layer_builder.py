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
        "hardtanh": "HARDTANH",
        "hardsigmoid": "HARDSIGMOID",
        "hardswish": "HARDSWISH",
        "silu": "SILU",
        "softplus": "SOFTPLUS",
        "mish": "MISH",
        "softsign": "SOFTSIGN",
        "gelu": "GELU",
        # A-group: unary operations
        "abs": "ABS",
        "clip": "CLIP",
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
    input_shape = [1, int(in_ch), int(H), int(W)]
    out_H, out_W = _infer_conv2d_output_hw(H, W, kernel=kernel, stride=stride, padding=padding)
    if out_H <= 0 or out_W <= 0:
        raise ValueError(f"Invalid CONV2D output shape: H={out_H}, W={out_W}")
    output_shape = [1, int(out_ch), int(out_H), int(out_W)]
    layers.append({
        "kind": "CONV2D",
        "params": {},
        "meta": {
            "in_channels": int(in_ch),
            "out_channels": int(out_ch),
            "kernel_size": int(kernel),
            "stride": int(stride),
            "padding": int(padding),
            "input_shape": input_shape,
            "output_shape": output_shape,
        },
    })
    return out_H, out_W


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
    """Append pooling layer and return output H, W."""
    input_shape = [1, int(in_ch), int(H), int(W)]
    out_H, out_W = _infer_pool2d_output_hw(H, W, kernel=kernel, stride=stride, padding=padding)
    if out_H <= 0 or out_W <= 0:
        raise ValueError(f"Invalid {kind} output shape: H={out_H}, W={out_W}")
    output_shape = [1, int(in_ch), int(out_H), int(out_W)]
    layers.append({
        "kind": kind,
        "params": {},
        "meta": {
            "kernel_size": int(kernel),
            "stride": int(stride),
            "padding": int(padding),
            "input_shape": input_shape,
            "output_shape": output_shape,
        },
    })
    return out_H, out_W


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
    input_shape = [1, int(in_ch), int(W)]
    out_W = _infer_conv1d_output_w(W, kernel=kernel, stride=stride, padding=padding)
    if out_W <= 0:
        raise ValueError(f"Invalid CONV1D output width: {out_W}")
    output_shape = [1, int(out_ch), int(out_W)]
    layers.append({
        "kind": "CONV1D",
        "params": {},
        "meta": {
            "in_channels": int(in_ch),
            "out_channels": int(out_ch),
            "kernel_size": int(kernel),
            "stride": int(stride),
            "padding": int(padding),
            "input_shape": input_shape,
            "output_shape": output_shape,
        },
    })
    return out_W


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
    input_shape = [1, int(in_ch), int(D), int(H), int(W)]
    out_D, out_H, out_W = _infer_conv3d_output_dhw(D, H, W, kernel=kernel, stride=stride, padding=padding)
    if out_D <= 0 or out_H <= 0 or out_W <= 0:
        raise ValueError(f"Invalid CONV3D output shape: D={out_D}, H={out_H}, W={out_W}")
    output_shape = [1, int(out_ch), int(out_D), int(out_H), int(out_W)]
    layers.append({
        "kind": "CONV3D",
        "params": {},
        "meta": {
            "in_channels": int(in_ch),
            "out_channels": int(out_ch),
            "kernel_size": int(kernel),
            "stride": int(stride),
            "padding": int(padding),
            "input_shape": input_shape,
            "output_shape": output_shape,
        },
    })
    return out_D, out_H, out_W


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
    input_shape = [1, int(in_ch), int(W)]
    out_W = _infer_pool1d_output_w(W, kernel=kernel, stride=stride, padding=padding)
    if out_W <= 0:
        raise ValueError(f"Invalid {kind} output width: {out_W}")
    output_shape = [1, int(in_ch), int(out_W)]
    layers.append({
        "kind": kind,
        "params": {},
        "meta": {
            "kernel_size": int(kernel),
            "stride": int(stride),
            "padding": int(padding),
            "input_shape": input_shape,
            "output_shape": output_shape,
        },
    })
    return out_W


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
    input_shape = [1, int(in_ch), int(D), int(H), int(W)]
    out_D, out_H, out_W = _infer_pool3d_output_dhw(D, H, W, kernel=kernel, stride=stride, padding=padding)
    if out_D <= 0 or out_H <= 0 or out_W <= 0:
        raise ValueError(f"Invalid {kind} output shape: D={out_D}, H={out_H}, W={out_W}")
    output_shape = [1, int(in_ch), int(out_D), int(out_H), int(out_W)]
    layers.append({
        "kind": kind,
        "params": {},
        "meta": {
            "kernel_size": int(kernel),
            "stride": int(stride),
            "padding": int(padding),
            "input_shape": input_shape,
            "output_shape": output_shape,
        },
    })
    return out_D, out_H, out_W


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


def append_bias(layers: List[Dict[str, Any]]) -> None:
    """Append BIAS layer (element-wise bias addition).

    Params will be auto-filled by factory.create_network based on previous layer output size.
    """
    layers.append({
        "kind": "BIAS",
        "params": {},  # Will be filled with {"c": tensor} by factory
        "meta": {},
    })


def append_scale(layers: List[Dict[str, Any]]) -> None:
    """Append SCALE layer (element-wise scaling).

    Params will be auto-filled by factory.create_network based on previous layer output size.
    """
    layers.append({
        "kind": "SCALE",
        "params": {},  # Will be filled with {"a": tensor} by factory
        "meta": {},
    })


def append_bn(layers: List[Dict[str, Any]]) -> None:
    """Append BN (Batch Normalization) layer.

    Params will be auto-filled by factory.create_network based on previous layer output size.
    """
    layers.append({
        "kind": "BN",
        "params": {},  # Will be filled with {"A": tensor, "c": tensor} by factory
        "meta": {},
    })


def append_act(layers: List[Dict[str, Any]], act_kind: str, *, act_params: Dict[str, Any] = None) -> None:
    """Append activation layer with optional parameters."""
    meta = {}
    if act_params:
        # Map activation parameters to meta fields
        if act_kind == "LRELU" and "lrelu_alpha" in act_params:
            meta["negative_slope"] = float(act_params["lrelu_alpha"])
        elif act_kind == "HARDTANH":
            if "hardtanh_min" in act_params:
                meta["min_val"] = float(act_params["hardtanh_min"])
            if "hardtanh_max" in act_params:
                meta["max_val"] = float(act_params["hardtanh_max"])
        elif act_kind == "HARDSIGMOID":
            if "hardsigmoid_alpha" in act_params:
                meta["alpha"] = float(act_params["hardsigmoid_alpha"])
            if "hardsigmoid_beta" in act_params:
                meta["beta"] = float(act_params["hardsigmoid_beta"])
        elif act_kind == "CLIP":
            if "clip_min" in act_params:
                meta["min"] = float(act_params["clip_min"])
            if "clip_max" in act_params:
                meta["max"] = float(act_params["clip_max"])
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
