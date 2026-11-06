#===- act/back_end/interval_tf/tf_cnn.py - CNN Interval Transfer Func ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   CNN Interval Transfer Functions. Provides transfer functions for CNN layers
#   to enable the abstraction framework to handle convolutional neural networks.

import math
import torch
import torch.nn.functional as F
from collections.abc import Callable
from act.back_end.core import Bounds, Fact, Layer
from act.back_end.interval_tf.tf_mlp import _fact
from act.back_end.utils import avgpool2d_denominators, split_weight


def tf_conv2d(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for Conv2d layer.
    
    Linearizes the convolution operation using im2col transformation.
    """
    # Extract convolution parameters
    weight = L.params["weight"]  # [out_channels, in_channels, kernel_h, kernel_w]
    assert isinstance(weight, torch.Tensor)
    bias = L.params.get("bias", None)
    if bias is not None:
        assert isinstance(bias, torch.Tensor)
    stride = L.params.get("stride", 1)
    padding = L.params.get("padding", 0)
    dilation = L.params.get("dilation", 1)
    groups = L.params.get("groups", 1)
    assert isinstance(groups, int)
    
    # Normalize stride/padding/dilation to tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # Get weight dimensions
    out_channels, in_channels_per_group, kernel_h, kernel_w = weight.shape
    in_channels = in_channels_per_group * groups
    
    B_in = Bin.lb.shape[0]
    actual_input_size = Bin.lb[0].numel()
    if Bin.lb.shape != Bin.ub.shape:
        raise ValueError(
            f"conv2d layer {L.id}: lower/upper shape mismatch "
            f"{tuple(Bin.lb.shape)} vs {tuple(Bin.ub.shape)}"
        )
    if Bin.lb.dim() == 4:
        _, bound_channels, in_h, in_w = Bin.lb.shape
        if bound_channels != in_channels:
            raise ValueError(
                f"conv2d layer {L.id}: bounds have {bound_channels} channels, "
                f"weight expects {in_channels}"
            )
    elif Bin.lb.dim() == 2:
        meta_shape = L.params.get("input_shape")
        if meta_shape is None or len(meta_shape) != 4:
            raise ValueError(
                f"conv2d layer {L.id}: flattened bounds require a 4-D input_shape"
            )
        _, meta_channels, in_h, in_w = (int(dim) for dim in meta_shape)
        if meta_channels != in_channels:
            raise ValueError(
                f"conv2d layer {L.id}: input_shape has {meta_channels} channels, "
                f"weight expects {in_channels}"
            )
    else:
        raise ValueError(
            f"conv2d layer {L.id}: expected 2-D flattened or 4-D bounds, "
            f"got {tuple(Bin.lb.shape)}"
        )
    if in_channels * in_h * in_w != actual_input_size:
        raise ValueError(
            f"conv2d layer {L.id}: shape {(in_channels, in_h, in_w)} has "
            f"{in_channels * in_h * in_w} elements, expected {actual_input_size}"
        )
    
    input_shape = (B_in, in_channels, in_h, in_w)
    
    # Compute output dimensions using standard conv formula
    out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    output_shape = (B_in, out_channels, out_h, out_w)
    expected_output_size = out_channels * out_h * out_w
    if out_h <= 0 or out_w <= 0 or len(L.out_vars) != expected_output_size:
        raise ValueError(
            f"conv2d layer {L.id}: computed output shape "
            f"{(out_channels, out_h, out_w)} does not match "
            f"{len(L.out_vars)} output variables"
        )
    
    # Compute bounds via torch
    lb_4d = Bin.lb.reshape(B_in, in_channels, in_h, in_w)
    ub_4d = Bin.ub.reshape(B_in, in_channels, in_h, in_w)

    conv_kw = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    lb_out, ub_out = _conv_bound_pair(F.conv2d, lb_4d, ub_4d, weight, **conv_kw)
    
    if bias is not None:
        lb_out = lb_out + bias.view(1, -1, 1, 1)
        ub_out = ub_out + bias.view(1, -1, 1, 1)

    B_output = Bounds(lb=lb_out.reshape(B_in, -1), ub=ub_out.reshape(B_in, -1))

    actual_output_size = B_output.lb.shape[1]
    spatial_size_per_channel = actual_output_size // out_channels
    if bias is not None:
        b_equiv = bias.repeat_interleave(spatial_size_per_channel)
    else:
        b_equiv = Bin.lb.new_zeros(actual_output_size)
    
    # Store conv params for constraint (no Toeplitz materialization)
    return _fact(L, B_output, "EQ", {
        "tag": f"conv2d:{L.id}",
        "weight": weight,
        "b": b_equiv,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "conv_params": {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups
        }
    })


def tf_maxpool2d(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for MaxPool2d layer.
    
    Uses interval arithmetic to bound the max pooling operation.
    """
    # Extract pooling parameters
    kernel_size = L.params["kernel_size"]
    stride = L.params.get("stride", kernel_size)
    padding = L.params.get("padding", 0)
    dilation = L.params.get("dilation", 1)
    
    # Shape information
    input_shape = L.params["input_shape"]  # [batch, channels, height, width]
    output_shape = L.params["output_shape"]  # [batch, channels, out_h, out_w]
    input_shape = tuple(int(dim) for dim in input_shape)
    output_shape = tuple(int(dim) for dim in output_shape)

    B_in = Bin.lb.shape[0]
    _, channels, in_h, in_w = input_shape
    _, _, out_h, out_w = output_shape
    
    # For max pooling, we need to consider all possible inputs in each pool window
    # The output bounds are the max of upper bounds and max of lower bounds in each window
    
    # Reshape bounds for pooling operation
    input_lb = Bin.lb.view(B_in, channels, in_h, in_w)
    input_ub = Bin.ub.view(B_in, channels, in_h, in_w)
    
    # Apply max pooling to bounds
    # For lower bound: take max of lower bounds in each window
    # For upper bound: take max of upper bounds in each window
    output_lb = F.max_pool2d(input_lb, kernel_size, stride, padding, dilation)
    output_ub = F.max_pool2d(input_ub, kernel_size, stride, padding, dilation)
    
    # Flatten output bounds
    assert tuple(output_lb.shape) == (B_in, channels, out_h, out_w), (
        f"maxpool2d output shape mismatch: got {tuple(output_lb.shape)}, expected {(B_in, channels, out_h, out_w)}"
    )
    assert output_lb[0].numel() == len(L.out_vars), (
        f"maxpool2d out_vars length {len(L.out_vars)} != output elements {output_lb[0].numel()}"
    )
    B_output = Bounds(output_lb.reshape(B_in, -1), output_ub.reshape(B_in, -1))
    
    # Create constraints for max pooling
    return _fact(L, B_output, "INEQ", {
        "tag": f"maxpool2d:{L.id}",
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "input_shape": input_shape,
        "output_shape": output_shape
    })

def tf_avgpool1d(L: Layer, Bin: Bounds) -> Fact:
    kernel_size = L.params["kernel_size"]
    stride = L.params.get("stride", kernel_size)
    padding = L.params.get("padding", 0)

    input_shape = L.params["input_shape"]
    output_shape = L.params["output_shape"]

    b, c, w = input_shape
    lb_in = Bin.lb.view(b, c, w)
    ub_in = Bin.ub.view(b, c, w)

    lb_out = F.avg_pool1d(lb_in, kernel_size, stride, padding)
    ub_out = F.avg_pool1d(ub_in, kernel_size, stride, padding)

    B_output = Bounds(lb_out.view(-1), ub_out.view(-1))
    return _fact(L, B_output, "EQ", {
        "tag": f"avgpool1d:{L.id}",
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input_shape": input_shape,
        "output_shape": output_shape
    })

def tf_maxpool3d(L: Layer, Bin: Bounds) -> Fact:
    kernel_size = L.params["kernel_size"]
    stride = L.params.get("stride", kernel_size)
    padding = L.params.get("padding", 0)
    dilation = L.params.get("dilation", 1)

    input_shape = L.params["input_shape"]   # [b, c, d, h, w]
    output_shape = L.params["output_shape"] # [b, c, od, oh, ow]

    b, c, d, h, w = input_shape
    lb_in = Bin.lb.view(b, c, d, h, w)
    ub_in = Bin.ub.view(b, c, d, h, w)

    lb_out = F.max_pool3d(lb_in, kernel_size, stride, padding, dilation)
    ub_out = F.max_pool3d(ub_in, kernel_size, stride, padding, dilation)
    assert lb_out.shape == tuple(output_shape), f"maxpool3d output shape mismatch: got {tuple(lb_out.shape)}, expected {tuple(output_shape)}"
    assert lb_out.numel() == len(L.out_vars), f"maxpool3d out_vars length {len(L.out_vars)} != output elements {lb_out.numel()}"

    B = Bounds(lb_out.view(-1), ub_out.view(-1))
    assert torch.all(B.lb <= B.ub), "maxpool3d produced invalid bounds (lb > ub)"
    return _fact(L, B, "INEQ", {
        "tag": f"maxpool3d:{L.id}",
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "input_shape": input_shape,
        "output_shape": output_shape,
    })

def tf_pad(L: Layer, Bin: Bounds) -> Fact:
    pads = L.params.get("pad", None)
    if pads is None:
        pads = L.params.get("pads", None)
    if pads is None:
        raise KeyError(f"pad/pads not found in params for PAD layer {L.id}")
    assert len(pads) % 2 == 0, f"pad expects pairs, got pads={pads}"

    mode = L.params.get("mode", "constant")
    value = float(L.params.get("value", 0.0))

    in_shape = tuple(L.params["input_shape"])
    lb_in = Bin.lb.view(*in_shape)
    ub_in = Bin.ub.view(*in_shape)

    lb_out = F.pad(lb_in, pads, mode=mode, value=value)
    ub_out = F.pad(ub_in, pads, mode=mode, value=value)
    assert lb_out.numel() == len(L.out_vars), f"pad out_vars length {len(L.out_vars)} != output elements {lb_out.numel()}"

    B = Bounds(lb_out.reshape(-1), ub_out.reshape(-1))
    assert torch.all(B.lb <= B.ub), "pad produced invalid bounds (lb > ub)"
    return _fact(L, B, "EQ", {
        "tag": f"pad:{L.id}",
        "pads": list(pads),
        "mode": mode,
        "value": value,
    })

def tf_flatten(L: Layer, Bin: Bounds) -> Fact:
    B_in = Bin.lb.shape[0]

    if "input_shape" in L.params:
        raw_input_shape = L.params["input_shape"]
        input_shape = tuple(int(dim) for dim in raw_input_shape)
    else:
        input_shape = (B_in, int(Bin.lb[0].numel()))

    if "output_shape" in L.params:
        raw_output_shape = L.params["output_shape"]
        output_shape = tuple(int(dim) for dim in raw_output_shape)
    else:
        output_shape = (B_in, int(Bin.lb[0].numel()))

    axis      = L.params.get("axis", None)        # ONNX Flatten(axis=...)
    start_dim = L.params.get("start_dim", None)   # torch.flatten(start_dim, end_dim)
    end_dim   = L.params.get("end_dim", None)

    lb_flat = Bin.lb.view(B_in, -1)
    ub_flat = Bin.ub.view(B_in, -1)
    assert lb_flat.shape[1] == len(L.out_vars), (
        f"flatten out_vars length {len(L.out_vars)} != output elements {lb_flat.shape[1]}"
    )
    if "output_shape" in L.params:
        prod_all = 1
        for dim in output_shape:
            prod_all *= int(dim)
        prod_strip = 1
        for dim in output_shape[1:]:
            prod_strip *= int(dim)
        assert lb_flat.shape[1] in (prod_all, prod_strip), (
            f"flatten output numel {lb_flat.shape[1]} matches neither "
            f"prod(output_shape)={prod_all} nor prod(output_shape[1:])={prod_strip}"
        )
    B_out = Bounds(lb_flat, ub_flat)
    # Note: bounds validity is checked in analyze.py with detailed debug info

    return _fact(L, B_out, "EQ", {
        "tag":          f"flatten:{L.id}",
        "input_shape":  input_shape,
        "output_shape": output_shape,
        "axis":         axis,
        "start_dim":    start_dim,
        "end_dim":      end_dim,
    })


def tf_avgpool2d(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for AvgPool2d layer.
    
    Uses linear transformation to handle average pooling.
    """
    # Extract pooling parameters
    kernel_size = L.params["kernel_size"]
    stride = L.params.get("stride", kernel_size)
    padding = L.params.get("padding", 0)
    ceil_mode = bool(L.params.get("ceil_mode", False))
    count_include_pad = bool(L.params.get("count_include_pad", True))
    divisor_override = L.params.get("divisor_override")
    
    # Input/output shape information
    input_shape = L.params["input_shape"]
    output_shape = L.params["output_shape"]
    input_shape = tuple(int(dim) for dim in input_shape)
    output_shape = tuple(int(dim) for dim in output_shape)

    B_in = Bin.lb.shape[0]
    _, channels, in_h, in_w = input_shape
    _, _, out_h, out_w = output_shape

    input_lb = Bin.lb.view(B_in, channels, in_h, in_w)
    input_ub = Bin.ub.view(B_in, channels, in_h, in_w)
    pool_kwargs = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
    }
    output_lb = F.avg_pool2d(input_lb, **pool_kwargs)
    output_ub = F.avg_pool2d(input_ub, **pool_kwargs)
    assert tuple(output_lb.shape) == (B_in, channels, out_h, out_w), (
        f"avgpool2d output shape mismatch: got {tuple(output_lb.shape)}, expected {(B_in, channels, out_h, out_w)}"
    )
    assert output_lb[0].numel() == len(L.out_vars), (
        f"avgpool2d out_vars length {len(L.out_vars)} != output elements {output_lb[0].numel()}"
    )
    B_output = Bounds(output_lb.reshape(B_in, -1), output_ub.reshape(B_in, -1))

    W_equiv = _avgpool2d_to_linear_matrix(
        input_shape,
        output_shape,
        kernel_size,
        stride,
        padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        device=Bin.lb.device,
        dtype=Bin.lb.dtype,
    )
    
    # Create constraints
    return _fact(L, B_output, "EQ", {
        "tag": f"avgpool2d:{L.id}",
        "W": W_equiv,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
        "input_shape": input_shape,
        "output_shape": output_shape
    })


def _avgpool2d_to_linear_matrix(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    kernel_size: int,
    stride: int,
    padding: int,
    *,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override=None,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Convert AvgPool2d to equivalent linear transformation matrix."""
    _, channels, in_h, in_w = input_shape
    _, _, out_h, out_w = output_shape

    input_flat_size = channels * in_h * in_w
    output_flat_size = channels * out_h * out_w

    W_equiv = torch.zeros(
        output_flat_size,
        input_flat_size,
        device=device,
        dtype=dtype or torch.get_default_dtype(),
    )

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    kernel_h, kernel_w = kernel_size

    c, out_y, out_x, k_y, k_x = torch.meshgrid(
        torch.arange(channels, device=W_equiv.device),
        torch.arange(out_h, device=W_equiv.device),
        torch.arange(out_w, device=W_equiv.device),
        torch.arange(kernel_h, device=W_equiv.device),
        torch.arange(kernel_w, device=W_equiv.device),
        indexing="ij",
    )

    in_y = out_y * stride[0] - padding[0] + k_y
    in_x = out_x * stride[1] - padding[1] + k_x
    valid = (in_y >= 0) & (in_y < in_h) & (in_x >= 0) & (in_x < in_w)

    denominators = avgpool2d_denominators(
        (in_h, in_w),
        (out_h, out_w),
        kernel_size,
        stride,
        padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        device=W_equiv.device,
        dtype=W_equiv.dtype,
    )
    weight_vals = denominators.reciprocal().view(1, out_h, out_w, 1, 1)

    out_idx = (c * (out_h * out_w) + out_y * out_w + out_x)[valid]
    in_idx = (c * (in_h * in_w) + in_y * in_w + in_x)[valid]
    scatter_vals = weight_vals.expand_as(valid)[valid]
    W_equiv.index_put_((out_idx, in_idx), scatter_vals, accumulate=True)

    return W_equiv


# -------- Additional CNN Layers --------

def _conv_nd_tf(
    L: Layer,
    Bin: Bounds,
    conv_fn: Callable[..., torch.Tensor],
    to_matrix: Callable[..., torch.Tensor],
    name: str,
    extra_kw: dict[str, object],
) -> Fact:
    """Shared Conv1d / Conv3d / ConvTranspose2d transfer function.

    Shapes are [batch, channels, *spatial]; ``to_matrix`` builds the dense
    constraint matrix. ``extra_kw`` holds kind-specific conv keywords
    (ConvTranspose2d's ``output_padding``), placed after ``padding``.
    """
    # Extract convolution parameters
    weight = L.params["weight"]
    assert isinstance(weight, torch.Tensor)
    bias = L.params.get("bias", None)
    if bias is not None:
        assert isinstance(bias, torch.Tensor)
    groups = L.params.get("groups", 1)
    assert isinstance(groups, int)
    conv_kw = {
        "stride": L.params.get("stride", 1),
        "padding": L.params.get("padding", 0),
        **extra_kw,
        "dilation": L.params.get("dilation", 1),
        "groups": groups,
    }

    # Input/output shape information
    input_shape = tuple(int(dim) for dim in L.params["input_shape"])
    output_shape = tuple(int(dim) for dim in L.params["output_shape"])

    B_in = Bin.lb.shape[0]
    out_channels = output_shape[1]
    lb_in = Bin.lb.view(B_in, *input_shape[1:])
    ub_in = Bin.ub.view(B_in, *input_shape[1:])

    lb_out, ub_out = _conv_bound_pair(conv_fn, lb_in, ub_in, weight, **conv_kw)
    if bias is not None:
        bias_view = bias.view(1, -1, *([1] * (lb_out.dim() - 2)))
        lb_out = lb_out + bias_view
        ub_out = ub_out + bias_view
    expected_shape = (B_in, *output_shape[1:])
    assert tuple(lb_out.shape) == expected_shape, (
        f"{name} output shape mismatch: got {tuple(lb_out.shape)}, expected {expected_shape}"
    )
    assert lb_out[0].numel() == len(L.out_vars), (
        f"{name} out_vars length {len(L.out_vars)} != output elements {lb_out[0].numel()}"
    )
    B_output = Bounds(lb_out.reshape(B_in, -1), ub_out.reshape(B_in, -1))

    W_equiv = to_matrix(weight, input_shape, output_shape, **conv_kw)

    spatial_size = math.prod(output_shape[2:])
    if bias is not None:
        b_equiv = bias.repeat(spatial_size)
    else:
        b_equiv = Bin.lb.new_zeros(out_channels * spatial_size)
    
    # Create constraints
    return _fact(L, B_output, "EQ", {
        "tag": f"{name}:{L.id}",
        "W": W_equiv,
        "b": b_equiv,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "conv_params": conv_kw,
    })


def tf_conv1d(L: Layer, Bin: Bounds) -> Fact:
    """Transfer function for Conv1d layer."""
    # Conv1d weight: [out_channels, in_channels, kernel_w]
    # Conv1d input_shape:  [batch, channels, width]
    # Conv1d output_shape: [batch, out_channels, out_w]
    return _conv_nd_tf(L, Bin, F.conv1d, _conv1d_to_linear_matrix, "conv1d", {})


def tf_conv3d(L: Layer, Bin: Bounds) -> Fact:
    """Transfer function for Conv3d layer."""
    # Conv3d weight: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
    # Conv3d input_shape:  [batch, channels, depth, height, width]
    # Conv3d output_shape: [batch, out_channels, out_d, out_h, out_w]
    return _conv_nd_tf(L, Bin, F.conv3d, _conv3d_to_linear_matrix, "conv3d", {})


def tf_convtranspose2d(L: Layer, Bin: Bounds) -> Fact:
    """Transfer function for ConvTranspose2d layer."""
    # ConvTranspose2d weight: [in_channels, out_channels, kernel_h, kernel_w]
    # (channel order reversed relative to Conv2d)
    # Extract parameters
    extra_kw = {"output_padding": L.params.get("output_padding", 0)}
    return _conv_nd_tf(
        L, Bin, F.conv_transpose2d, _convtranspose2d_to_linear_matrix, "convtranspose2d", extra_kw
    )

def tf_upsample(L: Layer, Bin: Bounds) -> Fact:
    # input_shape comes through as a list after JSON deserialization;
    # coerce to tuple of ints for downstream torch ops.
    in_shape = tuple(int(dim) for dim in L.params["input_shape"])
    B_in = Bin.lb.shape[0]
    x_lb = Bin.lb.view(B_in, *in_shape[1:])
    x_ub = Bin.ub.view(B_in, *in_shape[1:])

    size = L.params.get("size", None)
    scale_factor = L.params.get("scale_factor", None)
    mode = L.params.get("mode", "nearest")
    align_corners = bool(L.params.get("align_corners", False))
    assert size is not None or scale_factor is not None, "upsample requires size or scale_factor"

    # F.interpolate scale_factor must be float or tuple of float
    y_lb = F.interpolate(
        x_lb,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners if "linear" in mode else None,
    )
    y_ub = F.interpolate(
        x_ub,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners if "linear" in mode else None,
    )

    if "output_shape" in L.params:
        out_shape = tuple(int(dim) for dim in L.params["output_shape"])
        expected_shape = (B_in, *out_shape[1:])
        assert tuple(y_lb.shape) == expected_shape, f"upsample output shape mismatch: got {tuple(y_lb.shape)}, expected {expected_shape}"
    assert y_lb[0].numel() == len(L.out_vars), f"upsample out_vars length {len(L.out_vars)} != output elements {y_lb[0].numel()}"

    B = Bounds(y_lb.reshape(B_in, -1), y_ub.reshape(B_in, -1))
    assert torch.all(B.lb <= B.ub), "upsample produced invalid bounds (lb > ub)"
    return _fact(L, B, "EQ", {
        "tag": f"upsample:{L.id}",
        "mode": mode,
        "size": list(size) if size is not None else None,
        "scale_factor": scale_factor,
        "input_shape": in_shape,
        "output_shape": list(y_lb.shape),
    })


# -------- Helper functions for new conv layers --------

def _conv_bound_pair(
    conv_fn: Callable[..., torch.Tensor],
    lb_in: torch.Tensor,
    ub_in: torch.Tensor,
    weight: torch.Tensor,
    **conv_kw,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(lb, ub) = (conv(lb_in, W+) + conv(ub_in, W-), conv(ub_in, W+) + conv(lb_in, W-))
    fused into 2 conv launches via batch-stacking. W+ = weight.clamp(min=0), W- = weight.clamp(max=0).
    """
    assert lb_in.shape == ub_in.shape, (
        f"_conv_bound_pair: lb/ub shape mismatch ({tuple(lb_in.shape)} vs {tuple(ub_in.shape)})"
    )
    W_pos, W_neg = split_weight(weight)
    B = lb_in.shape[0]
    in_for_pos = torch.cat([lb_in, ub_in], dim=0)
    in_for_neg = torch.cat([ub_in, lb_in], dim=0)
    out_pos = conv_fn(in_for_pos, W_pos, None, **conv_kw)
    out_neg = conv_fn(in_for_neg, W_neg, None, **conv_kw)
    out = out_pos + out_neg
    return out[:B], out[B:]


def _conv1d_to_linear_matrix(
    weight: torch.Tensor,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """Convert Conv1d to equivalent linear transformation matrix."""
    _, in_channels, in_w = input_shape
    _, out_channels, out_w = output_shape

    input_flat_size = in_channels * in_w
    output_flat_size = out_channels * out_w

    W_equiv = weight.new_zeros(output_flat_size, input_flat_size)

    kernel_w = weight.shape[2]
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups

    dev = weight.device
    out_c, out_x, in_c, k_x = torch.meshgrid(
        torch.arange(out_channels, device=dev),
        torch.arange(out_w, device=dev),
        torch.arange(in_channels_per_group, device=dev),
        torch.arange(kernel_w, device=dev),
        indexing="ij",
    )

    group_idx = out_c // out_channels_per_group
    actual_in_c = group_idx * in_channels_per_group + in_c
    in_x = out_x * stride - padding + k_x * dilation
    valid = (in_x >= 0) & (in_x < in_w)

    out_idx = (out_c * out_w + out_x)[valid]
    in_idx = (actual_in_c * in_w + in_x)[valid]
    scatter_vals = weight[out_c, in_c, k_x][valid]
    W_equiv.index_put_((out_idx, in_idx), scatter_vals, accumulate=True)

    return W_equiv


def _conv3d_to_linear_matrix(
    weight: torch.Tensor,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """Convert Conv3d to equivalent linear transformation matrix."""
    _, in_channels, in_d, in_h, in_w = input_shape
    _, out_channels, out_d, out_h, out_w = output_shape

    input_flat_size = in_channels * in_d * in_h * in_w
    output_flat_size = out_channels * out_d * out_h * out_w

    W_equiv = weight.new_zeros(output_flat_size, input_flat_size)

    kernel_d, kernel_h, kernel_w = weight.shape[2], weight.shape[3], weight.shape[4]

    # Handle stride/padding as tuples or ints
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups

    out_c, out_d_idx, out_h_idx, out_w_idx, in_c, k_d, k_h, k_w = torch.meshgrid(
        torch.arange(out_channels, device=weight.device),
        torch.arange(out_d, device=weight.device),
        torch.arange(out_h, device=weight.device),
        torch.arange(out_w, device=weight.device),
        torch.arange(in_channels_per_group, device=weight.device),
        torch.arange(kernel_d, device=weight.device),
        torch.arange(kernel_h, device=weight.device),
        torch.arange(kernel_w, device=weight.device),
        indexing="ij",
    )

    group_idx = out_c // out_channels_per_group
    actual_in_c = group_idx * in_channels_per_group + in_c
    in_d_idx = out_d_idx * stride[0] - padding[0] + k_d * dilation[0]
    in_h_idx = out_h_idx * stride[1] - padding[1] + k_h * dilation[1]
    in_w_idx = out_w_idx * stride[2] - padding[2] + k_w * dilation[2]
    valid = (
        (in_d_idx >= 0)
        & (in_d_idx < in_d)
        & (in_h_idx >= 0)
        & (in_h_idx < in_h)
        & (in_w_idx >= 0)
        & (in_w_idx < in_w)
    )

    out_idx = (
        out_c * out_d * out_h * out_w
        + out_d_idx * out_h * out_w
        + out_h_idx * out_w
        + out_w_idx
    )[valid]
    in_idx = (
        actual_in_c * in_d * in_h * in_w
        + in_d_idx * in_h * in_w
        + in_h_idx * in_w
        + in_w_idx
    )[valid]
    scatter_vals = weight[out_c, in_c, k_d, k_h, k_w][valid]
    W_equiv.index_put_((out_idx, in_idx), scatter_vals, accumulate=True)

    return W_equiv


def _convtranspose2d_to_linear_matrix(
    weight: torch.Tensor,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """Convert ConvTranspose2d to equivalent linear transformation matrix."""
    _, in_channels, in_h, in_w = input_shape
    _, out_channels, out_h, out_w = output_shape

    input_flat_size = in_channels * in_h * in_w
    output_flat_size = out_channels * out_h * out_w

    W_equiv = weight.new_zeros(output_flat_size, input_flat_size)

    kernel_h, kernel_w = weight.shape[2], weight.shape[3]

    # Handle stride/padding as tuples or ints
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups

    in_c, in_y, in_x, out_c, k_y, k_w = torch.meshgrid(
        torch.arange(in_channels, device=weight.device),
        torch.arange(in_h, device=weight.device),
        torch.arange(in_w, device=weight.device),
        torch.arange(out_channels_per_group, device=weight.device),
        torch.arange(kernel_h, device=weight.device),
        torch.arange(kernel_w, device=weight.device),
        indexing="ij",
    )

    group_idx = in_c // in_channels_per_group
    actual_out_c = group_idx * out_channels_per_group + out_c
    out_y = in_y * stride[0] - padding[0] + k_y * dilation[0]
    out_x = in_x * stride[1] - padding[1] + k_w * dilation[1]
    valid = (out_y >= 0) & (out_y < out_h) & (out_x >= 0) & (out_x < out_w)

    in_idx = (in_c * in_h * in_w + in_y * in_w + in_x)[valid]
    out_idx = (actual_out_c * out_h * out_w + out_y * out_w + out_x)[valid]
    scatter_vals = weight[in_c, out_c, k_y, k_w][valid]
    W_equiv.index_put_((out_idx, in_idx), scatter_vals, accumulate=True)

    return W_equiv
