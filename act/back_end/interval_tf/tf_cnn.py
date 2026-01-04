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

import torch
import torch.nn.functional as F
from typing import List, Tuple
from act.back_end.core import Bounds, Con, ConSet, Fact, Layer
from act.back_end.utils import affine_bounds, pwl_meta, bound_var_interval, scale_interval


def tf_conv2d(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for Conv2d layer.
    
    Linearizes the convolution operation using im2col transformation.
    """
    # Extract convolution parameters
    weight = L.params["weight"]  # [out_channels, in_channels, kernel_h, kernel_w]
    bias = L.params.get("bias", None)
    stride = L.meta.get("stride", 1)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)
    groups = L.meta.get("groups", 1)
    
    # Input shape information
    input_shape = L.meta["input_shape"]  # [batch, channels, height, width]
    output_shape = L.meta["output_shape"]  # [batch, out_channels, out_h, out_w]
    
    batch_size, in_channels, in_h, in_w = input_shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    _, _, out_h, out_w = output_shape
    
    # Flatten input bounds for processing
    input_flat_size = in_channels * in_h * in_w
    output_flat_size = out_channels * out_h * out_w
    
    # Create equivalent linear transformation matrix using im2col
    # This converts the convolution to matrix multiplication
    W_equiv = _conv2d_to_linear_matrix(
        weight, input_shape, output_shape, stride, padding, dilation, groups
    )
    
    # Apply affine transformation
    if bias is not None:
        b_equiv = bias.repeat(out_h * out_w)  # Broadcast bias across spatial dimensions
    else:
        b_equiv = torch.zeros(output_flat_size, dtype=weight.dtype, device=weight.device)
    
    # Compute bounds using affine transformation
    W_pos = torch.clamp(W_equiv, min=0)
    W_neg = torch.clamp(W_equiv, max=0)
    
    # Reshape input bounds to flat format
    input_bounds_flat = Bounds(
        Bin.lb.view(-1),  # [input_flat_size]
        Bin.ub.view(-1)   # [input_flat_size]
    )
    
    # Apply linear transformation
    B_output = affine_bounds(W_pos, W_neg, b_equiv, input_bounds_flat)
    
    # Create constraints
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"conv2d:{L.id}",
        "W": W_equiv,
        "b": b_equiv,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "conv_params": {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups
        }
    }))
    
    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)

def tf_maxpool1d(L: Layer, Bin: Bounds) -> Fact:
    kernel_size = L.meta["kernel_size"]
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)

    input_shape = L.meta["input_shape"]   # [batch, channels, width]
    output_shape = L.meta["output_shape"] # [batch, channels, out_w]

    b, c, w = input_shape
    _, _, out_w = output_shape

    lb_in = Bin.lb.view(b, c, w)
    ub_in = Bin.ub.view(b, c, w)

    lb_out = F.max_pool1d(lb_in, kernel_size, stride, padding, dilation)
    ub_out = F.max_pool1d(ub_in, kernel_size, stride, padding, dilation)
    assert lb_out.shape == (b, c, out_w), f"maxpool1d output shape mismatch: got {tuple(lb_out.shape)}, expected {(b, c, out_w)}"
    assert lb_out.numel() == len(L.out_vars), f"maxpool1d out_vars length {len(L.out_vars)} != output elements {lb_out.numel()}"

    B = Bounds(lb_out.view(-1), ub_out.view(-1))
    assert torch.all(B.lb <= B.ub), "maxpool1d produced invalid bounds (lb > ub)"
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"maxpool1d:{L.id}",
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "input_shape": input_shape,
        "output_shape": output_shape,
    }))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)


def tf_maxpool2d(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for MaxPool2d layer.
    
    Uses interval arithmetic to bound the max pooling operation.
    """
    # Extract pooling parameters
    kernel_size = L.meta["kernel_size"]
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)
    
    # Shape information
    input_shape = L.meta["input_shape"]  # [batch, channels, height, width]
    output_shape = L.meta["output_shape"]  # [batch, channels, out_h, out_w]
    
    batch_size, channels, in_h, in_w = input_shape
    _, _, out_h, out_w = output_shape
    
    # For max pooling, we need to consider all possible inputs in each pool window
    # The output bounds are the max of upper bounds and max of lower bounds in each window
    
    # Reshape bounds for pooling operation
    input_lb = Bin.lb.view(batch_size, channels, in_h, in_w)
    input_ub = Bin.ub.view(batch_size, channels, in_h, in_w)
    
    # Apply max pooling to bounds
    # For lower bound: take max of lower bounds in each window
    # For upper bound: take max of upper bounds in each window
    output_lb = F.max_pool2d(input_lb, kernel_size, stride, padding, dilation)
    output_ub = F.max_pool2d(input_ub, kernel_size, stride, padding, dilation)
    
    # Flatten output bounds
    B_output = Bounds(output_lb.view(-1), output_ub.view(-1))
    
    # Create constraints for max pooling
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"maxpool2d:{L.id}",
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "input_shape": input_shape,
        "output_shape": output_shape
    }))
    
    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)

def tf_avgpool1d(L: Layer, Bin: Bounds) -> Fact:
    kernel_size = L.meta["kernel_size"]
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)

    input_shape = L.meta["input_shape"]
    output_shape = L.meta["output_shape"]

    b, c, w = input_shape
    lb_in = Bin.lb.view(b, c, w)
    ub_in = Bin.ub.view(b, c, w)

    lb_out = F.avg_pool1d(lb_in, kernel_size, stride, padding)
    ub_out = F.avg_pool1d(ub_in, kernel_size, stride, padding)

    B_output = Bounds(lb_out.view(-1), ub_out.view(-1))
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"avgpool1d:{L.id}",
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input_shape": input_shape,
        "output_shape": output_shape
    }))
    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)

def tf_maxpool3d(L: Layer, Bin: Bounds) -> Fact:
    kernel_size = L.meta["kernel_size"]
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)

    input_shape = L.meta["input_shape"]   # [b, c, d, h, w]
    output_shape = L.meta["output_shape"] # [b, c, od, oh, ow]

    b, c, d, h, w = input_shape
    lb_in = Bin.lb.view(b, c, d, h, w)
    ub_in = Bin.ub.view(b, c, d, h, w)

    lb_out = F.max_pool3d(lb_in, kernel_size, stride, padding, dilation)
    ub_out = F.max_pool3d(ub_in, kernel_size, stride, padding, dilation)
    assert lb_out.shape == tuple(output_shape), f"maxpool3d output shape mismatch: got {tuple(lb_out.shape)}, expected {tuple(output_shape)}"
    assert lb_out.numel() == len(L.out_vars), f"maxpool3d out_vars length {len(L.out_vars)} != output elements {lb_out.numel()}"

    B = Bounds(lb_out.view(-1), ub_out.view(-1))
    assert torch.all(B.lb <= B.ub), "maxpool3d produced invalid bounds (lb > ub)"
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"maxpool3d:{L.id}",
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "input_shape": input_shape,
        "output_shape": output_shape,
    }))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)

def tf_pad(L: Layer, Bin: Bounds) -> Fact:
    pads = L.meta.get("pad", None)
    if pads is None:
        pads = L.meta.get("pads", None)
    if pads is None:
        raise KeyError(f"pad/pads not found in meta for PAD layer {L.id}")
    assert len(pads) % 2 == 0, f"pad expects pairs, got pads={pads}"

    mode = L.meta.get("mode", "constant")
    value = float(L.meta.get("value", 0.0))

    in_shape = tuple(L.meta["input_shape"])
    lb_in = Bin.lb.view(*in_shape)
    ub_in = Bin.ub.view(*in_shape)

    lb_out = F.pad(lb_in, pads, mode=mode, value=value)
    ub_out = F.pad(ub_in, pads, mode=mode, value=value)
    assert lb_out.numel() == len(L.out_vars), f"pad out_vars length {len(L.out_vars)} != output elements {lb_out.numel()}"

    B = Bounds(lb_out.reshape(-1), ub_out.reshape(-1))
    assert torch.all(B.lb <= B.ub), "pad produced invalid bounds (lb > ub)"
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"pad:{L.id}",
        "pads": list(pads),
        "mode": mode,
        "value": value,
    }))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)

def tf_flatten(L: Layer, Bin: Bounds) -> Fact:
    lb = Bin.lb
    ub = Bin.ub

    if "input_shape" in L.meta:
        input_shape = tuple(L.meta["input_shape"])
    else:
        input_shape = (int(lb.numel()),)

    if "output_shape" in L.meta:
        output_shape = tuple(L.meta["output_shape"])
    else:
        output_shape = (int(lb.numel()),)

    axis      = L.meta.get("axis", None)        # ONNX Flatten(axis=...)
    start_dim = L.meta.get("start_dim", None)   # torch.flatten(start_dim, end_dim)
    end_dim   = L.meta.get("end_dim", None)

    lb_flat = lb.view(-1)
    ub_flat = ub.view(-1)
    assert lb_flat.numel() == len(L.out_vars), f"flatten out_vars length {len(L.out_vars)} != output elements {lb_flat.numel()}"
    if "output_shape" in L.meta:
        expected = int(torch.tensor(output_shape).prod().item())
        assert lb_flat.numel() == expected, f"flatten output numel {lb_flat.numel()} != expected {expected}"
    B_out = Bounds(lb_flat, ub_flat)
    assert torch.all(B_out.lb <= B_out.ub), "flatten produced invalid bounds (lb > ub)"

    C = ConSet()
    C.replace(Con(
        "EQ",
        tuple(L.out_vars + L.in_vars),
        {
            "tag":          f"flatten:{L.id}",
            "input_shape":  input_shape,
            "output_shape": output_shape,
            "axis":         axis,
            "start_dim":    start_dim,
            "end_dim":      end_dim,
        },
    ))

    C.add_box(L.id, L.out_vars, B_out)
    return Fact(B_out, C)

def _conv2d_to_linear_matrix(
    weight: torch.Tensor,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """
    Convert Conv2d operation to equivalent linear transformation matrix.
    
    This uses the im2col algorithm to unfold the convolution into matrix multiplication.
    """
    batch_size, in_channels, in_h, in_w = input_shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    _, _, out_h, out_w = output_shape
    
    # Create input and output flat sizes
    input_flat_size = in_channels * in_h * in_w
    output_flat_size = out_channels * out_h * out_w
    
    # Initialize the equivalent weight matrix
    W_equiv = torch.zeros(output_flat_size, input_flat_size, dtype=weight.dtype, device=weight.device)
    
    # Convert stride, padding, dilation to tuples if they're integers
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # For each output position, find corresponding input positions
    for out_c in range(out_channels):
        for out_y in range(out_h):
            for out_x in range(out_w):
                # Calculate output linear index
                out_idx = out_c * (out_h * out_w) + out_y * out_w + out_x
                
                # For each kernel position
                for in_c in range(in_channels):
                    for k_y in range(kernel_h):
                        for k_x in range(kernel_w):
                            # Calculate input position
                            in_y = out_y * stride[0] - padding[0] + k_y * dilation[0]
                            in_x = out_x * stride[1] - padding[1] + k_x * dilation[1]
                            
                            # Check bounds
                            if 0 <= in_y < in_h and 0 <= in_x < in_w:
                                # Calculate input linear index
                                in_idx = in_c * (in_h * in_w) + in_y * in_w + in_x
                                
                                # Set weight in equivalent matrix
                                W_equiv[out_idx, in_idx] = weight[out_c, in_c, k_y, k_x]
    
    return W_equiv


def tf_avgpool2d(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for AvgPool2d layer.
    
    Uses linear transformation to handle average pooling.
    """
    # Extract pooling parameters
    kernel_size = L.meta["kernel_size"]
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)
    
    # Input/output shape information
    input_shape = L.meta["input_shape"]
    output_shape = L.meta["output_shape"]
    
    batch_size, channels, in_h, in_w = input_shape
    _, _, out_h, out_w = output_shape
    
    # Create equivalent linear transformation for average pooling
    input_flat_size = channels * in_h * in_w
    output_flat_size = channels * out_h * out_w
    
    W_equiv = _avgpool2d_to_linear_matrix(
        input_shape, output_shape, kernel_size, stride, padding
    )
    
    # No bias for average pooling
    b_equiv = torch.zeros(output_flat_size, dtype=Bin.lb.dtype, device=Bin.lb.device)
    
    # Apply linear transformation
    W_pos = torch.clamp(W_equiv, min=0)
    W_neg = torch.clamp(W_equiv, max=0)
    
    input_bounds_flat = Bounds(Bin.lb.view(-1), Bin.ub.view(-1))
    B_output = affine_bounds(W_pos, W_neg, b_equiv, input_bounds_flat)
    
    # Create constraints
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"avgpool2d:{L.id}",
        "W": W_equiv,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input_shape": input_shape,
        "output_shape": output_shape
    }))
    
    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)


def _avgpool2d_to_linear_matrix(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    kernel_size: int,
    stride: int,
    padding: int
) -> torch.Tensor:
    """Convert AvgPool2d to equivalent linear transformation matrix."""
    batch_size, channels, in_h, in_w = input_shape
    _, _, out_h, out_w = output_shape
    
    input_flat_size = channels * in_h * in_w
    output_flat_size = channels * out_h * out_w
    
    W_equiv = torch.zeros(output_flat_size, input_flat_size)
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    kernel_h, kernel_w = kernel_size
    
    for c in range(channels):
        for out_y in range(out_h):
            for out_x in range(out_w):
                out_idx = c * (out_h * out_w) + out_y * out_w + out_x
                
                # Count valid kernel positions
                valid_count = 0
                
                for k_y in range(kernel_h):
                    for k_x in range(kernel_w):
                        in_y = out_y * stride[0] - padding[0] + k_y
                        in_x = out_x * stride[1] - padding[1] + k_x
                        
                        if 0 <= in_y < in_h and 0 <= in_x < in_w:
                            in_idx = c * (in_h * in_w) + in_y * in_w + in_x
                            valid_count += 1
                
                # Set weights for average (1/count for each valid position)
                if valid_count > 0:
                    weight_val = 1.0 / valid_count
                    
                    for k_y in range(kernel_h):
                        for k_x in range(kernel_w):
                            in_y = out_y * stride[0] - padding[0] + k_y
                            in_x = out_x * stride[1] - padding[1] + k_x
                            
                            if 0 <= in_y < in_h and 0 <= in_x < in_w:
                                in_idx = c * (in_h * in_w) + in_y * in_w + in_x
                                W_equiv[out_idx, in_idx] = weight_val
    
    return W_equiv


# -------- Additional CNN Layers --------

def tf_conv1d(L: Layer, Bin: Bounds) -> Fact:
    """Transfer function for Conv1d layer."""
    # Extract convolution parameters
    weight = L.params["weight"]  # [out_channels, in_channels, kernel_w]
    bias = L.params.get("bias", None)
    stride = L.meta.get("stride", 1)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)
    groups = L.meta.get("groups", 1)
    
    # Input/output shape information
    input_shape = L.meta["input_shape"]   # [batch, channels, width]
    output_shape = L.meta["output_shape"] # [batch, out_channels, out_w]
    
    # Convert to equivalent linear transformation matrix
    W_equiv = _conv1d_to_linear_matrix(
        weight, input_shape, output_shape, stride, padding, dilation, groups
    )
    
    # Apply affine transformation with bias
    if bias is not None:
        b_equiv = bias.repeat(output_shape[-1])  # Repeat for spatial dimensions
    else:
        b_equiv = torch.zeros(W_equiv.shape[0], device=weight.device, dtype=weight.dtype)
    
    # Compute bounds using affine transformation
    W_pos = torch.clamp(W_equiv, min=0)
    W_neg = torch.clamp(W_equiv, max=0)
    
    # Apply linear transformation
    B_output = affine_bounds(W_pos, W_neg, b_equiv, Bin)
    
    # Create constraints
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"conv1d:{L.id}",
        "W": W_equiv,
        "b": b_equiv,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "conv_params": {
            "stride": stride, "padding": padding, "dilation": dilation, "groups": groups
        }
    }))
    
    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)


def tf_conv3d(L: Layer, Bin: Bounds) -> Fact:
    """Transfer function for Conv3d layer."""
    # Extract convolution parameters
    weight = L.params["weight"]  # [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
    bias = L.params.get("bias", None)
    stride = L.meta.get("stride", 1)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)
    groups = L.meta.get("groups", 1)
    
    # Input/output shape information
    input_shape = L.meta["input_shape"]   # [batch, channels, depth, height, width]
    output_shape = L.meta["output_shape"] # [batch, out_channels, out_d, out_h, out_w]
    
    # Convert to equivalent linear transformation matrix
    W_equiv = _conv3d_to_linear_matrix(
        weight, input_shape, output_shape, stride, padding, dilation, groups
    )
    
    # Apply affine transformation with bias
    if bias is not None:
        out_d, out_h, out_w = output_shape[-3:]
        b_equiv = bias.repeat(out_d * out_h * out_w)
    else:
        b_equiv = torch.zeros(W_equiv.shape[0], device=weight.device, dtype=weight.dtype)
    
    # Compute bounds using affine transformation
    W_pos = torch.clamp(W_equiv, min=0)
    W_neg = torch.clamp(W_equiv, max=0)
    
    B_output = affine_bounds(W_pos, W_neg, b_equiv, Bin)
    
    # Create constraints
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"conv3d:{L.id}",
        "W": W_equiv,
        "b": b_equiv,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "conv_params": {
            "stride": stride, "padding": padding, "dilation": dilation, "groups": groups
        }
    }))
    
    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)


def tf_convtranspose2d(L: Layer, Bin: Bounds) -> Fact:
    """Transfer function for ConvTranspose2d layer."""
    # Extract parameters
    weight = L.params["weight"]  # [in_channels, out_channels, kernel_h, kernel_w]
    bias = L.params.get("bias", None)
    stride = L.meta.get("stride", 1)
    padding = L.meta.get("padding", 0)
    output_padding = L.meta.get("output_padding", 0)
    dilation = L.meta.get("dilation", 1)
    groups = L.meta.get("groups", 1)
    
    # Input/output shape information
    input_shape = L.meta["input_shape"]
    output_shape = L.meta["output_shape"]
    
    # Convert to equivalent linear transformation matrix
    W_equiv = _convtranspose2d_to_linear_matrix(
        weight, input_shape, output_shape, stride, padding, output_padding, dilation, groups
    )
    
    # Apply affine transformation with bias
    if bias is not None:
        out_h, out_w = output_shape[-2:]
        b_equiv = bias.repeat(out_h * out_w)
    else:
        b_equiv = torch.zeros(W_equiv.shape[0], device=weight.device, dtype=weight.dtype)
    
    # Compute bounds
    W_pos = torch.clamp(W_equiv, min=0)
    W_neg = torch.clamp(W_equiv, max=0)
    
    B_output = affine_bounds(W_pos, W_neg, b_equiv, Bin)
    
    # Create constraints
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"convtranspose2d:{L.id}",
        "W": W_equiv,
        "b": b_equiv,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "conv_params": {
            "stride": stride, "padding": padding, "output_padding": output_padding,
            "dilation": dilation, "groups": groups
        }
    }))
    
    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)

def tf_upsample(L: Layer, Bin: Bounds) -> Fact:
    in_shape = tuple(L.meta["input_shape"])
    x_lb = Bin.lb.view(*in_shape)
    x_ub = Bin.ub.view(*in_shape)

    size = L.meta.get("size", None)
    scale_factor = L.meta.get("scale_factor", None)
    mode = L.meta.get("mode", "nearest")
    align_corners = bool(L.meta.get("align_corners", False))
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

    if "output_shape" in L.meta:
        expected_shape = tuple(L.meta["output_shape"])
        assert tuple(y_lb.shape) == expected_shape, f"upsample output shape mismatch: got {tuple(y_lb.shape)}, expected {expected_shape}"
    assert y_lb.numel() == len(L.out_vars), f"upsample out_vars length {len(L.out_vars)} != output elements {y_lb.numel()}"

    B = Bounds(y_lb.reshape(-1), y_ub.reshape(-1))
    assert torch.all(B.lb <= B.ub), "upsample produced invalid bounds (lb > ub)"
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"upsample:{L.id}",
        "mode": mode,
        "size": list(size) if size is not None else None,
        "scale_factor": scale_factor,
        "input_shape": in_shape,
        "output_shape": list(y_lb.shape),
    }))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)


# -------- Helper functions for new conv layers --------

def _conv1d_to_linear_matrix(
    weight: torch.Tensor,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """Convert Conv1d to equivalent linear transformation matrix."""
    batch_size, in_channels, in_w = input_shape
    _, out_channels, out_w = output_shape
    
    input_flat_size = in_channels * in_w
    output_flat_size = out_channels * out_w
    
    W_equiv = torch.zeros(output_flat_size, input_flat_size, device=weight.device, dtype=weight.dtype)
    
    kernel_w = weight.shape[2]
    
    for out_c in range(out_channels):
        for out_x in range(out_w):
            for in_c in range(in_channels // groups):
                for k_x in range(kernel_w):
                    in_x = out_x * stride - padding + k_x * dilation
                    
                    if 0 <= in_x < in_w:
                        group_idx = (out_c // (out_channels // groups))
                        actual_in_c = group_idx * (in_channels // groups) + in_c
                        
                        out_idx = out_c * out_w + out_x
                        in_idx = actual_in_c * in_w + in_x
                        
                        W_equiv[out_idx, in_idx] += weight[out_c, in_c, k_x]
    
    return W_equiv


def _conv3d_to_linear_matrix(
    weight: torch.Tensor,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """Convert Conv3d to equivalent linear transformation matrix."""
    batch_size, in_channels, in_d, in_h, in_w = input_shape
    _, out_channels, out_d, out_h, out_w = output_shape
    
    input_flat_size = in_channels * in_d * in_h * in_w
    output_flat_size = out_channels * out_d * out_h * out_w
    
    W_equiv = torch.zeros(output_flat_size, input_flat_size, device=weight.device, dtype=weight.dtype)
    
    kernel_d, kernel_h, kernel_w = weight.shape[2], weight.shape[3], weight.shape[4]
    
    # Handle stride/padding as tuples or ints
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    
    for out_c in range(out_channels):
        for out_d_idx in range(out_d):
            for out_h_idx in range(out_h):
                for out_w_idx in range(out_w):
                    for in_c in range(in_channels // groups):
                        for k_d in range(kernel_d):
                            for k_h in range(kernel_h):
                                for k_w in range(kernel_w):
                                    in_d_idx = out_d_idx * stride[0] - padding[0] + k_d * dilation[0]
                                    in_h_idx = out_h_idx * stride[1] - padding[1] + k_h * dilation[1]
                                    in_w_idx = out_w_idx * stride[2] - padding[2] + k_w * dilation[2]
                                    
                                    if (0 <= in_d_idx < in_d and 0 <= in_h_idx < in_h and 0 <= in_w_idx < in_w):
                                        group_idx = (out_c // (out_channels // groups))
                                        actual_in_c = group_idx * (in_channels // groups) + in_c
                                        
                                        out_idx = (out_c * out_d * out_h * out_w + 
                                                 out_d_idx * out_h * out_w +
                                                 out_h_idx * out_w + out_w_idx)
                                        in_idx = (actual_in_c * in_d * in_h * in_w +
                                                in_d_idx * in_h * in_w +
                                                in_h_idx * in_w + in_w_idx)
                                        
                                        W_equiv[out_idx, in_idx] += weight[out_c, in_c, k_d, k_h, k_w]
    
    return W_equiv


def _convtranspose2d_to_linear_matrix(
    weight: torch.Tensor,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """Convert ConvTranspose2d to equivalent linear transformation matrix."""
    batch_size, in_channels, in_h, in_w = input_shape
    _, out_channels, out_h, out_w = output_shape
    
    input_flat_size = in_channels * in_h * in_w
    output_flat_size = out_channels * out_h * out_w
    
    W_equiv = torch.zeros(output_flat_size, input_flat_size, device=weight.device, dtype=weight.dtype)
    
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
    
    # Transpose convolution: each input position contributes to multiple output positions
    for in_c in range(in_channels):
        for in_y in range(in_h):
            for in_x in range(in_w):
                for out_c in range(out_channels // groups):
                    for k_y in range(kernel_h):
                        for k_w in range(kernel_w):
                            out_y = in_y * stride[0] - padding[0] + k_y * dilation[0]
                            out_x = in_x * stride[1] - padding[1] + k_w * dilation[1]
                            
                            if (0 <= out_y < out_h and 0 <= out_x < out_w):
                                group_idx = (in_c // (in_channels // groups))
                                actual_out_c = group_idx * (out_channels // groups) + out_c
                                
                                in_idx = in_c * in_h * in_w + in_y * in_w + in_x
                                out_idx = actual_out_c * out_h * out_w + out_y * out_w + out_x
                                
                                W_equiv[out_idx, in_idx] += weight[in_c, out_c, k_y, k_w]
    
    return W_equiv
