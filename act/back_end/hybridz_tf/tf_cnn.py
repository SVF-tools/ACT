#===- act/back_end/hybridz_tf/tf_cnn.py - HybridZ CNN Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ CNN Transfer Functions. Implements HybridZ-based transfer functions
#   for CNN layers including convolution, pooling, and tensor reshaping
#   operations.
#
#===---------------------------------------------------------------------===#


import torch
import torch.nn.functional as F
from typing import List, Tuple
from act.back_end.core import Bounds, Fact, Layer, ConSet


@torch.no_grad()
def hybridz_tf_conv2d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 2D convolution with enhanced precision."""
    # Extract convolution parameters
    weight = L.params["weight"]  # (out_channels, in_channels, kernel_h, kernel_w)
    bias = L.params.get("bias", None)
    stride = L.meta.get("stride", 1)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)
    groups = L.meta.get("groups", 1)
    
    # Input shape: (batch, in_channels, height, width) - for bounds propagation batch=1
    input_shape = L.meta.get("input_shape", None)  # (channels, height, width)
    if Bin.lb.dim() == 1:
        # Flatten input needs to be reshaped
        if input_shape is None:
            raise ValueError("CONV2D got flat bounds but meta.input_shape is missing")

        # input_shape may be (N,C,H,W) or (C,H,W)
        if len(input_shape) == 4:
            _, C, H, W = input_shape
        elif len(input_shape) == 3:
            C, H, W = input_shape
        else:
            raise ValueError(f"Unexpected input_shape={input_shape}")
        Bin_reshaped_lb = Bin.lb.view(1, C, H, W)
        Bin_reshaped_ub = Bin.ub.view(1, C, H, W)
    elif Bin.lb.dim() == 3:
        Bin_reshaped_lb = Bin.lb.unsqueeze(0)
        Bin_reshaped_ub = Bin.ub.unsqueeze(0)
    else:
        Bin_reshaped_lb = Bin.lb
        Bin_reshaped_ub = Bin.ub
    
    # Apply convolution to bounds
    # For HybridZ: more precise bound computation considering kernel structure
    weight_pos = torch.clamp(weight, min=0)
    weight_neg = torch.clamp(weight, max=0)
    
    # Lower bound: positive weights * lower bounds + negative weights * upper bounds
    lb_conv = F.conv2d(Bin_reshaped_lb, weight_pos, bias=None, stride=stride, 
                       padding=padding, dilation=dilation, groups=groups)
    lb_conv += F.conv2d(Bin_reshaped_ub, weight_neg, bias=None, stride=stride,
                        padding=padding, dilation=dilation, groups=groups)
    
    # Upper bound: positive weights * upper bounds + negative weights * lower bounds  
    ub_conv = F.conv2d(Bin_reshaped_ub, weight_pos, bias=None, stride=stride,
                       padding=padding, dilation=dilation, groups=groups)
    ub_conv += F.conv2d(Bin_reshaped_lb, weight_neg, bias=None, stride=stride,
                        padding=padding, dilation=dilation, groups=groups)
    
    if bias is not None:
        lb_conv += bias.view(1, -1, 1, 1)
        ub_conv += bias.view(1, -1, 1, 1)
    
    # Flatten output if needed
    lb = lb_conv.reshape(-1)
    ub = ub_conv.reshape(-1)
    assert lb.numel() == len(L.out_vars)
    
    Bout = Bounds(lb=lb, ub=ub)
    
    # Generate convolution constraints
    cons = ConSet()
    cons.add_op( f"conv2d:{L.id}", list(L.out_vars + L.in_vars), weight=weight, 
                bias=bias if bias is not None else torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype),
                stride=stride, padding=padding, dilation=dilation, groups=groups, input_shape=L.meta.get("input_shape"), output_shape=L.meta.get("output_shape"),)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_maxpool2d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 2D max pooling."""
    kernel_size = L.meta.get("kernel_size", 2)
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)
    
    # Reshape input if flattened
    in_shape = L.meta.get("input_shape")  # May be (N,C,H,W) or (C,H,W)
    if len(Bin.lb.shape) == 1 and in_shape:
        # input_shape may be (N,C,H,W) or (C,H,W)
        if len(in_shape) == 4:
            _, C, H, W = in_shape
        elif len(in_shape) == 3:
            C, H, W = in_shape
        else:
            raise ValueError(f"Unexpected input_shape={in_shape}")
        Bin_lb = Bin.lb.view(1, C, H, W)
        Bin_ub = Bin.ub.view(1, C, H, W)
    else:
        Bin_lb = Bin.lb.unsqueeze(0) if len(Bin.lb.shape) == 3 else Bin.lb
        Bin_ub = Bin.ub.unsqueeze(0) if len(Bin.ub.shape) == 3 else Bin.ub
    
    # Max pooling: upper bounds of pooling regions
    # For HybridZ: track which neurons contribute to maximum
    lb_pool = F.max_pool2d(Bin_lb, kernel_size, stride=stride, padding=padding)
    ub_pool = F.max_pool2d(Bin_ub, kernel_size, stride=stride, padding=padding)
    
    # For max pooling, lower bound is more complex - use max of lower bounds in each region
    # This is conservative but sound
    lb = lb_pool.squeeze(0).flatten() if len(L.out_vars) != lb_pool.numel() else lb_pool.squeeze(0)
    ub = ub_pool.squeeze(0).flatten() if len(L.out_vars) != ub_pool.numel() else ub_pool.squeeze(0)

    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    # Max pooling generates max constraints
    cons.add_op( f"maxpool2d:{L.id}", list(L.out_vars + L.in_vars), kernel_size=kernel_size, 
        stride=stride, padding=padding, input_shape=in_shape, output_shape=L.meta.get("output_shape"),)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_avgpool2d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 2D average pooling."""
    kernel_size = L.meta.get("kernel_size", 2)
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)
    
    # Reshape input if needed
    in_shape = L.meta.get("input_shape")  # (N,C,H,W) or (C,H,W)
    if len(Bin.lb.shape) == 1 and in_shape:
        # input_shape may be (N,C,H,W) or (C,H,W)
        if len(in_shape) == 4:
            _, C, H, W = in_shape
        elif len(in_shape) == 3:
            C, H, W = in_shape
        else:
            raise ValueError(f"Unexpected input_shape={in_shape}")
        Bin_lb = Bin.lb.view(1, C, H, W)
        Bin_ub = Bin.ub.view(1, C, H, W)
    else:
        Bin_lb = Bin.lb.unsqueeze(0) if len(Bin.lb.shape) == 3 else Bin.lb
        Bin_ub = Bin.ub.unsqueeze(0) if len(Bin.ub.shape) == 3 else Bin.ub
    
    # Average pooling is linear - exact bounds
    lb_pool = F.avg_pool2d(Bin_lb, kernel_size, stride=stride, padding=padding)
    ub_pool = F.avg_pool2d(Bin_ub, kernel_size, stride=stride, padding=padding)
    
    lb = lb_pool.squeeze(0).flatten() if len(L.out_vars) != lb_pool.numel() else lb_pool.squeeze(0)
    ub = ub_pool.squeeze(0).flatten() if len(L.out_vars) != ub_pool.numel() else ub_pool.squeeze(0)
    
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_op(
        f"avgpool2d:{L.id}", list(L.out_vars + L.in_vars), kernel_size=kernel_size, stride=stride,
        padding=padding, input_shape=in_shape, output_shape=L.meta.get("output_shape"),)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_flatten(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for tensor flattening."""
    # Flattening is just reshaping - bounds remain the same
    start_dim = L.meta.get("start_dim", 1)
    end_dim = L.meta.get("end_dim", -1)
    
    # Simple reshape - no change in bounds
    lb = Bin.lb.flatten()
    ub = Bin.ub.flatten()
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_op(f"flatten:{L.id}", list(L.out_vars + L.in_vars), start_dim=start_dim, end_dim=end_dim, input_shape=L.meta.get("input_shape"), output_shape=L.meta.get("output_shape"))
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_reshape(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for general tensor reshaping."""
    target_shape = L.meta.get("target_shape")

    # Reshape bounds preserving values
    lb = Bin.lb.reshape(target_shape) if target_shape else Bin.lb
    ub = Bin.ub.reshape(target_shape) if target_shape else Bin.ub

    # Flatten for output variables
    lb = lb.flatten()
    ub = ub.flatten()
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"reshape:{L.id}", list(L.out_vars + L.in_vars), target_shape=target_shape, input_shape=L.meta.get("input_shape"), output_shape=L.meta.get("output_shape"))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_conv1d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 1D convolution."""
    weight = L.params["weight"]  # (out_channels, in_channels, kernel_w)
    bias = L.params.get("bias", None)
    stride = L.meta.get("stride", 1)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)
    groups = L.meta.get("groups", 1)

    input_shape = L.meta.get("input_shape")
    if Bin.lb.dim() == 1 and input_shape:
        if len(input_shape) == 3:
            N, C, W = input_shape
        elif len(input_shape) == 2:
            C, W = input_shape
            N = 1
        else:
            raise ValueError(f"Unexpected input_shape={input_shape}")
        Bin_lb = Bin.lb.view(1, C, W)
        Bin_ub = Bin.ub.view(1, C, W)
    else:
        Bin_lb = Bin.lb.unsqueeze(0) if Bin.lb.dim() == 2 else Bin.lb
        Bin_ub = Bin.ub.unsqueeze(0) if Bin.ub.dim() == 2 else Bin.ub

    weight_pos = torch.clamp(weight, min=0)
    weight_neg = torch.clamp(weight, max=0)

    lb_conv = F.conv1d(Bin_lb, weight_pos, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
    lb_conv += F.conv1d(Bin_ub, weight_neg, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)

    ub_conv = F.conv1d(Bin_ub, weight_pos, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
    ub_conv += F.conv1d(Bin_lb, weight_neg, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)

    if bias is not None:
        lb_conv += bias.view(1, -1, 1)
        ub_conv += bias.view(1, -1, 1)

    lb = lb_conv.reshape(-1)
    ub = ub_conv.reshape(-1)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"conv1d:{L.id}", list(L.out_vars + L.in_vars), weight=weight,
                bias=bias if bias is not None else torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype),
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                input_shape=input_shape, output_shape=L.meta.get("output_shape"))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_conv3d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 3D convolution."""
    weight = L.params["weight"]  # (out_channels, in_channels, kernel_d, kernel_h, kernel_w)
    bias = L.params.get("bias", None)
    stride = L.meta.get("stride", 1)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)
    groups = L.meta.get("groups", 1)

    input_shape = L.meta.get("input_shape")
    if Bin.lb.dim() == 1 and input_shape:
        if len(input_shape) == 5:
            N, C, D, H, W = input_shape
        elif len(input_shape) == 4:
            C, D, H, W = input_shape
            N = 1
        else:
            raise ValueError(f"Unexpected input_shape={input_shape}")
        Bin_lb = Bin.lb.view(1, C, D, H, W)
        Bin_ub = Bin.ub.view(1, C, D, H, W)
    else:
        Bin_lb = Bin.lb.unsqueeze(0) if Bin.lb.dim() == 4 else Bin.lb
        Bin_ub = Bin.ub.unsqueeze(0) if Bin.ub.dim() == 4 else Bin.ub

    weight_pos = torch.clamp(weight, min=0)
    weight_neg = torch.clamp(weight, max=0)

    lb_conv = F.conv3d(Bin_lb, weight_pos, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
    lb_conv += F.conv3d(Bin_ub, weight_neg, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)

    ub_conv = F.conv3d(Bin_ub, weight_pos, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
    ub_conv += F.conv3d(Bin_lb, weight_neg, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)

    if bias is not None:
        lb_conv += bias.view(1, -1, 1, 1, 1)
        ub_conv += bias.view(1, -1, 1, 1, 1)

    lb = lb_conv.reshape(-1)
    ub = ub_conv.reshape(-1)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"conv3d:{L.id}", list(L.out_vars + L.in_vars), weight=weight,
                bias=bias if bias is not None else torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype),
                stride=stride, padding=padding, dilation=dilation, groups=groups,
                input_shape=input_shape, output_shape=L.meta.get("output_shape"))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_maxpool1d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 1D max pooling."""
    kernel_size = L.meta.get("kernel_size", 2)
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)

    in_shape = L.meta.get("input_shape")
    if len(Bin.lb.shape) == 1 and in_shape:
        if len(in_shape) == 3:
            N, C, W = in_shape
        elif len(in_shape) == 2:
            C, W = in_shape
            N = 1
        else:
            raise ValueError(f"Unexpected input_shape={in_shape}")
        Bin_lb = Bin.lb.view(1, C, W)
        Bin_ub = Bin.ub.view(1, C, W)
    else:
        Bin_lb = Bin.lb.unsqueeze(0) if len(Bin.lb.shape) == 2 else Bin.lb
        Bin_ub = Bin.ub.unsqueeze(0) if len(Bin.ub.shape) == 2 else Bin.ub

    lb_pool = F.max_pool1d(Bin_lb, kernel_size, stride=stride, padding=padding, dilation=dilation)
    ub_pool = F.max_pool1d(Bin_ub, kernel_size, stride=stride, padding=padding, dilation=dilation)

    lb = lb_pool.reshape(-1)
    ub = ub_pool.reshape(-1)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"maxpool1d:{L.id}", list(L.out_vars + L.in_vars), kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                input_shape=in_shape, output_shape=L.meta.get("output_shape"))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_maxpool3d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 3D max pooling."""
    kernel_size = L.meta.get("kernel_size", 2)
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)

    in_shape = L.meta.get("input_shape")
    if len(Bin.lb.shape) == 1 and in_shape:
        if len(in_shape) == 5:
            N, C, D, H, W = in_shape
        elif len(in_shape) == 4:
            C, D, H, W = in_shape
            N = 1
        else:
            raise ValueError(f"Unexpected input_shape={in_shape}")
        Bin_lb = Bin.lb.view(1, C, D, H, W)
        Bin_ub = Bin.ub.view(1, C, D, H, W)
    else:
        Bin_lb = Bin.lb.unsqueeze(0) if len(Bin.lb.shape) == 4 else Bin.lb
        Bin_ub = Bin.ub.unsqueeze(0) if len(Bin.ub.shape) == 4 else Bin.ub

    lb_pool = F.max_pool3d(Bin_lb, kernel_size, stride=stride, padding=padding, dilation=dilation)
    ub_pool = F.max_pool3d(Bin_ub, kernel_size, stride=stride, padding=padding, dilation=dilation)

    lb = lb_pool.reshape(-1)
    ub = ub_pool.reshape(-1)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"maxpool3d:{L.id}", list(L.out_vars + L.in_vars), kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                input_shape=in_shape, output_shape=L.meta.get("output_shape"))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_avgpool1d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 1D average pooling."""
    kernel_size = L.meta.get("kernel_size", 2)
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)

    in_shape = L.meta.get("input_shape")
    if len(Bin.lb.shape) == 1 and in_shape:
        if len(in_shape) == 3:
            N, C, W = in_shape
        elif len(in_shape) == 2:
            C, W = in_shape
            N = 1
        else:
            raise ValueError(f"Unexpected input_shape={in_shape}")
        Bin_lb = Bin.lb.view(1, C, W)
        Bin_ub = Bin.ub.view(1, C, W)
    else:
        Bin_lb = Bin.lb.unsqueeze(0) if len(Bin.lb.shape) == 2 else Bin.lb
        Bin_ub = Bin.ub.unsqueeze(0) if len(Bin.ub.shape) == 2 else Bin.ub

    lb_pool = F.avg_pool1d(Bin_lb, kernel_size, stride=stride, padding=padding)
    ub_pool = F.avg_pool1d(Bin_ub, kernel_size, stride=stride, padding=padding)

    lb = lb_pool.reshape(-1)
    ub = ub_pool.reshape(-1)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"avgpool1d:{L.id}", list(L.out_vars + L.in_vars), kernel_size=kernel_size,
                stride=stride, padding=padding,
                input_shape=in_shape, output_shape=L.meta.get("output_shape"))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_pad(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for padding."""
    pads = L.meta.get("pad", L.meta.get("pads", None))
    if pads is None:
        raise KeyError(f"pad/pads not found in meta for PAD layer {L.id}")

    mode = L.meta.get("mode", "constant")
    value = float(L.meta.get("value", 0.0))

    in_shape = tuple(L.meta["input_shape"])
    lb_in = Bin.lb.view(*in_shape)
    ub_in = Bin.ub.view(*in_shape)

    lb_out = F.pad(lb_in, pads, mode=mode, value=value)
    ub_out = F.pad(ub_in, pads, mode=mode, value=value)

    lb = lb_out.reshape(-1)
    ub = ub_out.reshape(-1)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"pad:{L.id}", list(L.out_vars + L.in_vars), pads=list(pads), mode=mode, value=value)

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_upsample(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for upsampling."""
    in_shape = tuple(L.meta["input_shape"])
    x_lb = Bin.lb.view(*in_shape)
    x_ub = Bin.ub.view(*in_shape)

    size = L.meta.get("size", None)
    scale_factor = L.meta.get("scale_factor", None)
    mode = L.meta.get("mode", "nearest")
    align_corners = bool(L.meta.get("align_corners", False))

    y_lb = F.interpolate(x_lb, size=size, scale_factor=scale_factor, mode=mode,
                         align_corners=align_corners if "linear" in mode else None)
    y_ub = F.interpolate(x_ub, size=size, scale_factor=scale_factor, mode=mode,
                         align_corners=align_corners if "linear" in mode else None)

    lb = y_lb.reshape(-1)
    ub = y_ub.reshape(-1)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"upsample:{L.id}", list(L.out_vars + L.in_vars), mode=mode,
                size=list(size) if size is not None else None, scale_factor=scale_factor,
                input_shape=in_shape, output_shape=list(y_lb.shape))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_transpose(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for transpose (identity for bounds)."""
    lb = Bin.lb.clone()
    ub = Bin.ub.clone()
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"transpose:{L.id}", list(L.out_vars + L.in_vars), perm=L.meta.get("perm"))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_squeeze(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for squeeze (identity for bounds)."""
    lb = Bin.lb.clone()
    ub = Bin.ub.clone()
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"squeeze:{L.id}", list(L.out_vars + L.in_vars), dims=L.meta.get("dims"))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_unsqueeze(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for unsqueeze (identity for bounds)."""
    lb = Bin.lb.clone()
    ub = Bin.ub.clone()
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"unsqueeze:{L.id}", list(L.out_vars + L.in_vars), dims=L.meta.get("dims"))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_slice(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for slice operation."""
    inp_shape = tuple(L.meta["input_shape"])
    x_lb = Bin.lb.view(*inp_shape)
    x_ub = Bin.ub.view(*inp_shape)

    starts = L.meta.get("starts", [])
    ends = L.meta.get("ends", [])
    axes = L.meta.get("axes", list(range(len(inp_shape))))
    steps = L.meta.get("steps", [1] * len(axes))

    # Build slice objects for each dimension
    slices = [slice(None)] * len(inp_shape)
    for i, axis in enumerate(axes):
        s = starts[i]
        e = ends[i]
        st = steps[i]
        if e > inp_shape[axis]:
            e = inp_shape[axis]
        slices[axis] = slice(s, e, st)

    out_lb = x_lb[tuple(slices)]
    out_ub = x_ub[tuple(slices)]

    lb = out_lb.reshape(-1)
    ub = out_ub.reshape(-1)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"slice:{L.id}", list(L.out_vars + L.in_vars), starts=starts, ends=ends, axes=axes, steps=steps, input_shape=inp_shape)

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_gather(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for gather operation."""
    inp_shape = tuple(L.meta["input_shape"])
    axis = int(L.meta.get("axis", 0))
    x_lb = Bin.lb.view(*inp_shape)
    x_ub = Bin.ub.view(*inp_shape)

    raw_idx = L.meta["indices"]
    if isinstance(raw_idx, (list, tuple)):
        indices = torch.tensor(raw_idx, dtype=torch.long, device=x_lb.device)
    else:
        indices = raw_idx.to(x_lb.device).long()

    out_lb = torch.index_select(x_lb, dim=axis, index=indices)
    out_ub = torch.index_select(x_ub, dim=axis, index=indices)

    lb = out_lb.reshape(-1)
    ub = out_ub.reshape(-1)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"gather:{L.id}", list(L.out_vars + L.in_vars), axis=axis,
                indices=indices.detach().cpu().tolist(), input_shape=inp_shape, output_shape=list(out_lb.shape))

    return Fact(bounds=Bout, cons=cons)