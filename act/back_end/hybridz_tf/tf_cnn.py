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


def _assert_shape_match(L: Layer, Bin: Bounds, expected_ndim: int) -> Tuple:
    """Assert input bounds match metadata shape and return validated shape tuple.

    STRICT MODE: No fallback behavior. All shapes must be in standard format:
      - 1D: (N, C, W) where N=batch=1
      - 2D: (N, C, H, W) where N=batch=1
      - 3D: (N, C, D, H, W) where N=batch=1

    Args:
        L: Layer with input_shape metadata
        Bin: Input bounds (must be 1D flattened)
        expected_ndim: Expected dimensionality (3=1D, 4=2D, 5=3D)

    Returns:
        Validated input_shape as tuple

    Raises:
        AssertionError: If shape validation fails
    """
    # Bounds must be 1D (flattened)
    assert Bin.lb.dim() == 1, (
        f"Layer {L.id} ({L.kind}): bounds must be 1D (flattened), got {Bin.lb.shape}"
    )

    # input_shape must exist - NO FALLBACK
    assert "input_shape" in L.meta, (
        f"Layer {L.id} ({L.kind}): missing 'input_shape' in metadata. "
        f"CNN layers require input_shape for strict shape validation."
    )

    input_shape = tuple(L.meta["input_shape"])

    # Validate shape has correct dimensionality - NO FALLBACK for (C,H,W) format
    assert len(input_shape) == expected_ndim, (
        f"Layer {L.id} ({L.kind}): expected {expected_ndim}D input_shape, "
        f"got {len(input_shape)}D shape {input_shape}. "
        f"Shapes must be in standard format (N,C,...) with N=batch dimension."
    )

    # Validate bounds numel matches shape
    expected_numel = 1
    for d in input_shape:
        expected_numel *= int(d)
    actual_numel = Bin.lb.numel()
    assert actual_numel == expected_numel, (
        f"Layer {L.id} ({L.kind}): bounds size mismatch. "
        f"Expected {expected_numel} from input_shape={input_shape}, got {actual_numel}."
    )

    return input_shape


@torch.no_grad()
def hybridz_tf_conv2d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 2D convolution with strict shape validation."""
    # STRICT MODE: Assert input_shape metadata exists and matches bounds
    input_shape = _assert_shape_match(L, Bin, expected_ndim=4)
    N, C, H, W = input_shape

    # Extract convolution parameters
    weight = L.params["weight"]  # (out_channels, in_channels, kernel_h, kernel_w)
    bias = L.params.get("bias", None)
    stride = L.meta.get("stride", 1)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)
    groups = L.meta.get("groups", 1)

    # Reshape using validated shape (no guessing)
    Bin_reshaped_lb = Bin.lb.view(N, C, H, W)
    Bin_reshaped_ub = Bin.ub.view(N, C, H, W)
    
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
    """HybridZ transfer function for 2D max pooling with strict shape validation."""
    # STRICT MODE: Assert input_shape metadata exists and matches bounds
    input_shape = _assert_shape_match(L, Bin, expected_ndim=4)
    N, C, H, W = input_shape

    kernel_size = L.meta.get("kernel_size", 2)
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)

    # Reshape using validated shape (no guessing)
    Bin_lb = Bin.lb.view(N, C, H, W)
    Bin_ub = Bin.ub.view(N, C, H, W)
    
    # Max pooling: upper bounds of pooling regions
    # For HybridZ: track which neurons contribute to maximum
    lb_pool = F.max_pool2d(Bin_lb, kernel_size, stride=stride, padding=padding)
    ub_pool = F.max_pool2d(Bin_ub, kernel_size, stride=stride, padding=padding)

    # Flatten to 1D (bounds must always be 1D)
    lb = lb_pool.reshape(-1)
    ub = ub_pool.reshape(-1)

    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    # Max pooling generates max constraints
    cons.add_op( f"maxpool2d:{L.id}", list(L.out_vars + L.in_vars), kernel_size=kernel_size,
        stride=stride, padding=padding, input_shape=input_shape, output_shape=L.meta.get("output_shape"),)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_avgpool2d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 2D average pooling with strict shape validation."""
    # STRICT MODE: Assert input_shape metadata exists and matches bounds
    input_shape = _assert_shape_match(L, Bin, expected_ndim=4)
    N, C, H, W = input_shape

    kernel_size = L.meta.get("kernel_size", 2)
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)

    # Reshape using validated shape (no guessing)
    Bin_lb = Bin.lb.view(N, C, H, W)
    Bin_ub = Bin.ub.view(N, C, H, W)
    
    # Average pooling is linear - exact bounds
    lb_pool = F.avg_pool2d(Bin_lb, kernel_size, stride=stride, padding=padding)
    ub_pool = F.avg_pool2d(Bin_ub, kernel_size, stride=stride, padding=padding)

    # Flatten to 1D (bounds must always be 1D)
    lb = lb_pool.reshape(-1)
    ub = ub_pool.reshape(-1)

    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_op(
        f"avgpool2d:{L.id}", list(L.out_vars + L.in_vars), kernel_size=kernel_size, stride=stride,
        padding=padding, input_shape=input_shape, output_shape=L.meta.get("output_shape"),)
    
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
    """HybridZ transfer function for 1D convolution with strict shape validation."""
    # STRICT MODE: Assert input_shape metadata exists and matches bounds
    input_shape = _assert_shape_match(L, Bin, expected_ndim=3)
    N, C, W = input_shape

    weight = L.params["weight"]  # (out_channels, in_channels, kernel_w)
    bias = L.params.get("bias", None)
    stride = L.meta.get("stride", 1)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)
    groups = L.meta.get("groups", 1)

    # Reshape using validated shape (no guessing)
    Bin_lb = Bin.lb.view(N, C, W)
    Bin_ub = Bin.ub.view(N, C, W)

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
    """HybridZ transfer function for 3D convolution with strict shape validation."""
    # STRICT MODE: Assert input_shape metadata exists and matches bounds
    input_shape = _assert_shape_match(L, Bin, expected_ndim=5)
    N, C, D, H, W = input_shape

    weight = L.params["weight"]  # (out_channels, in_channels, kernel_d, kernel_h, kernel_w)
    bias = L.params.get("bias", None)
    stride = L.meta.get("stride", 1)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)
    groups = L.meta.get("groups", 1)

    # Reshape using validated shape (no guessing)
    Bin_lb = Bin.lb.view(N, C, D, H, W)
    Bin_ub = Bin.ub.view(N, C, D, H, W)

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
    """HybridZ transfer function for 1D max pooling with strict shape validation."""
    # STRICT MODE: Assert input_shape metadata exists and matches bounds
    input_shape = _assert_shape_match(L, Bin, expected_ndim=3)
    N, C, W = input_shape

    kernel_size = L.meta.get("kernel_size", 2)
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)

    # Reshape using validated shape (no guessing)
    Bin_lb = Bin.lb.view(N, C, W)
    Bin_ub = Bin.ub.view(N, C, W)

    lb_pool = F.max_pool1d(Bin_lb, kernel_size, stride=stride, padding=padding, dilation=dilation)
    ub_pool = F.max_pool1d(Bin_ub, kernel_size, stride=stride, padding=padding, dilation=dilation)

    lb = lb_pool.reshape(-1)
    ub = ub_pool.reshape(-1)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"maxpool1d:{L.id}", list(L.out_vars + L.in_vars), kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                input_shape=input_shape, output_shape=L.meta.get("output_shape"))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_maxpool3d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 3D max pooling with strict shape validation."""
    # STRICT MODE: Assert input_shape metadata exists and matches bounds
    input_shape = _assert_shape_match(L, Bin, expected_ndim=5)
    N, C, D, H, W = input_shape

    kernel_size = L.meta.get("kernel_size", 2)
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)

    # Reshape using validated shape (no guessing)
    Bin_lb = Bin.lb.view(N, C, D, H, W)
    Bin_ub = Bin.ub.view(N, C, D, H, W)

    lb_pool = F.max_pool3d(Bin_lb, kernel_size, stride=stride, padding=padding, dilation=dilation)
    ub_pool = F.max_pool3d(Bin_ub, kernel_size, stride=stride, padding=padding, dilation=dilation)

    lb = lb_pool.reshape(-1)
    ub = ub_pool.reshape(-1)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"maxpool3d:{L.id}", list(L.out_vars + L.in_vars), kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                input_shape=input_shape, output_shape=L.meta.get("output_shape"))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_avgpool1d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 1D average pooling with strict shape validation."""
    # STRICT MODE: Assert input_shape metadata exists and matches bounds
    input_shape = _assert_shape_match(L, Bin, expected_ndim=3)
    N, C, W = input_shape

    kernel_size = L.meta.get("kernel_size", 2)
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)

    # Reshape using validated shape (no guessing)
    Bin_lb = Bin.lb.view(N, C, W)
    Bin_ub = Bin.ub.view(N, C, W)

    lb_pool = F.avg_pool1d(Bin_lb, kernel_size, stride=stride, padding=padding)
    ub_pool = F.avg_pool1d(Bin_ub, kernel_size, stride=stride, padding=padding)

    lb = lb_pool.reshape(-1)
    ub = ub_pool.reshape(-1)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"avgpool1d:{L.id}", list(L.out_vars + L.in_vars), kernel_size=kernel_size,
                stride=stride, padding=padding,
                input_shape=input_shape, output_shape=L.meta.get("output_shape"))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_pad(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for padding with strict shape validation."""
    pads = L.meta.get("pad", L.meta.get("pads", None))
    if pads is None:
        raise KeyError(f"pad/pads not found in meta for PAD layer {L.id}")

    mode = L.meta.get("mode", "constant")
    value = float(L.meta.get("value", 0.0))

    # STRICT MODE: Assert input_shape metadata exists
    assert "input_shape" in L.meta, (
        f"Layer {L.id} (PAD): missing 'input_shape' in metadata."
    )
    in_shape = tuple(L.meta["input_shape"])

    # Validate bounds match input_shape
    expected_numel = 1
    for d in in_shape:
        expected_numel *= int(d)
    assert Bin.lb.numel() == expected_numel, (
        f"Layer {L.id} (PAD): bounds size mismatch. "
        f"Expected {expected_numel} from input_shape={in_shape}, got {Bin.lb.numel()}."
    )

    # Reshape using validated shape (no guessing)
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
    """HybridZ transfer function for upsampling with strict shape validation."""
    # STRICT MODE: Assert input_shape metadata exists
    assert "input_shape" in L.meta, (
        f"Layer {L.id} (UPSAMPLE): missing 'input_shape' in metadata."
    )
    in_shape = tuple(L.meta["input_shape"])

    # Validate bounds match input_shape
    expected_numel = 1
    for d in in_shape:
        expected_numel *= int(d)
    assert Bin.lb.numel() == expected_numel, (
        f"Layer {L.id} (UPSAMPLE): bounds size mismatch. "
        f"Expected {expected_numel} from input_shape={in_shape}, got {Bin.lb.numel()}."
    )

    # Reshape using validated shape (no guessing)
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
    """HybridZ transfer function for slice operation with strict shape validation."""
    # STRICT MODE: Assert input_shape metadata exists
    assert "input_shape" in L.meta, (
        f"Layer {L.id} (SLICE): missing 'input_shape' in metadata."
    )
    inp_shape = tuple(L.meta["input_shape"])

    # Validate bounds match input_shape
    expected_numel = 1
    for d in inp_shape:
        expected_numel *= int(d)
    assert Bin.lb.numel() == expected_numel, (
        f"Layer {L.id} (SLICE): bounds size mismatch. "
        f"Expected {expected_numel} from input_shape={inp_shape}, got {Bin.lb.numel()}."
    )

    # Reshape using validated shape (no guessing)
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
    """HybridZ transfer function for gather operation with strict shape validation."""
    # STRICT MODE: Assert input_shape metadata exists
    assert "input_shape" in L.meta, (
        f"Layer {L.id} (GATHER): missing 'input_shape' in metadata."
    )
    inp_shape = tuple(L.meta["input_shape"])

    # Validate bounds match input_shape
    expected_numel = 1
    for d in inp_shape:
        expected_numel *= int(d)
    assert Bin.lb.numel() == expected_numel, (
        f"Layer {L.id} (GATHER): bounds size mismatch. "
        f"Expected {expected_numel} from input_shape={inp_shape}, got {Bin.lb.numel()}."
    )

    axis = int(L.meta.get("axis", 0))

    # Reshape using validated shape (no guessing)
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