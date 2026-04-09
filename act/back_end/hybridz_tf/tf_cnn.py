# ===- act/back_end/hybridz_tf/tf_cnn.py - HybridZ CNN Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ CNN Transfer Functions. Implements HybridZ-based transfer functions
#   for CNN layers including convolution, pooling, and tensor reshaping
#   operations.
#
# ===---------------------------------------------------------------------===#


import torch
import torch.nn.functional as F
from act.back_end.core import Bounds, Fact, Layer, ConSet


@torch.no_grad()
def hybridz_tf_conv2d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 2D convolution with enhanced precision."""
    weight = L.params["weight"]
    bias = L.params.get("bias", None)
    stride = L.params.get("stride", 1)
    padding = L.params.get("padding", 0)
    dilation = L.params.get("dilation", 1)
    groups = L.params.get("groups", 1)
    input_shape = L.params.get("input_shape")

    if input_shape is not None and len(input_shape) == 4:
        _, C, H, W = input_shape
    elif input_shape is not None and len(input_shape) == 3:
        C, H, W = input_shape
    else:
        out_c, in_c_per_g, kH, kW = weight.shape
        C = in_c_per_g * groups
        spatial = Bin.lb.numel() // C
        H = W = int(spatial**0.5)
        if H * W != spatial:
            for h in range(int(spatial**0.5) + 10, 0, -1):
                if spatial % h == 0:
                    H, W = h, spatial // h
                    break

    Bin_lb = Bin.lb.view(1, C, H, W)
    Bin_ub = Bin.ub.view(1, C, H, W)

    weight_pos = torch.clamp(weight, min=0)
    weight_neg = torch.clamp(weight, max=0)

    lb_conv = F.conv2d(
        Bin_lb,
        weight_pos,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    lb_conv += F.conv2d(
        Bin_ub,
        weight_neg,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    ub_conv = F.conv2d(
        Bin_ub,
        weight_pos,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    ub_conv += F.conv2d(
        Bin_lb,
        weight_neg,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    if bias is not None:
        lb_conv += bias.view(1, -1, 1, 1)
        ub_conv += bias.view(1, -1, 1, 1)

    Bout = Bounds(lb=lb_conv.view(-1), ub=ub_conv.view(-1))

    return Fact(bounds=Bout, cons=ConSet())


@torch.no_grad()
def hybridz_tf_maxpool2d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 2D max pooling."""
    kernel_size = L.params.get("kernel_size", 2)
    stride = L.params.get("stride", kernel_size)
    padding = L.params.get("padding", 0)
    input_shape = L.params.get("input_shape")

    if input_shape is not None and len(input_shape) == 4:
        _, C, H, W = input_shape
    elif input_shape is not None and len(input_shape) == 3:
        C, H, W = input_shape
    else:
        total = Bin.lb.numel()
        C = 1
        spatial = total
        H = W = int(spatial**0.5)

    lb_in = Bin.lb.view(1, C, H, W)
    ub_in = Bin.ub.view(1, C, H, W)

    lb_pool = F.max_pool2d(lb_in, kernel_size, stride, padding)
    ub_pool = F.max_pool2d(ub_in, kernel_size, stride, padding)

    Bout = Bounds(lb=lb_pool.view(-1), ub=ub_pool.view(-1))

    cons = ConSet()
    cons.add_op(
        f"maxpool2d:{L.id}",
        list(L.out_vars + L.in_vars),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        input_shape=input_shape,
    )

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_flatten(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for tensor flattening."""
    # Flattening is just reshaping - bounds remain the same
    start_dim = L.params.get("start_dim", 1)
    end_dim = L.params.get("end_dim", -1)

    # Simple reshape - no change in bounds
    lb = Bin.lb.flatten()
    ub = Bin.ub.flatten()
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(
        f"flatten:{L.id}",
        list(L.out_vars + L.in_vars),
        start_dim=start_dim,
        end_dim=end_dim,
        input_shape=L.params.get("input_shape"),
        output_shape=L.params.get("output_shape"),
    )

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_reshape(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for general tensor reshaping."""
    target_shape = L.params.get("target_shape")

    # Reshape bounds preserving values
    lb = Bin.lb.reshape(target_shape).flatten() if target_shape else Bin.lb.flatten()
    ub = Bin.ub.reshape(target_shape).flatten() if target_shape else Bin.ub.flatten()
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(
        f"reshape:{L.id}",
        list(L.out_vars + L.in_vars),
        target_shape=target_shape,
        input_shape=L.params.get("input_shape"),
        output_shape=L.params.get("output_shape"),
    )

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_avgpool2d(L: Layer, Bin: Bounds) -> Fact:
    kernel_size = L.params.get("kernel_size", 2)
    stride = L.params.get("stride", kernel_size)
    padding = L.params.get("padding", 0)
    input_shape = L.params.get("input_shape")
    output_shape = L.params.get("output_shape")

    if input_shape is not None and len(input_shape) == 4:
        _, C, H, W = input_shape
    elif input_shape is not None and len(input_shape) == 3:
        C, H, W = input_shape
    else:
        total = Bin.lb.numel()
        C = 1
        H = W = int(total**0.5)

    lb_in = Bin.lb.view(1, C, H, W)
    ub_in = Bin.ub.view(1, C, H, W)

    lb_pool = F.avg_pool2d(lb_in, kernel_size, stride, padding)
    ub_pool = F.avg_pool2d(ub_in, kernel_size, stride, padding)

    Bout = Bounds(lb=lb_pool.view(-1), ub=ub_pool.view(-1))
    cons = ConSet()
    cons.add_op(
        f"avgpool2d:{L.id}",
        list(L.out_vars + L.in_vars),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        input_shape=input_shape,
        output_shape=output_shape,
    )
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_conv1d(L: Layer, Bin: Bounds) -> Fact:
    weight = L.params["weight"]
    bias = L.params.get("bias", None)
    stride = L.params.get("stride", 1)
    padding = L.params.get("padding", 0)
    dilation = L.params.get("dilation", 1)
    groups = L.params.get("groups", 1)
    input_shape = L.params.get("input_shape")

    if input_shape is not None and len(input_shape) == 3:
        _, C, W = input_shape
    elif input_shape is not None and len(input_shape) == 2:
        C, W = input_shape
    else:
        out_c, in_c_per_g, kW = weight.shape
        C = in_c_per_g * groups
        W = Bin.lb.numel() // C

    lb_in = Bin.lb.view(1, C, W)
    ub_in = Bin.ub.view(1, C, W)

    weight_pos = torch.clamp(weight, min=0)
    weight_neg = torch.clamp(weight, max=0)

    lb_conv = F.conv1d(
        lb_in,
        weight_pos,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    lb_conv += F.conv1d(
        ub_in,
        weight_neg,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    ub_conv = F.conv1d(
        ub_in,
        weight_pos,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    ub_conv += F.conv1d(
        lb_in,
        weight_neg,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    if bias is not None:
        lb_conv += bias.view(1, -1, 1)
        ub_conv += bias.view(1, -1, 1)

    Bout = Bounds(lb=lb_conv.view(-1), ub=ub_conv.view(-1))
    cons = ConSet()
    cons.add_op(f"conv1d:{L.id}", list(L.out_vars + L.in_vars))
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_conv3d(L: Layer, Bin: Bounds) -> Fact:
    weight = L.params["weight"]
    bias = L.params.get("bias", None)
    stride = L.params.get("stride", 1)
    padding = L.params.get("padding", 0)
    dilation = L.params.get("dilation", 1)
    groups = L.params.get("groups", 1)
    input_shape = L.params.get("input_shape")

    if input_shape is not None and len(input_shape) == 5:
        _, C, D, H, W = input_shape
    elif input_shape is not None and len(input_shape) == 4:
        C, D, H, W = input_shape
    else:
        out_c, in_c_per_g = weight.shape[:2]
        C = in_c_per_g * groups
        spatial = Bin.lb.numel() // C
        side = int(round(spatial ** (1 / 3)))
        D = H = W = side

    lb_in = Bin.lb.view(1, C, D, H, W)
    ub_in = Bin.ub.view(1, C, D, H, W)

    weight_pos = torch.clamp(weight, min=0)
    weight_neg = torch.clamp(weight, max=0)

    lb_conv = F.conv3d(
        lb_in,
        weight_pos,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    lb_conv += F.conv3d(
        ub_in,
        weight_neg,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    ub_conv = F.conv3d(
        ub_in,
        weight_pos,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    ub_conv += F.conv3d(
        lb_in,
        weight_neg,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    if bias is not None:
        lb_conv += bias.view(1, -1, 1, 1, 1)
        ub_conv += bias.view(1, -1, 1, 1, 1)

    Bout = Bounds(lb=lb_conv.view(-1), ub=ub_conv.view(-1))
    cons = ConSet()
    cons.add_op(f"conv3d:{L.id}", list(L.out_vars + L.in_vars))
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_convtranspose2d(L: Layer, Bin: Bounds) -> Fact:
    weight = L.params["weight"]
    bias = L.params.get("bias", None)
    stride = L.params.get("stride", 1)
    padding = L.params.get("padding", 0)
    output_padding = L.params.get("output_padding", 0)
    dilation = L.params.get("dilation", 1)
    groups = L.params.get("groups", 1)
    input_shape = L.params.get("input_shape")

    if input_shape is not None and len(input_shape) == 4:
        _, C, H, W = input_shape
    elif input_shape is not None and len(input_shape) == 3:
        C, H, W = input_shape
    else:
        in_c = weight.shape[0]
        C = in_c
        spatial = Bin.lb.numel() // C
        H = W = int(spatial**0.5)

    lb_in = Bin.lb.view(1, C, H, W)
    ub_in = Bin.ub.view(1, C, H, W)

    weight_pos = torch.clamp(weight, min=0)
    weight_neg = torch.clamp(weight, max=0)

    lb_conv = F.conv_transpose2d(
        lb_in,
        weight_pos,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )
    lb_conv += F.conv_transpose2d(
        ub_in,
        weight_neg,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )

    ub_conv = F.conv_transpose2d(
        ub_in,
        weight_pos,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )
    ub_conv += F.conv_transpose2d(
        lb_in,
        weight_neg,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )

    if bias is not None:
        lb_conv += bias.view(1, -1, 1, 1)
        ub_conv += bias.view(1, -1, 1, 1)

    Bout = Bounds(lb=lb_conv.view(-1), ub=ub_conv.view(-1))
    cons = ConSet()
    cons.add_op(f"convtranspose2d:{L.id}", list(L.out_vars + L.in_vars))
    return Fact(bounds=Bout, cons=cons)
