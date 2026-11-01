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
try:
    import numpy as np
    import scipy.sparse as sp
except ImportError:
    np = None
    sp = None
from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import HZono, SparseHZono
from act.back_end.hybridz_tf.tf_mlp import _hz_fact, _sparse_apply_per_batch_linear, _sparse_available
from act.back_end.utils import avgpool2d_denominators, avgpool2d_output_hw
import act.back_end.interval_tf.tf_cnn as interval


# --- HZ transfer functions (CNN) ---

def tf_conv2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        input_shape = L.params.get("input_shape")
        if input_shape is not None:
            tf._hz_cache[L.id] = hz_conv2d(
                hz_in, L.params["weight"], L.params.get("bias"),
                L.params.get("stride", 1), L.params.get("padding", 0),
                L.params.get("dilation", 1), L.params.get("groups", 1), input_shape,
            )
        else:
            hz_in = None
    fact = interval.tf_conv2d(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_maxpool2d(L, bounds, tf):
    # MaxPool is not affine in the HZ rows; keep interval bounds for soundness.
    tf._hz_cache[L.id] = None
    return interval.tf_maxpool2d(L, bounds)


def _pair(x):
    return (int(x), int(x)) if isinstance(x, int) else (int(x[0]), int(x[1]))


def _spatial_shape(input_shape):
    if len(input_shape) == 4:
        _, C, H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError(f"Unexpected input_shape={input_shape}")
    return int(C), int(H), int(W)


def sparse_conv2d_matrix_from_layer(layer):
    input_shape = layer.params.get("input_shape")
    if input_shape is None:
        raise ValueError("missing conv2d input_shape")
    C, H, W = _spatial_shape(tuple(int(d) for d in input_shape))
    weight = layer.params["weight"].detach().cpu().double().numpy()
    stride = _pair(layer.params.get("stride", 1))
    padding = _pair(layer.params.get("padding", 0))
    dilation = _pair(layer.params.get("dilation", 1))
    groups = int(layer.params.get("groups", 1))
    OC, ICg, KH, KW = weight.shape
    OH = (H + 2 * padding[0] - dilation[0] * (KH - 1) - 1) // stride[0] + 1
    OW = (W + 2 * padding[1] - dilation[1] * (KW - 1) - 1) // stride[1] + 1
    out_per_group = OC // groups
    rows, cols, data = [], [], []
    for oc in range(OC):
        group = oc // out_per_group
        c0 = group * ICg
        for oh in range(OH):
            for ow in range(OW):
                r = oc * OH * OW + oh * OW + ow
                for icg in range(ICg):
                    ic = c0 + icg
                    for kh in range(KH):
                        ih = oh * stride[0] - padding[0] + kh * dilation[0]
                        if ih < 0 or ih >= H:
                            continue
                        for kw in range(KW):
                            iw = ow * stride[1] - padding[1] + kw * dilation[1]
                            if iw < 0 or iw >= W:
                                continue
                            rows.append(r)
                            cols.append(ic * H * W + ih * W + iw)
                            data.append(weight[oc, icg, kh, kw])
    mat = sp.csr_matrix((data, (rows, cols)), shape=(OC * OH * OW, C * H * W))
    bias = layer.params.get("bias")
    b = None
    if bias is not None:
        b = np.repeat(bias.detach().cpu().double().numpy().reshape(-1), OH * OW)
    return mat, b


def sparse_convtranspose2d_matrix_from_layer(layer):
    input_shape = layer.params.get("input_shape")
    if input_shape is None:
        raise ValueError("missing convtranspose2d input_shape")
    C, H, W = _spatial_shape(tuple(int(d) for d in input_shape))
    weight = layer.params["weight"].detach().cpu().double().numpy()
    stride = _pair(layer.params.get("stride", 1))
    padding = _pair(layer.params.get("padding", 0))
    output_padding = _pair(layer.params.get("output_padding", 0))
    dilation = _pair(layer.params.get("dilation", 1))
    groups = int(layer.params.get("groups", 1))
    IC, OCg, KH, KW = weight.shape
    OH = (H - 1) * stride[0] - 2 * padding[0] + dilation[0] * (KH - 1) + output_padding[0] + 1
    OW = (W - 1) * stride[1] - 2 * padding[1] + dilation[1] * (KW - 1) + output_padding[1] + 1
    in_per_group = IC // groups
    OC = OCg * groups
    rows, cols, data = [], [], []
    for ic in range(IC):
        group = ic // in_per_group
        oc0 = group * OCg
        for ih in range(H):
            for iw in range(W):
                cidx = ic * H * W + ih * W + iw
                for ocg in range(OCg):
                    oc = oc0 + ocg
                    for kh in range(KH):
                        oh = ih * stride[0] - padding[0] + kh * dilation[0]
                        if oh < 0 or oh >= OH:
                            continue
                        for kw in range(KW):
                            ow = iw * stride[1] - padding[1] + kw * dilation[1]
                            if ow < 0 or ow >= OW:
                                continue
                            rows.append(oc * OH * OW + oh * OW + ow)
                            cols.append(cidx)
                            data.append(weight[ic, ocg, kh, kw])
    mat = sp.csr_matrix((data, (rows, cols)), shape=(OC * OH * OW, C * H * W))
    bias = layer.params.get("bias")
    b = None
    if bias is not None:
        b = np.repeat(bias.detach().cpu().double().numpy().reshape(-1), OH * OW)
    return mat, b


def sparse_avgpool2d_matrix_from_layer(layer):
    input_shape = layer.params.get("input_shape")
    if input_shape is None:
        raise ValueError("missing avgpool2d input_shape")
    C, H, W = _spatial_shape(tuple(int(d) for d in input_shape))
    kernel = _pair(layer.params["kernel_size"])
    raw_stride = layer.params.get("stride")
    stride = _pair(raw_stride if raw_stride is not None else layer.params["kernel_size"])
    padding = _pair(layer.params.get("padding", 0))
    ceil_mode = bool(layer.params.get("ceil_mode", False))
    count_include_pad = bool(layer.params.get("count_include_pad", True))
    divisor_override = layer.params.get("divisor_override")
    OH, OW = avgpool2d_output_hw(
        (H, W), kernel, stride, padding, ceil_mode
    )
    denominators = avgpool2d_denominators(
        (H, W),
        (OH, OW),
        kernel,
        stride,
        padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
        dtype=torch.float64,
    ).cpu().numpy()
    rows, cols, data = [], [], []
    for c in range(C):
        for oh in range(OH):
            for ow in range(OW):
                r = c * OH * OW + oh * OW + ow
                for kh in range(kernel[0]):
                    ih = oh * stride[0] - padding[0] + kh
                    if ih < 0 or ih >= H:
                        continue
                    for kw in range(kernel[1]):
                        iw = ow * stride[1] - padding[1] + kw
                        if iw < 0 or iw >= W:
                            continue
                        rows.append(r)
                        cols.append(c * H * W + ih * W + iw)
                        data.append(1.0 / float(denominators[oh, ow]))
    return sp.csr_matrix((data, (rows, cols)), shape=(C * OH * OW, C * H * W)), None


def sparse_hz_apply_layer(L, hz: SparseHZono, input_bounds: Bounds, result, tf):
    if not _sparse_available():
        return True, None, "scipy_unavailable"
    k = L.kind.upper()
    if k == "CONV2D":
        W, b = sparse_conv2d_matrix_from_layer(L)
        return True, _sparse_apply_per_batch_linear(hz, W, b), None
    if k == "CONVTRANSPOSE2D":
        W, b = sparse_convtranspose2d_matrix_from_layer(L)
        return True, _sparse_apply_per_batch_linear(hz, W, b), None
    if k == "AVGPOOL2D":
        W, b = sparse_avgpool2d_matrix_from_layer(L)
        return True, _sparse_apply_per_batch_linear(hz, W, b), None
    if k == "MAXPOOL2D":
        return True, None, "unsupported_sparse_maxpool2d"
    return False, None, None


# --- HZ conv2d (zonotope domain) ---

def _conv2d_generators(
    G, weight, B, C, H, W, stride, padding, dilation, groups, n_out_per_sample
):
    """Apply conv2d to a generator matrix ``(B*C*H*W, ng)`` and return
    a generator matrix ``(B*n_out_per_sample, ng)``. Each generator
    column is convolved independently per batch element by stacking
    ``ng * B`` images into conv2d's leading "batch" axis.
    """
    if G.shape[1] == 0:
        return G.new_zeros(B * n_out_per_sample, 0)
    ng = G.shape[1]
    imgs = G.t().contiguous().view(ng, B, C, H, W).reshape(ng * B, C, H, W)
    out = F.conv2d(
        imgs,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    _, Cp, Hp, Wp = out.shape
    return (
        out.view(ng, B, Cp, Hp, Wp)
        .permute(1, 2, 3, 4, 0)
        .contiguous()
        .reshape(B * Cp * Hp * Wp, ng)
    )


def hz_conv2d(
    hz: HZono, weight, bias, stride, padding, dilation, groups, input_shape
) -> HZono:
    if len(input_shape) == 4:
        _, C, H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError(f"Unexpected input_shape={input_shape}, expected 3D or 4D")
    weight = weight.to(hz.c)

    spatial_in = C * H * W
    B = hz.c.numel() // spatial_in
    c_img = hz.c.view(B, C, H, W)
    out_c = F.conv2d(
        c_img,
        weight,
        bias=bias.to(hz.c) if bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    _, Cp, Hp, Wp = out_c.shape
    new_c = out_c.reshape(-1, 1)
    n_out_per_sample = Cp * Hp * Wp

    new_Gc = _conv2d_generators(
        hz.Gc, weight, B, C, H, W, stride, padding, dilation, groups, n_out_per_sample
    )
    new_Gb = _conv2d_generators(
        hz.Gb, weight, B, C, H, W, stride, padding, dilation, groups, n_out_per_sample
    )

    return HZono(
        c=new_c,
        Gc=new_Gc,
        Gb=new_Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


def _spatial_op_generators(G, op_fn, B, C, H, W, n_out_per_sample):
    """Apply a linear BCHW spatial operator to each HZ generator column.

    If y = S(x) + b and x = c + G xi, then each output generator is
    G'[:, j] = vec(S(unvec(G[:, j]))), without materializing the matrix for S.
    """
    if G.shape[1] == 0:
        return G.new_zeros(B * n_out_per_sample, 0)
    ng = G.shape[1]
    imgs = G.t().contiguous().view(ng, B, C, H, W).reshape(ng * B, C, H, W)
    out = op_fn(imgs)
    _, Cp, Hp, Wp = out.shape
    return (
        out.view(ng, B, Cp, Hp, Wp)
        .permute(1, 2, 3, 4, 0)
        .contiguous()
        .reshape(B * Cp * Hp * Wp, ng)
    )


def _hz_spatial_affine(hz: HZono, op_fn, input_shape, bias=None) -> HZono:
    if len(input_shape) == 4:
        _, C, H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError(f"Unexpected input_shape={input_shape}")
    spatial_in = C * H * W
    B = hz.c.numel() // spatial_in
    out_c = op_fn(hz.c.view(B, C, H, W))
    _, Cp, Hp, Wp = out_c.shape
    if bias is not None:
        out_c = out_c + bias.to(hz.c).view(1, -1, 1, 1)
    n_out = Cp * Hp * Wp
    return HZono(
        c=out_c.reshape(-1, 1),
        Gc=_spatial_op_generators(hz.Gc, op_fn, B, C, H, W, n_out),
        Gb=_spatial_op_generators(hz.Gb, op_fn, B, C, H, W, n_out),
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


def hz_avgpool2d(
    hz,
    kernel_size,
    stride,
    padding,
    input_shape,
    *,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
) -> HZono:
    op = lambda x: F.avg_pool2d(
        x,
        kernel_size=kernel_size,
        stride=stride if stride is not None else kernel_size,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
    return _hz_spatial_affine(hz, op, input_shape)


def tf_avgpool2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        ishape = L.params.get("input_shape")
        if ishape is not None:
            tf._hz_cache[L.id] = hz_avgpool2d(
                hz_in,
                L.params.get("kernel_size"),
                L.params.get("stride"),
                L.params.get("padding", 0),
                ishape,
                ceil_mode=bool(L.params.get("ceil_mode", False)),
                count_include_pad=bool(L.params.get("count_include_pad", True)),
                divisor_override=L.params.get("divisor_override"),
            )
        else:
            hz_in = None
    fact = interval.tf_avgpool2d(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def hz_convtranspose2d(
    hz, weight, bias, stride, padding, output_padding, dilation, groups, input_shape
) -> HZono:
    weight = weight.to(hz.c)
    op = lambda x: F.conv_transpose2d(
        x,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )
    return _hz_spatial_affine(hz, op, input_shape, bias=bias)


def tf_convtranspose2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        ishape = L.params.get("input_shape")
        if ishape is not None:
            tf._hz_cache[L.id] = hz_convtranspose2d(
                hz_in,
                L.params["weight"],
                L.params.get("bias"),
                L.params.get("stride", 1),
                L.params.get("padding", 0),
                L.params.get("output_padding", 0),
                L.params.get("dilation", 1),
                L.params.get("groups", 1),
                ishape,
            )
        else:
            hz_in = None
    fact = interval.tf_convtranspose2d(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact
