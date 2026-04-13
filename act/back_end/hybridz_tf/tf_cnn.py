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
from act.back_end.core import Bounds, Fact
from act.back_end.solver.solver_hz import HZono, hz_compute_bounds
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
        return Fact(bounds=hz_compute_bounds(tf._hz_cache[L.id]), cons=fact.cons)
    return fact


def tf_maxpool2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        bds = hz_compute_bounds(hz_in)
        shape = L.params.get("input_shape")
        done = False
        if shape is not None:
            _, C, H, W = shape
            if hz_in is not None:
                _, idx = F.max_pool2d(
                    bds.lb.view(1, C, H, W),
                    kernel_size=L.params.get("kernel_size", 2),
                    stride=L.params.get("stride", L.params.get("kernel_size", 2)),
                    padding=L.params.get("padding", 0),
                    return_indices=True,
                )
                w = idx.reshape(-1)
                tf._hz_cache[L.id] = HZono(
                    c=hz_in.c[w], Gc=hz_in.Gc[w], Gb=hz_in.Gb[w],
                    Ac=hz_in.Ac.clone(), Ab=hz_in.Ab.clone(), b=hz_in.b.clone(),
                )
                done = True
        if not done:
            hz_in = None
    fact = interval.tf_maxpool2d(L, bounds)
    if hz_in is not None:
        return Fact(bounds=hz_compute_bounds(tf._hz_cache[L.id]), cons=fact.cons)
    return fact


# --- HZ conv2d (zonotope domain) ---

def _conv2d_generators(
    G, weight, C, H, W, stride, padding, dilation, groups, n_out, dtype, device
):
    """Apply conv2d to a generator matrix (Gc or Gb)."""
    if G.shape[1] == 0:
        return torch.zeros((n_out, 0), dtype=dtype, device=device)
    ncols = G.shape[1]
    imgs = G.t().contiguous().view(ncols, C, H, W)
    out = F.conv2d(
        imgs,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return out.permute(1, 2, 3, 0).contiguous().reshape(-1, ncols)


def hz_conv2d(
    hz: HZono, weight, bias, stride, padding, dilation, groups, input_shape
) -> HZono:
    """Apply conv2d to a hybrid zonotope: convolve center and each generator column."""
    dtype, device = hz.c.dtype, hz.c.device
    # Inline shape extraction (no parse_input_shape dependency)
    _, C, H, W = input_shape
    weight = weight.to(dtype=dtype, device=device)

    c_img = hz.c.view(C, H, W).unsqueeze(0)
    out_c = F.conv2d(
        c_img,
        weight,
        bias=bias.to(dtype=dtype, device=device) if bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    new_c = out_c.reshape(-1, 1)
    n_out = new_c.shape[0]

    new_Gc = _conv2d_generators(
        hz.Gc, weight, C, H, W, stride, padding, dilation, groups, n_out, dtype, device
    )
    new_Gb = _conv2d_generators(
        hz.Gb, weight, C, H, W, stride, padding, dilation, groups, n_out, dtype, device
    )

    return HZono(
        c=new_c,
        Gc=new_Gc,
        Gb=new_Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
    )
