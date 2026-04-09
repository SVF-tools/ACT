#===- act/back_end/hybridz_tf/tf_mlp.py - HybridZ MLP Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ MLP Transfer Functions. Each function has the signature
#   (Layer, Bounds) -> Fact, computing interval bounds and generating ConSet
#   constraints.  HZ zonotope tracking is handled externally by HybridzTF.apply().
#
#===---------------------------------------------------------------------===#

import torch
from act.back_end.core import Bounds, Fact, Layer, ConSet


@torch.no_grad()
def hybridz_tf_dense(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for dense/linear layers with zonotope precision."""
    W = L.params["weight"]
    b = L.params.get("bias", None)

    if W.shape[1] != Bin.lb.shape[0]:
        raise ValueError(f"Dense layer input mismatch: W expects {W.shape[1]}, got {Bin.lb.shape[0]}")

    W_pos = torch.clamp(W, min=0)
    W_neg = torch.clamp(W, max=0)

    lb = W_pos @ Bin.lb + W_neg @ Bin.ub
    ub = W_pos @ Bin.ub + W_neg @ Bin.lb

    if b is not None:
        lb = lb + b
        ub = ub + b

    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"dense:{L.id}", list(L.out_vars + L.in_vars), W=W, b=b)

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_bias(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for bias addition."""
    c = L.params["c"]

    lb = Bin.lb + c
    ub = Bin.ub + c
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"bias:{L.id}", list(L.out_vars + L.in_vars), c=c)

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_scale(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for element-wise scaling."""
    a = L.params["a"]

    a_pos = torch.clamp(a, min=0)
    a_neg = torch.clamp(a, max=0)

    lb = a_pos * Bin.lb + a_neg * Bin.ub
    ub = a_pos * Bin.ub + a_neg * Bin.lb
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"scale:{L.id}", list(L.out_vars + L.in_vars), a=a)

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_relu(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for ReLU activation with precise constraint handling."""
    lb = torch.clamp(Bin.lb, min=0)
    ub = torch.clamp(Bin.ub, min=0)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()

    slope = torch.zeros_like(Bin.lb)
    shift = torch.zeros_like(Bin.lb)

    idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]
    idx_on = torch.where(Bin.lb >= 0)[0]
    idx_off = torch.where(Bin.ub <= 0)[0]
    if len(idx_amb) > 0:
        slope = Bin.lb[idx_amb] / torch.clamp(Bin.ub[idx_amb] - Bin.lb[idx_amb], min=1e-12)
        shift = -slope * Bin.lb[idx_amb]
    cons.add_op(f"relu:{L.id}", list(L.out_vars + L.in_vars), idx_on=idx_on, idx_off=idx_off, idx_amb=idx_amb, slope=slope, shift=shift)

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_lrelu(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for LeakyReLU."""
    alpha = float(L.params.get("negative_slope", 0.01))

    lb = torch.where(Bin.lb >= 0, Bin.lb, alpha * Bin.lb)
    ub = torch.where(Bin.ub <= 0, alpha * Bin.ub, Bin.ub)
    Bout = Bounds(lb=lb, ub=ub)

    idx_on = torch.where(Bin.lb >= 0)[0]
    idx_off = torch.where(Bin.ub <= 0)[0]
    idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]

    slope = torch.zeros_like(Bin.lb)
    shift = torch.zeros_like(Bin.lb)

    if len(idx_amb) > 0:
        y_at_ub = Bin.ub[idx_amb]
        y_at_lb = alpha * Bin.lb[idx_amb]
        denom = Bin.ub[idx_amb] - Bin.lb[idx_amb]
        slope[idx_amb] = torch.where(denom > 1e-8, (y_at_ub - y_at_lb) / denom, torch.ones_like(denom))
        shift[idx_amb] = y_at_lb - slope[idx_amb] * Bin.lb[idx_amb]

    cons = ConSet()
    cons.add_op(f"lrelu:{L.id}", list(L.out_vars + L.in_vars), alpha=alpha, idx_on=idx_on, idx_off=idx_off, idx_amb=idx_amb, slope=slope[idx_amb], shift=shift[idx_amb])

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_tanh(L: Layer, Bin: Bounds) -> Fact:
    """Tanh with piecewise linear HZ encoding. Returns Fact."""
    lb = torch.tanh(Bin.lb)
    ub = torch.tanh(Bin.ub)
    Bout = Bounds(lb=torch.minimum(lb, ub), ub=torch.maximum(lb, ub))

    cons = ConSet()
    cons.add_op(f"tanh:{L.id}", list(L.out_vars + L.in_vars))

    return Fact(bounds=Bout, cons=cons)

@torch.no_grad()
def hybridz_tf_sigmoid(L: Layer, Bin: Bounds) -> Fact:
    """Sigmoid with piecewise linear HZ encoding. Returns Fact."""
    lb = torch.sigmoid(Bin.lb)
    ub = torch.sigmoid(Bin.ub)
    Bout = Bounds(lb=torch.minimum(lb, ub), ub=torch.maximum(lb, ub))

    cons = ConSet()
    cons.add_op(f"sigmoid:{L.id}", list(L.out_vars + L.in_vars))
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_abs(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for absolute value."""
    idx_pos = torch.where(Bin.lb >= 0)[0]
    idx_neg = torch.where(Bin.ub <= 0)[0]
    idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]
    lb = torch.where(idx_amb[:, None] == torch.arange(len(Bin.lb))[None, :],
                     torch.zeros_like(Bin.lb),
                     torch.where(Bin.lb >= 0, Bin.lb, -Bin.ub))
    ub = torch.maximum(torch.abs(Bin.lb), torch.abs(Bin.ub))
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"abs:{L.id}", list(L.out_vars + L.in_vars), idx_pos=idx_pos, idx_neg=idx_neg, idx_amb=idx_amb)

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_add(L: Layer, Bin1: Bounds, Bin2: Bounds) -> Fact:
    """HybridZ transfer function for element-wise addition."""
    lb = Bin1.lb + Bin2.lb
    ub = Bin1.ub + Bin2.ub
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"add:{L.id}", list(L.out_vars + L.in_vars),)

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_mul(L: Layer, Bin1: Bounds, Bin2: Bounds) -> Fact:
    """HybridZ transfer function for element-wise multiplication with McCormick relaxation."""
    lx, ux = Bin1.lb, Bin1.ub
    ly, uy = Bin2.lb, Bin2.ub

    corners = torch.stack([lx * ly, lx * uy, ux * ly, ux * uy])

    lb = torch.min(corners, dim=0)[0]
    ub = torch.max(corners, dim=0)[0]
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"mcc:{L.id}", list(L.out_vars + L.in_vars),
                lx=Bin1.lb, ux=Bin1.ub, ly=Bin2.lb, uy=Bin2.ub)
    return Fact(bounds=Bout, cons=cons)
