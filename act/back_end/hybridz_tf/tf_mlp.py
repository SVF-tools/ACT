#===- act/back_end/hybridz_tf/tf_mlp.py - HybridZ MLP Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ MLP Transfer Functions. Implements HybridZ-based transfer functions
#   for MLP layers including dense, activation, and basic arithmetic operations.
#
#===---------------------------------------------------------------------===#


import torch
from typing import Optional
from act.back_end.core import Bounds, Fact, Layer, ConSet


@torch.no_grad()
def hybridz_tf_dense(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for dense/linear layers with zonotope precision."""
    # Extract parameters
    W = L.params["W"]  # (out_features, in_features)
    b = L.params.get("b", None)
    
    # Apply linear transformation with HybridZ operations
    # For now, use interval arithmetic as base implementation
    # TODO: Integrate actual HybridZ zonotope operations
    
    # Compute bounds: y = W @ x + b
    if W.shape[1] != Bin.lb.shape[0]:
        raise ValueError(f"Dense layer input mismatch: W expects {W.shape[1]}, got {Bin.lb.shape[0]}")
    
    # Interval multiplication: [a,b] * [c,d] with mixed signs
    W_pos = torch.clamp(W, min=0)
    W_neg = torch.clamp(W, max=0)
    
    # Compute output bounds
    lb = W_pos @ Bin.lb + W_neg @ Bin.ub
    ub = W_pos @ Bin.ub + W_neg @ Bin.lb
    
    if b is not None:
        lb = lb + b
        ub = ub + b
    
    Bout = Bounds(lb=lb, ub=ub)
    
    # Generate constraint for dense layer
    cons = ConSet()
    # cons.add_dense(L.id, L.in_vars, L.out_vars, W, b)
    cons.add_op(f"dense:{L.id}", list(L.out_vars + L.in_vars), W=W, b=b)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_bias(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for bias addition."""
    c = L.params["c"]
    
    # Simple translation
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
    
    # Handle positive/negative scaling
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
    # Determine ReLU phases
    idx_on = torch.where(Bin.lb >= 0)[0]  # Always active
    idx_off = torch.where(Bin.ub <= 0)[0]  # Always inactive
    idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]  # Ambiguous
    
    # Compute output bounds
    lb = torch.clamp(Bin.lb, min=0)
    ub = torch.clamp(Bin.ub, min=0)
    Bout = Bounds(lb=lb, ub=ub)
    
    # HybridZ-specific ReLU constraint generation
    cons = ConSet()
    
    # For ambiguous neurons, use HybridZ slope computation
    slope = torch.zeros_like(Bin.lb)
    shift = torch.zeros_like(Bin.lb)
    
    if len(idx_amb) > 0:
        # HybridZ: More precise slope computation
        slope = Bin.lb[idx_amb] / torch.clamp(Bin.ub[idx_amb] - Bin.lb[idx_amb], min=1e-12)
        shift = -slope * Bin.lb[idx_amb]
    else:
        s = torch.empty(0, dtype=Bin.lb.dtype, device=Bin.lb.device)
        t = torch.empty(0, dtype=Bin.lb.dtype, device=Bin.lb.device)
        
    cons.add_op(f"relu:{L.id}", list(L.out_vars + L.in_vars), idx_on=idx_on, idx_off=idx_off, idx_amb=idx_amb, slope=slope, shift=shift)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_lrelu(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for LeakyReLU."""
    alpha = float(L.meta.get("negative_slope", 0.01))

    # Determine phases
    idx_on = torch.where(Bin.lb >= 0)[0]
    idx_off = torch.where(Bin.ub <= 0)[0]
    idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]
    
    # Output bounds
    lb = torch.where(Bin.lb >= 0, Bin.lb, alpha * Bin.lb)
    ub = torch.where(Bin.ub <= 0, alpha * Bin.ub, Bin.ub)
    Bout = Bounds(lb=lb, ub=ub)
    
    # HybridZ slope computation for ambiguous region
    slope = torch.zeros_like(Bin.lb)
    shift = torch.zeros_like(Bin.lb)
    
    if len(idx_amb) > 0:
        y_at_ub = Bin.ub[idx_amb]
        y_at_lb = alpha * Bin.lb[idx_amb]
        denom = Bin.ub[idx_amb] - Bin.lb[idx_amb]
        slope[idx_amb] = torch.where(denom > 1e-8, (y_at_ub - y_at_lb) / denom, torch.ones_like(denom))
        shift[idx_amb] = y_at_lb - slope[idx_amb] * Bin.lb[idx_amb]
    
    cons = ConSet()
    cons.add_op(f"lrelu:{L.id}", list(L.out_vars + L.in_vars), alpha=alpha, idx_on=idx_on, idx_off=idx_off, idx_amb=idx_amb,
         slope=slope[idx_amb], shift=shift[idx_amb])
    
    return Fact(bounds=Bout, cons=cons)

@torch.no_grad()
def hybridz_tf_tanh(L: Layer, Bin: Bounds) -> Fact:
    lb = torch.tanh(Bin.lb)
    ub = torch.tanh(Bin.ub)

    lb2 = torch.minimum(lb, ub)
    ub2 = torch.maximum(lb, ub)

    Bout = Bounds(lb=lb2, ub=ub2)
    cons = ConSet()
    cons.add_op(f"tanh:{L.id}", list(L.out_vars + L.in_vars))

    return Fact(bounds=Bout, cons=cons)

@torch.no_grad()
def hybridz_tf_sigmoid(L: Layer, Bin: Bounds) -> Fact:
    lb = torch.sigmoid(Bin.lb)
    ub = torch.sigmoid(Bin.ub)
    lb2 = torch.minimum(lb, ub)
    ub2 = torch.maximum(lb, ub)

    Bout = Bounds(lb=lb2, ub=ub2)

    cons = ConSet()
    cons.add_op(f"sigmoid:{L.id}", list(L.out_vars + L.in_vars))
    return Fact(bounds=Bout, cons=cons)

@torch.no_grad()
def hybridz_tf_abs(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for absolute value."""
    # Determine phases
    idx_pos = torch.where(Bin.lb >= 0)[0]  # Always positive
    idx_neg = torch.where(Bin.ub <= 0)[0]  # Always negative
    crosses_zero = (Bin.lb < 0) & (Bin.ub > 0)  # Ambiguous: crosses zero
    idx_amb = torch.where(crosses_zero)[0]

    # Output bounds:
    # - if x >= 0: |x| = x, so lb = l, ub = u
    # - if x <= 0: |x| = -x, so lb = -u, ub = -l (min is at u, max is at l)
    # - if x crosses zero: lb = 0 (minimum is at x=0), ub = max(|l|, |u|)
    lb = torch.where(crosses_zero, torch.zeros_like(Bin.lb),
                     torch.where(Bin.lb >= 0, Bin.lb, -Bin.ub))
    ub = torch.maximum(torch.abs(Bin.lb), torch.abs(Bin.ub))
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_op(f"abs:{L.id}", list(L.out_vars + L.in_vars), idx_pos=idx_pos, idx_neg=idx_neg, idx_amb=idx_amb)

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_silu(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for SILU (Swish) activation: x * sigmoid(x)."""
    # SILU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    # Compute sigmoid bounds
    s_lb = 1 / (1 + torch.exp(-Bin.lb))
    s_ub = 1 / (1 + torch.exp(-Bin.ub))

    # SILU is x * sigmoid(x), use interval multiplication
    # Consider all combinations of x and sigmoid(x) bounds
    cand = torch.stack([
        Bin.lb * s_lb,
        Bin.lb * s_ub,
        Bin.ub * s_lb,
        Bin.ub * s_ub
    ], dim=0)

    lb = torch.min(cand, dim=0).values
    ub = torch.max(cand, dim=0).values
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"silu:{L.id}", list(L.out_vars + L.in_vars), s_lb=s_lb, s_ub=s_ub)

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_add(L: Layer, Bin1: Bounds, Bin2: Bounds) -> Fact:
    """HybridZ transfer function for element-wise addition."""
    # Simple interval addition
    lb = Bin1.lb + Bin2.lb
    ub = Bin1.ub + Bin2.ub
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_op(f"add:{L.id}", list(L.out_vars + L.in_vars),)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()  
def hybridz_tf_mul(L: Layer, Bin1: Bounds, Bin2: Bounds) -> Fact:
    """HybridZ transfer function for element-wise multiplication with McCormick relaxation."""
    # McCormick envelope for bilinear terms
    # z = x * y, with x ∈ [lx, ux], y ∈ [ly, uy]
    lx, ux = Bin1.lb, Bin1.ub
    ly, uy = Bin2.lb, Bin2.ub
    
    # Four corner points
    corners = torch.stack([
        lx * ly,  # lower-left
        lx * uy,  # lower-right
        ux * ly,  # upper-left
        ux * uy   # upper-right
    ])
    
    lb = torch.min(corners, dim=0)[0]
    ub = torch.max(corners, dim=0)[0]
    Bout = Bounds(lb=lb, ub=ub)
    
    # McCormick constraints
    cons = ConSet()
    cons.add_op(f"mcc:{L.id}", list(L.out_vars + L.in_vars), lx=lx, ux=ux, ly=ly, uy=uy)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_bn(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for batch normalization."""
    A = L.params["A"]  # Combined scale factor
    c = L.params["c"]  # Combined bias

    # BN is affine: y = A * x + c
    lb = torch.where(A >= 0, A * Bin.lb + c, A * Bin.ub + c)
    ub = torch.where(A >= 0, A * Bin.ub + c, A * Bin.lb + c)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"bn:{L.id}", list(L.out_vars + L.in_vars), A=A, c=c)

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_relu6(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for ReLU6: clamp(x, 0, 6)."""
    lb = torch.clamp(Bin.lb, min=0.0, max=6.0)
    ub = torch.clamp(Bin.ub, min=0.0, max=6.0)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"relu6:{L.id}", list(L.out_vars + L.in_vars))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_sub(L: Layer, Bin1: Bounds, Bin2: Bounds) -> Fact:
    """HybridZ transfer function for element-wise subtraction."""
    # [a,b] - [c,d] = [a-d, b-c]
    lb = Bin1.lb - Bin2.ub
    ub = Bin1.ub - Bin2.lb
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"sub:{L.id}", list(L.out_vars + L.in_vars))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_square(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for square: x^2."""
    # Clamp inputs to prevent numerical overflow when squaring large values
    # sqrt(1e150) ~ 1e75, so we clamp to 1e150 to keep squared values < 1e300 (within float64 range)
    MAX_VAL = 1e150
    l = torch.clamp(Bin.lb, min=-MAX_VAL, max=MAX_VAL)
    u = torch.clamp(Bin.ub, min=-MAX_VAL, max=MAX_VAL)

    # If interval crosses zero, lb = 0
    # Otherwise lb = min(l^2, u^2)
    crosses_zero = (l <= 0) & (u >= 0)
    lb = torch.where(crosses_zero, torch.zeros_like(l), torch.minimum(l * l, u * u))
    ub = torch.maximum(l * l, u * u)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"square:{L.id}", list(L.out_vars + L.in_vars))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_power(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for power: x^p."""
    p = float(L.meta.get("exponent", L.meta.get("p", 2.0)))

    # Clamp inputs to prevent numerical overflow when raising to power
    MAX_VAL = 1e150 ** (1.0/max(p, 1.0))  # Ensure result stays < 1e150
    l = torch.clamp(Bin.lb, min=0.0, max=MAX_VAL)
    u = torch.clamp(Bin.ub, min=0.0, max=MAX_VAL)

    # For positive base values
    f = lambda x: torch.pow(x, p)
    lb = f(l)
    ub = f(u)

    # Ensure lb <= ub
    lb2 = torch.minimum(lb, ub)
    ub2 = torch.maximum(lb, ub)
    Bout = Bounds(lb=lb2, ub=ub2)

    cons = ConSet()
    cons.add_op(f"power:{L.id}", list(L.out_vars + L.in_vars), p=p)

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_div(L: Layer, Bin1: Bounds, Bin2: Bounds) -> Fact:
    """HybridZ transfer function for element-wise division."""
    ly, uy = Bin2.lb, Bin2.ub

    # Check if denominator crosses zero
    crosses_zero = (ly <= 0) & (uy >= 0)

    # Compute corners for division
    cand = torch.stack([
        Bin1.lb / ly,
        Bin1.lb / uy,
        Bin1.ub / ly,
        Bin1.ub / uy,
    ], dim=0)

    lb = torch.min(cand, dim=0).values
    ub = torch.max(cand, dim=0).values

    # Conservative bounds when denominator crosses zero
    big = 1e6
    lb = torch.where(crosses_zero, torch.full_like(lb, -big), lb)
    ub = torch.where(crosses_zero, torch.full_like(ub, +big), ub)

    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"div:{L.id}", list(L.out_vars + L.in_vars), safe=not torch.any(crosses_zero).item())

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_matmul(L: Layer, Bin1: Bounds, Bin2: Bounds) -> Fact:
    """HybridZ transfer function for matrix multiplication."""
    x_shape = L.meta["x_shape"]      # (m, k)
    y_shape = L.meta["y_shape"]      # (k, n)

    m, k = x_shape
    k2, n = y_shape
    assert k == k2, "matmul: inner dim mismatch"

    # Reshape back to matrix form
    X_lb = Bin1.lb.view(m, k)
    X_ub = Bin1.ub.view(m, k)
    Y_lb = Bin2.lb.view(k, n)
    Y_ub = Bin2.ub.view(k, n)

    out_lb = []
    out_ub = []
    for i in range(m):
        for j in range(n):
            xs_lb = X_lb[i, :]
            xs_ub = X_ub[i, :]
            ys_lb = Y_lb[:, j]
            ys_ub = Y_ub[:, j]

            # Four corners for each element
            p1 = xs_lb * ys_lb
            p2 = xs_lb * ys_ub
            p3 = xs_ub * ys_lb
            p4 = xs_ub * ys_ub

            lo_ij = torch.min(torch.min(p1, p2), torch.min(p3, p4)).sum()
            hi_ij = torch.max(torch.max(p1, p2), torch.max(p3, p4)).sum()

            out_lb.append(lo_ij)
            out_ub.append(hi_ij)

    lb = torch.stack(out_lb, dim=0)
    ub = torch.stack(out_ub, dim=0)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"matmul:{L.id}", list(L.out_vars + L.in_vars), x_shape=x_shape, y_shape=y_shape)

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_max(L: Layer, By_list: list) -> Fact:
    """HybridZ transfer function for element-wise maximum."""
    lb = By_list[0].lb.clone()
    ub = By_list[0].ub.clone()

    for b in By_list[1:]:
        lb = torch.maximum(lb, b.lb)
        ub = torch.maximum(ub, b.ub)

    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"max:{L.id}", list(L.out_vars + L.in_vars), k=len(By_list))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_min(L: Layer, By_list: list) -> Fact:
    """HybridZ transfer function for element-wise minimum."""
    lb = By_list[0].lb.clone()
    ub = By_list[0].ub.clone()

    for b in By_list[1:]:
        lb = torch.minimum(lb, b.lb)
        ub = torch.minimum(ub, b.ub)

    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"min:{L.id}", list(L.out_vars + L.in_vars), k=len(By_list))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_pow(L: Layer, Bin1: Bounds, Bin2: Bounds) -> Fact:
    """HybridZ transfer function for element-wise power: x^y."""
    # Clamp base to prevent overflow: limit base to reasonable range
    MAX_BASE = 1e50  # Prevents overflow with typical exponents
    x_lb = torch.clamp(Bin1.lb, min=1e-8, max=MAX_BASE)
    x_ub = torch.clamp(Bin1.ub, min=1e-8, max=MAX_BASE)

    # Conservative interval arithmetic for x^y
    cand = torch.stack([
        torch.pow(x_lb, Bin2.lb),
        torch.pow(x_lb, Bin2.ub),
        torch.pow(x_ub, Bin2.lb),
        torch.pow(x_ub, Bin2.ub),
    ], dim=0)

    lb = torch.min(cand, dim=0).values
    ub = torch.max(cand, dim=0).values

    # Clamp result to finite values
    MAX_VAL = 1e150
    lb = torch.clamp(lb, min=-MAX_VAL, max=MAX_VAL)
    ub = torch.clamp(ub, min=-MAX_VAL, max=MAX_VAL)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"pow:{L.id}", list(L.out_vars + L.in_vars))

    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_concat(L: Layer, Bs: list) -> Fact:
    """HybridZ transfer function for concatenation."""
    lb = torch.cat([b.lb for b in Bs], dim=0)
    ub = torch.cat([b.ub for b in Bs], dim=0)
    Bout = Bounds(lb=lb, ub=ub)

    cons = ConSet()
    cons.add_op(f"concat:{L.id}", list(L.out_vars + L.in_vars))

    return Fact(bounds=Bout, cons=cons)