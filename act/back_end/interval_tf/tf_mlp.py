#===- act/back_end/interval_tf/tf_mlp.py - MLP Interval Transfer Func ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   MLP Interval Transfer Functions. Provides interval-based transfer functions
#   for multi-layer perceptron operations including linear layers and
#   activation functions.
#
#===---------------------------------------------------------------------===#

import torch
from typing import List
from act.back_end.core import Bounds, Con, ConSet, Fact, Layer
from act.back_end.utils import affine_bounds, pwl_meta, bound_var_interval, scale_interval

# -------- MLP Basics --------
def tf_dense(L: Layer, Bin: Bounds) -> Fact:
    # Handle parameter compatibility: schema defines W, but optimization uses W_pos/W_neg
    W = L.params["W"]
    W_pos = L.params.get("W_pos", torch.clamp(W, min=0))
    W_neg = L.params.get("W_neg", torch.clamp(W, max=0))
    b = L.params.get("b", torch.zeros(W.shape[0], device=W.device, dtype=W.dtype))
    
    B = affine_bounds(W_pos, W_neg, b, Bin)
    C = ConSet(); C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"dense:{L.id}", "W": W, "b": b}))
    C.add_box(L.id, L.out_vars, B); return Fact(B,C)

def tf_bias(L: Layer, Bin: Bounds) -> Fact:
    c=L.params["c"]; B=Bounds(Bin.lb+c, Bin.ub+c)
    C=ConSet(); C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"bias:{L.id}", "c": c}))
    C.add_box(L.id, L.out_vars, B); return Fact(B,C)

def tf_scale(L: Layer, Bin: Bounds) -> Fact:
    a=L.params["a"]
    lb=torch.where(a>=0, a*Bin.lb, a*Bin.ub); ub=torch.where(a>=0, a*Bin.ub, a*Bin.lb)
    B=Bounds(lb,ub); C=ConSet(); C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"scale:{L.id}", "a": a}))
    C.add_box(L.id, L.out_vars, B); return Fact(B,C)

def tf_relu(L: Layer, Bin: Bounds) -> Fact:
    l,u=Bin.lb,Bin.ub; on=l>=0; off=u<=0; amb=~(on|off)
    lb=torch.where(off,0.0,torch.where(on,l,0.0)); ub=torch.where(off,0.0,torch.where(on,u,u))
    if torch.any(amb):
        s=u[amb]/torch.clamp(u[amb]-l[amb],min=1e-12); t=-s*l[amb]
    else: s=t=torch.empty(0, dtype=l.dtype, device=l.device)
    B=Bounds(lb,ub); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"relu:{L.id}",
        "idx_on": torch.nonzero(on,as_tuple=True)[0],
        "idx_off": torch.nonzero(off,as_tuple=True)[0],
        "idx_amb": torch.nonzero(amb,as_tuple=True)[0],
        "slope": s, "shift": t}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_lrelu(L: Layer, Bin: Bounds) -> Fact:
    a=float(L.meta["alpha"]); l,u=Bin.lb,Bin.ub; on=l>=0; off=u<=0; amb=~(on|off)
    lb=torch.minimum(a*torch.minimum(l,0.0), torch.maximum(l,0.0))
    ub=torch.maximum(a*torch.maximum(u,0.0), torch.maximum(u,0.0))
    if torch.any(amb):
        s=(u[amb]-a*l[amb])/torch.clamp(u[amb]-l[amb],min=1e-12); t=a*l[amb]-s*l[amb]
    else: s=t=torch.empty(0, dtype=l.dtype, device=l.device)
    B=Bounds(lb,ub); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"lrelu:{L.id}","alpha":a,
        "idx_on": torch.nonzero(on,as_tuple=True)[0],
        "idx_off": torch.nonzero(off,as_tuple=True)[0],
        "idx_amb": torch.nonzero(amb,as_tuple=True)[0],
        "slope": s, "shift": t}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_abs(L: Layer, Bin: Bounds) -> Fact:
    l,u=Bin.lb,Bin.ub; pos=l>=0; neg=u<=0; amb=~(pos|neg)
    lb=torch.minimum(torch.zeros_like(l), torch.minimum(torch.abs(l), torch.abs(u)))
    ub=torch.maximum(torch.abs(l), torch.abs(u))
    B=Bounds(lb,ub); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"abs:{L.id}",
        "idx_pos": torch.nonzero(pos,as_tuple=True)[0],
        "idx_neg": torch.nonzero(neg,as_tuple=True)[0],
        "idx_amb": torch.nonzero(amb,as_tuple=True)[0]}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_clip(L: Layer, Bin: Bounds) -> Fact:
    a,b=L.params["a"],L.params["b"]; B=Bounds(torch.clamp(Bin.lb,a,b), torch.clamp(Bin.ub,a,b))
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"clip:{L.id}","a":a,"b":b}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_add(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    B=Bounds(Bx.lb+By.lb, Bx.ub+By.ub); C=ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.meta["x_vars"] + L.meta["y_vars"]), {"tag":f"add:{L.id}"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)
    
def tf_sub(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    B = Bounds(Bx.lb - By.lb, Bx.ub - By.ub)
    assert B.lb.numel() == len(L.out_vars), f"sub out_vars length {len(L.out_vars)} != output elements {B.lb.numel()}"
    assert torch.all(B.lb <= B.ub), "sub produced invalid bounds (lb > ub)"
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.meta["x_vars"] + L.meta["y_vars"]), {"tag": f"sub:{L.id}"}))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)

def tf_mul(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    cand=torch.stack([Bx.lb*By.lb, Bx.lb*By.ub, Bx.ub*By.lb, Bx.ub*By.ub], dim=0)
    B=Bounds(torch.min(cand,0).values, torch.max(cand,0).values); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.meta["x_vars"] + L.meta["y_vars"]),
        {"tag":f"mcc:{L.id}","lx":Bx.lb,"ux":Bx.ub,"ly":By.lb,"uy":By.ub}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)
    
def tf_div(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    ly, uy = By.lb, By.ub
    crosses_zero = (ly <= 0) & (uy >= 0)
    assert not torch.any(torch.isclose(ly, torch.zeros_like(ly)) & torch.isclose(uy, torch.zeros_like(uy))), "div denominator interval collapses to zero"
    cand = torch.stack([
        Bx.lb / ly,
        Bx.lb / uy,
        Bx.ub / ly,
        Bx.ub / uy,
    ], dim=0)
    lb = torch.min(cand, dim=0).values
    ub = torch.max(cand, dim=0).values
    big = 1e6
    lb = torch.where(crosses_zero, torch.full_like(lb, -big), lb)
    ub = torch.where(crosses_zero, torch.full_like(ub, +big), ub)
    B = Bounds(lb, ub)
    assert B.lb.numel() == len(L.out_vars), f"div out_vars length {len(L.out_vars)} != output elements {B.lb.numel()}"
    assert torch.all(torch.isfinite(B.lb) & torch.isfinite(B.ub)), "div produced non-finite bounds"
    assert torch.all(B.lb <= B.ub), "div produced invalid bounds (lb > ub)"
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.meta["x_vars"] + L.meta["y_vars"]), {"tag": f"div:{L.id}", "safe": not torch.any(crosses_zero).item()}))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)

def tf_matmul(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    x_shape = L.meta["x_shape"]      # (m, k)
    y_shape = L.meta["y_shape"]      # (k, n)
    out_shape = L.meta["output_shape"]  # (m, n)

    m, k = x_shape
    k2, n = y_shape
    assert k == k2, "matmul: inner dim mismatch"

    # Reshape back to matrix form
    X_lb = Bx.lb.view(m, k)
    X_ub = Bx.ub.view(m, k)
    Y_lb = By.lb.view(k, n)
    Y_ub = By.ub.view(k, n)

    out_lb = []
    out_ub = []
    for i in range(m):
        for j in range(n):
            # Collect the 4 corners of x[i, :] * y[:, j]
            xs_lb = X_lb[i, :]    # [k]
            xs_ub = X_ub[i, :]    # [k]
            ys_lb = Y_lb[:, j]    # [k]
            ys_ub = Y_ub[:, j]    # [k]

            p1 = xs_lb * ys_lb
            p2 = xs_lb * ys_ub
            p3 = xs_ub * ys_lb
            p4 = xs_ub * ys_ub

            lo_ij = torch.min(torch.min(p1, p2), torch.min(p3, p4)).sum()
            hi_ij = torch.max(torch.max(p1, p2), torch.max(p3, p4)).sum()

            out_lb.append(lo_ij)
            out_ub.append(hi_ij)

    out_lb = torch.stack(out_lb, dim=0)
    out_ub = torch.stack(out_ub, dim=0)

    B = Bounds(out_lb, out_ub)
    C = ConSet()

    C.replace(Con("INEQ",
                  tuple(L.out_vars + L.meta["x_vars"] + L.meta["y_vars"]),
                  {"tag": f"matmul:{L.id}",
                   "x_shape": x_shape,
                   "y_shape": y_shape,
                   "out_shape": out_shape}))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)

def tf_concat(L: Layer, Bs: List[Bounds]) -> Fact:
    B=Bounds(torch.cat([b.lb for b in Bs],0), torch.cat([b.ub for b in Bs],0))
    C=ConSet(); C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_bn(L: Layer, Bin: Bounds) -> Fact:
    A,c=L.params["A"],L.params["c"]
    lb=torch.where(A>=0, A*Bin.lb+c, A*Bin.ub+c); ub=torch.where(A>=0, A*Bin.ub+c, A*Bin.lb+c)
    B=Bounds(lb,ub); C=ConSet(); C.replace(Con("EQ", tuple(L.out_vars+L.in_vars), {"tag":f"bn:{L.id}","A":A,"c":c}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

# -------- Less-common MLP-ish --------
def tf_sigmoid(L: Layer, Bin: Bounds) -> Fact:
    f=lambda x: 1/(1+torch.exp(-x)); B=Bounds(f(Bin.lb), f(Bin.ub))
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"sigmoid:{L.id}","segs":pwl_meta(Bin.lb,Bin.ub,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_tanh(L: Layer, Bin: Bounds) -> Fact:
    B=Bounds(torch.tanh(Bin.lb), torch.tanh(Bin.ub))
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"tanh:{L.id}","segs":pwl_meta(Bin.lb,Bin.ub,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_softplus(L: Layer, Bin: Bounds) -> Fact:
    f=lambda x: torch.log1p(torch.exp(x)); B=Bounds(f(Bin.lb), f(Bin.ub))
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"softplus:{L.id}","segs":pwl_meta(Bin.lb,Bin.ub,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_silu(L: Layer, Bin: Bounds) -> Fact:
    s_lb, s_ub = 1/(1+torch.exp(-Bin.lb)), 1/(1+torch.exp(-Bin.ub))
    cand=torch.stack([Bin.lb*s_lb, Bin.lb*s_ub, Bin.ub*s_lb, Bin.ub*s_ub],0)
    B=Bounds(torch.min(cand,0).values, torch.max(cand,0).values)
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"silu:{L.id}","s_lb":s_lb,"s_ub":s_ub}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_max(L: Layer, By_list: List[Bounds]) -> Fact:
    lb=torch.maximum.reduce([b.lb for b in By_list]); ub=torch.maximum.reduce([b.ub for b in By_list])
    B=Bounds(lb,ub); all_y=sum((L.meta["y_vars_list"][i] for i in range(len(By_list))), [])
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+all_y), {"tag":f"max:{L.id}","k":len(By_list),"mode":"convex"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_min(L: Layer, By_list: List[Bounds]) -> Fact:
    lb=torch.minimum.reduce([b.lb for b in By_list]); ub=torch.minimum.reduce([b.ub for b in By_list])
    B=Bounds(lb,ub); all_y=sum((L.meta["y_vars_list"][i] for i in range(len(By_list))), [])
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+all_y), {"tag":f"min:{L.id}","k":len(By_list),"mode":"convex"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_square(L: Layer, Bin: Bounds) -> Fact:
    l,u=Bin.lb,Bin.ub
    lb=torch.where((l<=0)&(u>=0), 0.0, torch.minimum(l*l, u*u)); ub=torch.maximum(l*l, u*u)
    B=Bounds(lb,ub); C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"square:{L.id}","segs":pwl_meta(l,u,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_power(L: Layer, Bin: Bounds) -> Fact:
    p=float(L.meta["p"]); f=lambda x: torch.pow(torch.clamp(x,min=0.0), p)
    B=Bounds(f(Bin.lb), f(Bin.ub)); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"power:{L.id}","p":p,"segs":pwl_meta(Bin.lb,Bin.ub,2)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

# -------- Additional Activations --------
def tf_relu6(L: Layer, Bin: Bounds) -> Fact:
    """ReLU6: clamp(x, 0, 6)"""
    l, u = Bin.lb, Bin.ub
    lb = torch.clamp(l, min=0.0, max=6.0)
    ub = torch.clamp(u, min=0.0, max=6.0)
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"relu6:{L.id}"}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_hardtanh(L: Layer, Bin: Bounds) -> Fact:
    """HardTanh: clamp(x, min_val, max_val)"""
    min_val = float(L.meta.get("min_val", -1.0))
    max_val = float(L.meta.get("max_val", 1.0))
    l, u = Bin.lb, Bin.ub
    lb = torch.clamp(l, min=min_val, max=max_val)
    ub = torch.clamp(u, min=min_val, max=max_val)
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"hardtanh:{L.id}", "min_val": min_val, "max_val": max_val}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_hardsigmoid(L: Layer, Bin: Bounds) -> Fact:
    """HardSigmoid: clamp(alpha * x + beta, 0, 1)"""
    alpha = float(L.meta.get("alpha", 1/6))
    beta = float(L.meta.get("beta", 0.5))
    l, u = Bin.lb, Bin.ub
    # Apply linear transformation then clamp
    l_linear = alpha * l + beta
    u_linear = alpha * u + beta
    lb = torch.clamp(l_linear, min=0.0, max=1.0)
    ub = torch.clamp(u_linear, min=0.0, max=1.0)
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"hardsigmoid:{L.id}", "alpha": alpha, "beta": beta}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_hardswish(L: Layer, Bin: Bounds) -> Fact:
    """HardSwish: x * hardsigmoid(x)"""
    l, u = Bin.lb, Bin.ub
    # HardSwish bounds are complex, use conservative approximation
    lb = torch.where(l >= 3, l, torch.where(l <= -3, torch.zeros_like(l), torch.minimum(l, torch.zeros_like(l))))
    ub = torch.where(u >= 3, u, torch.where(u <= -3, torch.zeros_like(u), torch.maximum(u, torch.zeros_like(u))))
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"hardswish:{L.id}"}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_mish(L: Layer, Bin: Bounds) -> Fact:
    """Mish: x * tanh(softplus(x))"""
    l, u = Bin.lb, Bin.ub
    # Conservative bounds for Mish activation
    lb = torch.where(l >= 0, 0.0 * l, l)  # Negative values bounded by input
    ub = torch.where(u <= 0, 0.0 * u, u)  # Positive values bounded by input
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"mish:{L.id}"}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_softsign(L: Layer, Bin: Bounds) -> Fact:
    """SoftSign: x / (1 + |x|)"""
    l, u = Bin.lb, Bin.ub
    # SoftSign is bounded between -1 and 1
    lb = l / (1 + torch.abs(l))
    ub = u / (1 + torch.abs(u))
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {"tag": f"softsign:{L.id}"}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

# -------- Tensor Operations --------
def tf_reshape(L: Layer, Bin: Bounds) -> Fact:
    """Reshape: identity operation for bounds propagation"""
    # Reshape doesn't change the values, only the tensor shape
    B = Bounds(Bin.lb.clone(), Bin.ub.clone())
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"reshape:{L.id}", "target_shape": L.meta.get("target_shape")}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_transpose(L: Layer, Bin: Bounds) -> Fact:
    """Transpose: permute dimensions (identity for bounds)"""
    # Transpose doesn't change the values, only the dimension order
    B = Bounds(Bin.lb.clone(), Bin.ub.clone())
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"transpose:{L.id}", "perm": L.meta.get("perm")}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_squeeze(L: Layer, Bin: Bounds) -> Fact:
    """Squeeze: remove singleton dimensions (identity for bounds)"""
    B = Bounds(Bin.lb.clone(), Bin.ub.clone())
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"squeeze:{L.id}", "dims": L.meta.get("dims")}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_unsqueeze(L: Layer, Bin: Bounds) -> Fact:
    """Unsqueeze: add singleton dimensions (identity for bounds)"""
    B = Bounds(Bin.lb.clone(), Bin.ub.clone())
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"unsqueeze:{L.id}", "dims": L.meta.get("dims")}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_tile(L: Layer, Bin: Bounds) -> Fact:
    """Tile: repeat tensor along dimensions"""
    # Conservative bounds: same as input for each repetition
    repeats = L.meta.get("repeats")
    inp_shape = tuple(L.meta["input_shape"])
    x_lb = Bin.lb.view(*inp_shape)
    x_ub = Bin.ub.view(*inp_shape)
    out_lb = x_lb.repeat(*repeats)
    out_ub = x_ub.repeat(*repeats)
    B = Bounds(out_lb.reshape(-1), out_ub.reshape(-1))
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"tile:{L.id}", "repeats": repeats}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_expand(L: Layer, Bin: Bounds) -> Fact:
    """Expand: broadcast tensor to larger shape"""
    # Broadcasting doesn't change values, only shape
    B = Bounds(Bin.lb.clone(), Bin.ub.clone())
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"expand:{L.id}", "shape": L.meta.get("shape")}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)
    
def tf_slice(L: Layer, Bin: Bounds) -> Fact:
    inp_shape = tuple(L.meta["input_shape"])  # e.g. (1, 3, 32, 32)
    x_lb = Bin.lb.view(*inp_shape)
    x_ub = Bin.ub.view(*inp_shape)

    starts = L.meta.get("starts", [])
    ends   = L.meta.get("ends", [])
    axes   = L.meta.get("axes", list(range(len(inp_shape))))
    steps  = L.meta.get("steps", [1] * len(axes))

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
    assert out_lb.numel() == len(L.out_vars), f"slice out_vars length {len(L.out_vars)} != output elements {out_lb.numel()}"
    assert torch.all(out_lb <= out_ub), "slice produced invalid bounds (lb > ub)"

    B = Bounds(out_lb.reshape(-1), out_ub.reshape(-1))

    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"slice:{L.id}",
        "starts": starts,
        "ends": ends,
        "axes": axes,
        "steps": steps,
        "input_shape": inp_shape,
    }))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)


def tf_gather(L: Layer, Bin: Bounds) -> Fact:

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

    B = Bounds(out_lb.reshape(-1), out_ub.reshape(-1))

    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"gather:{L.id}",
        "axis": axis,
        "indices": indices.detach().cpu().tolist(),
        "input_shape": inp_shape,
        "output_shape": list(out_lb.shape),
    }))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)

def tf_index_select(L: Layer, Bin: Bounds) -> Fact:

    inp_shape = tuple(L.meta["input_shape"])
    dim = int(L.meta["dim"])
    assert 0 <= dim < len(inp_shape), f"index_select dim {dim} out of range for input shape {inp_shape}"
    x_lb = Bin.lb.view(*inp_shape)
    x_ub = Bin.ub.view(*inp_shape)

    raw_idx = L.meta["indices"]
    if isinstance(raw_idx, (list, tuple)):
        indices = torch.tensor(raw_idx, dtype=torch.long, device=x_lb.device)
    else:
        indices = raw_idx.to(x_lb.device).long()
    assert indices.numel() > 0, "index_select received empty indices"

    out_lb = torch.index_select(x_lb, dim=dim, index=indices)
    out_ub = torch.index_select(x_ub, dim=dim, index=indices)
    assert out_lb.numel() == len(L.out_vars), f"index_select out_vars length {len(L.out_vars)} != output elements {out_lb.numel()}"
    assert torch.all(out_lb <= out_ub), "index_select produced invalid bounds (lb > ub)"

    B = Bounds(out_lb.reshape(-1), out_ub.reshape(-1))

    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"index_select:{L.id}",
        "dim": dim,
        "indices": indices.detach().cpu().tolist(),
        "input_shape": inp_shape,
        "output_shape": list(out_lb.shape),
    }))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)

def tf_permute(L, ctx):
    (lx, ux) = ctx.get_predecessor_bounds(L.id, 0)
    perm = L.meta["perm"]    
    assert len(perm) == lx.dim(), f"permute length {len(perm)} != tensor dim {lx.dim()}"
    lx = lx.permute(*perm)
    ux = ux.permute(*perm)
    assert lx.shape == ux.shape, "permute produced mismatched bound shapes"
    return lx, ux

def tf_reorder(L, ctx):
    (lx, ux) = ctx.get_predecessor_bounds(L.id, 0)
    raw_order = L.meta["order"]
    order = raw_order if torch.is_tensor(raw_order) else torch.tensor(raw_order, device=lx.device, dtype=torch.long)
    dim = L.meta.get("dim", 0)
    assert order.numel() == lx.shape[dim], f"reorder order length {order.numel()} != dim size {lx.shape[dim]}"
    lx = lx.index_select(L.meta.get("dim", 0), order)
    ux = ux.index_select(L.meta.get("dim", 0), order)
    assert lx.shape == ux.shape, "reorder produced mismatched bound shapes"
    return lx, ux

def tf_scale_shift(L, ctx):
    (lx, ux) = ctx.get_predecessor_bounds(L.id, 0)
    s = L.params["scale"]
    b = L.params.get("shift", 0.)
    l2 = lx * s + b
    u2 = ux * s + b
    lo = torch.minimum(l2, u2)
    hi = torch.maximum(l2, u2)
    assert lo.shape == hi.shape, "scale_shift produced mismatched bound shapes"
    assert torch.all(lo <= hi), "scale_shift produced invalid bounds (lb > ub)"
    return lo, hi

def tf_stack(L, ctx):
    lbs, ubs = [], []
    for i in range(len(L.inputs)):
        lb, ub = ctx.get_predecessor_bounds(L.id, i)
        lbs.append(lb); ubs.append(ub)
    dim = L.meta.get("axis", 0)
    stacked_lb = torch.stack(lbs, dim=dim)
    stacked_ub = torch.stack(ubs, dim=dim)
    assert stacked_lb.shape == stacked_ub.shape, "stack produced mismatched bound shapes"
    assert torch.all(stacked_lb <= stacked_ub), "stack produced invalid bounds (lb > ub)"
    return stacked_lb, stacked_ub
