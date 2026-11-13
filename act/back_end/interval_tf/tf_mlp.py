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
import math
from typing import Any
from collections.abc import Mapping
from act.back_end.core import Bounds, Con, ConSet, Fact, Layer
from act.back_end.layer_schema import step_function
from act.back_end.utils import affine_bounds, four_corner_envelope, pwl_meta


def _sanitize_smooth_bounds(lb: torch.Tensor, ub: torch.Tensor) -> Bounds:
    lb_s = torch.nan_to_num(lb, nan=-float("inf"), neginf=-float("inf"), posinf=float("inf"))
    ub_s = torch.nan_to_num(ub, nan=float("inf"), neginf=-float("inf"), posinf=float("inf"))
    return Bounds(torch.minimum(lb_s, ub_s), torch.maximum(lb_s, ub_s))


def _periodic_critical_exists(lo: torch.Tensor, hi: torch.Tensor, offset: float, period: float) -> torch.Tensor:
    eps = torch.finfo(lo.dtype).eps * 16.0
    a = (lo - offset) / period - eps
    b = (hi - offset) / period + eps
    return torch.floor(b) >= torch.ceil(a)


def _quantize_params_flat(params: Mapping[str, Any], error_label: str, n: int, device: torch.device, dtype: torch.dtype):
    scale = params["scale"].to(device=device, dtype=dtype).flatten()
    zero_point = params["zero_point"].to(device=device, dtype=dtype).flatten()
    if scale.numel() == 1:
        scale = scale.expand(n)
    elif scale.numel() != n:
        raise NotImplementedError(f"{error_label}: scale with {scale.numel()} entries cannot broadcast to flat size {n}")
    if zero_point.numel() == 1:
        zero_point = zero_point.expand(n)
    elif zero_point.numel() != n:
        raise NotImplementedError(f"{error_label}: zero_point with {zero_point.numel()} entries cannot broadcast to flat size {n}")
    if torch.any(scale <= 0):
        raise ValueError(f"{error_label}: scale must be positive")
    qmin = torch.full((n,), float(params["qmin"]), device=device, dtype=dtype)
    qmax = torch.full((n,), float(params["qmax"]), device=device, dtype=dtype)
    return scale, zero_point, qmin, qmax


def _quantize_qdq_value(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, qmin: torch.Tensor, qmax: torch.Tensor) -> torch.Tensor:
    lo_code = qmin - zero_point
    hi_code = qmax - zero_point
    return scale * torch.clamp(torch.round(x / scale), min=lo_code, max=hi_code)


def _periodic_interval(fn, lo: torch.Tensor, hi: torch.Tensor, max_offset: float, min_offset: float) -> Bounds:
    """Interval image of a 2*pi-periodic function with unit amplitude (sin/cos)."""
    two_pi = 2.0 * math.pi
    finite = torch.isfinite(lo) & torch.isfinite(hi) & (torch.abs(lo) <= 1.0e12) & (torch.abs(hi) <= 1.0e12)
    hi = torch.maximum(hi, lo)
    width = hi - lo
    endpoint_lo = fn(lo)
    endpoint_hi = fn(hi)
    base_lb = torch.minimum(endpoint_lo, endpoint_hi)
    base_ub = torch.maximum(endpoint_lo, endpoint_hi)
    full_period = width >= two_pi
    has_max = _periodic_critical_exists(lo, hi, max_offset, two_pi)
    has_min = _periodic_critical_exists(lo, hi, min_offset, two_pi)
    lb = torch.where(has_min | full_period, torch.full_like(base_lb, -1.0), base_lb)
    ub = torch.where(has_max | full_period, torch.full_like(base_ub, 1.0), base_ub)
    lb = torch.where(finite, lb, torch.full_like(lb, -1.0))
    ub = torch.where(finite, ub, torch.full_like(ub, 1.0))
    return Bounds(lb, ub)


def _sin_interval(lo: torch.Tensor, hi: torch.Tensor) -> Bounds:
    return _periodic_interval(torch.sin, lo, hi, 0.5 * math.pi, -0.5 * math.pi)


def _cos_interval(lo: torch.Tensor, hi: torch.Tensor) -> Bounds:
    return _periodic_interval(torch.cos, lo, hi, 0.0, math.pi)


def _fact(L: Layer, B: Bounds, con_kind: str, meta: dict[str, Any]) -> Fact:
    """Fact of B with one con_kind constraint over out_vars + in_vars, then the box."""
    C = ConSet()
    C.replace(Con(con_kind, tuple(L.out_vars + L.in_vars), meta))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)

# -------- MLP Basics --------
def tf_dense(L: Layer, Bin: Bounds) -> Fact:
    # Parameter names aligned with PyTorch: weight, bias, weight_pos, weight_neg
    W = L.params["weight"]
    if L.params.get("token_wise", False):
        assert isinstance(W, torch.Tensor)
        weight = W.to(device=Bin.lb.device, dtype=Bin.lb.dtype)
        in_features = weight.shape[1]
        if len(L.in_vars) % in_features:
            raise ValueError(
                f"token-wise DENSE {L.id}: {len(L.in_vars)} inputs are not divisible "
                f"by in_features={in_features}"
            )
        tokens = len(L.in_vars) // in_features
        batch = Bin.lb.shape[0]
        bias = L.params.get("bias")
        if bias is None:
            bias = weight.new_zeros(weight.shape[0])
        else:
            assert isinstance(bias, torch.Tensor)
            bias = bias.to(device=Bin.lb.device, dtype=Bin.lb.dtype)
        # One affine map per token; affine_bounds keeps 0 * inf = 0 for unbounded inputs.
        token_box = affine_bounds(
            weight.clamp(min=0), weight.clamp(max=0), bias,
            Bounds(Bin.lb.reshape(batch * tokens, in_features), Bin.ub.reshape(batch * tokens, in_features)),
        )
        box = Bounds(token_box.lb.reshape(batch, -1), token_box.ub.reshape(batch, -1))
        return _fact(L, box, "EQ", {
            "tag": f"dense:{L.id}", "W": weight, "b": bias,
            "token_wise": True, "token_count": tokens,
        })
    W_pos = L.params.get("weight_pos", torch.clamp(W, min=0))
    W_neg = L.params.get("weight_neg", torch.clamp(W, max=0))
    b = L.params.get("bias", torch.zeros(W.shape[0]))
    
    B = affine_bounds(W_pos, W_neg, b, Bin)
    return _fact(L, B, "EQ", {"tag": f"dense:{L.id}", "W": W, "b": b})

def tf_bias(L: Layer, Bin: Bounds) -> Fact:
    c=L.params["c"]
    B=Bounds(Bin.lb+c, Bin.ub+c)
    return _fact(L, B, "EQ", {"tag": f"bias:{L.id}", "c": c})

def tf_scale(L: Layer, Bin: Bounds) -> Fact:
    # Bounds are [B, n], but SCALE parameters can retain spatial axes.
    a=L.params["a"].reshape(-1)
    lb=torch.where(a>=0, a*Bin.lb, a*Bin.ub); ub=torch.where(a>=0, a*Bin.ub, a*Bin.lb)
    return _fact(L, Bounds(lb,ub), "EQ", {"tag": f"scale:{L.id}", "a": a})

def tf_relu(L: Layer, Bin: Bounds) -> Fact:
    l,u=Bin.lb,Bin.ub; on=l>=0; off=u<=0; amb=~(on|off)
    lb=torch.where(off,0.0,torch.where(on,l,0.0)); ub=torch.where(off,0.0,torch.where(on,u,u))
    if torch.any(amb):
        ua,la=u[amb],l[amb]; gap=ua-la
        finite=torch.isfinite(gap) & (gap>1e-12)
        s=torch.where(finite, ua/torch.clamp(gap,min=1e-12), torch.ones_like(gap))
        t=torch.where(finite, -s*la, torch.zeros_like(gap))
    else: s=t=torch.empty(0)
    return _fact(L, Bounds(lb,ub), "INEQ", {"tag":f"relu:{L.id}",
        "idx_on": torch.nonzero(on,as_tuple=True)[0],
        "idx_off": torch.nonzero(off,as_tuple=True)[0],
        "idx_amb": torch.nonzero(amb,as_tuple=True)[0],
        "slope": s, "shift": t})

def tf_lrelu(L: Layer, Bin: Bounds) -> Fact:
    a=float(L.params["negative_slope"])
    l,u=Bin.lb,Bin.ub; on=l>=0; off=u<=0; amb=~(on|off)
    z=torch.zeros_like(l)
    lb=torch.minimum(a*torch.minimum(l,z), torch.maximum(l,z))
    ub=torch.maximum(a*torch.maximum(u,z), torch.maximum(u,z))
    if torch.any(amb):
        la,ua=l[amb],u[amb]; gap=ua-la
        # Guard against inf-inf=NaN: when gap is not finite, use slope=max(a,1)
        finite=torch.isfinite(gap) & (gap>1e-12)
        s=torch.where(finite, (ua-a*la)/torch.clamp(gap,min=1e-12), torch.full_like(gap,max(a,1.0)))
        t=torch.where(finite, a*la-s*la, torch.zeros_like(gap))
    else: s=t=torch.empty(0)
    return _fact(L, Bounds(lb,ub), "INEQ", {"tag":f"lrelu:{L.id}","alpha":a,
        "idx_on": torch.nonzero(on,as_tuple=True)[0],
        "idx_off": torch.nonzero(off,as_tuple=True)[0],
        "idx_amb": torch.nonzero(amb,as_tuple=True)[0],
        "slope": s, "shift": t})

def tf_abs(L: Layer, Bin: Bounds) -> Fact:
    l,u=Bin.lb,Bin.ub; pos=l>=0; neg=u<=0; amb=~(pos|neg)
    lb=torch.minimum(torch.zeros_like(l), torch.minimum(torch.abs(l), torch.abs(u)))
    ub=torch.maximum(torch.abs(l), torch.abs(u))
    return _fact(L, Bounds(lb,ub), "INEQ", {"tag":f"abs:{L.id}",
        "idx_pos": torch.nonzero(pos,as_tuple=True)[0],
        "idx_neg": torch.nonzero(neg,as_tuple=True)[0],
        "idx_amb": torch.nonzero(amb,as_tuple=True)[0]})

def tf_clip(L: Layer, Bin: Bounds) -> Fact:
    a,b=L.params["a"],L.params["b"]; B=Bounds(torch.clamp(Bin.lb,a,b), torch.clamp(Bin.ub,a,b))
    return _fact(L, B, "INEQ", {"tag":f"clip:{L.id}","a":a,"b":b})

def tf_add(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    B=Bounds(Bx.lb+By.lb, Bx.ub+By.ub); C=ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.params["x_vars"] + L.params["y_vars"]), {"tag":f"add:{L.id}"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)
    
def tf_sub(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    B = Bounds(Bx.lb - By.ub, Bx.ub - By.lb)
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.params["x_vars"] + L.params["y_vars"]), {"tag": f"sub:{L.id}"}))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)

def tf_mul(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    lb,ub=four_corner_envelope(Bx.lb, Bx.ub, By.lb, By.ub)
    B=Bounds(lb,ub); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.params["x_vars"] + L.params["y_vars"]),
        {"tag":f"mcc:{L.id}","lx":Bx.lb,"ux":Bx.ub,"ly":By.lb,"uy":By.ub}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)
    
def tf_div(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    ly, uy = By.lb, By.ub
    crosses_zero = (ly <= 0) & (uy >= 0)
    cand = torch.stack([
        Bx.lb / ly,
        Bx.lb / uy,
        Bx.ub / ly,
        Bx.ub / uy,
    ], dim=0)
    lb = torch.min(cand, dim=0).values
    ub = torch.max(cand, dim=0).values
    lb = torch.where(crosses_zero, torch.full_like(lb, -float("inf")), lb)
    ub = torch.where(crosses_zero, torch.full_like(ub, float("inf")), ub)
    B = Bounds(lb, ub)
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.params["x_vars"] + L.params["y_vars"]), {"tag": f"div:{L.id}", "safe": not torch.any(crosses_zero).item()}))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)

def tf_matmul(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    batch_size = Bx.lb.shape[0]
    x_shape = tuple(L.params["x_shape"])
    y_shape = tuple(L.params["y_shape"])
    A_lb = Bx.lb.view(batch_size, *x_shape).unsqueeze(-1)
    A_ub = Bx.ub.view(batch_size, *x_shape).unsqueeze(-1)
    B_lb = By.lb.view(batch_size, *y_shape).unsqueeze(-3)
    B_ub = By.ub.view(batch_size, *y_shape).unsqueeze(-3)
    lo, hi = four_corner_envelope(A_lb, A_ub, B_lb, B_ub, sum_axis=-2)
    out_lb = lo.reshape(batch_size, -1)
    out_ub = hi.reshape(batch_size, -1)
    Bres = Bounds(out_lb, out_ub)
    C = ConSet(); C.add_box(L.id, L.out_vars, Bres)
    return Fact(Bres, C)


def tf_arg_extremum(L: Layer, Bin: Bounds) -> Fact:
    batch_size = Bin.lb.shape[0]
    in_shape = L.params.get("input_shape")
    axis = int(L.params.get("axis", 0))
    if in_shape is not None and axis < 0:
        axis += len(in_shape)
    axis_dim = int(in_shape[axis]) if in_shape else 1
    n_out = len(L.out_vars)
    lb = Bin.lb.new_zeros(batch_size, n_out)
    ub = Bin.lb.new_full((batch_size, n_out), float(max(0, axis_dim - 1)))
    Bout = Bounds(lb, ub)
    C = ConSet(); C.add_box(L.id, L.out_vars, Bout)
    return Fact(Bout, C)


def tf_scatter_nd(L: Layer, Bdata: Bounds, Bidx: Bounds, Bupdates: Bounds) -> Fact:
    batch_size = Bdata.lb.shape[0]
    n = len(L.out_vars)
    data_lb = Bdata.lb.view(batch_size, -1)
    data_ub = Bdata.ub.view(batch_size, -1)
    if data_lb.shape[1] != n:
        repeat = (n + data_lb.shape[1] - 1) // data_lb.shape[1]
        data_lb = data_lb[:, :n] if data_lb.shape[1] > n else data_lb.repeat(1, repeat)[:, :n]
        data_ub = data_ub[:, :n] if data_ub.shape[1] > n else data_ub.repeat(1, repeat)[:, :n]
    if Bupdates.lb.numel() > 0:
        updates_lb = Bupdates.lb.view(batch_size, -1)
        updates_ub = Bupdates.ub.view(batch_size, -1)
        u_min = updates_lb.min(dim=1, keepdim=True).values.expand(batch_size, n)
        u_max = updates_ub.max(dim=1, keepdim=True).values.expand(batch_size, n)
        lb = torch.minimum(data_lb, u_min)
        ub = torch.maximum(data_ub, u_max)
    else:
        lb, ub = data_lb, data_ub
    Bout = Bounds(lb, ub)
    C = ConSet(); C.add_box(L.id, L.out_vars, Bout)
    return Fact(Bout, C)

def tf_concat(L: Layer, Bs: list[Bounds]) -> Fact:
    """Concatenates tensors on dim=-1 (feature axis)."""
    B=Bounds(torch.cat([b.lb for b in Bs], dim=-1), torch.cat([b.ub for b in Bs], dim=-1))
    C=ConSet(); C.add_box(L.id,L.out_vars,B)
    # Only axis 1 is flat append order in every operand layout.
    if L.params.get("concat_dim") == 1 and len(L.in_vars) == len(L.out_vars):
        C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {"tag": f"concat:{L.id}"}))
    return Fact(B,C)

def tf_constant(L: Layer, Bin: Bounds) -> Fact:
    # CONSTANT bounds come from L.params["value"], not from Bin. The graph
    # may carry spurious predecessor edges (e.g. ONNX initializers wired to
    # topological ancestors) whose Bounds have unrelated shape; ignoring
    # Bin keeps the output shaped to L.out_vars regardless of routing.
    B_size = Bin.lb.shape[0]
    val = L.params["value"].reshape(-1).to(Bin.lb)
    val_b = val.unsqueeze(0).expand(B_size, -1).contiguous()
    Bout = Bounds(val_b.clone(), val_b.clone())
    C = ConSet(); C.add_box(L.id, L.out_vars, Bout)
    C.replace(Con("EQ", tuple(L.out_vars), {"tag": f"constant:{L.id}", "value": val}))
    return Fact(Bout, C)

def tf_step(L: Layer, Bin: Bounds) -> Fact:
    """SIGN / ROUND: monotone non-decreasing steps, so the box is exactly [f(lb), f(ub)]."""
    f = step_function(L.kind, L.params)
    B = Bounds(f(Bin.lb), f(Bin.ub))
    C = ConSet(); C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)

def _bcast(b: torch.Tensor, B: int, n: int) -> torch.Tensor:
    if b.dim() == 0:
        return b.reshape(1, 1).expand(B, n)
    if b.dim() == 1:
        if b.numel() == 1:
            return b.reshape(1, 1).expand(B, n)
        if b.numel() == n:
            return b.reshape(1, n).expand(B, n)
    if b.dim() == 2:
        if b.shape == (B, n):
            return b
        if b.shape == (1, n):
            return b.expand(B, n)
        if b.shape == (B, 1):
            return b.expand(B, n)
        if b.shape == (1, 1):
            return b.expand(B, n)
    raise ValueError(f"COMPARE/WHERE bcast: cannot align shape {tuple(b.shape)} -> {(B, n)}")


def tf_compare(L: Layer, Bx: Bounds, By: Bounds) -> Fact:
    op = L.params["op"]
    batch_size = Bx.lb.shape[0]
    n = len(L.out_vars)
    lb_x, ub_x = _bcast(Bx.lb, batch_size, n), _bcast(Bx.ub, batch_size, n)
    lb_y, ub_y = _bcast(By.lb, batch_size, n), _bcast(By.ub, batch_size, n)
    if op == "lt":
        defin_t, defin_f = ub_x < lb_y, lb_x >= ub_y
    elif op == "le":
        defin_t, defin_f = ub_x <= lb_y, lb_x > ub_y
    elif op == "gt":
        defin_t, defin_f = lb_x > ub_y, ub_x <= lb_y
    elif op == "ge":
        defin_t, defin_f = lb_x >= ub_y, ub_x < lb_y
    elif op == "eq":
        is_pt = (lb_x == ub_x) & (lb_y == ub_y)
        defin_t = is_pt & (lb_x == lb_y)
        defin_f = (ub_x < lb_y) | (lb_x > ub_y)
    elif op == "ne":
        is_pt = (lb_x == ub_x) & (lb_y == ub_y)
        defin_t = (ub_x < lb_y) | (lb_x > ub_y)
        defin_f = is_pt & (lb_x == lb_y)
    else:
        raise ValueError(f"COMPARE: unknown op '{op}'")
    z, o = torch.zeros_like(lb_x), torch.ones_like(lb_x)
    lb = torch.where(defin_t, o, z)
    ub = torch.where(defin_f, z, o)
    Bout = Bounds(lb, ub); C = ConSet(); C.add_box(L.id, L.out_vars, Bout)
    return Fact(Bout, C)


def tf_where(L: Layer, Bcond: Bounds, Bx: Bounds, By: Bounds) -> Fact:
    batch_size = Bcond.lb.shape[0]
    n = len(L.out_vars)
    cond_lb, cond_ub = _bcast(Bcond.lb, batch_size, n), _bcast(Bcond.ub, batch_size, n)
    lb_x, ub_x = _bcast(Bx.lb, batch_size, n), _bcast(Bx.ub, batch_size, n)
    lb_y, ub_y = _bcast(By.lb, batch_size, n), _bcast(By.ub, batch_size, n)
    cond_true = cond_lb >= 0.5
    cond_false = cond_ub < 0.5
    lb = torch.where(cond_true, lb_x, torch.where(cond_false, lb_y, torch.minimum(lb_x, lb_y)))
    ub = torch.where(cond_true, ub_x, torch.where(cond_false, ub_y, torch.maximum(ub_x, ub_y)))
    Bout = Bounds(lb, ub); C = ConSet(); C.add_box(L.id, L.out_vars, Bout)
    return Fact(Bout, C)


def tf_reduce_sum(L: Layer, Bin: Bounds) -> Fact:
    batch_size = Bin.lb.shape[0]
    axes = L.params.get("axes")
    keepdims = bool(L.params.get("keepdims", 0))
    in_shape = L.params.get("input_shape")
    lb_in, ub_in = Bin.lb, Bin.ub
    if in_shape is not None and len(in_shape) > 0:
        lb_in = lb_in.view(batch_size, *in_shape)
        ub_in = ub_in.view(batch_size, *in_shape)
    dim = tuple(int(a) + 1 for a in axes) if axes else tuple(range(1, lb_in.dim()))
    lb_out = lb_in.sum(dim=dim, keepdim=keepdims)
    ub_out = ub_in.sum(dim=dim, keepdim=keepdims)
    Bout = Bounds(lb_out.reshape(batch_size, -1), ub_out.reshape(batch_size, -1))
    C = ConSet(); C.add_box(L.id, L.out_vars, Bout)
    # The concrete operator numbers axes with the batch, and the box without it;
    # they name the same axes only under a singleton leading input dimension.
    if axes and in_shape is not None and in_shape[0] == 1 and all(int(a) > 0 for a in axes):
        C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
            "tag": f"reduce_sum:{L.id}", "input_shape": tuple(in_shape), "axes": tuple(int(a) for a in axes),
        }))
    return Fact(Bout, C)

def tf_mean(L: Layer, Bin: Bounds) -> Fact:
    batch_size = Bin.lb.shape[0]
    dims = L.params.get("dim")
    keepdim = bool(L.params.get("keepdim", 0))
    in_shape = L.params.get("input_shape")
    lb_in, ub_in = Bin.lb, Bin.ub
    if in_shape is not None and len(in_shape) > 0:
        lb_in = lb_in.view(batch_size, *in_shape)
        ub_in = ub_in.view(batch_size, *in_shape)
    dim = tuple(int(a) + 1 for a in dims) if dims else tuple(range(1, lb_in.dim()))
    lb_out = lb_in.mean(dim=dim, keepdim=keepdim)
    ub_out = ub_in.mean(dim=dim, keepdim=keepdim)
    Bout = Bounds(lb_out.reshape(batch_size, -1), ub_out.reshape(batch_size, -1))
    C = ConSet(); C.add_box(L.id, L.out_vars, Bout)
    # A singleton leading input dimension may be the concrete batch axis.
    if dims and in_shape is not None and all(int(a) >= (in_shape[0] == 1) for a in dims):
        C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
            "tag": f"mean:{L.id}", "input_shape": tuple(in_shape), "axes": tuple(int(a) for a in dims),
        }))
    return Fact(Bout, C)

def tf_bn(L: Layer, Bin: Bounds) -> Fact:
    A,c=L.params["A"],L.params["c"]
    lb=torch.where(A>=0, A*Bin.lb+c, A*Bin.ub+c); ub=torch.where(A>=0, A*Bin.ub+c, A*Bin.lb+c)
    return _fact(L, Bounds(lb,ub), "EQ", {"tag":f"bn:{L.id}","A":A,"c":c})

# -------- Less-common MLP-ish --------
def _monotone_pwl(name: str, f):
    """Build a TF for an increasing activation f: endpoint bounds plus a 2-segment PWL INEQ."""
    def tf(L: Layer, Bin: Bounds) -> Fact:
        B=Bounds(f(Bin.lb), f(Bin.ub))
        return _fact(L, B, "INEQ", {"tag":f"{name}:{L.id}","segs":pwl_meta(Bin.lb,Bin.ub,2)})
    return tf

tf_sigmoid = _monotone_pwl("sigmoid", lambda x: 1/(1+torch.exp(-x)))
tf_tanh = _monotone_pwl("tanh", torch.tanh)
tf_erf = _monotone_pwl("erf", torch.erf)
tf_softplus = _monotone_pwl("softplus", lambda x: torch.log1p(torch.exp(x)))

def tf_sqrt(L: Layer, Bin: Bounds) -> Fact:
    lo_e = torch.clamp(Bin.lb, min=0.0)
    hi_e = torch.clamp(Bin.ub, min=0.0)
    B=_sanitize_smooth_bounds(torch.sqrt(lo_e), torch.sqrt(hi_e))
    return _fact(L, B, "INEQ", {"tag":f"sqrt:{L.id}"})

def tf_sin(L: Layer, Bin: Bounds) -> Fact:
    B=_sin_interval(Bin.lb, Bin.ub)
    return _fact(L, B, "INEQ", {"tag":f"sin:{L.id}"})

def tf_cos(L: Layer, Bin: Bounds) -> Fact:
    B=_cos_interval(Bin.lb, Bin.ub)
    return _fact(L, B, "INEQ", {"tag":f"cos:{L.id}"})

def tf_quantize(L: Layer, Bin: Bounds) -> Fact:
    n = Bin.lb.numel()
    lo = Bin.lb.reshape(-1)
    hi = torch.maximum(Bin.ub.reshape(-1), lo)
    scale, zero_point, qmin, qmax = _quantize_params_flat(L.params, f"quantize:{L.id}", n, lo.device, lo.dtype)
    z_lo = _quantize_qdq_value(lo, scale, zero_point, qmin, qmax)
    z_hi = _quantize_qdq_value(hi, scale, zero_point, qmin, qmax)
    B=_sanitize_smooth_bounds(torch.minimum(z_lo, z_hi).reshape_as(Bin.lb), torch.maximum(z_lo, z_hi).reshape_as(Bin.ub))
    return _fact(L, B, "INEQ", {"tag":f"quantize:{L.id}", "scale": L.params["scale"], "zero_point": L.params["zero_point"], "qmin": L.params["qmin"], "qmax": L.params["qmax"]})


def tf_silu(L: Layer, Bin: Bounds) -> Fact:
    s_lb, s_ub = 1/(1+torch.exp(-Bin.lb)), 1/(1+torch.exp(-Bin.ub))
    lb,ub=four_corner_envelope(Bin.lb, Bin.ub, s_lb, s_ub)
    return _fact(L, Bounds(lb,ub), "INEQ", {"tag":f"silu:{L.id}","s_lb":s_lb,"s_ub":s_ub})

def tf_max(L: Layer, By_list: list[Bounds]) -> Fact:
    lb = torch.stack([b.lb for b in By_list], dim=0).amax(dim=0)
    ub = torch.stack([b.ub for b in By_list], dim=0).amax(dim=0)
    B=Bounds(lb,ub)
    # Flatten list of y_vars_list efficiently
    all_y = [v for sub in L.params.get("y_vars_list", []) for v in sub]
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+all_y), {"tag":f"max:{L.id}","k":len(By_list),"mode":"convex"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_min(L: Layer, By_list: list[Bounds]) -> Fact:
    lb = torch.stack([b.lb for b in By_list], dim=0).amin(dim=0)
    ub = torch.stack([b.ub for b in By_list], dim=0).amin(dim=0)
    B=Bounds(lb,ub)
    # Flatten list of y_vars_list efficiently
    all_y = [v for sub in L.params.get("y_vars_list", []) for v in sub]
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+all_y), {"tag":f"min:{L.id}","k":len(By_list),"mode":"convex"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_square(L: Layer, Bin: Bounds) -> Fact:
    l,u=Bin.lb,Bin.ub
    lb=torch.where((l<=0)&(u>=0), 0.0, torch.minimum(l*l, u*u)); ub=torch.maximum(l*l, u*u)
    return _fact(L, Bounds(lb,ub), "INEQ", {"tag":f"square:{L.id}","segs":pwl_meta(l,u,2)})

def tf_power(L: Layer, Bin: Bounds) -> Fact:
    p=float(L.params["p"]); f=lambda x: torch.pow(torch.clamp(x,min=0.0), p)
    B=Bounds(f(Bin.lb), f(Bin.ub))
    return _fact(L, B, "INEQ", {"tag":f"power:{L.id}","p":p,"segs":pwl_meta(Bin.lb,Bin.ub,2)})

# -------- Additional Activations --------
def tf_relu6(L: Layer, Bin: Bounds) -> Fact:
    """ReLU6: clamp(x, 0, 6)"""
    l, u = Bin.lb, Bin.ub
    lb = torch.clamp(l, min=0.0, max=6.0)
    ub = torch.clamp(u, min=0.0, max=6.0)
    return _fact(L, Bounds(lb, ub), "INEQ", {"tag": f"relu6:{L.id}"})

def tf_hardtanh(L: Layer, Bin: Bounds) -> Fact:
    """HardTanh: clamp(x, min_val, max_val)"""
    min_val = float(L.params.get("min_val", -1.0))
    max_val = float(L.params.get("max_val", 1.0))
    l, u = Bin.lb, Bin.ub
    lb = torch.clamp(l, min=min_val, max=max_val)
    ub = torch.clamp(u, min=min_val, max=max_val)
    return _fact(L, Bounds(lb, ub), "INEQ", {"tag": f"hardtanh:{L.id}", "min_val": min_val, "max_val": max_val})

def tf_hardsigmoid(L: Layer, Bin: Bounds) -> Fact:
    """HardSigmoid: clamp(alpha * x + beta, 0, 1)"""
    alpha = float(L.params.get("alpha", 1/6))
    beta = float(L.params.get("beta", 0.5))
    l, u = Bin.lb, Bin.ub
    # Apply linear transformation then clamp
    l_linear = alpha * l + beta
    u_linear = alpha * u + beta
    lb = torch.clamp(l_linear, min=0.0, max=1.0)
    ub = torch.clamp(u_linear, min=0.0, max=1.0)
    return _fact(L, Bounds(lb, ub), "INEQ", {"tag": f"hardsigmoid:{L.id}", "alpha": alpha, "beta": beta})

def tf_hardswish(L: Layer, Bin: Bounds) -> Fact:
    """HardSwish: x * clamp(x+3, 0, 6) / 6.

    Has a single global minimum at x = -1.5 with f = -0.375 (the dip
    inside [-3, 0]); endpoint-only evaluation is unsound when the input
    range crosses x = -1.5.
    """
    HS_MIN_X = -1.5
    HS_MIN_Y = -0.375
    f = lambda x: x * torch.clamp(x + 3, min=0.0, max=6.0) / 6.0
    f_lb, f_ub = f(Bin.lb), f(Bin.ub)
    contains_min = (Bin.lb <= HS_MIN_X) & (Bin.ub >= HS_MIN_X)
    lb = torch.where(contains_min, torch.full_like(f_lb, HS_MIN_Y), torch.minimum(f_lb, f_ub))
    ub = torch.maximum(f_lb, f_ub)
    return _fact(L, Bounds(lb, ub), "INEQ", {"tag": f"hardswish:{L.id}"})

def tf_mish(L: Layer, Bin: Bounds) -> Fact:
    """Mish: x * tanh(softplus(x)).

    Has a single global minimum at x ≈ -1.1924 with f ≈ -0.30884.
    Monotone-decreasing for x < min_x (towards 0 as x → -∞) and
    monotone-increasing for x > min_x.
    """
    MISH_MIN_X = -1.1924
    MISH_MIN_Y = -0.30884
    f = lambda x: x * torch.tanh(torch.nn.functional.softplus(x))
    f_lb, f_ub = f(Bin.lb), f(Bin.ub)
    contains_min = (Bin.lb <= MISH_MIN_X) & (Bin.ub >= MISH_MIN_X)
    lb = torch.where(contains_min, torch.full_like(f_lb, MISH_MIN_Y), torch.minimum(f_lb, f_ub))
    ub = torch.maximum(f_lb, f_ub)
    return _fact(L, Bounds(lb, ub), "INEQ", {"tag": f"mish:{L.id}"})

def tf_softsign(L: Layer, Bin: Bounds) -> Fact:
    """SoftSign: x / (1 + |x|)"""
    l, u = Bin.lb, Bin.ub
    # SoftSign is bounded between -1 and 1
    lb = l / (1 + torch.abs(l))
    ub = u / (1 + torch.abs(u))
    return _fact(L, Bounds(lb, ub), "INEQ", {"tag": f"softsign:{L.id}"})

# -------- Tensor Operations --------
def _identity_eq(name: str, param_key: str):
    """Build a TF for a shape-only op: bounds pass through, EQ Con carries param_key."""
    def tf(L: Layer, Bin: Bounds) -> Fact:
        B = Bounds(Bin.lb.clone(), Bin.ub.clone())
        return _fact(L, B, "EQ", {"tag": f"{name}:{L.id}", param_key: L.params.get(param_key)})
    return tf

tf_reshape = _identity_eq("reshape", "target_shape")
tf_squeeze = _identity_eq("squeeze", "dims")
tf_unsqueeze = _identity_eq("unsqueeze", "dims")

def _transpose_flat_indices(
    L: Layer,
    n: int,
    device: torch.device,
    input_shape=None,
) -> torch.Tensor:
    input_shape = L.params.get("input_shape") or input_shape
    perm = L.params.get("perm")
    if perm is None:
        return torch.arange(n, device=device)
    if input_shape is None:
        raise ValueError(f"transpose layer {L.id} requires input_shape")
    input_shape = tuple(int(d) for d in input_shape)
    perm = tuple(int(d) for d in perm)
    if math.prod(input_shape) != n:
        raise ValueError(
            f"transpose layer {L.id}: input_shape {input_shape} has "
            f"{math.prod(input_shape)} elements, expected {n}"
        )
    if sorted(perm) != list(range(len(input_shape))):
        raise ValueError(f"transpose layer {L.id}: invalid permutation {perm}")
    return torch.arange(n, device=device).reshape(input_shape).permute(*perm).reshape(-1)


def tf_transpose(L: Layer, Bin: Bounds) -> Fact:
    perm_raw = L.params.get("perm")
    input_shape = L.params.get("input_shape") or (
        1, *tuple(int(d) for d in Bin.lb.shape[1:])
    )
    if perm_raw is None:
        B = Bounds(Bin.lb.clone(), Bin.ub.clone())
    else:
        perm = tuple(int(d) for d in perm_raw)
        width = Bin.lb[0].numel()
        rows = _transpose_flat_indices(L, width, Bin.lb.device, input_shape)
        input_shape = tuple(int(d) for d in input_shape)
        if len(L.in_vars) != width or len(L.out_vars) != width:
            raise ValueError(
                f"transpose layer {L.id}: expected {width} input/output variables, "
                f"got {len(L.in_vars)}/{len(L.out_vars)}"
            )
        # Bounds laid out as [lanes, *input_shape[1:]] permute in place when the
        # leading (batch) axis stays first; any other per-lane shape, e.g. an
        # unbatched (n, 1) column, permutes its flattened lane coordinates.
        if Bin.lb.ndim == len(perm) and perm[0] == 0 and tuple(Bin.lb.shape[1:]) == input_shape[1:]:
            B = Bounds(
                Bin.lb.permute(*perm).contiguous(),
                Bin.ub.permute(*perm).contiguous(),
            )
        else:
            B = Bounds(
                Bin.lb.reshape(Bin.lb.shape[0], width).index_select(1, rows),
                Bin.ub.reshape(Bin.ub.shape[0], width).index_select(1, rows),
            )
    return _fact(L, B, "EQ", {
        "tag": f"transpose:{L.id}",
        "perm": perm_raw,
        "input_shape": input_shape,
    })

def tf_expand(L: Layer, Bin: Bounds) -> Fact:
    batch_size = Bin.lb.shape[0]
    in_shape = L.params.get("input_shape")
    out_shape = L.params.get("output_shape") or L.params.get("shape")
    n_out = len(L.out_vars)
    relation = None
    if in_shape is not None and out_shape is not None:
        in_shape = tuple(int(d) for d in in_shape)
        out_shape = tuple(int(d) for d in out_shape)
        lb = Bin.lb.view(batch_size, *in_shape).broadcast_to(batch_size, *out_shape).reshape(batch_size, -1).clone()
        ub = Bin.ub.view(batch_size, *in_shape).broadcast_to(batch_size, *out_shape).reshape(batch_size, -1).clone()
        relation = {"tag": f"expand:{L.id}", "input_shape": in_shape, "output_shape": out_shape}
    elif Bin.lb.shape[1] == n_out:
        lb, ub = Bin.lb.clone(), Bin.ub.clone()
    else:
        width = Bin.lb.shape[1]
        repeat = (n_out + width - 1) // width
        lb = Bin.lb.repeat(1, repeat)[:, :n_out].clone()
        ub = Bin.ub.repeat(1, repeat)[:, :n_out].clone()
    Bout = Bounds(lb, ub)
    C = ConSet(); C.add_box(L.id, L.out_vars, Bout)
    if relation is not None:
        C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), relation))
    return Fact(Bout, C)
    
def tf_slice(L: Layer, Bin: Bounds) -> Fact:
    batch_size = Bin.lb.shape[0]
    inp_shape = tuple(L.params["input_shape"])  # e.g. (1, 3, 32, 32)
    x_lb = Bin.lb.view(batch_size, *inp_shape)
    x_ub = Bin.ub.view(batch_size, *inp_shape)

    starts = L.params.get("starts", [])
    ends   = L.params.get("ends", [])
    axes   = L.params.get("axes", list(range(len(inp_shape))))
    steps  = L.params.get("steps", [1] * len(axes))

    # Build slice objects for each dimension
    slices = [slice(None)] * (len(inp_shape) + 1)
    for i, axis in enumerate(axes):
        axis = int(axis)
        s = starts[i]
        e = ends[i]
        st = steps[i]
        if e > inp_shape[axis]:
            e = inp_shape[axis]
        slices[axis + 1] = slice(s, e, st)

    out_lb = x_lb[tuple(slices)]
    out_ub = x_ub[tuple(slices)]
    assert out_lb.shape[0] == batch_size, f"slice batch mismatch {out_lb.shape[0]} != {batch_size}"
    assert out_lb[0].numel() == len(L.out_vars), f"slice out_vars length {len(L.out_vars)} != output elements {out_lb[0].numel()}"
    assert torch.all(out_lb <= out_ub), "slice produced invalid bounds (lb > ub)"

    Bout = Bounds(out_lb.reshape(batch_size, -1), out_ub.reshape(batch_size, -1))

    return _fact(L, Bout, "EQ", {
        "tag": f"slice:{L.id}",
        "starts": starts,
        "ends": ends,
        "axes": axes,
        "steps": steps,
        "input_shape": inp_shape,
    })


def tf_gather(L: Layer, Bin: Bounds) -> Fact:

    batch_size = Bin.lb.shape[0]
    inp_shape = tuple(L.params["input_shape"])
    axis = int(L.params.get("axis", 0))
    x_lb = Bin.lb.view(batch_size, *inp_shape)
    x_ub = Bin.ub.view(batch_size, *inp_shape)

    raw_idx = L.params["indices"]
    if isinstance(raw_idx, (list, tuple)):
        indices = torch.tensor(raw_idx, dtype=torch.long)
    else:
        indices = raw_idx.to(x_lb.device).long()

    out_lb = torch.index_select(x_lb, dim=axis + 1, index=indices)
    out_ub = torch.index_select(x_ub, dim=axis + 1, index=indices)

    Bout = Bounds(out_lb.reshape(batch_size, -1), out_ub.reshape(batch_size, -1))

    return _fact(L, Bout, "EQ", {
        "tag": f"gather:{L.id}",
        "axis": axis,
        "indices": indices.detach().cpu().tolist(),
        "input_shape": inp_shape,
        "output_shape": list(out_lb.shape),
    })
