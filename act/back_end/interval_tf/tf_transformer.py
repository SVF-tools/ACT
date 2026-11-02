#===- act/back_end/interval_tf/tf_transformer.py - Transformer Interval TF ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Transformer Interval Transfer Functions. Provides interval-based transfer
#   functions for transformer components including attention mechanisms.
#
#===---------------------------------------------------------------------===#

import torch
from typing import List, Optional, Tuple, cast
from act.back_end.core import Bounds, Con, ConSet, Fact, Layer
from act.back_end.utils import pwl_meta, scale_interval, four_corner_envelope
from act.back_end.interval_tf.tf_mlp import tf_concat
from act.back_end.interval_tf.tf_attention import (
    LinearBounds, att_scores_dual_planar, _GELU_MIN_X, _GELU_MIN_Y,
)

# tf_embedding is provided by act.back_end.interval_tf.tf_rnn (signature
# (L, Bin) -> Fact). The previous transformer-local definition had a wrong
# signature and would shadow the rnn one via `from tf_transformer import *`
# in interval_tf.py — both EMBEDDING and EMBEDDING_TF would have raised
# TypeError at runtime. Single source of truth lives in tf_rnn.

def _round_nan_outward(lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Round 0*inf / inf-inf NaNs outward to a sound extended-real interval.

    Interval TFs are seeded with +/-inf placeholders on the first worklist
    visit, before a predecessor has converged. The LayerNorm std-normalisation
    product cx*inv and the GELU polynomial x*(1+tanh(...)) then evaluate the
    indeterminate 0*inf at an unbounded endpoint, which IEEE returns as NaN.
    NaN is unsound as a bound and, worse, invisible to the fixpoint change test
    (every NaN comparison is false), so it would stick forever and never refine.
    Widening NaN to -inf (lb) / +inf (ub) keeps the interval a sound
    over-approximation and lets a later visit with finite predecessor bounds
    recompute a finite result.
    """
    lb = torch.where(torch.isnan(lb), lb.new_full((), float("-inf")), lb)
    ub = torch.where(torch.isnan(ub), ub.new_full((), float("inf")), ub)
    return lb, ub

def tf_posenc(L: Layer, Bin: Bounds) -> Fact:
    P = cast(torch.Tensor, L.params["pos_vec"]); B=Bounds(Bin.lb+P, Bin.ub+P); C=ConSet()
    C.replace(Con("EQ", tuple(L.out_vars+L.in_vars), {"tag":f"posenc:{L.id}"})); C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def _layernorm_box(L: Layer, Bin: Bounds) -> Bounds:
    """Bound normalization independently over each trailing gamma-sized row."""
    if Bin.lb.dim() < 2:
        raise ValueError(f"LAYERNORM expects batched bounds [B, *], got shape {tuple(Bin.lb.shape)}")
    variant = L.params.get("variant", L.params.get("layer_norm", "standard"))
    if variant == "no":
        output = Bin.copy()
    else:
        gamma = cast(torch.Tensor, L.params["gamma"]).to(Bin.lb).flatten()
        beta = cast(torch.Tensor, L.params["beta"]).to(Bin.lb).flatten()
        width = gamma.numel()
        if width == 0 or Bin.lb.flatten(1).shape[1] % width:
            raise ValueError(f"LAYERNORM {L.id}: gamma width must divide the input width")
        lower = Bin.lb.reshape(Bin.lb.shape[0], -1, width)
        upper = Bin.ub.reshape(Bin.ub.shape[0], -1, width)
        cx_lb = lower - upper.mean(-1, keepdim=True)
        cx_ub = upper - lower.mean(-1, keepdim=True)
        if variant == "no_var":
            sh_lb, sh_ub = cx_lb, cx_ub
        else:
            # Variance includes differences between coordinate centers, even
            # when the input box has zero radius.
            square_lb = torch.where(
                (cx_lb <= 0) & (cx_ub >= 0), torch.zeros_like(cx_lb),
                torch.minimum(cx_lb.square(), cx_ub.square()),
            )
            v_lo = square_lb.mean(-1, keepdim=True)
            v_hi = torch.maximum(cx_lb.square(), cx_ub.square()).mean(-1, keepdim=True)
            eps = float(cast(float, L.params.get("eps", 1e-5)))
            inv_lb, inv_ub = torch.rsqrt(v_hi + eps), torch.rsqrt(v_lo + eps)
            sh_lb, sh_ub = scale_interval(cx_lb, cx_ub, inv_lb, inv_ub)
        lb = torch.where(gamma >= 0, gamma * sh_lb + beta, gamma * sh_ub + beta)
        ub = torch.where(gamma >= 0, gamma * sh_ub + beta, gamma * sh_lb + beta)
        lb, ub = _round_nan_outward(lb.reshape_as(Bin.lb), ub.reshape_as(Bin.ub))
        output = Bounds(lb, ub)
    return output


def tf_layernorm(L: Layer, Bin: Bounds) -> Fact:
    """Attach the existing LayerNorm constraints to its shared interval image."""
    output = _layernorm_box(L, Bin)
    C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"layernorm:{L.id}"}))
    C.add_box(L.id,L.out_vars,output); return Fact(output,C)

def _gelu_box(L: Layer, Bin: Bounds) -> Bounds:
    """Bound the declared GELU mode, sharing its image with the dual kernels."""
    approximate = cast(str, L.params.get("approximate", "none"))
    if approximate == "none":
        f_lb = torch.nn.functional.gelu(Bin.lb)
        f_ub = torch.nn.functional.gelu(Bin.ub)
        contains_min = (Bin.lb <= _GELU_MIN_X) & (Bin.ub >= _GELU_MIN_X)
        lb = torch.where(contains_min, torch.full_like(f_lb, _GELU_MIN_Y),
                         torch.minimum(f_lb, f_ub))
        ub = torch.maximum(f_lb, f_ub)
    elif approximate == "tanh":
        # The tanh factor is monotone; its product with x is not.
        def factor(x: torch.Tensor) -> torch.Tensor:
            return 0.5 * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x**3)))
        lb, ub = scale_interval(Bin.lb, Bin.ub, factor(Bin.lb), factor(Bin.ub))
        point = Bin.lb == Bin.ub
        value = torch.nn.functional.gelu(Bin.lb, approximate="tanh")
        lb, ub = torch.where(point, value, lb), torch.where(point, value, ub)
    else:
        raise ValueError(f"GELU {L.id}: unsupported approximate={approximate!r}")
    lb, ub = _round_nan_outward(lb, ub)
    return Bounds(lb, ub)


def tf_gelu(L: Layer, Bin: Bounds) -> Fact:
    """Attach GELU constraints to the shared, mode-correct interval image."""
    B = _gelu_box(L, Bin); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"gelu:{L.id}","segs":pwl_meta(Bin.lb,Bin.ub,3)}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_att_scores(L: Layer, Bq: Bounds, Bk: Bounds) -> Fact:
    if L.params.get("attn_mode") == "dual_planar":
        return _tf_att_scores_dual_planar(L)
    batch_size = Bq.lb.shape[0]
    if Bk.lb.shape[0] != batch_size:
        raise ValueError(f"ATT_SCORES expects matching batch dims, got {batch_size} and {Bk.lb.shape[0]}")
    s=Bq.lb.new_tensor(1.0/float(cast(float, cast(object, L.params["dk"]))))
    lb,ub=four_corner_envelope(Bq.lb, Bq.ub, Bk.lb, Bk.ub, sum_axis=-1, keepdim=True, scale=s)
    mask = L.params.get("mask")
    if isinstance(mask, torch.Tensor): lb=lb+mask; ub=ub+mask
    B=Bounds(lb,ub); C=ConSet()
    q_vars = cast(List[int], cast(object, L.params["q_vars"]))
    k_vars = cast(List[int], cast(object, L.params["k_vars"]))
    C.replace(Con("INEQ", tuple(L.out_vars + q_vars + k_vars), {"tag":f"att_scores:{L.id}","scale":float(s.item()),"mcc":True}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def _tf_att_scores_dual_planar(L: Layer) -> Fact:
    """ReLU-catalyzed dual-planar QKt scores as a tighter interval box.

    Reads per-head query/key ``LinearBounds`` frames from the layer params,
    fuses the two attention planes (rule slope when ``clamp_alpha`` is unset,
    optimized-alpha init when set), and concretizes to a sound box. The McCormick
    ``att_scores`` path stays the default; this mode is the tighter dual-path
    bound and emits its own ``att_dual_planar`` tag for the solver export.
    """
    fused = att_scores_dual_planar(
        cast(LinearBounds, cast(object, L.params["q_lb"])),
        cast(LinearBounds, cast(object, L.params["k_lb"])),
        head_size=cast(int, L.params["head_size"]),
        k=float(cast(float, cast(object, L.params.get("k_thresh", 1.0)))),
        clamp_alpha=bool(L.params.get("clamp_alpha", False)),
        mask=cast(Optional[torch.Tensor], L.params.get("mask")),
    )
    lo, hi = fused.concretize()
    bs = lo.shape[0]
    B = Bounds(lo.reshape(bs, -1), hi.reshape(bs, -1))
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars), {"tag": f"att_dual_planar:{L.id}"}))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)


def softmax_rowsize(L: Layer, bounds: Bounds) -> Optional[int]:
    if "rowsize" in L.params:
        row = L.get_int("rowsize")
        if row < 1 or bounds.lb.flatten(1).shape[1] % row:
            raise ValueError(f"SOFTMAX {L.id}: rowsize must divide the feature width")
        return row
    input_shape = L.params.get("input_shape")
    axis = L.get_int("axis", -1)
    if input_shape is None:
        return int(bounds.lb.shape[-1])
    shape = tuple(int(d) for d in cast(Tuple[int, ...], input_shape))
    axis = axis if axis >= 0 else len(shape) + axis
    if axis == len(shape) - 1 and bounds.lb.numel() % shape[axis] == 0:
        return int(shape[axis])
    return None


def tf_softmax(L: Layer, Bin: Bounds) -> Fact:
    B=Bounds(torch.zeros_like(Bin.lb), torch.ones_like(Bin.ub))
    rowsize=softmax_rowsize(L, Bin); mode=L.params.get("mode","simplex"); tag=f"softmax:{mode}:{L.id}"
    C=ConSet()
    if rowsize is not None:
        if mode=="simplex": C.replace(Con("INEQ", tuple(L.out_vars), {"tag":tag,"rowsize":rowsize}))
        elif mode=="pwl":  C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":tag,"rowsize":rowsize,"segs":{"K":3}}))
        else:              C.replace(Con("BIN",  tuple(L.out_vars+L.in_vars), {"tag":tag,"rowsize":rowsize,"K":4,"sos2":True}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_att_mix(L: Layer, Bw: Bounds, Bv: Bounds) -> Fact:
    batch_size = Bw.lb.shape[0]
    if Bv.lb.shape[0] != batch_size:
        raise ValueError(f"ATT_MIX expects matching batch dims, got {batch_size} and {Bv.lb.shape[0]}")
    lo,hi=four_corner_envelope(Bw.lb, Bw.ub, Bv.lb, Bv.ub, sum_axis=-1, keepdim=True)
    B=Bounds(lo,hi); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + cast(List[int], cast(object, L.params["w_vars"])) + cast(List[int], cast(object, L.params["v_vars"]))), {"tag":f"att_mix:{L.id}","mcc":True,"rowsize":L.params["rowsize"]}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_mha_split(L: Layer, Bin: Bounds) -> Fact:
    """Project and select query/key/value slices for explicit attention graphs."""
    weight_value = L.params.get("weight")
    if not isinstance(weight_value, torch.Tensor):
        return Fact(Bin.copy(), ConSet())
    weight = weight_value
    batch_size = Bin.lb.shape[0]
    input_shape_value = cast(Tuple[int, ...], cast(object, L.params.get("input_shape", (1, Bin.lb.shape[1]))))
    input_shape = tuple(int(d) for d in input_shape_value)
    hidden_size = int(cast(int, cast(object, L.params.get("hidden_size", weight.shape[1]))))
    seq_len = int(input_shape[-2]) if len(input_shape) >= 3 else max(Bin.lb.shape[1] // max(hidden_size, 1), 1)
    x_lb = Bin.lb.reshape(batch_size, seq_len, hidden_size)
    x_ub = Bin.ub.reshape(batch_size, seq_len, hidden_size)
    W = weight.to(device=Bin.lb.device, dtype=Bin.lb.dtype)
    W_pos = torch.clamp(W, min=0)
    W_neg = torch.clamp(W, max=0)
    bias = L.params.get("bias")
    b = bias.to(device=Bin.lb.device, dtype=Bin.lb.dtype) if isinstance(bias, torch.Tensor) else Bin.lb.new_zeros(W.shape[0])
    proj_lb = x_lb @ W_pos.T + x_ub @ W_neg.T + b
    proj_ub = x_ub @ W_pos.T + x_lb @ W_neg.T + b
    role = str(L.params.get("role", ""))
    if role in {"query", "key"}:
        position = int(cast(int, cast(object, L.params.get("position", 0))))
        out_lb = proj_lb[:, position, :]
        out_ub = proj_ub[:, position, :]
    elif role == "value":
        feature = int(cast(int, cast(object, L.params.get("feature", 0))))
        out_lb = proj_lb[:, :, feature]
        out_ub = proj_ub[:, :, feature]
    else:
        out_lb = proj_lb.reshape(batch_size, -1)
        out_ub = proj_ub.reshape(batch_size, -1)
    B = Bounds(out_lb.reshape(batch_size, -1), out_ub.reshape(batch_size, -1))
    C = ConSet(); C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)
def tf_mha_join(L: Layer, Bs: List[Bounds]) -> Fact:
    shaped = [Bounds(b.lb.unsqueeze(-1), b.ub.unsqueeze(-1)) if b.lb.dim() == 1 else b for b in Bs]
    return tf_concat(L, shaped)
def tf_mask_add(L: Layer, Bin: Bounds) -> Fact:
    M = cast(torch.Tensor, L.params["M"]); B=Bounds(Bin.lb+M, Bin.ub+M); C=ConSet()
    C.replace(Con("EQ", tuple(L.out_vars+L.in_vars), {"tag":f"mask:{L.id}"})); C.add_box(L.id,L.out_vars,B); return Fact(B,C)
