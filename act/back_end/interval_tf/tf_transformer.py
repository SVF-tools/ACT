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
from typing import List
from act.back_end.core import Bounds, Con, ConSet, Fact, Layer
from act.back_end.utils import pwl_meta, bound_var_interval, scale_interval
from act.back_end.interval_tf.tf_mlp import tf_concat

def tf_embedding(L: Layer) -> Fact:
    E=L.params["emb_vec"]; B=Bounds(E.clone(), E.clone()); C=ConSet(); C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_posenc(L: Layer, Bin: Bounds) -> Fact:
    P=L.params["pos_vec"]; B=Bounds(Bin.lb+P, Bin.ub+P); C=ConSet()
    C.replace(Con("EQ", tuple(L.out_vars+L.in_vars), {"tag":f"posenc:{L.id}"})); C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_layernorm(L: Layer, Bin: Bounds) -> Fact:
    mu_lb, mu_ub = torch.mean(Bin.lb), torch.mean(Bin.ub)
    cx_lb, cx_ub = Bin.lb - mu_ub, Bin.ub - mu_lb
    v_lo, v_hi = bound_var_interval(Bin.lb, Bin.ub)
    eps=float(L.meta.get("eps",1e-5))
    inv_lb = 1.0/torch.sqrt(torch.tensor(v_hi+eps, dtype=Bin.lb.dtype, device=Bin.lb.device))
    inv_ub = 1.0/torch.sqrt(torch.tensor(max(v_lo,0.0)+eps, dtype=Bin.lb.dtype, device=Bin.lb.device))
    sh_lb, sh_ub = scale_interval(cx_lb, cx_ub, inv_lb, inv_ub)
    gamma,beta=L.params["gamma"],L.params["beta"]
    lb=torch.where(gamma>=0, gamma*sh_lb+beta, gamma*sh_ub+beta)
    ub=torch.where(gamma>=0, gamma*sh_ub+beta, gamma*sh_lb+beta)
    B=Bounds(lb,ub); C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"layernorm:{L.id}"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_gelu(L: Layer, Bin: Bounds) -> Fact:
    f=lambda x: 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi, dtype=x.dtype, device=x.device))*(x+0.044715*(x**3))))
    B=Bounds(f(Bin.lb), f(Bin.ub)); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"gelu:{L.id}","segs":pwl_meta(Bin.lb,Bin.ub,3)}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_att_scores(L: Layer, Bq: Bounds, Bk: Bounds) -> Fact:
    s=torch.tensor(1.0/float(L.meta["dk"]), dtype=Bq.lb.dtype, device=Bq.lb.device)
    lo=torch.minimum(torch.minimum(Bq.lb*Bk.lb, Bq.lb*Bk.ub), torch.minimum(Bq.ub*Bk.lb, Bq.ub*Bk.ub))
    hi=torch.maximum(torch.maximum(Bq.lb*Bk.lb, Bq.lb*Bk.ub), torch.maximum(Bq.ub*Bk.lb, Bq.ub*Bk.ub))
    lb=s*lo.sum(dim=-1); ub=s*hi.sum(dim=-1)
    if L.meta.get("mask") is not None: lb=lb+L.meta["mask"]; ub=ub+L.meta["mask"]
    B=Bounds(lb,ub); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.meta["q_vars"] + L.meta["k_vars"]), {"tag":f"att_scores:{L.id}","scale":float(s),"mcc":True}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_softmax(L: Layer, Bin: Bounds) -> Fact:
    B=Bounds(torch.zeros_like(Bin.lb), torch.ones_like(Bin.ub))
    rowsize=int(L.meta["rowsize"]); mode=L.meta.get("mode","simplex"); tag=f"softmax:{mode}:{L.id}"
    C=ConSet()
    if mode=="simplex": C.replace(Con("INEQ", tuple(L.out_vars), {"tag":tag,"rowsize":rowsize}))
    elif mode=="pwl":  C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":tag,"rowsize":rowsize,"segs":{"K":3}}))
    else:              C.replace(Con("BIN",  tuple(L.out_vars+L.in_vars), {"tag":tag,"rowsize":rowsize,"K":4,"sos2":True}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_att_mix(L: Layer, Bw: Bounds, Bv: Bounds) -> Fact:
    lo=torch.minimum(torch.minimum(Bw.lb*Bv.lb, Bw.lb*Bv.ub), torch.minimum(Bw.ub*Bv.lb, Bw.ub*Bv.ub)).sum(dim=-1)
    hi=torch.maximum(torch.maximum(Bw.lb*Bv.lb, Bw.lb*Bv.ub), torch.maximum(Bw.ub*Bv.lb, Bw.ub*Bv.ub)).sum(dim=-1)
    B=Bounds(lo,hi); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.meta["w_vars"] + L.meta["v_vars"]), {"tag":f"att_mix:{L.id}","mcc":True,"rowsize":L.meta["rowsize"]}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_mha_split(L: Layer, Bin: Bounds) -> Fact: return Fact(Bin.copy(), ConSet())
def tf_mha_join(L: Layer, Bs: List[Bounds]) -> Fact: return tf_concat(L, Bs)
def tf_mask_add(L: Layer, Bin: Bounds) -> Fact:
    M=L.params["M"]; B=Bounds(Bin.lb+M, Bin.ub+M); C=ConSet()
    C.replace(Con("EQ", tuple(L.out_vars+L.in_vars), {"tag":f"mask:{L.id}"})); C.add_box(L.id,L.out_vars,B); return Fact(B,C)