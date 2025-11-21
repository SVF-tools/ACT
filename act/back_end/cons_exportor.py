#===- act/back_end/cons_exportor.py - Constraint Set Export Utilities ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Constraint set export utilities for external solver integration.
#   Provides export functionality for constraint sets to various formats.
#
#===---------------------------------------------------------------------===#

import numpy as np
import torch
from typing import Optional, Tuple
from act.back_end.core import ConSet
from act.back_end.solver.solver_base import Solver
from act.util.device_manager import get_default_device, get_default_dtype

def to_numpy(x) -> np.ndarray:
    try:
        if isinstance(x, torch.Tensor):
            # Use current default dtype and ensure proper device handling
            current_dtype = get_default_dtype()
            return x.detach().to("cpu", dtype=current_dtype).numpy()
    except Exception:
        pass
    # Use the global dtype for numpy conversion too
    current_dtype = get_default_dtype()
    if current_dtype == torch.float16:
        np_dtype = np.float16
    elif current_dtype == torch.float32:
        np_dtype = np.float32
    else:  # torch.float64
        np_dtype = np.float64
    return np.asarray(x, dtype=np_dtype)

def export_to_solver(globalC: ConSet, solver: Solver,
                     objective: Optional[Tuple[np.ndarray, float]]=None, sense="min") -> int:
    # Use device manager to get optimal device hint
    default_device = get_default_device()
    dev_hint = str(default_device)  # Use global device manager default
    
    # Only initialize solver if it hasn't been pre-configured
    if hasattr(solver, 'n') and solver.n == 0:
        print(f"🔧 export_to_solver: Initializing solver (current vars: {solver.n})")
        solver.begin("verify", device=dev_hint)
    else:
        print(f"🔧 export_to_solver: Solver already initialized (current vars: {getattr(solver, 'n', 'unknown')})")

    # 1) global var set and merged boxes
    all_ids=set(); boxes={}
    templates=list(globalC)
    for con in templates:
        all_ids.update(con.var_ids)
        tag = con.meta.get("tag","")

        if tag.startswith("box:"):
            lb = to_numpy(con.meta["lb"])
            ub = to_numpy(con.meta["ub"])

            # 统一成一维向量
            lb = lb.reshape(-1)
            ub = ub.reshape(-1)

            n_box = lb.shape[0]
            n_ids = len(con.var_ids)

            if n_box == n_ids:
                # 原来的“每个 var_id 一个 bound”模式
                for i, vid in enumerate(con.var_ids):
                    cur = boxes.get(vid, (-np.inf, +np.inf))
                    boxes[vid] = (
                        max(cur[0], float(lb[i])),
                        min(cur[1], float(ub[i])),
                    )

            elif n_ids == 1 and n_box > 1:
                base = con.var_ids[0]
                for offset in range(n_box):
                    vid = base + offset
                    # 这些变量是真实存在的，必须参与 all_ids 统计
                    all_ids.add(vid)

                    cur = boxes.get(vid, (-np.inf, +np.inf))
                    boxes[vid] = (
                        max(cur[0], float(lb[offset])),
                        min(cur[1], float(ub[offset])),
                    )

            elif n_box == 1 and n_ids > 1:
                # 兼容“标量 lb/ub 广播到一组 var_ids”
                for vid in con.var_ids:
                    cur = boxes.get(vid, (-np.inf, +np.inf))
                    boxes[vid] = (
                        max(cur[0], float(lb[0])),
                        min(cur[1], float(ub[0])),
                    )

            else:
                raise AssertionError(
                    f"box shape mismatch: lb len={n_box}, var_ids len={n_ids}"
                )

        # 非 box 约束在这一步只是参与 all_ids 统计，box 逻辑之外不用动


        # if tag.startswith("box:"):
        #     lb = to_numpy(con.meta["lb"]); ub = to_numpy(con.meta["ub"])
        #     assert lb.shape[0] == len(con.var_ids), \
        #     f"box lb length mismatch: {lb.shape[0]} vs {len(con.var_ids)}"
        #     assert ub.shape[0] == len(con.var_ids), \
        #     f"box ub length mismatch: {ub.shape[0]} vs {len(con.var_ids)}"
        #     for i, vid in enumerate(con.var_ids):
        #         cur=boxes.get(vid, (-np.inf, +np.inf))
        #         boxes[vid]=(max(cur[0], float(lb[i])), min(cur[1], float(ub[i])))

    nvars = max(all_ids)+1 if all_ids else 0
    solver.add_vars(nvars)
    if boxes:
        idxs=sorted(boxes.keys())
        lb=np.array([boxes[i][0] for i in idxs],dtype=np.float64)
        ub=np.array([boxes[i][1] for i in idxs],dtype=np.float64)
        solver.set_bounds(idxs, lb, ub)

    # 2) materialize per-tag
    for con in templates:
        tag = con.meta.get("tag","")
        if tag.startswith("box:"): continue

        if tag.startswith("dense:"):
            W = to_numpy(con.meta["W"]); b = to_numpy(con.meta["b"])
            assert W.ndim == 2, f"dense: W must be 2D, got {W.shape}"
            # W has shape (n_out, n_in), so we know the dimensions
            n_out, n_in = W.shape
            assert b.shape[0] == n_out, f"dense: b len {b.shape[0]} != n_out {n_out}"
            assert len(con.var_ids) == n_out + n_in, \
            f"dense: expected {n_out + n_in} var_ids, got {len(con.var_ids)}"
            # Take the first n_out variables as outputs, the rest as inputs
            y = list(con.var_ids[:n_out])
            x = list(con.var_ids[n_out:])
            for i, yi in enumerate(y):
                solver.add_lin_eq([yi]+x, [1.0]+[-float(W[i,j]) for j in range(W.shape[1])], float(b[i]))

        elif tag.startswith("bias:"):
            n = len(con.var_ids) // 2
            assert 2 * n == len(con.var_ids), f"bias: var_ids length not even: {len(con.var_ids)}"
            y = list(con.var_ids[:n]); x = list(con.var_ids[n:])
            c = to_numpy(con.meta["c"])
            assert c.shape[0] == n, f"bias: c length {c.shape[0]} != {n}"
            for i, yi in enumerate(y): solver.add_lin_eq([yi,x[i]],[1.0,-1.0], float(c[i]))

        elif tag.startswith("scale:"):
            n=len(con.var_ids)//2; y=list(con.var_ids[:n]); x=list(con.var_ids[n:])
            a=to_numpy(con.meta["a"])
            for i, yi in enumerate(y): solver.add_lin_eq([yi,x[i]],[1.0,-float(a[i])], 0.0)

        elif tag.startswith("bn:"):
            n=len(con.var_ids)//2; y=list(con.var_ids[:n]); x=list(con.var_ids[n:])
            A=to_numpy(con.meta["A"]); c=to_numpy(con.meta["c"])
            for i, yi in enumerate(y): solver.add_lin_eq([yi,x[i]],[1.0,-float(A[i])], float(c[i]))

        elif tag.startswith("add:"):
            n=len(con.var_ids)//3
            z=list(con.var_ids[:n]); x=list(con.var_ids[n:2*n]); y=list(con.var_ids[2*n:])
            for i, zi in enumerate(z): solver.add_lin_eq([zi,x[i],y[i]],[1.0,-1.0,-1.0], 0.0)

        elif tag.startswith("relu:"):
            meta=con.meta; n=len(con.var_ids)//2; z=list(con.var_ids[:n]); y=list(con.var_ids[n:])
            for i in to_numpy(meta["idx_on"]).astype(int):  solver.add_lin_eq([z[i],y[i]],[1.0,-1.0],0.0)
            for i in to_numpy(meta["idx_off"]).astype(int): solver.add_lin_eq([z[i]],[1.0],0.0)
            slope=to_numpy(meta["slope"]); shift=to_numpy(meta["shift"])
            for k, i in enumerate(to_numpy(meta["idx_amb"]).astype(int)):
                solver.add_lin_le([z[i]], [-1.0], 0.0)
                solver.add_lin_le([y[i], z[i]], [1.0, -1.0], 0.0)
                solver.add_lin_le([z[i], y[i]], [1.0, -float(slope[k])], float(shift[k]))

        elif tag.startswith("lrelu:"):
            meta=con.meta; alpha=float(meta["alpha"]); n=len(con.var_ids)//2
            z=list(con.var_ids[:n]); y=list(con.var_ids[n:])
            for i in to_numpy(meta["idx_on"]).astype(int):  solver.add_lin_eq([z[i],y[i]],[1.0,-1.0],0.0)
            for i in to_numpy(meta["idx_off"]).astype(int): solver.add_lin_eq([z[i],y[i]],[1.0, alpha],0.0)
            for i in to_numpy(meta["idx_amb"]).astype(int):
                solver.add_lin_le([y[i],z[i]],[ 1.0,-1.0],0.0)
                solver.add_lin_le([y[i],z[i]],[ alpha,-1.0],0.0)
            slope=to_numpy(meta["slope"]); shift=to_numpy(meta["shift"])
            for k, i in enumerate(to_numpy(meta["idx_amb"]).astype(int)):
                solver.add_lin_le([z[i], y[i]], [1.0, -float(slope[k])], float(shift[k]))

        elif tag.startswith("abs:"):
            meta=con.meta; n=len(con.var_ids)//2; z=list(con.var_ids[:n]); y=list(con.var_ids[n:])
            for i in to_numpy(meta["idx_pos"]).astype(int): solver.add_lin_eq([z[i],y[i]],[1.0,-1.0],0.0)
            for i in to_numpy(meta["idx_neg"]).astype(int): solver.add_lin_eq([z[i],y[i]],[1.0, 1.0],0.0)
            for i in to_numpy(meta["idx_amb"]).astype(int):
                solver.add_lin_le([y[i], z[i]],[ 1.0,-1.0],0.0)
                solver.add_lin_le([y[i], z[i]],[-1.0,-1.0],0.0)

        elif tag.startswith("mcc:"):
            meta=con.meta; n=len(con.var_ids)//3
            z=list(con.var_ids[:n]); x=list(con.var_ids[n:2*n]); y=list(con.var_ids[2*n:])
            lx,ux,ly,uy = map(to_numpy, (meta["lx"], meta["ux"], meta["ly"], meta["uy"]))
            for i in range(n):
                solver.add_lin_ge([z[i],y[i],x[i]],[1.0, -float(lx[i]), -float(ly[i])], -float(lx[i]*ly[i]))
                solver.add_lin_ge([z[i],y[i],x[i]],[1.0, -float(ux[i]), -float(uy[i])], -float(ux[i]*uy[i]))
                solver.add_lin_le([z[i],y[i],x[i]],[1.0, -float(lx[i]), -float(uy[i])], -float(lx[i]*uy[i]))
                solver.add_lin_le([z[i],y[i],x[i]],[1.0, -float(ux[i]), -float(ly[i])], -float(ux[i]*ly[i]))

        elif tag.startswith(("max:", "min:")):
            k=int(con.meta["k"]); n_out=len(con.var_ids)//(1+k)
            z=list(con.var_ids[:n_out]); pos=n_out; blocks=[]
            for _ in range(k): blocks.append(list(con.var_ids[pos:pos+n_out])); pos+=n_out
            if tag.startswith("max:"):
                for yi in blocks:
                    for j in range(n_out): solver.add_lin_ge([z[j], yi[j]],[1.0,-1.0],0.0)
            else:
                for yi in blocks:
                    for j in range(n_out): solver.add_lin_le([z[j], yi[j]],[1.0,-1.0],0.0)

        elif tag.startswith("softmax:simplex:"):
            rowsize=int(con.meta["rowsize"]); W=list(con.var_ids)
            assert len(W)%rowsize==0
            for r in range(len(W)//rowsize):
                row=W[r*rowsize:(r+1)*rowsize]; solver.add_ge_zero(row); solver.add_sum_eq(row, 1.0)
        
        elif tag == "in:linpoly":
            # Input specification: A·x ≤ b (linear polytope constraint)
            A = to_numpy(con.meta["A"])
            b = to_numpy(con.meta["b"])
            vids = list(con.var_ids)
            for i in range(A.shape[0]):
                solver.add_lin_le(vids, list(A[i, :]), float(b[i]))
        
        else:
            pass

    # 3) objective (optional)
    if objective is None: solver.set_objective_linear([],[],0.0,"min")
    else:
        c,c0 = objective; vids=list(range(len(c))); coeffs=[float(ci) for ci in c]
        solver.set_objective_linear(vids, coeffs, float(c0), sense)
    return nvars
