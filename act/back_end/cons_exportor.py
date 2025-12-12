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

# def export_to_solver(globalC: ConSet, solver: Solver,
#                      objective: Optional[Tuple[np.ndarray, float]]=None, sense="min") -> int:
#     # Use device manager to get optimal device hint
#     default_device = get_default_device()
#     dev_hint = str(default_device)  # Use global device manager default
    
#     # Only initialize solver if it hasn't been pre-configured
#     if hasattr(solver, 'n') and solver.n == 0:
#         print(f"🔧 export_to_solver: Initializing solver (current vars: {solver.n})")
#         solver.begin("verify", device=dev_hint)
#     else:
#         print(f"🔧 export_to_solver: Solver already initialized (current vars: {getattr(solver, 'n', 'unknown')})")

#     # 1) global var set and merged boxes
#     all_ids=set(); boxes={}
#     templates=list(globalC)
#     for con in templates:
#         all_ids.update(con.var_ids)
#         tag = con.meta.get("tag","")

#         if tag.startswith("box:"):
#             lb = to_numpy(con.meta["lb"])
#             ub = to_numpy(con.meta["ub"])

#             # 统一成一维向量
#             lb = lb.reshape(-1)
#             ub = ub.reshape(-1)

#             n_box = lb.shape[0]
#             n_ids = len(con.var_ids)

#             if n_box == n_ids:
#                 # 原来的“每个 var_id 一个 bound”模式
#                 for i, vid in enumerate(con.var_ids):
#                     cur = boxes.get(vid, (-np.inf, +np.inf))
#                     boxes[vid] = (
#                         max(cur[0], float(lb[i])),
#                         min(cur[1], float(ub[i])),
#                     )

#             elif n_ids == 1 and n_box > 1:
#                 base = con.var_ids[0]
#                 for offset in range(n_box):
#                     vid = base + offset
#                     # 这些变量是真实存在的，必须参与 all_ids 统计
#                     all_ids.add(vid)

#                     cur = boxes.get(vid, (-np.inf, +np.inf))
#                     boxes[vid] = (
#                         max(cur[0], float(lb[offset])),
#                         min(cur[1], float(ub[offset])),
#                     )

#             elif n_box == 1 and n_ids > 1:
#                 # 兼容“标量 lb/ub 广播到一组 var_ids”
#                 for vid in con.var_ids:
#                     cur = boxes.get(vid, (-np.inf, +np.inf))
#                     boxes[vid] = (
#                         max(cur[0], float(lb[0])),
#                         min(cur[1], float(ub[0])),
#                     )

#             else:
#                 raise AssertionError(
#                     f"box shape mismatch: lb len={n_box}, var_ids len={n_ids}"
#                 )

#         # 非 box 约束在这一步只是参与 all_ids 统计，box 逻辑之外不用动


#         # if tag.startswith("box:"):
#         #     lb = to_numpy(con.meta["lb"]); ub = to_numpy(con.meta["ub"])
#         #     assert lb.shape[0] == len(con.var_ids), \
#         #     f"box lb length mismatch: {lb.shape[0]} vs {len(con.var_ids)}"
#         #     assert ub.shape[0] == len(con.var_ids), \
#         #     f"box ub length mismatch: {ub.shape[0]} vs {len(con.var_ids)}"
#         #     for i, vid in enumerate(con.var_ids):
#         #         cur=boxes.get(vid, (-np.inf, +np.inf))
#         #         boxes[vid]=(max(cur[0], float(lb[i])), min(cur[1], float(ub[i])))

#     nvars = max(all_ids)+1 if all_ids else 0
#     solver.add_vars(nvars)
#     if boxes:
#         idxs = sorted(boxes.keys())
#         lb = np.array([boxes[i][0] for i in idxs], dtype=np.float64)
#         ub = np.array([boxes[i][1] for i in idxs], dtype=np.float64)

#         # 🔍 调试：检查有没有 lb>ub 的变量
#         bad = lb > ub
#         if np.any(bad):
#             print("🚨 [BUG] In export_to_solver: found inconsistent box bounds!")
#             for vid, lo, hi, flag in zip(idxs, lb, ub, bad):
#                 if flag:
#                     print(f"    var {vid}: lb={lo}, ub={hi}")
#             raise RuntimeError("Inconsistent box bounds (lb>ub) - check globalC.add_box / merging logic.")

#         solver.set_bounds(idxs, lb, ub)

#     # if boxes:
#     #     idxs=sorted(boxes.keys())
#     #     lb=np.array([boxes[i][0] for i in idxs],dtype=np.float64)
#     #     ub=np.array([boxes[i][1] for i in idxs],dtype=np.float64)
#     #     solver.set_bounds(idxs, lb, ub)

#     # 2) materialize per-tag
#     for con in templates:
#         tag = con.meta.get("tag","")
#         if tag.startswith("box:"): continue

#         if tag.startswith("dense:"):
#             W = to_numpy(con.meta["W"]); b = to_numpy(con.meta["b"])
#             assert W.ndim == 2, f"dense: W must be 2D, got {W.shape}"
#             # W has shape (n_out, n_in), so we know the dimensions
#             n_out, n_in = W.shape
#             assert b.shape[0] == n_out, f"dense: b len {b.shape[0]} != n_out {n_out}"
#             assert len(con.var_ids) == n_out + n_in, \
#             f"dense: expected {n_out + n_in} var_ids, got {len(con.var_ids)}"
#             # Take the first n_out variables as outputs, the rest as inputs
#             y = list(con.var_ids[:n_out])
#             x = list(con.var_ids[n_out:])
#             for i, yi in enumerate(y):
#                 solver.add_lin_eq([yi]+x, [1.0]+[-float(W[i,j]) for j in range(W.shape[1])], float(b[i]))

#         elif tag.startswith("bias:"):
#             n = len(con.var_ids) // 2
#             assert 2 * n == len(con.var_ids), f"bias: var_ids length not even: {len(con.var_ids)}"
#             y = list(con.var_ids[:n]); x = list(con.var_ids[n:])
#             c = to_numpy(con.meta["c"])
#             assert c.shape[0] == n, f"bias: c length {c.shape[0]} != {n}"
#             for i, yi in enumerate(y): solver.add_lin_eq([yi,x[i]],[1.0,-1.0], float(c[i]))

#         elif tag.startswith("scale:"):
#             n=len(con.var_ids)//2; y=list(con.var_ids[:n]); x=list(con.var_ids[n:])
#             a=to_numpy(con.meta["a"])
#             for i, yi in enumerate(y): solver.add_lin_eq([yi,x[i]],[1.0,-float(a[i])], 0.0)

#         elif tag.startswith("bn:"):
#             n=len(con.var_ids)//2; y=list(con.var_ids[:n]); x=list(con.var_ids[n:])
#             A=to_numpy(con.meta["A"]); c=to_numpy(con.meta["c"])
#             for i, yi in enumerate(y): solver.add_lin_eq([yi,x[i]],[1.0,-float(A[i])], float(c[i]))

#         elif tag.startswith("add:"):
#             n=len(con.var_ids)//3
#             z=list(con.var_ids[:n]); x=list(con.var_ids[n:2*n]); y=list(con.var_ids[2*n:])
#             for i, zi in enumerate(z): solver.add_lin_eq([zi,x[i],y[i]],[1.0,-1.0,-1.0], 0.0)

#         elif tag.startswith("relu:"):
#             meta=con.meta; n=len(con.var_ids)//2; z=list(con.var_ids[:n]); y=list(con.var_ids[n:])
#             for i in to_numpy(meta["idx_on"]).astype(int):  solver.add_lin_eq([z[i],y[i]],[1.0,-1.0],0.0)
#             for i in to_numpy(meta["idx_off"]).astype(int): solver.add_lin_eq([z[i]],[1.0],0.0)
#             slope=to_numpy(meta["slope"]); shift=to_numpy(meta["shift"])
#             for k, i in enumerate(to_numpy(meta["idx_amb"]).astype(int)):
#                 solver.add_lin_le([z[i]], [-1.0], 0.0)
#                 solver.add_lin_le([y[i], z[i]], [1.0, -1.0], 0.0)
#                 solver.add_lin_le([z[i], y[i]], [1.0, -float(slope[k])], float(shift[k]))

#         elif tag.startswith("lrelu:"):
#             meta=con.meta; alpha=float(meta["alpha"]); n=len(con.var_ids)//2
#             z=list(con.var_ids[:n]); y=list(con.var_ids[n:])
#             for i in to_numpy(meta["idx_on"]).astype(int):  solver.add_lin_eq([z[i],y[i]],[1.0,-1.0],0.0)
#             for i in to_numpy(meta["idx_off"]).astype(int): solver.add_lin_eq([z[i],y[i]],[1.0, alpha],0.0)
#             for i in to_numpy(meta["idx_amb"]).astype(int):
#                 solver.add_lin_le([y[i],z[i]],[ 1.0,-1.0],0.0)
#                 solver.add_lin_le([y[i],z[i]],[ alpha,-1.0],0.0)
#             slope=to_numpy(meta["slope"]); shift=to_numpy(meta["shift"])
#             for k, i in enumerate(to_numpy(meta["idx_amb"]).astype(int)):
#                 solver.add_lin_le([z[i], y[i]], [1.0, -float(slope[k])], float(shift[k]))

#         elif tag.startswith("abs:"):
#             meta=con.meta; n=len(con.var_ids)//2; z=list(con.var_ids[:n]); y=list(con.var_ids[n:])
#             for i in to_numpy(meta["idx_pos"]).astype(int): solver.add_lin_eq([z[i],y[i]],[1.0,-1.0],0.0)
#             for i in to_numpy(meta["idx_neg"]).astype(int): solver.add_lin_eq([z[i],y[i]],[1.0, 1.0],0.0)
#             for i in to_numpy(meta["idx_amb"]).astype(int):
#                 solver.add_lin_le([y[i], z[i]],[ 1.0,-1.0],0.0)
#                 solver.add_lin_le([y[i], z[i]],[-1.0,-1.0],0.0)

#         elif tag.startswith("mcc:"):
#             meta=con.meta; n=len(con.var_ids)//3
#             z=list(con.var_ids[:n]); x=list(con.var_ids[n:2*n]); y=list(con.var_ids[2*n:])
#             lx,ux,ly,uy = map(to_numpy, (meta["lx"], meta["ux"], meta["ly"], meta["uy"]))
#             for i in range(n):
#                 solver.add_lin_ge([z[i],y[i],x[i]],[1.0, -float(lx[i]), -float(ly[i])], -float(lx[i]*ly[i]))
#                 solver.add_lin_ge([z[i],y[i],x[i]],[1.0, -float(ux[i]), -float(uy[i])], -float(ux[i]*uy[i]))
#                 solver.add_lin_le([z[i],y[i],x[i]],[1.0, -float(lx[i]), -float(uy[i])], -float(lx[i]*uy[i]))
#                 solver.add_lin_le([z[i],y[i],x[i]],[1.0, -float(ux[i]), -float(ly[i])], -float(ux[i]*ly[i]))

#         elif tag.startswith(("max:", "min:")):
#             k=int(con.meta["k"]); n_out=len(con.var_ids)//(1+k)
#             z=list(con.var_ids[:n_out]); pos=n_out; blocks=[]
#             for _ in range(k): blocks.append(list(con.var_ids[pos:pos+n_out])); pos+=n_out
#             if tag.startswith("max:"):
#                 for yi in blocks:
#                     for j in range(n_out): solver.add_lin_ge([z[j], yi[j]],[1.0,-1.0],0.0)
#             else:
#                 for yi in blocks:
#                     for j in range(n_out): solver.add_lin_le([z[j], yi[j]],[1.0,-1.0],0.0)

#         elif tag.startswith("softmax:simplex:"):
#             rowsize=int(con.meta["rowsize"]); W=list(con.var_ids)
#             assert len(W)%rowsize==0
#             for r in range(len(W)//rowsize):
#                 row=W[r*rowsize:(r+1)*rowsize]; solver.add_ge_zero(row); solver.add_sum_eq(row, 1.0)
        
#         elif tag == "in:linpoly":
#             # Input specification: A·x ≤ b (linear polytope constraint)
#             A = to_numpy(con.meta["A"])
#             b = to_numpy(con.meta["b"])
#             vids = list(con.var_ids)
#             for i in range(A.shape[0]):
#                 solver.add_lin_le(vids, list(A[i, :]), float(b[i]))
        
#         else:
#             pass

#     # 3) objective (optional)
#     if objective is None: solver.set_objective_linear([],[],0.0,"min")
#     else:
#         c,c0 = objective; vids=list(range(len(c))); coeffs=[float(ci) for ci in c]
#         solver.set_objective_linear(vids, coeffs, float(c0), sense)
#     return nvars

def export_to_solver(globalC: ConSet, solver: Solver,
                     objective: Optional[Tuple[np.ndarray, float]] = None,
                     sense: str = "min") -> int:
    # Use device manager to get optimal device hint
    default_device = get_default_device()
    dev_hint = str(default_device)

    # Only initialize solver if it hasn't been pre-configured
    if hasattr(solver, "n") and solver.n == 0:
        print(f"🔧 export_to_solver: Initializing solver (current vars: {solver.n})")
        solver.begin("verify", device=dev_hint)
    else:
        print(
            f"🔧 export_to_solver: Solver already initialized "
            f"(current vars: {getattr(solver, 'n', 'unknown')})"
        )

    # ------------------------------------------------------------------
    # 1) 收集所有变量 ID，并预先计算“每个 var 的最后写入 layer_id”
    # ------------------------------------------------------------------
    all_ids: set[int] = set()
    boxes: dict[int, Tuple[float, float]] = {}
    templates = list(globalC)
    
    # print("===== DUMP globalC (first 20 cons) =====")
    # for idx, con in enumerate(templates[:20]):
    #     tag = con.meta.get("tag","")
    #     print(f"[{idx}] tag={tag}, var_ids={con.var_ids}, meta_keys={list(con.meta.keys())}")
    # print("========================================")
    # Debug: print all constraint tags to trace what is being exported
    for con in templates:
        print("[DEBUG TAG]", con.meta.get("tag", ""))


    # 这些是为了 debug 统计用
    used_in_cons: set[int] = set()
    used_in_box: set[int] = set()

    last_box_layer_for_var: dict[int, int] = {}  # vid -> max layer_id

    for con in templates:
        all_ids.update(con.var_ids)
        tag = con.meta.get("tag", "")
        if not tag.startswith("box:"):
            continue

        # 优先用 meta['layer_id']，没有的话再从 tag 中解析
        if "layer_id" in con.meta:
            layer_id = int(con.meta["layer_id"])
        else:
            try:
                layer_id = int(tag.split("box:")[-1])
            except Exception:
                continue

        for vid in con.var_ids:
            prev = last_box_layer_for_var.get(vid, -1)
            if layer_id > prev:
                last_box_layer_for_var[vid] = layer_id

    # ------------------------------------------------------------------
    # 2) 第二遍：真正合并 box，只保留“最后写入层”的那一个
    # ------------------------------------------------------------------
    for con in templates:
        tag = con.meta.get("tag", "")
        if not tag.startswith("box:"):
            continue

        lb = to_numpy(con.meta["lb"])
        ub = to_numpy(con.meta["ub"])

        # 统一成一维向量
        lb = lb.reshape(-1)
        ub = ub.reshape(-1)

        # 🔥 先在“单个 box 自己”层面检查有没有 lb>ub
        local_bad = lb > ub
        if np.any(local_bad):
            print("🔥 [LOCAL BAD BOX] =================================================")
            print(f"  tag       = {tag}")
            print(f"  var_ids   = {con.var_ids}")
            print(f"  lb.shape  = {lb.shape}, ub.shape = {ub.shape}")
            for i, vid in enumerate(con.var_ids):
                if i < len(lb) and local_bad[i]:
                    print(f"    --> var {vid}: lb={lb[i]}, ub={ub[i]}  (LOCAL BAD)")
            print("==================================================================")

        n_box = lb.shape[0]
        n_ids = len(con.var_ids)

        # 取出当前 box 的 layer_id
        if "layer_id" in con.meta:
            layer_id = int(con.meta["layer_id"])
        else:
            try:
                layer_id = int(tag.split("box:")[-1])
            except Exception:
                layer_id = -1

        if n_box == n_ids:
            # 标准：每个 var_id 一个 bounds
            for i, vid in enumerate(con.var_ids):
                # ⭐ 核心：只保留这个 var 的“最后写入层”的 box
                last_writer = last_box_layer_for_var.get(vid, layer_id)
                if layer_id != last_writer:
                    # 说明还有更靠后的 layer 给这个 vid 写过 box，当前 box 是“旧的”，丢弃
                    continue

                cur = boxes.get(vid, (-np.inf, +np.inf))
                boxes[vid] = (
                    max(cur[0], float(lb[i])),
                    min(cur[1], float(ub[i])),
                )
                used_in_box.add(vid)

        elif n_box == 1 and n_ids > 1:
            # 标量 lb/ub 广播到一组 var_ids，同样应用“最后写入层”规则
            for vid in con.var_ids:
                last_writer = last_box_layer_for_var.get(vid, layer_id)
                if layer_id != last_writer:
                    continue

                cur = boxes.get(vid, (-np.inf, +np.inf))
                boxes[vid] = (
                    max(cur[0], float(lb[0])),
                    min(cur[1], float(ub[0])),
                )
                used_in_box.add(vid)

        else:
            print("🚨 [BUG] box shape mismatch in export_to_solver:")
            print(f"    tag      = {tag}")
            print(f"    var_ids  = {con.var_ids}")
            print(f"    lb.shape = {lb.shape}, ub.shape = {ub.shape}")
            raise AssertionError(
                f"box shape mismatch: lb len={n_box}, var_ids len={n_ids}"
            )

    # ------------------------------------------------------------------
    # 3) 创建 MILP 变量 + 应用 box bounds（过滤掉 lb>ub 的）
    # ------------------------------------------------------------------
    nvars = max(all_ids) + 1 if all_ids else 0

    print("[DEBUG] last_box_layer_for_var (sample):",
          list(sorted(last_box_layer_for_var.items()))[:20])
    print("[DEBUG] boxes keys (sample):", list(sorted(boxes.keys()))[:20])
    print("[DEBUG] nvars:", nvars)

    solver.add_vars(nvars)
    
    all_vids = list(range(nvars))
    lb_all = np.full(nvars, -np.inf, dtype=np.float64)
    ub_all = np.full(nvars,  np.inf, dtype=np.float64)
    solver.set_bounds(all_vids, lb_all, ub_all)

    if boxes:
        idxs = sorted(boxes.keys())
        lb = np.array([boxes[i][0] for i in idxs], dtype=np.float64)
        ub = np.array([boxes[i][1] for i in idxs], dtype=np.float64)

        bad = lb > ub
        if np.any(bad):
            print("🚨 [WARN] In export_to_solver: found inconsistent box bounds! Relaxing to wide interval.")
            lb_relaxed = np.minimum(lb, ub) - 1e-3
            ub_relaxed = np.maximum(lb, ub) + 1e-3
            solver.set_bounds(idxs, lb_relaxed, ub_relaxed)
        else:
            solver.set_bounds(idxs, lb, ub)

    # ------------------------------------------------------------------
    # 4) 其他 tag（dense/relu/mcc/softmax/in:linpoly/...）+ used_in_cons 标记
    # ------------------------------------------------------------------
    for con in templates:
        tag = con.meta.get("tag", "")
        if tag.startswith("box:"):
            continue

        # 统一：凡是这个约束涉及的 var_ids，都视为“出现在约束中”
        used_in_cons.update(con.var_ids)

        if tag.startswith("dense:"):
            W = to_numpy(con.meta["W"])
            b = to_numpy(con.meta["b"])
            assert W.ndim == 2, f"dense: W must be 2D, got {W.shape}"
            n_out, n_in = W.shape
            assert b.shape[0] == n_out, f"dense: b len {b.shape[0]} != n_out {n_out}"
            assert len(con.var_ids) == n_out + n_in, \
                f"dense: expected {n_out + n_in} var_ids, got {len(con.var_ids)}"
            # Take the first n_out variables as outputs, the rest as inputs
            y = list(con.var_ids[:n_out])
            x = list(con.var_ids[n_out:])
            for i, yi in enumerate(y):
                solver.add_lin_eq(
                    [yi] + x,
                    [1.0] + [-float(W[i, j]) for j in range(W.shape[1])],
                    float(b[i]),
                )

        elif tag.startswith("bias:"):
            n = len(con.var_ids) // 2
            assert 2 * n == len(con.var_ids), \
                f"bias: var_ids length not even: {len(con.var_ids)}"
            y = list(con.var_ids[:n])
            x = list(con.var_ids[n:])
            c = to_numpy(con.meta["c"])
            assert c.shape[0] == n, f"bias: c length {c.shape[0]} != {n}"
            for i, yi in enumerate(y):
                solver.add_lin_eq([yi, x[i]], [1.0, -1.0], float(c[i]))

        elif tag.startswith("scale:"):
            n = len(con.var_ids) // 2
            y = list(con.var_ids[:n])
            x = list(con.var_ids[n:])
            a = to_numpy(con.meta["a"])
            for i, yi in enumerate(y):
                solver.add_lin_eq([yi, x[i]], [1.0, -float(a[i])], 0.0)

        elif tag.startswith("bn:"):
            n = len(con.var_ids) // 2
            y = list(con.var_ids[:n])
            x = list(con.var_ids[n:])
            A = to_numpy(con.meta["A"])
            c = to_numpy(con.meta["c"])
            for i, yi in enumerate(y):
                solver.add_lin_eq([yi, x[i]], [1.0, -float(A[i])], float(c[i]))

        elif tag.startswith("add:"):
            n = len(con.var_ids) // 3
            z = list(con.var_ids[:n])
            x = list(con.var_ids[n:2 * n])
            y = list(con.var_ids[2 * n:])
            for i, zi in enumerate(z):
                solver.add_lin_eq([zi, x[i], y[i]], [1.0, -1.0, -1.0], 0.0)


        elif tag.startswith("relu:"):
            """
            ReLU encoding (ACT 原版语义):

            var_ids = [z | y]
              - z: post-activation (output), z = ReLU(y), z >= 0
              - y: pre-activation (input)

            meta:
              - idx_on  : indices where y_lb >= 0  → z = y
              - idx_off : indices where y_ub <= 0  → z = 0
              - idx_amb : ambiguous indices where y_lb < 0 < y_ub
                          use standard triangular relaxation:
                              z >= 0
                              z >= y
                              z <= slope * y + shift
            """
            meta = con.meta
            n = len(con.var_ids) // 2
            assert 2 * n == len(con.var_ids), \
                f"relu: var_ids length not even: {len(con.var_ids)}"

            # 注意：这里沿用 ACT 原来的语义：
            #   z = ReLU 输出（post-activation）
            #   y = ReLU 输入（pre-activation）
            z = list(con.var_ids[:n])
            y = list(con.var_ids[n:])

            idx_on  = to_numpy(meta["idx_on"]).astype(int)
            idx_off = to_numpy(meta["idx_off"]).astype(int)
            idx_amb = to_numpy(meta["idx_amb"]).astype(int)

            slope = to_numpy(meta["slope"])
            shift = to_numpy(meta["shift"])

            # 1) 稳定 ON：y_lb >= 0 → z_i = y_i
            for i in idx_on:
                solver.add_lin_eq(
                    [z[i], y[i]],
                    [1.0, -1.0],
                    0.0,
                )

            # 2) 稳定 OFF：y_ub <= 0 → z_i = 0（y 不再约束）
            for i in idx_off:
                solver.add_lin_eq(
                    [z[i]],
                    [1.0],
                    0.0,
                )

            # 3) ambiguous：三角松弛
            for k, i in enumerate(idx_amb):
                # (a) z >= 0   →  -z <= 0
                solver.add_lin_le(
                    [z[i]],
                    [-1.0],
                    0.0,
                )

                # (b) z >= y   →  y - z <= 0
                solver.add_lin_le(
                    [y[i], z[i]],
                    [1.0, -1.0],
                    0.0,
                )

                # (c) z <= slope * y + shift
                #     → z - slope*y <= shift
                solver.add_lin_le(
                    [z[i], y[i]],
                    [1.0, -float(slope[k])],
                    float(shift[k]),
                )



        elif tag.startswith("lrelu:"):
            meta = con.meta
            alpha = float(meta["alpha"])
            n = len(con.var_ids) // 2
            z = list(con.var_ids[:n])
            y = list(con.var_ids[n:])
            for i in to_numpy(meta["idx_on"]).astype(int):
                solver.add_lin_eq([z[i], y[i]], [1.0, -1.0], 0.0)
            for i in to_numpy(meta["idx_off"]).astype(int):
                solver.add_lin_eq([z[i], y[i]], [1.0, alpha], 0.0)
            for i in to_numpy(meta["idx_amb"]).astype(int):
                solver.add_lin_le([y[i], z[i]], [1.0, -1.0], 0.0)
                solver.add_lin_le([y[i], z[i]], [alpha, -1.0], 0.0)
            slope = to_numpy(meta["slope"])
            shift = to_numpy(meta["shift"])
            for k, i in enumerate(to_numpy(meta["idx_amb"]).astype(int)):
                solver.add_lin_le(
                    [z[i], y[i]],
                    [1.0, -float(slope[k])],
                    float(shift[k]),
                )
                
        elif tag.startswith("top1:"):
            meta = con.meta

            # 所有 logit 的变量 id
            y_vars = list(con.var_ids)

            # 目标类别 t（假设是单个 index）
            t_idx  = int(to_numpy(meta["t_index"]).item())
            v_id   = int(meta["v_id"])
            margin = float(meta.get("margin", 0.0))

            # 语义：v >= y_j - y_t - margin,  对所有 j != t
            for j, yj in enumerate(y_vars):
                if j == t_idx:
                    continue

                # v >= y_j - y_t - margin
                # => v - yj + y_t >= -margin
                solver.add_lin_ge(
                    [v_id, yj, y_vars[t_idx]],
                    [1.0, -1.0, 1.0],
                    -margin,
                )
                
        elif tag.startswith("range:"):
            meta = con.meta

            # 约定：var_ids[0] 是 violation 变量 v，其余是输出 y_j
            v_id = con.var_ids[0]
            y    = list(con.var_ids[1:])

            lb = to_numpy(meta["lb"]).reshape(-1)
            ub = to_numpy(meta["ub"]).reshape(-1)

            assert len(y) == lb.shape[0] == ub.shape[0], \
                f"range: len(y)={len(y)}, lb={lb.shape}, ub={ub.shape}"

            # v >= 0  ->  v >= 0
            solver.add_lin_ge([v_id], [1.0], 0.0)

            # 对每个维度 j：
            #   v >= lb_j - y_j      <=>  v + y_j >= lb_j
            #   v >= y_j - ub_j      <=>  v - y_j >= -ub_j
            for j, yj in enumerate(y):
                solver.add_lin_ge(
                    [v_id, yj],
                    [1.0, 1.0],
                    float(lb[j]),
                )
                solver.add_lin_ge(
                    [v_id, yj],
                    [1.0, -1.0],
                    float(-ub[j]),
                )



        elif tag.startswith("abs:"):
            meta = con.meta
            n = len(con.var_ids) // 2
            z = list(con.var_ids[:n])
            y = list(con.var_ids[n:])
            for i in to_numpy(meta["idx_pos"]).astype(int):
                solver.add_lin_eq([z[i], y[i]], [1.0, -1.0], 0.0)
            for i in to_numpy(meta["idx_neg"]).astype(int):
                solver.add_lin_eq([z[i], y[i]], [1.0, 1.0], 0.0)
            for i in to_numpy(meta["idx_amb"]).astype(int):
                solver.add_lin_le([y[i], z[i]], [1.0, -1.0], 0.0)
                solver.add_lin_le([y[i], z[i]], [-1.0, -1.0], 0.0)

        elif tag.startswith("mcc:"):
            meta = con.meta
            n = len(con.var_ids) // 3
            z = list(con.var_ids[:n])
            x = list(con.var_ids[n:2 * n])
            y = list(con.var_ids[2 * n:])
            lx, ux, ly, uy = map(
                to_numpy,
                (meta["lx"], meta["ux"], meta["ly"], meta["uy"]),
            )
            for i in range(n):
                solver.add_lin_ge(
                    [z[i], y[i], x[i]],
                    [1.0, -float(lx[i]), -float(ly[i])],
                    -float(lx[i] * ly[i]),
                )
                solver.add_lin_ge(
                    [z[i], y[i], x[i]],
                    [1.0, -float(ux[i]), -float(uy[i])],
                    -float(ux[i] * uy[i]),
                )
                solver.add_lin_le(
                    [z[i], y[i], x[i]],
                    [1.0, -float(lx[i]), -float(uy[i])],
                    -float(lx[i] * uy[i]),
                )
                solver.add_lin_le(
                    [z[i], y[i], x[i]],
                    [1.0, -float(ux[i]), -float(ly[i])],
                    -float(ux[i] * ly[i]),
                )

        elif tag.startswith(("max:", "min:")):
            k = int(con.meta["k"])
            n_out = len(con.var_ids) // (1 + k)
            z = list(con.var_ids[:n_out])
            pos = n_out
            blocks = []
            for _ in range(k):
                blocks.append(list(con.var_ids[pos:pos + n_out]))
                pos += n_out
            if tag.startswith("max:"):
                for yi in blocks:
                    for j in range(n_out):
                        solver.add_lin_ge([z[j], yi[j]], [1.0, -1.0], 0.0)
            else:
                for yi in blocks:
                    for j in range(n_out):
                        solver.add_lin_le([z[j], yi[j]], [1.0, -1.0], 0.0)

        elif tag.startswith("softmax:simplex:"):
            rowsize = int(con.meta["rowsize"])
            W = list(con.var_ids)
            assert len(W) % rowsize == 0
            for r in range(len(W) // rowsize):
                row = W[r * rowsize:(r + 1) * rowsize]
                solver.add_ge_zero(row)
                solver.add_sum_eq(row, 1.0)

        elif tag == "in:linpoly":
            # Input specification: A·x ≤ b (linear polytope constraint)
            A = to_numpy(con.meta["A"])
            b = to_numpy(con.meta["b"])
            vids = list(con.var_ids)
            for i in range(A.shape[0]):
                solver.add_lin_le(vids, list(A[i, :]), float(b[i]))

        else:
            # 其他 tag 暂时忽略
            pass

    # ------------------------------------------------------------------
    # 4.5) Debug：哪一些变量只出现在 box / 只出现在约束？
    # ------------------------------------------------------------------
    only_box = sorted(used_in_box - used_in_cons)
    only_cons = sorted(used_in_cons - used_in_box)
    print(f"[DEBUG] vars only in boxes (count={len(only_box)}): sample={only_box[:20]}")
    print(f"[DEBUG] vars only in constraints (count={len(only_cons)}): sample={only_cons[:20]}")

    # ------------------------------------------------------------------
    # 5) objective (optional) —— 这里加“假目标”防止 UNBOUNDED
    # ------------------------------------------------------------------
    if objective is None:
        # 如果有 box，就找一个有 box 的变量做 dummy 目标
        if boxes:
            v0 = min(boxes.keys())
            solver.set_objective_linear([v0], [1.0], 0.0, "min")
        else:
            # 完全没有 box 的话，只能退回常数目标（此时 UNBOUNDED 还是可能的）
            solver.set_objective_linear([], [], 0.0, "min")
    else:
        c, c0 = objective
        vids = list(range(len(c)))
        coeffs = [float(ci) for ci in c]
        print(f"[DEBUG] objective: linear over {len(vids)} vars, sense={sense}")
        solver.set_objective_linear(vids, coeffs, float(c0), sense)

    return nvars
