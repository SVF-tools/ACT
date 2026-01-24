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
from act.back_end.layer_util import validate_conset_ops
from act.util.device_manager import get_default_device, get_default_dtype

TANH_EPS = 1e-9
TANH_IDENTITY_WINDOW = 0.25  # treat tanh(x) ≈ x within this window
TANH_IDENTITY_TOL = 1e-6    # symmetric tolerance when approximating identity

def _tanh_value(x: float) -> float:
    return float(np.tanh(x))

def _tanh_derivative(x: float) -> float:
    t = np.tanh(x)
    return float(1.0 - t * t)

def _add_tanh_convex_segment(solver: Solver, yi: int, zi: int, lo: float, hi: float) -> None:
    if hi - lo <= TANH_EPS:
        return
    f_lo = _tanh_value(lo)
    f_hi = _tanh_value(hi)
    slope_sec = (f_hi - f_lo) / (hi - lo)
    intercept_sec = f_lo - slope_sec * lo
    solver.add_lin_le([zi, yi], [1.0, -float(slope_sec)], float(intercept_sec))
    slope_tan = _tanh_derivative(hi)
    intercept_tan = f_hi - slope_tan * hi
    solver.add_lin_ge([zi, yi], [1.0, -float(slope_tan)], float(intercept_tan))

def _add_tanh_concave_segment(solver: Solver, yi: int, zi: int, lo: float, hi: float) -> None:
    if hi - lo <= TANH_EPS:
        return
    f_lo = _tanh_value(lo)
    f_hi = _tanh_value(hi)
    slope_sec = (f_hi - f_lo) / (hi - lo)
    intercept_sec = f_lo - slope_sec * lo
    solver.add_lin_ge([zi, yi], [1.0, -float(slope_sec)], float(intercept_sec))

    slope_tan = _tanh_derivative(lo)
    intercept_tan = f_lo - slope_tan * lo
    solver.add_lin_le([zi, yi], [1.0, -float(slope_tan)], float(intercept_tan))

def _add_tanh_small_band(solver: Solver, yi: int, zi: int, lo: float, hi: float) -> None:
    diff = max(abs(_tanh_value(lo) - lo), abs(_tanh_value(hi) - hi))
    delta = max(diff, TANH_EPS)
    solver.add_lin_le([zi, yi], [1.0, -1.0], float(delta))
    solver.add_lin_ge([zi, yi], [1.0, -1.0], float(-delta))

def _add_tanh_pwl_k_segments(solver: Solver, yi: int, zi: int, lo: float, hi: float, K: int) -> None:
    """
    Add K-tangent PWL approximation for z = tanh(y) over [lo, hi].

    CORRECT APPROACH (avoiding under-approximation):
    - Use ONE GLOBAL SECANT for over/under-approximation (valid everywhere in [lo,hi])
    - Use K TANGENT LINES for tighter bounds at K sampling points (all valid everywhere)

    For convex region (x < 0):
      - Global secant as UPPER bound: z <= secant(y)
      - K tangents as LOWER bounds: z >= tangent_i(y) for all i

    For concave region (x > 0):
      - Global secant as LOWER bound: z >= secant(y)
      - K tangents as UPPER bounds: z <= tangent_i(y) for all i

    This ensures all constraints are valid over the entire range [lo, hi],
    avoiding the under-approximation bug from per-segment secants.
    """
    if not np.isfinite(lo) or not np.isfinite(hi):
        return

    # Handle degenerate case
    if hi - lo <= TANH_EPS:
        val = _tanh_value(0.5 * (hi + lo))
        solver.add_lin_ge([zi], [1.0], float(val))
        solver.add_lin_ge([zi], [-1.0], float(-val))
        return

    # Handle small range near zero where tanh(x) ≈ x
    max_abs = max(abs(lo), abs(hi))
    if max_abs <= TANH_IDENTITY_WINDOW:
        delta = max(abs(_tanh_value(lo) - lo), abs(_tanh_value(hi) - hi), TANH_IDENTITY_TOL)
        solver.add_lin_le([zi, yi], [1.0, -1.0], float(delta))
        solver.add_lin_ge([zi, yi], [1.0, -1.0], float(-delta))
        return

    # Global secant line (valid everywhere in [lo, hi])
    f_lo = _tanh_value(lo)
    f_hi = _tanh_value(hi)
    slope_global_sec = (f_hi - f_lo) / (hi - lo)
    intercept_global_sec = f_lo - slope_global_sec * lo

    # Handle different regions
    if hi <= -TANH_EPS:
        # Entirely in convex region (x < 0)
        # Secant is UPPER bound, tangents are LOWER bounds
        solver.add_lin_le([zi, yi], [1.0, -float(slope_global_sec)], float(intercept_global_sec))

        # Add K tangent lines as lower bounds (all valid in convex region)
        for k in range(K):
            t = lo + (k + 0.5) * (hi - lo) / K  # Sample point in k-th segment
            f_t = _tanh_value(t)
            slope_tan = _tanh_derivative(t)
            intercept_tan = f_t - slope_tan * t
            solver.add_lin_ge([zi, yi], [1.0, -float(slope_tan)], float(intercept_tan))

    elif lo >= TANH_EPS:
        # Entirely in concave region (x > 0)
        # Secant is LOWER bound, tangents are UPPER bounds
        solver.add_lin_ge([zi, yi], [1.0, -float(slope_global_sec)], float(intercept_global_sec))

        # Add K tangent lines as upper bounds (all valid in concave region)
        for k in range(K):
            t = lo + (k + 0.5) * (hi - lo) / K  # Sample point in k-th segment
            f_t = _tanh_value(t)
            slope_tan = _tanh_derivative(t)
            intercept_tan = f_t - slope_tan * t
            solver.add_lin_le([zi, yi], [1.0, -float(slope_tan)], float(intercept_tan))

    else:
        # Range crosses zero - DO NOT use per-region secants (they cause under-approximation)
        # Per-region secants are only valid in their respective regions, not globally
        # Using them together over-constrains the LP → UNSAT → false CERTIFIED

        # Instead: use global bounds + tangents only
        # tanh is monotonic, so global bounds are simply [tanh(lo), tanh(hi)]
        solver.add_lin_ge([zi], [1.0], float(f_lo))  # z >= tanh(lo)
        solver.add_lin_le([zi], [1.0], float(f_hi))  # z <= tanh(hi)

        # Add tangent constraints in negative part (convex region)
        # Tangents in convex region are global lower bounds
        if lo < -TANH_EPS:
            neg_hi = min(hi, -TANH_EPS)
            K_neg = max(1, K // 2)
            for k in range(K_neg):
                t = lo + (k + 0.5) * (neg_hi - lo) / K_neg
                f_t = _tanh_value(t)
                slope_tan = _tanh_derivative(t)
                intercept_tan = f_t - slope_tan * t
                solver.add_lin_ge([zi, yi], [1.0, -float(slope_tan)], float(intercept_tan))

        # Add tangent constraints in positive part (concave region)
        # Tangents in concave region are global upper bounds
        if hi > TANH_EPS:
            pos_lo = max(lo, TANH_EPS)
            K_pos = max(1, K - (K // 2))
            for k in range(K_pos):
                t = pos_lo + (k + 0.5) * (hi - pos_lo) / K_pos
                f_t = _tanh_value(t)
                slope_tan = _tanh_derivative(t)
                intercept_tan = f_t - slope_tan * t
                solver.add_lin_le([zi, yi], [1.0, -float(slope_tan)], float(intercept_tan))

        # If range is very close to zero, add small band constraint
        if abs(lo) < TANH_EPS and abs(hi) < TANH_EPS:
            _add_tanh_small_band(solver, yi, zi, lo, hi)

def _add_tanh_constraints_for_var(solver: Solver, yi: int, zi: int, lo: float, hi: float) -> None:
    """Legacy 2-tangent approximation (fallback when K is not specified)."""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return

    max_abs = max(abs(lo), abs(hi))
    if max_abs <= TANH_IDENTITY_WINDOW:
        delta = max(abs(_tanh_value(lo) - lo), abs(_tanh_value(hi) - hi), TANH_IDENTITY_TOL)
        solver.add_lin_le([zi, yi], [1.0, -1.0], float(delta))
        solver.add_lin_ge([zi, yi], [1.0, -1.0], float(-delta))
        return

    if hi - lo <= TANH_EPS:
        val = _tanh_value(0.5 * (hi + lo))
        solver.add_lin_ge([zi], [1.0], float(val))
        solver.add_lin_ge([zi], [-1.0], float(-val))
        return

    if hi <= -TANH_EPS:
        _add_tanh_convex_segment(solver, yi, zi, lo, hi)
        return
    if lo >= TANH_EPS:
        _add_tanh_concave_segment(solver, yi, zi, lo, hi)
        return

    added = False
    if lo < -TANH_EPS:
        neg_hi = min(hi, -TANH_EPS)
        _add_tanh_convex_segment(solver, yi, zi, lo, neg_hi)
        added = True
    if hi > TANH_EPS:
        pos_lo = max(lo, TANH_EPS)
        _add_tanh_concave_segment(solver, yi, zi, pos_lo, hi)
        added = True
    if not added:
        _add_tanh_small_band(solver, yi, zi, lo, hi)

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
    validate_conset_ops(globalC)
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
            lb = to_numpy(con.meta["lb"]); ub = to_numpy(con.meta["ub"])
            for i, vid in enumerate(con.var_ids):
                cur=boxes.get(vid, (-np.inf, +np.inf))
                boxes[vid]=(max(cur[0], float(lb[i])), min(cur[1], float(ub[i])))

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
            # W has shape (n_out, n_in), so we know the dimensions
            n_out, n_in = W.shape
            # Take the first n_out variables as outputs, the rest as inputs
            y = list(con.var_ids[:n_out])
            x = list(con.var_ids[n_out:])
            for i, yi in enumerate(y):
                solver.add_lin_eq([yi]+x, [1.0]+[-float(W[i,j]) for j in range(W.shape[1])], float(b[i]))

        elif tag.startswith("bias:"):
            n=len(con.var_ids)//2; y=list(con.var_ids[:n]); x=list(con.var_ids[n:])
            c=to_numpy(con.meta["c"])
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
            # LRELU: z = alpha * y when y < 0, so constraint is z - alpha*y = 0
            for i in to_numpy(meta["idx_off"]).astype(int): solver.add_lin_eq([z[i],y[i]],[1.0, -alpha],0.0)
            for i in to_numpy(meta["idx_amb"]).astype(int):
                solver.add_lin_le([y[i],z[i]],[ 1.0,-1.0],0.0)
                solver.add_lin_le([y[i],z[i]],[ alpha,-1.0],0.0)
            slope=to_numpy(meta["slope"]); shift=to_numpy(meta["shift"])
            for k, i in enumerate(to_numpy(meta["idx_amb"]).astype(int)):
                solver.add_lin_le([z[i], y[i]], [1.0, -float(slope[k])], float(shift[k]))

        elif tag.startswith("tanh:"):
            n = len(con.var_ids) // 2
            z_vars = list(con.var_ids[:n])
            y_vars = list(con.var_ids[n:])

            # Extract K from PWL metadata if available
            segs = con.meta.get("segs", {})
            K = segs.get("K", None)

            for zi, yi in zip(z_vars, y_vars):
                bounds = boxes.get(yi)
                if bounds is None:
                    continue
                lo, hi = bounds

                # Use K-segment PWL if K is specified, otherwise use legacy 2-tangent
                if K is not None and K >= 2:
                    _add_tanh_pwl_k_segments(solver, yi, zi, float(lo), float(hi), int(K))
                else:
                    _add_tanh_constraints_for_var(solver, yi, zi, float(lo), float(hi))

        elif tag.startswith("top1:"):
            meta = con.meta; y_vars = list(con.var_ids)
            t_idx  = int(to_numpy(meta["t_index"]).item()); v_id = int(meta["v_id"])
            margin = float(meta.get("margin", 0.0))
            for j, yj in enumerate(y_vars):
                if j == t_idx:
                    continue
                solver.add_lin_ge([v_id, yj, y_vars[t_idx]], [1.0, -1.0, 1.0], -margin)
                
        elif tag.startswith("range:"):
            meta = con.meta; v_id = con.var_ids[0]; y = list(con.var_ids[1:])
            lb = to_numpy(meta["lb"]).reshape(-1); ub = to_numpy(meta["ub"]).reshape(-1)
            solver.add_lin_ge([v_id], [1.0], 0.0)
            for j, yj in enumerate(y): 
                solver.add_lin_ge([v_id, yj], [1.0, 1.0], float(lb[j]))
                solver.add_lin_ge([v_id, yj], [1.0, -1.0], float(-ub[j]))

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
