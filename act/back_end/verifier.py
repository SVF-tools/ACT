#===- act/back_end/verifier.py - Spec-free Verification Engine ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Spec-free, input-free verification (single-shot).
#   Assumes ACT Net already encodes input and output specifications via
#   INPUT_SPEC and ASSERT layers produced by torch2act.TorchToACT.
#
# Architecture:
#   1. Extract seed bounds and constraints from INPUT_SPEC layers
#   2. Create entry_fact (Fact with bounds + all constraints)
#   3. Pass entry_fact to analyze() for abstract interpretation
#   4. Export all constraints via export_to_solver() (includes LIN_POLY)
#   5. Add negated ASSERT property and solve
#
#===---------------------------------------------------------------------===#

# Public API:
#   - verify_once(net, solver, timelimit=None) -> VerifResult
#
# Notes:
#   * Spec-free verification: all constraints extracted from ACT Net layers.
#   * Returns counterexample as torch.Tensor if FALSIFIED.
#   * Caller validates counterexamples using model inference (model_factory).
#   * INPUT_SPEC constraints (including LIN_POLY) are propagated through analyze().

from __future__ import annotations
import time
import heapq
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any

import numpy as np
import torch

# ACT backend imports
from act.back_end.core import Bounds, Con, ConSet, Fact
from act.back_end.solver.solver_base import Solver, SolveStatus
from act.back_end.layer_schema import LayerKind
from act.back_end.utils import validate_constraints

# Front-end enums (kinds)
from act.front_end.specs import InKind, OutKind

# -----------------------------------------------------------------------------
# Verification Status and Results
# -----------------------------------------------------------------------------

class VerifStatus:
    """Verification result status codes."""
    CERTIFIED = "CERTIFIED"      # Property proven safe
    FALSIFIED = "FALSIFIED"      # Property violated (counterexample found)
    UNKNOWN = "UNKNOWN"          # Inconclusive result

@dataclass
class VerifResult:
    """Verification result with optional counterexample input."""
    status: str                                    # CERTIFIED | FALSIFIED | UNKNOWN
    counterexample: Optional[torch.Tensor] = None  # Input tensor (only if FALSIFIED)
    stats: Dict[str, Any] = field(default_factory=dict)  # Solver metadata

# -----------------------------------------------------------------------------
# ACT Net extraction helpers
# -----------------------------------------------------------------------------

def find_entry_layer_id(net) -> int:
    """Return the id of the single INPUT layer."""
    candidates = [L.id for L in net.layers if L.kind == LayerKind.INPUT.value]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one INPUT layer, found {len(candidates)}.")
    return candidates[0]

def get_input_ids(net) -> List[int]:
    """Return input variable IDs (out_vars of INPUT layer)."""
    entry = find_entry_layer_id(net)
    return list(net.by_id[entry].out_vars)

def get_output_ids(net) -> List[int]:
    """Return output variable IDs (in_vars of ASSERT layer)."""
    assert_layer = net.layers[-1]
    if assert_layer.kind != LayerKind.ASSERT.value:
        raise ValueError("Expected last layer to be ASSERT.")
    return list(assert_layer.in_vars)

def gather_input_spec_layers(net):
    """Return list of INPUT_SPEC layers."""
    return [L for L in net.layers if L.kind == LayerKind.INPUT_SPEC.value]

def get_assert_layer(net):
    """Return the ASSERT layer (must be last)."""
    assert_layer = net.layers[-1]
    if assert_layer.kind != LayerKind.ASSERT.value:
        raise ValueError("Expected last layer to be ASSERT.")
    return assert_layer

# -----------------------------------------------------------------------------
# Seed and input spec helpers
# -----------------------------------------------------------------------------

def seed_from_input_specs(spec_layers) -> Bounds:
    """
    Create seed Bounds from INPUT_SPEC layers.
    Prefers BOX, then LINF_BALL, raises if only LIN_POLY exists.
    
    Note: This extracts only box bounds for seeding abstract interpretation.
    All constraints (including LIN_POLY) are added via add_all_input_specs().
    """
    # BOX first
    for L in spec_layers:
        if L.meta.get("kind") == InKind.BOX and "lb" in L.params and "ub" in L.params:
            return Bounds(L.params["lb"].clone(), L.params["ub"].clone())
    
    # LINF_BALL next
    for L in spec_layers:
        if L.meta.get("kind") == InKind.LINF_BALL:
            if "lb" in L.params and "ub" in L.params:
                return Bounds(L.params["lb"].clone(), L.params["ub"].clone())
            center = L.params.get("center")
            eps = L.meta.get("eps")
            if center is not None and eps is not None:
                e = torch.tensor(eps, dtype=center.dtype, device=center.device)
                return Bounds(center - e, center + e)
    
    # LIN_POLY only -> error
    if any(L.meta.get("kind") == InKind.LIN_POLY for L in spec_layers):
        raise ValueError("LIN_POLY requires a seed box (BOX or LINF_BALL).")
    
    # No usable spec at all
    raise ValueError("No valid input specification found for seeding.")


def add_all_input_specs(globalC: ConSet, input_ids: List[int], spec_layers) -> None:
    """
    Add all INPUT_SPEC constraints to constraint set.
    
    This function adds:
    - BOX constraints (box bounds)
    - LINF_BALL constraints (converted to box)
    - LIN_POLY constraints (linear polytope A·x ≤ b)
    
    The LIN_POLY constraints are tagged with "in:linpoly" and will be
    exported to the solver via export_to_solver() in cons_exportor.py.
    """
    for L in spec_layers:
        k = L.meta.get("kind")
        if k == InKind.BOX:
            lb = L.params["lb"].flatten()
            ub = L.params["ub"].flatten()
            globalC.add_box(-1, input_ids, Bounds(lb, ub))
        elif k == InKind.LINF_BALL:
            if "lb" in L.params and "ub" in L.params:
                lb = L.params["lb"].flatten()
                ub = L.params["ub"].flatten()
                globalC.add_box(-1, input_ids, Bounds(lb, ub))
            else:
                center = L.params["center"]
                eps = L.meta["eps"]
                e = torch.tensor(eps, dtype=center.dtype, device=center.device)
                lb = (center - e).flatten()
                ub = (center + e).flatten()
                globalC.add_box(-1, input_ids, Bounds(lb, ub))
        elif k == InKind.LIN_POLY:
            A, b = L.params["A"], L.params["b"]
            globalC.replace(Con("INEQ", tuple(input_ids), {"tag": "in:linpoly", "A": A, "b": b}))
        else:
            raise NotImplementedError(f"Unsupported INPUT_SPEC kind: {k}")


def input_satisfies_specs(x_np: np.ndarray, spec_layers, tol: float = 1e-7) -> bool:
    """
    Check whether a candidate input satisfies all INPUT_SPEC layers.
    Supports BOX, LINF_BALL, and LIN_POLY.
    """
    x = torch.from_numpy(x_np).reshape(-1)

    for L in spec_layers:
        k = L.meta.get("kind")
        if k == InKind.BOX:
            lb = L.params["lb"].flatten()
            ub = L.params["ub"].flatten()
            if torch.any(x < lb - tol) or torch.any(x > ub + tol):
                return False

        elif k == InKind.LINF_BALL:
            if "lb" in L.params and "ub" in L.params:
                lb = L.params["lb"].flatten()
                ub = L.params["ub"].flatten()
            else:
                center = L.params["center"].flatten()
                eps = torch.tensor(L.meta["eps"], dtype=center.dtype, device=center.device)
                lb = center - eps
                ub = center + eps
            if torch.any(x < lb - tol) or torch.any(x > ub + tol):
                return False

        elif k == InKind.LIN_POLY:
            A, b = L.params["A"], L.params["b"]
            x_flat = x.to(dtype=A.dtype, device=A.device, non_blocking=True)
            lhs = torch.mv(A, x_flat)
            if torch.any(lhs > b + tol):
                return False

        else:
            raise NotImplementedError(f"Unsupported INPUT_SPEC kind: {k}")

    return True


def check_violation_at_point(net, x_np: np.ndarray, assert_layer) -> bool:
    """
    Evaluate the ACT Net at a point and check if the ASSERT property is violated.
    Used to filter spurious counterexamples after solver returns SAT.
    """
    from act.back_end.analyze import analyze

    x_tensor = torch.from_numpy(x_np)
    point_bounds = Bounds(x_tensor, x_tensor)
    entry_fact = Fact(bounds=point_bounds, cons=ConSet())

    entry_id = find_entry_layer_id(net)
    _, after, _ = analyze(net, entry_id, entry_fact)

    output_layer_id = net.layers[-2].id  # Layer before ASSERT
    y_bounds = after[output_layer_id].bounds
    y_mid = ((y_bounds.lb + y_bounds.ub) / 2).cpu().numpy()

    k = assert_layer.meta.get("kind")
    if k == OutKind.TOP1_ROBUST:
        t = int(assert_layer.meta["y_true"])
        others = [i for i in range(len(y_mid)) if i != t]
        return (y_mid[others] - y_mid[t]).max() >= 0.0
    elif k == OutKind.MARGIN_ROBUST:
        t = int(assert_layer.meta["y_true"])
        margin = float(assert_layer.meta["margin"])
        others = [i for i in range(len(y_mid)) if i != t]
        return (y_mid[others] - y_mid[t]).max() >= margin
    elif k == OutKind.LINEAR_LE:
        c = np.asarray(assert_layer.params["c"], dtype=float)
        d = float(assert_layer.meta["d"])
        return float(np.dot(c, y_mid)) >= d + 1e-8
    elif k == OutKind.RANGE:
        lb = assert_layer.params.get("lb")
        ub = assert_layer.params.get("ub")
        if lb is not None and np.any(y_mid < np.asarray(lb, dtype=float) - 1e-8):
            return True
        if ub is not None and np.any(y_mid > np.asarray(ub, dtype=float) + 1e-8):
            return True
        return False
    else:
        raise NotImplementedError(f"ASSERT kind not supported: {k}")

def _upper_bound_violation(out_bounds: Optional[Bounds], t: int, margin: float = 0.0) -> float:
    """
    给 TOP1 / MARGIN_ROBUST 里的 violation 变量 v 一个有限的上界：

        TOP1:   v >= max_j (y[j] - y[t])
        MARGIN: v >= max_j (y[j] - y[t] + margin)

    用抽象 bounds 做个粗上界：
        y[j] ∈ [lb_j, ub_j], y[t] ∈ [lb_t, ub_t]
        ⇒ y[j] - y[t] ≤ ub_j - lb_t

    所以：
        TOP1:   v_max = max_j (ub_j - lb_t)
        MARGIN: v_max = max_j (ub_j - lb_t + margin)

    如果拿不到 bounds，就用一个保守常数。
    """
    if out_bounds is None:
        return 1e3  # fallback：保守常数，避免无界

    lb = out_bounds.lb.detach().cpu().numpy().reshape(-1)
    ub = out_bounds.ub.detach().cpu().numpy().reshape(-1)
    
    import numpy as np
    print("[DEBUG] ub finite?", np.isfinite(ub).all(), 
          "ub[min,max]=", np.nanmin(ub), np.nanmax(ub))

    lb_t = float(lb[t])
    # 对每个 j 的上界：
    diff = ub - lb_t + margin
    v_max = float(diff.max())
    # 至少给一点正数，避免 v_max 为 0 或负数导致数值奇怪
    return max(v_max, 1e-3)

def add_negated_assert_to_solver(
    solver: Solver,
    out_ids: List[int],
    assert_layer,
    out_bounds: Optional[Bounds] = None,
):
    """
    Add the *negation* of ASSERT property as constraints to solver.

    返回:
        - objective: dict 或 None
          如果需要一个标量“违例变量”并设置目标，则返回 {"var": idx, "sense": "max" 或 "min"}。
    """
    from act.back_end.cons_exportor import to_numpy
    k = assert_layer.meta.get("kind")
    objective = None

    # ---------------- LINEAR_LE ----------------
    if k == OutKind.LINEAR_LE:
        # Property: c·y ≤ d
        # Negation: c·y ≥ d + ε
        coeffs = list(to_numpy(assert_layer.params["c"]))
        d = float(assert_layer.meta["d"])
        solver.add_lin_ge(out_ids, coeffs, d + 1e-6)

    # ---------------- TOP1_ROBUST ----------------
    elif k == OutKind.TOP1_ROBUST:
        # Property: y[t] > y[j] for all j≠t
        # Negation (存在反例): ∃j: y[j] - y[t] ≥ 0
        #
        # 我们不用显式的离散“选 j”，而是只加一个连续 witness 变量 v，
        # 再用约束把 v 约束成:
        #   v >= y[j] - y[t]    ∀j≠t
        #   v >= 0
        #   v <= v_max          (v_max 从抽象 out_bounds 推出，只是为了避免 unbounded)
        #
        # 这样在 SAT 情况下我们可以读出一个 candidate input，再用真实网络检查。
        t = int(assert_layer.meta["y_true"])

        # 1) 新建违例变量 v（不能复用任何现有变量索引）
        v = solver.n
        solver.add_vars(1)

        # 2) 给 v 一个有限上界（仅用于数值稳定 / 防止无界）
        v_max = _upper_bound_violation(out_bounds, t, margin=0.0)

        # v >= 0
        solver.add_lin_ge([v], [1.0], 0.0)
        # v <= v_max
        solver.add_lin_le([v], [1.0], float(v_max))

        # 3) 对所有 j≠t 施加: v >= y[j] - y[t]
        #    即：v - y[j] + y[t] >= 0
        yt = out_ids[t]
        for j, oj in enumerate(out_ids):
            if j == t:
                continue
            solver.add_lin_ge(
                [v, oj, yt],
                [1.0, -1.0, 1.0],
                0.0,
            )

        # 把目标交给外面统一设置
        objective = {"var": v, "sense": "max"}

    # ---------------- MARGIN_ROBUST ----------------
    elif k == OutKind.MARGIN_ROBUST:
        # Property: y[t] - y[j] > margin  ∀j≠t
        # Negation: ∃j: y[j] - y[t] + margin ≥ 0
        #
        # 形式同上，只是把 diff 换成 (y[j] - y[t] + margin)
        t = int(assert_layer.meta["y_true"])
        margin = float(assert_layer.meta["margin"])

        v = solver.n
        solver.add_vars(1)

        v_max = _upper_bound_violation(out_bounds, t, margin=margin)

        # v >= 0
        solver.add_lin_ge([v], [1.0], 0.0)
        # v <= v_max
        solver.add_lin_le([v], [1.0], float(v_max))

        yt = out_ids[t]
        for j, oj in enumerate(out_ids):
            if j == t:
                continue
            # v >= y[j] - y[t] + margin
            # v - y[j] + y[t] >= margin
            solver.add_lin_ge(
                [v, oj, yt],
                [1.0, -1.0, 1.0],
                margin,
            )

        objective = {"var": v, "sense": "max"}

    # ---------------- RANGE ----------------
    elif k == OutKind.RANGE:
        # Property: lb ≤ y ≤ ub
        # Negation（当前只编码 y > ub 支路，保持与原始实现一致）:
        #   ∃i: y[i] > ub[i]
        ub = assert_layer.params.get("ub")
        if ub is not None:
            for i, yi in enumerate(out_ids):
                solver.add_lin_ge([yi], [1.0], float(ub[i].item()) + 1e-6)

    else:
        raise NotImplementedError(f"Unsupported ASSERT kind: {k}")

    return objective


# def add_negated_assert_to_solver(solver: Solver, out_ids: List[int], assert_layer, out_bounds: Optional[Bounds] = None,):
#     """
#     Add the negation of ASSERT property as constraints to solver.
#     Returns optional objective info for disjunctive properties.
#     """
#     from act.back_end.cons_exportor import to_numpy
#     k = assert_layer.meta.get("kind")
#     objective = None
    
#     if k == OutKind.LINEAR_LE:
#         # Property: c·y ≤ d  →  Negation: c·y ≥ d + ε
#         coeffs = list(to_numpy(assert_layer.params["c"]))
#         d = float(assert_layer.meta["d"])
#         solver.add_lin_ge(out_ids, coeffs, d + 1e-6)
        
#     elif k == OutKind.TOP1_ROBUST:
#         # Property: y[t] > y[j] for all j≠t
#         # Negation: ∃j: y[j] - y[t] ≥ 0
#         #
#         # Encode:
#         #   v >= y[j] - y[t]    ∀j≠t
#         #   v >= 0
#         #   v <= v_max          (from abstract bounds)
#         #
#         # Maximize v:
#         #   v* > 0  → violation exists
#         #   v* ≤ 0  → no violation
#         t = int(assert_layer.meta["y_true"])
#         v = solver.n
#         solver.add_vars(1)

#         v_max = _upper_bound_violation(out_bounds, t, margin=0.0)

#         # v >= 0
#         solver.add_lin_ge([v], [1.0], 0.0)
#         # v <= v_max  →  -v >= -v_max
#         solver.add_lin_ge([v], [-1.0], -v_max)

#         for j, oj in enumerate(out_ids):
#             if j != t:
#                 # v >= y[j] - y[t]
#                 # v - y[j] + y[t] >= 0
#                 solver.add_lin_ge([v, oj, out_ids[t]], [1.0, -1.0, 1.0], 0.0)

#         objective = {"var": v, "sense": "max"}

#     elif k == OutKind.MARGIN_ROBUST:
#         # Property: y[t] - y[j] > margin  ∀j≠t
#         # Negation: ∃j: y[j] - y[t] + margin ≥ 0
        
#         # Encode:
#         #   v >= y[j] - y[t] + margin    ∀j≠t
#         #   v >= 0
#         #   v <= v_max                   (from bounds, +margin)
        
#         # Maximize v:
#         #   v* > 0  → violation exists
#         #   v* ≤ 0  → no violation
#         t = int(assert_layer.meta["y_true"])
#         margin = float(assert_layer.meta["margin"])
#         v = solver.n
#         solver.add_vars(1)

#         v_max = _upper_bound_violation(out_bounds, t, margin=margin)

#         # v >= 0
#         solver.add_lin_ge([v], [1.0], 0.0)
#         # v <= v_max  →  -v >= -v_max
#         solver.add_lin_ge([v], [-1.0], -v_max)

#         for j, oj in enumerate(out_ids):
#             if j != t:
#                 # v >= y[j] - y[t] + margin
#                 # v - y[j] + y[t] >= margin
#                 solver.add_lin_ge(
#                     [v, oj, out_ids[t]],
#                     [1.0, -1.0, 1.0],
#                     margin,
#                 )

#         objective = {"var": v, "sense": "max"}
        
#     elif k == OutKind.RANGE:
#         # Property: lb ≤ y ≤ ub  →  Negation: y > ub (or y < lb)
#         ub = assert_layer.params.get("ub")
#         if ub is not None:
#             for i, yi in enumerate(out_ids):
#                 solver.add_lin_ge([yi], [1.0], float(ub[i].item()) + 1e-6)
#     else:
#         raise NotImplementedError(f"Unsupported ASSERT kind: {k}")

#     return objective
    
    # if k == OutKind.LINEAR_LE:
    #     # Property: c·y ≤ d  →  Negation: c·y ≥ d + ε
    #     coeffs = list(to_numpy(assert_layer.params["c"]))
    #     d = float(assert_layer.meta["d"])
    #     solver.add_lin_ge(out_ids, coeffs, d + 1e-6)
        
    # elif k == OutKind.TOP1_ROBUST:
    #     # Property: y[t] > y[j] for all j≠t  →  Negation: ∃j: y[j] ≥ y[t]
    #     t = int(assert_layer.meta["y_true"])
    #     v = solver.n
    #     solver.add_vars(1)
    #     for j, oj in enumerate(out_ids):
    #         if j != t:
    #             solver.add_lin_ge([v, oj, out_ids[t]], [1.0, -1.0, 1.0], 0.0)
    #     solver.add_lin_ge([v], [1.0], 0.0)  # v >= 0 to witness violation
    #     objective = {"var": v, "sense": "max"}
        
    # elif k == OutKind.MARGIN_ROBUST:
    #     # Property: y[t] - y[j] > margin for all j≠t  →  Negation: ∃j: y[j] ≥ y[t] - margin
    #     t = int(assert_layer.meta["y_true"])
    #     margin = float(assert_layer.meta["margin"])
    #     v = solver.n
    #     solver.add_vars(1)
    #     for j, oj in enumerate(out_ids):
    #         if j != t:
    #             solver.add_lin_ge([v, oj, out_ids[t]], [1.0, -1.0, 1.0], margin)
    #     solver.add_lin_ge([v], [1.0], 0.0)
    #     objective = {"var": v, "sense": "max"}
        
    # elif k == OutKind.RANGE:
    #     # Property: lb ≤ y ≤ ub  →  Negation: y > ub (or y < lb)
    #     ub = assert_layer.params.get("ub")
    #     if ub is not None:
    #         for i, yi in enumerate(out_ids):
    #             solver.add_lin_ge([yi], [1.0], float(ub[i].item()) + 1e-6)
    # else:
    #     raise NotImplementedError(f"Unsupported ASSERT kind: {k}")

    # return objective


# -----------------------------------------------------------------------------
# Core solver workflow (shared by verify_once and BaB)
# -----------------------------------------------------------------------------

@torch.no_grad()
def setup_and_solve(
    net,
    input_bounds: Bounds,
    solver: Solver,
    timelimit: Optional[float] = None
) -> tuple[str, Optional[np.ndarray], Dict[str, Any]]:
    """
    Core verification workflow: setup constraints and solve.
    
    This function encapsulates the common verification pattern:
    1. Extract network structure (entry layer, input/output IDs, specs)
    2. Create entry_fact with input_bounds and all INPUT_SPEC constraints
    3. Run abstract interpretation (analyze)
    4. Export constraints to solver
    5. Add negated ASSERT property
    6. Solve and return status + counterexample (if found)
    
    Args:
        net: ACT network
        input_bounds: Input region bounds (seed box or refinement box)
        solver: Solver instance
        timelimit: Optional timeout in seconds
    
    Returns:
        Tuple of (status, counterexample_input, stats):
        - status: SolveStatus.SAT/UNSAT/UNKNOWN
        - counterexample_input: np.ndarray if SAT, else None
        - stats: Dict with metadata (ncons, status, etc.)
    """
    from act.back_end.analyze import analyze
    from act.back_end.cons_exportor import export_to_solver
    from act.back_end.cons_exportor import to_numpy

    
    # Extract network structure
    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    output_ids = get_output_ids(net)
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)
    
    out_bounds: Optional[Bounds] = None
    
    # Create entry_fact with ALL input constraints
    entry_fact = Fact(bounds=input_bounds, cons=ConSet())
    add_all_input_specs(entry_fact.cons, input_ids, spec_layers)
    
    # Analyze with full input specification (propagates constraints)
    before, after, globalC = analyze(net, entry_id, entry_fact)

    import logging
    logger = logging.getLogger(__name__)
    # Add output box from last hidden layer to bound objective
    try:
        assert_preds = net.preds.get(assert_layer.id, [])
        if assert_preds:
            last_hid = assert_preds[0]
            out_bounds = after[last_hid].bounds
            if out_bounds is not None:
                logger.info(
                    "[DEBUG] out_bounds: lb[min,max]=[%.4f, %.4f], ub[min,max]=[%.4f, %.4f]",
                    out_bounds.lb.min().item(), out_bounds.lb.max().item(),
                    out_bounds.ub.min().item(), out_bounds.ub.max().item(),
                )
                # ====== STEP B-1: 检查 out_bounds 内部是否存在 lb>ub ======
                bad = out_bounds.lb > out_bounds.ub + 1e-9
                if bad.any():
                    idx = bad.nonzero(as_tuple=False)
                    logger.error("🔥 [STEP B-1] out_bounds has lb>ub at %d dims", idx.shape[0])
                    # 打印前 20 个有问题的维度
                    for k in idx[:20]:
                        flat_idx = int(k.view(-1)[0].item())
                        logger.error(
                            "    dim %d: lb=%.6f, ub=%.6f",
                            flat_idx,
                            out_bounds.lb.view(-1)[flat_idx].item(),
                            out_bounds.ub.view(-1)[flat_idx].item(),
                        )
                # =======================================================
            else:
                logger.info("[DEBUG] out_bounds: None")
            
            # globalC.add_box(last_hid, output_ids, out_bounds.copy())
    except Exception:
        logger.exception("Error when adding output box / computing out_bounds")

    # Debug: report constraint sizes and bounds stats
    try:
        import logging
        logger = logging.getLogger(__name__)
        box_cons = [c for c in globalC if c.meta.get("tag", "").startswith("box:")]
        logger.info(
            "    [DEBUG] constraints total=%d boxes=%d input_ids=%d "
            "seed_lb[min,max]=[%.4f, %.4f] seed_ub[min,max]=[%.4f, %.4f]",
            len(globalC),
            len(box_cons),
            len(input_ids),
            input_bounds.lb.min().item(),
            input_bounds.lb.max().item(),
            input_bounds.ub.min().item(),
            input_bounds.ub.max().item(),
        )
    except Exception:
        pass
    
    # Validate constraints (validation runs if enabled, logging only if debug_tf also enabled)
    validate_constraints(globalC, after, net)
    
    # Export all constraints to solver (including LIN_POLY)
    export_to_solver(globalC, solver, objective=None, sense="min")
    
    # 🔒 强制把输入变量限制在 seed box 里
    try:
        from act.back_end.cons_exportor import to_numpy
        lb_np = to_numpy(input_bounds.lb).reshape(-1)
        ub_np = to_numpy(input_bounds.ub).reshape(-1)

        if lb_np.shape[0] != len(input_ids) or ub_np.shape[0] != len(input_ids):
            raise ValueError(
                f"Input bounds size mismatch in setup_and_solve: "
                f"lb={lb_np.shape[0]}, ub={ub_np.shape[0]}, input_ids={len(input_ids)}"
            )
        # 关键：这里会覆盖 export_to_solver 里给这些 var 设置的任何 bounds
        solver.set_bounds(input_ids, lb_np, ub_np)
        logger.info(
            "[DEBUG] enforce input box on solver: id[0]=%d, lb[0]=%.4f, ub[0]=%.4f",
            input_ids[0], lb_np[0], ub_np[0],
        )
    except Exception as e:
        logger.warning("Failed to set input bounds on solver: %s", e)
    
    # 🔒 关键补丁：确保输入变量真的被 seed_bounds 限制在 box 内
    try:
        from act.back_end.cons_exportor import to_numpy
        lb_np = to_numpy(input_bounds.lb).reshape(-1)
        ub_np = to_numpy(input_bounds.ub).reshape(-1)

        if lb_np.shape[0] != len(input_ids) or ub_np.shape[0] != len(input_ids):
            raise ValueError(
                f"Input bounds size mismatch in setup_and_solve: "
                f"lb={lb_np.shape[0]}, ub={ub_np.shape[0]}, input_ids={len(input_ids)}"
            )

        solver.set_bounds(input_ids, lb_np, ub_np)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Failed to set input bounds on solver: %s", e)
    
    obj_info = add_negated_assert_to_solver(solver, output_ids, assert_layer, out_bounds=out_bounds,)
    
    # Solve (feasibility check only)
    if obj_info:
        solver.set_objective_linear([obj_info["var"]], [1.0], 0.0, sense=obj_info.get("sense", "max"))
    else:
        solver.set_objective_linear([], [], 0.0, sense="min")
    solver.optimize(timelimit)
    
    # Extract result
    st = solver.status()
    ce_input = None
    if st == SolveStatus.SAT and solver.has_solution():
        ce_input = solver.get_values(input_ids)
    
    stats = {"status": st, "ncons": len(globalC)}
    # # === 新增：用目标值重解释 TOP1 / MARGIN 这种带 obj 的情况 ===
    # if obj_info is not None and st == SolveStatus.SAT:
    #     v_idx = obj_info.get("var")
    #     try:
    #         v_val = float(solver.get_values([v_idx])[0])
    #         stats["violation_var"] = v_val
    #     except Exception as e:
    #         stats["violation_var_error"] = str(e)
    # return st, ce_input, stats
    # === 用目标值重解释 TOP1 / MARGIN 这种带 obj 的情况 ===
    # if obj_info is not None and st == SolveStatus.SAT and solver.has_solution():
    #     v_idx = obj_info["var"]
    #     try:
    #         v_val = float(solver.get_values([v_idx])[0])
    #         stats["violation_var"] = v_val

    #         # 如果 v <= 0（加点容差），说明在抽象约束下
    #         #  max_j d_j ≤ 0  →  没有违反 ASSERT 的点
    #         if v_val <= 1e-6:
    #             st = SolveStatus.UNSAT
    #             ce_input = None
    #     except Exception as e:
    #         stats["violation_var_error"] = str(e)
    
    # 记录 violation_var（仅当有目标的情况：TOP1 / MARGIN / RANGE 里的某些实现）
    if obj_info is not None and solver.has_solution() and st == SolveStatus.SAT:
        v_idx = obj_info["var"]
        try:
            v_val = float(solver.get_values([v_idx])[0])
            stats["violation_var"] = v_val
        except Exception as e:
            stats["violation_var_error"] = str(e)

    return st, ce_input, stats


# -----------------------------------------------------------------------------
# Single-shot verification
# -----------------------------------------------------------------------------

@torch.no_grad()
def verify_once(net, solver: Solver, timelimit: Optional[float] = None) -> VerifResult:
    """
    Single-shot verification without refinement.
    Returns CERTIFIED/FALSIFIED/UNKNOWN with optional counterexample input.
    """
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)
    seed_bounds = seed_from_input_specs(spec_layers)
    
    # Sanity check: seed width matches input vars
    input_ids = get_input_ids(net)
    if seed_bounds.lb.numel() != len(input_ids) or seed_bounds.ub.numel() != len(input_ids):
        raise ValueError(
            f"Seed bounds/input_ids mismatch: lb={seed_bounds.lb.numel()}, "
            f"ub={seed_bounds.ub.numel()}, input_ids={len(input_ids)}"
        )
    
    # Core solver workflow
    status, ce_input, stats = setup_and_solve(net, seed_bounds, solver, timelimit)

    # Filter out spurious counterexamples using input specs and point evaluation
    if status == SolveStatus.SAT and ce_input is not None:
        stats.setdefault("ce_checks", {})
        stats["ce_checks"]["input_sat"] = bool(input_satisfies_specs(ce_input, spec_layers))
        stats["ce_checks"]["output_violated"] = bool(check_violation_at_point(net, ce_input, assert_layer))

        if (not input_satisfies_specs(ce_input, spec_layers)) or \
           (not check_violation_at_point(net, ce_input, assert_layer)):
            status = SolveStatus.UNKNOWN
            ce_input = None
    
    # Interpret result
    if status == SolveStatus.SAT and ce_input is not None:
        ce_x = torch.from_numpy(ce_input)
        return VerifResult(VerifStatus.FALSIFIED, counterexample=ce_x, stats=stats)
    
    if status == SolveStatus.UNSAT:
        return VerifResult(VerifStatus.CERTIFIED, stats=stats)
    
    return VerifResult(VerifStatus.UNKNOWN, stats=stats)

# @torch.no_grad()
# def verify_once(net, solver: Solver, timelimit: Optional[float] = None) -> VerifResult:
#     spec_layers = gather_input_spec_layers(net)
#     assert_layer = get_assert_layer(net)
#     seed_bounds = seed_from_input_specs(spec_layers)

#     input_ids = get_input_ids(net)
#     if seed_bounds.lb.numel() != len(input_ids) or seed_bounds.ub.numel() != len(input_ids):
#         raise ValueError(
#             f"Seed bounds/input_ids mismatch: lb={seed_bounds.lb.numel()}, "
#             f"ub={seed_bounds.ub.numel()}, input_ids={len(input_ids)}"
#         )

#     status, ce_input, stats = setup_and_solve(net, seed_bounds, solver, timelimit)
#     k = assert_layer.meta.get("kind")

#     # 1) 有可行解且拿到 CE：一定是 FALSIFIED（对所有性质都成立）
#     if status == SolveStatus.SAT and ce_input is not None:
#         ce_x = torch.from_numpy(ce_input)
#         return VerifResult(VerifStatus.FALSIFIED, counterexample=ce_x, stats=stats)

#     # 2) TOP1_ROBUST / MARGIN_ROBUST：用 violation_var 来解释
#     # if k in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
#     #     v_val = stats.get("violation_var", None)

#     #     # 有解并且我们拿到 v 的值
#     #     if status == SolveStatus.SAT and v_val is not None:
#     #         # v*>0 ⇒ 存在 j: y[j]-y[t](+margin) ≥ 0 ⇒ 有反例
#     #         if v_val > 1e-6:
#     #             # 理论上应该已经在上面的 CE 分支返回了，
#     #             # 如果 ce_input 恰好是 None，我们也至少标成 FALSIFIED（但没有具体 CE）
#     #             return VerifResult(VerifStatus.FALSIFIED, stats=stats)
#     #         else:
#     #             # v*≤0 ⇒ ∀j: y[j]-y[t](+margin) ≤ 0 ⇒ 性质在抽象域下被证明安全
#     #             return VerifResult(VerifStatus.CERTIFIED, stats=stats)

#     #     # 只要没有 SAT+v，就不敢宣称 CERTIFIED（包含 UNSAT/UNKNOWN）
#     #     return VerifResult(VerifStatus.UNKNOWN, stats=stats)
#     if k in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
#         # ✅ 1) 否定性质不可行 → 没反例 → CERTIFIED
#         if status == SolveStatus.UNSAT:
#             return VerifResult(VerifStatus.CERTIFIED, stats=stats)

#         v_val = stats.get("violation_var", None)

#         # ✅ 2) 有解 + 有 v 值：用 v 解释 SAT
#         if status == SolveStatus.SAT and v_val is not None:
#             if v_val > 1e-6:
#                 # 有解但可能没拿到 ce_input，也能标成 FALSIFIED
#                 ce_x = torch.from_numpy(ce_input) if ce_input is not None else None
#                 return VerifResult(VerifStatus.FALSIFIED, counterexample=ce_x, stats=stats)
#             else:
#                 return VerifResult(VerifStatus.CERTIFIED, stats=stats)

#         # ❓ 3) 其他奇怪情况（比如 Gurobi UNKNOWN / TIME_LIMIT）
#         return VerifResult(VerifStatus.UNKNOWN, stats=stats)


#     # 3) 线性性质（LINEAR_LE / RANGE 等）：沿用原来的语义
#     if status == SolveStatus.UNSAT:
#         return VerifResult(VerifStatus.CERTIFIED, stats=stats)

#     if status == SolveStatus.SAT and ce_input is not None:
#         ce_x = torch.from_numpy(ce_input)
#         return VerifResult(VerifStatus.FALSIFIED, counterexample=ce_x, stats=stats)

#     return VerifResult(VerifStatus.UNKNOWN, stats=stats)

