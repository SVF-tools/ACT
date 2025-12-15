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
from scipy import stats
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
    # Use concrete PyTorch evaluation for robustness (avoid midpoint loss)
    from act.pipeline.verification.act2torch import ACTToTorch

    model = getattr(net, "_ce_eval_model", None)
    if model is None:
        model = ACTToTorch(net).run()
        model.eval()
        setattr(net, "_ce_eval_model", model)

    x_tensor = torch.from_numpy(x_np)
    entry_shape = net.by_id[find_entry_layer_id(net)].meta.get("shape")
    try:
        if entry_shape is not None:
            x_tensor = x_tensor.view(*entry_shape)
        else:
            x_tensor = x_tensor.view(1, -1)
    except Exception:
        x_tensor = x_tensor.view(1, -1)

    first_param = next(model.parameters(), None)
    if first_param is not None:
        x_tensor = x_tensor.to(dtype=first_param.dtype, device=first_param.device)

    with torch.no_grad():
        res = model(x_tensor)

    if isinstance(res, dict):
        return not bool(res.get("output_satisfied", False))

    # Fallback: raw tensor output
    y_mid = res.flatten().detach().cpu().numpy()
    k = assert_layer.meta.get("kind")
    if k == OutKind.TOP1_ROBUST:
        t = int(assert_layer.meta["y_true"])
        others = [i for i in range(len(y_mid)) if i != t]
        return (y_mid[others] - y_mid[t]).max() >= 0.0
    elif k == OutKind.MARGIN_ROBUST:
        t = int(assert_layer.meta["y_true"])
        margin = float(assert_layer.meta["margin"])
        others = [i for i in range(len(y_mid)) if i != t]
        return (y_mid[others] - y_mid[t] + margin).max() >= 0.0
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
    Provide a finite upper bound for the violation variable v in TOP1/MARGIN_ROBUST:

        TOP1:   v >= max_j (y[j] - y[t])
        MARGIN: v >= max_j (y[j] - y[t] + margin)

    Use abstract bounds for a coarse upper bound:
        y[j] ∈ [lb_j, ub_j], y[t] ∈ [lb_t, ub_t]
        ⇒ y[j] - y[t] ≤ ub_j - lb_t

    Therefore:
        TOP1:   v_max = max_j (ub_j - lb_t)
        MARGIN: v_max = max_j (ub_j - lb_t + margin)

    If bounds are unavailable, fall back to a conservative constant.
    """
    if out_bounds is None:
        return 1e6  # fallback: conservative constant to avoid unboundedness

    lb = out_bounds.lb.detach().cpu().numpy().reshape(-1)
    ub = out_bounds.ub.detach().cpu().numpy().reshape(-1)

    lb_t = float(lb[t])
    diff = ub - lb_t + margin
    v_max = float(np.nanmax(diff))
    if not np.isfinite(v_max):
        return 1e6
    return max(v_max, 1e3)

def add_negated_assert_to_solver(
    solver: Solver,
    out_ids: List[int],
    assert_layer,
    out_bounds: Optional[Bounds] = None,
):
    """
    Add the *negation* of the ASSERT property as constraints to the solver.

    Current version encodes feasibility only (SAT/UNSAT); no scalar violation
    objective is introduced:
      - LINEAR_LE:      c·y ≤ d        →  ¬prop: c·y ≥ d + ε
      - TOP1_ROBUST:    y[t] > y[j]    →  ¬prop: ∃j: y[j] - y[t] ≥ 0
      - MARGIN_ROBUST:  y[t] - y[j] > margin
                         →  ¬prop: ∃j: y[j] - y[t] + margin ≥ 0
      - RANGE:          lb ≤ y ≤ ub    →  ¬prop: ∃i: y[i] < lb[i] or y[i] > ub[i]

    For properties with an existential quantifier, use binary selector
    variables with big-M to encode the OR.
    Return value remains None for legacy compatibility and is unused.
    """
    from act.back_end.cons_exportor import to_numpy
    k = assert_layer.meta.get("kind")
    objective = None  # Legacy placeholder; always None now

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
        # Negation: ∃j: y[j] - y[t] ≥ 0
        t = int(assert_layer.meta["y_true"])
        v = solver.n
        solver.add_vars(1)
        v_max = _upper_bound_violation(out_bounds, t, margin=0.0)
        solver.add_lin_ge([v], [1.0], 0.0)       # v >= 0
        solver.add_lin_ge([v], [-1.0], -v_max)   # v <= v_max
        for j, oj in enumerate(out_ids):
            if j == t:
                continue
            # v >= y[j] - y[t]
            solver.add_lin_ge([v, oj, out_ids[t]], [1.0, -1.0, 1.0], 0.0)
        objective = {"var": v, "sense": "max"}

    # ---------------- MARGIN_ROBUST ----------------
    elif k == OutKind.MARGIN_ROBUST:
        # Property: y[t] - y[j] > margin  for all j != t
        # Negation: exists j: y[j] - y[t] + margin >= 0
        t = int(assert_layer.meta["y_true"])
        margin = float(assert_layer.meta["margin"])
        v = solver.n
        solver.add_vars(1)
        v_max = _upper_bound_violation(out_bounds, t, margin=margin)
        if not np.isfinite(v_max) or v_max < 1e-3:
            v_max = 1e6
        solver.add_lin_ge([v], [1.0], 0.0)       # v >= 0
        solver.add_lin_ge([v], [-1.0], -v_max)   # v <= v_max
        for j, oj in enumerate(out_ids):
            if j == t:
                continue
            # v >= y[j] - y[t] + margin
            solver.add_lin_ge(
                [v, oj, out_ids[t]],
                [1.0, -1.0, 1.0],
                margin,
            )
        objective = {"var": v, "sense": "max"}



    elif k == OutKind.RANGE:
        from act.back_end.cons_exportor import to_numpy

        lb_t = assert_layer.params.get("lb", None)
        ub_t = assert_layer.params.get("ub", None)
        if lb_t is None and ub_t is None:
            raise ValueError("RANGE assert requires lb and/or ub.")

        lb = None
        ub = None
        if lb_t is not None:
            lb = to_numpy(lb_t).reshape(-1)
        if ub_t is not None:
            ub = to_numpy(ub_t).reshape(-1)

        n_out = len(out_ids)
        if lb is not None and lb.shape[0] != n_out:
            raise ValueError(f"RANGE: lb length {lb.shape[0]} != len(out_ids)={n_out}")
        if ub is not None and ub.shape[0] != n_out:
            raise ValueError(f"RANGE: ub length {ub.shape[0]} != len(out_ids)={n_out}")

        # 1) Add violation variable v
        v = solver.n
        solver.add_vars(1)

        # 2) Estimate an upper bound v_max from output abstract bounds to avoid unboundedness
        #    y_i ∈ [y_lb_i, y_ub_i]
        #    lb_i - y_i ≤ lb_i - y_lb_i
        #    y_i - ub_i ≤ y_ub_i - ub_i
        v_max_terms = []

        if out_bounds is not None:
            y_lb = to_numpy(out_bounds.lb).reshape(-1)
            y_ub = to_numpy(out_bounds.ub).reshape(-1)

            if lb is not None:
                # Maximum possible lb_i - y_i
                v_max_terms.append(np.max(lb - y_lb))
            if ub is not None:
                # Maximum possible y_i - ub_i
                v_max_terms.append(np.max(y_ub - ub))
        # If bounds are unavailable, use a conservative constant
        v_max = max(v_max_terms) if v_max_terms else 1e6
        if (not np.isfinite(v_max)) or v_max < 1e-3:
            v_max = 1e6

        # 3) Constraints: 0 <= v <= v_max
        solver.add_lin_ge([v], [1.0], 0.0)        # v >= 0
        solver.add_lin_ge([v], [-1.0], -v_max)    # v <= v_max

        # 4) For each dimension i add v >= lb_i - y_i and/or v >= y_i - ub_i
        for i, yi in enumerate(out_ids):
            if lb is not None:
                # v >= lb_i - y_i   <=>  v + y_i >= lb_i
                solver.add_lin_ge(
                    [v, yi],
                    [1.0, 1.0],
                    float(lb[i]),
                )
            if ub is not None:
                # v >= y_i - ub_i   <=>  v - y_i >= -ub_i
                solver.add_lin_ge(
                    [v, yi],
                    [1.0, -1.0],
                    float(-ub[i]),
                )

        # Maximize v
        objective = {"var": v, "sense": "max"}

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
    
    Workflow:
      1. Extract entry/input/output/spec/assert from the ACT Net
      2. Build entry_fact from input_bounds plus all INPUT_SPEC constraints
      3. Run analyze() for abstract propagation to get globalC + layer bounds
      4. export_to_solver(globalC, solver)
      5. Add the negated ASSERT property (add_negated_assert_to_solver)
      6. Solve feasibility only and extract a CE input from the solver

    Returns:
        (status, counterexample_input, stats)
        - status: SolveStatus.SAT/UNSAT/UNKNOWN
        - counterexample_input: np.ndarray (only non-None when SAT with a solution)
        - stats: debug/statistics info
    """
    from act.back_end.analyze import analyze
    from act.back_end.cons_exportor import export_to_solver
    from act.back_end.cons_exportor import to_numpy
    import os

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

    # Optional debug/relaxation: drop internal box constraints (keep input seed)
    if os.environ.get("ACT_SKIP_INTERNAL_BOXES"):
        filtered = ConSet()
        for con in globalC:
            tag = con.meta.get("tag", "")
            layer_id = con.meta.get("layer_id", None)
            if not tag.startswith("box:"):
                filtered.replace(con)
            else:
                if layer_id is None:
                    try:
                        layer_id = int(str(tag).split("box:")[-1])
                    except Exception:
                        layer_id = 0
                if int(layer_id) < 0:
                    filtered.replace(con)
        globalC = filtered

    import logging
    logger = logging.getLogger(__name__)

    # Grab an output box from the ASSERT predecessor for big-M estimation
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
                # Extra sanity check: any dimensions with lb>ub?
                bad = out_bounds.lb > out_bounds.ub + 1e-9
                if bad.any():
                    idx = bad.nonzero(as_tuple=False)
                    logger.error("🔥 [STEP B-1] out_bounds has lb>ub at %d dims", idx.shape[0])
                    for k in idx[:20]:
                        flat_idx = int(k.view(-1)[0].item())
                        logger.error(
                            "    dim %d: lb=%.6f, ub=%.6f",
                            flat_idx,
                            out_bounds.lb.view(-1)[flat_idx].item(),
                            out_bounds.ub.view(-1)[flat_idx].item(),
                        )
            else:
                logger.info("[DEBUG] out_bounds: None")
    except Exception:
        logger.exception("Error when adding output box / computing out_bounds")

    # Debug: report constraint sizes and bounds stats
    try:
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

    # Enforce input variables inside the seed box (override any solver bounds)
    try:
        lb_np = to_numpy(input_bounds.lb).reshape(-1)
        ub_np = to_numpy(input_bounds.ub).reshape(-1)

        if lb_np.shape[0] != len(input_ids) or ub_np.shape[0] != len(input_ids):
            raise ValueError(
                f"Input bounds size mismatch in setup_and_solve: "
                f"lb={lb_np.shape[0]}, ub={ub_np.shape[0]}, input_ids={len(input_ids)}"
            )

        solver.set_bounds(input_ids, lb_np, ub_np)
        logger.info(
            "[DEBUG] enforce input box on solver: id[0]=%d, lb[0]=%.4f, ub[0]=%.4f",
            input_ids[0], lb_np[0], ub_np[0],
        )
    except Exception as e:
        logger.warning("Failed to set input bounds on solver: %s", e)

    logger.info(
    "Before call add_negated_assert_to_solver, ASSERT kind=%s, y_true=%s, margin=%s",
    assert_layer.meta.get("kind"),
    assert_layer.meta.get("y_true", None),
    assert_layer.meta.get("margin", None),
)

    # Add the negated ASSERT property (may return objective info)
    objective = add_negated_assert_to_solver(
        solver,
        output_ids,
        assert_layer,
        out_bounds=out_bounds,
    )
    
    logger.info(
    "After call add_negated_assert_to_solver, ASSERT kind=%s, y_true=%s, margin=%s",
    assert_layer.meta.get("kind"),
    assert_layer.meta.get("y_true", None),
    assert_layer.meta.get("margin", None),
)


    # Set objective if provided (e.g., violation variable for TOP1/MARGIN)
    if objective and "var" in objective:
        solver.set_objective_linear([objective["var"]], [1.0], 0.0, sense=objective.get("sense", "max"))
    else:
        solver.set_objective_linear([], [], 0.0, sense="min")
    solver.optimize(timelimit)

    # Extract result
    st = solver.status()
    ce_input = None
    if st == SolveStatus.SAT and solver.has_solution():
        ce_input = solver.get_values(input_ids)

    stats: Dict[str, Any] = {
        "status": st,
        "ncons": len(globalC),
    }
    if objective and "var" in objective and solver.has_solution():
        try:
            stats["violation_var"] = float(solver.get_values([objective["var"]])[0])
        except Exception:
            pass

    return st, ce_input, stats



# -----------------------------------------------------------------------------
# Single-shot verification
# -----------------------------------------------------------------------------

@torch.no_grad()
def verify_once(net, solver: Solver, timelimit: Optional[float] = None) -> VerifResult:
    """
    Single-shot verification without refinement.

    Semantics (uniform across ASSERT kinds):
      - SolveStatus.UNSAT  →  CERTIFIED (property proven with abstraction + MILP)
      - SolveStatus.SAT and CE passes secondary checks (input meets INPUT_SPEC and output truly violates)
                           →  FALSIFIED (returns counterexample as torch.Tensor)
      - Otherwise          →  UNKNOWN (no conclusion)
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

    # --- DEBUG: keep the solver's raw CE even if later discarded ---
    if ce_input is not None:
        try:
            stats["raw_ce_input"] = ce_input.copy()  # np.ndarray
        except Exception:
            stats["raw_ce_input"] = ce_input
    # ------------------------------------------------------------

    # Filter spurious CEs: must satisfy input spec and truly violate output property
    if status == SolveStatus.SAT and ce_input is not None:
        stats.setdefault("ce_checks", {})
        in_ok = input_satisfies_specs(ce_input, spec_layers)
        out_bad = check_violation_at_point(net, ce_input, assert_layer)

        stats["ce_checks"]["input_sat"] = bool(in_ok)
        stats["ce_checks"]["output_violated"] = bool(out_bad)

        if not (in_ok and out_bad):
            # SAT solution but CE not trustworthy → mark UNKNOWN and drop CE
            status = SolveStatus.UNKNOWN
            ce_input = None

    # Standardize status explanations
    if status == SolveStatus.SAT and ce_input is not None:
        print("[DEBUG VERIFY_ONCE] FINAL SAT+CE -> FALSIFIED")
        # Genuine counterexample
        ce_x = torch.from_numpy(ce_input)
        return VerifResult(VerifStatus.FALSIFIED, counterexample=ce_x, stats=stats)

    # Use violation_var (objective-based) to interpret status if available
    v_val = stats.get("violation_var", None)
    if v_val is not None:
        if float(v_val) > 1e-8:
            raw_ce = ce_input if ce_input is not None else stats.get("raw_ce_input", None)
            ce_x = torch.from_numpy(raw_ce) if raw_ce is not None else None
            return VerifResult(VerifStatus.FALSIFIED, counterexample=ce_x, stats=stats)
        else:
            return VerifResult(VerifStatus.CERTIFIED, stats=stats)

    if status == SolveStatus.UNSAT:
        # Negated property infeasible → original property proven
        return VerifResult(VerifStatus.CERTIFIED, stats=stats)

    # Fallback: if we have a validated CE from solver stats, treat as FALSIFIED
    ce_checks = stats.get("ce_checks", {})
    if ce_checks.get("input_sat", False) and ce_checks.get("output_violated", False):
        print("[DEBUG VERIFY_ONCE] FALLBACK CE_CHECKS -> FALSIFIED")
        raw_ce = ce_input
        if raw_ce is None:
            raw_ce = stats.get("raw_ce_input", None)
        if raw_ce is not None:
            try:
                ce_x = torch.from_numpy(raw_ce)
            except Exception:
                ce_x = torch.as_tensor(raw_ce)
        else:
            ce_x = None
        return VerifResult(
            status=VerifStatus.FALSIFIED,
            counterexample=ce_x,
            stats=stats,
        )

    # All other cases (UNKNOWN / TIME_LIMIT / NUMERIC ISSUE, etc.)
    return VerifResult(VerifStatus.UNKNOWN, stats=stats)


# -----------------------------------------------------------------------------
# Debug helper: fixed-input LP with simple linear property
# -----------------------------------------------------------------------------

@torch.no_grad()
def debug_fixed_input_lp(
    net,
    solver: Solver,
    x_ce: torch.Tensor,
    t: int,
    j_alt: int,
    timelimit: Optional[float] = None,
) -> tuple[str, Optional[np.ndarray], Dict[str, Any]]:
    """
    Debug-only helper:
      - Pin the input to a concrete counterexample x_ce (lb = ub = x_ce).
      - Replace ASSERT with a simple linear property y[j_alt] - y[t] <= 0,
        i.e., negation y[j_alt] - y[t] >= 0.
      - Build and solve the MILP; return status/ce_input/stats.
    """
    from types import SimpleNamespace
    from act.back_end.analyze import analyze
    from act.back_end.cons_exportor import export_to_solver, to_numpy

    # Flatten CE and build fixed bounds
    x_flat = x_ce.flatten()
    seed_bounds = Bounds(x_flat.clone(), x_flat.clone())

    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    output_ids = get_output_ids(net)
    assert_layer = get_assert_layer(net)

    # Synthetic ASSERT: property y[j_alt] - y[t] <= 0 (LINEAR_LE)
    c = torch.zeros(len(output_ids), dtype=x_flat.dtype, device=x_flat.device)
    c[j_alt] = 1.0
    c[t] = -1.0
    linear_assert = SimpleNamespace(
        meta={"kind": OutKind.LINEAR_LE, "d": 0.0},
        params={"c": c},
    )

    # Entry fact with only fixed box (skip other input specs to isolate network encoding)
    entry_fact = Fact(bounds=seed_bounds, cons=ConSet())

    before, after, globalC = analyze(net, entry_id, entry_fact)

    # Extract output bounds for big-M if available
    out_bounds: Optional[Bounds] = None
    try:
        assert_preds = net.preds.get(assert_layer.id, [])
        if assert_preds:
            last_hid = assert_preds[0]
            out_bounds = after[last_hid].bounds
    except Exception:
        out_bounds = None

    # Export constraints
    export_to_solver(globalC, solver, objective=None, sense="min")

    # Pin inputs exactly to CE
    x_np = to_numpy(x_flat).reshape(-1)
    solver.set_bounds(input_ids, x_np, x_np)

    # Add negated linear property
    add_negated_assert_to_solver(
        solver,
        output_ids,
        linear_assert,
        out_bounds=out_bounds,
    )

    solver.set_objective_linear([], [], 0.0, sense="min")
    solver.optimize(timelimit)

    st = solver.status()
    ce_input = None
    if st == SolveStatus.SAT and solver.has_solution():
        ce_input = solver.get_values(input_ids)

    stats: Dict[str, Any] = {
        "status": st,
        "ncons": len(globalC),
        "debug_mode": "fixed_input_lp",
    }

    return st, ce_input, stats
