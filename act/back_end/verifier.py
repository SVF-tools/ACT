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
    
        raise ValueError("No valid input specification found for seeding.")
    # Flatten bounds to match flat input_ids
    return Bounds(seed.lb.flatten(), seed.ub.flatten())

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

def add_negated_assert_to_solver(solver: Solver, out_ids: List[int], assert_layer):
    """
    Add the negation of ASSERT property as constraints to solver.
    Returns optional objective info for disjunctive properties.
    """
    from act.back_end.cons_exportor import to_numpy
    k = assert_layer.meta.get("kind")
    objective = None
    
    if k == OutKind.LINEAR_LE:
        # Property: c·y ≤ d  →  Negation: c·y ≥ d + ε
        coeffs = list(to_numpy(assert_layer.params["c"]))
        d = float(assert_layer.meta["d"])
        solver.add_lin_ge(out_ids, coeffs, d + 1e-6)
        
    elif k == OutKind.TOP1_ROBUST:
        # Property: y[t] > y[j] for all j≠t  →  Negation: ∃j: y[j] ≥ y[t]
        t = int(assert_layer.meta["y_true"])
        v = solver.n
        solver.add_vars(1)
        for j, oj in enumerate(out_ids):
            if j != t:
                solver.add_lin_ge([v, oj, out_ids[t]], [1.0, -1.0, 1.0], 0.0)
        solver.add_lin_ge([v], [1.0], 0.0)  # v >= 0 to witness violation
        objective = {"var": v, "sense": "max"}
        
    elif k == OutKind.MARGIN_ROBUST:
        # Property: y[t] - y[j] > margin for all j≠t  →  Negation: ∃j: y[j] ≥ y[t] - margin
        t = int(assert_layer.meta["y_true"])
        margin = float(assert_layer.meta["margin"])
        v = solver.n
        solver.add_vars(1)
        for j, oj in enumerate(out_ids):
            if j != t:
                solver.add_lin_ge([v, oj, out_ids[t]], [1.0, -1.0, 1.0], -margin)
        solver.add_lin_ge([v], [1.0], 0.0)
        objective = {"var": v, "sense": "max"}
        
    elif k == OutKind.RANGE:
        # Property: lb ≤ y ≤ ub  →  Negation: y > ub (or y < lb)
        ub = assert_layer.params.get("ub")
        if ub is not None:
            for i, yi in enumerate(out_ids):
                solver.add_lin_ge([yi], [1.0], float(ub[i].item()) + 1e-6)
    else:
        raise NotImplementedError(f"Unsupported ASSERT kind: {k}")

    return objective


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
    
    # Extract network structure
    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    output_ids = get_output_ids(net)
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)
    
    # Create entry_fact with ALL input constraints
    entry_fact = Fact(bounds=input_bounds, cons=ConSet())
    add_all_input_specs(entry_fact.cons, input_ids, spec_layers)
    
    # Analyze with full input specification (propagates constraints)
    before, after, globalC = analyze(net, entry_id, entry_fact)

    # Add output box from last hidden layer to bound objective
    try:
        assert_preds = net.preds.get(assert_layer.id, [])
        if assert_preds:
            last_hid = assert_preds[0]
            out_bounds = after[last_hid].bounds
            globalC.add_box(last_hid, output_ids, out_bounds.copy())
    except Exception:
        pass

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
    obj_info = add_negated_assert_to_solver(solver, output_ids, assert_layer)
    
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
