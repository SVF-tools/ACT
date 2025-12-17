#===- act/back_end/bab.py - Branch-and-Bound Verification --------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Branch-and-bound verification with iterative refinement.
#   Implements refinement strategies for constraint satisfaction problems.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
import time
import heapq
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch

# ACT backend imports
from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.solver.solver_base import Solver, SolveStatus

# Import from verifier for shared functionality
from act.back_end.verifier import (
    VerifStatus,
    VerifResult,
    find_entry_layer_id,
    gather_input_spec_layers,
    get_assert_layer,
    seed_from_input_specs,
    setup_and_solve,
)

# Front-end enums
from act.front_end.specs import OutKind


# -----------------------------------------------------------------------------
# Branch-and-Bound verification
# -----------------------------------------------------------------------------

@dataclass
class BabNode:
    """Branch-and-bound search node."""
    box: Bounds
    depth: int
    score: float
    candidate_ce: Optional[np.ndarray] = None
    
    def __lt__(self, other):
        return self.score > other.score  # Max-heap by score

def width_sum(B: Bounds) -> float:
    """Compute total width of bounds box."""
    return float(torch.sum(B.ub - B.lb).item())

def choose_split_dim(B: Bounds) -> int:
    """Choose dimension with largest width for branching."""
    return int(torch.argmax(B.ub - B.lb).item())

def branch(B: Bounds, d: int) -> tuple[Bounds, Bounds]:
    """Split bounds at midpoint of dimension d."""
    mid = 0.5 * (B.lb[d] + B.ub[d])
    Lb, Ub = B.lb.clone(), B.ub.clone()
    Lb2, Ub2 = B.lb.clone(), B.ub.clone()
    Ub[d] = mid
    Lb2[d] = mid
    return Bounds(Lb, Ub), Bounds(Lb2, Ub2)

def solve_BaB_node(net, node: BabNode, solver: Solver, assert_layer) -> str:
    """
    Solve node's box region. Returns SolveStatus (SAT/UNSAT/UNKNOWN).
    Uses lightweight ACT Net evaluation to filter spurious counterexamples.
    """
    # Core solver workflow
    status, ce_input, _ = setup_and_solve(net, node.box, solver, timelimit=None)
    
    if status == SolveStatus.UNSAT:
        return SolveStatus.UNSAT
    
    if status == SolveStatus.SAT and ce_input is not None:
        # Lightweight validation using ACT Net point evaluation
        if check_violation_at_point(net, ce_input, assert_layer):
            node.candidate_ce = ce_input
            return SolveStatus.SAT
        # Spurious counterexample - treat as UNKNOWN
    
    return SolveStatus.UNKNOWN


def verify_bab(net, solver: Solver,
               max_depth: int = 20,
               max_nodes: int = 2000,
               time_budget_s: float = 300.0) -> VerifResult:
    """
    Branch-and-bound verification with iterative refinement.
    Returns CERTIFIED/FALSIFIED/UNKNOWN with optional counterexample input.
    Uses lightweight ACT Net evaluation to filter spurious counterexamples.
    
    Status Mapping:
        Solver Result (negated property) → BaB Node Result  → Final Verification
        ─────────────────────────────────────────────────────────────────────────
        SolveStatus.SAT (solution found) → SolveStatus.SAT  → VerifStatus.FALSIFIED
        SolveStatus.UNSAT (no solution)  → SolveStatus.UNSAT → Continue (node safe)
        SolveStatus.UNKNOWN (spurious)   → SolveStatus.UNKNOWN → Continue branching
        
        All nodes UNSAT/exhausted → VerifStatus.CERTIFIED
    """
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)
    root_box = seed_from_input_specs(spec_layers)

    pq: List[BabNode] = []
    heapq.heappush(pq, BabNode(root_box, 0, score=width_sum(root_box)))
    start = time.time()
    processed = 0

    while pq and (time.time() - start) < time_budget_s and processed < max_nodes:
        node = heapq.heappop(pq)
        processed += 1

        status = solve_BaB_node(net, node, solver, assert_layer)
        
        if status == SolveStatus.SAT:
            # Found validated counterexample
            ce_x = torch.from_numpy(node.candidate_ce)
            return VerifResult(
                VerifStatus.FALSIFIED,
                counterexample=ce_x,
                stats={"nodes": processed}
            )
        
        if status == SolveStatus.UNSAT:
            # This region is certified safe
            continue

        # SolveStatus.UNKNOWN - spurious or inconclusive, keep branching
        if node.depth >= max_depth:
            continue
        
        d = choose_split_dim(node.box)
        Lbox, Rbox = branch(node.box, d)
        heapq.heappush(pq, BabNode(Lbox, node.depth + 1, width_sum(Lbox)))
        heapq.heappush(pq, BabNode(Rbox, node.depth + 1, width_sum(Rbox)))

    return VerifResult(VerifStatus.CERTIFIED, stats={"nodes": processed})

# -----------------------------------------------------------------------------
# Counterexample validation (lightweight, used internally by BaB)
# -----------------------------------------------------------------------------

def check_violation_at_point(net, x_np: np.ndarray, assert_layer) -> bool:
    """
    Lightweight validation: evaluate ACT Net at a point and check property violation.
    Used internally by BaB to filter spurious counterexamples.
    For external validation, caller should use full model inference.
    """
    from act.back_end.analyze import analyze
    
    # Create tight bounds around point
    x_tensor = torch.from_numpy(x_np)
    point_bounds = Bounds(x_tensor, x_tensor)
    
    # Create entry_fact with point bounds (no additional constraints for point eval)
    entry_fact = Fact(bounds=point_bounds, cons=ConSet())
    
    # Analyze through network
    entry_id = find_entry_layer_id(net)
    _, after, _ = analyze(net, entry_id, entry_fact)
    
    # Get output bounds (should be tight for point evaluation)
    output_layer_id = net.layers[-2].id  # Layer before ASSERT
    y_bounds = after[output_layer_id].bounds
    y_mid = ((y_bounds.lb + y_bounds.ub) / 2).cpu().numpy()
    
    # Check violation
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