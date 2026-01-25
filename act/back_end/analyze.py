#===- act/back_end/analyze.py - Network Analysis Functions --------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Network analysis functions for ACT verification framework.
#   Provides analysis capabilities for neural network structures and properties.
#
#===---------------------------------------------------------------------===#

import torch
from collections import deque
from typing import Dict, Tuple
from act.back_end.core import Bounds, Fact, Net, ConSet
from act.back_end.utils import box_join, changed_or_maskdiff, update_cache
from act.back_end.transfer_functions import dispatch_tf, set_transfer_function_mode

# Initialize default transfer function mode
def initialize_tf_mode(mode: str = "interval"):
    """Initialize transfer function mode. Call this before using analyze()."""
    set_transfer_function_mode(mode)

@torch.no_grad()
def analyze(net: Net, entry_id: int, entry_fact: Fact, eps: float=1e-9) -> Tuple[Dict[int, Fact], Dict[int, Fact], ConSet]:
    """Perform abstract interpretation on the network starting from entry_fact.

    This implements a worklist-based forward abstract interpretation algorithm
    that propagates bounds through the network topology. The algorithm handles
    both sequential networks and DAGs with merge points (e.g., residual networks).

    DAG Handling (Problem 5 Review):
    --------------------------------
    For networks with skip connections (residual blocks), layers may have multiple
    predecessors. At merge points (e.g., ADD layers), bounds from all predecessors
    are joined using box_join (interval hull). The worklist algorithm ensures that:

    1. Layers are re-processed when any predecessor's bounds change
    2. Fixpoint is reached when no more changes propagate (convergence)
    3. Infinite bounds (±inf) are handled correctly during early iterations
       when some predecessors haven't been fully processed yet

    Note: The changed_or_maskdiff function handles nan values that can arise from
    operations like 0 * inf = nan during matmul with infinite bounds. It detects
    finiteness changes explicitly to ensure proper convergence.

    Args:
        net: ACT network structure containing layers and topology (preds/succs)
        entry_id: ID of the entry (INPUT) layer
        entry_fact: Initial Fact containing bounds and constraints for the input
        eps: Convergence epsilon for fixpoint iteration

    Returns:
        Tuple of (before, after, globalC):
          - before: Dict mapping layer_id -> Fact (pre-transfer bounds)
          - after: Dict mapping layer_id -> Fact (post-transfer bounds)
          - globalC: ConSet containing all collected constraints
    """
    # Auto-initialize transfer function mode if not set
    try:
        from act.back_end.transfer_functions import get_transfer_function
        get_transfer_function()  # Check if already initialized
    except RuntimeError:
        initialize_tf_mode("interval")  # Default to interval mode

    before: Dict[int, Fact] = {}
    after:  Dict[int, Fact] = {}
    globalC = ConSet()

    # Initialize all layers with infinite bounds (±inf).
    # These will be refined as the worklist propagates finite bounds.
    for L in net.layers:
        n = len(L.out_vars)
        hi = torch.full((n,), float("inf"), device=entry_fact.bounds.lb.device, dtype=entry_fact.bounds.lb.dtype)
        lo = torch.full((n,), -float("inf"), device=entry_fact.bounds.lb.device, dtype=entry_fact.bounds.lb.dtype)
        before[L.id] = Fact(bounds=Bounds(lo.clone(), hi.clone()), cons=ConSet())
        after[L.id]  = Fact(bounds=Bounds(lo.clone(), hi.clone()), cons=ConSet())
        L.cache.clear()

    # Seed entry with provided Fact (includes all input constraints)
    before[entry_id] = entry_fact

    # Worklist algorithm: process layers in order, re-add successors on change
    WL = deque([entry_id])
    while WL:
        lid = WL.popleft()
        L = net.by_id[lid]

        # DAG merge: join bounds from all predecessors using box_join
        # For layers with multiple predecessors (e.g., ADD in residual blocks),
        # we compute the interval hull of all predecessor bounds.
        if net.preds.get(lid):
            preds_list = net.preds[lid]
            # Initialize from first predecessor
            first_bounds = after[preds_list[0]].bounds
            Bjoin = Bounds(lb=first_bounds.lb.clone(), ub=first_bounds.ub.clone())
            Cjoin = ConSet()
            for con in after[preds_list[0]].cons:
                Cjoin.replace(con)
            # Join with remaining predecessors (for DAG merge points)
            for pid in preds_list[1:]:
                Bjoin = box_join(Bjoin, after[pid].bounds)
                for con in after[pid].cons:
                    Cjoin.replace(con)
            before[lid] = Fact(Bjoin, Cjoin)

        # Apply transfer function to compute output bounds
        out_fact = dispatch_tf(L, before, after, net)

        # Check if bounds changed significantly (handles nan from inf arithmetic)
        if changed_or_maskdiff(L, out_fact.bounds, None, eps):
            after[lid] = out_fact
            update_cache(L, out_fact.bounds, None)
            # Collect constraints globally
            for con in out_fact.cons:
                globalC.replace(con)
            # Add all successors to worklist for re-processing
            for sid in net.succs.get(lid, []):
                WL.append(sid)

    return before, after, globalC
