#===- act/back_end/utils.py - Backend Utility Functions ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Backend utility functions for ACT verification framework.
#   Provides common utilities for bounds manipulation and tensor operations.
#
#===---------------------------------------------------------------------===#

import torch
from typing import Dict, Any, Tuple, Optional
from act.back_end.core import Bounds, ConSet
from act.util.options import PerformanceOptions

EPS = 1e-12

def box_join(a: Bounds, b: Bounds) -> Bounds:
    return Bounds(lb=torch.minimum(a.lb, b.lb), ub=torch.maximum(a.ub, b.ub))

def changed_or_maskdiff(L, B: Bounds, masks: Optional[Dict[str, torch.Tensor]], eps=1e-9) -> bool:
    plb = L.cache.get("prev_lb"); pub = L.cache.get("prev_ub")
    if plb is None or pub is None: return True
    if torch.any(torch.abs(plb - B.lb) > eps) or torch.any(torch.abs(pub - B.ub) > eps): return True
    prev = L.cache.get("masks")
    if (masks is None) ^ (prev is None): return True
    if masks is None: return False
    for k in masks.keys():
        if (k not in prev) or (masks[k].shape != prev[k].shape) or torch.any(masks[k] != prev[k]):
            return True
    return False

def update_cache(L, B: Bounds, masks: Optional[Dict[str, torch.Tensor]]):
    L.cache["prev_lb"] = B.lb.clone(); L.cache["prev_ub"] = B.ub.clone()
    L.cache["masks"] = None if masks is None else {k: v.clone() for k,v in masks.items()}

def affine_bounds(W_pos, W_neg, b, Bin: Bounds) -> Bounds:
    """Compute bounds for affine transformation using interval arithmetic.
    
    Uses pre-computed W_pos/W_neg split for efficiency:
    - W_pos = clamp(W, min=0): positive weights
    - W_neg = clamp(W, max=0): negative weights
    
    Interval arithmetic:
    - Lower bound: W_pos @ lb + W_neg @ ub + b
    - Upper bound: W_pos @ ub + W_neg @ lb + b
    
    Args:
        W_pos: Positive part of weight matrix [out_features, in_features]
        W_neg: Negative part of weight matrix [out_features, in_features]
        b: Bias vector [out_features]
        Bin: Input bounds with shape [batch, in_features] or [in_features]
    
    Returns:
        Output bounds with shape [out_features]
    """
    # Validate and squeeze batch dimension if present
    if Bin.lb.ndim == 2:
        batch_size = Bin.lb.shape[0]
        assert batch_size == 1, (
            f"affine_bounds expects batch_size=1 for symbolic verification, "
            f"got batch_size={batch_size}. Bounds shape: {Bin.lb.shape}"
        )
        lb_vec = Bin.lb.squeeze(0)  # [1, in_features] → [in_features]
        ub_vec = Bin.ub.squeeze(0)
    elif Bin.lb.ndim == 1:
        lb_vec = Bin.lb
        ub_vec = Bin.ub
    else:
        raise ValueError(
            f"affine_bounds expects 1D or 2D bounds, got {Bin.lb.ndim}D with shape {Bin.lb.shape}"
        )
    
    # Compute bounds using interval arithmetic
    lb = W_pos @ lb_vec + W_neg @ ub_vec + b
    ub = W_pos @ ub_vec + W_neg @ lb_vec + b
    return Bounds(lb, ub)

def pwl_meta(l: torch.Tensor, u: torch.Tensor, K: int) -> Dict[str, Any]:
    return {"K": int(K), "mid": 0.5*(l+u)}

def bound_var_interval(l: torch.Tensor, u: torch.Tensor) -> Tuple[float, float]:
    r = 0.5*(u-l); v_hi = float(torch.mean((2*r)**2))
    return (0.0, v_hi)

def scale_interval(cx_lo, cx_hi, inv_lo, inv_hi):
    cand = torch.stack([cx_lo*inv_lo, cx_lo*inv_hi, cx_hi*inv_lo, cx_hi*inv_hi], dim=0)
    return torch.min(cand, dim=0).values, torch.max(cand, dim=0).values


def validate_constraints(globalC, after: Dict, net) -> bool:
    """Validate constraint set for common errors (targeted validation).
    
    This function performs targeted validation by:
    1. Collecting only the variable IDs referenced by constraints in globalC
    2. Building var_bounds dict for only those variables from the 'after' facts
    3. Validating constraint dimensions and bounds existence
    
    Checks (when enabled):
    - All variable IDs referenced in constraints have bounds in 'after' facts
    - LIN_POLY dimensions match variable count
    - No NaN/Inf in constraint parameters
    
    Args:
        globalC: Constraint set to validate (ConSet)
        after: Dictionary mapping layer_id -> Fact (from analyze())
        net: ACT network with layer definitions
    
    Returns:
        True if all checks pass, False otherwise
    """
    from act.back_end.core import Bounds
    from act.util.options import PerformanceOptions
    
    if not PerformanceOptions.validate_constraints:
        return True  # Skip validation when disabled
    
    # Step 1: Collect all variable IDs referenced by constraints
    var_ids_used = set()
    for con in globalC:
        var_ids_used.update(con.var_ids)
    
    # Step 2: Build var_bounds dict for only the variables referenced by constraints
    var_bounds = {}
    for layer_id, fact in after.items():
        layer = net.by_id[layer_id]
        for i, var_id in enumerate(layer.out_vars):
            if var_id in var_ids_used:
                # Extract individual bounds for this variable
                var_bounds[var_id] = Bounds(
                    lb=fact.bounds.lb[i:i+1],  # Keep as 1D tensor with single element
                    ub=fact.bounds.ub[i:i+1]
                )
    
    # Step 3: Validate constraints
    all_valid = True
    issues = []
    
    for i, con in enumerate(globalC):
        # Check variable IDs exist
        for var_id in con.var_ids:
            if var_id not in var_bounds:
                issues.append(f"Constraint {i}: Variable ID {var_id} not in var_bounds")
                all_valid = False
        
        # Check LIN_POLY dimensions
        if con.kind == 'LIN_POLY':
            expected_vars = con.A.shape[1]
            actual_vars = len(con.var_ids)
            if expected_vars != actual_vars:
                issues.append(
                    f"Constraint {i}: A.shape[1]={expected_vars} != len(var_ids)={actual_vars}"
                )
                all_valid = False
            
            # Check for NaN/Inf
            if torch.isnan(con.A).any() or torch.isinf(con.A).any():
                issues.append(f"Constraint {i}: A matrix contains NaN/Inf")
                all_valid = False
            if torch.isnan(con.b).any() or torch.isinf(con.b).any():
                issues.append(f"Constraint {i}: b vector contains NaN/Inf")
                all_valid = False
    
    # Write to debug file (GUARDED - only if debug_tf is also enabled)
    if PerformanceOptions.debug_tf:
        with open(PerformanceOptions.debug_output_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"CONSTRAINT VALIDATION (Targeted)\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total constraints: {len(globalC)}\n")
            f.write(f"Unique variables referenced: {len(var_ids_used)}\n")
            f.write(f"Variables with bounds found: {len(var_bounds)}\n")
            f.write(f"Status: {'✅ VALID' if all_valid else '❌ INVALID'}\n")
            if issues:
                f.write(f"\nIssues found:\n")
                for issue in issues:
                    f.write(f"  - {issue}\n")
            f.write("\n")
    
    return all_valid
