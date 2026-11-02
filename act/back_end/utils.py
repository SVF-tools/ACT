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
from act.back_end.core import Bounds
from act.util.options import PerformanceOptions
from act.util.format_utils import rule

EPS = 1e-12


def pair_2d(value) -> Tuple[int, int]:
    if isinstance(value, int):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError(f"expected a 2-D parameter, got {value}")
    return int(value[0]), int(value[1])


def avgpool2d_output_hw(
    input_hw,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode: bool = False,
) -> Tuple[int, int]:
    h, w = pair_2d(input_hw)
    kh, kw = pair_2d(kernel_size)
    sh, sw = pair_2d(kernel_size if stride is None else stride)
    ph, pw = pair_2d(padding)

    def _out(n: int, k: int, s: int, p: int) -> int:
        if ceil_mode:
            out = (n + 2 * p - k + s - 1) // s + 1
            if (out - 1) * s >= n + p:
                out -= 1
            return out
        return (n + 2 * p - k) // s + 1

    return _out(h, kh, sh, ph), _out(w, kw, sw, pw)


def avgpool2d_denominators(
    input_hw,
    output_hw,
    kernel_size,
    stride=None,
    padding=0,
    *,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
    device=None,
    dtype=None,
) -> torch.Tensor:
    h, w = pair_2d(input_hw)
    oh, ow = pair_2d(output_hw)
    kh, kw = pair_2d(kernel_size)
    sh, sw = pair_2d(kernel_size if stride is None else stride)
    ph, pw = pair_2d(padding)
    expected = avgpool2d_output_hw(
        (h, w), (kh, kw), (sh, sw), (ph, pw), ceil_mode
    )
    if (oh, ow) != expected:
        raise ValueError(
            f"avgpool2d output shape {(oh, ow)} does not match expected {expected}"
        )

    if divisor_override is not None:
        divisor = int(divisor_override)
        if divisor <= 0:
            raise ValueError("avgpool2d divisor_override must be positive")
        return torch.full(
            (oh, ow), float(divisor), device=device, dtype=dtype or torch.float32
        )

    out_y = torch.arange(oh, device=device, dtype=torch.long) * sh - ph
    out_x = torch.arange(ow, device=device, dtype=torch.long) * sw - pw

    def _count(starts, kernel: int, lower: int, upper: int):
        lo = torch.maximum(starts, torch.full_like(starts, lower))
        hi = torch.minimum(starts + kernel, torch.full_like(starts, upper))
        return (hi - lo).clamp_min(0)

    if count_include_pad:
        count_y = _count(out_y, kh, -ph, h + ph)
        count_x = _count(out_x, kw, -pw, w + pw)
    else:
        count_y = _count(out_y, kh, 0, h)
        count_x = _count(out_x, kw, 0, w)
    denom = count_y[:, None] * count_x[None, :]
    if torch.any(denom <= 0):
        raise ValueError("avgpool2d produced an empty pooling window")
    return denom.to(dtype=dtype or torch.float32)

def split_weight(W):
    return W.clamp(min=0), W.clamp(max=0)

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
    """Batched affine-transform bounds via interval arithmetic.

    Maps [B, n_in] input bounds to [B, n_out] output bounds for any B>=1.

    Args:
        W_pos: clamp(W, min=0), shape [out_features, in_features]
        W_neg: clamp(W, max=0), shape [out_features, in_features]
        b: bias vector, shape [out_features]
        Bin: input bounds, lb/ub shape [B, in_features]

    Returns:
        Bounds with lb/ub shape [B, out_features].
    """
    assert Bin.lb.dim() == 2, f"affine_bounds expects 2D [B, n_in] bounds, got shape {tuple(Bin.lb.shape)}"
    lo, hi = Bin.lb, Bin.ub
    if not (torch.isinf(lo).any() or torch.isinf(hi).any()):
        lb = lo @ W_pos.T + hi @ W_neg.T + b
        ub = hi @ W_pos.T + lo @ W_neg.T + b
        return Bounds(lb, ub)
    return _affine_bounds_inf_safe(W_pos, W_neg, b, lo, hi)


def _affine_bounds_inf_safe(W_pos, W_neg, b, lo, hi) -> Bounds:
    # Interval-arithmetic convention 0 * (±inf) = 0: a zero weight drops its
    # input coordinate from the affine map, so an unbounded input range times a
    # zero weight must contribute 0. IEEE evaluates 0*inf as NaN, which the fused
    # matmul then spreads across the whole output row (an unsound NaN "bound").
    # Compute the finite contribution with infinities zeroed, then re-inject the
    # true infinities from the sign pattern of the already-split nonnegative
    # (W_pos) / nonpositive (W_neg) weights, which never multiplies a zero weight
    # by an infinity. NaN inputs are left intact so genuine upstream NaN surfaces.
    dt = lo.dtype
    lo0 = torch.where(torch.isinf(lo), torch.zeros_like(lo), lo)
    hi0 = torch.where(torch.isinf(hi), torch.zeros_like(hi), hi)
    lb = lo0 @ W_pos.T + hi0 @ W_neg.T + b
    ub = hi0 @ W_pos.T + lo0 @ W_neg.T + b

    wp = (W_pos > 0).to(dt)   # strictly positive weights (W > 0)
    wn = (W_neg < 0).to(dt)   # strictly negative weights (W < 0)
    lo_neg = (lo == float("-inf")).to(dt)
    hi_pos = (hi == float("inf")).to(dt)
    # min_x (Wx+b) is -inf iff some coord has (lo=-inf & W>0) or (hi=+inf & W<0)
    lb_neg = (lo_neg @ wp.T + hi_pos @ wn.T) > 0
    # max_x (Wx+b) is +inf iff some coord has (hi=+inf & W>0) or (lo=-inf & W<0)
    ub_pos = (hi_pos @ wp.T + lo_neg @ wn.T) > 0
    lb = torch.where(lb_neg, torch.full_like(lb, float("-inf")), lb)
    ub = torch.where(ub_pos, torch.full_like(ub, float("inf")), ub)
    return Bounds(lb, ub)

def pwl_meta(l: torch.Tensor, u: torch.Tensor, K: int) -> Dict[str, Any]:
    return {"K": int(K), "mid": 0.5*(l+u)}

def four_corner_envelope(xl, xu, yl, yu, sum_axis=None, keepdim=False, scale=1.0):
    """Four-corner McCormick envelope (lb, ub) of x*y over [xl,xu]x[yl,yu].

    ``sum_axis`` contracts the envelope along a matmul/attention axis; ``scale``
    is a post-sum factor (e.g. 1/dk). ``amin``/``amax`` over the stacked corners
    are bit-identical to the nested-``minimum`` / ``min(stack,0).values`` idioms.
    """
    p = torch.stack([xl*yl, xl*yu, xu*yl, xu*yu], dim=0)
    lb = p.amin(dim=0); ub = p.amax(dim=0)
    if sum_axis is not None:
        lb = lb.sum(dim=sum_axis, keepdim=keepdim); ub = ub.sum(dim=sum_axis, keepdim=keepdim)
    if scale != 1.0:
        lb = lb * scale; ub = ub * scale
    return lb, ub


def scale_interval(cx_lo, cx_hi, inv_lo, inv_hi):
    return four_corner_envelope(cx_lo, cx_hi, inv_lo, inv_hi)


def validate_constraints(globalC, after: Dict[int, Any], net) -> bool:
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
            f.write(f"\n{rule()}\n")
            f.write(f"CONSTRAINT VALIDATION (Targeted)\n")
            f.write(f"{rule()}\n")
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
