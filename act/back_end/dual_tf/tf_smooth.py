#===- act/back_end/dual_tf/tf_smooth.py - Smooth Activation Dual TF -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Smooth activation dual backward (Sigmoid, Tanh). Tangent-line relaxation.
#   Unified dual_smooth_backward() works for any S-shaped function given f, f'.
#     - v >= 0: use lower bound (tangent/chord), contrib = v * b_lower
#     - v < 0: use upper bound (chord/tangent), contrib = v * b_upper
#
#===---------------------------------------------------------------------===#

import torch
from typing import Tuple, Callable
from act.back_end.core import Bounds

# -------- Activation Functions --------
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid: f(x) = 1 / (1 + exp(-x))"""
    return torch.sigmoid(x)

def dsigmoid(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid derivative: f'(x) = f(x) * (1 - f(x))"""
    s = torch.sigmoid(x); return s * (1 - s)

def tanh(x: torch.Tensor) -> torch.Tensor:
    """Tanh: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    return torch.tanh(x)

def dtanh(x: torch.Tensor) -> torch.Tensor:
    """Tanh derivative: f'(x) = 1 - tanh(x)^2"""
    return 1 - torch.tanh(x) ** 2

# -------- Linear Relaxation --------
@torch.no_grad()
def compute_smooth_relaxation(
    l: torch.Tensor, u: torch.Tensor,
    func: Callable[[torch.Tensor], torch.Tensor],
    dfunc: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute linear relaxation for S-shaped f(x) on [l, u].
    Returns (k_lower, b_lower, k_upper, b_upper) for y = k*x + b bounds.
    
    S-shaped (convex x<0, concave x>0): l>=0 lower=tangent@l, u<=0 upper=tangent@u.
    """
    assert (l <= u).all(), f"Invalid bounds: l > u at {(l > u).nonzero().flatten().tolist()[:5]}"
    
    f_l, f_u = func(l), func(u)
    
    # Chord slope
    denom = (u - l).clamp(min=1e-12)
    k_chord = (f_u - f_l) / denom
    
    # Initialize with chord (always sound)
    k_lower, k_upper = k_chord.clone(), k_chord.clone()
    b_lower = f_l - k_lower * l
    b_upper = f_l - k_upper * l
    
    # l >= 0 (concave): tangent at l for lower bound
    mask_pos = l >= 0
    if mask_pos.any():
        k_tan = dfunc(l)
        k_lower[mask_pos] = k_tan[mask_pos]
        b_lower[mask_pos] = f_l[mask_pos] - k_tan[mask_pos] * l[mask_pos]
    
    # u <= 0 (convex): tangent at u for upper bound
    mask_neg = u <= 0
    if mask_neg.any():
        k_tan = dfunc(u)
        k_upper[mask_neg] = k_tan[mask_neg]
        b_upper[mask_neg] = f_u[mask_neg] - k_tan[mask_neg] * u[mask_neg]
    
    return k_lower, b_lower, k_upper, b_upper

# -------- Smooth Backward --------
@torch.no_grad()
def dual_smooth_backward(
    nu: torch.Tensor, bounds: Bounds,
    func: Callable[[torch.Tensor], torch.Tensor],
    dfunc: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Smooth activation backward: v_out = v*k, contrib = sum(v*b). Adaptive bound selection."""
    l, u = bounds.lb.flatten(), bounds.ub.flatten()
    v = nu.flatten()
    
    # Align sizes if needed
    n = min(v.numel(), l.numel())
    if v.numel() != l.numel(): l, u, v = l[:n], u[:n], v[:n]
    
    # Compute relaxation and select bound based on sign of v
    k_lower, b_lower, k_upper, b_upper = compute_smooth_relaxation(l, u, func, dfunc)
    v_pos = v >= 0
    k = torch.where(v_pos, k_lower, k_upper)
    b = torch.where(v_pos, b_lower, b_upper)
    
    v_out = v * k
    contrib = (v * b).sum()
    
    return v_out, contrib

# -------- Sigmoid / Tanh --------
@torch.no_grad()
def dual_sigmoid_backward(nu: torch.Tensor, bounds: Bounds) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid backward: tangent-line relaxation for f(x) = 1/(1+exp(-x))."""
    return dual_smooth_backward(nu, bounds, sigmoid, dsigmoid)

@torch.no_grad()
def dual_tanh_backward(nu: torch.Tensor, bounds: Bounds) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tanh backward: tangent-line relaxation for f(x) = tanh(x)."""
    return dual_smooth_backward(nu, bounds, tanh, dtanh)
