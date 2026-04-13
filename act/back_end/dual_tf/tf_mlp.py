# ===- act/back_end/dual_tf/tf_mlp.py - MLP Dual Transfer Functions ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   MLP dual transfer functions for Lagrangian dual bound computation.
#   ReLU (Linear-adaptive): slope depends on sign of dual variable
#     - v >= 0: use lower bound (slope=0, contrib=0)
#     - v < 0: use upper bound (slope=u/(u-l), contrib=-v*d*l)
#   Dense: v_{i-1}=W^T@v, contrib=-b^T@v
#
# ===---------------------------------------------------------------------===#

import torch
from typing import Tuple, Optional
from act.back_end.core import Bounds


# -------- Helpers --------
def _align(a: torch.Tensor, n: int) -> torch.Tensor:
    """Align tensor to size n by truncating or tiling."""
    if a.numel() == n:
        return a.flatten()
    elif a.numel() > n:
        return a.flatten()[:n]
    else:
        return a.flatten().repeat((n + a.numel() - 1) // a.numel())[:n]


# -------- ReLU --------
@torch.no_grad()
def get_relu_masks(
    l: torch.Tensor, u: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get boolean masks: (on, off, amb) for ReLU neurons."""
    on, off = l >= 0, u <= 0
    return on, off, ~(on | off)


@torch.no_grad()
def dual_relu_backward(
    nu: torch.Tensor, bounds: Bounds
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ReLU backward pass for dual bound computation (Wong-Kolter style).

    Uses FIXED upper-bound slope for all crossing neurons (not adaptive).
    This matches the formulation in provable.py which achieves 91% certification.

    For crossing neurons (l < 0 < u):
    - Slope: d = u / (u - l) (upper bound relaxation)
    - Contribution: [nu]_+ * l (computed AFTER applying slope)

    Returns: (v_out, contribution)
    """
    l, u = bounds.lb, bounds.ub
    v = nu

    n = min(v.shape[-1], l.shape[-1])
    if v.shape[-1] != l.shape[-1]:
        l, u, v = l[..., :n], u[..., :n], v[..., :n]

    on, off, amb = get_relu_masks(l, u)

    d = torch.zeros_like(l)
    d = torch.where(on, torch.ones_like(d), d)
    if amb.any():
        denom = (u - l).clamp(min=1e-12)
        d = torch.where(amb, u / denom, d)

    v_out = d * v

    contrib = torch.zeros(v.shape[0], dtype=v.dtype, device=v.device)
    if amb.any():
        crossing_contrib = torch.where(amb, v_out.clamp(min=0) * l, torch.zeros_like(l))
        contrib = crossing_contrib.sum(dim=-1)

    return v_out, contrib


# -------- Dense --------
@torch.no_grad()
def dual_dense_backward(
    nu: torch.Tensor, W: torch.Tensor, b: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dense backward: v_out = W^T @ v, contrib = -b^T @ v."""
    assert W.dim() == 2
    assert nu.shape[-1] == W.shape[0]

    v_out = nu @ W
    contrib = (
        -(nu * b).sum(dim=-1)
        if b is not None
        else torch.zeros(nu.shape[0], dtype=nu.dtype, device=nu.device)
    )
    return v_out, contrib


# -------- Bias / Scale / BatchNorm --------
@torch.no_grad()
def dual_bias_backward(
    nu: torch.Tensor, c: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bias backward (y=x+c): v_out=v, contrib=-c^T@v."""
    c_flat = _align(c, nu.shape[-1])
    return nu, -(nu * c_flat).sum(dim=-1)


@torch.no_grad()
def dual_scale_backward(
    nu: torch.Tensor, a: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale backward (y=a*x): v_out=a*v, contrib=0."""
    a_aligned = _align(a, nu.shape[-1])
    return a_aligned * nu, torch.zeros(nu.shape[0], dtype=nu.dtype, device=nu.device)


@torch.no_grad()
def dual_bn_backward(
    nu: torch.Tensor, A: torch.Tensor, c: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BatchNorm backward (y=A*x+c): v_out=A*v, contrib=-c^T@v."""
    A_aligned = _align(A, nu.shape[-1])
    c_aligned = _align(c, nu.shape[-1])
    return A_aligned * nu, -(nu * c_aligned).sum(dim=-1)


# -------- Identity-like --------
@torch.no_grad()
def dual_identity_backward(nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Identity backward (Flatten, Reshape, etc.): v_out=v, contrib=0."""
    return nu, torch.zeros(nu.shape[0], dtype=nu.dtype, device=nu.device)
