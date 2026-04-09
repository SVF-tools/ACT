# ===- act/back_end/solver/solver_hz.py - Hybrid Zonotope Solver ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Hybrid Zonotope (HZ) solver for computing tight bounds during verification.
#
#   The HZ domain represents reachable sets as:
#     Z = {c + Gc*xi_c + Gb*xi_b | Ac*xi_c + Ab*xi_b = b,
#          xi_c in [-1,1]^ng, xi_b in {-1,1}^nb}
#   This captures continuous uncertainty (Gc) and discrete switching (Gb),
#   enabling exact ReLU/LeakyReLU encoding and piecewise tanh/sigmoid.
#
#   Solver hierarchy:
#     GurobiSolver (MILP, exact) > HZSolver (HZ, tight) > interval (box, fast)
#
#   Provides:
#     1. HZono dataclass — the HZ data container
#     2. Algebraic operations — linear maps, translations, Minkowski sums
#     3. Activation encodings — exact (ReLU, LeakyReLU) and piecewise (tanh, sigmoid)
#     4. Conv2d — HZ propagation through convolution layers
#     5. Order reduction — Girard's method for complexity control
#     6. Bounds computation — dispatch to Gurobi MILP / SciPy LP / fast path
#     7. HZSolver(Solver) — subclass for the solver hierarchy
#
#   This module is used by HybridzTF to compute tighter bounds than interval
#   arithmetic alone.  Transfer functions in hybridz_tf/ do NOT import this
#   module — only the HybridzTF orchestrator does.
#
# ===---------------------------------------------------------------------===#

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from act.back_end.core import Bounds
from act.back_end.utils import parse_input_shape

# Gurobi availability — delegated to solver_gurobi (single source of truth)
try:
    from act.back_end.solver.solver_gurobi import GurobiSolver, is_gurobi_available

    _HAS_GUROBI = is_gurobi_available()
except ImportError:
    _HAS_GUROBI = False

try:
    import numpy as np
    from scipy.optimize import linprog

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ============================================================================
# 1. HZono dataclass
# ============================================================================


@dataclass
class HZono:
    """Hybrid Zonotope data container.

    Z = {c + Gc @ xi_c + Gb @ xi_b | Ac @ xi_c + Ab @ xi_b = b,
         xi_c in [-1,1]^ng, xi_b in {-1,1}^nb}
    """

    c: torch.Tensor  # (n, 1)   center vector
    Gc: torch.Tensor  # (n, ng)  continuous generator matrix
    Gb: torch.Tensor  # (n, nb)  binary generator matrix
    Ac: torch.Tensor  # (nc, ng) continuous constraint matrix
    Ab: torch.Tensor  # (nc, nb) binary constraint matrix
    b: torch.Tensor  # (nc, 1)  constraint RHS vector


# ============================================================================
# 2. Algebraic operations
# ============================================================================


def hz_multiply(hz: HZono, R: torch.Tensor) -> HZono:
    """Linear map: c'=R@c, Gc'=R@Gc, Gb'=R@Gb, constraints unchanged."""
    R = R.to(dtype=hz.c.dtype, device=hz.c.device)
    return HZono(
        c=R @ hz.c,
        Gc=R @ hz.Gc,
        Gb=R @ hz.Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
    )


def hz_add_const(hz: HZono, v: torch.Tensor) -> HZono:
    """Translate center: c'=c+v, generators and constraints unchanged."""
    v = v.to(dtype=hz.c.dtype, device=hz.c.device)
    if v.ndim == 1:
        v = v.view(-1, 1)
    return HZono(
        c=hz.c + v,
        Gc=hz.Gc.clone(),
        Gb=hz.Gb.clone(),
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
    )


def hz_minkowski_sum(hz1: HZono, hz2: HZono) -> HZono:
    """Minkowski sum: c1+c2, block-diag generators, block constraints."""
    dtype, device = hz1.c.dtype, hz1.c.device

    new_c = hz1.c + hz2.c.to(dtype=dtype, device=device)
    new_Gc = torch.cat([hz1.Gc, hz2.Gc.to(dtype=dtype, device=device)], dim=1)
    new_Gb = torch.cat([hz1.Gb, hz2.Gb.to(dtype=dtype, device=device)], dim=1)

    nc1, nc2 = hz1.Ac.shape[0], hz2.Ac.shape[0]
    ng1, ng2 = hz1.Gc.shape[1], hz2.Gc.shape[1]
    nb1, nb2 = hz1.Gb.shape[1], hz2.Gb.shape[1]

    Ac_top = torch.cat(
        [hz1.Ac, torch.zeros((nc1, ng2), dtype=dtype, device=device)], dim=1
    )
    Ac_bot = torch.cat(
        [
            torch.zeros((nc2, ng1), dtype=dtype, device=device),
            hz2.Ac.to(dtype=dtype, device=device),
        ],
        dim=1,
    )
    new_Ac = torch.cat([Ac_top, Ac_bot], dim=0)

    Ab_top = torch.cat(
        [hz1.Ab, torch.zeros((nc1, nb2), dtype=dtype, device=device)], dim=1
    )
    Ab_bot = torch.cat(
        [
            torch.zeros((nc2, nb1), dtype=dtype, device=device),
            hz2.Ab.to(dtype=dtype, device=device),
        ],
        dim=1,
    )
    new_Ab = torch.cat([Ab_top, Ab_bot], dim=0)

    new_b = torch.cat([hz1.b, hz2.b.to(dtype=dtype, device=device)], dim=0)

    return HZono(c=new_c, Gc=new_Gc, Gb=new_Gb, Ac=new_Ac, Ab=new_Ab, b=new_b)


def hz_from_bounds(bounds: Bounds, dtype, device) -> HZono:
    """Create fresh, unconstrained HZ from interval Bounds (axis-aligned box)."""
    lb = bounds.lb.flatten().to(dtype=dtype, device=device)
    ub = bounds.ub.flatten().to(dtype=dtype, device=device)
    n = lb.shape[0]
    c = ((lb + ub) / 2.0).view(-1, 1)
    rad = (ub - lb) / 2.0
    return HZono(
        c=c,
        Gc=torch.diag(rad),
        Gb=torch.zeros((n, 0), dtype=dtype, device=device),
        Ac=torch.zeros((0, n), dtype=dtype, device=device),
        Ab=torch.zeros((0, 0), dtype=dtype, device=device),
        b=torch.zeros((0, 1), dtype=dtype, device=device),
    )


# ============================================================================
# 3. Activation encodings
# ============================================================================


def hz_apply_relu(hz: HZono) -> HZono:
    """Exact ReLU via equality constraints + linking equality.

    Per unstable neuron i with bounds [alpha, beta] (alpha < 0 < beta):
      ng += 4 (xi1, xi2, xi3, xi4)
      nb += 1 (z)
      nc += 3 equalities
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    bounds = hz_compute_bounds(hz)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    active = lb >= 0
    inactive = ub <= 0
    unstable = ~active & ~inactive
    k = int(unstable.sum().item())

    out_Gc = torch.zeros((n, ng + 4 * k), dtype=dtype, device=device)
    out_Gb = torch.zeros((n, nb + k), dtype=dtype, device=device)
    out_c = torch.zeros((n, 1), dtype=dtype, device=device)

    if active.any():
        out_c[active] = hz.c[active]
        out_Gc[active, :ng] = hz.Gc[active]
        out_Gb[active, :nb] = hz.Gb[active]

    if k == 0:
        return HZono(
            c=out_c,
            Gc=out_Gc[:, :ng],
            Gb=out_Gb[:, :nb],
            Ac=hz.Ac.clone(),
            Ab=hz.Ab.clone(),
            b=hz.b.clone(),
        )

    unstable_idx = torch.where(unstable)[0]
    alpha = lb[unstable_idx]
    beta = ub[unstable_idx]
    t = torch.arange(k, device=device)

    col_xi1 = ng + t
    col_xi2 = ng + k + t
    col_xi3 = ng + 2 * k + t
    col_xi4 = ng + 3 * k + t
    col_z = nb + t

    out_c[unstable_idx, 0] = beta / 2.0
    out_Gc[unstable_idx, col_xi2] = -beta / 2.0

    ng_new = ng + 4 * k
    nb_new = nb + k

    eq_Ac = torch.zeros((3 * k, ng_new), dtype=dtype, device=device)
    eq_Ab = torch.zeros((3 * k, nb_new), dtype=dtype, device=device)
    eq_b = torch.zeros((3 * k, 1), dtype=dtype, device=device)

    r1 = 3 * t
    r2 = 3 * t + 1

    eq_Ac[r1, col_xi1] = 1.0
    eq_Ac[r1, col_xi3] = 1.0
    eq_Ab[r1, col_z] = 1.0
    eq_b[r1, 0] = 1.0

    eq_Ac[r2, col_xi2] = 1.0
    eq_Ac[r2, col_xi4] = 1.0
    eq_Ab[r2, col_z] = -1.0
    eq_b[r2, 0] = 1.0

    for j in range(k):
        idx_i = int(unstable_idx[j].item())
        eq_Ac[3 * j + 2, col_xi1[j]] = alpha[j] / 2.0
        eq_Ac[3 * j + 2, col_xi2[j]] = -beta[j] / 2.0
        eq_Ac[3 * j + 2, :ng] -= hz.Gc[idx_i]
        eq_Ab[3 * j + 2, :nb] -= hz.Gb[idx_i]
        eq_Ab[3 * j + 2, col_z[j]] = alpha[j] / 2.0
        eq_b[3 * j + 2, 0] = hz.c[idx_i, 0] - beta[j] / 2.0

    old_Ac_ext = torch.cat(
        [hz.Ac, torch.zeros((nc, 4 * k), dtype=dtype, device=device)], dim=1
    )
    old_Ab_ext = torch.cat(
        [hz.Ab, torch.zeros((nc, k), dtype=dtype, device=device)], dim=1
    )

    return HZono(
        c=out_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=torch.cat([old_Ac_ext, eq_Ac], dim=0),
        Ab=torch.cat([old_Ab_ext, eq_Ab], dim=0),
        b=torch.cat([hz.b, eq_b], dim=0),
    )


def hz_apply_leaky_relu(hz: HZono, alpha_arg: float) -> HZono:
    """Exact LeakyReLU via equality constraints + box equalities with slack.

    Per unstable neuron: ng += 6, nb += 1, nc += 5.
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]
    a = alpha_arg

    bounds = hz_compute_bounds(hz)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    active = lb >= 0
    inactive = ub <= 0
    unstable = ~active & ~inactive
    k = int(unstable.sum().item())

    out_Gc = torch.zeros((n, ng + 6 * k), dtype=dtype, device=device)
    out_Gb = torch.zeros((n, nb + k), dtype=dtype, device=device)
    out_c = torch.zeros((n, 1), dtype=dtype, device=device)

    if active.any():
        out_c[active] = hz.c[active]
        out_Gc[active, :ng] = hz.Gc[active]
        out_Gb[active, :nb] = hz.Gb[active]

    if inactive.any():
        out_c[inactive] = a * hz.c[inactive]
        out_Gc[inactive, :ng] = a * hz.Gc[inactive]
        out_Gb[inactive, :nb] = a * hz.Gb[inactive]

    if k == 0:
        return HZono(
            c=out_c,
            Gc=out_Gc[:, :ng],
            Gb=out_Gb[:, :nb],
            Ac=hz.Ac.clone(),
            Ab=hz.Ab.clone(),
            b=hz.b.clone(),
        )

    unstable_idx = torch.where(unstable)[0]
    l = lb[unstable_idx]
    u = ub[unstable_idx]
    t = torch.arange(k, device=device)

    col_g1 = ng + t
    col_g2 = ng + k + t
    col_s1p = ng + 2 * k + t
    col_s1m = ng + 3 * k + t
    col_s2p = ng + 4 * k + t
    col_s2m = ng + 5 * k + t
    col_z = nb + t

    out_c[unstable_idx, 0] = (u + a * l) / 4.0
    out_Gc[unstable_idx, col_g1] = a * l / 2.0
    out_Gc[unstable_idx, col_g2] = -u / 2.0
    out_Gb[unstable_idx, col_z] = (a * l - u) / 4.0

    ng_total = ng + 6 * k
    nb_total = nb + k

    eq_Ac = torch.zeros((5 * k, ng_total), dtype=dtype, device=device)
    eq_Ab = torch.zeros((5 * k, nb_total), dtype=dtype, device=device)
    eq_b = torch.zeros((5 * k, 1), dtype=dtype, device=device)

    r0 = 5 * t
    r1 = 5 * t + 1
    r2 = 5 * t + 2
    r3 = 5 * t + 3

    eq_Ac[r0, col_g1] = 1.0
    eq_Ac[r0, col_s1p] = 1.0
    eq_Ab[r0, col_z] = 0.5
    eq_b[r0, 0] = 0.5

    eq_Ac[r1, col_g1] = -1.0
    eq_Ac[r1, col_s1m] = 1.0
    eq_Ab[r1, col_z] = 0.5
    eq_b[r1, 0] = 0.5

    eq_Ac[r2, col_g2] = 1.0
    eq_Ac[r2, col_s2p] = 1.0
    eq_Ab[r2, col_z] = -0.5
    eq_b[r2, 0] = 0.5

    eq_Ac[r3, col_g2] = -1.0
    eq_Ac[r3, col_s2m] = 1.0
    eq_Ab[r3, col_z] = -0.5
    eq_b[r3, 0] = 0.5

    for j in range(k):
        idx_i = int(unstable_idx[j].item())
        eq_Ac[5 * j + 4, :ng] = hz.Gc[idx_i]
        eq_Ac[5 * j + 4, col_g1[j]] = -l[j] / 2.0
        eq_Ac[5 * j + 4, col_g2[j]] = u[j] / 2.0
        eq_Ab[5 * j + 4, :nb] = hz.Gb[idx_i]
        eq_Ab[5 * j + 4, col_z[j]] = -(l[j] - u[j]) / 4.0
        eq_b[5 * j + 4, 0] = (u[j] + l[j]) / 4.0 - hz.c[idx_i, 0]

    old_Ac_ext = torch.cat(
        [hz.Ac, torch.zeros((nc, 6 * k), dtype=dtype, device=device)], dim=1
    )
    old_Ab_ext = torch.cat(
        [hz.Ab, torch.zeros((nc, k), dtype=dtype, device=device)], dim=1
    )

    return HZono(
        c=out_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=torch.cat([old_Ac_ext, eq_Ac], dim=0),
        Ab=torch.cat([old_Ab_ext, eq_Ab], dim=0),
        b=torch.cat([hz.b, eq_b], dim=0),
    )


def hz_apply_piecewise(hz: HZono, func, dfunc, K: int = 2) -> HZono:
    """Piecewise linear approximation for monotone activations (tangent parallelogram)."""
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    bounds = hz_compute_bounds(hz)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    wide = (ub - lb) > 1e-12
    narrow = ~wide
    wide_idx = torch.where(wide)[0]
    m = int(wide_idx.sum() if wide_idx.ndim == 0 else wide_idx.shape[0])

    new_c = hz.c.clone()
    new_c[narrow] = func(hz.c[narrow])
    new_Gc_base = hz.Gc.clone()
    new_Gc_base[narrow] = 0.0
    new_Gb_base = hz.Gb.clone()
    new_Gb_base[narrow] = 0.0

    if m == 0:
        return HZono(
            c=new_c,
            Gc=new_Gc_base,
            Gb=new_Gb_base,
            Ac=hz.Ac.clone(),
            Ab=hz.Ab.clone(),
            b=hz.b.clone(),
        )

    lb_w, ub_w = lb[wide_idx], ub[wide_idx]
    centers_x_k, centers_y_k = [], []
    g1_x_k, g1_y_k, g2_x_k, g2_y_k = [], [], [], []

    for k_idx in range(K):
        a = lb_w + k_idx * (ub_w - lb_w) / K
        b = lb_w + (k_idx + 1) * (ub_w - lb_w) / K
        fa, fb = func(a), func(b)
        la, lb_slope = dfunc(a), dfunc(b)
        cx, cy = (a + b) / 2.0, (fa + fb) / 2.0
        nearly_linear = (la - lb_slope).abs() < 1e-10

        denom = lb_slope - la
        safe_denom = torch.where(nearly_linear, torch.ones_like(denom), denom)
        p1 = (fb - fa + lb_slope * a - la * b) / safe_denom
        p2 = a + b - p1
        g1x_tang = (p1 - a) / 2.0
        g1y_tang = lb_slope * (p1 - a) / 2.0
        g2x_tang = (p2 - a) / 2.0
        g2y_tang = la * (p2 - a) / 2.0

        hw = (b - a) / 2.0
        slope = (fb - fa) / (b - a + 1e-30)
        t_pts = torch.linspace(0.0, 1.0, 50, dtype=dtype, device=device).unsqueeze(1)
        pts = a.unsqueeze(0) + t_pts * (b - a).unsqueeze(0)
        f_pts = func(pts)
        resid = f_pts - (slope.unsqueeze(0) * pts + (fa - slope * a).unsqueeze(0))
        max_err = resid.abs().max(dim=0).values
        g1x_lin, g1y_lin = hw, slope * hw
        g2x_lin, g2y_lin = torch.zeros_like(hw), max_err

        g1x = torch.where(nearly_linear, g1x_lin, g1x_tang)
        g1y = torch.where(nearly_linear, g1y_lin, g1y_tang)
        g2x = torch.where(nearly_linear, g2x_lin, g2x_tang)
        g2y = torch.where(nearly_linear, g2y_lin, g2y_tang)

        # Soundness check
        dx = pts - cx.unsqueeze(0)
        dy = f_pts - cy.unsqueeze(0)
        det = g1y * g2x - g1x * g2y
        safe_det = torch.where(det.abs() < 1e-30, torch.ones_like(det), det)
        xi1 = (dy * g2x.unsqueeze(0) - dx * g2y.unsqueeze(0)) / safe_det.unsqueeze(0)
        xi2 = (dy * g1x.unsqueeze(0) - dx * g1y.unsqueeze(0)) / (-safe_det.unsqueeze(0))
        max_xi = torch.max(xi1.abs().max(dim=0).values, xi2.abs().max(dim=0).values)
        scale_factor = torch.where(max_xi > 1.0, max_xi * 1.01, torch.ones_like(max_xi))
        scale_factor = torch.where(
            det.abs() < 1e-30, torch.ones_like(scale_factor), scale_factor
        )
        g1x *= scale_factor
        g1y *= scale_factor
        g2x *= scale_factor
        g2y *= scale_factor

        centers_x_k.append(cx)
        centers_y_k.append(cy)
        g1_x_k.append(g1x)
        g1_y_k.append(g1y)
        g2_x_k.append(g2x)
        g2_y_k.append(g2y)

    cy_sum = torch.zeros(m, dtype=dtype, device=device)
    for k_idx in range(K):
        cy_sum = cy_sum + centers_y_k[k_idx]
    new_c[wide_idx] = (cy_sum / 2.0).unsqueeze(1)
    new_Gc_base[wide_idx] = 0.0
    new_Gb_base[wide_idx] = 0.0

    n_real = 2 * K * m
    n_slack = 4 * K * m
    Gc_new = torch.zeros((n, n_real + n_slack), dtype=dtype, device=device)
    for k_idx in range(K):
        g1_cols = torch.arange(k_idx * m, (k_idx + 1) * m, device=device)
        g2_cols = torch.arange(
            K * m + k_idx * m, K * m + (k_idx + 1) * m, device=device
        )
        for j in range(m):
            Gc_new[wide_idx[j], g1_cols[j]] = g1_y_k[k_idx][j]
            Gc_new[wide_idx[j], g2_cols[j]] = g2_y_k[k_idx][j]

    Gb_new = torch.zeros((n, K * m), dtype=dtype, device=device)
    for k_idx in range(K):
        z_cols = torch.arange(k_idx * m, (k_idx + 1) * m, device=device)
        for j in range(m):
            Gb_new[wide_idx[j], z_cols[j]] = -centers_y_k[k_idx][j] / 2.0

    out_Gc = torch.cat([new_Gc_base, Gc_new], dim=1)
    out_Gb = torch.cat([new_Gb_base, Gb_new], dim=1)
    ng_total = ng + n_real + n_slack
    nb_total = nb + K * m

    n_box = 4 * K * m
    n_eq_total = n_box + m + m
    eq_Ac = torch.zeros((n_eq_total, ng_total), dtype=dtype, device=device)
    eq_Ab = torch.zeros((n_eq_total, nb_total), dtype=dtype, device=device)
    eq_b = torch.zeros((n_eq_total, 1), dtype=dtype, device=device)

    for k_idx in range(K):
        for j in range(m):
            g1_col = ng + k_idx * m + j
            g2_col = ng + K * m + k_idx * m + j
            z_col = nb + k_idx * m + j
            s_base = ng + n_real + (k_idx * m + j) * 4
            r = 4 * (k_idx * m + j)
            eq_Ac[r, g1_col] = 1.0
            eq_Ac[r, s_base] = 1.0
            eq_Ab[r, z_col] = -0.5
            eq_b[r, 0] = 0.5
            eq_Ac[r + 1, g1_col] = -1.0
            eq_Ac[r + 1, s_base + 1] = 1.0
            eq_Ab[r + 1, z_col] = -0.5
            eq_b[r + 1, 0] = 0.5
            eq_Ac[r + 2, g2_col] = 1.0
            eq_Ac[r + 2, s_base + 2] = 1.0
            eq_Ab[r + 2, z_col] = -0.5
            eq_b[r + 2, 0] = 0.5
            eq_Ac[r + 3, g2_col] = -1.0
            eq_Ac[r + 3, s_base + 3] = 1.0
            eq_Ab[r + 3, z_col] = -0.5
            eq_b[r + 3, 0] = 0.5

    for j in range(m):
        idx_i = int(wide_idx[j].item())
        r = n_box + j
        rhs_val = 0.0
        for k_idx in range(K):
            g1_col = ng + k_idx * m + j
            g2_col = ng + K * m + k_idx * m + j
            z_col = nb + k_idx * m + j
            rhs_val += centers_x_k[k_idx][j].item() / 2.0
            eq_Ac[r, g1_col] = -g1_x_k[k_idx][j]
            eq_Ac[r, g2_col] = -g2_x_k[k_idx][j]
            eq_Ab[r, z_col] = centers_x_k[k_idx][j] / 2.0
        eq_Ac[r, :ng] = hz.Gc[idx_i]
        eq_Ab[r, :nb] = hz.Gb[idx_i]
        eq_b[r, 0] = rhs_val - hz.c[idx_i, 0].item()

    for j in range(m):
        r = n_box + m + j
        for k_idx in range(K):
            eq_Ab[r, nb + k_idx * m + j] = 1.0
        eq_b[r, 0] = float(K - 2)

    old_Ac_ext = torch.cat(
        [hz.Ac, torch.zeros((nc, n_real + n_slack), dtype=dtype, device=device)], dim=1
    )
    old_Ab_ext = torch.cat(
        [hz.Ab, torch.zeros((nc, K * m), dtype=dtype, device=device)], dim=1
    )

    return HZono(
        c=new_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=torch.cat([old_Ac_ext, eq_Ac], dim=0),
        Ab=torch.cat([old_Ab_ext, eq_Ab], dim=0),
        b=torch.cat([hz.b, eq_b], dim=0),
    )


def hz_apply_sigmoid(hz: HZono, K: int = 2) -> HZono:
    """Piecewise linear sigmoid via tangent parallelogram encoding."""
    return hz_apply_piecewise(
        hz, torch.sigmoid, lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)), K
    )


def hz_apply_tanh(hz: HZono, K: int = 2) -> HZono:
    """Piecewise linear tanh via tangent parallelogram encoding."""
    return hz_apply_piecewise(hz, torch.tanh, lambda x: 1 - torch.tanh(x) ** 2, K)


# ============================================================================
# 4. Conv2d
# ============================================================================


_parse_input_shape = parse_input_shape


def _conv2d_generators(
    G, weight, C, H, W, stride, padding, dilation, groups, n_out, dtype, device
):
    """Apply conv2d to a generator matrix (Gc or Gb)."""
    if G.shape[1] == 0:
        return torch.zeros((n_out, 0), dtype=dtype, device=device)
    ncols = G.shape[1]
    imgs = G.t().contiguous().view(ncols, C, H, W)
    out = F.conv2d(
        imgs,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return out.permute(1, 2, 3, 0).contiguous().reshape(-1, ncols)


def hz_conv2d(
    hz: HZono, weight, bias, stride, padding, dilation, groups, input_shape
) -> HZono:
    """Apply conv2d to a hybrid zonotope: convolve center and each generator column."""
    dtype, device = hz.c.dtype, hz.c.device
    C, H, W = _parse_input_shape(input_shape)
    weight = weight.to(dtype=dtype, device=device)

    c_img = hz.c.view(C, H, W).unsqueeze(0)
    out_c = F.conv2d(
        c_img,
        weight,
        bias=bias.to(dtype=dtype, device=device) if bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    new_c = out_c.reshape(-1, 1)
    n_out = new_c.shape[0]

    new_Gc = _conv2d_generators(
        hz.Gc, weight, C, H, W, stride, padding, dilation, groups, n_out, dtype, device
    )
    new_Gb = _conv2d_generators(
        hz.Gb, weight, C, H, W, stride, padding, dilation, groups, n_out, dtype, device
    )

    return HZono(
        c=new_c,
        Gc=new_Gc,
        Gb=new_Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
    )


# ============================================================================
# 5. Order reduction
# ============================================================================


def hz_reduce(hz: HZono, max_order: float = 10.0) -> HZono:
    """Reduce HZ complexity via Girard's method (sound over-approximation)."""
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    if n == 0:
        return hz

    max_ng = max(int(max_order * n), n + 1)
    max_nb = max(2 * n, 1)

    # Step 1: Relax excess binary generators to continuous
    if nb > max_nb:
        col_norms = hz.Gb.abs().sum(dim=0)
        _, sorted_idx = col_norms.sort()
        n_relax = nb - max_nb
        relax_idx = sorted_idx[:n_relax]
        keep_idx = sorted_idx[n_relax:]
        extra_Gc = hz.Gb[:, relax_idx]
        extra_Ac = (
            hz.Ab[:, relax_idx]
            if nc > 0
            else torch.zeros((0, n_relax), dtype=dtype, device=device)
        )
        hz = HZono(
            c=hz.c,
            Gc=torch.cat([hz.Gc, extra_Gc], dim=1),
            Gb=hz.Gb[:, keep_idx],
            Ac=torch.cat([hz.Ac, extra_Ac], dim=1)
            if nc > 0
            else torch.zeros((0, ng + n_relax), dtype=dtype, device=device),
            Ab=hz.Ab[:, keep_idx]
            if nc > 0
            else torch.zeros((0, max_nb), dtype=dtype, device=device),
            b=hz.b.clone(),
        )
        ng = hz.Gc.shape[1]
        nb = hz.Gb.shape[1]

    # Step 2: Reduce continuous generators
    if ng > max_ng:
        col_norms = hz.Gc.abs().sum(dim=0)
        _, sorted_idx = col_norms.sort(descending=True)
        keep_idx = sorted_idx[: max_ng - n]
        drop_idx = sorted_idx[max_ng - n :]
        Gc_keep = hz.Gc[:, keep_idx]
        new_Gc = torch.cat(
            [Gc_keep, torch.diag(hz.Gc[:, drop_idx].abs().sum(dim=1))], dim=1
        )

        if nc > 0:
            drop_set = set(drop_idx.tolist())
            keep_rows = [
                r
                for r in range(nc)
                if not any(abs(hz.Ac[r, c].item()) > 1e-15 for c in drop_set)
            ]
            if keep_rows:
                krt = torch.tensor(keep_rows, dtype=torch.long, device=device)
                new_Ac = torch.cat(
                    [
                        hz.Ac[krt][:, keep_idx],
                        torch.zeros((len(keep_rows), n), dtype=dtype, device=device),
                    ],
                    dim=1,
                )
                new_Ab = hz.Ab[krt]
                new_b = hz.b[krt]
            else:
                new_Ac = torch.zeros((0, new_Gc.shape[1]), dtype=dtype, device=device)
                new_Ab = torch.zeros((0, nb), dtype=dtype, device=device)
                new_b = torch.zeros((0, 1), dtype=dtype, device=device)
        else:
            new_Ac = torch.zeros((0, new_Gc.shape[1]), dtype=dtype, device=device)
            new_Ab = torch.zeros((0, nb), dtype=dtype, device=device)
            new_b = torch.zeros((0, 1), dtype=dtype, device=device)

        hz = HZono(c=hz.c, Gc=new_Gc, Gb=hz.Gb, Ac=new_Ac, Ab=new_Ab, b=new_b)

    return hz


# ============================================================================
# 6. Bounds computation
# ============================================================================


def _hz_is_unconstrained(hz: HZono) -> bool:
    """Check if Ac, Ab, b are all near-zero (no active constraints)."""
    tol = 1e-12
    return (
        torch.all(torch.abs(hz.Ac) < tol).item()
        and torch.all(torch.abs(hz.Ab) < tol).item()
        and torch.all(torch.abs(hz.b) < tol).item()
    )


def _hz_bounds_unconstrained(hz: HZono) -> Bounds:
    """Fast path: lb = c - |Gc|_rowsum - |Gb|_rowsum."""
    n = hz.c.shape[0]
    dtype, device = hz.c.dtype, hz.c.device
    absGc = (
        hz.Gc.abs().sum(dim=1, keepdim=True)
        if hz.Gc.numel()
        else torch.zeros((n, 1), dtype=dtype, device=device)
    )
    absGb = (
        hz.Gb.abs().sum(dim=1, keepdim=True)
        if hz.Gb.numel()
        else torch.zeros((n, 1), dtype=dtype, device=device)
    )
    rad = absGc + absGb
    return Bounds(lb=(hz.c - rad).flatten(), ub=(hz.c + rad).flatten())


def hz_compute_bounds(hz: HZono) -> Bounds:
    """Compute bounds: unconstrained fast path -> Gurobi -> SciPy -> unconstrained fallback."""
    if _hz_is_unconstrained(hz):
        return _hz_bounds_unconstrained(hz)
    if _HAS_GUROBI:
        try:
            return _hz_compute_bounds_gurobi(hz)
        except Exception:
            pass
    if _HAS_SCIPY:
        try:
            return _hz_compute_bounds_scipy(hz)
        except Exception:
            pass
    return _hz_bounds_unconstrained(hz)


def _hz_compute_bounds_gurobi(hz: HZono) -> Bounds:
    """Delegate to GurobiSolver.compute_hz_bounds (single source for all Gurobi code)."""
    return GurobiSolver.compute_hz_bounds(hz)


def _hz_compute_bounds_scipy(hz: HZono) -> Bounds:
    """SciPy LP relaxation (fallback, treats binary generators as continuous)."""
    n = int(hz.c.shape[0])
    p = int(hz.Gc.shape[1])
    q = int(hz.Gb.shape[1])
    c_np = hz.c.detach().cpu().numpy().astype("float64").reshape(-1)
    Gc_np = hz.Gc.detach().cpu().numpy().astype("float64")
    Gb_np = hz.Gb.detach().cpu().numpy().astype("float64")
    Ac_np = hz.Ac.detach().cpu().numpy().astype("float64")
    Ab_np = hz.Ab.detach().cpu().numpy().astype("float64")
    b_np = hz.b.detach().cpu().numpy().astype("float64").reshape(-1)

    A_eq = (
        np.concatenate([Ac_np, Ab_np], axis=1) if (Ac_np.size or Ab_np.size) else None
    )
    b_eq = b_np if (A_eq is not None) else None
    var_bounds = [(-1.0, 1.0)] * (p + q)

    LB = np.empty((n,), dtype=np.float64)
    UB = np.empty((n,), dtype=np.float64)
    for i in range(n):
        obj = np.concatenate([Gc_np[i], Gb_np[i]], axis=0)
        res_min = linprog(
            c=obj, A_eq=A_eq, b_eq=b_eq, bounds=var_bounds, method="highs"
        )
        if not res_min.success:
            raise RuntimeError(
                f"[linprog] MIN infeasible at dim {i}: {res_min.message}"
            )
        LB[i] = c_np[i] + res_min.fun
        res_max = linprog(
            c=-obj, A_eq=A_eq, b_eq=b_eq, bounds=var_bounds, method="highs"
        )
        if not res_max.success:
            raise RuntimeError(
                f"[linprog] MAX infeasible at dim {i}: {res_max.message}"
            )
        UB[i] = c_np[i] - res_max.fun

    dtype, device = hz.c.dtype, hz.c.device
    return Bounds(
        lb=torch.from_numpy(LB).to(device=device, dtype=dtype).flatten(),
        ub=torch.from_numpy(UB).to(device=device, dtype=dtype).flatten(),
    )


# ============================================================================
# 7. HZSolver — Solver subclass for hybrid zonotope bounds
# ============================================================================


from act.back_end.solver.solver_base import Solver, SolverCaps


class HZSolver(Solver):
    """Hybrid Zonotope bounds solver.

    Precision hierarchy:
      GurobiSolver (MILP, exact) > HZSolver (HZ, tight) > interval (box, fast)

    Operates on HZono objects directly via compute_bounds(), not via CSP.
    """

    def __init__(self):
        self._last_bounds = None

    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=False)

    def compute_bounds(self, hz: HZono) -> Bounds:
        return hz_compute_bounds(hz)

    def begin(self, name="verify", device=None):
        pass

    def status(self):
        return "UNKNOWN"

    def has_solution(self):
        return False

    @property
    def n(self):
        return 0

    def _csp_unsupported(self, *a, **kw):
        raise NotImplementedError("HZSolver operates on HZono, not CSP")

    add_vars = set_bounds = add_binary_vars = _csp_unsupported
    add_lin_eq = add_lin_ge = add_lin_le = _csp_unsupported
    set_objective_linear = optimize = get_values = _csp_unsupported
