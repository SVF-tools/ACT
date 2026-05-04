# ===- act/back_end/hybridz_tf/tf_mlp.py - HybridZ MLP Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ MLP Transfer Functions. Implements HybridZ-based transfer functions
#   for MLP layers including dense, activation, and element-wise operations.
#
# ===---------------------------------------------------------------------===#

import torch
import torch.nn.functional as F
from act.back_end.core import Bounds, Fact
from act.back_end.solver.solver_hz import (
    HZono,
    hz_multiply,
    hz_add_const,
    hz_minkowski_sum,
    hz_from_bounds,
    hz_compute_bounds,
)
import act.back_end.interval_tf.tf_mlp as interval


# ============================================================================
# HZ layer functions: HZono -> Optional[HZono] per layer kind
# Each takes (L, hz_in, tf) and returns the transformed HZono or None.
# ============================================================================


# --- HZ transfer functions (MLP) ---


def tf_dense(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        dtype, device = hz_in.c.dtype, hz_in.c.device
        hz = hz_multiply(hz_in, L.params["weight"])
        b = L.params.get("bias")
        if b is not None:
            b_col = b.to(dtype=dtype, device=device)
            hz = hz_add_const(hz, b_col.view(-1, 1) if b_col.ndim == 1 else b_col)
        tf._hz_cache[L.id] = hz
    fact = interval.tf_dense(L, bounds)
    if hz_in is not None:
        return Fact(bounds=hz_compute_bounds(tf._hz_cache[L.id]), cons=fact.cons)
    return fact


def tf_bias(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        dtype, device = hz_in.c.dtype, hz_in.c.device
        c = L.params["c"].to(dtype=dtype, device=device)
        tf._hz_cache[L.id] = hz_add_const(hz_in, c.view(-1, 1) if c.ndim == 1 else c)
    fact = interval.tf_bias(L, bounds)
    if hz_in is not None:
        return Fact(bounds=hz_compute_bounds(tf._hz_cache[L.id]), cons=fact.cons)
    return fact


def tf_scale(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        dtype, device = hz_in.c.dtype, hz_in.c.device
        a = L.params["a"].to(dtype=dtype, device=device).flatten()
        tf._hz_cache[L.id] = hz_multiply(hz_in, torch.diag(a))
    fact = interval.tf_scale(L, bounds)
    if hz_in is not None:
        return Fact(bounds=hz_compute_bounds(tf._hz_cache[L.id]), cons=fact.cons)
    return fact


def tf_relu(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = hz_reduce(hz_apply_relu(hz_in))
    fact = interval.tf_relu(L, bounds)
    if hz_in is not None:
        return Fact(bounds=hz_compute_bounds(tf._hz_cache[L.id]), cons=fact.cons)
    return fact


def tf_lrelu(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = hz_reduce(
            hz_apply_leaky_relu(hz_in, float(L.params.get("negative_slope", 0.01)))
        )
    fact = interval.tf_lrelu(L, bounds)
    if hz_in is not None:
        return Fact(bounds=hz_compute_bounds(tf._hz_cache[L.id]), cons=fact.cons)
    return fact


def tf_tanh(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = hz_reduce(hz_apply_tanh(hz_in, K=tf._tanh_K))
    fact = interval.tf_tanh(L, bounds)
    if hz_in is not None:
        return Fact(bounds=hz_compute_bounds(tf._hz_cache[L.id]), cons=fact.cons)
    return fact


def tf_sigmoid(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = hz_reduce(hz_apply_sigmoid(hz_in, K=tf._sigmoid_K))
    fact = interval.tf_sigmoid(L, bounds)
    if hz_in is not None:
        return Fact(bounds=hz_compute_bounds(tf._hz_cache[L.id]), cons=fact.cons)
    return fact


def tf_abs(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        dtype, device = hz_in.c.dtype, hz_in.c.device
        bds = hz_compute_bounds(hz_in)
        lb_out = torch.where(
            bds.lb >= 0,
            bds.lb,
            torch.where(bds.ub <= 0, -bds.ub, torch.zeros_like(bds.lb)),
        )
        tf._hz_cache[L.id] = hz_from_bounds(
            Bounds(lb=lb_out, ub=torch.maximum(bds.lb.abs(), bds.ub.abs())),
            dtype,
            device,
        )
    fact = interval.tf_abs(L, bounds)
    if hz_in is not None:
        return Fact(bounds=hz_compute_bounds(tf._hz_cache[L.id]), cons=fact.cons)
    return fact


def tf_bn(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        dtype, device = hz_in.c.dtype, hz_in.c.device
        A = L.params["A"].to(dtype=dtype, device=device).flatten()
        c = L.params["c"].to(dtype=dtype, device=device)
        hz = hz_multiply(hz_in, torch.diag(A))
        tf._hz_cache[L.id] = hz_add_const(hz, c.view(-1, 1) if c.ndim == 1 else c)
    fact = interval.tf_bn(L, bounds)
    if hz_in is not None:
        return Fact(bounds=hz_compute_bounds(tf._hz_cache[L.id]), cons=fact.cons)
    return fact


def tf_add(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            tf._hz_cache[L.id] = hz_minkowski_sum(hz_in, hz2)
        else:
            hz_in = None
    fact = interval.tf_add(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )
    if hz_in is not None:
        return Fact(bounds=hz_compute_bounds(tf._hz_cache[L.id]), cons=fact.cons)
    return fact


def tf_sub(L, bounds, tf):
    """SUB: y = x - z. HZ form: minkowski_sum(hz_x, -hz_z)."""
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            hz2_neg = HZono(
                c=-hz2.c, Gc=-hz2.Gc, Gb=-hz2.Gb,
                Ac=hz2.Ac.clone(), Ab=hz2.Ab.clone(), b=hz2.b.clone(),
            )
            tf._hz_cache[L.id] = hz_minkowski_sum(hz_in, hz2_neg)
        else:
            hz_in = None
    fact = interval.tf_sub(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )
    if hz_in is not None:
        return Fact(bounds=hz_compute_bounds(tf._hz_cache[L.id]), cons=fact.cons)
    return fact


def tf_mul(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        dtype, device = hz_in.c.dtype, hz_in.c.device
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            b1, b2 = hz_compute_bounds(hz_in), hz_compute_bounds(hz2)
            corners = torch.stack(
                [b1.lb * b2.lb, b1.lb * b2.ub, b1.ub * b2.lb, b1.ub * b2.ub]
            )
            tf._hz_cache[L.id] = hz_from_bounds(
                Bounds(lb=corners.min(0)[0], ub=corners.max(0)[0]), dtype, device
            )
        else:
            hz_in = None
    fact = interval.tf_mul(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )
    if hz_in is not None:
        return Fact(bounds=hz_compute_bounds(tf._hz_cache[L.id]), cons=fact.cons)
    return fact


def tf_concat(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        parts = [tf._hz_cache.get(pid) for pid in preds]
        if all(p is not None for p in parts):
            result = parts[0]
            for p in parts[1:]:
                result = hz_minkowski_sum(result, p)
            tf._hz_cache[L.id] = result
        else:
            hz_in = None
    fact = interval.tf_concat(
        L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
    )
    if hz_in is not None:
        return Fact(bounds=hz_compute_bounds(tf._hz_cache[L.id]), cons=fact.cons)
    return fact


# --- HZ activation encodings (zonotope domain) ---


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
    """Exact LeakyReLU via the same encoding as ReLU.

    Per unstable neuron: ng += 4 (xi1, xi2, xi3, xi4), nb += 1 (z), nc += 3
    (graph eq 1, graph eq 2, linking eq) -- identical to hz_apply_relu.

    Decomposition: y = max(s*x, x) where s = alpha_arg. On the unstable
    branch, using the same switching mechanism as ReLU (z=+1 -> inactive
    with xi2 forced to 1; z=-1 -> active with xi1 forced to 1), we set
    the output as::

        y_h = beta/2 + (s*alpha/2) xi1 - (beta/2) xi2 + (s*alpha/2) z

    which degenerates exactly to ReLU's ``y_h = (beta/2)(1 - xi2)`` when
    s = 0. The graph equalities (xi1+xi3+z=1, xi2+xi4-z=1) and the linking
    equality (that ties x_h to xi1, xi2, z) are identical to ReLU.
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]
    s = alpha_arg
    assert 0.0 <= s <= 1.0, f"hz_apply_leaky_relu: slope must be in [0, 1], got {s}"

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

    if inactive.any():
        out_c[inactive] = s * hz.c[inactive]
        out_Gc[inactive, :ng] = s * hz.Gc[inactive]
        out_Gb[inactive, :nb] = s * hz.Gb[inactive]

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

    # Output encoding: y_h = beta/2 + (s*alpha/2) xi1 - (beta/2) xi2 + (s*alpha/2) z
    out_c[unstable_idx, 0] = beta / 2.0
    out_Gc[unstable_idx, col_xi1] = s * alpha / 2.0
    out_Gc[unstable_idx, col_xi2] = -beta / 2.0
    out_Gb[unstable_idx, col_z] = s * alpha / 2.0

    ng_new = ng + 4 * k
    nb_new = nb + k

    eq_Ac = torch.zeros((3 * k, ng_new), dtype=dtype, device=device)
    eq_Ab = torch.zeros((3 * k, nb_new), dtype=dtype, device=device)
    eq_b = torch.zeros((3 * k, 1), dtype=dtype, device=device)

    r1 = 3 * t
    r2 = 3 * t + 1

    # Graph equality 1: xi1 + xi3 + z = 1
    eq_Ac[r1, col_xi1] = 1.0
    eq_Ac[r1, col_xi3] = 1.0
    eq_Ab[r1, col_z] = 1.0
    eq_b[r1, 0] = 1.0

    # Graph equality 2: xi2 + xi4 - z = 1
    eq_Ac[r2, col_xi2] = 1.0
    eq_Ac[r2, col_xi4] = 1.0
    eq_Ab[r2, col_z] = -1.0
    eq_b[r2, 0] = 1.0

    # Linking equality: ties x_h to (xi1, xi2, z)
    # Same form as ReLU; x_h has the same input expression.
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


# --- HZ order reduction ---


def hz_reduce(hz: HZono, max_order: float = 3.0) -> HZono:
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
            has_dropped = hz.Ac[:, drop_idx].abs().max(dim=1).values > 1e-15
            keep_mask = ~has_dropped
            krt = torch.where(keep_mask)[0]
            if krt.numel() > 0:
                new_Ac = torch.cat(
                    [
                        hz.Ac[krt][:, keep_idx],
                        torch.zeros((krt.numel(), n), dtype=dtype, device=device),
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