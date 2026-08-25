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
try:
    import numpy as np
    import scipy.sparse as sp
except ImportError:
    np = None
    sp = None
from act.back_end.core import Bounds, Fact
from act.back_end.solver.solver_hz import (
    HZono,
    SparseHZono,
    hz_multiply,
    hz_add_const,
    hz_from_bounds,
    hz_fresh_col_ids,
    hz_compute_bounds,
    hz_concat,
    hz_sgm_add,
    hz_sub,
    sparse_hz_add_const,
    sparse_hz_add_same_frame,
    sparse_hz_concat,
    sparse_hz_fast_bounds,
    sparse_hz_from_bounds,
    sparse_hz_gather_rows,
    sparse_hz_is_point,
    sparse_hz_linear,
    sparse_hz_pad_frame,
    sparse_hz_reduce_sum_rows,
    sparse_hz_scale,
    sparse_hz_sub_same_frame,
)
import act.back_end.interval_tf.tf_mlp as interval
import act.back_end.interval_tf.tf_cnn as interval_cnn


def _hz_fact(fact: Fact, hz: HZono) -> Fact:
    hb = hz_compute_bounds(hz)
    lb = torch.maximum(hb.lb.reshape_as(fact.bounds.lb), fact.bounds.lb)
    ub = torch.minimum(hb.ub.reshape_as(fact.bounds.ub), fact.bounds.ub)
    return Fact(
        bounds=Bounds(lb=lb, ub=ub),
        cons=fact.cons,
    )


def _hz_exceeds_limit(tf, L, hz: HZono) -> bool:
    ngnb = hz.Gc.shape[1] + hz.Gb.shape[1]
    return max(len(L.out_vars), ngnb) > tf._HZ_MAX_INPUT_DIM


def _hz_apply_per_batch_linear(hz: HZono, W: torch.Tensor, B: int) -> HZono:
    in_dim = W.shape[1]
    out_dim = W.shape[0]
    if B == 1:
        return hz_multiply(hz, W)
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    c3 = hz.c.view(B, in_dim, 1)
    new_c = (W @ c3).reshape(B * out_dim, 1)
    if ng:
        new_Gc = (W @ hz.Gc.view(B, in_dim, ng)).reshape(B * out_dim, ng)
    else:
        new_Gc = hz.Gc.new_zeros(B * out_dim, 0)
    if nb:
        new_Gb = (W @ hz.Gb.view(B, in_dim, nb)).reshape(B * out_dim, nb)
    else:
        new_Gb = hz.Gb.new_zeros(B * out_dim, 0)
    return HZono(
        c=new_c, Gc=new_Gc, Gb=new_Gb,
        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


def _hz_add_per_channel(hz: HZono, v: torch.Tensor, B: int) -> HZono:
    v = v.to(dtype=hz.c.dtype, device=hz.c.device).flatten()
    if B > 1:
        v = v.repeat(B)
    return hz_add_const(hz, v.view(-1, 1))


def _hz_scale_per_channel(hz: HZono, a: torch.Tensor, B: int) -> HZono:
    a = a.to(dtype=hz.c.dtype, device=hz.c.device).flatten()
    if B > 1:
        a = a.repeat(B)
    a_col = a.view(-1, 1)
    return HZono(
        c=a_col * hz.c,
        Gc=a_col * hz.Gc,
        Gb=a_col * hz.Gb,
        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


def _hz_is_point(hz: HZono) -> bool:
    gc_zero = hz.Gc.numel() == 0 or bool((hz.Gc.abs() <= 1e-12).all())
    gb_zero = hz.Gb.numel() == 0 or bool((hz.Gb.abs() <= 1e-12).all())
    return gc_zero and gb_zero


def _broadcast_flat(v: torch.Tensor, n: int) -> torch.Tensor:
    v = v.flatten()
    if v.numel() == n:
        return v
    if v.numel() == 1:
        return v.expand(n)
    if n % v.numel() == 0:
        return v.repeat(n // v.numel())
    raise ValueError(f"cannot broadcast {v.numel()} values to {n}")


def _hz_scale_elementwise(hz: HZono, a: torch.Tensor) -> HZono:
    a = _broadcast_flat(a.to(dtype=hz.c.dtype, device=hz.c.device), hz.c.shape[0])
    acol = a.view(-1, 1)
    return HZono(
        c=acol * hz.c,
        Gc=acol * hz.Gc,
        Gb=acol * hz.Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


def _prod(shape) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def _shared_const_block(flat: torch.Tensor, shape, batch: int):
    dim = _prod(shape)
    if dim == 0:
        return None
    if flat.numel() == dim:
        return flat.view(*shape)
    if batch > 0 and flat.numel() == batch * dim:
        blocks = flat.view(batch, dim)
        if bool(torch.allclose(blocks, blocks[:1].expand_as(blocks))):
            return blocks[0].view(*shape)
    return None


def _hz_matmul_const(L, hz: HZono, const: torch.Tensor, *, variable_is_left: bool):
    dtype, device = hz.c.dtype, hz.c.device
    x_shape = tuple(int(d) for d in L.params["x_shape"])
    y_shape = tuple(int(d) for d in L.params["y_shape"])
    in_shape = x_shape if variable_is_left else y_shape
    in_dim = _prod(in_shape)
    if in_dim == 0 or hz.c.shape[0] % in_dim != 0:
        return None
    B = hz.c.shape[0] // in_dim
    C = const.to(dtype=dtype, device=device).flatten()
    if variable_is_left:
        W = _shared_const_block(C, y_shape, B)
        if W is None:
            return None
        eye = torch.eye(in_dim, dtype=dtype, device=device).view(in_dim, *x_shape)
        out = torch.matmul(eye, W).reshape(in_dim, -1)
    else:
        W = _shared_const_block(C, x_shape, B)
        if W is None:
            return None
        eye = torch.eye(in_dim, dtype=dtype, device=device).view(in_dim, *y_shape)
        out = torch.matmul(W, eye).reshape(in_dim, -1)
    return _hz_apply_per_batch_linear(hz, out.t().contiguous(), B)


def _sparse_available() -> bool:
    return np is not None and sp is not None


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)


def _sparse_param_vector(value, n: int):
    arr = _to_numpy(value).reshape(-1)
    if arr.size == n:
        return arr
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=np.float64)
    if n % arr.size == 0:
        return np.tile(arr, n // arr.size)
    raise ValueError(f"cannot broadcast {arr.size} sparse values to {n}")


def _sparse_apply_per_batch_linear(hz: SparseHZono, W, bias=None) -> SparseHZono:
    Wsp = W.tocsr().astype(np.float64) if sp.issparse(W) else sp.csr_matrix(_to_numpy(W))
    in_dim = int(Wsp.shape[1])
    if in_dim == 0 or hz.n_out % in_dim != 0:
        raise ValueError(f"sparse batch linear shape mismatch: {hz.n_out} vs {Wsp.shape}")
    B = hz.n_out // in_dim
    M = sp.kron(sp.eye(B, format="csr"), Wsp, format="csr") if B != 1 else Wsp
    b = None
    if bias is not None:
        b0 = _to_numpy(bias).reshape(-1)
        b = np.tile(b0, B) if B > 1 else b0
    return sparse_hz_linear(hz, M, b)


def _sparse_matmul_const(L, hz: SparseHZono, const, *, variable_is_left: bool):
    x_shape = tuple(int(d) for d in L.params["x_shape"])
    y_shape = tuple(int(d) for d in L.params["y_shape"])
    in_shape = x_shape if variable_is_left else y_shape
    in_dim = _prod(in_shape)
    if in_dim == 0 or hz.n_out % in_dim != 0:
        return None
    B = hz.n_out // in_dim
    C = torch.as_tensor(_to_numpy(const), dtype=torch.float64).flatten()
    if variable_is_left:
        W = _shared_const_block(C, y_shape, B)
        if W is None:
            return None
        eye = torch.eye(in_dim, dtype=torch.float64).view(in_dim, *x_shape)
        out = torch.matmul(eye, W).reshape(in_dim, -1)
    else:
        W = _shared_const_block(C, x_shape, B)
        if W is None:
            return None
        eye = torch.eye(in_dim, dtype=torch.float64).view(in_dim, *y_shape)
        out = torch.matmul(W, eye).reshape(in_dim, -1)
    return _sparse_apply_per_batch_linear(hz, sp.csr_matrix(out.t().numpy()))


def _sparse_triplets(parts, shape):
    parts = [
        (
            np.asarray(rows, dtype=np.int64).reshape(-1),
            np.asarray(cols, dtype=np.int64).reshape(-1),
            np.asarray(data, dtype=np.float64).reshape(-1),
        )
        for rows, cols, data in parts
        if np.asarray(data).size
    ]
    if not parts:
        return sp.csr_matrix(shape, dtype=np.float64)
    return sp.coo_matrix(
        (
            np.concatenate([part[2] for part in parts]),
            (
                np.concatenate([part[0] for part in parts]),
                np.concatenate([part[1] for part in parts]),
            ),
        ),
        shape=shape,
        dtype=np.float64,
    ).tocsr()


def _sparse_relu_bounds(hz: SparseHZono, input_bounds: Bounds):
    hb = sparse_hz_fast_bounds(hz)
    fact_lb = _to_numpy(input_bounds.lb).reshape(-1)
    fact_ub = _to_numpy(input_bounds.ub).reshape(-1)
    hz_lb = _to_numpy(hb.lb).reshape(-1)
    hz_ub = _to_numpy(hb.ub).reshape(-1)
    if fact_lb.size != hz.n_out or hz_lb.size != hz.n_out:
        raise ValueError("sparse ReLU bounds shape mismatch")
    lb = np.maximum(fact_lb, hz_lb)
    ub = np.minimum(fact_ub, hz_ub)
    if np.any(lb > ub):
        raise ValueError("sparse ReLU received inconsistent bounds")
    return lb, ub


def sparse_hz_apply_relu_exact(
    hz: SparseHZono,
    lb,
    ub,
    slots,
    n_cont: int,
    n_bin: int,
) -> SparseHZono:
    """Apply the compressed exact ReLU graph in one shared sparse frame."""
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    active_idx = np.flatnonzero(lb >= 0.0).astype(np.int64)
    unstable_idx = np.flatnonzero((lb < 0.0) & (ub > 0.0)).astype(np.int64)
    k = int(unstable_idx.size)
    if len(slots) != k:
        raise ValueError("sparse ReLU slot count mismatch")

    padded = sparse_hz_pad_frame(hz, n_cont, n_bin)
    out_c = np.zeros(hz.n_out, dtype=np.float64)
    gc_parts = []
    gb_parts = []
    if active_idx.size:
        out_c[active_idx] = hz.c[active_idx]
        active_gc = padded.Gc[active_idx].tocoo()
        gc_parts.append(
            (active_idx[active_gc.row], active_gc.col, active_gc.data)
        )
        active_gb = padded.Gb[active_idx].tocoo()
        gb_parts.append(
            (active_idx[active_gb.row], active_gb.col, active_gb.data)
        )

    eq_c_parts = []
    eq_b_parts = []
    ineq_c_parts = []
    ineq_b_parts = []
    if k:
        slot_array = np.asarray(slots, dtype=np.int64)
        xi1_cols = slot_array[:, 0]
        xi2_cols = slot_array[:, 1]
        z_cols = slot_array[:, 2]
        rows = np.arange(k, dtype=np.int64)
        alpha = lb[unstable_idx]
        beta = ub[unstable_idx]

        out_c[unstable_idx] = beta / 2.0
        gc_parts.append((unstable_idx, xi2_cols, -beta / 2.0))

        eq_c_parts.extend(
            [
                (rows, xi1_cols, alpha / 2.0),
                (rows, xi2_cols, -beta / 2.0),
            ]
        )
        pre_gc = hz.Gc[unstable_idx].tocoo()
        eq_c_parts.append((pre_gc.row, pre_gc.col, -pre_gc.data))
        eq_b_parts.append((rows, z_cols, alpha / 2.0))
        pre_gb = hz.Gb[unstable_idx].tocoo()
        eq_b_parts.append((pre_gb.row, pre_gb.col, -pre_gb.data))

        ineq_c_parts.extend(
            [
                (rows, xi1_cols, -np.ones(k, dtype=np.float64)),
                (k + rows, xi2_cols, -np.ones(k, dtype=np.float64)),
            ]
        )
        ineq_b_parts.extend(
            [
                (rows, z_cols, -np.ones(k, dtype=np.float64)),
                (k + rows, z_cols, np.ones(k, dtype=np.float64)),
            ]
        )
        eq_rhs = hz.c[unstable_idx] - beta / 2.0
    else:
        eq_rhs = np.zeros(0, dtype=np.float64)

    out_Gc = _sparse_triplets(gc_parts, (hz.n_out, n_cont))
    out_Gb = _sparse_triplets(gb_parts, (hz.n_out, n_bin))
    eq_Ac = _sparse_triplets(eq_c_parts, (k, n_cont))
    eq_Ab = _sparse_triplets(eq_b_parts, (k, n_bin))
    ineq_Ac = _sparse_triplets(ineq_c_parts, (2 * k, n_cont))
    ineq_Ab = _sparse_triplets(ineq_b_parts, (2 * k, n_bin))
    return SparseHZono(
        c=out_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=sp.vstack([padded.Ac, eq_Ac], format="csr"),
        Ab=sp.vstack([padded.Ab, eq_Ab], format="csr"),
        b=np.concatenate([padded.b, eq_rhs]),
        Auc=sp.vstack([padded.Auc, ineq_Ac], format="csr"),
        Aub=sp.vstack([padded.Aub, ineq_Ab], format="csr"),
        ub=np.concatenate([padded.ub, np.zeros(2 * k, dtype=np.float64)]),
        frame_id=hz.frame_id,
    )


def _sparse_apply_relu(L, hz: SparseHZono, input_bounds: Bounds, tf):
    lb, ub = _sparse_relu_bounds(hz, input_bounds)
    unstable_idx = np.flatnonzero((lb < 0.0) & (ub > 0.0)).astype(np.int64)
    reservation = tf._sparse_relu_slots_for(hz, L.id, unstable_idx)
    if reservation is None:
        return None, "sparse_relu_size_limit"
    slots, n_cont, n_bin = reservation
    return sparse_hz_apply_relu_exact(hz, lb, ub, slots, n_cont, n_bin), None


def _transpose_rows(
    L, n_rows: int, width: int, device, input_shape
) -> torch.Tensor:
    local = interval._transpose_flat_indices(
        L, width, device, input_shape=input_shape
    )
    width = int(local.numel())
    if width == 0 or n_rows % width != 0:
        raise ValueError(
            f"transpose: {n_rows} HZ rows are incompatible with width {width}"
        )
    batch = n_rows // width
    offsets = torch.arange(batch, device=local.device).unsqueeze(1) * width
    return (local.unsqueeze(0) + offsets).reshape(-1)


def sparse_hz_apply_layer(L, hz: SparseHZono, input_bounds: Bounds, result: Fact, tf):
    if not _sparse_available():
        return True, None, "scipy_unavailable"
    k = L.kind.upper()
    if k == "DENSE":
        out = _sparse_apply_per_batch_linear(hz, L.params["weight"], L.params.get("bias"))
        return True, out, None
    if k == "BIAS":
        return True, sparse_hz_add_const(hz, _sparse_param_vector(L.params["c"], hz.n_out)), None
    if k == "SCALE":
        return True, sparse_hz_scale(hz, _sparse_param_vector(L.params["a"], hz.n_out)), None
    if k == "BN":
        out = sparse_hz_scale(hz, _sparse_param_vector(L.params["A"], hz.n_out))
        return True, sparse_hz_add_const(out, _sparse_param_vector(L.params["c"], hz.n_out)), None
    if k == "RELU":
        out, reason = _sparse_apply_relu(L, hz, input_bounds, tf)
        return True, out, reason
    if k == "MUL":
        preds = tf._net.preds.get(L.id, [])
        other = tf._sparse_hz_cache.get(preds[1]) if len(preds) > 1 else None
        if other is None:
            return True, None, "missing_sparse_mul_input"
        if sparse_hz_is_point(other):
            return True, sparse_hz_scale(hz, other.c), None
        if sparse_hz_is_point(hz):
            return True, sparse_hz_scale(other, hz.c), None
        return True, None, "unsupported_sparse_mul_var_var"
    if k == "MATMUL":
        preds = tf._net.preds.get(L.id, [])
        other = tf._sparse_hz_cache.get(preds[1]) if len(preds) > 1 else None
        if other is None:
            return True, None, "missing_sparse_matmul_input"
        out = None
        if sparse_hz_is_point(other):
            out = _sparse_matmul_const(L, hz, other.c, variable_is_left=True)
        elif sparse_hz_is_point(hz):
            out = _sparse_matmul_const(L, other, hz.c, variable_is_left=False)
        return (True, out, None) if out is not None else (True, None, "unsupported_sparse_matmul")
    if k in {"FLATTEN", "RESHAPE", "SQUEEZE", "UNSQUEEZE"}:
        return True, hz, None
    if k == "TRANSPOSE":
        width = input_bounds.lb.numel() // input_bounds.lb.shape[0]
        rows = _transpose_rows(L, hz.n_out, width, result.bounds.lb.device, (1, *tuple(int(d) for d in input_bounds.lb.shape[1:])),)
        return True, sparse_hz_gather_rows(hz, rows.detach().cpu().numpy()), None
    if k == "UPSAMPLE":
        rows = _row_indices_upsample_nearest(L, hz.n_out, result.bounds.lb.numel())
        return (
            (True, sparse_hz_gather_rows(hz, rows.detach().cpu().numpy()), None)
            if rows is not None
            else (True, None, "unsupported_sparse_upsample")
        )
    if k == "SLICE":
        rows = _row_indices_slice(L, hz.n_out)
        return (
            (True, sparse_hz_gather_rows(hz, rows.detach().cpu().numpy()), None)
            if rows is not None and rows.numel() == result.bounds.lb.numel()
            else (True, None, "unsupported_sparse_slice")
        )
    if k == "GATHER":
        rows = _row_indices_gather(L, hz.n_out)
        return (
            (True, sparse_hz_gather_rows(hz, rows.detach().cpu().numpy()), None)
            if rows is not None and rows.numel() == result.bounds.lb.numel()
            else (True, None, "unsupported_sparse_gather")
        )
    if k == "EXPAND":
        rows = _row_indices_expand(L, hz.n_out)
        return (
            (True, sparse_hz_gather_rows(hz, rows.detach().cpu().numpy()), None)
            if rows is not None and rows.numel() == result.bounds.lb.numel()
            else (True, None, "unsupported_sparse_expand")
        )
    if k == "REDUCE_SUM":
        rows = _row_indices_reduce_sum(L, hz.n_out, result.bounds.lb.numel())
        return (
            (True, sparse_hz_reduce_sum_rows(hz, rows.detach().cpu().numpy(), result.bounds.lb.numel()), None)
            if rows is not None
            else (True, None, "unsupported_sparse_reduce_sum")
        )
    if k == "ADD":
        preds = tf._net.preds.get(L.id, [])
        other = tf._sparse_hz_cache.get(preds[1]) if len(preds) > 1 else None
        return (
            (True, sparse_hz_add_same_frame(hz, other), None)
            if other is not None
            else (True, None, "missing_sparse_add_input")
        )
    if k == "SUB":
        preds = tf._net.preds.get(L.id, [])
        other = tf._sparse_hz_cache.get(preds[1]) if len(preds) > 1 else None
        return (
            (True, sparse_hz_sub_same_frame(hz, other), None)
            if other is not None
            else (True, None, "missing_sparse_sub_input")
        )
    if k == "CONCAT":
        preds = tf._net.preds.get(L.id, [])
        parts = [tf._sparse_hz_cache.get(pid) for pid in preds]
        return (
            (True, sparse_hz_concat(parts), None)
            if parts and all(p is not None for p in parts)
            else (True, None, "missing_sparse_concat_input")
        )
    if k == "CONSTANT":
        return True, sparse_hz_from_bounds(result.bounds, frame_id=hz.frame_id), None
    if k in {"LRELU", "SIGMOID", "TANH", "MAXPOOL2D"}:
        return True, None, f"unsupported_sparse_nonlinear:{k}"
    return False, None, None


# ============================================================================
# HZ layer functions: HZono -> Optional[HZono] per layer kind
# Each takes (L, hz_in, tf) and returns the transformed HZono or None.
# ============================================================================


# --- HZ transfer functions (MLP) ---


def tf_dense(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    sigmoid_in = tf._sigmoid_affine_inputs.get(L.id)
    if sigmoid_in is not None:
        W = L.params["weight"].to(sigmoid_in.c)
        bias = L.params.get("bias")
        bias = (
            sigmoid_in.c.new_zeros(W.shape[0])
            if bias is None
            else bias.to(sigmoid_in.c).flatten()
        )
        hz = hz_apply_sigmoid(sigmoid_in, tf._sigmoid_K, affine=(W, bias))
        fact = interval.tf_dense(L, bounds)
        if _hz_exceeds_limit(tf, L, hz):
            tf._hz_cache.pop(L.id, None)
            return fact
        tf._hz_cache[L.id] = hz
        return _hz_fact(fact, hz)
    if hz_in is not None:
        W = L.params["weight"].to(hz_in.c)
        in_dim = W.shape[1]
        B = hz_in.c.shape[0] // in_dim
        hz = _hz_apply_per_batch_linear(hz_in, W, B)
        bias = L.params.get("bias")
        if bias is not None:
            hz = _hz_add_per_channel(hz, bias, B)
        tf._hz_cache[L.id] = hz
    fact = interval.tf_dense(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_bias(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        c = L.params["c"].to(hz_in.c)
        if c.ndim == 1:
            B = hz_in.c.shape[0] // c.numel()
            tf._hz_cache[L.id] = _hz_add_per_channel(hz_in, c, B)
        else:
            tf._hz_cache[L.id] = hz_add_const(hz_in, c)
    fact = interval.tf_bias(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_scale(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        a = L.params["a"].to(hz_in.c).flatten()
        B = hz_in.c.shape[0] // a.numel()
        tf._hz_cache[L.id] = _hz_scale_per_channel(hz_in, a, B)
    fact = interval.tf_scale(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_relu(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    fact = interval.tf_relu(L, bounds)
    if hz_in is not None:
        hz_out = hz_apply_relu(hz_in)
        if _hz_exceeds_limit(tf, L, hz_out):
            tf._hz_cache.pop(L.id, None)
        else:
            tf._hz_cache[L.id] = hz_out
            return _hz_fact(fact, hz_out)
    return fact


def tf_lrelu(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    fact = interval.tf_lrelu(L, bounds)
    if hz_in is not None:
        hz_out = hz_apply_leaky_relu(hz_in, float(L.params["negative_slope"]))
        if _hz_exceeds_limit(tf, L, hz_out):
            tf._hz_cache.pop(L.id, None)
        else:
            tf._hz_cache[L.id] = hz_out
            return _hz_fact(fact, hz_out)
    return fact


def tf_tanh(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    fact = interval.tf_tanh(L, bounds)
    if hz_in is not None:
        hz_out = hz_apply_tanh(hz_in, K=tf._tanh_K)
        if _hz_exceeds_limit(tf, L, hz_out):
            tf._hz_cache.pop(L.id, None)
        else:
            tf._hz_cache[L.id] = hz_out
            return _hz_fact(fact, hz_out)
    return fact


def tf_sigmoid(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    fact = interval.tf_sigmoid(L, bounds)
    target = tf._sigmoid_affine_targets.get(L.id)
    if target is not None:
        tf._sigmoid_affine_inputs.pop(target, None)
        if hz_in is not None:
            W = tf._net.by_id[target].params["weight"]
            if hz_in.c.shape[0] == W.shape[1]:
                tf._sigmoid_affine_inputs[target] = hz_in
                tf._hz_cache.pop(L.id, None)
                return fact
    if hz_in is not None:
        hz_out = hz_apply_sigmoid(hz_in, K=tf._sigmoid_K)
        if _hz_exceeds_limit(tf, L, hz_out):
            tf._hz_cache.pop(L.id, None)
        else:
            tf._hz_cache[L.id] = hz_out
            return _hz_fact(fact, hz_out)
    return fact


def tf_erf(L, bounds, tf):
    tf._hz_cache[L.id] = None
    return interval.tf_erf(L, bounds)


def tf_sqrt(L, bounds, tf):
    tf._hz_cache[L.id] = None
    return interval.tf_sqrt(L, bounds)


def tf_sin(L, bounds, tf):
    tf._hz_cache[L.id] = None
    return interval.tf_sin(L, bounds)


def tf_cos(L, bounds, tf):
    tf._hz_cache[L.id] = None
    return interval.tf_cos(L, bounds)


def tf_quantize(L, bounds, tf):
    tf._hz_cache[L.id] = None
    return interval.tf_quantize(L, bounds)


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
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_bn(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        A, c = L.params["A"], L.params["c"]
        B = hz_in.c.shape[0] // A.numel()
        tf._hz_cache[L.id] = _hz_add_per_channel(
            _hz_scale_per_channel(hz_in, A, B), c, B
        )
    fact = interval.tf_bn(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_add(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            tf._hz_cache[L.id] = hz_sgm_add(hz_in, hz2)
        else:
            hz_in = None
    fact = interval.tf_add(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_mul(L, bounds, tf):
    bx = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0)
    by = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)
    fact = interval.tf_mul(L, bx, by)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            if _hz_is_point(hz2):
                tf._hz_cache[L.id] = _hz_scale_elementwise(hz_in, hz2.c)
            elif _hz_is_point(hz_in):
                tf._hz_cache[L.id] = _hz_scale_elementwise(hz2, hz_in.c)
            else:
                tf._hz_cache.pop(L.id, None)
                return fact
        else:
            hz_in = None
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_div(L, bounds, tf):
    bx = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0)
    by = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)
    fact = interval.tf_div(L, bx, by)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None and _hz_is_point(hz2):
            denom = _broadcast_flat(
                hz2.c.to(dtype=hz_in.c.dtype, device=hz_in.c.device),
                hz_in.c.shape[0],
            )
            if bool((denom.abs() > 1e-12).all()):
                tf._hz_cache[L.id] = _hz_scale_elementwise(hz_in, 1.0 / denom)
                return _hz_fact(fact, tf._hz_cache[L.id])
        tf._hz_cache.pop(L.id, None)
    return fact


def tf_constant(L, bounds, tf):
    val = L.params["value"].flatten()
    n = val.numel()
    if bounds is not None and n > 0:
        in_numel = int(bounds.lb.numel())
        if in_numel > 0 and in_numel % n == 0:
            B = in_numel // n
            if B > 1:
                val = val.repeat(B)
                n = val.numel()
    tf._hz_cache[L.id] = HZono(
        c=val.view(-1, 1),
        Gc=val.new_zeros(n, 0),
        Gb=val.new_zeros(n, 0),
        Ac=val.new_zeros(0, 0),
        Ab=val.new_zeros(0, 0),
        b=val.new_zeros(0, 1),
    )
    return interval.tf_constant(L, bounds)


def tf_sign(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_sign(L, bounds)


def tf_compare(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_compare(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )


def tf_where(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_where(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 2),
    )


def tf_matmul(L, bounds, tf):
    bx = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0)
    by = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)
    fact = interval.tf_matmul(L, bx, by)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            out = None
            if _hz_is_point(hz2):
                out = _hz_matmul_const(L, hz_in, hz2.c, variable_is_left=True)
            elif _hz_is_point(hz_in):
                out = _hz_matmul_const(L, hz2, hz_in.c, variable_is_left=False)
            if out is not None:
                tf._hz_cache[L.id] = out
                return _hz_fact(fact, tf._hz_cache[L.id])
    tf._hz_cache.pop(L.id, None)
    return fact


def tf_arg_extremum(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_arg_extremum(L, bounds)


def _row_indices_upsample_nearest(L, n_in: int, n_out: int):
    mode = str(L.params.get("mode", "nearest")).lower()
    if mode != "nearest":
        return None
    in_shape = L.params.get("input_shape")
    if in_shape is None:
        return None
    in_shape = tuple(int(d) for d in in_shape)
    if _prod(in_shape) != int(n_in) or len(in_shape) < 3:
        return None
    view_shape = (1, *in_shape) if len(in_shape) == 3 else in_shape
    spatial_rank = len(view_shape) - 2
    size = L.params.get("size")
    scale_factor = L.params.get("scale_factor")
    if size is not None and isinstance(size, (list, tuple)):
        size = tuple(int(s) for s in size)
        if len(size) > spatial_rank:
            size = size[-spatial_rank:]
    if scale_factor is not None and isinstance(scale_factor, (list, tuple)):
        scale_factor = tuple(float(s) for s in scale_factor)
        if len(scale_factor) > spatial_rank:
            scale_factor = scale_factor[-spatial_rank:]
    if size is None and scale_factor is None:
        out_shape = L.params.get("output_shape")
        if out_shape is None:
            return None
        out_shape = tuple(int(d) for d in out_shape)
        out_view_shape = (1, *out_shape) if len(out_shape) == 3 else out_shape
        if len(out_view_shape) != len(view_shape):
            return None
        size = out_view_shape[2:]
    base = torch.arange(int(n_in), dtype=torch.float64).view(*view_shape)
    idx = F.interpolate(base, size=size, scale_factor=scale_factor, mode="nearest")
    idx = idx.reshape(-1).long()
    return idx if idx.numel() == int(n_out) else None


def tf_upsample(L, bounds, tf):
    fact = interval_cnn.tf_upsample(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is None:
        return fact
    rows = _row_indices_upsample_nearest(L, hz_in.c.shape[0], fact.bounds.lb.numel())
    if rows is None:
        tf._hz_cache.pop(L.id, None)
        return fact
    tf._hz_cache[L.id] = _hz_gather_rows(hz_in, rows.to(device=hz_in.c.device))
    return _hz_fact(fact, tf._hz_cache[L.id])


def _row_indices_slice(L, n: int):
    if "input_shape" not in L.params:
        return None
    inp_shape = tuple(int(d) for d in L.params["input_shape"])
    per = _prod(inp_shape)
    if per == 0 or int(n) % per != 0:
        return None
    batch = int(n) // per
    idx = torch.arange(int(n)).view(batch, *inp_shape)
    starts = L.params.get("starts", [])
    ends = L.params.get("ends", [])
    axes = L.params.get("axes", list(range(len(inp_shape))))
    steps = L.params.get("steps", [1] * len(axes))
    slices = [slice(None)] * (len(inp_shape) + 1)
    for i, axis in enumerate(axes):
        axis = int(axis)
        end = ends[i]
        if end > inp_shape[axis]:
            end = inp_shape[axis]
        slices[axis + 1] = slice(starts[i], end, steps[i])
    return idx[tuple(slices)].reshape(-1)


def _row_indices_gather(L, n: int):
    if "input_shape" not in L.params:
        return None
    inp_shape = tuple(int(d) for d in L.params["input_shape"])
    per = _prod(inp_shape)
    if per == 0 or int(n) % per != 0:
        return None
    batch = int(n) // per
    axis = int(L.params.get("axis", 0))
    if axis < 0:
        axis += len(inp_shape)
    raw_idx = L.params["indices"]
    if isinstance(raw_idx, (list, tuple)):
        indices = torch.tensor(raw_idx, dtype=torch.long)
    elif hasattr(raw_idx, "detach"):
        indices = raw_idx.detach().cpu().long()
    else:
        indices = torch.as_tensor(raw_idx, dtype=torch.long)
    idx = torch.arange(int(n)).view(batch, *inp_shape)
    return torch.index_select(idx, dim=axis + 1, index=indices.reshape(-1)).reshape(-1)


def _row_indices_expand(L, n: int):
    in_shape = L.params.get("input_shape")
    out_shape = L.params.get("output_shape") or L.params.get("shape")
    if in_shape is None or out_shape is None:
        return None
    in_shape = tuple(int(d) for d in in_shape)
    out_shape = tuple(int(d) for d in out_shape)
    per = _prod(in_shape)
    if per == 0 or int(n) % per != 0:
        return None
    batch = int(n) // per
    try:
        return torch.arange(int(n)).view(batch, *in_shape).broadcast_to(
            batch, *out_shape
        ).reshape(-1)
    except RuntimeError:
        return None


def _row_indices_reduce_sum(L, n_in: int, n_out: int):
    in_shape = L.params.get("input_shape")
    if in_shape is None:
        return None
    in_shape = tuple(int(d) for d in in_shape)
    per = _prod(in_shape)
    if per == 0 or int(n_in) % per != 0:
        return None
    batch = int(n_in) // per
    axes = L.params.get("axes")
    axes = list(range(len(in_shape))) if not axes else [int(a) for a in axes]
    axes = [(a + len(in_shape)) if a < 0 else a for a in axes]
    keepdims = bool(L.params.get("keepdims", 0))
    out_shape = []
    for i, dim in enumerate(in_shape):
        if i in axes:
            if keepdims:
                out_shape.append(1)
        else:
            out_shape.append(dim)
    if _prod(out_shape) * batch != int(n_out):
        return None
    out_idx = torch.arange(int(n_out)).view(batch, *out_shape)
    view_shape = [batch]
    for i, dim in enumerate(in_shape):
        view_shape.append(1 if i in axes else dim)
    return out_idx.reshape(tuple(view_shape)).broadcast_to(batch, *in_shape).reshape(-1)


def tf_scatter_nd(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_scatter_nd(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 2),
    )


def tf_reduce_sum(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    fact = interval.tf_reduce_sum(L, bounds)
    if hz_in is not None:
        rows = _row_indices_reduce_sum(
            L, hz_in.c.shape[0], fact.bounds.lb.numel()
        )
        if rows is None:
            tf._hz_cache[L.id] = hz_from_bounds(
                fact.bounds, fact.bounds.lb.dtype, fact.bounds.lb.device
            )
        else:
            rows = rows.to(device=hz_in.c.device)
            out_n = int(fact.bounds.lb.numel())
            c = hz_in.c.new_zeros(out_n, 1)
            Gc = hz_in.Gc.new_zeros(out_n, hz_in.Gc.shape[1])
            Gb = hz_in.Gb.new_zeros(out_n, hz_in.Gb.shape[1])
            c.index_add_(0, rows, hz_in.c)
            if hz_in.Gc.shape[1]:
                Gc.index_add_(0, rows, hz_in.Gc)
            if hz_in.Gb.shape[1]:
                Gb.index_add_(0, rows, hz_in.Gb)
            tf._hz_cache[L.id] = HZono(
                c=c,
                Gc=Gc,
                Gb=Gb,
                Ac=hz_in.Ac,
                Ab=hz_in.Ab,
                b=hz_in.b,
                eq_mask=hz_in.eq_mask,
                col_ids=hz_in.col_ids,
                bcol_ids=hz_in.bcol_ids,
            )
    return fact


def tf_concat(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        parts = [tf._hz_cache.get(pid) for pid in preds]
        if all(p is not None for p in parts):
            tf._hz_cache[L.id] = hz_concat(parts)
        else:
            hz_in = None
    fact = interval.tf_concat(
        L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
    )
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_sub(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            tf._hz_cache[L.id] = hz_sub(hz_in, hz2)
        else:
            hz_in = None
    fact = interval.tf_sub(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_flatten(L, bounds, tf):
    fact = interval_cnn.tf_flatten(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = _hz_rebind(hz_in)
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_reshape(L, bounds, tf):
    fact = interval.tf_reshape(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = _hz_rebind(hz_in)
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def _hz_rebind(hz: HZono) -> HZono:
    return HZono(
        c=hz.c,
        Gc=hz.Gc,
        Gb=hz.Gb,
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        eq_mask=hz.eq_mask,
        col_ids=hz.col_ids,
        bcol_ids=hz.bcol_ids,
    )


def _hz_gather_rows(hz: HZono, row_idx: torch.Tensor) -> HZono:
    ri = row_idx.to(device=hz.c.device, dtype=torch.long)
    return HZono(
        c=hz.c[ri],
        Gc=hz.Gc[ri],
        Gb=hz.Gb[ri],
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        eq_mask=hz.eq_mask,
        col_ids=hz.col_ids,
        bcol_ids=hz.bcol_ids,
    )


def tf_squeeze(L, bounds, tf):
    fact = interval.tf_squeeze(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = _hz_rebind(hz_in)
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_unsqueeze(L, bounds, tf):
    fact = interval.tf_unsqueeze(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = _hz_rebind(hz_in)
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_transpose(L, bounds, tf):
    fact = interval.tf_transpose(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        width = bounds.lb.numel() // bounds.lb.shape[0]
        rows = _transpose_rows(L, hz_in.c.shape[0], width,
            fact.bounds.lb.device,
            (1, *tuple(int(d) for d in bounds.lb.shape[1:])),
        )
        tf._hz_cache[L.id] = _hz_gather_rows(hz_in, rows)
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_slice(L, bounds, tf):
    fact = interval.tf_slice(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        rows = _row_indices_slice(L, hz_in.c.shape[0])
        if rows is not None and rows.numel() == fact.bounds.lb.numel():
            tf._hz_cache[L.id] = _hz_gather_rows(hz_in, rows)
            return _hz_fact(fact, tf._hz_cache[L.id])
        tf._hz_cache.pop(L.id, None)
    return fact


def tf_gather(L, bounds, tf):
    fact = interval.tf_gather(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        rows = _row_indices_gather(L, hz_in.c.shape[0])
        if rows is not None and rows.numel() == fact.bounds.lb.numel():
            tf._hz_cache[L.id] = _hz_gather_rows(hz_in, rows)
            return _hz_fact(fact, tf._hz_cache[L.id])
        tf._hz_cache.pop(L.id, None)
    return fact


def tf_expand(L, bounds, tf):
    fact = interval.tf_expand(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        rows = _row_indices_expand(L, hz_in.c.shape[0])
        if rows is not None and rows.numel() == fact.bounds.lb.numel():
            tf._hz_cache[L.id] = _hz_gather_rows(hz_in, rows)
            return _hz_fact(fact, tf._hz_cache[L.id])
        tf._hz_cache.pop(L.id, None)
    return fact


# --- HZ activation encodings (zonotope domain) ---


def _relu_extend_ids(hz: HZono, k: int):
    if hz.col_ids is None:
        return None, None
    if hz.col_ids.numel() != hz.Gc.shape[1]:
        return None, None
    if hz.bcol_ids is None:
        if hz.Gb.shape[1] != 0:
            return None, None
        base_bids = torch.zeros(0, dtype=torch.long, device=hz.c.device)
    elif hz.bcol_ids.numel() == hz.Gb.shape[1]:
        base_bids = hz.bcol_ids.to(hz.c.device)
    else:
        return None, None
    return (
        torch.cat([hz.col_ids.to(hz.c.device), hz_fresh_col_ids(2 * k, hz.c.device)]),
        torch.cat([base_bids, hz_fresh_col_ids(k, hz.c.device)]),
    )


def _hz_apply_relu_family(hz: HZono, negative_slope: float) -> HZono:
    """Exact compressed LeakyReLU/ReLU graph encoding.

    For each unstable neuron with bounds [a, b], a < 0 < b, add xi1/xi2 and
    binary z. The linking equality plus xi1 + z >= 0 and xi2 - z >= 0 exactly
    selects y = s*x on x <= 0 or y = x on x >= 0; s=0 gives ReLU.
    """
    s = float(negative_slope)
    assert 0.0 <= s <= 1.0, f"negative_slope must be in [0, 1], got {s}"

    device = hz.c.device
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
    unstable_idx = torch.where(unstable)[0]
    k = int(unstable_idx.numel())

    out_c = hz.c.new_zeros(n, 1)
    out_Gc = hz.c.new_zeros(n, ng + 2 * k)
    out_Gb = hz.c.new_zeros(n, nb + k)

    if active.any():
        out_c[active] = hz.c[active]
        out_Gc[active, :ng] = hz.Gc[active]
        out_Gb[active, :nb] = hz.Gb[active]
    if inactive.any() and s != 0.0:
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
            eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
            col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
            bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
        )

    alpha = lb[unstable_idx]
    beta = ub[unstable_idx]
    t = torch.arange(k, device=device)
    col_xi1 = ng + t
    col_xi2 = ng + k + t
    col_z = nb + t

    out_c[unstable_idx, 0] = beta / 2.0
    if s != 0.0:
        out_Gc[unstable_idx, col_xi1] = s * alpha / 2.0
        out_Gb[unstable_idx, col_z] = s * alpha / 2.0
    out_Gc[unstable_idx, col_xi2] = -beta / 2.0

    ng_new = ng + 2 * k
    nb_new = nb + k
    rows = nc + 3 * k
    Ac_out = hz.c.new_zeros(rows, ng_new)
    Ab_out = hz.c.new_zeros(rows, nb_new)
    b_out = hz.c.new_zeros(rows, 1)
    if nc > 0:
        Ac_out[:nc, :ng] = hz.Ac
        Ab_out[:nc, :nb] = hz.Ab
        b_out[:nc] = hz.b

    eq_row = nc + t
    Ac_out[eq_row, col_xi1] = alpha / 2.0
    Ac_out[eq_row, col_xi2] = -beta / 2.0
    Ac_out[eq_row, :ng] = -hz.Gc[unstable_idx]
    if nb > 0:
        Ab_out[eq_row, :nb] = -hz.Gb[unstable_idx]
    Ab_out[eq_row, col_z] = alpha / 2.0
    b_out[eq_row, 0] = hz.c[unstable_idx, 0] - beta / 2.0

    ineq1 = nc + k + t
    ineq2 = nc + 2 * k + t
    Ac_out[ineq1, col_xi1] = -1.0
    Ab_out[ineq1, col_z] = -1.0
    Ac_out[ineq2, col_xi2] = -1.0
    Ab_out[ineq2, col_z] = 1.0

    old_mask = (
        hz.eq_mask.to(device)
        if hz.eq_mask is not None
        else torch.ones(nc, dtype=torch.bool, device=device)
    )
    eq_mask = torch.cat(
        [
            old_mask,
            torch.ones(k, dtype=torch.bool, device=device),
            torch.zeros(2 * k, dtype=torch.bool, device=device),
        ]
    )
    col_ids, bcol_ids = _relu_extend_ids(hz, k)

    return HZono(
        c=out_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=Ac_out,
        Ab=Ab_out,
        b=b_out,
        eq_mask=eq_mask,
        col_ids=col_ids,
        bcol_ids=bcol_ids,
    )


def hz_apply_relu(hz: HZono) -> HZono:
    return _hz_apply_relu_family(hz, 0.0)


def hz_apply_leaky_relu(hz: HZono, alpha_arg: float) -> HZono:
    return _hz_apply_relu_family(hz, alpha_arg)


def _project_s_curve_affine(
    hz, affine, new_c, lb_w, ub_w, wide_idx, owner,
    center_x, center_y, g1_x, g1_y, g2_x, g2_y,
):
    W, bias = affine
    W, bias = W.to(hz.c), bias.to(hz.c).flatten()
    device = hz.c.device
    ng, nb, nc = hz.Gc.shape[1], hz.Gb.shape[1], hz.Ac.shape[0]
    m, r = int(wide_idx.numel()), int(owner.numel())

    x_min = center_x - g1_x.abs() - g2_x.abs()
    x_max = center_x + g1_x.abs() + g2_x.abs()
    y_min_seg = center_y - g1_y.abs() - g2_y.abs()
    y_max_seg = center_y + g1_y.abs() + g2_y.abs()
    y_min = hz.c.new_full((m,), float("inf"))
    y_max = hz.c.new_full((m,), float("-inf"))
    y_min.scatter_reduce_(0, owner, y_min_seg, reduce="amin", include_self=True)
    y_max.scatter_reduce_(0, owner, y_max_seg, reduce="amax", include_self=True)
    y_mid, y_rad = (y_min + y_max) / 2.0, (y_max - y_min) / 2.0
    new_c[wide_idx] = y_mid.unsqueeze(1)

    det = g1_x * g2_y - g2_x * g1_y
    safe_det = torch.where(det.abs() < 1e-14, torch.ones_like(det), det)
    nx = torch.stack([g2_y, -g2_y, -g1_y, g1_y], dim=1) / safe_det[:, None]
    ny = torch.stack([-g2_x, g2_x, g1_x, -g1_x], dim=1) / safe_det[:, None]
    rhs = 1.0 + nx * center_x[:, None] + ny * center_y[:, None]
    degenerate = det.abs() < 1e-14
    if degenerate.any():
        nx[degenerate] = hz.c.new_tensor([1.0, -1.0, 0.0, 0.0])
        ny[degenerate] = hz.c.new_tensor([0.0, 0.0, 1.0, -1.0])
        rhs[degenerate] = torch.stack(
            [x_max, -x_min, y_max_seg, -y_min_seg], dim=1
        )[degenerate]
    rhs += 64.0 * torch.finfo(hz.c.dtype).eps * (1.0 + rhs.abs())

    facet_owner = owner.repeat_interleave(4)
    nx, ny, rhs = nx.flatten(), ny.flatten(), rhs.flatten()
    x_lo, x_hi = lb_w[facet_owner], ub_w[facet_owner]
    y_lo, y_hi = y_min[facet_owner], y_max[facet_owner]
    max_lhs = (
        torch.where(nx >= 0, nx * x_hi, nx * x_lo)
        + torch.where(ny >= 0, ny * y_hi, ny * y_lo)
    )
    big_m = (max_lhs - rhs).clamp_min(0.0)
    big_m += 64.0 * torch.finfo(hz.c.dtype).eps * (1.0 + max_lhs.abs())

    ng_total, nb_total = ng + m, nb + r
    rows = m + 4 * r
    Ac = hz.c.new_zeros(rows, ng_total)
    Ab = hz.c.new_zeros(rows, nb_total)
    b = hz.c.new_zeros(rows, 1)
    segment_cols = nb + torch.arange(r, device=device)
    Ab[owner, segment_cols] = 1.0
    b[:m, 0] = torch.bincount(owner, minlength=m).to(hz.c) - 2.0

    ineq = m + torch.arange(4 * r, device=device)
    input_rows = wide_idx[facet_owner]
    Ac[ineq, :ng] = nx[:, None] * hz.Gc[input_rows]
    Ac[ineq, ng + facet_owner] = ny * y_rad[facet_owner]
    if nb:
        Ab[ineq, :nb] = nx[:, None] * hz.Gb[input_rows]
    Ab[ineq, segment_cols.repeat_interleave(4)] = -big_m / 2.0
    b[ineq, 0] = (
        rhs + big_m / 2.0
        - nx * hz.c[input_rows, 0] - ny * y_mid[facet_owner]
    )

    old_Ac = torch.cat([hz.Ac, hz.c.new_zeros(nc, m)], dim=1)
    old_Ab = torch.cat([hz.Ab, hz.c.new_zeros(nc, r)], dim=1)
    old_mask = (
        hz.eq_mask.to(device)
        if hz.eq_mask is not None
        else torch.ones(nc, dtype=torch.bool, device=device)
    )
    col_ids = bcol_ids = None
    if hz.col_ids is not None and hz.col_ids.numel() == ng:
        base_binary_ids = (
            torch.zeros(0, dtype=torch.long, device=device)
            if hz.bcol_ids is None and nb == 0 else hz.bcol_ids
        )
        if base_binary_ids is not None and base_binary_ids.numel() == nb:
            col_ids = torch.cat([hz.col_ids.to(device), hz_fresh_col_ids(m, device)])
            bcol_ids = torch.cat(
                [base_binary_ids.to(device), hz_fresh_col_ids(r, device)]
            )

    out_Gc = hz.c.new_zeros(W.shape[0], ng_total)
    out_Gc[:, ng:] = W[:, wide_idx] * y_rad
    out = HZono(
        c=W @ new_c + bias.view(-1, 1),
        Gc=out_Gc,
        Gb=hz.c.new_zeros(W.shape[0], nb_total),
        Ac=torch.cat([old_Ac, Ac], dim=0),
        Ab=torch.cat([old_Ab, Ab], dim=0),
        b=torch.cat([hz.b, b], dim=0),
        eq_mask=torch.cat([
            old_mask,
            torch.ones(m, dtype=torch.bool, device=device),
            torch.zeros(4 * r, dtype=torch.bool, device=device),
        ]),
        col_ids=col_ids,
        bcol_ids=bcol_ids,
    )
    if hasattr(hz, "full_col_ids"):
        out.full_col_ids = hz.full_col_ids
    return out


def hz_apply_piecewise(
    hz: HZono,
    func,
    dfunc,
    K: int = 2,
    *,
    inflection: float = 0.0,
    affine=None,
) -> HZono:
    """Sound inflection-split S-curve enclosure for monotone activations.

    Each wide neuron's range [l, u] is split at the inflection point into up
    to K segments per side. Per segment, two continuous generators span the
    endpoint-tangent parallelogram and one binary selects the segment:
    sum(z) = count - 2; |xi| <= (1 - z) / 2 zeros deselected segments.
    This is a sound enclosure, not an exact graph encoding.
    """
    K = max(int(K), 1)
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
        out = HZono(
            c=new_c,
            Gc=new_Gc_base,
            Gb=new_Gb_base,
            Ac=hz.Ac.clone(),
            Ab=hz.Ab.clone(),
            b=hz.b.clone(),
            eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
            col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
            bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
        )
        if affine is not None:
            W, bias = affine
            out = hz_add_const(hz_multiply(out, W.to(out.c)), bias.to(out.c))
        return out

    lb_w, ub_w = lb[wide_idx], ub[wide_idx]
    p = torch.clamp(torch.full_like(lb_w, float(inflection)), min=lb_w, max=ub_w)
    segment_ids = torch.arange(K, dtype=dtype, device=device).unsqueeze(1)
    left_width = (p - lb_w).unsqueeze(0) / K
    right_width = (ub_w - p).unsqueeze(0) / K
    a_left = lb_w.unsqueeze(0) + segment_ids * left_width
    a_right = p.unsqueeze(0) + segment_ids * right_width
    a_grid = torch.cat([a_left, a_right], dim=0)
    b_grid = torch.cat([a_left + left_width, a_right + right_width], dim=0)
    owner_grid = torch.arange(m, device=device).unsqueeze(0).expand(2 * K, -1)
    nondeg = (b_grid - a_grid) > 1e-12
    a = a_grid[nondeg]
    b_seg = b_grid[nondeg]
    owner = owner_grid[nondeg].to(dtype=torch.long)
    r = int(a.numel())

    fa, fb = func(a), func(b_seg)
    la, lb_slope = dfunc(a), dfunc(b_seg)
    centers_x = (a + b_seg) / 2.0
    centers_y = (fa + fb) / 2.0
    nearly_linear = (la - lb_slope).abs() < 1e-10

    denom = lb_slope - la
    safe_denom = torch.where(nearly_linear, torch.ones_like(denom), denom)
    p1 = (fb - fa + lb_slope * a - la * b_seg) / safe_denom
    p2 = a + b_seg - p1
    g1x_tang = (p1 - a) / 2.0
    g1y_tang = lb_slope * (p1 - a) / 2.0
    g2x_tang = (p2 - a) / 2.0
    g2y_tang = la * (p2 - a) / 2.0

    hw = (b_seg - a) / 2.0
    slope = (fb - fa) / (b_seg - a + 1e-30)
    t_pts = torch.linspace(0.0, 1.0, 50, dtype=dtype, device=device).view(50, 1)
    pts = a.unsqueeze(0) + t_pts * (b_seg - a).unsqueeze(0)
    f_pts = func(pts)
    resid = f_pts - (slope.unsqueeze(0) * pts + (fa - slope * a).unsqueeze(0))
    max_err = resid.abs().max(dim=0).values
    g1x_lin, g1y_lin = hw, slope * hw
    g2x_lin, g2y_lin = torch.zeros_like(hw), max_err

    g1_x = torch.where(nearly_linear, g1x_lin, g1x_tang)
    g1_y = torch.where(nearly_linear, g1y_lin, g1y_tang)
    g2_x = torch.where(nearly_linear, g2x_lin, g2x_tang)
    g2_y = torch.where(nearly_linear, g2y_lin, g2y_tang)

    dx = pts - centers_x.unsqueeze(0)
    dy = f_pts - centers_y.unsqueeze(0)
    det = g1_y * g2_x - g1_x * g2_y
    safe_det = torch.where(det.abs() < 1e-30, torch.ones_like(det), det)
    xi1 = (dy * g2_x.unsqueeze(0) - dx * g2_y.unsqueeze(0)) / safe_det.unsqueeze(0)
    xi2 = (dy * g1_x.unsqueeze(0) - dx * g1_y.unsqueeze(0)) / (-safe_det.unsqueeze(0))
    max_xi = torch.maximum(xi1.abs().amax(dim=0), xi2.abs().amax(dim=0))
    scale_factor = torch.where(max_xi > 1.0, max_xi * 1.01, torch.ones_like(max_xi))
    scale_factor = torch.where(det.abs() < 1e-30, torch.ones_like(scale_factor), scale_factor)
    g1_x = g1_x * scale_factor
    g1_y = g1_y * scale_factor
    g2_x = g2_x * scale_factor
    g2_y = g2_y * scale_factor

    if affine is not None:
        return _project_s_curve_affine(
            hz, affine, new_c, lb_w, ub_w, wide_idx, owner,
            centers_x, centers_y, g1_x, g1_y, g2_x, g2_y,
        )

    center_y_sum = torch.bincount(owner, weights=centers_y, minlength=m).to(dtype)
    center_x_sum = torch.bincount(owner, weights=centers_x, minlength=m).to(dtype)
    seg_count = torch.bincount(owner, minlength=m).to(dtype)
    new_c[wide_idx] = (center_y_sum / 2.0).unsqueeze(1)
    new_Gc_base[wide_idx] = 0.0
    new_Gb_base[wide_idx] = 0.0

    n_real = 2 * r
    ng_total = ng + n_real
    nb_total = nb + r
    g1_cols = ng + torch.arange(r, device=device)
    g2_cols = ng + r + torch.arange(r, device=device)
    z_cols = nb + torch.arange(r, device=device)
    wide_rows = wide_idx[owner]

    Gc_new = hz.c.new_zeros(n, n_real)
    Gc_new[wide_rows, g1_cols - ng] = g1_y
    Gc_new[wide_rows, g2_cols - ng] = g2_y
    Gb_new = hz.c.new_zeros(n, r)
    Gb_new[wide_rows, z_cols - nb] = -centers_y / 2.0
    out_Gc = torch.cat([new_Gc_base, Gc_new], dim=1)
    out_Gb = torch.cat([new_Gb_base, Gb_new], dim=1)

    n_eq_total = 2 * m
    n_le_total = 4 * r
    eq_Ac = hz.c.new_zeros(n_eq_total, ng_total)
    eq_Ab = hz.c.new_zeros(n_eq_total, nb_total)
    eq_b = hz.c.new_zeros(n_eq_total, 1)

    link_rows = torch.arange(m, device=device)
    sum_rows = m + link_rows
    eq_Ac[link_rows[owner], g1_cols] = -g1_x
    eq_Ac[link_rows[owner], g2_cols] = -g2_x
    eq_Ab[link_rows[owner], z_cols] = centers_x / 2.0
    eq_Ac[link_rows, :ng] = hz.Gc[wide_idx]
    eq_Ab[link_rows, :nb] = hz.Gb[wide_idx]
    eq_b[link_rows, 0] = center_x_sum / 2.0 - hz.c[wide_idx, 0]
    eq_Ab[sum_rows[owner], z_cols] = 1.0
    eq_b[sum_rows, 0] = seg_count - 2.0

    ineq_Ac = hz.c.new_zeros(n_le_total, ng_total)
    ineq_Ab = hz.c.new_zeros(n_le_total, nb_total)
    ineq_b = hz.c.new_full((n_le_total, 1), 0.5)
    box_rows = 4 * torch.arange(r, device=device)
    ineq_Ac[box_rows, g1_cols] = 1.0
    ineq_Ac[box_rows + 1, g1_cols] = -1.0
    ineq_Ac[box_rows + 2, g2_cols] = 1.0
    ineq_Ac[box_rows + 3, g2_cols] = -1.0
    ineq_Ab[box_rows, z_cols] = 0.5
    ineq_Ab[box_rows + 1, z_cols] = 0.5
    ineq_Ab[box_rows + 2, z_cols] = 0.5
    ineq_Ab[box_rows + 3, z_cols] = 0.5

    old_Ac_ext = torch.cat([hz.Ac, hz.c.new_zeros(nc, n_real)], dim=1)
    old_Ab_ext = torch.cat([hz.Ab, hz.c.new_zeros(nc, r)], dim=1)
    new_col_ids = None
    new_bcol_ids = None
    if hz.col_ids is not None and hz.col_ids.numel() == ng:
        if hz.bcol_ids is None:
            base_bcol_ids = (
                torch.zeros(0, dtype=torch.long, device=hz.c.device)
                if nb == 0
                else None
            )
        elif hz.bcol_ids.numel() == nb:
            base_bcol_ids = hz.bcol_ids.to(hz.c.device)
        else:
            base_bcol_ids = None
        if base_bcol_ids is not None:
            new_col_ids = torch.cat(
                [hz.col_ids.to(hz.c.device), hz_fresh_col_ids(n_real, hz.c.device)]
            )
            new_bcol_ids = torch.cat(
                [base_bcol_ids, hz_fresh_col_ids(r, hz.c.device)]
            )

    old_mask = (
        hz.eq_mask.to(device)
        if hz.eq_mask is not None
        else torch.ones(nc, dtype=torch.bool, device=device)
    )
    out = HZono(
        c=new_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=torch.cat([old_Ac_ext, eq_Ac, ineq_Ac], dim=0),
        Ab=torch.cat([old_Ab_ext, eq_Ab, ineq_Ab], dim=0),
        b=torch.cat([hz.b, eq_b, ineq_b], dim=0),
        eq_mask=torch.cat(
            [
                old_mask,
                torch.ones(n_eq_total, dtype=torch.bool, device=device),
                torch.zeros(n_le_total, dtype=torch.bool, device=device),
            ]
        ),
        col_ids=new_col_ids,
        bcol_ids=new_bcol_ids,
    )
    if hasattr(hz, "full_col_ids"):
        out.full_col_ids = hz.full_col_ids
    return out


def hz_apply_sigmoid(hz: HZono, K: int = 2, *, affine=None) -> HZono:
    return hz_apply_piecewise(
        hz,
        torch.sigmoid,
        lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)),
        K,
        inflection=0.0,
        affine=affine,
    )


def hz_apply_tanh(hz: HZono, K: int = 2) -> HZono:
    return hz_apply_piecewise(hz, torch.tanh, lambda x: 1 - torch.tanh(x) ** 2, K)
