#===- act/back_end/hybridz_tf/tf_transformer.py - HybridZ Transformer TF -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ transfer functions for transformer operators.
#
#===---------------------------------------------------------------------===#

import torch

try:
    import numpy as np
    import scipy.sparse as sp
except ImportError:
    np = None
    sp = None

import act.back_end.interval_tf.tf_transformer as interval
from act.back_end.core import Bounds, Fact
from act.back_end.hybridz_tf.tf_mlp import _broadcast_flat, _matmul_term_rows
from act.back_end.solver.solver_hz import (
    hz_add_const,
    hz_bounds_are_liftable,
    hz_concat,
    hz_lift_bounds,
    hz_multiply,
    hz_tighten_bounds,
    sparse_abs_row_sum,
    sparse_hz_add_const,
    sparse_hz_concat,
    sparse_hz_lift_bounds,
    sparse_hz_linear,
)
from act.back_end.utils import scale_interval


def _dense_lift(
    L,
    fact,
    tf,
    output_equalities=None,
    equality_rhs=None,
    output_inequalities=None,
    inequality_rhs=None,
):
    hz = tf._hz_cache.get(L.id)
    if hz is None or not hz_bounds_are_liftable(fact.bounds):
        tf._hz_cache.pop(L.id, None)
        return fact
    out = hz_lift_bounds(
        hz,
        fact.bounds,
        output_equalities=output_equalities,
        equality_rhs=equality_rhs,
        output_inequalities=output_inequalities,
        inequality_rhs=inequality_rhs,
    )
    if max(out.Gc.shape[1] + out.Gb.shape[1], out.c.shape[0]) > tf._HZ_MAX_INPUT_DIM:
        tf._hz_cache.pop(L.id, None)
    else:
        tf._hz_cache[L.id] = out
    return fact


def _sparse_simplex(rowsize: int, n_out: int):
    if sp is None or rowsize <= 0 or n_out % rowsize:
        return None, None
    groups = n_out // rowsize
    rows = np.repeat(np.arange(groups, dtype=np.int64), rowsize)
    cols = np.arange(n_out, dtype=np.int64)
    matrix = sp.csr_matrix(
        (np.ones(n_out, dtype=np.float64), (rows, cols)),
        shape=(groups, n_out),
    )
    return matrix, np.ones(groups, dtype=np.float64)


def _softmax_ratio_inequalities(bounds: Bounds, rowsize: int, differences=None):
    if sp is None or rowsize <= 1 or bounds.lb.numel() % rowsize:
        return None, None
    lb = bounds.lb.detach().cpu().double().numpy().reshape(-1, rowsize)
    ub = bounds.ub.detach().cpu().double().numpy().reshape(-1, rowsize)
    matrix_rows, matrix_cols, matrix_data = [], [], []
    row = 0
    for group in range(lb.shape[0]):
        if rowsize <= 8:
            pairs = ((i, j) for i in range(rowsize) for j in range(i + 1, rowsize))
        else:
            ref = int(np.argmax((lb[group] + ub[group]) * 0.5))
            pairs = ((i, ref) for i in range(rowsize) if i != ref)
        offset = group * rowsize
        for i, j in pairs:
            if differences is None:
                lower_delta = lb[group, i] - ub[group, j]
                upper_delta = ub[group, i] - lb[group, j]
            else:
                difference_lb, difference_ub = differences
                lower_delta = -difference_ub[group, i, j]
                upper_delta = -difference_lb[group, i, j]
            lower = (
                np.nextafter(np.exp(lower_delta), 0.0)
                if -745.0 < lower_delta <= np.log(1e12)
                else 0.0
            )
            upper = np.inf
            if upper_delta <= np.log(1e12):
                upper = np.nextafter(
                    np.exp(upper_delta) if upper_delta > -745.0 else 0.0,
                    np.inf,
                )
            if np.isfinite(upper):
                matrix_rows.extend((row, row))
                matrix_cols.extend((offset + i, offset + j))
                matrix_data.extend((1.0, -upper))
                row += 1
            if lower > 0.0:
                matrix_rows.extend((row, row))
                matrix_cols.extend((offset + i, offset + j))
                matrix_data.extend((-1.0, lower))
                row += 1
    if row == 0:
        return None, None
    matrix = sp.csr_matrix(
        (matrix_data, (matrix_rows, matrix_cols)),
        shape=(row, bounds.lb.numel()),
        dtype=np.float64,
    )
    return matrix, np.zeros(row, dtype=np.float64)


def _attention_score_differences(L, bounds: Bounds, rowsize: int, tf):
    tf._softmax_score_contexts.pop(L.id, None)
    predecessors = tf._net.preds.get(L.id, [])
    if len(predecessors) != 1:
        return None
    n_scores = bounds.lb.numel()
    scale = np.ones(n_scores, dtype=np.float64)
    shift = np.zeros(n_scores, dtype=np.float64)
    current = predecessors[0]
    while True:
        score_layer = tf._net.by_id.get(current)
        if score_layer is None:
            return None
        kind = score_layer.kind.upper()
        if kind == "MATMUL":
            break
        layer_predecessors = tf._net.preds.get(current, [])
        if len(layer_predecessors) != 1:
            return None
        if kind in {"RESHAPE", "FLATTEN", "SQUEEZE", "UNSQUEEZE"}:
            current = layer_predecessors[0]
            continue
        if kind in {"SCALE", "BIAS"}:
            key = "a" if kind == "SCALE" else "c"
            raw_value = score_layer.params[key]
            value_count = int(raw_value.numel())
            if value_count == 0 or (
                value_count not in (1, n_scores) and n_scores % value_count != 0
            ):
                return None
            value = (
                _broadcast_flat(raw_value, n_scores)
                .detach().cpu().double().numpy()
            )
            if kind == "SCALE":
                scale *= value
            else:
                shift += scale * value
            current = layer_predecessors[0]
            continue
        return None

    score_predecessors = tf._net.preds.get(score_layer.id, [])
    if len(score_predecessors) != 2:
        return None
    q_hz = tf._sparse_hz_cache.get(score_predecessors[0])
    k_hz = tf._sparse_hz_cache.get(score_predecessors[1])
    if q_hz is None or k_hz is None or q_hz.frame_id != k_hz.frame_id:
        return None
    q_bounds = tf._net.get_predecessor_bounds(
        score_layer.id, tf._after, tf._before, 0
    )
    k_bounds = tf._net.get_predecessor_bounds(
        score_layer.id, tf._after, tf._before, 1
    )
    rows = _matmul_term_rows(score_layer, int(q_bounds.lb.shape[0]))
    if rows is None:
        return None
    q_rows, k_rows = rows
    if q_rows.shape[0] != n_scores or n_scores % rowsize:
        return None
    groups, reduction = n_scores // rowsize, q_rows.shape[1]
    q_lower = q_bounds.lb.detach().cpu().double().numpy().reshape(-1)
    q_upper = q_bounds.ub.detach().cpu().double().numpy().reshape(-1)
    lower = np.zeros((groups, rowsize, rowsize), dtype=np.float64)
    upper = np.zeros_like(lower)
    score_scale = scale.reshape(groups, rowsize)
    score_shift = shift.reshape(groups, rowsize)
    key_cache = {}
    pairs = rowsize * rowsize
    term_rows = np.repeat(np.arange(pairs, dtype=np.int64), reduction)
    term_cols = np.arange(pairs * reduction, dtype=np.int64)
    for group in range(groups):
        block = k_rows[group * rowsize : (group + 1) * rowsize]
        key = (block.tobytes(), score_scale[group].tobytes())
        delta = key_cache.get(key)
        if delta is None:
            left = np.broadcast_to(
                block[None, :, :], (rowsize, rowsize, reduction)
            ).reshape(-1)
            right = np.broadcast_to(
                block[:, None, :], (rowsize, rowsize, reduction)
            ).reshape(-1)
            left_scale = np.broadcast_to(
                score_scale[group][None, :, None],
                (rowsize, rowsize, reduction),
            ).reshape(-1)
            right_scale = np.broadcast_to(
                score_scale[group][:, None, None],
                (rowsize, rowsize, reduction),
            ).reshape(-1)
            center = (
                left_scale * k_hz.c[left] - right_scale * k_hz.c[right]
            ).reshape(rowsize, rowsize, reduction)
            continuous = (
                sp.diags(left_scale) @ k_hz.Gc[left]
                - sp.diags(right_scale) @ k_hz.Gc[right]
            ).tocsr()
            radius = sparse_abs_row_sum(continuous).reshape(
                rowsize, rowsize, reduction
            )
            if k_hz.n_bin:
                binary = (
                    sp.diags(left_scale) @ k_hz.Gb[left]
                    - sp.diags(right_scale) @ k_hz.Gb[right]
                ).tocsr()
                radius += sparse_abs_row_sum(binary).reshape(
                    rowsize, rowsize, reduction
                )
            else:
                binary = sp.csr_matrix((center.size, 0), dtype=np.float64)
            delta = (
                center,
                continuous,
                binary,
                center - radius,
                center + radius,
            )
            key_cache[key] = delta
        center, continuous, binary, delta_lower, delta_upper = delta
        q_term_rows = q_rows[group * rowsize]
        ql, qu = q_lower[q_term_rows], q_upper[q_term_rows]
        corners = np.stack(
            (ql * delta_lower, ql * delta_upper, qu * delta_lower, qu * delta_upper),
            axis=0,
        )
        interval_lower = np.min(corners, axis=0).sum(axis=-1)
        interval_upper = np.max(corners, axis=0).sum(axis=-1)

        q_mid = np.broadcast_to(
            (ql + qu) * 0.5, (pairs, reduction)
        ).reshape(-1)
        delta_mid = ((delta_lower + delta_upper) * 0.5).reshape(-1)
        q_operator = sp.csr_matrix(
            (delta_mid, (term_rows, np.tile(q_term_rows, pairs))),
            shape=(pairs, q_hz.n_out),
        )
        delta_operator = sp.csr_matrix(
            (q_mid, (term_rows, term_cols)),
            shape=(pairs, pairs * reduction),
        )
        affine_center = (
            np.asarray(q_operator @ q_hz.c).reshape(-1)
            + np.asarray(delta_operator @ center.reshape(-1)).reshape(-1)
            - np.add.reduceat(q_mid * delta_mid, term_cols[::reduction])
        ).reshape(rowsize, rowsize)
        affine_continuous = (
            q_operator @ q_hz.Gc + delta_operator @ continuous
        ).tocsr()
        affine_radius = sparse_abs_row_sum(affine_continuous).reshape(
            rowsize, rowsize
        )
        if q_hz.n_bin:
            affine_binary = (
                q_operator @ q_hz.Gb + delta_operator @ binary
            ).tocsr()
            affine_radius += sparse_abs_row_sum(affine_binary).reshape(
                rowsize, rowsize
            )
        affine_radius += (
            (qu - ql)[None, None, :] * (delta_upper - delta_lower) * 0.25
        ).sum(axis=-1)
        lower[group] = np.maximum(
            interval_lower, affine_center - affine_radius
        )
        upper[group] = np.minimum(
            interval_upper, affine_center + affine_radius
        )
        shift_delta = score_shift[group][None, :] - score_shift[group][:, None]
        lower[group] = np.nextafter(lower[group] + shift_delta, -np.inf)
        upper[group] = np.nextafter(upper[group] + shift_delta, np.inf)

    tf._softmax_score_contexts[L.id] = {
        "q_hz": q_hz,
        "k_hz": k_hz,
        "q_bounds": q_bounds,
        "k_bounds": k_bounds,
        "q_rows": q_rows,
        "k_rows": k_rows,
        "scale": scale,
        "shift": shift,
    }
    return lower, upper


def _softmax_box_from_differences(bounds: Bounds, differences) -> Bounds:
    lower, upper = differences
    with np.errstate(over="ignore"):
        lower_denominator = np.sum(np.exp(upper), axis=-1)
        upper_denominator = np.sum(np.exp(lower), axis=-1)
    probability_lower = np.divide(
        1.0,
        lower_denominator,
        out=np.zeros_like(lower_denominator),
        where=lower_denominator > 0,
    )
    probability_upper = np.divide(
        1.0,
        upper_denominator,
        out=np.ones_like(upper_denominator),
        where=upper_denominator > 0,
    )
    probability_lower = np.nextafter(probability_lower, 0.0)
    probability_upper = np.nextafter(probability_upper, np.inf)
    return Bounds(
        torch.from_numpy(probability_lower.reshape(-1)).to(bounds.lb).reshape_as(bounds.lb),
        torch.from_numpy(probability_upper.reshape(-1)).to(bounds.ub).reshape_as(bounds.ub),
    )


def _sparse_lift(
    L,
    hz,
    fact,
    tf,
    *,
    simplex_rowsize=None,
    output_equalities=None,
    equality_rhs=None,
    output_inequalities=None,
    inequality_rhs=None,
):
    if not hz_bounds_are_liftable(fact.bounds):
        return None, "nonfinite_sparse_transformer_bounds"
    reservation = tf._sparse_cont_slots_for(hz, L.id, fact.bounds.lb.numel())
    if reservation is None:
        return None, "sparse_transformer_size_limit"
    slots, n_cont = reservation
    equalities, rhs = output_equalities, equality_rhs
    if simplex_rowsize is not None:
        equalities, rhs = _sparse_simplex(
            int(simplex_rowsize), fact.bounds.lb.numel()
        )
    elif isinstance(equalities, torch.Tensor):
        equalities = sp.csr_matrix(equalities.detach().cpu().double().numpy())
        rhs = rhs.detach().cpu().double().numpy()
    inequalities, upper = output_inequalities, inequality_rhs
    if isinstance(inequalities, torch.Tensor):
        inequalities = sp.csr_matrix(
            inequalities.detach().cpu().double().numpy()
        )
        upper = upper.detach().cpu().double().numpy()
    return (
        sparse_hz_lift_bounds(
            hz,
            fact.bounds,
            slots,
            n_cont,
            output_equalities=equalities,
            equality_rhs=rhs,
            output_inequalities=inequalities,
            inequality_rhs=upper,
        ),
        None,
    )


def _softmax_box(bounds: Bounds, rowsize) -> Bounds:
    if rowsize is None or not hz_bounds_are_liftable(bounds):
        return Bounds(torch.zeros_like(bounds.lb), torch.ones_like(bounds.ub))
    lb = bounds.lb.reshape(-1, rowsize)
    ub = bounds.ub.reshape(-1, rowsize)
    eye = torch.eye(rowsize, dtype=torch.bool, device=lb.device).unsqueeze(0)
    other_ub = ub.unsqueeze(1).expand(-1, rowsize, -1).masked_fill(
        eye, float("-inf")
    )
    other_lb = lb.unsqueeze(1).expand(-1, rowsize, -1).masked_fill(
        eye, float("-inf")
    )
    lower_den = torch.logaddexp(lb, torch.logsumexp(other_ub, dim=-1))
    upper_den = torch.logaddexp(ub, torch.logsumexp(other_lb, dim=-1))
    return Bounds(
        torch.exp(lb - lower_den).reshape_as(bounds.lb),
        torch.exp(ub - upper_den).reshape_as(bounds.ub),
    )


def _layernorm_box(L, bounds: Bounds) -> Bounds:
    gamma = L.params["gamma"].flatten().to(bounds.lb)
    beta = L.params["beta"].flatten().to(bounds.lb)
    width = int(gamma.numel())
    if width == 0 or bounds.lb.shape[-1] % width:
        raise ValueError("LayerNorm parameters do not divide the flattened input")
    lb = bounds.lb.reshape(-1, width)
    ub = bounds.ub.reshape(-1, width)
    centered_lb = lb - ub.mean(dim=-1, keepdim=True)
    centered_ub = ub - lb.mean(dim=-1, keepdim=True)
    if L.params.get("variant", L.params.get("layer_norm", "standard")) == "no_var":
        norm_lb, norm_ub = centered_lb, centered_ub
    else:
        max_abs = torch.maximum(centered_lb.abs(), centered_ub.abs())
        variance_ub = (max_abs * max_abs).mean(dim=-1, keepdim=True)
        eps = bounds.lb.new_tensor(float(L.params.get("eps", 1e-5)))
        norm_lb, norm_ub = scale_interval(
            centered_lb,
            centered_ub,
            torch.rsqrt(variance_ub + eps),
            torch.rsqrt(eps).expand_as(variance_ub),
        )
    out_lb = torch.where(gamma >= 0, gamma * norm_lb + beta, gamma * norm_ub + beta)
    out_ub = torch.where(gamma >= 0, gamma * norm_ub + beta, gamma * norm_lb + beta)
    return Bounds(out_lb.reshape_as(bounds.lb), out_ub.reshape_as(bounds.ub))


def _layernorm_equalities(L, fact):
    gamma = L.params.get("gamma")
    beta = L.params.get("beta")
    if not isinstance(gamma, torch.Tensor) or not isinstance(beta, torch.Tensor):
        return None, None
    width = int(gamma.numel())
    total = int(fact.bounds.lb.numel())
    if width == 0 or total % width:
        return None, None
    gamma = gamma.flatten().to(fact.bounds.lb)
    beta = beta.flatten().to(fact.bounds.lb)
    if bool((gamma.abs() <= torch.finfo(gamma.dtype).eps).any()):
        return None, None
    weights = gamma.reciprocal()
    groups = total // width
    matrix = fact.bounds.lb.new_zeros((groups, total))
    cols = torch.arange(total, device=matrix.device).view(groups, width)
    matrix.scatter_(1, cols, weights.expand(groups, -1))
    return matrix, (weights * beta).sum().expand(groups)


def _mha_split_operator(L, n_in: int, n_out: int):
    weight = L.params.get("weight")
    if sp is None or not isinstance(weight, torch.Tensor):
        return None
    hidden = int(L.params.get("hidden_size", weight.shape[1]))
    input_shape = tuple(
        int(d) for d in L.params.get("input_shape", (1, hidden))
    )
    sequence = int(input_shape[-2]) if len(input_shape) >= 2 else 1
    per_sample = sequence * hidden
    if per_sample <= 0 or n_in % per_sample:
        return None
    matrix = weight.detach().cpu().double().numpy()
    bias = L.params.get("bias")
    bias_vector = (
        bias.detach().cpu().double().numpy().reshape(-1)
        if isinstance(bias, torch.Tensor)
        else np.zeros(matrix.shape[0], dtype=np.float64)
    )
    role = str(L.params.get("role", ""))
    if role in {"query", "key"}:
        position = int(L.params.get("position", 0))
        if position < 0 or position >= sequence:
            return None
        single = sp.hstack(
            [
                sp.csr_matrix((matrix.shape[0], position * hidden)),
                sp.csr_matrix(matrix),
                sp.csr_matrix(
                    (matrix.shape[0], (sequence - position - 1) * hidden)
                ),
            ],
            format="csr",
        )
        single_bias = bias_vector
    elif role == "value":
        feature = int(L.params.get("feature", 0))
        if feature < 0 or feature >= matrix.shape[0]:
            return None
        single = sp.kron(
            sp.eye(sequence, format="csr"),
            sp.csr_matrix(matrix[feature : feature + 1]),
            format="csr",
        )
        single_bias = np.full(sequence, bias_vector[feature], dtype=np.float64)
    else:
        single = sp.kron(
            sp.eye(sequence, format="csr"), sp.csr_matrix(matrix), format="csr"
        )
        single_bias = np.tile(bias_vector, sequence)
    batch = n_in // per_sample
    operator = sp.block_diag([single] * batch, format="csr")
    full_bias = np.tile(single_bias, batch)
    return (operator, full_bias) if operator.shape == (n_out, n_in) else None


def tf_posenc(L, bounds, tf):
    fact = interval.tf_posenc(L, bounds)
    hz = tf._hz_cache.get(L.id)
    if hz is not None:
        tf._hz_cache[L.id] = hz_add_const(
            hz, _broadcast_flat(L.params["pos_vec"], hz.c.shape[0])
        )
    return fact


def tf_layernorm(L, bounds, tf):
    fact = interval.tf_layernorm(L, bounds)
    fact = Fact(bounds=_layernorm_box(L, bounds), cons=fact.cons)
    fact.cons.add_box(L.id, L.out_vars, fact.bounds)
    equalities, rhs = _layernorm_equalities(L, fact)
    return _dense_lift(L, fact, tf, equalities, rhs)


def tf_gelu(L, bounds, tf):
    return _dense_lift(L, interval.tf_gelu(L, bounds), tf)


def tf_att_scores(L, bounds, tf):
    fact = interval.tf_att_scores(
        L,
        tf._after[L.params["q_src"]].bounds,
        tf._after[L.params["k_src"]].bounds,
    )
    return _dense_lift(L, fact, tf)


def tf_softmax(L, bounds, tf):
    fact = interval.tf_softmax(L, bounds)
    rowsize = interval.softmax_rowsize(L, bounds)
    softmax_bounds = _softmax_box(bounds, rowsize)
    differences = (
        _attention_score_differences(L, bounds, rowsize, tf)
        if rowsize is not None else None
    )
    if differences is not None:
        softmax_bounds = hz_tighten_bounds(
            softmax_bounds,
            _softmax_box_from_differences(bounds, differences),
        )
    tf._softmax_differences[L.id] = differences
    fact = Fact(bounds=softmax_bounds, cons=fact.cons)
    fact.cons.add_box(L.id, L.out_vars, fact.bounds)
    hz = tf._hz_cache.get(L.id)
    equalities = rhs = inequalities = upper = None
    if hz is not None and rowsize is not None:
        sparse_equalities, sparse_rhs = _sparse_simplex(
            rowsize, fact.bounds.lb.numel()
        )
        if sparse_equalities is not None:
            equalities = torch.from_numpy(sparse_equalities.toarray()).to(hz.c)
            rhs = torch.from_numpy(sparse_rhs).to(hz.c)
        sparse_inequalities, sparse_upper = _softmax_ratio_inequalities(
            bounds, rowsize, differences
        )
        if sparse_inequalities is not None:
            inequalities = torch.from_numpy(sparse_inequalities.toarray()).to(hz.c)
            upper = torch.from_numpy(sparse_upper).to(hz.c)
    return _dense_lift(
        L, fact, tf, equalities, rhs, inequalities, upper
    )


def tf_att_mix(L, bounds, tf):
    fact = interval.tf_att_mix(
        L,
        tf._after[L.params["w_src"]].bounds,
        tf._after[L.params["v_src"]].bounds,
    )
    return _dense_lift(L, fact, tf)


def tf_mha_split(L, bounds, tf):
    fact = interval.tf_mha_split(L, bounds)
    hz = tf._hz_cache.get(L.id)
    if hz is not None:
        affine = _mha_split_operator(L, hz.c.shape[0], fact.bounds.lb.numel())
        if affine is not None:
            operator, bias = affine
            dense_operator = torch.from_numpy(operator.toarray()).to(hz.c)
            tf._hz_cache[L.id] = hz_add_const(
                hz_multiply(hz, dense_operator), torch.from_numpy(bias).to(hz.c)
            )
            return fact
    return _dense_lift(L, fact, tf)


def tf_mha_join(L, bounds, tf):
    inputs = tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
    fact = interval.tf_mha_join(L, inputs)
    parts = [tf._hz_cache.get(pid) for pid in tf._net.preds.get(L.id, [])]
    if parts and all(part is not None for part in parts):
        tf._hz_cache[L.id] = hz_concat(parts, batch_size=inputs[0].lb.shape[0])
    else:
        tf._hz_cache.pop(L.id, None)
    return fact


def tf_mask_add(L, bounds, tf):
    fact = interval.tf_mask_add(L, bounds)
    hz = tf._hz_cache.get(L.id)
    if hz is not None:
        tf._hz_cache[L.id] = hz_add_const(
            hz, _broadcast_flat(L.params["M"], hz.c.shape[0])
        )
    return fact


def sparse_hz_apply_layer(L, hz, input_bounds, result, tf):
    kind = L.kind.upper()
    if kind in {"POSENC", "MASK_ADD"}:
        key = "pos_vec" if kind == "POSENC" else "M"
        return True, sparse_hz_add_const(
            hz, _broadcast_flat(L.params[key], hz.n_out)
        ), None
    if kind == "MHA_JOIN":
        parts = [
            tf._sparse_hz_cache.get(pid)
            for pid in tf._net.preds.get(L.id, [])
        ]
        inputs = tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
        return (
            (True, sparse_hz_concat(parts, batch_size=inputs[0].lb.shape[0]), None)
            if parts and all(part is not None for part in parts)
            else (True, None, "missing_sparse_mha_join_input")
        )
    if kind == "SOFTMAX":
        rowsize = interval.softmax_rowsize(L, input_bounds)
        differences = tf._softmax_differences.get(L.id)
        inequalities, upper = (
            _softmax_ratio_inequalities(input_bounds, rowsize, differences)
            if rowsize is not None
            else (None, None)
        )
        out, reason = _sparse_lift(
            L,
            hz,
            result,
            tf,
            simplex_rowsize=rowsize,
            output_inequalities=inequalities,
            inequality_rhs=upper,
        )
        return True, out, reason
    if kind == "MHA_SPLIT":
        affine = _mha_split_operator(L, hz.n_out, result.bounds.lb.numel())
        if affine is not None:
            operator, bias = affine
            return True, sparse_hz_linear(hz, operator, bias), None
        out, reason = _sparse_lift(L, hz, result, tf)
        return True, out, reason
    if kind == "LAYERNORM":
        equalities, rhs = _layernorm_equalities(L, result)
        out, reason = _sparse_lift(
            L, hz, result, tf, output_equalities=equalities, equality_rhs=rhs
        )
        return True, out, reason
    if kind in {"GELU", "ATT_SCORES", "ATT_MIX"}:
        out, reason = _sparse_lift(L, hz, result, tf)
        return True, out, reason
    return False, None, None
