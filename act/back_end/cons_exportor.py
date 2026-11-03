#===- act/back_end/cons_exportor.py - Constraint Set Export Utilities ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Constraint set export utilities for external solver integration.
#   Provides export functionality for constraint sets to various formats.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import logging
from importlib import import_module
import math
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import cached_property

import numpy as np
import torch
from typing import TYPE_CHECKING, Any
from collections.abc import Callable
from act.back_end.core import Bounds, ConSet, Fact, Layer, Net, get_topo_order
from act.back_end.interval_tf.tf_mlp import _quantize_params_flat
from act.back_end.solver.solver_base import BatchLPProblem
from act.back_end.layer_util import validate_conset_ops
from act.front_end.specs import InKind, OutKind, OutputSpec

if TYPE_CHECKING:
    from act.back_end.solver.solver_torchlp import _SparseLP

# Module logger
logger = logging.getLogger(__name__)

# =============================================================================
# [BATCHED-API] export_to_batch_problem and helpers
# =============================================================================
#
# Build a BatchLPProblem from analyze() output + per-instance input box.
# Canonical per-tag forms per Oracle §I keep m_eq/m_le uniform across N so the
# block-diagonal sparse representation has fixed block shape. ASSERT negation
# is encoded inline here from batched ASSERT tensors.
#
# =============================================================================

# Numerical tolerances used by batched ASSERT encoding.
_ASSERT_EPS = 1e-6
_RANGE_SLACK_CAP = 1e6
_TANH_LIN_EPS = 1e-9
_TANH_BAND = 0.25
_TANH_BAND_TOL = 1e-6


def _as_td(x: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Exact extraction of the repeated coercion idiom: tensors move via .to
    # (no-op when already matching), non-tensors lift via torch.as_tensor.
    # Preserves device/dtype/contiguity semantics; adds no .clone().
    return (
        x.to(device=device, dtype=dtype)
        if isinstance(x, torch.Tensor)
        else torch.as_tensor(x, device=device, dtype=dtype)
    )


def _coerce_b_tensor(
    val: Any,
    expected_b: int,
    expected_n: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Bring meta tensor to ``[B, n]`` shape.

    Accepts a 1-D ``[n]`` (broadcasts over batch), a 2-D ``[B, n]``, or a 0-D
    scalar (broadcasts both axes). Raises ValueError on any other shape.
    """
    t = val if isinstance(val, torch.Tensor) else torch.as_tensor(val)
    t = t.to(device=device, dtype=dtype)
    if t.dim() == 0:
        return t.expand(expected_b, expected_n).contiguous()
    if t.dim() == 1:
        if t.shape[0] != expected_n:
            raise ValueError(
                f"{name}: 1-D tensor length {t.shape[0]} != expected_n "
                f"{expected_n}"
            )
        return t.unsqueeze(0).expand(expected_b, -1).contiguous()
    if t.dim() == 2:
        if t.shape[0] == 1 and expected_b > 1:
            t = t.expand(expected_b, -1)
        if t.shape != (expected_b, expected_n):
            raise ValueError(
                f"{name}: 2-D tensor shape {tuple(t.shape)} != "
                f"({expected_b}, {expected_n})"
            )
        return t.contiguous()
    raise ValueError(
        f"{name}: unsupported tensor dim {t.dim()} shape {tuple(t.shape)}"
    )


class _RowAcc:
    """Row accumulator for one constraint kind (EQ or LE).

    Each entry records a single per-batch row with structure shared across
    instances. The variable IDs are fixed; only the coefficients and the RHS
    can vary across the batch dimension N.

    Fields per entry:
        var_ids: List[int]            -- column ids referenced by this row
        vals:    torch.Tensor[N, k]   -- coefficients (k == len(var_ids))
        rhs:     torch.Tensor[N]      -- right-hand side per instance
    """

    __slots__ = ("n_batch", "device", "dtype", "_rows")

    def __init__(self, n_batch: int, device: torch.device, dtype: torch.dtype):
        self.n_batch = n_batch
        self.device = device
        self.dtype = dtype
        self._rows: list[tuple[list[int], torch.Tensor, torch.Tensor]] = []

    def add(
        self,
        var_ids: list[int],
        vals: torch.Tensor,
        rhs: torch.Tensor,
    ) -> None:
        if vals.shape != (self.n_batch, len(var_ids)):
            raise ValueError(
                f"_RowAcc.add: vals shape {tuple(vals.shape)} != "
                f"({self.n_batch}, {len(var_ids)})"
            )
        if rhs.shape != (self.n_batch,):
            raise ValueError(
                f"_RowAcc.add: rhs shape {tuple(rhs.shape)} != "
                f"({self.n_batch},)"
            )
        self._rows.append((list(var_ids), vals, rhs))

    def add_block(
        self,
        col_block: torch.Tensor,
        val_block: torch.Tensor,
        rhs_block: torch.Tensor,
    ) -> None:
        """Add m rows in one call.

        Args:
            col_block: ``[m, k]`` long tensor of column ids (uniform across N).
            val_block: ``[N, m, k]`` coefficient values per (instance, row).
            rhs_block: ``[N, m]`` right-hand side values per instance.
        """
        if col_block.dim() != 2:
            raise ValueError(
                f"_RowAcc.add_block: col_block dim {col_block.dim()} != 2"
            )
        m, k = col_block.shape
        if val_block.shape != (self.n_batch, m, k):
            raise ValueError(
                f"_RowAcc.add_block: val_block shape {tuple(val_block.shape)} "
                f"!= ({self.n_batch}, {m}, {k})"
            )
        if rhs_block.shape != (self.n_batch, m):
            raise ValueError(
                f"_RowAcc.add_block: rhs_block shape {tuple(rhs_block.shape)} "
                f"!= ({self.n_batch}, {m})"
            )
        cols_list = col_block.tolist()
        for r in range(m):
            self._rows.append(
                (cols_list[r], val_block[:, r, :], rhs_block[:, r])
            )

    def m(self) -> int:
        return len(self._rows)

    def build_sparse(
        self, nvars_total: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialise block-diagonal sparse A and stacked b.

        Returns:
            A: ``torch.sparse_coo_tensor`` of shape (n_batch*m, n_batch*nvars).
            b: ``[n_batch, m]`` dense tensor.
        """
        n_batch = self.n_batch
        m = self.m()
        if m == 0:
            empty_idx = torch.zeros(
                (2, 0), device=self.device, dtype=torch.long
            )
            empty_val = torch.zeros(
                (0,), device=self.device, dtype=self.dtype
            )
            A_empty = torch.sparse_coo_tensor(
                empty_idx, empty_val, (n_batch * 0, n_batch * nvars_total)
            )
            b_empty = torch.zeros(
                (n_batch, 0), device=self.device, dtype=self.dtype
            )
            return A_empty, b_empty

        # Build row/col index tensors and values for every entry.
        row_chunks: list[torch.Tensor] = []
        col_chunks: list[torch.Tensor] = []
        val_chunks: list[torch.Tensor] = []
        rhs_chunks: list[torch.Tensor] = []
        batch_ids = torch.arange(n_batch, device=self.device, dtype=torch.long)
        for row_idx, (var_ids, vals, rhs) in enumerate(self._rows):
            k = len(var_ids)
            # global_row[n, j] = n * m + row_idx     (j replicates over k vars)
            row_global = (batch_ids * m + row_idx).repeat_interleave(k)
            var_id_t = torch.tensor(
                var_ids, device=self.device, dtype=torch.long
            )
            # global_col[n, j] = n * nvars_total + var_ids[j]
            col_global = (
                batch_ids.unsqueeze(1) * nvars_total + var_id_t.unsqueeze(0)
            ).reshape(-1)
            row_chunks.append(row_global)
            col_chunks.append(col_global)
            val_chunks.append(vals.reshape(-1))
            rhs_chunks.append(rhs)

        rows_flat = torch.cat(row_chunks)
        cols_flat = torch.cat(col_chunks)
        vals_flat = torch.cat(val_chunks)
        indices = torch.stack([rows_flat, cols_flat], dim=0)
        sparse_a = torch.sparse_coo_tensor(
            indices, vals_flat, (n_batch * m, n_batch * nvars_total)
        )
        b = torch.stack(rhs_chunks, dim=1)
        return sparse_a.coalesce(), b


# -----------------------------------------------------------------------------
# Bounds aggregation
# -----------------------------------------------------------------------------


def _build_box_tensors_batched(
    templates: list[Any],
    nvars_total: int,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge every ``box:`` template into ``[N, nvars_total]`` lb/ub tensors.

    Box tensors may be ``[n]`` (broadcast across N) or ``[N, n]``. Multiple
    box templates per variable intersect (tighter wins).
    """
    lb_global = torch.full(
        (N, nvars_total), -float("inf"), device=device, dtype=dtype
    )
    ub_global = torch.full(
        (N, nvars_total), float("inf"), device=device, dtype=dtype
    )
    for con in templates:
        tag = con.meta.get("tag", "")
        if not tag.startswith("box:"):
            continue
        var_ids = list(con.var_ids)
        if not var_ids:
            continue
        n_vars = len(var_ids)
        lb_raw = con.meta["lb"]
        ub_raw = con.meta["ub"]
        def norm(raw: Any, name: str) -> torch.Tensor:
            """Normalise one box endpoint to ``[N, n_vars]``."""
            if isinstance(raw, torch.Tensor) and raw.numel() == n_vars:
                return raw.to(device=device, dtype=dtype).reshape(n_vars).unsqueeze(0).expand(N, -1)
            if isinstance(raw, torch.Tensor) and raw.numel() == N * n_vars:
                return raw.to(device=device, dtype=dtype).reshape(N, n_vars)
            raise ValueError(
                f"box: {name} has unexpected numel={getattr(raw, 'numel', lambda: '?')()}, "
                f"expected {n_vars} or {N * n_vars} for tag {tag!r}"
            )

        lb_b = norm(lb_raw, "lb")
        ub_b = norm(ub_raw, "ub")
        vid_idx = torch.tensor(var_ids, device=device, dtype=torch.long)
        lb_global[:, vid_idx] = torch.maximum(lb_global[:, vid_idx], lb_b)
        ub_global[:, vid_idx] = torch.minimum(ub_global[:, vid_idx], ub_b)
    return lb_global, ub_global


# -----------------------------------------------------------------------------
# Per-tag canonical emitters
# -----------------------------------------------------------------------------
#
# Each emitter appends rows into the EQ or LE accumulator with shape:
#   var_ids: list[int]
#   vals_per_b: [N, k]
#   rhs_per_b: [N]


def _emit_dense(
    con: Any,
    eq: _RowAcc,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """``y_i - sum_j W[i,j] * x_j = b[i]`` — one eq per output element."""
    W = con.meta["W"]
    b = con.meta["b"]
    W_t = _as_td(W, device, dtype)
    b_t = _as_td(b, device, dtype)
    n_out, n_in = W_t.shape
    var_ids_all = list(con.var_ids)
    token_count = int(con.meta["token_count"]) if con.meta.get("token_wise", False) else 1
    total_out, total_in = n_out * token_count, n_in * token_count
    y = var_ids_all[:total_out]
    x = var_ids_all[total_out:]
    if len(x) != total_in:
        raise ValueError(
            f"dense: var_ids length mismatch: {len(x)} input vars vs "
            f"expected {total_in} ({token_count} x W.shape[1]={n_in})"
        )
    m = total_out
    col_block = torch.empty((m, 1 + n_in), device=device, dtype=torch.long)
    col_block[:, 0] = torch.tensor(y, device=device, dtype=torch.long)
    x_rows = torch.tensor(x, device=device, dtype=torch.long).reshape(token_count, n_in)
    col_block[:, 1:] = x_rows[:, None, :].expand(token_count, n_out, n_in).reshape(m, n_in)
    val_per_row = torch.empty((m, 1 + n_in), device=device, dtype=dtype)
    val_per_row[:, 0] = 1.0
    val_per_row[:, 1:] = -W_t.repeat(token_count, 1)
    val_block = val_per_row.unsqueeze(0).expand(N, m, 1 + n_in).contiguous()
    rhs_block = b_t.reshape(n_out).repeat(token_count).unsqueeze(0).expand(N, m).contiguous()
    eq.add_block(col_block, val_block, rhs_block)


def _emit_affine_pairs(
    y: list[int],
    x: list[int],
    a: torch.Tensor,
    c: torch.Tensor,
    eq: _RowAcc,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Emit ``y_i - a_i*x_i = c_i`` equalities."""
    if len(y) != len(x) or a.numel() != len(y) or c.numel() != len(y):
        raise ValueError("affine pairs: incompatible variable and coefficient counts")
    n = len(y)
    col_block = torch.stack(
        [
            torch.tensor(y, device=device, dtype=torch.long),
            torch.tensor(x, device=device, dtype=torch.long),
        ],
        dim=1,
    )
    val_per_row = torch.empty((n, 2), device=device, dtype=dtype)
    val_per_row[:, 0] = 1.0
    val_per_row[:, 1] = -a.reshape(n)
    eq.add_block(
        col_block,
        val_per_row.unsqueeze(0).expand(N, n, 2).contiguous(),
        c.reshape(n).unsqueeze(0).expand(N, n).contiguous(),
    )


def _emit_bias(
    con: Any,
    eq: _RowAcc,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """``y_i - x_i = c_i`` per element (bias layer)."""
    n = len(con.var_ids) // 2
    y = list(con.var_ids[:n])
    x = list(con.var_ids[n:])
    c_raw = con.meta["c"]
    c_t = _as_td(c_raw, device, dtype).reshape(-1)
    if c_t.numel() != n:
        raise ValueError(f"bias: c length {c_t.numel()} != n {n}")
    _emit_affine_pairs(
        y, x, torch.ones(n, device=device, dtype=dtype), c_t, eq, N, device, dtype
    )


def _emit_scale(
    con: Any,
    eq: _RowAcc,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """``y_i - a_i * x_i = 0`` per element (scale layer)."""
    n = len(con.var_ids) // 2
    y = list(con.var_ids[:n])
    x = list(con.var_ids[n:])
    a_raw = con.meta["a"]
    a_t = _as_td(a_raw, device, dtype).reshape(-1)
    if a_t.numel() != n:
        raise ValueError(f"scale: a length {a_t.numel()} != n {n}")
    _emit_affine_pairs(
        y, x, a_t, torch.zeros(n, device=device, dtype=dtype), eq, N, device, dtype
    )


def _emit_bn(
    con: Any,
    eq: _RowAcc,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """``y_i - A_i * x_i = c_i`` per element (batch-norm-fused affine)."""
    n = len(con.var_ids) // 2
    y = list(con.var_ids[:n])
    x = list(con.var_ids[n:])
    A_raw = con.meta["A"]
    c_raw = con.meta["c"]
    A_t = _as_td(A_raw, device, dtype).reshape(-1)
    c_t = _as_td(c_raw, device, dtype).reshape(-1)
    if A_t.numel() != n or c_t.numel() != n:
        raise ValueError(
            f"bn: A.numel={A_t.numel()} c.numel={c_t.numel()} != n {n}"
        )
    _emit_affine_pairs(y, x, A_t, c_t, eq, N, device, dtype)


def _emit_add_sub(
    con: Any,
    eq: _RowAcc,
    sign_y: float,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """``z_i - x_i - sign_y * y_i = 0``."""
    n = len(con.var_ids) // 3
    z = list(con.var_ids[:n])
    x = list(con.var_ids[n:2 * n])
    y = list(con.var_ids[2 * n:])
    m = n
    col_block = torch.stack([
        torch.tensor(z, device=device, dtype=torch.long),
        torch.tensor(x, device=device, dtype=torch.long),
        torch.tensor(y, device=device, dtype=torch.long),
    ], dim=1)
    val_per_row = torch.tensor(
        [1.0, -1.0, -sign_y], device=device, dtype=dtype
    ).unsqueeze(0).expand(m, -1)
    val_block = val_per_row.unsqueeze(0).expand(N, m, 3).contiguous()
    rhs_block = torch.zeros((N, m), device=device, dtype=dtype)
    eq.add_block(col_block, val_block, rhs_block)


def _emit_flatten(
    con: Any,
    eq: _RowAcc,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Flatten is a C-order identity reshape: ``y_i - x_i = 0``."""
    input_shape_raw = con.meta.get("input_shape")
    if input_shape_raw is None:
        if len(con.var_ids) % 2 != 0:
            raise ValueError(
                f"flatten: var_ids length {len(con.var_ids)} is not even"
            )
        n_in = len(con.var_ids) // 2
    else:
        input_shape = tuple(int(v) for v in input_shape_raw)
        dims = input_shape[1:] if len(input_shape) > 1 else input_shape
        n_in = int(np.prod(dims))
    n_out = len(con.var_ids) - n_in
    if n_out != n_in:
        raise ValueError(f"flatten: n_out {n_out} != n_in {n_in}")
    y = list(con.var_ids[:n_out])
    x = list(con.var_ids[n_out:])
    _emit_copy_rows(y, x, eq, N, device, dtype)


def _emit_transpose(
    con: Any,
    eq: _RowAcc,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if len(con.var_ids) % 2 != 0:
        raise ValueError(
            f"transpose: var_ids length {len(con.var_ids)} is not even"
    )
    n = len(con.var_ids) // 2
    perm_raw = con.meta.get("perm")
    if perm_raw is None:
        idx = torch.arange(n, device=device)
    else:
        input_shape_raw = con.meta.get("input_shape")
        if input_shape_raw is None:
            raise ValueError("transpose: input_shape is required when perm is set")
        input_shape = tuple(int(v) for v in input_shape_raw)
        perm = tuple(int(v) for v in perm_raw)
        if not input_shape or input_shape[0] != 1:
            raise ValueError(
                f"transpose: input_shape must have a singleton batch dimension, "
                f"got {input_shape}"
            )
        if int(np.prod(input_shape)) != n:
            raise ValueError(
                f"transpose: input_shape {input_shape} has "
                f"{int(np.prod(input_shape))} elements, expected {n}"
            )
        if sorted(perm) != list(range(len(input_shape))):
            raise ValueError(f"transpose: invalid permutation {perm}")
        if perm[0] != 0:
            raise ValueError(
                f"transpose: batch-axis permutation is unsupported: {perm}"
            )
        idx = (
            torch.arange(n, device=device)
            .reshape(input_shape)
            .permute(*perm)
            .reshape(-1)
        )
    y = torch.tensor(con.var_ids[:n], device=device, dtype=torch.long)
    x = torch.tensor(
        con.var_ids[n:], device=device, dtype=torch.long
    ).index_select(0, idx)
    _emit_copy_rows(y.tolist(), x.tolist(), eq, N, device, dtype)


def _emit_mask_add(
    con: Any,
    eq: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """MASK_ADD is affine: ``y_i = x_i + M_i``.

    The interval transfer function stores only ``tag=mask:*``. Recover the
    constant from propagated boxes and fail if the boxes do not prove a fixed
    offset for every batch lane.
    """
    if len(con.var_ids) % 2 != 0:
        raise ValueError(f"mask: var_ids length {len(con.var_ids)} is not even")
    n = len(con.var_ids) // 2
    y = list(con.var_ids[:n])
    x = list(con.var_ids[n:])
    y_idx = torch.tensor(y, device=device, dtype=torch.long)
    x_idx = torch.tensor(x, device=device, dtype=torch.long)
    off_lb = lb_global[:, y_idx] - lb_global[:, x_idx]
    off_ub = ub_global[:, y_idx] - ub_global[:, x_idx]
    if not torch.all(torch.isfinite(off_lb) & torch.isfinite(off_ub)):
        raise ValueError("mask: cannot recover finite mask offsets from boxes")
    if not torch.allclose(off_lb, off_ub, rtol=1e-7, atol=1e-9):
        raise ValueError("mask: output/input boxes do not encode a fixed offset")
    vals = torch.tensor(
        [[1.0, -1.0]], device=device, dtype=dtype
    ).expand(N, -1).contiguous()
    for i in range(n):
        eq.add([y[i], x[i]], vals, off_lb[:, i].contiguous())


def _emit_conv2d(
    con: Any,
    eq: _RowAcc,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """``y[c_o, h_o, w_o] - sum_{c_i, dh, dw} W[c_o, c_i, dh, dw] * x[c_i, h_i, w_i] = b[c_o, h_o, w_o]``.

    One equality per output element. Input position is
    ``h_i = h_o * stride_h - pad_h + dh * dil_h`` (similarly for ``w_i``);
    out-of-bounds positions contribute 0 (zero-padding convention).

    The weight tensor is shared across N (broadcast); only variable IDs and
    the bias RHS differ from instance to instance (here they are uniform).
    Currently restricted to ``groups == 1``; mixed groups would need
    per-group slicing on ``c_i``.
    """
    weight_raw = con.meta["weight"]
    bias_raw = con.meta["b"]
    conv_params = con.meta["conv_params"]
    input_shape = con.meta["input_shape"]
    output_shape = con.meta["output_shape"]

    stride_h, stride_w = conv_params["stride"]
    pad_h, pad_w = conv_params["padding"]
    dil_h, dil_w = conv_params["dilation"]
    groups = int(conv_params["groups"])
    if groups != 1:
        raise NotImplementedError(
            f"conv2d: groups={groups} not supported (only groups=1)"
        )

    # input_shape is (B_meta, C_in, H_in, W_in); B_meta is informational only.
    _, C_in, H_in, W_in = (int(v) for v in input_shape)
    _, C_out, H_out, W_out = (int(v) for v in output_shape)

    W_t = _as_td(weight_raw, device, dtype)
    if W_t.dim() != 4:
        raise ValueError(
            f"conv2d: weight must be 4-D [C_out, C_in, K_h, K_w]; "
            f"got shape {tuple(W_t.shape)}"
        )
    W_C_out, W_C_in, K_h, K_w = (int(v) for v in W_t.shape)
    if W_C_out != C_out or W_C_in != C_in:
        raise ValueError(
            f"conv2d: weight shape {tuple(W_t.shape)} inconsistent with "
            f"C_out={C_out}, C_in={C_in} from output/input_shape"
        )

    n_out_per_instance = C_out * H_out * W_out
    n_in_per_instance = C_in * H_in * W_in
    var_ids_all = list(con.var_ids)
    if len(var_ids_all) != n_out_per_instance + n_in_per_instance:
        raise ValueError(
            f"conv2d: var_ids length {len(var_ids_all)} != "
            f"{n_out_per_instance} (out) + {n_in_per_instance} (in)"
        )
    y_ids = var_ids_all[:n_out_per_instance]
    x_ids = var_ids_all[n_out_per_instance:]

    b_t = _as_td(bias_raw, device, dtype).reshape(-1)
    if b_t.numel() != n_out_per_instance:
        raise ValueError(
            f"conv2d: b length {b_t.numel()} != n_out_per_instance "
            f"{n_out_per_instance}"
        )

    K = C_in * K_h * K_w

    h_o_grid = torch.arange(H_out, device=device, dtype=torch.long)
    dh_grid = torch.arange(K_h, device=device, dtype=torch.long)
    h_in_map = h_o_grid.unsqueeze(1) * stride_h - pad_h + dh_grid.unsqueeze(0) * dil_h
    h_valid = (h_in_map >= 0) & (h_in_map < H_in)
    h_in_clamped = h_in_map.clamp(min=0, max=max(H_in - 1, 0))

    w_o_grid = torch.arange(W_out, device=device, dtype=torch.long)
    dw_grid = torch.arange(K_w, device=device, dtype=torch.long)
    w_in_map = w_o_grid.unsqueeze(1) * stride_w - pad_w + dw_grid.unsqueeze(0) * dil_w
    w_valid = (w_in_map >= 0) & (w_in_map < W_in)
    w_in_clamped = w_in_map.clamp(min=0, max=max(W_in - 1, 0))

    c_in_idx = torch.arange(C_in, device=device, dtype=torch.long)
    h_in_full = h_in_clamped.view(H_out, 1, 1, K_h, 1).expand(
        H_out, W_out, C_in, K_h, K_w
    )
    w_in_full = w_in_clamped.view(1, W_out, 1, 1, K_w).expand(
        H_out, W_out, C_in, K_h, K_w
    )
    c_in_full = c_in_idx.view(1, 1, C_in, 1, 1).expand(
        H_out, W_out, C_in, K_h, K_w
    )
    valid_full = (
        h_valid.view(H_out, 1, 1, K_h, 1)
        & w_valid.view(1, W_out, 1, 1, K_w)
    ).expand(H_out, W_out, C_in, K_h, K_w)

    # NCHW C-order flat index: c_i*H_in*W_in + h_i*W_in + w_i. Out-of-bounds
    # entries get the placeholder x_ids[0] paired with coefficient 0; the
    # sparse coalesce sums them into the genuine entry (or stays zero).
    x_flat_idx = (
        c_in_full * (H_in * W_in) + h_in_full * W_in + w_in_full
    )
    x_ids_t = torch.tensor(x_ids, device=device, dtype=torch.long)
    safe_flat = x_flat_idx.clamp(min=0, max=n_in_per_instance - 1)
    x_var_for_kernel = x_ids_t[safe_flat.reshape(-1)].reshape(
        H_out, W_out, C_in, K_h, K_w
    )
    x_var_for_kernel = torch.where(
        valid_full,
        x_var_for_kernel,
        torch.full_like(x_var_for_kernel, x_ids[0]),
    )

    W_kernel_per_out = (-W_t).reshape(C_out, K)
    valid_kernel = valid_full.reshape(H_out * W_out, K).to(dtype)

    m = n_out_per_instance
    y_ids_t = torch.tensor(y_ids, device=device, dtype=torch.long)
    x_cols_per_row = (
        x_var_for_kernel.reshape(H_out * W_out, K)
        .unsqueeze(0).expand(C_out, H_out * W_out, K)
        .reshape(m, K)
    )
    col_block = torch.empty((m, 1 + K), device=device, dtype=torch.long)
    col_block[:, 0] = y_ids_t
    col_block[:, 1:] = x_cols_per_row

    val_per_row = torch.empty((m, 1 + K), device=device, dtype=dtype)
    val_per_row[:, 0] = 1.0
    val_per_row[:, 1:] = (
        W_kernel_per_out.unsqueeze(1) * valid_kernel.unsqueeze(0)
    ).reshape(m, K)
    val_block = val_per_row.unsqueeze(0).expand(N, m, 1 + K).contiguous()

    rhs_block = b_t.unsqueeze(0).expand(N, m).contiguous()

    eq.add_block(col_block, val_block, rhs_block)


def _emit_relu_canonical(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Canonical 3-ineq RELU form for ALL elements.

    Rows per neuron i (z_i = relu(y_i)):
      (a)  z_i >= 0                ⇒  -z_i <= 0
      (b)  z_i >= y_i              ⇒  y_i - z_i <= 0
      (c)  z_i <= slope_i * y_i + shift_i
                                   ⇒  z_i - slope_i * y_i <= shift_i

    Degenerate slopes by phase (derived from per-N y_i bounds):
        ON  (lb_i >= 0): slope=1,                 shift=0
        OFF (ub_i <= 0): slope=0,                 shift=0
        AMB:             slope=ub/(ub-lb),        shift=-lb*slope
    """
    n = len(con.var_ids) // 2
    z = list(con.var_ids[:n])
    y = list(con.var_ids[n:])
    # Per-N bounds on y from the global box tensors.
    y_idx = torch.tensor(y, device=device, dtype=torch.long)
    y_lb = lb_global[:, y_idx]  # [N, n]
    y_ub = ub_global[:, y_idx]  # [N, n]
    on = y_lb >= 0
    off = y_ub <= 0
    amb = ~(on | off)
    gap = y_ub - y_lb
    safe_gap = torch.where(
        amb & (gap > _TANH_LIN_EPS),
        gap,
        torch.ones_like(gap),
    )
    slope_amb = torch.where(
        amb & (gap > _TANH_LIN_EPS),
        y_ub / safe_gap,
        torch.zeros_like(gap),
    )
    shift_amb = -y_lb * slope_amb
    # Final slope/shift selectors.
    slope = torch.where(on, torch.ones_like(gap), torch.where(off, torch.zeros_like(gap), slope_amb))
    shift = torch.where(on, torch.zeros_like(gap), torch.where(off, torch.zeros_like(gap), shift_amb))
    for i in range(n):
        vals_a = torch.full((N, 1), -1.0, device=device, dtype=dtype)
        rhs_a = torch.zeros((N,), device=device, dtype=dtype)
        le.add([z[i]], vals_a, rhs_a)
        vals_b = torch.tensor(
            [[1.0, -1.0]], device=device, dtype=dtype
        ).expand(N, -1).contiguous()
        rhs_b = torch.zeros((N,), device=device, dtype=dtype)
        le.add([y[i], z[i]], vals_b, rhs_b)
        vals_c = torch.empty((N, 2), device=device, dtype=dtype)
        vals_c[:, 0] = 1.0
        vals_c[:, 1] = -slope[:, i]
        rhs_c = shift[:, i].clone()
        le.add([z[i], y[i]], vals_c, rhs_c)


def _emit_lrelu_canonical(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Canonical 3-ineq LRELU with phase-degenerate slopes.

    Rows per neuron i (z = lrelu(y, alpha)):
      (a)  z_i >= y_i           ⇒  y_i - z_i <= 0
      (b)  z_i >= alpha * y_i   ⇒  alpha * y_i - z_i <= 0
      (c)  z_i <= slope_i * y_i + shift_i

    Slope/shift by phase:
        ON  (lb >= 0): slope=1,                                       shift=0
        OFF (ub <= 0): slope=alpha,                                   shift=0
        AMB:           slope=(ub - alpha*lb)/(ub-lb),                 shift=alpha*lb - slope*lb
    """
    n = len(con.var_ids) // 2
    z = list(con.var_ids[:n])
    y = list(con.var_ids[n:])
    alpha = float(con.meta["alpha"])
    y_idx = torch.tensor(y, device=device, dtype=torch.long)
    y_lb = lb_global[:, y_idx]
    y_ub = ub_global[:, y_idx]
    on = y_lb >= 0
    off = y_ub <= 0
    amb = ~(on | off)
    gap = y_ub - y_lb
    safe_gap = torch.where(
        amb & (gap > _TANH_LIN_EPS),
        gap,
        torch.ones_like(gap),
    )
    slope_amb = torch.where(
        amb & (gap > _TANH_LIN_EPS),
        (y_ub - alpha * y_lb) / safe_gap,
        torch.full_like(gap, max(alpha, 1.0)),
    )
    shift_amb = alpha * y_lb - slope_amb * y_lb
    one = torch.ones_like(gap)
    zero = torch.zeros_like(gap)
    slope = torch.where(on, one, torch.where(off, torch.full_like(gap, alpha), slope_amb))
    shift = torch.where(on, zero, torch.where(off, zero, shift_amb))
    for i in range(n):
        vals_a = torch.tensor(
            [[1.0, -1.0]], device=device, dtype=dtype
        ).expand(N, -1).contiguous()
        rhs_a = torch.zeros((N,), device=device, dtype=dtype)
        le.add([y[i], z[i]], vals_a, rhs_a)
        vals_b = torch.tensor(
            [[alpha, -1.0]], device=device, dtype=dtype
        ).expand(N, -1).contiguous()
        rhs_b = torch.zeros((N,), device=device, dtype=dtype)
        le.add([y[i], z[i]], vals_b, rhs_b)
        vals_c = torch.empty((N, 2), device=device, dtype=dtype)
        vals_c[:, 0] = 1.0
        vals_c[:, 1] = -slope[:, i]
        rhs_c = shift[:, i].clone()
        le.add([z[i], y[i]], vals_c, rhs_c)


def _emit_envelope_rows(
    le: _RowAcc,
    z: list[int],
    y: list[int],
    row_lo: torch.Tensor,
    row_hi: torch.Tensor,
    slope_lo: torch.Tensor,
    b_lo: torch.Tensor,
    slope_hi: torch.Tensor,
    b_hi: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Emit four lower and upper envelope rows for ``z`` over ``y``."""
    n = len(z)
    z_cols = torch.tensor(z, device=device, dtype=torch.long)[:, None]
    zy_cols = torch.stack(
        [
            torch.tensor(z, device=device, dtype=torch.long),
            torch.tensor(y, device=device, dtype=torch.long),
        ],
        dim=1,
    )
    le.add_block(
        z_cols,
        -torch.ones((N, n, 1), device=device, dtype=dtype),
        -row_lo,
    )
    le.add_block(
        z_cols,
        torch.ones((N, n, 1), device=device, dtype=dtype),
        row_hi,
    )
    vals_lo = torch.empty((N, n, 2), device=device, dtype=dtype)
    vals_lo[:, :, 0] = -1.0
    vals_lo[:, :, 1] = slope_lo
    le.add_block(zy_cols, vals_lo, -b_lo)
    vals_hi = torch.empty((N, n, 2), device=device, dtype=dtype)
    vals_hi[:, :, 0] = 1.0
    vals_hi[:, :, 1] = -slope_hi
    le.add_block(zy_cols, vals_hi, b_hi)


def _emit_tanh_canonical(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Canonical 4-ineq TANH form for ALL elements.

    Rows per element (z = tanh(y) over [lo, hi]):
      (a) z >= tanh(lo)               (monotone lower)
      (b) z <= tanh(hi)               (monotone upper)
      (c) z >= slope_lo * y + b_lo    (lower envelope)
      (d) z <= slope_hi * y + b_hi    (upper envelope)

    Case selection (per element, per N):
        identity-band  : max(|lo|, |hi|) <= 0.25
            slope_lo=1, b_lo=-delta  (delta = max(|tanh(lo)-lo|, |tanh(hi)-hi|, eps))
            slope_hi=1, b_hi=+delta
        cross-zero     : lo < 0 < hi
            slope_lo=tanh'(lo)  (tangent at lo, valid on convex region)
            slope_hi=tanh'(hi)  (tangent at hi, valid on concave region)
        convex segment : hi <= 0
            slope_lo=tanh'(hi)  (tangent at hi, valid lower bound)
            slope_hi=secant(lo,hi)
        concave segment: lo >= 0
            slope_lo=secant(lo,hi)
            slope_hi=tanh'(lo)
        point bounds   : |hi-lo| < eps (tanh nearly constant)
            slope_lo=0, b_lo=tanh((lo+hi)/2) (forces z >= tanh-mid)
            slope_hi=0, b_hi=tanh((lo+hi)/2) (forces z <= tanh-mid)

    All branches emit exactly 4 rows. Trivially-satisfied tangents are kept
    by intent (uniform structure across N).
    """
    n = len(con.var_ids) // 2
    z = list(con.var_ids[:n])
    y = list(con.var_ids[n:])
    y_idx = torch.tensor(y, device=device, dtype=torch.long)
    lo = lb_global[:, y_idx]
    hi = ub_global[:, y_idx]
    # Sanitise non-finite bounds by clamping to a wide finite range; the
    # monotone rows still hold and the envelope rows become trivial.
    big = 1e30
    lo_s = torch.where(torch.isfinite(lo), lo, torch.full_like(lo, -big))
    hi_s = torch.where(torch.isfinite(hi), hi, torch.full_like(hi, big))
    # Order safety: ensure hi >= lo
    hi_s = torch.maximum(hi_s, lo_s)
    f_lo = torch.tanh(lo_s)
    f_hi = torch.tanh(hi_s)
    dfdy_lo = 1.0 - f_lo * f_lo
    dfdy_hi = 1.0 - f_hi * f_hi
    gap = hi_s - lo_s
    safe_gap = torch.where(gap > _TANH_LIN_EPS, gap, torch.ones_like(gap))
    secant_slope = (f_hi - f_lo) / safe_gap
    secant_intercept = f_lo - secant_slope * lo_s
    tang_lo_intercept = f_lo - dfdy_lo * lo_s
    tang_hi_intercept = f_hi - dfdy_hi * hi_s
    max_abs = torch.maximum(lo_s.abs(), hi_s.abs())
    identity_band = max_abs <= _TANH_BAND
    point_bounds = ~identity_band & (gap <= _TANH_LIN_EPS)
    convex_seg = ~identity_band & ~point_bounds & (hi_s <= -_TANH_LIN_EPS)
    concave_seg = ~identity_band & ~point_bounds & (lo_s >= _TANH_LIN_EPS)
    # Identity-band: |tanh(x) - x| is bounded by tanh(0.25) - 0.25 ~ 0.005.
    delta_band = torch.maximum(
        torch.maximum((f_lo - lo_s).abs(), (f_hi - hi_s).abs()),
        torch.full_like(lo_s, _TANH_BAND_TOL),
    )
    # Point case: tanh-mid forces z to the (nearly) constant value.
    mid = 0.5 * (lo_s + hi_s)
    f_mid = torch.tanh(mid)
    # Compose per-case slope/intercept tensors.
    zero = torch.zeros_like(lo_s)
    one = torch.ones_like(lo_s)
    # Per-case slope/intercept selection (soundness mandate: each line must
    # be a valid bound on tanh throughout [lo, hi] in its case).
    #   identity-band : both envelopes use slope=1 with +/- delta_band
    #                   (since |tanh(x) - x| <= delta_band on this range)
    #   convex (hi<=0): tanh convex on (-inf, 0]; tangent_at_hi is a global
    #                   lower bound; secant from (lo,f(lo)) to (hi,f(hi)) is
    #                   a global upper bound for convex segment
    #   concave(lo>=0): tanh concave on [0, inf); secant is global lower
    #                   bound; tangent_at_lo is global upper bound
    #   cross-zero    : no single sloped line is uniformly above (or below)
    #                   tanh across [lo, hi] for all asymmetries -- secant
    #                   flips role depending on |lo| vs hi. Fall back to
    #                   horizontal bounds (redundant with monotone rows 1/2
    #                   but preserves uniform 4-row structure).
    #   point         : both envelopes horizontal at tanh(mid)
    slope_lo = torch.where(
        identity_band, one,
        torch.where(
            point_bounds, zero,
            torch.where(
                convex_seg, dfdy_hi,
                torch.where(concave_seg, secant_slope, zero),
            ),
        ),
    )
    b_lo = torch.where(
        identity_band, -delta_band,
        torch.where(
            point_bounds, f_mid,
            torch.where(
                convex_seg, tang_hi_intercept,
                torch.where(concave_seg, secant_intercept, f_lo),
            ),
        ),
    )
    slope_hi = torch.where(
        identity_band, one,
        torch.where(
            point_bounds, zero,
            torch.where(
                convex_seg, secant_slope,
                torch.where(concave_seg, dfdy_lo, zero),
            ),
        ),
    )
    b_hi = torch.where(
        identity_band, delta_band,
        torch.where(
            point_bounds, f_mid,
            torch.where(
                convex_seg, secant_intercept,
                torch.where(concave_seg, tang_lo_intercept, f_hi),
            ),
        ),
    )
    _emit_envelope_rows(
        le, z, y, f_lo, f_hi, slope_lo, b_lo, slope_hi, b_hi, N, device, dtype
    )


def _emit_sigmoid_canonical(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Canonical 4-ineq SIGMOID envelope, mirroring TANH by curvature."""
    _emit_scurve(
        con, le, lb_global, ub_global, N, device, dtype,
        torch.sigmoid, lambda value: torch.sigmoid(value) * (1.0 - torch.sigmoid(value)),
    )


def _emit_erf_canonical(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Canonical 4-ineq ERF envelope, mirroring SIGMOID by curvature."""
    _emit_scurve(
        con, le, lb_global, ub_global, N, device, dtype,
        torch.erf, lambda value: (2.0 / float(np.sqrt(np.pi))) * torch.exp(-value * value),
    )


def _emit_scurve(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
    fn: Callable[[torch.Tensor], torch.Tensor],
    dfn: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Emit the shared sigmoid-like curvature envelope."""
    n = len(con.var_ids) // 2
    z = list(con.var_ids[:n])
    y = list(con.var_ids[n:])
    y_idx = torch.tensor(y, device=device, dtype=torch.long)
    lo = lb_global[:, y_idx]
    hi = ub_global[:, y_idx]
    big = 1e30
    lo_s = torch.where(torch.isfinite(lo), lo, torch.full_like(lo, -big))
    hi_s = torch.where(torch.isfinite(hi), hi, torch.full_like(hi, big))
    hi_s = torch.maximum(hi_s, lo_s)
    f_lo = fn(lo_s)
    f_hi = fn(hi_s)
    df_lo = dfn(lo_s)
    df_hi = dfn(hi_s)
    gap = hi_s - lo_s
    safe_gap = torch.where(gap > _TANH_LIN_EPS, gap, torch.ones_like(gap))
    secant_slope = (f_hi - f_lo) / safe_gap
    secant_intercept = f_lo - secant_slope * lo_s
    tang_lo_intercept = f_lo - df_lo * lo_s
    tang_hi_intercept = f_hi - df_hi * hi_s
    point_bounds = gap <= _TANH_LIN_EPS
    convex_seg = ~point_bounds & (hi_s <= -_TANH_LIN_EPS)
    concave_seg = ~point_bounds & (lo_s >= _TANH_LIN_EPS)
    mid = 0.5 * (lo_s + hi_s)
    f_mid = fn(mid)
    zero = torch.zeros_like(lo_s)
    slope_lo = torch.where(
        point_bounds, zero,
        torch.where(convex_seg, df_hi, torch.where(concave_seg, secant_slope, zero)),
    )
    b_lo = torch.where(
        point_bounds, f_mid,
        torch.where(convex_seg, tang_hi_intercept, torch.where(concave_seg, secant_intercept, f_lo)),
    )
    slope_hi = torch.where(
        point_bounds, zero,
        torch.where(convex_seg, secant_slope, torch.where(concave_seg, df_lo, zero)),
    )
    b_hi = torch.where(
        point_bounds, f_mid,
        torch.where(convex_seg, secant_intercept, torch.where(concave_seg, tang_lo_intercept, f_hi)),
    )
    _emit_envelope_rows(
        le, z, y, f_lo, f_hi, slope_lo, b_lo, slope_hi, b_hi, N, device, dtype
    )



def _emit_sqrt_canonical(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Canonical 4-ineq SQRT envelope on clamped domain x >= 0."""
    n = len(con.var_ids) // 2
    z = list(con.var_ids[:n])
    y = list(con.var_ids[n:])
    y_idx = torch.tensor(y, device=device, dtype=torch.long)
    lo = lb_global[:, y_idx]
    hi = ub_global[:, y_idx]
    lo_e = torch.clamp(torch.where(torch.isfinite(lo), lo, torch.zeros_like(lo)), min=0.0)
    hi_e = torch.clamp(torch.where(torch.isfinite(hi), hi, lo_e), min=0.0)
    hi_e = torch.maximum(hi_e, lo_e)
    f_lo = torch.sqrt(lo_e)
    f_hi = torch.sqrt(hi_e)
    gap = hi_e - lo_e
    safe_gap = torch.where(gap > _TANH_LIN_EPS, gap, torch.ones_like(gap))
    slope_lo = (f_hi - f_lo) / safe_gap
    b_lo = f_lo - slope_lo * lo_e
    slope_hi = 0.5 / torch.sqrt(torch.clamp(hi_e, min=_TANH_LIN_EPS))
    b_hi = f_hi - slope_hi * hi_e
    degenerate = (gap <= _TANH_LIN_EPS) | (hi_e <= _TANH_LIN_EPS)
    mid_val = torch.sqrt(0.5 * (lo_e + hi_e))
    zero = torch.zeros_like(lo_e)
    slope_lo = torch.where(degenerate, zero, slope_lo)
    b_lo = torch.where(degenerate, mid_val, b_lo)
    slope_hi = torch.where(degenerate, zero, slope_hi)
    b_hi = torch.where(degenerate, mid_val, b_hi)
    _emit_envelope_rows(
        le, z, y, f_lo, f_hi, slope_lo, b_lo, slope_hi, b_hi, N, device, dtype
    )


def _emit_periodic_box_canonical(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
    op_name: str,
) -> None:
    """Emit the four-row constant box relaxation for non-monotone periodic ops."""
    n = len(con.var_ids) // 2
    z = list(con.var_ids[:n])
    z_idx = torch.tensor(z, device=device, dtype=torch.long)
    lo = lb_global[:, z_idx]
    hi = ub_global[:, z_idx]
    if not torch.all(torch.isfinite(lo) & torch.isfinite(hi)):
        raise ValueError(f"{op_name}: finite output bounds are required")
    for i in range(n):
        vals_a = torch.full((N, 1), -1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_a, -lo[:, i])
        vals_b = torch.full((N, 1), 1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_b, hi[:, i])
        vals_c = torch.full((N, 1), -1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_c, -lo[:, i])
        vals_d = torch.full((N, 1), 1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_d, hi[:, i])


def _quantize_qdq_value(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, qmin: torch.Tensor, qmax: torch.Tensor) -> torch.Tensor:
    return scale * torch.clamp(torch.round(x / scale), min=qmin - zero_point, max=qmax - zero_point)


def _emit_quantize_canonical(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Canonical QDQ envelope: saturating box plus guarded +/- scale/2 band."""
    n = len(con.var_ids) // 2
    z = list(con.var_ids[:n])
    y = list(con.var_ids[n:])
    y_idx = torch.tensor(y, device=device, dtype=torch.long)
    lo = lb_global[:, y_idx]
    hi = torch.maximum(ub_global[:, y_idx], lo)
    if not torch.all(torch.isfinite(lo) & torch.isfinite(hi)):
        raise ValueError("quantize: finite input bounds are required")
    scale_1d, zp_1d, qmin_1d, qmax_1d = _quantize_params_flat(con.meta, "quantize", n, device, dtype)
    scale = scale_1d.unsqueeze(0).expand(N, n)
    zp = zp_1d.unsqueeze(0).expand(N, n)
    qmin = qmin_1d.unsqueeze(0).expand(N, n)
    qmax = qmax_1d.unsqueeze(0).expand(N, n)
    z_lo_raw = _quantize_qdq_value(lo, scale, zp, qmin, qmax)
    z_hi_raw = _quantize_qdq_value(hi, scale, zp, qmin, qmax)
    z_lo = torch.minimum(z_lo_raw, z_hi_raw)
    z_hi = torch.maximum(z_lo_raw, z_hi_raw)
    q_lo = torch.round(lo / scale) + zp
    q_hi = torch.round(hi / scale) + zp
    unsaturated = (qmin <= q_lo) & (q_hi <= qmax)
    slope_lo = torch.where(unsaturated, torch.ones_like(scale), torch.zeros_like(scale))
    b_lo = torch.where(unsaturated, -0.5 * scale, z_lo)
    slope_hi = torch.where(unsaturated, torch.ones_like(scale), torch.zeros_like(scale))
    b_hi = torch.where(unsaturated, 0.5 * scale, z_hi)
    for i in range(n):
        vals_a = torch.full((N, 1), -1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_a, -z_lo[:, i])
        vals_b = torch.full((N, 1), 1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_b, z_hi[:, i])
        vals_c = torch.empty((N, 2), device=device, dtype=dtype)
        vals_c[:, 0] = -1.0
        vals_c[:, 1] = slope_lo[:, i]
        le.add([z[i], y[i]], vals_c, -b_lo[:, i])
        vals_d = torch.empty((N, 2), device=device, dtype=dtype)
        vals_d[:, 0] = 1.0
        vals_d[:, 1] = -slope_hi[:, i]
        le.add([z[i], y[i]], vals_d, b_hi[:, i])


def _emit_clipped_affine_hull(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    alpha: float,
    beta: float,
    clamp_min: float,
    clamp_max: float,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if alpha <= 0.0:
        raise ValueError("clipped affine hull: alpha must be positive")
    if clamp_min > clamp_max:
        raise ValueError(
            f"clipped affine hull: clamp_min {clamp_min} > clamp_max {clamp_max}"
        )
    n = len(con.var_ids) // 2
    z = list(con.var_ids[:n])
    x = list(con.var_ids[n:])
    x_idx = torch.tensor(x, device=device, dtype=torch.long)
    lo = lb_global[:, x_idx]
    hi = torch.maximum(ub_global[:, x_idx], lo)
    if not torch.all(torch.isfinite(lo) & torch.isfinite(hi)):
        raise ValueError("clipped affine hull: finite input bounds are required")
    for i in range(n):
        vals_min = torch.full((N, 1), -1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_min, torch.full((N,), -clamp_min, device=device, dtype=dtype))

        vals_max = torch.full((N, 1), 1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_max, torch.full((N,), clamp_max, device=device, dtype=dtype))

        upper_excess = torch.clamp(alpha * hi[:, i] + beta - clamp_max, min=0.0)
        vals_lower = torch.empty((N, 2), device=device, dtype=dtype)
        vals_lower[:, 0] = -1.0
        vals_lower[:, 1] = alpha
        le.add([z[i], x[i]], vals_lower, upper_excess - beta)

        lower_deficit = torch.clamp(clamp_min - (alpha * lo[:, i] + beta), min=0.0)
        vals_upper = torch.empty((N, 2), device=device, dtype=dtype)
        vals_upper[:, 0] = 1.0
        vals_upper[:, 1] = -alpha
        le.add([z[i], x[i]], vals_upper, beta + lower_deficit)


def _emit_abs_canonical(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Canonical 4-ineq ABS form for ALL elements (z = |y|).

    Rows per element:
      (a) z >= 0     ⇒  -z <= 0
      (b) z >= y     ⇒  y - z <= 0
      (c) z >= -y    ⇒  -y - z <= 0
      (d) z <= max(|lb|, |ub|) (constant per-N upper envelope)
    """
    n = len(con.var_ids) // 2
    z = list(con.var_ids[:n])
    y = list(con.var_ids[n:])
    y_idx = torch.tensor(y, device=device, dtype=torch.long)
    lo = lb_global[:, y_idx]
    hi = ub_global[:, y_idx]
    big = 1e30
    lo_s = torch.where(torch.isfinite(lo), lo, torch.full_like(lo, -big))
    hi_s = torch.where(torch.isfinite(hi), hi, torch.full_like(hi, big))
    upper_env = torch.maximum(lo_s.abs(), hi_s.abs())
    for i in range(n):
        vals_a = torch.full((N, 1), -1.0, device=device, dtype=dtype)
        rhs_a = torch.zeros((N,), device=device, dtype=dtype)
        le.add([z[i]], vals_a, rhs_a)
        vals_b = torch.tensor(
            [[1.0, -1.0]], device=device, dtype=dtype
        ).expand(N, -1).contiguous()
        rhs_b = torch.zeros((N,), device=device, dtype=dtype)
        le.add([y[i], z[i]], vals_b, rhs_b)
        vals_c = torch.tensor(
            [[-1.0, -1.0]], device=device, dtype=dtype
        ).expand(N, -1).contiguous()
        rhs_c = torch.zeros((N,), device=device, dtype=dtype)
        le.add([y[i], z[i]], vals_c, rhs_c)
        vals_d = torch.full((N, 1), 1.0, device=device, dtype=dtype)
        rhs_d = upper_env[:, i].clone()
        le.add([z[i]], vals_d, rhs_d)


def _emit_square_canonical(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Convex relaxation for ``z = x*x`` over finite input boxes."""
    n = len(con.var_ids) // 2
    z = list(con.var_ids[:n])
    x = list(con.var_ids[n:])
    x_idx = torch.tensor(x, device=device, dtype=torch.long)
    lo = lb_global[:, x_idx]
    hi = ub_global[:, x_idx]
    if not torch.all(torch.isfinite(lo) & torch.isfinite(hi)):
        raise ValueError("square: finite input bounds are required")
    hi = torch.maximum(hi, lo)
    for i in range(n):
        # Tangent at lo: z >= 2*lo*x - lo^2.
        vals_lo = torch.empty((N, 2), device=device, dtype=dtype)
        vals_lo[:, 0] = -1.0
        vals_lo[:, 1] = 2.0 * lo[:, i]
        rhs_lo = lo[:, i] * lo[:, i]
        le.add([z[i], x[i]], vals_lo, rhs_lo)
        # Tangent at hi: z >= 2*hi*x - hi^2.
        vals_hi = torch.empty((N, 2), device=device, dtype=dtype)
        vals_hi[:, 0] = -1.0
        vals_hi[:, 1] = 2.0 * hi[:, i]
        rhs_hi = hi[:, i] * hi[:, i]
        le.add([z[i], x[i]], vals_hi, rhs_hi)
        # Secant upper: z <= (lo+hi)*x - lo*hi.
        vals_sec = torch.empty((N, 2), device=device, dtype=dtype)
        vals_sec[:, 0] = 1.0
        vals_sec[:, 1] = -(lo[:, i] + hi[:, i])
        rhs_sec = -(lo[:, i] * hi[:, i])
        le.add([z[i], x[i]], vals_sec, rhs_sec)


def _emit_relu_power_canonical(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Canonical relaxation for POWER's ReLU-before-power semantics."""
    p = float(con.meta["p"])
    if abs(p - 2.0) > 1e-12:
        raise NotImplementedError(f"power: only p=2 is supported, got p={p}")
    n = len(con.var_ids) // 2
    z = list(con.var_ids[:n])
    x = list(con.var_ids[n:])
    x_idx = torch.tensor(x, device=device, dtype=torch.long)
    lo = lb_global[:, x_idx]
    hi = torch.maximum(ub_global[:, x_idx], lo)
    if not torch.all(torch.isfinite(lo) & torch.isfinite(hi)):
        raise ValueError("power: finite input bounds are required")
    zero = torch.zeros((N,), device=device, dtype=dtype)
    for i in range(n):
        vals_nonneg = torch.full((N, 1), -1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_nonneg, zero)

        active_lo = torch.clamp(lo[:, i], min=0.0)
        active_hi = torch.clamp(hi[:, i], min=0.0)
        active_mid = 0.5 * (active_lo + active_hi)
        for tangent in (active_lo, active_mid, active_hi):
            vals_tangent = torch.empty((N, 2), device=device, dtype=dtype)
            vals_tangent[:, 0] = -1.0
            vals_tangent[:, 1] = 2.0 * tangent
            le.add([z[i], x[i]], vals_tangent, tangent * tangent)

        inactive = hi[:, i] <= 0.0
        active = lo[:, i] >= 0.0
        gap = hi[:, i] - lo[:, i]
        active_slope = lo[:, i] + hi[:, i]
        active_intercept = -(lo[:, i] * hi[:, i])
        mixed_slope = (hi[:, i] * hi[:, i]) / torch.clamp(gap, min=_TANH_LIN_EPS)
        mixed_intercept = -mixed_slope * lo[:, i]
        slope = torch.where(inactive, zero, torch.where(active, active_slope, mixed_slope))
        intercept = torch.where(
            inactive, zero, torch.where(active, active_intercept, mixed_intercept),
        )
        vals_upper = torch.empty((N, 2), device=device, dtype=dtype)
        vals_upper[:, 0] = 1.0
        vals_upper[:, 1] = -slope
        le.add([z[i], x[i]], vals_upper, intercept)


def _emit_layernorm_box(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Sound layernorm relaxation using propagated output interval rows.

    Exact layernorm couples all inputs through mean and variance. The interval
    transfer function already computes a sound output box; exporting those rows
    preserves soundness and restores batch coverage without adding nonlinear
    terms to the LP.
    """
    n = len(con.var_ids) // 2
    z = list(con.var_ids[:n])
    z_idx = torch.tensor(z, device=device, dtype=torch.long)
    lo = lb_global[:, z_idx]
    hi = ub_global[:, z_idx]
    if not torch.all(torch.isfinite(lo) & torch.isfinite(hi)):
        raise ValueError("layernorm: finite output bounds are required")
    for i in range(n):
        vals_lb = torch.full((N, 1), -1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_lb, -lo[:, i])
        vals_ub = torch.full((N, 1), 1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_ub, hi[:, i])


def _emit_attn_dual_planar(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Export ReLU-catalyzed dual-planar attention scores as their output box.

    The fused bound is affine in the perturbed embeddings, not in the per-layer
    score variables, so the LP path uses the propagated output interval rows
    (mirrors the layernorm export) to stay sound without introducing nonlinear
    terms. The score variables are the only ids on this constraint.
    """
    z = list(con.var_ids)
    z_idx = torch.tensor(z, device=device, dtype=torch.long)
    lo = lb_global[:, z_idx]
    hi = ub_global[:, z_idx]
    if not torch.all(torch.isfinite(lo) & torch.isfinite(hi)):
        raise ValueError("att_dual_planar: finite output bounds are required")
    for i in range(len(z)):
        vals_lb = torch.full((N, 1), -1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_lb, -lo[:, i])
        vals_ub = torch.full((N, 1), 1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_ub, hi[:, i])


def _emit_mcc(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """4-ineq McCormick envelope for ``z = x * y`` (uniform across phases).

    Bounds (lx, ux) on x and (ly, uy) on y are read from the box-tensor
    aggregator so this works even when the meta tensors were produced under
    a per-N analyze. Rows per element:
      (a) z >= lx*y + ly*x - lx*ly         ⇒  -z + lx*y + ly*x <= -lx*ly
      (b) z >= ux*y + uy*x - ux*uy         ⇒  -z + ux*y + uy*x <= -ux*uy
      (c) z <= lx*y + uy*x - lx*uy         ⇒   z - lx*y - uy*x <= -lx*uy
      (d) z <= ux*y + ly*x - ux*ly         ⇒   z - ux*y - ly*x <= -ux*ly
    """
    n = len(con.var_ids) // 3
    z = list(con.var_ids[:n])
    x = list(con.var_ids[n:2 * n])
    y = list(con.var_ids[2 * n:])
    x_idx = torch.tensor(x, device=device, dtype=torch.long)
    y_idx = torch.tensor(y, device=device, dtype=torch.long)
    lx = lb_global[:, x_idx]
    ux = ub_global[:, x_idx]
    ly = lb_global[:, y_idx]
    uy = ub_global[:, y_idx]
    for i in range(n):
        # (a) -z + lx*y + ly*x <= -lx*ly
        vals_a = torch.empty((N, 3), device=device, dtype=dtype)
        vals_a[:, 0] = -1.0
        vals_a[:, 1] = lx[:, i]   # coeff on y
        vals_a[:, 2] = ly[:, i]   # coeff on x
        rhs_a = -lx[:, i] * ly[:, i]
        le.add([z[i], y[i], x[i]], vals_a, rhs_a)
        # (b) -z + ux*y + uy*x <= -ux*uy
        vals_b = torch.empty((N, 3), device=device, dtype=dtype)
        vals_b[:, 0] = -1.0
        vals_b[:, 1] = ux[:, i]
        vals_b[:, 2] = uy[:, i]
        rhs_b = -ux[:, i] * uy[:, i]
        le.add([z[i], y[i], x[i]], vals_b, rhs_b)
        # (c) z - lx*y - uy*x <= -lx*uy
        vals_c = torch.empty((N, 3), device=device, dtype=dtype)
        vals_c[:, 0] = 1.0
        vals_c[:, 1] = -lx[:, i]
        vals_c[:, 2] = -uy[:, i]
        rhs_c = -lx[:, i] * uy[:, i]
        le.add([z[i], y[i], x[i]], vals_c, rhs_c)
        # (d) z - ux*y - ly*x <= -ux*ly
        vals_d = torch.empty((N, 3), device=device, dtype=dtype)
        vals_d[:, 0] = 1.0
        vals_d[:, 1] = -ux[:, i]
        vals_d[:, 2] = -ly[:, i]
        rhs_d = -ux[:, i] * ly[:, i]
        le.add([z[i], y[i], x[i]], vals_d, rhs_d)


def _emit_softmax_simplex(
    con: Any,
    eq: _RowAcc,
    le: _RowAcc,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """``softmax:simplex:`` rows: per row of size ``rowsize``,
    one ``sum=1`` equality plus ``rowsize`` non-negativity inequalities."""
    rowsize = int(con.meta["rowsize"])
    W = list(con.var_ids)
    if len(W) % rowsize != 0:
        raise ValueError(
            f"softmax:simplex: var count {len(W)} not divisible by rowsize "
            f"{rowsize}"
        )
    for r in range(len(W) // rowsize):
        row = W[r * rowsize:(r + 1) * rowsize]
        # sum_eq: sum(row) = 1
        vals_eq = torch.ones((N, rowsize), device=device, dtype=dtype)
        rhs_eq = torch.ones((N,), device=device, dtype=dtype)
        eq.add(row, vals_eq, rhs_eq)
        # ge_zero (rowsize rows): -var_i <= 0
        for vi in row:
            vals = torch.full((N, 1), -1.0, device=device, dtype=dtype)
            rhs = torch.zeros((N,), device=device, dtype=dtype)
            le.add([vi], vals, rhs)


def _emit_slice_gather(
    con: Any,
    eq: _RowAcc,
    is_gather: bool,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Slice/gather rows: out_vid - in_vid[mapped] = 0 per output element."""
    inp_shape = tuple(int(v) for v in con.meta["input_shape"])
    n_in = int(np.prod(inp_shape))
    n_out = len(con.var_ids) - n_in
    out_vids = list(con.var_ids[:n_out])
    in_vids = list(con.var_ids[n_out:])
    if is_gather:
        axis = int(con.meta["axis"])
        indices = [int(i) for i in con.meta["indices"]]
        idx = np.arange(n_in).reshape(inp_shape)
        mapped = np.take(idx, indices, axis=axis).ravel()
    else:
        starts = con.meta["starts"]
        ends = con.meta["ends"]
        axes = con.meta["axes"]
        steps = con.meta["steps"]
        idx = np.arange(n_in).reshape(inp_shape)
        slc = [slice(None)] * len(inp_shape)
        for i, ax in enumerate(axes):
            s = int(starts[i])
            e = min(int(ends[i]), inp_shape[int(ax)])
            st = int(steps[i])
            slc[int(ax)] = slice(s, e, st)
        mapped = idx[tuple(slc)].ravel()
    for k in range(n_out):
        ov = out_vids[k]
        iv = in_vids[int(mapped[k])]
        vals = torch.tensor(
            [[1.0, -1.0]], device=device, dtype=dtype
        ).expand(N, -1).contiguous()
        rhs = torch.zeros((N,), device=device, dtype=dtype)
        eq.add([ov, iv], vals, rhs)


def _emit_copy_rows(
    y: list[int], x: list[int], eq: _RowAcc, N: int,
    device: torch.device, dtype: torch.dtype,
) -> None:
    """Emit ``y_i - x_i = 0`` for one operator's resolved operand mapping."""
    col_block = torch.tensor(list(zip(y, x)), device=device, dtype=torch.long)
    val_block = torch.tensor([1.0, -1.0], device=device, dtype=dtype).expand(N, len(y), 2).contiguous()
    eq.add_block(col_block, val_block, torch.zeros((N, len(y)), device=device, dtype=dtype))


def _emit_concat(
    con: Any, eq: _RowAcc, N: int, device: torch.device, dtype: torch.dtype,
) -> None:
    """Axis-1 concatenation lists every operand's variables in output order."""
    n = len(con.var_ids) // 2
    _emit_copy_rows(list(con.var_ids[:n]), list(con.var_ids[n:]), eq, N, device, dtype)


def _emit_expand(
    con: Any, eq: _RowAcc, N: int, device: torch.device, dtype: torch.dtype,
) -> None:
    """Each broadcast output copies its source element."""
    input_shape = tuple(int(v) for v in con.meta["input_shape"])
    mapped = np.broadcast_to(
        np.arange(int(np.prod(input_shape))).reshape(input_shape),
        tuple(int(v) for v in con.meta["output_shape"]),
    ).ravel()
    n_out = mapped.size
    if len(con.var_ids) != n_out + int(np.prod(input_shape)):
        raise ValueError(f"expand: {len(con.var_ids)} variables for {n_out} outputs")
    x = con.var_ids[n_out:]
    _emit_copy_rows(list(con.var_ids[:n_out]), [x[int(j)] for j in mapped], eq, N, device, dtype)


def _emit_constant(
    con: Any, eq: _RowAcc, N: int, device: torch.device, dtype: torch.dtype,
) -> None:
    """Fix each output: ``y_i = value_i``."""
    value = _as_td(con.meta["value"], device, dtype).reshape(-1)
    if value.numel() != len(con.var_ids):
        raise ValueError(f"constant: {value.numel()} values for {len(con.var_ids)} outputs")
    eq.add_block(
        torch.tensor(con.var_ids, device=device, dtype=torch.long)[:, None],
        torch.ones((N, value.numel(), 1), device=device, dtype=dtype),
        value.expand(N, -1).contiguous(),
    )


def _emit_reduction(
    con: Any, eq: _RowAcc, mean: bool, N: int, device: torch.device, dtype: torch.dtype,
) -> None:
    """``group*y_j - sum(x_i) = 0`` for a mean, ``y_j - sum(x_i) = 0`` for a sum.

    Multiplying out the integer group size keeps a mean's coefficients exact
    instead of rounding ``1/group`` in the proof.
    """
    input_shape = tuple(int(v) for v in con.meta["input_shape"])
    axes = tuple(int(v) for v in con.meta["axes"])
    n_in = int(np.prod(input_shape))
    kept = tuple(1 if axis in axes else size for axis, size in enumerate(input_shape))
    n_out = int(np.prod(kept))
    if len(con.var_ids) != n_out + n_in:
        raise ValueError(f"reduction: {len(con.var_ids)} variables for {n_out}+{n_in}")
    group = n_in // n_out
    # Moving the reduced axes last leaves the kept axes in output order.
    members = np.moveaxis(
        np.arange(n_in).reshape(input_shape), axes, tuple(range(-len(axes), 0)),
    ).reshape(n_out, group)
    y, x = con.var_ids[:n_out], con.var_ids[n_out:]
    col_block = torch.tensor(
        [[y[j], *(x[int(i)] for i in owned)] for j, owned in enumerate(members)],
        device=device, dtype=torch.long,
    )
    values = torch.full((N, n_out, group + 1), -1.0, device=device, dtype=dtype)
    values[:, :, 0] = float(group) if mean else 1.0
    eq.add_block(col_block, values, torch.zeros((N, n_out), device=device, dtype=dtype))


def _emit_in_linpoly(
    con: Any,
    le: _RowAcc,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """``in:linpoly``: ``A x <= b`` rows on the input variables."""
    A_raw = con.meta["A"]
    b_raw = con.meta["b"]
    A_t = _as_td(A_raw, device, dtype)
    b_t = _as_td(b_raw, device, dtype)
    vids = list(con.var_ids)
    if A_t.dim() != 2 or A_t.shape[1] != len(vids):
        raise ValueError(
            f"in:linpoly: A shape {tuple(A_t.shape)} incompatible with "
            f"{len(vids)} input vars"
        )
    for i in range(A_t.shape[0]):
        vals = A_t[i].unsqueeze(0).expand(N, -1).contiguous()
        rhs = b_t[i].expand(N).contiguous()
        le.add(vids, vals, rhs)


# -----------------------------------------------------------------------------
# Inline ASSERT (negated) encoder
# -----------------------------------------------------------------------------


def _emit_assert_canonical(
    assert_layer: Any,
    out_ids: list[int],
    le: _RowAcc,
    *,
    N: int,
    nvars_net: int,
    nvars_total: int,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Encode the negated ASSERT spec from batched ASSERT tensors.

    All ASSERT params are already produced in batched ``[B, ...]`` form by
    ``OutputSpec.encode_linear`` (front-end). Slack variables (TOP1/RANGE)
    live at ``nvars_net`` (shared id across N; per-N values via [N, nvars]
    lb/ub tensors).
    """
    from act.front_end.specs import OutKind

    kind = assert_layer.params.get("kind")
    n_out = len(out_ids)

    if kind == OutKind.LINEAR_LE:
        # Negated LINEAR_LE in LE form: -c · y <= -d - eps.
        c_raw = assert_layer.params["c"]
        d_raw = assert_layer.params["d"]
        _c_probe = c_raw if isinstance(c_raw, torch.Tensor) else torch.as_tensor(c_raw)
        # Multi-row conjunction = more than one row PER batch lane (numel > N*n_out).
        # A batched single-row LINEAR_LE is [N, n_out] (numel == N*n_out) and must
        # NOT trip this guard, otherwise B>1 batches are misread as a conjunction.
        if _c_probe.shape[-1] == n_out and _c_probe.numel() > N * n_out:
            raise NotImplementedError(
                "Multi-row LINEAR_LE (conjunction) is not supported by the "
                "LP/export tier: its negation is disjunctive and needs one solve "
                "per row. Use solver_tier in {dual, dual_alpha, dual_alpha_eta}."
            )
        c_b = _coerce_b_tensor(c_raw, N, n_out, device, dtype, "c")
        d_b = _as_td(d_raw, device, dtype).reshape(-1)
        if d_b.numel() == 1:
            d_b = d_b.expand(N).contiguous()
        elif d_b.numel() != N:
            raise ValueError(
                f"LINEAR_LE: d length {d_b.numel()} != N {N}"
            )
        vals = -c_b
        rhs = -d_b - _ASSERT_EPS
        le.add(list(out_ids), vals, rhs)
        return

    if kind == OutKind.UNSAFE_LINEAR:
        # M rows of c_i · y <= d_i + eps; params are batched as
        # C[B*M, n_out] and thresholds[B, M].
        M = int(assert_layer.params.get("M", 1))
        C_raw = assert_layer.params.get("C")
        if C_raw is None:
            # Fall back to "c" (3-D or 2-D)
            C_raw = assert_layer.params["c"]
        C_t = _as_td(C_raw, device, dtype)
        if C_t.dim() == 3:
            C_t = C_t.reshape(C_t.shape[0] * C_t.shape[1], C_t.shape[2])
        if C_t.shape != (N * M, n_out):
            # Expand from [M, n_out] when B-uniform.
            if C_t.shape == (M, n_out):
                C_t = C_t.unsqueeze(0).expand(N, -1, -1).reshape(N * M, n_out)
            else:
                raise ValueError(
                    f"UNSAFE_LINEAR: C shape {tuple(C_t.shape)} incompatible "
                    f"with N={N} M={M} n_out={n_out}"
                )
        d_raw = assert_layer.params.get("thresholds")
        if d_raw is None:
            d_raw = assert_layer.params["d"]
        d_t = _as_td(d_raw, device, dtype)
        if d_t.dim() == 1 and d_t.numel() == M:
            d_t = d_t.unsqueeze(0).expand(N, -1)
        elif d_t.shape == (1, M):
            d_t = d_t.expand(N, -1)
        if d_t.shape != (N, M):
            raise ValueError(
                f"UNSAFE_LINEAR: d shape {tuple(d_t.shape)} != ({N}, {M})"
            )
        C_view = C_t.reshape(N, M, n_out)
        for i in range(M):
            vals = C_view[:, i, :]
            rhs = d_t[:, i] + _ASSERT_EPS
            le.add(list(out_ids), vals, rhs)
        return

    def per_batch(raw: Any, value_dtype: torch.dtype, name: str) -> torch.Tensor:
        """Normalise a scalar or tensor ASSERT field to one value per batch."""
        values = (
            raw.to(device=device, dtype=value_dtype).reshape(-1)
            if isinstance(raw, torch.Tensor)
            else torch.tensor([raw], device=device, dtype=value_dtype)
        )
        if values.numel() == 1 and N > 1:
            values = values.expand(N)
        if values.numel() != N:
            raise ValueError(f"{kind}: {name} length {values.numel()} != N {N}")
        return values

    if kind == OutKind.TOP1_ROBUST:
        # Slack v at slot nvars_net (per §K). K rows: for j in 0..K-1:
        #   v - y_j + y_{t_b} >= 0   ⇒   -v + y_j - y_{t_b} <= 0
        # Plus one standalone v >= 0 row ⇒ -v <= 0. Total K+1 rows per instance.
        if nvars_total != nvars_net + 1:
            raise ValueError(
                f"TOP1_ROBUST requires 1 slack var; nvars_total={nvars_total} "
                f"vs nvars_net={nvars_net}"
            )
        slack_id = nvars_net
        y_true_b = per_batch(assert_layer.params["y_true"], torch.long, "y_true")
        k_classes = n_out
        # For each j, vars referenced are [slack, y_j, y_{t_b}]. y_{t_b}
        # differs per instance, so we emit per-j rows where vars include
        # y_{t_b} as the third var, but indexed per N.
        for j in range(k_classes):
            row_vars = [slack_id] + list(out_ids)
            vals = torch.zeros((N, 1 + k_classes), device=device, dtype=dtype)
            vals[:, 0] = -1.0  # coeff on slack v
            j_eq_t = (y_true_b == j)
            # coeff on y_j (column 1 + j): 1 if j != t_b, else 0
            vals[:, 1 + j] = torch.where(
                j_eq_t, torch.zeros((N,), device=device, dtype=dtype),
                torch.ones((N,), device=device, dtype=dtype),
            )
            # coeff on y_{t_b} (column 1 + t_b): -1 if j != t_b, else 0
            t_col = 1 + y_true_b
            # Use scatter to set per-row column. j_eq_t is shape [N].
            t_col_val = torch.where(
                j_eq_t, torch.zeros((N,), device=device, dtype=dtype),
                torch.full((N,), -1.0, device=device, dtype=dtype),
            )
            vals.scatter_(1, t_col.unsqueeze(1), t_col_val.unsqueeze(1))
            rhs = torch.zeros((N,), device=device, dtype=dtype)
            le.add(row_vars, vals, rhs)
        # Standalone v >= 0  ⇒ -v <= 0
        vals_v = torch.full((N, 1), -1.0, device=device, dtype=dtype)
        rhs_v = torch.zeros((N,), device=device, dtype=dtype)
        le.add([slack_id], vals_v, rhs_v)
        return

    if kind == OutKind.MARGIN_ROBUST:
        # For i != t: -y_t + y_i <= -margin + eps.
        # We emit k_classes rows per instance (mirroring TOP1's row count for
        # easy reasoning); the j == t_b row degenerates to 0 <= -margin + eps
        # (trivially satisfied when margin > 0, or strict when margin <= 0).
        k_classes = n_out
        y_true_b = per_batch(assert_layer.params["y_true"], torch.long, "y_true")
        margin_b = per_batch(assert_layer.params["margin"], dtype, "margin")
        for j in range(k_classes):
            row_vars = list(out_ids)
            vals = torch.zeros((N, k_classes), device=device, dtype=dtype)
            j_eq_t = (y_true_b == j)
            # coeff on y_j (col j): 1 if j != t_b else 0
            vals[:, j] = torch.where(
                j_eq_t, torch.zeros((N,), device=device, dtype=dtype),
                torch.ones((N,), device=device, dtype=dtype),
            )
            # coeff on y_{t_b}: -1 if j != t_b else 0
            t_col_val = torch.where(
                j_eq_t, torch.zeros((N,), device=device, dtype=dtype),
                torch.full((N,), -1.0, device=device, dtype=dtype),
            )
            vals.scatter_(1, y_true_b.unsqueeze(1), t_col_val.unsqueeze(1))
            # rhs: margin - eps for normal rows; for the degenerate j == t_b
            # row the coefficients are all zero so we need rhs >= 0 to keep
            # "0 <= rhs" trivially true. Pick +1.0 (any positive constant
            # works; we want a uniform finite value that won't be silently
            # treated as +inf).
            rhs_normal = margin_b - _ASSERT_EPS
            rhs_trivial = torch.full((N,), 1.0, device=device, dtype=dtype)
            rhs = torch.where(j_eq_t, rhs_trivial, rhs_normal)
            le.add(row_vars, vals, rhs)
        return

    if kind == OutKind.RANGE:
        # Slack v at slot nvars_net. 2K rows + 2 slack rows = 2K+2 rows total.
        # Lower-side: for each i: v + y_i >= lb[i]  ⇒  -v - y_i <= -lb[i]
        # Upper-side: for each i: v - y_i >= -ub[i] ⇒  -v + y_i <= ub[i]
        # Slack: v >= 0  ⇒  -v <= 0
        #        v <= v_max  ⇒  v <= v_max
        if nvars_total != nvars_net + 1:
            raise ValueError(
                f"RANGE requires 1 slack var; nvars_total={nvars_total} "
                f"vs nvars_net={nvars_net}"
            )
        slack_id = nvars_net
        lb_raw = assert_layer.params.get("lb")
        ub_raw = assert_layer.params.get("ub")
        # Missing bounds become trivial rows to preserve a uniform structure.
        if lb_raw is not None:
            lb_b = _coerce_b_tensor(lb_raw, N, n_out, device, dtype, "lb")
        else:
            lb_b = torch.full(
                (N, n_out), -float("inf"), device=device, dtype=dtype
            )
        if ub_raw is not None:
            ub_b = _coerce_b_tensor(ub_raw, N, n_out, device, dtype, "ub")
        else:
            ub_b = torch.full(
                (N, n_out), float("inf"), device=device, dtype=dtype
            )
        for i in range(n_out):
            # Lower side: -v - y_i <= -lb[i]
            vals_l = torch.tensor(
                [[-1.0, -1.0]], device=device, dtype=dtype
            ).expand(N, -1).contiguous()
            rhs_l = -lb_b[:, i]
            le.add([slack_id, out_ids[i]], vals_l, rhs_l)
            # Upper side: -v + y_i <= ub[i]
            vals_u = torch.tensor(
                [[-1.0, 1.0]], device=device, dtype=dtype
            ).expand(N, -1).contiguous()
            rhs_u = ub_b[:, i].clone()
            le.add([slack_id, out_ids[i]], vals_u, rhs_u)
        # Slack v >= 0
        vals_v0 = torch.full((N, 1), -1.0, device=device, dtype=dtype)
        rhs_v0 = torch.zeros((N,), device=device, dtype=dtype)
        le.add([slack_id], vals_v0, rhs_v0)
        # Slack v <= v_max
        vals_v1 = torch.full((N, 1), 1.0, device=device, dtype=dtype)
        rhs_v1 = torch.full(
            (N,), _RANGE_SLACK_CAP, device=device, dtype=dtype
        )
        le.add([slack_id], vals_v1, rhs_v1)
        return

    raise NotImplementedError(
        f"_emit_assert_canonical: unsupported ASSERT kind {kind!r}"
    )


# -----------------------------------------------------------------------------
# Public entry: export_to_batch_problem
# -----------------------------------------------------------------------------


def export_to_batch_problem(
    net: Any,
    globalC: ConSet,
    assert_layer: Any,
    input_box_per_b: Bounds,
    *,
    objective: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> BatchLPProblem:
    """[BATCHED-API] Build a BatchLPProblem from analyze() output.

    All ``N`` problem instances share the variable-id schema; only the
    coefficient and bound TENSORS vary along the leading batch dim. Per-tag
    handlers emit canonical rows (uniform row count per layer across N) so the
    block-diagonal sparse A_eq / A_le have fixed block shape.

    ASSERT encoding is emitted directly from batched tensors. Slack variables
    for TOP1/RANGE live at
    ``nvars_net`` (one shared variable id across N).

    Args:
        net: ACT ``Net`` used to discover input variable ids; only consulted
            for ``find_entry_layer_id`` (the constraint templates already
            carry all variable references).
        globalC: ConSet produced by ``analyze(net, ...)``; box templates carry
            the propagated bounds.
        assert_layer: the final ASSERT layer with pre-encoded params from
            ``OutputSpec.encode_linear``.
        input_box_per_b: ``Bounds`` whose ``lb``/``ub`` are
            ``[N, *input_shape]``; the FIRST dimension determines ``N``.
        objective: optional ``(obj_c, obj_const)`` pair where ``obj_c`` is
            ``[N, nvars_total]`` and ``obj_const`` is ``[N]``. Defaults to
            feasibility (zeros). Sense is hardcoded to ``"min"`` per design;
            verifications run as pure feasibility checks.

    Returns:
        A ``BatchLPProblem`` ready for ``Solver.solve_batch``.
    """
    if input_box_per_b.lb.dim() < 2:
        raise ValueError(
            f"export_to_batch_problem: input_box_per_b must be batched "
            f"[N, *input_shape]; got dim={input_box_per_b.lb.dim()} "
            f"shape={tuple(input_box_per_b.lb.shape)}"
        )
    N = int(input_box_per_b.lb.shape[0])
    device = input_box_per_b.lb.device
    dtype = input_box_per_b.lb.dtype

    validate_conset_ops(globalC)

    templates = list(globalC)

    # ---- Pass 1: variable-id discovery ----
    all_ids: set[int] = set()
    for con in templates:
        all_ids.update(con.var_ids)
    nvars_net = (max(all_ids) + 1) if all_ids else 0

    # ---- Slack allocation per ASSERT kind ----
    from act.front_end.specs import OutKind
    kind = assert_layer.params.get("kind")
    needs_slack = kind in (OutKind.TOP1_ROBUST, OutKind.RANGE)
    nvars_total = nvars_net + (1 if needs_slack else 0)

    # ---- Pass 2: build [N, nvars_total] box tensors ----
    lb_global, ub_global = _build_box_tensors_batched(
        templates, nvars_total, N, device, dtype,
    )

    # Intersect with the per-N input-box override.
    input_layer_ids = [layer.id for layer in net.layers if layer.kind == "INPUT"]
    if len(input_layer_ids) != 1:
        raise ValueError(f"Expected exactly one INPUT layer, found {len(input_layer_ids)}.")
    entry_id = input_layer_ids[0]
    input_ids = list(net.by_id[entry_id].out_vars)
    flat_lb = input_box_per_b.lb.reshape(N, -1).to(device=device, dtype=dtype)
    flat_ub = input_box_per_b.ub.reshape(N, -1).to(device=device, dtype=dtype)
    if flat_lb.shape[1] != len(input_ids):
        raise ValueError(
            f"export_to_batch_problem: input_box_per_b has "
            f"{flat_lb.shape[1]} input dims but entry layer expects "
            f"{len(input_ids)}"
        )
    in_idx = torch.tensor(input_ids, device=device, dtype=torch.long)
    lb_global[:, in_idx] = torch.maximum(lb_global[:, in_idx], flat_lb)
    ub_global[:, in_idx] = torch.minimum(ub_global[:, in_idx], flat_ub)

    # Slack-var bounds (per-N value via the shared id).
    if needs_slack:
        # All ASSERT kinds using slack expect v >= 0; RANGE also clamps
        # v <= v_max via an explicit row (we keep ub at +inf here and let
        # the inline row enforce the upper cap).
        lb_global[:, nvars_net] = 0.0

    # ---- Pass 3: per-tag emission ----
    eq = _RowAcc(N, device, dtype)
    le = _RowAcc(N, device, dtype)

    def linear(emit: Callable[..., None]) -> Callable[[Any], None]:
        """Bind an equality emitter ``emit(con, eq, N, device, dtype)``."""
        return lambda con: emit(con, eq, N, device, dtype)

    def relaxed(emit: Callable[..., None]) -> Callable[[Any], None]:
        """Bind a relaxation emitter ``emit(con, le, lb_global, ub_global, N, device, dtype)``."""
        return lambda con: emit(con, le, lb_global, ub_global, N, device, dtype)

    # Unconditional tag prefixes; skipped, conditional and rejected tags stay explicit below.
    rules: dict[str, Callable[[Any], None]] = {
        "dense:": linear(_emit_dense),
        "conv2d:": linear(_emit_conv2d),
        "bias:": linear(_emit_bias),
        "scale:": linear(_emit_scale),
        "bn:": linear(_emit_bn),
        "add:": lambda con: _emit_add_sub(con, eq, +1.0, N, device, dtype),
        "sub:": lambda con: _emit_add_sub(con, eq, -1.0, N, device, dtype),
        "flatten:": linear(_emit_flatten),
        "reshape:": linear(_emit_flatten),
        "transpose:": linear(_emit_transpose),
        "relu:": relaxed(_emit_relu_canonical),
        "lrelu:": relaxed(_emit_lrelu_canonical),
        "relu6:": lambda con: _emit_clipped_affine_hull(
            con, le, lb_global, ub_global, 1.0, 0.0, 0.0, 6.0, N, device, dtype),
        "tanh:": relaxed(_emit_tanh_canonical),
        "sigmoid:": relaxed(_emit_sigmoid_canonical),
        "erf:": relaxed(_emit_erf_canonical),
        "sqrt:": relaxed(_emit_sqrt_canonical),
        "sin:": lambda con: _emit_periodic_box_canonical(
            con, le, lb_global, ub_global, N, device, dtype, "sin"),
        "cos:": lambda con: _emit_periodic_box_canonical(
            con, le, lb_global, ub_global, N, device, dtype, "cos"),
        "quantize:": relaxed(_emit_quantize_canonical),
        "hardtanh:": lambda con: _emit_clipped_affine_hull(
            con, le, lb_global, ub_global, 1.0, 0.0,
            float(con.meta.get("min_val", -1.0)), float(con.meta.get("max_val", 1.0)), N, device, dtype),
        "hardsigmoid:": lambda con: _emit_clipped_affine_hull(
            con, le, lb_global, ub_global, float(con.meta.get("alpha", 1.0 / 6.0)),
            float(con.meta.get("beta", 0.5)), 0.0, 1.0, N, device, dtype),
        "abs:": relaxed(_emit_abs_canonical),
        "square:": relaxed(_emit_square_canonical),
        "power:": relaxed(_emit_relu_power_canonical),
        "layernorm:": relaxed(_emit_layernorm_box),
        "att_dual_planar:": relaxed(_emit_attn_dual_planar),
        "mask:": lambda con: _emit_mask_add(con, eq, lb_global, ub_global, N, device, dtype),
        "slice:": lambda con: _emit_slice_gather(con, eq, is_gather=False, N=N, device=device, dtype=dtype),
        "gather:": lambda con: _emit_slice_gather(con, eq, is_gather=True, N=N, device=device, dtype=dtype),
        "concat:": linear(_emit_concat),
        "expand:": linear(_emit_expand),
        "constant:": linear(_emit_constant),
        "reduce_sum:": lambda con: _emit_reduction(con, eq, False, N, device, dtype),
        "mean:": lambda con: _emit_reduction(con, eq, True, N, device, dtype),
    }

    for con in templates:
        tag = str(con.meta.get("tag", ""))
        head, colon, _ = tag.partition(":")
        emit = rules.get(head + colon)
        if emit is not None:
            emit(con)
        elif tag.startswith(("box:", "att_scores:", "att_mix:")):
            continue
        elif tag.startswith("mcc:"):
            if len(con.var_ids) % 3 != 0:
                continue
            _emit_mcc(con, le, lb_global, ub_global, N, device, dtype)
        elif tag.startswith("softmax:simplex:"):
            _emit_softmax_simplex(con, eq, le, N, device, dtype)
        elif tag == "in:linpoly":
            _emit_in_linpoly(con, le, N, device, dtype)
        elif tag.startswith(("max:", "min:", "div:", "clip:")):
            raise NotImplementedError(
                f"export_to_batch_problem: tag {tag!r} not implemented"
            )
        else:
            raise NotImplementedError(
                f"export_to_batch_problem: unsupported tag {tag!r}"
            )

    out_ids = list(assert_layer.in_vars)
    _emit_assert_canonical(
        assert_layer, out_ids, le,
        N=N, nvars_net=nvars_net, nvars_total=nvars_total,
        lb_global=lb_global, ub_global=ub_global,
        device=device, dtype=dtype,
    )

    # ---- Pass 4: build sparse tensors ----
    A_eq, b_eq = eq.build_sparse(nvars_total)
    A_le, b_le = le.build_sparse(nvars_total)

    # ---- Pass 5: objective ----
    if objective is None:
        obj_c = torch.zeros((N, nvars_total), device=device, dtype=dtype)
        obj_const = torch.zeros((N,), device=device, dtype=dtype)
    else:
        obj_c_raw, obj_const_raw = objective
        obj_c = obj_c_raw.to(device=device, dtype=dtype)
        obj_const = obj_const_raw.to(device=device, dtype=dtype)
        if obj_c.shape != (N, nvars_total):
            raise ValueError(
                f"objective[0] shape {tuple(obj_c.shape)} != "
                f"({N}, {nvars_total})"
            )
        if obj_const.shape != (N,):
            raise ValueError(
                f"objective[1] shape {tuple(obj_const.shape)} != ({N},)"
            )

    return BatchLPProblem(
        nvars=nvars_total,
        m_eq=eq.m(),
        m_le=le.m(),
        lb=lb_global,
        ub=ub_global,
        A_eq_blockdiag=A_eq,
        b_eq=b_eq,
        A_le_blockdiag=A_le,
        b_le=b_le,
        obj_c=obj_c,
        obj_const=obj_const,
    )

# Terminal network export support.
TERMINAL_LAYER_KINDS = frozenset({
    "INPUT", "INPUT_SPEC", "ASSERT", "DENSE", "CONV2D", "BIAS", "SCALE",
    "ADD", "SUB", "FLATTEN", "RESHAPE", "TRANSPOSE", "SLICE", "GATHER", "RELU",
    "BN", "CONSTANT", "CONCAT", "EXPAND", "REDUCE_SUM", "MEAN",
})
_EXACT_TAGS = frozenset({
    "box", "dense", "conv2d", "bias", "scale", "add", "sub", "flatten",
    "reshape", "transpose", "slice", "gather", "relu", "in",
    "bn", "constant", "concat", "expand", "reduce_sum", "mean",
})
_LANE_BASE_CACHE = 64


@dataclass(frozen=True)
class _PhaseRows:
    """Where one literal's sign and phase-equality rows live in its lane LP."""
    layer_id: int
    neuron: int
    sign: int
    pre_var: int
    post_var: int
    le_row: int
    eq_row: int


@dataclass
class TerminalExport:
    """CPU sparse lanes with a Torch representation materialized on demand."""
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    spec_c: torch.Tensor
    spec_d: torch.Tensor
    out_kind: str
    net: Net
    input_bounds: Bounds
    split_signs: dict[int, torch.Tensor]
    relu_upper_rows: tuple[int, ...]
    relu_layers: tuple[Layer, ...]
    _lanes: tuple[Any, ...] = field(repr=False)
    _phase_rows: tuple[tuple[_PhaseRows, ...], ...] = field(repr=False)

    @cached_property
    def problem(self) -> BatchLPProblem:
        """Pack corrected lanes only when a consumer needs BatchLPProblem."""
        from scipy import sparse

        lanes = self._lanes
        n = len(lanes[0].lb)
        me = max(p.eq.shape[0] for p in lanes)
        ml = max(p.le.shape[0] for p in lanes)
        ref = self.spec_c

        def tensor(arrays: list[np.ndarray]) -> torch.Tensor:
            """Move a stacked numeric block to the export device."""
            return torch.as_tensor(np.stack(arrays), device=ref.device, dtype=ref.dtype)

        # Pad the public batched view without mutating the solver's CPU lanes.
        eqs = [sparse.vstack((p.eq, sparse.csr_matrix((me - p.eq.shape[0], n))),
                             format="csr") for p in lanes]
        les = [sparse.vstack((p.le, sparse.csr_matrix((ml - p.le.shape[0], n))),
                             format="csr") for p in lanes]
        return BatchLPProblem(
            n, me, ml, tensor([p.lb for p in lanes]), tensor([p.ub for p in lanes]),
            _torch_blocks(eqs, ref),
            tensor([np.pad(p.eq_rhs, (0, me - p.eq.shape[0])) for p in lanes]),
            _torch_blocks(les, ref),
            tensor([np.pad(p.le_rhs, (0, ml - p.le.shape[0])) for p in lanes]),
            ref.new_zeros((len(lanes), n)), ref.new_zeros(len(lanes)),
        )


def _scipy_blocks(matrix: torch.Tensor, rows: int, cols: int, count: int) -> list[Any]:
    """Convert COO once, then slice sparse lane blocks without densifying."""
    from scipy import sparse
    matrix = matrix.detach().coalesce().cpu()
    indices = matrix.indices().numpy()
    whole = sparse.coo_matrix((matrix.values().double().numpy(), indices),
                              shape=matrix.shape).tocsr()
    return [whole[k * rows:(k + 1) * rows, k * cols:(k + 1) * cols]
            for k in range(count)]


def _torch_blocks(blocks: list[Any], reference: torch.Tensor) -> torch.Tensor:
    """Pack CPU sparse blocks in the existing BatchLPProblem convention."""
    from scipy import sparse
    whole = sparse.block_diag(blocks, format="coo")
    indices = torch.as_tensor(np.stack((whole.row, whole.col)), device=reference.device)
    values = torch.as_tensor(whole.data, device=reference.device, dtype=reference.dtype)
    return torch.sparse_coo_tensor(indices, values, whole.shape).coalesce()


def _derive_boxes(p: _SparseLP, input_ids: tuple[int, ...],
                  relus: tuple[Layer, ...]) -> None:
    """Recover interval boxes from exported equations, with directed rounding.

    Interval TF boxes are only exporter scaffolding, never proof premises.
    It never assumes the nearest-rounded interval TF boxes are sound.
    """
    solver_torchlp = import_module("act.back_end.solver.solver_torchlp")
    _add, _fraction_float, _mul = (
        solver_torchlp._add, solver_torchlp._fraction_float, solver_torchlp._mul,
    )

    known = set(input_ids)
    pending = set(range(p.eq.shape[0]))
    relu_pending = [(x, z) for layer in relus for x, z in zip(layer.in_vars, layer.out_vars)]
    while pending or relu_pending:
        progress = False
        for row in tuple(pending):
            start, end = p.eq.indptr[row:row + 2]
            terms = list(zip(p.eq.indices[start:end], p.eq.data[start:end]))
            unknown = [int(j) for j, a in terms if a != 0 and j not in known]
            if not unknown:
                pending.remove(row)
                continue
            if len(unknown) != 1:
                continue
            pivot = unknown[0]
            a = next(float(a) for j, a in terms if j == pivot)
            lower = upper = float(p.eq_rhs[row])
            for j, v in terms:
                if j == pivot or v == 0:
                    continue
                lower = _add(lower, -max(_mul(v, float(p.lb[j]), True), _mul(v, float(p.ub[j]), True)))
                upper = _add(upper, -min(_mul(v, float(p.lb[j])), _mul(v, float(p.ub[j]))))
            if abs(a) == 1:
                values = (lower / a, upper / a)
                p.lb[pivot], p.ub[pivot] = min(values), max(values)
            else:
                values = (Fraction(lower) / Fraction(a), Fraction(upper) / Fraction(a))
                p.lb[pivot] = _fraction_float(min(values), False)
                p.ub[pivot] = _fraction_float(max(values), True)
            known.add(pivot)
            pending.remove(row)
            progress = True
        for x, z in tuple(relu_pending):
            if x in known:
                p.lb[z], p.ub[z] = max(0.0, p.lb[x]), max(0.0, p.ub[x])
                known.add(z)
                relu_pending.remove((x, z))
                progress = True
        if not progress and (pending or relu_pending):
            raise NotImplementedError("unsupported_nontriangular_affine_equations")
    if len(known) != len(p.lb) or not np.isfinite(p.lb).all() or not np.isfinite(p.ub).all():
        raise NotImplementedError("unsupported_unbounded_variable_schema")


@dataclass(frozen=True)
class _TerminalHeader:
    """A supported model's validated literals and property; no lane LP yet."""
    entry: Layer
    assertion: Layer
    specs: list[Layer]
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    signs: dict[int, torch.Tensor]
    spec_c: torch.Tensor
    spec_d: torch.Tensor
    out_kind: str


def export_terminal_problem(
    net: Net, input_bounds: Bounds, split_signs: dict[int, torch.Tensor] | None = None,
    *, reference_bounds: dict[int, Bounds] | None = None,
    lane_bases: dict[tuple[bytes, bytes], Any] | None = None,
) -> TerminalExport:
    """Export all K lanes once, without trusting supplied intermediate boxes.

    ``reference_bounds`` is accepted for caller symmetry but deliberately not
    used: root replay of a core cannot inherit conditioned lane bounds.
    Signs have shape [K,n] or identical [K,M,n] copies. Unsupported inputs or
    incomplete affine encodings raise ``NotImplementedError`` explicitly.
    The LP lives on the host, so the export is built there too, whatever the
    default device; only candidate validation returns to the input's device.
    ``lane_bases`` caches one split-free LP per input box across calls.
    """
    with torch.device("cpu"):
        return _export_on_host(net, input_bounds, split_signs, lane_bases)


def _terminal_header(
    net: Net, input_bounds: Bounds, split_signs: dict[int, torch.Tensor] | None,
) -> _TerminalHeader:
    """Reject unsupported models and encode the literals and property, without any lane LP."""
    from act.back_end.verifier import gather_input_spec_layers

    if input_bounds.lb.ndim < 2 or input_bounds.lb.shape != input_bounds.ub.shape:
        raise ValueError("terminal input bounds require matching [K,*shape]")
    if not (torch.isfinite(input_bounds.lb).all() and torch.isfinite(input_bounds.ub).all()):
        raise ValueError("nonfinite_input_bounds")
    if bool((input_bounds.lb > input_bounds.ub).any()):
        raise ValueError("inverted_input_bounds")
    unsupported = {layer.kind for layer in net.layers} - TERMINAL_LAYER_KINDS
    if unsupported:
        raise NotImplementedError(f"unsupported_layers:{','.join(sorted(unsupported))}")
    # Concrete execution follows predecessor edges, while the exporter uses
    # variable IDs (and explicit binary operand metadata). Both descriptions
    # must encode the same complete operation before any proof is trusted.
    defined: set[int] = set()
    for lid in get_topo_order(net):
        layer = net.by_id[lid]
        predecessors = net.preds.get(layer.id, [])
        arity = (0 if layer.kind == "INPUT" else 2 if layer.kind in {"ADD", "SUB"}
                 else len(predecessors) if layer.kind == "CONCAT" else 1)
        operands = [net.by_id[lid].out_vars for lid in predecessors]
        # A CONSTANT reads no operand, whatever routing edges its importer left.
        if layer.kind != "CONSTANT" and (
            len(predecessors) != arity
            or layer.in_vars != [variable for operand in operands for variable in operand]
        ):
            raise NotImplementedError(f"unsupported_operand_mapping:{layer.id}")
        if layer.kind in {"ADD", "SUB"} and (
            layer.params.get("x_vars") != operands[0]
            or layer.params.get("y_vars") != operands[1]
            or any(len(operand) != len(layer.out_vars) for operand in operands)
        ):
            raise NotImplementedError(f"unsupported_operand_mapping:{layer.id}")
        # A proof variable denotes one scalar value, not a mutable storage
        # slot. Only coordinate-preserving wrappers may reuse an existing ID.
        identity = (layer.kind in {"INPUT_SPEC", "FLATTEN", "RESHAPE", "ASSERT"}
                    and len(layer.out_vars) == len(layer.in_vars))
        if (len(set(layer.out_vars)) != len(layer.out_vars)
                or any(v in defined and not (identity and v == layer.in_vars[j])
                       for j, v in enumerate(layer.out_vars))):
            raise NotImplementedError(f"unsupported_variable_alias:{layer.id}")
        defined.update(layer.out_vars)
    specs = gather_input_spec_layers(net)
    for layer in specs:
        kind = layer.params["kind"]
        if kind not in {InKind.BOX, InKind.LINF_BALL, InKind.LIN_POLY, InKind.LP_EMBEDDING}:
            raise NotImplementedError(f"unsupported_input:{kind}")
        if kind == InKind.LP_EMBEDDING and (layer.input_geometry is None
                                          or layer.input_geometry.p != math.inf):
            raise NotImplementedError("unsupported_finite_p_input")
    entry, = (layer for layer in net.layers if layer.kind == "INPUT")
    assertion = net.layers[-1]
    if assertion.kind != "ASSERT":
        raise ValueError("terminal network must end in ASSERT")
    k_count = input_bounds.lb.shape[0]
    input_ids, output_ids = tuple(entry.out_vars), tuple(assertion.in_vars)
    signs: dict[int, torch.Tensor] = {}
    for lid, value in (split_signs or {}).items():
        layer = net.by_id[lid]
        if layer.kind != "RELU":
            raise NotImplementedError(f"unsupported_split_layer:{lid}")
        if value.ndim == 3:
            if not bool((value == value[:, :1, :]).all()):
                raise ValueError("spec_rows_have_different_lane_phases")
            value = value[:, 0, :]
        if value.shape != (k_count, len(layer.in_vars)) or not bool(((value == 0) | (value.abs() == 1)).all()):
            raise ValueError("invalid_split_signs")
        signs[lid] = value.to(device=input_bounds.lb.device)
    ref = input_bounds.lb.double().cpu()
    params = assertion.params
    if "C" in params and "thresholds" in params:
        m = int(params["M"])
        c = torch.as_tensor(params["C"], device=ref.device, dtype=ref.dtype).reshape(-1, m, len(output_ids))
        d = torch.as_tensor(params["thresholds"], device=ref.device, dtype=ref.dtype).reshape(-1, m)
        c, d = c.expand(k_count, -1, -1), d.expand(k_count, -1)
    else:
        spec = OutputSpec(kind=params["kind"], **{key: params.get(key)
                          for key in ("c", "d", "y_true", "margin", "lb", "ub")})
        encoded = spec.encode_linear(k_count, len(output_ids), ref.device, ref.dtype)
        c = encoded["C"].reshape(k_count, -1, len(output_ids))
        d = encoded["thresholds"]
    if not torch.isfinite(c).all() or not torch.isfinite(d).all():
        raise ValueError("nonfinite_output_spec")
    return _TerminalHeader(entry, assertion, specs, input_ids, output_ids, signs, c, d, params["kind"])


@dataclass(frozen=True)
class _LaneBase:
    """Split-free exported LP of one input box; literals only append rows to it."""
    lp: Any
    relus: tuple[Layer, ...]
    upper_rows: tuple[int, ...]
    pre_vars: np.ndarray
    post_vars: np.ndarray


def _lane_base(net: Net, head: _TerminalHeader, lower: torch.Tensor, upper: torch.Tensor) -> _LaneBase:
    """Export one ``[1,*shape]`` box, with outward boxes and repaired ReLU upper rows."""
    from act.back_end.interval_tf.interval_tf import IntervalTF
    solver_torchlp = import_module("act.back_end.solver.solver_torchlp")
    _SparseLP, _fraction_float = solver_torchlp._SparseLP, solver_torchlp._fraction_float
    from act.back_end.verifier import add_all_input_specs

    entry, assertion, specs = head.entry, head.assertion, head.specs
    input_ids = head.input_ids
    k_count = 1
    seed = Bounds(lower.double().cpu(), upper.double().cpu())
    before: dict[int, Fact] = {}
    after: dict[int, Fact] = {}
    constraints = ConSet()
    tf = IntervalTF()
    for lid in get_topo_order(net):
        original_layer = net.by_id[lid]
        layer = replace(original_layer, params={
            key: value.to(seed.lb) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
            for key, value in original_layer.params.items()
        }, cache={})
        preds = net.preds.get(lid, [])
        box = after[preds[0]].bounds if preds else Bounds(seed.lb.flatten(1), seed.ub.flatten(1))
        before[lid] = Fact(box, ConSet())
        fact = tf.apply(layer, box, net, before, after)
        after[lid] = fact
        for con in fact.cons:
            if con.meta.get("tag", "").split(":")[0] not in _EXACT_TAGS:
                raise NotImplementedError(f"unsupported_constraint:{con.meta.get('tag')}")
            constraints.replace(con)
    add_all_input_specs(constraints, list(entry.out_vars), specs)
    # Reuse the canonical exporter but remove its deliberately trivial ASSERT
    # row. Property negation belongs below, one ordinary row or one unsafe set.
    dummy_params = OutputSpec(kind=OutKind.LINEAR_LE,
        c=seed.lb.new_zeros(len(assertion.in_vars)), d=seed.lb.new_tensor(-1.0)
    ).encode_linear(k_count, len(assertion.in_vars), seed.lb.device, seed.lb.dtype)
    dummy = Layer(assertion.id, "ASSERT", dummy_params,
                  list(assertion.in_vars), list(assertion.out_vars))
    problem = export_to_batch_problem(net, constraints, dummy, seed)
    n = problem.nvars
    eqs = _scipy_blocks(problem.A_eq_blockdiag, problem.m_eq, n, k_count)
    les = _scipy_blocks(problem.A_le_blockdiag, problem.m_le, n, k_count)
    eq_rhs, le_rhs = problem.b_eq.cpu().numpy(), problem.b_le.cpu().numpy()
    relus = tuple(net.by_id[int(con.meta["tag"].split(":")[1])]
                  for con in constraints if con.meta.get("tag", "").startswith("relu:"))
    # Only RELU and original linear input constraints emit inequality rows.
    upper_rows = tuple(3 * j + 2 for j in range(sum(len(l.out_vars) for l in relus)))
    p = _SparseLP(eqs[0], eq_rhs[0].copy(), les[0][:-1].tocsr(), le_rhs[0, :-1].copy(),
                  np.full(n, -np.inf), np.full(n, np.inf))
    p.lb[list(input_ids)] = problem.lb[0, list(input_ids)].cpu().numpy()
    p.ub[list(input_ids)] = problem.ub[0, list(input_ids)].cpu().numpy()
    _derive_boxes(p, input_ids, relus)
    le, rhs = p.le.tolil(), list(p.le_rhs)
    count = 0
    for layer in relus:
        for x, z in zip(layer.in_vars, layer.out_vars):
            row = upper_rows[count]
            count += 1
            # Replace EVERY upper row. A float slope is safe only when
            # its intercept covers BOTH exact endpoints of the chord.
            le.rows[row], le.data[row] = [z], [1.0]
            rhs[row] = max(0.0, p.ub[x])
            if p.lb[x] < 0 < p.ub[x]:
                lower, upper = Fraction(float(p.lb[x])), Fraction(float(p.ub[x]))
                slope = float(upper / (upper - lower))
                intercept = max(-Fraction(slope) * lower, (1 - Fraction(slope)) * upper)
                le.rows[row], le.data[row] = [z, x], [1.0, -slope]
                rhs[row] = _fraction_float(intercept, True)
    p.le, p.le_rhs = le.tocsr(), np.asarray(rhs)
    pre_vars = np.array([x for layer in relus for x in layer.in_vars], dtype=np.int64)
    post_vars = np.array([z for layer in relus for z in layer.out_vars], dtype=np.int64)
    return _LaneBase(p, relus, upper_rows, pre_vars, post_vars)


def _lane_with_literals(
    base: _LaneBase, literals: np.ndarray, layers: list[int], neurons: list[int],
) -> tuple[_SparseLP, tuple[_PhaseRows, ...]]:
    """Append this lane's fixed-phase equations and sign rows to its base LP."""
    from scipy import sparse
    _SparseLP = import_module("act.back_end.solver.solver_torchlp")._SparseLP

    p = base.lp
    width = len(p.lb)
    lower, upper = p.lb[base.pre_vars], p.ub[base.pre_vars]
    active = (literals > 0) | (lower >= 0)
    fixed = active | (literals < 0) | (upper <= 0)
    fixed_rows = np.cumsum(fixed) - 1
    rows = np.concatenate((fixed_rows[fixed], fixed_rows[active]))
    columns = np.concatenate((base.post_vars[fixed], base.pre_vars[active]))
    values = np.concatenate((np.ones(int(fixed.sum())), -np.ones(int(active.sum()))))
    equations = sparse.csr_matrix((values, (rows, columns)), shape=(int(fixed.sum()), width))
    chosen = np.nonzero(literals)[0]
    signs = sparse.csr_matrix(
        (-literals[chosen], (np.arange(chosen.size), base.pre_vars[chosen])), shape=(chosen.size, width),
    )
    phases = tuple(
        _PhaseRows(layers[j], neurons[j], int(literals[j]), int(base.pre_vars[j]), int(base.post_vars[j]),
                   len(p.le_rhs) + index, p.eq.shape[0] + int(fixed_rows[j]))
        for index, j in enumerate(chosen.tolist())
    )
    return _SparseLP(
        sparse.vstack((p.eq, equations), format="csr"), np.concatenate((p.eq_rhs, np.zeros(equations.shape[0]))),
        sparse.vstack((p.le, signs), format="csr"), np.concatenate((p.le_rhs, np.zeros(chosen.size))),
        p.lb, p.ub,
    ), phases


def _export_on_host(
    net: Net, input_bounds: Bounds, split_signs: dict[int, torch.Tensor] | None,
    lane_bases: dict[tuple[bytes, bytes], _LaneBase] | None = None,
) -> TerminalExport:
    """Build one TerminalExport with host tensors and sparse matrices."""
    head = _terminal_header(net, input_bounds, split_signs)
    input_ids, output_ids, signs = head.input_ids, head.output_ids, head.signs
    k_count = input_bounds.lb.shape[0]
    flat_lower = input_bounds.lb.detach().double().cpu().flatten(1)
    flat_upper = input_bounds.ub.detach().double().cpu().flatten(1)
    bases = {} if lane_bases is None else lane_bases
    lanes: list[_SparseLP] = []
    phase_rows: list[tuple[_PhaseRows, ...]] = []
    base: _LaneBase | None = None
    literal_matrix = None
    for k in range(k_count):
        key = (flat_lower[k].numpy().tobytes(), flat_upper[k].numpy().tobytes())
        base = bases.get(key)
        if base is None:
            if len(bases) >= _LANE_BASE_CACHE:
                bases.clear()
            base = bases[key] = _lane_base(net, head, input_bounds.lb[k:k + 1], input_bounds.ub[k:k + 1])
        if literal_matrix is None:
            # Every base of one network orders the same neurons.
            layers = [layer.id for layer in base.relus for _ in layer.in_vars]
            neurons = [j for layer in base.relus for j in range(len(layer.in_vars))]
            literal_matrix = (
                torch.cat([
                    signs[layer.id].cpu().double() if layer.id in signs
                    else torch.zeros(k_count, len(layer.in_vars), dtype=torch.float64)
                    for layer in base.relus
                ], 1).numpy() if base.relus else np.zeros((k_count, 0))
            )
        lane, lane_phases = _lane_with_literals(base, literal_matrix[k], layers, neurons)
        lanes.append(lane)
        phase_rows.append(lane_phases)
    assert base is not None
    upper_rows, relus = base.upper_rows, base.relus
    return TerminalExport(input_ids, output_ids, head.spec_c, head.spec_d, head.out_kind, net, input_bounds,
                          signs, upper_rows, relus, tuple(lanes), tuple(phase_rows))
