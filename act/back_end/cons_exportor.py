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

import logging

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
from act.back_end.core import Bounds, ConSet
from act.back_end.solver.solver_base import BatchLPProblem
from act.back_end.layer_util import validate_conset_ops

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
        self._rows: List[Tuple[List[int], torch.Tensor, torch.Tensor]] = []

    def add(
        self,
        var_ids: List[int],
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        row_chunks: List[torch.Tensor] = []
        col_chunks: List[torch.Tensor] = []
        val_chunks: List[torch.Tensor] = []
        rhs_chunks: List[torch.Tensor] = []
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
    templates: List[Any],
    nvars_total: int,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # Box meta tensors are stored flat per add_box (ConSet.add_box reshapes
        # to (-1,)); however when bounds are batched [B, n] the underlying
        # reshape collapses to (B*n,) or the upstream
        # pipeline stores [B, n] directly. Normalise:
        if isinstance(lb_raw, torch.Tensor) and lb_raw.numel() == n_vars:
            lb_b = lb_raw.to(device=device, dtype=dtype).reshape(n_vars)
            lb_b = lb_b.unsqueeze(0).expand(N, -1)
        elif isinstance(lb_raw, torch.Tensor) and lb_raw.numel() == N * n_vars:
            lb_b = lb_raw.to(device=device, dtype=dtype).reshape(N, n_vars)
        else:
            raise ValueError(
                f"box: lb has unexpected numel={getattr(lb_raw, 'numel', lambda: '?')()}, "
                f"expected {n_vars} or {N * n_vars} for tag {tag!r}"
            )
        if isinstance(ub_raw, torch.Tensor) and ub_raw.numel() == n_vars:
            ub_b = ub_raw.to(device=device, dtype=dtype).reshape(n_vars)
            ub_b = ub_b.unsqueeze(0).expand(N, -1)
        elif isinstance(ub_raw, torch.Tensor) and ub_raw.numel() == N * n_vars:
            ub_b = ub_raw.to(device=device, dtype=dtype).reshape(N, n_vars)
        else:
            raise ValueError(
                f"box: ub has unexpected numel for tag {tag!r}"
            )
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
    W_t = (
        W.to(device=device, dtype=dtype)
        if isinstance(W, torch.Tensor)
        else torch.as_tensor(W, device=device, dtype=dtype)
    )
    b_t = (
        b.to(device=device, dtype=dtype)
        if isinstance(b, torch.Tensor)
        else torch.as_tensor(b, device=device, dtype=dtype)
    )
    n_out, n_in = W_t.shape
    var_ids_all = list(con.var_ids)
    y = var_ids_all[:n_out]
    x = var_ids_all[n_out:]
    if len(x) != n_in:
        raise ValueError(
            f"dense: var_ids length mismatch: {len(x)} input vars vs "
            f"W.shape[1]={n_in}"
        )
    for i in range(n_out):
        row_vars = [y[i]] + x
        # vals[n, 0] = 1.0; vals[n, 1:] = -W[i, :]
        vals = torch.empty((N, 1 + n_in), device=device, dtype=dtype)
        vals[:, 0] = 1.0
        vals[:, 1:] = -W_t[i].unsqueeze(0).expand(N, -1)
        rhs = b_t[i].expand(N).contiguous()
        eq.add(row_vars, vals, rhs)


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
    c_t = (
        c_raw.to(device=device, dtype=dtype)
        if isinstance(c_raw, torch.Tensor)
        else torch.as_tensor(c_raw, device=device, dtype=dtype)
    ).reshape(-1)
    if c_t.numel() != n:
        raise ValueError(f"bias: c length {c_t.numel()} != n {n}")
    for i in range(n):
        vals = torch.tensor(
            [[1.0, -1.0]], device=device, dtype=dtype
        ).expand(N, -1).contiguous()
        rhs = c_t[i].expand(N).contiguous()
        eq.add([y[i], x[i]], vals, rhs)


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
    a_t = (
        a_raw.to(device=device, dtype=dtype)
        if isinstance(a_raw, torch.Tensor)
        else torch.as_tensor(a_raw, device=device, dtype=dtype)
    ).reshape(-1)
    if a_t.numel() != n:
        raise ValueError(f"scale: a length {a_t.numel()} != n {n}")
    for i in range(n):
        vals = torch.empty((N, 2), device=device, dtype=dtype)
        vals[:, 0] = 1.0
        vals[:, 1] = -a_t[i]
        rhs = torch.zeros((N,), device=device, dtype=dtype)
        eq.add([y[i], x[i]], vals, rhs)


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
    A_t = (
        A_raw.to(device=device, dtype=dtype)
        if isinstance(A_raw, torch.Tensor)
        else torch.as_tensor(A_raw, device=device, dtype=dtype)
    ).reshape(-1)
    c_t = (
        c_raw.to(device=device, dtype=dtype)
        if isinstance(c_raw, torch.Tensor)
        else torch.as_tensor(c_raw, device=device, dtype=dtype)
    ).reshape(-1)
    if A_t.numel() != n or c_t.numel() != n:
        raise ValueError(
            f"bn: A.numel={A_t.numel()} c.numel={c_t.numel()} != n {n}"
        )
    for i in range(n):
        vals = torch.empty((N, 2), device=device, dtype=dtype)
        vals[:, 0] = 1.0
        vals[:, 1] = -A_t[i]
        rhs = c_t[i].expand(N).contiguous()
        eq.add([y[i], x[i]], vals, rhs)


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
    for i in range(n):
        vals = torch.tensor(
            [[1.0, -1.0, -sign_y]], device=device, dtype=dtype
        ).expand(N, -1).contiguous()
        rhs = torch.zeros((N,), device=device, dtype=dtype)
        eq.add([z[i], x[i], y[i]], vals, rhs)


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
    vals = torch.tensor(
        [[1.0, -1.0]], device=device, dtype=dtype
    ).expand(N, -1).contiguous()
    rhs = torch.zeros((N,), device=device, dtype=dtype)
    for i in range(n_out):
        eq.add([y[i], x[i]], vals, rhs)


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

    W_t = (
        weight_raw.to(device=device, dtype=dtype)
        if isinstance(weight_raw, torch.Tensor)
        else torch.as_tensor(weight_raw, device=device, dtype=dtype)
    )
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

    b_t = (
        bias_raw.to(device=device, dtype=dtype)
        if isinstance(bias_raw, torch.Tensor)
        else torch.as_tensor(bias_raw, device=device, dtype=dtype)
    ).reshape(-1)
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
    # Emit per-neuron 3 ineq rows.
    for i in range(n):
        # Row (a): -z_i <= 0
        vals_a = torch.full((N, 1), -1.0, device=device, dtype=dtype)
        rhs_a = torch.zeros((N,), device=device, dtype=dtype)
        le.add([z[i]], vals_a, rhs_a)
        # Row (b): y_i - z_i <= 0
        vals_b = torch.tensor(
            [[1.0, -1.0]], device=device, dtype=dtype
        ).expand(N, -1).contiguous()
        rhs_b = torch.zeros((N,), device=device, dtype=dtype)
        le.add([y[i], z[i]], vals_b, rhs_b)
        # Row (c): z_i - slope_i * y_i <= shift_i
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
    cross_zero = ~(identity_band | point_bounds | convex_seg | concave_seg)
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
    for i in range(n):
        # (a) z >= tanh(lo)  ⇒  -z <= -f_lo
        vals_a = torch.full((N, 1), -1.0, device=device, dtype=dtype)
        rhs_a = -f_lo[:, i]
        le.add([z[i]], vals_a, rhs_a)
        # (b) z <= tanh(hi)
        vals_b = torch.full((N, 1), 1.0, device=device, dtype=dtype)
        rhs_b = f_hi[:, i]
        le.add([z[i]], vals_b, rhs_b)
        # (c) z >= slope_lo*y + b_lo  ⇒  -z + slope_lo*y <= -b_lo
        vals_c = torch.empty((N, 2), device=device, dtype=dtype)
        vals_c[:, 0] = -1.0
        vals_c[:, 1] = slope_lo[:, i]
        rhs_c = -b_lo[:, i]
        le.add([z[i], y[i]], vals_c, rhs_c)
        # (d) z <= slope_hi*y + b_hi  ⇒  z - slope_hi*y <= b_hi
        vals_d = torch.empty((N, 2), device=device, dtype=dtype)
        vals_d[:, 0] = 1.0
        vals_d[:, 1] = -slope_hi[:, i]
        rhs_d = b_hi[:, i]
        le.add([z[i], y[i]], vals_d, rhs_d)


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
    f_lo = torch.sigmoid(lo_s)
    f_hi = torch.sigmoid(hi_s)
    df_lo = f_lo * (1.0 - f_lo)
    df_hi = f_hi * (1.0 - f_hi)
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
    f_mid = torch.sigmoid(mid)
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
    for i in range(n):
        vals_a = torch.full((N, 1), -1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_a, -f_lo[:, i])
        vals_b = torch.full((N, 1), 1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_b, f_hi[:, i])
        vals_c = torch.empty((N, 2), device=device, dtype=dtype)
        vals_c[:, 0] = -1.0
        vals_c[:, 1] = slope_lo[:, i]
        le.add([z[i], y[i]], vals_c, -b_lo[:, i])
        vals_d = torch.empty((N, 2), device=device, dtype=dtype)
        vals_d[:, 0] = 1.0
        vals_d[:, 1] = -slope_hi[:, i]
        le.add([z[i], y[i]], vals_d, b_hi[:, i])


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
    f_lo = torch.erf(lo_s)
    f_hi = torch.erf(hi_s)
    df_lo = (2.0 / float(np.sqrt(np.pi))) * torch.exp(-lo_s * lo_s)
    df_hi = (2.0 / float(np.sqrt(np.pi))) * torch.exp(-hi_s * hi_s)
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
    f_mid = torch.erf(mid)
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
    for i in range(n):
        vals_a = torch.full((N, 1), -1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_a, -f_lo[:, i])
        vals_b = torch.full((N, 1), 1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_b, f_hi[:, i])
        vals_c = torch.empty((N, 2), device=device, dtype=dtype)
        vals_c[:, 0] = -1.0
        vals_c[:, 1] = slope_lo[:, i]
        le.add([z[i], y[i]], vals_c, -b_lo[:, i])
        vals_d = torch.empty((N, 2), device=device, dtype=dtype)
        vals_d[:, 0] = 1.0
        vals_d[:, 1] = -slope_hi[:, i]
        le.add([z[i], y[i]], vals_d, b_hi[:, i])



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
    for i in range(n):
        vals_a = torch.full((N, 1), -1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_a, -f_lo[:, i])
        vals_b = torch.full((N, 1), 1.0, device=device, dtype=dtype)
        le.add([z[i]], vals_b, f_hi[:, i])
        vals_c = torch.empty((N, 2), device=device, dtype=dtype)
        vals_c[:, 0] = -1.0
        vals_c[:, 1] = slope_lo[:, i]
        le.add([z[i], y[i]], vals_c, -b_lo[:, i])
        vals_d = torch.empty((N, 2), device=device, dtype=dtype)
        vals_d[:, 0] = 1.0
        vals_d[:, 1] = -slope_hi[:, i]
        le.add([z[i], y[i]], vals_d, b_hi[:, i])


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


def _emit_sin_canonical(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    _emit_periodic_box_canonical(con, le, lb_global, ub_global, N, device, dtype, "sin")


def _emit_cos_canonical(
    con: Any,
    le: _RowAcc,
    lb_global: torch.Tensor,
    ub_global: torch.Tensor,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    _emit_periodic_box_canonical(con, le, lb_global, ub_global, N, device, dtype, "cos")



def _quantize_params_flat(con: Any, n: int, device: torch.device, dtype: torch.dtype):
    scale = con.meta["scale"].to(device=device, dtype=dtype).flatten()
    zero_point = con.meta["zero_point"].to(device=device, dtype=dtype).flatten()
    if scale.numel() == 1:
        scale = scale.expand(n)
    elif scale.numel() != n:
        raise NotImplementedError(f"quantize: scale with {scale.numel()} entries cannot broadcast to flat size {n}")
    if zero_point.numel() == 1:
        zero_point = zero_point.expand(n)
    elif zero_point.numel() != n:
        raise NotImplementedError(f"quantize: zero_point with {zero_point.numel()} entries cannot broadcast to flat size {n}")
    if torch.any(scale <= 0):
        raise ValueError("quantize: scale must be positive")
    qmin = torch.full((n,), float(con.meta["qmin"]), device=device, dtype=dtype)
    qmax = torch.full((n,), float(con.meta["qmax"]), device=device, dtype=dtype)
    return scale, zero_point, qmin, qmax


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
    scale_1d, zp_1d, qmin_1d, qmax_1d = _quantize_params_flat(con, n, device, dtype)
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
    A_t = (
        A_raw.to(device=device, dtype=dtype)
        if isinstance(A_raw, torch.Tensor)
        else torch.as_tensor(A_raw, device=device, dtype=dtype)
    )
    b_t = (
        b_raw.to(device=device, dtype=dtype)
        if isinstance(b_raw, torch.Tensor)
        else torch.as_tensor(b_raw, device=device, dtype=dtype)
    )
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
    out_ids: List[int],
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
        d_b = (
            d_raw.to(device=device, dtype=dtype).reshape(-1)
            if isinstance(d_raw, torch.Tensor)
            else torch.as_tensor(d_raw, device=device, dtype=dtype).reshape(-1)
        )
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
        C_t = (
            C_raw.to(device=device, dtype=dtype)
            if isinstance(C_raw, torch.Tensor)
            else torch.as_tensor(C_raw, device=device, dtype=dtype)
        )
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
        d_t = (
            d_raw.to(device=device, dtype=dtype)
            if isinstance(d_raw, torch.Tensor)
            else torch.as_tensor(d_raw, device=device, dtype=dtype)
        )
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
        y_true_raw = assert_layer.params["y_true"]
        y_true_b = (
            y_true_raw.to(device=device, dtype=torch.long).reshape(-1)
            if isinstance(y_true_raw, torch.Tensor)
            else torch.tensor(
                [int(y_true_raw)], device=device, dtype=torch.long
            )
        )
        if y_true_b.numel() == 1 and N > 1:
            y_true_b = y_true_b.expand(N)
        if y_true_b.numel() != N:
            raise ValueError(
                f"TOP1_ROBUST: y_true length {y_true_b.numel()} != N {N}"
            )
        k_classes = n_out
        # For each j, vars referenced are [slack, y_j, y_{t_b}]. y_{t_b}
        # differs per instance, so we emit per-j rows where vars include
        # y_{t_b} as the third var, but indexed per N.
        for j in range(k_classes):
            # We need to model "v - y_j + y_{t_b} <= 0" per instance. The
            # third var index differs per instance ⇒ structure isn't uniform
            # across N for j != t. Workaround: when j == t_b, the row reduces
            # to v <= 0... wait actually y_j == y_{t_b} so it cancels to v <= 0.
            # For j != t_b, the row uses two distinct y vars per instance.
            # We re-encode by listing ALL K output vars on each row with
            # per-N coefficients, plus the slack.
            #   vals[n] on y_k = (+1 if k == j else 0) - (+1 if k == t_b[n] else 0)
            #   ⇒ for k=j: vals[n] = 1 - delta_{j, t_b[n]}
            #   ⇒ for k=t_b[n]: vals[n] = -1 (if k != j) or 0 (if k == j)
            #   ⇒ for k != j, k != t_b[n]: 0
            # vals on slack v: -1.0
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
        # Also tighten the slack's lower bound in lb_global (helps Adam solver
        # initialise well). Upper bound stays +inf.
        return

    if kind == OutKind.MARGIN_ROBUST:
        # For i != t: -y_t + y_i <= -margin + eps.
        # We emit k_classes rows per instance (mirroring TOP1's row count for
        # easy reasoning); the j == t_b row degenerates to 0 <= -margin + eps
        # (trivially satisfied when margin > 0, or strict when margin <= 0).
        k_classes = n_out
        y_true_raw = assert_layer.params["y_true"]
        y_true_b = (
            y_true_raw.to(device=device, dtype=torch.long).reshape(-1)
            if isinstance(y_true_raw, torch.Tensor)
            else torch.tensor(
                [int(y_true_raw)], device=device, dtype=torch.long
            )
        )
        if y_true_b.numel() == 1 and N > 1:
            y_true_b = y_true_b.expand(N)
        if y_true_b.numel() != N:
            raise ValueError(
                f"MARGIN_ROBUST: y_true length {y_true_b.numel()} != N {N}"
            )
        margin_raw = assert_layer.params["margin"]
        margin_b = (
            margin_raw.to(device=device, dtype=dtype).reshape(-1)
            if isinstance(margin_raw, torch.Tensor)
            else torch.tensor([float(margin_raw)], device=device, dtype=dtype)
        )
        if margin_b.numel() == 1 and N > 1:
            margin_b = margin_b.expand(N)
        if margin_b.numel() != N:
            raise ValueError(
                f"MARGIN_ROBUST: margin length {margin_b.numel()} != N {N}"
            )
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
        # When one side is missing, materialise both sides with sentinel infinity; for
        # uniform structure we materialise both sides with sentinel infinity
        # so the missing side becomes a trivial constraint.
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
    objective: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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

    for con in templates:
        tag = str(con.meta.get("tag", ""))
        if tag.startswith("box:"):
            continue
        if tag.startswith("dense:"):
            _emit_dense(con, eq, N, device, dtype)
        elif tag.startswith("conv2d:"):
            _emit_conv2d(con, eq, N, device, dtype)
        elif tag.startswith("bias:"):
            _emit_bias(con, eq, N, device, dtype)
        elif tag.startswith("scale:"):
            _emit_scale(con, eq, N, device, dtype)
        elif tag.startswith("bn:"):
            _emit_bn(con, eq, N, device, dtype)
        elif tag.startswith("add:"):
            _emit_add_sub(con, eq, +1.0, N, device, dtype)
        elif tag.startswith("sub:"):
            _emit_add_sub(con, eq, -1.0, N, device, dtype)
        elif tag.startswith(("flatten:", "reshape:", "transpose:")):
            _emit_flatten(con, eq, N, device, dtype)
        elif tag.startswith("relu:"):
            _emit_relu_canonical(con, le, lb_global, ub_global, N, device, dtype)
        elif tag.startswith("lrelu:"):
            _emit_lrelu_canonical(con, le, lb_global, ub_global, N, device, dtype)
        elif tag.startswith("relu6:"):
            _emit_clipped_affine_hull(con, le, lb_global, ub_global, 1.0, 0.0, 0.0, 6.0, N, device, dtype)
        elif tag.startswith("tanh:"):
            _emit_tanh_canonical(con, le, lb_global, ub_global, N, device, dtype)
        elif tag.startswith("sigmoid:"):
            _emit_sigmoid_canonical(con, le, lb_global, ub_global, N, device, dtype)
        elif tag.startswith("erf:"):
            _emit_erf_canonical(con, le, lb_global, ub_global, N, device, dtype)
        elif tag.startswith("sqrt:"):
            _emit_sqrt_canonical(con, le, lb_global, ub_global, N, device, dtype)
        elif tag.startswith("sin:"):
            _emit_sin_canonical(con, le, lb_global, ub_global, N, device, dtype)
        elif tag.startswith("cos:"):
            _emit_cos_canonical(con, le, lb_global, ub_global, N, device, dtype)
        elif tag.startswith("quantize:"):
            _emit_quantize_canonical(con, le, lb_global, ub_global, N, device, dtype)
        elif tag.startswith("hardtanh:"):
            _emit_clipped_affine_hull(
                con, le, lb_global, ub_global, 1.0, 0.0,
                float(con.meta.get("min_val", -1.0)),
                float(con.meta.get("max_val", 1.0)),
                N, device, dtype,
            )
        elif tag.startswith("hardsigmoid:"):
            _emit_clipped_affine_hull(
                con, le, lb_global, ub_global,
                float(con.meta.get("alpha", 1.0 / 6.0)),
                float(con.meta.get("beta", 0.5)),
                0.0, 1.0, N, device, dtype,
            )
        elif tag.startswith("abs:"):
            _emit_abs_canonical(con, le, lb_global, ub_global, N, device, dtype)
        elif tag.startswith("square:"):
            _emit_square_canonical(con, le, lb_global, ub_global, N, device, dtype)
        elif tag.startswith("power:"):
            _emit_relu_power_canonical(con, le, lb_global, ub_global, N, device, dtype)
        elif tag.startswith("layernorm:"):
            _emit_layernorm_box(con, le, lb_global, ub_global, N, device, dtype)
        elif tag.startswith("att_dual_planar:"):
            _emit_attn_dual_planar(con, le, lb_global, ub_global, N, device, dtype)
        elif tag.startswith("mask:"):
            _emit_mask_add(con, eq, lb_global, ub_global, N, device, dtype)
        elif tag.startswith(("att_scores:", "att_mix:")):
            continue
        elif tag.startswith("mcc:"):
            if len(con.var_ids) % 3 != 0:
                continue
            _emit_mcc(con, le, lb_global, ub_global, N, device, dtype)
        elif tag.startswith("softmax:simplex:"):
            _emit_softmax_simplex(con, eq, le, N, device, dtype)
        elif tag.startswith("slice:"):
            _emit_slice_gather(con, eq, is_gather=False, N=N, device=device, dtype=dtype)
        elif tag.startswith("gather:"):
            _emit_slice_gather(con, eq, is_gather=True, N=N, device=device, dtype=dtype)
        elif tag == "in:linpoly":
            _emit_in_linpoly(con, le, N, device, dtype)
        elif tag.startswith(("max:", "min:", "div:", "clip:")):
            # Currently unsupported by the canonical batched form; fail loud
            # rather than silently produce a wrong LP.
            raise NotImplementedError(
                f"export_to_batch_problem: tag {tag!r} not implemented"
            )
        else:
            # Unknown tag: fail loud per directive (A).
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


# =============================================================================
# Inline test battery for the batched exporter. Run via:
#   python -m act.back_end.cons_exportor
# =============================================================================


def _dense_block_rows(A_blockdiag: torch.Tensor, N: int, m: int, nvars: int):
    """Return ``[N, m, nvars]`` dense view of a block-diagonal sparse matrix.

    Used for test assertions only; production code must NEVER call this on
    large matrices.
    """
    A_dense = A_blockdiag.to_dense()
    rows = A_dense.view(N, m, N, nvars)
    eye = torch.arange(N)
    out = rows[eye, :, eye, :]
    return out


def _build_relu_test_net(B: int, n: int, lb: torch.Tensor, ub: torch.Tensor):  # pragma: no cover
    """Net = INPUT -> INPUT_SPEC (BOX [B,n]) -> RELU -> ASSERT(LINEAR_LE).

    The ASSERT layer is required only because export_to_batch_problem
    consults it; it does not affect the RELU encoding under test.
    """
    from act.back_end.core import Layer, Net
    from act.back_end.layer_schema import LayerKind
    from act.front_end.specs import OutputSpec, OutKind

    device = lb.device
    dtype = lb.dtype
    in_v = list(range(n))
    out_v = list(range(n, 2 * n))
    spec_layer = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.zeros(n, device=device, dtype=dtype),
        d=torch.tensor(1.0, device=device, dtype=dtype),
    ).encode_linear(B=B, n_out=n, device=device, dtype=dtype)
    layers = [
        Layer(
            id=0, kind=LayerKind.INPUT.value,
            params={"shape": (B, n), "dtype": str(dtype)},
            in_vars=[], out_vars=in_v,
        ),
        Layer(
            id=1, kind=LayerKind.INPUT_SPEC.value,
            params={"kind": "BOX", "lb": lb, "ub": ub},
            in_vars=in_v, out_vars=in_v,
        ),
        Layer(
            id=2, kind=LayerKind.RELU.value,
            params={}, in_vars=in_v, out_vars=out_v,
        ),
        Layer(
            id=3, kind=LayerKind.ASSERT.value,
            params=spec_layer, in_vars=out_v, out_vars=out_v,
        ),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2]}
    succs = {0: [1], 1: [2], 2: [3], 3: []}
    return Net(layers=layers, preds=preds, succs=succs)


def _run_analyze(net, lb, ub):
    from act.back_end.analyze import analyze
    from act.back_end.core import Bounds, Con, ConSet, Fact
    from act.front_end.specs import InKind

    input_layer_ids = [layer.id for layer in net.layers if layer.kind == "INPUT"]
    if len(input_layer_ids) != 1:
        raise ValueError(f"Expected exactly one INPUT layer, found {len(input_layer_ids)}.")
    entry_id = input_layer_ids[0]
    input_ids = list(net.by_id[entry_id].out_vars)
    spec_layers = [layer for layer in net.layers if layer.kind == "INPUT_SPEC"]
    seed = Bounds(lb.clone(), ub.clone())
    entry_fact = Fact(bounds=seed, cons=ConSet())
    for spec_layer in spec_layers:
        kind = spec_layer.params.get("kind")
        if kind == InKind.BOX:
            entry_fact.cons.add_box(
                -1, input_ids,
                Bounds(spec_layer.params["lb"], spec_layer.params["ub"]),
            )
        elif kind == InKind.LIN_POLY:
            entry_fact.cons.replace(
                Con(
                    "INEQ", tuple(input_ids),
                    {
                        "tag": "in:linpoly",
                        "A": spec_layer.params["A"],
                        "b": spec_layer.params["b"],
                    },
                )
            )
        else:
            raise NotImplementedError(f"Unsupported INPUT_SPEC kind: {kind}")
    _before, _after, globalC = analyze(net, entry_id, entry_fact)
    return globalC


def _test_export_relu_canonical():  # pragma: no cover
    """3 ineq rows per RELU neuron, ON/OFF/AMB slopes match Oracle §I."""
    torch.manual_seed(0)
    B = 1
    n = 6
    lb = torch.tensor([[-2.0, -0.5, 1.0, 2.0, -1.0, -3.0]])
    ub = torch.tensor([[-0.1, 0.5, 3.0, 5.0, 2.0, -2.0]])
    net = _build_relu_test_net(B, n, lb, ub)
    globalC = _run_analyze(net, lb, ub)
    assert_layer = net.layers[-1]
    bp = export_to_batch_problem(
        net, globalC, assert_layer,
        Bounds(lb=lb, ub=ub),
    )
    expected_relu_rows = 3 * n
    assert bp.m_le >= expected_relu_rows + 1, (
        f"expected at least {expected_relu_rows} + 1 rows; got {bp.m_le}"
    )
    A_le_dense = _dense_block_rows(bp.A_le_blockdiag, bp.N, bp.m_le, bp.nvars)
    rows = A_le_dense[0]
    rhs = bp.b_le[0]
    nvars_net = 2 * n
    for i in range(n):
        z_id = n + i
        y_id = i
        row_a = rows[3 * i]
        row_b = rows[3 * i + 1]
        row_c = rows[3 * i + 2]
        assert float(row_a[z_id]) == -1.0, f"row_a[z_{i}] != -1"
        assert float(rhs[3 * i]) == 0.0
        assert float(row_b[y_id]) == 1.0 and float(row_b[z_id]) == -1.0
        assert float(rhs[3 * i + 1]) == 0.0
        assert float(row_c[z_id]) == 1.0
        lb_i = float(lb[0, i])
        ub_i = float(ub[0, i])
        if lb_i >= 0:
            expected_slope = 1.0
            expected_shift = 0.0
        elif ub_i <= 0:
            expected_slope = 0.0
            expected_shift = 0.0
        else:
            expected_slope = ub_i / (ub_i - lb_i)
            expected_shift = -lb_i * expected_slope
        coeff_on_y = float(row_c[y_id])
        assert abs(coeff_on_y - (-expected_slope)) < 1e-6, (
            f"neuron {i}: coeff_on_y={coeff_on_y} != -{expected_slope}"
        )
        assert abs(float(rhs[3 * i + 2]) - expected_shift) < 1e-6, (
            f"neuron {i}: rhs={float(rhs[3 * i + 2])} != {expected_shift}"
        )


def _build_lrelu_test_net(B, n, lb, ub, alpha):  # pragma: no cover
    from act.back_end.core import Layer, Net
    from act.back_end.layer_schema import LayerKind
    from act.front_end.specs import OutputSpec, OutKind

    device = lb.device
    dtype = lb.dtype
    in_v = list(range(n))
    out_v = list(range(n, 2 * n))
    spec = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.zeros(n, device=device, dtype=dtype),
        d=torch.tensor(1.0, device=device, dtype=dtype),
    ).encode_linear(B=B, n_out=n, device=device, dtype=dtype)
    layers = [
        Layer(
            id=0, kind=LayerKind.INPUT.value,
            params={"shape": (B, n), "dtype": str(dtype)},
            in_vars=[], out_vars=in_v,
        ),
        Layer(
            id=1, kind=LayerKind.INPUT_SPEC.value,
            params={"kind": "BOX", "lb": lb, "ub": ub},
            in_vars=in_v, out_vars=in_v,
        ),
        Layer(
            id=2, kind=LayerKind.LRELU.value,
            params={"alpha": alpha}, in_vars=in_v, out_vars=out_v,
        ),
        Layer(
            id=3, kind=LayerKind.ASSERT.value,
            params=spec, in_vars=out_v, out_vars=out_v,
        ),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2]}
    succs = {0: [1], 1: [2], 2: [3], 3: []}
    return Net(layers=layers, preds=preds, succs=succs)


def _test_export_lrelu_canonical():  # pragma: no cover
    """3 ineq per LRELU neuron with phase-degenerate slopes."""
    B = 1
    n = 4
    alpha = 0.1
    lb = torch.tensor([[-2.0, 0.5, -1.0, -3.0]])
    ub = torch.tensor([[-0.1, 2.0, 1.0, -2.0]])
    net = _build_lrelu_test_net(B, n, lb, ub, alpha)
    globalC = _run_analyze(net, lb, ub)
    bp = export_to_batch_problem(
        net, globalC, net.layers[-1],
        Bounds(lb=lb, ub=ub),
    )
    assert bp.m_le >= 3 * n + 1, f"expected >= {3 * n + 1}, got {bp.m_le}"
    A_dense = _dense_block_rows(bp.A_le_blockdiag, bp.N, bp.m_le, bp.nvars)
    rows = A_dense[0]
    rhs = bp.b_le[0]
    for i in range(n):
        z_id = n + i
        y_id = i
        row_a = rows[3 * i]
        row_b = rows[3 * i + 1]
        row_c = rows[3 * i + 2]
        assert float(row_a[y_id]) == 1.0 and float(row_a[z_id]) == -1.0
        assert abs(float(row_b[y_id]) - alpha) < 1e-9
        assert float(row_b[z_id]) == -1.0
        assert float(row_c[z_id]) == 1.0
        lb_i = float(lb[0, i])
        ub_i = float(ub[0, i])
        if lb_i >= 0:
            exp_slope = 1.0; exp_shift = 0.0
        elif ub_i <= 0:
            exp_slope = alpha; exp_shift = 0.0
        else:
            exp_slope = (ub_i - alpha * lb_i) / (ub_i - lb_i)
            exp_shift = alpha * lb_i - exp_slope * lb_i
        assert abs(float(row_c[y_id]) - (-exp_slope)) < 1e-6
        assert abs(float(rhs[3 * i + 2]) - exp_shift) < 1e-6


def _build_tanh_test_net(B, n, lb, ub):  # pragma: no cover
    from act.back_end.core import Layer, Net
    from act.back_end.layer_schema import LayerKind
    from act.front_end.specs import OutputSpec, OutKind

    device = lb.device
    dtype = lb.dtype
    in_v = list(range(n))
    out_v = list(range(n, 2 * n))
    spec = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.zeros(n, device=device, dtype=dtype),
        d=torch.tensor(2.0, device=device, dtype=dtype),
    ).encode_linear(B=B, n_out=n, device=device, dtype=dtype)
    layers = [
        Layer(
            id=0, kind=LayerKind.INPUT.value,
            params={"shape": (B, n), "dtype": str(dtype)},
            in_vars=[], out_vars=in_v,
        ),
        Layer(
            id=1, kind=LayerKind.INPUT_SPEC.value,
            params={"kind": "BOX", "lb": lb, "ub": ub},
            in_vars=in_v, out_vars=in_v,
        ),
        Layer(
            id=2, kind=LayerKind.TANH.value,
            params={}, in_vars=in_v, out_vars=out_v,
        ),
        Layer(
            id=3, kind=LayerKind.ASSERT.value,
            params=spec, in_vars=out_v, out_vars=out_v,
        ),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2]}
    succs = {0: [1], 1: [2], 2: [3], 3: []}
    return Net(layers=layers, preds=preds, succs=succs)


def _test_export_tanh_canonical_5_cases():  # pragma: no cover
    """100 random intervals spanning 5 cases: 4 valid ineq per element each.

    Validity check: for 50 random y ∈ [lo, hi], the LP-permitted z range
    must INCLUDE tanh(y) (otherwise the relaxation is unsound).
    """
    torch.manual_seed(42)
    n = 5
    intervals = []
    for _ in range(20):
        intervals.append(
            (torch.rand(n) * 0.4 - 0.2, torch.rand(n) * 0.4 - 0.2)
        )
    for _ in range(20):
        lo = -torch.rand(n) * 3.0 - 0.5
        hi = -torch.rand(n) * 0.5
        intervals.append((torch.minimum(lo, hi), torch.maximum(lo, hi)))
    for _ in range(20):
        lo = torch.rand(n) * 0.5
        hi = torch.rand(n) * 3.0 + 0.5
        intervals.append((torch.minimum(lo, hi), torch.maximum(lo, hi)))
    for _ in range(20):
        lo = -torch.rand(n) * 3.0 - 0.1
        hi = torch.rand(n) * 3.0 + 0.1
        intervals.append((lo, hi))
    for _ in range(20):
        center = (torch.rand(n) - 0.5) * 4.0
        half = torch.rand(n) * 1e-3 + 1e-6
        intervals.append((center - half, center + half))
    for k, (lo, hi) in enumerate(intervals):
        lo_b = torch.where(lo <= hi, lo, hi).unsqueeze(0)
        hi_b = torch.where(hi >= lo, hi, lo).unsqueeze(0)
        if (hi_b - lo_b).min() < 0:
            continue
        net = _build_tanh_test_net(1, n, lo_b, hi_b)
        globalC = _run_analyze(net, lo_b, hi_b)
        bp = export_to_batch_problem(
            net, globalC, net.layers[-1],
            Bounds(lb=lo_b, ub=hi_b),
        )
        expected_tanh_rows = 4 * n
        assert bp.m_le >= expected_tanh_rows, (
            f"case {k}: expected >= {expected_tanh_rows}, got {bp.m_le}"
        )
        A_dense = _dense_block_rows(bp.A_le_blockdiag, bp.N, bp.m_le, bp.nvars)
        rows = A_dense[0]
        rhs = bp.b_le[0]
        torch.manual_seed(k)
        for sample in range(50):
            y_samp = lo_b[0] + (hi_b[0] - lo_b[0]) * torch.rand(n)
            z_samp = torch.tanh(y_samp)
            for i in range(n):
                z_id = n + i
                y_id = i
                for r in range(4):
                    row = rows[4 * i + r]
                    coeff_z = float(row[z_id])
                    coeff_y = float(row[y_id])
                    lhs = coeff_z * float(z_samp[i]) + coeff_y * float(y_samp[i])
                    assert lhs <= float(rhs[4 * i + r]) + 1e-4, (
                        f"case {k} sample {sample} neuron {i} row {r}: "
                        f"tanh point ({float(y_samp[i])}, {float(z_samp[i])}) "
                        f"violates row coeffs (z={coeff_z}, y={coeff_y}, "
                        f"rhs={float(rhs[4 * i + r])}) -- lhs={lhs}"
                    )


def _build_dense_test_net(B, n_in, n_out, W, b, lb_in, ub_in):  # pragma: no cover
    from act.back_end.core import Layer, Net
    from act.back_end.layer_schema import LayerKind
    from act.front_end.specs import OutputSpec, OutKind

    device = lb_in.device
    dtype = lb_in.dtype
    in_v = list(range(n_in))
    out_v = list(range(n_in, n_in + n_out))
    spec = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.zeros(n_out, device=device, dtype=dtype),
        d=torch.tensor(100.0, device=device, dtype=dtype),
    ).encode_linear(B=B, n_out=n_out, device=device, dtype=dtype)
    layers = [
        Layer(
            id=0, kind=LayerKind.INPUT.value,
            params={"shape": (B, n_in), "dtype": str(dtype)},
            in_vars=[], out_vars=in_v,
        ),
        Layer(
            id=1, kind=LayerKind.INPUT_SPEC.value,
            params={"kind": "BOX", "lb": lb_in, "ub": ub_in},
            in_vars=in_v, out_vars=in_v,
        ),
        Layer(
            id=2, kind=LayerKind.DENSE.value,
            params={
                "weight": W, "in_features": n_in, "out_features": n_out,
                "weight_pos": W.clamp(min=0), "weight_neg": W.clamp(max=0),
                "bias": b, "input_shape": (n_in,),
            },
            in_vars=in_v, out_vars=out_v,
        ),
        Layer(
            id=3, kind=LayerKind.ASSERT.value,
            params=spec, in_vars=out_v, out_vars=out_v,
        ),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2]}
    succs = {0: [1], 1: [2], 2: [3], 3: []}
    return Net(layers=layers, preds=preds, succs=succs)


def _test_export_dense_uniform():  # pragma: no cover
    """m_eq must equal W.shape[0] (one eq per output) for any N."""
    for B in (1, 4, 8):
        n_in, n_out = 4, 3
        W = torch.tensor(
            [[1.0, 2.0, -1.0, 0.5],
             [0.0, 1.0, 1.0, 1.0],
             [-0.5, 0.0, 2.0, 1.0]],
        )
        b = torch.tensor([0.1, -0.2, 0.3])
        lb_in = torch.full((B, n_in), -1.0)
        ub_in = torch.full((B, n_in), 1.0)
        net = _build_dense_test_net(B, n_in, n_out, W, b, lb_in, ub_in)
        globalC = _run_analyze(net, lb_in, ub_in)
        bp = export_to_batch_problem(
            net, globalC, net.layers[-1],
            Bounds(lb=lb_in, ub=ub_in),
        )
        assert bp.m_eq == n_out, (
            f"B={B}: expected m_eq={n_out}, got {bp.m_eq}"
        )
        assert bp.N == B
        A_dense = _dense_block_rows(
            bp.A_eq_blockdiag, bp.N, bp.m_eq, bp.nvars
        )
        for nb in range(B):
            for i in range(n_out):
                row = A_dense[nb, i]
                rhs = float(bp.b_eq[nb, i])
                assert float(row[n_in + i]) == 1.0, (
                    f"B={B} n={nb}: coeff on y_{i} != 1"
                )
                for j in range(n_in):
                    assert abs(float(row[j]) - (-float(W[i, j]))) < 1e-9, (
                        f"B={B} n={nb}: coeff on x_{j} != -W[{i},{j}]"
                    )
                assert abs(rhs - float(b[i])) < 1e-9


def _build_top1_test_net(B, K, y_true, lb_in, ub_in, W, bias):  # pragma: no cover
    from act.back_end.core import Layer, Net
    from act.back_end.layer_schema import LayerKind
    from act.front_end.specs import OutputSpec, OutKind

    n_in = lb_in.shape[1]
    device = lb_in.device
    dtype = lb_in.dtype
    in_v = list(range(n_in))
    out_v = list(range(n_in, n_in + K))
    spec = OutputSpec(
        kind=OutKind.TOP1_ROBUST,
        y_true=y_true,
    ).encode_linear(B=B, n_out=K, device=device, dtype=dtype)
    layers = [
        Layer(
            id=0, kind=LayerKind.INPUT.value,
            params={"shape": (B, n_in), "dtype": str(dtype)},
            in_vars=[], out_vars=in_v,
        ),
        Layer(
            id=1, kind=LayerKind.INPUT_SPEC.value,
            params={"kind": "BOX", "lb": lb_in, "ub": ub_in},
            in_vars=in_v, out_vars=in_v,
        ),
        Layer(
            id=2, kind=LayerKind.DENSE.value,
            params={
                "weight": W, "in_features": n_in, "out_features": K,
                "weight_pos": W.clamp(min=0), "weight_neg": W.clamp(max=0),
                "bias": bias, "input_shape": (n_in,),
            },
            in_vars=in_v, out_vars=out_v,
        ),
        Layer(
            id=3, kind=LayerKind.ASSERT.value,
            params=spec, in_vars=out_v, out_vars=out_v,
        ),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2]}
    succs = {0: [1], 1: [2], 2: [3], 3: []}
    return Net(layers=layers, preds=preds, succs=succs)


def _test_export_top1_robust_batched():  # pragma: no cover
    """K+1 rows per instance with slack at slot nvars_net."""
    torch.manual_seed(7)
    for trial in range(4):
        B = int(torch.randint(1, 6, (1,)).item())
        K = int(torch.randint(2, 6, (1,)).item())
        n_in = 3
        y_true = torch.randint(0, K, (B,), dtype=torch.long)
        lb_in = torch.full((B, n_in), -0.5)
        ub_in = torch.full((B, n_in), 0.5)
        W = torch.randn(K, n_in) * 0.1
        bias = torch.zeros(K)
        net = _build_top1_test_net(B, K, y_true, lb_in, ub_in, W, bias)
        globalC = _run_analyze(net, lb_in, ub_in)
        bp = export_to_batch_problem(
            net, globalC, net.layers[-1],
            Bounds(lb=lb_in, ub=ub_in),
        )
        assert bp.nvars == n_in + K + 1, (
            f"trial {trial}: expected nvars=n_in+K+1={n_in + K + 1}, "
            f"got {bp.nvars}"
        )
        slack_id = n_in + K
        for nb in range(B):
            assert float(bp.lb[nb, slack_id]) == 0.0, (
                f"trial {trial} n={nb}: slack lb should be 0"
            )
        assert bp.m_eq == K, f"trial {trial}: dense m_eq={bp.m_eq} != K={K}"
        assert bp.m_le >= K + 1, (
            f"trial {trial}: top1 m_le={bp.m_le} should be >= K+1={K + 1}"
        )


def _build_simple_dense_relu_dense_top1_net(
    B, n_in, n_hidden, K, y_true, W1, b1, W2, b2, lb_in, ub_in,
):
    from act.back_end.core import Layer, Net
    from act.back_end.layer_schema import LayerKind
    from act.front_end.specs import OutputSpec, OutKind

    device = lb_in.device
    dtype = lb_in.dtype
    in_v = list(range(n_in))
    h_pre = list(range(n_in, n_in + n_hidden))
    h_post = list(range(n_in + n_hidden, n_in + 2 * n_hidden))
    out_v = list(
        range(n_in + 2 * n_hidden, n_in + 2 * n_hidden + K)
    )
    spec = OutputSpec(
        kind=OutKind.TOP1_ROBUST, y_true=y_true,
    ).encode_linear(B=B, n_out=K, device=device, dtype=dtype)
    layers = [
        Layer(
            id=0, kind=LayerKind.INPUT.value,
            params={"shape": (B, n_in), "dtype": str(dtype)},
            in_vars=[], out_vars=in_v,
        ),
        Layer(
            id=1, kind=LayerKind.INPUT_SPEC.value,
            params={"kind": "BOX", "lb": lb_in, "ub": ub_in},
            in_vars=in_v, out_vars=in_v,
        ),
        Layer(
            id=2, kind=LayerKind.DENSE.value,
            params={
                "weight": W1, "in_features": n_in, "out_features": n_hidden,
                "weight_pos": W1.clamp(min=0), "weight_neg": W1.clamp(max=0),
                "bias": b1, "input_shape": (n_in,),
            },
            in_vars=in_v, out_vars=h_pre,
        ),
        Layer(
            id=3, kind=LayerKind.RELU.value,
            params={}, in_vars=h_pre, out_vars=h_post,
        ),
        Layer(
            id=4, kind=LayerKind.DENSE.value,
            params={
                "weight": W2, "in_features": n_hidden, "out_features": K,
                "weight_pos": W2.clamp(min=0), "weight_neg": W2.clamp(max=0),
                "bias": b2, "input_shape": (n_hidden,),
            },
            in_vars=h_post, out_vars=out_v,
        ),
        Layer(
            id=5, kind=LayerKind.ASSERT.value,
            params=spec, in_vars=out_v, out_vars=out_v,
        ),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]}
    succs = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []}
    return Net(layers=layers, preds=preds, succs=succs)





def _test_export_n1_batch_problem_self_consistent():  # pragma: no cover
    torch.manual_seed(123)
    B = 1
    n_in, n_hidden, K = 4, 6, 3
    y_true = torch.tensor([1], dtype=torch.long)
    W1 = torch.randn(n_hidden, n_in) * 0.5
    b1 = torch.randn(n_hidden) * 0.1
    W2 = torch.randn(K, n_hidden) * 0.5
    b2 = torch.randn(K) * 0.1
    lb_in = torch.full((B, n_in), -1.0)
    ub_in = torch.full((B, n_in), 1.0)
    net = _build_simple_dense_relu_dense_top1_net(
        B, n_in, n_hidden, K, y_true, W1, b1, W2, b2, lb_in, ub_in,
    )
    globalC = _run_analyze(net, lb_in, ub_in)
    bp = export_to_batch_problem(
        net, globalC, net.layers[-1], Bounds(lb=lb_in, ub=ub_in),
    )

    assert bp.N == 1
    assert bp.nvars >= max(max(con.var_ids) for con in globalC) + 1
    assert bp.lb.shape == bp.ub.shape == (1, bp.nvars)
    assert bp.A_eq_blockdiag.shape == (bp.m_eq, bp.nvars)
    assert bp.A_le_blockdiag.shape == (bp.m_le, bp.nvars)
    assert bp.m_eq > 0 and bp.m_le > 0


def _build_conv2d_test_net(  # pragma: no cover
    B, C_in, H_in, W_in, C_out, K_h, K_w, stride, padding,
    weight, bias_flat, lb_in, ub_in,
):
    from act.back_end.core import Layer, Net
    from act.back_end.layer_schema import LayerKind
    from act.front_end.specs import OutputSpec, OutKind

    device = lb_in.device
    dtype = lb_in.dtype
    sh, sw = stride
    ph, pw = padding
    H_out = (H_in + 2 * ph - (K_h - 1) - 1) // sh + 1
    W_out = (W_in + 2 * pw - (K_w - 1) - 1) // sw + 1
    n_in_flat = C_in * H_in * W_in
    n_out_flat = C_out * H_out * W_out
    in_v = list(range(n_in_flat))
    out_v = list(range(n_in_flat, n_in_flat + n_out_flat))
    spec_layer = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.zeros(n_out_flat, device=device, dtype=dtype),
        d=torch.tensor(1.0, device=device, dtype=dtype),
    ).encode_linear(B=B, n_out=n_out_flat, device=device, dtype=dtype)
    layers = [
        Layer(
            id=0, kind=LayerKind.INPUT.value,
            params={"shape": (B, C_in, H_in, W_in), "dtype": str(dtype)},
            in_vars=[], out_vars=in_v,
        ),
        Layer(
            id=1, kind=LayerKind.INPUT_SPEC.value,
            params={
                "kind": "BOX",
                "lb": lb_in.reshape(B, -1),
                "ub": ub_in.reshape(B, -1),
            },
            in_vars=in_v, out_vars=in_v,
        ),
        Layer(
            id=2, kind=LayerKind.CONV2D.value,
            params={
                "weight": weight,
                "in_channels": C_in, "out_channels": C_out,
                "kernel_size": K_h if K_h == K_w else (K_h, K_w),
                "stride": stride, "padding": padding,
                "dilation": 1, "groups": 1,
                "input_shape": (1, C_in, H_in, W_in),
                "output_shape": (1, C_out, H_out, W_out),
            },
            in_vars=in_v, out_vars=out_v,
        ),
        Layer(
            id=3, kind=LayerKind.ASSERT.value,
            params=spec_layer, in_vars=out_v, out_vars=out_v,
        ),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2]}
    succs = {0: [1], 1: [2], 2: [3], 3: []}
    return Net(layers=layers, preds=preds, succs=succs), (H_out, W_out)


def _test_export_conv2d_n1_parity_vs_torch():  # pragma: no cover
    """Conv2D LP matrix matches torch.nn.functional.conv2d on a random input."""
    torch.manual_seed(2025)
    B = 1
    C_in, H_in, W_in = 3, 4, 4
    C_out, K_h, K_w = 2, 3, 3
    stride = (1, 1)
    padding = (1, 1)
    weight = torch.randn(C_out, C_in, K_h, K_w, dtype=torch.float64) * 0.5
    lb_in = torch.full((B, C_in, H_in, W_in), -1.0, dtype=torch.float64)
    ub_in = torch.full((B, C_in, H_in, W_in), 1.0, dtype=torch.float64)
    bias_flat = torch.zeros(C_out * H_in * W_in, dtype=torch.float64)
    net, (H_out, W_out) = _build_conv2d_test_net(
        B, C_in, H_in, W_in, C_out, K_h, K_w, stride, padding,
        weight, bias_flat, lb_in, ub_in,
    )
    globalC = _run_analyze(
        net, lb_in.reshape(B, -1), ub_in.reshape(B, -1)
    )
    bp = export_to_batch_problem(
        net, globalC, net.layers[-1],
        Bounds(lb=lb_in.reshape(B, -1), ub=ub_in.reshape(B, -1)),
    )

    n_in_flat = C_in * H_in * W_in
    n_out_flat = C_out * H_out * W_out
    assert bp.m_eq == n_out_flat, (
        f"expected m_eq={n_out_flat}, got {bp.m_eq}"
    )

    A_dense = _dense_block_rows(
        bp.A_eq_blockdiag, bp.N, bp.m_eq, bp.nvars
    )
    rows = A_dense[0]
    rhs = bp.b_eq[0]

    x_sample = torch.randn(B, C_in, H_in, W_in, dtype=torch.float64)
    y_ref = torch.nn.functional.conv2d(
        x_sample, weight, bias=None, stride=stride, padding=padding,
    )

    x_flat = x_sample.reshape(B, n_in_flat)
    y_flat_ref = y_ref.reshape(B, n_out_flat)

    for r in range(n_out_flat):
        row = rows[r]
        coef_on_y = float(row[n_in_flat + r])
        assert coef_on_y == 1.0, (
            f"row {r}: coef on y_{r} = {coef_on_y} (expected 1.0)"
        )
        lhs = (
            float(y_flat_ref[0, r])
            + float((row[:n_in_flat] * x_flat[0]).sum())
        )
        assert abs(lhs - float(rhs[r])) < 1e-9, (
            f"row {r}: lhs={lhs} != rhs={float(rhs[r])} "
            f"(y_ref + (-W flat) . x diff)"
        )


def _test_export_conv2d_batched_N_4():  # pragma: no cover
    """B=4: each LP instance gets its own input bounds; W is shared (broadcast)."""
    torch.manual_seed(2026)
    B = 4
    C_in, H_in, W_in = 1, 4, 4
    C_out, K_h, K_w = 3, 3, 3
    stride = (1, 1)
    padding = (1, 1)
    weight = torch.randn(C_out, C_in, K_h, K_w, dtype=torch.float64) * 0.4
    lb_in = torch.randn(B, C_in, H_in, W_in, dtype=torch.float64) - 0.5
    ub_in = lb_in + torch.rand(B, C_in, H_in, W_in, dtype=torch.float64) + 0.5
    bias_flat = torch.zeros(C_out * H_in * W_in, dtype=torch.float64)
    net, (H_out, W_out) = _build_conv2d_test_net(
        B, C_in, H_in, W_in, C_out, K_h, K_w, stride, padding,
        weight, bias_flat, lb_in, ub_in,
    )
    globalC = _run_analyze(
        net, lb_in.reshape(B, -1), ub_in.reshape(B, -1)
    )
    bp = export_to_batch_problem(
        net, globalC, net.layers[-1],
        Bounds(lb=lb_in.reshape(B, -1), ub=ub_in.reshape(B, -1)),
    )
    assert bp.N == B
    n_out_flat = C_out * H_out * W_out
    assert bp.m_eq == n_out_flat
    A_dense = _dense_block_rows(
        bp.A_eq_blockdiag, bp.N, bp.m_eq, bp.nvars
    )
    for nb in range(1, B):
        diff = (A_dense[0] - A_dense[nb]).abs().max().item()
        assert diff < 1e-12, (
            f"conv2d coefficients must be uniform across N "
            f"(W shared); instance {nb} differs by {diff}"
        )
    n_in_flat = C_in * H_in * W_in
    for nb in range(B):
        for k in range(n_in_flat):
            v = float(bp.lb[nb, k])
            expected = float(lb_in.reshape(B, n_in_flat)[nb, k])
            assert abs(v - expected) < 1e-12


def _test_export_conv2d_stride2_pad0():  # pragma: no cover
    """Stride=2, pad=0: shrinking spatial output, dropped kernel taps at edges."""
    torch.manual_seed(2027)
    B = 1
    C_in, H_in, W_in = 1, 4, 4
    C_out, K_h, K_w = 1, 3, 3
    stride = (2, 2)
    padding = (0, 0)
    weight = torch.ones(C_out, C_in, K_h, K_w, dtype=torch.float64)
    lb_in = torch.full((B, C_in, H_in, W_in), -1.0, dtype=torch.float64)
    ub_in = torch.full((B, C_in, H_in, W_in), 1.0, dtype=torch.float64)
    bias_flat = torch.zeros(C_out * 1 * 1, dtype=torch.float64)
    net, (H_out, W_out) = _build_conv2d_test_net(
        B, C_in, H_in, W_in, C_out, K_h, K_w, stride, padding,
        weight, bias_flat, lb_in, ub_in,
    )
    assert (H_out, W_out) == (1, 1), (
        f"expected 1x1 output for 4x4 input k=3 s=2 p=0; got {H_out}x{W_out}"
    )
    globalC = _run_analyze(
        net, lb_in.reshape(B, -1), ub_in.reshape(B, -1)
    )
    bp = export_to_batch_problem(
        net, globalC, net.layers[-1],
        Bounds(lb=lb_in.reshape(B, -1), ub=ub_in.reshape(B, -1)),
    )
    n_in_flat = C_in * H_in * W_in
    n_out_flat = C_out * H_out * W_out
    assert bp.m_eq == n_out_flat
    A_dense = _dense_block_rows(
        bp.A_eq_blockdiag, bp.N, bp.m_eq, bp.nvars
    )
    row = A_dense[0, 0]
    coef_y = float(row[n_in_flat])
    assert coef_y == 1.0
    x_coefs = row[:n_in_flat].reshape(C_in, H_in, W_in)
    receptive = x_coefs[0, 0:3, 0:3]
    assert torch.allclose(
        receptive, -torch.ones_like(receptive)
    ), f"top-left 3x3 receptive coefs should be -1; got {receptive}"
    untouched_mask = torch.ones((H_in, W_in), dtype=torch.bool)
    untouched_mask[0:3, 0:3] = False
    untouched = x_coefs[0][untouched_mask]
    assert untouched.abs().max().item() < 1e-12, (
        f"positions outside 3x3 receptive field must have coefficient 0; "
        f"got max abs {untouched.abs().max().item()}"
    )


def _build_unary_layer_test_net(B, n, kind, params, lb, ub):  # pragma: no cover
    from act.back_end.core import Layer, Net
    from act.back_end.layer_schema import LayerKind
    from act.front_end.specs import OutputSpec, OutKind

    device = lb.device
    dtype = lb.dtype
    in_v = list(range(n))
    out_v = list(range(n, 2 * n))
    spec = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.zeros(n, device=device, dtype=dtype),
        d=torch.tensor(-1.0e6, device=device, dtype=dtype),
    ).encode_linear(B=B, n_out=n, device=device, dtype=dtype)
    layers = [
        Layer(
            id=0, kind=LayerKind.INPUT.value,
            params={"shape": (B, n), "dtype": str(dtype)},
            in_vars=[], out_vars=in_v,
        ),
        Layer(
            id=1, kind=LayerKind.INPUT_SPEC.value,
            params={"kind": "BOX", "lb": lb, "ub": ub},
            in_vars=in_v, out_vars=in_v,
        ),
        Layer(id=2, kind=kind, params=params, in_vars=in_v, out_vars=out_v),
        Layer(id=3, kind=LayerKind.ASSERT.value, params=spec, in_vars=out_v, out_vars=out_v),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2]}
    succs = {0: [1], 1: [2], 2: [3], 3: []}
    return Net(layers=layers, preds=preds, succs=succs)


def _export_unary_layer(B, n, kind, params, lb, ub):
    net = _build_unary_layer_test_net(B, n, kind, params, lb, ub)
    globalC = _run_analyze(net, lb, ub)
    bp = export_to_batch_problem(net, globalC, net.layers[-1], Bounds(lb=lb, ub=ub))
    return net, bp


def _assert_layer_rows_hold(bp, x, y):
    A_le = _dense_block_rows(bp.A_le_blockdiag, bp.N, bp.m_le, bp.nvars)
    A_eq = _dense_block_rows(bp.A_eq_blockdiag, bp.N, bp.m_eq, bp.nvars)
    n = x.shape[1]
    vals = torch.cat([x, y], dim=1).to(dtype=bp.lb.dtype)
    for b in range(bp.N):
        if bp.m_eq:
            eq_lhs = A_eq[b].matmul(vals[b])
            assert torch.allclose(eq_lhs, bp.b_eq[b], atol=1e-7, rtol=1e-7), (
                f"eq rows failed: lhs={eq_lhs} rhs={bp.b_eq[b]}"
            )
        if bp.m_le:
            le_lhs = A_le[b].matmul(vals[b])
            assert bool(torch.all(le_lhs <= bp.b_le[b] + 1e-6)), (
                f"le rows failed: max_violation={(le_lhs - bp.b_le[b]).max()}"
            )
    assert vals.shape[1] == 2 * n


def _test_export_flatten_identity():  # pragma: no cover
    from act.back_end.layer_schema import LayerKind
    B, n = 2, 6
    lb = torch.full((B, n), -1.0, dtype=torch.float64)
    ub = torch.full((B, n), 2.0, dtype=torch.float64)
    _net, bp = _export_unary_layer(
        B, n, LayerKind.FLATTEN.value,
        {"input_shape": (B, 2, 3), "output_shape": (B, 6)}, lb, ub,
    )
    assert bp.m_eq == n
    x = torch.randn(B, n, dtype=torch.float64)
    _assert_layer_rows_hold(bp, x, x)


def _test_export_sigmoid_relaxation():  # pragma: no cover
    from act.back_end.layer_schema import LayerKind
    B, n = 2, 4
    lb = torch.tensor([[-3.0, -1.0, 0.2, -2.0], [-0.5, 0.0, 1.0, -4.0]], dtype=torch.float64)
    ub = torch.tensor([[3.0, 0.5, 2.0, -0.5], [0.5, 1.5, 4.0, 2.0]], dtype=torch.float64)
    _net, bp = _export_unary_layer(B, n, LayerKind.SIGMOID.value, {}, lb, ub)
    assert bp.m_le >= 4 * n
    x = 0.5 * (lb + ub)
    y = torch.sigmoid(x)
    _assert_layer_rows_hold(bp, x, y)


def _test_export_relu6_hull_relaxation():  # pragma: no cover
    from act.back_end.layer_schema import LayerKind
    B, n = 1, 5
    lb = torch.tensor([[-2.0, -0.1, 0.5, 5.0, 7.0]], dtype=torch.float64)
    ub = torch.tensor([[1.0, 2.0, 3.0, 8.0, 9.0]], dtype=torch.float64)
    _net, bp = _export_unary_layer(B, n, LayerKind.RELU6.value, {}, lb, ub)
    assert bp.m_le > 2 * n
    x = torch.tensor([[-1.0, 1.0, 2.0, 6.0, 8.0]], dtype=torch.float64)
    _assert_layer_rows_hold(bp, x, torch.clamp(x, min=0.0, max=6.0))


def _test_export_hardsigmoid_hull_relaxation():  # pragma: no cover
    from act.back_end.layer_schema import LayerKind
    B, n = 1, 4
    lb = torch.full((B, n), -4.0, dtype=torch.float64)
    ub = torch.full((B, n), 4.0, dtype=torch.float64)
    params = {"alpha": 1.0 / 6.0, "beta": 0.5}
    _net, bp = _export_unary_layer(B, n, LayerKind.HARDSIGMOID.value, params, lb, ub)
    assert bp.m_le > 2 * n
    x = torch.tensor([[-4.0, -1.0, 1.0, 4.0]], dtype=torch.float64)
    y = torch.clamp(x / 6.0 + 0.5, min=0.0, max=1.0)
    _assert_layer_rows_hold(bp, x, y)


def _test_export_hardtanh_hull_relaxation():  # pragma: no cover
    from act.back_end.layer_schema import LayerKind
    B, n = 1, 4
    lb = torch.full((B, n), -2.0, dtype=torch.float64)
    ub = torch.full((B, n), 2.0, dtype=torch.float64)
    params = {"min_val": -1.0, "max_val": 1.0}
    _net, bp = _export_unary_layer(B, n, LayerKind.HARDTANH.value, params, lb, ub)
    assert bp.m_le > 2 * n
    x = torch.tensor([[-2.0, -0.5, 0.5, 2.0]], dtype=torch.float64)
    _assert_layer_rows_hold(bp, x, torch.clamp(x, min=-1.0, max=1.0))


def _test_export_mask_add_equality():  # pragma: no cover
    from act.back_end.layer_schema import LayerKind
    B, n = 1, 4
    lb = torch.full((B, n), -1.0, dtype=torch.float64)
    ub = torch.full((B, n), 1.0, dtype=torch.float64)
    mask = torch.tensor([0.0, -10000.0, 0.0, -10000.0], dtype=torch.float64)
    _net, bp = _export_unary_layer(B, n, LayerKind.MASK_ADD.value, {"M": mask}, lb, ub)
    x = torch.tensor([[0.1, 0.2, -0.3, 0.4]], dtype=torch.float64)
    _assert_layer_rows_hold(bp, x, x + mask.unsqueeze(0))


def _test_export_power_p2_relu_square_hull():  # pragma: no cover
    from act.back_end.layer_schema import LayerKind
    B, n = 1, 4
    lb = torch.full((B, n), -1.0, dtype=torch.float64)
    ub = torch.full((B, n), 2.0, dtype=torch.float64)
    _net, bp = _export_unary_layer(B, n, LayerKind.POWER.value, {"p": 2.0}, lb, ub)
    x = torch.tensor([[-1.0, 0.0, 1.0, 2.0]], dtype=torch.float64)
    _assert_layer_rows_hold(bp, x, torch.clamp(x, min=0.0).pow(2.0))


def _test_export_square_hull():  # pragma: no cover
    from act.back_end.layer_schema import LayerKind
    B, n = 1, 4
    lb = torch.full((B, n), -2.0, dtype=torch.float64)
    ub = torch.full((B, n), 1.5, dtype=torch.float64)
    _net, bp = _export_unary_layer(B, n, LayerKind.SQUARE.value, {}, lb, ub)
    x = torch.tensor([[-2.0, -0.5, 0.5, 1.5]], dtype=torch.float64)
    _assert_layer_rows_hold(bp, x, x * x)


def _test_export_layernorm_box_relaxation():  # pragma: no cover
    from act.back_end.layer_schema import LayerKind
    B, n = 1, 8
    lb = torch.full((B, n), -1.0, dtype=torch.float64)
    ub = torch.full((B, n), 1.0, dtype=torch.float64)
    gamma = torch.ones(n, dtype=torch.float64)
    beta = torch.zeros(n, dtype=torch.float64)
    eps = 1e-5
    params = {"gamma": gamma, "beta": beta, "eps": eps}
    _net, bp = _export_unary_layer(B, n, LayerKind.LAYERNORM.value, params, lb, ub)
    x = torch.linspace(-0.7, 0.7, n, dtype=torch.float64).unsqueeze(0)
    y = torch.nn.functional.layer_norm(x, (n,), gamma, beta, eps)
    _assert_layer_rows_hold(bp, x, y)


_BATCHED_TESTS = [  # pragma: no cover
    _test_export_relu_canonical,
    _test_export_lrelu_canonical,
    _test_export_tanh_canonical_5_cases,
    _test_export_dense_uniform,
    _test_export_top1_robust_batched,
    _test_export_n1_batch_problem_self_consistent,
    _test_export_conv2d_n1_parity_vs_torch,
    _test_export_conv2d_batched_N_4,
    _test_export_conv2d_stride2_pad0,
    _test_export_flatten_identity,
    _test_export_sigmoid_relaxation,
    _test_export_relu6_hull_relaxation,
    _test_export_hardsigmoid_hull_relaxation,
    _test_export_hardtanh_hull_relaxation,
    _test_export_mask_add_equality,
    _test_export_power_p2_relu_square_hull,
    _test_export_square_hull,
    _test_export_layernorm_box_relaxation,
]


def _run_batched_tests() -> int:
    passed = failed = 0
    for fn in _BATCHED_TESTS:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    import sys
    from act.util.device_manager import initialize_device
    initialize_device("cpu", "float64")
    print("Running cons_exportor batched self-tests\n")
    sys.exit(_run_batched_tests())
