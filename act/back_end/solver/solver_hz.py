from __future__ import annotations

import logging
import time

import torch
from dataclasses import dataclass, replace
from typing import Optional, TYPE_CHECKING
from act.back_end.core import Bounds
from act.back_end.solver.solver_base import Solver, SolverCaps
from act.front_end.specs import OutKind
from act.util.stats import VerifyResult, VerifyStatus

if TYPE_CHECKING:
    from act.back_end.solver.solver_base import BatchLPProblem, BatchLPSolution
    from act.front_end.specs import OutputSpec

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import scipy.sparse as sp
    from scipy.optimize import Bounds as SciPyBounds
    from scipy.optimize import LinearConstraint, linprog, milp

    _HAS_SCIPY = True
except ImportError:
    np = None
    sp = None
    _HAS_SCIPY = False

try:
    import highspy
except ImportError:
    highspy = None


# ============================================================================
# 1. HZono dataclass
# ============================================================================


@dataclass
class HZono:
    """Z = {c + Gc @ xi_c + Gb @ xi_b | Ac @ xi_c + Ab @ xi_b = b,
    xi_c in [-1,1]^ng, xi_b in {-1,1}^nb}"""

    c: torch.Tensor  # (n, 1)
    Gc: torch.Tensor  # (n, ng)
    Gb: torch.Tensor  # (n, nb)
    Ac: torch.Tensor  # (nc, ng)
    Ab: torch.Tensor  # (nc, nb)
    b: torch.Tensor  # (nc, 1)
    eq_mask: Optional[torch.Tensor] = None
    col_ids: Optional[torch.Tensor] = None
    bcol_ids: Optional[torch.Tensor] = None


@dataclass
class SparseHZono:
    c: "np.ndarray"
    Gc: "sp.csr_matrix"
    Gb: "sp.csr_matrix"
    Ac: "sp.csr_matrix"
    Ab: "sp.csr_matrix"
    b: "np.ndarray"
    Auc: Optional["sp.csr_matrix"] = None
    Aub: Optional["sp.csr_matrix"] = None
    ub: Optional["np.ndarray"] = None
    frame_id: Optional[int] = None
    exact: bool = True

    def __post_init__(self) -> None:
        _require_sparse()
        self.c = np.asarray(self.c, dtype=np.float64).reshape(-1)
        self.b = np.asarray(self.b, dtype=np.float64).reshape(-1)
        self.Gc = _as_csr(self.Gc)
        self.Gb = _as_csr(self.Gb)
        self.Ac = _as_csr(self.Ac)
        self.Ab = _as_csr(self.Ab)

        n_out = int(self.c.size)
        n_cont = int(self.Gc.shape[1])
        n_bin = int(self.Gb.shape[1])
        if self.Gc.shape[0] != n_out or self.Gb.shape[0] != n_out:
            raise ValueError(
                "SparseHZono value shape mismatch: "
                f"c={n_out}, Gc={self.Gc.shape}, Gb={self.Gb.shape}"
            )
        if self.Ac.shape[1] != n_cont or self.Ab.shape[1] != n_bin:
            raise ValueError(
                "SparseHZono equality column mismatch: "
                f"Gc_cols={n_cont}, Gb_cols={n_bin}, Ac={self.Ac.shape}, Ab={self.Ab.shape}"
            )
        if self.Ac.shape[0] != self.Ab.shape[0] or self.Ac.shape[0] != self.b.size:
            raise ValueError(
                "SparseHZono equality row mismatch: "
                f"Ac={self.Ac.shape}, Ab={self.Ab.shape}, b={self.b.size}"
            )

        if self.Auc is None and self.Aub is None and self.ub is None:
            self.Auc = sparse_empty(0, n_cont)
            self.Aub = sparse_empty(0, n_bin)
            self.ub = np.zeros(0, dtype=np.float64)
        elif self.Auc is None or self.Aub is None or self.ub is None:
            raise ValueError("upper constraints require Auc, Aub, and ub together")
        else:
            self.Auc = _as_csr(self.Auc, shape=(self.Auc.shape[0], n_cont))
            self.Aub = _as_csr(self.Aub, shape=(self.Aub.shape[0], n_bin))
            self.ub = np.asarray(self.ub, dtype=np.float64).reshape(-1)
            if self.Auc.shape[0] != self.Aub.shape[0] or self.Auc.shape[0] != self.ub.size:
                raise ValueError(
                    "SparseHZono upper row mismatch: "
                    f"Auc={self.Auc.shape}, Aub={self.Aub.shape}, ub={self.ub.size}"
                )

    @property
    def n_out(self) -> int:
        return int(self.c.size)

    @property
    def n_cont(self) -> int:
        return int(self.Gc.shape[1])

    @property
    def n_bin(self) -> int:
        return int(self.Gb.shape[1])

    @property
    def n_eq(self) -> int:
        return int(self.Ac.shape[0])

    @property
    def n_ineq(self) -> int:
        return int(self.Auc.shape[0])


def hz_tighten_bounds(base: Bounds, candidate: Bounds) -> Bounds:
    candidate_lb = candidate.lb.reshape_as(base.lb).to(base.lb)
    candidate_ub = candidate.ub.reshape_as(base.ub).to(base.ub)
    lb = torch.maximum(base.lb, candidate_lb)
    ub = torch.minimum(base.ub, candidate_ub)
    conflict = lb > ub
    if bool(conflict.any()):
        scale = torch.maximum(
            torch.maximum(lb.abs(), ub.abs()),
            torch.ones((), dtype=lb.dtype, device=lb.device),
        )
        tolerance = 128 * torch.finfo(lb.dtype).eps * scale
        if bool(((lb - ub > tolerance) & conflict).any()):
            raise ValueError("HZ and interval bounds have a non-numerical conflict")
    return Bounds(
        lb=torch.where(conflict, base.lb, lb),
        ub=torch.where(conflict, base.ub, ub),
    )


def hz_bounds_are_liftable(bounds: Bounds) -> bool:
    return bool(
        torch.isfinite(bounds.lb).all()
        and torch.isfinite(bounds.ub).all()
        and (bounds.lb <= bounds.ub).all()
    )


_NEXT_COL_ID = [-1]


def hz_fresh_col_ids(k: int, device=None) -> torch.Tensor:
    k = int(k)
    start = _NEXT_COL_ID[0]
    _NEXT_COL_ID[0] = start - k
    return torch.arange(start, start - k, -1, dtype=torch.long, device=device)


# ============================================================================
# 2. Algebraic operations
# ============================================================================


def hz_multiply(hz: HZono, R: torch.Tensor) -> HZono:
    R = R.to(dtype=hz.c.dtype, device=hz.c.device)
    return HZono(
        c=R @ hz.c,
        Gc=R @ hz.Gc,
        Gb=R @ hz.Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=_clone_ids(hz.eq_mask),
        col_ids=_clone_ids(hz.col_ids),
        bcol_ids=_clone_ids(hz.bcol_ids),
    )


def hz_add_const(hz: HZono, v: torch.Tensor) -> HZono:
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
        eq_mask=_clone_ids(hz.eq_mask),
        col_ids=_clone_ids(hz.col_ids),
        bcol_ids=_clone_ids(hz.bcol_ids),
    )


def hz_minkowski_sum(hz1: HZono, hz2: HZono) -> HZono:
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
    if hz1.eq_mask is None and hz2.eq_mask is None:
        new_eq_mask = None
    else:
        m1 = hz1.eq_mask if hz1.eq_mask is not None else torch.ones(
            nc1, dtype=torch.bool, device=device
        )
        m2 = hz2.eq_mask if hz2.eq_mask is not None else torch.ones(
            nc2, dtype=torch.bool, device=device
        )
        new_eq_mask = torch.cat([m1.to(device), m2.to(device)], dim=0)
    new_col_ids = None
    if hz1.col_ids is not None and hz2.col_ids is not None:
        new_col_ids = torch.cat([hz1.col_ids.to(device), hz2.col_ids.to(device)])
    new_bcol_ids = None
    if hz1.bcol_ids is not None and hz2.bcol_ids is not None:
        new_bcol_ids = torch.cat([hz1.bcol_ids.to(device), hz2.bcol_ids.to(device)])
    return HZono(
        c=new_c,
        Gc=new_Gc,
        Gb=new_Gb,
        Ac=new_Ac,
        Ab=new_Ab,
        b=new_b,
        eq_mask=new_eq_mask,
        col_ids=new_col_ids,
        bcol_ids=new_bcol_ids,
    )


def hz_from_bounds(
    bounds: Bounds,
    dtype,
    device,
    *,
    track_ids: bool = False,
    col_ids: Optional[torch.Tensor] = None,
) -> HZono:
    """Convert an interval box to an HZ, optionally seeding generator ids."""
    lb = bounds.lb.flatten().to(dtype=dtype, device=device)
    ub = bounds.ub.flatten().to(dtype=dtype, device=device)
    n = lb.shape[0]
    c = ((lb + ub) / 2.0).view(-1, 1)
    rad = (ub - lb) / 2.0
    nz = rad > 0
    ng = int(nz.sum().item())
    idx = torch.where(nz)[0]
    Gc = torch.zeros((n, ng), dtype=dtype, device=device)
    if ng:
        Gc[idx, torch.arange(ng, device=device)] = rad[idx]
    ids = None
    if col_ids is not None:
        full_ids = col_ids.to(device=device)
        if full_ids.numel() == n:
            ids = full_ids[idx]
        elif full_ids.numel() == ng:
            ids = full_ids
        else:
            ids = None
    elif track_ids:
        full_ids = hz_fresh_col_ids(n, device=device)
        ids = full_ids[idx]
    hz = HZono(
        c=c,
        Gc=Gc,
        Gb=torch.zeros((n, 0), dtype=dtype, device=device),
        Ac=torch.zeros((0, ng), dtype=dtype, device=device),
        Ab=torch.zeros((0, 0), dtype=dtype, device=device),
        b=torch.zeros((0, 1), dtype=dtype, device=device),
        col_ids=ids,
        bcol_ids=torch.zeros(0, dtype=torch.long, device=device)
        if ids is not None
        else None,
    )
    if col_ids is not None and col_ids.numel() == n:
        hz.full_col_ids = col_ids.to(device=device)
    elif track_ids:
        hz.full_col_ids = full_ids
    return hz


def hz_lift_bounds(
    hz: HZono,
    bounds: Bounds,
    *,
    output_equalities: Optional[torch.Tensor] = None,
    equality_rhs: Optional[torch.Tensor] = None,
    output_inequalities: Optional[torch.Tensor] = None,
    inequality_rhs: Optional[torch.Tensor] = None,
) -> HZono:
    dtype, device = hz.c.dtype, hz.c.device
    lb = bounds.lb.flatten().to(dtype=dtype, device=device)
    ub = bounds.ub.flatten().to(dtype=dtype, device=device)
    if not bool(torch.isfinite(lb).all() and torch.isfinite(ub).all()):
        raise ValueError("cannot lift non-finite bounds into a dense HZ")
    if not bool((lb <= ub).all()):
        raise ValueError("cannot lift inconsistent bounds into a dense HZ")

    center = (lb + ub) * 0.5
    radius = (ub - lb) * 0.5
    n_out = int(center.numel())
    old_cont = int(hz.Gc.shape[1])
    old_bin = int(hz.Gb.shape[1])
    Gc = torch.zeros((n_out, old_cont + n_out), dtype=dtype, device=device)
    if n_out:
        rows = torch.arange(n_out, device=device)
        Gc[rows, old_cont + rows] = radius
    Gb = torch.zeros((n_out, old_bin), dtype=dtype, device=device)
    Ac = torch.cat(
        [hz.Ac, torch.zeros((hz.Ac.shape[0], n_out), dtype=dtype, device=device)],
        dim=1,
    )
    Ab = hz.Ab.clone()
    b = hz.b.clone()
    eq_mask = _constraint_mask(hz)

    def add_rows(matrix, rhs, equality):
        nonlocal Ac, Ab, b, eq_mask
        if matrix is None:
            return
        matrix = matrix.to(dtype=dtype, device=device)
        if matrix.ndim != 2 or matrix.shape[1] != n_out:
            raise ValueError("output constraint shape mismatch")
        rhs = (
            torch.zeros(matrix.shape[0], dtype=dtype, device=device)
            if rhs is None
            else rhs.to(dtype=dtype, device=device).reshape(-1)
        )
        if rhs.numel() != matrix.shape[0]:
            raise ValueError("output constraint rhs shape mismatch")
        Ac = torch.cat([Ac, matrix @ Gc], dim=0)
        Ab = torch.cat(
            [Ab, torch.zeros((matrix.shape[0], old_bin), dtype=dtype, device=device)],
            dim=0,
        )
        b = torch.cat([b, (rhs - matrix @ center).view(-1, 1)], dim=0)
        eq_mask = torch.cat(
            [
                eq_mask,
                torch.full(
                    (matrix.shape[0],), equality, dtype=torch.bool, device=device
                ),
            ]
        )

    add_rows(output_equalities, equality_rhs, True)
    add_rows(output_inequalities, inequality_rhs, False)
    col_ids = None
    if hz.col_ids is not None:
        col_ids = torch.cat(
            [hz.col_ids.to(device), hz_fresh_col_ids(n_out, device=device)]
        )
    return HZono(
        c=center.view(-1, 1),
        Gc=Gc,
        Gb=Gb,
        Ac=Ac,
        Ab=Ab,
        b=b,
        eq_mask=eq_mask,
        col_ids=col_ids,
        bcol_ids=_clone_ids(hz.bcol_ids),
    )


def _require_sparse() -> None:
    if not _HAS_SCIPY:
        raise RuntimeError("Sparse HybridZ requires scipy")


def _as_csr(mat, *, shape=None):
    _require_sparse()
    out = mat if sp.issparse(mat) else sp.csr_matrix(mat, dtype=np.float64)
    out = out.tocsr().astype(np.float64)
    if shape is not None and out.shape != shape:
        if out.shape[0] != shape[0] or out.shape[1] > shape[1]:
            raise ValueError(f"CSR shape mismatch: {out.shape} vs {shape}")
        out = sp.hstack(
            [out, sp.csr_matrix((out.shape[0], shape[1] - out.shape[1]))],
            format="csr",
        )
    out.eliminate_zeros()
    return out


def _torch_to_csr(t: torch.Tensor):
    arr = t.detach().cpu().numpy().astype(np.float64)
    return sp.csr_matrix(arr)


def _bounds_to_numpy(bounds: Bounds):
    return tuple(
        value.detach().cpu().double().numpy().reshape(-1)
        for value in (bounds.lb, bounds.ub)
    )


def sparse_empty(rows: int, cols: int):
    _require_sparse()
    return sp.csr_matrix((int(rows), int(cols)), dtype=np.float64)


def sparse_abs_row_sum(mat):
    _require_sparse()
    matrix = mat if sp.isspmatrix_csr(mat) else sp.csr_matrix(mat)
    matrix.sum_duplicates()
    result = np.zeros(matrix.shape[0], dtype=np.float64)
    rows = np.flatnonzero(np.diff(matrix.indptr))
    if rows.size:
        result[rows] = np.add.reduceat(np.abs(matrix.data), matrix.indptr[rows])
    return result


def _sparse_generator_radius(Gc, Gb):
    radius = sparse_abs_row_sum(Gc)
    return radius + (sparse_abs_row_sum(Gb) if Gb.shape[1] else 0.0)


def sparse_pad_cols(mat, cols: int):
    mat = _as_csr(mat)
    cols = int(cols)
    if mat.shape[1] == cols:
        return mat
    if mat.shape[1] > cols:
        raise ValueError(f"cannot shrink sparse matrix from {mat.shape[1]} to {cols}")
    return sp.hstack([mat, sparse_empty(mat.shape[0], cols - mat.shape[1])], format="csr")


def sparse_hz_pad_frame(hz: SparseHZono, n_cont: int, n_bin: int) -> SparseHZono:
    return SparseHZono(
        c=hz.c,
        Gc=sparse_pad_cols(hz.Gc, n_cont),
        Gb=sparse_pad_cols(hz.Gb, n_bin),
        Ac=sparse_pad_cols(hz.Ac, n_cont),
        Ab=sparse_pad_cols(hz.Ab, n_bin),
        b=hz.b,
        Auc=sparse_pad_cols(hz.Auc, n_cont),
        Aub=sparse_pad_cols(hz.Aub, n_bin),
        ub=hz.ub,
        frame_id=hz.frame_id,
        exact=hz.exact,
    )


def sparse_hz_from_bounds(
    bounds: Bounds,
    *,
    frame_id: Optional[int] = None,
    drop_zero_radius: bool = True,
) -> SparseHZono:
    _require_sparse()
    lb = bounds.lb.detach().cpu().numpy().astype(np.float64).reshape(-1)
    ub = bounds.ub.detach().cpu().numpy().astype(np.float64).reshape(-1)
    center = (lb + ub) * 0.5
    rad = (ub - lb) * 0.5
    rows = (
        np.nonzero(np.abs(rad) > 1e-12)[0].astype(np.int32)
        if drop_zero_radius
        else np.arange(rad.size, dtype=np.int32)
    )
    cols = np.arange(rows.size, dtype=np.int32)
    Gc = sp.csr_matrix(
        (rad[rows], (rows, cols)),
        shape=(rad.size, rows.size),
        dtype=np.float64,
    )
    return SparseHZono(
        c=center,
        Gc=Gc,
        Gb=sparse_empty(rad.size, 0),
        Ac=sparse_empty(0, rows.size),
        Ab=sparse_empty(0, 0),
        b=np.zeros(0, dtype=np.float64),
        Auc=sparse_empty(0, rows.size),
        Aub=sparse_empty(0, 0),
        ub=np.zeros(0, dtype=np.float64),
        frame_id=frame_id,
        exact=True,
    )


def sparse_hz_lift_bounds(
    hz: SparseHZono,
    bounds: Bounds,
    slots,
    n_cont: int,
    *,
    output_equalities=None,
    equality_rhs=None,
    output_inequalities=None,
    inequality_rhs=None,
) -> SparseHZono:
    lb, ub = _bounds_to_numpy(bounds)
    if not np.isfinite(lb).all() or not np.isfinite(ub).all():
        raise ValueError("cannot lift non-finite bounds into a sparse HZ")
    if np.any(lb > ub):
        raise ValueError("cannot lift inconsistent bounds into a sparse HZ")
    slots = np.asarray(slots, dtype=np.int64).reshape(-1)
    if slots.size != lb.size:
        raise ValueError(f"sparse lift slot mismatch: {slots.size} vs {lb.size}")

    rows = np.arange(lb.size, dtype=np.int64)
    center = (lb + ub) * 0.5
    radius = (ub - lb) * 0.5
    nonzero = radius != 0.0
    Gc = sp.csr_matrix(
        (radius[nonzero], (rows[nonzero], slots[nonzero])),
        shape=(lb.size, int(n_cont)),
        dtype=np.float64,
    )
    Gb = sparse_empty(lb.size, hz.n_bin)

    def lift_rows(matrix, rhs, kind):
        if matrix is None:
            return (
                sparse_empty(0, int(n_cont)),
                sparse_empty(0, hz.n_bin),
                np.zeros(0, dtype=np.float64),
            )
        matrix = _as_csr(matrix)
        if matrix.shape[1] != lb.size:
            raise ValueError(
                f"sparse output {kind} shape mismatch: {matrix.shape} vs {lb.size}"
            )
        rhs = (
            np.zeros(matrix.shape[0], dtype=np.float64)
            if rhs is None
            else np.asarray(rhs, dtype=np.float64).reshape(-1)
        )
        if rhs.size != matrix.shape[0]:
            raise ValueError(f"sparse output {kind} rhs shape mismatch")
        return (
            (matrix @ Gc).tocsr(),
            sparse_empty(matrix.shape[0], hz.n_bin),
            rhs - np.asarray(matrix @ center).reshape(-1),
        )

    Ac, Ab, b = lift_rows(output_equalities, equality_rhs, "equality")
    Auc, Aub, upper = lift_rows(
        output_inequalities, inequality_rhs, "inequality"
    )
    return SparseHZono(
        c=center,
        Gc=Gc,
        Gb=Gb,
        Ac=Ac,
        Ab=Ab,
        b=b,
        Auc=Auc,
        Aub=Aub,
        ub=upper,
        frame_id=hz.frame_id,
        exact=False,
    )


def sparse_hz_reframe_point(hz: SparseHZono, target: SparseHZono) -> SparseHZono:
    if not sparse_hz_is_point(hz) or hz.n_eq or hz.n_ineq:
        raise ValueError("only unconstrained point HZs can be reframed")
    return replace(
        hz,
        Gc=sparse_empty(hz.n_out, target.n_cont),
        Gb=sparse_empty(hz.n_out, target.n_bin),
        Ac=sparse_empty(0, target.n_cont),
        Ab=sparse_empty(0, target.n_bin),
        Auc=sparse_empty(0, target.n_cont),
        Aub=sparse_empty(0, target.n_bin),
        frame_id=target.frame_id,
        exact=hz.exact and target.exact,
    )


def sparse_hz_linear(hz: SparseHZono, W, bias=None) -> SparseHZono:
    Wsp = _as_csr(W)
    if Wsp.shape[1] != hz.n_out:
        raise ValueError(f"linear shape mismatch: W={Wsp.shape}, n_out={hz.n_out}")
    b = (
        np.zeros(Wsp.shape[0], dtype=np.float64)
        if bias is None
        else np.asarray(bias, dtype=np.float64).reshape(-1)
    )
    if b.size != Wsp.shape[0]:
        raise ValueError(f"bias shape mismatch: bias={b.size}, rows={Wsp.shape[0]}")
    Gc = (Wsp @ hz.Gc).tocsr()
    Gb = (Wsp @ hz.Gb).tocsr() if hz.n_bin else sparse_empty(Wsp.shape[0], 0)
    Gc.eliminate_zeros()
    Gb.eliminate_zeros()
    return SparseHZono(
        c=np.asarray(Wsp @ hz.c).reshape(-1) + b,
        Gc=Gc,
        Gb=Gb,
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        Auc=hz.Auc,
        Aub=hz.Aub,
        ub=hz.ub,
        frame_id=hz.frame_id,
        exact=hz.exact,
    )


def sparse_hz_intersect_bounds(hz: SparseHZono, bounds: Bounds) -> SparseHZono:
    lb, ub = _bounds_to_numpy(bounds)
    if lb.size != hz.n_out or not np.isfinite(lb).all() or not np.isfinite(ub).all():
        raise ValueError("sparse HZ box intersection requires finite matching bounds")
    if np.any(lb > ub):
        raise ValueError("sparse HZ box intersection received inconsistent bounds")
    radius = _sparse_generator_radius(hz.Gc, hz.Gb)
    upper_rows = np.flatnonzero(ub < hz.c + radius)
    lower_rows = np.flatnonzero(lb > hz.c - radius)
    return replace(
        hz,
        Auc=sp.vstack(
            [hz.Auc, hz.Gc[upper_rows], -hz.Gc[lower_rows]], format="csr"
        ),
        Aub=sp.vstack(
            [hz.Aub, hz.Gb[upper_rows], -hz.Gb[lower_rows]], format="csr"
        ),
        ub=np.concatenate(
            [
                hz.ub,
                ub[upper_rows] - hz.c[upper_rows],
                hz.c[lower_rows] - lb[lower_rows],
            ]
        ),
        exact=False,
    )


def sparse_hz_add_const(hz: SparseHZono, bias) -> SparseHZono:
    b = np.asarray(
        bias.detach().cpu().double().numpy() if isinstance(bias, torch.Tensor) else bias,
        dtype=np.float64,
    ).reshape(-1)
    if b.size == 1:
        b = np.full(hz.n_out, float(b[0]), dtype=np.float64)
    if b.size != hz.n_out:
        raise ValueError(f"bias shape mismatch: bias={b.size}, n_out={hz.n_out}")
    return SparseHZono(
        c=hz.c + b,
        Gc=hz.Gc,
        Gb=hz.Gb,
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        Auc=hz.Auc,
        Aub=hz.Aub,
        ub=hz.ub,
        frame_id=hz.frame_id,
        exact=hz.exact,
    )


def sparse_hz_scale(hz: SparseHZono, scale) -> SparseHZono:
    s = np.asarray(
        scale.detach().cpu().double().numpy() if isinstance(scale, torch.Tensor) else scale,
        dtype=np.float64,
    ).reshape(-1)
    if s.size == 1:
        s = np.full(hz.n_out, float(s[0]), dtype=np.float64)
    if s.size != hz.n_out:
        raise ValueError(f"scale shape mismatch: scale={s.size}, n_out={hz.n_out}")
    D = sp.diags(s, offsets=0, shape=(hz.n_out, hz.n_out), format="csr")
    return sparse_hz_linear(hz, D, None)


def sparse_hz_gather_rows(hz: SparseHZono, rows) -> SparseHZono:
    idx = np.asarray(rows, dtype=np.int64).reshape(-1)
    return SparseHZono(
        c=hz.c[idx],
        Gc=hz.Gc[idx].tocsr(),
        Gb=hz.Gb[idx].tocsr() if hz.n_bin else sparse_empty(idx.size, 0),
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        Auc=hz.Auc,
        Aub=hz.Aub,
        ub=hz.ub,
        frame_id=hz.frame_id,
        exact=hz.exact,
    )


def sparse_hz_reduce_sum_rows(hz: SparseHZono, rows, n_out: int) -> SparseHZono:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if rows.size != hz.n_out:
        raise ValueError(f"reduce rows mismatch: rows={rows.size}, n_out={hz.n_out}")
    src = np.arange(rows.size, dtype=np.int64)
    R = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, src)),
        shape=(int(n_out), hz.n_out),
    )
    return sparse_hz_linear(hz, R, None)


def _sparse_same_frame(parts) -> bool:
    frames = [p.frame_id for p in parts]
    return all(f is not None for f in frames) and all(f == frames[0] for f in frames)


def _sparse_vstack(mats, cols: int):
    mats = [sparse_pad_cols(m, cols) for m in mats if m.shape[0]]
    return sp.vstack(mats, format="csr") if mats else sparse_empty(0, cols)


def _sparse_concat_arrays(arrs):
    arrs = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrs if np.asarray(a).size]
    return np.concatenate(arrs) if arrs else np.zeros(0, dtype=np.float64)


def _sparse_constraint_prefix(Ac_x, Ab_x, b_x, Ac_y, Ab_y, b_y) -> int:
    count = min(int(Ac_x.shape[0]), int(Ac_y.shape[0]))
    if count == 0:
        return 0
    dc = (Ac_x[:count] - Ac_y[:count]).tocsr()
    db = (Ab_x[:count] - Ab_y[:count]).tocsr()
    dc.eliminate_zeros()
    db.eliminate_zeros()
    same = (
        (np.asarray(dc.getnnz(axis=1)).reshape(-1) == 0)
        & (np.asarray(db.getnnz(axis=1)).reshape(-1) == 0)
        & (np.asarray(b_x[:count]) == np.asarray(b_y[:count]))
    )
    different = np.flatnonzero(~same)
    return int(different[0]) if different.size else count


def _sparse_merge_constraints(parts, c_name, b_name, rhs_name, n_cont, n_bin):
    base_index = max(
        range(len(parts)), key=lambda i: getattr(parts[i], c_name).shape[0]
    )
    base = parts[base_index]
    Ac_base = sparse_pad_cols(getattr(base, c_name), n_cont)
    Ab_base = sparse_pad_cols(getattr(base, b_name), n_bin)
    rhs_base = np.asarray(getattr(base, rhs_name), dtype=np.float64).reshape(-1)
    Ac_parts, Ab_parts, rhs_parts = [Ac_base], [Ab_base], [rhs_base]
    for index, part in enumerate(parts):
        if index == base_index:
            continue
        Ac = sparse_pad_cols(getattr(part, c_name), n_cont)
        Ab = sparse_pad_cols(getattr(part, b_name), n_bin)
        rhs = np.asarray(getattr(part, rhs_name), dtype=np.float64).reshape(-1)
        prefix = _sparse_constraint_prefix(
            Ac_base, Ab_base, rhs_base, Ac, Ab, rhs
        )
        Ac_parts.append(Ac[prefix:])
        Ab_parts.append(Ab[prefix:])
        rhs_parts.append(rhs[prefix:])
    return (
        _sparse_vstack(Ac_parts, n_cont),
        _sparse_vstack(Ab_parts, n_bin),
        _sparse_concat_arrays(rhs_parts),
    )


def _sparse_merge_all_constraints(parts, n_cont, n_bin):
    return (
        *_sparse_merge_constraints(parts, "Ac", "Ab", "b", n_cont, n_bin),
        *_sparse_merge_constraints(parts, "Auc", "Aub", "ub", n_cont, n_bin),
    )


def _sparse_truncate_rows(matrix, max_terms: int):
    matrix = matrix.tocsr()
    omitted_radius = np.zeros(matrix.shape[0], dtype=np.float64)
    if max_terms <= 0 or np.diff(matrix.indptr).max(initial=0) <= max_terms:
        return matrix, omitted_radius
    rows, cols, data = [], [], []
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        row_data = matrix.data[start:stop]
        row_cols = matrix.indices[start:stop]
        if row_data.size > max_terms:
            selected = np.argpartition(np.abs(row_data), -max_terms)[-max_terms:]
            omitted = np.ones(row_data.size, dtype=bool)
            omitted[selected] = False
            omitted_radius[row] = np.abs(row_data[omitted]).sum()
            row_data, row_cols = row_data[selected], row_cols[selected]
        rows.extend([row] * row_data.size)
        cols.extend(row_cols.tolist())
        data.extend(row_data.tolist())
    return sp.csr_matrix(
        (data, (rows, cols)), shape=matrix.shape, dtype=np.float64
    ), omitted_radius


def _sparse_add_error_generators(Gc, radius, slots, n_cont: int):
    radius = np.asarray(radius, dtype=np.float64).reshape(-1)
    slots = np.asarray(slots, dtype=np.int64).reshape(-1)
    nonzero = np.flatnonzero(radius != 0.0)
    if nonzero.size:
        Gc = Gc + sp.csr_matrix(
            (radius[nonzero], (nonzero, slots[nonzero])),
            shape=(radius.size, int(n_cont)),
        )
    return Gc.tocsr()


def sparse_hz_matmul_relaxation(
    x: SparseHZono,
    y: SparseHZono,
    x_bounds: Bounds,
    y_bounds: Bounds,
    output_bounds: Bounds,
    x_rows,
    y_rows,
    slots,
    n_cont: int,
) -> SparseHZono:
    if not _sparse_same_frame([x, y]):
        raise ValueError("sparse variable MatMul requires one shared frame")
    x_rows = np.asarray(x_rows, dtype=np.int64)
    y_rows = np.asarray(y_rows, dtype=np.int64)
    if x_rows.ndim != 2 or x_rows.shape != y_rows.shape:
        raise ValueError("MatMul term rows must have matching shapes")
    lbx, ubx = _bounds_to_numpy(x_bounds)
    lby, uby = _bounds_to_numpy(y_bounds)
    slots = np.asarray(slots, dtype=np.int64).reshape(-1)
    n_out, reduction = x_rows.shape
    if n_out != output_bounds.lb.numel() or slots.size != n_out:
        raise ValueError("MatMul output rows, bounds, and slots do not match")
    if x_rows.size and (x_rows.min() < 0 or x_rows.max() >= x.n_out):
        raise ValueError("MatMul left term row is out of range")
    if y_rows.size and (y_rows.min() < 0 or y_rows.max() >= y.n_out):
        raise ValueError("MatMul right term row is out of range")

    n_bin = max(x.n_bin, y.n_bin)
    xp = sparse_hz_pad_frame(x, int(n_cont), n_bin)
    yp = sparse_hz_pad_frame(y, int(n_cont), n_bin)
    out_idx = np.arange(n_out, dtype=np.int64)
    Ac, Ab, b, Auc, Aub, ub = _sparse_merge_all_constraints(
        [xp, yp], int(n_cont), n_bin
    )
    lx, ux = lbx[x_rows], ubx[x_rows]
    ly, uy = lby[y_rows], uby[y_rows]
    cx, cy = (lx + ux) * 0.5, (ly + uy) * 0.5
    term_rows = np.repeat(out_idx, reduction)
    X0 = sp.csr_matrix(
        (cy.reshape(-1), (term_rows, x_rows.reshape(-1))),
        shape=(n_out, x.n_out),
    )
    Y0 = sp.csr_matrix(
        (cx.reshape(-1), (term_rows, y_rows.reshape(-1))),
        shape=(n_out, y.n_out),
    )
    center = (
        np.asarray(X0 @ xp.c).reshape(-1)
        + np.asarray(Y0 @ yp.c).reshape(-1)
        - np.sum(cx * cy, axis=1)
    )
    Gc = (X0 @ xp.Gc + Y0 @ yp.Gc).tocsr()
    Gc, omitted = _sparse_truncate_rows(Gc, 256)
    radius = np.nextafter(
        np.sum((ux - lx) * (uy - ly), axis=1) * 0.25 + omitted,
        np.inf,
    )
    Gc = _sparse_add_error_generators(Gc, radius, slots, n_cont)
    Gb = (
        (X0 @ xp.Gb + Y0 @ yp.Gb).tocsr()
        if n_bin else sparse_empty(n_out, 0)
    )
    out = SparseHZono(
        c=center,
        Gc=Gc,
        Gb=Gb,
        Ac=Ac,
        Ab=Ab,
        b=b,
        Auc=Auc,
        Aub=Aub,
        ub=ub,
        frame_id=x.frame_id,
        exact=False,
    )
    return sparse_hz_intersect_bounds(out, output_bounds)


def softmax_ratio_weighted_extreme(
    values,
    probability_lower,
    probability_upper,
    groups,
    score_differences,
    minimize: bool,
    score_lower=None,
    score_upper=None,
):
    order = np.argsort(values, axis=1)
    if not minimize:
        order = order[:, ::-1]
    ordered_values = np.take_along_axis(values, order, axis=1)
    ordered_lower = np.take_along_axis(probability_lower, order, axis=1)
    ordered_upper = np.take_along_axis(probability_upper, order, axis=1)
    selected = ordered_lower.copy()
    remaining = np.maximum(0.0, 1.0 - selected.sum(axis=1))
    for column in range(values.shape[1]):
        added = np.minimum(
            remaining, ordered_upper[:, column] - ordered_lower[:, column]
        )
        selected[:, column] += added
        remaining -= added
    extrema = np.min(values, axis=1) if minimize else np.max(values, axis=1)
    result = np.sum(selected * ordered_values, axis=1)
    result = np.where(remaining <= 1e-10, result, extrema)
    result = np.nextafter(result, -np.inf if minimize else np.inf)

    if score_lower is None or score_upper is None:
        return result
    score_lower = np.asarray(score_lower, dtype=np.float64)
    score_upper = np.asarray(score_upper, dtype=np.float64)
    if score_lower.shape != values.shape or score_upper.shape != values.shape:
        return result
    objective = values if minimize else -values
    order = np.argsort(objective, axis=1)
    ordered_objective = np.take_along_axis(objective, order, axis=1).astype(
        np.longdouble
    )
    ordered_lower = np.take_along_axis(score_lower, order, axis=1).astype(
        np.longdouble
    )
    ordered_upper = np.take_along_axis(score_upper, order, axis=1).astype(
        np.longdouble
    )
    shift = np.max(ordered_upper, axis=1, keepdims=True)
    exp_lower = np.exp(ordered_lower - shift)
    exp_upper = np.exp(ordered_upper - shift)
    prefix_num = np.concatenate(
        [
            np.zeros((values.shape[0], 1), dtype=np.longdouble),
            np.cumsum(ordered_objective * exp_upper, axis=1),
        ],
        axis=1,
    )
    prefix_den = np.concatenate(
        [
            np.zeros((values.shape[0], 1), dtype=np.longdouble),
            np.cumsum(exp_upper, axis=1),
        ],
        axis=1,
    )
    lower_num = np.concatenate(
        [
            np.zeros((values.shape[0], 1), dtype=np.longdouble),
            np.cumsum(ordered_objective * exp_lower, axis=1),
        ],
        axis=1,
    )
    lower_den = np.concatenate(
        [
            np.zeros((values.shape[0], 1), dtype=np.longdouble),
            np.cumsum(exp_lower, axis=1),
        ],
        axis=1,
    )
    candidates = (prefix_num + lower_num[:, -1:] - lower_num) / (
        prefix_den + lower_den[:, -1:] - lower_den
    )
    vertex = np.asarray(np.min(candidates, axis=1), dtype=np.float64)
    vertex -= 32.0 * np.finfo(np.float64).eps * (1.0 + np.abs(vertex))
    if minimize:
        return np.maximum(result, vertex)
    return np.minimum(result, -vertex)


def _softmax_taylor_coefficients(
    score_bounds: Bounds,
    score_rows,
    groups,
    weights,
    score_differences,
    probability_lower,
    probability_upper,
):
    score_rows = np.asarray(score_rows, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64)
    n_out, reduction = score_rows.shape
    score_lb, score_ub = _bounds_to_numpy(score_bounds)
    score_lb, score_ub = score_lb[score_rows], score_ub[score_rows]
    score_center = 0.5 * (score_lb + score_ub)
    reference = np.argmax(score_center, axis=1)
    rows = np.arange(n_out, dtype=np.int64)
    difference_lb = score_lb - score_ub[rows, reference, None]
    difference_ub = score_ub - score_lb[rows, reference, None]
    if score_differences is not None:
        diff_lb, diff_ub = (
            np.asarray(part, dtype=np.float64) for part in score_differences
        )
        if (
            diff_lb.ndim == 3
            and diff_lb.shape == diff_ub.shape
            and diff_lb.shape[0] > int(groups.max(initial=-1))
            and diff_lb.shape[1:] == (reduction, reduction)
        ):
            difference_lb = np.maximum(
                difference_lb, diff_lb[groups, reference]
            )
            difference_ub = np.minimum(
                difference_ub, diff_ub[groups, reference]
            )
    difference_center = 0.5 * (difference_lb + difference_ub)
    difference_center[rows, reference] = 0.0
    difference_radius = np.maximum(
        difference_center - np.minimum(difference_lb, difference_center),
        np.maximum(difference_ub, difference_center) - difference_center,
    )
    difference_radius[rows, reference] = 0.0

    shifted = difference_center - np.max(difference_center, axis=1, keepdims=True)
    probability_center = np.exp(shifted)
    probability_center /= probability_center.sum(axis=1, keepdims=True)
    weighted_center = np.sum(probability_center * weights, axis=1)
    affine = probability_center * (weights - weighted_center[:, None])
    intercept = weighted_center - np.sum(affine * difference_center, axis=1)

    upper_shift = score_ub[:, None, :] - score_lb[:, :, None]
    lower_shift = score_lb[:, None, :] - score_ub[:, :, None]
    diagonal = np.arange(reduction)
    upper_shift[:, diagonal, diagonal] = 0.0
    lower_shift[:, diagonal, diagonal] = 0.0

    def inverse_exp_sum(shifts):
        maximum = np.max(shifts, axis=2, keepdims=True)
        log_denominator = maximum[:, :, 0] + np.log(
            np.exp(shifts - maximum).sum(axis=2)
        )
        return np.exp(-log_denominator)

    raw_probability_lower = np.nextafter(inverse_exp_sum(upper_shift), 0.0)
    raw_probability_upper = np.nextafter(inverse_exp_sum(lower_shift), np.inf)
    probability_lower = np.asarray(probability_lower, dtype=np.float64)
    probability_upper = np.asarray(probability_upper, dtype=np.float64)
    tightened_lower = np.maximum(raw_probability_lower, probability_lower)
    tightened_upper = np.minimum(raw_probability_upper, probability_upper)
    consistent = tightened_lower <= tightened_upper
    raw_probability_lower = np.where(
        consistent, tightened_lower, raw_probability_lower
    )
    raw_probability_upper = np.where(
        consistent, tightened_upper, raw_probability_upper
    )

    def weighted_extreme(coefficients, minimize):
        return softmax_ratio_weighted_extreme(
            coefficients,
            raw_probability_lower,
            raw_probability_upper,
            groups,
            score_differences,
            minimize,
            score_lb,
            score_ub,
        )

    weighted_lower = weighted_extreme(weights, True)
    weighted_upper = weighted_extreme(weights, False)
    candidates = np.stack(
        [
            raw_probability_lower,
            raw_probability_upper,
            np.clip(0.25, raw_probability_lower, raw_probability_upper),
            np.clip(0.5, raw_probability_lower, raw_probability_upper),
        ],
        axis=0,
    )
    diagonal_factor = np.max(
        candidates * np.abs(1.0 - 2.0 * candidates), axis=0
    )
    weight_pair = weights[:, :, None] + weights[:, None, :]
    hessian = (
        raw_probability_upper[:, :, None]
        * raw_probability_upper[:, None, :]
        * np.maximum(
            np.abs(weight_pair - 2.0 * weighted_lower[:, None, None]),
            np.abs(weight_pair - 2.0 * weighted_upper[:, None, None]),
        )
    )
    hessian[:, diagonal, diagonal] = diagonal_factor * np.maximum(
        np.abs(weights - weighted_lower[:, None]),
        np.abs(weights - weighted_upper[:, None]),
    )
    taylor_radius = 0.5 * np.einsum(
        "nij,ni,nj->n", hessian, difference_radius, difference_radius
    )
    delta_lower = difference_lb - difference_center
    delta_upper = difference_ub - difference_center
    delta_span = np.max(delta_upper, axis=1) - np.min(delta_lower, axis=1)
    weight_span = np.max(weights, axis=1) - np.min(weights, axis=1)
    directional_radius = np.nextafter(
        0.125 * weight_span * np.square(delta_span), np.inf
    )
    taylor_radius = np.minimum(taylor_radius, directional_radius)

    valid_log = (
        np.all(raw_probability_lower > 0.0, axis=1)
        & np.all(raw_probability_upper >= raw_probability_lower, axis=1)
    )
    log_lower = np.log(np.maximum(raw_probability_lower, np.finfo(float).tiny))
    log_upper = np.log(np.maximum(raw_probability_upper, np.finfo(float).tiny))
    probability_width = raw_probability_upper - raw_probability_lower
    secant_slope = np.divide(
        log_upper - log_lower,
        probability_width,
        out=1.0 / np.maximum(raw_probability_lower, np.finfo(float).tiny),
        where=probability_width > 1e-12,
    )
    secant_intercept = log_lower - secant_slope * raw_probability_lower
    tangent_point = np.clip(
        probability_center, raw_probability_lower, raw_probability_upper
    )
    tangent_slope = 1.0 / tangent_point
    tangent_intercept = np.log(tangent_point) - 1.0
    log_coefficient = -affine
    upper_uses_tangent = log_coefficient >= 0.0
    upper_slope = np.where(upper_uses_tangent, tangent_slope, secant_slope)
    upper_intercept = np.where(
        upper_uses_tangent, tangent_intercept, secant_intercept
    )
    lower_slope = np.where(upper_uses_tangent, secant_slope, tangent_slope)
    lower_intercept = np.where(
        upper_uses_tangent, secant_intercept, tangent_intercept
    )
    upper_coefficients = weights + log_coefficient * upper_slope
    lower_coefficients = weights + log_coefficient * lower_slope
    residual_upper = (
        np.sum(log_coefficient * upper_intercept, axis=1)
        + weighted_extreme(upper_coefficients, False)
        - intercept
    )
    residual_lower = (
        np.sum(log_coefficient * lower_intercept, axis=1)
        + weighted_extreme(lower_coefficients, True)
        - intercept
    )
    guard = 1e-10 * (
        1.0 + np.maximum(np.abs(residual_lower), np.abs(residual_upper))
    )
    residual_lower -= guard
    residual_upper += guard
    error_lower = np.maximum(-taylor_radius, residual_lower)
    error_upper = np.minimum(taylor_radius, residual_upper)
    consistent = valid_log & (error_lower <= error_upper)
    error_lower = np.where(consistent, error_lower, -taylor_radius)
    error_upper = np.where(consistent, error_upper, taylor_radius)
    return affine, intercept, error_lower, error_upper


def _attention_score_affine(
    context,
    affine,
    probability_rows,
    n_cont: int,
    n_bin: int,
):
    if not isinstance(context, dict):
        return None
    q = context.get("q_hz")
    k = context.get("k_hz")
    if not isinstance(q, SparseHZono) or not isinstance(k, SparseHZono):
        return None
    if not _sparse_same_frame([q, k]):
        return None

    probability_rows = np.asarray(probability_rows, dtype=np.int64)
    affine = np.asarray(affine, dtype=np.float64)
    q_rows = np.asarray(context.get("q_rows"), dtype=np.int64)
    k_rows = np.asarray(context.get("k_rows"), dtype=np.int64)
    scale = np.asarray(context.get("scale"), dtype=np.float64).reshape(-1)
    shift = np.asarray(context.get("shift"), dtype=np.float64).reshape(-1)
    if (
        probability_rows.ndim != 2
        or affine.shape != probability_rows.shape
        or q_rows.ndim != 2
        or q_rows.shape != k_rows.shape
        or q_rows.shape[0] != scale.size
        or shift.size != scale.size
        or probability_rows.size == 0
        or probability_rows.min() < 0
        or probability_rows.max() >= q_rows.shape[0]
    ):
        return None

    selected_q = q_rows[probability_rows]
    selected_k = k_rows[probability_rows]
    if not np.all(selected_q == selected_q[:, :1, :]):
        return None
    n_out, width = probability_rows.shape
    reduction = q_rows.shape[1]
    coefficients = affine * scale[probability_rows]
    q_term_rows = selected_q[:, 0, :]
    mixed_rows = np.broadcast_to(
        np.arange(n_out * reduction, dtype=np.int64).reshape(n_out, 1, reduction),
        (n_out, width, reduction),
    )
    mixed_coefficients = np.broadcast_to(
        coefficients[:, :, None], (n_out, width, reduction)
    )
    key_operator = sp.csr_matrix(
        (
            mixed_coefficients.reshape(-1),
            (mixed_rows.reshape(-1), selected_k.reshape(-1)),
        ),
        shape=(n_out * reduction, k.n_out),
    )

    kp = sparse_hz_pad_frame(k, int(n_cont), n_bin)
    qp = sparse_hz_pad_frame(q, int(n_cont), n_bin)
    mixed_center = np.asarray(key_operator @ kp.c).reshape(n_out, reduction)
    mixed_Gc = (key_operator @ kp.Gc).tocsr()
    mixed_Gb = (
        (key_operator @ kp.Gb).tocsr()
        if n_bin else sparse_empty(n_out * reduction, 0)
    )
    k_bounds = context.get("k_bounds")
    q_bounds = context.get("q_bounds")
    if not isinstance(k_bounds, Bounds) or not isinstance(q_bounds, Bounds):
        return None
    k_lb, k_ub = _bounds_to_numpy(k_bounds)
    k_lb, k_ub = k_lb[selected_k], k_ub[selected_k]
    positive = coefficients[:, :, None] >= 0.0
    interval_lower = np.sum(
        mixed_coefficients * np.where(positive, k_lb, k_ub), axis=1
    )
    interval_upper = np.sum(
        mixed_coefficients * np.where(positive, k_ub, k_lb), axis=1
    )
    mixed_radius = _sparse_generator_radius(mixed_Gc, mixed_Gb).reshape(
        n_out, reduction
    )
    mixed_lower = np.maximum(interval_lower, mixed_center - mixed_radius)
    mixed_upper = np.minimum(interval_upper, mixed_center + mixed_radius)
    consistent = mixed_lower <= mixed_upper
    mixed_lower = np.where(consistent, mixed_lower, interval_lower)
    mixed_upper = np.where(consistent, mixed_upper, interval_upper)

    q_lb, q_ub = _bounds_to_numpy(q_bounds)
    q_lb, q_ub = q_lb[q_term_rows], q_ub[q_term_rows]
    q_center = qp.c[q_term_rows]
    q_radius = _sparse_generator_radius(qp.Gc, qp.Gb)[q_term_rows]
    q_lower = np.maximum(q_lb, q_center - q_radius)
    q_upper = np.minimum(q_ub, q_center + q_radius)
    consistent = q_lower <= q_upper
    q_lower = np.where(consistent, q_lower, q_lb)
    q_upper = np.where(consistent, q_upper, q_ub)

    q_mid = 0.5 * (q_lower + q_upper)
    k_mid = 0.5 * (mixed_lower + mixed_upper)
    product_error = np.nextafter(
        np.sum(
            0.25 * (q_upper - q_lower) * (mixed_upper - mixed_lower),
            axis=1,
        ),
        np.inf,
    )
    output_rows = np.repeat(np.arange(n_out, dtype=np.int64), reduction)
    q_operator = sp.csr_matrix(
        (k_mid.reshape(-1), (output_rows, q_term_rows.reshape(-1))),
        shape=(n_out, q.n_out),
    )
    key_output_rows = np.broadcast_to(
        np.arange(n_out, dtype=np.int64)[:, None, None], selected_k.shape
    )
    key_operator = sp.csr_matrix(
        (
            (mixed_coefficients * q_mid[:, None, :]).reshape(-1),
            (key_output_rows.reshape(-1), selected_k.reshape(-1)),
        ),
        shape=(n_out, k.n_out),
    )
    key_center = np.asarray(key_operator @ kp.c).reshape(-1)
    key_Gc = (key_operator @ kp.Gc).tocsr()
    key_Gb = (
        (key_operator @ kp.Gb).tocsr()
        if n_bin else sparse_empty(n_out, 0)
    )
    center = (
        np.asarray(q_operator @ qp.c).reshape(-1)
        + key_center
        - np.sum(q_mid * k_mid, axis=1)
        + np.sum(affine * shift[probability_rows], axis=1)
    )
    Gc = (q_operator @ qp.Gc + key_Gc).tocsr()
    Gb = (
        (q_operator @ qp.Gb + key_Gb).tocsr()
        if n_bin else sparse_empty(n_out, 0)
    )
    return center, Gc, Gb, product_error, (qp, kp)


def _sparse_hz_softmax_value_fused(
    scores: SparseHZono,
    values: SparseHZono,
    score_bounds: Bounds,
    probability_rows,
    value_rows,
    slots,
    n_cont: int,
    p0,
    probability_lower,
    probability_upper,
    mid,
    cross_radius,
    score_differences,
    score_context,
) -> SparseHZono:
    probability_rows = np.asarray(probability_rows, dtype=np.int64)
    value_rows = np.asarray(value_rows, dtype=np.int64)
    n_out, reduction = probability_rows.shape
    groups = probability_rows[:, 0] // reduction
    affine, intercept, error_lower, error_upper = _softmax_taylor_coefficients(
        score_bounds,
        probability_rows,
        groups,
        mid,
        score_differences,
        probability_lower,
        probability_upper,
    )
    error_lower -= cross_radius
    error_upper += cross_radius
    error_center = 0.5 * (error_lower + error_upper)
    error_radius = np.nextafter(0.5 * (error_upper - error_lower), np.inf)

    n_bin = max(scores.n_bin, values.n_bin)
    value_padded = sparse_hz_pad_frame(values, int(n_cont), n_bin)
    direct_score = _attention_score_affine(
        score_context, affine, probability_rows, int(n_cont), n_bin
    )
    output_rows = np.repeat(np.arange(n_out, dtype=np.int64), reduction)
    if direct_score is None:
        score_padded = sparse_hz_pad_frame(scores, int(n_cont), n_bin)
        score_operator = sp.csr_matrix(
            (affine.reshape(-1), (output_rows, probability_rows.reshape(-1))),
            shape=(n_out, scores.n_out),
        )
        score_center = np.asarray(score_operator @ score_padded.c).reshape(-1)
        score_Gc = (score_operator @ score_padded.Gc).tocsr()
        score_Gb = (
            (score_operator @ score_padded.Gb).tocsr()
            if n_bin else sparse_empty(n_out, 0)
        )
        score_error = np.zeros(n_out, dtype=np.float64)
        constraint_parts = [score_padded, value_padded]
    else:
        score_center, score_Gc, score_Gb, score_error, score_parts = direct_score
        constraint_parts = [*score_parts, value_padded]
    value_operator = sp.csr_matrix(
        (p0.reshape(-1), (output_rows, value_rows.reshape(-1))),
        shape=(n_out, values.n_out),
    )
    error_radius = np.nextafter(error_radius + score_error, np.inf)
    center = (
        score_center
        + np.asarray(value_operator @ value_padded.c).reshape(-1)
        + intercept
        - np.sum(p0 * mid, axis=1)
        + error_center
    )
    Gc = (score_Gc + value_operator @ value_padded.Gc).tocsr()
    Gc = _sparse_add_error_generators(Gc, error_radius, slots, n_cont)
    Gb = (
        (score_Gb + value_operator @ value_padded.Gb).tocsr()
        if n_bin else sparse_empty(n_out, 0)
    )
    Ac, Ab, b, Auc, Aub, ub = _sparse_merge_all_constraints(
        constraint_parts, int(n_cont), n_bin
    )
    return SparseHZono(
        c=center,
        Gc=Gc,
        Gb=Gb,
        Ac=Ac,
        Ab=Ab,
        b=b,
        Auc=Auc,
        Aub=Aub,
        ub=ub,
        frame_id=value_padded.frame_id,
        exact=False,
    )


def _softmax_value_cross_radius(
    probability_lower,
    probability_upper,
    reference,
    value_radius,
    score_lower,
    score_upper,
    score_differences,
    groups,
):
    width = int(probability_lower.shape[1])
    if width > 8:
        positive = np.maximum(probability_upper - reference, 0.0)
        negative = np.maximum(reference - probability_lower, 0.0)
        mass = np.minimum(positive.sum(axis=1), negative.sum(axis=1))

        def radius_for(weights):
            independent = np.sum(
                np.maximum(
                    np.abs(probability_lower - reference),
                    np.abs(probability_upper - reference),
                )
                * weights,
                axis=1,
            )
            order = np.argsort(-weights, axis=1)
            sorted_weights = np.take_along_axis(weights, order, axis=1)
            sorted_positive = np.take_along_axis(positive, order, axis=1)
            sorted_negative = np.take_along_axis(negative, order, axis=1)

            def weighted_mass(capacity, target):
                before = np.cumsum(capacity, axis=1) - capacity
                used = np.clip(target[:, None] - before, 0.0, capacity)
                return np.sum(used * sorted_weights, axis=1)

            relaxed = np.nextafter(
                weighted_mass(sorted_positive, mass)
                + weighted_mass(sorted_negative, mass),
                np.inf,
            )
            capacity = positive + negative
            switches = np.divide(
                weights * (positive - negative),
                capacity,
                out=np.zeros_like(weights),
                where=capacity > 0.0,
            )
            max_weight = np.max(weights, axis=1, keepdims=True)
            multipliers = np.concatenate(
                [-max_weight, switches, max_weight], axis=1
            )
            lower_side = negative[:, None, :] * (
                weights[:, None, :] + multipliers[:, :, None]
            )
            upper_side = positive[:, None, :] * (
                weights[:, None, :] - multipliers[:, :, None]
            )
            lagrangian = np.nextafter(
                np.min(
                    np.sum(np.maximum(lower_side, upper_side), axis=2), axis=1
                ),
                np.inf,
            )
            depth = min(width, 8)
            disjoint = np.full(weights.shape[0], -np.inf, dtype=np.float64)
            for assignment in range(1 << depth):
                positive_capacity = sorted_positive.copy()
                negative_capacity = sorted_negative.copy()
                positive_side = (
                    (assignment >> np.arange(depth, dtype=np.int64)) & 1
                ).astype(bool)
                positive_capacity[:, np.flatnonzero(~positive_side)] = 0.0
                negative_capacity[:, np.flatnonzero(positive_side)] = 0.0
                branch_mass = np.minimum(
                    positive_capacity.sum(axis=1),
                    negative_capacity.sum(axis=1),
                )
                branch = weighted_mass(
                    positive_capacity, branch_mass
                ) + weighted_mass(negative_capacity, branch_mass)
                disjoint = np.maximum(disjoint, branch)
            disjoint = np.nextafter(disjoint, np.inf)
            return np.minimum(
                independent, np.minimum(relaxed, np.minimum(disjoint, lagrangian))
            )

        return reference, radius_for(value_radius)

    signs = 2.0 * (
        (
            np.arange(1 << width, dtype=np.uint64)[:, None]
            >> np.arange(width, dtype=np.uint64)
        )
        & 1
    ).astype(np.float64) - 1.0
    weights = (value_radius[:, None, :] * signs[None, :, :]).reshape(-1, width)
    lower = np.repeat(probability_lower, signs.shape[0], axis=0)
    upper = np.repeat(probability_upper, signs.shape[0], axis=0)
    repeated_score_lower = np.repeat(score_lower, signs.shape[0], axis=0)
    repeated_score_upper = np.repeat(score_upper, signs.shape[0], axis=0)
    extrema = softmax_ratio_weighted_extreme(
        weights,
        lower,
        upper,
        np.repeat(np.asarray(groups, dtype=np.int64), signs.shape[0]),
        score_differences,
        False,
        repeated_score_lower,
        repeated_score_upper,
    ).reshape(probability_lower.shape[0], -1)
    signed_weights = weights.reshape(probability_lower.shape[0], -1, width)
    optimized = reference.copy()
    objective = np.zeros(width + 1, dtype=np.float64)
    objective[-1] = 1.0
    equality = np.zeros((1, width + 1), dtype=np.float64)
    equality[0, :width] = 1.0
    for row in range(probability_lower.shape[0]):
        inequalities = np.empty((signs.shape[0], width + 1), dtype=np.float64)
        inequalities[:, :width] = -signed_weights[row]
        inequalities[:, -1] = -1.0
        result = linprog(
            objective,
            A_ub=inequalities,
            b_ub=-extrema[row],
            A_eq=equality,
            b_eq=np.ones(1, dtype=np.float64),
            bounds=[
                *zip(probability_lower[row], probability_upper[row]),
                (0.0, None),
            ],
            method="highs-ds",
        )
        if result.success:
            optimized[row] = result.x[:width]
    radius = (
        extrema - np.einsum("nsi,ni->ns", signed_weights, optimized)
    ).max(axis=1)
    radius = np.nextafter(np.maximum(radius, 0.0), np.inf)
    independent = np.sum(
        np.maximum(
            np.abs(probability_lower - optimized),
            np.abs(probability_upper - optimized),
        )
        * value_radius,
        axis=1,
    )
    return optimized, np.minimum(independent, radius)


def sparse_hz_softmax_value_relaxation(
    probabilities: SparseHZono,
    values: SparseHZono,
    probability_bounds: Bounds,
    value_bounds: Bounds,
    probability_rows,
    value_rows,
    slots,
    n_cont: int,
    *,
    scores: SparseHZono,
    score_bounds: Bounds,
    score_differences=None,
    score_context=None,
) -> SparseHZono:
    if not _sparse_same_frame([probabilities, values, scores]):
        raise ValueError("Softmax-value operands require one shared frame")
    probability_rows = np.asarray(probability_rows, dtype=np.int64)
    value_rows = np.asarray(value_rows, dtype=np.int64)
    if probability_rows.ndim != 2 or probability_rows.shape != value_rows.shape:
        raise ValueError("Softmax-value term rows must have matching shapes")
    n_out, reduction = probability_rows.shape
    slots = np.asarray(slots, dtype=np.int64).reshape(-1)
    if slots.size != n_out:
        raise ValueError("Softmax-value output rows and slots do not match")
    if probability_rows.size and (
        probability_rows.min() < 0
        or probability_rows.max() >= probabilities.n_out
        or probability_rows.max() >= scores.n_out
    ):
        raise ValueError("Softmax probability row is out of range")
    probability_lb, probability_ub = _bounds_to_numpy(probability_bounds)
    value_lb, value_ub = _bounds_to_numpy(value_bounds)
    if value_rows.size and (
        value_rows.min() < 0 or value_rows.max() >= values.n_out
        or value_rows.max() >= value_lb.size
    ):
        raise ValueError("Softmax value row is out of range")
    if probability_rows.size and probability_rows.max() >= probability_lb.size:
        raise ValueError("Softmax probability bound row is out of range")
    mid = (value_lb[value_rows] + value_ub[value_rows]) * 0.5
    value_radius = (value_ub[value_rows] - value_lb[value_rows]) * 0.5
    probability_lower = probability_lb[probability_rows]
    probability_upper = probability_ub[probability_rows]
    probability_mid = (probability_lower + probability_upper) * 0.5
    delta = 1.0 - probability_mid.sum(axis=1)
    capacity = np.where(
        delta[:, None] >= 0.0,
        probability_upper - probability_mid,
        probability_mid - probability_lower,
    )
    order = np.argsort(value_radius, axis=1)
    ordered_capacity = np.take_along_axis(capacity, order, axis=1)
    before = np.cumsum(ordered_capacity, axis=1) - ordered_capacity
    selected = np.clip(
        np.abs(delta)[:, None] - before, 0.0, ordered_capacity
    )
    adjustment = np.zeros_like(probability_mid)
    np.put_along_axis(adjustment, order, selected, axis=1)
    reference = probability_mid + np.sign(delta)[:, None] * adjustment
    score_lb, score_ub = _bounds_to_numpy(score_bounds)
    score_lb, score_ub = score_lb[probability_rows], score_ub[probability_rows]
    groups = probability_rows[:, 0] // reduction
    reference, residual = _softmax_value_cross_radius(
        probability_lower,
        probability_upper,
        reference,
        value_radius,
        score_lb,
        score_ub,
        score_differences,
        groups,
    )
    return _sparse_hz_softmax_value_fused(
        scores,
        values,
        score_bounds,
        probability_rows,
        value_rows,
        slots,
        n_cont,
        reference,
        probability_lower,
        probability_upper,
        mid,
        residual,
        score_differences,
        score_context,
    )


def sparse_hz_concat(parts) -> SparseHZono:
    parts = list(parts)
    if not parts:
        raise ValueError("sparse_hz_concat requires at least one part")
    if not _sparse_same_frame(parts):
        raise ValueError("sparse concat requires one shared frame")
    n_cont = max(p.n_cont for p in parts)
    n_bin = max(p.n_bin for p in parts)
    padded = [sparse_hz_pad_frame(p, n_cont, n_bin) for p in parts]
    return SparseHZono(
        c=np.concatenate([p.c for p in padded]),
        Gc=sp.vstack([p.Gc for p in padded], format="csr"),
        Gb=sp.vstack([p.Gb for p in padded], format="csr"),
        Ac=_sparse_vstack([p.Ac for p in padded], n_cont),
        Ab=_sparse_vstack([p.Ab for p in padded], n_bin),
        b=_sparse_concat_arrays([p.b for p in padded]),
        Auc=_sparse_vstack([p.Auc for p in padded], n_cont),
        Aub=_sparse_vstack([p.Aub for p in padded], n_bin),
        ub=_sparse_concat_arrays([p.ub for p in padded]),
        frame_id=padded[0].frame_id,
        exact=all(p.exact for p in padded),
    )


def sparse_hz_add_same_frame(x: SparseHZono, y: SparseHZono) -> SparseHZono:
    if not _sparse_same_frame([x, y]):
        raise ValueError("sparse add requires one shared frame")
    if x.n_out != y.n_out:
        raise ValueError(f"sparse add shape mismatch: {x.n_out} vs {y.n_out}")
    n_cont = max(x.n_cont, y.n_cont)
    n_bin = max(x.n_bin, y.n_bin)
    xp = sparse_hz_pad_frame(x, n_cont, n_bin)
    yp = sparse_hz_pad_frame(y, n_cont, n_bin)
    Gc = (xp.Gc + yp.Gc).tocsr()
    Gb = (xp.Gb + yp.Gb).tocsr()
    Gc.eliminate_zeros()
    Gb.eliminate_zeros()
    return SparseHZono(
        c=xp.c + yp.c,
        Gc=Gc,
        Gb=Gb,
        Ac=_sparse_vstack([xp.Ac, yp.Ac], n_cont),
        Ab=_sparse_vstack([xp.Ab, yp.Ab], n_bin),
        b=_sparse_concat_arrays([xp.b, yp.b]),
        Auc=_sparse_vstack([xp.Auc, yp.Auc], n_cont),
        Aub=_sparse_vstack([xp.Aub, yp.Aub], n_bin),
        ub=_sparse_concat_arrays([xp.ub, yp.ub]),
        frame_id=xp.frame_id,
        exact=xp.exact and yp.exact,
    )


def sparse_hz_sub_same_frame(x: SparseHZono, y: SparseHZono) -> SparseHZono:
    return sparse_hz_add_same_frame(x, sparse_hz_scale(y, -1.0))


def sparse_hz_is_point(hz: SparseHZono, tol: float = 1e-12) -> bool:
    return (
        (hz.Gc.nnz == 0 or bool(np.all(np.abs(hz.Gc.data) <= tol)))
        and (hz.Gb.nnz == 0 or bool(np.all(np.abs(hz.Gb.data) <= tol)))
    )


def sparse_hz_fast_bounds(hz: SparseHZono) -> Bounds:
    abs_gc = np.asarray(np.abs(hz.Gc).sum(axis=1)).reshape(-1)
    abs_gb = np.asarray(np.abs(hz.Gb).sum(axis=1)).reshape(-1) if hz.n_bin else 0.0
    rad = abs_gc + abs_gb
    return Bounds(
        lb=torch.from_numpy(hz.c - rad).reshape(1, -1),
        ub=torch.from_numpy(hz.c + rad).reshape(1, -1),
    )


def _clone_ids(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if t is None else t.clone()


def _align(
    ids_x: torch.Tensor,
    ids_y: torch.Tensor,
    Gx: torch.Tensor,
    Gy: torch.Tensor,
):
    """Merge two generator matrices by column id, preserving shared factors."""
    n = Gx.shape[0]
    dtype, device = Gx.dtype, Gx.device
    pos: dict[int, int] = {}
    merged_ids: list[int] = []
    for idv in ids_x.tolist():
        if idv not in pos:
            pos[idv] = len(merged_ids)
            merged_ids.append(idv)
    for idv in ids_y.tolist():
        if idv not in pos:
            pos[idv] = len(merged_ids)
            merged_ids.append(idv)
    x_map = torch.tensor([pos[v] for v in ids_x.tolist()], dtype=torch.long, device=device)
    y_map = torch.tensor([pos[v] for v in ids_y.tolist()], dtype=torch.long, device=device)
    G = torch.zeros(n, len(merged_ids), dtype=dtype, device=device)
    if Gx.shape[1]:
        G.index_add_(1, x_map, Gx)
    if Gy.shape[1]:
        G.index_add_(1, y_map, Gy.to(dtype=dtype, device=device))
    return G, torch.tensor(merged_ids, dtype=torch.long, device=device), x_map, y_map


def _scatter_cols(A: torch.Tensor, col_map: torch.Tensor, n_merged: int) -> torch.Tensor:
    """Lift constraints into the merged generator-column space."""
    out = A.new_zeros(A.shape[0], n_merged)
    if A.shape[1]:
        out[:, col_map] = A
    return out


def _constraint_mask(hz: HZono) -> torch.Tensor:
    if hz.eq_mask is not None:
        return hz.eq_mask
    return torch.ones(int(hz.Ac.shape[0]), dtype=torch.bool, device=hz.Ac.device)


def _shared_constraint_prefix(
    Ac_x: torch.Tensor,
    Ac_y: torch.Tensor,
    Ab_x: torch.Tensor,
    Ab_y: torch.Tensor,
    b_x: torch.Tensor,
    b_y: torch.Tensor,
    eq_x: Optional[torch.Tensor],
    eq_y: Optional[torch.Tensor],
) -> int:
    """Return the common prefix length of identical constraints."""
    m = min(int(Ac_x.shape[0]), int(Ac_y.shape[0]))
    if m == 0:
        return 0
    same = (Ac_x[:m] == Ac_y[:m]).all(dim=1)
    if Ab_x.shape[1]:
        same &= (Ab_x[:m] == Ab_y[:m]).all(dim=1)
    same &= (b_x[:m] == b_y[:m]).reshape(m, -1).all(dim=1)
    if eq_x is not None or eq_y is not None:
        ex = eq_x if eq_x is not None else torch.ones(
            int(Ac_x.shape[0]), dtype=torch.bool, device=Ac_x.device
        )
        ey = eq_y if eq_y is not None else torch.ones(
            int(Ac_y.shape[0]), dtype=torch.bool, device=Ac_x.device
        )
        same &= ex[:m].to(Ac_x.device) == ey[:m].to(Ac_x.device)
    return m if bool(same.all()) else int((~same).nonzero()[0, 0])


def hz_sgm_add(hz_x: HZono, hz_y: HZono) -> HZono:
    """Add HZs while preserving shared generator identities.

    Matching col_ids denote the same latent factor, so correlated terms can
    combine exactly instead of being duplicated as independent Minkowski terms.
    """
    if hz_x.col_ids is None or hz_y.col_ids is None:
        return hz_minkowski_sum(hz_x, hz_y)
    n = int(hz_x.c.shape[0])
    if int(hz_y.c.shape[0]) != n:
        raise ValueError(f"hz_sgm_add: shape mismatch {n} vs {hz_y.c.shape[0]}")
    dtype, device = hz_x.c.dtype, hz_x.c.device
    bx = hz_x.bcol_ids if hz_x.bcol_ids is not None else torch.zeros(
        0, dtype=torch.long, device=device
    )
    by = hz_y.bcol_ids if hz_y.bcol_ids is not None else torch.zeros(
        0, dtype=torch.long, device=device
    )
    Gc, cids, xc_map, yc_map = _align(hz_x.col_ids, hz_y.col_ids, hz_x.Gc, hz_y.Gc)
    Gb, bids, xb_map, yb_map = _align(bx, by, hz_x.Gb, hz_y.Gb)
    Ac_x = _scatter_cols(hz_x.Ac, xc_map, Gc.shape[1])
    Ac_y = _scatter_cols(hz_y.Ac.to(dtype=dtype, device=device), yc_map, Gc.shape[1])
    Ab_x = _scatter_cols(hz_x.Ab, xb_map, Gb.shape[1])
    Ab_y = _scatter_cols(hz_y.Ab.to(dtype=dtype, device=device), yb_map, Gb.shape[1])
    b_x = hz_x.b.to(dtype=dtype, device=device)
    b_y = hz_y.b.to(dtype=dtype, device=device)
    k = _shared_constraint_prefix(
        Ac_x, Ac_y, Ab_x, Ab_y, b_x, b_y, hz_x.eq_mask, hz_y.eq_mask
    )
    if hz_x.eq_mask is None and hz_y.eq_mask is None:
        eq_mask = None
    else:
        eq_mask = torch.cat(
            [_constraint_mask(hz_x).to(device), _constraint_mask(hz_y).to(device)[k:]],
            dim=0,
        )
    return HZono(
        c=hz_x.c + hz_y.c.to(dtype=dtype, device=device),
        Gc=Gc,
        Gb=Gb,
        Ac=torch.cat([Ac_x, Ac_y[k:]], dim=0),
        Ab=torch.cat([Ab_x, Ab_y[k:]], dim=0),
        b=torch.cat([b_x, b_y[k:]], dim=0),
        eq_mask=eq_mask,
        col_ids=cids,
        bcol_ids=bids,
    )


def hz_negate(hz: HZono) -> HZono:
    return HZono(
        c=-hz.c,
        Gc=-hz.Gc,
        Gb=-hz.Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=_clone_ids(hz.eq_mask),
        col_ids=_clone_ids(hz.col_ids),
        bcol_ids=_clone_ids(hz.bcol_ids),
    )


def hz_sub(hz_x: HZono, hz_y: HZono) -> HZono:
    return hz_sgm_add(hz_x, hz_negate(hz_y))


def hz_concat(parts) -> "HZono | None":
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    if any(p.col_ids is None for p in parts):
        return _hz_concat_independent(parts)
    dtype, device = parts[0].c.dtype, parts[0].c.device
    cpos: dict[int, int] = {}
    cids: list[int] = []
    for p in parts:
        for idv in p.col_ids.tolist():
            if idv not in cpos:
                cpos[idv] = len(cids)
                cids.append(idv)
    bpos: dict[int, int] = {}
    bids: list[int] = []
    for p in parts:
        pb = p.bcol_ids if p.bcol_ids is not None else torch.zeros(
            0, dtype=torch.long, device=device
        )
        for idv in pb.tolist():
            if idv not in bpos:
                bpos[idv] = len(bids)
                bids.append(idv)
    ngm, nbm = len(cids), len(bids)
    cs, Gcs, Gbs, Acs, Abs, bs, eqs = [], [], [], [], [], [], []
    for p in parts:
        cmap = torch.tensor([cpos[v] for v in p.col_ids.tolist()], dtype=torch.long, device=device)
        pb = p.bcol_ids if p.bcol_ids is not None else torch.zeros(
            0, dtype=torch.long, device=device
        )
        bmap = torch.tensor([bpos[v] for v in pb.tolist()], dtype=torch.long, device=device)
        Gc_p = p.c.new_zeros(p.c.shape[0], ngm)
        if p.Gc.shape[1]:
            Gc_p[:, cmap] = p.Gc.to(dtype=dtype, device=device)
        Gb_p = p.c.new_zeros(p.c.shape[0], nbm)
        if p.Gb.shape[1] and nbm:
            Gb_p[:, bmap] = p.Gb.to(dtype=dtype, device=device)
        Ac_p = p.Ac.new_zeros(p.Ac.shape[0], ngm)
        if p.Ac.shape[1]:
            Ac_p[:, cmap] = p.Ac.to(dtype=dtype, device=device)
        Ab_p = p.Ab.new_zeros(p.Ab.shape[0], nbm)
        if p.Ab.shape[1] and nbm:
            Ab_p[:, bmap] = p.Ab.to(dtype=dtype, device=device)
        cs.append(p.c.to(dtype=dtype, device=device))
        Gcs.append(Gc_p)
        Gbs.append(Gb_p)
        Acs.append(Ac_p)
        Abs.append(Ab_p)
        bs.append(p.b.to(dtype=dtype, device=device))
        eqs.append(_constraint_mask(p).to(device))
    return HZono(
        c=torch.cat(cs, 0),
        Gc=torch.cat(Gcs, 0),
        Gb=torch.cat(Gbs, 0),
        Ac=torch.cat(Acs, 0),
        Ab=torch.cat(Abs, 0),
        b=torch.cat(bs, 0),
        eq_mask=torch.cat(eqs, 0) if any(p.eq_mask is not None for p in parts) else None,
        col_ids=torch.tensor(cids, dtype=torch.long, device=device),
        bcol_ids=torch.tensor(bids, dtype=torch.long, device=device),
    )


def _hz_concat_independent(parts) -> HZono:
    dtype, device = parts[0].c.dtype, parts[0].c.device
    ng_tot = sum(int(p.Gc.shape[1]) for p in parts)
    nb_tot = sum(int(p.Gb.shape[1]) for p in parts)
    nc_tot = sum(int(p.Ac.shape[0]) for p in parts)
    Ac = torch.zeros(nc_tot, ng_tot, dtype=dtype, device=device)
    Ab = torch.zeros(nc_tot, nb_tot, dtype=dtype, device=device)
    cs, Gcs, Gbs, bs, eqs = [], [], [], [], []
    goff = boff = roff = 0
    for p in parts:
        n_p, ng_p = int(p.c.shape[0]), int(p.Gc.shape[1])
        nb_p, nc_p = int(p.Gb.shape[1]), int(p.Ac.shape[0])
        Gc_p = torch.zeros(n_p, ng_tot, dtype=dtype, device=device)
        Gc_p[:, goff:goff + ng_p] = p.Gc.to(dtype=dtype, device=device)
        Gb_p = torch.zeros(n_p, nb_tot, dtype=dtype, device=device)
        Gb_p[:, boff:boff + nb_p] = p.Gb.to(dtype=dtype, device=device)
        cs.append(p.c.to(dtype=dtype, device=device))
        Gcs.append(Gc_p)
        Gbs.append(Gb_p)
        if nc_p:
            Ac[roff:roff + nc_p, goff:goff + ng_p] = p.Ac.to(dtype=dtype, device=device)
            Ab[roff:roff + nc_p, boff:boff + nb_p] = p.Ab.to(dtype=dtype, device=device)
            bs.append(p.b.to(dtype=dtype, device=device))
            eqs.append(_constraint_mask(p).to(device))
        goff += ng_p
        boff += nb_p
        roff += nc_p
    return HZono(
        c=torch.cat(cs, 0),
        Gc=torch.cat(Gcs, 0),
        Gb=torch.cat(Gbs, 0),
        Ac=Ac,
        Ab=Ab,
        b=torch.cat(bs, 0) if bs else torch.zeros(0, 1, dtype=dtype, device=device),
        eq_mask=torch.cat(eqs, 0) if eqs else None,
    )


# ============================================================================
# 3. Bounds computation
# ============================================================================


def _hz_is_unconstrained(hz: HZono) -> bool:
    tol = 1e-12
    return (
        torch.all(torch.abs(hz.Ac) < tol).item()
        and torch.all(torch.abs(hz.Ab) < tol).item()
        and torch.all(torch.abs(hz.b) < tol).item()
    )


def _hz_bounds_unconstrained(hz: HZono) -> Bounds:
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
    return Bounds(lb=(hz.c - rad).reshape(1, -1), ub=(hz.c + rad).reshape(1, -1))


def _hz_compute_bounds_scipy(hz: HZono) -> Bounds:
    model = _lower_hz_milp(hz)
    if model.n_var == 0:
        LB = UB = model.value_center.copy()
    else:
        constraints = (
            LinearConstraint(model.A, model.row_lb, model.row_ub)
            if model.A.shape[0] else None
        )
        bounds = SciPyBounds(model.var_lb, model.var_ub)
        LB = np.empty(model.value_center.size, dtype=np.float64)
        UB = np.empty(model.value_center.size, dtype=np.float64)
        for i in range(model.value_center.size):
            obj = model.value_matrix.getrow(i).toarray().reshape(-1)
            options = {"presolve": True, "mip_rel_gap": 0.0}
            res_min = milp(
                obj,
                integrality=model.integrality,
                bounds=bounds,
                constraints=constraints,
                options=options,
            )
            res_max = milp(
                -obj,
                integrality=model.integrality,
                bounds=bounds,
                constraints=constraints,
                options=options,
            )
            if not res_min.success or not res_max.success:
                raise RuntimeError(f"HybridZ bound MILP failed at output {i}")
            LB[i] = model.value_center[i] + res_min.fun
            UB[i] = model.value_center[i] - res_max.fun

    dtype, device = hz.c.dtype, hz.c.device
    return Bounds(
        lb=torch.from_numpy(LB).to(device=device, dtype=dtype).reshape(1, -1),
        ub=torch.from_numpy(UB).to(device=device, dtype=dtype).reshape(1, -1),
    )


def hz_compute_bounds(hz: HZono, *, exact: bool = False) -> Bounds:
    """Compute box bounds from a hybrid zonotope.

    Args:
        hz: The hybrid zonotope.
        exact: If False (default), always use the fast unconstrained
            over-approximation (|Gc| + |Gb| radius). This is sound but
            may be wider than necessary.  If True, solve per-dimension
            LP/MILP to obtain tight bounds when equality constraints
            exist.  Use ``exact=True`` only at the final output layer
            where tight bounds matter for verification; intermediate
            layers benefit from the 1000×+ speed-up of the fast path
            with negligible precision loss (the full zonotope structure
            is still propagated via ``_hz_cache``).
    """
    if _hz_is_unconstrained(hz):
        return _hz_bounds_unconstrained(hz)
    if not exact:
        return _hz_bounds_unconstrained(hz)
    if _HAS_SCIPY:
        try:
            return _hz_compute_bounds_scipy(hz)
        except Exception as e:
            # Intentional: scipy linprog failures fall back to the unconstrained bounds estimate.
            logger.debug("suppressed: %s", e)
    return _hz_bounds_unconstrained(hz)


# ============================================================================
# 4. HZSolver
# ============================================================================


@dataclass(frozen=True)
class _HZMILP:
    value_center: "np.ndarray"
    value_matrix: "sp.csr_matrix"
    A: "sp.csr_matrix"
    row_lb: "np.ndarray"
    row_ub: "np.ndarray"
    var_lb: "np.ndarray"
    var_ub: "np.ndarray"
    integrality: "np.ndarray"
    n_cont: int
    n_bin: int
    cont_source: "np.ndarray"
    bin_source: "np.ndarray"

    @property
    def n_var(self) -> int:
        return self.n_cont + self.n_bin


@dataclass(frozen=True)
class _MILPResult:
    status: str
    x: Optional["np.ndarray"]
    nodes: int = 0


@dataclass(frozen=True)
class _MIPObjectiveResult:
    upper_bound: Optional[float]
    x: Optional["np.ndarray"]
    nodes: int = 0


def _row_sum(mat) -> "np.ndarray":
    return np.asarray(mat.sum(axis=1), dtype=np.float64).reshape(-1)


def _coalesce_antiparallel_rows(A, row_lb, row_ub):
    A = A.tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()
    row_lb = np.asarray(row_lb, dtype=np.float64).copy()
    row_ub = np.asarray(row_ub, dtype=np.float64).copy()
    representatives = {}
    keep = []
    lower = []
    upper = []
    for row in range(A.shape[0]):
        start, stop = A.indptr[row], A.indptr[row + 1]
        indices = A.indices[start:stop]
        values = A.data[start:stop]
        key = (indices.tobytes(), values.tobytes())
        negated = (indices.tobytes(), (-values).tobytes())
        if key in representatives:
            target = representatives[key]
            lower[target] = max(lower[target], row_lb[row])
            upper[target] = min(upper[target], row_ub[row])
        elif negated in representatives:
            target = representatives[negated]
            lower[target] = max(lower[target], -row_ub[row])
            upper[target] = min(upper[target], -row_lb[row])
        else:
            representatives[key] = len(keep)
            keep.append(row)
            lower.append(row_lb[row])
            upper.append(row_ub[row])
    if len(keep) == A.shape[0]:
        return A, row_lb, row_ub
    return (
        A[np.asarray(keep, dtype=np.int64)],
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )


def _lower_hz_milp(hz: "HZono | SparseHZono") -> _HZMILP:
    _require_sparse()
    if isinstance(hz, SparseHZono):
        c, Gc, Gb = hz.c, hz.Gc, hz.Gb
        n_cont, n_bin = hz.n_cont, hz.n_bin
        eq_Ac, eq_Ab, eq_b = hz.Ac, hz.Ab, hz.b
        le_Ac, le_Ab, le_b = hz.Auc, hz.Aub, hz.ub
    elif isinstance(hz, HZono):
        c = hz.c.detach().cpu().double().numpy().reshape(-1)
        Gc, Gb = _torch_to_csr(hz.Gc), _torch_to_csr(hz.Gb)
        n_cont, n_bin = Gc.shape[1], Gb.shape[1]
        Ac, Ab = _torch_to_csr(hz.Ac), _torch_to_csr(hz.Ab)
        b = hz.b.detach().cpu().double().numpy().reshape(-1)
        mask = (
            np.ones(Ac.shape[0], dtype=bool)
            if hz.eq_mask is None
            else hz.eq_mask.detach().cpu().numpy().astype(bool).reshape(-1)
        )
        if mask.size != Ac.shape[0]:
            raise ValueError("HZ eq_mask length does not match constraint rows")
        eq_Ac, eq_Ab, eq_b = Ac[mask], Ab[mask], b[mask]
        le_Ac, le_Ab, le_b = Ac[~mask], Ab[~mask], b[~mask]
    else:
        raise TypeError(f"unsupported HZ representation: {type(hz).__name__}")

    used_cont = np.asarray(Gc.getnnz(axis=0)).reshape(-1) != 0
    used_bin = np.asarray(Gb.getnnz(axis=0)).reshape(-1) != 0
    if eq_Ac.shape[0]:
        used_cont |= np.asarray(eq_Ac.getnnz(axis=0)).reshape(-1) != 0
        used_bin |= np.asarray(eq_Ab.getnnz(axis=0)).reshape(-1) != 0
    if le_Ac.shape[0]:
        used_cont |= np.asarray(le_Ac.getnnz(axis=0)).reshape(-1) != 0
        used_bin |= np.asarray(le_Ab.getnnz(axis=0)).reshape(-1) != 0
    cont_source = np.flatnonzero(used_cont).astype(np.int64, copy=False)
    bin_source = np.flatnonzero(used_bin).astype(np.int64, copy=False)
    Gc, Gb = Gc[:, cont_source], Gb[:, bin_source]
    eq_Ac, eq_Ab = eq_Ac[:, cont_source], eq_Ab[:, bin_source]
    le_Ac, le_Ab = le_Ac[:, cont_source], le_Ab[:, bin_source]
    n_cont, n_bin = cont_source.size, bin_source.size

    value_center = np.asarray(c, dtype=np.float64).reshape(-1) - _row_sum(Gb)
    value_matrix = sp.hstack([Gc, 2.0 * Gb], format="csr")
    blocks, lowers, uppers = [], [], []
    if eq_Ac.shape[0]:
        blocks.append(sp.hstack([eq_Ac, 2.0 * eq_Ab], format="csr"))
        rhs = np.asarray(eq_b, dtype=np.float64).reshape(-1) + _row_sum(eq_Ab)
        lowers.append(rhs)
        uppers.append(rhs)
    if le_Ac.shape[0]:
        blocks.append(sp.hstack([le_Ac, 2.0 * le_Ab], format="csr"))
        rhs = np.asarray(le_b, dtype=np.float64).reshape(-1) + _row_sum(le_Ab)
        lowers.append(np.full(rhs.size, -np.inf, dtype=np.float64))
        uppers.append(rhs)
    A = sp.vstack(blocks, format="csr") if blocks else sparse_empty(0, n_cont + n_bin)
    row_lb = np.concatenate(lowers) if lowers else np.zeros(0, dtype=np.float64)
    row_ub = np.concatenate(uppers) if uppers else np.zeros(0, dtype=np.float64)
    A, row_lb, row_ub = _coalesce_antiparallel_rows(A, row_lb, row_ub)
    return _HZMILP(
        value_center=value_center,
        value_matrix=value_matrix,
        A=A,
        row_lb=row_lb,
        row_ub=row_ub,
        var_lb=np.concatenate([
            -np.ones(n_cont, dtype=np.float64),
            np.zeros(n_bin, dtype=np.float64),
        ]),
        var_ub=np.ones(n_cont + n_bin, dtype=np.float64),
        integrality=np.concatenate([
            np.zeros(n_cont, dtype=np.int32),
            np.ones(n_bin, dtype=np.int32),
        ]),
        n_cont=n_cont,
        n_bin=n_bin,
        cont_source=cont_source,
        bin_source=bin_source,
    )


def _combined_constraints(model: _HZMILP, extra_A, extra_lb, extra_ub):
    if extra_A is None or extra_A.shape[0] == 0:
        return model.A, model.row_lb, model.row_ub
    A = sp.vstack([model.A, extra_A], format="csr")
    return (
        A,
        np.concatenate([model.row_lb, np.asarray(extra_lb, dtype=np.float64)]),
        np.concatenate([model.row_ub, np.asarray(extra_ub, dtype=np.float64)]),
    )


def _valid_milp_point(
    model: _HZMILP,
    x: "np.ndarray",
    A,
    row_lb,
    row_ub,
    tol: float,
) -> bool:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size != model.n_var or not np.all(np.isfinite(x)):
        return False
    if np.any(x < model.var_lb - tol) or np.any(x > model.var_ub + tol):
        return False
    if model.n_bin:
        z = x[model.n_cont:]
        if np.any(np.abs(z - np.rint(z)) > tol):
            return False
    if A.shape[0]:
        values = np.asarray(A @ x, dtype=np.float64).reshape(-1)
        finite_lb = np.isfinite(row_lb)
        finite_ub = np.isfinite(row_ub)
        if np.any(values[finite_lb] < row_lb[finite_lb] - tol):
            return False
        if np.any(values[finite_ub] > row_ub[finite_ub] + tol):
            return False
    return True


def _solve_hz_feasibility(
    model: _HZMILP,
    deadline: float,
    *,
    extra_A=None,
    extra_lb=None,
    extra_ub=None,
    feasibility_tol: float = 1e-7,
) -> _MILPResult:
    A, row_lb, row_ub = _combined_constraints(model, extra_A, extra_lb, extra_ub)
    if model.n_var == 0:
        x = np.zeros(0, dtype=np.float64)
        status = "feasible" if _valid_milp_point(
            model, x, A, row_lb, row_ub, feasibility_tol
        ) else "infeasible"
        return _MILPResult(status, x if status == "feasible" else None)
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        return _MILPResult("unknown", None)
    constraints = (
        LinearConstraint(A, row_lb, row_ub) if A.shape[0] else None
    )
    try:
        result = milp(
            c=np.zeros(model.n_var, dtype=np.float64),
            integrality=model.integrality,
            bounds=SciPyBounds(model.var_lb, model.var_ub),
            constraints=constraints,
            options={
                "presolve": True,
                "time_limit": max(1e-3, remaining),
                "mip_rel_gap": 0.0,
            },
        )
    except Exception as exc:
        logger.debug("HybridZ MILP failed: %s", exc)
        return _MILPResult("unknown", None)
    nodes = int(getattr(result, "mip_node_count", 0) or 0)
    x = getattr(result, "x", None)
    if x is not None and _valid_milp_point(
        model, x, A, row_lb, row_ub, feasibility_tol
    ):
        return _MILPResult("feasible", np.asarray(x, dtype=np.float64), nodes)
    if int(getattr(result, "status", -1)) == 2:
        return _MILPResult("infeasible", None, nodes)
    return _MILPResult("unknown", None, nodes)


def _add_highs_model(solver, model: _HZMILP) -> None:
    solver.addVars(model.n_var, model.var_lb, model.var_ub)
    if model.A.shape[0]:
        A = model.A.tocsr()
        solver.addRows(
            A.shape[0],
            model.row_lb,
            model.row_ub,
            A.nnz,
            A.indptr.astype(np.int32, copy=False),
            A.indices.astype(np.int32, copy=False),
            A.data,
        )


def _set_highs_integrality(solver, model: _HZMILP) -> None:
    if model.n_bin:
        binary = np.arange(model.n_cont, model.n_var, dtype=np.int32)
        types = np.full(model.n_bin, highspy.HighsVarType.kInteger, dtype=object)
        solver.changeColsIntegrality(model.n_bin, binary, types)


class _HighsLPRelaxation:
    def __init__(self, model: _HZMILP):
        if highspy is None:
            raise RuntimeError("highspy is unavailable")
        self.model = model
        self.solver = highspy.Highs()
        self.solver.setOptionValue("output_flag", False)
        self.solver.setOptionValue("solver", "simplex")
        self.solver.setOptionValue("presolve", "on")
        self.solver.setOptionValue("simplex_strategy", 1)
        self.solver.setOptionValue("threads", 1)
        self.solver.setOptionValue("primal_feasibility_tolerance", 1e-8)
        self.solver.setOptionValue("dual_feasibility_tolerance", 1e-8)
        self.solver.setOptionValue("mip_feasibility_tolerance", 1e-8)
        A = model.A.tocsr()
        lp = highspy.HighsLp()
        lp.num_col_ = model.n_var
        lp.num_row_ = A.shape[0]
        lp.col_cost_ = np.zeros(model.n_var, dtype=np.float64)
        lp.col_lower_ = model.var_lb
        lp.col_upper_ = model.var_ub
        lp.row_lower_ = model.row_lb
        lp.row_upper_ = model.row_ub
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_ = A.indptr.astype(np.int32, copy=False)
        lp.a_matrix_.index_ = A.indices.astype(np.int32, copy=False)
        lp.a_matrix_.value_ = A.data
        self.solver.passModel(lp)
        self.solver.changeObjectiveSense(highspy.ObjSense.kMaximize)
        self.indices = np.arange(model.n_var, dtype=np.int32)
        self.costs = np.zeros(model.n_var, dtype=np.float64)

    def maximize(self, coeff, offset: float, deadline: float) -> Optional[float]:
        coeff = coeff.tocsr()
        if coeff.nnz == 0:
            return float(offset)
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return None
        self.costs.fill(0.0)
        self.costs[coeff.indices] = coeff.data
        self.solver.changeColsCost(self.model.n_var, self.indices, self.costs)
        self.solver.setOptionValue("time_limit", remaining)
        self.solver.run()
        if self.solver.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return None
        value = float(offset + self.solver.getObjectiveValue())
        return value + 1e-7 * (1.0 + abs(value))

    def maximize_mip(
        self,
        coeff,
        offset: float,
        deadline: float,
        *,
        cutoff: float,
        feasibility_tol: float,
    ) -> _MIPObjectiveResult:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return _MIPObjectiveResult(None, None)
        coeff = coeff.tocsr()
        costs = np.zeros(self.model.n_var, dtype=np.float64)
        costs[coeff.indices] = -coeff.data
        solver = highspy.Highs()
        solver.setOptionValue("output_flag", False)
        solver.setOptionValue("threads", 1)
        solver.setOptionValue("presolve", "on")
        solver.setOptionValue("mip_rel_gap", 0.0)
        solver.setOptionValue("mip_heuristic_effort", 0.0)
        solver.setOptionValue("mip_lp_solver", "ipm")
        _add_highs_model(solver, self.model)
        solver.changeColsCost(self.model.n_var, self.indices, costs)
        _set_highs_integrality(solver, self.model)
        guard = 2e-7 * (1.0 + abs(cutoff))
        solver.changeObjectiveSense(highspy.ObjSense.kMinimize)
        solver.setOptionValue("objective_bound", offset - cutoff + guard)
        solver.setOptionValue(
            "objective_target",
            offset - cutoff - 2.0 * feasibility_tol,
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return _MIPObjectiveResult(None, None)
        solver.setOptionValue("time_limit", remaining)
        solver.run()
        status = solver.getModelStatus()
        info = solver.getInfo()
        nodes = max(0, int(getattr(info, "mip_node_count", 0) or 0))
        safe_statuses = {
            highspy.HighsModelStatus.kInfeasible,
            highspy.HighsModelStatus.kObjectiveBound,
        }
        if status in safe_statuses:
            return _MIPObjectiveResult(
                np.nextafter(cutoff, -np.inf),
                None,
                nodes,
            )
        bound_statuses = {
            highspy.HighsModelStatus.kOptimal,
            highspy.HighsModelStatus.kTimeLimit,
            highspy.HighsModelStatus.kObjectiveTarget,
        }
        dual = (
            float(getattr(info, "mip_dual_bound", np.inf))
            if status in bound_statuses
            else np.inf
        )
        upper = None
        if np.isfinite(dual):
            value = float(offset - dual)
            upper = value + 1e-7 * (1.0 + abs(value))
        solution = solver.getSolution()
        x = None
        if bool(solution.value_valid):
            candidate = np.asarray(solution.col_value, dtype=np.float64)
            if _valid_milp_point(
                self.model,
                candidate,
                self.model.A,
                self.model.row_lb,
                self.model.row_ub,
                feasibility_tol,
            ):
                x = candidate
        return _MIPObjectiveResult(upper, x, nodes)


def sparse_hz_obbt_bounds(
    hz: SparseHZono,
    lb,
    ub,
    rows,
    *,
    time_limit: float,
):
    lb = np.asarray(lb, dtype=np.float64).reshape(-1).copy()
    ub = np.asarray(ub, dtype=np.float64).reshape(-1).copy()
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if highspy is None or rows.size == 0 or time_limit <= 0.0:
        return lb, ub
    model = _lower_hz_milp(hz)
    if model.value_center.size != lb.size or ub.size != lb.size:
        raise ValueError("sparse OBBT bounds do not match HZ outputs")
    if model.n_var == 0:
        return np.maximum(lb, model.value_center), np.minimum(ub, model.value_center)

    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("solver", "simplex")
    solver.setOptionValue("presolve", "off")
    solver.setOptionValue("threads", 1)
    _add_highs_model(solver, model)

    indices = np.arange(model.n_var, dtype=np.int32)
    costs = np.zeros(model.n_var, dtype=np.float64)
    deadline = time.monotonic() + float(time_limit)
    priority = rows[np.argsort(np.minimum(-lb[rows], ub[rows]))]
    primary = {
        int(row): (-1.0 if ub[row] <= -lb[row] else 1.0)
        for row in priority
    }
    for opposite in (False, True):
        for row in priority:
            if lb[row] >= 0.0 or ub[row] <= 0.0:
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return lb, ub
            coeff = model.value_matrix.getrow(int(row))
            costs.fill(0.0)
            costs[coeff.indices] = coeff.data
            sense = primary[int(row)] * (-1.0 if opposite else 1.0)
            solver.setOptionValue("time_limit", remaining)
            solver.changeColsCost(model.n_var, indices, sense * costs)
            solver.run()
            if solver.getModelStatus() != highspy.HighsModelStatus.kOptimal:
                return lb, ub
            objective = float(solver.getInfo().objective_function_value)
            value = model.value_center[row] + sense * objective
            guard = 1e-7 * (1.0 + abs(value))
            if sense > 0.0:
                lb[row] = max(lb[row], value - guard)
            else:
                ub[row] = min(ub[row], value + guard)
    return lb, ub


class HZSolver(Solver):
    """Open-source Hybrid Zonotope bounds and verdict solver."""

    def __init__(self, time_limit: float = 30.0, tolerance: float = 1e-7):
        self._last_bounds: Optional[Bounds] = None
        self.time_limit = float(time_limit)
        self.tolerance = float(tolerance)
        self.last_stats: dict[str, object] = {}

    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=False, supports_csp=False, supports_hz=True)

    def compute_bounds(self, hz: HZono, *, exact: bool = False) -> Bounds:
        self._last_bounds = hz_compute_bounds(hz, exact=exact)
        return self._last_bounds

    def _unknown_results(self, batch_size: int, reason: str) -> list[VerifyResult]:
        return [
            VerifyResult(
                VerifyStatus.UNKNOWN,
                metadata={"lane": lane, "source": "hybridz", "reason": reason},
            )
            for lane in range(batch_size)
        ]

    @staticmethod
    def _recover_input(
        model: _HZMILP,
        x: "np.ndarray",
        input_hz: Optional[SparseHZono],
        input_shape: tuple[int, ...],
        lane: int,
    ) -> Optional[torch.Tensor]:
        if input_hz is None or input_hz.n_out != int(np.prod(input_shape)):
            return None
        xi_c = np.zeros(input_hz.n_cont, dtype=np.float64)
        input_cont = model.cont_source < input_hz.n_cont
        xi_c[model.cont_source[input_cont]] = x[:model.n_cont][input_cont]
        xi_b = -np.ones(input_hz.n_bin, dtype=np.float64)
        input_bin = model.bin_source < input_hz.n_bin
        z = x[model.n_cont:]
        xi_b[model.bin_source[input_bin]] = 2.0 * z[input_bin] - 1.0
        value = input_hz.c.copy()
        if input_hz.n_cont:
            value += np.asarray(input_hz.Gc @ xi_c).reshape(-1)
        if input_hz.n_bin:
            value += np.asarray(input_hz.Gb @ xi_b).reshape(-1)
        full = torch.from_numpy(value.reshape(input_shape).copy())
        return full[lane].clone()

    def evaluate_spec(
        self,
        output_hz: "HZono | SparseHZono | None",
        out_spec: "OutputSpec",
        *,
        batch_size: int,
        n_out: int,
        input_hz: Optional[SparseHZono] = None,
        input_shape: Optional[tuple[int, ...]] = None,
        timelimit: Optional[float] = None,
    ) -> list[VerifyResult]:
        """Decide an output specification over a propagated Hybrid Zonotope."""
        B = int(batch_size)
        if output_hz is None:
            return self._unknown_results(B, "missing_hz_state")
        if not _HAS_SCIPY:
            return self._unknown_results(B, "scipy_unavailable")
        try:
            model = _lower_hz_milp(output_hz)
        except Exception as exc:
            return self._unknown_results(B, f"lowering_failed:{type(exc).__name__}")
        if model.value_center.size != B * int(n_out):
            return self._unknown_results(B, "output_shape_mismatch")

        encoded = out_spec.encode_linear(
            B=B,
            n_out=int(n_out),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        C = encoded["C"].detach().cpu().double().numpy()
        thresholds = encoded["thresholds"].detach().cpu().double().numpy()
        M = int(encoded["M"])
        is_unsafe_linear = encoded["kind"] == OutKind.UNSAFE_LINEAR
        started = time.monotonic()
        budget = float(self.time_limit if timelimit is None else timelimit)
        deadline = started + budget
        lp_deadline = min(deadline, started + max(1.0, 0.2 * budget))
        feasibility_tol = max(self.tolerance, 1e-7)
        solves = 0
        nodes = 0
        warm_lp = None
        if not is_unsafe_linear and highspy is not None:
            try:
                warm_lp = _HighsLPRelaxation(model)
            except Exception as exc:
                logger.debug("HybridZ warm LP setup failed: %s", exc)

        def solve(extra_A=None, extra_lb=None, extra_ub=None) -> _MILPResult:
            nonlocal solves, nodes
            result = _solve_hz_feasibility(
                model,
                deadline,
                extra_A=extra_A,
                extra_lb=extra_lb,
                extra_ub=extra_ub,
                feasibility_tol=feasibility_tol,
            )
            solves += 1
            nodes += result.nodes
            return result

        def maximize_lp(coeff, offset: float) -> Optional[float]:
            nonlocal solves
            if warm_lp is None:
                return None
            solves += 1
            return warm_lp.maximize(coeff, offset, lp_deadline)

        def maximize_mip(
            coeff, offset: float, cutoff: float
        ) -> _MIPObjectiveResult:
            nonlocal solves, nodes
            result = warm_lp.maximize_mip(
                coeff,
                offset,
                deadline,
                cutoff=cutoff,
                feasibility_tol=feasibility_tol,
            )
            solves += 1
            nodes += result.nodes
            return result

        exact_witness = (
            isinstance(output_hz, SparseHZono)
            and output_hz.exact
            and input_hz is not None
            and output_hz.frame_id is not None
            and output_hz.frame_id == input_hz.frame_id
            and input_shape is not None
        )
        representation = "sparse" if isinstance(output_hz, SparseHZono) else "dense"
        results: list[VerifyResult] = []

        def metadata(lane: int, reason: str) -> dict[str, object]:
            return {
                "lane": lane,
                "source": "hybridz_milp",
                "representation": representation,
                "reason": reason,
            }

        def falsified(lane: int, witness, reason: str) -> Optional[VerifyResult]:
            counterexample = self._recover_input(
                model, witness, input_hz, input_shape, lane
            )
            if counterexample is None:
                return None
            return VerifyResult(
                VerifyStatus.FALSIFIED,
                counterexample=counterexample,
                metadata=metadata(lane, reason),
            )

        for lane in range(B):
            start, stop = lane * n_out, (lane + 1) * n_out
            C_lane = C[lane * M:(lane + 1) * M]
            t_lane = thresholds[lane]
            value_matrix = model.value_matrix[start:stop]
            coeff = (sp.csr_matrix(C_lane) @ value_matrix).tocsr()
            const = C_lane @ model.value_center[start:stop]
            lane_result: Optional[VerifyResult] = None

            if is_unsafe_linear:
                expanded = solve(
                    coeff,
                    np.full(M, -np.inf, dtype=np.float64),
                    t_lane + self.tolerance - const,
                )
                if expanded.status == "infeasible":
                    lane_result = VerifyResult(
                        VerifyStatus.CERTIFIED,
                        metadata=metadata(lane, "expanded_unsafe_infeasible"),
                    )
                elif expanded.status == "feasible" and exact_witness:
                    values = const + np.asarray(coeff @ expanded.x).reshape(-1)
                    witness = expanded.x if np.all(values <= t_lane - self.tolerance) else None
                    if witness is None:
                        contracted = solve(
                            coeff,
                            np.full(M, -np.inf, dtype=np.float64),
                            t_lane - self.tolerance - const,
                        )
                        if contracted.status == "feasible":
                            values = const + np.asarray(coeff @ contracted.x).reshape(-1)
                            if np.all(values <= t_lane - self.tolerance):
                                witness = contracted.x
                    if witness is not None:
                        lane_result = falsified(
                            lane, witness, "exact_unsafe_witness"
                        )
                if lane_result is None:
                    lane_result = VerifyResult(
                        VerifyStatus.UNKNOWN,
                        metadata=metadata(lane, "unsafe_region_undecided"),
                    )
            else:
                hard_rows = []
                continuous_upper = sparse_abs_row_sum(
                    coeff[:, :model.n_cont]
                )
                binary_upper = (
                    np.asarray(
                        coeff[:, model.n_cont:].maximum(0.0).sum(axis=1)
                    ).reshape(-1)
                    if model.n_bin else np.zeros(M, dtype=np.float64)
                )
                box_upper = const + continuous_upper + binary_upper
                for row in range(M):
                    cutoff = float(t_lane[row] - self.tolerance)
                    if box_upper[row] < cutoff:
                        continue
                    relaxed = (
                        None
                        if model.n_bin >= 512
                        else maximize_lp(coeff[row], float(const[row]))
                    )
                    if relaxed is not None and relaxed < cutoff:
                        continue
                    hard_rows.append((
                        row,
                        float(box_upper[row]) if relaxed is None else relaxed,
                    ))
                    if time.monotonic() >= deadline:
                        break

                undecided = (
                    len(hard_rows) > 0
                    or len(hard_rows) < M and time.monotonic() >= deadline
                )
                hard_rows.sort(key=lambda item: item[1], reverse=True)
                for row, _ in hard_rows:
                    cutoff = float(t_lane[row] - self.tolerance)
                    if warm_lp is not None:
                        optimized = maximize_mip(
                            coeff[row], float(const[row]), cutoff
                        )
                        if (
                            optimized.upper_bound is not None
                            and optimized.upper_bound < cutoff
                        ):
                            undecided = False
                            continue
                        witness = optimized.x
                        if witness is not None and exact_witness:
                            value = const[row] + float(
                                (coeff[row] @ witness).item()
                            )
                            if value >= t_lane[row] + self.tolerance:
                                lane_result = falsified(
                                    lane, witness, "exact_violation_witness"
                                )
                        if lane_result is not None:
                            break
                        undecided = True
                        break
                    expanded = solve(
                        coeff[row],
                        np.array([cutoff - const[row]]),
                        np.array([np.inf]),
                    )
                    if expanded.status == "infeasible":
                        undecided = False
                        continue
                    witness = None
                    if expanded.status == "feasible":
                        value = const[row] + float((coeff[row] @ expanded.x).item())
                        if exact_witness:
                            if value >= t_lane[row] + self.tolerance:
                                witness = expanded.x
                            else:
                                contracted = solve(
                                    coeff[row],
                                    np.array([
                                        t_lane[row] + self.tolerance - const[row]
                                    ]),
                                    np.array([np.inf]),
                                )
                                if contracted.status == "feasible":
                                    witness = contracted.x
                    if witness is not None:
                        lane_result = falsified(
                            lane, witness, "exact_violation_witness"
                        )
                        if lane_result is not None:
                            break
                    undecided = True
                    break
                if lane_result is None:
                    lane_result = VerifyResult(
                        VerifyStatus.UNKNOWN if undecided else VerifyStatus.CERTIFIED,
                        metadata=metadata(
                            lane,
                            "objective_bound_undecided" if undecided
                            else "objective_bounds_safe",
                        ),
                    )
            results.append(lane_result)

        self.last_stats = {
            "elapsed": time.monotonic() - started,
            "solves": solves,
            "nodes": nodes,
            "n_cont": model.n_cont,
            "n_bin": model.n_bin,
            "n_rows": int(model.A.shape[0]),
            "representation": representation,
        }
        for result in results:
            result.metadata.update(self.last_stats)
        return results

    def solve_batch(
        self,
        problem: "BatchLPProblem",
        timelimit: Optional[float] = None,
    ) -> "BatchLPSolution":
        """HZSolver does not accept BatchLPProblem inputs.

        HZSolver operates on HZono domains via compute_bounds() and
        evaluate_spec(), not on LP/CSP batch problems. Callers that
        need batch LP solving should use TorchLPSolver or GurobiSolver.
        """
        raise NotImplementedError(
            "HZSolver does not accept BatchLPProblem; use evaluate_spec()."
        )
