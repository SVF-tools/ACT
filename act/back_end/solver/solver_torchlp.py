from __future__ import annotations
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import cached_property
import math, time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional
import numpy as np
import torch

from act.util.counterexample import validate_counterexamples
from act.back_end.core import Bounds, Net
from act.back_end.solver.solver_dual import unproven_spec_rows
from act.config.config import TorchLPConfig
from act.front_end.specs import InKind, OutKind
from act.util.stats import VerifyStatus
from act.back_end.solver.solver_base import (
    Solver,
    SolveStatus,
    SolverCaps,
    BatchLPProblem,
    BatchLPSolution,
)
from act.util.device_manager import get_default_device, get_default_dtype

if TYPE_CHECKING:
    from act.back_end.cons_exportor import TerminalExport, _TerminalHeader

class TorchLPSolver(Solver):
    """Continuous LP solver using Torch + Adam with penalty and box projection.

    - Supports GPU via device hint in begin(...).
    - LP-only: no integrality constraints (no binary vars, no SOS2).
    """
    def __init__(self, config: Optional[TorchLPConfig] = None):
        cfg = config or TorchLPConfig()
        self._device = get_default_device()
        self._dtype = get_default_dtype()
        # parameters
        self.rho_eq = cfg.rho_eq
        self.rho_ineq = cfg.rho_ineq
        self.max_iter = cfg.max_iter  # lighter default; see large-n overrides below
        self.tol_feas = cfg.tol_feas
        self.lr = cfg.lr
        self.beta1 = cfg.beta1
        self.beta2 = cfg.beta2
        self.weight_decay = cfg.weight_decay
        self._large_n_threshold = cfg.large_n_threshold
        self._large_n_max_iter = cfg.large_n_max_iter
        self._large_n_tol = cfg.large_n_tol
        self._log_every = 200
        self._stagnation_patience = cfg.stagnation_patience
        self._stagnation_tol = cfg.stagnation_tol
        self._feas_check_stride = cfg.feas_check_stride

    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=True)

    def solve_batch(
        self,
        problem: BatchLPProblem,
        timelimit: Optional[float] = None,
    ) -> BatchLPSolution:
        """Native batched LP solve via Adam + penalty on block-diagonal sparse.

        Operates uniformly over all N instances (no scalar special-case for
        N=1) using ``torch.sparse.mm`` against the ``[N*m, N*nvars]``
        block-diagonal matrices in ``problem``. The per-N feasibility gate
        preserves the C2 soundness invariant: penalty-on-Adam CANNOT certify
        infeasibility, so any instance whose
        final residual exceeds ``tol_feas`` is reported as UNKNOWN, never
        UNSAT. The returned iterate is the box-clamped last state for all
        N; callers must check ``statuses[i] == SAT`` before consuming
        ``x[i]``.
        """
        N = problem.N
        nvars = problem.nvars
        m_eq = problem.m_eq
        m_le = problem.m_le

        device = problem.lb.device
        dtype = problem.lb.dtype

        total_vars = N * nvars
        eff_max_iter = self.max_iter
        eff_tol_feas = self.tol_feas
        if total_vars > self._large_n_threshold:
            eff_max_iter = min(eff_max_iter, self._large_n_max_iter)
            eff_tol_feas = max(eff_tol_feas, self._large_n_tol)

        t_end = None if timelimit is None else (time.time() + float(timelimit))

        lb = problem.lb
        ub = problem.ub

        lb_finite = torch.where(torch.isfinite(lb), lb, torch.zeros_like(lb))
        ub_finite = torch.where(torch.isfinite(ub), ub, torch.zeros_like(ub))
        mid = 0.5 * (lb_finite + ub_finite)
        both_inf = (~torch.isfinite(lb)) & (~torch.isfinite(ub))
        mid = torch.where(both_inf, torch.zeros_like(mid), mid)
        x = torch.nn.Parameter(mid.clone().contiguous(), requires_grad=True)

        has_eq = m_eq > 0
        has_le = m_le > 0
        A_eq = problem.A_eq_blockdiag.coalesce() if has_eq else None
        A_le = problem.A_le_blockdiag.coalesce() if has_le else None
        b_eq = problem.b_eq
        b_le = problem.b_le
        obj_c = problem.obj_c
        obj_const = problem.obj_const
        is_max = (problem.sense == "max")

        opt = torch.optim.Adam(
            [x],
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
        )

        best_max_viol = math.inf
        iters_since_improve = 0

        for it in range(eff_max_iter):
            if t_end is not None and time.time() >= t_end:
                break

            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                obj = (
                    0.001 * (x * x).sum()
                    + obj_const.sum()
                    + (obj_c * x).sum()
                )
                if is_max:
                    obj = -obj
                x_flat = x.reshape(N * nvars).unsqueeze(1)
                if has_eq and A_eq is not None:
                    Ax_eq = torch.sparse.mm(A_eq, x_flat).squeeze(1)
                    v_eq_mat = Ax_eq.view(N, m_eq) - b_eq
                    obj = obj + self.rho_eq * (v_eq_mat * v_eq_mat).sum()
                if has_le and A_le is not None:
                    Ax_le = torch.sparse.mm(A_le, x_flat).squeeze(1)
                    v_le_mat = Ax_le.view(N, m_le) - b_le
                    obj = obj + self.rho_ineq * (torch.relu(v_le_mat) ** 2).sum()
                obj.backward()
            opt.step()

            with torch.no_grad():
                x.data.clamp_(lb, ub)

            if it % self._feas_check_stride != 0:
                continue

            with torch.no_grad():
                x_flat_no = x.reshape(N * nvars).unsqueeze(1)
                max_viol_per_n = torch.zeros(N, device=device, dtype=dtype)
                if has_eq and A_eq is not None:
                    v_eq_no = (
                        torch.sparse.mm(A_eq, x_flat_no).squeeze(1).view(N, m_eq)
                        - b_eq
                    )
                    max_viol_per_n = torch.maximum(
                        max_viol_per_n, v_eq_no.abs().max(dim=1).values
                    )
                if has_le and A_le is not None:
                    v_le_no = (
                        torch.sparse.mm(A_le, x_flat_no).squeeze(1).view(N, m_le)
                        - b_le
                    )
                    max_viol_per_n = torch.maximum(
                        max_viol_per_n, torch.relu(v_le_no).max(dim=1).values
                    )
                global_max_viol = float(max_viol_per_n.max().item())

            if global_max_viol < best_max_viol - self._stagnation_tol:
                best_max_viol = global_max_viol
                iters_since_improve = 0
            else:
                iters_since_improve += self._feas_check_stride

            if (
                global_max_viol <= eff_tol_feas
                or iters_since_improve >= self._stagnation_patience
            ):
                break

        with torch.no_grad():
            max_viol_final = torch.zeros(N, device=device, dtype=dtype)
            x_flat_final = x.reshape(N * nvars).unsqueeze(1)
            if has_eq and A_eq is not None:
                Ax_eq_f = torch.sparse.mm(A_eq, x_flat_final).squeeze(1)
                v_eq_final = Ax_eq_f.view(N, m_eq) - b_eq
                max_viol_final = torch.maximum(
                    max_viol_final,
                    v_eq_final.abs().max(dim=1).values,
                )
            if has_le and A_le is not None:
                Ax_le_f = torch.sparse.mm(A_le, x_flat_final).squeeze(1)
                v_le_final = Ax_le_f.view(N, m_le) - b_le
                max_viol_final = torch.maximum(
                    max_viol_final,
                    torch.relu(v_le_final).max(dim=1).values,
                )

            # Penalty-on-Adam CANNOT certify infeasibility (C2 invariant).
            statuses = tuple(
                SolveStatus.SAT
                if float(max_viol_final[i].item()) <= eff_tol_feas
                else SolveStatus.UNKNOWN
                for i in range(N)
            )
            x_out = x.detach().clone()

        return BatchLPSolution(
            statuses=statuses,
            x=x_out,
            max_viol=max_viol_final,
        )

# Certified terminal LP: BaB terminal pruning and CLIMB core admission share this
# proof-checking boundary. Solver statuses are never certificates: dual residuals
# are checked independently here. Fixed-phase elimination composes chain networks
# in float64 with rigorous error relaxation and uses exact rational arithmetic
# elsewhere, then relaxes outward when converting back to floats. SciPy is optional
# and imported only at the solve boundary.

# Independent lane LPs solved in one HiGHS call. The budget fraction keeps time
# for per-lane fallback when a block does not converge.
_BLOCK_MAX_LANES = 32
_BLOCK_MAX_NONZEROS = 2_000_000
_BLOCK_BUDGET_FRACTION = 0.5
_ELASTIC_PENALTY = 1.0e4
_UNIT_ROUNDOFF = 2.0 ** -53
# Keeps an exact-zero error estimate outward once rows are composed.
_CHAIN_ERROR_FLOOR = 1e-300
_CHAIN_KINDS = frozenset({"INPUT", "INPUT_SPEC", "ASSERT", "DENSE", "RELU", "FLATTEN", "RESHAPE"})
_CHAIN_INPUT_KINDS = frozenset({InKind.BOX, InKind.LINF_BALL, InKind.LIN_POLY})
# Entries of one lane batch's composed [K, neurons, inputs] map.
_CHAIN_MAX_ELEMENTS = 1 << 26

def _add(a: float, b: float, up: bool = False) -> float:
    """Round one addition outward, preserving exact zero identities."""
    if a == 0:
        return b
    if b == 0:
        return a
    if a == -b:
        return 0.0
    return math.nextafter(a + b, math.inf if up else -math.inf)


def _mul(a: float, b: float, up: bool = False) -> float:
    """Round one product outward, including underflow."""
    if a == 0 or b == 0:
        return 0.0
    if abs(a) == 1 or abs(b) == 1:
        return a * b
    return math.nextafter(a * b, math.inf if up else -math.inf)


def _outward_products(a: np.ndarray, b: np.ndarray, up: bool) -> np.ndarray:
    """Round products outward; exact factors of zero or one keep their identities."""
    product = a * b
    exact = (a == 0) | (b == 0) | (np.abs(a) == 1) | (np.abs(b) == 1)
    return np.where(exact, product, np.nextafter(product, np.inf if up else -np.inf))


def _outward_sum(terms: list[float], up: bool) -> float:
    """Exactly rounded sum, moved outward only when the rounding lost something."""
    total = math.fsum(terms)
    if not math.isfinite(total):
        return total
    return math.nextafter(total, math.inf if up else -math.inf) if math.fsum((*terms, -total)) else total


def _fraction_float(value: Fraction, up: bool) -> float:
    """Convert an exact rational in the requested rounding direction."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite affine coefficient")
    exact = Fraction(result)
    if (up and exact < value) or (not up and exact > value):
        result = math.nextafter(result, math.inf if up else -math.inf)
    return result


@dataclass
class _SparseLP:
    """One CPU sparse instance; immutable by convention after certification."""
    eq: Any
    eq_rhs: np.ndarray
    le: Any
    le_rhs: np.ndarray
    lb: np.ndarray
    ub: np.ndarray


def certified_lower_bound(
    A_eq: Any, b_eq: np.ndarray, A_le: Any, b_le: np.ndarray,
    lb: np.ndarray, ub: np.ndarray, c: np.ndarray,
    eq_dual: np.ndarray, le_dual: np.ndarray, constant: float = 0.0,
) -> Optional[float]:
    """Verify min-LP duals with outward-rounded residual box correction.

    For ``A_le x <= b_le``, multipliers must be nonpositive. Equality
    multipliers are free. Products round outward and each sum is computed exactly
    before one outward step, so accumulation error never compounds. Any residual ``c-A.T*y-E.T*z`` is minimized over
    the finite box, rather than assumed zero within a solver tolerance.
    This checker does not inspect a solver status or a reported objective.
    """
    arrays = (b_eq, b_le, lb, ub, c, eq_dual, le_dual)
    if any(not np.isfinite(a).all() for a in arrays) or not math.isfinite(constant):
        return None
    if (np.shape(lb) != np.shape(c) or np.shape(ub) != np.shape(c)
            or np.shape(eq_dual) != np.shape(b_eq)
            or np.shape(le_dual) != np.shape(b_le)
            or np.any(lb > ub) or np.any(le_dual > 0)):
        return None
    columns, lower_terms, upper_terms = [], [], []
    value_terms = [np.array([constant], dtype=np.float64)]
    for matrix, rhs, dual in ((A_eq, b_eq, eq_dual), (A_le, b_le, le_dual)):
        matrix = matrix.tocsr()
        if matrix.shape != (len(rhs), len(c)) or not np.isfinite(matrix.data).all():
            return None
        value_terms.append(_outward_products(dual, rhs, False))
        owners = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
        multiplier = dual[owners]
        used = multiplier != 0
        columns.append(matrix.indices[used])
        lower_terms.append(-_outward_products(matrix.data[used], multiplier[used], True))
        upper_terms.append(-_outward_products(matrix.data[used], multiplier[used], False))
    lo, hi = np.array(c, dtype=np.float64), np.array(c, dtype=np.float64)
    column = np.concatenate(columns)
    if column.size:
        order = np.argsort(column, kind="stable")
        touched, starts = np.unique(column[order], return_index=True)
        stops = np.append(starts[1:], column.size)
        low_values = np.concatenate(lower_terms)[order].tolist()
        high_values = np.concatenate(upper_terms)[order].tolist()
        c_values = c[touched].tolist()
        segments = list(zip(c_values, starts.tolist(), stops.tolist()))
        lo[touched] = [_outward_sum([base, *low_values[i:j]], False) for base, i, j in segments]
        hi[touched] = [_outward_sum([base, *high_values[i:j]], True) for base, i, j in segments]
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        return None
    corrections = np.stack([_outward_products(r, x, False) for r in (lo, hi) for x in (lb, ub)]).min(0)
    value = _outward_sum(np.concatenate((*value_terms, corrections)).tolist(), False)
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class TerminalCertificate:
    """Replayable numerical evidence, not a cached solver verdict."""
    kind: str
    lower_bound: float
    eq_dual: np.ndarray
    le_dual: np.ndarray
    _problem: _SparseLP = field(repr=False)
    _objective: np.ndarray = field(repr=False)
    _constant: float = field(repr=False)
    _out_kind: str = field(repr=False)
    _reference: torch.Tensor = field(repr=False)
    row_index: int = -1

    def verify(self) -> bool:
        """Replay this immutable evidence once; ``replace`` makes a new, unverified one."""
        return self._verified

    @cached_property
    def _verified(self) -> bool:
        """Recompute the corrected bound and the shared property boundary."""
        if (self.kind not in {"lp_residual", "phase1_residual"}
                or not bool(torch.isfinite(self._reference).all())):
            return False
        p = self._problem
        lower = certified_lower_bound(
            p.eq, p.eq_rhs, p.le, p.le_rhs, p.lb, p.ub, self._objective,
            self.eq_dual, self.le_dual, self._constant,
        )
        return (lower is not None and math.isfinite(self.lower_bound) and lower >= self.lower_bound
                and self._crosses_boundary(lower))

    def _crosses_boundary(self, lower: float) -> bool:
        """Apply the shared property boundary to one checked lower bound."""
        kind = OutKind.UNSAFE_LINEAR if self.kind == "phase1_residual" else self._out_kind
        return not bool(unproven_spec_rows(
            self._reference.double().new_tensor(lower), kind, self._reference,
        ).item())


@dataclass
class TerminalResult:
    """K-aligned outcomes; only ``has_lp_certificate`` authorizes retirement."""
    statuses: tuple[VerifyStatus, ...]
    witnesses: torch.Tensor
    certificates: tuple[tuple[TerminalCertificate, ...], ...]
    reasons: tuple[str, ...]
    tiers: tuple[str, ...]
    certificate_kinds: tuple[str, ...]
    _required_rows: int = field(default=1, repr=False)
    _export: Optional["TerminalExport"] = field(default=None, repr=False)

    def has_lp_certificate(self, lane: int) -> bool:
        """Reject empty, forged, nonfinite, or incomplete row evidence."""
        evidence = self.certificates[lane]
        if self.statuses[lane] != VerifyStatus.CERTIFIED or not evidence:
            return False
        if not all(isinstance(c, TerminalCertificate) and c.verify() for c in evidence):
            return False
        return (any(c.kind == "phase1_residual" for c in evidence)
                or sorted(c.row_index for c in evidence) == list(range(self._required_rows)))

    @property
    def certified(self) -> torch.Tensor:
        """Return the lane proof mask after independent evidence replay."""
        return torch.tensor([self.has_lp_certificate(k) for k in range(len(self.statuses))],
                            device=self.witnesses.device, dtype=torch.bool)


def _affine_reduce(p: _SparseLP, input_ids: tuple[int, ...],
                   deadline: Optional[float]) -> Optional[tuple[_SparseLP, dict[int, dict[int, Fraction]]]]:
    """Eliminate fixed-phase equations exactly; relax only float conversion.

    Sparse rational composition avoids treating rounded condensed matrices as
    the original model. The per-solve coefficient cap bounds this optional
    optimization's memory; generic solving remains available beyond the cap.
    """
    from scipy import sparse
    forms = {j: {i: Fraction(1)} for i, j in enumerate(input_ids)}
    pending = set(range(p.eq.shape[0]))
    work = 0
    while pending:
        progress = False
        for row in tuple(pending):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("terminal_deadline")
            start, end = p.eq.indptr[row:row + 2]
            terms = list(zip(p.eq.indices[start:end], p.eq.data[start:end]))
            unknown = [j for j, a in terms if a and j not in forms]
            if not unknown:
                pending.remove(row)
                continue
            if len(unknown) != 1:
                continue
            pivot = unknown[0]
            divisor = Fraction(float(next(a for j, a in terms if j == pivot)))
            result = {-1: Fraction(float(p.eq_rhs[row]))}
            for j, a in terms:
                if j == pivot or not a:
                    continue
                for i, coefficient in forms[j].items():
                    result[i] = result.get(i, Fraction(0)) - Fraction(float(a)) * coefficient
            forms[pivot] = {i: q / divisor for i, q in result.items() if q}
            work += len(forms[pivot])
            if work > 200_000:
                return None
            pending.remove(row)
            progress = True
        if not progress:
            break
    if len(forms) != len(p.lb):
        return None  # At least one phase is still genuinely unfixed.
    lb, ub = p.lb[list(input_ids)], p.ub[list(input_ids)]
    rows, cols, data, rhs = [], [], [], []

    def append(terms: list[tuple[int, float]], bound: float) -> None:
        """Outward-relax one substituted inequality in input coordinates."""
        exact = _compose(forms, terms)
        target = Fraction(float(bound)) - exact.pop(-1, Fraction(0))
        row = len(rhs)
        for i, coefficient in exact.items():
            value = float(coefficient)
            error = Fraction(value) - coefficient
            target += max(error * Fraction(float(lb[i])), error * Fraction(float(ub[i])))
            if value:
                rows.append(row)
                cols.append(i)
                data.append(value)
        rhs.append(_fraction_float(target, True))

    for matrix, targets, equality in ((p.le, p.le_rhs, False), (p.eq, p.eq_rhs, True)):
        for row in range(matrix.shape[0]):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("terminal_deadline")
            start, end = matrix.indptr[row:row + 2]
            terms = list(zip(matrix.indices[start:end], matrix.data[start:end]))
            append(terms, targets[row])
            if equality:
                append([(j, -a) for j, a in terms], -targets[row])
    for j in range(len(p.lb)):
        if j not in input_ids:
            append([(j, 1.0)], p.ub[j])
            append([(j, -1.0)], -p.lb[j])
    n = len(input_ids)
    reduced = _SparseLP(sparse.csr_matrix((0, n)), np.empty(0),
        sparse.csr_matrix((data, (rows, cols)), shape=(len(rhs), n)),
        np.asarray(rhs), lb.copy(), ub.copy())
    return reduced, forms


@dataclass(frozen=True)
class _ChainForms:
    """One lane's output as a float map of its inputs, with a rounding-error bound."""
    weight: np.ndarray
    constant: np.ndarray
    abs_weight: np.ndarray
    abs_constant: np.ndarray
    scale: np.ndarray
    gamma: float
    fixing_error: np.ndarray

    def row(self, coefficients: np.ndarray, threshold: float) -> tuple[np.ndarray, float, float]:
        """Return ``coefficients @ y`` as (input map, constant, bound on their joint error)."""
        magnitude = np.abs(coefficients)
        error = 2.0 * (self.gamma + (len(coefficients) + 2) * _UNIT_ROUNDOFF) * (
            magnitude @ self.abs_weight @ self.scale + magnitude @ self.abs_constant + abs(threshold)
        ) + magnitude @ self.fixing_error + _CHAIN_ERROR_FLOOR
        return coefficients @ self.weight, float(coefficients @ self.constant), float(error)


def _chain_reduce(
    head: _TerminalHeader, net: Net, input_bounds: Bounds,
    region_bounds: Optional[dict[int, Bounds]] = None,
) -> Optional[dict[int, tuple[_SparseLP, _ChainForms]]]:
    """Eliminate phases of whole lane batches on chain networks, in input space.

    Each layer composes the current affine map, so a neuron with no literal is
    fixed when the map's range over the box has one sign: that range covers the
    lane's region, and earlier phases already hold there. A lane with an open
    neuron is omitted, since no elimination could fix it either. Rows use float64
    composition with every right-hand side relaxed by a bound on its rounding
    error, so each LP relaxes its exact lane. ``None`` means not a chain network.
    ``region_bounds`` are sound ReLU input intervals on each lane's own region: they
    fix more phases, which makes the LP valid for that region and no wider one.
    """
    if not all(
        layer.kind in _CHAIN_KINDS and not layer.params.get("token_wise", False)
        and len(net.preds.get(layer.id, [])) == (layer.kind != "INPUT")
        for layer in net.layers
    ) or any(layer.params["kind"] not in _CHAIN_INPUT_KINDS for layer in head.specs):
        return None
    from scipy import sparse

    count = input_bounds.lb.shape[0]
    lower = input_bounds.lb.detach().double().cpu().flatten(1)
    upper = input_bounds.ub.detach().double().cpu().flatten(1)
    width = lower.shape[1]
    scale = torch.maximum(lower.abs(), upper.abs())
    relu_width = max((len(l.out_vars) for l in net.layers if l.kind == "RELU"), default=1)
    chunk = max(1, _CHAIN_MAX_ELEMENTS // max(1, relu_width * width))
    if chunk < count:
        reduced: dict[int, tuple[_SparseLP, _ChainForms]] = {}
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            part = replace(
                head, signs={lid: value[start:stop] for lid, value in head.signs.items()},
                spec_c=head.spec_c[start:stop], spec_d=head.spec_d[start:stop],
            )
            box = Bounds(input_bounds.lb[start:stop], input_bounds.ub[start:stop])
            region = None if region_bounds is None else {
                lid: Bounds(value.lb[start:stop], value.ub[start:stop]) for lid, value in region_bounds.items()
            }
            part_reduced = _chain_reduce(part, net, box, region)
            assert part_reduced is not None
            reduced.update((start + lane, lp) for lane, lp in part_reduced.items())
        return reduced
    weight = torch.eye(width, dtype=torch.float64).expand(count, -1, -1)
    constant = lower.new_zeros(count, width)
    abs_weight, abs_constant = weight, constant
    fixing_error = constant.clone()
    gamma = 0.0
    fixed = torch.ones(count, dtype=torch.bool)
    lanes, row_blocks, rhs_blocks = [], [], []
    for layer in net.layers:
        if layer.kind == "DENSE":
            matrix = layer.params["weight"].detach().double().cpu()
            bias = layer.params.get("bias")
            bias = lower.new_zeros(matrix.shape[0]) if bias is None else bias.detach().double().cpu().flatten()
            gamma += (matrix.shape[1] + 2) * _UNIT_ROUNDOFF
            weight, abs_weight = matrix @ weight, matrix.abs() @ abs_weight
            constant = constant @ matrix.T + bias
            abs_constant = abs_constant @ matrix.abs().T + bias.abs()
            fixing_error = fixing_error @ matrix.abs().T
        elif layer.kind == "RELU":
            gamma += (width + 2) * _UNIT_ROUNDOFF
            error = (
                2.0 * gamma * ((abs_weight @ scale[..., None]).squeeze(-1) + abs_constant)
                + fixing_error + _CHAIN_ERROR_FLOOR
            )
            positive, negative = weight.clamp(min=0), weight.clamp(max=0)
            low = (positive @ lower[..., None] + negative @ upper[..., None]).squeeze(-1) + constant - error
            high = (positive @ upper[..., None] + negative @ lower[..., None]).squeeze(-1) + constant + error
            own_low, own_high = low, high
            tolerance = torch.zeros_like(low)
            if region_bounds is not None and layer.id in region_bounds:
                region = region_bounds[layer.id]
                low = torch.maximum(low, region.lb.detach().flatten(1).double().cpu())
                high = torch.minimum(high, region.ub.detach().flatten(1).double().cpu())
                # Float32 passes round these intervals, so a phase they imply may be
                # off by this much near zero; charging it keeps every row a relaxation.
                tolerance = 4 * torch.finfo(region.lb.dtype).eps * (torch.maximum(low.abs(), high.abs()) + 1)
            literal = head.signs[layer.id].cpu().to(lower) if layer.id in head.signs else torch.zeros_like(low)
            active = (literal > 0) | ((literal == 0) & (low >= -tolerance))
            inactive = (literal < 0) | ((literal == 0) & ~active & (high <= tolerance))
            fixed &= (active | inactive).all(1)
            lane, neuron = torch.nonzero(literal, as_tuple=True)
            lanes.append(lane)
            row_blocks.append(-literal[lane, neuron, None] * weight[lane, neuron])
            rhs_blocks.append(literal[lane, neuron] * constant[lane, neuron] + error[lane, neuron])
            charged = (literal == 0) & ((active & (own_low < 0)) | (inactive & (own_high > 0)))
            weight, abs_weight = weight * active[..., None], abs_weight * active[..., None]
            constant, abs_constant = constant * active, abs_constant * active
            fixing_error = fixing_error * active + tolerance * charged
    lane_ids = torch.cat(lanes) if lanes else torch.empty(0, dtype=torch.long)
    rows = torch.cat(row_blocks).numpy() if lanes else np.empty((0, width))
    rhs = torch.cat(rhs_blocks).numpy() if lanes else np.empty(0)
    extra_rows, extra_rhs = [], []
    for layer in head.specs:
        if layer.params["kind"] == InKind.LIN_POLY:
            extra_rows.append(layer.params["A"].detach().double().cpu().numpy().reshape(-1, width))
            extra_rhs.append(layer.params["b"].detach().double().cpu().numpy().reshape(-1))
    unsafe = head.out_kind == OutKind.UNSAFE_LINEAR
    spec_c, spec_d = head.spec_c.double().cpu().numpy(), head.spec_d.double().cpu().numpy()
    grouped = lane_ids.numpy()
    result: dict[int, tuple[_SparseLP, _ChainForms]] = {}
    for k in torch.where(fixed)[0].tolist():
        forms = _ChainForms(
            weight[k].numpy(), constant[k].numpy(), abs_weight[k].numpy(), abs_constant[k].numpy(),
            scale[k].numpy(), gamma, fixing_error[k].numpy(),
        )
        own = grouped == k
        lane_rows, lane_rhs = [rows[own], *extra_rows], [rhs[own], *extra_rhs]
        if unsafe:
            # The whole conjunction shares this LP, each row with its own error.
            parts = [forms.row(spec_c[k, i], float(spec_d[k, i])) for i in range(spec_d.shape[1])]
            lane_rows.append(np.stack([part[0] for part in parts]))
            lane_rhs.append(np.array([spec_d[k, i] - parts[i][1] + parts[i][2] for i in range(len(parts))]))
        result[k] = (_SparseLP(
            sparse.csr_matrix((0, width)), np.empty(0),
            sparse.csr_matrix(np.vstack(lane_rows)), np.concatenate(lane_rhs),
            lower[k].numpy().copy(), upper[k].numpy().copy(),
        ), forms)
    return result


def _compose(forms: dict[int, dict[int, Fraction]],
             terms: list[tuple[int, float]]) -> dict[int, Fraction]:
    """Compose a sparse row using the affine tier's exact expressions."""
    result: dict[int, Fraction] = {}
    for j, a in terms:
        if a:
            for i, q in forms[j].items():
                result[i] = result.get(i, Fraction(0)) + Fraction(float(a)) * q
    return {i: q for i, q in result.items() if q}


def _objective(p: _SparseLP, output_ids: tuple[int, ...], row: np.ndarray,
               threshold: float, forms: Optional[dict[int, dict[int, Fraction]] | _ChainForms],
               ) -> tuple[np.ndarray, float]:
    """Return a conservative minimum of threshold minus one output row."""
    if isinstance(forms, _ChainForms):
        objective, constant, error = forms.row(-row, threshold)
        return objective, threshold + constant - error
    objective = np.zeros(len(p.lb))
    if forms is None:
        objective[list(output_ids)] = -row
        return objective, threshold
    exact = _compose(forms, list(zip(output_ids, -row)))
    constant = Fraction(threshold) + exact.pop(-1, Fraction(0))
    for i, q in exact.items():
        objective[i] = float(q)
        error = q - Fraction(objective[i])
        constant += min(error * Fraction(float(p.lb[i])), error * Fraction(float(p.ub[i])))
    return objective, _fraction_float(constant, False)


def _solve(p: _SparseLP, c: np.ndarray, deadline: Optional[float]) -> Any:
    """Call HiGHS at the CPU single-instance boundary with the shared budget."""
    from scipy.optimize import linprog
    remaining = None if deadline is None else deadline - time.monotonic()
    if remaining is not None and remaining <= 0:
        raise TimeoutError("terminal_deadline")
    options = {} if remaining is None else {"time_limit": remaining}
    return linprog(c, A_ub=p.le, b_ub=p.le_rhs, A_eq=p.eq, b_eq=p.eq_rhs,
                   bounds=np.column_stack((p.lb, p.ub)), method="highs", options=options)


def _certificate(p: _SparseLP, c: np.ndarray, constant: float, result: Any,
                 kind: str, out_kind: str, reference: torch.Tensor) -> Optional[TerminalCertificate]:
    """Treat returned dual arrays as untrusted data, irrespective of status."""
    eq = getattr(getattr(result, "eqlin", None), "marginals", None)
    le = getattr(getattr(result, "ineqlin", None), "marginals", None)
    if eq is None or le is None:
        return None
    lower = certified_lower_bound(p.eq, p.eq_rhs, p.le, p.le_rhs, p.lb, p.ub,
                                  c, np.asarray(eq), np.asarray(le), constant)
    if lower is None:
        return None
    certificate = TerminalCertificate(kind, lower, np.array(eq), np.array(le), p,
                                      c.copy(), constant, out_kind, reference)
    return certificate if certificate._crosses_boundary(lower) else None


def _phase_problem(p: _SparseLP) -> tuple[_SparseLP, np.ndarray]:
    """Construct a bounded phase-I LP over one extra violation column t.

    t relaxes every row (equalities in both directions). Feasible original
    points lift to t=0, so any residual-certified positive lower bound is a
    Farkas-style infeasibility certificate for the original LP.
    """
    from scipy import sparse
    rows = sparse.csr_matrix(sparse.vstack((p.le, p.eq, -p.eq)))
    assert rows.shape is not None
    rhs = np.concatenate((p.le_rhs, p.eq_rhs, -p.eq_rhs))
    # Bound the largest possible violation outward, so phase I is feasible
    # even when the original LP has no feasible point.
    # Proofs never rely on this cap: a feasible point always lifts to t=0.
    largest = np.maximum(rows.data * p.lb[rows.indices], rows.data * p.ub[rows.indices])
    owners = np.repeat(np.arange(rows.shape[0]), np.diff(rows.indptr))
    violation = np.bincount(owners, weights=largest, minlength=rows.shape[0]) - rhs
    cap = max(1.0, float(violation.max(initial=1.0))) * (1.0 + 1e-9)
    # Append the violation column in place: it is every row's last sorted entry.
    pointers = rows.indptr + np.arange(rows.shape[0] + 1)
    last = pointers[1:] - 1
    data, indices = np.empty(rows.nnz + rows.shape[0]), np.empty(rows.nnz + rows.shape[0], dtype=rows.indices.dtype)
    kept = np.ones(data.size, dtype=bool)
    kept[last] = False
    data[kept], indices[kept] = rows.data, rows.indices
    data[last], indices[last] = -1.0, len(p.lb)
    phase = _SparseLP(sparse.csr_matrix((0, len(p.lb) + 1)), np.empty(0),
        sparse.csr_matrix((data, indices, pointers), shape=(rows.shape[0], len(p.lb) + 1)), rhs,
        np.append(p.lb, 0.0), np.append(p.ub, cap))
    c = np.zeros(len(phase.lb))
    c[-1] = 1.0
    return phase, c


def _phase_one(p: _SparseLP, deadline: Optional[float], reference: torch.Tensor
               ) -> Optional[TerminalCertificate]:
    """Solve phase I and verify its positive minimum."""
    phase, c = _phase_problem(p)
    result = _solve(phase, c, deadline)
    return _certificate(phase, c, 0.0, result, "phase1_residual", OutKind.UNSAFE_LINEAR, reference)


def _elastic_problem(p: _SparseLP, c: np.ndarray, reference: float) -> tuple[_SparseLP, np.ndarray]:
    """Keep one inequality-form lane LP feasible inside a shared block.

    The extra column only helps solving: certificates are checked on the lane's
    own LP, valid for any multipliers, so it never enters a proof.
    """
    from scipy import sparse
    elastic = _SparseLP(
        sparse.csr_matrix((0, len(p.lb) + 1)), np.empty(0),
        sparse.hstack((p.le, -np.ones((p.le.shape[0], 1))), format="csr"), p.le_rhs,
        np.append(p.lb, 0.0), np.append(p.ub, np.inf),
    )
    return elastic, np.append(c, _ELASTIC_PENALTY * max(1.0, abs(reference)))


def _phase_point(result: Any, x: np.ndarray) -> SimpleNamespace:
    """Hand on a phase I point without its violation column."""
    return SimpleNamespace(status=result.status, x=x)


def _solve_blocks(
    jobs: list[tuple[tuple[int, int], _SparseLP, np.ndarray]], deadline: Optional[float],
) -> dict[tuple[int, int], Any]:
    """Solve inequality-form jobs block-diagonally, under lane and nonzero caps.

    Each job's last column is auxiliary. A result carries that job's own primal
    and row duals, plus ``exact`` when the column is zero, so the primal solves
    the unrelaxed lane LP. A failed block leaves its jobs to per-lane solving.
    """
    from scipy import sparse
    solved: dict[tuple[int, int], Any] = {}
    position = 0
    while position < len(jobs):
        stop, nonzeros = position, 0
        while stop < len(jobs) and stop - position < _BLOCK_MAX_LANES and (
            stop == position or nonzeros + jobs[stop][1].le.nnz <= _BLOCK_MAX_NONZEROS
        ):
            nonzeros += jobs[stop][1].le.nnz
            stop += 1
        chunk = jobs[position:stop]
        position = stop
        if len(chunk) < 2:
            continue
        columns = sum(len(lp.lb) for _, lp, _ in chunk)
        block = _SparseLP(
            sparse.csr_matrix((0, columns)), np.empty(0),
            sparse.block_diag([lp.le for _, lp, _ in chunk], format="csr"),
            np.concatenate([lp.le_rhs for _, lp, _ in chunk]),
            np.concatenate([lp.lb for _, lp, _ in chunk]),
            np.concatenate([lp.ub for _, lp, _ in chunk]),
        )
        limit = None
        if deadline is not None:
            now = time.monotonic()
            limit = now + _BLOCK_BUDGET_FRACTION * (deadline - now)
        try:
            result = _solve(block, np.concatenate([c for _, _, c in chunk]), limit)
        except (TimeoutError, ValueError):
            continue
        marginals = getattr(getattr(result, "ineqlin", None), "marginals", None)
        if getattr(result, "status", None) != 0 or marginals is None or result.x is None:
            continue
        column = row = 0
        for key, lp, _ in chunk:
            end_column, end_row = column + len(lp.lb), row + lp.le.shape[0]
            primal = result.x[column:end_column]
            solved[key] = SimpleNamespace(
                status=0, x=primal[:-1],
                eqlin=SimpleNamespace(marginals=np.empty(0)),
                ineqlin=SimpleNamespace(marginals=marginals[row:end_row]),
                exact=bool(primal[-1] <= 0),
            )
            column, row = end_column, end_row
    return solved


def solve_terminal_batch(
    net: Net, input_bounds: Bounds, split_signs: Optional[dict[int, torch.Tensor]] = None,
    *, reference_bounds: Optional[dict[int, Bounds]] = None,
    deadline: Optional[float] = None, prefer_affine: bool = True,
    lane_bases: Optional[dict[tuple[bytes, bytes], Any]] = None,
) -> TerminalResult:
    """Solve K terminal lanes, batching export and concrete candidate checks.

    Ordinary properties need every violation row bounded; UNSAFE_LINEAR
    instead asks whether the entire closed unsafe polytope is feasible.
    FALSIFIED is produced only after original-domain/concrete validation.
    ``prefer_affine=False`` keeps phase rows, as ``terminal_literal_costs`` needs.
    ``reference_bounds`` may fix phases of chain lanes, so a proof then covers only
    the region those intervals hold on; pass none for a proof on the whole box.
    """
    count = input_bounds.lb.shape[0]
    statuses = [VerifyStatus.UNKNOWN] * count
    reasons, tiers, kinds = ["unresolved"] * count, ["generic"] * count, [""] * count
    evidence: list[tuple[TerminalCertificate, ...]] = [()] * count
    witnesses = torch.full_like(input_bounds.lb, math.nan)
    candidates: list[torch.Tensor] = []
    required = 1
    try:
        from act.back_end.cons_exportor import TerminalExport, _terminal_header, export_terminal_problem
        from scipy import sparse
        if deadline is not None and (not math.isfinite(deadline) or time.monotonic() >= deadline):
            raise TimeoutError("terminal_deadline")
        chain: Optional[dict[int, tuple[_SparseLP, _ChainForms]]] = None
        if prefer_affine:
            with torch.device("cpu"):
                head = _terminal_header(net, input_bounds, split_signs)
                chain = _chain_reduce(head, net, input_bounds, reference_bounds)
        if chain is not None and len(chain) == count:
            # Every lane is eliminated, so no sparse lane export is needed.
            exported = TerminalExport(
                head.input_ids, head.output_ids, head.spec_c, head.spec_d, head.out_kind, net,
                input_bounds, head.signs, (), (), (), (),
            )
        else:
            exported = export_terminal_problem(
                net, input_bounds, split_signs, reference_bounds=reference_bounds, lane_bases=lane_bases,
            )
    except (ImportError, NotImplementedError, ValueError, TimeoutError) as error:
        reason = "no_solver" if isinstance(error, ImportError) else str(error)
        status = VerifyStatus.TIMEOUT if isinstance(error, TimeoutError) else VerifyStatus.UNKNOWN
        return TerminalResult((status,) * count, witnesses, tuple(evidence), (reason,) * count,
                              tuple(tiers), tuple(kinds))
    unsafe = exported.out_kind == OutKind.UNSAFE_LINEAR
    required = 1 if unsafe else exported.spec_d.shape[1]
    spec_c, spec_d = exported.spec_c.cpu().numpy(), exported.spec_d.cpu().numpy()

    def lane_failure(k: int, error: Exception) -> None:
        """Record why one lane stayed open; no other lane is affected."""
        if isinstance(error, TimeoutError):
            statuses[k], reasons[k] = VerifyStatus.TIMEOUT, "terminal_deadline"
        elif isinstance(error, ImportError):
            reasons[k] = "no_solver"
        else:
            reasons[k] = f"numerical_failure:{error}"

    prepared: dict[int, tuple[_SparseLP, Optional[dict[int, dict[int, Fraction]] | _ChainForms]]] = dict(chain or {})
    tiers = ["affine" if k in prepared else tier for k, tier in enumerate(tiers)]
    for k, original in enumerate(exported._lanes):
        if k in prepared:
            continue
        p = original
        try:
            if unsafe:
                # Append the entire conjunction BEFORE affine elimination so
                # all rows, including their rounding error, share one owner.
                rows = sparse.lil_matrix((len(spec_d[k]), len(p.lb)))
                rows[:, list(exported.output_ids)] = spec_c[k]
                p = replace(p, le=sparse.vstack((p.le, rows), format="csr"),
                            le_rhs=np.concatenate((p.le_rhs, spec_d[k])))
            reduction = (_affine_reduce(p, exported.input_ids, deadline)
                         if prefer_affine and chain is None else None)
            forms = None
            if reduction is not None:
                p, forms = reduction
                tiers[k] = "affine"
            prepared[k] = (p, forms)
        except (TimeoutError, ImportError, ValueError, OverflowError) as error:
            lane_failure(k, error)
    # Phase I is always a pure inequality LP and decides feasibility on its own, so
    # every unsafe lane blocks it; elsewhere only eliminated lanes block, row by row.
    jobs: list[tuple[tuple[int, int], _SparseLP, np.ndarray]] = []
    phase_problems: dict[int, tuple[_SparseLP, np.ndarray]] = {}
    objectives: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
    for k, (p, forms) in prepared.items():
        try:
            for row in range(1 if unsafe else required):
                if unsafe:
                    objectives[k, row] = (np.zeros(len(p.lb)), 0.0)
                    phase_problems[k] = _phase_problem(p)
                    jobs.append(((k, row), *phase_problems[k]))
                else:
                    objectives[k, row] = _objective(
                        p, exported.output_ids, spec_c[k, row], float(spec_d[k, row]), forms,
                    )
                    if forms is not None:
                        jobs.append(((k, row), *_elastic_problem(
                            p, objectives[k, row][0], float(spec_d[k, row]),
                        )))
        except (ValueError, OverflowError) as error:
            lane_failure(k, error)
    blocks = _solve_blocks(jobs, deadline)
    for k, (p, forms) in prepared.items():
        try:
            proofs = []
            for row in range(1 if unsafe else required):
                if (k, row) not in objectives:
                    continue
                reference = input_bounds.lb.new_tensor(max(1.0, abs(float(spec_d[k, row]))))
                c, constant = objectives[k, row]
                block = blocks.get((k, row))
                certificate = None
                if block is not None and unsafe:
                    phase, phase_c = phase_problems[k]
                    certificate = _certificate(phase, phase_c, 0.0, block, "phase1_residual",
                                               OutKind.UNSAFE_LINEAR, reference)
                    if certificate is not None:
                        proofs = [certificate]
                        break
                elif block is not None:
                    certificate = _certificate(p, c, constant, block, "lp_residual",
                                               exported.out_kind, reference)
                exact = block is not None and block.exact
                result = block
                if certificate is None and not exact:
                    if unsafe:
                        # Phase I alone decides a conjunction: its positive minimum is
                        # the proof, and a zero minimum carries a feasible point.
                        phase, phase_c = _phase_problem(p)
                        result = _solve(phase, phase_c, deadline)
                        certificate = _certificate(phase, phase_c, 0.0, result, "phase1_residual",
                                                   OutKind.UNSAFE_LINEAR, reference)
                        if certificate is not None:
                            proofs = [certificate]
                            break
                        if getattr(result, "x", None) is not None:
                            result = _phase_point(result, result.x[:-1])
                    else:
                        result = _solve(p, c, deadline)
                        certificate = _certificate(
                            p, c, constant, result, "lp_residual", exported.out_kind, reference)
                if certificate is not None:
                    proofs.append(replace(certificate, row_index=row))
                    continue
                assert result is not None
                if getattr(result, "x", None) is not None:
                    x = np.asarray(result.x)
                    selected = x if forms is not None else x[list(exported.input_ids)]
                    while len(candidates) <= row:
                        candidates.append(torch.full_like(input_bounds.lb, math.nan))
                    candidates[row][k] = torch.as_tensor(selected, device=witnesses.device,
                                                         dtype=witnesses.dtype).reshape_as(witnesses[k])
                if not unsafe and getattr(result, "status", None) == 2:
                    certificate = _phase_one(p, deadline, reference)
                    if certificate is not None:
                        proofs = [certificate]
                        break
                reasons[k] = "solver_timeout" if getattr(result, "status", None) == 1 else "uncertified_lp"
                if getattr(result, "status", None) == 1:
                    statuses[k] = VerifyStatus.TIMEOUT
            if proofs and (len(proofs) == required or proofs[0].kind == "phase1_residual"):
                evidence[k] = tuple(proofs)
                statuses[k], reasons[k], kinds[k] = VerifyStatus.CERTIFIED, "verified_certificate", proofs[0].kind
        except (TimeoutError, ImportError, ValueError, OverflowError) as error:
            lane_failure(k, error)
    for batch in candidates:
        if deadline is not None and time.monotonic() >= deadline:
            for k in range(count):
                if statuses[k] == VerifyStatus.UNKNOWN:
                    statuses[k], reasons[k] = VerifyStatus.TIMEOUT, "terminal_deadline"
            break
        try:
            valid = validate_counterexamples(
                exported.net, batch, box=exported.input_bounds,
                split_signs=exported.split_signs, tolerance=0.0,
            )
        except (ValueError, RuntimeError, NotImplementedError) as error:
            for k in range(count):
                if statuses[k] == VerifyStatus.UNKNOWN:
                    reasons[k] = f"candidate_validation_failed:{error}"
            continue
        for k in torch.nonzero(valid, as_tuple=False).flatten().tolist():
            statuses[k], witnesses[k] = VerifyStatus.FALSIFIED, batch[k]
            evidence[k], reasons[k], kinds[k] = (), "validated_counterexample", ""
    return TerminalResult(tuple(statuses), witnesses, tuple(evidence), tuple(reasons),
                          tuple(tiers), tuple(kinds), required, exported)


def terminal_literal_costs(
    result: TerminalResult, lane: int,
) -> Optional[tuple[tuple[tuple[int, int, int], ...], np.ndarray, np.ndarray]]:
    """Charge each literal from a checked, uneliminated source LP's duals.

    Returns ``(literal keys, costs [rows, literals], slack [rows])``, or None
    when a dual cannot be attributed to its original row. Costs only propose a
    subset: a changed core still needs its own LP certificate. Deleting a row
    keeps the ReLU triangle, on which ``z-x`` (active) or ``z`` (inactive) lies
    in ``[0, reach]`` with reach ``max(0,-l)`` or ``max(0,u)``; that is the
    violation range of the sign row too, unlike the chord intercept at zero.
    Since ``c = A.T*y + r``, only a negative multiplier loses bound.
    """
    export = result._export
    if export is None or result.tiers[lane] != "generic" or not result.has_lp_certificate(lane):
        return None
    original = export._lanes[lane]
    equalities = original.eq.shape[0]
    phases = export._phase_rows[lane]
    proofs = sorted(result.certificates[lane], key=lambda proof: proof.row_index)
    costs = np.zeros((len(proofs), len(phases)))
    le_rows = np.array([phase.le_row for phase in phases], dtype=np.int64)
    eq_rows = np.array([phase.eq_row for phase in phases], dtype=np.int64)
    pre_vars = np.array([phase.pre_var for phase in phases], dtype=np.int64)
    post_vars = np.array([phase.post_var for phase in phases], dtype=np.int64)
    signs = np.array([phase.sign for phase in phases], dtype=np.float64)
    reach = np.where(signs > 0, -original.lb[pre_vars], original.ub[pre_vars]).clip(min=0.0)
    for row, proof in enumerate(proofs):
        problem = proof._problem
        # Phase I stacks (le, eq, -eq) under one artificial column; equality
        # multipliers split into two nonpositive inequality multipliers.
        phase_one = proof.kind == "phase1_residual"
        inequalities = problem.le.shape[0] - (2 * equalities if phase_one else 0)
        if len(proof.le_dual) != problem.le.shape[0] or (
            len(proof.eq_dual) != (0 if phase_one else equalities)
        ):
            return None
        eq_matrix, eq_index = (problem.le, inequalities + eq_rows) if phase_one else (problem.eq, eq_rows)
        pre_in_eq = np.asarray(eq_matrix[eq_index, pre_vars]).ravel() if len(phases) else np.empty(0)
        if len(phases) and (
            not np.array_equal(np.asarray(problem.le[le_rows, pre_vars]).ravel(), -signs)
            or not np.all(np.asarray(eq_matrix[eq_index, post_vars]).ravel() == 1)
            or not np.all((pre_in_eq == 0) | (pre_in_eq == -1))
        ):
            return None
        equality = (
            proof.le_dual[eq_index] - proof.le_dual[eq_index + equalities] if phase_one
            else proof.eq_dual[eq_index]
        )
        costs[row] = ((-proof.le_dual[le_rows]).clip(min=0.0) + (-equality).clip(min=0.0)) * reach
    if not np.isfinite(costs).all():
        return None
    return (
        tuple((phase.layer_id, phase.neuron, phase.sign) for phase in phases), costs,
        np.array([proof.lower_bound for proof in proofs]),
    )
