from __future__ import annotations
import math, time
from typing import Optional
import torch

from act.back_end.solver.solver_base import (
    Solver,
    SolveStatus,
    SolverCaps,
    BatchLPProblem,
    BatchLPSolution,
    _empty_blockdiag,
)
from act.util.device_manager import get_default_device, get_default_dtype

class TorchLPSolver(Solver):
    """Continuous LP solver using Torch + Adam with penalty and box projection.

    - Supports GPU via device hint in begin(...).
    - LP-only: no integrality constraints (no binary vars, no SOS2).
    """
    def __init__(self):
        self._device = get_default_device()
        self._dtype = get_default_dtype()
        # parameters
        self.rho_eq = 10.0
        self.rho_ineq = 10.0
        self.max_iter = 2000  # lighter default; see large-n overrides below
        self.tol_feas = 1e-4
        self.lr = 1e-2
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.weight_decay = 0.0
        self._large_n_threshold = 20000
        self._large_n_max_iter = 800
        self._large_n_tol = 1e-3
        self._log_every = 200
        self._stagnation_patience = 300
        self._stagnation_tol = 1e-5
        self._feas_check_stride = 5

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
        stagnation_steps = 0
        cached_v_eq: Optional[torch.Tensor] = None
        cached_v_le: Optional[torch.Tensor] = None

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
                    cached_v_eq = v_eq_mat.detach()
                if has_le and A_le is not None:
                    Ax_le = torch.sparse.mm(A_le, x_flat).squeeze(1)
                    v_le_mat = Ax_le.view(N, m_le) - b_le
                    obj = obj + self.rho_ineq * (torch.relu(v_le_mat) ** 2).sum()
                    cached_v_le = v_le_mat.detach()
                obj.backward()
            opt.step()

            with torch.no_grad():
                x.data.clamp_(lb, ub)

            with torch.no_grad():
                if it % self._feas_check_stride == 0:
                    x_flat_no = x.reshape(N * nvars).unsqueeze(1)
                    if has_eq and A_eq is not None:
                        Ax_eq_no = torch.sparse.mm(A_eq, x_flat_no).squeeze(1)
                        cached_v_eq = Ax_eq_no.view(N, m_eq) - b_eq
                    if has_le and A_le is not None:
                        Ax_le_no = torch.sparse.mm(A_le, x_flat_no).squeeze(1)
                        cached_v_le = Ax_le_no.view(N, m_le) - b_le

                max_viol_per_n = torch.zeros(N, device=device, dtype=dtype)
                if has_eq and cached_v_eq is not None:
                    max_viol_per_n = torch.maximum(
                        max_viol_per_n,
                        cached_v_eq.abs().max(dim=1).values,
                    )
                if has_le and cached_v_le is not None:
                    max_viol_per_n = torch.maximum(
                        max_viol_per_n,
                        torch.relu(cached_v_le).max(dim=1).values,
                    )
                global_max_viol = float(max_viol_per_n.max().item())

            if global_max_viol < best_max_viol - self._stagnation_tol:
                best_max_viol = global_max_viol
                stagnation_steps = 0
            else:
                stagnation_steps += 1

            if (
                global_max_viol <= eff_tol_feas
                or stagnation_steps >= self._stagnation_patience
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


def _test_solve_batch_n1_sat_feasible():
    N, nvars = 1, 1
    problem = BatchLPProblem(
        nvars=nvars,
        m_eq=0,
        m_le=0,
        lb=torch.zeros(N, nvars),
        ub=torch.ones(N, nvars),
        A_eq_blockdiag=_empty_blockdiag(N, 0, nvars),
        b_eq=torch.zeros(N, 0),
        A_le_blockdiag=_empty_blockdiag(N, 0, nvars),
        b_le=torch.zeros(N, 0),
        obj_c=torch.tensor([[1.0]]),
        obj_const=torch.zeros(N),
    )
    sol = TorchLPSolver().solve_batch(problem)
    assert sol.statuses == (SolveStatus.SAT,), f"got {sol.statuses}"
    assert sol.x.shape == (N, nvars), f"got shape {tuple(sol.x.shape)}"
    assert sol.max_viol.shape == (N,)
    assert float(sol.max_viol[0].item()) <= 1e-4, (
        f"max_viol[0]={float(sol.max_viol[0].item())} > tol"
    )
    x00 = float(sol.x[0, 0].item())
    assert 0.0 <= x00 <= 1.0, f"x[0,0]={x00} outside [0, 1]"


def _test_solve_batch_n4_homogeneous():
    N, nvars = 4, 1
    problem = BatchLPProblem(
        nvars=nvars,
        m_eq=0,
        m_le=0,
        lb=torch.zeros(N, nvars),
        ub=torch.ones(N, nvars),
        A_eq_blockdiag=_empty_blockdiag(N, 0, nvars),
        b_eq=torch.zeros(N, 0),
        A_le_blockdiag=_empty_blockdiag(N, 0, nvars),
        b_le=torch.zeros(N, 0),
        obj_c=torch.ones(N, nvars),
        obj_const=torch.zeros(N),
    )
    sol = TorchLPSolver().solve_batch(problem)
    assert sol.statuses == (SolveStatus.SAT,) * N, f"got {sol.statuses}"
    assert sol.x.shape == (N, nvars)
    assert sol.max_viol.shape == (N,)
    for i in range(N):
        assert float(sol.max_viol[i].item()) <= 1e-4
        assert 0.0 <= float(sol.x[i, 0].item()) <= 1.0
    x0 = sol.x[0, 0].item()
    for i in range(1, N):
        assert sol.x[i, 0].item() == x0, (
            f"sub-LP {i} diverged from sub-LP 0: {sol.x[i,0].item()} vs {x0}"
        )


def _test_solve_batch_n4_mixed():
    N, nvars = 4, 1
    m_le = 1
    # Block-diagonal A_le shape (N*m_le, N*nvars) = (4, 4). Instances 0,1 have
    # no non-zero row (trivial 0 <= 1); instances 2,3 encode x >= 3 (i.e.
    # -x <= -3), infeasible against box [0, 1]. Block (i, i) lives at row i,
    # column i because nvars = m_le = 1.
    indices = torch.tensor([[2, 3], [2, 3]], dtype=torch.long)
    values = torch.tensor([-1.0, -1.0])
    A_le = torch.sparse_coo_tensor(indices, values, (N * m_le, N * nvars))
    b_le = torch.tensor([[1.0], [1.0], [-3.0], [-3.0]])
    problem = BatchLPProblem(
        nvars=nvars,
        m_eq=0,
        m_le=m_le,
        lb=torch.zeros(N, nvars),
        ub=torch.ones(N, nvars),
        A_eq_blockdiag=_empty_blockdiag(N, 0, nvars),
        b_eq=torch.zeros(N, 0),
        A_le_blockdiag=A_le,
        b_le=b_le,
        obj_c=torch.zeros(N, nvars),
        obj_const=torch.zeros(N),
    )
    sol = TorchLPSolver().solve_batch(problem)
    assert sol.statuses == (
        SolveStatus.SAT,
        SolveStatus.SAT,
        SolveStatus.UNKNOWN,
        SolveStatus.UNKNOWN,
    ), f"got {sol.statuses}"
    assert sol.x.shape == (N, nvars)
    assert sol.max_viol.shape == (N,)


def _test_solve_batch_max_viol_correct():
    """SAT instances have max_viol <= tol_feas; UNKNOWN have max_viol > tol_feas."""
    N, nvars = 4, 1
    m_le = 1
    indices = torch.tensor([[2, 3], [2, 3]], dtype=torch.long)
    values = torch.tensor([-1.0, -1.0])
    A_le = torch.sparse_coo_tensor(indices, values, (N * m_le, N * nvars))
    b_le = torch.tensor([[1.0], [1.0], [-3.0], [-3.0]])
    problem = BatchLPProblem(
        nvars=nvars,
        m_eq=0,
        m_le=m_le,
        lb=torch.zeros(N, nvars),
        ub=torch.ones(N, nvars),
        A_eq_blockdiag=_empty_blockdiag(N, 0, nvars),
        b_eq=torch.zeros(N, 0),
        A_le_blockdiag=A_le,
        b_le=b_le,
        obj_c=torch.zeros(N, nvars),
        obj_const=torch.zeros(N),
    )
    solver = TorchLPSolver()
    sol = solver.solve_batch(problem)
    tol = solver.tol_feas
    sat_viols = [float(sol.max_viol[i].item()) for i in (0, 1)]
    unk_viols = [float(sol.max_viol[i].item()) for i in (2, 3)]
    for v in sat_viols:
        assert v <= tol, f"SAT instance had max_viol={v} > tol={tol}"
    for v in unk_viols:
        assert v > tol, f"UNKNOWN instance had max_viol={v} <= tol={tol}"


if __name__ == "__main__":
    import sys
    tests = [
        _test_solve_batch_n1_sat_feasible,
        _test_solve_batch_n4_homogeneous,
        _test_solve_batch_n4_mixed,
        _test_solve_batch_max_viol_correct,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(failed)
