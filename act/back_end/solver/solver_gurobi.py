
from __future__ import annotations

import math
import time
from typing import Any, Optional, TYPE_CHECKING
import numpy as np
import torch
import os
from act.back_end.core import Net
from act.back_end.solver.solver_base import (
    Solver,
    SolverCaps,
    SolveStatus,
)
from act.config.config import BaBConfig, GurobiConfig
from act.front_end.specs import OutKind
from act.util.path_config import get_project_root
from act.util.stats import VerifyResult, VerifyStatus

if TYPE_CHECKING:
    from act.back_end.cons_exportor import TerminalExport
    from act.back_end.solver.solver_base import BatchLPProblem, BatchLPSolution

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    print("Warning: Gurobi not available. Some operations will use alternative solvers.")
    gp = None
    GRB = None
    GUROBI_AVAILABLE = False


def is_gurobi_available() -> bool:
    return bool(GUROBI_AVAILABLE)

def setup_gurobi_license():
    """Setup Gurobi license path based on current folder layout."""
    if 'GRB_LICENSE_FILE' not in os.environ:
        if 'ACTHOME' in os.environ:
            license_path = os.path.join(os.environ['ACTHOME'], 'modules', 'gurobi', 'gurobi.lic')
            print(f"[ACT] Using ACTHOME environment variable: {os.path.relpath(os.environ['ACTHOME'])}")
        else:
            project_root = get_project_root()
            license_path = os.path.join(project_root, 'modules', 'gurobi', 'gurobi.lic')
            print(f"[ACT] Auto-detecting project root: {os.path.relpath(project_root)}")
        
        license_path = os.path.abspath(license_path)
        
        if os.path.exists(license_path):
            os.environ['GRB_LICENSE_FILE'] = license_path
            print(f"[ACT] Gurobi license found: {os.path.relpath(license_path)}")
        else:
            print(f"[WARN] Gurobi license not found: {os.path.relpath(license_path)}")
            print(f"[INFO] Please place gurobi.lic in: {os.path.relpath(os.path.dirname(license_path))}")
    else:
        print(f"[ACT] Using existing Gurobi license: {os.path.relpath(os.environ['GRB_LICENSE_FILE'])}")

setup_gurobi_license()


class GurobiSolver(Solver):
    """Gurobi backend for exact LP/MILP solving (CPU-only)."""

    def capabilities(self) -> SolverCaps:
        return SolverCaps(False)

    def __init__(self, config: Optional[GurobiConfig] = None):
        self._cfg: GurobiConfig = config or GurobiConfig()
        if not GUROBI_AVAILABLE:
            raise RuntimeError("gurobipy is not available in this environment.")

    def solve_batch(
        self,
        problem: "BatchLPProblem",  # noqa: F821
        timelimit: Optional[float] = None,
    ) -> "BatchLPSolution":  # noqa: F821
        """Solve a batch of N independent LPs.

        N=1: builds a single gp.Model, solves, returns BatchLPSolution[N=1].
        N>1: raises NotImplementedError — Gurobi's multi-scenario API does not
             expose truly parallel solving for varying constraint matrices.
        """
        from act.back_end.solver.solver_base import BatchLPSolution
        import torch

        if problem.N != 1:
            raise NotImplementedError(
                f"GurobiSolver.solve_batch: N={problem.N} not supported. "
                f"Gurobi does not expose a truly parallel multi-LP API for "
                f"varying constraint matrices. Use TorchLPSolver for N>1, "
                f"or constrain BaB to bab_max_batch_size=1 and "
                f"verify_lp_batched is skipped (set lp_enabled=False)."
            )

        nvars = problem.nvars
        lb = problem.lb[0].cpu().numpy().astype(np.float64)
        ub = problem.ub[0].cpu().numpy().astype(np.float64)

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", self._cfg.output_flag)
        env.start()
        m = gp.Model("verify_batch_n1", env=env)
        m.Params.MIPGap = self._cfg.mip_gap
        m.Params.Threads = self._cfg.threads
        x = m.addMVar(nvars, lb=lb, ub=ub, name="x")

        # Decompose block-diagonal sparse: for N=1 the block is the full matrix.
        if problem.m_eq > 0:
            A_eq = (
                problem.A_eq_blockdiag.to_dense()[: problem.m_eq, :nvars]
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            b_eq = problem.b_eq[0].cpu().numpy().astype(np.float64)
            m.addConstr(A_eq @ x == b_eq)

        if problem.m_le > 0:
            A_le = (
                problem.A_le_blockdiag.to_dense()[: problem.m_le, :nvars]
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            b_le = problem.b_le[0].cpu().numpy().astype(np.float64)
            m.addConstr(A_le @ x <= b_le)

        obj_c = problem.obj_c[0].cpu().numpy().astype(np.float64)
        obj_const = float(problem.obj_const[0].item())
        sense = GRB.MINIMIZE if problem.sense == "min" else GRB.MAXIMIZE
        m.setObjective(obj_c @ x + obj_const, sense)

        active_timelimit = self._cfg.time_limit if self._cfg.time_limit is not None else timelimit
        if active_timelimit is not None:
            m.Params.TimeLimit = float(active_timelimit)
        m.optimize()

        dtype = problem.lb.dtype
        device = problem.lb.device

        if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            status = SolveStatus.SAT
            x_val = torch.as_tensor(x.X, dtype=dtype, device=device).unsqueeze(0)
            max_viol = torch.zeros(1, dtype=dtype, device=device)
        elif m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            status = SolveStatus.UNSAT
            x_val = torch.zeros_like(problem.lb)
            max_viol = torch.full((1,), float("inf"), dtype=dtype, device=device)
        else:
            status = SolveStatus.UNKNOWN
            x_val = torch.zeros_like(problem.lb)
            max_viol = torch.full((1,), float("nan"), dtype=dtype, device=device)

        return BatchLPSolution(
            statuses=(status,),
            x=x_val,
            max_viol=max_viol,
        )

def _optimize(
    exported: TerminalExport,
    relus: list[tuple[int, int, float, float]],
    stop_at: float,
) -> tuple[str, torch.Tensor | None]:
    """Build exact ReLUs over exported affine rows and solve within the deadline."""
    if not GUROBI_AVAILABLE:
        return "unavailable", None
    assert gp is not None and GRB is not None
    grb = GRB
    problem = exported.problem
    # Reuse the numerical owner's sparse lane; never densify exported blocks.
    lane = exported._lanes[0]
    upper_rows = set(exported.relu_upper_rows)
    keep = [row for row in range(problem.m_le) if row not in upper_rows]
    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model("act_root_milp", env=env) as model:
                model.Params.Threads = 1
                model.Params.DualReductions = 0
                x = model.addMVar(problem.nvars, lb=lane.lb, ub=lane.ub, name="x")
                if problem.m_eq:
                    model.addConstr(lane.eq @ x == lane.eq_rhs)
                if keep:
                    model.addConstr(lane.le[keep] @ x <= lane.le_rhs[keep])
                for pre, post, lower, upper in relus:
                    if lower >= 0.0:
                        model.addConstr(x[post] <= x[pre])
                    elif upper <= 0.0:
                        model.addConstr(x[post] <= 0.0)
                    else:
                        active = model.addVar(vtype=grb.BINARY, name=f"relu_{post}")
                        model.addConstr(x[post] <= upper * active)
                        model.addConstr(x[post] <= x[pre] - lower * (1.0 - active))
                # UNSAFE is a closed conjunction. Never shift its boundary to
                # make a zero-margin unsafe point disappear from the MILP.
                c = exported.spec_c[0].detach().cpu().numpy()
                d = exported.spec_d[0].detach().cpu().numpy()
                model.addConstr(c @ x[list(exported.output_ids)] <= d)
                model.setObjective(0.0)
                remaining = stop_at - time.monotonic()
                if remaining <= 0.0:
                    return "deadline", None
                model.Params.TimeLimit = remaining
                model.optimize()
                if model.Status == grb.INFEASIBLE:
                    return "infeasible", None
                candidate = None
                if model.SolCount > 0:
                    candidate = torch.as_tensor(
                        x.X[list(exported.input_ids)],
                        device=exported.input_bounds.lb.device,
                        dtype=exported.input_bounds.lb.dtype,
                    ).reshape_as(exported.input_bounds.lb)
                status = {
                    grb.OPTIMAL: "optimal",
                    grb.TIME_LIMIT: "time_limit",
                    grb.INF_OR_UNBD: "infeasible_or_unbounded",
                }.get(model.Status, f"solver_status_{model.Status}")
                return status, candidate
    except gp.GurobiError as error:
        reason = "license_unavailable" if error.errno == grb.Error.NO_LICENSE else "solver_error"
        return f"{reason}: {error}", None

def solve_milp(net: Net, config: BaBConfig, *, deadline: float) -> VerifyResult:
    """Attempt optional root escalation; preserve an explicit reason on every exit."""
    started = time.monotonic()
    details: dict[str, Any] = {"attempted": False}

    def finish(
        reason: str,
        status: VerifyStatus = VerifyStatus.UNKNOWN,
        candidate: torch.Tensor | None = None,
    ) -> VerifyResult:
        """Attach escalation provenance without admitting a CLIMB certificate."""
        details.update(reason=reason, elapsed_s=time.monotonic() - started)
        metadata: dict[str, Any] = {"milp_escalation": dict(details)}
        if status == VerifyStatus.CERTIFIED:
            metadata["certificate_kind"] = "milp_status_infeasible"
        return VerifyResult(status, counterexample=candidate, metadata=metadata)

    if not config.milp_escalation:
        return finish("disabled")
    if not math.isfinite(deadline) or not math.isfinite(config.milp_timeout) or config.milp_timeout <= 0:
        return finish("invalid_time_budget")
    stop_at = min(deadline, started + config.milp_timeout)
    if started >= stop_at:
        return finish("deadline")
    from act.util.counterexample import validate_counterexamples
    from act.back_end.cons_exportor import export_terminal_problem
    from act.back_end.verifier import gather_input_spec_layers, seed_from_input_specs

    nvars = 1 + max((v for layer in net.layers for v in layer.out_vars), default=-1)
    if nvars > config.milp_max_variables:
        return finish("variable_limit")
    try:
        bounds = seed_from_input_specs(gather_input_spec_layers(net))
        if bounds.lb.shape[0] != 1:
            return finish("unsupported_batch")
        exported = export_terminal_problem(net, bounds)
    except ImportError as error:
        return finish(f"unavailable_dependency: {error}")
    except (NotImplementedError, ValueError) as error:
        return finish(f"unsupported: {error}")
    if exported.out_kind != OutKind.UNSAFE_LINEAR:
        return finish("unsupported_output")
    problem = exported.problem
    finite = (
        problem.lb, problem.ub, problem.b_eq, problem.b_le,
        problem.A_eq_blockdiag.coalesce().values(),
        problem.A_le_blockdiag.coalesce().values(), exported.spec_c, exported.spec_d,
    )
    if not all(bool(torch.isfinite(tensor).all()) for tensor in finite):
        return finish("nonfinite_encoding")
    relus = [
        (pre, post, float(problem.lb[0, pre]), float(problem.ub[0, pre]))
        for layer in exported.relu_layers
        for pre, post in zip(layer.in_vars, layer.out_vars, strict=True)
    ]
    if len(exported.relu_upper_rows) != len(relus):
        return finish("incomplete_relu_encoding")
    unstable = sum(lower < 0.0 < upper for _, _, lower, upper in relus)
    nvars = problem.nvars + unstable
    ncons = problem.m_eq + problem.m_le + unstable + exported.spec_d.shape[1]
    details.update(variables=nvars, constraints=ncons, binaries=unstable)
    if nvars > config.milp_max_variables:
        return finish("variable_limit")
    if ncons > config.milp_max_constraints:
        return finish("constraint_limit")
    if time.monotonic() >= stop_at:
        return finish("deadline")
    details["attempted"] = True
    solver_status, candidate = _optimize(exported, relus, stop_at)
    details["solver_status"] = solver_status
    if solver_status == "infeasible":
        return finish("infeasible", VerifyStatus.CERTIFIED)
    if candidate is not None:
        if bool(validate_counterexamples(
            exported.net, candidate, box=exported.input_bounds,
            split_signs=exported.split_signs, tolerance=0.0,
        ).all()):
            return finish("validated_incumbent", VerifyStatus.FALSIFIED, candidate)
        return finish("invalid_incumbent")
    return finish(solver_status)
