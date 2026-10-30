#===- act/back_end/solver/milp_escalate.py - Optional exact MILP -----------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2026 ACT Team
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
"""Small, single-lane exact ReLU MILP with explicit solver-status trust.

This separate module isolates optional Gurobi escalation from certified LP
proofs: exact solver statuses may resolve an instance, but cannot justify
CLIMB cores. It also keeps Gurobi off the default verification path.

This module is imported only after the optional escalation gate opens. It
shares export, support checks and concrete validation with terminal LP; it
does not produce residual certificates or CLIMB cores.
"""

import math
import time
from typing import Any

import torch

from act.back_end.core import Net
from act.back_end.solver.terminal_lp import (
    TerminalExport,
    export_terminal_problem,
    validate_terminal_candidates,
)
from act.config.config import BaBConfig
from act.front_end.specs import OutKind
from act.util.path_config import import_gurobi
from act.util.stats import VerifyResult, VerifyStatus


def _optimize(
    exported: TerminalExport,
    relus: list[tuple[int, int, float, float]],
    stop_at: float,
) -> tuple[str, torch.Tensor | None]:
    """Build exact ReLUs over exported affine rows and solve within the deadline."""
    available, gp, grb = import_gurobi(ensure_license=True)
    if not available:
        return "unavailable", None
    assert gp is not None and grb is not None
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
    from act.back_end.verifier import gather_input_spec_layers, seed_from_input_specs

    # Reject oversized variable schemas before allocating export matrices.
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
        if bool(validate_terminal_candidates(exported, candidate).all()):
            return finish("validated_incumbent", VerifyStatus.FALSIFIED, candidate)
        return finish("invalid_incumbent")
    return finish(solver_status)
