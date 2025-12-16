
from __future__ import annotations
from typing import List, Optional
import numpy as np
import os
from act.back_end.solver.solver_base import Solver, SolverCaps, SolveStatus
from act.util.path_config import get_project_root

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    print("Warning: Gurobi not available. Some operations will use alternative solvers.")
    GUROBI_AVAILABLE = False

def setup_gurobi_license():
    """Setup Gurobi license path based on current folder layout."""
    if 'GRB_LICENSE_FILE' not in os.environ:
        if 'ACTHOME' in os.environ:
            license_path = os.path.join(os.environ['ACTHOME'], 'modules', 'gurobi', 'gurobi.lic')
            print(f"[ACT] Using ACTHOME environment variable: {os.environ['ACTHOME']}")
        else:
            project_root = get_project_root()
            license_path = os.path.join(project_root, 'modules', 'gurobi', 'gurobi.lic')
            print(f"[ACT] Auto-detecting project root: {project_root}")
        
        license_path = os.path.abspath(license_path)
        
        if os.path.exists(license_path):
            os.environ['GRB_LICENSE_FILE'] = license_path
            print(f"[ACT] Gurobi license found and set: {license_path}")
        else:
            print(f"[WARN] Gurobi license not found at: {license_path}")
            print(f"[INFO] Please ensure gurobi.lic is placed in: {os.path.dirname(license_path)}")
    else:
        print(f"[ACT] Using existing Gurobi license: {os.environ['GRB_LICENSE_FILE']}")

setup_gurobi_license()


class GurobiSolver(Solver):
    """Gurobi backend for exact LP/MILP solving (CPU-only)."""

    def capabilities(self) -> SolverCaps:
        return SolverCaps(False)

    def __init__(self):
        if not GUROBI_AVAILABLE:
            raise RuntimeError("gurobipy is not available in this environment.")
        self.m = None
        self._x = []

    @property
    def n(self) -> int:
        return len(self._x)

    def begin(self, name: str = "verify", device: Optional[str] = None):
        # device hint ignored (CPU solver)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 0
        self.m.Params.DualReductions = 0
        self._x = []

    def add_vars(self, n: int) -> None:
        new = self.m.addVars(n, lb=-GRB.INFINITY, ub=+GRB.INFINITY, name="x")
        self._x.extend(list(new.values()))

    def set_bounds(self, idxs: List[int], lb: np.ndarray, ub: np.ndarray) -> None:
        for idx, lo, hi in zip(idxs, lb, ub):
            self._x[idx].LB = float(lo)
            self._x[idx].UB = float(hi)

    def add_binary_vars(self, n: int) -> List[int]:
        start = len(self._x)
        new = self.m.addVars(n, vtype=GRB.BINARY, name="b")
        self._x.extend(list(new.values()))
        return list(range(start, start + n))

    def _lexpr(self, vids: List[int], coeffs: List[float]):
        e = gp.LinExpr()
        for i, a in zip(vids, coeffs):
            e.addTerms(float(a), self._x[i])
        return e

    def add_lin_eq(self, vids: List[int], coeffs: List[float], rhs: float) -> None:
        self.m.addConstr(self._lexpr(vids, coeffs) == float(rhs))

    def add_lin_le(self, vids: List[int], coeffs: List[float], rhs: float) -> None:
        self.m.addConstr(self._lexpr(vids, coeffs) <= float(rhs))

    def add_lin_ge(self, vids: List[int], coeffs: List[float], rhs: float) -> None:
        self.m.addConstr(self._lexpr(vids, coeffs) >= float(rhs))

    def add_sum_eq(self, vids: List[int], rhs: float) -> None:
        self.m.addConstr(gp.quicksum(self._x[i] for i in vids) == float(rhs))

    def add_ge_zero(self, vids: List[int]) -> None:
        for i in vids:
            self.m.addConstr(self._x[i] >= 0.0)

    def add_sos2(self, var_ids: List[int], weights: Optional[List[float]] = None) -> None:
        self.m.addSOS(GRB.SOS_TYPE2, [self._x[i] for i in var_ids], weights)

    def set_objective_linear(self, vids: List[int], coeffs: List[float], const: float = 0.0, sense: str = "min") -> None:
        e = self._lexpr(vids, coeffs) + float(const)
        self.m.setObjective(e, GRB.MINIMIZE if sense == "min" else GRB.MAXIMIZE)

    def optimize(self, timelimit: Optional[float] = None) -> None:
        if timelimit is not None:
            self.m.Params.TimeLimit = float(timelimit)
        self.m.update()
        self.m.optimize()

        if self.m.Status in [3, 4, 5, 12]:  # infeasible, inf_or_unbd, unbounded, numeric
            self.m.write("debug_model.lp")
        # If still INF_OR_UNBD, disable DualReductions and rerun once
        if self.m.Status == GRB.INF_OR_UNBD:
            # Handle only on the first encounter
            self.m.Params.DualReductions = 0
            self.m.optimize()

    def status(self) -> str:
        """
        Map Gurobi status to {SAT, UNSAT, UNKNOWN}.

        Semantics (from the ACT framework perspective):
          - SAT    : At least one feasible solution satisfies all constraints
                     (in a “find counterexample” model, this means a CE candidate exists)
          - UNSAT  : Gurobi proves infeasibility (no feasible solutions)
                     (in a “find counterexample” model, proves no counterexample exists)
          - UNKNOWN: All other situations:
                     - not started / interrupted with no incumbent
                     - numeric issues
                     - infeasible-or-unbounded / truly unbounded
                     - any case where feasibility cannot be trusted

        Notes:
        - Return UNSAT only when Gurobi explicitly reports INFEASIBLE.
        - If there is a solution but not optimal (time/node limits or interruption):
          with an incumbent (SolCount > 0) we treat as SAT (a feasible point exists),
          because existence of a feasible solution does not depend on optimality.
        - Numeric issues or (INF_OR_UNBD / UNBOUNDED) are always UNKNOWN, even if
          SolCount > 0, to preserve soundness.
        """
        if self.m is None:
            return SolveStatus.UNKNOWN

        st = self.m.Status
        try:
            solcnt = int(getattr(self.m, "SolCount", 0))
        except Exception:
            solcnt = 0

        # 1. Explicitly infeasible
        if st == GRB.INFEASIBLE:
            return SolveStatus.UNKNOWN

        # 2. Numeric issues / infeasible-or-unbounded / unbounded ⇒ untrusted, treat as UNKNOWN
        if st in (GRB.INF_OR_UNBD, GRB.UNBOUNDED, GRB.NUMERIC):
            return SolveStatus.UNKNOWN

        # 3. Normal termination, proven feasible solution exists
        if st in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            if solcnt > 0:
                return SolveStatus.SAT
            # In theory OPTIMAL/SUBOPTIMAL implies a solution; if not, stay conservative
            return SolveStatus.UNKNOWN

        # 4. Limits triggered but an incumbent exists:
        early_stop_with_incumbent = {
            GRB.TIME_LIMIT,
            GRB.NODE_LIMIT,
            GRB.ITERATION_LIMIT,
            GRB.SOLUTION_LIMIT,
            GRB.INTERRUPTED,
            GRB.CUTOFF,
            GRB.USER_OBJ_LIMIT,
        }
        if st in early_stop_with_incumbent:
            if solcnt > 0:
                return SolveStatus.SAT
            # No incumbent means feasibility cannot be determined
            return SolveStatus.UNKNOWN

        # 5. All other statuses (LOADED, NOT_STARTED, INPROGRESS, etc.) → UNKNOWN
        return SolveStatus.UNKNOWN


    def has_solution(self) -> bool:
        return self.m is not None and getattr(self.m, 'SolCount', 0) > 0

    def get_values(self, vids: List[int]) -> np.ndarray:
        return np.array([self._x[i].X for i in vids], dtype=float)

    def get_counterexample(self, input_ids: List[int]) -> np.ndarray:
        # Gurobi returns exact LP/MILP solutions; just proxy.
        return self.get_values(input_ids)
