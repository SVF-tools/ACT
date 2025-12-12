
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
        print("Gurobi m.Status :", self.m.Status)
        print("Gurobi SolCount :", self.m.SolCount)

        if self.m.Status in [3, 4, 5, 12]:  # infeasible, inf_or_unbd, unbounded, numeric
            self.m.write("debug_model.lp")
            print("[DEBUG] wrote LP model to debug_model.lp")
        # ★ 如果仍然拿到了 INF_OR_UNBD，再关掉 DualReductions 重跑一遍
        if self.m.Status == GRB.INF_OR_UNBD:
            # 仅在第一次遇到时处理即可
            self.m.Params.DualReductions = 0
            self.m.optimize()

    def status(self) -> str:
        """
        Map Gurobi status to {SAT, UNSAT, UNKNOWN}.

        语义约定（从 ACT 框架角度）：
          - SAT    : 至少找到一个满足所有约束的可行解
                     （在“找反例”的模型中 = 找到了反例候选）
          - UNSAT  : Gurobi 严格证明模型不可行（无任何可行解）
                     （在“找反例”的模型中 = 证明不存在反例）
          - UNKNOWN: 其他所有情况：
                     - 未开始 / 被中断且无 incumbent
                     - 数值问题
                     - 不可行-或-无界 / 真无界
                     - 没法确信可行性结论的情况

        注意：
        - 只有在 Gurobi 明确给出 INFEASIBLE 时才返回 UNSAT。
        - 有解但未最优（时间限制 / 节点限制 / 中断）：
          如果有 incumbent（SolCount > 0），我们认为 SAT（存在可行点），
          因为“存在可行解”并不依赖最优性。
        - 数值问题或 (INF_OR_UNBD / UNBOUNDED) 一律 UNKNOWN，
          即使 SolCount > 0 也不信任，保证 soundness。
        """
        if self.m is None:
            return SolveStatus.UNKNOWN

        st = self.m.Status
        try:
            solcnt = int(getattr(self.m, "SolCount", 0))
        except Exception:
            solcnt = 0

        # 1. 明确不可行
        if st == GRB.INFEASIBLE:
            return SolveStatus.UNKNOWN

        # 2. 数值问题 / 不可行或无界 / 真无界 ⇒ 不可信，统一 UNKNOWN
        #    即便 SolCount > 0 也不要冒险当作 SAT
        if st in (GRB.INF_OR_UNBD, GRB.UNBOUNDED, GRB.NUMERIC):
            return SolveStatus.UNKNOWN

        # 3. 正常终止，证明存在可行解
        #    OPTIMAL: 有一个被证明为最优的可行解
        #    SUBOPTIMAL: 有可行 incumbent，但由于数值/界裁剪等原因未完全证明最优
        if st in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            if solcnt > 0:
                return SolveStatus.SAT
            # 理论上 OPTIMAL/SUBOPTIMAL 一定有解；如果没有，就保守 UNKNOWN
            return SolveStatus.UNKNOWN

        # 4. 限制条件触发但有 incumbent 的情况：
        #    TIME_LIMIT, NODE_LIMIT, ITERATION_LIMIT, SOLUTION_LIMIT,
        #    INTERRUPTED, CUTOFF, USER_OBJ_LIMIT 等。
        #    —— 只要有 incumbent，就说明“存在一个可行解”，可以当作 SAT。
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
            # 没有任何 incumbent，则无法判断可行性
            return SolveStatus.UNKNOWN

        # 5. 其他所有状态（例如 LOADED, NOT_STARTED, INPROGRESS 等）一律 UNKNOWN
        return SolveStatus.UNKNOWN


    def has_solution(self) -> bool:
        return self.m is not None and getattr(self.m, 'SolCount', 0) > 0

    def get_values(self, vids: List[int]) -> np.ndarray:
        return np.array([self._x[i].X for i in vids], dtype=float)

    def get_counterexample(self, input_ids: List[int]) -> np.ndarray:
        # Gurobi returns exact LP/MILP solutions; just proxy.
        return self.get_values(input_ids)
