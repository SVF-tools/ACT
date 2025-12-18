from __future__ import annotations
import math, os, time
from typing import List, Optional, Tuple
import numpy as np
import torch

from act.back_end.solver.solver_base import Solver, SolveStatus, SolverCaps
from act.util.device_manager import get_default_device, get_default_dtype

class TorchLPSolver(Solver):
    """Continuous LP solver using Torch + Adam with penalty and box projection.

    - Supports GPU via device hint in begin(...).
    - No integrality: add_binary_vars() creates [0,1] continuous vars.
    - add_sos2 is a no-op.
    """
    def __init__(self):
        self._device = get_default_device()
        self._dtype = get_default_dtype()
        self._n = 0
        self._x = None                 # torch.nn.Parameter
        self._lb = None                # torch.Tensor [n]
        self._ub = None                # torch.Tensor [n]
        self._eq = []                  # rows: (vids, coeffs, rhs)
        self._le = []
        self._ge = []
        self._objective = ([], [], 0.0, "min")
        self._status = SolveStatus.UNKNOWN
        self._has_solution = False
        self._sol = None
        self._timelimit = None

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

    @property
    def n(self) -> int:
        return self._n

    def begin(self, name: str = "verify", device: Optional[str] = None):
        # Use global device manager for default, allow override
        if device is not None:
            self._device = torch.device(device)
        # else keep the device_manager default from __init__
        
        self._n = 0
        self._x = None
        self._lb = None
        self._ub = None
        self._eq.clear(); self._le.clear(); self._ge.clear()
        self._objective = ([], [], 0.0, "min")
        self._status = SolveStatus.UNKNOWN
        self._has_solution = False
        self._sol = None

    def add_vars(self, n: int) -> None:
        if n <= 0:
            return
        if self._n == 0:
            self._n = n
            # Create tensors on the correct device and dtype
            self._lb = torch.full((n,), -np.inf, device=self._device, dtype=self._dtype)
            self._ub = torch.full((n,), +np.inf, device=self._device, dtype=self._dtype)
        else:
            old_n = self._n
            self._n += n
            # Extend tensors on the correct device and dtype
            self._lb = torch.cat([self._lb, torch.full((n,), -np.inf, device=self._device, dtype=self._dtype)])
            self._ub = torch.cat([self._ub, torch.full((n,), +np.inf, device=self._device, dtype=self._dtype)])

    def add_binary_vars(self, n: int) -> List[int]:
        start = self._n
        self.add_vars(n)
        idxs = list(range(start, start + n))
        # relax to [0,1]
        self._lb[idxs] = 0.0
        self._ub[idxs] = 1.0
        return idxs

    def set_bounds(self, idxs: List[int], lb: np.ndarray, ub: np.ndarray) -> None:
        # Convert to tensors with correct device and dtype
        lb_t = torch.as_tensor(lb, device=self._device, dtype=self._dtype)
        ub_t = torch.as_tensor(ub, device=self._device, dtype=self._dtype)
        self._lb[idxs] = lb_t
        self._ub[idxs] = ub_t

    def add_lin_eq(self, vids: List[int], coeffs: List[float], rhs: float) -> None:
        self._eq.append((vids, coeffs, rhs))

    def add_lin_le(self, vids: List[int], coeffs: List[float], rhs: float) -> None:
        self._le.append((vids, coeffs, rhs))

    def add_lin_ge(self, vids: List[int], coeffs: List[float], rhs: float) -> None:
        vids2 = vids
        coeffs2 = [-float(a) for a in coeffs]
        rhs2 = -float(rhs)
        self._le.append((vids2, coeffs2, rhs2))

    def add_sum_eq(self, vids: List[int], rhs: float) -> None:
        coeffs = [1.0] * len(vids)
        self.add_lin_eq(vids, coeffs, rhs)

    def add_ge_zero(self, vids: List[int]) -> None:
        for i in vids:
            self.add_lin_le([i], [-1.0], 0.0)

    def add_sos2(self, var_ids: List[int], weights: Optional[List[float]] = None) -> None:
        return  # no-op

    def set_objective_linear(self, vids: List[int], coeffs: List[float], const: float = 0.0, sense: str = "min") -> None:
        self._objective = (vids, coeffs, float(const), "min" if sense != "max" else "max")

    def optimize(self, timelimit: Optional[float] = None) -> None:
        self._timelimit = timelimit
        t_end = None if timelimit is None else (time.time() + float(timelimit))

        eff_max_iter = self.max_iter
        eff_tol_feas = self.tol_feas
        if self._n > self._large_n_threshold:
            eff_max_iter = min(eff_max_iter, self._large_n_max_iter)
            eff_tol_feas = max(eff_tol_feas, self._large_n_tol)

        # initialize x at box center (or zeros where infinite)
        if self._x is None:
            lb = torch.where(torch.isfinite(self._lb), self._lb, torch.zeros_like(self._lb))
            ub = torch.where(torch.isfinite(self._ub), self._ub, torch.zeros_like(self._ub))
            mid = 0.5 * (lb + ub)
            both_inf = (~torch.isfinite(self._lb)) & (~torch.isfinite(self._ub))
            mid = torch.where(both_inf, torch.zeros_like(mid), mid)
            self._x = torch.nn.Parameter(mid.clone().to(device=self._device, dtype=self._dtype), requires_grad=True)
            # self._x = torch.nn.Parameter(mid.to(device=self._device, dtype=self._dtype), requires_grad=True)
        else:
            self._x = torch.nn.Parameter(self._x.detach().to(device=self._device, dtype=self._dtype), requires_grad=True)

        vids, coeffs, c0, sense = self._objective
        is_max = (sense == "max")
        if vids:
            vids_t = torch.as_tensor(vids, device=self._device, dtype=torch.long)
            coeffs_t = torch.as_tensor(coeffs, device=self._device, dtype=self._dtype)
        else:
            vids_t = None
            coeffs_t = None
        const_term = float(c0)

        lb = self._lb.to(device=self._device, dtype=self._dtype)
        ub = self._ub.to(device=self._device, dtype=self._dtype)

        def build_sparse(rows):
            if not rows:
                zero = torch.zeros((0,), device=self._device, dtype=self._dtype)
                return None, zero
            m = len(rows)
            b = torch.empty((m,), device=self._device, dtype=self._dtype)
            ri: List[int] = []
            ci: List[int] = []
            vals: List[float] = []
            for r, (vids_row, coeffs_row, rhs) in enumerate(rows):
                b[r] = float(rhs)
                ri.extend([r] * len(vids_row))
                ci.extend(vids_row)
                vals.extend(coeffs_row)
            if not vals:
                idx = torch.zeros((2, 0), device=self._device, dtype=torch.long)
                val_t = torch.zeros((0,), device=self._device, dtype=self._dtype)
            else:
                idx = torch.tensor([ri, ci], device=self._device, dtype=torch.long)
                val_t = torch.as_tensor(vals, device=self._device, dtype=self._dtype)
            A = torch.sparse_coo_tensor(idx, val_t, (m, self._n), device=self._device, dtype=self._dtype)
            return A.coalesce(), b

        Aeq, beq = build_sparse(self._eq)
        Ale, ble = build_sparse(self._le)

        opt = torch.optim.Adam([self._x], lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        self._status = SolveStatus.UNKNOWN
        self._has_solution = False
        best_max_viol = math.inf
        stagnation_steps = 0
        viol_eq = None
        viol_le = None

        for it in range(eff_max_iter):
            if t_end is not None and time.time() >= t_end:
                break

            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                obj = 0.001 * (self._x * self._x).sum() + const_term
                if vids_t is not None and coeffs_t is not None and vids_t.numel() > 0:
                    obj = obj + torch.dot(coeffs_t, self._x.index_select(0, vids_t))
                if is_max:
                    obj = -obj

                viol_eq = None if Aeq is None else torch.sparse.mm(Aeq, self._x.unsqueeze(1)).squeeze(1) - beq
                if viol_eq is not None:
                    obj = obj + self.rho_eq * (viol_eq * viol_eq).sum()

                viol_le = None if Ale is None else torch.sparse.mm(Ale, self._x.unsqueeze(1)).squeeze(1) - ble
                if viol_le is not None:
                    obj = obj + self.rho_ineq * (torch.relu(viol_le) ** 2).sum()

                obj.backward()
            opt.step()

            with torch.no_grad():
                self._x.data.clamp_(lb, ub)

            recompute = (it % self._feas_check_stride == 0)
            with torch.no_grad():
                if recompute:
                    v_eq = None if Aeq is None else torch.sparse.mm(Aeq, self._x.unsqueeze(1)).squeeze(1) - beq
                    v_le = None if Ale is None else torch.sparse.mm(Ale, self._x.unsqueeze(1)).squeeze(1) - ble
                else:
                    v_eq = viol_eq
                    v_le = viol_le

                max_viol = 0.0
                if v_eq is not None and v_eq.numel() > 0:
                    max_viol = max(max_viol, float(v_eq.abs().max().item()))
                if v_le is not None and v_le.numel() > 0:
                    max_viol = max(max_viol, float(torch.relu(v_le).max().item()))

            if max_viol < best_max_viol - self._stagnation_tol:
                best_max_viol = max_viol
                stagnation_steps = 0
            else:
                stagnation_steps += 1

            if max_viol <= eff_tol_feas or stagnation_steps >= self._stagnation_patience:
                break

        self._status = SolveStatus.SAT
        self._has_solution = True
        self._sol = self._x.detach().clone()

    def status(self) -> str:
        return self._status

    def has_solution(self) -> bool:
        return bool(self._has_solution)

    def get_values(self, vids: List[int]) -> np.ndarray:
        assert self._sol is not None, "No solution available"
        with torch.no_grad():
            return self._sol[vids].detach().cpu().to(self._dtype).numpy()

    def get_counterexample(self, input_ids: List[int]) -> np.ndarray:
        # Already projected to box during optimize(); can return directly.
        return self.get_values(input_ids)
