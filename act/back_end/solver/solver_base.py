
#===- act/back_end/solver/solver_base.py - Base Solver Interface -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Base Solver Interface. Defines abstract base class and common interfaces
#   for constraint satisfaction problem solvers.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    import torch
    from act.back_end.core import Bounds, Net


class SolveStatus:
    """Solver status codes (SAT/UNSAT terminology for verification)."""
    SAT = "SAT"              # Satisfiable - solution found
    UNSAT = "UNSAT"          # Unsatisfiable - no solution exists
    UNKNOWN = "UNKNOWN"      # Inconclusive (timeout/spurious/error)

class SolverCaps:
    def __init__(self, supports_gpu: bool = False, supports_csp: bool = True, supports_hz: bool = False, supports_dual: bool = False):
        self.supports_gpu = supports_gpu
        self.supports_csp = supports_csp
        self.supports_hz = supports_hz
        self.supports_dual = supports_dual

class BoundResult(ABC):
    """Result of one certified-bound call.

    Concrete solvers retain their optimization state on their result types;
    branching depends only on the certified row bounds and optional witnesses.
    """

    @abstractmethod
    def row_lower_bounds(self) -> "torch.Tensor":
        """Return the certified lower bound for each packed specification row."""

    @abstractmethod
    def row_witness_inputs(self) -> Optional["torch.Tensor"]:
        """Return a witness input per packed row when the solver produced one."""


class BatchBoundResult(ABC):
    """Decoded result of one K-lane output-spec batch.

    The interface is consumed by Branch-and-Bound. Solver-specific warm-start
    state remains on the concrete result type.
    """

    @abstractmethod
    def lane_solution(self) -> "BatchLPSolution":
        """Return the per-lane feasibility solution."""

    @abstractmethod
    def lane_lower_bounds(self) -> "torch.Tensor":
        """Return the certified lower bound of the worst row per lane."""

    @abstractmethod
    def spec_row_slack(self) -> Optional["torch.Tensor"]:
        """Return per-lane specification-row slack when it is available."""

    @abstractmethod
    def lane_witness_inputs(self) -> Optional["torch.Tensor"]:
        """Return a candidate input per lane when it is available."""

    @abstractmethod
    def intermediate_bounds(self) -> Optional[Dict[int, "Bounds"]]:
        """Return intermediate bounds used by branching when available."""

    @abstractmethod
    def neuron_sensitivities(self) -> Optional[Dict[int, "torch.Tensor"]]:
        """Return per-neuron branching sensitivities when available."""


class Solver:
    """Abstract solver interface used by the exporter and verification pipeline.

    Three **alternative** capability surfaces (not orthogonal axes): every
    current subclass implements exactly one of the three primary capability
    flags (``supports_csp`` / ``supports_hz`` / ``supports_dual``), so the
    surfaces partition solvers into specialised families rather than
    composing freely. Subclasses implement only the methods that match
    their declared capability and leave the rest raising
    ``NotImplementedError``:

      * ``compute_bounds(domain_obj)`` — domain-element → box bounds.
        Implemented by ``GurobiSolver`` and ``HZSolver``.
      * ``solve_batch(BatchLPProblem)`` — batched LP/MILP. Implemented by
        ``GurobiSolver``, ``TorchLPSolver``, ``HZSolver``.
      * ``compute_certified_bound(net, bounds_dict, c)`` — linear-relaxation
        dual lower bound on ``c @ output``. Implemented by ``DualSolver``.

    Callers dispatch by reading ``capabilities()`` rather than relying on
    method presence; ``raise NotImplementedError`` is the contract for
    "this solver does not support that surface".
    """

    # --- Capabilities / lifecycle ---
    def capabilities(self) -> SolverCaps:  # pragma: no cover - abstract
        return SolverCaps(False)

    # --- Domain-based bounds ---
    def compute_bounds(self, domain_obj) -> object:  # pragma: no cover - abstract
        raise NotImplementedError

    # --- Batched API [BATCHED-API] ---
    def solve_batch(
        self,
        problem: "BatchLPProblem",
        timelimit: Optional[float] = None,
    ) -> "BatchLPSolution":
        raise NotImplementedError(f"{type(self).__name__}.solve_batch")

    # --- Certified backward-dual bound [DUAL-API] ---
    def compute_certified_bound(
        self,
        net: "Net",
        bounds_dict: "Dict[int, Bounds]",
        c: "torch.Tensor",
        M: int = 1,
        return_sce: bool = False,
        enable_grad: bool = False,
    ) -> BoundResult:
        """Sound certified lower bound ``L <= min_x c @ f(x)`` for x in the
        input region described by ``bounds_dict``, computed via backward
        propagation through ``net``'s graph.

        Args:
            net: ACT Net (DAG-aware; reverse-topo'd internally).
            bounds_dict: forward-pass bounds keyed by layer id, batched
                ``[B, *shape]``.
            c: ``Tensor[B*M, num_classes]`` linear coefficients for the
                objective ``c @ output``. When ``M > 1`` (multi-spec-row
                verification) rows are packed sample-major (``b*M+j``).
            M: number of spec rows packed into ``c``'s leading axis per
                sample. DualSolver-specific (lazy M-broadcast);
                LP/MILP solvers ignore this. Defaults to 1.
            return_sce: if True, expose an input witness per packed row via
                ``BoundResult.row_witness_inputs()``. Defaults to False.
            enable_grad: if True, allow gradient flow through the bound
                computation (for adversarial / robust-training loops).

        Returns:
            A :class:`BoundResult` with one certified lower bound per packed
            (sample, specification) row.

        Subclasses with ``capabilities().supports_dual == True`` must
        override this; LP/MILP-only solvers inherit the NotImplementedError.
        """
        raise NotImplementedError(f"{type(self).__name__}.compute_certified_bound")


# --- Batched API types [BATCHED-API] ---

from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass(frozen=True)
class BatchLPProblem:
    """N independent linear programs sharing variable-id schema.

    All tensors have leading batch dim N. nvars is shared across N
    (caller aligns). Constraint counts m_eq, m_le are uniform across N
    via canonical row emission in cons_exportor (no per-N ragged shapes).

    Sparse storage convention:
        A_eq_blockdiag and A_le_blockdiag are 2-D torch.sparse_coo_tensors
        with shape (N*m_eq, N*nvars) and (N*m_le, N*nvars) respectively,
        constructed so that block (i, i) holds the i-th instance's
        constraint rows. This is the only sparse layout torch.sparse.mm
        handles efficiently as of PyTorch 2.x.
    """

    nvars: int
    m_eq: int
    m_le: int
    lb: torch.Tensor
    ub: torch.Tensor
    A_eq_blockdiag: torch.Tensor
    b_eq: torch.Tensor
    A_le_blockdiag: torch.Tensor
    b_le: torch.Tensor
    obj_c: torch.Tensor
    obj_const: torch.Tensor
    sense: str = "min"

    @property
    def N(self) -> int:
        return int(self.lb.shape[0])

    def __post_init__(self) -> None:
        N = self.N
        if self.lb.shape != (N, self.nvars):
            raise ValueError(f"lb shape {tuple(self.lb.shape)} != ({N}, {self.nvars})")
        if self.ub.shape != (N, self.nvars):
            raise ValueError(f"ub shape {tuple(self.ub.shape)} != ({N}, {self.nvars})")
        if self.b_eq.shape != (N, self.m_eq):
            raise ValueError(f"b_eq shape {tuple(self.b_eq.shape)} != ({N}, {self.m_eq})")
        if self.b_le.shape != (N, self.m_le):
            raise ValueError(f"b_le shape {tuple(self.b_le.shape)} != ({N}, {self.m_le})")
        if self.obj_c.shape != (N, self.nvars):
            raise ValueError(f"obj_c shape {tuple(self.obj_c.shape)} != ({N}, {self.nvars})")
        if self.obj_const.shape != (N,):
            raise ValueError(f"obj_const shape {tuple(self.obj_const.shape)} != ({N},)")
        if not self.A_eq_blockdiag.is_sparse:
            raise ValueError("A_eq_blockdiag must be sparse")
        if self.A_eq_blockdiag.shape != (N * self.m_eq, N * self.nvars):
            raise ValueError(
                f"A_eq_blockdiag shape {tuple(self.A_eq_blockdiag.shape)} "
                f"!= ({N * self.m_eq}, {N * self.nvars})"
            )
        if not self.A_le_blockdiag.is_sparse:
            raise ValueError("A_le_blockdiag must be sparse")
        if self.A_le_blockdiag.shape != (N * self.m_le, N * self.nvars):
            raise ValueError(
                f"A_le_blockdiag shape {tuple(self.A_le_blockdiag.shape)} "
                f"!= ({N * self.m_le}, {N * self.nvars})"
            )
        if self.sense not in ("min", "max"):
            raise ValueError(f"sense={self.sense!r}")


@dataclass(frozen=True)
class BatchLPSolution:
    """Per-N status + iterate.

    statuses[i] in {SAT, UNSAT, UNKNOWN}. SAT means x[i] is feasible
    within max_viol[i] <= tol_feas; iterate clamped to [lb, ub]. UNSAT
    means the backend proved infeasibility (only Gurobi N=1 produces
    UNSAT; TorchLPSolver cannot conclude UNSAT and uses UNKNOWN per
    soundness invariant — penalty-on-Adam cannot certify infeasibility).
    """

    statuses: Tuple[str, ...]
    x: torch.Tensor
    max_viol: torch.Tensor

    def __post_init__(self) -> None:
        N = int(self.x.shape[0])
        if len(self.statuses) != N:
            raise ValueError(f"statuses len {len(self.statuses)} != N={N}")
        if self.max_viol.shape != (N,):
            raise ValueError(f"max_viol shape {tuple(self.max_viol.shape)} != ({N},)")
        valid = {"SAT", "UNSAT", "UNKNOWN"}
        for i, s in enumerate(self.statuses):
            if s not in valid:
                raise ValueError(f"statuses[{i}]={s!r} not in {valid}")
