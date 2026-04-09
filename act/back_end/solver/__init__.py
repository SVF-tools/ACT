#===- act/back_end/solver/__init__.py - Constraint Solvers --------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Solvers for constraint satisfaction. Provides various solver implementations
#   including Gurobi and PyTorch-based solvers.
#
#===---------------------------------------------------------------------===#

from .solver_base import Solver, SolverCaps, SolveStatus
from .solver_torch import TorchLPSolver
from .solver_gurobi import GurobiSolver
from .hz_bounds import hz_compute_bounds

__all__ = [
    'Solver', 'SolverCaps', 'SolveStatus',
    'TorchLPSolver', 'GurobiSolver',
    'hz_compute_bounds',
]