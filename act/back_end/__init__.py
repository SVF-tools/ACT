# ===- act/back_end/__init__.py - ACT Backend Verification Framework ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Torch-native DNN verification abstraction framework. Provides core
#   data structures (Bounds, Con, ConSet, Fact, Layer, Net).
#
# ===---------------------------------------------------------------------===#

"""
- Transfer functions for MLP and Transformer layers
- Worklist-based bounds propagation analysis
- Constraint export to LP/MILP solvers
- Branch-and-bound verification with spurious CE elimination
- Input/output specification handling
- Device-aware tensor management

Example usage:
    >>> from act.back_end import *
    >>> # Create verification problem
    >>> I = InputSpec(kind=InKind.BOX, lb=torch.tensor([-1.0]), ub=torch.tensor([1.0]))
    >>> O = OutputSpec(kind=OutKind.MARGIN_ROBUST, y_true=0, margin=0.0)
    >>> # Run verification
    >>> result = verify_once(net, entry_id, input_ids, output_ids, I, O, seed_bounds, solver)
"""

# Core data structures
from .core import Bounds, Con, ConSet, Fact, Layer, Net

# Device management (moved to front_end)
# from .device_manager import initialize_device_dtype, ensure_initialized, summary, temp_device_dtype, wrap_model_fn

# Utilities
from .utils import box_join, changed_or_maskdiff, update_cache, affine_bounds
from .utils import pwl_meta, bound_var_interval, scale_interval

# Transfer function interface
from .transfer_functions import (
    TransferFunction,
    set_transfer_function,
    get_transfer_function,
    set_transfer_function_mode,
    dispatch_tf,
)

# Transfer function implementations
from .interval_tf import IntervalTF
from .hybridz_tf import HybridzTF

# Analysis algorithms
from .analyze import analyze, initialize_tf_mode

# Analysis algorithms
from .analyze import dispatch_tf, analyze

# Solver interfaces
from .solver.solver_base import Solver, SolverCaps, SolveStatus
from .solver.solver_gurobi import GurobiSolver

# Verification types (canonical location: act/util/stats.py)
from act.util.stats import VerifyStatus, VerifyResult

# Note: TorchLPSolver and some verification functions are available
# via direct import to avoid circular dependencies:
# from act.back_end.verifier import verify_once
# from act.back_end.solver.solver_interval import TorchLPSolver

__all__ = [
    # Core
    "Bounds",
    "Con",
    "ConSet",
    "Fact",
    "Layer",
    "Net",
    # Device management
    "initialize_device_dtype",
    "ensure_initialized",
    "summary",
    "temp_device_dtype",
    "wrap_model_fn",
    # Utilities
    "box_join",
    "changed_or_maskdiff",
    "update_cache",
    "affine_bounds",
    "pwl_meta",
    "bound_var_interval",
    "scale_interval",
    # Transfer function interface
    "TransferFunction",
    "IntervalTF",
    "HybridzTF",
    "set_transfer_function",
    "get_transfer_function",
    "set_transfer_function_mode",
    "dispatch_tf",
    # Analysis
    "analyze",
    "initialize_tf_mode",
    # Export
    "export_to_solver",
    "to_numpy",
    # Solvers
    "Solver",
    "SolverCaps",
    "SolveStatus",
    "GurobiSolver",
    "TorchLPSolver",
    # Specs
    "InputSpec",
    "OutputSpec",
    "InKind",
    "OutKind",
    "VerifyStatus",
    "VerifyResult",
    "seed_from_input_spec",
    "add_input_spec",
    "verify_once",
    # Config
    "BackendConfig",
    "GenerationConfig",
    # BaB
    "BabNode",
    "SubproblemBatch",
    "split_subproblems",
    "BaBConfig",
    "verify_bab",
    "check_violation_at_point",
    "BranchingStrategy",
    "Scheduler",
]
