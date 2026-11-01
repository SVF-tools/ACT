#===- act/back_end/dual_tf/__init__.py - Dual Transfer Functions --------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Dual transfer functions module for Lagrangian dual bound (wong & kolter-style) computation.
#   Implements backward pass for certified bound computation.
#   - Precision driven: computes tight bounds on the dual objective by backward Lagrangian method 
#   - Adaptive Optimization: computes the bound via dual variables which can be on-demand optimized with gradient-based methods  
#   - Spurious counterexample: greedy spurious counterexample generation via linear boundary 
#
#===---------------------------------------------------------------------===#

"""Dual transfer functions: certified bounds via backward Lagrangian relaxation.

Entry point: ``DualSolver.evaluate_spec`` (in ``act.back_end.solver.solver_dual``).
"""

from .dual_tf import DualTF
from .tf_forward import compute_forward_bounds
