# ===- act/back_end/bab/__init__.py - BaB Package -------------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#

from act.back_end.config import BaBConfig
from act.back_end.bab.bab import verify_bab, check_violation_at_point
from act.back_end.bab.node import BabNode, SubproblemBatch, split_subproblems
from act.back_end.bab.branching.branching import BranchingStrategy
from act.back_end.bab.branching.bounding import BoundingStrategy

__all__ = [
    "BaBConfig",
    "verify_bab",
    "BabNode",
    "SubproblemBatch",
    "split_subproblems",
    "check_violation_at_point",
    "BranchingStrategy",
    "BoundingStrategy",
]
