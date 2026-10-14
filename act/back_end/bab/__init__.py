# ===- act/back_end/bab/__init__.py - BaB Package -------------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#

from act.config.config import BaBConfig
from act.back_end.bab.bab import verify_bab, verify_bab_batched
from act.back_end.bab.node import SubproblemBatch
from act.back_end.bab.branching.branching import BranchingStrategy
from act.back_end.bab.branching.bounding import BoundingStrategy

__all__ = [
    "BaBConfig",
    "verify_bab",
    "verify_bab_batched",
    "SubproblemBatch",
    "BranchingStrategy",
    "BoundingStrategy",
]
