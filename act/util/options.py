#===- util.options.py ----ACT Native Parameters & Option Definitions---------#
#
#                 ACT: Abstract Constraints Transformer
#
# Copyright (C) <2025->  ACT Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Authors: ACT Team   
#
# Purpose:
#   Global performance and debugging options for ACT.
#
#===----------------------------------------------------------------------===#
    
import os
from act.util.path_config import get_pipeline_log_dir


class PerformanceOptions:
    """Global performance and debugging options.
    
    Attributes:
        debug_tf: Enable debug logging in transfer functions (default: True)
        validate_constraints: Enable constraint validation (default: True)
        debug_output_file: Path to debug log file (default: "act/pipeline/log/act_debug_tf.log")
        debug_tf_max_constraints: Maximum number of constraints to log per layer (default: 50)
    
    """
    debug_tf: bool = True
    validate_constraints: bool = True
    debug_output_file: str = os.path.join(get_pipeline_log_dir(), "act_debug_tf.log")
    debug_tf_max_constraints: int = 50
    
