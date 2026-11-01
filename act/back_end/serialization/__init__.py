#===- act/back_end/serialization/__init__.py - Serialization Package ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Serialization package for ACT neural networks providing JSON serialization,
#   network analysis, validation, and utility functions for model persistence.
#
#===---------------------------------------------------------------------====#

# Core serialization functionality
from .serialization import (
    TensorEncoder,
    LayerSerializer, 
    NetSerializer,
    save_net_to_file,
    load_net_from_file,
    save_net_to_string,
    load_net_from_string,
)

__all__ = [
    # Core serialization
    'TensorEncoder',
    'LayerSerializer',
    'NetSerializer', 
    'save_net_to_file',
    'load_net_from_file',
    'save_net_to_string',
    'load_net_from_string',
]
