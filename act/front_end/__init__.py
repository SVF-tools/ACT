#===- act/front_end/__init__.py - ACT Frontend Preprocessing Module ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   ACT Front-End module providing unified specification system and utilities
#   for DNN verification. Integrates with spec creators for data/model/spec
#   loading and synthesis.
#
#===---------------------------------------------------------------------===#

"""
ACT Front-End Module

Key Features:
- Unified specification system (InputSpec/OutputSpec)
- Spec creators (TorchVision, VNNLib)
- Device-aware tensor management
- Verifiable model wrappers

Usage:
    >>> from act.front_end import InputSpec, OutputSpec, InKind, OutKind
    >>> 
    >>> # Create specifications
    >>> input_spec = InputSpec(kind=InKind.LINF_BALL, center=data, eps=0.1)
    >>> output_spec = OutputSpec(kind=OutKind.SAFETY, ...)
"""

# Core specification system
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind

# Unified spec loader
from act.front_end.spec_loader import SpecLoader

# Device management - import only when needed to avoid triggering argparse at import time
# from act.util.device_manager import get_default_device, get_default_dtype, get_current_settings

__all__ = [
    # Specifications
    'InputSpec', 'OutputSpec', 'InKind', 'OutKind',
    
    # Unified loader
    'SpecLoader',
    
    # Device management (available via act.util.device_manager)
    # 'get_default_device', 'get_default_dtype', 'get_current_settings',
]
