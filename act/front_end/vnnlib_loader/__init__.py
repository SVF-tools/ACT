#===- act/front_end/vnnlib/__init__.py - VNNLIB Module Exports --------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Module initialization for VNNLIB benchmark support.
#   Exports key functions and classes for VNNLIB-based verification.
#
#===---------------------------------------------------------------------===#

"""
VNNLIB Benchmark Support for ACT.

This module provides tools for working with VNNLIB benchmarks from VNN-COMP,
including ONNX model conversion, VNNLIB specification parsing, and spec generation.

Main Components:
- VNNLibSpecCreator: Create InputSpec/OutputSpec from VNNLIB benchmarks
- Data/Model Loader: Download and load VNN-COMP benchmarks
- ONNX Converter: Convert ONNX models to PyTorch
- VNNLIB Parser: Parse VNNLIB SMT-LIB format files

Example Usage:
    >>> from act.front_end.vnnlib_loader import (
    ...     VNNLibSpecCreator,
    ...     download_vnnlib_category,
    ...     list_downloaded_pairs
    ... )
    >>> 
    >>> # Download a benchmark category
    >>> download_vnnlib_category("mnist_fc")
    >>> 
    >>> # Create specs for all downloaded instances
    >>> creator = VNNLibSpecCreator()
    >>> results = creator.create_specs_for_data_model_pairs()
"""

from __future__ import annotations

# Spec creator (main interface)
from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator

# Data/model loading
from act.front_end.vnnlib_loader.data_model_loader import (
    download_vnnlib_category,
    list_downloaded_pairs,
    load_vnnlib_pair,
    list_available_categories,
    list_local_categories,
    get_category_info
)

# ONNX conversion utilities
from act.front_end.vnnlib_loader.onnx_converter import (
    convert_onnx_to_pytorch,
    get_onnx_input_shape,
    get_onnx_output_shape,
    get_onnx_metadata,
    validate_onnx_file,
    ONNXConversionError
)

# VNNLIB parsing utilities
from act.front_end.vnnlib_loader.vnnlib_parser import (
    parse_vnnlib_to_tensors,
    parse_vnnlib_queries,
    validate_vnnlib_file,
    VNNLibParseError
)


__all__ = [
    # Main spec creator
    'VNNLibSpecCreator',
    
    # Data/model loading
    'download_vnnlib_category',
    'list_downloaded_pairs',
    'load_vnnlib_pair',
    'list_available_categories',
    'list_local_categories',
    'get_category_info',
    
    # ONNX conversion
    'convert_onnx_to_pytorch',
    'get_onnx_input_shape',
    'get_onnx_output_shape',
    'get_onnx_metadata',
    'validate_onnx_file',
    'ONNXConversionError',
    
    # VNNLIB parsing
    'parse_vnnlib_to_tensors',
    'parse_vnnlib_queries',
    'validate_vnnlib_file',
    'VNNLibParseError'
]


# Version info
__version__ = '0.1.0'
__author__ = 'ACT Team'
