#===- act/front_end/vnnlib/__init__.py - VNNLIB Package ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Package marker for VNNLIB benchmark support; import from the submodules.
#
#===---------------------------------------------------------------------===#

"""
VNNLIB Benchmark Support for ACT.

This module provides tools for working with VNNLIB benchmarks from VNN-COMP,
including ONNX model conversion, VNNLIB specification parsing, and spec generation.

Submodules:
- create_specs: VNNLibSpecCreator builds InputSpec/OutputSpec from VNNLIB benchmarks
- data_model_loader: Download and load VNN-COMP benchmarks
- onnx_converter: Convert ONNX models to PyTorch
- vnnlib_parser: Parse VNNLIB SMT-LIB format files
"""
