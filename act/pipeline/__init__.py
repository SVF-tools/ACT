#===- act/pipeline/__init__.py - ACT Pipeline Module -------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   ACT Pipeline module for PyTorch model generation and testing utilities.
#   Provides tools for converting between PyTorch models and ACT Nets,
#   verifier validation, and utility functions.
#
#===---------------------------------------------------------------------===#

"""ACT Pipeline Module - Model Generation and Testing Utilities.

This module provides utilities for PyTorch model generation, ACT conversion,
verifier validation, and performance analysis.

Key Components:
    - ModelFactory: Create PyTorch models from YAML configurations
    - TorchToACT: Convert PyTorch models to ACT representation
    - VerifierValidator: Validate verifier correctness with concrete tests

All verification utilities are located in the verification/ submodule.

Example:
    # Create PyTorch model from pre-generated nets/*.json
    from act.pipeline import ModelFactory, TorchToACT
    
    factory = ModelFactory()
    model = factory.create_model("mnist_mlp_small", load_weights=True)
    
    # Convert to ACT format
    converter = TorchToACT()
    act_net = converter.convert(model, input_shape=(1, 784))
"""

# Core imports
from act.pipeline.verification.model_factory import ModelFactory
from act.pipeline.verification.torch2act import TorchToACT

__all__ = [
    # Core model factory and conversion
    'ModelFactory',
    'TorchToACT',
]
