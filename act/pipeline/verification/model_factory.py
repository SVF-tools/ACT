#!/usr/bin/env python3
#===- act/pipeline/model_factory.py - PyTorch Model Factory ------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   PyTorch model factory for spec-free verification testing. Creates
#   verifiable PyTorch models from examples_config.yaml specifications
#   with embedded input/output constraints for automatic verification.
#
# Key Features:
#   - Spec-free models: InputSpecLayer/OutputSpecLayer embedded in model
#   - Weight consistency: Loads shared weights from JSON (same as ACT Nets)
#   - VerifiableModel: Returns models with automatic constraint checking
#   - Bidirectional testing: Validates PyTorch→ACT→PyTorch round trips
#   - Comprehensive coverage: 12 test networks across 4 verification scenarios
#
# Architecture:
#   Each model is wrapped with verification layers:
#   1. InputLayer: Declares input shape/dtype/device
#   2. InputSpecLayer: Input constraints (BOX, L_INF, LIN_POLY)
#   3. Model layers: nn.Linear, nn.Conv2d, nn.ReLU, etc.
#   4. OutputSpecLayer: Output constraints (SAFETY, TOP1_ROBUST, etc.)
#
# Note: Preprocessing (normalization, flatten, etc.) should be handled by
#   data loader (e.g., torchvision.transforms) before wrapping the model.
#
# Test Scenarios (examples_config.yaml):
#   - mnist_robust_*: Classification robustness (ε-ball perturbations)
#   - cifar_margin_*: Classification margin constraints
#   - control_*: Control system safety (state bounds)
#   - reachability_*: Reachability analysis (target regions)
#
# Weight Consistency:
#   Models and ACT Nets load identical weights from JSON files, ensuring:
#   - PyTorch inference ≡ ACT forward bounds (numerically identical)
#   - Round-trip conversion preserves all parameters
#   - Verification results match between PyTorch and ACT
#
# Usage:
#   factory = ModelFactory()
#   model = factory.create_model("mnist_robust_easy", load_weights=True)
#   
#   # Model is VerifiableModel with constraint checking
#   results = model(input_tensor)
#   print(f"Output: {results['output']}")
#   print(f"Constraints satisfied: {results['output_satisfied']}")
#
# Testing:
#   python act/pipeline/model_factory.py  # Tests all 12 networks
#
#===---------------------------------------------------------------------===#

import yaml
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

from act.back_end.core import Net, Layer
from act.back_end.serialization.serialization import NetSerializer
from act.pipeline.verification.act2torch import ACTToTorch
from act.util.device_manager import get_default_dtype, get_default_device

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating PyTorch models from examples_config.yaml."""
    
    def __init__(self, 
                 config_path: str = "act/back_end/examples/examples_config.yaml",
                 nets_dir: str = "act/back_end/examples/nets"):
        """
        Initialize factory with configuration file and pre-load all ACT Nets.
        
        Args:
            config_path: Path to examples_config.yaml
            nets_dir: Directory containing pre-generated ACT Net JSON files
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.nets_dir = Path(nets_dir)
        
        # Pre-load all ACT Nets for fast access (avoids repeated file I/O)
        self.nets: Dict[str, Net] = {}
        self._load_all_nets()
    
    def _load_all_nets(self) -> None:
        """
        Pre-load all ACT Nets from JSON files at initialization.
        
        This eager loading strategy:
        - Avoids repeated file I/O during model creation
        - Validates all nets exist and are valid at init time
        - Enables O(1) lookup via get_act_net()
        - Costs ~10-20MB memory for typical test suites
        """
        for name in self.config['networks'].keys():
            net_path = self.nets_dir / f"{name}.json"
            
            if not net_path.exists():
                logger.warning(f"ACT Net file not found: {net_path}. Skipping '{name}'.")
                continue
            
            try:
                with open(net_path, 'r') as f:
                    net_dict = json.load(f)
                act_net, _ = NetSerializer.deserialize_net(net_dict)
                self.nets[name] = act_net
                logger.debug(f"Pre-loaded ACT Net '{name}' from {net_path}")
            except Exception as e:
                logger.error(f"Failed to load ACT Net '{name}' from {net_path}: {e}")
                continue
        
        logger.info(f"Pre-loaded {len(self.nets)} ACT Nets from {self.nets_dir}")
    
    def get_act_net(self, name: str) -> Net:
        """
        Get pre-loaded ACT Net by name.
        
        Args:
            name: Network name from examples_config.yaml
            
        Returns:
            Pre-loaded ACT Net
            
        Raises:
            KeyError: If network name not found or failed to load
        """
        if name not in self.nets:
            available = ", ".join(self.nets.keys())
            raise KeyError(f"ACT Net '{name}' not available. Available: {available}")
        
        return self.nets[name]
    
    def create_model(self, name: str, load_weights: bool = True) -> nn.Module:
        """
        Create PyTorch model from configuration.
        
        Args:
            name: Network name from examples_config.yaml
            load_weights: If True, load weights from corresponding ACT Net JSON file
            
        Returns:
            PyTorch nn.Module ready for inference or training
            
        Raises:
            KeyError: If network name not found in config
            ValueError: If network architecture is invalid
        """
        if name not in self.config['networks']:
            available = ", ".join(self.config['networks'].keys())
            raise KeyError(f"Network '{name}' not found. Available: {available}")
        
        spec = self.config['networks'][name]
        
        # Get pre-loaded ACT Net if weights should be transferred
        act_net = None
        if load_weights:
            act_net = self.get_act_net(name)  # O(1) lookup, no file I/O
            logger.debug(f"Using pre-loaded ACT Net '{name}'")
        
        # Build PyTorch module using ACTToTorch converter
        if act_net is not None:
            converter = ACTToTorch(act_net)
            model = converter.run()
        else:
            # Fallback: build from config with random weights
            raise ValueError(f"Cannot create model without ACT Net. Set load_weights=True or ensure {name}.json exists.")
        
        logger.info(f"Created PyTorch model '{name}' with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
    
    def generate_test_input(self, name: str, test_case: str = "center") -> torch.Tensor:
        """
        Generate strategic test input considering both INPUT metadata and INPUT_SPEC constraints.
        
        Args:
            name: Network name from examples_config.yaml
            test_case: One of "center" (safe), "boundary" (risky), "random" (uncertain)
            
        Returns:
            Input tensor strategically placed for verification testing
            
        Test Case Strategy:
        - center: Input at center of constraint region (expected PASS)
        - boundary: Input near boundary of constraints (expected UNCERTAIN/FAIL)
        - random: Random input in constraint region (expected varied results)
        """
        if name not in self.config['networks']:
            raise KeyError(f"Network '{name}' not found")
        
        spec = self.config['networks'][name]
        layers_spec = spec['layers']
        
        # Find INPUT and INPUT_SPEC layers
        input_layer = None
        input_spec_layer = None
        for layer_spec in layers_spec:
            if layer_spec['kind'] == 'INPUT':
                input_layer = layer_spec
            elif layer_spec['kind'] == 'INPUT_SPEC':
                input_spec_layer = layer_spec
        
        if input_layer is None:
            raise ValueError(f"No INPUT layer found in network '{name}'")
        
        # Get INPUT metadata
        input_meta = input_layer.get('meta', {})
        shape = input_meta.get('shape')
        if shape is None:
            raise ValueError(f"INPUT layer missing 'shape' in network '{name}'")
        
        # Use device_manager's dtype/device for test inputs
        # This ensures test inputs match the model's configuration
        dtype = get_default_dtype()
        device = get_default_device()
        
        # Get INPUT_SPEC constraints if present
        if input_spec_layer is not None:
            spec_meta = input_spec_layer.get('meta', {})
            spec_kind = spec_meta.get('kind')
            
            if spec_kind == 'BOX':
                lb_val = spec_meta.get('lb_val', 0.0)
                ub_val = spec_meta.get('ub_val', 1.0)
                
                if test_case == 'center':
                    # Center of box: (lb + ub) / 2
                    value = (lb_val + ub_val) / 2.0
                    tensor = torch.full(shape, value, dtype=dtype)
                elif test_case == 'boundary':
                    # Near upper boundary: ub - small_epsilon
                    value = ub_val - 0.001
                    tensor = torch.full(shape, value, dtype=dtype)
                elif test_case == 'random':
                    # Random within bounds
                    tensor = torch.rand(*shape, dtype=dtype) * (ub_val - lb_val) + lb_val
                else:
                    raise ValueError(f"Unknown test_case '{test_case}'")
            
            elif spec_kind == 'LINF_BALL':
                center_val = spec_meta.get('center_val', 0.5)
                eps = spec_meta.get('eps', 0.1)
                
                if test_case == 'center':
                    # At center of L∞ ball
                    tensor = torch.full(shape, center_val, dtype=dtype)
                elif test_case == 'boundary':
                    # Near boundary: center + eps - small_delta
                    value = center_val + eps - 0.001
                    tensor = torch.full(shape, value, dtype=dtype)
                elif test_case == 'random':
                    # Random within L∞ ball
                    perturbation = (torch.rand(*shape, dtype=dtype) - 0.5) * 2.0 * eps
                    tensor = torch.full(shape, center_val, dtype=dtype) + perturbation
                else:
                    raise ValueError(f"Unknown test_case '{test_case}'")
            
            else:
                # LIN_POLY or unknown: fallback to uniform random in value_range
                value_range = input_meta.get('value_range', [0.0, 1.0])
                tensor = torch.rand(*shape, dtype=dtype) * (value_range[1] - value_range[0]) + value_range[0]
        
        else:
            # No INPUT_SPEC: use uniform random in value_range
            value_range = input_meta.get('value_range', [0.0, 1.0])
            tensor = torch.rand(*shape, dtype=dtype) * (value_range[1] - value_range[0]) + value_range[0]
        
        return tensor
    
    def list_networks(self) -> List[str]:
        """List all available network names."""
        return list(self.config['networks'].keys())
    
    def get_network_info(self, name: str) -> Dict[str, Any]:
        """Get metadata about a network without creating it."""
        if name not in self.config['networks']:
            raise KeyError(f"Network '{name}' not found")
        
        spec = self.config['networks'][name]
        
        return {
            'name': name,
            'description': spec.get('description', 'No description'),
            'architecture_type': spec.get('architecture_type', 'unknown'),
            'input_shape': spec.get('input_shape', 'unknown'),
            'num_layers': len([l for l in spec['layers'] if l['kind'] not in ['INPUT', 'INPUT_SPEC', 'ASSERT']]),
            'metadata': spec.get('metadata', {})
        }


def main():
    """Test model factory with all example networks and verify spec-free verification."""
    logging.basicConfig(level=logging.INFO)
    
    factory = ModelFactory()
    
    print("=" * 80)
    print("PyTorch Model Factory - Spec-Free Verification Testing")
    print("=" * 80)
    
    all_passed = True
    total_tests = 0
    passed_tests = 0
    
    for name in factory.list_networks():
        print(f"\n{'=' * 80}")
        print(f"Network: {name}")
        print("=" * 80)
        
        # Get network info
        info = factory.get_network_info(name)
        print(f"Description: {info['description']}")
        print(f"Architecture: {info['architecture_type']}")
        print(f"Input shape: {info['input_shape']}")
        
        # Create model with VerifiableModel wrapper
        try:
            model = factory.create_model(name, load_weights=True)
            print(f"\n✅ Created VerifiableModel model")
            
            # Test with 3 strategic test cases
            test_cases = ['center', 'boundary', 'random']
            
            for test_case in test_cases:
                print(f"\n📊 Test Case: {test_case}")
                print("-" * 80)
                
                try:
                    # Generate strategic input
                    input_tensor = factory.generate_test_input(name, test_case)
                    print(f"  Input shape: {list(input_tensor.shape)}")
                    print(f"  Input range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
                    
                    # Run model with automatic constraint checking
                    results = model(input_tensor)
                    
                    # Check if results is a dict (VerifiableModel) or tensor (legacy)
                    if isinstance(results, dict):
                        # VerifiableModel returns dict with verification info
                        output = results['output']
                        input_satisfied = results['input_satisfied']
                        input_explanation = results['input_explanation']
                        output_satisfied = results['output_satisfied']
                        output_explanation = results['output_explanation']
                        
                        print(f"\n  📥 {input_explanation}")
                        print(f"  📤 {output_explanation}")
                        print(f"  Output shape: {list(output.shape)}")
                        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
                        
                        # Track test success
                        total_tests += 1
                        if input_satisfied and output_satisfied:
                            passed_tests += 1
                            print(f"  ✅ Test PASSED (both constraints satisfied)")
                        elif not input_satisfied:
                            print(f"  ⚠️  Test UNCERTAIN (input constraint violated)")
                        else:
                            print(f"  ❌ Test FAILED (output constraint violated)")
                    
                    else:
                        # Legacy nn.Sequential (no verification)
                        output = results
                        print(f"  ⚠️  Legacy model (no constraint checking)")
                        print(f"  Output shape: {list(output.shape)}")
                        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
                        total_tests += 1
                
                except Exception as e:
                    print(f"  ❌ Test case '{test_case}' failed: {e}")
                    import traceback
                    traceback.print_exc()
                    all_passed = False
            
        except Exception as e:
            print(f"\n❌ Failed to create/test model '{name}': {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"📊 Verification Test Summary:")
    print(f"   Total tests: {total_tests}")
    print(f"   ✅ Passed: {passed_tests}")
    print(f"   ⚠️  Uncertain/Failed: {total_tests - passed_tests}")
    if total_tests > 0:
        success_rate = (passed_tests / total_tests) * 100
        print(f"   Success rate: {success_rate:.1f}%")
    print("=" * 80)
    
    if all_passed:
        print("✅ All models created and tested successfully")
    else:
        print("⚠️  Some models had issues - see details above")
    print("=" * 80)


if __name__ == "__main__":
    main()
