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
#   verifiable PyTorch models from pre-generated Net JSON files with
#   embedded input/output constraints for automatic verification.
#
# Key Features:
#   - Directory-scan discovery: finds networks from nets/*.json + manifest
#   - Spec-free models: InputSpecLayer/OutputSpecLayer embedded in model
#   - Weight consistency: loads shared weights from JSON (same as ACT Nets)
#   - VerifiableModel: returns models with automatic constraint checking
#   - Strategic test inputs: center / boundary / random placement
#
# Usage:
#   factory = ModelFactory()
#   model   = factory.create_model("mnist_robust_easy", load_weights=True)
#   results = model(input_tensor)
#   x_test  = factory.generate_test_input("mnist_robust_easy", "center")
#
#===---------------------------------------------------------------------===#

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

from act.back_end.core import Net
from act.back_end.serialization.serialization import NetSerializer
from act.pipeline.verification.act2torch import ACTToTorch
from act.util.device_manager import get_default_dtype, get_default_device
from act.util.path_config import get_examples_nets_dir
from act.util.format_utils import rule

logger = logging.getLogger(__name__)


def _load_manifest(manifest_path: Path) -> List[str]:
    """Load network names from manifest JSON file."""
    payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    return list(payload.get('nets', []))


def _discover_net_names(nets_dir: Path, manifest_path: Optional[Path]) -> List[str]:
    """
    Discover network names from manifest and/or directory scan.

    Priority: manifest first, then directory glob.
    Duplicates are removed while preserving order.
    """
    names: List[str] = []
    if manifest_path is None:
        # Try default manifest locations
        for candidate in [nets_dir / '_meta' / 'manifest.json',
                          nets_dir / 'manifest.json']:
            if candidate.exists():
                manifest_path = candidate
                break

    if manifest_path is not None and manifest_path.exists():
        names.extend(str(n) for n in _load_manifest(manifest_path))

    names.extend(p.stem for p in sorted(nets_dir.glob('*.json')))

    ordered: List[str] = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


class ModelFactory:
    """Factory for creating PyTorch models from pre-generated Net JSONs.

    Networks are discovered by scanning ``nets_dir/*.json`` and optional
    ``manifest.json``.  No YAML configuration file is needed -- all metadata
    (input shape, layer count, constraints) is extracted directly from
    the loaded Net objects.
    """

    def __init__(
        self,
        nets_dir: str = get_examples_nets_dir(),
        manifest_path: Optional[str] = None,
    ):
        """
        Initialize factory with nets directory and optional manifest.
        
        Args:
            nets_dir: Directory containing pre-generated ACT Net JSON files
            manifest_path: Optional path to manifest.json listing network names
        """
        self.nets_dir = Path(nets_dir)
        self.manifest_path = Path(manifest_path) if manifest_path else None
        
        # Pre-load all ACT Nets for fast access (avoids repeated file I/O)
        self.net_names = _discover_net_names(self.nets_dir, self.manifest_path)
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
        # Callers that only need VNNLIB/TorchVision models construct the factory
        # before nets are generated; consumers that need nets check list_networks().
        if not self.nets_dir.exists():
            logger.debug(f"Nets dir not found: {self.nets_dir}")
            return
        
        for name in self.net_names:
            net_path = self.nets_dir / f"{name}.json"
            
            if not net_path.exists():
                raise FileNotFoundError(
                    f"ACT Net '{name}' is listed in the manifest but {net_path} does not exist"
                )
            
            with open(net_path, 'r') as f:
                net_dict = json.load(f)
            act_net = NetSerializer.deserialize_net(net_dict)
            self.nets[name] = act_net
            logger.debug(f"Pre-loaded ACT Net '{name}' from {net_path}")
        
        logger.info(f"Pre-loaded {len(self.nets)} ACT Nets from {self.nets_dir}")
    
    def get_act_net(self, name: str) -> Net:
        """
        Get pre-loaded ACT Net by name.
        
        Args:
            name: Network name (stem of .json file in nets/)
            
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
        Create PyTorch model from pre-loaded Net.
        
        Args:
            name: Network name (stem of .json file in nets/)
            load_weights: If True, load weights from corresponding ACT Net JSON file
            
        Returns:
            PyTorch nn.Module ready for inference or training
            
        Raises:
            KeyError: If network name not found
            ValueError: If network architecture is invalid
        """
        if name not in self.nets:
            available = ", ".join(self.nets.keys())
            raise KeyError(f"Network '{name}' not found. Available: {available}")
        
        if not load_weights:
            raise ValueError("ModelFactory requires load_weights=True (Net JSONs are the source of truth).")
        
        act_net = self.get_act_net(name)
        
        # Convert Net to PyTorch model (auto-detects DAG mode)
        converter = ACTToTorch(act_net)
        model = converter.run()
        
        logger.info(f"Created PyTorch model '{name}' with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
    
    def _find_layer(self, net: Net, kind: str) -> Optional[Any]:
        """Find first layer of given kind in network."""
        for layer in getattr(net, 'layers', []):
            if getattr(layer, 'kind', None) == kind:
                return layer
        return None
    
    def _infer_box_bounds(self, params: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """Infer box bounds from lb/ub parameter tensors."""
        lb = params.get('lb')
        ub = params.get('ub')
        if lb is None or ub is None:
            return None
        lb_t = torch.as_tensor(lb)
        ub_t = torch.as_tensor(ub)
        return float(lb_t.min().item()), float(ub_t.max().item())
    
    def generate_test_input(self, name: str, test_case: str = "center") -> torch.Tensor:
        """
        Generate strategic test input considering both INPUT params and INPUT_SPEC constraints.
        
        Args:
            name: Network name (stem of .json file in nets/)
            test_case: One of "center" (safe), "boundary" (risky), "random" (uncertain)
            
        Returns:
            Input tensor strategically placed for verification testing
            
        Test Case Strategy:
        - center: Input at center of constraint region (expected PASS)
        - boundary: Input near boundary of constraints (expected UNCERTAIN/FAIL)
        - random: Random input in constraint region (expected varied results)
        """
        if name not in self.nets:
            raise KeyError(f"Network '{name}' not found")
        
        act_net = self.get_act_net(name)
        input_layer = self._find_layer(act_net, 'INPUT')
        input_spec_layer = self._find_layer(act_net, 'INPUT_SPEC')
        
        if input_layer is None:
            raise ValueError(f"No INPUT layer found in network '{name}'")
        
        # ACT Layer has flat params (no separate meta).
        # Shape, dtype, value_range etc. are all in layer.params.
        input_params = input_layer.params or {}
        shape = input_params.get('shape')
        if shape is None:
            raise ValueError(f"INPUT layer missing 'shape' in network '{name}'")
        
        # Use device_manager's dtype/device for test inputs
        dtype = get_default_dtype()
        device = get_default_device()
        
        geometry = act_net.input_geometry
        if geometry is not None:
            center = geometry.center.to(device=device, dtype=dtype)
            match test_case:
                case "center":
                    tensor = center.clone()
                case "boundary":
                    direction = torch.zeros_like(center)
                    direction[..., 0] = 1.0
                    tensor = center + direction * geometry.radii.to(center)
                case "random":
                    tensor = center + (2 * torch.rand_like(center) - 1) * geometry.radii.to(center)
                case _:
                    raise ValueError(f"Unknown test_case '{test_case}'")
            return geometry.project(tensor)

        # Get INPUT_SPEC constraints if present
        if input_spec_layer is not None:
            spec_params = input_spec_layer.params or {}
            spec_kind = str(spec_params.get('kind'))
            
            if spec_kind == 'BOX':
                lb_val = spec_params.get('lb_val')
                ub_val = spec_params.get('ub_val')
                if lb_val is None or ub_val is None:
                    bounds = self._infer_box_bounds(spec_params)
                    if bounds:
                        lb_val, ub_val = bounds
                if lb_val is None or ub_val is None:
                    lb_val, ub_val = 0.0, 1.0
                
                if test_case == 'center':
                    value = (lb_val + ub_val) / 2.0
                    tensor = torch.full(shape, value, dtype=dtype, device=device)
                elif test_case == 'boundary':
                    value = ub_val - 0.001
                    tensor = torch.full(shape, value, dtype=dtype, device=device)
                elif test_case == 'random':
                    tensor = torch.rand(*shape, dtype=dtype, device=device) * (ub_val - lb_val) + lb_val
                else:
                    raise ValueError(f"Unknown test_case '{test_case}'")
            
            elif spec_kind == 'LINF_BALL':
                center_val = spec_params.get('center_val', 0.5)
                eps = spec_params.get('eps', 0.1)
                
                if test_case == 'center':
                    tensor = torch.full(shape, center_val, dtype=dtype, device=device)
                elif test_case == 'boundary':
                    value = center_val + eps - 0.001
                    tensor = torch.full(shape, value, dtype=dtype, device=device)
                elif test_case == 'random':
                    perturbation = (torch.rand(*shape, dtype=dtype, device=device) - 0.5) * 2.0 * eps
                    tensor = torch.full(shape, center_val, dtype=dtype, device=device) + perturbation
                else:
                    raise ValueError(f"Unknown test_case '{test_case}'")
            
            else:
                # LIN_POLY or unknown: fallback to uniform random in value_range
                value_range = input_params.get('value_range', [0.0, 1.0])
                tensor = torch.rand(*shape, dtype=dtype, device=device) * (value_range[1] - value_range[0]) + value_range[0]
        
        else:
            # No INPUT_SPEC: use uniform random in value_range
            value_range = input_params.get('value_range', [0.0, 1.0])
            tensor = torch.rand(*shape, dtype=dtype, device=device) * (value_range[1] - value_range[0]) + value_range[0]
        
        return tensor
    
    def list_networks(self) -> List[str]:
        """List all available network names."""
        return list(self.nets.keys())
    
    def get_network_info(self, name: str) -> Dict[str, Any]:
        """Get info about a network without creating it."""
        if name not in self.nets:
            raise KeyError(f"Network '{name}' not found")
        
        net = self.nets[name]
        meta = getattr(net, 'meta', {}) or {}
        input_layer = self._find_layer(net, 'INPUT')
        input_shape = None
        if input_layer is not None:
            input_shape = (input_layer.params or {}).get('shape')
        
        num_layers = len([l for l in net.layers if l.kind not in ['INPUT', 'INPUT_SPEC', 'ASSERT']])
        
        return {
            'name': name,
            'description': meta.get('description', name),
            'architecture_type': meta.get('architecture_type', 'unknown'),
            'input_shape': input_shape or 'unknown',
            'num_layers': num_layers,
            'metadata': meta,
        }


def main():
    """Test model factory with all example networks and verify spec-free verification."""
    logging.basicConfig(level=logging.INFO)
    
    factory = ModelFactory()
    
    print(rule())
    print("PyTorch Model Factory - Spec-Free Verification Testing")
    print(rule())
    
    failures: List[str] = []
    total_tests = 0
    passed_tests = 0
    
    network_names = factory.list_networks()
    if not network_names:
        raise RuntimeError(
            f"No ACT Nets found in {factory.nets_dir}; run `python -m act.back_end --generate` first"
        )
    
    for name in network_names:
        print(f"\n{rule()}")
        print(f"Network: {name}")
        print(rule())
        
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
                print(rule(80, "-"))
                
                try:
                    # Generate strategic input
                    input_tensor = factory.generate_test_input(name, test_case)
                    print(f"  Input shape: {list(input_tensor.shape)}")
                    print(f"  Input range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
                    
                    # Run model with automatic constraint checking
                    results = model(input_tensor)
                    
                    # VerifiableModel emits a dict with verification info;
                    # a raw nn.Module emits a bare output tensor.
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
                        # Constraint outcomes are properties of the (random-weight)
                        # network on this input, not conversion errors.
                        if input_satisfied and output_satisfied:
                            passed_tests += 1
                            print(f"  ✅ Both constraints satisfied")
                        elif not input_satisfied:
                            print(f"  ℹ️  Input outside spec region (property not applicable)")
                        else:
                            print(f"  ℹ️  Output property violated on this input (counterexample)")
                    
                    else:
                        # Legacy nn.Module (no verification)
                        output = results
                        print(f"  ⚠️  Legacy model (no constraint checking)")
                        print(f"  Output shape: {list(output.shape)}")
                        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
                        total_tests += 1
                
                except Exception as e:
                    print(f"  ❌ Test case '{test_case}' failed: {e}")
                    import traceback
                    traceback.print_exc()
                    failures.append(f"{name}/{test_case}: {e}")
            
        except Exception as e:
            print(f"\n❌ Failed to create/test model '{name}': {e}")
            import traceback
            traceback.print_exc()
            failures.append(f"{name}: {e}")
    
    # Print summary
    print("\n" + rule())
    print(f"📊 Verification Test Summary:")
    print(f"   Total tests: {total_tests}")
    print(f"   ✅ Both constraints satisfied: {passed_tests}")
    print(f"   ℹ️  Input outside spec / property violated: {total_tests - passed_tests}")
    if total_tests == 0:
        raise RuntimeError("No ACT→PyTorch test cases were executed")
    print(rule())
    
    if failures:
        raise RuntimeError(
            f"{len(failures)} ACT→PyTorch model/test-case error(s):\n  " + "\n  ".join(failures)
        )
    print("✅ All models created and tested successfully")
    print(rule())


if __name__ == "__main__":
    main()
