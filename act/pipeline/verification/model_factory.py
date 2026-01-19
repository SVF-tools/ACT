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
#   verifiable PyTorch models directly from ACT Net JSONs.
#
#===---------------------------------------------------------------------===#

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from act.back_end.core import Net
from act.back_end.serialization.serialization import NetSerializer
from act.pipeline.verification.act2torch import ACTToTorch
from act.util.device_manager import get_default_dtype, get_default_device

logger = logging.getLogger(__name__)

DEFAULT_NETS_DIR = "act/back_end/examples/nets"


def _load_manifest(manifest_path: Path) -> List[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(payload.get("nets", []))


def _discover_net_names(nets_dir: Path, manifest_path: Optional[Path]) -> List[str]:
    names: List[str] = []
    if manifest_path is None:
        default_manifest = nets_dir / "manifest.json"
        if default_manifest.exists():
            manifest_path = default_manifest

    if manifest_path is not None and manifest_path.exists():
        try:
            names.extend(str(n) for n in _load_manifest(manifest_path))
        except Exception as e:
            logger.warning("Failed to read manifest %s: %s", manifest_path, e)

    names.extend(p.stem for p in nets_dir.glob("*.json"))

    ordered: List[str] = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


class ModelFactory:
    """Factory for creating PyTorch models from ACT Net JSONs."""

    def __init__(
        self,
        nets_dir: str = DEFAULT_NETS_DIR,
        manifest_path: Optional[str] = None,
    ):
        self.nets_dir = Path(nets_dir)
        self.manifest_path = Path(manifest_path) if manifest_path else None

        self.net_names = _discover_net_names(self.nets_dir, self.manifest_path)
        self.nets: Dict[str, Net] = {}
        self._load_all_nets()

    def _load_all_nets(self) -> None:
        if not self.nets_dir.exists():
            logger.warning("Nets dir not found: %s", self.nets_dir)
            return

        for name in self.net_names:
            net_path = self.nets_dir / f"{name}.json"
            if not net_path.exists():
                logger.warning("ACT Net file not found: %s. Skipping '%s'.", net_path, name)
                continue
            try:
                with open(net_path, "r") as f:
                    net_dict = json.load(f)
                act_net, _ = NetSerializer.deserialize_net(net_dict)
                self.nets[name] = act_net
                logger.debug("Pre-loaded ACT Net '%s' from %s", name, net_path)
            except Exception as e:
                logger.error("Failed to load ACT Net '%s' from %s: %s", name, net_path, e)
                continue

        logger.info("Pre-loaded %d ACT Nets from %s", len(self.nets), self.nets_dir)

    def get_act_net(self, name: str) -> Net:
        if name not in self.nets:
            available = ", ".join(self.nets.keys())
            raise KeyError(f"ACT Net '{name}' not available. Available: {available}")
        return self.nets[name]

    def create_model(self, name: str, load_weights: bool = True) -> nn.Module:
        if name not in self.nets:
            available = ", ".join(self.nets.keys())
            raise KeyError(f"Network '{name}' not found. Available: {available}")

        if not load_weights:
            raise ValueError("ModelFactory requires load_weights=True (ACT Net JSONs are the source of truth).")

        act_net = self.get_act_net(name)
        converter = ACTToTorch(act_net)
        model = converter.run()

        logger.info(
            "Created PyTorch model '%s' with %d parameters",
            name,
            sum(p.numel() for p in model.parameters()),
        )
        return model

    def _find_layer(self, net: Net, kind: str) -> Optional[Any]:
        for layer in getattr(net, "layers", []):
            if getattr(layer, "kind", None) == kind:
                return layer
        return None

    def _infer_box_bounds(self, params: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        lb = params.get("lb")
        ub = params.get("ub")
        if lb is None or ub is None:
            return None
        lb_t = torch.as_tensor(lb)
        ub_t = torch.as_tensor(ub)
        return float(lb_t.min().item()), float(ub_t.max().item())

    def generate_test_input(self, name: str, test_case: str = "center") -> torch.Tensor:
        if name not in self.nets:
            raise KeyError(f"Network '{name}' not found")

        act_net = self.get_act_net(name)
        input_layer = self._find_layer(act_net, "INPUT")
        input_spec_layer = self._find_layer(act_net, "INPUT_SPEC")

        if input_layer is None:
            raise ValueError(f"No INPUT layer found in network '{name}'")

        input_meta = input_layer.meta or {}
        shape = input_meta.get("shape")
        if shape is None:
            raise ValueError(f"INPUT layer missing 'shape' in network '{name}'")

        dtype = get_default_dtype()
        device = get_default_device()

        if input_spec_layer is not None:
            spec_meta = input_spec_layer.meta or {}
            spec_kind = str(spec_meta.get("kind"))

            if spec_kind == "BOX":
                lb_val = spec_meta.get("lb_val")
                ub_val = spec_meta.get("ub_val")
                if lb_val is None or ub_val is None:
                    bounds = self._infer_box_bounds(input_spec_layer.params or {})
                    if bounds:
                        lb_val, ub_val = bounds
                if lb_val is None or ub_val is None:
                    lb_val, ub_val = 0.0, 1.0

                if test_case == "center":
                    value = (lb_val + ub_val) / 2.0
                    tensor = torch.full(shape, value, dtype=dtype, device=device)
                elif test_case == "boundary":
                    value = ub_val - 0.001
                    tensor = torch.full(shape, value, dtype=dtype, device=device)
                elif test_case == "random":
                    tensor = torch.rand(*shape, dtype=dtype, device=device) * (ub_val - lb_val) + lb_val
                else:
                    raise ValueError(f"Unknown test_case '{test_case}'")

            elif spec_kind == "LINF_BALL":
                center_val = spec_meta.get("center_val", 0.5)
                eps = spec_meta.get("eps", 0.1)

                if test_case == "center":
                    tensor = torch.full(shape, center_val, dtype=dtype, device=device)
                elif test_case == "boundary":
                    value = center_val + eps - 0.001
                    tensor = torch.full(shape, value, dtype=dtype, device=device)
                elif test_case == "random":
                    perturbation = (torch.rand(*shape, dtype=dtype, device=device) - 0.5) * 2.0 * eps
                    tensor = torch.full(shape, center_val, dtype=dtype, device=device) + perturbation
                else:
                    raise ValueError(f"Unknown test_case '{test_case}'")

            else:
                value_range = input_meta.get("value_range", [0.0, 1.0])
                tensor = torch.rand(*shape, dtype=dtype, device=device) * (value_range[1] - value_range[0]) + value_range[0]
        else:
            value_range = input_meta.get("value_range", [0.0, 1.0])
            tensor = torch.rand(*shape, dtype=dtype, device=device) * (value_range[1] - value_range[0]) + value_range[0]

        return tensor

    def list_networks(self) -> List[str]:
        return list(self.nets.keys())

    def get_network_info(self, name: str) -> Dict[str, Any]:
        if name not in self.nets:
            raise KeyError(f"Network '{name}' not found")

        net = self.nets[name]
        meta = getattr(net, "meta", {}) or {}
        input_layer = self._find_layer(net, "INPUT")
        input_shape = None
        if input_layer is not None:
            input_shape = (input_layer.meta or {}).get("shape")

        num_layers = len([l for l in net.layers if l.kind not in ["INPUT", "INPUT_SPEC", "ASSERT"]])

        return {
            "name": name,
            "description": meta.get("description", "No description"),
            "architecture_type": meta.get("architecture_type", "unknown"),
            "input_shape": input_shape or "unknown",
            "num_layers": num_layers,
            "metadata": meta,
        }


def main():
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

        info = factory.get_network_info(name)
        print(f"Description: {info['description']}")
        print(f"Architecture: {info['architecture_type']}")
        print(f"Input shape: {info['input_shape']}")

        try:
            model = factory.create_model(name, load_weights=True)
            print("\n✅ Created VerifiableModel model")

            test_cases = ["center", "boundary", "random"]

            for test_case in test_cases:
                print(f"\n📊 Test Case: {test_case}")
                print("-" * 80)

                try:
                    input_tensor = factory.generate_test_input(name, test_case)
                    print(f"  Input shape: {list(input_tensor.shape)}")
                    print(f"  Input range: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")

                    results = model(input_tensor)

                    if isinstance(results, dict):
                        output = results["output"]
                        input_satisfied = results["input_satisfied"]
                        input_explanation = results["input_explanation"]
                        output_satisfied = results["output_satisfied"]
                        output_explanation = results["output_explanation"]

                        print(f"\n  📥 {input_explanation}")
                        print(f"  📤 {output_explanation}")
                        print(f"  Output shape: {list(output.shape)}")
                        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

                        total_tests += 1
                        if input_satisfied and output_satisfied:
                            passed_tests += 1
                            print("  ✅ Test PASSED (both constraints satisfied)")
                        elif not input_satisfied:
                            print("  ⚠️  Test UNCERTAIN (input constraint violated)")
                        else:
                            print("  ❌ Test FAILED (output constraint violated)")
                    else:
                        output = results
                        print("  ⚠️  Legacy model (no constraint checking)")
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

    print("\n" + "=" * 80)
    print("📊 Verification Test Summary:")
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
