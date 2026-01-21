#===- act/back_end/net_factory/factory.py - Simplified NetFactory ---------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   NetFactory orchestration and I/O.
#   Core responsibilities:
#   - Load YAML config and manage output paths
#   - Orchestrate sampling, building, and serialization
#   - Generate weight tensors and handle layer variables
#   - Write optional manifest for downstream tools
#
#   Flow:
#   config -> ConfigSampler -> layer_builder -> Net -> JSON
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import json
import random
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from act.back_end.core import Layer, Net
from act.back_end.serialization.serialization import NetSerializer
from act.front_end.specs import InKind, OutKind
from act.util.device_manager import get_default_dtype
from act.util.path_config import get_examples_gen_config_path

from .layer_builder import build_cnn_layers, build_mlp_layers
from .sampler import ConfigSampler


# ============================================================================
# Internal Utility Functions
# ============================================================================


def _stable_u32_from_bytes(data: bytes) -> int:
    """Extract stable u32 from bytes."""
    return int.from_bytes(data[:4], byteorder="little", signed=False)


def _derive_seed(base_seed: int, idx: int, instance_id: str) -> int:
    """Derive deterministic seed from base_seed, index, and instance_id."""
    import hashlib
    payload = f"{base_seed}|{idx}|{instance_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return _stable_u32_from_bytes(digest)


def _choose(rng: random.Random, items: List[Any], *, name: str) -> Any:
    """Randomly choose from items with error handling."""
    if not items:
        raise ValueError(f"Config.{name} must be non-empty")
    return rng.choice(list(items))


class NetFactory:
    """Simplified generator-driven factory for ACT Nets."""

    def __init__(
        self,
        gen_config_path: str = get_examples_gen_config_path(),
        *,
        output_dir: Optional[str] = None,
        base_seed: Optional[int] = None,
        num_instances: Optional[int] = None,
        name_prefix: Optional[str] = None,
        write_manifest: Optional[bool] = None,
        manifest_path: Optional[str] = None,
    ):
        self.config_path = str(gen_config_path)
        self.config = self._load_config(self.config_path)
        common = self.config["common"]

        # Initialize sampler
        self.sampler = ConfigSampler(self.config)

        # Setup generation parameters
        base_seed = base_seed if base_seed is not None else common.get("base_seed")
        self.base_seed = int(base_seed) if base_seed is not None else int(secrets.randbits(32))
        self.num_instances = int(num_instances) if num_instances is not None else int(common["num_instances"])
        self.name_prefix = str(name_prefix) if name_prefix is not None else str(common["name_prefix"])

        # Setup output paths
        output_dir = output_dir or common["output_dir"]
        self.output_dir = Path(str(output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.write_manifest = bool(write_manifest) if write_manifest is not None else bool(common["write_manifest"])
        if manifest_path is None:
            manifest_path = common.get("manifest_path")
        self.manifest_path = Path(manifest_path) if manifest_path else (self.output_dir / "manifest.json")

    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file must be a mapping: {cfg_path}")
        return data

    def generate_weight_tensor(self, kind: str, meta: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Generate minimal weight tensors that satisfy schema requirements."""
        if kind == "DENSE":
            in_features = meta.get("in_features", 1)
            out_features = meta.get("out_features", 1)
            return torch.randn(out_features, in_features) * 0.1
        if kind == "CONV2D":
            in_channels = meta.get("in_channels", 1)
            out_channels = meta.get("out_channels", 1)
            kernel_size = meta.get("kernel_size", 3)
            if isinstance(kernel_size, int):
                weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
            else:
                weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
            return torch.randn(*weight_shape) * 0.1
        return None

    def _input_spec_params(
        self, meta: Dict[str, Any], input_shape: List[int], dtype: torch.dtype
    ) -> Dict[str, Any]:
        """Generate INPUT_SPEC layer parameters."""
        if meta["kind"] == InKind.BOX:
            lb_val = float(meta.get("lb_val", 0.0))
            ub_val = float(meta.get("ub_val", 1.0))
            return {
                "lb": torch.full(input_shape, lb_val, dtype=dtype),
                "ub": torch.full(input_shape, ub_val, dtype=dtype),
            }
        if meta["kind"] == InKind.LINF_BALL:
            center_val = float(meta.get("center_val", 0.5))
            eps = float(meta.get("eps", 0.0))
            center = torch.full(input_shape, center_val, dtype=dtype)
            return {"center": center, "lb": center - eps, "ub": center + eps}
        raise ValueError(f"Unsupported INPUT_SPEC kind '{meta.get('kind')}'")

    def _assert_params(self, params: Dict[str, Any], meta: Dict[str, Any], dtype: torch.dtype) -> Dict[str, Any]:
        """Convert ASSERT layer parameters to tensors."""
        kind = meta.get("kind")
        if kind == OutKind.LINEAR_LE and isinstance(params.get("c"), list):
            params["c"] = torch.as_tensor(params["c"], dtype=dtype)
        elif kind == OutKind.RANGE:
            if isinstance(params.get("lb"), list):
                params["lb"] = torch.as_tensor(params["lb"], dtype=dtype)
            if isinstance(params.get("ub"), list):
                params["ub"] = torch.as_tensor(params["ub"], dtype=dtype)
        return params

    def _generate_layer_variables(
        self, kind: str, layer_index: int, var_counter: int, meta: Dict[str, Any], layers: List[Layer]
    ) -> Tuple[List[int], List[int], int]:
        """Generate input/output variables for a layer."""
        if kind == "INPUT":
            out_num_vars = torch.Size(meta["shape"]).numel()
            out_vars = list(range(var_counter, var_counter + out_num_vars))
            return [], out_vars, var_counter + out_num_vars

        if kind == "DENSE":
            in_vars = layers[layer_index - 1].out_vars
            out_features = int(meta["out_features"])
            out_vars = list(range(var_counter, var_counter + out_features))
            return in_vars, out_vars, var_counter + out_features

        if kind in ["RELU", "SIGMOID", "TANH"]:
            in_vars = layers[layer_index - 1].out_vars
            out_vars = list(range(var_counter, var_counter + len(in_vars)))
            return in_vars, out_vars, var_counter + len(in_vars)

        if kind in ["CONV2D", "MAXPOOL2D", "AVGPOOL2D"]:
            in_vars = layers[layer_index - 1].out_vars
            out_num_vars = torch.Size(meta["output_shape"]).numel()
            out_vars = list(range(var_counter, var_counter + out_num_vars))
            return in_vars, out_vars, var_counter + out_num_vars

        if kind == "FLATTEN":
            in_vars = layers[layer_index - 1].out_vars
            out_vars = list(range(var_counter, var_counter + len(in_vars)))
            return in_vars, out_vars, var_counter + len(in_vars)

        if kind == "ADD":
            x_vars = meta["x_vars"]
            y_vars = meta["y_vars"]
            in_vars = list(x_vars) + list(y_vars)
            out_vars = list(range(var_counter, var_counter + len(x_vars)))
            return in_vars, out_vars, var_counter + len(x_vars)

        if kind in ["INPUT_SPEC", "ASSERT"]:
            prev_vars = layers[layer_index - 1].out_vars
            return prev_vars, list(prev_vars), var_counter

        raise NotImplementedError(f"Unsupported layer kind '{kind}'")

    def _sample_instance(self, idx: int) -> Dict[str, Any]:
        """Sample a single network instance configuration."""
        instance_id = f"{self.name_prefix}{int(self.base_seed)}_idx{int(idx):05d}"
        seed = int(_derive_seed(int(self.base_seed), int(idx), instance_id))
        rng = random.Random(seed)

        # Sampler returns (family, model_cfg) directly
        family, model_cfg = self.sampler.sample_family(rng)
        num_classes = int(model_cfg["num_classes"])

        input_spec = self.sampler.sample_input_spec(rng)
        output_spec = self.sampler.sample_output_spec(rng, num_classes=num_classes)

        return {
            "instance_id": instance_id,
            "seed": seed,
            "family": family,
            "model_cfg": model_cfg,
            "input_spec": input_spec,
            "output_spec": output_spec,
        }

    def _build_network_spec(self, instance: Dict[str, Any], *, dtype: str) -> Dict[str, Any]:
        """Build network specification from sampled instance."""
        model_cfg = instance["model_cfg"]
        input_shape = list(model_cfg["input_shape"])
        num_classes = int(model_cfg["num_classes"])

        layers: List[Dict[str, Any]] = []

        # INPUT layer
        input_meta: Dict[str, Any] = {
            "shape": input_shape,
            "dtype": str(dtype),
            "num_classes": num_classes,
            "value_range": list(instance["input_spec"]["value_range"]),
        }
        layers.append({"kind": "INPUT", "params": {}, "meta": input_meta})

        # INPUT_SPEC layer
        in_kind = str(instance["input_spec"]["kind"])
        spec_meta: Dict[str, Any] = {"kind": in_kind}
        if in_kind == "BOX":
            spec_meta["lb_val"] = float(instance["input_spec"]["lb_val"])
            spec_meta["ub_val"] = float(instance["input_spec"]["ub_val"])
        elif in_kind == "LINF_BALL":
            spec_meta["center_val"] = float(instance["input_spec"]["center_val"])
            spec_meta["eps"] = float(instance["input_spec"]["eps"])
        else:
            raise ValueError(f"Input spec kind '{in_kind}' is not supported")
        layers.append({"kind": "INPUT_SPEC", "params": {}, "meta": spec_meta})

        # Model layers
        if instance["family"] == "mlp":
            build_mlp_layers(layers, cfg=model_cfg)
        elif instance["family"] == "cnn2d":
            rng = random.Random(int(instance["seed"]))
            build_cnn_layers(layers, cfg=model_cfg, rng=rng)
        else:
            raise ValueError(f"Unsupported model family: {instance['family']}")

        # ASSERT layer
        out_kind = str(instance["output_spec"]["kind"])
        out_meta: Dict[str, Any] = {"kind": out_kind}
        out_params: Dict[str, Any] = {}

        if out_kind == "TOP1_ROBUST":
            out_meta["y_true"] = int(instance["output_spec"]["y_true"])
        elif out_kind == "MARGIN_ROBUST":
            out_meta["y_true"] = int(instance["output_spec"]["y_true"])
            out_meta["margin"] = float(instance["output_spec"]["margin"])
        elif out_kind == "LINEAR_LE":
            out_params["c"] = list(instance["output_spec"]["c"])
            out_meta["d"] = float(instance["output_spec"]["d"])
        elif out_kind == "RANGE":
            out_params["lb"] = list(instance["output_spec"]["lb"])
            out_params["ub"] = list(instance["output_spec"]["ub"])
        else:
            raise ValueError(f"Output spec kind '{out_kind}' is not supported")

        layers.append({"kind": "ASSERT", "params": out_params, "meta": out_meta})

        return {"layers": layers}

    def create_network(self, name: str, spec: Dict[str, Any]) -> Net:
        """Create Net object from specification."""
        dtype = get_default_dtype()
        dtype_str = str(dtype)

        layers = []
        var_counter = 0

        layer_specs = list(spec["layers"])
        for i, layer_spec in enumerate(layer_specs):
            params = layer_spec.get("params", {}).copy()
            meta = layer_spec.get("meta", {}).copy()
            kind = layer_spec["kind"]

            # Handle ADD layer inputs
            if kind == "ADD":
                inputs = layer_spec.get("inputs") or {}
                x_src = inputs.get("x")
                y_src = inputs.get("y")
                if x_src is None or y_src is None:
                    raise ValueError(f"{kind} layer requires inputs {{'x': idx, 'y': idx}} in spec")
                if x_src >= len(layers) or y_src >= len(layers):
                    raise ValueError(f"{kind} inputs must reference earlier layers (x={x_src}, y={y_src})")
                meta["x_vars"] = list(layers[x_src].out_vars)
                meta["y_vars"] = list(layers[y_src].out_vars)

            # Generate variables
            in_vars, out_vars, var_counter = self._generate_layer_variables(kind, i, var_counter, meta, layers)

            # Fill in parameters based on layer kind
            if kind == "INPUT":
                meta["dtype"] = dtype_str
            elif kind == "INPUT_SPEC":
                params.update(self._input_spec_params(meta, layers[0].meta["shape"], dtype))
            elif kind == "ASSERT":
                params = self._assert_params(params, meta, dtype)
            elif kind == "DENSE" and "W" not in params:
                weight = self.generate_weight_tensor(kind, meta)
                if weight is not None:
                    params["W"] = weight
                if meta.get("bias_enabled", False):
                    out_features = meta.get("out_features", 10)
                    params["b"] = torch.zeros(out_features, dtype=dtype)
            elif kind == "CONV2D" and "weight" not in params:
                weight = self.generate_weight_tensor(kind, meta)
                if weight is not None:
                    params["weight"] = weight

            layer = Layer(id=i, kind=kind, params=params, meta=meta, in_vars=in_vars, out_vars=out_vars)
            layers.append(layer)

        # Build predecessors and successors
        preds: Dict[int, List[int]] = {}
        for i, layer_spec in enumerate(layer_specs):
            spec_preds = layer_spec.get("preds")
            if spec_preds is None:
                preds[i] = [i - 1] if i > 0 else []
            else:
                preds[i] = list(spec_preds)

        succs: Dict[int, List[int]] = {i: [] for i in range(len(layers))}
        for i, p_list in preds.items():
            for p in p_list:
                succs[p].append(i)

        net = Net(layers=layers, preds=preds, succs=succs)
        net.meta = {"name": name}
        return net

    def save_network(self, net: Net, name: str) -> None:
        """Save network to JSON file."""
        output_path = self.output_dir / f"{name}.json"
        net_dict = NetSerializer.serialize_net(net, metadata={"generated_by": "NetFactory"})
        with open(output_path, "w") as f:
            json.dump(net_dict, f, indent=2)
        print(f"Saved: {output_path}")

    def _write_manifest(self, names: List[str]) -> None:
        """Write manifest file with generation metadata."""
        payload = {
            "base_seed": int(self.base_seed),
            "num_instances": int(self.num_instances),
            "name_prefix": self.name_prefix,
            "nets": list(names),
            "config_path": self.config_path,
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def generate(self) -> List[str]:
        """Generate all network instances."""
        print(f"Generating {self.num_instances} networks...")
        common = self.config["common"]
        dtype = str(common["dtype"])

        names: List[str] = []
        for idx in range(self.num_instances):
            instance = self._sample_instance(idx)
            name = instance["instance_id"]
            spec = self._build_network_spec(instance, dtype=dtype)
            net = self.create_network(name, spec)
            self.save_network(net, name)
            names.append(name)

        if self.write_manifest:
            self._write_manifest(names)

        print(f"All networks generated in {self.output_dir}")
        return names


if __name__ == "__main__":
    factory = NetFactory()
    factory.generate()
