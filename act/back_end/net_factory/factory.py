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


# ============================================================================
# ConfigSampler - Generic YAML-based Sampling
# ============================================================================


class ConfigSampler:
    """Generic sampler that uses YAML-defined sampling rules."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _sample_value(self, rng: random.Random, rule: Any) -> Any:
        """
        Sample a value based on a rule definition.

        Rules:
        - {choice: [v1, v2, ...]} -> random choice
        - {range: [lo, hi]} -> random int in [lo, hi]
        - {weighted: {k1: w1, k2: w2, ...}} -> weighted choice
        - {repeat: {count: rule, value: rule}} -> list of sampled values
        - {probability: p} -> boolean (True with probability p)
        - {const: v} -> constant value v
        - plain value -> return as-is
        """
        if not isinstance(rule, dict):
            return rule

        if "const" in rule:
            return rule["const"]

        if "choice" in rule:
            items = rule["choice"]
            if not items:
                raise ValueError("choice rule must have non-empty list")
            return rng.choice(items)

        if "range" in rule:
            lo, hi = rule["range"]
            lo, hi = int(lo), int(hi)
            if hi < lo:
                lo, hi = hi, lo
            return rng.randint(lo, hi)

        if "weighted" in rule:
            weights = rule["weighted"]
            if not weights:
                raise ValueError("weighted rule must have non-empty dict")
            items = list(weights.keys())
            probs = list(weights.values())
            total = sum(probs)
            if total <= 0:
                raise ValueError("weighted rule must have positive total weight")
            normalized = [p / total for p in probs]
            return rng.choices(items, weights=normalized)[0]

        if "repeat" in rule:
            repeat_rule = rule["repeat"]
            count = self._sample_value(rng, repeat_rule["count"])
            value_rule = repeat_rule["value"]
            return [self._sample_value(rng, value_rule) for _ in range(int(count))]

        if "probability" in rule:
            p = float(rule["probability"])
            return rng.random() < p

        raise ValueError(f"Unknown sampling rule: {rule}")

    def _sample_dict(self, rng: random.Random, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sample all values in a dict spec."""
        result = {}
        for key, value in spec.items():
            if isinstance(value, dict):
                # Check if it's a sampling rule or nested dict
                is_rule = any(k in value for k in ["choice", "range", "weighted", "repeat", "probability", "const"])
                if is_rule:
                    result[key] = self._sample_value(rng, value)
                else:
                    result[key] = self._sample_dict(rng, value)
            else:
                result[key] = value
        return result

    def sample_family(self, rng: random.Random) -> Tuple[str, Dict[str, Any]]:
        """
        Sample a family and its parameters.
        Returns: (family_name, sampled_params)
        """
        # Sample family from selection strategy
        family_selection = self.config["family_selection"]
        if "weighted" in family_selection:
            weights = family_selection["weighted"]
            family_names = list(weights.keys())
            weight_values = list(weights.values())
            total = sum(weight_values)
            normalized = [w / total for w in weight_values]
            family = rng.choices(family_names, weights=normalized)[0]
        else:
            raise ValueError("family_selection must have 'weighted' strategy")

        # Sample family parameters
        families = self.config["families"]
        params_spec = families[family]
        params = self._sample_dict(rng, params_spec)

        # Convert types for compatibility with layer_builder
        if "input_shape" in params:
            params["input_shape"] = tuple(int(x) for x in params["input_shape"])
        if "hidden_sizes" in params:
            params["hidden_sizes"] = tuple(int(x) for x in params["hidden_sizes"])
        if "conv_channels" in params:
            params["conv_channels"] = tuple(int(x) for x in params["conv_channels"])

        return family, params

    def sample_input_spec(self, rng: random.Random) -> Dict[str, Any]:
        """Sample input specification."""
        spec_config = self.config["input_spec"]
        kind = self._sample_value(rng, spec_config["kind"])
        value_range = self._sample_value(rng, spec_config["value_range"])
        lo, hi = float(value_range[0]), float(value_range[1])
        if hi < lo:
            lo, hi = hi, lo

        if kind == "BOX":
            shrink_range = spec_config.get("box_shrink_range", [0.0, 0.2])
            span = hi - lo
            shrink_a = rng.random() * shrink_range[1]
            shrink_b = rng.random() * shrink_range[1]
            lb_val = lo + span * shrink_a
            ub_val = hi - span * shrink_b
            if ub_val < lb_val:
                lb_val, ub_val = lo, hi
            return {
                "kind": "BOX",
                "value_range": (lo, hi),
                "lb_val": float(lb_val),
                "ub_val": float(ub_val),
            }

        if kind == "LINF_BALL":
            center_val = lo + (hi - lo) * rng.random()
            eps = self._sample_value(rng, spec_config["eps"])
            eps = min(float(eps), 0.5 * (hi - lo)) if (hi > lo) else 0.0
            return {
                "kind": "LINF_BALL",
                "value_range": (lo, hi),
                "center_val": float(center_val),
                "eps": float(eps),
            }

        raise ValueError(f"Unsupported input_spec kind '{kind}'")

    def sample_output_spec(self, rng: random.Random, *, num_classes: int) -> Dict[str, Any]:
        """Sample output specification."""
        spec_config = self.config["output_spec"]
        kind = self._sample_value(rng, spec_config["kind"])
        y_true = int(rng.randrange(int(num_classes)))

        if kind == "TOP1_ROBUST":
            return {"kind": "TOP1_ROBUST", "y_true": y_true}

        if kind == "MARGIN_ROBUST":
            margin = self._sample_value(rng, spec_config["margin"])
            return {"kind": "MARGIN_ROBUST", "y_true": y_true, "margin": float(margin)}

        if kind == "LINEAR_LE":
            c_range = spec_config["linear_le_c_range"]
            d_range = spec_config["linear_le_d_range"]
            c_lo, c_hi = c_range[0], c_range[1]
            d_lo, d_hi = d_range[0], d_range[1]
            c_vals = [c_lo + (c_hi - c_lo) * rng.random() for _ in range(int(num_classes))]
            d_val = d_lo + (d_hi - d_lo) * rng.random()
            return {"kind": "LINEAR_LE", "c": [float(x) for x in c_vals], "d": float(d_val)}

        if kind == "RANGE":
            bounds = self._sample_value(rng, spec_config["range_bounds"])
            lo, hi = bounds[0], bounds[1]
            lb_vals = []
            ub_vals = []
            for _ in range(int(num_classes)):
                a = lo + (hi - lo) * rng.random()
                b = lo + (hi - lo) * rng.random()
                lb_vals.append(min(a, b))
                ub_vals.append(max(a, b))
            return {
                "kind": "RANGE",
                "lb": [float(x) for x in lb_vals],
                "ub": [float(x) for x in ub_vals],
            }

        raise ValueError(f"Unsupported output_spec kind '{kind}'")


# ============================================================================
# NetFactory - Main Orchestration
# ============================================================================


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

        if kind == "CONV1D":
            in_channels = meta.get("in_channels", 1)
            out_channels = meta.get("out_channels", 1)
            kernel_size = meta.get("kernel_size", 3)
            if isinstance(kernel_size, int):
                weight_shape = (out_channels, in_channels, kernel_size)
            else:
                weight_shape = (out_channels, in_channels, kernel_size[0])
            return torch.randn(*weight_shape) * 0.1

        if kind == "CONV2D":
            in_channels = meta.get("in_channels", 1)
            out_channels = meta.get("out_channels", 1)
            kernel_size = meta.get("kernel_size", 3)
            if isinstance(kernel_size, int):
                weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
            else:
                weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
            return torch.randn(*weight_shape) * 0.1

        if kind == "CONV3D":
            in_channels = meta.get("in_channels", 1)
            out_channels = meta.get("out_channels", 1)
            kernel_size = meta.get("kernel_size", 3)
            if isinstance(kernel_size, int):
                weight_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
            else:
                weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])
            return torch.randn(*weight_shape) * 0.1

        if kind == "CONVTRANSPOSE2D":
            # Note: ConvTranspose2D weight shape is [in_channels, out_channels, k, k]
            in_channels = meta.get("in_channels", 1)
            out_channels = meta.get("out_channels", 1)
            kernel_size = meta.get("kernel_size", 3)
            if isinstance(kernel_size, int):
                weight_shape = (in_channels, out_channels, kernel_size, kernel_size)
            else:
                weight_shape = (in_channels, out_channels, kernel_size[0], kernel_size[1])
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

        # All activation functions (element-wise operations)
        activation_kinds = [
            "RELU", "SIGMOID", "TANH", "LRELU", "RELU6", "HARDTANH", "HARDSIGMOID",
            "HARDSWISH", "SILU", "SOFTPLUS", "MISH", "SOFTSIGN", "GELU", "ABS",
            "CLIP", "SQUARE", "POWER"
        ]
        if kind in activation_kinds:
            in_vars = layers[layer_index - 1].out_vars
            out_vars = list(range(var_counter, var_counter + len(in_vars)))
            return in_vars, out_vars, var_counter + len(in_vars)

        # All CNN spatial layers (conv, pool, upsample, pad)
        cnn_spatial_kinds = [
            "CONV1D", "CONV2D", "CONV3D", "CONVTRANSPOSE2D",
            "MAXPOOL1D", "MAXPOOL2D", "MAXPOOL3D",
            "AVGPOOL1D", "AVGPOOL2D", "AVGPOOL3D",
            "UPSAMPLE", "PAD"
        ]
        if kind in cnn_spatial_kinds:
            in_vars = layers[layer_index - 1].out_vars
            out_num_vars = torch.Size(meta["output_shape"]).numel()
            out_vars = list(range(var_counter, var_counter + out_num_vars))
            return in_vars, out_vars, var_counter + out_num_vars

        if kind == "FLATTEN":
            in_vars = layers[layer_index - 1].out_vars
            out_vars = list(range(var_counter, var_counter + len(in_vars)))
            return in_vars, out_vars, var_counter + len(in_vars)

        # Multi-input operations (ADD, SUB, MUL, DIV, POW)
        if kind in ["ADD", "SUB", "MUL", "DIV", "POW"]:
            x_vars = meta["x_vars"]
            y_vars = meta["y_vars"]
            in_vars = list(x_vars) + list(y_vars)
            out_vars = list(range(var_counter, var_counter + len(x_vars)))
            return in_vars, out_vars, var_counter + len(x_vars)

        if kind == "MATMUL":
            x_vars = meta["x_vars"]
            y_vars = meta["y_vars"]
            in_vars = list(x_vars) + list(y_vars)
            output_shape = meta["output_shape"]
            out_num_vars = output_shape[0] * output_shape[1]
            out_vars = list(range(var_counter, var_counter + out_num_vars))
            return in_vars, out_vars, var_counter + out_num_vars

        # Tensor slice/gather operations
        if kind in ["SLICE", "GATHER", "INDEX_SELECT"]:
            in_vars = layers[layer_index - 1].out_vars
            if "output_shape" in meta:
                out_num_vars = torch.Size(meta["output_shape"]).numel()
            else:
                # Conservative: assume same size as input
                out_num_vars = len(in_vars)
            out_vars = list(range(var_counter, var_counter + out_num_vars))
            return in_vars, out_vars, var_counter + out_num_vars

        if kind in ["INPUT_SPEC", "ASSERT"]:
            prev_vars = layers[layer_index - 1].out_vars
            return prev_vars, list(prev_vars), var_counter

        raise NotImplementedError(f"Unsupported layer kind '{kind}'")

    def _get_family_tag(self, family: str, cfg: Dict[str, Any]) -> str:
        """
        Generate family tag.

        Rules:
            MLP: mlp_{variant}  (plain/block/residual)
            CNN2D: cnn2d_plain or resnet (if variant=stage)
        """
        if family == "mlp":
            variant = cfg.get("variant", "plain")
            return f"mlp_{variant}"

        elif family == "cnn2d":
            variant = cfg.get("variant", "plain")
            if variant == "stage":
                return "resnet"
            return "cnn2d_plain"

        else:
            return family

    def _format_input_shape(self, input_shape: tuple) -> str:
        """
        Format input shape (remove batch dimension).

        Examples:
            (1, 6) -> "6"
            (1, 3, 8) -> "3x8"
            (1, 3, 16, 16) -> "3x16x16"
        """
        dims = input_shape[1:] if input_shape[0] == 1 else input_shape
        return "x".join(str(d) for d in dims)

    def _format_structure(self, family: str, cfg: Dict[str, Any]) -> str:
        """
        Format structure summary.

        Rules:
            mlp_plain: 32x64x64 (hidden_sizes)
            mlp_block: 64x4 (block_width x num_blocks)
            mlp_residual: 128x2 (residual_width x num_residual_blocks)
            cnn2d_plain: 8x16x32 (conv_channels)
            resnet: 16x3x2 (base_channels x stages x blocks_per_stage)
        """
        if family == "mlp":
            variant = cfg.get("variant", "plain")

            if variant == "plain":
                hidden = cfg.get("hidden_sizes", ())
                return "x".join(str(h) for h in hidden)

            elif variant == "block":
                width = cfg.get("block_width", 64)
                num_blocks = cfg.get("num_blocks", 3)
                return f"{width}x{num_blocks}"

            elif variant == "residual":
                width = cfg.get("residual_width", 128)
                num_blocks = cfg.get("num_residual_blocks", 2)
                return f"{width}x{num_blocks}"

        elif family == "cnn2d":
            variant = cfg.get("variant", "plain")

            if variant == "plain":
                channels = cfg.get("conv_channels", ())
                return "x".join(str(c) for c in channels)

            elif variant == "stage":
                base = cfg.get("base_channels", 16)
                stages = cfg.get("num_stages", 3)
                blocks = cfg.get("blocks_per_stage", 2)
                return f"{base}x{stages}x{blocks}"

        return "default"

    def _generate_semantic_name(
        self, family: str, model_cfg: Dict[str, Any], seed: int
    ) -> str:
        """
        Generate semantic filename: {family_tag}_{input}_{structure}_{seed}

        Examples:
            mlp_plain_6_32x64x64_12345
            resnet_3x16x16_16x3x2_98765
        """
        family_tag = self._get_family_tag(family, model_cfg)
        input_str = self._format_input_shape(model_cfg["input_shape"])
        structure_str = self._format_structure(family, model_cfg)
        return f"{family_tag}_{input_str}_{structure_str}_{seed}"

    def _sample_instance(self, idx: int) -> Dict[str, Any]:
        """Sample a single network instance configuration."""
        temp_id = f"{self.name_prefix}{int(self.base_seed)}_idx{int(idx):05d}"
        seed = int(_derive_seed(int(self.base_seed), int(idx), temp_id))
        rng = random.Random(seed)

        # Sample family and configuration
        family, model_cfg = self.sampler.sample_family(rng)
        num_classes = int(model_cfg["num_classes"])

        # Generate semantic instance name
        instance_id = self._generate_semantic_name(family, model_cfg, seed)

        # Sample input and output specifications
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

            # Handle multi-input layer inputs (ADD, SUB, DIV, MUL, POW, MATMUL)
            if kind in ["ADD", "SUB", "DIV", "MUL", "POW", "MATMUL"]:
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
            elif kind in ["CONV1D", "CONV2D", "CONV3D", "CONVTRANSPOSE2D"] and "weight" not in params:
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
