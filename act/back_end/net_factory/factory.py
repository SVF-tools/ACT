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
#   - TF-aware network generation (via tf_capabilities)
#
#   Flow:
#   config -> ConfigSampler -> layer_builder -> Net -> JSON
#
#   TF-Aware Generation:
#   NetFactory uses tf_capabilities module to dynamically determine allowed
#   layers based on target Transfer Functions (IntervalTF, HybridzTF, DualTF).
#   This enables generating networks that are compatible with specific TFs.
#
#   Validation Strategy:
#   NetFactory relies on ACT's existing validation infrastructure:
#   - Layer.__post_init__() → validate_layer() (params/meta against REGISTRY)
#   - Net.__post_init__() → validate_graph() (layer IDs, variable validity)
#   Invalid networks raise ValueError during construction.
#   See README.md "Validation Strategy" for three-layer design details.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import json
import logging
import random
import secrets
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import torch
import yaml

from act.back_end.core import Layer, Net
from act.back_end.serialization.serialization import NetSerializer
from .tf_capabilities import get_allowed_layers
from act.front_end.specs import InKind, OutKind
from act.util.device_manager import get_default_dtype
from act.util.path_config import get_examples_gen_config_path

from .layer_builder import build_cnn_layers, build_mlp_layers

logger = logging.getLogger(__name__)


# ============================================================================
# Coverage Target - Default Layer List (Legacy Fallback)
# ============================================================================
# This list defines default layer types for coverage tracking.
# When tf_targets is specified, layers are dynamically determined from
# tf_capabilities module based on each TF's registered layer support.
#
# Note: This constant is kept for backward compatibility. New code should
# use get_allowed_layers() from tf_capabilities for dynamic layer sets.

_DEFAULT_COVERAGE_LAYERS = [
    # Linear and normalization operations
    "DENSE", "BIAS", "SCALE", "BN",
    # Activations (A-group: element-wise, low risk)
    "RELU", "LRELU", "ABS", "CLIP", "SQUARE", "POWER",
    "SIGMOID", "TANH", "SOFTPLUS", "SILU", "RELU6",
    "HARDTANH", "HARDSIGMOID", "HARDSWISH", "MISH", "SOFTSIGN",
    # Multi-input operations (B-group: fork-merge, medium risk)
    "ADD", "SUB", "MUL", "DIV", "POW", "MAX", "MIN", "MATMUL", "CONCAT",
    # CNN operations (D-group: convolution/pooling, medium risk)
    "CONV1D", "CONV2D", "CONV3D", "CONVTRANSPOSE2D",
    "MAXPOOL1D", "MAXPOOL2D", "MAXPOOL3D",
    "AVGPOOL1D", "AVGPOOL2D",
    "PAD", "UPSAMPLE", "FLATTEN",
    # Tensor operations (C-group: shape transformation, high risk)
    "RESHAPE", "TRANSPOSE", "SQUEEZE", "UNSQUEEZE",
    "TILE", "EXPAND", "SLICE", "GATHER", "INDEX_SELECT",
]

# Legacy alias for backward compatibility
COVERAGE_TARGET_LAYERS = _DEFAULT_COVERAGE_LAYERS


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


# ============================================================================
# ConfigSampler - Generic YAML-based Sampling
# ============================================================================


class ConfigSampler:
    """
    Generic sampler that uses YAML-defined sampling rules.

    Args:
        config: YAML configuration dictionary.
        allowed_layers: Set of allowed layer kinds for filtering.
            When specified, families that require unsupported layers
            are filtered out during sampling.
    """

    # Layer requirements for each network family
    # Maps family name to set of layers that MUST be supported
    _FAMILY_REQUIRED_LAYERS = {
        "mlp": {"DENSE", "RELU"},  # Basic MLP needs DENSE and activation
        "cnn2d": {"CONV2D", "DENSE", "RELU"},  # CNN needs CONV2D, DENSE, activation
    }

    # Activation functions available for sampling
    _ALL_ACTIVATIONS = frozenset({
        "RELU", "LRELU", "SIGMOID", "TANH", "RELU6",
        "HARDTANH", "HARDSIGMOID", "HARDSWISH", "SILU",
        "SOFTPLUS", "MISH", "SOFTSIGN", "ABS",
    })

    def __init__(
        self,
        config: Dict[str, Any],
        allowed_layers: Optional[FrozenSet[str]] = None
    ):
        self.config = config
        self.allowed_layers = allowed_layers or frozenset(_DEFAULT_COVERAGE_LAYERS)

        # Compute available activations
        self.available_activations = self._ALL_ACTIVATIONS & self.allowed_layers
        if not self.available_activations:
            # Fallback to RELU if no activations available
            self.available_activations = frozenset({"RELU"})
            logger.warning("No activations in allowed_layers, defaulting to RELU")

        # Compute available families
        self.available_families = self._compute_available_families()
        logger.debug(f"ConfigSampler: available_families={self.available_families}")

    def _compute_available_families(self) -> List[str]:
        """Compute families whose required layers are all in allowed_layers."""
        available = []
        for family, required in self._FAMILY_REQUIRED_LAYERS.items():
            # Check if family is in config
            if family not in self.config.get("families", {}):
                continue
            # Check if all required layers are allowed
            if required <= self.allowed_layers:
                available.append(family)
            else:
                missing = required - self.allowed_layers
                logger.debug(f"Family '{family}' excluded: missing {missing}")
        return available

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

        Filters families based on allowed_layers from tf_capabilities.
        Only families whose required layers are all supported will be sampled.

        Returns:
            (family_name, sampled_params)

        Raises:
            ValueError: If no families are available for the current allowed_layers.
        """
        # Check available families
        if not self.available_families:
            raise ValueError(
                f"No families available for allowed_layers. "
                f"Required: {self._FAMILY_REQUIRED_LAYERS}, "
                f"Allowed: {len(self.allowed_layers)} layers"
            )

        # Sample family from selection strategy (filtered)
        family_selection = self.config["family_selection"]
        if "weighted" in family_selection:
            weights = family_selection["weighted"]
            # Filter to only available families
            filtered_weights = {
                k: v for k, v in weights.items()
                if k in self.available_families
            }
            if not filtered_weights:
                raise ValueError(
                    f"No families in config match available_families: "
                    f"{self.available_families}"
                )
            family_names = list(filtered_weights.keys())
            weight_values = list(filtered_weights.values())
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

        # Filter activation functions based on allowed_layers
        params = self._filter_activations(params)

        return family, params

    def _filter_activations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter activation functions in params to only use allowed ones."""
        # Check if 'activation' is in params and filter it
        if "activation" in params:
            act = params["activation"]
            if isinstance(act, str) and act.upper() not in self.allowed_layers:
                # Replace with a random allowed activation
                params["activation"] = list(self.available_activations)[0]
                logger.debug(f"Replaced activation '{act}' with '{params['activation']}'")
        return params

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
    """
    Generator-driven factory for ACT Nets with TF-aware layer support.

    Args:
        gen_config_path: Path to YAML generation config file.
        output_dir: Output directory for generated networks.
        base_seed: Base seed for deterministic generation.
        num_instances: Number of networks to generate.
        name_prefix: Prefix for generated network names.
        write_manifest: Whether to write manifest.json.
        manifest_path: Custom manifest file path.
        tf_targets: List of target TFs for layer filtering.
            - None: Use all TFs (intersection by default)
            - ["interval"]: Only IntervalTF layers
            - ["interval", "hybridz"]: Intersection/union of both
            - ["interval", "hybridz", "dual"]: All TFs
        registry_mode: How to combine multiple TF layer sets.
            - "intersection": Layers supported by ALL targets (default, safest)
            - "union": Layers supported by ANY target (broader coverage)

    Example:
        >>> # Generate networks compatible with all TFs (safest)
        >>> factory = NetFactory(tf_targets=["interval", "hybridz", "dual"])

        >>> # Generate networks for IntervalTF only (full coverage)
        >>> factory = NetFactory(tf_targets=["interval"])

        >>> # Generate networks compatible with interval OR hybridz
        >>> factory = NetFactory(tf_targets=["interval", "hybridz"], registry_mode="union")
    """

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
        tf_targets: Optional[List[str]] = None,
        registry_mode: str = "intersection",
    ):
        self.config_path = str(gen_config_path)
        self.config = self._load_config(self.config_path)
        common = self.config["common"]

        # TF-aware layer support
        self.tf_targets = tf_targets
        self.registry_mode = registry_mode
        self.allowed_layers = self._compute_allowed_layers(tf_targets, registry_mode)
        logger.info(
            f"NetFactory: tf_targets={tf_targets}, mode={registry_mode}, "
            f"allowed_layers={len(self.allowed_layers)}"
        )

        # Initialize sampler with allowed layers for filtering
        self.sampler = ConfigSampler(self.config, allowed_layers=self.allowed_layers)

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

        # Coverage tracking - use allowed_layers instead of static list
        self.coverage_mode = common.get("coverage_mode", "basic")
        self.coverage_max_attempts = int(common.get("coverage_max_attempts", 1000))
        self.coverage_report = bool(common.get("coverage_report", True))
        self._init_coverage_stats()
        self.total_networks_generated = 0

    def _compute_allowed_layers(
        self,
        tf_targets: Optional[List[str]],
        registry_mode: str
    ) -> FrozenSet[str]:
        """
        Compute allowed layers based on TF targets and registry mode.

        Falls back to _DEFAULT_COVERAGE_LAYERS if tf_capabilities fails.
        """
        try:
            return get_allowed_layers(tf_targets, registry_mode)
        except Exception as e:
            logger.warning(
                f"Failed to get TF capabilities: {e}. "
                f"Using default coverage layers."
            )
            return frozenset(_DEFAULT_COVERAGE_LAYERS)

    def _init_coverage_stats(self) -> None:
        """Initialize coverage stats from allowed_layers."""
        # Filter to only layers that can be generated (exclude base layers)
        base_layers = {'INPUT', 'INPUT_SPEC', 'OUTPUT_SPEC', 'ASSERT'}
        identity_layers = {'DROPOUT', 'IDENTITY'}
        generatable_layers = self.allowed_layers - base_layers - identity_layers
        self.coverage_stats = {layer: 0 for layer in sorted(generatable_layers)}

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

        # Element-wise normalization layers (BIAS, SCALE, BN)
        if kind in ["BIAS", "SCALE", "BN"]:
            # Check for explicit predecessors (for fork structures)
            preds_indices = meta.get("preds_indices", None)
            if preds_indices:
                pred_idx = preds_indices[0] if isinstance(preds_indices, list) else preds_indices
                in_vars = layers[pred_idx].out_vars
            else:
                in_vars = layers[layer_index - 1].out_vars
            out_vars = list(range(var_counter, var_counter + len(in_vars)))
            return in_vars, out_vars, var_counter + len(in_vars)

        # All activation functions (element-wise operations)
        activation_kinds = [
            "RELU", "SIGMOID", "TANH", "LRELU", "RELU6", "HARDTANH", "HARDSIGMOID",
            "HARDSWISH", "SILU", "SOFTPLUS", "MISH", "SOFTSIGN", "ABS",
            "CLIP", "SQUARE", "POWER"
        ]
        if kind in activation_kinds:
            # Check for explicit predecessors (for fork structures)
            preds_indices = meta.get("preds_indices", None)
            if preds_indices:
                pred_idx = preds_indices[0] if isinstance(preds_indices, list) else preds_indices
                in_vars = layers[pred_idx].out_vars
            else:
                in_vars = layers[layer_index - 1].out_vars
            out_vars = list(range(var_counter, var_counter + len(in_vars)))
            return in_vars, out_vars, var_counter + len(in_vars)

        # All CNN spatial layers (conv, pool, upsample, pad)
        cnn_spatial_kinds = [
            "CONV1D", "CONV2D", "CONV3D", "CONVTRANSPOSE2D",
            "MAXPOOL1D", "MAXPOOL2D", "MAXPOOL3D",
            "AVGPOOL1D", "AVGPOOL2D",
            "UPSAMPLE", "PAD"
        ]
        if kind in cnn_spatial_kinds:
            # Check for explicit predecessors (for fork structures)
            preds_indices = meta.get("preds_indices", None)
            if preds_indices:
                pred_idx = preds_indices[0] if isinstance(preds_indices, list) else preds_indices
                in_vars = layers[pred_idx].out_vars
            else:
                in_vars = layers[layer_index - 1].out_vars
            out_num_vars = torch.Size(meta["output_shape"]).numel()
            out_vars = list(range(var_counter, var_counter + out_num_vars))
            return in_vars, out_vars, var_counter + out_num_vars

        if kind == "FLATTEN":
            # Check for explicit predecessors (for fork structures)
            preds_indices = meta.get("preds_indices", None)
            if preds_indices:
                pred_idx = preds_indices[0] if isinstance(preds_indices, list) else preds_indices
                in_vars = layers[pred_idx].out_vars
            else:
                in_vars = layers[layer_index - 1].out_vars
            out_vars = list(range(var_counter, var_counter + len(in_vars)))
            return in_vars, out_vars, var_counter + len(in_vars)

        # Multi-input operations (ADD, SUB, MUL, DIV, POW, MAX, MIN)
        if kind in ["ADD", "SUB", "MUL", "DIV", "POW", "MAX", "MIN"]:
            x_vars = meta.get("x_vars", [])
            y_vars = meta.get("y_vars", [])
            if not x_vars or not y_vars:
                # Fallback: get from predecessor layers
                if len(layers) >= 2:
                    x_vars = layers[layer_index - 2].out_vars
                    y_vars = layers[layer_index - 1].out_vars
                    # Update meta for later use
                    meta["x_vars"] = x_vars
                    meta["y_vars"] = y_vars
            in_vars = list(x_vars) + list(y_vars)
            out_vars = list(range(var_counter, var_counter + len(x_vars)))
            return in_vars, out_vars, var_counter + len(x_vars)

        # MATMUL - matrix multiplication
        if kind == "MATMUL":
            x_vars = meta.get("x_vars", [])
            y_vars = meta.get("y_vars", [])
            if not x_vars or not y_vars:
                # Fallback: get from predecessor layers
                if len(layers) >= 2:
                    x_vars = layers[layer_index - 2].out_vars
                    y_vars = layers[layer_index - 1].out_vars
                    meta["x_vars"] = x_vars
                    meta["y_vars"] = y_vars
            in_vars = list(x_vars) + list(y_vars)
            # For MATMUL, output size depends on matrix dimensions
            # Conservative fallback: use first input size
            if "output_shape" in meta:
                out_num_vars = torch.Size(meta["output_shape"]).numel()
            else:
                out_num_vars = len(x_vars)
            out_vars = list(range(var_counter, var_counter + out_num_vars))
            return in_vars, out_vars, var_counter + out_num_vars

        # Tensor slice/gather operations
        if kind in ["SLICE", "GATHER", "INDEX_SELECT"]:
            # Check for explicit predecessors (for fork structures)
            preds_indices = meta.get("preds_indices", None)
            if preds_indices:
                pred_idx = preds_indices[0] if isinstance(preds_indices, list) else preds_indices
                in_vars = layers[pred_idx].out_vars
            else:
                in_vars = layers[layer_index - 1].out_vars
            if "output_shape" in meta:
                out_num_vars = torch.Size(meta["output_shape"]).numel()
            else:
                # Conservative: assume same size as input
                out_num_vars = len(in_vars)
            out_vars = list(range(var_counter, var_counter + out_num_vars))
            return in_vars, out_vars, var_counter + out_num_vars

        # Shape transformation operations (element count preserved)
        if kind in ["RESHAPE", "TRANSPOSE", "SQUEEZE", "UNSQUEEZE"]:
            # Check for explicit predecessors (for fork structures)
            preds_indices = meta.get("preds_indices", None)
            if preds_indices:
                pred_idx = preds_indices[0] if isinstance(preds_indices, list) else preds_indices
                in_vars = layers[pred_idx].out_vars
            else:
                in_vars = layers[layer_index - 1].out_vars
            # These operations preserve number of elements
            out_vars = list(range(var_counter, var_counter + len(in_vars)))
            return in_vars, out_vars, var_counter + len(in_vars)

        # Expansion operations (element count increases)
        if kind in ["TILE", "EXPAND"]:
            # Check for explicit predecessors (for fork structures)
            preds_indices = meta.get("preds_indices", None)
            if preds_indices:
                pred_idx = preds_indices[0] if isinstance(preds_indices, list) else preds_indices
                in_vars = layers[pred_idx].out_vars
            else:
                in_vars = layers[layer_index - 1].out_vars
            if "output_shape" in meta:
                out_num_vars = torch.Size(meta["output_shape"]).numel()
            else:
                # Conservative: assume 2x expansion
                out_num_vars = len(in_vars) * 2
            out_vars = list(range(var_counter, var_counter + out_num_vars))
            return in_vars, out_vars, var_counter + out_num_vars

        # CONCAT - multi-input concatenation
        if kind == "CONCAT":
            # Get all predecessor variables from meta
            preds_indices = meta.get("preds_indices", [])
            if not preds_indices and layer_index >= 2:
                # Fallback: use last two layers
                preds_indices = [layer_index - 2, layer_index - 1]
            in_vars = []
            for pred_idx in preds_indices:
                if pred_idx < len(layers):
                    in_vars.extend(layers[pred_idx].out_vars)
            # Output has all concatenated variables
            out_vars = list(range(var_counter, var_counter + len(in_vars)))
            return in_vars, out_vars, var_counter + len(in_vars)

        if kind in ["INPUT_SPEC", "ASSERT"]:
            prev_vars = layers[layer_index - 1].out_vars
            return prev_vars, list(prev_vars), var_counter

        raise NotImplementedError(f"Unsupported layer kind '{kind}'")

    def _get_family_tag(self, family: str, cfg: Dict[str, Any]) -> str:
        """
        Generate family tag.

        Rules:
            MLP: mlp_{variant}  (plain/block/residual)
            CNN2D: cnn2d_plain, cnn2d_residual, or resnet (if variant=stage)
        """
        if family == "mlp":
            variant = cfg.get("variant", "plain")
            return f"mlp_{variant}"

        elif family == "cnn2d":
            variant = cfg.get("variant", "plain")
            if variant == "stage":
                return "resnet"
            elif variant == "residual":
                return "cnn2d_residual"
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
            cnn2d_residual: 32x3 (residual_channels x num_residual_blocks)
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

            elif variant == "residual":
                channels = cfg.get("residual_channels", 32)
                num_blocks = cfg.get("num_residual_blocks", 3)
                return f"{channels}x{num_blocks}"

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
        """Create Net object from specification.

        Note: Validation is automatic via Layer.__post_init__() and Net.__post_init__():
        - validate_layer(): Checks params/meta against REGISTRY (layer_util.py)
        - validate_graph(): Checks layer IDs and variable validity (layer_util.py)
        Invalid networks will raise ValueError during construction.
        """
        dtype = get_default_dtype()
        dtype_str = str(dtype)

        layers = []
        var_counter = 0

        layer_specs = list(spec["layers"])
        for i, layer_spec in enumerate(layer_specs):
            params = layer_spec.get("params", {}).copy()
            meta = layer_spec.get("meta", {}).copy()
            kind = layer_spec["kind"]

            # Handle multi-input layer inputs (ADD, SUB, DIV, MUL, POW, MAX, MIN, MATMUL)
            if kind in ["ADD", "SUB", "DIV", "MUL", "POW", "MAX", "MIN", "MATMUL"]:
                inputs = layer_spec.get("inputs") or {}
                x_src = inputs.get("x")
                y_src = inputs.get("y")
                if x_src is None or y_src is None:
                    raise ValueError(f"{kind} layer requires inputs {{'x': idx, 'y': idx}} in spec")
                if x_src >= len(layers) or y_src >= len(layers):
                    raise ValueError(f"{kind} inputs must reference earlier layers (x={x_src}, y={y_src})")
                meta["x_vars"] = list(layers[x_src].out_vars)
                meta["y_vars"] = list(layers[y_src].out_vars)

            # Handle CONCAT - multi-input concatenation
            if kind == "CONCAT":
                preds = layer_spec.get("preds", [])
                if not preds and len(layers) >= 2:
                    # Fallback: use last two layers
                    preds = [len(layers) - 2, len(layers) - 1]
                # Store preds in meta for variable generation
                meta["preds_indices"] = preds

            # Store preds_indices for ALL layers that have explicit preds (for fork structures)
            if "preds" in layer_spec and "preds_indices" not in meta:
                meta["preds_indices"] = layer_spec["preds"]

            # Generate variables
            in_vars, out_vars, var_counter = self._generate_layer_variables(kind, i, var_counter, meta, layers)

            # Handle MAX/MIN - add y_vars_list for interval_tf
            if kind in ["MAX", "MIN"]:
                preds = layer_spec.get("preds", [])
                if preds:
                    # interval_tf expects y_vars_list: list of var lists from each predecessor
                    meta["y_vars_list"] = [list(layers[p].out_vars) for p in preds if p < len(layers)]

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
            elif kind == "BIAS" and "c" not in params:
                # Auto-fill bias vector: c = zeros with size matching input
                num_elements = len(in_vars)
                params["c"] = torch.zeros(num_elements, dtype=dtype)
            elif kind == "SCALE" and "a" not in params:
                # Auto-fill scale vector: a = ones with size matching input
                num_elements = len(in_vars)
                params["a"] = torch.ones(num_elements, dtype=dtype)
            elif kind == "BN" and "A" not in params:
                # Auto-fill batch norm parameters: A = ones, c = zeros
                num_elements = len(in_vars)
                params["A"] = torch.ones(num_elements, dtype=dtype)
                params["c"] = torch.zeros(num_elements, dtype=dtype)

            # Layer.__post_init__() automatically calls validate_layer() for strict validation
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

        # Net.__post_init__() automatically calls validate_graph() for structure validation
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
            # TF-aware generation metadata
            "tf_targets": self.tf_targets,
            "registry_mode": self.registry_mode,
            "allowed_layers_count": len(self.allowed_layers),
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _record_network_layers(self, net: Net) -> None:
        """Record layer types from a generated network for coverage tracking."""
        for layer in net.layers:
            kind = layer.kind.upper()
            if kind in self.coverage_stats:
                self.coverage_stats[kind] += 1

    def _get_uncovered_layers(self) -> List[str]:
        """Get list of layers with zero occurrences."""
        return [layer for layer, count in self.coverage_stats.items() if count == 0]

    def _get_coverage_rate(self) -> float:
        """Calculate coverage rate as percentage."""
        covered = sum(1 for count in self.coverage_stats.values() if count > 0)
        total = len(self.coverage_stats)
        return (covered / total * 100.0) if total > 0 else 100.0

    def _generate_minimal_template(self, layer_kind: str) -> Optional[Dict[str, Any]]:
        """Generate minimal network template for a specific layer type.

        Returns network spec dict or None if layer cannot be generated standalone.
        """
        dtype = str(self.config["common"]["dtype"])

        # Simple 1D input shape for minimal templates
        input_shape = [1, 4]

        # Element-wise operations: INPUT -> INPUT_SPEC -> layer -> ASSERT
        if layer_kind in ["BIAS", "SCALE", "BN", "RELU", "SIGMOID", "TANH", "LRELU",
                          "RELU6", "HARDTANH", "HARDSIGMOID", "HARDSWISH", "SILU",
                          "SOFTPLUS", "MISH", "SOFTSIGN", "ABS", "CLIP", "SQUARE", "POWER"]:
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": input_shape, "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": layer_kind, "params": {}, "meta": {}},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        # DENSE: INPUT -> INPUT_SPEC -> DENSE -> ASSERT
        if layer_kind == "DENSE":
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": input_shape, "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": "DENSE", "params": {}, "meta": {"in_features": 4, "out_features": 4, "bias_enabled": True}},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        # CNN layers: use 2D input
        if layer_kind in ["CONV2D", "MAXPOOL2D", "AVGPOOL2D", "PAD", "UPSAMPLE"]:
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": [1, 1, 4, 4], "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": layer_kind, "params": {}, "meta": self._get_cnn_layer_meta(layer_kind, [1, 1, 4, 4])},
                    {"kind": "FLATTEN", "params": {}, "meta": {"start_dim": 1}},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        # 1D/3D CNN layers
        if layer_kind in ["CONV1D", "MAXPOOL1D", "AVGPOOL1D"]:
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": [1, 1, 8], "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": layer_kind, "params": {}, "meta": self._get_cnn_layer_meta(layer_kind, [1, 1, 8])},
                    {"kind": "FLATTEN", "params": {}, "meta": {"start_dim": 1}},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        if layer_kind in ["CONV3D", "MAXPOOL3D"]:
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": [1, 1, 4, 4, 4], "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": layer_kind, "params": {}, "meta": self._get_cnn_layer_meta(layer_kind, [1, 1, 4, 4, 4])},
                    {"kind": "FLATTEN", "params": {}, "meta": {"start_dim": 1}},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        # Shape operations
        if layer_kind == "FLATTEN":
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": [1, 2, 2], "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": "FLATTEN", "params": {}, "meta": {"start_dim": 1}},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        # Multi-input operations: fork two inputs
        if layer_kind in ["ADD", "SUB", "MUL", "DIV", "POW"]:
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": input_shape, "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": "RELU", "params": {}, "meta": {}},  # branch 1
                    {"kind": "SIGMOID", "params": {}, "meta": {}},  # branch 2
                    {"kind": layer_kind, "params": {}, "meta": {}, "inputs": {"x": 2, "y": 3}, "preds": [2, 3]},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        # MAX/MIN: special handling for y_vars_list
        if layer_kind in ["MAX", "MIN"]:
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": input_shape, "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": "RELU", "params": {}, "meta": {}},  # branch 1
                    {"kind": "SIGMOID", "params": {}, "meta": {}},  # branch 2
                    {"kind": layer_kind, "params": {}, "meta": {}, "inputs": {"x": 2, "y": 3}, "preds": [2, 3]},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        # MATMUL: needs real fork with parallel branches from INPUT_SPEC
        if layer_kind == "MATMUL":
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": [1, 2, 2], "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": "RELU", "params": {}, "meta": {}},  # branch 1 (from INPUT_SPEC by default)
                    {"kind": "SIGMOID", "params": {}, "meta": {}, "preds": [1]},  # branch 2 (also from INPUT_SPEC)
                    {"kind": "RESHAPE", "params": {}, "meta": {"target_shape": [2, 2]}, "preds": [2]},  # reshape branch 1
                    {"kind": "RESHAPE", "params": {}, "meta": {"target_shape": [2, 2]}, "preds": [3]},  # reshape branch 2
                    {"kind": "MATMUL", "params": {}, "meta": {"x_shape": [2, 2], "y_shape": [2, 2], "output_shape": [2, 2]}, "inputs": {"x": 4, "y": 5}, "preds": [4, 5]},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        # CONCAT: fork two inputs
        if layer_kind == "CONCAT":
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": input_shape, "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": "RELU", "params": {}, "meta": {}},  # branch 1
                    {"kind": "SIGMOID", "params": {}, "meta": {}},  # branch 2
                    {"kind": "CONCAT", "params": {}, "meta": {"concat_dim": 0}, "preds": [2, 3]},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        # Shape transformations: minimal examples
        if layer_kind == "RESHAPE":
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": [1, 2, 2], "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": "RESHAPE", "params": {}, "meta": {"target_shape": [1, 4]}},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        if layer_kind == "TRANSPOSE":
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": [1, 2, 3], "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": "TRANSPOSE", "params": {}, "meta": {"perm": [0, 2, 1]}},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        if layer_kind == "SQUEEZE":
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": [1, 1, 4], "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": "SQUEEZE", "params": {}, "meta": {"dims": [1]}},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        if layer_kind == "UNSQUEEZE":
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": [1, 4], "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": "UNSQUEEZE", "params": {}, "meta": {"dims": [1]}},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        if layer_kind == "TILE":
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": [1, 2], "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": "TILE", "params": {}, "meta": {"repeats": [1, 2], "input_shape": [1, 2], "output_shape": [1, 4]}},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        if layer_kind == "EXPAND":
            # tf_expand returns bounds with same length as input (no expansion), so use non-expanding shape
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": [1, 1], "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": "EXPAND", "params": {}, "meta": {"shape": [1, 1], "input_shape": [1, 1], "output_shape": [1, 1]}},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        if layer_kind == "SLICE":
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": [1, 8], "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": "SLICE", "params": {}, "meta": {"starts": [0], "ends": [4], "axes": [1], "input_shape": [1, 8], "output_shape": [1, 4]}},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        if layer_kind == "GATHER":
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": [1, 8], "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": "GATHER", "params": {}, "meta": {"indices": [0, 2, 4], "axis": 1, "input_shape": [1, 8], "output_shape": [1, 3]}},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        if layer_kind == "INDEX_SELECT":
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": [1, 8], "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": "INDEX_SELECT", "params": {}, "meta": {"indices": [0, 2, 4], "dim": 1, "input_shape": [1, 8], "output_shape": [1, 3]}},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        if layer_kind == "CONVTRANSPOSE2D":
            return {
                "layers": [
                    {"kind": "INPUT", "params": {}, "meta": {"shape": [1, 1, 2, 2], "dtype": dtype}},
                    {"kind": "INPUT_SPEC", "params": {}, "meta": {"kind": "BOX"}},
                    {"kind": "CONVTRANSPOSE2D", "params": {}, "meta": {
                        "in_channels": 1, "out_channels": 1, "kernel_size": 3,
                        "stride": 2, "padding": 1, "output_padding": 1,
                        "input_shape": [1, 1, 2, 2], "output_shape": [1, 1, 4, 4]
                    }},
                    {"kind": "FLATTEN", "params": {}, "meta": {"start_dim": 1}},
                    {"kind": "ASSERT", "params": {}, "meta": {"kind": "TOP1_ROBUST", "target": 0}},
                ],
            }

        return None

    def _get_cnn_layer_meta(self, layer_kind: str, input_shape: List[int]) -> Dict[str, Any]:
        """Get minimal meta for CNN layers."""
        if layer_kind == "CONV2D":
            return {
                "in_channels": 1, "out_channels": 1, "kernel_size": 3,
                "stride": 1, "padding": 1,
                "input_shape": input_shape, "output_shape": [1, 1, 4, 4]
            }
        elif layer_kind == "CONV1D":
            return {
                "in_channels": 1, "out_channels": 1, "kernel_size": 3,
                "stride": 1, "padding": 1,
                "input_shape": input_shape, "output_shape": [1, 1, 8]
            }
        elif layer_kind == "CONV3D":
            return {
                "in_channels": 1, "out_channels": 1, "kernel_size": 3,
                "stride": 1, "padding": 1,
                "input_shape": input_shape, "output_shape": [1, 1, 4, 4, 4]
            }
        elif layer_kind in ["MAXPOOL2D", "AVGPOOL2D"]:
            return {
                "kernel_size": 2, "stride": 2, "padding": 0,
                "input_shape": input_shape, "output_shape": [1, 1, 2, 2]
            }
        elif layer_kind in ["MAXPOOL1D", "AVGPOOL1D"]:
            return {
                "kernel_size": 2, "stride": 2, "padding": 0,
                "input_shape": input_shape, "output_shape": [1, 1, 4]
            }
        elif layer_kind == "MAXPOOL3D":
            return {
                "kernel_size": 2, "stride": 2, "padding": 0,
                "input_shape": input_shape, "output_shape": [1, 1, 2, 2, 2]
            }
        elif layer_kind == "PAD":
            return {
                "pad": [1, 1, 1, 1], "mode": "constant", "value": 0.0,
                "input_shape": input_shape, "output_shape": [1, 1, 6, 6]
            }
        elif layer_kind == "UPSAMPLE":
            return {
                "scale_factor": 2.0, "mode": "nearest",
                "input_shape": input_shape, "output_shape": [1, 1, 8, 8]
            }
        return {}

    def _print_coverage_report(self) -> None:
        """Print coverage statistics with TF-aware information."""
        if not self.coverage_report:
            return

        covered = sum(1 for count in self.coverage_stats.values() if count > 0)
        total = len(self.coverage_stats)
        rate = self._get_coverage_rate()

        print(f"\n{'='*60}")
        print(f"Layer Coverage Report")
        print(f"{'='*60}")

        # TF-aware info
        if self.tf_targets:
            print(f"TF Targets: {self.tf_targets} (mode: {self.registry_mode})")
        else:
            print(f"TF Targets: all (mode: {self.registry_mode})")
        print(f"Allowed Layers: {len(self.allowed_layers)}")
        print(f"Trackable Layers: {total}")

        print(f"\nCoverage Rate: {covered}/{total} ({rate:.1f}%)")
        print(f"Total Networks Generated: {self.total_networks_generated}")

        uncovered = self._get_uncovered_layers()
        if uncovered:
            print(f"\nUncovered Layers ({len(uncovered)}):")
            for layer in sorted(uncovered):
                print(f"  - {layer}")
        else:
            print(f"\nAll target layers covered!")

        print(f"\nCovered Layers (count > 0):")
        for layer in sorted(self.coverage_stats.keys()):
            count = self.coverage_stats[layer]
            if count > 0:
                print(f"  {layer}: {count}")
        print(f"{'='*60}\n")

    def generate(self) -> List[str]:
        """Generate all network instances with TF-aware layer support."""
        common = self.config["common"]
        dtype = str(common["dtype"])
        names: List[str] = []

        if self.coverage_mode == "full":
            print(f"Generating networks in FULL coverage mode...")
            print(f"Target: {len(self.coverage_stats)} layer types")
            print(f"Max attempts: {self.coverage_max_attempts}")

            # Generate networks until all layers covered or max attempts reached
            for idx in range(self.coverage_max_attempts):
                instance = self._sample_instance(idx)
                name = instance["instance_id"]
                spec = self._build_network_spec(instance, dtype=dtype)
                net = self.create_network(name, spec)
                self.save_network(net, name)
                names.append(name)
                self.total_networks_generated += 1

                # Update coverage
                self._record_network_layers(net)
                uncovered = self._get_uncovered_layers()

                # Progress report every 50 networks
                if (idx + 1) % 50 == 0:
                    rate = self._get_coverage_rate()
                    print(f"  Generated {idx + 1} networks, coverage: {rate:.1f}%, uncovered: {len(uncovered)}")

                # Stop if all covered
                if len(uncovered) == 0:
                    print(f"\n✓ All layers covered after {idx + 1} networks!")
                    break
            else:
                print(f"\n⚠ Reached max attempts ({self.coverage_max_attempts}), some layers may be uncovered")

            # Generate minimal templates for remaining uncovered layers
            # Only generate templates for layers that are in allowed_layers
            uncovered = self._get_uncovered_layers()
            if uncovered:
                # Filter to only layers in allowed_layers
                generatable = [l for l in uncovered if l in self.allowed_layers]
                if generatable:
                    print(f"\nGenerating minimal templates for {len(generatable)} uncovered layers...")
                    for layer_kind in sorted(generatable):
                        template_spec = self._generate_minimal_template(layer_kind)
                        if template_spec:
                            try:
                                name = f"template_{layer_kind.lower()}_minimal"
                                net = self.create_network(name, template_spec)
                                self.save_network(net, name)
                                names.append(name)
                                self.total_networks_generated += 1
                                self._record_network_layers(net)
                                print(f"  Generated template for {layer_kind}")
                            except Exception as e:
                                print(f"  Failed to generate template for {layer_kind}: {e}")
                        else:
                            print(f"  - Skipped {layer_kind} (complex, requires manual implementation)")

        else:  # basic mode
            print(f"Generating {self.num_instances} networks in BASIC mode...")
            for idx in range(self.num_instances):
                instance = self._sample_instance(idx)
                name = instance["instance_id"]
                spec = self._build_network_spec(instance, dtype=dtype)
                net = self.create_network(name, spec)
                self.save_network(net, name)
                names.append(name)
                self.total_networks_generated += 1

                # Track coverage even in basic mode
                self._record_network_layers(net)

        if self.write_manifest:
            self._write_manifest(names)

        print(f"\nAll networks generated in {self.output_dir}")

        # Print coverage report
        self._print_coverage_report()

        return names


if __name__ == "__main__":
    factory = NetFactory()
    factory.generate()
