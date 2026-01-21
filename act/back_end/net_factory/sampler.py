#===- act/back_end/net_factory/sampler_v2.py - Generic YAML Sampler ------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Generic rule-based sampler that reads YAML config and samples values.
#   Eliminates model-specific sampling logic by using declarative rules:
#   - choice: random selection from list
#   - range: random integer in [lo, hi]
#   - weighted: weighted random choice
#   - repeat: repeat sampling N times
#   - probability: boolean with probability p
#   - const: constant value
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple, Union


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

    def sample_family(self, rng: random.Random) -> Tuple[str, str, Dict[str, Any]]:
        """
        Sample a family and its parameters.
        Returns: (family_name, family_type, sampled_params)
        """
        # Sample family from selection strategy
        family_selection = self.config["family_selection"]
        if "weighted" in family_selection:
            weights = family_selection["weighted"]
            family_names = list(weights.keys())
            weight_values = list(weights.values())
            total = sum(weight_values)
            normalized = [w / total for w in weight_values]
            family_name = rng.choices(family_names, weights=normalized)[0]
        else:
            raise ValueError("family_selection must have 'weighted' strategy")

        # Get family configuration and type
        families = self.config["families"]
        if family_name not in families:
            raise ValueError(f"Family '{family_name}' not found in families section")

        family_config = families[family_name]
        family_type = family_config.get("type")
        if not family_type:
            raise ValueError(f"Family '{family_name}' missing required 'type' field")

        # Sample family parameters (excluding 'type')
        params_spec = {k: v for k, v in family_config.items() if k != "type"}
        params = self._sample_dict(rng, params_spec)

        # Convert types for compatibility with layer_builder
        if "input_shape" in params:
            params["input_shape"] = tuple(int(x) for x in params["input_shape"])
        if "hidden_sizes" in params:
            params["hidden_sizes"] = tuple(int(x) for x in params["hidden_sizes"])
        if "conv_channels" in params:
            params["conv_channels"] = tuple(int(x) for x in params["conv_channels"])

        return family_name, family_type, params

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
