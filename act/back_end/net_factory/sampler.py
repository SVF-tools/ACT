#===- act/back_end/net_factory/sampler.py - Config Sampling Logic --------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Sampling-only module for NetFactory.
#   - Reads config dicts and samples architecture/spec parameters.
#   - Uses caller-provided RNG for deterministic runs.
#   - Produces plain dicts consumed by layer_builder/factory.
#   - No file I/O and no Net/Layer construction here.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import random
from typing import Any, Dict, List

from .utils import choose, randint_inclusive


class ConfigSampler:
    """Samples architecture and spec configs from YAML config."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def sample_family(self, rng: random.Random) -> str:
        """Sample model family (mlp or cnn2d)."""
        gen = self.config["generator"]
        fams = list(gen["families"])
        if len(fams) == 1:
            return str(fams[0])

        has_mlp = "mlp" in fams
        has_cnn = "cnn2d" in fams
        if has_mlp and has_cnn:
            return "mlp" if (rng.random() < float(gen["p_mlp"])) else "cnn2d"
        return str(rng.choice(fams))

    def sample_mlp(self, rng: random.Random, *, num_classes: int) -> Dict[str, Any]:
        """Sample MLP configuration."""
        cfg = self.config["mlp"]
        input_shape = choose(rng, cfg["input_shapes"], name="mlp.input_shapes")
        depth = randint_inclusive(rng, cfg["depth_range"])
        if depth <= 0:
            raise ValueError(f"mlp.depth_range produced non-positive depth={depth}")

        widths = [int(choose(rng, cfg["width_choices"], name="mlp.width_choices")) for _ in range(depth)]
        activation = str(choose(rng, cfg["activation_choices"], name="mlp.activation_choices"))

        # Determine variant
        block_p = float(cfg["block_p"])
        residual_p = float(cfg["residual_p"])
        r = rng.random()
        if r < residual_p:
            variant = "residual"
        elif r < residual_p + block_p:
            variant = "block"
        else:
            variant = "plain"

        return {
            "input_shape": tuple(int(x) for x in input_shape),
            "hidden_sizes": tuple(widths),
            "variant": variant,
            "num_blocks": int(randint_inclusive(rng, cfg["block_count_range"])),
            "block_width": int(choose(rng, cfg["block_width_choices"], name="mlp.block_width_choices")),
            "post_block_activation": bool(rng.random() < float(cfg["post_block_activation_p"])),
            "num_residual_blocks": int(randint_inclusive(rng, cfg["residual_blocks_range"])),
            "residual_width": int(choose(rng, cfg["residual_width_choices"], name="mlp.residual_width_choices")),
            "activation": activation,
            "use_bias": True,
            "num_classes": int(num_classes),
        }

    def sample_cnn2d(self, rng: random.Random, *, num_classes: int) -> Dict[str, Any]:
        """Sample CNN configuration."""
        cfg = self.config["cnn"]
        input_shape = choose(rng, cfg["input_shapes"], name="cnn.input_shapes")
        if int(input_shape[2]) > 32 or int(input_shape[3]) > 32:
            raise ValueError(f"cnn.input_shapes must have H,W <= 32, got {input_shape}")

        variant = "stage" if (rng.random() < float(cfg["stage_variant_p"])) else "plain"
        blocks = randint_inclusive(rng, cfg["num_blocks_range"])
        if blocks <= 0:
            raise ValueError(f"cnn.num_blocks_range produced non-positive blocks={blocks}")

        # Sample conv channels
        conv_channels: List[int] = []
        for _ in range(blocks):
            ch = int(choose(rng, cfg["channels_choices"], name="cnn.channels_choices"))
            conv_channels.append(ch)

        # Sample stage parameters
        stages = randint_inclusive(rng, cfg["stages_range"])
        base_channels = int(choose(rng, cfg["base_channels_choices"], name="cnn.base_channels_choices"))
        channel_mult = int(choose(rng, cfg["channel_mult_choices"], name="cnn.channel_mult_choices"))

        # Limit max channels to 64
        max_channels = base_channels * (channel_mult ** (stages - 1))
        if max_channels > 64:
            stages = max(1, min(stages, 3))
            while stages > 1 and base_channels * (channel_mult ** (stages - 1)) > 64:
                stages -= 1
            max_channels = base_channels * (channel_mult ** (stages - 1))
            if max_channels > 64:
                base_channels = min(base_channels, 64)

        return {
            "input_shape": tuple(int(x) for x in input_shape),
            "conv_channels": tuple(conv_channels),
            "variant": variant,
            "stages": int(stages),
            "blocks_per_stage": int(randint_inclusive(rng, cfg["blocks_per_stage_range"])),
            "base_channels": int(base_channels),
            "channel_mult": int(channel_mult),
            "downsample": str(choose(rng, cfg["downsample_choices"], name="cnn.downsample_choices")),
            "double_conv_p": float(choose(rng, cfg["double_conv_p_choices"], name="cnn.double_conv_p_choices")),
            "head_pool_to_1x1": True,
            "kernel_sizes": int(choose(rng, cfg["kernel_choices"], name="cnn.kernel_choices")),
            "strides": int(choose(rng, cfg["stride_choices"], name="cnn.stride_choices")),
            "paddings": int(choose(rng, cfg["padding_choices"], name="cnn.padding_choices")),
            "activation": str(choose(rng, cfg["activation_choices"], name="cnn.activation_choices")),
            "use_bias": True,
            "use_maxpool": bool(rng.random() < float(cfg["use_maxpool_p"])),
            "maxpool_kernel": 2,
            "maxpool_stride": 2,
            "num_classes": int(num_classes),
            "fc_hidden": int(choose(rng, cfg["fc_hidden_choices"], name="cnn.fc_hidden_choices")),
        }

    def sample_input_spec(self, rng: random.Random) -> Dict[str, Any]:
        """Sample input specification."""
        cfg = self.config["input_spec"]
        kinds = list(cfg["kind_choices"])

        # Choose kind
        if len(kinds) == 1:
            kind = kinds[0]
        else:
            has_box = "BOX" in kinds
            has_linf = "LINF_BALL" in kinds
            if has_box and has_linf and len(kinds) == 2:
                kind = "BOX" if (rng.random() < float(cfg["p_box"])) else "LINF_BALL"
            else:
                kind = rng.choice(kinds)

        # Sample value range
        value_range = choose(rng, cfg["value_range_choices"], name="input_spec.value_range_choices")
        lo, hi = float(value_range[0]), float(value_range[1])
        if hi < lo:
            lo, hi = hi, lo

        if kind == "BOX":
            span = hi - lo
            shrink_a = rng.random() * 0.2
            shrink_b = rng.random() * 0.2
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
            eps = float(choose(rng, cfg["eps_choices"], name="input_spec.eps_choices"))
            eps = min(eps, 0.5 * (hi - lo)) if (hi > lo) else 0.0
            return {
                "kind": "LINF_BALL",
                "value_range": (lo, hi),
                "center_val": float(center_val),
                "eps": float(eps),
            }

        raise ValueError(f"Unsupported input_spec kind '{kind}'")

    def sample_output_spec(self, rng: random.Random, *, num_classes: int) -> Dict[str, Any]:
        """Sample output specification."""
        cfg = self.config["output_spec"]
        kinds = list(cfg["kind_choices"])

        # Choose kind
        if len(kinds) == 1:
            kind = kinds[0]
        else:
            has_top1 = "TOP1_ROBUST" in kinds
            has_margin = "MARGIN_ROBUST" in kinds
            if has_top1 and has_margin and len(kinds) == 2:
                kind = "TOP1_ROBUST" if (rng.random() < float(cfg["p_top1"])) else "MARGIN_ROBUST"
            else:
                kind = rng.choice(kinds)

        y_true = int(rng.randrange(int(num_classes)))

        if kind == "TOP1_ROBUST":
            return {"kind": "TOP1_ROBUST", "y_true": y_true}

        if kind == "MARGIN_ROBUST":
            margin = float(choose(rng, cfg["margin_choices"], name="output_spec.margin_choices"))
            return {"kind": "MARGIN_ROBUST", "y_true": y_true, "margin": float(margin)}

        if kind == "LINEAR_LE":
            c_lo, c_hi = cfg["linear_le_c_range"]
            d_lo, d_hi = cfg["linear_le_d_range"]
            c_vals = [c_lo + (c_hi - c_lo) * rng.random() for _ in range(int(num_classes))]
            d_val = d_lo + (d_hi - d_lo) * rng.random()
            return {"kind": "LINEAR_LE", "c": [float(x) for x in c_vals], "d": float(d_val)}

        if kind == "RANGE":
            lo, hi = choose(rng, cfg["range_choices"], name="output_spec.range_choices")
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
