#!/usr/bin/env python3
#===- act/back_end/confignet_spec.py - ConfigNet Spec + Sampling -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   ConfigNet dataclasses, reproducible seeding, and sampling.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


def _stable_u32_from_bytes(data: bytes) -> int:
    return int.from_bytes(data[:4], byteorder="little", signed=False)


def derive_seed(base_seed: int, idx: int, instance_id: Optional[str] = None) -> int:
    """
    Derive a stable per-instance seed from (base_seed, idx, instance_id).
    """
    if not isinstance(base_seed, int):
        raise TypeError(f"base_seed must be int, got {type(base_seed)}")
    if not isinstance(idx, int):
        raise TypeError(f"idx must be int, got {type(idx)}")

    payload = f"{base_seed}|{idx}|{instance_id or ''}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return _stable_u32_from_bytes(digest)


# --------------------------
# Model Configs
# --------------------------

class ModelFamily(str, Enum):
    MLP = "mlp"
    CNN2D = "cnn2d"


def _to_basic(obj: Any, *, strict: bool = False) -> Any:
    """Convert objects to YAML/JSON-serializable primitives."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [_to_basic(x, strict=strict) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_basic(v, strict=strict) for k, v in obj.items()}
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return _to_basic(obj.to_dict(), strict=strict)
    if is_dataclass(obj):
        return _to_basic(asdict(obj), strict=strict)
    if strict:
        raise TypeError(f"Object is not JSON-serializable: {type(obj)}")
    return str(obj)


@dataclass(frozen=True)
class MLPConfig:
    """
    MLP architecture configuration.
    """
    input_shape: Tuple[int, ...]
    hidden_sizes: Tuple[int, ...]
    variant: str = "plain"
    num_blocks: int = 4
    block_width: Optional[int] = None
    post_block_activation: bool = True
    activation: str = "relu"
    use_bias: bool = True
    dropout_p: float = 0.0
    num_classes: int = 10

    def __post_init__(self):
        if len(self.input_shape) < 2:
            raise ValueError(f"MLPConfig.input_shape must include batch dim, got {self.input_shape}")
        if self.input_shape[0] != 1:
            raise ValueError(f"ConfigNet assumes batch=1, got input_shape={self.input_shape}")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be >= 2")
        if any(h <= 0 for h in self.hidden_sizes):
            raise ValueError(f"hidden_sizes must be positive, got {self.hidden_sizes}")
        if not (0.0 <= self.dropout_p < 1.0):
            raise ValueError(f"dropout_p must be in [0,1), got {self.dropout_p}")
        if self.variant not in ("plain", "block"):
            raise ValueError(f"MLPConfig.variant must be 'plain' or 'block', got {self.variant}")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be >= 1")
        if self.block_width is not None and self.block_width <= 0:
            raise ValueError("block_width must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_shape": list(self.input_shape),
            "hidden_sizes": list(self.hidden_sizes),
            "variant": self.variant,
            "num_blocks": int(self.num_blocks),
            "block_width": int(self.block_width) if self.block_width is not None else None,
            "post_block_activation": bool(self.post_block_activation),
            "activation": self.activation,
            "use_bias": bool(self.use_bias),
            "dropout_p": float(self.dropout_p),
            "num_classes": int(self.num_classes),
        }


@dataclass(frozen=True)
class CNN2DConfig:
    """
    Simple CNN2D architecture configuration.
    """
    input_shape: Tuple[int, int, int, int]
    conv_channels: Tuple[int, ...]
    variant: str = "plain"
    stages: int = 3
    blocks_per_stage: int = 2
    base_channels: int = 16
    channel_mult: int = 2
    downsample: str = "maxpool"
    double_conv_p: float = 0.5
    head_pool_to_1x1: bool = True
    kernel_sizes: Union[int, Tuple[int, ...]] = 3
    strides: Union[int, Tuple[int, ...]] = 1
    paddings: Union[int, Tuple[int, ...]] = 1
    activation: str = "relu"
    use_bias: bool = True
    use_maxpool: bool = False
    maxpool_kernel: int = 2
    maxpool_stride: int = 2
    num_classes: int = 10
    fc_hidden: int = 64

    def __post_init__(self):
        if len(self.input_shape) != 4:
            raise ValueError(f"CNN2DConfig.input_shape must be (1,C,H,W), got {self.input_shape}")
        if self.input_shape[0] != 1:
            raise ValueError(f"ConfigNet assumes batch=1, got input_shape={self.input_shape}")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be >= 2")
        if any(c <= 0 for c in self.conv_channels):
            raise ValueError(f"conv_channels must be positive, got {self.conv_channels}")
        if self.maxpool_kernel <= 0 or self.maxpool_stride <= 0:
            raise ValueError("maxpool params must be positive")
        if self.fc_hidden <= 0:
            raise ValueError("fc_hidden must be positive")
        if self.variant not in ("plain", "stage"):
            raise ValueError(f"CNN2DConfig.variant must be 'plain' or 'stage', got {self.variant}")
        if self.stages <= 0 or self.blocks_per_stage <= 0:
            raise ValueError("stages and blocks_per_stage must be >= 1")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if self.channel_mult <= 0:
            raise ValueError("channel_mult must be positive")
        if self.downsample not in ("maxpool", "avgpool", "stride2_conv"):
            raise ValueError("downsample must be maxpool|avgpool|stride2_conv")
        if not (0.0 <= self.double_conv_p <= 1.0):
            raise ValueError("double_conv_p must be in [0,1]")

    def to_dict(self) -> Dict[str, Any]:
        def _val(v: Union[int, Tuple[int, ...]]) -> Union[int, List[int]]:
            return int(v) if isinstance(v, int) else [int(x) for x in v]

        return {
            "input_shape": list(self.input_shape),
            "conv_channels": list(self.conv_channels),
            "variant": self.variant,
            "stages": int(self.stages),
            "blocks_per_stage": int(self.blocks_per_stage),
            "base_channels": int(self.base_channels),
            "channel_mult": int(self.channel_mult),
            "downsample": self.downsample,
            "double_conv_p": float(self.double_conv_p),
            "head_pool_to_1x1": bool(self.head_pool_to_1x1),
            "kernel_sizes": _val(self.kernel_sizes),
            "strides": _val(self.strides),
            "paddings": _val(self.paddings),
            "activation": self.activation,
            "use_bias": bool(self.use_bias),
            "use_maxpool": bool(self.use_maxpool),
            "maxpool_kernel": int(self.maxpool_kernel),
            "maxpool_stride": int(self.maxpool_stride),
            "num_classes": int(self.num_classes),
            "fc_hidden": int(self.fc_hidden),
        }



# --------------------------
# Spec Configs
# --------------------------

@dataclass(frozen=True)
class InputSpecConfig:
    """
    Input spec configuration for InputSpecLayer construction.
    """
    kind: str
    value_range: Tuple[float, float] = (0.0, 1.0)

    lb: Optional[List[float]] = None
    ub: Optional[List[float]] = None
    lb_val: Optional[float] = None
    ub_val: Optional[float] = None

    center: Optional[List[float]] = None
    center_val: Optional[float] = 0.5
    eps: Optional[float] = 0.03

    A: Optional[List[List[float]]] = None
    b: Optional[List[float]] = None
    derive_poly_from_box: bool = True

    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "value_range": list(self.value_range),
            "lb": list(self.lb) if self.lb is not None else None,
            "ub": list(self.ub) if self.ub is not None else None,
            "lb_val": self.lb_val,
            "ub_val": self.ub_val,
            "center": list(self.center) if self.center is not None else None,
            "center_val": self.center_val,
            "eps": self.eps,
            "A": self.A if self.A is not None else None,
            "b": list(self.b) if self.b is not None else None,
            "derive_poly_from_box": bool(self.derive_poly_from_box),
            "meta": _to_basic(self.meta),
        }


@dataclass(frozen=True)
class OutputSpecConfig:
    """
    Output spec configuration for OutputSpecLayer construction.
    """
    kind: str
    y_true: Optional[int] = None
    margin: float = 0.0

    c: Optional[List[float]] = None
    d: Optional[float] = None

    lb: Optional[List[float]] = None
    ub: Optional[List[float]] = None

    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "y_true": self.y_true,
            "margin": float(self.margin),
            "c": list(self.c) if self.c is not None else None,
            "d": self.d,
            "lb": list(self.lb) if self.lb is not None else None,
            "ub": list(self.ub) if self.ub is not None else None,
            "meta": _to_basic(self.meta),
        }


# --------------------------
# Instance-level containers
# --------------------------

@dataclass(frozen=True)
class InstanceSpec:
    """
    Fully specified instance: id + seed + model config + spec config.
    """
    instance_id: str
    seed: int
    family: ModelFamily
    model_cfg: Union[MLPConfig, CNN2DConfig]
    input_spec: InputSpecConfig
    output_spec: OutputSpecConfig
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "seed": int(self.seed),
            "family": self.family.value if isinstance(self.family, Enum) else str(self.family),
            "model_cfg": _to_basic(self.model_cfg),
            "input_spec": _to_basic(self.input_spec),
            "output_spec": _to_basic(self.output_spec),
            "meta": _to_basic(self.meta),
        }


# --------------------------
# Sampler configuration
# --------------------------

@dataclass(frozen=True)
class ConfigNetConfig:
    """
    Sampler configuration.
    """
    num_instances: int = 10
    base_seed: int = 0
    instance_id_prefix: str = "cfg"

    families: Tuple[ModelFamily, ...] = (ModelFamily.MLP, ModelFamily.CNN2D)
    p_mlp: float = 0.5

    num_classes_choices: Tuple[int, ...] = (10,)

    mlp_input_shapes: Tuple[Tuple[int, ...], ...] = ((1, 784), (1, 1, 28, 28))
    mlp_depth_range: Tuple[int, int] = (2, 4)
    mlp_width_choices: Tuple[int, ...] = (32, 64, 128)
    mlp_activation_choices: Tuple[str, ...] = ("relu", "tanh", "sigmoid")
    mlp_dropout_p_choices: Tuple[float, ...] = (0.0,)
    mlp_block_p: float = 0.6
    mlp_block_count_range: Tuple[int, int] = (2, 4)
    mlp_block_width_choices: Tuple[int, ...] = (32, 64, 128)
    mlp_post_block_activation_p: float = 0.8

    cnn_input_shapes: Tuple[Tuple[int, int, int, int], ...] = ((1, 1, 28, 28), (1, 3, 32, 32))
    cnn_num_blocks_range: Tuple[int, int] = (1, 3)
    cnn_channels_choices: Tuple[int, ...] = (8, 16, 32)
    cnn_kernel_choices: Tuple[int, ...] = (3,)
    cnn_stride_choices: Tuple[int, ...] = (1,)
    cnn_padding_choices: Tuple[int, ...] = (1,)
    cnn_use_maxpool_p: float = 0.3
    cnn_fc_hidden_choices: Tuple[int, ...] = (32, 64, 128)
    cnn_activation_choices: Tuple[str, ...] = ("relu", "tanh", "sigmoid")
    cnn_stage_variant_p: float = 0.5
    cnn_stages_range: Tuple[int, int] = (1, 3)
    cnn_blocks_per_stage_range: Tuple[int, int] = (1, 2)
    cnn_base_channels_choices: Tuple[int, ...] = (8, 16, 32)
    cnn_channel_mult_choices: Tuple[int, ...] = (2,)
    cnn_downsample_choices: Tuple[str, ...] = ("maxpool", "avgpool", "stride2_conv")
    cnn_double_conv_p_choices: Tuple[float, ...] = (0.5, 0.7)

    input_kind_choices: Tuple[str, ...] = ("BOX", "LINF_BALL")
    p_box: float = 0.5
    value_range_choices: Tuple[Tuple[float, float], ...] = ((0.0, 1.0),)
    eps_choices: Tuple[float, ...] = (0.03, 0.05, 0.1)

    output_kind_choices: Tuple[str, ...] = ("TOP1_ROBUST", "MARGIN_ROBUST", "LINEAR_LE", "RANGE")
    p_top1: float = 0.5
    margin_choices: Tuple[float, ...] = (0.0, 0.1, 1.0)
    output_linear_le_c_range: Tuple[float, float] = (-1.0, 1.0)
    output_linear_le_d_range: Tuple[float, float] = (-1.0, 1.0)
    output_range_choices: Tuple[Tuple[float, float], ...] = ((-1.0, 1.0),)



# --------------------------
# Sampler
# --------------------------


def _randint_inclusive(rng: random.Random, lo_hi: Tuple[int, int]) -> int:
    lo, hi = int(lo_hi[0]), int(lo_hi[1])
    if hi < lo:
        lo, hi = hi, lo
    return rng.randint(lo, hi)


def _choose(rng: random.Random, items: Sequence[Any], *, name: str) -> Any:
    if not items:
        raise ValueError(f"ConfigNetConfig.{name} must be non-empty")
    return rng.choice(list(items))


def _sample_family(rng: random.Random, cfg: ConfigNetConfig) -> ModelFamily:
    fams = list(cfg.families)
    if not fams:
        raise ValueError("ConfigNetConfig.families must be non-empty")

    if len(fams) == 1:
        return fams[0]

    has_mlp = ModelFamily.MLP in fams
    has_cnn = ModelFamily.CNN2D in fams

    if has_mlp and has_cnn:
        return ModelFamily.MLP if (rng.random() < float(cfg.p_mlp)) else ModelFamily.CNN2D

    return rng.choice(fams)


def _make_instance_id(prefix: str, base_seed: int, idx: int) -> str:
    return f"{prefix}_{int(base_seed)}_{int(idx):05d}"


def _sample_mlp(rng: random.Random, cfg: ConfigNetConfig, *, num_classes: int) -> Tuple[MLPConfig, Dict[str, Any]]:
    input_shape = _choose(rng, cfg.mlp_input_shapes, name="mlp_input_shapes")

    variant = "block" if (rng.random() < float(cfg.mlp_block_p)) else "plain"
    depth = _randint_inclusive(rng, cfg.mlp_depth_range)
    if depth <= 0:
        raise ValueError(f"mlp_depth_range produced non-positive depth={depth}")

    widths = []
    for _ in range(depth):
        widths.append(int(_choose(rng, cfg.mlp_width_choices, name="mlp_width_choices")))
    hidden_sizes = tuple(widths)

    activation = str(_choose(rng, cfg.mlp_activation_choices, name="mlp_activation_choices"))
    dropout_p = float(_choose(rng, cfg.mlp_dropout_p_choices, name="mlp_dropout_p_choices"))

    num_blocks = _randint_inclusive(rng, cfg.mlp_block_count_range)
    block_width = int(_choose(rng, cfg.mlp_block_width_choices, name="mlp_block_width_choices"))
    post_block_activation = bool(rng.random() < float(cfg.mlp_post_block_activation_p))
    if variant == "block":
        dropout_p = 0.0

    model_cfg = MLPConfig(
        input_shape=tuple(int(x) for x in input_shape),
        hidden_sizes=hidden_sizes,
        variant=variant,
        num_blocks=num_blocks,
        block_width=block_width,
        post_block_activation=post_block_activation,
        activation=activation,
        use_bias=True,
        dropout_p=dropout_p,
        num_classes=int(num_classes),
    )
    meta = {
        "depth": depth,
        "hidden_sizes": list(hidden_sizes),
        "variant": variant,
        "num_blocks": int(num_blocks),
        "block_width": int(block_width),
        "post_block_activation": bool(post_block_activation),
        "activation": activation,
        "dropout_p": dropout_p,
        "input_shape": list(model_cfg.input_shape),
        "num_classes": num_classes,
    }
    return model_cfg, meta


def _sample_cnn2d(rng: random.Random, cfg: ConfigNetConfig, *, num_classes: int) -> Tuple[CNN2DConfig, Dict[str, Any]]:
    input_shape = _choose(rng, cfg.cnn_input_shapes, name="cnn_input_shapes")
    if int(input_shape[2]) > 32 or int(input_shape[3]) > 32:
        raise ValueError(f"cnn_input_shapes must have H,W <= 32, got {input_shape}")

    variant = "stage" if (rng.random() < float(cfg.cnn_stage_variant_p)) else "plain"
    blocks = _randint_inclusive(rng, cfg.cnn_num_blocks_range)
    if blocks <= 0:
        raise ValueError(f"cnn_num_blocks_range produced non-positive blocks={blocks}")

    conv_channels: List[int] = []
    prev = int(_choose(rng, cfg.cnn_channels_choices, name="cnn_channels_choices"))
    conv_channels.append(prev)
    for _ in range(1, blocks):
        prev = int(_choose(rng, cfg.cnn_channels_choices, name="cnn_channels_choices"))
        conv_channels.append(prev)

    kernel_sizes = int(_choose(rng, cfg.cnn_kernel_choices, name="cnn_kernel_choices"))
    strides = int(_choose(rng, cfg.cnn_stride_choices, name="cnn_stride_choices"))
    paddings = int(_choose(rng, cfg.cnn_padding_choices, name="cnn_padding_choices"))
    activation = str(_choose(rng, cfg.cnn_activation_choices, name="cnn_activation_choices"))
    use_maxpool = bool(rng.random() < float(cfg.cnn_use_maxpool_p))
    fc_hidden = int(_choose(rng, cfg.cnn_fc_hidden_choices, name="cnn_fc_hidden_choices"))

    stages = _randint_inclusive(rng, cfg.cnn_stages_range)
    blocks_per_stage = _randint_inclusive(rng, cfg.cnn_blocks_per_stage_range)
    base_channels = int(_choose(rng, cfg.cnn_base_channels_choices, name="cnn_base_channels_choices"))
    channel_mult = int(_choose(rng, cfg.cnn_channel_mult_choices, name="cnn_channel_mult_choices"))
    downsample = str(_choose(rng, cfg.cnn_downsample_choices, name="cnn_downsample_choices"))
    double_conv_p = float(_choose(rng, cfg.cnn_double_conv_p_choices, name="cnn_double_conv_p_choices"))

    max_channels = base_channels * (channel_mult ** (stages - 1))
    if max_channels > 64:
        stages = max(1, min(stages, 3))
        while stages > 1 and base_channels * (channel_mult ** (stages - 1)) > 64:
            stages -= 1
        max_channels = base_channels * (channel_mult ** (stages - 1))
        if max_channels > 64:
            base_channels = min(base_channels, 64)

    model_cfg = CNN2DConfig(
        input_shape=tuple(int(x) for x in input_shape),
        conv_channels=tuple(conv_channels),
        variant=variant,
        stages=int(stages),
        blocks_per_stage=int(blocks_per_stage),
        base_channels=int(base_channels),
        channel_mult=int(channel_mult),
        downsample=str(downsample),
        double_conv_p=float(double_conv_p),
        head_pool_to_1x1=True,
        kernel_sizes=kernel_sizes,
        strides=strides,
        paddings=paddings,
        activation=activation,
        use_bias=True,
        use_maxpool=use_maxpool,
        maxpool_kernel=2,
        maxpool_stride=2,
        num_classes=int(num_classes),
        fc_hidden=int(fc_hidden),
    )
    meta = {
        "blocks": blocks,
        "conv_channels": conv_channels,
        "variant": variant,
        "stages": int(stages),
        "blocks_per_stage": int(blocks_per_stage),
        "base_channels": int(base_channels),
        "channel_mult": int(channel_mult),
        "downsample": str(downsample),
        "double_conv_p": float(double_conv_p),
        "kernel_sizes": kernel_sizes if isinstance(kernel_sizes, int) else list(kernel_sizes),
        "strides": strides if isinstance(strides, int) else list(strides),
        "paddings": paddings if isinstance(paddings, int) else list(paddings),
        "activation": activation,
        "use_maxpool": use_maxpool,
        "fc_hidden": fc_hidden,
        "input_shape": list(model_cfg.input_shape),
        "num_classes": num_classes,
    }
    return model_cfg, meta


def _sample_input_spec(
    rng: random.Random,
    cfg: ConfigNetConfig,
    *,
    input_shape: Tuple[int, ...],
) -> Tuple[InputSpecConfig, Dict[str, Any]]:
    kinds = list(cfg.input_kind_choices)
    if not kinds:
        raise ValueError("input_kind_choices must be non-empty")

    if len(kinds) == 1:
        kind = kinds[0]
    else:
        has_box = "BOX" in kinds
        has_linf = "LINF_BALL" in kinds
        if has_box and has_linf and len(kinds) == 2:
            kind = "BOX" if (rng.random() < float(cfg.p_box)) else "LINF_BALL"
        else:
            kind = rng.choice(kinds)

    value_range = _choose(rng, cfg.value_range_choices, name="value_range_choices")
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

        in_cfg = InputSpecConfig(
            kind="BOX",
            value_range=(lo, hi),
            lb_val=float(lb_val),
            ub_val=float(ub_val),
            meta={"sampled": {"shape": list(input_shape)}},
        )
        meta = {"kind": "BOX", "value_range": (lo, hi), "lb_val": float(lb_val), "ub_val": float(ub_val)}
        return in_cfg, meta

    if kind == "LINF_BALL":
        center_val = lo + (hi - lo) * rng.random()
        eps = float(_choose(rng, cfg.eps_choices, name="eps_choices"))
        eps = min(eps, 0.5 * (hi - lo)) if (hi > lo) else 0.0

        in_cfg = InputSpecConfig(
            kind="LINF_BALL",
            value_range=(lo, hi),
            center_val=float(center_val),
            eps=float(eps),
            meta={"sampled": {"shape": list(input_shape)}},
        )
        meta = {"kind": "LINF_BALL", "value_range": (lo, hi), "center_val": float(center_val), "eps": float(eps)}
        return in_cfg, meta

    if kind == "LIN_POLY":
        span = hi - lo
        shrink_a = rng.random() * 0.2
        shrink_b = rng.random() * 0.2
        lb_val = lo + span * shrink_a
        ub_val = hi - span * shrink_b
        if ub_val < lb_val:
            lb_val, ub_val = lo, hi

        in_cfg = InputSpecConfig(
            kind="LIN_POLY",
            value_range=(lo, hi),
            lb_val=float(lb_val),
            ub_val=float(ub_val),
            derive_poly_from_box=True,
            meta={"sampled": {"shape": list(input_shape)}},
        )
        meta = {"kind": "LIN_POLY", "value_range": (lo, hi), "lb_val": float(lb_val), "ub_val": float(ub_val)}
        return in_cfg, meta

    in_cfg = InputSpecConfig(
        kind="BOX",
        value_range=(lo, hi),
        lb_val=lo,
        ub_val=hi,
        meta={"fallback": True, "reason": f"unsupported kind={kind}"},
    )
    meta = {"kind": "FALLBACK_BOX", "value_range": (lo, hi), "lb_val": lo, "ub_val": hi}
    return in_cfg, meta


def _sample_output_spec(
    rng: random.Random,
    cfg: ConfigNetConfig,
    *,
    num_classes: int,
) -> Tuple[OutputSpecConfig, Dict[str, Any]]:
    kinds = list(cfg.output_kind_choices)
    if not kinds:
        raise ValueError("output_kind_choices must be non-empty")

    if len(kinds) == 1:
        kind = kinds[0]
    else:
        has_top1 = "TOP1_ROBUST" in kinds
        has_margin = "MARGIN_ROBUST" in kinds
        if has_top1 and has_margin and len(kinds) == 2:
            kind = "TOP1_ROBUST" if (rng.random() < float(cfg.p_top1)) else "MARGIN_ROBUST"
        else:
            kind = rng.choice(kinds)

    y_true = int(rng.randrange(int(num_classes)))

    if kind == "TOP1_ROBUST":
        out_cfg = OutputSpecConfig(
            kind="TOP1_ROBUST",
            y_true=y_true,
            margin=0.0,
            meta={},
        )
        meta = {"kind": "TOP1_ROBUST", "y_true": y_true, "margin": 0.0}
        return out_cfg, meta

    if kind == "MARGIN_ROBUST":
        margin = float(_choose(rng, cfg.margin_choices, name="margin_choices"))
        out_cfg = OutputSpecConfig(
            kind="MARGIN_ROBUST",
            y_true=y_true,
            margin=float(margin),
            meta={},
        )
        meta = {"kind": "MARGIN_ROBUST", "y_true": y_true, "margin": float(margin)}
        return out_cfg, meta

    if kind == "LINEAR_LE":
        c_lo, c_hi = cfg.output_linear_le_c_range
        d_lo, d_hi = cfg.output_linear_le_d_range
        c_vals = [c_lo + (c_hi - c_lo) * rng.random() for _ in range(int(num_classes))]
        d_val = d_lo + (d_hi - d_lo) * rng.random()
        out_cfg = OutputSpecConfig(
            kind="LINEAR_LE",
            c=[float(x) for x in c_vals],
            d=float(d_val),
            meta={},
        )
        meta = {"kind": "LINEAR_LE", "c": list(c_vals), "d": float(d_val)}
        return out_cfg, meta

    if kind == "RANGE":
        lo, hi = _choose(rng, cfg.output_range_choices, name="output_range_choices")
        lb_vals = []
        ub_vals = []
        for _ in range(int(num_classes)):
            a = lo + (hi - lo) * rng.random()
            b = lo + (hi - lo) * rng.random()
            lb_vals.append(min(a, b))
            ub_vals.append(max(a, b))
        out_cfg = OutputSpecConfig(
            kind="RANGE",
            lb=[float(x) for x in lb_vals],
            ub=[float(x) for x in ub_vals],
            meta={},
        )
        meta = {"kind": "RANGE", "lb": list(lb_vals), "ub": list(ub_vals)}
        return out_cfg, meta

    out_cfg = OutputSpecConfig(kind="TOP1_ROBUST", y_true=y_true, margin=0.0, meta={"fallback": True})
    meta = {"kind": "FALLBACK_TOP1_ROBUST", "y_true": y_true, "margin": 0.0}
    return out_cfg, meta


def sample_instances(cfg: ConfigNetConfig) -> List[InstanceSpec]:
    """
    Determinism contract:
      - instance_id = f"{prefix}_{base_seed}_{idx:05d}"
      - seed = derive_seed(base_seed, idx, instance_id)  (stable u32)
      - per-instance RNG = random.Random(seed)
    """
    base_seed = int(cfg.base_seed)
    num_instances = int(cfg.num_instances)
    prefix = str(cfg.instance_id_prefix)

    if num_instances < 0:
        raise ValueError(f"num_instances must be >= 0, got {num_instances}")

    out: List[InstanceSpec] = []

    for idx in range(num_instances):
        instance_id = _make_instance_id(prefix, base_seed, idx)
        seed = int(derive_seed(base_seed, idx, instance_id))
        rng = random.Random(seed)
        rng_family = random.Random(derive_seed(seed, 0, "family"))
        rng_model = random.Random(derive_seed(seed, 0, "model"))
        rng_spec = random.Random(derive_seed(seed, 0, "spec"))

        family = _sample_family(rng_family, cfg)
        num_classes = int(_choose(rng_family, cfg.num_classes_choices, name="num_classes_choices"))

        if family == ModelFamily.MLP:
            model_cfg, model_meta = _sample_mlp(rng_model, cfg, num_classes=num_classes)
            input_shape = tuple(model_cfg.input_shape)
        elif family == ModelFamily.CNN2D:
            model_cfg, model_meta = _sample_cnn2d(rng_model, cfg, num_classes=num_classes)
            input_shape = tuple(model_cfg.input_shape)
        else:
            raise ValueError(f"Unknown family: {family}")

        input_spec, in_meta = _sample_input_spec(rng_spec, cfg, input_shape=input_shape)
        output_spec, out_meta = _sample_output_spec(rng_spec, cfg, num_classes=num_classes)

        meta: Dict[str, Any] = {
            "base_seed": base_seed,
            "idx": idx,
            "seed": seed,
            "model": model_meta,
            "input_spec": in_meta,
            "output_spec": out_meta,
        }

        inst = InstanceSpec(
            instance_id=instance_id,
            seed=seed,
            family=family,
            model_cfg=model_cfg,
            input_spec=input_spec,
            output_spec=output_spec,
            meta=meta,
        )
        out.append(inst)

    return out
