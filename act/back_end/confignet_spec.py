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
import os
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from act.front_end.specs import InKind, OutKind

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

logger = logging.getLogger(__name__)


# --------------------------
# Seeds / reproducibility
# --------------------------

def seed_everything(seed: int, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: base seed
        deterministic: if True, request deterministic algorithms (can reduce performance).
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed)}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextmanager
def seeded(seed: int, deterministic: bool = False):
    """Seed everything, then restore RNG/global states after."""
    old_py = random.getstate()
    old_hash = os.environ.get("PYTHONHASHSEED", None)

    old_np = None
    if np is not None:
        old_np = np.random.get_state()

    old_torch = torch.get_rng_state()
    old_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    old_det = torch.are_deterministic_algorithms_enabled()
    old_cudnn_det = torch.backends.cudnn.deterministic
    old_cudnn_bench = torch.backends.cudnn.benchmark

    try:
        seed_everything(seed, deterministic=deterministic)
        yield
    finally:
        random.setstate(old_py)
        if old_hash is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = old_hash

        if np is not None and old_np is not None:
            np.random.set_state(old_np)

        torch.set_rng_state(old_torch)
        if old_cuda is not None:
            torch.cuda.set_rng_state_all(old_cuda)

        torch.use_deterministic_algorithms(old_det, warn_only=True)
        torch.backends.cudnn.deterministic = old_cudnn_det
        torch.backends.cudnn.benchmark = old_cudnn_bench


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
    TEMPLATE = "template"


def _tensor_to_list(t: torch.Tensor) -> Any:
    return t.detach().cpu().tolist()


def _json_safe(obj: Any, *, strict: bool = False) -> Any:
    """Convert objects to JSON-safe values for auditing."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if torch.is_tensor(obj):
        return _tensor_to_list(obj)
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x, strict=strict) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, strict=strict) for k, v in obj.items()}
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return _json_safe(obj.to_dict(), strict=strict)
    if is_dataclass(obj):
        return _json_safe(asdict(obj), strict=strict)
    if strict:
        raise TypeError(f"Object is not JSON-serializable: {type(obj)}")
    return str(obj)


def _enum_value(v: Any) -> Any:
    return v.value if isinstance(v, Enum) else v


@dataclass(frozen=True)
class MLPConfig:
    """
    MLP architecture configuration.
    """
    input_shape: Tuple[int, ...]
    hidden_sizes: Tuple[int, ...]
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_shape": list(self.input_shape),
            "hidden_sizes": list(self.hidden_sizes),
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

    def to_dict(self) -> Dict[str, Any]:
        def _val(v: Union[int, Tuple[int, ...]]) -> Union[int, List[int]]:
            return int(v) if isinstance(v, int) else [int(x) for x in v]

        return {
            "input_shape": list(self.input_shape),
            "conv_channels": list(self.conv_channels),
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


@dataclass(frozen=True)
class TemplateConfig:
    """
    Template network configuration (built from ACT examples).
    """
    template_name: str
    dataset: Optional[str] = None
    activation: Optional[str] = None
    input_shape: Optional[Tuple[int, ...]] = None
    num_classes: Optional[int] = None
    overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_name": str(self.template_name),
            "dataset": self.dataset,
            "activation": self.activation,
            "input_shape": list(self.input_shape) if self.input_shape is not None else None,
            "num_classes": int(self.num_classes) if self.num_classes is not None else None,
            "overrides": _json_safe(self.overrides),
        }


# --------------------------
# Spec Configs
# --------------------------

@dataclass(frozen=True)
class InputSpecConfig:
    """
    Input spec configuration for InputSpecLayer construction.
    """
    kind: InKind
    value_range: Tuple[float, float] = (0.0, 1.0)

    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    lb_val: Optional[float] = None
    ub_val: Optional[float] = None

    center: Optional[torch.Tensor] = None
    center_val: Optional[float] = 0.5
    eps: Optional[float] = 0.03

    A: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None
    derive_poly_from_box: bool = True

    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": _enum_value(self.kind),
            "value_range": list(self.value_range),
            "lb": _tensor_to_list(self.lb) if self.lb is not None else None,
            "ub": _tensor_to_list(self.ub) if self.ub is not None else None,
            "lb_val": self.lb_val,
            "ub_val": self.ub_val,
            "center": _tensor_to_list(self.center) if self.center is not None else None,
            "center_val": self.center_val,
            "eps": self.eps,
            "A": _tensor_to_list(self.A) if self.A is not None else None,
            "b": _tensor_to_list(self.b) if self.b is not None else None,
            "derive_poly_from_box": bool(self.derive_poly_from_box),
            "meta": _json_safe(self.meta),
        }


@dataclass(frozen=True)
class OutputSpecConfig:
    """
    Output spec configuration for OutputSpecLayer construction.
    """
    kind: OutKind
    y_true: Optional[int] = None
    margin: float = 0.0

    c: Optional[torch.Tensor] = None
    d: Optional[float] = None

    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None

    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": _enum_value(self.kind),
            "y_true": self.y_true,
            "margin": float(self.margin),
            "c": _tensor_to_list(self.c) if self.c is not None else None,
            "d": self.d,
            "lb": _tensor_to_list(self.lb) if self.lb is not None else None,
            "ub": _tensor_to_list(self.ub) if self.ub is not None else None,
            "meta": _json_safe(self.meta),
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
    model_cfg: Union[MLPConfig, CNN2DConfig, TemplateConfig]
    input_spec: InputSpecConfig
    output_spec: OutputSpecConfig
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "seed": int(self.seed),
            "family": self.family.value if isinstance(self.family, Enum) else str(self.family),
            "model_cfg": _json_safe(self.model_cfg),
            "input_spec": _json_safe(self.input_spec),
            "output_spec": _json_safe(self.output_spec),
            "meta": _json_safe(self.meta),
        }


@dataclass
class GeneratedInstance:
    """
    Generated instance bundle: spec + wrapped torch model.
    """
    instance_spec: InstanceSpec
    wrapped_model: torch.nn.Module


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

    cnn_input_shapes: Tuple[Tuple[int, int, int, int], ...] = ((1, 1, 28, 28), (1, 3, 32, 32))
    cnn_num_blocks_range: Tuple[int, int] = (1, 3)
    cnn_channels_choices: Tuple[int, ...] = (8, 16, 32)
    cnn_kernel_choices: Tuple[int, ...] = (3,)
    cnn_stride_choices: Tuple[int, ...] = (1,)
    cnn_padding_choices: Tuple[int, ...] = (1,)
    cnn_use_maxpool_p: float = 0.3
    cnn_fc_hidden_choices: Tuple[int, ...] = (32, 64, 128)
    cnn_activation_choices: Tuple[str, ...] = ("relu", "tanh", "sigmoid")

    input_kind_choices: Tuple[InKind, ...] = (InKind.BOX, InKind.LINF_BALL)
    p_box: float = 0.5
    value_range_choices: Tuple[Tuple[float, float], ...] = ((0.0, 1.0),)
    eps_choices: Tuple[float, ...] = (0.03, 0.05, 0.1)

    output_kind_choices: Tuple[OutKind, ...] = (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST, OutKind.LINEAR_LE, OutKind.RANGE)
    p_top1: float = 0.5
    margin_choices: Tuple[float, ...] = (0.0, 0.1, 1.0)
    output_linear_le_c_range: Tuple[float, float] = (-1.0, 1.0)
    output_linear_le_d_range: Tuple[float, float] = (-1.0, 1.0)
    output_range_choices: Tuple[Tuple[float, float], ...] = ((-1.0, 1.0),)

    template_names: Optional[Tuple[str, ...]] = None


# --------------------------
# Sampler
# --------------------------

_TEMPLATE_FACTORY = None


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    return str(obj)


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


def _get_template_factory():
    global _TEMPLATE_FACTORY
    if _TEMPLATE_FACTORY is None:
        from act.pipeline.verification.model_factory import ModelFactory
        _TEMPLATE_FACTORY = ModelFactory()
    return _TEMPLATE_FACTORY


def _resolve_template_catalog(cfg: ConfigNetConfig) -> Dict[str, Any]:
    factory = _get_template_factory()
    names = list(factory.config.get("networks", {}).keys())
    if cfg.template_names:
        names = [n for n in cfg.template_names if n in names] or list(cfg.template_names)
    if not names:
        raise ValueError("No template networks available for ModelFamily.TEMPLATE sampling")
    return {"factory": factory, "names": names}


def _get_template_info(name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    input_shape = spec.get("input_shape", None)
    dataset = None
    num_classes = None
    activation = None

    if input_shape is None:
        for layer in spec.get("layers", []):
            if layer.get("kind") == "INPUT":
                meta = layer.get("meta", {})
                input_shape = meta.get("shape", None)
                dataset = meta.get("dataset_name", None)
                num_classes = meta.get("num_classes", None)
                break
    else:
        input_shape = list(input_shape)
        for layer in spec.get("layers", []):
            if layer.get("kind") == "INPUT":
                meta = layer.get("meta", {})
                dataset = meta.get("dataset_name", None)
                num_classes = meta.get("num_classes", None)
                break

    return {
        "template_name": name,
        "input_shape": tuple(int(x) for x in input_shape) if input_shape is not None else None,
        "dataset": dataset,
        "num_classes": int(num_classes) if num_classes is not None else None,
        "activation": activation,
    }


def _build_template_overrides(input_spec: InputSpecConfig, output_spec: OutputSpecConfig) -> Dict[str, Any]:
    eps = 0.0
    if input_spec.kind == InKind.LINF_BALL and input_spec.eps is not None:
        eps = float(input_spec.eps)
    return {
        "eps": float(eps),
        "norm": "inf",
        "assert_kind": output_spec.kind.value,
        "y_true": int(output_spec.y_true if output_spec.y_true is not None else 0),
        "targeted": False,
        "target_label": None,
    }


def _make_instance_id(prefix: str, base_seed: int, idx: int) -> str:
    return f"{prefix}_{int(base_seed)}_{int(idx):05d}"


def _sample_mlp(rng: random.Random, cfg: ConfigNetConfig, *, num_classes: int) -> Tuple[MLPConfig, Dict[str, Any]]:
    input_shape = _choose(rng, cfg.mlp_input_shapes, name="mlp_input_shapes")

    depth = _randint_inclusive(rng, cfg.mlp_depth_range)
    if depth <= 0:
        raise ValueError(f"mlp_depth_range produced non-positive depth={depth}")

    widths = []
    for _ in range(depth):
        widths.append(int(_choose(rng, cfg.mlp_width_choices, name="mlp_width_choices")))
    hidden_sizes = tuple(widths)

    activation = str(_choose(rng, cfg.mlp_activation_choices, name="mlp_activation_choices"))
    dropout_p = float(_choose(rng, cfg.mlp_dropout_p_choices, name="mlp_dropout_p_choices"))

    model_cfg = MLPConfig(
        input_shape=tuple(int(x) for x in input_shape),
        hidden_sizes=hidden_sizes,
        activation=activation,
        use_bias=True,
        dropout_p=dropout_p,
        num_classes=int(num_classes),
    )
    meta = {
        "depth": depth,
        "hidden_sizes": list(hidden_sizes),
        "activation": activation,
        "dropout_p": dropout_p,
        "input_shape": list(model_cfg.input_shape),
        "num_classes": num_classes,
    }
    return model_cfg, meta


def _sample_cnn2d(rng: random.Random, cfg: ConfigNetConfig, *, num_classes: int) -> Tuple[CNN2DConfig, Dict[str, Any]]:
    input_shape = _choose(rng, cfg.cnn_input_shapes, name="cnn_input_shapes")

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

    model_cfg = CNN2DConfig(
        input_shape=tuple(int(x) for x in input_shape),
        conv_channels=tuple(conv_channels),
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
        has_box = InKind.BOX in kinds
        has_linf = InKind.LINF_BALL in kinds
        if has_box and has_linf and len(kinds) == 2:
            kind = InKind.BOX if (rng.random() < float(cfg.p_box)) else InKind.LINF_BALL
        else:
            kind = rng.choice(kinds)

    value_range = _choose(rng, cfg.value_range_choices, name="value_range_choices")
    lo, hi = float(value_range[0]), float(value_range[1])
    if hi < lo:
        lo, hi = hi, lo

    if kind == InKind.BOX:
        span = hi - lo
        shrink_a = rng.random() * 0.2
        shrink_b = rng.random() * 0.2
        lb_val = lo + span * shrink_a
        ub_val = hi - span * shrink_b
        if ub_val < lb_val:
            lb_val, ub_val = lo, hi

        in_cfg = InputSpecConfig(
            kind=InKind.BOX,
            value_range=(lo, hi),
            lb_val=float(lb_val),
            ub_val=float(ub_val),
            meta={"sampled": {"shape": list(input_shape)}},
        )
        meta = {"kind": "BOX", "value_range": (lo, hi), "lb_val": float(lb_val), "ub_val": float(ub_val)}
        return in_cfg, meta

    if kind == InKind.LINF_BALL:
        center_val = lo + (hi - lo) * rng.random()
        eps = float(_choose(rng, cfg.eps_choices, name="eps_choices"))
        eps = min(eps, 0.5 * (hi - lo)) if (hi > lo) else 0.0

        in_cfg = InputSpecConfig(
            kind=InKind.LINF_BALL,
            value_range=(lo, hi),
            center_val=float(center_val),
            eps=float(eps),
            meta={"sampled": {"shape": list(input_shape)}},
        )
        meta = {"kind": "LINF_BALL", "value_range": (lo, hi), "center_val": float(center_val), "eps": float(eps)}
        return in_cfg, meta

    if kind == InKind.LIN_POLY:
        span = hi - lo
        shrink_a = rng.random() * 0.2
        shrink_b = rng.random() * 0.2
        lb_val = lo + span * shrink_a
        ub_val = hi - span * shrink_b
        if ub_val < lb_val:
            lb_val, ub_val = lo, hi

        in_cfg = InputSpecConfig(
            kind=InKind.LIN_POLY,
            value_range=(lo, hi),
            lb_val=float(lb_val),
            ub_val=float(ub_val),
            derive_poly_from_box=True,
            meta={"sampled": {"shape": list(input_shape)}},
        )
        meta = {"kind": "LIN_POLY", "value_range": (lo, hi), "lb_val": float(lb_val), "ub_val": float(ub_val)}
        return in_cfg, meta

    in_cfg = InputSpecConfig(
        kind=InKind.BOX,
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
        has_top1 = OutKind.TOP1_ROBUST in kinds
        has_margin = OutKind.MARGIN_ROBUST in kinds
        if has_top1 and has_margin and len(kinds) == 2:
            kind = OutKind.TOP1_ROBUST if (rng.random() < float(cfg.p_top1)) else OutKind.MARGIN_ROBUST
        else:
            kind = rng.choice(kinds)

    y_true = int(rng.randrange(int(num_classes)))

    if kind == OutKind.TOP1_ROBUST:
        out_cfg = OutputSpecConfig(
            kind=OutKind.TOP1_ROBUST,
            y_true=y_true,
            margin=0.0,
            meta={},
        )
        meta = {"kind": "TOP1_ROBUST", "y_true": y_true, "margin": 0.0}
        return out_cfg, meta

    if kind == OutKind.MARGIN_ROBUST:
        margin = float(_choose(rng, cfg.margin_choices, name="margin_choices"))
        out_cfg = OutputSpecConfig(
            kind=OutKind.MARGIN_ROBUST,
            y_true=y_true,
            margin=float(margin),
            meta={},
        )
        meta = {"kind": "MARGIN_ROBUST", "y_true": y_true, "margin": float(margin)}
        return out_cfg, meta

    if kind == OutKind.LINEAR_LE:
        c_lo, c_hi = cfg.output_linear_le_c_range
        d_lo, d_hi = cfg.output_linear_le_d_range
        c_vals = [c_lo + (c_hi - c_lo) * rng.random() for _ in range(int(num_classes))]
        d_val = d_lo + (d_hi - d_lo) * rng.random()
        c = torch.tensor(c_vals, dtype=torch.float32)
        out_cfg = OutputSpecConfig(
            kind=OutKind.LINEAR_LE,
            c=c,
            d=float(d_val),
            meta={},
        )
        meta = {"kind": "LINEAR_LE", "c": list(c_vals), "d": float(d_val)}
        return out_cfg, meta

    if kind == OutKind.RANGE:
        lo, hi = _choose(rng, cfg.output_range_choices, name="output_range_choices")
        lb_vals = []
        ub_vals = []
        for _ in range(int(num_classes)):
            a = lo + (hi - lo) * rng.random()
            b = lo + (hi - lo) * rng.random()
            lb_vals.append(min(a, b))
            ub_vals.append(max(a, b))
        lb = torch.tensor(lb_vals, dtype=torch.float32)
        ub = torch.tensor(ub_vals, dtype=torch.float32)
        out_cfg = OutputSpecConfig(
            kind=OutKind.RANGE,
            lb=lb,
            ub=ub,
            meta={},
        )
        meta = {"kind": "RANGE", "lb": list(lb_vals), "ub": list(ub_vals)}
        return out_cfg, meta

    out_cfg = OutputSpecConfig(kind=OutKind.TOP1_ROBUST, y_true=y_true, margin=0.0, meta={"fallback": True})
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
        elif family == ModelFamily.TEMPLATE:
            catalog = _resolve_template_catalog(cfg)
            name = str(rng_model.choice(catalog["names"]))
            spec = catalog["factory"].config["networks"].get(name, {})
            info = _get_template_info(name, spec)
            if info["input_shape"] is None:
                raise ValueError(f"Template '{name}' is missing input_shape metadata")
            if info["num_classes"] is None:
                info["num_classes"] = int(_choose(rng, cfg.num_classes_choices, name="num_classes_choices"))
            input_shape = tuple(info["input_shape"])
            num_classes = int(info["num_classes"])
            model_cfg = TemplateConfig(
                template_name=info["template_name"],
                dataset=info["dataset"],
                activation=info["activation"],
                input_shape=input_shape,
                num_classes=num_classes,
                overrides={},
            )
            model_meta = {
                "template_name": info["template_name"],
                "input_shape": list(input_shape),
                "num_classes": num_classes,
                "dataset": info["dataset"],
            }
        else:
            raise ValueError(f"Unknown family: {family}")

        input_spec, in_meta = _sample_input_spec(rng_spec, cfg, input_shape=input_shape)
        output_spec, out_meta = _sample_output_spec(rng_spec, cfg, num_classes=num_classes)

        if family == ModelFamily.TEMPLATE:
            overrides = _build_template_overrides(input_spec, output_spec)
            model_cfg = TemplateConfig(
                template_name=model_cfg.template_name,
                dataset=model_cfg.dataset,
                activation=model_cfg.activation,
                input_shape=model_cfg.input_shape,
                num_classes=model_cfg.num_classes,
                overrides=overrides,
            )
            model_meta = dict(model_meta)
            model_meta["overrides"] = _to_plain(overrides)

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


# --------------------------
# Input sampling
# --------------------------

def _resolve_input_shape(instance_spec: InstanceSpec) -> Tuple[int, ...]:
    model_cfg = instance_spec.model_cfg
    if not hasattr(model_cfg, "input_shape"):
        raise ValueError("instance_spec.model_cfg missing input_shape")
    shape = tuple(int(x) for x in getattr(model_cfg, "input_shape"))
    if len(shape) < 2:
        raise ValueError(f"input_shape must include batch dim, got {shape}")
    if shape[0] != 1:
        raise ValueError(f"ConfigNet assumes batch=1, got input_shape={shape}")
    return shape


def _prod(shape: Tuple[int, ...]) -> int:
    p = 1
    for s in shape:
        p *= int(s)
    return p