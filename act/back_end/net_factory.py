#===- act/back_end/net_factory.py - Generator-Driven Network Factory -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Random network generator for ACT Net JSONs driven by a generator config
#   (config_gen_act_net.yaml). Removes runtime dependency on examples_config.yaml.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import copy
import hashlib
import json
import random
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from act.back_end.core import Layer, Net
from act.back_end.serialization.serialization import NetSerializer
from act.util.device_manager import get_default_dtype

DEFAULT_GEN_CONFIG = "act/back_end/examples/config_gen_act_net.yaml"
DEFAULT_NETS_DIR = "act/back_end/examples/nets"
DEFAULT_NAME_PREFIX = "cfg_seed"

_DEFAULT_CONFIG: Dict[str, Any] = {
    "generator": {
        "num_instances": 5,
        "base_seed": None,
        "name_prefix": DEFAULT_NAME_PREFIX,
        "output_dir": DEFAULT_NETS_DIR,
        "write_manifest": True,
        "manifest_path": None,
        "families": ["mlp", "cnn2d"],
        "p_mlp": 0.5,
        "num_classes_choices": [10],
        "dtype": "torch.float64",
    },
    "mlp": {
        "input_shapes": [[1, 784], [1, 1, 28, 28]],
        "depth_range": [2, 4],
        "width_choices": [32, 64, 128],
        "activation_choices": ["relu", "tanh", "sigmoid"],
        "dropout_p_choices": [0.0],
        "block_p": 0.6,
        "block_count_range": [2, 6],
        "block_width_choices": [32, 64, 128],
        "post_block_activation_p": 0.8,
        "residual_p": 0.2,
        "residual_blocks_range": [1, 3],
        "residual_width_choices": [32, 64, 128],
    },
    "cnn": {
        "input_shapes": [[1, 1, 28, 28], [1, 3, 32, 32]],
        "num_blocks_range": [1, 3],
        "channels_choices": [8, 16, 32],
        "kernel_choices": [3],
        "stride_choices": [1],
        "padding_choices": [1],
        "use_maxpool_p": 0.3,
        "fc_hidden_choices": [32, 64, 128],
        "activation_choices": ["relu", "tanh", "sigmoid"],
        "stage_variant_p": 0.5,
        "stages_range": [1, 3],
        "blocks_per_stage_range": [1, 2],
        "base_channels_choices": [8, 16, 32],
        "channel_mult_choices": [2],
        "downsample_choices": ["maxpool", "avgpool", "stride2_conv"],
        "double_conv_p_choices": [0.5, 0.7],
    },
    "input_spec": {
        "kind_choices": ["BOX", "LINF_BALL"],
        "p_box": 0.5,
        "value_range_choices": [[0.0, 1.0]],
        "eps_choices": [0.03, 0.05, 0.1],
    },
    "output_spec": {
        "kind_choices": ["TOP1_ROBUST", "MARGIN_ROBUST", "LINEAR_LE", "RANGE"],
        "p_top1": 0.5,
        "margin_choices": [0.0, 0.1, 1.0],
        "linear_le_c_range": [-1.0, 1.0],
        "linear_le_d_range": [-1.0, 1.0],
        "range_choices": [[-1.0, 1.0]],
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_gen_config(path: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    cfg_path = Path(path)
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return _deep_merge(_DEFAULT_CONFIG, data)


def _stable_u32_from_bytes(data: bytes) -> int:
    return int.from_bytes(data[:4], byteorder="little", signed=False)


def _derive_seed(base_seed: int, idx: int, instance_id: str) -> int:
    payload = f"{base_seed}|{idx}|{instance_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return _stable_u32_from_bytes(digest)


def _randint_inclusive(rng: random.Random, lo_hi: List[int]) -> int:
    lo, hi = int(lo_hi[0]), int(lo_hi[1])
    if hi < lo:
        lo, hi = hi, lo
    return rng.randint(lo, hi)


def _choose(rng: random.Random, items: List[Any], *, name: str) -> Any:
    if not items:
        raise ValueError(f"Config.{name} must be non-empty")
    return rng.choice(list(items))


def _prod(shape: Tuple[int, ...]) -> int:
    p = 1
    for s in shape:
        p *= int(s)
    return p


def _ensure_batch1(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(shape) < 2:
        raise ValueError(f"input_shape must include batch dim, got {shape}")
    if int(shape[0]) != 1:
        raise ValueError(f"Generator assumes batch=1, got {shape}")
    return tuple(int(x) for x in shape)


def _activation_kind(name: str) -> str:
    name = (name or "relu").lower()
    if name == "relu":
        return "RELU"
    if name == "tanh":
        return "TANH"
    if name == "sigmoid":
        return "SIGMOID"
    raise ValueError(f"Unsupported activation '{name}'")


def _infer_conv2d_output_hw(
    H: int,
    W: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> Tuple[int, int]:
    def out_dim(x: int) -> int:
        return int((x + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1)

    return out_dim(H), out_dim(W)


def _infer_pool2d_output_hw(
    H: int,
    W: int,
    kernel: int,
    stride: int,
    padding: int = 0,
) -> Tuple[int, int]:
    def out_dim(x: int) -> int:
        return int((x + 2 * padding - (kernel - 1) - 1) // stride + 1)

    return out_dim(H), out_dim(W)


def _as_block_param(v: Any, i: int, n_blocks: int, name: str) -> int:
    if isinstance(v, int):
        return int(v)
    t = tuple(int(x) for x in v)
    if len(t) == 1:
        return int(t[0])
    if len(t) == n_blocks:
        return int(t[i])
    raise ValueError(f"{name} must be int or tuple of len 1 or len {n_blocks}, got len={len(t)}")


def _append_conv2d(
    layers: List[Dict[str, Any]],
    *,
    in_ch: int,
    out_ch: int,
    H: int,
    W: int,
    kernel: int,
    stride: int,
    padding: int,
) -> Tuple[int, int]:
    input_shape = [1, int(in_ch), int(H), int(W)]
    out_H, out_W = _infer_conv2d_output_hw(H, W, kernel=kernel, stride=stride, padding=padding)
    if out_H <= 0 or out_W <= 0:
        raise ValueError(f"Invalid CONV2D output shape: H={out_H}, W={out_W}")
    output_shape = [1, int(out_ch), int(out_H), int(out_W)]
    layers.append(
        {
            "kind": "CONV2D",
            "params": {},
            "meta": {
                "in_channels": int(in_ch),
                "out_channels": int(out_ch),
                "kernel_size": int(kernel),
                "stride": int(stride),
                "padding": int(padding),
                "input_shape": input_shape,
                "output_shape": output_shape,
            },
        }
    )
    return out_H, out_W


def _append_pool2d(
    layers: List[Dict[str, Any]],
    *,
    kind: str,
    in_ch: int,
    H: int,
    W: int,
    kernel: int,
    stride: int,
    padding: int = 0,
) -> Tuple[int, int]:
    input_shape = [1, int(in_ch), int(H), int(W)]
    out_H, out_W = _infer_pool2d_output_hw(H, W, kernel=kernel, stride=stride, padding=padding)
    if out_H <= 0 or out_W <= 0:
        raise ValueError(f"Invalid {kind} output shape: H={out_H}, W={out_W}")
    output_shape = [1, int(in_ch), int(out_H), int(out_W)]
    layers.append(
        {
            "kind": kind,
            "params": {},
            "meta": {
                "kernel_size": int(kernel),
                "stride": int(stride),
                "padding": int(padding),
                "input_shape": input_shape,
                "output_shape": output_shape,
            },
        }
    )
    return out_H, out_W


    def _build_mlp_layers(layers: List[Dict[str, Any]], *, cfg: Dict[str, Any]) -> None:
        shape = _ensure_batch1(tuple(cfg["input_shape"]))
        in_features = int(shape[1]) if len(shape) == 2 else _prod(shape[1:])

        if len(shape) > 2:
            layers.append({"kind": "FLATTEN", "params": {}, "meta": {"start_dim": 1}})

        act_kind = _activation_kind(cfg.get("activation", "relu"))

        variant = cfg.get("variant", "plain")
        if variant == "plain":
            for h in cfg["hidden_sizes"]:
                layers.append(
                    {
                        "kind": "DENSE",
                        "params": {},
                    "meta": {
                        "in_features": int(in_features),
                        "out_features": int(h),
                        "bias_enabled": bool(cfg.get("use_bias", True)),
                    },
                }
            )
            layers.append({"kind": act_kind, "params": {}, "meta": {}})
            in_features = int(h)
    else:
        width = int(cfg.get("block_width") or (cfg["hidden_sizes"][0] if cfg["hidden_sizes"] else 64))
        layers.append(
            {
                "kind": "DENSE",
                "params": {},
                "meta": {
                    "in_features": int(in_features),
                    "out_features": int(width),
                    "bias_enabled": bool(cfg.get("use_bias", True)),
                },
            }
        )
        layers.append({"kind": act_kind, "params": {}, "meta": {}})
        in_features = int(width)

        for _ in range(int(cfg.get("num_blocks", 1))):
            layers.append(
                {
                    "kind": "DENSE",
                    "params": {},
                    "meta": {
                        "in_features": int(in_features),
                        "out_features": int(in_features),
                        "bias_enabled": bool(cfg.get("use_bias", True)),
                    },
                }
            )
            layers.append({"kind": act_kind, "params": {}, "meta": {}})
            layers.append(
                {
                    "kind": "DENSE",
                    "params": {},
                    "meta": {
                        "in_features": int(in_features),
                        "out_features": int(in_features),
                        "bias_enabled": bool(cfg.get("use_bias", True)),
                    },
                }
            )
            if cfg.get("post_block_activation", True):
                layers.append({"kind": act_kind, "params": {}, "meta": {}})
        elif variant == "residual":
            width = int(cfg.get("residual_width") or (cfg["hidden_sizes"][0] if cfg["hidden_sizes"] else in_features))
            if in_features != width:
                layers.append(
                    {
                        "kind": "DENSE",
                        "params": {},
                        "meta": {
                            "in_features": int(in_features),
                            "out_features": int(width),
                            "bias_enabled": bool(cfg.get("use_bias", True)),
                        },
                    }
                )
                layers.append({"kind": act_kind, "params": {}, "meta": {}})
                in_features = int(width)

            num_blocks = int(cfg.get("num_residual_blocks", cfg.get("num_blocks", 1)))
            for _ in range(num_blocks):
                skip_idx = len(layers) - 1
                layers.append(
                    {
                        "kind": "DENSE",
                        "params": {},
                        "meta": {
                            "in_features": int(in_features),
                            "out_features": int(in_features),
                            "bias_enabled": bool(cfg.get("use_bias", True)),
                        },
                    }
                )
                layers.append({"kind": act_kind, "params": {}, "meta": {}})
                layers.append(
                    {
                        "kind": "DENSE",
                        "params": {},
                        "meta": {
                            "in_features": int(in_features),
                            "out_features": int(in_features),
                            "bias_enabled": bool(cfg.get("use_bias", True)),
                        },
                    }
                )
                main_idx = len(layers) - 1
                layers.append(
                    {
                        "kind": "ADD",
                        "params": {},
                        "meta": {},
                        "inputs": {"x": skip_idx, "y": main_idx},
                        "preds": [skip_idx, main_idx],
                    }
                )
                layers.append({"kind": act_kind, "params": {}, "meta": {}})
        else:
            raise ValueError(f"Unsupported MLP variant '{variant}'")

        layers.append(
            {
                "kind": "DENSE",
                "params": {},
            "meta": {
                "in_features": int(in_features),
                "out_features": int(cfg["num_classes"]),
                "bias_enabled": True,
            },
        }
    )


def _build_cnn_layers(
    layers: List[Dict[str, Any]],
    *,
    cfg: Dict[str, Any],
    rng: random.Random,
) -> None:
    shape = _ensure_batch1(tuple(cfg["input_shape"]))
    if len(shape) != 4:
        raise ValueError(f"CNN2D expects input_shape=(1,C,H,W), got {shape}")
    _, in_ch, H, W = shape
    in_ch = int(in_ch)
    act_kind = _activation_kind(cfg.get("activation", "relu"))

    if cfg.get("variant", "plain") == "plain":
        n_blocks = len(cfg["conv_channels"])
        for i, out_ch in enumerate(cfg["conv_channels"]):
            k = _as_block_param(cfg["kernel_sizes"], i, n_blocks, "kernel_sizes")
            s = _as_block_param(cfg["strides"], i, n_blocks, "strides")
            p = _as_block_param(cfg["paddings"], i, n_blocks, "paddings")

            H, W = _append_conv2d(
                layers,
                in_ch=in_ch,
                out_ch=int(out_ch),
                H=H,
                W=W,
                kernel=int(k),
                stride=int(s),
                padding=int(p),
            )
            layers.append({"kind": act_kind, "params": {}, "meta": {}})
            in_ch = int(out_ch)

            if cfg.get("use_maxpool", False):
                H, W = _append_pool2d(
                    layers,
                    kind="MAXPOOL2D",
                    in_ch=in_ch,
                    H=H,
                    W=W,
                    kernel=int(cfg.get("maxpool_kernel", 2)),
                    stride=int(cfg.get("maxpool_stride", 2)),
                    padding=0,
                )

        layers.append({"kind": "FLATTEN", "params": {}, "meta": {"start_dim": 1}})
        feat = int(in_ch * H * W)
        layers.append(
            {
                "kind": "DENSE",
                "params": {},
                "meta": {
                    "in_features": int(feat),
                    "out_features": int(cfg["fc_hidden"]),
                    "bias_enabled": True,
                },
            }
        )
        layers.append({"kind": act_kind, "params": {}, "meta": {}})
        layers.append(
            {
                "kind": "DENSE",
                "params": {},
                "meta": {
                    "in_features": int(cfg["fc_hidden"]),
                    "out_features": int(cfg["num_classes"]),
                    "bias_enabled": True,
                },
            }
        )
        return

    ch = int(cfg["base_channels"])
    H, W = _append_conv2d(
        layers,
        in_ch=in_ch,
        out_ch=ch,
        H=H,
        W=W,
        kernel=3,
        stride=1,
        padding=1,
    )
    layers.append({"kind": act_kind, "params": {}, "meta": {}})

    for stage in range(int(cfg["stages"])):
        if stage > 0:
            next_ch = min(64, ch * int(cfg["channel_mult"]))
            if cfg["downsample"] == "stride2_conv":
                H, W = _append_conv2d(
                    layers,
                    in_ch=ch,
                    out_ch=next_ch,
                    H=H,
                    W=W,
                    kernel=3,
                    stride=2,
                    padding=1,
                )
                layers.append({"kind": act_kind, "params": {}, "meta": {}})
                ch = next_ch
            else:
                pool_kind = "MAXPOOL2D" if cfg["downsample"] == "maxpool" else "AVGPOOL2D"
                H, W = _append_pool2d(
                    layers,
                    kind=pool_kind,
                    in_ch=ch,
                    H=H,
                    W=W,
                    kernel=2,
                    stride=2,
                    padding=0,
                )
                if next_ch != ch:
                    H, W = _append_conv2d(
                        layers,
                        in_ch=ch,
                        out_ch=next_ch,
                        H=H,
                        W=W,
                        kernel=1,
                        stride=1,
                        padding=0,
                    )
                    layers.append({"kind": act_kind, "params": {}, "meta": {}})
                    ch = next_ch

        for _ in range(int(cfg["blocks_per_stage"])):
            make_double_conv = (rng.random() < float(cfg["double_conv_p"]))
            if make_double_conv:
                H, W = _append_conv2d(
                    layers,
                    in_ch=ch,
                    out_ch=ch,
                    H=H,
                    W=W,
                    kernel=3,
                    stride=1,
                    padding=1,
                )
                layers.append({"kind": act_kind, "params": {}, "meta": {}})
                H, W = _append_conv2d(
                    layers,
                    in_ch=ch,
                    out_ch=ch,
                    H=H,
                    W=W,
                    kernel=3,
                    stride=1,
                    padding=1,
                )
                layers.append({"kind": act_kind, "params": {}, "meta": {}})
            else:
                H, W = _append_conv2d(
                    layers,
                    in_ch=ch,
                    out_ch=ch,
                    H=H,
                    W=W,
                    kernel=3,
                    stride=1,
                    padding=1,
                )
                layers.append({"kind": act_kind, "params": {}, "meta": {}})

    while cfg.get("head_pool_to_1x1", True) and (H > 1 or W > 1):
        H, W = _append_pool2d(
            layers,
            kind="AVGPOOL2D",
            in_ch=ch,
            H=H,
            W=W,
            kernel=2,
            stride=2,
            padding=0,
        )
        if H <= 0 or W <= 0:
            raise ValueError("Invalid spatial dims after head pooling")

    layers.append({"kind": "FLATTEN", "params": {}, "meta": {"start_dim": 1}})
    feat = int(ch * H * W)
    layers.append(
        {
            "kind": "DENSE",
            "params": {},
            "meta": {
                "in_features": int(feat),
                "out_features": int(cfg["num_classes"]),
                "bias_enabled": True,
            },
        }
    )


class NetFactory:
    """Generator-driven factory that writes ACT Net JSONs."""

    def __init__(
        self,
        gen_config_path: str = DEFAULT_GEN_CONFIG,
        *,
        output_dir: Optional[str] = None,
        base_seed: Optional[int] = None,
        num_instances: Optional[int] = None,
        name_prefix: Optional[str] = None,
        write_manifest: Optional[bool] = None,
        manifest_path: Optional[str] = None,
    ):
        self.config_path = str(gen_config_path)
        self.config = _load_gen_config(self.config_path)
        gen = self.config.get("generator", {})

        self.base_seed = int(base_seed) if base_seed is not None else gen.get("base_seed")
        if self.base_seed is None:
            self.base_seed = int(secrets.randbits(32))

        self.num_instances = int(num_instances) if num_instances is not None else int(gen.get("num_instances", 0))
        self.name_prefix = str(name_prefix) if name_prefix is not None else str(gen.get("name_prefix", DEFAULT_NAME_PREFIX))

        output_dir = output_dir or gen.get("output_dir", DEFAULT_NETS_DIR)
        self.output_dir = Path(str(output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.write_manifest = bool(write_manifest) if write_manifest is not None else bool(gen.get("write_manifest", True))
        manifest_default = gen.get("manifest_path")
        if manifest_path is None:
            manifest_path = manifest_default
        self.manifest_path = Path(manifest_path) if manifest_path else (self.output_dir / "manifest.json")

    def generate_weight_tensor(self, kind: str, meta: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Generate minimal weight tensors that satisfy schema requirements."""
        if kind == "DENSE":
            in_features = meta.get("in_features", 10)
            out_features = meta.get("out_features", 10)
            return torch.randn(out_features, in_features) * 0.1
        if kind in ["CONV2D", "CONV1D", "CONV3D"]:
            in_channels = meta.get("in_channels", 1)
            out_channels = meta.get("out_channels", 1)
            kernel_size = meta.get("kernel_size", 3)
            if isinstance(kernel_size, int):
                if kind == "CONV1D":
                    weight_shape = (out_channels, in_channels, kernel_size)
                elif kind == "CONV2D":
                    weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
                else:
                    weight_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
            else:
                if kind == "CONV1D":
                    weight_shape = (out_channels, in_channels, kernel_size[0])
                elif kind == "CONV2D":
                    weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
                else:
                    weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])
            return torch.randn(*weight_shape) * 0.1
        return None

    def _generate_input_spec_params(self, params: Dict[str, Any], meta: Dict[str, Any], input_shape: Optional[List[int]]) -> None:
        if not input_shape:
            raise ValueError("Cannot generate INPUT_SPEC params: input shape is required but not provided")

        spec_kind = str(meta.get("kind"))
        if spec_kind == "BOX":
            lb_val = meta.get("lb_val", 0.0)
            ub_val = meta.get("ub_val", 1.0)
            params["lb"] = torch.full(input_shape, lb_val)
            params["ub"] = torch.full(input_shape, ub_val)
        elif spec_kind == "LINF_BALL":
            eps = meta.get("eps")
            if eps is None:
                raise ValueError("LINF_BALL requires 'eps' in meta")
            center_val = meta.get("center_val", 0.5)
            params["center"] = torch.full(input_shape, center_val)
            params["lb"] = params["center"] - eps
            params["ub"] = params["center"] + eps

    def _generate_assert_params(self, params: Dict[str, Any], meta: Dict[str, Any], output_shape: Optional[List[int]]) -> None:
        if not output_shape:
            raise ValueError("Cannot generate ASSERT params: output shape is required but not provided")

        assert_kind = str(meta.get("kind"))
        if assert_kind == "TOP1_ROBUST":
            if "y_true" not in meta:
                raise ValueError("TOP1_ROBUST requires 'y_true' in meta")
        elif assert_kind == "MARGIN_ROBUST":
            if "y_true" not in meta:
                raise ValueError("MARGIN_ROBUST requires 'y_true' in meta")
            if "margin" not in meta:
                raise ValueError("MARGIN_ROBUST requires 'margin' in meta")
        elif assert_kind == "LINEAR_LE":
            if "c" in params and isinstance(params["c"], list):
                params["c"] = torch.tensor(params["c"], dtype=torch.float32)
            if "d" not in meta:
                raise ValueError("LINEAR_LE requires 'd' in meta")
        elif assert_kind == "RANGE":
            if "lb" in params and isinstance(params["lb"], list):
                params["lb"] = torch.tensor(params["lb"], dtype=torch.float32)
            if "ub" in params and isinstance(params["ub"], list):
                params["ub"] = torch.tensor(params["ub"], dtype=torch.float32)
            if "lb" not in params or "ub" not in params:
                raise ValueError("RANGE requires both 'lb' and 'ub' in params")

    def _generate_layer_variables(
        self,
        kind: str,
        layer_index: int,
        var_counter: int,
        meta: Dict[str, Any],
        layers: List[Layer],
    ) -> Tuple[List[int], List[int], int]:
        if kind == "INPUT":
            shape = meta.get("shape", [])
            out_num_vars = torch.Size(shape).numel() if shape else 1
            out_vars = list(range(var_counter, var_counter + out_num_vars))
            var_counter += out_num_vars
            return [], out_vars, var_counter

        if kind == "DENSE":
            in_features = meta.get("in_features", 1)
            out_features = meta.get("out_features", 1)
            if layers and layer_index > 0:
                prev_out_vars = layers[layer_index - 1].out_vars
                if len(prev_out_vars) != in_features:
                    raise ValueError(f"DENSE layer expects {in_features} inputs but got {len(prev_out_vars)}")
                in_vars = prev_out_vars
            else:
                in_vars = []
            out_vars = list(range(var_counter, var_counter + out_features))
            var_counter += out_features
            return in_vars, out_vars, var_counter

        if kind in ["RELU", "SIGMOID", "TANH"]:
            if layers and layer_index > 0:
                in_vars = layers[layer_index - 1].out_vars
                out_vars = list(range(var_counter, var_counter + len(in_vars)))
                var_counter += len(in_vars)
                return in_vars, out_vars, var_counter
            raise ValueError(f"Activation layer '{kind}' cannot be the first layer in network")

        if kind.startswith("CONV"):
            if layers and layer_index > 0:
                in_vars = layers[layer_index - 1].out_vars
            else:
                raise ValueError(f"Convolutional layer '{kind}' cannot be the first layer in network")

            output_shape = meta.get("output_shape")
            if output_shape:
                out_num_vars = torch.Size(output_shape).numel()
            else:
                raise ValueError(f"Convolutional layer '{kind}' requires 'output_shape' in meta")

            out_vars = list(range(var_counter, var_counter + out_num_vars))
            var_counter += out_num_vars
            return in_vars, out_vars, var_counter

        if kind in ["MAXPOOL2D", "AVGPOOL2D", "ADAPTIVEAVGPOOL2D"]:
            if layers and layer_index > 0:
                in_vars = layers[layer_index - 1].out_vars
            else:
                raise ValueError(f"Pooling layer '{kind}' cannot be the first layer in network")

            output_shape = meta.get("output_shape")
            if output_shape:
                out_num_vars = torch.Size(output_shape).numel()
            else:
                raise ValueError(f"Pooling layer '{kind}' requires 'output_shape' in meta")

            out_vars = list(range(var_counter, var_counter + out_num_vars))
            var_counter += out_num_vars
            return in_vars, out_vars, var_counter

        if kind == "FLATTEN":
            if layers and layer_index > 0:
                in_vars = layers[layer_index - 1].out_vars
                out_vars = list(range(var_counter, var_counter + len(in_vars)))
                var_counter += len(in_vars)
                return in_vars, out_vars, var_counter
            raise ValueError("Flatten layer cannot be the first layer in network")

        if kind in ["ADD", "SUB", "MUL", "DIV"]:
            x_vars = meta.get("x_vars")
            y_vars = meta.get("y_vars")
            if x_vars is None or y_vars is None:
                raise ValueError(f"{kind} layer requires meta['x_vars'] and meta['y_vars']")
            if len(x_vars) != len(y_vars):
                raise ValueError(f"{kind} layer expects x_vars and y_vars of same length")
            in_vars = list(x_vars) + list(y_vars)
            out_vars = list(range(var_counter, var_counter + len(x_vars)))
            var_counter += len(x_vars)
            return in_vars, out_vars, var_counter

        if kind in ["INPUT_SPEC", "ASSERT"]:
            if layers and layer_index > 0:
                prev_vars = layers[layer_index - 1].out_vars
                return prev_vars, prev_vars.copy(), var_counter
            raise ValueError(f"Layer '{kind}' cannot be the first layer in network")

        supported_types = [
            "INPUT",
            "DENSE",
            "RELU",
            "SIGMOID",
            "TANH",
            "CONV1D",
            "CONV2D",
            "CONV3D",
            "MAXPOOL2D",
            "AVGPOOL2D",
            "ADAPTIVEAVGPOOL2D",
            "FLATTEN",
            "ADD",
            "SUB",
            "MUL",
            "DIV",
            "INPUT_SPEC",
            "ASSERT",
        ]
        raise NotImplementedError(
            f"Layer type '{kind}' is not supported. Supported types: {supported_types}."
        )

    def _sample_family(self, rng: random.Random) -> str:
        gen = self.config.get("generator", {})
        fams = list(gen.get("families", []))
        if not fams:
            raise ValueError("generator.families must be non-empty")
        if len(fams) == 1:
            return str(fams[0])

        has_mlp = "mlp" in fams
        has_cnn = "cnn2d" in fams
        if has_mlp and has_cnn:
            return "mlp" if (rng.random() < float(gen.get("p_mlp", 0.5))) else "cnn2d"
        return str(rng.choice(fams))

    def _sample_mlp(self, rng: random.Random, *, num_classes: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cfg = self.config.get("mlp", {})
        input_shape = _choose(rng, cfg.get("input_shapes", []), name="mlp.input_shapes")
        depth = _randint_inclusive(rng, cfg.get("depth_range", [2, 4]))
        if depth <= 0:
            raise ValueError(f"mlp.depth_range produced non-positive depth={depth}")

        widths = [int(_choose(rng, cfg.get("width_choices", []), name="mlp.width_choices")) for _ in range(depth)]
        hidden_sizes = tuple(widths)

        activation = str(_choose(rng, cfg.get("activation_choices", []), name="mlp.activation_choices"))
        dropout_p = float(_choose(rng, cfg.get("dropout_p_choices", []), name="mlp.dropout_p_choices"))

        block_p = float(cfg.get("block_p", 0.0))
        residual_p = float(cfg.get("residual_p", 0.0))
        r = rng.random()
        if r < residual_p:
            variant = "residual"
        elif r < residual_p + block_p:
            variant = "block"
        else:
            variant = "plain"
        num_blocks = _randint_inclusive(rng, cfg.get("block_count_range", [2, 4]))
        block_width = int(_choose(rng, cfg.get("block_width_choices", []), name="mlp.block_width_choices"))
        post_block_activation = bool(rng.random() < float(cfg.get("post_block_activation_p", 0.5)))
        residual_blocks = _randint_inclusive(rng, cfg.get("residual_blocks_range", [1, 2]))
        residual_width = int(_choose(rng, cfg.get("residual_width_choices", []), name="mlp.residual_width_choices"))

        if variant == "block":
            dropout_p = 0.0

        model_cfg = {
            "input_shape": tuple(int(x) for x in input_shape),
            "hidden_sizes": hidden_sizes,
            "variant": variant,
            "num_blocks": int(num_blocks),
            "block_width": int(block_width),
            "post_block_activation": bool(post_block_activation),
            "num_residual_blocks": int(residual_blocks),
            "residual_width": int(residual_width),
            "activation": activation,
            "use_bias": True,
            "dropout_p": float(dropout_p),
            "num_classes": int(num_classes),
        }
        meta = {
            "depth": int(depth),
            "hidden_sizes": list(hidden_sizes),
            "variant": variant,
            "num_blocks": int(num_blocks),
            "block_width": int(block_width),
            "post_block_activation": bool(post_block_activation),
            "num_residual_blocks": int(residual_blocks),
            "residual_width": int(residual_width),
            "activation": activation,
            "dropout_p": float(dropout_p),
            "input_shape": list(model_cfg["input_shape"]),
            "num_classes": int(num_classes),
        }
        return model_cfg, meta

    def _sample_cnn2d(self, rng: random.Random, *, num_classes: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cfg = self.config.get("cnn", {})
        input_shape = _choose(rng, cfg.get("input_shapes", []), name="cnn.input_shapes")
        if int(input_shape[2]) > 32 or int(input_shape[3]) > 32:
            raise ValueError(f"cnn.input_shapes must have H,W <= 32, got {input_shape}")

        variant = "stage" if (rng.random() < float(cfg.get("stage_variant_p", 0.0))) else "plain"
        blocks = _randint_inclusive(rng, cfg.get("num_blocks_range", [1, 2]))
        if blocks <= 0:
            raise ValueError(f"cnn.num_blocks_range produced non-positive blocks={blocks}")

        conv_channels: List[int] = []
        prev = int(_choose(rng, cfg.get("channels_choices", []), name="cnn.channels_choices"))
        conv_channels.append(prev)
        for _ in range(1, blocks):
            prev = int(_choose(rng, cfg.get("channels_choices", []), name="cnn.channels_choices"))
            conv_channels.append(prev)

        kernel_sizes = int(_choose(rng, cfg.get("kernel_choices", []), name="cnn.kernel_choices"))
        strides = int(_choose(rng, cfg.get("stride_choices", []), name="cnn.stride_choices"))
        paddings = int(_choose(rng, cfg.get("padding_choices", []), name="cnn.padding_choices"))
        activation = str(_choose(rng, cfg.get("activation_choices", []), name="cnn.activation_choices"))
        use_maxpool = bool(rng.random() < float(cfg.get("use_maxpool_p", 0.0)))
        fc_hidden = int(_choose(rng, cfg.get("fc_hidden_choices", []), name="cnn.fc_hidden_choices"))

        stages = _randint_inclusive(rng, cfg.get("stages_range", [1, 3]))
        blocks_per_stage = _randint_inclusive(rng, cfg.get("blocks_per_stage_range", [1, 2]))
        base_channels = int(_choose(rng, cfg.get("base_channels_choices", []), name="cnn.base_channels_choices"))
        channel_mult = int(_choose(rng, cfg.get("channel_mult_choices", []), name="cnn.channel_mult_choices"))
        downsample = str(_choose(rng, cfg.get("downsample_choices", []), name="cnn.downsample_choices"))
        double_conv_p = float(_choose(rng, cfg.get("double_conv_p_choices", []), name="cnn.double_conv_p_choices"))

        max_channels = base_channels * (channel_mult ** (stages - 1))
        if max_channels > 64:
            stages = max(1, min(stages, 3))
            while stages > 1 and base_channels * (channel_mult ** (stages - 1)) > 64:
                stages -= 1
            max_channels = base_channels * (channel_mult ** (stages - 1))
            if max_channels > 64:
                base_channels = min(base_channels, 64)

        model_cfg = {
            "input_shape": tuple(int(x) for x in input_shape),
            "conv_channels": tuple(conv_channels),
            "variant": variant,
            "stages": int(stages),
            "blocks_per_stage": int(blocks_per_stage),
            "base_channels": int(base_channels),
            "channel_mult": int(channel_mult),
            "downsample": str(downsample),
            "double_conv_p": float(double_conv_p),
            "head_pool_to_1x1": True,
            "kernel_sizes": kernel_sizes,
            "strides": strides,
            "paddings": paddings,
            "activation": activation,
            "use_bias": True,
            "use_maxpool": use_maxpool,
            "maxpool_kernel": 2,
            "maxpool_stride": 2,
            "num_classes": int(num_classes),
            "fc_hidden": int(fc_hidden),
        }
        meta = {
            "blocks": int(blocks),
            "conv_channels": conv_channels,
            "variant": variant,
            "stages": int(stages),
            "blocks_per_stage": int(blocks_per_stage),
            "base_channels": int(base_channels),
            "channel_mult": int(channel_mult),
            "downsample": str(downsample),
            "double_conv_p": float(double_conv_p),
            "kernel_sizes": kernel_sizes,
            "strides": strides,
            "paddings": paddings,
            "activation": activation,
            "use_maxpool": use_maxpool,
            "fc_hidden": int(fc_hidden),
            "input_shape": list(model_cfg["input_shape"]),
            "num_classes": int(num_classes),
        }
        return model_cfg, meta

    def _sample_input_spec(
        self,
        rng: random.Random,
        *,
        input_shape: Tuple[int, ...],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cfg = self.config.get("input_spec", {})
        kinds = list(cfg.get("kind_choices", []))
        if not kinds:
            raise ValueError("input_spec.kind_choices must be non-empty")

        if len(kinds) == 1:
            kind = kinds[0]
        else:
            has_box = "BOX" in kinds
            has_linf = "LINF_BALL" in kinds
            if has_box and has_linf and len(kinds) == 2:
                kind = "BOX" if (rng.random() < float(cfg.get("p_box", 0.5))) else "LINF_BALL"
            else:
                kind = rng.choice(kinds)

        value_range = _choose(rng, cfg.get("value_range_choices", []), name="input_spec.value_range_choices")
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

            in_cfg = {
                "kind": "BOX",
                "value_range": (lo, hi),
                "lb_val": float(lb_val),
                "ub_val": float(ub_val),
                "meta": {"sampled": {"shape": list(input_shape)}},
            }
            meta = {
                "kind": "BOX",
                "value_range": (lo, hi),
                "lb_val": float(lb_val),
                "ub_val": float(ub_val),
            }
            return in_cfg, meta

        if kind == "LINF_BALL":
            center_val = lo + (hi - lo) * rng.random()
            eps = float(_choose(rng, cfg.get("eps_choices", []), name="input_spec.eps_choices"))
            eps = min(eps, 0.5 * (hi - lo)) if (hi > lo) else 0.0

            in_cfg = {
                "kind": "LINF_BALL",
                "value_range": (lo, hi),
                "center_val": float(center_val),
                "eps": float(eps),
                "meta": {"sampled": {"shape": list(input_shape)}},
            }
            meta = {
                "kind": "LINF_BALL",
                "value_range": (lo, hi),
                "center_val": float(center_val),
                "eps": float(eps),
            }
            return in_cfg, meta

        in_cfg = {
            "kind": "BOX",
            "value_range": (lo, hi),
            "lb_val": float(lo),
            "ub_val": float(hi),
            "meta": {"fallback": True, "reason": f"unsupported kind={kind}"},
        }
        meta = {"kind": "FALLBACK_BOX", "value_range": (lo, hi), "lb_val": lo, "ub_val": hi}
        return in_cfg, meta

    def _sample_output_spec(
        self,
        rng: random.Random,
        *,
        num_classes: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cfg = self.config.get("output_spec", {})
        kinds = list(cfg.get("kind_choices", []))
        if not kinds:
            raise ValueError("output_spec.kind_choices must be non-empty")

        if len(kinds) == 1:
            kind = kinds[0]
        else:
            has_top1 = "TOP1_ROBUST" in kinds
            has_margin = "MARGIN_ROBUST" in kinds
            if has_top1 and has_margin and len(kinds) == 2:
                kind = "TOP1_ROBUST" if (rng.random() < float(cfg.get("p_top1", 0.5))) else "MARGIN_ROBUST"
            else:
                kind = rng.choice(kinds)

        y_true = int(rng.randrange(int(num_classes)))

        if kind == "TOP1_ROBUST":
            out_cfg = {"kind": "TOP1_ROBUST", "y_true": y_true, "margin": 0.0, "meta": {}}
            meta = {"kind": "TOP1_ROBUST", "y_true": y_true, "margin": 0.0}
            return out_cfg, meta

        if kind == "MARGIN_ROBUST":
            margin = float(_choose(rng, cfg.get("margin_choices", []), name="output_spec.margin_choices"))
            out_cfg = {"kind": "MARGIN_ROBUST", "y_true": y_true, "margin": float(margin), "meta": {}}
            meta = {"kind": "MARGIN_ROBUST", "y_true": y_true, "margin": float(margin)}
            return out_cfg, meta

        if kind == "LINEAR_LE":
            c_lo, c_hi = cfg.get("linear_le_c_range", [-1.0, 1.0])
            d_lo, d_hi = cfg.get("linear_le_d_range", [-1.0, 1.0])
            c_vals = [c_lo + (c_hi - c_lo) * rng.random() for _ in range(int(num_classes))]
            d_val = d_lo + (d_hi - d_lo) * rng.random()
            out_cfg = {"kind": "LINEAR_LE", "c": [float(x) for x in c_vals], "d": float(d_val), "meta": {}}
            meta = {"kind": "LINEAR_LE", "c": list(c_vals), "d": float(d_val)}
            return out_cfg, meta

        if kind == "RANGE":
            lo, hi = _choose(rng, cfg.get("range_choices", []), name="output_spec.range_choices")
            lb_vals = []
            ub_vals = []
            for _ in range(int(num_classes)):
                a = lo + (hi - lo) * rng.random()
                b = lo + (hi - lo) * rng.random()
                lb_vals.append(min(a, b))
                ub_vals.append(max(a, b))
            out_cfg = {
                "kind": "RANGE",
                "lb": [float(x) for x in lb_vals],
                "ub": [float(x) for x in ub_vals],
                "meta": {},
            }
            meta = {"kind": "RANGE", "lb": list(lb_vals), "ub": list(ub_vals)}
            return out_cfg, meta

        out_cfg = {"kind": "TOP1_ROBUST", "y_true": y_true, "margin": 0.0, "meta": {"fallback": True}}
        meta = {"kind": "FALLBACK_TOP1_ROBUST", "y_true": y_true, "margin": 0.0}
        return out_cfg, meta

    def _build_network_spec(self, instance: Dict[str, Any], *, dtype: str) -> Dict[str, Any]:
        model_cfg = instance["model_cfg"]
        input_shape = list(model_cfg["input_shape"])
        num_classes = int(model_cfg["num_classes"])

        layers: List[Dict[str, Any]] = []

        input_meta: Dict[str, Any] = {
            "shape": input_shape,
            "dtype": str(dtype),
            "desc": "Generated by NetFactory",
            "num_classes": num_classes,
            "value_range": list(instance["input_spec"]["value_range"]),
        }
        layers.append({"kind": "INPUT", "params": {}, "meta": input_meta})

        in_kind = str(instance["input_spec"]["kind"])
        spec_meta: Dict[str, Any] = {"kind": in_kind}
        if in_kind == "BOX":
            spec_meta["lb_val"] = float(instance["input_spec"].get("lb_val", instance["input_spec"]["value_range"][0]))
            spec_meta["ub_val"] = float(instance["input_spec"].get("ub_val", instance["input_spec"]["value_range"][1]))
        elif in_kind == "LINF_BALL":
            spec_meta["center_val"] = float(instance["input_spec"].get("center_val", sum(instance["input_spec"]["value_range"]) / 2.0))
            spec_meta["eps"] = float(instance["input_spec"].get("eps", 0.0))
        else:
            raise ValueError(f"Input spec kind '{in_kind}' is not supported")

        layers.append({"kind": "INPUT_SPEC", "params": {}, "meta": spec_meta})

        if instance["family"] == "mlp":
            _build_mlp_layers(layers, cfg=model_cfg)
            arch = "mlp"
        elif instance["family"] == "cnn2d":
            rng = random.Random(int(instance["seed"]))
            _build_cnn_layers(layers, cfg=model_cfg, rng=rng)
            arch = "cnn"
        else:
            raise ValueError(f"Unsupported model family: {instance['family']}")

        out_kind = str(instance["output_spec"]["kind"])
        out_meta: Dict[str, Any] = {"kind": out_kind}
        out_params: Dict[str, Any] = {}

        if out_kind == "TOP1_ROBUST":
            out_meta["y_true"] = int(instance["output_spec"].get("y_true", 0))
        elif out_kind == "MARGIN_ROBUST":
            out_meta["y_true"] = int(instance["output_spec"].get("y_true", 0))
            out_meta["margin"] = float(instance["output_spec"].get("margin", 0.0))
        elif out_kind == "LINEAR_LE":
            out_params["c"] = list(instance["output_spec"].get("c", [1.0] * num_classes))
            out_meta["d"] = float(instance["output_spec"].get("d", 0.0))
        elif out_kind == "RANGE":
            out_params["lb"] = list(instance["output_spec"].get("lb", [0.0] * num_classes))
            out_params["ub"] = list(instance["output_spec"].get("ub", [0.0] * num_classes))
        else:
            out_meta["y_true"] = int(instance["output_spec"].get("y_true", 0))

        layers.append({"kind": "ASSERT", "params": out_params, "meta": out_meta})

        return {
            "description": f"Generated ({instance['instance_id']})",
            "architecture_type": arch,
            "input_shape": input_shape,
            "layers": layers,
        }

    def _sample_instance(self, idx: int) -> Dict[str, Any]:
        gen = self.config.get("generator", {})
        instance_id = f"{self.name_prefix}{int(self.base_seed)}_idx{int(idx):05d}"
        seed = int(_derive_seed(int(self.base_seed), int(idx), instance_id))
        rng = random.Random(seed)

        family = self._sample_family(rng)
        num_classes = int(_choose(rng, gen.get("num_classes_choices", []), name="generator.num_classes_choices"))

        if family == "mlp":
            model_cfg, model_meta = self._sample_mlp(rng, num_classes=num_classes)
        elif family == "cnn2d":
            model_cfg, model_meta = self._sample_cnn2d(rng, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown family: {family}")

        input_spec, in_meta = self._sample_input_spec(rng, input_shape=tuple(model_cfg["input_shape"]))
        output_spec, out_meta = self._sample_output_spec(rng, num_classes=num_classes)

        meta: Dict[str, Any] = {
            "base_seed": int(self.base_seed),
            "idx": int(idx),
            "seed": int(seed),
            "model": model_meta,
            "input_spec": in_meta,
            "output_spec": out_meta,
        }

        return {
            "instance_id": instance_id,
            "seed": seed,
            "family": family,
            "model_cfg": model_cfg,
            "input_spec": input_spec,
            "output_spec": output_spec,
            "meta": meta,
        }

    def create_network(self, name: str, spec: Dict[str, Any]) -> Net:
        current_dtype = str(get_default_dtype())

        layers = []
        var_counter = 0

        layer_specs = list(spec["layers"])
        for i, layer_spec in enumerate(layer_specs):
            params = layer_spec.get("params", {}).copy()
            meta = layer_spec.get("meta", {}).copy()
            kind = layer_spec["kind"]

            if kind in ["ADD", "SUB", "MUL", "DIV"]:
                inputs = layer_spec.get("inputs") or {}
                x_src = inputs.get("x")
                y_src = inputs.get("y")
                if x_src is None or y_src is None:
                    raise ValueError(f"{kind} layer requires inputs {{'x': idx, 'y': idx}} in spec")
                if x_src >= len(layers) or y_src >= len(layers):
                    raise ValueError(f"{kind} inputs must reference earlier layers (x={x_src}, y={y_src})")
                meta["x_vars"] = list(layers[x_src].out_vars)
                meta["y_vars"] = list(layers[y_src].out_vars)

            in_vars, out_vars, var_counter = self._generate_layer_variables(kind, i, var_counter, meta, layers)

            if kind == "INPUT" and "dtype" in meta:
                meta["dtype"] = current_dtype

            input_shape = None
            if i > 0 and layers[i - 1].kind == "INPUT":
                input_shape = layers[i - 1].meta.get("shape")

            output_shape = None
            if i > 0:
                for j in range(i - 1, -1, -1):
                    prev_layer = layers[j]
                    if prev_layer.kind == "DENSE":
                        out_features = prev_layer.meta.get("out_features")
                        if out_features:
                            output_shape = [1, out_features]
                            break

            if kind == "INPUT_SPEC":
                self._generate_input_spec_params(params, meta, input_shape)
            elif kind == "ASSERT":
                self._generate_assert_params(params, meta, output_shape)
            elif kind == "DENSE" and "W" not in params:
                weight = self.generate_weight_tensor(kind, meta)
                if weight is not None:
                    params["W"] = weight
                if meta.get("bias_enabled", False):
                    out_features = meta.get("out_features", 10)
                    params["b"] = torch.zeros(out_features)
            elif kind.startswith("CONV") and "weight" not in params:
                weight = self.generate_weight_tensor(kind, meta)
                if weight is not None:
                    params["weight"] = weight

            layer = Layer(
                id=i,
                kind=kind,
                params=params,
                meta=meta,
                in_vars=in_vars,
                out_vars=out_vars,
            )
            layers.append(layer)

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
        net.meta = {
            "name": name,
            "description": spec.get("description", ""),
            "architecture_type": spec.get("architecture_type", ""),
            "input_shape": spec.get("input_shape", []),
        }
        return net

    def save_network(self, net: Net, name: str) -> None:
        output_path = self.output_dir / f"{name}.json"
        net_dict = NetSerializer.serialize_net(net, metadata={"generated_by": "NetFactory"})
        with open(output_path, "w") as f:
            json.dump(net_dict, f, indent=2)
        print(f"Saved: {output_path}")

    def _write_manifest(self, names: List[str]) -> None:
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
        print(f"Generating {self.num_instances} networks...")
        gen = self.config.get("generator", {})
        dtype = str(gen.get("dtype", "torch.float64"))

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
