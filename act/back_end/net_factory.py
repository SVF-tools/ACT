#===- act/back_end/net_factory.py - YAML-Driven ACT Net Factory ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Generator-driven ACT Net generator. Samples MLP/CNN2D
#   configs, fills INPUT_SPEC/ASSERT params, builds layer vars/weights, and
#   writes Net JSONs + optional manifest for downstream verification.
#   Runtime uses JSON nets under act/back_end/examples/nets; YAML is config-only.
#   CLI entry: python -m act.back_end --generate (see act/back_end/cli.py).
#   INPUT.meta.dtype is set from device_manager default (get_default_dtype()).
#
#===---------------------------------------------------------------------===#
# ASSERT semantics are documented in act/back_end/README.md.
#===---------------------------------------------------------------------===#

from __future__ import annotations

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
from act.front_end.specs import InKind, OutKind
from act.util.device_manager import get_default_dtype
from act.util.path_config import get_examples_gen_config_path


def _load_gen_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping: {cfg_path}")
    return data


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


def _append_dense(
    layers: List[Dict[str, Any]],
    *,
    in_features: int,
    out_features: int,
    use_bias: bool,
) -> None:
    layers.append(
        {
            "kind": "DENSE",
            "params": {},
            "meta": {
                "in_features": int(in_features),
                "out_features": int(out_features),
                "bias_enabled": bool(use_bias),
            },
        }
    )


def _append_act(layers: List[Dict[str, Any]], act_kind: str) -> None:
    layers.append({"kind": act_kind, "params": {}, "meta": {}})


def _append_add(layers: List[Dict[str, Any]], *, skip_idx: int, main_idx: int) -> None:
    layers.append(
        {
            "kind": "ADD",
            "params": {},
            "meta": {},
            "inputs": {"x": skip_idx, "y": main_idx},
            "preds": [skip_idx, main_idx],
        }
    )

def _build_mlp_layers(layers: List[Dict[str, Any]], *, cfg: Dict[str, Any]) -> None:
    shape = _ensure_batch1(tuple(cfg["input_shape"]))
    in_features = int(shape[1]) if len(shape) == 2 else _prod(shape[1:])

    if len(shape) > 2:
        layers.append({"kind": "FLATTEN", "params": {}, "meta": {"start_dim": 1}})

    act_kind = _activation_kind(cfg["activation"])
    use_bias = bool(cfg["use_bias"])
    variant = cfg["variant"]
    if variant == "plain":
        for h in cfg["hidden_sizes"]:
            _append_dense(layers, in_features=in_features, out_features=int(h), use_bias=use_bias)
            _append_act(layers, act_kind)
            in_features = int(h)
    elif variant == "block":
        width = int(cfg["block_width"])
        _append_dense(layers, in_features=in_features, out_features=width, use_bias=use_bias)
        _append_act(layers, act_kind)
        in_features = int(width)

        for _ in range(int(cfg["num_blocks"])):
            _append_dense(layers, in_features=in_features, out_features=in_features, use_bias=use_bias)
            _append_act(layers, act_kind)
            _append_dense(layers, in_features=in_features, out_features=in_features, use_bias=use_bias)
            if cfg["post_block_activation"]:
                _append_act(layers, act_kind)
    elif variant == "residual":
        width = int(cfg["residual_width"])
        if in_features != width:
            _append_dense(layers, in_features=in_features, out_features=width, use_bias=use_bias)
            _append_act(layers, act_kind)
            in_features = int(width)

        num_blocks = int(cfg["num_residual_blocks"])
        for _ in range(num_blocks):
            skip_idx = len(layers) - 1
            _append_dense(layers, in_features=in_features, out_features=in_features, use_bias=use_bias)
            _append_act(layers, act_kind)
            _append_dense(layers, in_features=in_features, out_features=in_features, use_bias=use_bias)
            main_idx = len(layers) - 1
            _append_add(layers, skip_idx=skip_idx, main_idx=main_idx)
            _append_act(layers, act_kind)
    else:
        raise ValueError(f"Unsupported MLP variant '{variant}'")

    _append_dense(layers, in_features=in_features, out_features=int(cfg["num_classes"]), use_bias=True)


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
    act_kind = _activation_kind(cfg["activation"])

    if cfg["variant"] == "plain":
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
            _append_act(layers, act_kind)
            in_ch = int(out_ch)

            if cfg["use_maxpool"]:
                H, W = _append_pool2d(
                    layers,
                    kind="MAXPOOL2D",
                    in_ch=in_ch,
                    H=H,
                    W=W,
                    kernel=int(cfg["maxpool_kernel"]),
                    stride=int(cfg["maxpool_stride"]),
                    padding=0,
                )

        layers.append({"kind": "FLATTEN", "params": {}, "meta": {"start_dim": 1}})
        feat = int(in_ch * H * W)
        _append_dense(layers, in_features=int(feat), out_features=int(cfg["fc_hidden"]), use_bias=True)
        _append_act(layers, act_kind)
        _append_dense(layers, in_features=int(cfg["fc_hidden"]), out_features=int(cfg["num_classes"]), use_bias=True)
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
    _append_act(layers, act_kind)

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
                _append_act(layers, act_kind)
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
                    _append_act(layers, act_kind)
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
                _append_act(layers, act_kind)
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
                _append_act(layers, act_kind)
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
                _append_act(layers, act_kind)

    while cfg["head_pool_to_1x1"] and (H > 1 or W > 1):
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
    _append_dense(layers, in_features=int(feat), out_features=int(cfg["num_classes"]), use_bias=True)


class NetFactory:
    """Generator-driven factory that writes ACT Net JSONs."""

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
        self.config = _load_gen_config(self.config_path)
        gen = self.config["generator"]

        base_seed = base_seed if base_seed is not None else gen.get("base_seed")
        self.base_seed = int(base_seed) if base_seed is not None else int(secrets.randbits(32))
        self.num_instances = int(num_instances) if num_instances is not None else int(gen["num_instances"])
        self.name_prefix = str(name_prefix) if name_prefix is not None else str(gen["name_prefix"])

        output_dir = output_dir or gen["output_dir"]
        self.output_dir = Path(str(output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.write_manifest = bool(write_manifest) if write_manifest is not None else bool(gen["write_manifest"])
        if manifest_path is None:
            manifest_path = gen.get("manifest_path")
        self.manifest_path = Path(manifest_path) if manifest_path else (self.output_dir / "manifest.json")

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
        self,
        meta: Dict[str, Any],
        input_shape: List[int],
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
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

    def _assert_params(
        self,
        params: Dict[str, Any],
        meta: Dict[str, Any],
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
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
        self,
        kind: str,
        layer_index: int,
        var_counter: int,
        meta: Dict[str, Any],
        layers: List[Layer],
    ) -> Tuple[List[int], List[int], int]:
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

    def _sample_family(self, rng: random.Random) -> str:
        gen = self.config["generator"]
        fams = list(gen["families"])
        if len(fams) == 1:
            return str(fams[0])

        has_mlp = "mlp" in fams
        has_cnn = "cnn2d" in fams
        if has_mlp and has_cnn:
            return "mlp" if (rng.random() < float(gen["p_mlp"])) else "cnn2d"
        return str(rng.choice(fams))

    def _sample_mlp(self, rng: random.Random, *, num_classes: int) -> Dict[str, Any]:
        cfg = self.config["mlp"]
        input_shape = _choose(rng, cfg["input_shapes"], name="mlp.input_shapes")
        depth = _randint_inclusive(rng, cfg["depth_range"])
        if depth <= 0:
            raise ValueError(f"mlp.depth_range produced non-positive depth={depth}")

        widths = [int(_choose(rng, cfg["width_choices"], name="mlp.width_choices")) for _ in range(depth)]
        hidden_sizes = tuple(widths)

        activation = str(_choose(rng, cfg["activation_choices"], name="mlp.activation_choices"))

        block_p = float(cfg["block_p"])
        residual_p = float(cfg["residual_p"])
        r = rng.random()
        if r < residual_p:
            variant = "residual"
        elif r < residual_p + block_p:
            variant = "block"
        else:
            variant = "plain"
        num_blocks = _randint_inclusive(rng, cfg["block_count_range"])
        block_width = int(_choose(rng, cfg["block_width_choices"], name="mlp.block_width_choices"))
        post_block_activation = bool(rng.random() < float(cfg["post_block_activation_p"]))
        residual_blocks = _randint_inclusive(rng, cfg["residual_blocks_range"])
        residual_width = int(_choose(rng, cfg["residual_width_choices"], name="mlp.residual_width_choices"))

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
            "num_classes": int(num_classes),
        }
        return model_cfg

    def _sample_cnn2d(self, rng: random.Random, *, num_classes: int) -> Dict[str, Any]:
        cfg = self.config["cnn"]
        input_shape = _choose(rng, cfg["input_shapes"], name="cnn.input_shapes")
        if int(input_shape[2]) > 32 or int(input_shape[3]) > 32:
            raise ValueError(f"cnn.input_shapes must have H,W <= 32, got {input_shape}")

        variant = "stage" if (rng.random() < float(cfg["stage_variant_p"])) else "plain"
        blocks = _randint_inclusive(rng, cfg["num_blocks_range"])
        if blocks <= 0:
            raise ValueError(f"cnn.num_blocks_range produced non-positive blocks={blocks}")

        conv_channels: List[int] = []
        prev = int(_choose(rng, cfg["channels_choices"], name="cnn.channels_choices"))
        conv_channels.append(prev)
        for _ in range(1, blocks):
            prev = int(_choose(rng, cfg["channels_choices"], name="cnn.channels_choices"))
            conv_channels.append(prev)

        kernel_sizes = int(_choose(rng, cfg["kernel_choices"], name="cnn.kernel_choices"))
        strides = int(_choose(rng, cfg["stride_choices"], name="cnn.stride_choices"))
        paddings = int(_choose(rng, cfg["padding_choices"], name="cnn.padding_choices"))
        activation = str(_choose(rng, cfg["activation_choices"], name="cnn.activation_choices"))
        use_maxpool = bool(rng.random() < float(cfg["use_maxpool_p"]))
        fc_hidden = int(_choose(rng, cfg["fc_hidden_choices"], name="cnn.fc_hidden_choices"))

        stages = _randint_inclusive(rng, cfg["stages_range"])
        blocks_per_stage = _randint_inclusive(rng, cfg["blocks_per_stage_range"])
        base_channels = int(_choose(rng, cfg["base_channels_choices"], name="cnn.base_channels_choices"))
        channel_mult = int(_choose(rng, cfg["channel_mult_choices"], name="cnn.channel_mult_choices"))
        downsample = str(_choose(rng, cfg["downsample_choices"], name="cnn.downsample_choices"))
        double_conv_p = float(_choose(rng, cfg["double_conv_p_choices"], name="cnn.double_conv_p_choices"))

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
        return model_cfg

    def _sample_input_spec(
        self,
        rng: random.Random,
    ) -> Dict[str, Any]:
        cfg = self.config["input_spec"]
        kinds = list(cfg["kind_choices"])
        if len(kinds) == 1:
            kind = kinds[0]
        else:
            has_box = "BOX" in kinds
            has_linf = "LINF_BALL" in kinds
            if has_box and has_linf and len(kinds) == 2:
                kind = "BOX" if (rng.random() < float(cfg["p_box"])) else "LINF_BALL"
            else:
                kind = rng.choice(kinds)

        value_range = _choose(rng, cfg["value_range_choices"], name="input_spec.value_range_choices")
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
            eps = float(_choose(rng, cfg["eps_choices"], name="input_spec.eps_choices"))
            eps = min(eps, 0.5 * (hi - lo)) if (hi > lo) else 0.0

            return {
                "kind": "LINF_BALL",
                "value_range": (lo, hi),
                "center_val": float(center_val),
                "eps": float(eps),
            }

        raise ValueError(f"Unsupported input_spec kind '{kind}'")

    def _sample_output_spec(
        self,
        rng: random.Random,
        *,
        num_classes: int,
    ) -> Dict[str, Any]:
        cfg = self.config["output_spec"]
        kinds = list(cfg["kind_choices"])
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
            margin = float(_choose(rng, cfg["margin_choices"], name="output_spec.margin_choices"))
            return {"kind": "MARGIN_ROBUST", "y_true": y_true, "margin": float(margin)}

        if kind == "LINEAR_LE":
            c_lo, c_hi = cfg["linear_le_c_range"]
            d_lo, d_hi = cfg["linear_le_d_range"]
            c_vals = [c_lo + (c_hi - c_lo) * rng.random() for _ in range(int(num_classes))]
            d_val = d_lo + (d_hi - d_lo) * rng.random()
            return {"kind": "LINEAR_LE", "c": [float(x) for x in c_vals], "d": float(d_val)}

        if kind == "RANGE":
            lo, hi = _choose(rng, cfg["range_choices"], name="output_spec.range_choices")
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

    def _build_network_spec(self, instance: Dict[str, Any], *, dtype: str) -> Dict[str, Any]:
        model_cfg = instance["model_cfg"]
        input_shape = list(model_cfg["input_shape"])
        num_classes = int(model_cfg["num_classes"])

        layers: List[Dict[str, Any]] = []

        input_meta: Dict[str, Any] = {
            "shape": input_shape,
            "dtype": str(dtype),
            "num_classes": num_classes,
            "value_range": list(instance["input_spec"]["value_range"]),
        }
        layers.append({"kind": "INPUT", "params": {}, "meta": input_meta})

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

        if instance["family"] == "mlp":
            _build_mlp_layers(layers, cfg=model_cfg)
        elif instance["family"] == "cnn2d":
            rng = random.Random(int(instance["seed"]))
            _build_cnn_layers(layers, cfg=model_cfg, rng=rng)
        else:
            raise ValueError(f"Unsupported model family: {instance['family']}")

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

    def _sample_instance(self, idx: int) -> Dict[str, Any]:
        gen = self.config["generator"]
        instance_id = f"{self.name_prefix}{int(self.base_seed)}_idx{int(idx):05d}"
        seed = int(_derive_seed(int(self.base_seed), int(idx), instance_id))
        rng = random.Random(seed)

        family = self._sample_family(rng)
        num_classes = int(_choose(rng, gen["num_classes_choices"], name="generator.num_classes_choices"))

        if family == "mlp":
            model_cfg = self._sample_mlp(rng, num_classes=num_classes)
        elif family == "cnn2d":
            model_cfg = self._sample_cnn2d(rng, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown family: {family}")

        input_spec = self._sample_input_spec(rng)
        output_spec = self._sample_output_spec(rng, num_classes=num_classes)

        return {
            "instance_id": instance_id,
            "seed": seed,
            "family": family,
            "model_cfg": model_cfg,
            "input_spec": input_spec,
            "output_spec": output_spec,
        }

    def create_network(self, name: str, spec: Dict[str, Any]) -> Net:
        dtype = get_default_dtype()
        dtype_str = str(dtype)

        layers = []
        var_counter = 0

        layer_specs = list(spec["layers"])
        for i, layer_spec in enumerate(layer_specs):
            params = layer_spec.get("params", {}).copy()
            meta = layer_spec.get("meta", {}).copy()
            kind = layer_spec["kind"]

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

            in_vars, out_vars, var_counter = self._generate_layer_variables(kind, i, var_counter, meta, layers)

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
        net.meta = {"name": name}
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
        gen = self.config["generator"]
        dtype = str(gen["dtype"])

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
