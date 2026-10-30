#!/usr/bin/env python3
# ===- act/pipeline/act2torch.py - ACT to Torch Converter ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# ACT → PyTorch Converter (Schema-Driven)
# =======================================
# REGISTRY (layer_schema.py) validates params; ACT_TO_TORCH maps LayerKind →
# torch.nn.Module for restoration.
#
# Layer Routing:
#   INPUT      → InputLayer
#   INPUT_SPEC → InputSpecLayer
#   <MODEL>    → ActGraphModule (arbitrary DAG restoration)
#   ASSERT     → OutputSpecLayer
#
# ===---------------------------------------------------------------------===#

from typing import Optional, Dict, Any, Tuple, cast
from typing import Set, List
import importlib
import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
import logging

from act.back_end.core import Net, Layer, Bounds, get_topo_order
from act.back_end.interval_tf.tf_transformer import softmax_rowsize
from act.back_end.layer_schema import ACT_TO_TORCH, LayerKind, REGISTRY
from act.util.device_manager import get_default_dtype, get_default_device

logger = logging.getLogger(__name__)


def _axis_shift(input_shape: Any, x: torch.Tensor) -> int:
    """Offset to map a stored slice/gather axis onto the live tensor's axes.

    Returns 0 when ``input_shape`` already spans every live axis (ONNX-style
    full shape) and 1 when the live tensor carries one extra leading dim the
    stored per-sample axes do not count.
    """
    if input_shape is not None and x.dim() == len(tuple(input_shape)):
        return 0
    return 1


def _align_elementwise_param(
    param: torch.Tensor, x: torch.Tensor, input_shape: Any = None,
) -> torch.Tensor:
    """Align a flat SCALE/BIAS param to ``x`` for elementwise combination.

    A param spanning the full per-sample tensor (e.g. a position embedding)
    is reshaped to ``x``'s per-sample shape. For a fixed structural leading
    dimension, a full-shape param follows the model tensor instead of a
    verifier lane; a trailing-feature param is returned unchanged.
    """
    if (
        input_shape is not None and len(input_shape) == x.dim()
        and int(input_shape[0]) != 1 and tuple(input_shape) == tuple(x.shape)
        and param.numel() == x.numel()
    ):
        return param.reshape_as(x)
    if x.dim() >= 2 and param.numel() == x[0].numel() and param.numel() != x.shape[-1]:
        return param.reshape(x.shape[1:])
    return param


class ACTErf(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.erf(x)


class ACTSqrt(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(x, min=0.0))


class ACTSin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class ACTCos(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x)


class ACTQuantize(nn.Module):
    def __init__(
        self,
        scale: object = None,
        zero_point: object = None,
        qmin: int = 0,
        qmax: int = 255,
    ) -> None:
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(1.0 if scale is None else scale))
        self.register_buffer(
            "zero_point", torch.as_tensor(0 if zero_point is None else zero_point)
        )
        self.qmin = float(qmin)
        self.qmax = float(qmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.get_buffer("scale").to(device=x.device, dtype=x.dtype)
        zero_point = self.get_buffer("zero_point").to(device=x.device, dtype=x.dtype)
        quantized = torch.clamp(
            torch.round(x / scale),
            min=self.qmin - zero_point,
            max=self.qmax - zero_point,
        )
        return scale * quantized


class ACTMatMul(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y)


class ACTMaskAdd(nn.Module):
    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.get_buffer("mask").to(device=x.device, dtype=x.dtype)
        return x + mask


class ACTSoftmax(nn.Module):
    """Execute explicit contiguous rows using the analysis interpretation."""

    def __init__(self, layer: Layer) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row = softmax_rowsize(self.layer, Bounds(x, x))
        assert row is not None
        return x.reshape(x.shape[0], -1, row).softmax(-1).reshape_as(x)


class ACTLayerNorm(nn.Module):
    def __init__(
        self,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        eps: float,
        variant: str,
    ) -> None:
        super().__init__()
        self.register_buffer("gamma", gamma.detach().clone())
        self.register_buffer("beta", beta.detach().clone())
        self.eps = eps
        self.variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.variant == "no":
            return x
        gamma = self.get_buffer("gamma").to(device=x.device, dtype=x.dtype)
        beta = self.get_buffer("beta").to(device=x.device, dtype=x.dtype)
        if self.variant == "no_var":
            width = gamma.numel()
            features = x.shape[1:].numel()
            if width == 0 or features % width:
                raise ValueError("LayerNorm gamma width must divide the input width")
            rows = x.reshape(x.shape[0], -1, width)
            centered = rows - rows.mean(dim=-1, keepdim=True)
            return (centered * gamma.flatten() + beta.flatten()).reshape_as(x)
        import torch.nn.functional as F

        return F.layer_norm(x, gamma.shape, weight=gamma, bias=beta, eps=self.eps)


class ACTMHASplit(nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        role: str,
        position: int,
        feature: int,
        hidden_size: Optional[int],
    ) -> None:
        super().__init__()
        self.register_buffer(
            "weight", weight.detach().clone() if isinstance(weight, torch.Tensor) else None
        )
        self.register_buffer(
            "bias", bias.detach().clone() if isinstance(bias, torch.Tensor) else None
        )
        self.role = role
        self.position = position
        self.feature = feature
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.get_buffer("weight")
        if weight is None:
            return x
        import torch.nn.functional as F

        weight = weight.to(device=x.device, dtype=x.dtype)
        bias = self.get_buffer("bias")
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        hidden = self.hidden_size or int(weight.shape[1])
        if hidden != weight.shape[1]:
            raise ValueError("MHA_SPLIT hidden_size must match the projection input width")
        projected = F.linear(x.reshape(x.shape[0], -1, hidden), weight, bias)
        if self.role in {"query", "key"}:
            return projected[:, self.position, :]
        if self.role == "value":
            return projected[:, :, self.feature]
        return projected


class ACTAttentionScores(nn.Module):
    def __init__(self, dk: float, mask: Optional[torch.Tensor]) -> None:
        super().__init__()
        self.dk = dk
        self.register_buffer(
            "mask", mask.detach().clone() if isinstance(mask, torch.Tensor) else None
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        output = (query * key).sum(dim=-1, keepdim=True) / self.dk
        mask = self.get_buffer("mask")
        if mask is not None:
            output = output + mask.to(device=output.device, dtype=output.dtype)
        return output


class ACTAttentionMix(nn.Module):
    def forward(self, weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return (weights * values).sum(dim=-1, keepdim=True)


class ACTMHAJoin(nn.Module):
    def __init__(self, sequence_length: int, hidden_size: Optional[int]) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        output = torch.cat([value.flatten(1) for value in inputs], dim=1)
        hidden_size = self.hidden_size or int(output.shape[1] // self.sequence_length)
        return output.reshape(output.shape[0], self.sequence_length, hidden_size)


class ActGraphModule(nn.Module):
    """DAG-aware nn.Module for ACT body graphs reconstructed by ACTToTorch.

    Holds the body sub-graph between INPUT_SPEC and ASSERT. Performs forward by
    topological-order traversal, caching activations per layer id. Handles:
      - Module-backed layers (DENSE, CONV2D, RELU, ...): single-input nn.Module call.
      - Functional layers (ADD, CONCAT, MUL, SCALE, BIAS): inline tensor ops; tensor
        constants (SCALE["a"], BIAS["c"]) registered as buffers for .to() propagation.
      - BN-fused pairs (SCALE + BIAS with is_batchnorm_decomposition): SCALE owns the
        reconstructed BatchNorm module; BIAS's activation aliases SCALE's so successors
        reading the BIAS id get the fused BN output.
    """

    def __init__(
        self,
        act_net,
        topo_order,
        layer_modules,
        source_ids,
        exit_id,
        bn_aliases,
        body_preds,
    ):
        super().__init__()
        self.layers = nn.ModuleDict(
            {str(lid): mod for lid, mod in layer_modules.items() if mod is not None}
        )
        self._net = act_net
        self._topo = list(topo_order)
        self._sources = set(source_ids)
        self._exit = exit_id
        self._bn_aliases = dict(bn_aliases)
        self._body_preds = {k: list(v) for k, v in body_preds.items()}

        for lid in self._topo:
            layer = act_net.by_id[lid]
            if str(lid) in self.layers:
                continue
            if lid in self._bn_aliases:
                continue
            kind = layer.kind
            if kind == LayerKind.SCALE.value and isinstance(layer.params.get("a"), torch.Tensor):
                self.register_buffer(f"_scale_a_{lid}", layer.params["a"].detach().clone())
            elif kind == LayerKind.BIAS.value and isinstance(layer.params.get("c"), torch.Tensor):
                self.register_buffer(f"_bias_c_{lid}", layer.params["c"].detach().clone())
            elif kind == LayerKind.CONSTANT.value and isinstance(layer.params.get("value"), torch.Tensor):
                self.register_buffer(f"_const_value_{lid}", layer.params["value"].detach().clone())

    def forward(self, x):
        activations = {}
        for lid in self._topo:
            if lid in self._bn_aliases:
                activations[lid] = activations[self._bn_aliases[lid]]
                continue

            pred_sources = self._body_preds.get(lid, [])
            if pred_sources:
                inp_tensors = []
                for pred in pred_sources:
                    if pred is None:
                        inp_tensors.append(x)
                        continue
                    try:
                        inp_tensors.append(activations[pred])
                    except KeyError as e:
                        raise RuntimeError(
                            f"ActGraphModule: layer {lid} (kind={self._net.by_id[lid].kind}) "
                            f"missing predecessor activation for layer id {e.args[0]}. "
                            f"Check topological sort vs. body_preds."
                        )
            elif lid in self._sources:
                inp_tensors = [x]
            else:
                raise RuntimeError(
                    f"ActGraphModule: layer {lid} (kind={self._net.by_id[lid].kind}) "
                    f"has no recorded predecessors and is not a source node."
                )

            mod = self.layers[str(lid)] if str(lid) in self.layers else None
            layer = self._net.by_id[lid]

            if mod is None:
                out = self._apply_functional(layer, inp_tensors)
            else:
                if layer.kind == LayerKind.DENSE.value and layer.params.get("token_wise"):
                    out = mod(inp_tensors[0])
                elif layer.kind == LayerKind.DENSE.value and inp_tensors[0].dim() >= 3:
                    import torch.nn.functional as F
                    output_shape = layer.params.get("output_shape")
                    if output_shape is not None and len(output_shape) >= 3:
                        in_features = inp_tensors[0].shape[-1]
                        out_features = int(output_shape[-1])
                        weight = layer.params["weight"].to(
                            device=inp_tensors[0].device, dtype=inp_tensors[0].dtype
                        )[:out_features, :in_features]
                        bias_param = layer.params.get("bias")
                        bias = (
                            bias_param.to(device=inp_tensors[0].device, dtype=inp_tensors[0].dtype)[:out_features]
                            if isinstance(bias_param, torch.Tensor) else None
                        )
                        out = F.linear(inp_tensors[0], weight, bias)
                    else:
                        out = mod(inp_tensors[0])
                else:
                    out = mod(*inp_tensors)
                # nn.RNN / LSTM / GRU return (output, hidden); MHA returns
                # (output, attn_weights). Verification only consumes the
                # primary output tensor, so drop the auxiliary state.
                if isinstance(out, tuple):
                    out = out[0]
            activations[lid] = out

        if self._exit not in activations:
            raise RuntimeError(
                f"ActGraphModule: exit layer id {self._exit} has no activation. "
                f"Topological order or exit_id inconsistent with body graph."
            )
        return activations[self._exit]

    def _apply_functional(self, layer, inputs):
        kind = layer.kind
        if kind == LayerKind.ADD.value:
            if len(inputs) < 2:
                raise RuntimeError(
                    f"ActGraphModule: ADD layer {layer.id} expects at least 2 inputs, "
                    f"got {len(inputs)}."
                )
            out = inputs[0]
            for t in inputs[1:]:
                out = out + t
            return out
        if kind == LayerKind.CONCAT.value:
            if len(inputs) < 2:
                raise RuntimeError(
                    f"ActGraphModule: CONCAT layer {layer.id} expects at least 2 inputs, "
                    f"got {len(inputs)}."
                )
            return torch.cat(inputs, dim=layer.params.get("concat_dim", 1))
        if kind == LayerKind.MUL.value:
            if len(inputs) < 2:
                raise RuntimeError(
                    f"ActGraphModule: MUL layer {layer.id} expects at least 2 inputs, "
                    f"got {len(inputs)}."
                )
            out = inputs[0]
            for t in inputs[1:]:
                out = out * t
            return out
        if kind == LayerKind.SCALE.value:
            return inputs[0] * _align_elementwise_param(
                getattr(self, f"_scale_a_{layer.id}"), inputs[0],
                layer.params.get("input_shape"),
            )
        if kind == LayerKind.BIAS.value:
            return inputs[0] + _align_elementwise_param(
                getattr(self, f"_bias_c_{layer.id}"), inputs[0],
                layer.params.get("input_shape"),
            )
        if kind == LayerKind.TRANSPOSE.value:
            perm = layer.params.get("perm")
            if perm is None:
                return inputs[0]
            return inputs[0].permute(*perm).contiguous()
        if kind == LayerKind.UNSQUEEZE.value:
            dims = layer.params.get("dims") or []
            out = inputs[0]
            for d in sorted(dims):
                out = out.unsqueeze(d)
            return out
        if kind == LayerKind.SQUEEZE.value:
            dims = layer.params.get("dims") or []
            out = inputs[0]
            for d in sorted(dims, reverse=True):
                out = out.squeeze(d)
            return out
        if kind == LayerKind.RESHAPE.value:
            target = layer.params.get("target_shape") or layer.params.get("output_shape")
            if target is None:
                return inputs[0]
            input_shape = layer.params.get("input_shape")
            if input_shape is not None and (int(input_shape[0]) != 1 or int(target[0]) != 1):
                # A fixed-shape ONNX leading dimension belongs to the model,
                # not to the verifier's independent sample lanes.
                return inputs[0].reshape(tuple(int(d) for d in target))
            # Native templates carry one conceptual lane; preserve the live B.
            return inputs[0].reshape(inputs[0].shape[0], *target[1:])
        if kind == LayerKind.MAX.value:
            if len(inputs) < 2:
                raise RuntimeError(
                    f"ActGraphModule: MAX layer {layer.id} expects at least 2 inputs, "
                    f"got {len(inputs)}."
                )
            out = inputs[0]
            for t in inputs[1:]:
                out = torch.maximum(out, t)
            return out
        if kind == LayerKind.MIN.value:
            if len(inputs) < 2:
                raise RuntimeError(
                    f"ActGraphModule: MIN layer {layer.id} expects at least 2 inputs, "
                    f"got {len(inputs)}."
                )
            out = inputs[0]
            for t in inputs[1:]:
                out = torch.minimum(out, t)
            return out
        if kind == LayerKind.CONSTANT.value:
            val = getattr(self, f"_const_value_{layer.id}")
            target_shape = layer.params.get("output_shape") or layer.params.get("input_shape")
            if target_shape is not None:
                val = val.reshape(*target_shape)
            # A CONSTANT source receives x as inputs[0]; broadcast its leading
            # (batch) axis to the runtime batch so it concatenates/adds against
            # batched activations.
            if inputs and val.dim() >= 1 and val.shape[0] == 1:
                batch = inputs[0].shape[0]
                if batch != 1:
                    val = val.expand(batch, *val.shape[1:])
            return val.clone()
        if kind == LayerKind.SIGN.value:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"ActGraphModule: SIGN layer {layer.id} expects exactly 1 input, "
                    f"got {len(inputs)}."
                )
            return torch.sign(inputs[0])
        if kind == LayerKind.REDUCE_SUM.value:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"ActGraphModule: REDUCE_SUM layer {layer.id} expects exactly 1 input, "
                    f"got {len(inputs)}."
                )
            axes = layer.params.get("axes")
            keepdims = bool(layer.params.get("keepdims", 0))
            if axes is None:
                axes = tuple(range(inputs[0].dim()))
            return torch.sum(inputs[0], dim=tuple(int(a) for a in axes), keepdim=keepdims)
        if kind == LayerKind.COMPARE.value:
            if len(inputs) != 2:
                raise RuntimeError(
                    f"ActGraphModule: COMPARE layer {layer.id} expects exactly 2 inputs, "
                    f"got {len(inputs)}."
                )
            op = layer.params["op"]
            return getattr(torch, op)(inputs[0], inputs[1]).to(inputs[0].dtype)
        if kind == LayerKind.WHERE.value:
            if len(inputs) != 3:
                raise RuntimeError(
                    f"ActGraphModule: WHERE layer {layer.id} expects exactly 3 inputs, "
                    f"got {len(inputs)}."
                )
            return torch.where(inputs[0].bool(), inputs[1], inputs[2])
        if kind == LayerKind.MATMUL.value:
            if len(inputs) != 2:
                raise RuntimeError(
                    f"ActGraphModule: MATMUL layer {layer.id} expects exactly 2 inputs, "
                    f"got {len(inputs)}."
                )
            return torch.matmul(inputs[0], inputs[1])
        if kind == LayerKind.ARG_EXTREMUM.value:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"ActGraphModule: ARG_EXTREMUM layer {layer.id} expects exactly 1 input, "
                    f"got {len(inputs)}."
                )
            op = layer.params["op"]
            axis = int(layer.params.get("axis", 0))
            keepdims = bool(layer.params.get("keepdims", 0))
            fn = torch.argmax if op == "argmax" else torch.argmin
            return fn(inputs[0], dim=axis, keepdim=keepdims).to(inputs[0].dtype)
        if kind == LayerKind.UPSAMPLE.value:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"ActGraphModule: UPSAMPLE layer {layer.id} expects exactly 1 input, "
                    f"got {len(inputs)}."
                )
            import torch.nn.functional as F
            mode = str(layer.params.get("mode", "nearest")).lower()
            scale_factor = layer.params.get("scale_factor")
            size = layer.params.get("size")
            kwargs: Dict[str, Any] = {"mode": mode}
            if mode != "nearest" and layer.params.get("align_corners") is not None:
                kwargs["align_corners"] = bool(layer.params["align_corners"])
            if size is not None:
                kwargs["size"] = tuple(int(s) for s in size)
            elif scale_factor is not None:
                kwargs["scale_factor"] = tuple(float(s) for s in scale_factor)
            return F.interpolate(inputs[0], **kwargs)
        if kind == LayerKind.EXPAND.value:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"ActGraphModule: EXPAND layer {layer.id} expects exactly 1 input, "
                    f"got {len(inputs)}."
                )
            target = layer.params.get("output_shape") or layer.params.get("shape")
            if target is None:
                return inputs[0].clone()
            out_shape = torch.broadcast_shapes(
                inputs[0].shape, tuple(int(d) for d in target)
            )
            return inputs[0].broadcast_to(out_shape).clone()
        if kind == LayerKind.SCATTER_ND.value:
            if len(inputs) != 3:
                raise RuntimeError(
                    f"ActGraphModule: SCATTER_ND layer {layer.id} expects exactly 3 inputs, "
                    f"got {len(inputs)}."
                )
            data, idx, upd = inputs
            out = data.clone()
            idx_long = idx.long()
            if idx_long.dim() == 1:
                idx_long = idx_long.unsqueeze(-1)
            indices_per_dim = tuple(idx_long[..., d] for d in range(idx_long.shape[-1]))
            out.index_put_(indices_per_dim, upd, accumulate=False)
            return out
        if kind == LayerKind.PAD.value:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"ActGraphModule: PAD layer {layer.id} expects exactly 1 input, "
                    f"got {len(inputs)}."
                )
            import torch.nn.functional as F
            pad = layer.params.get("pad")
            if pad is None:
                pad = layer.params.get("pads")
            if pad is None:
                return inputs[0].clone()
            pad_list = [int(p) for p in pad]
            mode = str(layer.params.get("mode", "constant")).lower()
            if mode == "constant":
                value = float(layer.params.get("value", 0.0))
                return F.pad(inputs[0], pad_list, mode=mode, value=value)
            return F.pad(inputs[0], pad_list, mode=mode)
        if kind == LayerKind.MASK_ADD.value:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"ActGraphModule: MASK_ADD layer {layer.id} expects exactly 1 input, "
                    f"got {len(inputs)}."
                )
            M = layer.params["M"].to(
                device=inputs[0].device, dtype=inputs[0].dtype
            )
            return inputs[0] + M
        if kind == LayerKind.MHA_SPLIT.value:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"ActGraphModule: MHA_SPLIT layer {layer.id} expects exactly 1 input, "
                    f"got {len(inputs)}."
                )
            import torch.nn.functional as F
            x = inputs[0]
            weight = layer.params["weight"].to(device=x.device, dtype=x.dtype)
            bias_param = layer.params.get("bias")
            bias = bias_param.to(device=x.device, dtype=x.dtype) if isinstance(bias_param, torch.Tensor) else None
            if x.dim() >= 3:
                projected = F.linear(x.reshape(x.shape[0], -1, weight.shape[1]), weight, bias)
            else:
                projected = F.linear(x.reshape(x.shape[0], -1, weight.shape[1]), weight, bias)
            role = str(layer.params.get("role", ""))
            if role in {"query", "key"}:
                position = int(layer.params.get("position", 0))
                hidden = int(layer.params.get("hidden_size", projected.shape[-1]))
                seq_len = projected.shape[1] if projected.dim() >= 3 else max(projected.shape[-1] // max(hidden, 1), 1)
                projected = projected.reshape(x.shape[0], seq_len, hidden)
                return projected[:, position, :]
            if role == "value":
                feature = int(layer.params.get("feature", 0))
                hidden = int(layer.params.get("hidden_size", projected.shape[-1]))
                seq_len = projected.shape[1] if projected.dim() >= 3 else max(projected.shape[-1] // max(hidden, 1), 1)
                projected = projected.reshape(x.shape[0], seq_len, hidden)
                return projected[:, :, feature]
            return projected
        if kind == LayerKind.ATT_SCORES.value:
            if len(inputs) != 2:
                raise RuntimeError(
                    f"ActGraphModule: ATT_SCORES layer {layer.id} expects exactly 2 inputs, "
                    f"got {len(inputs)}."
                )
            scale = float(layer.params["dk"])
            return (inputs[0] * inputs[1]).sum(dim=-1, keepdim=True) / scale
        if kind == LayerKind.ATT_MIX.value:
            if len(inputs) != 2:
                raise RuntimeError(
                    f"ActGraphModule: ATT_MIX layer {layer.id} expects exactly 2 inputs, "
                    f"got {len(inputs)}."
                )
            return (inputs[0] * inputs[1]).sum(dim=-1, keepdim=True)
        if kind == LayerKind.MHA_JOIN.value:
            if not inputs:
                raise RuntimeError(f"ActGraphModule: MHA_JOIN layer {layer.id} expects inputs.")
            out = torch.cat(inputs, dim=-1)
            seq_len = int(layer.params.get("seq_len", 1))
            hidden = int(layer.params.get("hidden_size", out.shape[-1]))
            if seq_len == 1:
                return out.reshape(out.shape[0], 1, hidden)
            return out.reshape(out.shape[0], seq_len, hidden)
        if kind == LayerKind.POSENC.value:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"ActGraphModule: POSENC layer {layer.id} expects exactly 1 input, "
                    f"got {len(inputs)}."
                )
            P = layer.params["pos_vec"].to(
                device=inputs[0].device, dtype=inputs[0].dtype
            )
            return inputs[0] + P
        if kind == LayerKind.LAYERNORM.value:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"ActGraphModule: LAYERNORM layer {layer.id} expects exactly 1 input, "
                    f"got {len(inputs)}."
                )
            import torch.nn.functional as F
            gamma = layer.params["gamma"].to(
                device=inputs[0].device, dtype=inputs[0].dtype
            )
            beta = layer.params["beta"].to(
                device=inputs[0].device, dtype=inputs[0].dtype
            )
            eps = float(layer.params.get("eps", 1e-5))
            return F.layer_norm(
                inputs[0], gamma.shape, weight=gamma, bias=beta, eps=eps
            )
        if kind == LayerKind.SQUARE.value:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"ActGraphModule: SQUARE layer {layer.id} expects exactly 1 input, "
                    f"got {len(inputs)}."
                )
            return inputs[0] * inputs[0]
        if kind == LayerKind.POWER.value:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"ActGraphModule: POWER layer {layer.id} expects exactly 1 input, "
                    f"got {len(inputs)}."
                )
            p = float(layer.params["p"])
            return torch.pow(torch.clamp(inputs[0], min=0.0), p)
        if kind == LayerKind.SUB.value:
            if len(inputs) != 2:
                raise RuntimeError(
                    f"ActGraphModule: SUB layer {layer.id} expects exactly 2 inputs, "
                    f"got {len(inputs)}."
                )
            return inputs[0] - inputs[1]
        if kind == LayerKind.DIV.value:
            if len(inputs) != 2:
                raise RuntimeError(
                    f"ActGraphModule: DIV layer {layer.id} expects exactly 2 inputs, "
                    f"got {len(inputs)}."
                )
            return inputs[0] / inputs[1]
        if kind == LayerKind.ABS.value:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"ActGraphModule: ABS layer {layer.id} expects exactly 1 input, "
                    f"got {len(inputs)}."
                )
            return torch.abs(inputs[0])
        if kind == LayerKind.BN.value:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"ActGraphModule: BN layer {layer.id} expects exactly 1 input, "
                    f"got {len(inputs)}."
                )
            A = layer.params["A"].to(device=inputs[0].device, dtype=inputs[0].dtype)
            c = layer.params["c"].to(device=inputs[0].device, dtype=inputs[0].dtype)
            return A * inputs[0] + c
        if kind == LayerKind.SLICE.value:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"ActGraphModule: SLICE layer {layer.id} expects exactly 1 input, "
                    f"got {len(inputs)}."
                )
            starts = layer.params["starts"]
            ends = layer.params["ends"]
            axes = layer.params.get("axes")
            steps = layer.params.get("steps")
            x = inputs[0]
            if axes is None:
                axes = list(range(len(starts)))
            if steps is None:
                steps = [1] * len(starts)
            # Axis indices may be per-sample (excluding batch) or over the full
            # input_shape (which already includes a leading dim). Shift by +1
            # only when the live tensor carries an extra leading axis.
            shift = _axis_shift(layer.params.get("input_shape"), x)
            for s, e, a, st in zip(starts, ends, axes, steps):
                dim = int(a) + shift
                st = int(st)
                if st == 1:
                    x = x.narrow(dim, int(s), int(e) - int(s))
                else:
                    # narrow cannot express a stride, so a strided step must be
                    # realised by explicit indexing to match the analyzed Net.
                    end = min(int(e), x.shape[dim])
                    idx = torch.arange(int(s), end, st, device=x.device)
                    x = x.index_select(dim, idx)
            return x
        if kind == LayerKind.GATHER.value:
            if len(inputs) != 1:
                raise RuntimeError(
                    f"ActGraphModule: GATHER layer {layer.id} expects exactly 1 input, "
                    f"got {len(inputs)}."
                )
            indices = layer.params["indices"]
            axis = int(layer.params.get("axis", 0))
            # Same axis convention as SLICE (see _axis_shift).
            shift = _axis_shift(layer.params.get("input_shape"), inputs[0])
            idx = torch.as_tensor(indices, dtype=torch.long, device=inputs[0].device)
            return torch.index_select(inputs[0], dim=axis + shift, index=idx)
        if kind == LayerKind.MEAN.value:
            keepdim = bool(layer.params.get("keepdim", 0))
            shift = _axis_shift(layer.params.get("input_shape"), inputs[0])
            dims = [int(d) + shift for d in layer.params["dim"]]
            return inputs[0].mean(dim=dims, keepdim=keepdim)
        raise NotImplementedError(
            f"ActGraphModule: functional layer kind '{kind}' (id={layer.id}) not supported."
        )


class ACTToTorch:
    """
    Convert ACT Net to PyTorch nn.Module using dynamic restoration.

    Usage:
        converter = ACTToTorch(act_net)
        model = converter.run()  # Returns VerifiableModel
    """

    def __init__(self, act_net: Net):
        """
        Initialize converter with ACT Net.

        Args:
            act_net: ACT Net object (contains architecture + weights)

        Raises:
            TypeError: If act_net is not a Net instance
        """
        if not isinstance(act_net, Net):
            raise TypeError(f"ACTToTorch expects a Net object, got {type(act_net)}")
        self.act_net = act_net

    def run(self) -> nn.Module:
        """
        Convert ACT Net to PyTorch nn.Module.

        Iterates through ACT layers, creates corresponding PyTorch layers,
        transfers weights, and assembles into VerifiableModel model.

        Returns:
            VerifiableModel model with embedded constraint checking

        Raises:
            ValueError: If ACT Net invariants are violated or restoration fails
        """
        from act.front_end.verifiable_model import (
            InputLayer,
            InputSpecLayer,
            OutputSpecLayer,
            VerifiableModel,
        )
        from act.front_end.spec_creator_base import LabeledInputTensor
        from act.front_end.specs import InputSpec, InKind, OutputSpec, OutKind

        target_dtype = get_default_dtype()
        target_device = get_default_device()

        def _to_target_float_tensor(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(dtype=target_dtype, device=target_device)
            return value

        def _to_target_tensor(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.to(device=target_device)
            return value

        input_act = None
        input_spec_acts = []
        assert_act = None
        body_acts = []

        for idx, act_layer in enumerate(self.act_net.layers):
            if act_layer.kind == LayerKind.INPUT.value:
                if input_act is not None:
                    raise ValueError(
                        f"ACTToTorch expects exactly one INPUT layer, found ids "
                        f"{input_act.id} and {act_layer.id}."
                    )
                input_act = act_layer
                continue

            if act_layer.kind == LayerKind.INPUT_SPEC.value:
                input_spec_acts.append(act_layer)
                continue

            if act_layer.kind == LayerKind.ASSERT.value:
                if assert_act is not None:
                    raise ValueError(
                        f"ACTToTorch expects exactly one ASSERT layer, found ids "
                        f"{assert_act.id} and {act_layer.id}."
                    )
                if idx != len(self.act_net.layers) - 1:
                    raise ValueError(
                        f"ASSERT layer {act_layer.id} must be the last ACT layer, "
                        f"but appears at position {idx} of {len(self.act_net.layers) - 1}."
                    )
                assert_act = act_layer
                continue

            body_acts.append(act_layer)

        if input_act is None:
            raise ValueError("ACTToTorch expects exactly one INPUT layer, found 0.")
        if not input_spec_acts:
            raise ValueError("ACTToTorch expects at least one INPUT_SPEC layer, found 0.")
        if assert_act is None:
            raise ValueError("ACTToTorch expects exactly one ASSERT layer, found 0.")

        ip = input_act.params
        shape_value = ip.get("shape")
        if not isinstance(shape_value, (list, tuple)):
            raise ValueError(
                f"ACTToTorch: INPUT layer {input_act.id} has invalid shape param {shape_value!r}."
            )
        shape = tuple(shape_value)

        dv = ip.get("dtype")
        if isinstance(dv, str):
            dname = dv.split(".", 1)[1] if dv.startswith("torch.") else dv
            try:
                torch_dtype = getattr(torch, dname)
            except AttributeError as exc:
                raise ValueError(
                    f"ACTToTorch: INPUT layer {input_act.id} has unknown dtype {dv!r}."
                ) from exc
        elif isinstance(dv, torch.dtype):
            torch_dtype = dv
        else:
            raise ValueError(
                f"ACTToTorch: INPUT layer {input_act.id} has invalid dtype param {dv!r}."
            )

        labeled_input = ip.get("labeled_input")
        if labeled_input is None:
            dummy = torch.zeros(shape, dtype=torch_dtype, device=target_device)
            labeled_input = LabeledInputTensor(tensor=dummy, label=None)
        elif isinstance(labeled_input, LabeledInputTensor):
            labeled_input = LabeledInputTensor(
                tensor=labeled_input.tensor.to(device=target_device, dtype=torch_dtype),
                label=labeled_input.label.to(device=target_device)
                if labeled_input.label is not None
                else None,
            )
        else:
            raise ValueError(
                f"ACTToTorch: INPUT layer {input_act.id} has invalid labeled_input type "
                f"{type(labeled_input).__name__}."
            )

        optional_kwargs = {}
        for key in (
            "desc",
            "layout",
            "dataset_name",
            "num_classes",
            "value_range",
            "scale_hint",
            "distribution",
            "sample_id",
            "domain",
            "channels",
        ):
            if key in ip:
                optional_kwargs[key] = ip[key]

        input_layer_mod = InputLayer(
            labeled_input=labeled_input,
            shape=shape,
            dtype=torch_dtype,
            **optional_kwargs,
        )

        # Multiple INPUT_SPEC layers are allowed (the verifier handles each
        # spec independently via add_all_input_specs). The PyTorch model
        # only needs ONE spec for concrete input-satisfaction checks in
        # find_concrete_counterexample; pick the BOX/LINF_BALL seed using
        # the same priority as seed_from_input_specs. LIN_POLY refinements
        # are kept on the ACT Net for the symbolic verifier path; at the
        # concrete inference level they are not enforced (any sampled x
        # outside the polytope simply fails to be a true counterexample to
        # the joint spec, which is benign for soundness).
        seed_spec_act = next(
            (L for L in input_spec_acts
             if L.params.get("kind") in ("BOX", "LINF_BALL", "LP_EMBEDDING")),
            None,
        )
        if seed_spec_act is None:
            raise ValueError(
                f"ACTToTorch: at least one BOX, LINF_BALL, or LP_EMBEDDING INPUT_SPEC required "
                f"for PyTorch model reconstruction; got kinds="
                f"{[L.params.get('kind') for L in input_spec_acts]}."
            )

        input_spec_params = seed_spec_act.params
        input_kind_str = input_spec_params.get("kind")
        if not isinstance(input_kind_str, str):
            raise ValueError(
                f"ACTToTorch: INPUT_SPEC layer {seed_spec_act.id} has invalid kind "
                f"param {input_kind_str!r}."
            )
        try:
            input_spec_kind = getattr(InKind, input_kind_str)
        except AttributeError as exc:
            raise ValueError(
                f"ACTToTorch: INPUT_SPEC layer {seed_spec_act.id} has unknown kind "
                f"{input_kind_str!r}."
            ) from exc
        input_spec_dict = {"kind": input_spec_kind}
        if "eps" in input_spec_params:
            input_spec_dict["eps"] = _to_target_float_tensor(input_spec_params["eps"])
        if input_spec_kind == InKind.LP_EMBEDDING:
            if "p_norm" in input_spec_params:
                input_spec_dict["p_norm"] = _to_target_float_tensor(input_spec_params["p_norm"])
            elif "p_norm" in seed_spec_act.cache:
                input_spec_dict["p_norm"] = _to_target_float_tensor(seed_spec_act.cache["p_norm"])
            if "perturbed_positions" in input_spec_params:
                input_spec_dict["perturbed_positions"] = _to_target_tensor(
                    input_spec_params["perturbed_positions"]
                )
            elif "perturbed_positions" in seed_spec_act.cache:
                input_spec_dict["perturbed_positions"] = _to_target_tensor(
                    seed_spec_act.cache["perturbed_positions"]
                )
        for param_key in ["lb", "ub", "center", "A", "b"]:
            if param_key in input_spec_params:
                input_spec_dict[param_key] = _to_target_float_tensor(
                    input_spec_params[param_key]
                )
        input_spec_mod = InputSpecLayer(InputSpec(**input_spec_dict))

        body_ids = {layer.id for layer in body_acts}
        for layer in body_acts:
            missing = [pred for pred in self.act_net.preds.get(layer.id, [])
                       if pred not in self.act_net.by_id]
            if missing:
                raise ValueError(
                    f"ACTToTorch: layer {layer.id} references missing predecessors {missing}"
                )
        body_preds = {
            lid: [p if p in body_ids else None for p in self.act_net.preds.get(lid, [])]
            for lid in body_ids
        }
        body_succs = {
            lid: [s for s in self.act_net.succs.get(lid, []) if s in body_ids]
            for lid in body_ids
        }

        # Scheduling deduplicates dependencies; numerical inputs keep their
        # original order and multiplicity (e.g. h + h reads h twice).
        topo_order = [lid for lid in get_topo_order(self.act_net) if lid in body_ids]

        layer_modules: Dict[int, Optional[nn.Module]] = {}
        skip_ids: Set[int] = set()
        bn_aliases: Dict[int, int] = {}
        layer_indices = {layer.id: idx for idx, layer in enumerate(self.act_net.layers)}

        for act_layer in body_acts:
            if act_layer.kind != LayerKind.SCALE.value:
                continue
            if not act_layer.params.get("is_batchnorm_decomposition"):
                continue

            scale_id = act_layer.id
            bias_id = None
            for succ_id in body_succs.get(scale_id, []):
                succ_layer = self.act_net.by_id[succ_id]
                if succ_layer.kind == LayerKind.BIAS.value and succ_layer.params.get(
                    "paired_with_scale"
                ):
                    bias_id = succ_layer.id
                    break
            if bias_id is None:
                bias_layer = self._find_paired_bias(layer_indices[scale_id])
                if bias_layer is not None and bias_layer.id in body_ids:
                    bias_id = bias_layer.id
            if bias_id is None:
                raise ValueError(
                    f"ACTToTorch: SCALE layer {scale_id} is marked as batchnorm decomposition "
                    f"but no paired BIAS layer was found."
                )

            bn_module = self._restore_batchnorm(act_layer)
            if bn_module is None:
                raise ValueError(
                    f"ACTToTorch: failed to restore BatchNorm module from SCALE layer {scale_id}."
                )

            skip_ids.add(bias_id)
            bn_aliases[bias_id] = scale_id
            layer_modules[scale_id] = bn_module

        for act_layer in body_acts:
            lid = act_layer.id
            if lid in skip_ids or lid in layer_modules:
                continue
            if act_layer.kind in {
                LayerKind.SCALE.value,
                LayerKind.BIAS.value,
                LayerKind.ADD.value,
                LayerKind.CONCAT.value,
                LayerKind.MUL.value,
            }:
                layer_modules[lid] = None
            else:
                layer_modules[lid] = self._build_from_schema(act_layer)

        assert_preds = self.act_net.preds.get(assert_act.id, [])
        if len(assert_preds) != 1:
            raise ValueError(
                f"ACTToTorch: ASSERT layer {assert_act.id} must have exactly one predecessor; "
                f"found {assert_preds}."
            )
        exit_id = assert_preds[0]
        if exit_id is None or exit_id not in body_ids:
            raise ValueError(
                f"ACTToTorch: ASSERT layer {assert_act.id} must consume a body-layer output; "
                f"got predecessor {exit_id}."
            )

        source_ids = {
            lid for lid in body_ids
            if all(p is None for p in body_preds[lid])
        }
        inner_model = ActGraphModule(
            act_net=self.act_net,
            topo_order=topo_order,
            layer_modules=layer_modules,
            source_ids=source_ids,
            exit_id=exit_id,
            bn_aliases=bn_aliases,
            body_preds=body_preds,
        )
        inner_model = inner_model.to(device=target_device, dtype=torch_dtype)

        output_params = assert_act.params
        output_kind_str = output_params.get("kind")
        if not isinstance(output_kind_str, str):
            raise ValueError(
                f"ACTToTorch: ASSERT layer {assert_act.id} has invalid kind param "
                f"{output_kind_str!r}."
            )
        try:
            output_spec_kind = getattr(OutKind, output_kind_str)
        except AttributeError as exc:
            raise ValueError(
                f"ACTToTorch: ASSERT layer {assert_act.id} has unknown kind {output_kind_str!r}."
            ) from exc
        # Batch-native handoff: pass the on-disk batched ASSERT params straight
        # through to OutputSpec / OutputSpecLayer. The leading B axis is the
        # same B that the model forward consumes, so no per-sample stripping is
        # needed. On-disk shapes (from OutputSpec.encode_linear):
        #   LINEAR_LE     c: [B, n_out]      d: [B]
        #   UNSAFE_LINEAR c: [B, N, n_out]   d: [B, N]
        #   TOP1_ROBUST   y_true: [B]
        #   MARGIN_ROBUST y_true: [B]        margin: [B]
        #   RANGE         lb: [B, n_out]     ub: [B, n_out]
        # OutputSpecLayer.forward dispatches on dim() to handle both these
        # batched shapes and the single-sample shapes still emitted by the
        # VNNLIB and TorchVision front-end loaders.
        output_spec_dict = {"kind": output_spec_kind}
        if "y_true" in output_params:
            output_spec_dict["y_true"] = _to_target_tensor(
                output_params["y_true"]
            )
        if "margin" in output_params:
            output_spec_dict["margin"] = _to_target_float_tensor(
                output_params["margin"]
            )
        if "d" in output_params:
            output_spec_dict["d"] = _to_target_float_tensor(output_params["d"])
        for param_key in ["c", "lb", "ub"]:
            if param_key in output_params:
                output_spec_dict[param_key] = _to_target_float_tensor(
                    output_params[param_key]
                )
        output_spec_mod = OutputSpecLayer(OutputSpec(**output_spec_dict))

        model = VerifiableModel(
            input_layer=input_layer_mod,
            input_spec=input_spec_mod,
            model=inner_model,
            output_spec=output_spec_mod,
        )
        model.eval()
        logger.info(
            f"Reconstructed VerifiableModel: body has {len(body_acts)} layer(s), "
            f"inner_model type={type(inner_model).__name__}, "
            f"INPUT_SPEC={len(input_spec_acts)}, BN_fused={len(bn_aliases)}."
        )
        return model

    def _find_paired_bias(self, scale_idx: int) -> Optional[Layer]:
        """Find BIAS layer paired with SCALE at given index."""
        layers = self.act_net.layers
        for j in range(scale_idx + 1, len(layers)):
            layer = layers[j]
            if layer.kind == LayerKind.BIAS.value and layer.params.get(
                "paired_with_scale"
            ):
                return layer
            # Stop if we hit a non-BIAS layer
            if layer.kind != "BIAS":
                break
        return None

    def _restore_batchnorm(self, scale_layer: Layer) -> Optional[nn.Module]:
        """Restore BatchNorm from SCALE layer with batchnorm_* params."""
        params = scale_layer.params

        bn_module_path = cast(str, params.get("batchnorm_module"))
        if not bn_module_path:
            return None

        # Parse module path
        mod_name, cls_name = bn_module_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(mod_name), cls_name)

        # Create BatchNorm instance
        args = cast(List[Any], params.get("batchnorm_args", []))
        kwargs = cast(Dict[str, Any], params.get("batchnorm_kwargs", {}))
        bn = cls(*args, **kwargs)

        # Load state from batchnorm_state
        bn_state = cast(Dict[str, Any], params.get("batchnorm_state", {}))
        if bn_state:
            state_dict = {}
            for key in [
                "weight",
                "bias",
                "running_mean",
                "running_var",
                "num_batches_tracked",
            ]:
                if key in bn_state:
                    state_dict[key] = bn_state[key]
            if state_dict:
                bn.load_state_dict(state_dict, strict=False)

        return bn

    def _build_from_schema(self, act_layer: Layer) -> Optional[nn.Module]:
        """Build PyTorch module from REGISTRY params + ACT_TO_TORCH mapping."""
        kind = act_layer.kind
        params = act_layer.params

        if kind not in REGISTRY:
            raise ValueError(f"Layer kind '{kind}' not found in REGISTRY")
        spec = REGISTRY[kind]

        # Recurrent layers need bespoke construction (the schema's positional
        # arg order does not line up with nn.RNN's constructor — see
        # _build_rnn_family for the full reasoning).
        if kind in (LayerKind.RNN.value, LayerKind.GRU.value, LayerKind.LSTM.value):
            return self._build_rnn_family(act_layer)

        if kind == LayerKind.ERF.value:
            return ACTErf()
        if kind == LayerKind.SQRT.value:
            return ACTSqrt()
        if kind == LayerKind.SIN.value:
            return ACTSin()
        if kind == LayerKind.COS.value:
            return ACTCos()
        if kind == LayerKind.QUANTIZE.value:
            return ACTQuantize(
                scale=params.get("scale"),
                zero_point=params.get("zero_point"),
                qmin=int(cast(Any, params.get("qmin", 0))),
                qmax=int(cast(Any, params.get("qmax", 255))),
            )
        if kind == LayerKind.MATMUL.value:
            return ACTMatMul()
        if kind == LayerKind.MASK_ADD.value:
            return ACTMaskAdd(cast(torch.Tensor, params["M"]))
        if kind == LayerKind.SOFTMAX.value and "rowsize" in params:
            return ACTSoftmax(act_layer)
        if kind == LayerKind.LAYERNORM.value:
            variant = str(params.get("variant", params.get("layer_norm", "standard")))
            return ACTLayerNorm(
                gamma=cast(torch.Tensor, params["gamma"]),
                beta=cast(torch.Tensor, params["beta"]),
                eps=float(cast(Any, params.get("eps", 1e-5))),
                variant=variant,
            )
        if kind == LayerKind.MHA_SPLIT.value:
            hidden_size = params.get("hidden_size")
            return ACTMHASplit(
                weight=cast(Optional[torch.Tensor], params.get("weight")),
                bias=cast(Optional[torch.Tensor], params.get("bias")),
                role=str(params.get("role", "")),
                position=int(cast(Any, params.get("position", 0))),
                feature=int(cast(Any, params.get("feature", 0))),
                hidden_size=int(cast(Any, hidden_size)) if hidden_size is not None else None,
            )
        if kind == LayerKind.ATT_SCORES.value:
            return ACTAttentionScores(
                dk=float(cast(Any, params["dk"])),
                mask=cast(Optional[torch.Tensor], params.get("mask")),
            )
        if kind == LayerKind.ATT_MIX.value:
            return ACTAttentionMix()
        if kind == LayerKind.MHA_JOIN.value:
            hidden_size = params.get("hidden_size")
            return ACTMHAJoin(
                sequence_length=int(cast(Any, params.get("seq_len", 1))),
                hidden_size=int(cast(Any, hidden_size)) if hidden_size is not None else None,
            )

        cls = ACT_TO_TORCH.get(kind)
        if cls is None:
            if "requires_graph_restoration" in spec.get("params_optional", []):
                logger.warning(
                    f"Skipping {kind} layer (id={act_layer.id}): "
                    f"no direct nn.Module mapping; handled functionally by ActGraphModule"
                )
            return None

        # Build positional args from params_required (excluding tensors)
        # Tensors are auto-detected via isinstance() - they go to state_dict, not constructor
        args = []
        for key in spec.get("params_required", []):
            if key not in params:
                raise ValueError(
                    f"Layer '{kind}' (id={act_layer.id}) missing required param '{key}'"
                )
            # Skip tensor params - they go to state_dict
            if isinstance(params[key], torch.Tensor):
                continue
            args.append(params[key])

        # Build kwargs from params_optional (names match PyTorch directly)
        # Only include kwargs that PyTorch module accepts
        kwargs = {}
        # Check if bias exists in params to set bias kwarg for Linear/Conv.
        # CONVTRANSPOSE2D must be in the bias=False list: leaving its bias
        # at the nn.ConvTranspose2d default (True) materialises a randomly
        # initialised phantom bias that the abstract analysis has no record
        # of, breaking the per-neuron soundness invariant.
        if "bias" in act_layer.params:
            kwargs["bias"] = True
        elif kind in (
            LayerKind.DENSE.value,
            LayerKind.CONV1D.value,
            LayerKind.CONV2D.value,
            LayerKind.CONV3D.value,
            LayerKind.CONVTRANSPOSE2D.value,
        ):
            kwargs["bias"] = False
        # Pass through common kwargs from params
        for key in (
            "stride",
            "padding",
            "dilation",
            "groups",
            "start_dim",
        ):
            if key in params:
                kwargs[key] = params[key]
        if kind == LayerKind.AVGPOOL2D.value:
            for key in ("ceil_mode", "count_include_pad", "divisor_override"):
                if key in params:
                    kwargs[key] = params[key]
        if kind == LayerKind.GELU.value and "approximate" in params:
            kwargs["approximate"] = params["approximate"]

        # Create module instance
        m = cls(*args, **kwargs)

        # Load state_dict directly (param names already match PyTorch)
        if act_layer.params:
            state_dict = {}
            for key, value in act_layer.params.items():
                if not isinstance(value, torch.Tensor):
                    continue
                state_dict[key] = value

            if state_dict:
                m.load_state_dict(state_dict, strict=False)

        return m

    def _build_rnn_family(self, act_layer: Layer) -> nn.Module:
        """Build a single-layer nn.RNN / nn.LSTM / nn.GRU.
        """
        kind = act_layer.kind
        params = act_layer.params
        if int(cast(Any, params.get("num_layers", 1))) != 1:
            raise ValueError(
                f"ACTToTorch: {kind} layer {act_layer.id} has num_layers="
                f"{params['num_layers']}, only single-layer is supported."
            )

        ctor_kwargs: Dict[str, Any] = {
            "input_size":   int(cast(Any, params["input_size"])),
            "hidden_size":  int(cast(Any, params["hidden_size"])),
            "num_layers":   1,
            "bidirectional": bool(params.get("bidirectional", False)),
            "batch_first":  bool(params.get("batch_first", False)),
            "bias":         isinstance(params.get("bias_ih_l0"), torch.Tensor),
        }
        if kind == LayerKind.RNN.value:
            ctor_kwargs["nonlinearity"] = params.get("nonlinearity", "tanh")

        rnn = ACT_TO_TORCH[kind](**ctor_kwargs)
        target_dtype = next(rnn.parameters()).dtype
        sd = {k: v.detach().clone().to(dtype=target_dtype)
              for k, v in params.items() if isinstance(v, torch.Tensor)}
        rnn.load_state_dict(sd, strict=True)
        return rnn
