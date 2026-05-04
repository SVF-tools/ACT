# ===- act/back_end/hybridz_tf/hybridz_tf.py - HybridZ Transfer Function -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ Transfer Function Implementation.
#
#   Each hz_tf_* is a complete TF for one layer kind, combining
#   HZ zonotope propagation with interval_tf constraint generation.
#   hz_tf_* live in tf_mlp.py / tf_cnn.py alongside their layer types.
#   HZ domain ops co-locate with the hz_tf_* that use them.
#
# ===---------------------------------------------------------------------===#

""" """

import torch
from typing import Dict, Optional
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.transfer_functions import TransferFunction
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_hz import HZono, hz_from_bounds, hz_compute_bounds

import act.back_end.hybridz_tf.tf_mlp as hz_mlp
import act.back_end.hybridz_tf.tf_cnn as hz_cnn
import act.back_end.hybridz_tf.tf_rnn as hz_rnn
import act.back_end.hybridz_tf.tf_transformer as hz_transformer
import act.back_end.interval_tf.tf_mlp as interval_mlp
import act.back_end.interval_tf.tf_cnn as interval_cnn


class HybridzTF(TransferFunction):
    def __init__(self):
        self._hz_cache: Dict[int, HZono] = {}
        self._cache_net_id: Optional[int] = None
        self._tanh_K: int = 1
        self._sigmoid_K: int = 1

    _LAYER_REGISTRY = {
        # Identity / spec
        LayerKind.INPUT.value: lambda L, b, tf: Fact(bounds=b, cons=ConSet()),
        LayerKind.INPUT_SPEC.value: lambda L, b, tf: Fact(bounds=b, cons=ConSet()),
        LayerKind.ASSERT.value: lambda L, b, tf: Fact(bounds=b, cons=ConSet()),
        # MLP: HZ + interval
        LayerKind.DENSE.value: lambda L, b, tf: hz_mlp.tf_dense(L, b, tf),
        LayerKind.BIAS.value: lambda L, b, tf: hz_mlp.tf_bias(L, b, tf),
        LayerKind.SCALE.value: lambda L, b, tf: hz_mlp.tf_scale(L, b, tf),
        LayerKind.RELU.value: lambda L, b, tf: hz_mlp.tf_relu(L, b, tf),
        LayerKind.LRELU.value: lambda L, b, tf: hz_mlp.tf_lrelu(L, b, tf),
        LayerKind.TANH.value: lambda L, b, tf: hz_mlp.tf_tanh(L, b, tf),
        LayerKind.SIGMOID.value: lambda L, b, tf: hz_mlp.tf_sigmoid(L, b, tf),
        LayerKind.ABS.value: lambda L, b, tf: hz_mlp.tf_abs(L, b, tf),
        LayerKind.BN.value: lambda L, b, tf: hz_mlp.tf_bn(L, b, tf),
        # Multi-input: HZ + interval
        LayerKind.ADD.value: lambda L, b, tf: hz_mlp.tf_add(L, b, tf),
        LayerKind.SUB.value: lambda L, b, tf: hz_mlp.tf_sub(L, b, tf),
        LayerKind.MUL.value: lambda L, b, tf: hz_mlp.tf_mul(L, b, tf),
        LayerKind.CONCAT.value: lambda L, b, tf: hz_mlp.tf_concat(L, b, tf),
        # CNN: HZ + interval
        LayerKind.CONV2D.value: lambda L, b, tf: hz_cnn.tf_conv2d(L, b, tf),
        LayerKind.MAXPOOL2D.value: lambda L, b, tf: hz_cnn.tf_maxpool2d(L, b, tf),
        # Activations: interval-only
        LayerKind.CLIP.value: lambda L, b, tf: interval_mlp.tf_clip(L, b),
        LayerKind.SOFTPLUS.value: lambda L, b, tf: interval_mlp.tf_softplus(L, b),
        LayerKind.SILU.value: lambda L, b, tf: interval_mlp.tf_silu(L, b),
        LayerKind.RELU6.value: lambda L, b, tf: interval_mlp.tf_relu6(L, b),
        LayerKind.HARDTANH.value: lambda L, b, tf: interval_mlp.tf_hardtanh(L, b),
        LayerKind.HARDSIGMOID.value: lambda L, b, tf: interval_mlp.tf_hardsigmoid(L, b),
        LayerKind.HARDSWISH.value: lambda L, b, tf: interval_mlp.tf_hardswish(L, b),
        LayerKind.MISH.value: lambda L, b, tf: interval_mlp.tf_mish(L, b),
        LayerKind.SOFTSIGN.value: lambda L, b, tf: interval_mlp.tf_softsign(L, b),
        LayerKind.SQUARE.value: lambda L, b, tf: interval_mlp.tf_square(L, b),
        LayerKind.POWER.value: lambda L, b, tf: interval_mlp.tf_power(L, b),
        LayerKind.MAX.value: lambda L, b, tf: interval_mlp.tf_max(
            L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
        ),
        LayerKind.MIN.value: lambda L, b, tf: interval_mlp.tf_min(
            L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
        ),
        # CNN: interval-only
        LayerKind.AVGPOOL2D.value: lambda L, b, tf: interval_cnn.tf_avgpool2d(L, b),
        LayerKind.CONV1D.value: lambda L, b, tf: interval_cnn.tf_conv1d(L, b),
        LayerKind.CONV3D.value: lambda L, b, tf: interval_cnn.tf_conv3d(L, b),
        LayerKind.CONVTRANSPOSE2D.value: lambda L, b, tf: (
            interval_cnn.tf_convtranspose2d(L, b)
        ),
        LayerKind.FLATTEN.value: lambda L, b, tf: interval_cnn.tf_flatten(L, b),
        # Shape ops: interval-only
        LayerKind.RESHAPE.value: lambda L, b, tf: interval_mlp.tf_reshape(L, b),
        LayerKind.TRANSPOSE.value: lambda L, b, tf: interval_mlp.tf_transpose(L, b),
        LayerKind.SQUEEZE.value: lambda L, b, tf: interval_mlp.tf_squeeze(L, b),
        LayerKind.UNSQUEEZE.value: lambda L, b, tf: interval_mlp.tf_unsqueeze(L, b),
        LayerKind.TILE.value: lambda L, b, tf: interval_mlp.tf_tile(L, b),
        LayerKind.EXPAND.value: lambda L, b, tf: interval_mlp.tf_expand(L, b),
        # RNN
        LayerKind.LSTM.value: lambda L, b, tf: hz_rnn.tf_lstm(L, b, tf),
        LayerKind.GRU.value: lambda L, b, tf: hz_rnn.tf_gru(L, b, tf),
        LayerKind.RNN.value: lambda L, b, tf: hz_rnn.tf_rnn(L, b, tf),
        LayerKind.EMBEDDING.value: lambda L, b, tf: hz_rnn.tf_embedding(L, b, tf),
        LayerKind.EMBEDDING_TF.value: lambda L, b, tf: hz_rnn.tf_embedding(L, b, tf),
        # Transformer
        LayerKind.POSENC.value: lambda L, b, tf: hz_transformer.tf_posenc(L, b, tf),
        LayerKind.LAYERNORM.value: lambda L, b, tf: hz_transformer.tf_layernorm(L, b, tf),
        LayerKind.GELU.value: lambda L, b, tf: hz_transformer.tf_gelu(L, b, tf),
        LayerKind.ATT_SCORES.value: lambda L, b, tf: hz_transformer.tf_att_scores(L, b, tf),
        LayerKind.SOFTMAX.value: lambda L, b, tf: hz_transformer.tf_softmax(L, b, tf),
        LayerKind.ATT_MIX.value: lambda L, b, tf: hz_transformer.tf_att_mix(L, b, tf),
        LayerKind.MHA_SPLIT.value: lambda L, b, tf: hz_transformer.tf_mha_split(L, b, tf),
        LayerKind.MHA_JOIN.value: lambda L, b, tf: hz_transformer.tf_mha_join(L, b, tf),
        LayerKind.MASK_ADD.value: lambda L, b, tf: hz_transformer.tf_mask_add(L, b, tf),
    }

    @property
    def name(self) -> str:
        return "HybridzTF"

    def supports_layer(self, layer_kind: str) -> bool:
        return layer_kind.upper() in self._LAYER_REGISTRY

    _HZ_MAX_INPUT_DIM = 1024

    def _hz_from_bounds(self, bounds: Bounds) -> Optional[HZono]:
        lb, ub = bounds.lb.flatten(), bounds.ub.flatten()
        n = lb.shape[0]
        if n > self._HZ_MAX_INPUT_DIM:
            return None
        dtype, device = lb.dtype, lb.device
        c = ((lb + ub) / 2.0).view(-1, 1)
        rad = (ub - lb) / 2.0
        return HZono(
            c=c,
            Gc=torch.diag(rad),
            Gb=torch.zeros((n, 0), dtype=dtype, device=device),
            Ac=torch.zeros((0, n), dtype=dtype, device=device),
            Ab=torch.zeros((0, 0), dtype=dtype, device=device),
            b=torch.zeros((0, 1), dtype=dtype, device=device),
        )

    def apply(
        self,
        L: Layer,
        input_bounds: Bounds,
        net: Net,
        before: Dict[int, Fact],
        after: Dict[int, Fact],
    ) -> Fact:
        k = L.kind.upper()
        if k not in self._LAYER_REGISTRY:
            raise NotImplementedError(f"HybridzTF: Unsupported layer kind '{k}'")

        net_id = id(net)
        if self._cache_net_id != net_id:
            self._hz_cache.clear()
            self._cache_net_id = net_id

        self._net = net
        self._before = before
        self._after = after

        if k in ("INPUT", "INPUT_SPEC"):
            hz_init = self._hz_from_bounds(input_bounds)
            if hz_init is not None:
                self._hz_cache[L.id] = hz_init
        elif k != "ASSERT":
            preds = net.preds.get(L.id, [])
            if preds and preds[0] in self._hz_cache:
                self._hz_cache[L.id] = self._hz_cache[preds[0]]

        n_out = len(L.out_vars)
        if n_out >= self._HZ_MAX_INPUT_DIM and k not in (
            "INPUT",
            "INPUT_SPEC",
            "ASSERT",
        ):
            self._hz_cache.pop(L.id, None)

        hz_before = self._hz_cache.get(L.id)
        result = self._LAYER_REGISTRY[k](L, input_bounds, self)

        if hz_before is not None and self._hz_cache.get(L.id) is hz_before:
            self._hz_cache[L.id] = hz_from_bounds(
                result.bounds, result.bounds.lb.dtype, result.bounds.lb.device
            )

        return result
