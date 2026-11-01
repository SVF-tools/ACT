#===- act/back_end/interval_tf/interval_tf.py - Interval Transfer Func --====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Interval Transfer Function Implementation. Implements the IntervalTF class
#   that provides interval-based transfer functions for standard bounds
#   propagation analysis.
#
#===---------------------------------------------------------------------===#

from typing import Dict
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.transfer_functions import RegistryTF
from act.back_end.layer_schema import LayerKind
from act.back_end.interval_tf.tf_mlp import *
from act.back_end.interval_tf.tf_cnn import *
from act.back_end.interval_tf.tf_rnn import *
from act.back_end.interval_tf.tf_transformer import *


class IntervalTF(RegistryTF):
    """Interval-based transfer functions for standard bounds propagation."""
    
    # Layer kind to function mapping
    _LAYER_REGISTRY = {
        # Identity/constraint layers
        LayerKind.INPUT.value: lambda L, bounds, tf: Fact(bounds=bounds, cons=ConSet()),
        LayerKind.INPUT_SPEC.value: lambda L, bounds, tf: Fact(bounds=bounds, cons=ConSet()),
        LayerKind.ASSERT.value: lambda L, bounds, tf: Fact(bounds=bounds, cons=ConSet()),
        
        # MLP operations
        LayerKind.DENSE.value: lambda L, bounds, tf: tf_dense(L, bounds),
        LayerKind.BIAS.value: lambda L, bounds, tf: tf_bias(L, bounds),
        LayerKind.SCALE.value: lambda L, bounds, tf: tf_scale(L, bounds),
        LayerKind.RELU.value: lambda L, bounds, tf: tf_relu(L, bounds),
        LayerKind.LRELU.value: lambda L, bounds, tf: tf_lrelu(L, bounds),
        LayerKind.ABS.value: lambda L, bounds, tf: tf_abs(L, bounds),
        LayerKind.CLIP.value: lambda L, bounds, tf: tf_clip(L, bounds),

        # Multi-input operations
        LayerKind.ADD.value: lambda L, bounds, tf: tf_add(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)),
        LayerKind.MUL.value: lambda L, bounds, tf: tf_mul(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)),
        LayerKind.SUB.value: lambda L, bounds, tf: tf_sub(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)),
        LayerKind.DIV.value: lambda L, bounds, tf: tf_div(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)),
        LayerKind.CONCAT.value: lambda L, bounds, tf: tf_concat(L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)),
        LayerKind.BN.value: lambda L, bounds, tf: tf_bn(L, bounds),

        # CNN operations
        LayerKind.CONV2D.value: lambda L, bounds, tf: tf_conv2d(L, bounds),
        LayerKind.CONV1D.value: lambda L, bounds, tf: tf_conv1d(L, bounds),
        LayerKind.CONV3D.value: lambda L, bounds, tf: tf_conv3d(L, bounds),
        LayerKind.CONVTRANSPOSE2D.value: lambda L, bounds, tf: tf_convtranspose2d(L, bounds),
        LayerKind.MAXPOOL2D.value: lambda L, bounds, tf: tf_maxpool2d(L, bounds),
        LayerKind.MAXPOOL3D.value: lambda L, bounds, tf: tf_maxpool3d(L, bounds),
        LayerKind.AVGPOOL1D.value: lambda L, bounds, tf: tf_avgpool1d(L, bounds),
        LayerKind.AVGPOOL2D.value: lambda L, bounds, tf: tf_avgpool2d(L, bounds),
        LayerKind.PAD.value:      lambda L, bounds, tf: tf_pad(L, bounds),
        LayerKind.FLATTEN.value: lambda L, bounds, tf: tf_flatten(L, bounds),

        # RNN operations
        LayerKind.LSTM.value: lambda L, bounds, tf: tf_lstm(L, bounds),
        LayerKind.GRU.value: lambda L, bounds, tf: tf_gru(L, bounds),
        LayerKind.RNN.value: lambda L, bounds, tf: tf_rnn(L, bounds),
        LayerKind.EMBEDDING.value: lambda L, bounds, tf: tf_embedding(L, bounds),

        # Activation functions
        LayerKind.SIGMOID.value: lambda L, bounds, tf: tf_sigmoid(L, bounds),
        LayerKind.TANH.value: lambda L, bounds, tf: tf_tanh(L, bounds),
        LayerKind.ERF.value: lambda L, bounds, tf: tf_erf(L, bounds),
        LayerKind.SQRT.value: lambda L, bounds, tf: tf_sqrt(L, bounds),
        LayerKind.SIN.value: lambda L, bounds, tf: tf_sin(L, bounds),
        LayerKind.COS.value: lambda L, bounds, tf: tf_cos(L, bounds),
        LayerKind.QUANTIZE.value: lambda L, bounds, tf: tf_quantize(L, bounds),
        LayerKind.SOFTPLUS.value: lambda L, bounds, tf: tf_softplus(L, bounds),
        LayerKind.SILU.value: lambda L, bounds, tf: tf_silu(L, bounds),
        LayerKind.RELU6.value: lambda L, bounds, tf: tf_relu6(L, bounds),
        LayerKind.HARDTANH.value: lambda L, bounds, tf: tf_hardtanh(L, bounds),
        LayerKind.HARDSIGMOID.value: lambda L, bounds, tf: tf_hardsigmoid(L, bounds),
        LayerKind.HARDSWISH.value: lambda L, bounds, tf: tf_hardswish(L, bounds),
        LayerKind.MISH.value: lambda L, bounds, tf: tf_mish(L, bounds),
        LayerKind.SOFTSIGN.value: lambda L, bounds, tf: tf_softsign(L, bounds),

        # Element-wise operations
        LayerKind.MAX.value: lambda L, bounds, tf: tf_max(L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)),
        LayerKind.MIN.value: lambda L, bounds, tf: tf_min(L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)),
        LayerKind.SQUARE.value: lambda L, bounds, tf: tf_square(L, bounds),
        LayerKind.POWER.value: lambda L, bounds, tf: tf_power(L, bounds),
        LayerKind.SIGN.value: lambda L, bounds, tf: tf_sign(L, bounds),
        LayerKind.REDUCE_SUM.value: lambda L, bounds, tf: tf_reduce_sum(L, bounds),
        LayerKind.MEAN.value: lambda L, bounds, tf: tf_mean(L, bounds),
        LayerKind.CONSTANT.value: lambda L, bounds, tf: tf_constant(L, bounds),
        LayerKind.COMPARE.value: lambda L, bounds, tf: tf_compare(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)),
        LayerKind.WHERE.value: lambda L, bounds, tf: tf_where(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 2)),
        LayerKind.MATMUL.value: lambda L, bounds, tf: tf_matmul(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)),
        LayerKind.ARG_EXTREMUM.value: lambda L, bounds, tf: tf_arg_extremum(L, bounds),
        LayerKind.UPSAMPLE.value: lambda L, bounds, tf: tf_upsample(L, bounds),
        LayerKind.SCATTER_ND.value: lambda L, bounds, tf: tf_scatter_nd(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 2)),

        # Tensor operations
        LayerKind.RESHAPE.value: lambda L, bounds, tf: tf_reshape(L, bounds),
        LayerKind.TRANSPOSE.value: lambda L, bounds, tf: tf_transpose(L, bounds),
        LayerKind.SQUEEZE.value: lambda L, bounds, tf: tf_squeeze(L, bounds),
        LayerKind.UNSQUEEZE.value: lambda L, bounds, tf: tf_unsqueeze(L, bounds),
        LayerKind.EXPAND.value: lambda L, bounds, tf: tf_expand(L, bounds),
        LayerKind.SLICE.value: lambda L, bounds, tf: tf_slice(L, bounds),
        LayerKind.GATHER.value: lambda L, bounds, tf: tf_gather(L, bounds),

        # Transformer operations
        LayerKind.EMBEDDING_TF.value: lambda L, bounds, tf: tf_embedding(L, bounds),
        LayerKind.POSENC.value: lambda L, bounds, tf: tf_posenc(L, bounds),
        LayerKind.LAYERNORM.value: lambda L, bounds, tf: tf_layernorm(L, bounds),
        LayerKind.GELU.value: lambda L, bounds, tf: tf_gelu(L, bounds),
        LayerKind.ATT_SCORES.value: lambda L, bounds, tf: tf_att_scores(L,
            tf._after[L.params["q_src"]].bounds,
            tf._after[L.params["k_src"]].bounds),
        LayerKind.SOFTMAX.value: lambda L, bounds, tf: tf_softmax(L, bounds),
        LayerKind.ATT_MIX.value: lambda L, bounds, tf: tf_att_mix(L,
            tf._after[L.params["w_src"]].bounds,
            tf._after[L.params["v_src"]].bounds),
        LayerKind.MHA_SPLIT.value: lambda L, bounds, tf: tf_mha_split(L, bounds),
        LayerKind.MHA_JOIN.value: lambda L, bounds, tf: tf_mha_join(L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)),
        LayerKind.MASK_ADD.value: lambda L, bounds, tf: tf_mask_add(L, bounds),
    }
    
    def __init__(self) -> None:
        super().__init__("IntervalTF")
        
    def apply(self, L: Layer, input_bounds: Bounds, net: Net,
              before: Dict[int, Fact], after: Dict[int, Fact]) -> Fact:
        """Apply interval transfer function to layer L."""
        k = self._check_supported(L.kind)
        self._set_context(net, before, after)
        
        transfer_fn = self._LAYER_REGISTRY[k]
        return transfer_fn(L, input_bounds, self)