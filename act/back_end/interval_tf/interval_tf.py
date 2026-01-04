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

import torch
from typing import Dict, List
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.transfer_functions import TransferFunction
from act.back_end.layer_schema import LayerKind
from act.back_end.interval_tf.tf_mlp import *
from act.back_end.interval_tf.tf_cnn import *
from act.back_end.interval_tf.tf_rnn import *
from act.back_end.interval_tf.tf_transformer import *


class IntervalTF(TransferFunction):
    """Interval-based transfer functions for standard bounds propagation."""
    
    # Layer kind to function mapping
    _LAYER_REGISTRY = {
        # Identity/constraint layers
        LayerKind.INPUT.value: lambda L, bounds, tf: Fact(bounds=bounds, cons=ConSet()),
        LayerKind.INPUT_SPEC.value: lambda L, bounds, tf: Fact(bounds=bounds, cons=ConSet()),
        LayerKind.ASSERT.value: lambda L, bounds, tf: Fact(bounds=bounds, cons=ConSet()),
        
        # MLP operations
        LayerKind.DENSE.value: lambda L, bounds, tf: tf_dense(L, bounds),
        "BIAS": lambda L, bounds, tf: tf_bias(L, bounds),
        "SCALE": lambda L, bounds, tf: tf_scale(L, bounds),
        LayerKind.RELU.value: lambda L, bounds, tf: tf_relu(L, bounds),
        "LRELU": lambda L, bounds, tf: tf_lrelu(L, bounds),
        "ABS": lambda L, bounds, tf: tf_abs(L, bounds),
        "CLIP": lambda L, bounds, tf: tf_clip(L, bounds),
        
        # Multi-input operations  
        "ADD": lambda L, bounds, tf: tf_add(L, 
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0), 
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)),
        "MUL": lambda L, bounds, tf: tf_mul(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)),
        "CONCAT": lambda L, bounds, tf: tf_concat(L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)),
        "BN": lambda L, bounds, tf: tf_bn(L, bounds),
        
        # CNN operations
        "CONV2D": lambda L, bounds, tf: tf_conv2d(L, bounds),
        "CONV1D": lambda L, bounds, tf: tf_conv1d(L, bounds),
        "CONV3D": lambda L, bounds, tf: tf_conv3d(L, bounds),
        "CONVTRANSPOSE2D": lambda L, bounds, tf: tf_convtranspose2d(L, bounds),
        "MAXPOOL2D": lambda L, bounds, tf: tf_maxpool2d(L, bounds),
        "AVGPOOL2D": lambda L, bounds, tf: tf_avgpool2d(L, bounds),
        "FLATTEN": lambda L, bounds, tf: tf_flatten(L, bounds),
        
        # RNN operations
        "LSTM": lambda L, bounds, tf: tf_lstm(L, bounds),
        "GRU": lambda L, bounds, tf: tf_gru(L, bounds),
        "RNN": lambda L, bounds, tf: tf_rnn(L, bounds),
        "EMBEDDING": lambda L, bounds, tf: tf_embedding(L, bounds),
        
        # Activation functions
        "SIGMOID": lambda L, bounds, tf: tf_sigmoid(L, bounds),
        "TANH": lambda L, bounds, tf: tf_tanh(L, bounds),
        "SOFTPLUS": lambda L, bounds, tf: tf_softplus(L, bounds),
        "SILU": lambda L, bounds, tf: tf_silu(L, bounds),
        "RELU6": lambda L, bounds, tf: tf_relu6(L, bounds),
        "HARDTANH": lambda L, bounds, tf: tf_hardtanh(L, bounds),
        "HARDSIGMOID": lambda L, bounds, tf: tf_hardsigmoid(L, bounds),
        "HARDSWISH": lambda L, bounds, tf: tf_hardswish(L, bounds),
        "MISH": lambda L, bounds, tf: tf_mish(L, bounds),
        "SOFTSIGN": lambda L, bounds, tf: tf_softsign(L, bounds),
        
        # Element-wise operations
        "MAX": lambda L, bounds, tf: tf_max(L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)),
        "MIN": lambda L, bounds, tf: tf_min(L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)),
        "SQUARE": lambda L, bounds, tf: tf_square(L, bounds),
        "POWER": lambda L, bounds, tf: tf_power(L, bounds),
        
        # Tensor operations
        "RESHAPE": lambda L, bounds, tf: tf_reshape(L, bounds),
        "TRANSPOSE": lambda L, bounds, tf: tf_transpose(L, bounds),
        "SQUEEZE": lambda L, bounds, tf: tf_squeeze(L, bounds),
        "UNSQUEEZE": lambda L, bounds, tf: tf_unsqueeze(L, bounds),
        "TILE": lambda L, bounds, tf: tf_tile(L, bounds),
        "EXPAND": lambda L, bounds, tf: tf_expand(L, bounds),
        
        # Transformer operations
        "EMBEDDING_TF": lambda L, bounds, tf: tf_embedding(L, bounds),
        "POSENC": lambda L, bounds, tf: tf_posenc(L, bounds),
        "LAYERNORM": lambda L, bounds, tf: tf_layernorm(L, bounds),
        "GELU": lambda L, bounds, tf: tf_gelu(L, bounds),
        "ATT_SCORES": lambda L, bounds, tf: tf_att_scores(L,
            tf._before[L.meta["q_src"]].bounds,
            tf._before[L.meta["k_src"]].bounds),
        "SOFTMAX": lambda L, bounds, tf: tf_softmax(L, bounds),
        "ATT_MIX": lambda L, bounds, tf: tf_att_mix(L,
            tf._before[L.meta["w_src"]].bounds, 
            tf._before[L.meta["v_src"]].bounds),
        "MHA_SPLIT": lambda L, bounds, tf: tf_mha_split(L, bounds),
        "MHA_JOIN": lambda L, bounds, tf: tf_mha_join(L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)),
        "MASK_ADD": lambda L, bounds, tf: tf_mask_add(L, bounds),
    }
    
    @property
    def name(self) -> str:
        return "IntervalTF"
        
    def supports_layer(self, layer_kind: str) -> bool:
        """Check if this transfer function supports the given layer kind."""
        return layer_kind.upper() in self._LAYER_REGISTRY
        
    def apply(self, L: Layer, input_bounds: Bounds, net: Net,
              before: Dict[int, Fact], after: Dict[int, Fact]) -> Fact:
        """Apply interval transfer function to layer L."""
        k = L.kind.upper()
        if k not in self._LAYER_REGISTRY:
            raise NotImplementedError(f"IntervalTF: Unsupported layer kind '{k}'")
            
        # Store context for lambdas
        self._net = net
        self._before = before
        self._after = after
        
        transfer_fn = self._LAYER_REGISTRY[k]
        return transfer_fn(L, input_bounds, self)