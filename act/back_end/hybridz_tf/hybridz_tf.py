#===- act/back_end/hybridz_tf/hybridz_tf.py - HybridZ Transfer Function -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ Transfer Function Implementation. Implements the HybridzTF class
#   that provides zonotope-based transfer functions with enhanced precision
#   over interval methods.
#
#===---------------------------------------------------------------------===#

"""
"""

import torch
from typing import Dict, List
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.transfer_functions import TransferFunction
from act.back_end.hybridz_tf.tf_mlp import *
from act.back_end.hybridz_tf.tf_cnn import *
from act.back_end.hybridz_tf.tf_rnn import *
from act.back_end.hybridz_tf.tf_transformer import *


class HybridzTF(TransferFunction):
    """HybridZ-based transfer functions with zonotope operations."""
    
    # Layer kind to function mapping for HybridZ operations
    _LAYER_REGISTRY = {
        # Identity/constraint layers
        "INPUT": lambda L, bounds, tf: Fact(bounds=bounds, cons=ConSet()),
        "INPUT_SPEC": lambda L, bounds, tf: Fact(bounds=bounds, cons=ConSet()),
        "ASSERT": lambda L, bounds, tf: Fact(bounds=bounds, cons=ConSet()),
        
        # MLP operations (with HybridZ precision)
        "DENSE": lambda L, bounds, tf: hybridz_tf_dense(L, bounds),
        "BIAS": lambda L, bounds, tf: hybridz_tf_bias(L, bounds),
        "SCALE": lambda L, bounds, tf: hybridz_tf_scale(L, bounds),
        "BN": lambda L, bounds, tf: hybridz_tf_bn(L, bounds),
        "RELU": lambda L, bounds, tf: hybridz_tf_relu(L, bounds),
        "RELU6": lambda L, bounds, tf: hybridz_tf_relu6(L, bounds),
        "LRELU": lambda L, bounds, tf: hybridz_tf_lrelu(L, bounds),
        "TANH": lambda L, bounds, tf: hybridz_tf_tanh(L, bounds),
        "SIGMOID": lambda L, bounds, tf: hybridz_tf_sigmoid(L, bounds),
        "ABS": lambda L, bounds, tf: hybridz_tf_abs(L, bounds),
        "SILU": lambda L, bounds, tf: hybridz_tf_silu(L, bounds),
        "SQUARE": lambda L, bounds, tf: hybridz_tf_square(L, bounds),
        "POWER": lambda L, bounds, tf: hybridz_tf_power(L, bounds),

        # Multi-input operations
        "ADD": lambda L, bounds, tf: hybridz_tf_add(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)),
        "SUB": lambda L, bounds, tf: hybridz_tf_sub(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)),
        "MUL": lambda L, bounds, tf: hybridz_tf_mul(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)),
        "DIV": lambda L, bounds, tf: hybridz_tf_div(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)),
        "POW": lambda L, bounds, tf: hybridz_tf_pow(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)),
        "MATMUL": lambda L, bounds, tf: hybridz_tf_matmul(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)),
        "MAX": lambda L, bounds, tf: hybridz_tf_max(L,
            [tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, i)
             for i in range(len(L.inputs))]),
        "MIN": lambda L, bounds, tf: hybridz_tf_min(L,
            [tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, i)
             for i in range(len(L.inputs))]),
        "CONCAT": lambda L, bounds, tf: hybridz_tf_concat(L,
            [tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, i)
             for i in range(len(L.inputs))]),

        # CNN operations
        "CONV1D": lambda L, bounds, tf: hybridz_tf_conv1d(L, bounds),
        "CONV2D": lambda L, bounds, tf: hybridz_tf_conv2d(L, bounds),
        "CONV3D": lambda L, bounds, tf: hybridz_tf_conv3d(L, bounds),
        "MAXPOOL1D": lambda L, bounds, tf: hybridz_tf_maxpool1d(L, bounds),
        "MAXPOOL2D": lambda L, bounds, tf: hybridz_tf_maxpool2d(L, bounds),
        "MAXPOOL3D": lambda L, bounds, tf: hybridz_tf_maxpool3d(L, bounds),
        "AVGPOOL1D": lambda L, bounds, tf: hybridz_tf_avgpool1d(L, bounds),
        "AVGPOOL2D": lambda L, bounds, tf: hybridz_tf_avgpool2d(L, bounds),
        "PAD": lambda L, bounds, tf: hybridz_tf_pad(L, bounds),
        "UPSAMPLE": lambda L, bounds, tf: hybridz_tf_upsample(L, bounds),
        "FLATTEN": lambda L, bounds, tf: hybridz_tf_flatten(L, bounds),
        "RESHAPE": lambda L, bounds, tf: hybridz_tf_reshape(L, bounds),

        # Tensor operations
        "TRANSPOSE": lambda L, bounds, tf: hybridz_tf_transpose(L, bounds),
        "SQUEEZE": lambda L, bounds, tf: hybridz_tf_squeeze(L, bounds),
        "UNSQUEEZE": lambda L, bounds, tf: hybridz_tf_unsqueeze(L, bounds),
        "SLICE": lambda L, bounds, tf: hybridz_tf_slice(L, bounds),
        "GATHER": lambda L, bounds, tf: hybridz_tf_gather(L, bounds),

        # RNN operations
        "LSTM": lambda L, bounds, tf: hybridz_tf_lstm(L, bounds),
        "GRU": lambda L, bounds, tf: hybridz_tf_gru(L, bounds),
        "RNN": lambda L, bounds, tf: hybridz_tf_rnn(L, bounds),
        "EMBEDDING": lambda L, bounds, tf: hybridz_tf_embedding(L, bounds),
        
        # Transformer operations
        "LAYERNORM": lambda L, bounds, tf: hybridz_tf_layernorm(L, bounds),
        "GELU": lambda L, bounds, tf: hybridz_tf_gelu(L, bounds),
        "SOFTMAX": lambda L, bounds, tf: hybridz_tf_softmax(L, bounds),
        "POSENC": lambda L, bounds, tf: hybridz_tf_posenc(L, bounds),
    }
    
    @property
    def name(self) -> str:
        return "HybridzTF"
        
    def supports_layer(self, layer_kind: str) -> bool:
        """Check if HybridZ supports this layer kind."""
        return layer_kind.upper() in self._LAYER_REGISTRY
        
    def apply(self, L: Layer, input_bounds: Bounds, net: Net,
              before: Dict[int, Fact], after: Dict[int, Fact]) -> Fact:
        """Apply HybridZ transfer function to layer L."""
        k = L.kind.upper()
        if k not in self._LAYER_REGISTRY:
            raise NotImplementedError(f"HybridzTF: Unsupported layer kind '{k}'")
            
        # Store context for lambdas
        self._net = net
        self._before = before
        self._after = after
        
        transfer_fn = self._LAYER_REGISTRY[k]
        return transfer_fn(L, input_bounds, self)