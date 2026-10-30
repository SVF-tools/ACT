#===- act/back_end/dual_tf/__init__.py - Dual Transfer Functions --------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Dual transfer functions module for Lagrangian dual bound (wong & kolter-style) computation.
#   Implements backward pass for certified bound computation.
#   - Precision driven: computes tight bounds on the dual objective by backward Lagrangian method 
#   - Adaptive Optimization: computes the bound via dual variables which can be on-demand optimized with gradient-based methods  
#   - Spurious counterexample: greedy spurious counterexample generation via linear boundary 
#
#===---------------------------------------------------------------------===#

# Disable pyright import-cycle error for this module (circular imports are intentional)
# pyright: reportImportCycles=false
"""Dual transfer functions: certified bounds via backward Lagrangian relaxation.

Entry point: ``DualSolver.evaluate_spec`` (in ``act.back_end.solver.solver_dual``).
Per-layer kernels live in ``tf_mlp`` (MLP), ``tf_cnn`` (CNN), ``tf_smooth``
(sigmoid/tanh), ``tf_rnn`` (LSTM/GRU stubs), ``tf_transformer`` (attention
and projection kernels). Forward dual-track pass in ``compute_forward_bounds``.

Batch convention: all tensors are ``[B, *layer_shape]``. ``compute_certified_bound``
requires ``c.dim() == 2``; ``compute_forward_bounds`` auto-promotes 1-D inputs.
Backward handlers gate on ``enable_grad`` for robust-training use.
"""

# Core DualTF class + ADD/CONCAT dispatch (lives in dual_tf.py beside the class)
from .dual_tf import (
    DualTF, backward_add, backward_concat, backward_div,
    forward_add, forward_concat, forward_div,
)

# MLP batched kernels + dispatch (backward) and forward registry handlers
from .tf_mlp import (
    dual_relu_backward, dual_dense_backward, get_relu_masks,
    dual_bias_backward, dual_scale_backward, dual_bn_backward, dual_identity_backward,
    backward_dense, backward_relu, backward_bias, backward_scale,
    backward_bn, backward_identity,
    forward_dense, forward_relu, forward_bias, forward_scale,
    forward_bn, forward_lrelu, forward_identity, forward_reshape,
)

# Forward bounds
from .tf_forward import (
    compute_forward_bounds, Frame, ForwardFrame, ForwardPassStats,
    ForwardPrefixCache, input_support, forward_frame_row_lower_bounds,
)

from .tf_cnn import (
    dual_conv2d_backward,
    backward_conv1d, forward_conv1d,
    backward_conv2d, backward_maxpool2d, backward_avgpool2d,
    forward_conv2d, forward_maxpool2d, forward_avgpool2d,
)

# Smooth activation batched kernels + dispatch (backward) and forward handlers
from .tf_smooth import (
    dual_smooth_backward, dual_sigmoid_backward, dual_tanh_backward,
    compute_smooth_relaxation, sigmoid, dsigmoid, tanh, dtanh,
    backward_sigmoid, backward_tanh,
    forward_sigmoid, forward_tanh,
    forward_square, backward_square,
)

# Recurrent stubs and implemented transformer registry handlers
from .tf_rnn import forward_lstm, backward_lstm, forward_gru, backward_gru
from .tf_transformer import (
    forward_attention, backward_attention,
    forward_mha_split, backward_mha_split,
    forward_layernorm, backward_layernorm,
    forward_gelu, backward_gelu,
)

__all__ = [
    'DualTF', 'backward_add', 'backward_concat', 'forward_add', 'forward_concat',
    'forward_div', 'backward_div',
    'dual_relu_backward', 'dual_dense_backward', 'get_relu_masks',
    'dual_bias_backward', 'dual_scale_backward', 'dual_bn_backward', 'dual_identity_backward',
    'backward_dense', 'backward_relu', 'backward_bias', 'backward_scale',
    'backward_bn', 'backward_identity',
    'forward_dense', 'forward_relu', 'forward_bias', 'forward_scale',
    'forward_bn', 'forward_lrelu', 'forward_identity', 'forward_reshape',
    'compute_forward_bounds',
    'Frame', 'ForwardFrame', 'ForwardPassStats', 'ForwardPrefixCache',
    'input_support', 'forward_frame_row_lower_bounds',
    'dual_conv2d_backward',
    'backward_conv1d', 'forward_conv1d',
    'backward_conv2d', 'backward_maxpool2d', 'backward_avgpool2d',
    'forward_conv2d', 'forward_maxpool2d', 'forward_avgpool2d',
    'dual_smooth_backward', 'dual_sigmoid_backward', 'dual_tanh_backward',
    'compute_smooth_relaxation', 'sigmoid', 'dsigmoid', 'tanh', 'dtanh',
    'backward_sigmoid', 'backward_tanh',
    'forward_sigmoid', 'forward_tanh',
    'forward_square', 'backward_square',
    'forward_lstm', 'backward_lstm', 'forward_gru', 'backward_gru',
    'forward_attention', 'backward_attention',
    'forward_mha_split', 'backward_mha_split',
    'forward_layernorm', 'backward_layernorm',
    'forward_gelu', 'backward_gelu',
]
