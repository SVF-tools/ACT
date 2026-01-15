#===- act/back_end/dual_tf/__init__.py - Dual Transfer Functions --------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Dual transfer functions module for Lagrangian dual bound (wong & kolter, CROWN-bounding-based style) computation .
#   Implements backward pass for certified bound computation.
#   - Precision driven: computes tight bounds on the dual objective by backward Lagrangian method 
#   - Adaptive Optimization: computes the bound via dual variables which can be on-demand optimized with gradient-based methods  
#   - Spurious counterexample: greedy spurious counterexample generation via linear boundary 
#
# Key Components:
#   - dual_tf: DualTF class following TransferFunction pattern
#   - tf_mlp: ReLU and Dense dual backward functions
#
#===---------------------------------------------------------------------===#

from .dual_tf import (
    DualTF,
    compute_dual_bound,
    compute_robust_loss_bound,
)

from .tf_mlp import (
    dual_relu_backward,
    dual_dense_backward,
    compute_relu_slopes,
)

from .forward_bounds import (
    compute_forward_bounds,
)

# placeholder for future implementation
from .tf_mlp import (
    dual_maxpool2d_backward,
    dual_avgpool2d_backward,
    dual_lstm_backward,
    dual_gru_backward,
    dual_attention_backward,
    dual_layernorm_backward,
)

__all__ = [
    'DualTF',
    'compute_dual_bound',
    'compute_robust_loss_bound',
    'dual_relu_backward',
    'dual_dense_backward',
    'compute_relu_slopes',
    'compute_forward_bounds',
    # Pooling Layers
    'dual_maxpool2d_backward',
    'dual_avgpool2d_backward',
    # RNN
    'dual_lstm_backward',
    'dual_gru_backward',
    # Transformer
    'dual_attention_backward',
    'dual_layernorm_backward',
]
