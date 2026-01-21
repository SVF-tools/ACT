#===- act/back_end/dual_tf/tf_transformer.py - Transformer Dual TFs -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Transformer dual backward functions (Attention, LayerNorm, GELU).
#   Softmax/LayerNorm are challenging due to non-element-wise ops.
#   Forward bounds handled by IntervalTF (see tf_forward.py).
#
#===---------------------------------------------------------------------===#

import torch
from typing import Tuple, Optional, Dict
from act.back_end.core import Bounds

# -------- Multi-Head Attention --------
@torch.no_grad()
def dual_attention_backward(
    nu: torch.Tensor, W_Q: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor,
    W_O: Optional[torch.Tensor] = None,
    b_Q: Optional[torch.Tensor] = None, b_K: Optional[torch.Tensor] = None,
    b_V: Optional[torch.Tensor] = None, b_O: Optional[torch.Tensor] = None,
    num_heads: int = 1, bounds_dict: Optional[Dict[str, Bounds]] = None,
    attention_weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention backward via linearization around reference weights. (Pending)"""
    raise NotImplementedError("dual_attention_backward: pending")

# -------- LayerNorm --------
@torch.no_grad()
def dual_layernorm_backward(
    nu: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5,
    normalized_shape: Optional[Tuple[int, ...]] = None, bounds: Optional[Bounds] = None,
    reference_mean: Optional[torch.Tensor] = None, reference_var: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LayerNorm backward via linearization around reference mean/var. (Pending)"""
    raise NotImplementedError("dual_layernorm_backward: pending")

# -------- GELU --------
@torch.no_grad()
def dual_gelu_backward(nu: torch.Tensor, bounds: Bounds) -> Tuple[torch.Tensor, torch.Tensor]:
    """GELU backward: uses linear relaxation similar to smooth activations. (Pending)"""
    raise NotImplementedError("dual_gelu_backward: pending")
