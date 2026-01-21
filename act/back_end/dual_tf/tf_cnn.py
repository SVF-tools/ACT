#===- act/back_end/dual_tf/tf_cnn.py - CNN Dual Transfer Functions ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   CNN dual backward functions. Conv2D: v_out=ConvT(v,W), contrib=-b^T@v.
#   Forward bounds handled by IntervalTF (see tf_forward.py).
#
#===---------------------------------------------------------------------===#

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

# -------- Conv2D --------
@torch.no_grad()
def dual_conv2d_backward(
    nu: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
    stride: int = 1, padding: int = 0,
    input_shape: Optional[tuple] = None, output_shape: Optional[tuple] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Conv2D backward: v_out = ConvTranspose(v, W), contrib = -b^T @ sum_spatial(v)."""
    assert weight.dim() == 4, f"weight must be 4D [oC,iC,kH,kW], got shape {weight.shape}"
    
    oC, iC, kH, kW = weight.shape
    v = nu.flatten()
    n = v.numel()
    
    # Determine output spatial dims
    if output_shape is not None:
        if len(output_shape) == 4: _, oC, oH, oW = output_shape
        elif len(output_shape) == 3: oC, oH, oW = output_shape
        else: oH = oW = int((n // oC) ** 0.5) if oC > 0 else 1
    else:
        spatial = n // oC if oC > 0 else n
        oH = oW = int(spatial ** 0.5) if spatial > 0 else 1
    
    # Reshape to 4D
    try:
        expected = oC * oH * oW
        if n == expected:
            v_4d = v.view(1, oC, oH, oW)
        elif n > expected:
            v_4d = v[:expected].view(1, oC, oH, oW)
        else:
            v_pad = torch.zeros(expected, dtype=nu.dtype, device=nu.device)
            v_pad[:n] = v
            v_4d = v_pad.view(1, oC, oH, oW)
    except RuntimeError:
        # Fallback: return identity
        contrib = torch.tensor(0.0, dtype=nu.dtype, device=nu.device)
        if bias is not None:
            m = min(oC, v.numel())
            contrib = -(bias[:m] @ v[:m])
        return nu, contrib
    
    # Transposed conv for backward
    v_out = F.conv_transpose2d(v_4d, weight, None, stride=stride, padding=padding).flatten()
    
    # Bias contribution
    if bias is not None:
        v_per_ch = v_4d.sum(dim=(2, 3)).squeeze(0)  # [oC]
        contrib = -(bias @ v_per_ch)
    else:
        contrib = torch.tensor(0.0, dtype=nu.dtype, device=nu.device)
    
    return v_out, contrib

# -------- Pooling (placeholders) --------
@torch.no_grad()
def dual_maxpool2d_backward(
    nu: torch.Tensor, kernel_size: int = 2, stride: Optional[int] = None, padding: int = 0,
    input_shape: Optional[tuple] = None, output_shape: Optional[tuple] = None,
    indices: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MaxPool2d backward: distributes gradient to max locations. (Pending)"""
    raise NotImplementedError("dual_maxpool2d_backward: pending")

@torch.no_grad()
def dual_avgpool2d_backward(
    nu: torch.Tensor, kernel_size: int = 2, stride: Optional[int] = None, padding: int = 0,
    input_shape: Optional[tuple] = None, output_shape: Optional[tuple] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """AvgPool2d backward: distributes gradient uniformly. (Pending)"""
    raise NotImplementedError("dual_avgpool2d_backward: pending")
