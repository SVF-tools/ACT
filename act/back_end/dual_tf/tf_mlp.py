#===- act/back_end/dual_tf/tf_mlp.py - MLP Dual Transfer Functions ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   MLP dual transfer functions for Lagrangian dual bound computation.
#
# Mathematical Background:
#   For ReLU z = max(0, y) with y in [l, u]:
#   - If l >= 0: slope d = 1 (always active)
#   - If u <= 0: slope d = 0 (always inactive)
#   - If l < 0 < u: slope d = u/(u-l), contribution = [v]+ * l
#
#   For Dense y = W @ x + b:
#   - Backward: v_{i-1} = W^T @ v
#   - Contribution: -b^T @ v
#
#===---------------------------------------------------------------------===#

import torch
from typing import Tuple, Optional

from act.back_end.core import Bounds, Layer


# =============================================================================
# Size Alignment Helpers
# =============================================================================

def _align_size(a: torch.Tensor, target_size: int) -> torch.Tensor:
    """Align tensor a to target_size by truncating or tiling."""
    if a.numel() == target_size:
        return a.flatten()
    elif a.numel() > target_size:
        return a.flatten()[:target_size]
    else:
        repeats = (target_size + a.numel() - 1) // a.numel()
        return a.flatten().repeat(repeats)[:target_size]


# =============================================================================
# ReLU Dual Transfer Functions
# =============================================================================

@torch.no_grad()
def compute_relu_slopes(lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    """
    Compute ReLU relaxation slopes for the triangle relaxation.
    
    For each neuron i with pre-activation bounds [l_i, u_i]:
    - d_i = 1 if l_i >= 0 (always active)
    - d_i = 0 if u_i <= 0 (always inactive)  
    - d_i = u_i / (u_i - l_i) if l_i < 0 < u_i (ambiguous)
    
    Args:
        lb: Lower bounds tensor [n]
        ub: Upper bounds tensor [n]
        
    Returns:
        slopes: Tensor of slopes [n]
    """
    n = lb.shape[0]
    d = torch.zeros(n, dtype=lb.dtype, device=lb.device)
    
    # Always active: d = 1
    active = lb >= 0
    d[active] = 1.0
    
    # Always inactive: d = 0 (already initialized to 0)
    
    # Ambiguous: d = u / (u - l)
    ambiguous = (lb < 0) & (ub > 0)
    denom = (ub[ambiguous] - lb[ambiguous]).clamp(min=1e-12)
    d[ambiguous] = ub[ambiguous] / denom
    
    return d


@torch.no_grad()
def get_relu_masks(
    lb: torch.Tensor, 
    ub: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get boolean masks for ReLU neuron classification.
    
    Args:
        lb: Lower bounds [n]
        ub: Upper bounds [n]
        
    Returns:
        Tuple of (mask_on, mask_off, mask_amb) boolean tensors
    """
    mask_on = lb >= 0
    mask_off = ub <= 0
    mask_amb = ~(mask_on | mask_off)
    return mask_on, mask_off, mask_amb


@torch.no_grad()
def dual_relu_backward(
    nu: torch.Tensor,
    bounds: Bounds
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass through ReLU layer for dual computation.
    
    Computes:
    - v_{i-1} = D @ v_i (element-wise multiplication with slopes)
    - contribution = sum [v_i]+ * l_i for ambiguous neurons
    
    Args:
        nu: Incoming v vector from next layer [n]
        bounds: Pre-activation bounds (lb, ub)
        
    Returns:
        Tuple of (nu_out, contribution):
        - nu_out: v vector to pass to previous layer
        - contribution: Scalar contribution to dual objective
    """
    lb, ub = bounds.lb.flatten(), bounds.ub.flatten()
    nu_flat = nu.flatten()
    
    # Handle size mismatch between nu and bounds
    n_nu = nu_flat.numel()
    n_bounds = lb.numel()
    
    if n_nu != n_bounds:
        # Align sizes - use minimum common size
        n_common = min(n_nu, n_bounds)
        lb = lb[:n_common]
        ub = ub[:n_common]
        nu_flat = nu_flat[:n_common]
    
    # Compute slopes
    d = compute_relu_slopes(lb, ub)
    
    # Backward: v_{i-1} = D @ v_i
    nu_out = d * nu_flat
    
    # Contribution from ambiguous neurons: sum [v]+ * l
    mask_on, mask_off, mask_amb = get_relu_masks(lb, ub)
    
    # [v]+ = max(v, 0)
    nu_pos = nu_flat.clamp(min=0)
    
    # Contribution: only from ambiguous neurons
    contribution = (nu_pos[mask_amb] * lb[mask_amb]).sum()
    
    return nu_out, contribution


# =============================================================================
# Dense (Linear) Dual Transfer Functions
# =============================================================================

@torch.no_grad()
def dual_dense_backward(
    nu: torch.Tensor,
    W: torch.Tensor,
    b: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass through Dense layer for dual computation.
    
    Computes:
    - v_{i-1} = W^T @ v_i
    - contribution = -b^T @ v_i (if bias exists)
    
    Args:
        nu: Incoming v vector [out_features]
        W: Weight matrix [out_features, in_features]
        b: Optional bias vector [out_features]
        
    Returns:
        Tuple of (nu_out, contribution):
        - nu_out: v vector for previous layer [in_features]
        - contribution: Scalar contribution to dual objective
    """
    # Backward: v_{i-1} = W^T @ v_i
    nu_out = W.T @ nu
    
    # Contribution from bias: -b^T @ v
    if b is not None:
        contribution = -(b @ nu)
    else:
        contribution = torch.tensor(0.0, dtype=nu.dtype, device=nu.device)
    
    return nu_out, contribution


# =============================================================================
# Bias / Scale / BatchNorm Dual Transfer Functions
# =============================================================================

@torch.no_grad()
def dual_bias_backward(
    nu: torch.Tensor,
    c: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass through Bias layer (y = x + c).
    
    Backward: v_{i-1} = v_i (identity)
    Contribution: -c^T @ v_i
    """
    # Handle size mismatch
    nu_flat = nu.flatten()
    c_flat = c.flatten()
    if c_flat.numel() != nu_flat.numel():
        if c_flat.numel() > nu_flat.numel():
            c_flat = c_flat[:nu_flat.numel()]
        else:
            c_flat = c_flat.repeat((nu_flat.numel() + c_flat.numel() - 1) // c_flat.numel())[:nu_flat.numel()]
    contribution = -(c_flat @ nu_flat)
    return nu, contribution


@torch.no_grad()
def dual_scale_backward(
    nu: torch.Tensor,
    a: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass through Scale layer (y = a * x).
    
    Backward: v_{i-1} = a * v_i
    Contribution: 0
    """
    # Handle size mismatch
    if a.numel() != nu.numel():
        if a.numel() > nu.numel():
            a = a.flatten()[:nu.numel()].view(nu.shape)
        else:
            a = a.flatten().repeat((nu.numel() + a.numel() - 1) // a.numel())[:nu.numel()].view(nu.shape)
    nu_out = a * nu
    contribution = torch.tensor(0.0, dtype=nu.dtype, device=nu.device)
    return nu_out, contribution


@torch.no_grad()
def dual_bn_backward(
    nu: torch.Tensor,
    A: torch.Tensor,
    c: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass through BatchNorm layer (y = A * x + c).
    
    Backward: v_{i-1} = A * v_i
    Contribution: -c^T @ v_i
    """
    # Handle size mismatch
    nu_flat = nu.flatten()
    if A.numel() != nu.numel():
        if A.numel() > nu.numel():
            A = A.flatten()[:nu.numel()].view(nu.shape)
        else:
            A = A.flatten().repeat((nu.numel() + A.numel() - 1) // A.numel())[:nu.numel()].view(nu.shape)
    if c.numel() != nu_flat.numel():
        c_flat = c.flatten()
        if c_flat.numel() > nu_flat.numel():
            c_flat = c_flat[:nu_flat.numel()]
        else:
            c_flat = c_flat.repeat((nu_flat.numel() + c_flat.numel() - 1) // c_flat.numel())[:nu_flat.numel()]
    else:
        c_flat = c.flatten()
    nu_out = A * nu
    contribution = -(c_flat @ nu_flat)
    return nu_out, contribution


# =============================================================================
# Identity-like Layers (Flatten, Reshape, etc.)
# =============================================================================

@torch.no_grad()
def dual_identity_backward(
    nu: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for identity-like layers (Flatten, Reshape, etc.).
    
    Backward: v_{i-1} = v_i
    Contribution: 0
    """
    contribution = torch.tensor(0.0, dtype=nu.dtype, device=nu.device)
    return nu, contribution


# =============================================================================
# CNN Layers (Placeholders)
# =============================================================================

# =============================================================================
# Conv2D Dual Transfer Functions
# =============================================================================

@torch.no_grad()
def dual_conv2d_backward(
    nu: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    input_shape: Optional[tuple] = None,
    output_shape: Optional[tuple] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass through Conv2D layer for dual computation.
    
    Uses transposed convolution to compute v_{i-1} = ConvTranspose(v_i, W).
    
    Args:
        nu: Incoming v vector from next layer (flattened)
        weight: Conv2D weights [out_channels, in_channels, kH, kW]
        bias: Optional bias [out_channels]
        stride: Convolution stride
        padding: Convolution padding
        input_shape: Input spatial shape (C, H, W) for reshaping
        output_shape: Output spatial shape (C, H, W) for reshaping
        
    Returns:
        Tuple of (nu_out, contribution)
    """
    import torch.nn.functional as F
    
    out_channels, in_channels, kH, kW = weight.shape
    
    # Try to infer output spatial shape from nu
    nu_flat = nu.flatten()
    n_out = nu_flat.numel()
    
    # Determine output spatial dimensions
    oC, oH, oW = out_channels, 1, 1  # Default initialization
    
    if output_shape is not None:
        # Use provided output shape - handle (N, C, H, W) or (C, H, W) formats
        if len(output_shape) == 4:
            _, oC, oH, oW = output_shape  # (N, C, H, W)
        elif len(output_shape) == 3:
            oC, oH, oW = output_shape  # (C, H, W)
        else:
            # Fallback to inference
            oC = out_channels
            spatial_size = n_out // oC if oC > 0 else n_out
            oH = oW = int(spatial_size ** 0.5) if spatial_size > 0 else 1
    else:
        # Infer from size: assume square spatial dimensions
        oC = out_channels
        spatial_size = n_out // oC if oC > 0 else n_out
        oH = oW = int(spatial_size ** 0.5) if spatial_size > 0 else 1
    
    # Reshape nu to 4D
    try:
        if n_out == oC * oH * oW:
            nu_4d = nu_flat.view(1, oC, oH, oW)
        else:
            # Size mismatch - truncate or pad
            expected = oC * oH * oW
            if n_out > expected:
                nu_4d = nu_flat[:expected].view(1, oC, oH, oW)
            else:
                nu_padded = torch.zeros(expected, dtype=nu.dtype, device=nu.device)
                nu_padded[:n_out] = nu_flat
                nu_4d = nu_padded.view(1, oC, oH, oW)
    except RuntimeError:
        # Fallback: return identity
        contribution = torch.tensor(0.0, dtype=nu.dtype, device=nu.device)
        if bias is not None:
            contribution = -(bias @ nu_flat[:out_channels] if nu_flat.numel() >= out_channels else bias[:nu_flat.numel()] @ nu_flat)
        return nu, contribution
    
    # Compute v_{i-1} using transposed convolution
    # ConvTranspose2d: output_size = (input_size - 1) * stride - 2 * padding + kernel_size
    nu_out_4d = F.conv_transpose2d(nu_4d, weight, None, stride=stride, padding=padding)
    nu_out = nu_out_4d.flatten()
    
    # Contribution from bias: -b^T @ v
    if bias is not None:
        # Sum v over spatial dimensions for each channel
        nu_per_channel = nu_4d.sum(dim=(2, 3)).squeeze(0)  # [out_channels]
        contribution = -(bias @ nu_per_channel)
    else:
        contribution = torch.tensor(0.0, dtype=nu.dtype, device=nu.device)
    
    return nu_out, contribution


# =============================================================================
# Pooling Layers (MaxPool2d, AvgPool2d)
# =============================================================================

def dual_maxpool2d_backward(nu: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """MaxPool2d dual backward (placeholder)."""
    raise NotImplementedError("dual_maxpool2d_backward")


def dual_avgpool2d_backward(nu: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """AvgPool2d dual backward (placeholder)."""
    raise NotImplementedError("dual_avgpool2d_backward")


# =============================================================================
# RNN Layers (Placeholders)
# =============================================================================

def dual_lstm_backward(nu: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """LSTM dual backward (placeholder)."""
    raise NotImplementedError("dual_lstm_backward")


def dual_gru_backward(nu: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """GRU dual backward (placeholder)."""
    raise NotImplementedError("dual_gru_backward")


# =============================================================================
# Transformer Layers (Placeholders)
# =============================================================================

def dual_attention_backward(nu: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Multi-head attention dual backward (placeholder)."""
    raise NotImplementedError("dual_attention_backward")


def dual_layernorm_backward(nu: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """LayerNorm dual backward (placeholder)."""
    raise NotImplementedError("dual_layernorm_backward")
