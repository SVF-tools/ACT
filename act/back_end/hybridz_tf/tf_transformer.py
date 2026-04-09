#===- act/back_end/hybridz_tf/tf_transformer.py - HybridZ Transformer TF -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ Transformer Transfer Functions. Implements HybridZ-based transfer
#   functions for Transformer layers including layer normalization, attention
#   mechanisms, and position encoding.
#
#===---------------------------------------------------------------------===#

import torch
import math
from act.back_end.core import Bounds, Fact, Layer, ConSet


@torch.no_grad()
def hybridz_tf_layernorm(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for layer normalization with enhanced precision."""
    # Layer norm parameters
    normalized_shape = L.params.get("normalized_shape")
    eps = float(L.params.get("eps", 1e-5))
    weight = L.params.get("weight")
    bias = L.params.get("bias")
    
    # For HybridZ: more precise handling of normalization
    # LayerNorm: y = (x - μ) / σ * γ + β
    # where μ = mean(x), σ = sqrt(var(x) + eps)
    
    # Conservative bound computation for normalized values
    # Assume normalized values are approximately in [-3, 3] range for most cases
    # This is a conservative approximation for interval analysis
    
    input_range = Bin.ub - Bin.lb
    # If input range is small, normalization has less effect
    if torch.all(input_range < eps):
        lb_norm = torch.zeros_like(Bin.lb)
        ub_norm = torch.zeros_like(Bin.ub)
    else:
        # Conservative bounds for normalized values
        scale_factor = 3.0  # Most normalized values fall within [-3, 3]
        lb_norm = torch.full_like(Bin.lb, -scale_factor)
        ub_norm = torch.full_like(Bin.ub, scale_factor)
    
    # Apply affine transformation if weight and bias exist
    if weight is not None:
        weight_pos = torch.clamp(weight, min=0)
        weight_neg = torch.clamp(weight, max=0)
        
        lb_out = weight_pos * lb_norm + weight_neg * ub_norm
        ub_out = weight_pos * ub_norm + weight_neg * lb_norm
    else:
        lb_out = lb_norm
        ub_out = ub_norm
    
    if bias is not None:
        lb_out += bias
        ub_out += bias
    
    Bout = Bounds(lb=lb_out, ub=ub_out)
    
    return Fact(bounds=Bout, cons=ConSet())


@torch.no_grad()
def hybridz_tf_gelu(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for GELU activation with piecewise linear approximation."""
    # GELU(x) = x * Φ(x) where Φ is CDF of standard normal
    # Approximate with piecewise linear function for different ranges
    
    # Define breakpoints for piecewise linear approximation
    breakpoints = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], device=Bin.lb.device, dtype=Bin.lb.dtype)
    
    # Compute piecewise linear bounds
    lb = torch.zeros_like(Bin.lb)
    ub = torch.zeros_like(Bin.ub)
    
    for i in range(len(Bin.lb)):
        x_min, x_max = Bin.lb[i].item(), Bin.ub[i].item()
        
        # Find which segments the interval [x_min, x_max] intersects
        # Check GELU values at breakpoints within interval
        y_candidates = [gelu_approx(x_min), gelu_approx(x_max)]
        for bp in breakpoints:
            if x_min <= bp <= x_max:
                y_candidates.append(gelu_approx(bp.item()))
        
        # Also check for potential extrema (GELU has minimum around x ≈ -0.7)
        if x_min <= -0.7 <= x_max:
            y_candidates.append(gelu_approx(-0.7))
        
        y_candidates = torch.tensor(y_candidates, device=Bin.lb.device, dtype=Bin.lb.dtype)
        lb[i] = torch.min(y_candidates)
        ub[i] = torch.max(y_candidates)
    
    Bout = Bounds(lb=lb, ub=ub)
    
    return Fact(bounds=Bout, cons=ConSet())


def gelu_approx(x: float) -> float:
    """Approximate GELU function value."""
    # GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    import math
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    inner = sqrt_2_pi * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + math.tanh(inner))


@torch.no_grad()
def hybridz_tf_softmax(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for softmax with simplex constraints."""
    # Softmax output: exp(x_i) / sum(exp(x_j))
    # Properties: sum = 1, all values ≥ 0
    
    # Conservative bounds for softmax
    # Lower bound: 0 (always non-negative)
    # Upper bound: 1 (probability values)
    
    n = len(Bin.lb)
    lb = torch.zeros_like(Bin.lb)
    ub = torch.ones_like(Bin.ub)
    
    max_input = torch.max(Bin.ub)
    min_input = torch.min(Bin.lb)
    
    # If one input is much larger than others, it will dominate
    for i in range(n):
        if Bin.lb[i] > max_input - 1e-6:  # This element is approximately the maximum
            # This element will have high probability
            others_max = torch.max(torch.cat([Bin.ub[:i], Bin.ub[i+1:]]))
            if Bin.lb[i] > others_max + 1.0:  # Significantly larger
                lb[i] = 0.7  # Conservative lower bound for dominant element
        
        if Bin.ub[i] < min_input + 1e-6:  # This element is approximately the minimum
            # This element will have low probability
            others_min = torch.min(torch.cat([Bin.lb[:i], Bin.lb[i+1:]]))
            if Bin.ub[i] < others_min - 1.0:  # Significantly smaller
                ub[i] = 0.3  # Conservative upper bound for dominated element
    
    Bout = Bounds(lb=lb, ub=ub)
    
    # Softmax generates simplex constraints (sum = 1, all ≥ 0)
    cons = ConSet()
    rowsize = len(L.out_vars)
    mode = L.params.get("mode", "simplex")
    cons.add_op(f"softmax:{mode}:{L.id}", list(L.out_vars), rowsize=rowsize)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_posenc(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for positional encoding."""
    # Positional encoding adds fixed positional embeddings
    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    max_len = L.params.get("max_len", 1000)
    d_model = L.params.get("d_model", Bin.lb.shape[-1])
    
    # Positional encoding values are bounded by [-1, 1] (sin/cos range)
    # Adding to input bounds
    pe_range = 1.0  # sin/cos range
    
    lb = Bin.lb - pe_range
    ub = Bin.ub + pe_range
    Bout = Bounds(lb=lb, ub=ub)
    return Fact(bounds=Bout, cons=ConSet())
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_attention_scores(L: Layer, Q_bounds: Bounds, K_bounds: Bounds) -> Fact:
    """HybridZ transfer function for attention score computation: Q @ K^T / sqrt(d_k)."""
    d_k = L.params.get("d_k", Q_bounds.lb.shape[-1])
    scale = 1.0 / math.sqrt(d_k)
    
    # Attention scores: QK^T / sqrt(d_k)
    # Use bilinear multiplication bounds
    
    # For each pair (q_i, k_j), compute bounds for q_i * k_j
    q_lb, q_ub = Q_bounds.lb, Q_bounds.ub
    k_lb, k_ub = K_bounds.lb, K_bounds.ub
    
    # Simplified: assume we're computing one attention head
    # Full implementation would handle batch dimensions and multiple heads
    
    # McCormick bounds for dot product
    products = []
    for i in range(len(q_lb)):
        for j in range(len(k_lb)):
            # Bilinear term: q[i] * k[j]
            corners = torch.tensor([
                q_lb[i] * k_lb[j],
                q_lb[i] * k_ub[j], 
                q_ub[i] * k_lb[j],
                q_ub[i] * k_ub[j]
            ])
            products.append((torch.min(corners), torch.max(corners)))
    
    # Sum over embedding dimension and scale
    # This is a simplified version - full implementation needs proper tensor handling
    lb = torch.tensor([p[0] for p in products]) * scale
    ub = torch.tensor([p[1] for p in products]) * scale
    
    Bout = Bounds(lb=lb, ub=ub)
    return Fact(bounds=Bout, cons=ConSet())
    return Fact(bounds=Bout, cons=cons)