#===- act/back_end/hybridz_tf/tf_rnn.py - HybridZ RNN Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ RNN Transfer Functions. Implements HybridZ-based transfer functions
#   for RNN layers including LSTM, GRU, and basic RNN cells.
#
#===---------------------------------------------------------------------===#

import torch
from typing import Tuple
from act.back_end.core import Bounds, Fact, Layer, ConSet


def hybridz_tf_lstm(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for LSTM cells."""
    # LSTM is complex with internal gates - conservative approximation
    # For now, use interval-based bounds with HybridZ constraint generation
    
    input_size = L.meta.get("input_size")
    hidden_size = L.meta.get("hidden_size") 
    
    # Conservative bounds for LSTM output
    # Hidden state typically bounded by tanh activation [-1, 1]
    # Cell state can have wider range
    
    # Split input into sequence and initial hidden/cell states if needed
    seq_len = L.meta.get("seq_len", 1)
    batch_size = L.meta.get("batch_size", 1)
    
    # Output bounds (hidden states)
    lb = torch.full((hidden_size,), -1.0, device=Bin.lb.device, dtype=Bin.lb.dtype)
    ub = torch.full((hidden_size,), 1.0, device=Bin.lb.device, dtype=Bin.lb.dtype)
    
    # If we have tighter input bounds, we might get tighter output bounds
    input_range = torch.max(Bin.ub) - torch.min(Bin.lb)
    if input_range < 2.0:  # Input is somewhat bounded
        # Scale output bounds proportionally
        scale_factor = min(1.0, input_range / 2.0)
        lb *= scale_factor
        ub *= scale_factor
    
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_lstm(L.id, L.in_vars, L.out_vars, input_size, hidden_size)
    
    return Fact(bounds=Bout, cons=cons)


def hybridz_tf_gru(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for GRU cells."""
    # GRU is simpler than LSTM but still complex
    
    input_size = L.meta.get("input_size")
    hidden_size = L.meta.get("hidden_size")
    
    # GRU output (hidden state) bounded by tanh activation
    lb = torch.full((hidden_size,), -1.0, device=Bin.lb.device, dtype=Bin.lb.dtype)
    ub = torch.full((hidden_size,), 1.0, device=Bin.lb.device, dtype=Bin.lb.dtype)
    
    # Tighter bounds based on input
    input_range = torch.max(Bin.ub) - torch.min(Bin.lb)
    if input_range < 2.0:
        scale_factor = min(1.0, input_range / 2.0)
        lb *= scale_factor
        ub *= scale_factor
    
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_gru(L.id, L.in_vars, L.out_vars, input_size, hidden_size)
    
    return Fact(bounds=Bout, cons=cons)


def hybridz_tf_rnn(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for basic RNN cells."""
    # Basic RNN: h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
    
    input_size = L.meta.get("input_size")
    hidden_size = L.meta.get("hidden_size")
    nonlinearity = L.meta.get("nonlinearity", "tanh")  # tanh or relu
    
    if nonlinearity == "tanh":
        # Tanh activation bounds output to [-1, 1]
        lb = torch.full((hidden_size,), -1.0, device=Bin.lb.device, dtype=Bin.lb.dtype)
        ub = torch.full((hidden_size,), 1.0, device=Bin.lb.device, dtype=Bin.lb.dtype)
    elif nonlinearity == "relu":
        # ReLU activation bounds output to [0, +inf), but use conservative upper bound
        lb = torch.zeros((hidden_size,), device=Bin.lb.device, dtype=Bin.lb.dtype)
        # Conservative upper bound based on input range
        input_max = torch.max(torch.abs(Bin.ub), torch.abs(Bin.lb))
        ub = torch.full((hidden_size,), float(input_max * 2), device=Bin.lb.device, dtype=Bin.lb.dtype)
    else:
        raise ValueError(f"Unsupported RNN nonlinearity: {nonlinearity}")
    
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_rnn(L.id, L.in_vars, L.out_vars, input_size, hidden_size, nonlinearity)
    
    return Fact(bounds=Bout, cons=cons)


def hybridz_tf_embedding(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for embedding lookup."""
    # Embedding lookup: discrete input indices -> continuous embeddings
    
    num_embeddings = L.meta.get("num_embeddings")
    embedding_dim = L.meta.get("embedding_dim")
    weight = L.params.get("weight")  # (num_embeddings, embedding_dim)
    
    if weight is not None:
        # Bounds based on embedding table values
        lb = torch.min(weight, dim=0)[0]  # Minimum over vocabulary
        ub = torch.max(weight, dim=0)[0]  # Maximum over vocabulary
    else:
        # Conservative bounds if no weight provided
        lb = torch.full((embedding_dim,), -1.0, device=Bin.lb.device, dtype=Bin.lb.dtype)
        ub = torch.full((embedding_dim,), 1.0, device=Bin.lb.device, dtype=Bin.lb.dtype)
    
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_embedding(L.id, L.in_vars, L.out_vars, num_embeddings, embedding_dim, weight)
    
    return Fact(bounds=Bout, cons=cons)