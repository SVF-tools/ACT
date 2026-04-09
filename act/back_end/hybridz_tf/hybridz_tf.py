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
from typing import Dict, List, Optional
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.transfer_functions import TransferFunction
from act.back_end.solver.hz_bounds import (
    HZono, hz_multiply, hz_add_const, hz_minkowski_sum, hz_from_bounds, hz_reduce,
    hz_compute_bounds, hz_apply_relu, hz_apply_leaky_relu, hz_apply_tanh, hz_apply_sigmoid,
    hz_conv2d,
)
from act.back_end.hybridz_tf.tf_mlp import *
from act.back_end.hybridz_tf.tf_cnn import *
from act.back_end.hybridz_tf.tf_rnn import *
from act.back_end.hybridz_tf.tf_transformer import *


class HybridzTF(TransferFunction):
    """HybridZ-based transfer functions with zonotope operations."""
    
    def __init__(self):
        self._hz_cache: Dict[int, HZono] = {}
        self._cache_net_id: Optional[int] = None
        self._tanh_K: int = 2
        self._sigmoid_K: int = 2
    
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
        "RELU": lambda L, bounds, tf: hybridz_tf_relu(L, bounds),
        "LRELU": lambda L, bounds, tf: hybridz_tf_lrelu(L, bounds),
        "TANH": lambda L, bounds, tf: hybridz_tf_tanh(L, bounds),
        "SIGMOID": lambda L, bounds, tf: hybridz_tf_sigmoid(L, bounds), 
        "ABS": lambda L, bounds, tf: hybridz_tf_abs(L, bounds),
        
        # Multi-input operations  
        "ADD": lambda L, bounds, tf: hybridz_tf_add(L, 
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0), 
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)),
        "MUL": lambda L, bounds, tf: hybridz_tf_mul(L,
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
            tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)),
        
        # CNN operations
        "CONV2D": lambda L, bounds, tf: hybridz_tf_conv2d(L, bounds),
        "MAXPOOL2D": lambda L, bounds, tf: hybridz_tf_maxpool2d(L, bounds),
        "AVGPOOL2D": lambda L, bounds, tf: hybridz_tf_avgpool2d(L, bounds),
        "FLATTEN": lambda L, bounds, tf: hybridz_tf_flatten(L, bounds),
        "RESHAPE": lambda L, bounds, tf: hybridz_tf_reshape(L, bounds),
        
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
    
    # Max input dimension for HZ tracking.
    _HZ_MAX_INPUT_DIM = 1024
    
    def _hz_from_bounds(self, bounds: Bounds) -> Optional[HZono]:
        """Create initial HZ from Bounds: c=(lb+ub)/2, Gc=diag((ub-lb)/2)."""
        lb, ub = bounds.lb.flatten(), bounds.ub.flatten()
        n = lb.shape[0]
        if n > self._HZ_MAX_INPUT_DIM:
            return None
        dtype, device = lb.dtype, lb.device
        c = ((lb + ub) / 2.0).view(-1, 1)
        rad = (ub - lb) / 2.0
        return HZono(c=c, Gc=torch.diag(rad),
                     Gb=torch.zeros((n, 0), dtype=dtype, device=device),
                     Ac=torch.zeros((0, n), dtype=dtype, device=device),
                     Ab=torch.zeros((0, 0), dtype=dtype, device=device),
                     b=torch.zeros((0, 1), dtype=dtype, device=device))
    
    def _hz_transform(self, L: Layer, hz_in: HZono) -> Optional[HZono]:
        """Compute HZ output for layer L. Returns None if unsupported."""
        k = L.kind.upper()
        dtype, device = hz_in.c.dtype, hz_in.c.device
        
        if k == "DENSE":
            hz = hz_multiply(hz_in, L.params["weight"])
            b = L.params.get("bias")
            if b is not None:
                b_col = b.to(dtype=dtype, device=device)
                hz = hz_add_const(hz, b_col.view(-1, 1) if b_col.ndim == 1 else b_col)
            return hz
        if k == "BIAS":
            c = L.params["c"].to(dtype=dtype, device=device)
            return hz_add_const(hz_in, c.view(-1, 1) if c.ndim == 1 else c)
        if k == "SCALE":
            a = L.params["a"].to(dtype=dtype, device=device).flatten()
            return hz_multiply(hz_in, torch.diag(a))
        if k == "RELU":
            return hz_reduce(hz_apply_relu(hz_in))
        if k == "LRELU":
            return hz_reduce(hz_apply_leaky_relu(hz_in, float(L.params.get("negative_slope", 0.01))))
        if k == "TANH":
            return hz_apply_tanh(hz_in, K=self._tanh_K)
        if k == "SIGMOID":
            return hz_apply_sigmoid(hz_in, K=self._sigmoid_K)
        if k == "ABS":
            bds = hz_compute_bounds(hz_in)
            lb_out = torch.where(bds.lb >= 0, bds.lb, torch.where(bds.ub <= 0, -bds.ub, torch.zeros_like(bds.lb)))
            return hz_from_bounds(Bounds(lb=lb_out, ub=torch.maximum(bds.lb.abs(), bds.ub.abs())), dtype, device)
        if k == "CONV2D":
            return hz_conv2d(hz_in, L.params["weight"], L.params.get("bias"),
                              L.params.get("stride", 1), L.params.get("padding", 0),
                              L.params.get("dilation", 1), L.params.get("groups", 1),
                              L.params.get("input_shape"))
        if k == "MAXPOOL2D":
            import torch.nn.functional as F
            bds = hz_compute_bounds(hz_in)
            shape = L.params.get("input_shape")
            if shape is not None:
                C, H, W = shape[-3:]
                _, idx = F.max_pool2d(bds.lb.view(1, C, H, W),
                                      kernel_size=L.params.get("kernel_size", 2),
                                      stride=L.params.get("stride", L.params.get("kernel_size", 2)),
                                      padding=L.params.get("padding", 0), return_indices=True)
                w = idx.reshape(-1)
                return HZono(c=hz_in.c[w], Gc=hz_in.Gc[w], Gb=hz_in.Gb[w],
                             Ac=hz_in.Ac.clone(), Ab=hz_in.Ab.clone(), b=hz_in.b.clone())
            return None
        if k in ("FLATTEN", "RESHAPE"):
            return hz_in
        if k == "ADD":
            preds = self._net.preds.get(L.id, [])
            hz2 = self._hz_cache.get(preds[1]) if len(preds) > 1 else None
            return hz_minkowski_sum(hz_in, hz2) if hz2 is not None else None
        if k == "MUL":
            preds = self._net.preds.get(L.id, [])
            hz2 = self._hz_cache.get(preds[1]) if len(preds) > 1 else None
            if hz2 is not None:
                b1, b2 = hz_compute_bounds(hz_in), hz_compute_bounds(hz2)
                corners = torch.stack([b1.lb*b2.lb, b1.lb*b2.ub, b1.ub*b2.lb, b1.ub*b2.ub])
                return hz_from_bounds(Bounds(lb=corners.min(0)[0], ub=corners.max(0)[0]), dtype, device)
            return None
        
        return None  # No HZ transform for this layer type
    
    def apply(self, L: Layer, input_bounds: Bounds, net: Net,
              before: Dict[int, Fact], after: Dict[int, Fact]) -> Fact:
        """Apply HybridZ transfer function to layer L."""
        k = L.kind.upper()
        if k not in self._LAYER_REGISTRY:
            raise NotImplementedError(f"HybridzTF: Unsupported layer kind '{k}'")
        
        # Reset cache if network changed
        net_id = id(net)
        if self._cache_net_id != net_id:
            self._hz_cache.clear()
            self._cache_net_id = net_id
        
        # Store context for lambdas
        self._net = net
        self._before = before
        self._after = after
        
        # Seed HZ cache for input layers
        if k in ("INPUT", "INPUT_SPEC"):
            hz_init = self._hz_from_bounds(input_bounds)
            if hz_init is not None:
                self._hz_cache[L.id] = hz_init
        
        # Propagate HZ from predecessor
        if k not in ("INPUT", "INPUT_SPEC", "ASSERT"):
            preds = net.preds.get(L.id, [])
            if preds and preds[0] in self._hz_cache:
                self._hz_cache[L.id] = self._hz_cache[preds[0]]
        
        # HZ processing before entering the transfer function
        hz_in = self._hz_cache.get(L.id)
        hz_bounds = None
        if hz_in is not None:
            hz_out = self._hz_transform(L, hz_in)
            if hz_out is not None:
                self._hz_cache[L.id] = hz_out
                hz_bounds = hz_compute_bounds(hz_out)
        
        # Call transfer function (original signature, no tf)
        transfer_fn = self._LAYER_REGISTRY[k]
        fact = transfer_fn(L, input_bounds, self)
        
        # Create fresh HZ for layers without HZ transform
        if hz_in is not None and hz_bounds is None:
            self._hz_cache[L.id] = hz_from_bounds(
                fact.bounds, fact.bounds.lb.dtype, fact.bounds.lb.device)
        
        # Return HZ bounds if tighter
        if hz_bounds is not None:
            return Fact(bounds=hz_bounds, cons=fact.cons)
        return fact