#===- act/back_end/core.py - ACT Core Data Structures ------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Core data structures for ACT verification framework including Layer,
#   Net, Bounds, and constraint set definitions.
#
#===---------------------------------------------------------------------===#

# core.py
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

# Import validation functions
from act.back_end.layer_util import validate_layer, validate_graph, validate_wrapper_graph

# Supported layer types: Please see them in act/back_end/layer_schema.py
@dataclass
class Layer:
    id: int                                     # Unique layer identifier
    kind: str                                   # UPPER name (e.g., "DENSE", "CONV2D", "RELU")
    params: Dict[str, torch.Tensor]            # Numeric tensors (weights, biases) on device
    meta: Dict[str, Any]                       # Non-numeric metadata (shapes, strides, etc.)
    in_vars: List[int]                         # Input variable indices 
    out_vars: List[int]                        # Output variable indices
    cache: Dict[str, torch.Tensor] = field(default_factory=dict)  # Runtime cache tensors

    def __post_init__(self):
        validate_layer(self)

    def is_validation(self) -> bool:
        return self.kind == "ASSERT"
    
    def get_bounds_for_var(self, fact: 'Fact', var_id: int, is_output: bool = True) -> Tuple[float, float]:
        """Get (lb, ub) for variable var_id from fact. Use is_output=True for out_vars, False for in_vars."""
        var_list = self.out_vars if is_output else self.in_vars
        
        if var_id not in var_list:
            raise ValueError(
                f"Variable {var_id} not in layer {self.id} "
                f"{'output' if is_output else 'input'} vars {var_list}"
            )
        
        # Find position of var_id in the list
        position = var_list.index(var_id)
        
        # Retrieve bounds at that position
        lb = fact.bounds.lb[position].item()
        ub = fact.bounds.ub[position].item()
        
        return lb, ub
    
    def get_all_var_bounds(self, fact: 'Fact', is_output: bool = True) -> Dict[int, Tuple[float, float]]:
        """Get dict of {var_id: (lb, ub)} for all variables. Use is_output=True for out_vars, False for in_vars."""
        var_list = self.out_vars if is_output else self.in_vars
        
        bounds_dict = {}
        for var_id in var_list:
            bounds_dict[var_id] = self.get_bounds_for_var(fact, var_id, is_output)
        
        return bounds_dict
    
    def get_input_shape(self) -> Optional[Tuple[int, ...]]:
        """Get input shape from metadata, or None if not stored."""
        if self.kind == "INPUT":
            return self.meta.get("shape")
        return self.meta.get("input_shape")
    
    def get_output_shape(self) -> Optional[Tuple[int, ...]]:
        """Get output shape from metadata, or None if not stored."""
        if self.kind == "INPUT":
            return self.meta.get("shape")
        return self.meta.get("output_shape")
    
    def get_num_input_vars(self) -> int:
        """Get number of input variables (flattened dimension)."""
        return len(self.in_vars)
    
    def get_num_output_vars(self) -> int:
        """Get number of output variables (flattened dimension)."""
        return len(self.out_vars)

@dataclass
class Net:
    layers: List[Layer]
    preds: Dict[int, List[int]]
    succs: Dict[int, List[int]]
    by_id: Dict[int, Layer] = field(init=False)
    
    def __post_init__(self):
        self.by_id = {L.id: L for L in self.layers}
        # Validate the graph structure
        validate_graph(self.layers)
        validate_wrapper_graph(self.layers)

    # helpers
    def last_validation(self) -> Optional[Layer]:
        for L in reversed(self.layers):
            if L.is_validation(): return L
        return None

    def assert_last_is_validation(self) -> None:
        if not self.layers or not self.layers[-1].is_validation():
            raise ValueError(f"Expected last layer to be ASSERT, got {self.layers[-1].kind if self.layers else 'EMPTY'}")
    
    def get_predecessor_bounds(self, layer_id: int, after: Dict[int, 'Fact'], 
                                before: Dict[int, 'Fact'], pred_index: int = 0) -> 'Bounds':
        """Get bounds from predecessor at pred_index (default 0) of layer_id."""
        if layer_id not in self.preds or pred_index >= len(self.preds[layer_id]):
            raise IndexError(f"Layer {layer_id} has no predecessor at index {pred_index}")
        
        pred_id = self.preds[layer_id][pred_index]
        return after[pred_id].bounds if pred_id in after else before[pred_id].bounds
    
    def get_all_predecessor_bounds(self, layer_id: int, after: Dict[int, 'Fact'], 
                                     before: Dict[int, 'Fact']) -> List['Bounds']:
        """Get list of bounds from all predecessors of layer_id."""
        if layer_id not in self.preds:
            return []
        return [self.get_predecessor_bounds(layer_id, after, before, i) 
                for i in range(len(self.preds[layer_id]))]
    
    def get_layer_shape(self, layer_id: int, facts: Dict[int, 'Fact'], 
                        is_output: bool = True) -> Tuple[int, ...]:
        """Get shape from bounds tensor in facts dict. Use is_output=True for output shape, False for input."""
        if layer_id not in facts:
            raise KeyError(f"Layer {layer_id} not found in facts dictionary")
        return facts[layer_id].bounds.lb.shape
        
        
@dataclass(eq=True, frozen=True)
class Bounds:
    lb: torch.Tensor
    ub: torch.Tensor
    def copy(self) -> "Bounds": return Bounds(self.lb.clone(), self.ub.clone())

@dataclass(eq=False)
class Con:
    kind: str                      # 'EQ' | 'INEQ' | 'BIN'
    var_ids: Tuple[int, ...]
    meta: Dict[str, Any] = field(default_factory=dict)
    # Optional numeric payloads (unused internally; only for compatibility)
    A: Any=None; b: Any=None; C: Any=None; d: Any=None
    def signature(self) -> Tuple[str, Tuple[int, ...], str]:
        return (self.kind, self.var_ids, self.meta.get("tag",""))

@dataclass
class ConSet:
    S: Dict[Tuple[str, Tuple[int, ...], str], Con] = field(default_factory=dict)
    
    def replace(self, c: Con): 
        self.S[c.signature()] = c
    
    def add_box(self, layer_id: int, var_ids: List[int], B: Bounds):
        self.replace(Con("INEQ", tuple(var_ids), {"tag": f"box:{layer_id}", "lb": B.lb.clone(), "ub": B.ub.clone()}))
    
    def __iter__(self):
        """Iterate over constraints (Con objects). Makes ConSet iterable."""
        return iter(self.S.values())
    
    def __len__(self):
        """Return number of constraints. Enables len(ConSet)."""
        return len(self.S)

@dataclass
class Fact:
    bounds: Bounds
    cons: ConSet