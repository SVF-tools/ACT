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
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union

# Import validation functions
from act.back_end.layer_util import validate_layer, validate_graph, validate_wrapper_graph
from act.back_end.layer_schema import LayerKind
from act.front_end.specs import InputGeometry, normalize_input_geometry

# Type alias for union-typed parameter values
ParamValue = Union[torch.Tensor, int, str, tuple, bool, None]

# Supported layer types: Please see them in act/back_end/layer_schema.py
@dataclass
class Layer:
    id: int                                     # Unique layer identifier
    kind: str                                   # UPPER name (e.g., "DENSE", "CONV2D", "RELU")
    params: Dict[str, Any]                     # Schema-validated, including nested metadata
    in_vars: List[int]                         # Input variable indices 
    out_vars: List[int]                        # Output variable indices
    cache: Dict[str, Any] = field(default_factory=dict)  # Runtime values and metadata
    input_geometry: Optional[InputGeometry] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        validate_layer(self)
        self.refresh_input_geometry()

    def refresh_input_geometry(self) -> None:
        """Rebuild derived geometry after construction or explicit sample slicing."""
        self.input_geometry = (
            normalize_input_geometry(self.params)
            if self.kind == LayerKind.INPUT_SPEC.value else None
        )

    def is_validation(self) -> bool:
        return self.kind == "ASSERT"
    
    def get_tensor(self, key: str, default: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Get tensor parameter. Returns default if missing or not a tensor."""
        val = self.params.get(key)
        if val is None or not isinstance(val, torch.Tensor):
            return default
        return val
    
    def get_scalar(self, key: str, default: Any = None) -> Any:
        """Get non-tensor parameter."""
        return self.params.get(key, default)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer parameter."""
        val = self.params.get(key, default)
        return int(val) if val is not None else default
    
    def get_tuple(self, key: str, default: tuple = ()) -> tuple:
        """Get tuple parameter."""
        val = self.params.get(key, default)
        return tuple(val) if val is not None else default
    
    def is_tensor(self, key: str) -> bool:
        """Check if parameter is a tensor."""
        return isinstance(self.params.get(key), torch.Tensor)
    
    

@dataclass
class Net:
    layers: List[Layer]
    preds: Dict[int, List[int]]
    succs: Dict[int, List[int]]
    by_id: Dict[int, Layer] = field(init=False)
    _topo_cache: Dict[bool, List[int]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _input_geometry_id: Optional[int] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        self.by_id = {L.id: L for L in self.layers}
        geometry_ids = [layer.id for layer in self.layers if layer.input_geometry is not None]
        if len(geometry_ids) > 1:
            raise ValueError("A Net supports at most one LP_EMBEDDING input specification")
        self._input_geometry_id = geometry_ids[0] if geometry_ids else None
        # Validate the graph structure
        validate_graph(self.layers)
        validate_wrapper_graph(self.layers)

    # helpers

    @property
    def input_geometry(self) -> Optional[InputGeometry]:
        """Return the canonical input domain without scanning the graph."""
        return (
            self.by_id[self._input_geometry_id].input_geometry
            if self._input_geometry_id is not None else None
        )

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
    
        
        
def topological_sort(net: Net, reverse: bool = False) -> List[int]:
    """Kahn topological order over the ACT DAG.

    ``reverse=False`` returns forward order (each layer after its
    predecessors); ``reverse=True`` returns reverse order (each layer after
    its successors). Neighbour lists are order-preserving deduped via
    ``dict.fromkeys`` so duplicate edges cannot inflate an in-degree that
    never reaches zero, while first-occurrence order keeps tie-breaking
    deterministic.

    Raises:
        ValueError: If the graph has a cycle or disconnected layers.
    """
    in_edges = net.succs if reverse else net.preds
    out_edges = net.preds if reverse else net.succs
    layer_ids = [layer.id for layer in net.layers]
    in_deg: Dict[int, int] = {
        lid: len(dict.fromkeys(in_edges.get(lid, []))) for lid in layer_ids
    }
    queue = deque(lid for lid in layer_ids if in_deg[lid] == 0)
    order: List[int] = []
    while queue:
        lid = queue.popleft()
        order.append(lid)
        for nxt in dict.fromkeys(out_edges.get(lid, [])):
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(layer_ids):
        raise ValueError(
            f"topological_sort: graph has cycle or disconnected layers "
            f"({len(order)}/{len(layer_ids)} sorted)"
        )
    return order


def get_topo_order(net: Net, reverse: bool = False) -> List[int]:
    """Memoized ``topological_sort(net, reverse)`` keyed on ``reverse``.

    Caches the forward and reverse Kahn orders on the (mutable) ``Net`` in
    ``net._topo_cache``; returns the identical list ``topological_sort`` would.
    The cache is valid only while the layer set is immutable (as during BaB); a
    cheap cache-hit guard asserts the current layer-id set still matches.
    """
    cache = net._topo_cache
    order = cache.get(reverse)
    if order is None:
        order = topological_sort(net, reverse)
        cache[reverse] = order
        return order
    assert set(order) == {layer.id for layer in net.layers}, (
        "get_topo_order: net layer set changed since cache was populated"
    )
    return order


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
        self.replace(Con("INEQ", tuple(var_ids),
                         {"tag": f"box:{layer_id}",
                          "lb": B.lb.detach().reshape(-1).clone(),
                          "ub": B.ub.detach().reshape(-1).clone()}))
        
    
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
