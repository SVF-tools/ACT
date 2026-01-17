#===- act/pipeline/torch2act.py - Torch to ACT Converter ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Spec-free PyTorch → ACT converter for verification. Converts wrapped
#   PyTorch models (containing InputLayer, InputSpecLayer, and OutputSpecLayer)
#   into ACT Net graphs with embedded constraints for formal verification.
#
# Key Features:
#   - Spec-free: Constraints embedded in model, not passed separately
#   - Input-free: Input specifications extracted from wrapper layers
#   - Bidirectional: Paired with act2torch.py for round-trip conversion
#   - Weight preservation: Transfers all model parameters to ACT format
#   - Unified tracing: Graph-based parsing for models
#
# Architecture:
#   InputLayer           → INPUT      (declares input shape/dtype/device)
#   InputSpecLayer       → INPUT_SPEC (input constraints: BOX, L_INF, LIN_POLY)
#   nn.Linear            → DENSE      (fully connected layers)
#   nn.Conv2d            → CONV2D     (convolutional layers)
#   nn.ReLU              → RELU       (activation functions)
#   OutputSpecLayer      → ASSERT     (output constraints: SAFETY, classification)
#
# Contract:
#   - Exactly one InputLayer must be present (defines input shape)
#   - Optional InputSpecLayer for input constraints
#   - Optional OutputSpecLayer for output constraints
#   - All wrapper layers converted to ACT layer graph
#
# Usage:
#   
#   # Convert wrapped PyTorch model to ACT Net
#   converter = TorchToACT(pytorch_model)
#   act_net = converter.run()
#   
#   # ACT Net ready for verification
#   from act.back_end.verifier import verify_once
#   result = verify_once(act_net)
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn
import torch.fx as fx
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.ops import StochasticDepth

from act.util.model_inference import model_inference
from act.front_end.model_synthesis import model_synthesis
from act.back_end.core import Net, Layer
from act.back_end.layer_schema import LayerKind
from act.back_end.layer_util import create_layer
from act.back_end.solver.solver_torch import TorchLPSolver
from act.back_end.solver.solver_gurobi import GurobiSolver
from act.util.options import PerformanceOptions
from act.pipeline.verification.utils import _prod, _normalize_tuple



# -----------------------------------------------------------------------------
# Unified graph-based tracing for PyTorch models
# -----------------------------------------------------------------------------

class _LayerGraphBuilder:
    """
    Build ACT layer graph from nn.Module.
    
    Works with any nn.Module - uses torch.fx for graph extraction and onnx2pytorch models. The resulting graph is a DAG.
    """
    
    # Dispatch tables for FX call_method operations
    _METADATA_METHODS = frozenset({'size', 'dim', 'numel'})
    _PASSTHROUGH_METHODS = frozenset({'contiguous', 'to', 'float', 'double', 'half', 'cpu', 'cuda', 'detach'})
    _RESHAPE_METHODS = frozenset({'view', 'reshape', 'flatten'})
    
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...], dtype: torch.dtype = torch.float64):
        self.model = model
        self.input_shape = input_shape
        self.dtype = dtype
        
        # Layer building state
        self.layers: List[Layer] = []
        self.next_var = 0
        self.prev_out: List[int] = []
        self.shape: Tuple[int, ...] = input_shape
        
        # Graph tracking (populated by FX or ONNX path)
        self.node_outputs: Dict[str, List[int]] = {}
        self.node_shapes: Dict[str, Tuple[int, ...]] = {}
        self.node_to_layer_id: Dict[str, int] = {}
        self.graph_edges: Dict[str, List[str]] = {}
        self.modules: Dict[str, nn.Module] = {}
        
        # torch.fx specific
        self.fx_graph: Optional[fx.Graph] = None
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    def build_layer_graph(self) -> Tuple[List[Layer], Dict[int, List[int]], Dict[int, List[int]]]:
        """
        Build ACT layer graph from the model.
        
        Returns:
            Tuple of (layers, preds, succs) forming a DAG
        """
        # Initialize input vars
        n_inputs = _prod(self.input_shape)
        self.prev_out = self._alloc_ids(n_inputs)
        
        # Extract computation graph
        onnx_graph, onnx_mapping = self._extract_graph()
        
        # Pre-register placeholder nodes (network inputs)
        self._pre_register_nodes()
        
        # Process graph
        if onnx_graph is not None and onnx_mapping is not None:
            self._process_onnx_graph(onnx_graph, onnx_mapping)
        else:
            self._process_fx_graph()
        
        # Build and validate graph structure
        preds, succs = self._build_preds_succs()
        return self.layers, preds, succs
        
    def _alloc_ids(self, n: int) -> List[int]:
        """Allocate n consecutive variable IDs."""
        ids = list(range(self.next_var, self.next_var + n))
        self.next_var += n
        return ids
    
    def _same_size_forward(self) -> List[int]:
        """Allocate same number of output vars as current prev_out."""
        return self._alloc_ids(len(self.prev_out))
    
    def _add_layer(self, kind: str, params: Dict[str, torch.Tensor], meta: Dict[str, Any],
                   in_vars: List[int], out_vars: List[int]) -> int:
        """Add a layer and return its ID."""
        layer = create_layer(
            id=len(self.layers),
            kind=kind,
            params=params,
            meta=meta,
            in_vars=in_vars,
            out_vars=out_vars,
        )
        self.layers.append(layer)
        return layer.id
    
    def _register_node(self, name: str, layer_id: Optional[int] = None) -> None:
        """Register node's output vars, shape, and layer mapping."""
        self.node_outputs[name] = self.prev_out.copy()
        self.node_shapes[name] = self.shape
        self.node_to_layer_id[name] = layer_id if layer_id is not None else (len(self.layers) - 1)
    
    def _pre_register_nodes(self) -> None:
        """Pre-register nodes so successor nodes can look up input vars."""
        if self.fx_graph is None:
            return
        for node in self.fx_graph.nodes:
            if node.op == 'placeholder':
                self.node_outputs[node.name] = self.prev_out.copy()
                self.node_shapes[node.name] = self.shape
                self.node_to_layer_id[node.name] = -1
    
    def _get_predecessor_state(self, node: fx.Node) -> bool:
        """Set state from first valid predecessor. Returns True if found."""
        if node.args and isinstance(node.args[0], fx.Node):
            pred_name = node.args[0].name
            if pred_name in self.node_outputs:
                self.prev_out = self.node_outputs[pred_name].copy()
                self.shape = self.node_shapes[pred_name]
                return True
        return False
    
    # -------------------------------------------------------------------------
    # Model tracing
    # -------------------------------------------------------------------------
    
    def _extract_graph(self) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """Extract computation graph (torch.fx or ONNX)."""
        # Check for onnx2pytorch model first
        if hasattr(self.model, 'onnx_model') and hasattr(self.model, 'mapping'):
            try:
                onnx_model = self.model.onnx_model
                if hasattr(onnx_model, 'graph') and hasattr(onnx_model.graph, 'node'):
                    onnx_graph = onnx_model.graph
                    self.modules = dict(self.model.named_modules())
                    self._build_onnx_graph_edges(onnx_graph)
                    return onnx_graph, self.model.mapping
            except Exception as e:
                warnings.warn(f"ONNX graph parsing failed: {e}, trying torch.fx")
        
        # Use torch.fx tracing
        try:
            traced = fx.symbolic_trace(self.model)
            self.fx_graph = traced.graph
            self.modules = dict(traced.named_modules())
            self._build_fx_graph_edges()
            return None, None
        except Exception as e:
            raise RuntimeError(f"Failed to extract graph with torch.fx: {e}")
    
    def _build_fx_graph_edges(self) -> None:
        """Build graph edge dictionary from torch.fx graph."""
        if self.fx_graph is None:
            return
        for node in self.fx_graph.nodes:
            self.graph_edges[node.name] = [
                arg.name for arg in node.args if isinstance(arg, fx.Node)
            ]
    
    def _build_onnx_graph_edges(self, onnx_graph: Any) -> None:
        """Build graph edge dictionary from ONNX graph."""
        output_to_node: Dict[str, Any] = {}
        for node in onnx_graph.node:
            for output in node.output:
                output_to_node[output] = node
        
        graph_inputs = {gi.name for gi in onnx_graph.input}
        for node in onnx_graph.node:
            for output in node.output:
                preds = [inp for inp in node.input 
                         if inp in output_to_node or inp in graph_inputs]
                self.graph_edges[output] = preds
    
    # -------------------------------------------------------------------------
    # FX Graph Processing
    # -------------------------------------------------------------------------
    
    def _process_fx_graph(self) -> None:
        """Process model using torch.fx graph."""
        if self.fx_graph is None:
            return
        for node in self.fx_graph.nodes:
            if node.op == 'placeholder':
                self._handle_placeholder(node)
            elif node.op == 'call_module':
                self._handle_call_module(node)
            elif node.op == 'call_function':
                self._handle_call_function(node)
            elif node.op == 'call_method':
                self._handle_call_method(node)
            elif node.op == 'get_attr':
                self._handle_get_attr(node)
            elif node.op == 'output':
                self._handle_output(node)
    
    def _handle_placeholder(self, node: fx.Node) -> None:
        """Handle placeholder node - already pre-registered."""
        pass
    
    def _handle_call_module(self, node: fx.Node) -> None:
        """Handle call_module node."""
        module = self.modules.get(node.target)
        if module is None:
            raise ValueError(f"Module '{node.target}' not found in traced model")
        
        self._get_predecessor_state(node)
        self._convert_module(module)
        self._register_node(node.name)
    
    def _handle_call_function(self, node: fx.Node) -> None:
        """Handle call_function node."""
        target_name = str(node.target).lower()
        
        handlers = {
            'add': self._process_add_operation,
            'cat': self._process_concat_operation,
            'concat': self._process_concat_operation,
            'flatten': self._process_flatten_function,
            'mul': self._process_mul_operation,
            'mean': self._process_mean_operation,
            'getitem': self._process_getitem_operation,
            'stochastic_depth': self._process_passthrough_function,
            'dropout': self._process_passthrough_function,
        }
        
        for key, handler in handlers.items():
            if key in target_name:
                handler(node)
                return
        
        raise NotImplementedError(
            f"Unsupported function in graph: {node.target}\n"
            f"  Add support in _handle_call_function() or use a simpler model."
        )
    
    def _handle_call_method(self, node: fx.Node) -> None:
        """Handle call_method node."""
        method_name = node.target
        
        if method_name in self._METADATA_METHODS:
            # Return ints/tuples, not tensors - just register for graph continuity
            if node.args and isinstance(node.args[0], fx.Node):
                pred_name = node.args[0].name
                if pred_name in self.node_to_layer_id:
                    self.node_to_layer_id[node.name] = self.node_to_layer_id[pred_name]
        
        elif method_name in self._PASSTHROUGH_METHODS:
            if node.args and isinstance(node.args[0], fx.Node):
                pred_name = node.args[0].name
                if pred_name in self.node_outputs:
                    self.node_outputs[node.name] = self.node_outputs[pred_name].copy()
                    self.node_shapes[node.name] = self.node_shapes[pred_name]
                    self.node_to_layer_id[node.name] = self.node_to_layer_id.get(pred_name, len(self.layers) - 1)
                    self.prev_out = self.node_outputs[node.name]
                    self.shape = self.node_shapes[node.name]
        
        elif method_name in self._RESHAPE_METHODS:
            if self._get_predecessor_state(node):
                self._create_flatten_layer(node.name)
        
        else:
            raise NotImplementedError(
                f"Unsupported tensor method: .{method_name}()\n"
                f"  Add support in _handle_call_method() or use explicit layers."
            )
    
    def _handle_get_attr(self, node: fx.Node) -> None:
        """Handle get_attr node."""
        if node.args and isinstance(node.args[0], fx.Node):
            pred_name = node.args[0].name
            if pred_name in self.node_to_layer_id:
                self.node_to_layer_id[node.name] = self.node_to_layer_id[pred_name]
    
    def _handle_output(self, node: fx.Node) -> None:
        """Handle output node."""
        if node.args and isinstance(node.args[0], fx.Node):
            pred_name = node.args[0].name
            if pred_name in self.node_outputs:
                self.prev_out = self.node_outputs[pred_name].copy()
                self.shape = self.node_shapes[pred_name]
    
    # -------------------------------------------------------------------------
    # ONNX Graph Processing
    # -------------------------------------------------------------------------
    
    def _process_onnx_graph(self, onnx_graph: Any, onnx_mapping: Dict[str, Any]) -> None:
        """Process model using ONNX graph."""
        # Register graph inputs
        input_names = [gi.name for gi in onnx_graph.input 
                       if not any(init.name == gi.name for init in onnx_graph.initializer)]
        if input_names:
            self.node_outputs[input_names[0]] = self.prev_out.copy()
            self.node_shapes[input_names[0]] = self.shape
            self.node_to_layer_id[input_names[0]] = -1
        
        # ONNX operation handlers
        onnx_handlers = {
            'Conv': lambda n, o: self._process_onnx_module(n, o, onnx_mapping, nn.Conv2d, self._convert_conv2d),
            'Gemm': lambda n, o: self._process_onnx_module(n, o, onnx_mapping, nn.Linear, self._convert_linear),
            'MatMul': lambda n, o: self._process_onnx_module(n, o, onnx_mapping, nn.Linear, self._convert_linear),
            'MaxPool': lambda n, o: self._process_onnx_module(n, o, onnx_mapping, nn.MaxPool2d, self._convert_pool2d),
            'AveragePool': lambda n, o: self._process_onnx_module(n, o, onnx_mapping, nn.AvgPool2d, self._convert_pool2d),
            'BatchNormalization': lambda n, o: self._process_onnx_batchnorm(n, o, onnx_mapping),
            'Relu': self._process_onnx_relu,
            'Add': self._process_onnx_add,
            'GlobalAveragePool': self._process_onnx_global_avgpool,
            'Flatten': self._process_onnx_flatten,
            'Reshape': self._process_onnx_reshape,
        }
        
        for onnx_node in onnx_graph.node:
            if not onnx_node.output:
                continue
            
            output_name = onnx_node.output[0]
            self._set_onnx_predecessor_state(onnx_node)
            
            handler = onnx_handlers.get(onnx_node.op_type)
            if handler:
                handler(onnx_node, output_name)
            else:
                raise NotImplementedError(
                    f"Unsupported ONNX operation: {onnx_node.op_type}\n"
                    f"  Node: {onnx_node.name}"
                )
        
        # Set final output state
        output_names = [go.name for go in onnx_graph.output]
        if output_names and output_names[0] in self.node_outputs:
            self.prev_out = self.node_outputs[output_names[0]].copy()
            self.shape = self.node_shapes[output_names[0]]
    
    def _set_onnx_predecessor_state(self, onnx_node: Any) -> None:
        """Set state from ONNX node's first valid input."""
        for inp in onnx_node.input:
            if inp in self.node_outputs:
                self.prev_out = self.node_outputs[inp].copy()
                self.shape = self.node_shapes[inp]
                break
    
    def _get_onnx_module(self, output_name: str, onnx_mapping: Dict[str, Any]) -> Optional[nn.Module]:
        """Find PyTorch module corresponding to ONNX node output."""
        if output_name in onnx_mapping:
            module_name = onnx_mapping[output_name]
            if isinstance(module_name, str) and module_name in self.modules:
                return self.modules[module_name]
        return None
    
    def _process_onnx_module(self, node: Any, output_name: str, onnx_mapping: Dict[str, Any],
                             expected_type: type, converter: Any) -> None:
        """Generic ONNX handler for ops with PyTorch module mapping."""
        module = self._get_onnx_module(output_name, onnx_mapping)
        if module is not None and isinstance(module, expected_type):
            converter(module)
        else:
            raise NotImplementedError(f"ONNX {node.op_type} without PyTorch module: {node.name}")
        self._register_node(output_name)
    
    def _process_onnx_batchnorm(self, node: Any, output_name: str, onnx_mapping: Dict[str, Any]) -> None:
        """Process ONNX BatchNormalization."""
        module = self._get_onnx_module(output_name, onnx_mapping)
        if module is not None:
            bn = module.bnu if hasattr(module, 'bnu') else module
            if isinstance(bn, _BatchNorm):
                self._convert_batchnorm(bn)
        self._register_node(output_name, len(self.layers) - 1 if self.layers else 0)
    
    def _process_onnx_relu(self, node: Any, output_name: str) -> None:
        """Process ONNX Relu."""
        out_vars = self._same_size_forward()
        self._add_layer(LayerKind.RELU.value, {}, {}, self.prev_out, out_vars)
        self.prev_out = out_vars
        self._register_node(output_name)
    
    def _process_onnx_add(self, node: Any, output_name: str) -> None:
        """Process ONNX Add."""
        inputs = [inp for inp in node.input if inp in self.node_outputs]
        if len(inputs) >= 2:
            x_vars = self.node_outputs[inputs[0]]
            y_vars = self.node_outputs[inputs[1]]
            x_shape = self.node_shapes[inputs[0]]
            
            out_vars = self._alloc_ids(len(x_vars))
            layer_id = self._add_layer(
                LayerKind.ADD.value, {},
                {"x_vars": x_vars, "y_vars": y_vars, "input_shape": x_shape, "output_shape": x_shape},
                x_vars + y_vars, out_vars
            )
            self.prev_out = out_vars
            self.shape = x_shape
            self._register_node(output_name, layer_id)
    
    def _process_onnx_global_avgpool(self, node: Any, output_name: str) -> None:
        """Process ONNX GlobalAveragePool."""
        if len(self.shape) == 4:
            batch, channels, h, w = self.shape
            output_shape = (1, channels, 1, 1)
            out_vars = self._alloc_ids(channels)
            self._add_layer(LayerKind.ADAPTIVEAVGPOOL2D.value, {}, {"output_size": (1, 1)},
                           self.prev_out, out_vars)
            self.shape = output_shape
            self.prev_out = out_vars
        self._register_node(output_name)
    
    def _process_onnx_flatten(self, node: Any, output_name: str) -> None:
        """Process ONNX Flatten."""
        self._create_flatten_layer(output_name)
    
    def _process_onnx_reshape(self, node: Any, output_name: str) -> None:
        """Process ONNX Reshape."""
        self._register_node(output_name, len(self.layers) - 1 if self.layers else 0)
    
    # -------------------------------------------------------------------------
    # Graph Structure Building
    # -------------------------------------------------------------------------
    
    def _build_preds_succs(self) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """Build preds and succs dictionaries, validating DAG property.
        
        Uses a hybrid approach:
        1. Build edges from FX/ONNX graph for mapped nodes
        2. For unmapped intermediate layers (e.g., SCALE from BatchNorm), 
           connect them sequentially based on layer creation order
        """
        n_layers = len(self.layers)
        
        # Identify which layers are mapped to FX/ONNX nodes
        mapped_layer_ids = set(lid for lid in self.node_to_layer_id.values() if lid >= 0)
        
        preds: Dict[int, List[int]] = {i: [] for i in range(n_layers)}
        succs: Dict[int, List[int]] = {i: [] for i in range(n_layers)}
        
        # Track layers that take input directly from the placeholder (network input)
        takes_input_from_placeholder: Set[int] = set()
        
        # First, build FX/ONNX graph edges for mapped layers
        for node_name, pred_names in self.graph_edges.items():
            layer_id = self.node_to_layer_id.get(node_name)
            if layer_id is None or layer_id < 0:
                continue
            
            for pred_name in pred_names:
                pred_layer_id = self.node_to_layer_id.get(pred_name)
                if pred_layer_id is None:
                    continue
                
                if pred_layer_id < 0:
                    # Predecessor is the input placeholder - this layer takes network input
                    takes_input_from_placeholder.add(layer_id)
                    continue
                
                if pred_layer_id != layer_id and pred_layer_id not in preds[layer_id]:
                    preds[layer_id].append(pred_layer_id)
        
        # Second, connect unmapped layers (SCALE, BIAS from BatchNorm, etc.) sequentially
        # These layers are internal to a multi-layer conversion and should connect to i-1
        for i in range(1, n_layers):
            if i not in mapped_layer_ids:
                # Unmapped layer - must connect to previous layer
                preds[i] = [i - 1]
            elif not preds[i] and i not in takes_input_from_placeholder:
                # Mapped but no FX predecessors AND not taking input from placeholder
                # -> connect to previous layer (internal layer within multi-layer conversion)
                preds[i] = [i - 1]
            # else: Mapped layer with FX predecessors OR takes network input - keep as-is
        
        # Build succs from preds
        for i in range(n_layers):
            for pred_id in preds[i]:
                if i not in succs[pred_id]:
                    succs[pred_id].append(i)
        
        self._assert_dag(preds, succs, n_layers)
        return preds, succs
    
    def _assert_dag(self, preds: Dict[int, List[int]], succs: Dict[int, List[int]], n_layers: int) -> None:
        """Assert graph is a DAG using Kahn's algorithm."""
        if n_layers == 0:
            return
        
        in_degree = {i: len(preds.get(i, [])) for i in range(n_layers)}
        queue = [i for i in range(n_layers) if in_degree[i] == 0]
        visited = 0
        
        while queue:
            node = queue.pop(0)
            visited += 1
            for succ in succs.get(node, []):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        
        if visited != n_layers:
            cycle_nodes = [i for i in range(n_layers) if in_degree[i] > 0]
            raise ValueError(f"Layer graph contains a cycle! Nodes: {cycle_nodes}")
    
    # -------------------------------------------------------------------------
    # Layer Conversion - Module Dispatcher
    # -------------------------------------------------------------------------
    
    def _convert_module(self, mod: nn.Module) -> None:
        """Convert a PyTorch module to ACT layer(s)."""
        converters = {
            nn.Flatten: self._convert_flatten,
            nn.Linear: self._convert_linear,
            nn.ReLU: self._convert_relu,
            nn.Conv2d: self._convert_conv2d,
            nn.MaxPool2d: self._convert_pool2d,
            nn.AvgPool2d: self._convert_pool2d,
            nn.AdaptiveAvgPool2d: self._convert_adaptive_avgpool2d,
            _BatchNorm: self._convert_batchnorm,
            nn.SiLU: lambda m: self._convert_activation(m, LayerKind.SILU),
            nn.Sigmoid: lambda m: self._convert_activation(m, LayerKind.SIGMOID),
            nn.Tanh: lambda m: self._convert_activation(m, LayerKind.TANH),
            nn.LeakyReLU: lambda m: self._convert_activation(m, LayerKind.LRELU, {"negative_slope": m.negative_slope}),
        }
        
        # No-op modules
        if isinstance(mod, (nn.Dropout, StochasticDepth)):
            return
        
        for mod_type, converter in converters.items():
            if isinstance(mod, mod_type):
                converter(mod)
                return
        
        raise NotImplementedError(f"Unsupported module: {type(mod).__name__}")
    
    # -------------------------------------------------------------------------
    # Layer Conversion - Specific Converters
    # -------------------------------------------------------------------------
    
    def _create_flatten_layer(self, node_name: Optional[str] = None) -> List[int]:
        """Create FLATTEN layer, optionally register node."""
        out_vars = self._same_size_forward()
        output_shape = (1, _prod(self.shape[1:]))
        layer_id = self._add_layer(
            LayerKind.FLATTEN.value, {},
            {"input_shape": self.shape, "output_shape": output_shape},
            self.prev_out, out_vars
        )
        self.prev_out = out_vars
        self.shape = output_shape
        if node_name:
            self._register_node(node_name, layer_id)
        return out_vars
    
    def _convert_flatten(self, mod: nn.Flatten) -> None:
        """Convert nn.Flatten."""
        self._create_flatten_layer()
    
    def _convert_linear(self, mod: nn.Linear) -> None:
        """Convert nn.Linear to DENSE layer."""
        out_features = int(mod.out_features)
        W = mod.weight.detach()
        b = mod.bias.detach() if mod.bias is not None else torch.zeros(out_features, dtype=W.dtype, device=W.device)
        
        out_vars = self._alloc_ids(out_features)
        self._add_layer(
            LayerKind.DENSE.value,
            {"W": W, "b": b},
            {"input_shape": self.shape, "output_shape": (1, out_features),
             "in_features": int(mod.in_features), "out_features": out_features,
             "bias_enabled": mod.bias is not None},
            self.prev_out, out_vars
        )
        self.shape = (1, out_features)
        self.prev_out = out_vars
    
    def _convert_relu(self, mod: nn.ReLU) -> None:
        """Convert nn.ReLU."""
        out_vars = self._same_size_forward()
        self._add_layer(LayerKind.RELU.value, {},
                       {"input_shape": self.shape, "output_shape": self.shape},
                       self.prev_out, out_vars)
        self.prev_out = out_vars
    
    def _convert_conv2d(self, mod: nn.Conv2d) -> None:
        """Convert nn.Conv2d."""
        weight = mod.weight.detach()
        bias = mod.bias.detach() if mod.bias is not None else None
        
        # Infer input shape if flattened
        if len(self.shape) == 2:
            n_features = self.shape[1]
            channels = mod.in_channels
            spatial = int((n_features / channels) ** 0.5)
            input_shape = (1, channels, spatial, spatial)
        else:
            input_shape = self.shape
        
        batch, in_c, in_h, in_w = input_shape
        out_c = mod.out_channels
        out_h = (in_h + 2 * mod.padding[0] - mod.dilation[0] * (mod.kernel_size[0] - 1) - 1) // mod.stride[0] + 1
        out_w = (in_w + 2 * mod.padding[1] - mod.dilation[1] * (mod.kernel_size[1] - 1) - 1) // mod.stride[1] + 1
        output_shape = (1, out_c, out_h, out_w)
        
        params = {"weight": weight}
        if bias is not None:
            params["bias"] = bias
        
        out_vars = self._alloc_ids(out_c * out_h * out_w)
        self._add_layer(
            LayerKind.CONV2D.value, params,
            {"input_shape": input_shape, "output_shape": output_shape,
             "kernel_size": mod.kernel_size, "stride": mod.stride,
             "padding": mod.padding, "dilation": mod.dilation,
             "groups": mod.groups, "in_channels": in_c, "out_channels": out_c},
            self.prev_out, out_vars
        )
        self.shape = output_shape
        self.prev_out = out_vars
    
    def _convert_pool2d(self, mod: Union[nn.MaxPool2d, nn.AvgPool2d]) -> None:
        """Convert MaxPool2d or AvgPool2d."""
        if len(self.shape) != 4:
            raise ValueError(f"Pool2d requires 4D input shape, got {len(self.shape)}D")
        
        is_max = isinstance(mod, nn.MaxPool2d)
        kind = LayerKind.MAXPOOL2D if is_max else LayerKind.AVGPOOL2D
        
        batch, in_c, in_h, in_w = self.shape
        ks = _normalize_tuple(mod.kernel_size)
        st = _normalize_tuple(mod.stride if mod.stride else mod.kernel_size)
        pad = _normalize_tuple(mod.padding, (0, 0))
        
        out_h = (in_h + 2 * pad[0] - ks[0]) // st[0] + 1
        out_w = (in_w + 2 * pad[1] - ks[1]) // st[1] + 1
        output_shape = (1, in_c, out_h, out_w)
        
        out_vars = self._alloc_ids(in_c * out_h * out_w)
        self._add_layer(
            kind.value, {},
            {"kernel_size": mod.kernel_size, "stride": mod.stride or mod.kernel_size,
             "padding": mod.padding, "input_shape": self.shape, "output_shape": output_shape},
            self.prev_out, out_vars
        )
        self.shape = output_shape
        self.prev_out = out_vars
    
    def _convert_adaptive_avgpool2d(self, mod: nn.AdaptiveAvgPool2d) -> None:
        """Convert nn.AdaptiveAvgPool2d."""
        if len(self.shape) != 4:
            raise ValueError(f"AdaptiveAvgPool2d requires 4D input, got {len(self.shape)}D")
        
        batch, in_c, in_h, in_w = self.shape
        out_size = mod.output_size
        out_h, out_w = (out_size, out_size) if isinstance(out_size, int) else out_size
        output_shape = (1, in_c, out_h, out_w)
        
        out_vars = self._alloc_ids(in_c * out_h * out_w)
        self._add_layer(LayerKind.ADAPTIVEAVGPOOL2D.value, {},
                       {"output_size": (out_h, out_w)},
                       self.prev_out, out_vars)
        self.shape = output_shape
        self.prev_out = out_vars
    
    def _convert_batchnorm(self, mod: _BatchNorm) -> None:
        """Convert BatchNorm to SCALE + BIAS layers."""
        gamma = mod.weight.detach() if mod.weight is not None else torch.ones(
            mod.num_features, dtype=mod.running_mean.dtype, device=mod.running_mean.device)
        beta = mod.bias.detach() if mod.bias is not None else torch.zeros(
            mod.num_features, dtype=mod.running_mean.dtype, device=mod.running_mean.device)
        
        scale = gamma / torch.sqrt(mod.running_var.detach() + mod.eps)
        bias = beta - scale * mod.running_mean.detach()
        
        n_channels = mod.num_features
        actual_size = len(self.prev_out)
        if actual_size % n_channels != 0:
            raise ValueError(f"BatchNorm: input size {actual_size} not divisible by {n_channels}")
        
        spatial = actual_size // n_channels
        scale_full = scale.repeat_interleave(spatial) if spatial > 1 else scale
        bias_full = bias.repeat_interleave(spatial) if spatial > 1 else bias
        
        # SCALE layer
        out_scale = self._same_size_forward()
        self._add_layer("SCALE", {"a": scale_full}, {}, self.prev_out, out_scale)
        self.prev_out = out_scale
        
        # BIAS layer
        out_bias = self._same_size_forward()
        self._add_layer("BIAS", {"c": bias_full}, {}, self.prev_out, out_bias)
        self.prev_out = out_bias
    
    def _convert_activation(self, mod: nn.Module, kind: LayerKind, 
                           extra_meta: Optional[Dict[str, Any]] = None) -> None:
        """Convert activation function."""
        out_vars = self._same_size_forward()
        meta = {"input_shape": self.shape, "output_shape": self.shape}
        if extra_meta:
            meta.update(extra_meta)
        self._add_layer(kind.value, {}, meta, self.prev_out, out_vars)
        self.prev_out = out_vars
    
    # -------------------------------------------------------------------------
    # FX Function Handlers
    # -------------------------------------------------------------------------
    
    def _process_add_operation(self, node: fx.Node) -> None:
        """Process ADD operation (skip connection merge)."""
        inputs = [a for a in node.args if isinstance(a, fx.Node)]
        if len(inputs) < 2:
            return
        
        x_name, y_name = inputs[0].name, inputs[1].name
        if x_name not in self.node_outputs or y_name not in self.node_outputs:
            return
        
        x_vars = self.node_outputs[x_name]
        y_vars = self.node_outputs[y_name]
        x_shape = self.node_shapes[x_name]
        
        out_vars = self._alloc_ids(len(x_vars))
        layer_id = self._add_layer(
            LayerKind.ADD.value, {},
            {"x_vars": x_vars, "y_vars": y_vars, "input_shape": x_shape, "output_shape": x_shape},
            x_vars + y_vars, out_vars
        )
        self.prev_out = out_vars
        self.shape = x_shape
        self._register_node(node.name, layer_id)
    
    def _process_concat_operation(self, node: fx.Node) -> None:
        """Process CONCAT operation."""
        if node.args and isinstance(node.args[0], (list, tuple)):
            inputs = [a for a in node.args[0] if isinstance(a, fx.Node)]
        else:
            inputs = [a for a in node.args if isinstance(a, fx.Node)]
        
        if not inputs:
            return
        
        all_vars = []
        total_size = 0
        for inp in inputs:
            if inp.name in self.node_outputs:
                vars_list = self.node_outputs[inp.name]
                all_vars.extend(vars_list)
                total_size += len(vars_list)
        
        if not all_vars:
            return
        
        out_vars = self._alloc_ids(total_size)
        dim = node.kwargs.get('dim', 1) if hasattr(node, 'kwargs') else 1
        
        layer_id = self._add_layer(
            LayerKind.CONCAT.value, {},
            {"concat_dim": dim,
             "input_shapes": [self.node_shapes.get(n.name) for n in inputs],
             "output_shape": (1, total_size)},
            all_vars, out_vars
        )
        self.prev_out = out_vars
        self.shape = (1, total_size)
        self._register_node(node.name, layer_id)
    
    def _process_flatten_function(self, node: fx.Node) -> None:
        """Process torch.flatten()."""
        if self._get_predecessor_state(node):
            self._create_flatten_layer(node.name)
    
    def _process_mul_operation(self, node: fx.Node) -> None:
        """Process MUL operation."""
        inputs = [a for a in node.args if isinstance(a, fx.Node)]
        
        if len(inputs) >= 2:
            x_name, y_name = inputs[0].name, inputs[1].name
            if x_name in self.node_outputs and y_name in self.node_outputs:
                x_vars = self.node_outputs[x_name]
                y_vars = self.node_outputs[y_name]
                x_shape = self.node_shapes[x_name]
                
                out_vars = self._alloc_ids(len(x_vars))
                layer_id = self._add_layer(
                    LayerKind.MUL.value, {},
                    {"input_shape": x_shape, "output_shape": x_shape},
                    x_vars + y_vars, out_vars
                )
                self.prev_out = out_vars
                self.shape = x_shape
                self._register_node(node.name, layer_id)
        
        elif len(inputs) == 1:
            x_name = inputs[0].name
            if x_name in self.node_outputs:
                x_vars = self.node_outputs[x_name]
                x_shape = self.node_shapes[x_name]
                scalar = node.args[1] if len(node.args) > 1 else 1.0
                if not isinstance(scalar, (int, float)):
                    scalar = 1.0
                
                scale_tensor = torch.full((len(x_vars),), float(scalar), dtype=self.dtype)
                out_vars = self._alloc_ids(len(x_vars))
                layer_id = self._add_layer(
                    "SCALE", {"a": scale_tensor},
                    {"input_shape": x_shape, "output_shape": x_shape},
                    x_vars, out_vars
                )
                self.prev_out = out_vars
                self.shape = x_shape
                self._register_node(node.name, layer_id)
    
    def _process_mean_operation(self, node: fx.Node) -> None:
        """Process torch.mean()."""
        if not self._get_predecessor_state(node):
            return
        
        out_vars = self._alloc_ids(1)
        output_shape = (1, 1)
        layer_id = self._add_layer(
            "MEAN", {},
            {"input_shape": self.shape, "output_shape": output_shape},
            self.prev_out, out_vars
        )
        self.prev_out = out_vars
        self.shape = output_shape
        self._register_node(node.name, layer_id)
    
    def _process_getitem_operation(self, node: fx.Node) -> None:
        """Process indexing operation."""
        inputs = [a for a in node.args if isinstance(a, fx.Node)]
        if inputs and inputs[0].name in self.node_outputs:
            pred_name = inputs[0].name
            self.node_outputs[node.name] = self.node_outputs[pred_name].copy()
            self.node_shapes[node.name] = self.node_shapes[pred_name]
            self.node_to_layer_id[node.name] = self.node_to_layer_id.get(pred_name, len(self.layers) - 1)
            self.prev_out = self.node_outputs[node.name]
            self.shape = self.node_shapes[node.name]
    
    def _process_passthrough_function(self, node: fx.Node) -> None:
        """Process no-op functions (dropout, stochastic_depth)."""
        inputs = [a for a in node.args if isinstance(a, fx.Node)]
        if inputs and inputs[0].name in self.node_outputs:
            pred_name = inputs[0].name
            self.node_outputs[node.name] = self.node_outputs[pred_name].copy()
            self.node_shapes[node.name] = self.node_shapes[pred_name]
            self.node_to_layer_id[node.name] = self.node_to_layer_id.get(pred_name, len(self.layers) - 1)
            self.prev_out = self.node_outputs[node.name]
            self.shape = self.node_shapes[node.name]


# -----------------------------------------------------------------------------
# Public tracing function (API - do not change signature)
# -----------------------------------------------------------------------------

def trace_model(model: nn.Module, input_shape: Tuple[int, ...], 
                dtype: torch.dtype = torch.float64) -> Tuple[List[Layer], Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Trace a PyTorch model and return ACT layers with graph structure.
    
    Args:
        model: Any nn.Module to trace
        input_shape: Input shape including batch dimension (e.g., (1, 3, 32, 32))
        dtype: Data type for tensors
        
    Returns:
        Tuple of (layers, preds, succs) forming a DAG
    """
    builder = _LayerGraphBuilder(model, input_shape, dtype)
    return builder.build_layer_graph()


# -----------------------------------------------------------------------------
# TorchToACT Converter
# -----------------------------------------------------------------------------

class TorchToACT:
    """
    Convert a wrapped nn.Module to ACT Net.
    
    Requirements:
      - Exactly one InputLayer (defines input shape)
      - At least one InputSpecLayer
      - Ends with OutputSpecLayer (ASSERT)
    """
    _WRAPPER_TYPES = ("InputLayer", "InputSpecLayer", "OutputSpecLayer")
    
    def __init__(self, wrapped: nn.Module):
        if not isinstance(wrapped, nn.Module):
            raise TypeError("TorchToACT expects an nn.Module.")
        
        self.m = wrapped
        mods = list(self.m)
        
        # Validate wrapper structure
        self._validate_wrapper(mods)
        
        # Extract InputLayer
        input_layers = [x for x in mods if type(x).__name__ == "InputLayer"]
        self.input_layer = input_layers[0]
        
        shape = getattr(self.input_layer, "shape", None)
        if shape is None:
            raise AssertionError("InputLayer must expose a 'shape' attribute.")
        
        # State
        self.layers: List[Layer] = []
        self.prev_out: List[int] = []
        self.shape: Tuple[int, ...] = tuple(int(s) for s in shape)
        self._model_preds: Dict[int, List[int]] = {}
        self._model_succs: Dict[int, List[int]] = {}
        self._wrapper_offset: int = 0
    
    def _validate_wrapper(self, mods: List[nn.Module]) -> None:
        """Validate wrapper layer structure."""
        has_input_spec = any(type(x).__name__ == "InputSpecLayer" for x in mods)
        has_output_spec = any(type(x).__name__ == "OutputSpecLayer" for x in mods)
        
        if not has_input_spec:
            raise AssertionError("Wrapper must include InputSpecLayer.")
        if not has_output_spec:
            raise AssertionError("Wrapper must include OutputSpecLayer.")
        if type(mods[-1]).__name__ != "OutputSpecLayer":
            raise AssertionError("Wrapper should end with OutputSpecLayer.")
    
    def run(self) -> Net:
        """Convert wrapped PyTorch model to ACT Net."""
        # Emit INPUT layer
        new_layers, out_vars = self.input_layer.to_act_layers(len(self.layers), [])
        self.layers.extend(new_layers)
        self.prev_out = out_vars
        
        # Process InputSpecLayers
        for mod in self.m:
            if type(mod).__name__ == "InputSpecLayer" and hasattr(mod, 'to_act_layers'):
                new_layers, out_vars = mod.to_act_layers(len(self.layers), self.prev_out)
                self.layers.extend(new_layers)
                self.prev_out = out_vars
        
        # Trace inner model
        self._trace_inner_model()
        
        # Process OutputSpecLayers
        for mod in self.m:
            if type(mod).__name__ == "OutputSpecLayer" and hasattr(mod, 'to_act_layers'):
                new_layers, out_vars = mod.to_act_layers(len(self.layers), self.prev_out)
                self.layers.extend(new_layers)
                self.prev_out = out_vars
        
        # Build and validate network
        preds, succs = self._build_layer_graph()
        net = Net(layers=self.layers, preds=preds, succs=succs)
        
        from act.back_end.layer_util import validate_graph
        validate_graph(self.layers)
        net.assert_last_is_validation()
        
        return net
    
    def _trace_inner_model(self) -> None:
        """Find and trace the inner model."""
        inner = self._find_inner_model()
        if inner is None:
            self._model_preds = {}
            self._model_succs = {}
            self._wrapper_offset = len(self.layers)
            return
        
        dtype = getattr(self.input_layer, 'dtype', torch.float64)
        model_layers, model_preds, model_succs = trace_model(inner, self.shape, dtype)
        
        # Offset layer IDs
        offset = len(self.layers)
        for layer in model_layers:
            layer.id += offset
        
        self.layers.extend(model_layers)
        if model_layers:
            self.prev_out = model_layers[-1].out_vars
        
        self._model_preds = {k + offset: [v + offset for v in vals] for k, vals in model_preds.items()}
        self._model_succs = {k + offset: [v + offset for v in vals] for k, vals in model_succs.items()}
        self._wrapper_offset = offset
    
    def _find_inner_model(self) -> Optional[nn.Module]:
        """Find actual model inside wrapper (skip wrapper layers)."""
        for mod in self.m:
            if type(mod).__name__ not in self._WRAPPER_TYPES:
                return mod
        return None
    
    def _build_layer_graph(self) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """Build layer graph combining wrapper and model layers."""
        n = len(self.layers)
        preds: Dict[int, List[int]] = {i: [] for i in range(n)}
        succs: Dict[int, List[int]] = {i: [] for i in range(n)}
        
        # Copy model graph
        for lid, ps in self._model_preds.items():
            if lid < n:
                preds[lid] = ps
        for lid, ss in self._model_succs.items():
            if lid < n:
                succs[lid] = ss
        
        # Connect wrapper layers (before model)
        for i in range(1, self._wrapper_offset):
            if not preds[i]:
                preds[i] = [i - 1]
                if i not in succs[i - 1]:
                    succs[i - 1].append(i)
        
        # Connect wrapper to first model layer
        if self._wrapper_offset > 0 and self._wrapper_offset < n:
            first_model = self._wrapper_offset
            last_wrapper = self._wrapper_offset - 1
            if not preds[first_model]:
                preds[first_model] = [last_wrapper]
            elif last_wrapper not in preds[first_model]:
                preds[first_model].insert(0, last_wrapper)
            if first_model not in succs[last_wrapper]:
                succs[last_wrapper].append(first_model)
        
        # Connect last model layer to ASSERT
        assert_id = n - 1
        if self._model_succs:
            last_model = max(self._model_succs.keys())
            if assert_id not in succs.get(last_model, []):
                succs[last_model].append(assert_id)
            if last_model not in preds[assert_id]:
                preds[assert_id].append(last_model)
        elif self._wrapper_offset > 0:
            last_wrapper = self._wrapper_offset - 1
            if last_wrapper not in preds[assert_id]:
                preds[assert_id].append(last_wrapper)
            if assert_id not in succs[last_wrapper]:
                succs[last_wrapper].append(assert_id)
        
        return preds, succs


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def _setup_debug_logging() -> None:
    """Initialize debug logging if enabled."""
    if PerformanceOptions.debug_tf:
        with open(PerformanceOptions.debug_output_file, 'w') as f:
            f.write(f"ACT Torch2ACT Conversion Debug Log\n{'='*80}\n\n")
        print(f"Debug logging to: {PerformanceOptions.debug_output_file}")


def _synthesize_models() -> Dict[str, nn.Module]:
    """Synthesize and test wrapped models."""
    print("\n[Step 1] Synthesizing wrapped models...")
    wrapped = model_synthesis()
    print(f"  Generated {len(wrapped)} wrapped models")
    
    print("\n[Step 2] Testing model inference...")
    successful = model_inference(wrapped)
    print(f"  {len(successful)} models passed inference tests")
    
    if not successful:
        print("  ERROR: No successful models to verify!")
        exit(1)
    
    return successful


def _initialize_solvers() -> List[Tuple[str, Any]]:
    """Initialize available solvers."""
    print("\n[Step 3] Initializing solvers...")
    solvers = []
    
    try:
        gurobi = GurobiSolver()
        gurobi.begin("act_verification")
        solvers.append(("Gurobi", gurobi))
        print("  Gurobi solver available")
    except Exception as e:
        print(f"  Gurobi initialization failed: {e}")
    
    try:
        torch_solver = TorchLPSolver()
        torch_solver.begin("act_verification")
        solvers.append(("TorchLP", torch_solver))
        print(f"  TorchLP solver available (device: {torch_solver._device})")
    except Exception as e:
        print(f"  TorchLP initialization failed: {e}")
    
    if not solvers:
        print("  No solvers available - conversion only")
    
    return solvers


def _process_models(models: Dict[str, nn.Module], solvers: List[Tuple[str, Any]]) -> Dict[str, Dict]:
    """Process all models: convert and optionally verify."""
    import gc
    
    print(f"\n[Step 4] Processing {len(models)} models...")
    results = {"conversion": {}, "verification": {}}
    
    for idx, (model_id, model) in enumerate(models.items(), 1):
        print(f"\n  [{idx}/{len(models)}] {model_id}")
        
        # Convert
        try:
            net = model.to_act_net()
            layer_types = " -> ".join(layer.kind for layer in net.layers)
            print(f"    Conversion OK: {len(net.layers)} layers ({layer_types})")
            results["conversion"][model_id] = "SUCCESS"
        except Exception as e:
            results["conversion"][model_id] = f"FAILED: {str(e)[:100]}"
            print(f"    Conversion FAILED: {e}")
            continue
        
        # Verify (if solvers available)
        if solvers:
            for solver_name, solver in solvers:
                # Verification currently skipped for testing
                print(f"    Verification ({solver_name}): SKIPPED")
        
        del net
        if idx % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return results


def _print_summary(results: Dict[str, Dict], total: int) -> None:
    """Print final summary."""
    successes = sum(1 for v in results["conversion"].values() if v == "SUCCESS")
    print(f"\n[Summary]")
    print(f"  Conversions: {successes}/{total} ({100*successes/total:.1f}%)")
    
    failures = {k: v for k, v in results["conversion"].items() if v != "SUCCESS"}
    if failures:
        print(f"  Failed conversions: {len(failures)}")
        for model_id, error in list(failures.items())[:5]:
            print(f"    - {model_id}: {error}")


def main():
    """Main entry point for PyTorch to ACT conversion and verification."""
    _setup_debug_logging()
    print("Starting Torch to ACT Verification Demo")
    
    models = _synthesize_models()
    solvers = _initialize_solvers()
    results = _process_models(models, solvers)
    _print_summary(results, len(models))
    
    if PerformanceOptions.debug_tf:
        print(f"\nDebug log: {PerformanceOptions.debug_output_file}")
    
    print("\nTorch to ACT conversion completed!")


if __name__ == "__main__":
    main()
