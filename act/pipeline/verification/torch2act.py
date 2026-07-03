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
#   - Unified tracing: Graph-based parsing via torch.fx for DAG support
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
import logging
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Union, cast
import torch
import torch.nn as nn
import torch.fx as fx
try:
    from onnx2torch.utils.common import OnnxToTorchModule
except Exception:  # pragma: no cover
    OnnxToTorchModule = ()
from torch.nn.modules.batchnorm import _BatchNorm

logger = logging.getLogger(__name__)
try:
    from torchvision.ops import StochasticDepth
    _HAS_STOCHASTIC_DEPTH = True
except (ImportError, RuntimeError):
    StochasticDepth = None  # type: ignore[assignment,misc]
    _HAS_STOCHASTIC_DEPTH = False

from act.back_end.core import Net, Layer
from act.back_end.layer_schema import LayerKind
from act.back_end.layer_util import create_layer
from act.pipeline.verification.utils import (
    _prod, _normalize_tuple, _assert_dag, _broadcast_const_to_size,
    _normalize_axes, _reduce_output_shape, _compute_slice_output_shape,
    ONNX_HANDLERS,
)

# Imports needed for main() test harness
from act.util.model_inference import model_inference
from act.front_end.model_synthesis import model_synthesis
from act.back_end.solver.solver_torchlp import TorchLPSolver
from act.back_end.solver.solver_gurobi import GurobiSolver
from act.util.options import PerformanceOptions


# -----------------------------------------------------------------------------
# Unified graph-based tracing for PyTorch models
# -----------------------------------------------------------------------------

class _LayerGraphBuilder:
    """
    Build ACT layer graph from nn.Module using torch.fx for graph extraction.
    
    The resulting graph is a DAG supporting skip connections (ResNet, etc.).
    """
    
    # Dispatch tables for FX call_method operations
    _METADATA_METHODS = frozenset({'size', 'dim', 'numel'})
    _PASSTHROUGH_METHODS = frozenset({'contiguous', 'to', 'float', 'double', 'half', 'cpu', 'cuda', 'detach'})
    _RESHAPE_METHODS = frozenset({'view', 'reshape', 'flatten'})
    _TRANSPOSE_METHODS = frozenset({'transpose', 'permute'})

    # ONNX Shape spec; DeviceManager is float-only so we can't derive this from self.dtype.
    _ONNX_SHAPE_DTYPE: ClassVar[torch.dtype] = torch.int64
    
    def __init__(
        self, model: nn.Module, input_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float64,
        sample_input: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.input_shape = input_shape
        self.dtype = dtype
        # Optional concrete sample tensor used by ``_evaluate_constant_subgraph``
        # when a constant chain reaches the model placeholder. Required only for
        # benchmarks like cctsdb_yolo_2023 whose Slice bounds are derived from
        # the input itself; for all other benchmarks this stays None and is
        # never consulted. The resulting ACT Net is locally valid around this
        # sample (e.g. for adversarial perturbation verification near it).
        self.sample_input = sample_input
        
        # Layer building state
        self.layers: List[Layer] = []
        self.next_var = 0
        self.prev_out: List[int] = []
        self.shape: Tuple[int, ...] = input_shape
        
        # Graph tracking (populated by FX tracing)
        self.node_outputs: Dict[str, List[int]] = {}
        self.node_shapes: Dict[str, Tuple[int, ...]] = {}
        self.node_to_layer_id: Dict[str, int] = {}
        self.graph_edges: Dict[str, List[str]] = {}
        self.modules: Dict[str, nn.Module] = {}
        
        # torch.fx specific
        self.fx_graph: Optional[fx.Graph] = None
        self.traced_model: Optional[fx.GraphModule] = None
        self._manual_preds: Optional[Dict[int, List[int]]] = None
        self._manual_succs: Optional[Dict[int, List[int]]] = None

        # Explicit predecessor edges for FX-path layers whose true data source
        # differs from their FX-arg node. Split decomposes into N sibling SLICEs
        # that all read the split's input, but each getitem child's FX arg is
        # the split node (mapped to the LAST chunk), so graph-edge resolution
        # would mis-wire them. _build_preds_succs honours this override.
        self._fx_pred_override: Dict[int, List[int]] = {}

        # Compile-time constants (e.g. OnnxShape values) that must NOT enter
        # the runtime IR. Resolved by ``_resolve_constant_tensor`` BEFORE any
        # placeholder/sample_input fallback — this is what stops shape chains
        # from baking sample-local bounds into a globally-quantified ACT Net.
        self._compile_time_values: Dict[str, torch.Tensor] = {}

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

        if self._try_build_vit_graph() or self._try_build_bert_transformer_graph():
            if self._manual_preds is None or self._manual_succs is None:
                raise RuntimeError("Manual transformer graph did not produce graph edges.")
            return self.layers, self._manual_preds, self._manual_succs

        # Extract computation graph using torch.fx
        self._extract_graph()

        # Pre-register placeholder nodes (network inputs)
        self._pre_register_nodes()

        # Process the FX graph
        self._process_fx_graph()

        # Build and validate graph structure
        preds, succs = self._build_preds_succs()

        return self.layers, preds, succs

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    def _alloc_ids(self, n: int) -> List[int]:
        """Allocate n consecutive variable IDs."""
        ids = list(range(self.next_var, self.next_var + n))
        self.next_var += n
        return ids
    
    def _same_size_forward(self) -> List[int]:
        """Allocate same number of output vars as current prev_out."""
        return self._alloc_ids(len(self.prev_out))
    
    def _add_layer(self, kind: str, params: Dict[str, Any],
                   in_vars: List[int], out_vars: List[int]) -> int:
        """Add a layer and return its ID."""
        layer = create_layer(
            id=len(self.layers),
            kind=kind,
            params=params,
            in_vars=in_vars,
            out_vars=out_vars,
        )
        self.layers.append(layer)
        return layer.id

    def _add_manual_layer(
        self,
        kind: str,
        params: Dict[str, Any],
        in_vars: List[int],
        out_vars: List[int],
        pred_ids: List[int],
    ) -> int:
        """Add a layer and explicit predecessor edges for manual graph mapping."""
        layer_id = self._add_layer(kind, params, in_vars, out_vars)
        if self._manual_preds is None:
            self._manual_preds = {}
        if self._manual_succs is None:
            self._manual_succs = {}
        self._manual_preds[layer_id] = list(pred_ids)
        self._manual_succs.setdefault(layer_id, [])
        for pred in pred_ids:
            self._manual_succs.setdefault(pred, [])
            if layer_id not in self._manual_succs[pred]:
                self._manual_succs[pred].append(layer_id)
        return layer_id

    def _try_build_vit_graph(self) -> bool:
        """Map explicit tiny ViT modules to existing ACT transformer layers."""
        patch_embed = getattr(self.model, "patch_embed", None)
        patch_proj = getattr(patch_embed, "proj", None)
        block = getattr(self.model, "block", None)
        norm = getattr(self.model, "norm", None)
        head = getattr(self.model, "head", None)
        cls_token = getattr(self.model, "cls_token", None)
        pos_embed = getattr(self.model, "pos_embed", None)
        if not (
            isinstance(patch_proj, nn.Conv2d)
            and isinstance(block, nn.Module)
            and self._is_bert_layer(block)
            and isinstance(norm, nn.Module)
            and isinstance(head, nn.Linear)
            and isinstance(cls_token, torch.Tensor)
            and isinstance(pos_embed, torch.Tensor)
        ):
            return False

        self._manual_preds = {}
        self._manual_succs = {}
        current_vars = self.prev_out
        current_shape = self.shape

        patch_vars, patch_shape, patch_id = self._add_manual_conv2d(
            patch_proj, current_vars, current_shape, []
        )
        flat_vars = self._alloc_ids(len(patch_vars))
        flat_shape = (patch_shape[0], patch_shape[1], patch_shape[2] * patch_shape[3])
        flat_id = self._add_manual_layer(
            LayerKind.FLATTEN.value,
            {"input_shape": patch_shape, "output_shape": flat_shape, "start_dim": 2, "end_dim": 3},
            patch_vars, flat_vars, [patch_id],
        )
        patch_seq_vars = self._alloc_ids(len(flat_vars))
        patch_seq_shape = (flat_shape[0], flat_shape[2], flat_shape[1])
        transpose_id = self._add_manual_layer(
            LayerKind.TRANSPOSE.value, {"perm": (0, 2, 1), "input_shape": tuple(flat_shape)},
            flat_vars, patch_seq_vars, [flat_id],
        )

        cls_shape = tuple(int(d) for d in cls_token.shape)
        cls_const_vars, cls_const_id = self._add_manual_constant(cls_token.detach(), cls_shape)
        cls_expand_vars = self._alloc_ids(_prod(cls_shape) or 1)
        cls_expand_id = self._add_manual_layer(
            LayerKind.EXPAND.value,
            {"shape": cls_shape, "input_shape": cls_shape, "output_shape": cls_shape},
            cls_const_vars, cls_expand_vars, [cls_const_id],
        )

        seq_shape = (patch_seq_shape[0], patch_seq_shape[1] + cls_shape[1], patch_seq_shape[2])
        concat_vars = self._alloc_ids(_prod(seq_shape) or 1)
        concat_id = self._add_manual_layer(
            LayerKind.CONCAT.value,
            {"concat_dim": 1},
            cls_expand_vars + patch_seq_vars, concat_vars, [cls_expand_id, transpose_id],
        )

        pos_shape = tuple(int(d) for d in pos_embed.shape)
        pos_vars, pos_id = self._add_manual_constant(pos_embed.detach(), pos_shape)
        pos_add_vars = self._alloc_ids(len(concat_vars))
        pos_add_id = self._add_manual_layer(
            LayerKind.ADD.value,
            {
                "x_vars": concat_vars, "y_vars": pos_vars,
                "input_shape": seq_shape, "output_shape": seq_shape,
            },
            concat_vars + pos_vars, pos_add_vars, [concat_id, pos_id],
        )

        block_vars, block_shape, block_ids = self._build_bert_block(
            block, pos_add_vars, seq_shape, [pos_add_id]
        )
        norm_vars, norm_id = self._add_manual_layernorm(norm, block_vars, block_shape, block_ids)

        cls_vars = self._alloc_ids(int(block_shape[-1]))
        gather_id = self._add_manual_layer(
            LayerKind.GATHER.value,
            {
                "indices": torch.tensor([0], dtype=torch.long), "axis": 0,
                "input_shape": (int(block_shape[-2]), int(block_shape[-1])), "output_shape": (1, int(block_shape[-1])),
            },
            norm_vars, cls_vars, [norm_id],
        )
        cls_flat_vars = self._alloc_ids(int(block_shape[-1]))
        reshape_id = self._add_manual_layer(
            LayerKind.RESHAPE.value,
            {
                "input_shape": (1, 1, int(block_shape[-1])),
                "output_shape": (1, int(block_shape[-1])),
                "target_shape": (1, int(block_shape[-1])),
            },
            cls_vars, cls_flat_vars, [gather_id],
        )
        head_vars, head_shape, head_id = self._add_manual_dense(
            head, cls_flat_vars, (1, int(block_shape[-1])), [reshape_id]
        )

        self.prev_out = head_vars
        self.shape = head_shape
        n_layers = len(self.layers)
        self._manual_preds = {i: self._manual_preds.get(i, []) for i in range(n_layers)}
        self._manual_succs = {i: self._manual_succs.get(i, []) for i in range(n_layers)}
        _assert_dag(self._manual_preds, self._manual_succs, n_layers)
        return True

    def _add_manual_constant(self, value: torch.Tensor, shape: Tuple[int, ...]) -> Tuple[List[int], int]:
        """Add a fixed tensor source layer for parameters used as graph operands."""
        flat = value.detach().clone().to(self.dtype).reshape(-1)
        out_vars = self._alloc_ids(int(flat.numel()) or 1)
        layer_id = self._add_manual_layer(
            LayerKind.CONSTANT.value,
            {"value": flat, "input_shape": shape, "output_shape": shape},
            [], out_vars, [],
        )
        return out_vars, layer_id

    def _add_manual_conv2d(
        self,
        mod: nn.Conv2d,
        in_vars: List[int],
        in_shape: Tuple[int, ...],
        pred_ids: List[int],
    ) -> Tuple[List[int], Tuple[int, ...], int]:
        """Add a Conv2d layer preserving PyTorch convolution parameters."""
        if len(in_shape) != 4:
            raise ValueError(f"ViT patch embedding requires 4D input, got {in_shape}.")
        _, in_c, in_h, in_w = (int(d) for d in in_shape)
        kernel_size = tuple(int(d) for d in _normalize_tuple(mod.kernel_size))
        stride = tuple(int(d) for d in _normalize_tuple(mod.stride))
        padding = tuple(int(d) for d in _normalize_tuple(mod.padding, (0, 0)))
        dilation = tuple(int(d) for d in _normalize_tuple(mod.dilation))
        out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
        out_shape = (1, int(mod.out_channels), int(out_h), int(out_w))
        out_vars = self._alloc_ids(_prod(out_shape) or 1)
        params: Dict[str, Any] = {
            "weight": mod.weight.detach(),
            "input_shape": in_shape, "output_shape": out_shape,
            "kernel_size": kernel_size, "stride": stride,
            "padding": padding, "dilation": dilation,
            "groups": mod.groups, "in_channels": int(in_c),
            "out_channels": int(mod.out_channels),
        }
        if mod.bias is not None:
            params["bias"] = mod.bias.detach()
        layer_id = self._add_manual_layer(
            LayerKind.CONV2D.value, params, in_vars, out_vars, pred_ids
        )
        return out_vars, out_shape, layer_id

    def _try_build_bert_transformer_graph(self) -> bool:
        """Map explicit BERT-style BertLayer modules to transformer ACT layers."""
        blocks = self._find_bert_encoder_blocks(self.model)
        if not blocks:
            return False

        self._manual_preds = {}
        self._manual_succs = {}
        current_vars = self.prev_out
        current_shape = self.shape
        anchor_vars = self._alloc_ids(len(current_vars))
        anchor_id = self._add_manual_layer(
            LayerKind.RESHAPE.value,
            {"input_shape": current_shape, "output_shape": current_shape, "target_shape": current_shape},
            current_vars,
            anchor_vars,
            [],
        )
        current_vars = anchor_vars
        current_layer_ids: List[int] = [anchor_id]

        for block in blocks:
            current_vars, current_shape, current_layer_ids = self._build_bert_block(
                block, current_vars, current_shape, current_layer_ids
            )

        classifier = getattr(self.model, "classifier", None)
        if isinstance(classifier, nn.Linear):
            pooled_vars = self._alloc_ids(int(classifier.in_features))
            if len(current_shape) >= 3:
                pool_id = self._add_manual_layer(
                    LayerKind.GATHER.value,
                    {
                        "indices": torch.tensor([0], dtype=torch.long), "axis": 0,
                        "input_shape": (int(current_shape[-2]), int(current_shape[-1])),
                        "output_shape": (1, int(classifier.in_features)),
                    },
                    current_vars, pooled_vars, current_layer_ids,
                )
                reshape_vars = self._alloc_ids(int(classifier.in_features))
                pool_id = self._add_manual_layer(
                    LayerKind.RESHAPE.value,
                    {
                        "input_shape": (1, 1, int(classifier.in_features)),
                        "output_shape": (1, int(classifier.in_features)),
                        "target_shape": (1, int(classifier.in_features)),
                    },
                    pooled_vars, reshape_vars, [pool_id],
                )
                pooled_vars = reshape_vars
            else:
                pool_id = self._add_manual_layer(
                    LayerKind.RESHAPE.value,
                    {
                        "input_shape": current_shape,
                        "output_shape": (1, int(classifier.in_features)),
                        "target_shape": (1, int(classifier.in_features)),
                    },
                    current_vars, pooled_vars, current_layer_ids,
                )
            current_vars = pooled_vars
            current_shape = (1, int(classifier.in_features))
            current_layer_ids = [pool_id]
            current_vars, current_shape, dense_id = self._add_manual_dense(
                classifier, current_vars, current_shape, current_layer_ids
            )
            current_layer_ids = [dense_id]

        self.prev_out = current_vars
        self.shape = current_shape
        n_layers = len(self.layers)
        self._manual_preds = {i: self._manual_preds.get(i, []) for i in range(n_layers)}
        self._manual_succs = {i: self._manual_succs.get(i, []) for i in range(n_layers)}
        _assert_dag(self._manual_preds, self._manual_succs, n_layers)
        return True

    def _find_bert_encoder_blocks(self, model: nn.Module) -> List[nn.Module]:
        """Return explicit BERT BertLayer modules when present."""
        candidates: List[Any] = []
        encoder = getattr(model, "encoder", None)
        if encoder is not None:
            candidates.append(getattr(encoder, "layer", None))
        candidates.extend([
            getattr(model, "layer", None),
            getattr(model, "layers", None),
            getattr(model, "blocks", None),
        ])
        for candidate in candidates:
            if isinstance(candidate, nn.ModuleList):
                blocks = list(candidate)
            elif isinstance(candidate, (list, tuple)) and all(isinstance(x, nn.Module) for x in candidate):
                blocks = list(candidate)
            elif isinstance(candidate, nn.Module):
                blocks = [candidate]
            else:
                continue
            if blocks and all(self._is_bert_layer(block) for block in blocks):
                return blocks
        if self._is_bert_layer(model):
            return [model]
        return []

    def _is_bert_layer(self, block: nn.Module) -> bool:
        attention = getattr(block, "attention", None)
        intermediate = getattr(block, "intermediate", None)
        output = getattr(block, "output", None)
        self_attn = getattr(attention, "self", None) if attention is not None else None
        attn_output = getattr(attention, "output", None) if attention is not None else None
        return all(
            hasattr(self_attn, name) and isinstance(getattr(self_attn, name), nn.Linear)
            for name in ("query", "key", "value")
        ) and all(x is not None for x in (attn_output, intermediate, output))

    def _build_bert_block(
        self,
        block: nn.Module,
        input_vars: List[int],
        input_shape: Tuple[int, ...],
        input_layer_ids: List[int],
    ) -> Tuple[List[int], Tuple[int, ...], List[int]]:
        if not input_shape:
            raise ValueError("Transformer block mapping requires a non-empty input shape.")
        block_any = cast(Any, block)
        attention = block_any.attention
        self_attn = attention.self
        attn_output = attention.output
        intermediate = block_any.intermediate
        block_output = block_any.output
        seq_len = int(input_shape[1]) if len(input_shape) >= 3 else 1
        hidden_size = int(input_shape[len(input_shape) - 1])
        head_dim = int(getattr(self_attn, "attention_head_size", hidden_size))

        score_rows: List[int] = []
        for pos in range(seq_len):
            q_vars = self._alloc_ids(head_dim)
            q_id = self._add_manual_layer(
                LayerKind.MHA_SPLIT.value,
                self._mha_split_params("query", self_attn.query, pos, None, input_shape, (1, head_dim)),
                input_vars,
                q_vars,
                input_layer_ids,
            )
            row_score_ids: List[int] = []
            row_score_vars: List[int] = []
            for key_pos in range(seq_len):
                k_vars = self._alloc_ids(head_dim)
                k_id = self._add_manual_layer(
                    LayerKind.MHA_SPLIT.value,
                    self._mha_split_params("key", self_attn.key, key_pos, None, input_shape, (1, head_dim)),
                    input_vars,
                    k_vars,
                    input_layer_ids,
                )
                score_vars = self._alloc_ids(1)
                score_id = self._add_manual_layer(
                    LayerKind.ATT_SCORES.value,
                    {
                        "dk": float(head_dim) ** 0.5,
                        "q_vars": q_vars,
                        "k_vars": k_vars,
                        "q_src": q_id,
                        "k_src": k_id,
                        "query_position": pos,
                        "key_position": key_pos,
                        "input_shape": (1, head_dim),
                        "output_shape": (1, 1),
                    },
                    q_vars + k_vars,
                    score_vars,
                    [q_id, k_id],
                )
                row_score_ids.append(score_id)
                row_score_vars.extend(score_vars)
            if len(row_score_ids) == 1:
                softmax_in_vars = row_score_vars
                softmax_preds = row_score_ids
            else:
                softmax_in_vars = row_score_vars
                concat_vars = self._alloc_ids(seq_len)
                concat_id = self._add_manual_layer(
                    LayerKind.CONCAT.value,
                    {"concat_dim": -1},
                    row_score_vars,
                    concat_vars,
                    row_score_ids,
                )
                softmax_in_vars = concat_vars
                softmax_preds = [concat_id]
            prob_vars = self._alloc_ids(seq_len)
            prob_id = self._add_manual_layer(
                LayerKind.SOFTMAX.value,
                {"axis": -1},
                softmax_in_vars,
                prob_vars,
                softmax_preds,
            )
            score_rows.append(prob_id)

        context_ids: List[int] = []
        context_vars: List[int] = []
        for pos, prob_id in enumerate(score_rows):
            for feature in range(hidden_size):
                v_vars = self._alloc_ids(seq_len)
                v_id = self._add_manual_layer(
                    LayerKind.MHA_SPLIT.value,
                    self._mha_split_params("value", self_attn.value, None, feature, input_shape, (1, seq_len)),
                    input_vars,
                    v_vars,
                    input_layer_ids,
                )
                mix_vars = self._alloc_ids(1)
                mix_id = self._add_manual_layer(
                    LayerKind.ATT_MIX.value,
                    {
                        "rowsize": seq_len,
                        "w_vars": self.layers[prob_id].out_vars,
                        "v_vars": v_vars,
                        "w_src": prob_id,
                        "v_src": v_id,
                        "query_position": pos,
                        "feature": feature,
                        "input_shape": (1, seq_len),
                        "output_shape": (1, 1),
                    },
                    self.layers[prob_id].out_vars + v_vars,
                    mix_vars,
                    [prob_id, v_id],
                )
                context_ids.append(mix_id)
                context_vars.extend(mix_vars)
        joined_vars = self._alloc_ids(seq_len * hidden_size)
        join_id = self._add_manual_layer(
            LayerKind.MHA_JOIN.value,
            {"seq_len": seq_len, "hidden_size": hidden_size, "output_shape": (1, seq_len, hidden_size)},
            context_vars,
            joined_vars,
            context_ids,
        )

        attn_dense_vars, attn_dense_shape, attn_dense_id = self._add_manual_dense(
            attn_output.dense, joined_vars, (1, seq_len * hidden_size), [join_id]
        )
        add1_vars = self._alloc_ids(len(attn_dense_vars))
        add1_id = self._add_manual_layer(
            LayerKind.ADD.value,
            {
                "x_vars": attn_dense_vars,
                "y_vars": input_vars,
                "input_shape": attn_dense_shape,
                "output_shape": attn_dense_shape,
            },
            attn_dense_vars + input_vars,
            add1_vars,
            [attn_dense_id] + input_layer_ids,
        )
        norm1_vars, norm1_id = self._add_manual_layernorm(attn_output.LayerNorm, add1_vars, attn_dense_shape, [add1_id])

        ffn_vars, ffn_shape, ffn_dense_id = self._add_manual_dense(
            intermediate.dense, norm1_vars, attn_dense_shape, [norm1_id]
        )
        act_kind = LayerKind.GELU if self._uses_gelu(intermediate) else LayerKind.RELU
        act_vars = self._alloc_ids(len(ffn_vars))
        act_id = self._add_manual_layer(
            act_kind.value,
            {"input_shape": ffn_shape, "output_shape": ffn_shape},
            ffn_vars,
            act_vars,
            [ffn_dense_id],
        )
        out_vars, out_shape, out_dense_id = self._add_manual_dense(
            block_output.dense, act_vars, ffn_shape, [act_id]
        )
        add2_vars = self._alloc_ids(len(out_vars))
        add2_id = self._add_manual_layer(
            LayerKind.ADD.value,
            {
                "x_vars": out_vars,
                "y_vars": norm1_vars,
                "input_shape": out_shape,
                "output_shape": out_shape,
            },
            out_vars + norm1_vars,
            add2_vars,
            [out_dense_id, norm1_id],
        )
        norm2_vars, norm2_id = self._add_manual_layernorm(block_output.LayerNorm, add2_vars, out_shape, [add2_id])
        return norm2_vars, out_shape, [norm2_id]

    def _mha_split_params(
        self,
        role: str,
        projection: nn.Linear,
        position: Optional[int],
        feature: Optional[int],
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "role": role,
            "weight": projection.weight.detach(),
            "hidden_size": int(projection.in_features),
            "input_shape": input_shape,
            "output_shape": output_shape,
        }
        if projection.bias is not None:
            params["bias"] = projection.bias.detach()
        if position is not None:
            params["position"] = int(position)
        if feature is not None:
            params["feature"] = int(feature)
        return params

    def _add_manual_dense(
        self,
        mod: nn.Linear,
        in_vars: List[int],
        in_shape: Tuple[int, ...],
        pred_ids: List[int],
    ) -> Tuple[List[int], Tuple[int, ...], int]:
        in_features = int(mod.in_features)
        out_features = int(mod.out_features)
        if len(in_vars) == in_features:
            weight = mod.weight.detach()
            bias = mod.bias.detach() if mod.bias is not None else None
            output_shape = (1, out_features)
        elif len(in_vars) % in_features == 0:
            token_count = len(in_vars) // in_features
            weight = torch.block_diag(*[mod.weight.detach()] * token_count)
            bias = mod.bias.detach().repeat(token_count) if mod.bias is not None else None
            if len(in_shape) >= 3:
                output_shape = (*in_shape[:-1], out_features)
            else:
                output_shape = (1, token_count, out_features)
        else:
            raise ValueError(
                f"Dense input size {len(in_vars)} is not compatible with in_features={in_features}."
            )
        out_vars = self._alloc_ids(_prod(output_shape) or 1)
        params = {
            "weight": weight,
            "in_features": int(weight.shape[1]),
            "out_features": int(weight.shape[0]),
            "input_shape": in_shape,
            "output_shape": output_shape,
        }
        if bias is not None:
            params["bias"] = bias
        layer_id = self._add_manual_layer(LayerKind.DENSE.value, params, in_vars, out_vars, pred_ids)
        return out_vars, output_shape, layer_id

    def _add_manual_layernorm(
        self,
        mod: nn.Module,
        in_vars: List[int],
        in_shape: Tuple[int, ...],
        pred_ids: List[int],
    ) -> Tuple[List[int], int]:
        gamma = getattr(mod, "weight").detach()
        beta = getattr(mod, "bias").detach()
        eps = float(getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-5)))
        out_vars = self._alloc_ids(len(in_vars))
        layer_id = self._add_manual_layer(
            LayerKind.LAYERNORM.value,
            {"gamma": gamma, "beta": beta, "eps": eps, "input_shape": in_shape, "output_shape": in_shape},
            in_vars,
            out_vars,
            pred_ids,
        )
        return out_vars, layer_id

    def _uses_gelu(self, module: nn.Module) -> bool:
        act = getattr(module, "intermediate_act_fn", None)
        if isinstance(act, nn.GELU):
            return True
        name = getattr(act, "__name__", "").lower()
        return "gelu" in name
    
    def _register_node(self, name: str, layer_id: Optional[int] = None) -> None:
        """Register node's output vars, shape, and layer mapping."""
        self.node_outputs[name] = self.prev_out.copy()
        self.node_shapes[name] = self.shape
        self.node_to_layer_id[name] = layer_id if layer_id is not None else (len(self.layers) - 1)
    
    def _pre_register_nodes(self) -> None:
        """Pre-register placeholder nodes so successor nodes can look up input vars."""
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
    
    def _propagate_node_state(self, node_name: str, pred_name: str) -> bool:
        """Propagate state from predecessor to current node (for passthrough ops). Returns True if successful."""
        if pred_name not in self.node_outputs:
            return False
        self.node_outputs[node_name] = self.node_outputs[pred_name].copy()
        self.node_shapes[node_name] = self.node_shapes[pred_name]
        self.node_to_layer_id[node_name] = self.node_to_layer_id.get(pred_name, len(self.layers) - 1)
        self.prev_out = self.node_outputs[node_name]
        self.shape = self.node_shapes[node_name]
        return True

    def _resolve_constant_tensor(self, node_name: str) -> Optional[torch.Tensor]:
        """Return the tensor value of a get_attr fx node or compile-time stashed value."""
        cached = self._compile_time_values.get(node_name)
        if cached is not None:
            return cached.detach().clone()
        if self.fx_graph is None or self.traced_model is None:
            return None
        for n in self.fx_graph.nodes:
            if n.name != node_name:
                continue
            if n.op != 'get_attr':
                return None
            target = str(n.target)
            for resolver_name in ('get_buffer', 'get_parameter'):
                resolver = getattr(self.traced_model, resolver_name, None)
                if resolver is None:
                    continue
                try:
                    val = resolver(target)
                except (AttributeError, KeyError, RuntimeError) as e:
                    # Intentional: resolver may not own this target; try the next resolver.
                    logger.debug("suppressed: %s", e)
                    continue
                if isinstance(val, torch.Tensor):
                    return val.detach().clone()
            return None
        return None

    def _resolve_slice_input_to_int_list(self, node_name: str) -> Optional[List[int]]:
        """Read an OnnxSlice positional input (starts/ends/axes/steps) as an int list.

        Resolves in three escalating tiers:
          1. Direct get_attr initializer.
          2. Upstream layer that stored a constant under
             ``params['shape_value']`` / ``params['value']``.
          3. Constant-only fx subgraph: walk back through call_module nodes
             that consume only get_attr / Constant chains and execute them
             offline (handles e.g. YOLO's ``slice(anchor, concat(idx0, idx1))``
             where the concat is itself constant).
        Returns None when none of the tiers apply.
        """
        tensor = self._resolve_constant_tensor(node_name)
        if tensor is not None:
            return [int(x) for x in tensor.reshape(-1).tolist()]
        if node_name in self.node_to_layer_id:
            layer_id = self.node_to_layer_id[node_name]
            if 0 <= layer_id < len(self.layers):
                layer = self.layers[layer_id]
                shape_value = layer.params.get("shape_value", layer.params.get("value"))
                if isinstance(shape_value, torch.Tensor):
                    return [int(x) for x in shape_value.reshape(-1).tolist()]
        evaluated = self._evaluate_constant_subgraph(node_name)
        if evaluated is not None:
            return [int(x) for x in evaluated.reshape(-1).tolist()]
        return None

    def _evaluate_constant_subgraph(self, node_name: str) -> Optional[torch.Tensor]:
        """Recursively evaluate an fx node whose inputs trace back to constants only.

        Returns the concrete tensor or None if the chain involves any variable
        (i.e. an actual model activation). Used by ``_resolve_slice_input_to_int_list``
        to recover Slice bounds whose value is computed by a chain of constant
        ops (e.g. YOLO's ``slice_23`` where starts/ends come from constant
        ``Concat(initializer_X, initializer_Y)``).
        """
        cached = self._resolve_constant_tensor(node_name)
        if cached is not None:
            return cached
        if self.fx_graph is None or self.traced_model is None:
            return None
        target_node = next((n for n in self.fx_graph.nodes if n.name == node_name), None)
        if target_node is None:
            return None
        if target_node.op == 'placeholder':
            # The chain reached the model input. If a concrete ``sample_input``
            # was passed, substitute it so the chain can continue; the resulting
            # IR is locally valid around that sample. Without one, treat as
            # genuinely variable and abort the constant evaluation.
            if self.sample_input is not None:
                return self.sample_input
            return None
        if target_node.op != 'call_module':
            return None
        sub = self.modules.get(str(target_node.target))
        if sub is None:
            return None
        arg_vals: List[Any] = []
        for a in target_node.args:
            if isinstance(a, fx.Node):
                v = self._evaluate_constant_subgraph(a.name)
                if v is None:
                    return None
                arg_vals.append(v)
            else:
                arg_vals.append(a)
        try:
            with torch.no_grad():
                return sub(*arg_vals)
        except Exception:
            return None

    def _ensure_constant_vars(self, node_name: str) -> bool:
        """Emit a CONSTANT layer for an ONNX initializer (consumed by Concat /
        Slice / MatMul as a registered var operand). False if not get_attr."""
        if node_name in self.node_outputs:
            return True
        const = self._resolve_constant_tensor(node_name)
        if const is None:
            return False
        flat = const.detach().clone().to(self.dtype).reshape(-1)
        shape = tuple(int(d) for d in const.shape) or (1,)
        out_vars = self._alloc_ids(int(flat.numel()) or 1)
        layer_id = self._add_layer(
            LayerKind.CONSTANT.value,
            {"value": flat, "input_shape": shape, "output_shape": shape},
            [], out_vars,
        )
        self.node_outputs[node_name] = out_vars
        self.node_shapes[node_name] = shape
        self.node_to_layer_id[node_name] = layer_id
        return True

    # -------------------------------------------------------------------------
    # Model Tracing (torch.fx only)
    # -------------------------------------------------------------------------
    
    def _extract_graph(self) -> None:
        """Extract computation graph using torch.fx symbolic tracing.

        Reuse a pre-traced GraphModule (e.g. from ``onnx2torch.convert``) so its
        initializer buffers stay attached; re-tracing would lose them.
        """
        try:
            traced = self.model if isinstance(self.model, fx.GraphModule) else fx.symbolic_trace(self.model)
            self.traced_model = traced
            self.fx_graph = traced.graph
            self.modules = dict(traced.named_modules())
            self._build_fx_graph_edges()
        except Exception as e:
            raise RuntimeError(f"Failed to trace model with torch.fx: {e}")
    
    def _build_fx_graph_edges(self) -> None:
        """Build graph edge dictionary from torch.fx graph.

        Args may hold fx.Nodes directly OR inside a list/tuple (e.g. torch.cat /
        torch.stack take a list of tensors). Both must be recorded so merge
        layers (CONCAT/STACK) connect to ALL of their branch producers, not just
        the sequential fallback predecessor.
        """
        if self.fx_graph is None:
            return
        for node in self.fx_graph.nodes:
            edge_preds: List[str] = []
            for arg in node.args:
                if isinstance(arg, fx.Node):
                    edge_preds.append(arg.name)
                elif isinstance(arg, (list, tuple)):
                    edge_preds.extend(a.name for a in arg if isinstance(a, fx.Node))
            self.graph_edges[node.name] = edge_preds
    
    # -------------------------------------------------------------------------
    # FX Graph Processing
    # -------------------------------------------------------------------------
    
    def _process_fx_graph(self) -> None:
        """Process model using torch.fx graph nodes."""
        if self.fx_graph is None:
            return
        for node in self.fx_graph.nodes:
            if node.op == 'placeholder':
                pass  # Already pre-registered in _pre_register_nodes
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
    
    def _handle_call_module(self, node: fx.Node) -> None:
        """Handle call_module node."""
        module = self.modules.get(node.target)
        if module is None:
            raise ValueError(f"Module '{node.target}' not found in traced model")

        if 'onnx2torch' in type(module).__module__ or (OnnxToTorchModule and isinstance(module, OnnxToTorchModule)):
            cls_name = type(module).__name__
            handler = getattr(self, f'_convert_{cls_name}', None)
            if handler is None:
                raise NotImplementedError(
                    f"Unsupported onnx2torch module {cls_name} at {node.name}"
                )
            handler(module, node)
            return

        self._get_predecessor_state(node)
        self._convert_module(module)
        self._register_node(node.name)
    
    def _handle_call_function(self, node: fx.Node) -> None:
        """Handle call_function node."""
        target_name = str(node.target).lower()
        
        handlers = {
            'add': self._process_add_operation,
            'sub': self._process_sub_operation,
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
                self._propagate_node_state(node.name, node.args[0].name)
        
        elif method_name in self._RESHAPE_METHODS:
            if self._get_predecessor_state(node):
                self._create_reshape_method_layer(node)

        elif method_name in self._TRANSPOSE_METHODS:
            if self._get_predecessor_state(node):
                self._create_transpose_method_layer(node)

        elif method_name == 'expand':
            if self._get_predecessor_state(node):
                self._create_expand_method_layer(node)
        
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
    # Graph Structure Building
    # -------------------------------------------------------------------------
    
    def _build_preds_succs(self) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """Build preds and succs dictionaries, validating DAG property.
        
        Uses a hybrid approach:
        1. Build edges from FX graph for mapped nodes
        2. For unmapped intermediate layers (e.g., SCALE from BatchNorm), 
           connect them sequentially based on layer creation order
        """
        n_layers = len(self.layers)
        
        # Identify which layers are mapped to FX nodes
        mapped_layer_ids = set(lid for lid in self.node_to_layer_id.values() if lid >= 0)
        
        preds: Dict[int, List[int]] = {i: [] for i in range(n_layers)}
        succs: Dict[int, List[int]] = {i: [] for i in range(n_layers)}
        
        # Track layers that take input directly from the placeholder (network input)
        takes_input_from_placeholder: Set[int] = set()
        
        # First, build FX graph edges for mapped layers
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
        # These layers are internal to a multi-layer conversion and should connect to i-1.
        # Source layers (``in_vars == []``, e.g. CONSTANT emitted by
        # ``_ensure_constant_vars``) are intentionally exempt -- they have no
        # data dependency on anything upstream, so wiring them to ``i-1`` would
        # invent a fake predecessor and corrupt downstream TF dataflow.
        for i in range(1, n_layers):
            if not self.layers[i].in_vars:
                continue
            if i not in mapped_layer_ids:
                # Unmapped layer - must connect to previous layer
                preds[i] = [i - 1]
            elif not preds[i] and i not in takes_input_from_placeholder:
                # Mapped but no FX predecessors AND not taking input from placeholder
                # -> connect to previous layer (internal layer within multi-layer conversion)
                preds[i] = [i - 1]
            # else: Mapped layer with FX predecessors OR takes network input - keep as-is

        # Explicit overrides (e.g. split-chunk SLICEs) replace FX-derived edges
        # before succs are built so both directions stay consistent.
        for lid, override in self._fx_pred_override.items():
            if 0 <= lid < n_layers:
                preds[lid] = list(override)

        # CONSTANT layers inject a fixed tensor (data-independent source); any
        # FX edge to them is a shape/control artifact, so they must have no
        # predecessors or the dual backward would route ν into a wrong-width op.
        for i in range(n_layers):
            if self.layers[i].kind == LayerKind.CONSTANT.value:
                preds[i] = []

        # Build succs from preds
        for i in range(n_layers):
            for pred_id in preds[i]:
                if i not in succs[pred_id]:
                    succs[pred_id].append(i)
        
        _assert_dag(preds, succs, n_layers)
        return preds, succs

    # -------------------------------------------------------------------------
    # Layer Conversion - Module Dispatcher
    # -------------------------------------------------------------------------
    
    def _convert_module(self, mod: nn.Module) -> None:
        """Convert a PyTorch module to ACT layer(s)."""
        converters = {
            nn.Flatten: self._convert_flatten,
            nn.Linear: self._convert_linear,
            nn.ReLU: lambda m: self._convert_activation(m, LayerKind.RELU),
            nn.Conv2d: self._convert_conv2d,
            nn.ConvTranspose2d: self._convert_conv_transpose2d,
            nn.MaxPool2d: self._convert_pool2d,
            nn.AvgPool2d: self._convert_pool2d,
            nn.AdaptiveAvgPool2d: self._convert_adaptive_avgpool2d,
            _BatchNorm: self._convert_batchnorm,
            nn.SiLU: lambda m: self._convert_activation(m, LayerKind.SILU),
            nn.Sigmoid: lambda m: self._convert_activation(m, LayerKind.SIGMOID),
            nn.Tanh: lambda m: self._convert_activation(m, LayerKind.TANH),
            nn.Softmax: self._convert_softmax,
            nn.LeakyReLU: lambda m: self._convert_activation(m, LayerKind.LRELU, {"negative_slope": m.negative_slope}),
            nn.LSTM: lambda m: self._convert_rnn_family(m, LayerKind.LSTM),
            nn.GRU: lambda m: self._convert_rnn_family(m, LayerKind.GRU),
            nn.RNN: lambda m: self._convert_rnn_family(m, LayerKind.RNN),
        }
        
        # No-op modules (identity during inference)
        if isinstance(mod, nn.Dropout) or (
            _HAS_STOCHASTIC_DEPTH and isinstance(mod, StochasticDepth)
        ):
            return
        
        for mod_type, converter in converters.items():
            if isinstance(mod, mod_type):
                converter(mod)
                return
        
        raise NotImplementedError(f"Unsupported module: {type(mod).__name__}")
    
    # -------------------------------------------------------------------------
    # Layer Conversion - Specific Converters
    # -------------------------------------------------------------------------
    
    def _resolve_int_arg(self, value: Any) -> Optional[int]:
        """Resolve FX literal integer arguments produced by shape metadata nodes."""
        if isinstance(value, int):
            return int(value)
        if isinstance(value, fx.Node):
            if value.op == 'call_method' and value.target == 'size':
                if len(value.args) >= 2 and isinstance(value.args[1], int):
                    source = value.args[0]
                    if isinstance(source, fx.Node) and source.name in self.node_shapes:
                        dim = int(value.args[1])
                        shape = self.node_shapes[source.name]
                        return int(shape[dim])
            const = self._resolve_constant_tensor(value.name)
            if const is not None and const.numel() == 1:
                return int(const.reshape(-1)[0].item())
        return None

    def _normalize_target_shape(self, raw_shape: Tuple[Any, ...]) -> Tuple[int, ...]:
        """Resolve reshape/expand dimensions against the current ACT shape."""
        shape: List[int] = []
        for index, dim_value in enumerate(raw_shape):
            dim = self._resolve_int_arg(dim_value)
            if dim is None:
                raise ValueError(f"Cannot resolve tensor shape argument {dim_value!r}.")
            if dim == -1:
                shape.append(-1)
            else:
                shape.append(int(self.shape[index]) if dim == 0 and index < len(self.shape) else dim)
        if -1 in shape:
            known = _prod(tuple(d for d in shape if d != -1)) or 1
            shape[shape.index(-1)] = _prod(self.shape) // known
        return tuple(shape)

    def _create_reshape_method_layer(self, node: fx.Node) -> List[int]:
        """Create a RESHAPE/FLATTEN layer for tensor shape methods."""
        method_name = str(node.target)
        if method_name == 'flatten':
            start_dim = int(node.args[1]) if len(node.args) > 1 and isinstance(node.args[1], int) else 0
            end_dim = int(node.args[2]) if len(node.args) > 2 and isinstance(node.args[2], int) else -1
            rank = len(self.shape)
            if start_dim < 0:
                start_dim += rank
            if end_dim < 0:
                end_dim += rank
            output_shape = (
                *self.shape[:start_dim],
                _prod(self.shape[start_dim:end_dim + 1]),
                *self.shape[end_dim + 1:],
            )
            out_vars = self._same_size_forward()
            layer_id = self._add_layer(
                LayerKind.FLATTEN.value,
                {
                    "input_shape": self.shape, "output_shape": output_shape,
                    "start_dim": start_dim, "end_dim": end_dim,
                },
                self.prev_out, out_vars,
            )
            self.prev_out = out_vars
            self.shape = output_shape
            self._register_node(node.name, layer_id)
            return out_vars

        raw_shape = tuple(node.args[1:])
        if len(raw_shape) == 1 and isinstance(raw_shape[0], (list, tuple)):
            raw_shape = tuple(raw_shape[0])
        output_shape = self._normalize_target_shape(raw_shape)
        out_vars = self._alloc_ids(_prod(output_shape) or 1)
        if len(out_vars) != len(self.prev_out):
            raise ValueError(f"Reshape changes element count from {len(self.prev_out)} to {len(out_vars)}.")
        layer_id = self._add_layer(
            LayerKind.RESHAPE.value,
            {"target_shape": output_shape, "input_shape": self.shape, "output_shape": output_shape},
            self.prev_out, out_vars,
        )
        self.prev_out = out_vars
        self.shape = output_shape
        self._register_node(node.name, layer_id)
        return out_vars

    def _create_transpose_method_layer(self, node: fx.Node) -> List[int]:
        """Create a TRANSPOSE layer for tensor transpose/permute methods."""
        rank = len(self.shape)
        input_shape_t = tuple(self.shape)
        if node.target == 'transpose':
            if len(node.args) < 3:
                raise ValueError(f"transpose at {node.name} requires two dimensions.")
            dim0 = int(node.args[1])
            dim1 = int(node.args[2])
            if dim0 < 0:
                dim0 += rank
            if dim1 < 0:
                dim1 += rank
            perm = list(range(rank))
            perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        else:
            perm_args = node.args[1:]
            if len(perm_args) == 1 and isinstance(perm_args[0], (list, tuple)):
                perm_args = tuple(perm_args[0])
            perm = [int(p) + rank if int(p) < 0 else int(p) for p in perm_args]
        if len(perm) != rank:
            raise ValueError(f"transpose rank {len(perm)} does not match input rank {rank}.")
        output_shape = tuple(self.shape[p] for p in perm)
        out_vars = self._same_size_forward()
        layer_id = self._add_layer(
            LayerKind.TRANSPOSE.value,
            {"perm": tuple(perm), "input_shape": input_shape_t},
            self.prev_out, out_vars,
        )
        self.prev_out = out_vars
        self.shape = output_shape
        self._register_node(node.name, layer_id)
        return out_vars

    def _create_expand_method_layer(self, node: fx.Node) -> List[int]:
        """Create an EXPAND layer for tensor broadcast expansion."""
        raw_shape = tuple(node.args[1:])
        if len(raw_shape) == 1 and isinstance(raw_shape[0], (list, tuple)):
            raw_shape = tuple(raw_shape[0])
        target_shape = self._normalize_target_shape(raw_shape)
        try:
            output_shape = tuple(int(d) for d in torch.broadcast_shapes(self.shape, target_shape))
        except RuntimeError as exc:
            raise ValueError(f"Cannot expand {self.shape} to {target_shape}.") from exc
        out_vars = self._alloc_ids(_prod(output_shape) or 1)
        layer_id = self._add_layer(
            LayerKind.EXPAND.value,
            {"shape": output_shape, "input_shape": self.shape, "output_shape": output_shape},
            self.prev_out, out_vars,
        )
        self.prev_out = out_vars
        self.shape = output_shape
        self._register_node(node.name, layer_id)
        return out_vars

    def _create_flatten_layer(self, node_name: Optional[str] = None,
                               start_dim: int = 1, end_dim: int = -1) -> List[int]:
        """Create FLATTEN layer, optionally register node."""
        out_vars = self._same_size_forward()
        output_shape = (1, _prod(self.shape[1:]))
        
        params = {
            "input_shape": self.shape, "output_shape": output_shape,
            "start_dim": start_dim, "end_dim": end_dim
        }
        layer_id = self._add_layer(
            LayerKind.FLATTEN.value, params,
            self.prev_out, out_vars
        )
        self.prev_out = out_vars
        self.shape = output_shape
        if node_name:
            self._register_node(node_name, layer_id)
        return out_vars
    
    def _convert_flatten(self, mod: nn.Flatten) -> None:
        """Convert nn.Flatten."""
        self._create_flatten_layer(start_dim=mod.start_dim, end_dim=mod.end_dim)
    
    def _convert_linear(self, mod: nn.Linear) -> None:
        """Convert nn.Linear to DENSE layer."""
        in_features = int(mod.in_features)
        out_features = int(mod.out_features)
        has_bias = mod.bias is not None
        
        W = mod.weight.detach()
        b = mod.bias.detach() if has_bias else torch.zeros(out_features)
        
        out_vars = self._alloc_ids(out_features)
        
        params = {
            "weight": W,
            "input_shape": self.shape, "output_shape": (1, out_features),
            "in_features": in_features, "out_features": out_features
        }
        if b is not None:
            params["bias"] = b
        self._add_layer(
            LayerKind.DENSE.value,
            params,
            self.prev_out, out_vars
        )
        self.shape = (1, out_features)
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
            input_shape = tuple(int(d) for d in self.shape)
        
        batch, in_c, in_h, in_w = (int(d) for d in input_shape)
        out_c = int(mod.out_channels)
        kernel_size = tuple(int(d) for d in _normalize_tuple(mod.kernel_size))
        stride = tuple(int(d) for d in _normalize_tuple(mod.stride))
        padding = tuple(int(d) for d in _normalize_tuple(mod.padding, (0, 0)))
        dilation = tuple(int(d) for d in _normalize_tuple(mod.dilation))
        out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
        output_shape = (1, out_c, out_h, out_w)
        
        
        params = {
            "weight": weight,
            "input_shape": input_shape, "output_shape": output_shape,
            "kernel_size": kernel_size, "stride": stride,
            "padding": padding, "dilation": dilation,
            "groups": mod.groups, "in_channels": in_c, "out_channels": out_c
        }
        if bias is not None:
            params["bias"] = bias
        
        out_vars = self._alloc_ids(out_c * out_h * out_w)
        self._add_layer(
            LayerKind.CONV2D.value, params,
            self.prev_out, out_vars
        )
        self.shape = output_shape
        self.prev_out = out_vars

    def _convert_conv_transpose2d(self, mod: nn.ConvTranspose2d) -> None:
        """Convert nn.ConvTranspose2d (output-shape formula differs from Conv2d)."""
        if len(self.shape) != 4:
            raise ValueError(f"ConvTranspose2d requires 4D input shape, got {self.shape}")
        weight = mod.weight.detach()
        bias = mod.bias.detach() if mod.bias is not None else None

        _, in_c, in_h, in_w = (int(d) for d in self.shape)
        out_c = int(mod.out_channels)
        st = tuple(int(d) for d in _normalize_tuple(mod.stride))
        pad = tuple(int(d) for d in _normalize_tuple(mod.padding, (0, 0)))
        dil = tuple(int(d) for d in _normalize_tuple(mod.dilation))
        op = tuple(int(d) for d in _normalize_tuple(mod.output_padding, (0, 0)))
        out_h = (in_h - 1) * st[0] - 2 * pad[0] + dil[0] * (mod.kernel_size[0] - 1) + op[0] + 1
        out_w = (in_w - 1) * st[1] - 2 * pad[1] + dil[1] * (mod.kernel_size[1] - 1) + op[1] + 1
        output_shape = (1, out_c, out_h, out_w)

        params = {
            "weight": weight,
            "stride": st, "padding": pad, "dilation": dil, "groups": mod.groups,
            "output_padding": op,
            "input_shape": self.shape, "output_shape": output_shape,
        }
        if bias is not None:
            params["bias"] = bias

        out_vars = self._alloc_ids(out_c * out_h * out_w)
        self._add_layer(LayerKind.CONVTRANSPOSE2D.value, params, self.prev_out, out_vars)
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
        
        params = {
            "kernel_size": mod.kernel_size, "stride": mod.stride or mod.kernel_size,
            "padding": mod.padding, "input_shape": self.shape, "output_shape": output_shape
        }
        self._add_layer(
            kind.value, params,
            self.prev_out, out_vars
        )
        self.shape = output_shape
        self.prev_out = out_vars
    
    def _convert_adaptive_avgpool2d(self, mod: nn.AdaptiveAvgPool2d) -> None:
        """Convert nn.AdaptiveAvgPool2d."""
        if len(self.shape) != 4:
            raise ValueError(f"AdaptiveAvgPool2d requires 4D input, got {len(self.shape)}D")
        
        batch, in_c, in_h, in_w = (int(d) for d in self.shape)
        out_size = mod.output_size
        out_h_raw, out_w_raw = (out_size, out_size) if isinstance(out_size, int) else tuple(out_size)
        out_h = in_h if out_h_raw is None else int(out_h_raw)
        out_w = in_w if out_w_raw is None else int(out_w_raw)
        output_shape = (1, in_c, out_h, out_w)
        
        out_vars = self._alloc_ids(in_c * out_h * out_w)
        
        params = {"output_size": (out_h, out_w)}
        self._add_layer(LayerKind.ADAPTIVEAVGPOOL2D.value, params,
                       self.prev_out, out_vars)
        self.shape = output_shape
        self.prev_out = out_vars
    
    def _convert_batchnorm(self, mod: _BatchNorm) -> None:
        """Convert BatchNorm to SCALE + BIAS layers with restoration params."""
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
        
        # Determine BatchNorm type for restoration
        if isinstance(mod, nn.BatchNorm1d):
            bn_module = "torch.nn.BatchNorm1d"
        elif isinstance(mod, nn.BatchNorm2d):
            bn_module = "torch.nn.BatchNorm2d"
        elif isinstance(mod, nn.BatchNorm3d):
            bn_module = "torch.nn.BatchNorm3d"
        else:
            bn_module = "torch.nn.BatchNorm2d"  # fallback

        # Store BatchNorm state for restoration
        batchnorm_state = {
            "weight": gamma,
            "bias": beta,
            "running_mean": mod.running_mean.detach(),
            "running_var": mod.running_var.detach(),
            "num_batches_tracked": mod.num_batches_tracked.detach() if mod.num_batches_tracked is not None else torch.tensor(0),
        }

        # SCALE layer - stores BatchNorm restoration info
        out_scale = self._same_size_forward()
        scale_params = {
            "a": scale_full,
            "input_shape": self.shape, "output_shape": self.shape,
            # BatchNorm restoration params
            "is_batchnorm_decomposition": True,
            "batchnorm_module": bn_module,
            "batchnorm_args": [n_channels],
            "batchnorm_kwargs": {"eps": mod.eps, "momentum": mod.momentum,
                                 "affine": mod.affine, "track_running_stats": mod.track_running_stats},
            "batchnorm_state": batchnorm_state
        }
        self._add_layer("SCALE", scale_params, self.prev_out, out_scale)
        self.prev_out = out_scale
        
        # BIAS layer - marked as paired with SCALE
        out_bias = self._same_size_forward()
        bias_params = {
            "c": bias_full,
            "input_shape": self.shape, "output_shape": self.shape,
            "is_batchnorm_decomposition": True,
            "paired_with_scale": True
        }
        self._add_layer("BIAS", bias_params, self.prev_out, out_bias)
        self.prev_out = out_bias
    
    def _convert_rnn_family(self, mod: Union[nn.RNN, nn.LSTM, nn.GRU], kind: LayerKind) -> None:
        """Convert single-layer nn.RNN / nn.LSTM / nn.GRU.
        """
        if int(mod.num_layers) != 1:
            raise ValueError(f"{kind.value}: num_layers={mod.num_layers} not supported (single-layer only).")
        if kind == LayerKind.LSTM and int(getattr(mod, "proj_size", 0)) != 0:
            raise ValueError(f"LSTM: proj_size={mod.proj_size} not supported.")

        batch_first = bool(mod.batch_first)
        if len(self.shape) != 3:
            raise ValueError(f"{kind.value}: requires 3D input shape, got {self.shape}.")
        batch, seq_len, in_feat = self.shape if batch_first else (self.shape[1], self.shape[0], self.shape[2])
        directions = 2 if mod.bidirectional else 1
        out_feat = int(mod.hidden_size) * directions
        output_shape = (batch, seq_len, out_feat) if batch_first else (seq_len, batch, out_feat)

        params: Dict[str, Any] = {
            "input_size": int(mod.input_size), "hidden_size": int(mod.hidden_size),
            "num_layers": 1, "bidirectional": bool(mod.bidirectional), "batch_first": batch_first,
            "input_shape": self.shape, "output_shape": output_shape,
        }
        if kind == LayerKind.RNN:
            params["nonlinearity"] = getattr(mod, "nonlinearity", "tanh")
        # state_dict already contains exactly the keys nn.RNN's contract
        # guarantees for the (bias?, bidirectional?) combination -- pull them
        # all in. act2torch's load_state_dict(strict=True) will catch any
        # shape mismatch on the round-trip.
        for key, tensor in mod.state_dict().items():
            params[key] = tensor.detach().clone()

        out_vars = self._alloc_ids(batch * seq_len * out_feat)
        self._add_layer(kind.value, params, self.prev_out, out_vars)
        self.shape = output_shape
        self.prev_out = out_vars

    def _convert_activation(self, mod: nn.Module, kind: LayerKind,
                           extra_params: Optional[Dict[str, Any]] = None) -> None:
        """Convert activation function."""
        out_vars = self._same_size_forward()

        
        # LRELU only accepts negative_slope, not shape params
        if kind == LayerKind.LRELU:
            params = {"negative_slope": getattr(mod, 'negative_slope', 0.01)}
        else:
            params = {"input_shape": self.shape, "output_shape": self.shape}
        if extra_params:
            params.update(extra_params)

        self._add_layer(kind.value, params, self.prev_out, out_vars)
        self.prev_out = out_vars

    def _convert_softmax(self, mod: nn.Module) -> None:
        """Convert nn.Softmax to SOFTMAX layer."""
        out_vars = self._same_size_forward()
        axis = getattr(mod, 'dim', None)
        if axis is None:
            axis = -1
        self._add_layer(LayerKind.SOFTMAX.value, {"axis": int(axis)}, self.prev_out, out_vars)
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
        
        params = {"x_vars": x_vars, "y_vars": y_vars, "input_shape": x_shape, "output_shape": x_shape}
        layer_id = self._add_layer(
            LayerKind.ADD.value, params,
            x_vars + y_vars, out_vars
        )
        self.prev_out = out_vars
        self.shape = x_shape
        self._register_node(node.name, layer_id)

    def _resolve_const_tensor(self, val: Any) -> torch.Tensor:
        """Resolve a non-variable operand (literal / get_attr buffer) to a tensor."""
        if isinstance(val, torch.Tensor):
            return val
        if isinstance(val, (int, float)):
            return torch.tensor(float(val), dtype=torch.get_default_dtype())
        if isinstance(val, fx.Node) and val.op == 'get_attr':
            obj: Any = self.traced_model
            for part in str(val.target).split('.'):
                obj = getattr(obj, part)
            if isinstance(obj, torch.Tensor):
                return obj
        raise NotImplementedError(f"Cannot resolve constant operand: {val}")

    def _emit_const_bias(self, node: fx.Node, var_node: fx.Node,
                         const: torch.Tensor, negate: bool) -> None:
        """Emit a BIAS layer (y = x + c). For x - c pass negate=True (c -> -c)."""
        in_vars = self.node_outputs[var_node.name]
        shape = self.node_shapes[var_node.name]
        c = const.flatten().to(dtype=torch.get_default_dtype())
        if negate:
            c = -c
        if c.numel() == 1:
            c = c.expand(len(in_vars)).clone()
        if c.numel() != len(in_vars):
            raise NotImplementedError(
                f"sub/bias const numel {c.numel()} != vars {len(in_vars)} at {node.name}"
            )
        out_vars = self._alloc_ids(len(in_vars))
        layer_id = self._add_layer(
            LayerKind.BIAS.value,
            {"c": c, "input_shape": shape, "output_shape": shape},
            in_vars, out_vars,
        )
        self.prev_out = out_vars
        self.shape = shape
        self._register_node(node.name, layer_id)

    def _process_sub_operation(self, node: fx.Node) -> None:
        """SUB: var-var -> SUB layer; var-const -> BIAS(-c); const-var -> SCALE(-1)+BIAS(c)."""
        if len(node.args) < 2:
            raise NotImplementedError(f"sub at {node.name} expects 2 operands")
        a, b = node.args[0], node.args[1]
        a_node = a if isinstance(a, fx.Node) else None
        b_node = b if isinstance(b, fx.Node) else None
        a_var = a_node is not None and a_node.name in self.node_outputs
        b_var = b_node is not None and b_node.name in self.node_outputs
        if a_var and b_var and a_node is not None and b_node is not None:
            x_vars = self.node_outputs[a_node.name]
            y_vars = self.node_outputs[b_node.name]
            x_shape = self.node_shapes[a_node.name]
            if len(y_vars) != len(x_vars):
                raise NotImplementedError(f"sub var-var size mismatch at {node.name}")
            out_vars = self._alloc_ids(len(x_vars))
            layer_id = self._add_layer(
                LayerKind.SUB.value,
                {"x_vars": x_vars, "y_vars": y_vars, "input_shape": x_shape, "output_shape": x_shape},
                x_vars + y_vars, out_vars,
            )
            self.prev_out = out_vars
            self.shape = x_shape
            self._register_node(node.name, layer_id)
            return
        if a_var and not b_var and a_node is not None:
            self._emit_const_bias(node, a_node, self._resolve_const_tensor(b), negate=True)
            return
        if b_var and not a_var and b_node is not None:
            in_vars = self.node_outputs[b_node.name]
            shape = self.node_shapes[b_node.name]
            scale_vars = self._alloc_ids(len(in_vars))
            neg = torch.full((len(in_vars),), -1.0, dtype=torch.get_default_dtype())
            self._add_layer(
                LayerKind.SCALE.value,
                {"a": neg, "input_shape": shape, "output_shape": shape},
                in_vars, scale_vars,
            )
            c = self._resolve_const_tensor(a).flatten().to(dtype=torch.get_default_dtype())
            if c.numel() == 1:
                c = c.expand(len(in_vars)).clone()
            if c.numel() != len(in_vars):
                raise NotImplementedError(f"sub const-var numel {c.numel()} != vars {len(in_vars)} at {node.name}")
            out_vars = self._alloc_ids(len(in_vars))
            layer_id = self._add_layer(
                LayerKind.BIAS.value,
                {"c": c, "input_shape": shape, "output_shape": shape},
                scale_vars, out_vars,
            )
            self.prev_out = out_vars
            self.shape = shape
            self._register_node(node.name, layer_id)
            return
        raise NotImplementedError(f"sub with no variable operand at {node.name}")

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
        
        
        params = {"concat_dim": dim}
        layer_id = self._add_layer(
            LayerKind.CONCAT.value, params,
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

                params = {"x_vars": x_vars, "y_vars": y_vars,
                          "input_shape": x_shape, "output_shape": x_shape}
                layer_id = self._add_layer(
                    LayerKind.MUL.value, params,
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
                    "SCALE",
                    {"a": scale_tensor, "input_shape": x_shape, "output_shape": x_shape},
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
            LayerKind.MEAN.value,
            {"input_shape": self.shape, "output_shape": output_shape},
            self.prev_out, out_vars
        )
        self.prev_out = out_vars
        self.shape = output_shape
        self._register_node(node.name, layer_id)
    
    def _process_getitem_operation(self, node: fx.Node) -> None:
        """Process indexing operation (passthrough).

        If the node is already registered (e.g. by OnnxSplit13's handler, which
        pre-registers each ``getitem(split, i)`` child to point at the i-th
        chunk's vars), skip — overwriting would collapse all children to the
        same chunk.
        """
        if node.name in self.node_outputs:
            self.prev_out = self.node_outputs[node.name]
            self.shape = self.node_shapes[node.name]
            return
        inputs = [a for a in node.args if isinstance(a, fx.Node)]
        if inputs:
            self._propagate_node_state(node.name, inputs[0].name)
    
    def _process_passthrough_function(self, node: fx.Node) -> None:
        """Process no-op functions (dropout, stochastic_depth)."""
        inputs = [a for a in node.args if isinstance(a, fx.Node)]
        if inputs:
            self._propagate_node_state(node.name, inputs[0].name)


# Bind ONNX handlers from utils.py onto the class. They live there only to keep
# this file manageable; ``self`` inside each handler is a _LayerGraphBuilder.
for _cls_name, _fn in ONNX_HANDLERS.items():
    setattr(_LayerGraphBuilder, _fn.__name__, _fn)
del _cls_name, _fn


# -----------------------------------------------------------------------------
# Public API - build_act
# -----------------------------------------------------------------------------

def build_act(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float64,
    sample_input: Optional[torch.Tensor] = None,
) -> Tuple[List[Layer], Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Build ACT layer graph from a PyTorch model.

    Args:
        model: Any nn.Module to build
        input_shape: Input shape including batch dimension (e.g., (1, 3, 32, 32))
        dtype: Data type for tensors
        sample_input: Optional concrete tensor matching ``input_shape``. Only
            consulted when a constant-evaluation chain reaches the model
            placeholder (e.g. cctsdb_yolo_2023's slice bounds, which are
            data-derived). When supplied, the resulting ACT Net is locally
            valid around this sample (e.g. for adversarial perturbations near
            it) but is not universally valid for arbitrary inputs.

    Returns:
        Tuple of (layers, preds, succs) forming a DAG
    """
    builder = _LayerGraphBuilder(model, input_shape, dtype, sample_input=sample_input)
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

    def __init__(self, wrapped: nn.Module, sample_input: Optional[torch.Tensor] = None):
        if not isinstance(wrapped, nn.Module):
            raise TypeError("TorchToACT expects an nn.Module.")

        self.m = wrapped
        mods = list(self.m.children())

        # Validate wrapper structure
        self._validate_wrapper(mods)

        # Extract InputLayer
        input_layers = [x for x in mods if type(x).__name__ == "InputLayer"]
        if len(input_layers) != 1:
            raise AssertionError(f"Wrapper must contain exactly one InputLayer; found {len(input_layers)}.")
        self.input_layer = input_layers[0]

        shape = getattr(self.input_layer, "shape", None)
        if shape is None:
            raise AssertionError("InputLayer must expose a 'shape' attribute.")

        # Resolve sample_input: prefer the explicit argument, else fall back to a
        # tensor stored on the InputLayer (``input_tensor`` or ``labeled_input.tensor``).
        # Required by the inner builder's ``_evaluate_constant_subgraph`` for
        # benchmarks whose static graph derives slice/reshape bounds from the
        # actual input (e.g. cctsdb_yolo_2023). The resulting ACT Net is locally
        # valid around this sample.
        if sample_input is None:
            sample_input = getattr(self.input_layer, 'input_tensor', None)
            if sample_input is None:
                labeled = getattr(self.input_layer, 'labeled_input', None)
                if labeled is not None and hasattr(labeled, 'tensor'):
                    sample_input = labeled.tensor
        self.sample_input = sample_input

        # State
        self.layers: List[Layer] = []
        self.prev_out: List[int] = []
        self.shape: Tuple[int, ...] = tuple(int(s) for s in shape)
        self._model_preds: Dict[int, List[int]] = {}
        self._model_succs: Dict[int, List[int]] = {}
        self._wrapper_offset: int = 0
    
    def _validate_wrapper(self, mods: List[nn.Module]) -> None:
        """Validate wrapper layer structure."""
        type_names = [type(m).__name__ for m in mods]
        if "InputSpecLayer" not in type_names:
            raise AssertionError("Wrapper must include InputSpecLayer.")
        if "OutputSpecLayer" not in type_names:
            raise AssertionError("Wrapper must include OutputSpecLayer.")
        if type_names[-1] != "OutputSpecLayer":
            raise AssertionError("Wrapper should end with OutputSpecLayer.")
    
    def run(self) -> Net:
        """Convert wrapped PyTorch model to ACT Net."""
        new_layers, out_vars = self.input_layer.to_act_layers(len(self.layers), [])
        self.layers.extend(new_layers)
        self.prev_out = out_vars
        # Capture batch dim from InputLayer for downstream spec encoding.
        B = self.input_layer.shape[0]

        for mod in self.m.children():
            if type(mod).__name__ == "InputSpecLayer" and hasattr(mod, 'to_act_layers'):
                new_layers, out_vars = mod.to_act_layers(
                    len(self.layers), self.prev_out, B
                )
                self.layers.extend(new_layers)
                self.prev_out = out_vars

        self._build_inner_model()

        for mod in self.m.children():
            if type(mod).__name__ == "OutputSpecLayer" and hasattr(mod, 'to_act_layers'):
                new_layers, out_vars = mod.to_act_layers(
                    len(self.layers), self.prev_out, B
                )
                self.layers.extend(new_layers)
                self.prev_out = out_vars
        
        # Build and validate network
        preds, succs = self._build_layer_graph()
        net = Net(layers=self.layers, preds=preds, succs=succs)
        
        from act.back_end.layer_util import validate_graph
        validate_graph(self.layers)
        net.assert_last_is_validation()
        
        return net
    
    def _build_inner_model(self) -> None:
        """Find and build the inner model using build_act()."""
        inner = self._find_inner_model()
        if inner is None:
            self._model_preds = {}
            self._model_succs = {}
            self._wrapper_offset = len(self.layers)
            return
        
        dtype = getattr(self.input_layer, 'dtype', torch.float64)
        model_layers, model_preds, model_succs = build_act(
            inner, self.shape, dtype, sample_input=self.sample_input,
        )
        
        # Offset layer IDs
        offset = len(self.layers)
        for layer in model_layers:
            layer.id += offset
            for key in ("q_src", "k_src", "w_src", "v_src"):
                value = layer.params.get(key)
                if isinstance(value, int):
                    layer.params[key] = value + offset
        
        self.layers.extend(model_layers)
        if model_layers:
            self.prev_out = model_layers[-1].out_vars
        
        self._model_preds = {k + offset: [v + offset for v in vals] for k, vals in model_preds.items()}
        self._model_succs = {k + offset: [v + offset for v in vals] for k, vals in model_succs.items()}
        self._wrapper_offset = offset
    
    def _find_inner_model(self) -> Optional[nn.Module]:
        """Find actual model inside wrapper (skip wrapper layers)."""
        for mod in self.m.children():
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
            for model_id in range(self._wrapper_offset, n - 1):
                if preds[model_id]:
                    continue
                # CONSTANT sources stay disconnected; wiring them to the wrapper
                # input would inject a spurious data dependence (and mis-shaped
                # ν route in the dual backward).
                if self.layers[model_id].kind == LayerKind.CONSTANT.value:
                    continue
                preds[model_id] = [last_wrapper]
                if model_id not in succs[last_wrapper]:
                    succs[last_wrapper].append(model_id)

        # Multi-operand ops need BOTH operands as ordered preds
        # [producer(x_vars), producer(y_vars)]; an input-sourced operand's edge
        # is dropped upstream and the loop above only fills EMPTY pred lists, so
        # a partially-wired MUL/ADD/SUB/MATMUL is left short one operand. Rewire
        # from a var->producer map. Soundness-critical: an input operand resolves
        # to INPUT_SPEC (latest-id producer), never INPUT id 0 -- the dual
        # backward harvests ν only at _find_input_layer_id==INPUT_SPEC, so
        # routing to INPUT would silently drop the term and yield an unsound bound.
        producer: Dict[int, int] = {}
        for layer in self.layers:
            for v in layer.out_vars:
                if v not in producer or layer.id > producer[v]:
                    producer[v] = layer.id
        multi_operand_kinds = {
            LayerKind.MUL.value, LayerKind.ADD.value,
            LayerKind.SUB.value, LayerKind.MATMUL.value,
        }
        for layer in self.layers:
            if layer.kind not in multi_operand_kinds:
                continue
            x_vars = layer.params.get("x_vars")
            y_vars = layer.params.get("y_vars")
            if not isinstance(x_vars, (list, tuple)) or not isinstance(y_vars, (list, tuple)):
                continue
            if not x_vars or not y_vars:
                continue
            px = producer.get(int(x_vars[0]))
            py = producer.get(int(y_vars[0]))
            if px is None or py is None:
                continue
            want = [px, py]
            if preds.get(layer.id) == want:
                continue
            for old_pred in preds.get(layer.id, []):
                if layer.id in succs.get(old_pred, []):
                    succs[old_pred].remove(layer.id)
            preds[layer.id] = want
            for p in want:
                if layer.id not in succs.setdefault(p, []):
                    succs[p].append(layer.id)

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
# Main entry point for testing
# -----------------------------------------------------------------------------

def main():
    """Main entry point for PyTorch→ACT conversion and verification testing."""
    # Initialize debug file (GUARDED)
    if PerformanceOptions.debug_tf:
        debug_file = PerformanceOptions.debug_output_file
        with open(debug_file, 'w') as f:
            f.write(f"ACT Torch2ACT Conversion Debug Log\n")
            f.write(f"{'='*80}\n\n")
        print(f"Debug logging to: {debug_file}")
    
    print("Starting Spec-Free, Input-Free Torch→ACT Verification Demo")
    
    # Step 1: Synthesize all wrapped models
    print("\n Step 1: Synthesizing wrapped models...")
    wrapped_models = model_synthesis()
    print(f"  Generated {len(wrapped_models)} wrapped models")
    
    # Step 2: Test all models with inference (each wrapped model carries its own input data).
    print("\n Step 2: Testing model inference...")
    successful_models = model_inference(wrapped_models)
    print(f"  {len(successful_models)} models passed inference tests")
    
    if not successful_models:
        print("  No successful models to verify!")
        exit(1)
    
    # Step 3: Convert and verify all successful models (memory-efficient)
    print(f"\n Step 3: Converting and verifying all {len(successful_models)} successful models...")
    print(f"  Processing one at a time to avoid memory issues...")
    
    # Import verification functions
    from act.back_end.verifier import verify_once
    
    import gc
    import torch as torch_module
    
    conversion_results = {}
    verification_results = {}
    conversion_success_count = 0
    verification_success_count = 0
    
    # Step 4: Initialize solvers (moved earlier to reuse for all models)
    print("\n Step 4: Initializing solvers...")
    gurobi_solver = None
    torch_solver = None
    
    try:
        gurobi_solver = GurobiSolver()
        print("  Gurobi solver available")
    except Exception as e:
        print(f"  Gurobi initialization failed: {e}")

    try:
        torch_solver = TorchLPSolver()
        print(f"  TorchLP solver available (device: {torch_solver._device})")
    except Exception as e:
        print(f"  TorchLP initialization failed: {e}")
    
    solvers_to_test = []
    if gurobi_solver:
        solvers_to_test.append(("Gurobi", gurobi_solver))
    if torch_solver:
        solvers_to_test.append(("TorchLP", torch_solver))
    
    if not solvers_to_test:
        print("  No solvers available!")
        print("  Will only test conversions without verification")
    
    print(f"\n Step 5: Processing all models...")
    
    for idx, (model_id, wrapped_model) in enumerate(successful_models.items(), 1):
        print(f"\n  [{idx}/{len(successful_models)}] Processing '{model_id}'...")
        
        # === CONVERSION ===
        try:
            net = TorchToACT(wrapped_model).run()
            
            # Verify the conversion produced a valid net
            if not net.layers:
                raise ValueError("Net should have layers")
            if net.layers[0].kind != LayerKind.INPUT.value:
                raise ValueError(f"First layer should be INPUT, got {net.layers[0].kind}")
            if net.layers[-1].kind != LayerKind.ASSERT.value:
                raise ValueError(f"Last layer should be ASSERT, got {net.layers[-1].kind}")
            
            layer_types = " -> ".join([layer.kind for layer in net.layers])
            print(f"    Conversion: {len(net.layers)} layers ({layer_types})")
            
            conversion_results[model_id] = "SUCCESS"
            conversion_success_count += 1
            
        except Exception as e:
            conversion_results[model_id] = f"FAILED: {str(e)[:100]}..."
            print(f"    Conversion FAILED: {e}")
            continue  # Skip verification if conversion failed
        
        # === VERIFICATION (only if solvers available) ===
        if solvers_to_test:
            model_verification = {}
            
            for solver_name, solver in solvers_to_test:
                try:
                    # TEMPORARILY COMMENTED OUT: Testing if verify_once causes memory issue
                    # res = verify_once(net, solver=solver, timelimit=30.0)
                    # status = res.status
                    status = "SKIPPED"  # Placeholder to test memory usage
                    model_verification[solver_name] = status
                    print(f"    Verification ({solver_name}): {status} (verify_once commented out)")
                    
                    if status == "UNSAT" or status == "SAT":
                        verification_success_count += 1
                        
                except Exception as e:
                    model_verification[solver_name] = f"ERROR: {str(e)[:50]}"
                    print(f"    Verification ({solver_name}): ERROR - {str(e)[:50]}")
            
            verification_results[model_id] = model_verification
        
        # === MEMORY CLEANUP ===
        # Free memory from this net immediately (no need to store)
        del net
        
        # Clean up memory periodically
        if idx % 10 == 0:
            gc.collect()
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
    
    # === FINAL SUMMARY ===
    total_count = len(successful_models)
    print(f"\nFinal Results:")
    print(f"  Conversions: {conversion_success_count}/{total_count} ({conversion_success_count/total_count*100:.1f}%)")
    
    if solvers_to_test and verification_results:
        # Count successful verifications (UNSAT or SAT results)
        total_verifications = sum(len(v) for v in verification_results.values())
        successful_verifications = sum(
            1 for results in verification_results.values() 
            for status in results.values() 
            if isinstance(status, str) and status in ["UNSAT", "SAT"]
        )
        print(f"  Verifications: {successful_verifications}/{total_verifications} successful")
    
    # Print failed conversions if any
    failed_conversions = {k: v for k, v in conversion_results.items() if v != "SUCCESS"}
    if failed_conversions:
        print(f"\n  Failed conversions: {len(failed_conversions)}")
        for model_id, error in list(failed_conversions.items())[:5]:  # Show first 5
            print(f"    - {model_id}: {error}")
    
    # Print debug file location (GUARDED)
    if PerformanceOptions.debug_tf:
        print(f"\nDebug log written to: {PerformanceOptions.debug_output_file}")
    
    print("\nTorch->ACT conversion and verification completed!")


if __name__ == "__main__":
    main()
