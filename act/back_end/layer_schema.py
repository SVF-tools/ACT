#===- act/back_end/layer_schema.py - ACT Layer Schema and Registry -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   ACT layer schema definitions, strict registry, wrapper checks,
#   and validation for layer types and parameters.
#
#===---------------------------------------------------------------------===#

"""
- Enums like DataFormat/PaddingMode and dataclasses like ConvMeta/PoolMeta/NormMeta are *convenience types*.
  They provide defaults and IDE/type hints, but they are **not required** for validation. In a verification
  toolchain where you want a slim, explicit surface, they can be replaced by plain strings/tuples stored in
  `Layer.meta` and validated by a central registry.

WHAT THIS FILE PROVIDES (concise):
1) LayerKind enum + Layer dataclass (the only structured types you need).
2) A single REGISTRY that lists **all allowed params/meta keys** per kind.
3) Strict validators in layer_util.py: `validate_layer`, `validate_graph`, and `validate_wrapper_graph` (for wrapper layout).
4) `create_layer(...)` helper that validates on creation.
5) Clear header on how to add new kinds/keys.
6) A tiny usage example runnable via `python layer_util.py`.

HOW TO ADD NEW STUFF (READ THIS):
- Add a new LAYER KIND:
    1. Append a value to LayerKind (e.g., MYOP = "MYOP").
    2. Add REGISTRY[LayerKind.MYOP.value] = {
           "params_required": [...],
           "params_optional": [...],
           "meta_required":   [...],
           "meta_optional":   [...],
       }
    3. Done. The validator will enforce that only those keys are used.

- Add a NEW PARAM or META KEY to an existing kind:
    * If every instance MUST have it, put it in the corresponding `..._required` list.
    * Otherwise, add it to `..._optional`.
    * Re-run; unknown keys will fail with a message that suggests the closest valid key
      or tells you to add the key to REGISTRY.

WRAPPER LAYOUT (validated by `validate_wrapper_graph` in layer_util.py):
InputLayer → InputSpecLayer → Model → OutputSpecLayer
- Exactly one `INPUT`
- ≥1 `INPUT_SPEC`
- Final layer must be `ASSERT`
- Preprocessing (normalization, resizing, channel conversion) handled by data loader
  (e.g., torchvision.transforms.Compose or create_preprocessing_pipeline())
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
import enum

# Import Layer from core to avoid circular import issues
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .core import Layer

try:
    import torch
    Tensor = torch.Tensor
except Exception:  # typing only
    Tensor = "torch.Tensor"  # type: ignore

# -------------------------------
# Minimal enum of operation kinds
# -------------------------------
class LayerKind(str, enum.Enum):
    # Wrapper & specs
    INPUT = "INPUT"           # meta: shape (required), params: center (optional)
    INPUT_SPEC = "INPUT_SPEC" # meta: kind ('BOX'|'LINF_BALL'|'LIN_POLY'), constraints in meta
    ASSERT = "ASSERT"         # meta: kind ('LINEAR_LE'|'TOP1_ROBUST'|'MARGIN_ROBUST'|'RANGE'), fields in meta

    # Core MLP/CNN ops (subset can be extended easily)
    DENSE = "DENSE"
    CONV1D = "CONV1D"
    CONV2D = "CONV2D"
    CONV3D = "CONV3D"
    CONVTRANSPOSE2D = "CONVTRANSPOSE2D"

    # Pooling
    MAXPOOL1D = "MAXPOOL1D"
    MAXPOOL2D = "MAXPOOL2D"
    MAXPOOL3D = "MAXPOOL3D"
    AVGPOOL1D = "AVGPOOL1D"
    AVGPOOL2D = "AVGPOOL2D"
    AVGPOOL3D = "AVGPOOL3D"
    ADAPTIVEAVGPOOL2D = "ADAPTIVEAVGPOOL2D"

    # Activations / elementwise
    RELU = "RELU"
    LRELU = "LRELU"
    PRELU = "PRELU"
    SIGMOID = "SIGMOID"
    TANH = "TANH"
    SOFTPLUS = "SOFTPLUS"
    SILU = "SILU"
    GELU = "GELU"
    RELU6 = "RELU6"
    HARDTANH = "HARDTANH"
    HARDSIGMOID = "HARDSIGMOID"
    HARDSWISH = "HARDSWISH"
    MISH = "MISH"
    SOFTSIGN = "SOFTSIGN"
    ABS = "ABS"
    CLIP = "CLIP"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    POW = "POW"
    MIN = "MIN"
    MAX = "MAX"

    # Tensor plumbing
    CONCAT = "CONCAT"
    STACK = "STACK"
    RESHAPE = "RESHAPE"
    FLATTEN = "FLATTEN"
    TRANSPOSE = "TRANSPOSE"
    SQUEEZE = "SQUEEZE"
    UNSQUEEZE = "UNSQUEEZE"
    TILE = "TILE"
    EXPAND = "EXPAND"
    UPSAMPLE = "UPSAMPLE"

    # Sequences & attention
    EMBEDDING = "EMBEDDING"
    RNN = "RNN"
    GRU = "GRU"
    LSTM = "LSTM"
    SOFTMAX = "SOFTMAX"
    MHA = "MHA"
    POSENC = "POSENC"

# -------------------------------------------
# Strict schema: flat registry (easy to edit)
# -------------------------------------------
REGISTRY: Dict[str, Dict[str, List[str]]] = {
    # Wrapper & specs
    LayerKind.INPUT.value:       {
        "params_required": [],
        "params_optional": [
            "labeled_input",    # Optional: LabeledInputTensor (tensor + label pair) for self-contained model inference
        ], 
        "meta_required": [
            "shape",            # Required tuple: input shape including batch=1 (e.g., (1, 784) or (1, 3, 32, 32))
            "dtype",            # Required str: tensor data type (e.g., "torch.float32", "torch.float64") - CRITICAL for verification soundness
        ], 
        "meta_optional": [
            "desc",             # Optional str: human-readable description (default: "input")
            # Tier 1: Essential metadata for data characterization
            "layout",           # Optional str: data format - "CHW" (channel-first), "HWC" (channel-last), "FLAT" (flattened)
            "dataset_name",     # Optional str: dataset identifier (e.g., "mnist", "cifar10", "custom_data")
            # Tier 2: Important metadata for verification context
            "num_classes",      # Optional int: number of output classes for classification tasks
            "value_range",      # Optional tuple: (min, max) actual value range in data (e.g., (0.0, 1.0) or (0.0, 255.0))
            "scale_hint",       # Optional str: scale description - "0-1", "0-255", "normalized", "unknown"
            "distribution",     # Optional str: data distribution - "uniform", "normal", "normalized", "unknown", or custom (free-form)
            # Tier 3: Optional metadata for debugging and tracking
            "sample_id",        # Optional int/str: sample identifier for tracking individual inputs
            "domain",           # Optional str: problem domain - "vision", "tabular", "text", "audio"
            "channels",         # Optional int: number of channels (e.g., 1 for grayscale, 3 for RGB)
        ]
    },
    LayerKind.INPUT_SPEC.value:  {"params_required": [], "params_optional": ["lb","ub","center","A","b"], "meta_required": ["kind"], "meta_optional": ["eps","lb_val","ub_val","center_val"]},
    LayerKind.ASSERT.value:      {"params_required": [], "params_optional": ["c","lb","ub"], "meta_required": ["kind"], "meta_optional": ["d","y_true","margin"]},

    # Dense/CNN
    LayerKind.DENSE.value:       {"params_required": ["W"], "params_optional": ["b","W_pos","W_neg"], "meta_required": [], "meta_optional": ["activation","input_shape","output_shape","bias_enabled","in_features","out_features"]},
    LayerKind.CONV1D.value:      {"params_required": ["weight"], "params_optional": ["bias","weight_pos","weight_neg"], "meta_required": ["stride","padding","dilation","groups"], "meta_optional": ["transposed","output_padding","padding_mode","input_shape","output_shape","data_format"]},
    LayerKind.CONV2D.value:      {"params_required": ["weight"], "params_optional": ["bias"], "meta_required": ["input_shape","output_shape"], "meta_optional": ["stride","padding","dilation","groups","kernel_size","in_channels","out_channels","transposed","output_padding","padding_mode","data_format"]},
    LayerKind.CONV3D.value:      {"params_required": ["weight"], "params_optional": ["bias","weight_pos","weight_neg"], "meta_required": ["stride","padding","dilation","groups"], "meta_optional": ["transposed","output_padding","padding_mode","input_shape","output_shape","data_format"]},
    LayerKind.CONVTRANSPOSE2D.value: {"params_required": ["weight"], "params_optional": ["bias"], "meta_required": ["stride","padding","dilation","groups"], "meta_optional": ["transposed","output_padding","padding_mode","input_shape","output_shape","data_format"]},

    # Pooling
    LayerKind.MAXPOOL1D.value:   {"params_required": [], "params_optional": [], "meta_required": ["kernel_size"], "meta_optional": ["stride","padding","dilation","ceil_mode","count_include_pad","output_size"]},
    LayerKind.MAXPOOL2D.value:   {"params_required": [], "params_optional": [], "meta_required": ["kernel_size"], "meta_optional": ["stride","padding","dilation","ceil_mode","count_include_pad","output_size","input_shape","output_shape"]},
    LayerKind.MAXPOOL3D.value:   {"params_required": [], "params_optional": [], "meta_required": ["kernel_size"], "meta_optional": ["stride","padding","dilation","ceil_mode","count_include_pad","output_size"]},
    LayerKind.AVGPOOL1D.value:   {"params_required": [], "params_optional": [], "meta_required": ["kernel_size"], "meta_optional": ["stride","padding","dilation","ceil_mode","count_include_pad","output_size"]},
    LayerKind.AVGPOOL2D.value:   {"params_required": [], "params_optional": [], "meta_required": ["kernel_size"], "meta_optional": ["stride","padding","dilation","ceil_mode","count_include_pad","output_size","input_shape","output_shape"]},
    LayerKind.AVGPOOL3D.value:   {"params_required": [], "params_optional": [], "meta_required": ["kernel_size"], "meta_optional": ["stride","padding","dilation","ceil_mode","count_include_pad","output_size"]},
    LayerKind.ADAPTIVEAVGPOOL2D.value: {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["output_size"]},

    # Activations / elementwise
    LayerKind.RELU.value:        {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["input_shape","output_shape"]},
    LayerKind.LRELU.value:       {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["negative_slope"]},
    LayerKind.PRELU.value:       {"params_required": ["weight"], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.SIGMOID.value:     {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.TANH.value:        {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.SOFTPLUS.value:    {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.SILU.value:        {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["input_shape","output_shape"]},
    LayerKind.GELU.value:        {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["approximate"]},
    LayerKind.RELU6.value:       {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.HARDTANH.value:    {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["min_val","max_val"]},
    LayerKind.HARDSIGMOID.value: {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["alpha","beta"]},
    LayerKind.HARDSWISH.value:   {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.SOFTSIGN.value:    {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.ABS.value:         {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.CLIP.value:        {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["min","max"]},
    LayerKind.ADD.value:         {"params_required": [], "params_optional": ["bias"], "meta_required": [], "meta_optional": ["broadcast","axis","input_shape","output_shape","original_shape","x_vars","y_vars"]},
    LayerKind.SUB.value:         {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["broadcast","axis"]},
    LayerKind.MUL.value:         {"params_required": [], "params_optional": ["scale"], "meta_required": [], "meta_optional": ["broadcast","axis","input_shape","output_shape","original_shape"]},
    LayerKind.DIV.value:         {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["broadcast","axis"]},
    LayerKind.POW.value:         {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["broadcast","axis"]},
    LayerKind.MIN.value:         {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["broadcast","axis"]},
    LayerKind.MAX.value:         {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["broadcast","axis"]},

    # Tensor plumbing
    LayerKind.CONCAT.value:      {"params_required": [], "params_optional": [], "meta_required": ["concat_dim"], "meta_optional": []},
    LayerKind.STACK.value:       {"params_required": [], "params_optional": [], "meta_required": ["axis"], "meta_optional": []},
    LayerKind.RESHAPE.value:     {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["target_shape"]},
    LayerKind.FLATTEN.value:     {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["start_dim","end_dim","input_shape","output_shape"]},
    LayerKind.TRANSPOSE.value:   {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["perm"]},
    LayerKind.SQUEEZE.value:     {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["dims"]},
    LayerKind.UNSQUEEZE.value:   {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["dims"]},
    LayerKind.TILE.value:        {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["repeats"]},
    LayerKind.EXPAND.value:      {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["shape"]},
    LayerKind.UPSAMPLE.value:    {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["mode","align_corners","scale_factor","size"]},

    # Sequences / attention
    LayerKind.EMBEDDING.value:   {"params_required": ["weight"], "params_optional": [], "meta_required": ["num_embeddings","embedding_dim"], "meta_optional": ["padding_idx","max_norm","norm_type","scale_grad_by_freq","sparse"]},
    LayerKind.RNN.value:         {"params_required": [], "params_optional": ["weight_ih_l0","weight_hh_l0","bias_ih_l0","bias_hh_l0"], "meta_required": ["input_size","hidden_size","num_layers","bidirectional"], "meta_optional": ["dropout","batch_first","nonlinearity","proj_size","gate_order","packed_sequence"]},
    LayerKind.GRU.value:         {"params_required": [], "params_optional": ["weight_ih_l0","weight_hh_l0","bias_ih_l0","bias_hh_l0"], "meta_required": ["input_size","hidden_size","num_layers","bidirectional"], "meta_optional": ["dropout","batch_first","nonlinearity","proj_size","gate_order","packed_sequence"]},
    LayerKind.LSTM.value:        {"params_required": [], "params_optional": ["weight_ih_l0","weight_hh_l0","bias_ih_l0","bias_hh_l0"], "meta_required": ["input_size","hidden_size","num_layers","bidirectional"], "meta_optional": ["dropout","batch_first","nonlinearity","proj_size","gate_order","packed_sequence"]},
    LayerKind.SOFTMAX.value:     {"params_required": [], "params_optional": [], "meta_required": ["axis"], "meta_optional": []},
    LayerKind.MHA.value:         {"params_required": [], "params_optional": ["in_proj_weight","in_proj_bias","q_proj.weight","q_proj.bias","k_proj.weight","k_proj.bias","v_proj.weight","v_proj.bias","out_proj.weight","out_proj.bias","bias_k","bias_v","rel_pos_bias"], "meta_required": ["num_heads"], "meta_optional": ["head_dim","scale","dropout","add_zero_attn","batch_first","causal","mask_kind","mask_format","axis","qkv_layout","posenc_kind","rope_theta"]},
    LayerKind.POSENC.value:      {"params_required": [], "params_optional": ["weight","slopes"], "meta_required": [], "meta_optional": ["kind","seq_len","embedding_dim","theta"]},
}

# Supported exporter op tags (base name before ":").
SUPPORTED_EXPORT_OPS = {
    "box",
    "dense",
    "bias",
    "scale",
    "bn",
    "add",
    "relu",
    "lrelu",
    "tanh",
    "sigmoid",
    "abs",
    "mcc",
    "conv2d",
    "maxpool2d",
    "avgpool2d",
    "flatten",
    "reshape",
    "top1",
    "range",
    "max",
    "min",
    "softmax",
    "in",
    "posenc",
    "layernorm",
    "gelu",
    "att_scores",
    "att_mix",
    "mask",
}
