#===- act/front_end/bert_loader/compact_bert.py - Compact checkpoint ------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
"""Strict compact BERT inference from raw three-table embedding sums.

The compact checkpoint format implies no-variance normalization when its legacy
config omits ``layer_norm``. Explicit incompatible variants are rejected.
These models are inference graphs: callers must keep them in evaluation mode.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertTokenizer
from transformers.models.bert.modeling_bert import BertEncoder, BertLayer, BertPooler

from act.util.device_manager import get_current_settings
from act.util.path_config import get_data_root


class CompactCheckpointError(ValueError):
    """An invalid or unsupported compact checkpoint."""

    def __init__(self, detail: str) -> None:
        """Keep the boundary failure available to callers."""
        self.detail = detail
        super().__init__(detail)


class NoVarLayerNorm(nn.Module):
    """Affine mean-centering without variance normalization."""

    variant: str = "no_var"

    def __init__(self, hidden_size: int) -> None:
        """Create learned scale and shift."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Normalize along the feature axis."""
        return self.weight * (inputs - inputs.mean(dim=-1, keepdim=True)) + self.bias


class CompactBertFromEmbeddings(nn.Module):
    """Route A: explicit HF encoder blocks for dedicated ACT lowering."""

    def __init__(self, config: BertConfig) -> None:
        """Build only the supported ReLU, absolute-position encoder."""
        super().__init__()
        if (config.hidden_act != "relu" or config.is_decoder or config.add_cross_attention
                or getattr(config, "position_embedding_type", "absolute") != "absolute"
                or getattr(config, "pruned_heads", {})
                or getattr(config, "layer_norm", "no_var") != "no_var"
                or config.model_type != "bert"):
            raise CompactCheckpointError("requires ReLU/no_var absolute-position BERT encoder")
        dimensions = (config.hidden_size, config.num_attention_heads, config.num_hidden_layers,
                      config.intermediate_size, config.vocab_size,
                      config.max_position_embeddings, config.type_vocab_size, config.num_labels)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise CompactCheckpointError("BERT dimensions must be positive integers")
        if config.hidden_size % config.num_attention_heads or config.num_labels < 2:
            raise CompactCheckpointError("invalid head dimensions or classifier size")
        self.config = deepcopy(config)
        self.config._attn_implementation = "eager"
        self.embeddings = nn.Sequential(OrderedDict(
            LayerNorm=NoVarLayerNorm(config.hidden_size)))
        self.encoder = BertEncoder(self.config)
        for block in self.encoder.layer:
            assert isinstance(block, BertLayer)
            block.attention.output.set_submodule("LayerNorm", NoVarLayerNorm(config.hidden_size))
            block.output.set_submodule("LayerNorm", NoVarLayerNorm(config.hidden_size))
        self.pooler = BertPooler(self.config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.eval()

    def forward(self, embedding_sum: torch.Tensor) -> torch.Tensor:
        """Classify unmasked embedding sums shaped [batch, sequence, hidden]."""
        hidden = self.encoder(self.embeddings(embedding_sum), return_dict=False)[0]
        return self.classifier(self.pooler(hidden))


class CompactBertRouteBBlock(nn.Module):
    """Equivalent inference block consisting only of FX-traceable operations."""

    def __init__(self, source: BertLayer, sequence_length: int) -> None:
        """Copy trained projections and norms, without a parallel operator system."""
        super().__init__()
        self.q_proj = deepcopy(source.attention.self.query)
        self.k_proj = deepcopy(source.attention.self.key)
        self.v_proj = deepcopy(source.attention.self.value)
        self.attn_proj = deepcopy(source.attention.output.dense)
        self.attn_norm = deepcopy(source.attention.output.LayerNorm)
        self.ff_in = deepcopy(source.intermediate.dense)
        self.relu = nn.ReLU()
        self.ff_out = deepcopy(source.output.dense)
        self.ff_norm = deepcopy(source.output.LayerNorm)
        self.softmax = nn.Softmax(dim=-1)
        self.sequence_length = sequence_length
        self.num_heads = source.attention.self.num_attention_heads
        self.head_dim = source.attention.self.attention_head_size
        self.hidden_size = self.num_heads * self.head_dim

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        """Separate head features from token coordinates."""
        return value.reshape(-1, self.sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Evaluate self-attention followed by the residual feed-forward block."""
        query, key, value = (self._heads(proj(hidden))
                             for proj in (self.q_proj, self.k_proj, self.v_proj))
        probabilities = self.softmax(torch.matmul(query, key.transpose(-1, -2))
                                     / math.sqrt(self.head_dim))
        context = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
        context = context.reshape(-1, self.sequence_length, self.hidden_size)
        attention = self.attn_norm(self.attn_proj(context) + hidden)
        return self.ff_norm(self.ff_out(self.relu(self.ff_in(attention))) + attention)


class CompactBertRouteBModel(nn.Module):
    """Whole-tensor route B, specialized to a verification sequence length."""

    def __init__(self, source: CompactBertFromEmbeddings, sequence_length: int) -> None:
        """Preserve inference semantics, rejecting training-mode dropout."""
        super().__init__()
        if any(module.training for module in source.modules()):
            raise CompactCheckpointError("route B requires an evaluation-mode source")
        if type(sequence_length) is not int or not 1 <= sequence_length <= source.config.max_position_embeddings:
            raise CompactCheckpointError("sequence length is outside the position table")
        self.input_norm = deepcopy(source.embeddings.get_submodule("LayerNorm"))
        self.stages = nn.ModuleList()
        for block in source.encoder.layer:
            assert isinstance(block, BertLayer)
            self.stages.append(CompactBertRouteBBlock(block, sequence_length))
        self.pool_proj = deepcopy(source.pooler.dense)
        self.pool_activation = nn.Tanh()
        self.output_proj = deepcopy(source.classifier)
        self.eval()

    def forward(self, embedding_sum: torch.Tensor) -> torch.Tensor:
        """Classify the same raw embedding sum as route A."""
        hidden = self.input_norm(embedding_sum)
        for stage in self.stages:
            hidden = stage(hidden)
        return self.output_proj(self.pool_activation(self.pool_proj(hidden[:, 0])))


@dataclass(frozen=True, slots=True)
class CompactBertTokenizedInput:
    """Checkpoint WordPiece tokens with aligned integer IDs."""

    wordpiece_tokens: list[str]
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor


@dataclass(frozen=True, slots=True)
class LoadedCompactBert:
    """Validated inference model and the three input embedding tables."""

    model: CompactBertFromEmbeddings
    tokenizer: BertTokenizer
    word_embeddings: torch.Tensor
    position_embeddings: torch.Tensor
    token_type_embeddings: torch.Tensor
    model_dir: Path
    checkpoint_dir: Path
    config: BertConfig
    consumed_checkpoint_keys: tuple[str, ...]
    unused_checkpoint_keys: tuple[str, ...]


def resolve_checkpoint(model_dir: str | Path) -> Path:
    """Resolve a data-root-relative directory and require its checkpoint artifacts."""
    directory = (Path(get_data_root()) / model_dir).resolve()
    marker = (directory / "checkpoint").read_text(encoding="utf-8").strip()
    if not marker or not marker.isascii() or not marker.isdecimal():
        raise CompactCheckpointError("checkpoint marker must be an ASCII nonnegative integer")
    checkpoint = directory / f"ckpt-{int(marker)}"
    for filename in ("config.json", "pytorch_model.bin", "vocab.txt"):
        if not (checkpoint / filename).is_file():
            raise FileNotFoundError(checkpoint / filename)
    return checkpoint


def load_compact_bert(
    model_dir: str | Path, *, device: torch.device | str | None = None,
    dtype: torch.dtype | None = None, strict: bool = True,
) -> LoadedCompactBert:
    """Load explicit parameter mappings and reject missing, malformed or unused weights."""
    checkpoint = resolve_checkpoint(model_dir)
    values = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    required = {"hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size",
                "vocab_size", "max_position_embeddings", "type_vocab_size", "hidden_act"}
    if not isinstance(values, dict) or not required.issubset(values):
        raise CompactCheckpointError("config must specify all architecture dimensions and hidden_act")
    config = BertConfig.from_dict(values)
    model = CompactBertFromEmbeddings(config)
    default_device, default_dtype = get_current_settings()
    target_device = default_device if device is None else device
    target_dtype = default_dtype if dtype is None else dtype
    model.to(device=target_device, dtype=target_dtype)
    state = torch.load(checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise CompactCheckpointError("weights must be a string-to-tensor state dictionary")
    key_map = {key: key if key.startswith("classifier.") else f"bert.{key}"
               for key in model.state_dict()}
    table_keys = tuple(f"bert.embeddings.{name}_embeddings.weight"
                       for name in ("word", "position", "token_type"))
    consumed = set(key_map.values()) | set(table_keys)
    missing, unused = sorted(consumed - state.keys()), tuple(sorted(state.keys() - consumed))
    if missing or (strict and unused):
        raise CompactCheckpointError(f"missing checkpoint keys: {missing}; unused keys: {unused}")
    expected = {source: model.state_dict()[target].shape for target, source in key_map.items()}
    expected.update(zip(table_keys, ((config.vocab_size, config.hidden_size),
                                    (config.max_position_embeddings, config.hidden_size),
                                    (config.type_vocab_size, config.hidden_size))))
    for key, shape in expected.items():
        tensor = state[key]
        if tensor.shape != shape or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise CompactCheckpointError(f"invalid shape, dtype or nonfinite values for {key}")
    vocab = (checkpoint / "vocab.txt").read_text(encoding="utf-8").splitlines()
    if (len(vocab) != config.vocab_size or len(set(vocab)) != len(vocab) or "" in vocab
            or not {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}.issubset(vocab)):
        raise CompactCheckpointError("vocab must match the word table and contain special tokens")
    model.load_state_dict({target: state[source] for target, source in key_map.items()}, strict=True)
    tables = [state[key].to(device=target_device, dtype=target_dtype) for key in table_keys]
    tokenizer = BertTokenizer(
        vocab_file=str(checkpoint / "vocab.txt"),
        do_lower_case=True,
    )
    return LoadedCompactBert(model, tokenizer, tables[0], tables[1], tables[2], checkpoint.parent, checkpoint,
                             config, tuple(sorted(consumed)), unused)


def tokenize_act_tokens(tokens: Sequence[str], tokenizer: BertTokenizer) -> CompactBertTokenizedInput:
    """Use checkpoint WordPiece tokenization on space-joined ACT tokens."""
    pieces = ["[CLS]", *tokenizer.tokenize(" ".join(tokens)), "[SEP]"]
    ids = torch.tensor(tokenizer.convert_tokens_to_ids(pieces), dtype=torch.long)
    return CompactBertTokenizedInput(pieces, ids, torch.zeros_like(ids))


def embedding_sum(
    loaded: LoadedCompactBert, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sum word, absolute position and token-type embeddings before normalization."""
    ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
    if ids.dim() != 2 or ids.dtype not in (torch.int32, torch.int64):
        raise CompactCheckpointError("input_ids must be integer [L] or [B, L]")
    ids = ids.to(device=loaded.word_embeddings.device)
    types = torch.zeros_like(ids) if token_type_ids is None else token_type_ids
    types = types.unsqueeze(0) if types.dim() == 1 else types
    if types.shape != ids.shape or types.dtype not in (torch.int32, torch.int64):
        raise CompactCheckpointError("token_type_ids must be integer IDs aligned with input_ids")
    types = types.to(device=ids.device)
    positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    return (F.embedding(ids, loaded.word_embeddings)
            + F.embedding(positions, loaded.position_embeddings)
            + F.embedding(types, loaded.token_type_embeddings))
