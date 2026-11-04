#===- act/front_end/bert_loader/create_specs.py - BERT Specs ------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Creates embedding-space InputSpec and OutputSpec pairs for SST/Yelp BERT
#   classifiers whose verification graph starts after token embedding lookup,
#   and builds the compact BERT model named by one SST/Yelp query-index row.
#
#===---------------------------------------------------------------------===#

"""Specification creator and query index for BERT verify-from-embeddings tasks."""

from __future__ import annotations

import csv
import hashlib
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn

from act.front_end.bert_loader.compact_bert import (
    CompactBertTokenizedInput,
    CompactBertRouteBModel,
    embedding_sum,
    load_compact_bert,
    resolve_checkpoint,
    tokenize_act_tokens,
)
from act.front_end.spec_creator_base import BaseSpecCreator, LabeledInputTensor, SpecTuple
from act.front_end.specs import InKind, InputSpec, OutKind, OutputSpec
from act.front_end.bert_loader.data_model_loader import (
    BertEmbeddingClassifier,
    BertExample,
    BertVocabulary,
    QueryIndexError,
    _load_query_examples,
    find_bert_dataset_name,
    load_bert_dataset,
    sample_correctly_classified,
)
from act.util.path_config import get_bert_artifact_root, get_data_root

logger = logging.getLogger(__name__)

_VALID_POSITION_MODES = frozenset(("prefix", "sweep"))
_VALID_CONVERSION_ROUTES = frozenset(("a", "b"))
_SPECIAL_WORDPIECES = frozenset(("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"))
MAX_COMPACT_VERIFY_LENGTH = 20
REQUIRED_COLUMNS = ("dataset", "split", "depth", "example_id", "position", "p", "eps")


class BertSpecCreator(BaseSpecCreator):
    """Create verification specifications from BERT dataset-model pairs."""

    def create_specs_for_data_model_pairs(
        self,
        validate_shapes: bool = True,
        *,
        dataset_names: list[str] | None = None,
        model_names: list[str] | None = None,
        num_samples: int = 1,
        split: str = "test",
        max_verify_length: int | None = None,
        epsilon: float | None = 0.1,
        p_norm: float | None = 1.0,
        perturbed_words: int | None = 1,
        checkpoint_dir: str | Path | None = None,
        position_mode: str | None = None,
        conversion_route: str | None = None,
    ) -> list[SpecTuple]:
        """Create embedding-space specs for BERT datasets.

        Args:
            dataset_names: Dataset names, defaulting to SST.
            model_names: ``embedding_classifier`` or ``compact_bert``.
            num_samples: Number of correctly classified examples per pair.
            split: Dataset split to read.
            max_verify_length: Maximum token length; defaults to 20 for compact
                BERT and 8 for the embedding classifier.
            epsilon: Embedding-space perturbation radius.
            p_norm: Lp norm metadata carried by ``LP_EMBEDDING``.
            perturbed_words: Number of positions from index zero (including CLS).
            checkpoint_dir: Compact BERT model directory containing ``checkpoint``.
            position_mode: ``prefix`` or ``sweep`` over eligible WordPieces.
            conversion_route: ``a`` for manual BERT lowering or ``b`` for FX.
            validate_shapes: Whether to validate spec/model shape compatibility.

        Returns:
            List of ``(dataset, model_name, model_from_embeddings, labeled_embeddings, spec_pairs)``.
        """
        if epsilon is None:
            epsilon = float(self.config.get("eps", 0.1))
        if p_norm is None:
            p_norm = float(self.config.get("p", 1.0))
        if perturbed_words is None:
            perturbed_words = int(self.config.get("perturbed_words", 1))
        if not (math.isfinite(epsilon) and epsilon >= 0):
            raise ValueError("epsilon must be finite and >= 0")
        if math.isnan(p_norm) or p_norm < 1:
            raise ValueError("p_norm must be at least 1")
        if checkpoint_dir is None:
            configured_checkpoint = self.config.get("checkpoint_dir")
            checkpoint_dir = cast(str | Path | None, configured_checkpoint)
        if position_mode is None:
            position_mode = str(self.config.get("position_mode", "prefix"))
        if position_mode not in _VALID_POSITION_MODES:
            raise ValueError("position_mode must be 'prefix' or 'sweep'")
        if conversion_route is None:
            conversion_route = str(self.config.get("conversion_route", "b"))
        if conversion_route not in _VALID_CONVERSION_ROUTES:
            raise ValueError("conversion_route must be 'a' or 'b'")
        datasets = [find_bert_dataset_name(name) for name in (dataset_names or ["sst"])]
        models = model_names or [
            "compact_bert" if checkpoint_dir is not None else "embedding_classifier"
        ]
        results: list[SpecTuple] = []

        for dataset in datasets:
            examples = load_bert_dataset(dataset, split)
            for model_name in models:
                if model_name == "embedding_classifier":
                    vocabulary = BertVocabulary(
                        examples,
                        embedding_dim=int(self.config.get("embedding_dim", 8)),
                    )
                    model = BertEmbeddingClassifier().eval()
                    token_cap = 8 if max_verify_length is None else max_verify_length

                    def embed_tokens(example: BertExample) -> tuple[BertExample, torch.Tensor] | None:
                        embeddings = vocabulary.lookup(example.tokens)
                        return None if embeddings.shape[1] > token_cap else (example, embeddings)

                    selected = sample_correctly_classified(
                        examples, model, embed_tokens, num_samples=num_samples, sample_kind="text"
                    )
                    labeled_embeddings = [
                        LabeledInputTensor(
                            tensor=embeddings,
                            label=torch.tensor([predicted], dtype=torch.int64),
                        )
                        for _, embeddings, predicted in selected
                    ]
                    spec_pairs = [
                        cast(
                            tuple[InputSpec, OutputSpec],
                            self._create_spec_pair(
                                embeddings=embeddings,
                                predicted=predicted,
                                epsilon=epsilon,
                                p_norm=p_norm,
                                perturbed_words=perturbed_words,
                            ),
                        )
                        for _, embeddings, predicted in selected
                    ]
                    if validate_shapes:
                        spec_pairs = self._validate_and_filter_specs(
                            spec_pairs, model, labeled_embeddings[0].tensor
                        )
                    if spec_pairs:
                        results.append((dataset, model_name, model, labeled_embeddings, spec_pairs))
                    continue

                if model_name != "compact_bert":
                    logger.warning("Skipping unsupported BERT model '%s'", model_name)
                    continue
                if checkpoint_dir is None:
                    raise ValueError("compact_bert requires checkpoint_dir")
                results.extend(self.create_compact_specs_for_examples(
                    dataset,
                    examples,
                    checkpoint_dir=checkpoint_dir,
                    num_samples=num_samples,
                    max_verify_length=max_verify_length,
                    epsilon=epsilon,
                    p_norm=p_norm,
                    perturbed_words=perturbed_words,
                    position_mode=position_mode,
                    conversion_route=conversion_route,
                    validate_shapes=validate_shapes,
                ))

        return cast(list[SpecTuple], results)

    def create_compact_specs_for_examples(
        self,
        dataset: str,
        examples: Sequence[BertExample],
        *,
        checkpoint_dir: str | Path,
        epsilon: float,
        p_norm: float,
        num_samples: int = 1,
        max_verify_length: int | None = None,
        perturbed_words: int = 1,
        position_mode: str = "prefix",
        position: int | None = None,
        conversion_route: str | None = None,
        validate_shapes: bool = True,
    ) -> list[SpecTuple]:
        """Build compact models and specs, optionally for one eligible sweep position."""
        route = conversion_route or str(self.config.get("conversion_route", "b"))
        if route not in _VALID_CONVERSION_ROUTES:
            raise ValueError("conversion_route must be 'a' or 'b'")
        loaded = load_compact_bert(checkpoint_dir)
        token_cap = MAX_COMPACT_VERIFY_LENGTH if max_verify_length is None else max_verify_length

        def embed_wordpieces(example: Any) -> tuple[CompactBertTokenizedInput, torch.Tensor] | None:
            tokenized = tokenize_act_tokens(example.tokens, loaded.tokenizer)
            if tokenized.input_ids.numel() > token_cap:
                return None
            return tokenized, embedding_sum(loaded, tokenized.input_ids, tokenized.token_type_ids)

        selected = sample_correctly_classified(
            examples, loaded.model, embed_wordpieces,
            num_samples=num_samples, sample_kind="compact BERT",
        )
        results = []
        for tokenized, embeddings, predicted in selected:
            model = (
                loaded.model if route == "a"
                else CompactBertRouteBModel(loaded.model, int(embeddings.shape[-2]))
            )
            labeled = LabeledInputTensor(
                tensor=embeddings,
                label=torch.tensor([predicted], dtype=torch.int64, device=embeddings.device),
            )
            created = self._create_spec_pair(
                embeddings=embeddings,
                predicted=predicted,
                epsilon=epsilon,
                p_norm=p_norm,
                perturbed_words=perturbed_words,
                position_mode=position_mode,
                wordpiece_tokens=tokenized.wordpiece_tokens,
                position=position,
            )
            pairs = created if isinstance(created, list) else [created]
            if validate_shapes:
                pairs = self._validate_and_filter_specs(pairs, model, embeddings)
            if pairs:
                results.append((dataset, "compact_bert", model, [labeled], pairs))
        return results

    def _create_spec_pair(
        self,
        *,
        embeddings: torch.Tensor,
        predicted: int,
        epsilon: float,
        p_norm: float,
        perturbed_words: int,
        position_mode: str = "prefix",
        wordpiece_tokens: Sequence[str] | None = None,
        position: int | None = None,
    ) -> tuple[InputSpec, OutputSpec] | list[tuple[InputSpec, OutputSpec]]:
        """Create a prefix spec or one spec for every eligible WordPiece."""
        length = embeddings.shape[-2]
        if position_mode == "prefix":
            count = max(0, min(perturbed_words, length))
            positions = torch.arange(count, dtype=torch.long, device=embeddings.device)
            return self._single_spec_pair(embeddings, predicted, epsilon, p_norm, positions)
        if position_mode not in _VALID_POSITION_MODES:
            raise ValueError("position_mode must be 'prefix' or 'sweep'")
        if wordpiece_tokens is None or len(wordpiece_tokens) != length:
            raise ValueError(
                "sweep mode requires WordPiece tokens aligned with the embedding sequence"
            )
        eligible = [
            index for index in range(1, length - 1)
            if (wordpiece_tokens[index] not in _SPECIAL_WORDPIECES
                and not wordpiece_tokens[index].startswith("##"))
        ]
        if position is not None and position not in eligible:
            raise ValueError(
                f"position {position} is not a sweep position (L={length}, eligible {eligible})"
            )
        return [
            self._single_spec_pair(
                embeddings,
                predicted,
                epsilon,
                p_norm,
                torch.tensor([index], dtype=torch.long, device=embeddings.device),
            )
            for index in ([position] if position is not None else eligible)
        ]

    @staticmethod
    def _single_spec_pair(
        embeddings: torch.Tensor,
        predicted: int,
        epsilon: float,
        p_norm: float,
        positions: torch.Tensor,
    ) -> tuple[InputSpec, OutputSpec]:
        """Create one embedding input/output robustness spec pair."""
        input_spec = InputSpec(
            kind=InKind.LP_EMBEDDING,
            center=embeddings.clone(),
            eps=torch.tensor([epsilon], dtype=embeddings.dtype),
            p_norm=p_norm,
            perturbed_positions=positions,
        )
        output_spec = OutputSpec(
            kind=OutKind.MARGIN_ROBUST,
            y_true=torch.tensor([predicted], dtype=torch.int64),
            margin=torch.tensor([0.0], dtype=embeddings.dtype),
        )
        return input_spec, output_spec


@dataclass(frozen=True)
class TextQuery:
    """One SST/Yelp verification query parsed from a query-index row."""

    dataset: str
    split: str
    depth: int
    example_id: int
    position: int
    p: float
    eps: float
    checkpoint_sha256: str | None = None

    @property
    def model_dir(self) -> Path:
        """Return the compact BERT directory that owns its checkpoint marker."""
        return Path(get_bert_artifact_root()) / "models" / f"model_{self.dataset}_{self.depth}"


def _query_field(
    row: Mapping[str, str | None],
    column: str,
    where: str,
    convert: Callable[[str], Any] = str,
    minimum: int | None = None,
) -> Any:
    """Return one required query-row cell, converted and checked against ``minimum``."""
    raw = (row.get(column) or "").strip()
    if not raw:
        raise QueryIndexError(f"{where}: column '{column}' is empty")
    try:
        value = convert(raw)
    except ValueError as error:
        kind = "an integer" if convert is int else "a number"
        raise QueryIndexError(f"{where}: {column}={raw!r} is not {kind}") from error
    if minimum is not None and value < minimum:
        raise QueryIndexError(f"{where}: {column}={value} must be >= {minimum}")
    return value


def parse_query_row(row: Mapping[str, str | None], where: str = "query") -> TextQuery:
    """Validate one CSV row and return its immutable verification query."""
    dataset_name = _query_field(row, "dataset", where)
    try:
        dataset = find_bert_dataset_name(dataset_name)
    except ValueError as error:
        raise QueryIndexError(f"{where}: {error}") from error
    p = _query_field(row, "p", where, float)
    if p not in (1.0, math.inf):
        raise QueryIndexError(f"{where}: p={p} is not supported; use inf or 1")
    eps = _query_field(row, "eps", where, float)
    if not (math.isfinite(eps) and eps > 0.0):
        raise QueryIndexError(f"{where}: eps={eps} must be finite and > 0")
    checksum = (row.get("checkpoint_sha256") or "").strip().lower() or None
    if checksum is not None and not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise QueryIndexError(f"{where}: checkpoint_sha256 is not a hex SHA-256 digest")
    return TextQuery(
        dataset=dataset,
        split=_query_field(row, "split", where),
        depth=_query_field(row, "depth", where, int, minimum=1),
        example_id=_query_field(row, "example_id", where, int, minimum=0),
        position=_query_field(row, "position", where, int, minimum=1),
        p=p,
        eps=eps,
        checkpoint_sha256=checksum,
    )


def read_query(index_path: str | Path, query_id: int) -> TextQuery:
    """Read and validate data row ``query_id`` (zero-based) from a CSV index."""
    path = Path(index_path)
    if query_id < 0:
        raise QueryIndexError(f"{path}: query id {query_id} must be >= 0")
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as error:
        raise QueryIndexError(f"cannot read query index {path}: {error}") from error
    with handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise QueryIndexError(f"{path}: missing required columns {missing}")
        rows = 0
        for row_number, row in enumerate(reader):
            if row_number == query_id:
                return parse_query_row(row, where=f"{path} query {query_id}")
            rows += 1
    raise QueryIndexError(f"{path}: query id {query_id} out of range ({rows} data rows)")


def _recorded_sha256(relative_path: str) -> str:
    """Return the manifest digest for a data-root-relative BERT artifact."""
    sums_path = Path(get_bert_artifact_root()) / "SHA256SUMS"
    try:
        lines = sums_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise QueryIndexError(f"cannot read checkpoint manifest {sums_path}: {error}") from error
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[1].lstrip("*") == relative_path:
            return parts[0].lower()
    raise QueryIndexError(f"{sums_path} has no entry for {relative_path}")


def verify_checkpoint_sha256(query: TextQuery) -> Path:
    """Verify an optional row checksum against the manifest and weights file."""
    if query.checkpoint_sha256 is None:
        raise QueryIndexError("query has no checkpoint_sha256 to verify")
    checkpoint_file = resolve_checkpoint(query.model_dir) / "pytorch_model.bin"
    relative = checkpoint_file.relative_to(Path(get_data_root()).resolve()).as_posix()
    recorded = _recorded_sha256(relative)
    if query.checkpoint_sha256 != recorded:
        raise QueryIndexError(
            f"checkpoint_sha256 {query.checkpoint_sha256} != SHA256SUMS {recorded} for {relative}"
        )
    with checkpoint_file.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != recorded:
            raise QueryIndexError(f"{checkpoint_file} does not match SHA256SUMS {recorded}")
    return checkpoint_file


def build_query_model(query: TextQuery) -> nn.Module:
    """Build the synthesized compact-BERT model for one verified query."""
    if not query.model_dir.is_dir():
        raise QueryIndexError(f"no compact BERT model directory {query.model_dir}")
    if query.checkpoint_sha256 is not None:
        verify_checkpoint_sha256(query)
    examples = _load_query_examples(query.dataset, query.split, query.example_id)
    try:
        compact_specs = BertSpecCreator().create_compact_specs_for_examples(
            query.dataset,
            [examples[query.example_id]],
            checkpoint_dir=query.model_dir,
            epsilon=query.eps,
            p_norm=query.p,
            position_mode="sweep",
            position=query.position,
        )
    except ValueError as error:
        raise QueryIndexError(f"example {query.example_id}: {error}") from error
    # Through importlib: synthesis loads the whole back end and imports this module itself.
    synthesis = import_module("act.front_end.model_synthesis")
    (model,) = synthesis.synthesize_models_from_specs(compact_specs).values()
    logger.info(
        "query %s d%d %s example %d position %d (p=%s, eps=%g)",
        query.dataset, query.depth, query.split, query.example_id,
        query.position, query.p, query.eps,
    )
    return model
