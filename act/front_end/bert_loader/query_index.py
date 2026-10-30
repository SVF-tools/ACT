#===- act/front_end/bert_loader/query_index.py - SST/Yelp Query Index ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Reads one row of an SST/Yelp query-index CSV and builds the wrapped
#   compact BERT model of that query through the BERT creator path.
#
#===---------------------------------------------------------------------===#

"""Query-index rows for checkpoint-backed SST/Yelp verification queries."""

from __future__ import annotations

import csv
import hashlib
import logging
import math
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn

from act.front_end.bert_loader.compact_bert import resolve_checkpoint
from act.front_end.bert_loader.create_specs import BertSpecCreator
from act.front_end.bert_loader.data_loader import (
    BertExample,
    _load_sst,
    _load_yelp,
    find_bert_dataset_name,
)
from act.front_end.model_synthesis import synthesize_models_from_specs
from act.util.path_config import get_bert_artifact_root, get_data_root

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "dataset",
    "split",
    "depth",
    "example_id",
    "position",
    "p",
    "eps",
)
CHECKSUM_COLUMN = "checkpoint_sha256"
YELP_PARTITION_SEED = 20260928
YELP_PARTITION_SPLITS: tuple[str, ...] = ("calibration", "evaluation")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class QueryIndexError(ValueError):
    """A query-index file, row, or named query is invalid."""


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


def _field(row: Mapping[str, str | None], column: str, where: str) -> str:
    value = (row.get(column) or "").strip()
    if not value:
        raise QueryIndexError(f"{where}: column '{column}' is empty")
    return value


def _int_field(row: Mapping[str, str | None], column: str, minimum: int, where: str) -> int:
    raw = _field(row, column, where)
    try:
        value = int(raw)
    except ValueError as error:
        raise QueryIndexError(f"{where}: {column}={raw!r} is not an integer") from error
    if value < minimum:
        raise QueryIndexError(f"{where}: {column}={value} must be >= {minimum}")
    return value


def _float_field(row: Mapping[str, str | None], column: str, where: str) -> float:
    raw = _field(row, column, where)
    try:
        return float(raw)
    except ValueError as error:
        raise QueryIndexError(f"{where}: {column}={raw!r} is not a number") from error


def parse_query_row(row: Mapping[str, str | None], where: str = "query") -> TextQuery:
    """Validate one CSV row and return its immutable verification query."""
    dataset_name = _field(row, "dataset", where)
    try:
        dataset = find_bert_dataset_name(dataset_name)
    except ValueError as error:
        raise QueryIndexError(f"{where}: {error}") from error
    p = _float_field(row, "p", where)
    if p not in (1.0, math.inf):
        raise QueryIndexError(f"{where}: p={p} is not supported; use inf or 1")
    eps = _float_field(row, "eps", where)
    if not (math.isfinite(eps) and eps > 0.0):
        raise QueryIndexError(f"{where}: eps={eps} must be finite and > 0")
    checksum = (row.get(CHECKSUM_COLUMN) or "").strip().lower() or None
    if checksum is not None and not _SHA256_PATTERN.fullmatch(checksum):
        raise QueryIndexError(f"{where}: {CHECKSUM_COLUMN} is not a hex SHA-256 digest")
    return TextQuery(
        dataset=dataset,
        split=_field(row, "split", where),
        depth=_int_field(row, "depth", 1, where),
        example_id=_int_field(row, "example_id", 0, where),
        position=_int_field(row, "position", 1, where),
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
    digest = hashlib.sha256()
    with checkpoint_file.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != recorded:
        raise QueryIndexError(f"{checkpoint_file} does not match SHA256SUMS {recorded}")
    return checkpoint_file


def yelp_partition_indices(num_examples: int) -> dict[str, tuple[int, ...]]:
    """Return the fixed 20/80 source-row partition of Yelp ``test.csv``."""
    if num_examples < 0:
        raise ValueError("num_examples must be >= 0")
    shuffled = list(range(num_examples))
    random.Random(YELP_PARTITION_SEED).shuffle(shuffled)
    calibration_size = num_examples // 5
    return {
        "calibration": tuple(sorted(shuffled[:calibration_size])),
        "evaluation": tuple(sorted(shuffled[calibration_size:])),
    }


def _raw_dataset_file(dataset: str, split: str) -> Path:
    """Return the expected raw source file without triggering fixture fallback."""
    root = Path(get_data_root()) / dataset
    if dataset == "sst":
        return root / ("train-nodes.tsv" if split == "train" else f"{split}.txt")
    if dataset == "yelp":
        return root / f"{split}.csv"
    raise QueryIndexError(f"unsupported BERT dataset {dataset!r}")


def _load_query_examples(query: TextQuery) -> list[BertExample]:
    """Load raw query data and enforce Yelp calibration/evaluation membership."""
    partitioned_yelp = query.dataset == "yelp" and query.split in YELP_PARTITION_SPLITS
    source_split = "test" if partitioned_yelp else query.split
    raw_file = _raw_dataset_file(query.dataset, source_split)
    if not raw_file.is_file():
        raise QueryIndexError(
            f"no raw {query.dataset} data for split '{query.split}': expected {raw_file}"
        )
    loader = _load_sst if query.dataset == "sst" else _load_yelp
    examples = loader(raw_file.parent, source_split)
    if not examples:
        raise QueryIndexError(f"raw {query.dataset} data has no usable examples: {raw_file}")
    if query.example_id >= len(examples):
        raise QueryIndexError(
            f"example_id {query.example_id} out of range for {query.dataset} "
            f"{query.split} ({len(examples)} examples)"
        )
    if partitioned_yelp and query.example_id not in yelp_partition_indices(len(examples))[query.split]:
        raise QueryIndexError(
            f"example_id {query.example_id} is not in Yelp {query.split} split"
        )
    return examples


def build_query_model(query: TextQuery) -> nn.Module:
    """Build the synthesized compact-BERT model for one verified query."""
    if not query.model_dir.is_dir():
        raise QueryIndexError(f"no compact BERT model directory {query.model_dir}")
    if query.checkpoint_sha256 is not None:
        verify_checkpoint_sha256(query)
    examples = _load_query_examples(query)
    try:
        compact_specs = BertSpecCreator(config_dict={}).create_compact_specs_for_examples(
            [examples[query.example_id]],
            checkpoint_dir=query.model_dir,
            epsilon=query.eps,
            p_norm=query.p,
            position_mode="sweep",
            position=query.position,
        )
    except ValueError as error:
        raise QueryIndexError(f"example {query.example_id}: {error}") from error
    wrapped_models = synthesize_models_from_specs(
        [(query.dataset, "compact_bert", model, labeled, pairs)
         for model, labeled, pairs in compact_specs]
    )
    (model,) = wrapped_models.values()
    logger.info(
        "query %s d%d %s example %d position %d (p=%s, eps=%g)",
        query.dataset,
        query.depth,
        query.split,
        query.example_id,
        query.position,
        query.p,
        query.eps,
    )
    return model
