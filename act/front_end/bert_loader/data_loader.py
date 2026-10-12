#===- act/front_end/bert_loader/data_loader.py - BERT Data Loading ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Loads SST/Yelp text examples, builds token vocabularies, materializes clean
#   embedding tensors, and samples correctly classified verification inputs for
#   BERT-style embedding-space verification.
#
#===---------------------------------------------------------------------===#

"""BERT-style text dataset utilities for verify-from-embeddings front ends."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import csv
import hashlib
import logging
from pathlib import Path
import re
from typing import cast, override

import torch
import torch.nn as nn

from act.util.path_config import get_data_root

logger = logging.getLogger(__name__)

BERT_DATASETS: dict[str, str] = {
    "sst": "SST binary sentiment",
    "bert": "Alias for SST binary sentiment",
    "yelp": "Yelp binary sentiment",
}

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class BertExample:
    """A tokenized text classification example.

    Args:
        label: Integer class label.
        tokens: Tokenized sentence.
        word_labels: Optional SST node labels aligned with ``tokens``.
    """

    label: int
    tokens: list[str]
    word_labels: list[int] | None = None


class BertVocabulary:
    """Vocabulary and deterministic embedding table for text examples."""

    def __init__(self, examples: Sequence[BertExample], embedding_dim: int = 8) -> None:
        """Build a vocabulary and embedding matrix.

        Args:
            examples: Tokenized examples used to collect vocabulary entries.
            embedding_dim: Width of each token embedding.

        Raises:
            ValueError: If ``embedding_dim`` is too small for the classifier.
        """
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")

        self.embedding_dim: int = embedding_dim
        words = sorted({token for example in examples for token in example.tokens})
        self.token_to_id: dict[str, int] = {"[PAD]": 0, "[UNK]": 1}
        self.token_to_id.update({word: i + 2 for i, word in enumerate(words)})
        self.embeddings: torch.Tensor = torch.stack(
            [self._embedding_for_token(token) for token in self.token_to_id]
        )

    def encode(self, tokens: Sequence[str]) -> torch.Tensor:
        """Encode tokens as vocabulary IDs.

        Args:
            tokens: Token sequence.

        Returns:
            Long tensor of vocabulary IDs.
        """
        ids = [self.token_to_id.get(token, self.token_to_id["[UNK]"]) for token in tokens]
        return torch.tensor(ids, dtype=torch.long)

    def lookup(self, tokens: Sequence[str]) -> torch.Tensor:
        """Look up clean token embeddings.

        Args:
            tokens: Token sequence.

        Returns:
            Embedding tensor with shape ``[1, L, D]``.
        """
        ids = self.encode(tokens)
        return self.embeddings.index_select(0, ids).unsqueeze(0)

    def _embedding_for_token(self, token: str) -> torch.Tensor:
        """Create a deterministic embedding for a token.

        Args:
            token: Vocabulary token.

        Returns:
            Float tensor of shape ``[embedding_dim]``.
        """
        vector = torch.zeros(self.embedding_dim, dtype=torch.float32)
        if token == "[PAD]":
            return vector

        vector[0] = _sentiment_axis(token)
        vector[1] = 1.0 if token != "[UNK]" else 0.0
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(2, self.embedding_dim):
            vector[i] = (digest[i] / 255.0 - 0.5) * 0.1
        return vector


class BertEmbeddingClassifier(nn.Module):
    """Tiny classifier whose public forward path consumes embeddings."""

    def __init__(self) -> None:
        """Initialize an embedding-space binary sentiment classifier."""
        super().__init__()

    @override
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Classify directly from clean or perturbed embeddings.

        Args:
            embeddings: Tensor with shape ``[B, L, D]`` or ``[L, D]``.

        Returns:
            Logits with shape ``[B, 2]``.

        Raises:
            ValueError: If the embedding tensor is malformed.
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        if embeddings.dim() != 3 or embeddings.shape[-1] < 1:
            raise ValueError("embeddings must have shape [B, L, D]")

        score = embeddings[..., 0].mean(dim=1)
        return torch.stack([-score, score], dim=-1)


def list_bert_datasets() -> list[str]:
    """List available BERT dataset names.

    Returns:
        Sorted dataset names and aliases.
    """
    return sorted(BERT_DATASETS)


def find_bert_dataset_name(name: str) -> str:
    """Resolve a BERT dataset or alias name.

    Args:
        name: Dataset name or alias.

    Returns:
        Normalized dataset name.

    Raises:
        ValueError: If the name is unknown.
    """
    key = name.lower()
    if key == "bert":
        return "sst"
    if key in BERT_DATASETS:
        return key
    raise ValueError(f"BERT dataset '{name}' not found; available: {list_bert_datasets()}")


def load_bert_dataset(dataset_name: str = "sst", split: str = "test") -> list[BertExample]:
    """Load a BERT dataset from disk or deterministic fixture data.

    Args:
        dataset_name: BERT dataset name, currently ``sst`` or ``yelp``.
        split: Dataset split name.

    Returns:
        Tokenized examples.

    Raises:
        ValueError: If ``dataset_name`` is not supported.
    """
    dataset = find_bert_dataset_name(dataset_name)
    root = Path(get_data_root()) / dataset
    if dataset == "sst":
        examples = _load_sst(root, split)
    elif dataset == "yelp":
        examples = _load_yelp(root, split)
    else:
        raise ValueError(f"Unsupported BERT dataset '{dataset_name}'")

    if examples:
        logger.info("Loaded %d %s examples from %s", len(examples), dataset, root)
        return examples

    logger.warning("Using deterministic %s fixture because raw data is unavailable", dataset)
    return _synthetic_examples(dataset)


def sample_correctly_classified(
    examples: Sequence[BertExample],
    model: nn.Module,
    vocabulary: BertVocabulary,
    *,
    num_samples: int,
    max_verify_length: int,
) -> list[tuple[BertExample, torch.Tensor, int]]:
    """Select correctly classified examples within the verification length cap.

    Args:
        examples: Candidate examples.
        model: Embedding-space classifier.
        vocabulary: Vocabulary used for embedding lookup.
        num_samples: Number of examples to return.
        max_verify_length: Maximum allowed embedding sequence length.

    Returns:
        Tuples of ``(example, embedding_tensor, predicted_label)``.

    Raises:
        ValueError: If no eligible examples can be found.
    """
    selected: list[tuple[BertExample, torch.Tensor, int]] = []
    _ = model.eval()

    with torch.no_grad():
        for example in examples:
            embeddings = vocabulary.lookup(example.tokens)
            if embeddings.shape[1] > max_verify_length:
                continue
            logits = cast(torch.Tensor, model(embeddings))
            predicted = int(logits.argmax(dim=-1).item())
            if predicted != example.label:
                continue
            selected.append((example, embeddings, predicted))
            if len(selected) >= num_samples:
                break

    if not selected:
        raise ValueError(
            "No correctly classified text samples found within max_verify_length"
        )
    return selected


def _load_sst(root: Path, split: str) -> list[BertExample]:
    """Load SST examples from the expected raw file layout."""
    if split == "train":
        path = root / "train-nodes.tsv"
        if not path.exists():
            return []
        examples: list[BertExample] = []
        rows = path.read_text(encoding="utf-8").splitlines()[1:]
        for line in rows:
            text, label = line.split("\t")[:2]
            examples.append(BertExample(label=int(label), tokens=tokenize(text)))
        return examples

    path = root / f"{split}.txt"
    if not path.exists():
        return []

    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_sst_tree_line(line)
        if parsed is not None:
            examples.append(parsed)
    return examples


def _load_yelp(root: Path, split: str) -> list[BertExample]:
    """Load Yelp examples from ``label,text`` CSV files."""
    path = root / f"{split}.csv"
    if not path.exists():
        return []

    examples: list[BertExample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = int(row["label"])
            if label in (1, 2):
                label -= 1
            text = row.get("text", "").replace("\\n", " ").replace('\\"', '"')
            examples.append(BertExample(label=label, tokens=tokenize(text)))
    return examples


def _parse_sst_tree_line(line: str) -> BertExample | None:
    """Parse one SST parenthesized sentiment tree line."""
    segments = line.split(" ")
    if not segments or len(segments[0]) < 2:
        return None
    label = int(segments[0][1])
    if label < 2:
        binary_label = 0
    elif label >= 3:
        binary_label = 1
    else:
        return None

    tokens: list[str] = []
    word_labels: list[int] = []
    for i in range(len(segments) - 1):
        marker = segments[i]
        next_segment = segments[i + 1]
        if (
            marker.startswith("(")
            and len(marker) > 1
            and marker[1] in {"0", "1", "2", "3", "4"}
            and not next_segment.startswith("(")
        ):
            end = next_segment.find(")")
            token = next_segment[:end] if end >= 0 else next_segment
            tokens.append(_normalize_sst_token(token))
            word_labels.append(int(marker[1]))

    return BertExample(label=binary_label, tokens=tokens, word_labels=word_labels)


def _synthetic_examples(dataset: str) -> list[BertExample]:
    """Return deterministic binary sentiment examples for fixture mode."""
    if dataset == "yelp":
        return [
            BertExample(label=1, tokens=["great", "service", "and", "delightful", "food"]),
            BertExample(label=0, tokens=["bad", "service", "and", "awful", "food"]),
            BertExample(label=1, tokens=["excellent", "fresh", "meal"]),
        ]
    return [
        BertExample(label=1, tokens=["a", "great", "movie"]),
        BertExample(label=0, tokens=["a", "bad", "movie"]),
        BertExample(label=1, tokens=["excellent", "acting"]),
    ]


def tokenize(sentence: str) -> list[str]:
    """Tokenize a sentence with NLTK when available, otherwise regex.

    Args:
        sentence: Raw text.

    Returns:
        Token strings.
    """
    try:
        import nltk  # type: ignore[import-untyped]

        return nltk.word_tokenize(sentence)
    except (ImportError, LookupError) as exc:  # nltk or its punkt data not installed
        logger.debug("Falling back to regex tokenization: %s", exc)
        return _TOKEN_RE.findall(sentence)


def _normalize_sst_token(token: str) -> str:
    """Normalize SST bracket placeholder tokens."""
    if token == "-LRB-":
        return "("
    if token == "-RRB-":
        return ")"
    return token


def _sentiment_axis(token: str) -> float:
    """Map known sentiment words to a deterministic classification axis."""
    positive = {"amazing", "delightful", "excellent", "fresh", "good", "great", "love"}
    negative = {"awful", "bad", "boring", "hate", "poor", "terrible", "worst"}
    lowered = token.lower()
    if lowered in positive:
        return 2.0
    if lowered in negative:
        return -2.0
    return 0.0
