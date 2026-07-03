#===- act/front_end/bert_loader/__init__.py - BERT Loader Exports -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Exposes BERT-style text dataset loading and specification creation for
#   embedding-space verification tasks.
#
#===---------------------------------------------------------------------===#

"""BERT-style text front-end exports for embedding-space verification."""

from __future__ import annotations

from act.front_end.bert_loader.create_specs import BertSpecCreator, create_bert_specs
from act.front_end.bert_loader.certify import (
    CertifiedRadiusResult,
    RadiusSearchOptions,
    RadiusStep,
    certify_radius,
    configure_bab_for_method,
    falsified_counterexample_violates,
    soundness_sample_certified,
)
from act.front_end.bert_loader.data_loader import (
    BertEmbeddingClassifier,
    BertExample,
    BertVocabulary,
    find_bert_dataset_name,
    list_bert_datasets,
    load_bert_dataset,
    sample_correctly_classified,
)

__all__ = [
    "BertEmbeddingClassifier",
    "BertExample",
    "BertSpecCreator",
    "BertVocabulary",
    "CertifiedRadiusResult",
    "RadiusSearchOptions",
    "RadiusStep",
    "certify_radius",
    "configure_bab_for_method",
    "create_bert_specs",
    "falsified_counterexample_violates",
    "soundness_sample_certified",
    "find_bert_dataset_name",
    "list_bert_datasets",
    "load_bert_dataset",
    "sample_correctly_classified",
]
