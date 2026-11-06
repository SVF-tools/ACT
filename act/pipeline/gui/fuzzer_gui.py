#===- act/pipeline/gui/fuzzer_gui.py - Fuzzing observation and control ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   The ACTFuzzer lifecycle for a GUI, built on the shared ``common_gui`` API:
#     options  - ``fuzzing_options()``: every tunable FuzzingConfig field
#                (``FUZZING_OPTIONS``) as an ``OptionDefault`` with its
#                pipeline.yaml default
#     records  - FuzzStart, FuzzStep (one per fuzzed batch), FuzzConfigChange,
#                FuzzEnd, delivered through the ``common_gui`` channel
#     control  - ``FuzzControl``, a ``RunControl`` over FUZZING_OPTIONS passed
#                to ACTFuzzer: stop, pause/resume, and live config updates,
#                applied between batches
#   Without a sink the loop reads ``common_gui.enabled`` once per batch; without
#   a control it never checks one. Results are unchanged by observation.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple

import torch

from act.pipeline.gui.common_gui import OptionDefault, Record, RunControl

if TYPE_CHECKING:  # ACTFuzzer imports these records, so the fuzzer module loads lazily here
    from act.pipeline.fuzzing.actfuzzer import FuzzingConfig, FuzzingReport


def fuzzing_options(config_path: Optional[str | Path] = None) -> Tuple[OptionDefault, ...]:
    """Every tunable fuzzing option with its pipeline.yaml default."""
    from act.config.config import read_fuzzing_section
    from act.pipeline.fuzzing.actfuzzer import FUZZING_OPTIONS, FuzzingConfig

    section = read_fuzzing_section(config_path)
    defaults = {name: getattr(FuzzingConfig, name, None) for name in FuzzingConfig.__dataclass_fields__}
    return tuple(OptionDefault(o, section.get(o.name, defaults.get(o.name))) for o in FUZZING_OPTIONS)


@dataclass(frozen=True)
class FuzzStart(Record):
    """The run as ACTFuzzer starts it. ``config`` is the resolved FuzzingConfig;
    ``weights`` the normalized strategy selection probabilities."""
    config: FuzzingConfig
    weights: Mapping[str, float]
    batch_size: int
    device: str
    seed_count: int


@dataclass(frozen=True)
class FuzzStep(Record):
    """One ACTFuzzer batch as ACT ran it.

    iteration: samples processed once this batch is done (ACTFuzzer's counter).
    seeds: the selected ``FuzzingSeed`` batch; ``inputs``: its mutated batch [B, ...].
    violations: [B] bool, lanes ACT's property checker accepted as counterexamples.
    counterexamples: the ``Counterexample`` objects added by this batch, in lane order.
    """
    iteration: int
    elapsed_s: float
    strategy: Optional[str]
    seeds: Any
    inputs: torch.Tensor
    violations: torch.Tensor
    counterexamples: Tuple[Any, ...]
    coverage: float
    corpus_size: int


@dataclass(frozen=True)
class FuzzConfigChange(Record):
    """A control update at a batch boundary: ``applied`` (normalized values) when
    accepted, otherwise ``rejected`` holds the reason and nothing changed."""
    iteration: int
    requested: Mapping[str, Any]
    applied: Mapping[str, Any]
    rejected: str


@dataclass(frozen=True)
class FuzzEnd(Record):
    """The final report. stop_reason: "max_iterations" | "timeout" | "first_violation" | "stopped"."""
    report: FuzzingReport
    stop_reason: str


class FuzzControl(RunControl):
    """RunControl of one ACTFuzzer over FUZZING_OPTIONS; requests take effect between batches.
    ACTFuzzer reports each applied update as a FuzzConfigChange."""

    def __init__(self) -> None:
        from act.pipeline.fuzzing.actfuzzer import FUZZING_OPTIONS

        super().__init__(FUZZING_OPTIONS)


__all__: Tuple[str, ...] = (
    "fuzzing_options", "FuzzStart", "FuzzStep", "FuzzConfigChange", "FuzzEnd", "FuzzControl",
)
