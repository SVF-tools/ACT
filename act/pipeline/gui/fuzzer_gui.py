#===- act/pipeline/gui/fuzzer_gui.py - Fuzzing observation and control ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   The ACTFuzzer lifecycle for a GUI, without touching fuzzer internals:
#     options  - ``fuzzing_options()``: every tunable FuzzingConfig field with
#                its pipeline.yaml default, admissible values, and live flag
#     records  - FuzzStart, FuzzStep (one per fuzzed batch), FuzzConfigChange,
#                FuzzEnd, delivered through ``act.pipeline.gui.verifier_gui`` sinks
#     control  - ``FuzzControl`` passed to ACTFuzzer: stop, pause/resume, and
#                live config updates, applied between batches
#   Without a sink the loop reads ``verifier_gui.enabled`` once per batch; without
#   a control it never checks one. Results are unchanged by observation.
#   Record tensors are live references, valid during the sink call.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple

import torch

if TYPE_CHECKING:  # ACTFuzzer imports these records, so the fuzzer module loads lazily here
    from act.pipeline.fuzzing.actfuzzer import FuzzingConfig, FuzzingOption, FuzzingReport


@dataclass(frozen=True)
class FuzzOption:
    """A tunable FuzzingConfig field and its default in the pipeline YAML."""
    option: FuzzingOption
    default: Any


def fuzzing_options(config_path: Optional[str | Path] = None) -> Tuple[FuzzOption, ...]:
    """Every tunable fuzzing option with its pipeline.yaml default."""
    from act.config.config import read_fuzzing_section
    from act.pipeline.fuzzing.actfuzzer import FUZZING_OPTIONS, FuzzingConfig

    section = read_fuzzing_section(config_path)
    defaults = {name: getattr(FuzzingConfig, name, None) for name in FuzzingConfig.__dataclass_fields__}
    return tuple(FuzzOption(o, section.get(o.name, defaults.get(o.name))) for o in FUZZING_OPTIONS)


@dataclass(frozen=True)
class FuzzStart:
    """The run as ACTFuzzer starts it. ``config`` is the resolved FuzzingConfig;
    ``weights`` the normalized strategy selection probabilities."""
    config: FuzzingConfig
    weights: Mapping[str, float]
    batch_size: int
    device: str
    seed_count: int


@dataclass(frozen=True)
class FuzzStep:
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
class FuzzConfigChange:
    """A control update at a batch boundary: ``applied`` (normalized values) when
    accepted, otherwise ``rejected`` holds the reason and nothing changed."""
    iteration: int
    requested: Mapping[str, Any]
    applied: Mapping[str, Any]
    rejected: str


@dataclass(frozen=True)
class FuzzEnd:
    """The final report. stop_reason: "max_iterations" | "timeout" | "first_violation" | "stopped"."""
    report: FuzzingReport
    stop_reason: str


class FuzzControl:
    """Thread-safe run control for one ACTFuzzer; requests take effect between batches.

    ``update`` validates against FUZZING_OPTIONS at once (ValueError for a field
    that is unknown, not live, or invalid) and queues the values; later updates
    of the same field win. ACTFuzzer reports each applied batch as a FuzzConfigChange.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._changes: Dict[str, Any] = {}
        self._paused = False
        self._stop = False

    @property
    def paused(self) -> bool:
        with self._cond:
            return self._paused

    @property
    def stop_requested(self) -> bool:
        with self._cond:
            return self._stop

    def request_stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    def pause(self) -> None:
        with self._cond:
            self._paused = True

    def resume(self) -> None:
        with self._cond:
            self._paused = False
            self._cond.notify_all()

    def update(self, **changes: Any) -> Dict[str, Any]:
        from act.pipeline.fuzzing.actfuzzer import validate_live_changes

        checked = validate_live_changes(changes)
        with self._cond:
            self._changes.update(checked)
            self._cond.notify_all()
        return checked

    def take_changes(self) -> Dict[str, Any]:
        """Pending updates, cleared (called by ACTFuzzer at a batch boundary)."""
        with self._cond:
            changes, self._changes = self._changes, {}
            return changes

    def wait_while_paused(self) -> float:
        """Block until resumed, stopped, or updated; returns the seconds waited."""
        start = time.monotonic()
        with self._cond:
            self._cond.wait_for(lambda: not self._paused or self._stop or bool(self._changes))
        return time.monotonic() - start


__all__: Tuple[str, ...] = (
    "FuzzOption", "fuzzing_options", "FuzzStart", "FuzzStep", "FuzzConfigChange", "FuzzEnd", "FuzzControl",
)
