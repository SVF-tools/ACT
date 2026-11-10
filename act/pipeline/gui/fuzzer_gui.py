#===- act/pipeline/gui/fuzzer_gui.py - Fuzzing observation and control ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   The ACTFuzzer lifecycle for a GUI, built on the shared ``act_gui`` API:
#     options  - ``fuzzing_options()``: every tunable FuzzingConfig field
#                (``FUZZING_OPTIONS``) as an ``OptionDefault`` with its
#                pipeline.yaml default
#     records  - FuzzStart, FuzzStep (one per fuzzed batch), FuzzConfigChange,
#                FuzzEnd, delivered through the ``act_gui`` channel
#     control  - ``FuzzControl``, a ``RunControl`` over FUZZING_OPTIONS passed
#                to ACTFuzzer: stop, pause/resume, and live config updates,
#                applied between batches
#   Without a sink the loop reads ``act_gui.enabled`` once per batch; without
#   a control it never checks one. Results are unchanged by observation.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple

import torch

from act.pipeline.gui.act_gui import OptionDefault, Record, RunControl, observe

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


# Self-check: python -m act.pipeline.gui.fuzzer_gui
def _fuzzer_for(net, control: Optional["FuzzControl"] = None, max_iterations: int = 40) -> "ACTFuzzer":
    import random
    import tempfile
    from pathlib import Path

    import numpy as np

    from act.front_end.spec_creator_base import LabeledInputTensor
    from act.front_end.verifiable_model import InputLayer
    from act.pipeline.fuzzing.actfuzzer import ACTFuzzer, FuzzingConfig
    from act.pipeline.verification.act2torch import ACTToTorch

    model = ACTToTorch(net).run()
    rows = next(c for c in model.children() if isinstance(c, InputLayer)).input_tensor
    seeds = [LabeledInputTensor(tensor=rows[i:i + 1].clone(), label=None) for i in range(rows.shape[0])]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    cfg = FuzzingConfig.from_yaml(max_iterations=max_iterations, timeout_seconds=120.0,
                                  save_counterexamples=False, output_dir=Path(tempfile.mkdtemp()),
                                  report_interval=10_000, verbose=0, trace_level=0)
    return ACTFuzzer(wrapped_model=model, initial_seeds=seeds, config=cfg, control=control)


def _check_fuzzing(factory) -> None:
    import threading
    from typing import Dict, List
    from act.pipeline.gui.self_check_support import Recorder, make_relu_net
    from act.front_end.verifiable_model import VerifiableModel

    control = FuzzControl()
    assert isinstance(control, RunControl)
    try:
        control.update(coverage_strategy="neuron")
    except ValueError:
        pass
    else:
        raise AssertionError("a non-live option changed during a run")

    options = {o.option.name: o for o in fuzzing_options()}
    assert options["perturb_scale"].option.live and not options["coverage_strategy"].option.live
    net = make_relu_net(factory, 0.5)
    VerifiableModel.set_strict_mode(False)

    plain = _fuzzer_for(net).fuzz()
    sink = Recorder()
    with observe(sink):
        observed = _fuzzer_for(net).fuzz()
    assert [ce.input.tolist() for ce in observed.counterexamples] == [ce.input.tolist() for ce in plain.counterexamples]
    steps = [r for r in sink if isinstance(r, FuzzStep)]
    start, end = sink[0], sink[-1]
    assert isinstance(start, FuzzStart) and isinstance(end, FuzzEnd)
    assert end.stop_reason == "max_iterations" and steps[-1].iteration == observed.total_iterations
    assert sum(len(s.counterexamples) for s in steps) == len(observed.counterexamples)

    for bad in ({"coverage_strategy": "BestInputCov"}, {"perturb_scale": 2.0}, {"mutation_weights": {"x": 1}},
                {"max_iterations": 0}, {"stop_on_first_violation": "yes"}, {"pgd_restarts": 1.5}):
        try:
            FuzzControl().update(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")

    # Live update after batch 3, then a paused update applied without running a batch, then stop.
    control, sink = FuzzControl(), Recorder()
    paused, applied = threading.Event(), threading.Event()
    batches: List[FuzzStep] = []

    def drive(record) -> None:
        sink.append(record)
        if isinstance(record, FuzzStep):
            batches.append(record)
            if len(batches) == 3:
                control.update(mutation_weights={"random": 1.0}, perturb_scale=0.3,
                               seed_selection_strategy="random", pgd_restarts=2)
            if len(batches) == 6:
                control.pause()
                paused.set()
        if isinstance(record, FuzzConfigChange) and paused.is_set():
            applied.set()

    def observed_fuzz():
        with observe(drive):
            return fuzzer.fuzz()

    fuzzer = _fuzzer_for(net, control, max_iterations=10_000)
    result: Dict[str, object] = {}
    worker = threading.Thread(target=lambda: result.update(report=observed_fuzz()))
    worker.start()
    assert paused.wait(60)
    control.update(perturb_mode="adaptive_perdim")
    assert applied.wait(60) and control.paused
    batches_while_paused = len(batches)
    control.resume()
    control.request_stop()
    worker.join(60)
    changes = [r for r in sink if isinstance(r, FuzzConfigChange)]
    assert [c.iteration for c in changes] == [batches[2].iteration, batches[5].iteration]
    assert not any(c.rejected for c in changes) and batches_while_paused == 6
    assert fuzzer.stop_reason == "stopped" and result["report"] is not None
    assert {s.strategy for s in batches[3:]} <= {"random"}
    assert fuzzer.mutation_engine.perturb_mode == "adaptive_perdim" and fuzzer.seed_corpus.strategy == "random"
    try:
        fuzzer.update_config(activation_threshold=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-live field changed")


def _self_check() -> None:
    import tempfile
    from act.back_end.transfer_functions import set_solver_mode, set_transfer_function_mode
    from act.pipeline.gui.self_check_support import make_factory
    from act.util.device_manager import initialize_device

    initialize_device("cpu", "float64")
    torch.set_num_threads(1)
    set_solver_mode(None)
    set_transfer_function_mode("interval")
    with tempfile.TemporaryDirectory() as tmp:
        _check_fuzzing(make_factory(tmp))
    print("fuzzer GUI self-check passed")


if __name__ == "__main__":
    # Use the canonical module; ACTFuzzer imports its record classes from there.
    from act.pipeline.gui import fuzzer_gui

    fuzzer_gui._self_check()
