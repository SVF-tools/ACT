#===- act/pipeline/gui/act_gui.py - Shared GUI observation API ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   ACT-side support for ACT-GUI: observation and control of verification and
#   fuzzing runs. This module is what verifier_gui and fuzzer_gui share, so both
#   expose one interface:
#     channel  - ``observe(sink)`` registers a sink, ``emit(record)`` delivers
#                to every sink, and hot paths read the module attribute
#                ``enabled`` (True iff a sink is registered) before building
#                a record; ``observed_phase`` brackets a function with
#                PhaseEvent start/end/error.
#     records  - every record derives from ``Record`` and carries ``t_ns``
#                (perf_counter_ns at construction).
#     options  - ``Option`` describes one tunable engine setting (kind,
#                admissible values, live flag); ``OptionDefault`` pairs it
#                with its YAML default; ``validate_changes`` normalizes a set
#                of values against an option table.
#     control  - ``RunControl`` stops, pauses/resumes, and queues validated
#                live option changes for a running engine, which applies them
#                at its own safe points.
#
#   Cost model: with no sink registered, an instrumented hot path reads
#   ``enabled`` and does nothing else; phase decorators call straight through.
#   Records reference live engine tensors (no copies); they are valid only
#   during the sink call, and a sink copies what it keeps. A sink that raises
#   is detached and logged once; the engine run continues unchanged.
#
#   Self-check: ``python -m act.pipeline.gui.act_gui`` (run in CI) runs
#   ``verify_once`` in every domain under a sink and drives one ACTFuzzer
#   through its whole lifecycle with a FuzzControl, asserting the records a GUI
#   reads and that observation leaves verification and fuzzing results unchanged.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import functools
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple, TypeVar, cast

import torch

if TYPE_CHECKING:
    from act.pipeline.fuzzing.actfuzzer import ACTFuzzer
    from act.pipeline.gui.fuzzer_gui import FuzzControl

logger = logging.getLogger(__name__)

# The only symbol hot paths read. True iff at least one sink is registered.
enabled: bool = False
_SINKS: List["Sink"] = []


def now_ns() -> int:
    return time.perf_counter_ns()


@dataclass(frozen=True)
class Record:
    """Base of every observation record; ``t_ns`` is stamped at construction."""
    t_ns: int = field(default_factory=now_ns, kw_only=True)


@dataclass(frozen=True)
class PhaseEvent(Record):
    """Start/end/error of an instrumented function (``observed_phase``)."""
    phase: str
    status: str  # "start" | "end" | "error"
    meta: Mapping[str, Any] = field(default_factory=dict)


class Sink(Protocol):
    """Receives records. Optional boolean attributes (e.g. ``wants_relaxation``)
    ask producers for records that cost extra work; see ``wants``."""

    def __call__(self, record: Record) -> None: ...


def wants(flag: str) -> bool:
    """True when a registered sink sets the boolean attribute ``flag``."""
    return enabled and any(getattr(s, flag, False) for s in _SINKS)


def _refresh() -> None:
    global enabled
    enabled = bool(_SINKS)


@contextmanager
def observe(sink: Sink) -> Iterator[Sink]:
    """Register ``sink`` for the duration of the block. Nestable; sinks are called in registration order."""
    _SINKS.append(sink)
    _refresh()
    try:
        yield sink
    finally:
        try:
            _SINKS.remove(sink)
        except ValueError:
            pass  # already detached after an error
        _refresh()


def emit(record: Record) -> None:
    """Deliver ``record`` to every sink. A sink that raises is detached; the engine run is never affected."""
    for sink in tuple(_SINKS):
        try:
            with torch.no_grad():
                sink(record)
        except Exception:  # noqa: BLE001 - observer isolation, see module docstring
            logger.warning("act.pipeline.gui.act_gui: detaching sink %r after it raised", sink, exc_info=True)
            try:
                _SINKS.remove(sink)
            except ValueError:
                pass
            _refresh()


F = TypeVar("F", bound=Callable[..., Any])


def observed_phase(name: str) -> Callable[[F], F]:
    """Emit PhaseEvent start/end/error around a function. Calls straight through when nothing observes."""

    def deco(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enabled:
                return fn(*args, **kwargs)
            emit(PhaseEvent(name, "start"))
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                emit(PhaseEvent(name, "error", {"error": type(exc).__name__}))
                raise
            emit(PhaseEvent(name, "end"))
            return result

        return cast(F, wrapper)

    return deco


@dataclass(frozen=True)
class Option:
    """One tunable engine setting: its value type, admissible values, and
    whether a running engine may change it (``live``).

    kind: "int" | "float" | "bool" | "choice" | "weights" (choice -> weight >= 0).
    """
    name: str
    kind: str
    live: bool
    description: str
    choices: Tuple[str, ...] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    exclusive_minimum: bool = False

    def validate(self, value: Any) -> Any:
        """The value normalized to its kind; ValueError when it is not admissible."""
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{self.name} must be a boolean, got {value!r}")
            return value
        if self.kind == "choice":
            if value not in self.choices:
                raise ValueError(f"{self.name} must be one of {list(self.choices)}, got {value!r}")
            return value
        if self.kind == "weights":
            if not isinstance(value, dict) or set(value) - set(self.choices):
                raise ValueError(f"{self.name} keys must be among {list(self.choices)}, got {value!r}")
            weights = {k: self._number(float, v, f"{self.name}[{k}]") for k, v in value.items()}
            if sum(weights.values()) <= 0.0:
                raise ValueError(f"{self.name} must sum to > 0")
            return weights
        return self._number(int if self.kind == "int" else float, value, self.name)

    def _number(self, kind: type, value: Any, label: str) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (kind is int and value != int(value)):
            raise ValueError(f"{label} must be {'an integer' if kind is int else 'a number'}, got {value!r}")
        number = kind(value)
        low = self.minimum
        if low is not None and (number <= low if self.exclusive_minimum else number < low):
            raise ValueError(f"{label} must be {'>' if self.exclusive_minimum else '>='} {low}, got {number}")
        if self.maximum is not None and number > self.maximum:
            raise ValueError(f"{label} must be <= {self.maximum}, got {number}")
        return number


@dataclass(frozen=True)
class OptionDefault:
    """An option and its default in the engine's YAML configuration."""
    option: Option
    default: Any


def validate_changes(options: Sequence[Option], changes: Mapping[str, Any], *,
                     live_only: bool = False) -> Dict[str, Any]:
    """Normalized ``changes``; ValueError for an unknown, invalid, or (with
    ``live_only``) non-live option. Nothing is returned unless every value passes."""
    table = {option.name: option for option in options}
    checked: Dict[str, Any] = {}
    for name, value in changes.items():
        option = table.get(name)
        if option is None:
            raise ValueError(f"{name} is not an option")
        if live_only and not option.live:
            raise ValueError(f"{name} cannot change during a run")
        checked[name] = option.validate(value)
    return checked


class RunControl:
    """Thread-safe control of one running engine; the engine applies requests at its own safe points.

    ``update`` validates live option changes at once (ValueError for a field that is
    unknown, not live, or invalid) and queues them; later updates of a field win.
    """

    def __init__(self, options: Sequence[Option]) -> None:
        self.options: Tuple[Option, ...] = tuple(options)
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
        checked = validate_changes(self.options, changes, live_only=True)
        with self._cond:
            self._changes.update(checked)
            self._cond.notify_all()
        return checked

    def take_changes(self) -> Dict[str, Any]:
        """Pending updates, cleared (called by the engine at a safe point)."""
        with self._cond:
            changes, self._changes = self._changes, {}
            return changes

    def wait_while_paused(self) -> float:
        """Block until resumed, stopped, or updated; returns the seconds waited."""
        start = time.monotonic()
        with self._cond:
            self._cond.wait_for(lambda: not self._paused or self._stop or bool(self._changes))
        return time.monotonic() - start


# ---------------------------------------------------------------------------
# Self-check (``python -m act.pipeline.gui.act_gui``). Engine imports stay inside
# the checks: verifier, solver, and fuzzer modules import this module at load time.
# ---------------------------------------------------------------------------


class _Recorder(list[object]):
    wants_relaxation = True

    def __call__(self, record) -> None:
        self.append(record)


def _relu_net(factory, bound: float):
    """Active, inactive, unstable, and zero ReLU neurons; outputs must stay in [-bound, bound]."""
    from act.back_end.net_factory import _lt_assert_range, _lt_input

    return factory.create_network(f"gui_relu_{bound}", {"layers": _lt_input([2, 4], -1., 1.) + [
        {"kind": "DENSE", "params": {"in_features": 4, "out_features": 4,
                                     "weight": torch.diag(torch.tensor([1., 1., 1., 0.])),
                                     "bias": torch.tensor([2., -2., 0., 0.])}},
        {"kind": "RELU", "params": {}},
        {"kind": "DENSE", "params": {"in_features": 4, "out_features": 2,
                                     "weight": torch.tensor([[1., -1., 1., -1.], [-1., 1., -1., 1.]]),
                                     "bias": torch.zeros(2)}},
        _lt_assert_range([-bound, -bound], [bound, bound]),
    ]})


def _run(net, domain: str, sink=None, **kwargs):
    from contextlib import nullcontext

    from act.back_end.transfer_functions import set_solver_mode, set_transfer_function_mode
    from act.back_end.verifier import verify_once

    set_transfer_function_mode("hybridz" if domain == "hybridz" else "interval")
    set_solver_mode(domain if domain in ("dual", "hybridz") else None)
    with nullcontext() if sink is None else observe(sink):
        return [r.status for r in verify_once(net, **kwargs)]


def _check_options() -> None:
    from act.config.config import DualConfig
    from act.pipeline.gui import verifier_gui as obs
    from act.pipeline.gui.fuzzer_gui import FuzzControl

    options = {o.option.name: o for o in obs.verification_options()}
    assert not any(o.option.live for o in options.values()) and options["hybridz_tolerance"].default > 0
    dual_fields = {name.split(".", 1)[1] for name in options if name.startswith("dual_config.")}
    assert dual_fields <= set(DualConfig.__dataclass_fields__), dual_fields
    assert validate_changes(obs.VERIFICATION_OPTIONS, {"dual_iters": 2}) == {"dual_iters": 2}
    for bad in ({"dual_iters": -1}, {"unknown": 1}):
        try:
            validate_changes(obs.VERIFICATION_OPTIONS, bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    control = FuzzControl()
    assert isinstance(control, RunControl)
    try:
        control.update(coverage_strategy="neuron")
    except ValueError:
        pass
    else:
        raise AssertionError("a non-live option changed during a run")


def _check_observe(factory) -> None:
    from act.back_end.solver.solver_dual import DualSolver
    from act.back_end.transfer_functions import set_transfer_function_mode
    from act.front_end.specs import OutKind, OutputSpec
    from act.pipeline.gui import verifier_gui as obs

    net = _relu_net(factory, 10.0)
    for domain, form, kwargs in (("interval", "box", {}), ("hybridz", None, {}),
                                 ("dual", "linear", {"dual_iters": 2})):
        sink = _Recorder()
        assert _run(net, domain, sink, **kwargs) == _run(net, domain, **kwargs), domain
        kinds = {type(r).__name__ for r in sink}
        assert {"PhaseEvent", "LayerBounds", "Relaxation"} <= kinds, (domain, kinds)
        relaxations = [r for r in sink if isinstance(r, obs.Relaxation)]
        assert all(r.status.dtype == torch.int8 for r in relaxations)
        assert form is None or any(r.form == form for r in relaxations), domain
        if domain == "dual":
            backward = [r for r in sink if isinstance(r, obs.DualBackward)]
            assert backward and all(r.alpha_optimized for r in backward)
            assert any(isinstance(r, obs.SpecBound) for r in sink)

    # The chunked dual path reports the same per-layer alpha records.
    set_transfer_function_mode("interval")
    sink = _Recorder()
    spec = OutputSpec(kind=OutKind.RANGE, lb=torch.full((2,), -10.0), ub=torch.full((2,), 10.0))
    with observe(sink):
        DualSolver(n_iters=1).evaluate_spec(net, spec, chunk_size=1)
    assert any(isinstance(r, obs.DualBackward) for r in sink)

    def failing(record) -> None:
        raise RuntimeError("sink failure")
    with observe(failing):
        _run(net, "interval")
        assert not enabled, "a failing sink is detached"
    assert not enabled and not obs.wants_relaxation()


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
    from act.front_end.verifiable_model import VerifiableModel
    from act.pipeline.gui.fuzzer_gui import (
        FuzzConfigChange, FuzzControl, FuzzEnd, FuzzStart, FuzzStep, fuzzing_options,
    )

    options = {o.option.name: o for o in fuzzing_options()}
    assert options["perturb_scale"].option.live and not options["coverage_strategy"].option.live
    net = _relu_net(factory, 0.5)
    VerifiableModel.set_strict_mode(False)

    plain = _fuzzer_for(net).fuzz()
    sink = _Recorder()
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
    control, sink = FuzzControl(), _Recorder()
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

    from act.back_end.net_factory import NetFactory
    from act.back_end.transfer_functions import set_solver_mode, set_transfer_function_mode
    from act.util.device_manager import initialize_device

    initialize_device("cpu", "float64")
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        factory = NetFactory({"common": {"output_dir": tmp, "base_seed": 42, "num_instances": 1,
                                         "name_prefix": "gui", "write_manifest": False}})
        try:
            _check_options()
            _check_observe(factory)
            _check_fuzzing(factory)
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")
    print("act.pipeline.gui.act_gui self-check passed")


__all__: Tuple[str, ...] = (
    "enabled", "now_ns", "Record", "PhaseEvent", "Sink", "wants", "observe", "emit", "observed_phase",
    "Option", "OptionDefault", "validate_changes", "RunControl",
)

if __name__ == "__main__":
    # Run in the imported module: under ``-m`` this file is also ``__main__``, and
    # engine code emits to the sinks of act.pipeline.gui.act_gui, not of __main__.
    from act.pipeline.gui import act_gui

    act_gui._self_check()
