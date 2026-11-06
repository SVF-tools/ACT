#===- act/pipeline/gui/common_gui.py - Shared GUI observation API -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   What verifier_gui and fuzzer_gui share, so both expose one interface:
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
#===---------------------------------------------------------------------===#

from __future__ import annotations

import functools
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple, TypeVar, cast

import torch

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
            logger.warning("act.pipeline.gui.common_gui: detaching sink %r after it raised", sink, exc_info=True)
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


__all__: Tuple[str, ...] = (
    "enabled", "now_ns", "Record", "PhaseEvent", "Sink", "wants", "observe", "emit", "observed_phase",
    "Option", "OptionDefault", "validate_changes", "RunControl",
)
