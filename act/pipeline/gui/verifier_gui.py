#===- act/pipeline/gui/verifier_gui.py - Propagation observer API --------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Read-only observation of bound propagation, layer by layer and neuron by
#   neuron, while verification runs. Consumers (e.g. ACT-GUI) register a sink
#   with ``observe(sink)``; instrumented engine sites emit records to it.
#
#   Cost model: with no sink registered, an instrumented hot path reads the
#   module attribute ``enabled`` and does nothing else; phase decorators call
#   straight through. Records reference live engine tensors (no copies); they
#   are valid only during the sink call, and a sink copies what it keeps.
#   A sink that raises is detached and logged once; verification continues.
#   ReLU relaxation descriptions are dispatched by domain (``DESCRIBERS``)
#   outside the numerical kernels and run only when a sink asks for them.
#
#   Record identity (``RecordKey``) is shared by every producer so that later
#   producers (BaB nodes, backward refinement) reuse the same model:
#     scope/parent - where the bound holds: "root" for a verify_once run, a
#                    BaB node id later (parent = the node it was derived from)
#     domain       - "interval" | "hybridz" | "dual"
#     pass_        - "forward" | "backward" | "refine"
#     layer_id     - ACT layer id
#     stage        - "out": the layer's output state; "pre": the input state of
#                    an activation layer as the producer used it
#     version      - commit counter per (scope, domain, pass_, layer_id, stage)
#                    within one producer call; worklist revisits increase it
#     origin       - "computed" | "inherited" (copied from a parent scope)
#   Tensors keep ACT's layout: dim 0 is the batch lane, the rest is the
#   layer's flattened or shaped output (neuron i = flat index i in
#   ``layer.out_vars`` order).
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, List, Mapping, Optional, Protocol, Tuple, TypeVar, Union, cast

import torch

logger = logging.getLogger(__name__)

# The only symbol hot paths read. True iff at least one sink is registered.
enabled: bool = False
_SINKS: List["Sink"] = []

DOMAINS = ("interval", "hybridz", "dual")
PASSES = ("forward", "backward", "refine")
STAGES = ("out", "pre")
ORIGINS = ("computed", "inherited")


@dataclass(frozen=True)
class RecordKey:
    domain: str
    pass_: str
    layer_id: int
    stage: str = "out"
    version: int = 0
    scope: str = "root"
    parent: Optional[str] = None
    origin: str = "computed"


@dataclass(frozen=True)
class PhaseEvent:
    """Start/end/error of an instrumented function (``observed_phase``)."""
    phase: str
    status: str  # "start" | "end" | "error"
    t_ns: int
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LayerBounds:
    """Per-neuron box of one layer as committed by the producer. lb/ub: [B, ...]."""
    key: RecordKey
    kind: str
    lb: torch.Tensor
    ub: torch.Tensor
    t_ns: int
    changed: bool = True  # analyze(): False when the worklist re-evaluated the layer without a change


@dataclass(frozen=True)
class Relaxation:
    """How the producer's domain over-approximates one activation layer, per neuron. Tensors [B, n].

    status: int8, 1 = stable active (pre_lb >= 0), 0 = stable inactive (pre_ub <= 0), -1 = unstable.
    form: "box" (interval), "linear" (dual), "exact_split" (hybridz).
    Slopes/intercepts describe y >= lower_slope * x + lower_intercept and y <= upper_slope * x + upper_intercept
    for linear forms; they are None for forms without a linear envelope.
    """
    key: RecordKey
    status: torch.Tensor
    form: str
    lower_slope: Optional[torch.Tensor] = None
    lower_intercept: Optional[torch.Tensor] = None
    upper_slope: Optional[torch.Tensor] = None
    upper_intercept: Optional[torch.Tensor] = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DualBackward:
    """Dual backward pass data at one ReLU layer.

    alpha: lower-envelope slope, [B, M, n] (per spec row) or [B, n]; meaningful for unstable neurons only.
    alpha_optimized: False when no alpha iterations ran (alpha is the initial chord slope u/(u-l)).
    nu: dual variable at this layer, when the producer reports it.
    """
    key: RecordKey
    alpha: Optional[torch.Tensor]
    alpha_optimized: bool
    nu: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class SpecBound:
    """Dual backward result at the output: per lane, per spec row. key.layer_id is the ASSERT layer."""
    key: RecordKey
    margins: torch.Tensor
    slack: torch.Tensor
    iterations: int
    stop_reason: str


Record = Union[PhaseEvent, LayerBounds, Relaxation, DualBackward, SpecBound]


class Sink(Protocol):
    """Receives records. ``wants_relaxation`` (optional attribute, default False) asks producers to
    compute ``Relaxation`` records, which cost extra work and are skipped otherwise."""

    def __call__(self, record: Record) -> None: ...


def now_ns() -> int:
    return time.perf_counter_ns()


def wants_relaxation() -> bool:
    return enabled and any(getattr(s, "wants_relaxation", False) for s in _SINKS)


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
    """Deliver ``record`` to every sink. A sink that raises is detached; verification is never affected."""
    for sink in tuple(_SINKS):
        try:
            with torch.no_grad():
                sink(record)
        except Exception:  # noqa: BLE001 - observer isolation, see module docstring
            logger.warning("act.pipeline.gui.verifier_gui: detaching sink %r after it raised", sink, exc_info=True)
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
            emit(PhaseEvent(name, "start", now_ns()))
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                emit(PhaseEvent(name, "error", now_ns(), {"error": type(exc).__name__}))
                raise
            emit(PhaseEvent(name, "end", now_ns()))
            return result

        return cast(F, wrapper)

    return deco


def relu_status(pre_lb: torch.Tensor, pre_ub: torch.Tensor) -> torch.Tensor:
    """1 stable active, 0 stable inactive, -1 unstable (same shape as the inputs, int8)."""
    status = torch.full_like(pre_lb, -1, dtype=torch.int8)
    status = torch.where(pre_lb >= 0, torch.ones_like(status), status)
    status = torch.where(pre_ub <= 0, torch.zeros_like(status), status)
    return status


# ReLU descriptions per domain. Bounds and alpha tensors are live references;
# backward dual lower slopes may have a specification axis [B, M, n].

def describe_interval(key: RecordKey, pre: Any, post: Any, **_: Any) -> Relaxation:
    return Relaxation(
        key, relu_status(pre.lb, pre.ub), "box",
        extra={"pre_lb": pre.lb, "pre_ub": pre.ub, "lb": post.lb, "ub": post.ub},
    )


def describe_hybridz(key: RecordKey, pre: Any, post: Any, *, tf: Any) -> Relaxation:
    pre, extra, exact = tf._observed_relu
    return Relaxation(
        key, relu_status(pre.lb, pre.ub), "exact_split" if exact else "box",
        extra={"pre_lb": pre.lb, "pre_ub": pre.ub, "lb": post.lb, "ub": post.ub, **extra},
    )


def describe_dual(key: RecordKey, pre: Any, post: Any = None, *,
                  alpha: Optional[torch.Tensor] = None, **_: Any) -> Relaxation:
    lb = pre.lb.flatten(1)
    ub = pre.__dict__.get("_repaired_ub", pre.ub.flatten(1))
    status = relu_status(lb, ub)
    unstable = (lb < 0) & (ub > 0)
    chord = ub / (ub - lb).clamp(min=1e-12)
    upper = torch.where(unstable, chord, (lb >= 0).to(lb))
    upper = torch.where(ub <= 0, torch.zeros_like(upper), upper)
    intercept = torch.where(unstable, -upper * lb, torch.zeros_like(lb))
    lower = upper if alpha is None else alpha
    on, off = lb >= 0, ub <= 0
    if lower.dim() == 3:
        on, off = on.unsqueeze(1), off.unsqueeze(1)
    lower = torch.where(off, torch.zeros_like(lower), torch.where(on, torch.ones_like(lower), lower))
    return Relaxation(
        key, status, "linear", lower, torch.zeros_like(lower), upper, intercept,
        {"pre_lb": lb, "pre_ub": ub},
    )


DESCRIBERS: Mapping[str, Callable[..., Relaxation]] = {
    "interval": describe_interval,
    "hybridz": describe_hybridz,
    "dual": describe_dual,
}


__all__: Tuple[str, ...] = (
    "enabled", "RecordKey", "PhaseEvent", "LayerBounds", "Relaxation", "DualBackward", "SpecBound", "Record",
    "Sink", "observe", "emit", "observed_phase", "wants_relaxation", "relu_status", "now_ns",
    "DESCRIBERS", "DOMAINS", "PASSES", "STAGES", "ORIGINS",
)
