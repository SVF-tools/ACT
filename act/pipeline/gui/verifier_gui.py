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
#   neuron, while verification runs. Records are delivered through the shared
#   ``act_gui`` channel (``observe``/``emit``/``enabled``); instrumented
#   engine sites build them only while a sink is registered. ReLU relaxation
#   descriptions are dispatched by domain (``DESCRIBERS``) outside the
#   numerical kernels and run only when a sink sets ``wants_relaxation``.
#   ``verification_options()`` lists the verify_once settings a GUI may tune,
#   in the same ``Option`` form as ``fuzzer_gui.fuzzing_options()``.
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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple

import torch

from act.pipeline.gui.act_gui import Option, OptionDefault, Record, wants

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
class LayerBounds(Record):
    """Per-neuron box of one layer as committed by the producer. lb/ub: [B, ...]."""
    key: RecordKey
    kind: str
    lb: torch.Tensor
    ub: torch.Tensor
    changed: bool = True  # analyze(): False when the worklist re-evaluated the layer without a change


@dataclass(frozen=True)
class Relaxation(Record):
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
class DualBackward(Record):
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
class SpecBound(Record):
    """Dual backward result at the output: per lane, per spec row. key.layer_id is the ASSERT layer."""
    key: RecordKey
    margins: torch.Tensor
    slack: torch.Tensor
    iterations: int
    stop_reason: str


def wants_relaxation() -> bool:
    return wants("wants_relaxation")


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


def dual_relu_box(pre: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """The [B, n] pre-activation box the dual backward pass used (its repaired upper bound if any)."""
    return pre.lb.flatten(1), pre.__dict__.get("_repaired_ub", pre.ub.flatten(1))


def chord_slope(lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    return ub / (ub - lb).clamp(min=1e-12)


def describe_dual(key: RecordKey, pre: Any, post: Any = None, *,
                  alpha: Optional[torch.Tensor] = None, **_: Any) -> Relaxation:
    lb, ub = dual_relu_box(pre)
    status = relu_status(lb, ub)
    unstable = (lb < 0) & (ub > 0)
    chord = chord_slope(lb, ub)
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


# verify_once keywords; "dual_config.<field>" names a DualConfig field.
VERIFICATION_OPTIONS: Tuple[Option, ...] = (
    Option("dual_iters", "int", False, "alpha optimization iterations; 0 keeps the fixed chord slopes", minimum=0),
    Option("dual_refresh_forward", "bool", False, "recompute forward bounds from the current alpha each iteration"),
    Option("dual_config.lr_alpha", "float", False, "Adam learning rate for alpha", minimum=0.0,
           exclusive_minimum=True),
    Option("dual_config.lr_beta", "float", False, "Adam learning rate for split multipliers", minimum=0.0,
           exclusive_minimum=True),
    Option("dual_config.lr_decay", "float", False, "learning-rate decay per iteration", minimum=0.0,
           maximum=1.0, exclusive_minimum=True),
    Option("dual_config.per_class_alpha", "bool", False, "separate alpha per output class"),
    Option("dual_config.stagnation_patience", "int", False, "non-improving iterations before stopping; 0 disables",
           minimum=0),
    Option("dual_config.stagnation_tol", "float", False, "bound improvement that resets stagnation", minimum=0.0),
    Option("dual_config.stop_when_verified", "bool", False, "stop once every lane is certified"),
    Option("dual_config.max_time", "float", False, "optimization time limit in seconds; 0 disables", minimum=0.0),
    Option("dual_config.forward_lin_max_perturbed", "int", False,
           "perturbed input dims up to which forward bounds stay linear", minimum=0),
    Option("hybridz_tolerance", "float", False, "MILP feasibility and specification tolerance", minimum=0.0,
           exclusive_minimum=True),
)


def verification_options(config_path: Optional[str | Path] = None) -> Tuple[OptionDefault, ...]:
    """Every tunable verify_once option with its default (backend YAML for DualConfig and HybridZ fields)."""
    from act.config.config import BackendConfig

    backend = BackendConfig.from_yaml(config_path)
    defaults: dict[str, Any] = {"dual_iters": 0, "dual_refresh_forward": False,
                                "hybridz_tolerance": backend.hybridz.tolerance}
    defaults.update({f"dual_config.{name}": value for name, value in vars(backend.dual).items()})
    return tuple(OptionDefault(o, defaults[o.name]) for o in VERIFICATION_OPTIONS)


__all__: Tuple[str, ...] = (
    "RecordKey", "LayerBounds", "Relaxation", "DualBackward", "SpecBound", "wants_relaxation", "relu_status",
    "DESCRIBERS", "DOMAINS", "PASSES", "STAGES", "ORIGINS", "VERIFICATION_OPTIONS", "verification_options",
)
