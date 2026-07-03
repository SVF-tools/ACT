#===- act/front_end/bert_loader/certify.py - BERT Radius Certification ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Provides SST/Yelp method selection helpers and a certified-radius driver
#   that uses ACT's batched BaB verifier as the epsilon decision oracle.
#
#===---------------------------------------------------------------------===#

"""Certified-radius search for embedding-space BERT verification."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from act.back_end.bab.bab import check_violations_batched, verify_bab_batched
from act.back_end.config import BaBConfig, BertMethodSelection, select_bert_method
from act.back_end.core import Net
from act.back_end.solver.solver_base import Solver
from act.front_end.specs import InKind
from act.pipeline.verification.act2torch import ACTToTorch
from act.util.stats import VerifyResult, VerifyStatus


@dataclass(frozen=True)
class RadiusSearchOptions:
    """Options for binary-search certified-radius verification.

    Args:
        method: Public text method selector.
        p: Embedding perturbation norm metadata.
        perturbed_words: Number of token positions perturbed together.
        eps: Initial radius used when callers request one direct check.
        max_eps: Upper end of the binary-search interval.
        num_verify_iters: Number of binary-search iterations.
        k: Rule threshold for rule-slope alpha selection.
        alpha_opt_steps: Optimization steps for optimized-alpha refinement.
        time_budget_s: Per-oracle BaB wall-clock budget.
        max_batch_size: BaB subproblem batch cap.
    """

    method: str = "planar"
    p: float = 2.0
    perturbed_words: int = 1
    eps: float = 1e-5
    max_eps: float = 0.01
    num_verify_iters: int = 5
    k: int = 1
    alpha_opt_steps: int = 1000
    time_budget_s: float = 300.0
    max_batch_size: int | str | None = None

    def selection(self) -> BertMethodSelection:
        """Return the resolved ACT method selector."""
        return select_bert_method(self.method)


@dataclass(frozen=True)
class RadiusStep:
    """One certified-radius oracle decision."""

    iteration: int
    radius: float
    safe: bool
    lower: float
    upper: float
    status: VerifyStatus


@dataclass(frozen=True)
class CertifiedRadiusResult:
    """Certified-radius search result.

    Args:
        method: Canonical method name.
        certified_radius: Largest certified lower endpoint found by search.
        upper_radius: Final uncertified upper endpoint.
        verified: Whether the certified lower endpoint is positive or max_eps
            itself was certified.
        decision: Last oracle result.
        trace: Binary-search trace.
    """

    method: str
    certified_radius: float
    upper_radius: float
    verified: bool
    decision: VerifyResult
    trace: list[RadiusStep] = field(default_factory=list)


def configure_bab_for_method(
    method: str,
    *,
    base_config: BaBConfig | None = None,
    p: float = 2.0,
    perturbed_words: int = 1,
    eps: float = 1e-5,
    max_eps: float = 0.01,
    num_verify_iters: int = 5,
    k: int = 1,
    alpha_opt_steps: int = 1000,
) -> BaBConfig:
    """Create a BaB config with a user-facing text method applied.

    Args:
        method: Public text method name.
        base_config: Existing BaB settings to preserve.
        p: Embedding perturbation norm metadata.
        perturbed_words: Number of token positions perturbed together.
        eps: Initial direct-check radius.
        max_eps: Binary-search upper radius.
        num_verify_iters: Binary-search iterations.
        k: Rule-slope rule threshold.
        alpha_opt_steps: Optimized-alpha optimization steps.

    Returns:
        A copied BaBConfig with method fields and solver tier resolved.
    """
    selection = select_bert_method(method)
    cfg = copy.deepcopy(base_config) if base_config is not None else BaBConfig()
    cfg.method = selection.method
    cfg.baf = selection.baf
    cfg.alpha_mode = selection.alpha_mode
    cfg.solver_tier = selection.solver_tier
    cfg.p = float(p)
    cfg.perturbed_words = int(perturbed_words)
    cfg.eps = float(eps)
    cfg.max_eps = float(max_eps)
    cfg.num_verify_iters = int(num_verify_iters)
    cfg.k = int(k)
    cfg.alpha_opt_steps = int(alpha_opt_steps)
    cfg.dual_n_iters = int(alpha_opt_steps) if selection.alpha_mode == "optimized" else cfg.dual_n_iters
    return cfg


def certify_radius(
    net: Net,
    *,
    solver_factory: Callable[[], Solver],
    options: RadiusSearchOptions,
    config: BaBConfig | None = None,
    perturbed_positions: Sequence[int] | torch.Tensor | None = None,
) -> CertifiedRadiusResult:
    """Binary-search the certified embedding radius using BaB decisions.

    Args:
        net: Single-instance ACT network with an LP_EMBEDDING input spec.
        solver_factory: Factory returning a fresh solver for LP-tier calls.
        options: Radius and method options.
        config: Optional base BaB configuration.
        perturbed_positions: Token positions for this radius query.

    Returns:
        Certified-radius result and decision trace.
    """
    method_cfg = configure_bab_for_method(
        options.method,
        base_config=config,
        p=options.p,
        perturbed_words=options.perturbed_words,
        eps=options.eps,
        max_eps=options.max_eps,
        num_verify_iters=options.num_verify_iters,
        k=options.k,
        alpha_opt_steps=options.alpha_opt_steps,
    )
    lower = 0.0
    upper = float(options.max_eps)
    trace: list[RadiusStep] = []
    last = _run_decision(
        net,
        radius=upper,
        solver_factory=solver_factory,
        config=method_cfg,
        options=options,
        perturbed_positions=perturbed_positions,
    )
    if last.status == VerifyStatus.CERTIFIED:
        return CertifiedRadiusResult(
            method=method_cfg.method or options.selection().method,
            certified_radius=upper,
            upper_radius=upper,
            verified=upper > 0.0,
            decision=last,
            trace=[RadiusStep(0, upper, True, upper, upper, last.status)],
        )

    for iteration in range(1, int(options.num_verify_iters) + 1):
        mid = 0.5 * (lower + upper)
        decision = _run_decision(
            net,
            radius=mid,
            solver_factory=solver_factory,
            config=method_cfg,
            options=options,
            perturbed_positions=perturbed_positions,
        )
        safe = decision.status == VerifyStatus.CERTIFIED
        if safe:
            lower = mid
        else:
            upper = mid
        trace.append(RadiusStep(iteration, mid, safe, lower, upper, decision.status))
        last = decision

    return CertifiedRadiusResult(
        method=method_cfg.method or options.selection().method,
        certified_radius=lower,
        upper_radius=upper,
        verified=lower > 0.0,
        decision=last,
        trace=trace,
    )


def soundness_sample_certified(
    net: Net,
    radius: float,
    *,
    num_samples: int = 128,
    seed: int = 0,
    perturbed_positions: Sequence[int] | torch.Tensor | None = None,
) -> bool:
    """Sample an embedding ball and ensure no concrete property violation.

    Args:
        net: ACT network with ASSERT output spec.
        radius: Radius to sample.
        num_samples: Number of random concrete points.
        seed: Random seed.
        perturbed_positions: Optional token positions to perturb.

    Returns:
        True iff every sample satisfies the output spec.
    """
    query_net = _net_with_radius(net, radius, perturbed_positions=perturbed_positions)
    model = ACTToTorch(query_net).run().eval()
    assert_layer = query_net.layers[-1]
    spec = _embedding_spec_layer(query_net).params
    center = spec["center"]
    p_norm = float(torch.as_tensor(spec.get("p_norm", 2.0)).reshape(-1)[0].item())
    positions = spec.get("perturbed_positions")
    samples = _sample_embedding_region(
        center,
        radius,
        p_norm=p_norm,
        perturbed_positions=positions,
        num_samples=num_samples,
        seed=seed,
    )
    violations = check_violations_batched(model, samples, assert_layer)
    return not bool(violations.any().item())


def falsified_counterexample_violates(net: Net, result: VerifyResult) -> bool:
    """Check that a reported counterexample concretely violates the ASSERT spec."""
    if result.status != VerifyStatus.FALSIFIED or result.counterexample is None:
        return False
    model = ACTToTorch(net).run().eval()
    assert_layer = net.layers[-1]
    input_shape = _input_shape(net)
    ce = result.counterexample.reshape(1, *input_shape[1:]) if len(input_shape) > 1 else result.counterexample.reshape(1, -1)
    violations = check_violations_batched(model, ce, assert_layer)
    return bool(violations[0].item())


def _run_decision(
    net: Net,
    *,
    radius: float,
    solver_factory: Callable[[], Solver],
    config: BaBConfig,
    options: RadiusSearchOptions,
    perturbed_positions: Sequence[int] | torch.Tensor | None,
) -> VerifyResult:
    query_net = _net_with_radius(net, radius, perturbed_positions=perturbed_positions)
    return verify_bab_batched(
        query_net,
        solver_factory=solver_factory,
        config=config,
        max_batch_size=options.max_batch_size,
        time_budget_s=options.time_budget_s,
    )


def _net_with_radius(
    net: Net,
    radius: float,
    *,
    perturbed_positions: Sequence[int] | torch.Tensor | None,
) -> Net:
    query_net = copy.deepcopy(net)
    spec_layer = _embedding_spec_layer(query_net)
    center = spec_layer.params["center"]
    spec_layer.params["eps"] = torch.as_tensor(
        [float(radius)], device=center.device, dtype=center.dtype,
    )
    if perturbed_positions is not None:
        spec_layer.params["perturbed_positions"] = torch.as_tensor(
            perturbed_positions, device=center.device,
        )
    lb, ub = _embedding_box(
        center,
        float(radius),
        spec_layer.params.get("perturbed_positions"),
    )
    spec_layer.params["lb"] = lb
    spec_layer.params["ub"] = ub
    return query_net


def _embedding_spec_layer(net: Net) -> Any:
    for layer in net.layers:
        if layer.kind == "INPUT_SPEC" and layer.params.get("kind") == InKind.LP_EMBEDDING:
            return layer
    raise ValueError("certified-radius search requires an LP_EMBEDDING INPUT_SPEC")


def _input_shape(net: Net) -> tuple[int, ...]:
    for layer in net.layers:
        if layer.kind == "INPUT":
            shape = layer.params.get("shape")
            if not isinstance(shape, (list, tuple)):
                raise ValueError("INPUT layer shape must be a sequence")
            return tuple(int(v) for v in shape)
    raise ValueError("network has no INPUT layer")


def _embedding_box(
    center: torch.Tensor,
    radius: float,
    perturbed_positions: torch.Tensor | Sequence[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = _position_mask(center, perturbed_positions).unsqueeze(-1).expand_as(center)
    eps = torch.as_tensor(radius, device=center.device, dtype=center.dtype)
    return torch.where(mask, center - eps, center), torch.where(mask, center + eps, center)


def _position_mask(
    center: torch.Tensor,
    perturbed_positions: torch.Tensor | Sequence[int] | None,
) -> torch.Tensor:
    from act.front_end.specs import normalize_position_mask

    return normalize_position_mask(
        perturbed_positions,
        int(center.shape[-2]),
        batch_shape=tuple(center.shape[:-2]),
        device=center.device,
    )


def _sample_embedding_region(
    center: torch.Tensor,
    radius: float,
    *,
    p_norm: float,
    perturbed_positions: torch.Tensor | Sequence[int] | None,
    num_samples: int,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device=center.device).manual_seed(seed)
    samples = center.repeat(num_samples, *([1] * (center.dim() - 1)))
    mask = _position_mask(center, perturbed_positions).unsqueeze(-1).expand_as(center)[0]
    if not bool(mask.any().item()) or radius <= 0.0:
        return samples
    dims = int(mask.sum().item())
    direction = torch.randn(num_samples, dims, generator=gen, device=center.device, dtype=center.dtype)
    if p_norm == float("inf"):
        delta = (torch.rand(num_samples, dims, generator=gen, device=center.device, dtype=center.dtype) * 2.0 - 1.0) * radius
    else:
        norm = torch.linalg.vector_norm(direction, ord=p_norm, dim=-1, keepdim=True).clamp(min=1e-12)
        unit = direction / norm
        scale = torch.rand(num_samples, 1, generator=gen, device=center.device, dtype=center.dtype) ** (1.0 / max(dims, 1))
        delta = unit * scale * radius
    flat = samples.reshape(num_samples, -1)
    flat_mask = mask.flatten()
    flat[:, flat_mask] = flat[:, flat_mask] + delta
    return samples
