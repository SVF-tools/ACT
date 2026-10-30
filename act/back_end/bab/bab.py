# ===- act/back_end/bab/bab.py - BaB Verification Engine -----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   BaB loop on a single-spec instance. Subproblems are explored in K-batched
#   waves via solve_batch with CE validation per SAT lane; optional certificate
#   reuse is delegated to ``act.back_end.bab.climb.ClimbSession``. Dual-tier
#   bound policy lives here, concrete CE checks in ``violation``, branching
#   decisions in ``branching/``, and child construction in ``node.py``.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

import logging
import math
import time
from functools import wraps
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Union, cast

import torch

from act.config.config import (
    BaBConfig,
    DualConfig,
    NEURON_BRANCHING_METHODS,
    TOP_K_BOUNDINGS,
    TOP_K_INCOMPATIBLE_BOUNDINGS,
    VALID_BOUNDINGS,
    VALID_SOLVER_TIERS,
)
from act.back_end.bab.node import (
    SubproblemBatch,
    slice_branching_state,
    split_input,
    split_neurons,
    tensor_storage_bytes,
)
from act.back_end.bab.climb import ClimbSession
from act.back_end.bab.frontier_memory import (
    FrontierMemoryPlanner, allow_per_class_alpha, allow_warm_state,
)
from act.back_end.bab.branching.branching import (
    BranchingStrategy,
    SplitDecision,
    _assert_witness_residual_decision_unstable,
    _build_branching_strategy as _build_branching_strategy_impl,
    witness_relu_preactivations,
)
from act.back_end.bab.branching.multi_split import (
    _multi_split_from_decision,
    enumerate_unstable_candidates,
    gain_tested_decision,
    groups_to_tensors,
    lanes_with_relu_candidates,
    presplit_root,
)
from act.back_end.bab.branching.bounding import (
    BoundingStrategy,
    RandomBounding,
    TopKBounding,
    DiverseTopKBounding,
    MCTSBounding,
    OrderFunction,
    DepthLowerBoundOrder,
    GreedyOrder,
    SAOrder,
)

from act.back_end.core import Bounds, Layer, Net
from act.back_end.dual_tf.tf_forward import compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.bab.violation import _check_input_specs_batched, check_violations_batched
from act.back_end.solver.solver_base import Solver, SolveStatus
from act.back_end.solver.solver_dual import (
    DualBatchResult,
    DualSolver,
    _alpha_tree_select_spec_rows,
    _alpha_tree_leaves,
    expand_bounds_dict,
    unproven_spec_rows,
)
from act.back_end.verifier import (
    gather_input_spec_layers,
    get_assert_layer,
    get_input_ids,
    seed_from_input_specs,
    setup_and_solve_batch,
)
from act.front_end.specs import OutKind, OutputSpec
from act.util.stats import VerifyStatus, VerifyResult

log = logging.getLogger(__name__)

_monotonic = time.monotonic


# ---------------------------------------------------------------------------
# Strategy factories
# ---------------------------------------------------------------------------


def _build_branching_strategy(
    method: str, *, dual_solver: Any = None, neuron_only: bool = False
) -> BranchingStrategy:
    return _build_branching_strategy_impl(
        method, dual_solver=dual_solver, neuron_only=neuron_only
    )


def _build_bounding(
    bounding: str,
    *,
    depth_weight: float = 1.0,
    bound_weight: float = 1.0,
    cooling_rate: float = 0.99,
    mcts_exploration: float = 1.0,
    mcts_lambda: float = 0.5,
    mcts_virtual_loss: float = 1.0,
    top_k: int = 0,
) -> BoundingStrategy:
    if bounding not in VALID_BOUNDINGS:
        raise ValueError(
            f"Unknown bounding={bounding!r}. Valid: {VALID_BOUNDINGS}."
        )
    if top_k > 0 and bounding in TOP_K_INCOMPATIBLE_BOUNDINGS:
        raise ValueError(
            f"top_k={top_k} is not supported by bounding={bounding!r}: it does not "
            f"rank the pool by an order function. Use one of "
            f"{TOP_K_BOUNDINGS}, or leave top_k=0 (unbounded)."
        )
    if bounding == "random":
        return RandomBounding()
    if bounding == "diverse_split_signs":
        return DiverseTopKBounding(
            DepthLowerBoundOrder(depth_weight=depth_weight, bound_weight=bound_weight),
            k=top_k,
        )
    # MCTS pins depth_bound_blend: W2 replaces its scoring with UCB1, so the order is moot.
    order_name = "depth_bound_blend" if bounding == "mcts" else bounding
    order: OrderFunction
    if order_name == "depth_bound_blend":
        order = DepthLowerBoundOrder(depth_weight=depth_weight, bound_weight=bound_weight)
    elif order_name == "greedy":
        order = GreedyOrder()
    elif order_name == "annealed":
        order = SAOrder(cooling_rate=cooling_rate)
    else:
        raise ValueError(f"No order registered for bounding {bounding!r}")
    if bounding == "mcts":
        return MCTSBounding(
            order,
            exploration=mcts_exploration,
            lambda_=mcts_lambda,
            virtual_loss=mcts_virtual_loss,
        )
    return TopKBounding(order, k=top_k)


# ---------------------------------------------------------------------------
# BaB engine
# ---------------------------------------------------------------------------


def _net_bound_elements(net: Net) -> int:
    """Total bound-carrying variables; a proxy for per-lane memory cost."""
    return sum(len(l.out_vars) for l in net.layers if l.out_vars)


def _select_spec_rows(
    state: Optional[Dict[int, torch.Tensor]],
    keep_rows: torch.Tensor,
) -> Optional[Dict[int, torch.Tensor]]:
    """Slice the spec axis of per-layer incremental dual state."""
    if state is None:
        return None
    return {
        lid: tensor.index_select(1, keep_rows.to(tensor.device))
        if tensor.dim() >= 3
        else tensor
        for lid, tensor in state.items()
    }


def _neuron_branching_supported(config: BaBConfig) -> bool:
    return (
        config.branching_method in NEURON_BRANCHING_METHODS
        and config.solver_tier in ("dual_alpha", "dual_alpha_eta")
    )


def _witness_residual_branching_active(config: BaBConfig) -> bool:
    return config.branching_method == "witness_residual"


def _keep_branch_lanes(
    branch_batch: SubproblemBatch,
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
    witness_preact: Optional[Dict[int, torch.Tensor]],
    keep: torch.Tensor,
) -> tuple[
    SubproblemBatch,
    Optional[Dict[int, Bounds]],
    Optional[Dict[int, torch.Tensor]],
    Optional[Dict[int, torch.Tensor]],
]:
    """Apply one lane selector to branch state, including spec-packed sensitivities."""
    kept_idx = torch.where(keep.to(branch_batch.lb.device))[0]
    bounds_out, nu_out = slice_branching_state(
        bounds_dict, nu_per_layer, kept_idx, branch_batch.batch_size
    )
    witness_out = (
        None
        if witness_preact is None
        else {
            lid: tensor.index_select(0, kept_idx.to(tensor.device))
            for lid, tensor in witness_preact.items()
        }
    )
    return branch_batch.select(kept_idx), bounds_out, nu_out, witness_out


def _assert_multi_split_invariants(
    stats: Dict[str, int], active_k: Optional[int] = None
) -> None:
    requested = stats["multi_split_k_requested"]
    used = stats["multi_split_k_used"]
    waves = stats["multi_split_wave_count"]
    clamped = stats["multi_split_clamped_wave_count"]
    starved = stats["multi_split_lane_starved_count"]
    counters_ok = waves >= 0 and clamped >= 0 and 0 <= starved <= waves
    metadata_ok = (waves == 0 and used == 1) or (
        waves > 0 and 2 <= used <= requested
    )
    active_ok = active_k is None or 1 <= active_k <= requested
    assert requested >= 1 and counters_ok and metadata_ok and active_ok, (
        "MULTI-SPLIT invariant violated: split depth or wave counters are inconsistent "
        f"(k_requested={requested}, k_used={used}, active_k={active_k}, "
        f"waves={waves}, clamped_waves={clamped}, starved_waves={starved})"
    )


def _terminal_input_axes(batch: SubproblemBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose only cuts whose midpoint is finite and strictly inside the box."""
    midpoint = (batch.lb + batch.ub) / 2
    valid = torch.isfinite(midpoint) & (midpoint > batch.lb) & (midpoint < batch.ub)
    widths = batch.widths().masked_fill(~valid, -torch.inf)
    return valid.any(1), widths.argmax(1)


def _solve_dual_batch(
    *,
    net: Net,
    assert_layer: Layer,
    batched_bounds: Bounds,
    k_actual: int,
    batch: SubproblemBatch,
    config: BaBConfig,
    dual_config: DualConfig,
    optimize: bool,
    keep_rows: Optional[torch.Tensor] = None,
    root_bounds_dict: Optional[Dict[int, Bounds]] = None,
    round_policy: Optional[Any] = None,
    time_cap: Optional[float] = None,
    memory_planner: Optional[FrontierMemoryPlanner] = None,
) -> DualBatchResult:
    """Prepare BaB-specific bounds/state policy, then call ``DualSolver``."""
    solver = DualSolver()
    # A mixed pool may pad absent alpha leaves. Padding is storage, not a
    # certificate or a valid optimizer initialization: solve such waves cold.
    if (batch.warm_state_present is not None
            and not bool(batch.warm_state_present.all())):
        batch.incremental_alpha = None
        batch.incremental_eta = None
    call_started = _monotonic() if time_cap is not None else 0.0

    def _remaining_cap() -> Optional[float]:
        if time_cap is None:
            return None
        return max(0.0, time_cap - (_monotonic() - call_started))

    # Reused root boxes have no lane-matched symbolic frame.
    forward_frame = None
    base_reference_bounds = None
    input_split_child = (
        batch.depths.numel() > 0
        and bool((batch.depths.max() > 0).item())
        and not batch.split_signs
    )
    use_root_dict = root_bounds_dict is not None and not input_split_child
    if use_root_dict:
        assert root_bounds_dict is not None
        bounds_dict = expand_bounds_dict(root_bounds_dict, k_actual)
        lane_box = Bounds(batched_bounds.lb, batched_bounds.ub)
        for layer in net.layers:
            kind = (
                layer.kind.upper()
                if isinstance(layer.kind, str)
                else layer.kind
            )
            if (
                kind in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value)
                and layer.id in bounds_dict
            ):
                bounds_dict[layer.id] = lane_box
        if config.root_bounds_reuse in ("split_refresh", "split_refresh_linear") and batch.split_signs:
            from act.back_end.bab.bound_refresh import refresh_bounds
            base_reference_bounds = bounds_dict
            refreshed = refresh_bounds(
                net, bounds_dict, batch.split_signs,
                linear=config.root_bounds_reuse == "split_refresh_linear",
                support_refinement=config.climb_lin_support_refinement,
                rows_cap=config.per_subproblem_refine_rows_cap,
                forward_lin_max_perturbed=dual_config.forward_lin_max_perturbed,
            )
            bounds_dict = refreshed.bounds
            refine_mode = config.per_subproblem_refine
            refine_rows_cap = config.per_subproblem_refine_rows_cap
            refine_iters = config.per_subproblem_refine_iters
            if round_policy is not None:
                if round_policy.refine_mode is not None:
                    refine_mode = round_policy.refine_mode
                if round_policy.refine_rows_cap is not None:
                    refine_rows_cap = round_policy.refine_rows_cap
                if round_policy.refine_iters is not None:
                    refine_iters = round_policy.refine_iters
            if refine_mode != "none":
                bounds_dict = solver.refine_intermediate_bounds_batched(
                    net,
                    bounds_dict,
                    split_signs=batch.split_signs,
                    mode=refine_mode,
                    rows_cap=refine_rows_cap,
                    optimize_iters=refine_iters,
                    max_tensor_mib=config.intermediate_refine_max_tensor_mib,
                )
    else:
        bounds_dict, forward_frame, _ = compute_forward_bounds(
            net,
            batched_bounds.lb,
            batched_bounds.ub,
            forward_lin_max_perturbed=dual_config.forward_lin_max_perturbed,
        )
    if base_reference_bounds is None:
        base_reference_bounds = bounds_dict

    out_kind = assert_layer.params["kind"]
    if not isinstance(out_kind, str):
        raise TypeError(
            f"ASSERT kind must be str, got {type(out_kind).__name__}"
        )
    out_kind = OutKind(out_kind)
    fields = OutputSpec(kind=out_kind)._gather_rows(
        rows=None,
        batch_size=1,
        device=batched_bounds.lb.device,
        dtype=batched_bounds.lb.dtype,
        shared_ndim={},
        source=assert_layer.params,
        source_batch_size=1,
        drop_singleton_batch=True,
    )
    out_spec = OutputSpec(
        kind=out_kind,
        y_true=fields.get("y_true"),
        margin=fields.get("margin"),
        c=fields.get("c"),
        d=fields.get("d"),
        lb=fields.get("lb"),
        ub=fields.get("ub"),
    )
    eta_enabled = config.solver_tier == "dual_alpha_eta"
    is_child_batch = (
        bool((batch.depths > 0).all().item()) if batch.depths.numel() else False
    )
    child_n_iters = int(config.child_n_iters)
    remaining_cap = _remaining_cap()
    if remaining_cap is not None and remaining_cap <= 0.0:
        raise TimeoutError("BaB dual batch time cap exhausted before solving")
    max_time = dual_config.max_time
    if remaining_cap is not None:
        max_time = (
            remaining_cap
            if dual_config.max_time <= 0.0
            else min(dual_config.max_time, remaining_cap)
        )
    capped_dual_config = replace(
        dual_config,
        max_time=max_time,
        n_iters=(
            child_n_iters
            if optimize and is_child_batch and child_n_iters > 0
            else dual_config.n_iters
        ),
    )
    memory_enabled = batched_bounds.lb.device.type == "cuda" and config.frontier_mem_safety > 0
    if memory_enabled and capped_dual_config.per_class_alpha:
        m_specs = int(keep_rows.numel()) if keep_rows is not None else int(assert_layer.params["M"])
        relu_width = sum(len(layer.out_vars) for layer in net.layers if layer.kind == LayerKind.RELU)
        estimated_alpha = k_actual * m_specs * relu_width * batched_bounds.lb.element_size()
        if not allow_per_class_alpha(estimated_alpha, enabled=True):
            capped_dual_config = replace(capped_dual_config, per_class_alpha=False)
    result = solver.solve_spec_batch(
        net,
        bounds_dict,
        out_spec,
        optimize=optimize,
        dual_config=capped_dual_config,
        input_shape=tuple(batched_bounds.lb.shape[1:]),
        keep_rows=keep_rows,
        split_signs=batch.split_signs if eta_enabled else None,
        eta=batch.incremental_eta if eta_enabled else None,
        incremental_alphas=(
            batch.incremental_alpha
            if dual_config.incremental_start_enabled
            else None
        ),
        incremental_etas=(
            batch.incremental_eta
            if eta_enabled and dual_config.incremental_start_enabled
            else None
        ),
        optimize_alpha=not (
            config.eta_only_children and is_child_batch
        ),
        refresh_forward=not use_root_dict,
        return_nu=_neuron_branching_supported(config),
        reuse_bounds_for_branching=use_root_dict,
        forward_frame=forward_frame,
    )
    result = replace(result, base_reference_bounds=base_reference_bounds)
    if optimize:
        batch.incremental_alpha = result.alpha_state
        assert batch.warm_state_present is not None
        batch.warm_state_present.fill_(result.alpha_state is not None)
        if eta_enabled:
            batch.incremental_eta = result.eta_state
        if memory_planner is not None:
            layout = "per_row" if capped_dual_config.per_class_alpha else "shared"
            memory_planner.alpha_waves[layout] += 1
        if memory_enabled and not allow_warm_state(
            tensor_storage_bytes((
                *_alpha_tree_leaves(batch.incremental_alpha),
                *_alpha_tree_leaves(batch.incremental_eta),
            )), k_actual, enabled=True,
        ):
            batch.incremental_alpha = batch.incremental_eta = None
            batch.warm_state_present.fill_(False)
            result = replace(result, alpha_state=None, eta_state=None)
            if memory_planner is not None:
                memory_planner.warm_drop_count += 1
    return result


def _auto_batch_budget_bytes(safety: float) -> float:
    """Memory the auto sizer may use: min(safety*total, 90% of what this
    process can reclaim), so it shares the GPU with other processes."""
    free, total = torch.cuda.mem_get_info()
    reclaimable = free + torch.cuda.memory_reserved()
    return min(float(total) * safety, float(reclaimable) * 0.9)


def _auto_initial_batch(
    net: Net, config: BaBConfig, budget_bytes: Optional[float] = None,
) -> int:
    """Conservative first batch from net size; the loop recalibrates it from
    the measured per-lane peak after the first real round."""
    safety = float(config.auto_batch_safety)
    cap = int(config.auto_batch_cap)
    floor = int(config.auto_batch_floor)
    per_lane = 4.0 * max(1, _net_bound_elements(net)) * 256.0
    k = int((_auto_batch_budget_bytes(safety) if budget_bytes is None else budget_bytes) / per_lane)
    return max(floor, min(cap, k))


def _auto_recalibrate_batch(
    peak_bytes: float, max_k_seen: int, config: BaBConfig,
    budget_bytes: Optional[float] = None,
) -> int:
    """Batch for the next round = budget / measured bytes-per-lane.

    ``peak_bytes / max_k_seen`` over-estimates the marginal per-lane cost (it
    folds in the one-time root/presolve peak), so the sizer errs toward fewer
    lanes - safe against OOM while still ramping up on small nets with spare
    memory."""
    safety = float(config.auto_batch_safety)
    cap = int(config.auto_batch_cap)
    floor = int(config.auto_batch_floor)
    bpl = max(peak_bytes / max(1, max_k_seen), 1.0)
    k = int((_auto_batch_budget_bytes(safety) if budget_bytes is None else budget_bytes) / bpl)
    return max(floor, min(cap, k))


def _root_oom_guard(function: Callable[..., VerifyResult]) -> Callable[..., VerifyResult]:
    """Root setup has no committed frontier yet; OOM cannot prove its domain."""
    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> VerifyResult:
        """Translate only allocator OOM, leaving all other errors visible."""
        try:
            return function(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            return VerifyResult(VerifyStatus.UNKNOWN, metadata={
                "reason": "root_out_of_memory", "nodes": 0, "iterations": 0,
                "pool_remaining": 1, "oom_failures": 1,
                "any_dropped_frontier_memory": False,
            })
    return guarded


@_root_oom_guard
@torch.no_grad()
def verify_bab_batched(
    net: Net,
    solver_factory: Callable[[], Solver],
    config: Optional[BaBConfig] = None,
    *,
    max_batch_size: Union[int, str],
    time_budget_s: Optional[float] = None,
    dual_config: Optional[DualConfig] = None,
    verbose: bool = False,
    trace: Optional[List[Dict[str, Any]]] = None,
) -> VerifyResult:
    """[BATCHED-API] K-batched Branch-and-Bound verification (single instance).

    Per iteration::

        K       = min(len(pool), max_batch_size, max_nodes - processed)
        batch   = pool.pop(K)                       # [K, D_flat]
        sol     = setup_and_solve_batch(net, [K,*input_shape] bounds, solver_factory())
        # decode per-lane:
        #   UNSAT       -> prune (region certified)
        #   SAT + violation (check_violations_batched) -> FALSIFIED (terminate)
        #   SAT spurious / UNKNOWN -> branch (or drop at max_depth)

    Soundness: returns CERTIFIED only when the pool drains via UNSAT pruning
    with every processed sub-box resolved (``all_resolved_unsat`` and
    ``pool.empty``). If the time/node budget exhausts with unproven sub-boxes
    remaining (branched-then-never-revisited, or dropped at ``max_depth``),
    returns UNKNOWN with
    ``metadata['reason'] == 'budget_exhausted_with_unproven_subboxes'``.

    Args:
        net: ACT network with a single-instance INPUT_SPEC (B=1 seed).
        solver_factory: callable returning a fresh ``Solver`` per iteration
            (no state leakage across iterations).
        config: ``BaBConfig``; ``max_depth`` and ``max_nodes`` cap the search
            tree.
        max_batch_size: required cap on K, an int >= 1 or ``"auto"`` (sizes K
            from GPU memory; ``config.auto_batch_cap`` on CPU). There is no
            implicit default: callers pass the central
            ``BackendConfig.bab_max_batch_size`` or an explicit value.
        time_budget_s: wall-clock budget (default 300 s).
        verbose: reserved.
        trace: if a list is supplied, one dict per iteration is appended with
            keys iter, K, pool_before, pool_after, pruned_unsat, branched,
            falsified_lane, max_depth, spec_rows_kept, elapsed_s.
    """
    run_started = _monotonic()
    if config is None:
        config = BaBConfig()
    budget_s = time_budget_s if time_budget_s is not None else 300.0
    if not math.isfinite(budget_s) or budget_s < 0.0:
        raise ValueError(
            f"time_budget_s must be finite and non-negative, got {budget_s}"
        )
    presolve_budget_s = (
        config.presolve_time_fraction * budget_s
        if config.presolve_time_fraction > 0.0
        else budget_s
    )

    def _elapsed() -> float:
        return _monotonic() - run_started

    def _remaining() -> float:
        return max(0.0, budget_s - _elapsed())

    def _root_time_cap() -> float:
        return max(0.0, min(presolve_budget_s - _elapsed(), _remaining()))

    assert_layer = get_assert_layer(net)
    if dual_config is None:
        dual_config = DualConfig()
    climb = ClimbSession.from_config(
        config, net, cast(str, assert_layer.params["kind"]), dual_config
    )
    auto_batch = isinstance(max_batch_size, str) and max_batch_size == "auto"
    if auto_batch:
        effective_batch = int(config.auto_batch_cap)
    else:
        effective_batch = int(max_batch_size)
    if effective_batch < 1:
        raise ValueError(f"max_batch_size must be >= 1, got {effective_batch}")
    max_k_seen = 0

    fsb_dual_solver = None
    if config.branching_method == "fsb":
        fsb_dual_solver = DualSolver()
    # "gain" measures child bounds directly; its fallback (when no measured
    # decision is available) is BaBSR — it reuses the dual ν scores and
    # degrades to width-based only when ν/bounds are absent, which is strictly
    # better than a random fallback.
    brancher_method = (
        "babsr" if config.branching_method == "gain" else config.branching_method
    )
    brancher = _build_branching_strategy(
        brancher_method, dual_solver=fsb_dual_solver, neuron_only=climb is not None
    )

    multi_split_stats: Dict[str, int] = {
        "multi_split_k_requested": int(config.multi_split_levels),
        "multi_split_k_used": 1,
        "multi_split_wave_count": 0,
        "multi_split_clamped_wave_count": 0,
        "multi_split_lane_starved_count": 0,
    }
    terminal_stats: Dict[str, Any] = {
        "terminal_lp_calls": 0,
        "terminal_lp_lanes": 0,
        "terminal_lp_certified": 0,
        "terminal_input_fallback_lanes": 0,
        "terminal_retired_lanes": 0,
        "terminal_failure_reasons": {},
    } if config.terminal_lp_enabled else {}
    planner: Optional[FrontierMemoryPlanner] = None

    def _branching_metadata() -> Dict[str, Any]:
        _assert_multi_split_invariants(multi_split_stats)
        meta: Dict[str, Any] = dict(multi_split_stats)
        meta.update(terminal_stats)
        if planner is not None:
            meta.update(planner.metadata())
        meta.update(
            any_dropped_max_depth=any_dropped_max_depth,
            dropped_max_depth_nodes=dropped_max_depth_nodes,
            dropped_frontier_cap_nodes=dropped_frontier_cap_nodes,
            any_retired_no_relu_candidate=any_retired_no_relu_candidate,
            any_retired_terminal=any_retired_terminal,
        )
        meta["bounding_top_k_effective"] = int(getattr(pool, "k", 0))
        if _witness_residual_branching_active(config):
            meta["witness_residual_fallback_count"] = int(getattr(brancher, "fallback_count", 0))
            meta["witness_residual_diff_from_babsr_count"] = int(
                getattr(brancher, "different_from_babsr_count", 0)
            )
        if climb is not None:
            meta.update(climb.metadata())
        return meta

    pool = _build_bounding(
        config.bounding,
        depth_weight=config.bounding_depth_weight,
        bound_weight=config.bounding_bound_weight,
        cooling_rate=config.sa_cooling_rate,
        mcts_exploration=config.mcts_exploration,
        mcts_lambda=config.mcts_lambda,
        mcts_virtual_loss=config.mcts_virtual_loss,
        top_k=config.top_k,
    )
    llm_probe: Any = None
    _llm: Any = None
    _wave_index = 0
    if config.llm_probe_enabled:
        from act.pipeline.verification import llm_probe as _llm
        llm_probe = _llm.build_llm_probe(config)

    provenance = bool(config.provenance_enabled or (climb is not None and config.climb_merge_enabled)) or isinstance(
        pool, MCTSBounding
    )
    if provenance and not isinstance(pool, (TopKBounding, MCTSBounding)):
        raise ValueError(
            "provenance_enabled requires a bounding that preserves node_id/parent_id, "
            "not 'random'"
        )
    if isinstance(pool, MCTSBounding) and config.presplit_levels > 0:
        raise ValueError(
            "bounding='mcts' is incompatible with presplit_levels>0: the "
            "pre-split root batch carries no node_id/parent_id provenance"
        )
    node_counter = 0
    fanout = max(2, int(config.input_split_fanout))
    frontier_cap = int(config.frontier_cap)

    spec_layers = gather_input_spec_layers(net)
    root_bounds = seed_from_input_specs(spec_layers)
    planner = FrontierMemoryPlanner(
        root_bounds.lb.device, config.frontier_mem_safety, config.frontier_offload,
        4 * max(1, _net_bound_elements(net)) * 256,
    )
    if auto_batch and root_bounds.lb.device.type == "cuda":
        effective_batch = _auto_initial_batch(
            net, config, planner.snapshot(pool).available if planner.active else None,
        )
    batch_policy = (
        replace(
            config, auto_batch_cap=config.auto_batch_cap if auto_batch else effective_batch,
            auto_batch_floor=config.auto_batch_floor if auto_batch else 1,
        ) if planner.active else config
    )
    if root_bounds.lb.shape[0] != 1:
        raise ValueError("verify_bab_batched requires one root sample; slice the net before BaB")
    input_shape: tuple[int, ...] = tuple(root_bounds.lb.shape[1:])

    per_lane_dim = int(root_bounds.lb[0].numel())
    n_input_vars = len(get_input_ids(net))
    if n_input_vars != per_lane_dim:
        raise ValueError(
            f"verify_bab_batched: INPUT layer declares {n_input_vars} variables "
            f"but the per-lane input dim is {per_lane_dim}. The net was likely "
            f"converted with a batched input shape (B baked into INPUT vars); "
            f"synthesize per-instance (B=1) models before BaB."
        )

    root_batch = SubproblemBatch.from_bounds(root_bounds)
    if climb is not None:
        root_batch.decision_literals = torch.empty((1, 0), dtype=torch.long, device=root_batch.lb.device)
    if provenance:
        n = root_batch.batch_size
        root_batch.node_id = torch.arange(
            node_counter,
            node_counter + n,
            device=root_batch.lb.device,
            dtype=torch.long,
        )
        root_batch.parent_id = torch.full(
            (n,), -1, device=root_batch.lb.device, dtype=torch.long
        )
        node_counter += n

    # Root spec-pruning presolve (ALL-rows kinds, dual tiers): rows certified
    # on the root box stay certified on every sub-box, so descendants only
    # carry the unproven rows.
    spec_keep_rows: Optional[torch.Tensor] = None
    presolve_tier = config.solver_tier
    root_fwd: Optional[Dict[int, Bounds]] = None
    refine_mode = config.intermediate_refine
    reuse_mode = config.root_bounds_reuse
    if presolve_tier in ("dual", "dual_alpha", "dual_alpha_eta") and (
        reuse_mode != "none" or refine_mode != "none"
    ):
        root_fwd, _, _ = compute_forward_bounds(
            net,
            root_bounds.lb,
            root_bounds.ub,
            forward_lin_max_perturbed=dual_config.forward_lin_max_perturbed,
        )
        if refine_mode != "none":
            root_fwd = DualSolver().refine_intermediate_bounds(
                net,
                root_fwd,
                mode=refine_mode,
                blowup_ratio=config.intermediate_refine_ratio,
                max_tensor_mib=config.intermediate_refine_max_tensor_mib,
            )
    # Per-node bound reuse is governed solely by root_bounds_reuse: root_fwd may
    # exist just for the root presolve/refine above, and passing it to descendant
    # solves would freeze every child's intermediate bounds at root tightness
    # (fatal for input-split BaB, where the whole gain comes from recomputing
    # intermediates on the smaller box).
    node_root_fwd: Optional[Dict[int, Bounds]] = (
        root_fwd if reuse_mode != "none" else None
    )
    presolve: Optional[DualBatchResult] = None
    if (
        presolve_tier in ("dual", "dual_alpha", "dual_alpha_eta")
        and assert_layer.params.get("kind") != OutKind.UNSAFE_LINEAR
        and _root_time_cap() > 0.0
    ):
        try:
            presolve = _solve_dual_batch(
                net=net,
                assert_layer=assert_layer,
                batched_bounds=Bounds(root_bounds.lb, root_bounds.ub),
                k_actual=root_batch.batch_size,
                batch=root_batch,
                config=config,
                dual_config=dual_config,
                optimize=presolve_tier in ("dual_alpha", "dual_alpha_eta"),
                root_bounds_dict=root_fwd,
                time_cap=_root_time_cap(),
            )
        except TimeoutError:
            presolve = None
    if presolve is not None:
        presolve_slack = presolve.spec_row_slack()
        if presolve_slack is not None:
            assert presolve.thresholds is not None
            unproven = unproven_spec_rows(
                presolve_slack, assert_layer.params["kind"],
                torch.maximum(presolve.margins.abs(), presolve.thresholds.abs()),
            ).any(dim=0)
            total_rows = int(unproven.numel())
            if not bool(unproven.any().item()):
                return VerifyResult(
                    VerifyStatus.CERTIFIED,
                    metadata={
                        "nodes": root_batch.batch_size,
                        "iterations": 0,
                        "pool_remaining": 0,
                        "spec_rows_total": total_rows,
                        "spec_rows_kept": 0,
                        "resolved_by": "root_presolve",
                        **(climb.metadata() if climb is not None else {}),
                    },
                )
            keep = torch.where(unproven)[0]
            if int(keep.numel()) < total_rows:
                spec_keep_rows = keep
                root_batch.incremental_alpha = _alpha_tree_select_spec_rows(
                    root_batch.incremental_alpha, keep,
                )
                root_batch.incremental_eta = _select_spec_rows(
                    root_batch.incremental_eta, keep,
                )
                root_batch.split_signs = _select_spec_rows(
                    root_batch.split_signs, keep,
                )
        presplit_k = int(config.presplit_levels)
        presolve_bounds = presolve.intermediate_bounds()
        presolve_sensitivities = presolve.neuron_sensitivities()
        if (
            presplit_k > 0
            and root_batch.batch_size == 1
            and presolve_bounds is not None
            and presolve_sensitivities is not None
        ):
            presplit = presplit_root(
                root_batch,
                net,
                presolve_bounds,
                presolve_sensitivities,
                presplit_k,
            )
            if presplit is not None:
                if provenance:
                    assert root_batch.node_id is not None
                    presplit.parent_id = root_batch.node_id.expand(presplit.batch_size).clone()
                    presplit.node_id = torch.arange(node_counter, node_counter + presplit.batch_size, device=presplit.lb.device)
                root_batch = presplit
                node_counter += root_batch.batch_size

    # A completed root solve can replace a second solve after its optimization
    # window expires, but only while its spec axis and lane domain still match.
    if spec_keep_rows is not None or bool((root_batch.depths > 0).any().item()):
        presolve = None

    pool.push(planner.prepare_push(pool, root_batch, allow_evict=True))
    del root_batch
    presplit = presolve_bounds = presolve_sensitivities = None
    any_dropped_frontier_cap = False
    dropped_frontier_cap_nodes = 0
    if frontier_cap > 0 and len(pool) > frontier_cap:
        dropped_frontier_cap_nodes = pool.evict_to(frontier_cap)
        if dropped_frontier_cap_nodes > 0:
            any_dropped_frontier_cap = True

    start = run_started
    processed = 0
    iteration = 0

    def _trace_row(
        k: int,
        pool_before: int,
        pruned_unsat: int,
        branched: int,
        falsified_lane: Optional[int],
        max_depth: int,
    ) -> Dict[str, Any]:
        return {
            "iter": iteration,
            "K": k,
            "pool_before": pool_before,
            "pool_after": len(pool),
            "pruned_unsat": pruned_unsat,
            "branched": branched,
            "falsified_lane": falsified_lane,
            "max_depth": max_depth,
            "spec_rows_kept": (
                int(spec_keep_rows.numel()) if spec_keep_rows is not None else None
            ),
            "elapsed_s": _monotonic() - start,
        }

    any_dropped_max_depth = False
    any_retired_no_relu_candidate = False
    any_retired_terminal = False
    any_oom_exhausted = False
    pending_oom_nodes = 0
    dropped_max_depth_nodes = 0
    pending_children: List[SubproblemBatch] = []
    _last_input_widths: Optional[list[float]] = None

    def _enqueue_children(
        parents: SubproblemBatch, children: SubproblemBatch, parent_index: torch.Tensor,
    ) -> None:
        """Share provenance, frontier caps and CLIMB accounting for every split."""
        nonlocal node_counter, any_dropped_frontier_cap, dropped_frontier_cap_nodes
        if provenance:
            assert parents.node_id is not None
            children.parent_id = parents.node_id.index_select(
                0, parent_index.to(parents.node_id.device),
            )
            nc = children.batch_size
            children.node_id = torch.arange(
                node_counter, node_counter + nc, device=children.lb.device, dtype=torch.long,
            )
            node_counter += nc
        # Commit terminal fallback and ordinary children together. A later
        # allocation failure must never requeue parents of already-pushed rows.
        if planner.active:
            pending_children.append(children)
        else:
            pool.push(children)
            if climb is not None:
                climb.after_children(children.batch_size, len(pool))
            if frontier_cap > 0 and len(pool) > frontier_cap:
                dropped = pool.evict_to(frontier_cap)
                dropped_frontier_cap_nodes += dropped
                any_dropped_frontier_cap |= dropped > 0

    def _retire_lanes(
        keep: torch.Tensor,
        branch_batch: SubproblemBatch,
        bounds_dict: Optional[Dict[int, Bounds]],
        nu_per_layer: Optional[Dict[int, torch.Tensor]],
        witness_preact: Optional[Dict[int, torch.Tensor]],
    ) -> tuple[
        Optional[SubproblemBatch],
        Optional[Dict[int, Bounds]],
        Optional[Dict[int, torch.Tensor]],
        Optional[Dict[int, torch.Tensor]],
    ]:
        """Retire unresolved CLIMB lanes without interpreting them as proofs."""
        nonlocal any_retired_no_relu_candidate
        assert climb is not None
        retired = int((~keep).sum().item())
        if retired == 0:
            return branch_batch, bounds_dict, nu_per_layer, witness_preact
        climb.retire_lanes_without_relu_candidate(retired)
        any_retired_no_relu_candidate = True
        if not bool(keep.any().item()):
            return None, None, None, None
        return _keep_branch_lanes(
            branch_batch, bounds_dict, nu_per_layer, witness_preact, keep
        )

    def _recover_wave() -> bool:
        """Restore uncommitted domains after the failed traceback is released."""
        nonlocal any_dropped_max_depth, any_oom_exhausted, any_retired_no_relu_candidate, any_retired_terminal
        nonlocal checkpoint_tensors, climb_checkpoint, dropped_max_depth_nodes, effective_batch
        nonlocal frontier_checkpoint, node_counter, original_batch, pending_oom_nodes
        nonlocal processed
        restored = original_batch is None
        prepop_relieved = False
        if not wave_committed and planner.active:
            if climb_checkpoint is not None:
                assert climb is not None
                cast(TopKBounding, pool).restore(frontier_checkpoint)
                climb.restore_frontier(climb_checkpoint)
                planner.offloaded, planner.evicted_nodes = memory_checkpoint
                frontier_checkpoint = climb_checkpoint = None
                checkpoint_tensors = ()
            # Completed complement probes retain their consumed node budget
            # and valid learned cores even when the main wave retries.
            node_counter = counter_before
            (any_dropped_max_depth, any_retired_no_relu_candidate,
             any_retired_terminal, dropped_max_depth_nodes) = retirement_before
            terminal_calls = {
                key: terminal_stats[key]
                for key in ("terminal_lp_calls", "terminal_lp_lanes")
                if key in terminal_stats
            }
            terminal_stats.clear()
            terminal_stats.update(terminal_before)
            terminal_stats.update(terminal_calls)
            try:
                if original_batch is None:
                    before_relief = planner.offload_count, planner.warm_drop_count
                    planner._relieve(pool)
                    prepop_relieved = before_relief != (
                        planner.offload_count, planner.warm_drop_count,
                    )
                else:
                    original_batch = original_batch.drop_warm_state()
                    planner._relieve(pool)
                    if planner.offloaded:
                        original_batch = original_batch.to("cpu")
                    pool.push(original_batch)
                    restored = True
            except torch.cuda.OutOfMemoryError:
                # Retain the domain count even if requeue itself cannot
                # allocate. This is an UNKNOWN, never an empty-pool proof.
                if original_batch is not None:
                    pending_oom_nodes = original_batch.batch_size
        if wave_committed and not wave_accounted:
            processed += k_actual
        if trace is not None and len(trace) == trace_before:
            trace.append(_trace_row(
                k_actual or min(effective_batch, _pool_before), _pool_before,
                0, 0, None, batch_max_depth,
            ))
        failed_k = k_actual or k_requested or effective_batch
        if in_prepop_climb and prepop_relieved:
            # Pre-pop CLIMB work gathers the full frontier, independently of K.
            # Retry only after offload or warm-state removal actually helped.
            planner.oom_retries += 1
            return True
        if not (planner.active and restored and not wave_committed
                and not in_prepop_climb and failed_k > 1):
            if original_batch is not None and not restored and not wave_committed:
                pending_oom_nodes = original_batch.batch_size
            planner.oom_failures += 1
            any_oom_exhausted = True
            return False
        effective_batch = max(1, min(effective_batch // 2, failed_k // 2))
        batch_policy.auto_batch_cap = min(batch_policy.auto_batch_cap, effective_batch)
        batch_policy.auto_batch_floor = min(batch_policy.auto_batch_floor, effective_batch)
        planner.oom_retries += 1
        return True

    while not pool.empty:
        elapsed = _monotonic() - start
        if elapsed >= budget_s or processed >= config.max_nodes:
            break

        original_batch: Optional[SubproblemBatch] = None
        in_prepop_climb = False
        wave_committed = False
        wave_accounted = False
        k_actual = 0
        k_requested = min(len(pool), effective_batch, config.max_nodes - processed)
        _pool_before = len(pool)
        batch_max_depth = 0
        iteration_before = iteration
        trace_before = len(trace) if trace is not None else 0
        pending_children.clear()
        # Optimizer output may mutate warm state, never domain geometry.
        counter_before: int = node_counter
        retirement_before: tuple[bool, bool, bool, int] = (
            any_dropped_max_depth, any_retired_no_relu_candidate,
            any_retired_terminal, dropped_max_depth_nodes,
        )
        terminal_before: Dict[str, Any] = dict(terminal_stats) if planner.active else {}
        if planner.active and terminal_stats:
            terminal_before["terminal_failure_reasons"] = dict(terminal_stats["terminal_failure_reasons"])
        frontier_checkpoint: Optional[SubproblemBatch] = None
        climb_checkpoint: Optional[tuple[bool, int, int, int]] = None
        checkpoint_tensors: tuple[torch.Tensor, ...] = ()
        memory_checkpoint: tuple[bool, int] = planner.offloaded, planner.evicted_nodes
        with planner.wave():
            if climb is not None:
                in_prepop_climb = True
                climb.before_pop(cast(TopKBounding, pool))
                in_prepop_climb = False
            if pool.empty:
                break

            remaining_nodes = config.max_nodes - processed
            k_requested = min(len(pool), effective_batch, remaining_nodes)
            if k_requested <= 0:
                break

            iteration += 1
            _wave_t0 = time.time()
            _pool_before = len(pool)
            _wave_policy = None
            _wave_split_used = None
            if llm_probe is not None and _llm is not None:
                _wave_policy = llm_probe.begin_wave(_llm.build_frontier_stats(
                    wave_index=_wave_index,
                    pool_size=len(pool),
                    effective_batch=effective_batch,
                    remaining_nodes=remaining_nodes,
                    elapsed_s=elapsed,
                    remaining_s=max(0.0, budget_s - elapsed),
                    input_widths=_last_input_widths,
                ))
                if _wave_policy.k_requested is not None:
                    k_requested = max(1, min(_wave_policy.k_requested, len(pool), effective_batch, remaining_nodes))

            planned_k = planner.plan_wave(pool, k_requested)
            if planner.active and 0 < planned_k < k_requested:
                effective_batch = planned_k
            k_requested = planned_k
            if not k_requested or pool.empty:
                if pool.empty:
                    if trace is not None:
                        trace.append(_trace_row(0, _pool_before, 0, 0, None, 0))
                    break
                raise torch.cuda.OutOfMemoryError("one wave lane exceeds memory budget")
            original_batch = pool.pop(batch_size=k_requested)
            k_actual = original_batch.batch_size
            batch_max_depth = int(original_batch.depths.max().item()) if trace is not None else 0
            batch = original_batch.to(root_bounds.lb.device) if planner.active else original_batch
            if climb is not None:
                climb.after_pop(len(pool) + k_actual)

            if input_shape:
                k_lb = batch.lb.reshape(k_actual, *input_shape)
                k_ub = batch.ub.reshape(k_actual, *input_shape)
            else:
                k_lb = batch.lb
                k_ub = batch.ub
            batched_bounds = Bounds(k_lb, k_ub)
            wave_start_bytes = planner.begin_bound()

            solver_tier = config.solver_tier
            neuron_branching_supported = _neuron_branching_supported(config)
            bounds_dict_for_branching: Optional[Dict[int, Bounds]] = None
            nu_per_layer_for_branching: Optional[Dict[int, torch.Tensor]] = None
            witness_input_for_branching: Optional[torch.Tensor] = None
            dual_solve_result: Optional[DualBatchResult] = None
            if solver_tier == "lp":
                solver = solver_factory()
                solution = setup_and_solve_batch(
                    net, batched_bounds, solver, timelimit=None,
                )
            elif solver_tier in ("dual", "dual_alpha", "dual_alpha_eta"):
                has_root = bool((batch.depths == 0).any().item())
                optimize = solver_tier != "dual"
                if has_root and presolve is not None and _root_time_cap() <= 0.0:
                    dual_solve_result = presolve
                else:
                    # The root window caps optimization, not search. If setup uses
                    # up that window, a fixed bound still permits root branching.
                    modes = (True, False) if has_root and optimize else (optimize,)
                    for optimize in modes:
                        dual_time_cap = (
                            _root_time_cap() if has_root and optimize else _remaining()
                        )
                        if dual_time_cap <= 0.0:
                            continue
                        try:
                            dual_solve_result = _solve_dual_batch(
                                net=net,
                                assert_layer=assert_layer,
                                batched_bounds=batched_bounds,
                                k_actual=k_actual,
                                batch=batch,
                                config=config,
                                dual_config=dual_config,
                                optimize=optimize,
                                keep_rows=spec_keep_rows,
                                root_bounds_dict=node_root_fwd,
                                round_policy=_wave_policy,
                                time_cap=dual_time_cap,
                                memory_planner=planner,
                            )
                        except TimeoutError:
                            continue
                        break
                presolve = None
                if dual_solve_result is None:
                    pool.push(batch)
                    if trace is not None:
                        trace.append(_trace_row(
                            k_actual, _pool_before, 0, 0, None, batch_max_depth,
                        ))
                    break
                solution = dual_solve_result.lane_solution()
                bounds_dict_for_branching = dual_solve_result.intermediate_bounds()
                nu_per_layer_for_branching = dual_solve_result.neuron_sensitivities()
                witness_input_for_branching = dual_solve_result.lane_witness_inputs()
            else:
                raise ValueError(
                    f"Unknown solver_tier={solver_tier!r}. Valid: {VALID_SOLVER_TIERS}."
                )

            if climb is not None:
                climb.after_main_bound(k_actual)

            n_unsat = sum(1 for s in solution.statuses if s == SolveStatus.UNSAT)
            branched_lanes = 0
            node_lower_bound = (-solution.max_viol).detach()
            if batch.lower_bound is not None:
                # Bound inheritance: a child region is a subset of its parent, so
                # the parent's certified lower bound stays valid; clamping removes
                # per-subproblem optimization regressions (observed: re-optimized
                # children reporting bounds below their parent's).
                node_lower_bound = torch.maximum(
                    node_lower_bound, batch.lower_bound.to(node_lower_bound.device)
                )

            sat_lane_idx = [
                i for i, s in enumerate(solution.statuses) if s == SolveStatus.SAT
            ]
            if sat_lane_idx:
                input_ids = get_input_ids(net)
                input_index = torch.tensor(
                    input_ids, device=solution.x.device, dtype=torch.long,
                )
                sat_idx_t = torch.tensor(
                    sat_lane_idx, device=solution.x.device, dtype=torch.long,
                )
                x_full = solution.x.index_select(0, sat_idx_t)
                x_input_flat = x_full.index_select(1, input_index)
                x_input_shaped = (
                    x_input_flat.reshape(len(sat_lane_idx), *input_shape)
                    if input_shape
                    else x_input_flat
                )
                in_region = _check_input_specs_batched(x_input_shaped, spec_layers)
                violations = check_violations_batched(net, x_input_shaped, assert_layer) & in_region
                for j, lane in enumerate(sat_lane_idx):
                    if bool(violations[j].item()):
                        if trace is not None:
                            trace.append(_trace_row(
                                k_actual, _pool_before, n_unsat, 0, lane, batch_max_depth,
                            ))
                        return VerifyResult(
                            VerifyStatus.FALSIFIED,
                            counterexample=x_input_shaped[j].detach().cpu().clone(),
                            metadata={
                                "nodes": processed + k_actual,
                                "iterations": iteration,
                                "lane": lane,
                                "K": k_actual,
                                "nodes_minted": node_counter,
                                "any_dropped_frontier_cap": any_dropped_frontier_cap,
                                **_branching_metadata(),
                            },
                        )

            unresolved_idx = torch.tensor(
                [i for i, status in enumerate(solution.statuses) if status != SolveStatus.UNSAT],
                device=batch.lb.device,
                dtype=torch.long,
            )
            if climb is not None:
                assert dual_solve_result is not None
                climb.learn(batch, solution.statuses, dual_solve_result, k_actual)
                processed += climb.bound_complements(
                    lambda candidate, cap: _solve_dual_batch(
                        net=net, assert_layer=assert_layer,
                        batched_bounds=Bounds(
                            candidate.lb.reshape(candidate.batch_size, *input_shape),
                            candidate.ub.reshape(candidate.batch_size, *input_shape),
                        ),
                        k_actual=candidate.batch_size, batch=candidate, config=config,
                        dual_config=dual_config, optimize=True, keep_rows=spec_keep_rows,
                        root_bounds_dict=node_root_fwd, time_cap=cap,
                    ),
                    _remaining, min(effective_batch, config.max_nodes - processed - k_actual),
                )

            if int(unresolved_idx.numel()) > 0:
                unresolved = batch.select(unresolved_idx)
                unresolved.lower_bound = node_lower_bound.index_select(
                    0, unresolved_idx.to(node_lower_bound.device)
                )
                if climb is not None:
                    if planner.active:
                        memory_checkpoint = planner.offloaded, planner.evicted_nodes
                        frontier_checkpoint = cast(TopKBounding, pool).checkpoint()
                        climb_checkpoint = climb.checkpoint_frontier()
                        checkpoint_tensors = (
                            tuple(frontier_checkpoint.tensors()) if frontier_checkpoint is not None else ()
                        )
                    unresolved, kept_rows = climb.before_split(cast(TopKBounding, pool), unresolved)
                    unresolved_idx = unresolved_idx.index_select(0, kept_rows)
                if config.terminal_lp_enabled and unresolved.batch_size:
                    from act.back_end.solver.terminal_lp import solve_terminal_batch

                    terminal_bounds, terminal_nu = slice_branching_state(
                        bounds_dict_for_branching, nu_per_layer_for_branching,
                        unresolved_idx, k_actual,
                    )
                    has_candidate = lanes_with_relu_candidates(
                        unresolved, terminal_bounds, terminal_nu,
                    )
                    terminal_rows = torch.where(~has_candidate)[0]
                    if terminal_rows.numel():
                        terminal_batch = unresolved.select(terminal_rows)
                        terminal_count = terminal_batch.batch_size
                        terminal_result = solve_terminal_batch(
                            net, Bounds(
                                terminal_batch.lb.reshape(terminal_count, *input_shape),
                                terminal_batch.ub.reshape(terminal_count, *input_shape),
                            ),
                            terminal_batch.split_signs, deadline=run_started + budget_s,
                        )
                        terminal_stats["terminal_lp_calls"] += 1
                        terminal_stats["terminal_lp_lanes"] += terminal_count
                        proved = torch.tensor(
                            [terminal_result.has_lp_certificate(i) for i in range(terminal_count)],
                            device=batch.lb.device, dtype=torch.bool,
                        )
                        n_unsat += int(proved.sum())
                        terminal_stats["terminal_lp_certified"] += int(proved.sum())
                        for local, status in enumerate(terminal_result.statuses):
                            if status == VerifyStatus.FALSIFIED:
                                lane = int(unresolved_idx[terminal_rows[local]])
                                if trace is not None:
                                    trace.append(_trace_row(
                                        k_actual, _pool_before, n_unsat, branched_lanes,
                                        lane, batch_max_depth,
                                    ))
                                return VerifyResult(
                                    VerifyStatus.FALSIFIED,
                                    counterexample=terminal_result.witnesses[local].detach().cpu().clone(),
                                    metadata={
                                        "nodes": processed + k_actual, "iterations": iteration,
                                        "lane": lane, "K": k_actual, "nodes_minted": node_counter,
                                        "resolved_by": "terminal_lp",
                                        "any_dropped_frontier_cap": any_dropped_frontier_cap,
                                        **_branching_metadata(),
                                    },
                                )
                        if climb is not None and bool(proved.any()):
                            climb.learn_terminal(
                                terminal_batch.select(torch.where(proved)[0]),
                                deadline=run_started + budget_s,
                            )
                        splittable, axes = _terminal_input_axes(terminal_batch)
                        fallback = (~proved & splittable
                                    & (terminal_batch.depths < int(config.max_depth)))
                        if _remaining() <= 0:
                            fallback.fill_(False)
                        fallback_rows = torch.where(fallback)[0]
                        if fallback_rows.numel():
                            fallback_batch = terminal_batch.select(fallback_rows)
                            fallback_children, fallback_parents = split_input(
                                fallback_batch, axes.index_select(0, fallback_rows),
                            )
                            _enqueue_children(fallback_batch, fallback_children, fallback_parents)
                            branched_lanes += fallback_batch.batch_size
                            terminal_stats["terminal_input_fallback_lanes"] += fallback_batch.batch_size
                        for local in torch.where(~proved & ~fallback)[0].tolist():
                            any_retired_terminal = True
                            terminal_stats["terminal_retired_lanes"] += 1
                            failure = terminal_result.reasons[local] or "unproved_terminal"
                            if _remaining() <= 0:
                                failure += ":deadline_exhausted"
                            elif terminal_batch.depths[local] >= int(config.max_depth):
                                failure += ":max_depth"
                            elif not bool(splittable[local]):
                                failure += ":input_unsplittable"
                            failures = terminal_stats["terminal_failure_reasons"]
                            failures[failure] = failures.get(failure, 0) + 1
                            if climb is not None:
                                climb.retire_lanes_without_relu_candidate(1)
                        keep = torch.where(has_candidate)[0]
                        unresolved = unresolved.select(keep)
                        unresolved_idx = unresolved_idx.index_select(0, keep)
                branch_mask = unresolved.depths < int(config.max_depth)
                if bool((~branch_mask).any().item()):
                    any_dropped_max_depth = True
                    dropped_max_depth_nodes += int((~branch_mask).sum().item())
                branch_idx = torch.where(branch_mask)[0]
                if int(branch_idx.numel()) > 0:
                    branch_batch = unresolved.select(branch_idx)
                    children: Optional[SubproblemBatch] = None
                    parent_index = torch.empty(0, dtype=torch.long, device=branch_batch.lb.device)
                    if neuron_branching_supported:
                        full_branch_idx = unresolved_idx.index_select(
                            0, branch_idx.to(unresolved_idx.device)
                        )
                        bd_branch, nu_branch = slice_branching_state(
                            bounds_dict_for_branching,
                            nu_per_layer_for_branching,
                            full_branch_idx,
                            k_actual,
                        )
                        witness_preact_branch: Optional[Dict[int, torch.Tensor]] = None
                        if _witness_residual_branching_active(config) and witness_input_for_branching is not None:
                            witness_branch = witness_input_for_branching.index_select(
                                0,
                                full_branch_idx.to(witness_input_for_branching.device),
                            )
                            witness_preact_branch = witness_relu_preactivations(
                                net,
                                witness_branch,
                                input_shape,
                                dual_config,
                            )
                        multi = None
                        multi_k = int(config.multi_split_levels)
                        if llm_probe is not None and _llm is not None and llm_probe.wants_neuron:
                            # neuron_topk>0 => never bail on candidate count: enumerate the
                            # full set (limit=None) and let advise_neuron_groups truncate to
                            # the top-K by score, so the LLM always decides (with a bounded
                            # view) instead of falling back to FSB. neuron_topk==0 keeps the
                            # legacy "bail to FSB when > max_candidates_total" behavior.
                            _neuron_topk = int(config.llm_probe_neuron_topk)
                            _cand_dicts = enumerate_unstable_candidates(
                                branch_batch, bd_branch, nu_branch,
                                limit=None if _neuron_topk > 0
                                else config.llm_probe_max_candidates_total,
                            )
                            if _cand_dicts:
                                _ngroups = llm_probe.advise_neuron_groups(_llm.build_frontier_stats(
                                    wave_index=_wave_index,
                                    pool_size=len(pool),
                                    effective_batch=effective_batch,
                                    remaining_nodes=remaining_nodes,
                                    elapsed_s=elapsed,
                                    branch_batch_size=branch_batch.batch_size,
                                    candidates=[_llm.CandidateSummary(**_d) for _d in _cand_dicts],
                                ))
                                if _ngroups is not None:
                                    _tl, _tn, _keff = groups_to_tensors(_ngroups, branch_batch)
                                    if _tl is not None and _tn is not None:
                                        multi = split_neurons(branch_batch, net, _tl, _tn, _keff)
                                        _wave_split_used = _keff
                        # Joint splitting is orthogonal to how a split is SCORED, so
                        # it is no longer keyed on branching_method == "gain": the k
                        # neurons come from the BaBSR heuristic inside
                        # _multi_split_from_decision either way. The enclosing
                        # neuron_branching_supported guard already restricts this to a
                        # neuron-branching method on a dual_alpha* tier.
                        if multi is None and multi_k > 1:
                            # Adaptive split depth: fan out so children roughly
                            # fill one bounding batch; n_branch lanes x 2^k <=
                            # max_batch_size keeps the frontier from flooding
                            # the pool. Note this needs
                            # effective_batch >= 4 * branch_batch.batch_size before
                            # k_adaptive can exceed 1 at all, so joint splitting
                            # stays dormant on waves where most lanes are
                            # unresolved. The clamp is a memory guard and must stay
                            # (forcing exact k risks OOM); the user's lever is
                            # --bab-max-batch-size.
                            k_adaptive = max(
                                1,
                                min(
                                    multi_k,
                                    int(math.log2(max(2, effective_batch // max(1, branch_batch.batch_size)))),
                                ),
                            )
                            if _wave_policy is not None and _wave_policy.split_k is not None and _llm is not None:
                                k_adaptive = _llm.clip_split_k(
                                    _wave_policy.split_k,
                                    branch_batch_size=branch_batch.batch_size,
                                    effective_batch=effective_batch,
                                    multi_split_levels=multi_k,
                                )
                            if k_adaptive < multi_k:
                                if multi_split_stats["multi_split_clamped_wave_count"] == 0:
                                    log.warning(
                                        "joint multi-split clamped: requested k=%d but "
                                        "effective_batch=%d / branch lanes=%d allows only "
                                        "k=%d (needs effective_batch >= %d for k=%d); "
                                        "raise --bab-max-batch-size to lift this",
                                        multi_k, effective_batch, branch_batch.batch_size,
                                        k_adaptive, branch_batch.batch_size * (2 ** multi_k),
                                        multi_k,
                                    )
                                multi_split_stats["multi_split_clamped_wave_count"] += 1
                            _wave_split_used = k_adaptive
                            if k_adaptive > 1:
                                multi = _multi_split_from_decision(
                                    branch_batch, net, bd_branch, nu_branch, k_adaptive,
                                    steer=climb.steer_scores if climb is not None and config.climb_steer else None,
                                )
                                if multi is not None:
                                    multi_split_stats["multi_split_wave_count"] += 1
                                    multi_split_stats["multi_split_k_used"] = k_adaptive
                                    if multi[0].batch_size != branch_batch.batch_size * (2 ** k_adaptive):
                                        multi_split_stats["multi_split_lane_starved_count"] += 1
                        if multi is not None:
                            _assert_multi_split_invariants(
                                multi_split_stats, active_k=_wave_split_used
                            )
                            children, parent_index = multi
                        else:
                            decision = None
                            if climb is not None and config.branching_method == "gain":
                                branch_batch, bd_branch, nu_branch, witness_preact_branch = _retire_lanes(
                                    lanes_with_relu_candidates(branch_batch, bd_branch, nu_branch),
                                    branch_batch, bd_branch, nu_branch, witness_preact_branch,
                                )
                            if branch_batch is not None and config.branching_method == "gain":
                                decision = gain_tested_decision(
                                    branch_batch,
                                    net,
                                    assert_layer,
                                    config,
                                    dual_config,
                                    spec_keep_rows,
                                    node_root_fwd,
                                    bd_branch,
                                    nu_branch,
                                    input_shape,
                                    solve_dual=_solve_dual_batch,
                                    steer=climb.steer_scores if climb is not None and config.climb_steer else None,
                                )
                            if branch_batch is not None and decision is None:
                                extra_branch_kwargs = (
                                    {"witness_preact_per_layer": witness_preact_branch}
                                    if _witness_residual_branching_active(config)
                                    else {}
                                )
                                scores = cast(Any, brancher).compute_scores(
                                    branch_batch,
                                    net,
                                    bounds_dict=bd_branch,
                                    nu_per_layer=nu_branch,
                                    **extra_branch_kwargs,
                                )
                                if climb is not None:
                                    scores = climb.steer_scores(scores)
                                decision = cast(SplitDecision, cast(Any, brancher).select(scores))
                                if climb is not None and decision.lane_mask is not None:
                                    branch_batch, bd_branch, nu_branch, witness_preact_branch = _retire_lanes(
                                        decision.lane_mask,
                                        branch_batch, bd_branch, nu_branch, witness_preact_branch,
                                    )
                                if branch_batch is not None and _witness_residual_branching_active(config):
                                    _assert_witness_residual_decision_unstable(
                                        decision, bd_branch
                                    )
                            if branch_batch is not None and decision is not None:
                                if decision.kind == "input_axis":
                                    decision.fanout = fanout
                                children, parent_index = decision.apply(branch_batch, net)
                    else:
                        scores = brancher.compute_scores(branch_batch, net)
                        legacy_decision = cast(Any, brancher).select(scores)
                        if isinstance(legacy_decision, SplitDecision):
                            decision = legacy_decision
                            split_dims = decision.input_axes(branch_batch)
                        else:
                            split_dims = torch.as_tensor(
                                legacy_decision,
                                device=branch_batch.lb.device,
                                dtype=torch.long,
                            ).reshape(-1)
                            decision = SplitDecision(
                                kind="input_axis",
                                input_axis=split_dims,
                                fanout=fanout,
                            )
                        widths = branch_batch.widths()
                        _last_input_widths = (
                            widths.mean(dim=0).tolist() if widths.shape[1] <= 32 else None
                        )
                        if _wave_policy is not None and _wave_policy.input_split_dim is not None:
                            # LLM-advised input dimension (already range-clipped). Lanes where
                            # the advised dim has zero width keep the brancher's choice:
                            # splitting a zero-width dim yields identical children (livelock).
                            advised = torch.full_like(split_dims, int(_wave_policy.input_split_dim))
                            has_width = widths.gather(1, advised.unsqueeze(1)).squeeze(1) > 0
                            split_dims = torch.where(has_width, advised, split_dims)
                        decision.input_axis = split_dims
                        if _wave_policy is not None and _wave_policy.input_split_fanout is not None:
                            decision.fanout = int(_wave_policy.input_split_fanout)
                        children, parent_index = decision.apply(branch_batch, net)

                    if children is not None:
                        assert branch_batch is not None
                        branched_lanes += branch_batch.batch_size
                        _enqueue_children(branch_batch, children, parent_index)

            if pending_children:
                incoming = pending_children[0]
                for extra_children in pending_children[1:]:
                    # Both sources and the concat destination coexist at peak.
                    sources = tuple(incoming.tensors()) + tuple(extra_children.tensors())
                    temporary = sum(t.numel() * t.element_size() for t in sources)
                    if not planner.snapshot(
                        pool, incoming, (*batch.tensors(), *extra_children.tensors(), *checkpoint_tensors),
                        temporary,
                    ).fits:
                        raise torch.cuda.OutOfMemoryError("staged children concatenation budget")
                    incoming = incoming.concat(extra_children)
                incoming = planner.prepare_push(
                    pool, incoming, (*batch.tensors(), *checkpoint_tensors),
                    allow_evict=effective_batch == 1,
                )
                pool.push(incoming)
                wave_committed = True
                if climb is not None:
                    climb.after_children(incoming.batch_size, len(pool))
                pending_children.clear()
            if frontier_cap > 0 and len(pool) > frontier_cap:
                dropped = pool.evict_to(frontier_cap)
                dropped_frontier_cap_nodes += dropped
                any_dropped_frontier_cap |= dropped > 0

            processed += k_actual
            wave_accounted = True
            wave_committed = True
            if isinstance(pool, MCTSBounding):
                # Only a committed, concretely checked wave earns a visit/reward.
                assert batch.node_id is not None
                n_unstable = max(1, sum(
                    int(((b.lb < 0) & (b.ub > 0)).sum().item())
                    for b in (bounds_dict_for_branching or {}).values()
                ) // k_actual)
                pool.observe(
                    batch.node_id, node_lower_bound, solution.statuses,
                    batch.depths, n_unstable,
                )

            if trace is not None:
                trace.append(_trace_row(
                    k_actual, _pool_before, n_unsat, branched_lanes, None, batch_max_depth,
                ))

            if isinstance(pool, MCTSBounding):
                log.info(
                    "mcts: nodes=%d n_tot=%d frontier N[parent] histogram=%s",
                    processed,
                    pool.n_tot,
                    pool.frontier_parent_visit_histogram(),
                )

            if planner.active:
                planner.observe_wave(
                    torch.cuda.max_memory_allocated(planner.device) - wave_start_bytes, k_actual,
                )
            if (auto_batch or planner.active) and root_bounds.lb.device.type == "cuda":
                max_k_seen = max(max_k_seen, k_actual)
                effective_batch = _auto_recalibrate_batch(
                    (planner.lane_bytes * max_k_seen if planner.active
                     else torch.cuda.max_memory_allocated(root_bounds.lb.device)),
                    max_k_seen, batch_policy,
                    budget_bytes=(planner.snapshot(pool).available if planner.active else None),
                )

            if llm_probe is not None and _llm is not None:
                llm_probe.end_wave(_llm.WaveOutcome(
                    wave_index=_wave_index,
                    pool_before=_pool_before,
                    pool_after=len(pool),
                    k_requested_used=k_actual,
                    split_k_used=_wave_split_used if _wave_split_used is not None else 1,
                    refine_iters_used=(_wave_policy.refine_iters if (_wave_policy is not None and _wave_policy.refine_iters is not None) else 0),
                    certified_count=0,
                    falsified_found=False,
                    branched_count=0,
                    best_lb_before=None,
                    best_lb_after=None,
                    wave_time_s=time.time() - _wave_t0,
                    fallback_used=False,
                ))
                _wave_index += 1

            if planner.active:
                # Finished-wave tensors must not masquerade as outside memory
                # when planning the next wave.
                original_batch = batch = batched_bounds = dual_solve_result = solution = None
                bounds_dict_for_branching = nu_per_layer_for_branching = None
                witness_input_for_branching = unresolved = branch_batch = children = None
                terminal_batch = fallback_batch = fallback_children = incoming = None
                extra_children = sources = terminal_result = None
                bd_branch = nu_branch = witness_preact_branch = terminal_bounds = terminal_nu = None
                multi = witness_branch = x_full = x_input_flat = x_input_shaped = None
                k_lb = k_ub = node_lower_bound = None
                frontier_checkpoint = climb_checkpoint = None
                checkpoint_tensors = ()

        if planner.wave_oom:
            # Leave the exception handler before allocating during recovery:
            # its traceback otherwise retains the failing solver's work tensors.
            # The pool commits each mutation atomically. No children are pushed
            # before the whole active-planner wave succeeds, so the exact
            # original rows are still a complete cover of unfinished work.
            iteration = max(iteration, iteration_before + 1)
            pending_children.clear()
            batch = batched_bounds = dual_solve_result = solution = None
            bounds_dict_for_branching = nu_per_layer_for_branching = None
            witness_input_for_branching = unresolved = branch_batch = children = None
            terminal_batch = fallback_batch = fallback_children = incoming = None
            extra_children = sources = terminal_result = None
            bd_branch = nu_branch = witness_preact_branch = terminal_bounds = terminal_nu = None
            multi = witness_branch = x_full = x_input_flat = x_input_shaped = None
            k_lb = k_ub = node_lower_bound = None
            presolve = None
            if not _recover_wave():
                break
    pool_remaining = len(pool) + pending_oom_nodes
    elapsed_total = _monotonic() - start
    exhausted_time = elapsed_total >= budget_s
    exhausted_nodes = processed >= config.max_nodes

    spec_rows_kept = (
        int(spec_keep_rows.numel()) if spec_keep_rows is not None else None
    )

    if (
        not any_dropped_max_depth
        and not any_dropped_frontier_cap
        and not any_retired_no_relu_candidate
        and not any_retired_terminal
        and not any_oom_exhausted
        and planner.evicted_nodes == 0
        and pool_remaining == 0
    ):
        return VerifyResult(
            VerifyStatus.CERTIFIED,
            metadata={
                "nodes": processed,
                "spec_rows_kept": spec_rows_kept,
                "pool_remaining": 0,
                "iterations": iteration,
                "exhausted_budget_time": exhausted_time,
                "exhausted_budget_nodes": exhausted_nodes,
                "nodes_minted": node_counter,
                "any_dropped_frontier_cap": any_dropped_frontier_cap,
                **_branching_metadata(),
            },
        )

    return VerifyResult(
        VerifyStatus.UNKNOWN,
        metadata={
            "nodes": processed,
            "spec_rows_kept": spec_rows_kept,
            "pool_remaining": pool_remaining,
            "iterations": iteration,
            "exhausted_budget_time": exhausted_time,
            "exhausted_budget_nodes": exhausted_nodes,
            "nodes_minted": node_counter,
            "any_dropped_frontier_cap": any_dropped_frontier_cap,
            "unrequeued_oom_nodes": pending_oom_nodes,
            "reason": (
                "budget_exhausted_with_unproven_subboxes"
                if pool_remaining > 0 and (exhausted_time or exhausted_nodes)
                else "out_of_memory"
                if any_oom_exhausted
                else "frontier_memory_eviction"
                if planner.evicted_nodes
                else "frontier_cap_eviction"
                if any_dropped_frontier_cap
                else "terminal_lp_unproved_retirement"
                if any_retired_terminal
                else "climb_retired_without_relu_candidate"
                if any_retired_no_relu_candidate
                else "max_depth_unproved_retirement"
                if any_dropped_max_depth
                else "budget_exhausted_with_unproven_subboxes"
            ),
            **_branching_metadata(),
        },
    )


@torch.no_grad()
def verify_bab(
    net: Net,
    solver: Solver,
    config: Optional[BaBConfig] = None,
    *,
    max_depth: Optional[int] = None,
    max_nodes: Optional[int] = None,
    max_subproblems: Optional[int] = None,
    time_budget_s: Optional[float] = None,
    timelimit: Optional[float] = None,
    verbose: bool = False,
    dual_config: Optional[DualConfig] = None,
    trace: Optional[List[Dict[str, Any]]] = None,
) -> VerifyResult:
    """Single-solver Branch-and-Bound entry: one subproblem per iteration.

    Thin wrapper over ``verify_bab_batched`` with K=1. Constructs a solver factory
    from the supplied solver instance's type so each BaB iteration gets a fresh
    instance. Prefer ``verify_bab_batched`` directly for batched (K>1) solving.
    """
    if config is None:
        config = BaBConfig(
            max_depth=max_depth if max_depth is not None else 20,
            max_nodes=(max_nodes or max_subproblems or 2000),
            verbose=verbose,
        )
    budget = (
        time_budget_s if time_budget_s is not None
        else (timelimit if timelimit is not None else 300.0)
    )
    solver_tier = config.solver_tier
    if solver_tier not in VALID_SOLVER_TIERS:
        raise ValueError(
            f"Unknown solver_tier={solver_tier!r}. Valid: {VALID_SOLVER_TIERS}."
        )
    solver_type = type(solver)
    return verify_bab_batched(
        net=net,
        solver_factory=lambda: solver_type(),
        config=config,
        max_batch_size=1,
        time_budget_s=budget,
        verbose=verbose,
        dual_config=dual_config,
        trace=trace,
    )


if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    from act.back_end.net_factory import LAYER_TESTING_SPECS, NetFactory
    from act.config.config import BackendConfig
    from act.util.device_manager import get_default_dtype, initialize_device

    initialize_device("cpu", "float64")
    torch.set_num_threads(1)
    _ = torch.manual_seed(42)
    with TemporaryDirectory(prefix="act-bab-shared-") as directory:
        factory = NetFactory(BackendConfig.from_yaml().generation.net_factory,
                             output_dir=directory, write_manifest=False)
        upper = torch.ones(1, 1, dtype=get_default_dtype())
        from act.back_end.bab.frontier_memory import (
            MemoryBudget, PER_CLASS_ALPHA_MAX_BYTES, WARM_STATE_MAX_BYTES_PER_LANE,
        )
        from act.back_end.solver.solver_dual import MixedAlpha
        from dataclasses import fields

        # Statistics visit backing storage, not copies or materialized views.
        backing = torch.arange(48, dtype=upper.dtype).reshape(4, 12)
        assert tensor_storage_bytes((backing[:1, :2], backing[2:, 3:])) == backing.untyped_storage().nbytes()
        transport = SubproblemBatch(
            lb=backing[:2, :2], ub=backing[:2, :2] + 1,
            depths=torch.tensor([1, 2]), node_id=torch.tensor([1, 2]),
            parent_id=torch.tensor([-1, -1]), decision_literals=torch.tensor([[1, 0], [2, -1]]),
            parent_margins=upper.new_tensor([.2, .3]), lower_bound=upper.new_tensor([-.1, .4]),
            incremental_alpha={2: {"nested": [upper.new_full((2, 3, 2), .5)]}},
            incremental_eta={2: upper.new_zeros((2, 3, 2))},
            split_signs={2: upper.new_ones((2, 3, 2))},
        )
        pool_factories = (
            RandomBounding, lambda: TopKBounding(SAOrder()),
            lambda: DiverseTopKBounding(SAOrder()), lambda: MCTSBounding(SAOrder()),
        )
        for make_pool in pool_factories:
            state_pool = make_pool()
            state_pool.push(transport)
            with (
                patch.object(SubproblemBatch, "select", side_effect=AssertionError("statistics allocated")),
                patch.object(TopKBounding, "view_all", side_effect=AssertionError("statistics gathered")),
            ):
                assert state_pool.storage_bytes() == tensor_storage_bytes(state_pool.resident_tensors())
                assert state_pool.projected_push_bytes(transport) > 0
            state_pool.to("cpu")
            transported = state_pool.pop(2)
            assert transported.node_id is not None
            transported = transported.select(torch.argsort(transported.node_id))
            for field in fields(transport):
                expected, actual = getattr(transport, field.name), getattr(transported, field.name)
                if isinstance(expected, torch.Tensor):
                    assert torch.equal(expected, actual), (type(state_pool), field.name)
                elif isinstance(expected, dict):
                    assert all(torch.equal(a, b) for a, b in zip(
                        _alpha_tree_leaves(expected), _alpha_tree_leaves(actual),
                    ))
            state_pool.push(transported)
            state_pool.drop_warm_state()
            dropped_state = state_pool.pop(2)
            assert dropped_state.incremental_alpha is dropped_state.incremental_eta is None
            assert dropped_state.warm_state_present is not None
            assert not bool(dropped_state.warm_state_present.any())
            assert dropped_state.split_signs is not None
            state_pool.push(dropped_state)
            assert state_pool.evict_to(0) == 2 and state_pool.empty

            # Failure on a late allocation leaves payload and scheduling intact.
            for operation in ("push", "pop"):
                atomic_pool = make_pool()
                atomic_pool.push(transport)
                resident_ids = tuple(id(t) for t in atomic_pool.resident_tensors())
                parents_before = dict(atomic_pool.parent) if isinstance(atomic_pool, MCTSBounding) else None
                schedule_before = vars(atomic_pool.order).copy() if isinstance(atomic_pool, TopKBounding) else None
                allocs = [0]
                allocator = torch.cat if operation == "push" else torch.Tensor.index_select

                def fail_late(*args: Any, **kwargs: Any) -> torch.Tensor:
                    """Fail after earlier output tensors have already allocated."""
                    allocs[0] += 1
                    if allocs[0] == 4:
                        raise torch.cuda.OutOfMemoryError("injected late allocation")
                    return allocator(*args, **kwargs)

                target, attribute = (torch, "cat") if operation == "push" else (torch.Tensor, "index_select")
                try:
                    with patch.object(target, attribute, fail_late):
                        if operation == "push":
                            atomic_pool.push(replace(transport, node_id=torch.tensor([3, 4])))
                        else:
                            atomic_pool.pop(1)
                except torch.cuda.OutOfMemoryError:
                    pass
                else:
                    raise AssertionError((type(atomic_pool), operation, allocs))
                assert tuple(id(t) for t in atomic_pool.resident_tensors()) == resident_ids
                assert len(atomic_pool) == 2
                if isinstance(atomic_pool, MCTSBounding):
                    assert atomic_pool.parent == parents_before
                if isinstance(atomic_pool, TopKBounding):
                    assert vars(atomic_pool.order) == schedule_before

        nested_warm = transport.select(torch.tensor([0]))
        nested_cold = replace(
            transport.select(torch.tensor([1])),
            incremental_alpha={2: {"nested": [upper.new_full((1, 3, 2), .6)], "missing": upper}},
            incremental_eta=None, split_signs=None, warm_state_present=torch.tensor([True]),
        )
        missing = nested_warm.concat(nested_cold)
        assert missing.warm_state_present is not None
        assert missing.warm_state_present.tolist() == [False, True]
        assert missing.incremental_eta is not None and not bool(missing.incremental_eta[2][1].any())
        assert missing.split_signs is not None and not bool(missing.split_signs[2][1].any())
        assert allow_per_class_alpha(PER_CLASS_ALPHA_MAX_BYTES, enabled=True)
        assert not allow_per_class_alpha(PER_CLASS_ALPHA_MAX_BYTES + 1, enabled=True)
        assert allow_per_class_alpha(PER_CLASS_ALPHA_MAX_BYTES + 1, enabled=False)
        assert allow_warm_state(2 * WARM_STATE_MAX_BYTES_PER_LANE, 2, enabled=True)
        assert not allow_warm_state(2 * WARM_STATE_MAX_BYTES_PER_LANE + 1, 2, enabled=True)

        inactive = FrontierMemoryPlanner(upper.device, .6, True, 100)
        inactive_pool = TopKBounding()
        with patch("torch.cuda.mem_get_info", side_effect=AssertionError("CPU queried CUDA")):
            assert inactive.plan_wave(inactive_pool, 4) == 4
            assert inactive.prepare_push(inactive_pool, transport) is transport
            assert inactive.begin_bound() == 0
        calibration_pool = TopKBounding()
        calibration_pool.push(SubproblemBatch.from_bounds(Bounds(
            -upper.expand(16, -1), upper.expand(16, -1),
        )))
        calibrated = FrontierMemoryPlanner(upper.device, .6, False, 10000)
        calibrated.active = True
        allocator_peak = [10_000_000]  # old root peak is not marginal wave memory
        with (
            patch("torch.cuda.memory_allocated", return_value=2000),
            patch("torch.cuda.reset_peak_memory_stats",
                  side_effect=lambda *_: allocator_peak.__setitem__(0, 2000)),
        ):
            bound_start = calibrated.begin_bound()
        allocator_peak[0] = max(allocator_peak[0], 2200)
        calibrated.observe_wave(allocator_peak[0] - bound_start, 2)
        batch_limit = BaBConfig(auto_batch_floor=1, auto_batch_cap=16)
        expected_k = _auto_recalibrate_batch(200, 2, batch_limit, budget_bytes=1000)
        with patch.object(calibrated, "snapshot", return_value=MemoryBudget(1000, 0, 0, 0, 0, 0)):
            assert calibrated.plan_wave(calibration_pool, 16) == expected_k == 10
            calibrated.observe_wave(40, 2)
            assert calibrated.plan_wave(calibration_pool, 16) == 10
            calibrated.observe_wave(600, 3)
            assert calibrated.plan_wave(calibration_pool, 16) == 5
        # Script pressure by byte state, not timing or actual accelerator use.
        pressure_pool = TopKBounding()
        pressure_pool.push(transport)
        simulated = FrontierMemoryPlanner(upper.device, .6, False, 100)
        simulated.active = True
        def pressure_snapshot(*args: Any, **kwargs: Any) -> MemoryBudget:
            """Warm storage is relieved before shrinking the requested wave."""
            resident_pressure = pressure_pool._resident_batch()
            assert resident_pressure is not None
            warm = resident_pressure.incremental_alpha is not None
            return MemoryBudget(250 if not warm else 50, 0, 0, 0, 0, 0)
        with patch.object(simulated, "snapshot", side_effect=pressure_snapshot):
            assert simulated.plan_wave(pressure_pool, 4) == 2
        assert simulated.warm_drop_count == simulated.wave_shrinks == 1
        assert simulated.evicted_nodes == 0 and len(pressure_pool) == 2
        with patch.object(simulated, "snapshot", side_effect=lambda *_args, **_kw:
                          MemoryBudget(100, 0, 100 * len(pressure_pool), 0, 0, 0)):
            assert simulated.plan_wave(pressure_pool, 1) == 1
        assert simulated.evicted_nodes == 2 and pressure_pool.empty
        offloader = FrontierMemoryPlanner(upper.device, .6, True, 100)
        offloader.active = True
        pressure_pool.push(transport)
        with (
            patch.object(offloader, "_host_available", return_value=10**9),
            patch.object(offloader, "snapshot", side_effect=lambda *_args, **_kw:
                         MemoryBudget(1000 if offloader.offloaded else 0, 0, 0, 0, 0, 0)),
        ):
            assert offloader.plan_wave(pressure_pool, 4) == 4
        assert offloader.offload_count == 1 and offloader.warm_drop_count == 0
        byte_probe = FrontierMemoryPlanner(upper.device, .6, False, 100)
        owned_bytes = tensor_storage_bytes((*pressure_pool.resident_tensors(), *transport.tensors()))
        with (
            patch("torch.cuda.mem_get_info", side_effect=[(5000, 10000), (1000, 10000)]),
            patch("torch.cuda.memory_reserved", return_value=2000),
            patch("torch.cuda.memory_allocated", return_value=owned_bytes + 500),
        ):
            first_budget = byte_probe.snapshot(pressure_pool, transport, (transport.lb,), temporary=77)
            changed_budget = byte_probe.snapshot(pressure_pool, transport, (transport.lb,), temporary=77)
        assert first_budget.outside == changed_budget.outside == 500
        assert first_budget.inflight == 0 and first_budget.temporary == 77
        assert first_budget.limit == 6000 and changed_budget.limit == 2700
        assert byte_probe.pop_temporary(pressure_pool) >= sum(
            t.numel() * t.element_size() for t in pressure_pool.resident_tensors()
        )
        byte_probe.active = True
        with patch.object(byte_probe, "snapshot", side_effect=lambda pool, incoming=None, inflight=(), temporary=0:
                          MemoryBudget(100000, 0, len(pool) * 100000, 0, 0, temporary)):
            try:
                byte_probe.prepare_push(pressure_pool, transport)
            except torch.cuda.OutOfMemoryError:
                pass
            else:
                raise AssertionError("push peak ignored before K backoff")
            assert len(pressure_pool) == 2 and byte_probe.evicted_nodes == 0
            byte_probe.prepare_push(pressure_pool, transport, allow_evict=True)
        assert pressure_pool.empty and byte_probe.evicted_nodes == 2

        # Two revived parents must stay genuinely cold when K=1 leaves one
        # pending while the first parent's real solve pushes warm children.

        cold_net = factory.create_network(
            "merge_cold_lifecycle", LAYER_TESTING_SPECS["layer_testing_climb_reuse"]()
        )
        cold_config = BaBConfig(
            climb_enabled=True, solver_tier="dual_alpha_eta", branching_method="babsr",
            bounding="depth_bound_blend", root_bounds_reuse="plain",
            climb_merge_enabled=True, climb_presplit_propagation_enabled=False,
        )
        cold_session = ClimbSession(cold_config, cold_net, OutKind.LINEAR_LE)
        cold_lid = cold_session.codec.layer_ids[0]
        cold_box = seed_from_input_specs(gather_input_spec_layers(cold_net))
        cold_siblings = SubproblemBatch(
            lb=cold_box.lb.expand(3, -1).clone(), ub=cold_box.ub.expand(3, -1).clone(),
            depths=torch.full((3,), 2, dtype=torch.long),
            split_signs={cold_lid: upper.new_tensor([
                [[-1., 1.]], [[-1., -1.]], [[1., -1.]],
            ])},
            decision_literals=torch.tensor([[-1, 2], [-1, -2], [1, -2]]),
            node_id=torch.tensor([11, 12, 21]), parent_id=torch.tensor([10, 10, 20]),
        )
        cold_pool = TopKBounding()
        cold_pool.push(cold_siblings)
        cold_session._insert_cores(torch.tensor([[1, 2]]))
        cold_session.before_pop(cold_pool)
        assert cold_session.metrics.merged_pairs == 1
        cold_active = cold_pool.pop(1)
        assert cold_active.node_id is not None
        cold_base, _, _ = compute_forward_bounds(cold_net, cold_active.lb, cold_active.ub)
        cold_result = _solve_dual_batch(
            net=cold_net, assert_layer=cold_net.layers[-1],
            batched_bounds=Bounds(cold_active.lb, cold_active.ub),
            k_actual=1, batch=cold_active, config=cold_config,
            dual_config=DualConfig(n_iters=0), optimize=True, root_bounds_dict=cold_base,
        )
        assert cold_result.solution.statuses == (SolveStatus.SAT,)
        warm_children, child_parents = split_neurons(
            cold_active, cold_net, torch.tensor([[cold_lid]]), torch.tensor([[0]]), 1,
        )
        assert warm_children.incremental_alpha is not None
        warm_children.node_id = torch.tensor([31, 32])
        warm_children.parent_id = cold_active.node_id[child_parents]
        cold_pool.push(warm_children)
        cold_pool.replace_all(cold_pool.view_all())
        # A merged lane stays cold beside warm lanes; zero padding is never a warm start.
        frontier = cold_pool.view_all()
        assert frontier.node_id is not None and frontier.warm_state_present is not None
        assert frontier.warm_state_present.tolist() == (frontier.node_id >= 31).tolist()
        cold_pool.pop(len(cold_pool))
        cold_pool.push(warm_children)
        assert cold_pool.pop().incremental_alpha is not None
        from act.back_end.bab.branching.multi_split import _collect_neuron_candidates

        steering_spec = LAYER_TESTING_SPECS["layer_testing_climb_reuse"]()
        steering_spec["layers"][2]["params"].update(
            out_features=4, weight=upper.new_tensor([[1., 0.], [1., 0.], [1., 0.], [0., 0.]]),
            bias=upper.new_tensor([0., 0., 0., 2.]),
        )
        steering_spec["layers"][4]["params"].update(
            in_features=4, weight=upper.new_tensor([[1., -1., 0., 0.]]),
        )
        steering_net = factory.create_network("steering_consumers", steering_spec)
        steering_box = seed_from_input_specs(gather_input_spec_layers(steering_net))
        scoring_batch = SubproblemBatch.from_bounds(steering_box)
        scoring_bounds, _, _ = compute_forward_bounds(steering_net, scoring_batch.lb, scoring_batch.ub)
        consumer = ClimbSession(BaBConfig(climb_steer=True), steering_net, OutKind.LINEAR_LE)
        score_lid = consumer.codec.layer_ids[0]
        consumer.library.insert(torch.tensor([[1]]))
        scoring_nu = {score_lid: upper.new_tensor([[[.2, .4, .6, 100.]]])}
        for steer, selected in ((None, 2), (consumer.steer_scores, 0)):
            candidates = _collect_neuron_candidates(scoring_batch, scoring_bounds, scoring_nu, steer)
            assert candidates is not None and torch.isneginf(candidates[0][0, 3])
            decision = gain_tested_decision(
                scoring_batch, steering_net, steering_net.layers[-1],
                BaBConfig(solver_tier="dual_alpha_eta"), DualConfig(n_iters=0),
                None, scoring_bounds, scoring_bounds, scoring_nu, (2,),
                _solve_dual_batch, n_candidates=1, steer=steer,
            )
            assert decision is not None and decision.neuron_idx is not None
            assert decision.neuron_idx.item() == selected
            joint = _multi_split_from_decision(
                scoring_batch, steering_net, scoring_bounds, scoring_nu, 2, steer,
            )
            assert joint is not None and joint[0].split_signs is not None
            chosen = joint[0].split_signs[score_lid].ne(0).any(0).any(0)
            assert torch.equal(chosen, torch.tensor(
                [steer is not None, steer is None, True, False], device=chosen.device,
            ))
        assert scoring_batch.split_signs is None
        assert torch.equal(scoring_batch.lb, steering_box.lb)
        assert torch.equal(scoring_batch.ub, steering_box.ub)
        # Closed unsafe sets own the tie. Exercise both ALL and EXISTS row
        # reductions, finite proofs and the actual interval/dual/BaB surfaces.
        from act.back_end.solver.solver_dual import certification_tolerance
        from act.back_end.transfer_functions import set_solver_mode
        from act.back_end.verifier import verify_once

        for precision in ("float32", "float64"):
            initialize_device("cpu", precision)
            ref = torch.ones(2, 2, dtype=get_default_dtype())
            band = certification_tolerance(ref)
            for kind in (
                OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST,
                OutKind.UNSAFE_LINEAR, OutKind.LINEAR_LE, OutKind.RANGE,
            ):
                closed = kind in OutputSpec.CLOSED_KINDS
                for slack, expected in (
                    (ref * 0, not closed), (band * .5, not closed),
                    (band * 2, True), (-band, False),
                    (ref * torch.nan, False), (ref * torch.inf, False),
                    (-ref * torch.inf, False),
                ):
                    assert bool((~unproven_spec_rows(slack, kind, ref) == expected).all())
                mixed = band * 2
                mixed[0, 0] = -1
                rows = ~unproven_spec_rows(mixed, kind, ref)
                reduced = rows.any(1) if kind == OutKind.UNSAFE_LINEAR else rows.all(1)
                assert reduced.tolist() == [kind == OutKind.UNSAFE_LINEAR, True]
            tie_spec = LAYER_TESTING_SPECS["layer_testing_climb_reuse"]()
            tie_spec["layers"] = tie_spec["layers"][:3] + [{
                "kind": "ASSERT", "params": {"kind": "TOP1_ROBUST", "y_true": 0},
            }]
            tie_net = factory.create_network(f"tie_{precision}", tie_spec)
            tie_bounds = seed_from_input_specs(gather_input_spec_layers(tie_net))
            tie_forward, _, _ = compute_forward_bounds(
                tie_net, tie_bounds.lb, tie_bounds.ub
            )
            for mode in ("interval", "dual"):
                set_solver_mode(mode)
                assert verify_once(tie_net)[0].status != VerifyStatus.CERTIFIED
            assert not bool(DualSolver().evaluate_spec(
                tie_net, OutputSpec(OutKind.TOP1_ROBUST, y_true=torch.tensor([0])),
                num_classes=2, bounds_dict=tie_forward,
            ).certified.any())
            verdict_tie = verify_bab_batched(
                tie_net, DualSolver,
                BaBConfig(solver_tier="dual_alpha", max_depth=1, max_nodes=3),
                max_batch_size=2, time_budget_s=30, dual_config=DualConfig(n_iters=1),
            )
            assert verdict_tie.status != VerifyStatus.CERTIFIED
            equality = LAYER_TESTING_SPECS["layer_testing_climb_admission"]()
            equality["layers"][-1]["params"].update(c=[0.], d=0.)
            equality_net = factory.create_network(f"equality_{precision}", equality)
            for mode in ("interval", "dual"):
                set_solver_mode(mode)
                assert verify_once(equality_net)[0].status == VerifyStatus.CERTIFIED
        initialize_device("cpu", "float64")
        set_solver_mode("dual")
        # Root symbolic frames are intentionally not reused for child boxes:
        # their coordinates may have reset. Finite-p geometry stays original.
        lp_net = factory.create_network(
            "replay_geometry", LAYER_TESTING_SPECS["layer_testing_lp_dense"]()
        )
        lp_box = seed_from_input_specs(gather_input_spec_layers(lp_net))
        lp_root, lp_frame, _ = compute_forward_bounds(lp_net, lp_box.lb, lp_box.ub)
        assert lp_frame is not None and lp_net.input_geometry is not None
        saved_radii = lp_net.input_geometry.radii.clone()
        half = SubproblemBatch.from_bounds(lp_box).concat(SubproblemBatch.from_bounds(lp_box))
        half.lb[0, 0] = 0
        half.ub[1, 0] = 0
        half.depths += 1
        # Retained zero signs must not trick root reuse into changing the balls.
        half.split_signs = {}
        geometry_result: Optional[torch.Tensor] = None
        for reuse in ("none", "plain", "split_refresh", "split_refresh_linear"):
            lp_result = _solve_dual_batch(
                net=lp_net, assert_layer=get_assert_layer(lp_net),
                batched_bounds=Bounds(half.lb.reshape(2, 2, 2), half.ub.reshape(2, 2, 2)),
                k_actual=2, batch=half, config=BaBConfig(root_bounds_reuse=reuse),
                dual_config=DualConfig(n_iters=0), optimize=False,
                root_bounds_dict=lp_root if reuse != "none" else None,
            )
            assert torch.allclose(lp_result.lower_bounds, half.lb.new_full((2,), .25))
            assert torch.equal(lp_net.input_geometry.radii, saved_radii)
            if reuse == "none":
                geometry_result = lp_result.lower_bounds
            else:
                assert geometry_result is not None
                assert torch.equal(lp_result.lower_bounds, geometry_result)
        # Shared [B,n] alpha must broadcast child eta/signs over both spec rows.
        spec = LAYER_TESTING_SPECS["layer_testing_climb_reuse"]()
        spec["layers"][-1]["params"].update(c=[[1.], [-1.]], d=[.1, .1])
        net = factory.create_network("shared_multirow", spec)
        layout_spec = LAYER_TESTING_SPECS["layer_testing_climb_reuse"]()
        layout_spec["layers"][-1]["params"].update(c=[[1.], [-1.], [.5]], d=[.1, .1, .2])
        layout_net = factory.create_network("mixed_three_rows", layout_spec)
        layout_box = seed_from_input_specs(gather_input_spec_layers(layout_net))
        layout_lid = next(layer.id for layer in layout_net.layers if layer.kind == LayerKind.RELU)
        shared_lane = SubproblemBatch(
            lb=layout_box.lb, ub=layout_box.ub, depths=torch.ones(1, dtype=torch.long),
            incremental_alpha={layout_lid: upper.new_full((1, 2), .25)},
            incremental_eta={layout_lid: upper.new_zeros((1, 1, 2))},
            split_signs={layout_lid: upper.new_zeros((1, 1, 2))},
        )
        per_row_lane = replace(
            shared_lane, incremental_alpha={layout_lid: upper.new_tensor([[[.3, .6], [.4, .7], [.5, .8]]])},
            incremental_eta={layout_lid: upper.new_zeros((1, 3, 2))},
            split_signs={layout_lid: upper.new_zeros((1, 3, 2))},
        )
        mixed_pool = TopKBounding()
        mixed_pool.push(shared_lane)
        mixed_pool.push(per_row_lane)
        mixed_pool.to("cpu")
        mixed_layout = mixed_pool.pop(2)
        assert mixed_layout.incremental_alpha is not None
        assert mixed_layout.warm_state_present is not None
        assert mixed_layout.split_signs is not None and mixed_layout.incremental_eta is not None
        assert isinstance(mixed_layout.incremental_alpha[layout_lid], MixedAlpha)
        assert tensor_storage_bytes(_alpha_tree_leaves(mixed_layout.incremental_alpha)) == (
            8 * upper.element_size() + 2 * 8
        )
        assert bool(mixed_layout.warm_state_present.all())
        assert mixed_layout.split_signs[layout_lid].shape == mixed_layout.incremental_eta[layout_lid].shape == (2, 3, 2)
        layout_config = BaBConfig(solver_tier="dual_alpha_eta")
        layout_dual = DualConfig(n_iters=0, incremental_start_enabled=True)
        def solve_layout(candidate: SubproblemBatch) -> DualBatchResult:
            """Run the real mixed-layout and genuine cold-start consumers."""
            return _solve_dual_batch(
                net=layout_net, assert_layer=get_assert_layer(layout_net),
                batched_bounds=Bounds(candidate.lb, candidate.ub),
                k_actual=candidate.batch_size, batch=candidate,
                config=layout_config, dual_config=layout_dual, optimize=True,
            )
        independent = [solve_layout(shared_lane), solve_layout(per_row_lane)]
        layout_result = solve_layout(mixed_layout)
        assert torch.allclose(layout_result.margins, torch.cat([r.margins for r in independent]))
        assert layout_result.m_specs == 3 and bool(mixed_layout.warm_state_present.all())
        assert layout_result.reference_bounds is not None
        refine_bounds = layout_result.reference_bounds
        refine_solver = DualSolver()
        uncapped = refine_solver.refine_intermediate_bounds_batched(
            layout_net, refine_bounds, mode="all", rows_cap=2, lane_chunk=2,
        )
        two_queries_mib = 8 * sum(
            box.lb[0].numel() * box.lb.element_size() for box in refine_bounds.values()
        ) / 1024**2
        with patch.object(DualSolver, "compute_certified_bound", autospec=True,
                          side_effect=DualSolver.compute_certified_bound) as refine_spy:
            capped = refine_solver.refine_intermediate_bounds_batched(
                layout_net, refine_bounds, mode="all", rows_cap=2, lane_chunk=2,
                max_tensor_mib=two_queries_mib,
            )
        assert len(refine_spy.call_args_list) >= 2
        assert all(torch.equal(capped[lid].lb, box.lb) and torch.equal(capped[lid].ub, box.ub)
                   for lid, box in uncapped.items())
        with patch.object(DualSolver, "compute_certified_bound",
                          side_effect=AssertionError("unaffordable optional refinement")):
            skipped = refine_solver.refine_intermediate_bounds_batched(
                layout_net, refine_bounds, mode="all", max_tensor_mib=1e-12,
            )
        assert all(torch.equal(skipped[lid].lb, box.lb) and torch.equal(skipped[lid].ub, box.ub)
                   for lid, box in refine_bounds.items())
        cold_layout = mixed_layout.drop_warm_state()
        cold_reference = SubproblemBatch(
            lb=cold_layout.lb, ub=cold_layout.ub, depths=cold_layout.depths,
            split_signs=cold_layout.split_signs,
        )
        assert torch.equal(cold_layout.lb, mixed_layout.lb)
        assert cold_layout.split_signs is not None
        assert torch.equal(cold_layout.split_signs[layout_lid], mixed_layout.split_signs[layout_lid])
        with patch.object(DualSolver, "solve_spec_batch", autospec=True,
                          side_effect=DualSolver.solve_spec_batch) as cold_spy:
            cold_result = solve_layout(cold_layout)
        assert cold_spy.call_args.kwargs["incremental_alphas"] is None
        assert cold_spy.call_args.kwargs["incremental_etas"] is None
        assert torch.equal(cold_result.margins, solve_layout(cold_reference).margins)
        from act.back_end.bab.climb import certificate_replay
        replay_cold = replace(mixed_layout, warm_state_present=torch.tensor([True, False]))
        assert layout_result.c_rows is not None and layout_result.thresholds is not None
        with patch.object(DualSolver, "compute_certified_bound",
                          side_effect=AssertionError("cold placeholders replayed")):
            assert certificate_replay(
                net=layout_net, batch=replay_cold, reference_bounds=layout_result.reference_bounds,
                c_rows=layout_result.c_rows, thresholds=layout_result.thresholds,
                m_specs=3, out_kind=OutKind.LINEAR_LE, with_costs=False,
            ) is None
        from act.back_end.bab.bound_refresh import refresh_bounds

        for mode in ("split_refresh", "split_refresh_linear"):
            for count in (1, 2):
                mode_spec = LAYER_TESTING_SPECS["layer_testing_climb_reuse"]()
                if count == 2:
                    mode_spec["layers"][-1]["params"].update(c=[[1.], [-1.]], d=[.1, .1])
                mode_net = factory.create_network(f"{mode}_{count}", mode_spec)
                mode_box = seed_from_input_specs(gather_input_spec_layers(mode_net))
                mode_base, _, _ = compute_forward_bounds(mode_net, mode_box.lb, mode_box.ub)
                mode_batch = SubproblemBatch.from_bounds(mode_box, depth=1)
                mode_batch = mode_batch.concat(mode_batch)
                mode_batch.split_signs = {
                    3: mode_batch.lb.new_tensor([[[-1., 0.]], [[1., 0.]]]).expand(-1, count, -1).clone()
                }
                mode_config = BaBConfig(
                    solver_tier="dual_alpha_eta", root_bounds_reuse=mode,
                    climb_enabled=True, branching_method="babsr", bounding="depth_bound_blend",
                )
                mode_result = _solve_dual_batch(
                    net=mode_net, assert_layer=get_assert_layer(mode_net),
                    batched_bounds=Bounds(mode_batch.lb, mode_batch.ub),
                    k_actual=2, batch=mode_batch, config=mode_config,
                    dual_config=DualConfig(n_iters=2, per_class_alpha=False),
                    optimize=True, root_bounds_dict=mode_base,
                )
                assert mode_result.m_specs == count
                assert mode_result.base_reference_bounds is not None
                mode_session = ClimbSession(mode_config, mode_net, OutKind.LINEAR_LE)
                mode_session.learn(mode_batch, mode_result.solution.statuses, mode_result, 2)
                if mode == "split_refresh_linear":
                    assert mode_result.solution.statuses == (SolveStatus.UNSAT,) * 2
                    assert mode_session.library.core_count > 0
        # Complement probes use the same solve path and original-domain base
        # as main lanes, with no recursive probing of their learned cores.
        complement_net = factory.create_network(
            "complement", LAYER_TESTING_SPECS["layer_testing_climb_admission"](),
        )
        complement_box = seed_from_input_specs(gather_input_spec_layers(complement_net))
        complement_base, _, _ = compute_forward_bounds(
            complement_net, complement_box.lb, complement_box.ub,
        )
        complement_config = BaBConfig(
            climb_enabled=True, climb_coarsening_enabled=False,
            climb_complement_bounding=True, solver_tier="dual_alpha_eta",
            branching_method="babsr", bounding="depth_bound_blend",
        )

        def solve_complement(candidate: SubproblemBatch, cap: float) -> DualBatchResult:
            """Exercise the real shared bounder and split-free result snapshot."""
            assert torch.equal(candidate.lb, complement_box.lb)
            assert candidate.incremental_alpha is candidate.incremental_eta is None
            result = _solve_dual_batch(
                net=complement_net, assert_layer=get_assert_layer(complement_net),
                batched_bounds=Bounds(candidate.lb, candidate.ub),
                k_actual=candidate.batch_size, batch=candidate,
                config=complement_config, dual_config=DualConfig(n_iters=30),
                optimize=True, root_bounds_dict=complement_base, time_cap=cap,
            )
            assert result.base_reference_bounds is not None
            assert torch.equal(result.base_reference_bounds[2].lb, complement_base[2].lb)
            return result

        for complement_mode in ("split_refresh", "split_refresh_linear"):
            complement_config = replace(complement_config, root_bounds_reuse=complement_mode)
            complement_session = ClimbSession(complement_config, complement_net, OutKind.LINEAR_LE)
            # The inactive core is already checked in climb's real replay smoke.
            complement_session.library.insert(torch.tensor([[-1]]))
            assert complement_session.bound_complements(solve_complement, lambda: 30., 1) == 1
            assert complement_session.metrics.complement_calls == 1
            assert complement_session.library.pack(complement_box.lb.device).shape == (1, 0)
            assert complement_session.bound_complements(solve_complement, lambda: 30., 1) == 0

            missing_base = ClimbSession(complement_config, complement_net, OutKind.LINEAR_LE)
            missing_base.library.insert(torch.tensor([[-1]]))
            try:
                missing_base.bound_complements(
                    lambda candidate, cap: replace(solve_complement(candidate, cap), base_reference_bounds=None),
                    lambda: 30., 1,
                )
            except ValueError as exc:
                assert "base_reference_bounds" in str(exc)
            else:
                raise AssertionError("complement used conditioned bounds as base")
        failing_complement_spec = LAYER_TESTING_SPECS["layer_testing_climb_admission"]()
        failing_complement_spec["layers"][-1]["params"]["c"] = [1.]
        complement_net = factory.create_network("failed_complement", failing_complement_spec)
        complement_base, _, _ = compute_forward_bounds(
            complement_net, complement_box.lb, complement_box.ub,
        )
        failed_complement = ClimbSession(complement_config, complement_net, OutKind.LINEAR_LE)
        failed_complement.library.insert(torch.tensor([[-1]]))
        assert failed_complement.bound_complements(solve_complement, lambda: 30., 1) == 1
        assert torch.equal(failed_complement.library.pack(complement_box.lb.device),
                           torch.tensor([[-1]], device=complement_box.lb.device))
        policy_trace: list[dict[str, int | float | None]] = []
        policy_run = verify_bab_batched(
            net, DualSolver,
            BaBConfig(
                solver_tier="dual_alpha_eta", branching_method="babsr",
                root_bounds_reuse="split_refresh_linear", bounding="depth_bound_blend",
                climb_enabled=True, climb_merge_enabled=True, climb_complement_bounding=True,
                climb_steer=True, climb_coarsening_enabled=False,
                max_nodes=8, max_depth=3,
            ),
            max_batch_size=1, dual_config=DualConfig(
                n_iters=2, per_class_alpha=False, incremental_start_enabled=False,
            ), trace=policy_trace,
        )
        assert policy_run.metadata["nodes"] <= 8
        assert policy_run.metadata["climb_complement_row_passes"] > 0
        assert policy_run.metadata["iterations"] == len(policy_trace)
        assert (sum(int(wave["K"]) for wave in policy_trace if isinstance(wave["K"], int))
                + policy_run.metadata["climb_complement_row_passes"] == policy_run.metadata["nodes"])
        finite_spec = LAYER_TESTING_SPECS["layer_testing_lp_dense"]()
        finite_spec["layers"].insert(-1, {"kind": "RELU", "params": {}})
        finite_net = factory.create_network("linear_finite_p", finite_spec)
        finite_bounds, _, _ = compute_forward_bounds(finite_net, lp_box.lb, lp_box.ub)
        finite_phase = {4: lp_box.lb.new_ones(1, 2, 1)}
        finite_refresh = refresh_bounds(finite_net, finite_bounds, finite_phase, linear=True)
        assert finite_refresh.linear_refined
        assert finite_refresh.bounds[4].ub.item() <= 10.0
        assert finite_refresh.bounds[4].ub.item() >= 9.99
        assert finite_net.input_geometry is not None
        assert torch.equal(finite_net.input_geometry.radii, saved_radii)
        trace: list[dict[str, int | float | None]] = []
        verdict = verify_bab_batched(
            net, DualSolver,
            BaBConfig(
                solver_tier="dual_alpha_eta", branching_method="babsr",
                root_bounds_reuse="plain", bounding="depth_bound_blend",
                max_depth=4, max_nodes=16,
            ),
            max_batch_size=4, time_budget_s=60,
            dual_config=DualConfig(n_iters=1, per_class_alpha=False), trace=trace,
        )
        assert verdict.status == VerifyStatus.UNKNOWN
        assert len(trace) > 1
        assert any(wave["K"] > 1 for wave in trace if isinstance(wave["K"], int))

        # A real candidate-less M=2 network remains UNKNOWN with the legacy
        # gate and drains through checked LP proofs when the gate is enabled.
        terminal_config = BaBConfig(
            solver_tier="dual_alpha_eta", branching_method="babsr",
            bounding="depth_bound_blend", climb_enabled=True,
            climb_coarsening_enabled=False, max_depth=4, max_nodes=31,
        )
        import sys

        with patch.dict(sys.modules, {"act.back_end.solver.terminal_lp": None}):
            gate_off = verify_bab_batched(
                net, DualSolver, terminal_config, max_batch_size=4,
                dual_config=DualConfig(n_iters=0),
            )
        assert gate_off.status == VerifyStatus.UNKNOWN
        assert gate_off.metadata["reason"] == "climb_retired_without_relu_candidate"
        assert "terminal_lp_calls" not in gate_off.metadata
        for climb_enabled in (False, True):
            terminal_trace: List[Dict[str, Any]] = []
            gate_on = verify_bab_batched(
                net, DualSolver, replace(
                    terminal_config, terminal_lp_enabled=True, climb_enabled=climb_enabled,
                ),
                max_batch_size=4, dual_config=DualConfig(n_iters=0), trace=terminal_trace,
            )
            assert gate_on.status == VerifyStatus.CERTIFIED, gate_on.metadata
            assert gate_on.metadata["terminal_lp_certified"] > 0
            assert gate_on.metadata["iterations"] == len(terminal_trace)
            assert all(len(row) == 10 for row in terminal_trace)
            assert sum(row["K"] for row in terminal_trace) == gate_on.metadata["nodes"]
        from act.back_end.solver.terminal_lp import solve_terminal_batch
        import numpy as np
        from scipy import sparse
        from types import SimpleNamespace
        from act.back_end.solver import terminal_lp as terminal_lp_module

        # Concrete ADD supports three operands, but the reused exporter is
        # binary. A valid Net must not certify a different, truncated graph.
        from act.back_end.bab.violation import _forward_for_violation_check
        from act.back_end.solver.milp_escalate import solve_milp

        add_box = Bounds(upper.new_zeros((1, 1)), upper.new_ones((1, 1)))
        add_layers = [
            Layer(0, "INPUT", {"shape": (1, 1), "dtype": str(upper.dtype)}, [], [0]),
            Layer(1, "INPUT_SPEC", {"kind": "BOX", "lb": add_box.lb, "ub": add_box.ub}, [0], [0]),
            *[Layer(i, "DENSE", {
                "weight": upper.new_ones((1, 1)), "bias": upper.new_zeros(1),
                "in_features": 1, "out_features": 1,
            }, [0], [i - 1]) for i in (2, 3, 4)],
            Layer(5, "ADD", {"x_vars": [1], "y_vars": [2]}, [1, 2, 3], [4]),
            Layer(6, "ASSERT", OutputSpec(
                OutKind.LINEAR_LE, c=upper.new_tensor([1.]), d=upper.new_tensor(2.5),
            ).encode_linear(1, 1, upper.device, upper.dtype), [4], []),
        ]
        add_preds = {0: [], 1: [0], 2: [1], 3: [1], 4: [1], 5: [2, 3, 4], 6: [5]}
        add_succs = {0: [1], 1: [2, 3, 4], 2: [5], 3: [5], 4: [5], 5: [6], 6: []}
        add_net = Net(add_layers, add_preds, add_succs)
        assert _forward_for_violation_check(add_net, add_box.ub).item() == 3.
        for prefer_affine in (False, True):
            rejected_add = solve_terminal_batch(add_net, add_box, prefer_affine=prefer_affine)
            assert rejected_add.statuses == (VerifyStatus.UNKNOWN,), rejected_add.statuses
            assert "unsupported_operand" in rejected_add.reasons[0]
        unsafe_add = Net([
            *add_layers[:-1], replace(add_layers[-1], params=OutputSpec(
                OutKind.UNSAFE_LINEAR, c=upper.new_tensor([[-1.]]), d=upper.new_tensor([-2.5]),
            ).encode_linear(1, 1, upper.device, upper.dtype)),
        ], add_preds, add_succs)
        rejected_milp = solve_milp(
            unsafe_add, BaBConfig(milp_escalation=True), deadline=time.monotonic() + 30,
        )
        assert rejected_milp.status == VerifyStatus.UNKNOWN
        assert not rejected_milp.metadata["milp_escalation"]["attempted"]
        for predecessors, inputs in (([2, 3], [1, 2]), ([2, 2], [1, 1])):
            binary_layers = [
                *add_layers[:5], replace(add_layers[5], in_vars=inputs, params={
                    "x_vars": inputs[:1], "y_vars": inputs[1:],
                }), add_layers[-1],
            ]
            binary_preds = {**add_preds, 5: predecessors}
            binary_succs = {**add_succs, 2: [5], 3: [5] if 3 in predecessors else [], 4: []}
            binary_net = Net(binary_layers, binary_preds, binary_succs)
            assert _forward_for_violation_check(binary_net, add_box.ub).item() == 2.
            for prefer_affine in (False, True):
                assert solve_terminal_batch(
                    binary_net, add_box, prefer_affine=prefer_affine,
                ).has_lp_certificate(0)
            malformed = Net([
                *binary_layers[:5], replace(binary_layers[5], params={
                    "x_vars": [3], "y_vars": inputs[1:],
                }), binary_layers[-1],
            ], binary_preds, binary_succs)
            assert solve_terminal_batch(malformed, add_box).statuses == (VerifyStatus.UNKNOWN,)

        # Residual proofs, not optimizer labels/objectives, own certification.
        problem = terminal_lp_module._SparseLP(
            sparse.csr_matrix((0, 1)), np.empty(0),
            sparse.csr_matrix([[1.]]), np.array([1.]),
            np.array([-1.]), np.array([1.]),
        )
        objective = np.array([-1.])
        proof_args = (
            problem.eq, problem.eq_rhs, problem.le, problem.le_rhs,
            problem.lb, problem.ub, objective, np.empty(0),
        )
        corrected = terminal_lp_module.certified_lower_bound(*proof_args, np.array([-1.]))
        assert corrected is not None and corrected <= -1.
        for invalid_dual in (1., float("nan"), float("inf")):
            assert terminal_lp_module.certified_lower_bound(
                *proof_args, np.array([invalid_dual]),
            ) is None
        forged_optimizer = SimpleNamespace(
            status=0, fun=1e99, eqlin=SimpleNamespace(marginals=np.empty(0)),
            ineqlin=SimpleNamespace(marginals=np.zeros(1)),
        )
        for optimizer_status in (0, 2):
            forged_optimizer.status = optimizer_status
            assert terminal_lp_module._certificate(
                problem, objective, 0., forged_optimizer, "lp_residual",
                OutKind.LINEAR_LE, upper.new_tensor(1.),
            ) is None
        impossible = replace(
            problem, le=sparse.csr_matrix([[1.], [-1.]]),
            le_rhs=np.array([-.5, -.5]),
        )
        phase_proof = terminal_lp_module._phase_one(impossible, None, upper.new_tensor(1.))
        assert phase_proof is not None and phase_proof.verify()
        assert phase_proof.kind == "phase1_residual"
        assert not replace(phase_proof, le_dual=np.full_like(phase_proof.le_dual, np.nan)).verify()
        assert not replace(phase_proof, lower_bound=float("inf")).verify()
        terminal_box = seed_from_input_specs(gather_input_spec_layers(net))
        twice = Bounds(terminal_box.lb.repeat(2, 1), terminal_box.ub.repeat(2, 1))
        relu_id = next(layer.id for layer in net.layers if layer.kind == LayerKind.RELU)
        fixed_phases = {relu_id: twice.lb.new_tensor(
            [[[1., 1.], [1., 1.]], [[-1., -1.], [-1., -1.]]],
        )}
        generic = solve_terminal_batch(net, twice, fixed_phases, prefer_affine=False)
        affine = solve_terminal_batch(net, twice, fixed_phases)
        assert bool(generic.certified.all()) and bool(affine.certified.all())
        assert generic.statuses == affine.statuses
        assert generic.tiers == ("generic", "generic") and affine.tiers == ("affine", "affine")
        unsupported = solve_terminal_batch(lp_net, lp_box)
        assert unsupported.statuses == (VerifyStatus.UNKNOWN,)
        assert unsupported.reasons == ("unsupported_finite_p_input",)
        with patch("act.back_end.solver.terminal_lp._solve", side_effect=ImportError("solver unavailable")):
            no_solver = solve_terminal_batch(net, terminal_box)
        assert no_solver.statuses == (VerifyStatus.UNKNOWN,)
        assert no_solver.reasons == ("no_solver",)
        with patch("act.back_end.solver.terminal_lp._solve", side_effect=TimeoutError("smoke deadline")):
            timed_out = solve_terminal_batch(net, terminal_box)
        assert timed_out.statuses == (VerifyStatus.TIMEOUT,) and not bool(timed_out.certified.any())

        real_bound = _solve_dual_batch
        mixed_seen: List[SubproblemBatch] = []

        def unresolved_bound(**kwargs: Any) -> DualBatchResult:
            """Withhold main proofs to isolate terminal orchestration paths."""
            mixed_seen.append(kwargs["batch"])
            result = real_bound(**kwargs)
            return replace(
                result, row_slack=None,
                solution=replace(
                    result.solution, statuses=(SolveStatus.UNKNOWN,) * kwargs["k_actual"],
                ),
            )

        def mixed_candidates(
            candidate: SubproblemBatch, bounds: Any, nu: Any,
        ) -> torch.Tensor:
            """One mixed K=2 wave: input fallback beside a normal neuron split."""
            assert nu is not None
            if candidate.batch_size == 2:
                assert all(value.shape[0] == 4 for value in nu.values())
                assert candidate.split_signs is not None
                assert all(value.shape[1] == 2 for value in candidate.split_signs.values())
                return torch.tensor([False, True], device=candidate.lb.device)
            return torch.full(
                (candidate.batch_size,), bool((candidate.depths == 0).all()),
                device=candidate.lb.device,
            )

        def unproved_terminal(*args: Any, **kwargs: Any) -> Any:
            """Withhold real LP proofs; status alone must not prune a lane."""
            result = solve_terminal_batch(*args, **kwargs)
            return replace(
                result, statuses=(VerifyStatus.CERTIFIED,) * len(result.statuses),
                certificates=((),) * len(result.statuses),
                reasons=("withheld_certificate",) * len(result.statuses),
            )

        mixed_trace: List[Dict[str, Any]] = []
        with (
            patch(f"{__name__}._solve_dual_batch", unresolved_bound),
            patch(f"{__name__}.lanes_with_relu_candidates", mixed_candidates),
            patch("act.back_end.solver.terminal_lp.solve_terminal_batch", unproved_terminal),
        ):
            mixed_result = verify_bab_batched(
                net, DualSolver, replace(
                    terminal_config, terminal_lp_enabled=True, provenance_enabled=True,
                    max_depth=2, max_nodes=7,
                ), max_batch_size=4, dual_config=DualConfig(n_iters=0), trace=mixed_trace,
            )
        assert mixed_result.status == VerifyStatus.UNKNOWN
        assert mixed_result.metadata["reason"] == "terminal_lp_unproved_retirement"
        assert mixed_result.metadata["terminal_lp_certified"] == 0
        assert mixed_result.metadata["terminal_input_fallback_lanes"] == 1
        assert mixed_trace[1]["branched"] == 2 and mixed_trace[1]["pool_after"] == 4
        assert mixed_result.metadata["terminal_failure_reasons"] == {"withheld_certificate:max_depth": 4}
        final_mixed = mixed_seen[-1]
        assert final_mixed.batch_size == 4 and final_mixed.split_signs is not None
        narrow = (final_mixed.widths() < 2).any(1)
        assert int(narrow.sum()) == 2
        phases = next(iter(final_mixed.split_signs.values()))
        assert bool((phases[narrow, 0].ne(0).sum(1) == 1).all())
        assert bool((phases[~narrow, 0].ne(0).sum(1) == 2).all())
        assert final_mixed.decision_literals is not None
        assert bool((final_mixed.decision_literals[narrow].ne(0).sum(1) == 1).all())
        assert final_mixed.node_id is not None and final_mixed.parent_id is not None
        assert final_mixed.node_id.unique().numel() == 4
        assert final_mixed.parent_id[narrow].unique().numel() == 1
        with (
            patch(f"{__name__}._solve_dual_batch", unresolved_bound),
            patch(f"{__name__}.lanes_with_relu_candidates", mixed_candidates),
            patch("act.back_end.solver.terminal_lp.solve_terminal_batch", unproved_terminal),
        ):
            budget_with_retirement = verify_bab_batched(
                net, DualSolver, replace(
                    terminal_config, terminal_lp_enabled=True, max_depth=1, max_nodes=2,
                ), max_batch_size=1, dual_config=DualConfig(n_iters=0),
            )
        assert budget_with_retirement.status == VerifyStatus.UNKNOWN
        assert budget_with_retirement.metadata["pool_remaining"] == 1
        assert budget_with_retirement.metadata["exhausted_budget_nodes"]
        assert budget_with_retirement.metadata["terminal_retired_lanes"] == 1
        assert budget_with_retirement.metadata["reason"] == "budget_exhausted_with_unproven_subboxes"

        # Inject at every owned wave boundary, including after terminal
        # fallback children have been staged but before normal children exist.
        planner_init = FrontierMemoryPlanner.__init__
        real_push, real_pop, real_apply = TopKBounding.push, TopKBounding.pop, SplitDecision.apply

        def simulated_init(self: FrontierMemoryPlanner, *args: Any, **kwargs: Any) -> None:
            """Activate deterministic policy testing without touching CUDA."""
            planner_init(self, *args, **kwargs)
            self.active = True

        for oom_site in ("pop", "solve", "split_after_terminal", "push", "k1", "root"):
            injected = [False]
            requeued: List[SubproblemBatch] = []
            attempted: List[SubproblemBatch] = []
            solved_ids: List[int] = []
            oom_trace: List[Dict[str, Any]] = []

            def observed_pop(self: TopKBounding, batch_size: int = 1) -> SubproblemBatch:
                """A failed pop must leave the original pool untouched."""
                if oom_site == "pop" and batch_size == 2 and not injected[0]:
                    injected[0] = True
                    raise torch.cuda.OutOfMemoryError("injected pop")
                value = real_pop(self, batch_size)
                if value.batch_size == 2 and bool((value.depths == 1).all()):
                    attempted.append(value.select(torch.arange(2)))
                return value

            def observed_push(self: TopKBounding, value: SubproblemBatch) -> None:
                """Record the exact original domains returned by recovery."""
                if oom_site == "push" and value.batch_size == 4 and not injected[0]:
                    injected[0] = True
                    raise torch.cuda.OutOfMemoryError("injected child push")
                if injected[0] and value.batch_size == 2 and bool((value.depths == 1).all()):
                    requeued.append(value)
                real_push(self, value)

            def failing_bound(**kwargs: Any) -> DualBatchResult:
                """Fail one whole K=2 wave, or the first real K=1 wave."""
                candidate = kwargs["batch"]
                if ((oom_site == "solve" and candidate.batch_size == 2)
                        or (oom_site == "k1" and kwargs.get("memory_planner") is not None)
                        or oom_site == "root") and not injected[0]:
                    injected[0] = True
                    raise torch.cuda.OutOfMemoryError("injected bound")
                result = unresolved_bound(**kwargs)
                if candidate.batch_size == 1 and bool((candidate.depths == 1).all()):
                    solved_ids.extend(candidate.node_id.tolist())
                return result

            def failing_split(self: SplitDecision, candidate: SubproblemBatch, network: Net) -> Any:
                """Fail after the sibling terminal fallback has been staged."""
                if (oom_site == "split_after_terminal"
                        and bool((candidate.depths == 1).all()) and not injected[0]):
                    injected[0] = True
                    raise torch.cuda.OutOfMemoryError("injected post-terminal split")
                return real_apply(self, candidate, network)

            with (
                patch.object(FrontierMemoryPlanner, "__init__", simulated_init),
                patch.object(FrontierMemoryPlanner, "begin_bound", return_value=0),
                patch.object(FrontierMemoryPlanner, "snapshot", return_value=MemoryBudget(10**12, 0, 0, 0, 0, 0)),
                patch("torch.cuda.max_memory_allocated", return_value=0),
                patch(f"{__name__}._build_bounding",
                      side_effect=lambda *_args, **_kw: TopKBounding(lambda depths, lower_bound: -depths)),
                patch.object(TopKBounding, "push", observed_push),
                patch.object(TopKBounding, "pop", observed_pop),
                patch.object(SplitDecision, "apply", failing_split),
                patch(f"{__name__}._solve_dual_batch", failing_bound),
                patch(f"{__name__}.lanes_with_relu_candidates", mixed_candidates),
                patch("act.back_end.solver.terminal_lp.solve_terminal_batch", unproved_terminal),
            ):
                recovered = verify_bab_batched(
                    net, DualSolver, replace(
                        terminal_config, terminal_lp_enabled=True, provenance_enabled=True,
                        climb_enabled=False, frontier_mem_safety=.6, frontier_offload=False,
                        max_depth=3, max_nodes=3,
                    ), max_batch_size=2, dual_config=DualConfig(n_iters=0), trace=oom_trace,
                )
            assert injected[0], oom_site
            assert recovered.status == VerifyStatus.UNKNOWN, (oom_site, recovered.metadata)
            assert recovered.metadata["iterations"] == len(oom_trace)
            assert all(len(row) == 10 for row in oom_trace)
            if oom_site == "root":
                assert recovered.metadata["reason"] == "root_out_of_memory"
                assert recovered.metadata["nodes"] == 0 and not oom_trace
            elif oom_site == "k1":
                assert recovered.metadata["reason"] == "out_of_memory"
                assert recovered.metadata["nodes"] == recovered.metadata["oom_retries"] == 0
                assert recovered.metadata["pool_remaining"] == 1 and len(oom_trace) == 1
            else:
                assert recovered.metadata["oom_retries"] == 1
                assert recovered.metadata["oom_failures"] == 0
                assert recovered.metadata["nodes"] == 3
                assert recovered.metadata["pool_remaining"] == 4
                assert sorted(solved_ids) == [1, 2], (oom_site, solved_ids)
                assert sum(row["K"] for row in oom_trace) == 5
                if oom_site != "pop":
                    assert len(requeued) == 1 and attempted
                    for field in ("lb", "ub", "depths", "node_id", "parent_id", "decision_literals"):
                        left, right = getattr(attempted[0], field), getattr(requeued[0], field)
                        assert (left is right is None) or torch.equal(left, right), (oom_site, field)
                    assert attempted[0].split_signs is not None and requeued[0].split_signs is not None
                    assert all(torch.equal(value, requeued[0].split_signs[lid])
                               for lid, value in attempted[0].split_signs.items())
                assert recovered.metadata["terminal_input_fallback_lanes"] == 2

        # Pre-pop CLIMB gathers the whole frontier: shrinking K cannot relieve it.
        real_before_pop, real_view = ClimbSession.before_pop, TopKBounding.view_all
        for remedy in ("offload", "cold", "exhausted"):
            for max_k in (1, 2):
                prepop_pool = TopKBounding()
                prepop_planners: list[FrontierMemoryPlanner] = []
                prepop_failures: list[int] = []
                prepop_pops: list[int] = []
                prepop_cover: list[SubproblemBatch] = []
                prepop_pressure = [False]

                def prepop_init(self: FrontierMemoryPlanner, *args: Any, **kwargs: Any) -> None:
                    """Activate the real policy with deterministic CPU budgets."""
                    simulated_init(self, *args, **kwargs)
                    prepop_planners.append(self)

                def needs_relief(frontier: BoundingStrategy) -> bool:
                    """Model pressure removed by offload or optional-state removal."""
                    resident = frontier._resident_batch()
                    return prepop_pressure[0] and (
                        remedy == "exhausted"
                        or (not prepop_planners[-1].offloaded if remedy == "offload"
                            else resident is not None and resident.incremental_alpha is not None)
                    )

                def prepop_snapshot(
                    self: FrontierMemoryPlanner, frontier: BoundingStrategy, *args: Any, **kwargs: Any,
                ) -> MemoryBudget:
                    """Expose the same pressure to the existing planner."""
                    return MemoryBudget(0 if needs_relief(frontier) else 10**12, 0, 0, 0, 0, 0)

                def prepop_hook(session: ClimbSession, frontier: TopKBounding) -> None:
                    """Enable the real propagation path after the root branches."""
                    if len(frontier) >= 2:
                        prepop_pressure[0] = True
                        if not session.library.core_count:
                            session.library.insert(torch.tensor([[1, 2]]))
                    real_before_pop(session, frontier)

                def prepop_view(frontier: TopKBounding) -> SubproblemBatch:
                    """Fail the whole-pool gather until the real mitigation helps."""
                    if needs_relief(frontier):
                        if not prepop_cover:
                            prepop_cover.append(real_view(frontier))
                        prepop_failures.append(len(frontier))
                        raise torch.cuda.OutOfMemoryError("pre-pop frontier pressure")
                    return real_view(frontier)

                def prepop_pop(frontier: TopKBounding, batch_size: int = 1) -> SubproblemBatch:
                    """Observe the first actual wave after pre-pop recovery."""
                    if prepop_pressure[0]:
                        prepop_pops.append(batch_size)
                    return real_pop(frontier, batch_size)

                prepop_trace: List[Dict[str, Any]] = []
                with (
                    patch.object(FrontierMemoryPlanner, "__init__", prepop_init),
                    patch.object(FrontierMemoryPlanner, "begin_bound", return_value=0),
                    patch.object(FrontierMemoryPlanner, "snapshot", prepop_snapshot),
                    patch.object(FrontierMemoryPlanner, "_host_available", return_value=10**9),
                    patch("torch.cuda.max_memory_allocated", return_value=0),
                    patch(f"{__name__}._build_bounding", return_value=prepop_pool),
                    patch(f"{__name__}._solve_dual_batch", unresolved_bound),
                    patch.object(ClimbSession, "before_pop", prepop_hook),
                    patch.object(TopKBounding, "view_all", prepop_view),
                    patch.object(TopKBounding, "pop", prepop_pop),
                ):
                    prepop_result = verify_bab_batched(
                        cold_net, DualSolver, BaBConfig(
                            solver_tier="dual_alpha_eta", branching_method="babsr",
                            bounding="depth_bound_blend", climb_enabled=True,
                            climb_coarsening_enabled=False, provenance_enabled=True,
                            frontier_mem_safety=.6, frontier_offload=remedy == "offload",
                            max_nodes=3,
                        ), max_batch_size=max_k, dual_config=DualConfig(n_iters=0),
                        trace=prepop_trace,
                    )
                assert prepop_result.status == VerifyStatus.UNKNOWN
                assert prepop_result.metadata["iterations"] == len(prepop_trace)
                assert all(len(row) == 10 for row in prepop_trace)
                assert prepop_result.metadata["frontier_memory_evicted_nodes"] == 0
                if remedy == "exhausted":
                    assert prepop_result.metadata["reason"] == "out_of_memory"
                    assert prepop_failures == [2, 2] and not prepop_pops
                    assert prepop_result.metadata["nodes"] == 1
                    assert prepop_result.metadata["pool_remaining"] == 2
                    remaining = real_view(prepop_pool)
                    for field in ("lb", "ub", "depths", "node_id", "parent_id", "decision_literals"):
                        assert torch.equal(getattr(prepop_cover[0], field), getattr(remaining, field))
                    assert prepop_result.metadata["oom_failures"] == 1
                else:
                    assert prepop_result.metadata["reason"] != "out_of_memory", (
                        remedy, max_k, prepop_result.metadata
                    )
                    assert prepop_failures == [2] and prepop_pops[0] == max_k
                    assert prepop_result.metadata["nodes"] == 3
                    assert prepop_result.metadata["oom_failures"] == 0
                    if remedy == "offload":
                        assert prepop_result.metadata["frontier_offload_count"] == 1
                    else:
                        assert prepop_result.metadata["frontier_warm_drop_count"] == 1
                assert prepop_result.metadata["oom_retries"] == 1

        # CLIMB frontier replacement belongs to the same transaction as children.
        real_before_split = ClimbSession.before_split
        merge_injected = [False]
        merge_pool = TopKBounding()

        def merge_then_fail(
            session: ClimbSession, frontier: TopKBounding, candidate: SubproblemBatch,
        ) -> tuple[SubproblemBatch, torch.Tensor]:
            """Fail after an actual pending/active sibling pair revives its parent."""
            if not merge_injected[0] and bool((candidate.depths == 1).all()):
                session._insert_cores(torch.tensor([[2]]))
                merged = real_before_split(session, frontier, candidate)
                assert session.metrics.merged_pairs == 1 and merged[0].batch_size == 0
                merge_injected[0] = True
                raise torch.cuda.OutOfMemoryError("after pending-active merge")
            return real_before_split(session, frontier, candidate)

        with (
            patch.object(FrontierMemoryPlanner, "__init__", simulated_init),
            patch.object(FrontierMemoryPlanner, "begin_bound", return_value=0),
            patch.object(FrontierMemoryPlanner, "snapshot", return_value=MemoryBudget(10**12, 0, 0, 0, 0, 0)),
            patch("torch.cuda.max_memory_allocated", return_value=0),
            patch(f"{__name__}._build_bounding", return_value=merge_pool),
            patch(f"{__name__}._solve_dual_batch", unresolved_bound),
            patch.object(ClimbSession, "before_split", merge_then_fail),
        ):
            merge_oom = verify_bab_batched(
                cold_net, DualSolver, BaBConfig(
                    solver_tier="dual_alpha_eta", branching_method="babsr",
                    bounding="depth_bound_blend", climb_enabled=True,
                    climb_merge_enabled=True,
                    frontier_mem_safety=.6, frontier_offload=False, max_nodes=8,
                ), max_batch_size=1, dual_config=DualConfig(n_iters=0),
            )
        resident = merge_pool.view_all()
        assert resident.node_id is not None and resident.decision_literals is not None
        assert merge_injected[0] and set(resident.node_id.tolist()) == {1, 2}
        assert not bool(resident.decision_literals.eq(0).all(1).any())
        assert merge_oom.metadata["climb_merged_pairs"] == 0
        assert merge_oom.metadata["nodes"] == 1 and merge_oom.metadata["pool_remaining"] == 2

        # A real failed frame's tensor must die before recovery allocates.
        import weakref
        temporary_refs: list[weakref.ReferenceType[torch.Tensor]] = []
        live_at_recovery: list[bool] = []
        real_relieve = FrontierMemoryPlanner._relieve

        def fail_with_temporary(**kwargs: Any) -> DualBatchResult:
            """Leave the only strong work-buffer reference in the failing frame."""
            if kwargs.get("memory_planner") is not None:
                temporary = upper.new_ones(1024)
                temporary_refs.append(weakref.ref(temporary))
                raise torch.cuda.OutOfMemoryError("live solver work buffer")
            return unresolved_bound(**kwargs)

        def observe_recovery(planner: FrontierMemoryPlanner, frontier: BoundingStrategy) -> None:
            """Observe lifetime at the first allocating recovery boundary."""
            live_at_recovery.append(temporary_refs[-1]() is not None)
            real_relieve(planner, frontier)

        with (
            patch.object(FrontierMemoryPlanner, "__init__", simulated_init),
            patch.object(FrontierMemoryPlanner, "begin_bound", return_value=0),
            patch.object(FrontierMemoryPlanner, "snapshot", return_value=MemoryBudget(10**12, 0, 0, 0, 0, 0)),
            patch.object(FrontierMemoryPlanner, "_relieve", observe_recovery),
            patch(f"{__name__}._solve_dual_batch", fail_with_temporary),
        ):
            lifetime_result = verify_bab_batched(
                cold_net, DualSolver, BaBConfig(
                    solver_tier="dual_alpha_eta", branching_method="babsr",
                    bounding="depth_bound_blend", frontier_mem_safety=.6,
                    frontier_offload=False,
                ), max_batch_size=1, dual_config=DualConfig(n_iters=0),
            )
        assert live_at_recovery == [False] and temporary_refs[-1]() is None
        assert lifetime_result.status == VerifyStatus.UNKNOWN

        # A deferred merge keeps an offloaded pending pool on its storage device.
        host_siblings = SubproblemBatch(
            cold_box.lb.expand(2, -1), cold_box.ub.expand(2, -1), torch.ones(2, dtype=torch.long),
            node_id=torch.tensor([1, 2]), parent_id=torch.tensor([0, 0]),
            decision_literals=torch.tensor([[-1], [1]]),
            split_signs={cold_lid: upper.new_tensor([[[-1., 0.]], [[1., 0.]]])},
        )
        host_pool = TopKBounding()
        host_pool.push(host_siblings)
        host_session = ClimbSession(
            BaBConfig(climb_merge_enabled=True, climb_presplit_propagation_enabled=False),
            cold_net, OutKind.LINEAR_LE,
        )
        host_session._insert_cores(torch.tensor([[2]]))
        host_session.before_pop(host_pool)
        assert len(host_pool) == 1 and host_session.metrics.merged_pairs == 1
        assert host_pool.view_all().lb.device.type == "cpu"

        warm_host = TopKBounding()
        warm_host.push(SubproblemBatch(
            -upper, upper, torch.zeros(1, dtype=torch.long),
            incremental_alpha={cold_lid: upper.new_ones((1, 512))},
        ))
        restore_planner = FrontierMemoryPlanner(torch.device("meta"), .6, True, 100)
        restore_planner.active = True
        with (
            patch.object(restore_planner, "snapshot", return_value=MemoryBudget(1000, 0, 0, 0, 0, 0)),
            patch.object(restore_planner, "_host_available", return_value=10**9),
        ):
            assert restore_planner.plan_wave(warm_host, 1) == 1
        assert restore_planner.warm_drop_count == 1
        assert warm_host.pop().incremental_alpha is None

        # Adjacent floats and point boxes cannot spawn identical children.
        point = upper.new_tensor([[1.], [0.]])
        adjacent = torch.nextafter(point, torch.full_like(point, torch.inf))
        splittable, _ = _terminal_input_axes(SubproblemBatch.from_bounds(Bounds(point, adjacent)))
        assert not bool(splittable.any())
        # The terminal owner validates this counterexample on the concrete
        # original model; the main bound is deliberately withheld.
        falsified_spec = LAYER_TESTING_SPECS["layer_testing_climb_reuse"]()
        falsified_spec["layers"][-1]["params"]["d"] = -.1
        falsified_net = factory.create_network("terminal_falsified", falsified_spec)
        falsified_trace: List[Dict[str, Any]] = []
        with (
            patch(f"{__name__}._solve_dual_batch", unresolved_bound),
            patch(f"{__name__}.lanes_with_relu_candidates",
                  side_effect=lambda candidate, *_: torch.zeros(candidate.batch_size, dtype=torch.bool)),
        ):
            terminal_sat = verify_bab_batched(
                falsified_net, DualSolver, replace(terminal_config, terminal_lp_enabled=True),
                max_batch_size=2, dual_config=DualConfig(n_iters=0), trace=falsified_trace,
            )
        assert terminal_sat.status == VerifyStatus.FALSIFIED
        assert terminal_sat.counterexample is not None
        assert terminal_sat.metadata["resolved_by"] == "terminal_lp"
        assert terminal_sat.metadata["iterations"] == len(falsified_trace) == 1
        assert falsified_trace[0]["falsified_lane"] == terminal_sat.metadata["lane"] == 0
        deadline_clock = [0.0]

        def expired_terminal(*args: Any, **kwargs: Any) -> Any:
            """Expire at the terminal boundary without relying on elapsed time."""
            assert kwargs["deadline"] == 10.
            kwargs["deadline"] = time.monotonic() - 1
            result = solve_terminal_batch(*args, **kwargs)
            deadline_clock[0] = 10.
            return result

        with (
            patch(f"{__name__}._monotonic", lambda: deadline_clock[0]),
            patch(f"{__name__}._solve_dual_batch", unresolved_bound),
            patch(f"{__name__}.lanes_with_relu_candidates",
                  side_effect=lambda candidate, *_: torch.zeros(candidate.batch_size, dtype=torch.bool)),
            patch("act.back_end.solver.terminal_lp.solve_terminal_batch", expired_terminal),
        ):
            terminal_timeout = verify_bab_batched(
                net, DualSolver, replace(terminal_config, terminal_lp_enabled=True),
                max_batch_size=2, time_budget_s=10., dual_config=DualConfig(n_iters=0),
            )
        assert terminal_timeout.status == VerifyStatus.UNKNOWN
        assert terminal_timeout.metadata["terminal_input_fallback_lanes"] == 0
        assert terminal_timeout.metadata["exhausted_budget_time"]
        assert all(reason.endswith(":deadline_exhausted")
                   for reason in terminal_timeout.metadata["terminal_failure_reasons"])

        # Control the clock at solver boundaries, not by sleeping. A root-only
        # deadline must leave the remaining search budget available to children.
        original_solve = _solve_dual_batch
        original_forward = compute_forward_bounds
        for case in (
            "completed", "timeout", "skipped", "pruned", "presplit",
            "main_timeout", "global", "default",
        ):
            clock = [0.0]
            calls: list[tuple[tuple[int, ...], bool, Optional[list[int]]]] = []
            controlled_trace: list[Dict[str, Any]] = []
            controlled_net = net
            if case == "pruned":
                pruned_spec = LAYER_TESTING_SPECS["layer_testing_climb_reuse"]()
                pruned_spec["layers"][-1]["params"].update(
                    c=[[1.], [-1.]], d=[10., .1]
                )
                controlled_net = factory.create_network("pruned_rows", pruned_spec)

            def controlled_solve(**kwargs: Any) -> DualBatchResult:
                """Expire only the requested window at a deterministic boundary."""
                rows = kwargs.get("keep_rows")
                calls.append((
                    tuple(kwargs["batch"].depths.tolist()), kwargs["optimize"],
                    None if rows is None else rows.tolist(),
                ))
                if (case == "timeout" and len(calls) == 1) or (
                    case == "main_timeout" and len(calls) == 2
                ):
                    clock[0] = 2.0
                    raise TimeoutError("controlled root work exhausted")
                result = original_solve(**kwargs)
                if case != "main_timeout":
                    clock[0] = 10.0 if case == "global" else 2.0
                return result

            def controlled_forward(*args: Any, **kwargs: Any) -> Any:
                """Cover root setup consuming its window before presolve starts."""
                result = original_forward(*args, **kwargs)
                if case == "skipped":
                    clock[0] = 2.0
                return result

            with (
                patch(f"{__name__}._monotonic", lambda: clock[0]),
                patch("act.back_end.solver.solver_dual._monotonic", lambda: clock[0]),
                patch(f"{__name__}._solve_dual_batch", controlled_solve),
                patch(f"{__name__}.compute_forward_bounds", controlled_forward),
            ):
                controlled = verify_bab_batched(
                    controlled_net, DualSolver,
                    BaBConfig(
                        solver_tier="dual_alpha_eta", branching_method="babsr",
                        root_bounds_reuse="plain", bounding="depth_bound_blend",
                        max_depth=1, max_nodes=3,
                        presolve_time_fraction=0.0 if case == "default" else .1,
                        presplit_levels=int(case == "presplit"),
                    ),
                    max_batch_size=2, time_budget_s=10,
                    dual_config=DualConfig(n_iters=1), trace=controlled_trace,
                )
            assert controlled.status == VerifyStatus.UNKNOWN, (case, controlled)
            if case == "global":
                assert not controlled_trace and len(calls) == 1
                assert controlled.metadata["exhausted_budget_time"]
                continue
            assert controlled_trace, (case, controlled.metadata, calls)
            assert any(all(depth > 0 for depth in depths) for depths, _, _ in calls), (
                case, calls
            )
            assert not controlled.metadata["exhausted_budget_time"]
            root_calls = [(optimize, rows) for depths, optimize, rows in calls if depths == (0,)]
            if case in ("completed", "presplit"):
                assert root_calls == [(True, None)], (case, calls)
            elif case == "timeout":
                assert root_calls == [(True, None), (False, None)], calls
            elif case == "skipped":
                assert root_calls == [(False, None)], calls
            elif case == "pruned":
                assert root_calls == [(True, None), (False, [1])], calls
                assert controlled.metadata["spec_rows_kept"] == 1
            elif case == "main_timeout":
                assert root_calls == [(True, None), (True, None), (False, None)], calls
            else:
                assert root_calls == [(True, None), (True, None)], calls

        # Root/mixed batches retain the base optimizer policy. Only optimized
        # all-child batches use the override, carried solely by DualConfig.
        bounds = seed_from_input_specs(gather_input_spec_layers(net))
        root = SubproblemBatch.from_bounds(bounds)
        child = SubproblemBatch.from_bounds(bounds, depth=1)
        children = child.concat(child)
        mixed = root.concat(child)
        base_dual = DualConfig(n_iters=3, max_time=7.0)
        for batch, optimize, child_iters, time_cap, expected_iters in (
            (root, True, 2, 5.0, 3),
            (children, True, 2, 5.0, 2),
            (mixed, True, 2, 5.0, 3),
            (children, False, 2, 5.0, 3),
            (children, True, 0, None, 3),
            (children, True, 2, None, 2),
        ):
            with (
                patch(f"{__name__}._monotonic", return_value=0.0),
                patch("act.back_end.solver.solver_dual._monotonic", return_value=0.0),
                patch.object(
                    DualSolver, "solve_spec_batch", autospec=True,
                    side_effect=DualSolver.solve_spec_batch,
                ) as solve_spy,
            ):
                _solve_dual_batch(
                    net=net, assert_layer=get_assert_layer(net),
                    batched_bounds=Bounds(batch.lb, batch.ub),
                    k_actual=batch.batch_size, batch=batch,
                    config=BaBConfig(
                        solver_tier="dual_alpha_eta", child_n_iters=child_iters,
                        eta_only_children=True,
                    ),
                    dual_config=base_dual, optimize=optimize, time_cap=time_cap,
                )
            assert solve_spy.call_count == 1
            kwargs = solve_spy.call_args.kwargs
            assert "n_iters" not in kwargs
            assert kwargs["dual_config"].n_iters == expected_iters
            assert kwargs["dual_config"].max_time == (7.0 if time_cap is None else 5.0)
            assert kwargs["optimize_alpha"] == bool((batch.depths == 0).any().item())
            assert base_dual.n_iters == 3 and base_dual.max_time == 7.0
