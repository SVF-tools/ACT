#===- act/back_end/verifier.py - Spec-free Verification Engine ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Spec-free, input-free verification. Assumes the ACT Net already encodes
#   both input and output specifications via INPUT_SPEC and ASSERT layers
#   (produced by torch2act.TorchToACT).
#
# Architecture — verify_once:
#   1. Seed [B, *input_shape] bounds from INPUT_SPEC layers (no CSP).
#   2. analyze() propagates batched bounds through every TF op.
#   3. Read pre-encoded [B*M, n_out] linear-form C / [B, M] thresholds / M
#      from the ASSERT layer params (produced upstream by
#      OutputSpec.encode_linear at FE construction time).
#   4. INTERVAL CERTIFICATION: one tensor pass computes margin_max under
#      output bounds; sample b is CERTIFIED iff every M lane passes.
#   5. CONCRETE FALSIFICATION (when model_fn given): one batched forward at
#      box centre; samples whose concrete output meets-or-exceeds threshold
#      become FALSIFIED. Remaining samples are UNKNOWN.
#   6. Return List[VerifyResult] of length B (one per input lane).
#
#===---------------------------------------------------------------------===#

# Public API:
#   - verify_once(net, *, model_fn=None, collect_facts=False)
#       Pure-tensor batched single-shot verifier. Returns List[VerifyResult]
#       by default, or (results, facts_or_none) when collect_facts=True.
#   - setup_and_solve_batch(net, input_bounds_per_b, solver, timelimit=None)
#       Batch-native CSP setup helper used by LP and BaB refinement.
#   - find_entry_layer_id / get_input_ids / get_output_ids /
#     gather_input_spec_layers / get_assert_layer / seed_from_input_specs /
#     add_all_input_specs (helpers).
#
# Notes:
#   * Spec-free verification: all constraints extracted from ACT Net layers.
#   * verify_once returns one VerifyResult per lane (len(result) == B).
#   * INPUT_SPEC constraints (including LIN_POLY) are propagated through
#     analyze(); they enter via add_all_input_specs into entry_fact.cons.
#     LIN_POLY constraints are not consumed by verify_once's interval
#     certification; they are preserved for the batch-native solver path.

from __future__ import annotations
from typing import Optional, List, Callable, Dict, Any, TYPE_CHECKING, Tuple, Literal, overload

import torch
import copy

# ACT backend imports
from act.back_end.core import Bounds, Con, ConSet, Fact, Net
from act.back_end.solver.solver_base import Solver, SolveStatus, BatchLPSolution
from act.back_end.layer_schema import LayerKind
from act.back_end.utils import validate_constraints

if TYPE_CHECKING:
    from act.back_end.analyze import AnalyzeCache

# Front-end enums (kinds)
from act.front_end.specs import InKind, OutKind, OutputSpec, normalize_position_mask

# Verification types (canonical location: act/util/stats.py)
from act.util.stats import VerifyStatus, VerifyResult

# -----------------------------------------------------------------------------
# Sequential per-sample slicing (for B>1 BaB)
# -----------------------------------------------------------------------------

def _slice_first_dim(value: Any, sample_idx: int, expected_b: int) -> Any:
    if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == expected_b:
        return value[sample_idx:sample_idx + 1]
    return value


def slice_net_to_sample(net: Net, sample_idx: int) -> Net:
    from act.front_end.spec_creator_base import LabeledInputTensor

    mutable_kinds = {
        LayerKind.INPUT.value,
        LayerKind.INPUT_SPEC.value,
        LayerKind.ASSERT.value,
    }
    layers = []
    for layer in net.layers:
        if layer.kind not in mutable_kinds:
            layers.append(layer)
            continue
        layer2 = copy.copy(layer)
        layer2.params = dict(layer.params)
        layer2.in_vars = list(layer.in_vars)
        layer2.out_vars = list(layer.out_vars)
        layer2.cache = dict(layer.cache)
        layers.append(layer2)
    net2 = copy.copy(net)
    net2.layers = layers
    net2.preds = net.preds
    net2.succs = net.succs
    net2.by_id = {layer.id: layer for layer in layers}

    entry_id = find_entry_layer_id(net2)
    input_layer = net2.by_id[entry_id]
    shape = input_layer.params.get("shape") or []
    shape_t = tuple(shape) if isinstance(shape, (list, tuple)) else ()
    B = int(shape_t[0]) if shape_t else 1
    if shape_t and int(shape_t[0]) == B:
        input_layer.params["shape"] = (1,) + tuple(shape_t[1:])
    li = input_layer.params.get("labeled_input")
    if isinstance(li, LabeledInputTensor):
        new_tensor = _slice_first_dim(li.tensor, sample_idx, B)
        new_label = _slice_first_dim(li.label, sample_idx, B) if li.label is not None else None
        input_layer.__dict__["params"]["labeled_input"] = LabeledInputTensor(
            tensor=new_tensor, label=new_label,
        )

    for spec_layer in gather_input_spec_layers(net2):
        for key in ("center", "eps", "lb", "ub", "A", "b"):
            val = spec_layer.params.get(key)
            if val is not None:
                spec_layer.params[key] = _slice_first_dim(val, sample_idx, B)

    assert_layer = get_assert_layer(net2)
    m_raw = assert_layer.params.get("M", 1)
    if isinstance(m_raw, torch.Tensor):
        m_rows = int(m_raw.item())
    elif isinstance(m_raw, int):
        m_rows = m_raw
    else:
        raise ValueError(f"ASSERT M must be int or tensor, got {m_raw!r}")
    for key in ("y_true", "margin", "c", "d", "lb", "ub"):
        val = assert_layer.params.get(key)
        if val is not None:
            assert_layer.params[key] = _slice_first_dim(val, sample_idx, B)
    # C is [B*M, n_out] — first dim is B*M not B, so slice rows manually
    c_big = assert_layer.params.get("C")
    if isinstance(c_big, torch.Tensor) and c_big.shape[0] == B * m_rows:
        assert_layer.params["C"] = c_big[sample_idx * m_rows:(sample_idx + 1) * m_rows]
    thresholds = assert_layer.params.get("thresholds")
    if isinstance(thresholds, torch.Tensor) and thresholds.shape[0] == B:
        assert_layer.params["thresholds"] = thresholds[sample_idx:sample_idx + 1]

    return net2


# -----------------------------------------------------------------------------
# ACT Net extraction helpers
# -----------------------------------------------------------------------------

def find_entry_layer_id(net) -> int:
    """Return the id of the single INPUT layer."""
    candidates = [L.id for L in net.layers if L.kind == LayerKind.INPUT.value]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one INPUT layer, found {len(candidates)}.")
    return candidates[0]

def get_input_ids(net) -> List[int]:
    """Return input variable IDs (out_vars of INPUT layer)."""
    entry = find_entry_layer_id(net)
    return list(net.by_id[entry].out_vars)

def get_output_ids(net) -> List[int]:
    """Return output variable IDs (in_vars of ASSERT layer)."""
    assert_layer = net.layers[-1]
    if assert_layer.kind != LayerKind.ASSERT.value:
        raise ValueError("Expected last layer to be ASSERT.")
    return list(assert_layer.in_vars)

def gather_input_spec_layers(net):
    """Return list of INPUT_SPEC layers."""
    return [L for L in net.layers if L.kind == LayerKind.INPUT_SPEC.value]

def get_assert_layer(net):
    """Return the ASSERT layer (must be last)."""
    assert_layer = net.layers[-1]
    if assert_layer.kind != LayerKind.ASSERT.value:
        raise ValueError("Expected last layer to be ASSERT.")
    return assert_layer

# -----------------------------------------------------------------------------
# Seed and input spec helpers
# -----------------------------------------------------------------------------

def seed_from_input_specs(spec_layers) -> Bounds:
    """
    Create seed Bounds from INPUT_SPEC layers.
    Prefers BOX, then LINF_BALL, raises if only LIN_POLY exists.
    
    Note: This extracts only box bounds for seeding abstract interpretation.
    All constraints (including LIN_POLY) are added via add_all_input_specs().
    """
    # BOX first
    for spec_layer in spec_layers:
        if spec_layer.params.get("kind") == InKind.BOX and "lb" in spec_layer.params and "ub" in spec_layer.params:
            return Bounds(spec_layer.params["lb"].clone(), spec_layer.params["ub"].clone())
    
    # LINF_BALL next
    for spec_layer in spec_layers:
        if spec_layer.params.get("kind") == InKind.LINF_BALL:
            if "lb" in spec_layer.params and "ub" in spec_layer.params:
                return Bounds(spec_layer.params["lb"].clone(), spec_layer.params["ub"].clone())
            center = spec_layer.params.get("center")
            eps = spec_layer.params.get("eps")
            if center is not None and eps is not None:
                e = eps.to(device=center.device, dtype=center.dtype) if torch.is_tensor(eps) else center.new_tensor(eps)
                return Bounds(center - e, center + e)

    # LP_EMBEDDING seeds the enclosing box; finite-p precision is recovered by
    # the dual input contribution, which reads p_norm/perturbed_positions.
    for spec_layer in spec_layers:
        if spec_layer.params.get("kind") == InKind.LP_EMBEDDING:
            if "lb" in spec_layer.params and "ub" in spec_layer.params:
                return Bounds(spec_layer.params["lb"].clone(), spec_layer.params["ub"].clone())
            center = spec_layer.params.get("center")
            eps = spec_layer.params.get("eps")
            if center is None or eps is None:
                raise ValueError("LP_EMBEDDING requires center/eps or lb/ub for seeding.")
            e = eps.to(device=center.device, dtype=center.dtype) if torch.is_tensor(eps) else center.new_tensor(eps)
            lb = center.clone()
            ub = center.clone()
            mask = normalize_position_mask(
                spec_layer.params.get("perturbed_positions"),
                int(center.shape[-2]),
                batch_shape=tuple(center.shape[:-2]),
                device=center.device,
            )
            expanded = mask.unsqueeze(-1).expand_as(center)
            return Bounds(torch.where(expanded, center - e, lb), torch.where(expanded, center + e, ub))
    
    # LIN_POLY only -> error
    if any(spec_layer.params.get("kind") == InKind.LIN_POLY for spec_layer in spec_layers):
        raise ValueError("LIN_POLY requires a seed box (BOX or LINF_BALL).")
    
    raise ValueError("No valid input specification found for seeding.")


def _setup_verify_context(net):
    spec_layers = gather_input_spec_layers(net)
    seed_bounds = seed_from_input_specs(spec_layers)
    if seed_bounds.lb.dim() < 2:
        message = (
            f"_setup_verify_context: INPUT_SPEC seed must be batched [B, *input_shape], got dim={seed_bounds.lb.dim()} "
            f"shape={tuple(seed_bounds.lb.shape)}."
        )
        raise ValueError(message)
    B = int(seed_bounds.lb.shape[0])
    return spec_layers, seed_bounds, B


def add_all_input_specs(globalC: ConSet, input_ids: List[int], spec_layers) -> None:
    """
    Add all INPUT_SPEC constraints to constraint set.
    
    This function adds:
    - BOX constraints (box bounds)
    - LINF_BALL constraints (converted to box)
    - LP_EMBEDDING/LIN_POLY constraints (box seed or linear polytope A·x ≤ b)
    
    The LIN_POLY constraints are tagged with "in:linpoly" and will be
    exported by export_to_batch_problem() in cons_exportor.py.
    """
    for L in spec_layers:
        k = L.params.get("kind")
        if k == InKind.BOX:
            globalC.add_box(-1, input_ids, Bounds(L.params["lb"], L.params["ub"]))
        elif k == InKind.LINF_BALL:
            if "lb" in L.params and "ub" in L.params:
                globalC.add_box(-1, input_ids, Bounds(L.params["lb"], L.params["ub"]))
            else:
                center = L.params["center"]
                eps = L.params["eps"]
                e = eps.to(device=center.device, dtype=center.dtype) if torch.is_tensor(eps) else center.new_tensor(eps)
                globalC.add_box(-1, input_ids, Bounds(center - e, center + e))
        elif k == InKind.LP_EMBEDDING:
            if "lb" in L.params and "ub" in L.params:
                globalC.add_box(-1, input_ids, Bounds(L.params["lb"], L.params["ub"]))
            else:
                center = L.params["center"]
                eps = L.params["eps"]
                e = eps.to(device=center.device, dtype=center.dtype) if torch.is_tensor(eps) else center.new_tensor(eps)
                globalC.add_box(-1, input_ids, Bounds(center - e, center + e))
        elif k == InKind.LIN_POLY:
            A, b = L.params["A"], L.params["b"]
            globalC.replace(Con("INEQ", tuple(input_ids), {"tag": "in:linpoly", "A": A, "b": b}))
        else:
            raise NotImplementedError(f"Unsupported INPUT_SPEC kind: {k}")




@torch.no_grad()
def setup_and_solve_batch(
    net,
    input_bounds_per_b: Bounds,
    solver: Solver,
    timelimit: Optional[float] = None,
    *,
    cache: Optional["AnalyzeCache"] = None,
) -> BatchLPSolution:
    """[BATCHED-API] Orchestrate analyze → export_to_batch_problem → solve_batch.

    ``input_bounds_per_b`` must already be a tensor-view batch
    ``[B, *input_shape]``; B=1 is just
    the length-one batch case, not a scalar special case.
    """
    from act.back_end.analyze import analyze
    from act.back_end.cons_exportor import export_to_batch_problem

    if input_bounds_per_b.lb.dim() < 2 or input_bounds_per_b.ub.dim() < 2:
        raise ValueError(
            f"setup_and_solve_batch: input_bounds_per_b must be batched "
            f"[B, *input_shape], got lb={tuple(input_bounds_per_b.lb.shape)} "
            f"ub={tuple(input_bounds_per_b.ub.shape)}"
        )

    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)

    entry_fact = Fact(bounds=input_bounds_per_b, cons=ConSet())
    add_all_input_specs(entry_fact.cons, input_ids, spec_layers)

    _before, after, globalC = analyze(net, entry_id, entry_fact, cache=cache)
    validate_constraints(globalC, after, net)

    problem = export_to_batch_problem(
        net=net,
        globalC=globalC,
        assert_layer=assert_layer,
        input_box_per_b=input_bounds_per_b,
    )
    solution = solver.solve_batch(problem, timelimit=timelimit)

    expected_n = int(input_bounds_per_b.lb.shape[0])
    if len(solution.statuses) != expected_n:
        raise ValueError(
            f"setup_and_solve_batch: solver returned {len(solution.statuses)} "
            f"statuses for B={expected_n}"
        )
    valid_statuses = {SolveStatus.SAT, SolveStatus.UNSAT, SolveStatus.UNKNOWN}
    unexpected = [status for status in solution.statuses if status not in valid_statuses]
    if unexpected:
        raise ValueError(
            f"setup_and_solve_batch: unexpected solver statuses {unexpected}"
        )
    if solution.max_viol.shape != (expected_n,):
        raise ValueError(
            f"setup_and_solve_batch: max_viol shape "
            f"{tuple(solution.max_viol.shape)} != ({expected_n},)"
        )
    return solution


@torch.no_grad()
def verify_lp_batched(
    net,
    solver_factory: Callable[[], Solver],
    timelimit: Optional[float] = None,
) -> List[VerifyResult]:
    """[BATCHED-API] Run one native batched LP verification pass.

    The ACT net supplies a batched INPUT_SPEC seed ``[B, *input_shape]`` and a
    batched ASSERT layer. ``setup_and_solve_batch`` solves all B LPs at once;
    this function decodes each solver lane to a ``VerifyResult`` and validates
    SAT candidates concretely before reporting FALSIFIED.
    """
    import importlib

    _, seed_bounds, batch_size = _setup_verify_context(net)
    if seed_bounds.ub.dim() < 2:
        raise ValueError(
            f"verify_lp_batched: seed bounds must be [B, *input_shape], "
            f"got lb={tuple(seed_bounds.lb.shape)} ub={tuple(seed_bounds.ub.shape)}"
        )
    solver = solver_factory()
    solution = setup_and_solve_batch(
        net,
        Bounds(seed_bounds.lb.clone(), seed_bounds.ub.clone()),
        solver,
        timelimit=timelimit,
    )
    if len(solution.statuses) != batch_size:
        raise ValueError(
            f"verify_lp_batched: solver returned {len(solution.statuses)} "
            f"statuses for B={batch_size}"
        )
    if solution.x.dim() != 2 or solution.x.shape[0] != batch_size:
        raise ValueError(
            f"verify_lp_batched: solution.x must be [B, nvars], got "
            f"shape={tuple(solution.x.shape)} for B={batch_size}"
        )

    input_ids = get_input_ids(net)
    input_index = torch.tensor(input_ids, device=solution.x.device, dtype=torch.long)
    x_candidates = solution.x.index_select(1, input_index).reshape_as(seed_bounds.lb)
    assert_layer = get_assert_layer(net)

    sat_mask = torch.tensor(
        [status in (SolveStatus.SAT, "FEASIBLE") for status in solution.statuses],
        device=x_candidates.device,
        dtype=torch.bool,
    )
    violations = torch.zeros(batch_size, device=x_candidates.device, dtype=torch.bool)
    if bool(sat_mask.any().item()):
        violation_module = importlib.import_module("act.back_end.bab.violation")
        sat_idx = torch.where(sat_mask)[0]
        checked_sat = violation_module.check_violations_batched(
            net, x_candidates.index_select(0, sat_idx), assert_layer,
        )
        if checked_sat.shape != (int(sat_idx.numel()),):
            raise ValueError(
                f"verify_lp_batched: check_violations_batched returned "
                f"shape={tuple(checked_sat.shape)} expected ({int(sat_idx.numel())},)"
            )
        violations.scatter_(
            0, sat_idx, checked_sat.to(device=x_candidates.device, dtype=torch.bool),
        )

    results: List[VerifyResult] = []
    x_cpu = x_candidates.detach().cpu()
    max_viol_cpu = solution.max_viol.detach().cpu()
    for lane, status in enumerate(solution.statuses):
        metadata: Dict[str, Any] = {
            "lane": lane,
            "B": batch_size,
            "solver_status": status,
            "max_viol": float(max_viol_cpu[lane].item()),
        }
        if status in (SolveStatus.SAT, "FEASIBLE"):
            if bool(violations[lane].item()):
                results.append(
                    VerifyResult(
                        VerifyStatus.FALSIFIED,
                        counterexample=x_cpu[lane].clone(),
                        metadata=metadata,
                    )
                )
            else:
                metadata["validation"] = "no_verified_violation"
                results.append(VerifyResult(VerifyStatus.UNKNOWN, metadata=metadata))
        elif status in (SolveStatus.UNSAT, "INFEASIBLE"):
            results.append(VerifyResult(VerifyStatus.CERTIFIED, metadata=metadata))
        elif status == "TIMEOUT":
            results.append(VerifyResult(VerifyStatus.TIMEOUT, metadata=metadata))
        elif status == SolveStatus.UNKNOWN:
            results.append(VerifyResult(VerifyStatus.UNKNOWN, metadata=metadata))
        else:
            raise ValueError(f"verify_lp_batched: unexpected solver status {status!r}")
    return results


# -----------------------------------------------------------------------------
# Single-shot verification
# -----------------------------------------------------------------------------


def _get_output_layer_bounds(net, after: Dict[int, Fact]) -> Bounds:
    """Return the Bounds tensor produced by the network's output layer.

    The output layer is the unique predecessor of the ASSERT layer; the
    returned Bounds is shaped ``[B, n_out]``.
    """
    assert_layer = get_assert_layer(net)
    pred_ids = net.preds.get(assert_layer.id, [])
    if len(pred_ids) != 1:
        raise ValueError(
            f"ASSERT layer {assert_layer.id} must have exactly one "
            f"predecessor (the network output), got predecessors={pred_ids}"
        )
    return after[pred_ids[0]].bounds


@overload
def verify_once(
    net,
    *,
    model_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    timelimit: Optional[float] = None,
    hybridz_tolerance: Optional[float] = None,
) -> List[VerifyResult]:
    ...


@overload
def verify_once(
    net,
    *,
    model_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    collect_facts: Literal[False],
    timelimit: Optional[float] = None,
    hybridz_tolerance: Optional[float] = None,
) -> List[VerifyResult]:
    ...


@overload
def verify_once(
    net,
    *,
    model_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    collect_facts: Literal[True],
    timelimit: Optional[float] = None,
    hybridz_tolerance: Optional[float] = None,
) -> Tuple[List[VerifyResult], Optional[Dict[int, Any]]]:
    ...


@torch.no_grad()
def verify_once(
    net,
    *,
    model_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    collect_facts: bool = False,
    timelimit: Optional[float] = None,
    hybridz_tolerance: Optional[float] = None,
) -> List[VerifyResult] | Tuple[List[VerifyResult], Optional[Dict[int, Any]]]:
    """Single-shot, pure-tensor batched verifier.

    Pipeline:

      1. Seed bounds from INPUT_SPEC layers (already shaped ``[B, *input_shape]``).
      2. ``analyze`` propagates batched bounds through every layer.
      3. Read pre-encoded ``C`` / ``thresholds`` / ``M`` from the ASSERT
         layer params (encoding lives in ``OutputSpec.encode_linear`` on the
         front-end; verify_once does no kind-dispatch).
      4. INTERVAL CERTIFICATION: in one tensor pass, compute the
         per-row interval upper bound of ``C @ y`` and compare to the
         per-lane threshold; ALL of a sample's M lanes must pass for that
         sample to be CERTIFIED.
      5. CONCRETE FALSIFICATION (only if ``model_fn`` given): evaluate the
         model at the box centre; any sample where a lane's concrete
         margin meets-or-exceeds the threshold is FALSIFIED.
      6. Remaining samples are UNKNOWN.

    Args:
        net: an ACT ``Net`` whose first layer is INPUT, last layer is ASSERT,
            and whose INPUT_SPEC layers carry already-batchified
            ``[B, *input_shape]`` lb/ub.
        model_fn: optional callable mapping ``x: [B, *input_shape] ->
            [B, n_out]`` for concrete falsification. If omitted, the
            FALSIFIED status is never produced (FALSIFIED requires evidence).
        collect_facts: when true, return the verifier results together with
            the fact map used by validation: analyze() ``after`` facts for the
            interval/hybridz path, or dual pre-activation forward bounds for the
            dual path.
        timelimit: optional HybridZ verdict-solver wall-clock limit in seconds.
        hybridz_tolerance: optional HybridZ MILP feasibility/spec tolerance.

    Returns:
        ``List[VerifyResult]`` of length ``B`` (one per input lane), or
        ``(results, facts_or_none)`` when ``collect_facts`` is true. Each
        result carries ``status`` plus a ``metadata['lane'] = i`` and any
        ``counterexample`` (a ``torch.Tensor`` of shape ``[*input_shape]``)
        for FALSIFIED lanes.
    """
    from act.back_end.analyze import analyze
    from act.back_end.transfer_functions import get_transfer_function

    # 1. Extract structure and seed.
    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    output_ids = get_output_ids(net)
    spec_layers, seed_bounds, B = _setup_verify_context(net)
    assert_layer = get_assert_layer(net)

    # Standalone solver modes own their verdict logic; the interval/LP path
    # below remains authoritative for non-standalone solver modes.
    from act.back_end.transfer_functions import (
        ensure_active_tf,
        is_dual_solver_active,
        is_hybridz_solver_active,
    )
    active_tf = ensure_active_tf("interval")
    is_dual = is_dual_solver_active()
    is_hybridz = is_hybridz_solver_active()

    out_spec = None
    if is_dual or is_hybridz:
        def _unbatch(val: Any) -> Any:
            # ASSERT params are pre-batchified; OutputSpec expects the
            # canonical unbroadcasted property while y_true remains per lane.
            if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == B:
                return val[0]
            return val

        out_spec = OutputSpec(
            kind=assert_layer.params.get("kind"),
            c=_unbatch(assert_layer.params.get("c")),
            d=_unbatch(assert_layer.params.get("d")),
            y_true=assert_layer.params.get("y_true"),
            margin=_unbatch(assert_layer.params.get("margin")),
            lb=_unbatch(assert_layer.params.get("lb")),
            ub=_unbatch(assert_layer.params.get("ub")),
        )

    if is_dual:
        from act.back_end.solver.solver_dual import DualSolver
        num_classes = len(output_ids)
        # DualSolver is now self-contained: no tf parameter, evaluate_spec
        # computes its own pre-activation forward bounds internally from the net.
        solver = DualSolver()
        result = solver.evaluate_spec(
            net,
            out_spec,
            num_classes=num_classes,
            collect_bounds=collect_facts,
        )
        results = result.to_verify_results()
        if collect_facts:
            return results, solver.last_forward_bounds
        return results

    # 2. Build entry_fact (with all INPUT_SPEC constraints) and analyze.
    entry_fact = Fact(bounds=seed_bounds, cons=ConSet())
    add_all_input_specs(entry_fact.cons, input_ids, spec_layers)
    _before, after, _globalC = analyze(net, entry_id, entry_fact)

    # 3. Pull output bounds (pre-ASSERT layer's Fact).
    output_bounds = _get_output_layer_bounds(net, after)
    output_lb = output_bounds.lb
    output_ub = output_bounds.ub
    if output_lb.dim() != 2 or output_lb.shape[0] != B:
        raise ValueError(
            f"verify_once: output bounds must be [B={B}, n_out], got "
            f"shape={tuple(output_lb.shape)}. Some TF op on this network's "
            f"path collapsed the leading batch dimension."
        )
    n_out = output_lb.shape[1]
    if n_out != len(output_ids):
        raise ValueError(
            f"verify_once: output_lb has n_out={n_out} but ASSERT.in_vars "
            f"has length {len(output_ids)}"
        )
    device = output_lb.device
    dtype = output_lb.dtype

    if is_hybridz:
        from act.back_end.hybridz_tf import HybridzTF
        from act.back_end.solver.solver_hz import HZSolver

        output_layer_id = net.preds[assert_layer.id][0]
        output_hz = None
        input_hz = None
        exact_box_input = (
            len(spec_layers) == 1
            and spec_layers[0].params.get("kind") in (InKind.BOX, InKind.LINF_BALL)
        )
        if isinstance(active_tf, HybridzTF):
            sparse_output = active_tf.get_sparse_hz(output_layer_id)
            output_hz = sparse_output
            if sparse_output is None:
                output_hz = active_tf.get_hz(output_layer_id)
            if exact_box_input and sparse_output is not None:
                output_frame = sparse_output.frame_id
                for layer in reversed(spec_layers):
                    candidate = active_tf.get_sparse_hz(layer.id)
                    if (
                        candidate is not None
                        and candidate.frame_id == output_frame
                        and candidate.n_out == seed_bounds.lb.numel()
                    ):
                        input_hz = candidate
                        break
        solver = HZSolver(
            time_limit=30.0 if timelimit is None else timelimit,
            tolerance=1e-7 if hybridz_tolerance is None else hybridz_tolerance,
        )
        assert out_spec is not None
        results = solver.evaluate_spec(
            output_hz,
            out_spec,
            batch_size=B,
            n_out=n_out,
            input_hz=input_hz,
            input_shape=tuple(seed_bounds.lb.shape),
            timelimit=timelimit,
        )
        for result in results:
            if result.counterexample is not None:
                result.counterexample = result.counterexample.to(
                    device=seed_bounds.lb.device,
                    dtype=seed_bounds.lb.dtype,
                )
        return (results, after) if collect_facts else results

    # 4. Read pre-encoded ASSERT params (produced by OutputSpec.encode_linear
    # at FE construction time). Dispatch on ``kind`` because UNSAFE_LINEAR
    # has EXISTS-row safety semantics while the four other kinds (LINEAR_LE,
    # TOP1_ROBUST, MARGIN_ROBUST, RANGE) share an ALL-rows form.
    C = assert_layer.params["C"].to(device=device, dtype=dtype)
    thresholds = assert_layer.params["thresholds"].to(device=device, dtype=dtype)
    M = int(assert_layer.params["M"])
    kind = assert_layer.params.get("kind")
    is_unsafe_linear = kind == OutKind.UNSAFE_LINEAR
    assert C.dim() == 2 and C.shape == (B * M, n_out), (
        f"verify_once: ASSERT params['C'].shape={tuple(C.shape)} "
        f"expected ({B * M}, {n_out})"
    )
    assert thresholds.shape == (B, M), (
        f"verify_once: ASSERT params['thresholds'].shape="
        f"{tuple(thresholds.shape)} expected ({B}, {M})"
    )

    C_pos = C.clamp(min=0)
    C_neg = C.clamp(max=0)
    lb_exp = output_lb.repeat_interleave(M, dim=0)
    ub_exp = output_ub.repeat_interleave(M, dim=0)

    if is_unsafe_linear:
        # UNSAFE polytope = {y : C y <= d}. Property is SAFE iff for all y in
        # the box, EXISTS row i with c_i @ y > d_i (i.e. y leaves the polytope
        # on row i). Sound under-approximation: EXISTS row i such that
        # min_{y in box} (c_i @ y) > d_i. min(c_i @ y) = c_i_pos @ lb + c_i_neg @ ub.
        margin_min = (C_pos * lb_exp + C_neg * ub_exp).sum(dim=-1)
        certified = (margin_min.view(B, M) > thresholds).any(dim=-1)
    else:
        # LINEAR_LE / TOP1_ROBUST / MARGIN_ROBUST / RANGE: certified iff for
        # all y in the box, ALL rows max_y (c_i @ y) < d_i.
        margin_max = (C_pos * ub_exp + C_neg * lb_exp).sum(dim=-1)
        certified = (margin_max.view(B, M) < thresholds).all(dim=-1)

    # 5. Concrete falsification (optional).
    falsified = torch.zeros(B, dtype=torch.bool, device=device)
    counterexamples: List[Optional[torch.Tensor]] = [None] * B
    if model_fn is not None:
        x_center = 0.5 * (seed_bounds.lb + seed_bounds.ub)
        y_concrete = model_fn(x_center)
        if y_concrete.dim() != 2 or y_concrete.shape != (B, n_out):
            raise ValueError(
                f"verify_once: model_fn returned shape "
                f"{tuple(y_concrete.shape)}, expected ({B}, {n_out})"
            )
        y_concrete = y_concrete.to(device=device, dtype=dtype)
        C_view = C.view(B, M, n_out)
        concrete_violation = torch.einsum("bmn,bn->bm", C_view, y_concrete)
        if is_unsafe_linear:
            # Concrete y is in the UNSAFE polytope iff ALL rows c_i @ y <= d_i;
            # that is the violation condition for UNSAFE_LINEAR.
            falsified = (~certified) & (
                (concrete_violation <= thresholds).all(dim=-1)
            )
        else:
            # ALL-rows kinds: FALSIFIED iff ANY lane's concrete margin
            # meets-or-exceeds threshold.
            falsified = (~certified) & (
                (concrete_violation >= thresholds).any(dim=-1)
            )
        if falsified.any():
            x_center_cpu = x_center.detach().cpu()
            # B1 (oracle-verified): single sync via .tolist() replaces B per-element .item() syncs.
            # torch.where returns ascending indices; lane order is preserved.
            for i in torch.where(falsified)[0].tolist():
                counterexamples[i] = x_center_cpu[i].clone()

    # 6. Assemble per-lane results.
    results: List[VerifyResult] = []
    cert_list = certified.tolist()
    fals_list = falsified.tolist()
    for i in range(B):
        meta: Dict[str, Any] = {"lane": i, "B": B, "M": M}
        if cert_list[i]:
            results.append(
                VerifyResult(VerifyStatus.CERTIFIED, metadata=meta)
            )
        elif fals_list[i]:
            results.append(
                VerifyResult(
                    VerifyStatus.FALSIFIED,
                    counterexample=counterexamples[i],
                    metadata=meta,
                )
            )
        else:
            results.append(
                VerifyResult(VerifyStatus.UNKNOWN, metadata=meta)
            )
    return (results, after) if collect_facts else results
