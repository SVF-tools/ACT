#!/usr/bin/env python3
#===- act/pipeline/verification/per_neuron_bounds.py - Per-Neuron Bounds --====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Level-2 (per-neuron) numerical validation. This module checks that ACT’s
#   abstract bounds (lb/ub) over-approximate the concrete activations produced
#   by a reference PyTorch forward pass, neuron-by-neuron, for a single input.
#
# Key Features:
#   - Bounds extraction:
#       Reuses verifier-produced per-layer Bounds(lb, ub): analyze() facts for
#       interval/hybridz, and dual pre-activation forward bounds for dual.
#   - Concrete activation tracing (hook-based):
#       Captures hookable PyTorch module outputs, or activation inputs when
#       comparing against dual pre-activation bounds.
#   - Built-in alignment to ACT layer IDs:
#       Aligns hook events to ACT layers using a strict hookable-order strategy
#       (with optional shape sanity checks from ACT layer params).
#   - Per-neuron violation detection:
#       A neuron is flagged when the concrete activation falls outside [lb, ub]
#       beyond the small floating-point noise tolerance used by the checker.
#   - Debug-oriented reporting:
#       Computes per-layer statistics and returns the top-K worst violations
#       (largest gaps) for fast bug localization.
#
# Pipeline:
#   once per net    : validate() → bounds_from_facts(verifier facts)
#                       → precomputed_bounds (shared by all samples)
#   once per sample : run_per_neuron_bounds_check(input_tensor, precomputed_bounds)
#     → collect_concrete_activations()          : hooks → concrete_by_layer + meta
#     → compare_bounds_per_neuron()             : gaps/violations/topk report
#
# Numerical Policy:
#   - Zero tolerance by default:
#       gap = max(lb - a, a - ub); violation iff gap > tol_abs + tol_rel * max(|lb|, |ub|)
#       with PerNeuronCheckConfig defaults tol_abs = tol_rel = 0 (any deviation is
#       unsound; FP noise is the transfer functions' outward-rounding job). The
#       CLI resolves --bounds-tolerance 'auto' to the 100-ulp dtype noise floor;
#       pass '0,0' there for strict zero.
#       Reversed intervals (lb > ub) are still surfaced by the same gap test.
#   - nan_policy="error":
#       Any NaN/Inf encountered in concrete or bounds yields ERROR status.
#   - topk:
#       When violations occur, returns the K most severe violating neurons
#       (largest gap) to simplify debugging.
#
# Outputs (dict):
#   - status: PASS / FAIL / ERROR
#   - violations_total: total number of violating neurons
#   - violations_topk: list of worst-K violations (layer_id, neuron_index, gap, ...)
#   - layerwise_stats: per-layer summary (num_violations, max_gap, mean_gap, ranges)
#   - alignment: meta describing the alignment mode and event/layer counts
#   - total_checks: total number of neurons compared
#   - worst_gap: maximum gap observed across all layers
#
# Usage:
#   result = run_per_neuron_bounds_check(
#       act_net=act_net,
#       model=torch_model,
#       input_tensor=x,
#       config=PerNeuronCheckConfig(topk=10),
#       precomputed_bounds=bounds_by_layer,
#   )
#
# Design Notes:
#   - Alignment is strict by design: mismatches are surfaced as explicit errors
#     (kind/type/shape) rather than silently producing incorrect matches.
#   - Only “hookable” modules are traced to keep the activation stream stable
#     and comparable to ACT layer kinds.
#
#===---------------------------------------------------------------------===#


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from act.back_end.core import Bounds, Layer
from act.back_end.layer_schema import (
    TRANSFORMER_KINDS,
    LayerKind,
)


_ACT_KIND_TO_MODULE = {
    LayerKind.DENSE.value: "Linear",
    LayerKind.CONV1D.value: "Conv1d",
    LayerKind.CONV2D.value: "Conv2d",
    LayerKind.CONV3D.value: "Conv3d",
    LayerKind.RELU.value: "ReLU",
    LayerKind.SIGMOID.value: "Sigmoid",
    LayerKind.TANH.value: "Tanh",
    LayerKind.SILU.value: "SiLU",
    LayerKind.LRELU.value: "LeakyReLU",
    LayerKind.SIN.value: "ACTSin",
    LayerKind.COS.value: "ACTCos",
    LayerKind.ERF.value: "ACTErf",
    LayerKind.SQRT.value: "ACTSqrt",
    LayerKind.QUANTIZE.value: "ACTQuantize",
    LayerKind.ATT_SCORES.value: "ACTAttentionScores",
    LayerKind.ATT_MIX.value: "ACTAttentionMix",
    LayerKind.MHA_SPLIT.value: "ACTMHASplit",
    LayerKind.MHA_JOIN.value: "ACTMHAJoin",
    LayerKind.SOFTMAX.value: "Softmax",
    LayerKind.LAYERNORM.value: "ACTLayerNorm",
    LayerKind.GELU.value: "GELU",
    LayerKind.MASK_ADD.value: "ACTMaskAdd",
    LayerKind.MATMUL.value: "ACTMatMul",
    LayerKind.FLATTEN.value: "Flatten",
    LayerKind.MAXPOOL1D.value: "MaxPool1d",
    LayerKind.MAXPOOL2D.value: "MaxPool2d",
    LayerKind.MAXPOOL3D.value: "MaxPool3d",
    LayerKind.AVGPOOL1D.value: "AvgPool1d",
    LayerKind.AVGPOOL2D.value: "AvgPool2d",
    LayerKind.AVGPOOL3D.value: "AvgPool3d",
    LayerKind.ADAPTIVEAVGPOOL2D.value: "AdaptiveAvgPool2d",
}

# ``DualSolver`` calls ``compute_forward_bounds(post_activation=False)``.  Its
# nonlinear relaxation handlers therefore store the incoming (pre-activation)
# box, while affine, shape, pooling, and bilinear handlers store their output
# box.  Keep that distinction in ACT-layer terms: module names are only a
# tracing detail and do not define which tensor a verifier bound represents.
_PRE_ACTIVATION_BOUND_KINDS = frozenset(
    {
        LayerKind.RELU.value,
        LayerKind.SIGMOID.value,
        LayerKind.TANH.value,
        LayerKind.SILU.value,
        LayerKind.LRELU.value,
        LayerKind.SIN.value,
        LayerKind.COS.value,
        LayerKind.ERF.value,
        LayerKind.SQRT.value,
        LayerKind.QUANTIZE.value,
        LayerKind.SOFTMAX.value,
        LayerKind.LAYERNORM.value,
        LayerKind.GELU.value,
    }
)

_POST_ACTIVATION_BOUND_KINDS = frozenset(
    {
        LayerKind.DENSE.value,
        LayerKind.CONV1D.value,
        LayerKind.CONV2D.value,
        LayerKind.CONV3D.value,
        LayerKind.ATT_SCORES.value,
        LayerKind.ATT_MIX.value,
        LayerKind.MHA_SPLIT.value,
        LayerKind.MHA_JOIN.value,
        LayerKind.MASK_ADD.value,
        LayerKind.MATMUL.value,
        LayerKind.FLATTEN.value,
        LayerKind.MAXPOOL1D.value,
        LayerKind.MAXPOOL2D.value,
        LayerKind.MAXPOOL3D.value,
        LayerKind.AVGPOOL1D.value,
        LayerKind.AVGPOOL2D.value,
        LayerKind.AVGPOOL3D.value,
        LayerKind.ADAPTIVEAVGPOOL2D.value,
    }
)

assert not (_PRE_ACTIVATION_BOUND_KINDS & _POST_ACTIVATION_BOUND_KINDS)
assert (
    _PRE_ACTIVATION_BOUND_KINDS | _POST_ACTIVATION_BOUND_KINDS
) == frozenset(_ACT_KIND_TO_MODULE)


def check_hookable_alignment(act_net, model: torch.nn.Module) -> Optional[str]:
    """Return a Level-2 skip reason for structurally un-alignable models.

    Ordinary mismatches intentionally return ``None`` so the strict hookable-order
    path continues to surface them as hard errors.
    """
    hookable_kinds = set(_ACT_KIND_TO_MODULE.values())
    hookable_modules = sum(
        1
        for module in model.modules()
        if module is not model and module.__class__.__name__ in hookable_kinds
    )
    layers = getattr(act_net, "layers", [])
    hookable_layers = sum(
        1 for layer in layers if _ACT_KIND_TO_MODULE.get(layer.kind) in hookable_kinds
    )

    if hookable_layers == 0:
        return (
            "reference model and ACT net expose no 1:1 hookable layers; "
            "per-neuron bounds were not checked"
        )

    if hookable_modules == hookable_layers:
        return None

    layer_kinds = {getattr(layer, "kind", None) for layer in layers}
    if layer_kinds & TRANSFORMER_KINDS:
        return (
            "transformer lowering is not 1:1 with torch modules "
            f"(hookable events={hookable_modules} vs layers={hookable_layers}); "
            "Level-2 alignment pending — Level 1 still enforced"
        )

    if hookable_modules == 0 and hookable_layers > 0:
        return (
            "reference model exposes no hookable torch modules "
            "(ONNX-converted graph?); per-neuron alignment not applicable"
        )

    return None


def bounds_from_facts(act_net, after) -> Tuple[Dict[int, Bounds], List[str]]:
    raw_bounds: Dict[int, Bounds] = {}
    for lid, fact_or_bounds in after.items():
        raw_bounds[int(lid)] = (
            fact_or_bounds
            if isinstance(fact_or_bounds, Bounds)
            else fact_or_bounds.bounds
        )
    return _validate_bounds_by_layer(act_net, raw_bounds)


def _validate_bounds_by_layer(
    act_net,
    raw_bounds: Dict[int, Bounds],
) -> Tuple[Dict[int, Bounds], List[str]]:
    """Apply the common layer-presence, shape, and finite checks to bounds."""
    errors: List[str] = []
    bounds_by_layer: Dict[int, Bounds] = {}

    for layer in getattr(act_net, "layers", []):
        lid = layer.id
        if lid not in raw_bounds:
            errors.append(f"Missing bounds for layer_id={lid} (kind={layer.kind})")
            continue
        bounds = raw_bounds[lid]
        lb = bounds.lb
        ub = bounds.ub
        if lb.shape != ub.shape:
            errors.append(
                f"Bounds shape mismatch at layer_id={lid}: lb={tuple(lb.shape)} ub={tuple(ub.shape)}"
            )
            continue
        if not torch.isfinite(lb).all() or not torch.isfinite(ub).all():
            errors.append(f"Non-finite bounds at layer_id={lid}")
            continue
        bounds_by_layer[lid] = Bounds(lb=lb, ub=ub)

    return bounds_by_layer, errors


def sample_inputs_from_spec(
    act_net,
    num_samples: int,
    *,
    device,
    dtype,
    seed: int = 42,
) -> List[torch.Tensor]:
    """Sample deterministic concrete inputs from ACT INPUT_SPEC layers."""
    from act.back_end.verifier import gather_input_spec_layers, seed_from_input_specs

    spec_layers = gather_input_spec_layers(act_net)
    seed_bounds = seed_from_input_specs(spec_layers)
    lb = seed_bounds.lb.to(device=device, dtype=dtype)
    ub = seed_bounds.ub.to(device=device, dtype=dtype)
    if lb.dim() < 2:
        lb = lb.unsqueeze(0)
        ub = ub.unsqueeze(0)

    samples: List[torch.Tensor] = []
    if num_samples <= 0:
        return samples

    center = lb + 0.5 * (ub - lb)
    samples.append(center)
    generator = torch.Generator(device=lb.device)
    generator.manual_seed(seed)
    for _ in range(1, num_samples):
        rand = torch.rand(lb.shape, device=lb.device, dtype=lb.dtype, generator=generator)
        samples.append(lb + rand * (ub - lb))
    geometry = act_net.input_geometry
    if geometry is not None:
        samples = [geometry.project(x) for x in samples]
    return samples


def collect_concrete_activations(
    act_net,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    *,
    strict_single_call_per_module: bool = False,
    pre_activation: bool = False,
) -> Tuple[Dict[int, torch.Tensor], List[str], List[str], Dict[str, Any]]:
    """
    Collect concrete activations and align them to ACT layer IDs.
    """
    errors: List[str] = []
    warnings: List[str] = []
    call_counts: Dict[int, int] = {}
    hookable_events: List[Tuple[str, Any, Any]] = []
    hooks = []

    def _hook(module, inputs, output):
        module_id = id(module)
        call_counts[module_id] = call_counts.get(module_id, 0) + 1
        if strict_single_call_per_module and call_counts[module_id] > 1:
            errors.append(f"Module called multiple times: {module.__class__.__name__}")
        module_type = module.__class__.__name__
        module_input = inputs[0] if inputs else None
        hookable_events.append((module_type, module_input, output))

    hookable_kinds = set(_ACT_KIND_TO_MODULE.values())

    for module in model.modules():
        if module is model:
            continue
        if module.__class__.__name__ in hookable_kinds:
            hooks.append(module.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            model(input_tensor)
    finally:
        for h in hooks:
            h.remove()

    hookable_layers = [
        L for L in getattr(act_net, "layers", [])
        if _ACT_KIND_TO_MODULE.get(L.kind) in hookable_kinds
    ]

    if len(hookable_events) != len(hookable_layers):
        errors.append(
            f"Hookable count mismatch: events={len(hookable_events)} layers={len(hookable_layers)}"
        )

    def _numel(shape: Tuple[int, ...]) -> int:
        prod = 1
        for s in shape:
            prod *= int(s)
        return int(prod)

    def _drop_batch_if_and_only_if_batch1(
        raw_shape: Tuple[int, ...],
        expected_shape: Tuple[int, ...] | None,
    ) -> Tuple[Tuple[int, ...], bool, str]:
        """Match per-sample shape: raw and expected may carry any leading
        batch dim (B=1 or B>1). Comparison strips the leading dim from both
        sides if their per-sample ranks line up."""
        if expected_shape is None:
            return raw_shape, False, "expected_shape_missing"
        if not raw_shape:
            return raw_shape, False, "raw_shape_empty"
        if len(raw_shape) == len(expected_shape) + 1:
            candidate = tuple(raw_shape[1:])
            if candidate == expected_shape:
                return candidate, True, "dropped_batch"
            return raw_shape, False, "drop_would_not_match_expected"
        if len(raw_shape) == len(expected_shape):
            if tuple(raw_shape[1:]) == tuple(expected_shape[1:]):
                return raw_shape, True, "per_sample_matched"
            return raw_shape, False, "per_sample_shape_mismatch"
        return raw_shape, False, "rank_mismatch"

    mapping: Dict[int, torch.Tensor] = {}

    for idx, layer in enumerate(hookable_layers):
        if idx >= len(hookable_events):
            break
        module_type, module_input, module_output = hookable_events[idx]
        expected = _ACT_KIND_TO_MODULE.get(layer.kind)
        if expected is None:
            errors.append(
                f"Unsupported ACT kind at position {idx}: act_kind={layer.kind}"
            )
        elif expected != module_type:
            errors.append(
                f"Kind/type mismatch at position {idx}: act_kind={layer.kind} event_type={module_type}"
            )
        tensor_source = (
            module_input
            if pre_activation and layer.kind in _PRE_ACTIVATION_BOUND_KINDS
            else module_output
        )
        if not torch.is_tensor(tensor_source):
            errors.append(
                f"Non-tensor activation at layer_id={layer.id}: "
                f"kind={layer.kind} module={module_type}"
            )
            continue
        tensor = tensor_source.detach()
        expected_shape = None

        params = getattr(layer, "params", {}) or {}
        if "output_shape" in params:
            expected_shape = tuple(int(x) for x in params["output_shape"])
        elif "shape" in params:
            expected_shape = tuple(int(x) for x in params["shape"])
        if expected_shape is not None:
            raw_shape = tuple(int(x) for x in tensor.shape)
            no_batch_shape, dropped, drop_reason = _drop_batch_if_and_only_if_batch1(
                raw_shape,
                expected_shape,
            )
            if not dropped:
                ev_numel = _numel(raw_shape)
                exp_numel = _numel(expected_shape)
                if ev_numel != exp_numel:
                    errors.append(
                        f"Shape mismatch at layer_id={layer.id}: "
                        f"event_raw={raw_shape} event_no_batch={no_batch_shape} "
                        f"expected={expected_shape} "
                        f"dropped_batch={dropped} drop_reason={drop_reason} "
                        f"event_numel={ev_numel} expected_numel={exp_numel}"
                    )
        mapping[layer.id] = tensor

    info = {
        "mode": "hookable_order_strict_pre_activation" if pre_activation else "hookable_order_strict",
        "hookable_events": len(hookable_events),
        "hookable_layers": len(hookable_layers),
    }

    return mapping, errors, warnings, info


def _is_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


def compare_bounds_per_neuron(
    *,
    bounds_by_layer: Dict[int, Bounds],
    concrete_by_layer: Dict[int, torch.Tensor],
    layer_by_id: Dict[int, Layer],
    topk: int = 10,
    nan_policy: str = "error",
    tol_abs: float = 0.0,
    tol_rel: float = 0.0,
) -> Dict[str, Any]:
    """
    Compare per-neuron concrete activations against abstract bounds. Zero
    tolerance by default; a nonzero tol_abs/tol_rel must be passed explicitly.
    """
    errors: List[str] = []
    warnings: List[str] = []
    violations_topk: List[Dict[str, Any]] = []
    layerwise_stats: List[Dict[str, Any]] = []
    violations_total = 0

    if set(bounds_by_layer.keys()) != set(concrete_by_layer.keys()):
        missing = set(bounds_by_layer.keys()) - set(concrete_by_layer.keys())
        extra = set(concrete_by_layer.keys()) - set(bounds_by_layer.keys())
        errors.append(f"Layer key mismatch: missing={sorted(missing)} extra={sorted(extra)}")

    if errors:
        return {
            "status": "ERROR",
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "errors": errors,
            "warnings": warnings,
        }

    candidates: List[Dict[str, Any]] = []

    for layer_id, bounds in bounds_by_layer.items():
        concrete = concrete_by_layer[layer_id]
        layer = layer_by_id.get(layer_id)
        kind = layer.kind if layer is not None else "UNKNOWN"
        lb = bounds.lb
        ub = bounds.ub

        if nan_policy == "error":
            if not _is_finite(concrete) or not _is_finite(lb) or not _is_finite(ub):
                errors.append(f"Non-finite value at layer_id={layer_id}")
                continue

        concrete_flat = concrete.reshape(-1)
        lb_flat = lb.reshape(-1)
        ub_flat = ub.reshape(-1)
        if concrete_flat.numel() != lb_flat.numel():
            errors.append(
                f"Shape mismatch at layer_id={layer_id}: "
                f"concrete_numel={concrete_flat.numel()} bounds_numel={lb_flat.numel()}"
            )
            continue

        diff_low = lb_flat - concrete_flat
        diff_high = concrete_flat - ub_flat
        gap = torch.maximum(diff_low, diff_high)
        gap = torch.clamp(gap, min=0.0)

        tol = tol_abs + tol_rel * torch.maximum(lb_flat.abs(), ub_flat.abs())
        violations_mask = gap > tol
        num_violations = int(violations_mask.sum().item())
        violations_total += num_violations

        if num_violations > 0:
            gap_vals = gap[violations_mask]
            max_gap = float(gap_vals.max().item())
            mean_gap = float(gap_vals.mean().item())
        else:
            max_gap = 0.0
            mean_gap = 0.0

        layerwise_stats.append(
            {
                "layer_id": int(layer_id),
                "kind": kind,
                "shape": list(lb.shape),
                "num_neurons": int(concrete_flat.numel()),
                "num_violations": int(num_violations),
                "max_gap": float(max_gap),
                "mean_gap": float(mean_gap),
                "lb_min": float(lb_flat.min().item()) if lb_flat.numel() > 0 else 0.0,
                "lb_max": float(lb_flat.max().item()) if lb_flat.numel() > 0 else 0.0,
                "ub_min": float(ub_flat.min().item()) if ub_flat.numel() > 0 else 0.0,
                "ub_max": float(ub_flat.max().item()) if ub_flat.numel() > 0 else 0.0,
                "concrete_min": float(concrete_flat.min().item()) if concrete_flat.numel() > 0 else 0.0,
                "concrete_max": float(concrete_flat.max().item()) if concrete_flat.numel() > 0 else 0.0,
                "layer_status": "FAIL" if num_violations > 0 else "PASS",
            }
        )

        if topk > 0:
            k = min(int(topk), int(concrete_flat.numel()))
            if k > 0:
                vals, idxs = torch.topk(gap, k=k)
                for v, i in zip(vals.tolist(), idxs.tolist()):
                    i = int(i)
                    if v <= float(tol[i].item()):
                        continue
                    candidates.append(
                        {
                            "layer_id": int(layer_id),
                            "kind": kind,
                            "neuron_index": i,
                            "gap": float(v),
                            "concrete": float(concrete_flat[i].item()),
                            "lb": float(lb_flat[i].item()),
                            "ub": float(ub_flat[i].item()),
                        }
                    )

    if errors:
        return {
            "status": "ERROR",
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "errors": errors,
            "warnings": warnings,
        }

    candidates.sort(key=lambda x: x["gap"], reverse=True)
    violations_topk = candidates[: int(topk)]

    status = "FAIL" if violations_total > 0 else "PASS"
    return {
        "status": status,
        "violations_total": int(violations_total),
        "violations_topk": violations_topk,
        "layerwise_stats": layerwise_stats,
        "errors": errors,
        "warnings": warnings,
    }

@dataclass(frozen=True)
class PerNeuronCheckConfig:
    """Per-neuron check knobs.

    topk: number of most-severe violating neurons to report per check.
    nan_policy: "error" => NaN/Inf in concrete or bounds yields ERROR status.
    tol_abs / tol_rel: violation threshold ``tol_abs + tol_rel * max(|lb|, |ub|)``.
    Defaults are ZERO (any deviation outside [lb, ub] is unsound). The CLI's
    ``--bounds-tolerance`` defaults to 'auto' (100 ulp of the run dtype) and
    accepts '0,0' for strict zero.
    """

    topk: int = 10
    nan_policy: str = "error"
    tol_abs: float = 0.0
    tol_rel: float = 0.0


def run_per_neuron_bounds_check(
    *,
    act_net,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    config: PerNeuronCheckConfig = PerNeuronCheckConfig(),
    precomputed_bounds: Dict[int, Bounds],
    pre_activation: bool = False,
) -> Dict[str, Any]:
    """
    Full per-neuron bounds validation pipeline for a single input sample.
    """
    errors: List[str] = []
    warnings: List[str] = []

    bounds_by_layer = precomputed_bounds

    concrete_by_layer, event_errors, event_warnings, alignment_meta = collect_concrete_activations(
        act_net,
        model,
        input_tensor,
        pre_activation=pre_activation,
    )
    if event_errors:
        errors.extend(event_errors)
    if event_warnings:
        warnings.extend(event_warnings)

    if errors:
        return {
            "status": "ERROR",
            "errors": errors,
            "warnings": warnings,
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "alignment": alignment_meta,
            "total_checks": 0,
            "worst_gap": 0.0,
        }

    missing_bounds = [lid for lid in concrete_by_layer.keys() if lid not in bounds_by_layer]
    if missing_bounds:
        return {
            "status": "ERROR",
            "errors": [f"Missing bounds for layer_ids={sorted(missing_bounds)}"],
            "warnings": warnings,
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "alignment": alignment_meta,
            "total_checks": 0,
            "worst_gap": 0.0,
        }

    bounds_for_compare = {lid: bounds_by_layer[lid] for lid in concrete_by_layer.keys()}
    compare = compare_bounds_per_neuron(
        bounds_by_layer=bounds_for_compare,
        concrete_by_layer=concrete_by_layer,
        layer_by_id=getattr(act_net, "by_id", {}),
        topk=config.topk,
        nan_policy=config.nan_policy,
        tol_abs=config.tol_abs,
        tol_rel=config.tol_rel,
    )

    if compare.get("status") == "ERROR":
        return {
            "status": "ERROR",
            "errors": compare.get("errors", []),
            "warnings": warnings + compare.get("warnings", []),
            "violations_total": 0,
            "violations_topk": [],
            "layerwise_stats": [],
            "alignment": alignment_meta,
            "total_checks": 0,
            "worst_gap": 0.0,
        }

    layerwise_stats = compare.get("layerwise_stats", [])
    total_checks = sum(int(s.get("num_neurons", 0)) for s in layerwise_stats)
    worst_gap = 0.0
    for s in layerwise_stats:
        worst_gap = max(worst_gap, float(s.get("max_gap", 0.0)))

    compare["alignment"] = alignment_meta
    compare["warnings"] = warnings + compare.get("warnings", [])
    compare["total_checks"] = int(total_checks)
    compare["worst_gap"] = float(worst_gap)
    return compare
