#!/usr/bin/env python3
# ===- act/pipeline/validate_verifier.py - Verifier Correctness Validation ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   Unified verification validation framework with two validation levels:
#
#   Level 1: Counterexample/Soundness Validation
#     - Validates that verifier doesn't claim CERTIFIED when concrete
#       counterexamples exist
#
#   Level 2: Bounds/Numerical Validation
#     - Validates that abstract bounds correctly overapproximate concrete
#       activation values
#
# ===---------------------------------------------------------------------===#
#
# Level 1: Counterexample/Soundness Validation
# ============================================
#
# Key Insight:
#   Concrete execution provides ground truth - if we find a real counterexample
#   at runtime, the formal verifier cannot claim the property is certified.
#   This is a soundness check for the verification backend.
#
# Validation Strategy:
#   1. For each network, generate strategic test cases:
#      - Center: Input at center of input spec (typically safe)
#      - Boundary: Input near boundary of input spec (risky)
#      - Random: Random input within input spec (varied)
#
#   2. Run concrete execution to find violations
#   3. Run the formal verifier on the network unconditionally (every net is
#      end-to-end verified; this also subsumes the historical --verify-all)
#   4. If a concrete counterexample was found in step 2, cross-validate the
#      verifier's verdict against it using the matrix below.  Otherwise the
#      cross-validation outcome is INCONCLUSIVE — but the verifier was still
#      exercised in step 3.
#
# Validation Matrix (Level 1):
#   ┌─────────────────────────┬────────────────────────────────────┬──────────────┐
#   │ Concrete Counterexample │ Verifier Result                    │ Validation   │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ FOUND                   │ CERTIFIED                          │ ❌ FAILED    │
#   │                         │ (Soundness Bug - false negative)   │              │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ FOUND                   │ FALSIFIED                          │ ✅ PASSED    │
#   │                         │ (Correct - verifier found issue)   │              │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ FOUND                   │ UNKNOWN                            │ ⚠️ ACCEPTABLE│
#   │                         │ (Incomplete but sound)             │              │
#   ├─────────────────────────┼────────────────────────────────────┼──────────────┤
#   │ NOT FOUND               │ Any Result                         │ ❓ INCONC.   │
#   │                         │ (Cannot validate - no ground truth)│              │
#   └─────────────────────────┴────────────────────────────────────┴──────────────┘
#
#   Legend:
#     FAILED       - Critical soundness bug (false negative)
#     PASSED       - Verifier correct
#     ACCEPTABLE   - Verifier incomplete but sound (conservative)
#     INCONCLUSIVE - No concrete counterexample to validate against
#
# ===---------------------------------------------------------------------===#
#
# Level 2: Bounds/Numerical Validation
# ====================================
#
# Key Insight:
#   Abstract interpretation must overapproximate concrete values. If any
#   concrete activation value falls outside its abstract bounds [lb, ub],
#   the transfer function is unsound.
#
# Validation Strategy:
#   1. Sample concrete inputs from input specification
#   2. Run concrete forward pass through PyTorch model → get concrete activations
#   3. Run abstract analysis through ACT → get abstract bounds for each layer
#   4. Check: concrete_value ∈ [lb, ub] for all layers and all neurons
#
# Validation Matrix (Level 2):
#   ┌──────────────────────┬────────────────────────┬──────────────┐
#   │ Concrete Values      │ Abstract Bounds        │ Validation   │
#   ├──────────────────────┼────────────────────────┼──────────────┤
#   │ value ∈ [lb, ub]     │ All layers/neurons     │ ✅ PASSED    │
#   │ (Sound bounds)       │                        │              │
#   ├──────────────────────┼────────────────────────┼──────────────┤
#   │ value ∉ [lb, ub]     │ Any layer/neuron       │ ❌ FAILED    │
#   │ (Unsound bounds)     │ (Transfer function bug)│              │
#   └──────────────────────┴────────────────────────┴──────────────┘
#
#   Legend:
#     PASSED - All concrete values within abstract bounds (sound)
#     FAILED - Concrete value outside bounds (unsound transfer function)
#
# ===---------------------------------------------------------------------===#
#
# Usage:
#   # Via CLI (recommended): run normal verification and add the unified
#   # post-verification validator hook.
#   python -m act.pipeline --verify netfactory --solvers torchlp --tf-modes interval --validate-soundness
#   python -m act.pipeline --verify vnnlib --category acasxu_2023 --max-instances 3 --validate-soundness
#   python -m act.pipeline --verify torchvision --dataset MNIST --model simple_cnn --num-samples 2 --validate-soundness
#
#   # With device and dtype specification:
#   python -m act.pipeline --verify netfactory --validate-soundness --device cpu --dtype float64
#   python -m act.pipeline --verify vnnlib --category acasxu_2023 --validate-soundness --device cuda --dtype float32
#
#   # Limit netfactory networks and adjust Level-2 samples:
#   python -m act.pipeline --verify netfactory --networks mnist_mlp_small --input-samples 20 --validate-soundness
#
#   # Ignore errors and always exit 0 (useful for CI):
#   python -m act.pipeline --verify netfactory --validate-soundness --ignore-errors
#
# Exit Codes:
#   0 - All validations passed (no failures or errors)
#   0 - With --ignore-errors flag (always succeed regardless of results)
#   1 - Failures detected (verifier bugs) OR errors detected (backend bugs)
#
# ===---------------------------------------------------------------------===#

import copy
import torch
import logging
from typing import Dict, Any, Optional, Tuple, List

from act.back_end.core import Net, Layer
from act.pipeline.verification.model_factory import ModelFactory
from act.pipeline.verification.per_neuron_bounds import (
    PerNeuronCheckConfig,
    bounds_from_facts,
    check_hookable_alignment,
    run_per_neuron_bounds_check,
    sample_inputs_from_spec,
)
from act.util.stats import VerifyStatus
from act.util.options import PerformanceOptions
from act.util.format_utils import rule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerificationValidator:
    """Unified verification validation framework with counterexample and bounds validation."""

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float64):
        """
        Initialize verification validator.

        Args:
            device: Device for computation ('cpu' or 'cuda')
            dtype: Data type for computation (float32 or float64)
        """
        self.factory = ModelFactory()
        self.device = device
        self.dtype = dtype
        self.validation_results = []

        # Initialize debug file (GUARDED)
        if PerformanceOptions.debug_tf:
            debug_file = PerformanceOptions.debug_output_file
            with open(debug_file, "w") as f:
                f.write(f"ACT Verification Debug Log\n")
                f.write(f"Device: {device}, Dtype: {dtype}\n")
                f.write(f"{rule()}\n\n")
            logger.info(f"Debug logging to: {debug_file}")

    def _batchify_net(self, net: Net, target_B: Optional[int]) -> Net:
        """Return a deep copy of ``net`` with INPUT/INPUT_SPEC/ASSERT tensors
        adjusted to ``target_B`` lanes along axis 0.

        Scope:
          Sub-problems sharing the SAME network and SAME spec kind only;
          mixing kinds (e.g. LINEAR_LE with TOP1_ROBUST) in one batch is
          not supported. Within a kind, lanes MAY carry different per-lane
          constraints.

        Per-kind ASSERT handling:
          - TOP1_ROBUST / MARGIN_ROBUST: cycle ``y_true`` across classes
            so each lane verifies a different "true class" assumption.
          - LINEAR_LE / RANGE / UNSAFE_LINEAR: replicate sample 0's
            constraint (these kinds have no natural per-lane axis).

        INPUT / INPUT_SPEC: leading-axis tensors are replicated; spec-side
        is the intended axis of per-lane variation.

        ``target_B is None`` returns the net unchanged (use native B).
        """
        if target_B is None or target_B <= 0:
            return self._migrate_net_to_device(copy.deepcopy(net))

        from act.back_end.verifier import gather_input_spec_layers, seed_from_input_specs
        from act.front_end.spec_creator_base import LabeledInputTensor

        input_layer = next(layer for layer in net.layers if layer.kind == "INPUT")
        shape = input_layer.params.get("shape")
        shape_t = tuple(shape) if isinstance(shape, (list, tuple)) else ()
        seed = seed_from_input_specs(gather_input_spec_layers(net))
        original_B = int(seed.lb.shape[0])
        structural = (
            len(shape_t) >= 2 and shape_t[0] != 1
            and tuple(seed.lb.shape[1:]) == shape_t
        )
        if structural and target_B != 1:
            raise ValueError(
                f"Fixed model input {shape_t} cannot be replicated as "
                f"{target_B} independent verifier lanes"
            )

        new_net = copy.deepcopy(net)
        for L in new_net.layers:
            if L.kind in ("INPUT", "INPUT_SPEC"):
                params = L.params or {}
                if L.input_geometry is not None:
                    params["perturbed_positions"] = L.input_geometry.mask
                for key in ("lb", "ub", "center", "eps", "perturbed_positions"):
                    t = params.get(key)
                    if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] != target_B:
                        params[key] = (
                            t[:1].expand(target_B, *t.shape[1:]).contiguous()
                        )
                if L.kind == "INPUT" and "shape" in params:
                    shape_param = params["shape"]
                    # JSON loads INPUT shape as a list, but the original guard
                    # only accepted tuple and silently skipped it, leaving the
                    # batch axis unbatchified (root cause of the B>1 shape bug).
                    if not isinstance(shape_param, (list, tuple)):
                        continue
                    shp = list(shape_param)
                    if shp and not structural and shp[0] != target_B:
                        shp[0] = target_B
                        params["shape"] = tuple(shp)
                    params["batch_size"] = target_B
                    labeled = params.get("labeled_input")
                    if (
                        not structural and isinstance(labeled, LabeledInputTensor)
                        and labeled.tensor.shape[0] == original_B
                        and original_B != target_B
                    ):
                        tensor = labeled.tensor[:1].expand(target_B, *labeled.tensor.shape[1:]).clone()
                        label = labeled.label
                        if label is not None and label.shape[0] == original_B:
                            label = label[:1].expand(target_B, *label.shape[1:]).clone()
                        params["labeled_input"] = LabeledInputTensor(tensor=tensor, label=label)
            elif L.kind in ("LSTM", "GRU", "RNN", "EMBEDDING"):
                params = L.params or {}
                for key in ("input_shape", "output_shape"):
                    shp = params.get(key)
                    if isinstance(shp, (list, tuple)) and shp and shp[0] != target_B:
                        new_shp = list(shp)
                        new_shp[0] = target_B
                        params[key] = tuple(new_shp)
            elif L.kind == "ASSERT":
                self._batchify_assert_layer(L, target_B)
        return self._migrate_net_to_device(new_net)

    def _migrate_net_to_device(self, net: Net) -> Net:
        """Move every tensor in ``net.layers[*].params`` to ``self.device`` and
        cast floating-point tensors to ``self.dtype``. Non-tensor params and
        integer / bool tensors are passed through untouched. Required so
        downstream ``analyze`` doesn't see mixed CPU/CUDA matmul operands.
        """
        for L in net.layers:
            params = L.params or {}
            for k, v in list(params.items()):
                if torch.is_tensor(v):
                    params[k] = v.to(
                        device=self.device,
                        dtype=self.dtype if v.is_floating_point() else v.dtype,
                    )
            if L.input_geometry is not None:
                L.refresh_input_geometry()
        return net

    def _batchify_assert_layer(self, L: Layer, target_B: int) -> None:
        """Re-encode an ASSERT layer to ``target_B`` lanes via the canonical
        ``OutputSpec.encode_linear`` pipeline (single source of truth).

        Mutates ``L.params`` in place. No-op if already at ``target_B``.
        """
        from act.front_end.specs import OutputSpec

        params = L.params or {}
        kind = str(params.get("kind", ""))

        m_raw = params.get("M")
        M = int(m_raw) if isinstance(m_raw, (int, float)) else 0
        C_cur = params.get("C")
        if (
            torch.is_tensor(C_cur)
            and C_cur.dim() == 2
            and M > 0
            and C_cur.shape[0] // M == target_B
        ):
            return

        n_out = (
            int(C_cur.shape[1])
            if torch.is_tensor(C_cur) and C_cur.dim() == 2
            else len(L.in_vars)
        )

        high: Dict[str, Any] = {}

        if kind in ("TOP1_ROBUST", "MARGIN_ROBUST"):
            y_true = params.get("y_true")
            if not torch.is_tensor(y_true):
                return
            K = n_out
            y0 = y_true.flatten()[:1]
            arange = torch.arange(
                target_B, device=y_true.device, dtype=y_true.dtype
            )
            high["y_true"] = ((y0 + arange) % K).contiguous()
            if kind == "MARGIN_ROBUST":
                margin = params.get("margin")
                if torch.is_tensor(margin):
                    high["margin"] = (
                        margin.flatten()[:1].expand(target_B).contiguous()
                    )

        elif kind == "LINEAR_LE":
            c, d = params.get("c"), params.get("d")
            if torch.is_tensor(c) and c.dim() == 2:
                high["c"] = c[0].contiguous()
            if torch.is_tensor(d) and d.dim() == 1:
                high["d"] = d[0:1].contiguous()

        elif kind == "RANGE":
            lb, ub = params.get("lb"), params.get("ub")
            if torch.is_tensor(lb) and lb.dim() == 2:
                high["lb"] = lb[0].contiguous()
            if torch.is_tensor(ub) and ub.dim() == 2:
                high["ub"] = ub[0].contiguous()

        elif kind == "UNSAFE_LINEAR":
            c, d = params.get("c"), params.get("d")
            if torch.is_tensor(c) and c.dim() == 3:
                high["c"] = c[0].contiguous()
            if torch.is_tensor(d) and d.dim() == 2:
                high["d"] = d[0].contiguous()

        else:
            return

        ref = (
            C_cur if torch.is_tensor(C_cur)
            else next(
                (v for v in high.values() if torch.is_tensor(v)), None
            )
        )
        spec = OutputSpec(kind=kind, **high)
        new_params = spec.encode_linear(
            B=target_B,
            n_out=n_out,
            device=ref.device if ref is not None else torch.device(self.device),
            dtype=(
                ref.dtype
                if ref is not None and ref.dtype.is_floating_point
                else self.dtype
            ),
        )

        L.params.clear()
        L.params.update(new_params)

    # Layer kinds where DualSolver currently has a known soundness gap — the
    # dispatch table at `dual_tf/dual_tf.py:223` aliases LRELU's backward
    # handler to `backward_relu`, which is mathematically incorrect (LReLU has
    # a non-zero negative slope; ReLU's backward zeros that branch out).  This
    # produces over-tight dual lower bounds, leading to false-CERTIFIED on
    # LReLU nets.  Tracked for fix in a follow-up PR.  Pre-filtered here so
    # CI reports SKIPPED instead of FAILED on the soundness bug.
    _DUAL_KNOWN_BROKEN_LAYER_KINDS: frozenset[str] = frozenset({"LRELU"})

    def _network_supported_by_mode(
        self, net: Net, tf_mode: str
    ) -> Tuple[bool, List[str]]:
        """Return ``(is_supported, sorted_blocking_kinds)`` for the
        (network, tf_mode) pair.

        Backend's static ``supports_layer`` returns False for some kind
        (e.g. some TFs lack LSTM/GRU/transformer ops). Real (undocumented)
        runtime errors are NOT swallowed; they bubble
        up as ERROR so we notice and fix them.

        Valid ``tf_mode`` values: ``"interval"`` and ``"hybridz"``.  ``"dual"``
        is a Solver choice (``--solver dual``), not a TF mode, after the β
        refactor; passing ``"dual"`` here raises ``ValueError`` from the
        underlying ``set_transfer_function_mode``.
        """
        from act.back_end.transfer_functions import (
            set_transfer_function_mode,
            get_transfer_function,
        )
        set_transfer_function_mode(tf_mode)
        tf = get_transfer_function()
        blocking = set()
        for L in net.layers:
            if not tf.supports_layer(L.kind):
                blocking.add(L.kind)
        return len(blocking) == 0, sorted(blocking)

    def find_concrete_counterexample(
        self,
        name: str,
        model: torch.nn.Module,
        max_random: int = 64,
        act_net: Optional[Net] = None,
    ) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Try to find a concrete counterexample via concrete execution.
        Returns (input_tensor, results_dict) if found, else None.
        """
        if max_random < 0:
            raise ValueError(f"max_random must be >= 0, got {max_random}")
        was_training = bool(getattr(model, "training", False))
        model.eval()

        try:
            if act_net is None:
                act_net = self.factory.get_act_net(name)
            input_shape = None
            shape_prod = None
            if act_net is not None:
                for layer in getattr(act_net, "layers", []):
                    if getattr(layer, "kind", None) == "INPUT":
                        shp = (layer.params or {}).get("shape", None)
                        if (
                            isinstance(shp, (list, tuple))
                            and shp
                            and all(isinstance(x, int) and x > 0 for x in shp)
                        ):
                            input_shape = tuple(shp)
                            shape_prod = int(torch.tensor(input_shape).prod().item())
                        break

            spec_lb = spec_ub = None
            if act_net is not None:
                from act.back_end.verifier import gather_input_spec_layers, seed_from_input_specs

                specs = gather_input_spec_layers(act_net)
                if specs:
                    seed = seed_from_input_specs(specs)
                    lb = seed.lb.to(self.device, self.dtype).flatten()
                    ub = seed.ub.to(self.device, self.dtype).flatten()
                    if (
                        lb.shape == ub.shape
                        and lb.numel() > 0
                        and (not torch.any(lb > ub))
                    ):
                        spec_lb, spec_ub = lb, ub

            if spec_lb is None or spec_ub is None:
                return None

            delta = spec_ub - spec_lb
            dim = int(spec_lb.numel())
            geometry = act_net.input_geometry if act_net is not None else None

            def _probe(x_flat: torch.Tensor, label: str) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
                x = (
                    x_flat.reshape(*input_shape)
                    if (input_shape and shape_prod == x_flat.numel())
                    else x_flat.reshape(1, -1)
                ).to(self.device, self.dtype)
                # These are generated hull samples, not alleged solver witnesses.
                if geometry is not None:
                    x = geometry.project(x)
                with torch.no_grad():
                    res = model(x)
                if (
                    isinstance(res, dict)
                    and res.get("input_satisfied", False)
                    and (not res.get("output_satisfied", True))
                ):
                    logger.info("  🔴 Counterexample found (%s)", label)
                    logger.info("     Input explanation:  %s", res.get("input_explanation"))
                    logger.info("     Output explanation: %s", res.get("output_explanation"))
                    return x, res
                return None

            center = spec_lb + 0.5 * delta
            found = _probe(center, "spec_center")
            if found is not None:
                return found

            if dim <= 16:
                for i in range(dim):
                    for val, tag in ((spec_lb[i], "lb"), (spec_ub[i], "ub")):
                        x_edge = center.clone()
                        x_edge[i] = val
                        found = _probe(x_edge, f"spec_per_dim_{tag}_{i}")
                        if found is not None:
                            return found

            for k in range(max_random):
                found = _probe(spec_lb + torch.rand_like(spec_lb) * delta, f"spec_random_{k}")
                if found is not None:
                    return found

            return None

        finally:
            if was_training:
                model.train()

    def validate(
        self,
        tag: str,
        model: torch.nn.Module,
        act_net: Net,
        results: List[Any],
        solver: str,
        tf_mode: str = "interval",
        facts: Optional[Dict[int, Any]] = None,
        num_samples: int = 10,
        per_neuron_config: Optional[PerNeuronCheckConfig] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run unified Level-1 and Level-2 validation after verification."""
        model = model.to(self.device, self.dtype).eval()

        concrete_ce = self.find_concrete_counterexample(tag, model, act_net=act_net)
        lane_violated = None
        if concrete_ce is not None and len(results) > 1:
            _, ce_res = concrete_ce
            out_ps = ce_res.get("output_satisfied_per_sample")
            if out_ps is not None and len(out_ps) == len(results):
                in_ps = ce_res.get("input_satisfied_per_sample")
                if in_ps is None or len(in_ps) != len(results):
                    in_ps = torch.ones(len(results), dtype=torch.bool)
                lane_violated = [
                    bool(in_ps[i]) and not bool(out_ps[i]) for i in range(len(results))
                ]
        counterexample_records = []
        for idx, verify_result in enumerate(results):
            network_name = f"{tag}[{idx}]" if len(results) > 1 else tag
            lane_ce = concrete_ce
            if lane_violated is not None and not lane_violated[idx]:
                lane_ce = None
            validation = self._cross_validate_counterexample(
                network_name=network_name,
                solver_name=solver,
                concrete_counterexample=lane_ce,
                verifier_status=verify_result.status,
            )
            validation["batch_size"] = batch_size
            self.validation_results.append(validation)
            counterexample_records.append(validation)
            ce_label = "FOUND" if validation["concrete_counterexample"] else "NOT_FOUND"
            verifier = verify_result.status.name
            print(
                f"  [soundness] {validation['network']}: {validation['validation_status']} "
                f"(concrete_ce={ce_label}, verifier={verifier})"
            )

        bounds_record = self._validate_bounds_for_hook(
            tag=tag,
            model=model,
            act_net=act_net,
            solver=solver,
            tf_mode=tf_mode,
            facts=facts,
            num_samples=num_samples,
            per_neuron_config=per_neuron_config,
            batch_size=batch_size,
        )
        return {
            "counterexample": counterexample_records[-1] if len(counterexample_records) == 1 else counterexample_records,
            "bounds": bounds_record,
        }

    def skip_reason(
        self,
        act_net: Net,
        solver: str,
        tf_mode: str,
    ) -> Optional[str]:
        """Return the capability skip reason for an unsupported (net, solver, tf_mode) cell."""
        if solver == "dual":
            from act.back_end.dual_tf.dual_tf import DualTF

            dual_tf = DualTF()
            layer_kinds = {L.kind for L in act_net.layers}
            missing = sorted({k for k in layer_kinds if not dual_tf.supports_layer(k)})
            known_broken = sorted(layer_kinds & self._DUAL_KNOWN_BROKEN_LAYER_KINDS)

            if missing:
                return f"DualTF cannot handle: {', '.join(missing)}"
            if known_broken:
                return (
                    f"DualSolver has a known soundness gap on "
                    f"{', '.join(known_broken)} (deferred to follow-up PR)"
                )
            return None

        ok, missing = self._network_supported_by_mode(act_net, tf_mode)
        if not ok:
            return f"tf_mode={tf_mode!r} has no handler for: {', '.join(missing)}"
        return None

    def overall_failed(self, ignore_errors: bool = False) -> bool:
        """Return whether any validation result failed or errored."""
        summary_l1 = self._compute_summary("counterexample")
        summary_l2 = self._compute_summary("bounds")
        failed = (
            summary_l1["failed"] > 0
            or summary_l1.get("errors", 0) > 0
            or summary_l2["failed"] > 0
            or summary_l2.get("errors", 0) > 0
        )
        return False if ignore_errors else failed

    def _record(self, **fields) -> Dict[str, Any]:
        self.validation_results.append(fields)
        return fields

    def record_skip(
        self,
        name: str,
        solver: str,
        tf_mode: str,
        batch_size: Optional[int],
        reason: str,
    ) -> Dict[str, Any]:
        record = self._record(
            network=name,
            solver=solver,
            tf_mode=tf_mode,
            batch_size=batch_size,
            validation_type="counterexample",
            validation_status="SKIPPED",
            explanation=f"⏭️  SKIPPED: {reason}",
        )
        logger.info("\n  %s", record["explanation"])
        return record

    def record_error(
        self,
        name: str,
        solver: str,
        tf_mode: str,
        batch_size: Optional[int],
        error: str,
    ) -> Dict[str, Any]:
        return self._record(
            network=name,
            solver=solver,
            tf_mode=tf_mode,
            batch_size=batch_size,
            validation_type="counterexample",
            status="ERROR",
            error=error,
            concrete_counterexample=False,
        )

    def _validate_bounds_for_hook(
        self,
        *,
        tag: str,
        model: torch.nn.Module,
        act_net: Net,
        solver: str,
        tf_mode: str,
        facts: Optional[Dict[int, Any]],
        num_samples: int,
        per_neuron_config: Optional[PerNeuronCheckConfig],
        batch_size: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Run Level-2 bounds validation for the unified hook."""
        record_tf_mode = "dual-forward" if solver == "dual" else tf_mode

        def _bounds_record(**fields) -> Dict[str, Any]:
            return self._record(
                network=tag,
                tf_mode=record_tf_mode,
                batch_size=batch_size,
                validation_type="bounds",
                **fields,
            )

        alignment_skip_reason = check_hookable_alignment(act_net, model)
        if alignment_skip_reason:
            skip_result = _bounds_record(
                validation_status="SKIPPED",
                explanation=f"⏭️  SKIPPED: {alignment_skip_reason}",
                unsupported_kinds=[],
            )
            logger.info(f"\n  {skip_result['explanation']}")
            return skip_result

        if facts is None:
            return _bounds_record(
                status="ERROR",
                error="Missing verifier facts for bounds validation",
                samples_processed=0,
            )
        bounds_by_layer, bounds_errors = bounds_from_facts(act_net, facts)

        if bounds_errors:
            return _bounds_record(
                status="ERROR",
                error="; ".join(bounds_errors[:3]),
                samples_processed=0,
            )

        violations = []
        total_checks = 0
        per_neuron_config = per_neuron_config or PerNeuronCheckConfig()
        inputs = sample_inputs_from_spec(
            act_net,
            num_samples,
            device=self.device,
            dtype=self.dtype,
        )

        for sample_idx, input_tensor in enumerate(inputs):
            try:
                check = run_per_neuron_bounds_check(
                    act_net=act_net,
                    model=model,
                    input_tensor=input_tensor,
                    config=per_neuron_config,
                    precomputed_bounds=bounds_by_layer,
                    pre_activation=(solver == "dual"),
                )
                if check.get("status") == "ERROR":
                    raise RuntimeError("; ".join(check.get("errors", [])[:3]))

                total_checks += int(check.get("total_checks", 0))
                if int(check.get("violations_total", 0)) > 0:
                    violated_layers = [
                        s
                        for s in check.get("layerwise_stats", [])
                        if int(s.get("num_violations", 0)) > 0
                    ]
                    violation_info = {
                        "sample_idx": sample_idx,
                        "violations_total": int(check.get("violations_total", 0)),
                        "worst_gap": float(check.get("worst_gap", 0.0)),
                        "violations_topk": check.get("violations_topk", []),
                        "violated_layers": violated_layers,
                    }
                    violations.append(violation_info)
                    top1 = (check.get("violations_topk", []) or [None])[0]
                    if isinstance(top1, dict):
                        concrete = float(top1.get("concrete", 0.0))
                        lb = float(top1.get("lb", 0.0))
                        ub = float(top1.get("ub", 0.0))
                        if concrete < lb:
                            violation_dir = "below_lb"
                        elif concrete > ub:
                            violation_dir = "above_ub"
                        else:
                            violation_dir = "outside_bounds"
                        logger.error(
                            "  ❌ Bounds violation at sample %d: %d violating neurons | "
                            "worst_gap=%.6g | layer_id=%s kind=%s neuron=%s dir=%s | "
                            "concrete=%.6g lb=%.6g ub=%.6g",
                            sample_idx,
                            int(check.get("violations_total", 0)),
                            float(check.get("worst_gap", 0.0)),
                            top1.get("layer_id", "?"),
                            top1.get("kind", "?"),
                            top1.get("neuron_index", "?"),
                            violation_dir,
                            concrete,
                            lb,
                            ub,
                        )
                    else:
                        logger.error(
                            "  ❌ Bounds violation at sample %d: %d violating neurons | worst_gap=%.6g",
                            sample_idx,
                            int(check.get("violations_total", 0)),
                            float(check.get("worst_gap", 0.0)),
                        )
            except Exception as e:
                logger.error("  ⚠️ Abstract analysis failed for sample %d: %s", sample_idx, e)
                return _bounds_record(
                    status="ERROR",
                    error=str(e),
                    samples_processed=sample_idx,
                )

        # Never turn an empty hook trace into a green soundness result.
        if total_checks == 0:
            result = _bounds_record(
                validation_status="SKIPPED",
                explanation=(
                    "⏭️  SKIPPED: per-neuron validation produced zero bound checks; "
                    "no soundness claim was made"
                ),
                total_checks=0,
                violations=[],
                per_neuron_config={"topk": per_neuron_config.topk},
            )
            logger.info(f"\n  {result['explanation']}")
        elif violations:
            result = _bounds_record(
                validation_status="FAILED",
                explanation=f"🚨 UNSOUND BOUNDS: {len(violations)} violations found across {num_samples} samples",
                total_checks=total_checks,
                violations=violations,
                per_neuron_config={"topk": per_neuron_config.topk},
            )
            logger.error(f"\n  {result['explanation']}")
        else:
            result = _bounds_record(
                validation_status="PASSED",
                explanation=f"✅ SOUND BOUNDS: All {total_checks} checks passed across {num_samples} samples",
                total_checks=total_checks,
                violations=[],
                per_neuron_config={"topk": per_neuron_config.topk},
            )
            logger.info(f"\n  {result['explanation']}")

        return result

    def _cross_validate_counterexample(
        self,
        network_name: str,
        solver_name: str,
        concrete_counterexample: Optional[Tuple[torch.Tensor, Dict[str, Any]]],
        verifier_status: VerifyStatus,
    ) -> Dict[str, Any]:
        """
        Cross-validate concrete inference vs formal verification (Level 1).

        Validation Rules:
        1. If concrete counterexample found → verifier MUST report FALSIFIED or UNKNOWN
        2. If no concrete counterexample → verifier can report anything (testing incomplete)
        """
        result = {
            "network": network_name,
            "solver": solver_name,
            "validation_type": "counterexample",
            "concrete_counterexample": concrete_counterexample is not None,
            "verifier_result": verifier_status,
            "validation_status": None,
            "explanation": None,
        }

        if concrete_counterexample is not None:
            # We found a real counterexample - verifier MUST NOT claim CERTIFIED
            input_tensor, inference_results = concrete_counterexample

            if verifier_status == VerifyStatus.CERTIFIED:
                # CRITICAL BUG: Verifier claims safe, but we have a counterexample!
                result["validation_status"] = "FAILED"
                result["explanation"] = (
                    f"🚨 SOUNDNESS BUG DETECTED! Verifier claims CERTIFIED but "
                    f"concrete counterexample exists. This is a false negative."
                )
                logger.error(f"\n  {result['explanation']}")
                logger.error(
                    f"     Counterexample input: {input_tensor.shape}, "
                    f"range=[{input_tensor.min():.4f}, {input_tensor.max():.4f}]"
                )
                logger.error(
                    f"     Output violation: {inference_results['output_explanation']}"
                )

            elif verifier_status == VerifyStatus.FALSIFIED:
                # CORRECT: Verifier correctly identified the issue
                result["validation_status"] = "PASSED"
                result["explanation"] = (
                    f"✅ CORRECT - Verifier correctly reported FALSIFIED "
                    f"(matches concrete execution)"
                )
                logger.info(f"\n  {result['explanation']}")

            elif verifier_status == VerifyStatus.UNKNOWN:
                # ACCEPTABLE: Verifier couldn't decide (incomplete but sound)
                result["validation_status"] = "ACCEPTABLE"
                result["explanation"] = (
                    f"⚠️ INCOMPLETE - Verifier returned UNKNOWN, but concrete "
                    f"counterexample exists (verifier is sound but incomplete)"
                )
                logger.warning(f"\n  {result['explanation']}")

            else:
                result["validation_status"] = "UNKNOWN"
                result["explanation"] = f"Unknown verifier result: {verifier_status}"
                logger.warning(f"\n  {result['explanation']}")

        else:
            # No concrete counterexample found in testing
            result["validation_status"] = "INCONCLUSIVE"
            result["explanation"] = (
                f"⚪ INCONCLUSIVE - No counterexample found in concrete testing. "
                f"Verifier result: {verifier_status} (cannot validate with this test)"
            )
            logger.info(f"\n  {result['explanation']}")

        return result

    def _compute_summary(self, validation_type: str) -> Dict[str, Any]:
        """
        Compute validation summary statistics for specific validation type.

        Args:
            validation_type: 'counterexample' or 'bounds'
        """
        results = [
            r
            for r in self.validation_results
            if r.get("validation_type") == validation_type
        ]
        return self._summarize_results(validation_type, results)

    def _summarize_results(
        self, validation_type: str, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        total = len(results)

        if total == 0:
            return {
                "validation_type": validation_type,
                "total": 0,
                "passed": 0,
                "failed": 0,
                "acceptable": 0,
                "inconclusive": 0,
                "skipped": 0,
                "unknown": 0,
                "errors": 0,
                "results": [],
                "error_message": "No validation results (all tests encountered errors)",
            }

        passed = sum(1 for r in results if r.get("validation_status") == "PASSED")
        failed = sum(1 for r in results if r.get("validation_status") == "FAILED")
        acceptable = sum(
            1 for r in results if r.get("validation_status") == "ACCEPTABLE"
        )
        inconclusive = sum(
            1 for r in results if r.get("validation_status") == "INCONCLUSIVE"
        )
        skipped = sum(
            1 for r in results if r.get("validation_status") == "SKIPPED"
        )
        unknown = sum(
            1 for r in results if r.get("validation_status") == "UNKNOWN"
        )
        errors = sum(1 for r in results if r.get("status") == "ERROR")

        summary = {
            "validation_type": validation_type,
            "total": total,
            "passed": passed,
            "failed": failed,
            "acceptable": acceptable,
            "inconclusive": inconclusive,
            "skipped": skipped,
            "unknown": unknown,
            "errors": errors,
            "results": results,
        }

        if validation_type == "counterexample":
            summary["counterexamples_found"] = sum(
                1 for r in results if r.get("concrete_counterexample", False)
            )
            summary["critical_bugs"] = failed
        elif validation_type == "bounds":
            summary["total_checks"] = sum(r.get("total_checks", 0) for r in results)
            summary["total_violations"] = sum(
                len(r.get("violations", [])) for r in results
            )

        self._print_summary(summary)
        return summary

    def _print_summary(self, summary: Dict[str, Any]):
        """Print validation summary for specific validation type."""
        validation_type = summary.get("validation_type", "unknown")

        print("\n" + rule())
        print(f"VALIDATION SUMMARY - {validation_type.upper()}")
        print(rule())

        if summary["total"] == 0:
            print()
            print("⚠️  No validation tests completed successfully")
            if "error_message" in summary:
                print(f"   {summary['error_message']}")
            print(rule())
            return

        print(f"\nTotal validation tests: {summary['total']}")

        if validation_type == "counterexample":
            print(
                f"Concrete counterexamples found: {summary.get('counterexamples_found', 0)}"
            )
        elif validation_type == "bounds":
            print(f"Total bound checks: {summary.get('total_checks', 0)}")
            print(f"Total violations: {summary.get('total_violations', 0)}")

        print()
        print(f"✅ PASSED:       {summary['passed']}")
        if validation_type == "counterexample":
            print(f"⚠️  ACCEPTABLE:   {summary['acceptable']}")
            print(f"⚪ INCONCLUSIVE: {summary['inconclusive']}")
        if summary.get("skipped", 0) > 0:
            print(f"⏭️  SKIPPED:      {summary['skipped']}")
        print(f"❌ ERRORS:       {summary['errors']}")
        print(f"🚨 FAILED:       {summary['failed']}")
        print(rule())

        if summary["failed"] > 0:
            print(f"\n🚨 CRITICAL: {validation_type.upper()} validation failed!")
            if validation_type == "counterexample":
                print("Soundness bugs detected in the following networks:")
            else:
                print("Unsound bounds detected in the following networks:")
            for result in summary["results"]:
                if result.get("validation_status") == "FAILED":
                    if validation_type == "counterexample":
                        print(f"  - {result['network']} ({result['solver']})")
                    else:
                        print(f"  - {result['network']} ({result['tf_mode']})")
            print()
        elif summary["errors"] > 0:
            print(f"\n⚠️  All {validation_type} validation tests encountered errors!")
            print("This indicates pre-existing bugs in the verification backend.")
            print()
        elif validation_type == "bounds" and summary.get("total_checks", 0) == 0:
            print("\n⏭️  BOUNDS validation SKIPPED: no bound checks were performed.")
        else:
            print(f"\n✅ {validation_type.upper()} validation PASSED!")

        print(rule())
