# ===- act/util/counterexample.py - Counterexample validation -------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
"""The single concrete counterexample check shared by fuzzing and verification.

Candidate producers (fuzzer mutations, solver witnesses, BaB terminal LPs,
concrete probes) never decide on their own that an input is a counterexample:
they call this module. A counterexample is a finite input that satisfies every
input constraint and whose concrete output violates the output property.

- ``input_constraint_satisfied`` checks one input constraint per lane; the
  model wrapper ``InputSpecLayer`` and ACT ``INPUT_SPEC`` layers both use it.
- ``output_violations`` applies ``OutputSpec``'s concrete boundary semantics;
  ``OutputSpecLayer``, the fuzzer and ACT ``ASSERT`` layers all use it.
- ``validate_counterexamples`` combines both with a concrete forward of an
  ACT ``Net``, optionally restricted to a BaB node box and ReLU split phases.

Batch lanes keep their input-specification and ASSERT alignment throughout.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from act.front_end.specs import InKind, InputGeometry, OutputSpec

if TYPE_CHECKING:
    from act.back_end.core import Bounds, Layer, Net


# Entries keep their Net and a hit requires ``is`` identity, so a recycled
# id() rebuilds instead of returning another net's module.
_VIOLATION_CHECK_MODULE_CACHE: dict[int, tuple[Net, torch.nn.Module]] = {}


def clear_violation_check_module_cache() -> None:
    """Release reconstructed models at the end of a verification dispatch."""
    _VIOLATION_CHECK_MODULE_CACHE.clear()


def get_concrete_model(net: Net, reference: torch.Tensor) -> torch.nn.Module:
    """Reuse the concrete model in the candidate device and analysis dtype."""
    key = id(net)
    cached = _VIOLATION_CHECK_MODULE_CACHE.get(key)
    if cached is not None and cached[0] is net:
        return cached[1]
    from act.pipeline.verification.act2torch import ACTToTorch

    module = ACTToTorch(net).run().to(device=reference.device, dtype=reference.dtype)
    _VIOLATION_CHECK_MODULE_CACHE[key] = (net, module)
    return module


def concrete_forward(net: Net | torch.nn.Module, candidates: torch.Tensor) -> torch.Tensor:
    """Evaluate concrete inputs; reconstruction and inference errors propagate."""
    from act.util.model_inference import infer_single_model

    module = net if isinstance(net, torch.nn.Module) else get_concrete_model(net, candidates)
    module.eval()
    success, output, error = infer_single_model("ce_validate_batched", module, candidates)
    if not success or output is None:
        raise RuntimeError(f"Counterexample model forward failed: {error}")
    if output.dim() < 2 or output.shape[0] != candidates.shape[0]:
        raise ValueError(f"Concrete output must preserve candidate lanes, got {tuple(output.shape)}")
    return output.flatten(1)


def input_constraint_satisfied(
    x: torch.Tensor, kind: str, params: Mapping[str, Any], *,
    geometry: InputGeometry | None = None, tolerance: float = 0.0,
) -> torch.Tensor:
    """Per-lane membership of ``x`` [B, ...] in one input constraint.

    ``params`` tensors are already aligned with the lanes (or broadcast):
    BOX/LINF_BALL use ``lb``/``ub``, LIN_POLY uses ``A`` ([m, n] or per-lane
    [B, m, n]) and ``b``. An ``InputGeometry`` (LP_EMBEDDING, or a norm ball
    stored on an ACT layer) takes precedence over the box form.
    """
    if geometry is not None:
        return geometry.contains(x, tol=tolerance)
    lanes = x.shape[0]  # 1-D inputs ([B]) have one value per lane, so never flatten(1)
    flat = x.reshape(lanes, -1)
    match kind:
        case InKind.BOX | InKind.LINF_BALL:
            lb, ub = params["lb"].to(x), params["ub"].to(x)
            return ((x >= lb - tolerance) & (x <= ub + tolerance)).reshape(lanes, -1).all(1)
        case InKind.LIN_POLY:
            a, b = params["A"].to(x), params["b"].to(x)
            lhs = torch.bmm(a, flat.unsqueeze(-1)).squeeze(-1) if a.dim() == 3 else flat @ a.T
            return (lhs <= b).all(1)
        case _:
            raise NotImplementedError(f"Unsupported concrete input constraint: {kind}")


def check_input_specs_batched(
    candidates: torch.Tensor, spec_layers: Sequence[Layer], *, tolerance: float = 1e-7,
) -> torch.Tensor:
    """Check finite candidates against every ACT INPUT_SPEC layer, without dropping lanes."""
    valid = torch.isfinite(candidates).reshape(candidates.shape[0], -1).all(1)
    for layer in spec_layers:
        valid &= input_constraint_satisfied(
            candidates, layer.params["kind"], layer.params,
            geometry=layer.input_geometry, tolerance=tolerance,
        )
    return valid


def output_violations(
    outputs: torch.Tensor, spec: OutputSpec, *, rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-lane (violated, severity); non-finite outputs never count as violations."""
    violated, severity = spec.violation(outputs, rows=rows)
    finite = torch.isfinite(outputs.reshape(outputs.shape[0], -1)).all(1) & torch.isfinite(severity)
    return violated & finite, severity


def check_output_violations(outputs: torch.Tensor, assert_layer: Layer) -> torch.Tensor:
    """Use the canonical ASSERT rows and OutputSpec's concrete boundary semantics."""
    params = assert_layer.params
    rows = int(params["M"])
    spec = OutputSpec(
        kind=params["kind"],
        c=params["C"].to(outputs).reshape(-1, rows, outputs.shape[1]),
        d=params["thresholds"].to(outputs),
        y_true=params.get("y_true"), margin=params.get("margin"),
        lb=params.get("lb"), ub=params.get("ub"),
    )
    return output_violations(outputs, spec)[0]


def _forward_with_phases(
    net: Net, candidates: torch.Tensor, split_signs: Mapping[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concrete forward that also checks each split ReLU's pre-activation sign."""
    from act.pipeline.verification.act2torch import ActGraphModule

    module = get_concrete_model(net, candidates)
    observed: dict[int, torch.Tensor] = {}
    handles = []
    try:
        for graph in module.modules():
            if not isinstance(graph, ActGraphModule):
                continue
            for lid in split_signs:
                if str(lid) not in graph.layers:
                    raise NotImplementedError(f"unsupported_phase_capture:{lid}")

                def capture(_module: torch.nn.Module, args: tuple[torch.Tensor, ...],
                            layer_id: int = lid) -> None:
                    observed[layer_id] = args[0].flatten(1).detach()
                handles.append(graph.layers[str(lid)].register_forward_pre_hook(capture))
        outputs = concrete_forward(module, candidates)
    finally:
        for handle in handles:
            handle.remove()
    phases = torch.ones(candidates.shape[0], dtype=torch.bool, device=candidates.device)
    for lid, signs in split_signs.items():
        if lid not in observed:
            return outputs, torch.zeros_like(phases)
        values = observed[lid]
        phases &= (torch.isfinite(values) & ((signs == 0) | (values * signs >= 0))).all(1)
    return outputs, phases


@torch.no_grad()
def validate_counterexamples(
    net: Net, candidates: torch.Tensor, *,
    model_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    box: Bounds | None = None,
    split_signs: Mapping[int, torch.Tensor] | None = None,
    tolerance: float = 1e-7,
) -> torch.Tensor:
    """Return a per-lane mask of legal inputs that concretely violate the property.

    ``box`` and ``split_signs`` restrict legality to a BaB node: the candidate
    must lie in the node box and follow every split ReLU phase.
    """
    if candidates.dim() < 2:
        raise ValueError("Counterexample candidates must have shape [B, *input_shape]")
    if box is not None and candidates.shape != box.lb.shape:
        raise ValueError("node candidates must preserve the node's K lanes")
    spec_layers = [layer for layer in net.layers if layer.kind == "INPUT_SPEC"]
    assert_layer = next(layer for layer in net.layers if layer.kind == "ASSERT")
    valid = check_input_specs_batched(candidates, spec_layers, tolerance=tolerance)
    if box is not None:
        valid &= ((candidates >= box.lb) & (candidates <= box.ub)).flatten(1).all(1)
    if not bool(valid.any()):
        return valid
    # Keep lane-specific targets aligned, while preventing illegal candidates
    # from poisoning the concrete forward of otherwise valid lanes.
    filler = torch.zeros_like(candidates) if box is None else box.lb
    safe = torch.where(valid.reshape(-1, *([1] * (candidates.ndim - 1))), candidates, filler)
    if split_signs:
        outputs, phases = _forward_with_phases(net, safe, split_signs)
        valid &= phases
    else:
        outputs = concrete_forward(net, safe) if model_fn is None else model_fn(safe)
    if outputs.dim() < 2 or outputs.shape[0] != candidates.shape[0]:
        raise ValueError(f"Concrete output must preserve candidate lanes, got {tuple(outputs.shape)}")
    return valid & check_output_violations(outputs.flatten(1), assert_layer)
