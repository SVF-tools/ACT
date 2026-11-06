#===- act/back_end/layer_util.py - Layer Utility Functions --------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Layer utility functions for ACT layers and networks. Separated from
#   layer_schema.py to avoid circular import issues.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from typing import Any
import difflib
import logging

logger = logging.getLogger(__name__)

# Import validation components
try:
    # Try relative import first (when used as module)
    from .layer_schema import REGISTRY, LayerKind, SUPPORTED_EXPORT_OPS
    from act.front_end.specs import InKind
except ImportError:
    # Fallback to absolute import (when run standalone)
    import sys
    # Use path_config for consistent project root detection
    from act.util.path_config import get_project_root
    project_root = get_project_root()
    sys.path.insert(0, project_root)
    from act.back_end.layer_schema import REGISTRY, LayerKind, SUPPORTED_EXPORT_OPS
    from act.front_end.specs import InKind

# Import Layer from core to avoid circular import issues
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from .core import Layer
    except ImportError:
        # Will import at runtime when needed to avoid circular import
        pass

import torch
Tensor = torch.Tensor

# ------------------------------
# Strict validation & helpers
# ------------------------------
def _missing(required: list[str], got: dict[str, Any]) -> list[str]:
    return [k for k in required if k not in got]

def _unknown(allowed: list[str], got: dict[str, Any]) -> list[str]:
    return [k for k in got.keys() if k not in allowed]

def _suggest(key: str, candidates: list[str]) -> list[str]:
    return difflib.get_close_matches(key, candidates, n=3, cutoff=0.6)

def _format_unknown(kind: str, category: str, unknowns: list[str], allowed: list[str]) -> str:
    parts = []
    for u in unknowns:
        sugg = _suggest(u, allowed)
        if sugg:
            parts.append(f"'{u}' (did you mean {', '.join(sugg)}?)")
        else:
            parts.append(f"'{u}' (no close match)")
    hint = f"Add to REGISTRY['{kind}']['{category}'] in layer_schema.py if intentional."
    return f"Unknown {category}: " + ", ".join(parts) + f". {hint}"

def validate_layer(layer: "Layer") -> None:
    """Strict validation against REGISTRY with friendly messages."""
    kind = layer.kind
    if kind not in REGISTRY:
        raise ValueError(f"Kind '{kind}' not in REGISTRY. Add it to REGISTRY in layer_schema.py.")

    spec = REGISTRY[kind]

    # Type check for params
    for name, val in layer.params.items():
        # Special validation for labeled_input (must be LabeledInputTensor with tensor + int/list)
        if name == 'labeled_input':
            from act.front_end.spec_creator_base import LabeledInputTensor
            if not isinstance(val, LabeledInputTensor):
                raise TypeError(
                    f"{kind}.params['labeled_input'] must be LabeledInputTensor, got {type(val)}. "
                    f"Use LabeledInputTensor(tensor=..., label=...) from spec_creator_base."
                )
            # Validate tensor component
            try:
                import torch  # noqa
                if not isinstance(val.tensor, Tensor):  # type: ignore[arg-type]
                    raise TypeError(
                        f"{kind}.params['labeled_input'].tensor must be torch.Tensor, got {type(val.tensor)}."
                    )
            except ImportError as e:
                # Intentional: torch is optional here; skip tensor-type check when torch is unavailable.
                logger.debug("suppressed: %s", e)
            # Validate label component (accepts int, list[int], or torch.Tensor)
            if val.label is not None:
                try:
                    import torch
                    if not isinstance(val.label, (int, list, torch.Tensor)):
                        raise TypeError(
                            f"{kind}.params['labeled_input'].label must be int, list[int], or torch.Tensor, got {type(val.label)}."
                        )
                except ImportError:
                    # If torch not available, only accept int or list
                    if not isinstance(val.label, (int, list)):
                        raise TypeError(
                            f"{kind}.params['labeled_input'].label must be int or list[int], got {type(val.label)}."
                        )
                
                # Additional validation for list types (ensure all elements are ints)
                if isinstance(val.label, list) and not all(isinstance(x, int) for x in val.label):
                    raise TypeError(
                        f"{kind}.params['labeled_input'].label list contains non-int elements."
                    )
            continue
        # Tensor params are auto-detected at runtime via isinstance(val, torch.Tensor).
        # No explicit 'tensors' list validation needed - params can be any type (tensor or scalar).
        # Type validation is implicitly handled by downstream code that uses the params.

    miss_p = _missing(spec['params_required'], layer.params)
    
    allowed_p = spec['params_required'] + spec['params_optional']

    unk_p = _unknown(allowed_p, layer.params)

    errs: list[str] = []
    if miss_p:
        errs.append(f"Missing required PARAMS: {miss_p}. Add them or relax schema in REGISTRY['{kind}']['params_required'].")
    if unk_p:
        errs.append(_format_unknown(kind, "params_optional/params_required", unk_p, allowed_p))

    # Critical op sanity
    if kind == LayerKind.MHA.value:
        has_any = any(k in layer.params for k in ("in_proj_weight","q_proj.weight","k_proj.weight","v_proj.weight","out_proj.weight"))
        if not has_any:
            errs.append("MHA requires in_proj_* or split {q,k,v}_proj.* or out_proj.weight.")

    if errs:
        raise ValueError(f"Layer(id={layer.id}, kind={kind}) schema violation:\n- " + "\n- ".join(errs))

def validate_graph(layers: list["Layer"]) -> None:
    seen = set()
    for ly in layers:
        if ly.id in seen:
            raise ValueError(f"Duplicate layer id {ly.id}")
        seen.add(ly.id)
        validate_layer(ly)
    for ly in layers:
        for v in ly.in_vars + ly.out_vars:
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"Invalid var id {v} in layer {ly.id}")

def validate_wrapper_graph(layers: list["Layer"]) -> None:
    """Hard assertions for the wrapper layout."""
    if not layers:
        raise ValueError("Empty graph")

    kinds = [ly.kind for ly in layers]
    input_count = kinds.count(LayerKind.INPUT.value)
    input_spec_count = kinds.count(LayerKind.INPUT_SPEC.value)
    
    if input_count != 1:
        raise ValueError(f"Wrapper must have exactly one INPUT layer, found {input_count}.")
    if input_spec_count < 1:
        raise ValueError("Wrapper must include at least one INPUT_SPEC layer.")
    if kinds[-1] != LayerKind.ASSERT.value:
        raise ValueError(f"Last layer must be ASSERT, found {kinds[-1]}.")

    first_spec_idx = kinds.index(LayerKind.INPUT_SPEC.value)
    input_idx = kinds.index(LayerKind.INPUT.value)

    # INPUT_SPEC should come after INPUT
    if first_spec_idx < input_idx:
        raise ValueError("INPUT must come before INPUT_SPEC")
    
    # Between INPUT and INPUT_SPEC, only allow model layers (not wrapper types)
    wrapper_types = {LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, LayerKind.ASSERT.value}
    for i in range(input_idx + 1, first_spec_idx):
        if kinds[i] in wrapper_types:
            raise ValueError(
                f"Unexpected wrapper layer {kinds[i]} between INPUT and INPUT_SPEC at position {i}. "
                f"Preprocessing should be handled by data loader (e.g., torchvision.transforms)."
            )

    # All INPUT_SPEC layers must form a contiguous prefix block right after
    # INPUT. Once a non-INPUT_SPEC, non-wrapper layer appears, no more
    # INPUT_SPECs or INPUTs are allowed.
    seen_model_layer = False
    for i, k in enumerate(kinds[first_spec_idx+1:-1], start=first_spec_idx+1):
        if k == LayerKind.INPUT.value:
            raise ValueError(
                f"Unexpected {k} after the first INPUT_SPEC at index {i}."
            )
        if k == LayerKind.INPUT_SPEC.value:
            if seen_model_layer:
                raise ValueError(
                    f"INPUT_SPEC at index {i} appears after model layers; "
                    "all INPUT_SPEC layers must form a contiguous block "
                    "immediately after INPUT."
                )
        else:
            seen_model_layer = True


def validate_conset_ops(conset) -> None:
    for con in conset:
        tag = str(con.meta.get("tag", ""))
        if not tag:
            continue
        op = tag.split(":", 1)[0]
        if op and op not in SUPPORTED_EXPORT_OPS:
            raise ValueError(
                f"Unsupported op tag '{op}' (tag='{tag}'). "
                "Add it to SUPPORTED_EXPORT_OPS in layer_schema.py "
                "and exporter handling if intentional."
            )

def create_layer(id: int, kind: str, params: dict[str, Any],
                 in_vars: list[int], out_vars: list[int]) -> "Layer":
    """Create and validate a layer."""
    try:
        from .core import Layer
    except ImportError:
        import sys
        # Use path_config for consistent project root detection
        from act.util.path_config import get_project_root
        project_root = get_project_root()
        sys.path.insert(0, project_root)
        from act.back_end.core import Layer
    
    ly = Layer(id=id, kind=kind, params=params, in_vars=in_vars, out_vars=out_vars)
    validate_layer(ly)
    return ly

# ---------------------
# Tiny example (run file)
# ---------------------
if __name__ == "__main__":
    import sys
    # Add parent directory to path for absolute imports
    from act.util.path_config import get_project_root
    project_root = get_project_root()
    sys.path.insert(0, project_root)
        
    from act.back_end.core import Layer
    from act.back_end.layer_schema import LayerKind
    from act.front_end.specs import OutKind, OutputSpec
    from act.util.device_manager import get_default_device, get_default_dtype

    device, dtype = get_default_device(), get_default_dtype()
    input_shape = (1, 1, 2, 2)
    input_vars = list(range(4))
    flat_vars = list(range(4, 8))
    output_vars = list(range(8, 10))
    layers: list[Layer] = []

    # INPUT
    layers.append(create_layer(
            id=0, kind=LayerKind.INPUT.value,
            params={"shape": input_shape, "dtype": str(dtype)},
            in_vars=[], out_vars=input_vars,
    ))
    # SPEC (directly after INPUT - no adapters)
    lb_tensor = torch.full(input_shape, -1.0, device=device, dtype=dtype)
    ub_tensor = torch.full(input_shape, 1.0, device=device, dtype=dtype)
    layers.append(create_layer(
            id=1, kind=LayerKind.INPUT_SPEC.value,
            params={"kind": InKind.BOX, "lb": lb_tensor, "ub": ub_tensor},
            in_vars=input_vars, out_vars=input_vars,
    ))
    # Model toy
    layers.append(create_layer(
            id=2, kind=LayerKind.FLATTEN.value,
            params={"input_shape": input_shape, "output_shape": (1, 4)},
            in_vars=input_vars, out_vars=flat_vars,
    ))
    weight = torch.zeros((2, 4), device=device, dtype=dtype)
    bias = torch.zeros(2, device=device, dtype=dtype)
    layers.append(create_layer(
            id=3, kind=LayerKind.DENSE.value,
            params={"weight": weight, "bias": bias, "in_features": 4, "out_features": 2},
            in_vars=flat_vars, out_vars=output_vars,
    ))
    output_spec = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.ones(2, device=device, dtype=dtype),
        d=torch.ones(1, device=device, dtype=dtype),
    )
    layers.append(create_layer(
            id=4, kind=LayerKind.ASSERT.value,
            params=output_spec.encode_linear(B=1, n_out=2, device=device, dtype=dtype),
            in_vars=output_vars, out_vars=output_vars,
    ))

    validate_graph(layers)
    validate_wrapper_graph(layers)
    print("OK — wrapper model passes with", len(layers), "layers.")
