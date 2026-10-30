#===- act/front_end/vnnlib_loader/vnncomp_instance.py - Instance Loading ---===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===----------------------------------------------------------------------===#

"""Load one VNN-COMP instance through the existing specification pipeline."""

from dataclasses import dataclass
from pathlib import Path

from torch import nn

from act.front_end.model_synthesis import (
    merge_split_relus,
    synthesize_models_and_seeds_from_specs,
)
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.vnnlib_loader.create_specs import create_specs_from_paths


@dataclass(frozen=True)
class VnncompInstance:
    """Original model and synthesized disjuncts with their aligned seeds."""

    raw_model: nn.Module
    models: tuple[nn.Module, ...]
    seeds: tuple[tuple[LabeledInputTensor, ...], ...]
    inputs: tuple[LabeledInputTensor, ...]
    merged_relus: int


def load_vnncomp_instance(
    onnx_path: str | Path,
    vnnlib_path: str | Path,
    *,
    merge_relus: bool = False,
) -> VnncompInstance:
    """Load and synthesize all properties without choosing a solver policy."""
    for path in (Path(onnx_path), Path(vnnlib_path)):
        if not path.is_file():
            raise FileNotFoundError(path)
    source, name, raw_model, inputs, specs = create_specs_from_paths(
        onnx_path, vnnlib_path,
    )
    model, merged = merge_split_relus(raw_model) if merge_relus else (raw_model, 0)
    models, seeds = synthesize_models_and_seeds_from_specs(
        [(source, name, model, inputs, specs)],
    )
    if not models:
        raise ValueError("Instance has no synthesized verification models")
    return VnncompInstance(
        raw_model=raw_model,
        models=tuple(models.values()),
        seeds=tuple(tuple(seeds[key]) for key in models),
        inputs=tuple(inputs),
        merged_relus=merged,
    )
