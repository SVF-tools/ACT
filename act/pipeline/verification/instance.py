#===- act/pipeline/verification/instance.py - Instance Verification -------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===----------------------------------------------------------------------===#

"""Shared conversion and disjunct accounting for verification entry points."""

from collections.abc import Iterator, Sequence
from pathlib import Path
import time
from typing import Literal

import torch
from torch import nn

from act.back_end.core import Net
from act.back_end.serialization.serialization import save_net_to_file
from act.pipeline.verification.torch2act import TorchToACT
from act.util.stats import VerifyResult, VerifyStatus


def convert_model(wrapped: nn.Module, save_path: str | None = None) -> Net:
    """Convert once and optionally save the exact Net returned to the solver."""
    net = TorchToACT(wrapped).run()
    if save_path is not None:
        save_net_to_file(net, save_path)
    return net


class DisjunctResults:
    """Aggregate complete lane results without treating missing work as proof."""

    def __init__(self, model_count: int) -> None:
        """Require at least one disjunct to prevent vacuous certification."""
        if model_count <= 0:
            raise ValueError("An instance must contain at least one model")
        self.model_count = model_count
        self.results: list[tuple[VerifyResult, ...]] = []

    def pending(self, deadline: float | None) -> Iterator[tuple[int, float | None]]:
        """Allocate each remaining model a share of the monotonic budget."""
        for index in range(self.model_count):
            if self.counterexample is not None:
                break
            budget = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                budget = remaining / (self.model_count - index)
            yield index, budget

    def record(self, results: Sequence[VerifyResult]) -> None:
        """Record all returned lanes, including an unresolved empty result."""
        if len(self.results) >= self.model_count:
            raise ValueError("More results than instance models")
        self.results.append(tuple(results))

    def save_path(self, path: str | None, index: int) -> str | None:
        """Suffix multi-model exports with their zero-based disjunct index."""
        if path is None or self.model_count == 1:
            return path
        target = Path(path)
        return str(target.with_name(f"{target.stem}.d{index}{target.suffix}"))

    @property
    def counterexample(self) -> torch.Tensor | None:
        """Return a witness already validated by the underlying verifier."""
        return next(
            (result.counterexample
             for lanes in self.results for result in lanes
             if result.status == VerifyStatus.FALSIFIED
             and result.counterexample is not None),
            None,
        )

    @property
    def verdict(self) -> Literal["sat", "unsat", "unknown", "timeout"]:
        """Require nonempty certified results for every disjunct for UNSAT."""
        if self.counterexample is not None:
            return "sat"
        if len(self.results) == self.model_count and all(
            lanes and all(r.status == VerifyStatus.CERTIFIED for r in lanes)
            for lanes in self.results
        ):
            return "unsat"
        if any(
            not lanes or any(
                r.status not in (VerifyStatus.CERTIFIED, VerifyStatus.TIMEOUT)
                for r in lanes
            )
            for lanes in self.results
        ):
            return "unknown"
        return "timeout"
