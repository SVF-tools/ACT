"""
Property violation detection for ACTFuzzer.

Checks if model outputs violate OutputSpec properties and records counterexamples.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING, Optional, List, Tuple, cast
import time
import torch

from act.front_end.specs import OutKind, OutputSpec
from act.util.counterexample import output_violations

if TYPE_CHECKING:
    from act.pipeline.fuzzing.corpus import FuzzingSeed


INPUT_FEASIBILITY_ATTR = "_act_input_satisfied_per_sample"


@dataclass
class Counterexample:
    """
    Counterexample with full details.
    
    Represents an input that violates the OutputSpec property.
    This is the primary output of ACTFuzzer.
    
    Attributes:
        input: Input tensor that caused violation (perturbed)
        output: Model's output on this input
        seed_input: Original unperturbed input
        kind: Type of violation (TOP1_ROBUST, MARGIN_ROBUST, etc.)
        spec_row: OutputSpec row backing this counterexample
        severity: Signed violation margin returned by OutputSpec.violation()
        timestamp: When the counterexample was found
        true_class: Label the spec row expected, resolved when the counterexample
            was built. ``None`` for kinds that carry no label (RANGE, LINEAR_LE,
            UNSAFE_LINEAR).
    """
    input: torch.Tensor
    output: torch.Tensor
    seed_input: torch.Tensor
    kind: str
    spec_row: int
    severity: float
    timestamp: float
    true_class: Optional[int] = None

    def summary(self) -> str:
        """One-line summary of the counterexample."""
        details = f"spec_row={self.spec_row}, severity={self.severity:.6g}"
        if self.true_class is None:
            return f"{self.kind}: {details}"
        predicted_class = int(self.output.reshape(-1).argmax().item())
        return (
            f"{self.kind}: expected {self.true_class}, got {predicted_class} "
            f"({details})"
        )

    def to_dict(self) -> dict[str, object]:
        """Convert the counterexample to its serialized representation."""
        return {
            "input": self.input,
            "output": self.output,
            "seed_input": self.seed_input,
            "kind": self.kind,
            "spec_row": self.spec_row,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "true_class": self.true_class,
        }

    def save(self, path: str | Path) -> None:
        """Save counterexample to disk."""
        torch.save(self.to_dict(), path)

    @staticmethod
    def load(path: str | Path) -> Counterexample:
        """Load counterexample from disk."""
        data = cast(dict[str, Any], torch.load(path, weights_only=True))
        return Counterexample(
            input=data["input"],
            output=data["output"],
            seed_input=data["seed_input"],
            kind=data["kind"],
            spec_row=data["spec_row"],
            severity=data["severity"],
            timestamp=data["timestamp"],
            true_class=data["true_class"],
        )


# =============================================================================
# Property Checking
# =============================================================================

class PropertyChecker:
    """
    Vectorized property checker for violation detection.
    
    Supports all OutKind types through OutputSpec.violation():
    - TOP1_ROBUST: Top prediction must equal true label
    - MARGIN_ROBUST: Margin to runner-up must exceed threshold
    - RANGE: Output must be within [lb, ub]
    - LINEAR_LE: Linear constraint c^T y <= d must hold
    - UNSAFE_LINEAR: Output must not enter the unsafe linear region
    
    Example:
        >>> checker = PropertyChecker(output_spec)
        >>> violations = checker.check(inputs, outputs, labels)
        >>> # violations is List[Counterexample | None] of length B
    """
    
    def __init__(self, output_spec: Optional[OutputSpec]):
        """Initialize property checker."""
        self.spec = output_spec
        self.infeasible_candidates = 0
    
    def check(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        seeds: 'FuzzingSeed'
    ) -> Tuple[torch.Tensor, List[Counterexample]]:
        """
        Check B samples for violations in parallel.
        
        Args:
            inputs: Input tensors [B, C, H, W] or [B, D]
            outputs: Model outputs [B, num_classes]
            seeds: FuzzingSeed batch with labels, original tensors, and indices
        
        Returns:
            Tuple of (violations_mask BoolTensor[B], List[Counterexample] for violations only)
        """
        if self.spec is None:
            raise ValueError("PropertyChecker requires an OutputSpec")

        input_satisfied = getattr(outputs, INPUT_FEASIBILITY_ATTR, None)
        if not isinstance(input_satisfied, torch.Tensor):
            raise RuntimeError(
                "PropertyChecker requires lane-aware input feasibility metadata "
                "from VerifiableModel.forward"
            )
        input_satisfied = input_satisfied.to(
            device=outputs.device, dtype=torch.bool
        ).reshape(-1)
        if input_satisfied.shape != (inputs.shape[0],):
            raise RuntimeError(
                "Input feasibility mask must have shape "
                f"({inputs.shape[0]},), got {tuple(input_satisfied.shape)}"
            )
        
        violations_mask, severity = output_violations(
            outputs, self.spec, rows=seeds.original_index
        )
        rejected_mask = violations_mask & ~input_satisfied
        rejected_count = int(rejected_mask.sum().item())
        if rejected_count:
            self.infeasible_candidates += rejected_count
            print(
                "⚠️  [PropertyChecker] Rejected "
                f"{rejected_count} infeasible counterexample candidate(s) "
                f"(total: {self.infeasible_candidates})"
            )
        violations_mask = violations_mask & input_satisfied
        return self._build_results(
            inputs, outputs, violations_mask, severity, seeds=seeds
        )
    
    def _label_targets(self) -> Optional[torch.Tensor]:
        """Flattened y_true labels, or None when the spec kind carries no label."""
        assert self.spec is not None
        if self.spec.kind not in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            return None
        if self.spec.y_true is None:
            raise ValueError(f"{self.spec.kind} requires an OutputSpec with y_true")
        return self.spec.y_true.reshape(-1)
    
    def _build_results(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        violations_mask: torch.Tensor,
        severity: torch.Tensor,
        seeds: 'FuzzingSeed',
    ) -> Tuple[torch.Tensor, List[Counterexample]]:
        """Build Counterexample list from violation mask and return (mask, list)."""
        assert self.spec is not None
        timestamp = time.time()
        violation_indices = violations_mask.nonzero(as_tuple=True)[0]
        targets = self._label_targets()
        
        counterexamples: List[Counterexample] = []
        
        for idx in violation_indices:
            i = int(idx.item())
            spec_row = int(seeds.original_index[i].item())
            true_class: Optional[int] = None
            if targets is not None:
                # y_true is either shared across rows ([1]) or one label per row ([N]).
                target_index = 0 if targets.numel() == 1 else spec_row
                true_class = int(targets[target_index].item())
            counterexamples.append(Counterexample(
                input=inputs[i].detach().cpu(),
                output=outputs[i].detach().cpu(),
                seed_input=seeds.original_tensor[i].detach().cpu(),
                kind=self.spec.kind,
                spec_row=spec_row,
                severity=float(severity[i].item()),
                timestamp=timestamp,
                true_class=true_class,
            ))
        
        return violations_mask, counterexamples
