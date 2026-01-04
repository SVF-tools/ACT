"""
Coverage tracking for ACTFuzzer.

Tracks neuron coverage (method-level metrics) during fuzzing to guide exploration.
Implements two coverage strategies:
1. PerInputCov (PIC): per-input mutation threshold coverage (stores only per-input coverage values; a neuron is marked covered once |activation| > threshold. Coverage = (#covered) / (#total neurons)).
2. GlobalCov (GLC): global union threshold coverage (a neuron stays covered once it exceeds the threshold across all mutated inputs).

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Tuple, List
import torch
import torch.nn as nn


NeuronId = Tuple[str, int]


class CoverageStrategy(ABC):
    """
    Coverage strategy interface.

    A strategy owns its own coverage state (covered set, totals, stats) and can be
    plugged into a tracker/engine (similar to `MutationStrategy` + `MutationEngine`).
    """

    def __init__(self, model: nn.Module, threshold: float = 0.1):
        self.model = model
        self.threshold = threshold

    @abstractmethod
    def update(self, input_tensor: torch.Tensor, activations: Dict[str, torch.Tensor]) -> float:
        """Update coverage with new activations; returns coverage delta (0..1)."""
        raise NotImplementedError

    @abstractmethod
    def get_coverage(self) -> float:
        """Return coverage in [0, 1]."""
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return strategy-specific coverage stats (JSON friendly)."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset internal coverage state."""
        raise NotImplementedError

    # Optional capabilities (implemented only by some strategies)
    def get_uncovered_neurons(self) -> Set[NeuronId]:
        raise NotImplementedError

    def get_covered_neurons(self) -> Set[NeuronId]:
        raise NotImplementedError


def _activation_to_neuron_vector(activation: torch.Tensor) -> torch.Tensor:
    """
    Convert an activation tensor into a per-"neuron" 1D vector (batch=first sample).
    - Linear/ReLU typical: (B, N) -> (N,)
    - Conv typical: (B, C, H, W) -> (C,) via max abs over spatial dims
    - Other: flatten (first sample) -> (K,)
    """
    a = activation
    if a.dim() >= 1 and a.size(0) > 0:
        a0 = a[0]
    else:
        a0 = a

    if a.dim() == 4:
        # a0: (C, H, W)
        return a0.abs().amax(dim=(1, 2))
    if a.dim() == 2:
        return a0.flatten()
    return a0.flatten()


class PerInputCov(CoverageStrategy):
    """
    Per-input neuron coverage.

    - Each update() computes coverage for that specific input only.
    - We store per-input coverage values (history), not a global union of covered neurons.
    - get_coverage() returns the best (max) per-input coverage seen so far (monotonic).
    """

    def __init__(self, model: nn.Module, threshold: float = 0.1):
        super().__init__(model, threshold)
        self._layer_neuron_counts: Dict[str, int] = {}
        self.coverage_history: List[float] = []
        self.last_input_coverage: float = 0.0
        self.best_input_coverage: float = 0.0

    def update(self, input_tensor: torch.Tensor, activations: Dict[str, torch.Tensor]) -> float:
        covered_this_input: Set[NeuronId] = set()

        for layer_name, activation in activations.items():
            vec = _activation_to_neuron_vector(activation)
            if vec.numel() == 0:
                continue
            self._layer_neuron_counts.setdefault(layer_name, int(vec.numel()))
            fired = (vec.abs() > float(self.threshold)).nonzero(as_tuple=True)[0].tolist()
            for idx in fired:
                covered_this_input.add((layer_name, int(idx)))

        total_neurons = int(sum(self._layer_neuron_counts.values()))
        self.last_input_coverage = (len(covered_this_input) / total_neurons) if total_neurons > 0 else 0.0
        self.coverage_history.append(float(self.last_input_coverage))

        old_best = float(self.best_input_coverage)
        if self.last_input_coverage > self.best_input_coverage:
            self.best_input_coverage = float(self.last_input_coverage)
        return max(0.0, float(self.best_input_coverage) - old_best)

    def get_coverage(self) -> float:
        return float(self.best_input_coverage)

    def get_stats(self) -> Dict[str, Any]:
        n = len(self.coverage_history)
        avg = (sum(self.coverage_history) / n) if n else 0.0
        return {
            "coverage": float(self.get_coverage()),
            "inputs_seen": int(n),
            "last_input_coverage": float(self.last_input_coverage),
            "best_input_coverage": float(self.best_input_coverage),
            "avg_input_coverage": float(avg),
            "total_neurons_seen": int(sum(self._layer_neuron_counts.values())),
            "layers_seen": int(len(self._layer_neuron_counts)),
        }

    def reset(self) -> None:
        self._layer_neuron_counts.clear()
        self.coverage_history.clear()
        self.last_input_coverage = 0.0
        self.best_input_coverage = 0.0


class GlobalCov(CoverageStrategy):
    """
    Global union neuron coverage.

    A neuron is covered if it has fired at least once across all inputs.
    """

    def __init__(self, model: nn.Module, threshold: float = 0.1):
        super().__init__(model, threshold)

        self.all_neurons: Set[NeuronId] = set()
        self._layer_neuron_counts: Dict[str, int] = {}
        self.covered_neurons: Set[NeuronId] = set()
        self.last_newly_covered_count: int = 0

    def _ensure_layer_registered(self, layer_name: str, neuron_count: int) -> None:
        neuron_count = int(neuron_count)
        if neuron_count <= 0:
            return
        if layer_name in self._layer_neuron_counts:
            return
        self._layer_neuron_counts[layer_name] = neuron_count
        for idx in range(neuron_count):
            self.all_neurons.add((layer_name, idx))

    def update(self, input_tensor: torch.Tensor, activations: Dict[str, torch.Tensor]) -> float:
        old_count = len(self.covered_neurons)

        for layer_name, activation in activations.items():
            vec = _activation_to_neuron_vector(activation)
            if vec.numel() == 0:
                continue
            self._ensure_layer_registered(layer_name, int(vec.numel()))
            fired_indices = (vec.abs() > float(self.threshold)).nonzero(as_tuple=True)[0].tolist()
            for idx in fired_indices:
                self.covered_neurons.add((layer_name, int(idx)))

        new_count = len(self.covered_neurons)
        self.last_newly_covered_count = new_count - old_count
        total_neurons = len(self.all_neurons)
        return (self.last_newly_covered_count / total_neurons) if total_neurons > 0 else 0.0

    def get_coverage(self) -> float:
        total_neurons = len(self.all_neurons)
        if total_neurons == 0:
            return 0.0
        return len(self.covered_neurons) / total_neurons

    def get_uncovered_neurons(self) -> Set[NeuronId]:
        return self.all_neurons - self.covered_neurons

    def get_covered_neurons(self) -> Set[NeuronId]:
        return self.covered_neurons.copy()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "coverage": float(self.get_coverage()),
            "covered_neurons": int(len(self.covered_neurons)),
            "total_neurons": int(len(self.all_neurons)),
            "last_newly_covered": int(self.last_newly_covered_count),
            "layers_seen": int(len(self._layer_neuron_counts)),
        }

    def reset(self) -> None:
        self.covered_neurons.clear()
        self.all_neurons.clear()
        self._layer_neuron_counts.clear()
        self.last_newly_covered_count = 0

class CoverageTracker:
    """
    Coverage "engine" that owns multiple coverage strategies `CoverageTracker` delegates to a chosen strategy.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.1,
        strategy: str = "pic"
    ):
        self.model = model
        self.threshold = threshold

        # Initialize coverage strategies
        pic = PerInputCov(model=model, threshold=threshold)
        glc = GlobalCov(model=model, threshold=threshold)

        # Canonical strategy names, a placeholder for future dynamic strategy registration
        self.strategies: Dict[str, CoverageStrategy] = {"pic": pic, "glc": glc}

        if strategy not in self.strategies:
            raise ValueError(f"Unknown coverage strategy '{strategy}'. Valid: {list(self.strategies.keys())}")
        self.strategy = strategy

    def update(
        self,
        input_tensor: torch.Tensor,
        activations: Dict[str, torch.Tensor],
        strategy: Optional[str] = None,
        
    ) -> float:
        
        s = self.strategy if strategy is None else strategy
        if s not in self.strategies:
            raise ValueError(f"Unknown coverage strategy '{s}'. Valid: {list(self.strategies.keys())}")
        return self.strategies[s].update(input_tensor, activations)

    def get_coverage(self) -> float:
        return self.strategies[self.strategy].get_coverage()

    def get_stats(self) -> Dict[str, Any]:
        base = {
            "strategy": self.strategy,
            "available_strategies": ["pic", "glc"],
        }
        return {**base, **self.strategies[self.strategy].get_stats()}

    # Convenience delegation for strategies that support uncovered/covered queries
    def get_uncovered_neurons(self) -> Set[NeuronId]:
        strat = self.strategies[self.strategy]
        return strat.get_uncovered_neurons()

    def get_covered_neurons(self) -> Set[NeuronId]:
        strat = self.strategies[self.strategy]
        return strat.get_covered_neurons()

