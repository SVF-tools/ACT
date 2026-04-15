"""
Coverage tracking for ACTFuzzer.

Tracks neuron coverage (method-level metrics) during fuzzing to guide exploration.

## How Coverage Is Collected

1. **Activation Capture**: MutationEngine registers forward hooks on computational layers
   (ReLU, Linear, Conv2d). During inference, these hooks capture layer activations into
   an activation_map dict keyed by layer name.

2. **Coverage Update**: After inference, ACTFuzzer calls CoverageTracker.update() with the
   mutated input and the activation_map. The tracker delegates to the active CoverageStrategy.

3. **Neuron Firing**: A neuron is considered "fired" (covered) if |activation| > threshold.
   For multi-dimensional activations (e.g., Conv2d [N,C,H,W]), spatial dims are max-pooled
   to produce per-channel coverage: one neuron per channel.

4. **Coverage Delta**: update() returns (global_delta, interesting_mask) — the coverage
   improvement and a per-sample BoolTensor marking which samples are interesting.
   ACTFuzzer uses these for energy computation and corpus scheduling.

## Coverage Strategies

1. **BestInputCov (BIC)**: Per-input coverage tracking. Each input gets its own coverage
   score (fraction of neurons fired). get_coverage() returns the best (max) per-input
   coverage seen so far. Does NOT maintain a global union — only individual input scores.

2. **GlobalCov (GLC)**: Global union coverage. A neuron is covered once it fires in ANY
   input across ALL iterations (monotonic). Coverage state is stored as per-layer
   BoolTensors on the activation device for O(1) vectorized lookup.

## Statistics

get_stats() returns strategy-specific metrics:
- BestInputCov: coverage, inputs_seen, last/best/avg input coverage, total neurons, layers
- GlobalCov: coverage, covered/total neurons, last newly covered count, layers

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Tuple
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
    def update(
        self, input_tensor: torch.Tensor, activations: Dict[str, torch.Tensor]
    ) -> Tuple[float, torch.Tensor]:
        """
        Update coverage with batch activations.
        
        Batch-native: processes all N samples in one vectorized call.
        
        Returns:
            (global_delta, interesting_mask) where:
            - global_delta: overall coverage improvement (float, 0..1)
            - interesting_mask: BoolTensor of shape (N,), True for samples that
              contributed to coverage improvement (should be added to corpus)
        """
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


def _activation_to_neuron_matrix(activation: torch.Tensor) -> torch.Tensor:
    """
    Convert (N,...) activation tensor to (N, neurons) matrix for coverage.
    
    - (N, neurons) [dim==2]: already correct → return as-is
    - (N, C, H, W) [dim==4]: max-pool spatial dims → (N, C)
    - Other: flatten all but batch dim → (N, K)
    """
    if activation.dim() == 2:
        return activation  # (N, neurons)
    if activation.dim() == 4:
        return activation.abs().amax(dim=(2, 3))  # (N, C)
    return activation.flatten(start_dim=1)  # (N, K)


class BestInputCov(CoverageStrategy):
    """
    Per-input neuron coverage.

    - Each update() computes coverage for that specific input only.
    - get_coverage() returns the best (max) per-input coverage seen so far (monotonic).
    """

    def __init__(self, model: nn.Module, threshold: float = 0.1):
        super().__init__(model, threshold)
        self._layer_neuron_counts: Dict[str, int] = {}
        self._inputs_seen: int = 0
        self._sum_coverage: float = 0.0
        self.last_input_coverage: float = 0.0
        self.best_input_coverage: float = 0.0

    def update(
        self, input_tensor: torch.Tensor, activations: Dict[str, torch.Tensor]
    ) -> Tuple[float, torch.Tensor]:
        """
        Update per-input neuron coverage with batch activations.
        
        Batch-native: computes coverage for all N samples in one vectorized call.
        A sample is "interesting" if its individual coverage exceeds the pre-batch best.
        
        Returns:
            (global_delta, interesting_mask):
            - global_delta: improvement in best_input_coverage (float)
            - interesting_mask: BoolTensor (N,), True for samples exceeding previous best
        """
        for layer_name, activation in activations.items():
            mat = _activation_to_neuron_matrix(activation)
            if mat.numel() == 0:
                continue
            self._layer_neuron_counts.setdefault(layer_name, mat.shape[1])
        
        total_neurons = int(sum(self._layer_neuron_counts.values()))
        if total_neurons == 0:
            N = input_tensor.shape[0]
            return 0.0, torch.zeros(N, dtype=torch.bool)
        
        old_best = float(self.best_input_coverage)
        N = input_tensor.shape[0]
        
        total_fired = torch.zeros(N)
        for layer_name, activation in activations.items():
            mat = _activation_to_neuron_matrix(activation)
            if mat.numel() == 0:
                continue
            total_fired += (mat > float(self.threshold)).sum(dim=1).float()
        
        sample_covs = total_fired / total_neurons  # (N,)
        
        interesting_mask = sample_covs > old_best
        
        # Running stats — single .sum()/.max() reductions, no per-sample .tolist()
        self._inputs_seen += N
        self._sum_coverage += float(sample_covs.sum())
        self.last_input_coverage = float(sample_covs[-1])
        batch_best = float(sample_covs.max())
        if batch_best > self.best_input_coverage:
            self.best_input_coverage = batch_best
        
        global_delta = max(0.0, float(self.best_input_coverage) - old_best)
        return global_delta, interesting_mask

    def get_coverage(self) -> float:
        return float(self.best_input_coverage)

    def get_stats(self) -> Dict[str, Any]:
        avg = (self._sum_coverage / self._inputs_seen) if self._inputs_seen else 0.0
        return {
            "coverage": float(self.get_coverage()),
            "inputs_seen": self._inputs_seen,
            "last_input_coverage": float(self.last_input_coverage),
            "best_input_coverage": float(self.best_input_coverage),
            "avg_input_coverage": float(avg),
            "total_neurons_seen": int(sum(self._layer_neuron_counts.values())),
            "layers_seen": int(len(self._layer_neuron_counts)),
        }

    def reset(self) -> None:
        self._layer_neuron_counts.clear()
        self._inputs_seen = 0
        self._sum_coverage = 0.0
        self.last_input_coverage = 0.0
        self.best_input_coverage = 0.0


class GlobalCov(CoverageStrategy):
    """
    Global union neuron coverage.

    A neuron is covered if it has fired at least once across all inputs.
    Coverage state is stored as per-layer BoolTensors on the same device as
    activations, enabling O(1) vectorized lookup instead of Python set iteration.
    """

    def __init__(self, model: nn.Module, threshold: float = 0.1):
        super().__init__(model, threshold)

        self._layer_neuron_counts: Dict[str, int] = {}
        self._covered_masks: Dict[str, torch.Tensor] = {}
        self.last_newly_covered_count: int = 0

    def _ensure_layer_registered(
        self, layer_name: str, neuron_count: int, device: torch.device
    ) -> None:
        neuron_count = int(neuron_count)
        if neuron_count <= 0:
            return
        if layer_name in self._layer_neuron_counts:
            return
        self._layer_neuron_counts[layer_name] = neuron_count
        self._covered_masks[layer_name] = torch.zeros(
            neuron_count, dtype=torch.bool, device=device
        )

    def update(
        self, input_tensor: torch.Tensor, activations: Dict[str, torch.Tensor]
    ) -> Tuple[float, torch.Tensor]:
        """
        Update global union neuron coverage with batch activations.
        
        Batch-native: processes all N samples in one call. A sample is "interesting"
        if it fires any neuron not covered before this batch (AFL-style: any input
        hitting a new edge is interesting, multiple samples can share credit).
        
        All tensor operations stay on input_tensor.device 
        
        Returns:
            (global_delta, interesting_mask):
            - global_delta: fraction of newly covered neurons (float)
            - interesting_mask: BoolTensor (N,), True for samples that fire
              at least one previously-uncovered neuron
        """
        N = input_tensor.shape[0]
        old_covered = self._total_covered()
        interesting_mask = torch.zeros(N, dtype=torch.bool)
        
        for layer_name, activation in activations.items():
            mat = _activation_to_neuron_matrix(activation)  # (N, neurons)
            if mat.numel() == 0:
                continue
            n_neurons = mat.shape[1]
            self._ensure_layer_registered(layer_name, n_neurons, device=mat.device)
            
            fired_mask = mat > float(self.threshold)  # (N, neurons)
            already_covered = self._covered_masks[layer_name]  # (neurons,)
            
            fired_any = fired_mask.any(dim=0)  # (neurons,)
            newly_covered = fired_any & ~already_covered  # (neurons,)
            
            if newly_covered.any():
                # (N, neurons) & (1, neurons) → (N, neurons) → any(dim=1) → (N,)
                interesting_mask |= (fired_mask & newly_covered.unsqueeze(0)).any(dim=1)
            
            # Update coverage mask in-place (bitwise OR)
            self._covered_masks[layer_name] = already_covered | fired_any
        
        new_covered = self._total_covered()
        self.last_newly_covered_count = new_covered - old_covered
        total_neurons = self._total_neurons()
        global_delta = (self.last_newly_covered_count / total_neurons) if total_neurons > 0 else 0.0
        return global_delta, interesting_mask

    def _total_covered(self) -> int:
        return sum(int(m.sum()) for m in self._covered_masks.values())

    def _total_neurons(self) -> int:
        return sum(self._layer_neuron_counts.values())

    def get_coverage(self) -> float:
        total = self._total_neurons()
        if total == 0:
            return 0.0
        return self._total_covered() / total

    def get_uncovered_neurons(self) -> Set[NeuronId]:
        result: Set[NeuronId] = set()
        for layer_name, mask in self._covered_masks.items():
            for idx in (~mask).nonzero(as_tuple=True)[0].tolist():
                result.add((layer_name, int(idx)))
        return result

    def get_covered_neurons(self) -> Set[NeuronId]:
        result: Set[NeuronId] = set()
        for layer_name, mask in self._covered_masks.items():
            for idx in mask.nonzero(as_tuple=True)[0].tolist():
                result.add((layer_name, int(idx)))
        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "coverage": float(self.get_coverage()),
            "covered_neurons": self._total_covered(),
            "total_neurons": self._total_neurons(),
            "last_newly_covered": int(self.last_newly_covered_count),
            "layers_seen": int(len(self._layer_neuron_counts)),
        }

    def reset(self) -> None:
        self._covered_masks.clear()
        self._layer_neuron_counts.clear()
        self.last_newly_covered_count = 0

class CoverageTracker:
    """
    Coverage engine that delegates to coverage strategies.
    
    Supports runtime strategy switching via update(strategy=...).
    Strategies are lazily initialized on first use.
    """

    _REGISTRY = {"BestInputCov": BestInputCov, "GlobalCov": GlobalCov}

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.1,
        strategy: str = "BestInputCov"
    ):
        self.model = model
        self.threshold = threshold
        self.strategy = strategy
        self._strategies: Dict[str, CoverageStrategy] = {}
        
        if strategy not in self._REGISTRY:
            raise ValueError(f"Unknown coverage strategy '{strategy}'. Valid: {list(self._REGISTRY.keys())}")

    def _get_strategy(self, name: str) -> CoverageStrategy:
        """Get or create strategy by name (lazy init)."""
        if name not in self._strategies:
            if name not in self._REGISTRY:
                raise ValueError(f"Unknown coverage strategy '{name}'. Valid: {list(self._REGISTRY.keys())}")
            self._strategies[name] = self._REGISTRY[name](model=self.model, threshold=self.threshold)
        return self._strategies[name]

    def update(
        self,
        input_tensor: torch.Tensor,
        activations: Dict[str, torch.Tensor],
        strategy: Optional[str] = None
    ) -> Tuple[float, torch.Tensor]:
        """
        Update coverage with activations from a fuzzing iteration.
        
        Batch-native: processes all N samples in one vectorized call.
        Called by ACTFuzzer after each inference.
        
        Args:
            input_tensor: Mutated input batch [N, ...]
            activations: Dict of layer activations from MutationEngine hooks
            strategy: Override default strategy for this call (optional)
        
        Returns:
            (global_delta, interesting_mask):
            - global_delta: overall coverage improvement (float, 0..1)
            - interesting_mask: BoolTensor (N,), True for interesting samples
        """
        s = strategy if strategy is not None else self.strategy
        return self._get_strategy(s).update(input_tensor, activations)

    def get_coverage(self, strategy: Optional[str] = None) -> float:
        s = strategy if strategy is not None else self.strategy
        return self._get_strategy(s).get_coverage()

    def get_stats(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        s = strategy if strategy is not None else self.strategy
        return {"strategy": s, **self._get_strategy(s).get_stats()}

    def get_uncovered_neurons(self, strategy: Optional[str] = None) -> Set[NeuronId]:
        s = strategy if strategy is not None else self.strategy
        return self._get_strategy(s).get_uncovered_neurons()

    def get_covered_neurons(self, strategy: Optional[str] = None) -> Set[NeuronId]:
        s = strategy if strategy is not None else self.strategy
        return self._get_strategy(s).get_covered_neurons()

