"""
Coverage tracking for ACTFuzzer.

Tracks neuron coverage (DeepXplore-style) during fuzzing to guide exploration.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from typing import Dict, Set
import torch
import torch.nn as nn


class CoverageTracker:
    """
    Track neuron coverage during fuzzing.
    
    Metrics:
    - Neuron coverage: % of neurons that have activated (output > threshold)
    - Total neurons: Count of all neurons in model
    - Covered neurons: Set of (layer_name, neuron_idx) tuples
    
    Example:
        >>> tracker = CoverageTracker(model)
        >>> activations = instrumentor.capture_activations(input_tensor)
        >>> coverage_delta = tracker.update(input_tensor, activations)
        >>> print(f"Coverage: {tracker.get_coverage():.2%}")
    """
    
    def __init__(self, model: nn.Module, threshold: float = 0.1):
        """
        Initialize coverage tracker.
        
        Args:
            model: Model to track coverage for
            threshold: Activation threshold (neuron is "active" if output > threshold)
        """
        self.model = model
        self.threshold = threshold
        
        # Track covered neurons as (layer_name, neuron_idx)
        self.covered_neurons: Set[tuple] = set()
        
        # Count total neurons
        self.total_neurons = self._count_neurons()
    
    def _count_neurons(self) -> int:
        """Count total neurons in model."""
        count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                count += module.out_features
            elif isinstance(module, nn.Conv2d):
                # For Conv2d, count output channels
                count += module.out_channels
        
        return count
    
    def update(self, 
               input_tensor: torch.Tensor,
               activations: Dict[str, torch.Tensor]
              ) -> float:
        """
        Update coverage with new activations.
        
        Args:
            input_tensor: Input that was tested (unused, for future extensions)
            activations: Dict of layer activations from instrumentation
        
        Returns:
            Coverage delta (increase in coverage from 0.0 to 1.0)
        """
        old_count = len(self.covered_neurons)
        
        # Process activations
        for layer_name, activation in activations.items():
            # Check if this is a layer we track
            if 'relu' in layer_name.lower() or 'linear' in layer_name.lower() or 'conv' in layer_name.lower():
                # Find neurons that fired (activation > threshold)
                fired_mask = (activation.abs() > self.threshold)
                
                # Get indices of fired neurons
                # Handle different tensor shapes
                if fired_mask.dim() == 2:
                    # Linear layer: (batch, neurons)
                    fired_indices = fired_mask[0].nonzero(as_tuple=True)[0].tolist()
                elif fired_mask.dim() == 4:
                    # Conv layer: (batch, channels, height, width)
                    # Track by channel
                    fired_indices = fired_mask[0].any(dim=(1, 2)).nonzero(as_tuple=True)[0].tolist()
                else:
                    # Flatten and track
                    fired_indices = fired_mask.flatten().nonzero(as_tuple=True)[0].tolist()
                
                # Add to covered set
                for idx in fired_indices:
                    self.covered_neurons.add((layer_name, idx))
        
        # Compute coverage delta
        new_count = len(self.covered_neurons)
        delta = (new_count - old_count) / self.total_neurons if self.total_neurons > 0 else 0.0
        
        return delta
    
    def get_coverage(self) -> float:
        """
        Get current coverage percentage.
        
        Returns:
            Coverage from 0.0 to 1.0
        """
        if self.total_neurons == 0:
            return 0.0
        
        return len(self.covered_neurons) / self.total_neurons
    
    def get_stats(self) -> Dict[str, float]:
        """Get detailed coverage statistics."""
        coverage = self.get_coverage()
        
        # Count covered neurons per layer
        layer_coverage = {}
        for layer_name, _ in self.covered_neurons:
            layer_coverage[layer_name] = layer_coverage.get(layer_name, 0) + 1
        
        return {
            "coverage": coverage,
            "covered_neurons": len(self.covered_neurons),
            "total_neurons": self.total_neurons,
            "layers_with_coverage": len(layer_coverage),
            "avg_neurons_per_layer": (
                sum(layer_coverage.values()) / len(layer_coverage)
                if layer_coverage else 0
            ),
        }
    
    def reset(self):
        """Reset coverage tracking."""
        self.covered_neurons.clear()


class DistinctNeuronCoverageTracker(CoverageTracker):
    """
    Enhanced coverage tracker that explicitly tracks distinct neurons.
    
    Key differences from base CoverageTracker:
    - Maintains explicit mapping of all neurons in the model
    - Tracks which specific neurons (by layer and index) have been covered
    - Provides methods to query uncovered neurons
    - Tracks activation statistics for each covered neuron
    
    A neuron is marked as "covered" the first time its activation exceeds the threshold.
    Once covered, it remains covered (distinct neuron coverage).
    
    Example:
        >>> tracker = DistinctNeuronCoverageTracker(model, threshold=0.1)
        >>> activations = instrumentor.capture_activations(input_tensor)
        >>> coverage_delta = tracker.update(input_tensor, activations)
        >>> print(f"Coverage: {tracker.get_coverage():.2%}")
        >>> print(f"Coverage delta: {coverage_delta:.4f}")
        >>> print(f"Uncovered: {len(tracker.get_uncovered_neurons())} neurons")
        >>> print(f"Newly covered: {tracker.get_newly_covered_count()} neurons")
    """
    
    def __init__(self, model: nn.Module, threshold: float = 0.1):
        """
        Initialize distinct neuron coverage tracker.
        
        Args:
            model: Model to track coverage for
            threshold: Activation threshold (neuron is "covered" if output > threshold)
        """
        super().__init__(model, threshold)
        
        # Build explicit mapping of all neurons in the model
        self.all_neurons: Set[tuple] = self._build_neuron_map()
        
        # Track activation statistics for covered neurons
        self.neuron_activation_count: Dict[tuple, int] = {}
        self.neuron_max_activation: Dict[tuple, float] = {}
        
        # Track most recent update stats (for debugging/analysis)
        self.last_newly_covered_count: int = 0
        
    def _build_neuron_map(self) -> Set[tuple]:
        """
        Build explicit set of all neurons in the model.
        
        Returns:
            Set of (layer_name, neuron_idx) tuples for all neurons
        """
        all_neurons = set()
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Add all output neurons for Linear layer
                for idx in range(module.out_features):
                    all_neurons.add((name, idx))
            elif isinstance(module, nn.Conv2d):
                # Add all output channels for Conv2d layer
                for idx in range(module.out_channels):
                    all_neurons.add((name, idx))
        
        return all_neurons
    
    def update(self, 
               input_tensor: torch.Tensor,
               activations: Dict[str, torch.Tensor]
              ) -> float:
        """
        Update coverage with new activations.
        
        Args:
            input_tensor: Input that was tested
            activations: Dict of layer activations from instrumentation
        
        Returns:
            Coverage delta (increase in coverage from 0.0 to 1.0)
        """
        old_count = len(self.covered_neurons)
        
        # Process activations
        for layer_name, activation in activations.items():
            # Check if this is a layer we track
            if 'relu' in layer_name.lower() or 'linear' in layer_name.lower() or 'conv' in layer_name.lower():
                # Find neurons that fired (activation > threshold)
                fired_mask = (activation.abs() > self.threshold)
                
                # Get indices of fired neurons and their activation values
                if fired_mask.dim() == 2:
                    # Linear layer: (batch, neurons)
                    fired_indices = fired_mask[0].nonzero(as_tuple=True)[0].tolist()
                    activation_values = activation[0].abs().tolist()
                elif fired_mask.dim() == 4:
                    # Conv layer: (batch, channels, height, width)
                    # Track by channel, use max activation per channel
                    fired_indices = fired_mask[0].any(dim=(1, 2)).nonzero(as_tuple=True)[0].tolist()
                    activation_values = activation[0].amax(dim=(1, 2)).abs().tolist()
                else:
                    # Flatten and track
                    fired_indices = fired_mask.flatten().nonzero(as_tuple=True)[0].tolist()
                    activation_values = activation.flatten().abs().tolist()
                
                # Add to covered set and update statistics
                for idx in fired_indices:
                    neuron_key = (layer_name, idx)
                    
                    # Mark as covered (set automatically handles duplicates)
                    self.covered_neurons.add(neuron_key)
                    
                    # Update statistics
                    self.neuron_activation_count[neuron_key] = (
                        self.neuron_activation_count.get(neuron_key, 0) + 1
                    )
                    
                    # Update max activation
                    if idx < len(activation_values):
                        current_max = self.neuron_max_activation.get(neuron_key, 0.0)
                        self.neuron_max_activation[neuron_key] = max(
                            current_max, activation_values[idx]
                        )
        
        # Compute coverage delta
        new_count = len(self.covered_neurons)
        self.last_newly_covered_count = new_count - old_count
        delta = self.last_newly_covered_count / self.total_neurons if self.total_neurons > 0 else 0.0
        
        return delta
    
    def get_newly_covered_count(self) -> int:
        """
        Get the number of neurons newly covered in the last update.
        
        Returns:
            Number of distinct neurons covered for the first time in last update
        """
        return self.last_newly_covered_count
    
    def get_uncovered_neurons(self) -> Set[tuple]:
        """
        Get set of neurons that have NOT been covered yet.
        
        Returns:
            Set of (layer_name, neuron_idx) tuples for uncovered neurons
        """
        return self.all_neurons - self.covered_neurons
    
    def get_covered_neurons(self) -> Set[tuple]:
        """
        Get set of neurons that have been covered.
        
        Returns:
            Set of (layer_name, neuron_idx) tuples for covered neurons
        """
        return self.covered_neurons.copy()
    
    def is_neuron_covered(self, layer_name: str, neuron_idx: int) -> bool:
        """
        Check if a specific neuron has been covered.
        
        Args:
            layer_name: Name of the layer
            neuron_idx: Index of the neuron in that layer
        
        Returns:
            True if neuron has been covered (activated > threshold at least once)
        """
        return (layer_name, neuron_idx) in self.covered_neurons
    
    def get_neuron_stats(self, layer_name: str, neuron_idx: int) -> Dict[str, float]:
        """
        Get activation statistics for a specific neuron.
        
        Args:
            layer_name: Name of the layer
            neuron_idx: Index of the neuron
        
        Returns:
            Dict with 'activation_count' and 'max_activation', or empty if not covered
        """
        neuron_key = (layer_name, neuron_idx)
        
        if neuron_key not in self.covered_neurons:
            return {}
        
        return {
            'activation_count': self.neuron_activation_count.get(neuron_key, 0),
            'max_activation': self.neuron_max_activation.get(neuron_key, 0.0),
        }
    
    def get_stats(self) -> Dict[str, any]:
        """Get detailed distinct neuron coverage statistics."""
        base_stats = super().get_stats()
        
        # Add distinct neuron specific stats
        uncovered = self.get_uncovered_neurons()
        
        # Coverage by layer
        layer_coverage_stats = {}
        for layer_name, neuron_idx in self.all_neurons:
            if layer_name not in layer_coverage_stats:
                layer_coverage_stats[layer_name] = {'total': 0, 'covered': 0}
            layer_coverage_stats[layer_name]['total'] += 1
        
        for layer_name, neuron_idx in self.covered_neurons:
            if layer_name in layer_coverage_stats:
                layer_coverage_stats[layer_name]['covered'] += 1
        
        # Compute per-layer coverage percentages
        layer_coverage_pct = {
            layer: stats['covered'] / stats['total'] if stats['total'] > 0 else 0.0
            for layer, stats in layer_coverage_stats.items()
        }
        
        distinct_stats = {
            'distinct_neurons_total': len(self.all_neurons),
            'distinct_neurons_covered': len(self.covered_neurons),
            'distinct_neurons_uncovered': len(uncovered),
            'coverage_percentage': self.get_coverage(),
            'layer_coverage': layer_coverage_pct,
            'neurons_activated_multiple_times': sum(
                1 for count in self.neuron_activation_count.values() if count > 1
            ),
        }
        
        # Merge with base stats
        return {**base_stats, **distinct_stats}
    
    def reset(self):
        """Reset all coverage tracking."""
        super().reset()
        self.neuron_activation_count.clear()
        self.neuron_max_activation.clear()
        self.last_newly_covered_count = 0
