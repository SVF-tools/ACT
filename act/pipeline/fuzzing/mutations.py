"""
Mutation strategies for ACTFuzzer.

Implements gradient-guided, activation-guided, boundary, and random mutations.
All mutations automatically respect InputSpec constraints via projection.

## Adaptive Perturbation Sizing

NOTE: We use "perturb_size" (not "epsilon") to avoid confusion with InputSpec.eps (L∞ radius).
- InputSpec.eps: Defines constraint boundaries (e.g., center ± eps for LINF_BALL)
- Mutation perturb_size: Controls mutation perturbation magnitude (exploration granularity)

This module supports adaptive perturbation sizing that scales with InputSpec bounds to ensure
consistent exploration across different problem scales.

### What is perturb_scale?

`perturb_scale` is the **fraction of the feasible range** that each mutation perturbation covers.

**Interpretation Formula:**
    steps_to_traverse = 1 / perturb_scale

**Calculation:**
    range / perturb_size = range / (range * perturb_scale) = 1 / perturb_scale

**Examples:**
    - perturb_scale=0.1  → Each perturbation covers 10% of range → Takes ~10 steps to traverse from lb to ub
    - perturb_scale=0.2  → Each perturbation covers 20% of range → Takes ~5 steps to traverse from lb to ub
    - perturb_scale=0.05 → Each perturbation covers 5% of range  → Takes ~20 steps to traverse from lb to ub

### Perturbation Modes

1. **adaptive_scalar** (default):
   - Computes single perturb_size from mean range: perturb_size = mean(ub - lb) * perturb_scale
   - Best for: Uniform ranges (e.g., VNNLib BOX constraints with consistent bounds)
   - Example: VNNLib with lb=0.0, ub=1.0 → range=1.0, perturb_size=0.1 (10 steps)

2. **adaptive_perdim** (advanced):
   - Computes per-dimension perturb_size tensor: perturb_size[i] = (ub[i] - lb[i]) * perturb_scale
   - Best for: Non-uniform ranges (e.g., different features with vastly different scales)
   - Example: lb=[0, -100], ub=[1, 100] → perturb_size=[0.1, 20.0] (10 steps per dimension)

3. **fixed** (legacy):
   - Uses hardcoded perturb_size values (0.01 for gradient/activation, 0.005 for boundary/random)
   - Best for: Backward compatibility or when InputSpec is not available
   - Note: May be too large for tight bounds or too small for wide bounds

### Configuration

Set in `act/pipeline/fuzzing/config.yaml`:
```yaml
perturb_mode: "adaptive_scalar"  # Options: "adaptive_scalar", "adaptive_perdim", "fixed"
perturb_scale: 0.1               # Fraction of range per step (default: 0.1 = 10 steps)
```

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import torch.nn as nn
import numpy as np

from act.front_end.specs import InputSpec, InKind


class MutationStrategy(ABC):
    """Base class for mutation strategies."""
    
    @abstractmethod
    def mutate(self, 
               input_tensor: torch.Tensor,
               model: nn.Module,
               activations: Optional[Dict[str, torch.Tensor]] = None
              ) -> torch.Tensor:
        """
        Apply mutation to input tensor.
        
        Args:
            input_tensor: Seed input
            model: Model for gradient computation
            activations: Activations from previous inference (optional)
        
        Returns:
            Mutated input tensor
        """
        pass


class GradientMutation(MutationStrategy):
    """
    Gradient-guided mutation with FGSM and PGD support.
    
    Computes gradients to maximize output variance, then applies
    either FGSM (single-step) or PGD (iterative) perturbation.
    """
    
    def __init__(self, perturb_size: Union[float, torch.Tensor] = 8/255, num_steps: int = 10, alpha: Optional[float] = None):
        """
        Initialize gradient mutation.
        
        Args:
            perturb_size: Mutation perturbation magnitude for FGSM (scalar or per-dimension tensor)
            num_steps: Number of PGD iterations (default: 1 for FGSM, use >1 for PGD)
            alpha: Step size per PGD iteration (default: perturb_size / num_steps * 2.5 if num_steps > 1)
        """
        self.perturb_size = perturb_size
        self.num_steps = num_steps
        
        # For PGD, set alpha if num_steps > 1
        if num_steps > 1:
            self.alpha = alpha if alpha is not None else self.perturb_size / num_steps * 2.5
        
    
    def mutate(self, input_tensor, model, activations=None):
        """Apply gradient-based perturbation."""
        if self.num_steps == 1:
            return self.mutate_fgsm(input_tensor, model, activations)
        else:
            return self.mutate_pgd(input_tensor, model, activations)
    
    def mutate_fgsm(self, input_tensor, model, activations=None):
        """Apply FGSM gradient-based perturbation (single-step)."""
        # Enable gradients
        x = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = model(x)
        
        # Extract output tensor if dict (from VerifiableModel)
        if isinstance(output, dict):
            output = output['output']
        
        # Compute loss: maximize output variance
        loss = output.var()
        
        # Get gradient w.r.t. input only (avoid accumulating grads on model params)
        grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0].detach()
        
        # FGSM: sign of gradient
        # Handle both scalar and tensor perturb_size
        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        perturbation = perturb_size * torch.sign(grad)
        
        # Apply perturbation
        return input_tensor + perturbation
    
    def mutate_pgd(self, input_tensor, model, activations=None):
        """
        Apply PGD (Projected Gradient Descent) gradient-based perturbation.
        
        Uses iterative gradient ascent with projection to epsilon-ball.
        More powerful than FGSM but computationally more expensive.
        
        Args:
            input_tensor: Seed input
            model: Model for gradient computation
            activations: Activations from previous inference (optional)
        
        Returns:
            Mutated input tensor
        """
        # Store original input
        x_orig = input_tensor.clone().detach()
        
        # Handle both scalar and tensor perturb_size (tensor enables per-dimension eps)
        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        
        # Initialize perturbed input with small random noise
        # Note: Tensor bounds are not supported by Tensor.uniform_, so sample in [-1, 1] and scale.
        x = x_orig + torch.empty_like(x_orig).uniform_(-1.0, 1.0) * perturb_size
        
        # PGD iterations
        for _ in range(self.num_steps):
            # Enable gradients
            x = x.clone().detach().requires_grad_(True)
            
            # Forward pass
            output = model(x)
            
            # Extract output tensor if dict (from VerifiableModel)
            if isinstance(output, dict):
                output = output['output']
            
            # Compute loss: maximize output variance
            loss = output.var()
            
            # Get gradient w.r.t. input only (avoid accumulating grads on model params)
            grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0].detach()
            
            # Take a step in the direction of the gradient (maximize variance)
            x_adv = x.detach() + self.alpha * torch.sign(grad)
            
            # Project back to epsilon-ball around original input
            perturbation = torch.clamp(x_adv - x_orig, -perturb_size, perturb_size)
            x = x_orig + perturbation
        
        return x.detach()


class ActivationMutation(MutationStrategy):
    """
    Mutation to maximize neuron activation changes.
    
    Uses random direction weighted by recent activation patterns.
    """
    
    def __init__(self, perturb_size: Union[float, torch.Tensor] = 0.01):
        """
        Initialize activation mutation.
        
        Args:
            perturb_size: Mutation perturbation magnitude (scalar or per-dimension tensor)
        """
        self.perturb_size = perturb_size
    
    def mutate(self, input_tensor, model, activations=None):
        """Apply activation-guided perturbation."""
        # Random direction (future: weight by inactive neurons)
        direction = torch.randn_like(input_tensor)
        
        # Normalize and scale
        direction = direction / (direction.norm() + 1e-8)
        # Handle both scalar and tensor perturb_size
        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        perturbation = perturb_size * direction
        
        return input_tensor + perturbation


class BoundaryMutation(MutationStrategy):
    """
    Mutation toward InputSpec boundaries.
    
    Explores edge cases where properties are more likely to fail.
    """
    
    def __init__(self, perturb_size: Union[float, torch.Tensor] = 0.005):
        """
        Initialize boundary mutation.
        
        Args:
            perturb_size: Mutation perturbation magnitude toward boundary (scalar or per-dimension tensor)
        """
        self.perturb_size = perturb_size
    
    def mutate(self, input_tensor, model, activations=None):
        """Push toward boundaries (will be projected by engine)."""
        # Random direction
        direction = torch.sign(torch.randn_like(input_tensor))
        
        # Scale
        # Handle both scalar and tensor perturb_size
        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        perturbation = perturb_size * direction
        
        return input_tensor + perturbation


class RandomMutation(MutationStrategy):
    """Random Gaussian perturbation (baseline)."""
    
    def __init__(self, perturb_size: Union[float, torch.Tensor] = 0.005):
        """
        Initialize random mutation.
        
        Args:
            perturb_size: Standard deviation of Gaussian noise (scalar or per-dimension tensor)
        """
        self.perturb_size = perturb_size
    
    def mutate(self, input_tensor, model, activations=None):
        """Apply random Gaussian noise."""
        # Handle both scalar and tensor perturb_size
        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        noise = torch.randn_like(input_tensor) * perturb_size
        return input_tensor + noise


class MutationEngine:
    """
    Mutation engine with strategy selection and constraint projection.
    
    Features:
    - Weighted random strategy selection
    - Automatic InputSpec projection
    - Activation capture via forward hooks
    - Statistics tracking
    
    Example:
        >>> engine = MutationEngine(model, input_spec, weights, device)
        >>> mutated = engine.mutate(seed_tensor)
        >>> activations = engine.get_last_activations()
    """
    
    def __init__(self,
                 model: nn.Module,
                 input_spec: Optional[InputSpec],
                 weights: Dict[str, float],
                 device: torch.device,
                 perturb_mode: str = "fixed",
                 perturb_scale: float = 0.1):
        """
        Initialize mutation engine.
        
        Args:
            model: Model for gradient computation
            input_spec: InputSpec for constraint projection
            weights: Strategy weights (e.g., {"gradient": 0.4, "random": 0.1})
            device: Torch device
            perturb_mode: Perturbation size computation mode ("adaptive_scalar", "adaptive_perdim", "fixed")
            perturb_scale: Fraction of range per mutation perturbation (e.g., 0.1 = 10% = ~10 steps to traverse)
        """
        self.model = model
        self.input_spec = input_spec
        self.device = device
        self.perturb_mode = perturb_mode
        self.perturb_scale = perturb_scale
        
        # Compute perturb_size based on mode
        perturb_size = self._compute_adaptive_perturb_size()
        
        # Initialize strategies with computed perturb_size
        self.strategies = {
            "gradient": GradientMutation(perturb_size=perturb_size),
            "activation": ActivationMutation(perturb_size=perturb_size),
            "boundary": BoundaryMutation(perturb_size=perturb_size * 0.5),  # Half perturb_size for boundary (more conservative)
            "random": RandomMutation(perturb_size=perturb_size * 0.5)       # Half perturb_size for random (more conservative)
        }
        
        # Normalize weights
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
        
        # Statistics
        self.total_mutations = 0
        self.last_activations: Dict[str, torch.Tensor] = {}
        self.last_strategy: Optional[str] = None  # NEW: track last mutation strategy
        self.last_gradients: Optional[Dict[str, torch.Tensor]] = None  # NEW: for Level 3 tracing
        self.last_loss: Optional[float] = None  # NEW: for Level 3 tracing
        
        # Neuron activation statistics
        self.last_neuron_stats: Dict[str, Dict[str, Any]] = {}  # Per-layer neuron statistics (latest hook result)
        self.last_network_neuron_stats: Dict[str, Any] = {}  # Whole-network aggregated stats (latest forward)
        # Controls for printing per-hook neuron stats (can be very verbose)
        self.neuron_stats_enabled: bool = True
        # Print once per *forward* (network-level), not per layer hook.
        self.neuron_stats_print: bool = True
        self.neuron_stats_print_per_layer: bool = False
        self.neuron_stats_topk: int = 8                # print/store top-k |activation| neurons per layer
        self.neuron_stats_threshold: float = 0.001       # for counting "active" neurons (|a| > threshold)
        self._forward_id: int = 0
        
        # Setup hooks for activation capture
        self._setup_hooks()
    
    def _compute_adaptive_perturb_size(self) -> Union[float, torch.Tensor]:
        """
        Compute perturb_size based on InputSpec bounds and perturb_mode.
        
        Note: We use "perturb_size" to avoid confusion with InputSpec.eps (L∞ radius constraint).
        
        Returns:
            - float: Scalar perturb_size (for "adaptive_scalar" or "fixed" modes)
            - torch.Tensor: Per-dimension perturb_size (for "adaptive_perdim" mode)
        
        Algorithm:
            1. adaptive_scalar: perturb_size = mean(ub - lb) * perturb_scale
               - Single perturb_size value computed from mean range
               - Best for uniform ranges (e.g., VNNLib BOX constraints)
            
            2. adaptive_perdim: perturb_size = (ub - lb) * perturb_scale
               - Tensor of perturb_size values, one per dimension
               - Best for non-uniform ranges (different feature scales)
            
            3. fixed: Uses hardcoded defaults (backward compatibility)
               - gradient/activation: 0.01
               - boundary/random: 0.005
        
        Interpretation:
            perturb_scale represents the fraction of range each perturbation covers.
            steps_to_traverse = 1 / perturb_scale
            
            Examples:
                - perturb_scale=0.1  → 10% per perturbation → ~10 steps to traverse
                - perturb_scale=0.2  → 20% per perturbation → ~5 steps to traverse
                - perturb_scale=0.05 → 5% per perturbation  → ~20 steps to traverse
        """
        if self.perturb_mode == "fixed":
            # Legacy fixed perturbation sizes (backward compatibility)
            print(f"[MutationEngine] Using fixed perturb_size mode (legacy)")
            print(f"  - Gradient/Activation perturb_size: 0.01")
            print(f"  - Boundary/Random perturb_size: 0.005")
            return 0.01  # Default for gradient/activation (will be halved for boundary/random)
        
        if self.input_spec is None:
            print(f"[MutationEngine] No InputSpec provided, falling back to fixed perturb_size=0.01")
            return 0.01
        
        # Extract bounds based on InputSpec kind
        if self.input_spec.kind == InKind.BOX:
            lb = self.input_spec.lb
            ub = self.input_spec.ub
        elif self.input_spec.kind == InKind.LINF_BALL:
            # For L∞ ball, range is 2*eps around center
            # Note: InputSpec.eps is the L∞ radius (constraint boundary), different from mutation perturb_size
            lb = self.input_spec.center - self.input_spec.eps
            ub = self.input_spec.center + self.input_spec.eps
        else:
            # LIN_POLY or other unsupported kinds
            print(f"[MutationEngine] Unsupported InputSpec kind '{self.input_spec.kind}', falling back to fixed perturb_size=0.01")
            return 0.01
        
        # Compute range
        range_tensor = ub - lb  # Shape: same as input tensor
        
        if self.perturb_mode == "adaptive_scalar":
            # Compute single perturb_size from mean range
            mean_range = range_tensor.mean().item()
            perturb_size = mean_range * self.perturb_scale
            
            # Diagnostic output
            print(f"[MutationEngine] Adaptive Scalar Perturbation Size:")
            print(f"  - perturb_scale: {self.perturb_scale} (fraction of range per perturbation)")
            print(f"  - mean_range: {mean_range:.6f}")
            print(f"  - computed perturb_size: {perturb_size:.6f}")
            print(f"  - steps_to_traverse: ~{1/self.perturb_scale:.1f} steps")
            print(f"  - interpretation: Each mutation perturbation covers {self.perturb_scale*100:.1f}% of the range")
            
            return perturb_size
        
        elif self.perturb_mode == "adaptive_perdim":
            # Compute per-dimension perturb_size tensor
            perturb_size_tensor = range_tensor * self.perturb_scale
            
            # Diagnostic output
            print(f"[MutationEngine] Adaptive Per-Dimension Perturbation Size:")
            print(f"  - perturb_scale: {self.perturb_scale} (fraction of range per perturbation)")
            print(f"  - range shape: {range_tensor.shape}")
            print(f"  - perturb_size shape: {perturb_size_tensor.shape}")
            print(f"  - perturb_size range: [{perturb_size_tensor.min().item():.6f}, {perturb_size_tensor.max().item():.6f}]")
            print(f"  - perturb_size mean: {perturb_size_tensor.mean().item():.6f}")
            print(f"  - steps_to_traverse: ~{1/self.perturb_scale:.1f} steps per dimension")
            print(f"  - interpretation: Each mutation perturbation covers {self.perturb_scale*100:.1f}% of each dimension's range")
            
            return perturb_size_tensor
        
        else:
            raise ValueError(f"Unknown perturb_mode: {self.perturb_mode}. "
                           f"Valid options: 'adaptive_scalar', 'adaptive_perdim', 'fixed'")
    
    def _setup_hooks(self):
        """Setup forward hooks to capture activations."""
        hook_count = 0
        
        # Track hook trigger count for debugging
        self._hook_trigger_count = 0

        # Reset buffers at the start of each full-model forward
        def _pre_forward_hook(module, inputs):
            self._forward_id += 1
            self.last_activations.clear()
            self.last_neuron_stats.clear()
            self.last_network_neuron_stats.clear()
            self._hook_trigger_count = 0

        # Print once at the end of each full-model forward
        def _post_forward_hook(module, inputs, output):
            if self.neuron_stats_enabled and self.neuron_stats_print:
                self._compute_network_neuron_stats()
                self._print_network_neuron_stats()

        # Attach hooks on the *whole model* so we print once per forward pass.
        # Works for VerifiableModel/nn.Sequential etc.
        self.model.register_forward_pre_hook(_pre_forward_hook)
        self.model.register_forward_hook(_post_forward_hook)
        
        def make_hook(name):
            def hook(module, input, output):
                # Increment trigger counter
                self._hook_trigger_count += 1
                
                # Store activation (handle both tensor and dict outputs)
                activation = None
                if isinstance(output, torch.Tensor):
                    activation = output.detach()
                    self.last_activations[name] = activation
                elif isinstance(output, dict) and 'output' in output:
                    activation = output['output'].detach()
                    self.last_activations[name] = activation
                
                # Compute neuron-level statistics
                if activation is not None:
                    self._compute_neuron_stats(name, activation)
            
            return hook
        
        # Register hooks on computational layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Linear, nn.Conv2d)):
                module.register_forward_hook(make_hook(name))
                hook_count += 1
        
        print(f"[MutationEngine] Registered {hook_count} activation hooks on model layers")
        if hook_count == 0:
            print(f"[MutationEngine] WARNING: No hooks registered! Model may not contain ReLU/Linear/Conv2d layers.")

    def _activation_to_neuron_vector(self, activation: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Convert an activation tensor into a 1D 'neuron' vector for stats.
        - Linear/ReLU typical: (B, N) -> per-neuron
        - Conv typical: (B, C, H, W) -> per-channel (reduce spatial)
        - Other shapes: flatten per-sample
        Returns:
            (vec, mode) where vec is 1D tensor on CPU and mode describes mapping.
        """
        # Ensure we only look at first sample in batch (most of this code assumes batch=1 anyway)
        a = activation
        if a.dim() >= 1 and a.size(0) > 0:
            a0 = a[0]
        else:
            a0 = a

        # Conv: treat each channel as a neuron (reduce H,W with max abs)
        if a.dim() == 4:
            # a0: (C, H, W)
            vec = a0.abs().amax(dim=(1, 2))
            mode = "conv:per_channel(max_abs)"
        # Linear/ReLU common: (N,) or (B,N)
        elif a.dim() == 2:
            # a0: (N,)
            vec = a0.flatten()
            mode = "dense:per_unit"
        else:
            vec = a0.flatten()
            mode = f"flat:{tuple(a0.shape)}"

        return vec.detach().to("cpu"), mode

    def _compute_neuron_stats(self, layer_name: str, activation: torch.Tensor) -> Dict[str, Any]:
        """
        Compute and optionally print neuron-level statistics for a single activation tensor.
        Stores latest stats in self.last_neuron_stats[layer_name].
        """
        if not self.neuron_stats_enabled:
            return {}

        vec, mode = self._activation_to_neuron_vector(activation)
        if vec.numel() == 0:
            stats: Dict[str, Any] = {
                "layer": layer_name,
                "shape": tuple(activation.shape),
                "mode": mode,
                "num_neurons": 0,
            }
            self.last_neuron_stats[layer_name] = stats
            return stats

        # Basic stats
        abs_vec = vec.abs()
        thr = float(self.neuron_stats_threshold)
        active_count = int((abs_vec > thr).sum().item())
        num = int(vec.numel())

        # Top-k by absolute value
        k = max(1, int(self.neuron_stats_topk))
        k = min(k, num)
        top_vals, top_idx = torch.topk(abs_vec, k=k, largest=True, sorted=True)
        # Keep signed values too
        signed_top_vals = vec[top_idx]

        stats = {
            "layer": layer_name,
            "shape": tuple(activation.shape),
            "mode": mode,
            "num_neurons": num,
            "threshold": thr,
            "active_count": active_count,
            "active_ratio": active_count / num if num > 0 else 0.0,
            "min": float(vec.min().item()),
            "max": float(vec.max().item()),
            "mean": float(vec.mean().item()),
            "std": float(vec.std(unbiased=False).item()) if num > 1 else 0.0,
            "topk": [
                {
                    "index": int(i.item()),
                    "abs": float(av.item()),
                    "value": float(v.item()),
                }
                for i, av, v in zip(top_idx, top_vals, signed_top_vals)
            ],
        }

        self.last_neuron_stats[layer_name] = stats

        if self.neuron_stats_print_per_layer:
            topk_str = ", ".join(
                [f"{t['index']}:{t['value']:.4f}(|.|={t['abs']:.4f})" for t in stats["topk"]]
            )
            print(
                f"[ActivationStats:Layer] layer={layer_name} shape={stats['shape']} mode={mode} "
                f"neurons={num} active(|a|>{thr})={active_count}({stats['active_ratio']:.2%}) "
                f"mean={stats['mean']:.6f} std={stats['std']:.6f} min={stats['min']:.6f} max={stats['max']:.6f} "
                f"top{len(stats['topk'])}=[{topk_str}]"
            )

        return stats

    def _compute_network_neuron_stats(self) -> Dict[str, Any]:
        """
        Aggregate neuron stats across the whole network based on last_activations
        captured during the most recent forward pass.
        """
        if not self.neuron_stats_enabled:
            self.last_network_neuron_stats = {}
            return self.last_network_neuron_stats

        thr = float(self.neuron_stats_threshold)
        topk = max(1, int(self.neuron_stats_topk))

        total_neurons = 0
        active_neurons = 0
        global_min = None
        global_max = None
        sum_val = 0.0
        sum_sq = 0.0

        # Global top-k by abs across all layers: store (abs, layer, idx, value)
        # We'll keep a small Python list and prune to topk.
        global_top: List[Tuple[float, str, int, float]] = []

        for layer_name, act in self.last_activations.items():
            vec, mode = self._activation_to_neuron_vector(act)
            if vec.numel() == 0:
                continue

            num = int(vec.numel())
            total_neurons += num

            abs_vec = vec.abs()
            active_neurons += int((abs_vec > thr).sum().item())

            vmin = float(vec.min().item())
            vmax = float(vec.max().item())
            global_min = vmin if global_min is None else min(global_min, vmin)
            global_max = vmax if global_max is None else max(global_max, vmax)

            # Mean/std via sums to avoid concatenating huge tensors
            sum_val += float(vec.sum().item())
            sum_sq += float((vec * vec).sum().item())

            # Merge top-k candidates from this layer
            k = min(topk, num)
            top_vals, top_idx = torch.topk(abs_vec, k=k, largest=True, sorted=True)
            signed_top_vals = vec[top_idx]
            for i, av, sv in zip(top_idx.tolist(), top_vals.tolist(), signed_top_vals.tolist()):
                global_top.append((float(av), layer_name, int(i), float(sv)))

            # prune occasionally
            if len(global_top) > topk * 8:
                global_top.sort(key=lambda x: x[0], reverse=True)
                global_top = global_top[:topk]

        if total_neurons == 0:
            stats: Dict[str, Any] = {
                "forward_id": self._forward_id,
                "layers_seen": len(self.last_activations),
                "total_neurons": 0,
                "active_neurons": 0,
                "active_ratio": 0.0,
                "threshold": thr,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "topk": [],
            }
            self.last_network_neuron_stats = stats
            return stats

        mean = sum_val / total_neurons
        var = max(0.0, (sum_sq / total_neurons) - (mean * mean))
        std = var ** 0.5

        global_top.sort(key=lambda x: x[0], reverse=True)
        global_top = global_top[:topk]

        stats = {
            "forward_id": self._forward_id,
            "layers_seen": len(self.last_activations),
            "total_neurons": total_neurons,
            "active_neurons": active_neurons,
            "active_ratio": active_neurons / total_neurons if total_neurons > 0 else 0.0,
            "threshold": thr,
            "min": global_min,
            "max": global_max,
            "mean": mean,
            "std": std,
            "topk": [
                {"abs": av, "layer": layer, "index": idx, "value": val}
                for (av, layer, idx, val) in global_top
            ],
            "hook_triggers": int(self._hook_trigger_count),
        }

        self.last_network_neuron_stats = stats
        return stats

    def _print_network_neuron_stats(self) -> None:
        stats = self.last_network_neuron_stats
        if not stats:
            return
        topk = stats.get("topk", [])
        topk_str = ", ".join(
            [f"{t['layer']}[{t['index']}]:{t['value']:.4f}(|.|={t['abs']:.4f})" for t in topk]
        )
        print(
            f"[ActivationStats:Net] forward={stats['forward_id']} layers={stats['layers_seen']} "
            f"neurons={stats['total_neurons']} active(|a|>{stats['threshold']})={stats['active_neurons']}({stats['active_ratio']:.2%}) "
            f"mean={stats['mean']:.6f} std={stats['std']:.6f} min={stats['min']:.6f} max={stats['max']:.6f} "
            f"hook_triggers={stats.get('hook_triggers')} "
            f"top{len(topk)}=[{topk_str}]"
        )

    def get_last_neuron_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get last computed neuron stats per layer (latest hook result for each layer)."""
        return self.last_neuron_stats

    def get_last_network_neuron_stats(self) -> Dict[str, Any]:
        """Get last computed whole-network neuron stats (latest forward)."""
        return self.last_network_neuron_stats
    
    def mutate(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply random mutation strategy and project to InputSpec.
        
        Args:
            input_tensor: Seed input
        
        Returns:
            Mutated input satisfying InputSpec constraints
        """
        # Select strategy
        strategy_names = list(self.weights.keys())
        strategy_probs = list(self.weights.values())
        strategy_name = np.random.choice(strategy_names, p=strategy_probs)
        strategy = self.strategies[strategy_name]
        
        # NEW: Store strategy for tracing
        self.last_strategy = strategy_name
        
        # Apply mutation
        input_device = input_tensor.to(self.device)
        mutated = strategy.mutate(
            input_device,
            self.model,
            self.last_activations
        )
        
        # Project to InputSpec constraints
        mutated = self._project(mutated)
        
        self.total_mutations += 1
        return mutated
    
    def _project(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Project tensor to satisfy InputSpec constraints.
        
        Supports:
        - BOX: Clip to [lb, ub]
        - LINF_BALL: Clamp to L∞ ball around center
        - LIN_POLY: (TODO) Project to linear polytope
        
        Note: InputSpec bounds should always match tensor shape (enforced by spec creators).
        """
        if self.input_spec is None:
            return tensor
        
        if self.input_spec.kind == InKind.BOX:
            # Box constraints: clip to bounds
            lb = self.input_spec.lb.to(tensor.device)
            ub = self.input_spec.ub.to(tensor.device)
            
            # Verify shape consistency (should be guaranteed by spec creators)
            assert lb.shape == tensor.shape, (
                f"Shape mismatch in BOX projection: "
                f"input_spec.lb.shape={lb.shape} != tensor.shape={tensor.shape}. "
                f"This indicates a bug in the spec creator - bounds should be reshaped during spec creation."
            )
            assert ub.shape == tensor.shape, (
                f"Shape mismatch in BOX projection: "
                f"input_spec.ub.shape={ub.shape} != tensor.shape={tensor.shape}. "
                f"This indicates a bug in the spec creator - bounds should be reshaped during spec creation."
            )
            
            return torch.clamp(tensor, lb, ub)
        
        elif self.input_spec.kind == InKind.LINF_BALL:
            # L∞ ball: clamp perturbation to epsilon
            center = self.input_spec.center.to(tensor.device)
            eps = self.input_spec.eps
            
            # Verify shape consistency (center has batch dimension matching tensor)
            assert center.shape == tensor.shape, (
                f"Shape mismatch in LINF_BALL projection: "
                f"input_spec.center.shape={center.shape} != tensor.shape={tensor.shape}. "
                f"This indicates a bug in the spec creator - center should have batch dimension."
            )
            
            delta = tensor - center
            delta = torch.clamp(delta, -eps, eps)
            return center + delta
        
        elif self.input_spec.kind == InKind.LIN_POLY:
            # Linear polytope: Ax <= b
            # TODO: Implement quadratic programming projection
            # For now, just return the tensor
            return tensor
        
        return tensor
    
    def get_last_activations(self) -> Dict[str, torch.Tensor]:
        """Get activations from last inference."""
        # Note: We don't reset hook_trigger_count here to allow debugging across iterations
        return self.last_activations
    
    def reset_hook_trigger_count(self):
        """Reset the hook trigger counter (for debugging)."""
        self._hook_trigger_count = 0
    
    def debug_activations(self):
        """Print debug information about last_activations."""
        print(f"[MutationEngine] Hook trigger count since last check: {self._hook_trigger_count}")
        if not self.last_activations:
            print("[MutationEngine] WARNING: last_activations is EMPTY!")
            print("  This means no hooks have been triggered yet, or hooks were not registered properly.")
        else:
            print(f"[MutationEngine] last_activations contains {len(self.last_activations)} layers:")
            for name, act in list(self.last_activations.items())[:5]:  # Show first 5
                print(f"  - {name}: shape {act.shape}, mean={act.mean().item():.6f}")
    
    def get_last_gradients(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get gradients from last mutation (Level 3 tracing only)."""
        return self.last_gradients
    
    def get_last_loss(self) -> Optional[float]:
        """Get loss value from last mutation (Level 3 tracing only)."""
        return self.last_loss
    
    def get_stats(self) -> Dict:
        """Get mutation statistics."""
        # Extract perturb_size info
        perturb_size_info = {}
        for strategy_name, strategy in self.strategies.items():
            perturb_size = strategy.perturb_size
            if isinstance(perturb_size, torch.Tensor):
                perturb_size_info[strategy_name] = {
                    "type": "tensor",
                    "shape": list(perturb_size.shape),
                    "min": perturb_size.min().item(),
                    "max": perturb_size.max().item(),
                    "mean": perturb_size.mean().item()
                }
            else:
                perturb_size_info[strategy_name] = {
                    "type": "scalar",
                    "value": perturb_size
                }
        
        return {
            "total_mutations": self.total_mutations,
            "strategy_weights": self.weights,
            "perturb_mode": self.perturb_mode,
            "perturb_scale": self.perturb_scale,
            "perturb_size_values": perturb_size_info,
        }
