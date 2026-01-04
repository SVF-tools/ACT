"""
Mutation strategies for ACTFuzzer.

Implements gradient-guided, activation-guided, boundary, and random mutations.
All mutations automatically respect InputSpec constraints via projection.

Gradient-guided now accommodates two mutated input generation methods: FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent).
    1) FGSM: single-step gradient-based perturbation.
    2) PGD: iterative gradient-based perturbation.

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
from typing import Dict, Optional, Union
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


class FGSMMutation(MutationStrategy):
    """
    FGSM-style gradient-guided mutation (single-step).

    Computes gradients to maximize output variance, then applies a single-step
    sign-gradient perturbation.
    """

    def __init__(self, perturb_size: Union[float, torch.Tensor] = 8/255):
        """
        Initialize FGSM mutation.

        Args:
            perturb_size: Mutation perturbation magnitude (scalar or per-dimension tensor)
        """
        self.perturb_size = perturb_size

    def mutate(self, input_tensor, model, activations=None):
        """Apply FGSM gradient-based perturbation (single-step)."""
        # Enable gradients
        x = input_tensor.clone().detach().requires_grad_(True)

        # Forward pass
        output = model(x)

        # Extract output tensor if dict (from VerifiableModel)
        if isinstance(output, dict):
            output = output['output']

        # Compute loss: maximize output variance (unsupervised, label-free)
        loss = output.var()

        # Get gradient w.r.t. input only (avoid accumulating grads on model params)
        grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0].detach()

        # FGSM: sign of gradient
        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        perturbation = perturb_size * torch.sign(grad)

        # Apply perturbation
        return input_tensor + perturbation


class PGDMutation(MutationStrategy):
    """
    PGD-style gradient-guided mutation (iterative).

    Implementation follows the notebook approach:
    - Define a feasible box around x0: [x0 - perturb_size, x0 + perturb_size]
    - Optional random start within the feasible box
    - Iterative sign-gradient ascent with projection back to the feasible box

    Note: Global InputSpec constraints are enforced by MutationEngine projection after mutation.
    """

    def __init__(
        self,
        perturb_size: Union[float, torch.Tensor] = 8/255,
        num_steps: int = 10,
        step_size: Optional[float] = None,
        random_start: bool = True,
    ):
        """
        Initialize PGD mutation.

        Args:
            perturb_size: L_infinity radius of local feasible box around the seed (scalar or per-dimension tensor)
            num_steps: Number of PGD iterations
            step_size: Per-iteration step size (if None, computed from feasible box range / steps as in notebook)
            random_start: Whether to start uniformly within the feasible box (recommended)
        """
        self.perturb_size = perturb_size
        self.num_steps = int(num_steps)
        self.step_size = step_size
        self.random_start = random_start

    def mutate(self, input_tensor, model, activations=None):
        """Apply PGD mutation with notebook-style projection and step size heuristic."""
        x0 = input_tensor.detach()

        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        x_low = x0 - perturb_size
        x_high = x0 + perturb_size

        # Default step size: spread movement across the available range (notebook heuristic)
        if self.step_size is None:
            # (x_high - x_low) == 2*perturb_size; take max range element as scalar step size
            step_size = float((x_high - x_low).abs().max().item()) / max(self.num_steps, 1)
            step_size = max(step_size, 1e-6)
        else:
            step_size = float(self.step_size)

        # Random start inside feasible box
        if self.random_start:
            x_adv = x_low + torch.rand_like(x0) * (x_high - x_low)
        else:
            x_adv = x0.clone()

        # Ensure start in-bounds
        x_adv = torch.max(torch.min(x_adv, x_high), x_low).detach()

        for _ in range(self.num_steps):
            x_adv.requires_grad_(True)

            # Forward pass
            output = model(x_adv)

            # Extract output tensor if dict (from VerifiableModel)
            if isinstance(output, dict):
                output = output['output']

            # Loss: maximize output variance (unsupervised, label-free)
            loss = output.var()

            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0].detach()

            # Gradient ascent on loss
            x_adv = (x_adv + step_size * torch.sign(grad)).detach()

            # Project back to feasible box
            x_adv = torch.max(torch.min(x_adv, x_high), x_low).detach()

        return x_adv.detach()




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
            "gradient": FGSMMutation(perturb_size=perturb_size),
            "pgd": PGDMutation(perturb_size=perturb_size),
            "activation": ActivationMutation(perturb_size=perturb_size),
            "boundary": BoundaryMutation(perturb_size=perturb_size * 0.5),  # Half perturb_size for boundary (more conservative)
            "random": RandomMutation(perturb_size=perturb_size * 0.5)       # Half perturb_size for random (more conservative)
        }

        # Validate and normalize weights
        unknown = set(weights.keys()) - set(self.strategies.keys())
        if unknown:
            raise ValueError(
                f"Unknown mutation strategy keys in weights: {sorted(unknown)}. "
                f"Valid options: {sorted(self.strategies.keys())}"
            )
        total = sum(float(v) for v in weights.values())
        if total <= 0.0:
            raise ValueError(f"Mutation weights must sum to > 0. Got total={total}.")
        self.weights = {k: float(v) / total for k, v in weights.items()}
        
        # Statistics
        self.total_mutations = 0
        self.last_activations: Dict[str, torch.Tensor] = {}
        self.last_strategy: Optional[str] = None  # NEW: track last mutation strategy
        self.last_gradients: Optional[Dict[str, torch.Tensor]] = None  # NEW: for Level 3 tracing
        self.last_loss: Optional[float] = None  # NEW: for Level 3 tracing
        
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

        
        def make_hook(name):
            def hook(module, input, output):
                # Store activation (handle both tensor and dict outputs)
                if isinstance(output, torch.Tensor):
                    self.last_activations[name] = output.detach()
                elif isinstance(output, dict) and 'output' in output:
                    self.last_activations[name] = output['output'].detach()
            
            return hook
        
        # Register hooks on computational layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Linear, nn.Conv2d)):
                module.register_forward_hook(make_hook(name))
                
    
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
        return self.last_activations
    
    
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
