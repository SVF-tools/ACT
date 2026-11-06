"""
Mutation strategies for ACTFuzzer.

Implements gradient-guided, activation-guided, boundary, and random mutations.
All mutations automatically respect InputSpec constraints via projection.

Gradient-guided now accommodates two mutated input generation methods: FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent).
    1) FGSM: single-step gradient-based perturbation.
    2) PGD: iterative gradient-based perturbation.

## Batch Tensor-Based Mutation

All mutation strategies operate on batched inputs [B, C, H, W] for GPU parallelism.
The MutationEngine selects a single strategy per batch and applies it to all seeds
simultaneously, enabling efficient gradient computation (FGSM/PGD) across the batch.
The batch size is aligned with model synthesis (N VNNLib instances), so InputSpec bounds
are already [N, ...] and match the batch dimension directly. After mutation, projection
ensures each sample respects its InputSpec bounds (BOX per-sample bounds via original_index,
or LINF_BALL eps-ball around each seed's original_tensor).

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

3. **fixed** (conventional):
   - Computes perturb_size from mean range: perturb_size = mean(ub - lb)
   - Note: Uses full feasible range as perturbation size 

### Configuration

Set in `act/config/pipeline.yaml`:
```yaml
perturb_mode: "adaptive_scalar"  # Options: "adaptive_scalar", "adaptive_perdim", "fixed"
perturb_scale: 0.1               # Fraction of range per step (default: 0.1 = 10 steps)
```

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING, cast
import torch
import torch.nn as nn
import numpy as np

from act.front_end.specs import InputSpec, InKind, OutKind, OutputSpec
from act.front_end.verifiable_model import VerifiableModel
from act.pipeline.fuzzing.checker import INPUT_FEASIBILITY_ATTR
from act.util.device_manager import get_default_device

if TYPE_CHECKING:
    from act.pipeline.fuzzing.corpus import FuzzingSeed


LOGIT_GUIDED_KINDS = frozenset({OutKind.TOP1_ROBUST})

PERTURB_MODES = ("adaptive_scalar", "adaptive_perdim", "fixed")
MUTATION_STRATEGIES = ("gradient", "pgd", "activation", "boundary", "random")

SIGN_STE_TIGHT_EPS = 0.1
SIGN_STE_LOOSE_EPS = 5.0

# A distance in pre-activation units, not a numerical tolerance: it says how
# near the flip boundary a neuron must sit to be worth steering, which depends
# on the network's activation scale and not on the working float precision.
BOUNDARY_MARGIN_THRESHOLD = 1e-4
BOUNDARY_MARGIN_SCALER = 10.0


class _SignSTEFunction(torch.autograd.Function):
    """``torch.sign`` forward with a tunable-eps straight-through backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.eps = eps
        return torch.sign(x)

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        (x,) = ctx.saved_tensors
        (grad_output,) = grad_outputs
        return grad_output.masked_fill(x.abs() >= ctx.eps, 0.0) / ctx.eps, None


class SignSTE(nn.Module):
    """Differentiable stand-in for a binarized network's ``sign`` activation.

    ``torch.sign`` has zero gradient everywhere it is defined, so a binarized
    network hands back an input gradient of exactly zero and every
    gradient-guided strategy degenerates into a random walk.

    The forward pass is bit-exact ``torch.sign``. That is what keeps this
    module honest: the fuzzer's own inference — and therefore every
    counterexample ``PropertyChecker`` accepts — runs the unmodified network,
    so installing this module cannot manufacture a false positive. Only the
    backward pass is a straight-through estimate, using the tunable-eps rule
    ``grad[|x| >= eps] = 0; grad /= eps``.

    ``eps`` trades faithfulness against reach. A tight ``eps`` only propagates
    through pre-activations that are genuinely close to flipping, which is
    the accurate local model; a loose ``eps`` keeps signal alive on networks
    whose pre-activations are far from the sign boundary.
    """

    def __init__(self, eps: float = SIGN_STE_LOOSE_EPS):
        """
        Initialize the estimator.

        Args:
            eps: Backward-pass gate width; gradient is dropped where
                ``|x| >= eps`` and rescaled by ``1 / eps``.
        """
        super().__init__()
        self.eps = float(eps)
        self.pre_activation: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``torch.sign(x)`` with a straight-through gradient path."""
        self.pre_activation = x
        return _SignSTEFunction.apply(x, self.eps)


def is_sign_module(module: nn.Module) -> bool:
    """Whether ``module`` is an ONNX-imported wrapper around ``torch.sign``.

    Located by the wrapped callable rather than by module path, so any importer
    that exposes the op as a ``.function`` attribute is recognised. The name is
    matched exactly: ``softsign`` also contains "sign" but is a different,
    already-differentiable op, and replacing it would corrupt the forward pass.

    Args:
        module: Candidate module from the attacked network.

    Returns:
        True when the module simply applies ``torch.sign``.
    """
    function = getattr(module, "function", None)
    return callable(function) and getattr(function, "__name__", "") == "sign"


class MutationStrategy(ABC):
    """Base class for mutation strategies."""
    
    @abstractmethod
    def mutate(self, 
               input_tensor: torch.Tensor,
               model: nn.Module,
               activations: Optional[Dict[str, torch.Tensor]] = None,
               rows: Optional[torch.Tensor] = None
              ) -> torch.Tensor:
        """
        Apply mutation to input tensor.
        
        Args:
            input_tensor: Seed input
            model: Inner network for gradient computation
            activations: Activations from previous inference (optional)
            rows: [B] int64 — spec row backing each batch lane (optional)
        
        Returns:
            Mutated input tensor
        """
        pass


class GradientMutation(MutationStrategy, ABC):
    """
    Base class for the gradient-guided strategies (FGSM, PGD).

    Both ascend the same objective, and this is the only place it is defined:
    the violation severity of the model's ``OutputSpec``. Severity is signed so
    that ``severity > 0`` means the property is broken, which makes it a direct
    gradient-ascent target for every ``OutKind``. For ``TOP1_ROBUST`` it is
    exactly the Carlini-Wagner margin ``max_{j != t} z_j - z_t`` that PGD used
    to compute inline; routing through ``OutputSpec.severity`` generalises that
    loss to the other four kinds instead of falling back to a label-free proxy.

    When ``logit_sink`` is supplied the objective is scored on the network's
    pre-softmax logits instead of its output. A saturating softmax flattens the
    severity landscape to a constant, which starves gradient ascent even though
    the property itself is still attackable; the logits it consumes are not
    saturated. Only ``TOP1_ROBUST`` qualifies: its severity is a pure ranking
    comparison. A gap-valued severity such as ``MARGIN_ROBUST`` does not,
    because softmax preserves order but not gaps, so the logit severity and the
    true severity cross zero at different points. This only ever changes the
    ATTACK OBJECTIVE — the violation decision stays with ``PropertyChecker`` on
    the model's true output.
    """

    def __init__(
        self,
        output_spec: OutputSpec,
        perturb_size: Union[float, torch.Tensor],
        logit_sink: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Initialize a gradient-guided mutation.

        Args:
            output_spec: Property to attack. Supplies the ascent objective.
            perturb_size: Mutation perturbation magnitude (scalar or per-dimension tensor)
            logit_sink: Dict a forward hook fills with the trailing Softmax's
                input under key ``"z"``, or None to score the model's output.
        """
        self.output_spec = output_spec
        self.perturb_size = perturb_size
        self.logit_sink = logit_sink
        self.sign_stes: List[SignSTE] = []

    def _attack_output(self, x: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Forward ``x`` and return the tensor the objective scores."""
        if self.logit_sink is None:
            return model(x)
        self.logit_sink.pop("z", None)
        output = model(x)
        return self.logit_sink.pop("z", output)

    def _severity(
        self,
        x: torch.Tensor,
        model: nn.Module,
        rows: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Per-lane violation severity of ``x`` under the attack objective."""
        return self.output_spec.severity(self._attack_output(x, model), rows=rows)

    def _severity_sum(
        self,
        x: torch.Tensor,
        model: nn.Module,
        rows: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward ``x`` and return the batch's total violation severity.

        The returned scalar stays inside the autograd graph, so
        ``torch.autograd.grad(loss, x)`` yields the direction that drives the
        batch toward violating the property. Lanes are summed because they are
        independent: ``d severity_i / d x_j`` is zero for ``i != j``, so the sum
        gives each lane its own gradient.

        Args:
            x: Input tensor [B, ...] that requires grad.
            model: Inner network; returns logits [B, n_out].
            rows: [B] int64 mapping each lane onto its spec row, or None for
                identity. The corpus samples with replacement, so lane ``i`` is
                generally NOT spec row ``i``.

        Returns:
            Scalar tensor to ascend.
        """
        severity = self._severity(x, model, rows).sum()
        return severity + self._boundary_proximity()

    def _boundary_proximity(self) -> Union[torch.Tensor, float]:
        """Reward for parking sign pre-activations near their flip boundary.

        A binarized layer only responds to an input perturbation when one of its
        pre-activations crosses zero, so the neurons worth attacking are the
        ones already sitting close to it. Rewarding proximity keeps ascent
        supplied with neurons that a further step can actually flip, which is
        also where the straight-through estimator's gradient is non-zero.

        The first estimator is excluded because it reads the input pixels
        directly, so its proximity is a statement about the perturbation rather
        than about the network's internal state.

        Returns:
            Scalar tensor to add to the ascent objective, or ``0.0`` when fewer
            than two estimators are installed.
        """
        if len(self.sign_stes) < 2:
            return 0.0
        per_layer: List[torch.Tensor] = []
        for ste in self.sign_stes[1:]:
            pre_activation = ste.pre_activation
            if pre_activation is None:
                continue
            margin = (BOUNDARY_MARGIN_THRESHOLD - pre_activation.abs()).clamp(min=0)
            per_layer.append(margin.reshape(margin.shape[0], -1).mean(dim=1).sum())
        if not per_layer:
            return 0.0
        stacked = torch.stack(per_layer).mean()
        return stacked / (BOUNDARY_MARGIN_SCALER * BOUNDARY_MARGIN_THRESHOLD)

    def input_gradient(
        self,
        x: torch.Tensor,
        model: nn.Module,
        rows: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Gradient of the batch's total severity w.r.t. ``x``.

        Args:
            x: Input tensor [B, ...]; a fresh leaf is taken from it.
            model: Inner network to differentiate through.
            rows: [B] int64 mapping each lane onto its spec row, or None.

        Returns:
            Detached gradient tensor shaped like ``x``.
        """
        leaf = x.detach().requires_grad_(True)
        loss = self._severity_sum(leaf, model, rows)
        return torch.autograd.grad(
            loss, leaf, retain_graph=False, create_graph=False
        )[0].detach()


class FGSMMutation(GradientMutation):
    """
    FGSM-style gradient-guided mutation (single-step).

    Takes one sign-gradient step along the violation-severity gradient.
    """

    def __init__(
        self,
        output_spec: OutputSpec,
        perturb_size: Union[float, torch.Tensor] = 8/255,
        logit_sink: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Initialize FGSM mutation.

        Args:
            output_spec: Property to attack (supplies the ascent objective)
            perturb_size: Mutation perturbation magnitude (scalar or per-dimension tensor)
            logit_sink: Pre-softmax logit capture (see :class:`GradientMutation`)
        """
        super().__init__(output_spec, perturb_size, logit_sink=logit_sink)

    def mutate(self, input_tensor, model, activations=None, rows=None):
        """Apply FGSM gradient-based perturbation (single-step).
        
        Args:
            input_tensor: Seed input tensor
            model: Inner network for gradient computation
            activations: Activations from previous inference (unused)
            rows: [B] int64 — spec row backing each batch lane
        """
        grad = self.input_gradient(input_tensor, model, rows)

        # FGSM: sign of gradient
        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        perturbation = perturb_size * torch.sign(grad)

        # Apply perturbation
        return input_tensor + perturbation


class PGDMutation(GradientMutation):
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
        output_spec: OutputSpec,
        perturb_size: Union[float, torch.Tensor] = 8/255,
        num_steps: int = 50,
        step_size: Optional[float] = None,
        random_start: bool = True,
        restarts: int = 1,
        restarts_binarized: int = 1,
        logit_sink: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Initialize PGD mutation.

        Args:
            output_spec: Property to attack (supplies the ascent objective)
            perturb_size: L_infinity radius of local feasible box around the seed (scalar or per-dimension tensor)
            num_steps: Number of PGD iterations
            step_size: Per-iteration step size (if None, computed from feasible box range / steps as in notebook)
            random_start: Whether to start uniformly within the feasible box (recommended)
            restarts: Independent random starts per mutation; the best lane-wise
                result is kept. 1 reproduces a single-start attack exactly.
            restarts_binarized: Restart count used instead of ``restarts`` once
                sign estimators are installed, so the larger budget a binarized
                network needs costs ordinary networks nothing.
            logit_sink: Pre-softmax logit capture (see :class:`GradientMutation`)
        """
        super().__init__(output_spec, perturb_size, logit_sink=logit_sink)
        self.num_steps = int(num_steps)
        self.step_size = step_size
        self.random_start = random_start
        self.restarts = max(int(restarts), 1)
        self.restarts_binarized = max(int(restarts_binarized), 1)

    def _apply_ste_eps(self, restart: int) -> None:
        """Point the installed sign STEs at this restart's gate width."""
        if not self.sign_stes:
            return
        # A faithfully tight gate is dead when pre-activations sit far from zero.
        eps = SIGN_STE_LOOSE_EPS if restart % 2 == 0 else SIGN_STE_TIGHT_EPS
        for ste in self.sign_stes:
            ste.eps = eps

    def _ascend(
        self,
        x0: torch.Tensor,
        x_low: torch.Tensor,
        x_high: torch.Tensor,
        step_size: float,
        model: nn.Module,
        rows: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run one restart: random start, then projected sign-gradient ascent."""
        # Random start inside feasible box
        if self.random_start:
            x_adv = x_low + torch.rand_like(x0) * (x_high - x_low)
        else:
            x_adv = x0.clone()

        # Ensure start in-bounds
        x_adv = torch.max(torch.min(x_adv, x_high), x_low).detach()

        for _ in range(self.num_steps):
            grad = self.input_gradient(x_adv, model, rows)

            # Gradient ascent on loss
            x_adv = (x_adv + step_size * torch.sign(grad)).detach()

            # Project back to feasible box
            x_adv = torch.max(torch.min(x_adv, x_high), x_low).detach()

        return x_adv.detach()

    def mutate(self, input_tensor, model, activations=None, rows=None):
        """Apply PGD mutation.
        
        Runs :attr:`restarts` independent random starts and keeps, per lane,
        whichever reached the highest violation severity. Restarts stop early
        once every lane already violates, so the extra budget is only spent
        while it can still buy something.
        
        Args:
            input_tensor: Seed input tensor [B, C, H, W] or [1, C, H, W]
            model: Inner network for gradient computation
            activations: Activations from previous inference (unused by PGD)
            rows: [B] int64 — spec row backing each batch lane
        
        Returns:
            Adversarially perturbed input tensor [B, C, H, W]
        """
        x0 = input_tensor.detach()

        perturb_size = self.perturb_size.to(input_tensor.device) if isinstance(self.perturb_size, torch.Tensor) else self.perturb_size
        x_low = x0 - perturb_size
        x_high = x0 + perturb_size

        # Default step size: spread movement across the available range (notebook heuristic)
        if self.step_size is None:
            # (x_high - x_low) == 2*perturb_size; take max range element as scalar step size
            step_size = float((x_high - x_low).abs().max().item()) / max(self.num_steps, 1)
            step_size = max(step_size, float(torch.finfo(x0.dtype).eps))
        else:
            step_size = float(self.step_size)

        restarts = self.restarts_binarized if self.sign_stes else self.restarts

        self._apply_ste_eps(0)
        best_x = self._ascend(x0, x_low, x_high, step_size, model, rows)
        if restarts == 1:
            return best_x

        with torch.no_grad():
            best_severity = self._severity(best_x, model, rows)

        for restart in range(1, restarts):
            if bool((best_severity > 0).all()):
                break
            self._apply_ste_eps(restart)
            x_adv = self._ascend(x0, x_low, x_high, step_size, model, rows)
            with torch.no_grad():
                severity = self._severity(x_adv, model, rows)
            keep = (severity > best_severity).reshape(
                -1, *([1] * (x_adv.dim() - 1))
            )
            best_x = torch.where(keep, x_adv, best_x)
            best_severity = torch.maximum(severity, best_severity)

        return best_x




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
    
    def mutate(self, input_tensor, model, activations=None, rows=None):
        """Apply activation-guided perturbation.
        
        Args:
            input_tensor: Seed input tensor
            model: Model (unused)
            activations: Activations from previous inference (unused currently)
            rows: Spec rows (unused, kept for interface consistency)
        """
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
    
    def mutate(self, input_tensor, model, activations=None, rows=None):
        """Push toward boundaries (will be projected by engine).
        
        Args:
            input_tensor: Seed input tensor
            model: Model (unused)
            activations: Activations (unused)
            rows: Spec rows (unused, kept for interface consistency)
        """
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
    
    def mutate(self, input_tensor, model, activations=None, rows=None):
        """Apply random Gaussian noise.
        
        Args:
            input_tensor: Seed input tensor
            model: Model (unused)
            activations: Activations (unused)
            rows: Spec rows (unused, kept for interface consistency)
        """
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
    - Single strategy per mutation call for GPU parallelism
    
    Example:
        >>> engine = MutationEngine(model, input_spec, weights)
        >>> mutated = engine.mutate([seed1, seed2])
        >>> activations = engine.get_activation_map()
    """
    
    def __init__(self,
                 model: nn.Module,
                 input_spec: Optional[InputSpec],
                 weights: Dict[str, float],
                 perturb_mode: str = "fixed",
                 perturb_scale: float = 0.1,
                 pgd_restarts: int = 1,
                 pgd_restarts_binarized: int = 1):
        """
        Initialize mutation engine.
        
        Args:
            model: VerifiableModel for gradient computation. Hooks are registered
                across its whole module tree, but the gradient strategies attack
                its inner network (see :meth:`_resolve_attack_target`).
            input_spec: InputSpec for constraint projection (batched bounds from model synthesis)
            weights: Strategy weights (e.g., {"gradient": 0.4, "random": 0.1})
            perturb_mode: Perturbation size computation mode ("adaptive_scalar", "adaptive_perdim", "fixed")
            perturb_scale: Fraction of range per mutation perturbation (e.g., 0.1 = 10% = ~10 steps to traverse)
            pgd_restarts: PGD random starts per mutation.
            pgd_restarts_binarized: PGD random starts per mutation once sign
                estimators are installed.
        
        Raises:
            TypeError: If ``model`` is not a VerifiableModel.
            ValueError: If ``model`` carries no OutputSpec, or weights are invalid.
        """
        self.model = model
        self.attack_model, self.output_spec, self.logit_sink = (
            self._resolve_attack_target(model)
        )
        self.input_spec = input_spec
        self.device = get_default_device()

        self.strategies = {
            "gradient": FGSMMutation(
                self.output_spec,
                perturb_size=0.0,
                logit_sink=self.logit_sink,
            ),
            "pgd": PGDMutation(
                self.output_spec,
                perturb_size=0.0,
                restarts=pgd_restarts,
                restarts_binarized=pgd_restarts_binarized,
                logit_sink=self.logit_sink,
            ),
            "activation": ActivationMutation(perturb_size=0.0),
            "boundary": BoundaryMutation(perturb_size=0.0),
            "random": RandomMutation(perturb_size=0.0),
        }
        self.set_perturbation(perturb_mode, perturb_scale)
        self.set_weights(weights)
        
        # Statistics
        self.total_mutations = 0
        self.activation_map: Dict[str, torch.Tensor] = {}
        self.last_strategy: Optional[str] = None  # Track last mutation strategy for tracing
        self.last_gradients: Optional[Dict[str, torch.Tensor]] = None  # For Level 3 tracing: gradient capture
        self.last_loss: Optional[float] = None  # For Level 3 tracing: loss value capture
        self._pending_rows: Optional[torch.Tensor] = None
        self._feasibility_forward_active = False
        self._gradient_signal_checked = False
        
        # Setup hooks for activation capture
        self._setup_hooks()
        self._setup_feasibility_hooks()
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """Validate and normalize the strategy selection weights."""
        unknown = set(weights.keys()) - set(MUTATION_STRATEGIES)
        if unknown:
            raise ValueError(
                f"Unknown mutation strategy keys in weights: {sorted(unknown)}. "
                f"Valid options: {sorted(MUTATION_STRATEGIES)}"
            )
        total = sum(float(v) for v in weights.values())
        if total <= 0.0:
            raise ValueError(f"Mutation weights must sum to > 0. Got total={total}.")
        self.weights = {k: float(v) / total for k, v in weights.items()}

    def set_perturbation(self, mode: str, scale: float) -> None:
        """Set how perturbation sizes are computed (see FuzzingConfig.perturb_mode)."""
        if mode not in PERTURB_MODES:
            raise ValueError(f"Unknown perturb_mode: {mode}. Valid options: {list(PERTURB_MODES)}")
        self.perturb_mode = mode
        self.perturb_scale = scale
        # Adaptive modes set each strategy's size per seed in mutate(). Fixed
        # mode keeps a range-aware baseline; without an InputSpec, 0.01.
        if mode == "fixed" or self.input_spec is None:
            size = self._compute_fixed_perturb_size()
            for strategy in self.strategies.values():
                strategy.perturb_size = size

    def set_pgd_restarts(self, restarts: int, restarts_binarized: int) -> None:
        """Set PGD random starts per mutation (ordinary and binarized networks)."""
        pgd = cast(PGDMutation, self.strategies["pgd"])
        pgd.restarts = max(int(restarts), 1)
        pgd.restarts_binarized = max(int(restarts_binarized), 1)

    @staticmethod
    def _resolve_attack_target(
        model: nn.Module,
    ) -> tuple[nn.Module, OutputSpec, Optional[Dict[str, torch.Tensor]]]:
        """Split a wrapped model into the network to attack and the property to attack it with.
        
        The gradient strategies attack the INNER network rather than the
        VerifiableModel wrapper. ``InputLayer`` and ``InputSpecLayer`` are tensor
        pass-throughs, so the logits are identical, but every wrapper forward runs
        ``.sum().item()`` in each spec-layer branch — roughly 4 CUDA syncs that PGD
        would otherwise pay ``num_steps`` times per mutation. Coverage hooks are
        unaffected: they are registered across the whole module tree, and the
        inner network is part of it.
        
        A network whose last layer is a Softmax also gets a forward hook that
        captures that Softmax's input, so the gradient strategies can ascend
        on unsaturated logits. The hook is only installed for
        :data:`LOGIT_GUIDED_KINDS`, whose severity depends on the output
        through its RANKING alone, which softmax preserves. Every other
        ``OutKind`` scores output values or gaps, neither of which softmax
        preserves, so logits would silently redefine it. The Softmax must be the LAST
        leaf module: an interior Softmax (transformer attention) is not the
        model's output layer, so its input is not a logit vector.
        
        Args:
            model: VerifiableModel from model synthesis.
        
        Returns:
            Tuple of (inner network returning logits, output property to
            ascend, pre-softmax logit sink or None).
        
        Raises:
            TypeError: If ``model`` is not a VerifiableModel, so no inner network
                and no property can be identified.
            ValueError: If the wrapper carries no OutputSpec; the gradient
                strategies have no objective without one.
        """
        if not isinstance(model, VerifiableModel):
            raise TypeError(
                f"MutationEngine requires a VerifiableModel to source the attack "
                f"objective, got {type(model).__name__}."
            )
        output_spec = model.output_spec.spec
        if output_spec is None:
            raise ValueError(
                "MutationEngine requires an OutputSpec: the gradient strategies "
                "ascend its violation severity."
            )
        inner = model.model
        if output_spec.kind not in LOGIT_GUIDED_KINDS:
            return inner, output_spec, None
        leaves = [m for m in inner.modules() if next(m.children(), None) is None]
        if not leaves or not isinstance(leaves[-1], nn.Softmax):
            return inner, output_spec, None
        logit_sink: Dict[str, torch.Tensor] = {}
        
        def capture_logits(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            logit_sink["z"] = args[0]
        
        leaves[-1].register_forward_pre_hook(capture_logits)
        return inner, output_spec, logit_sink
    
    def _install_sign_stes(self) -> List[SignSTE]:
        """Swap every ``sign`` module in the attacked network for a SignSTE.

        Returns:
            The installed estimators, empty when the network has no sign
            modules.
        """
        stes: List[SignSTE] = []
        for name, module in list(self.attack_model.named_modules()):
            if not is_sign_module(module):
                continue
            parent = self.attack_model
            *path, attribute = name.split(".")
            for step in path:
                parent = getattr(parent, step)
            ste = SignSTE()
            setattr(parent, attribute, ste)
            stes.append(ste)
        return stes

    def _ensure_gradient_signal(
        self,
        strategy: GradientMutation,
        batch_input: torch.Tensor,
        rows: torch.Tensor,
    ) -> None:
        """Install sign STEs once, if the network's input gradient is dead.

        Runs on the first gradient-guided mutation. An input gradient of
        exactly zero means no strategy can steer, which for a binarized network
        is caused by ``torch.sign``; :class:`SignSTE` restores a backward
        path without altering the forward pass. Probed rather than assumed, so
        ordinary networks pay one forward/backward and nothing else.

        Args:
            strategy: Gradient strategy supplying the objective to probe.
            batch_input: Current batch [B, ...] to probe at.
            rows: [B] int64 spec row backing each lane.
        """
        if self._gradient_signal_checked:
            return
        self._gradient_signal_checked = True

        grad = strategy.input_gradient(batch_input, self.attack_model, rows)
        if float(grad.norm().item()) != 0.0:
            return

        stes = self._install_sign_stes()
        if not stes:
            print(
                "[MutationEngine] Input gradient is exactly zero and the network "
                "has no sign modules; gradient strategies cannot steer."
            )
            return
        for candidate in self.strategies.values():
            if isinstance(candidate, GradientMutation):
                candidate.sign_stes = stes

        retry = strategy.input_gradient(batch_input, self.attack_model, rows)
        print(
            f"[MutationEngine] Input gradient was exactly zero: installed a "
            f"sign STE on {len(stes)} module(s) "
            f"(eps={SIGN_STE_LOOSE_EPS}); gradient norm after retry: "
            f"{float(retry.norm().item()):.6g}"
        )

    def _compute_fixed_perturb_size(self) -> float:
        """Return the range-aware perturbation baseline used by fixed mode."""
        if self.input_spec is None:
            print(
                "[MutationEngine] No InputSpec provided, falling back to "
                "fixed perturb_size=0.01"
            )
            return 0.01
        
        # Extract bounds based on InputSpec kind
        # BOX: explicit lb/ub bounds define the feasible region directly
        # LINF_BALL: feasible region is [center - eps, center + eps] (L∞ ball around center)
        if self.input_spec.kind == InKind.BOX:
            # BOX constraints: lb and ub are directly specified
            assert self.input_spec.lb is not None and self.input_spec.ub is not None
            lb = self.input_spec.lb
            ub = self.input_spec.ub
        elif self.input_spec.kind == InKind.LINF_BALL:
            # L∞ ball constraints: range is 2*eps around center point
            # The feasible region is all points x such that ||x - center||_∞ <= eps
            assert self.input_spec.center is not None and self.input_spec.eps is not None
            center = self.input_spec.center
            eps = torch.as_tensor(self.input_spec.eps, device=center.device, dtype=center.dtype)
            lb = center - eps
            ub = center + eps
        elif self.input_spec.kind == InKind.LP_EMBEDDING:
            lb, ub = self.input_spec.materialize_box_seed()
        else:
            print(
                f"[MutationEngine] Unsupported InputSpec kind "
                f"'{self.input_spec.kind}', falling back to fixed "
                "perturb_size=0.01"
            )
            return 0.01

        return (ub - lb).mean().item()
    
    def _setup_hooks(self):
        """Setup forward hooks to capture activations."""
        def make_hook(name):
            def hook(module, input, output):
                # Store activation (handle both tensor and dict outputs)
                if isinstance(output, torch.Tensor):
                    self.activation_map[name] = output.detach()
                elif isinstance(output, dict) and 'output' in output:
                    self.activation_map[name] = output['output'].detach()
            
            return hook
        
        # Register hooks on computational layers (ReLU, Linear, Conv2d)
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Linear, nn.Conv2d)):
                module.register_forward_hook(make_hook(name))

    def _setup_feasibility_hooks(self) -> None:
        """Carry lane rows and input feasibility through the fuzzer inference."""
        def inject_rows(module, args, kwargs):
            rows = self._pending_rows
            if rows is None:
                return None
            self._pending_rows = None
            self._feasibility_forward_active = True
            if "rows" in kwargs:
                raise RuntimeError("Fuzzer inference rows were provided twice")
            kwargs["rows"] = rows
            return args, kwargs

        def attach_feasibility(module, args, kwargs, output):
            if not self._feasibility_forward_active:
                return None
            self._feasibility_forward_active = False
            if not isinstance(output, dict):
                raise RuntimeError(
                    "VerifiableModel.forward must return input feasibility metadata"
                )
            model_output = output.get("output")
            input_satisfied = output.get("input_satisfied_per_sample")
            if not isinstance(model_output, torch.Tensor) or not isinstance(
                input_satisfied, torch.Tensor
            ):
                raise RuntimeError(
                    "VerifiableModel.forward did not return a per-sample input "
                    "feasibility mask"
                )
            setattr(model_output, INPUT_FEASIBILITY_ATTR, input_satisfied)
            return None

        self.model.register_forward_pre_hook(inject_rows, with_kwargs=True)
        self.model.register_forward_hook(attach_feasibility, with_kwargs=True)
                
    def mutate(self, seeds: 'FuzzingSeed') -> torch.Tensor:
        """
        Apply mutation to seeds.
        
        A single strategy is selected for all seeds, enabling GPU parallelism
        for gradient-based strategies (FGSM/PGD).
        
        Args:
            seeds: FuzzingSeed batch with .tensor [B,C,H,W] and .original_index [B] int64
        
        Returns:
            Mutated tensor [B, C, H, W] satisfying InputSpec constraints
        """
        if not seeds:
            raise ValueError("Empty seed batch")
        
        B = len(seeds)
        
        # Use batch tensor directly: [B, C, H, W]
        batch_input = seeds.tensor.to(self.device)
        # Spec row backing each lane. The corpus samples WITH REPLACEMENT, so
        # lane i is not spec row i; identity rows would attack another
        # instance's property (and gather another instance's bounds below).
        rows = seeds.original_index.to(self.device)
        
        # Select strategy (same for all samples)
        strategy_names = list(self.weights.keys())
        strategy_probs = list(self.weights.values())
        strategy_name = np.random.choice(strategy_names, p=strategy_probs)
        strategy = self.strategies[strategy_name]
        
        # Store strategy for tracing
        self.last_strategy = strategy_name
        
        # Dynamic per-seed scale: s_b = 1 - (1 - s₀)^{n_b+1}
        if self.perturb_mode != "fixed" and self.input_spec is not None:
            if self.input_spec.kind == InKind.LINF_BALL:
                assert self.input_spec.eps is not None
                _orig = seeds.original_tensor.to(self.device)
                _eps = torch.as_tensor(self.input_spec.eps, device=_orig.device, dtype=_orig.dtype)
                _lb = _orig - _eps
                _ub = _orig + _eps
            else:
                if self.input_spec.kind == InKind.LP_EMBEDDING:
                    _lb, _ub = self.input_spec.materialize_box_seed()
                    _lb = _lb.to(self.device)
                    _ub = _ub.to(self.device)
                else:
                    assert self.input_spec.lb is not None and self.input_spec.ub is not None
                    _lb = self.input_spec.lb.to(self.device)
                    _ub = self.input_spec.ub.to(self.device)
                if _lb.shape[0] > 1:
                    assert bool((rows < _lb.shape[0]).all()), (
                        f"original_index out of range for InputSpec bounds: "
                        f"max={int(rows.max().item())} >= {_lb.shape[0]} rows"
                    )
                    _lb, _ub = _lb[rows], _ub[rows]
                elif _lb.shape[0] == 1 and B > 1:
                    _lb = _lb.expand(B, *_lb.shape[1:])
                    _ub = _ub.expand(B, *_ub.shape[1:])
                else:
                    _lb, _ub = _lb[:B], _ub[:B]
            n = seeds.select_count.float().to(self.device)
            s_b = 1.0 - (1.0 - self.perturb_scale) ** (n + 1.0)
            reach = torch.max(batch_input - _lb, _ub - batch_input)
            shape = [B] + [1] * (reach.dim() - 1)
            perturb_size = s_b.view(*shape) * reach
            if self.perturb_mode == "adaptive_scalar":
                perturb_size = perturb_size.mean(dim=list(range(1, perturb_size.dim())), keepdim=True)
            strategy.perturb_size = perturb_size
        
        if isinstance(strategy, GradientMutation):
            self._ensure_gradient_signal(strategy, batch_input, rows)
        
        # Apply mutation
        mutated = strategy.mutate(
            batch_input,
            self.attack_model,
            self.activation_map,
            rows=rows
        )
        
        # Project to InputSpec constraints
        mutated = self._project(mutated, seeds)

        # The next wrapped-model inference must check each lane against the
        # InputSpec row from which that seed originated.
        self._pending_rows = rows
        
        self.total_mutations += B
        return mutated
    
    def _project(self, tensor: torch.Tensor, seeds: 'Optional[FuzzingSeed]' = None) -> torch.Tensor:
        """
        Project mutated tensor back into the feasible region defined by InputSpec constraints.
        
        Since fuzzing batch size is aligned with model synthesis (N VNNLib instances),
        the InputSpec bounds (lb/ub) are already [N, ...] and match the batch dimension directly.
        
        **BOX (InKind.BOX)**:
            Clamps to per-sample lb/ub bounds from InputSpec (already batch-aligned).
            Uses seeds.original_index tensor to gather correct per-sample bounds.
        
        **LINF_BALL (InKind.LINF_BALL)**:
            Clamps perturbation delta to [-eps, +eps] around each seed's ORIGINAL input,
            preserving the L∞ distance invariant across mutation chains.
        
        **LIN_POLY (InKind.LIN_POLY)**:
            Not yet implemented — returns tensor unchanged.
        
        Args:
            tensor: Mutated input tensor [B, ...] to project
            seeds: FuzzingSeed batch with original_index [B] and original_tensor [B, ...]
        
        Returns:
            Projected tensor [B, ...] satisfying InputSpec constraints
        """
        if self.input_spec is None:
            return tensor
        
        B = tensor.shape[0]
        
        if self.input_spec.kind in (InKind.BOX, InKind.LP_EMBEDDING):
            if self.input_spec.kind == InKind.LP_EMBEDDING:
                lb, ub = self.input_spec.materialize_box_seed()
                lb = lb.to(tensor.device)
                ub = ub.to(tensor.device)
            else:
                assert self.input_spec.lb is not None and self.input_spec.ub is not None
                lb = self.input_spec.lb.to(tensor.device)
                ub = self.input_spec.ub.to(tensor.device)
            
            # bounds: use seeds.original_index to gather correct bounds
            if lb.shape[0] > 1 and seeds is not None:
                indices = seeds.original_index.to(lb.device)
                assert bool((indices < lb.shape[0]).all()), (
                    f"original_index out of range for InputSpec bounds: "
                    f"max={int(indices.max().item())} >= {lb.shape[0]} rows"
                )
                lb = lb[indices]  # (B, ...) vectorized gather
                ub = ub[indices]
            elif lb.shape[0] == 1 and B > 1:
                lb = lb.expand(B, *lb.shape[1:])
                ub = ub.expand(B, *ub.shape[1:])
            else:
                lb = lb[:B]
                ub = ub[:B]
            
            return torch.clamp(tensor, lb, ub)
        
        elif self.input_spec.kind == InKind.LINF_BALL:
            eps = self.input_spec.eps
            assert eps is not None
            
            assert seeds is not None and len(seeds) == B, \
                f"LINF_BALL projection requires seeds (got {len(seeds) if seeds else 0}, expected {B})"
            
            # Use original_tensor as center to maintain L∞ distance from original
            center = seeds.original_tensor.to(tensor.device)
            eps = torch.as_tensor(eps, device=tensor.device, dtype=tensor.dtype)
            
            delta = tensor - center
            delta = torch.clamp(delta, -eps, eps)
            projected = center + delta

            # Reconstructing a point exactly at ±eps can round one ULP outside
            # the ball. Pull only those coordinates one representable value
            # toward the center so the exact InputSpecLayer check agrees with
            # this projection.
            overshot = (projected - center).abs() > eps
            return torch.where(
                overshot, torch.nextafter(projected, center), projected
            )
        
        elif self.input_spec.kind == InKind.LIN_POLY:
            # TODO: Implement quadratic programming projection
            return tensor
        
        return tensor
    
    def get_activation_map(self) -> Dict[str, torch.Tensor]:
        """Get activations from last inference."""
        return self.activation_map
    
    
    def get_last_gradients(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get gradients from last mutation (Level 3 tracing only)."""
        return self.last_gradients
    
    def get_last_loss(self) -> Optional[float]:
        """Get loss value from last mutation (Level 3 tracing only)."""
        return self.last_loss
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mutation statistics."""
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
