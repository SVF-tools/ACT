# ===- act/back_end/bab/branching/branching.py - Branching ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Branching strategies for Branch-and-Bound.
#
#   A branching strategy decides *which dimension (or neuron) to split* for each
#   subproblem in a batch. Two paradigms are supported:
#     1. Input splitting  — bisect (or N-ary section) the input domain along a dim.
#     2. Neuron splitting  — fix an unstable ReLU's activation phase (on/off).
#
#   Joint multi-neuron splitting (2^k sign combinations of each lane's top-k
#   BaBSR neurons, FM 2026) lives in ``act.back_end.bab.branching.multi_split``.
#
#   Strategies (subclasses of ``BranchingStrategy``):
#     * ``RandomBranching`` — uniform-random over eligible dims (width-weighted for
#       input splits; masked to unstable neurons for neuron splits). Baseline.
#     * ``BaBSRBranching`` — BaBSR score ``|bias_term + intercept_term|`` over candidate
#       neurons from the dual ν/bounds (relaxation intercept ``(-l)·u/(u-l)`` weighted by
#       ν, with the pre-activation-bias term). Intercept-only backup
#       ``(-l)·u/(u-l)·clamp(-ν,0)``; width-based fallback when ν/bounds are absent.
#       "Branch and Bound for Piecewise Linear Neural Network Verification",
#       Bunel et al., JMLR 2020.
#     * ``FSBBranching`` — Filtered Smart Branching; extends BaBSR by re-scoring the
#       top candidates with the dual solver and picking the best bound improvement.
#       "Improved Branch and Bound for Neural Network Verification via Lagrangian
#       Decomposition", De Palma et al., 2021.
#
#   Result types: ``BranchingScores`` (per-dim / per-neuron scores) and
#   ``SplitDecision`` (``input_axis`` + ``fanout`` for input splits;
#   ``layer_id`` / ``neuron_idx`` for neuron splits). Factory:
#   ``_build_branching_strategy``.
#
#   All tensor shapes follow the (N, D) batch convention for batch-parallel scoring.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from act.config.config import DualConfig
from act.back_end.bab.node import (
    SubproblemBatch,
    _layer_neuron_count,
    _state_spec_row_count,
    split_input,
    split_input_nary,
    split_neurons,
)
from act.back_end.core import Bounds, Layer, Net
from act.back_end.solver.solver_base import BoundResult
from act.back_end.dual_tf.tf_forward import compute_forward_bounds
from act.back_end.layer_schema import LayerKind
from act.util.device_manager import get_default_device, get_default_dtype


# ---------------------------------------------------------------------------
# Branching result types
# ---------------------------------------------------------------------------


@dataclass
class BranchingScores:
    flat: Optional[torch.Tensor] = None
    per_layer: Optional[Dict[int, torch.Tensor]] = None
    intercept_per_layer: Optional[Dict[int, torch.Tensor]] = None
    input_fallback: Optional[torch.Tensor] = None
    """Width-masked input-axis scores kept alongside neuron scores, so a lane
    with no splittable neuron falls back to a *wide* input dimension instead of
    a hard-coded dim 0 (zero width on any sparse-perturbation benchmark)."""
    babsr_per_layer: Optional[Dict[int, torch.Tensor]] = None
    """Original candidates retained when FSB replaces scores with measured gains."""


@dataclass
class SplitDecision:
    kind: str
    input_axis: Optional[torch.Tensor | int] = None
    fanout: int = 2
    layer_id: Optional[torch.Tensor] = None
    neuron_idx: Optional[torch.Tensor] = None
    lane_mask: Optional[torch.Tensor] = None
    """Scored lanes covered by a neuron-only decision; tensors contain kept rows."""

    def input_axes(self, batch: SubproblemBatch) -> torch.Tensor:
        if self.input_axis is None:
            raise ValueError("input-axis decision missing input_axis")
        input_axis = torch.as_tensor(
            self.input_axis, device=batch.lb.device, dtype=torch.long
        ).reshape(-1)
        if input_axis.numel() == 1:
            input_axis = input_axis.expand(batch.batch_size)
        if input_axis.numel() != batch.batch_size:
            raise ValueError(
                f"input-axis decision has {input_axis.numel()} lanes for batch size "
                f"{batch.batch_size}"
            )
        return input_axis.contiguous()

    def apply(
        self,
        batch: SubproblemBatch,
        net: Net,
    ) -> tuple[SubproblemBatch, torch.Tensor]:
        fanout = max(2, int(self.fanout))
        if self.kind == "input_axis":
            dims = self.input_axes(batch)
            if fanout == 2:
                return split_input(batch, dims)
            return split_input_nary(batch, dims, fanout)

        if self.kind == "neuron":
            if self.layer_id is None or self.neuron_idx is None:
                raise ValueError("neuron decision missing layer_id or neuron_idx")

            layer_id_tensor = self.layer_id.reshape(-1)
            neuron_idx_tensor = self.neuron_idx.reshape(-1)
            if layer_id_tensor.numel() == 0 or neuron_idx_tensor.numel() == 0:
                raise ValueError("neuron decision tensors must be non-empty")

            representative_layer_id = int(layer_id_tensor[0].item())
            if representative_layer_id < 0:
                raise ValueError(
                    f"neuron decision has negative layer_id {representative_layer_id}; "
                    "input-axis splits must be emitted as "
                    "SplitDecision(kind='input_axis')"
                )

            n_lanes = batch.batch_size
            layer_ids = (
                layer_id_tensor.expand(n_lanes)
                if layer_id_tensor.numel() == 1
                else layer_id_tensor
            )
            neuron_indices = (
                neuron_idx_tensor.expand(n_lanes)
                if neuron_idx_tensor.numel() == 1
                else neuron_idx_tensor
            )
            if layer_ids.numel() != n_lanes or neuron_indices.numel() != n_lanes:
                raise ValueError(
                    f"neuron decision has {layer_ids.numel()}/{neuron_indices.numel()} "
                    f"entries for batch size {n_lanes}"
                )

            return split_neurons(
                batch,
                net,
                layer_ids.unsqueeze(1),
                neuron_indices.unsqueeze(1),
                1,
                first_sign=+1.0,
            )

        raise ValueError(f"Unknown SplitDecision.kind: {self.kind!r}")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BranchingStrategy(ABC):
    """Abstract branching strategy for Branch-and-Bound.

    Lifecycle (called by the BaB engine per iteration)::

        scores   = strategy.compute_scores(batch, net, unstable_mask)
        decision = strategy.select(scores)
        children, parent_index = decision.apply(batch, net)

    Subclass contract
    ~~~~~~~~~~~~~~~~~
    * ``compute_scores`` **must** return ``(N, D)`` float tensor.
    * Dimensions that must not be split should receive score ``-inf``
      (or ``0`` when ``select`` uses ``argmax``).
    * ``select`` defaults to row-wise ``argmax``; override for
      stochastic or top-k selection.
    """

    @abstractmethod
    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | BranchingScores:
        """Score every candidate split dimension.

        Args:
            batch:          Current subproblems ``(N, D)``.
            net:            ACT network (for layer structure / neuron info).
            unstable_mask:  ``(D,)`` or ``(N, D)`` bool tensor marking
                            neurons eligible for splitting.
                            ``None`` ⇒ all dimensions are candidates
                            (input-split mode).

        Returns:
            ``(N, D)`` float tensor — higher score = better split.
        """
        ...

    def select(self, scores: torch.Tensor | BranchingScores) -> torch.Tensor | SplitDecision:
        """Pick one split dimension per subproblem.

        Default implementation: deterministic ``argmax`` per row.
        Override for stochastic or multi-split selection.

        Args:
            scores: ``(N, D)`` branching scores.

        Returns:
            ``(N,)`` long tensor of selected dimension indices.
        """
        if isinstance(scores, BranchingScores):
            if scores.flat is None:
                raise ValueError("Base BranchingStrategy requires flat scores")
            scores = scores.flat
        return _argmax_splittable(scores)


def _argmax_splittable(scores: torch.Tensor) -> torch.Tensor:
    """Row-wise ``argmax`` that refuses a masked-out (``-inf``) winner.

    Input-axis scorers mask zero-width dimensions to ``-inf``. A lane whose best
    score is still ``-inf`` has nothing to split: bisecting a zero-width
    dimension yields two children identical to the parent, so the frontier keeps
    re-expanding them and BaB livelocks. Failing loudly is the sound option —
    silently emitting no children would let ``bab.py`` drain the pool and report
    CERTIFIED.
    """
    best = scores.argmax(dim=-1)
    best_score = scores.gather(-1, best.unsqueeze(-1)).squeeze(-1)
    if not bool(torch.isfinite(best_score).all().item()):
        raise ValueError(
            "no splittable input dimension: every candidate dimension is "
            "zero-width or masked out in at least one lane"
        )
    return best


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------


class RandomBranching(BranchingStrategy):
    """Uniform-random branching over eligible dimensions.

    Supports both paradigms:

    * **Input split** (``unstable_mask is None``):
      Random scores weighted by domain width — wider dimensions are more
      likely to be chosen, and zero-width dimensions are excluded.

    * **Neuron split** (``unstable_mask`` provided):
      Uniform-random scores masked to unstable neurons.  Stable neurons
      receive score 0 and are never selected.
    """

    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N, D = batch.batch_size, batch.input_dim
        device = batch.lb.device

        if unstable_mask is not None:
            # Neuron-split mode: zero-out stable neurons
            scores = torch.rand(N, D, device=device)
            mask = unstable_mask.float()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(N, -1)  # (D,) → (N, D)
            scores = scores * mask
        else:
            embedding_mask = _perturbed_embedding_input_mask(net, batch)
            widths = batch.widths()  # (N, D)
            if embedding_mask is None:
                scores = torch.rand(N, D, device=device)
            else:
                mask = embedding_mask.unsqueeze(0).expand(N, -1)
                scores = widths.masked_fill(~mask, float("-inf"))
            scores = scores.masked_fill(widths <= 0, float("-inf"))

        return scores


class InputBranching(BranchingStrategy):
    """Deterministic widest-dimension input branching.

    Scores each input dimension by its current domain width, so ``select``
    bisects the widest side of the box - the classic input-split rule for
    low-dimensional domains (e.g. ACAS Xu), where halving the dominant side
    maximizes the worst-case per-child bound tightening.
    """

    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        widths = batch.widths()
        scores = widths.masked_fill(widths <= 0, float("-inf"))
        embedding_mask = _perturbed_embedding_input_mask(net, batch)
        if embedding_mask is not None:
            mask = embedding_mask.unsqueeze(0).expand_as(widths)
            return scores.masked_fill(~mask, float("-inf"))
        return scores


# ---------------------------------------------------------------------------
# Score-based branching (width-weighted, optional slope-gradient scoring)
# ---------------------------------------------------------------------------


class BaBSRBranching(BranchingStrategy):
    """BaBSR neuron-split scoring (Bunel et al., 1909.06588).

    For each ambiguous ReLU (``l<0<u``) with backward coefficient ν and pre-activation
    bias ``b``, scores the estimated bound improvement of splitting it as
    ``|bias_term + intercept_term|`` where ``s = u/(u-l)``,
    ``bias_term = max(b·ν·(s-1), b·ν·s)`` and ``intercept_term = clamp(ν, max=0)·(-l)·u/(u-l)``.
    An intercept-only score (``(-l)·u/(u-l)·clamp(-ν, 0)``) is kept as the backup.
    When ν / intermediate bounds are unavailable it falls back to width-based scores
    compatible with ``RandomBranching``.
    """

    def __init__(
        self,
        decision_threshold: float = 1e-3,
        intercept_fallback_max: int = 2,
        sparsest_layer: Optional[int] = None,
        neuron_only: bool = False,
    ) -> None:
        self.decision_threshold = decision_threshold
        self.intercept_fallback_max = intercept_fallback_max
        self.sparsest_layer = sparsest_layer
        self.neuron_only = neuron_only
        self.icp_score_counter = 0

    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
        *,
        bounds_dict: Optional[Dict[int, Bounds]] = None,
        nu_per_layer: Optional[Dict[int, torch.Tensor]] = None,
    ) -> BranchingScores:
        if bounds_dict is None or nu_per_layer is None:
            return BranchingScores(flat=self._baseline_scores(batch, unstable_mask, net))

        per_layer: Dict[int, torch.Tensor] = {}
        intercept_per_layer: Dict[int, torch.Tensor] = {}
        for lid, bounds in bounds_dict.items():
            if self.neuron_only and _layer_kind_upper(net.by_id[lid]) != LayerKind.RELU.value:
                continue
            if self.sparsest_layer is not None and lid != self.sparsest_layer:
                continue
            if lid not in nu_per_layer:
                continue
            lb = bounds.lb.flatten(start_dim=1)
            ub = bounds.ub.flatten(start_dim=1)
            N, n_neurons = lb.shape
            ambiguous = (lb < 0) & (ub > 0)
            nu = nu_per_layer[lid]
            if nu.dim() > 2:
                nu = nu.flatten(start_dim=1)
            if nu.shape[0] == N:
                nu_view = nu.reshape(N, 1, n_neurons)
            elif nu.shape[0] % N == 0:
                nu_view = nu.reshape(N, nu.shape[0] // N, n_neurons)
            else:
                continue

            split_mask = self._already_split_mask(batch, lid, n_neurons)
            effective = ambiguous & ~split_mask
            denom = torch.clamp(ub - lb, min=torch.finfo(lb.dtype).eps)
            slope = (ub / denom).unsqueeze(1)
            relax_intercept = ((-lb) * ub / denom).unsqueeze(1)
            preact_bias = _preact_bias_of(net, lid)
            if preact_bias.numel() != n_neurons:
                preact_bias = torch.zeros(n_neurons, dtype=lb.dtype, device=lb.device)
            preact_bias = preact_bias.reshape(1, 1, -1).to(device=nu_view.device, dtype=nu_view.dtype)
            bias_term = torch.maximum(
                preact_bias * nu_view * (slope - 1.0),
                preact_bias * nu_view * slope,
            )
            intercept_term = nu_view.clamp(max=0.0) * relax_intercept
            primary = (bias_term + intercept_term).abs().mean(dim=1)
            primary = primary.masked_fill(~effective, float("-inf"))
            intercept = ((-lb) * ub / denom) * nu_view.mean(dim=1).neg().clamp(min=0.0)
            intercept = intercept.masked_fill(~effective, float("-inf"))
            per_layer[lid] = primary
            intercept_per_layer[lid] = intercept

        if not per_layer:
            return BranchingScores(flat=self._baseline_scores(batch, unstable_mask, net))
        return BranchingScores(
            flat=None,
            per_layer=per_layer,
            intercept_per_layer=intercept_per_layer,
            input_fallback=self._baseline_scores(batch, None, net),
        )

    def _baseline_scores(
        self,
        batch: SubproblemBatch,
        unstable_mask: Optional[torch.Tensor] = None,
        net: Optional[Net] = None,
    ) -> torch.Tensor:
        widths = batch.ub - batch.lb

        embedding_mask = _perturbed_embedding_input_mask(net=net, batch=batch)
        if embedding_mask is not None:
            scores = widths.masked_fill(~embedding_mask.unsqueeze(0), float("-inf"))
        elif batch.incremental_alpha is None:
            scores = widths * torch.rand_like(widths)
        else:
            scores = widths

        if unstable_mask is not None:
            mask = unstable_mask.float()
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(batch.batch_size, -1)
            scores = scores * mask

        # Masked last: -inf * 0 from the unstable mask would be NaN.
        return scores.masked_fill(widths <= 0, float("-inf"))

    def _already_split_mask(
        self,
        batch: SubproblemBatch,
        lid: int,
        n_neurons: int,
    ) -> torch.Tensor:
        if batch.split_signs is None or lid not in batch.split_signs:
            return torch.zeros(
                (batch.batch_size, n_neurons),
                device=batch.lb.device,
                dtype=torch.bool,
            )
        signs = batch.split_signs[lid]
        return (signs != 0).any(dim=1)

    @staticmethod
    def _input_axis_fallback(scores: BranchingScores) -> SplitDecision:
        if scores.input_fallback is None:
            raise ValueError(
                "no splittable input dimension: neuron branching fell back to an "
                "input-axis split but no width scores were recorded"
            )
        return SplitDecision(
            kind="input_axis", input_axis=_argmax_splittable(scores.input_fallback)
        )

    def select(self, scores: torch.Tensor | BranchingScores) -> torch.Tensor | SplitDecision:
        if self.neuron_only:
            return self.select_neuron_only(
                BranchingScores(flat=scores) if isinstance(scores, torch.Tensor) else scores
            )
        if isinstance(scores, torch.Tensor):
            return _argmax_splittable(scores)
        if scores.flat is not None:
            return SplitDecision(kind="input_axis", input_axis=_argmax_splittable(scores.flat))
        per_layer = scores.per_layer
        if not per_layer:
            return self._input_axis_fallback(scores)

        N = next(iter(per_layer.values())).shape[0]
        device = next(iter(per_layer.values())).device
        decisions_layer = torch.zeros(N, dtype=torch.long, device=device)
        decisions_neuron = torch.zeros(N, dtype=torch.long, device=device)

        for n in range(N):
            best_lid: Optional[int] = None
            best_idx = 0
            best_val = float("-inf")
            for lid, score in per_layer.items():
                val, idx = score[n].max(dim=0)
                if float(val.item()) > best_val:
                    best_val = float(val.item())
                    best_lid = lid
                    best_idx = int(idx.item())

            if best_lid is not None and best_val > self.decision_threshold:
                decisions_layer[n] = best_lid
                decisions_neuron[n] = best_idx
                self.icp_score_counter = 0
                continue

            intercept = scores.intercept_per_layer
            if intercept is not None and self.icp_score_counter < self.intercept_fallback_max:
                ic_lid: Optional[int] = None
                ic_idx = 0
                ic_val = float("-inf")
                for lid, score in intercept.items():
                    val, idx = score[n].max(dim=0)
                    if float(val.item()) > ic_val:
                        ic_val = float(val.item())
                        ic_lid = lid
                        ic_idx = int(idx.item())
                if ic_lid is not None and ic_val > float("-inf"):
                    decisions_layer[n] = ic_lid
                    decisions_neuron[n] = ic_idx
                    self.icp_score_counter += 1
                    continue

            self.icp_score_counter = 0
            return self._input_axis_fallback(scores)

        return SplitDecision(kind="neuron", layer_id=decisions_layer, neuron_idx=decisions_neuron)


    def select_neuron_only(self, scores: BranchingScores) -> SplitDecision:
        """Select finite ReLU candidates and flag candidate-less lanes for retirement."""
        per_layer = scores.per_layer or {}
        intercept = scores.intercept_per_layer or {}
        fallback = scores.babsr_per_layer or per_layer
        sample = next(iter(per_layer.values()), None)
        if sample is None:
            sample = next(iter(fallback.values()), None)
        if sample is None:
            sample = next(iter(intercept.values()), None)
        if sample is None:
            sample = scores.flat if scores.flat is not None else scores.input_fallback
        if sample is None:
            raise ValueError("neuron-only scores must identify the scored batch")

        n_lanes = sample.shape[0]
        keep = torch.zeros(n_lanes, dtype=torch.bool, device=sample.device)
        layers = torch.zeros(n_lanes, dtype=torch.long, device=sample.device)
        neurons = torch.zeros_like(layers)
        for lane in range(n_lanes):
            lid, neuron, value = _lane_argmax(per_layer, lane)
            if lid is not None and value > self.decision_threshold:
                self.icp_score_counter = 0
            else:
                lid = None
                if intercept and self.icp_score_counter < self.intercept_fallback_max:
                    lid, neuron, _ = _lane_argmax(intercept, lane)
                    if lid is not None:
                        self.icp_score_counter += 1
                if lid is None:
                    self.icp_score_counter = 0
                    for table in (fallback, intercept):
                        lid, neuron, _ = _lane_argmax(table, lane)
                        if lid is not None:
                            break
            if lid is not None:
                layers[lane] = lid
                neurons[lane] = neuron
                keep[lane] = True

        rows = torch.where(keep)[0]
        return SplitDecision(
            kind="neuron",
            layer_id=layers.index_select(0, rows),
            neuron_idx=neurons.index_select(0, rows),
            lane_mask=keep,
        )


def _lane_argmax(
    per_layer: Dict[int, torch.Tensor], lane: int
) -> Tuple[Optional[int], int, float]:
    """Return the best finite candidate, including zero and negative scores."""
    best_lid: Optional[int] = None
    best_idx = 0
    best_value = float("-inf")
    for lid, score in per_layer.items():
        row = score[lane]
        if row.numel() == 0:
            continue
        value, index = row.masked_fill(~torch.isfinite(row), float("-inf")).max(dim=0)
        if float(value.item()) > best_value:
            best_lid = lid
            best_idx = int(index.item())
            best_value = float(value.item())
    return best_lid, best_idx, best_value


_WITNESS_GAP_TOL = 1e-6


def witness_relu_preactivations(
    net: Net,
    witness_input: torch.Tensor,
    input_shape: tuple[int, ...],
    dual_config: DualConfig,
) -> Optional[Dict[int, torch.Tensor]]:
    """Return concrete per-ReLU pre-activations at a dual witness."""
    if witness_input.numel() == 0:
        return None
    x = witness_input
    if input_shape and x.dim() == 2 and x.shape[1] == int(math.prod(input_shape)):
        x = x.reshape(x.shape[0], *input_shape)
    bounds_dict = None
    try:
        bounds_dict, _, _ = compute_forward_bounds(
            net,
            x,
            x,
            forward_lin_max_perturbed=dual_config.forward_lin_max_perturbed,
        )
    except ValueError:
        # compute_forward_bounds raises ValueError for layer kinds it cannot
        # propagate; those nets use BaBSR scoring.
        bounds_dict = None
    if bounds_dict is None:
        return None
    preactivations = {
        layer_id: bounds.lb.flatten(start_dim=1)
        for layer_id, bounds in bounds_dict.items()
        if _layer_kind_upper(net.by_id[layer_id]) == LayerKind.RELU.value
    }
    return preactivations or None


def _layer_kind_upper(layer: Layer) -> str:
    kind = layer.kind
    return kind.value if isinstance(kind, LayerKind) else kind.upper()


class WitnessResidualBranching(BaBSRBranching):
    """Branch on the realized ReLU relaxation gap at the dual witness.

    For an unstable ReLU with pre-activation interval ``[l, u]`` and witness
    pre-activation ``z``, the triangle upper relaxation is
    ``u / (u - l) * (z - l)``.  The score is the realized residual magnitude
    ``Relaxed(z) - ReLU(z)`` masked to currently unstable, unsplit neurons.
    Missing witness data falls back to BaBSR so the strategy remains usable on
    paths whose solver does not expose an SCE.
    """

    def __init__(
        self, decision_threshold: float = 1e-12, neuron_only: bool = False
    ) -> None:
        super().__init__(decision_threshold=decision_threshold, neuron_only=neuron_only)
        self.fallback_count = 0
        self.different_from_babsr_count = 0

    track_babsr_disagreement: bool = False
    """Opt-in: compare each decision against BaBSR and count divergences.

    Off by default because it costs a second full BaBSR scoring pass on every
    branching step and produces a number the verifier never consumes. Set it on
    the class when studying how far this heuristic drifts from BaBSR.
    """

    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
        *,
        bounds_dict: Optional[Dict[int, Bounds]] = None,
        nu_per_layer: Optional[Dict[int, torch.Tensor]] = None,
        witness_preact_per_layer: Optional[Dict[int, torch.Tensor]] = None,
    ) -> BranchingScores:
        if bounds_dict is None or witness_preact_per_layer is None:
            self.fallback_count += int(batch.batch_size)
            return super().compute_scores(
                batch,
                net,
                unstable_mask,
                bounds_dict=bounds_dict,
                nu_per_layer=nu_per_layer,
            )

        per_layer: Dict[int, torch.Tensor] = {}
        for lid, bounds in bounds_dict.items():
            z_raw = witness_preact_per_layer.get(lid)
            if z_raw is None:
                continue
            lb = bounds.lb.flatten(start_dim=1)
            ub = bounds.ub.flatten(start_dim=1)
            z = z_raw.flatten(start_dim=1).to(device=lb.device, dtype=lb.dtype)
            n_neurons = min(lb.shape[-1], ub.shape[-1], z.shape[-1])
            if n_neurons <= 0 or lb.shape[0] != batch.batch_size or z.shape[0] != batch.batch_size:
                continue
            lb_n = lb[:, :n_neurons]
            ub_n = ub[:, :n_neurons]
            z_n = z[:, :n_neurons]
            ambiguous = (lb_n < 0) & (ub_n > 0)
            split_mask = self._already_split_mask(batch, lid, n_neurons)
            effective = ambiguous & ~split_mask
            denom = torch.clamp(ub_n - lb_n, min=torch.finfo(lb_n.dtype).eps)
            relaxed = ub_n / denom * (z_n - lb_n)
            relu = z_n.clamp(min=0.0)
            # On z in [l, u] the triangle relaxation dominates ReLU, so the gap
            # is non-negative. A negative gap means the witness pre-activation
            # escaped this layer's own bounds (stale bounds, or a witness from a
            # different subproblem). clamp(min=0) neutralises it; abs() would
            # instead turn the violation into a HIGH score and steer branching
            # straight at the corrupted neuron.
            gap = relaxed - relu
            assert not bool(
                (gap[effective] < -_WITNESS_GAP_TOL).any().item()
            ), f"witness pre-activation outside layer {lid} bounds"
            score = gap.clamp(min=0.0).masked_fill(~effective, float("-inf"))
            per_layer[lid] = score

        if not per_layer or not any(bool(torch.isfinite(score).any().item()) for score in per_layer.values()):
            self.fallback_count += int(batch.batch_size)
            return super().compute_scores(
                batch,
                net,
                unstable_mask,
                bounds_dict=bounds_dict,
                nu_per_layer=nu_per_layer,
            )

        if self.track_babsr_disagreement:
            self._record_babsr_disagreement(
                batch, net, unstable_mask, bounds_dict, nu_per_layer, per_layer
            )
        return BranchingScores(
            flat=None,
            per_layer=per_layer,
            intercept_per_layer=None,
            input_fallback=self._baseline_scores(batch, None, net),
        )

    def _record_babsr_disagreement(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor],
        bounds_dict: Dict[int, Bounds],
        nu_per_layer: Optional[Dict[int, torch.Tensor]],
        witness_scores: Dict[int, torch.Tensor],
    ) -> None:
        if nu_per_layer is None:
            return
        babsr = super().compute_scores(
            batch,
            net,
            unstable_mask,
            bounds_dict=bounds_dict,
            nu_per_layer=nu_per_layer,
        )
        if babsr.per_layer is None:
            return
        wr_choice, wr_valid = self._best_pairs(witness_scores)
        babsr_choice, babsr_valid = self._best_pairs(babsr.per_layer)
        if wr_choice is None or babsr_choice is None:
            return
        both_valid = wr_valid & babsr_valid
        differ = (wr_choice != babsr_choice).any(dim=1) & both_valid
        self.different_from_babsr_count += int(differ.sum().item())

    def _best_pairs(
        self, scores: Dict[int, torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Argmax over the concatenated (layer, neuron) axis, one row per lane.

        Returns ``([K, 2]`` tensor of ``(layer_id, neuron_idx)``, ``[K]`` validity
        mask). Concatenating first keeps this to a single ``argmax`` and one host
        sync, instead of a ``.item()`` per lane per layer.
        """
        if not scores:
            return None, torch.zeros(0, dtype=torch.bool)
        layer_ids = sorted(scores)
        flat = torch.cat([scores[lid] for lid in layer_ids], dim=1)
        widths = [scores[lid].shape[1] for lid in layer_ids]
        offsets = torch.tensor(
            [0, *widths[:-1]], device=flat.device, dtype=torch.long
        ).cumsum(0)
        owner = torch.repeat_interleave(
            torch.tensor(layer_ids, device=flat.device, dtype=torch.long),
            torch.tensor(widths, device=flat.device, dtype=torch.long),
        )
        start = torch.repeat_interleave(
            offsets, torch.tensor(widths, device=flat.device, dtype=torch.long)
        )
        best_val, best_col = flat.max(dim=1)
        pairs = torch.stack([owner[best_col], best_col - start[best_col]], dim=1)
        return pairs, torch.isfinite(best_val)


def _assert_witness_residual_decision_unstable(
    decision: SplitDecision,
    bounds_dict: Optional[Dict[int, Bounds]],
) -> None:
    """Check that every non-fallback witness decision splits an unstable ReLU."""
    if decision.kind != "neuron":
        return
    layer_ids = decision.layer_id
    neuron_indices = decision.neuron_idx
    violation: Optional[str] = None
    if layer_ids is None or neuron_indices is None:
        violation = (
            f"neuron decision missing tensors (has_layers={layer_ids is not None}, "
            f"has_neurons={neuron_indices is not None})"
        )
    elif bounds_dict is None:
        violation = (
            f"missing bounds (lanes={layer_ids.numel()}, "
            f"decisions={neuron_indices.numel()})"
        )
    else:
        layer_ids = layer_ids.reshape(-1)
        neuron_indices = neuron_indices.reshape(-1)
        for lane, (layer_raw, neuron_raw) in enumerate(
            zip(layer_ids.tolist(), neuron_indices.tolist())
        ):
            layer_id = int(layer_raw)
            neuron = int(neuron_raw)
            bounds = bounds_dict.get(layer_id)
            if bounds is None:
                violation = (
                    f"decision has no layer bounds (lane={lane}, layer={layer_id}, "
                    f"neuron={neuron}, available_layers={len(bounds_dict)})"
                )
                break
            lb = bounds.lb.flatten(start_dim=1)
            ub = bounds.ub.flatten(start_dim=1)
            if lane >= lb.shape[0] or neuron >= lb.shape[1] or neuron < 0:
                violation = (
                    f"decision index out of range (lane={lane}, layer={layer_id}, "
                    f"neuron={neuron}, bound_lanes={lb.shape[0]}, "
                    f"bound_neurons={lb.shape[1]})"
                )
                break
            lower = float(lb[lane, neuron].item())
            upper = float(ub[lane, neuron].item())
            if not lower < 0.0 < upper:
                violation = (
                    f"non-fallback decision selected stable neuron (lane={lane}, "
                    f"layer={layer_id}, neuron={neuron}, lb={lower:g}, ub={upper:g})"
                )
                break
    assert violation is None, f"WITNESS RESIDUAL invariant violated: {violation}"


def _preact_bias_of(net: Net, lid: int) -> torch.Tensor:
    layer = net.by_id[lid]
    n_neurons = len(layer.out_vars)
    preds = net.preds.get(lid, [])
    if preds:
        pred = net.by_id[preds[0]]
        bias = pred.params.get("bias")
        if isinstance(bias, torch.Tensor):
            return bias.detach().clone()
        for value in pred.params.values():
            if isinstance(value, torch.Tensor):
                return torch.zeros(n_neurons, dtype=value.dtype, device=value.device)
    dtype = get_default_dtype()
    device = get_default_device()
    for value in layer.params.values():
        if isinstance(value, torch.Tensor):
            dtype = value.dtype
            device = value.device
            break
    return torch.zeros(n_neurons, dtype=dtype, device=device)


def _perturbed_embedding_input_mask(net: Optional[Net], batch: SubproblemBatch) -> Optional[torch.Tensor]:
    """Expand one root sample's canonical token mask over its coordinates."""
    if net is None or net.input_geometry is None:
        return None
    geometry = net.input_geometry
    if geometry.mask.shape[0] != 1:
        raise ValueError("BaB branching requires one root sample; slice the net before branching")
    flat = geometry.mask.unsqueeze(-1).expand_as(geometry.center).reshape(-1)
    if flat.numel() != batch.input_dim:
        raise ValueError(
            f"BaB input dimension {batch.input_dim} differs from geometry width {flat.numel()}"
        )
    return flat.to(device=batch.lb.device)


class FSBBranching(BaBSRBranching):
    def __init__(
        self,
        dual_solver: Any,
        branching_candidates: int = 3,
        decision_threshold: float = 1e-3,
        intercept_fallback_max: int = 2,
        sparsest_layer: Optional[int] = None,
        neuron_only: bool = False,
    ) -> None:
        super().__init__(
            decision_threshold=decision_threshold,
            intercept_fallback_max=intercept_fallback_max,
            sparsest_layer=sparsest_layer,
            neuron_only=neuron_only,
        )
        self.dual_solver = dual_solver
        self.branching_candidates = branching_candidates

    def compute_scores(
        self,
        batch: SubproblemBatch,
        net: Net,
        unstable_mask: Optional[torch.Tensor] = None,
        *,
        bounds_dict: Optional[Dict[int, Bounds]] = None,
        nu_per_layer: Optional[Dict[int, torch.Tensor]] = None,
    ) -> BranchingScores:
        bsr = super().compute_scores(
            batch,
            net,
            unstable_mask,
            bounds_dict=bounds_dict,
            nu_per_layer=nu_per_layer,
        )
        if bsr.per_layer is None or not bsr.per_layer:
            return bsr

        hypothesis_list: List[Dict[int, torch.Tensor]] = []
        candidate_metadata: List[Tuple[int, torch.Tensor]] = []
        candidate_validity: List[torch.Tensor] = []
        topk = self.branching_candidates
        for lid, score in bsr.per_layer.items():
            if self.neuron_only:
                score = score.masked_fill(~torch.isfinite(score), float("-inf"))
            k = min(topk, score.shape[-1])
            if k > 0:
                top_values, top_idx = score.topk(k, dim=-1)
                for idx in range(top_idx.shape[-1]):
                    if not bool(torch.isfinite(top_values[:, idx]).any().item()):
                        continue
                    neuron_per_lane = top_idx[:, idx]
                    valid = torch.isfinite(top_values[:, idx])
                    candidate_validity.append(valid)
                    hypothesis_list.append(
                        self._clone_split_signs_with_hypothesis(
                            batch, lid,
                            neuron_per_lane.masked_fill(~valid, -1) if self.neuron_only else neuron_per_lane,
                            net,
                        )
                    )
                    candidate_metadata.append((lid, neuron_per_lane))
        if bsr.intercept_per_layer is not None:
            for lid, score in bsr.intercept_per_layer.items():
                if self.neuron_only:
                    score = score.masked_fill(~torch.isfinite(score), float("-inf"))
                k = min(topk, score.shape[-1])
                if k > 0:
                    top_values, top_idx = score.topk(k, dim=-1)
                    for idx in range(top_idx.shape[-1]):
                        if not bool(torch.isfinite(top_values[:, idx]).any().item()):
                            continue
                        neuron_per_lane = top_idx[:, idx]
                        valid = torch.isfinite(top_values[:, idx])
                        candidate_validity.append(valid)
                        hypothesis_list.append(
                            self._clone_split_signs_with_hypothesis(
                                batch, lid,
                                neuron_per_lane.masked_fill(~valid, -1) if self.neuron_only else neuron_per_lane,
                                net,
                            )
                        )
                        candidate_metadata.append((lid, neuron_per_lane))

        if not hypothesis_list:
            return bsr

        N = batch.batch_size
        baseline = batch.parent_margins
        result = self._evaluate_hypotheses(net, bounds_dict, hypothesis_list)
        stacked_margins = result.row_lower_bounds().to(
            device=batch.lb.device,
            dtype=batch.lb.dtype,
        )
        if stacked_margins.dim() == 1:
            stacked_margins = stacked_margins.reshape(1, -1)
        if stacked_margins.dim() > 2:
            stacked_margins = stacked_margins.reshape(stacked_margins.shape[0], N, -1).mean(dim=-1)
        if stacked_margins.shape[0] != len(hypothesis_list):
            raise ValueError("KFSB solver returned wrong candidate dimension")
        if stacked_margins.shape[1] != N:
            stacked_margins = stacked_margins.reshape(len(hypothesis_list), N, -1).mean(dim=-1)
        improvements = (
            stacked_margins - baseline.to(stacked_margins.device).unsqueeze(0)
            if baseline is not None
            else stacked_margins
        )
        if self.neuron_only:
            improvements = improvements.masked_fill(
                ~torch.stack(candidate_validity) | ~torch.isfinite(improvements),
                float("-inf"),
            )

        if bool(torch.isneginf(improvements).all().item()):
            return bsr

        best_cand_idx = improvements.argmax(dim=0)
        final_per_layer: Dict[int, torch.Tensor] = {}
        for lane in range(N):
            ci = int(best_cand_idx[lane].item())
            lid, neuron_tensor = candidate_metadata[ci]
            neuron_idx = int(neuron_tensor[lane].item())
            source = bsr.per_layer.get(lid) if bsr.per_layer is not None else None
            if source is None and bsr.intercept_per_layer is not None:
                source = bsr.intercept_per_layer.get(lid)
            if source is None:
                continue
            if lid not in final_per_layer:
                final_per_layer[lid] = torch.full_like(source, float("-inf"))
            final_per_layer[lid][lane, neuron_idx] = improvements[ci, lane]

        return BranchingScores(
            flat=None,
            per_layer=final_per_layer,
            intercept_per_layer=bsr.intercept_per_layer,
            input_fallback=bsr.input_fallback,
            babsr_per_layer=bsr.per_layer,
        )

    def _evaluate_hypotheses(
        self,
        net: Net,
        bounds_dict: Optional[Dict[int, Bounds]],
        split_signs: List[Dict[int, torch.Tensor]],
    ) -> BoundResult:
        if bounds_dict is None:
            raise ValueError("FSB hypothesis evaluation requires bounds_dict")
        c = self._zero_objective(net, bounds_dict)
        return self.dual_solver.compute_certified_bound(
            net,
            bounds_dict,
            c,
            M=1,
            split_signs=split_signs,
            optimize=False,
        )


    def _zero_objective(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
    ) -> torch.Tensor:
        assert_layers = [
            layer for layer in net.layers if layer.kind == LayerKind.ASSERT.value
        ]
        if assert_layers:
            assert_layer = assert_layers[-1]
            preds = net.preds.get(assert_layer.id, [])
            n_out = len(net.by_id[preds[0]].out_vars) if preds else 1
        else:
            n_out = 1
        sample = next(iter(bounds_dict.values()))
        B = sample.lb.shape[0]
        return torch.zeros((B, n_out), dtype=sample.lb.dtype, device=sample.lb.device)

    def _clone_split_signs_with_hypothesis(
        self,
        batch: SubproblemBatch,
        lid: int,
        neuron_idx_per_lane: torch.Tensor,
        net: Net,
    ) -> Dict[int, torch.Tensor]:
        # This is a one-sided +1 scoring hypothesis, not a partition into children.
        hypo: Dict[int, torch.Tensor] = {}
        if batch.split_signs is not None:
            for key, value in batch.split_signs.items():
                hypo[key] = value.clone()

        n_neurons = _layer_neuron_count(net.by_id[lid])
        n_specs = _state_spec_row_count(batch)

        if lid not in hypo:
            hypo[lid] = torch.zeros(
                (batch.batch_size, n_specs, n_neurons),
                device=batch.lb.device,
                dtype=batch.lb.dtype,
            )
        for lane in range(batch.batch_size):
            idx = int(neuron_idx_per_lane[lane].item())
            if idx < 0:
                continue
            hypo[lid][lane, :, idx] = +1.0
        return hypo


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def _build_branching_strategy(
    method: str,
    *,
    dual_solver: Any = None,
    branching_candidates: int = 3,
    neuron_only: bool = False,
) -> BranchingStrategy:
    if method == "random":
        return RandomBranching()
    if method == "width":
        return InputBranching()
    if method == "babsr":
        return BaBSRBranching(neuron_only=neuron_only)
    if method == "witness_residual":
        return WitnessResidualBranching(neuron_only=neuron_only)
    if method == "fsb":
        if dual_solver is None:
            raise ValueError("FSB branching requires a dual_solver instance (inject via factory).")
        return FSBBranching(
            dual_solver=dual_solver,
            branching_candidates=branching_candidates,
            neuron_only=neuron_only,
        )
    raise ValueError(f"Unknown branching method: {method!r}")
