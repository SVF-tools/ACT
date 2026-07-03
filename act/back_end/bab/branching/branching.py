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
#   Joint multi-neuron splitting (``_multi_split_from_decision``): split each lane's
#   top-k BaBSR-scored neurons together into all 2^k sign combinations (one "skip"
#   instead of k greedy single splits); gains are super-additive.
#     * ``MultiNeuronSplitBranching`` — "Mining Verdict Boundaries for Neural Network
#       Verification", Jiawei Ren, Guanqin Zhang, Zhenya Zhang, Yulei Sui, FM 2026.
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
#   ``SplitDecision`` (``input_axis`` / ``cut_dim`` + ``fanout`` for input splits;
#   ``layer_id`` / ``neuron_idx`` for neuron splits). Factory:
#   ``_build_branching_strategy``.
#
#   All tensor shapes follow the (N, D) batch convention for batch-parallel scoring.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from act.back_end.bab.node import SubproblemBatch
from act.back_end.core import Bounds, Net
from act.front_end.specs import InKind


# ---------------------------------------------------------------------------
# Branching result types
# ---------------------------------------------------------------------------


@dataclass
class BranchingScores:
    flat: Optional[torch.Tensor] = None
    per_layer: Optional[Dict[int, torch.Tensor]] = None
    intercept_per_layer: Optional[Dict[int, torch.Tensor]] = None


@dataclass
class SplitDecision:
    kind: str
    input_axis: Optional[torch.Tensor | int] = None
    cut_dim: Optional[torch.Tensor] = None
    fanout: int = 2
    layer_id: Optional[torch.Tensor] = None
    neuron_idx: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BranchingStrategy(ABC):
    """Abstract branching strategy for Branch-and-Bound.

    Lifecycle (called by the BaB engine per iteration)::

        scores     = strategy.compute_scores(batch, net, unstable_mask)
        split_dims = strategy.select(scores)
        left, right = split_subproblems(batch, split_dims)

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
        return scores.argmax(dim=-1)


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
                scores = torch.rand(N, D, device=device) * (widths > 0).float()
            else:
                mask = embedding_mask.unsqueeze(0).expand(N, -1)
                scores = widths.masked_fill(~mask, float("-inf"))

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
        embedding_mask = _perturbed_embedding_input_mask(net, batch)
        if embedding_mask is not None:
            mask = embedding_mask.unsqueeze(0).expand_as(widths)
            return widths.masked_fill(~mask, float("-inf"))
        return widths


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
    ) -> None:
        self.decision_threshold = decision_threshold
        self.intercept_fallback_max = intercept_fallback_max
        self.sparsest_layer = sparsest_layer
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

        return scores

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

    def select(self, scores: torch.Tensor | BranchingScores) -> torch.Tensor | SplitDecision:
        if isinstance(scores, torch.Tensor):
            return scores.argmax(dim=-1)
        if scores.flat is not None:
            return SplitDecision(kind="input_axis", input_axis=scores.flat.argmax(dim=-1))
        per_layer = scores.per_layer
        if not per_layer:
            return SplitDecision(kind="input_axis", input_axis=0)

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
            return SplitDecision(kind="input_axis", input_axis=0)

        return SplitDecision(kind="neuron", layer_id=decisions_layer, neuron_idx=decisions_neuron)


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
    dtype = torch.float32
    device = torch.device("cpu")
    for value in layer.params.values():
        if isinstance(value, torch.Tensor):
            dtype = value.dtype
            device = value.device
            break
    return torch.zeros(n_neurons, dtype=dtype, device=device)


def _perturbed_embedding_input_mask(net: Optional[Net], batch: SubproblemBatch) -> Optional[torch.Tensor]:
    """Flat input mask for LP_EMBEDDING perturbed token coordinates.

    The mask is stored on the batch after first discovery so fallback input
    branching can still reuse it when a strategy helper lacks the net handle.
    """
    cached = getattr(batch, "_perturbed_embedding_mask", None)
    if isinstance(cached, torch.Tensor):
        return cached.to(device=batch.lb.device, dtype=torch.bool)
    if net is None:
        return None
    for layer in net.layers:
        if layer.kind != "INPUT_SPEC" or layer.params.get("kind") != InKind.LP_EMBEDDING:
            continue
        center = layer.params.get("center")
        if not isinstance(center, torch.Tensor) or center.dim() < 2:
            continue
        token_count = int(center.shape[-2])
        embed_dim = int(center.shape[-1])
        from act.front_end.specs import normalize_position_mask

        positions = layer.params.get("perturbed_positions")
        if (
            isinstance(positions, torch.Tensor)
            and positions.dtype == torch.bool
            and positions.numel() != token_count
            and positions.numel() % token_count == 0
        ):
            positions = positions.reshape(-1, token_count)[0]
        token_mask = normalize_position_mask(
            positions, token_count, device=batch.lb.device,
        )
        flat = token_mask.unsqueeze(-1).expand(token_count, embed_dim).reshape(-1)
        if flat.numel() < batch.input_dim:
            pad = torch.zeros(batch.input_dim - flat.numel(), device=batch.lb.device, dtype=torch.bool)
            flat = torch.cat([flat, pad], dim=0)
        elif flat.numel() > batch.input_dim:
            flat = flat[:batch.input_dim]
        setattr(batch, "_perturbed_embedding_mask", flat)
        return flat
    return None


def attention_nu_exposure_candidates(
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Deferred hook for future attention-neuron splitting.

    Attention BaBSR needs a stable exposure of catalytic attention ν and
    pre-activation intervals.  Phase P4 intentionally ships input splitting
    only, so this hook documents the integration point and returns no
    candidates until that exposure contract exists.
    """
    del bounds_dict, nu_per_layer
    return None


class FSBBranching(BaBSRBranching):
    def __init__(
        self,
        dual_solver: Any,
        branching_candidates: int = 3,
        decision_threshold: float = 1e-3,
        intercept_fallback_max: int = 2,
        sparsest_layer: Optional[int] = None,
    ) -> None:
        super().__init__(
            decision_threshold=decision_threshold,
            intercept_fallback_max=intercept_fallback_max,
            sparsest_layer=sparsest_layer,
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
        topk = self.branching_candidates
        for lid, score in bsr.per_layer.items():
            k = min(topk, score.shape[-1])
            if k > 0:
                top_values, top_idx = score.topk(k, dim=-1)
                for idx in range(top_idx.shape[-1]):
                    if not bool(torch.isfinite(top_values[:, idx]).any().item()):
                        continue
                    neuron_per_lane = top_idx[:, idx]
                    hypothesis_list.append(
                        self._clone_split_signs_with_hypothesis(batch, lid, neuron_per_lane, net)
                    )
                    candidate_metadata.append((lid, neuron_per_lane))
        if bsr.intercept_per_layer is not None:
            for lid, score in bsr.intercept_per_layer.items():
                k = min(topk, score.shape[-1])
                if k > 0:
                    top_values, top_idx = score.topk(k, dim=-1)
                    for idx in range(top_idx.shape[-1]):
                        if not bool(torch.isfinite(top_values[:, idx]).any().item()):
                            continue
                        neuron_per_lane = top_idx[:, idx]
                        hypothesis_list.append(
                            self._clone_split_signs_with_hypothesis(batch, lid, neuron_per_lane, net)
                        )
                        candidate_metadata.append((lid, neuron_per_lane))

        if not hypothesis_list:
            return bsr

        N = batch.batch_size
        baseline = batch.parent_margins
        try:
            result = self._evaluate_hypotheses(net, bounds_dict, hypothesis_list)
            stacked_margins = result.margins.to(device=batch.lb.device, dtype=batch.lb.dtype)
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
        except Exception:
            improvements = self._evaluate_hypotheses_serial(
                net,
                bounds_dict,
                hypothesis_list,
                baseline,
                N,
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

        return BranchingScores(flat=None, per_layer=final_per_layer, intercept_per_layer=bsr.intercept_per_layer)

    def _evaluate_hypotheses(
        self,
        net: Net,
        bounds_dict: Optional[Dict[int, Bounds]],
        split_signs: List[Dict[int, torch.Tensor]],
    ) -> Any:
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

    def _evaluate_hypotheses_serial(
        self,
        net: Net,
        bounds_dict: Optional[Dict[int, Bounds]],
        hypothesis_list: List[Dict[int, torch.Tensor]],
        baseline: Optional[torch.Tensor],
        N: int,
    ) -> torch.Tensor:
        improvements = torch.full(
            (len(hypothesis_list), N),
            float("-inf"),
            device=next(iter(hypothesis_list[0].values())).device,
            dtype=next(iter(hypothesis_list[0].values())).dtype,
        )
        for c_idx, hypo in enumerate(hypothesis_list):
            try:
                result = self._evaluate_hypothesis(net, bounds_dict, hypo)
                margins = result.margins.reshape(N, -1).mean(dim=-1).to(improvements)
                improvements[c_idx] = margins - baseline.to(margins.device) if baseline is not None else margins
            except Exception:
                # Recoverable: an individual hypothetical split may be incompatible with a legacy solver/mock.
                continue
        return improvements

    def _evaluate_hypothesis(
        self,
        net: Net,
        bounds_dict: Optional[Dict[int, Bounds]],
        split_signs: Dict[int, torch.Tensor],
    ) -> Any:
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
        assert_layers = [layer for layer in net.layers if layer.kind == "ASSERT"]
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
        hypo: Dict[int, torch.Tensor] = {}
        if batch.split_signs is not None:
            for key, value in batch.split_signs.items():
                hypo[key] = value.clone()

        n_neurons = len(net.by_id[lid].out_vars)
        n_specs = 1
        if batch.incremental_alpha is not None and lid in batch.incremental_alpha:
            n_specs = int(batch.incremental_alpha[lid].shape[1])
        elif batch.incremental_eta is not None and lid in batch.incremental_eta:
            n_specs = int(batch.incremental_eta[lid].shape[1])

        if lid not in hypo:
            hypo[lid] = torch.zeros(
                (batch.batch_size, n_specs, n_neurons),
                device=batch.lb.device,
                dtype=batch.lb.dtype,
            )
        for lane in range(batch.batch_size):
            idx = int(neuron_idx_per_lane[lane].item())
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
) -> BranchingStrategy:
    if method == "random":
        return RandomBranching()
    if method == "width":
        return InputBranching()
    if method == "babsr":
        return BaBSRBranching()
    if method == "fsb":
        if dual_solver is None:
            raise ValueError("FSB branching requires a dual_solver instance (inject via factory).")
        return FSBBranching(dual_solver=dual_solver, branching_candidates=branching_candidates)
    raise ValueError(f"Unknown branching method: {method!r}")


# ---------------------------------------------------------------------------
# Joint multi-neuron splitting (verdict-boundary)
# ---------------------------------------------------------------------------


def _collect_neuron_candidates(
    branch_batch: SubproblemBatch,
    bounds_dict: Dict[int, Bounds],
    nu_per_layer: Dict[int, torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per-lane BaBSR scores (area x |nu|) over all splittable neurons.

    Returns ``(scores, layer_ids, neuron_ids)``, each ``[K, C]`` over the
    concatenated candidate axis; stable or already-split neurons score -inf.
    """
    kb = branch_batch.batch_size
    device = branch_batch.lb.device
    cand_layers: List[torch.Tensor] = []
    cand_neurons: List[torch.Tensor] = []
    cand_scores: List[torch.Tensor] = []
    for lid, nut in nu_per_layer.items():
        b = bounds_dict.get(lid)
        if b is None:
            continue
        lb = b.lb.flatten(start_dim=1)
        ub = b.ub.flatten(start_dim=1)
        if lb.shape[0] != kb:
            continue
        n = min(lb.shape[-1], nut.shape[-1])
        amb = (lb[:, :n] < 0) & (ub[:, :n] > 0)
        already = branch_batch.split_signs.get(lid) if branch_batch.split_signs else None
        if already is not None:
            amb &= already[:, 0, :n].to(device) == 0
        area = (-lb[:, :n] * ub[:, :n] / (ub[:, :n] - lb[:, :n]).clamp(min=1e-12)).clamp(min=0)
        nv = nut.reshape(kb, -1, nut.shape[-1])[:, :, :n].abs().sum(dim=1)
        sc = torch.where(amb, area * nv, torch.full_like(area, float("-inf")))
        cand_scores.append(sc)
        cand_layers.append(torch.full((kb, n), lid, device=device, dtype=torch.long))
        cand_neurons.append(
            torch.arange(n, device=device, dtype=torch.long).expand(kb, n)
        )
    if not cand_scores:
        return None
    return (
        torch.cat(cand_scores, dim=1),
        torch.cat(cand_layers, dim=1),
        torch.cat(cand_neurons, dim=1),
    )


def enumerate_unstable_candidates(
    branch_batch: SubproblemBatch,
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if bounds_dict is None or nu_per_layer is None:
        return []
    cand = _collect_neuron_candidates(branch_batch, bounds_dict, nu_per_layer)
    if cand is None:
        return []
    scores, layers, neurons = cand
    finite_mask = torch.isfinite(scores)
    total = int(finite_mask.sum().item())
    if total == 0 or (limit is not None and total > int(limit)):
        return []
    out: List[Dict[str, Any]] = []
    for lane in range(scores.shape[0]):
        flat_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for col in torch.where(finite_mask[lane])[0].tolist():
            lid = int(layers[lane, col].item())
            nidx = int(neurons[lane, col].item())
            b = bounds_dict.get(lid)
            if b is not None and lid not in flat_cache:
                flat_cache[lid] = (b.lb.flatten(start_dim=1), b.ub.flatten(start_dim=1))
            lb = float(flat_cache[lid][0][lane, nidx].item()) if lid in flat_cache else 0.0
            ub = float(flat_cache[lid][1][lane, nidx].item()) if lid in flat_cache else 0.0
            denom = ub - lb
            area = float(max(0.0, (-lb * ub) / denom)) if denom > 1e-12 else 0.0
            score = float(scores[lane, col].item())
            out.append({
                "lane": lane, "layer_id": lid, "neuron_idx": nidx,
                "score": score, "lb": lb, "ub": ub,
                "nu": (score / area) if area > 1e-12 else None, "area": area,
            })
    return out


def _multi_split_from_groups(
    batch: SubproblemBatch,
    net: Net,
    top_layers: torch.Tensor,
    top_neurons: torch.Tensor,
    k_eff: int,
) -> Tuple[SubproblemBatch, torch.Tensor]:
    # Soundness: the 2^k_eff sign-combination children exactly partition each
    # lane's region for ANY (top_layers, top_neurons) — BaBSR- or LLM-chosen.
    n_lanes = batch.batch_size
    n_children = 2 ** k_eff
    device = batch.lb.device
    parent_index = torch.arange(n_lanes, device=device).repeat(n_children)

    def _gather(state: Optional[Dict[int, torch.Tensor]]) -> Optional[Dict[int, torch.Tensor]]:
        if state is None:
            return None
        return {
            l: t.index_select(0, parent_index.to(t.device)) for l, t in state.items()
        }

    m_specs = 1
    if batch.incremental_alpha:
        m_specs = int(next(iter(batch.incremental_alpha.values())).shape[1])
    elif batch.split_signs:
        m_specs = int(next(iter(batch.split_signs.values())).shape[1])

    signs = _gather(batch.split_signs) or {}
    for lid_val in torch.unique(top_layers).tolist():
        lid_int = int(lid_val)
        layer = net.by_id[lid_int]
        n_neurons = int(layer.out_vars[-1] - layer.out_vars[0] + 1)
        if lid_int not in signs:
            signs[lid_int] = torch.zeros(
                n_children * n_lanes, m_specs, n_neurons,
                device=device, dtype=batch.lb.dtype,
            )
        else:
            signs[lid_int] = signs[lid_int].clone()
        for bit in range(k_eff):
            lane_sel = torch.where(top_layers[:, bit] == lid_val)[0]
            if lane_sel.numel() == 0:
                continue
            neuron_sel = top_neurons[lane_sel, bit].to(device=device, dtype=torch.long)
            for j in range(n_children):
                sign_val = 1.0 if (j >> bit) & 1 else -1.0
                rows = j * n_lanes + lane_sel
                signs[lid_int][rows, :, neuron_sel] = sign_val

    children = SubproblemBatch(
        lb=batch.lb.index_select(0, parent_index),
        ub=batch.ub.index_select(0, parent_index),
        depths=batch.depths.index_select(0, parent_index) + k_eff,
        incremental_alpha=_gather(batch.incremental_alpha),
        incremental_eta=_gather(batch.incremental_eta),
        split_signs=signs,
        parent_margins=(
            batch.parent_margins.index_select(0, parent_index)
            if batch.parent_margins is not None
            else None
        ),
        lower_bound=(
            batch.lower_bound.index_select(0, parent_index)
            if batch.lower_bound is not None
            else None
        ),
    )
    return children, parent_index


def _multi_split_from_decision(
    batch: SubproblemBatch,
    net: Net,
    bounds_dict: Optional[Dict[int, Bounds]],
    nu_per_layer: Optional[Dict[int, torch.Tensor]],
    k_levels: int,
) -> Optional[Tuple[SubproblemBatch, torch.Tensor]]:
    """Joint top-k neuron split: each lane emits all 2^k sign combinations.

    Verdict-boundary multi-neuron splitting — "Mining Verdict Boundaries for
    Neural Network Verification", Jiawei Ren, Guanqin Zhang, Zhenya Zhang,
    Yulei Sui, FM 2026. Candidates are scored by the BaBSR heuristic
    (``_collect_neuron_candidates``).

    The 2^k children exactly partition each lane's region (every selected
    neuron is constrained to >=0 or <=0 in both directions across the
    combination set), so replacing the lane by its children is sound. Joint
    splits are super-additive in bound gain versus greedy single splits.
    Returns None when fewer than ``k_levels`` finite candidates exist in some
    lane (caller falls back to single-split).
    """
    if bounds_dict is None or nu_per_layer is None:
        return None
    cand = _collect_neuron_candidates(batch, bounds_dict, nu_per_layer)
    if cand is None:
        return None
    all_scores, all_layers, all_neurons = cand
    finite_per_lane = torch.isfinite(all_scores).sum(dim=1)
    k_eff = min(k_levels, int(finite_per_lane.min().item()))
    if k_eff < 2:
        return None
    top = torch.topk(all_scores, k=k_eff, dim=1).indices
    top_layers = all_layers.gather(1, top)
    top_neurons = all_neurons.gather(1, top)
    return _multi_split_from_groups(batch, net, top_layers, top_neurons, k_eff)
