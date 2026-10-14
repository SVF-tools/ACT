# ===- act/back_end/bab/branching/bounding.py - Subproblem Bounding ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Subproblem pool management for Branch-and-Bound.
#
#   Strategies selectable via ``--bab-bounding`` / ``BaBConfig.bounding``:
#
#   +---------------------+---------------------+--------------------------------+-------------+
#   | value               | pool class          | order function                 | --bab-top-k |
#   +=====================+=====================+================================+=============+
#   | depth_bound_blend   | TopKBounding        | DepthLowerBoundOrder:          | honoured    |
#   |                     |                     | 0.5*norm(depth) + 0.5*urgency  |             |
#   +---------------------+---------------------+--------------------------------+-------------+
#   | greedy              | TopKBounding        | GreedyOrder: best-first on     | honoured    |
#   |                     |                     | |lb| (Oliva-Greedy)            |             |
#   +---------------------+---------------------+--------------------------------+-------------+
#   | annealed            | TopKBounding        | SAOrder: Gumbel noise with     | honoured    |
#   |                     |                     | temp = cooling_rate**step      |             |
#   |                     |                     | (Oliva-SA)                     |             |
#   +---------------------+---------------------+--------------------------------+-------------+
#   | diverse_split_signs | DiverseTopKBounding | any order, then hash/soft      | honoured    |
#   |                     |                     | repulsion over split-sign      |             |
#   |                     |                     | signatures                     |             |
#   +---------------------+---------------------+--------------------------------+-------------+
#   | random              | RandomBounding      | none — uniform sampling        | ValueError  |
#   +---------------------+---------------------+--------------------------------+-------------+
#   | mcts                | MCTSBounding        | DepthLowerBoundOrder, pinned;  | ValueError  |
#   |                     |                     | observer-only, pop is plain    |             |
#   |                     |                     | top-k until UCB1 lands         |             |
#   +---------------------+---------------------+--------------------------------+-------------+
#
#   The first four share the ``TopKBounding`` pool and so honour ``top_k``; the
#   last two do not rank by an order function, so ``top_k`` is rejected rather
#   than silently ignored. ``top_k`` caps a single ``pop`` without discarding
#   anything — unlike ``evict_to``, which drops worst-priority leaves to honour a
#   frontier cap and therefore forces a sound ``UNKNOWN``.
#
#   ``GreedyOrder`` / ``SAOrder`` implement the Oliva order-leading exploration of the
#   BaB tree — "Efficient Neural Network Verification via Order Leading Exploration of
#   Branch-and-Bound Trees", Guanqin Zhang, Kota Fukuda, Zhenya Zhang, H.M.N. Dilum
#   Bandara, Shiping Chen, Jianjun Zhao, Yulei Sui, ECOOP 2025 (arXiv:2507.17453).
#   ``MCTSBounding`` is specified in ``docs/design/mcts_bab.md``.
#
#   A bounding strategy maintains a *pool* of pending subproblems and
#   decides which ones to process next.  All data flows through
#   ``SubproblemBatch`` (tensor-native) so that:
#
#     * ``push`` and ``pop`` operate on batches, not individual nodes.
#     * Internal storage can be a single tensor block (GPU-friendly).
#     * Future batch-parallel BaB pops N subproblems at once for
#       vectorised solving.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Callable, Dict, List, Literal, Optional, Protocol, Sequence, Set, Tuple

import torch

from act.back_end.bab.node import SubproblemBatch
from act.back_end.solver.solver_base import SolveStatus
from act.util.device_manager import get_default_dtype


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BoundingStrategy(ABC):
    """Abstract subproblem pool for Branch-and-Bound.

    Lifecycle (called by the BaB engine)::

        pool.push(root_batch)
        while not pool.empty:
            batch = pool.pop(batch_size=N)
            …solve / branch…
            pool.push(children_batch)

    Subclass contract
    ~~~~~~~~~~~~~~~~~
    * ``push`` accepts any-sized ``SubproblemBatch``.
    * ``pop(k)`` returns *at most* ``k`` subproblems; fewer if the
      pool is smaller.  Raises ``IndexError`` on empty pool.
    * ``__len__`` returns the current pool size.
    """

    @abstractmethod
    def push(self, batch: SubproblemBatch) -> None:
        """Enqueue a batch of subproblems.

        Args:
            batch: ``(N, D)`` subproblems to add to the pool.
        """
        ...

    @abstractmethod
    def pop(self, batch_size: int = 1) -> SubproblemBatch:
        """Dequeue subproblems for the next bounding iteration.

        Args:
            batch_size: Maximum number of subproblems to return.

        Returns:
            ``(M, D)`` batch where ``M <= batch_size``.

        Raises:
            IndexError: If the pool is empty.
        """
        ...

    @abstractmethod
    def evict_to(self, cap: int) -> int:
        """Drop pending subproblems until at most cap remain."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of pending subproblems."""
        ...

    @property
    def empty(self) -> bool:
        """True when no subproblems remain."""
        return len(self) == 0


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------


class RandomBounding(BoundingStrategy):
    """Uniform-random subproblem selection.

    ``pop(k)`` selects ``k`` subproblems uniformly at random from the
    pool (without replacement).

    Internal storage is fully tensor-native: three tensors ``(M, D)``,
    ``(M, D)``, ``(M,)`` for lower bounds, upper bounds, and depths
    respectively.
    """

    def __init__(self) -> None:
        self._lb: Optional[torch.Tensor] = None  # (M, D)
        self._ub: Optional[torch.Tensor] = None  # (M, D)
        self._depths: Optional[torch.Tensor] = None  # (M,)

    # -- BoundingStrategy interface -----------------------------------------

    def push(self, batch: SubproblemBatch) -> None:
        if self._lb is None:
            self._lb = batch.lb.clone()
            self._ub = batch.ub.clone()
            self._depths = batch.depths.clone()
        else:
            assert self._ub is not None and self._depths is not None
            self._lb = torch.cat([self._lb, batch.lb], dim=0)
            self._ub = torch.cat([self._ub, batch.ub], dim=0)
            self._depths = torch.cat([self._depths, batch.depths], dim=0)

    def pop(self, batch_size: int = 1) -> SubproblemBatch:
        if self.empty:
            raise IndexError("pop from empty pool")

        n = min(batch_size, len(self))
        assert self._lb is not None and self._ub is not None and self._depths is not None
        perm = torch.randperm(len(self), device=self._lb.device)
        selected = perm[:n]
        remaining = perm[n:]

        result = SubproblemBatch(
            lb=self._lb[selected],
            ub=self._ub[selected],
            depths=self._depths[selected],
        )

        if len(remaining) > 0:
            self._lb = self._lb[remaining]
            self._ub = self._ub[remaining]
            self._depths = self._depths[remaining]
        else:
            self._lb = None
            self._ub = None
            self._depths = None

        return result

    def evict_to(self, cap: int) -> int:
        total = len(self)
        if total <= cap or cap <= 0:
            return 0
        assert self._lb is not None and self._ub is not None and self._depths is not None
        self._lb = self._lb[:cap]
        self._ub = self._ub[:cap]
        self._depths = self._depths[:cap]
        return total - cap

    def __len__(self) -> int:
        return 0 if self._lb is None else self._lb.shape[0]


# ---------------------------------------------------------------------------
# Top-k priority selection (quantitative total order: depth + lower bound)
# ---------------------------------------------------------------------------


def _clone_optional_dict(
    d: Optional[Dict[int, torch.Tensor]],
) -> Optional[Dict[int, torch.Tensor]]:
    return {key: tensor.clone() for key, tensor in d.items()} if d is not None else None


def _index_optional_dict(
    d: Optional[Dict[int, torch.Tensor]], idx: torch.Tensor
) -> Optional[Dict[int, torch.Tensor]]:
    if d is None:
        return None
    return {key: tensor.index_select(0, idx.to(tensor.device)) for key, tensor in d.items()}


def _merge_optional_dict(
    existing: Optional[Dict[int, torch.Tensor]],
    n_existing: int,
    incoming: Optional[Dict[int, torch.Tensor]],
    n_incoming: int,
) -> Optional[Dict[int, torch.Tensor]]:
    # Per-key concat with key-union; a subproblem missing a key is padded with
    # zeros (e.g. split_signs keys differ per branch — a missing layer means "not
    # split", i.e. all-zero signs). Keeps the pool lossless across heterogeneous
    # incremental-state/split structures.
    if existing is None and incoming is None:
        return None
    existing = existing or {}
    incoming = incoming or {}
    merged: Dict[int, torch.Tensor] = {}
    for key in sorted(set(existing) | set(incoming)):
        te = existing.get(key)
        ti = incoming.get(key)
        ref = te if te is not None else ti
        assert ref is not None
        if te is not None and ti is not None:
            assert te.shape[1:] == ti.shape[1:], (
                f"optional dict key {key} trailing shape mismatch: "
                f"existing {tuple(te.shape[1:])} vs incoming {tuple(ti.shape[1:])}"
            )
        trailing = ref.shape[1:]
        if te is None:
            te = torch.zeros((n_existing, *trailing), dtype=ref.dtype, device=ref.device)
        if ti is None:
            ti = torch.zeros((n_incoming, *trailing), dtype=ref.dtype, device=ref.device)
        merged[key] = torch.cat([te, ti], dim=0)
    return merged


class OrderFunction(Protocol):
    def __call__(self, depths: torch.Tensor, lower_bound: torch.Tensor) -> torch.Tensor:
        ...


def _advance_order_schedule(order: OrderFunction) -> None:
    """Tick a stateful order's schedule once per wave.

    Scoring and scheduling must stay separate: ``pop`` skips ``_priority_scores``
    whenever the pool already fits the wave, and ``evict_to`` scores without
    consuming a wave. Folding the tick into ``__call__`` therefore made the
    annealing temperature depend on pool size and eviction pressure rather than
    on elapsed waves.
    """
    advance = getattr(order, "advance_schedule", None)
    if advance is not None:
        advance()


class DepthLowerBoundOrder:
    def __init__(self, depth_weight: float = 0.5, bound_weight: float = 0.5) -> None:
        self.depth_weight = depth_weight
        self.bound_weight = bound_weight

    def __call__(self, depths: torch.Tensor, lower_bound: torch.Tensor) -> torch.Tensor:
        dtype = lower_bound.dtype
        eps = torch.finfo(dtype).eps
        d = depths.to(dtype=dtype)
        d_norm = (d - d.min()) / (d.max() - d.min()).clamp(min=eps)
        urgency = (lower_bound.max() - lower_bound) / (
            lower_bound.max() - lower_bound.min()
        ).clamp(min=eps)
        return self.depth_weight * d_norm + self.bound_weight * urgency


class GreedyOrder(DepthLowerBoundOrder):
    """Oliva-Greedy: best-first by lower bound (``|lb|``).

    "Efficient Neural Network Verification via Order Leading Exploration of
    Branch-and-Bound Trees", Guanqin Zhang, Kota Fukuda, Zhenya Zhang,
    H.M.N. Dilum Bandara, Shiping Chen, Jianjun Zhao, Yulei Sui, ECOOP 2025.
    """

    def __init__(self) -> None:
        super().__init__(depth_weight=0.0, bound_weight=1.0)


class SAOrder:
    """Oliva-SA: temperature-annealed exploration order.

    "Efficient Neural Network Verification via Order Leading Exploration of
    Branch-and-Bound Trees", Guanqin Zhang, Kota Fukuda, Zhenya Zhang,
    H.M.N. Dilum Bandara, Shiping Chen, Jianjun Zhao, Yulei Sui, ECOOP 2025.

    ``temp = cooling_rate ** step`` cools each call, so selection explores early and
    converges to greedy (``|lb|`` best-first) as it cools.
    """

    def __init__(self, cooling_rate: float = 0.99) -> None:
        self.cooling_rate = cooling_rate
        self.step = 0

    def advance_schedule(self) -> None:
        self.step += 1

    def __call__(self, depths: torch.Tensor, lower_bound: torch.Tensor) -> torch.Tensor:
        dtype = lower_bound.dtype
        eps = torch.finfo(dtype).eps
        temp = max(self.cooling_rate ** self.step, 1e-6)
        base = (lower_bound.max() - lower_bound) / (
            lower_bound.max() - lower_bound.min()
        ).clamp(min=eps)
        u = torch.rand_like(base).clamp(min=eps, max=1.0 - eps)
        gumbel = -torch.log(-torch.log(u))
        return base / temp + gumbel


class TopKBounding(BoundingStrategy):
    """Priority pool: keep the top-k subproblems chosen by an order callable.

    The BaB tensor (batch) size is capped by compute resources, so when the pool
    holds more subproblems than the requested batch size, the next wave keeps only
    the k highest-priority ones; the rest stay pooled. Priority comes from a
    swappable order strategy (default :class:`DepthLowerBoundOrder` — a
    50/50 blend of depth and lower bound).

    ``k`` caps how many subproblems a single ``pop`` may return, independently of
    the caller's ``batch_size``. ``k = 0`` means unbounded, matching the
    ``BaBConfig.frontier_cap`` convention. Unlike ``evict_to``, capping ``pop``
    never discards a subproblem: the remainder stays pooled for later waves, so
    a smaller ``k`` only re-sorts priorities more often.

    Storage is lossless: bounds, depth, lower bound, parent margins and every
    incremental-state dict (including split_signs, which neuron-split BaB requires) are
    preserved across push/pop.
    """

    def __init__(
        self,
        order: Optional[OrderFunction] = None,
        select_probe: Optional[Callable[[SubproblemBatch], None]] = None,
        *,
        k: int = 0,
    ) -> None:
        if k < 0:
            raise ValueError(f"top-k must be non-negative, got k={k}")
        self.order: OrderFunction = order if order is not None else DepthLowerBoundOrder()
        self.k = int(k)
        self.select_probe = select_probe
        self._lb: Optional[torch.Tensor] = None
        self._ub: Optional[torch.Tensor] = None
        self._depths: Optional[torch.Tensor] = None
        self._lower_bound: Optional[torch.Tensor] = None
        self._parent_margins: Optional[torch.Tensor] = None
        self._node_id: Optional[torch.Tensor] = None
        self._parent_id: Optional[torch.Tensor] = None
        self._incremental_alpha: Optional[Dict[int, torch.Tensor]] = None
        self._incremental_eta: Optional[Dict[int, torch.Tensor]] = None
        self._split_signs: Optional[Dict[int, torch.Tensor]] = None

    def push(self, batch: SubproblemBatch) -> None:
        size_before = len(self)
        n_new = batch.batch_size
        device, dtype = batch.lb.device, batch.lb.dtype
        lower = (
            batch.lower_bound
            if batch.lower_bound is not None
            else torch.zeros(n_new, dtype=dtype, device=device)
        )
        parent = (
            batch.parent_margins
            if batch.parent_margins is not None
            else torch.zeros(n_new, dtype=dtype, device=device)
        )
        prev_lb, prev_ub, prev_depths = self._lb, self._ub, self._depths
        prev_lower, prev_parent = self._lower_bound, self._parent_margins
        if prev_lb is None:
            self._lb = batch.lb.clone()
            self._ub = batch.ub.clone()
            self._depths = batch.depths.clone()
            self._lower_bound = lower.clone()
            self._parent_margins = parent.clone()
            self._node_id = batch.node_id.clone() if batch.node_id is not None else None
            self._parent_id = batch.parent_id.clone() if batch.parent_id is not None else None
            self._incremental_alpha = _clone_optional_dict(batch.incremental_alpha)
            self._incremental_eta = _clone_optional_dict(batch.incremental_eta)
            self._split_signs = _clone_optional_dict(batch.split_signs)
            self._assert_push_transition(size_before, n_new)
            return

        assert prev_ub is not None and prev_depths is not None
        assert prev_lower is not None and prev_parent is not None
        assert (self._node_id is None) == (batch.node_id is None)
        assert (self._parent_id is None) == (batch.parent_id is None)
        n_old = prev_lb.shape[0]
        self._incremental_alpha = _merge_optional_dict(self._incremental_alpha, n_old, batch.incremental_alpha, n_new)
        self._incremental_eta = _merge_optional_dict(self._incremental_eta, n_old, batch.incremental_eta, n_new)
        self._split_signs = _merge_optional_dict(self._split_signs, n_old, batch.split_signs, n_new)
        self._lb = torch.cat([prev_lb, batch.lb], dim=0)
        self._ub = torch.cat([prev_ub, batch.ub], dim=0)
        self._depths = torch.cat([prev_depths, batch.depths], dim=0)
        self._lower_bound = torch.cat([prev_lower, lower.to(prev_lower)], dim=0)
        self._parent_margins = torch.cat([prev_parent, parent.to(prev_parent)], dim=0)
        if self._node_id is not None:
            assert batch.node_id is not None
            self._node_id = torch.cat([self._node_id, batch.node_id.to(self._node_id.device)], dim=0)
        if self._parent_id is not None:
            assert batch.parent_id is not None
            self._parent_id = torch.cat([self._parent_id, batch.parent_id.to(self._parent_id.device)], dim=0)
        self._assert_push_transition(size_before, n_new)

    def pop(self, batch_size: int = 1) -> SubproblemBatch:
        lb = self._lb
        if lb is None:
            raise IndexError("pop from empty pool")
        total = lb.shape[0]
        n = min(batch_size, total)
        if self.k > 0:
            n = min(n, self.k)
        _advance_order_schedule(self.order)
        if n >= total:
            selected = torch.arange(total, device=lb.device)
            remaining: Optional[torch.Tensor] = None
        else:
            order = torch.argsort(self._priority_scores(), descending=True)
            selected = order[:n]
            remaining = order[n:]

        result = self._build(selected)
        if self.select_probe is not None:
            self.select_probe(result)
        if remaining is None or remaining.numel() == 0:
            self._clear()
        else:
            self._restrict(remaining)
        self._assert_pop_transition(total, result.batch_size)
        return result

    def _assert_push_transition(self, size_before: int, added: int) -> None:
        size_after = len(self)
        assert size_after == size_before + added, (
            "TOP-K BOUNDING invariant violated: push changed pool by wrong size "
            f"(before={size_before}, batch_size={added}, after={size_after}, "
            f"expected_after={size_before + added})"
        )

    def _assert_pop_transition(self, size_before: int, popped: int) -> None:
        size_after = len(self)
        cap_ok = self.k <= 0 or popped <= self.k
        bookkeeping_ok = size_before - popped == size_after
        assert cap_ok and bookkeeping_ok, (
            "TOP-K BOUNDING invariant violated: pop cap or pool bookkeeping failed "
            f"(before={size_before}, popped={popped}, after={size_after}, "
            f"k={self.k}, expected_after={size_before - popped})"
        )

    def view_all(self) -> SubproblemBatch:
        """Return a lossless, non-destructive view of the full frontier."""
        if self._lb is None:
            raise IndexError("view of empty pool")
        indices = torch.arange(self._lb.shape[0], device=self._lb.device)
        return self._build(indices)

    def replace_all(self, batch: SubproblemBatch) -> None:
        """Replace the full frontier without scoring or advancing schedules."""
        self._clear()
        if batch.batch_size > 0:
            self.push(batch)

    def _priority_scores(self) -> torch.Tensor:
        depths_t, lb = self._depths, self._lower_bound
        assert depths_t is not None and lb is not None
        return self.order(depths_t, lb)

    def _build(self, idx: torch.Tensor) -> SubproblemBatch:
        lb, ub, depths = self._lb, self._ub, self._depths
        lower, parent = self._lower_bound, self._parent_margins
        assert lb is not None and ub is not None and depths is not None
        assert lower is not None and parent is not None
        idx = idx.to(lb.device)
        return SubproblemBatch(
            lb=lb.index_select(0, idx),
            ub=ub.index_select(0, idx),
            depths=depths.index_select(0, idx),
            incremental_alpha=_index_optional_dict(self._incremental_alpha, idx),
            incremental_eta=_index_optional_dict(self._incremental_eta, idx),
            split_signs=_index_optional_dict(self._split_signs, idx),
            parent_margins=parent.index_select(0, idx),
            lower_bound=lower.index_select(0, idx),
            node_id=(
                self._node_id.index_select(0, idx.to(self._node_id.device))
                if self._node_id is not None
                else None
            ),
            parent_id=(
                self._parent_id.index_select(0, idx.to(self._parent_id.device))
                if self._parent_id is not None
                else None
            ),
        )

    def _restrict(self, idx: torch.Tensor) -> None:
        kept = self._build(idx)
        self._lb, self._ub, self._depths = kept.lb, kept.ub, kept.depths
        self._lower_bound, self._parent_margins = kept.lower_bound, kept.parent_margins
        self._node_id, self._parent_id = kept.node_id, kept.parent_id
        self._incremental_alpha, self._incremental_eta, self._split_signs = (
            kept.incremental_alpha,
            kept.incremental_eta,
            kept.split_signs,
        )

    def _clear(self) -> None:
        self._lb = self._ub = self._depths = None
        self._lower_bound = self._parent_margins = None
        self._node_id = self._parent_id = None
        self._incremental_alpha = self._incremental_eta = self._split_signs = None

    def evict_to(self, cap: int) -> int:
        total = len(self)
        if total <= cap or cap <= 0:
            return 0
        order = torch.argsort(self._priority_scores(), descending=True)
        self._restrict(order[:cap])
        return total - cap

    def __len__(self) -> int:
        return 0 if self._lb is None else self._lb.shape[0]


class DiverseTopKBounding(TopKBounding):
    """Top-k priority pool with optional diversity-aware scheduling.

    ``diversity_mode='hash'`` preserves the original exact split-sign
    de-duplication behaviour.  ``'soft'`` uses a value-weighted farthest-point
    selector over split-sign vectors (neuron splitting) or box centres (input
    splitting / no split signs).  This is scheduling-only: no node is pruned, no
    bound is changed, and all non-selected nodes remain in the pool for later
    waves.  ``'none'`` is an explicit off switch and is equivalent to
    :class:`TopKBounding` selection.
    """

    def __init__(
        self,
        order: Optional[OrderFunction] = None,
        select_probe: Optional[Callable[[SubproblemBatch], None]] = None,
        *,
        k: int = 0,
        diversity_mode: Literal["hash", "soft", "none"] = "hash",
        diversity_weight: float = 1.0,
    ) -> None:
        super().__init__(order, select_probe=select_probe, k=k)
        if diversity_mode not in {"hash", "soft", "none"}:
            raise ValueError(
                "diversity_mode must be one of 'hash', 'soft', or 'none', "
                f"got {diversity_mode!r}"
            )
        self.diversity_mode = diversity_mode
        self.diversity_weight = float(diversity_weight)

    def pop(self, batch_size: int = 1) -> SubproblemBatch:
        lb = self._lb
        if lb is None:
            raise IndexError("pop from empty pool")
        total = lb.shape[0]
        n = min(batch_size, total)
        if self.k > 0:
            n = min(n, self.k)
        _advance_order_schedule(self.order)
        if n >= total:
            selected = torch.arange(total, device=lb.device)
            remaining: Optional[torch.Tensor] = None
        else:
            order = torch.argsort(self._priority_scores(), descending=True)
            if self.diversity_mode == "none":
                selected = order[:n]
            elif self.diversity_mode == "soft":
                selected = self._soft_diverse_select(order, n)
            else:
                selected = self._dedup_select(order, n)
            selected_mask = torch.zeros(total, dtype=torch.bool, device=lb.device)
            selected_mask[selected] = True
            remaining = order[~selected_mask.index_select(0, order)]

        result = self._build(selected)
        if self.select_probe is not None:
            self.select_probe(result)
        if remaining is None or remaining.numel() == 0:
            self._clear()
        else:
            self._restrict(remaining)
        self._assert_pop_transition(total, result.batch_size)
        return result

    def _dedup_select(self, order: torch.Tensor, n: int) -> torch.Tensor:
        signatures = self._split_sign_signatures()
        if signatures is None:
            return order[:n]

        selected: List[int] = []
        selected_set: Set[int] = set()
        seen: Set[Tuple[int, ...]] = set()
        ordered_indices = [int(i) for i in order.detach().cpu().tolist()]

        for idx in ordered_indices:
            signature = signatures[idx]
            if signature in seen:
                continue
            selected.append(idx)
            selected_set.add(idx)
            seen.add(signature)
            if len(selected) == n:
                break

        if len(selected) < n:
            for idx in ordered_indices:
                if idx in selected_set:
                    continue
                selected.append(idx)
                selected_set.add(idx)
                if len(selected) == n:
                    break

        return torch.tensor(selected, dtype=torch.long, device=order.device)

    def _split_sign_signatures(self) -> Optional[List[Tuple[int, ...]]]:
        split_signs = self._split_signs
        if not split_signs:
            return None

        pieces: List[torch.Tensor] = []
        for layer_id in sorted(split_signs):
            value = split_signs[layer_id]
            if value.shape[0] == 0:
                continue
            pieces.append(value.detach().reshape(value.shape[0], -1).to(device="cpu"))
        if not pieces:
            return None

        features = torch.cat(pieces, dim=1)
        if features.shape[1] == 0:
            return None
        return [tuple(int(v) for v in row.tolist()) for row in features]

    def _soft_diverse_select(self, order: torch.Tensor, n: int) -> torch.Tensor:
        features = self._diversity_features()
        if features is None or features.shape[0] < 2:
            return order[:n]

        candidate_features = features.index_select(0, order.to(features.device))
        distances = torch.cdist(candidate_features, candidate_features, p=2)
        max_distance = distances.max().clamp(min=torch.finfo(distances.dtype).eps)
        distances = distances / max_distance

        lb = self._lb
        assert lb is not None
        priorities = self._priority_scores().index_select(0, order.to(lb.device))
        priorities = priorities.to(device=features.device, dtype=features.dtype)
        priority_span = (priorities.max() - priorities.min()).clamp(
            min=torch.finfo(priorities.dtype).eps
        )
        priorities = (priorities - priorities.min()) / priority_span

        selected_positions: List[int] = [0]
        remaining = torch.ones(order.shape[0], dtype=torch.bool, device=features.device)
        remaining[0] = False
        min_dist_to_selected = distances[0].clone()

        while len(selected_positions) < n and bool(remaining.any().item()):
            scores = priorities + self.diversity_weight * min_dist_to_selected
            scores = scores.masked_fill(~remaining, -torch.inf)
            next_pos = int(torch.argmax(scores).item())
            selected_positions.append(next_pos)
            remaining[next_pos] = False
            min_dist_to_selected = torch.minimum(
                min_dist_to_selected, distances[next_pos]
            )

        if len(selected_positions) < n:
            for pos in range(order.shape[0]):
                if pos not in selected_positions:
                    selected_positions.append(pos)
                    if len(selected_positions) == n:
                        break
        return order[torch.tensor(selected_positions, dtype=torch.long, device=order.device)]

    def _diversity_features(self) -> Optional[torch.Tensor]:
        split_features = self._split_sign_features()
        if split_features is not None:
            return split_features
        return self._box_center_features()

    def _split_sign_features(self) -> Optional[torch.Tensor]:
        split_signs = self._split_signs
        if not split_signs:
            return None
        pieces: List[torch.Tensor] = []
        for layer_id in sorted(split_signs):
            value = split_signs[layer_id]
            if value.shape[0] == 0:
                continue
            pieces.append(
                value.detach().reshape(value.shape[0], -1).to(dtype=get_default_dtype())
            )
        if not pieces:
            return None
        features = torch.cat(pieces, dim=1)
        if features.shape[1] == 0:
            return None
        return features

    def _box_center_features(self) -> Optional[torch.Tensor]:
        if self._lb is None or self._ub is None:
            return None
        centers = (
            ((self._lb + self._ub) / 2.0)
            .detach()
            .to(dtype=get_default_dtype())
        )
        if centers.ndim > 2:
            centers = centers.reshape(centers.shape[0], -1)
        if centers.shape[1] == 0:
            return None
        return centers


ROOT_PARENT = -1


def _mcts_visit_accounting_valid(
    parent: Dict[int, int], visits: Dict[int, int], n_tot: int
) -> bool:
    if n_tot < 0 or any(count < 0 for count in visits.values()):
        return False
    child_visit_sums: Dict[int, int] = {}
    for node, count in visits.items():
        parent_id = parent.get(node, ROOT_PARENT)
        if parent_id != ROOT_PARENT:
            child_visit_sums[parent_id] = child_visit_sums.get(parent_id, 0) + count
    if any(visits.get(node, 0) < total for node, total in child_visit_sums.items()):
        return False
    root_visits = sum(
        count
        for node, count in visits.items()
        if parent.get(node, ROOT_PARENT) == ROOT_PARENT
    )
    return root_visits == n_tot


def _mcts_visit_diagnostics(
    parent: Dict[int, int], visits: Dict[int, int]
) -> tuple[int, int, int]:
    child_visit_sums: Dict[int, int] = {}
    for node, count in visits.items():
        parent_id = parent.get(node, ROOT_PARENT)
        if parent_id != ROOT_PARENT:
            child_visit_sums[parent_id] = child_visit_sums.get(parent_id, 0) + count
    root_visits = sum(
        count
        for node, count in visits.items()
        if parent.get(node, ROOT_PARENT) == ROOT_PARENT
    )
    min_visit = min(visits.values(), default=0)
    max_child_excess = max(
        (total - visits.get(node, 0) for node, total in child_visit_sums.items()),
        default=0,
    )
    return root_visits, min_visit, max_child_excess


class MCTSBounding(BoundingStrategy):
    """MCTS side tables (``N``/``Q``) over the BaB tree, maintained as a pure observer.

    ``order`` governs **eviction priority only**; selection is UCB1 (W2). At this
    observer stage ``pop`` is plain top-k by ``order`` — no UCB1 term is applied
    yet — while ``evict_to`` keeps the lb-based ``order`` priority so a frontier
    cap always drops the least promising leaves.

    Storage is lossless: bounds, depth, lower bound, parent margins, ``node_id`` /
    ``parent_id`` provenance and every incremental-state dict (including
    split_signs) are preserved across push/pop.
    """

    def __init__(
        self,
        order: Optional[OrderFunction] = None,
        select_probe: Optional[Callable[[SubproblemBatch], None]] = None,
        *,
        exploration: float = 1.0,
        lambda_: float = 0.5,
        virtual_loss: float = 1.0,  # inert at K=1; reserved for the batched extension
    ) -> None:
        self.order: OrderFunction = order if order is not None else DepthLowerBoundOrder()
        self.select_probe = select_probe
        self.exploration: float = float(exploration)
        self.lambda_: float = float(lambda_)
        self.virtual_loss: float = float(virtual_loss)
        self.parent: Dict[int, int] = {}
        self.N: Dict[int, int] = {}
        self.Q: Dict[int, float] = {}
        self.n_tot: int = 0
        self._lb: Optional[torch.Tensor] = None
        self._ub: Optional[torch.Tensor] = None
        self._depths: Optional[torch.Tensor] = None
        self._lower_bound: Optional[torch.Tensor] = None
        self._parent_margins: Optional[torch.Tensor] = None
        self._node_id: Optional[torch.Tensor] = None
        self._parent_id: Optional[torch.Tensor] = None
        self._incremental_alpha: Optional[Dict[int, torch.Tensor]] = None
        self._incremental_eta: Optional[Dict[int, torch.Tensor]] = None
        self._split_signs: Optional[Dict[int, torch.Tensor]] = None

    def push(self, batch: SubproblemBatch) -> None:
        if batch.node_id is None or batch.parent_id is None:
            missing = "node_id" if batch.node_id is None else "parent_id"
            raise ValueError(
                f"MCTSBounding.push requires provenance, but batch.{missing} is None"
            )
        n_new = batch.batch_size
        device, dtype = batch.lb.device, batch.lb.dtype
        lower = (
            batch.lower_bound
            if batch.lower_bound is not None
            else torch.zeros(n_new, dtype=dtype, device=device)
        )
        parent = (
            batch.parent_margins
            if batch.parent_margins is not None
            else torch.zeros(n_new, dtype=dtype, device=device)
        )
        prev_lb = self._lb
        if prev_lb is None:
            self._lb = batch.lb.clone()
            self._ub = batch.ub.clone()
            self._depths = batch.depths.clone()
            self._lower_bound = lower.clone()
            self._parent_margins = parent.clone()
            self._node_id = batch.node_id.clone()
            self._parent_id = batch.parent_id.clone()
            self._incremental_alpha = _clone_optional_dict(batch.incremental_alpha)
            self._incremental_eta = _clone_optional_dict(batch.incremental_eta)
            self._split_signs = _clone_optional_dict(batch.split_signs)
        else:
            prev_ub, prev_depths = self._ub, self._depths
            prev_lower, prev_parent = self._lower_bound, self._parent_margins
            prev_node, prev_parent_id = self._node_id, self._parent_id
            assert prev_ub is not None and prev_depths is not None
            assert prev_lower is not None and prev_parent is not None
            assert prev_node is not None and prev_parent_id is not None
            n_old = prev_lb.shape[0]
            self._incremental_alpha = _merge_optional_dict(self._incremental_alpha, n_old, batch.incremental_alpha, n_new)
            self._incremental_eta = _merge_optional_dict(self._incremental_eta, n_old, batch.incremental_eta, n_new)
            self._split_signs = _merge_optional_dict(self._split_signs, n_old, batch.split_signs, n_new)
            self._lb = torch.cat([prev_lb, batch.lb], dim=0)
            self._ub = torch.cat([prev_ub, batch.ub], dim=0)
            self._depths = torch.cat([prev_depths, batch.depths], dim=0)
            self._lower_bound = torch.cat([prev_lower, lower.to(prev_lower)], dim=0)
            self._parent_margins = torch.cat([prev_parent, parent.to(prev_parent)], dim=0)
            self._node_id = torch.cat([prev_node, batch.node_id.to(prev_node.device)], dim=0)
            self._parent_id = torch.cat([prev_parent_id, batch.parent_id.to(prev_parent_id.device)], dim=0)

        for nid, pid in zip(batch.node_id.tolist(), batch.parent_id.tolist()):
            self.parent[int(nid)] = int(pid)

    def pop(self, batch_size: int = 1) -> SubproblemBatch:
        lb = self._lb
        if lb is None:
            raise IndexError("pop from empty pool")
        total = lb.shape[0]
        n = min(batch_size, total)
        _advance_order_schedule(self.order)
        if n >= total:
            selected = torch.arange(total, device=lb.device)
            remaining: Optional[torch.Tensor] = None
        else:
            order = torch.argsort(self._priority_scores(), descending=True)
            selected = order[:n]
            remaining = order[n:]

        result = self._build(selected)
        if self.select_probe is not None:
            self.select_probe(result)
        if remaining is None or remaining.numel() == 0:
            self._clear()
        else:
            self._restrict(remaining)
        return result

    def evict_to(self, cap: int) -> int:
        total = len(self)
        if total <= cap or cap <= 0:
            return 0
        order = torch.argsort(self._priority_scores(), descending=True)
        self._restrict(order[:cap])
        return total - cap

    def __len__(self) -> int:
        return 0 if self._lb is None else self._lb.shape[0]

    def _priority_scores(self) -> torch.Tensor:
        depths_t, lb = self._depths, self._lower_bound
        assert depths_t is not None and lb is not None
        return self.order(depths_t, lb)

    def _build(self, idx: torch.Tensor) -> SubproblemBatch:
        lb, ub, depths = self._lb, self._ub, self._depths
        lower, parent = self._lower_bound, self._parent_margins
        node_id, parent_id = self._node_id, self._parent_id
        assert lb is not None and ub is not None and depths is not None
        assert lower is not None and parent is not None
        assert node_id is not None and parent_id is not None
        idx = idx.to(lb.device)
        return SubproblemBatch(
            lb=lb.index_select(0, idx),
            ub=ub.index_select(0, idx),
            depths=depths.index_select(0, idx),
            incremental_alpha=_index_optional_dict(self._incremental_alpha, idx),
            incremental_eta=_index_optional_dict(self._incremental_eta, idx),
            split_signs=_index_optional_dict(self._split_signs, idx),
            parent_margins=parent.index_select(0, idx),
            lower_bound=lower.index_select(0, idx),
            node_id=node_id.index_select(0, idx.to(node_id.device)),
            parent_id=parent_id.index_select(0, idx.to(parent_id.device)),
        )

    def _restrict(self, idx: torch.Tensor) -> None:
        kept = self._build(idx)
        self._lb, self._ub, self._depths = kept.lb, kept.ub, kept.depths
        self._lower_bound, self._parent_margins = kept.lower_bound, kept.parent_margins
        self._node_id, self._parent_id = kept.node_id, kept.parent_id
        self._incremental_alpha, self._incremental_eta, self._split_signs = (
            kept.incremental_alpha,
            kept.incremental_eta,
            kept.split_signs,
        )

    def _clear(self) -> None:
        self._lb = self._ub = self._depths = None
        self._lower_bound = self._parent_margins = None
        self._node_id = self._parent_id = None
        self._incremental_alpha = self._incremental_eta = self._split_signs = None

    def observe(
        self,
        node_ids: torch.Tensor,
        lower_bounds: torch.Tensor,
        statuses: Sequence[str],
        depths: torch.Tensor,
        n_unstable: int,
    ) -> None:
        """Backpropagate one wave of solve results into ``N``/``Q``.

        Callers must invoke this only after counterexample validation, so a
        ``SAT`` status here always denotes a spurious, still-unresolved lane.
        """
        ids = node_ids.detach().cpu()
        lb = lower_bounds.detach().cpu().to(dtype=get_default_dtype())
        depth = depths.detach().cpu().to(dtype=get_default_dtype())
        blended = self.lambda_ * depth / max(n_unstable, 1) + (
            1.0 - self.lambda_
        ) * self._rank01(ids, lb)
        blended = torch.where(
            torch.isnan(lb) | torch.isnan(blended),
            torch.full_like(blended, -math.inf),
            blended,
        )

        for i, (node, status) in enumerate(zip(ids.tolist(), statuses)):
            reward = -math.inf if status == SolveStatus.UNSAT else float(blended[i])
            self.n_tot += 1
            visited = int(node)
            while visited != ROOT_PARENT:
                self.N[visited] = self.N.get(visited, 0) + 1
                visited = self.parent[visited]
            valued = int(node)
            while valued != ROOT_PARENT and self.Q.get(valued, -math.inf) < reward:
                self.Q[valued] = reward
                valued = self.parent[valued]
        assert _mcts_visit_accounting_valid(self.parent, self.N, self.n_tot), (
            "MCTS invariant violated: visit-tree accounting is inconsistent "
            f"(n_tot={self.n_tot}, root_visits="
            f"{_mcts_visit_diagnostics(self.parent, self.N)[0]}, min_visit="
            f"{_mcts_visit_diagnostics(self.parent, self.N)[1]}, "
            f"max_child_excess={_mcts_visit_diagnostics(self.parent, self.N)[2]})"
        )

    def frontier_parent_visit_histogram(self) -> Dict[int, int]:
        parent_ids = self._parent_id
        if parent_ids is None:
            return {}
        counts = Counter(self.N.get(int(pid), 0) for pid in parent_ids.tolist())
        return dict(sorted(counts.items()))

    @staticmethod
    def _rank01(node_ids: torch.Tensor, lower_bounds: torch.Tensor) -> torch.Tensor:
        n = int(lower_bounds.numel())
        if n < 2:
            return torch.zeros_like(lower_bounds)
        # Rank lexicographically on (lb, node_id): scale-free in [0, 1] and
        # invariant to the order the wave's results arrive in.
        by_id = torch.argsort(node_ids, stable=True)
        order = by_id[torch.argsort(lower_bounds[by_id], stable=True)]
        ranks = torch.empty_like(lower_bounds)
        ranks[order] = torch.arange(n, dtype=lower_bounds.dtype, device=lower_bounds.device)
        return ranks / (n - 1)
