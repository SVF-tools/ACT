"""
Batched seed corpus for GPU-accelerated fuzzing.

FuzzingSeed — every field carries a leading batch dim B (no scalar path).
SeedCorpus  — parallel-tensor pool.
  Selection: energy-weighted sampling with replacement → FuzzingSeed(B).
  Insertion: boolean-mask filtered add() with per-spec-row byte-hash dedup.
  Storage:   N parallel 1-D/N-D tensors grown via torch.cat on insert.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from typing import Optional, Union
import numpy as np
import torch

from act.front_end.spec_creator_base import LabeledInputTensor
from act.util.device_manager import get_default_device

SELECTION_STRATEGIES = ("energy", "random")


class FuzzingSeed:
    """
    Batched fuzzing seed. All fields are tensors with leading batch dim B.
    A single FuzzingSeed instance represents B seeds. 
    
    Attributes:
        tensor:          [B, ...] Input tensors (current, may be mutated)
        original_tensor: [B, ...] Original clean inputs (NEVER mutated)
        original_index:  [B] int64 — synthesized spec-row index
        label:           [B] int64 — ground truth label (-1 = no label)
        energy:          [B] float — seed energy (higher = more interesting)
        depth:           [B] int64 — how many mutations from original seed
        id:              [B] int64 — unique seed identifiers (auto-generated)
        parent_id:       [B] int64 — parent seed IDs (-1 = no parent)
        select_count:    [B] int64 — how many times this seed has been selected
    """
    
    _id_counter: int = 0
    
    @classmethod
    def _next_ids(cls, n: int) -> torch.Tensor:
        """Generate n unique sequential int64 IDs."""
        start = cls._id_counter
        cls._id_counter += n
        return torch.arange(start, start + n, dtype=torch.long)
    
    def __init__(
        self,
        tensor: torch.Tensor,
        original_tensor: Optional[torch.Tensor] = None,
        original_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        id: Optional[torch.Tensor] = None,
        parent_id: Optional[torch.Tensor] = None,
        select_count: Optional[torch.Tensor] = None,
    ):
        B = tensor.shape[0]
        self.tensor = tensor
        self.original_tensor = original_tensor if original_tensor is not None else tensor.clone()
        self.original_index = original_index if original_index is not None else torch.zeros(B, dtype=torch.long)
        self.label = label if label is not None else torch.full((B,), -1, dtype=torch.long)
        self.energy = energy if energy is not None else torch.ones(B)
        self.depth = depth if depth is not None else torch.zeros(B, dtype=torch.long)
        self.id = id if id is not None else FuzzingSeed._next_ids(B)
        self.parent_id = parent_id if parent_id is not None else torch.full((B,), -1, dtype=torch.long)
        self.select_count = select_count if select_count is not None else torch.zeros(B, dtype=torch.long)
    
    def __len__(self) -> int:
        """Return batch size B."""
        return self.tensor.shape[0]
    
    def __getitem__(self, idx: Union[int, slice, torch.Tensor]) -> 'FuzzingSeed':
        """Slice the batch. Returns FuzzingSeed with sub-batch."""
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        return FuzzingSeed(
            tensor=self.tensor[idx],
            original_tensor=self.original_tensor[idx],
            original_index=self.original_index[idx],
            label=self.label[idx],
            energy=self.energy[idx],
            depth=self.depth[idx],
            id=self.id[idx],
            parent_id=self.parent_id[idx],
            select_count=self.select_count[idx],
        )


class SeedCorpus:
    """
    AFL-style seed corpus with energy-based scheduling.
    
    Features:
    - Energy-based seed selection (higher energy = more likely to select)
    - Random selection as fallback
    - Automatic seed deduplication by spec row and tensor hash
    - All storage on device_manager's device (GPU when available)
    
    Example:
        >>> corpus = SeedCorpus(initial_seeds, strategy="energy")
        >>> batch = corpus.select(32)       # FuzzingSeed with B=32
        >>> corpus.add(child, mask)   # Add interesting children
    """
    
    def __init__(self, 
                 initial_seeds: list[LabeledInputTensor],
                 strategy: str = "energy"):
        """
        Initialize seed corpus from LabeledInputTensor list.
        
        Converts list of LabeledInputTensor into parallel tensor storage.
        
        Args:
            initial_seeds: Initial seeds from spec creators
            strategy: Selection strategy ("energy" or "random")
        """
        if strategy not in SELECTION_STRATEGIES:
            raise ValueError(f"Unknown seed selection strategy: {strategy}. Valid options: {list(SELECTION_STRATEGIES)}")
        self.strategy = strategy
        self.seen_hashes: set[tuple[int, int]] = set()
        self._device = get_default_device()
        
        # Build parallel lists from initial seeds, then stack to tensors
        tensors = []
        indices = []
        labels = []

        for i, labeled_tensor in enumerate(initial_seeds):
            t = labeled_tensor.tensor.to(self._device)
            label_val = int(labeled_tensor.label.item()) if isinstance(labeled_tensor.label, torch.Tensor) else labeled_tensor.label
            
            seed_hash = (i, self._hash_tensor(t))
            self.seen_hashes.add(seed_hash)
            
            tensors.append(t)
            indices.append(i)
            labels.append(label_val if label_val is not None else -1)
        
        N = len(tensors)
        device = self._device
        # Stack into parallel tensors on device manager's device
        self._tensors = torch.cat(tensors, dim=0)                                       # [N, ...]
        self._original_tensors = self._tensors.clone()                                  # [N, ...]
        self._original_indices = torch.tensor(indices, dtype=torch.long, device=device) # [N]
        self._labels = torch.tensor(labels, dtype=torch.long, device=device)            # [N]
        self._energies = torch.ones(N, device=device)                                   # [N]
        self._depths = torch.zeros(N, dtype=torch.long, device=device)                  # [N]
        self._ids = FuzzingSeed._next_ids(N)                                            # [N]
        self._parent_ids = torch.full((N,), -1, dtype=torch.long, device=device)        # [N]
        self._select_counts = torch.zeros(N, dtype=torch.long, device=device)           # [N]
    
    def _hash_tensor(self, tensor: torch.Tensor) -> int:
        """Compute hash of tensor for deduplication."""
        return hash(tensor.flatten().cpu().numpy().tobytes())
    
    def select(self, n: int) -> FuzzingSeed:
        """
        Select n seeds based on strategy. Returns a FuzzingSeed batch of size n.
        
        Sampling is with replacement — high-energy seeds may appear multiple times,
        which is intentional for exploitation.
        
        Args:
            n: Number of seeds to select (required).
               
        Returns:
            FuzzingSeed batch with B=n.
        """
        corpus_size = self._tensors.shape[0]
        if corpus_size == 0:
            raise ValueError("Corpus is empty!")
        
        if self.strategy == "energy":
            energies = self._energies.cpu().numpy()  # numpy requires CPU
            total = energies.sum()
            if total == 0:
                probs = np.ones(corpus_size) / corpus_size
            else:
                probs = energies / total
            indices = np.random.choice(corpus_size, size=n, p=probs, replace=True)
        else:
            indices = np.random.choice(corpus_size, size=n, replace=True)
        
        idx = torch.from_numpy(indices).long().to(self._device)
        result = FuzzingSeed(
            tensor=self._tensors[idx],
            original_tensor=self._original_tensors[idx],
            original_index=self._original_indices[idx],
            label=self._labels[idx],
            energy=self._energies[idx],
            depth=self._depths[idx],
            id=self._ids[idx],
            parent_id=self._parent_ids[idx],
            select_count=self._select_counts[idx],
        )
        self._select_counts.scatter_add_(0, idx, torch.ones_like(idx))
        return result
    
    def add(self, seeds: FuzzingSeed, mask: torch.Tensor):
        """
        Add interesting seeds from a batch to the corpus.
        
        Only seeds where mask[i] == True are added, with dedup checking.
        
        Args:
            seeds: FuzzingSeed batch (all children from one iteration)
            mask: BoolTensor[B] indicating which seeds are interesting
        """
        if not mask.any():
            return
        
        # Filter to interesting seeds
        interesting_idx = mask.nonzero(as_tuple=True)[0]
        
        # Dedup and collect indices to actually add
        keep = []
        for i in interesting_idx.tolist():
            original_index = int(seeds.original_index[i].item())
            seed_hash = (original_index, self._hash_tensor(seeds.tensor[i:i+1]))
            if seed_hash not in self.seen_hashes:
                self.seen_hashes.add(seed_hash)
                keep.append(i)
        
        if not keep:
            return
        
        idx = torch.tensor(keep, dtype=torch.long, device=self._device)
        self._tensors = torch.cat([self._tensors, seeds.tensor[idx]], dim=0)
        self._original_tensors = torch.cat([self._original_tensors, seeds.original_tensor[idx]], dim=0)
        self._original_indices = torch.cat([self._original_indices, seeds.original_index[idx]])
        self._labels = torch.cat([self._labels, seeds.label[idx]])
        self._energies = torch.cat([self._energies, seeds.energy[idx]])
        self._depths = torch.cat([self._depths, seeds.depth[idx]])
        self._ids = torch.cat([self._ids, seeds.id[idx]])
        self._parent_ids = torch.cat([self._parent_ids, seeds.parent_id[idx]])
        self._select_counts = torch.cat([self._select_counts, seeds.select_count[idx]])
    
    def __len__(self) -> int:
        """Return corpus size."""
        return self._tensors.shape[0]
    
    def __iter__(self):
        """Iterate over seeds as single-element FuzzingSeed batches."""
        for i in range(len(self)):
            yield FuzzingSeed(
                tensor=self._tensors[i:i+1],
                original_tensor=self._original_tensors[i:i+1],
                original_index=self._original_indices[i:i+1],
                label=self._labels[i:i+1],
                energy=self._energies[i:i+1],
                depth=self._depths[i:i+1],
                id=self._ids[i:i+1],
                parent_id=self._parent_ids[i:i+1],
                select_count=self._select_counts[i:i+1],
            )
    
    def get_stats(self) -> dict[str, object]:
        """Get corpus statistics."""
        n = len(self)
        if n == 0:
            return {
                "total_seeds": 0,
                "avg_energy": 0.0,
                "max_depth": 0,
            }
        
        return {
            "total_seeds": n,
            "avg_energy": float(self._energies.mean()),
            "max_energy": float(self._energies.max()),
            "max_depth": int(self._depths.max().item()),
            "strategy": self.strategy,
        }
