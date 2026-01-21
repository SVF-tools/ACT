#===- act/back_end/net_factory/utils.py - NetFactory Utility Functions ---===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

from __future__ import annotations

import hashlib
import random
from typing import Any, List, Tuple


def stable_u32_from_bytes(data: bytes) -> int:
    """Extract stable u32 from bytes."""
    return int.from_bytes(data[:4], byteorder="little", signed=False)


def derive_seed(base_seed: int, idx: int, instance_id: str) -> int:
    """Derive deterministic seed from base_seed, index, and instance_id."""
    payload = f"{base_seed}|{idx}|{instance_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return stable_u32_from_bytes(digest)


def randint_inclusive(rng: random.Random, lo_hi: List[int]) -> int:
    """Sample random int from [lo, hi] inclusive."""
    lo, hi = int(lo_hi[0]), int(lo_hi[1])
    if hi < lo:
        lo, hi = hi, lo
    return rng.randint(lo, hi)


def choose(rng: random.Random, items: List[Any], *, name: str) -> Any:
    """Randomly choose from items with error handling."""
    if not items:
        raise ValueError(f"Config.{name} must be non-empty")
    return rng.choice(list(items))


def prod(shape: Tuple[int, ...]) -> int:
    """Compute product of shape dimensions."""
    p = 1
    for s in shape:
        p *= int(s)
    return p


def ensure_batch1(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Ensure shape has batch=1 as first dimension."""
    if len(shape) < 2:
        raise ValueError(f"input_shape must include batch dim, got {shape}")
    if int(shape[0]) != 1:
        raise ValueError(f"Generator assumes batch=1, got {shape}")
    return tuple(int(x) for x in shape)


def activation_kind(name: str) -> str:
    """Map activation name to layer kind."""
    name = (name or "relu").lower()
    mapping = {"relu": "RELU", "tanh": "TANH", "sigmoid": "SIGMOID"}
    if name not in mapping:
        raise ValueError(f"Unsupported activation '{name}'")
    return mapping[name]


def infer_conv2d_output_hw(
    H: int,
    W: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> Tuple[int, int]:
    """Compute Conv2D output spatial dimensions."""
    def out_dim(x: int) -> int:
        return int((x + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1)
    return out_dim(H), out_dim(W)


def infer_pool2d_output_hw(
    H: int,
    W: int,
    kernel: int,
    stride: int,
    padding: int = 0,
) -> Tuple[int, int]:
    """Compute Pool2D output spatial dimensions."""
    def out_dim(x: int) -> int:
        return int((x + 2 * padding - (kernel - 1) - 1) // stride + 1)
    return out_dim(H), out_dim(W)


def as_block_param(v: Any, i: int, n_blocks: int, name: str) -> int:
    """Extract per-block parameter from int or tuple."""
    if isinstance(v, int):
        return int(v)
    t = tuple(int(x) for x in v)
    if len(t) == 1:
        return int(t[0])
    if len(t) == n_blocks:
        return int(t[i])
    raise ValueError(f"{name} must be int or tuple of len 1 or len {n_blocks}, got len={len(t)}")
