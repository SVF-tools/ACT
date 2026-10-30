#===- act/back_end/dual_tf/tf_forward.py - Forward Bounds ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
"""Batched DAG-aware forward bound propagation for DualTF.

Per-layer linear state lives in ``lin_state`` and frame state in ``frame_dict``.
Traversal uses Kahn's algorithm over ``net.preds`` / ``net.succs``. ADD and
CONCAT read predecessor state explicitly for fan-out / fan-in DAGs such as
ResNet skips. Returned bounds stay batch-first flattened ``[B, n]`` tensors,
with activation bounds stored PRE-activation unless ``post_activation=True``.
"""
#===---------------------------------------------------------------------===#

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportMissingParameterType=false, reportUntypedFunctionDecorator=false, reportDeprecated=false

import logging
import math
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, cast

from act.back_end.core import Bounds, Layer, Net, topological_sort
from act.back_end.layer_schema import LayerKind
from act.back_end.utils import pair_2d
from act.config.config import DualConfig
from act.front_end.specs import InputGeometry, lp_norm
from act.util.device_manager import get_default_device, get_default_dtype

log = logging.getLogger(__name__)


@dataclass
class LinearBound:
    A_lb: Optional[torch.Tensor]
    b_lb: torch.Tensor
    A_ub: Optional[torch.Tensor]
    b_ub: torch.Tensor

@dataclass(frozen=True, slots=True)
class Frame:
    """Symbolic coordinates, their current box, and optional root provenance."""

    box: Bounds
    geometry: InputGeometry | None = None
    columns: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class ForwardFrame:
    """Output affine relation over one explicit symbolic frame."""

    lid: int
    lin: LinearBound
    frame: Frame


@dataclass(frozen=True, slots=True)
class ForwardPassStats:
    """Frame provenance counts for the complete pass, including reused nodes."""

    lp_spec: bool
    layers: int
    ball_frames: int
    box_frames: int
    frame_resets: int


def input_support(a: torch.Tensor, frame: Frame) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound ``a*x`` over the root balls intersected with the current box.

    Each support problem uses a superset of that intersection. Their bounds
    may be intersected, but the child box must never redefine a root ball.
    Omitted fixed coordinates are already in the caller's affine intercept.
    """
    x_L, x_U = frame.box.lb, frame.box.ub
    positive, negative = a.clamp(min=0), a.clamp(max=0)
    lower = torch.einsum("bmi,bi->bm", positive, x_L) + torch.einsum("bmi,bi->bm", negative, x_U)
    upper = torch.einsum("bmi,bi->bm", positive, x_U) + torch.einsum("bmi,bi->bm", negative, x_L)
    geometry = frame.geometry
    if geometry is None or geometry.p == math.inf:
        return lower, upper
    center = geometry.center.to(a).flatten(1)
    radii = geometry.radii.to(a).flatten(1)
    if frame.columns is None:
        norms = lp_norm(a.reshape(*a.shape[:2], -1, geometry.embed_dim), geometry.q)
    else:
        center = center.index_select(1, frame.columns)
        # Group compressed columns by their ORIGINAL token, not by adjacency
        # in the compressed matrix. Scaling avoids overflow for p near one.
        groups = (frame.columns // geometry.embed_dim).expand_as(a)
        norms = a.new_zeros(*a.shape[:2], radii.shape[1]).scatter_reduce(
            2, groups, a.abs(), reduce="amax", include_self=False,
        )
        if geometry.q != math.inf:
            scale = norms.gather(2, groups)
            scale = torch.where(scale == 0, 1.0, scale)
            powers = torch.zeros_like(norms).scatter_add(2, groups, (a.abs() / scale).pow(geometry.q))
            norms = norms * powers.clamp_min(torch.finfo(a.dtype).tiny).pow(1 / geometry.q)
    middle = torch.einsum("bmi,bi->bm", a, center)
    radius = (norms * radii.unsqueeze(1)).sum(-1)
    return torch.maximum(lower, middle - radius), torch.minimum(upper, middle + radius)


def forward_frame_row_lower_bounds(frame: ForwardFrame, c: torch.Tensor, M: int) -> torch.Tensor:
    """Concretize packed output rows without discarding output correlations."""
    if c.dim() != 2 or M < 1 or c.shape[0] % M:
        raise ValueError("Forward objectives must have shape [B*M, outputs]")
    B = c.shape[0] // M
    if frame.lin.b_lb.shape[0] not in (1, B) or c.shape[1] != frame.lin.b_lb.shape[1]:
        raise ValueError("Forward frame batch/output shape does not match its objectives")
    rows = c.reshape(B, M, -1)
    positive, negative = rows.clamp(min=0), rows.clamp(max=0)
    lin = frame.lin
    intercept = (torch.einsum("bmo,bo->bm", positive, lin.b_lb)
                 + torch.einsum("bmo,bo->bm", negative, lin.b_ub))
    if lin.A_lb is None and lin.A_ub is None:
        coefficients = rows
    else:
        assert lin.A_lb is not None and lin.A_ub is not None
        coefficients = (torch.einsum("bmo,boi->bmi", positive, lin.A_lb)
                        + torch.einsum("bmo,boi->bmi", negative, lin.A_ub))
    return (input_support(coefficients, frame.frame)[0] + intercept).reshape(B * M)


# Above this layer input dim, ``_fwd_X`` returns ``None`` to skip the
# dense linear-bound A matrix; caller ``forward_X`` falls back to
# interval-only via ``_box_X`` + ``_reset_forward_box``. Sound (interval
# is always sound), looser bounds at the bailed layer.
_DENSE_LIN_BOUND_MAX_DIM: int = 10_000


_LIN_BOUND_FREE_FRACTION: float = 0.8

# Peak transient working set of one lane of the symbolic composition, as a
# multiple of that lane's persistent coefficient pair
# ``2 * n_out * n_sym * itemsize``. Measured as ``max_memory_allocated()``
# deltas over the three helpers on toy CUDA layers (float32, B=4): CONV2D 2.38
# (3->64 channels, VGG conv1 shape) and 3.45 (64->64), DENSE 2.25 (widening)
# and 4.02 (square), MAXPOOL2D 2.51 — each rounded up to the next integer.
# Allocations that do NOT scale with the lane count sit outside this model and
# no lane split can shrink them: ``W.abs()`` plus the einsum's weight copies
# for DENSE (they follow ``n_out * n_in``, and dominate whenever
# ``n_in >> n_out``), and the im2col buffer for spatially tiny convolutions.
_LIN_BOUND_CHUNK_TRANSIENT_CONV2D: int = 4
_LIN_BOUND_CHUNK_TRANSIENT_DENSE: int = 4
_LIN_BOUND_CHUNK_TRANSIENT_MAXPOOL2D: int = 3

_LIN_BOUND_REPORTED: set[Tuple[int, str]] = set()


def _report_lin_bound_once(layer_id: int, mode: str, level: int, msg: str, *args: object) -> None:
    """Emit ``msg`` the first time a ``(layer id, mode)`` pair degrades.

    Forward bounds are recomputed for every BaB node, so an unconditional log
    call here produces one line per node per layer.
    """
    key = (layer_id, mode)
    if key in _LIN_BOUND_REPORTED:
        return
    _LIN_BOUND_REPORTED.add(key)
    log.log(level, msg, *args)


def _lin_bound_lane_chunk(B: int, n_out: int, n_sym: int, ref: torch.Tensor,
                          k_transient: int, layer_id: int, mode: str) -> Optional[int]:
    """Plan how many lanes a symbolic ``[B, n_out, n_sym]`` layer may do at once.

    Returns ``B`` when the whole batch fits ~80% of the free CUDA memory (the
    unguarded path, bit-for-bit unchanged), a chunk size ``1 <= c < B`` when the
    batch must be split lane-wise, or ``None`` when even a single lane's working
    set overruns the budget or the concatenated result cannot fit at all — the
    caller then returns ``None`` and ``forward_X`` degrades that layer to the
    interval box. CPU allocations are never split.

    Chunking only moves the *transient* peak: the result ``[B, n_out, n_sym]``
    pair is concatenated back and stays resident either way, so it is checked
    against the full free memory separately.
    """
    if not ref.is_cuda:
        return B
    pair_bytes = 2 * n_out * n_sym * ref.element_size()
    if pair_bytes <= 0:
        return B
    free_bytes, _ = torch.cuda.mem_get_info(ref.device)
    budget = free_bytes * _LIN_BOUND_FREE_FRACTION
    persistent = B * pair_bytes
    if persistent <= budget:
        return B

    chunk = min(B, int(budget // (pair_bytes * k_transient)))
    if chunk >= 1 and persistent <= free_bytes:
        _report_lin_bound_once(
            layer_id, mode, logging.INFO,
            "forward linear bound splits %d lanes into chunks of %d for a "
            "[%d, %d, %d] coefficient pair (%.2f GiB resident, %.2f GiB CUDA free)",
            B, chunk, B, n_out, n_sym, persistent / 2 ** 30, free_bytes / 2 ** 30,
        )
        return chunk

    _report_lin_bound_once(
        layer_id, mode, logging.WARNING,
        "forward linear bound needs ~%.2f GiB for a [%d, %d, %d] coefficient pair "
        "(%.3f GiB for a single lane) with only %.2f GiB CUDA free; falling back "
        "to interval for this layer",
        persistent / 2 ** 30, B, n_out, n_sym,
        pair_bytes * k_transient / 2 ** 30, free_bytes / 2 ** 30,
    )
    return None


def _lin_lanes(lin: LinearBound, start: int, end: int) -> LinearBound:
    """View of lanes ``[start, end)`` — slices, so no copy is made."""
    return LinearBound(
        A_lb=None if lin.A_lb is None else lin.A_lb[start:end],
        b_lb=lin.b_lb[start:end],
        A_ub=None if lin.A_ub is None else lin.A_ub[start:end],
        b_ub=lin.b_ub[start:end],
    )


def _cat_lanes(parts: List[LinearBound]) -> LinearBound:
    """Reassemble lane chunks along the batch axis."""
    return LinearBound(
        A_lb=None if parts[0].A_lb is None else torch.cat(
            [cast(torch.Tensor, part.A_lb) for part in parts], dim=0),
        b_lb=torch.cat([part.b_lb for part in parts], dim=0),
        A_ub=None if parts[0].A_ub is None else torch.cat(
            [cast(torch.Tensor, part.A_ub) for part in parts], dim=0),
        b_ub=torch.cat([part.b_ub for part in parts], dim=0),
    )


def _concretize(lin: LinearBound, frame: Frame
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concretize dual-track affine bounds over their actual symbolic frame.

    Lazy identity: when ``A_lb is None`` the affine bound is the identity,
    so ``lb = x_L + b_lb`` (and analogously for ub). This avoids materializing
    a dense ``[B, n, n]`` eye matrix at the INPUT layer, which for 224×224×3
    input would request 181 GB.
    """
    x_L, x_U = frame.box.lb, frame.box.ub
    n_x = x_L.shape[1]
    if lin.A_lb is None:
        b_lb = lin.b_lb
        if b_lb.shape[1] != n_x:
            n_b = min(b_lb.shape[1], n_x)
            lb = x_L[..., :n_b] + b_lb[..., :n_b]
        else:
            lb = x_L + b_lb
    else:
        lb = input_support(lin.A_lb, frame)[0] + lin.b_lb
    if lin.A_ub is None:
        b_ub = lin.b_ub
        if b_ub.shape[1] != n_x:
            n_b = min(b_ub.shape[1], n_x)
            ub = x_U[..., :n_b] + b_ub[..., :n_b]
        else:
            ub = x_U + b_ub
    else:
        ub = input_support(lin.A_ub, frame)[1] + lin.b_ub
    return lb, ub

def _identity_lin(B: int, n: int, device, dtype) -> LinearBound:
    """Identity ``LinearBound`` over n features.

    Lazy identity: A_lb / A_ub are returned as ``None`` (sentinel for
    identity). Materializing ``torch.eye(n)`` for 224×224×3 input = 181 GB
    OOM-killed the verifier; the sentinel avoids it entirely. Downstream
    ``_fwd_X`` helpers and ``_concretize`` recognise None as identity.
    """
    zeros = torch.zeros(B, n, device=device, dtype=dtype)
    return LinearBound(A_lb=None, b_lb=zeros, A_ub=None, b_ub=zeros.clone())

def _reset_lin(lb: torch.Tensor, ub: torch.Tensor, device, dtype
               ) -> Tuple[LinearBound, torch.Tensor, torch.Tensor]:
    B, n = lb.shape[0], lb.shape[1]
    return _identity_lin(B, n, device, dtype), lb.clone(), ub.clone()

def _entry_lin_frame(lb_in: torch.Tensor, ub_in: torch.Tensor,
                     max_perturbed: int, device, dtype,
                     geometry: InputGeometry | None = None,
                     ) -> Tuple[LinearBound, Frame]:
    """Entry LinearBound/Frame, symbolic only in the perturbed input dims.

    ``pert`` is the union over the batch of dims with ``ub > lb``. When
    ``0 < n_pert <= max_perturbed``, the entry bound is the explicit one-hot
    selection ``A[:, i, k] = 1`` iff input dim ``i`` is the ``k``-th perturbed
    dim; unperturbed dims fold into the constant term ``b`` and the frame
    shrinks to the perturbed columns. Downstream ``_fwd_X`` handlers then see
    ``A is not None`` and never take the ``_DENSE_LIN_BOUND_MAX_DIM`` bail,
    so e.g. VGG16 with <=100 perturbed pixels keeps linear bounds at every
    conv. Otherwise (``n_pert == 0`` or above the cap) fall back to the lazy
    identity over all dims, preserving the previous behaviour exactly.
    """
    B, input_dim = lb_in.shape
    pert = (ub_in > lb_in).any(dim=0)
    n_pert = int(pert.sum().item())
    if n_pert == 0 or n_pert > max_perturbed:
        return _identity_lin(B, input_dim, device, dtype), Frame(Bounds(lb_in, ub_in), geometry)
    selection = torch.zeros(input_dim, n_pert, device=device, dtype=dtype)
    selection[pert.nonzero(as_tuple=True)[0], torch.arange(n_pert, device=device)] = 1.0
    A = selection.unsqueeze(0).expand(B, input_dim, n_pert)
    b = torch.where(pert, torch.zeros_like(lb_in), lb_in)
    lin = LinearBound(A_lb=A, b_lb=b, A_ub=A, b_ub=b.clone())
    return lin, Frame(Bounds(lb_in[:, pert], ub_in[:, pert]), geometry, pert.nonzero(as_tuple=True)[0])

def _match_lin_input_dim(lin: LinearBound, n_in: int) -> LinearBound:
    """Pad or truncate the current output-feature axis to size n_in.

    Invariant: ``A_lb`` and ``A_ub`` are paired — either both ``None``
    (identity sentinel) or both ``Tensor``. We check both for explicit
    type narrowing.
    """
    if lin.A_lb is None or lin.A_ub is None:
        curr_out = lin.b_lb.shape[1]
    else:
        curr_out = lin.A_lb.shape[1]
    if curr_out == n_in:
        return lin

    B = lin.b_lb.shape[0]
    if lin.A_lb is None or lin.A_ub is None:
        if curr_out < n_in:
            pad = n_in - curr_out
            zeros_b = torch.zeros(B, pad, device=lin.b_lb.device, dtype=lin.b_lb.dtype)
            return LinearBound(
                A_lb=None,
                b_lb=torch.cat([lin.b_lb, zeros_b], dim=1),
                A_ub=None,
                b_ub=torch.cat([lin.b_ub, zeros_b.clone()], dim=1),
            )
        return LinearBound(
            A_lb=None,
            b_lb=lin.b_lb[:, :n_in],
            A_ub=None,
            b_ub=lin.b_ub[:, :n_in],
        )

    input_dim = lin.A_lb.shape[2]
    if curr_out < n_in:
        pad = n_in - curr_out
        zeros_A = torch.zeros(B, pad, input_dim, device=lin.A_lb.device, dtype=lin.A_lb.dtype)
        zeros_b = torch.zeros(B, pad, device=lin.b_lb.device, dtype=lin.b_lb.dtype)
        return LinearBound(
            A_lb=torch.cat([lin.A_lb, zeros_A], dim=1),
            b_lb=torch.cat([lin.b_lb, zeros_b], dim=1),
            A_ub=torch.cat([lin.A_ub, zeros_A.clone()], dim=1),
            b_ub=torch.cat([lin.b_ub, zeros_b.clone()], dim=1),
        )

    return LinearBound(
        A_lb=lin.A_lb[:, :n_in, :],
        b_lb=lin.b_lb[:, :n_in],
        A_ub=lin.A_ub[:, :n_in, :],
        b_ub=lin.b_ub[:, :n_in],
    )

def _intersect_boxes(lb_a: torch.Tensor, ub_a: torch.Tensor,
                     lb_b: torch.Tensor, ub_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Intersect two sound boxes; EVERY crossing degrades to box ``b`` per entry.

    Both boxes enclose the same true value set, so in exact arithmetic they
    overlap and ``max(lbs) <= min(ubs)``. In float32 the linear-track
    concretization (box ``a``) and the interval track (box ``b``) are
    independent accumulations; when the true width is below the accumulation
    noise the pair can cross (RC2) — on layers whose partial sums dwarf the
    cancelled result (late VGG convs / FC, partials 1e5-1e7 vs results
    O(1-100)) by a small multiple of the PARTIAL-sum ulp. Any crossed entry,
    whatever its magnitude, falls back to box ``b`` verbatim (the
    centre-radius interval track, ordered by construction), which is sound
    and merely looser. A tolerance-gated "repair" that swaps the crossed pair
    to ``[min(ubs), max(lbs)]`` is UNSOUND: that interval lies between the two
    boxes and may contain points of neither — e.g. true range [1.0, 2.0],
    interval box [1.0001, 2.0001], linear box [0.0, 1.0] (1 ulp low at 1e5
    partial sums) swapped to [1.0, 1.0001], excluding the true value 2.0.
    Genuine producer bugs that emit inverted boxes directly remain visible to
    the downstream degenerate-interval check.
    """
    lb = torch.maximum(lb_a, lb_b)
    ub = torch.minimum(ub_a, ub_b)
    crossed = lb > ub
    out_lb = torch.where(crossed, lb_b, lb)
    out_ub = torch.where(crossed, ub_b, ub)
    return out_lb, out_ub

def _align_batch(a: torch.Tensor, n: int) -> torch.Tensor:
    if a.shape[1] == n:
        return a
    if a.shape[1] > n:
        return a[:, :n]
    repeats = (n + a.shape[1] - 1) // a.shape[1]
    return a.repeat(1, repeats)[:, :n]

def _shape_list(shape_param: object) -> Optional[list[int]]:
    return [int(v) for v in shape_param] if isinstance(shape_param, (tuple, list)) else None

def _int_param(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    return value if isinstance(value, int) else default


def _box_dense(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interval DENSE as ``m = c Wᵀ + b``, ``rho = r |W|ᵀ``, ``[m-rho, m+rho]``.

    ``rho >= 0`` structurally, so ``ub - lb = 2*rho >= 0`` per neuron whatever
    the rounding; the ``W⁺l + W⁻u`` / ``W⁺u + W⁻l`` form sums two independent
    accumulations that float32 cancellation can order the wrong way.
    """
    W = layer.params["weight"]
    b = layer.params.get("bias")
    if layer.params.get("token_wise", False):
        if lb.shape[1] % W.shape[1]:
            raise ValueError(f"token-wise DENSE {layer.id}: incompatible input width")
        shape = lb.shape
        lb, ub = lb.reshape(-1, W.shape[1]), ub.reshape(-1, W.shape[1])
    else:
        shape = None
    lb = _align_batch(lb, W.shape[1])
    ub = _align_batch(ub, W.shape[1])
    centre = (lb + ub) * 0.5
    radius = (ub - lb) * 0.5
    mid = centre @ W.T
    spread = radius @ W.abs().T
    if b is not None:
        mid = mid + _align(b.flatten(), W.shape[0])
    lower, upper = mid - spread, mid + spread
    if shape is not None:
        lower, upper = lower.reshape(shape[0], -1), upper.reshape(shape[0], -1)
    return lower, upper


def _box_bias(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    c = _align(layer.params["c"], lb.shape[1])
    return lb + c, ub + c


def _box_scale(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    a = _align(layer.params["a"], lb.shape[1])
    out_lb = torch.where(a >= 0, a * lb, a * ub)
    out_ub = torch.where(a >= 0, a * ub, a * lb)
    return out_lb, out_ub


def _box_bn(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    A_bn = _align(layer.params["A"], lb.shape[1])
    c = _align(layer.params["c"], lb.shape[1])
    out_lb = torch.where(A_bn >= 0, A_bn * lb + c, A_bn * ub + c)
    out_ub = torch.where(A_bn >= 0, A_bn * ub + c, A_bn * lb + c)
    return out_lb, out_ub


def _box_relu(lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact interval propagation for ReLU."""
    return lb.clamp(min=0), ub.clamp(min=0)


def _box_lrelu(lb: torch.Tensor, ub: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact interval propagation for LeakyReLU."""
    alpha_tensor = torch.full_like(lb, alpha)
    out_lb = torch.where(lb >= 0, lb, alpha_tensor * lb)
    out_ub = torch.where(ub <= 0, alpha_tensor * ub, ub)
    return out_lb, out_ub


def _store_forward_state(bounds_dict: Dict[int, Bounds],
                          box_state: Dict[int, Bounds],
                          lin_state: Dict[int, LinearBound],
                          frame_dict: Dict[int, Frame],
                          layer_id: int,
                          stored: Bounds,
                          out_box: Bounds,
                          lin: LinearBound,
                          frame: Frame) -> None:
    """Store public bounds plus internal forward state for a layer."""
    bounds_dict[layer_id] = stored.copy()
    box_state[layer_id] = out_box.copy()
    lin_state[layer_id] = lin
    frame_dict[layer_id] = frame


def _reset_forward_box(lb: torch.Tensor, ub: torch.Tensor, device, dtype
                       ) -> Tuple[LinearBound, Frame]:
    """Reset dual-track state on a concrete box."""
    lin, x_L, x_U = _reset_lin(lb, ub, device, dtype)
    return lin, Frame(Bounds(x_L, x_U))


def _sum_interval_bounds(boxes: List[Bounds]) -> Bounds:
    """Sum predecessor boxes element-wise, trimming to the smallest width."""
    lbs = [box.lb.flatten(start_dim=1) for box in boxes]
    ubs = [box.ub.flatten(start_dim=1) for box in boxes]
    n = min(lb.shape[1] for lb in lbs)
    lbs = [lb[:, :n] for lb in lbs]
    ubs = [ub[:, :n] for ub in ubs]
    return Bounds(sum(lbs[1:], lbs[0]), sum(ubs[1:], ubs[0]))


def _sum_linear_bounds(lins: List[LinearBound]) -> LinearBound:
    """Sum dual-track affine bounds from multiple predecessors.

    ``A = None`` denotes one identity, never a sum of identities. Materialize
    it before addition so repeated branches keep their multiplicity. The ADD
    handler uses interval summation when every predecessor is lazy, avoiding
    a quadratic allocation for wide reset frames.
    """
    A_lbs = [lin.A_lb for lin in lins]
    A_ubs = [lin.A_ub for lin in lins]

    def _materialize(A: Optional[torch.Tensor], example: torch.Tensor) -> torch.Tensor:
        if A is not None:
            return A
        B, n = example.shape
        eye = torch.eye(n, device=example.device, dtype=example.dtype)
        return eye.unsqueeze(0).expand(B, n, n)

    A_lb_sum = sum((_materialize(A, lins[i].b_lb) for i, A in enumerate(A_lbs[1:], 1)),
                   _materialize(A_lbs[0], lins[0].b_lb))
    A_ub_sum = sum((_materialize(A, lins[i].b_ub) for i, A in enumerate(A_ubs[1:], 1)),
                   _materialize(A_ubs[0], lins[0].b_ub))

    return LinearBound(
        A_lb=A_lb_sum,
        b_lb=sum((lin.b_lb for lin in lins[1:]), lins[0].b_lb),
        A_ub=A_ub_sum,
        b_ub=sum((lin.b_ub for lin in lins[1:]), lins[0].b_ub),
    )


_MAXPOOL_CACHE_KEYS = ("maxpool_argmax_flat", "maxpool_dominant", "maxpool_lb", "maxpool_ub")


class ForwardPrefixCache:
    """Own alpha-independent state for exactly one optimization call.

    Filled once by the ordinary traversal; reuse skips alpha-independent nodes
    and reads their private state without recloning every affine matrix.
    Public outputs are copied so callers cannot modify cached state.
    """

    def __init__(self, net: Net, input_lb: torch.Tensor, input_ub: torch.Tensor, *,
                 alpha_relu_ids: set[int],
                 post_activation: bool = False,
                 forward_lin_max_perturbed: int | None = None) -> None:
        """Capture the immutable domain and propagation options."""
        self._net = net
        self._inputs = (input_lb.detach().clone(), input_ub.detach().clone())
        self._strides = (input_lb.stride(), input_ub.stride())
        geometry = net.input_geometry
        self._geometry = geometry.snapshot() if geometry is not None else None
        self._layout = tuple((layer.id, layer.kind, tuple(net.preds.get(layer.id, ())))
                             for layer in net.layers)
        self._alpha_ids = frozenset(alpha_relu_ids)
        if any(lid not in net.by_id or net.by_id[lid].kind != LayerKind.RELU.value
               for lid in self._alpha_ids):
            raise ValueError("ForwardPrefixCache alpha ids must name ReLU layers")
        self._options = (post_activation, forward_lin_max_perturbed if forward_lin_max_perturbed is not None
                         else DualConfig().forward_lin_max_perturbed,
                         get_default_device(), get_default_dtype())
        independent: set[int] = set()
        for lid in topological_sort(net):
            if lid not in self._alpha_ids and all(pid in independent for pid in net.preds.get(lid, ())):
                independent.add(lid)
        self._independent = independent
        self._bounds: dict[int, Bounds] = {}
        self._boxes: dict[int, Bounds] = {}
        self._lins: dict[int, LinearBound] = {}
        self._frames: dict[int, Frame] = {}
        self._maxpool: dict[int, dict[str, torch.Tensor]] = {}

    def _validate(self, net: Net, inputs: tuple[torch.Tensor, torch.Tensor],
                  options: tuple[bool, int, torch.device, torch.dtype],
                  alpha_ids: frozenset[int]) -> None:
        """Reject a different domain, lane layout, graph or relaxation setup."""
        layout = tuple((layer.id, layer.kind, tuple(net.preds.get(layer.id, ())))
                       for layer in net.layers)
        if net is not self._net or layout != self._layout or options != self._options or alpha_ids != self._alpha_ids:
            raise ValueError("ForwardPrefixCache: graph, alpha layers or forward options changed")
        if tuple(t.stride() for t in inputs) != self._strides:
            raise ValueError("ForwardPrefixCache: input lane layout changed")
        pairs = list(zip(inputs, self._inputs))
        geometry = net.input_geometry
        saved = self._geometry
        if (geometry is None) != (saved is None):
            raise ValueError("ForwardPrefixCache: input geometry changed")
        if geometry is not None and saved is not None:
            if geometry.p != saved.p:
                raise ValueError("ForwardPrefixCache: input norm changed")
            pairs.extend(((geometry.center, saved.center), (geometry.eps, saved.eps), (geometry.mask, saved.mask)))
        if any(a.shape != b.shape or a.dtype != b.dtype or a.device != b.device
               or not torch.equal(a, b) for a, b in pairs):
            raise ValueError("ForwardPrefixCache: input box or root geometry changed")


@torch.no_grad()
def compute_forward_bounds(net: Net, input_lb: torch.Tensor, input_ub: torch.Tensor,
                           post_activation: bool = False,
                           alphas: Optional[Dict[int, torch.Tensor]] = None,
                           forward_lin_max_perturbed: Optional[int] = None,
                           prefix_cache: ForwardPrefixCache | None = None,
                           ) -> tuple[Dict[int, Bounds], ForwardFrame | None, ForwardPassStats]:
    """Forward bounds, natively batched with singleton auto-promotion.

    ``forward_lin_max_perturbed`` caps the number of perturbed input dims for
    which the linear track stays symbolic (see :func:`_entry_lin_frame`);
    ``None`` resolves to the DualConfig default at call time rather than being
    frozen at import time.
    """
    # Lazy import to break circular dep (dual_tf imports compute_forward_bounds)
    from .dual_tf import DualTF

    if forward_lin_max_perturbed is None:
        forward_lin_max_perturbed = DualConfig().forward_lin_max_perturbed

    device, dtype = get_default_device(), get_default_dtype()
    if prefix_cache is not None:
        prefix_cache._validate(
            net, (input_lb, input_ub),
            (post_activation, forward_lin_max_perturbed, device, dtype),
            frozenset(alphas or ()),
        )
    if (
        input_lb.dtype != dtype or input_lb.device != device
        or input_ub.dtype != dtype or input_ub.device != device
    ):
        input_lb = input_lb.to(device=device, dtype=dtype)
        input_ub = input_ub.to(device=device, dtype=dtype)

    if input_lb.dim() < 2:
        input_lb = input_lb.unsqueeze(0)
        input_ub = input_ub.unsqueeze(0)

    B = input_lb.shape[0]
    lb_in = input_lb.reshape(B, -1)
    ub_in = input_ub.reshape(B, -1)

    bounds_dict: Dict[int, Bounds] = {}
    box_state: Dict[int, Bounds] = {}
    lin_state: Dict[int, LinearBound] = {}
    frame_dict: Dict[int, Frame] = {}
    topo_order = topological_sort(net)
    by_id = getattr(net, "by_id", {layer.id: layer for layer in net.layers})
    entry_box = Bounds(lb_in, ub_in)
    entry_lin = entry_frame = None
    if prefix_cache is None or not prefix_cache._bounds:
        geometry = net.input_geometry
        entry_lin, entry_frame = _entry_lin_frame(
            lb_in, ub_in, forward_lin_max_perturbed, device, dtype,
            geometry.to(lb_in) if geometry is not None else None,
        )
    if prefix_cache is not None and prefix_cache._bounds:
        bounds_dict = {lid: bounds.copy() for lid, bounds in prefix_cache._bounds.items()}
        box_state, lin_state, frame_dict = (
            dict(prefix_cache._boxes), dict(prefix_cache._lins), dict(prefix_cache._frames),
        )
        for lid, metadata in prefix_cache._maxpool.items():
            for key in _MAXPOOL_CACHE_KEYS:
                by_id[lid].cache.pop(key, None)
            by_id[lid].cache.update(deepcopy(metadata))

    for lid in topo_order:
        if lid in bounds_dict:
            continue
        layer = by_id[lid]
        lid = layer.id
        kind = layer.kind.upper()
        preds = list(net.preds.get(lid, []) or [])

        if not preds:
            assert entry_lin is not None and entry_frame is not None, "Cached source state is missing"
            if kind == LayerKind.CONSTANT.value:
                # CONSTANT is a data-independent source; materialize its value
                # broadcast to batch B (taken from entry_box), no input frame.
                handler = DualTF._FORWARD_REGISTRY[kind]
                stored, out, lin, frame = handler(
                    layer, [entry_box], [], [], [], post_activation, device, dtype,
                )
                _store_forward_state(
                    bounds_dict, box_state, lin_state, frame_dict,
                    lid, stored, out, lin, frame,
                )
                continue
            if kind not in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value):
                raise ValueError(f"compute_forward_bounds: layer {lid} kind '{kind}' has no predecessors and is not INPUT / INPUT_SPEC")
            _store_forward_state(
                bounds_dict,
                box_state,
                lin_state,
                frame_dict,
                lid,
                entry_box,
                entry_box,
                entry_lin,
                entry_frame,
            )
            continue

        missing = [pid for pid in preds if pid not in box_state or pid not in lin_state or pid not in frame_dict]
        if missing:
            raise ValueError(f"compute_forward_bounds: layer {lid} missing predecessor state for {missing}")

        if len(preds) >= 2 or kind in (LayerKind.ADD.value, LayerKind.CONCAT.value):
            handler = DualTF._FORWARD_REGISTRY.get(kind)
            if handler is None:
                raise ValueError(
                    f"compute_forward_bounds: unknown multi-pred layer kind '{kind}' "
                    f"at layer {lid}. Registered kinds: "
                    f"{sorted(DualTF._FORWARD_REGISTRY.keys())}"
                )
            pred_boxes  = [box_state[pid]   for pid in preds]
            pred_lins   = [lin_state[pid]   for pid in preds]
            pred_frames = [frame_dict[pid]  for pid in preds]
            stored, out, lin, frame = handler(
                layer, pred_boxes, pred_lins, pred_frames, preds,
                post_activation, device, dtype,
            )
            _store_forward_state(bounds_dict, box_state, lin_state, frame_dict,
                                 lid, stored, out, lin, frame)
            continue

        if kind == LayerKind.RELU.value and alphas is not None:
            parent_box = box_state[preds[0]]
            parent_lin = lin_state[preds[0]]
            parent_frame = frame_dict[preds[0]]
            pre_lb, pre_ub = parent_box.lb, parent_box.ub
            alpha = alphas.get(lid)
            new_lin = _fwd_relu(parent_lin, pre_lb, pre_ub, alpha=alpha)
            int_lb, int_ub = _box_relu(pre_lb, pre_ub)
            if new_lin is None:
                lb, ub = int_lb, int_ub
                out = Bounds(lb, ub)
                stored = out if post_activation else Bounds(pre_lb, pre_ub)
                lin, frame = _reset_forward_box(lb, ub, device, dtype)
            else:
                lin = new_lin
                lin_lb, lin_ub = _concretize(lin, parent_frame)
                lb, ub = _intersect_boxes(lin_lb, lin_ub, int_lb, int_ub)
                out = Bounds(lb, ub)
                stored = out if post_activation else Bounds(pre_lb, pre_ub)
                frame = parent_frame
                if post_activation:
                    lin, frame = _reset_forward_box(lb, ub, device, dtype)
            _store_forward_state(bounds_dict, box_state, lin_state, frame_dict,
                                 lid, stored, out, lin, frame)
            continue

        handler = DualTF._FORWARD_REGISTRY.get(kind)
        if handler is None:
            raise ValueError(
                f"compute_forward_bounds: unknown layer kind '{kind}' at layer {lid}. "
                f"Registered kinds: {sorted(DualTF._FORWARD_REGISTRY.keys())}"
            )
        pred_boxes  = [box_state[preds[0]]]
        pred_lins   = [lin_state[preds[0]]]
        pred_frames = [frame_dict[preds[0]]]
        stored, out, lin, frame = handler(
            layer, pred_boxes, pred_lins, pred_frames, preds,
            post_activation, device, dtype,
        )
        _store_forward_state(bounds_dict, box_state, lin_state, frame_dict,
                             lid, stored, out, lin, frame)

    if prefix_cache is not None and not prefix_cache._bounds:
        stable = prefix_cache._independent
        (prefix_cache._bounds, prefix_cache._boxes,
         prefix_cache._lins, prefix_cache._frames) = deepcopy((
            {lid: value for lid, value in bounds_dict.items() if lid in stable},
            {lid: value for lid, value in box_state.items() if lid in stable},
            {lid: value for lid, value in lin_state.items() if lid in stable},
            {lid: value for lid, value in frame_dict.items() if lid in stable},
        ), {id(net.input_geometry): prefix_cache._geometry} if net.input_geometry is not None else None)
        prefix_cache._maxpool = {
            lid: {key: by_id[lid].cache[key].detach().clone()
                  for key in _MAXPOOL_CACHE_KEYS
                  if key in by_id[lid].cache}
            for lid in stable if by_id[lid].kind == LayerKind.MAXPOOL2D.value
        }
    output_frame = None
    output_preds = net.preds.get(net.layers[-1].id, ())
    if len(output_preds) == 1:
        output_id = output_preds[0]
        output_frame = ForwardFrame(output_id, lin_state[output_id], frame_dict[output_id])
        if prefix_cache is not None:
            geometry = output_frame.frame.geometry
            output_frame = deepcopy(
                output_frame, {id(geometry): geometry.snapshot()} if geometry is not None else None,
            )
    ball_frames = sum(frame.geometry is not None and frame.geometry.p != math.inf
                      for frame in frame_dict.values())
    stats = ForwardPassStats(
        lp_spec=net.input_geometry is not None, layers=len(bounds_dict),
        ball_frames=ball_frames, box_frames=len(frame_dict) - ball_frames,
        frame_resets=sum(bool(net.preds.get(lid)) and all(
            frame is not frame_dict[pid] for pid in net.preds[lid]
        ) for lid, frame in frame_dict.items()),
    )
    return bounds_dict, output_frame, stats


def _fwd_dense(layer: Layer, lin: LinearBound) -> Optional[LinearBound]:
    """Compose dual-track affine bounds through a dense layer.

    Lazy identity: if input ``A_lb`` is None (identity), output A would
    be ``W`` broadcast to ``[B, out_c, in_c]``. For wide DENSE layers
    (e.g. simple_cnn FC1 with in_c=200704), this allocates several GB
    transiently (W_broadcast_lb, W_broadcast_ub, W_pos, W_neg). When
    ``in_c > _DENSE_LIN_BOUND_MAX_DIM``, we return ``None`` instead, signalling
    the caller to fall back to interval-only forward via
    :func:`_reset_forward_box`. Same pattern as :func:`_fwd_conv2d` and
    :func:`_fwd_relu`.

    Under CUDA memory pressure the batch is composed in lane chunks (see
    :func:`_lin_bound_lane_chunk`) and reassembled; interval fallback is kept
    only for the case where a single lane does not fit.
    """
    W = layer.params["weight"]
    lin = _match_lin_input_dim(lin, W.shape[1])
    B = lin.b_lb.shape[0]
    n_sym = W.shape[1] if lin.A_lb is None else lin.A_lb.shape[2]
    chunk = _lin_bound_lane_chunk(B, W.shape[0], n_sym, lin.b_lb,
                                  _LIN_BOUND_CHUNK_TRANSIENT_DENSE, layer.id, "dense")
    if chunk is None:
        return None
    if chunk >= B:
        return _fwd_dense_lanes(layer, lin)

    parts: List[LinearBound] = []
    for start in range(0, B, chunk):
        part = _fwd_dense_lanes(layer, _lin_lanes(lin, start, min(start + chunk, B)))
        if part is None:
            return None
        parts.append(part)
    return _cat_lanes(parts)


def _fwd_dense_lanes(layer: Layer, lin: LinearBound) -> Optional[LinearBound]:
    """Dense composition for one lane chunk; ``lin`` must already be dim-matched."""
    W = layer.params["weight"]
    b = layer.params.get("bias")
    B = lin.b_lb.shape[0]
    bias_vec = torch.zeros(W.shape[0], device=lin.b_lb.device, dtype=lin.b_lb.dtype)
    if b is not None:
        bias_vec = _align(b.flatten(), W.shape[0])

    # Centre-radius composition; see _fwd_conv2d for the identity and the
    # float32 cancellation rationale.
    b_centre = (lin.b_lb + lin.b_ub) * 0.5
    b_radius = (lin.b_ub - lin.b_lb) * 0.5
    b_mid = torch.einsum("oc,bc->bo", W, b_centre) + bias_vec
    b_spread = torch.einsum("oc,bc->bo", W.abs(), b_radius)

    if lin.A_lb is None and lin.A_ub is None:
        if W.shape[1] > _DENSE_LIN_BOUND_MAX_DIM:
            return None
        W_broadcast_lb = W.unsqueeze(0).expand(B, -1, -1).contiguous()
        W_broadcast_ub = W.unsqueeze(0).expand(B, -1, -1).contiguous()
        return LinearBound(A_lb=W_broadcast_lb, b_lb=b_mid - b_spread,
                           A_ub=W_broadcast_ub, b_ub=b_mid + b_spread)

    A_centre = (lin.A_lb + lin.A_ub) * 0.5
    A_radius = (lin.A_ub - lin.A_lb) * 0.5
    A_mid = torch.einsum("oc,bci->boi", W, A_centre)
    A_spread = torch.einsum("oc,bci->boi", W.abs(), A_radius)
    return LinearBound(
        A_lb=A_mid - A_spread,
        b_lb=b_mid - b_spread,
        A_ub=A_mid + A_spread,
        b_ub=b_mid + b_spread,
    )


def _fwd_relu(lin: LinearBound, lb: torch.Tensor, ub: torch.Tensor,
              alpha: Optional[torch.Tensor] = None
              ) -> Optional[LinearBound]:
    """Apply forward ReLU linear relaxation with per-batch alpha choice.

    Lazy identity: if input A is None (identity), output A becomes a
    diagonal scaling matrix (alpha or up_slope on the diagonal). For high-dim
    inputs this would materialize a sparse-on-diagonal dense tensor — we
    return ``None`` instead, signalling the caller to fall back to
    interval-only forward via :func:`_reset_forward_box`. ReLU after INPUT
    is rare in practice (networks start with DENSE/CONV); the common case
    (RELU after DENSE/CONV) hits the second branch with a materialized A_lb.
    """
    on = lb >= 0
    off = ub <= 0
    amb = ~(on | off)
    denom = (ub - lb).clamp(min=1e-12)
    up_slope = torch.where(
        amb,
        ub / denom,
        torch.where(on, torch.ones_like(lb), torch.zeros_like(lb)),
    )
    up_inter = torch.where(amb, -up_slope * lb, torch.zeros_like(lb))
    if alpha is None:
        low_slope = (ub > -lb).to(lb.dtype)
    else:
        low_slope = alpha.to(device=lb.device, dtype=lb.dtype)
    low_slope = torch.where(on, torch.ones_like(low_slope), low_slope)
    low_slope = torch.where(off, torch.zeros_like(low_slope), low_slope)

    if lin.A_lb is None or lin.A_ub is None:
        B, n = lb.shape
        if n > _DENSE_LIN_BOUND_MAX_DIM:
            return None
        eye = torch.eye(n, device=lb.device, dtype=lb.dtype).unsqueeze(0).expand(B, n, n)
        A_lb_in = lin.A_lb if lin.A_lb is not None else eye
        A_ub_in = lin.A_ub if lin.A_ub is not None else eye
        return LinearBound(
            A_lb=low_slope.unsqueeze(-1) * A_lb_in,
            b_lb=low_slope * lin.b_lb,
            A_ub=up_slope.unsqueeze(-1) * A_ub_in,
            b_ub=up_slope * lin.b_ub + up_inter,
        )

    return LinearBound(
        A_lb=low_slope.unsqueeze(-1) * lin.A_lb,
        b_lb=low_slope * lin.b_lb,
        A_ub=up_slope.unsqueeze(-1) * lin.A_ub,
        b_ub=up_slope * lin.b_ub + up_inter,
    )


def _shared_explicit_frame(
    lins: List[LinearBound], frames: List[Frame], widths: List[int],
) -> bool:
    """Whether operand relations use the same explicit symbolic coordinates."""
    return all(frame is frames[0] for frame in frames[1:]) and all(
        lin.A_lb is not None and lin.A_ub is not None
        and lin.A_lb.shape[1] == width and lin.b_lb.shape[1] == width
        for lin, width in zip(lins, widths, strict=True)
    )


def _fwd_elementwise_planes(
    lin: LinearBound,
    planes: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> Optional[LinearBound]:
    """Compose signed elementwise envelopes over an existing explicit frame."""
    if lin.A_lb is None or lin.A_ub is None:
        return None
    k_lo, b_lo, k_hi, b_hi = planes
    if lin.A_lb.shape[1] != k_lo.shape[1] or lin.b_lb.shape[1] != k_lo.shape[1]:
        return None
    lp, ln = k_lo.clamp(min=0), k_lo.clamp(max=0)
    up, un = k_hi.clamp(min=0), k_hi.clamp(max=0)
    return LinearBound(
        A_lb=lp.unsqueeze(-1) * lin.A_lb + ln.unsqueeze(-1) * lin.A_ub,
        b_lb=lp * lin.b_lb + ln * lin.b_ub + b_lo,
        A_ub=up.unsqueeze(-1) * lin.A_ub + un.unsqueeze(-1) * lin.A_lb,
        b_ub=up * lin.b_ub + un * lin.b_lb + b_hi,
    )


def _fwd_rowwise_planes(
    lin: LinearBound, row: int,
    planes: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> Optional[LinearBound]:
    """Compose independent row maps, preserving the original symbolic axis."""
    if lin.A_lb is None or lin.A_ub is None:
        return None
    B, n, symbols = lin.A_lb.shape
    a_lo, c_lo, a_hi, c_hi = planes
    if n % row or a_lo.shape[0] != B * (n // row):
        return None
    rows, outputs = n // row, a_lo.shape[-2]
    A_lb, A_ub = (a.reshape(B, rows, row, symbols) for a in (lin.A_lb, lin.A_ub))
    b_lb, b_ub = (b.reshape(B, rows, row) for b in (lin.b_lb, lin.b_ub))
    lo, hi = (a.reshape(B, rows, outputs, row) for a in (a_lo, a_hi))
    lp, ln = lo.clamp(min=0), lo.clamp(max=0)
    up, un = hi.clamp(min=0), hi.clamp(max=0)
    lower_a = torch.einsum("brij,brjs->bris", lp, A_lb) + torch.einsum("brij,brjs->bris", ln, A_ub)
    upper_a = torch.einsum("brij,brjs->bris", up, A_ub) + torch.einsum("brij,brjs->bris", un, A_lb)
    lower_b = torch.einsum("brij,brj->bri", lp, b_lb) + torch.einsum("brij,brj->bri", ln, b_ub)
    upper_b = torch.einsum("brij,brj->bri", up, b_ub) + torch.einsum("brij,brj->bri", un, b_lb)
    return LinearBound(
        A_lb=lower_a.reshape(B, rows * outputs, symbols),
        b_lb=(lower_b + c_lo.reshape(B, rows, outputs)).reshape(B, rows * outputs),
        A_ub=upper_a.reshape(B, rows * outputs, symbols),
        b_ub=(upper_b + c_hi.reshape(B, rows, outputs)).reshape(B, rows * outputs),
    )


def _fwd_bias(layer: Layer, lin: LinearBound) -> LinearBound:
    """Compose dual-track affine bounds through a bias layer.

    Bias is purely additive on b, so A_lb / A_ub pass through unchanged —
    including the None (identity) sentinel.
    """
    c = _align(layer.params["c"], lin.b_lb.shape[1])
    return LinearBound(
        A_lb=lin.A_lb,
        b_lb=lin.b_lb + c,
        A_ub=lin.A_ub,
        b_ub=lin.b_ub + c,
    )


def _fwd_scale(layer: Layer, lin: LinearBound) -> Optional[LinearBound]:
    """Compose dual-track affine bounds through an element-wise scale.

    Lazy identity: SCALE following INPUT is rare; if input A is None,
    materialize a diagonal eye·a representation. Common case (SCALE after
    DENSE/CONV) hits the second branch with materialized A_lb.
    """
    a = _align(layer.params["a"], lin.b_lb.shape[1])
    a_pos = a.clamp(min=0)
    a_neg = a.clamp(max=0)
    a_pos_A = a_pos.view(1, -1, 1)
    a_neg_A = a_neg.view(1, -1, 1)

    if lin.A_lb is None or lin.A_ub is None:
        B, n = lin.b_lb.shape
        if n > _DENSE_LIN_BOUND_MAX_DIM:
            return None
        eye = torch.eye(n, device=lin.b_lb.device, dtype=lin.b_lb.dtype).unsqueeze(0).expand(B, n, n)
        A_lb_in = lin.A_lb if lin.A_lb is not None else eye
        A_ub_in = lin.A_ub if lin.A_ub is not None else eye
        return LinearBound(
            A_lb=a_pos_A * A_lb_in + a_neg_A * A_ub_in,
            b_lb=a_pos * lin.b_lb + a_neg * lin.b_ub,
            A_ub=a_pos_A * A_ub_in + a_neg_A * A_lb_in,
            b_ub=a_pos * lin.b_ub + a_neg * lin.b_lb,
        )
    return LinearBound(
        A_lb=a_pos_A * lin.A_lb + a_neg_A * lin.A_ub,
        b_lb=a_pos * lin.b_lb + a_neg * lin.b_ub,
        A_ub=a_pos_A * lin.A_ub + a_neg_A * lin.A_lb,
        b_ub=a_pos * lin.b_ub + a_neg * lin.b_lb,
    )


def _fwd_bn(layer: Layer, lin: LinearBound) -> Optional[LinearBound]:
    """Compose dual-track affine bounds through batch normalization.

    Lazy identity: BN following INPUT (rare) materializes diagonal.
    """
    A_bn = _align(layer.params["A"], lin.b_lb.shape[1])
    c = _align(layer.params["c"], lin.b_lb.shape[1])
    A_pos = A_bn.clamp(min=0)
    A_neg = A_bn.clamp(max=0)
    A_pos_A = A_pos.view(1, -1, 1)
    A_neg_A = A_neg.view(1, -1, 1)

    if lin.A_lb is None or lin.A_ub is None:
        B, n = lin.b_lb.shape
        if n > _DENSE_LIN_BOUND_MAX_DIM:
            return None
        eye = torch.eye(n, device=lin.b_lb.device, dtype=lin.b_lb.dtype).unsqueeze(0).expand(B, n, n)
        A_lb_in = lin.A_lb if lin.A_lb is not None else eye
        A_ub_in = lin.A_ub if lin.A_ub is not None else eye
        return LinearBound(
            A_lb=A_pos_A * A_lb_in + A_neg_A * A_ub_in,
            b_lb=A_pos * lin.b_lb + A_neg * lin.b_ub + c,
            A_ub=A_pos_A * A_ub_in + A_neg_A * A_lb_in,
            b_ub=A_pos * lin.b_ub + A_neg * lin.b_lb + c,
        )
    return LinearBound(
        A_lb=A_pos_A * lin.A_lb + A_neg_A * lin.A_ub,
        b_lb=A_pos * lin.b_lb + A_neg * lin.b_ub + c,
        A_ub=A_pos_A * lin.A_ub + A_neg_A * lin.A_lb,
        b_ub=A_pos * lin.b_ub + A_neg * lin.b_lb + c,
    )


def _fwd_lrelu(lin: LinearBound, lb: torch.Tensor, ub: torch.Tensor, alpha: float) -> Optional[LinearBound]:
    """Apply forward triangle linear relaxation for LeakyReLU.

    Lazy identity: LReLU after INPUT (rare) materializes diagonal.
    """
    on = lb >= 0
    off = ub <= 0
    amb = ~(on | off)
    denom = (ub - lb).clamp(min=1e-12)
    alpha_tensor = torch.full_like(lb, alpha)
    up_slope = torch.where(
        amb,
        (ub - alpha * lb) / denom,
        torch.where(on, torch.ones_like(lb), alpha_tensor),
    )
    up_inter = torch.where(amb, alpha * lb - up_slope * lb, torch.zeros_like(lb))
    low_slope = torch.where(on, torch.ones_like(lb), alpha_tensor)

    if lin.A_lb is None or lin.A_ub is None:
        B, n = lb.shape
        if n > _DENSE_LIN_BOUND_MAX_DIM:
            return None
        eye = torch.eye(n, device=lb.device, dtype=lb.dtype).unsqueeze(0).expand(B, n, n)
        A_lb_in = lin.A_lb if lin.A_lb is not None else eye
        A_ub_in = lin.A_ub if lin.A_ub is not None else eye
        return LinearBound(
            A_lb=low_slope.unsqueeze(-1) * A_lb_in,
            b_lb=low_slope * lin.b_lb,
            A_ub=up_slope.unsqueeze(-1) * A_ub_in,
            b_ub=up_slope * lin.b_ub + up_inter,
        )
    return LinearBound(
        A_lb=low_slope.unsqueeze(-1) * lin.A_lb,
        b_lb=low_slope * lin.b_lb,
        A_ub=up_slope.unsqueeze(-1) * lin.A_ub,
        b_ub=up_slope * lin.b_ub + up_inter,
    )


def _fwd_conv2d(layer: Layer, lin: LinearBound) -> Optional[LinearBound]:
    """Propagate dual-track affine bounds through Conv2D via batched F.conv2d.

    Under CUDA memory pressure the batch is convolved in lane chunks (see
    :func:`_lin_bound_lane_chunk`) and reassembled; interval fallback is kept
    only for the case where a single lane does not fit.
    """
    weight = cast(torch.Tensor, layer.params["weight"])
    bias = layer.params.get("bias")
    stride = layer.params.get("stride", 1)
    padding = layer.params.get("padding", 0)
    dilation = layer.params.get("dilation", 1)
    groups = layer.params.get("groups", 1)
    if isinstance(stride, (list, tuple)):
        stride = stride[0]
    if isinstance(padding, (list, tuple)):
        padding = padding[0]
    if isinstance(dilation, (list, tuple)):
        dilation = dilation[0]

    out_c, in_c_per_g, _, _ = weight.shape
    in_c = in_c_per_g * groups

    if lin.A_lb is None or lin.A_ub is None:
        B, curr_dim = lin.b_lb.shape
        if curr_dim > _DENSE_LIN_BOUND_MAX_DIM:
            return None
        device, dtype = lin.b_lb.device, lin.b_lb.dtype
        eye = torch.eye(curr_dim, device=device, dtype=dtype).unsqueeze(0).expand(B, curr_dim, curr_dim)
        A_lb_in = lin.A_lb if lin.A_lb is not None else eye
        A_ub_in = lin.A_ub if lin.A_ub is not None else eye
        lin = LinearBound(A_lb=A_lb_in, b_lb=lin.b_lb,
                          A_ub=A_ub_in, b_ub=lin.b_ub)

    B, curr_dim, input_dim = lin.A_lb.shape

    in_h = in_w = 0
    input_shape = _shape_list(layer.params.get("input_shape"))
    if input_shape is not None and len(input_shape) >= 3:
        shape = input_shape
        if len(shape) == 4:
            _, _, h, w = shape
        else:
            _, h, w = shape[-3], shape[-2], shape[-1]
        if h * w * in_c == curr_dim:
            in_h, in_w = h, w

    if in_h == 0 or in_w == 0:
        spatial = curr_dim // in_c if in_c > 0 else 0
        side = int(spatial ** 0.5) if spatial > 0 else 0
        if spatial > 0 and side * side * in_c == curr_dim:
            in_h = in_w = side

    if in_h == 0 or in_w == 0:
        return None

    k_h, k_w = int(weight.shape[2]), int(weight.shape[3])
    stride_i = cast(int, stride)
    padding_i = cast(int, padding)
    dilation_i = cast(int, dilation)
    out_h = (in_h + 2 * padding_i - dilation_i * (k_h - 1) - 1) // stride_i + 1
    out_w = (in_w + 2 * padding_i - dilation_i * (k_w - 1) - 1) // stride_i + 1
    chunk = _lin_bound_lane_chunk(B, out_c * out_h * out_w, input_dim, lin.b_lb,
                                  _LIN_BOUND_CHUNK_TRANSIENT_CONV2D, layer.id, "conv2d")
    if chunk is None:
        return None

    def conv_A(A_mat: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        lanes = A_mat.shape[0]
        A_t = A_mat.transpose(1, 2).contiguous().view(lanes * input_dim, in_c, in_h, in_w)
        out = F.conv2d(A_t, kernel, None, stride, padding, dilation, groups)
        return out.flatten(start_dim=1).reshape(lanes, input_dim, -1).transpose(1, 2).contiguous()

    def conv_b(vec: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        b_4d = vec.view(vec.shape[0], in_c, in_h, in_w)
        return F.conv2d(b_4d, kernel, None, stride, padding, dilation, groups).flatten(start_dim=1)

    # Centre-radius composition, algebraically identical to the split-sign
    # form: W+ @ A_lb + W- @ A_ub == W @ Ac - |W| @ Ar with Ac=(A_lb+A_ub)/2,
    # Ar=(A_ub-A_lb)/2 (and the mirrored identity for the upper track). The
    # split-sign form runs four INDEPENDENT accumulations whose float32
    # rounding crosses by ~n*eps*sum|W||b| once the tracks nearly coincide
    # (RC2's linear-track sibling; vgg spec2 inverted by 1e2 at ~1e5
    # magnitudes). Sharing the centre term keeps bit-equal input tracks
    # bit-equal on output, so concretized boxes cannot invert along affine
    # chains. cuDNN is disabled for the radius convs: Winograd's subtractive
    # transforms emit small negatives for a zero/nonnegative radius (see
    # _fwd_conv2d_interval).
    def compose(sub: LinearBound) -> LinearBound:
        A_centre = (sub.A_lb + sub.A_ub) * 0.5
        A_radius = (sub.A_ub - sub.A_lb) * 0.5
        b_centre = (sub.b_lb + sub.b_ub) * 0.5
        b_radius = (sub.b_ub - sub.b_lb) * 0.5
        A_mid = conv_A(A_centre, weight)
        b_mid = conv_b(b_centre, weight)
        with torch.backends.cudnn.flags(enabled=False):
            A_spread = conv_A(A_radius, weight.abs())
            b_spread = conv_b(b_radius, weight.abs())
        A_lb_new = A_mid - A_spread
        A_ub_new = A_mid + A_spread
        b_lb_new = b_mid - b_spread
        b_ub_new = b_mid + b_spread

        if bias is not None:
            out_spatial = b_lb_new.shape[1] // out_c
            bias_bc = bias.view(out_c, 1).expand(out_c, out_spatial).reshape(-1)
            b_lb_new = b_lb_new + bias_bc
            b_ub_new = b_ub_new + bias_bc

        return LinearBound(A_lb=A_lb_new, b_lb=b_lb_new, A_ub=A_ub_new, b_ub=b_ub_new)

    if chunk >= B:
        return compose(lin)
    return _cat_lanes([
        compose(_lin_lanes(lin, start, min(start + chunk, B)))
        for start in range(0, B, chunk)
    ])


def _fwd_conv2d_interval(layer: Layer, lb: torch.Tensor, ub: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback interval Conv2D for when the linear-relaxation path cannot infer shape.

    Centre-radius form: ``m = c ⊛ W + b``, ``rho = r ⊛ |W|``, ``[m-rho, m+rho]``.
    ``rho >= 0`` structurally, so ``ub - lb = 2*rho >= 0`` per neuron whatever
    the rounding; the ``W⁺l + W⁻u`` / ``W⁺u + W⁻l`` form sums two independent
    4608-term accumulations that float32 cancellation can order the wrong way
    (VGG16 L30: 11 515 inverted neurons, worst excess 3.9e-1). Two convs, not four.

    The radius conv runs with cuDNN disabled: cuDNN picks Winograd for 3x3, whose
    transforms contain subtractions, so it returns small *negative* values for a
    non-negative input/kernel pair (measured -1.3 where neighbouring tiles carry a
    large radius). Plain accumulation keeps ``rho >= 0`` exact, hence ``ub >= lb``.
    """
    weight = cast(torch.Tensor, layer.params["weight"])
    bias = layer.params.get("bias")
    stride = layer.params.get("stride", 1)
    padding = layer.params.get("padding", 0)
    dilation = layer.params.get("dilation", 1)
    groups = layer.params.get("groups", 1)
    if isinstance(stride, (list, tuple)):
        stride = stride[0]
    if isinstance(padding, (list, tuple)):
        padding = padding[0]
    if isinstance(dilation, (list, tuple)):
        dilation = dilation[0]

    B = lb.shape[0]
    _, in_c_per_g, _, _ = weight.shape
    in_c = in_c_per_g * groups
    input_shape = _shape_list(layer.params.get("input_shape"))
    if input_shape is not None and len(input_shape) >= 3:
        shape = input_shape
        if len(shape) == 4:
            _, _, in_h, in_w = shape
        else:
            _, in_h, in_w = shape[-3], shape[-2], shape[-1]
    else:
        spatial = lb.shape[1] // in_c if in_c > 0 else 0
        side = int(spatial ** 0.5) if spatial > 0 else 0
        if side * side * in_c != lb.shape[1]:
            raise ValueError(
                f"_fwd_conv2d_interval: cannot infer spatial shape for "
                f"{lb.shape[1]} features with in_c={in_c}; layer {layer.id} "
                f"needs an explicit 'input_shape' param"
            )
        in_h = in_w = side

    try:
        centre = ((lb + ub) * 0.5).view(B, in_c, in_h, in_w)
        radius = ((ub - lb) * 0.5).view(B, in_c, in_h, in_w)
    except RuntimeError as e:
        raise ValueError(
            f"_fwd_conv2d_interval: reshape to [B={B}, {in_c}, {in_h}, {in_w}] "
            f"failed for lb.shape={tuple(lb.shape)}"
        ) from e

    conv_kw = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    mid = F.conv2d(centre, weight, None, **conv_kw)
    with torch.backends.cudnn.flags(enabled=False):
        spread = F.conv2d(radius, weight.abs(), None, **conv_kw)
    if bias is not None:
        mid = mid + bias.view(1, -1, 1, 1)
    return (mid - spread).flatten(start_dim=1), (mid + spread).flatten(start_dim=1)


def _fwd_maxpool2d(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """MaxPool2D interval propagation with runtime batch size."""
    kernel_size = layer.params.get("kernel_size", 2)
    stride = layer.params.get("stride", kernel_size)
    padding = layer.params.get("padding", 0)
    dilation = layer.params.get("dilation", 1)
    input_shape = _shape_list(layer.params.get("input_shape"))
    if input_shape is None:
        raise ValueError(
            f"_fwd_maxpool2d: layer {layer.id} missing required 'input_shape' param"
        )

    shape = input_shape
    if len(shape) == 4:
        _, c, h, w = shape
    else:
        c, h, w = shape[-3], shape[-2], shape[-1]
    B = lb.shape[0]
    lb_out = F.max_pool2d(lb.view(B, c, h, w), kernel_size, stride, padding, dilation)
    ub_out = F.max_pool2d(ub.view(B, c, h, w), kernel_size, stride, padding, dilation)
    return lb_out.flatten(start_dim=1), ub_out.flatten(start_dim=1)


def _fwd_maxpool2d_lin(
    layer: Layer, lin: LinearBound, lb: torch.Tensor, ub: torch.Tensor,
    frame: Frame,
) -> Optional[Tuple[LinearBound, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """DeepPoly MaxPool2D on the dual-track affine bounds (explicit ``A`` only).

    Per window let ``i*`` be the argmax of the interval lower bounds. Since
    ``max(x_window) >= x_{i*}`` always, input ``i*``'s affine lower bound is a
    sound lower bound for the pool output. When ``i*`` dominates —
    ``lb_{i*} >= max_{j != i*} ub_j`` — the pool output *is* ``x_{i*}``, so its
    affine upper bound is exact; otherwise the constant ``max_j ub_j`` is used,
    lifted to also cover the concretized max of the gathered lower row over
    ``frame``. ``ub_max`` comes from the interval track while the lower row
    comes from the linear track; once the box width collapses (deep input
    splits) the two independent float32 accumulations can cross by a few ulps,
    and a crossed (lower_fn, upper_const) pair is amplified by every following
    affine layer's |W| mass until the backward's degenerate-interval check
    raises. Raising the upper constant to ``max(ub_max, max_frame lower_fn)``
    is sound (uppers may only grow) and restores the exact-arithmetic
    invariant ``upper >= lower`` pointwise on the frame.
    ``ub_second`` (largest ub among the window's other inputs) is computed via
    ``F.unfold`` with the argmax slot and zero-filled padding slots masked to
    ``-inf``.

    Under CUDA memory pressure the gathers run in lane chunks (see
    :func:`_lin_bound_lane_chunk`); the window statistics above them do not
    scale with ``n_sym`` and stay batched.

    Returns ``None`` on the lazy-identity path (``A is None``), on a feature
    width mismatch, or when a single lane's gathered ``[1, n_out, n_sym]``
    coefficients would overrun CUDA memory — the caller then keeps the
    interval-reset behaviour. Otherwise returns ``(lin_out, idx_flat, dominant, lb_max,
    ub_max)`` where ``idx_flat`` is the flat argmax index per output (long
    ``[B, n_out]``), ``dominant`` is a bool ``[B, n_out]`` mask, and
    ``lb_max``/``ub_max`` are the interval pool box ``[B, n_out]``.
    """
    if lin.A_lb is None or lin.A_ub is None:
        return None
    kernel_size = pair_2d(layer.params.get("kernel_size", 2))
    stride = pair_2d(layer.params.get("stride", layer.params.get("kernel_size", 2)))
    padding = pair_2d(layer.params.get("padding", 0))
    dilation = pair_2d(layer.params.get("dilation", 1))
    input_shape = _shape_list(layer.params.get("input_shape"))
    if input_shape is None:
        raise ValueError(
            f"_fwd_maxpool2d_lin: layer {layer.id} missing required 'input_shape' param"
        )
    shape = input_shape
    if len(shape) == 4:
        _, c, h, w = shape
    else:
        c, h, w = shape[-3], shape[-2], shape[-1]
    B, n_in = lb.shape
    if n_in != c * h * w or lin.A_lb.shape[1] != n_in or lin.b_lb.shape[1] != n_in:
        return None

    lb_4d = lb.view(B, c, h, w)
    ub_4d = ub.view(B, c, h, w)
    lb_max_4d, idx_plane = F.max_pool2d(
        lb_4d, kernel_size, stride, padding, dilation, return_indices=True
    )
    ub_max_4d = F.max_pool2d(ub_4d, kernel_size, stride, padding, dilation)
    n_out = c * lb_max_4d.shape[2] * lb_max_4d.shape[3]
    n_sym = lin.A_lb.shape[2]
    chunk = _lin_bound_lane_chunk(B, n_out, n_sym, lin.b_lb,
                                  _LIN_BOUND_CHUNK_TRANSIENT_MAXPOOL2D, layer.id, "maxpool2d")
    if chunk is None:
        return None

    # Window membership via im2col: map each kernel slot to its input plane
    # index. Unfold zero-fills padding, so shift indices by one to tell a
    # padded slot (0) from plane position 0; float64 keeps indices exact.
    plane = torch.arange(
        1, h * w + 1, device=lb.device, dtype=torch.float64
    ).view(1, 1, h, w)
    plane_unf = F.unfold(
        plane, kernel_size, dilation=dilation, padding=padding, stride=stride
    )
    pad_slot = plane_unf == 0
    slot_plane = plane_unf.long() - 1
    k2, spatial_out = plane_unf.shape[1], plane_unf.shape[2]

    # Largest ub among the window's OTHER inputs: mask the argmax-lb slot and
    # the padded slots (unfold zero-fill) to -inf, then reduce over the kernel.
    ub_unf = F.unfold(
        ub_4d, kernel_size, dilation=dilation, padding=padding, stride=stride
    ).view(B, c, k2, spatial_out)
    is_argmax = slot_plane.view(1, 1, k2, spatial_out) == idx_plane.view(B, c, 1, spatial_out)
    ub_second = ub_unf.masked_fill(
        is_argmax | pad_slot.view(1, 1, k2, spatial_out), float("-inf")
    ).amax(dim=2)

    lb_max = lb_max_4d.flatten(start_dim=1)
    ub_max = ub_max_4d.flatten(start_dim=1)
    dominant = (lb_max_4d.view(B, c, spatial_out) >= ub_second).view(B, n_out)

    channel_base = (torch.arange(c, device=lb.device) * (h * w)).view(1, c, 1)
    idx_flat = (idx_plane.view(B, c, spatial_out) + channel_base).view(B, n_out)

    x_L, x_U = frame.box.lb, frame.box.ub

    def gather_lanes(start: int, end: int) -> LinearBound:
        idx = idx_flat[start:end]
        gather_idx = idx.unsqueeze(-1).expand(end - start, n_out, n_sym)
        dom = dominant[start:end]
        A_lb_out = cast(torch.Tensor, lin.A_lb)[start:end].gather(1, gather_idx)
        b_lb_out = lin.b_lb[start:end].gather(1, idx)
        A_ub_out = cast(torch.Tensor, lin.A_ub)[start:end].gather(1, gather_idx).masked_fill(
            ~dom.unsqueeze(-1), 0.0)
        lower_row_max = (
            torch.einsum("boi,bi->bo", A_lb_out.clamp(min=0), x_U[start:end])
            + torch.einsum("boi,bi->bo", A_lb_out.clamp(max=0), x_L[start:end])
            + b_lb_out
        )
        b_ub_out = torch.where(
            dom, lin.b_ub[start:end].gather(1, idx),
            torch.maximum(ub_max[start:end], lower_row_max),
        )
        return LinearBound(A_lb=A_lb_out, b_lb=b_lb_out, A_ub=A_ub_out, b_ub=b_ub_out)

    if chunk >= B:
        lin_out = gather_lanes(0, B)
    else:
        lin_out = _cat_lanes([
            gather_lanes(start, min(start + chunk, B)) for start in range(0, B, chunk)
        ])
    return lin_out, idx_flat, dominant, lb_max, ub_max


def _fwd_avgpool2d(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """AvgPool2D interval propagation with runtime batch size."""
    kernel_size = layer.params.get("kernel_size", 2)
    stride = layer.params.get("stride", kernel_size)
    padding = layer.params.get("padding", 0)
    ceil_mode = bool(layer.params.get("ceil_mode", False))
    count_include_pad = bool(layer.params.get("count_include_pad", True))
    divisor_override = layer.params.get("divisor_override")
    input_shape = _shape_list(layer.params.get("input_shape"))
    if input_shape is None:
        raise ValueError(
            f"_fwd_avgpool2d: layer {layer.id} missing required 'input_shape' param"
        )

    shape = input_shape
    if len(shape) == 4:
        _, c, h, w = shape
    else:
        c, h, w = shape[-3], shape[-2], shape[-1]
    B = lb.shape[0]
    pool_kwargs = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "divisor_override": divisor_override,
    }
    lb_out = F.avg_pool2d(lb.view(B, c, h, w), **pool_kwargs)
    ub_out = F.avg_pool2d(ub.view(B, c, h, w), **pool_kwargs)
    return lb_out.flatten(start_dim=1), ub_out.flatten(start_dim=1)


def _align(a: torch.Tensor, n: int) -> torch.Tensor:
    """Align a 1-D parameter tensor to size n."""
    a = a.flatten()
    if a.numel() == n:
        return a
    if a.numel() > n:
        return a[:n]
    return a.repeat((n + a.numel() - 1) // a.numel())[:n]
