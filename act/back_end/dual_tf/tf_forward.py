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

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from act.back_end.core import Bounds, Layer, Net
from act.back_end.layer_schema import LayerKind
from act.util.device_manager import get_default_device, get_default_dtype


@dataclass
class LinearBound:
    A_lb: Optional[torch.Tensor]
    b_lb: torch.Tensor
    A_ub: Optional[torch.Tensor]
    b_ub: torch.Tensor

Frame = Tuple[torch.Tensor, torch.Tensor]  # (x_L, x_U) over which lin is defined


# Above this layer input dim, ``_fwd_X`` returns ``None`` to skip the
# dense linear-bound A matrix; caller ``forward_X`` falls back to
# interval-only via ``_box_X`` + ``_reset_forward_box``. Sound (interval
# is always sound), looser bounds at the bailed layer.
_DENSE_LIN_BOUND_MAX_DIM: int = 10_000


def _concretize(lin: LinearBound, x_L: torch.Tensor, x_U: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concretize dual-track affine bounds over a batched input box.

    Lazy identity: when ``A_lb is None`` the affine bound is the identity,
    so ``lb = x_L + b_lb`` (and analogously for ub). This avoids materializing
    a dense ``[B, n, n]`` eye matrix at the INPUT layer, which for 224×224×3
    input would request 181 GB.
    """
    n_x = x_L.shape[1]
    if lin.A_lb is None:
        b_lb = lin.b_lb
        if b_lb.shape[1] != n_x:
            n_b = min(b_lb.shape[1], n_x)
            lb = x_L[..., :n_b] + b_lb[..., :n_b]
        else:
            lb = x_L + b_lb
    else:
        A_lb_p = lin.A_lb.clamp(min=0)
        A_lb_n = lin.A_lb.clamp(max=0)
        lb = (
            torch.einsum("boi,bi->bo", A_lb_p, x_L)
            + torch.einsum("boi,bi->bo", A_lb_n, x_U)
            + lin.b_lb
        )
    if lin.A_ub is None:
        b_ub = lin.b_ub
        if b_ub.shape[1] != n_x:
            n_b = min(b_ub.shape[1], n_x)
            ub = x_U[..., :n_b] + b_ub[..., :n_b]
        else:
            ub = x_U + b_ub
    else:
        A_ub_p = lin.A_ub.clamp(min=0)
        A_ub_n = lin.A_ub.clamp(max=0)
        ub = (
            torch.einsum("boi,bi->bo", A_ub_p, x_U)
            + torch.einsum("boi,bi->bo", A_ub_n, x_L)
            + lin.b_ub
        )
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
    return torch.maximum(lb_a, lb_b), torch.minimum(ub_a, ub_b)

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
    W = layer.params["weight"]
    b = layer.params.get("bias")
    lb = _align_batch(lb, W.shape[1])
    ub = _align_batch(ub, W.shape[1])
    W_pos = W.clamp(min=0)
    W_neg = W.clamp(max=0)
    out_lb = lb @ W_pos.transpose(0, 1) + ub @ W_neg.transpose(0, 1)
    out_ub = ub @ W_pos.transpose(0, 1) + lb @ W_neg.transpose(0, 1)
    if b is not None:
        bias_vec = _align(b.flatten(), W.shape[0])
        out_lb = out_lb + bias_vec
        out_ub = out_ub + bias_vec
    return out_lb, out_ub


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
                       ) -> Tuple[LinearBound, Tuple[torch.Tensor, torch.Tensor]]:
    """Reset dual-track state on a concrete box."""
    lin, x_L, x_U = _reset_lin(lb, ub, device, dtype)
    return lin, (x_L, x_U)


def _topological_sort(net: Net) -> List[int]:
    """Return a Kahn topological order over the ACT DAG."""
    layer_ids = [layer.id for layer in net.layers]
    in_deg: Dict[int, int] = {lid: len(set(net.preds.get(lid, []))) for lid in layer_ids}
    queue = deque(lid for lid in layer_ids if in_deg[lid] == 0)
    order: List[int] = []
    while queue:
        lid = queue.popleft()
        order.append(lid)
        for succ in net.succs.get(lid, []):
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)
    if len(order) != len(layer_ids):
        raise ValueError(f"compute_forward_bounds: graph has cycle or disconnected layers ({len(order)}/{len(layer_ids)} sorted)")
    return order


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

    Lazy identity: ``A = None`` denotes identity. Summing identities is
    a diagonal=k·I matrix; we still materialize only when at least one
    predecessor has a non-None A (i.e. is no longer identity). If all are
    None, result remains None (single-identity case is impossible because
    callers only sum from >=2 predecessors).
    """
    A_lbs = [lin.A_lb for lin in lins]
    A_ubs = [lin.A_ub for lin in lins]

    def _materialize(A: Optional[torch.Tensor], example: torch.Tensor) -> torch.Tensor:
        if A is not None:
            return A
        B, n = example.shape
        eye = torch.eye(n, device=example.device, dtype=example.dtype)
        return eye.unsqueeze(0).expand(B, n, n)

    if all(A is None for A in A_lbs):
        A_lb_sum = None
    else:
        any_real = next(A for A in A_lbs if A is not None)
        A_lb_sum = sum((_materialize(A, lins[i].b_lb) for i, A in enumerate(A_lbs[1:], 1)),
                       _materialize(A_lbs[0], lins[0].b_lb))

    if all(A is None for A in A_ubs):
        A_ub_sum = None
    else:
        any_real = next(A for A in A_ubs if A is not None)
        A_ub_sum = sum((_materialize(A, lins[i].b_ub) for i, A in enumerate(A_ubs[1:], 1)),
                       _materialize(A_ubs[0], lins[0].b_ub))

    return LinearBound(
        A_lb=A_lb_sum,
        b_lb=sum((lin.b_lb for lin in lins[1:]), lins[0].b_lb),
        A_ub=A_ub_sum,
        b_ub=sum((lin.b_ub for lin in lins[1:]), lins[0].b_ub),
    )


@torch.no_grad()
def compute_forward_bounds(net: Net, input_lb: torch.Tensor, input_ub: torch.Tensor,
                           post_activation: bool = False,
                           alphas: Optional[Dict[int, torch.Tensor]] = None) -> Dict[int, Bounds]:
    """Forward bounds, natively batched with singleton auto-promotion."""
    # Lazy import to break circular dep (dual_tf imports compute_forward_bounds)
    from .dual_tf import DualTF

    device, dtype = get_default_device(), get_default_dtype()
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
    input_dim = lb_in.shape[1]

    bounds_dict: Dict[int, Bounds] = {}
    box_state: Dict[int, Bounds] = {}
    lin_state: Dict[int, LinearBound] = {}
    frame_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    topo_order = _topological_sort(net)
    by_id = getattr(net, "by_id", {layer.id: layer for layer in net.layers})
    entry_box = Bounds(lb_in, ub_in)
    entry_lin = _identity_lin(B, input_dim, device, dtype)
    entry_frame = (lb_in, ub_in)

    for lid in topo_order:
        layer = by_id[lid]
        lid = layer.id
        kind = layer.kind.upper()
        preds = list(net.preds.get(lid, []) or [])

        if not preds:
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
            x_L, x_U = parent_frame
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
                lin_lb, lin_ub = _concretize(lin, x_L, x_U)
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

    return bounds_dict


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
    """
    W = layer.params["weight"]
    b = layer.params.get("bias")
    lin = _match_lin_input_dim(lin, W.shape[1])
    W_pos = W.clamp(min=0)
    W_neg = W.clamp(max=0)
    B = lin.b_lb.shape[0]
    bias_vec = torch.zeros(W.shape[0], device=lin.b_lb.device, dtype=lin.b_lb.dtype)
    if b is not None:
        bias_vec = _align(b.flatten(), W.shape[0])

    if lin.A_lb is None and lin.A_ub is None:
        if W.shape[1] > _DENSE_LIN_BOUND_MAX_DIM:
            return None
        W_broadcast_lb = W.unsqueeze(0).expand(B, -1, -1).contiguous()
        W_broadcast_ub = W.unsqueeze(0).expand(B, -1, -1).contiguous()
        b_lb_new = torch.einsum("oc,bc->bo", W_pos, lin.b_lb) + torch.einsum("oc,bc->bo", W_neg, lin.b_ub) + bias_vec
        b_ub_new = torch.einsum("oc,bc->bo", W_pos, lin.b_ub) + torch.einsum("oc,bc->bo", W_neg, lin.b_lb) + bias_vec
        return LinearBound(A_lb=W_broadcast_lb, b_lb=b_lb_new,
                           A_ub=W_broadcast_ub, b_ub=b_ub_new)

    return LinearBound(
        A_lb=torch.einsum("oc,bci->boi", W_pos, lin.A_lb) + torch.einsum("oc,bci->boi", W_neg, lin.A_ub),
        b_lb=torch.einsum("oc,bc->bo", W_pos, lin.b_lb) + torch.einsum("oc,bc->bo", W_neg, lin.b_ub) + bias_vec,
        A_ub=torch.einsum("oc,bci->boi", W_pos, lin.A_ub) + torch.einsum("oc,bci->boi", W_neg, lin.A_lb),
        b_ub=torch.einsum("oc,bc->bo", W_pos, lin.b_ub) + torch.einsum("oc,bc->bo", W_neg, lin.b_lb) + bias_vec,
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
    """Propagate dual-track affine bounds through Conv2D via batched F.conv2d."""
    weight = layer.params["weight"]
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

    W_pos = weight.clamp(min=0)
    W_neg = weight.clamp(max=0)

    def conv_A(A_mat: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        A_t = A_mat.transpose(1, 2).contiguous().view(B * input_dim, in_c, in_h, in_w)
        out = F.conv2d(A_t, kernel, None, stride, padding, dilation, groups)
        return out.flatten(start_dim=1).reshape(B, input_dim, -1).transpose(1, 2).contiguous()

    def conv_b(vec: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        b_4d = vec.view(B, in_c, in_h, in_w)
        return F.conv2d(b_4d, kernel, None, stride, padding, dilation, groups).flatten(start_dim=1)

    A_lb_new = conv_A(lin.A_lb, W_pos) + conv_A(lin.A_ub, W_neg)
    A_ub_new = conv_A(lin.A_ub, W_pos) + conv_A(lin.A_lb, W_neg)
    b_lb_new = conv_b(lin.b_lb, W_pos) + conv_b(lin.b_ub, W_neg)
    b_ub_new = conv_b(lin.b_ub, W_pos) + conv_b(lin.b_lb, W_neg)

    if bias is not None:
        out_spatial = b_lb_new.shape[1] // out_c
        bias_bc = bias.view(out_c, 1).expand(out_c, out_spatial).reshape(-1)
        b_lb_new = b_lb_new + bias_bc
        b_ub_new = b_ub_new + bias_bc

    return LinearBound(A_lb=A_lb_new, b_lb=b_lb_new, A_ub=A_ub_new, b_ub=b_ub_new)


def _fwd_conv2d_interval(layer: Layer, lb: torch.Tensor, ub: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback interval Conv2D for when the linear-relaxation path cannot infer shape."""
    weight = layer.params["weight"]
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
        lb_4d = lb.view(B, in_c, in_h, in_w)
        ub_4d = ub.view(B, in_c, in_h, in_w)
    except RuntimeError as e:
        raise ValueError(
            f"_fwd_conv2d_interval: reshape to [B={B}, {in_c}, {in_h}, {in_w}] "
            f"failed for lb.shape={tuple(lb.shape)}"
        ) from e

    W_pos = weight.clamp(min=0)
    W_neg = weight.clamp(max=0)
    conv_kw = dict(stride=stride, padding=padding, dilation=dilation, groups=groups)
    lb_out = F.conv2d(lb_4d, W_pos, None, **conv_kw) + F.conv2d(ub_4d, W_neg, None, **conv_kw)
    ub_out = F.conv2d(ub_4d, W_pos, None, **conv_kw) + F.conv2d(lb_4d, W_neg, None, **conv_kw)
    if bias is not None:
        bias_4d = bias.view(1, -1, 1, 1)
        lb_out = lb_out + bias_4d
        ub_out = ub_out + bias_4d
    return lb_out.flatten(start_dim=1), ub_out.flatten(start_dim=1)


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


def _fwd_avgpool2d(layer: Layer, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """AvgPool2D interval propagation with runtime batch size."""
    kernel_size = layer.params.get("kernel_size", 2)
    stride = layer.params.get("stride", kernel_size)
    padding = layer.params.get("padding", 0)
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
    lb_out = F.avg_pool2d(lb.view(B, c, h, w), kernel_size, stride, padding)
    ub_out = F.avg_pool2d(ub.view(B, c, h, w), kernel_size, stride, padding)
    return lb_out.flatten(start_dim=1), ub_out.flatten(start_dim=1)


def _align(a: torch.Tensor, n: int) -> torch.Tensor:
    """Align a 1-D parameter tensor to size n."""
    a = a.flatten()
    if a.numel() == n:
        return a
    if a.numel() > n:
        return a[:n]
    return a.repeat((n + a.numel() - 1) // a.numel())[:n]
