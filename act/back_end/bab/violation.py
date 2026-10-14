# ===- act/back_end/bab/violation.py - Counterexample Checking -----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------====#
#
# Purpose:
#   Concrete counterexample checks for BaB candidates. A candidate batch is
#   run through the reconstructed PyTorch model (cached per net until
#   ``clear_violation_check_module_cache``) and tested against the ASSERT
#   parameters; ``_check_input_specs_batched`` confirms each candidate lies
#   inside the INPUT_SPEC region.
#
# ===---------------------------------------------------------------------====#

from __future__ import annotations

from typing import List, cast

import torch

from act.back_end.core import Layer, Net
from act.front_end.specs import InKind, OutKind, normalize_position_mask
from act.util.model_inference import infer_single_model


def _as_batched_vector(
    value: object,
    n_batch: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    t = t.to(device=device, dtype=dtype)
    if t.dim() == 0:
        return t.expand(n_batch, width).contiguous()
    if t.dim() == 1:
        if t.numel() == width:
            return t.unsqueeze(0).expand(n_batch, -1).contiguous()
        if width == 1 and t.numel() == n_batch:
            return t.reshape(n_batch, 1).contiguous()
    if t.dim() == 2:
        if t.shape == (1, width):
            return t.expand(n_batch, -1).contiguous()
        if t.shape == (n_batch, width):
            return t.contiguous()
    raise ValueError(
        f"{name}: expected scalar, ({width},), (1,{width}), or "
        f"({n_batch},{width}); got {tuple(t.shape)}"
    )


def _as_batched_index(
    value: object,
    n_batch: int,
    n_out: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    t = t.to(device=device, dtype=torch.long).reshape(-1)
    if t.numel() == 1:
        t = t.expand(n_batch)
    if t.numel() != n_batch:
        raise ValueError(
            f"{name}: expected 1 or {n_batch} indices, got {t.numel()}"
        )
    if bool(((t < 0) | (t >= n_out)).any().item()):
        raise ValueError(f"{name}: index out of range for n_out={n_out}: {t.tolist()}")
    return t.contiguous()


# Per-net cache of reconstructed PyTorch nn.Module used for CE validation.
# Without this, every check_violations_batched call rebuilds the module from
# scratch via ACTToTorch.run() — costly under K-batched BaB which can invoke
# CE validation dozens of times for a single net.  Cleared per top-level
# verify-all dispatch via clear_violation_check_module_cache().
_VIOLATION_CHECK_MODULE_CACHE: dict[int, torch.nn.Module] = {}


def clear_violation_check_module_cache() -> None:
    _VIOLATION_CHECK_MODULE_CACHE.clear()


def _module_float_dtype(module: torch.nn.Module, default: torch.dtype) -> torch.dtype:
    for tensor in module.parameters():
        if tensor.is_floating_point():
            return tensor.dtype
    for tensor in module.buffers():
        if tensor.is_floating_point():
            return tensor.dtype
    return default


def _forward_for_violation_check(net: object, x_batch: torch.Tensor) -> torch.Tensor:
    if isinstance(net, torch.nn.Module):
        module = net
    else:
        key = id(net)
        cached = _VIOLATION_CHECK_MODULE_CACHE.get(key)
        if cached is None:
            from act.pipeline.verification.act2torch import ACTToTorch

            cached = ACTToTorch(cast(Net, net)).run()
            # ACTToTorch emits mixed float32 weights + float64 buffers; unify to
            # the analysis dtype or the internal forward clashes float32/float64.
            if x_batch.is_floating_point():
                cached = cached.to(dtype=x_batch.dtype)
            _VIOLATION_CHECK_MODULE_CACHE[key] = cached
        module = cached
    _ = module.eval()
    target_dtype = _module_float_dtype(module, x_batch.dtype)
    if x_batch.dtype != target_dtype:
        x_batch = x_batch.to(dtype=target_dtype)
    success, output, error = infer_single_model("ce_validate_batched", module, x_batch)
    if not success or output is None:
        raise RuntimeError(f"check_violations_batched: model forward failed: {error}")
    if output.dim() < 2:
        raise ValueError(
            f"check_violations_batched: model output must be batched, got "
            f"shape={tuple(output.shape)}"
        )
    return output.reshape(output.shape[0], -1)


@torch.no_grad()
def check_violations_batched(net: object, x_batch: torch.Tensor, assert_layer: Layer) -> torch.Tensor:
    """[BATCHED-API] Return a ``[N]`` bool tensor for concrete ASSERT violations.

    ``x_batch`` is always treated as a tensor-view batch ``[N, *input_shape]``;
    N=1 is represented by a length-one leading dimension. ASSERT parameters are
    read directly from ``assert_layer.params`` in their batch-native form.
    """
    if x_batch.dim() < 2:
        raise ValueError(
            f"check_violations_batched: x_batch must be [N, *input_shape], "
            f"got shape={tuple(x_batch.shape)}"
        )
    y_batch = _forward_for_violation_check(net, x_batch)
    n_batch = int(x_batch.shape[0])
    if int(y_batch.shape[0]) != n_batch:
        raise ValueError(
            f"check_violations_batched: output batch {int(y_batch.shape[0])} "
            f"!= input batch {n_batch}"
        )
    n_out = int(y_batch.shape[1])
    device = y_batch.device
    dtype = y_batch.dtype
    params = assert_layer.params
    kind = params.get("kind")
    eps = 1e-8

    if kind == OutKind.TOP1_ROBUST:
        y_true = _as_batched_index(params["y_true"], n_batch, n_out, device, "y_true")
        y_true_scores = y_batch.gather(1, y_true.unsqueeze(1)).squeeze(1)
        mask = torch.ones_like(y_batch, dtype=torch.bool)
        _ = mask.scatter_(1, y_true.unsqueeze(1), False)
        other_scores = y_batch.masked_fill(~mask, -float("inf"))
        return (other_scores.max(dim=1).values - y_true_scores) >= 0

    if kind == OutKind.MARGIN_ROBUST:
        y_true = _as_batched_index(params["y_true"], n_batch, n_out, device, "y_true")
        margin = _as_batched_vector(
            params["margin"], n_batch, 1, device, dtype, "margin"
        ).reshape(n_batch)
        y_true_scores = y_batch.gather(1, y_true.unsqueeze(1)).squeeze(1)
        mask = torch.ones_like(y_batch, dtype=torch.bool)
        _ = mask.scatter_(1, y_true.unsqueeze(1), False)
        other_scores = y_batch.masked_fill(~mask, -float("inf"))
        # ``-margin``, not ``+margin``: ``encode_linear`` emits rows ``e_j - e_t``
        # with ``thresholds = -margin``, so a lane is certified iff
        # ``max_j(z_j - z_t) < -margin``. The negative sign is deliberate.
        return (other_scores.max(dim=1).values - y_true_scores) >= -margin

    if kind == OutKind.LINEAR_LE:
        c_raw = params["c"]
        c_t = (c_raw if isinstance(c_raw, torch.Tensor) else torch.as_tensor(c_raw)).to(device=device, dtype=dtype)
        if c_t.dim() <= 1:
            rows = 1
        elif c_t.dim() == 2:
            if c_t.shape[1] != n_out:
                raise ValueError(f"LINEAR_LE: c cols {c_t.shape[1]} != n_out {n_out}")
            rows = int(c_t.shape[0])
        else:
            rows = int(c_t.shape[1])
        if rows <= 1:
            coeff = _as_batched_vector(params["c"], n_batch, n_out, device, dtype, "c")
            bound = _as_batched_vector(params["d"], n_batch, 1, device, dtype, "d").reshape(n_batch)
            return (coeff * y_batch).sum(dim=1) >= bound + eps
        if c_t.dim() == 2:
            c_view = c_t.unsqueeze(0).expand(n_batch, -1, -1)
        else:
            c_view = c_t if c_t.shape[0] == n_batch else c_t.expand(n_batch, -1, -1)
        d_raw = params["d"]
        d_t = (d_raw if isinstance(d_raw, torch.Tensor) else torch.as_tensor(d_raw)).to(device=device, dtype=dtype).flatten()
        if d_t.numel() == rows:
            d_view = d_t.unsqueeze(0).expand(n_batch, -1)
        elif d_t.numel() == n_batch * rows:
            d_view = d_t.reshape(n_batch, rows)
        else:
            raise ValueError(f"LINEAR_LE: d numel {d_t.numel()} incompatible with rows {rows}")
        lhs = torch.einsum("bmo,bo->bm", c_view.contiguous(), y_batch)
        return (lhs > d_view + eps).any(dim=1)

    if kind == OutKind.RANGE:
        result = torch.zeros(n_batch, dtype=torch.bool, device=device)
        lb_raw = params.get("lb")
        ub_raw = params.get("ub")
        if lb_raw is not None:
            lb = _as_batched_vector(lb_raw, n_batch, n_out, device, dtype, "lb")
            result = result | (y_batch < lb - eps).any(dim=1)
        if ub_raw is not None:
            ub = _as_batched_vector(ub_raw, n_batch, n_out, device, dtype, "ub")
            result = result | (y_batch > ub + eps).any(dim=1)
        return result

    if kind == OutKind.UNSAFE_LINEAR:
        m_raw = params.get("M", 1)
        if isinstance(m_raw, torch.Tensor):
            m_rows = int(m_raw.item())
        elif isinstance(m_raw, int):
            m_rows = m_raw
        else:
            raise ValueError(f"UNSAFE_LINEAR: M must be int, got {m_raw!r}")
        c_raw = params.get("C", params.get("c"))
        if c_raw is None:
            raise ValueError("UNSAFE_LINEAR requires C or c params")
        c_tensor = c_raw if isinstance(c_raw, torch.Tensor) else torch.as_tensor(c_raw)
        c_tensor = c_tensor.to(device=device, dtype=dtype)
        if c_tensor.dim() == 2:
            if c_tensor.shape == (m_rows, n_out):
                c_view = c_tensor.unsqueeze(0).expand(n_batch, -1, -1).contiguous()
            elif c_tensor.shape == (n_batch * m_rows, n_out):
                c_view = c_tensor.reshape(n_batch, m_rows, n_out).contiguous()
            else:
                raise ValueError(
                    f"UNSAFE_LINEAR: C shape {tuple(c_tensor.shape)} incompatible "
                    f"with N={n_batch}, M={m_rows}, n_out={n_out}"
                )
        elif c_tensor.dim() == 3:
            if c_tensor.shape == (1, m_rows, n_out):
                c_view = c_tensor.expand(n_batch, -1, -1).contiguous()
            elif c_tensor.shape == (n_batch, m_rows, n_out):
                c_view = c_tensor.contiguous()
            else:
                raise ValueError(
                    f"UNSAFE_LINEAR: c shape {tuple(c_tensor.shape)} incompatible "
                    f"with N={n_batch}, M={m_rows}, n_out={n_out}"
                )
        else:
            raise ValueError(f"UNSAFE_LINEAR: unsupported C dim {c_tensor.dim()}")
        d_raw = params.get("thresholds", params.get("d"))
        if d_raw is None:
            raise ValueError("UNSAFE_LINEAR requires thresholds or d params")
        d_tensor = d_raw if isinstance(d_raw, torch.Tensor) else torch.as_tensor(d_raw)
        d_tensor = d_tensor.to(device=device, dtype=dtype)
        if d_tensor.dim() == 1 and d_tensor.numel() == m_rows:
            d_view = d_tensor.unsqueeze(0).expand(n_batch, -1).contiguous()
        elif d_tensor.shape == (1, m_rows):
            d_view = d_tensor.expand(n_batch, -1).contiguous()
        elif d_tensor.shape == (n_batch, m_rows):
            d_view = d_tensor.contiguous()
        else:
            raise ValueError(
                f"UNSAFE_LINEAR: d shape {tuple(d_tensor.shape)} incompatible "
                f"with N={n_batch}, M={m_rows}"
            )
        lhs = torch.einsum("bmo,bo->bm", c_view, y_batch)
        return (lhs <= d_view + eps).all(dim=1)

    raise NotImplementedError(f"ASSERT kind not supported: {kind}")


def _check_input_specs_batched(x_batch: torch.Tensor, spec_layers: List[Layer]) -> torch.Tensor:
    result = torch.ones(x_batch.shape[0], device=x_batch.device, dtype=torch.bool)
    tol = 1e-7
    for layer in spec_layers:
        kind = layer.params.get("kind")
        if kind not in (InKind.BOX, InKind.LINF_BALL, InKind.LP_EMBEDDING):
            continue
        lb = layer.params.get("lb")
        ub = layer.params.get("ub")
        if isinstance(lb, torch.Tensor) and isinstance(ub, torch.Tensor):
            lb_t = lb.to(device=x_batch.device, dtype=x_batch.dtype)
            ub_t = ub.to(device=x_batch.device, dtype=x_batch.dtype)
            result &= ((x_batch >= lb_t - tol) & (x_batch <= ub_t + tol)).flatten(start_dim=1).all(dim=1)
        if kind != InKind.LP_EMBEDDING:
            continue
        center = layer.params.get("center")
        eps = layer.params.get("eps")
        p_norm = layer.params.get("p_norm")
        if not isinstance(center, torch.Tensor) or eps is None or p_norm is None:
            result &= torch.zeros_like(result)
            continue
        center_t = center.to(device=x_batch.device, dtype=x_batch.dtype)
        if isinstance(eps, torch.Tensor):
            eps_t = eps.to(device=x_batch.device, dtype=x_batch.dtype)
        elif isinstance(eps, (int, float, bool)):
            eps_t = center_t.new_tensor(float(eps))
        else:
            result &= torch.zeros_like(result)
            continue
        if isinstance(p_norm, torch.Tensor):
            p_value = float(p_norm.reshape(-1)[0].item())
        elif isinstance(p_norm, (int, float, bool)):
            p_value = float(p_norm)
        else:
            result &= torch.zeros_like(result)
            continue
        positions_raw = layer.params.get("perturbed_positions")
        positions = (
            positions_raw
            if positions_raw is None or isinstance(positions_raw, (torch.Tensor, list, tuple))
            else None
        )
        mask = normalize_position_mask(
            positions,
            int(center_t.shape[-2]),
            batch_shape=tuple(center_t.shape[:-2]),
            device=x_batch.device,
        )
        if mask.shape[0] == 1 and x_batch.shape[0] != 1:
            mask_b = mask.expand(x_batch.shape[0], *mask.shape[1:])
            center_b = center_t.expand_as(x_batch)
        else:
            mask_b = mask
            center_b = center_t.expand_as(x_batch)
        delta = x_batch - center_b
        clean = (~mask_b).unsqueeze(-1).expand_as(delta)
        if bool(clean.any().item()):
            clean_ok = torch.where(clean, delta.abs(), torch.zeros_like(delta)).flatten(start_dim=1).amax(dim=1) <= tol
            result &= clean_ok
        perturbed_delta = delta[mask_b.unsqueeze(-1).expand_as(delta)].reshape(x_batch.shape[0], -1, center_t.shape[-1])
        if perturbed_delta.numel() == 0:
            continue
        if p_value == float("inf"):
            norms = perturbed_delta.abs().amax(dim=-1)
        elif p_value == 1.0:
            norms = perturbed_delta.abs().sum(dim=-1)
        elif p_value == 2.0:
            norms = torch.linalg.vector_norm(perturbed_delta, ord=2, dim=-1)
        else:
            norms = torch.linalg.vector_norm(perturbed_delta, ord=p_value, dim=-1)
        result &= (norms <= eps_t.reshape(-1)[0] + tol).all(dim=1)
    return result
