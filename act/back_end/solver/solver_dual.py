#===- act/back_end/solver/solver_dual.py - Dual Bounds Solver ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# DualSolver: linear-relaxation dual certified lower-bound solver.
# STRICT batched API ([B, *shape] only). Raises ValueError on 1-D input.
# Mirrors HZSolver precedent in solver_hz.py.
#===---------------------------------------------------------------------===#
# pyright: reportMissingImports=false, reportImportCycles=false
# justification: torch C-extension stubs are absent in CI; DualSolver and verifier share result utilities during type analysis

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import torch
from act.back_end.core import Bounds, Layer, Net
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_base import Solver, SolverCaps
from act.front_end.specs import OutputSpec, OutKind
from act.util.device_manager import get_default_device, get_default_dtype
from act.util.stats import SpecBatchResult

if TYPE_CHECKING:
    from act.back_end.dual_tf.dual_tf import DualTF


@dataclass(frozen=True)
class DualResult:
    """Result of ``compute_certified_bound``. Fields depend on caller flags."""

    margins: torch.Tensor
    sce: Optional[Any] = None
    alpha_state: Optional[Dict[int, torch.Tensor]] = None
    eta_state: Optional[Dict[int, torch.Tensor]] = None
    nu_per_layer: Optional[Dict[int, torch.Tensor]] = None


def expand_bounds_dict(bounds_dict: Dict[int, Bounds], M: int) -> Dict[int, Bounds]:
    """Expand each batched Bounds entry from [B, *shape] to [B*M, *shape].

    The dual solver threads ``M`` through ``compute_certified_bound`` and
    broadcasts inside the activation handlers (lazy M-broadcast), avoiding the
    M× memory blowup; this explicit expansion is for callers that need a
    materialized M-expanded bounds dict.

    repeat_interleave aligns with row b*M+j sharing sample b's bounds. All
    entries must already be batched (lb.dim() >= 2). M=1 returns the dict
    unchanged.
    """
    if M <= 0:
        raise ValueError(f"expand_bounds_dict: M must be positive, got {M}")
    if M == 1:
        return dict(bounds_dict)
    out: Dict[int, Bounds] = {}
    for lid, bounds in bounds_dict.items():
        if bounds.lb.dim() < 2:
            raise ValueError(
                f"expand_bounds_dict: layer {lid} bounds must be batched "
                f"[B, *shape], got dim={bounds.lb.dim()} shape={tuple(bounds.lb.shape)}"
            )
        out[lid] = Bounds(
            lb=bounds.lb.repeat_interleave(M, dim=0),
            ub=bounds.ub.repeat_interleave(M, dim=0),
        )
    return out


def _alpha_tree_leaves(tree: Any):
    """Yield the tensor leaves of a dual-alpha pytree.

    A per-layer alpha is a pytree whose shape is interpreted by the matching
    backward kernel: RELU is a single tensor leaf; future per-kind allocators
    may nest leaves in lists/dicts. ``None`` marks a fixed-slope kind with no
    alpha. The optimizer, projection, and keep-best clone all walk these leaves
    so they cannot drift out of agreement on the pytree shape.
    """
    if tree is None:
        return
    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _alpha_tree_leaves(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _alpha_tree_leaves(value)
    else:
        raise TypeError(f"unsupported alpha pytree node: {type(tree)!r}")


def _clone_alpha_tree(tree: Any) -> Any:
    """Detach-clone every leaf of an alpha pytree, preserving its structure."""
    if tree is None:
        return None
    if isinstance(tree, torch.Tensor):
        return tree.detach().clone()
    if isinstance(tree, dict):
        return {key: _clone_alpha_tree(value) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_clone_alpha_tree(value) for value in tree)
    raise TypeError(f"unsupported alpha pytree node: {type(tree)!r}")


def _reverse_topological_sort(net: Net) -> List[int]:
    """Kahn's algorithm on net.succs.

    Returns layer IDs in reverse-topological order: every layer appears
    after all its successors.

    Raises:
        ValueError: If the graph contains a cycle or disconnected layers.
    """
    in_deg: Dict[int, int] = {layer.id: len(net.succs.get(layer.id, [])) for layer in net.layers}
    queue: List[int] = [lid for lid, degree in in_deg.items() if degree == 0]
    order: List[int] = []
    while queue:
        lid = queue.pop(0)
        order.append(lid)
        for pred in set(net.preds.get(lid, [])):
            in_deg[pred] -= 1
            if in_deg[pred] == 0:
                queue.append(pred)
    if len(order) != len(net.layers):
        raise ValueError(
            f"DualSolver: graph has cycle or disconnected layers "
            f"({len(order)}/{len(net.layers)} sorted)"
        )
    return order


# Floor on the L2 witness denominator so a zero-coefficient block yields a
# center witness instead of a divide-by-zero.
_DUAL_NORM_EPS = 1e-12


def _resolve_perturbation_norm(value: Any) -> float:
    """Normalize a spec ``p_norm`` field to a float, defaulting to ``inf``.

    LP_EMBEDDING carries the input perturbation norm ``p``; box / L_inf specs
    omit it. A missing field (``None``) maps to ``inf`` so the dual solver keeps
    the exact box concretization with zero behavior change.

    Args:
        value: Raw ``p_norm`` from the input-spec layer params. Accepts ``None``,
            a number, or a string such as ``"inf"`` / ``"2"``.

    Returns:
        The perturbation norm ``p`` as a float (``float('inf')`` for L_inf).
    """
    if value is None:
        return float("inf")
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("inf", "+inf", "infinity", "linf", "l_inf"):
            return float("inf")
        return float(token)
    return float(value)


def _dual_norm_exponent(p: float) -> float:
    """Return the Hölder dual exponent ``q`` with ``1/p + 1/q = 1``.

    Used to evaluate the exact ``min`` of a linear form over an Lp input ball
    (``min_{‖δ‖_p ≤ ε} ν·δ = −ε‖ν‖_q``): p=inf→q=1 (box), p=2→q=2, p=1→q=inf.
    """
    if p == float("inf"):
        return 1.0
    if p == 1.0:
        return float("inf")
    return p / (p - 1.0)


class DualSolver(Solver):
    """Dual (linear-relaxation) certified bounds solver. Strict [B, *shape] API."""

    _AFFINE_CONTRIB_KINDS = {
        LayerKind.DENSE.value,
        LayerKind.CONV2D.value,
        LayerKind.BIAS.value,
        LayerKind.BN.value,
        LayerKind.ADD.value,
    }

    def __init__(self, n_iters: int = 0):
        # DualTF is a backward-handler registry (not a TransferFunction);
        # instantiate it internally so DualSolver is self-contained and callers
        # need no knowledge of it.
        from act.back_end.dual_tf.dual_tf import DualTF
        self.tf = DualTF()
        self.n_iters = n_iters
        self._last_bounds: Optional[Bounds] = None
        self.last_forward_bounds: Optional[Dict[int, Bounds]] = None

    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=True, supports_csp=False, supports_dual=True)

    def compute_certified_bound(
        self, net: Net, bounds_dict: Dict[int, Bounds],
        c: torch.Tensor, M: int = 1,
        return_sce: bool = False,
        enable_grad: bool = False,
        alpha: Optional[Dict[int, torch.Tensor]] = None,
        eta: Optional[Dict[int, torch.Tensor]] = None,
        split_signs: Optional[Union[Dict[int, torch.Tensor], List[Dict[int, torch.Tensor]]]] = None,
        optimize: bool = False,
        n_iters: int = 50,
        lr_alpha: float = 0.1,
        lr_beta: float = 0.1,
        lr_decay: float = 0.98,
        incremental_alphas: Optional[Dict[int, torch.Tensor]] = None,
        incremental_etas: Optional[Dict[int, torch.Tensor]] = None,
        return_optimized: bool = False,
        per_class_alpha: bool = True,
        return_nu_per_layer: bool = False,
        optimize_alpha: bool = True,
        refresh_forward: bool = True,
        start_lid: Optional[int] = None,
    ) -> DualResult:
        """Batched certified lower bound on c^T @ output (DAG-aware).

        Implements ``Solver.compute_certified_bound``; see base for the
        full contract. DualSolver realises this via reverse-topological
        backward propagation of a per-layer accumulator:
          nu_accum[lid] = sum over all successors s of ν routed by s's handler to lid.

        Each handler returns per-pred νs; the outer loop distributes them to preds.

        Unknown layer kind raises ValueError (no silent identity fallback for soundness).

        Lazy M-broadcast: ``c`` has shape ``[B*M, num_classes]`` packed
        sample-major (row ``b*M+j`` = sample b's j-th spec row), but
        ``bounds_dict`` entries stay at ``[B, *shape]``. Activation handlers
        (RELU/SIGMOID/TANH) view nu as ``[B, M, n]`` and broadcast bounds
        ``[B, 1, n]`` against it — mathematically equivalent to the legacy
        M-expanded path, with M× lower bounds memory.

        Caveats:
            Uses GLOBAL intermediate bounds — bounds_dict stays at [B, *shape] across
            BaB lanes within the same sample (lazy-M-broadcast design). Per-lane
            intermediate-bound tightening (a stricter dual variant) is OUT OF SCOPE
            for this solver; bounds may be looser than a per-lane refinement would
            give, but the dual lower bound remains SOUND.

        When ``optimize=True``, runs joint Adam optimization over ReLU α/η via
        ``_optimize_alpha_eta``. If ``return_optimized=True``, also returns the optimized
        α state for incremental-starting. When ``optimize=False``, executes a single-pass
        backward; with ``alpha=None`` it uses the default fixed-slope relaxation.

        eta: Per-layer η multiplier — Lagrange (KKT) multiplier for branch-split
            constraints on activation pre-activations. Keyed by activation layer id
            (RELU, LRELU, SIGMOID, TANH, GELU). η ≥ 0 invariant (enforced by clamp).
        split_signs: Per-layer split direction. {-1: inactive, +1: active, 0: unsplit}.
            Same key set as eta.
        η is applied to the TRUE pre-activation variable (immediately AFTER the
        activation handler in the reverse-topological backward loop):
        nu_pre = slope · nu_post − η · sign, so the multiplier acts on the
        affine pre-activation unscaled. Applying it before the handler would
        scale η by the relaxation slope, forcing the effective multiplier to 0
        on inactive-split (slope = 0) neurons and discarding the z ≤ 0
        constraint's input-region information. Sound for any η ≥ 0.
        """
        if isinstance(split_signs, list):
            # KFSB path: accept K split hypotheses and return stacked margins
            # [K, N, ...], evaluating each hypothesis through the single-hypothesis
            # backward pass (no leading-K vectorization).
            if not split_signs:
                raise ValueError("split_signs list cannot be empty")
            if optimize:
                raise ValueError("list-form split_signs is only supported with optimize=False")
            if return_optimized:
                raise ValueError("return_optimized requires optimize=True and single split_signs")
            stacked_split_signs = self._stack_split_sign_hypotheses(split_signs)
            normalized_split_signs = self._unstack_split_sign_hypotheses(stacked_split_signs)
            margins = []
            sce_values = []
            for hypo in normalized_split_signs:
                result = self.compute_certified_bound(
                    net,
                    bounds_dict,
                    c,
                    M=M,
                    return_sce=return_sce,
                    enable_grad=enable_grad,
                    alpha=alpha,
                    eta=eta,
                    split_signs=hypo,
                    optimize=False,
                    n_iters=n_iters,
                    lr_alpha=lr_alpha,
                    lr_beta=lr_beta,
                    lr_decay=lr_decay,
                    incremental_alphas=incremental_alphas,
                    incremental_etas=incremental_etas,
                    return_optimized=False,
                    per_class_alpha=per_class_alpha,
                    return_nu_per_layer=False,
                )
                margins.append(result.margins)
                if return_sce:
                    sce_values.append(result.sce)
            stacked_sce = None
            if return_sce and sce_values and all(sce is not None for sce in sce_values):
                stacked_sce = torch.stack(cast(List[torch.Tensor], sce_values), dim=0)
            return DualResult(margins=torch.stack(margins, dim=0), sce=stacked_sce)

        bounds_dict = self._harden_split_bounds(bounds_dict, split_signs)

        if optimize:
            bound, sce, alpha_state, eta_state = self._optimize_alpha_eta(
                net,
                bounds_dict,
                c,
                M=M,
                n_iters=n_iters,
                lr_alpha=lr_alpha,
                lr_beta=lr_beta,
                lr_decay=lr_decay,
                incremental_alphas=incremental_alphas,
                incremental_etas=incremental_etas,
                split_signs=split_signs,
                return_sce=return_sce,
                per_class_alpha=per_class_alpha,
                optimize_alpha=optimize_alpha,
                refresh_forward=refresh_forward,
                start_lid=start_lid,
            )
            if return_optimized:
                return DualResult(
                    margins=bound,
                    sce=sce if return_sce else None,
                    alpha_state=alpha_state if alpha_state else None,
                    eta_state=eta_state if eta_state else None,
                )
            return DualResult(
                margins=bound,
                sce=sce if return_sce else None,
            )

        if c.dim() != 2:
            raise ValueError(
                f"c must be 2-D [B*M, num_classes], got shape {tuple(c.shape)}. "
                "Use c.unsqueeze(0) for single instance.")
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")
        if c.shape[0] % M != 0:
            raise ValueError(
                f"c batch dim {c.shape[0]} not divisible by M={M}; "
                f"expected c.shape[0] == B*M for some integer B"
            )
        with torch.set_grad_enabled(enable_grad):
            assert len(bounds_dict) > 0, "bounds_dict cannot be empty"
            device, dtype = get_default_device(), get_default_dtype()
            if c.dtype != dtype or c.device != device:
                c = c.to(device=device, dtype=dtype)
            B = c.shape[0]

            for _ in range(self.n_iters):
                pass

            if start_lid is not None:
                # Interior start: c is a linear functional on layer
                # ``start_lid``'s output; only its ancestors are visited by
                # the backward loop (non-ancestors never enter nu_accum).
                # Used by refine_intermediate_bounds.
                output_lid = start_lid
            else:
                assert_layer = None
                for layer in net.layers:
                    k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
                    if k == LayerKind.ASSERT.value:
                        assert_layer = layer
                        break
                if assert_layer is None:
                    raise ValueError("DualSolver.compute_certified_bound: net has no ASSERT layer")

                assert_preds = net.preds.get(assert_layer.id, [])
                if len(assert_preds) != 1:
                    raise ValueError(
                        f"DualSolver.compute_certified_bound: ASSERT layer {assert_layer.id} must have "
                        f"exactly 1 predecessor, got {len(assert_preds)}"
                    )

                output_lid = assert_preds[0]
            nu_accum: Dict[int, torch.Tensor] = {output_lid: c.clone()}
            nu_snapshot: Dict[int, torch.Tensor] = {}
            obj = torch.zeros(B, dtype=c.dtype, device=c.device)

            topo_order = _reverse_topological_sort(net)
            registry = self.tf._BACKWARD_REGISTRY

            for lid in topo_order:
                layer = net.by_id[lid]
                k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind

                if k in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value, LayerKind.ASSERT.value):
                    continue

                if lid not in nu_accum:
                    continue

                nu_here = nu_accum.pop(lid)
                handler = registry.get(k)
                if handler is None:
                    raise ValueError(
                        f"DualSolver.compute_certified_bound: unknown layer kind '{k}' at layer {lid}; "
                        f"soundness requires explicit backward handler. "
                        f"Supported kinds: {sorted(registry.keys())}"
                    )

                if return_nu_per_layer and k == LayerKind.RELU.value:
                    nu_snapshot[lid] = nu_here.detach().clone()

                preds = list(net.preds.get(lid, []))
                if alpha is None:
                    pred_nus, contrib = handler(layer, nu_here, bounds_dict, preds, M)
                else:
                    pred_nus, contrib = handler(
                        layer, nu_here, bounds_dict, preds, M, alpha=alpha.get(lid)
                    )

                if eta is not None and split_signs is not None and lid in eta:
                    # Split Lagrangian on the TRUE pre-activation variable:
                    # nu_pre = D nu_post - eta * sign.
                    # Applying it BEFORE the handler would scale eta by the
                    # relaxation slope D - forcing beta = 0 on inactive-split
                    # (D = 0) neurons, which silently discards the z <= 0
                    # constraint's restriction of the input region (the only
                    # channel carrying it: the y = 0 relaxation says nothing
                    # about x). Sound for any eta >= 0: on the child region
                    # sign * z >= 0, so -eta * sign * z <= 0 only lowers the
                    # minimized Lagrangian below the true child minimum.
                    eta_l = eta[lid].to(device=nu_here.device, dtype=nu_here.dtype)
                    signs_l = split_signs[lid].to(device=nu_here.device, dtype=nu_here.dtype)
                    if eta_l.dim() == 3:
                        eta_l = eta_l.reshape(-1, eta_l.shape[-1])
                    if signs_l.dim() == 3:
                        signs_l = signs_l.reshape(-1, signs_l.shape[-1])
                    pn = pred_nus[0]
                    n_clip = min(pn.shape[-1], eta_l.shape[-1])
                    pn = pn.clone()
                    pn[..., :n_clip] = (
                        pn[..., :n_clip]
                        - eta_l[..., :n_clip] * signs_l[..., :n_clip]
                    )
                    pred_nus = [pn, *pred_nus[1:]]

                if len(pred_nus) != len(preds):
                    raise ValueError(
                        f"handler {k} at layer {lid} returned {len(pred_nus)} pred_nus, "
                        f"expected {len(preds)}"
                    )
                if contrib.shape != (B,):
                    raise ValueError(
                        f"handler {k} at layer {lid} contrib shape {tuple(contrib.shape)}, "
                        f"expected ({B},)"
                    )

                if k in self._AFFINE_CONTRIB_KINDS:
                    contrib = -contrib

                obj = obj + contrib
                for pred_id, pred_nu in zip(preds, pred_nus):
                    if pred_id in nu_accum:
                        nu_accum[pred_id] = nu_accum[pred_id] + pred_nu
                    else:
                        nu_accum[pred_id] = pred_nu.clone()

            input_lid = self._find_input_layer_id(net)
            if input_lid is None:
                return DualResult(
                    margins=obj,
                    sce=None if return_sce else None,
                    nu_per_layer=nu_snapshot if return_nu_per_layer else None,
                )

            nu_final = nu_accum.get(input_lid)
            if nu_final is None:
                return DualResult(
                    margins=obj,
                    sce=None if return_sce else None,
                    nu_per_layer=nu_snapshot if return_nu_per_layer else None,
                )

            input_contrib, sce = self._input_contribution_from_nu(
                net,
                input_lid,
                nu_final,
                bounds_dict,
                M=M,
                return_sce=return_sce,
                enable_grad=enable_grad,
            )
            obj = obj + input_contrib
            return DualResult(
                margins=obj,
                sce=sce if return_sce else None,
                nu_per_layer=nu_snapshot if return_nu_per_layer else None,
            )

    def _stack_split_sign_hypotheses(
        self,
        split_signs: List[Dict[int, torch.Tensor]],
    ) -> Dict[int, torch.Tensor]:
        layer_ids = sorted({lid for hypo in split_signs for lid in hypo})
        stacked: Dict[int, torch.Tensor] = {}
        for lid in layer_ids:
            template = next(hypo[lid] for hypo in split_signs if lid in hypo)
            entries = [hypo.get(lid, torch.zeros_like(template)) for hypo in split_signs]
            stacked[lid] = torch.stack(entries, dim=0)
        return stacked

    def _unstack_split_sign_hypotheses(
        self,
        stacked_split_signs: Dict[int, torch.Tensor],
    ) -> List[Dict[int, torch.Tensor]]:
        first = next(iter(stacked_split_signs.values()))
        hypotheses: List[Dict[int, torch.Tensor]] = []
        for idx in range(first.shape[0]):
            hypotheses.append({lid: signs[idx] for lid, signs in stacked_split_signs.items()})
        return hypotheses

    def _init_alpha(
        self,
        layer: Layer,
        bounds_dict: Dict[int, Bounds],
        B: int,
        M: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        per_class_alpha: bool,
        optimize_alpha: bool,
        incremental_alphas: Optional[Dict[int, torch.Tensor]],
    ) -> Any:
        """Per-kind dual-alpha allocation returning a pytree of leaves, or None.

        RELU allocates an optimizable lower-envelope slope ``[B, M, n]`` (or
        ``[B, n]`` when ``per_class_alpha`` is off), warm-started from
        ``incremental_alphas`` when present. SOFTMAX and LAYERNORM are
        fixed-slope for now and allocate no alpha (``None``), so only RELU
        contributes optimizer leaves; their backward kernels interpret the
        absent alpha as the fixed relaxation. Each kind owns the shape of its
        own pytree, which the matching backward kernel reads back via
        ``alpha.get(lid)``.
        """
        k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        if k in (LayerKind.ATT_SCORES.value, LayerKind.ATT_MIX.value):
            return self._init_attention_alpha(
                layer, bounds_dict, device, dtype,
                optimize_alpha=optimize_alpha,
                incremental_alphas=incremental_alphas,
            )
        if k != LayerKind.RELU.value:
            return None
        b = bounds_dict.get(layer.id)
        if b is None:
            return None
        if incremental_alphas is not None and layer.id in incremental_alphas:
            alpha_init = (
                incremental_alphas[layer.id]
                .detach()
                .clone()
                .to(device=device, dtype=dtype)
                .clamp(0.0, 1.0)
            )
        else:
            lb_flat = b.lb.to(device=device, dtype=dtype).flatten(start_dim=1)
            ub_flat = b.ub.to(device=device, dtype=dtype).flatten(start_dim=1)
            n_neurons = lb_flat.shape[-1]
            denom = (ub_flat - lb_flat).clamp(min=1e-12)
            alpha_init_bn = (ub_flat / denom).clamp(0.0, 1.0).detach()
            if per_class_alpha:
                alpha_init = (
                    alpha_init_bn.unsqueeze(1)
                    .expand(B, M, n_neurons)
                    .contiguous()
                )
            else:
                alpha_init = alpha_init_bn.contiguous()
        return torch.nn.Parameter(alpha_init) if optimize_alpha else alpha_init.detach()

    def _init_attention_alpha(
        self,
        layer: Layer,
        bounds_dict: Dict[int, Bounds],
        device: torch.device,
        dtype: torch.dtype,
        *,
        optimize_alpha: bool,
        incremental_alphas: Optional[Dict[int, torch.Tensor]],
    ) -> Any:
        """Allocate the bilinear-attention fusion-slope pytree for one core.

        The pytree is ``{omega_l, omega_u}`` with per-output ``[B, 1]`` slopes,
        warm-started at the rule init derived from the same local input boxes the
        backward kernel reads (so allocator, kernel, keep-best clone and the
        ``[0, 1]`` projection all agree on the shape). Returns ``None`` when the
        predecessor boxes are missing so the kernel falls back to its rule slope.
        """
        from act.back_end.dual_tf.tf_transformer import (
            attention_rule_alpha, _attention_input_boxes,
        )
        try:
            x_l, x_u, y_l, y_u, _scale, _mask = _attention_input_boxes(layer, bounds_dict)
        except KeyError:
            return None
        x_l = x_l.to(device=device, dtype=dtype)
        x_u = x_u.to(device=device, dtype=dtype)
        y_l = y_l.to(device=device, dtype=dtype)
        y_u = y_u.to(device=device, dtype=dtype)
        if incremental_alphas is not None and layer.id in incremental_alphas:
            prior = cast(Dict[str, torch.Tensor], cast(object, incremental_alphas[layer.id]))
            tree = {
                "omega_l": prior["omega_l"].detach().clone().to(device=device, dtype=dtype).clamp(0.0, 1.0),
                "omega_u": prior["omega_u"].detach().clone().to(device=device, dtype=dtype).clamp(0.0, 1.0),
            }
        else:
            tree = attention_rule_alpha(x_l, x_u, y_l, y_u)
        if optimize_alpha:
            return {key: torch.nn.Parameter(val.detach().clone()) for key, val in tree.items()}
        return {key: val.detach() for key, val in tree.items()}

    def _optimize_alpha_eta(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        c: torch.Tensor,
        M: int = 1,
        n_iters: int = 50,
        lr_alpha: float = 0.1,
        lr_beta: float = 0.1,
        lr_decay: float = 0.98,
        incremental_alphas: Optional[Dict[int, torch.Tensor]] = None,
        incremental_etas: Optional[Dict[int, torch.Tensor]] = None,
        split_signs: Optional[Dict[int, torch.Tensor]] = None,
        return_sce: bool = False,
        per_class_alpha: bool = True,
        optimize_alpha: bool = True,
        refresh_forward: bool = True,
        start_lid: Optional[int] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Dict[int, torch.Tensor],
        Dict[int, torch.Tensor],
    ]:
        """Joint α/η optimization: iterative dual lower-bound refinement.

        Each ReLU gets a learnable lower-envelope slope α constrained to [0, 1].
        Each split layer with a nonzero split sign gets a learnable η multiplier
        constrained to η ≥ 0.

        Returns:
            ``(best_bounds, best_sce, alpha_state, eta_state)`` where
            ``best_bounds`` has shape ``[B*M]``, ``best_sce`` is optional,
            ``alpha_state`` maps ReLU layer id to optimized α, and ``eta_state``
            maps split layer id to optimized η.
        """
        if c.dim() != 2:
            raise ValueError(
                f"c must be 2-D [B*M, n_out], got shape {tuple(c.shape)}"
            )
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")
        BM = c.shape[0]
        if BM % M != 0:
            raise ValueError(
                f"c batch dim {BM} not divisible by M={M}; expected B*M rows"
            )
        B = BM // M
        device, dtype = c.device, c.dtype

        input_lid = self._find_input_layer_id(net)
        if input_lid is None:
            raise ValueError("DualSolver._optimize_alpha_eta: net has no INPUT/INPUT_SPEC layer")
        by_id = getattr(net, "by_id", {layer.id: layer for layer in net.layers})
        input_layer = by_id[input_lid]
        input_bounds = bounds_dict.get(input_lid)
        if input_bounds is None:
            if "lb" not in input_layer.params or "ub" not in input_layer.params:
                raise ValueError(
                        f"DualSolver._optimize_alpha_eta: input layer {input_lid} has no bounds"
                )
            input_lb = cast(torch.Tensor, input_layer.params["lb"])
            input_ub = cast(torch.Tensor, input_layer.params["ub"])
        else:
            input_lb = input_bounds.lb
            input_ub = input_bounds.ub
        input_lb = input_lb.to(device=device, dtype=dtype)
        input_ub = input_ub.to(device=device, dtype=dtype)

        ancestor_lids: Optional[set[int]] = None
        if start_lid is not None:
            # Interior-start objectives only depend on ancestor layers; alpha
            # parameters outside that cone receive no gradient and would crash
            # the optimizer.
            ancestor_lids = {start_lid}
            stack = [start_lid]
            while stack:
                for p in net.preds.get(stack.pop(), []):
                    if p not in ancestor_lids:
                        ancestor_lids.add(p)
                        stack.append(p)

        alphas: Dict[int, Any] = {}
        for layer in net.layers:
            if ancestor_lids is not None and layer.id not in ancestor_lids:
                continue
            tree = self._init_alpha(
                layer, bounds_dict, B, M, device, dtype,
                per_class_alpha=per_class_alpha,
                optimize_alpha=optimize_alpha,
                incremental_alphas=incremental_alphas,
            )
            if tree is not None:
                alphas[layer.id] = tree

        etas: Dict[int, torch.nn.Parameter] = {}
        if split_signs is not None:
            for lid, signs in split_signs.items():
                signs_init = signs.detach().to(device=device, dtype=dtype)
                if not (signs_init != 0).any():
                    continue
                if incremental_etas is not None and lid in incremental_etas:
                    eta_init = (
                        incremental_etas[lid]
                        .detach()
                        .clone()
                        .to(device=device, dtype=dtype)
                        .clamp(min=0)
                    )
                    if eta_init.shape != signs_init.shape:
                        eta_init = torch.zeros_like(signs_init)
                else:
                    eta_init = torch.zeros_like(signs_init)
                etas[lid] = torch.nn.Parameter(eta_init)

        if not alphas and not etas:
            result = self.compute_certified_bound(
                net, bounds_dict, c, M=M, return_sce=return_sce,
                start_lid=start_lid,
            )
            return result.margins.detach(), result.sce, {}, {}

        param_groups: List[Dict[str, object]] = []
        alpha_params = [
            leaf
            for tree in alphas.values()
            for leaf in _alpha_tree_leaves(tree)
            if isinstance(leaf, torch.nn.Parameter)
        ]
        eta_params = list(etas.values())
        if alpha_params:
            param_groups.append({"params": alpha_params, "lr": lr_alpha})
        if eta_params:
            param_groups.append({"params": eta_params, "lr": lr_beta})
        if not param_groups:
            result = self.compute_certified_bound(
                net, bounds_dict, c, M=M, return_sce=return_sce,
                alpha=cast(Dict[int, torch.Tensor], alphas) if alphas else None,
                start_lid=start_lid,
            )
            return (
                result.margins.detach(),
                result.sce,
                {lid: _clone_alpha_tree(tree) for lid, tree in alphas.items()},
                {},
            )
        optimizer = torch.optim.Adam(param_groups)
        scheduler = (
            torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
            if lr_decay < 1.0
            else None
        )
        best_bounds = torch.full((BM,), float("-inf"), device=device, dtype=dtype)
        best_sce: Optional[torch.Tensor] = None
        best_alpha_state: Dict[int, Any] = {
            lid: _clone_alpha_tree(tree) for lid, tree in alphas.items()
        }
        best_eta_state: Dict[int, torch.Tensor] = {
            lid: e.detach().clone() for lid, e in etas.items()
        }

        from act.back_end.dual_tf.tf_forward import compute_forward_bounds

        with torch.enable_grad():
            for _ in range(n_iters):
                optimizer.zero_grad()
                alpha_tensors = cast(Dict[int, torch.Tensor], alphas)
                eta_tensors = cast(Dict[int, torch.Tensor], etas)
                if refresh_forward:
                    forward_alphas = {
                        lid: a[:, 0, :] if a.dim() == 3 else a
                        for lid, a in alpha_tensors.items()
                        if isinstance(a, torch.Tensor)
                    }
                    fresh_bounds = compute_forward_bounds(
                        net,
                        input_lb,
                        input_ub,
                        post_activation=False,
                        alphas=forward_alphas,
                    )
                else:
                    # Fixed intermediate bounds (root-reuse mode): alpha/eta
                    # only enter the backward pass; sound for any valid
                    # bounds_dict.
                    fresh_bounds = bounds_dict
                result = self.compute_certified_bound(
                    net,
                    fresh_bounds,
                    c,
                    M=M,
                    return_sce=return_sce,
                    enable_grad=True,
                    alpha=alpha_tensors,
                    eta=eta_tensors,
                    split_signs=split_signs,
                    start_lid=start_lid,
                )
                bound_bm = result.margins
                sce = result.sce

                (-bound_bm.sum()).backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                with torch.no_grad():
                    for tree in alphas.values():
                        for leaf in _alpha_tree_leaves(tree):
                            leaf.data.clamp_(0.0, 1.0)
                    for e in etas.values():
                        e.data.clamp_(min=0)

                    improved = bound_bm > best_bounds
                    if improved.any():
                        best_bounds = torch.where(improved, bound_bm.detach(), best_bounds)
                        best_alpha_state = {
                            lid: _clone_alpha_tree(tree) for lid, tree in alphas.items()
                        }
                        best_eta_state = {
                            lid: e.detach().clone() for lid, e in etas.items()
                        }
                    if return_sce and sce is not None:
                        if best_sce is None:
                            best_sce = sce.detach().clone()
                        else:
                            best_sce[improved] = sce[improved].detach()

        if n_iters <= 0:
            result = self.compute_certified_bound(
                net,
                bounds_dict,
                c,
                M=M,
                return_sce=return_sce,
                enable_grad=False,
                alpha=cast(Dict[int, torch.Tensor], alphas),
                eta=cast(Dict[int, torch.Tensor], etas),
                split_signs=split_signs,
                start_lid=start_lid,
            )
            best_bounds = result.margins
            best_sce = result.sce

        return best_bounds.detach(), best_sce, best_alpha_state, best_eta_state

    def _harden_split_bounds(
        self,
        bounds_dict: Dict[int, Bounds],
        split_signs: Optional[Dict[int, torch.Tensor]],
    ) -> Dict[int, Bounds]:
        """Fix split ReLU phases in the relaxation itself.

        sign=+1 asserts pre-activation >= 0 (lb clamped to 0: the relaxation
        collapses to the exact identity); sign=-1 asserts <= 0 (ub clamped to
        0: exact zero). Applied alongside the eta Lagrangian term, the split
        tightens the bound through both channels. Sound: clamping encodes
        exactly the branch's split assumption.
        """
        if not split_signs or isinstance(split_signs, list):
            return bounds_dict
        out = dict(bounds_dict)
        for lid, signs in split_signs.items():
            b = out.get(lid)
            if b is None:
                continue
            s = signs[:, 0, :] if signs.dim() == 3 else signs
            if not bool((s != 0).any().item()):
                continue
            lb = b.lb.flatten(start_dim=1).clone()
            ub = b.ub.flatten(start_dim=1).clone()
            n = min(lb.shape[-1], s.shape[-1])
            s_n = s[..., :n].to(device=lb.device)
            lb[..., :n] = torch.where(s_n > 0, lb[..., :n].clamp(min=0.0), lb[..., :n])
            ub[..., :n] = torch.where(s_n < 0, ub[..., :n].clamp(max=0.0), ub[..., :n])
            # An infeasible branch (split contradicts a stable phase) yields a
            # degenerate [x, x] interval; the branch represents an empty input
            # region, so any bound for it is sound.
            ub[..., :n] = torch.maximum(ub[..., :n], lb[..., :n])
            out[lid] = Bounds(lb.view_as(b.lb), ub.view_as(b.ub))
        return out

    def refine_intermediate_bounds(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        mode: str = "auto",
        blowup_ratio: float = 10.0,
        max_rows_per_call: int = 4096,
        optimize_iters: int = 20,
    ) -> Dict[int, Bounds]:
        """Metric-driven backward refinement of selected pre-activation bounds.

        Forward-mode concretization loses correlation at wide fan-in affine
        layers; a backward pass from the affected layer to the input keeps it.
        Selection is architecture-agnostic: an activation layer qualifies if it
        has unstable neurons and (mode="auto") its mean pre-activation width
        exceeds ``blowup_ratio`` x the median width of all activation layers;
        mode="all" refines every unstable activation layer. Refined bounds are
        intersected with the forward bounds (both are valid over-approximations,
        so the intersection is sound). Layers are processed in topological
        order so later refinements consume earlier ones.
        """
        if mode == "none":
            return bounds_dict
        if mode not in ("auto", "all", "tail"):
            raise ValueError(
                f"intermediate_refine mode must be none|auto|all|tail, got {mode!r}"
            )

        stats = []
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k != LayerKind.RELU.value or layer.id not in bounds_dict:
                continue
            b = bounds_dict[layer.id]
            lb, ub = b.lb.flatten(start_dim=1), b.ub.flatten(start_dim=1)
            unstable = int(((lb < 0) & (ub > 0)).sum().item())
            stats.append((layer.id, unstable, float((ub - lb).mean().item())))
        if not stats:
            return bounds_dict

        median_width = sorted(s[2] for s in stats)[len(stats) // 2]
        threshold = max(median_width, 1e-9) * blowup_ratio
        if mode == "tail":
            unstable_lids = [lid for lid, unstable, _ in stats if unstable > 0]
            selected = unstable_lids[-2:]
        else:
            selected = [
                lid for lid, unstable, width in stats
                if unstable > 0 and (mode == "all" or width > threshold)
            ]
        if not selected:
            return bounds_dict

        out = dict(bounds_dict)
        for lid in selected:
            preds = net.preds.get(lid, [])
            if len(preds) != 1:
                continue
            pred_lid = preds[0]
            b = out[lid]
            lb0 = b.lb.flatten(start_dim=1)
            ub0 = b.ub.flatten(start_dim=1)
            if lb0.shape[0] != 1:
                continue
            n = lb0.shape[-1]
            device, dtype = lb0.device, lb0.dtype
            # Only unstable neurons need refinement: stable phases make the
            # relaxation exact regardless of bound width, so querying them
            # would spend backward rows for zero tightening.
            amb_idx = torch.where((lb0[0] < 0) & (ub0[0] > 0))[0]
            n_amb = int(amb_idx.numel())
            if n_amb == 0:
                continue
            lb_new = torch.empty(n_amb, device=device, dtype=dtype)
            ub_new = torch.empty(n_amb, device=device, dtype=dtype)
            for s in range(0, n_amb, max_rows_per_call):
                e = min(s + max_rows_per_call, n_amb)
                eye = torch.zeros(e - s, n, device=device, dtype=dtype)
                eye[torch.arange(e - s), amb_idx[s:e]] = 1.0
                rows = torch.cat([eye, -eye], dim=0)
                res = self.compute_certified_bound(
                    net, out, rows.contiguous(), M=int(rows.shape[0]),
                    start_lid=pred_lid,
                    optimize=optimize_iters > 0,
                    n_iters=optimize_iters,
                    lr_alpha=0.25,
                    lr_decay=0.98,
                    per_class_alpha=True,
                    refresh_forward=False,
                )
                lb_new[s:e] = res.margins[: e - s]
                ub_new[s:e] = -res.margins[e - s:]
            lb_ref = lb0[0].clone()
            ub_ref = ub0[0].clone()
            lb_ref[amb_idx] = torch.maximum(lb_ref[amb_idx], lb_new)
            ub_ref[amb_idx] = torch.minimum(ub_ref[amb_idx], ub_new)
            ub_ref = torch.maximum(ub_ref, lb_ref)
            refined = Bounds(
                lb_ref.view_as(b.lb[0]).unsqueeze(0).clone(),
                ub_ref.view_as(b.ub[0]).unsqueeze(0).clone(),
            )
            out[lid] = refined
            if pred_lid in out and out[pred_lid].lb.shape == refined.lb.shape:
                out[pred_lid] = refined
        return out

    def refine_intermediate_bounds_batched(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        split_signs: Optional[Dict[int, torch.Tensor]] = None,
        mode: str = "tail",
        rows_cap: int = 64,
        optimize_iters: int = 0,
        lane_chunk: int = 32,
    ) -> Dict[int, Bounds]:
        """K-lane per-subproblem sparse refinement of pre-activation bounds.

        Batched counterpart of ``refine_intermediate_bounds`` for the BaB loop:
        ``bounds_dict`` entries are ``[K, *shape]`` (one lane per subproblem).
        Split constraints are applied by hardening the bounds FIRST (sign=+1
        clamps lb to 0, sign=-1 clamps ub to 0); the backward pass then sees
        the hardened relaxation slopes, which propagates each lane's splits
        relationally to downstream layers - the tightening that the interval
        refresh cannot provide. ``split_signs`` is NOT forwarded to
        ``compute_certified_bound`` (its eta machinery is shaped for the final
        spec's M, not the refine rows); hardening alone carries the split.

        Rows are the per-neuron one-hot +/- queries for the UNION of unstable
        neurons across lanes, capped at ``rows_cap`` by descending interval
        width. Each refined bound is intersected per lane with the existing
        bound (both are valid over-approximations: sound). Layers are visited
        in topological order so later refinements consume earlier ones.
        """
        if mode == "none":
            return bounds_dict
        if mode not in ("tail", "all"):
            raise ValueError(
                f"per_subproblem_refine mode must be none|tail|all, got {mode!r}"
            )

        out = self._harden_split_bounds(bounds_dict, split_signs)

        stats: List[tuple[int, int]] = []
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k != LayerKind.RELU.value or layer.id not in out:
                continue
            b = out[layer.id]
            lb, ub = b.lb.flatten(start_dim=1), b.ub.flatten(start_dim=1)
            n_unstable = int(((lb < 0) & (ub > 0)).any(dim=0).sum().item())
            stats.append((layer.id, n_unstable))
        unstable_lids = [lid for lid, n_unstable in stats if n_unstable > 0]
        if not unstable_lids:
            return out
        selected = unstable_lids[-2:] if mode == "tail" else unstable_lids

        for lid in selected:
            preds = net.preds.get(lid, [])
            if len(preds) != 1:
                continue
            pred_lid = preds[0]
            b = out[lid]
            lb0 = b.lb.flatten(start_dim=1)
            ub0 = b.ub.flatten(start_dim=1)
            k_lanes = lb0.shape[0]
            n = lb0.shape[-1]
            device, dtype = lb0.device, lb0.dtype
            amb_union = ((lb0 < 0) & (ub0 > 0)).any(dim=0)
            amb_idx = torch.where(amb_union)[0]
            n_amb = int(amb_idx.numel())
            if n_amb == 0:
                continue
            if n_amb > rows_cap:
                width = (ub0 - lb0).amax(dim=0)[amb_idx]
                amb_idx = amb_idx[torch.topk(width, k=rows_cap).indices]
                n_amb = rows_cap
            eye = torch.zeros(n_amb, n, device=device, dtype=dtype)
            eye[torch.arange(n_amb), amb_idx] = 1.0
            rows = torch.cat([eye, -eye], dim=0)
            m_rows = int(rows.shape[0])
            margins = torch.empty(k_lanes, m_rows, device=device, dtype=dtype)
            for k0 in range(0, k_lanes, lane_chunk):
                k1 = min(k0 + lane_chunk, k_lanes)
                sub = {
                    l: Bounds(bb.lb[k0:k1], bb.ub[k0:k1]) for l, bb in out.items()
                }
                c = rows.repeat(k1 - k0, 1).contiguous()
                res = self.compute_certified_bound(
                    net, sub, c, M=m_rows,
                    start_lid=pred_lid,
                    optimize=optimize_iters > 0,
                    n_iters=optimize_iters,
                    lr_alpha=0.25,
                    lr_decay=0.98,
                    per_class_alpha=True,
                    refresh_forward=False,
                )
                margins[k0:k1] = res.margins.view(k1 - k0, m_rows)
            lb_new = margins[:, :n_amb]
            ub_new = -margins[:, n_amb:]
            lb_ref = lb0.clone()
            ub_ref = ub0.clone()
            lb_ref[:, amb_idx] = torch.maximum(lb_ref[:, amb_idx], lb_new)
            ub_ref[:, amb_idx] = torch.minimum(ub_ref[:, amb_idx], ub_new)
            ub_ref = torch.maximum(ub_ref, lb_ref)
            refined = Bounds(
                lb_ref.view_as(b.lb).clone(),
                ub_ref.view_as(b.ub).clone(),
            )
            out[lid] = refined
            if pred_lid in out and out[pred_lid].lb.shape == refined.lb.shape:
                out[pred_lid] = refined
        return out

    def recompute_bounds_and_nu(
        self,
        net: Net,
        bounds_dict: Dict[int, Bounds],
        c: torch.Tensor,
        M: int,
        alpha_state: Optional[Dict[int, torch.Tensor]] = None,
        eta_state: Optional[Dict[int, torch.Tensor]] = None,
        split_signs: Optional[Dict[int, torch.Tensor]] = None,
        per_class_alpha: bool = True,
    ) -> Tuple[Dict[int, Bounds], Optional[Dict[int, torch.Tensor]]]:
        """Forward bounds and per-RELU ν at the converged (α, η), from one pass.

        BaBSR/FSB scoring pairs each RELU's slope/intercept (from interval bounds)
        with its backward multiplier ν. Both MUST come from the same forward pass or
        the heuristic mixes an un-optimized interval with an optimized multiplier.
        Returns ``(fresh_bounds, nu_per_layer)``; soundness is unaffected (the
        certified bound is produced separately — this output is heuristic-only).
        """
        from act.back_end.dual_tf.tf_forward import compute_forward_bounds

        device, dtype = c.device, c.dtype
        input_lid = self._find_input_layer_id(net)
        if input_lid is None:
            return bounds_dict, None
        input_bounds = bounds_dict.get(input_lid)
        if input_bounds is None:
            return bounds_dict, None
        input_lb = input_bounds.lb.to(device=device, dtype=dtype)
        input_ub = input_bounds.ub.to(device=device, dtype=dtype)

        with torch.no_grad():
            forward_alphas = (
                {lid: (a[:, 0, :] if a.dim() == 3 else a) for lid, a in alpha_state.items()}
                if alpha_state
                else None
            )
            fresh_bounds = compute_forward_bounds(
                net, input_lb, input_ub, post_activation=False, alphas=forward_alphas,
            )
            result = self.compute_certified_bound(
                net,
                fresh_bounds,
                c,
                M=M,
                enable_grad=False,
                alpha=alpha_state if alpha_state else None,
                eta=eta_state if eta_state else None,
                split_signs=split_signs,
                return_nu_per_layer=True,
            )
        return fresh_bounds, result.nu_per_layer

    def evaluate_spec(
        self, net: Net,
        out_spec: OutputSpec,
        bounds_dict: Optional[Dict[int, Bounds]] = None,
        num_classes: Optional[int] = None,
        chunk_size: Optional[int] = None,
        enable_grad: bool = False,
        collect_bounds: bool = False,
    ) -> SpecBatchResult:
        """Dual bound evaluation for any OutputSpec — self-contained entry point.

        Refactor note: ``bounds_dict`` is optional. When omitted (the typical
        case), the solver gathers the net's INPUT_SPEC seed bounds and computes
        per-layer pre-activation forward bounds internally via
        ``compute_forward_bounds(post_activation=False)``. Callers who already
        have a bounds_dict (e.g. BaB refinement loops) may pass it explicitly to
        skip the recomputation. When ``collect_bounds`` is true, the solver
        stores the same dict on ``last_forward_bounds`` so post-verification
        soundness checks validate exactly the bounds used by the dual certificate.

        Strategy: dispatch on ``out_spec.kind`` into two branches that share
        ``compute_certified_bound`` but use opposite sign conventions and
        opposite row aggregators.

        - ALL-rows kinds (LINEAR_LE, TOP1_ROBUST, MARGIN_ROBUST, RANGE):
          ``encode_linear`` emits (C, thresholds) in UB-cert form (CERTIFIED
          iff ``UB(C @ y) < threshold``). Pass ``-C`` / ``-thresholds`` to
          ``compute_certified_bound`` and compare; ``slack >= 0`` means the
          row passes. Certified iff every row passes (``.all()``).
        - EXISTS-row kind (UNSAFE_LINEAR): the unsafe polytope is
          ``P = {y : c_i^T y <= d_i for ALL i}``. SAFE iff for all reachable
          y, some row i satisfies ``c_i^T y > d_i`` (escape). Sound
          strengthening via quantifier swap (mirrors ``verifier.py:574-580``):
          certify SAFE iff there exists a row i with ``LB_dual(c_i^T y) > d_i``.
          ``encode_linear`` emits UNSAFE_LINEAR in LB-cert form, so pass ``+C``
          / ``+thresholds`` directly (no sign flip). Certified iff any row
          escapes (``.any()``).

        Raises:
            ValueError: if net lacks ASSERT layer, ASSERT has != 1 predecessor,
                or (when bounds_dict is supplied) the output layer's bounds are
                missing / unbatched.
        """
        if bounds_dict is None:
            from act.back_end.dual_tf.tf_forward import compute_forward_bounds
            from act.back_end.verifier import (
                gather_input_spec_layers,
                seed_from_input_specs,
            )
            spec_layers = gather_input_spec_layers(net)
            seed_bounds = seed_from_input_specs(spec_layers)
            bounds_dict = compute_forward_bounds(
                net, seed_bounds.lb, seed_bounds.ub, post_activation=False,
            )

        if collect_bounds:
            self.last_forward_bounds = bounds_dict

        sample = next(iter(bounds_dict.values()))
        device = sample.lb.device
        dtype = sample.lb.dtype
        if sample.lb.dim() < 2:
            raise ValueError(
                "DualSolver.evaluate_spec: bounds_dict entries must be batched "
                f"[B, *shape]; got dim={sample.lb.dim()}"
            )
        B = sample.lb.shape[0]

        assert_layer = None
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k == LayerKind.ASSERT.value:
                assert_layer = layer
                break
        if assert_layer is None:
            raise ValueError("DualSolver.evaluate_spec: net has no ASSERT layer")
        assert_preds = net.preds.get(assert_layer.id, [])
        if len(assert_preds) != 1:
            raise ValueError(
                f"ASSERT layer must have exactly 1 predecessor, got {len(assert_preds)}"
            )
        output_lid = assert_preds[0]
        if output_lid not in bounds_dict:
            raise ValueError(
                f"DualSolver.evaluate_spec: bounds_dict missing output layer "
                f"{output_lid} (ASSERT predecessor); run forward analysis first."
            )
        out_bounds = bounds_dict[output_lid]
        if out_bounds.lb.dim() < 2:
            raise ValueError(
                f"DualSolver.evaluate_spec: output layer {output_lid} bounds "
                f"must be batched; got dim={out_bounds.lb.dim()}"
            )
        n_out = int(out_bounds.lb.flatten(start_dim=1).shape[-1])

        if out_spec.kind == OutKind.UNSAFE_LINEAR:
            # EXISTS-row branch. encode_linear emits LB-cert form for
            # UNSAFE_LINEAR (specs.py:179-201) — pass +C / +thresholds
            # directly. Certified iff any row escapes the unsafe polytope.
            # Slack semantics is ASYMMETRIC vs ALL-rows kinds below:
            # here ``slack > 0`` means the row certifies; ``min_slack`` is
            # NOT a meaningful summary (use ``slack.max(dim=-1)`` instead).
            fe_params = out_spec.encode_linear(B=B, n_out=n_out, device=device, dtype=dtype)
            C = fe_params["C"].contiguous()
            thresholds = fe_params["thresholds"].contiguous()
            N = int(fe_params["M"])
            active_mask = torch.ones(B, N, dtype=torch.bool, device=device)

            with torch.set_grad_enabled(enable_grad):
                if chunk_size is None or N <= chunk_size:
                    result = self.compute_certified_bound(
                        net, bounds_dict, C, M=N, enable_grad=enable_grad,
                    )
                    margins_flat = result.margins
                else:
                    margins_flat = self._chunked_eval(
                        net, bounds_dict, C, B, N, n_out, chunk_size, enable_grad,
                    )
                margins = margins_flat.view(B, N)
                slack = margins - thresholds
                certified = ((slack > 0) & active_mask).any(dim=-1)

            return SpecBatchResult(
                margins=margins,
                slack=slack,
                active_mask=active_mask,
                certified=certified,
            )

        fe_params = out_spec.encode_linear(B=B, n_out=n_out, device=device, dtype=dtype)
        C_neg = -fe_params["C"].contiguous()
        thresholds_neg = -fe_params["thresholds"].contiguous()
        M = int(fe_params["M"])
        active_mask = torch.ones(B, M, dtype=torch.bool, device=device)

        with torch.set_grad_enabled(enable_grad):
            if chunk_size is None or M <= chunk_size:
                result = self.compute_certified_bound(
                    net, bounds_dict, C_neg, M=M, enable_grad=enable_grad,
                )
                margins_flat = result.margins
            else:
                margins_flat = self._chunked_eval(
                    net, bounds_dict, C_neg, B, M, n_out, chunk_size, enable_grad,
                )

            margins = margins_flat.view(B, M)
            slack = margins - thresholds_neg
            violations = (slack < 0) & active_mask
            certified = ~violations.any(dim=-1)

        return SpecBatchResult(
            margins=margins,
            slack=slack,
            active_mask=active_mask,
            certified=certified,
        )

    def _chunked_eval(
        self, net: Net, bounds_dict: Dict[int, Bounds],
        C_neg: torch.Tensor, B: int, M: int, n_out: int,
        chunk_size: int, enable_grad: bool,
    ) -> torch.Tensor:
        """Evaluate sign-flipped C in chunks along the M dimension.

        For large M (e.g. CIFAR-100 K=100), trades time for memory by
        processing chunk_size specs per sample at a time.

        Chunked evaluation invariant: slicing the leading B*M axis at arbitrary
        chunk_size is bit-identical to unchunked evaluation, because each
        (sample, spec) row is fully independent in the dual backward pass — no
        cross-row computation exists within a chunk or across chunk boundaries.
        """
        C_view = C_neg.view(B, M, n_out)
        chunks: List[torch.Tensor] = []
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            m_chunk = end - start
            # Slice specs [start:end] for all B samples — independent rows, invariant-safe.
            C_chunk = C_view[:, start:end, :].reshape(B * m_chunk, n_out).contiguous()
            result = self.compute_certified_bound(
                net, bounds_dict, C_chunk, M=m_chunk, enable_grad=enable_grad,
            )
            chunks.append(result.margins.view(B, m_chunk))
        return torch.cat(chunks, dim=1).reshape(B * M)

    def compute_robust_bound(
        self, net: Net, bounds_dict: Dict[int, Bounds],
        y_true: Union[int, torch.Tensor], num_classes: int,
        margin: float = 0.0,
        return_full: bool = False,
        enable_grad: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], SpecBatchResult]:
        """Dual certified robust bound for classification (top-1 or margin).

        Unified via evaluate_spec(). Retained as a first-class API for robust
        training loops and existing verification callers.

        Args:
            net: the ACT Net with an ASSERT layer.
            bounds_dict: layer bounds from forward analysis.
            y_true: [B] true class labels, or scalar for uniform label.
            num_classes: K (output dim of network's ASSERT predecessor).
            margin: if > 0 use MARGIN_ROBUST semantics (require y_t - y_j >= margin);
                    else use TOP1_ROBUST (require y_t - y_j >= 0).
            return_full: if True, return the full SpecBatchResult (has per-class
                         [B, K] margins useful for training losses). If False,
                         return legacy tuple (min_slack: Tensor[B], certified: Tensor[B] bool).
            enable_grad: if True, allow gradients to flow through the computation
                         (for robust training). Default False (inference/verification).

        Returns:
            SpecBatchResult if return_full else (Tensor[B], Tensor[B] bool).
        """
        sample = next(iter(bounds_dict.values()))
        device = sample.lb.device
        if isinstance(y_true, int):
            B = sample.lb.shape[0] if sample.lb.dim() >= 2 else 1
            y_true_t = torch.full((B,), y_true, dtype=torch.long, device=device)
        else:
            y_true_t = y_true.to(device=device, dtype=torch.long)

        kind = OutKind.MARGIN_ROBUST if margin > 0 else OutKind.TOP1_ROBUST
        out_spec = OutputSpec(
            kind=kind,
            y_true=y_true_t,
            margin=(
                torch.as_tensor([margin], device=device, dtype=sample.lb.dtype)
                if margin > 0
                else None
            ),
        )
        result = self.evaluate_spec(
            net, out_spec,
            bounds_dict=bounds_dict,
            num_classes=num_classes,
            enable_grad=enable_grad,
        )
        if return_full:
            return result
        return result.min_slack, result.certified

    def _find_input_layer_id(self, net: Net) -> Optional[int]:
        """Return the INPUT_SPEC layer id if present, else INPUT's id, else None."""
        input_spec_id = None
        input_id = None
        for layer in net.layers:
            k = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if k == LayerKind.INPUT_SPEC.value:
                input_spec_id = layer.id
            elif k == LayerKind.INPUT.value:
                input_id = layer.id
        return input_spec_id if input_spec_id is not None else input_id

    def _input_contribution_from_nu(self, net: Net, input_lid: int,
                                    nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                                    M: int = 1,
                                    return_sce: bool = False,
                                    enable_grad: bool = False
                                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Exact dual lower bound of ``nu @ x`` over the input region.

        For a box / L_inf spec (``p_norm`` unset or ``inf``) this is the box
        concretization ``lb·[nu]_+ + ub·[nu]_-``. For a finite-p LP_EMBEDDING
        spec it is the closed-form per-perturbed-word dual norm
        ``nu·center − Σ_block ‖half_block ⊙ nu_block‖_q`` (``q`` the Hölder dual
        of ``p``), which coincides with the box result exactly at p=inf, q=1.

        Lazy M-broadcast: ``nu`` has leading dim ``B*M`` (sample-major)
        while batched ``bounds_dict[input_lid]`` is ``[B, *shape]``. The
        contribution per (b, m) reuses the same bounds for all m via
        ``[B, 1, n]`` broadcast against ``[B, M, n]``. Bit-identical to
        legacy M-expanded path.

        The unbatched (``lb.dim() < 2``) and missing-bounds (lb/ub from
        ``input_layer.params``) paths are preserved: they broadcast a single
        ``[n]`` tensor against ``[BM, n]`` nu — the same as legacy with B=BM.
        """
        with torch.set_grad_enabled(enable_grad):
            BM = nu.shape[0]
            assert BM % M == 0, (
                f"_input_contribution_from_nu: nu batch {BM} not divisible by M={M}"
            )
            B = BM // M
            input_layer = net.by_id[input_lid]

            bounds = bounds_dict.get(input_lid)
            if bounds is None:
                if "lb" in input_layer.params and "ub" in input_layer.params:
                    lb = cast(torch.Tensor, input_layer.params["lb"])
                    ub = cast(torch.Tensor, input_layer.params["ub"])
                else:
                    raise ValueError(
                        f"_input_contribution_from_nu: input layer {input_lid} has no "
                        f"bounds in bounds_dict and no lb/ub params"
                    )
            else:
                lb = bounds.lb
                ub = bounds.ub

            # A finite input Lp ball (LP_EMBEDDING) is concretized by its exact
            # per-block dual norm; p=inf (or unset) falls through to the box
            # path below, bit-identical for every current vision / box spec.
            p_norm = _resolve_perturbation_norm(input_layer.params.get("p_norm"))
            if p_norm != float("inf"):
                return self._dual_norm_contribution(
                    input_layer, lb, ub, nu, B, M,
                    q=_dual_norm_exponent(p_norm),
                    return_sce=return_sce,
                )

            orig_shape = lb.shape
            v_flat = nu.flatten(start_dim=1)                       # [BM, n_in]

            if lb.dim() < 2:
                lb_b = lb.flatten().unsqueeze(0).expand(BM, -1)
                ub_b = ub.flatten().unsqueeze(0).expand(BM, -1)
                n = min(v_flat.shape[-1], lb_b.shape[-1])
                if v_flat.shape[-1] != lb_b.shape[-1]:
                    lb_b, ub_b, v_flat = lb_b[..., :n], ub_b[..., :n], v_flat[..., :n]
                assert (lb_b <= ub_b).all(), "Invalid input bounds: lb > ub"
                contrib = ((lb_b * v_flat.clamp(min=0)).sum(dim=-1)
                           + (ub_b * v_flat.clamp(max=0)).sum(dim=-1))
                sce = None
                if return_sce:
                    sce_flat = torch.where(v_flat > 0, lb_b, ub_b)
                    if sce_flat.shape[-1] == lb.flatten().numel():
                        sce = sce_flat.view(BM, *orig_shape)
                    else:
                        sce = sce_flat
                return contrib, sce

            lb_B = lb.flatten(start_dim=1)                         # [B, n_in]
            ub_B = ub.flatten(start_dim=1)                         # [B, n_in]
            n = min(v_flat.shape[-1], lb_B.shape[-1])
            if v_flat.shape[-1] != lb_B.shape[-1]:
                lb_B = lb_B[..., :n]
                ub_B = ub_B[..., :n]
                v_flat = v_flat[..., :n]
            assert (lb_B <= ub_B).all(), "Invalid input bounds: lb > ub"

            v = v_flat.view(B, M, n)                               # [B, M, n] view
            lb_bc = lb_B.unsqueeze(1)                              # [B, 1, n]
            ub_bc = ub_B.unsqueeze(1)                              # [B, 1, n]
            contrib_BM = ((lb_bc * v.clamp(min=0)).sum(dim=-1)
                          + (ub_bc * v.clamp(max=0)).sum(dim=-1))  # [B, M]
            contrib = contrib_BM.view(BM)

            sce = None
            if return_sce:
                sce_BMn = torch.where(v > 0, lb_bc, ub_bc)         # [B, M, n]
                sce_flat = sce_BMn.view(BM, n)
                total = int(torch.tensor(orig_shape[1:]).prod().item())
                sce = sce_flat.view(BM, *orig_shape[1:]) if sce_flat.shape[-1] == total else sce_flat
            return contrib, sce

    def _dual_norm_contribution(self, input_layer: Layer,
                                lb: torch.Tensor, ub: torch.Tensor,
                                nu: torch.Tensor, B: int, M: int,
                                q: float, return_sce: bool
                                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Per-perturbed-word-block dual-norm input contribution (finite Lp).

        Evaluates the inner minimization in closed form:
        ``nu·center − Σ_block ‖half_block ⊙ nu_block‖_q`` with
        ``center = (lb+ub)/2`` and ``half = (ub−lb)/2``. ``half`` equals the
        radius ε on perturbed coordinates and 0 on degenerate (non-perturbed)
        ones, so the latter contribute exactly ``nu·center`` with no penalty.
        Each embedding block is normed independently, so word balls stay
        decoupled. The dual norm is the exact value of ``min`` over the Lp ball,
        so no second-order-cone constraint is needed (q=1↔p=inf reproduces the
        box result, which the caller already routes through the box path).

        Args:
            input_layer: INPUT/INPUT_SPEC layer carrying ``p_norm`` and the
                optional ``perturbed_positions`` / ``embed_dim`` block metadata.
            lb: Input lower bounds, ``[B, *shape]`` (batched) or ``[*shape]``.
            ub: Input upper bounds, matching ``lb``.
            nu: Backward coefficient ``[B*M, *shape]`` (sample-major).
            B: Number of samples (``B*M == nu.shape[0]``).
            M: Spec rows per sample (lazy-M-broadcast factor).
            q: Hölder dual exponent of the spec norm ``p``.
            return_sce: Whether to also return the worst-case input witness.

        Returns:
            ``(contrib, sce)`` with ``contrib`` shape ``[B*M]`` and ``sce`` the
            ball minimizer (``None`` when ``return_sce`` is False).
        """
        orig_shape = lb.shape
        v_flat = nu.flatten(start_dim=1)                          # [BM, n_in]
        batched = lb.dim() >= 2
        if batched:
            lb_f = lb.flatten(start_dim=1)                        # [B, n_in]
            ub_f = ub.flatten(start_dim=1)
        else:
            lb_f = lb.flatten().unsqueeze(0)                      # [1, n_in]
            ub_f = ub.flatten().unsqueeze(0)
        n = min(v_flat.shape[-1], lb_f.shape[-1])
        if v_flat.shape[-1] != lb_f.shape[-1]:
            lb_f, ub_f, v_flat = lb_f[..., :n], ub_f[..., :n], v_flat[..., :n]
        assert (lb_f <= ub_f).all(), "Invalid input bounds: lb > ub"

        center = (lb_f + ub_f) * 0.5                              # [B|1, n]
        half = (ub_f - lb_f) * 0.5
        BM = v_flat.shape[0]
        blocks = self._perturbed_block_slices(input_layer.params, orig_shape, n)

        if batched:
            v = v_flat.reshape(B, M, n)                           # [B, M, n]
            center_bc = center.unsqueeze(1)                       # [B, 1, n]
            half_bc = half.unsqueeze(1)
            dot = (center_bc * v).sum(dim=-1)                     # [B, M]
            penalty = torch.zeros_like(dot)
            block_eps = input_layer.params.get("bab_block_eps")
            if isinstance(block_eps, torch.Tensor):
                eps_b = block_eps.to(device=v.device, dtype=v.dtype)
                for block_idx, (s, e) in enumerate(blocks):
                    penalty = penalty + eps_b[:, block_idx].unsqueeze(1) * torch.linalg.vector_norm(
                        v[..., s:e], ord=q, dim=-1
                    )
            else:
                for s, e in blocks:
                    penalty = penalty + torch.linalg.vector_norm(
                        half_bc[..., s:e] * v[..., s:e], ord=q, dim=-1
                    )
            contrib = (dot - penalty).reshape(BM)
        else:
            dot = (center * v_flat).sum(dim=-1)                   # [BM]
            penalty = torch.zeros_like(dot)
            for s, e in blocks:
                penalty = penalty + torch.linalg.vector_norm(
                    half[..., s:e] * v_flat[..., s:e], ord=q, dim=-1
                )
            contrib = dot - penalty

        sce = None
        if return_sce:
            sce = self._dual_norm_sce(
                center, half, v_flat, blocks, q, M, orig_shape, batched
            )
        return contrib, sce

    @staticmethod
    def _perturbed_block_slices(params: Dict[str, Any], orig_shape: torch.Size,
                                n: int) -> List[Tuple[int, int]]:
        """Coordinate ranges ``[start, end)`` of each per-word embedding block.

        Each word occupies ``embed_dim`` contiguous embedding coordinates, so the
        dual norm is taken over those ``D`` coordinates independently and word
        balls do not couple. Splitting every token (not only perturbed ones) is
        sound and format-agnostic: a non-perturbed token has zero half-width, so
        its block penalty ``‖0 ⊙ nu_block‖_q`` is exactly 0 — the box already
        encodes which coordinates carry width, so ``perturbed_positions`` (index
        list or bool mask, possibly per-sample) need not be parsed here. The block
        size is read from ``embed_dim`` or, failing that, the trailing
        ``[..., L, D]`` axis when ``perturbed_positions`` flags an embedding spec.
        Otherwise the whole input is one Lp ball (e.g. an image L2 spec).
        """
        embed_dim = params.get("embed_dim")
        if embed_dim is None:
            if params.get("perturbed_positions") is not None and len(orig_shape) >= 2:
                embed_dim = int(orig_shape[-1])
            else:
                return [(0, n)]
        d = int(embed_dim)
        if d <= 0 or n % d != 0:
            return [(0, n)]
        return [(i * d, (i + 1) * d) for i in range(n // d)]

    def _dual_norm_sce(self, center: torch.Tensor, half: torch.Tensor,
                       v_flat: torch.Tensor, blocks: List[Tuple[int, int]],
                       q: float, M: int, orig_shape: torch.Size,
                       batched: bool) -> torch.Tensor:
        """Worst-case input on the per-block Lp ball that attains the bound.

        The minimizer is ``center + δ*`` where, per block, ``δ*`` is the Hölder
        witness of ``‖half ⊙ nu‖_q``: the scaled-L2 ray ``−(half²⊙nu)/‖half⊙nu‖₂``
        for q=2 and a single max-coordinate spike for q=inf. Both lie on the ball
        boundary, so the witness is a sound counterexample candidate (the box
        corner is not, as it leaves the Lp ball). Non-perturbed coordinates stay
        at center (zero width).
        """
        BM = v_flat.shape[0]
        if batched:
            center_bm = center.repeat_interleave(M, dim=0)        # [B,n] -> [BM,n]
            half_bm = half.repeat_interleave(M, dim=0)
        else:
            center_bm = center.expand(BM, -1)                     # [1,n] -> [BM,n]
            half_bm = half.expand(BM, -1)
        sce_flat = center_bm.clone()
        for s, e in blocks:
            nu_b = v_flat[:, s:e]                                  # [BM, bs]
            h_b = half_bm[:, s:e]
            if q == 2.0:
                hn = h_b * nu_b
                norm = torch.linalg.vector_norm(hn, ord=2, dim=-1, keepdim=True)
                delta = -(h_b * hn) / norm.clamp_min(_DUAL_NORM_EPS)
            elif q == float("inf"):
                idx = (h_b * nu_b).abs().argmax(dim=-1, keepdim=True)
                spike = -torch.sign(nu_b.gather(-1, idx)) * h_b.gather(-1, idx)
                delta = torch.zeros_like(h_b)
                delta.scatter_(-1, idx, spike)
            else:
                delta = torch.zeros_like(h_b)
            sce_flat[:, s:e] = center_bm[:, s:e] + delta
        if batched:
            tail = orig_shape[1:]
            total = int(torch.tensor(tail).prod().item()) if len(tail) else 1
            return sce_flat.view(BM, *tail) if sce_flat.shape[-1] == total else sce_flat
        total = int(torch.tensor(orig_shape).prod().item())
        return sce_flat.view(BM, *orig_shape) if sce_flat.shape[-1] == total else sce_flat
