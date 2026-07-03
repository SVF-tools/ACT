#===- act/front_end/specs.py - Specification Data Types ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Defines InputSpec and OutputSpec data structures for verification
#   specifications including safety, robustness, and constraint types.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional
import torch

ScalarTensor = torch.Tensor | float | int

def normalize_position_mask(
    positions: "torch.Tensor | Sequence[int] | Sequence[bool] | None",
    token_count: int,
    *,
    batch_shape: "tuple[int, ...]" = (),
    device: "torch.device | None" = None,
) -> torch.Tensor:
    """Normalize an LP_EMBEDDING ``perturbed_positions`` field to a boolean
    token mask of shape ``(*batch_shape, token_count)``.

    Single source of truth for every consumer (specs / verifier / bab / node /
    branching / certify): ``None`` selects all positions; a boolean tensor is
    accepted at the exact mask shape, as a per-token vector broadcast over
    ``batch_shape``, or as a singleton-batch row; integer indices are
    bounds-checked and scattered onto the last axis.

    Raises:
        ValueError: If a boolean mask shape is incompatible or an integer
            index is out of range for ``token_count``.
    """
    shape = (*tuple(int(b) for b in batch_shape), int(token_count))
    if positions is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    pos = positions if isinstance(positions, torch.Tensor) else torch.as_tensor(positions)
    if device is not None:
        pos = pos.to(device=device)
    if pos.dtype == torch.bool:
        if tuple(pos.shape) == shape:
            return pos
        if pos.numel() == token_count:
            view = [1] * len(shape)
            view[-1] = token_count
            return pos.reshape(view).expand(shape)
        if len(shape) > 1 and pos.dim() == len(shape) and pos.shape[-1] == token_count and pos.shape[0] == 1:
            return pos.expand(shape)
        raise ValueError(
            f"boolean perturbed_positions shape {tuple(pos.shape)} incompatible with mask shape {shape}"
        )
    idx = pos.to(dtype=torch.long).flatten()
    mask = torch.zeros(shape, dtype=torch.bool, device=pos.device)
    if idx.numel() == 0:
        return mask
    if bool((idx < 0).any()) or bool((idx >= token_count).any()):
        raise ValueError(
            f"perturbed_positions contains out-of-range token index for length {token_count}"
        )
    mask.index_fill_(-1, idx, True)
    return mask


class InKind:
    """Supported input-constraint kinds for :class:`InputSpec`.

    Members:
        BOX: Axis-aligned box ``lb <= x <= ub`` directly on the input.
        LINF_BALL: L-infinity ball ``|x - center| <= eps`` around a clean input.
        LIN_POLY: Linear polytope ``A x <= b`` over the input variables.
        LP_EMBEDDING: Lp-norm ball ``||x - center||_p <= eps`` on the selected
            embedding-space token positions (``perturbed_positions``).
    """
    # Image/input perturbation regime: perturbation is applied directly in
    # pixel/feature space (the raw model input x in R^n), not in embedding space.
    #
    #   LINF_BALL:  { x : ||x - c||_inf <= eps }
    #               <=>  x_i in [c_i - eps, c_i + eps]  for every input coordinate i
    #   BOX:        { x : lb_i <= x_i <= ub_i }  (explicit per-coordinate box)
    #
    # Every pixel/feature moves freely within eps of the center c (LINF_BALL),
    # or within its explicit per-coordinate interval (BOX).
    # ONE ball (or box) over ALL input coordinates — perturbs pixels/features.
    BOX = "BOX"          # Axis-aligned box lb <= x <= ub on the input features.
    LINF_BALL = "LINF_BALL"  # L-infinity ball |x - center|_inf <= eps on the input features.
    LIN_POLY = "LIN_POLY"
    # Per-position embedding perturbation: selected token/patch i satisfy ||x_i - center_i||_p <= eps (p = p_norm).
    # perturbed_positions is a boolean position mask or integer index list; None selects all positions.
    # Unselected positions are pinned to center.
    # Analysis seeds the enclosing box; finite-p tightness is recovered by dual per-position input terms.
    LP_EMBEDDING = "LP_EMBEDDING"
    SYNONYM_SUB = "SYNONYM_SUB"

@dataclass
class InputSpec:
    """Input perturbation specification for a single verification instance.

    Supports two perturbation regimes:

    * **Image/input perturbation** (``BOX``, ``LINF_BALL``): perturbation is
      applied directly in pixel or raw-feature space (x in R^n).

      - LINF_BALL:  { x : ||x - center||_inf <= eps }
                    <=>  x_i in [center_i - eps, center_i + eps]  for every i
      - BOX:        { x : lb_i <= x_i <= ub_i }  (explicit per-coordinate box)

      ONE ball (or box) over ALL input coordinates — perturbs pixels/features.

    * **Embedding/position perturbation** (``LP_EMBEDDING``): perturbation is
      applied in the embedding space of a transformer model (BERT, ViT).
      For embeddings e in R^{L x d} with ``center`` c and
      P = ``perturbed_positions``, the admissible set is:

          { e :  ||e_t - c_t||_{p_norm} <= eps   for t in P,
                 e_t = c_t                         for t not in P }

      A SEPARATE per-token Lp ball of radius ``eps`` in the d-dimensional
      embedding space at each SELECTED token position t in P; all OTHER
      positions are PINNED to ``center``.

      Contrast: image = ONE ball over all input coordinates (perturbs pixels);
      LP_EMBEDDING = localized per-token balls at chosen sequence positions in
      EMBEDDING space (rest fixed) — perturbs token embeddings, not pixels.
    """
    kind: str
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    center: Optional[torch.Tensor] = None
    eps: Optional[ScalarTensor] = None
    A: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None
    p_norm: ScalarTensor = float("inf")
    perturbed_positions: torch.Tensor | Sequence[int] | Sequence[bool] | None = None
    budget: torch.Tensor | int | None = None
    synonym_table: Optional[Any] = None
    
    def __post_init__(self):
        """Ensure all numeric fields are tensors for architecture."""
        # Convert eps (scalar → 1-D tensor)
        if self.eps is not None and not isinstance(self.eps, torch.Tensor):
            self.eps = torch.tensor([float(self.eps)])

        if self.p_norm is not None and not isinstance(self.p_norm, torch.Tensor):
            self.p_norm = torch.tensor([float(self.p_norm)])

        if self.budget is not None and not isinstance(self.budget, torch.Tensor):
            self.budget = torch.tensor([int(self.budget)], dtype=torch.int64)

        if (
            self.perturbed_positions is not None
            and not isinstance(self.perturbed_positions, torch.Tensor)
        ):
            self.perturbed_positions = torch.tensor(self.perturbed_positions)
        
        # Convert lb, ub, center (list or scalar → tensor)
        for field in ['lb', 'ub', 'center']:
            val = getattr(self, field, None)
            if val is not None and not isinstance(val, torch.Tensor):
                if isinstance(val, (list, tuple)):
                    setattr(self, field, torch.tensor(val))
                else:
                    setattr(self, field, torch.tensor([float(val)]))
        
        # Convert A, b (list → tensor, keep None as is)
        for field in ['A', 'b']:
            val = getattr(self, field, None)
            if val is not None and not isinstance(val, torch.Tensor):
                if isinstance(val, (list, tuple)):
                    setattr(self, field, torch.tensor(val))

    def materialize_box_seed(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize the box seed used by interval and LP back ends.

        For ``LP_EMBEDDING`` the seed perturbs only the ``perturbed_positions``
        tokens by ``eps`` (under ``p_norm``) and pins every other position to
        ``center``; other kinds use their natural box.

        Args:
            None.

        Returns:
            A pair ``(lb, ub)`` whose tensors match the input shape.

        Raises:
            ValueError: If fields required by the input kind are missing or
                shape-incompatible.
            NotImplementedError: If the input kind does not define a box seed.
        """
        if self.kind == InKind.BOX:
            if self.lb is None or self.ub is None:
                raise ValueError("BOX requires both lb and ub")
            return self.lb.clone(), self.ub.clone()

        if self.kind == InKind.LINF_BALL:
            if self.center is None or self.eps is None:
                raise ValueError("LINF_BALL requires both center and eps")
            eps = self._eps_like(self.center)
            return self.center - eps, self.center + eps

        if self.kind == InKind.LP_EMBEDDING:
            if self.center is None or self.eps is None:
                raise ValueError("LP_EMBEDDING requires both center and eps")
            lb = self.center.clone()
            ub = self.center.clone()
            mask = self._embedding_position_mask(self.center)
            expanded_mask = mask.unsqueeze(-1).expand_as(self.center)
            eps = self._eps_like(self.center)
            full_lb = self.center - eps
            full_ub = self.center + eps
            lb = torch.where(expanded_mask, full_lb, lb)
            ub = torch.where(expanded_mask, full_ub, ub)
            return lb, ub

        raise NotImplementedError(
            f"Input kind {self.kind!r} does not define a box seed"
        )

    def _eps_like(self, center: torch.Tensor) -> torch.Tensor:
        """Normalize epsilon to the clean input tensor layout.

        Args:
            center: Clean input tensor that supplies target device and dtype.

        Returns:
            Epsilon tensor on the same device and dtype as ``center``.

        Raises:
            ValueError: If epsilon is missing.
        """
        if self.eps is None:
            raise ValueError(f"{self.kind} requires eps")
        if isinstance(self.eps, torch.Tensor):
            return self.eps.to(device=center.device, dtype=center.dtype)
        return torch.tensor([float(self.eps)], device=center.device, dtype=center.dtype)

    def _embedding_position_mask(self, center: torch.Tensor) -> torch.Tensor:
        """Build a boolean token-position mask for embedding-space specs.

        Args:
            center: Clean embedding tensor with embedding dimension last.

        Returns:
            Boolean tensor with shape ``center.shape[:-1]``.

        Raises:
            ValueError: If ``center`` is not an embedding tensor or the provided
                positions cannot be applied to its token dimension.
        """
        if center.dim() < 2:
            raise ValueError("LP_EMBEDDING center must have at least [L, D] shape")
        return normalize_position_mask(
            self.perturbed_positions,
            int(center.shape[-2]),
            batch_shape=tuple(center.shape[:-2]),
            device=center.device,
        )


class OutKind:
    LINEAR_LE   = "LINEAR_LE"
    TOP1_ROBUST = "TOP1_ROBUST"
    MARGIN_ROBUST = "MARGIN_ROBUST"
    RANGE = "RANGE"
    UNSAFE_LINEAR = "UNSAFE_LINEAR"

@dataclass
class OutputSpec:
    SLICEABLE_PARAM_KEYS: ClassVar[tuple[str, ...]] = (
        "y_true", "margin", "c", "d", "lb", "ub", "C", "thresholds"
    )

    kind: str
    c: Optional[torch.Tensor] = None
    d: Optional[torch.Tensor] = None
    y_true: Optional[torch.Tensor] = None
    margin: Optional[torch.Tensor] = None
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Ensure all numeric fields are tensors for batch-native architecture."""
        # Convert y_true (int/list → tensor)
        if self.y_true is not None and not isinstance(self.y_true, torch.Tensor):
            if isinstance(self.y_true, (list, tuple)):
                self.y_true = torch.tensor(self.y_true, dtype=torch.int64)
            else:
                self.y_true = torch.tensor([int(self.y_true)], dtype=torch.int64)
        
        # Convert margin (scalar → 1-D tensor)
        if self.margin is not None and not isinstance(self.margin, torch.Tensor):
            self.margin = torch.tensor([float(self.margin)])
        
        # Convert c, d, lb, ub (list/tuple → tensor; scalar → 1-D tensor).
        # ``d`` is scalar for LINEAR_LE but a vector for UNSAFE_LINEAR, so it
        # joins the list-aware conversion path below.
        for field in ['c', 'd', 'lb', 'ub']:
            val = getattr(self, field, None)
            if val is not None and not isinstance(val, torch.Tensor):
                if isinstance(val, (list, tuple)):
                    setattr(self, field, torch.tensor(val))
                else:
                    setattr(self, field, torch.tensor([float(val)]))

    def encode_linear(
        self,
        B: int,
        n_out: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Any]:
        """Produce ASSERT-layer params with both high-level fields (BaB) and
        pre-encoded linear form (verify_once).

        Output dict layout (all numeric values are ``torch.Tensor`` with a
        leading batch axis of size B; B=1 is the trivial case):

            {
              "kind":       str,                # for BaB MILP dispatch
              "y_true":     Tensor[B] long,     # TOP1_ROBUST, MARGIN_ROBUST
              "margin":     Tensor[B] float,    # MARGIN_ROBUST only
              "c":          Tensor[B, n_out],   # LINEAR_LE
                            Tensor[B, N, n_out],# UNSAFE_LINEAR
              "d":          Tensor[B],          # LINEAR_LE
                            Tensor[B, N],       # UNSAFE_LINEAR
              "lb":         Tensor[B, n_out],   # RANGE (if lb given)
              "ub":         Tensor[B, n_out],   # RANGE (if ub given)
              "C":          Tensor[B*M, n_out], # for verify_once
              "thresholds": Tensor[B, M],       # for verify_once
              "M":          int,                # for verify_once
            }

        Encoding orientation is "violation form": for each row
        ``C[b*M+j, :]``, the lane ``(b, j)`` is CERTIFIED iff
        ``margin_max(C[b*M+j] @ y) < thresholds[b, j]``. This method is the
        single source of truth for the five-kind row layouts consumed by
        ``verifier.verify_once`` (back-end).

        Args:
            B: batch size. Must be >= 1.
            n_out: number of output classes / output dimension.
                For TOP1/MARGIN this is also K (num_classes).
            device: target tensor device.
            dtype: target floating tensor dtype (``y_true`` uses ``torch.long``
                regardless).

        Returns:
            A dict with the layout above. All tensors carry the leading B
            axis. ``M`` is an int — kind-dependent (1, K-1, N, n_out, or
            2*n_out).

        Raises:
            ValueError: if required dataclass fields are missing or
                shape-incompatible with ``B`` / ``n_out``.
            NotImplementedError: if ``self.kind`` is not one of the five
                supported kinds.
        """
        params: Dict[str, Any] = {"kind": self.kind}

        if self.kind == OutKind.LINEAR_LE:
            if self.c is None or self.d is None:
                raise ValueError("LINEAR_LE requires both c and d")
            c_full = self.c.to(device=device, dtype=dtype)
            c_vec = c_full.flatten()
            d_t = self.d.to(device=device, dtype=dtype).flatten()
            if c_vec.shape[0] == n_out and d_t.numel() == 1:
                d_scalar = float(d_t.item())
                c_batched = c_vec.unsqueeze(0).expand(B, -1).contiguous()
                d_batched = torch.full((B,), d_scalar, device=device, dtype=dtype)
                params["c"] = c_batched
                params["d"] = d_batched
                c_rows = c_batched
                thresholds = d_batched.unsqueeze(1)
                m_specs = 1
            elif c_full.dim() == 2 and c_full.shape[1] == n_out:
                # Conjunction c_i.y <= d_i for all i. Reuses UNSAFE_LINEAR's row
                # LAYOUT only; kind MUST stay LINEAR_LE (AND-certify/OR-falsify) —
                # relabeling to UNSAFE_LINEAR flips polarity and is unsound.
                n_rows = int(c_full.shape[0])
                if d_t.numel() == 1:
                    d_t = d_t.expand(n_rows).contiguous()
                elif d_t.numel() != n_rows:
                    raise ValueError(
                        f"LINEAR_LE: d length {d_t.numel()} != rows {n_rows}"
                    )
                params["c"] = c_full.unsqueeze(0).expand(B, -1, -1).contiguous()
                params["d"] = d_t.unsqueeze(0).expand(B, -1).contiguous()
                c_rows = c_full.unsqueeze(0).expand(B, -1, -1).reshape(
                    B * n_rows, n_out
                ).contiguous()
                thresholds = d_t.unsqueeze(0).expand(B, -1).contiguous()
                m_specs = n_rows
            else:
                raise ValueError(
                    f"LINEAR_LE: c shape {tuple(self.c.shape)} incompatible "
                    f"with n_out {n_out}"
                )

        elif self.kind == OutKind.UNSAFE_LINEAR:
            if self.c is None or self.d is None:
                raise ValueError("UNSAFE_LINEAR requires both c and d")
            c_mat = self.c.to(device=device, dtype=dtype)
            if c_mat.dim() == 1:
                c_mat = c_mat.unsqueeze(0)
            N = c_mat.shape[0]
            if c_mat.shape[1] != n_out:
                raise ValueError(
                    f"UNSAFE_LINEAR: c cols {c_mat.shape[1]} != n_out {n_out}"
                )
            d_vec = self.d.to(device=device, dtype=dtype).flatten()
            if d_vec.shape[0] != N:
                raise ValueError(
                    f"UNSAFE_LINEAR: d length {d_vec.shape[0]} != N {N}"
                )
            params["c"] = c_mat.unsqueeze(0).expand(B, -1, -1).contiguous()
            params["d"] = d_vec.unsqueeze(0).expand(B, -1).contiguous()
            c_rows = c_mat.unsqueeze(0).expand(B, -1, -1).reshape(
                B * N, n_out
            ).contiguous()
            thresholds = d_vec.unsqueeze(0).expand(B, -1).contiguous()
            m_specs = N

        elif self.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            if self.y_true is None:
                raise ValueError(f"{self.kind} requires y_true")
            K = n_out
            if K < 2:
                raise ValueError(
                    f"{self.kind}: requires K >= 2 classes, got K={K}"
                )
            y_true_t = self.y_true.to(
                device=device, dtype=torch.long
            ).reshape(-1)
            if y_true_t.numel() == 1 and B > 1:
                y_true_t = y_true_t.repeat(B)
            if y_true_t.numel() != B:
                raise ValueError(
                    f"{self.kind}: y_true length {y_true_t.numel()} != B {B}"
                )
            if (y_true_t < 0).any() or (y_true_t >= K).any():
                raise ValueError(
                    f"{self.kind}: y_true contains out-of-range index "
                    f"(must lie in [0, K)); y_true={y_true_t.tolist()}, K={K}"
                )
            params["y_true"] = y_true_t
            # TOP1/MARGIN share row structure: drop the y_true row entirely
            # (m_specs = K-1) rather than masking with an active_mask.
            m_specs = K - 1
            j_full = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)
            one_hot_y = torch.nn.functional.one_hot(
                y_true_t, num_classes=K
            ).bool()
            keep_mask = ~one_hot_y
            j_kept = j_full[keep_mask].view(B, m_specs)
            eye_K = torch.eye(K, device=device, dtype=dtype)
            e_true = eye_K[y_true_t]
            e_others = eye_K[j_kept]
            # Violation form: row = e_j - e_{y_true}.
            c_per_sample = e_others - e_true.unsqueeze(1)
            c_rows = c_per_sample.reshape(B * m_specs, K).contiguous()
            if self.kind == OutKind.TOP1_ROBUST:
                thresholds = torch.zeros(
                    B, m_specs, device=device, dtype=dtype
                )
            else:
                if self.margin is None:
                    raise ValueError(
                        "MARGIN_ROBUST requires margin; use TOP1_ROBUST for "
                        "zero-margin semantics."
                    )
                margin_t = self.margin.to(
                    device=device, dtype=dtype
                ).reshape(-1)
                if margin_t.numel() == 1 and B > 1:
                    margin_t = margin_t.repeat(B)
                if margin_t.numel() != B:
                    raise ValueError(
                        f"MARGIN_ROBUST: margin length {margin_t.numel()} "
                        f"!= 1 or B {B}"
                    )
                params["margin"] = margin_t
                thresholds = (-margin_t).unsqueeze(1).expand(
                    B, m_specs
                ).contiguous()

        elif self.kind == OutKind.RANGE:
            if self.lb is None and self.ub is None:
                raise ValueError("RANGE requires lb and/or ub")
            eye = torch.eye(n_out, device=device, dtype=dtype)
            rows: List[torch.Tensor] = []
            thresh_rows: List[torch.Tensor] = []
            lb_vec: Optional[torch.Tensor] = None
            ub_vec: Optional[torch.Tensor] = None
            if self.lb is not None:
                lb_vec = self.lb.to(device=device, dtype=dtype).flatten()
                if lb_vec.shape[0] != n_out:
                    raise ValueError(
                        f"RANGE: lb length {lb_vec.shape[0]} != n_out {n_out}"
                    )
                rows.append(-eye)
                thresh_rows.append(-lb_vec)
            if self.ub is not None:
                ub_vec = self.ub.to(device=device, dtype=dtype).flatten()
                if ub_vec.shape[0] != n_out:
                    raise ValueError(
                        f"RANGE: ub length {ub_vec.shape[0]} != n_out {n_out}"
                    )
                rows.append(eye)
                thresh_rows.append(ub_vec)
            both_sides = lb_vec is not None and ub_vec is not None
            if both_sides:
                # Interleave [-e_0, +e_0, -e_1, +e_1, ...].
                stacked = torch.stack(
                    [rows[0], rows[1]], dim=1
                ).reshape(2 * n_out, n_out)
                thresh_stacked = torch.stack(
                    [thresh_rows[0], thresh_rows[1]], dim=1
                ).reshape(2 * n_out)
            else:
                stacked = rows[0]
                thresh_stacked = thresh_rows[0]
            m_specs = 2 * n_out if both_sides else n_out
            c_rows = stacked.unsqueeze(0).expand(B, -1, -1).reshape(
                B * m_specs, n_out
            ).contiguous()
            thresholds = thresh_stacked.unsqueeze(0).expand(
                B, -1
            ).contiguous()
            if lb_vec is not None:
                params["lb"] = lb_vec.unsqueeze(0).expand(
                    B, -1
                ).contiguous()
            if ub_vec is not None:
                params["ub"] = ub_vec.unsqueeze(0).expand(
                    B, -1
                ).contiguous()

        else:
            raise NotImplementedError(
                f"Unsupported ASSERT kind: {self.kind!r}. Supported: "
                f"LINEAR_LE, UNSAFE_LINEAR, TOP1_ROBUST, MARGIN_ROBUST, "
                f"RANGE."
            )

        assert c_rows.shape == (B * m_specs, n_out), (
            f"C.shape={tuple(c_rows.shape)} != ({B * m_specs}, {n_out})"
        )
        assert thresholds.shape == (B, m_specs), (
            f"thresholds.shape={tuple(thresholds.shape)} != ({B}, {m_specs})"
        )
        params["C"] = c_rows
        params["thresholds"] = thresholds
        params["M"] = m_specs
        return params
