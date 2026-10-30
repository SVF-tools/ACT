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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, ClassVar, Dict, List, Optional
import torch

ScalarTensor = torch.Tensor | float | int


def lp_norm(value: torch.Tensor, p: float, *, keepdim: bool = False) -> torch.Tensor:
    """Scale before exponentiation so extreme valid norms do not underflow."""
    scale = value.abs().amax(dim=-1, keepdim=True)
    unit = value / torch.where(scale == 0, 1.0, scale)
    norm = torch.linalg.vector_norm(unit, ord=p, dim=-1, keepdim=True) * scale
    return norm if keepdim else norm.squeeze(-1)


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
    if pos.numel() and (pos.is_floating_point() or pos.is_complex()):
        raise ValueError("perturbed_positions must contain integer indices or booleans")
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


class InKind(str, Enum):
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


@dataclass(frozen=True)
class InputGeometry:
    """Canonical token balls; a child box never changes this root domain."""

    center: torch.Tensor
    eps: torch.Tensor
    mask: torch.Tensor
    p: float

    @property
    def q(self) -> float:
        """Holder dual exponent used by backend linear support."""
        return 1.0 if self.p == math.inf else math.inf if self.p == 1 else self.p / (self.p - 1)

    @property
    def embed_dim(self) -> int:
        """Width of one independent token ball."""
        return self.center.shape[-1]

    @property
    def radii(self) -> torch.Tensor:
        """Per-token radii, with zero radius at pinned positions."""
        return self.eps * self.mask.unsqueeze(-1)

    def box(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Enclose the root balls without changing their centers or radii."""
        radii = self.radii
        return self.center - radii, self.center + radii

    def snapshot(self) -> "InputGeometry":
        """Own detached tensors for a verifier cache, including nonleaf inputs."""
        return InputGeometry(
            self.center.detach().clone(), self.eps.detach().clone(),
            self.mask.clone(), self.p,
        )

    def to(self, reference: torch.Tensor) -> "InputGeometry":
        """Move one domain at the solve boundary, not at every layer."""
        if self.center.device == reference.device and self.center.dtype == reference.dtype:
            return self
        return InputGeometry(
            self.center.to(reference), self.eps.to(reference),
            self.mask.to(device=reference.device), self.p,
        )

    def contains(self, x: torch.Tensor, tol: float = 0.0) -> torch.Tensor:
        """Check every token without flattening selected tokens across lanes."""
        shaped = x.reshape(x.shape[0], *self.center.shape[1:])
        norms = lp_norm(shaped - self.center.to(x), self.p)
        return (norms <= self.radii.to(x).squeeze(-1) + tol).flatten(1).all(1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Radially restrict generated samples; not a counterexample validator."""
        shaped = x.reshape(x.shape[0], *self.center.shape[1:])
        center = self.center.to(x)
        delta = shaped - center
        norm = lp_norm(delta, self.p, keepdim=True)
        radii = self.radii.to(x)
        scale = (radii / torch.where(norm == 0, 1.0, norm)).clamp(max=1)
        # Move rounded boundary samples inward; at unrepresentable tiny radii,
        # the center is the guaranteed feasible representable sample.
        scale = torch.where(norm > radii, scale * (1 - 4 * torch.finfo(x.dtype).eps), scale)
        candidate = center + delta * scale
        inside = lp_norm(candidate - center, self.p, keepdim=True) <= radii
        return torch.where(inside, candidate, center).reshape_as(x)


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

      ``p_norm`` is shared across the batch. ``eps`` is a scalar or one
      radius per sample, not a separate radius for each token or coordinate.

      Contrast: image = ONE ball over all input coordinates (perturbs pixels);
      LP_EMBEDDING = localized per-token balls at chosen sequence positions in
      EMBEDDING space (rest fixed) — perturbs token embeddings, not pixels.
    """
    kind: InKind
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
    
    def __post_init__(self) -> None:
        """Tensorize fields and normalize finite-p input parameters."""
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

        if self.kind == InKind.LP_EMBEDDING:
            if not isinstance(self.p_norm, torch.Tensor):
                raise ValueError("LP_EMBEDDING requires p_norm")
            values = self.p_norm.reshape(-1)
            if values.numel() == 0:
                raise ValueError("LP_EMBEDDING p_norm must not be empty")
            p = float(values[0])
            if math.isnan(p) or p < 1:
                raise ValueError(f"LP_EMBEDDING p_norm must be at least 1, got {p}")
            if not bool((values == p).all()):
                raise ValueError("LP_EMBEDDING p_norm must be shared across batch lanes")
            self.p_norm = values[:1].clone()
            if self.center is not None and self.eps is not None:
                self.eps = self._eps_like(self.center)

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
            geometry = self.input_geometry()
            lb, ub = geometry.box()
            assert self.center is not None
            return lb.reshape_as(self.center), ub.reshape_as(self.center)

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
            ValueError: If epsilon is missing, or an LP radius is invalid.
        """
        if self.eps is None:
            raise ValueError(f"{self.kind} requires eps")
        if self.kind == InKind.LP_EMBEDDING and not center.is_floating_point():
            raise ValueError("LP_EMBEDDING center must have a floating-point dtype")
        if isinstance(self.eps, torch.Tensor):
            eps = self.eps.to(device=center.device, dtype=center.dtype)
        else:
            eps = center.new_tensor([float(self.eps)])
        if self.kind != InKind.LP_EMBEDDING:
            return eps
        if not bool(torch.isfinite(eps).all()) or bool((eps < 0).any()):
            raise ValueError("LP_EMBEDDING eps must be finite and non-negative")
        if eps.numel() == 1:
            return eps.reshape((1,) * center.dim())
        batch_shape = tuple(center.shape[:-2])
        if tuple(eps.shape) not in (
            batch_shape, (*batch_shape, 1), (*batch_shape, 1, 1),
        ):
            raise ValueError(
                "LP_EMBEDDING eps must be scalar or one radius per sample; "
                f"got shape {tuple(eps.shape)} for center {tuple(center.shape)}"
            )
        return eps.reshape(*batch_shape, 1, 1)

    def _embedding_position_mask(self) -> torch.Tensor:
        """Build the token mask on the clean embedding tensor's device."""
        center = self.center
        if center is None:
            raise ValueError("LP_EMBEDDING requires center")
        if center.dim() < 2:
            raise ValueError("LP_EMBEDDING center must have at least [L, D] shape")
        return normalize_position_mask(
            self.perturbed_positions,
            int(center.shape[-2]),
            batch_shape=tuple(center.shape[:-2]),
            device=center.device,
        )

    def input_geometry(self) -> InputGeometry:
        """Normalize geometry once at a model or ACT-layer construction boundary."""
        center = self.center
        if center is None or self.kind != InKind.LP_EMBEDDING:
            raise ValueError("Input geometry requires an LP_EMBEDDING center")
        mask = self._embedding_position_mask()
        if center.numel() == 0 or not bool(torch.isfinite(center).all()):
            raise ValueError("LP_EMBEDDING center must have nonempty, finite token coordinates")
        eps = self._eps_like(center)
        assert isinstance(self.p_norm, torch.Tensor)
        if center.dim() == 2:
            center, eps, mask = center.unsqueeze(0), eps.unsqueeze(0), mask.unsqueeze(0)
        return InputGeometry(center, eps, mask, float(self.p_norm.item()))


def normalize_input_geometry(
    params: Mapping[str, ScalarTensor | str | Sequence[int] | None],
) -> InputGeometry | None:
    """Interpret raw INPUT_SPEC fields in the frontend, never in solver kernels."""
    if params.get("kind") != InKind.LP_EMBEDDING:
        return None
    center, eps, p = params.get("center"), params.get("eps"), params.get("p_norm")
    positions = params.get("perturbed_positions")
    if not isinstance(center, torch.Tensor) or not isinstance(eps, (torch.Tensor, int, float)):
        raise ValueError("LP_EMBEDDING requires center and eps")
    if not isinstance(p, (torch.Tensor, int, float)):
        raise ValueError("LP_EMBEDDING requires numeric p_norm")
    if positions is not None and not isinstance(positions, (torch.Tensor, list, tuple)):
        raise ValueError("LP_EMBEDDING positions must be indices or a boolean mask")
    spec = InputSpec(
        kind=InKind.LP_EMBEDDING, center=center, eps=eps,
        p_norm=p, perturbed_positions=positions,
    )
    geometry = spec.input_geometry()
    width = params.get("embed_dim")
    if width is not None and (not isinstance(width, int) or width != geometry.embed_dim):
        raise ValueError("LP_EMBEDDING embed_dim must match center's last dimension")
    return geometry


class OutKind(str, Enum):
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
    # Kinds whose unsafe set is closed, so severity == 0 already violates.
    CLOSED_KINDS: ClassVar[frozenset[OutKind]] = frozenset(
        {OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST, OutKind.UNSAFE_LINEAR}
    )

    kind: OutKind
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
        
        # Convert margin, c, d, lb, ub (list/tuple → tensor; scalar → 1-D
        # tensor). ``d`` is scalar for LINEAR_LE but a vector for
        # UNSAFE_LINEAR, and ``margin`` is scalar for a shared spec but a
        # vector once there is one row per batch lane, so both belong on this
        # list-aware conversion path.
        for field in ['margin', 'c', 'd', 'lb', 'ub']:
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
                    f"UNSAFE_LINEAR: d length {d_vec.shape[0]} != N {N}. A d of "
                    f"shape {tuple(self.d.shape)} means this model was synthesized "
                    f'with cd_group="shape", which batches per-lane d for fuzzing '
                    f"and is not consumable by the verifier."
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

    def _gather_rows(
        self,
        rows: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        shared_ndim: Mapping[str, int],
        *,
        source: Optional[Mapping[str, Any]] = None,
        source_batch_size: Optional[int] = None,
        drop_singleton_batch: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Select the spec parameters backing each batch lane.

        Walks :attr:`SLICEABLE_PARAM_KEYS` and returns present tensor fields
        aligned to ``batch_size`` lanes. By default fields come from this spec
        and a field is row-indexed when its rank exceeds
        ``shared_ndim[key]``. Encoded ASSERT dictionaries can instead be
        supplied through ``source``; in that mode, ``source_batch_size``
        identifies their leading batch axis by size.

        A singleton leading row expands to every lane; otherwise ``rows``
        gathers the backing row. Every tensor is moved explicitly to ``device``
        and floating tensors are also cast to ``dtype``.
        ``drop_singleton_batch`` removes a gathered leading ``[1]`` from
        rank-two-or-higher fields so a shared spec can be re-encoded for a new
        batch size.
        """
        gathered: Dict[str, torch.Tensor] = {}
        for key in self.SLICEABLE_PARAM_KEYS:
            value = source.get(key) if source is not None else getattr(self, key, None)
            if not isinstance(value, torch.Tensor):
                continue
            value = (
                value.to(device=device, dtype=dtype)
                if value.is_floating_point()
                else value.to(device=device)
            )
            if source_batch_size is None:
                base_ndim = shared_ndim.get(key)
                row_indexed = base_ndim is not None and value.dim() > base_ndim
            else:
                row_indexed = (
                    value.dim() >= 1
                    and value.shape[0] in (1, source_batch_size)
                )
            if row_indexed:
                if value.shape[0] == 1:
                    value = value.expand(batch_size, *value.shape[1:])
                elif rows is not None:
                    value = value.index_select(0, rows)
                elif value.shape[0] != batch_size:
                    raise ValueError(
                        f"{self.kind}: {key} carries {value.shape[0]} spec rows "
                        f"but the batch has {batch_size} lanes; pass rows= to "
                        f"map lanes onto spec rows"
                    )
            if drop_singleton_batch and value.dim() >= 2 and value.shape[0] == 1:
                value = value[0]
            gathered[key] = value
        return gathered

    @staticmethod
    def _runner_up_gap(z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-lane ``max_{j != target} z_j - z_target``.

        This is the Carlini-Wagner robustness gap: positive iff some non-target
        output outranks the target one. TOP1 and MARGIN differ only by an
        additive margin, so both read this single quantity.

        Args:
            z: Outputs of shape ``[B, n_out]``.
            target: Long tensor of shape ``[B]``, the target index per lane.

        Returns:
            Float tensor of shape ``[B]``.

        Raises:
            ValueError: If ``z`` has fewer than two columns, leaving no
                runner-up to compare against.
        """
        if z.shape[1] < 2:
            raise ValueError(
                f"TOP1/MARGIN robustness requires >= 2 classes, got {z.shape[1]}"
            )
        index = target.reshape(-1, 1)
        z_target = z.gather(1, index).squeeze(1)
        target_mask = torch.zeros_like(z, dtype=torch.bool).scatter_(1, index, True)
        max_other = z.masked_fill(target_mask, float("-inf")).max(dim=1).values
        return max_other - z_target

    @staticmethod
    def _linear_row_slack(
        z: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ) -> torch.Tensor:
        """Per-lane, per-row slack ``c_i . z - d_i`` of shape ``[B, N]``.

        Accepts the three ``c`` layouts the framework emits, mirroring the
        shape branches in ``OutputSpecLayer.forward``: ``[n_out]`` (a single
        row), ``[N, n_out]`` (a row block shared by every lane) and
        ``[B, N, n_out]`` (per-lane rows). ``d`` pairs as ``[1]``, ``[N]`` or
        ``[B, N]``; a singleton ``d`` applies to every row, as in
        :meth:`encode_linear`.

        Args:
            z: Outputs of shape ``[B, n_out]``.
            c: Constraint coefficients in one of the layouts above.
            d: Constraint right-hand sides.

        Returns:
            Float tensor of shape ``[B, N]``. Row ``i`` of lane ``b`` satisfies
            ``c_i . z <= d_i`` iff its slack is ``<= 0``.

        Raises:
            ValueError: If ``c`` has rank above 3, or ``d`` cannot be paired
                with the row count implied by ``c``.
        """
        if c.dim() == 3:
            slack = torch.einsum("bmn,bn->bm", c, z)
        elif c.dim() <= 2:
            c_rows = c if c.dim() == 2 else c.unsqueeze(0)
            slack = z @ c_rows.T
        else:
            raise ValueError(f"c must have rank 1, 2 or 3, got {c.dim()}")
        n_rows = slack.shape[1]
        if d.dim() <= 1:
            d_rows = d.reshape(-1)
            if d_rows.numel() not in (1, n_rows):
                raise ValueError(
                    f"d length {d_rows.numel()} is neither 1 nor the row count "
                    f"{n_rows} implied by c"
                )
        else:
            if tuple(d.shape) != tuple(slack.shape):
                raise ValueError(
                    f"d shape {tuple(d.shape)} != per-lane row shape "
                    f"{tuple(slack.shape)}"
                )
            d_rows = d
        return slack - d_rows

    def severity(
        self,
        y: torch.Tensor,
        rows: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score how badly ``y`` breaks this property, without deciding it.

        ``severity`` is a signed violation margin that doubles as an attack
        objective: it grows monotonically with how badly the property is
        broken, so ascending it drives an input toward a violation.

        =============== ====================================
        kind            severity
        =============== ====================================
        TOP1_ROBUST     ``max_{j != t} z_j - z_t``
        MARGIN_ROBUST   ``m - (z_t - max_{j != t} z_j)``
        RANGE           ``max(max_k(lb_k - z_k),
                        max_k(z_k - ub_k))``
        LINEAR_LE       ``max_i (c_i . z - d_i)``
        UNSAFE_LINEAR   ``-max_i (c_i . z - d_i)``
        =============== ====================================

        This is the gradient-guidance entry point, deliberately separated from
        :meth:`violation` so an attack may score a tensor that is *not* the
        model's declared output — for instance pre-softmax logits, whose
        ranking matches the saturating softmax but whose gradient does not
        vanish. Scoring such a surrogate tensor is sound only for kinds that
        depend on ``z`` through its ranking alone; kinds that constrain output
        *values* (``RANGE``, ``LINEAR_LE``, ``UNSAFE_LINEAR``) must be scored
        on the real output. Callers own that choice. The violation DECISION
        always stays with :meth:`violation` on the real output.

        Args:
            y: Tensor to score. The leading axis is the batch axis; any
                trailing axes are flattened, so ``[B, n_out]`` and
                ``[B, ...]`` are both accepted.
            rows: Long tensor of length ``B`` where ``rows[i]`` is the spec row
                backing batch lane ``i``. ``None`` means identity, lane ``i``
                <-> row ``i``.

        Returns:
            Float tensor of shape ``[B]``.

        Raises:
            ValueError: If fields required by ``self.kind`` are missing, a
                gathered field cannot be aligned to the ``B`` lanes, or
                ``self.kind`` is not one of the five supported output kinds.
        """
        z = y.reshape(y.shape[0], -1)
        batch_size = z.shape[0]
        if rows is not None:
            rows = rows.to(device=z.device, dtype=torch.long).reshape(-1)
            if rows.numel() != batch_size:
                raise ValueError(
                    f"rows length {rows.numel()} != batch size {batch_size}"
                )

        if self.kind == OutKind.TOP1_ROBUST:
            if self.y_true is None:
                raise ValueError("TOP1_ROBUST requires y_true")
            params = self._gather_rows(
                rows, batch_size, z.device, z.dtype, {"y_true": 0}
            )
            target = params["y_true"].reshape(-1).to(torch.long)
            return self._runner_up_gap(z, target)

        elif self.kind == OutKind.MARGIN_ROBUST:
            if self.y_true is None or self.margin is None:
                raise ValueError("MARGIN_ROBUST requires both y_true and margin")
            params = self._gather_rows(
                rows, batch_size, z.device, z.dtype, {"y_true": 0, "margin": 0}
            )
            target = params["y_true"].reshape(-1).to(torch.long)
            # m - (z_t - max_other) == m + (max_other - z_t) == m + gap.
            return params["margin"].reshape(-1) + self._runner_up_gap(z, target)

        elif self.kind == OutKind.RANGE:
            if self.lb is None and self.ub is None:
                raise ValueError("RANGE requires lb and/or ub")
            params = self._gather_rows(
                rows, batch_size, z.device, z.dtype, {"lb": 1, "ub": 1}
            )
            sides: List[torch.Tensor] = []
            if "lb" in params:
                sides.append((params["lb"] - z).max(dim=1).values)
            if "ub" in params:
                sides.append((z - params["ub"]).max(dim=1).values)
            return sides[0] if len(sides) == 1 else torch.maximum(*sides)

        elif self.kind == OutKind.LINEAR_LE:
            if self.c is None or self.d is None:
                raise ValueError("LINEAR_LE requires both c and d")
            params = self._gather_rows(
                rows, batch_size, z.device, z.dtype, {"c": 2, "d": 1}
            )
            slack = self._linear_row_slack(z, params["c"], params["d"])
            # Conjunction: certified iff EVERY row holds, so the worst row wins.
            return slack.max(dim=1).values

        elif self.kind == OutKind.UNSAFE_LINEAR:
            if self.c is None or self.d is None:
                raise ValueError("UNSAFE_LINEAR requires both c and d")
            params = self._gather_rows(
                rows, batch_size, z.device, z.dtype, {"c": 2, "d": 1}
            )
            slack = self._linear_row_slack(z, params["c"], params["d"])
            return -slack.max(dim=1).values

        raise ValueError(
            f"Unsupported output kind {self.kind!r}. Supported: "
            f"LINEAR_LE, UNSAFE_LINEAR, TOP1_ROBUST, MARGIN_ROBUST, RANGE."
        )

    def violation(
        self,
        y: torch.Tensor,
        rows: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate this output property against concrete model outputs.

        Single source of truth for *concrete* violation semantics: given a real
        forward pass, decide per batch lane whether the property is broken and
        by how much. Comparisons are exact, with no epsilon tolerance, because
        a genuine forward pass that breaks the property *is* a counterexample.
        (``bab.check_violations_batched`` deliberately keeps a tolerance — it
        validates LP-solver candidates, where numerical slop is expected.)

        The magnitude comes from :meth:`severity`; this method adds only the
        decision. ``TOP1_ROBUST``, ``MARGIN_ROBUST`` and ``UNSAFE_LINEAR``
        compare with ``>=`` rather than ``>`` (see :attr:`CLOSED_KINDS`). For
        ``TOP1_ROBUST`` an exact tie already falsifies the strict
        ``z_t > z_j`` requirement, and ``MARGIN_ROBUST`` is the same
        requirement shifted by ``m`` — a separation of exactly ``m`` already
        falsifies it. Both match ``bab.check_violations_batched`` and
        ``encode_linear`` (whose ``thresholds`` are ``0`` and ``-m``
        respectively, certified iff the row max is strictly below the
        threshold). For ``UNSAFE_LINEAR``: its unsafe set is the *closed*
        polytope ``(C z <= d).all()``, so a lane sitting exactly on a face is
        already unsafe.

        Note:
            ``argmax`` breaks ties toward the lowest index, so on an exact
            output tie ``TOP1_ROBUST`` can disagree with an ``argmax``-based
            check. That disagreement set has measure zero.

        Args:
            y: Concrete model outputs. The leading axis is the batch axis; any
                trailing axes are flattened, so ``[B, n_out]`` and
                ``[B, ...]`` are both accepted.
            rows: Long tensor of length ``B`` where ``rows[i]`` is the spec row
                backing batch lane ``i``. Required whenever lanes are not
                aligned with spec rows — for instance a corpus that samples
                with replacement. ``None`` means identity, lane ``i`` <-> row
                ``i``.

        Returns:
            ``(violated, severity)``: a bool tensor of shape ``[B]`` and a
            float tensor of shape ``[B]``.

        Raises:
            ValueError: If fields required by ``self.kind`` are missing, a
                gathered field cannot be aligned to the ``B`` lanes, or
                ``self.kind`` is not one of the five supported output kinds.
        """
        severity = self.severity(y, rows=rows)
        violated = (
            severity >= 0 if self.kind in self.CLOSED_KINDS else severity > 0
        )
        return violated, severity
