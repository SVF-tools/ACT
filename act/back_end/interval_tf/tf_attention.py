#===- act/back_end/interval_tf/tf_attention.py - Attention Bound Kernels =====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Precision attention bound kernels for the interval domain. Provides the
#   dual-planar linear-relaxation bound for the self-attention dot-product
#   (QKt and context) together with the ReLU-catalyzed fusion of the two
#   planes, the exp/reciprocal/divide softmax composition, and the layer-norm
#   relaxation. Bounds are carried as linear coefficients (lw/uw) plus biases
#   (lb/ub) over the perturbed-embedding block and concretized with the dual
#   norm, so the resulting box is tighter than per-element McCormick.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import math

import torch

# Guards a division by a vanishing interval width so the relaxation
# tangents/secants stay finite at degenerate (zero-width) inputs.
_EPS = 1e-12

# GELU global minimum, reused from tf_transformer.tf_gelu: the activation dips
# to _GELU_MIN_Y at _GELU_MIN_X, so any interval straddling the curvature bands
# is lower-bounded by this constant. -0.17004 sits just below the exact
# erf-GELU minimum (-0.169997), keeping it a sound floor for torch.erf GELU.
_GELU_MIN_X = -0.7517916
_GELU_MIN_Y = -0.17004
# Inflection points of erf-GELU: g''(x) = phi(x) * (2 - x^2) changes sign at
# +/- sqrt(2), so GELU is convex on [-sqrt2, sqrt2] and concave beyond it.
_GELU_INFLECTION = math.sqrt(2.0)


class LinearBounds:
    """Linear lower/upper bounds of a tensor over the perturbed input block.

    A tensor ``z`` is bracketed by two affine forms of the perturbation
    ``x^r`` (the embedding deltas of the perturbed words)::

        lw . x^r + lb  <=  z  <=  uw . x^r + ub

    where ``lw``/``uw`` have shape ``[B, length, dim_in, dim_out]`` and
    ``lb``/``ub`` have shape ``[B, length, dim_out]``. ``dim_in`` packs the
    perturbed words contiguously (``dim_in == embed_dim * perturbed_words``) so
    that :meth:`concretize` applies the dual norm independently per word.

    The class implements the linear-relaxation transfer functions for
    attention: multiplication, the dual-planar dot-product, the
    exp/reciprocal softmax, and layer-norm. The ``*_double`` variants emit the
    second (alternate) McCormick plane wherever a bilinear relaxation is taken;
    both planes are later fused by :func:`fuse_attention_planes`.
    """

    lw: torch.Tensor
    uw: torch.Tensor
    lb: torch.Tensor
    ub: torch.Tensor
    p: float
    q: float
    eps: float | torch.Tensor
    perturbed_words: int
    device: torch.device
    batch_size: int
    length: int
    dim_in: int
    dim_out: int

    def __init__(
        self,
        lw: torch.Tensor,
        uw: torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
        *,
        p: float,
        eps: float | torch.Tensor,
        perturbed_words: int,
    ) -> None:
        self.lw = lw
        self.uw = uw
        self.lb = lb
        self.ub = ub
        self.p = p
        # Dual norm exponent: q is the Holder conjugate of p (p=inf -> q=1,
        # p=2 -> q=2, p=1 -> q=inf). The inner maximization over the Lp ball
        # is solved exactly by the dual norm, so concretization is tight.
        self.q = 1.0 / (1.0 - 1.0 / p) if p != 1 else float("inf")
        # eps is stored verbatim: a python float keeps the original scalar-eps
        # path bit-identical, while a [B, perturbed_words] tensor carries the
        # original per-sample token radii, consumed block-wise in concretize.
        # Child boxes do not redefine them; a scalar never becomes a tensor.
        self.eps = eps
        self.perturbed_words = perturbed_words
        self.device = lw.device
        self.batch_size = lw.shape[0]
        self.length = lw.shape[1]
        self.dim_in = lw.shape[2]
        self.dim_out = lw.shape[3]

    # -- construction helpers --------------------------------------------------

    def _like(
        self,
        lw: torch.Tensor,
        uw: torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
    ) -> "LinearBounds":
        """Build a sibling bound that inherits the perturbation metadata."""
        return LinearBounds(
            lw, uw, lb, ub,
            p=self.p, eps=self.eps, perturbed_words=self.perturbed_words,
        )

    def clone(self) -> "LinearBounds":
        return self._like(
            self.lw.clone(), self.uw.clone(), self.lb.clone(), self.ub.clone()
        )

    # -- concretization --------------------------------------------------------

    def _eps_scale(self, eps_i: float | torch.Tensor) -> float | torch.Tensor:
        # A per-lane eps tensor holds one radius per (batch, block); reshape it
        # to [B, 1, 1] so it broadcasts against the [B, length, dim_out] dual
        # norm. A python-float scalar passes through untouched, so the scalar
        # path stays bit-identical to the original single-radius computation.
        if isinstance(eps_i, torch.Tensor):
            return eps_i.reshape(-1, 1, 1)
        return eps_i

    def _concretize_l(self, lw: torch.Tensor,
                      eps_i: float | torch.Tensor) -> torch.Tensor:
        # Lower envelope of the inner max: -eps * ||lw||_q minimizes lw . x^r
        # over the Lp ball, so subtracting it keeps the bound sound.
        return -self._eps_scale(eps_i) * torch.linalg.vector_norm(
            lw, ord=self.q, dim=-2)

    def _concretize_u(self, uw: torch.Tensor,
                      eps_i: float | torch.Tensor) -> torch.Tensor:
        return self._eps_scale(eps_i) * torch.linalg.vector_norm(
            uw, ord=self.q, dim=-2)

    def concretize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse the affine bounds to a box via the per-word dual norm.

        ``eps`` may be a scalar (one Lp ball shared by the whole batch) or a
        ``[B, perturbed_words]`` tensor carrying one radius per (lane, word)
        block. Each block's radius scales only that block's dual-norm tail.
        Radii describe the original token balls, not child-box half-widths.
        BaB intersects their support with child-box support in the forward
        frame; this method only consumes the radius it is handed.

        Returns:
            ``(lower, upper)`` boxes of shape ``[B, length, dim_out]``.
        """
        dim = self.lw.shape[-2] // self.perturbed_words
        res_l = self.lb.clone()
        res_u = self.ub.clone()
        eps = self.eps
        for i in range(self.perturbed_words):
            block = slice(dim * i, dim * (i + 1))
            eps_i = eps[..., i] if isinstance(eps, torch.Tensor) else eps
            res_l = res_l + self._concretize_l(self.lw[:, :, block, :], eps_i)
            res_u = res_u + self._concretize_u(self.uw[:, :, block, :], eps_i)
        return res_l, res_u

    def attention_bound_concretize(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Four-way concretization used to fuse the two attention planes.

        Each of ``lw``/``uw`` is concretized with both the lower (``-eps``) and
        upper (``+eps``) dual-norm tails, yielding ``(l_left, l_right, u_left,
        u_right)``. The fusion needs all four corners because soundness comes
        from a ReLU on the *difference* of the two planes, not from selecting
        or averaging either plane.
        """
        dim = self.lw.shape[-2] // self.perturbed_words
        res_l_left = self.lb.clone()
        res_l_right = self.lb.clone()
        res_u_left = self.ub.clone()
        res_u_right = self.ub.clone()
        eps = self.eps
        for i in range(self.perturbed_words):
            block = slice(dim * i, dim * (i + 1))
            eps_i = eps[..., i] if isinstance(eps, torch.Tensor) else eps
            res_l_left = res_l_left + self._concretize_l(self.lw[:, :, block, :], eps_i)
            res_l_right = res_l_right + self._concretize_u(self.lw[:, :, block, :], eps_i)
            res_u_left = res_u_left + self._concretize_l(self.uw[:, :, block, :], eps_i)
            res_u_right = res_u_right + self._concretize_u(self.uw[:, :, block, :], eps_i)
        return res_l_left, res_l_right, res_u_left, res_u_right

    def t(self) -> "LinearBounds":
        """Transpose the sequence and output axes (used to feed the context)."""
        return self._like(
            self.lw.transpose(1, 3), self.uw.transpose(1, 3),
            self.lb.transpose(1, 2), self.ub.transpose(1, 2),
        )

    def _new_slabs(
        self,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Return concretized bounds, sign masks, and zeroed coefficient slabs.

        The masks (``mask_pos``/``mask_neg``/``mask_both``) classify each output
        by the sign of its concretized interval; the zero slabs accumulate the
        per-branch linear relaxations added by :meth:`add_linear`.
        """
        l, u = self.concretize()
        mask_pos = torch.gt(l, 0).to(l.dtype)
        mask_neg = torch.lt(u, 0).to(l.dtype)
        mask_both = 1 - mask_pos - mask_neg
        lw = torch.zeros_like(self.lw)
        lb = torch.zeros_like(self.lb)
        uw = torch.zeros_like(self.uw)
        ub = torch.zeros_like(self.ub)
        return l, u, mask_pos, mask_neg, mask_both, lw, lb, uw, ub

    # -- affine algebra --------------------------------------------------------

    def add_linear(
        self,
        mask: torch.Tensor | None,
        w_out: torch.Tensor,
        b_out: torch.Tensor,
        type1: str,
        k: torch.Tensor,
        x0: float | torch.Tensor,
        y0: float | torch.Tensor,
        src: "LinearBounds | None" = None,
    ) -> None:
        """Accumulate ``k * (src - x0) + y0`` into ``(w_out, b_out)`` in place.

        The branch picks the source coefficients by the sign of the slope ``k``
        so the composed bound stays a valid lower/upper envelope: a positive
        slope keeps the same side, a negative slope flips to the opposite
        envelope. ``mask`` restricts the update to the selected outputs.
        """
        if mask is None:
            mask_w: float | torch.Tensor = 1.0
            mask_b: float | torch.Tensor = 1.0
        else:
            mask_w = mask.unsqueeze(2)
            mask_b = mask
        if src is None:
            src = self
        if type1 == "lower":
            w_pos, b_pos = src.lw, src.lb
            w_neg, b_neg = src.uw, src.ub
        else:
            w_pos, b_pos = src.uw, src.ub
            w_neg, b_neg = src.lw, src.lb
        mask_pos = torch.gt(k, 0).to(k.dtype)
        w_out += mask_w * mask_pos.unsqueeze(2) * w_pos * k.unsqueeze(2)
        b_out += mask_b * mask_pos * ((b_pos - x0) * k + y0)
        mask_neg = 1 - mask_pos
        w_out += mask_w * mask_neg.unsqueeze(2) * w_neg * k.unsqueeze(2)
        b_out += mask_b * mask_neg * ((b_neg - x0) * k + y0)

    def add(self, delta: "LinearBounds | float | torch.Tensor") -> "LinearBounds":
        """Add another bound or a constant bias to both envelopes."""
        if isinstance(delta, LinearBounds):
            return self._like(
                self.lw + delta.lw, self.uw + delta.uw,
                self.lb + delta.lb, self.ub + delta.ub,
            )
        return self._like(self.lw, self.uw, self.lb + delta, self.ub + delta)

    def matmul(self, weight: torch.Tensor) -> "LinearBounds":
        """Right-multiply by a constant 2-D weight ``[out, in]`` (dense affine).

        The weight is split into nonnegative/negative parts so the lower bound
        uses ``lw`` on the positive weights and ``uw`` on the negative ones
        (and vice versa for the upper bound), preserving soundness without a
        relaxation.
        """
        w = weight.t()
        pos_mask = torch.gt(w, 0).to(w.dtype)
        w_pos = w * pos_mask
        w_neg = w - w_pos
        return self._like(
            self.lw.matmul(w_pos) + self.uw.matmul(w_neg),
            self.lw.matmul(w_neg) + self.uw.matmul(w_pos),
            self.lb.matmul(w_pos) + self.ub.matmul(w_neg),
            self.lb.matmul(w_neg) + self.ub.matmul(w_pos),
        )

    def multiply(self, weight: float | torch.Tensor) -> "LinearBounds":
        """Multiply by a scalar or constant tensor (signed coefficient split).

        A negative factor swaps the lower and upper envelopes; a constant
        tensor applies the swap elementwise. The bound-times-bound product used
        by softmax lives in :meth:`_multiply_bound_plane`.
        """
        if isinstance(weight, torch.Tensor):
            pos_mask = torch.gt(weight, 0).to(weight.dtype)
            w_pos = weight * pos_mask
            w_neg = weight - w_pos
            return self._like(
                self.lw * w_pos + self.uw * w_neg,
                self.lw * w_neg + self.uw * w_pos,
                self.lb * w_pos + self.ub * w_neg,
                self.lb * w_neg + self.ub * w_pos,
            )
        if weight > 0:
            return self._like(
                self.lw * weight, self.uw * weight,
                self.lb * weight, self.ub * weight,
            )
        return self._like(
            self.uw * weight, self.lw * weight,
            self.ub * weight, self.lb * weight,
        )

    def get_bounds_xy(
        self,
        l_x: torch.Tensor,
        u_x: torch.Tensor,
        l_y: torch.Tensor,
        u_y: torch.Tensor,
        z: bool = False,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Planar McCormick coefficients for ``z = x * y`` over the box.

        ``z=False`` yields the lower plane anchored at ``(l_x, l_y)`` and the
        upper plane anchored at ``(l_x, u_y)``; ``z=True`` yields the alternate
        planes anchored at the upper corners. Both are valid envelopes; keeping
        both is what enables the catalytic tightening.
        """
        if not z:
            alpha_l = l_y
            beta_l = l_x
            gamma_l = -alpha_l * beta_l
            alpha_u = u_y
            beta_u = l_x
            gamma_u = -alpha_u * beta_u
        else:
            alpha_l = u_y
            beta_l = u_x
            gamma_l = -alpha_l * beta_l
            alpha_u = l_y
            beta_u = u_x
            gamma_u = -alpha_u * beta_u
        return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

    def _multiply_bound_plane(self, other: "LinearBounds", z: bool) -> "LinearBounds":
        """One planar McCormick relaxation of the product ``self * other``.

        Used by the softmax divide; the two ``z`` choices feed the catalytic
        fusion so the softmax can also benefit from the dual plane.
        """
        l_a, u_a = self.concretize()
        l_b, u_b = other.concretize()
        _, _, _, _, _, lw, lb, uw, ub = self._new_slabs()
        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = self.get_bounds_xy(
            l_a.reshape(-1), u_a.reshape(-1),
            l_b.reshape(-1), u_b.reshape(-1), z=z,
        )
        alpha_l = alpha_l.reshape(l_a.shape)
        beta_l = beta_l.reshape(l_a.shape)
        gamma_l = gamma_l.reshape(l_a.shape)
        alpha_u = alpha_u.reshape(l_a.shape)
        beta_u = beta_u.reshape(l_a.shape)
        gamma_u = gamma_u.reshape(l_a.shape)
        self.add_linear(None, lw, lb, "lower", alpha_l, 0, gamma_l)
        self.add_linear(None, lw, lb, "lower", beta_l, 0, 0, src=other)
        self.add_linear(None, uw, ub, "upper", alpha_u, 0, gamma_u)
        self.add_linear(None, uw, ub, "upper", beta_u, 0, 0, src=other)
        return self._like(lw, uw, lb, ub)

    # -- attention dot-products ------------------------------------------------

    def _dot_product_planes(self, other: "LinearBounds", z: bool) -> "LinearBounds":
        """Single planar relaxation of ``sum_k self[..,k] * other[..,k]``.

        Each summand is a bilinear term relaxed by :meth:`get_bounds_xy`; the
        linear parts are propagated through the stored ``lw``/``uw`` frames so
        the result stays affine in the perturbation. ``z`` selects which plane.
        """
        l_a, u_a = self.concretize()
        l_b, u_b = other.concretize()
        B, length, dim_out = l_a.shape
        other_length = l_b.shape[1]
        shape = (B, length, other_length, dim_out)

        # Whole-batch planar McCormick coefficients: broadcast the per-term
        # corners over the (length x other_length) output grid in one shot.
        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = self.get_bounds_xy(
            l_a.unsqueeze(2), u_a.unsqueeze(2),
            l_b.unsqueeze(1), u_b.unsqueeze(1), z=z,
        )
        alpha_l = alpha_l.expand(shape)
        beta_l = beta_l.expand(shape)
        alpha_u = alpha_u.expand(shape)
        beta_u = beta_u.expand(shape)

        def _w_self(frame_pos, frame_neg, weight):
            # Signed slope split summed over the shared dim_out contraction:
            # the positive part keeps frame_pos, the negative part flips to
            # frame_neg (replaces the per-batch matmul with one einsum).
            return (torch.einsum("bldk,blok->bldo", frame_pos, weight.clamp(min=0))
                    + torch.einsum("bldk,blok->bldo", frame_neg, weight.clamp(max=0)))

        def _w_other(weight, frame_pos, frame_neg):
            return (torch.einsum("blok,bodk->bldo", weight.clamp(min=0), frame_pos)
                    + torch.einsum("blok,bodk->bldo", weight.clamp(max=0), frame_neg))

        def _b_self(frame_pos, frame_neg, weight):
            return (torch.einsum("blk,blok->blo", frame_pos, weight.clamp(min=0))
                    + torch.einsum("blk,blok->blo", frame_neg, weight.clamp(max=0)))

        def _b_other(weight, frame_pos, frame_neg):
            return (torch.einsum("bok,blok->blo", frame_pos, weight.clamp(min=0))
                    + torch.einsum("bok,blok->blo", frame_neg, weight.clamp(max=0)))

        # Lower envelope: positive slope keeps lw, negative flips to uw.
        lw = _w_self(self.lw, self.uw, alpha_l) + _w_other(beta_l, other.lw, other.uw)
        lb = (gamma_l.sum(dim=-1)
              + _b_self(self.lb, self.ub, alpha_l) + _b_other(beta_l, other.lb, other.ub))
        # Upper envelope: mirror of the lower with the upper slopes.
        uw = _w_self(self.uw, self.lw, alpha_u) + _w_other(beta_u, other.uw, other.lw)
        ub = (gamma_u.sum(dim=-1)
              + _b_self(self.ub, self.lb, alpha_u) + _b_other(beta_u, other.ub, other.lb))

        return self._like(lw, uw, lb, ub)

    def _dot_product_degenerate(self, other: "LinearBounds") -> "LinearBounds":
        """Closed-form box product when there is no perturbation dimension."""
        l1, u1 = self.lb.unsqueeze(-2), self.ub.unsqueeze(-2)
        l2, u2 = other.lb.unsqueeze(1), other.ub.unsqueeze(1)
        prod1, prod2, prod3, prod4 = l1 * l2, l1 * u2, u1 * l2, u1 * u2
        l = torch.min(torch.min(prod1, prod2), torch.min(prod3, prod4)).sum(-1)
        u = torch.max(torch.max(prod1, prod2), torch.max(prod3, prod4)).sum(-1)
        w = l.unsqueeze(-2) * 0
        return self._like(w, w, l, u)

    def dot_product(self, other: "LinearBounds") -> "LinearBounds":
        """Single-planar attention bound for ``Q Kt`` / context.

        Implements the dual-norm linear bound of the multi-head dot-product:
        one valid plane per side, summed over the head dimension.
        """
        if self.dim_in == 0:
            return self._dot_product_degenerate(other)
        return self._dot_product_planes(other, z=False)

    def dot_product_double(
        self, other: "LinearBounds",
    ) -> tuple["LinearBounds", "LinearBounds"]:
        """Dual-planar attention bound: returns both McCormick planes.

        The pair is fused by :func:`fuse_attention_planes`; returning both is
        load-bearing because the catalytic ReLU acts on their difference.
        """
        if self.dim_in == 0:
            degenerate = self._dot_product_degenerate(other)
            return degenerate, degenerate.clone()
        return (
            self._dot_product_planes(other, z=False),
            self._dot_product_planes(other, z=True),
        )

    def context(self, value: "LinearBounds") -> "LinearBounds":
        """Attention-weighted value ``probs @ value`` (single plane)."""
        return self.dot_product(value.t())

    def context_double(
        self, value: "LinearBounds",
    ) -> tuple["LinearBounds", "LinearBounds"]:
        """Attention-weighted value keeping both planes for the fusion."""
        return self.dot_product_double(value.t())

    # -- elementwise relaxations (softmax / layer-norm building blocks) --------

    def reciprocal(self) -> "LinearBounds":
        """Bound ``1 / y`` for strictly positive ``y`` (softmax denominator).

        The lower bound is the tangent at the midpoint and the upper bound the
        secant; because the reciprocal slope is negative, the tangent uses the
        upper coefficient frame and the secant the lower one.
        """
        l, u = self.concretize()
        m = (l + u) / 2
        kl = -1 / m.pow(2)
        lw = self.uw * kl.unsqueeze(2)
        lb = self.ub * kl + 2 / m
        ku = -1.0 / (l * u)
        uw = self.lw * ku.unsqueeze(2)
        ub = self.lb * ku - ku * l + 1 / l
        return self._like(lw, uw, lb, ub)

    def divide(self, weight: "LinearBounds") -> "LinearBounds":
        """Divide by a positive bound via ``self * (1 / weight)`` (one plane)."""
        return self._multiply_bound_plane(weight.reciprocal(), z=False)

    def exp(self) -> "LinearBounds":
        """Bound ``exp(x)`` with a tangent lower and secant upper envelope.

        The tangent anchor stays below ``l + 1`` to keep its lower bound
        positive. Nonfinite exponentials remain visible to callers: a linear
        extrapolation of a convex exponential is not an upper envelope.
        """
        l, u = self.concretize()
        # Anchor < l + 1 keeps exp(anchor)*(l-anchor)+exp(anchor) > 0.
        m = torch.min((l + u) / 2, l + 0.99)
        kl = torch.exp(m)
        lw = self.lw * kl.unsqueeze(2)
        lb = kl * (self.lb - m + 1)
        width = u - l
        ku = (torch.exp(u) - torch.exp(l)) / torch.where(width > 0, width, torch.ones_like(width))
        uw = self.uw * ku.unsqueeze(2)
        ub = self.ub * ku - ku * l + torch.exp(l)
        return self._like(lw, uw, lb, ub)

    def _softmax_sum(self) -> tuple["LinearBounds", "LinearBounds"]:
        """Return ``exp(self)`` and the broadcast row-sum bound it divides by."""
        bounds_exp = self.exp()
        sum_lw = torch.sum(bounds_exp.lw, dim=-1, keepdim=True)
        sum_uw = torch.sum(bounds_exp.uw, dim=-1, keepdim=True)
        sum_lb = torch.sum(bounds_exp.lb, dim=-1, keepdim=True)
        sum_ub = torch.sum(bounds_exp.ub, dim=-1, keepdim=True)
        bounds_sum = self._like(
            sum_lw.expand(-1, -1, -1, self.dim_out),
            sum_uw.expand(-1, -1, -1, self.dim_out),
            sum_lb.expand(-1, -1, self.dim_out),
            sum_ub.expand(-1, -1, self.dim_out),
        )
        return bounds_exp, bounds_sum

    def softmax(self) -> "LinearBounds":
        """Softmax over the last axis as ``exp`` then ``divide`` by the sum."""
        bounds_exp, bounds_sum = self._softmax_sum()
        return bounds_exp.divide(bounds_sum)

    def softmax_double(self) -> tuple["LinearBounds", "LinearBounds"]:
        """Dual-planar softmax: both divide planes for the catalytic fusion."""
        bounds_exp, bounds_sum = self._softmax_sum()
        recip = bounds_sum.reciprocal()
        return (
            bounds_exp._multiply_bound_plane(recip, z=False),
            bounds_exp._multiply_bound_plane(recip, z=True),
        )

    def sqr(self) -> "LinearBounds":
        """Bound ``x^2``: secant upper and a sign-aware tangent lower.

        The lower anchor is pushed away from zero (``2u`` for negative inputs,
        ``2l`` for positive) so the tangent never dips below zero, keeping the
        squared output nonnegative.
        """
        l, u, mask_pos, mask_neg, _, lw, lb, uw, ub = self._new_slabs()
        k = u + l
        self.add_linear(None, uw, ub, "upper", k, l, l.pow(2))
        m = torch.max((l + u) / 2, 2 * u)
        self.add_linear(mask_neg, lw, lb, "lower", 2 * m, m, m.pow(2))
        m = torch.min((l + u) / 2, 2 * l)
        self.add_linear(mask_pos, lw, lb, "lower", 2 * m, m, m.pow(2))
        return self._like(lw, uw, lb, ub)

    def sqrt(self) -> "LinearBounds":
        """Bound ``sqrt(x)`` for positive ``x`` (layer-norm denominator)."""
        l, u, _, _, _, lw, lb, uw, ub = self._new_slabs()
        k = (torch.sqrt(u) - torch.sqrt(l)) / (u - l + _EPS)
        self.add_linear(None, lw, lb, "lower", k, l, torch.sqrt(l))
        m = (l + u) / 2
        k = 0.5 / torch.sqrt(m)
        self.add_linear(None, uw, ub, "upper", k, m, torch.sqrt(m))
        return self._like(lw, uw, lb, ub)

    def gelu(self) -> "LinearBounds":
        """Bound the exact erf GELU ``x * Phi(x)`` with a curvature-aware envelope.

        GELU has constant-sign curvature on each region split by the inflection
        points ``+/- sqrt(2)`` (``g''(x) = phi(x) * (2 - x^2)``): convex on
        ``[-sqrt2, sqrt2]`` and concave outside. On a single-curvature interval
        the midpoint tangent and the endpoint secant are exact one-sided
        envelopes -- tangent below and secant above when convex, with the roles
        swapped when concave -- so they yield a tight linear bound. An interval
        that straddles an inflection point mixes curvature, so it falls back to
        the min-aware box of :func:`tf_gelu` (constant ``_GELU_MIN_Y`` floor when
        the global minimum is enclosed, otherwise the monotone endpoint), a
        sound zero-slope envelope. The exact erf form matches ``F.gelu`` so the
        bound holds against the concrete activation, not just the tanh proxy.
        """
        l, u, _, _, _, lw, lb, uw, ub = self._new_slabs()

        inv_sqrt_2 = 1.0 / math.sqrt(2.0)
        inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)

        def gelu_f(x: torch.Tensor) -> torch.Tensor:
            return 0.5 * x * (1.0 + torch.erf(x * inv_sqrt_2))

        def gelu_df(x: torch.Tensor) -> torch.Tensor:
            # g'(x) = Phi(x) + x * phi(x): slope of the midpoint tangent line.
            return 0.5 * (1.0 + torch.erf(x * inv_sqrt_2)) + \
                x * inv_sqrt_2pi * torch.exp(-0.5 * x * x)

        g_l, g_u = gelu_f(l), gelu_f(u)
        m = (l + u) / 2
        g_m, dg_m = gelu_f(m), gelu_df(m)
        secant = (g_u - g_l) / (u - l + _EPS)

        convex = (l >= -_GELU_INFLECTION) & (u <= _GELU_INFLECTION)
        concave = (u <= -_GELU_INFLECTION) | (l >= _GELU_INFLECTION)

        # Min-aware box fallback for straddling intervals: the global minimum is
        # enclosed iff [l, u] brackets _GELU_MIN_X, else GELU is monotone there
        # and the extrema land on the endpoints (max is always at an endpoint).
        contains_min = (l <= _GELU_MIN_X) & (u >= _GELU_MIN_X)
        box_lo = torch.where(
            contains_min, torch.full_like(g_l, _GELU_MIN_Y),
            torch.minimum(g_l, g_u))
        box_hi = torch.maximum(g_l, g_u)

        zero = torch.zeros_like(l)
        kl = torch.where(convex, dg_m, torch.where(concave, secant, zero))
        xl0 = torch.where(concave, l, m)
        yl0 = torch.where(convex, g_m, torch.where(concave, g_l, box_lo))
        ku = torch.where(convex, secant, torch.where(concave, dg_m, zero))
        xu0 = torch.where(convex, l, m)
        yu0 = torch.where(convex, g_l, torch.where(concave, g_m, box_hi))

        self.add_linear(None, lw, lb, "lower", kl, xl0, yl0)
        self.add_linear(None, uw, ub, "upper", ku, xu0, yu0)
        return self._like(lw, uw, lb, ub)

    def layer_norm(
        self,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        variant: str = "standard",
        eps: float = _EPS,
    ) -> "LinearBounds":
        """Layer normalization relaxation with an optional no-variance mode.

        ``variant='no'`` is the identity, ``'no_var'`` subtracts the mean only
        (tighter, used by the artifact's tiny models), and ``'standard'`` also
        divides by ``sqrt(var + eps)``. The mean is an exact linear map; the
        variance path reuses :meth:`sqr`/:meth:`sqrt`/:meth:`divide`.
        """
        if variant == "no":
            return self

        dim = self.dim_out
        w_avg = torch.full(
            (dim, dim), 1.0 / dim, device=self.device, dtype=self.lw.dtype
        )
        minus_mu = self.add(self.matmul(w_avg).multiply(-1.0))

        if variant == "standard":
            variance = minus_mu.sqr().matmul(w_avg)
            normalized = minus_mu.divide(variance.add(eps).sqrt())
        elif variant == "no_var":
            normalized = minus_mu
        else:
            raise ValueError(f"layer_norm: unknown variant '{variant}'")

        return normalized.multiply(gamma).add(beta)


def rule_based_alpha(
    active_mask: torch.Tensor,
    upper: torch.Tensor,
    lower: torch.Tensor,
    k: float,
) -> torch.Tensor:
    """Rule-based ReLU slope for the catalytic fusion (rule-slope init).

    For an active (sign-crossing) entry the slope is 0 when the negative range
    dominates by more than ``k``, 1 when the positive range dominates, and the
    triangle slope ``upper / (upper - lower)`` otherwise. Inactive entries get
    slope 0.

    Args:
        active_mask: Float mask of entries whose plane difference crosses zero.
        upper: Upper corner of the plane difference.
        lower: Lower corner of the plane difference.
        k: Threshold balancing the positive and negative ranges.

    Returns:
        The per-entry slope in ``[0, 1]``.
    """
    mask_active = active_mask.bool()
    neg_range = (-lower).clamp(min=0) * active_mask
    pos_range = upper.clamp(min=0) * active_mask
    mask_zero = mask_active & (neg_range > k * pos_range)
    mask_one = mask_active & (pos_range > k * neg_range)
    mask_formula = mask_active & ~(mask_zero | mask_one)
    slope = torch.zeros_like(upper)
    slope[mask_one] = 1.0
    denominator = (upper - lower).clamp(min=1e-6)
    formula_slope = (upper / denominator).clamp(0.0, 1.0)
    slope[mask_formula] = formula_slope[mask_formula]
    return slope


def fuse_attention_planes(
    bound1: LinearBounds,
    bound2: LinearBounds,
    k: float,
    op_name: str,
    *,
    clamp_alpha: bool = False,
) -> LinearBounds:
    """Fuse the two attention planes into one sound bound (ReLU-catalyzed).

    The plane difference ``bound2 - bound1`` is concretized four ways; where a
    side stays positive the second plane is taken verbatim, and where it crosses
    zero a ReLU scaled by the rule-based slope is applied. Selecting or
    averaging a single plane would be unsound, so the fusion always keeps
    ``bound1`` as the base and adds the masked ReLU correction.

    Args:
        bound1: First plane (``z=False``) from ``dot_product_double``.
        bound2: Second plane (``z=True``) from ``dot_product_double``.
        k: Rule threshold passed to :func:`rule_based_alpha`.
        op_name: Operation tag (``"qk"``/``"softmax"``/``"qkv"``); selects the
            optimized-alpha registry key when the slope is later refined.
        clamp_alpha: When True clamp the slope to ``[1e-4, 1-1e-4]`` to mirror
            the optimized-alpha initialization warm start.

    Returns:
        The fused single-plane bound.
    """
    del op_name  # Reserved for the optimized-alpha registry key; unused here.
    diff = bound1._like(
        bound2.lw - bound1.lw,
        bound1.uw - bound2.uw,
        bound2.lb - bound1.lb,
        bound1.ub - bound2.ub,
    )
    l_min, l_max, u_min, u_max = diff.attention_bound_concretize()
    dtype = diff.lb.dtype

    def pos_mask(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return ((low > 0) & (high > 0)).to(dtype)

    def cross_mask(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return ((low < 0) & (high > 0)).to(dtype)

    upper_pos = pos_mask(u_min, u_max)
    upper_cross = cross_mask(u_min, u_max)
    lower_pos = pos_mask(l_min, l_max)
    lower_cross = cross_mask(l_min, l_max)

    omega_l = rule_based_alpha(lower_cross, l_max, l_min, k)
    omega_u = rule_based_alpha(upper_cross, u_max, u_min, k)
    if clamp_alpha:
        # The optimized-alpha mode seeds the optimizer at the rule slope clamped
        # into the open unit interval; only crossing entries carry a nonzero slope.
        omega_l = torch.where(
            lower_cross.bool(), omega_l.clamp(1e-4, 1 - 1e-4),
            torch.zeros_like(omega_l),
        )
        omega_u = torch.where(
            upper_cross.bool(), omega_u.clamp(1e-4, 1 - 1e-4),
            torch.zeros_like(omega_u),
        )

    lower_linear = lower_pos.unsqueeze(-2) * diff.lw
    upper_linear = upper_pos.unsqueeze(-2) * diff.uw
    lower_relu = lower_cross.unsqueeze(-2) * (omega_l.unsqueeze(-2) * diff.lw)
    upper_relu = upper_cross.unsqueeze(-2) * (omega_u.unsqueeze(-2) * diff.uw)

    eff_lw = lower_linear + lower_relu
    eff_uw = upper_linear + upper_relu
    eff_lb = lower_pos * diff.lb + lower_cross * (omega_l * diff.lb)
    eff_ub = upper_pos * diff.ub + upper_cross * (omega_u * diff.ub)

    return bound1._like(
        bound1.lw + eff_lw,
        bound1.uw - eff_uw,
        bound1.lb + eff_lb,
        bound1.ub - eff_ub,
    )


def att_scores_dual_planar(
    query: LinearBounds,
    key: LinearBounds,
    *,
    head_size: int,
    k: float = 1.0,
    clamp_alpha: bool = False,
    mask: torch.Tensor | None = None,
) -> LinearBounds:
    """Fused dual-planar attention scores ``scale * combine(Q Kt planes)``.

    Args:
        query: Linear bound of the per-head queries.
        key: Linear bound of the per-head keys.
        head_size: Attention head dimension; scores are scaled by
            ``1 / sqrt(head_size)``.
        k: Rule threshold for the fusion slope.
        clamp_alpha: Mirror the optimized-alpha slope clamping.
        mask: Optional additive attention mask broadcast onto both biases.

    Returns:
        The fused, scaled attention-score bound.
    """
    plane1, plane2 = query.dot_product_double(key)
    fused = fuse_attention_planes(plane1, plane2, k, "qk", clamp_alpha=clamp_alpha)
    scaled = fused.multiply(1.0 / math.sqrt(head_size))
    if mask is not None:
        scaled = scaled.add(mask)
    return scaled
