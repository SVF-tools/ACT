#===- act/front_end/specs.py - Specification Data Types ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Defines InputSpec and OutputSpec for verification specifications.
#   All specs are inherently batched where B=1 is single sample.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, Tuple
import torch


class InKind:
    """Input specification kinds."""
    BOX = "BOX"
    LINF_BALL = "LINF_BALL"
    LIN_POLY = "LIN_POLY"


class OutKind:
    """Output specification kinds."""
    LINEAR_LE = "LINEAR_LE"
    TOP1_ROBUST = "TOP1_ROBUST"
    MARGIN_ROBUST = "MARGIN_ROBUST"
    RANGE = "RANGE"


# =============================================================================
# Input Specification (Batch-Native)
# =============================================================================

@dataclass
class InputSpec:
    """
    Input specification for verification (batched, where B=1 is single sample).
    
    All tensors have shape [B, ...] where B is batch size.
    
    Attributes:
        kind: Constraint type (BOX, LINF_BALL, LIN_POLY)
        lb: Lower bound tensor [B, ...] for BOX
        ub: Upper bound tensor [B, ...] for BOX
        center: Center tensor [B, ...] for LINF_BALL
        eps: Perturbation radius - scalar or [B] tensor
        A: Constraint matrix [B, M, D] for LIN_POLY
        b: Constraint vector [B, M] for LIN_POLY
    
    Examples:
        >>> # BOX constraint
        >>> spec = InputSpec(kind=InKind.BOX, lb=lb_tensor, ub=ub_tensor)
        >>> print(spec.batch_size)
        
        >>> # L-infinity ball
        >>> spec = InputSpec(kind=InKind.LINF_BALL, center=images, eps=0.1)
        >>> lb, ub = spec.get_bounds()
    """
    kind: str
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    center: Optional[torch.Tensor] = None
    eps: Optional[Union[float, torch.Tensor]] = None
    A: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Validate required fields based on kind."""
        if self.kind == InKind.BOX:
            assert self.lb is not None and self.ub is not None, "BOX requires lb, ub"
        elif self.kind == InKind.LINF_BALL:
            assert self.center is not None and self.eps is not None, "LINF_BALL requires center, eps"
        elif self.kind == InKind.LIN_POLY:
            assert self.A is not None and self.b is not None, "LIN_POLY requires A, b"
    
    def _get_tensor(self) -> torch.Tensor:
        """Get the primary tensor for this spec kind."""
        if self.kind == InKind.BOX:
            return self.lb
        elif self.kind == InKind.LINF_BALL:
            return self.center
        elif self.kind == InKind.LIN_POLY:
            return self.A
        raise ValueError(f"Unknown kind: {self.kind}")
    
    @property
    def batch_size(self) -> int:
        """Get batch size (B dimension)."""
        return self._get_tensor().shape[0]
    
    @property
    def B(self) -> int:
        """Alias for batch_size."""
        return self.batch_size
    
    @property
    def device(self) -> torch.device:
        """Get device of tensors."""
        return self._get_tensor().device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of tensors."""
        return self._get_tensor().dtype
    
    @property
    def shape(self) -> torch.Size:
        """Get shape of primary tensor."""
        return self._get_tensor().shape
    
    def get_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (lb, ub) tensors.
        
        For LINF_BALL: lb=center-eps, ub=center+eps.
        For BOX: returns lb, ub directly.
        """
        if self.kind == InKind.BOX:
            return self.lb, self.ub
        elif self.kind == InKind.LINF_BALL:
            if isinstance(self.eps, torch.Tensor):
                # Broadcast eps to match center shape
                eps = self.eps.view(-1, *([1] * (self.center.dim() - 1)))
            else:
                eps = self.eps
            return self.center - eps, self.center + eps
        raise NotImplementedError(f"get_bounds not supported for {self.kind}")
    
    def to(self, device: Union[str, torch.device]) -> 'InputSpec':
        """Move spec to device."""
        def mv(t):
            return t.to(device) if isinstance(t, torch.Tensor) else t
        return InputSpec(
            kind=self.kind,
            lb=mv(self.lb),
            ub=mv(self.ub),
            center=mv(self.center),
            eps=mv(self.eps),
            A=mv(self.A),
            b=mv(self.b),
        )
    
    def __getitem__(self, idx) -> 'InputSpec':
        """Slice the spec to get a subset of samples."""
        def sl(t):
            return t[idx] if isinstance(t, torch.Tensor) else t
        eps_sliced = self.eps
        if isinstance(self.eps, torch.Tensor):
            eps_sliced = self.eps[idx]
        return InputSpec(
            kind=self.kind,
            lb=sl(self.lb),
            ub=sl(self.ub),
            center=sl(self.center),
            eps=eps_sliced,
            A=sl(self.A),
            b=sl(self.b),
        )
    
    def __len__(self) -> int:
        return self.batch_size
    
    def __repr__(self) -> str:
        shape = list(self._get_tensor().shape)
        if self.eps is not None and not isinstance(self.eps, torch.Tensor):
            return f"InputSpec({self.kind}, B={self.B}, shape={shape}, eps={self.eps})"
        return f"InputSpec({self.kind}, B={self.B}, shape={shape})"


# =============================================================================
# Output Specification (Batch-Native)
# =============================================================================

@dataclass
class OutputSpec:
    """
    Output specification for verification (batched, where B=1 is single sample).
    
    Attributes:
        kind: Constraint type (TOP1_ROBUST, MARGIN_ROBUST, LINEAR_LE, RANGE)
        c: Constraint vector [B, C] for LINEAR_LE
        d: Constraint bound - scalar or [B] for LINEAR_LE
        y_true: Ground truth labels - int or [B] tensor
        margin: Required margin - scalar or [B] tensor
        lb: Lower bound [B, C] for RANGE
        ub: Upper bound [B, C] for RANGE
        meta: Additional metadata
    
    Examples:
        >>> # TOP1 robustness
        >>> spec = OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=labels)
        >>> print(spec.batch_size)
        
        >>> # Margin robustness
        >>> spec = OutputSpec(kind=OutKind.MARGIN_ROBUST, y_true=labels, margin=0.5)
    """
    kind: str
    c: Optional[torch.Tensor] = None
    d: Optional[Union[float, torch.Tensor]] = None
    y_true: Optional[Union[int, torch.Tensor]] = None
    margin: Union[float, torch.Tensor] = 0.0
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate required fields based on kind."""
        if self.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            assert self.y_true is not None, f"{self.kind} requires y_true"
        elif self.kind == OutKind.LINEAR_LE:
            assert self.c is not None and self.d is not None, "LINEAR_LE requires c, d"
        elif self.kind == OutKind.RANGE:
            assert self.lb is not None or self.ub is not None, "RANGE requires lb or ub"
    
    def _get_tensor(self) -> torch.Tensor:
        """Get the primary tensor for this spec kind."""
        if self.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            if isinstance(self.y_true, torch.Tensor):
                return self.y_true
            # Single int label - no tensor available
            raise ValueError("Cannot get tensor from scalar y_true. Use tensor labels for batched specs.")
        elif self.kind == OutKind.LINEAR_LE:
            return self.c
        elif self.kind == OutKind.RANGE:
            return self.lb if self.lb is not None else self.ub
        raise ValueError(f"Unknown kind: {self.kind}")
    
    @property
    def batch_size(self) -> int:
        """Get batch size. Returns 1 for scalar y_true."""
        if self.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            if isinstance(self.y_true, torch.Tensor):
                return self.y_true.shape[0]
            return 1  # Scalar label = single sample
        return self._get_tensor().shape[0]
    
    @property
    def B(self) -> int:
        """Alias for batch_size."""
        return self.batch_size
    
    @property
    def device(self) -> torch.device:
        """Get device. Returns CPU for scalar y_true."""
        if self.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            if isinstance(self.y_true, torch.Tensor):
                return self.y_true.device
            return torch.device('cpu')
        return self._get_tensor().device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of primary tensor."""
        if self.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            if isinstance(self.y_true, torch.Tensor):
                return self.y_true.dtype
            return torch.long  # Default for labels
        return self._get_tensor().dtype
    
    def to(self, device: Union[str, torch.device]) -> 'OutputSpec':
        """Move spec to device."""
        def mv(t):
            return t.to(device) if isinstance(t, torch.Tensor) else t
        return OutputSpec(
            kind=self.kind,
            c=mv(self.c),
            d=mv(self.d),
            y_true=mv(self.y_true),
            margin=mv(self.margin),
            lb=mv(self.lb),
            ub=mv(self.ub),
            meta=self.meta.copy(),
        )
    
    def __getitem__(self, idx) -> 'OutputSpec':
        """Slice the spec to get a subset of samples."""
        def sl(t):
            return t[idx] if isinstance(t, torch.Tensor) else t
        
        # Handle scalar y_true
        y_true_sliced = self.y_true
        if isinstance(self.y_true, torch.Tensor):
            y_true_sliced = self.y_true[idx]
        
        # Handle scalar margin
        margin_sliced = self.margin
        if isinstance(self.margin, torch.Tensor):
            margin_sliced = self.margin[idx]
        
        return OutputSpec(
            kind=self.kind,
            c=sl(self.c),
            d=sl(self.d),
            y_true=y_true_sliced,
            margin=margin_sliced,
            lb=sl(self.lb),
            ub=sl(self.ub),
            meta=self.meta.copy(),
        )
    
    def __len__(self) -> int:
        return self.batch_size
    
    def __repr__(self) -> str:
        if self.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            if isinstance(self.y_true, torch.Tensor):
                n_unique = len(self.y_true.unique())
                return f"OutputSpec({self.kind}, B={self.B}, unique_labels={n_unique})"
            return f"OutputSpec({self.kind}, y_true={self.y_true})"
        return f"OutputSpec({self.kind}, B={self.B})"
