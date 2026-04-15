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
from dataclasses import dataclass
from typing import Optional
import torch

class InKind:
    BOX = "BOX"
    LINF_BALL = "LINF_BALL"
    LIN_POLY = "LIN_POLY"

@dataclass
class InputSpec:
    kind: str
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    center: Optional[torch.Tensor] = None
    eps: Optional[torch.Tensor] = None
    A: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Ensure all numeric fields are tensors for architecture."""
        # Convert eps (scalar → 1-D tensor)
        if self.eps is not None and not isinstance(self.eps, torch.Tensor):
            self.eps = torch.tensor([float(self.eps)])
        
        # Convert d (scalar → 1-D tensor)
        if hasattr(self, 'd') and self.d is not None and not isinstance(self.d, torch.Tensor):
            self.d = torch.tensor([float(self.d)])
        
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

class OutKind:
    LINEAR_LE   = "LINEAR_LE"
    TOP1_ROBUST = "TOP1_ROBUST"
    MARGIN_ROBUST = "MARGIN_ROBUST"
    RANGE = "RANGE"

@dataclass
class OutputSpec:
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
        
        # Convert d (scalar → 1-D tensor)
        if self.d is not None and not isinstance(self.d, torch.Tensor):
            self.d = torch.tensor([float(self.d)])
        
        # Convert c, lb, ub (list or scalar → tensor)
        for field in ['c', 'lb', 'ub']:
            val = getattr(self, field, None)
            if val is not None and not isinstance(val, torch.Tensor):
                if isinstance(val, (list, tuple)):
                    setattr(self, field, torch.tensor(val))
                else:
                    setattr(self, field, torch.tensor([float(val)]))
