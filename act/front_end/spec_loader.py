#===- act/front_end/spec_loader.py - Unified Spec Loader ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Unified interface for batch-loading specifications from various sources:
#   - TorchVision datasets (MNIST, CIFAR, etc.)
#   - VNNLIB benchmarks (VNN-COMP categories)
#   - CSV configuration files
#
#   All methods return batch-native (InputSpec, OutputSpec) pairs.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Union, Sequence
import logging
import torch
import torch.nn as nn

from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind

logger = logging.getLogger(__name__)


class SpecLoader:
    """
    Unified loader for batch-native specifications.
    
    Provides static methods for loading (InputSpec, OutputSpec) pairs
    from various data sources, all returning batch-native specs.
    
    Examples:
        >>> # Load from TorchVision dataset
        >>> in_spec, out_spec = SpecLoader.from_torchvision(
        ...     dataset=mnist_test, indices=[0,1,2,3], eps=0.1
        ... )
        >>> print(in_spec.batch_size)  # 4
        
        >>> # Load from VNNLIB files
        >>> in_spec, out_spec = SpecLoader.from_vnnlib(
        ...     vnnlib_paths=['prop1.vnnlib', 'prop2.vnnlib']
        ... )
    """
    
    @staticmethod
    def from_torchvision(
        dataset,
        indices: Sequence[int],
        eps: Union[float, torch.Tensor] = 0.03,
        input_kind: str = InKind.LINF_BALL,
        output_kind: str = OutKind.TOP1_ROBUST,
        clamp_bounds: bool = True,
    ) -> Tuple[InputSpec, OutputSpec]:
        """
        Create batched specs from a TorchVision dataset.
        
        Args:
            dataset: TorchVision dataset (e.g., MNIST, CIFAR10)
            indices: List of sample indices to load
            eps: Perturbation radius (scalar or [B] tensor)
            input_kind: Input constraint type (LINF_BALL or BOX)
            output_kind: Output constraint type (TOP1_ROBUST, MARGIN_ROBUST)
            clamp_bounds: If True, clamp bounds to [0, 1] for image data
            
        Returns:
            Tuple of (InputSpec, OutputSpec) with batch size = len(indices)
            
        Example:
            >>> from torchvision.datasets import MNIST
            >>> from torchvision import transforms
            >>> mnist = MNIST('./data', train=False, transform=transforms.ToTensor())
            >>> in_spec, out_spec = SpecLoader.from_torchvision(
            ...     mnist, indices=range(10), eps=0.1
            ... )
            >>> print(in_spec.batch_size)  # 10
        """
        # Collect samples
        images = []
        labels = []
        for idx in indices:
            img, label = dataset[idx]
            if isinstance(img, torch.Tensor):
                images.append(img)
            else:
                # PIL Image - need transform
                raise ValueError(
                    "Dataset must return tensors. "
                    "Apply transforms.ToTensor() to your dataset."
                )
            labels.append(label)
        
        # Stack into batched tensors
        center = torch.stack(images, dim=0)  # [B, C, H, W]
        y_true = torch.tensor(labels, dtype=torch.long)  # [B]
        
        # Create input spec
        if input_kind == InKind.LINF_BALL:
            in_spec = InputSpec(
                kind=InKind.LINF_BALL,
                center=center,
                eps=eps,
            )
        elif input_kind == InKind.BOX:
            # Compute bounds from eps
            if isinstance(eps, torch.Tensor):
                eps_broadcast = eps.view(-1, *([1] * (center.dim() - 1)))
            else:
                eps_broadcast = eps
            lb = center - eps_broadcast
            ub = center + eps_broadcast
            if clamp_bounds:
                lb = lb.clamp(0, 1)
                ub = ub.clamp(0, 1)
            in_spec = InputSpec(kind=InKind.BOX, lb=lb, ub=ub)
        else:
            raise ValueError(f"Unsupported input_kind: {input_kind}")
        
        # Create output spec
        out_spec = OutputSpec(kind=output_kind, y_true=y_true)
        
        logger.debug(
            f"Loaded {len(indices)} samples from TorchVision dataset: "
            f"input={in_spec}, output={out_spec}"
        )
        
        return in_spec, out_spec
    
    @staticmethod
    def from_tensors(
        images: torch.Tensor,
        labels: torch.Tensor,
        eps: Union[float, torch.Tensor] = 0.03,
        input_kind: str = InKind.LINF_BALL,
        output_kind: str = OutKind.TOP1_ROBUST,
        clamp_bounds: bool = True,
    ) -> Tuple[InputSpec, OutputSpec]:
        """
        Create batched specs from raw tensors.
        
        Args:
            images: Image tensor [B, C, H, W]
            labels: Label tensor [B]
            eps: Perturbation radius
            input_kind: Input constraint type
            output_kind: Output constraint type
            clamp_bounds: If True, clamp bounds to [0, 1]
            
        Returns:
            Tuple of (InputSpec, OutputSpec)
            
        Example:
            >>> images = torch.rand(16, 1, 28, 28)
            >>> labels = torch.randint(0, 10, (16,))
            >>> in_spec, out_spec = SpecLoader.from_tensors(images, labels, eps=0.1)
        """
        assert images.dim() >= 2, f"Images must be at least 2D, got {images.dim()}D"
        assert labels.dim() == 1, f"Labels must be 1D, got {labels.dim()}D"
        assert images.shape[0] == labels.shape[0], (
            f"Batch size mismatch: images={images.shape[0]}, labels={labels.shape[0]}"
        )
        
        if input_kind == InKind.LINF_BALL:
            in_spec = InputSpec(kind=InKind.LINF_BALL, center=images, eps=eps)
        elif input_kind == InKind.BOX:
            if isinstance(eps, torch.Tensor):
                eps_broadcast = eps.view(-1, *([1] * (images.dim() - 1)))
            else:
                eps_broadcast = eps
            lb = images - eps_broadcast
            ub = images + eps_broadcast
            if clamp_bounds:
                lb = lb.clamp(0, 1)
                ub = ub.clamp(0, 1)
            in_spec = InputSpec(kind=InKind.BOX, lb=lb, ub=ub)
        else:
            raise ValueError(f"Unsupported input_kind: {input_kind}")
        
        out_spec = OutputSpec(kind=output_kind, y_true=labels)
        
        return in_spec, out_spec
    
    @staticmethod
    def from_bounds(
        lb: torch.Tensor,
        ub: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        output_kind: str = OutKind.TOP1_ROBUST,
    ) -> Tuple[InputSpec, OutputSpec]:
        """
        Create specs from explicit box bounds.
        
        Args:
            lb: Lower bound tensor [B, ...]
            ub: Upper bound tensor [B, ...]
            y_true: Optional labels [B] for robustness specs
            output_kind: Output constraint type
            
        Returns:
            Tuple of (InputSpec, OutputSpec)
        """
        in_spec = InputSpec(kind=InKind.BOX, lb=lb, ub=ub)
        
        if y_true is not None:
            out_spec = OutputSpec(kind=output_kind, y_true=y_true)
        else:
            # No labels - create a placeholder spec
            out_spec = OutputSpec(
                kind=OutKind.RANGE,
                lb=torch.full((lb.shape[0], 1), -float('inf')),
                ub=torch.full((lb.shape[0], 1), float('inf')),
            )
        
        return in_spec, out_spec
    
    @staticmethod
    def from_vnnlib(
        vnnlib_paths: Sequence[Union[str, Path]],
        input_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
    ) -> Tuple[InputSpec, OutputSpec]:
        """
        Create batched specs from VNNLIB files.
        
        Parses multiple VNNLIB files and stacks them into batched specs.
        
        Args:
            vnnlib_paths: List of paths to VNNLIB files
            input_shapes: Optional list of input shapes per file
            
        Returns:
            Tuple of (InputSpec, OutputSpec) with batch size = len(vnnlib_paths)
            
        Note:
            Currently supports BOX constraints only. VNNLIB files with
            complex disjunctive constraints are not fully supported.
        """
        from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_to_tensors
        
        all_lb = []
        all_ub = []
        all_c = []
        all_d = []
        
        for i, vnnlib_path in enumerate(vnnlib_paths):
            path = Path(vnnlib_path)
            if not path.exists():
                raise FileNotFoundError(f"VNNLIB file not found: {path}")
            
            input_shape = input_shapes[i] if input_shapes else None
            
            # Parse VNNLIB file
            input_tensor, metadata = parse_vnnlib_to_tensors(
                str(path),
                input_shape=input_shape
            )
            
            # Extract bounds from metadata
            if 'lb' in metadata and 'ub' in metadata:
                all_lb.append(metadata['lb'])
                all_ub.append(metadata['ub'])
            else:
                # Use center tensor as both lb and ub (point constraint)
                all_lb.append(input_tensor.squeeze(0))
                all_ub.append(input_tensor.squeeze(0))
            
            # Extract output constraints
            if 'output_c' in metadata and 'output_d' in metadata:
                all_c.append(metadata['output_c'])
                all_d.append(metadata['output_d'])
        
        # Stack into batched tensors
        lb = torch.stack(all_lb, dim=0)
        ub = torch.stack(all_ub, dim=0)
        
        in_spec = InputSpec(kind=InKind.BOX, lb=lb, ub=ub)
        
        # Create output spec
        if all_c and all_d:
            c = torch.stack(all_c, dim=0)
            d = torch.stack(all_d, dim=0)
            out_spec = OutputSpec(kind=OutKind.LINEAR_LE, c=c, d=d)
        else:
            # Default to range constraint
            out_spec = OutputSpec(
                kind=OutKind.RANGE,
                lb=torch.full((lb.shape[0], 1), -float('inf')),
                ub=torch.full((lb.shape[0], 1), float('inf')),
            )
        
        logger.debug(
            f"Loaded {len(vnnlib_paths)} VNNLIB files: "
            f"input={in_spec}, output={out_spec}"
        )
        
        return in_spec, out_spec
    
    @staticmethod
    def from_csv(
        csv_path: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None,
        eps: float = 0.03,
    ) -> List[Tuple[str, InputSpec, OutputSpec]]:
        """
        Load specs from a CSV configuration file.
        
        CSV format:
            image_path,label[,eps]
            
        Args:
            csv_path: Path to CSV file
            data_root: Root directory for relative image paths
            eps: Default perturbation radius (if not in CSV)
            
        Returns:
            List of (image_path, InputSpec, OutputSpec) tuples
        """
        import csv
        from PIL import Image
        from torchvision import transforms
        
        csv_path = Path(csv_path)
        data_root = Path(data_root) if data_root else csv_path.parent
        
        results = []
        to_tensor = transforms.ToTensor()
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header if present
            
            for row in reader:
                if len(row) < 2:
                    continue
                
                img_path = data_root / row[0]
                label = int(row[1])
                sample_eps = float(row[2]) if len(row) > 2 else eps
                
                # Load image
                img = Image.open(img_path)
                img_tensor = to_tensor(img).unsqueeze(0)  # [1, C, H, W]
                
                in_spec = InputSpec(
                    kind=InKind.LINF_BALL,
                    center=img_tensor,
                    eps=sample_eps,
                )
                out_spec = OutputSpec(
                    kind=OutKind.TOP1_ROBUST,
                    y_true=torch.tensor([label]),
                )
                
                results.append((str(img_path), in_spec, out_spec))
        
        logger.debug(f"Loaded {len(results)} samples from CSV: {csv_path}")
        
        return results
    
    @staticmethod
    def collate(
        specs: Sequence[Tuple[InputSpec, OutputSpec]]
    ) -> Tuple[InputSpec, OutputSpec]:
        """
        Collate multiple single-sample specs into a batched spec.
        
        Args:
            specs: List of (InputSpec, OutputSpec) tuples, each with B=1
            
        Returns:
            Single (InputSpec, OutputSpec) tuple with B=len(specs)
        """
        if not specs:
            raise ValueError("Cannot collate empty list of specs")
        
        in_specs = [s[0] for s in specs]
        out_specs = [s[1] for s in specs]
        
        # Get the kind from first spec
        in_kind = in_specs[0].kind
        out_kind = out_specs[0].kind
        
        # Collate input specs
        if in_kind == InKind.LINF_BALL:
            centers = torch.cat([s.center for s in in_specs], dim=0)
            # Handle scalar vs tensor eps
            if all(isinstance(s.eps, (int, float)) for s in in_specs):
                eps = in_specs[0].eps  # Assume all same
            else:
                eps = torch.tensor([
                    s.eps if isinstance(s.eps, (int, float)) else s.eps.item()
                    for s in in_specs
                ])
            in_spec = InputSpec(kind=InKind.LINF_BALL, center=centers, eps=eps)
        elif in_kind == InKind.BOX:
            lb = torch.cat([s.lb for s in in_specs], dim=0)
            ub = torch.cat([s.ub for s in in_specs], dim=0)
            in_spec = InputSpec(kind=InKind.BOX, lb=lb, ub=ub)
        else:
            raise NotImplementedError(f"Collate not implemented for {in_kind}")
        
        # Collate output specs
        if out_kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            y_true = torch.cat([
                s.y_true if isinstance(s.y_true, torch.Tensor) 
                else torch.tensor([s.y_true])
                for s in out_specs
            ], dim=0)
            out_spec = OutputSpec(kind=out_kind, y_true=y_true)
        elif out_kind == OutKind.LINEAR_LE:
            c = torch.cat([s.c for s in out_specs], dim=0)
            d = torch.cat([
                s.d if isinstance(s.d, torch.Tensor) else torch.tensor([s.d])
                for s in out_specs
            ], dim=0)
            out_spec = OutputSpec(kind=OutKind.LINEAR_LE, c=c, d=d)
        else:
            raise NotImplementedError(f"Collate not implemented for {out_kind}")
        
        return in_spec, out_spec
