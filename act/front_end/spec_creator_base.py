"""
Base class for specification creators with shape validation.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
import yaml
import torch
from pathlib import Path

from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.util.path_config import get_spec_config_path, get_default_spec_config_path


@dataclass
class LabeledInputTensor:
    """
    Pairs an input tensor with its ground truth label.
    
    This unified data structure ensures tensor-label consistency throughout
    the verification pipeline, from data loading to spec generation.
    
    IMPORTANT: Tensors must include batch dimension (batch=1) following
    PyTorch convention for model inference.
    
    Attributes:
        tensor: Input tensor WITH batch dimension (1, C, H, W) or (1, F)
               Following PyTorch DataLoader convention for model inference
        label: Ground truth label (None if unavailable, e.g., for unlabeled data)
    
    Examples:
        >>> # Create from tensor and label (batch-native: both tensors)
        >>> tensor = torch.randn(1, 3, 32, 32)  # (B=1, C=3, H=32, W=32)
        >>> label = torch.tensor([5], dtype=torch.int64)  # (B=1,)
        >>> labeled = LabeledInputTensor(tensor, label=label)
        
        >>> # Tuple unpacking (both are tensors)
        >>> img, lbl = labeled
        >>> assert torch.equal(img, tensor) and torch.equal(lbl, label)
        
        >>> # Property access
        >>> assert labeled.tensor.shape == (1, 3, 32, 32)  # Batch dimension included
        >>> assert labeled.label.shape == (1,) and labeled.label.item() == 5
        
        >>> # Device management
        >>> cuda_labeled = labeled.to('cuda')
        >>> assert cuda_labeled.tensor.device.type == 'cuda'
        >>> assert cuda_labeled.label.device.type == 'cuda'
        
        >>> # Unlabeled data
        >>> unlabeled = LabeledInputTensor(tensor, label=None)
        >>> _, lbl = unlabeled
        >>> assert lbl is None
        
        Note:
            The batch dimension (first dimension = 1) is REQUIRED for:
            - PyTorch model inference (CNN layers expect BCHW format)
            - Consistency with PyTorch DataLoader convention
            - Direct use in model.forward() without shape manipulation
    """
    
    tensor: torch.Tensor
    label: Optional[torch.Tensor] = None  # (N,) tensor or None
    
    def __post_init__(self):
        """Validate inputs after initialization."""
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError(
                f"tensor must be torch.Tensor, got {type(self.tensor).__name__}"
            )
        
        # label must be tensor or None
        if self.label is not None and not isinstance(self.label, torch.Tensor):
            raise TypeError(
                f"label must be torch.Tensor or None, got {type(self.label).__name__}"
            )
    
    def __getitem__(self, key: int) -> Optional[torch.Tensor]:
        """
        Enable tuple-like unpacking (both tensor and label are tensors).
        
        Args:
            key: Index (0 for tensor, 1 for label)
        
        Returns:
            tensor if key==0, label tensor if key==1 (or None)
        
        Raises:
            IndexError: If key not in [0, 1]
        
        Examples:
            >>> labeled = LabeledInputTensor(
            ...     torch.randn(1, 3, 32, 32),
            ...     label=torch.tensor([5], dtype=torch.int64)
            ... )
            >>> tensor, label = labeled  # Unpacks via __getitem__
            >>> assert tensor.shape == (1, 3, 32, 32)
            >>> assert label.shape == (1,) and label.item() == 5
        """
        if key == 0:
            return self.tensor
        elif key == 1:
            return self.label
        else:
            raise IndexError(
                f"LabeledInputTensor index must be 0 (tensor) or 1 (label), got {key}"
            )
    
    def __len__(self) -> int:
        """Return length for tuple unpacking (always 2)."""
        return 2
    
    def to(self, device: Union[str, torch.device]) -> LabeledInputTensor:
        """
        Move tensor and label to specified device.
        
        Args:
            device: Target device ('cpu', 'cuda', torch.device, etc.)
        
        Returns:
            New LabeledInputTensor with both tensor and label on target device
        
        Examples:
            >>> tensor = torch.randn(1, 3, 32, 32)
            >>> label = torch.tensor([5], dtype=torch.int64)
            >>> labeled = LabeledInputTensor(tensor, label=label)
            >>> cuda_labeled = labeled.to('cuda')
            >>> assert cuda_labeled.tensor.device.type == 'cuda'
            >>> assert cuda_labeled.label.device.type == 'cuda'
        """
        return LabeledInputTensor(
            tensor=self.tensor.to(device),
            label=self.label.to(device) if self.label is not None else None
        )
    
    def cpu(self) -> LabeledInputTensor:
        """
        Move tensor to CPU.
        
        Returns:
            New LabeledInputTensor with tensor on CPU
        """
        return self.to('cpu')
    
    def cuda(self, device: Optional[int] = None) -> LabeledInputTensor:
        """
        Move tensor to CUDA device.
        
        Args:
            device: CUDA device index (None for current device)
        
        Returns:
            New LabeledInputTensor with tensor on CUDA device
        """
        if device is None:
            return self.to('cuda')
        else:
            return self.to(f'cuda:{device}')
    
    def detach(self) -> LabeledInputTensor:
        """
        Detach tensor and label from computation graph.
        
        Returns:
            New LabeledInputTensor with detached tensor and label
        """
        return LabeledInputTensor(
            tensor=self.tensor.detach(),
            label=self.label.detach() if self.label is not None else None
        )
    
    def clone(self) -> LabeledInputTensor:
        """
        Create a deep copy.
        
        Returns:
            New LabeledInputTensor with cloned tensor and label
        """
        return LabeledInputTensor(
            tensor=self.tensor.clone(),
            label=self.label.clone() if self.label is not None else None
        )
    
    @property
    def shape(self) -> torch.Size:
        """Convenience property for tensor shape."""
        return self.tensor.shape
    
    @property
    def device(self) -> torch.device:
        """Convenience property for tensor device."""
        return self.tensor.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Convenience property for tensor dtype."""
        return self.tensor.dtype
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        if self.label is not None:
            # Display label in readable format (int if single element, list otherwise)
            label_display = self.label.tolist() if self.label.numel() > 1 else self.label.item()
            label_str = f"label={label_display}"
        else:
            label_str = "label=None"
        return (
            f"LabeledInputTensor(shape={tuple(self.tensor.shape)}, "
            f"{label_str}, device={self.tensor.device})"
        )


class BaseSpecCreator(ABC):
    """
    Abstract base class for creating InputSpec/OutputSpec pairs.
    
    Provides shared configuration, validation, and spec generation utilities
    for different spec sources (TorchVision datasets, VNNLIB files, etc.)
    """
    
    def __init__(
        self,
        config_name: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize spec creator with configuration.
        
        Args:
            config_name: Named config from configs/specs/ (e.g., 'torchvision_classification')
            config_dict: Runtime configuration overrides
        """
        self.config = self._load_config(config_name)
        if config_dict:
            self.config.update(config_dict)
    
    def _load_config(self, config_name: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if config_name:
                config_path = get_spec_config_path(config_name)
            else:
                config_path = get_default_spec_config_path()
            
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  Could not load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Fallback default configuration"""
        return {
            'input_spec_types': ['BOX', 'LINF_BALL'],
            'output_spec_types': ['MARGIN_ROBUST', 'TOP1_ROBUST'],
            'perturbation': {'epsilon_values': [0.01, 0.03], 'norm': 'inf'},
            'combination_strategy': 'balanced'
        }
    
    # ==================== SHAPE VALIDATION ==================== #
    
    def _get_model_io_shapes(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor
    ) -> Tuple[torch.Size, torch.Size]:
        """
        Get model input and output shapes by running inference.
        
        Args:
            model: PyTorch model
            sample_input: Sample input tensor
        
        Returns:
            (input_shape, output_shape) tuple
        """
        model.eval()
        with torch.no_grad():
            # sample_input already has batch dimension (1, C, H, W)
            test_input = sample_input
            output = model(test_input)
        
        return sample_input.shape, output.shape
    
    def _validate_input_spec_shape(
        self,
        spec: InputSpec,
        expected_shape: torch.Size
    ) -> Tuple[bool, str]:
        """
        Validate InputSpec shape matches expected dimensions.
        
        Returns:
            (is_valid, error_message) tuple
        """
        if spec.kind == InKind.BOX:
            if spec.lb.shape != expected_shape:
                return False, f"BOX lb shape {spec.lb.shape} != expected {expected_shape}"
            if spec.ub.shape != expected_shape:
                return False, f"BOX ub shape {spec.ub.shape} != expected {expected_shape}"
        
        elif spec.kind == InKind.LINF_BALL:
            if spec.center.shape != expected_shape:
                return False, f"LINF_BALL center shape {spec.center.shape} != expected {expected_shape}"
        
        elif spec.kind == InKind.LIN_POLY:
            flat_size = expected_shape.numel()
            if spec.A.shape[1] != flat_size:
                return False, f"LIN_POLY A columns {spec.A.shape[1]} != flattened input {flat_size}"
        
        return True, ""
    
    def _validate_output_spec_shape(
        self,
        spec: OutputSpec,
        num_classes: int
    ) -> Tuple[bool, str]:
        """
        Validate OutputSpec is compatible with model output.
        
        Returns:
            (is_valid, error_message) tuple
        """
        if spec.kind in [OutKind.MARGIN_ROBUST, OutKind.TOP1_ROBUST]:
            y_true_valid_class = (0 <= spec.y_true).logical_and(spec.y_true < num_classes)

            if not y_true_valid_class.all():
                return False, f"Class label {spec.y_true} out of range [0, {num_classes})"
        
        elif spec.kind == OutKind.LINEAR_LE:
            if spec.c.numel() != num_classes:
                return False, f"LINEAR_LE coeff size {spec.c.numel()} != num_classes {num_classes}"
        
        elif spec.kind == OutKind.RANGE:
            # Range specs are always valid (can be per-output or global)
            pass
        
        return True, ""
    
    def validate_spec_pair_with_model(
        self,
        input_spec: InputSpec,
        output_spec: OutputSpec,
        model: torch.nn.Module,
        sample_input: torch.Tensor
    ) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation: spec shapes match model I/O.
        
        Returns:
            (is_valid, error_messages) tuple
        """
        errors = []
        
        # Get model I/O shapes
        try:
            input_shape, output_shape = self._get_model_io_shapes(model, sample_input)
            # Assume last dimension is classes for classification
            if len(output_shape) > 1:
                num_classes = output_shape[-1]
            else:
                num_classes = output_shape[0]
        except Exception as e:
            errors.append(f"Failed to infer model shapes: {e}")
            return False, errors
        
        # Validate input spec
        input_valid, input_err = self._validate_input_spec_shape(input_spec, input_shape)
        if not input_valid:
            errors.append(f"Input spec validation failed: {input_err}")
        
        # Validate output spec
        output_valid, output_err = self._validate_output_spec_shape(output_spec, num_classes)
        if not output_valid:
            errors.append(f"Output spec validation failed: {output_err}")
        
        return len(errors) == 0, errors
    
    # ==================== SPEC COMBINATION ==================== #
    
    def _create_spec_combinations(
        self,
        input_specs: List[InputSpec],
        output_specs: List[OutputSpec]
    ) -> List[Tuple[InputSpec, OutputSpec]]:
        """Create spec combinations based on strategy"""
        strategy = self.config.get('combination_strategy', 'balanced')
        
        if strategy == 'full':
            # Cartesian product
            return [(i, o) for i in input_specs for o in output_specs]
        elif strategy == 'minimal':
            # One-to-one pairing (truncate to min length)
            min_len = min(len(input_specs), len(output_specs))
            return list(zip(input_specs[:min_len], output_specs[:min_len]))
        else:  # 'balanced'
            # Balanced pairing (cycle shorter list)
            if not input_specs or not output_specs:
                return []
            pairs = []
            longer = input_specs if len(input_specs) >= len(output_specs) else output_specs
            shorter = output_specs if len(input_specs) >= len(output_specs) else input_specs
            is_input_longer = len(input_specs) >= len(output_specs)
            
            for i, item in enumerate(longer):
                paired_item = shorter[i % len(shorter)]
                if is_input_longer:
                    pairs.append((item, paired_item))
                else:
                    pairs.append((paired_item, item))
            return pairs
    
    @abstractmethod
    def create_specs_for_data_model_pairs(
        self,
        max_samples: Optional[int] = None,
        filter_fn: Optional[Callable] = None,
        validate_shapes: bool = True
    ) -> List[Tuple[str, str, torch.nn.Module, List[torch.Tensor], List[Tuple[InputSpec, OutputSpec]]]]:
        """
        Create specs for data-model pairs (must be implemented by subclasses).
        
        Args:
            max_samples: Maximum number of samples/instances to process
            filter_fn: Optional filter function (source, model) -> bool
            validate_shapes: Whether to validate spec shapes with model
        
        Returns:
            List of (data_source, model_name, pytorch_model, input_tensors, spec_pairs) tuples
            - data_source: Dataset/category identifier
            - model_name: Model identifier
            - pytorch_model: PyTorch nn.Module
            - input_tensors: List of input tensors
            - spec_pairs: List of (InputSpec, OutputSpec) tuples
        """
        pass
