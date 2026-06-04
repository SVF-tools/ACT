#===- act/front_end/torchvision/create_specs.py - TorchVision Specs ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Create InputSpec and OutputSpec from TorchVision dataset-model pairs.
#   Sample-based spec generation for image classification models.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import torch
import torch.nn as nn

from act.front_end.spec_creator_base import BaseSpecCreator, LabeledInputTensor
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.torchvision_loader.data_model_loader import (
    list_downloaded_pairs,
    load_dataset_model_pair
)
from act.util.device_manager import get_default_device

logger = logging.getLogger(__name__)


class TorchVisionSpecCreator(BaseSpecCreator):
    """
    Create verification specifications from TorchVision dataset-model pairs.
    
    Generates InputSpec and OutputSpec based on actual data samples:
    - Input specs: BOX or LINF_BALL perturbations around sample images
    - Output specs: Classification robustness properties (MARGIN_ROBUST)
    
    Example:
        >>> creator = TorchVisionSpecCreator(config_name="torchvision_classification")
        >>> results = creator.create_specs_for_data_model_pairs(
        ...     dataset_names=["MNIST"],
        ...     model_names=["simple_cnn"],
        ...     num_samples=10
        ... )
        >>> 
        >>> for data_source, model_name, pytorch_model, labeled_tensors, spec_pairs in results:
        ...     print(f"{data_source} + {model_name}: {len(spec_pairs)} spec pairs")
    """
    
    def __init__(
        self,
        config_name: Optional[str] = "torchvision_classification",
        config_dict: Optional[Dict] = None
    ):
        """
        Initialize TorchVision spec creator.
        
        Args:
            config_name: Name of YAML config file (without .yaml extension)
            config_dict: Direct config dict (overrides config_name if provided)
        """
        super().__init__(config_name, config_dict)
    
    def create_specs_for_data_model_pairs(
        self,
        dataset_names: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        num_samples: int = 10,
        start_index: int = 0,
        split: str = "test",
        validate_shapes: bool = True
    ) -> List[Tuple[str, str, nn.Module, List[LabeledInputTensor], List[Tuple[InputSpec, OutputSpec]]]]:
        """
        Create specs for TorchVision dataset-model pairs.
        
        Unified return format: List of (data_source, model_name, pytorch_model, labeled_tensors, spec_pairs)
        
        Args:
            dataset_names: List of dataset names (None = all downloaded)
            model_names: List of model names (None = all for each dataset)
            num_samples: Number of samples to generate specs for
            start_index: Starting index in dataset
            split: Dataset split ('train' or 'test')
            validate_shapes: Whether to validate specs against model
            
        Returns:
            List of tuples:
            - data_source: Dataset name (e.g., "MNIST")
            - model_name: Model name (e.g., "simple_cnn")
            - pytorch_model: torch.nn.Module
            - labeled_tensors: List of LabeledInputTensor instances
            - spec_pairs: List of (InputSpec, OutputSpec) tuples
            
        Example:
            >>> creator = TorchVisionSpecCreator()
            >>> results = creator.create_specs_for_data_model_pairs(
            ...     dataset_names=["MNIST"],
            ...     num_samples=5
            ... )
        """
        logger.info(
            f"Creating TorchVision specs: datasets={dataset_names}, "
            f"models={model_names}, samples={num_samples}"
        )
        
        # Get all downloaded pairs
        all_pairs = list_downloaded_pairs()
        
        if not all_pairs:
            logger.warning("No downloaded dataset-model pairs found")
            return []
        
        # Filter by dataset names if specified
        if dataset_names is not None:
            dataset_names_lower = [name.lower() for name in dataset_names]
            all_pairs = [
                p for p in all_pairs 
                if p['dataset'].lower() in dataset_names_lower
            ]
        
        # Filter by model names if specified
        if model_names is not None:
            model_names_lower = [name.lower() for name in model_names]
            all_pairs = [
                p for p in all_pairs 
                if p['model'].lower() in model_names_lower
            ]
        
        if not all_pairs:
            logger.warning("No pairs match the specified filters")
            return []
        
        logger.info(f"Processing {len(all_pairs)} dataset-model pairs")
        
        results = []
        
        for pair_info in all_pairs:
            dataset_name = pair_info['dataset']
            model_name = pair_info['model']
            
            try:
                # Load pair
                logger.info(f"Loading pair: {dataset_name} + {model_name}")
                pair_data = load_dataset_model_pair(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    split=split,
                    batch_size=1,
                    shuffle=False,
                    auto_download=False  # Already filtered to downloaded pairs
                )
                
                pytorch_model = pair_data['model']
                dataloader = pair_data['dataloader']
                
                # Generate specs for this pair
                result = self._create_specs_for_single_instance(
                    data_source=dataset_name,
                    model_name=model_name,
                    pytorch_model=pytorch_model,
                    dataloader=dataloader,
                    num_samples=num_samples,
                    start_index=start_index,
                    validate_shapes=validate_shapes
                )
                
                if result is not None:
                    results.append(result)
                
                # Memory optimization: Free dataset/dataloader after extracting input_tensors
                # pair_data contains the dataset (476 MB for MNIST) which is no longer needed
                import gc
                del pair_data, dataloader
                gc.collect()
                
            except Exception as e:
                logger.error(
                    f"Failed to create specs for {dataset_name} + {model_name}: {e}"
                )
        
        logger.info(f"Successfully created specs for {len(results)} pairs")
        return results
    
    def _create_specs_for_single_instance(
        self,
        data_source: str,
        model_name: str,
        pytorch_model: nn.Module,
        dataloader,
        num_samples: int,
        start_index: int,
        validate_shapes: bool
    ) -> Optional[Tuple[str, str, nn.Module, List[LabeledInputTensor], List[Tuple[InputSpec, OutputSpec]]]]:
        """
        Create specs for a single dataset-model pair.
        
        Returns:
            Tuple of (data_source, model_name, pytorch_model, labeled_tensors, spec_pairs)
            or None if failed
        """
        logger.info(f"Generating specs for {data_source} + {model_name}")
        
        # Collect samples as LabeledInputTensors
        labeled_tensors = []
        
        for idx, (images, targets) in enumerate(dataloader):
            if idx < start_index:
                continue
            if len(labeled_tensors) >= num_samples:
                break
            
            # DataLoader always returns CPU tensors regardless of torch.set_default_device
            device = get_default_device()
            tensor = images.to(device)
            label = targets.to(device)
            labeled_tensors.append(LabeledInputTensor(tensor=tensor, label=label))
        
        if not labeled_tensors:
            logger.warning(f"No samples collected for {data_source}")
            return None
        
        logger.info(f"Collected {len(labeled_tensors)} samples")
        
        # Generate spec pairs for EACH sample
        all_spec_pairs = []
        
        for labeled_tensor in labeled_tensors:
            # Unpack tensor and label
            tensor, label = labeled_tensor
            
            # Generate input specs for this sample
            sample_input_specs = self._generate_input_specs_for_sample(tensor)
            
            # Generate output specs for this sample's label
            sample_output_specs = self._generate_output_specs_for_label(label)
            
            # Create combinations for this sample
            sample_spec_pairs = self._create_spec_combinations(sample_input_specs, sample_output_specs)
            
            all_spec_pairs.extend(sample_spec_pairs)
        
        spec_pairs = all_spec_pairs
        
        logger.info(f"Generated {len(spec_pairs)} spec combinations")
        
        # Validate if requested
        if validate_shapes:
            validated_pairs = self._validate_and_filter_specs(
                spec_pairs,
                pytorch_model,
                labeled_tensors[0].tensor  # Use first sample for shape
            )
            
            if len(validated_pairs) < len(spec_pairs):
                logger.warning(
                    f"Filtered {len(spec_pairs) - len(validated_pairs)} invalid specs"
                )
            
            spec_pairs = validated_pairs
        
        if not spec_pairs:
            logger.warning(f"No valid specs generated for {data_source} + {model_name}")
            return None
        
        return (data_source, model_name, pytorch_model, labeled_tensors, spec_pairs)
    
    def _generate_input_specs_for_sample(self, sample_tensor: torch.Tensor) -> List[InputSpec]:
        """
        Generate input specifications for a single sample tensor.
        
        Creates BOX and/or LINF_BALL specs based on configuration.
        
        Args:
            sample_tensor: Single input sample tensor
            
        Returns:
            List of InputSpec objects for this sample
        """
        input_specs = []
        
        # Get epsilon values from config
        epsilons = self.config.get('epsilons', [0.01, 0.03, 0.05])
        
        # Get input kinds to generate
        input_kinds = self.config.get('input_kinds', ['BOX', 'LINF_BALL'])
        
        for kind in input_kinds:
            if kind == 'BOX':
                # BOX: lb and ub bounds
                for eps in epsilons:
                    lb = torch.clamp(sample_tensor - eps, 0.0, 1.0)
                    ub = torch.clamp(sample_tensor + eps, 0.0, 1.0)
                    
                    input_specs.append(InputSpec(
                        kind=InKind.BOX,
                        lb=lb,
                        ub=ub
                    ))
            
            elif kind == 'LINF_BALL':
                # LINF_BALL: center and epsilon
                for eps in epsilons:
                    input_specs.append(InputSpec(
                        kind=InKind.LINF_BALL,
                        center=sample_tensor.clone(),
                        eps=torch.tensor(eps)
                    ))
        
        return input_specs
    
    def _generate_output_specs_for_label(self, label: torch.Tensor) -> List[OutputSpec]:
        """
        Generate output specifications for a single label (batch-native).
        
        Creates MARGIN_ROBUST and/or TOP1_ROBUST specs based on configuration.
        
        Args:
            label: Ground truth label tensor (1,) shape, preserves device
            
        Returns:
            List of OutputSpec objects for this label
        """
        output_specs = []
        
        # Get output kinds to generate
        output_kinds = self.config.get('output_kinds', ['MARGIN_ROBUST'])
        
        # Get margin values
        margins = self.config.get('margins', [0.0])
        
        # No need to extract device - device_manager ensures all tensors use default device
        
        for kind in output_kinds:
            if kind == 'MARGIN_ROBUST':
                # MARGIN_ROBUST: classification with margin
                for margin in margins:
                    output_specs.append(OutputSpec(
                        kind=OutKind.MARGIN_ROBUST,
                        y_true=label.clone(),  # Use label tensor directly (already (1,) shape)
                        margin=torch.tensor([margin])
                    ))
            
            elif kind == 'TOP1_ROBUST':
                # TOP1_ROBUST: top-1 classification
                output_specs.append(OutputSpec(
                    kind=OutKind.TOP1_ROBUST,
                    y_true=label.clone()  # Use label tensor directly (already (1,) shape)
                ))
        
        return output_specs
    
    def _generate_input_specs(self, input_tensors: List[torch.Tensor]) -> List[InputSpec]:
        """
        Generate input specifications from sample tensors.
        
        Creates BOX and/or LINF_BALL specs based on configuration.
        For each input tensor, generates all configured spec types.
        
        Args:
            input_tensors: List of input sample tensors
            
        Returns:
            List of InputSpec objects (one set per input tensor)
        """
        input_specs = []
        
        # Get epsilon values from config
        epsilons = self.config.get('epsilons', [0.01, 0.03, 0.05])
        
        # Get input kinds to generate
        input_kinds = self.config.get('input_kinds', ['BOX', 'LINF_BALL'])
        
        # Generate specs for EACH input tensor
        for sample_idx, sample_tensor in enumerate(input_tensors):
            for kind in input_kinds:
                if kind == 'BOX':
                    # BOX: lb and ub bounds
                    for eps in epsilons:
                        lb = torch.clamp(sample_tensor - eps, 0.0, 1.0)
                        ub = torch.clamp(sample_tensor + eps, 0.0, 1.0)
                        
                        input_specs.append(InputSpec(
                            kind=InKind.BOX,
                            lb=lb,
                            ub=ub
                        ))
                
                elif kind == 'LINF_BALL':
                    # LINF_BALL: center and epsilon
                    for eps in epsilons:
                        input_specs.append(InputSpec(
                            kind=InKind.LINF_BALL,
                            center=sample_tensor.clone(),
                            eps=torch.tensor(eps)
                        ))
        
        logger.debug(f"Generated {len(input_specs)} input specs from {len(input_tensors)} samples")
        return input_specs
    
    def _generate_output_specs(self, labels: List[int]) -> List[OutputSpec]:
        """
        Generate output specifications for classification.
        
        Creates MARGIN_ROBUST and/or TOP1_ROBUST specs based on configuration.
        For each label, generates all configured output spec types.
        
        Args:
            labels: List of ground truth labels
            
        Returns:
            List of OutputSpec objects (one set per label)
        """
        output_specs = []
        
        # Get output kinds to generate
        output_kinds = self.config.get('output_kinds', ['MARGIN_ROBUST'])
        
        # Get margin values
        margins = self.config.get('margins', [0.0])
        
        # Generate specs for EACH label
        for y_true in labels:
            for kind in output_kinds:
                if kind == 'MARGIN_ROBUST':
                    # MARGIN_ROBUST: classification with margin
                    for margin in margins:
                        output_specs.append(OutputSpec(
                            kind=OutKind.MARGIN_ROBUST,
                            y_true=torch.tensor([y_true], dtype=torch.int64),  # Batch: (1,) shape
                            margin=torch.tensor([margin])  # Batch: (1,) shape
                        ))
                
                elif kind == 'TOP1_ROBUST':
                    # TOP1_ROBUST: top-1 classification
                    output_specs.append(OutputSpec(
                        kind=OutKind.TOP1_ROBUST,
                        y_true=torch.tensor([y_true], dtype=torch.int64)  # Batch: (1,) shape
                    ))
        
        logger.debug(f"Generated {len(output_specs)} output specs from {len(labels)} labels")
        return output_specs
    
    def _validate_and_filter_specs(
        self,
        spec_pairs: List[Tuple[InputSpec, OutputSpec]],
        pytorch_model: nn.Module,
        sample_input: torch.Tensor
    ) -> List[Tuple[InputSpec, OutputSpec]]:
        """
        Validate spec pairs against model and filter invalid ones.
        
        Args:
            spec_pairs: List of (InputSpec, OutputSpec) tuples
            pytorch_model: PyTorch model to validate against
            sample_input: Sample input tensor for shape inference
            
        Returns:
            Filtered list of valid spec pairs
        """
        valid_pairs = []
        
        for input_spec, output_spec in spec_pairs:
            try:
                is_valid, errors = self.validate_spec_pair_with_model(
                    input_spec,
                    output_spec,
                    pytorch_model,
                    sample_input
                )
                
                if is_valid:
                    valid_pairs.append((input_spec, output_spec))
                else:
                    logger.debug(
                        f"Spec pair validation failed: {input_spec.kind}, {output_spec.kind}, with errors:\n"
                        f"{"\n".join(errors)}"
                    )
            
            except Exception as e:
                logger.debug(f"Spec validation error: {e}")
        
        return valid_pairs


# Convenience function for quick usage
def create_torchvision_specs(
    dataset_names: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    num_samples: int = 10,
    config_name: str = "torchvision_classification"
) -> List[Tuple[str, str, nn.Module, List[LabeledInputTensor], List[Tuple[InputSpec, OutputSpec]]]]:
    """
    Convenience function to create TorchVision specs with default settings.
    
    Args:
        dataset_names: List of dataset names (None = all)
        model_names: List of model names (None = all)
        num_samples: Number of samples per pair
        config_name: Configuration preset name
        
    Returns:
        List of (data_source, model_name, pytorch_model, labeled_tensors, spec_pairs)
        
    Example:
        >>> results = create_torchvision_specs(["MNIST"], ["simple_cnn"], num_samples=5)
    """
    creator = TorchVisionSpecCreator(config_name=config_name)
    return creator.create_specs_for_data_model_pairs(
        dataset_names=dataset_names,
        model_names=model_names,
        num_samples=num_samples
    )
