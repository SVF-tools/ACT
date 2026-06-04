#===- act/front_end/vnnlib/create_specs.py - VNNLIB Spec Creator ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Create InputSpec and OutputSpec from VNNLIB benchmark instances.
#   Parses VNNLIB constraints and converts ONNX models to PyTorch.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import torch
import torch.nn as nn

from act.front_end.spec_creator_base import BaseSpecCreator, LabeledInputTensor
from act.front_end.specs import InputSpec, OutputSpec
from act.front_end.vnnlib_loader.data_model_loader import (
    list_downloaded_pairs,
    load_vnnlib_pair,
    list_local_categories
)
from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries

logger = logging.getLogger(__name__)


class VNNLibSpecCreator(BaseSpecCreator):
    """
    Create verification specifications from VNNLIB benchmark instances.
    
    Generates InputSpec and OutputSpec by parsing VNNLIB files:
    - Input specs: BOX constraints extracted from VNNLIB
    - Output specs: LINEAR_LE constraints from VNNLIB properties
    - Models: ONNX models converted to PyTorch
    
    Example:
        >>> creator = VNNLibSpecCreator(config_name="vnnlib_default")
        >>> results = creator.create_specs_for_data_model_pairs(
        ...     categories=["mnist_fc"],
        ...     max_instances=10
        ... )
        >>> 
        >>> for category, instance_id, pytorch_model, labeled_tensors, spec_pairs in results:
        ...     print(f"{category}/{instance_id}: {len(spec_pairs)} spec pairs")
    """
    
    def __init__(
        self,
        config_name: Optional[str] = "vnnlib_default",
        config_dict: Optional[Dict] = None
    ):
        """
        Initialize VNNLIB spec creator.
        
        Args:
            config_name: Name of YAML config file (without .yaml extension)
            config_dict: Direct config dict (overrides config_name if provided)
        """
        super().__init__(config_name, config_dict)
        
    def create_specs_for_data_model_pairs(
        self,
        categories: Optional[List[str]] = None,
        max_instances: Optional[int] = None,
        validate_shapes: bool = True
    ) -> List[Tuple[str, str, nn.Module, List[LabeledInputTensor], List[Tuple[InputSpec, OutputSpec]]]]:
        """
        Create specs for VNNLIB benchmark instances.
        
        Unified return format: List of (data_source, model_name, pytorch_model, labeled_tensors, spec_pairs)
        
        For VNNLIB:
        - data_source: Category name (e.g., "mnist_fc")
        - model_name: Instance identifier (e.g., "model_0_spec_5")
        - pytorch_model: ONNX model converted to PyTorch
        - labeled_tensors: List with single LabeledInputTensor from VNNLIB constraints
        - spec_pairs: List with single (InputSpec, OutputSpec) from VNNLIB
        
        Args:
            categories: List of benchmark categories (None = all downloaded)
            max_instances: Maximum instances per category (None = all)
            validate_shapes: Whether to validate specs against model
            
        Returns:
            List of tuples:
            - data_source: Category name
            - model_name: Instance identifier
            - pytorch_model: torch.nn.Module (converted from ONNX)
            - labeled_tensors: List containing single LabeledInputTensor
            - spec_pairs: List containing single (InputSpec, OutputSpec)
            
        Example:
            >>> creator = VNNLibSpecCreator()
            >>> results = creator.create_specs_for_data_model_pairs(
            ...     categories=["mnist_fc"],
            ...     max_instances=5
            ... )
        """
        logger.info(
            f"Creating VNNLIB specs: categories={categories}, "
            f"max_instances={max_instances}"
        )
        
        # Get all downloaded instances
        all_instances = list_downloaded_pairs()
        
        if not all_instances:
            logger.warning("No downloaded VNNLIB instances found")
            return []
        
        # Filter by categories if specified
        if categories is not None:
            categories_lower = [cat.lower() for cat in categories]
            all_instances = [
                inst for inst in all_instances
                if inst['category'].lower() in categories_lower
            ]
        
        if not all_instances:
            logger.warning("No instances match the specified categories")
            return []
        
        # Limit instances per category if specified
        if max_instances is not None:
            # Group by category and limit each
            category_instances = {}
            for inst in all_instances:
                cat = inst['category']
                if cat not in category_instances:
                    category_instances[cat] = []
                category_instances[cat].append(inst)
            
            # Take max_instances from each category
            all_instances = []
            for cat, instances in category_instances.items():
                all_instances.extend(instances[:max_instances])
        
        logger.info(f"Processing {len(all_instances)} VNNLIB instances")
        
        results = []
        
        # Cache converted models by (category, onnx_filename) so instances
        # sharing the same ONNX file reuse the same Python object.  This is
        # critical for model_synthesis.py which groups by id(pytorch_model).
        _model_cache: Dict[Tuple[str, str], nn.Module] = {}
        
        for instance_info in all_instances:
            category = instance_info['category']
            onnx_model = instance_info['onnx_model']
            vnnlib_spec = instance_info['vnnlib_spec']
            
            # Create instance identifier
            instance_id = f"{Path(onnx_model).stem}_{Path(vnnlib_spec).stem}"
            
            try:
                # Load instance
                logger.info(f"Loading instance: {category}/{instance_id}")
                instance_data = load_vnnlib_pair(
                    category=category,
                    onnx_model=onnx_model,
                    vnnlib_spec=vnnlib_spec,
                    auto_download=False  # Already filtered to downloaded
                )
                
                # Reuse cached model if same ONNX file was already converted
                cache_key = (category, onnx_model)
                if cache_key in _model_cache:
                    instance_data['model'] = _model_cache[cache_key]
                else:
                    _model_cache[cache_key] = instance_data['model']
                
                # Generate specs for this instance
                result = self._create_specs_for_single_instance(
                    category=category,
                    instance_id=instance_id,
                    instance_data=instance_data,
                    validate_shapes=validate_shapes
                )
                
                if result is not None:
                    results.append(result)
                
                # Memory optimization: Free instance_data after extracting model/specs
                # (model itself is kept alive via _model_cache)
                import gc
                del instance_data
                gc.collect()
                
            except Exception as e:
                logger.error(
                    f"Failed to create specs for {category}/{instance_id}: {e}"
                )
        
        logger.info(f"Successfully created specs for {len(results)} instances")
        return results
    
    def _create_specs_for_single_instance(
        self,
        category: str,
        instance_id: str,
        instance_data: Dict,
        validate_shapes: bool
    ) -> Optional[Tuple[str, str, nn.Module, List[LabeledInputTensor], List[Tuple[InputSpec, OutputSpec]]]]:
        """
        Create specs for a single VNNLIB instance.
        
        Returns:
            Tuple of (category, instance_id, pytorch_model, labeled_tensors, spec_pairs)
            or None if failed
        """
        logger.info(f"Generating specs for {category}/{instance_id}")
        
        pytorch_model = instance_data['model']
        labeled_tensor = instance_data['labeled_tensor']
        vnnlib_path = Path(instance_data['vnnlib_path'])
        
        # Parse VNNLIB to create specs
        # Pass input_shape to ensure specs match tensor shape (not flattened)
        # Pass true_label to promote RANGE to TOP1_ROBUST for classification
        try:
            queries = parse_vnnlib_queries(
                vnnlib_path,
                labeled_tensor=labeled_tensor,
            )
            if not queries:
                raise RuntimeError("parse_vnnlib_queries returned no queries")
            logger.info(
                f"Parsed VNNLIB specs: {len(queries)} queries, "
                f"first kind=({queries[0][0].kind}, {queries[0][1].kind})"
            )
        except Exception as e:
            logger.error(f"Failed to parse VNNLIB specs: {e}")
            return None
        
        spec_pairs = list(queries)
        
        # Validate if requested
        if validate_shapes:
            validated_pairs = self._validate_and_filter_specs(
                spec_pairs,
                pytorch_model,
                labeled_tensor.tensor
            )
            
            if not validated_pairs:
                logger.warning(f"Spec validation failed for {category}/{instance_id}")
                return None
            
            spec_pairs = validated_pairs
        
        # Return in unified format with labeled_tensors as list
        labeled_tensors = [labeled_tensor]
        
        return (category, instance_id, pytorch_model, labeled_tensors, spec_pairs)
    
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
    
    def list_categories(self) -> List[str]:
        """
        List locally downloaded VNNLIB benchmark categories.
        
        Returns:
            List of category names
            
        Example:
            >>> creator = VNNLibSpecCreator()
            >>> categories = creator.list_categories()
            >>> print(categories)
            ['mnist_fc', 'cifar10_resnet']
        """
        return list_local_categories()


# Convenience function for quick usage
def create_vnnlib_specs(
    categories: Optional[List[str]] = None,
    max_instances: Optional[int] = None,
    config_name: str = "vnnlib_default"
) -> List[Tuple[str, str, nn.Module, List[LabeledInputTensor], List[Tuple[InputSpec, OutputSpec]]]]:
    """
    Convenience function to create VNNLIB specs with default settings.
    
    Args:
        categories: List of benchmark categories (None = all)
        max_instances: Max instances per category (None = all)
        config_name: Configuration preset name
        
    Returns:
        List of (category, instance_id, pytorch_model, labeled_tensors, spec_pairs)
        
    Example:
        >>> results = create_vnnlib_specs(["mnist_fc"], max_instances=10)
    """
    creator = VNNLibSpecCreator(config_name=config_name)
    return creator.create_specs_for_data_model_pairs(
        categories=categories,
        max_instances=max_instances
    )
