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
from typing import Any, List, Tuple, Dict, Optional, override
import logging
import torch.nn as nn

from act.front_end.spec_creator_base import BaseSpecCreator, LabeledInputTensor
from act.front_end.specs import InputSpec, OutputSpec
from act.front_end.vnnlib_loader.data_model_loader import (
    list_downloaded_pairs,
    load_vnnlib_pair,
    list_local_categories
)
from act.front_end.vnnlib_loader.vnnlib_parser import (
    UnsupportedSpecError,
    VNNLibParseError,
    parse_vnnlib_queries,
)

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
        config_dict: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize VNNLIB spec creator.
        
        Args:
            config_name: Name of YAML config file (without .yaml extension)
            config_dict: Direct config dict (overrides config_name if provided)
        """
        super().__init__(config_name, config_dict)
        
    @override
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
        
        # Cache everything derived from the ONNX file(s) -- converted models and
        # declared input shape -- by (category, onnx filenames), so instances
        # sharing a file read and convert it once and reuse the same Python
        # object.  Object identity is critical for model_synthesis.py which
        # groups by id(pytorch_model).  Deliberately scoped to this call:
        # convert_onnx_to_pytorch() moves the model to the active device/dtype in
        # place, so a process-wide cache could hand back a model mutated for a
        # different run.
        _onnx_cache: Dict[Tuple[str, str, Optional[str]], Dict[str, Any]] = {}
        
        failures: List[str] = []
        for instance_info in all_instances:
            category = instance_info['category']
            onnx_model = instance_info['onnx_model']
            vnnlib_spec = instance_info['vnnlib_spec']

            # Dual-model (isomorphic) instances carry a second ONNX model (g).
            onnx_model_g = None
            if instance_info.get('is_dual_model') and len(instance_info.get('onnx_models', [])) > 1:
                onnx_model_g = instance_info['onnx_models'][1][1]

            # Create instance identifier
            instance_id = f"{Path(onnx_model).stem}_{Path(vnnlib_spec).stem}"
            
            try:
                # Load instance
                logger.info(f"Loading instance: {category}/{instance_id}")
                cache_key = (category, onnx_model, onnx_model_g)
                instance_data = load_vnnlib_pair(
                    category=category,
                    onnx_model=onnx_model,
                    vnnlib_spec=vnnlib_spec,
                    onnx_model_g=onnx_model_g,
                    auto_download=False,  # Already filtered to downloaded
                    onnx_cache=_onnx_cache.setdefault(cache_key, {})
                )
                
                # Generate specs for this instance
                result = self._create_specs_for_single_instance(
                    category=category,
                    instance_id=instance_id,
                    instance_data=instance_data,
                    validate_shapes=validate_shapes
                )
                
                results.append(result)
                
            except UnsupportedSpecError:
                raise
            except Exception as e:  # noqa: BLE001 — collect every failure, raise below
                logger.error(
                    f"Failed to create specs for {category}/{instance_id}: {e}"
                )
                failures.append(f"{category}/{instance_id}: {e}")
        
        if failures:
            raise RuntimeError(
                f"Spec creation failed for {len(failures)}/{len(all_instances)} VNNLIB instance(s):\n  "
                + "\n  ".join(failures)
            )
        logger.info(f"Successfully created specs for {len(results)} instances")
        return results
    
    def _create_specs_for_single_instance(
        self,
        category: str,
        instance_id: str,
        instance_data: Dict[str, Any],
        validate_shapes: bool
    ) -> Tuple[str, str, nn.Module, List[LabeledInputTensor], List[Tuple[InputSpec, OutputSpec]]]:
        """
        Create specs for a single VNNLIB instance.
        
        Returns:
            Tuple of (category, instance_id, pytorch_model, labeled_tensors, spec_pairs)
        """
        logger.info(f"Generating specs for {category}/{instance_id}")
        
        pytorch_model = instance_data['model']
        labeled_tensor = instance_data['labeled_tensor']
        vnnlib_path = Path(instance_data['vnnlib_path'])
        
        # Parse VNNLIB to create specs
        # Pass input_shape to ensure specs match tensor shape (not flattened)
        # Pass true_label to promote RANGE to TOP1_ROBUST for classification
        queries = parse_vnnlib_queries(
            vnnlib_path,
            labeled_tensor=labeled_tensor,
        )
        if not queries:
            raise RuntimeError(f"parse_vnnlib_queries returned no queries for {vnnlib_path}")
        logger.info(
            f"Parsed VNNLIB specs: {len(queries)} queries, "
            f"first kind=({queries[0][0].kind}, {queries[0][1].kind})"
        )
        
        spec_pairs = list(queries)
        
        # Validate if requested
        if validate_shapes:
            validated_pairs = self._validate_and_filter_specs(
                spec_pairs,
                pytorch_model,
                labeled_tensor.tensor
            )
            
            if not validated_pairs:
                raise RuntimeError(
                    f"No spec pair of {category}/{instance_id} matches the model's I/O shapes"
                )
            
            spec_pairs = validated_pairs
        
        # Return in unified format with labeled_tensors as list
        labeled_tensors = [labeled_tensor]
        
        return (category, instance_id, pytorch_model, labeled_tensors, spec_pairs)
    
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


def create_specs_from_paths(onnx_path, vnnlib_path, category: str = "custom"):
    """Build one spec_result from an arbitrary (onnx, vnnlib) pair (ONNX->torch,
    input-shape probe, VNNLIB parse+validate). Raises SystemExit on missing or
    invalid inputs. This is the single-instance entry point used by the
    VNN-COMP harness runner (act_run_instance.py)."""
    import torch as _torch

    from act.front_end.spec_creator_base import LabeledInputTensor
    from act.front_end.vnnlib_loader.data_model_loader import _parse_vnnlib_with_shape_probe
    from act.front_end.vnnlib_loader.onnx_converter import convert_onnx_to_pytorch, get_onnx_input_shape
    from act.front_end.vnnlib_loader.vnnlib_parser import extract_label_from_vnnlib

    onnx_p, vnnlib_p = Path(onnx_path), Path(vnnlib_path)
    if not onnx_p.exists():
        raise SystemExit(f"ONNX not found: {onnx_p}")
    if not vnnlib_p.exists():
        raise SystemExit(f"VNNLIB not found: {vnnlib_p}")
    model = convert_onnx_to_pytorch(onnx_p, simplify=True)
    model.eval()
    try:
        input_shape = get_onnx_input_shape(onnx_p)
    except Exception:
        input_shape = None
    try:
        input_tensor, _meta = _parse_vnnlib_with_shape_probe(vnnlib_p, model, input_shape)
    except (UnsupportedSpecError, VNNLibParseError) as exc:
        raise SystemExit(f"Unsupported/invalid VNNLIB spec for {vnnlib_p.name}: {exc}")
    lbl = extract_label_from_vnnlib(vnnlib_p)
    label = _torch.tensor([lbl], dtype=_torch.int64) if lbl is not None else None
    instance_data = {
        "model": model,
        "labeled_tensor": LabeledInputTensor(tensor=input_tensor, label=label),
        "vnnlib_path": str(vnnlib_p),
    }
    sr = VNNLibSpecCreator()._create_specs_for_single_instance(
        category, f"{onnx_p.stem}__{vnnlib_p.stem}", instance_data, validate_shapes=True)
    return sr
