#===- act/front_end/vnnlib/data_model_loader.py - VNNLIB Loader -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Download, list, and load VNNLIB benchmarks from VNN-COMP repository.
#   Mirrors torchvision/data_model_loader.py structure for unified interface.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
import ast
import logging
import json
import csv
import urllib.request
import shutil
import gzip
import torch
import torch.nn as nn

from act.util.path_config import get_vnnlib_data_root
from act.front_end.vnnlib_loader.onnx_converter import (
    convert_onnx_to_pytorch,
    get_onnx_input_shape,
    get_onnx_output_shape,
    ONNXConversionError
)
from act.front_end.vnnlib_loader.vnnlib_parser import (
    parse_vnnlib_to_tensors,
    extract_label_from_vnnlib,
    VNNLibParseError
)
from act.front_end.spec_creator_base import LabeledInputTensor

logger = logging.getLogger(__name__)


# VNN-COMP GitHub repository base URLs (try multiple sources)
VNNCOMP_REPO_URLS = [
    "https://raw.githubusercontent.com/VNN-COMP/vnncomp2026_benchmarks/main/benchmarks"
]
DEFAULT_BENCHMARK_VERSION = "2.0"


def _normalize_rel_path(rel_path: str) -> str:
    rel_path = rel_path.strip()
    if rel_path.startswith("./"):
        rel_path = rel_path[2:]
    return rel_path.lstrip("/")


def _parse_onnx_field(onnx_field: str) -> List[Tuple[Optional[str], str]]:
    """Parse standard ONNX paths or dual-model list-of-tuples fields."""
    onnx_field = onnx_field.strip()
    if onnx_field.startswith("["):
        raw_entries = ast.literal_eval(onnx_field)
        entries: List[Tuple[Optional[str], str]] = [
            (str(role) if role is not None else None, str(model_path))
            for role, model_path in raw_entries
        ]
        return [(role, _normalize_rel_path(model_path)) for role, model_path in entries]
    return [(None, _normalize_rel_path(onnx_field))]


class _CombinedDualModel(nn.Module):
    """Run two networks on a SHARED input and concatenate outputs as [f(x); g(x)].

    Used for isomorphic-equivalence specs; the concat order matches the parser's
    Y_f -> Y_<j>, Y_g -> Y_<numel(Y_f)+j> indexing.
    """

    def __init__(self, model_f: nn.Module, model_g: nn.Module):
        super().__init__()
        self.model_f = model_f
        self.model_g = model_g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        yf = self.model_f(x)
        yg = self.model_g(x)
        return torch.cat([yf.flatten(1), yg.flatten(1)], dim=1)


def download_vnnlib_category(
    category: str,
    root_dir: Optional[str] = None,
    force_redownload: bool = False
) -> Dict[str, Any]:
    """
    Download a VNNLIB benchmark category from VNN-COMP repository.
    
    Downloads instances.csv, ONNX models, and VNNLIB specifications
    for the specified category.
    
    Args:
        category: Benchmark category name (e.g., 'mnist_fc', 'cifar10_resnet')
        root_dir: Root directory for VNNLIB data (default: from path_config)
        force_redownload: If True, redownload even if exists
        
    Returns:
        Dict with download status and paths:
        - status: 'success' or 'error'
        - message: Status message
        - category_path: Path to category directory
        - num_instances: Number of instances downloaded
        
    Example:
        >>> result = download_vnnlib_category("mnist_fc")
        >>> print(result['num_instances'])
        100
    """
    if root_dir is None:
        root_dir = get_vnnlib_data_root()
    
    # Import here to avoid circular dependency
    from act.front_end.vnnlib_loader.category_mapping import CATEGORY_MAPPING
    
    info: Dict[str, Any] = CATEGORY_MAPPING.get(category, {})
    repo_name = str(info.get('repo_name', category))
    version = str(info.get('version', DEFAULT_BENCHMARK_VERSION))
    repo_prefix = f"{repo_name}/{version}"
    
    category_dir = Path(root_dir) / category
    onnx_dir = category_dir / "onnx"
    vnnlib_dir = category_dir / "vnnlib"
    instances_file = category_dir / "instances.csv"
    
    # Check if already exists
    if category_dir.exists() and instances_file.exists() and not force_redownload:
        logger.info(f"Category '{category}' already downloaded at {category_dir}")
        
        # Count existing instances
        num_instances = 0
        if instances_file.exists():
            with open(instances_file, 'r') as f:
                rows = list(csv.reader(f))
                if rows and len(rows[0]) >= 2 and rows[0][0].strip().lower() == 'onnx':
                    rows = rows[1:]
                num_instances = sum(1 for row in rows if len(row) >= 3)
        
        return {
            'status': 'success',
            'message': f"Category '{category}' already exists",
            'category_path': str(category_dir),
            'num_instances': num_instances
        }
    
    # Create directories
    onnx_dir.mkdir(parents=True, exist_ok=True)
    vnnlib_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading VNNLIB category: {category} (repo: {repo_name}, version: {version})")
    
    # Try multiple repository URLs
    successful_base_url = None
    for base_url in VNNCOMP_REPO_URLS:
        instances_url = f"{base_url}/{repo_prefix}/instances.csv"
        logger.info(f"Trying: {instances_url}")
        
        try:
            urllib.request.urlretrieve(instances_url, instances_file)
            successful_base_url = base_url
            logger.info(f"✓ Successfully downloaded from {base_url}")
            break
        except Exception as e:
            logger.debug(f"Failed to download from {base_url}: {e}")
            continue
    
    if successful_base_url is None:
        return {
            'status': 'error',
            'message': f"Failed to download instances.csv from all sources. Tried {len(VNNCOMP_REPO_URLS)} repositories."
        }
    
    try:
        # Parse instances.csv to get ONNX and VNNLIB files
        instances: List[Dict[str, Any]] = []
        onnx_files: set[str] = set()
        vnnlib_files: set[str] = set()
        with open(instances_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if rows and len(rows[0]) >= 2 and rows[0][0].strip().lower() == 'onnx':
                rows = rows[1:]
            
            for row in rows:
                if len(row) >= 3:
                    onnx_field, vnnlib_field, timeout = row[0], row[1], row[2]
                    onnx_entries = _parse_onnx_field(onnx_field)
                    vnnlib_file = _normalize_rel_path(vnnlib_field)
                    for _, onnx_file in onnx_entries:
                        onnx_files.add(onnx_file)
                    vnnlib_files.add(vnnlib_file)
                    instances.append({
                        'onnx': onnx_entries,
                        'vnnlib': vnnlib_file,
                        'timeout': timeout
                    })
        
        logger.info(f"Found {len(instances)} instances in category '{category}'")
        
        # Download ONNX and VNNLIB files using the successful base URL
        downloaded_onnx: set[str] = set()
        downloaded_vnnlib: set[str] = set()
        all_files: List[Tuple[str, set[str]]] = [(rel_path, downloaded_onnx) for rel_path in sorted(onnx_files)]
        all_files.extend((rel_path, downloaded_vnnlib) for rel_path in sorted(vnnlib_files))
        
        for idx, (rel_path, downloaded_set) in enumerate(all_files, 1):
            target_path = category_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Try compressed first, then uncompressed.
            tried_urls = []
            success = False
            
            for extension in ['.gz', '']:
                file_url = f"{successful_base_url}/{repo_prefix}/{rel_path}{extension}"
                tried_urls.append(file_url)
                
                try:
                    logger.debug(f"[{idx}/{len(all_files)}] Trying {rel_path}{extension}")
                    
                    if extension == '.gz':
                        # Download compressed file
                        gz_path = target_path.parent / f"{target_path.name}.gz"
                        urllib.request.urlretrieve(file_url, gz_path)
                        
                        # Decompress
                        with gzip.open(gz_path, 'rb') as f_in:
                            with open(target_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        
                        # Remove .gz file
                        gz_path.unlink()
                        logger.debug(f"  ✓ Decompressed {rel_path}")
                    else:
                        # Download uncompressed file
                        urllib.request.urlretrieve(file_url, target_path)
                    
                    downloaded_set.add(rel_path)
                    success = True
                    break
                except Exception as e:
                    logger.debug(f"  Failed with {extension}: {e}")
                    continue
            
            if not success:
                logger.warning(f"Failed to download {rel_path} from any URL")
        
        # Create metadata
        metadata = {
            'category': category,
            'repo_name': repo_name,
            'version': version,
            'num_instances': len(instances),
            'num_onnx_models': len(downloaded_onnx),
            'num_vnnlib_specs': len(downloaded_vnnlib),
            'source': successful_base_url,
            'paths': {
                'onnx': str(onnx_dir),
                'vnnlib': str(vnnlib_dir),
                'instances': str(instances_file)
            }
        }
        
        # Save metadata
        metadata_path = category_dir / "info.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(
            f"Successfully downloaded category '{category}': "
            f"{len(instances)} instances, {len(downloaded_onnx)} ONNX models, "
            f"{len(downloaded_vnnlib)} VNNLIB specs"
        )
        
        return {
            'status': 'success',
            'message': f"Downloaded {len(instances)} instances",
            'category_path': str(category_dir),
            'num_instances': len(instances)
        }
        
    except Exception as e:
        logger.error(f"Failed to download category '{category}': {e}")
        return {
            'status': 'error',
            'message': str(e)
        }


def list_downloaded_pairs(root_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all downloaded VNNLIB benchmark instances.
    
    Mirrors torchvision/data_model_loader.list_downloaded_pairs() interface.
    Each instance is treated as a (onnx_model, vnnlib_spec) pair.
    
    Args:
        root_dir: Root directory for VNNLIB data (default: from path_config)
        
    Returns:
        List of dicts with instance information:
        - category: Benchmark category
        - onnx_model: ONNX model filename
        - vnnlib_spec: VNNLIB spec filename
        - timeout: Verification timeout
        - paths: Dict with onnx_path and vnnlib_path
        
    Example:
        >>> pairs = list_downloaded_pairs()
        >>> for pair in pairs:
        ...     print(f"{pair['category']}: {pair['onnx_model']}")
    """
    if root_dir is None:
        root_dir = get_vnnlib_data_root()
    
    root_path = Path(root_dir)
    if not root_path.exists():
        return []
    
    all_instances = []
    
    for category_dir in root_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        instances_file = category_dir / "instances.csv"
        if not instances_file.exists():
            continue
        
        try:
            with open(instances_file, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if rows and len(rows[0]) >= 2 and rows[0][0].strip().lower() == 'onnx':
                    rows = rows[1:]
                
                for row in rows:
                    if len(row) >= 3:
                        onnx_field, vnnlib_file, timeout = row[0], row[1], row[2]
                        onnx_entries = _parse_onnx_field(onnx_field)
                        vnnlib_rel = _normalize_rel_path(vnnlib_file)
                        
                        onnx_paths = [category_dir / onnx_rel for _, onnx_rel in onnx_entries]
                        vnnlib_path = category_dir / vnnlib_rel
                        
                        # Only include if all referenced files exist
                        if all(onnx_path.exists() for onnx_path in onnx_paths) and vnnlib_path.exists():
                            first_onnx_rel = onnx_entries[0][1]
                            paths: Dict[str, Any] = {
                                'onnx': str(onnx_paths[0]),
                                'vnnlib': str(vnnlib_path)
                            }
                            if len(onnx_entries) > 1:
                                paths['onnx_models'] = [str(onnx_path) for onnx_path in onnx_paths]
                            
                            all_instances.append({
                                'category': category_dir.name,
                                'onnx_model': first_onnx_rel,
                                'onnx_models': onnx_entries,
                                'vnnlib_spec': vnnlib_rel,
                                'timeout': float(timeout) if timeout else None,
                                'is_dual_model': len(onnx_entries) > 1,
                                'paths': paths
                            })
        
        except Exception as e:
            logger.warning(f"Failed to read instances from {category_dir.name}: {e}")
    
    return all_instances


def _parse_vnnlib_with_shape_probe(vnnlib_path, pytorch_model, input_shape):
    """Parse VNNLib, reshape input tensor to a shape the model actually accepts.

    Fast path: parse with the ONNX-declared ``input_shape``. If that raises
    ``VNNLibParseError`` (shape-count mismatch), fall back to flat parsing and
    probe the model for a working shape via ``_probe_model_shape``.
    Raises ``RuntimeError`` if no working shape is found.
    """
    try:
        tensor, meta = parse_vnnlib_to_tensors(vnnlib_path, input_shape=input_shape)
        return tensor, meta
    except VNNLibParseError as e:
        logger.warning(f"Shape-matched parse failed ({e}); retrying with shape probe...")
        tensor, meta = parse_vnnlib_to_tensors(vnnlib_path, input_shape=None)
        probed = _probe_model_shape(pytorch_model, tensor.numel(), input_shape)
        if probed is None:
            raise RuntimeError(
                f"VNNLIB parsing failed: could not find a working shape for "
                f"{tensor.numel()}-element input in ONNX model (declared shape {input_shape})"
            )
        tensor = tensor.reshape(*probed)
        meta['input_shape_probed'] = probed
        logger.info(f"  ✓ Shape probe succeeded: reshaped to {probed}")
        return tensor, meta


def _probe_model_shape(pytorch_model, total_count: int, onnx_shape):
    """Find a tensor shape with numel==total_count that the model accepts.

    Returns the first shape whose dry-run forward pass succeeds, or None.
    Candidates (in order): onnx_shape if numel matches; perfect-square (1,1,s,s);
    ratio-based (k,*onnx_shape[1:]) when total_count is a multiple of onnx_numel;
    flat (1,total_count).
    """
    import math

    candidates = []
    if onnx_shape:
        onnx_numel = 1
        for d in onnx_shape:
            onnx_numel *= max(int(d), 1)
        if onnx_numel == total_count:
            candidates.append(tuple(onnx_shape))
        elif onnx_numel > 0 and total_count % onnx_numel == 0 and len(onnx_shape) > 1:
            k = total_count // onnx_numel
            candidates.append((k,) + tuple(onnx_shape[1:]))
    s = int(math.isqrt(total_count))
    if s * s == total_count:
        candidates.append((1, 1, s, s))
    candidates.append((1, total_count))

    for shape in candidates:
        try:
            x = torch.zeros(*shape)
            with torch.no_grad():
                pytorch_model(x)
            return shape
        except Exception as e:
            # Intentional: probing candidate input shapes; mismatches are expected, try the next.
            logger.debug("suppressed: %s", e)
            continue
    return None


def load_vnnlib_pair(
    category: str,
    onnx_model: str,
    vnnlib_spec: str,
    root_dir: Optional[str] = None,
    auto_download: bool = True,
    onnx_model_g: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load a VNNLIB benchmark instance (ONNX model + VNNLIB spec).
    
    Mirrors torchvision/data_model_loader.load_dataset_model_pair() interface.
    Returns PyTorch model and LabeledInputTensor from VNNLIB constraints.
    
    Args:
        category: Benchmark category
        onnx_model: ONNX model filename
        vnnlib_spec: VNNLIB spec filename
        root_dir: Root directory for VNNLIB data (default: from path_config)
        auto_download: If True, download category if not found locally
        onnx_model_g: Optional second ONNX model for dual-model instances
        
    Returns:
        Dict containing:
        - model: PyTorch nn.Module (converted from ONNX)
        - labeled_tensor: LabeledInputTensor with input tensor and ground truth label
        - vnnlib_metadata: Dict with constraint information
        - onnx_path: Path to ONNX file
        - vnnlib_path: Path to VNNLIB file
        - category: Benchmark category
        
    Example:
        >>> result = load_vnnlib_pair("mnist_fc", "model.onnx", "spec.vnnlib")
        >>> model = result['model']
        >>> labeled_tensor = result['labeled_tensor']
        >>> tensor, label = labeled_tensor
        >>> output = model(tensor.unsqueeze(0))
    """
    if root_dir is None:
        root_dir = get_vnnlib_data_root()
    
    category_dir = Path(root_dir) / category
    onnx_rel = _normalize_rel_path(onnx_model)
    vnnlib_rel = _normalize_rel_path(vnnlib_spec)
    onnx_path = category_dir / onnx_rel
    vnnlib_path = category_dir / vnnlib_rel
    onnx_path_g = category_dir / _normalize_rel_path(onnx_model_g) if onnx_model_g else None
    
    # Auto-download if not found
    if (
        not category_dir.exists()
        or not onnx_path.exists()
        or not vnnlib_path.exists()
        or (onnx_path_g is not None and not onnx_path_g.exists())
    ):
        if auto_download:
            logger.info(f"Category '{category}' not found locally. Downloading...")
            
            download_result = download_vnnlib_category(category, root_dir)
            
            if download_result['status'] != 'success':
                raise RuntimeError(
                    f"Failed to download category '{category}': "
                    f"{download_result['message']}"
                )
            
            logger.info("Download completed. Proceeding to load...")
        else:
            raise FileNotFoundError(
                f"VNNLIB instance not found: {category}/{onnx_model}\n"
                f"Set auto_download=True to download automatically."
            )
    
    # Check if files exist after potential download
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if onnx_path_g is not None and not onnx_path_g.exists():
        raise FileNotFoundError(f"Second ONNX model not found: {onnx_path_g}")
    if not vnnlib_path.exists():
        raise FileNotFoundError(f"VNNLIB spec not found: {vnnlib_path}")
    
    logger.info(f"Loading VNNLIB instance: {category}/{onnx_model}")
    
    # Convert ONNX to PyTorch
    logger.info("[1/3] Converting ONNX model to PyTorch...")
    try:
        pytorch_model = convert_onnx_to_pytorch(onnx_path, simplify=True)
        pytorch_model.eval()
        logger.info(f"  ✓ Model converted successfully")
    except ONNXConversionError as e:
        raise RuntimeError(f"ONNX conversion failed: {e}")
    
    pytorch_model_g = None
    if onnx_path_g is not None:
        try:
            pytorch_model_g = convert_onnx_to_pytorch(onnx_path_g, simplify=True)
            pytorch_model_g.eval()
            logger.info(f"  ✓ Second model converted successfully")
        except ONNXConversionError as e:
            raise RuntimeError(f"Second ONNX conversion failed: {e}")
    
    # Get input shape from ONNX
    logger.info("[2/3] Extracting input shape...")
    try:
        input_shape = get_onnx_input_shape(onnx_path)
        logger.info(f"  ✓ Input shape: {input_shape}")
    except ONNXConversionError as e:
        logger.warning(f"Failed to extract input shape: {e}")
        input_shape = None
    
    # Parse VNNLIB to get input tensor
    logger.info("[3/3] Parsing VNNLIB specification...")
    input_tensor, vnnlib_metadata = _parse_vnnlib_with_shape_probe(
        vnnlib_path, pytorch_model, input_shape
    )
    logger.info(
        f"  ✓ Parsed VNNLIB: {vnnlib_metadata['num_inputs']} inputs, "
        f"{vnnlib_metadata['num_outputs']} outputs"
    )
    
    # Extract ground truth label from VNNLIB comment (if available)
    ground_truth_label_int = extract_label_from_vnnlib(vnnlib_path)
    if ground_truth_label_int is not None:
        logger.info(f"  ✓ Ground truth label: {ground_truth_label_int}")
        ground_truth_label = torch.tensor([ground_truth_label_int], dtype=torch.int64)
    else:
        ground_truth_label = None
    
    # Create LabeledInputTensor pairing input with label (tensor)
    labeled_tensor = LabeledInputTensor(tensor=input_tensor, label=ground_truth_label)
    
    logger.info(f"Successfully loaded VNNLIB instance from '{category}'")
    
    result = {
        'model': pytorch_model,
        'labeled_tensor': labeled_tensor,
        'vnnlib_metadata': vnnlib_metadata,
        'onnx_path': str(onnx_path),
        'vnnlib_path': str(vnnlib_path),
        'category': category,
        'onnx_model': onnx_model,
        'vnnlib_spec': vnnlib_spec
    }
    if pytorch_model_g is not None and onnx_path_g is not None:
        result['model'] = _CombinedDualModel(pytorch_model, pytorch_model_g)
        result['model_f'] = pytorch_model
        result['model_g'] = pytorch_model_g
        result['onnx_path_g'] = str(onnx_path_g)
    return result


def model_inference_with_vnnlib(
    category: str,
    onnx_model: str,
    vnnlib_spec: str,
    root_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test inference for a single VNNLIB benchmark instance.
    
    Mirrors torchvision/data_model_loader.model_inference_with_dataset() interface
    for consistent CLI usage.
    
    Args:
        category: Benchmark category name
        onnx_model: ONNX model filename
        vnnlib_spec: VNNLIB spec filename
        root_dir: Root directory for VNNLIB data (default: from path_config)
        verbose: Whether to print detailed progress information
        
    Returns:
        Dictionary with test results:
        - status: 'success', 'failed', or 'error'
        - output_shape: Tuple of output shape (if successful)
        - error: Error message (if failed/error)
        - category: Benchmark category
        - model: Model name (stem of ONNX filename)
        - spec: VNNLIB spec filename
        
    Example:
        >>> result = model_inference_with_vnnlib("acasxu_2023", "model.onnx", "spec.vnnlib")
        >>> print(result['status'])
        'success'
    """
    from pathlib import Path
    from act.util.model_inference import infer_single_model
    
    model_name = Path(onnx_model).stem
    combo_id = f"{category}+{model_name}"
    
    try:
        # Load the pair
        if verbose:
            logger.info(f"Testing: {category} + {model_name}")
        
        result = load_vnnlib_pair(
            category=category,
            onnx_model=onnx_model,
            vnnlib_spec=vnnlib_spec,
            root_dir=root_dir,
            auto_download=True
        )
        
        model = result['model']
        labeled_tensor = result['labeled_tensor']
        input_tensor = labeled_tensor.tensor  # Already has batch dimension
        
        # Run inference
        success, output, error_msg = infer_single_model(
            combo_id, 
            model, 
            input_tensor  # Already (1, C, H, W)
        )
        
        if success:
            if output is None:
                raise RuntimeError("Inference succeeded without an output tensor")
            if verbose:
                logger.info(f"  ✓ SUCCESS - Output shape: {output.shape}")
            return {
                'category': category,
                'model': model_name,
                'spec': vnnlib_spec,
                'status': 'success',
                'output_shape': tuple(output.shape)
            }
        else:
            if verbose:
                logger.warning(f"  ✗ FAILED - {error_msg}")
            return {
                'category': category,
                'model': model_name,
                'spec': vnnlib_spec,
                'status': 'failed',
                'error': error_msg
            }
            
    except Exception as e:
        error_msg = str(e)[:100]
        if verbose:
            logger.error(f"  ✗ ERROR - {error_msg}")
        return {
            'category': category,
            'model': model_name,
            'spec': vnnlib_spec,
            'status': 'error',
            'error': error_msg
        }


def list_available_categories() -> List[str]:
    """
    List available benchmark categories from VNN-COMP repository.
    
    Note: This is a static list of common categories. For the complete
    and up-to-date list, visit:
    https://github.com/ChristopherBrix/vnncomp_benchmarks
    
    Returns:
        List of category names
    """
    # Common VNN-COMP benchmark categories (static list)
    common_categories = [
        'mnist_fc',
        'mnist_conv',
        'cifar10_resnet',
        'cifar10_cnn',
        'acasxu',
        'collins_yolo',
        'nn4sys',
        'oval',
        'reach_prob',
        'sri_resnet',
        'tllverifybench'
    ]
    
    return common_categories


def list_local_categories(root_dir: Optional[str] = None) -> List[str]:
    """
    List locally downloaded benchmark categories.
    
    Args:
        root_dir: Root directory for VNNLIB data (default: from path_config)
        
    Returns:
        List of category names that have been downloaded
    """
    if root_dir is None:
        root_dir = get_vnnlib_data_root()
    
    root_path = Path(root_dir)
    if not root_path.exists():
        return []
    
    categories = []
    for item in root_path.iterdir():
        if item.is_dir() and (item / "instances.csv").exists():
            categories.append(item.name)
    
    return sorted(categories)


def get_category_info(category: str, root_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a downloaded benchmark category.
    
    Args:
        category: Category name
        root_dir: Root directory for VNNLIB data (default: from path_config)
        
    Returns:
        Dict with category metadata, or None if not found
    """
    if root_dir is None:
        root_dir = get_vnnlib_data_root()
    
    category_dir = Path(root_dir) / category
    info_path = category_dir / "info.json"
    
    if not info_path.exists():
        return None
    
    try:
        with open(info_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read category info: {e}")
        return None


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    size_value = float(size_bytes)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_value < 1024:
            return f"{size_value:.2f} {unit}"
        size_value /= 1024
    return f"{size_value:.2f} TB"


def _get_directory_size(path: Path) -> int:
    """Calculate total size of directory in bytes."""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception as e:
        logger.warning(f"Failed to calculate directory size: {e}")
    return total
