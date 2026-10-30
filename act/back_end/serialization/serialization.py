#===- act/back_end/serialization.py - ACT Net JSON Serialization -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   JSON serialization and deserialization for ACT Net and Layer structures.
#   Provides robust tensor handling, schema validation, and round-trip
#   conversion capabilities for neural network verification models.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
import json
import base64
import io
from typing import Dict, List, Any, Optional
import numpy as np
import torch

from act.back_end.core import Layer, Net
from act.front_end.spec_creator_base import LabeledInputTensor

SERIALIZATION_VERSION = "2.0"
_VALUE_TAG = "__act_type__"
_TENSOR_KEYS = frozenset({"data", "dtype", "shape"})

class ACTSerializationError(Exception):
    """Custom exception for serialization errors."""
    pass

class TensorEncoder:
    """Handles PyTorch tensor encoding/decoding to/from JSON-compatible format."""
    
    @staticmethod
    def encode_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
        """Convert PyTorch tensor to JSON-serializable dictionary."""
        # Convert to numpy and encode as base64
        np_array = tensor.detach().cpu().numpy()
        buffer = io.BytesIO()
        np.save(buffer, np_array)
        encoded_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "data": encoded_data,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad
        }
    
    @staticmethod
    def decode_tensor(tensor_dict: Dict[str, Any], target_device: Optional[str] = None) -> torch.Tensor:
        """Convert JSON dictionary back to PyTorch tensor.

        Floating-point tensors are normalized to device_manager's default
        dtype so JSONs serialized at float64 still load cleanly into a
        float32 runtime (and vice versa). Integer / bool dtypes are
        preserved as-is — coercing them to float breaks invariants in
        downstream consumers (e.g. ``y_true`` must stay ``torch.long`` for
        ``y[arange(B), y_true]`` advanced indexing in
        ``OutputSpecLayer.forward``).
        """
        encoded_data = tensor_dict["data"]
        buffer = io.BytesIO(base64.b64decode(encoded_data.encode('utf-8')))
        np_array = np.load(buffer)

        tensor = torch.from_numpy(np_array)
        if tensor.is_floating_point():
            from act.util.device_manager import get_default_dtype
            tensor = tensor.to(dtype=get_default_dtype())

        if target_device is None:
            from act.util.device_manager import get_default_device
            device = get_default_device()
        else:
            device = torch.device(target_device)
        tensor = tensor.to(device=device)

        if tensor_dict.get("requires_grad", False):
            tensor.requires_grad_(True)

        return tensor


def _encode_value(value: Any) -> Any:
    """Encode supported nested metadata while retaining the v2 tensor payload."""
    if isinstance(value, torch.Tensor):
        return TensorEncoder.encode_tensor(value)
    if isinstance(value, LabeledInputTensor):
        return {
            _VALUE_TAG: "labeled_input",
            "tensor": _encode_value(value.tensor),
            "label": _encode_value(value.label),
        }
    if isinstance(value, tuple):
        return {_VALUE_TAG: "tuple", "items": [_encode_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("ACT metadata dictionary keys must be strings")
        encoded = {key: _encode_value(item) for key, item in value.items()}
        if _VALUE_TAG in value or _TENSOR_KEYS.issubset(value):
            return {_VALUE_TAG: "dict", "items": encoded}
        return encoded
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported ACT metadata type {type(value).__name__}")


def _decode_value(value: Any, target_device: Optional[str]) -> Any:
    """Decode recursive metadata and legacy flat v2.0 tensor payloads."""
    if isinstance(value, list):
        return [_decode_value(item, target_device) for item in value]
    if isinstance(value, dict):
        if _VALUE_TAG in value:
            tag = value[_VALUE_TAG]
            if tag == "tuple" and set(value) == {_VALUE_TAG, "items"}:
                if not isinstance(value["items"], list):
                    raise ValueError("ACT tuple metadata items must be a list")
                return tuple(_decode_value(item, target_device) for item in value["items"])
            if tag == "dict" and set(value) == {_VALUE_TAG, "items"}:
                if not isinstance(value["items"], dict):
                    raise ValueError("ACT dictionary metadata items must be a dictionary")
                return {
                    key: _decode_value(item, target_device)
                    for key, item in value["items"].items()
                }
            if tag == "labeled_input" and set(value) == {_VALUE_TAG, "tensor", "label"}:
                tensor = _decode_value(value["tensor"], target_device)
                label = _decode_value(value["label"], target_device)
                if not isinstance(tensor, torch.Tensor) or (
                    label is not None and not isinstance(label, torch.Tensor)
                ):
                    raise ValueError("ACT labeled_input requires a tensor and optional tensor label")
                return LabeledInputTensor(tensor=tensor, label=label)
            raise ValueError(f"Unsupported ACT metadata tag {tag!r}")
        if _TENSOR_KEYS.issubset(value):
            return TensorEncoder.decode_tensor(value, target_device)
        return {key: _decode_value(item, target_device) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported JSON metadata value {type(value).__name__}")


class LayerSerializer:
    """Handles Layer serialization/deserialization."""
    
    @staticmethod
    def serialize_layer(layer: Layer) -> Dict[str, Any]:
        """Convert Layer to JSON-serializable dictionary."""
        return {
            "id": layer.id,
            "kind": layer.kind,
            "params": _encode_value(layer.params),
            "in_vars": layer.in_vars,
            "out_vars": layer.out_vars,
            "cache": _encode_value(layer.cache),
        }
    
    @staticmethod
    def deserialize_layer(layer_dict: Dict[str, Any], target_device: Optional[str] = None) -> Layer:
        """Convert JSON dictionary to Layer object."""
        params_decoded = _decode_value(layer_dict.get("params", {}), target_device)
        cache_decoded = _decode_value(layer_dict.get("cache", {}), target_device)
        if not isinstance(params_decoded, dict) or not isinstance(cache_decoded, dict):
            raise ValueError("ACT layer params/cache must decode to dictionaries")
        return Layer(
            id=layer_dict["id"],
            kind=layer_dict["kind"],
            params=params_decoded,
            in_vars=layer_dict["in_vars"],
            out_vars=layer_dict["out_vars"],
            cache=cache_decoded
        )


class NetSerializer:
    """Handles Net serialization/deserialization."""
    
    @staticmethod
    def serialize_net(net: Net) -> Dict[str, Any]:
        """Convert Net to JSON-serializable dictionary."""
        layers_serialized = []
        for layer in net.layers:
            layers_serialized.append(LayerSerializer.serialize_layer(layer))
        
        return {
            "format_version": SERIALIZATION_VERSION,
            "act_net": {
                "layers": layers_serialized,
                "graph": {
                    "preds": {str(k): v for k, v in net.preds.items()},
                    "succs": {str(k): v for k, v in net.succs.items()}
                }
            }
        }
    
    @staticmethod
    def deserialize_net(net_dict: Dict[str, Any], target_device: Optional[str] = None) -> Net:
        """Convert JSON dictionary to Net object."""
        format_version = net_dict.get("format_version", "unknown")
        if format_version != SERIALIZATION_VERSION:
            raise ValueError(f"Unsupported format version: {format_version}. Expected {SERIALIZATION_VERSION}")
        
        act_net = net_dict["act_net"]
        
        # Deserialize layers
        layers = []
        for layer_dict in act_net["layers"]:
            layer = LayerSerializer.deserialize_layer(layer_dict, target_device)
            layers.append(layer)
        
        # Reconstruct graph structure
        graph = act_net.get("graph", {})
        preds = {int(k): v for k, v in graph.get("preds", {}).items()}
        succs = {int(k): v for k, v in graph.get("succs", {}).items()}
        
        # Create Net object with validation; malformed serialized graphs are errors.
        return Net(layers=layers, preds=preds, succs=succs)

class ACTJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for ACT objects."""
    
    def default(self, o):
        if isinstance(o, (Layer, Net)):
            raise ACTSerializationError("Use NetSerializer.serialize_net() instead of json.dumps() directly")
        return super().default(o)

# High-level API functions
def save_net_to_file(net: Net, filepath: str, indent: int = 2) -> None:
    """Save ACT Net to JSON file."""
    net_dict = NetSerializer.serialize_net(net)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(net_dict, f, indent=indent, ensure_ascii=False, cls=ACTJSONEncoder)

def load_net_from_file(filepath: str, target_device: Optional[str] = None) -> Net:
    """Load ACT Net from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        net_dict = json.load(f)
    return NetSerializer.deserialize_net(net_dict, target_device)

def save_net_to_string(net: Net, indent: int = 2) -> str:
    """Serialize ACT Net to JSON string."""
    net_dict = NetSerializer.serialize_net(net)
    return json.dumps(net_dict, indent=indent, ensure_ascii=False, cls=ACTJSONEncoder)

def load_net_from_string(json_str: str, target_device: Optional[str] = None) -> Net:
    """Deserialize ACT Net from JSON string."""
    net_dict = json.loads(json_str)
    return NetSerializer.deserialize_net(net_dict, target_device)

# Validation utilities
def validate_json_schema(net_dict: Dict[str, Any]) -> List[str]:
    """Validate JSON schema structure before deserialization."""
    errors = []
    
    # Check required top-level fields
    if "format_version" not in net_dict:
        errors.append("Missing 'format_version' field")
    
    if "act_net" not in net_dict:
        errors.append("Missing 'act_net' field")
        return errors
    
    act_net = net_dict["act_net"]
    
    # Check required act_net fields
    if "layers" not in act_net:
        errors.append("Missing 'layers' field in act_net")
    elif not isinstance(act_net["layers"], list):
        errors.append("'layers' must be a list")
    
    # Validate each layer structure
    for i, layer in enumerate(act_net.get("layers", [])):
        if not isinstance(layer, dict):
            errors.append(f"Layer {i} must be a dictionary")
            continue
            
        required_fields = ["id", "kind", "params", "in_vars", "out_vars"]
        for field in required_fields:
            if field not in layer:
                errors.append(f"Layer {i} missing required field '{field}'")
    
    return errors
