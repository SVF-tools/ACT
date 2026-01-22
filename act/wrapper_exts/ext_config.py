#!/usr/bin/env python3
"""Merged input-parser config utilities (types, vnnlib parser, model, dataset, specs)

This file consolidates configuration utilities for external verifier usage.
"""
from typing import Union, Optional, List, Tuple, Dict, Any
import os
import numpy as np
import torch
import re
import json
import onnx
import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
from onnx2torch import convert
import subprocess

from enum import Enum

# Import verification types from canonical location (act/util/stats.py)
from act.util.stats import VerifyStatus, VerifyResult


class SplitType(Enum):
    INPUT = "input"
    INPUT_GRAD = "input_grad"
    INPUT_SB = "input_sb"
    RELU_GRAD = "relu_grad"
    RELU_SR = "relu_babsr"
    RELU_CE = "relu_ce"


class SpecType(Enum):
    LOCAL_LP = "local_lp"
    LOCAL_VNNLIB = "local_vnnlib"
    SET_VNNLIB = "set_vnnlib"
    SET_BOX = "set_box"


class LPNormType(Enum):
    LINF = "inf"
    L2 = "2"
    L1 = "1"


class VNNLIBParser:

    def _var_extraction(var_string):

        match_indexed = re.match(r"^([A-Za-z_]+)_([0-9]+)$", var_string)
        if match_indexed:
            var_group = match_indexed.group(1)
            var_index = int(match_indexed.group(2))
            return var_group, var_index

        match_plain = re.match(r"^([A-Za-z_]+)$", var_string)
        if match_plain:
            var_group = match_plain.group(1)
            return var_group, -1

    def _num_extraction(var_string):
        match = re.match(r"^([+-]?)(\d+(\.\d*)?|\.\d+)$", var_string.strip())
        if match is None:
            return None
        sign = 1 if match.group(1) == "+" else -1
        return sign * float(match.group(2))

    @staticmethod
    def parse_term(input_clause):
        terms = input_clause.split()
        parsed_results = []
        sign_flag = None
        for term in terms:
            if term in ('+', '-'):
                sign_flag = 1 if term == '+' else -1
            else:
                num = VNNLIBParser._num_extraction(term)
                if num is None:
                    var_group, var_index = VNNLIBParser._var_extraction(term)

                    value = float(sign_flag) if sign_flag is not None else 1.0
                else:
                    var_group = "const"
                    var_index = -1

                    value = sign_flag * float(num) if sign_flag is not None else float(num)
                parsed_results.append((var_group, var_index, value))
                sign_flag = None

        return parsed_results

    @staticmethod
    def identify_declare_const(lines):
        input_vars, output_vars, anchors, utility = [], [], [], []
        for line in lines:
            if line.startswith("(declare-const"):
                match_indexed = re.match(r"\(declare-const ([A-Za-z_]+)_([0-9]+) [A-Za-z]+\)", line)
                if match_indexed:
                    var_group = match_indexed.group(1)
                    var_index = int(match_indexed.group(2))
                    if var_group == "X":
                        input_vars.append(("X", var_index))
                    elif var_group == "Y":
                        output_vars.append(("Y", var_index))
                    elif var_group == "X_hat":
                        anchors.append(("X_hat", var_index))
                    else:
                        utility.append((var_group, var_index))

                else:

                    match_plain = re.match(r"\(declare-const ([A-Za-z_]+) [A-Za-z]+\)", line)
                    if match_plain:
                        var_group = match_plain.group(1)
                        utility.append((var_group, -1))
        return input_vars, output_vars, anchors, utility

    @staticmethod
    def is_local(lines):
        return any("X_hat" in l or "eps" in l for l in lines)


class Model:
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        self.pytorch_model = None
        self._onnx_input_shape = None
        self._tf_input_shape = None

        if model_path is not None:
            self._auto_load_and_convert()

    def _auto_load_and_convert(self):
        if os.path.isfile(self.model_path):
            if self.model_path.endswith('.pt') or self.model_path.endswith('.pth'):
                self.load_pt_model()
            elif self.model_path.endswith('.onnx'):
                self._onnx_input_shape = self._get_onnx_input_shape(self.model_path)
                self.convert_onnx_to_pytorch()
            elif self.model_path.endswith('.h5') or self.model_path.endswith('.keras'):
                self._tf_input_shape = self._get_tf_input_shape(self.model_path)
                self.convert_tf_to_pytorch()
            else:
                raise ValueError(f"Unknown model file type: {self.model_path}")
        elif os.path.isdir(self.model_path):
            if os.path.exists(os.path.join(self.model_path, 'saved_model.pb')):
                self.convert_tf_to_pytorch()
            else:
                raise ValueError(f"Unknown model directory: {self.model_path}")
        else:
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

    def _get_onnx_input_shape(self, onnx_path):
        onnx_model = onnx.load(onnx_path)
        input_tensor = onnx_model.graph.input[0]
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:

            shape.append(dim.dim_value if dim.dim_value > 0 else 1)
        return tuple(shape)

    def _get_tf_input_shape(self, tf_path):
        try:
            if tf_path.endswith('.h5') or tf_path.endswith('.keras'):
                model = tf.keras.models.load_model(tf_path)
                return tuple(model.input_shape)
            elif os.path.isdir(tf_path) and os.path.exists(os.path.join(tf_path, 'saved_model.pb')):
                model = tf.keras.models.load_model(tf_path)
                return tuple(model.input_shape)
        except Exception as e:
            print(f"[WARN] Failed to get TF input shape: {e}")
        return None

    def get_expected_input_shape(self):

        if self._onnx_input_shape is not None:
            return self._onnx_input_shape
        if self._tf_input_shape is not None:
            return self._tf_input_shape
        if hasattr(self.pytorch_model, 'input_shape'):
            return self.pytorch_model.input_shape
        for shape in [(1, 1, 28, 28), (1, 3, 32, 32), (1, 5)]:
            try:
                dummy = torch.zeros(*shape)
                self.pytorch_model(dummy)
                return shape
            except Exception:
                continue
        raise RuntimeError("Cannot infer model input shape automatically.")

    def load_pt_model(self):
        try:
            self.pytorch_model = torch.load(self.model_path, map_location=self.device)
            self.pytorch_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {e}")

    def convert_onnx_to_pytorch(self):
        try:
            onnx_model = onnx.load(self.model_path)
            self.pytorch_model = convert(onnx_model)
            self.pytorch_model.eval()
            
            # Apply device and dtype settings
            try:
                from act.util.device_manager import get_default_device, get_default_dtype
                target_device = get_default_device()
                target_dtype = get_default_dtype()
                self.pytorch_model = self.pytorch_model.to(dtype=target_dtype, device=target_device)
            except Exception as dtype_e:
                # Fallback: convert to float64 by default for ACT consistency
                self.pytorch_model = self.pytorch_model.double()
                print(f"[WARN] Could not apply device_manager settings: {dtype_e}, using float64")
        except Exception as e:
            raise RuntimeError(f"Failed to convert ONNX model to PyTorch: {e}")

    def convert_tf_to_pytorch(self, onnx_path="tmp_model.onnx"):
        model_type = None
        if os.path.isfile(self.model_path):
            if self.model_path.endswith('.h5') or self.model_path.endswith('.keras'):
                model_type = 'keras'
        elif os.path.isdir(self.model_path):
            if os.path.exists(os.path.join(self.model_path, 'saved_model.pb')):
                model_type = 'saved_model'
        else:
            raise ValueError("Unknown model path type.")

        if model_type == 'keras':
            cmd = [
                "python", "-m", "tf2onnx.convert",
                "--keras", self.model_path,
                "--output", onnx_path
            ]
        elif model_type == 'saved_model':
            cmd = [
                "python", "-m", "tf2onnx.convert",
                "--saved-model", self.model_path,
                "--output", onnx_path
            ]
        else:
            raise ValueError("Model type not supported for conversion.")

        print(f"[INFO] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        try:
            onnx_model = onnx.load(onnx_path)
            self.pytorch_model = convert(onnx_model)
            self.pytorch_model.eval()
            
            # Apply device and dtype settings
            try:
                from act.util.device_manager import get_default_device, get_default_dtype
                target_device = get_default_device()
                target_dtype = get_default_dtype()
                self.pytorch_model = self.pytorch_model.to(dtype=target_dtype, device=target_device)
            except Exception as dtype_e:
                # Fallback: convert to float64 by default for ACT consistency
                self.pytorch_model = self.pytorch_model.double()
                print(f"[WARN] Could not apply device_manager settings: {dtype_e}, using float64")
        except Exception as e:
            raise RuntimeError(f"Failed to convert TF/Keras model to PyTorch: {e}")


class Dataset:
    def __init__(self,

                 dataset_path: str = None,
                 anchor_csv_path: str = None,
                 device: str = 'cpu',
                 spec_type: SpecType = None,
                 start: int = None,
                 end: int = None,
                 num_outputs: int = 10,
                 mean: Optional[Union[float, List[float]]] = None,
                 std: Optional[Union[float, List[float]]] = None,
                 preprocess: bool = False):

        self.dataset_path = dataset_path
        self.anchor_csv_path = anchor_csv_path
        self.device = device
        self.spec_type = SpecType(spec_type)
        self.start = start
        self.end = end
        self.num_outputs = num_outputs
        self.mean = mean
        self.std = std
        if isinstance(mean, list) and len(mean) == 1:
            self.mean = mean[0]
        if isinstance(std, list) and len(std) == 1:
            self.std = std[0]

        self.preprocess = preprocess

        self.input_center: Optional[torch.Tensor] = None

        self.labels: Optional[torch.Tensor] = None

        self.input_lb: Optional[torch.Tensor] = None
        self.input_ub: Optional[torch.Tensor] = None

        self.output_constraints: Optional[torch.Tensor] = None

        print("Init dataset")

        if dataset_path in ['mnist', 'cifar', 'cifar10']:
            self._download_and_load_builtin(dataset_path)
        elif dataset_path is not None:
            if dataset_path.endswith('.csv'):
                self.load_from_csv()
            elif dataset_path.endswith('.vnnlib'):
                print("Entered vnnlib")
                if self.spec_type == SpecType.SET_VNNLIB:
                    self.load_from_vnnlib(dataset_path)
                elif self.spec_type == SpecType.LOCAL_VNNLIB:
                    print("Entered local vnnlib")
                    self.load_from_vnnlib(dataset_path, anchor_csv_path=self.anchor_csv_path, x_hat=self.input_center)
            elif dataset_path.endswith('.json'):
                self.load_set_box_from_json()
            else:
                raise ValueError(f"Unknown dataset file type: {dataset_path}")

        self._apply_preprocessing()

    def _download_and_load_builtin(self, name):
        self._data_source = name
        data_root = os.environ.get('ACT_DATA_ROOT', None)
        if data_root is None:
            # try util path
            try:
                from act.util.path_config import get_data_root
                data_root = get_data_root()
            except Exception:
                data_root = '.'
        
        print(f"[ACT] Data directory: {data_root}")

        if name == 'mnist':
            dataset = torchvision.datasets.MNIST(
                root=data_root, train=False, download=True,
                transform=transforms.ToTensor())
            images = [img.numpy().squeeze() for img, _ in dataset]
            labels = [label for _, label in dataset]
        elif name in ['cifar', 'cifar10']:
            dataset = torchvision.datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.ToTensor())
            images = [img.numpy() for img, _ in dataset]
            labels = [label for _, label in dataset]
        else:
            raise ValueError(f"Unknown dataset name: {name}")

        if self.start is None or self.end is None:
            self.start = 0
            self.end = len(images)
        elif self.start < 0 or self.end > len(images) or self.start >= self.end:
            raise ValueError(f"Invalid start ({self.start}) or end ({self.end}) indices for dataset of size {len(images)}.")

        images = images[self.start:self.end]
        labels = labels[self.start:self.end]

        # Convert to numpy arrays first to avoid PyTorch performance warning
        images_array = np.array(images, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int64)
        
        self.input_center = torch.from_numpy(images_array).to(self.device)
        self.labels = torch.from_numpy(labels_array).to(self.device)
        self.spec_type = SpecType.LOCAL_LP

        print("Loaded dataset:", name)
        print(f"Dataset size: {len(images)} samples, input: {self.input_center}, labels: {self.labels}")

    # remaining methods (load_from_csv, load_from_vnnlib, helpers) are identical to the
    # previous implementation and kept here for brevity — full implementation follows.
    def load_from_csv(self) -> None:
        self._data_source = 'csv'

        if not self.dataset_path or not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path {self.dataset_path} not found.")

        with open(self.dataset_path, 'r') as f:
            first_line = f.readline().strip()

        has_header = 'label' in first_line.lower() or 'pixel' in first_line.lower()

        if has_header:
            print("Detected CSV header, skipping first row")

            raw = np.genfromtxt(self.dataset_path, delimiter=',', skip_header=1)
        else:
            print("No CSV header detected, reading all rows")
            raw = np.genfromtxt(self.dataset_path, delimiter=',')

        dataset_size = raw.shape[0]
        if self.start is None:
            self.start = 0
        if self.end is None or self.end > dataset_size:
            self.end = dataset_size

        if self.start < 0 or self.end > dataset_size or self.start >= self.end:
            raise ValueError(f"Invalid start ({self.start}) or end ({self.end}) indices for dataset of size {dataset_size}.")

        print(f"Loading CSV dataset: using samples {self.start} to {self.end-1} from total {dataset_size} samples")
        raw = raw[self.start:self.end]

        label_column = raw[:, 0]
        if np.any(np.isnan(label_column)):
            print(f"Warning: Found NaN values in label column. Replacing with 0.")
            label_column = np.nan_to_num(label_column, nan=0.0)

        labels = label_column.astype(int)

        labels = np.clip(labels, 0, 9)

        images = raw[:, 1:]

        self.input_center = torch.tensor(images, dtype=torch.float32).to(self.device)
        self.labels = torch.tensor(labels, dtype=torch.long).to(self.device)

        self.spec_type = SpecType.LOCAL_LP

    def load_from_vnnlib(self, file_path, anchor_csv_path=None, x_hat=None) -> None:
        self._data_source = 'vnnlib'

        if not self.dataset_path or not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path {self.dataset_path} not found.")

        with open(file_path, 'r') as f:
            lines = f.readlines()

        input_vars, output_vars, anchors, utility = VNNLIBParser.identify_declare_const(lines)

        is_local = VNNLIBParser.is_local(lines)

        idx_dict = {f"{x[0]}_{x[1]}": i for i, x in enumerate(sorted(anchors) + sorted(utility))}

        idx_dict_out = {f"{x[0]}_{x[1]}": i for i, x in enumerate(output_vars)}

        idx_dict_out["const_-1"] = len(output_vars)

        eps_dict = {}

        eps_scalar = None

        C_lb = np.zeros((len(input_vars), len(idx_dict)))

        C_ub = np.zeros((len(input_vars), len(idx_dict)))

        C_out = np.zeros((0, len(output_vars) + 1))

        if not is_local:

            input_lb = np.full(len(input_vars), -np.inf)
            input_ub = np.full(len(input_vars), np.inf)
            for line in lines:
                if not line.startswith("(assert"):
                    continue

                m_ge = re.match(r"\(assert \(>= X_(\d+) ([0-9.eE+-]+)\)\)", line.strip())
                m_le = re.match(r"\(assert \(<= X_(\d+) ([0-9.eE+-]+)\)\)", line.strip())
                if m_ge:
                    idx = int(m_ge.group(1))
                    val = float(m_ge.group(2))
                    input_lb[idx] = val
                    continue
                elif m_le:
                    idx = int(m_le.group(1))
                    val = float(m_le.group(2))
                    input_ub[idx] = val
                    continue

                if re.search(r"Y_\d+", line):
                    match = re.match(r"\(assert \(([><=]+) (.*)\)\)", line)
                    if not match:
                        continue
                    relation, expr = match.group(1), match.group(2)

                    first_term = re.match(r"\((.*)\) .*", expr).group(1) if expr.startswith("(") else expr.split()[0]
                    second_term = re.match(r".* \((.*)\)", expr).group(1) if expr.endswith(")") else expr.split()[-1]

                    greater_terms = VNNLIBParser.parse_term(first_term if relation == ">=" else second_term)
                    lesser_terms = VNNLIBParser.parse_term(second_term if relation == ">=" else first_term)

                    row = np.zeros((1, C_out.shape[1]))

                    for term in greater_terms:

                        row[0, idx_dict_out[f"{term[0]}_{term[1]}"]] += term[2]
                    for term in lesser_terms:

                        row[0, idx_dict_out[f"{term[0]}_{term[1]}"]] -= term[2]

                    C_out = np.vstack([C_out, row])
                    continue

            self.input_lb = torch.tensor(input_lb, dtype=torch.float32).unsqueeze(0).to(self.device)
            self.input_ub = torch.tensor(input_ub, dtype=torch.float32).unsqueeze(0).to(self.device)

            self.output_constraints = C_out
            self.spec_type = SpecType.SET_VNNLIB
            return

        count = 0
        for line in lines:
            print(f"Processing line {count}: {line.strip()}")
            count += 1
            if not line.startswith("(assert"):
                continue

            eps_match = re.match(r"\(assert \(= eps_(\d+) ([0-9.eE+-]+)\)\)", line.strip())
            if eps_match:
                idx = int(eps_match.group(1))
                eps_dict[idx] = float(eps_match.group(2))
                continue

            eps_scalar_match = re.match(r"\(assert \(= eps ([0-9.eE+-]+)\)\)", line.strip())
            if eps_scalar_match:
                eps_scalar = float(eps_scalar_match.group(1))
                print("eps_scalar", eps_scalar)
                continue

            if eps_match and eps_scalar_match:
                raise ValueError("Cannot define both per-dimension eps and set-based eps in the same vnnlib file.")

            match = re.match(r"\(assert \(([><=]+) (.*)\)\)", line)
            if not match:
                continue
            relation, expr = match.group(1), match.group(2)

            first_term = re.match(r"\((.*)\) .*", expr).group(1) if expr.startswith("(") else expr.split()[0]
            second_term = re.match(r".* \((.*)\)", expr).group(1) if expr.endswith(")") else expr.split()[-1]

            greater_terms = VNNLIBParser.parse_term(first_term if relation == ">=" else second_term)
            lesser_terms = VNNLIBParser.parse_term(second_term if relation == ">=" else first_term)

            if len(greater_terms) == 1 and greater_terms[0][0] == "X":
                print(greater_terms, lesser_terms)

                i = greater_terms[0][1]

                for term in lesser_terms:

                    print(f"Adding to C_lb: {term}, i: {i}, idx_dict: {idx_dict}")

                    C_lb[i, idx_dict[f"{term[0]}_{term[1]}"]] = term[2]

            elif len(lesser_terms) == 1 and lesser_terms[0][0] == "X":

                i = lesser_terms[0][1]

                for term in greater_terms:

                    C_ub[i, idx_dict[f"{term[0]}_{term[1]}"]] = term[2]

            else:

                row = np.zeros((1, C_out.shape[1]))

                for term in greater_terms:

                    row[0, idx_dict_out[f"{term[0]}_{term[1]}"]] += term[2]
                for term in lesser_terms:

                    row[0, idx_dict_out[f"{term[0]}_{term[1]}"]] -= term[2]

                C_out = np.vstack([C_out, row])
        print("Before entering is_local vnnlib")
        if is_local:
            all_input_lb = []
            all_input_ub = []

            if x_hat is None:
                print(f"Anchors loading")
                if anchor_csv_path is None:
                    raise ValueError("For local_vnnlib, must provide anchor_csv_path or x_hat.")
                
                # Check if the anchor CSV has header
                with open(anchor_csv_path, 'r') as f:
                    first_line = f.readline().strip()
                has_header = 'label' in first_line.lower() or 'pixel' in first_line.lower()
                
                if has_header:
                    print("Detected CSV header in anchor file, skipping first row")
                    raw_anchors = np.genfromtxt(anchor_csv_path, delimiter=',', skip_header=1)
                else:
                    print("No CSV header detected in anchor file, reading all rows")
                    raw_anchors = np.genfromtxt(anchor_csv_path, delimiter=',')
                
                if raw_anchors.ndim == 1:
                    raw_anchors = raw_anchors[None, :]
                
                # Extract labels from first column and pixel data from remaining columns
                if raw_anchors.shape[1] > 784:  # Has label column
                    anchor_labels = raw_anchors[:, 0].astype(int)
                    anchors = raw_anchors[:, 1:]  # Skip label column
                    print(f"Extracted labels from anchor CSV: {anchor_labels}")
                    # Set the labels for clean prediction
                    self.labels = torch.tensor(anchor_labels, dtype=torch.long).to(self.device)
                else:
                    anchors = raw_anchors
                    print("No label column detected in anchor CSV, using default labels")
                    # Set default labels if no label column
                    self.labels = torch.tensor([0] * raw_anchors.shape[0], dtype=torch.long).to(self.device)
                    
            else:
                anchors = np.array(x_hat)
                if anchors.ndim == 1:
                    anchors = anchors[None, :]
                # Set default labels when x_hat is provided directly
                self.labels = torch.tensor([0] * anchors.shape[0], dtype=torch.long).to(self.device)

            print(f"Anchors shape: {anchors.shape}")
            print(f"Labels: {self.labels}")
            
            # Apply normalization to anchor data if mean and std are provided
            if self.mean is not None and self.std is not None:
                print(f"Applying normalization to anchor data: mean={self.mean}, std={self.std}")
                anchors_tensor = torch.tensor(anchors, dtype=torch.float32).to(self.device)
                anchors_tensor = self._preprocessing_core(anchors_tensor)
                anchors = anchors_tensor.cpu().numpy()
                print(f"Anchor data range after normalization: [{anchors.min():.4f}, {anchors.max():.4f}]")
            else:
                print("No normalization applied to anchor data")

            if eps_scalar is None and len(eps_dict) == 0:
                raise ValueError("For local_vnnlib, eps must be provided.")

            if eps_dict:

                eps = [eps_dict[i] for i in range(len(eps_dict))]
            elif eps_scalar is not None:

                eps = [eps_scalar]
            else:
                raise ValueError("For local_vnnlib, eps must be provided or defined in vnnlib.")

            for i in range(anchors.shape[0]):
                x_hat = anchors[i]
                input_box = self._build_input_box(C_lb, C_ub, x_hat, eps)
                input_lb = [b[0] for b in input_box]
                input_ub = [b[1] for b in input_box]
                all_input_lb.append(input_lb)
                all_input_ub.append(input_ub)

            self.spec_type = SpecType.LOCAL_VNNLIB
            self.input_lb = torch.tensor(all_input_lb, dtype=torch.float32).to(self.device)
            self.input_ub = torch.tensor(all_input_ub, dtype=torch.float32).to(self.device)
            self.output_constraints = C_out

    def _build_input_box(self, C_lb, C_ub, x_hat=None, eps=None):

        n_x = C_lb.shape[0]
        x = []

        if x_hat is not None:

            x_hat = np.ones(n_x) * x_hat if len(x_hat) == 1 else np.array(x_hat)

            x.append(x_hat)

        n_e = C_lb.shape[1] - n_x

        if eps is not None:

            eps = np.ones(n_e) * eps if len(eps) == 1 else np.array(eps)
            x.append(eps)

        x_full = np.concatenate(x)

        print("x_hat shape:", x_hat.shape, "eps shape:", len(eps))
        print("C_lb shape:", C_lb.shape, "x_full shape:", x_full.shape)

        lb = np.matmul(C_lb, x_full)

        ub = np.matmul(C_ub, x_full)

        return [(lb[i], ub[i]) for i in range(n_x)]

    def load_set_box_from_json(self) -> None:
        self._data_source = 'json'

        if not self.dataset_path or not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path {self.dataset_path} not found.")

        with open(self.dataset_path, "r") as f:
            data = json.load(f)

        self.input_lb = torch.tensor(data['lb'], dtype=torch.float32).to(self.device)
        self.input_ub = torch.tensor(data['ub'], dtype=torch.float32).to(self.device)

        A = torch.tensor(data["output_constraints"]["A"], dtype=torch.float32)

        b = -torch.tensor(data["output_constraints"]["b"], dtype=torch.float32)
        self.output_constraints = torch.cat([A, b.unsqueeze(1)], dim=1)
        self.spec_type = SpecType.SET_BOX

        print("Loaded set-based box constraints from JSON file.")
        print(f"Input lower bounds shape: {self.input_lb.shape}, upper bounds shape: {self.input_ub.shape}")
        print(f"Input lb {self.input_lb}, ub {self.input_ub}")
        print(f"Output constraints shape: {self.output_constraints.shape}")
        print(f"Output constraints: {self.output_constraints}")

    def get_sample(self, i: int) -> Tuple[torch.Tensor, int]:
        if self.input_center is None or self.labels is None:
            raise ValueError("Inputs/labels not loaded. Did you call `load_local_lp_from_csv()`?")
        return self.input_center[i], self.labels[i]

    def _preprocessing_core(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is not None and self.std is not None:

            if isinstance(self.mean, (list, float)):
                mean = torch.tensor(self.mean, device=x.device, dtype=x.dtype)
            else:
                mean = self.mean

            if isinstance(self.std, (list, float)):
                std = torch.tensor(self.std, device=x.device, dtype=x.dtype)
            else:
                std = self.std

            if len(x.shape) == 4 and x.shape[1] == 3:

                if len(mean.shape) == 1 and mean.shape[0] == 3:

                    mean = mean.view(1, 3, 1, 1)
                    std = std.view(1, 3, 1, 1)
                    normalized = (x - mean) / std
                    return normalized
                else:
                    print(f"   Warning: mean/std shape mismatch for CIFAR-10 4D data")
                    return x

            elif len(x.shape) == 2 and x.shape[1] == 3072:

                x_temp = x.view(-1, 3, 32, 32)

                if len(mean.shape) == 1 and mean.shape[0] == 3:

                    mean = mean.view(1, 3, 1, 1)
                    std = std.view(1, 3, 1, 1)
                    normalized = (x_temp - mean) / std

                    result = normalized.view(x.shape)
                    return result
                else:
                    print(f"   Warning: mean/std shape mismatch for CIFAR-10 flattened data")
                    return x

            elif len(x.shape) == 2 and x.shape[1] == 784:
                if isinstance(mean, torch.Tensor) and mean.shape == torch.Size([]):

                    return (x - mean) / std
                else:
                    print(f"   Warning: unexpected mean/std for MNIST")
                    return x

            else:
                return (x - mean) / std

        return x

    def _apply_preprocessing(self):
        if not self.preprocess:
            return

        if self.input_center is None:
            return

        if hasattr(self, '_data_source'):
            if self._data_source in ['mnist', 'cifar', 'cifar10']:

                if self.mean is not None and self.std is not None:
                    self.input_center = self._preprocessing_core(self.input_center)
            elif self._data_source == 'csv':

                self._apply_csv_preprocessing()

        else:

            if self.mean is not None and self.std is not None:
                self.input_center = self._preprocessing_core(self.input_center)

    def _apply_csv_preprocessing(self):
        if self.input_center is None:
            return

        min_val = self.input_center.min().item()
        max_val = self.input_center.max().item()

        print(f"CSV data range: [{min_val:.4f}, {max_val:.4f}]")

        if max_val > 1.0:

            if max_val <= 255.0 and min_val >= 0.0:
                print("Detected raw pixel values [0-255], normalizing to [0-1]")
                self.input_center = self.input_center / 255.0

                if self.mean is not None and self.std is not None:
                    self.input_center = self._preprocessing_core(self.input_center)
            else:
                print("Unknown value range, applying user-specified normalization only")
                if self.mean is not None and self.std is not None:
                    self.input_center = self._preprocessing_core(self.input_center)
        else:

            print("Data appears to be already normalized, applying user-specified normalization")
            if self.mean is not None and self.std is not None:
                self.input_center = self._preprocessing_core(self.input_center)

    def __len__(self):
        return self.end - self.start


class BaseSpec:
    def __init__(self, dataset : Dataset = None, model : Model = None, status: VerifyStatus = VerifyStatus.UNKNOWN):
        self.dataset = dataset
        self.model = model


class InputSpec(BaseSpec):
    def __init__(self,
                 dataset : Dataset,
                 norm: Optional[LPNormType] = None,
                 epsilon: Optional[float] = None,
                 vnnlib_path: str = None,
                 ):

        super().__init__(dataset)
        if isinstance(self.dataset.spec_type, str):
            self.spec_type = SpecType(self.dataset.spec_type.lower())
        else:
            self.spec_type = self.dataset.spec_type

        if isinstance(norm, str):
            norm = LPNormType(norm.lower())

        self.norm = norm
        self.input_center = None
        self.epsilon = epsilon
        self.vnnlib_path = vnnlib_path
        self.input_lb = None
        self.input_ub = None

        if self.input_lb is not None and self.input_ub is not None:
            self._input_validation()

        if self.spec_type not in [SpecType.LOCAL_LP, SpecType.LOCAL_VNNLIB, SpecType.SET_VNNLIB, SpecType.SET_BOX]:
            raise ValueError(f"Unsupported specification type: {self.spec_type}")
        if self.spec_type == SpecType.LOCAL_LP and (norm is None or epsilon is None):
            raise ValueError("Norm type and epsilon value must be specified for local LP specifications")

        if self.spec_type == SpecType.LOCAL_LP:

            print(f"📊 [InputSpec] Epsilon = {epsilon}")

            if self.norm == LPNormType.LINF:
                if dataset.preprocess and hasattr(self.dataset, 'mean') and hasattr(self.dataset, 'std') and self.dataset.mean is not None and self.dataset.std is not None:

                    self.input_center = self.dataset.input_center

                    if isinstance(self.dataset.mean, list) and isinstance(self.dataset.std, list):

                        mean_tensor = torch.tensor(self.dataset.mean, dtype=torch.float32, device=self.dataset.input_center.device)
                        std_tensor = torch.tensor(self.dataset.std, dtype=torch.float32, device=self.dataset.input_center.device)

                        if len(self.dataset.input_center.shape) == 4:
                            mean_tensor = mean_tensor.view(1, -1, 1, 1)
                            std_tensor = std_tensor.view(1, -1, 1, 1)
                        elif len(self.dataset.input_center.shape) == 3:
                            mean_tensor = mean_tensor.view(-1, 1, 1)
                            std_tensor = std_tensor.view(-1, 1, 1)
                        elif len(self.dataset.input_center.shape) == 1:

                            C = len(self.dataset.mean)
                            pixels_per_channel = self.dataset.input_center.shape[0] // C
                            mean_tensor = mean_tensor.repeat_interleave(pixels_per_channel)
                            std_tensor = std_tensor.repeat_interleave(pixels_per_channel)
                        elif len(self.dataset.input_center.shape) == 2:
                            C = len(self.dataset.mean)
                            pixels_per_channel = self.dataset.input_center.shape[1] // C
                            mean_tensor = mean_tensor.repeat_interleave(pixels_per_channel).unsqueeze(0)
                            std_tensor = std_tensor.repeat_interleave(pixels_per_channel).unsqueeze(0)

                        original_pixels = self.dataset.input_center * std_tensor + mean_tensor
                    else:

                        mean = self.dataset.mean[0] if isinstance(self.dataset.mean, list) else self.dataset.mean
                        std = self.dataset.std[0] if isinstance(self.dataset.std, list) else self.dataset.std
                        original_pixels = self.dataset.input_center * std + mean

                    print(f"📊 Inverse normalized pixel value range: [{original_pixels.min():.6f}, {original_pixels.max():.6f}]")

                    lb_raw = torch.clamp(original_pixels - epsilon, 0.0, 1.0)
                    ub_raw = torch.clamp(original_pixels + epsilon, 0.0, 1.0)

                    print(f"📊 [0,1] space perturbation+clip range: LB=[{lb_raw.min():.6f}, {lb_raw.max():.6f}], UB=[{ub_raw.min():.6f}, {ub_raw.max():.6f}]")

                    if isinstance(self.dataset.mean, list) and isinstance(self.dataset.std, list):

                        self.input_lb = (lb_raw - mean_tensor) / std_tensor
                        self.input_ub = (ub_raw - mean_tensor) / std_tensor
                    else:

                        self.input_lb = (lb_raw - mean) / std
                        self.input_ub = (ub_raw - mean) / std

                    print(f"📊 Final normalized bounds: LB=[{self.input_lb.min():.6f}, {self.input_lb.max():.6f}], UB=[{self.input_ub.min():.6f}, {self.input_ub.max():.6f}]")

                    print(f"📊 Original center range: [{self.input_center.min():.6f}, {self.input_center.max():.6f}]")
                    print(f"📊 Perturbation interval width range: [{(self.input_ub - self.input_lb).min():.6f}, {(self.input_ub - self.input_lb).max():.6f}]")

                    original_lb = original_pixels - epsilon
                    original_ub = original_pixels + epsilon
                    clipped_lb = (original_lb < 0.0).sum().item()
                    clipped_ub = (original_ub > 1.0).sum().item()
                    total_pixels = original_pixels.numel()

                    print(f"📊 Physical clip statistics: LB clipped={clipped_lb}/{total_pixels} ({clipped_lb/total_pixels*100:.2f}%), UB clipped={clipped_ub}/{total_pixels} ({clipped_ub/total_pixels*100:.2f}%)")

                else:

                    self.input_center = self.dataset.input_center
                    original_pixels = self.dataset.input_center
                    self.input_lb = torch.clamp(original_pixels - epsilon, 0.0, 1.0)
                    self.input_ub = torch.clamp(original_pixels + epsilon, 0.0, 1.0)
                    self.input_center = (self.input_lb + self.input_ub) / 2.0

        elif self.spec_type == SpecType.LOCAL_VNNLIB:
            self.input_center = self.dataset.input_center
            self.input_lb = self.dataset.input_lb
            self.input_ub = self.dataset.input_ub

        elif self.spec_type in [SpecType.SET_VNNLIB, SpecType.SET_BOX]:
            self.input_lb = self.dataset.input_lb
            self.input_ub = self.dataset.input_ub
            # Calculate input_center as the midpoint of bounds
            self.input_center = (self.input_lb + self.input_ub) / 2.0

        else:
            raise ValueError(f"Unsupported spec type: {self.spec_type}")

    def _input_validation(self):
        if not torch.is_tensor(self.input_lb) or not torch.is_tensor(self.input_ub):
            raise ValueError("Input bounds must be torch tensors")
        if self.input_lb.shape != self.input_ub.shape:
            raise ValueError("Input bounds must have the same shape")
        if not torch.all(self.input_lb <= self.input_ub):
            raise ValueError("Lower bounds must be less than or equal to upper bounds")

    def _apply_preprocessing(self, tensor: torch.Tensor) -> torch.Tensor:
        mean, std = self.preprocessing.get("mean"), self.preprocessing.get("std")
        if mean is not None and std is not None:
            return (tensor - mean) / std
        return tensor

    def get_input_size(self) -> int:
        return self.input_lb.shape[0]


class OutputSpec(BaseSpec if 'BaseSpec' in globals() else object):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        if isinstance(self.dataset.spec_type, str):
            self.spec_type = SpecType(self.dataset.spec_type.lower())
        else:
            self.spec_type = self.dataset.spec_type
        self.labels = None
        self.output_constraints = None

        print(self.spec_type)
        if self.spec_type == SpecType.LOCAL_LP:
            self.labels = dataset.labels

        elif self.spec_type == SpecType.LOCAL_VNNLIB:
            self.labels = dataset.labels
            self.output_constraints = dataset.output_constraints

        elif self.spec_type == SpecType.SET_VNNLIB:
            self.output_constraints = dataset.output_constraints

        elif self.spec_type == SpecType.SET_BOX:
            self.output_constraints = dataset.output_constraints

        else:
            raise ValueError(f"Unsupported spec type: {self.spec_type}")


class Spec(BaseSpec if 'BaseSpec' in globals() else object):
    def __init__(self,
                 model : Model,
                 input_spec: InputSpec,
                 output_spec: OutputSpec):

        if input_spec.dataset != output_spec.dataset:
            raise ValueError("Input and output specifications must belong to the same dataset")

        if input_spec.spec_type != output_spec.spec_type:
            raise ValueError("Input and output specifications must have the same specification type")
        super().__init__(dataset=input_spec.dataset, model=model)
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.spec_type = self.input_spec.spec_type
