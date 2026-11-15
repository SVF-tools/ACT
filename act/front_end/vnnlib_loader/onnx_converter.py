#===- act/front_end/vnnlib/onnx_converter.py - ONNX to PyTorch -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Convert ONNX models to PyTorch nn.Module for unified verification interface.
#   Supports model validation and shape inference.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Tuple
import itertools
import logging
import math
import types
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from onnx import GraphProto, ModelProto, NodeProto, ValueInfoProto

logger = logging.getLogger(__name__)

_QDQ_REGISTERED = False


def _register_qdq_converters() -> None:
    """Register ACT-local onnx2torch converters for ONNX QuantizeLinear /
    DequantizeLinear (opset 10/13). Idempotent; imports onnx2torch lazily so
    non-VNNLIB workflows never require the onnx stack."""
    global _QDQ_REGISTERED
    if _QDQ_REGISTERED:
        return

    from typing import cast

    from onnx2torch.node_converters.registry import add_converter
    from onnx2torch.onnx_graph import OnnxGraph
    from onnx2torch.onnx_node import OnnxNode
    from onnx2torch.utils.common import (
        OnnxToTorchModule,
        OperationConverterResult,
        get_const_value,
        onnx_mapping_from_node,
    )

    def _reshape_axis_param(param: torch.Tensor, x: torch.Tensor, axis) -> torch.Tensor:
        if param.numel() == 1 or axis is None or param.dim() != 1 or x.dim() == 0:
            return param
        ax = axis + x.dim() if axis < 0 else axis
        shape = [1] * x.dim()
        shape[ax] = int(param.numel())
        return param.reshape(shape)

    def _qrange_from_zero_point(zero_point: torch.Tensor):
        if zero_point.dtype == torch.int8:
            return -128, 127, "int8"
        if zero_point.dtype == torch.uint8:
            return 0, 255, "uint8"
        if zero_point.dtype == torch.int32:
            return -(2**31), 2**31 - 1, "int32"
        raise NotImplementedError(f"QuantizeLinear zero_point dtype {zero_point.dtype} is not supported")

    class _OnnxQdqBase(nn.Module, OnnxToTorchModule):
        def __init__(self, scale, zero_point, axis=None):
            super().__init__()
            # Registered at the file's own dtype so the model-wide .to(dtype=...)
            # reaches it; zero_point stays integral, which .to() leaves alone.
            self.register_buffer("scale", scale.detach().clone())
            self.register_buffer("zero_point", zero_point.detach().clone())
            self.axis = axis
            self.qmin, self.qmax, self.dtype_name = _qrange_from_zero_point(cast(torch.Tensor, self.zero_point))

    class OnnxQuantizeLinear(_OnnxQdqBase):
        def forward(self, x, scale=None, zero_point=None):
            # ONNX semantics: the integer code, held in x's float dtype. The
            # following DequantizeLinear maps it back to s * (code - zp); torch2act
            # fuses a matching Q/DQ pair into one real-valued QUANTIZE layer.
            s = _reshape_axis_param(cast(torch.Tensor, self.scale).to(device=x.device, dtype=x.dtype), x, self.axis)
            zp = _reshape_axis_param(cast(torch.Tensor, self.zero_point).to(device=x.device, dtype=x.dtype), x, self.axis)
            return torch.clamp(torch.round(x / s) + zp, min=float(self.qmin), max=float(self.qmax))

    class OnnxDequantizeLinear(_OnnxQdqBase):
        def forward(self, q, scale=None, zero_point=None):
            # ONNX DequantizeLinear yields the scale's type; for an integral q
            # that is the only thing naming the working float precision.
            scale_buffer = cast(torch.Tensor, self.scale)
            dtype = scale_buffer.dtype if not q.is_floating_point() else q.dtype
            qf = q.to(dtype=dtype)
            s = _reshape_axis_param(cast(torch.Tensor, self.scale).to(device=q.device, dtype=dtype), qf, self.axis)
            zp = _reshape_axis_param(cast(torch.Tensor, self.zero_point).to(device=q.device, dtype=dtype), qf, self.axis)
            return s * (qf - zp)

    def _axis(node: "OnnxNode"):
        raw = node.attributes.get("axis")
        return None if raw is None else int(raw)

    def _const_tensor(name: str, graph: "OnnxGraph") -> torch.Tensor:
        value = get_const_value(name, graph)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        return value

    def _qdq_converter(module_class):
        def convert(node: "OnnxNode", graph: "OnnxGraph") -> "OperationConverterResult":
            return OperationConverterResult(
                torch_module=module_class(_const_tensor(node.input_values[1], graph),
                                          _const_tensor(node.input_values[2], graph), _axis(node)),
                onnx_mapping=onnx_mapping_from_node(node=node),
            )
        return convert

    for operation_type, module_class in (("QuantizeLinear", OnnxQuantizeLinear),
                                         ("DequantizeLinear", OnnxDequantizeLinear)):
        for version in (13, 10):
            add_converter(operation_type=operation_type, version=version)(_qdq_converter(module_class))

    _QDQ_REGISTERED = True


class ONNXConversionError(Exception):
    """Exception raised when ONNX conversion fails."""
    pass


def _fold_dequantize_initializers(onnx_model):
    """Fold DequantizeLinear nodes whose data input is an initializer.

    onnx2torch Conv/Gemm converters expect weights/biases as initializers, not
    computed node outputs. This rewrite is exact: y = scale * (q - zero_point),
    including per-axis scale/zero_point via the ONNX axis attribute.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    init_map = {init.name: numpy_helper.to_array(init) for init in onnx_model.graph.initializer}
    keep_nodes = []
    folded = 0
    for node in onnx_model.graph.node:
        if node.op_type != 'DequantizeLinear' or len(node.input) < 3 or node.input[0] not in init_map:
            keep_nodes.append(node)
            continue
        q = init_map[node.input[0]]
        scale = init_map.get(node.input[1])
        zp = init_map.get(node.input[2])
        if scale is None or zp is None:
            keep_nodes.append(node)
            continue
        axis = None
        for attr in node.attribute:
            if attr.name == 'axis':
                axis = int(onnx.helper.get_attribute_value(attr))
                break
        qf = q.astype(np.float64)
        sf = scale.astype(np.float64)
        zpf = zp.astype(np.float64)
        if axis is not None and sf.ndim == 1 and sf.size != 1 and qf.ndim > 0:
            ax = axis + qf.ndim if axis < 0 else axis
            shape = [1] * qf.ndim
            shape[ax] = sf.size
            sf = sf.reshape(shape)
            zpf = zpf.reshape(shape)
        # Computed wide, then stored back at the scale's own width: a wider
        # initializer would not match the types the rest of the graph declares.
        value = (sf * (qf - zpf)).astype(scale.dtype)
        onnx_model.graph.initializer.append(numpy_helper.from_array(value, name=node.output[0]))
        folded += 1
    if folded:
        del onnx_model.graph.node[:]
        onnx_model.graph.node.extend(keep_nodes)
        logger.info(f"Folded {folded} constant DequantizeLinear node(s) into initializers")
    return onnx_model


def _preprocess_onnx_for_onnx2torch(onnx_model):
    """Workarounds for onnx2torch quirks. Called on both main and retry paths.

    1. Dynamic batch dim: set first ``dim_value=0`` → 1 only when no node
       reads a tensor shape (``Shape``/``Size``), so the value cannot reach
       the computation and onnxsim may use static shapes (e.g. MatMul + Add
       → Gemm). Otherwise the dim stays symbolic: vit_2023 builds a Reshape
       target from Shape -> Gather -> Concat, and a fixed 1 would be folded
       into it, leaving a module that accepts batch 1 only.
       ``ClearField('dim_param')`` comes first since ``dim_value`` /
       ``dim_param`` are a protobuf oneof. Only the first dim is normalised;
       variable spatial dims are left alone so we don't mask real
       shape-concreteness errors.
    2. Empty Clip max input (cctsdb_yolo_2023): trim trailing empty input
       slots so onnx2torch's clip.py doesn't see them as present.
    """
    if not any(node.op_type in ("Shape", "Size") for node in onnx_model.graph.node):
        for inp in onnx_model.graph.input:
            dims = list(inp.type.tensor_type.shape.dim)
            if dims and dims[0].dim_value == 0:
                dims[0].ClearField('dim_param')
                dims[0].dim_value = 1
    for node in onnx_model.graph.node:
        if node.op_type == 'Clip':
            while len(node.input) > 1 and not node.input[-1]:
                del node.input[-1]
    onnx_model = _fold_dequantize_initializers(onnx_model)
    return onnx_model


def _ensure_unique_node_names(onnx_model: ModelProto) -> ModelProto:
    """Give graph nodes nonempty unique names without changing value edges."""
    reserved = {node.name for node in onnx_model.graph.node if node.name}
    seen: set[str] = set()
    for index, node in enumerate(onnx_model.graph.node):
        if node.name and node.name not in seen:
            seen.add(node.name)
            continue
        base = f"act_{node.op_type}_{index}"
        name = base
        suffix = 1
        while name in reserved:
            name = f"{base}_{suffix}"
            suffix += 1
        node.name = name
        reserved.add(name)
        seen.add(name)
    return onnx_model


PACKED_INPUT_NAME = "act_packed_input"


def _data_inputs(graph: GraphProto) -> list[ValueInfoProto]:
    """Graph inputs that are fed at run time (IR < 4 models also list initializers)."""
    initializers = {init.name for init in graph.initializer}
    return [inp for inp in graph.input if inp.name not in initializers]


def pack_onnx_inputs(onnx_model: ModelProto) -> ModelProto:
    """Replace several data inputs with one flat input ``[batch, sum(numel)]``.

    ACT verifies one input tensor per network, so a network with inputs
    ``x_1 .. x_k`` becomes ``f(Slice/Reshape(p))`` over their row-major
    concatenation ``p`` in graph-input order -- the order in which VNNLIB 2.0
    declares them (see ``vnnlib_parser._input_layout``). The rewrite is exact:
    the original graph runs unchanged on the reconstructed tensors.
    Single-input models are returned untouched.
    """
    import numpy as np
    from onnx import helper, numpy_helper

    graph = onnx_model.graph
    inputs = _data_inputs(graph)
    if len(inputs) < 2:
        return onnx_model
    elem_types = {inp.type.tensor_type.elem_type for inp in inputs}
    if len(elem_types) != 1:
        raise ONNXConversionError("multi-input ONNX model mixes input element types")
    batch_dims, row_shapes = [], []
    for inp in inputs:
        dims = list(inp.type.tensor_type.shape.dim)
        if len(dims) < 2 or any(d.dim_value <= 0 for d in dims[1:]):
            raise ONNXConversionError(
                f"multi-input ONNX model needs a batch axis and static row shape; "
                f"input {inp.name!r} has shape {_extract_shape_from_tensor(inp)}"
            )
        batch_dims.append(dims[0].dim_value)
        row_shapes.append([int(d.dim_value) for d in dims[1:]])
    if len(set(batch_dims)) != 1:
        raise ONNXConversionError(f"multi-input ONNX model has differing batch extents {batch_dims}")

    nodes, constants, offset = [], [], 0
    for index, (inp, row_shape) in enumerate(zip(inputs, row_shapes)):
        size = int(np.prod(row_shape))
        prefix = f"{PACKED_INPUT_NAME}_{index}"
        constants += [
            numpy_helper.from_array(np.array([offset], np.int64), f"{prefix}_start"),
            numpy_helper.from_array(np.array([offset + size], np.int64), f"{prefix}_end"),
            numpy_helper.from_array(np.array([1], np.int64), f"{prefix}_axis"),
            numpy_helper.from_array(np.array([-1, *row_shape], np.int64), f"{prefix}_shape"),
        ]
        nodes += [
            helper.make_node("Slice", [PACKED_INPUT_NAME, f"{prefix}_start", f"{prefix}_end", f"{prefix}_axis"],
                             [f"{prefix}_flat"], name=f"{prefix}_slice"),
            helper.make_node("Reshape", [f"{prefix}_flat", f"{prefix}_shape"], [inp.name],
                             name=f"{prefix}_reshape"),
        ]
        offset += size
    packed = helper.make_tensor_value_info(
        PACKED_INPUT_NAME, elem_types.pop(),
        [batch_dims[0] or inputs[0].type.tensor_type.shape.dim[0].dim_param or "batch", offset],
    )
    for inp in inputs:
        graph.input.remove(inp)
    graph.input.insert(0, packed)
    graph.initializer.extend(constants)
    original_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes + original_nodes)
    logger.info("Packed %d ONNX inputs into one [batch, %d] input", len(inputs), offset)
    return onnx_model


_INTEGER_TYPES = frozenset(range(2, 8)) | {12, 13}  # onnx.TensorProto (U)INT8..64
MAX_INTEGER_INPUT_CASES = 1 << 16


def _attribute(node, name: str, default=None):
    from onnx import helper

    return next((helper.get_attribute_value(a) for a in node.attribute if a.name == name), default)


@dataclass(frozen=True)
class IntegerInputCases:
    """Input coordinates a network reads only through ``Cast`` to an integer type.

    Such a coordinate takes the finitely many values ``trunc(lb) .. trunc(ub)``
    in a bounded box, and the part of the network computed from these
    coordinates alone (e.g. cctsdb_yolo_2023's patch mask, whose dynamic
    Slice/Range/ScatterND shapes follow the patch position) is fixed once they
    are. ``remainder`` is the network with that part cut off: its one input row
    is the declared input, flattened, followed by the ``extra`` values of those
    cut tensors, so every remaining shape is static. ``case_rows`` enumerates
    the distinct cases of a box as input rows of ``remainder``.
    """

    coordinates: tuple[int, ...]
    declared_shape: tuple[int, ...]
    extra: int
    remainder: "ModelProto"
    region: "ModelProto"  # declared input -> the cut tensors, in remainder input order

    @classmethod
    def find(cls, model: "ModelProto") -> "IntegerInputCases | None":
        """Cut ``model`` (prepared, shape-inferred) at its integer-cast inputs, if it has any."""
        import numpy as np
        from onnx import numpy_helper

        graph = model.graph
        inputs = _data_inputs(graph)
        if len(inputs) != 1:
            return None
        x = inputs[0]
        shape = [d.dim_value for d in x.type.tensor_type.shape.dim]
        if not shape or any(d <= 0 for d in shape):
            return None
        consts = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
        for node in graph.node:
            value = _attribute(node, "value") if node.op_type == "Constant" else None
            if value is not None:
                consts[node.output[0]] = numpy_helper.to_array(value)
        consumers: dict[str, list[NodeProto]] = {}
        for node in graph.node:
            for name in node.input:
                consumers.setdefault(name, []).append(node)
        flat = np.arange(math.prod(shape)).reshape(shape)

        def reads(node) -> set[int] | None:
            """Flat input coordinates a constant-index Gather/Slice of the input reads."""
            if node.op_type == "Gather" and node.input[1] in consts:
                return set(np.take(flat, consts[node.input[1]], axis=_attribute(node, "axis", 0)).ravel().tolist())
            args = list(node.input[1:]) + [""] * (5 - len(node.input))
            if node.op_type == "Slice" and all(not a or a in consts for a in args):
                starts, ends = consts[args[0]], consts[args[1]]
                axes = consts[args[2]] if args[2] else range(len(starts))
                steps = consts[args[3]] if args[3] else [1] * len(starts)
                index = [slice(None)] * flat.ndim
                for start, end, axis, step in zip(starts, ends, axes, steps):
                    index[int(axis)] = slice(int(start), int(end), int(step))
                return set(flat[tuple(index)].ravel().tolist())
            return None

        cast_only, other = [], set()
        for node in consumers.get(x.name, []):
            read = reads(node)
            uses = consumers.get(node.output[0], [])
            if (read is not None and node.op_type == "Gather" and uses
                    and all(u.op_type == "Cast" and _attribute(u, "to") in _INTEGER_TYPES for u in uses)):
                cast_only.append((node, read))
            else:
                other |= set(range(flat.size)) if read is None else read
        cut = [(node, read) for node, read in cast_only if not read & other]
        if not cut:
            return None

        # dyn: computed from the enumerated coordinates; dat: from the remaining input data.
        # Shape/Size read no data, so a static shape does not carry data dependence.
        removed = {id(node) for node, _ in cut}
        dyn = {name for node, _ in cut for name in node.output}
        dat: set[str] = set()
        for node in graph.node:
            if id(node) in removed:
                continue
            names = [n for n in node.input if n]
            if x.name in names:
                dat.update(node.output)
                continue
            if any(n in dyn for n in names):
                dyn.update(node.output)
            if node.op_type not in ("Shape", "Size") and any(n in dat for n in names):
                dat.update(node.output)
        region = removed | {id(n) for n in graph.node
                            if any(o in dyn for o in n.output) and not any(o in dat for o in n.output)}
        produced = {o for n in graph.node if id(n) in region for o in n.output}
        outputs = {o.name for o in graph.output}
        boundary = sorted({name for n in graph.node if id(n) not in region for name in n.input if name in produced}
                          | (produced & outputs))
        infos = {v.name: v for v in [*graph.value_info, *graph.output]}
        elem_type = x.type.tensor_type.elem_type
        boundary_shapes = []
        for name in boundary:
            info = infos.get(name)
            dims = [d.dim_value for d in info.type.tensor_type.shape.dim] if info is not None else []
            if info is None or info.type.tensor_type.elem_type != elem_type or any(d <= 0 for d in dims):
                raise ONNXConversionError(
                    f"tensor {name!r}, computed only from the integer-cast input coordinates "
                    f"{sorted(set().union(*(r for _, r in cut)))}, reaches the network without a static "
                    f"shape and the input's element type; ACT cannot enumerate these inputs")
            boundary_shapes.append(dims)
        return cls(
            coordinates=tuple(sorted(set().union(*(r for _, r in cut)))),
            declared_shape=tuple(shape),
            extra=sum(math.prod(s) for s in boundary_shapes),
            remainder=_remainder_model(model, x, region, boundary, boundary_shapes),
            region=_region_model(model, x.name, boundary),
        )

    def case_rows(self, lb: torch.Tensor, ub: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Input-row bounds ``(lb, ub)`` of ``remainder`` for each distinct case of the box.

        Each enumerated coordinate is fixed to a representative of its case,
        ``clamp(v, lb, ub)`` (so that ``trunc`` gives ``v``); the other
        coordinates keep their bounds, and the cut tensors follow as fixed
        values. Cases whose cut tensors coincide share one row. A row's
        leading ``prod(declared_shape)`` values are an input of the original
        network.
        """
        import numpy as np
        import onnxruntime as ort

        lo = lb.detach().reshape(-1).cpu().double().numpy()
        hi = ub.detach().reshape(-1).cpu().double().numpy()
        coords = list(self.coordinates)
        if not (np.isfinite(lo[coords]).all() and np.isfinite(hi[coords]).all()):
            raise ONNXConversionError(f"integer-cast inputs {coords} need finite bounds to be enumerated")
        ranges = [range(int(np.trunc(lo[i])), int(np.trunc(hi[i])) + 1) for i in coords]
        count = math.prod(len(r) for r in ranges)
        if count > MAX_INTEGER_INPUT_CASES:
            raise ONNXConversionError(
                f"integer-cast inputs {coords} span {count} cases (at most {MAX_INTEGER_INPUT_CASES})")
        session = ort.InferenceSession(self.region.SerializeToString(), providers=["CPUExecutionProvider"])
        feed = session.get_inputs()[0].name
        center = ((lo + hi) / 2).astype(np.float32)
        rows: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
        for case in itertools.product(*ranges):
            point = center.copy()
            point[coords] = np.clip(np.asarray(case, dtype=np.float64), lo[coords], hi[coords])
            values = session.run(None, {feed: point.reshape(self.declared_shape)})
            fixed = np.concatenate([np.asarray(v, dtype=np.float64).reshape(-1) for v in values])
            key = fixed.tobytes()
            if key not in rows:
                rows[key] = (point[coords].astype(np.float64), fixed)
        out = []
        for representative, fixed in rows.values():
            row_lo, row_hi = lo.copy(), hi.copy()
            row_lo[coords] = row_hi[coords] = representative
            pair = [torch.as_tensor(np.concatenate([b, fixed]), dtype=lb.dtype, device=lb.device).reshape(1, -1)
                    for b in (row_lo, row_hi)]
            out.append((pair[0], pair[1]))
        return out


def _forward_lane_by_lane(module: nn.Module, x: torch.Tensor, *args):
    """``forward`` of a converted graph that fixes one sample: run each lane of ``x`` alone.

    Tracing (TorchToACT, merge_split_relus) reads the class-level graph forward, so
    the analysed graph is unchanged; only eager batched calls loop.
    """
    forward = type(module).forward
    if x.shape[0] == 1:
        return forward(module, x, *args)
    return torch.cat([forward(module, x[i:i + 1], *args) for i in range(x.shape[0])])


def _region_model(model: "ModelProto", input_name: str, boundary: list[str]) -> "ModelProto":
    from onnx.utils import Extractor

    return Extractor(model).extract_model([input_name], boundary)


def _remainder_model(model: "ModelProto", x: "ValueInfoProto", region: set[int], boundary: list[str],
                     boundary_shapes: list[list[int]]) -> "ModelProto":
    """``model`` without ``region``: the input and the boundary tensors come from one row."""
    import numpy as np
    from onnx import helper, numpy_helper

    graph = model.graph
    shape = [d.dim_value for d in x.type.tensor_type.shape.dim]
    kept = [n for n in graph.node if id(n) not in region]
    feeds = [(x.name, shape)] if any(x.name in n.input for n in kept) else []
    feeds += list(zip(boundary, boundary_shapes))
    nodes, constants, offset = [], [], 0
    starts = {x.name: 0}
    for name, dims in [(x.name, shape), *zip(boundary, boundary_shapes)]:
        starts[name] = offset
        offset += math.prod(dims)
    for index, (name, dims) in enumerate(feeds):
        prefix = f"{PACKED_INPUT_NAME}_case_{index}"
        start = starts[name]
        constants += [
            numpy_helper.from_array(np.array([start], np.int64), f"{prefix}_start"),
            numpy_helper.from_array(np.array([start + math.prod(dims)], np.int64), f"{prefix}_end"),
            numpy_helper.from_array(np.array([1], np.int64), f"{prefix}_axis"),
            numpy_helper.from_array(np.array(dims, np.int64), f"{prefix}_shape"),
        ]
        nodes += [
            helper.make_node("Slice", [PACKED_INPUT_NAME, f"{prefix}_start", f"{prefix}_end", f"{prefix}_axis"],
                             [f"{prefix}_flat"], name=f"{prefix}_slice"),
            helper.make_node("Reshape", [f"{prefix}_flat", f"{prefix}_shape"], [name], name=f"{prefix}_reshape"),
        ]
    row = helper.make_tensor_value_info(PACKED_INPUT_NAME, x.type.tensor_type.elem_type, [1, offset])
    remainder = helper.make_model(
        helper.make_graph(nodes + kept, graph.name, [row], list(graph.output), [*graph.initializer, *constants]),
        opset_imports=model.opset_import, ir_version=model.ir_version)
    logger.info("Cut %d integer-cast-input nodes; the remainder reads one [1, %d] row", len(region), offset)
    return remainder


def _prepare_onnx_for_conversion(onnx_model: ModelProto) -> ModelProto:
    """Apply the same opset and graph preparation on main and raw retry."""
    try:
        from onnx import version_converter

        current_opset = max(
            (op.version for op in onnx_model.opset_import if not op.domain or op.domain == "ai.onnx"),
            default=0,
        )
        if 0 < current_opset < 13:
            logger.info("Upgrading ONNX opset %s → 13", current_opset)
            onnx_model = version_converter.convert_version(onnx_model, 13)
    except Exception as exc:
        logger.warning("Opset upgrade failed (%s), proceeding with original opset", exc)
    return _ensure_unique_node_names(_preprocess_onnx_for_onnx2torch(pack_onnx_inputs(onnx_model)))


def _infer_shapes(onnx_model: ModelProto) -> ModelProto:
    """Best-effort value shapes for onnx2torch, keeping symbolic dimensions.

    onnx2torch reads ranks from ``value_info`` (BatchNormalization picks its
    class from the input rank), and ONNX's own inference loses them behind
    shape arithmetic on a symbolic dimension. Graphs with symbolic inputs
    therefore use onnxruntime's symbolic shape inference. Its ``infer_shapes``
    entry point writes the model to the working directory when inference is
    incomplete, so its steps are run here and partial annotations are kept.
    """
    from onnx import shape_inference

    if any(dim.dim_value <= 0 for inp in _data_inputs(onnx_model.graph)
           for dim in inp.type.tensor_type.shape.dim):
        try:
            from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

            inference = SymbolicShapeInference(
                int_max=2**31 - 1, auto_merge=True, guess_output_rank=False, verbose=0,
            )
            inference._preprocess(onnx_model)
            while inference.run_:
                inference._infer_impl()
            inference._update_output_from_vi()
            return inference.out_mp_
        except Exception as e:
            logger.warning(f"Symbolic shape inference failed ({e}); using ONNX shape inference")
    try:
        return shape_inference.infer_shapes(onnx_model)
    except Exception as e:
        logger.warning(f"ONNX shape inference failed ({e}); proceeding without it")
        return onnx_model


def _simplify_and_infer(onnx_model: ModelProto, simplify: bool) -> ModelProto:
    """Optionally simplify, then propagate types/shapes through the graph."""
    if simplify:
        try:
            import onnxsim
            logger.info("Simplifying ONNX model")
            onnx_model, check = onnxsim.simplify(onnx_model)
            if not check:
                logger.warning("ONNX simplification check failed, using original model")
            onnx_model = _ensure_unique_node_names(onnx_model)
        except ImportError:
            logger.warning("onnxsim not available, skipping simplification")
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}, using original model")

    # VNN-COMP ONNX files can leave intermediate values with ValueType.UNKNOWN
    # (nn4sys) or lose annotations during onnxsim simplification; without this
    # step onnx2torch raises "Got unexpected input value type".
    return _infer_shapes(onnx_model)


def convert_onnx_to_pytorch(
    onnx_path: Path,
    simplify: bool = True
) -> nn.Module:
    """
    Convert ONNX model to PyTorch nn.Module.
    
    Args:
        onnx_path: Path to .onnx file
        simplify: Whether to simplify ONNX model before conversion
        
    Returns:
        PyTorch nn.Module equivalent to ONNX model
        
    Raises:
        ONNXConversionError: If conversion fails
    """
    return convert_onnx_instance(onnx_path, simplify, split_integer_inputs=False)[0]


def convert_onnx_instance(
    onnx_path: Path,
    simplify: bool = True,
    *,
    split_integer_inputs: bool = True,
) -> Tuple[nn.Module, IntegerInputCases | None]:
    """Convert an ONNX model; with integer-cast inputs, convert the remainder network instead.

    Returns the PyTorch module and, when ``split_integer_inputs`` found such
    inputs, their :class:`IntegerInputCases`, whose ``case_rows`` give the
    module's input rows. Raises ONNXConversionError if conversion fails.
    """
    if not onnx_path.exists():
        raise ONNXConversionError(f"ONNX file not found: {onnx_path}")
    
    try:
        # Import here to avoid requiring onnx for non-VNNLIB workflows
        import onnx
        _register_qdq_converters()
        from onnx2torch import convert
        
        # Load ONNX model
        logger.info(f"Loading ONNX model from {onnx_path}")
        onnx_model = onnx.load(str(onnx_path))
        
        onnx_model = _simplify_and_infer(_prepare_onnx_for_conversion(onnx_model), simplify)
        cases = IntegerInputCases.find(onnx_model) if split_integer_inputs else None
        if cases is not None:
            onnx_model = _simplify_and_infer(_ensure_unique_node_names(cases.remainder), simplify)

        # Convert to PyTorch. Simplification occasionally leaves intermediate
        # values with ValueType.UNKNOWN (nn4sys); only retry-without-simplify
        # for that specific upstream error so unrelated conversion bugs
        # (unsupported ops, shape errors) still surface normally.
        logger.info("Converting ONNX to PyTorch")
        try:
            pytorch_model = convert(onnx_model)
        except Exception as convert_err:
            if simplify and cases is None and "ValueType.UNKNOWN" in str(convert_err):
                logger.warning(
                    f"onnx2torch failed on simplified graph ({convert_err}); "
                    f"retrying with simplify=False"
                )
                raw_model = _infer_shapes(_prepare_onnx_for_conversion(onnx.load(str(onnx_path))))
                pytorch_model = convert(raw_model)
            else:
                raise
        pytorch_model.eval()
        if cases is not None:
            # The remainder keeps the original graph's one-sample shapes (cctsdb: a
            # rank-1 input with no batch axis), while each case is one input lane.
            pytorch_model.forward = types.MethodType(_forward_lane_by_lane, pytorch_model)
        
        # Convert model to match device_manager settings
        try:
            from act.util.device_manager import get_default_device, get_default_dtype
            target_device = get_default_device()
            target_dtype = get_default_dtype()
            
            # Move model to target device and dtype
            pytorch_model = pytorch_model.to(dtype=target_dtype, device=target_device)
            logger.info(f"Converted model to device={target_device}, dtype={target_dtype}")
        except Exception as e:
            logger.warning(f"Could not apply device_manager settings: {e}")
        
        logger.info(f"Successfully converted ONNX model: {onnx_path.name}")
        return pytorch_model, cases
        
    except ImportError as e:
        raise ONNXConversionError(
            f"Missing dependency for ONNX conversion: {e}\n"
            "Install with: pip install onnx onnx2torch onnxsim"
        )
    except Exception as e:
        raise ONNXConversionError(f"Failed to convert {onnx_path}: {str(e)}")


def get_onnx_input_shape(onnx_path: Path) -> Tuple[int, ...]:
    """
    Extract input shape from ONNX model.
    
    Args:
        onnx_path: Path to .onnx file
        
    Returns:
        Declared input shape; only a dynamic leading dimension is normalized
        to 1. Fixed leading extents remain model structure for the selected
        supported conversion operators.
        
    Raises:
        ONNXConversionError: If shape extraction fails
    """
    try:
        import onnx
        
        onnx_model = onnx.load(str(onnx_path))
        graph = onnx_model.graph
        
        data_inputs = _data_inputs(graph)
        if not data_inputs:
            raise ONNXConversionError("ONNX model has no inputs")
        if len(data_inputs) > 1:
            # Converted models take the packed input built by pack_onnx_inputs.
            data_inputs = _data_inputs(pack_onnx_inputs(onnx_model).graph)
        shape = _extract_shape_from_tensor(data_inputs[0])
        
        # Handle batch dimension - keep original, but normalize dynamic batch
        if not shape:
            raise ONNXConversionError("Failed to extract valid shape from ONNX model")
        
        if shape[0] == -1:
            # Dynamic batch: normalize to 1 for verification (requires concrete shape)
            shape = (1,) + tuple(shape[1:])
            logger.info(f"Normalized dynamic batch to 1: {shape}")
        else:
            # A fixed leading extent remains part of one model input.
            logger.info("Extracted fixed ONNX input shape: %s", shape)
        
        return tuple(shape)
        
    except ImportError:
        raise ONNXConversionError("onnx library not installed")
    except Exception as e:
        raise ONNXConversionError(f"Failed to extract shape from {onnx_path}: {str(e)}")


def _extract_shape_from_tensor(tensor) -> list:
    """
    Extract shape from ONNX tensor proto.
    
    Args:
        tensor: ONNX tensor (ValueInfoProto)
        
    Returns:
        List of dimension sizes (-1 for dynamic dimensions)
    """
    shape = []
    
    if hasattr(tensor, 'type') and hasattr(tensor.type, 'tensor_type'):
        tensor_type = tensor.type.tensor_type
        if hasattr(tensor_type, 'shape'):
            for dim in tensor_type.shape.dim:
                if hasattr(dim, 'dim_value'):
                    shape.append(dim.dim_value if dim.dim_value > 0 else -1)
                elif hasattr(dim, 'dim_param'):
                    # Dynamic dimension
                    shape.append(-1)
    
    return shape
