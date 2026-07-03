#===- act/pipeline/verification/onnx_quantization.py ---------------------====#
# ACT-local onnx2torch converters for ONNX QuantizeLinear/DequantizeLinear.
#===---------------------------------------------------------------------===#

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule, OperationConverterResult, onnx_mapping_from_node, get_const_value


def _reshape_axis_param(param: torch.Tensor, x: torch.Tensor, axis: int | None) -> torch.Tensor:
    if param.numel() == 1 or axis is None or param.dim() != 1 or x.dim() == 0:
        return param
    ax = axis + x.dim() if axis < 0 else axis
    shape = [1] * x.dim()
    shape[ax] = int(param.numel())
    return param.reshape(shape)


def _qrange_from_zero_point(zero_point: torch.Tensor) -> tuple[int, int, str]:
    if zero_point.dtype == torch.int8:
        return -128, 127, "int8"
    if zero_point.dtype == torch.uint8:
        return 0, 255, "uint8"
    if zero_point.dtype == torch.int32:
        return -(2**31), 2**31 - 1, "int32"
    raise NotImplementedError(f"QuantizeLinear zero_point dtype {zero_point.dtype} is not supported")


class OnnxQuantizeLinear(nn.Module, OnnxToTorchModule):
    def __init__(self, scale: torch.Tensor, zero_point: torch.Tensor, axis: int | None = None):
        super().__init__()
        self.register_buffer("scale", scale.detach().clone().to(dtype=torch.float32))
        self.register_buffer("zero_point", zero_point.detach().clone())
        self.axis = axis
        self.qmin, self.qmax, self.dtype_name = _qrange_from_zero_point(cast(torch.Tensor, self.zero_point))

    def forward(self, x: torch.Tensor, scale=None, zero_point=None) -> torch.Tensor:
        scale_buf = cast(torch.Tensor, self.scale)
        zp_buf = cast(torch.Tensor, self.zero_point)
        s = _reshape_axis_param(scale_buf.to(device=x.device, dtype=x.dtype), x, self.axis)
        zp = _reshape_axis_param(zp_buf.to(device=x.device, dtype=x.dtype), x, self.axis)
        return s * torch.clamp(torch.round(x / s), min=float(self.qmin) - zp, max=float(self.qmax) - zp)


class OnnxDequantizeLinear(nn.Module, OnnxToTorchModule):
    def __init__(self, scale: torch.Tensor, zero_point: torch.Tensor, axis: int | None = None):
        super().__init__()
        self.register_buffer("scale", scale.detach().clone().to(dtype=torch.float32))
        self.register_buffer("zero_point", zero_point.detach().clone())
        self.axis = axis
        self.qmin, self.qmax, self.dtype_name = _qrange_from_zero_point(cast(torch.Tensor, self.zero_point))

    def forward(self, q: torch.Tensor, scale=None, zero_point=None) -> torch.Tensor:
        dtype = torch.float32 if not q.is_floating_point() else q.dtype
        qf = q.to(dtype=dtype)
        scale_buf = cast(torch.Tensor, self.scale)
        zp_buf = cast(torch.Tensor, self.zero_point)
        s = _reshape_axis_param(scale_buf.to(device=q.device, dtype=dtype), qf, self.axis)
        zp = _reshape_axis_param(zp_buf.to(device=q.device, dtype=dtype), qf, self.axis)
        return s * (qf - zp)


def _axis(node: OnnxNode) -> int | None:
    raw = node.attributes.get("axis")
    return None if raw is None else int(raw)


def _const_tensor(name: str, graph: OnnxGraph) -> torch.Tensor:
    value = get_const_value(name, graph)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value


@add_converter(operation_type="QuantizeLinear", version=10)
@add_converter(operation_type="QuantizeLinear", version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    scale = _const_tensor(node.input_values[1], graph)
    zero_point = _const_tensor(node.input_values[2], graph)
    return OperationConverterResult(
        torch_module=OnnxQuantizeLinear(scale=scale, zero_point=zero_point, axis=_axis(node)),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )


@add_converter(operation_type="DequantizeLinear", version=10)
@add_converter(operation_type="DequantizeLinear", version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    scale = _const_tensor(node.input_values[1], graph)
    zero_point = _const_tensor(node.input_values[2], graph)
    return OperationConverterResult(
        torch_module=OnnxDequantizeLinear(scale=scale, zero_point=zero_point, axis=_axis(node)),
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
