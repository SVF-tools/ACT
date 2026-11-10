"""Network and sink fixtures shared by GUI module self-checks."""

import torch

from act.pipeline.gui.act_gui import Record


class Recorder(list[object]):
    wants_relaxation = True

    def __call__(self, record: Record) -> None:
        self.append(record)


def make_relu_net(factory, bound: float):
    """Small ReLU network with active, inactive, unstable, and zero neurons."""
    from act.back_end.net_factory import _lt_assert_range, _lt_input

    return factory.create_network(f"gui_relu_{bound}", {"layers": _lt_input([2, 4], -1., 1.) + [
        {"kind": "DENSE", "params": {"in_features": 4, "out_features": 4,
                                     "weight": torch.diag(torch.tensor([1., 1., 1., 0.])),
                                     "bias": torch.tensor([2., -2., 0., 0.])}},
        {"kind": "RELU", "params": {}},
        {"kind": "DENSE", "params": {"in_features": 4, "out_features": 2,
                                     "weight": torch.tensor([[1., -1., 1., -1.], [-1., 1., -1., 1.]]),
                                     "bias": torch.zeros(2)}},
        _lt_assert_range([-bound, -bound], [bound, bound]),
    ]})


def make_factory(output_dir: str):
    from act.back_end.net_factory import NetFactory

    return NetFactory({"common": {"output_dir": output_dir, "base_seed": 42,
                                  "num_instances": 1, "name_prefix": "gui",
                                  "write_manifest": False}})


__all__ = ["Recorder", "make_relu_net", "make_factory"]
