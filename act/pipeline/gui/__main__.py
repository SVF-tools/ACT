"""Self-check of ACT's GUI support (``python -m act.pipeline.gui``; run in CI).

Runs ``verify_once`` in every domain under an observe sink and drives one
ACTFuzzer through its whole lifecycle with a FuzzControl, asserting the records
a GUI reads and that observation leaves verification and fuzzing results unchanged.
"""
from __future__ import annotations

import random
import tempfile
import threading
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from act.back_end.net_factory import NetFactory, _lt_assert_range, _lt_input
from act.back_end.solver.solver_dual import DualSolver
from act.back_end.transfer_functions import set_solver_mode, set_transfer_function_mode
from act.back_end.verifier import verify_once
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.verifiable_model import InputLayer, VerifiableModel
from act.pipeline.fuzzing.actfuzzer import ACTFuzzer, FuzzingConfig
from act.pipeline.gui import verifier_gui as obs
from act.pipeline.gui.fuzzer_gui import (
    FuzzConfigChange, FuzzControl, FuzzEnd, FuzzStart, FuzzStep, fuzzing_options,
)
from act.pipeline.verification.act2torch import ACTToTorch
from act.util.device_manager import initialize_device


class Recorder(list[object]):
    wants_relaxation = True

    def __call__(self, record) -> None:
        self.append(record)


def relu_net(factory: NetFactory, bound: float):
    """Active, inactive, unstable, and zero ReLU neurons; outputs must stay in [-bound, bound]."""
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


def run(net, domain: str, sink=None, **kwargs):
    set_transfer_function_mode("hybridz" if domain == "hybridz" else "interval")
    set_solver_mode(domain if domain in ("dual", "hybridz") else None)
    with nullcontext() if sink is None else obs.observe(sink):
        return [r.status for r in verify_once(net, **kwargs)]


def check_observe(factory: NetFactory) -> None:
    net = relu_net(factory, 10.0)
    for domain, form, kwargs in (("interval", "box", {}), ("hybridz", None, {}),
                                 ("dual", "linear", {"dual_iters": 2})):
        sink = Recorder()
        assert run(net, domain, sink, **kwargs) == run(net, domain, **kwargs), domain
        kinds = {type(r).__name__ for r in sink}
        assert {"PhaseEvent", "LayerBounds", "Relaxation"} <= kinds, (domain, kinds)
        relaxations = [r for r in sink if isinstance(r, obs.Relaxation)]
        assert all(r.status.dtype == torch.int8 for r in relaxations)
        assert form is None or any(r.form == form for r in relaxations), domain
        if domain == "dual":
            backward = [r for r in sink if isinstance(r, obs.DualBackward)]
            assert backward and all(r.alpha_optimized for r in backward)
            assert any(isinstance(r, obs.SpecBound) for r in sink)

    # The chunked dual path reports the same per-layer alpha records.
    set_transfer_function_mode("interval")
    sink = Recorder()
    with obs.observe(sink):
        DualSolver(n_iters=1).evaluate_spec(net, _range_spec(10.0), chunk_size=1)
    assert any(isinstance(r, obs.DualBackward) for r in sink)

    def failing(record) -> None:
        raise RuntimeError("sink failure")
    with obs.observe(failing):
        run(net, "interval")
        assert not obs.enabled, "a failing sink is detached"
    assert not obs.enabled and not obs.wants_relaxation()


def _range_spec(bound: float):
    from act.front_end.specs import OutKind, OutputSpec
    return OutputSpec(kind=OutKind.RANGE, lb=torch.full((2,), -bound), ub=torch.full((2,), bound))


def fuzzer_for(net, control: FuzzControl | None = None, max_iterations: int = 40) -> ACTFuzzer:
    model = ACTToTorch(net).run()
    rows = next(c for c in model.children() if isinstance(c, InputLayer)).input_tensor
    seeds = [LabeledInputTensor(tensor=rows[i:i + 1].clone(), label=None) for i in range(rows.shape[0])]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    cfg = FuzzingConfig.from_yaml(max_iterations=max_iterations, timeout_seconds=120.0,
                                  save_counterexamples=False, output_dir=Path(tempfile.mkdtemp()),
                                  report_interval=10_000, verbose=0, trace_level=0)
    return ACTFuzzer(wrapped_model=model, initial_seeds=seeds, config=cfg, control=control)


def check_fuzzing(factory: NetFactory) -> None:
    options = {o.option.name: o for o in fuzzing_options()}
    assert options["perturb_scale"].option.live and not options["coverage_strategy"].option.live
    net = relu_net(factory, 0.5)
    VerifiableModel.set_strict_mode(False)

    plain = fuzzer_for(net).fuzz()
    sink = Recorder()
    with obs.observe(sink):
        observed = fuzzer_for(net).fuzz()
    assert [ce.input.tolist() for ce in observed.counterexamples] == [ce.input.tolist() for ce in plain.counterexamples]
    steps = [r for r in sink if isinstance(r, FuzzStep)]
    start, end = sink[0], sink[-1]
    assert isinstance(start, FuzzStart) and isinstance(end, FuzzEnd)
    assert end.stop_reason == "max_iterations" and steps[-1].iteration == observed.total_iterations
    assert sum(len(s.counterexamples) for s in steps) == len(observed.counterexamples)

    for bad in ({"coverage_strategy": "BestInputCov"}, {"perturb_scale": 2.0}, {"mutation_weights": {"x": 1}},
                {"max_iterations": 0}, {"stop_on_first_violation": "yes"}, {"pgd_restarts": 1.5}):
        try:
            FuzzControl().update(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")

    # Live update after batch 3, then a paused update applied without running a batch, then stop.
    control, sink = FuzzControl(), Recorder()
    paused, applied = threading.Event(), threading.Event()
    batches: list[FuzzStep] = []

    def drive(record) -> None:
        sink.append(record)
        if isinstance(record, FuzzStep):
            batches.append(record)
            if len(batches) == 3:
                control.update(mutation_weights={"random": 1.0}, perturb_scale=0.3,
                               seed_selection_strategy="random", pgd_restarts=2)
            if len(batches) == 6:
                control.pause()
                paused.set()
        if isinstance(record, FuzzConfigChange) and paused.is_set():
            applied.set()
    fuzzer = fuzzer_for(net, control, max_iterations=10_000)
    result: dict[str, object] = {}
    worker = threading.Thread(target=lambda: result.update(report=_observed(fuzzer, drive)))
    worker.start()
    assert paused.wait(60)
    control.update(perturb_mode="adaptive_perdim")
    assert applied.wait(60) and control.paused
    batches_while_paused = len(batches)
    control.resume()
    control.request_stop()
    worker.join(60)
    changes = [r for r in sink if isinstance(r, FuzzConfigChange)]
    assert [c.iteration for c in changes] == [batches[2].iteration, batches[5].iteration]
    assert not any(c.rejected for c in changes) and batches_while_paused == 6
    assert fuzzer.stop_reason == "stopped" and result["report"] is not None
    assert {s.strategy for s in batches[3:]} <= {"random"}
    assert fuzzer.mutation_engine.perturb_mode == "adaptive_perdim" and fuzzer.seed_corpus.strategy == "random"
    try:
        fuzzer.update_config(activation_threshold=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-live field changed")


def _observed(fuzzer: ACTFuzzer, sink):
    with obs.observe(sink):
        return fuzzer.fuzz()


if __name__ == "__main__":
    initialize_device("cpu", "float64")
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        factory = NetFactory({"common": {"output_dir": tmp, "base_seed": 42, "num_instances": 1,
                                         "name_prefix": "gui", "write_manifest": False}})
        try:
            check_observe(factory)
            check_fuzzing(factory)
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")
    print("act.pipeline.gui self-check passed")
