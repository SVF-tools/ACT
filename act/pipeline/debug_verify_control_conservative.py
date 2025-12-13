#!/usr/bin/env python3
from __future__ import annotations

import logging
import numpy as np
import torch

from act.pipeline.verification.model_factory import ModelFactory
from act.back_end.solver.solver_gurobi import GurobiSolver
from act.back_end.verifier import (
    verify_once,
    gather_input_spec_layers,
    seed_from_input_specs,
    get_assert_layer,
    check_violation_at_point,
)
from act.front_end.specs import InKind, OutKind


def main():
    # ------------------------------------------------------------------
    # 0. Basic setup
    # ------------------------------------------------------------------
    logging.basicConfig(level=logging.INFO)

    net_name = "control_conservative"
    print(f"[INFO] Loading ACT net: {net_name}")

    factory = ModelFactory()
    net = factory.get_act_net(net_name)

    # ------------------------------------------------------------------
    # 1. Run full verification with ACT + Gurobi
    # ------------------------------------------------------------------
    solver = GurobiSolver()
    res = verify_once(net, solver)

    print("====================================================")
    print(f"net name        : {net_name}")
    print(f"VerifStatus     : {res.status}")        # CERTIFIED / FALSIFIED / UNKNOWN
    print(f"SolveStatus(raw): {res.stats.get('status')}")
    print(f"ncons           : {res.stats.get('ncons')}")
    print(f"violation_var   : {res.stats.get('violation_var', None)}")
    print(f"stats dict      : {res.stats}")
    print("====================================================")

    if solver.m is not None:
        print(f"Gurobi m.Status : {solver.m.Status}")
        print(f"Gurobi SolCount : {getattr(solver.m, 'SolCount', 0)}")

    # ------------------------------------------------------------------
    # 2. STEP A: use real PyTorch model to check if the center is a counterexample
    # ------------------------------------------------------------------
    print("\n===== STEP A: check center point with real PyTorch model =====")

    # 2.1 Get seed_bounds from INPUT_SPEC and compute the center
    spec_layers = gather_input_spec_layers(net)
    seed_bounds = seed_from_input_specs(spec_layers)

    lb = seed_bounds.lb.flatten()
    ub = seed_bounds.ub.flatten()

    # Prefer LINF_BALL center when present
    center = None
    for L in spec_layers:
        k = L.meta.get("kind")
        if k == InKind.LINF_BALL:
            if "center" in L.params:
                center = L.params["center"].detach().clone().flatten()
            elif "lb" in L.params and "ub" in L.params:
                center = 0.5 * (L.params["lb"].flatten() + L.params["ub"].flatten())
            break

    if center is None:
        # Fallback: midpoint of seed_bounds
        center = 0.5 * (lb + ub)

    print(f"[STEP A] seed_bounds lb[min,max] = [{lb.min().item():.4f}, {lb.max().item():.4f}]")
    print(f"[STEP A] seed_bounds ub[min,max] = [{ub.min().item():.4f}, {ub.max().item():.4f}]")
    print(f"[STEP A] center range            = [{center.min().item():.4f}, {center.max().item():.4f}]")

    # ------------------------------------------------------------------
    # 2.2 Load PyTorch model (VerifiableModel wrapper)
    # ------------------------------------------------------------------
    torch_model = factory.create_model(net_name, load_weights=True)
    torch_model.eval()

    # Reshape center to model input shape using INPUT layer shape
    inp_layer = next(L for L in net.layers if L.kind == "INPUT")
    shape = inp_layer.meta.get("shape") or [center.numel()]  # e.g., [8] / [1, 8], etc.
    x = center.view(1, *shape)  # batch=1

    with torch.no_grad():
        out = torch_model(x)
        # VerifiableModel returns a dict containing 'output'
        if isinstance(out, dict):
            y = out["output"].view(-1)
        else:
            # Fallback for plain nn.Module
            y = out.view(-1)

    y_np = y.cpu().numpy()
    print(f"[STEP A] center output y.shape = {y_np.shape}")
    print(f"[STEP A] center output y       = {y_np}")

    # ------------------------------------------------------------------
    # 2.3 Read ASSERT layer; distinguish classification vs general control/safety properties
    # ------------------------------------------------------------------
    assert_layer = get_assert_layer(net)
    kind = assert_layer.meta.get("kind", assert_layer.kind)
    print(f"[STEP A] assert layer kind     = {kind}")
    print(f"[STEP A] assert layer meta     = {assert_layer.meta}")
    print(f"[STEP A] assert layer params   = {list(assert_layer.params.keys())}")

    if "y_true" in assert_layer.meta:
        # ==========================
        # Case 1: classification property (e.g., TOP1_ROBUST)
        # ==========================
        y_true = int(assert_layer.meta["y_true"])
        diffs = y_np - y_np[y_true]
        pred = int(diffs.argmax().item())

        print(f"[STEP A] y_true = {y_true}, argmax(pred) = {pred}")
        print(f"[STEP A] max_j (y[j] - y[t])  = {diffs.max():.6f}")
        print(f"[STEP A] diffs (y - y[t])[:10] = {diffs[:10]}")

        if pred != y_true:
            print("🚨 [STEP A] CENTER POINT IS A REAL COUNTEREXAMPLE (misclassified).")
        else:
            print("✅ [STEP A] center classified correctly; any CE must be other x in the ball.")

        # Also confirm using ACT check_violation_at_point
        center_np = center.cpu().numpy()
        violated = check_violation_at_point(net, center_np, assert_layer)
        print(f"[STEP A] check_violation_at_point(center) = {violated}")

    else:
        # ==========================
        # Case 2: general control/safety property
        # ==========================
        print("[STEP A] No 'y_true' in assert_layer.meta; treat spec as non-classification property.")
        center_np = center.cpu().numpy()
        violated = check_violation_at_point(net, center_np, assert_layer)
        print(f"[STEP A] generic spec check at center => violated = {violated}")

        if violated:
            print("🚨 [STEP A] CENTER POINT VIOLATES THE SPEC.")
        else:
            print("✅ [STEP A] center satisfies the spec.")

    # ------------------------------------------------------------------
    # 3. STEP D: run check_violation_at_point again as a sanity check
    # ------------------------------------------------------------------
    print("\n===== STEP D: ACT check_violation_at_point (sanity check) =====")
    center_np = center.cpu().numpy()
    violated_act = check_violation_at_point(net, center_np, assert_layer)
    print(f"[STEP D] check_violation_at_point(center) = {violated_act}")


if __name__ == "__main__":
    main()
