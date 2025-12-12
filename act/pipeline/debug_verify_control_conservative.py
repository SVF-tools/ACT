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
    # 0. 基本设置
    # ------------------------------------------------------------------
    logging.basicConfig(level=logging.INFO)

    net_name = "control_conservative"
    print(f"[INFO] Loading ACT net: {net_name}")

    factory = ModelFactory()
    net = factory.get_act_net(net_name)

    # ------------------------------------------------------------------
    # 1. 用 ACT + Gurobi 做一次完整验证
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
    # 2. STEP A：用 PyTorch 真模型检查“中心点”是不是反例
    # ------------------------------------------------------------------
    print("\n===== STEP A: check center point with real PyTorch model =====")

    # 2.1 从 INPUT_SPEC 里拿 seed_bounds，并算出中心点
    spec_layers = gather_input_spec_layers(net)
    seed_bounds = seed_from_input_specs(spec_layers)

    lb = seed_bounds.lb.flatten()
    ub = seed_bounds.ub.flatten()

    # 优先用 LINF_BALL 的 center，如果有的话
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
        # 退一步：直接用 seed_bounds 的中点
        center = 0.5 * (lb + ub)

    print(f"[STEP A] seed_bounds lb[min,max] = [{lb.min().item():.4f}, {lb.max().item():.4f}]")
    print(f"[STEP A] seed_bounds ub[min,max] = [{ub.min().item():.4f}, {ub.max().item():.4f}]")
    print(f"[STEP A] center range            = [{center.min().item():.4f}, {center.max().item():.4f}]")

    # ------------------------------------------------------------------
    # 2.2 拿 PyTorch 模型（VerifiableModel wrapper）
    # ------------------------------------------------------------------
    torch_model = factory.create_model(net_name, load_weights=True)
    torch_model.eval()

    # 用 INPUT 层的 shape 把 center reshape 成模型输入形状
    inp_layer = next(L for L in net.layers if L.kind == "INPUT")
    shape = inp_layer.meta.get("shape") or [center.numel()]  # 例如 [8] / [1, 8] 等
    x = center.view(1, *shape)  # batch=1

    with torch.no_grad():
        out = torch_model(x)
        # VerifiableModel 情况：返回 dict，里面有 'output'
        if isinstance(out, dict):
            y = out["output"].view(-1)
        else:
            # 兼容 plain nn.Module
            y = out.view(-1)

    y_np = y.cpu().numpy()
    print(f"[STEP A] center output y.shape = {y_np.shape}")
    print(f"[STEP A] center output y       = {y_np}")

    # ------------------------------------------------------------------
    # 2.3 读取 ASSERT 层，区分“分类性质” vs “一般控制/安全性质”
    # ------------------------------------------------------------------
    assert_layer = get_assert_layer(net)
    kind = assert_layer.meta.get("kind", assert_layer.kind)
    print(f"[STEP A] assert layer kind     = {kind}")
    print(f"[STEP A] assert layer meta     = {assert_layer.meta}")
    print(f"[STEP A] assert layer params   = {list(assert_layer.params.keys())}")

    if "y_true" in assert_layer.meta:
        # ==========================
        # 情况 1：分类性质（如 TOP1_ROBUST）
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

        # 也用 ACT 的 check_violation_at_point 再确认一遍
        center_np = center.cpu().numpy()
        violated = check_violation_at_point(net, center_np, assert_layer)
        print(f"[STEP A] check_violation_at_point(center) = {violated}")

    else:
        # ==========================
        # 情况 2：一般控制/安全性质
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
    # 3. STEP D：再跑一次 check_violation_at_point，当成 sanity check
    # ------------------------------------------------------------------
    print("\n===== STEP D: ACT check_violation_at_point (sanity check) =====")
    center_np = center.cpu().numpy()
    violated_act = check_violation_at_point(net, center_np, assert_layer)
    print(f"[STEP D] check_violation_at_point(center) = {violated_act}")


if __name__ == "__main__":
    main()
