#!/usr/bin/env python3
from __future__ import annotations

import logging
import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB

from act.pipeline.verification.model_factory import ModelFactory


def extract_two_linears(torch_model: torch.nn.Module):
    """
    从 VerifiableModel 里自动提取前两个 nn.Linear 层：
    control_conservative 是:  Linear(8->16) -> ReLU -> Linear(16->4)
    """
    linears = []
    for m in torch_model.modules():
        if isinstance(m, torch.nn.Linear):
            linears.append(m)
    if len(linears) < 2:
        raise RuntimeError(f"Expect at least 2 Linear layers, got {len(linears)}.")
    return linears[0], linears[1]


def compute_pre_bounds(W1: np.ndarray, b1: np.ndarray):
    """
    对于 x ∈ [-1,1]^8，计算 a1 = W1 x + b1 的简单区间上下界。

    l_i = b_i + sum_j W_ij * ( -1 if W_ij > 0 else 1 )
    u_i = b_i + sum_j W_ij * (  1 if W_ij > 0 else -1 )
    """
    in_dim = W1.shape[1]
    assert in_dim == 8, f"Expected input dim 8, got {in_dim}"

    lb_a1 = np.zeros(W1.shape[0])
    ub_a1 = np.zeros(W1.shape[0])

    for i in range(W1.shape[0]):
        row = W1[i, :]
        # x_j ∈ [-1,1]
        # 对每个 j 取能让 W_ij * x_j 最小/最大的位置
        contrib_lb = 0.0
        contrib_ub = 0.0
        for j in range(in_dim):
            w = row[j]
            if w >= 0:
                contrib_lb += w * (-1.0)
                contrib_ub += w * (+1.0)
            else:
                contrib_lb += w * (+1.0)
                contrib_ub += w * (-1.0)
        lb_a1[i] = b1[i] + contrib_lb
        ub_a1[i] = b1[i] + contrib_ub

    return lb_a1, ub_a1


def build_and_solve_milp(W1, b1, W2, b2, lb_a1, ub_a1, d=2.0):
    """
    使用 Gurobi 建 MILP：
      - 变量: x ∈ [-1,1]^8, a1 ∈ [lb_a1, ub_a1], h1 >=0, z ∈ {0,1}^16, y ∈ R^4
      - 约束: a1 = W1 x + b1
              ReLU 约束 (big-M)
              y = W2 h1 + b2
      - 目标: maximize sum(y_k)  (margin = sum(y_k) - d)
    """
    n_in = W1.shape[1]   # 8
    n_hid = W1.shape[0]  # 16
    n_out = W2.shape[0]  # 4

    m = gp.Model("control_conservative_milp")

    # 1. 输入变量 x ∈ [-1,1]^8
    x = m.addVars(n_in, lb=-1.0, ub=1.0, name="x")

    # 2. 第一层 pre-activation a1, post-activation h1, ReLU 二进制变量 z
    a1 = m.addVars(n_hid, lb=lb_a1, ub=ub_a1, name="a1")
    h1 = m.addVars(n_hid, lb=0.0, name="h1")
    z = m.addVars(n_hid, vtype=GRB.BINARY, name="z")

    # 3. 第二层输出 y ∈ R^4
    y = m.addVars(n_out, lb=-GRB.INFINITY, name="y")

    # -------------------------------
    # 约束 1: a1 = W1 x + b1
    # -------------------------------
    for i in range(n_hid):
        expr = gp.LinExpr()
        for j in range(n_in):
            expr += W1[i, j] * x[j]
        expr += b1[i]
        m.addConstr(a1[i] == expr, name=f"a1_def_{i}")

    # -------------------------------
    # 约束 2: ReLU 线性化
    #   h_i = max(0, a_i)
    #   使用 big-M:
    #     h_i ≥ 0
    #     h_i ≥ a_i
    #     h_i ≤ a_i - lb_a1[i]*(1 - z_i)
    #     h_i ≤ ub_a1[i]*z_i
    # -------------------------------
    for i in range(n_hid):
        li = float(lb_a1[i])
        ui = float(ub_a1[i])

        # ReLU 基本约束
        m.addConstr(h1[i] >= 0.0, name=f"relu_ge0_{i}")
        m.addConstr(h1[i] >= a1[i], name=f"relu_gea_{i}")
        # big-M 上界
        m.addConstr(h1[i] <= a1[i] - li * (1 - z[i]), name=f"relu_ub1_{i}")
        m.addConstr(h1[i] <= ui * z[i],               name=f"relu_ub2_{i}")

    # -------------------------------
    # 约束 3: y = W2 h1 + b2
    # -------------------------------
    for k in range(n_out):
        expr = gp.LinExpr()
        for i in range(n_hid):
            expr += W2[k, i] * h1[i]
        expr += b2[k]
        m.addConstr(y[k] == expr, name=f"y_def_{k}")

    # -------------------------------
    # 目标: maximize sum(y_k)
    # spec: sum(y_k) <= d  (这里 d=2.0)
    # margin = sum(y_k) - d
    # -------------------------------
    sum_y = gp.quicksum(y[k] for k in range(n_out))
    m.setObjective(sum_y, GRB.MAXIMIZE)

    m.Params.OutputFlag = 1
    m.optimize()

    status = m.Status
    if status == GRB.OPTIMAL:
        obj = m.ObjVal
        margin = obj - d
        print("=== MILP OPTIMAL ===")
        print(f"max sum(y)  = {obj:.6f}")
        print(f"margin      = sum(y) - {d} = {margin:.6f}")
        print("=> 如果 margin <= 0，则在线性约束下不存在违反 sum(y)<=2 的 CE。")
    elif status == GRB.INFEASIBLE:
        print("=== MILP INFEASIBLE ===")
        print("这说明在 [-1,1]^8 下居然没有任何可行点（一般不会发生，如果发生说明模型/约束写崩了）")
    else:
        print(f"=== MILP status = {status} ===")

    # 把最优解 x*, y* 打印出来（如果存在）
    if status == GRB.OPTIMAL:
        x_star = np.array([x[j].X for j in range(n_in)], dtype=float)
        y_star = np.array([y[k].X for k in range(n_out)], dtype=float)
        print("x* =", x_star)
        print("y* =", y_star)
        print("sum(y*) =", y_star.sum())
        return x_star, y_star, margin
    else:
        return None, None, None


def main():
    logging.basicConfig(level=logging.INFO)

    net_name = "control_conservative"
    print(f"[INFO] Loading PyTorch model for net: {net_name}")

    factory = ModelFactory()
    torch_model = factory.create_model(net_name, load_weights=True)
    torch_model.eval()

    # 提取两层全连接
    lin1, lin2 = extract_two_linears(torch_model)

    W1 = lin1.weight.detach().cpu().numpy()  # [16, 8]
    b1 = lin1.bias.detach().cpu().numpy()    # [16]
    W2 = lin2.weight.detach().cpu().numpy()  # [4, 16]
    b2 = lin2.bias.detach().cpu().numpy()    # [4]

    print("[INFO] W1.shape =", W1.shape, "b1.shape =", b1.shape)
    print("[INFO] W2.shape =", W2.shape, "b2.shape =", b2.shape)

    # 对第一层 pre-activation 做一个简单的区间 bound，用于 big-M
    lb_a1, ub_a1 = compute_pre_bounds(W1, b1)
    print("[INFO] pre-activation bounds (a1):")
    print("       lb_a1[min,max] =", float(lb_a1.min()), float(lb_a1.max()))
    print("       ub_a1[min,max] =", float(ub_a1.min()), float(ub_a1.max()))

    # 用 Gurobi 建 MILP 并求解 max sum(y)
    x_star, y_star, margin = build_and_solve_milp(W1, b1, W2, b2, lb_a1, ub_a1, d=2.0)

    # # 再把 x_star 用 PyTorch 模型跑一遍，做 sanity check
    # if x_star is not None:
    #     print("\n[CHECK] Evaluate x* with PyTorch VerifiableModel (包含 INPUT_SPEC 等模块)")

    #     # 根据你的 config: INPUT 的 meta.shape = [1, 8]
    #     shape = (1, 8)
    #     x_torch = torch.tensor(x_star, dtype=torch.float32).view(1, *shape)

    #     with torch.no_grad():
    #         out = torch_model(x_torch)
    #         if isinstance(out, dict):
    #             y_t = out["output"].view(-1)
    #         else:
    #             y_t = out.view(-1)
    #     y_np = y_t.cpu().numpy()
    #     print("[CHECK] y(PyTorch) =", y_np)
    #     print("[CHECK] sum(y(PyTorch)) =", float(y_np.sum()))
    #     print("[CHECK] margin(PyTorch) =", float(y_np.sum() - 2.0))
    
        # 再把 x_star 用 PyTorch 模型跑一遍，做 sanity check
    if x_star is not None:
        print("\n[CHECK] Evaluate x* with PyTorch VerifiableModel (包含 INPUT_SPEC 等模块)")

        # 用模型本身的 dtype / device，避免 Float vs Double 冲突
        first_param = next(torch_model.parameters())
        model_dtype = first_param.dtype
        model_device = first_param.device

        # 根据 config: INPUT meta.shape = [1, 8]
        shape = (1, 8)
        x_torch = torch.tensor(
            x_star,
            dtype=model_dtype,
            device=model_device,
        ).view(1, *shape)   # -> [1, 1, 8]

        with torch.no_grad():
            out = torch_model(x_torch)
            if isinstance(out, dict):
                y_t = out["output"].view(-1)
            else:
                y_t = out.view(-1)

        y_np = y_t.cpu().numpy()
        print("[CHECK] y(PyTorch) =", y_np)
        print("[CHECK] sum(y(PyTorch)) =", float(y_np.sum()))
        print("[CHECK] margin(PyTorch) =", float(y_np.sum() - 2.0))



if __name__ == "__main__":
    main()
