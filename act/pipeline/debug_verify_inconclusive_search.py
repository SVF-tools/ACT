#!/usr/bin/env python3
from __future__ import annotations

import logging
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB

from act.pipeline.verification.model_factory import ModelFactory
from act.back_end.solver.solver_gurobi import GurobiSolver
from act.back_end.verifier import (
    verify_once,
    gather_input_spec_layers,
    seed_from_input_specs,
    get_assert_layer,
    check_violation_at_point,
)
from act.front_end.specs import InKind


# Only verify control_* and reachability_* families
TARGET_NETS: List[str] = [
    "control_balanced",
    "control_strict",
    "reachability_loose",
    "reachability_moderate",
    "reachability_tight",
]


# ---------------------------------------------------------------
# Common helpers: extract input domain, center, and shape from ACT Net
# ---------------------------------------------------------------
def get_input_box_and_shape(net):
    spec_layers = gather_input_spec_layers(net)
    seed_bounds = seed_from_input_specs(spec_layers)

    lb = seed_bounds.lb.flatten()
    ub = seed_bounds.ub.flatten()

    # Prefer LINF_BALL center when available
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
        center = 0.5 * (lb + ub)

    print(f"[INFO] seed_bounds lb[min,max] = [{lb.min().item():.4f}, {lb.max().item():.4f}]")
    print(f"[INFO] seed_bounds ub[min,max] = [{ub.min().item():.4f}, {ub.max().item():.4f}]")
    print(f"[INFO] center range            = [{center.min().item():.4f}, {center.max().item():.4f}]")

    # meta.shape often includes a leading "1"; strip it
    inp_layer = next(L for L in net.layers if L.kind == "INPUT")
    shape_meta = list(inp_layer.meta.get("shape") or [center.numel()])

    if len(shape_meta) > 1 and shape_meta[0] == 1:
        logical_shape = shape_meta[1:]
    else:
        logical_shape = shape_meta

    return lb, ub, center, logical_shape


def build_torch_model(factory: ModelFactory, net_name: str) -> torch.nn.Module:
    model = factory.create_model(net_name, load_weights=True)
    model.eval()
    return model


# ---------------------------------------------------------------
# Unified margin computation for different ASSERT kinds
# ---------------------------------------------------------------
def eval_margin_and_output(
    torch_model: torch.nn.Module,
    x: torch.Tensor,
    assert_layer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        out = torch_model(x)
        if isinstance(out, dict):
            y = out["output"]
        else:
            y = out
        y = y.view(y.shape[0], -1)  # [B, out_dim]

    kind = assert_layer.meta.get("kind", assert_layer.kind)

    if kind == "TOP1_ROBUST":
        y_true = int(assert_layer.meta["y_true"])
        diffs = y - y[:, [y_true]]
        diffs[:, y_true] = float("-inf")
        margin, _ = diffs.max(dim=1)

    elif kind == "MARGIN_ROBUST":
        y_true = int(assert_layer.meta["y_true"])
        required_margin = float(assert_layer.meta["margin"])
        diffs = y - y[:, [y_true]]
        diffs[:, y_true] = float("-inf")
        comp_max, _ = diffs.max(dim=1)
        margin = comp_max - required_margin

    elif kind == "LINEAR_LE":
        c_list = assert_layer.params["c"]
        d = float(assert_layer.meta["d"])
        c = torch.tensor(c_list, dtype=y.dtype, device=y.device).view(-1, 1)
        score = (y @ c).view(-1)
        margin = score - d

    elif kind == "RANGE":
        lb_list = assert_layer.params["lb"]
        ub_list = assert_layer.params["ub"]
        lb_vec = torch.tensor(lb_list, dtype=y.dtype, device=y.device).view(1, -1)
        ub_vec = torch.tensor(ub_list, dtype=y.dtype, device=y.device).view(1, -1)
        viol_up, _ = (y - ub_vec).max(dim=1)
        viol_low, _ = (lb_vec - y).max(dim=1)
        margin = torch.max(viol_up, viol_low)

    else:
        raise NotImplementedError(f"Unsupported ASSERT kind: {kind}")

    return margin, y


# ---------------------------------------------------------------
# Random pointwise sampling — sanity check only, not exhaustive
# ---------------------------------------------------------------
def random_search_pointwise(
    torch_model: torch.nn.Module,
    lb: torch.Tensor,
    ub: torch.Tensor,
    shape,
    assert_layer,
    n_samples: int = 2000,
):
    first_param = next(torch_model.parameters())
    device = first_param.device
    dtype = first_param.dtype

    dim = lb.numel()
    lb_t = lb.to(device=device, dtype=dtype)
    ub_t = ub.to(device=device, dtype=dtype)

    best_margin = -1e9
    best_x = None
    best_y = None

    for i in range(n_samples):
        z = torch.rand(dim, device=device, dtype=dtype)
        x_flat = lb_t + (ub_t - lb_t) * z
        x = x_flat.view(1, *shape)

        margin, y = eval_margin_and_output(torch_model, x, assert_layer)
        m_val = float(margin.item())

        if m_val > best_margin:
            best_margin = m_val
            best_x = x.detach().cpu().numpy()[0]
            best_y = y.detach().cpu().numpy()[0]

        if (i + 1) % 200 == 0:
            print(
                f"[RANDOM] sample {i+1}/{n_samples}, "
                f"current best margin = {best_margin:.6f}"
            )

    return best_margin, best_x, best_y


# ---------------------------------------------------------------
# PGD gradient-ascent search — heuristic only
# ---------------------------------------------------------------
def pgd_search(
    torch_model: torch.nn.Module,
    lb: torch.Tensor,
    ub: torch.Tensor,
    shape,
    assert_layer,
    n_restarts: int = 10,
    iters: int = 100,
    step_size: float = 0.05,
):
    first_param = next(torch_model.parameters())
    device = first_param.device
    dtype = first_param.dtype

    dim = lb.numel()
    lb_t = lb.view(1, -1).to(device=device, dtype=dtype)
    ub_t = ub.view(1, -1).to(device=device, dtype=dtype)

    best_margin = -1e9
    best_x = None
    best_y = None

    for r in range(n_restarts):
        z0 = torch.rand(1, dim, device=device, dtype=dtype)
        x_flat = lb_t + (ub_t - lb_t) * z0
        x_flat.requires_grad_(True)

        optimizer = torch.optim.SGD([x_flat], lr=step_size)

        for _ in range(iters):
            optimizer.zero_grad()

            x = x_flat.view(1, *shape)
            out = torch_model(x)
            if isinstance(out, dict):
                y = out["output"]
            else:
                y = out
            y = y.view(1, -1)

            kind = assert_layer.meta.get("kind", assert_layer.kind)
            if kind == "TOP1_ROBUST":
                y_true = int(assert_layer.meta["y_true"])
                diffs = y - y[:, [y_true]]
                diffs[:, y_true] = float("-inf")
                margin = torch.max(diffs, dim=1).values
            elif kind == "MARGIN_ROBUST":
                y_true = int(assert_layer.meta["y_true"])
                req_m = float(assert_layer.meta["margin"])
                diffs = y - y[:, [y_true]]
                diffs[:, y_true] = float("-inf")
                comp_max = torch.max(diffs, dim=1).values
                margin = comp_max - req_m
            elif kind == "LINEAR_LE":
                c_list = assert_layer.params["c"]
                d = float(assert_layer.meta["d"])
                c = torch.tensor(c_list, dtype=y.dtype, device=y.device).view(-1, 1)
                score = (y @ c).view(-1)
                margin = score - d
            elif kind == "RANGE":
                lb_list = assert_layer.params["lb"]
                ub_list = assert_layer.params["ub"]
                lb_vec = torch.tensor(lb_list, dtype=y.dtype, device=y.device).view(1, -1)
                ub_vec = torch.tensor(ub_list, dtype=y.dtype, device=y.device).view(1, -1)
                viol_up = torch.max(y - ub_vec, dim=1).values
                viol_low = torch.max(lb_vec - y, dim=1).values
                margin = torch.max(viol_up, viol_low)
            else:
                raise NotImplementedError(f"Unsupported ASSERT kind: {kind}")

            loss = -margin.mean()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x_flat.data = torch.max(torch.min(x_flat, ub_t), lb_t)

        with torch.no_grad():
            x = x_flat.view(1, *shape)
            margin_final, y_final = eval_margin_and_output(torch_model, x, assert_layer)
            m_val = float(margin_final.item())
            if m_val > best_margin:
                best_margin = m_val
                best_x = x.detach().cpu().numpy()[0]
                best_y = y_final.detach().cpu().numpy()[0]

        print(
            f"[PGD] restart {r+1}/{n_restarts}, "
            f"margin = {m_val:.6f}, best so far = {best_margin:.6f}"
        )

    return best_margin, best_x, best_y


# ---------------------------------------------------------------
# Interval propagation: bounds for a single Linear layer
# ---------------------------------------------------------------
def compute_pre_bounds_box(
    W: np.ndarray, b: np.ndarray,
    lb_x: np.ndarray, ub_x: np.ndarray,
):
    """
    Compute interval bounds for one linear layer given box constraints
    x_j ∈ [lb_x[j], ub_x[j]]:
      a = W x + b
    """
    in_dim = W.shape[1]
    assert in_dim == lb_x.shape[0] == ub_x.shape[0]

    n_hid = W.shape[0]
    lb_a = np.zeros(n_hid, dtype=float)
    ub_a = np.zeros(n_hid, dtype=float)

    for i in range(n_hid):
        row = W[i, :]
        contrib_lb = 0.0
        contrib_ub = 0.0
        for j in range(in_dim):
            w = row[j]
            if w >= 0:
                contrib_lb += w * lb_x[j]
                contrib_ub += w * ub_x[j]
            else:
                contrib_lb += w * ub_x[j]
                contrib_ub += w * lb_x[j]
        lb_a[i] = b[i] + contrib_lb
        ub_a[i] = b[i] + contrib_ub

    return lb_a, ub_a


# ---------------------------------------------------------------
# Utility: extract Linear + ReLU sequences from torch_model
# ---------------------------------------------------------------
class LinearBlock:
    def __init__(self, W: np.ndarray, b: np.ndarray, has_relu: bool):
        self.W = W  # shape: [out_dim, in_dim]
        self.b = b  # shape: [out_dim]
        self.has_relu = has_relu  # Whether this Linear is immediately followed by a ReLU


def extract_linear_relu_blocks(torch_model: nn.Module) -> List[LinearBlock]:
    """
    Extract an execution-ordered sequence of [Linear (+ReLU?)] blocks from torch_model.
    Assumes a feed-forward Linear/ReLU network (control_*/reachability_* fit this).
    """
    modules: List[nn.Module] = []
    # modules() lists all modules (including top-level); keep only Linear/ReLU
    for m in torch_model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.ReLU):
            modules.append(m)

    blocks: List[LinearBlock] = []
    i = 0
    while i < len(modules):
        m = modules[i]
        if isinstance(m, nn.Linear):
            W = m.weight.detach().cpu().numpy()
            b = m.bias.detach().cpu().numpy()
            has_relu = False
            if i + 1 < len(modules) and isinstance(modules[i + 1], nn.ReLU):
                has_relu = True
                i += 1  # Consume the following ReLU
            blocks.append(LinearBlock(W, b, has_relu))
            i += 1
        else:
            # Encountering a bare ReLU (not preceded by Linear) should not happen; skip defensively
            i += 1

    if not blocks:
        raise RuntimeError("extract_linear_relu_blocks: no Linear layers found in model")

    return blocks


# ---------------------------------------------------------------
# Exact MILP: full-network MILP for any number of Linear + ReLU layers
# ---------------------------------------------------------------
def run_exact_milp_for_two_layer_mlp(
    net,
    torch_model: nn.Module,
    lb: torch.Tensor,
    ub: torch.Tensor,
    assert_layer,
) -> Optional[float]:
    """
    Build a fully concrete MILP with gurobipy to maximize violation margin
    under the given box constraints.

    No longer limited to “two Linear layers”; supports all Linear + ReLU
    feed-forward networks in torch_model.

    Currently focused on:
      - LINEAR_LE (control_*)
      - RANGE      (reachability_*)

    Returns:
      best_margin: float or None (if solving fails)
    """
    kind = assert_layer.meta.get("kind", assert_layer.kind)
    if kind not in ("LINEAR_LE", "RANGE"):
        raise RuntimeError(f"MILP currently supports LINEAR_LE / RANGE, got {kind}")

    # 1) Extract ordered Linear (+ReLU?) blocks from torch_model
    blocks = extract_linear_relu_blocks(torch_model)
    print(f"[MILP] extracted {len(blocks)} Linear blocks from torch_model")

    lb_x = lb.cpu().numpy().astype(float)
    ub_x = ub.cpu().numpy().astype(float)

    # 2) Use simple interval propagation (IBP) to get each layer's pre-activation bounds
    pre_lbs: List[np.ndarray] = []
    pre_ubs: List[np.ndarray] = []

    cur_lb = lb_x.copy()
    cur_ub = ub_x.copy()

    for idx, blk in enumerate(blocks):
        W, b = blk.W, blk.b
        lb_a, ub_a = compute_pre_bounds_box(W, b, cur_lb, cur_ub)
        pre_lbs.append(lb_a)
        pre_ubs.append(ub_a)

        # Post-activation bounds used by the next layer
        if blk.has_relu:
            cur_lb = np.maximum(lb_a, 0.0)
            cur_ub = np.maximum(ub_a, 0.0)
        else:
            cur_lb, cur_ub = lb_a, ub_a

    print("[MILP] final IBP output bounds (y):")
    print("       lb_y[min,max] =", float(cur_lb.min()), float(cur_lb.max()))
    print("       ub_y[min,max] =", float(cur_ub.min()), float(cur_ub.max()))

    # 3) Build MILP: input -> series of Linear(+ReLU?) -> output
    m = gp.Model(f"milp_{kind.lower()}")

    n_in = lb_x.shape[0]
    # Post-activation vector for each layer x_layers[k]
    x_layers: List[gp.tupledict] = []

    # Input layer variables
    x0 = m.addVars(n_in, lb=lb_x, ub=ub_x, name="x_0")
    x_layers.append(x0)

    # For each block, add pre-activation + ReLU + next-layer variables
    a_vars: List[gp.tupledict] = []  # pre-activation per layer
    z_vars: List[Optional[gp.tupledict]] = []

    for l_idx, blk in enumerate(blocks):
        W, b = blk.W, blk.b
        lb_a = pre_lbs[l_idx]
        ub_a = pre_ubs[l_idx]

        in_dim = W.shape[1]
        out_dim = W.shape[0]

        x_prev = x_layers[-1]

        # pre-activation: a_l
        a_l = m.addVars(out_dim, lb=lb_a, ub=ub_a, name=f"a_{l_idx}")
        a_vars.append(a_l)

        # a_l = W x_prev + b
        for i in range(out_dim):
            expr = gp.LinExpr()
            for j in range(in_dim):
                expr += W[i, j] * x_prev[j]
            expr += b[i]
            m.addConstr(a_l[i] == expr, name=f"a_def_{l_idx}_{i}")

        if blk.has_relu:
            # post-activation: h_l = ReLU(a_l)
            h_l = m.addVars(out_dim, lb=0.0, name=f"h_{l_idx}")
            z_l = m.addVars(out_dim, vtype=GRB.BINARY, name=f"z_{l_idx}")
            z_vars.append(z_l)

            for i in range(out_dim):
                li = float(lb_a[i])
                ui = float(ub_a[i])

                m.addConstr(h_l[i] >= 0.0, name=f"relu_ge0_{l_idx}_{i}")
                m.addConstr(h_l[i] >= a_l[i], name=f"relu_gea_{l_idx}_{i}")
                m.addConstr(h_l[i] <= a_l[i] - li * (1 - z_l[i]), name=f"relu_ub1_{l_idx}_{i}")
                m.addConstr(h_l[i] <= ui * z_l[i],            name=f"relu_ub2_{l_idx}_{i}")

            x_layers.append(h_l)
        else:
            # No activation: post-activation equals a_l
            z_vars.append(None)
            x_layers.append(a_l)

    # Final layer output y
    y = x_layers[-1]
    n_out = len(y)

    # 4) On this exact network, maximize assertion violation margin

    best_margin = -1e9
    best_desc = None

    if kind == "LINEAR_LE":
        # Property: c^T y <= d
        c_list = assert_layer.params["c"]
        d = float(assert_layer.meta["d"])
        c_arr = np.array(c_list, dtype=float)

        score_expr = gp.LinExpr()
        for i in range(n_out):
            score_expr += c_arr[i] * y[i]

        obj_expr = score_expr - d  # violation margin

        m.setObjective(obj_expr, GRB.MAXIMIZE)
        m.Params.OutputFlag = 1
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            best_margin = float(m.ObjVal)
            best_desc = "LINEAR_LE: max (c^T y - d)"
        else:
            print(f"[MILP] status = {m.Status} (not OPTIMAL; LINEAR_LE result unreliable)")
            return None

    elif kind == "RANGE":
        # Property: lb[i] <= y[i] <= ub[i]
        lb_list = np.array(assert_layer.params["lb"], dtype=float)
        ub_list = np.array(assert_layer.params["ub"], dtype=float)

        for i in range(n_out):
            # Upper-bound violation: y[i] - ub[i]
            expr_up = y[i] - ub_list[i]
            m.setObjective(expr_up, GRB.MAXIMIZE)
            m.Params.OutputFlag = 0
            m.optimize()
            if m.Status == GRB.OPTIMAL:
                val_up = float(m.ObjVal)
                if val_up > best_margin:
                    best_margin = val_up
                    best_desc = f"RANGE: max (y[{i}] - ub[{i}])"
            else:
                print(f"[MILP] status = {m.Status} @upper i={i} (skip this direction)")

            # Lower-bound violation: lb[i] - y[i]
            expr_low = lb_list[i] - y[i]
            m.setObjective(expr_low, GRB.MAXIMIZE)
            m.optimize()
            if m.Status == GRB.OPTIMAL:
                val_low = float(m.ObjVal)
                if val_low > best_margin:
                    best_margin = val_low
                    best_desc = f"RANGE: max (lb[{i}] - y[{i}])"
            else:
                print(f"[MILP] status = {m.Status} @lower i={i} (skip this direction)")

        if best_desc is None:
            print("[MILP] All RANGE directions were non-OPTIMAL; result unreliable")
            return None

    # 5) Print results and optionally extract a CE
    print("=== EXACT MILP (full ReLU MLP) ===")
    print(f"[MILP] kind           = {kind}")
    print(f"[MILP] best objective = {best_margin:.6f}")
    if best_desc is not None:
        print(f"[MILP] objective src = {best_desc}")
    if best_margin > 0:
        print("[MILP] ==> Found a strict CE (on exact network within the box)")

        # Output a CE x* (optional)
        n_in = lb_x.shape[0]
        ce_x = np.array([x_layers[0][j].X for j in range(n_in)], dtype=float)
        print(f"[MILP] example CE input x* = {ce_x}")
    else:
        print("[MILP] ==> No CE under exact network + box constraints")

    return best_margin


# ---------------------------------------------------------------
# Full check for a single net (ACT + Search + MILP triple validation)
# ---------------------------------------------------------------
def analyze_single_net(factory: ModelFactory, net_name: str):
    print("\n" + "=" * 80)
    print(f"[NET] {net_name}")
    print("=" * 80)

    # 1. ACT + Gurobi abstract verification
    net = factory.get_act_net(net_name)
    solver = GurobiSolver()
    res = verify_once(net, solver)
    print(f"[ACT] VerifStatus     : {res.status}")
    print(f"[ACT] SolveStatus(raw): {res.stats.get('status')}")
    print(f"[ACT] ncons           : {res.stats.get('ncons')}")
    print(f"[ACT] violation_var   : {res.stats.get('violation_var', None)}")

    # 2. Input domain box + center + shape
    lb, ub, center, shape = get_input_box_and_shape(net)

    # 3. PyTorch model + ASSERT layer
    torch_model = build_torch_model(factory, net_name)
    assert_layer = get_assert_layer(net)
    kind = assert_layer.meta.get("kind", assert_layer.kind)
    print(f"[ASSERT] kind = {kind}, meta = {assert_layer.meta}")

    # 4. Check if the center point is a CE
    print("\n----- CHECK CENTER POINT -----")
    center_np = center.cpu().numpy()
    violated_center = check_violation_at_point(net, center_np, assert_layer)
    print(f"[CENTER] check_violation_at_point(center) = {violated_center}")

    # 5. Random search (auxiliary)
    print("\n----- RANDOM SEARCH -----")
    best_margin_rand, best_x_rand, best_y_rand = random_search_pointwise(
        torch_model, lb, ub, shape, assert_layer, n_samples=2000
    )
    print(f"[RANDOM] best margin = {best_margin_rand:.6f}")
    if best_y_rand is not None:
        print(f"[RANDOM] best y = {best_y_rand}")
    print()

    # 6. PGD search (auxiliary)
    print("----- PGD SEARCH -----")
    best_margin_pgd, best_x_pgd, best_y_pgd = pgd_search(
        torch_model, lb, ub, shape, assert_layer,
        n_restarts=10, iters=100, step_size=0.05,
    )
    print(f"[PGD] best margin = {best_margin_pgd:.6f}")
    if best_y_pgd is not None:
        print(f"[PGD] best y = {best_y_pgd}")

    # 7. Full-network exact MILP check (the definitive source)
    print("\n----- EXACT MILP (full ReLU MLP) -----")
    milp_margin = None
    try:
        milp_margin = run_exact_milp_for_two_layer_mlp(
            net, torch_model, lb, ub, assert_layer
        )
    except Exception as e:
        print(f"[MILP] error building MILP for {net_name}: {e}")
        milp_margin = None

    # 8. Triple check: ACT abstraction + numeric search + exact MILP
    print("\n----- TRIPLE CHECK (ACT + Search + MILP) -----")
    act_status = res.status  # CERTIFIED / FALSIFIED / UNKNOWN

    # Numeric CE: center or random or PGD with margin>0 counts as CE
    concrete_ce = (
        bool(violated_center)
        or best_margin_rand > 0.0
        or best_margin_pgd > 0.0
    )

    if milp_margin is None:
        milp_ce = None
    else:
        milp_ce = milp_margin > 0.0

    print(f"[TRIPLE] ACT status         = {act_status}")
    print(f"[TRIPLE] center violation   = {violated_center}")
    print(
        f"[TRIPLE] search CE?         = {concrete_ce} "
        f"(rand={best_margin_rand:.6f}, pgd={best_margin_pgd:.6f})"
    )
    if milp_margin is None:
        print("[TRIPLE] MILP status        = ERROR/UNKNOWN (build or solve failed)")
    else:
        print(
            f"[TRIPLE] MILP max violation = {milp_margin:.6f} "
            f"(CE? {milp_ce})"
        )

    # Empirical ground truth: use MILP as the gold standard
    if milp_ce is True:
        truth = "UNSAFE (MILP found a CE)"
    elif milp_ce is False:
        truth = "SAFE (MILP proved no CE inside the box)"
    else:
        # If MILP failed, fall back to Search (no longer 100%)
        if concrete_ce:
            truth = "UNSAFE? (Search found a suspected CE, but MILP unavailable)"
        else:
            truth = "UNKNOWN (MILP unavailable; Search alone is not 100% reliable)"

    # Consistency between abstraction result and MILP truth
    if truth.startswith("UNSAFE") and act_status == "FALSIFIED":
        verdict = "CONSISTENT (ACT correctly reports UNSAFE)"
    elif truth.startswith("SAFE") and act_status == "CERTIFIED":
        verdict = "CONSISTENT (ACT correctly reports SAFE)"
    elif act_status == "UNKNOWN":
        verdict = f"INCONCLUSIVE (ACT=UNKNOWN, MILP says {truth})"
    else:
        verdict = f"POSSIBLE BUG? (ACT={act_status}, MILP truth={truth})"

    print(f"[TRIPLE] ground truth (MILP-based) = {truth}")
    print(f"[TRIPLE] verdict                   = {verdict}")

    # 9. Summary
    print("\n----- SUMMARY FOR NET -----")
    print(f"[SUMMARY] net               = {net_name}")
    print(f"[SUMMARY] ACT VerifStatus   = {res.status}")
    print(f"[SUMMARY] ACT raw status    = {res.stats.get('status')}")
    print(f"[SUMMARY] Center violated?  = {violated_center}")
    print(f"[SUMMARY] Random max margin = {best_margin_rand:.6f}")
    print(f"[SUMMARY] PGD max margin    = {best_margin_pgd:.6f}")
    print(f"[SUMMARY] MILP max margin   = {milp_margin if milp_margin is not None else 'N/A'}")
    print(f"[SUMMARY] MILP truth        = {truth}")
    print(f"[SUMMARY] Triple verdict    = {verdict}")
    print("[SUMMARY] Notes:")
    print("  - MILP is a global optimization on the exact network + exact box.")
    print("  - MILP best_margin > 0  ⇔ a CE exists in the box.")
    print("  - MILP best_margin ≤ 0  ⇔ no CE exists in the box.")
    print("  - When MILP status is OPTIMAL, this is the definitive conclusion (up to solver precision).")


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO)
    factory = ModelFactory()

    for net_name in TARGET_NETS:
        try:
            analyze_single_net(factory, net_name)
        except Exception as e:
            print(f"[ERROR] net {net_name} failed: {e}")


if __name__ == "__main__":
    main()
