#===- act/back_end/interval_tf/tf_rnn.py - RNN Interval Transfer Func ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   RNN interval transfer functions shared by interval and HybridZ modes.

from collections.abc import Callable
from math import prod
from typing import Any

import torch

from act.back_end.core import Bounds, Con, ConSet, Fact, Layer
from act.back_end.utils import affine_bounds, four_corner_envelope, split_weight


State = tuple[torch.Tensor, ...]
Step = Callable[
    [torch.Tensor, torch.Tensor, State, dict[str, torch.Tensor | None], int, str | None],
    State,
]


def tf_lstm(L: Layer, Bin: Bounds) -> Fact:
    """Apply the interval LSTM transfer function."""
    return _recurrent_fact(L, Bin, "lstm", strict_num_layers=False, hybridz=False)


def tf_gru(L: Layer, Bin: Bounds) -> Fact:
    """Apply the interval GRU transfer function."""
    return _recurrent_fact(L, Bin, "gru", strict_num_layers=False, hybridz=False)


def tf_rnn(L: Layer, Bin: Bounds) -> Fact:
    """Apply the interval vanilla RNN transfer function."""
    return _recurrent_fact(L, Bin, "rnn", strict_num_layers=False, hybridz=False)


def tf_embedding(L: Layer, Bin: Bounds) -> Fact:
    """Apply the interval embedding transfer function."""
    return _embedding_fact(L, hybridz=False)


def tf_lstm_hybridz(L: Layer, Bin: Bounds) -> Fact:
    """Apply the HybridZ LSTM interval handler."""
    return _recurrent_fact(L, Bin, "lstm", strict_num_layers=True, hybridz=True)


def tf_gru_hybridz(L: Layer, Bin: Bounds) -> Fact:
    """Apply the HybridZ GRU interval handler."""
    return _recurrent_fact(L, Bin, "gru", strict_num_layers=True, hybridz=True)


def tf_rnn_hybridz(L: Layer, Bin: Bounds) -> Fact:
    """Apply the HybridZ vanilla RNN interval handler."""
    return _recurrent_fact(L, Bin, "rnn", strict_num_layers=True, hybridz=True)


def tf_embedding_hybridz(L: Layer, Bin: Bounds) -> Fact:
    """Apply the HybridZ embedding interval handler."""
    return _embedding_fact(L, hybridz=True)


def _recurrent_fact(
    L: Layer,
    Bin: Bounds,
    operation: str,
    *,
    strict_num_layers: bool,
    hybridz: bool,
) -> Fact:
    """Run one recurrent layer through the common sequence engine."""
    cfg = _read_rnn_config(L, operation, strict_num_layers)
    seq_lb, seq_ub = _seq_view(Bin, cfg)
    nonlinearity: str | None = None
    if operation == "lstm":
        step = _lstm_step
        state = _zero_state(Bin, cfg["batch"], cfg["hidden_size"], 4)
    elif operation == "gru":
        step = _gru_step
        state = _zero_state(Bin, cfg["batch"], cfg["hidden_size"], 2)
    elif operation == "rnn":
        step = _rnn_step
        nonlinearity = L.params.get("nonlinearity", "tanh")
        if nonlinearity not in ("tanh", "relu"):
            if hybridz:
                raise ValueError(
                    f"hybridz tf_rnn: unsupported nonlinearity {nonlinearity!r}"
                )
            raise ValueError(f"Unsupported RNN nonlinearity: {nonlinearity}")
    else:
        raise ValueError(f"Unsupported recurrent operation: {operation}")

    if operation == "rnn":
        state = _zero_state(Bin, cfg["batch"], cfg["hidden_size"], 2)
    out_lb, out_ub = _run_sequence(
        seq_lb,
        seq_ub,
        cfg,
        state,
        step,
        nonlinearity,
        hybridz,
    )
    return _make_fact(
        L,
        cfg,
        out_lb,
        out_ub,
        operation,
        nonlinearity,
        hybridz=hybridz,
    )


def _embedding_fact(L: Layer, *, hybridz: bool) -> Fact:
    """Build conservative embedding bounds for one transfer-function mode."""
    if hybridz:
        weight: torch.Tensor = L.params["weight"]
        embedding_dim = int(L.params.get("embedding_dim", weight.shape[-1]))
        output_shape = L.params.get("output_shape")
        repeats = _embedding_repeats_hybridz(L, embedding_dim, output_shape)
        meta = {
            "tag": f"embedding:{L.id}",
            "embedding_dim": embedding_dim,
            "input_shape": L.params.get("input_shape"),
            "output_shape": output_shape,
        }
    else:
        weight = _get_tensor_param(L, "weight")
        num_embeddings = L.get_int("num_embeddings")
        embedding_dim = L.get_int("embedding_dim")
        padding_idx = L.params.get("padding_idx", None)
        input_shape = _get_shape_param(L, "input_shape")
        output_shape = _get_shape_param(L, "output_shape")
        repeats = prod(output_shape) // embedding_dim
        meta = {
            "tag": f"embedding:{L.id}",
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "padding_idx": padding_idx,
            "input_shape": input_shape,
            "output_shape": output_shape,
        }

    B_out = Bounds(
        weight.min(dim=0).values.repeat(repeats),
        weight.max(dim=0).values.repeat(repeats),
    )
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), meta))
    C.add_box(L.id, L.out_vars, B_out)
    return Fact(B_out, C)


def _embedding_repeats_hybridz(
    L: Layer,
    embedding_dim: int,
    output_shape: Any,
) -> int:
    """Validate HybridZ embedding output width and return table repetitions."""
    n_out = len(L.out_vars)
    if n_out % embedding_dim != 0:
        raise ValueError(
            f"hybridz tf_embedding: out_vars count {n_out} not divisible by "
            f"embedding_dim {embedding_dim} (output_shape={output_shape})."
        )
    return n_out // embedding_dim


def _read_rnn_config(
    L: Layer,
    operation: str,
    strict_num_layers: bool,
) -> dict[str, Any]:
    """Read layer-zero recurrent weights and preserve mode-specific checks."""
    num_layers = L.get_int("num_layers", 1)
    bidirectional = bool(L.get_scalar("bidirectional", False))
    if strict_num_layers and num_layers != 1:
        raise ValueError(
            f"hybridz tf_{operation}: only num_layers=1 supported, got {num_layers}."
        )
    if bidirectional:
        if strict_num_layers:
            raise NotImplementedError(
                f"hybridz tf_{operation}: bidirectional=True not supported (reverse "
                f"direction over weight_*_l0_reverse not implemented)."
            )
        raise NotImplementedError(
            f"tf_{operation}: bidirectional=True not supported (reverse-direction "
            f"sweep over weight_*_l0_reverse not implemented)."
        )

    input_shape = _get_shape_param(L, "input_shape")
    batch_first = bool(L.get_scalar("batch_first", False))
    if batch_first:
        batch, seq_len, _ = input_shape
    else:
        seq_len, batch, _ = input_shape
    return {
        "input_size": L.get_int("input_size"),
        "hidden_size": L.get_int("hidden_size"),
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "batch_first": batch_first,
        "input_shape": input_shape,
        "output_shape": _get_shape_param(L, "output_shape"),
        "batch": batch,
        "seq_len": seq_len,
        "weights": {
            "W_ih": _get_tensor_param(L, "weight_ih_l0"),
            "W_hh": _get_tensor_param(L, "weight_hh_l0"),
            "b_ih": L.get_tensor("bias_ih_l0"),
            "b_hh": L.get_tensor("bias_hh_l0"),
        },
    }


def _zero_state(Bin: Bounds, batch: int, hidden_size: int, count: int) -> State:
    """Create the lower and upper tensors for a recurrent cell state."""
    zero = Bin.lb.new_zeros(batch, hidden_size)
    return (zero,) * count


def _run_sequence(
    seq_lb: torch.Tensor,
    seq_ub: torch.Tensor,
    cfg: dict[str, Any],
    state: State,
    step: Step,
    nonlinearity: str | None,
    hybridz: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unroll one recurrent cell over normalized sequence bounds."""
    out_lb_steps: list[torch.Tensor] = []
    out_ub_steps: list[torch.Tensor] = []
    for time_step in range(cfg["seq_len"]):
        state = step(
            seq_lb[:, time_step, :],
            seq_ub[:, time_step, :],
            state,
            cfg["weights"],
            cfg["hidden_size"],
            nonlinearity,
        )
        out_lb_steps.append(state[0])
        out_ub_steps.append(state[1])
    return _stack_steps(out_lb_steps, out_ub_steps, cfg, hybridz)


def _lstm_step(
    x_lb: torch.Tensor,
    x_ub: torch.Tensor,
    state: State,
    weights: dict[str, torch.Tensor | None],
    hidden_size: int,
    _nonlinearity: str | None,
) -> State:
    """Propagate interval bounds through one LSTM cell."""
    h_lb, h_ub, c_lb, c_ub = state
    pre_lb, pre_ub = _affine_block(
        x_lb,
        x_ub,
        h_lb,
        h_ub,
        weights["W_ih"],
        weights["W_hh"],
        weights["b_ih"],
        weights["b_hh"],
    )
    i_lb, i_ub = (
        torch.sigmoid(pre_lb[:, :hidden_size]),
        torch.sigmoid(pre_ub[:, :hidden_size]),
    )
    f_lb, f_ub = (
        torch.sigmoid(pre_lb[:, hidden_size : 2 * hidden_size]),
        torch.sigmoid(pre_ub[:, hidden_size : 2 * hidden_size]),
    )
    g_lb, g_ub = (
        torch.tanh(pre_lb[:, 2 * hidden_size : 3 * hidden_size]),
        torch.tanh(pre_ub[:, 2 * hidden_size : 3 * hidden_size]),
    )
    o_lb, o_ub = (
        torch.sigmoid(pre_lb[:, 3 * hidden_size :]),
        torch.sigmoid(pre_ub[:, 3 * hidden_size :]),
    )
    fc_lb, fc_ub = _interval_mul(f_lb, f_ub, c_lb, c_ub)
    ig_lb, ig_ub = _interval_mul(i_lb, i_ub, g_lb, g_ub)
    new_c_lb, new_c_ub = fc_lb + ig_lb, fc_ub + ig_ub
    tanh_c_lb, tanh_c_ub = torch.tanh(new_c_lb), torch.tanh(new_c_ub)
    new_h_lb, new_h_ub = _interval_mul(o_lb, o_ub, tanh_c_lb, tanh_c_ub)
    return new_h_lb, new_h_ub, new_c_lb, new_c_ub


def _gru_step(
    x_lb: torch.Tensor,
    x_ub: torch.Tensor,
    state: State,
    weights: dict[str, torch.Tensor | None],
    hidden_size: int,
    _nonlinearity: str | None,
) -> State:
    """Propagate interval bounds through one Torch-compatible GRU cell."""
    h_lb, h_ub = state
    W_ih = weights["W_ih"]
    W_hh = weights["W_hh"]
    b_ih = weights["b_ih"]
    b_hh = weights["b_hh"]
    assert W_ih is not None and W_hh is not None

    rz_lb, rz_ub = _affine_block(
        x_lb,
        x_ub,
        h_lb,
        h_ub,
        W_ih[: 2 * hidden_size],
        W_hh[: 2 * hidden_size],
        None if b_ih is None else b_ih[: 2 * hidden_size],
        None if b_hh is None else b_hh[: 2 * hidden_size],
    )
    r_lb, r_ub = (
        torch.sigmoid(rz_lb[:, :hidden_size]),
        torch.sigmoid(rz_ub[:, :hidden_size]),
    )
    z_lb, z_ub = (
        torch.sigmoid(rz_lb[:, hidden_size : 2 * hidden_size]),
        torch.sigmoid(rz_ub[:, hidden_size : 2 * hidden_size]),
    )
    in_lb, in_ub = _affine_only(
        x_lb,
        x_ub,
        W_ih[2 * hidden_size : 3 * hidden_size],
        None if b_ih is None else b_ih[2 * hidden_size : 3 * hidden_size],
    )
    hn_lb, hn_ub = _affine_only(
        h_lb,
        h_ub,
        W_hh[2 * hidden_size : 3 * hidden_size],
        None if b_hh is None else b_hh[2 * hidden_size : 3 * hidden_size],
    )
    rhn_lb, rhn_ub = _interval_mul(r_lb, r_ub, hn_lb, hn_ub)
    n_lb, n_ub = torch.tanh(in_lb + rhn_lb), torch.tanh(in_ub + rhn_ub)
    one_minus_z_lb, one_minus_z_ub = 1.0 - z_ub, 1.0 - z_lb
    n_term_lb, n_term_ub = _interval_mul(
        one_minus_z_lb,
        one_minus_z_ub,
        n_lb,
        n_ub,
    )
    h_term_lb, h_term_ub = _interval_mul(z_lb, z_ub, h_lb, h_ub)
    return n_term_lb + h_term_lb, n_term_ub + h_term_ub


def _rnn_step(
    x_lb: torch.Tensor,
    x_ub: torch.Tensor,
    state: State,
    weights: dict[str, torch.Tensor | None],
    _hidden_size: int,
    nonlinearity: str | None,
) -> State:
    """Propagate interval bounds through one vanilla RNN cell."""
    h_lb, h_ub = state
    pre_lb, pre_ub = _affine_block(
        x_lb,
        x_ub,
        h_lb,
        h_ub,
        weights["W_ih"],
        weights["W_hh"],
        weights["b_ih"],
        weights["b_hh"],
    )
    if nonlinearity == "tanh":
        return torch.tanh(pre_lb), torch.tanh(pre_ub)
    return pre_lb.clamp(min=0), pre_ub.clamp(min=0)


def _affine_only(
    x_lb: torch.Tensor,
    x_ub: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a sign-split affine map to an interval."""
    assert weight is not None
    weight_pos, weight_neg = split_weight(weight)
    out = affine_bounds(
        weight_pos,
        weight_neg,
        weight.new_zeros(weight.shape[0]),
        Bounds(x_lb, x_ub),
    )
    if bias is None:
        return out.lb, out.ub
    return out.lb + bias, out.ub + bias


def _affine_block(
    x_lb: torch.Tensor,
    x_ub: torch.Tensor,
    h_lb: torch.Tensor,
    h_ub: torch.Tensor,
    weight_ih: torch.Tensor | None,
    weight_hh: torch.Tensor | None,
    bias_ih: torch.Tensor | None,
    bias_hh: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the input and recurrent affine blocks to interval endpoints."""
    ih_lb, ih_ub = _affine_only(x_lb, x_ub, weight_ih, bias_ih)
    hh_lb, hh_ub = _affine_only(h_lb, h_ub, weight_hh, bias_hh)
    return ih_lb + hh_lb, ih_ub + hh_ub


def _interval_mul(
    a_lb: torch.Tensor,
    a_ub: torch.Tensor,
    b_lb: torch.Tensor,
    b_ub: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the four-corner product envelope for two intervals."""
    return four_corner_envelope(a_lb, a_ub, b_lb, b_ub)


def _seq_view(Bin: Bounds, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize flat or time-major sequence bounds to batch-major tensors."""
    shape = cfg["input_shape"]
    time_dim = shape[1] if cfg["batch_first"] else shape[0]
    feature_dim = shape[2]
    runtime_batch = Bin.lb.numel() // (time_dim * feature_dim)
    cfg["batch"] = runtime_batch
    target = (
        (runtime_batch, time_dim, feature_dim)
        if cfg["batch_first"]
        else (time_dim, runtime_batch, feature_dim)
    )
    seq_lb = Bin.lb.reshape(target)
    seq_ub = Bin.ub.reshape(target)
    if not cfg["batch_first"]:
        seq_lb = seq_lb.permute(1, 0, 2).contiguous()
        seq_ub = seq_ub.permute(1, 0, 2).contiguous()
    return seq_lb, seq_ub


def _stack_steps(
    lb_steps: list[torch.Tensor],
    ub_steps: list[torch.Tensor],
    cfg: dict[str, Any],
    hybridz: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack recurrent outputs and retain each mode's declared layout."""
    out_lb = torch.stack(lb_steps, dim=1)
    out_ub = torch.stack(ub_steps, dim=1)
    if hybridz and not cfg["batch_first"]:
        out_lb = out_lb.permute(1, 0, 2).contiguous()
        out_ub = out_ub.permute(1, 0, 2).contiguous()
    return out_lb, out_ub


def _make_fact(
    L: Layer,
    cfg: dict[str, Any],
    out_lb: torch.Tensor,
    out_ub: torch.Tensor,
    operation: str,
    nonlinearity: str | None,
    *,
    hybridz: bool,
) -> Fact:
    """Build the recurrent box and constraint fact for one mode."""
    if hybridz:
        per_instance = len(L.out_vars)
        actual = out_lb.numel()
        if per_instance > 0 and actual % per_instance != 0:
            raise RuntimeError(
                f"hybridz tf_{operation}: produced {actual} outputs, not a multiple "
                f"of len(out_vars)={per_instance} (output_shape={cfg['output_shape']})."
            )
    meta = {
        "tag": f"{operation}:{L.id}",
        "input_size": cfg["input_size"],
        "hidden_size": cfg["hidden_size"],
        "num_layers": cfg["num_layers"],
        "bidirectional": cfg["bidirectional"],
        "batch_first": cfg["batch_first"],
        "input_shape": cfg["input_shape"],
        "output_shape": cfg["output_shape"],
    }
    if nonlinearity is not None:
        meta["nonlinearity"] = nonlinearity
    B_out = Bounds(out_lb, out_ub)
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), meta))
    C.add_box(L.id, L.out_vars, B_out)
    return Fact(B_out, C)


def _get_tensor_param(L: Layer, key: str) -> torch.Tensor:
    """Return a required tensor layer parameter."""
    value = L.get_tensor(key)
    if value is None:
        raise TypeError(f"Expected tensor param '{key}' on layer {L.id}")
    return value


def _get_shape_param(L: Layer, key: str) -> tuple[int, ...]:
    """Return an integer tuple layer parameter."""
    return tuple(int(dim) for dim in L.get_tuple(key))
