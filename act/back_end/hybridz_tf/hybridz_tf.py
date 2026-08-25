# ===- act/back_end/hybridz_tf/hybridz_tf.py - HybridZ Transfer Function -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ Transfer Function Implementation.
#
#   Each hz_tf_* is a complete TF for one layer kind, combining
#   HZ zonotope propagation with interval_tf constraint generation.
#   hz_tf_* live in tf_mlp.py / tf_cnn.py alongside their layer types.
#   HZ domain ops co-locate with the hz_tf_* that use them.
#
# ===---------------------------------------------------------------------===#

""" """

import torch
from typing import Dict, Optional
from act.config.config import HybridZConfig
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.transfer_functions import RegistryTF
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_hz import (
    HZono,
    SparseHZono,
    hz_from_bounds,
    sparse_hz_fast_bounds,
    sparse_hz_from_bounds,
)

import act.back_end.hybridz_tf.tf_mlp as hz_mlp
import act.back_end.hybridz_tf.tf_cnn as hz_cnn
import act.back_end.hybridz_tf.tf_rnn as hz_rnn
import act.back_end.hybridz_tf.tf_transformer as hz_transformer
import act.back_end.interval_tf.tf_mlp as interval_mlp
import act.back_end.interval_tf.tf_cnn as interval_cnn


class HybridzTF(RegistryTF):
    def __init__(self, config: Optional[HybridZConfig] = None):
        super().__init__("HybridzTF")
        cfg = config or HybridZConfig()
        self._hz_cache: Dict[int, HZono] = {}
        self._sparse_hz_cache: Dict[int, SparseHZono] = {}
        self._sparse_drop_reasons: Dict[int, str] = {}
        self._sigmoid_affine_targets: Dict[int, int] = {}
        self._sigmoid_affine_inputs: Dict[int, HZono] = {}
        self._cache_net_id: Optional[int] = None
        self._tanh_K: int = 2
        self._sigmoid_K: int = int(cfg.sigmoid_segments)
        if self._sigmoid_K < 1:
            raise ValueError("HybridZ sigmoid_segments must be positive")
        self._fuse_sigmoid_affine: bool = bool(cfg.fuse_sigmoid_affine)
        self._var_id_stride: int = 1
        setattr(self, "_HZ_MAX_INPUT_DIM", cfg.max_input_dim)
        self._sparse_next_frame_id: int = 0
        self._sparse_frame_widths: Dict[int, tuple[int, int]] = {}
        self._sparse_relu_slots: Dict[tuple[int, int, int], tuple[int, int, int]] = {}

    @staticmethod
    def _net_var_id_stride(net: Net) -> int:
        max_id = -1
        for layer in net.layers:
            if layer.in_vars:
                max_id = max(max_id, max(layer.in_vars))
            if layer.out_vars:
                max_id = max(max_id, max(layer.out_vars))
        return max(max_id + 1, 1)

    _LAYER_REGISTRY = {
        # Identity / spec
        LayerKind.INPUT.value: lambda L, b, tf: Fact(bounds=b, cons=ConSet()),
        LayerKind.INPUT_SPEC.value: lambda L, b, tf: Fact(bounds=b, cons=ConSet()),
        LayerKind.ASSERT.value: lambda L, b, tf: Fact(bounds=b, cons=ConSet()),
        # MLP: HZ + interval
        LayerKind.DENSE.value: lambda L, b, tf: hz_mlp.tf_dense(L, b, tf),
        LayerKind.BIAS.value: lambda L, b, tf: hz_mlp.tf_bias(L, b, tf),
        LayerKind.SCALE.value: lambda L, b, tf: hz_mlp.tf_scale(L, b, tf),
        LayerKind.RELU.value: lambda L, b, tf: hz_mlp.tf_relu(L, b, tf),
        LayerKind.LRELU.value: lambda L, b, tf: hz_mlp.tf_lrelu(L, b, tf),
        LayerKind.TANH.value: lambda L, b, tf: hz_mlp.tf_tanh(L, b, tf),
        LayerKind.SIGMOID.value: lambda L, b, tf: hz_mlp.tf_sigmoid(L, b, tf),
        LayerKind.ERF.value: lambda L, b, tf: hz_mlp.tf_erf(L, b, tf),
        LayerKind.SQRT.value: lambda L, b, tf: hz_mlp.tf_sqrt(L, b, tf),
        LayerKind.SIN.value: lambda L, b, tf: hz_mlp.tf_sin(L, b, tf),
        LayerKind.COS.value: lambda L, b, tf: hz_mlp.tf_cos(L, b, tf),
        LayerKind.QUANTIZE.value: lambda L, b, tf: hz_mlp.tf_quantize(L, b, tf),
        LayerKind.ABS.value: lambda L, b, tf: hz_mlp.tf_abs(L, b, tf),
        LayerKind.BN.value: lambda L, b, tf: hz_mlp.tf_bn(L, b, tf),
        # Multi-input: HZ + interval
        LayerKind.ADD.value: lambda L, b, tf: hz_mlp.tf_add(L, b, tf),
        LayerKind.MUL.value: lambda L, b, tf: hz_mlp.tf_mul(L, b, tf),
        LayerKind.SUB.value: lambda L, b, tf: hz_mlp.tf_sub(L, b, tf),
        LayerKind.DIV.value: lambda L, b, tf: hz_mlp.tf_div(L, b, tf),
        LayerKind.CONCAT.value: lambda L, b, tf: hz_mlp.tf_concat(L, b, tf),
        # CNN: HZ + interval
        LayerKind.CONV2D.value: lambda L, b, tf: hz_cnn.tf_conv2d(L, b, tf),
        LayerKind.MAXPOOL2D.value: lambda L, b, tf: hz_cnn.tf_maxpool2d(L, b, tf),
        # Activations: interval-only
        LayerKind.CLIP.value: lambda L, b, tf: interval_mlp.tf_clip(L, b),
        LayerKind.SOFTPLUS.value: lambda L, b, tf: interval_mlp.tf_softplus(L, b),
        LayerKind.SILU.value: lambda L, b, tf: interval_mlp.tf_silu(L, b),
        LayerKind.RELU6.value: lambda L, b, tf: interval_mlp.tf_relu6(L, b),
        LayerKind.HARDTANH.value: lambda L, b, tf: interval_mlp.tf_hardtanh(L, b),
        LayerKind.HARDSIGMOID.value: lambda L, b, tf: interval_mlp.tf_hardsigmoid(L, b),
        LayerKind.HARDSWISH.value: lambda L, b, tf: interval_mlp.tf_hardswish(L, b),
        LayerKind.MISH.value: lambda L, b, tf: interval_mlp.tf_mish(L, b),
        LayerKind.SOFTSIGN.value: lambda L, b, tf: interval_mlp.tf_softsign(L, b),
        LayerKind.SQUARE.value: lambda L, b, tf: interval_mlp.tf_square(L, b),
        LayerKind.POWER.value: lambda L, b, tf: interval_mlp.tf_power(L, b),
        LayerKind.SIGN.value: lambda L, b, tf: hz_mlp.tf_sign(L, b, tf),
        LayerKind.REDUCE_SUM.value: lambda L, b, tf: hz_mlp.tf_reduce_sum(L, b, tf),
        LayerKind.CONSTANT.value: lambda L, b, tf: hz_mlp.tf_constant(L, b, tf),
        LayerKind.COMPARE.value: lambda L, b, tf: hz_mlp.tf_compare(L, b, tf),
        LayerKind.WHERE.value: lambda L, b, tf: hz_mlp.tf_where(L, b, tf),
        LayerKind.MATMUL.value: lambda L, b, tf: hz_mlp.tf_matmul(L, b, tf),
        LayerKind.ARG_EXTREMUM.value: lambda L, b, tf: hz_mlp.tf_arg_extremum(L, b, tf),
        LayerKind.UPSAMPLE.value: lambda L, b, tf: hz_mlp.tf_upsample(L, b, tf),
        LayerKind.SCATTER_ND.value: lambda L, b, tf: hz_mlp.tf_scatter_nd(L, b, tf),
        LayerKind.MAX.value: lambda L, b, tf: interval_mlp.tf_max(
            L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
        ),
        LayerKind.MIN.value: lambda L, b, tf: interval_mlp.tf_min(
            L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
        ),
        # CNN: interval-only
        LayerKind.AVGPOOL1D.value: lambda L, b, tf: interval_cnn.tf_avgpool1d(L, b),
        LayerKind.AVGPOOL2D.value: lambda L, b, tf: hz_cnn.tf_avgpool2d(L, b, tf),
        LayerKind.MAXPOOL3D.value: lambda L, b, tf: interval_cnn.tf_maxpool3d(L, b),
        LayerKind.PAD.value:      lambda L, b, tf: interval_cnn.tf_pad(L, b),
        LayerKind.CONV1D.value: lambda L, b, tf: interval_cnn.tf_conv1d(L, b),
        LayerKind.CONV3D.value: lambda L, b, tf: interval_cnn.tf_conv3d(L, b),
        LayerKind.CONVTRANSPOSE2D.value: lambda L, b, tf: hz_cnn.tf_convtranspose2d(L, b, tf),
        LayerKind.FLATTEN.value: lambda L, b, tf: hz_mlp.tf_flatten(L, b, tf),
        LayerKind.RESHAPE.value: lambda L, b, tf: hz_mlp.tf_reshape(L, b, tf),
        LayerKind.TRANSPOSE.value: lambda L, b, tf: hz_mlp.tf_transpose(L, b, tf),
        LayerKind.SQUEEZE.value: lambda L, b, tf: hz_mlp.tf_squeeze(L, b, tf),
        LayerKind.UNSQUEEZE.value: lambda L, b, tf: hz_mlp.tf_unsqueeze(L, b, tf),
        LayerKind.EXPAND.value: lambda L, b, tf: hz_mlp.tf_expand(L, b, tf),
        LayerKind.SLICE.value: lambda L, b, tf: hz_mlp.tf_slice(L, b, tf),
        LayerKind.GATHER.value: lambda L, b, tf: hz_mlp.tf_gather(L, b, tf),
        # RNN
        LayerKind.LSTM.value: lambda L, b, tf: hz_rnn.tf_lstm(L, b, tf),
        LayerKind.GRU.value: lambda L, b, tf: hz_rnn.tf_gru(L, b, tf),
        LayerKind.RNN.value: lambda L, b, tf: hz_rnn.tf_rnn(L, b, tf),
        LayerKind.EMBEDDING.value: lambda L, b, tf: hz_rnn.tf_embedding(L, b, tf),
        LayerKind.EMBEDDING_TF.value: lambda L, b, tf: hz_rnn.tf_embedding(L, b, tf),
        # Transformer
        LayerKind.POSENC.value: lambda L, b, tf: hz_transformer.tf_posenc(L, b, tf),
        LayerKind.LAYERNORM.value: lambda L, b, tf: hz_transformer.tf_layernorm(L, b, tf),
        LayerKind.GELU.value: lambda L, b, tf: hz_transformer.tf_gelu(L, b, tf),
        LayerKind.ATT_SCORES.value: lambda L, b, tf: hz_transformer.tf_att_scores(L, b, tf),
        LayerKind.SOFTMAX.value: lambda L, b, tf: hz_transformer.tf_softmax(L, b, tf),
        LayerKind.ATT_MIX.value: lambda L, b, tf: hz_transformer.tf_att_mix(L, b, tf),
        LayerKind.MHA_SPLIT.value: lambda L, b, tf: hz_transformer.tf_mha_split(L, b, tf),
        LayerKind.MHA_JOIN.value: lambda L, b, tf: hz_transformer.tf_mha_join(L, b, tf),
        LayerKind.MASK_ADD.value: lambda L, b, tf: hz_transformer.tf_mask_add(L, b, tf),
    }

    def get_hz(self, layer_id: int) -> Optional[HZono]:
        return self._hz_cache.get(int(layer_id))

    def get_sparse_hz(self, layer_id: int) -> Optional[SparseHZono]:
        return self._sparse_hz_cache.get(int(layer_id))

    @staticmethod
    def _id_sig(ids: Optional[torch.Tensor]):
        if ids is None:
            return None
        vals = ids.detach().cpu().reshape(-1).tolist()
        return (tuple(ids.shape), tuple(int(v) for v in vals))

    @classmethod
    def _hz_sig(cls, hz: Optional[HZono]):
        if hz is None:
            return None
        eq_sig = None
        if hz.eq_mask is not None:
            eq_sig = (
                tuple(hz.eq_mask.shape),
                tuple(bool(v) for v in hz.eq_mask.detach().cpu().reshape(-1).tolist()),
            )
        return (
            tuple(hz.c.shape),
            tuple(hz.Gc.shape),
            tuple(hz.Gb.shape),
            tuple(hz.Ac.shape),
            tuple(hz.Ab.shape),
            tuple(hz.b.shape),
            eq_sig,
            cls._id_sig(hz.col_ids),
            cls._id_sig(hz.bcol_ids),
        )

    @staticmethod
    def _csr_sig(mat):
        return (tuple(mat.shape), int(mat.nnz))

    @classmethod
    def _sparse_hz_sig(cls, hz: Optional[SparseHZono]):
        if hz is None:
            return None
        return (
            int(hz.n_out),
            int(hz.n_cont),
            int(hz.n_bin),
            int(hz.n_eq),
            int(hz.n_ineq),
            cls._csr_sig(hz.Gc),
            cls._csr_sig(hz.Gb),
            cls._csr_sig(hz.Ac),
            cls._csr_sig(hz.Ab),
            cls._csr_sig(hz.Auc),
            cls._csr_sig(hz.Aub),
            hz.frame_id,
        )

    def side_state_signature(self, layer_id: int):
        lid = int(layer_id)
        target = self._sigmoid_affine_targets.get(lid, lid)
        return (
            self._hz_sig(self._hz_cache.get(lid)),
            self._sparse_hz_sig(self._sparse_hz_cache.get(lid)),
            self._sparse_drop_reasons.get(lid),
            self._hz_sig(self._sigmoid_affine_inputs.get(target)),
        )

    _HZ_MAX_INPUT_DIM = 1024
    _SPARSE_MAX_AFFINE_CELLS = 8_000_000

    @staticmethod
    def _find_sigmoid_affine_targets(net: Net) -> Dict[int, int]:
        shape_only = {
            LayerKind.FLATTEN.value,
            LayerKind.RESHAPE.value,
        }
        targets: Dict[int, int] = {}
        for layer in net.layers:
            if layer.kind.upper() != LayerKind.SIGMOID.value:
                continue
            current = layer.id
            chain = [current]
            while True:
                successors = net.succs.get(current, [])
                if len(successors) != 1:
                    break
                successor = successors[0]
                if net.preds.get(successor, []) != [current]:
                    break
                kind = net.by_id[successor].kind.upper()
                if kind == LayerKind.DENSE.value:
                    targets.update((node, successor) for node in chain)
                    break
                if kind not in shape_only:
                    break
                current = successor
                chain.append(current)
        return targets

    def _col_ids_from_vars(self, bounds: Bounds, var_ids) -> Optional[torch.Tensor]:
        if not var_ids:
            return None
        n = len(var_ids)
        total = int(bounds.lb.numel())
        if n <= 0 or total % n != 0:
            return None
        base = torch.tensor(var_ids, dtype=torch.long, device=bounds.lb.device)
        B = total // n
        if B == 1:
            return base
        offsets = (
            torch.arange(B, dtype=torch.long, device=bounds.lb.device).view(-1, 1)
            * self._var_id_stride
        )
        return (offsets + base.view(1, -1)).reshape(-1)

    def _hz_from_bounds(
        self,
        bounds: Bounds,
        *,
        col_ids: Optional[torch.Tensor] = None,
    ) -> Optional[HZono]:
        lb, ub = bounds.lb.flatten(), bounds.ub.flatten()
        rad = (ub - lb) / 2.0
        ng = int((rad > 0).sum().item())
        if ng > self._HZ_MAX_INPUT_DIM:
            return None
        ids = col_ids.to(device=lb.device) if col_ids is not None else None
        return hz_from_bounds(
            bounds,
            lb.dtype,
            lb.device,
            col_ids=ids,
        )

    def _sparse_from_bounds(self, bounds: Bounds) -> SparseHZono:
        frame_id = self._sparse_next_frame_id
        self._sparse_next_frame_id += 1
        hz = sparse_hz_from_bounds(bounds, frame_id=frame_id)
        self._sparse_frame_widths[frame_id] = (hz.n_cont, hz.n_bin)
        return hz

    def _sparse_relu_slots_for(
        self,
        hz: SparseHZono,
        layer_id: int,
        neurons,
    ) -> Optional[tuple[list[tuple[int, int, int]], int, int]]:
        if hz.frame_id is None:
            raise ValueError("sparse ReLU requires a generator frame")
        frame_id = int(hz.frame_id)
        n_cont, n_bin = self._sparse_frame_widths.get(
            frame_id, (hz.n_cont, hz.n_bin)
        )
        n_cont = max(n_cont, hz.n_cont)
        n_bin = max(n_bin, hz.n_bin)
        missing = sum(
            (frame_id, int(layer_id), int(neuron)) not in self._sparse_relu_slots
            for neuron in neurons
        )
        if hz.n_out * (n_cont + n_bin + 3 * missing) > self._SPARSE_MAX_AFFINE_CELLS:
            return None
        slots = []
        for neuron in neurons:
            key = (frame_id, int(layer_id), int(neuron))
            slot = self._sparse_relu_slots.get(key)
            if slot is None:
                slot = (n_cont, n_cont + 1, n_bin)
                self._sparse_relu_slots[key] = slot
                n_cont += 2
                n_bin += 1
            slots.append(slot)
        self._sparse_frame_widths[frame_id] = (n_cont, n_bin)
        return slots, n_cont, n_bin

    @staticmethod
    def _sparse_fact(fact: Fact, hz: SparseHZono) -> Fact:
        hb = sparse_hz_fast_bounds(hz)
        lb = hb.lb.to(dtype=fact.bounds.lb.dtype, device=fact.bounds.lb.device)
        ub = hb.ub.to(dtype=fact.bounds.ub.dtype, device=fact.bounds.ub.device)
        return Fact(
            bounds=Bounds(
                lb=torch.maximum(lb.reshape_as(fact.bounds.lb), fact.bounds.lb),
                ub=torch.minimum(ub.reshape_as(fact.bounds.ub), fact.bounds.ub),
            ),
            cons=fact.cons,
        )

    def _seed_sparse_cache(self, L: Layer, input_bounds: Bounds) -> None:
        k = L.kind.upper()
        try:
            if k in ("INPUT", "INPUT_SPEC"):
                self._sparse_hz_cache[L.id] = self._sparse_from_bounds(input_bounds)
                self._sparse_drop_reasons.pop(L.id, None)
            elif k != "ASSERT":
                preds = self._net.preds.get(L.id, [])
                if preds and preds[0] in self._sparse_hz_cache:
                    self._sparse_hz_cache[L.id] = self._sparse_hz_cache[preds[0]]
                    self._sparse_drop_reasons.pop(L.id, None)
                elif not preds:
                    self._sparse_hz_cache[L.id] = self._sparse_from_bounds(input_bounds)
                    self._sparse_drop_reasons.pop(L.id, None)
        except Exception as exc:
            self._drop_sparse_hz(L.id, f"sparse_seed_failed:{type(exc).__name__}")

    def _drop_sparse_hz(self, layer_id: int, reason: str) -> None:
        lid = int(layer_id)
        self._sparse_hz_cache.pop(lid, None)
        self._sparse_drop_reasons[lid] = reason

    def _sparse_exceeds_limit(self, hz: SparseHZono, out_dim: int) -> bool:
        gen = int(hz.n_cont + hz.n_bin)
        return gen > 0 and int(out_dim) * gen > self._SPARSE_MAX_AFFINE_CELLS

    def _propagate_sparse_hz(self, L: Layer, input_bounds: Bounds, result: Fact) -> Fact:
        k = L.kind.upper()
        if k in ("INPUT", "INPUT_SPEC", "ASSERT"):
            hz = self._sparse_hz_cache.get(L.id)
            return self._sparse_fact(result, hz) if hz is not None else result
        hz = self._sparse_hz_cache.get(L.id)
        if hz is None:
            return result
        if self._sparse_exceeds_limit(hz, result.bounds.lb.numel()):
            self._drop_sparse_hz(L.id, f"sparse_size_limit:{k}")
            return result
        try:
            for apply_sparse in (
                hz_mlp.sparse_hz_apply_layer,
                hz_cnn.sparse_hz_apply_layer,
            ):
                handled, out, drop_reason = apply_sparse(L, hz, input_bounds, result, self)
                if not handled:
                    continue
                if out is None:
                    self._drop_sparse_hz(L.id, drop_reason or f"unsupported_sparse_op:{k}")
                    return result
                self._sparse_hz_cache[L.id] = out
                self._sparse_drop_reasons.pop(L.id, None)
                return self._sparse_fact(result, out)
            self._drop_sparse_hz(L.id, f"unsupported_sparse_op:{k}")
        except Exception as exc:
            self._drop_sparse_hz(L.id, f"sparse_op_failed:{k}:{type(exc).__name__}")
        return result

    def apply(
        self,
        L: Layer,
        input_bounds: Bounds,
        net: Net,
        before: Dict[int, Fact],
        after: Dict[int, Fact],
    ) -> Fact:
        k = self._check_supported(L.kind)

        net_id = id(net)
        if self._cache_net_id != net_id:
            self._hz_cache.clear()
            self._sparse_hz_cache.clear()
            self._sparse_drop_reasons.clear()
            self._sigmoid_affine_inputs.clear()
            self._sparse_frame_widths.clear()
            self._sparse_relu_slots.clear()
            self._cache_net_id = net_id
            self._var_id_stride = self._net_var_id_stride(net)
            self._sparse_next_frame_id = 0
            self._sigmoid_affine_targets = (
                self._find_sigmoid_affine_targets(net)
                if self._fuse_sigmoid_affine
                else {}
            )

        self._set_context(net, before, after)
        self._seed_sparse_cache(L, input_bounds)

        if k in ("INPUT", "INPUT_SPEC"):
            hz_init = self._hz_from_bounds(
                input_bounds,
                col_ids=self._col_ids_from_vars(input_bounds, L.out_vars),
            )
            if hz_init is not None:
                self._hz_cache[L.id] = hz_init
        elif k != "ASSERT":
            preds = net.preds.get(L.id, [])
            if preds and preds[0] in self._hz_cache:
                self._hz_cache[L.id] = self._hz_cache[preds[0]]
            elif not preds:
                hz_init = self._hz_from_bounds(
                    input_bounds,
                    col_ids=self._col_ids_from_vars(input_bounds, L.in_vars),
                )
                if hz_init is not None:
                    self._hz_cache[L.id] = hz_init

        n_out = len(L.out_vars)
        hz_carried = self._hz_cache.get(L.id)
        ngnb = (
            hz_carried.Gc.shape[1] + hz_carried.Gb.shape[1]
            if hz_carried is not None
            else 0
        )
        if max(n_out, ngnb) > self._HZ_MAX_INPUT_DIM and k not in (
            "INPUT",
            "INPUT_SPEC",
            "ASSERT",
        ):
            self._hz_cache.pop(L.id, None)

        hz_before = self._hz_cache.get(L.id)
        result = self._LAYER_REGISTRY[k](L, input_bounds, self)
        result = self._propagate_sparse_hz(L, input_bounds, result)

        if (
            hz_before is not None
            and self._hz_cache.get(L.id) is hz_before
            and k not in ("INPUT", "INPUT_SPEC")
        ):
            self._hz_cache[L.id] = hz_from_bounds(
                result.bounds, result.bounds.lb.dtype, result.bounds.lb.device
            )

        return result
