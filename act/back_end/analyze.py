#===- act/back_end/analyze.py - Network Analysis Functions --------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Network analysis functions for ACT verification framework.
#   Provides analysis capabilities for neural network structures and properties.
#
#===---------------------------------------------------------------------===#

import heapq
import torch
from act.pipeline.gui import act_gui as _gui
from act.pipeline.gui import verifier_gui as _obs
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, cast
from act.back_end.core import Bounds, Fact, Net, ConSet, get_topo_order
from act.back_end.layer_schema import LayerKind
from act.back_end.utils import box_join, changed_or_maskdiff, update_cache
from act.back_end.transfer_functions import (
    dispatch_tf,
    get_transfer_function,
    set_transfer_function_mode,
)

# Initialize default transfer function mode
def initialize_tf_mode(mode: str = "interval", tf_config=None):
    """Initialize transfer function mode. Call this before using analyze()."""
    set_transfer_function_mode(mode, tf_config)


@dataclass
class AnalyzeCache:
    """Reusable dataflow state for monotone BaB input-box refinements."""
    before: Dict[int, Fact]
    after: Dict[int, Fact]
    globalC: ConSet


@_gui.observed_phase("analyze")
@torch.no_grad()
def analyze(
    net: Net,
    entry_id: int,
    entry_fact: Fact,
    eps: float = 1e-9,
    *,
    cache: Optional[AnalyzeCache] = None,
) -> Tuple[Dict[int, Fact], Dict[int, Fact], ConSet]:
    """
    Perform abstract interpretation on the network starting from entry_fact.
    Args:
        net: ACT network structure
        entry_id: ID of the entry (INPUT) layer
        entry_fact: Initial Fact containing bounds and constraints for the input
        eps: Convergence epsilon for fixpoint iteration
        cache: Optional state from a prior monotone-refinement analysis.

    Returns:
        Tuple of (before, after, globalC) containing propagated facts and global constraints
    """
    from act.back_end.transfer_functions import ensure_active_tf
    ensure_active_tf("interval")

    if cache is None:
        before: Dict[int, Fact] = {}
        after:  Dict[int, Fact] = {}
        globalC = ConSet()

        # Default bounds must carry the leading batch dim. Consumer TFs assume
        # bounds are [B, n] and reshape on that axis; a (n,) default would
        # silently broadcast in some paths and fail loudly in others.
        seed = entry_fact.bounds.lb
        B = seed.shape[0] if seed.dim() >= 2 else 1
        for layer in net.layers:
            n = len(layer.out_vars)
            hi = seed.new_full((B, n), float("inf"))
            lo = seed.new_full((B, n), -float("inf"))
            before[layer.id] = Fact(bounds=Bounds(lo.clone(), hi.clone()), cons=ConSet())
            after[layer.id] = Fact(bounds=Bounds(lo.clone(), hi.clone()), cons=ConSet())
            layer.cache.clear()

        # Seed entry with provided Fact (includes all input constraints)
        before[entry_id] = entry_fact

        # Seed every other zero-indegree source (e.g. CONSTANT layers emitted by
        # torch2act for ONNX initializers). Without this, source-layer bounds stay
        # at +/-inf forever because the worklist starts at entry_id and CONSTANTs
        # have no predecessor that would ever push them on. (Oracle finding #5.)
        seeds = [entry_id]
        for layer in net.layers:
            if layer.id == entry_id or net.preds.get(layer.id):
                continue
            if layer.kind == LayerKind.CONSTANT.value:
                B_size = entry_fact.bounds.lb.shape[0]
                raw_value = cast(object, layer.params["value"])
                if not isinstance(raw_value, torch.Tensor):
                    raise TypeError(
                        f"CONSTANT layer {layer.id} requires tensor param 'value', got {type(raw_value).__name__}."
                    )
                val = raw_value.reshape(-1).to(
                    device=entry_fact.bounds.lb.device,
                    dtype=entry_fact.bounds.lb.dtype,
                )
                val_b = val.unsqueeze(0).expand(B_size, -1).contiguous()  # [B, numel]
                before[layer.id] = Fact(bounds=Bounds(val_b.clone(), val_b.clone()), cons=ConSet())
            # Other zero-indegree kinds (none today) would be seeded similarly.
            seeds.append(layer.id)
    else:
        before = cache.before
        after = cache.after
        globalC = cache.globalC
        before[entry_id] = entry_fact
        seeds = [entry_id]

    tf = get_transfer_function()
    if _gui.enabled:
        from act.back_end.hybridz_tf import HybridzTF
        domain = "hybridz" if isinstance(tf, HybridzTF) else "interval"
        versions = {}
        for lid in seeds:
            layer = net.by_id[lid]
            if layer.kind in (LayerKind.INPUT.value, LayerKind.CONSTANT.value):
                bounds = before[lid].bounds
                versions[lid] = 0
                _gui.emit(_obs.LayerBounds(
                    _obs.RecordKey(domain, "forward", lid, "out", 0),
                    layer.kind, bounds.lb, bounds.ub,
                ))
    single_pass = bool(getattr(tf, "topological_single_pass", False))
    # The worklist pops layers in topological order, so a merge point (ADD,
    # CONCAT, ...) runs after every predecessor it can reach. In FIFO order a
    # short branch could reach the merge first and pull the +/-inf placeholder of
    # the longer branch; those infinite bounds reach later layers as NaN
    # (inf - inf) before the longer branch arrives.
    topo_index = {lid: i for i, lid in enumerate(get_topo_order(net))}
    WL: list[Tuple[int, int]] = []
    queued: set[int] = set()

    def push(lid: int) -> None:
        if lid not in queued:
            queued.add(lid)
            heapq.heappush(WL, (topo_index[lid], lid))

    for lid in (get_topo_order(net) if single_pass else seeds):
        push(lid)
    while WL:
        _, lid = heapq.heappop(WL); queued.discard(lid); layer = net.by_id[lid]

        # merge predecessors into before[lid]
        if net.preds.get(lid):
            preds_list = net.preds[lid]
            # Initialize from first predecessor (not infinite bounds)
            first_bounds = after[preds_list[0]].bounds
            Bjoin = Bounds(lb=first_bounds.lb.clone(), ub=first_bounds.ub.clone())
            Cjoin = ConSet()
            for con in after[preds_list[0]].cons: Cjoin.replace(con)
            # Join with remaining predecessors when shapes match (DAG merge points).
            # Multi-input ops with heterogeneous predecessor shapes (MATMUL, CONCAT,
            # SCATTER_ND, etc.) ignore Bin and pull each predecessor explicitly via
            # get_predecessor_bounds; the join is meaningless for them so we skip
            # rather than crash.
            for pid in preds_list[1:]:
                pb = after[pid].bounds
                if pb.lb.shape == Bjoin.lb.shape:
                    Bjoin = box_join(Bjoin, pb)
                for con in after[pid].cons: Cjoin.replace(con)
            before[lid] = Fact(Bjoin, Cjoin)

        out_fact = dispatch_tf(layer, before, after, net)
        side_sig = tf.side_state_signature(layer.id)
        side_changed = layer.cache.get("prev_tf_side_state") != side_sig

        if changed_or_maskdiff(layer, out_fact.bounds, None, eps) or side_changed:
            after[lid] = out_fact
            update_cache(layer, out_fact.bounds, None)
            layer.cache["prev_tf_side_state"] = side_sig
            for con in out_fact.cons: globalC.replace(con)
            if not single_pass:
                for sid in net.succs.get(lid, []): push(sid)
            if _gui.enabled:
                versions[lid] = versions.get(lid, 0) + 1
                bounds = after[lid].bounds
                key = _obs.RecordKey(domain, "forward", lid, "out", versions[lid])
                _gui.emit(_obs.LayerBounds(
                    key, layer.kind, bounds.lb, bounds.ub,
                ))
                if layer.kind == LayerKind.RELU.value and _obs.wants_relaxation():
                    _gui.emit(_obs.DESCRIBERS[domain](key, before[lid].bounds, bounds, tf=tf))
        elif _gui.enabled:
            bounds = after[lid].bounds
            _gui.emit(_obs.LayerBounds(
                _obs.RecordKey(domain, "forward", lid, "out", versions.get(lid, 0)),
                layer.kind, bounds.lb, bounds.ub, changed=False,
            ))

    return before, after, globalC
