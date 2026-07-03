#!/usr/bin/env python3
"""
Closed-loop LLM guidance for dual-batched Branch-and-Bound (BaB) verification.

Soundness boundary: the LLM only proposes search-scheduling decisions (split
depth/group, wave width, refinement effort); the BaB verifier alone computes
bounds and certifies/falsifies counterexamples. Invalid, missing, or
unavailable LLM guidance always falls back to the verifier's own baseline
behavior (disabled probe == baseline, bit-identical).

What this module provides
- Dataclasses describing the controller's inputs/outputs: FrontierStats,
  CandidateSummary, RoundAdvice, RoundPolicy, WaveOutcome, WaveRecord.
- LLMBackend implementations: MockBackend (offline/tests, no network) and
  OpenAICompatibleBackend (stdlib-only HTTP client; works with OpenRouter and
  direct OpenAI/GLM/MiniMax-compatible endpoints via _PROVIDER_PRESETS).
- LLMProbe: a stateful controller consulted once per BaB wave
  (begin_wave/end_wave) for split_k / k_requested / refine policy, plus an
  optional second post-solve consult (advise_neuron_groups) for joint
  neuron-group selection, gated by a per-backend cadence, bounded history,
  and a circuit breaker that disables the probe after repeated failures.
- build_llm_probe(config): factory that wires act/util/options.py
  `llm_probe_*` config fields into a concrete LLMProbe; imported lazily from
  act/back_end/bab/bab.py only when `llm_probe_enabled` is set.

Design docs: .sisyphus/plans/llm_probe_closed_loop_plan.md (M1) and
.sisyphus/plans/llm_probe_m2_plan.md (M2).
"""

import os
import json
import subprocess
import threading
import urllib.request
from collections import deque
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

_REFINE_MODES = ("none", "tail", "all")

_SYSTEM_PROMPT = (
    "You guide a sound neural-network branch-and-bound verifier. Return ONLY a JSON object. "
    "Default (scheduling) phase: optional integer fields split_k, k_requested, refine_iters, "
    "refine_rows_cap, optional string refine_mode in {none,tail,all}, optional integer horizon_hint, "
    "optional string rationale. "
    "phase='neuron_selection': return split_groups, a list of {lane, layer_id, neuron_idx} choosing "
    "which unstable neurons to split jointly per lane; the per-lane group size IS the split depth k and "
    "produces 2^k child subproblems. "
    "When the payload contains input_widths (input-domain-splitting mode, one mean width per input "
    "dimension), you may return optional integer fields input_dim (which dimension to bisect next; "
    "prefer dimensions with large width and high sensitivity) and input_fanout (2-8 equal segments). "
    "SPLIT-DEPTH (k) RULE — you MUST apply this deterministic rule, k capped at multi_split_levels; "
    "each 2^k split multiplies the pending pool, so throttle k only once a backlog builds. Let "
    "r = pool_size / max(1, effective_batch) (how many waves' worth of work is already queued; treat a "
    "null/absent pool_growth_rate_recent as 0): "
    "if r <= 1 -> split_k = multi_split_levels (the whole frontier fits in one wave, split DEEP for fast "
    "progress); else if r <= 2 -> split_k = 2; else -> split_k = 1 (a backlog is building, split shallow "
    "to avoid subproblem explosion). For neuron_selection size EVERY lane's group to this same k. State "
    "the computed r and chosen k in rationale. "
    "Omit a field to defer to the verifier's default. You never certify or falsify; you only schedule "
    "work within the provided limits."
)


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def _coerce_groups(value: Any) -> Optional[List[Dict[str, int]]]:
    if not isinstance(value, list):
        return None
    out: List[Dict[str, int]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        lane = _coerce_int(entry.get("lane"))
        layer_id = _coerce_int(entry.get("layer_id"))
        neuron_idx = _coerce_int(entry.get("neuron_idx"))
        if lane is None or layer_id is None or neuron_idx is None:
            continue
        out.append({"lane": lane, "layer_id": layer_id, "neuron_idx": neuron_idx})
    return out or None


@dataclass
class CandidateSummary:
    lane: int
    layer_id: int
    neuron_idx: int
    score: float
    lb: float
    ub: float
    nu: Optional[float] = None
    area: Optional[float] = None


@dataclass
class RoundAdvice:
    split_k: Optional[int] = None
    k_requested: Optional[int] = None
    refine_mode: Optional[str] = None
    refine_iters: Optional[int] = None
    refine_rows_cap: Optional[int] = None
    horizon_hint: Optional[int] = None
    rationale: Optional[str] = None
    split_groups: Optional[List[Dict[str, int]]] = None
    input_dim: Optional[int] = None
    input_fanout: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Any) -> "RoundAdvice":
        if not isinstance(data, dict):
            return cls()
        return cls(
            split_k=_coerce_int(data.get("split_k")),
            k_requested=_coerce_int(data.get("k_requested")),
            refine_mode=_coerce_str(data.get("refine_mode")),
            refine_iters=_coerce_int(data.get("refine_iters")),
            refine_rows_cap=_coerce_int(data.get("refine_rows_cap")),
            horizon_hint=_coerce_int(data.get("horizon_hint")),
            rationale=_coerce_str(data.get("rationale")),
            split_groups=_coerce_groups(data.get("split_groups")),
            input_dim=_coerce_int(data.get("input_dim")),
            input_fanout=_coerce_int(data.get("input_fanout")),
        )


@dataclass
class FrontierStats:
    wave_index: int
    pool_size: int
    effective_batch: int
    remaining_nodes: int
    elapsed_s: float
    branch_batch_size: int = 1
    remaining_s: Optional[float] = None
    depth_min: Optional[int] = None
    depth_max: Optional[int] = None
    lower_bound_min: Optional[float] = None
    lower_bound_max: Optional[float] = None
    candidates: List[CandidateSummary] = field(default_factory=list)
    input_widths: Optional[List[float]] = None


@dataclass
class WaveOutcome:
    wave_index: int
    pool_before: int
    pool_after: int
    k_requested_used: int
    split_k_used: int
    refine_iters_used: int
    certified_count: int
    falsified_found: bool
    branched_count: int
    best_lb_before: Optional[float]
    best_lb_after: Optional[float]
    wave_time_s: float
    fallback_used: bool


@dataclass
class WaveRecord:
    wave_index: int
    advice: RoundAdvice
    outcome: WaveOutcome
    valid_response: bool


@dataclass
class RoundPolicy:
    split_k: Optional[int] = None
    k_requested: Optional[int] = None
    refine_mode: Optional[str] = None
    refine_iters: Optional[int] = None
    refine_rows_cap: Optional[int] = None
    split_groups: Optional[Dict[int, List[Tuple[int, int]]]] = None
    input_split_dim: Optional[int] = None
    input_split_fanout: Optional[int] = None


def clip_input_split(dim: Any, fanout: Any, *, n_dims: int) -> Tuple[Optional[int], Optional[int]]:
    out_dim = _coerce_int(dim)
    if out_dim is not None and not (0 <= out_dim < max(1, int(n_dims))):
        out_dim = None
    out_fanout = _coerce_int(fanout)
    if out_fanout is not None:
        out_fanout = max(2, min(8, out_fanout))
    return out_dim, out_fanout


def clip_split_k(value: Any, *, branch_batch_size: int, effective_batch: int,
                 multi_split_levels: int) -> int:
    parsed = _coerce_int(value)
    if parsed is None:
        parsed = 1
    upper = max(1, int(multi_split_levels))
    bb = max(1, int(branch_batch_size))
    cap = max(1, int(effective_batch))
    feasible = 1
    while feasible < upper and (2 ** (feasible + 1)) * bb <= cap:
        feasible += 1
    return max(1, min(parsed, upper, feasible))


def clip_k_requested(value: Any, *, baseline: int, pool_size: int,
                     effective_batch: int, remaining_nodes: int) -> int:
    upper = max(1, min(int(pool_size), int(effective_batch), int(remaining_nodes)))
    parsed = _coerce_int(value)
    if parsed is None:
        parsed = int(baseline)
    return max(1, min(parsed, upper))


def clip_refine(mode: Any, iters: Any, rows_cap: Any, *, iters_cap: int,
                rows_cap_cap: int):
    out_mode = mode if mode in _REFINE_MODES else None
    out_iters = None
    parsed_iters = _coerce_int(iters)
    if parsed_iters is not None:
        out_iters = max(0, min(parsed_iters, int(iters_cap)))
    out_rows = None
    parsed_rows = _coerce_int(rows_cap)
    if parsed_rows is not None:
        out_rows = max(0, min(parsed_rows, int(rows_cap_cap)))
    return out_mode, out_iters, out_rows


def clip_split_groups(groups: Any, *, candidates: List[CandidateSummary],
                      branch_batch_size: int, effective_batch: int,
                      multi_split_levels: int) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    if not groups:
        return None
    legal: Dict[int, "set[Tuple[int, int]]"] = {}
    for cand in candidates:
        legal.setdefault(cand.lane, set()).add((cand.layer_id, cand.neuron_idx))
    per_lane: Dict[int, List[Tuple[int, int]]] = {}
    for entry in groups:
        if not isinstance(entry, dict):
            continue
        lane = entry.get("lane")
        lid = entry.get("layer_id")
        nidx = entry.get("neuron_idx")
        if lane is None or lid is None or nidx is None:
            continue
        lane_i = int(lane)
        key = (int(lid), int(nidx))
        if lane_i not in legal or key not in legal[lane_i]:
            continue
        ordered = per_lane.setdefault(lane_i, [])
        if key not in ordered:
            ordered.append(key)
    bb = max(1, int(branch_batch_size))
    cap = max(1, int(effective_batch))
    k_cap = 1
    while k_cap < int(multi_split_levels) and (2 ** (k_cap + 1)) * bb <= cap:
        k_cap += 1
    sizes = [len(per_lane.get(lane, [])) for lane in range(bb)]
    if not sizes or min(sizes) < 1:
        return None
    k_eff = min(min(sizes), k_cap, int(multi_split_levels))
    if k_eff < 1:
        return None
    return {lane: per_lane[lane][:k_eff] for lane in range(bb)}


def _extract_json(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 2:
            stripped = parts[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    start = stripped.find("{")
    if start == -1:
        return json.loads(stripped)
    depth = 0
    for i in range(start, len(stripped)):
        if stripped[i] == "{":
            depth += 1
        elif stripped[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(stripped[start:i + 1])
    return json.loads(stripped[start:])


class LLMBackend:
    def complete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class MockBackend(LLMBackend):
    def __init__(self, responder: Optional[Any] = None):
        self._responder = responder

    def complete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        responder = self._responder
        if callable(responder):
            result = responder(payload)
            return result if isinstance(result, dict) else {}
        if isinstance(responder, dict):
            return dict(responder)
        return {}


class OpenAICompatibleBackend(LLMBackend):
    def __init__(self, *, base_url: str, model: str, api_key: str,
                 temperature: float = 0.0, timeout: float = 30.0):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._api_key = api_key
        self._temperature = float(temperature)
        self._timeout = float(timeout)

    def complete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps({
            "model": self._model,
            "temperature": self._temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
        }).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            envelope = json.loads(response.read().decode("utf-8"))
        content = envelope["choices"][0]["message"]["content"]
        return _extract_json(content)


class ClaudeCLIBackend(LLMBackend):
    """Shells out to the local `claude` CLI (Claude Code) in non-interactive
    print mode, instead of a raw HTTP API call. Useful when a Claude
    subscription/session is already authenticated locally and no separate
    ANTHROPIC_API_KEY is configured. Stdlib `subprocess` only (no new
    dependency). Tool use is disabled (`--disallowedTools "*"`): the probe
    only ever needs one text completion per call, never agentic tool calls.
    """

    def __init__(self, *, model: str = "sonnet", timeout: float = 60.0,
                 binary: str = "claude"):
        self._model = model
        self._timeout = float(timeout)
        self._binary = binary

    def complete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        args = [
            self._binary, "-p", json.dumps(payload),
            "--output-format", "json",
            "--model", self._model,
            "--append-system-prompt", _SYSTEM_PROMPT,
            "--disallowedTools", "*",
        ]
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=self._timeout, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI exited {proc.returncode}: {proc.stderr[:200]}")
        envelope = json.loads(proc.stdout)
        if envelope.get("is_error"):
            raise RuntimeError(f"claude CLI error: {envelope.get('result')!r}")
        return _extract_json(str(envelope.get("result", "")))


class LLMProbe:
    def __init__(self, backend: LLMBackend, *, cadence: int = 1, history: int = 8,
                 max_failures: int = 3, multi_split_levels: int = 1,
                 refine_iters_cap: int = 0, refine_rows_cap_cap: int = 64,
                 max_candidates: int = 8, max_candidates_total: int = 1024,
                 neuron_topk: int = 0,
                 decisions: Tuple[str, ...] = ("split", "frontier", "refine"),
                 call_timeout: Optional[float] = None,
                 logger: Optional[Callable[[WaveRecord], None]] = None):
        self._backend = backend
        self._call_timeout = float(call_timeout) if call_timeout and call_timeout > 0 else None
        self._neuron_topk = max(0, int(neuron_topk))
        self._cadence = max(1, int(cadence))
        self._history: Deque[WaveRecord] = deque(maxlen=max(1, int(history)))
        self._max_failures = max(1, int(max_failures))
        self._multi_split_levels = int(multi_split_levels)
        self._refine_iters_cap = int(refine_iters_cap)
        self._refine_rows_cap_cap = int(refine_rows_cap_cap)
        self._max_candidates = int(max_candidates)
        self._max_candidates_total = int(max_candidates_total)
        self._decisions = set(decisions)
        self._logger = logger
        self._cached_advice = RoundAdvice()
        self._waves_since_call = 0
        self._consecutive_failures = 0
        self._pending: Tuple[RoundAdvice, bool] = (RoundAdvice(), False)
        self._branch_cache: Optional[RoundAdvice] = None
        self._branch_waves_since_call = 0
        self.disabled = False

    @property
    def history(self) -> Deque[WaveRecord]:
        return self._history

    @property
    def wants_neuron(self) -> bool:
        return "neuron" in self._decisions

    @property
    def call_count(self) -> int:
        return self._call_count

    def begin_wave(self, stats: FrontierStats) -> RoundPolicy:
        advice, valid = self._get_advice(stats)
        self._pending = (advice, valid)
        return self._to_policy(advice, stats)

    def end_wave(self, outcome: WaveOutcome) -> None:
        advice, valid = self._pending
        record = WaveRecord(
            wave_index=outcome.wave_index,
            advice=advice,
            outcome=outcome,
            valid_response=valid,
        )
        self._history.append(record)
        if self._logger is not None:
            self._logger(record)

    _call_count = 0

    def _complete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Backend-agnostic HARD wall-clock cap on a single LLM feedback call.

        ``urllib``'s timeout is a per-socket-operation timeout, not a total
        deadline, and different backends bound time differently (subprocess is
        total, HTTP is not). Running ``backend.complete`` in a daemon thread and
        ``join(call_timeout)`` guarantees the BaB loop resumes within
        ``call_timeout`` no matter which backend is used: on expiry we raise so
        the caller falls back to the verifier's baseline for this wave. The
        abandoned daemon thread cannot be killed (Python), but it never blocks
        process exit and dies when the backend's own timeout fires.
        """
        if self._call_timeout is None:
            return self._backend.complete(payload)
        box: Dict[str, Any] = {}
        def _worker() -> None:
            try:
                box["result"] = self._backend.complete(payload)
            except BaseException as exc:  # re-raised on the caller thread below
                box["error"] = exc
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join(self._call_timeout)
        if thread.is_alive():
            raise TimeoutError(f"LLM feedback exceeded {self._call_timeout}s hard deadline")
        if "error" in box:
            raise box["error"]
        return box.get("result", {})

    def _get_advice(self, stats: FrontierStats) -> Tuple[RoundAdvice, bool]:
        if self.disabled:
            return RoundAdvice(), False
        if self._waves_since_call > 0:
            self._waves_since_call -= 1
            return self._cached_advice, True
        try:
            raw = self._complete(self._build_payload(stats))
            advice = RoundAdvice.from_dict(raw)
            self._consecutive_failures = 0
            self._cached_advice = advice
            self._call_count += 1
            horizon = advice.horizon_hint if (advice.horizon_hint and advice.horizon_hint > 0) else self._cadence
            self._waves_since_call = max(1, int(horizon)) - 1
            return advice, True
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_failures:
                self.disabled = True
            return RoundAdvice(), False

    def _to_policy(self, advice: RoundAdvice, stats: FrontierStats) -> RoundPolicy:
        policy = RoundPolicy()
        if "split" in self._decisions and advice.split_k is not None:
            policy.split_k = clip_split_k(
                advice.split_k,
                branch_batch_size=stats.branch_batch_size,
                effective_batch=stats.effective_batch,
                multi_split_levels=self._multi_split_levels,
            )
        if "frontier" in self._decisions and advice.k_requested is not None:
            baseline = min(stats.pool_size, stats.effective_batch, stats.remaining_nodes)
            policy.k_requested = clip_k_requested(
                advice.k_requested,
                baseline=baseline,
                pool_size=stats.pool_size,
                effective_batch=stats.effective_batch,
                remaining_nodes=stats.remaining_nodes,
            )
        if "refine" in self._decisions:
            mode, iters, rows = clip_refine(
                advice.refine_mode,
                advice.refine_iters,
                advice.refine_rows_cap,
                iters_cap=self._refine_iters_cap,
                rows_cap_cap=self._refine_rows_cap_cap,
            )
            policy.refine_mode, policy.refine_iters, policy.refine_rows_cap = mode, iters, rows
        if "input_split" in self._decisions and stats.input_widths:
            policy.input_split_dim, policy.input_split_fanout = clip_input_split(
                advice.input_dim,
                advice.input_fanout,
                n_dims=len(stats.input_widths),
            )
        return policy

    def _recent_aggregates(self) -> Dict[str, Any]:
        gains: List[float] = []
        growth: List[float] = []
        stall = 0
        fallbacks = 0
        for record in self._history:
            outcome = record.outcome
            if outcome.best_lb_before is not None and outcome.best_lb_after is not None:
                gains.append(outcome.best_lb_after - outcome.best_lb_before)
            growth.append(float(outcome.pool_after - outcome.pool_before))
            if outcome.fallback_used or not record.valid_response:
                fallbacks += 1
        for record in reversed(self._history):
            outcome = record.outcome
            if (outcome.best_lb_before is not None and outcome.best_lb_after is not None
                    and (outcome.best_lb_after - outcome.best_lb_before) <= 1e-9):
                stall += 1
            else:
                break
        total = len(self._history)
        return {
            "stall_counter": stall,
            "mean_bound_gain": (sum(gains) / len(gains)) if gains else None,
            "pool_growth": (sum(growth) / len(growth)) if growth else None,
            "fallback_rate": (fallbacks / total) if total else 0.0,
        }

    def _build_payload(self, stats: FrontierStats) -> Dict[str, Any]:
        aggregates = self._recent_aggregates()
        return {
            "wave_index": stats.wave_index,
            "pool_size": stats.pool_size,
            "effective_batch": stats.effective_batch,
            "remaining_nodes": stats.remaining_nodes,
            "branch_batch_size": stats.branch_batch_size,
            "elapsed_s": stats.elapsed_s,
            "remaining_s": stats.remaining_s,
            "depth_min": stats.depth_min,
            "depth_max": stats.depth_max,
            "lower_bound_min": stats.lower_bound_min,
            "lower_bound_max": stats.lower_bound_max,
            "stall_counter": aggregates["stall_counter"],
            "mean_bound_gain_recent": aggregates["mean_bound_gain"],
            "pool_growth_rate_recent": aggregates["pool_growth"],
            "fallback_rate_recent": aggregates["fallback_rate"],
            "candidates": [asdict(c) for c in stats.candidates[: self._max_candidates]],
            "input_widths": stats.input_widths,
            "limits": {
                "multi_split_levels": self._multi_split_levels,
                "refine_iters_cap": self._refine_iters_cap,
                "refine_rows_cap_cap": self._refine_rows_cap_cap,
            },
        }

    def advise_neuron_groups(self, stats: FrontierStats) -> Optional[Dict[int, List[Tuple[int, int]]]]:
        if "neuron" not in self._decisions or self.disabled:
            return None
        # Optional top-K-by-score truncation of the neuron-selection candidate
        # set: smaller payload => faster serialize/inference/parse. Purely a
        # search-efficiency knob, never a soundness one — the verifier still
        # bounds/certifies EVERY subproblem the LLM's chosen group produces.
        # Applied before both the payload build and the legality clip so the
        # LLM only ever picks from, and is only ever validated against, the same
        # truncated set.
        if self._neuron_topk > 0 and len(stats.candidates) > self._neuron_topk:
            top = sorted(stats.candidates, key=lambda c: c.score, reverse=True)[: self._neuron_topk]
            stats = replace(stats, candidates=top)
        total = len(stats.candidates)
        if total == 0 or total > self._max_candidates_total:
            return None
        if self._branch_waves_since_call > 0:
            self._branch_waves_since_call -= 1
            advice = self._branch_cache
        else:
            try:
                raw = self._complete(self._build_branch_payload(stats))
                advice = RoundAdvice.from_dict(raw)
                self._consecutive_failures = 0
                self._branch_cache = advice
                self._branch_waves_since_call = self._cadence - 1
            except Exception:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_failures:
                    self.disabled = True
                return None
        if advice is None or advice.split_groups is None:
            return None
        return clip_split_groups(
            advice.split_groups,
            candidates=stats.candidates,
            branch_batch_size=stats.branch_batch_size,
            effective_batch=stats.effective_batch,
            multi_split_levels=self._multi_split_levels,
        )

    def _build_branch_payload(self, stats: FrontierStats) -> Dict[str, Any]:
        aggregates = self._recent_aggregates()
        return {
            "phase": "neuron_selection",
            "wave_index": stats.wave_index,
            "pool_size": stats.pool_size,
            "remaining_nodes": stats.remaining_nodes,
            "branch_batch_size": stats.branch_batch_size,
            "effective_batch": stats.effective_batch,
            "pool_growth_rate_recent": aggregates["pool_growth"],
            "stall_counter": aggregates["stall_counter"],
            "multi_split_levels": self._multi_split_levels,
            "candidates": [asdict(c) for c in stats.candidates],
        }


def build_frontier_stats(*, wave_index: int, pool_size: int, effective_batch: int,
                         remaining_nodes: int, elapsed_s: float,
                         branch_batch_size: int = 1, remaining_s: Optional[float] = None,
                         depth_min: Optional[int] = None, depth_max: Optional[int] = None,
                         lower_bound_min: Optional[float] = None,
                         lower_bound_max: Optional[float] = None,
                         candidates: Optional[List[CandidateSummary]] = None,
                         input_widths: Optional[List[float]] = None) -> FrontierStats:
    return FrontierStats(
        wave_index=wave_index,
        pool_size=pool_size,
        effective_batch=effective_batch,
        remaining_nodes=remaining_nodes,
        elapsed_s=elapsed_s,
        branch_batch_size=branch_batch_size,
        remaining_s=remaining_s,
        depth_min=depth_min,
        depth_max=depth_max,
        lower_bound_min=lower_bound_min,
        lower_bound_max=lower_bound_max,
        candidates=list(candidates) if candidates else [],
        input_widths=input_widths,
    )


def _make_jsonl_logger() -> Callable[[WaveRecord], None]:
    from act.util.path_config import get_pipeline_log_dir

    log_dir = os.path.join(get_pipeline_log_dir(), "llm_probe")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "llm_probe_waves.jsonl")

    def _log(record: WaveRecord) -> None:
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record)) + "\n")
        except OSError:
            pass

    return _log


_PROVIDER_PRESETS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPUAI_API_KEY"),
    "minimax": ("https://api.minimaxi.com/v1", "MINIMAX_API_KEY"),
}


def build_llm_probe(config: Any) -> Optional[LLMProbe]:
    if not getattr(config, "llm_probe_enabled", False):
        return None
    backend_name = getattr(config, "llm_probe_backend", "mock")
    timeout = getattr(config, "llm_probe_timeout", 30.0)
    if backend_name == "claude_cli":
        backend: LLMBackend = ClaudeCLIBackend(
            model=getattr(config, "llm_probe_model", "") or "sonnet",
            timeout=timeout,
        )
    elif backend_name in _PROVIDER_PRESETS:
        preset_url, preset_env = _PROVIDER_PRESETS[backend_name]
        base_url = getattr(config, "llm_probe_base_url", "") or preset_url
        api_key_env = getattr(config, "llm_probe_api_key_env", "") or preset_env
        backend = OpenAICompatibleBackend(
            base_url=base_url,
            model=getattr(config, "llm_probe_model", ""),
            api_key=os.environ.get(api_key_env, ""),
            temperature=getattr(config, "llm_probe_temperature", 0.0),
            timeout=timeout,
        )
    else:
        backend = MockBackend()
    decisions = tuple(
        token.strip()
        for token in getattr(config, "llm_probe_decisions", "split,frontier,refine").split(",")
        if token.strip()
    )
    logger = _make_jsonl_logger() if getattr(config, "llm_probe_log", False) else None
    return LLMProbe(
        backend,
        cadence=getattr(config, "llm_probe_cadence", 1),
        history=getattr(config, "llm_probe_history", 8),
        max_failures=getattr(config, "llm_probe_max_failures", 3),
        multi_split_levels=getattr(config, "multi_split_levels", 1),
        refine_iters_cap=getattr(config, "per_subproblem_refine_iters", 0),
        refine_rows_cap_cap=getattr(config, "per_subproblem_refine_rows_cap", 64),
        max_candidates=getattr(config, "llm_probe_max_candidates", 8),
        max_candidates_total=getattr(config, "llm_probe_max_candidates_total", 1024),
        neuron_topk=getattr(config, "llm_probe_neuron_topk", 512),
        decisions=decisions,
        call_timeout=timeout,
        logger=logger,
    )
