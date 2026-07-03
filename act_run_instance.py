#!/usr/bin/env python3
"""ACT VNN-COMP 2026 single-instance runner.

Wraps the ACT pipeline (arbitrary onnx+vnnlib load, ACTFuzzer mutation pre-attack for
FALSIFICATION, dual_alpha_eta BaB for CERTIFICATION) and emits the VNN-COMP result contract:

    line 1 : unsat | sat | timeout | unknown
    if sat : lines 2+ = the counterexample as a VNNLIB-1.0 flat assignment
             ``((X_0 v)...(Y_0 v)...)`` (see vnnlib_parser.write_vnncomp_result).

Both the verifier's CE (result.counterexample) and the fuzzer's CE (.input) are
in model input space; the output Y is (re)computed by running the raw
ONNX-converted model on the CE input, so it always matches the network the
harness re-executes. Invoked by run_instance.sh as:

    python act_run_instance.py <onnx> <vnnlib> <results_file> <timeout_s> [opts]

Tier homes: onnx+vnnlib loading and CE emission live in front_end
(vnnlib_loader.create_specs_from_paths / vnnlib_parser.write_vnncomp_result),
the ReluSplitter inverse in front_end.model_synthesis, the BaB profile in
back_end.config.build_vnncomp_bab_config, and the PGD pre-attack in
pipeline.fuzzing.actfuzzer.pgd_preattack.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

import torch

from act.util.device_manager import initialize_device


def main() -> None:
    ap = argparse.ArgumentParser(description="ACT VNN-COMP 2026 single-instance runner")
    ap.add_argument("onnx")
    ap.add_argument("vnnlib")
    ap.add_argument("output")
    ap.add_argument("timeout", type=float)
    ap.add_argument("--config", default="gain", choices=["fsb", "babsr", "gain", "gain+llm"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--fuzzing-seconds", type=float, default=10.0)
    ap.add_argument("--fuzzing-scale", type=float, default=0.5)
    ap.add_argument("--max-batch-size", default="auto",
                    help="int or 'auto' (net/GPU-aware, avoids OOM)")
    ap.add_argument("--margin", type=float, default=5.0,
                    help="seconds reserved before the harness kill (timeout+60)")
    ap.add_argument("--llm-backend", default="openrouter")
    ap.add_argument("--llm-model", default="google/gemini-2.5-flash-lite")
    ap.add_argument("--llm-timeout", type=float, default=30.0,
                    help="per-call LLM wall-clock cap; a slower reply falls back to baseline")
    ap.add_argument("--solver-tier", default="auto",
                    choices=["auto", "lp", "dual", "dual_alpha", "dual_alpha_eta"],
                    help="'auto' = cheap one-shot 'dual' bound, then escalate to 'dual_alpha_eta'")
    ap.add_argument("--input-split-dims", type=int, default=10,
                    help="input dimension threshold at or below which BaB switches to "
                         "input-domain splitting with full per-node bound recomputation "
                         "(the ACAS Xu regime); 0 disables the profile")
    args = ap.parse_args()

    cuda_ok = torch.cuda.is_available()
    print(f"[env] torch={torch.__version__} cuda_build={torch.version.cuda} "
          f"cuda_available={cuda_ok} "
          f"device_count={torch.cuda.device_count()} resolved_device={args.device}",
          file=sys.stderr, flush=True)
    if not cuda_ok:
        print("[env] WARNING: running on CPU - expect timeouts. torch cannot see a GPU; "
              "check the NVIDIA driver (nvidia-smi must work).",
              file=sys.stderr, flush=True)

    t0 = time.time()
    initialize_device(args.device, args.dtype)

    from act.back_end.bab.bab import clear_violation_check_module_cache, verify_bab_batched
    from act.back_end.config import build_vnncomp_bab_config
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.front_end.model_synthesis import merge_split_relus
    from act.front_end.model_synthesis import synthesize_models_from_specs
    from act.front_end.vnnlib_loader.create_specs import create_specs_from_paths
    from act.front_end.vnnlib_loader.vnnlib_parser import write_vnncomp_result
    from act.pipeline.fuzzing.actfuzzer import pgd_preattack
    from act.pipeline.verification.torch2act import TorchToACT
    from act.util.stats import VerifyStatus

    def remaining() -> float:
        return args.timeout - (time.time() - t0) - args.margin

    try:
        sr = create_specs_from_paths(args.onnx, args.vnnlib)
    except SystemExit as exc:
        print(f"[load failed] {exc}", file=sys.stderr)
        write_vnncomp_result(args.output, "unknown")
        return

    raw_model = sr[2]
    param = next(raw_model.parameters(), None)

    input_dim = int(sr[3][0].tensor.numel()) if sr[3] else 0
    low_dim = 0 < input_dim <= args.input_split_dims
    if low_dim:
        print(f"[profile] input_dim={input_dim} <= {args.input_split_dims}: "
              f"input-split BaB with per-node bound recompute", file=sys.stderr, flush=True)

    def raw_forward(x):
        if param is not None:
            x = x.to(device=param.device, dtype=param.dtype)
        with torch.no_grad():
            return raw_model(x)

    verify_model, n_merged = merge_split_relus(raw_model)
    if n_merged:
        print(f"[merge] fused {n_merged} split-ReLU neurons", file=sys.stderr, flush=True)
        sr = tuple(verify_model if i == 2 else v for i, v in enumerate(sr))

    # A VNNLIB 2.0 top-1 OR that did NOT pattern-merge (front-end Layer 1) yields
    # N disjunct models here; the instance is unsat only if EVERY disjunct is
    # infeasible and sat if ANY single disjunct is reachable. Verifying just the
    # first would be unsound. N == 1 reproduces the original single-model path
    # exactly (per-model budgets collapse to remaining()).
    wrapped_models = list(synthesize_models_from_specs([sr]).values())
    n_models = len(wrapped_models)

    if args.fuzzing_seconds > 0 and remaining() > 1.0:
        per_model_fuzz = min(args.fuzzing_seconds, remaining() / n_models)
        for wm in wrapped_models:
            if remaining() <= 1.0:
                break
            try:
                ce, _ = pgd_preattack(wm, sr[3], min(per_model_fuzz, remaining()), args.fuzzing_scale)
            except Exception as exc:
                ce = None
                print(f"[attack skipped] {exc}", file=sys.stderr)
            if ce is not None:
                x = ce.input if hasattr(ce, "input") else ce
                write_vnncomp_result(args.output, "sat", x=x, y=raw_forward(x))
                return

    if remaining() <= 1.0:
        write_vnncomp_result(args.output, "timeout")
        return

    def _verify_wrapped(wrapped, deadline):
        """Auto/escalation BaB flow on ONE disjunct model, capped at ``deadline``
        (wall-clock). When n_models == 1, deadline is set just past the global
        margin deadline so budget_left() == remaining() and this path is
        byte-identical to the original single-model logic."""
        net = TorchToACT(wrapped).run()

        def budget_left():
            return min(remaining(), deadline - time.time())

        def _verify(tier, budget):
            cfg = build_vnncomp_bab_config(args.config, llm_backend=args.llm_backend, llm_model=args.llm_model,
                                           llm_timeout=args.llm_timeout, solver_tier=tier)
            if low_dim:
                # Low-dim regime (ACAS Xu-style): bisect the input domain and recompute
                # every child's intermediate bounds on its own sub-box. Neuron splits and
                # frozen root bounds - the large-net defaults - certify nothing here: the
                # branching gain of an input split lives entirely in the recomputed
                # intermediate relaxations. Uncapped frontier, since any eviction makes
                # certification permanently impossible for the run.
                cfg.branching_method = "width"
                cfg.multi_split_levels = 1
                cfg.reuse_root_bounds = False
                cfg.intermediate_refine = "none"
                cfg.frontier_cap = 0
            clear_violation_check_module_cache()
            return verify_bab_batched(net, solver_factory=TorchLPSolver, config=cfg,
                                      max_batch_size=args.max_batch_size, time_budget_s=max(1.0, budget))

        if args.solver_tier == "auto":
            if low_dim:
                # With input splits + per-node recompute, the cheap one-shot 'dual'
                # bound is the workhorse (ACAS Xu prop_1 certifies in ~0.1s / 500
                # nodes vs 17s with alpha+eta); escalate only if it can't close.
                res = _verify("dual", budget_left())
                if res.status not in (VerifyStatus.CERTIFIED, VerifyStatus.FALSIFIED) and budget_left() > 1.0:
                    res = _verify("dual_alpha_eta", budget_left())
            else:
                # The one-shot 'dual' bound certifies tight nets (e.g. ViT attention) at the
                # root in ~0.2s; escalate to the iterative alpha+eta tier + BaB only if
                # still UNKNOWN.
                res = _verify("dual", min(budget_left(), 15.0))
                if res.status not in (VerifyStatus.CERTIFIED, VerifyStatus.FALSIFIED) and budget_left() > 1.0:
                    res = _verify("dual_alpha_eta", budget_left())
        else:
            res = _verify(args.solver_tier, budget_left())
        return res

    # Aggregate across disjuncts. Soundness (N > 1): 'unsat' requires EVERY
    # disjunct CERTIFIED; one FALSIFIED disjunct with a CE gives 'sat'; a
    # genuinely inconclusive verdict gives 'unknown'; otherwise the shortfall is
    # the clock -> 'timeout'.
    statuses = []
    falsified_ce = None
    try:
        unfinished = n_models
        for wm in wrapped_models:
            if remaining() <= 1.0:
                break
            deadline = time.time() + remaining() / max(1, unfinished)
            res = _verify_wrapped(wm, deadline)
            unfinished -= 1
            statuses.append(res.status)
            if res.status == VerifyStatus.FALSIFIED and res.counterexample is not None:
                falsified_ce = res.counterexample
                break
    except Exception as exc:
        print(f"[verify error] {exc}", file=sys.stderr)
        write_vnncomp_result(args.output, "unknown")
        return

    if falsified_ce is not None:
        write_vnncomp_result(args.output, "sat", x=falsified_ce, y=raw_forward(falsified_ce))
    elif len(statuses) == n_models and all(s == VerifyStatus.CERTIFIED for s in statuses):
        write_vnncomp_result(args.output, "unsat")
    elif any(s not in (VerifyStatus.CERTIFIED, VerifyStatus.TIMEOUT) for s in statuses):
        write_vnncomp_result(args.output, "unknown")
    else:
        write_vnncomp_result(args.output, "timeout")


if __name__ == "__main__":
    main()
