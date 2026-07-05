#!/usr/bin/env python3
"""ACT VNN-COMP 2026 single-instance runner (thin CLI entrypoint).

Wraps the ACT pipeline (arbitrary onnx+vnnlib load, ACTFuzzer mutation pre-attack for
FALSIFICATION, dual_alpha_eta BaB for CERTIFICATION) and emits the VNN-COMP result contract:

    line 1 : unsat | sat | timeout | unknown
    if sat : lines 2+ = the counterexample as a VNNLIB 2.0 command-line assignment
             (per-variable ``<name> <dtype> [shape]`` header + row-major values;
             see vnnlib_parser.write_vnncomp_result).

Invoked by run_instance.sh as:

    python act_run_instance.py <onnx> <vnnlib> <results_file> <timeout_s> [opts]

This entrypoint only parses CLI arguments; the orchestration (loading, PGD pre-attack,
auto/escalation BaB, disjunct aggregation, CE emission) lives in
act.pipeline.verification.vnncomp_runner.run_vnncomp_instance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# This runner lives in <repo>/vnncomp/; the repo root (which holds the `act`
# package) is one level up. Put it on sys.path so `import act` resolves without
# an editable install.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import torch

from act.pipeline.verification.vnncomp_runner import run_vnncomp_instance


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
    run_vnncomp_instance(args)


if __name__ == "__main__":
    main()
