"""Run ACT's batched BaB directly on a network saved by ACT-GUI.

From the ACT checkout:
    .venv/bin/python scripts/run_gui_bab.py
    .venv/bin/python scripts/run_gui_bab.py /path/to/network.json --time-budget 60
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


ACT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    ACT_ROOT.parent / "ACT-GUI" / "var" / "models" / "local-bab-deep-067b04b2ad0e.json"
)
sys.path.insert(0, str(ACT_ROOT))


class ProgressTrace(list[dict[str, Any]]):
    """Print each BaB wave as soon as the verifier appends its trace row."""

    def append(self, row: dict[str, Any]) -> None:
        super().append(row)
        print(
            f"wave {row['iter']:4d} | batch {row['K']:3d} | "
            f"frontier {row['pool_before']:4d} -> {row['pool_after']:4d} | "
            f"UNSAT {row['pruned_unsat']:3d} | branched {row['branched']:3d} | "
            f"max_depth {row['max_depth']:2d} | elapsed {row['elapsed_s']:.3f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--time-budget", type=float, default=30.0, metavar="SECONDS")
    parser.add_argument("--solver-tier", choices=("lp", "dual", "dual_alpha", "dual_alpha_eta"), default="lp")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"model file does not exist: {args.model}")
    if not 0 < args.time_budget <= 3600:
        parser.error("--time-budget must be in (0, 3600]")

    from act.back_end.bab.bab import verify_bab_batched
    from act.back_end.serialization.serialization import load_net_from_file
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.back_end.transfer_functions import set_solver_mode, set_transfer_function_mode
    from act.config.config import BackendConfig, PipelineConfig

    net = load_net_from_file(str(args.model.resolve()))
    set_solver_mode(None)
    set_transfer_function_mode("interval")
    pipeline_config = PipelineConfig.from_yaml()
    bab_config = replace(pipeline_config.bab, solver_tier=args.solver_tier)
    started = time.perf_counter()
    progress = ProgressTrace()
    print(f"running BaB on {args.model.resolve()} (solver_tier={args.solver_tier})", flush=True)
    result = verify_bab_batched(
        net=net,
        solver_factory=TorchLPSolver,
        config=bab_config,
        max_batch_size=BackendConfig.from_yaml().bab_max_batch_size,
        time_budget_s=args.time_budget,
        dual_config=pipeline_config.dual,
        trace=progress,
    )

    print(f"model: {args.model.resolve()}")
    print(f"status: {result.status.name}")
    print(f"elapsed_s: {time.perf_counter() - started:.4f}")
    print(f"metadata: {result.metadata}")
    if result.counterexample is not None:
        print(f"counterexample: {result.counterexample.detach().cpu().reshape(-1).tolist()}")


if __name__ == "__main__":
    main()
