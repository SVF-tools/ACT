"""Record BaB behavior before and after the wave-result refactor."""

import json
import tempfile
import unittest
from unittest.mock import patch

import torch

import act.back_end.bab.bab as bab
from act.back_end.analyze import initialize_tf_mode
from act.back_end.net_factory import LAYER_TESTING_SPECS, NetFactory
from act.back_end.solver.solver_torchlp import TorchLPSolver
from act.config.config import BackendConfig, BaBConfig, DualConfig, TorchLPConfig
from act.util.device_manager import initialize_device
from act.util.stats import VerifyStatus


class WaveBoundIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_device("cpu", "float64")
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory(prefix="act-wave-bound-")
        cls.addClassCleanup(cls.directory.cleanup)
        factory = NetFactory(
            BackendConfig.from_yaml().generation.net_factory,
            output_dir=cls.directory.name, write_manifest=False,
        )
        # Existing fixture: duplicate ReLUs cancel, so the property is safe,
        # while their independent relaxation leaves the root unresolved.
        cls.net = factory.create_network(
            "wave_bound", LAYER_TESTING_SPECS["layer_testing_climb_reuse"](),
        )

    def tearDown(self):
        initialize_tf_mode("interval")

    def report_case(self, name, result, trace):
        # Exclude elapsed time; keep deterministic search decisions for comparison.
        report = {
            "case": name,
            "status": result.status.name,
            "metadata": {key: result.metadata.get(key) for key in (
                "reason", "nodes", "iterations", "pool_remaining",
            )},
            "trace": [{key: row[key] for key in (
                "K", "branched", "pruned_unsat", "pool_after",
            )} for row in trace],
        }
        print("BAB_BEHAVIOR " + json.dumps(report, sort_keys=True), flush=True)

    def assert_one_unresolved_wave(self, result, trace):
        self.assertEqual(result.status, VerifyStatus.UNKNOWN)
        self.assertEqual(result.metadata["reason"], "max_depth_unproved_retirement")
        self.assertEqual(result.metadata["nodes"], 1)
        self.assertEqual(result.metadata["iterations"], 1)
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["K"], 1)
        self.assertEqual(trace[0]["branched"], 0)

    def test_lp_keeps_forward_modeling_inside_the_existing_pipeline(self):
        for mode in ("interval", "hybridz"):
            with self.subTest(forward=mode):
                initialize_tf_mode(mode)
                problems = []

                class RecordingSolver(TorchLPSolver):
                    def solve_batch(self, problem, timelimit=None):
                        problems.append(problem)
                        return super().solve_batch(problem, timelimit=timelimit)

                def make_solver():
                    return RecordingSolver(TorchLPConfig(max_iter=20))

                trace = []
                with patch.object(bab, "setup_and_solve_batch",
                                  wraps=bab.setup_and_solve_batch) as solve:
                    result = bab.verify_bab_batched(
                        self.net, make_solver, BaBConfig(
                            solver_tier="lp", branching_method="width",
                            max_depth=0, max_nodes=1,
                        ),
                        max_batch_size=1, time_budget_s=30, trace=trace,
                    )
                self.report_case(f"lp/{mode}", result, trace)
                self.assert_one_unresolved_wave(result, trace)
                solve.assert_called_once()
                self.assertEqual(len(problems), 1)
                self.assertEqual(problems[0].N, 1)
                self.assertGreater(problems[0].m_eq + problems[0].m_le, 0)

    def test_dual_variants_complete_a_search_after_real_splits(self):
        for tier in ("dual", "dual_alpha", "dual_alpha_eta"):
            with self.subTest(tier=tier):
                torch.manual_seed(42)
                initialize_tf_mode("interval")
                trace = []
                result = bab.verify_bab_batched(
                    self.net, bab.DualSolver, BaBConfig(
                        solver_tier=tier, branching_method="babsr", bounding="random",
                        max_depth=3, max_nodes=15,
                    ),
                    max_batch_size=2, time_budget_s=30,
                    dual_config=DualConfig(n_iters=5), trace=trace,
                )
                self.report_case(f"search/{tier}", result, trace)
                # The fixture's two identical ReLUs cancel exactly.
                self.assertEqual(result.status, VerifyStatus.CERTIFIED)
                self.assertEqual(result.metadata["pool_remaining"], 0)
                self.assertLessEqual(result.metadata["nodes"], 15)
                self.assertGreater(len(trace), 1)
                self.assertEqual(trace[0]["K"], 1)
                self.assertEqual(trace[0]["branched"], 1)
                self.assertTrue(any(row["K"] == 2 for row in trace))
                self.assertGreater(sum(row["pruned_unsat"] for row in trace), 0)

    def test_dual_variants_share_the_existing_dual_path(self):
        for tier in ("dual", "dual_alpha", "dual_alpha_eta"):
            with self.subTest(tier=tier):
                trace = []
                with patch.object(bab, "_solve_dual_batch",
                                  wraps=bab._solve_dual_batch) as solve:
                    result = bab.verify_bab_batched(
                        self.net, bab.DualSolver, BaBConfig(
                            solver_tier=tier, branching_method="babsr",
                            max_depth=0, max_nodes=1,
                        ),
                        max_batch_size=1, time_budget_s=30,
                        dual_config=DualConfig(n_iters=1), trace=trace,
                    )
                self.report_case(f"dispatch/{tier}", result, trace)
                self.assert_one_unresolved_wave(result, trace)
                self.assertGreaterEqual(solve.call_count, 2)  # presolve and wave
                for call in solve.call_args_list:
                    self.assertEqual(call.kwargs["config"].solver_tier, tier)
                    self.assertEqual(call.kwargs["optimize"], tier != "dual")


if __name__ == "__main__":
    unittest.main()
