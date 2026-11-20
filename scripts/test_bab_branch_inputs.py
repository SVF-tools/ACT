"""Check lane alignment at the BaB branching boundary."""

import unittest

import torch

from act.back_end.bab.wave import _BoundWave
from act.back_end.bab.node import SubproblemBatch
from act.back_end.core import Bounds
from act.back_end.solver.solver_base import BatchLPSolution
from act.back_end.solver.solver_dual import DualBatchResult


class BranchInputsTests(unittest.TestCase):
    def make_wave(self, dual=False):
        batch = SubproblemBatch(
            lb=torch.zeros(3, 1), ub=torch.ones(3, 1),
            depths=torch.zeros(3, dtype=torch.long),
        )
        solution = BatchLPSolution(
            statuses=("UNKNOWN",) * 3, x=torch.zeros(3, 1),
            max_viol=torch.ones(3),
        )
        if not dual:
            return _BoundWave.from_lp(batch, solution)
        bounds = torch.tensor([[10.], [20.], [30.]])
        result = DualBatchResult(
            solution, torch.zeros(3, 2), torch.zeros(3),
            bounds_dict={7: Bounds(bounds, bounds + 1)},
            nu_per_layer={7: torch.tensor([[100.], [101.], [200.], [201.], [300.], [301.]])},
            witness_input=torch.tensor([[1.], [2.], [3.]]), m_specs=2,
        )
        return _BoundWave.from_dual(batch, result)

    def test_lp_keeps_processed_batch_without_dual_evidence(self):
        wave = self.make_wave()
        rows = torch.tensor([2, 0])
        batch = wave.batch.select(rows)
        batch.lower_bound = torch.tensor([17., 19.])
        inputs = wave.for_branching(batch, rows, include_witness=True)
        self.assertIs(inputs.batch, batch)
        self.assertEqual(inputs.batch.lower_bound.tolist(), [17., 19.])
        self.assertIsNone(wave.dual)
        self.assertIsNone(inputs.bounds)
        self.assertIsNone(inputs.sensitivities)
        self.assertIsNone(inputs.witness_input)

    def test_dual_aligns_packed_specs_and_witness_without_rebuilding_batch(self):
        wave = self.make_wave(dual=True)
        rows = torch.tensor([2, 0])
        batch = wave.batch.select(rows)
        batch.lower_bound = torch.tensor([17., 19.])
        inputs = wave.for_branching(batch, rows, include_witness=True)
        self.assertIs(inputs.batch, batch)
        self.assertEqual(inputs.batch.lower_bound.tolist(), [17., 19.])
        self.assertEqual(inputs.bounds[7].lb.tolist(), [[30.], [10.]])
        self.assertEqual(inputs.sensitivities[7].tolist(), [[300.], [301.], [100.], [101.]])
        self.assertEqual(inputs.witness_input.tolist(), [[3.], [1.]])
        self.assertIsNone(wave.for_branching(batch, rows).witness_input)
        self.assertEqual(wave.dual.result.witness_input.tolist(), [[1.], [2.], [3.]])

    def test_empty_selection(self):
        wave = self.make_wave(dual=True)
        rows = torch.empty(0, dtype=torch.long)
        inputs = wave.for_branching(wave.batch.select(rows), rows, include_witness=True)
        self.assertEqual(inputs.bounds[7].lb.shape, (0, 1))
        self.assertEqual(inputs.sensitivities[7].shape, (0, 1))
        self.assertEqual(inputs.witness_input.shape, (0, 1))

    def test_rejects_lane_count_mismatch(self):
        wave = self.make_wave()
        with self.assertRaisesRegex(ValueError, "lane count"):
            _BoundWave.from_lp(wave.batch.select(torch.tensor([0])), wave.solution)
        with self.assertRaisesRegex(ValueError, "selected batch"):
            wave.for_branching(wave.batch, torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
