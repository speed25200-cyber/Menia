import unittest
import numpy as np
from research.audit_confidence_experience import fit_measure, paired_measure, summarize


class ConfidenceExperienceTests(unittest.TestCase):
    def test_distinct_measurements_recover_known_thresholds(self):
        length = np.repeat(np.arange(370, 431, 10), 3)
        confidence = .2 + .0001*(length-387.3)**2
        reproduction = 400 + .8*(length-412.5)
        c = fit_measure(length, confidence, 2)
        r = fit_measure(length, reproduction, 1)
        self.assertAlmostEqual(c["threshold"], 387.3)
        self.assertAlmostEqual(c["unbounded"], 387.3)
        self.assertAlmostEqual(r["threshold"], 412.5)
        self.assertFalse(r["outside_or_at_grid_boundary"])

    def test_confidence_boundary_is_recorded_not_hidden(self):
        length = np.arange(370, 431, 10)
        result = fit_measure(length, .2+.00001*(length-275)**2, 2)
        self.assertTrue(result["outside_or_at_grid_boundary"])
        self.assertEqual(result["threshold"], 320)
        self.assertAlmostEqual(result["unbounded"], 275)

    def test_pair_orientation_is_short_minus_long(self):
        short = {"confidence": {"threshold": 410}, "reproduction": {"threshold": 400}}
        long = {"confidence": {"threshold": 390}, "reproduction": {"threshold": 405}}
        self.assertEqual(paired_measure(short, long), {"confidence_shift": 20, "reproduction_shift": -5})

    def test_stratified_bootstrap_preserves_experiment_sizes(self):
        rows = [{"experiment": "concurrent", "confidence_shift": 4., "reproduction_shift": 1.}]*3
        rows += [{"experiment": "delayed", "confidence_shift": 12., "reproduction_shift": 1.}]
        result = summarize(rows, 2, draws=100)
        self.assertEqual(result["participants"], 4)
        self.assertEqual(result["metrics"]["confidence_shift"]["mean"], 6.)
        self.assertEqual(result["metrics"]["difference"]["percentile_95"], [5., 5.])


if __name__ == "__main__":
    unittest.main()
