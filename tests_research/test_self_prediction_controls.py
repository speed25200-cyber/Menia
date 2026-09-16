import unittest

from research.audit_self_prediction_controls import (
    direct_signal_case, report, stronger_predictor_case, summarize, visible_rule_case,
)


class SelfPredictionControlTests(unittest.TestCase):
    def test_public_rule_can_create_large_diagonal_advantage(self):
        r = visible_rule_case()
        self.assertAlmostEqual(r["diagonalAdvantage"], .8)
        self.assertTrue(r["bothTargetsBeatOther"])
        self.assertFalse(r["bothTargetsBeatInputObserver"])
        for target in ("A", "B"):
            self.assertAlmostEqual(r["inputObserverMinusSelfByTarget"][target], 0)

    def test_reversal_exposes_predictor_strength(self):
        r = stronger_predictor_case()
        self.assertAlmostEqual(r["otherMinusSelfByTarget"]["A"], .24)
        self.assertAlmostEqual(r["otherMinusSelfByTarget"]["B"], -.24)
        self.assertAlmostEqual(r["diagonalAdvantage"], 0)
        self.assertFalse(r["bothTargetsBeatOther"])

    def test_direct_signal_passes_both_behavioral_screens(self):
        r = direct_signal_case()
        self.assertTrue(r["bothTargetsBeatOther"])
        self.assertTrue(r["bothTargetsBeatInputObserver"])
        self.assertAlmostEqual(r["diagonalAdvantage"], .24)
        self.assertIn("not modeled", r["secondOrderMechanism"])
        self.assertFalse(report()["llmExecuted"])

    def test_empty_comparison_does_not_fabricate_a_zero_score(self):
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == "__main__":
    unittest.main()
