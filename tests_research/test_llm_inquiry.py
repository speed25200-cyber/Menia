import tempfile
import unittest
import numpy as np
from research.llm_inquiry import (points, replay_until, gains, summarize, verdicts, ScriptedDigits, scripted_symbols, Cached,
                                  evaluate, main)
from research.llm_atelier import MARKS


class InquiryTests(unittest.TestCase):
    def test_points_and_replay(self):
        pts = points(4, 3)
        self.assertEqual(sum(p["set"] == "R" for p in pts), 4)
        self.assertTrue(all(p["step"] == 0 for p in pts if p["set"] == "R"))
        history, p, g = replay_until(pts[-1]["life_record"], pts[-1]["step"])
        self.assertEqual(len(history), pts[-1]["step"])

    def test_a_model_that_knows_the_mark_prefers_place_one(self):
        digits, symbols = Cached(ScriptedDigits("mark")), Cached(scripted_symbols)
        result = gains(digits, symbols, [], 3, 5, 0)
        self.assertGreater(result["gains"][1], 0.9)
        self.assertAlmostEqual(result["gains"][2], 0.0, places=6)

    def test_a_fixed_body_model_has_no_preferred_place(self):
        rows = evaluate(Cached(ScriptedDigits("fixed")), Cached(scripted_symbols), points(4, 2), log=lambda m: None)
        self.assertTrue(all(r["best"] == 0 for r in rows))
        summary = summarize(rows)
        self.assertEqual(summary["start_mark_best"], 0.0)

    def test_verdicts(self):
        mark = summarize(evaluate(Cached(ScriptedDigits("mark")), Cached(scripted_symbols), points(4, 2), log=lambda m: None))
        fixed = summarize(evaluate(Cached(ScriptedDigits("fixed")), Cached(scripted_symbols), points(4, 2), log=lambda m: None))
        out = verdicts({"VM": mark, "F": fixed})
        self.assertTrue(out["I1"] and out["I2"] and out["global"])

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            main(["--out", tmp, "--r-lives", "2", "--m-lives", "1"])


if __name__ == "__main__":
    unittest.main()
