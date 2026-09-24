import json
import tempfile
import unittest
from pathlib import Path
from research.llm_mark_action import run, summarize, paired_difference, play, main
from research.llm_inquiry import Cached, ScriptedDigits


class ActionTests(unittest.TestCase):
    def test_a_reader_earns_points_from_the_mark_and_loses_them_under_lesion(self):
        rows = run(Cached(ScriptedDigits("mark")), "reader", 24)
        s = summarize(rows)
        self.assertAlmostEqual(s["intact"]["hit_when_reachable"], 0.85, delta=0.1)
        self.assertLess(s["lesion"]["hit_when_reachable"], 0.4)
        intact = [r for r in rows if r["condition"] == "intact"]
        lesion = [r for r in rows if r["condition"] == "lesion"]
        self.assertGreater(paired_difference(intact, lesion)["low"], 3)

    def test_a_blind_model_gains_nothing_from_the_mark(self):
        rows = run(Cached(ScriptedDigits("fixed")), "blind", 24)
        intact = [r for r in rows if r["condition"] == "intact"]
        lesion = [r for r in rows if r["condition"] == "lesion"]
        self.assertLess(abs(paired_difference(intact, lesion)["mean"]), 1.0)

    def test_lives_are_the_same_across_conditions_and_models(self):
        _, a = play(ScriptedDigits("mark"), 3, "intact")
        _, b = play(ScriptedDigits("fixed"), 3, "lesion")
        self.assertEqual(len(a), 12)
        self.assertEqual(a[0]["body"], b[0]["body"])

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            main(["--out", tmp + "/code", "--label", "code", "--lives", "12"])
            main(["--out", tmp + "/control", "--label", "control", "--lives", "12", "--scripted", "fixed",
                  "--conditions", "intact"])
            summary = json.loads(Path(tmp, "code", "summary.json").read_text())
            self.assertEqual(set(summary), {"intact", "lesion"})
            main(["verdicts", "--code", tmp + "/code", "--control", tmp + "/control", "--output", tmp + "/v.json"])
            main(["verdicts", "--code", tmp + "/code", "--control", tmp + "/control", "--check", tmp + "/v.json"])
            v = json.loads(Path(tmp, "v.json").read_text())["verdicts"]
            self.assertEqual((v["U1"], v["U2"], v["global"]), (True, True, True))


if __name__ == "__main__":
    unittest.main()
