import json
import tempfile
import unittest
from pathlib import Path
from research.llm_mark_reading import items, prompt, score, summarize, verdicts, main
from research.llm_inquiry import ScriptedDigits


class ReadingTests(unittest.TestCase):
    def test_prompts_cover_places_symbols_commands_positions(self):
        grid = items()
        self.assertEqual(len(grid), 256)
        text = prompt({"place": 1, "symbol": 3, "command": 2, "p": 6})
        self.assertIn("\nTour 1 : inspection du lieu 1, symbole □. Position 6, cible 2.\n", text)
        self.assertTrue(text.endswith("Tour 2 : commande C, de la case 6 à la case "))

    def test_a_reader_scores_high_on_the_mark_only_and_a_blind_model_scores_a_quarter(self):
        reader, blind = summarize(score(ScriptedDigits("mark"))), summarize(score(ScriptedDigits("fixed")))
        self.assertGreater(reader["P1"], 0.99)
        self.assertAlmostEqual(reader["P0"], 0.25, places=6)
        self.assertAlmostEqual(blind["P1"], 0.25, places=6)
        self.assertAlmostEqual(blind["P0"], 0.25, places=6)

    def test_verdicts(self):
        reader = {"digit_mass": 0.99, "P1": 0.7, "P0": 0.26}
        blind = {"digit_mass": 0.99, "P1": 0.26, "P0": 0.25}
        v = verdicts({"F": blind, "VM": blind, "VMI": blind, "VMW": blind, "VML": reader})
        self.assertEqual((v["L0"], v["L1"], v["L2"], v["global"]), (True, True, None, False))
        v = verdicts({"VMW": dict(blind, P1=0.4), "VML": dict(reader, P0=0.5)})
        self.assertEqual((v["L0"], v["L1"]), (False, False))
        v = verdicts({"VMW": dict(blind, digit_mass=0.2), "VML": dict(reader, digit_mass=0.2)})
        self.assertEqual((v["valid_VMW"], v["L0"], v["L1"]), (False, None, None))

    def test_cli_scores_and_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            main(["score", "--out", tmp + "/r", "--label", "VML", "--scripted", "mark"])
            main(["score", "--out", tmp + "/f", "--label", "F", "--scripted", "fixed"])
            main(["verdicts", "--reading", f"F={tmp}/f", f"VML={tmp}/r", "--output", tmp + "/v.json"])
            main(["verdicts", "--reading", f"F={tmp}/f", f"VML={tmp}/r", "--check", tmp + "/v.json"])
            self.assertTrue(json.loads(Path(tmp, "v.json").read_text())["verdicts"]["L1"])


if __name__ == "__main__":
    unittest.main()
