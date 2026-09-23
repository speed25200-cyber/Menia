import json
import tempfile
import unittest
from pathlib import Path
from research.llm_mark_reading import items, prompt, score, summarize, verdicts, main, by_command, hand_verdicts, long_verdicts
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

    def test_hand_verdicts_read_the_preferred_command_apart(self):
        rows = score(ScriptedDigits("mark"))
        for r in rows:  # a reader of command A only, which applies A's landing to every command
            if r["place"] == 1 and r["command"] != 0:
                r["p_implied"] = 0.05
        summary, commands = summarize(rows), by_command(rows)
        self.assertGreater(commands["P1_A"], 0.99)
        self.assertAlmostEqual(commands["P1_BCD"], 0.05)
        seeker = {"digit_mass": 0.99, "symbol_mass": 0.99, "start_mark_best": 0.9,
                  "start_mean_gain": {1: 0.3, 2: 0.0, 3: 0.0, 4: 0.0},
                  "mark_gain_before_change": 0.0, "mark_gain_after_first_move": 0.0}
        blind = dict(seeker, start_mark_best=0.06)
        v = hand_verdicts(summary, commands, {"VM": seeker, "F": blind})
        self.assertEqual((v["H1"], v["H2"], v["L1_all_commands"], v["H3"], v["H4"], v["global"]),
                         (True, False, False, True, True, True))

    def test_long_verdicts(self):
        point = lambda a: {"digit_mass": 0.99, "P0": 0.25, "P1_A": a, "P1_BCD": 0.25}
        v = long_verdicts({"850": point(0.35), "1350": point(0.7)}, point(0.312), "1350")
        self.assertEqual((v["G1"], v["G2"], v["global"]), (True, True, True))
        v = long_verdicts({"1350": point(0.45)}, point(0.312), "1350")
        self.assertEqual((v["G1"], v["G2"]), (False, True))
        v = long_verdicts({"1350": point(0.33)}, point(0.312), "1350")
        self.assertEqual((v["G1"], v["G2"]), (False, False))

    def test_cli_scores_and_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            main(["score", "--out", tmp + "/r", "--label", "VML", "--scripted", "mark"])
            main(["score", "--out", tmp + "/f", "--label", "F", "--scripted", "fixed"])
            main(["verdicts", "--reading", f"F={tmp}/f", f"VML={tmp}/r", "--output", tmp + "/v.json"])
            main(["verdicts", "--reading", f"F={tmp}/f", f"VML={tmp}/r", "--check", tmp + "/v.json"])
            self.assertTrue(json.loads(Path(tmp, "v.json").read_text())["verdicts"]["L1"])


if __name__ == "__main__":
    unittest.main()
