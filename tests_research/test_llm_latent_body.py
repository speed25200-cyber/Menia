import json
import tempfile
import unittest
from pathlib import Path
from research.origin_env import DELTAS, motor_delta, RING
from research.llm_atelier import build_prompt, place_rotation, move_line, inspect_line, ANSWER_INSTRUCTION
from research.llm_latent_body import (question_prompt, score_life, summarize, run_sets, ScriptedScorer, landing_positions,
                                      category, main)
from research.text_atelier import run_text_lives
from research.own_action_experiment import CHANGE_STEP


class PromptTests(unittest.TestCase):
    def test_default_prompt_unchanged_by_refactor(self):
        prompt = build_prompt("T-explicit", ["Tour 1 : commande A, de la case 0 à la case 2. Cible 4."], place_rotation(0))
        self.assertIn("Réponds sur deux lignes exactement, la commande d'abord :\nCOMMANDE: une seule lettre A, B, C, D ou un seul chiffre 1, 2, 3, 4\nNOTE: une phrase de ce que tu penses\n\nHistorique :\nTour 1 : commande A", prompt)
        self.assertIn("Il a laissé une marque", prompt)
        self.assertNotIn("Question", prompt)
        self.assertEqual(build_prompt("T-implicit", [], place_rotation(0)).count("(aucun tour joué)"), 1)

    def test_history_lines_match_the_published_format(self):
        self.assertEqual(move_line(3, 1, 5, 7, 2, 0), "Tour 4 : commande B, de la case 5 à la case 7. Cible 2.")
        self.assertEqual(move_line(0, 0, 0, 2, 2, 1), "Tour 1 : commande A, de la case 0 à la case 2. Cible 2. Point gagné.")
        self.assertEqual(inspect_line(4, 2, 1, 6, 3), "Tour 5 : inspection du lieu 2, symbole △. Position 6, cible 3.")

    def test_question_prompt_carries_position_and_command(self):
        prompt = question_prompt(["Tour 1 : commande C, de la case 1 à la case 0. Cible 5."], 0, 2)
        self.assertTrue(prompt.endswith("Question : tu es sur la case 0. Si tu donnes maintenant la commande C, sur quelle case arriveras-tu ?"))
        self.assertIn("Réponds par un seul chiffre de 0 à 7.", prompt)
        self.assertNotIn(ANSWER_INSTRUCTION, prompt)


class ScoringTests(unittest.TestCase):
    def test_landings_and_categories(self):
        self.assertEqual(sorted(landing_positions(0)), [1, 2, 6, 7])
        self.assertEqual(category(1, {1}, set(), False), "vu")
        self.assertEqual(category(1, set(), set(), False), "nouveau")
        self.assertEqual(category(1, {1}, set(), True), "vu_avant_seulement")
        self.assertEqual(category(1, {1}, {1}, True), "vu_apres")

    def test_copier_is_right_on_seen_commands_and_fixed_is_right_on_body_zero(self):
        lives = run_text_lives(None, "random", 5, 6)
        rows = []
        for life in lives:
            rows += score_life(ScriptedScorer("copier"), life)
        seen = [r for r in rows if r["category"] == "vu"]
        self.assertTrue(seen)
        self.assertTrue(all(r["correct"] for r in seen))
        summary = summarize(rows)
        self.assertEqual(summary["by_category"]["vu"]["accuracy"], 1.0)
        rows = []
        for life in lives:
            rows += score_life(ScriptedScorer("fixed"), life)
        for r in rows:
            life = lives[r["life"]]
            self.assertEqual(r["correct"], motor_delta(life["d"], r["command"]) == motor_delta(0, r["command"]))

    def test_change_set_categories_and_replay_guard(self):
        lives = run_text_lives(None, "random", 6, 3, forced_change_step=CHANGE_STEP)
        rows = score_life(ScriptedScorer("uniform"), lives[0])
        self.assertTrue(all(r["changed"] == (r["step"] >= CHANGE_STEP) for r in rows))
        self.assertTrue(all(r["category"] in ("vu", "nouveau") for r in rows if not r["changed"]))
        self.assertTrue(all(r["category"] in ("vu_apres", "vu_avant_seulement", "nouveau") for r in rows if r["changed"]))
        broken = dict(lives[0], tokens=[0, (lives[0]["tokens"][1] % 8) + 1] + lives[0]["tokens"][2:])
        with self.assertRaises(ValueError):
            score_life(ScriptedScorer("uniform"), broken)

    def test_cli_writes_files_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            main(["--out", tmp, "--backend", "scripted", "--scripted", "copier", "--episodes", "2"])
            out = Path(tmp)
            self.assertTrue((out / "rows-R.jsonl").exists() and (out / "rows-M.jsonl").exists())
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(set(summary), {"R", "M"})
            receipt = json.loads((out / "receipt.json").read_text())
            self.assertEqual(receipt["scripted"], "copier")
            self.assertEqual(set(receipt["files_sha256"]), {"lives-M.jsonl", "lives-R.jsonl", "rows-M.jsonl", "rows-R.jsonl"})


if __name__ == "__main__":
    unittest.main()
