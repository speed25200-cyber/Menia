import json
import tempfile
import unittest
from pathlib import Path
from research.llm_body_verdicts import cells, latent_verdicts, lora_verdicts, transformer_l5, validation_losses, read_rows
from research.llm_latent_body import run_sets, ScriptedScorer
from research.own_action_experiment import RANDOM_SEED
from research.text_atelier import run_text_lives

ROOT = Path("artifacts/own-action-channel")


def row(life, step, category, correct, confidence=0.9, digit_mass=0.9):
    return {"life": life, "step": step, "category": category, "correct": correct, "confidence": confidence, "digit_mass": digit_mass}


def write_rows(directory, name, rows):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"rows-{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


class CellTests(unittest.TestCase):
    def test_first_move_of_a_life_is_not_after_a_move(self):
        rows_r = [row(0, 2, "nouveau", False), row(0, 3, "nouveau", True), row(0, 5, "vu", True), row(1, 0, "nouveau", False)]
        rows_m = [row(0, 11, "vu", True), row(0, 12, "vu_avant_seulement", False), row(0, 16, "vu_apres", True)]
        c = cells(rows_r, rows_m)
        self.assertEqual(c["R_new"], 1 / 3)
        self.assertEqual(c["R_new_after_move"], 1.0)
        self.assertEqual(c["counts"]["R_new_after_move"], 1)
        self.assertEqual(c["M_12_23_seen_before_only"], 0.0)
        self.assertEqual(c["M_12_23_seen_after"], 1.0)
        self.assertEqual(c["M_16_23"], 1.0)
        self.assertEqual(c["M_before_change"], 1.0)

    def test_invalid_digit_mass_withholds_every_latent_criterion(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_rows(Path(tmp), "R", [row(0, 0, "vu", True, digit_mass=0.1)])
            write_rows(Path(tmp), "M", [row(0, 12, "vu_apres", True, digit_mass=0.1)])
            out = latent_verdicts(tmp, {"accuracy": 0.95})
        self.assertFalse(out["valid"])
        self.assertIsNone(out["L1"])
        self.assertIsNone(out["global"])
        self.assertTrue(out["L5"])


class LatentTests(unittest.TestCase):
    def test_scripted_copier_has_the_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_sets(ScriptedScorer("copier"), tmp, 6, log=lambda *_: None)
            out = latent_verdicts(tmp)
        self.assertTrue(out["valid"])
        self.assertTrue(out["L1"])
        self.assertEqual(out["cells"]["R_seen"], 1.0)
        self.assertIsNone(out["L5"])

    def test_qwen_lives_are_the_first_lives_of_the_transformer_set_r(self):
        lives = run_text_lives(None, "random", RANDOM_SEED, 3)
        logged = read_rows(ROOT / "lives-V-17-R.jsonl")[:3]
        self.assertEqual([l["actions"] for l in lives], [l["actions"] for l in logged])
        self.assertEqual([l["tokens"] for l in lives], [l["tokens"] for l in logged])

    def test_l5_recomputes_from_the_weights(self):
        l5 = transformer_l5(ROOT, episodes=8)
        self.assertEqual(sorted(l5["models"]), ["V-17", "V-29", "V-43"])
        self.assertGreater(l5["rows"], 0)


class LoraTests(unittest.TestCase):
    def test_verdicts_on_a_fabricated_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            write_rows(run / "base", "R", [row(0, 0, "nouveau", False), row(0, 1, "nouveau", False), row(0, 2, "vu", True)])
            write_rows(run / "base", "M", [row(0, 16, "vu_apres", False)])
            write_rows(run / "F", "R", [row(0, 0, "nouveau", False), row(0, 1, "nouveau", False, confidence=0.8), row(0, 2, "vu", False)])
            write_rows(run / "F", "M", [row(0, 16, "vu_apres", False)])
            write_rows(run / "VM", "R", [row(0, 0, "nouveau", False), row(0, 1, "nouveau", True), row(0, 2, "vu", True)])
            write_rows(run / "VM", "M", [row(0, 16, "vu_apres", True), row(0, 17, "vu_apres", True)])
            (run / "lora-F.log").write_text("Iter 1: Val loss 3.100, Val took 1.0s\nIter 600: Val loss 0.900, Val took 1.0s\n")
            (run / "lora-VM.log").write_text("Iter 1: Val loss 3.200, Val took 1.0s\nIter 600: Val loss 0.950, Val took 1.0s\n")
            out = lora_verdicts(run)
            self.assertEqual(validation_losses(run / "lora-VM.log"), [(1, 3.2), (600, 0.95)])
        self.assertTrue(out["A1"])
        self.assertTrue(out["A2"])
        self.assertTrue(out["A3"])
        self.assertTrue(out["A4"])
        self.assertTrue(out["global"])
        self.assertEqual(sorted(out["A5_models"]), ["VM", "base"])
        self.assertTrue(out["A5"])
        self.assertFalse(out["budget_sufficient"])

    def test_missing_model_leaves_its_criteria_undecided(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_rows(Path(tmp) / "base", "R", [row(0, 0, "nouveau", False)])
            write_rows(Path(tmp) / "base", "M", [row(0, 16, "nouveau", False)])
            out = lora_verdicts(tmp)
        self.assertTrue(out["A1"])
        self.assertIsNone(out["A3"])
        self.assertIsNone(out["global"])
        self.assertIsNone(out["budget_sufficient"])


if __name__ == "__main__":
    unittest.main()
