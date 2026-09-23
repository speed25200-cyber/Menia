import json
import tempfile
import unittest
from pathlib import Path
from research.origin_env import DELTAS, motor_delta, LIFE
from research.llm_lora_body import (childhood_text_lives, export_dataset, completion_prompt, life_text, HEADER, main,
                                    motor_examples)
from research.llm_latent_body import ScriptedScorer, score_life
from research.text_atelier import run_text_lives
from research.own_action_experiment import CHANGE_STEP
import re

MOVE = re.compile(r"Tour (\d+) : commande ([ABCD]), de la case (\d) à la case (\d)")


def implied_bodies(text):
    bodies = set()
    for _, c, a, b in MOVE.findall(text):
        delta = (int(b) - int(a)) % 8
        delta = delta - 8 if delta > 4 else delta
        bodies.add((DELTAS.index(delta) - "ABCD".index(c)) % 4)
    return bodies


class DataTests(unittest.TestCase):
    def test_fixed_regime_has_one_body_and_mutable_regime_sometimes_two(self):
        fixed = childhood_text_lives("F", 3, 20)
        self.assertTrue(all(implied_bodies(d["text"]) <= {0} for d in fixed))
        mutable = childhood_text_lives("VM", 3, 40)
        changed = sum(len(implied_bodies(d["text"])) > 1 for d in mutable)
        self.assertGreater(changed, 6)
        self.assertLess(changed, 34)
        variable = childhood_text_lives("V", 3, 40)
        self.assertTrue(all(len(implied_bodies(d["text"])) <= 1 for d in variable))
        self.assertEqual({d["d"] for d in variable}, {0, 1, 2, 3})

    def test_documents_have_the_prompt_format(self):
        doc = childhood_text_lives("V", 1, 1)[0]["text"]
        self.assertTrue(doc.startswith(HEADER[0]))
        self.assertIn("\nHistorique :\nTour 1 : ", doc)
        self.assertEqual(doc.count("Tour "), LIFE)

    def test_the_look_first_regime_is_vm_looking_before_it_acts(self):
        vm, vmi = childhood_text_lives("VM", 5, 60), childhood_text_lives("VMI", 5, 60)
        self.assertEqual([d["d"] for d in vm], [d["d"] for d in vmi])
        self.assertEqual([d["change_step"] for d in vm], [d["change_step"] for d in vmi])
        read_first = 0
        for doc in vmi:
            lines = doc["text"].split("Historique :\n")[1].splitlines()
            first_move = next(i for i, line in enumerate(lines) if "commande" in line)
            self.assertGreaterEqual(first_move, 2)
            self.assertTrue(all("inspection du lieu" in line for line in lines[:first_move]))
            read_first += any("lieu 1," in line for line in lines[:first_move])
        self.assertGreater(read_first / len(vmi), 0.6)

    def test_motor_examples_are_the_test_prompts_with_the_landing_digit(self):
        doc = childhood_text_lives("VMI", 5, 1)[0]
        lines = doc["text"].split("Historique :\n", 1)[1].splitlines()
        examples = motor_examples(doc)
        moves = [(i, line) for i, line in enumerate(lines) if " : commande " in line]
        self.assertEqual(len(examples), len(moves))
        for (i, line), example in zip(moves, examples):
            m = re.match(r"Tour (\d+) : commande (\w), de la case (\d) à la case (\d)", line)
            prompt = completion_prompt(lines[:i], int(m.group(3)), "ABCD".index(m.group(2)), int(m.group(1)) - 1)
            self.assertEqual(example, {"prompt": prompt, "completion": m.group(4)})
        with tempfile.TemporaryDirectory() as tmp:
            export_dataset("VMI", tmp, train=3, valid=2, objective="motor")
            row = json.loads(Path(tmp, "train.jsonl").read_text().splitlines()[0])
            self.assertEqual(set(row), {"prompt", "completion"})
            self.assertTrue(row["prompt"].endswith("à la case "))

    def test_raw_completion_masks_the_prompt(self):
        from research.llm_motor_lora import raw_process

        class Tokenizer:
            def encode(self, text, add_special_tokens=True):
                return [ord(c) for c in text]

        class Dataset:
            tokenizer, prompt_key, completion_key = Tokenizer(), "prompt", "completion"

        tokens, offset = raw_process(Dataset(), {"prompt": "à la case ", "completion": "7"})
        self.assertEqual(tokens[offset:], [ord("7")])
        self.assertEqual(offset, len("à la case "))

    def test_export_writes_train_and_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_dataset("F", tmp, train=5, valid=2, seed=4)
            train = [json.loads(l) for l in (Path(tmp) / "train.jsonl").read_text().splitlines()]
            valid = [json.loads(l) for l in (Path(tmp) / "valid.jsonl").read_text().splitlines()]
            self.assertEqual((len(train), len(valid)), (5, 2))
            self.assertTrue(all(set(d) == {"text"} for d in train))
            self.assertNotEqual(train[0]["text"], valid[0]["text"])


class CompletionTests(unittest.TestCase):
    def test_completion_prompt_ends_before_the_landing_square(self):
        prompt = completion_prompt(["Tour 1 : commande A, de la case 0 à la case 2. Cible 4."], 2, 1, 1)
        self.assertTrue(prompt.endswith("Tour 2 : commande B, de la case 2 à la case "))
        self.assertTrue(prompt.startswith(HEADER[0]))
        self.assertNotIn("Question", prompt)

    def test_scripted_scorers_work_in_completion_mode(self):
        lives = run_text_lives(None, "random", 5, 4, forced_change_step=CHANGE_STEP)
        rows = []
        for life in lives:
            rows += score_life(ScriptedScorer("copier"), life, builder=completion_prompt)
        seen = [r for r in rows if r["category"] in ("vu", "vu_apres")]
        self.assertTrue(seen and all(r["correct"] for r in seen))
        rows = []
        for life in lives:
            rows += score_life(ScriptedScorer("fixed"), life, builder=completion_prompt)
        for r in rows:
            self.assertEqual(r["correct"], motor_delta(0, r["command"]) == motor_delta(lives[r["life"]]["d"] if r["step"] < CHANGE_STEP else lives[r["life"]]["d_final"], r["command"]))

    def test_cli_evaluate_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            main(["export", "--regime", "VM", "--out", tmp + "/data", "--train", "3", "--valid", "1"])
            self.assertTrue((Path(tmp) / "data" / "train.jsonl").exists())
            main(["evaluate", "--out", tmp + "/eval", "--backend", "scripted", "--scripted", "copier", "--episodes", "2", "--label", "test"])
            receipt = json.loads((Path(tmp) / "eval" / "receipt.json").read_text())
            self.assertEqual(receipt["mode"], "completion")
            self.assertEqual(receipt["label"], "test")


if __name__ == "__main__":
    unittest.main()
