import json
import tempfile
import unittest
from pathlib import Path
from research.llm_atelier import (build_prompt, parse_reply, origin_mentions, place_rotation, run_episode, analyze,
                                  run_plan, ScriptedResponder, CONDITIONS, main, verify_manifest)
from research.origin_env import LIFE


class PromptTests(unittest.TestCase):
    def test_prompt_never_names_the_hidden_causes_or_the_mapping(self):
        for condition in CONDITIONS:
            prompt = build_prompt(condition, [], place_rotation(0))
            for forbidden in ("D=", "E=", "-2", "+2", "déplace de", "révèle"):
                self.assertNotIn(forbidden, prompt)
            self.assertEqual("construit ton corps" in prompt, condition.endswith("explicit"))

    def test_rotations_cover_every_place_for_the_mark(self):
        self.assertEqual(sorted(place_rotation(i)[0] for i in range(4)), [1, 2, 3, 4])
        for i in range(4):
            self.assertEqual(sorted(place_rotation(i)), [1, 2, 3, 4])

    def test_parser_is_strict(self):
        self.assertEqual(parse_reply("NOTE: ok\nCOMMANDE: B"), (1, "ok"))
        self.assertEqual(parse_reply("COMMANDE: 3")[0], 4 + 2)
        self.assertIsNone(parse_reply("COMMANDE: A ou B")[0])
        self.assertIsNone(parse_reply("COMMANDE: A\nCOMMANDE: 1")[0])
        self.assertIsNone(parse_reply("je choisis A")[0])
        self.assertEqual(origin_mentions("Je me demande qui m'a créé."), 1)
        self.assertEqual(origin_mentions("Je vais à gauche."), 0)


class EpisodeTests(unittest.TestCase):
    def test_mark_reader_is_credited_under_every_rotation(self):
        for rotation in range(4):
            place = place_rotation(rotation)[0]
            record = run_episode(ScriptedResponder("mark", mark_place=place), "T-explicit", 5, rotation)
            self.assertEqual(record["mark_place"], place)
            self.assertTrue(all(turn["inspected_cue"] == 0 for turn in record["turns"]))
        summary = analyze([record])
        self.assertEqual(summary["T-explicit"]["mark_share"], 1.0)
        self.assertEqual(summary["T-explicit"]["inspections_per_episode"], LIFE)

    def test_invalid_replies_cost_the_turn_and_are_counted(self):
        record = run_episode(ScriptedResponder("broken"), "T-implicit", 7, 0)
        self.assertEqual(len(record["turns"]), LIFE)
        self.assertTrue(all(not turn["valid"] for turn in record["turns"]))
        self.assertEqual(analyze([record])["T-implicit"]["invalid_rate"], 1.0)

    def test_plan_writes_logs_and_scores_origin_talk(self):
        with tempfile.TemporaryDirectory() as d:
            summary = run_plan(ScriptedResponder("talker"), d, episodes_per_condition=2, conditions=("C3-implicit",))
            self.assertEqual(summary["C3-implicit"]["origin_mention_rate"], 1.0)
            self.assertEqual(summary["C3-implicit"]["inspections_per_episode"], 0.0)
            lines = open(f"{d}/episodes.jsonl").read().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["condition"], "C3-implicit")
            random_summary = run_plan(ScriptedResponder("random", seed=3), d + "/r", episodes_per_condition=8)
            for condition in CONDITIONS:
                self.assertLess(abs(random_summary[condition]["mark_share"] - 0.25), 0.2)


class CommandLineTests(unittest.TestCase):
    def test_scripted_backend_writes_summary_and_receipt_for_selected_conditions(self):
        with tempfile.TemporaryDirectory() as d:
            summary = main(["--out", d, "--backend", "scripted", "--scripted", "mark", "--episodes", "2", "--conditions", "T-explicit", "C3-implicit"])
            self.assertEqual(set(summary), {"T-explicit", "C3-implicit"})
            receipt = json.load(open(f"{d}/receipt.json"))
            self.assertEqual(receipt["backend"], "scripted"); self.assertEqual(receipt["conditions"], ["T-explicit", "C3-implicit"])
            self.assertEqual(len(receipt["episodes_sha256"]), 64)
            first = json.loads(open(f"{d}/episodes.jsonl").readline())
            self.assertIn("seconds", first["turns"][0]); self.assertGreater(first["turns"][0]["prompt_characters"], 100)
        with self.assertRaises(SystemExit):
            main(["--out", "/tmp/x", "--backend", "scripted", "--conditions", "nope"])

    def test_manifest_verification_detects_a_changed_file(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_bytes(b"{}")
            manifest = {"repository": "r", "revision": "v", "files": [{"name": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}]}
            (Path(d) / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verify_manifest(d, Path(d) / "manifest.json")["revision"], "v")
            (Path(d) / "config.json").write_bytes(b"{ }")
            with self.assertRaises(ValueError):
                verify_manifest(d, Path(d) / "manifest.json")


if __name__ == "__main__":
    unittest.main()
