import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from research.llm_report import (journal, expected, counterfactual, build_items, score_items, verdicts, OracleScorer, QUESTIONS,
                                 main, agent_states)

STATE = {"life": 0, "t": 12, "pos": 3, "pos_conf": 0.9, "body": 2, "body_conf": 0.8, "intent": 5, "spot": 6, "read": 1,
         "trust": 0.1, "energy": 0.3, "satiety": 0.7, "module": "vis", "goal": "charger", "captured_estimate": True}


class ReportTests(unittest.TestCase):
    def test_journal_and_expected_answers(self):
        text = journal(STATE)
        self.assertIn("case 3", text)
        self.assertIn("atterri sur la case 6", text)
        self.assertEqual(expected(STATE, "trust"), 0)
        self.assertEqual(expected(STATE, "need"), 1)
        self.assertEqual(expected(STATE, "module"), 3)
        self.assertIn("il pleut", journal(STATE, distractor=True))

    def test_counterfactuals_change_the_answer(self):
        rng = np.random.default_rng(0)
        for kind in QUESTIONS:
            self.assertNotEqual(expected(counterfactual(STATE, kind, rng), kind), expected(STATE, kind))

    def test_oracle_passes_and_a_constant_reporter_fails(self):
        items = build_items([STATE, dict(STATE, pos=6, body=0, spot=5, trust=0.9, energy=0.8, module="pos", captured_estimate=False)])
        out = verdicts(score_items(items, OracleScorer(items), log=lambda m: None))
        self.assertTrue(out["global"])
        constant = lambda prompt: (np.eye(10)[0] * 0.9 + 0.01, 1.0)
        out = verdicts(score_items(items, constant, log=lambda m: None))
        self.assertFalse(out["global"])

    def test_published_items_are_those_the_builder_makes(self):
        path = Path("artifacts/llm-report/items.jsonl")
        items = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(items), 1854)
        self.assertEqual({i["variant"] for i in items}, {"real", "counterfactual", "distractor"})
        with tempfile.TemporaryDirectory() as tmp:
            main(["score", "--items", str(path), "--out", tmp])
            self.assertTrue(json.loads(Path(tmp, "summary.json").read_text())["global"])


class ReportProvenanceTests(unittest.TestCase):
    def test_items_rebuild_from_the_published_agent(self):
        """Replays the published agent in numpy: run in CI (x86), not on the Mac, where float rounding may differ."""
        items = [json.loads(line) for line in Path("artifacts/llm-report/items.jsonl").read_text().splitlines()]
        self.assertEqual(build_items(agent_states("artifacts/indicator-agent", 17)), items)


if __name__ == "__main__":
    unittest.main()
