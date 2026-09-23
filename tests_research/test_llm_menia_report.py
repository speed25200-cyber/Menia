import json
import re
import tempfile
import unittest
from pathlib import Path
import numpy as np
from menia.indicator_bridge import IndicatorAgentBridge
from research.llm_menia_report import (QUESTIONS, QUESTIONS_V2, OracleScorer, bridge_states, build_items, counterfactual,
                                       expected, main, render, score_items, verdicts)

CONTEXT = {"tick": 20, "workspace": {"position": {"square": 7, "confidence": 0.9, "age": 1},
                                     "body": {"body": 2, "confidence": 0.8, "age": 8},
                                     "vision": {"present": [1, 3, 4], "values": {"1": 0.8, "3": -0.5}, "age": 2},
                                     "interoception": {"energy": 0.75, "satiety": 0.25, "age": 0}},
           "best_known_object": 1, "lowest_need": "satiety", "goal": "stay", "last_writer": "intero", "alarm": "intero"}
PRONOUNS = {"je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles", "se", "lui", "leur", "y", "en", "son", "sa", "ses"}
ITEMS = Path("artifacts/menia-report/items.jsonl")
ITEMS_V2 = Path("artifacts/menia-report/items-v2.jsonl")


def words(text):
    return set(re.findall(r"[a-zàâçéèêëîïôöûùüÿœ]+", text.lower()))


class MeniaReportTests(unittest.TestCase):
    def test_journal_states_the_workspace_without_pronouns(self):
        text = render(CONTEXT)
        for part in ("sur la case 7 (contenu écrit 1 pas plus tôt)", "le corps 2 (contenu écrit au moins 8 pas plus tôt)",
                     "Objets vus sur les cases : 1, 3, 4.", "valeur connue est sur la case 1", "est la satiété (contenu écrit à ce pas)",
                     "les besoins (par une alarme).", "Décision de l'agent : rester sur place."):
            self.assertIn(part, text)
        self.assertIn("la pluie tombe", render(CONTEXT, distractor=True))
        self.assertNotIn("la pluie tombe", text)
        for question, _ in QUESTIONS.values():
            self.assertEqual(words(render(CONTEXT, distractor=True) + question) & PRONOUNS, set())

    def test_expected_answers(self):
        self.assertEqual([expected(CONTEXT, k) for k in QUESTIONS], [7, 2, 2, 3, 4, 1])
        self.assertEqual(expected(dict(CONTEXT, goal=5), "decision"), 2)
        self.assertEqual(expected(dict(CONTEXT, goal="charger"), "decision"), 1)

    def test_counterfactuals_change_the_answer_and_stay_coherent(self):
        rng = np.random.default_rng(0)
        for _ in range(30):
            for kind in QUESTIONS:
                cf = counterfactual(CONTEXT, kind, rng)
                self.assertNotEqual(expected(cf, kind), expected(CONTEXT, kind))
                if kind == "need":
                    needs = cf["workspace"]["interoception"]
                    self.assertEqual(cf["lowest_need"], "energy" if needs["energy"] < needs["satiety"] else "satiety")
                if kind == "best" or (kind == "decision" and isinstance(cf["goal"], int)):
                    square = cf["best_known_object"] if kind == "best" else cf["goal"]
                    self.assertIn(square, cf["workspace"]["vision"]["present"])
                if kind == "writer":
                    self.assertNotIn("alarme", render(cf))
        self.assertEqual(CONTEXT["workspace"]["vision"]["present"], [1, 3, 4])

    def test_oracle_passes_and_a_constant_reporter_fails(self):
        other = json.loads(json.dumps(CONTEXT))
        other["workspace"]["position"]["square"], other["goal"], other["last_writer"] = 0, "charger", "pos"
        items = build_items([{"life": 0, "t": 20, "context": CONTEXT}, {"life": 0, "t": 28, "context": other}])
        out = verdicts(score_items(items, OracleScorer(items), log=lambda m: None))
        self.assertTrue(out["global"])
        constant = lambda prompt: (np.eye(10)[1] * 0.9 + 0.01, 1.0)
        out = verdicts(score_items(items, constant, log=lambda m: None))
        self.assertFalse(out["global"])

    def test_published_items_are_scored_by_the_oracle(self):
        items = [json.loads(line) for line in ITEMS.read_text().splitlines()]
        self.assertEqual(len(items), 2160)
        self.assertEqual({i["variant"] for i in items}, {"real", "counterfactual", "distractor"})
        self.assertEqual({i["kind"] for i in items}, set(QUESTIONS))
        with tempfile.TemporaryDirectory() as tmp:
            main(["score", "--items", str(ITEMS), "--out", tmp])
            summary = json.loads(Path(tmp, "summary.json").read_text())
            self.assertTrue(summary["global"])
            main(["verdicts", "--rows", str(Path(tmp, "rows.jsonl")), "--check", str(Path(tmp, "summary.json"))])

    def test_second_version_labels_the_position_and_names_the_module(self):
        text = render(CONTEXT, version=2)
        self.assertIn("Position de l'agent selon l'espace de travail : case 7 (contenu écrit 1 pas plus tôt).", text)
        self.assertIn("Dernier module entré dans l'espace de travail : Intéro (par une alarme).", text)
        self.assertEqual(render(CONTEXT).splitlines()[2:5], text.splitlines()[2:5])
        for question, _ in QUESTIONS_V2.values():  # « en dernier » : préposition, pas pronom
            text_and_question = render(CONTEXT, distractor=True, version=2) + question.replace("en dernier", "dernier")
            self.assertEqual(words(text_and_question) & PRONOUNS, set())
        other = json.loads(json.dumps(CONTEXT))
        other["workspace"]["position"]["square"], other["goal"], other["last_writer"] = 0, "charger", "pos"
        items = build_items([{"life": 0, "t": 20, "context": CONTEXT}, {"life": 0, "t": 28, "context": other}], version=2)
        self.assertTrue(verdicts(score_items(items, OracleScorer(items), log=lambda m: None))["global"])

    def test_published_second_version_items_are_scored_by_the_oracle(self):
        items = [json.loads(line) for line in ITEMS_V2.read_text().splitlines()]
        self.assertEqual(len(items), 2154)
        self.assertTrue(all(i["journal"] == 2 for i in items))
        with tempfile.TemporaryDirectory() as tmp:
            main(["score", "--items", str(ITEMS_V2), "--out", tmp])
            self.assertTrue(json.loads(Path(tmp, "summary.json").read_text())["global"])


class MeniaReportProvenanceTests(unittest.TestCase):
    """Replays the published agent in numpy: run in CI (x86), not on the Mac, where float rounding may differ."""

    def test_items_rebuild_from_the_bridged_agent(self):
        items = [json.loads(line) for line in ITEMS.read_text().splitlines()]
        self.assertEqual(build_items(bridge_states()), items)

    def test_second_version_items_rebuild_from_the_version_6_agent(self):
        items = [json.loads(line) for line in ITEMS_V2.read_text().splitlines()]
        self.assertEqual(build_items(bridge_states("artifacts/indicator-agent-v6", 151), version=2), items)

    def test_the_bridged_agent_lives_the_evaluated_lives_of_set_M(self):
        records = [json.loads(line) for line in Path("artifacts/indicator-agent-v5/lives-113-M.jsonl").read_text().splitlines()]
        for i in range(3):
            bridge = IndicatorAgentBridge.from_artifacts(env_seed=930002 * 1000 + i, mode="change",
                                                         agent_seed=113 * 1_000_000 + 10_000 + i)
            self.assertEqual([bridge.cycle()["result"]["action"] for _ in range(48)], records[i]["actions"])


if __name__ == "__main__":
    unittest.main()
