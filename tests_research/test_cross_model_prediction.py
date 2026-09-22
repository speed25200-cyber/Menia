import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from research.cross_model_prediction import (
    CELLS, MODELS, SEED, analyze, append, canonical, collect, grade, make_plan,
    read_journal, reference, request_for, score,
)


class FixtureBackend:
    origin = "synthetic_fixture"
    metadata = {"description": "Software test, not an LLM", "packages": [("fake", "1")]}

    def __init__(self, fail=None, invalid=None):
        self.fail, self.invalid, self.seen = fail, invalid, []
        self.plan = make_plan()

    def generate(self, request):
        self.seen.append(request)
        call = request["call"]
        if call["id"] == self.fail:
            raise RuntimeError("Synthetic interruption")
        if call["id"] == self.invalid:
            return '{"p":true}', {}
        if call["kind"] == "forecast":
            return '{"p":0.8}' if call["model"] == "A" else '{"p":0.2}', {}
        task = self.plan["tasks"][call["task"]]
        success = (task["block"] % 3 != 0) == (call["model"] == "A")
        # Compute answer independently from reference(), only in this explicitly synthetic fixture.
        answer = sum(c == "A" for c in task["letters"]) if task["family"] == "countA" else sum(
            n * (-1)**i for i, n in enumerate(task["operands"]))
        return str(answer + (0 if success else 1)), {}


class CrossModelPredictionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/"audit.jsonl"

    def run_fixture(self, backend=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return collect(self.path, backend or FixtureBackend(), **kwargs)

    def test_plan_has_all_cells_no_overlap_and_prior_forecasts(self):
        p = make_plan()
        self.assertEqual(len(p["tasks"]), 60)
        self.assertEqual(len(p["calls"]), 408)
        self.assertEqual(len({t["question"] for t in p["tasks"]}), 60)
        for phase, count in (("calibration", 4), ("evaluation", 6)):
            for f, l in CELLS:
                self.assertEqual(sum(t["phase"] == phase and t["family"] == f and t["level"] == l for t in p["tasks"]), count)
        for t in p["tasks"][24:]:
            calls = [c for c in p["calls"] if c["task"] == t["id"]]
            self.assertEqual([c["kind"] for c in calls], ["forecast"]*8+["solve"]*2)
            self.assertEqual({(c["model"], c["target"], c["view"]) for c in calls[:8]},
                {(a,b,v) for a in MODELS for b in MODELS for v in ("normal", "swapped")})
        self.assertNotEqual(make_plan(SEED+1)["tasks"], p["tasks"])
        p["models"]["A"]["id"] = "changed"
        self.assertEqual(make_plan()["models"]["A"]["id"], "Qwen/Qwen3-4B")

    def test_same_information_and_calibration_only_with_name_pairing(self):
        self.run_fixture(limit=58)
        h, r, _ = read_journal(self.path)
        requests = [request_for(h["plan"], i, r) for i in range(48,56)]
        for target in MODELS:
            same = [x for x in requests if x["call"]["target"] == target and x["call"]["view"] == "normal"]
            self.assertEqual(same[0]["messages"], same[1]["messages"])
            payload = json.loads(same[0]["messages"][1]["content"])
            self.assertEqual(len(payload["history"]), 4)
            self.assertEqual(set(payload), {"target", "generation", "family", "level", "question", "history"})
            calibration_questions = {t["question"] for t in h["plan"]["tasks"][:24]}
            self.assertTrue(all(x["question"] in calibration_questions for x in payload["history"]))
        for predictor in MODELS:
            for target in MODELS:
                pair = [x for x in requests if x["call"]["model"] == predictor and x["call"]["target"] == target]
                self.assertEqual(pair[0]["call"]["seed"], pair[1]["call"]["seed"])
                a,b = [json.loads(x["messages"][1]["content"]) for x in pair]
                self.assertNotEqual(a["target"].pop("name"), b["target"].pop("name"))
                self.assertEqual(a,b)

    def test_complete_reconstruction_has_known_matrix_not_an_llm_result(self):
        r = self.run_fixture()
        self.assertTrue(r["complete"])
        self.assertEqual(r["origin"], "synthetic_fixture")
        self.assertFalse(r["executionAttested"])
        self.assertEqual(r["recordedResults"], 408)
        for view in r["views"].values():
            self.assertAlmostEqual(view["diagonalAdvantage"], .2)
            self.assertAlmostEqual(view["matrix"]["A->A"]["systemBrier"], .24)
            self.assertAlmostEqual(view["matrix"]["B->A"]["systemBrier"], .44)
            self.assertEqual(view["matrix"]["A->A"]["withinCell"][0]["auc"], .5)
        # Calibration is 2/4 for A and 2/4 for B because block 0 and 3 fail for A.
        self.assertAlmostEqual(r["baselines"]["A"]["empiricalCell"]["systemBrier"], .25)

    def test_invalid_forecast_preserved_and_excluded_only_from_valid_metric(self):
        r = self.run_fixture(FixtureBackend(invalid=48))
        call = make_plan()["calls"][48]
        cell = r["views"][call["view"]]["matrix"][f"{call['model']}->{call['target']}"]
        self.assertEqual((cell["n"], cell["valid"], cell["invalid"]), (36,35,1))
        self.assertIsNone(cell["auc"])
        self.assertTrue(r["complete"])

    def test_failure_retained_on_resume_and_never_retried(self):
        with self.assertRaises(RuntimeError):
            self.run_fixture(FixtureBackend(fail=0))
        backend = FixtureBackend()
        r = self.run_fixture(backend, resume=True)
        self.assertEqual(backend.seen[0]["call"]["id"], 1)
        self.assertTrue(r["finished"])
        self.assertFalse(r["complete"])
        self.assertEqual(r["statuses"]["error"], 1)

    def test_unrecorded_result_is_marked_interrupted_before_resume(self):
        self.run_fixture(limit=0)
        h, r, _ = read_journal(self.path)
        append(self.path, request_for(h["plan"], 0, r))
        backend = FixtureBackend()
        result = self.run_fixture(backend, resume=True, limit=1)
        self.assertEqual(result["statuses"], {"interrupted":1, "ok":1})
        self.assertEqual(backend.seen[0]["call"]["id"], 1)

    def test_existing_file_and_changed_environment_not_silently_replaced(self):
        self.run_fixture(limit=0)
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_fixture(limit=0)
        backend = FixtureBackend()
        backend.metadata = {"description":"different"}
        with self.assertRaises(ValueError):
            self.run_fixture(backend, resume=True)
        self.assertEqual(self.path.read_bytes(), before)

    def test_plan_prompt_and_history_type_tampering_rejected(self):
        self.run_fixture(limit=50)
        original = [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines()]
        for kind in ("plan", "prompt", "history"):
            events = copy.deepcopy(original)
            if kind == "plan": events[0]["plan"]["calls"][0]["seed"] += 1
            if kind == "prompt": events[1]["messages"][0]["content"] = "Different instructions"
            if kind == "history":
                event = events[1+2*48]
                payload = json.loads(event["messages"][1]["content"])
                payload["history"][0]["correct"] = int(payload["history"][0]["correct"])
                event["messages"][1]["content"] = canonical(payload)
            self.path.write_text("\n".join(canonical(x) for x in events)+"\n", encoding="utf-8")
            with self.assertRaises(ValueError, msg=kind): analyze(self.path)

    def test_missing_targets_are_not_counted_as_wrong_or_as_complete(self):
        r = self.run_fixture(limit=56)
        self.assertFalse(r["complete"])
        self.assertIsNone(r["views"]["normal"]["diagonalAdvantage"])
        self.assertEqual(r["views"]["normal"]["matrix"]["A->A"]["n"], 0)
        self.assertEqual(r["views"]["normal"]["matrix"]["A->A"]["missingTargets"], 36)

    def test_reference_strict_grade_and_auc_with_ties(self):
        task = dict(family="alternatingSum", operands=[19,23,14,17])
        self.assertEqual(reference(task), -7)
        self.assertTrue(grade(dict(status="ok", text="-7"), task))
        self.assertFalse(grade(dict(status="ok", text="Résultat : -7"), task))
        self.assertIsNone(grade(dict(status="error", text=""), task))
        self.assertEqual(score([(.9,True),(.1,True),(.1,False)])["auc"], .75)


if __name__ == "__main__":
    unittest.main()
