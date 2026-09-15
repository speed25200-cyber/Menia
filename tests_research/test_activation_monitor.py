import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research.activation_monitor import (
    ALPHAS, CELLS, COUNTS, DIM, analyze, append, canonical, collect, fit_bundle,
    forecast, make_plan, messages, predict, read_journal, ridge_fit,
)
from research.cross_model_prediction import make_plan as previous_plan


class SyntheticBackend:
    origin = "synthetic_fixture"
    metadata = {"description": "Artificial software control, not an LLM"}

    def __init__(self, path, signal=True, fail=False):
        self.path, self.signal, self.fail = path, signal, fail

    def generate(self, task, capture):
        if self.fail:
            raise RuntimeError("Synthetic backend failure")
        # Labels are independent of public prompt features; the internal signal
        # is planted solely to check whether the measurement can recover it.
        success = bool(np.random.default_rng(task["seed"]+81).integers(2))
        state = {name: [0.]*DIM for name in ("input", "middle", "final")}
        if self.signal:
            state["middle"][0] = 3. if success else -3.
        capture(state)
        last = json.loads(self.path.read_text(encoding="utf-8").splitlines()[-1])
        assert last["event"] == "state"
        assert (last["predictions"] is not None) == (task["split"] == "test")
        if task["letters"]:
            answer = sum(ch == "A" for ch in task["letters"])
        else:
            answer = sum((-1)**i*n for i, n in enumerate(task["operands"]))
        return str(answer + (0 if success else 1)), {"synthetic": True}


class ActivationMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name)/"signal.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            collect(cls.path, SyntheticBackend(cls.path))
        cls.header, cls.rows, _, cls.bundle = read_journal(cls.path)
        cls.report = analyze(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fixed_splits_are_disjoint_and_exclude_previous_pilot(self):
        plan = make_plan()
        self.assertEqual(len(plan["tasks"]), 672)
        questions = {t["question"] for t in plan["tasks"]}
        self.assertEqual(len(questions), 672)
        self.assertFalse(questions & {t["question"] for t in previous_plan()["tasks"]})
        for split, count in COUNTS.items():
            for f, level in CELLS:
                self.assertEqual(sum(t["split"] == split and (t["family"], t["level"]) == (f, level)
                                     for t in plan["tasks"]), count)

    def test_positive_control_and_wrong_state_controls(self):
        scores = self.report["scores"]
        self.assertTrue(self.report["complete"])
        self.assertEqual(self.report["origin"], "synthetic_fixture")
        self.assertLess(scores["internal"]["brier"], .05)
        for name in ("inputOnly", "betaCell", "shuffledLabels", "donorState"):
            self.assertGreater(scores[name]["brier"], .1)
        self.assertEqual(self.report["availableTestTargets"], 192)

    def test_refitting_never_reads_test_labels_or_test_states(self):
        poisoned = copy.deepcopy(self.rows)
        for row in poisoned[480:]:
            row["result"] = {"malformed": "must never be read for fitting"}
            row["state"] = {"secret": "must never be read for fitting"}
        self.assertEqual(canonical(fit_bundle(poisoned)), canonical(self.bundle))

    def test_zero_internal_features_give_no_advantage(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row["state"]["middle"] = [0.]*DIM
            row["state"]["final"] = [0.]*DIM
        bundle = fit_bundle(rows)
        for row in rows[480:]:
            p = forecast(bundle, rows[:480], row["task"], row["state"])
            self.assertAlmostEqual(p["internal"], p["inputOnly"], places=10)

    def test_ridge_dual_matches_independent_augmented_least_squares(self):
        rng = np.random.default_rng(21)
        x, y = rng.normal(size=(8, 19)), rng.integers(0, 2, 8)
        alpha = .1
        model = ridge_fit(x, y, alpha)
        z = (x-x.mean(0))/x.std(0)
        augmented = np.vstack((z, np.sqrt(len(x)*alpha)*np.eye(x.shape[1])))
        target = np.r_[y-y.mean(), np.zeros(x.shape[1])]
        independent = np.linalg.lstsq(augmented, target, rcond=None)[0]
        np.testing.assert_allclose(model["weights"], independent, atol=1e-12)

    def test_tampered_forecast_and_future_fit_order_rejected(self):
        events = [json.loads(s) for s in self.path.read_text(encoding="utf-8").splitlines()]
        altered = copy.deepcopy(events)
        state = next(e for e in altered if e["event"] == "state" and e["predictions"] is not None)
        state["predictions"]["internal"] = 1-state["predictions"]["internal"]
        path = Path(self.tmp.name)/"tampered.jsonl"
        path.write_text("\n".join(canonical(e) for e in altered)+"\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "forecast mismatch"):
            read_journal(path)
        altered = [e for e in events if e["event"] != "fit"]
        path.write_text("\n".join(canonical(e) for e in altered)+"\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen monitor"):
            read_journal(path)

    def test_pending_request_is_retained_as_interruption_on_resume(self):
        path = Path(self.tmp.name)/"interrupted.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            collect(path, SyntheticBackend(path), limit=0)
        task = make_plan()["tasks"][0]
        append(path, dict(event="request", task=task, messages=messages(task)))
        with contextlib.redirect_stdout(io.StringIO()):
            collect(path, SyntheticBackend(path), resume=True, limit=1)
        _, rows, pending, bundle = read_journal(path)
        self.assertEqual([r["result"]["status"] for r in rows], ["interrupted", "ok"])
        self.assertIsNone(pending)
        self.assertIsNone(bundle)
        self.assertFalse(analyze(path)["complete"])

    def test_technical_failure_is_recorded_without_fabricated_prediction(self):
        path = Path(self.tmp.name)/"error.jsonl"
        with self.assertRaisesRegex(RuntimeError, "Synthetic backend"):
            collect(path, SyntheticBackend(path, fail=True), limit=1)
        _, rows, pending, _ = read_journal(path)
        self.assertEqual(rows[0]["result"]["status"], "error")
        self.assertNotIn("state", rows[0])
        self.assertIsNone(pending)


if __name__ == "__main__":
    unittest.main()
