import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from research.sense_atelier import SenseAtelier, LIFE, RING, value, in_band, BAND
from research.indicator_agent import (Agent, Params, VARIANTS, train_hue_code, fit_conditional_logit, fit_logistic,
                                      schema_candidates, sigmoid, random_hue_code)
from research.indicator_experiment import (auroc, spearman, run_life, provisional_params, main as experiment_main,
                                           quality_space)
from research.audit_indicator_agent import audit


class WorldTests(unittest.TestCase):
    def test_same_seed_same_draws_whatever_the_actions(self):
        a, b = SenseAtelier(11, "fixed"), SenseAtelier(11, "fixed")
        a.reset(), b.reset()
        glitches_a, glitches_b, flashes_a, flashes_b = [], [], [], []
        for t in range(LIFE):
            oa, ta = a.step(t % 5, 0)
            ob, tb = b.step((t * 3 + 1) % 5, 3)
            flashes_a.append(a.flash_on[t]), flashes_b.append(b.flash_on[t])
            glitches_a.append(oa["pos"] is None), glitches_b.append(ob["pos"] is None)
        self.assertEqual(glitches_a, glitches_b)
        self.assertEqual(flashes_a, flashes_b)

    def test_value_and_band(self):
        self.assertAlmostEqual(float(value(0.1)), 1.0)
        self.assertLess(float(value(0.6)), -0.9)
        child = SenseAtelier(3, "childhood")
        self.assertFalse(any(in_band(h) for h in child.spawn_hues))
        band = SenseAtelier(3, "band")
        share = np.mean([in_band(h) for h in band.spawn_hues])
        self.assertGreater(share, 0.3)

    def test_forced_change(self):
        env = SenseAtelier(8, "change")
        env.reset()
        d0 = env.d
        for t in range(LIFE):
            _, truth = env.step(4, None)
            if t == 23:
                self.assertEqual(truth["d"], d0)
        self.assertNotEqual(env.d, d0)

    def test_capture_moves_the_spotlight_only_on_salient_events(self):
        env = SenseAtelier(21, "fixed")
        env.reset()
        for t in range(LIFE):
            _, truth = env.step(4, 0)
            if truth["captured"]:
                self.assertNotEqual(truth["spot"], 0)
            else:
                self.assertEqual(truth["spot"], 0)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.params = provisional_params(1)
        thetas = np.linspace(0, 1, 50, endpoint=False)
        self.params.random_code = random_hue_code(0.3, thetas, value(thetas), 1)

    def test_one_writer_per_step_and_unlimited(self):
        for variant, count in (("agent", 1), ("unlimited", 4), ("round_robin", 1)):
            life = run_life(self.params, variant, 5, "fixed", 5)
            self.assertTrue(all(len(r["writers"]) == count for r, _ in life["steps"]))

    def test_every_variant_runs(self):
        for variant in VARIANTS:
            life = run_life(self.params, variant, 6, "change", 6)
            self.assertEqual(len(life["steps"]), LIFE)

    def test_no_schema_binds_to_the_intent(self):
        life = run_life(self.params, "no_schema", 9, "fixed", 9)
        for rec, _ in life["steps"]:
            if rec["bound"] is not None:
                self.assertEqual(rec["bound"], rec["intent"])

    def test_schema_candidates(self):
        squares, rows = schema_candidates(2, [5, 2, 7], [0, 0, 1, 0, 0, 0, 0, 1], True, 1)
        self.assertEqual(squares, [2, 5, 7])
        self.assertEqual(rows.shape, (3, 6))
        self.assertEqual(rows[0][0], 1.0)
        self.assertEqual(rows[2][5], 1.0)


class LearningTests(unittest.TestCase):
    def test_hue_code_learns_a_smooth_sparse_code(self):
        rng = np.random.default_rng(0)
        thetas = rng.random(400)
        code, history = train_hue_code(thetas, value(thetas), 0, steps=600)
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        space = quality_space(code)
        self.assertGreater(space["spearman"], 0.5)

    def test_readers_fit_simple_rules(self):
        rng = np.random.default_rng(1)
        X = np.hstack([np.ones((500, 1)), rng.normal(size=(500, 2))])
        y = (X[:, 1] > 0).astype(float)
        w = fit_logistic(X, y)
        self.assertGreater(np.mean((sigmoid(X @ w) > 0.5) == y), 0.95)
        groups = [np.array([[1.0, 0.0], [0.0, 1.0]]) for _ in range(50)]
        w = fit_conditional_logit(groups, [1] * 50, 2)
        self.assertGreater(w[1], w[0])

    def test_auroc_and_spearman(self):
        self.assertEqual(auroc([0.9, 0.8, 0.1], [True, True, False]), 1.0)
        self.assertEqual(auroc([0.5, 0.5], [True, False]), 0.5)
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)


class PipelineTests(unittest.TestCase):
    def test_tiny_run_is_audited_without_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment_main(["--out", tmp, "--seeds", "3", "--childhood", "20", "--updates", "2", "--batch", "2", "--lives", "2"])
            out = audit(tmp, replay_lives=2, full=True, log=lambda m: None)
            self.assertEqual(out["problems"], [])
            criteria = json.loads(Path(tmp, "criteria.json").read_text())["criteria"]
            self.assertIn("global", criteria)


class VersionTwoTests(unittest.TestCase):
    def test_three_objects_and_decision_relevant_attention(self):
        from research.indicator_agent_v2 import AgentV2, discounted_returns
        params = provisional_params(2)
        params.q = np.zeros((2, 8))
        life = run_life(params, "agent", 12, "fixed", 12, version=2)
        self.assertEqual(len(life["steps"][0][1]["objects"]), 3)
        self.assertTrue(all(len(r["writers"]) == 1 for r, _ in life["steps"]))
        self.assertTrue(all("changes" in r for r, _ in life["steps"]))
        self.assertAlmostEqual(discounted_returns([1, 0, 1], horizon=2, gamma=0.5)[0], 1.0)

    def test_tiny_version_two_run_is_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment_main(["--version", "2", "--out", tmp, "--seeds", "4", "--childhood", "20", "--updates", "2",
                             "--batch", "3", "--lives", "2"])
            out = audit(tmp, replay_lives=2, full=True, log=lambda m: None)
            self.assertEqual(out["problems"], [])
            report = json.loads(Path(tmp, "report-4.json").read_text())
            self.assertEqual(report["settings"]["version"], 2)
            self.assertEqual(len(report["training"]["rounds"]), 2)


if __name__ == "__main__":
    unittest.main()
