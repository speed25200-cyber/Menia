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
        params.q = np.zeros((3, 8))
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


class VersionThreeTests(unittest.TestCase):
    def test_binned_needs_and_replayed_values(self):
        from research.indicator_agent_v2 import binned_features
        W = {"vis": {"values": [None] * 8, "presence": [0] * 8}, "pos": np.eye(8)[2], "intero": np.array([0.1, 0.9])}
        phi, candidate, _ = binned_features(W, 5)
        self.assertEqual(len(phi), 19)
        self.assertEqual(phi[:16].sum(), 1.0)
        self.assertEqual(int(np.argmax(phi[:16])), 3)
        self.assertIsNone(candidate)
        with tempfile.TemporaryDirectory() as tmp:
            experiment_main(["--version", "3", "--out", tmp, "--seeds", "6", "--childhood", "20", "--updates", "2",
                             "--batch", "3", "--lives", "2"])
            self.assertEqual(audit(tmp, replay_lives=2, full=True, log=lambda m: None)["problems"], [])


class VersionFourTests(unittest.TestCase):
    def test_alarms_take_the_workspace_and_values_follow_the_drive(self):
        from research.indicator_agent_v4 import AgentV4, drive, internal_rewards
        params = provisional_params(2)
        params.q = np.zeros((3, 19))
        agent = AgentV4(params, seed=1)
        agent.W["intero"] = np.array([0.5, 0.9])
        contents = {"intero": np.array([0.3, 0.9]), "body": agent.W["body"].copy()}
        self.assertEqual(agent._alarm(contents), "intero")
        agent.W["intero"] = np.array([0.33, 0.9])
        self.assertIsNone(agent._alarm(contents))
        contents["body"] = np.array([0.0, 1.0, 0.0, 0.0])
        self.assertEqual(agent._alarm(contents), "body")
        self.assertAlmostEqual(drive(1.0, 0.5), 0.0625)
        life = run_life(params, "agent", 12, "fixed", 12, version=4)
        steps = [r for r, _ in life["steps"]]
        self.assertTrue(all(len(r["writers"]) == 1 for r in steps))
        self.assertGreater(len({str(r["goal"]) for r in steps if r["decided"]}), 1)
        self.assertTrue(any(a["goal"] == "stay" and b["goal"] == "stay" and not b["decided"] for a, b in zip(steps, steps[1:])))
        rewards = internal_rewards(life)
        self.assertEqual(len(rewards), len(life["rewards"]))
        self.assertTrue(all(ri <= r + 1e-12 for ri, r in zip(rewards, life["rewards"])))
        with tempfile.TemporaryDirectory() as tmp:
            experiment_main(["--version", "4", "--out", tmp, "--seeds", "8", "--childhood", "20", "--updates", "2",
                             "--batch", "3", "--lives", "2"])
            self.assertEqual(audit(tmp, replay_lives=2, full=True, log=lambda m: None)["problems"], [])


class VersionFiveTests(unittest.TestCase):
    def test_empty_model_foresees_nothing_and_a_learned_one_arbitrates(self):
        from research.indicator_agent_v5 import NeedModel, simulate, need_observations, fit_model
        empty = NeedModel()
        self.assertEqual(simulate(empty, 0.5, 0.5, [("charger", 3, 0.0)], False),
                         simulate(empty, 0.5, 0.5, [("stay", 1, 0.0)], False))
        model = NeedModel(decay=(1 / 14, 1 / 20), charge=1.0, food=(1.0, 0.0), reward=(1.0, 0.0), travel=(0.5, 0.6))
        self.assertGreater(simulate(model, 0.25, 0.6, [("charger", 2, 0.0)], False),
                           simulate(model, 0.25, 0.6, [("food", 2, 0.6)], False))
        self.assertGreater(simulate(model, 0.9, 0.4, [("food", 2, 0.6)], False),
                           simulate(model, 0.9, 0.4, [("charger", 2, 0.0)], False))
        params = provisional_params(2)
        life = run_life(params, "agent", 12, "fixed", 12, version=5, learn=True)
        steps = [r for r, _ in life["steps"]]
        self.assertTrue(all(r["goal"] == "stay" or r["exploring"] for r in steps))
        self.assertTrue(all(len(r["writers"]) == 1 for r in steps))
        obs = need_observations(life)
        self.assertTrue(all(d >= 0 for d in obs["decay_e"]))
        self.assertEqual(fit_model({k: [] for k in obs}).to_json(), NeedModel().to_json())
        from research.sense_atelier import SenseAtelier, motor_delta
        env = SenseAtelier(5, "fixed", n_objects=3, charger_moves=True)
        env.reset()
        charger = env.charger
        env.p = (charger - 1) % 8
        env.objects.pop(env.p, None)
        env.step(next(a for a in range(4) if motor_delta(env.d, a) == 1), None)
        self.assertEqual(env.p, charger)
        self.assertNotEqual(env.charger, charger)
        self.assertEqual((env.energy, env.relocations), (1.0, 1))
        with tempfile.TemporaryDirectory() as tmp:
            experiment_main(["--version", "5", "--out", tmp, "--seeds", "9", "--childhood", "20", "--updates", "2",
                             "--batch", "3", "--lives", "2"])
            self.assertEqual(audit(tmp, replay_lives=2, full=True, log=lambda m: None)["problems"], [])
            self.assertIn("need", json.loads(Path(tmp, "params-9.json").read_text()))


if __name__ == "__main__":
    unittest.main()
