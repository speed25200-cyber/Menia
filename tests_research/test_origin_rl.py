import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from research.origin_env import Atelier, N_MOVE, N_ACTIONS, LIFE
from research.origin_neural import WorldModel
from research.origin_rl import (QNetwork, features, feature_size, Internal, intrinsic_reward, greedy_lives,
                                summarize, train_policy)


class QNetworkTests(unittest.TestCase):
    def test_gradients_against_finite_differences(self):
        net = QNetwork(7, hidden=5, seed=1)
        rng = np.random.default_rng(2)
        x = rng.normal(size=(6, 7)); actions = rng.integers(N_ACTIONS, size=6); targets = rng.normal(size=6) * 3
        _, grads = net.loss_and_grad(x, actions, targets)
        for key, parameter in net.p.items():
            for index in (tuple(0 for _ in parameter.shape), tuple(s - 1 for s in parameter.shape)):
                old = parameter[index]
                parameter[index] = old + 1e-6; plus, _ = net.loss_and_grad(x, actions, targets)
                parameter[index] = old - 1e-6; minus, _ = net.loss_and_grad(x, actions, targets)
                parameter[index] = old
                with self.subTest(parameter=key, index=index):
                    self.assertAlmostEqual(grads[key][index], (plus - minus) / 2e-6, places=5)

    def test_roundtrip_and_copy_independence(self):
        net = QNetwork(4, hidden=3, seed=0)
        clone = net.copy()
        net.p["W1"][0, 0] += 1
        self.assertNotEqual(clone.p["W1"][0, 0], net.p["W1"][0, 0])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "q.json"
            net.save(path, {})
            loaded = QNetwork.load(path)
            for k in net.p:
                np.testing.assert_array_equal(loaded.p[k], net.p[k])
            bad = json.loads(path.read_text()); bad["parameters"]["b2"] = [float("nan")] * 8
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                QNetwork.load(path)


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.model = WorldModel(hidden=6, seed=3)
        self.internal = Internal(self.model)
        self.env = Atelier("T", 4)
        self.obs = self.env.reset()
        self.h = self.model.zero()

    def test_features_have_the_declared_size_and_no_action_block(self):
        x = features(self.h, self.obs, 3)
        self.assertEqual(x.shape, (feature_size(6),))
        self.assertAlmostEqual(x[-1], 3 / LIFE)
        self.assertEqual(x[6:-1].sum(), 8)  # nine observation blocks minus the empty d block

    def test_prudence_penalizes_moves_only_and_surprise_rewards_inspections_only(self):
        for action in range(N_ACTIONS):
            h_next = self.internal.advance(self.h, self.obs, action)
            obs_next = dict(self.obs)
            prudence, _ = intrinsic_reward("prudence", 2.0, self.internal, self.h, self.obs, action, h_next, obs_next)
            surprise, _ = intrinsic_reward("surprise", 2.0, self.internal, self.h, self.obs, action, h_next, obs_next)
            if action < N_MOVE:
                self.assertLess(prudence, 0); self.assertEqual(surprise, 0.0)
            else:
                self.assertEqual(prudence, 0.0); self.assertGreater(surprise, 0)
        zero, _ = intrinsic_reward("prudence", 0.0, self.internal, self.h, self.obs, 0, self.h, self.obs)
        self.assertEqual(zero, 0.0)
        with self.assertRaises(ValueError):
            intrinsic_reward("greed", 1.0, self.internal, self.h, self.obs, 0, self.h, self.obs)

    def test_information_reward_is_entropy_difference_and_caches_after(self):
        h_next = self.internal.advance(self.h, self.obs, 0)
        obs_next, _, _ = self.env.step(0)
        before = self.internal.mean_motion_entropy(self.h, self.obs)
        after = self.internal.mean_motion_entropy(h_next, obs_next)
        reward, cached = intrinsic_reward("information", 3.0, self.internal, self.h, self.obs, 0, h_next, obs_next)
        self.assertAlmostEqual(reward, 3.0 * (before - after))
        self.assertAlmostEqual(cached, after)


class PolicyTests(unittest.TestCase):
    def test_greedy_lives_are_deterministic_and_summarized(self):
        model = WorldModel(hidden=6, seed=5)
        q = QNetwork(feature_size(6), hidden=4, seed=1)
        a = greedy_lives(model, q, "T", 21, 3)
        b = greedy_lives(model, q, "T", 21, 3)
        self.assertEqual([l["actions"] for l in a], [l["actions"] for l in b])
        s = summarize(a)
        self.assertEqual(s["moves_per_life"] + s["per_life"], LIFE)
        self.assertAlmostEqual(sum(s["share"]), 1.0 if s["inspections"] else 0.0)

    def test_tiny_training_runs_and_logs(self):
        model = WorldModel(hidden=6, seed=5)
        q, log, trajectory = train_policy(model, "T", 17, "prudence", 1.0, episodes=12, replay=200, batch=8,
                                          warmup=20, target_every=50, checkpoint_every=6, checkpoint_lives=2)
        self.assertEqual(len(trajectory), 2)
        self.assertTrue(all(np.isfinite(v).all() for v in q.p.values()))
        self.assertIn("extrinsic", log[0])


if __name__ == "__main__":
    unittest.main()
