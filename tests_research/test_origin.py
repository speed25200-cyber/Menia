import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from research.origin_env import (Atelier, encode, random_lives, motor_delta, INPUT, LIFE, N_MOVE, N_INSPECT,
                                 RING, SYMBOLS, DELTAS)
from research.origin_rational import OracleBayes
from research.origin_neural import (WorldModel, Imagination, decide, inspection_shares, hits_per_life,
                                    fit_probe, probe_predict, optimal_moves, replay_states, run_lives)


class EnvironmentTests(unittest.TestCase):
    def test_hidden_causes_never_leak_into_observations(self):
        for condition, shown in (("T", False), ("C3", False), ("C1", True)):
            env = Atelier(condition, 3)
            obs = env.reset()
            self.assertEqual(obs["d_shown"] >= 0, shown)
            for t in range(LIFE):
                obs, _, _ = env.step(t % 8)
                self.assertEqual(obs["d_shown"] >= 0, shown)
                if shown:
                    self.assertEqual(obs["d_shown"], env.d)
            self.assertEqual(set(obs), {"p", "g", "s", "last_action", "last_delta", "last_shift",
                                        "cue_channel", "cue_value", "d_shown"})

    def test_four_distinct_bodies(self):
        rows = [tuple(motor_delta(d, a) for a in range(N_MOVE)) for d in range(4)]
        self.assertEqual(len(set(rows)), 4)
        for row in rows:
            self.assertEqual(sorted(row), sorted(DELTAS))

    def test_cue_zero_is_informative_only_when_traced(self):
        def agreement(condition, k):
            hits = total = 0
            for n in range(400):
                env = Atelier(condition, 1000 + n)
                env.reset()
                obs, _, _ = env.step(N_MOVE + k)
                truth = env.d if k == 0 else env.e
                hits += obs["cue_value"] == truth
                total += 1
            return hits / total
        self.assertGreater(agreement("T", 0), 0.75)
        self.assertGreater(agreement("T", 1), 0.75)
        self.assertLess(agreement("C3", 0), 0.4)
        self.assertLess(agreement("T", 2), 0.4)
        self.assertLess(agreement("T", 3), 0.4)

    def test_sky_follows_world_cause_and_inspection_does_not_move(self):
        env = Atelier("T", 9)
        obs = env.reset()
        p, s = obs["p"], obs["s"]
        follows = 0
        for _ in range(200):
            obs, reward, _ = env.step(N_MOVE + 2)
            self.assertEqual(obs["p"], p)
            self.assertEqual(reward, 0)
            follows += obs["s"] == (s + env.e) % SYMBOLS
            s = obs["s"]
            if env.t >= LIFE:
                obs = env.reset(); p, s = obs["p"], obs["s"]
        self.assertGreater(follows / 200, 0.7)

    def test_encoding_is_one_hot_per_block_and_deterministic_data(self):
        env = Atelier("T", 2)
        obs = env.reset()
        x = encode(obs, 5)
        self.assertEqual(x.shape, (INPUT,))
        self.assertEqual(x.sum(), 9)  # p, g, s, last_action, last_delta, last_shift, cue_channel, cue_value, action
        X1, Y1, h1 = random_lives("T", 11, 3)
        X2, Y2, h2 = random_lives("T", 11, 3)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(h1, h2)


class OracleTests(unittest.TestCase):
    def test_exact_gains_point_to_the_right_cue(self):
        oracle = OracleBayes("T")
        gains = [oracle.eig(k, 0, 0) for k in range(N_INSPECT)]
        self.assertGreater(gains[0][0], 0.5)
        self.assertAlmostEqual(gains[0][1], 0.0)
        self.assertAlmostEqual(gains[1][0], 0.0)
        self.assertGreater(gains[1][1], 0.3)
        for k in (2, 3):
            self.assertAlmostEqual(gains[k][0], 0.0)
            self.assertAlmostEqual(gains[k][1], 0.0)
        for condition in ("C3",):
            self.assertAlmostEqual(OracleBayes(condition).eig(0, 0, 0)[0], 0.0)
        shown = OracleBayes("C1")
        shown.observe_shown(2)
        self.assertAlmostEqual(shown.eig(0, 0, 0)[0], 0.0)

    def test_a_single_move_identifies_the_body(self):
        oracle = OracleBayes("T")
        oracle.observe_move(0, 1, (0 + motor_delta(3, 1)) % RING)
        np.testing.assert_allclose(oracle.post.sum(axis=1), [0, 0, 0, 1])

    def test_decision_rule_uses_the_selected_gain_only(self):
        rng = np.random.default_rng(0)
        gains = np.array([[0.8, 0.0, 1.0], [0.0, 0.5, 1.0], [0.0, 0.0, 1.386], [0.0, 0.0, 1.386]])
        hit = np.array([0.1, 0.2, 0.0, 0.0])
        dist = np.array([2.0, 1.0, 3.0, 3.0])
        self.assertEqual(decide("self", gains, hit, dist, rng), N_MOVE + 0)
        self.assertEqual(decide("world", gains, hit, dist, rng), N_MOVE + 1)
        self.assertIn(decide("surprise", gains, hit, dist, rng), (N_MOVE + 2, N_MOVE + 3))
        self.assertEqual(decide("none", gains, hit, dist, rng), 1)
        confident = np.array([0.9, 0.0, 0.0, 0.0])
        self.assertEqual(decide("self", gains, confident, dist, rng), 0)
        self.assertEqual(decide("all", gains, confident, dist, rng), N_MOVE + 0)


class ModelTests(unittest.TestCase):
    def test_gradients_against_finite_differences(self):
        model = WorldModel(hidden=5, seed=3)
        X, Y, _ = random_lives("T", 5, 2)
        X, Y = X[:4], {k: v[:4] for k, v in Y.items()}
        _, grads = model.loss_and_grad(X, Y)
        for key, parameter in model.p.items():
            index = tuple(0 for _ in parameter.shape)
            old = parameter[index]
            parameter[index] = old + 1e-5
            plus, _ = model.loss_and_grad(X, Y)
            parameter[index] = old - 1e-5
            minus, _ = model.loss_and_grad(X, Y)
            parameter[index] = old
            with self.subTest(parameter=key):
                self.assertAlmostEqual(grads[key][index], (plus - minus) / 2e-5, places=6)

    def test_roundtrip_replay_and_corruption(self):
        model = WorldModel(hidden=6, seed=4)
        lives, states = run_lives(model, "T", "none", 21, 2, keep_states=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path, {})
            loaded = WorldModel.load(path)
            np.testing.assert_allclose(replay_states(loaded, lives[0]), states[0])
            bad = json.loads(path.read_text())
            bad["parameters"]["bz"] = [float("nan")] * 6
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                WorldModel.load(path)

    def test_imagination_is_causal_and_bounded(self):
        model = WorldModel(hidden=8, seed=5)
        env = Atelier("T", 8)
        obs = env.reset()
        gains = Imagination(model).inspection_gains(model.zero(), obs)
        self.assertEqual(gains.shape, (N_INSPECT, 3))
        self.assertTrue((gains >= 0).all())
        self.assertTrue((gains[:, :2] <= np.log(SYMBOLS) + 1e-9).all())

    def test_shares_hits_probe_and_optimal_moves(self):
        lives = [{"actions": [4, 4, 5, 0, 6], "rewards": [0, 0, 1, 0, 1], "d": 0},
                 {"actions": [0, 1, 2, 3, 0], "rewards": [0, 0, 0, 0, 0], "d": 1}]
        shares = inspection_shares(lives)
        self.assertEqual(shares["inspections"], 4)
        self.assertEqual(shares["share"], [0.5, 0.25, 0.25, 0.0])
        self.assertEqual(hits_per_life(lives), 1.0)
        self.assertEqual(inspection_shares([{"actions": [0], "rewards": [0], "d": 0}])["share"], [0, 0, 0, 0])
        rng = np.random.default_rng(1)
        features = rng.normal(size=(200, 3))
        labels = (features[:, 0] > 0).astype(int) + 2 * (features[:, 1] > 0)
        W = fit_probe(features, labels)
        self.assertGreater((probe_predict(W, features) == labels).mean(), 0.95)
        self.assertEqual(optimal_moves(0, 0, 2), {3})          # delta +2 is action 3 under D=0
        self.assertEqual(optimal_moves(1, 0, 2), {2})          # shifted body
        self.assertTrue(optimal_moves(0, 0, 4) <= {0, 3})      # unreachable: reduce distance


if __name__ == "__main__":
    unittest.main()
