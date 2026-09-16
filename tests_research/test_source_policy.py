import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from menia.source_agent import SourceAgent
from menia.source_environment import SourceWorld, streams
from menia.source_monitor import SourceMonitor
from menia.source_policy import SourcePolicy
from research.learn_source_policy import exploratory_log, evaluate, matched_donor


class SourcePolicyTests(unittest.TestCase):
    def test_unverified_truth_changes_training_outcomes_but_not_predecision_inputs(self):
        frames, truth = streams(86, batch=8, length=9)
        rng = np.random.default_rng(87)
        actions = rng.random(truth.shape) < .5
        costs = rng.uniform(.05, .45, truth.shape)
        original = exploratory_log(SourceMonitor(), frames, truth, actions, costs)
        changed = exploratory_log(SourceMonitor(), frames, np.where(actions, truth, 1-truth), actions, costs)
        for i in (0, 1, 2):
            np.testing.assert_array_equal(original[i], changed[i])
        np.testing.assert_array_equal(original[3][actions.ravel()], costs[actions])
        np.testing.assert_array_equal(changed[3][~actions.ravel()], 1-original[3][~actions.ravel()])

    def test_current_and_future_truth_cannot_change_current_q(self):
        frames, truth = streams(88, batch=4, length=9)
        actions = np.ones_like(truth, dtype=bool)
        costs = np.full_like(truth, .25)
        changed = truth.copy()
        changed[5:] = 1-changed[5:]
        a = exploratory_log(SourceMonitor(), frames, truth, actions, costs)[0].reshape(truth.shape)
        b = exploratory_log(SourceMonitor(), frames, changed, actions, costs)[0].reshape(truth.shape)
        np.testing.assert_array_equal(a[:6], b[:6])

    def test_learns_from_selected_costs_and_roundtrips(self):
        q = np.tile(np.linspace(0, 1, 81), 4)
        cost = np.full_like(q, .25)
        action = np.repeat([0, 1, 0, 1], 81)
        # Artificial regression fixture: action 0 costs q, action 1 costs .25.
        outcome = np.where(action, cost, q)
        learned = SourcePolicy.fit(q, cost, action, outcome)
        np.testing.assert_array_equal(learned.choose([.1, .8], .25), [False, True])
        blind = SourcePolicy.fit(q, cost, action, outcome, blind=True)
        np.testing.assert_array_equal(blind.values(.1, .25), blind.values(.8, .25))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"policy.json"
            learned.save(path)
            restored = SourcePolicy.load(path)
            np.testing.assert_array_equal(learned.values(q, cost), restored.values(q, cost))
            raw = learned.payload()
            raw["coefficients"][0][0] = float("nan")
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                SourcePolicy.load(path)

    def test_policy_drives_agent_before_verification_and_is_logged(self):
        model = SourceMonitor()
        model.p["V"][:] = 0
        model.p["bo"][:] = -3  # Old rule would accept; policy below verifies.
        coefficients = np.zeros((10, 2))
        coefficients[:9, 0] = .4
        coefficients[9, 1] = 1
        agent = SourceAgent(model, policy=SourcePolicy(coefficients))
        try:
            def verify():
                events = agent.memory.recent(agent.episode)
                self.assertEqual([r["kind"] for r in events], ["prediction", "decision"])
                self.assertEqual(events[-1]["payload"]["rule"], "minimize learned immediate action cost")
                return False
            result = agent.step(SourceWorld().next_packet(), verify)
            self.assertTrue(result["verified"])
            self.assertIsNotNone(agent.snapshot()["policy"])
        finally:
            agent.memory.close()

    def test_batched_evaluator_matches_real_agent_trajectory(self):
        model = SourceMonitor()
        coefficients = np.zeros((10, 2))
        coefficients[:9, 0] = np.linspace(.05, .45, 9)
        coefficients[9, 1] = 1
        policy = SourcePolicy(coefficients)
        world = SourceWorld(91, length=20)
        frames, truth = streams(91, batch=1, length=20)
        agent = SourceAgent(model, policy=policy)
        try:
            results = [agent.step(world.next_packet(), world.verify) for _ in range(20)]
            # Add a duplicate episode only for a defined sample SD in evaluate.
            metrics, _ = evaluate(model, policy, np.repeat(frames, 2, axis=1), np.repeat(truth, 2, axis=1), .25)
            errors = np.asarray([r["attributed_external"] for r in results]) != truth[:, 0]
            verifies = np.asarray([r["verified"] for r in results])
            self.assertAlmostEqual(metrics["loss"], (errors + .25*verifies).mean())
            self.assertAlmostEqual(metrics["verification_rate"], verifies.mean())
        finally:
            agent.memory.close()

    def test_donors_preserve_first_prediction_and_restore_choices(self):
        q = np.asarray([.01, .49, .51, .9, .99])
        np.testing.assert_array_equal(q >= .5, matched_donor(q) >= .5)
        coefficients = np.random.default_rng(4).normal(size=(10, 2))
        policy = SourcePolicy(coefficients)
        original = policy.choose(q, .25)
        policy.choose(matched_donor(q), .25)
        np.testing.assert_array_equal(original, policy.choose(q, .25))


if __name__ == "__main__":
    unittest.main()
