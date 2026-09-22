import tempfile
from pathlib import Path
import unittest
import numpy as np
from menia.source_monitor import SourceMonitor
from menia.source_agent import SourceAgent
from menia.source_environment import SourceWorld, streams
from research.evaluate_source_monitor import training_features


class SourceAgentTests(unittest.TestCase):
    def test_unverified_truth_cannot_enter_training_inputs_or_targets(self):
        frames, truth = streams(51, batch=8, length=9)
        mask = np.random.default_rng(52).random(truth.shape) < .5
        altered = np.where(mask, truth, 1-truth)
        x, target = training_features(frames, truth, mask)
        x2, target2 = training_features(frames, altered, mask)
        np.testing.assert_array_equal(x, x2)
        np.testing.assert_array_equal(target, target2)
        self.assertTrue(np.all(x[0, :, 3:] == 0))
        self.assertTrue(np.all(target[~mask] == 0))

    def test_prediction_and_decision_precede_verification(self):
        model = SourceMonitor()
        model.p["V"][:] = 0
        model.p["bo"][:] = 0
        agent = SourceAgent(model)
        try:
            def verify():
                events = agent.memory.recent(agent.episode)
                self.assertEqual([r["kind"] for r in events], ["prediction", "decision"])
                self.assertEqual(events[0]["payload"]["p_external"], .5)
                return True
            result = agent.step(SourceWorld().next_packet(), verify)
            self.assertTrue(result["verified"])
            self.assertTrue(result["attributed_external"])
            self.assertEqual(agent.feedback, (1., 1.))
            self.assertEqual(agent.memory.event(result["memory_event"])["kind"], "report")
            self.assertLess(result["decision_event"], result["verification_event"])
        finally:
            agent.memory.close()

    def test_inferred_source_is_not_a_certified_observation(self):
        model = SourceMonitor()
        model.p["V"][:] = 0
        model.p["bo"][:] = -3
        agent = SourceAgent(model)
        packet = SourceWorld().next_packet()
        try:
            def forbidden():
                self.fail("An unchosen verification exposed truth")
            result = agent.step(packet, forbidden)
            self.assertFalse(result["verified"])
            self.assertFalse(agent.memory.event(result["memory_event"])["payload"]["verified"])
            self.assertIsNone(agent.memory.latest(agent.episode, "source:0"))
            with self.assertRaises(ValueError):
                agent.step(packet, forbidden)
        finally:
            agent.memory.close()

    def test_failed_verification_stops_without_replaying(self):
        model = SourceMonitor()
        model.p["V"][:] = model.p["bo"][:] = 0
        agent = SourceAgent(model)
        try:
            with self.assertRaises(ValueError):
                agent.step(SourceWorld().next_packet(), lambda: "not a receipt")
            self.assertTrue(agent.stopped)
            with self.assertRaises(RuntimeError):
                agent.step(SourceWorld().next_packet(), lambda: True)
            self.assertIsNone(agent.memory.latest(agent.episode, "source:0"))
        finally:
            agent.memory.close()

    def test_checkpoints_preserve_sequence_predictions(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(7, 3, 5))
        with tempfile.TemporaryDirectory() as folder:
            for kind in ("recurrent", "reset", "window"):
                model = SourceMonitor(kind)
                path = Path(folder)/f"{kind}.json"
                model.save(path)
                restored = SourceMonitor.load(path)
                s1, s2 = model.zero(3), restored.zero(3)
                for row in x:
                    s1, q1 = model.step(row, s1)
                    s2, q2 = restored.step(row, s2)
                    np.testing.assert_array_equal(q1, q2)


if __name__ == "__main__":
    unittest.main()
