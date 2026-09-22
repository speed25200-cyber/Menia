import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research.evaluate_reliability import delayed_predictions, evaluate_policy
from research.recurrent import RecurrentMemory
from research.reliability import RecallPolicy, calibrate, model_fingerprint, wilson_lower
from research.session import CognitiveSession


class ReliabilityTests(unittest.TestCase):
    def test_unknown_is_not_a_memory_and_clear_removes_provenance(self):
        session = CognitiveSession(RecurrentMemory())
        for _ in range(4):
            event = session.observe()
            self.assertIsNone(event['answer_symbol'])
            self.assertIsNone(event['inferred_last_symbol'])
            self.assertEqual(event['uncertainty_reason'], 'no_observation')
        session.observe(2)
        event = session.observe()
        self.assertEqual(event['uncertainty_reason'], 'uncalibrated_recall')
        session.clear()
        self.assertIsNone(session.observation_age)
        self.assertEqual(session.observe()['uncertainty_reason'], 'no_observation')

    def test_confident_wrong_candidate_does_not_override_delay_or_observation(self):
        model = RecurrentMemory()
        # Force a confident class-zero prediction regardless of the input.
        model.p['V'][:] = 0
        model.p['bo'][:] = [20, -20, -20, -20]
        policy = RecallPolicy(model_fingerprint(model), max_age=1)
        session = CognitiveSession(model, policy=policy)
        event = session.observe(2)
        self.assertEqual(event['answer_symbol'], 2)
        self.assertEqual(event['answer_source'], 'observation')
        self.assertEqual(event['model_candidate'], 0)
        first = session.observe()
        self.assertEqual(first['answer_symbol'], 0)
        # An in-domain policy is not an infallible detector of wrong memories.
        self.assertFalse(session.assess(2)['answer_correct'])
        event = session.observe()
        self.assertTrue(event['abstained'])
        self.assertEqual(event['uncertainty_reason'], 'outside_calibrated_delay')
        assessment = session.assess(2)
        self.assertFalse(assessment['answered'])
        self.assertIsNone(assessment['answer_correct'])
        self.assertFalse(assessment['correct'])

    def test_fit_stops_at_first_bad_recall_age_and_checks_each_class(self):
        model = RecurrentMemory()
        labels = np.tile(np.arange(4), 128)
        p = np.broadcast_to(np.eye(4)[labels], (6, len(labels), 4)).copy()
        # Age zero is a present observation, not part of delayed recall fitting.
        p[0] = np.roll(p[0], 1, axis=-1)
        # One bad class cannot be hidden in a high aggregate accuracy.
        p[3, labels == 3] = [1, 0, 0, 0]
        policy, rows = calibrate(model, p, labels)
        self.assertEqual(policy.max_age, 2)
        self.assertFalse(rows[3]['passes_criterion'])
        self.assertTrue(rows[4]['passes_criterion'])
        # A missing class provides no basis for accepting a delay.
        missing, _ = calibrate(model, p[:, labels != 3], labels[labels != 3])
        self.assertEqual(missing.max_age, -1)

    def test_policy_binding_roundtrip_and_in_place_weight_change(self):
        model = RecurrentMemory()
        policy = RecallPolicy(model_fingerprint(model), 8)
        session = CognitiveSession(model, policy=policy)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'policy.json'
            policy.save(path)
            self.assertEqual(RecallPolicy.load(path, model), policy)
            with self.assertRaises(ValueError):
                RecallPolicy.load(path, RecurrentMemory(seed=29))
            data = json.loads(path.read_text())
            data['min_probability'] = float('nan')
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                RecallPolicy.load(path, model)
        model.p['bo'][0] += 1
        with self.assertRaises(ValueError):
            session.observe(2)
        self.assertEqual(session.sequence, 0)

    def test_no_answer_is_undefined_accuracy_not_perfect_accuracy(self):
        model = RecurrentMemory()
        p, labels = delayed_predictions(model, 21, batch=8, max_age=2)
        policy = RecallPolicy(model_fingerprint(model), -1)
        rows = evaluate_policy(p, labels, policy, [1, 2])
        for row in rows:
            score = row['calibrated_policy']
            self.assertEqual(score['coverage'], 0)
            self.assertIsNone(score['accuracy_when_answered'])

    def test_state_interventions_preserve_targets_and_change_only_internal_state(self):
        model = RecurrentMemory()
        p, labels = delayed_predictions(model, 31, batch=16, max_age=3)
        q, shuffled_labels = delayed_predictions(model, 31, batch=16, max_age=3,
                                                intervention='shuffle')
        np.testing.assert_array_equal(labels, shuffled_labels)
        self.assertFalse(np.array_equal(p, q))
        # Shuffling states preserves the population of predictions at each age.
        np.testing.assert_allclose(np.sort(p, axis=1), np.sort(q, axis=1))
        reset, reset_labels = delayed_predictions(model, 31, batch=16, max_age=3,
                                                 intervention='reset')
        np.testing.assert_array_equal(labels, reset_labels)
        np.testing.assert_allclose(reset, np.broadcast_to(reset[:, :1], reset.shape))

    def test_assessment_cannot_change_future_predictions_or_policy(self):
        model = RecurrentMemory()
        policy = RecallPolicy(model_fingerprint(model), 8)
        left, right = CognitiveSession(model, policy=policy), CognitiveSession(model, policy=policy)
        left.observe(2)
        right.observe(2)
        left.assess(0)
        right.assess(3)
        self.assertEqual(left.observe(), right.observe())
        self.assertEqual(left.policy, right.policy)

    def test_wilson_requires_evidence(self):
        self.assertEqual(wilson_lower(0, 0), 0)
        self.assertLess(wilson_lower(10, 10), .95)
        self.assertGreater(wilson_lower(128, 128), .95)


if __name__ == '__main__':
    unittest.main()
