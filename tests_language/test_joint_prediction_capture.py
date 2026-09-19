"""Timing, identity and isolation checks on a tiny random Qwen, not task performance."""
import copy
import unittest

import torch

from research.activation_monitor import messages, validate_state
from research.activation_monitor_gpu import sample_with_states
from research.cross_model_gpu import generate_text
from research.cross_model_prediction import SETTINGS
from research.joint_prediction_capture import sample_with_joint_capture
from research.output_confidence_trace import _ObservedTokenizer, trace_generation
from tests_language.test_cross_model_gpu import CrossModelBackendTests


class JointCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CrossModelBackendTests.setUpClass()
        cls.model = CrossModelBackendTests.model
        cls.tokenizer = CrossModelBackendTests.tokenizer
        cls.task = dict(question='hello hello hello hello', seed=421)
        cls.settings = dict(SETTINGS, max_new_tokens=8)

    def test_plain_states_confidence_and_rng_all_match(self):
        prompt = messages(self.task)
        seen = _ObservedTokenizer(self.tokenizer)
        baseline = generate_text(self.model, seen, prompt, self.task['seed'], settings=self.settings)
        expected_rng = torch.random.get_rng_state().clone()
        old_states = []
        self.assertEqual(sample_with_states(self.model, self.tokenizer, self.task, old_states.append,
                                           settings=self.settings), baseline)
        separate = trace_generation(self.model, self.tokenizer, prompt, self.task['seed'], settings=self.settings)
        captured = []
        actual = sample_with_joint_capture(self.model, self.tokenizer, prompt, self.task['seed'],
                                          captured.append, settings=self.settings)
        self.assertEqual(actual[:2], baseline)
        self.assertEqual(actual, separate)
        self.assertEqual(actual[2]['completion']['tokenIds'], seen.generated_ids)
        self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]['state'], old_states[0])
        self.assertEqual(captured[0]['preAnswer'], actual[2]['preAnswer'])
        self.assertEqual(set(captured[0]), {'state', 'preAnswer'})
        validate_state(captured[0]['state'])

    def test_callback_precedes_sampling_and_seed_does_not_enter_features(self):
        captures = []
        for seed in (421, 972):
            torch.manual_seed(seed)
            expected_rng = torch.random.get_rng_state().clone()
            def capture(value):
                self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
                self.assertNotIn('completion', value)
                self.assertNotIn('tokenIds', value['preAnswer'])
                captures.append(copy.deepcopy(value))
            sample_with_joint_capture(self.model, self.tokenizer, messages(self.task), seed,
                                      capture, settings=self.settings)
        self.assertEqual(captures[0], captures[1])

    def test_callback_failure_removes_every_hook_before_any_sample(self):
        seed = 811
        torch.manual_seed(seed)
        expected_rng = torch.random.get_rng_state().clone()
        before = [len(m._forward_hooks) for m in self.model.modules()]
        def fail(value):
            raise RuntimeError('Synthetic recording failure')
        with self.assertRaisesRegex(RuntimeError, 'Synthetic recording failure'):
            sample_with_joint_capture(self.model, self.tokenizer, messages(self.task), seed,
                                      fail, settings=self.settings)
        self.assertEqual(before, [len(m._forward_hooks) for m in self.model.modules()])
        self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
