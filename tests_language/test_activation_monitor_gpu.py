"""Offline hooks on a tiny random Qwen3; not pretrained-model performance."""
import unittest

from research.activation_monitor_gpu import sample_with_states
from research.activation_monitor import messages, validate_state
from research.cross_model_prediction import SETTINGS
from research.cross_model_gpu import generate_text
from tests_language import test_cross_model_gpu as fixtures


class ActivationHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.CrossModelBackendTests.setUpClass()
        cls.model = fixtures.CrossModelBackendTests.model
        cls.tokenizer = fixtures.CrossModelBackendTests.tokenizer

    def task(self, seed=41):
        return dict(question="hello hello hello hello", seed=seed)

    def test_hooks_capture_before_logits_once_and_leave_generation_identical(self):
        task = self.task()
        settings = dict(SETTINGS, max_new_tokens=8)
        baseline = generate_text(self.model, self.tokenizer, messages(task), task["seed"], settings=settings)
        captures = []
        def before_head(module, inputs):
            self.assertEqual(len(captures), 1)
        handle = self.model.lm_head.register_forward_pre_hook(before_head)
        try:
            observed = sample_with_states(self.model, self.tokenizer, task, captures.append, settings=settings)
        finally:
            handle.remove()
        self.assertEqual(observed, baseline)
        self.assertEqual(len(captures), 1)
        validate_state(captures[0])

    def test_prefix_states_do_not_contain_future_sampling_seed(self):
        captures = []
        for seed in (41, 92):
            sample_with_states(self.model, self.tokenizer, self.task(seed), captures.append,
                               settings=dict(SETTINGS, max_new_tokens=8))
        self.assertEqual(captures[0], captures[1])

    def test_failed_capture_removes_hooks_and_stops_before_sampling(self):
        called = []
        def fail(state):
            called.append(1)
            raise RuntimeError("Synthetic journal failure")
        before = [len(m._forward_hooks) for m in self.model.modules()]
        with self.assertRaisesRegex(RuntimeError, "Synthetic journal"):
            sample_with_states(self.model, self.tokenizer, self.task(), fail)
        self.assertEqual(called, [1])
        self.assertEqual(before, [len(m._forward_hooks) for m in self.model.modules()])


if __name__ == "__main__":
    unittest.main()
