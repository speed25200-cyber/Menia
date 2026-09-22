"""Numerical/instrumentation checks; tiny random Qwen, no Menia result."""
import math
import unittest

import torch

from research.cross_model_gpu import generate_text
from research.cross_model_prediction import SETTINGS
from research.output_confidence_trace import (
    _ObservedTokenizer, distribution_summary, trace_generation,
)
from tests_language import test_cross_model_gpu as fixtures


class OutputConfidenceTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.CrossModelBackendTests.setUpClass()
        cls.model = fixtures.CrossModelBackendTests.model
        cls.tokenizer = fixtures.CrossModelBackendTests.tokenizer

    def test_hand_computed_full_vocabulary_entropy(self):
        value = distribution_summary(torch.tensor([0., math.log(2), math.log(3)], dtype=torch.float64))
        self.assertEqual(value["topTokenId"], 2)
        self.assertAlmostEqual(value["maxProbability"], .5)
        self.assertAlmostEqual(value["topTwoMargin"], 1/6)
        entropy = -sum(p * math.log(p) for p in (1/6, 1/3, 1/2))
        self.assertAlmostEqual(value["entropyNats"], entropy)
        self.assertAlmostEqual(value["normalizedEntropy"], entropy/math.log(3))
        for invalid in (torch.ones((1, 3)), torch.tensor([0., float("inf")]), torch.ones(1)):
            with self.assertRaises(ValueError):
                distribution_summary(invalid)

    def test_exact_tokens_metrics_rng_and_pre_answer_logits(self):
        messages = [dict(role="user", content="hello hello")]
        settings = dict(SETTINGS, max_new_tokens=8)
        for seed in (19, 92):
            tokenizer = _ObservedTokenizer(self.tokenizer)
            baseline = generate_text(self.model, tokenizer, messages, seed, settings=settings)
            rng_after = torch.random.get_rng_state().clone()
            captures = []
            text, metrics, trace = trace_generation(self.model, self.tokenizer, messages, seed,
                settings=settings, on_prefill=captures.append)
            self.assertEqual((text, metrics), baseline)
            self.assertEqual(trace["completion"]["tokenIds"], tokenizer.generated_ids)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_after))
            self.assertEqual(captures, [trace["preAnswer"]])
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                                    return_token_type_ids=False)
            with torch.inference_mode():
                logits = self.model(**inputs).logits[0, -1]
            expected = distribution_summary(logits)
            for field in ("maxProbability", "topTwoMargin", "entropyNats", "normalizedEntropy"):
                self.assertAlmostEqual(trace["preAnswer"][field], expected[field], places=6)
            first_id = tokenizer.generated_ids[0]
            first_logp = float(logits.double().log_softmax(0)[first_id])
            self.assertAlmostEqual(trace["completion"]["tokenLogProbabilities"][0], first_logp, places=6)
            # Recompute every prefix without a KV cache. This independently
            # catches a one-step shift between head outputs and generated IDs.
            prefix = inputs.input_ids.clone()
            expected_logp = []
            for token_id in tokenizer.generated_ids:
                with torch.inference_mode():
                    row = self.model(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                                     use_cache=False).logits[0, -1].double()
                expected_logp.append(float(row.log_softmax(0)[token_id]))
                prefix = torch.cat((prefix, torch.tensor([[token_id]])), dim=1)
            for measured, expected in zip(trace["completion"]["tokenLogProbabilities"], expected_logp):
                self.assertAlmostEqual(measured, expected, places=6)
            self.assertAlmostEqual(trace["completion"]["sumLogProbability"], sum(expected_logp), places=6)

    def test_eos_is_retained_and_raw_distribution_is_not_filtered(self):
        # Force a known distribution through both paths; the token sampler still
        # runs with the frozen top-k/top-p/temperature settings.
        def fixed_head(module, inputs, output):
            result = torch.zeros_like(output)
            result[..., self.tokenizer.eos_token_id] = 10.
            return result
        handle = self.model.lm_head.register_forward_hook(fixed_head)
        try:
            text, metrics, trace = trace_generation(self.model, self.tokenizer,
                [dict(role="user", content="hello")], 17)
        finally:
            handle.remove()
        self.assertEqual(text, "")
        self.assertEqual(metrics["outputTokens"], 1)
        self.assertEqual(trace["completion"]["tokenIds"], [self.tokenizer.eos_token_id])
        expected = 1 / (1 + 63*math.exp(-10))
        self.assertAlmostEqual(trace["preAnswer"]["maxProbability"], expected)
        self.assertLess(trace["preAnswer"]["maxProbability"], 1.)
        self.assertAlmostEqual(trace["completion"]["sumLogProbability"], math.log(expected))

    def test_prefill_occurs_before_sampling_and_failure_removes_hook(self):
        seed = 121
        torch.manual_seed(seed)
        expected_rng = torch.random.get_rng_state().clone()
        before = len(self.model.lm_head._forward_hooks)
        def fail(value):
            self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
            raise RuntimeError("Recorder unavailable")
        with self.assertRaisesRegex(RuntimeError, "Recorder unavailable"):
            trace_generation(self.model, self.tokenizer, [dict(role="user", content="hello")],
                             seed, on_prefill=fail)
        self.assertEqual(before, len(self.model.lm_head._forward_hooks))


if __name__ == "__main__":
    unittest.main()
