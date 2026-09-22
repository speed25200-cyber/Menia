import copy
import unittest
from unittest.mock import patch

import torch
from transformers import GenerationConfig

from research.confidence_action_decode import decode_branch
from research.confidence_cached_action_decode import prefill_prefix, decode_cached_branch
from research.confidence_prefix_interventions import tensor_hash
from tests_language import test_cross_model_gpu as fixture


class CachedActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(242025)
        fixture.CrossModelBackendTests.setUpClass()
        cls.model = fixture.CrossModelBackendTests.model
        cls.tokenizer = fixture.CrossModelBackendTests.tokenizer
        cls.prefix = [1, 3, 4]

    def branch(self, suffix=(5, 3)):
        return dict(inputIds=self.prefix+list(suffix), candidateTokenIds=[3, 4], negative='verify', positive='accept')

    def capture(self, **kwargs):
        return prefill_prefix(self.model, self.prefix, 0, model_state_id='random-fixture-v1', **kwargs)

    def decode(self, saved, branch=None, **kwargs):
        return decode_cached_branch(self.model, self.tokenizer, branch or self.branch(), saved,
                                    model_state_id='random-fixture-v1', **kwargs)

    def test_baseline_matches_native_generation_and_full_replay_with_multiple_eos(self):
        saved = self.capture()
        for eos in (2, [2, 1]):
            with self.subTest(eos=eos), patch.object(self.model.generation_config, 'eos_token_id', eos):
                actual = self.decode(saved)
                replay = decode_branch(self.model, self.tokenizer, self.branch(), self.prefix, 0)
                ids = torch.tensor([self.branch()['inputIds']])
                with torch.inference_mode():
                    native = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                        generation_config=GenerationConfig(do_sample=False, use_cache=True,
                            max_new_tokens=2, eos_token_id=eos, pad_token_id=1))
                self.assertEqual(actual['tokenIds'], native[0, ids.shape[1]:].tolist())
                self.assertEqual(actual['tokenIds'], replay['tokenIds'])
                self.assertAlmostEqual(actual['conditionalPositive'], replay['conditionalPositive'], delta=1e-6)
                self.assertEqual(actual['passes'][0]['inputTokens'], 2)
                if len(actual['passes']) == 2:
                    self.assertEqual(actual['passes'][1]['inputTokens'], 1)
                    self.assertEqual(actual['passes'][1]['cacheTokensBefore'], ids.shape[1])

    def test_forks_do_not_contaminate_each_other_or_mutate_weights_rng_or_inputs(self):
        saved = self.capture(); first = self.branch(); second = self.branch((4, 5, 3))
        original = copy.deepcopy((first, second)); weights = {k: tensor_hash(v) for k, v in self.model.state_dict().items()}
        rng = torch.get_rng_state().clone(); cache = [tensor_hash(v) for pair in saved.kv for v in pair]
        a = self.decode(saved, first); b = self.decode(saved, second)
        self.assertEqual(self.decode(saved, first), a)
        self.assertEqual(self.decode(saved, second), b)
        self.assertEqual((first, second), original)
        self.assertEqual(cache, [tensor_hash(v) for pair in saved.kv for v in pair])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(weights, {k: tensor_hash(v) for k, v in self.model.state_dict().items()})
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.model.layers))

    def test_one_historical_patch_matches_reapplied_patch_and_last_block_has_no_future_effect(self):
        basis = torch.eye(32)[:, :2]
        for layer in (0, 1):
            original = prefill_prefix(self.model, self.prefix, layer, model_state_id='random-fixture-v1')
            donor = original.capture['before'].clone(); donor[:2] += 1.
            saved = prefill_prefix(self.model, self.prefix, layer, model_state_id='random-fixture-v1', donor=donor, basis=basis)
            baseline = self.decode(original); actual = self.decode(saved)
            replay = decode_branch(self.model, self.tokenizer, self.branch(), self.prefix, layer, donor=donor, basis=basis)
            self.assertEqual(actual['tokenIds'], replay['tokenIds'])
            for a, b in zip(actual['candidateLogits'], replay['candidateLogits']):
                self.assertAlmostEqual(a, b, delta=1e-6)
            self.assertGreater(saved.capture['displacementNorm'], 0)
            self.assertTrue(saved.capture['untouchedTokensEqual'])
            if layer == 0:
                self.assertNotEqual(actual['candidateLogits'], baseline['candidateLogits'])
            else:
                self.assertEqual(actual['candidateLogits'], baseline['candidateLogits'])
                self.assertEqual(original.cache_hash, saved.cache_hash)
            self.assertTrue(all(not block._forward_hooks for block in self.model.model.layers))

    def test_stale_state_corrupted_cache_prefix_and_budget_are_rejected(self):
        saved = self.capture()
        with self.assertRaises(ValueError): self.decode(saved, self.branch((5,)), max_input_tokens=4)
        broken = self.branch(); broken['inputIds'][0] = 5
        with self.assertRaises(ValueError): self.decode(saved, broken)
        with self.assertRaises(ValueError):
            decode_cached_branch(self.model, self.tokenizer, self.branch(), saved, model_state_id='other-adapter')
        with self.assertRaises(ValueError):
            decode_cached_branch(copy.deepcopy(self.model), self.tokenizer, self.branch(), saved, model_state_id='random-fixture-v1')
        with torch.inference_mode(): saved.kv[0][0].add_(1.)
        with self.assertRaises(ValueError): self.decode(saved)
        changed_model = copy.deepcopy(self.model)
        stale = prefill_prefix(changed_model, self.prefix, 0, model_state_id='modified')
        with torch.no_grad(): next(changed_model.parameters()).add_(.01)
        with self.assertRaises(ValueError):
            decode_cached_branch(changed_model, self.tokenizer, self.branch(), stale, model_state_id='modified')

    def test_prefill_hook_removed_after_exception_and_unsupported_cache_rejected(self):
        donor = torch.full((32,), float('nan'))
        with self.assertRaises(ValueError): self.capture(donor=donor, basis=torch.eye(32)[:, :2])
        self.assertTrue(all(not block._forward_hooks for block in self.model.model.layers))
        with patch.object(self.model.config, 'layer_types', ['sliding_attention', 'full_attention']):
            with self.assertRaises(ValueError): self.capture()


if __name__ == '__main__': unittest.main()
