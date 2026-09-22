import unittest

import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from research.confidence_cached_action_decode import _cache_hash, prefill_prefix
from research.prospective_cache_interventions import inverse_permutation, permute_prefix_cache


class CachePermutationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1701)
        self.model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=128, attention_dropout=0., _attn_implementation='sdpa')).eval()
        self.identity = 'tiny-random-test-only'
        self.saved = prefill_prefix(self.model, [1,7,12,4,9,3], 0, model_state_id=self.identity)
        self.permutation = [4,1,5,0,3,2]

    def transform(self, snapshot=None, **kwargs):
        arguments = dict(model_state_id=self.identity, layers=[0,1], permutation=self.permutation, mode='values_only')
        arguments.update(kwargs)
        return permute_prefix_cache(self.model, self.saved if snapshot is None else snapshot, **arguments)

    def logits(self, saved):
        cache = DynamicCache.from_legacy_cache(tuple(tuple(v.clone() for v in pair) for pair in saved.kv))
        ids = torch.tensor([[2,8]])
        with torch.inference_mode():
            return self.model(input_ids=ids, attention_mask=torch.ones((1,8),dtype=torch.long),
                position_ids=torch.tensor([[6,7]]), cache_position=torch.tensor([6,7]),
                past_key_values=cache, use_cache=True).logits

    def test_values_only_preserves_keys_source_tokens_and_restores_bitwise(self):
        changed, record = self.transform(layers=[0])
        self.assertEqual(record['changedTensors'], 1)
        self.assertGreater(record['absoluteL2'], 0)
        self.assertEqual(changed.prefix, self.saved.prefix)
        self.assertEqual(_cache_hash(self.saved.kv), self.saved.cache_hash)
        self.assertTrue(torch.equal(changed.kv[0][0], self.saved.kv[0][0]))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(changed.kv[1],self.saved.kv[1])))
        self.assertTrue(torch.equal(changed.kv[0][1],self.saved.kv[0][1].index_select(-2,torch.tensor(self.permutation))))
        restored, _ = self.transform(changed,layers=[0],permutation=inverse_permutation(self.permutation))
        self.assertEqual(restored.cache_hash,self.saved.cache_hash)
        self.assertTrue(torch.equal(self.logits(restored),self.logits(self.saved)))
        self.assertTrue(all(a.data_ptr()!=b.data_ptr() for pair,original in zip(changed.kv,self.saved.kv) for a,b in zip(pair,original)))

    def test_joint_permutation_is_an_attention_control_not_a_value_only_effect(self):
        joint, _ = self.transform(mode='keys_and_values')
        changed, _ = self.transform()
        baseline = self.logits(self.saved)
        torch.testing.assert_close(self.logits(joint),baseline,rtol=0,atol=1e-6)
        self.assertGreater(float((self.logits(changed)-baseline).abs().max()),1e-5)
        restored, _ = self.transform(joint,mode='keys_and_values',permutation=inverse_permutation(self.permutation))
        self.assertEqual(restored.cache_hash,self.saved.cache_hash)

    def test_rejects_ambiguous_permutations_and_changed_models(self):
        for permutation in ([0,0,2,3,4,5],[0,1,2],[False,1,2,3,4,5]):
            with self.assertRaises(ValueError): self.transform(permutation=permutation)
        for layers in ([],[1,0],[0,0],[2],[True]):
            with self.assertRaises(ValueError): self.transform(layers=layers)
        with self.assertRaises(ValueError): self.transform(mode='arbitrary')
        with self.assertRaises(ValueError): self.transform(model_state_id='other')
        with torch.no_grad(): next(self.model.parameters()).add_(.01)
        with self.assertRaises(ValueError): self.transform()


if __name__ == '__main__': unittest.main()
