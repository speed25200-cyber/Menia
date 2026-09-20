import copy
import unittest
from unittest.mock import patch

import torch
from transformers import GenerationConfig

from research.confidence_action_decode import decode_branch, parse_native_tokens
from research.confidence_prefix_interventions import forward_at_shared_prefix
from tests_language import test_cross_model_gpu as fixture


class NativeActionDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(242)
        fixture.CrossModelBackendTests.setUpClass()
        cls.model = fixture.CrossModelBackendTests.model; cls.tokenizer = fixture.CrossModelBackendTests.tokenizer

    def branch(self):
        return dict(inputIds=[1, 3, 4, 5, 3], candidateTokenIds=[3, 4], negative='verify', positive='accept')

    def test_actual_greedy_output_equals_transformers_and_preserves_weights_rng_and_inputs(self):
        model = self.model; branch = self.branch(); original = copy.deepcopy(branch)
        weights = {k: p.detach().clone() for k, p in model.named_parameters()}; rng = torch.get_rng_state().clone()
        result = decode_branch(model, self.tokenizer, branch, [1, 3, 4], 0)
        self.assertEqual(branch, original)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(p, weights[k]) for k, p in model.named_parameters()))
        ids = torch.tensor([branch['inputIds']])
        config = GenerationConfig(do_sample=False, use_cache=False, max_new_tokens=2,
                                  eos_token_id=2, pad_token_id=1)
        with torch.inference_mode():
            native = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=config)
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1].double()
        self.assertEqual(result['tokenIds'], native[0, len(branch['inputIds']):].tolist())
        self.assertEqual(result['tokenIds'][0], int(logits.argmax()))
        self.assertAlmostEqual(result['conditionalPositive'], float(torch.softmax(logits[[3, 4]], dim=0)[1]), delta=1e-7)
        self.assertAlmostEqual(result['candidateMass'], float(torch.softmax(logits, dim=0)[[3, 4]].sum()), delta=1e-7)
        self.assertFalse(result['settings']['candidateRestriction'])

    def test_strict_code_eos_parsing_and_semantic_remapping(self):
        for invalid in ([], [3], [3, 3], [3, 2, 2], [7, 2], [2]):
            self.assertFalse(parse_native_tokens(invalid, [3, 4], ['verify', 'accept'], 2)['validNativeResponse'])
        first = parse_native_tokens([3, 2], [3, 4], ['verify', 'accept'], 2)
        swapped = parse_native_tokens([3, 2], [4, 3], ['verify', 'accept'], 2)
        self.assertEqual(first['decision'], 'verify'); self.assertEqual(swapped['decision'], 'accept')
        self.assertTrue(first['validNativeResponse'] and swapped['validNativeResponse'])
        alternate = parse_native_tokens([3, 1], [3, 4], ['verify', 'accept'], [2, 1])
        self.assertTrue(alternate['validNativeResponse'])
        with self.assertRaises(ValueError): parse_native_tokens([3, 2], [3, 4], ['verify', 'accept'], [2, 2])

    def test_multiple_configured_eos_match_the_native_generator(self):
        branch = self.branch(); ids = torch.tensor([branch['inputIds']])
        with patch.object(self.model.generation_config, 'eos_token_id', [2, 1]):
            result = decode_branch(self.model, self.tokenizer, branch, [1, 3, 4], 0)
            config = GenerationConfig(do_sample=False, use_cache=False, max_new_tokens=2, eos_token_id=[2, 1], pad_token_id=1)
            with torch.inference_mode():
                reference = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=config)
        self.assertEqual(result['eosTokenIds'], [2, 1])
        self.assertEqual(result['tokenIds'], reference[0, len(branch['inputIds']):].tolist())

    def test_prefix_patch_on_each_pass_and_final_block_structural_control(self):
        model = self.model; branch = self.branch(); prefix = [1, 3, 4]
        ids = torch.tensor([branch['inputIds']]); inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        basis = torch.eye(32)[:, :2]
        for layer in (0, 1):
            base = decode_branch(model, self.tokenizer, branch, prefix, layer)
            _, capture = forward_at_shared_prefix(model, inputs, prefix, layer)
            donor = capture['before'].clone(); donor[:2] += 1.
            patched = decode_branch(model, self.tokenizer, branch, prefix, layer, donor=donor, basis=basis)
            manual, _ = forward_at_shared_prefix(model, inputs, prefix, layer, donor=donor, basis=basis)
            self.assertEqual(patched['tokenIds'][0], int(manual.argmax()))
            self.assertTrue(all(p['displacementNorm'] > 0 and p['untouchedTokensEqual'] for p in patched['passes']))
            self.assertEqual(len(patched['passes']), len(patched['tokenIds']))
            if layer == 1:
                self.assertEqual(base['tokenIds'], patched['tokenIds'])
                self.assertEqual(base['candidateLogits'], patched['candidateLogits'])
        self.assertTrue(all(len(layer._forward_hooks) == 0 for layer in model.model.layers))

    def test_budget_and_prefix_corruption_fail_before_decoding(self):
        with self.assertRaises(ValueError): decode_branch(self.model, self.tokenizer, self.branch(), [1, 3, 5], 0)
        with self.assertRaises(ValueError): decode_branch(self.model, self.tokenizer, self.branch(), [1, 3, 4], 0, max_input_tokens=5)
        with self.assertRaises(ValueError): decode_branch(self.model, self.tokenizer, self.branch(), [1, 3, 4], 0, max_new_tokens=1)


if __name__ == '__main__': unittest.main()
