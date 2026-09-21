"""Numerical/provenance tests on a tiny random model; not Qwen3-4B results."""
import copy
from dataclasses import replace
import unittest

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from research.confidence_cached_action_decode import prefill_prefix, decode_cached_branch, _cache_hash
from research.prospective_cache_interventions import permute_prefix_cache
from research.reader_permutation_numerics import widen_model, widen_snapshot, summarize, MODES, CONDITIONS
from tests_language import test_cross_model_gpu as fixture


class PermutationNumericsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026092073)
        fixture.CrossModelBackendTests.setUpClass()
        self.model = fixture.CrossModelBackendTests.model.to(torch.bfloat16).requires_grad_(False)
        self.tokenizer = fixture.CrossModelBackendTests.tokenizer
        self.source = prefill_prefix(self.model, [1,3,4], 0, model_state_id='original-bf16')
        self.branch = dict(inputIds=[1,3,4,5,3],candidateTokenIds=[3,4],negative='no',positive='yes')

    def test_widening_preserves_values_and_requires_an_explicit_new_state(self):
        before = {n:p.detach().clone() for n,p in self.model.named_parameters()}
        old_hash = self.source.cache_hash
        result = widen_model(self.model,self.source)
        self.assertTrue(result['allValuesPreserved'])
        for n,p in self.model.named_parameters():
            self.assertEqual(p.dtype,torch.float32)
            self.assertTrue(torch.equal(p,before[n].float()))
        with self.assertRaisesRegex(ValueError,'state changed'):
            decode_cached_branch(self.model,self.tokenizer,self.branch,self.source,model_state_id='original-bf16')
        widened = widen_snapshot(self.model,self.source,'widened-fp32')
        self.assertEqual(_cache_hash(self.source.kv),old_hash)
        self.assertFalse(widened.capture['precisionImport']['originalCacheRecomputed'])
        for a,b in zip(self.source.kv,widened.kv):
            for x,y in zip(a,b):
                self.assertTrue(torch.equal(x.float(),y))
                self.assertNotEqual(x.data_ptr(),y.data_ptr())
        decoded = decode_cached_branch(self.model,self.tokenizer,self.branch,widened,model_state_id='widened-fp32')
        self.assertTrue(decoded['snapshotUnchanged'])

    def test_precision_import_rejects_tampering_and_other_owners(self):
        widen_model(self.model,self.source)
        fake = copy.deepcopy(self.model)
        with self.assertRaisesRegex(ValueError,'owned by this model'):
            widen_snapshot(fake,self.source,'foreign')
        with self.assertRaisesRegex(ValueError,'new identity'):
            widen_snapshot(self.model,self.source,'original-bf16')
        corrupted = replace(self.source,cache_hash='0'*64)
        with self.assertRaisesRegex(ValueError,'Unchanged source'):
            widen_snapshot(self.model,corrupted,'corrupted')

    def test_math_attention_joint_control_after_exact_cache_widening(self):
        widen_model(self.model,self.source)
        source = widen_snapshot(self.model,self.source,'fp32')
        args = dict(model_state_id='fp32',layers=[0,1],permutation=[0,2,1])
        joint,_ = permute_prefix_cache(self.model,source,mode='keys_and_values',**args)
        with sdpa_kernel(SDPBackend.MATH):
            a = decode_cached_branch(self.model,self.tokenizer,self.branch,source,model_state_id='fp32')
            b = decode_cached_branch(self.model,self.tokenizer,self.branch,joint,model_state_id='fp32')
        self.assertLess(max(abs(x-y) for x,y in zip(a['candidateLogits'],b['candidateLogits'])),1e-6)
        self.assertEqual(a['tokenIds'],b['tokenIds'])
        self.assertEqual(_cache_hash(source.kv),source.cache_hash)

    def test_changed_parameters_are_not_silently_imported(self):
        with torch.no_grad():
            next(self.model.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError,'must still match'):
            widen_model(self.model,self.source)

    def test_summary_preserves_repeat_failures_and_rejects_missing_or_duplicate_rows(self):
        d = decode_cached_branch(self.model,self.tokenizer,self.branch,self.source,model_state_id='original-bf16')
        records = [dict(mode=m,repeat=r,condition=c,target=t,decoded=copy.deepcopy(d))
                   for m in MODES for r in range(2) for c in CONDITIONS for t in range(8)]
        report = summarize(records)
        self.assertEqual(report['decodes'],192)
        self.assertTrue(all(r['repeatExactlyEqual'] for r in report['contrasts']))
        records[-1]['decoded']['tokenIds'] = [0,0]
        changed = summarize(records)
        self.assertEqual(sum(not r['repeatExactlyEqual'] for r in changed['contrasts']),1)
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            summarize(records[:-1])
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            summarize(records+[records[0]])


if __name__ == '__main__':
    unittest.main()
