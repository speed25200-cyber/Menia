import copy
import unittest

import torch

from research.confidence_cached_action_decode import prefill_prefix, decode_cached_branch, _parameters
from research.retained_state_reader import RetainedStateReader
from tests_language import test_cross_model_gpu as fixture


class RetainedReaderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026092071)
        fixture.CrossModelBackendTests.setUpClass()
        self.producer = fixture.CrossModelBackendTests.model.eval().requires_grad_(False)
        self.tokenizer = fixture.CrossModelBackendTests.tokenizer
        self.saved = prefill_prefix(self.producer,[1,3,4],0,model_state_id='fixture')
        self.branch = dict(inputIds=[1,3,4,5,3],candidateTokenIds=[3,4],negative='no',positive='yes')
        self.reader = RetainedStateReader(self.producer,producer_state_id='fixture',rank=4,seed=97)

    def baseline(self):
        return decode_cached_branch(self.producer,self.tokenizer,self.branch,self.saved,model_state_id='fixture')

    def test_zero_adapter_matches_existing_native_path_and_leaves_rng(self):
        rng = torch.get_rng_state().clone()
        other = RetainedStateReader(self.producer,producer_state_id='fixture',rank=4,seed=97)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(other.adapter_hash(),self.reader.adapter_hash())
        baseline = self.baseline()
        result = self.reader.decode(self.tokenizer,self.saved,self.branch)
        actual = dict(result['decoded']); actual['modelStateId'] = baseline['modelStateId']
        for field in ('candidateLogits','conditionalPositive','candidateMass'):
            torch.testing.assert_close(torch.tensor(actual.pop(field)),torch.tensor(baseline.pop(field)),rtol=0,atol=1e-6)
        self.assertEqual(actual,baseline)
        self.assertEqual(result['producerStateId'],'fixture')
        state=self.reader.adapter_state()
        self.reader.load_adapter(state)
        self.assertEqual(self.reader.decode(self.tokenizer,self.saved,self.branch),result)
        broken=dict(state); broken.pop(next(iter(broken)))
        with self.assertRaises(ValueError): self.reader.load_adapter(broken)

    def test_updates_change_reader_but_not_producer_or_retained_state(self):
        baseline = self.baseline(); original_versions = _parameters(self.producer)
        initial = self.reader.adapter_hash()
        optimizer = torch.optim.AdamW(self.reader.adapter_parameters,lr=.02,weight_decay=0)
        before = float(self.reader.loss(self.saved,self.branch,4,2).detach())
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            loss = self.reader.loss(self.saved,self.branch,4,2); loss.backward()
            self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in self.reader.adapter_parameters))
            self.assertTrue(all(p.grad is None for p in self.producer.parameters()))
            optimizer.step()
        self.assertLess(float(self.reader.loss(self.saved,self.branch,4,2).detach()),before)
        self.assertNotEqual(self.reader.adapter_hash(),initial)
        self.assertEqual(_parameters(self.producer),original_versions)
        self.assertEqual(self.baseline(),baseline)
        self.assertNotEqual(self.reader.decode(self.tokenizer,self.saved,self.branch)['decoded']['candidateLogits'],baseline['candidateLogits'])

    def test_first_prediction_does_not_see_teacher_forced_target(self):
        a = self.reader.teacher_forced_logits(self.saved,self.branch,3)
        b = self.reader.teacher_forced_logits(self.saved,self.branch,4)
        self.assertTrue(torch.equal(a[0],b[0]))
        self.assertFalse(torch.equal(a[1],b[1]))
        self.assertTrue(a.requires_grad)

    def test_gradients_work_with_inference_mode_prefix_tensors(self):
        with torch.inference_mode():
            saved = prefill_prefix(self.producer,[1,3,4],0,model_state_id='fixture')
        self.assertTrue(saved.kv[0][0].is_inference())
        self.reader.loss(saved,self.branch,4,2).backward()
        self.assertTrue(any(p.grad is not None for p in self.reader.adapter_parameters))
        self.reader.guard(saved)

    def test_wrong_owner_prefix_mode_and_backbone_changes_are_rejected(self):
        other = copy.deepcopy(self.producer)
        wrong = prefill_prefix(other,[1,3,4],0,model_state_id='fixture')
        with self.assertRaises(ValueError): self.reader.guard(wrong)
        branch = dict(self.branch,inputIds=[1,4,4,5,3])
        with self.assertRaises(ValueError): self.reader.decode(self.tokenizer,self.saved,branch)
        with self.assertRaises(ValueError): self.reader.decode(self.tokenizer,self.saved,self.branch,max_input_tokens=5)
        module = next(iter(self.reader.modules.values())); module.enabled = False
        with self.assertRaises(ValueError): self.reader.guard(self.saved)
        module.enabled = True
        with torch.no_grad(): module.base.weight.add_(.01)
        with self.assertRaises(ValueError): self.reader.guard(self.saved)

    def test_changed_producer_and_corrupted_cache_are_rejected(self):
        with torch.no_grad(): self.saved.kv[0][0].add_(.1)
        with self.assertRaises(ValueError): self.reader.guard(self.saved)
        saved = prefill_prefix(self.producer,[1,3,4],0,model_state_id='fixture')
        with torch.no_grad(): next(self.producer.parameters()).add_(.01)
        with self.assertRaises(ValueError): self.reader.guard(saved)


if __name__ == '__main__': unittest.main()
