import copy
from dataclasses import replace
import unittest

import torch

from research.confidence_generation_continuity import capture_generated_prefix
from research.confidence_generation_numerics import replay_generated_tokens, compare_caches, compare_branches
from research.confidence_cached_action_decode import decode_cached_branch
from research.cross_model_prediction import SETTINGS
from tests_language.test_confidence_generation_continuity import controlled_trajectory
from tests_language import test_cross_model_gpu as fixture


class GenerationNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(260920)
        fixture.CrossModelBackendTests.setUpClass()
        cls.template = copy.deepcopy(fixture.CrossModelBackendTests.model)
        cls.tokenizer = fixture.CrossModelBackendTests.tokenizer

    def setUp(self): self.model = copy.deepcopy(self.template)

    def capture(self, layer=None):
        return capture_generated_prefix(self.model, self.tokenizer, [dict(role='user',content='hello')], 17,
            model_state_id='random-numerical-control', settings=dict(SETTINGS,max_new_tokens=3),
            max_context_tokens=64, generation_context=controlled_trajectory(self.model,layer=layer))[-1]

    def replay(self, snapshot, schedule='generation'):
        return replay_generated_tokens(self.model,snapshot,2,model_state_id='random-numerical-control',schedule=schedule)

    def decode(self, snapshot):
        branch = dict(inputIds=list(snapshot.prefix)+[5,4],candidateTokenIds=[3,4],negative='verify',positive='keep')
        return decode_cached_branch(self.model,self.tokenizer,branch,snapshot,model_state_id='random-numerical-control')

    def test_matching_schedule_is_exact_and_full_prefill_is_measured_separately(self):
        original = self.capture(); scheduled = self.replay(original); full = self.replay(original,'full')
        self.assertEqual(scheduled.capture['forwardInputLengths'],[2,1,1,1])
        self.assertEqual(full.capture['forwardInputLengths'],[5])
        self.assertEqual(scheduled.prefix,original.prefix); self.assertEqual(full.prefix,original.prefix)
        self.assertTrue(compare_caches(original,scheduled)['exact'])
        self.assertLess(compare_caches(original,full)['maximumAbsoluteDifference'],1e-6)
        a = self.decode(original); b = self.decode(scheduled)
        self.assertEqual(a,b); self.assertEqual(compare_branches(a,b)['maximumCandidateLogitDifference'],0.)

    def test_replay_erases_a_real_historical_perturbation_even_with_same_raw_tokens(self):
        clean = self.capture(); perturbed = self.capture(layer=0); replay = self.replay(perturbed)
        self.assertEqual(clean.prefix,perturbed.prefix)
        self.assertFalse(compare_caches(clean,perturbed)['exact'])
        self.assertGreater(compare_caches(clean,perturbed)['maximumAbsoluteDifference'],1e-5)
        self.assertTrue(compare_caches(clean,replay)['exact'])
        self.assertGreater(compare_branches(self.decode(clean),self.decode(perturbed))['maximumCandidateLogitDifference'],1e-5)

    def test_wrong_owner_parameters_cache_and_token_history_are_rejected(self):
        snapshot = self.capture()
        with self.assertRaises(ValueError): self.replay(snapshot,'unknown')
        with self.assertRaises(ValueError):
            replay_generated_tokens(copy.deepcopy(self.model),snapshot,2,model_state_id='random-numerical-control',schedule='full')
        with self.assertRaises(ValueError): compare_caches(snapshot,replace(snapshot,prefix=(0,)+snapshot.prefix[1:]))
        with torch.no_grad(): next(self.model.parameters()).add_(.01)
        with self.assertRaises(ValueError): self.replay(snapshot)
        snapshot = self.capture()
        snapshot.kv[0][0].add_(.1)
        with self.assertRaises(ValueError): self.replay(snapshot)


if __name__ == '__main__': unittest.main()
