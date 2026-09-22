import copy
from contextlib import contextmanager
import unittest

import torch

from research.confidence_generation_continuity import capture_generated_prefix
from research.confidence_cached_action_decode import prefill_prefix,decode_cached_branch
from research.confidence_prefix_interventions import tensor_hash
from research.cross_model_gpu import generate_text
from research.cross_model_prediction import SETTINGS
from tests_language import test_cross_model_gpu as fixture


@contextmanager
def controlled_trajectory(model,tokens=(5,3,2),layer=None,fail=False):
    """Synthetic logit fixture: predetermined emissions, optional historical patch.

    The actual native generator still samples the tokens and maintains its KV.
    These controls are not language-model task performance.
    """
    count = 0; handles = []
    def logits(module,args,output):
        nonlocal count
        if fail and count == 1: raise RuntimeError('Synthetic recording failure')
        value = torch.full_like(output,-100.)
        value[...,tokens[count]] = 100.; count += 1
        return value
    def perturb(module,args,output):
        value = output[0] if isinstance(output,tuple) else output
        changed = value.clone(); changed[:,-1,:2] += .7
        return (changed,*output[1:]) if isinstance(output,tuple) else changed
    handles.append(model.lm_head.register_forward_hook(logits))
    if layer is not None: handles.append(model.model.layers[layer].register_forward_hook(perturb))
    try: yield
    finally:
        for handle in handles: handle.remove()


class GenerationContinuityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(260920)
        fixture.CrossModelBackendTests.setUpClass()
        cls.template = copy.deepcopy(fixture.CrossModelBackendTests.model)
        cls.tokenizer = fixture.CrossModelBackendTests.tokenizer
        cls.messages = [dict(role='user',content='hello')]
        cls.settings = dict(SETTINGS,max_new_tokens=3)

    def setUp(self): self.model = copy.deepcopy(self.template)

    def capture(self,**kwargs):
        return capture_generated_prefix(self.model,self.tokenizer,self.messages,17,
            model_state_id='random-controlled-trajectory',settings=self.settings,max_context_tokens=64,**kwargs)

    def branch(self,saved):
        return dict(inputIds=list(saved.prefix)+[5,4],candidateTokenIds=[3,4],negative='verify',positive='accept')

    def decode(self,saved):
        return decode_cached_branch(self.model,self.tokenizer,self.branch(saved),saved,model_state_id='random-controlled-trajectory')

    def test_actual_generator_ids_metrics_rng_and_last_token_consumption(self):
        weights = {k:tensor_hash(v) for k,v in self.model.state_dict().items()}
        with controlled_trajectory(self.model):
            expected,metrics = generate_text(self.model,self.tokenizer,self.messages,17,settings=self.settings)
        rng = torch.get_rng_state().clone()
        text,actual,trace,saved = self.capture(generation_context=controlled_trajectory(self.model))
        self.assertEqual((text,actual),(expected,metrics)); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(trace['generatedTokenIds'],[5,3,2])
        self.assertEqual(trace['generationForwardInputLengths'],[2,1,1])
        self.assertEqual(trace['cacheTokensBeforeFinalToken'],4)
        self.assertEqual(trace['cacheTokensAfterFinalToken'],5)
        self.assertEqual(list(saved.prefix),trace['promptTokenIds']+[5,3,2])
        self.assertFalse(trace['textWasRetokenized'])
        self.assertEqual(weights,{k:tensor_hash(v) for k,v in self.model.state_dict().items()})

    def test_unperturbed_original_trajectory_matches_full_token_replay(self):
        _,_,_,saved = self.capture(generation_context=controlled_trajectory(self.model))
        replay = prefill_prefix(self.model,saved.prefix,0,model_state_id='random-controlled-trajectory')
        for a,b in zip(saved.kv,replay.kv):
            for x,y in zip(a,b): torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
        a = self.decode(saved); b = self.decode(replay)
        self.assertEqual(a['tokenIds'],b['tokenIds'])
        for x,y in zip(a['candidateLogits'],b['candidateLogits']): self.assertAlmostEqual(x,y,delta=1e-6)

    def test_hidden_history_changes_forks_with_identical_emitted_tokens_and_restores(self):
        _,_,_,baseline = self.capture(generation_context=controlled_trajectory(self.model))
        _,_,_,changed = self.capture(generation_context=controlled_trajectory(self.model,layer=0))
        self.assertEqual(baseline.prefix,changed.prefix)
        self.assertNotEqual(baseline.cache_hash,changed.cache_hash)
        replay = prefill_prefix(self.model,changed.prefix,0,model_state_id='random-controlled-trajectory')
        a = self.decode(baseline); b = self.decode(changed); c = self.decode(replay)
        self.assertGreater(max(abs(x-y) for x,y in zip(a['candidateLogits'],b['candidateLogits'])),1e-5)
        for x,y in zip(a['candidateLogits'],c['candidateLogits']): self.assertAlmostEqual(x,y,delta=1e-6)
        self.assertEqual(self.decode(baseline),a)
        self.assertTrue(all(not block._forward_hooks for block in self.model.model.layers))
        self.assertFalse(self.model._forward_hooks); self.assertFalse(self.model.lm_head._forward_hooks)

    def test_last_block_history_does_not_enter_future_cache(self):
        _,_,_,baseline = self.capture(generation_context=controlled_trajectory(self.model))
        _,_,_,changed = self.capture(generation_context=controlled_trajectory(self.model,layer=1))
        self.assertEqual(baseline.prefix,changed.prefix)
        self.assertEqual(baseline.cache_hash,changed.cache_hash)
        self.assertEqual(self.decode(baseline),self.decode(changed))

    def test_second_configured_stop_token_is_preserved_without_normalization(self):
        self.model.generation_config.eos_token_id = [2,1]
        _,_,trace,saved = self.capture(generation_context=controlled_trajectory(self.model,tokens=(5,3,1)))
        self.assertEqual(trace['generatedTokenIds'],[5,3,1])
        self.assertEqual(trace['eosTokenIds'],[2,1]); self.assertEqual(trace['finalTokenId'],1)
        self.assertEqual(saved.prefix[-1],1)
        self.assertEqual(trace['cacheTokensAfterFinalToken'],5)

    def test_failures_unfinished_answers_and_parameter_changes_are_rejected_and_hooks_removed(self):
        with self.assertRaises(RuntimeError): self.capture(generation_context=controlled_trajectory(self.model,fail=True))
        self.assertFalse(self.model._forward_hooks); self.assertFalse(self.model.lm_head._forward_hooks)
        with self.assertRaises(ValueError): self.capture(generation_context=controlled_trajectory(self.model,tokens=(5,3,4)))
        self.assertFalse(self.model._forward_hooks); self.assertFalse(self.model.lm_head._forward_hooks)
        with self.assertRaises(ValueError):
            capture_generated_prefix(self.model,self.tokenizer,self.messages,17,model_state_id='tiny',settings=self.settings,max_context_tokens=4)
        self.assertFalse(self.model._forward_hooks)
        @contextmanager
        def mutation():
            with torch.no_grad(): next(self.model.parameters()).add_(.01)
            with controlled_trajectory(self.model): yield
        with self.assertRaises(ValueError): self.capture(generation_context=mutation())
        self.assertFalse(self.model._forward_hooks); self.assertFalse(self.model.lm_head._forward_hooks)


if __name__ == '__main__': unittest.main()
