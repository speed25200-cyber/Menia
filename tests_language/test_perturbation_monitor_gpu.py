"""Actual tiny random Qwen forwards; all local and distinct from the A100 pilot."""
import unittest

import torch

from research.perturbation_monitor_gpu import rotate_vector,sample_with_intervention
from research.cross_model_prediction import SETTINGS
from tests_language import test_activation_monitor_gpu as fixtures


class PerturbationHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.ActivationHookTests.setUpClass()
        cls.model,cls.tokenizer=fixtures.ActivationHookTests.model,fixtures.ActivationHookTests.tokenizer

    def task(self,condition):
        return dict(question='hello hello hello hello',seed=41,noiseSeed=192,condition=condition)

    def run_case(self,condition):
        states=[]
        def before_head(module,inputs):
            self.assertEqual(len(states),1)
        h=self.model.lm_head.register_forward_pre_hook(before_head)
        try:
            out=sample_with_intervention(self.model,self.tokenizer,self.task(condition),states.append,
                                         settings=dict(SETTINGS,max_new_tokens=8))
        finally:
            h.remove()
        return out,states[0]

    def test_sham_identity_rotation_before_logits_and_model_restoration(self):
        weights={k:v.clone() for k,v in self.model.state_dict().items()}
        baseline,b=self.run_case('baseline')
        sham,s=self.run_case('sham')
        self.assertEqual(b,s)
        self.assertEqual(baseline[0],sham[0])
        for name in ('rotateHalf','rotateOne'):
            output,state=self.run_case(name)
            self.assertEqual(state['input'],b['input'])
            self.assertNotEqual(state['middle'],b['middle'])
            self.assertNotEqual(state['final'],b['final'])
            self.assertEqual(output[1]['intervention']['applications'],1)
            self.assertLess(output[1]['intervention']['normRelativeError'],.01)
        after,a=self.run_case('baseline')
        self.assertEqual((after,a),(baseline,b))
        for k,v in self.model.state_dict().items():
            self.assertTrue(torch.equal(v,weights[k]))

    def test_rotation_preserves_norm_uses_private_rng_and_has_expected_angle(self):
        for dtype in (torch.float32,torch.bfloat16):
            v=torch.arange(1,129,dtype=dtype)
            rng=torch.random.get_rng_state().clone()
            for strength in (.5,1.):
                out,error=rotate_vector(v,strength,56)
                self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
                self.assertLess(error,.01)
                cosine=float(torch.nn.functional.cosine_similarity(v.float(),out.float(),dim=0))
                self.assertAlmostEqual(cosine,1/(1+strength**2)**.5,places=3)

    def test_capture_failure_removes_all_hooks_and_restores_following_call(self):
        baseline,b=self.run_case('baseline')
        counts=[len(m._forward_hooks) for m in self.model.modules()]
        def fail(state):
            raise RuntimeError('Journal unavailable')
        with self.assertRaisesRegex(RuntimeError,'Journal unavailable'):
            sample_with_intervention(self.model,self.tokenizer,self.task('rotateOne'),fail)
        self.assertEqual(counts,[len(m._forward_hooks) for m in self.model.modules()])
        after,a=self.run_case('baseline')
        self.assertEqual((after,a),(baseline,b))


if __name__=='__main__':
    unittest.main()
