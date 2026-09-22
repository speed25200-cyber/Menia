import unittest
from unittest.mock import patch

import torch

from research.confidence_prefix_interventions import (
    projected_interchange,forward_at_shared_prefix,final_head_null_control)
from tests_language import test_cross_model_gpu as fixture


class ConfidencePrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.CrossModelBackendTests.setUpClass()
        cls.model=fixture.CrossModelBackendTests.model

    def inputs(self,ids):
        t=torch.tensor([ids],dtype=torch.long)
        return dict(input_ids=t,attention_mask=torch.ones_like(t))

    def test_projected_swap_preserves_complement_and_restores(self):
        r=torch.tensor([1.,2.,3.,4.],dtype=torch.float64)
        d=torch.tensor([-3.,5.,9.,0.],dtype=torch.float64)
        u=torch.tensor([[1.,0.],[0.,0.],[0.,1.],[0.,0.]],dtype=torch.float64)
        changed=projected_interchange(r,d,u)
        self.assertTrue(torch.equal(changed,torch.tensor([-3.,2.,9.,4.],dtype=torch.float64)))
        self.assertTrue(torch.equal(u.T@changed,u.T@d))
        self.assertTrue(torch.equal(projected_interchange(changed,r,u),r))
        self.assertTrue(torch.equal(projected_interchange(r,r,u),r))
        with self.assertRaises(ValueError):projected_interchange(r,d,u*2)
        with self.assertRaises(ValueError):projected_interchange(r,d,torch.eye(4,dtype=torch.float64))
        with self.assertRaises(ValueError):projected_interchange(r,d*float('nan'),u)

    def test_shared_state_is_causal_before_different_future_branches(self):
        # IDs are arbitrary tokens of a random miniature Qwen, not scored tasks.
        prefix=[1,2,3];model=self.model
        _,a=forward_at_shared_prefix(model,self.inputs(prefix+[4,5]),prefix,0)
        _,b=forward_at_shared_prefix(model,self.inputs(prefix+[6,7,8]),prefix,0)
        _,c=forward_at_shared_prefix(model,self.inputs(prefix),prefix,0)
        self.assertTrue(torch.allclose(a['before'],b['before'],atol=1e-7,rtol=1e-6))
        self.assertTrue(torch.allclose(a['before'],c['before'],atol=1e-7,rtol=1e-6))
        self.assertEqual(a['prefixHash'],b['prefixHash']);self.assertEqual(a['position'],2)
        with self.assertRaises(ValueError):forward_at_shared_prefix(model,self.inputs([1,2,4,5]),prefix,0)

    def test_real_prefix_swap_is_local_and_sham_leaves_output_and_weights_identical(self):
        model=self.model;prefix=[1,2,3];inputs=self.inputs(prefix+[4,5])
        weights={k:p.detach().clone() for k,p in model.named_parameters()};rng=torch.get_rng_state().clone()
        logits,recipient=forward_at_shared_prefix(model,inputs,prefix,0)
        _,donor=forward_at_shared_prefix(model,self.inputs([1,6,7]),[1,6,7],0)
        basis=torch.eye(len(recipient['before']))[:,:2]
        sham,control=forward_at_shared_prefix(model,inputs,prefix,0,donor=recipient['before'],basis=basis)
        changed,event=forward_at_shared_prefix(model,inputs,prefix,0,donor=donor['before'],basis=basis)
        self.assertTrue(torch.equal(sham,logits));self.assertEqual(control['displacementNorm'],0.)
        self.assertTrue(event['untouchedTokensEqual']);self.assertGreater(event['displacementNorm'],0.)
        self.assertTrue(torch.equal(event['after'],projected_interchange(recipient['before'],donor['before'],basis)))
        self.assertFalse(torch.equal(changed,logits))
        self.assertTrue(all(torch.equal(v,weights[k]) for k,v in model.named_parameters()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_hooks_removed_on_failure_and_unsupported_inputs_rejected(self):
        model=self.model;inputs=self.inputs([1,2,3,4]);layer=model.model.layers[0]
        count=len(layer._forward_hooks)
        with patch.object(model,'forward',side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):forward_at_shared_prefix(model,inputs,[1,2,3],0)
        self.assertEqual(len(layer._forward_hooks),count)
        padded=self.inputs([1,2,3,4]);padded['attention_mask'][0,0]=0
        with self.assertRaises(ValueError):forward_at_shared_prefix(model,padded,[1,2],0)
        model.train()
        try:
            with self.assertRaises(ValueError):forward_at_shared_prefix(model,inputs,[1,2],0)
        finally:model.eval()
        with self.assertRaises(ValueError):forward_at_shared_prefix(model,inputs,[1,2],-1)
        count=len(model.model.norm._forward_hooks)
        original=model.forward
        calls=[0]
        def fail_second(*args,**kwargs):
            calls[0]+=1
            if calls[0]==2:raise RuntimeError('injected after baseline')
            return original(*args,**kwargs)
        with patch.object(model,'forward',side_effect=fail_second):
            with self.assertRaises(RuntimeError):final_head_null_control(model,inputs,3,4,1.)
        self.assertEqual(len(model.model.norm._forward_hooks),count)

    def test_head_control_changes_confidence_without_any_correctness_labels(self):
        model=self.model;weights={k:p.detach().clone() for k,p in model.named_parameters()};rng=torch.get_rng_state().clone()
        for ids in ([1,2,3,4],[1,5,6,7]):
            for shift in (-1.,0.,1.):
                baseline,changed,event=final_head_null_control(model,self.inputs(ids),3,4,shift)
                self.assertAlmostEqual(event['realizedHeadProjection'],shift,delta=1e-6)
                self.assertAlmostEqual(event['actualLogOddsShift'],shift,delta=1e-6)
                p0=torch.softmax(baseline[[3,4]].double(),dim=0)[1]
                p1=torch.softmax(changed[[3,4]].double(),dim=0)[1]
                if shift>0:self.assertGreater(p1,p0)
                if shift<0:self.assertLess(p1,p0)
                if shift==0:self.assertTrue(torch.equal(changed,baseline))
        self.assertTrue(all(torch.equal(v,weights[k]) for k,v in model.named_parameters()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
