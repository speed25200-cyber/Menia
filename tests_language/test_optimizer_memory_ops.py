import copy
import math
import unittest

import torch

from research.optimizer_memory_ops import fingerprint,fork_optimizer,moment_summary,zero_gradient_step


class OptimizerMemoryTests(unittest.TestCase):
    def test_forks_do_not_alias_and_first_moment_is_only_changed_field(self):
        p=torch.nn.Parameter(torch.tensor([.7,-.3],dtype=torch.float64))
        opt=torch.optim.AdamW([p],lr=.01,weight_decay=0.)
        for i in range(3):
            opt.zero_grad();p.grad=torch.tensor([.4,-.2],dtype=torch.float64);opt.step()
        state=copy.deepcopy(opt.state_dict()); original=fingerprint(state); weights=p.detach().clone()
        q=torch.nn.Parameter(weights.clone()); fork,info=fork_optimizer([q],state,reset_first=True)
        self.assertEqual(info['after']['firstMomentNorm'],0.)
        for key in ('secondMomentNorm','minStep','maxStep','states'):
            self.assertEqual(info['before'][key],info['after'][key])
        zero_gradient_step(fork,[q]);self.assertTrue(torch.equal(q,weights))
        self.assertEqual(fingerprint(state),original)
        self.assertEqual(fingerprint(opt.state_dict()),original)

    def test_zero_gradient_drift_matches_adam_formula_but_none_skips(self):
        p=torch.nn.Parameter(torch.tensor([.5],dtype=torch.float64))
        opt=torch.optim.AdamW([p],lr=.03,betas=(.8,.95),eps=1e-8,weight_decay=0.)
        for i in range(4):
            p.grad=torch.tensor([.2+i*.1],dtype=torch.float64);opt.step()
        state=copy.deepcopy(opt.state_dict()); start=p.item()
        q=torch.nn.Parameter(p.detach().clone()); continued,info=fork_optimizer([q],state)
        m=state['state'][0]['exp_avg'].item();v=state['state'][0]['exp_avg_sq'].item();t=4; expected=start
        for i in range(5):
            t+=1;m*=.8;v*=.95
            expected-=.03*(m/(1-.8**t))/(math.sqrt(v/(1-.95**t))+1e-8)
            zero_gradient_step(continued,[q])
            self.assertAlmostEqual(q.item(),expected,places=12)
        self.assertNotEqual(q.item(),start)
        none=torch.nn.Parameter(p.detach().clone()); skipped,_=fork_optimizer([none],state)
        skipped.zero_grad(set_to_none=True);skipped.step()
        self.assertEqual(none.item(),start);self.assertEqual(moment_summary(skipped)['maxStep'],4)


if __name__=='__main__':unittest.main()
