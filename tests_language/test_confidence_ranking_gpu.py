import copy
import unittest

import torch
from torch import nn

from research.action_binding_gpu import train_step
from research.answer_confidence_data import input_messages
from research.answer_confidence_gpu import encode_confidence_training
from research.confidence_ranking_gpu import auxiliary_loss, example_terms, train_paired_step
from research.native_localization_gpu import install_adapters, adapter_state, load_adapter
from tests_language import test_cross_model_gpu as fixture


class ConfidenceRankingGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(240)
        fixture.CrossModelBackendTests.setUpClass()
        cls.template = copy.deepcopy(fixture.CrossModelBackendTests.model)
        cls.tokenizer = fixture.CrossModelBackendTests.tokenizer
        cls.tokenizer.add_tokens(['0', '1'])
        cls.codes = [cls.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in (0, 1)]

    def setup_model(self):
        model = copy.deepcopy(self.template)
        torch.manual_seed(241)
        modules = install_adapters(model, rank=2)
        # Nonzero B tests gradients through BOTH adapter factors.
        with torch.no_grad():
            for m in modules.values(): m.b.normal_(std=.01)
        params = [p for p in model.parameters() if p.requires_grad]
        encoded = [encode_confidence_training(self.tokenizer,
            dict(messages=input_messages('hello '*n, 'hello'), target=str(y)), model.device)
            for n, y in ((1, 1), (2, 0), (3, 1), (4, 1), (5, 0), (6, 1), (7, 0), (8, 0))]
        pairs = list(zip(encoded[::2], encoded[1::2]))
        return model, modules, params, encoded, pairs

    def test_causal_position_excludes_target_and_matches_individual_ce(self):
        model, _, _, encoded, _ = self.setup_model()
        item = encoded[0]; other = copy.deepcopy(item[0])
        other['input_ids'][0, -1] = self.codes[0]
        ce, score = example_terms(model, item, self.tokenizer.eos_token_id, self.codes)
        _, second = example_terms(model, (other, self.codes[0]), self.tokenizer.eos_token_id, self.codes)
        self.assertTrue(torch.equal(score, second))
        full = model(**item[0], use_cache=False).logits[0].float()
        reference = nn.functional.cross_entropy(full[-2:], torch.tensor([item[1], self.tokenizer.eos_token_id]))
        self.assertAlmostEqual(float(ce.detach()), float(reference.detach()), delta=1e-6)

    def test_auxiliary_gradient_direction_symmetry_and_exact_neutral_expectation(self):
        a = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(-.2, dtype=torch.float64, requires_grad=True)
        loss = auxiliary_loss(a, b, 1, 0, 'rank'); ga, gb = torch.autograd.grad(loss, (a, b))
        self.assertLess(float(ga), 0); self.assertGreater(float(gb), 0)
        self.assertEqual(float(loss.detach()), float(auxiliary_loss(b, a, 0, 1, 'rank').detach()))
        neutral = auxiliary_loss(a, b, 1, 0, 'neutral')
        reference = .5*(auxiliary_loss(a, b, 1, 0, 'rank')+auxiliary_loss(a, b, 0, 1, 'rank'))
        self.assertEqual(float(neutral.detach()), float(reference.detach()))
        for arm in ('ce', 'rank', 'neutral'):
            self.assertEqual(float(auxiliary_loss(a, b, 1, 1, arm).detach()), 0.)
        self.assertTrue(torch.autograd.gradcheck(lambda x, y: auxiliary_loss(x, y, 1, 0, 'rank'), (a, b)))

    def test_pairwise_backprop_matches_all_graphs_and_only_changes_adapters(self):
        for arm in ('ce', 'rank', 'neutral'):
            model, modules, params, encoded, pairs = self.setup_model()
            initial = adapter_state(modules)
            frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
            # Independent all-at-once objective; no pair helper in reference.
            logits = [model(**x, use_cache=False, logits_to_keep=2).logits[0].float() for x, _ in encoded]
            ce = torch.stack([nn.functional.cross_entropy(z, torch.tensor([code, self.tokenizer.eos_token_id]))
                              for z, (_, code) in zip(logits, encoded)]).mean()
            extra = ce*0
            for index in range(0, 8, 2):
                yi, yj = (int(encoded[k][1] == self.codes[1]) for k in (index, index+1))
                if yi != yj and arm != 'ce':
                    d = ((logits[index][0, self.codes[1]]-logits[index][0, self.codes[0]])-
                         (logits[index+1][0, self.codes[1]]-logits[index+1][0, self.codes[0]]))
                    target = .5 if arm == 'neutral' else float(yi > yj)
                    extra = extra+nn.functional.binary_cross_entropy_with_logits(d, d.new_tensor(target))/4
            objective = ce+extra; objective.backward()
            reference_grads = [p.grad.detach().clone() for p in params]
            norm = nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            clipped = [p.grad.detach().clone() for p in params]
            optimizer = torch.optim.SGD(params, lr=.01)
            before_rng = torch.get_rng_state().clone()
            result = train_paired_step(model, optimizer, params, pairs, self.tokenizer.eos_token_id, self.codes, arm)
            self.assertAlmostEqual(result['objective'], float(objective.detach()), delta=2e-6)
            self.assertAlmostEqual(result['gradientNorm'], float(norm), delta=2e-6)
            for got, wanted in zip(params, clipped):
                torch.testing.assert_close(got.grad, wanted, atol=1e-7, rtol=1e-5)
            self.assertTrue(all(torch.isfinite(g).all() for g in reference_grads))
            self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
            self.assertTrue(all(torch.equal(p, frozen[n]) for n, p in model.named_parameters() if not p.requires_grad))
            self.assertTrue(any(not torch.equal(v, initial[k]) for k, v in adapter_state(modules).items()))
            if arm == 'ce':
                paired_final = adapter_state(modules)
                load_adapter(modules, initial)
                train_step(model, torch.optim.SGD(params, lr=.01), params, encoded, self.tokenizer.eos_token_id)
                for k, value in adapter_state(modules).items():
                    torch.testing.assert_close(value, paired_final[k], atol=1e-7, rtol=1e-5)

    def test_wrong_trainable_set_and_training_mode_rejected(self):
        model, _, params, encoded, pairs = self.setup_model()
        optimizer = torch.optim.SGD(params, lr=.01)
        with self.assertRaises(ValueError):
            train_paired_step(model, optimizer, params[:-1], pairs, 2, self.codes, 'rank')
        model.train()
        with self.assertRaises(ValueError): example_terms(model, encoded[0], 2, self.codes)


if __name__ == '__main__':
    unittest.main()
