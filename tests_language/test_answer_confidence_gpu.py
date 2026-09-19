import unittest
import torch

from research.answer_confidence_gpu import assess_answer
from research.answer_confidence_data import input_messages
from tests_language import test_cross_model_gpu as fixture


class NativeConfidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.CrossModelBackendTests.setUpClass()
        cls.model=fixture.CrossModelBackendTests.model;cls.tokenizer=fixture.CrossModelBackendTests.tokenizer
        cls.tokenizer.add_tokens(['0','1'])

    def test_native_head_score_matches_full_forward_without_sampling_or_updates(self):
        model,tok=self.model,self.tokenizer
        old={k:p.detach().clone() for k,p in model.named_parameters()};rng=torch.get_rng_state().clone()
        result=assess_answer(model,tok,'hello','hello')
        messages=input_messages('hello','hello')
        ids=tok.encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
        inputs=torch.tensor([ids])
        with torch.no_grad():
            logits=model(input_ids=inputs,attention_mask=torch.ones_like(inputs),use_cache=False).logits[0,-1].double()
        probs=torch.softmax(logits,dim=0);i,j=result['codeTokenIds'];mass=float(probs[i]+probs[j])
        # The float32 head multiplies a different matrix shape when only the
        # last position is requested; compare at float32 precision, not 1e-12.
        self.assertAlmostEqual(result['candidateMass'],mass,delta=1e-7)
        self.assertAlmostEqual(result['conditionalCorrect'],float(probs[j])/mass,delta=1e-7)
        self.assertEqual(result['topTokenId'],int(logits.argmax()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,old[k]) for k,v in model.named_parameters()))
        self.assertEqual(result,assess_answer(model,tok,'hello','hello'))

    def test_overflow_and_training_mode_rejected(self):
        with self.assertRaises(ValueError):assess_answer(self.model,self.tokenizer,'hello','hello',max_input_tokens=1)
        self.model.train()
        try:
            with self.assertRaises(ValueError):assess_answer(self.model,self.tokenizer,'hello','hello')
        finally:self.model.eval()


if __name__=='__main__':unittest.main()
