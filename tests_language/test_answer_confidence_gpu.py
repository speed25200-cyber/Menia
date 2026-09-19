import unittest
import torch

from research.answer_confidence_gpu import assess_answer,encode_confidence_training
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

    def test_confidence_supervision_excludes_answer_targets_and_updates_only_adapters(self):
        from research.action_binding_gpu import supervised_loss,train_step
        from research.native_localization_gpu import install_adapters,adapter_state
        fixture.CrossModelBackendTests.setUpClass()
        model,tok=fixture.CrossModelBackendTests.model,fixture.CrossModelBackendTests.tokenizer
        tok.add_tokens(['0','1']);modules=install_adapters(model,rank=2)
        old=adapter_state(modules);base={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        example=dict(messages=input_messages('hello','hello'),target='0')
        inputs,code=encode_confidence_training(tok,example,model.device)
        loss=supervised_loss(model,inputs,code,tok.eos_token_id)
        full=model(**inputs,use_cache=False).logits[0].float()
        reference=torch.nn.functional.cross_entropy(full[-2:],torch.tensor([code,tok.eos_token_id]))
        self.assertAlmostEqual(float(loss.detach()),float(reference.detach()),delta=1e-6)
        alternate=dict(example,target='1');other,_=encode_confidence_training(tok,alternate,model.device)
        self.assertTrue(torch.equal(inputs['input_ids'][:,:-1],other['input_ids'][:,:-1]))
        self.assertTrue(torch.equal(model(**inputs,use_cache=False).logits[0,-2],model(**other,use_cache=False).logits[0,-2]))
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=0.,foreach=False)
        train_step(model,optimizer,params,[(inputs,code)]*8,tok.eos_token_id)
        self.assertTrue(any(not torch.equal(v,old[k]) for k,v in adapter_state(modules).items()))
        self.assertTrue(all(torch.equal(p,base[n]) for n,p in model.named_parameters() if not p.requires_grad))
        with self.assertRaises(ValueError):encode_confidence_training(tok,example,model.device,max_input_tokens=1)


if __name__=='__main__':unittest.main()
