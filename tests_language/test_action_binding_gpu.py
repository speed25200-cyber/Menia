import unittest
import torch
from research.action_binding_gpu import encode_training,supervised_loss,train_step
from research.native_localization_gpu import install_adapters,adapter_state,load_adapter
from tests_language import test_cross_model_gpu as fixture


class BindingTrainingTest(unittest.TestCase):
    def test_causal_two_token_loss_and_only_adapters_update(self):
        fixture.CrossModelBackendTests.setUpClass();model=fixture.CrossModelBackendTests.model;tok=fixture.CrossModelBackendTests.tokenizer
        tok.add_tokens(['1','2','A','B']);modules=install_adapters(model,rank=2)
        initial=adapter_state(modules);base={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        example=dict(messages=[dict(role='user',content='hello')],target='1');inputs,code=encode_training(tok,example,model.device)
        eos=tok.eos_token_id;loss=supervised_loss(model,inputs,code,eos)
        full=model(**inputs,use_cache=False).logits[0].float()
        independent=(torch.nn.functional.cross_entropy(full[-2:-1],torch.tensor([code]))+torch.nn.functional.cross_entropy(full[-1:],torch.tensor([eos])))/2
        self.assertAlmostEqual(float(loss.detach()),float(independent.detach()),places=6)
        changed={k:v.clone() for k,v in inputs.items()};changed['input_ids'][0,-1]=tok.encode('2',add_special_tokens=False)[0]
        self.assertTrue(torch.equal(model(**inputs,use_cache=False).logits[0,-2],model(**changed,use_cache=False).logits[0,-2]))
        parameters=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=0.,foreach=False)
        train_step(model,optimizer,parameters,[(inputs,code)]*8,eos)
        self.assertTrue(any(not torch.equal(v,initial[k]) for k,v in adapter_state(modules).items()))
        self.assertTrue(all(torch.equal(p,base[n]) for n,p in model.named_parameters() if not p.requires_grad))
        load_adapter(modules,initial)
        self.assertTrue(all(torch.equal(v,initial[k]) for k,v in adapter_state(modules).items()))


if __name__=='__main__':unittest.main()
