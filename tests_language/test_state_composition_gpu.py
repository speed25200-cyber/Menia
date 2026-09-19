import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from research import state_composition as s
from research import state_composition_gpu as runtime
from research.native_localization_gpu import install_adapters,adapter_state,file_hash
from tests_language.test_learning_diagnostic_gpu import model,tokenizer


def tok():
    t=tokenizer();t.add_tokens(['3']);return t


class CompositionRuntimeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_weighted_update_matches_independent_objective_and_freezes_base(self):
        with patch.dict(s.CONFIG,dict(layer=0)):
            first=model();install_adapters(first,rank=2);second=copy.deepcopy(first)
            params=[p for p in first.parameters() if p.requires_grad];other=[p for p in second.parameters() if p.requires_grad]
            base=first.model.embed_tokens.weight.detach().clone();opt=torch.optim.AdamW(params,lr=.0002,weight_decay=0.);ref=torch.optim.AdamW(other,lr=.0002,weight_decay=0.)
            b=dict(noiseSeed=51,shuffledTargets=[2,0,1],marker=1);g=dict(epoch=1,monitorFamily='hidden',publicPosition=2)
            def inputs(b,f,t,p,form,mapping):
                return dict(input_ids=torch.tensor([[1,2,3,4,5,6,7,8,10+mapping,12]])),[[1,2],[4,5]],'fixture'
            runtime.balanced_update(first,opt,params,b,'shuffled',g,[1,2,3,6],inputs)
            ref.zero_grad();loss=0.
            for mapping in (0,1):
                for pos,(truth,weight) in enumerate(((1,.0625),(0,.125),(1,.0625))):
                    x,spans,_=inputs(b,'hidden','monitor',pos,'trained',mapping)
                    logits,_=runtime.evaluate_forward(second,x,spans,b,'hidden',pos)
                    label=truth if mapping==0 else 1-truth
                    loss+=torch.nn.functional.cross_entropy(logits[None],torch.tensor([label+1]))*weight
            for truth,task in ((0,'marker_first'),(1,'marker_second')):
                for mapping in (0,1):
                    x,spans,_=inputs(b,'hidden',task,2,'trained',mapping);logits,_=runtime.evaluate_forward(second,x,spans,b,'hidden',2)
                    label=truth if mapping==0 else 1-truth
                    loss+=torch.nn.functional.cross_entropy(logits[None],torch.tensor([label+1]))*.125
            loss.backward();torch.nn.utils.clip_grad_norm_(other,1.);ref.step()
            self.assertTrue(all(torch.allclose(a,b,atol=2e-7,rtol=1e-5) for a,b in zip(params,other)))
            self.assertTrue(torch.equal(base,first.model.embed_tokens.weight));self.assertIsNone(first.model.embed_tokens.weight.grad)

    def test_real_tiny_qwen_warm_restart_eval_resume_and_parent_immutability(self):
        with tempfile.TemporaryDirectory() as d,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            stack.enter_context(patch.dict(s.CONFIG,dict(trainBlocks=2,testBlocks=2,epochs=2,replications=1,resamples=10,rank=2)))
            parent=Path(d)/'parent.jsonl';parent.write_bytes(b'synthetic composition parent')
            net=model(36,8);mods=install_adapters(net,rank=2)
            with torch.no_grad():
                for m in mods.values():m.b.fill_(.002)
            ck=parent.with_suffix('.r0-strong.safetensors');save_file(adapter_state(mods),str(ck));retained=ck.read_bytes()
            stack.enter_context(patch.object(s.previous,'PARENT_JOURNAL_SHA256',file_hash(parent)))
            stack.enter_context(patch.object(s.previous,'checkpoints',return_value={'r0-strong':file_hash(ck)}))
            stack.enter_context(patch('research.cross_model_gpu.environment',return_value={'gpu':'CPU synthetic fixture'}))
            stack.enter_context(patch('transformers.AutoTokenizer.from_pretrained',return_value=tok()))
            loader=stack.enter_context(patch('transformers.AutoModelForCausalLM.from_pretrained',side_effect=lambda *a,**k:model(36,8).to(torch.bfloat16)))
            stack.enter_context(patch('torch.cuda.synchronize'))
            p=Path(d)/'trial.jsonl';update=runtime.balanced_update;calls=0
            def train_stop(*a,**k):
                nonlocal calls
                calls+=1
                if calls==6:raise RuntimeError('Interrupted second arm')
                return update(*a,**k)
            with patch.object(runtime,'balanced_update',side_effect=train_stop),self.assertRaises(RuntimeError):runtime.run(p,parent)
            first=p.with_suffix('.r0-composed.safetensors').read_bytes()
            summarize=runtime.summarize_logits;calls=0
            def eval_stop(*a,**k):
                nonlocal calls
                calls+=1
                if calls==3:raise RuntimeError('Interrupted evaluation')
                return summarize(*a,**k)
            with patch.object(runtime,'summarize_logits',side_effect=eval_stop),self.assertRaises(RuntimeError):runtime.run(p,parent,resume=True)
            runtime.run(p,parent,resume=True);r=s.analyze(p)
            self.assertTrue(r['complete']);self.assertEqual(r['recorded'],768)
            self.assertEqual(r['trainingRestarts'],1);self.assertEqual(r['interruptedRequests'],1)
            self.assertEqual(ck.read_bytes(),retained);self.assertEqual(p.with_suffix('.r0-composed.safetensors').read_bytes(),first)
            clean=Path(d)/'clean.jsonl';runtime.run(clean,parent)
            for key in r['trained']:self.assertEqual(p.with_suffix('.'+key+'.safetensors').read_bytes(),clean.with_suffix('.'+key+'.safetensors').read_bytes())
            _,a,_,_,_=s.read_journal(p);_,b,_,_,_=s.read_journal(clean)
            self.assertEqual([x['result']['choiceLogits'] for x in a],[x['result']['choiceLogits'] for x in b])
            n=loader.call_count;before=p.read_bytes();runtime.run(p,parent,resume=True)
            self.assertEqual(loader.call_count,n);self.assertEqual(p.read_bytes(),before)
            p.with_suffix('.r0-shuffled.safetensors').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):runtime.run(p,parent,resume=True)


if __name__=='__main__':unittest.main()
