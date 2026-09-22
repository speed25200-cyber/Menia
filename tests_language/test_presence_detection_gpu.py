import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from research import presence_detection as study
from research import presence_detection_gpu as runtime
from research.native_localization_gpu import install_adapters
from tests_language.test_learning_diagnostic_gpu import model, tokenizer


class PresenceRuntimeTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_class_balanced_objective_matches_explicit_loss_and_freezes_base(self):
        with patch.dict(study.CONDITIONS,{'trained':dict(layer=0,strength=1.)}):
            first=model(); install_adapters(first,rank=2)
            second=copy.deepcopy(first);base=first.model.embed_tokens.weight.detach().clone()
            params=[p for p in first.parameters() if p.requires_grad]
            other=[p for p in second.parameters() if p.requires_grad]
            opt=torch.optim.AdamW(params,lr=.0002,weight_decay=0.)
            ref=torch.optim.AdamW(other,lr=.0002,weight_decay=0.)
            block=dict(noiseSeed=42,shuffledTargets=[2,0,1])
            def inputs(b,a,task,pos,marker=None):
                return dict(input_ids=torch.tensor([[1,2,3,4,5,6,7,8,9 if marker is None else marker+10,12]])),[[1,2],[4,5]],'fixture'
            rng=torch.random.get_rng_state().clone()
            result=runtime.balanced_update(first,opt,params,block,'shuffled',[1,2,3],inputs)
            ref.zero_grad();loss=0.
            # Shuffled presence labels are [1, 0, 1]: the single absent label weighs as much as both present ones.
            for pos,(label,weight) in enumerate(((1,.125),(0,.25),(1,.125))):
                x,spans,_=inputs(block,'shuffled','primary',pos)
                logits,_=runtime.evaluate_forward(second,x,spans,block,'shuffled',pos)
                loss=loss+torch.nn.functional.cross_entropy(logits[None],torch.tensor([[1,2,3][label]]))*weight
            for identity in (0,1):
                x,spans,_=inputs(block,'shuffled','marker',0,marker=identity)
                logits,_=runtime.evaluate_forward(second,x,spans,block,'shuffled',0)
                loss=loss+torch.nn.functional.cross_entropy(logits[None],torch.tensor([[2,3][identity]]))/4
            loss.backward();torch.nn.utils.clip_grad_norm_(other,1.);ref.step()
            self.assertTrue(all(torch.allclose(a,b,atol=2e-7,rtol=1e-5) for a,b in zip(params,other)))
            self.assertTrue(torch.equal(base,first.model.embed_tokens.weight));self.assertIsNone(first.model.embed_tokens.weight.grad)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()));self.assertEqual(len(result['traces']),5)
            self.assertEqual([t['changed'] for t in result['traces']],[False,True,True,False,False])

    def test_conditions_change_layer_and_strength_while_text_stays_identical(self):
        tok=tokenizer();b=study.plan()['blocks'][0]
        fake={'trained':dict(layer=0,strength=1.),'weaker':dict(layer=0,strength=.5),'later':dict(layer=1,strength=1.)}
        with patch.dict(study.CONDITIONS,fake,clear=True):
            # The final layer cannot reach the answer token through attention: keep 'later' below it.
            net=model(3)
            for layout in study.LAYOUTS:
                x,spans,h=runtime.encode(tok,b,'strong','primary',1,'cpu',layout)
                y,other,hh=runtime.encode(tok,b,'shuffled','primary',2,'cpu',layout)
                self.assertTrue(torch.equal(x.input_ids,y.input_ids));self.assertEqual((spans,h),(other,hh))
                outputs=[]
                for condition in fake:
                    with torch.inference_mode():
                        base,_=runtime.evaluate_forward(net,x,spans,b,'strong',0,condition)
                        sham,trace=runtime.evaluate_forward(net,x,spans,b,'strong',3,condition)
                        changed,t=runtime.evaluate_forward(net,x,spans,b,'strong',1,condition)
                        visible,v=runtime.evaluate_forward(net,x,spans,b,'visible',1,condition)
                    self.assertTrue(torch.equal(base,sham));self.assertFalse(torch.equal(base,changed))
                    self.assertTrue(torch.equal(base,visible));self.assertEqual(v['applications'],0)
                    self.assertEqual(trace['applications'],1);self.assertTrue(t['changed'])
                    outputs.append(changed)
                self.assertFalse(torch.equal(outputs[0],outputs[1]));self.assertFalse(torch.equal(outputs[0],outputs[2]))
                self.assertFalse(any(layer._forward_hooks for layer in net.model.layers))

    def test_real_tiny_qwen_resume_across_units_interrupted_eval_and_noop(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            stack.enter_context(patch.dict(study.CONFIG,dict(trainBlocks=1,testBlocks=1,epochs=1,rank=2,resamples=20)))
            stack.enter_context(patch.object(study,'INITIALIZATIONS',(31,32)))
            stack.enter_context(patch.dict(study.CONDITIONS,{k:study.CONDITIONS[k] for k in ('trained','later')},clear=True))
            stack.enter_context(patch('research.cross_model_gpu.environment',return_value={'gpu':'CPU synthetic fixture'}))
            stack.enter_context(patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer()))
            loader=stack.enter_context(patch('transformers.AutoModelForCausalLM.from_pretrained',side_effect=lambda *a,**k:model(36,8).to(torch.bfloat16)))
            stack.enter_context(patch('torch.cuda.synchronize'))
            path=Path(tmp)/'fixture.jsonl';original=runtime.balanced_update;calls=0
            def interrupted(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2: raise RuntimeError('Training interruption after a completed unit')
                return original(*args,**kwargs)
            with patch.object(runtime,'balanced_update',side_effect=interrupted),self.assertRaises(RuntimeError): runtime.run(path)
            retained=path.with_suffix('.r0-visible.safetensors').read_bytes()
            summarize=runtime.summarize_logits;calls=0
            def stopped(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2: raise RuntimeError('Evaluation interruption after one completed request')
                return summarize(*args,**kwargs)
            with patch.object(runtime,'summarize_logits',side_effect=stopped),self.assertRaises(RuntimeError): runtime.run(path,resume=True)
            runtime.run(path,resume=True);report=study.analyze(path)
            self.assertTrue(report['complete']);self.assertEqual(report['recorded'],288)
            self.assertEqual(report['trainingRestarts'],1);self.assertEqual(report['interruptedRequests'],1)
            self.assertEqual(report['unperturbedMismatches'],0)
            self.assertEqual(len(report['trained']),6);self.assertEqual(path.with_suffix('.r0-visible.safetensors').read_bytes(),retained)
            # Interrupted and resumed arms end in the same weights as an uninterrupted attempt.
            clean=Path(tmp)/'clean.jsonl';runtime.run(clean)
            for key in report['trained']:
                self.assertEqual(path.with_suffix(f'.{key}.safetensors').read_bytes(),clean.with_suffix(f'.{key}.safetensors').read_bytes())
            before=path.read_bytes();n=loader.call_count
            runtime.run(path,resume=True)
            self.assertEqual(path.read_bytes(),before);self.assertEqual(loader.call_count,n)
            path.with_suffix('.r1-strong.safetensors').write_bytes(b'corrupted fixture')
            with self.assertRaises(ValueError): runtime.run(path,resume=True)


if __name__=='__main__': unittest.main()
