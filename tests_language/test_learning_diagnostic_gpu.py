import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config,Qwen3ForCausalLM

from research import learning_diagnostic as study
from research import learning_diagnostic_gpu as runtime
from research.native_localization_gpu import adapter_state,install_adapters


def model(layers=2,width=16):
    torch.manual_seed(92)
    config=Qwen3Config(vocab_size=32,hidden_size=width,intermediate_size=width*2,num_hidden_layers=layers,
        num_attention_heads=2,num_key_value_heads=1,head_dim=width//2,max_position_embeddings=512)
    config._attn_implementation='sdpa';config._commit_hash=study.MODEL['revision']
    return Qwen3ForCausalLM(config).eval()


def tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    t=Tokenizer(WordLevel({'[UNK]':0,'0':1,'1':2,'2':3,'SIGNAL':4,'REPERE':5},unk_token='[UNK]'))
    t.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=t,unk_token='[UNK]')
    tok.chat_template='{% for m in messages %}{{ m["content"] }}{% endfor %}'
    return tok


class DiagnosticRuntimeTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_balanced_update_matches_explicit_objective_and_freezes_base(self):
        with patch.dict(study.CONFIG,dict(layer=0)):
            first=model();modules=install_adapters(first,rank=2);second=copy.deepcopy(first)
            base=first.model.embed_tokens.weight.detach().clone()
            parameters=[p for p in first.parameters() if p.requires_grad]
            reference=[p for p in second.parameters() if p.requires_grad]
            opt=torch.optim.AdamW(parameters,lr=.0002,weight_decay=0.)
            opt2=torch.optim.AdamW(reference,lr=.0002,weight_decay=0.)
            b=dict(noiseSeed=42,shuffledTargets=[2,0,1])
            def inputs_for(b,a,task,pos,marker=None):
                inputs=dict(input_ids=torch.tensor([[1,2,3,4,5,6,7,8,marker or 9,10]]))
                return inputs,[[1,2],[4,5]],'fixture'
            rng=torch.random.get_rng_state().clone()
            result=runtime.balanced_update(first,opt,parameters,b,'shuffled',[1,2,3],inputs_for)
            opt2.zero_grad();loss=0.
            for pos in range(3):
                x,spans,_=inputs_for(b,'shuffled','primary',pos)
                logits,_=runtime.evaluate_forward(second,x,spans,b,'shuffled',pos)
                loss=loss+torch.nn.functional.cross_entropy(logits[None],torch.tensor([[1,2,3][b['shuffledTargets'][pos]]]))/6
            for marker in (1,2):
                x,spans,_=inputs_for(b,'shuffled','marker',0,marker=marker)
                logits,_=runtime.evaluate_forward(second,x,spans,b,'shuffled',0)
                loss=loss+torch.nn.functional.cross_entropy(logits[None],torch.tensor([[1,2,3][marker]]))/4
            loss.backward();torch.nn.utils.clip_grad_norm_(reference,1.);opt2.step()
            for a,b in zip(parameters,reference): self.assertTrue(torch.allclose(a,b,atol=2e-7,rtol=1e-5))
            self.assertTrue(torch.equal(base,first.model.embed_tokens.weight));self.assertIsNone(first.model.embed_tokens.weight.grad)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            self.assertEqual(len(result['traces']),5)
            self.assertTrue(any(torch.count_nonzero(m.b)>0 for m in modules.values()))

    def test_visible_control_changes_tokens_hidden_assignment_does_not(self):
        t=tokenizer();b=study.plan()['blocks'][0]
        x,s,h=runtime.encode(t,b,'strong','primary',1,'cpu')
        y,ss,hh=runtime.encode(t,b,'shuffled','primary',2,'cpu')
        self.assertTrue(torch.equal(x.input_ids,y.input_ids));self.assertEqual((s,h),(ss,hh))
        x,_,_=runtime.encode(t,b,'visible','primary',1,'cpu')
        y,_,_=runtime.encode(t,b,'visible','primary',2,'cpu')
        self.assertFalse(torch.equal(x.input_ids,y.input_ids))

    def test_real_tiny_qwen_full_run_restart_restore_checkpoints_and_noop(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            stack.enter_context(patch.dict(study.CONFIG,dict(trainBlocks=2,testBlocks=2,epochs=1)))
            stack.enter_context(patch('research.cross_model_gpu.environment',return_value={'gpu':'CPU synthetic fixture'}))
            stack.enter_context(patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer()))
            loader=stack.enter_context(patch('transformers.AutoModelForCausalLM.from_pretrained',side_effect=lambda *a,**k:model(36,8).to(torch.bfloat16)))
            stack.enter_context(patch('torch.cuda.synchronize'))
            journal=Path(tmp)/'fixture.jsonl'
            original=runtime.balanced_update;calls=0
            def interrupted(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2: raise RuntimeError('Injected training interruption')
                return original(*args,**kwargs)
            with patch.object(runtime,'balanced_update',side_effect=interrupted),self.assertRaises(RuntimeError):
                runtime.run(journal)
            runtime.run(journal,resume=True)
            report=study.analyze(journal)
            self.assertTrue(report['complete']);self.assertEqual(report['recorded'],224)
            self.assertEqual(report['trainingRestarts'],1)
            self.assertEqual(len(list(Path(tmp).glob('*.safetensors'))),4)
            n=loader.call_count;before=journal.read_bytes()
            runtime.run(journal,resume=True)
            self.assertEqual(loader.call_count,n);self.assertEqual(journal.read_bytes(),before)
            journal.with_suffix('.visible.safetensors').write_bytes(b'invalid checkpoint fixture')
            with self.assertRaises(ValueError): runtime.run(journal,resume=True)


if __name__=='__main__': unittest.main()
