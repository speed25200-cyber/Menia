"""Actual tiny random Qwen training/forward API checks, CPU and offline."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from research.native_localization_gpu import adapter_state, encode, forward, install_adapters, intervention, load_adapter, summarize_logits
from research.native_localization import plan
from tests_language import test_cross_model_gpu as fixtures


class NativeRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(82)
        config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                          num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=512)
        config._attn_implementation='sdpa'
        self.model=Qwen3ForCausalLM(config).eval()
        self.inputs=dict(input_ids=torch.arange(1,31).reshape(1,-1))
        self.spans=[[i,i+1] for i in (1,5,9,13,17)]
        self.block=dict(layer=0,strength=.3,noiseSeed=79)

    def test_rotation_changes_native_logits_sham_and_restoration_exact(self):
        with torch.no_grad():
            normal,_=forward(self.model,self.inputs,self.spans,self.block,0)
            sham,stats=forward(self.model,self.inputs,self.spans,self.block,6)
            changed,stats2=forward(self.model,self.inputs,self.spans,self.block,2)
            again,_=forward(self.model,self.inputs,self.spans,self.block,0)
        self.assertTrue(torch.equal(normal,sham)); self.assertTrue(torch.equal(normal,again))
        self.assertFalse(torch.equal(normal,changed)); self.assertTrue(stats2['changed'])
        self.assertLess(stats2['normRelativeError'],.01)
        self.assertFalse(stats['changed'])

    def test_gradient_updates_only_adapters_and_reset_and_disable_restore(self):
        self.model.to(torch.bfloat16)
        weights={k:v.detach().clone() for k,v in self.model.state_dict().items()}
        with torch.no_grad(): original,_=forward(self.model,self.inputs,self.spans,self.block,0)
        modules=install_adapters(self.model,rank=2); initial=adapter_state(modules)
        params=[p for p in self.model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=.02)
        for _ in range(3):
            optimizer.zero_grad()
            logits,_=forward(self.model,self.inputs,self.spans,self.block,3)
            torch.nn.functional.cross_entropy(logits[None],torch.tensor([3])).backward(); optimizer.step()
        self.assertTrue(any(not torch.equal(v,initial[k]) for k,v in adapter_state(modules).items()))
        for name, module in modules.items():
            self.assertTrue(torch.equal(module.base.weight,weights[name+'.weight']))
            self.assertIsNone(module.base.weight.grad)
            module.enabled=False
        with torch.no_grad(): disabled,_=forward(self.model,self.inputs,self.spans,self.block,0)
        self.assertTrue(torch.equal(original,disabled))
        load_adapter(modules,initial)
        for module in modules.values(): module.enabled=True
        with torch.no_grad(): reset,_=forward(self.model,self.inputs,self.spans,self.block,0)
        self.assertTrue(torch.equal(original,reset))

    def test_hook_does_not_change_sampling_rng_and_cleans_up_on_error(self):
        for dtype in (torch.float32,torch.bfloat16):
            self.model.to(dtype)
            rng=torch.random.get_rng_state().clone()
            with torch.no_grad(): forward(self.model,self.inputs,self.spans,self.block,1)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        before=[len(m._forward_hooks) for m in self.model.modules()]
        with self.assertRaises(RuntimeError):
            with intervention(self.model,0,self.spans,1,.3,9): raise RuntimeError('fixture')
        self.assertEqual(before,[len(m._forward_hooks) for m in self.model.modules()])

    def test_actual_fast_tokenizer_spans_ignore_private_assignment(self):
        fixtures.CrossModelBackendTests.setUpClass(); tok=fixtures.CrossModelBackendTests.tokenizer
        b=plan()['blocks'][0]
        encoded,spans,h=encode(tok,b,'localize','cpu')
        other=dict(b,layer=99,noiseSeed=123,shuffledTargets=[5]*6)
        encoded2,spans2,h2=encode(tok,other,'localize','cpu')
        self.assertTrue(torch.equal(encoded.input_ids,encoded2.input_ids))
        self.assertEqual((spans,h),(spans2,h2))
        self.assertEqual(len(spans),5)
        out=summarize_logits(torch.tensor([1.,2.,3.,4.,5.,6.,10.]),list(range(6)))
        self.assertEqual(out['rawChoice'],-1); self.assertLess(out['choiceMass'],.1)

    def test_full_training_export_interruption_resume_and_completed_noop_on_cpu(self):
        from contextlib import ExitStack, redirect_stdout
        import io
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from research import native_localization as protocol
        from research import native_localization_gpu as runtime
        fixed=plan()
        kept={b['id'] for b in fixed['blocks'] if (b['split']=='train' and b['id'].endswith('000')) or
              (b['split']=='test' and int(b['id'].split('-')[1])<4)}
        fixed['blocks']=[b for b in fixed['blocks'] if b['id'] in kept]
        # Include one validation block so every requested report table is populated.
        val=next(b for b in plan()['blocks'] if b['split']=='validation')
        fixed['blocks'].append(val); kept.add(val['id'])
        fixed['training']=[r for r in fixed['training'] if r[0] in kept]
        fixed['evaluation']=[r for r in fixed['evaluation'] if r['block'] in kept]
        tok=Tokenizer(WordLevel({'[UNK]':0,**{str(i):i+1 for i in range(6)}},unk_token='[UNK]'))
        tok.pre_tokenizer=Whitespace()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,unk_token='[UNK]')
        tokenizer.chat_template='{% for m in messages %}{{ m["content"] }}{% endfor %}'
        def factory(*args,**kwargs):
            torch.manual_seed(79)
            c=Qwen3Config(vocab_size=16,hidden_size=8,intermediate_size=12,num_hidden_layers=36,
                          num_attention_heads=2,num_key_value_heads=1,head_dim=4,max_position_embeddings=512)
            c._attn_implementation='sdpa'; c._commit_hash=protocol.MODEL['revision']
            return Qwen3ForCausalLM(c).eval()
        calls=0
        real_forward=runtime.forward
        def interrupted(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==3: raise RuntimeError('Injected interruption during training')
            return real_forward(*args,**kwargs)
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack,redirect_stdout(io.StringIO()):
            journal=Path(tmp)/'cpu-fixture.jsonl'
            stack.enter_context(patch.object(protocol,'plan',return_value=fixed))
            stack.enter_context(patch.object(runtime,'plan',return_value=fixed))
            stack.enter_context(patch('research.cross_model_gpu.environment',return_value={'gpu':'CPU fixture'}))
            stack.enter_context(patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer))
            model_loader=stack.enter_context(patch('transformers.AutoModelForCausalLM.from_pretrained',side_effect=factory))
            stack.enter_context(patch('torch.cuda.synchronize'))
            with patch.object(runtime,'forward',side_effect=interrupted),self.assertRaisesRegex(RuntimeError,'Injected interruption'):
                runtime.run(journal)
            runtime.run(journal,resume=True)
            report=protocol.analyze(journal)
            self.assertTrue(report['complete']); self.assertEqual(report['trainingRestarts'],1)
            self.assertEqual(len(list(Path(tmp).glob('*.safetensors'))),2)
            before=journal.read_bytes(); loaded=model_loader.call_count
            runtime.run(journal,resume=True)
            self.assertEqual(model_loader.call_count,loaded)
            self.assertEqual(journal.read_bytes(),before)


if __name__=='__main__': unittest.main()
