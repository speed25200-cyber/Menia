import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from research import presence_specificity as s
from research import presence_specificity_gpu as runtime
from research.native_localization_gpu import install_adapters,adapter_state,file_hash
from tests_language.test_learning_diagnostic_gpu import model,tokenizer


class SpecificityRuntimeTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_encoding_all_questions_keeps_hidden_condition_secret(self):
        tok=tokenizer();b=s.plan()['blocks'][0]
        for q in s.QUESTIONS:
            first=runtime.encode(tok,b,'hidden',q,0,'cpu')
            for pos in (1,2,3):
                other=runtime.encode(tok,b,'hidden',q,pos,'cpu')
                self.assertTrue(torch.equal(first[0].input_ids,other[0].input_ids))
                self.assertEqual(first[1:],other[1:])

    def test_frozen_checkpoints_resume_and_noop_on_tiny_qwen(self):
        with tempfile.TemporaryDirectory() as d,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            stack.enter_context(patch.dict(s.CONFIG,dict(blocksPerReplication=2,replications=1,resamples=10,rank=2)))
            parent=Path(d)/'parent.jsonl';parent.write_bytes(b'synthetic fixture parent')
            frozen={}
            for i,arm in enumerate(('visible','strong','shuffled')):
                net=model(36,8);mods=install_adapters(net,rank=2)
                with torch.no_grad():
                    for m in mods.values(): m.b.fill_((i+1)*.002)
                f=parent.with_suffix(f'.r0-{arm}.safetensors');save_file(adapter_state(mods),str(f));frozen['r0-'+arm]=file_hash(f)
            before={k:parent.with_suffix('.'+k+'.safetensors').read_bytes() for k in frozen}
            stack.enter_context(patch.object(s,'PARENT_JOURNAL_SHA256',file_hash(parent)))
            stack.enter_context(patch.object(s,'checkpoints',return_value=frozen))
            stack.enter_context(patch('research.cross_model_gpu.environment',return_value={'gpu':'CPU synthetic fixture'}))
            stack.enter_context(patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer()))
            loader=stack.enter_context(patch('transformers.AutoModelForCausalLM.from_pretrained',side_effect=lambda *a,**k:model(36,8).to(torch.bfloat16)))
            stack.enter_context(patch('torch.cuda.synchronize'))
            path=Path(d)/'out.jsonl';summarize=runtime.summarize_logits;calls=0
            def stopped(*a,**k):
                nonlocal calls
                calls+=1
                if calls==3: raise RuntimeError('Interrupted forward')
                return summarize(*a,**k)
            with patch.object(runtime,'summarize_logits',side_effect=stopped),self.assertRaises(RuntimeError):runtime.run(path,parent)
            runtime.run(path,parent,resume=True);report=s.analyze(path)
            self.assertTrue(report['complete']);self.assertEqual(report['recorded'],256)
            self.assertEqual(report['interruptedRequests'],1);self.assertEqual(report['shamPairs'],64)
            clean=Path(d)/'clean.jsonl';runtime.run(clean,parent)
            _,a,_,_=s.read_journal(path);_,b,_,_=s.read_journal(clean)
            self.assertEqual([r['result']['choiceLogits'] for r in a],[r['result']['choiceLogits'] for r in b])
            for key,data in before.items():self.assertEqual(parent.with_suffix('.'+key+'.safetensors').read_bytes(),data)
            n=loader.call_count;unchanged=path.read_bytes();runtime.run(path,parent,resume=True)
            self.assertEqual(loader.call_count,n);self.assertEqual(path.read_bytes(),unchanged)
            parent.with_suffix('.r0-strong.safetensors').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checkpoint'):runtime.run(path,parent,resume=True)


if __name__=='__main__':unittest.main()
