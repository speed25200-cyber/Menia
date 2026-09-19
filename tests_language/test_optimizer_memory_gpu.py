import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file,load_file

from research import optimizer_memory as study
from research import optimizer_memory_gpu as runtime
from research.native_localization_gpu import install_adapters,adapter_state,load_adapter,file_hash
from tests_language.test_learning_diagnostic_gpu import model,tokenizer


class OptimizerMemoryRuntimeTests(unittest.TestCase):
    def test_real_qwen_reference_resume_and_counterfactual_integrity(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp,contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);parent=root/'parent.jsonl';composition=root/'composition.jsonl';journal=root/'memory.jsonl'
            base_plan=study.previous.plan();blocks=[b for b in base_plan['blocks'] if b['replication']==0 and b['split']=='train'][:2]
            groups=[dict(block=blocks[0]['id'],epoch=3,publicPosition=0,monitorFamily='hidden'),
                    dict(block=blocks[1]['id'],epoch=3,publicPosition=1,monitorFamily='hidden'),
                    dict(block=blocks[0]['id'],epoch=4,publicPosition=2,monitorFamily='visible')]
            old=dict(blocks=blocks,training=[dict(key='r0-composed',groups=groups)])
            tok=tokenizer();tok.add_tokens(['3']);choices=[tok.encode(str(i),add_special_tokens=False)[0] for i in range(4)]
            reference=model(36,8).to(torch.bfloat16);modules=install_adapters(reference,rank=8)
            initial=root/'parent.r0-strong.safetensors';save_file(adapter_state(modules),str(initial));parent_sha=file_hash(initial)
            params=[p for p in reference.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=.0002,weight_decay=0.)
            lookup={b['id']:b for b in blocks}
            def inputs(b,f,t,p,form='trained',mapping=0):return runtime.encode(tok,b,f,t,p,'cpu',form,mapping)
            for group in groups:runtime.balanced_update(reference,opt,params,lookup[group['block']],'composed',group,choices,inputs)
            expected=root/'expected.safetensors';save_file(adapter_state(modules),str(expected));expected_sha=file_hash(expected)
            stack.enter_context(patch.dict(study.CONFIG,dict(replications=1,prefixSteps=2,tailSteps=1,testBlocks=0,lexicalBlocks=0,resamples=20)))
            # This synthetic lineage deliberately uses tiny weights, not the published Qwen checkpoints.
            stack.enter_context(patch.object(study,'source_hash',return_value='1'*64))
            stack.enter_context(patch.object(study.previous,'plan',return_value=old))
            stack.enter_context(patch.object(study.diagnostic,'old_blocks',return_value=blocks))
            stack.enter_context(patch.object(study.diagnostic,'plan',return_value={'blocks':[]}))
            stack.enter_context(patch.object(study.diagnostic,'checkpoints',return_value={'r0-composed':expected_sha}))
            stack.enter_context(patch.object(study.previous.previous,'checkpoints',return_value={'r0-strong':parent_sha}))
            stack.enter_context(patch.object(runtime,'verify_sources',return_value={'r0-composed':expected_sha}))
            stack.enter_context(patch('research.cross_model_gpu.environment',return_value={'gpu':'CPU synthetic fixture'}))
            stack.enter_context(patch('transformers.AutoTokenizer.from_pretrained',return_value=tok))
            loader=stack.enter_context(patch('transformers.AutoModelForCausalLM.from_pretrained',side_effect=lambda *a,**k:model(36,8).to(torch.bfloat16)))
            stack.enter_context(patch('torch.cuda.synchronize'))
            original=runtime.balanced_update;calls=0
            def interrupt(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==3:raise RuntimeError('Injected interruption at first tail')
                return original(*args,**kwargs)
            with patch.object(runtime,'balanced_update',side_effect=interrupt),self.assertRaisesRegex(RuntimeError,'Injected'):
                runtime.run(journal,parent,composition)
            h,rows,done,pending,events=study.read_journal(journal)
            self.assertEqual(list(done),['r0-prefix']);self.assertFalse(rows)
            saved_prefix=runtime.checkpoint(journal,'r0-prefix').read_bytes()
            # Resume must restore the saved prefix and optimizer, not retrain it.
            with patch.object(runtime,'balanced_update',wraps=original) as updates:
                runtime.run(journal,parent,composition,resume=True)
                self.assertEqual(updates.call_count,2)  # carry and reset; zero-grad has no forward/backward.
            self.assertEqual(runtime.checkpoint(journal,'r0-prefix').read_bytes(),saved_prefix)
            self.assertEqual(file_hash(runtime.checkpoint(journal,'r0-carry')),expected_sha)
            report=study.analyze(journal)
            self.assertTrue(report['complete']);self.assertEqual(report['recorded'],64)
            self.assertEqual(report['trainingUpdates'],5);self.assertEqual(report['trainingRestarts'],1)
            self.assertEqual(report['shamPairs'],16)
            h,rows,done,pending,events=study.read_journal(journal)
            starts=[e for e in events if e['event']=='training_start' and e['fork'] is not None]
            self.assertEqual(len({e['initialHash'] for e in starts}),1)
            self.assertEqual(len({e['fork']['sourceStateHash'] for e in starts}),1)
            for event in starts:
                if event['unit']['arm']=='reset_m':self.assertEqual(event['fork']['after']['firstMomentNorm'],0.)
            before=journal.read_bytes();count=loader.call_count
            runtime.run(journal,parent,composition,resume=True)
            self.assertEqual(journal.read_bytes(),before);self.assertEqual(loader.call_count,count)
            # Invalid lineage and broken shams must not be accepted by the report.
            modified=copy.deepcopy(events)
            next(e for e in modified if e['event']=='training_complete' and e['key']=='r0-carry')['sha256']='0'*64
            corrupt=root/'corrupt.jsonl';corrupt.write_text('\n'.join(json.dumps(e) for e in modified)+'\n')
            with self.assertRaisesRegex(ValueError,'not reproduced'):study.read_journal(corrupt)
            modified=copy.deepcopy(events)
            target=next(r['request']['id'] for r in rows if r['request']['position']==3)
            next(e for e in modified if e['event']=='result' and e['id']==target)['choiceMass']*=.99
            corrupt.write_text('\n'.join(json.dumps(e) for e in modified)+'\n')
            with self.assertRaisesRegex(ValueError,'Sham differs'):study.analyze(corrupt)
            premature=[events[0],next(e for e in events if e['event']=='request')]
            corrupt.write_text('\n'.join(json.dumps(e) for e in premature)+'\n')
            with self.assertRaisesRegex(ValueError,'Premature request'):study.read_journal(corrupt)
            # The runner itself stops before any evaluation when the carry reference is wrong.
            with patch.object(study.diagnostic,'checkpoints',return_value={'r0-composed':'0'*64}),patch.object(runtime,'verify_sources',return_value={'r0-composed':'0'*64}):
                bad=root/'bad-reference.jsonl'
                with self.assertRaisesRegex(ValueError,'Exact reproduction failed'):runtime.run(bad,parent,composition)
                self.assertFalse(any(json.loads(line)['event']=='request' for line in bad.read_text().splitlines()))


if __name__=='__main__':unittest.main()
