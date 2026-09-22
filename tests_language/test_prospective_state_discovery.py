import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from research import prospective_state_discovery as study
from research.prospective_cache_interventions import inverse_permutation


class TokenizerFixture:
    eos_token_id=9
    def encode(self,text,**kwargs):
        chunks=text.split('<|im_end|>'); ids=[]
        for index,chunk in enumerate(chunks):
            if index: ids.append(9)
            ids.extend(map(ord,chunk))
        return ids
    def decode(self,tokens,**kwargs): return ''.join(chr(i) for i in tokens if i!=9)
    def apply_chat_template(self,messages,**kwargs):
        return ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n'


def constructed_events():
    p=study.plan(); tokenizer=TokenizerFixture(); states={}; events=[dict(event='header',origin='synthetic_test_fixture',
        metadata=dict(model=p['model'],weightUpdates=0),plan=p,planHash=study.digest(p),sourceHash=study.source_hash())]
    for case in p['cases']:
        prompt=tokenizer.encode(tokenizer.apply_chat_template(case['messages'])); generated=[ord('O'),ord('K'),9]; prefix=prompt+generated
        original=study.digest([case['id'],'original']); altered=study.digest([case['id'],'altered']); joint=study.digest([case['id'],'joint'])
        hashes=dict(actual=original,same_schedule=original,values_permuted=altered,joint_permuted=joint,restored=original)
        indices=study.permutation(len(prefix)); interventions={}
        for name in ('values_permuted','joint_permuted','restored'):
            perm=inverse_permutation(indices) if name=='restored' else indices
            interventions[name]=dict(mode='keys_and_values' if name=='joint_permuted' else 'values_only',layers=list(range(36)),
                permutation=perm,permutationHash=study.digest(perm),prefixTokens=len(prefix),
                sourceCacheHash=altered if name=='restored' else original,resultCacheHash=hashes[name],
                tokensUnchanged=True,sourceCacheUnchanged=True,parameterVersionsUnchanged=True,
                maximumAbsoluteDisplacement=1.,absoluteL2=2.,referenceL2=3.)
        exported=dict(file=f'prospective-state-20260920-v1.case{case["id"]:02d}.safetensors',bytes=100,sha256='f'*64,
            cacheHash=original,tensors=72) if case['id'] in p['exportActualCacheCases'] else None
        state=dict(event='state',case=case,text='OK',metrics=dict(inputTokens=len(prompt),outputTokens=3),
            trace=dict(promptTokenIds=prompt,generatedTokenIds=generated,promptHash=study.digest(prompt),generatedHash=study.digest(generated),
                finalTokenId=9,generationForwardInputLengths=[len(prompt),1,1],cacheTokensBeforeFinalToken=len(prefix)-1,
                cacheTokensAfterFinalToken=len(prefix),snapshotCacheHash=original,modelStateId='constructed-fixture'),
            branchInputHashes={k:study.digest(v['inputIds']) for k,v in study.compile_branches(tokenizer,prefix,case).items()},
            cacheHashes=hashes,checks={k:True for k in p['strictControls'][:4]},interventions=interventions,cacheExport=exported)
        states[case['id']]=state; events.append(state)
    for phase in ('forecast','task'):
        recorded=[]
        for call in study.calls(p):
            state=states[call['case']]; case=state['case']; altered=call['condition']=='values_permuted'
            bit=(0 if altered else 1) if phase=='forecast' else (case['expected']^int(altered))
            positive=.9 if bit else .1; config=p['branches'][phase]
            decoded=dict(candidateTokenIds=[48,49],candidateMeanings=[config['negative'],config['positive']],
                tokenIds=[48+bit,9],eosTokenIds=[9],validNativeResponse=True,decision=config['positive'] if bit else config['negative'],
                firstTokenMeaning=config['positive'] if bit else config['negative'],stoppedAtEOS=True,reachedTokenLimit=False,
                text=str(bit),firstTopIsCode=True,candidateLogits=[0.,math.log(positive/(1-positive))],conditionalPositive=positive,
                candidateMass=.999,snapshotCacheHash=state['cacheHashes'][call['condition']],modelStateId='constructed-fixture',
                snapshotUnchanged=True,settings=dict(do_sample=False,use_cache=True,candidateRestriction=False,max_new_tokens=2,interventionDuringBranch=False))
            events.append(dict(event='request',phase=phase,call=call,inputHash=state['branchInputHashes'][phase]))
            event=dict(event=phase,call=call,decoded=decoded,seconds=.01); events.append(event); recorded.append(event)
        if phase=='forecast': events.append(dict(event='forecasts_complete',count=120,forecastHash=study.digest(recorded)))
    events.append(dict(event='complete',states=24,forecasts=120,tasks=120))
    return events


def write_events(path,events):
    previous='0'*64
    with path.open('w',encoding='utf-8') as stream:
        for i,payload in enumerate(events):
            e=dict(sequence=i,previous=previous,payload=payload); e['sha256']=study.digest(e)
            stream.write(json.dumps(e,ensure_ascii=False)+'\n'); previous=e['sha256']


class ProspectiveDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.events=constructed_events()
    def read(self,events):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.jsonl'; write_events(path,events)
            return study.read_journal(path,TokenizerFixture(),require_actual=False)

    def test_balanced_cases_and_rotating_call_order(self):
        p=study.plan(); self.assertEqual(len(p['cases']),24); self.assertEqual(len(study.calls(p)),120)
        for count in (4,8):
            for target in range(count):
                pair=[c for c in p['cases'] if c['bindings']==count and c['targetIndex']==target]
                self.assertEqual([c['expected'] for c in pair],[0,1])
                self.assertTrue(all(sum(c['values'])==count//2 and c['values'][target]==c['expected'] for c in pair))

    def test_complete_roundtrip_and_paired_forecasts(self):
        result=study.summarize(self.read(self.events))
        self.assertEqual(result['units']['actual']['correct'],24)
        self.assertEqual(result['units']['values_permuted']['correct'],0)
        self.assertAlmostEqual(result['units']['actual']['brierConditionalAll'],.01)
        contrast=result['pairedContrasts']['values_permuted']
        self.assertEqual(contrast['forecastFollowsChangedValidTaskOutcome'],24)
        self.assertEqual(contrast['taskInvalidations'],0)

    def test_early_task_or_changed_request_is_rejected_even_when_rehashed(self):
        bad=copy.deepcopy(self.events); index=next(i for i,e in enumerate(bad) if e['event']=='request')
        bad[index]['phase']='task'
        with self.assertRaises(ValueError): self.read(bad)
        bad=copy.deepcopy(self.events); bad[index]['inputHash']='e'*64
        with self.assertRaises(ValueError): self.read(bad)
        bad=copy.deepcopy(self.events); barrier=next(i for i,e in enumerate(bad) if e['event']=='forecasts_complete'); bad.pop(barrier)
        with self.assertRaises(ValueError): self.read(bad)

    def test_altered_state_or_false_forecast_freeze_is_rejected(self):
        bad=copy.deepcopy(self.events); bad[1]['interventions']['values_permuted']['permutation'][0]=0
        with self.assertRaises(ValueError): self.read(bad)
        bad=copy.deepcopy(self.events); next(e for e in bad if e['event']=='forecasts_complete')['forecastHash']='e'*64
        with self.assertRaises(ValueError): self.read(bad)

    def test_invalid_task_is_failure_not_a_valid_recall_error(self):
        bad=copy.deepcopy(self.events)
        for e in bad:
            if e['event']=='task' and e['call']['condition']=='values_permuted':
                d=e['decoded']; d.update(tokenIds=[65,66],text='AB',decision=None,firstTokenMeaning=None,
                    validNativeResponse=False,firstTopIsCode=False,stoppedAtEOS=False,reachedTokenLimit=True)
        report=study.summarize(self.read(bad)); contrast=report['pairedContrasts']['values_permuted']
        self.assertEqual(contrast['taskInvalidations'],24); self.assertEqual(contrast['changedValidTaskOutcomes'],0)
        self.assertEqual(contrast['forecastFollowsChangedValidTaskOutcome'],0)


if __name__=='__main__': unittest.main()
