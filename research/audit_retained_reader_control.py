"""Audit recorded suffix-reader controls and exported weights, without retraining."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from research import retained_reader_control as c
from research import prospective_pair_discovery as parent
from research.confidence_prefix_interventions import tensor_hash
from research.audit_generation_numerics import validate_decode


def verify(path, tokenizer):
    import torch
    from safetensors.torch import load_file
    path=Path(path); events=c.read_events(path); p=c.plan()
    assert events[0]['origin']=='transformers_gpu' and events[-1]['event']=='complete'
    assert events[0]['metadata']['model']==p['model']
    reference=parent.read_journal(c.PARENT,tokenizer)
    counts=Counter(e['event'] for e in events)
    assert counts==dict(header=1,state=2,producer_task_before=8,producer_forecast=8,reader=1,zero_reader=8,
        training_example=8,updated=1,trained_reader=8,reload_verified=1,producer_task_after=8,complete=1)
    pos={event:[i for i,e in enumerate(events) if e['event']==event] for event in counts}
    for left,right in zip(('producer_forecast','reader','zero_reader','training_example','updated','trained_reader','reload_verified','producer_task_after'),
                          ('reader','zero_reader','training_example','updated','trained_reader','reload_verified','producer_task_after','complete')):
        assert max(pos[left])<min(pos[right]),(left,right)
    expected_calls=[]; expected_tasks=[]
    for cid in p['cases']:
        old=reference['states'][cid]; mask=old['bindingMask']
        targets=[mask['selectedRows'][0],next(t for t in range(old['case']['bindings']) if t not in mask['selectedRows'])]
        state=next(e for e in events if e['event']=='state' and e['case']==cid)
        assert state==dict(event='state',case=cid,targets=targets,trace=old['trace'],mask=mask,
                          cacheHashes={k:old['cacheHashes'][k] for k in ('actual','values_permuted')})
        for target in targets:
            expected_tasks.extend(dict(case=cid,target=target,condition=condition) for condition in ('actual','values_permuted'))
            expected_calls.extend(dict(case=cid,target=target,condition='values_permuted',mapping=mapping) for mapping in (0,1))
    for phase in ('producer_task_before','producer_task_after'):
        rows=[e for e in events if e['event']==phase]; assert [r['call'] for r in rows]==expected_tasks
        for r in rows: assert r['decoded']==reference['records']['task'][parent.key(r['call'])]['decoded']
    def label(call):
        key=parent.key({k:v for k,v in call.items() if k!='mapping'})
        d=reference['records']['task'][key]['decoded']
        value=reference['states'][call['case']]['case']['values'][call['target']]
        return int(d['validNativeResponse'] and d['decision']==('one' if value else 'zero'))
    for phase in ('producer_forecast','zero_reader','training_example','trained_reader'):
        rows=[e for e in events if e['event']==phase]; assert [r['call'] for r in rows]==expected_calls
    for e in (e for e in events if e['event']=='producer_forecast'):
        assert e['decoded']==reference['records']['forecast'][parent.key(e['call'])]['decoded'] and e['label']==label(e['call'])
    reader=events[pos['reader'][0]]; update=events[pos['updated'][0]]; complete=events[-1]
    assert reader['config']==dict(p['adapter'],targets=['q_proj','v_proj'],activation='first token after the retained prefix; entire query suffix and report')
    assert reader['parameters']==11796480 and reader['producerCopyExact']
    assert update['beforeAdapterHash']==reader['initialAdapterHash']!=update['afterAdapterHash']
    assert math.isfinite(update['gradientNorm']) and update['gradientNorm']>0 and update['seconds']>=0
    weights=path.parent/update['weights']['file']; assert weights.name==path.with_suffix('.adapter.safetensors').name
    assert weights.stat().st_size==update['weights']['bytes'] and hashlib.sha256(weights.read_bytes()).hexdigest()==update['weights']['sha256']
    state=load_file(str(weights)); assert len(state)==144 and sum(t.numel() for t in state.values())==reader['parameters']
    assert all(t.dtype==torch.float32 and bool(torch.isfinite(t).all()) for t in state.values())
    assert c.digest({name:tensor_hash(t) for name,t in state.items()})==update['afterAdapterHash']
    for layer in range(36):
        for module,out in (('q_proj',4096),('v_proj',1024)):
            stem=f'model.layers.{layer}.self_attn.{module}'
            assert state[stem+'.a'].shape==(32,2560) and state[stem+'.b'].shape==(out,32)
    zero_drift=[]; before_losses=[]; after_losses=[]; forecasts={name:[] for name in ('zero_reader','trained_reader')}
    for e in events:
        phase=e['event']
        if phase=='training_example':
            call=e['call']; target=tokenizer.encode(parent.plan()['forecastMappings'][call['mapping']]['codes'][label(call)],add_special_tokens=False)[0]
            assert e['label']==label(call) and e['target']==target and math.isfinite(e['loss']) and e['loss']>=0
            before_losses.append(e['loss'])
        if phase not in forecasts: continue
        call=e['call']; r=e['result']; d=r['decoded']; old=reference['records']['forecast'][parent.key(call)]['decoded']
        validate_decode(d,parent.plan()['forecastMappings'][call['mapping']],tokenizer)
        if d['firstTopIsCode']: assert d['candidateLogits'][d['candidateTokenIds'].index(d['tokenIds'][0])]==max(d['candidateLogits'])
        assert r['sourceCacheHash']==d['snapshotCacheHash']==reference['states'][call['case']]['cacheHashes']['values_permuted']
        expected_hash=reader['initialAdapterHash'] if phase=='zero_reader' else update['afterAdapterHash']
        assert r['adapterHash']==expected_hash and r['config']==reader['config']
        assert r['producerStateId']==old['modelStateId'] and d['modelStateId']==r['producerStateId']+'/suffix-reader/'+expected_hash
        if phase=='zero_reader':
            assert d['tokenIds']==old['tokenIds']
            zero_drift.extend(abs(a-b) for a,b in zip(d['candidateLogits'],old['candidateLogits']))
        else:
            assert e['label']==label(call) and math.isfinite(e['loss']) and e['loss']>=0
            after_losses.append(e['loss'])
        forecasts[phase].append(dict(valid=d['validNativeResponse'],correct=d['validNativeResponse'] and d['decision']==('correct' if label(call) else 'incorrect'),
                                    probability=d['conditionalPositive'],label=label(call)))
    assert math.isclose(math.fsum(before_losses)/8,update['meanLossBefore'],abs_tol=1e-12)
    assert math.isclose(math.fsum(after_losses)/8,complete['meanLossAfter'],abs_tol=1e-12)
    assert events[pos['reload_verified'][0]]==dict(event='reload_verified',adapterHash=update['afterAdapterHash'],branches=8)
    assert complete['counts']==p['counts'] and complete['producerUnchanged'] and complete['peakGPUBytes']>0
    return dict(schema='menia-retained-reader-control-audit-v1',verified=True,
        journalSHA256=hashlib.sha256(path.read_bytes()).hexdigest(),planHash=c.digest(p),sourceHash=c.source_hash(),
        weights=update['weights'],parameters=reader['parameters'],
        nonzeroUpdateMatrices=sum(bool(torch.count_nonzero(t)) for n,t in state.items() if n.endswith('.b')),
        zeroReaderNativeMatches=8,maximumZeroReaderCandidateLogitDrift=max(zero_drift),producerTasksUnchanged=8,
        meanLossBefore=update['meanLossBefore'],meanLossAfter=complete['meanLossAfter'],gradientNorm=update['gradientNorm'],
        updateSeconds=update['seconds'],peakGPUBytes=complete['peakGPUBytes'],
        forecasts={name:dict(n=len(rows),valid=sum(r['valid'] for r in rows),correct=sum(r['correct'] for r in rows),
            brierConditional=math.fsum((r['probability']-r['label'])**2 for r in rows)/len(rows)) for name,rows in forecasts.items()},
        scope='Tokens, code semantics, phases, parent records, arithmetic and exported adapter audited. No independent training replay, initial-weight export, held-out test, calibration or consciousness result.')


if __name__=='__main__':
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('journal',type=Path)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    spec=c.plan()['model']; tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False)
    result=verify(args.journal,tok)
    with args.output.open('x',encoding='utf-8',newline='\n') as f: f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result))
