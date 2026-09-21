"""Verify frozen order, inputs, labels, readout identities and exported tensors."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from research import prospective_reader_learning as study
from research import prospective_pair_discovery as parent
from research.audit_generation_numerics import validate_decode
from research.iphone_coupling_report import require


def finite(x):return type(x) in (int,float) and math.isfinite(x) and x>=0


def validate_order(events,p):
    require(bool(events) and events[0]['event']=='header','Initial header required')
    order=list(study.operations(p));index=0;pending=None;terminal=None;failure=None
    for e in events[1:]:
        require(terminal is None,'Event after terminal')
        if e['event']=='failure':terminal='failed';failure=e;continue
        if e['event']=='complete':
            require(index==len(order) and pending is None,'Premature completion')
            require(e['counts']==p['counts'] and e['producerUnchanged'] is True and finite(e['peakGPUBytes']),'Completion fields')
            terminal='completed';continue
        require(index<len(order),'Unexpected extra operation');kind,call=order[index]
        if pending is None:
            require(e==dict(event='request',kind=kind,call=call),'Operation request order')
            pending=e
        else:
            require(set(e)=={'event','call','result','seconds'} and e['event']==kind and e['call']==call,
                'Result differs from pending operation')
            require(finite(e['seconds']),'Finite duration');index+=1;pending=None
    return dict(complete=terminal=='completed',failure=failure,operations=index,pending=pending)


def validate_state(s,case,tokenizer,p,path):
    require(s['case']==case,'Exact case identity');t=s['trace']
    prompt=tokenizer.apply_chat_template(case['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tokenizer.encode(prompt,add_special_tokens=False);generated=t['generatedTokenIds'];prefix=ids+generated
    identity=p['model']['id']+'@'+p['model']['revision']+'/base-bfloat16-sdpa'
    require(t['schema']=='menia-generated-prefix-continuity-v1' and t['modelStateId']==identity and
        t['textWasRetokenized'] is False,'Generation identity')
    require(t['promptTokenIds']==ids and t['promptHash']==study.digest(ids) and t['generatedHash']==study.digest(generated),'Raw token hashes')
    require(1<=len(generated)<=p['settings']['max_new_tokens'] and generated[-1]==tokenizer.eos_token_id and
        t['finalTokenId']==generated[-1] and tokenizer.eos_token_id in t['eosTokenIds'],'Complete chat generation')
    require(t['generationForwardInputLengths']==[len(ids)]+[1]*(len(generated)-1) and
        t['cacheTokensBeforeFinalToken']==len(prefix)-1 and t['cacheTokensAfterFinalToken']==len(prefix),'Cache trajectory')
    require(s['metrics']['inputTokens']==len(ids) and s['metrics']['outputTokens']==len(generated) and
        s['text']==tokenizer.decode(generated,skip_special_tokens=True).strip(),'Generation counts/text')
    mask=parent.pair_swap(tokenizer,case,ids,len(prefix));require(s['bindingMask']==mask,'Public pair selection')
    branches=parent.compile_branches(tokenizer,prefix,case,p)
    require(s['branchInputHashes']=={k:study.digest(b['inputIds']) for k,b in branches.items()},'Exact branch inputs')
    require(all(len(b['inputIds'])+1<=p['maxContextTokens'] for b in branches.values()),'Untruncated query budget')
    hashes=s['cacheHashes'];require(set(hashes)==set(p['conditions']) and all(type(h) is str and len(h)==64 for h in hashes.values()),'Cache identities')
    require(hashes['actual']==t['snapshotCacheHash']==hashes['same_schedule']==hashes['restored'],'Exact replay/restoration')
    require(set(s['interventions'])=={'values_permuted','joint_permuted','restored'},'Intervention names')
    for name,r in s['interventions'].items():
        require(r['layers']==list(range(36)) and r['permutation']==mask['permutation'] and
            r['permutationHash']==mask['permutationHash'] and r['mode']==('keys_and_values' if name=='joint_permuted' else 'values_only'),'Intervention definition')
        require(r['sourceCacheHash']==hashes['values_permuted' if name=='restored' else 'actual'] and
            r['resultCacheHash']==hashes[name] and r['prefixTokens']==len(prefix),'Intervention source/result')
        require(all(r[k] is True for k in ('tokensUnchanged','sourceCacheUnchanged','parameterVersionsUnchanged')),'Mutation guard')
        require(all(finite(r[k]) for k in ('maximumAbsoluteDisplacement','absoluteL2','referenceL2')),'Finite displacements')
    ex=s['cacheExport'];require((ex is not None)==(case['id'] in p['exportCacheCases']),'Declared cache exports')
    if ex is not None:
        require(ex['file']==path.with_suffix(f'.case{case["id"]}.safetensors').name and ex['cacheHash']==hashes['actual'] and
            ex['tensors']==72 and ex['bytes']>0 and len(ex['sha256'])==64,'Cache export identity')
    return branches


def read_journal(path,tokenizer):
    path=Path(path);events=[];previous='0'*64
    for index,line in enumerate(path.read_text(encoding='utf-8').splitlines()):
        row=json.loads(line);require(set(row)=={'sequence','previous','payload','sha256'} and
            row['sequence']==index and row['previous']==previous,'Journal sequence')
        require(row['sha256']==study.digest({k:row[k] for k in ('sequence','previous','payload')}),'Journal hash')
        previous=row['sha256'];events.append(row['payload'])
    require(bool(events),'Empty journal');h=events[0];p=study.plan()
    require(h['plan']==p and h['planHash']==study.digest(p) and h['sourceHash']==study.source_hash(),'Frozen design')
    require(h['origin']=='transformers_gpu' and h['metadata']['model']==p['model'] and h['metadata']['settings']==p['settings'],'GPU/model settings')
    require(h['metadata']['deterministicAlgorithms'] is True and h['metadata']['tf32'] is False and
        h['metadata']['cublasWorkspace']==':4096:8','Numerical settings')
    status=validate_order(events,p);states={};branches={};weights={};tasks={};forecasts=[];records={};initial=None
    case_map={c['id']:c for c in p['cases']}
    for e in events[1:]:
        kind=e['event']
        if kind in ('request','complete','failure'):continue
        call=e['call'];r=e['result'];cid=call.get('case')
        if kind=='state':
            branches[cid]=validate_state(r,case_map[cid],tokenizer,p,path);states[cid]=r
        elif kind=='weights':
            arm=call['arm'];require(r['file']==path.with_suffix('.'+arm+'.safetensors').name and
                r['parameters']==11796480 and r['tensors']==144 and r['bytes']>0 and
                len(r['sha256'])==len(r['adapterHash'])==64,'Weight record')
            weights[arm]=r
            if arm=='initial':initial=r['adapterHash']
            else:require(r['adapterHash']!=initial,'Training made no parameter change')
        elif kind=='reset':require(r==dict(adapterHash=initial),'Both arms must reset to identical weights')
        elif kind=='step':
            batch=p['optimizer']['batch'];start=call['batch']*batch
            expected=study.training_examples(call['epoch'],p)[start:start+batch]
            require([x['example'] for x in r['examples']]==expected,'Matched training order')
            for x,example in zip(r['examples'],expected):
                require(case_map[example['case']]['split']=='train','Reserved label leaked into training')
                label=study.target_label(example['kind'],example,tasks,case_map)
                b=branches[example['case']][study.branch_name(example['kind'],example)]
                condition=study.memory_condition(call['arm'],example['condition'])
                require(x['label']==label and x['target']==b['candidateTokenIds'][label],'Measured training label or code')
                require(x['inputHash']==study.digest(b['inputIds']) and x['sourceCondition']==condition and
                    x['sourceCacheHash']==states[example['case']]['cacheHashes'][condition],'Training memory view')
                require(finite(x['loss']),'Finite training loss')
            require(math.isclose(r['meanLoss'],math.fsum(x['loss'] for x in r['examples'])/batch,rel_tol=0,abs_tol=1e-12) and
                finite(r['gradientNorm']),'Step arithmetic')
        elif kind in ('coding','forecast','task'):
            b=branches[cid][study.branch_name(kind,call)];d=r['decoded'];arm=call.get('arm','base')
            condition=study.memory_condition(arm,call['condition']);s=states[cid]
            validate_decode(d,p['taskMapping'] if kind=='task' else p['forecastMappings'][call['mapping']],tokenizer)
            require(r['inputHash']==study.digest(b['inputIds']) and r['sourceCondition']==condition,'Decode input/view')
            require(d['snapshotCacheHash']==s['cacheHashes'][condition] and d['snapshotUnchanged'] is True,'Decode cache identity')
            require(d['settings']==dict(do_sample=False,use_cache=True,candidateRestriction=False,max_new_tokens=2,interventionDuringBranch=False),'Native decoding settings')
            if d['firstTopIsCode']:require(d['candidateLogits'][d['candidateTokenIds'].index(d['tokenIds'][0])]==max(d['candidateLogits']),'Greedy code/logits')
            identity=s['trace']['modelStateId']
            if arm=='base':require(r['reader'] is None and d['modelStateId']==identity,'Producer report identity')
            else:
                rr=r['reader'];adapter=weights[arm]['adapterHash']
                require(rr['schema']=='menia-retained-state-reader-decode-v1' and rr['decoded']==d and rr['producerStateId']==identity and
                    rr['sourceCacheHash']==d['snapshotCacheHash'] and rr['adapterHash']==adapter and
                    d['modelStateId']==identity+'/suffix-reader/'+adapter,'Reader and producer identities')
                require(rr['config']==dict(p['adapter'],targets=['q_proj','v_proj'],
                    activation='first token after the retained prefix; entire query suffix and report'),'Reader configuration')
            records[kind,parent.key(call)]=e
            if kind=='task':tasks[cid,call['target'],call['condition']]=r
            if kind=='forecast' and call['split']=='reserved':forecasts.append(e)
        elif kind=='forecasts_complete':require(r==dict(recordsHash=study.digest(forecasts)),'Reserved forecast barrier')
    if status['complete']:
        for (kind,_),e in records.items():
            call=e['call']
            if call['split']!='reserved' or call['condition']!='actual':continue
            controls=p['conditions'] if call.get('arm')=='text' else ['same_schedule','restored']
            for condition in controls:
                other=records[kind,parent.key(dict(call,condition=condition))]
                require(e['result']['decoded']==other['result']['decoded'],'Exact replay, restoration or text-only invariance')
    return dict(header=h,events=events,states=states,weights=weights,records=records,chainEnd=previous,**status)


def audit_weights(path,weights):
    import torch
    from safetensors.torch import load_file
    from research.confidence_prefix_interventions import tensor_hash
    result=[]
    for arm,record in weights.items():
        file=path.parent/record['file'];raw=file.read_bytes()
        require(len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256'],'Weight file identity')
        tensors=load_file(str(file));names=set()
        for layer in range(36):
            for module,width in (('q_proj',4096),('v_proj',1024)):
                stem=f'model.layers.{layer}.self_attn.{module}'
                for part,shape in (('a',(32,2560)),('b',(width,32))):
                    name=stem+'.'+part;names.add(name);t=tensors[name]
                    require(t.dtype==torch.float32 and tuple(t.shape)==shape and bool(torch.isfinite(t).all()),'Weight dtype/shape/finite')
        require(set(tensors)==names and sum(t.numel() for t in tensors.values())==record['parameters'],'Weight inventory')
        require(study.digest({n:tensor_hash(t) for n,t in tensors.items()})==record['adapterHash'],'Weight tensor hash')
        nonzero=sum(bool(torch.count_nonzero(t)) for n,t in tensors.items() if n.endswith('.b'))
        require(arm!='initial' or nonzero==0,'Initial adapter must be inactive')
        result.append(dict(arm=arm,sha256=record['sha256'],nonzeroUpdateMatrices=nonzero,parameters=record['parameters']))
    return result


def verify(path,tokenizer):
    from research.audit_prospective_binding_discovery import cache_audit
    from research.prospective_reader_learning_metrics import summarize
    path=Path(path);data=read_journal(path,tokenizer)
    require(data['complete'] and data['failure'] is None,'Successful complete collection required for metrics')
    summary=summarize(data)
    audit=dict(schema='menia-prospective-reader-learning-audit-v1',verified=True,
        journalSHA256=hashlib.sha256(path.read_bytes()).hexdigest(),planHash=data['header']['planHash'],
        sourceHash=data['header']['sourceHash'],chainEnd=data['chainEnd'],operations=data['operations'],
        exportedWeights=audit_weights(path,data['weights']),
        exportedCaches=[cache_audit(path.parent,data['states'][cid]) for cid in study.plan()['exportCacheCases']],
        allReservedForecastsBeforeTasks=True,matchedInitializationAndTrainingOrder=True,
        scope='Recorded chronology, inputs, native tokens, semantics, label derivation and arithmetic checked. All three adapters and two prefix caches verified. Not an independent rerun of optimization or remote weight attestation.')
    return audit,summary


if __name__=='__main__':
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('journal',type=Path);args=parser.parse_args()
    spec=study.plan()['model'];tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False)
    audit,summary=verify(args.journal,tok)
    for suffix,value in (('.audit.json',audit),('.summary.json',summary)):
        with args.journal.with_suffix(suffix).open('x',encoding='utf-8',newline='\n') as f:
            f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(verified=True,functionalCriteriaPassed=summary['allFunctionalCriteriaPassed'],operations=audit['operations'])))
