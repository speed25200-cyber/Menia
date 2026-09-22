"""Targeted binding-position follow-up on already observed prospective cases.

The v1 collector and reader remain unchanged to preserve their frozen hashes.
This reader additionally checks targeted masks and exact parent reproduction.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from research.cross_model_prediction import digest
from research import prospective_state_discovery as parent_study
from research.prospective_binding_permutation import binding_swap
from research.iphone_coupling_report import require

PREPARATION = Path(__file__).resolve().parents[1]/'artifacts/prospective-binding-preparation/design.json'
CONDITIONS = ('actual', 'same_schedule', 'values_permuted', 'joint_permuted', 'restored')
FILES = tuple(dict.fromkeys(parent_study.FILES + ('prospective_binding_discovery.py',
    'prospective_binding_discovery_gpu.py','prospective_binding_permutation.py')))
PARENT_JOURNAL = Path(__file__).resolve().parents[1]/'artifacts/prospective-state-pilot/prospective-state-20260920-v1.jsonl'
MASKS = PREPARATION.with_name('tokenizer-check.json')



def source_hash():
    return digest({n:Path(__file__).with_name(n).read_text(encoding='utf-8') for n in FILES})


def plan():
    p=parent_study.plan()
    p.update(schema='menia-prospective-binding-discovery-plan-v1',
        parentJournalSHA256=hashlib.sha256(PARENT_JOURNAL.read_bytes()).hexdigest(),
        preparedMaskSHA256=hashlib.sha256(MASKS.read_bytes()).hexdigest(),
        baselinePolicy='Reproduce all 24 actual raw generations, metrics and cache hashes; reproduce all 48 actual branch records exactly. Stop and preserve any mismatch.',
        dataReuse='The same 24 already examined cases from the global-permutation discovery; no new held-out confirmation.',
        scope='Exploratory targeted-position intervention and untrained native forecast on reused cases. No trained comparator, decision utility, consciousness or novelty criterion.')
    p['intervention']=dict(p['intervention'],permutation='pair observed 0-bit positions with observed 1-bit positions; fix every other prefix position',
        restoration='inverse of the same involutive binding-position permutation')
    return p


def prepared_masks():
    data=json.loads(MASKS.read_text(encoding='utf-8'))
    require(data['sourceJournalSHA256']==plan()['parentJournalSHA256'],'Masks use another parent')
    require(len(data['masks'])==24 and [m['case'] for m in data['masks']]==list(range(24)),'Prepared mask cases')
    return {m['case']:m for m in data['masks']}


def parent_data(tokenizer):
    data=parent_study.read_journal(PARENT_JOURNAL,tokenizer)
    require(data['complete'] and data['failure'] is None,'Complete parent required')
    return data


def calls(p=None):
    p=plan() if p is None else p
    order=[]
    for case in p['cases']:
        shift=case['id']%len(CONDITIONS); rotated=CONDITIONS[shift:]+CONDITIONS[:shift]
        order.extend(dict(case=case['id'],condition=condition) for condition in rotated)
    return order


def compile_branches(tokenizer,prefix,case):
    from research.confidence_generation_continuity import compile_raw_continuations
    result=compile_raw_continuations(tokenizer,prefix,case['queries'])
    for key,branch in result.items():
        config=plan()['branches'][key]; ids=[tokenizer.encode(c,add_special_tokens=False) for c in config['codes']]
        require(all(len(i)==1 for i in ids) and ids[0]!=ids[1], 'Single distinct codes')
        branch.update(candidateTokenIds=[i[0] for i in ids],negative=config['negative'],positive=config['positive'])
    return result


def read_journal(path,tokenizer,*,require_actual=True,fixture_parent=None):
    from research.audit_generation_numerics import validate_decode
    from research.prospective_cache_interventions import inverse_permutation
    events=[]; previous='0'*64
    for index,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines()):
        e=json.loads(line)
        require(set(e)=={'sequence','previous','payload','sha256'} and e['sequence']==index and e['previous']==previous,'Journal order')
        require(e['sha256']==digest({k:e[k] for k in ('sequence','previous','payload')}),'Journal hash')
        events.append(e['payload']); previous=e['sha256']
    require(bool(events),'Empty journal'); header=events[0]; p=plan(); order=calls(p)
    require(header['event']=='header' and header['plan']==p and header['planHash']==digest(p) and header['sourceHash']==source_hash(),'Frozen design')
    if require_actual: require(header['origin']=='transformers_gpu','Actual GPU collection required')
    require(header['metadata']['model']==p['model'] and header['metadata']['weightUpdates']==0,'Model identity')
    require(fixture_parent is None or not require_actual,'Fixture parent forbidden for actual audit')
    reference=parent_data(tokenizer) if fixture_parent is None else fixture_parent
    masks=prepared_masks()
    states={}; inputs={}; forecasts={}; tasks={}; pending=None; barrier=False; complete=False; failure=None
    for e in events[1:]:
        require(not complete and failure is None,'Event after terminal')
        kind=e['event']
        if kind=='failure': failure=e; continue
        if kind=='state':
            require(not forecasts and pending is None and not barrier and len(states)<24,'State timing')
            case=p['cases'][len(states)]; require(e['case']==case,'State case')
            parent=reference['states'][case['id']]
            require(all(e[k]==parent[k] for k in ('case','text','metrics','trace')),'Parent generation or state changed')
            trace=e['trace']; prompt=tokenizer.apply_chat_template(case['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            prompt_ids=tokenizer.encode(prompt,add_special_tokens=False); generated=trace['generatedTokenIds']; prefix=prompt_ids+generated
            require(trace['promptTokenIds']==prompt_ids and trace['promptHash']==digest(prompt_ids) and trace['generatedHash']==digest(generated),'Raw prompt')
            require(1<=len(generated)<=32 and generated[-1]==tokenizer.eos_token_id and trace['finalTokenId']==generated[-1],'Completed generation')
            require(trace['generationForwardInputLengths']==[len(prompt_ids)]+[1]*(len(generated)-1) and
                trace['cacheTokensBeforeFinalToken']==len(prefix)-1 and trace['cacheTokensAfterFinalToken']==len(prefix),'Generation accounting')
            require(e['metrics']['inputTokens']==len(prompt_ids) and e['metrics']['outputTokens']==len(generated) and
                e['text']==tokenizer.decode(generated,skip_special_tokens=True).strip(),'Generation text or counts')
            branches=compile_branches(tokenizer,prefix,case)
            require(e['branchInputHashes']=={k:digest(v['inputIds']) for k,v in branches.items()},'Future queries changed')
            hashes=e['cacheHashes']; require(set(hashes)==set(CONDITIONS) and all(type(h)==str and len(h)==64 for h in hashes.values()),'Cache identities')
            require(hashes['actual']==trace['snapshotCacheHash']==hashes['same_schedule']==hashes['restored'],'Exact cache controls')
            require(e['checks']=={k:True for k in p['strictControls'][:4]},'State checks')
            require(set(e['interventions'])=={'values_permuted','joint_permuted','restored'},'Intervention records')
            mask=binding_swap(tokenizer,case,prompt_ids,len(prefix))
            require(e['bindingMask']==mask,'Targeted mask changed')
            if require_actual: require(mask==masks[case['id']],'Prepared mask differs from observed state')
            indices=mask['permutation']
            for name,record in e['interventions'].items():
                expected_mode='keys_and_values' if name=='joint_permuted' else 'values_only'
                expected_indices=inverse_permutation(indices) if name=='restored' else indices
                source=hashes['values_permuted'] if name=='restored' else hashes['actual']
                require(record['mode']==expected_mode and record['layers']==p['intervention']['layers'] and
                    record['permutation']==expected_indices and record['permutationHash']==digest(expected_indices),'Permutation changed')
                require(record['sourceCacheHash']==source and record['resultCacheHash']==hashes[name] and record['prefixTokens']==len(prefix),'Intervention identity')
                require(record['tokensUnchanged'] and record['sourceCacheUnchanged'] and record['parameterVersionsUnchanged'],'Mutation guard')
                require(all(type(record[k]) in (int,float) and math.isfinite(record[k]) and record[k]>=0
                    for k in ('maximumAbsoluteDisplacement','absoluteL2','referenceL2')),'Intervention magnitude')
            exported=e['cacheExport']
            require((exported is not None)==(case['id'] in p['exportActualCacheCases']),'Prespecified cache export')
            if exported is not None:
                require(exported['file']==f'prospective-binding-20260920-v1.case{case["id"]:02d}.safetensors' and
                    exported['cacheHash']==hashes['actual'] and exported['tensors']==72 and exported['bytes']>0 and len(exported['sha256'])==64,'Cache export identity')
            states[case['id']]=e; inputs[case['id']]=branches
        elif kind=='request':
            require(len(states)==24 and pending is None,'Request before states or pending request')
            phase='task' if barrier else 'forecast'; results=tasks if barrier else forecasts
            require(len(results)<120 and e==dict(event='request',phase=phase,call=order[len(results)],
                inputHash=digest(inputs[order[len(results)]['case']][phase]['inputIds'])),'Request order or input')
            pending=e
        elif kind in ('forecast','task'):
            require(pending is not None and kind==pending['phase'] and e['call']==pending['call'],'Unpaired result')
            case_id=e['call']['case']; condition=e['call']['condition']; key=(case_id,condition); decoded=e['decoded']
            validate_decode(decoded,p['branches'][kind],tokenizer)
            if decoded['firstTopIsCode']:
                position=decoded['candidateTokenIds'].index(decoded['tokenIds'][0])
                require(decoded['candidateLogits'][position]==max(decoded['candidateLogits']),'Native code contradicts greedy logits')
            require(decoded['snapshotCacheHash']==states[case_id]['cacheHashes'][condition] and
                decoded['modelStateId']==states[case_id]['trace']['modelStateId'] and decoded['snapshotUnchanged'],'Decoded state identity')
            require(decoded['settings']==dict(do_sample=False,use_cache=True,candidateRestriction=False,max_new_tokens=2,interventionDuringBranch=False),'Decoder settings')
            require(type(e['seconds']) in (int,float) and math.isfinite(e['seconds']) and e['seconds']>=0,'Duration')
            if condition=='actual': require(decoded==reference['forecasts' if kind=='forecast' else 'tasks'][key]['decoded'],'Parent branch changed')
            (forecasts if kind=='forecast' else tasks)[key]=e; pending=None
        elif kind=='forecasts_complete':
            require(not barrier and len(forecasts)==120 and not tasks and pending is None,'Forecast barrier order')
            require(e==dict(event='forecasts_complete',count=120,forecastHash=digest(list(forecasts.values()))),'Forecast freeze')
            barrier=True
        elif kind=='complete':
            require(barrier and len(tasks)==120 and pending is None and e==dict(event='complete',states=24,forecasts=120,tasks=120),'Incomplete experiment')
            for records in (forecasts,tasks):
                for case in range(24):
                    a=records[(case,'actual')]['decoded']
                    require(records[(case,'same_schedule')]['decoded']==a and records[(case,'restored')]['decoded']==a,'Branch controls failed')
            complete=True
        else: raise ValueError('Unknown event '+kind)
    return dict(header=header,states=states,forecasts=forecasts,tasks=tasks,complete=complete,failure=failure,chainEnd=previous)


def summarize(data):
    require(data['complete'] and data['failure'] is None,'Successful complete journal required')
    p=data['header']['plan']; units={}; outcomes={}
    for condition in CONDITIONS:
        values=[]
        for case in p['cases']:
            key=(case['id'],condition); f=data['forecasts'][key]['decoded']; task=data['tasks'][key]['decoded']
            correct=task['validNativeResponse'] and task['decision']==('one' if case['expected'] else 'zero')
            value=dict(case=case['id'],correct=bool(correct),taskValid=task['validNativeResponse'],
                forecastValid=f['validNativeResponse'],probability=f['conditionalPositive'],forecastMass=f['candidateMass'])
            values.append(value); outcomes[key]=value
        n=len(values); valid=[v for v in values if v['forecastValid']]
        units[condition]=dict(n=n,correct=sum(v['correct'] for v in values),taskValid=sum(v['taskValid'] for v in values),
            forecastValid=len(valid),meanConditionalForecast=math.fsum(v['probability'] for v in values)/n,
            meanForecastCandidateMass=math.fsum(v['forecastMass'] for v in values)/n,
            brierConditionalAll=math.fsum((v['probability']-int(v['correct']))**2 for v in values)/n,
            brierNativeForecasts=math.fsum((v['probability']-int(v['correct']))**2 for v in valid)/len(valid) if valid else None,
            cases=values)
    contrasts={}
    for condition in CONDITIONS[1:]:
        pairs=[(outcomes[(case['id'],'actual')],outcomes[(case['id'],condition)]) for case in p['cases']]
        changed=[(a,b) for a,b in pairs if a['correct']!=b['correct']]
        valid_changed=[(a,b) for a,b in changed if a['taskValid'] and b['taskValid']]
        contrasts[condition]=dict(correctToFailure=sum(a['correct'] and not b['correct'] for a,b in pairs),
            failureToCorrect=sum(not a['correct'] and b['correct'] for a,b in pairs),
            taskSuccessChange=math.fsum(int(b['correct'])-int(a['correct']) for a,b in pairs)/len(pairs),
            meanConditionalForecastChange=math.fsum(b['probability']-a['probability'] for a,b in pairs)/len(pairs),
            changedOutcomes=len(changed),nativeForecastsOnChangedOutcomes=sum(a['forecastValid'] and b['forecastValid'] for a,b in changed),
            taskInvalidations=sum(a['taskValid'] and not b['taskValid'] for a,b in pairs),
            changedValidTaskOutcomes=len(valid_changed),
            forecastFollowsChangedValidTaskOutcome=sum(a['forecastValid'] and b['forecastValid'] and
                (b['probability']-a['probability'])*(int(b['correct'])-int(a['correct']))>p['forecastDirectionMargin'] for a,b in valid_changed),
            forecastFollowsChangedOutcome=sum(a['forecastValid'] and b['forecastValid'] and
                (b['probability']-a['probability'])*(int(b['correct'])-int(a['correct']))>p['forecastDirectionMargin'] for a,b in changed))
    return dict(schema='menia-prospective-binding-discovery-summary-v1',planHash=data['header']['planHash'],sourceHash=data['header']['sourceHash'],
        chainEnd=data['chainEnd'],cases=24,forecasts=120,tasks=120,strictControlsPassed=True,units=units,pairedContrasts=contrasts,
        outcomeDefinition='One future deterministic native code/EOS recall. Invalid task output counts as failure; task validity is reported separately.',
        probabilityCaveat='Conditional first-token code probability, not an independently calibrated probability of success. All-format Brier remains descriptive if native format or candidate mass collapses.',
        seconds=math.fsum(e['seconds'] for e in list(data['forecasts'].values())+list(data['tasks'].values())),scope=p['scope'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args(); require(args.prepare,'Use --prepare')
    p=plan(); result=dict(plan=p,planHash=digest(p),sourceHash=source_hash())
    PREPARATION.parent.mkdir(parents=True,exist_ok=True)
    with PREPARATION.open('x',encoding='utf-8',newline='\n') as stream: stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('planHash','sourceHash')}))
