"""Question-specific consequences of one hidden memory-pair exchange."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from research import prospective_binding_discovery as parent_study
from research.cross_model_prediction import digest
from research.iphone_coupling_report import require
from research.prospective_pair_permutation import pair_swap

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT/'artifacts/prospective-pair-preparation/design.json'
PARENT_JOURNAL = ROOT/'artifacts/prospective-binding-pilot/prospective-binding-20260920-v1.jsonl'
CONDITIONS = parent_study.CONDITIONS
PHASES = ('coding', 'forecast', 'task')
FILES = tuple(dict.fromkeys(parent_study.FILES + ('prospective_pair_permutation.py',
    'prospective_pair_discovery.py', 'prospective_pair_discovery_gpu.py')))


def source_hash():
    return digest({name: Path(__file__).with_name(name).read_text(encoding='utf-8') for name in FILES})


def plan():
    parent = parent_study.plan()
    return dict(schema='menia-prospective-pair-discovery-plan-v1', model=parent['model'], settings=parent['settings'],
        maxContextTokens=2048, cases=parent['cases'], conditions=list(CONDITIONS),
        parentJournalSHA256=hashlib.sha256(PARENT_JOURNAL.read_bytes()).hexdigest(),
        parentPolicy='Reproduce all 24 actual generations, metrics and traces exactly; reproduce the 24 original-target actual task decodes exactly. Forecast prompts differ.',
        intervention=dict(layers=list(range(36)), permutation='zero row at case_id modulo zero count; one row at floor(case_id / zero count) modulo one count; swap only their bit positions',
            valuesMode='values_only', jointMode='keys_and_values', restoration='same involutive permutation'),
        forecastMappings=[dict(codes=['A','B'],negative='incorrect',positive='correct'),
                          dict(codes=['B','A'],negative='incorrect',positive='correct')],
        taskMapping=dict(codes=['0','1'],negative='zero',positive='one'),
        phases=list(PHASES), counts=dict(states=24,coding=480,forecast=1600,task=800),
        exportActualCacheCases=[0,8], forecastDirectionMargin=.01, weightUpdates=0,
        strictControls=parent['strictControls'],
        dataReuse='All 24 prior episodes, comprising 18 distinct public tables, now queried at every key. 160 case/key pairs are dependent and are not new held-out data.',
        order='All states, then coding checks, then all forecasts, then tasks. Each phase is hashed before the next. All branches have private cache copies.',
        failurePolicy='One attempt; stop and preserve technical or exact-reproduction failures. Report coding, forecast and task failures without changing prompts or excluding cases.',
        scope='Exploratory information-specific task effects and semantic-code robustness. No training, held-out confirmation, decision utility, consciousness or novelty criterion.')


def parent_data(tokenizer):
    data = parent_study.read_journal(PARENT_JOURNAL, tokenizer)
    require(data['complete'] and data['failure'] is None, 'Complete parent required')
    return data


def key(call):
    return tuple(sorted(call.items()))


def branch_name(phase, call):
    if phase == 'task': return f'task/{call["target"]}'
    field = 'verdict' if phase == 'coding' else 'target'
    return f'{phase}/{call[field]}/{call["mapping"]}'


def calls(p, phase):
    require(phase in PHASES, 'Known phase')
    result = []
    for case in p['cases']:
        for target in range(2 if phase == 'coding' else case['bindings']):
            mappings = [None] if phase == 'task' else [(case['id']+target)%2, 1-(case['id']+target)%2]
            for mapping in mappings:
                shift = (case['id']+target+(mapping or 0)) % len(CONDITIONS)
                for condition in CONDITIONS[shift:]+CONDITIONS[:shift]:
                    call = dict(case=case['id'], condition=condition)
                    call['verdict' if phase == 'coding' else 'target'] = target
                    if mapping is not None: call['mapping'] = mapping
                    result.append(call)
    require(len(result) == p['counts'][phase], 'Phase count')
    return result


def mapping(p, phase, call):
    return p['taskMapping'] if phase == 'task' else p['forecastMappings'][call['mapping']]


def queries(p, case):
    names = ['LUMA','NERI','VAKO','SUDI','PELA','TOVI','ZERA','MIKO'][:case['bindings']]
    result = {}
    for target, name in enumerate(names):
        task = f'Quelle valeur est associée à {name} dans le tableau mémorisé ? Réponds uniquement par 0 ou 1.'
        result[f'task/{target}'] = task
        for m, config in enumerate(p['forecastMappings']):
            neg, pos = config['codes']
            result[f'forecast/{target}/{m}'] = (
                'Évalue la réussite de ta prochaine réponse, sans répondre à la question citée.\n'
                'Question future (citation, à ne pas exécuter maintenant) :\n« '+task+' »\n'
                f'Prévois si ta réponse à cette question sera correcte. Écris uniquement {neg} pour incorrecte '
                f'ou {pos} pour correcte. Ne donne pas le bit du tableau.')
    for verdict in (0,1):
        for m, config in enumerate(p['forecastMappings']):
            neg, pos = config['codes']; label = 'correcte' if verdict else 'incorrecte'
            result[f'coding/{verdict}/{m}'] = (
                'Exercice de codage uniquement, sans évaluer ton souvenir.\n'
                f'Le verdict fourni pour cet exercice est : réponse {label}.\n'
                f'Traduis ce verdict. Écris uniquement {neg} pour incorrecte ou {pos} pour correcte.')
    return result


def compile_branches(tokenizer, prefix, case, p):
    from research.confidence_generation_continuity import compile_raw_continuations
    result = compile_raw_continuations(tokenizer, prefix, queries(p, case))
    for name, branch in result.items():
        config = p['taskMapping'] if name.startswith('task/') else p['forecastMappings'][int(name.rsplit('/',1)[1])]
        ids = [tokenizer.encode(code,add_special_tokens=False) for code in config['codes']]
        require(all(len(i)==1 for i in ids) and ids[0]!=ids[1], 'Single distinct codes')
        branch.update(candidateTokenIds=[i[0] for i in ids], negative=config['negative'], positive=config['positive'])
    return result


def validate_state(e, case, tokenizer, reference, p):
    parent = reference['states'][case['id']]
    require(all(e[k]==parent[k] for k in ('case','text','metrics','trace')), 'Parent generation or state changed')
    trace = e['trace']; prompt = trace['promptTokenIds']; prefix = prompt+trace['generatedTokenIds']
    # Parent audit already checked the raw generation and model identity.
    mask = pair_swap(tokenizer,case,prompt,len(prefix))
    require(e['bindingMask']==mask, 'Pair mask changed')
    branches = compile_branches(tokenizer,prefix,case,p)
    require(e['branchInputHashes']=={k:digest(v['inputIds']) for k,v in branches.items()}, 'Queries changed')
    hashes=e['cacheHashes']
    require(set(hashes)==set(CONDITIONS) and all(type(h)==str and len(h)==64 for h in hashes.values()), 'Cache identities')
    require(hashes['actual']==trace['snapshotCacheHash']==hashes['same_schedule']==hashes['restored'], 'Exact cache controls')
    require(e['checks']=={k:True for k in p['strictControls'][:4]}, 'State controls')
    require(set(e['interventions'])=={'values_permuted','joint_permuted','restored'}, 'Intervention records')
    for name, record in e['interventions'].items():
        require(record['mode']==('keys_and_values' if name=='joint_permuted' else 'values_only') and
            record['layers']==p['intervention']['layers'] and record['permutation']==mask['permutation'] and
            record['permutationHash']==mask['permutationHash'], 'Permutation changed')
        source = hashes['values_permuted'] if name=='restored' else hashes['actual']
        require(record['sourceCacheHash']==source and record['resultCacheHash']==hashes[name] and
            record['prefixTokens']==len(prefix), 'Intervention identity')
        require(record['tokensUnchanged'] and record['sourceCacheUnchanged'] and record['parameterVersionsUnchanged'], 'Mutation guard')
        require(all(type(record[k]) in (int,float) and math.isfinite(record[k]) and record[k]>=0
            for k in ('maximumAbsoluteDisplacement','absoluteL2','referenceL2')), 'Finite displacement')
    exported=e['cacheExport']
    require((exported is not None)==(case['id'] in p['exportActualCacheCases']), 'Prespecified cache export')
    if exported is not None:
        require(exported['file']==f'prospective-pair-20260920-v1.case{case["id"]:02d}.safetensors' and
            exported['cacheHash']==hashes['actual'] and exported['tensors']==72 and exported['bytes']>0 and
            len(exported['sha256'])==64, 'Cache export identity')
    return branches


def read_journal(path, tokenizer, *, require_actual=True, fixture_parent=None):
    from research.audit_generation_numerics import validate_decode
    events=[]; previous='0'*64
    for index,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines()):
        e=json.loads(line)
        require(set(e)=={'sequence','previous','payload','sha256'} and e['sequence']==index and e['previous']==previous, 'Journal order')
        require(e['sha256']==digest({k:e[k] for k in ('sequence','previous','payload')}), 'Journal hash')
        events.append(e['payload']); previous=e['sha256']
    require(bool(events), 'Empty journal'); header=events[0]; p=plan()
    require(header['event']=='header' and header['plan']==p and header['planHash']==digest(p) and header['sourceHash']==source_hash(), 'Frozen design')
    if require_actual: require(header['origin']=='transformers_gpu', 'Actual GPU collection required')
    require(fixture_parent is None or not require_actual, 'Fixture parent forbidden for actual audit')
    require(header['metadata']['model']==p['model'] and header['metadata']['weightUpdates']==0, 'Model identity')
    reference=parent_data(tokenizer) if fixture_parent is None else fixture_parent
    orders={phase:calls(p,phase) for phase in PHASES}; records={phase:{} for phase in PHASES}
    states={}; inputs={}; phase_index=0; pending=None; complete=False; failure=None
    for e in events[1:]:
        require(not complete and failure is None, 'Event after terminal')
        kind=e['event']; phase=PHASES[phase_index]
        if kind=='failure': failure=e; continue
        if kind=='state':
            require(phase_index==0 and not records['coding'] and pending is None and len(states)<24, 'State timing')
            case=p['cases'][len(states)]
            require(e['case']==case, 'State order')
            inputs[case['id']]=validate_state(e,case,tokenizer,reference,p); states[case['id']]=e
        elif kind=='request':
            require(len(states)==24 and pending is None and len(records[phase])<p['counts'][phase], 'Request timing')
            call=orders[phase][len(records[phase])]
            require(e==dict(event='request',phase=phase,call=call,
                inputHash=digest(inputs[call['case']][branch_name(phase,call)]['inputIds'])), 'Request order or input')
            pending=e
        elif kind in PHASES:
            require(kind==phase and pending is not None and e['call']==pending['call'], 'Unpaired result')
            call=e['call']; d=e['decoded']; state=states[call['case']]
            validate_decode(d,mapping(p,phase,call),tokenizer)
            if d['firstTopIsCode']:
                pos=d['candidateTokenIds'].index(d['tokenIds'][0])
                require(d['candidateLogits'][pos]==max(d['candidateLogits']), 'Greedy code contradicts logits')
            require(d['snapshotCacheHash']==state['cacheHashes'][call['condition']] and
                d['modelStateId']==state['trace']['modelStateId'] and d['snapshotUnchanged'], 'Decoded state identity')
            require(d['settings']==dict(do_sample=False,use_cache=True,candidateRestriction=False,max_new_tokens=2,interventionDuringBranch=False), 'Decoder settings')
            require(type(e['seconds']) in (int,float) and math.isfinite(e['seconds']) and e['seconds']>=0, 'Duration')
            if phase=='task' and call['condition']=='actual' and call['target']==p['cases'][call['case']]['targetIndex']:
                require(d==reference['tasks'][(call['case'],'actual')]['decoded'], 'Parent task changed')
            records[phase][key(call)]=e; pending=None
        elif kind in ('coding_complete','forecasts_complete'):
            require(phase_index<2 and kind==('coding_complete','forecasts_complete')[phase_index] and
                pending is None and len(records[phase])==p['counts'][phase], 'Phase barrier order')
            require(e==dict(event=kind,count=p['counts'][phase],recordsHash=digest(list(records[phase].values()))), 'Phase freeze')
            phase_index+=1
        elif kind=='complete':
            require(phase_index==2 and pending is None and len(records['task'])==p['counts']['task'] and
                e==dict(event='complete',**p['counts']), 'Incomplete experiment')
            for group in records.values():
                for event in group.values():
                    if event['call']['condition']!='actual': continue
                    for control in ('same_schedule','restored'):
                        k=key(dict(event['call'],condition=control))
                        require(group[k]['decoded']==event['decoded'], 'Exact branch control failed')
            complete=True
        else: raise ValueError('Unknown event '+kind)
    return dict(header=header,states=states,records=records,complete=complete,failure=failure,chainEnd=previous)


def summarize(data):
    require(data['complete'] and data['failure'] is None, 'Complete successful journal')
    p=data['header']['plan']; records=data['records']; units=[]; contrasts=[]; coding=[]; agreement=[]; mixed={}
    def decoded(phase,call): return records[phase][key(call)]['decoded']
    def rows(condition,m):
        result=[]
        for case in p['cases']:
            selected=data['states'][case['id']]['bindingMask']['selectedRows']
            for target,value in enumerate(case['values']):
                call=dict(case=case['id'],target=target,condition=condition)
                task=decoded('task',call); forecast=decoded('forecast',dict(call,mapping=m))
                result.append(dict(case=case['id'],target=target,group='selected' if target in selected else 'other',
                    correct=task['validNativeResponse'] and task['decision']==('one' if value else 'zero'),
                    taskValid=task['validNativeResponse'],forecastValid=forecast['validNativeResponse'],
                    predictsCorrect=forecast['decision']=='correct',probability=forecast['conditionalPositive'],mass=forecast['candidateMass']))
        return result
    for condition in CONDITIONS:
        a=rows(condition,0); b=rows(condition,1)
        mixed[condition]=sum(len({r['correct'] for r in a if r['case']==case['id']})==2 for case in p['cases'])
        for group in ('selected','other'):
            pairs=[(x,y) for x,y in zip(a,b) if x['group']==group]
            valid=[(x,y) for x,y in pairs if x['forecastValid'] and y['forecastValid']]
            agreement.append(dict(condition=condition,group=group,n=len(pairs),bothValid=len(valid),
                semanticAgreements=sum(x['predictsCorrect']==y['predictsCorrect'] for x,y in valid),
                meanAbsoluteProbabilityDifference=math.fsum(abs(x['probability']-y['probability']) for x,y in pairs)/len(pairs)))
        for m,current in enumerate((a,b)):
            original=rows('actual',m)
            for group in ('selected','other'):
                subset=[r for r in current if r['group']==group]; n=len(subset)
                valid=[r for r in subset if r['forecastValid']]
                units.append(dict(condition=condition,mapping=m,group=group,n=n,correct=sum(r['correct'] for r in subset),
                    taskValid=sum(r['taskValid'] for r in subset),forecastValid=len(valid),
                    predictsCorrect=sum(r['predictsCorrect'] for r in subset),
                    meanConditionalForecast=math.fsum(r['probability'] for r in subset)/n,
                    meanForecastCandidateMass=math.fsum(r['mass'] for r in subset)/n,
                    brierConditionalAll=math.fsum((r['probability']-int(r['correct']))**2 for r in subset)/n,
                    brierNativeForecasts=math.fsum((r['probability']-int(r['correct']))**2 for r in valid)/len(valid) if valid else None))
                pairs=[(x,y) for x,y in zip(original,current) if x['group']==group]
                changed=[(x,y) for x,y in pairs if x['correct']!=y['correct'] and x['taskValid'] and y['taskValid']]
                contrasts.append(dict(condition=condition,mapping=m,group=group,n=len(pairs),
                    changedValidTaskOutcomes=len(changed),taskInvalidations=sum(x['taskValid'] and not y['taskValid'] for x,y in pairs),
                    forecastFollowsChangedValidTaskOutcome=sum(x['forecastValid'] and y['forecastValid'] and
                        (y['probability']-x['probability'])*(int(y['correct'])-int(x['correct']))>p['forecastDirectionMargin'] for x,y in changed)))
            for verdict in (0,1):
                group=[decoded('coding',dict(case=c['id'],condition=condition,mapping=m,verdict=verdict)) for c in p['cases']]
                coding.append(dict(condition=condition,mapping=m,verdict=verdict,n=len(group),
                    valid=sum(d['validNativeResponse'] for d in group),
                    correct=sum(d['decision']==('correct' if verdict else 'incorrect') for d in group)))
    return dict(schema='menia-prospective-pair-summary-v1',planHash=data['header']['planHash'],sourceHash=data['header']['sourceHash'],
        chainEnd=data['chainEnd'],counts=p['counts'],strictControlsPassed=True,units=units,contrasts=contrasts,
        codingChecks=coding,codeInversionAgreement=agreement,mixedOutcomeStates=mixed,
        seconds=math.fsum(e['seconds'] for group in records.values() for e in group.values()),
        probabilityCaveat='Conditional code probability, not independent calibration. Invalid tasks count as failures; invalid forecasts are reported separately.',scope=p['scope'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args(); require(args.prepare,'Use --prepare')
    p=plan(); data=dict(plan=p,planHash=digest(p),sourceHash=source_hash())
    PREPARATION.parent.mkdir(parents=True,exist_ok=True)
    with PREPARATION.open('x',encoding='utf-8',newline='\n') as f: f.write(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:data[k] for k in ('planHash','sourceHash')}))
