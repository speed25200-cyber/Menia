"""Separate arithmetic from recorded tokens/logits; no collector metric imports."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path


def verify(journal, summary_path, tokenizer_receipt):
    events=[json.loads(line)['payload'] for line in Path(journal).read_text(encoding='utf-8').splitlines()]
    expected=json.loads(Path(summary_path).read_text(encoding='utf-8'))
    tokens=json.loads(Path(tokenizer_receipt).read_text(encoding='utf-8'))['codeTokenIds']
    p=events[0]['plan']; states={e['case']['id']:e for e in events if e['event']=='state'}
    assert len(states)==24 and events[-1]['event']=='complete'
    records={kind:[] for kind in ('coding','forecast','task')}
    with localcontext() as context:
        context.prec=60
        for event in events:
            kind=event['event']
            if kind not in records: continue
            call=event['call']; d=event['decoded']
            config=p['taskMapping'] if kind=='task' else p['forecastMappings'][call['mapping']]
            ids=[tokens[c][0] for c in config['codes']]
            assert d['candidateTokenIds']==ids and all(len(tokens[c])==1 for c in config['codes'])
            native=d['tokenIds']; valid=len(native)==2 and native[0] in ids and native[1] in d['eosTokenIds']
            bit=ids.index(native[0]) if valid else None
            logits=[Decimal(str(v)) for v in d['candidateLogits']]
            probability=float(1/(1+(logits[0]-logits[1]).exp()))
            assert abs(probability-d['conditionalPositive'])<1e-12
            records[kind].append(dict(call=call,valid=valid,bit=bit,probability=probability,
                                      mass=d['candidateMass'],seconds=event['seconds']))
    assert {k:len(v) for k,v in records.items()}=={k:p['counts'][k] for k in records}
    tasks={(r['call']['case'],r['call']['target'],r['call']['condition']):r for r in records['task']}
    observations=[]
    for f in records['forecast']:
        c=f['call']; state=states[c['case']]; target=c['target']
        task=tasks[c['case'],target,c['condition']]
        observations.append(dict(case=c['case'],target=target,condition=c['condition'],mapping=c['mapping'],
            group='selected' if target in state['bindingMask']['selectedRows'] else 'other',
            correct=task['valid'] and task['bit']==state['case']['values'][target],taskValid=task['valid'],
            forecastValid=f['valid'],predictsCorrect=f['valid'] and f['bit']==1,
            probability=f['probability'],mass=f['mass']))
    lookup={(r['case'],r['target'],r['condition'],r['mapping']):r for r in observations}
    def mean(values):
        values=list(values)
        return float(sum(Decimal(str(v)) for v in values)/len(values)) if values else None
    def brier(rows):
        return float(sum((Decimal(str(r['probability']))-int(r['correct']))**2 for r in rows)/len(rows)) if rows else None
    units=[]; contrasts=[]; coding=[]; agreement=[]; mixed={}
    for condition in p['conditions']:
        mixed[condition]=sum(len({r['correct'] for r in observations if r['case']==case and r['condition']==condition and r['mapping']==0})==2 for case in states)
        for group in ('selected','other'):
            pairs=[(r,lookup[r['case'],r['target'],condition,1]) for r in observations
                   if r['condition']==condition and r['mapping']==0 and r['group']==group]
            valid=[(a,b) for a,b in pairs if a['forecastValid'] and b['forecastValid']]
            agreement.append(dict(condition=condition,group=group,n=len(pairs),bothValid=len(valid),
                semanticAgreements=sum(a['predictsCorrect']==b['predictsCorrect'] for a,b in valid),
                meanAbsoluteProbabilityDifference=mean(abs(a['probability']-b['probability']) for a,b in pairs)))
        for m in (0,1):
            for group in ('selected','other'):
                selected=[r for r in observations if (r['condition'],r['mapping'],r['group'])==(condition,m,group)]
                valid=[r for r in selected if r['forecastValid']]
                units.append(dict(condition=condition,mapping=m,group=group,n=len(selected),
                    correct=sum(r['correct'] for r in selected),taskValid=sum(r['taskValid'] for r in selected),
                    forecastValid=len(valid),predictsCorrect=sum(r['predictsCorrect'] for r in selected),
                    meanConditionalForecast=mean(r['probability'] for r in selected),
                    meanForecastCandidateMass=mean(r['mass'] for r in selected),
                    brierConditionalAll=brier(selected),brierNativeForecasts=brier(valid)))
                paired=[(lookup[r['case'],r['target'],'actual',m],r) for r in selected]
                changed=[(a,b) for a,b in paired if a['correct']!=b['correct'] and a['taskValid'] and b['taskValid']]
                contrasts.append(dict(condition=condition,mapping=m,group=group,n=len(paired),
                    changedValidTaskOutcomes=len(changed),taskInvalidations=sum(a['taskValid'] and not b['taskValid'] for a,b in paired),
                    forecastFollowsChangedValidTaskOutcome=sum(a['forecastValid'] and b['forecastValid'] and
                        (b['probability']-a['probability'])*(int(b['correct'])-int(a['correct']))>p['forecastDirectionMargin'] for a,b in changed)))
            for verdict in (0,1):
                rows=[r for r in records['coding'] if (r['call']['condition'],r['call']['mapping'],r['call']['verdict'])==(condition,m,verdict)]
                coding.append(dict(condition=condition,mapping=m,verdict=verdict,n=len(rows),
                    valid=sum(r['valid'] for r in rows),correct=sum(r['valid'] and r['bit']==verdict for r in rows)))
    calculated=dict(units=units,contrasts=contrasts,codingChecks=coding,codeInversionAgreement=agreement,mixedOutcomeStates=mixed,
        seconds=math.fsum(r['seconds'] for rows in records.values() for r in rows))
    differences=[]
    def compare(a,b,path):
        if isinstance(a,dict):
            assert a.keys()==b.keys(),path
            for k in a: compare(a[k],b[k],path+'/'+k)
        elif isinstance(a,list):
            assert len(a)==len(b),path
            for i,(x,y) in enumerate(zip(a,b)): compare(x,y,path+'/'+str(i))
        elif type(a) is float:
            delta=abs(a-b); assert delta<=1e-12,(path,a,b); differences.append(delta)
        else: assert a==b,(path,a,b)
    for field,value in calculated.items(): compare(value,expected[field],field)
    return dict(schema='menia-pair-separate-arithmetic-v1',verified=True,
        journalSHA256=hashlib.sha256(Path(journal).read_bytes()).hexdigest(),
        summarySHA256=hashlib.sha256(Path(summary_path).read_bytes()).hexdigest(),
        auditorSourceHash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        branchRecords=sum(len(r) for r in records.values()),numericComparisons=len(differences),
        maximumAbsoluteDifference=max(differences),mixedOutcomeStates=mixed,
        scope='Separate decimal probability and score calculations on the same journal. No independent collection or model-weight attestation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('journal',type=Path)
    parser.add_argument('--summary',type=Path,required=True); parser.add_argument('--tokenizer-receipt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    result=verify(args.journal,args.summary,args.tokenizer_receipt)
    with args.output.open('x',encoding='utf-8',newline='\n') as f: f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result))
