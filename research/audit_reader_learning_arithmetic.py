"""Second implementation of reader-learning scores; imports no collector metrics.

Probabilities and loss sums use Decimal; AUROC uses midranks rather than pair
counting. Bootstrap draws the same fixed table indices with a separate loop,
and computes percentiles by explicit sorted interpolation. This is a check of
arithmetic on one journal, not an independent experiment or model attestation.
"""
import argparse
from collections import defaultdict
from decimal import Decimal,localcontext
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def decimal_mean(values):
    values=list(values)
    return float(sum(Decimal(str(x)) for x in values)/len(values)) if values else None


def ranks_auc(rows):
    positives=sum(r['label'] for r in rows);negatives=len(rows)-positives
    if not positives or not negatives:return None
    if any(not r['valid'] for r in rows):return 0.0
    ordered=sorted(rows,key=lambda r:r['probability']);rank_sum=0.;start=0
    while start<len(ordered):
        end=start+1
        while end<len(ordered) and ordered[end]['probability']==ordered[start]['probability']:end+=1
        rank_sum+=(start+1+end)/2*sum(r['label'] for r in ordered[start:end]);start=end
    return (rank_sum-positives*(positives+1)/2)/(positives*negatives)


def bootstrap(values,cfg):
    rng=np.random.default_rng(cfg['seed']);n=len(values);draws=[]
    for _ in range(cfg['resamples']):
        draws.append(math.fsum(values[i] for i in rng.integers(n,size=n))/n)
    draws.sort();tail=cfg['familyAlpha']/(2*cfg['comparisons'])
    def quantile(q):
        position=(len(draws)-1)*q;left=int(position);fraction=position-left
        return draws[left]+fraction*(draws[min(left+1,len(draws)-1)]-draws[left])
    return dict(tables=n,mean=decimal_mean(values),low=quantile(tail),high=quantile(1-tail),coverage=1-2*tail,
        seed=cfg['seed'],resamples=cfg['resamples'])


def branch(d,config,tokens):
    codes=[tokens[c][0] for c in config['codes']]
    assert all(len(tokens[c])==1 for c in config['codes']) and d['candidateTokenIds']==codes
    emitted=d['tokenIds'];valid=len(emitted)==2 and emitted[0] in codes and emitted[1] in d['eosTokenIds']
    semantic_bit=codes.index(emitted[0]) if valid else None
    with localcontext() as context:
        context.prec=70
        logit0,logit1=(Decimal(str(v)) for v in d['candidateLogits'])
        probability=float(1/(1+(logit0-logit1).exp()))
    assert abs(probability-d['conditionalPositive'])<=1e-12
    return dict(valid=valid,semanticBit=semantic_bit,probability=probability,candidateMass=d['candidateMass'])


def calculate(events,tokens):
    p=events[0]['plan'];cases={c['id']:c for c in p['cases']}
    states={e['result']['case']['id']:e['result'] for e in events if e['event']=='state'}
    task={};reports=[];coding=[]
    for e in events:
        if e['event']=='task':
            c=e['call'];task[c['case'],c['target'],c['condition']]=branch(e['result']['decoded'],p['taskMapping'],tokens)
    for e in events:
        if e['event'] not in ('forecast','coding'):continue
        c=e['call'];d=branch(e['result']['decoded'],p['forecastMappings'][c['mapping']],tokens)
        r=dict(c,**d)
        if e['event']=='coding':
            r['correct']=d['valid'] and d['semanticBit']==c['verdict'];coding.append(r);continue
        actual=task[c['case'],c['target'],c['condition']]
        r.update(label=int(actual['valid'] and actual['semanticBit']==cases[c['case']]['values'][c['target']]),
            taskValid=actual['valid'],group='selected' if c['target'] in states[c['case']]['bindingMask']['selectedRows'] else 'other')
        r['correct']=r['valid'] and r['semanticBit']==r['label']
        with localcontext() as context:
            context.prec=70;r['brier']=float((Decimal(str(r['probability']))-r['label'])**2)
        r['loss']=r['brier'] if r['valid'] else 1.
        reports.append(r)
    groups=defaultdict(list);codegroups=defaultdict(list)
    for r in reports:
        for group in ('all',r['group']):groups[r['split'],r['arm'],r['condition'],r['mapping'],group].append(r)
    for r in coding:codegroups[r['split'],r['arm'],r['condition'],r['mapping'],r['verdict']].append(r)
    units=[];codeunits=[]
    for (split,arm,condition,mapping,group),rows in sorted(groups.items()):
        valid=[r for r in rows if r['valid']]
        units.append(dict(split=split,arm=arm,condition=condition,mapping=mapping,group=group,n=len(rows),
            successes=sum(r['label'] for r in rows),taskValid=sum(r['taskValid'] for r in rows),valid=len(valid),
            semanticCorrect=sum(r['correct'] for r in rows),penalizedLoss=decimal_mean(r['loss'] for r in rows),
            conditionalBrierAll=decimal_mean(r['brier'] for r in rows),conditionalBrierValid=decimal_mean(r['brier'] for r in valid),
            meanProbability=decimal_mean(r['probability'] for r in rows),meanCandidateMass=decimal_mean(r['candidateMass'] for r in rows)))
    for (split,arm,condition,mapping,verdict),rows in sorted(codegroups.items()):
        codeunits.append(dict(split=split,arm=arm,condition=condition,mapping=mapping,verdict=verdict,n=len(rows),
            valid=sum(r['valid'] for r in rows),correct=sum(r['correct'] for r in rows)))
    primary=[];rank_comparisons=[];gates=[];cfg=p['primary']
    reserved=sorted(c['id'] for c in p['cases'] if c['split']=='reserved')
    for mapping in cfg['mappings']:
        lookup={(r['arm'],r['case'],r['target']):r for r in reports if
            r['split']=='reserved' and r['mapping']==mapping and r['condition']==cfg['condition']}
        table={arm:{cid:[lookup[arm,cid,i] for i in range(cases[cid]['bindings'])] for cid in reserved} for arm in p['arms']}
        for comparator in cfg['comparators']:
            gains=[decimal_mean(lookup[comparator,cid,i]['loss']-lookup['state',cid,i]['loss']
                for i in range(cases[cid]['bindings'])) for cid in reserved]
            ci=bootstrap(gains,cfg)
            primary.append(dict(mapping=mapping,comparator=comparator,metric='penalizedLossGain',interval=ci,
                passed=ci['mean']>=cfg['minimumMeanGain'] and ci['low']>0))
        eligible=[cid for cid in reserved if ranks_auc(table['state'][cid]) is not None]
        scores={arm:[ranks_auc(table[arm][cid]) for cid in eligible] for arm in p['arms']}
        means={arm:decimal_mean(v) for arm,v in scores.items()}
        gains=[a-b for a,b in zip(scores['state'],scores[cfg['discriminationComparator']])]
        ci=bootstrap(gains,cfg) if gains else None
        rank_comparisons.append(dict(mapping=mapping,comparator=cfg['discriminationComparator'],mixedTableIds=eligible,
            meanTableAUROC=means,interval=ci,passed=bool(ci and len(eligible)>=cfg['minimumMixedTables'] and
                means['state']>=cfg['minimumStateAUROC'] and ci['mean']>=cfg['minimumAUROCGain'] and ci['low']>0)))
        rows=[r for cid in reserved for r in table['state'][cid]];n=len(rows);successes=sum(r['label'] for r in rows)
        supplied=[r for r in coding if r['split']=='reserved' and r['arm']=='state' and
            r['mapping']==mapping and r['condition']==cfg['condition']]
        native_task=sum(r['taskValid'] for r in rows)/n;native_forecast=sum(r['valid'] for r in rows)/n
        good_coding=sum(r['correct'] for r in supplied)/len(supplied)
        gates.append(dict(mapping=mapping,successes=successes,failures=n-successes,
            classesPass=successes>=cfg['minimumPositiveTasks'] and n-successes>=cfg['minimumNegativeTasks'],
            nativeTaskFraction=native_task,nativeTaskPass=native_task>=cfg['minimumNativeTaskFraction'],
            nativeForecastFraction=native_forecast,nativeForecastPass=native_forecast>=cfg['minimumNativeFraction'],
            suppliedCodingFraction=good_coding,suppliedCodingPass=good_coding>=cfg['minimumSuppliedCodingFraction']))
    train_tasks=[r for (cid,target,condition),r in task.items() if cases[cid]['split']=='train' and condition=='values_permuted']
    train_outcomes=[int(r['valid'] and r['semanticBit']==cases[cid]['values'][target]) for (cid,target,condition),r in task.items()
        if cases[cid]['split']=='train' and condition=='values_permuted']
    assert len(train_tasks)==len(train_outcomes)
    prevalence=decimal_mean(train_outcomes)
    reserved_outcomes=[int(r['valid'] and r['semanticBit']==cases[cid]['values'][target]) for (cid,target,condition),r in task.items()
        if cases[cid]['split']=='reserved' and condition=='values_permuted']
    baseline=dict(trainingVOnlyPrevalence=prevalence,reservedVOnlyLoss=decimal_mean((prevalence-y)**2 for y in reserved_outcomes))
    learning=[]
    for arm in ('state','text'):
        for epoch in range(p['optimizer']['epochs']):
            steps=[e for e in events if e['event']=='step' and e['call']['arm']==arm and e['call']['epoch']==epoch]
            learning.append(dict(arm=arm,epoch=epoch,updates=len(steps),examples=sum(len(e['result']['examples']) for e in steps),
                meanTeacherForcedLoss=decimal_mean(e['result']['meanLoss'] for e in steps)))
    passed=all(r['passed'] for r in primary+rank_comparisons) and all(g[k] for g in gates for k in
        ('classesPass','nativeTaskPass','nativeForecastPass','suppliedCodingPass'))
    return dict(units=units,coding=codeunits,primaryLossComparisons=primary,primaryDiscriminationComparisons=rank_comparisons,
        gates=gates,allFunctionalCriteriaPassed=passed,constantDiagnostic=baseline,learning=learning,
        seconds=math.fsum(e.get('seconds',0) for e in events))


def compare_tree(actual,expected,path='root',differences=None):
    if differences is None:differences=[]
    if isinstance(actual,dict):
        assert actual.keys()==expected.keys(),path
        for k in actual:compare_tree(actual[k],expected[k],path+'/'+k,differences)
    elif isinstance(actual,list):
        assert len(actual)==len(expected),path
        for i,(a,b) in enumerate(zip(actual,expected)):compare_tree(a,b,path+'/'+str(i),differences)
    elif type(actual) is float:
        delta=abs(actual-expected);assert math.isfinite(delta) and delta<=1e-12,(path,actual,expected);differences.append(delta)
    else:assert actual==expected,(path,actual,expected)
    return differences


def verify(journal,summary_path,tokenizer_receipt):
    journal=Path(journal);summary_path=Path(summary_path);receipt=json.loads(Path(tokenizer_receipt).read_text(encoding='utf-8'))
    events=[json.loads(line)['payload'] for line in journal.read_text(encoding='utf-8').splitlines()]
    assert events[0]['origin']=='transformers_gpu' and events[-1]['event']=='complete'
    assert events[0]['plan']['model']==receipt['model']
    expected=json.loads(summary_path.read_text(encoding='utf-8'));computed=calculate(events,receipt['codeTokenIds'])
    differences=compare_tree(computed,{k:expected[k] for k in computed})
    return dict(schema='menia-reader-learning-separate-arithmetic-v1',verified=True,
        journalSHA256=hashlib.sha256(journal.read_bytes()).hexdigest(),summarySHA256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        auditorSourceHash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numericComparisons=len(differences),maximumAbsoluteDifference=max(differences),
        primaryLossComparisons=len(computed['primaryLossComparisons']),primaryRankComparisons=len(computed['primaryDiscriminationComparisons']),
        allFunctionalCriteriaPassed=computed['allFunctionalCriteriaPassed'],
        scope='Independent implementation of token interpretation, Decimal probabilities/losses, midrank AUROC and bootstrap interpolation. Same journal, tokenizer receipt and fixed NumPy RNG; no new model collection or optimizer replay. Candidate mass is read from the journal, not reconstructed from full vocabulary logits.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('journal',type=Path)
    parser.add_argument('--summary',type=Path,required=True);parser.add_argument('--tokenizer-receipt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=verify(args.journal,args.summary,args.tokenizer_receipt)
    with args.output.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))
