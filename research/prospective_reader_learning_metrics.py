"""Prespecified paired losses and within-table discrimination, including failures."""
from collections import defaultdict
import math

import numpy as np

from research import prospective_reader_learning as study


def penalized_loss(probability,label,valid):
    assert math.isfinite(probability) and 0<=probability<=1 and label in (0,1)
    return (probability-label)**2 if valid else 1.0


def table_auc(rows):
    positive=[r for r in rows if r['label']];negative=[r for r in rows if not r['label']]
    if not positive or not negative:return None
    if not all(r['valid'] for r in rows):return 0.0
    return math.fsum(float(a['probability']>b['probability'])+.5*(a['probability']==b['probability'])
        for a in positive for b in negative)/(len(positive)*len(negative))


def interval(values,config):
    assert values and all(math.isfinite(x) for x in values)
    values=np.asarray(values,dtype=np.float64)
    # Resample whole tables. A fresh fixed generator gives paired endpoints the
    # same draws when their set of eligible tables is the same.
    rng=np.random.default_rng(config['seed'])
    boot=values[rng.integers(0,len(values),(config['resamples'],len(values)))].mean(axis=1)
    tail=config['familyAlpha']/(2*config['comparisons'])
    low,high=np.quantile(boot,[tail,1-tail],method='linear')
    return dict(tables=len(values),mean=math.fsum(values)/len(values),low=float(low),high=float(high),
        coverage=1-2*tail,seed=config['seed'],resamples=config['resamples'])


def summarize(data):
    assert data['complete'] and data['failure'] is None
    p=data['header']['plan'];events=data['events'];case_map={c['id']:c for c in p['cases']}
    tasks={(e['call']['case'],e['call']['target'],e['call']['condition']):e['result']
        for e in events if e['event']=='task'}
    rows=[];coding=[]
    for e in events:
        kind=e['event']
        if kind not in ('forecast','coding'):continue
        call=e['call'];d=e['result']['decoded']
        label=study.target_label(kind,call,tasks,case_map)
        row=dict(call,label=label,valid=d['validNativeResponse'],probability=d['conditionalPositive'],
            correct=d['validNativeResponse'] and d['decision']==('correct' if label else 'incorrect'),
            candidateMass=d['candidateMass'])
        if kind=='coding':coding.append(row);continue
        row.update(taskValid=tasks[call['case'],call['target'],call['condition']]['decoded']['validNativeResponse'],
            group='selected' if call['target'] in data['states'][call['case']]['bindingMask']['selectedRows'] else 'other',
            loss=penalized_loss(row['probability'],label,row['valid']))
        rows.append(row)
    units=[];coding_units=[]
    groups=defaultdict(list);codegroups=defaultdict(list)
    for r in rows:
        for group in ('all',r['group']):groups[r['split'],r['arm'],r['condition'],r['mapping'],group].append(r)
    for r in coding:codegroups[r['split'],r['arm'],r['condition'],r['mapping'],r['verdict']].append(r)
    for (split,arm,condition,mapping,group),subset in sorted(groups.items()):
        n=len(subset);valid=[r for r in subset if r['valid']]
        units.append(dict(split=split,arm=arm,condition=condition,mapping=mapping,group=group,n=n,
            successes=sum(r['label'] for r in subset),taskValid=sum(r['taskValid'] for r in subset),
            valid=len(valid),semanticCorrect=sum(r['correct'] for r in subset),
            penalizedLoss=math.fsum(r['loss'] for r in subset)/n,
            conditionalBrierAll=math.fsum((r['probability']-r['label'])**2 for r in subset)/n,
            conditionalBrierValid=math.fsum((r['probability']-r['label'])**2 for r in valid)/len(valid) if valid else None,
            meanProbability=math.fsum(r['probability'] for r in subset)/n,
            meanCandidateMass=math.fsum(r['candidateMass'] for r in subset)/n))
    for (split,arm,condition,mapping,verdict),subset in sorted(codegroups.items()):
        coding_units.append(dict(split=split,arm=arm,condition=condition,mapping=mapping,verdict=verdict,
            n=len(subset),valid=sum(r['valid'] for r in subset),correct=sum(r['correct'] for r in subset)))
    cfg=p['primary'];primary=[];gates=[];discrimination=[]
    for mapping in cfg['mappings']:
        subsets={arm:groups['reserved',arm,cfg['condition'],mapping,'all'] for arm in study.ARMS}
        tables={arm:{cid:[r for r in subsets[arm] if r['case']==cid] for cid in sorted({r['case'] for r in subsets[arm]})}
            for arm in study.ARMS}
        for comparator in cfg['comparators']:
            gains=[]
            for cid in tables['state']:
                a=sorted(tables['state'][cid],key=lambda r:r['target']);b=sorted(tables[comparator][cid],key=lambda r:r['target'])
                assert [(r['target'],r['label']) for r in a]==[(r['target'],r['label']) for r in b]
                gains.append(math.fsum(y['loss']-x['loss'] for x,y in zip(a,b))/len(a))
            ci=interval(gains,cfg);passed=ci['mean']>=cfg['minimumMeanGain'] and ci['low']>0
            primary.append(dict(mapping=mapping,comparator=comparator,metric='penalizedLossGain',interval=ci,passed=passed))
        eligible=[cid for cid in tables['state'] if table_auc(tables['state'][cid]) is not None]
        aucs={arm:[table_auc(tables[arm][cid]) for cid in eligible] for arm in study.ARMS}
        gains=[a-b for a,b in zip(aucs['state'],aucs[cfg['discriminationComparator']])]
        ci=interval(gains,cfg) if gains else None
        means={arm:math.fsum(v)/len(v) if v else None for arm,v in aucs.items()}
        passed=bool(ci and len(eligible)>=cfg['minimumMixedTables'] and means['state']>=cfg['minimumStateAUROC']
            and ci['mean']>=cfg['minimumAUROCGain'] and ci['low']>0)
        discrimination.append(dict(mapping=mapping,comparator=cfg['discriminationComparator'],
            mixedTableIds=eligible,meanTableAUROC=means,interval=ci,passed=passed))
        current=subsets['state'];n=len(current);successes=sum(r['label'] for r in current)
        supplied=[r for r in coding if r['split']=='reserved' and r['arm']=='state' and
            r['condition']==cfg['condition'] and r['mapping']==mapping]
        gates.append(dict(mapping=mapping,successes=successes,failures=n-successes,
            classesPass=successes>=cfg['minimumPositiveTasks'] and n-successes>=cfg['minimumNegativeTasks'],
            nativeTaskFraction=sum(r['taskValid'] for r in current)/n,
            nativeTaskPass=sum(r['taskValid'] for r in current)/n>=cfg['minimumNativeTaskFraction'],
            nativeForecastFraction=sum(r['valid'] for r in current)/n,
            nativeForecastPass=sum(r['valid'] for r in current)/n>=cfg['minimumNativeFraction'],
            suppliedCodingFraction=sum(r['correct'] for r in supplied)/len(supplied),
            suppliedCodingPass=sum(r['correct'] for r in supplied)/len(supplied)>=cfg['minimumSuppliedCodingFraction']))
    # This constant uses training task outcomes only and is diagnostic, never tuned
    # against reserved labels or substituted for the frozen comparison arms.
    training=[r for r in rows if r['split']=='train' and r['arm']=='state' and r['mapping']==0]
    prior=math.fsum(r['label'] for r in training)/len(training)
    heldout=subsets['state']
    baseline=dict(trainingVOnlyPrevalence=prior,reservedVOnlyLoss=math.fsum((prior-r['label'])**2 for r in heldout)/len(heldout))
    passed=all(x['passed'] for x in primary+discrimination) and all(all(g[k] for k in
        ('classesPass','nativeTaskPass','nativeForecastPass','suppliedCodingPass')) for g in gates)
    steps=[e for e in events if e['event']=='step']
    learning=[dict(arm=arm,epoch=epoch,updates=len(s),examples=sum(len(e['result']['examples']) for e in s),
        meanTeacherForcedLoss=math.fsum(e['result']['meanLoss'] for e in s)/len(s))
        for arm in ('state','text') for epoch in range(p['optimizer']['epochs'])
        for s in [[e for e in steps if e['call']['arm']==arm and e['call']['epoch']==epoch]]]
    return dict(schema='menia-prospective-reader-learning-summary-v1',planHash=data['header']['planHash'],
        sourceHash=data['header']['sourceHash'],chainEnd=data['chainEnd'],counts=p['counts'],units=units,coding=coding_units,
        primaryLossComparisons=primary,primaryDiscriminationComparisons=discrimination,gates=gates,
        allFunctionalCriteriaPassed=passed,constantDiagnostic=baseline,learning=learning,
        seconds=math.fsum(e.get('seconds',0) for e in events),
        caveat='Intervals resample reserved tables conditional on one fitted optimization seed and this fixed-key synthetic task. Loss includes native-format penalties. AUROC requires both task classes and penalizes any invalid table forecast with zero. No independent calibration or consciousness test.',scope=p['scope'])
