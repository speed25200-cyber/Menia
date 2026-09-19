"""Separate grading, scalar outcome accounting and paired bootstrap arithmetic."""
from collections import Counter
import hashlib
import itertools
from math import fsum
from pathlib import Path
import re

import numpy as np

from research.audit_composition_diagnostic import compare
from research.audit_natural_errors import quantiles
from research.cross_model_prediction import CELLS
from research.iphone_coupling_report import require
from research.native_choice_journal import read_journal
from research.native_choice_plan import SEED


def correct(text, task):
    expected = sum(x=='A' for x in task['letters']) if task['family']=='countA' else sum((-1)**i*x for i,x in enumerate(task['operands']))
    text=text.strip()
    return bool(re.fullmatch(r'-?(?:0|[1-9][0-9]*)',text) and int(text)==expected)


def verify(path, report):
    data=read_journal(path)
    plan, results=data['header']['plan'],data['results']
    require(report['schema']=='menia-native-choice-report-v1' and report['complete'] and len(results)==plan['plannedRecords'] and data['pending'] is None and not data['failed'],'Incomplete audit input')
    require(all(report[k]==data['header'][k] for k in ('origin','sourceHash','planHash')),'Provenance')
    baseline={c['task']:r for c,r in zip(plan['calls'],results) if c['stage']=='baseline'}
    outputs={c['choice']:(req,r) for c,req,r in zip(plan['calls'],data['requests'],results) if c['stage']=='execute'}
    public={}
    for w,m in itertools.product(range(2),repeat=2):
        selected=[(c,r) for c,r in zip(plan['calls'],results) if c['stage']=='publicRisk' and (c['wording'],c['mapping'])==(w,m)]
        chosen=[]
        losses=[]
        best=[]
        good=0
        for c,r in selected:
            label=r['text'].strip()
            a='direct' if label==str(1+m) else 'verify' if label==str(2-m) else 'invalid'
            chosen.append(a)
            correct_action='direct' if 1-c['p']<=c['cost'] else 'verify'
            good+=a==correct_action
            losses.append(1-c['p'] if a=='direct' else c['cost'] if a=='verify' else 1.)
            best.append(min(1-c['p'],c['cost']))
        public[f'w{w}/m{m}']=dict(n=len(selected),optimalChoices=good,accuracy=good/len(selected),actions=dict(Counter(chosen)),
            meanExpectedRegret=fsum(a-b for a,b in zip(losses,best))/len(selected),passed=good/len(selected)>=plan['publicRiskMinimumAccuracy'])
    grouped={}
    for call,result in zip(plan['calls'],results):
        if call['stage']!='choice':continue
        task=plan['tasks'][call['task']]
        req,out=outputs[call['id']]
        chosen={'llm':'direct','exact_tool':'verify','invalid_choice':'invalid'}[req['engine']]
        y=correct(baseline[task['id']]['text'],task)
        delivered=correct(out['text'],task)
        key=f"{call['mode']}/{call['cost']:.1f}/w{call['wording']}/m{call['mapping']}"
        p=plan['calibration']['counts'][f"{task['family']}/{task['level']}"]['p']
        row=dict(id=task['id'],a=chosen,loss=call['cost'] if chosen=='verify' else 1-int(delivered) if chosen=='direct' else 1.,
            common=call['cost'] if chosen=='verify' else 1-int(y) if chosen=='direct' else 1.,
            beta=1-int(y) if 1-p<=call['cost'] else call['cost'],y=y,z=delivered,
            dt=result['seconds'],et=out['seconds'],dn=result['metrics']['outputTokens'],en=out['metrics'].get('outputTokens',0))
        grouped.setdefault(key,[]).append(row)
    native={}
    for key,rows in grouped.items():
        rows.sort(key=lambda r:r['id'])
        n=len(rows)
        native[key]=dict(n=n,actions=dict(Counter(r['a'] for r in rows)),directCorrect=sum(r['a']=='direct' and r['z'] for r in rows),
            deliveredCorrect=sum(r['z'] for r in rows),baselineCorrect=sum(r['y'] for r in rows),
            meanActualLoss=fsum(r['loss'] for r in rows)/n,meanCommonAnswerLoss=fsum(r['common'] for r in rows)/n,
            meanBetaCommonAnswerLoss=fsum(r['beta'] for r in rows)/n,alwaysDirectReferenceLoss=fsum(1-int(r['y']) for r in rows)/n,
            alwaysVerifyReferenceLoss=float(key.split('/')[1]),decisionSeconds=fsum(r['dt'] for r in rows),
            executionSeconds=fsum(r['et'] for r in rows),decisionTokens=sum(r['dn'] for r in rows),executionTokens=sum(r['en'] for r in rows))
    rng=np.random.default_rng(SEED+701)
    groups=[[t['id'] for t in plan['tasks'] if (t['family'],t['level'])==c] for c in CELLS]
    draws=np.concatenate([np.array(g)[rng.integers(0,len(g),size=(plan['bootstrapResamples'],len(g)))] for g in groups],axis=1)
    tail=plan['familyAlpha']/(2*plan['primaryComparisons'])
    contrasts=[]
    for cost,w,m in itertools.product((.2,.5,.8),range(2),range(2)):
        own=grouped[f'unassisted/{cost:.1f}/w{w}/m{m}']
        supplied=grouped[f'history/{cost:.1f}/w{w}/m{m}']
        d=[a['loss']-b['loss'] for a,b in zip(own,supplied)]
        boot=np.asarray(d)[draws].sum(1)/len(d)
        ci=quantiles(boot,[tail,1-tail])
        gain=fsum(d)/len(d)
        contrasts.append(dict(cost=cost,wording=w,mapping=m,unassistedMinusHistoryLoss=gain,
            interval95=quantiles(boot,[.025,.975]),familyInterval=ci,passed=bool(gain>=plan['minimumLossReduction'] and ci[0]>0)))
    transitions={}
    for mode,w,m in itertools.product(('unassisted','history'),range(2),range(2)):
        low=grouped[f'{mode}/0.2/w{w}/m{m}']
        high=grouped[f'{mode}/0.8/w{w}/m{m}']
        transitions[f'{mode}/w{w}/m{m}']=dict(Counter(a['a']+'->'+b['a'] for a,b in zip(low,high)))
    cells={}
    for family,level in CELLS:
        tasks=[t for t in plan['tasks'] if (t['family'],t['level'])==(family,level)]
        cells[f'{family}/{level}']=dict(n=len(tasks),correct=sum(correct(baseline[t['id']]['text'],t) for t in tasks))
    risk_gate=all(r['passed'] for r in public.values())
    expected=dict(recorded=len(results),planned=len(plan['calls']),complete=True,statuses={'ok':len(results)},
        actualLLMCalls=sum(r['engine']=='llm' for r in results),actualToolCalls=sum(r['engine']=='exact_tool' for r in results),
        invalidExecutionSlots=sum(r['engine']=='invalid_choice' for r in results),journalChainEnd=data['chainEnd'],
        publicRisk=public,publicRiskCriterion=risk_gate,native=native,primaryContrasts=contrasts,
        historyBenefitCriterion=bool(risk_gate and all(c['passed'] for c in contrasts)),costTransitions=transitions,
        baselineByCell=cells,baselineSeconds=fsum(r['seconds'] for r in baseline.values()),baselineOutputTokens=sum(r['metrics']['outputTokens'] for r in baseline.values()))
    delta=compare(expected,{k:report[k] for k in expected})
    return dict(schema='menia-native-choice-verification-v1',verified=True,origin=report['origin'],
        planHash=report['planHash'],sourceHash=report['sourceHash'],journalSHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        recorded=len(results),publicRiskTables=len(public),nativeTables=len(native),primaryContrasts=len(contrasts),maxAbsoluteDifference=delta,
        scope='Separate exact grading, scalar loss/latency accounting, native actions inferred from executed engine, paired bootstrap and manual quantiles. Shared plan and strict journal reader. Not an external collector replication.')
