"""Predeclared native choice outcomes; paired inference uses questions as units."""
from collections import Counter
import itertools

import numpy as np

from research.cross_model_prediction import CELLS, grade
from research.iphone_coupling_report import require
from research.native_choice_plan import COSTS, SEED, action, optimal
from research.native_choice_journal import read_journal


def observations(data):
    plan, results = data['header']['plan'], data['results']
    executed = {c['choice']:r for c,r in zip(plan['calls'],results) if c['stage']=='execute'}
    baseline = {c['task']:r for c,r in zip(plan['calls'],results) if c['stage']=='baseline'}
    rows = []
    for call,result in zip(plan['calls'],results):
        if call['stage']!='choice':
            continue
        task, output = plan['tasks'][call['task']], executed[call['id']]
        chosen = action(result['text'],call['mapping'])
        base_correct = bool(grade(baseline[task['id']],task))
        delivered_correct = bool(grade(output,task))
        loss = call['cost'] if chosen=='verify' else 1-float(delivered_correct) if chosen=='direct' else 1.
        common = call['cost'] if chosen=='verify' else 1-float(base_correct) if chosen=='direct' else 1.
        p = plan['calibration']['counts'][f"{task['family']}/{task['level']}"]['p']
        beta_action = optimal(p,call['cost'])
        beta_loss = call['cost'] if beta_action=='verify' else 1-float(base_correct)
        rows.append(dict(task=task,call=call,chosen=chosen,actualLoss=loss,commonAnswerLoss=common,
            betaCommonAnswerLoss=beta_loss,baselineCorrect=base_correct,deliveredCorrect=delivered_correct,
            decisionSeconds=result['seconds'],executionSeconds=output['seconds'],
            decisionTokens=result['metrics']['outputTokens'],executionTokens=output['metrics'].get('outputTokens',0),
            directCorrect=int(chosen=='direct' and delivered_correct)))
    return rows, baseline


def group_key(mode,cost,wording,mapping):
    return f'{mode}/{cost:.1f}/w{wording}/m{mapping}'


def analyze(path):
    data = read_journal(path)
    header, results = data['header'], data['results']
    plan = header['plan']
    complete = len(results)==plan['plannedRecords'] and data['pending'] is None and not data['failed']
    report = dict(schema='menia-native-choice-report-v1',origin=header['origin'],planHash=header['planHash'],
        sourceHash=header['sourceHash'],journalChainEnd=data['chainEnd'],complete=complete,recorded=len(results),
        planned=plan['plannedRecords'],statuses=dict(Counter(r['status'] for r in results)),
        actualLLMCalls=sum(r['engine']=='llm' for r in results),actualToolCalls=sum(r['engine']=='exact_tool' for r in results),
        invalidExecutionSlots=sum(r['engine']=='invalid_choice' for r in results))
    if not complete:
        return report
    rows, baselines = observations(data)
    risk = {}
    for w,m in itertools.product(range(2),repeat=2):
        selected = [(c,r) for c,r in zip(plan['calls'],results) if c['stage']=='publicRisk' and (c['wording'],c['mapping'])==(w,m)]
        actual = [action(r['text'],m) for c,r in selected]
        correct = sum(a==optimal(c['p'],c['cost']) for (c,r),a in zip(selected,actual))
        expected_losses = [c['cost'] if a=='verify' else 1-c['p'] if a=='direct' else 1. for (c,r),a in zip(selected,actual)]
        best_losses = [min(1-c['p'],c['cost']) for c,r in selected]
        risk[f'w{w}/m{m}'] = dict(n=len(selected),optimalChoices=correct,accuracy=correct/len(selected),
            actions=dict(Counter(actual)),meanExpectedRegret=float(np.mean(np.asarray(expected_losses)-best_losses)),
            passed=correct/len(selected)>=plan['publicRiskMinimumAccuracy'])
    by_key, native = {}, {}
    for mode,cost,w,m in itertools.product(('unassisted','history'),COSTS,range(2),range(2)):
        selected = sorted((r for r in rows if (r['call']['mode'],r['call']['cost'],r['call']['wording'],r['call']['mapping'])==(mode,cost,w,m)),key=lambda r:r['task']['id'])
        require(len(selected)==len(plan['tasks']),'Missing paired decisions')
        key=group_key(mode,cost,w,m)
        by_key[key]=selected
        native[key]=dict(n=len(selected),actions=dict(Counter(r['chosen'] for r in selected)),
            directCorrect=sum(r['directCorrect'] for r in selected),deliveredCorrect=sum(r['deliveredCorrect'] for r in selected),
            baselineCorrect=sum(r['baselineCorrect'] for r in selected),
            meanActualLoss=float(np.mean([r['actualLoss'] for r in selected])),
            meanCommonAnswerLoss=float(np.mean([r['commonAnswerLoss'] for r in selected])),
            meanBetaCommonAnswerLoss=float(np.mean([r['betaCommonAnswerLoss'] for r in selected])),
            alwaysDirectReferenceLoss=float(np.mean([1-float(r['baselineCorrect']) for r in selected])),
            alwaysVerifyReferenceLoss=cost,
            decisionSeconds=sum(r['decisionSeconds'] for r in selected),executionSeconds=sum(r['executionSeconds'] for r in selected),
            decisionTokens=sum(r['decisionTokens'] for r in selected),executionTokens=sum(r['executionTokens'] for r in selected))
    rng = np.random.default_rng(SEED+701)
    groups = [[i for i,t in enumerate(plan['tasks']) if (t['family'],t['level'])==cell] for cell in CELLS]
    require(all(len(g)==16 for g in groups),'Fixed per-category size')
    draws = np.concatenate([rng.choice(g,size=(plan['bootstrapResamples'],len(g)),replace=True) for g in groups],axis=1)
    tail=plan['familyAlpha']/(2*plan['primaryComparisons'])
    contrasts=[]
    for cost,w,m in itertools.product(COSTS,range(2),range(2)):
        own = by_key[group_key('unassisted',cost,w,m)]
        history = by_key[group_key('history',cost,w,m)]
        changes=np.asarray([a['actualLoss']-b['actualLoss'] for a,b in zip(own,history)])
        samples=changes[draws].mean(1)
        ci=np.quantile(samples,[tail,1-tail]).tolist()
        gain=float(changes.mean())
        contrasts.append(dict(cost=cost,wording=w,mapping=m,unassistedMinusHistoryLoss=gain,
            interval95=np.quantile(samples,[.025,.975]).tolist(),familyInterval=ci,
            passed=bool(gain>=plan['minimumLossReduction'] and ci[0]>0)))
    transitions={}
    for mode,w,m in itertools.product(('unassisted','history'),range(2),range(2)):
        low=by_key[group_key(mode,.2,w,m)]
        high=by_key[group_key(mode,.8,w,m)]
        transitions[f'{mode}/w{w}/m{m}']=dict(Counter(a['chosen']+'->'+b['chosen'] for a,b in zip(low,high)))
    baseline_cells={}
    for family,level in CELLS:
        tasks=[t for t in plan['tasks'] if (t['family'],t['level'])==(family,level)]
        baseline_cells[f'{family}/{level}']=dict(n=len(tasks),correct=sum(bool(grade(baselines[t['id']],t)) for t in tasks))
    risk_gate=all(r['passed'] for r in risk.values())
    report.update(publicRisk=risk,publicRiskCriterion=risk_gate,native=native,primaryContrasts=contrasts,
        historyBenefitCriterion=bool(risk_gate and all(c['passed'] for c in contrasts)),costTransitions=transitions,
        baselineByCell=baseline_cells,
        baselineSeconds=sum(r['seconds'] for r in baselines.values()),
        baselineOutputTokens=sum(r['metrics']['outputTokens'] for r in baselines.values()),
        scope='Predeclared effect of providing prior category frequency on native choices AND resulting answers. Same model and96 questions across24 conditions; paired resampling unit is question, not individual call. Known-risk gate is a design threshold, not population certainty. Point utility is assigned, separate from observed runtime. Common-answer benchmarks are descriptive, not exact counterfactual executions. No claim of native private-state readout, consciousness or novelty.')
    return report
