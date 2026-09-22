"""Paired capability interventions: fixed groups, pre-answer readouts, executed routes.

This trains an external monitor; it does not train Qwen or measure consciousness.
"""
from collections import Counter
import json
from pathlib import Path
import random
import re
import time

import numpy as np

from research.activation_monitor import (
    ALPHAS, DIM, PROJECTIONS, features, messages, predict, ridge_fit, validate_state,
    make_plan as activation_plan,
)
from research.cross_model_prediction import MODELS, SETTINGS, append, canonical, digest, grade, make_plan as cross_plan
from research.iphone_coupling_report import require, strict_json

SEED = 202609152
CELLS = (('countA', 8), ('alternatingSum', 2))
CONDITIONS = {'baseline': 0., 'sham': 0., 'rotateHalf': .5, 'rotateOne': 1.}
COUNTS = {'gate': 4, 'train': 48, 'validation': 12, 'test': 24}
NAMES = ('betaCell', 'oracleCondition', 'inputOnly', 'internal', 'shuffledLabels', 'donorState')
GATE_END, FIT_END, TOTAL = 32, 512, 704


def cell(task):
    return task['family'], task['level']


def make_plan():
    rng = random.Random(SEED)
    seen = {t['question'] for p in (activation_plan(), cross_plan()) for t in p['tasks']}
    tasks, qid = [], 0
    for split, count in COUNTS.items():
        for block in range(count):
            cells = list(CELLS)
            rng.shuffle(cells)
            for family, level in cells:
                while True:
                    letters = ''.join(rng.choice('ABCD') for _ in range(level)) if family == 'countA' else ''
                    operands = [rng.randrange(10, 100) for _ in range(level)] if not letters else []
                    question = (f'Combien de lettres A contient cette chaîne : {letters} ?' if letters else
                                f'Calcule {operands[0]} - {operands[1]}.')
                    if question not in seen:
                        seen.add(question)
                        break
                modes = list(CONDITIONS)
                rng.shuffle(modes)
                for condition in modes:
                    tasks.append(dict(id=len(tasks), qid=qid, split=split, family=family, level=level,
                        letters=letters, operands=operands, question=question, condition=condition,
                        seed=int(digest([SEED, qid, 'answer'])[:8], 16),
                        noiseSeed=int(digest([SEED, qid, 'direction'])[:8], 16)))
                qid += 1
    return dict(schema='menia-perturbation-plan-v1', seed=SEED, counts=COUNTS,
        model=MODELS['A'], settings=SETTINGS, conditions=CONDITIONS, projections=PROJECTIONS,
        projectionDimension=DIM, alphas=list(ALPHAS), tasks=tasks,
        intervention='Norm-preserving rotation at middle block, last prefix token, first prefill only',
        threshold=.8, verificationPointCost=.2, errorPointCost=1.)


def usable(rows, split):
    return [r for r in rows if r['task']['split'] == split and r['result']['status'] == 'ok']


def beta_key(task, condition=False):
    return f"{task['family']}:{task['level']}" + (':'+task['condition'] if condition else '')


def fit_bundle(rows):
    train, val = usable(rows, 'train'), usable(rows, 'validation')
    for group, minimum in ((train, 32), (val, 8)):
        require(all(sum(cell(r['task']) == c and r['task']['condition'] == k for r in group) >= minimum
                    for c in CELLS for k in CONDITIONS), 'Too few training/validation examples per condition')
    y = np.array([grade(r['result'], r['task']) for r in train], dtype=float)
    vy = np.array([grade(r['result'], r['task']) for r in val], dtype=float)
    shuffled = y.copy()
    rng = np.random.default_rng(SEED+9)
    for c in CELLS:
        for k in CONDITIONS:
            ids = [i for i,r in enumerate(train) if cell(r['task']) == c and r['task']['condition'] == k]
            shuffled[ids] = rng.permutation(y[ids])
    models, selection = {}, {}
    for name in ('inputOnly', 'internal', 'shuffledLabels'):
        x = np.array([features(r['task'], r['state'], internal=name != 'inputOnly') for r in train])
        vx = np.array([features(r['task'], r['state'], internal=name != 'inputOnly') for r in val])
        candidates, best = [], None
        for alpha in ALPHAS:
            model = ridge_fit(x, shuffled if name == 'shuffledLabels' else y, alpha)
            loss = float(np.mean((predict(model, vx)-vy)**2))
            candidates.append(dict(alpha=alpha, validationBrier=loss))
            if best is None or loss < best[0]:
                best = loss, model
        models[name], selection[name] = best[1], candidates
    beta = {}
    for aware in (False, True):
        for key in sorted({beta_key(r['task'], aware) for r in train}):
            labels = [y[i] for i,r in enumerate(train) if beta_key(r['task'], aware) == key]
            beta[key] = float((sum(labels)+1)/(len(labels)+2))
    return dict(models=models, selection=selection, beta=beta, trainCount=len(train), validationCount=len(val),
                fitDataHash=digest(train+val))


def forecast(bundle, rows, task, state):
    # Only the explicitly privileged oracle and matched donor receive condition.
    out = dict(betaCell=bundle['beta'][beta_key(task)], oracleCondition=bundle['beta'][beta_key(task, True)])
    for name, model in bundle['models'].items():
        out[name] = float(predict(model, features(task, state, internal=name != 'inputOnly')))
    donors = [r for r in usable(rows, 'train') if cell(r['task']) == cell(task) and r['task']['condition'] == task['condition']]
    donor = donors[int(digest([SEED, task['qid'], task['condition'], 'donor'])[:8], 16) % len(donors)]
    replaced = dict(state, middle=donor['state']['middle'], final=donor['state']['final'])
    out['donorState'] = float(predict(bundle['models']['internal'], features(task, replaced, internal=True)))
    return out


def actions(probabilities):
    return {k: 'direct' if p >= .8 else 'verify' for k,p in probabilities.items()}


def verify_question(question):
    """Execute a deterministic tool from the public question, not the scored label."""
    count = re.fullmatch(r'Combien de lettres A contient cette chaîne : ([ABCD]+) \?', question)
    if count:
        return str(Counter(count.group(1))['A'])
    subtract = re.fullmatch(r'Calcule ([0-9]+) - ([0-9]+)\.', question)
    require(subtract is not None, 'Unsupported verification request')
    return str(int(subtract.group(1))-int(subtract.group(2)))


def execute_routes(task, text, decisions):
    result = {}
    for name, action in decisions.items():
        started = time.perf_counter()
        final = text if action == 'direct' else verify_question(task['question'])
        result[name] = dict(action=action, text=final, seconds=time.perf_counter()-started)
    return result


def check_pair(group):
    """No outcome-based selection: only sham identity and identical visible input."""
    if len(group) != 4 or any(r['result']['status'] != 'ok' for r in group):
        return
    paired = {r['task']['condition']:r for r in group}
    b,s = paired['baseline'],paired['sham']
    require(b['result']['text'] == s['result']['text'] and b['state'] == s['state'], 'Sham changed computation')
    require(all(r['state']['input'] == b['state']['input'] for r in group), 'Paired input embeddings differ')


def check_runtime_metrics(metrics, task):
    require(metrics['dtype'] == 'torch.bfloat16' and metrics['device'] == 'cuda:0', 'GPU/dtype differs')
    for k in ('temperature','top_p','top_k','min_p','do_sample','max_new_tokens','repetition_penalty','renormalize_logits','use_cache'):
        require(metrics['effectiveGeneration'][k] == SETTINGS[k], 'Generation setting differs')
    require(0 < metrics['inputTokens'] <= SETTINGS['max_input_tokens'] and
            0 < metrics['outputTokens'] <= SETTINGS['max_new_tokens'], 'Token count outside protocol')
    trace = metrics['intervention']
    require(trace['condition'] == task['condition'] and trace['applications'] == (0 if task['condition'] == 'baseline' else 1), 'Intervention trace differs')
    require(np.isfinite(trace['normRelativeError']) and 0 <= trace['normRelativeError'] <= .01, 'Norm check failed')
    expected_change = np.sqrt(2-2/np.sqrt(1+CONDITIONS[task['condition']]**2))
    require(np.isfinite(trace['relativeChange']) and abs(trace['relativeChange']-expected_change) < .015, 'Rotation displacement differs')


def gate_report(rows):
    group = [r for r in rows if r['task']['split'] == 'gate']
    passed = len(group) == GATE_END and all(r['result']['status'] == 'ok' for r in group)
    sham_matches, input_matches = 0, 0
    for qid in sorted({r['task']['qid'] for r in group}):
        paired = {r['task']['condition']: r for r in group if r['task']['qid'] == qid and r['result']['status'] == 'ok'}
        if len(paired) != 4:
            passed = False
            continue
        b, s = paired['baseline'], paired['sham']
        sham_matches += b['result']['text'] == s['result']['text'] and b['state'] == s['state']
        input_matches += all(r['state']['input'] == b['state']['input'] for r in paired.values())
    passed = passed and sham_matches == input_matches == 8
    return dict(passed=bool(passed), results=len(group), shamExactMatches=int(sham_matches), identicalInputs=int(input_matches),
                criterion='Technical identity only; no accuracy or favorable-effect selection')


def close_tree(a, b):
    if isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), 'Fit fields differ')
        for key in a:
            close_tree(a[key], b[key])
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), 'Fit length differs')
        for x,y in zip(a,b):
            close_tree(x,y)
    elif isinstance(a, (int,float)):
        require(type(b) in (int,float) and np.isclose(a,b,atol=1e-8,rtol=1e-8), 'Refitted monitor mismatch')
    else:
        require(a == b, 'Fit value differs')


def read_journal(path):
    events = [strict_json(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    require(bool(events) and events[0].get('event') == 'header', 'Missing header')
    header, rows, pending, bundle, gate = events[0], [], None, None, None
    plan = make_plan()
    require(header['plan'] == plan and header['planHash'] == digest(plan), 'Fixed plan mismatch')
    require(header['origin'] in ('transformers_gpu', 'synthetic_fixture'), 'Unknown origin')
    for event in events[1:]:
        kind = event.get('event')
        if kind == 'gate':
            require(pending is None and gate is None and len(rows) == GATE_END, 'Gate order')
            gate = gate_report(rows)
            require(event == dict(event='gate', report=gate), 'Gate mismatch')
        elif kind == 'fit':
            require(pending is None and bundle is None and len(rows) == FIT_END, 'Fit must precede test')
            bundle = event['bundle']
            require(event['bundleHash'] == digest(bundle), 'Fit checksum')
            require(bundle['fitDataHash'] == digest(usable(rows,'train')+usable(rows,'validation')), 'Fit data mismatch')
        elif kind == 'request':
            require(pending is None and len(rows) < TOTAL, 'Unexpected request')
            task = plan['tasks'][len(rows)]
            require(event == dict(event='request', task=task, messages=messages(task)), 'Request mismatch')
            require(task['split'] == 'gate' or gate is not None and gate['passed'], 'Failed or missing technical gate')
            require(task['split'] != 'test' or bundle is not None, 'Test without frozen monitor')
            pending = dict(task=task)
        elif kind == 'state':
            require(pending is not None and 'state' not in pending and event['id'] == len(rows), 'State order')
            validate_state(event['state'])
            expected = forecast(bundle,rows,pending['task'],event['state']) if bundle else None
            if expected is None:
                require(event['predictions'] is None and event['actions'] is None, 'Premature decisions')
            else:
                recorded = event['predictions']
                require(isinstance(recorded,dict) and set(recorded) == set(expected), 'Forecast keys')
                require(all(type(recorded[k]) in (float,int) and np.isfinite(recorded[k]) and
                            0 <= recorded[k] <= 1 and abs(recorded[k]-v) < 1e-10 for k,v in expected.items()), 'Forecast mismatch')
                require(event['actions'] == actions(recorded), 'Action mismatch')
            pending.update(state=event['state'], predictions=event['predictions'], actions=event['actions'])
        elif kind == 'result':
            require(pending is not None and event['id'] == len(rows), 'Result order')
            require(event['status'] in ('ok','error','interrupted'), 'Result status')
            require(type(event['text']) is str, 'Raw output missing')
            require(type(event['seconds']) in (float,int) and np.isfinite(event['seconds']) and event['seconds'] >= 0, 'Duration')
            if event['status'] == 'ok':
                require('state' in pending, 'Capture missing')
                if header['origin'] == 'transformers_gpu':
                    check_runtime_metrics(event['metrics'], pending['task'])
                if pending['task']['split'] == 'test':
                    routes = event['routes']
                    require(set(routes) == set(NAMES), 'Routes missing')
                    for name, route in routes.items():
                        action = pending['actions'][name]
                        target = event['text'] if action == 'direct' else verify_question(pending['task']['question'])
                        require(route['action'] == action and route['text'] == target, 'Route mismatch')
                        require(type(route['seconds']) in (float,int) and np.isfinite(route['seconds']) and route['seconds'] >= 0, 'Tool duration')
                else:
                    require(event['routes'] is None, 'Premature routing')
            else:
                require(event['text'] == '' and event.get('routes') is None, 'Failed call has fabricated outcome')
            rows.append(dict(pending,result=event))
            pending = None
            if len(rows) % 4 == 0:
                check_pair(rows[-4:])
        else:
            raise ValueError('Unknown event')
    return header, rows, pending, bundle, gate


def collect(path, backend, *, resume=False, limit=None):
    path = Path(path)
    if resume:
        header, rows, pending, bundle, gate = read_journal(path)
        require(header['metadata'] == backend.metadata and header['origin'] == backend.origin, 'Resume environment differs')
        if pending:
            event = dict(event='result', id=len(rows), status='interrupted', text='', seconds=0., routes=None, errorType='UnrecordedResult')
            append(path,event)
            rows.append(dict(pending,result=event))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        plan = make_plan()
        header = dict(event='header',plan=plan,planHash=digest(plan),origin=backend.origin,metadata=backend.metadata)
        # Exclusive create; append() performs the durable write after reservation.
        with path.open('x',encoding='utf-8'):
            pass
        append(path,header)
        rows, bundle, gate = [], None, None
    start = len(rows)
    for task in header['plan']['tasks'][start:]:
        if limit is not None and len(rows)-start >= limit:
            break
        if len(rows) >= GATE_END and gate is None:
            gate = gate_report(rows)
            append(path,dict(event='gate',report=gate))
        require(gate is None or gate['passed'], 'Technical gate failed; preserve attempt for diagnosis')
        if task['split'] == 'test' and bundle is None:
            bundle = fit_bundle(rows)
            append(path,dict(event='fit',bundle=bundle,bundleHash=digest(bundle)))
        append(path,dict(event='request',task=task,messages=messages(task)))
        row = dict(task=task)
        def capture(state):
            require('state' not in row, 'Capture repeated')
            validate_state(state)
            probs = forecast(bundle,rows,task,state) if bundle else None
            decisions = actions(probs) if probs is not None else None
            append(path,dict(event='state',id=task['id'],state=state,predictions=probs,actions=decisions))
            row.update(state=state,predictions=probs,actions=decisions)
        started = time.perf_counter()
        try:
            text, metrics = backend.generate(task,capture)
            require('state' in row and type(text) is str, 'Missing capture/output')
            routes = execute_routes(task,text,row['actions']) if task['split'] == 'test' else None
            event = dict(event='result',id=task['id'],status='ok',text=text,seconds=time.perf_counter()-started,metrics=metrics,routes=routes)
        except (Exception,KeyboardInterrupt) as exc:
            event = dict(event='result',id=task['id'],status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'error',
                         text='',seconds=time.perf_counter()-started,routes=None,errorType=type(exc).__name__)
            append(path,event)
            raise
        append(path,event)
        rows.append(dict(row,result=event))
        if len(rows) % 4 == 0:
            check_pair(rows[-4:])
        print(f"{len(rows)}/{TOTAL} {task['split']} ok",flush=True)
    return len(rows)


def score(rows, name):
    if not rows:
        return dict(n=0,brier=None,auc=None,rawCorrect=0,finalCorrect=0,verify=0,pointLoss=None)
    p = np.array([r['predictions'][name] for r in rows])
    y = np.array([grade(r['result'],r['task']) for r in rows],dtype=float)
    good,bad = p[y == 1],p[y == 0]
    routes = [r['result']['routes'][name] for r in rows]
    final = [grade(dict(status='ok',text=route['text']),r['task']) for r,route in zip(rows,routes)]
    verify = np.array([route['action'] == 'verify' for route in routes])
    return dict(n=len(rows),rawCorrect=int(y.sum()),brier=float(np.mean((p-y)**2)),meanProbability=float(p.mean()),
        auc=float(np.mean((good[:,None] > bad)+.5*(good[:,None] == bad))) if len(good) and len(bad) else None,
        finalCorrect=sum(final),verify=int(verify.sum()),pointLoss=float(np.mean(1-np.array(final)+.2*verify)),
        routingSeconds=sum(route['seconds'] for route in routes))


def analyze(path, *, check_fit=True):
    header,rows,pending,bundle,gate = read_journal(path)
    if bundle and check_fit:
        close_tree(fit_bundle(rows),bundle)
    test = usable(rows,'test')
    paired = []
    for qid in sorted({r['task']['qid'] for r in test}):
        group = [r for r in test if r['task']['qid'] == qid]
        if len(group) == 4:
            paired.append(group)
    contrasts, draws = None, None
    if len(paired) == 48:
        rng = np.random.default_rng(SEED+20)
        strata = [[i for i,g in enumerate(paired) if cell(g[0]['task']) == c] for c in CELLS]
        draws = np.concatenate([rng.choice(ids,(2000,len(ids)),replace=True) for ids in strata],axis=1)
        contrasts = {}
        for name in NAMES:
            if name == 'internal':
                continue
            for metric in ('brier','pointLoss'):
                delta = np.array([score(g,name)[metric]-score(g,'internal')[metric] for g in paired])
                contrasts[name+':'+metric] = dict(otherMinusInternal=float(delta.mean()),
                    descriptiveQuestionClusterBootstrap95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist())
    effects = {}
    for condition in CONDITIONS:
        pairs = []
        for group in paired:
            lookup = {r['task']['condition']:r for r in group}
            b,r = lookup['baseline'],lookup[condition]
            pairs.append((grade(b['result'],b['task']),grade(r['result'],r['task']), b['result']['text'] == r['result']['text']))
        effects[condition] = dict(completeQuestionPairs=len(pairs),baselineCorrect=sum(b for b,_,_ in pairs),
            conditionCorrect=sum(v for _,v,_ in pairs),lost=sum(b and not v for b,v,_ in pairs),
            gained=sum(v and not b for b,v,_ in pairs),sameText=sum(s for _,_,s in pairs))
        if draws is not None:
            delta = np.array([int(b)-int(v) for b,v,_ in pairs])
            effects[condition].update(baselineMinusConditionAccuracy=float(delta.mean()),
                descriptiveQuestionClusterBootstrap95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist())
    return dict(schema='menia-perturbation-report-v1',origin=header['origin'],planHash=header['planHash'],
        model=header['plan']['model'],settings=header['plan']['settings'],recordedResults=len(rows),plannedResults=TOTAL,
        complete=len(rows) == TOTAL and all(r['result']['status'] == 'ok' for r in rows) and bool(gate and gate['passed']),
        statuses=dict(Counter(r['result']['status'] for r in rows)),pendingRequest=pending is not None,gate=gate,
        fitChecked=bool(bundle and check_fit),selection=bundle['selection'] if bundle else None,
        usedForFit={s:len(usable(rows,s)) for s in ('train','validation')},availableTestTargets=len(test),
        independentTestQuestions=len(paired),scores={n:score(test,n) for n in NAMES},contrasts=contrasts,pairedInterventionEffects=effects,
        withinCondition=[dict(family=c[0],level=c[1],condition=k,scores={n:score([r for r in test if cell(r['task']) == c and r['task']['condition'] == k],n) for n in NAMES}) for c in CELLS for k in CONDITIONS],
        interpretation='External monitor and executed deterministic routes on shared candidates. Fixed point costs plus measured routing time. Question-cluster uncertainty conditional on one fit. No native introspection or consciousness claim.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    report = json.dumps(analyze(args.journal),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(report,encoding='utf-8')
    print(report)
