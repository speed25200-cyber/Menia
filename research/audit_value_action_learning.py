"""Separate finite-grid arithmetic and frozen-checkpoint validation for Colab22."""
import hashlib
import itertools
import math
from pathlib import Path

from research.value_action_learning_journal import read_journal
from research.audit_composition_diagnostic import compare


def verify(path,report,check_weights=True):
    d=read_journal(path);p=d['header']['plan'];rows=d['results']
    if len(rows)!=9216 or len(d['checkpoints'])!=9 or d['pending'] or d['failed']:raise ValueError('Incomplete trial')
    if any(r['status']!='ok' for r in rows):raise ValueError('Failed calls')
    weights={};initial_hashes=[]
    if check_weights:
        from safetensors.torch import load_file
        import torch
        for rep in range(3):
            initial=Path(path).with_suffix(f'.r{rep}-initial.safetensors')
            sha=hashlib.sha256(initial.read_bytes()).hexdigest();base=load_file(str(initial));initial_hashes.append(sha)
            if not base or any(not torch.isfinite(v).all() for v in base.values()):raise ValueError('Invalid initial weights')
            weights[initial.name]=sha
            for arm in ('choice','linked','shuffled'):
                key=f'r{rep}-{arm}';event=d['checkpoints'][key];file=Path(path).with_suffix('.'+key+'.safetensors')
                if sha!=event['initializationHash']:raise ValueError('Unmatched initial weights')
                actual=hashlib.sha256(file.read_bytes()).hexdigest()
                if actual!=event['sha256']:raise ValueError('Checkpoint bytes changed')
                state=load_file(str(file))
                if set(state)!=set(base) or any(state[k].shape!=base[k].shape or not torch.isfinite(state[k]).all() for k in base):raise ValueError('Invalid trained weights')
                if not any(not torch.equal(state[k],base[k]) for k in base):raise ValueError('Weights unchanged')
                weights[file.name]=actual
        if len(set(initial_hashes))!=3:raise ValueError('Initializations not distinct')
    groups={};values={}
    for c in p['calls']:
        r=rows[c['id']];case=p['testCases'][c['case']];pair=('1','2') if c['symbols']=='digits' else ('A','B')
        raw=r['text'].strip();direct=pair[c['mapping']];verify_code=pair[1-c['mapping']]
        action='direct' if raw==direct else 'verify' if raw==verify_code else 'invalid'
        expected=('direct' if case['pPercent']+case['costCents']>100 else 'verify') if c['task']=='choice' else c['selected']
        good=int(action==expected)
        key=f'r{c["replication"]}/{c["arm"]}/{c["task"]}/w{c["wording"]}/{c["symbols"]}/o{c["order"]}/m{c["mapping"]}'
        if c['task']=='lookup':key+='/'+c['selected']
        groups.setdefault(key,[]).append((good,int(action=='invalid'),r['metrics']['outputTokens'],r['seconds']))
        if c['task']=='choice':values[(c['replication'],c['arm'],c['case'],c['wording'],c['symbols'],c['order'],c['mapping'])]=good
    tables={}
    for key,x in groups.items():
        if len(x)!=24:raise ValueError('Test group size')
        t={k:sum(v[i] for v in x) for i,k in enumerate(('correct','invalid','tokens'))}
        t.update(n=24,seconds=math.fsum(v[3] for v in x),accuracy=t['correct']/24,passed=t['correct']>=22);tables[key]=t
    contrasts=[];gates={}
    arms=('base','choice','linked','shuffled')
    for rep in range(3):
        for domain in ('trainedSurface','newWording','newSymbols','bothNew'):
            ws=[1] if domain in ('trainedSurface','newSymbols') else [2,3];sym='digits' if domain in ('trainedSurface','newWording') else 'letters'
            per_case={arm:[math.fsum(values[(rep,arm,k,w,sym,o,m)] for w,o,m in itertools.product(ws,(0,1),(0,1)))/(4*len(ws)) for k in range(24)] for arm in arms}
            scores={arm:math.fsum(v)/24 for arm,v in per_case.items()}
            contrasts.append(dict(replication=rep,domain=domain,accuracy=scores,caseAccuracies=per_case,
                                  linkedGain={a:scores['linked']-scores[a] for a in ('base','choice','shuffled')}))
        for arm,task in itertools.product(arms,('choice','lookup')):
            prefix=f'r{rep}/{arm}/{task}'
            subset=[t for k,t in tables.items() if k.startswith(prefix+'/')]
            if len(subset)!=(24 if task=='choice' else 8):raise ValueError('Missing presentation group')
            gates[prefix]=all(t['correct']>=22 for t in subset)
    criterion=all(gates[f'r{r}/linked/{task}'] for r,task in itertools.product(range(3),('choice','lookup'))) and all(v>=.05 for c in contrasts if c['domain']!='trainedSurface' for v in c['linkedGain'].values())
    expected=dict(schema='menia-value-action-learning-summary-v1',origin=d['header']['origin'],complete=True,planned=9216,recorded=9216,
                  planHash=d['header']['planHash'],sourceHash=d['header']['sourceHash'],checkpoints=d['checkpoints'],trainingSteps=576,
                  groups=tables,contrasts=contrasts,capabilityGates=gates,robustLearningCriterion=criterion,
                  tokenLimits=sum(int(r['metrics']['reachedTokenLimit']) for r in rows),scope=p['scope'])
    delta=compare(expected,report)
    return dict(schema='menia-value-action-learning-verification-v1',verified=True,recorded=9216,trainingSteps=576,groups=384,contrasts=12,
                journalSHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),maxAbsoluteDifference=delta,
                weightsChecked=check_weights,weights=weights,scope='Separate arithmetic and weights; shared plan and integrity reader, not an external replication.')
