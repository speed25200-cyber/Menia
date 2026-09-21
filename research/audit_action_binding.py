"""Separate finite-grid tally and saved-weight verification for Colab20."""
import hashlib
import itertools
import math
from pathlib import Path

from research.action_binding_journal import read_journal
from research.audit_composition_diagnostic import compare


def verify(path,report,check_weights=True):
    d=read_journal(path);p=d['header']['plan'];rows=d['results']
    if len(rows)!=6912 or len(d['checkpoints'])!=6 or d['pending'] or d['failed']:raise ValueError('Incomplete trial')
    if any(r['status']!='ok' for r in rows):raise ValueError('Failed calls')
    weights={}
    if check_weights:
        from safetensors.torch import load_file
        import torch
        for rep in range(3):
            initial=Path(path).with_suffix(f'.r{rep}-initial.safetensors')
            sha=hashlib.sha256(initial.read_bytes()).hexdigest();base=load_file(str(initial))
            if not base or any(not torch.isfinite(v).all() for v in base.values()):raise ValueError('Invalid initial weights')
            weights[initial.name]=sha
            for arm in ('fixed','permuted'):
                key=f'r{rep}-{arm}';event=d['checkpoints'][key];file=Path(path).with_suffix('.'+key+'.safetensors')
                if sha!=event['initializationHash']:raise ValueError('Unmatched initial weights')
                actual=hashlib.sha256(file.read_bytes()).hexdigest()
                if actual!=event['sha256']:raise ValueError('Checkpoint bytes changed')
                state=load_file(str(file))
                if set(state)!=set(base) or any(state[k].shape!=base[k].shape or not torch.isfinite(state[k]).all() for k in base):raise ValueError('Invalid trained weights')
                if not any(not torch.equal(state[k],base[k]) for k in base):raise ValueError('Weights unchanged')
                weights[file.name]=actual
    groups={};values={}
    for c in p['calls']:
        r=rows[c['id']];case=p['testCases'][c['case']];pair=('1','2') if c['symbols']=='digits' else ('A','B')
        raw=r['text'].strip();direct=pair[c['mapping']];verify_code=pair[1-c['mapping']]
        a='direct' if raw==direct else 'verify' if raw==verify_code else 'invalid'
        optimal='direct' if case['pPercent']+case['costCents']>100 else 'verify';good=int(a==optimal)
        costs=(100-case['pPercent'],case['costCents']);loss=costs[0] if a=='direct' else costs[1] if a=='verify' else 100
        k=f'r{c["replication"]}/{c["arm"]}/{c["information"]}/w{c["wording"]}/{c["symbols"]}/o{c["order"]}/m{c["mapping"]}'
        groups.setdefault(k,[]).append((good,int(a=='invalid'),int(a==('direct' if c['order']==0 else 'verify')),loss-min(costs),r['metrics']['outputTokens'],r['seconds']))
        values[(c['replication'],c['arm'],c['case'],c['information'],c['wording'],c['symbols'],c['order'],c['mapping'])]=good
    tables={}
    for key,x in groups.items():
        if len(x)!=16:raise ValueError('Test group size')
        t={k:sum(v[i] for v in x) for i,k in enumerate(('correct','invalid','first','regretCents','tokens'))}
        t.update(n=16,seconds=math.fsum(v[5] for v in x),accuracy=t['correct']/16,passed=t['correct']>=15);tables[key]=t
    contrasts=[];gates={}
    for rep in range(3):
        for info,domain in itertools.product(('probability','expectedLoss'),('trainedSurface','newWording','newSymbols','bothNew')):
            ws=[0] if domain in ('trainedSurface','newSymbols') else [1,2];ss='digits' if domain in ('trainedSurface','newWording') else 'letters'
            case_values={a:[math.fsum(values[(rep,a,k,info,w,ss,o,m)] for w,o,m in itertools.product(ws,(0,1),(0,1)))/(4*len(ws)) for k in range(16)] for a in ('base','fixed','permuted')}
            accuracies={a:math.fsum(x)/16 for a,x in case_values.items()}
            contrasts.append(dict(replication=rep,information=info,domain=domain,accuracy=accuracies,
                permutedGain={a:accuracies['permuted']-accuracies[a] for a in ('base','fixed')},caseAccuracies=case_values))
        for arm,info in itertools.product(('base','fixed','permuted'),('probability','expectedLoss')):
            prefix=f'r{rep}/{arm}/{info}'
            subset=[t for k,t in tables.items() if k.startswith(prefix+'/')]
            if len(subset)!=24:raise ValueError('Missing presentation group')
            gates[prefix]=all(t['correct']>=15 for t in subset)
    criterion=all(gates[f'r{r}/permuted/{i}'] for r,i in itertools.product(range(3),('probability','expectedLoss'))) and all(v>=.05 for c in contrasts if c['domain']!='trainedSurface' for v in c['permutedGain'].values())
    expected=dict(schema='menia-action-binding-summary-v1',origin=d['header']['origin'],complete=True,planned=6912,recorded=6912,
        planHash=d['header']['planHash'],sourceHash=d['header']['sourceHash'],checkpoints=d['checkpoints'],trainingSteps=384,
        groups=tables,contrasts=contrasts,capabilityGates=gates,robustLearningCriterion=criterion,
        tokenLimits=sum(int(r['metrics']['reachedTokenLimit']) for r in rows),scope=p['scope'])
    delta=compare(expected,report)
    return dict(schema='menia-action-binding-verification-v1',verified=True,recorded=6912,trainingSteps=384,groups=432,contrasts=24,
        journalSHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),maxAbsoluteDifference=delta,
        weightsChecked=check_weights,weights=weights,scope='Separate tally, weights and contrasts; shared plan and integrity reader, not external replication.')
