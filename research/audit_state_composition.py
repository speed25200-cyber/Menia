"""Independent arithmetic for experiment 12, prepared before its evaluation outcomes.

Shares the fixed plan and strict lineage reader, but enumerates targets, pair
probabilities, AUROC comparisons, block-count bootstrap and criteria separately.
No model calls, checkpoint selection, or source changes to the frozen runner.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from research import state_composition as study
from research.replay_controller_audit import compare


def audit(path,destination):
    h,rows,trained,pending,events=study.read_journal(path);report=study.analyze(path)
    assert report['complete'] and pending is None
    supplied=Path(path).with_suffix('.summary.json')
    summary_error=compare(report,json.loads(supplied.read_text(encoding='utf-8'))) if supplied.exists() else None
    cfg=h['plan']['config'];blocks={b['id']:b for b in h['plan']['blocks']};choices=h['metadata']['choiceTokenIds']
    keyed={r['request']['id']:r for r in rows};metrics={};max_error=0.;shams=0
    def check(value,expected):
        nonlocal max_error
        a=np.asarray(value,dtype=float);b=np.asarray(expected,dtype=float);assert a.shape==b.shape
        error=float(np.max(np.abs(a-b)));max_error=max(max_error,error)
        assert error<1e-10,(value,expected,error)
    for row in rows:
        r=row['request'];out=row['result'];block=blocks[r['block']];offset=2 if r['format']=='new_digits' else 0
        pair=(offset,offset+1);truth=int(r['position'] in (1,2)) if r['task']=='monitor' else int(block['marker']==(0 if r['task']=='marker_first' else 1))
        y=offset+(truth if r['mapping']==0 else 1-truth);l=out['choiceLogits']
        prediction=offset if l[offset]>=l[offset+1] else offset+1
        exp=[math.exp(x-max(l)) for x in l];total=math.fsum(exp);pair_total=math.fsum(exp[i] for i in pair)
        # Pair-normalized softmax is calculated separately to avoid underflow
        # if the unused digits greatly dominate a synthetic stress fixture.
        z=[math.exp(l[i]-max(l[j] for j in pair)) for i in pair];p=[v/math.fsum(z) for v in z]
        raw=choices.index(out['rawTokenId']) if out['rawTokenId'] in choices else -1;assert raw==out['rawChoice']
        latent=int(block['shuffledTargets'][0 if r['position']==3 else r['position']]!=0) if r['arm']=='shuffled' and r['family']=='hidden' and r['task']=='monitor' else truth
        training_target=offset+(latent if r['mapping']==0 else 1-latent)
        metrics[r['id']]=dict(target=y,correct=int(prediction==y),firstTokenCorrect=int(raw==y),firstTokenIsOption=int(raw in pair),
            trainingTargetAccuracy=int(prediction==training_target),choiceMass=out['choiceMass']*pair_total/total,fourDigitMass=out['choiceMass'],
            brier=math.fsum((v-int(i==y))**2 for i,v in zip(pair,p)),rawScore=l[offset+1]-l[offset],
            presenceScore=(l[offset+1]-l[offset])*(1 if r['mapping']==0 else -1))
    tables={};areas={};gates=[];contrast_keys=[];primary=[]
    for rep in range(cfg['replications']):
        for split in ('train','test'):
            bs=[b for b in blocks.values() if b['replication']==rep and b['split']==split];n=len(bs)
            draws=np.random.default_rng(h['plan']['seed']+900+rep).integers(0,n,(cfg['resamples'],n))
            counts=np.stack([np.bincount(d,minlength=n) for d in draws])
            def interval(v):return np.quantile(v,[.025,.975]).tolist()
            def boot_mean(v):return counts@np.asarray(v,dtype=float)/n
            def area(triples):
                wins=np.array([[math.fsum(float(a>b[0])+.5*float(a==b[0]) for a in row[1:]) for b in triples] for row in triples])
                point=math.fsum(float(a>b)+.5*float(a==b) for row in triples for a in row[1:] for b in [t[0] for t in triples])/(2*n*n)
                samples=np.einsum('bi,ij,bj->b',counts,wins,counts)/(2*n*n)
                return point,samples
            for form in (('trained',) if split=='train' else ('trained','paraphrase','new_digits')):
                if split=='train':streams=(('composed','hidden'),('shuffled','hidden'),('grammar','visible'))
                elif form=='trained':streams=(('base','hidden'),('parent','hidden'),('composed','hidden'),('shuffled','hidden'),('grammar','hidden'),('composed','visible'),('grammar','visible'))
                else:streams=(('composed','hidden'),('composed','visible'),('grammar','visible'))
                for (arm,family),task,mapping in itertools.product(streams,('monitor','marker_first','marker_second'),(0,1)):
                    key=f'{rep}/{split}/{arm}/{family}/{form}/{task}/{mapping}';table=report['tables'][key];groups=[]
                    for b in bs:
                        prefix=f"{rep}/{arm}/{family}/{b['id']}/{form}/{task}/{mapping}/"
                        a,c=(keyed[prefix+str(pos)]['result'] for pos in (0,3))
                        assert all(a[k]==c[k] for k in ('choiceLogits','choiceMass','rawTokenId','rawChoice'));shams+=1
                        groups.append([metrics[prefix+str(pos)] for pos in (0,1,2)])
                    items=[m for group in groups for m in group]
                    assert table['n']==3*n and table['blocks']==n
                    for field in ('correct','firstTokenCorrect','firstTokenIsOption','choiceMass','fourDigitMass','brier'):
                        check(table[field],statistics.mean(m[field] for m in items))
                    if split=='train':check(table['trainingTargetAccuracy'],statistics.mean(m['trainingTargetAccuracy'] for m in items))
                    else:assert 'trainingTargetAccuracy' not in table
                    targets=(2,3) if form=='new_digits' else (0,1)
                    acc=[statistics.mean(m['correct'] for m in items if m['target']==y) for y in targets]
                    balanced=statistics.mean(acc);check(table['correctByClass'],acc);check(table['balancedAccuracy'],balanced)
                    delta=[statistics.mean(m['rawScore'] for m in g[1:])-g[0]['rawScore'] for g in groups]
                    check(table['rawShift'],statistics.mean(delta));check(table['rawShiftInterval95'],interval(boot_mean(delta)))
                    if task=='monitor':
                        triples=[[m['presenceScore'] for m in g] for g in groups];point,samples=area(triples)
                        check(table['presenceAUROC'],point);check(table['presenceAUROCInterval95'],interval(samples))
                        ordering=statistics.mean(float(g[i]>g[0])+.5*float(g[i]==g[0]) for g in triples for i in (1,2))
                        check(table['withinBlockOrdering'],ordering);areas[(rep,split,arm,family,form,mapping)]=(point,samples)
                    else:
                        change=[statistics.mean(float(m['correct']) for m in g[1:])-g[0]['correct'] for g in groups]
                        check(table['accuracyChange'],statistics.mean(change));check(table['accuracyChangeInterval95'],interval(boot_mean(change)))
                    tables[key]=balanced
                    if split=='test' and arm=='composed' and form in ('trained','paraphrase'):
                        gate=dict(table=key,options=statistics.mean(m['firstTokenIsOption'] for m in items)>=cfg['optionGate'],
                                  accuracy=balanced>=cfg['accuracyGate'] if family=='visible' or task!='monitor' else True)
                        gates.append(gate)
        for form,mapping in itertools.product(('trained','paraphrase','new_digits'),(0,1)):
            a,ab=areas[(rep,'test','composed','hidden',form,mapping)]
            for other in (('chance','shuffled','grammar','parent','base') if form=='trained' else ('chance',)):
                b,bb=(.5,.5) if other=='chance' else areas[(rep,'test',other,'hidden',form,mapping)]
                cs=[c for c in report['contrasts'] if (c['replication'],c['format'],c['mapping'],c['other'])==(rep,form,mapping,other)];assert len(cs)==1
                c=cs[0];ci=interval(ab-bb);check(c['difference'],a-b);check(c['interval95'],ci)
                expected_primary=form in ('trained','paraphrase') and other in ('chance','shuffled','grammar');assert c['primary']==expected_primary
                contrast_keys.append((rep,form,mapping,other))
                if expected_primary:primary.append(ci[0]>0)
    assert set(tables)==set(report['tables']) and len(contrast_keys)==len(report['contrasts'])
    assert sorted(gates,key=lambda g:g['table'])==sorted(report['gates'],key=lambda g:g['table'])
    signals=all(primary);controls=all(g['accuracy'] and g['options'] for g in gates)
    assert report['signalRuleMet']==signals and report['controlRuleMet']==controls and report['fixedReadingRuleMet']==(signals and controls)
    assert report['shamPairs']==shams
    checkpoints={}
    for key,sha in trained.items():
        ck=Path(path).with_suffix('.'+key+'.safetensors')
        if ck.exists():
            assert hashlib.sha256(ck.read_bytes()).hexdigest()==sha;checkpoints[key]=sha
    for field,event in [('trainingUpdates','training_step'),('trainingRestarts','training_restart'),('interruptedRequests','interrupted_request'),('errors','error')]:
        assert report[field]==sum(e['event']==event for e in events)
    check(report['trainingSeconds'],math.fsum(e['seconds'] for e in events if e['event']=='training_step'))
    check(report['evaluationSeconds'],math.fsum(r['result']['seconds'] for r in rows))
    result=dict(schema='menia-state-composition-verification-v1',origin=h['metadata']['origin'],complete=True,
        journalSHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),planHash=h['planHash'],sourceHash=h['sourceHash'],
        independentMaxDifference=max_error,summaryMaxDifference=summary_error,evaluations=len(rows),shamPairs=shams,
        tables=len(tables),contrasts=len(contrast_keys),primaryContrasts=len(primary),controlGates=len(gates),checkpointsVerified=checkpoints,
        trainingUpdates=report['trainingUpdates'],errors=report['errors'],trainingRestarts=report['trainingRestarts'],interruptedRequests=report['interruptedRequests'],
        inputTokenRange=[min(r['inputTokens'] for r in rows),max(r['inputTokens'] for r in rows)],
        maxNormRelativeError=max(r['result']['intervention']['normRelativeError'] for r in rows),
        firstTokenOutsideOptions=sum(not m['firstTokenIsOption'] for m in metrics.values()),
        signalRuleMet=signals,controlRuleMet=controls,fixedReadingRuleMet=signals and controls,
        limitations='Shares strict reader and fixed plan; independently recomputes arithmetic and criteria. Missing checkpoint files are explicitly unverified. Not an external replication, proof of novelty, or consciousness test.')
    Path(destination).write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');return result


if __name__=='__main__':print(json.dumps(audit(sys.argv[1],sys.argv[2]),indent=2,ensure_ascii=False))
