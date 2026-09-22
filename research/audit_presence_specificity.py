"""Independent arithmetic audit, written before viewing experiment 11 outcomes.

Uses the strict protocol reader for lineage/order; enumerates its own targets,
binary predictions and pair counts. Bootstrap uses block multiplicities rather
than the report's indexed comparison matrices. No new model calls.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from research import presence_specificity as study
from research.replay_controller_audit import compare


def audit(path, destination):
    h,rows,pending,events=study.read_journal(path)
    report=study.analyze(path)
    assert report['complete'] and pending is None
    supplied=Path(path).with_suffix('.summary.json')
    summary_error=compare(report,json.loads(supplied.read_text(encoding='utf-8'))) if supplied.exists() else None
    config=h['plan']['config'];blocks={b['id']:b for b in h['plan']['blocks']}
    choices=h['metadata']['choiceTokenIds'];keyed={r['request']['id']:r for r in rows}
    independent={};max_error=0.;shams=0
    def check(value,expected):
        nonlocal max_error
        a=np.asarray(value,dtype=float);b=np.asarray(expected,dtype=float)
        assert a.shape==b.shape,(a.shape,b.shape)
        error=float(np.max(np.abs(a-b)));max_error=max(max_error,error)
        assert error<1e-10,(value,expected,error)
    def label(req,block):
        q=req['question']
        if q=='presence':return 1 if req['position']==1 or req['position']==2 else 0
        if q=='absence':return 0 if req['position']==1 or req['position']==2 else 1
        if q=='marker_first':return int(block['marker']==0)
        if q=='marker_second':return int(block['marker']==1)
        raise AssertionError(q)
    for r in rows:
        req=r['request'];out=r['result'];y=label(req,blocks[req['block']]);l=out['choiceLogits']
        prediction=0 if l[0]>=l[1] else 1
        exp=[math.exp(l[i]-max(l[:2])) for i in (0,1)];probs=[x/math.fsum(exp) for x in exp]
        raw=choices.index(out['rawTokenId']) if out['rawTokenId'] in choices else -1
        assert raw==out['rawChoice']
        independent[req['id']]=dict(target=y,correct=int(prediction==y),firstTokenCorrect=int(raw==y),
            firstTokenIsOption=int(raw in (0,1)),choiceMass=out['choiceMass'],
            brier=math.fsum((v-int(i==y))**2 for i,v in enumerate(probs)),
            rawScore=l[1]-l[0],probabilityOne=probs[1])
    results={};contrasts=[];gates=[]
    for rep in range(config['replications']):
        bs=[b for b in blocks.values() if b['replication']==rep];n=len(bs)
        draws=np.random.default_rng(h['plan']['seed']+900+rep).integers(0,n,(config['resamples'],n))
        counts=np.stack([np.bincount(draw,minlength=n) for draw in draws])
        def boot_mean(v):return counts@np.asarray(v)/n
        def interval(v):return np.quantile(v,[.025,.975]).tolist()
        def area(triples):
            counts_pair=np.array([[math.fsum(float(a>b[0])+.5*float(a==b[0]) for a in row[1:]) for b in triples] for row in triples])
            point=math.fsum(float(a>b)+.5*float(a==b) for row in triples for a in row[1:] for b in [t[0] for t in triples])/(2*n*n)
            boot=np.einsum('bi,ij,bj->b',counts,counts_pair,counts)/(2*n*n)
            return point,boot
        for arm in h['plan']['arms']:
            for family in ('hidden','visible'):
                for question in ('presence','absence','marker_first','marker_second'):
                    key=f'{rep}/{arm}/{family}/{question}';table=report['tables'][key]
                    triples=[];items=[]
                    for b in bs:
                        prefix=f"{rep}/{arm}/{family}/{b['id']}/{question}/"
                        a,c=(keyed[prefix+str(p)]['result'] for p in (0,3))
                        assert all(a[k]==c[k] for k in ('choiceLogits','choiceMass','rawTokenId','rawChoice'));shams+=1
                        group=[independent[prefix+str(p)] for p in range(3)]
                        triples.append([m['rawScore'] for m in group]);items.extend(group)
                    z=np.array(triples);shift=np.array([statistics.mean(t[1:])-t[0] for t in triples])
                    for k in ('correct','firstTokenCorrect','firstTokenIsOption','choiceMass','brier'):
                        check(table[k],statistics.mean(m[k] for m in items))
                    acc=[statistics.mean(m['correct'] for m in items if m['target']==y) for y in (0,1)]
                    check(table['correctByClass'],acc);check(table['balancedAccuracy'],statistics.mean(acc))
                    check(table['rawShift'],statistics.mean(shift));check(table['rawShiftInterval95'],interval(boot_mean(shift)))
                    raw,raw_boot=area(triples)
                    check(table['rawPresenceAUROC'],raw);check(table['rawPresenceAUROCInterval95'],interval(raw_boot));check(table['rawScores'],triples)
                    result=dict(shift=shift,balancedAccuracy=statistics.mean(acc))
                    if question in ('presence','absence'):
                        oriented=z*(1 if question=='presence' else -1);point,samples=area(oriented)
                        check(table['orientedPresenceAUROC'],point);check(table['orientedPresenceAUROCInterval95'],interval(samples))
                        paired=statistics.mean(float(row[i]>row[0])+.5*float(row[i]==row[0]) for row in oriented for i in (1,2))
                        check(table['withinBlockOrdering'],paired);result.update(auc=point,samples=samples)
                    else:
                        for y in (0,1):
                            selected=[i for i,b in enumerate(bs) if label(dict(question=question,position=0),b)==y]
                            check(table[f'rawShiftTarget{y}'],statistics.mean(shift[i] for i in selected))
                        for field,metric in [('probabilityOneShift','probabilityOne'),('accuracyChange','correct')]:
                            data=np.array([m[metric] for m in items],dtype=float).reshape(n,3)
                            effect=np.array([statistics.mean(v[1:])-v[0] for v in data])
                            check(table[field],statistics.mean(effect));check(table[field+'Interval95'],interval(boot_mean(effect)))
                    results[(rep,arm,family,question)]=result
        for c in [c for c in report['contrasts'] if c['replication']==rep]:
            if c['kind']=='orientedAUROC':
                a=results[(rep,'strong','hidden',c['question'])]
                b=results[(rep,c['other'],'hidden',c['question'])] if c['other']!='chance' else dict(auc=.5,samples=.5)
                point=a['auc']-b['auc'];samples=a['samples']-b['samples']
            else:
                a=results[(rep,c['arm'],c['family'],'presence')]['shift']
                q='absence' if c['kind']=='mappingInteraction' else c['question']
                b=results[(rep,c['arm'],c['family'],q)]['shift']
                point=statistics.mean(a-b);samples=boot_mean(a-b)
            check(c['difference'],point);ci=interval(samples);check(c['interval95'],ci)
            if c['primary']:contrasts.append(ci[0]>0)
        gate=all(results[(rep,a,'visible',q)]['balancedAccuracy']>=.90 for a in ('visible','strong') for q in ('presence','absence'))
        gates.append(gate)
    assert report['mappingSignalRuleMet']==all(contrasts)
    assert report['visibleInstructionGateMet']==all(gates)
    assert report['fixedReadingRuleMet']==(all(contrasts) and all(gates))
    assert shams==report['shamPairs']
    traces=[r['result']['intervention'] for r in rows]
    result=dict(schema='menia-presence-specificity-verification-v1',origin=h['metadata']['origin'],complete=True,
        journalSHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),planHash=h['planHash'],sourceHash=h['sourceHash'],
        independentMaxDifference=max_error,summaryMaxDifference=summary_error,evaluations=len(rows),shamPairs=shams,
        trainingEvents=sum(e['event'].startswith('training') for e in events),
        errors=report['errors'],interruptedRequests=report['interruptedRequests'],
        maxNormRelativeError=max(t['normRelativeError'] for t in traces),
        inputTokenRange=[min(r['inputTokens'] for r in rows),max(r['inputTokens'] for r in rows)],
        firstTokenOutsideOptions=sum(not m['firstTokenIsOption'] for m in independent.values()),
        mappingSignalRuleMet=all(contrasts),visibleInstructionGateMet=all(gates),fixedReadingRuleMet=all(contrasts) and all(gates),
        limitations='Shares strict lineage/order reader and fixed plan; independently recomputes targets, metrics, bootstrap and decision. Not an external replication, hardware attestation, or consciousness test.')
    Path(destination).write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':print(json.dumps(audit(sys.argv[1],sys.argv[2]),indent=2,ensure_ascii=False))
