"""Check the post-hoc diagnostic using a profile solver, not its Newton code."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import scipy
from scipy.optimize import brentq,minimize_scalar
from scipy.special import expit


def verify(journal,report):
    raw=Path(journal).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==report['parentJournalSHA256']
    events=[json.loads(line)['payload'] for line in raw.splitlines()]
    tasks={t['id']:t for t in events[0]['plan']['tasks']}
    requests={e['call']['id']:e['call'] for e in events if e['event']=='request'}
    answers={};scores={}
    for e in events:
        if e['event']=='answer':
            c=requests[e['id']];answers[(c['task'],c['producer'])]=e['text']
        elif e['event']=='judgment':
            c=requests[e['id']];scores[(c['task'],c['producer'],c['judge'])]=e['scores']['conditionalCorrect']
    def correct(task,producer):
        answer=answers[(task['id'],producer)].strip()
        truth=task['letters'].count('A') if task['family']=='countA' else sum((1 if i%2==0 else -1)*n for i,n in enumerate(task['operands']))
        return int(re.fullmatch(r'-?(?:0|[1-9][0-9]*)',answer) is not None and int(answer)==truth)
    results={};max_probability=0.;max_objective=0.;max_partition=0.
    for key,fit in report['calibrators'].items():
        calibration=[t for t in tasks.values() if t['split']=='calibration' and t['replication']==fit['replication']]
        assert len(calibration)==96 and [t['id'] for t in calibration]==fit['calibrationIds']
        y=np.asarray([correct(t,fit['producer']) for t in calibration],dtype=float)
        p=np.asarray([scores[(t['id'],fit['producer'],fit['judge'])] for t in calibration])
        clipped=np.clip(p,1e-12,1-1e-12);odds=np.log(clipped/(1-clipped))
        mean=float(odds.mean());scale=float(odds.std());scale=scale if scale>=1e-8 else 1.
        assert abs(mean-fit['mean'])<1e-10 and abs(scale-fit['scale'])<1e-10
        x=(odds-mean)/scale;n=len(y);prior=float(y.mean())
        def profile(b,with_intercept=False):
            bound=b*float(np.max(np.abs(x)))+40.
            a=brentq(lambda a:float(np.mean(expit(a+b*x))-prior),-bound,bound,xtol=1e-12,rtol=1e-14)
            z=a+b*x
            loss=float(np.where(y==1,np.logaddexp(0.,-z),np.logaddexp(0.,z)).mean()+b*b/(2*n))
            return (loss,a) if with_intercept else loss
        entropy=-prior*math.log(prior)-(1-prior)*math.log1p(-prior)
        bound=math.sqrt(2*n*entropy)
        found=minimize_scalar(profile,bounds=(0.,bound),method='bounded',options=dict(xatol=1e-11,maxiter=1000))
        assert found.success
        b=float(found.x) if found.fun<profile(0.) else 0.
        loss,a=profile(b,True)
        objective_difference=abs(loss-fit['objective']);max_objective=max(max_objective,objective_difference)
        assert objective_difference<1e-11
        test=[t for t in tasks.values() if t['split']=='test' and t['replication']==fit['replication']]
        target=np.asarray([correct(t,fit['producer']) for t in test])
        raw_p=np.asarray([scores[(t['id'],fit['producer'],fit['judge'])] for t in test])
        q=np.clip(raw_p,1e-12,1-1e-12);z=(np.log(q/(1-q))-mean)/scale
        predicted=expit(a+b*z)
        saved=report['results'][f"r{fit['replication']}/{fit['producer']}"]['judges'][fit['judge']]
        difference=float(np.max(np.abs(predicted-saved['calibratedProbabilities'])))
        max_probability=max(max_probability,difference);assert difference<1e-6
        assert abs(float(np.mean((predicted-target)**2))-saved['calibrated']['brier'])<1e-6
        cats=np.asarray([f"{t['family']}/{t['level']}" for t in test])
        positive=raw_p[target==1];negative=raw_p[target==0]
        concordance=(positive[:,None]>negative).astype(float)+.5*(positive[:,None]==negative)
        same=cats[target==1,None]==cats[None,target==0]
        part=saved['aucPartition'];assert part['pairs']==concordance.size
        for name,mask in [('within',same),('between',~same)]:
            count=int(mask.sum());assert part[name]['pairs']==count
            assert part[name]['twiceConcordance']==int(2*concordance[mask].sum())
            value=float(concordance[mask].mean()) if count else None
            if value is None:assert part[name]['auc'] is None
            else:max_partition=max(max_partition,abs(value-part[name]['auc']))
        results[key]=dict(profileIntercept=a,profileSlope=b,objectiveDifference=objective_difference,
                          maxTestProbabilityDifference=difference,independentBrier=float(np.mean((predicted-target)**2)))
    assert len(results)==27 and max_partition<1e-12
    return dict(schema='menia-confidence-calibration-audit-v1',verified=True,calibrators=27,
                parentJournalSHA256=report['parentJournalSHA256'],diagnosticSourceHash=report['sourceHash'],
                auditSourceSHA256=hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest(),
                numpy=np.__version__,scipy=scipy.__version__,maxObjectiveDifference=max_objective,
                maxProbabilityDifference=max_probability,maxAUCPartitionDifference=max_partition,checks=results,
                scope='Separate grading, vectorized AUC partition and scalar profile optimization with Brent roots. Same archived scores and calibration recipe; no independent experimental replication.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('journal',type=Path);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=verify(a.journal,json.loads(a.report.read_text(encoding='utf-8')))
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if a.output.exists():assert a.output.read_text(encoding='utf-8')==text
    else:a.output.write_text(text,encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','scope')},indent=2))
