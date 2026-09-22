"""Independent numerical readout of experiment 14; shared plan and strict event reader."""
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import fmean

import numpy as np

from research import optimizer_memory as s
from research.audit_composition_diagnostic import compare
from research.iphone_coupling_report import require,strict_json


def ci(xs):return np.quantile(xs,[.025,.975]).tolist()


def recompute(journal):
    h,rows,done,pending,events=s.read_journal(journal)
    require(pending is None and len(rows)==len(h['plan']['evaluation']),'Incomplete experiment')
    indexed={r['request']['id']:r['result'] for r in rows};choices=h['metadata']['choiceTokenIds']
    tables={};contrasts=[];shams=0
    for rep,split in itertools.product(range(s.CONFIG['replications']),('train','test','lexical')):
        bs=[b for b in h['plan']['blocks'] if b['replication']==rep and b['split']==split];n=len(bs)
        if not bs:continue
        rng=np.random.default_rng(s.SEED+rep*10+('train','test','lexical').index(split))
        draws=np.concatenate([rng.choice([i for i,b in enumerate(bs) if b['marker']==y],
            (s.CONFIG['resamples'],sum(b['marker']==y for b in bs))) for y in (0,1)],axis=1)
        forms=('trained',) if split=='train' else ('trained','paraphrase')
        streams=(('hidden','monitor'),) if split=='train' else (('hidden','monitor'),('visible','monitor'),('visible','marker_first'),('visible','marker_second'))
        native={};losses={}
        for arm,form,(family,task),mapping in itertools.product(s.ARMS,forms,streams,(0,1)):
            measurements=[];block_accuracy=[];block_loss=[];presence=[]
            class_correct={0:[],1:[]};class_ce={0:[],1:[]}
            for b in bs:
                key=f"{rep}/{arm}/{family}/{b['id']}/{form}/{task}/{mapping}/"
                a,c=(indexed[key+str(p)] for p in (0,3))
                require(all(a[k]==c[k] for k in ('choiceLogits','choiceMass','rawTokenId','rawChoice')),'Independent sham failure');shams+=1
                accuracies=[];ces=[];scores=[]
                for pos in (0,1,2):
                    out=indexed[key+str(pos)]
                    truth=int(pos!=0) if task=='monitor' else int(b['marker']==int(task=='marker_second'))
                    digit=truth^mapping;z=out['choiceLogits']
                    numerators=[math.exp(x-max(z)) for x in z];total=math.fsum(numerators)
                    correct=int(out['rawTokenId']==choices[digit])
                    ce=-math.log(out['choiceMass'])+math.log(total)+max(z)-z[digit]
                    pair=[math.exp(x-max(z[:2])) for x in z[:2]];pair=[x/math.fsum(pair) for x in pair]
                    measurements.append(dict(firstTokenCorrect=correct,firstTokenIsOption=int(out['rawTokenId'] in choices[:2]),
                        brier=math.fsum((pair[i]-int(i==digit))**2 for i in (0,1)),
                        choiceMass=out['choiceMass']*math.fsum(numerators[:2])/total,fourDigitMass=out['choiceMass']))
                    accuracies.append(correct);ces.append(ce);class_correct[truth].append(correct);class_ce[truth].append(ce)
                    scores.append((z[1]-z[0])*(-1 if mapping else 1))
                balance=lambda x:(x[0]+fmean(x[1:]))/2 if task=='monitor' else fmean(x)
                block_accuracy.append(balance(accuracies));block_loss.append(balance(ces));presence.append(scores)
            k=(arm,form,family,task,mapping);acc=np.array(block_accuracy);ce=np.array(block_loss)
            native[k]=acc;losses[k]=ce
            table={k:fmean(m[k] for m in measurements) for k in measurements[0]}
            table.update(blocks=n,n=3*n,nativeBalancedAccuracy=fmean(acc),nativeInterval95=ci(acc[draws].mean(1)),
                balancedCrossEntropy=fmean(ce),crossEntropyInterval95=ci(ce[draws].mean(1)))
            for label,bit in (('false',0),('true',1)):
                table[label+'Accuracy']=fmean(class_correct[bit]);table[label+'CrossEntropy']=fmean(class_ce[bit])
            if task=='monitor':
                wins=np.array([[sum(float(v>absent[0])+.5*float(v==absent[0]) for v in present[1:])
                                for present in presence] for absent in presence])
                table['presenceAUROC']=float(wins.mean()/2)
                table['presenceAUROCInterval95']=ci(wins[draws[:,:,None],draws[:,None,:]].mean((1,2))/2)
            tables[f'{rep}/{split}/{arm}/{family}/{form}/{task}/{mapping}']=table
        for form,(family,task),mapping in itertools.product(forms,streams,(0,1)):
            for a,b in (('carry','prefix'),('zero_grad','prefix'),('reset_m','carry')):
                k=lambda arm:(arm,form,family,task,mapping)
                diff=native[k(a)]-native[k(b)];dc=losses[k(a)]-losses[k(b)]
                contrasts.append(dict(replication=rep,split=split,format=form,family=family,task=task,mapping=mapping,
                    comparison=a+'-'+b,primary=split=='test' and form=='trained' and family=='hidden' and task=='monitor',
                    nativeDifference=fmean(diff),nativeInterval95=ci(diff[draws].mean(1)),
                    crossEntropyDifference=fmean(dc),crossEntropyInterval95=ci(dc[draws].mean(1))))
    return dict(tables=tables,contrasts=contrasts,shamPairs=shams,recorded=len(rows),planned=len(h['plan']['evaluation']),
        trained={k:e['sha256'] for k,e in done.items()},trainingUpdates=sum(e['event']=='training_step' for e in events),
        errors=sum(e['event']=='error' for e in events),trainingRestarts=sum(e['event']=='training_restart' for e in events),
        interruptedRequests=sum(e['event']=='interrupted_request' for e in events),
        evaluationSeconds=math.fsum(r['result']['seconds'] for r in rows))


def verify(journal,report):
    require(report['complete'],'Incomplete summary');fresh=recompute(journal)
    delta=compare(fresh,{k:report[k] for k in fresh})
    h,_,_,_,_=s.read_journal(journal)
    require(all(report[k]==h[k] for k in ('planHash','sourceHash')) and report['origin']==h['metadata']['origin'],'Provenance changed')
    return dict(schema='menia-optimizer-memory-verification-v1',verified=True,origin=report['origin'],
        journalSHA256=hashlib.sha256(Path(journal).read_bytes()).hexdigest(),planHash=h['planHash'],sourceHash=h['sourceHash'],
        tables=len(fresh['tables']),contrasts=len(fresh['contrasts']),shamPairs=fresh['shamPairs'],maxAbsoluteDifference=delta,
        scope='Independent target coding, full-vocabulary loss, raw-token accuracy, probabilities, AUROC, stratified paired bootstrap and contrasts. Shared strict reader and plan; not an external replication.')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('journal',type=Path)
    p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=verify(a.journal,strict_json(a.summary.read_text(encoding='utf-8')))
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
