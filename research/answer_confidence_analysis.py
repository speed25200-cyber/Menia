"""Fixed Brier gains, whole-question uncertainty and answer-drift controls."""
import math
import numpy as np

from research.answer_confidence_study import CRITERIA,evaluation_rows
from research.answer_confidence_plan import QUESTION_SEED,COUNTS
from research.answer_confidence_crossed import ARMS,summarize
from research.answer_confidence_baselines import forecast
from research.answer_confidence_journal import read_journal
from research.cross_model_prediction import CELLS,digest,grade
from research.iphone_coupling_report import require


def metrics(y,probabilities):
    y=np.asarray(y,dtype=float);p=np.asarray(probabilities,dtype=float)
    require(len(y)>0 and y.shape==p.shape and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Prediction array')
    good,bad=p[y==1],p[y==0]
    auc=float(np.mean((good[:,None]>bad)+.5*(good[:,None]==bad))) if len(good) and len(bad) else None
    bins=np.minimum((p*10).astype(int),9);reliability=[]
    for i in range(10):
        idx=bins==i
        reliability.append(dict(lower=i/10,upper=(i+1)/10,n=int(idx.sum()),
            meanProbability=float(p[idx].mean()) if idx.any() else None,accuracy=float(y[idx].mean()) if idx.any() else None))
    return dict(n=len(y),correct=int(y.sum()),brier=float(np.mean((p-y)**2)),auc=auc,
                meanProbability=float(p.mean()),reliability=reliability)


def probabilities(rows,bundle,producer):
    result={g:[r['judgments'][g][producer]['conditionalCorrect'] for r in rows] for g in ARMS}
    controls=[forecast(bundle,r['task'],r['answers'][producer],producer) for r in rows]
    result.update({k:[c[k] for c in controls] for k in ('betaCell','outputConfidence')})
    return result


def matched_quality(rows):
    out={}
    for judge in ARMS:
        for producer in ARMS:
            if producer==judge:continue
            for correct in (False,True):
                subset=[r for r in rows if grade(r['answers'][judge],r['task'])==correct and grade(r['answers'][producer],r['task'])==correct]
                own=[r['judgments'][judge][judge] for r in subset];other=[r['judgments'][judge][producer] for r in subset]
                same=[i for i,(a,b) in enumerate(zip(own,other)) if a['inputHash']==b['inputHash']]
                different=[i for i in range(len(subset)) if i not in same]
                delta=[a['conditionalCorrect']-b['conditionalCorrect'] for a,b in zip(own,other)]
                out[f'{judge}/{producer}/{int(correct)}']=dict(n=len(subset),identicalInputs=len(same),differentInputs=len(different),
                    meanOwnMinusOther=math.fsum(delta)/len(delta) if delta else None,
                    meanOwnMinusOtherDifferent=math.fsum(delta[i] for i in different)/len(different) if different else None,
                    maxIdenticalInputScoreDifference=max((abs(delta[i]) for i in same),default=0.))
    return out


def analyze(path):
    d=read_journal(path);require(d['complete'] and d['fitChecked'],'Only complete reconstructed trials')
    p=d['header']['plan'];all_rows=evaluation_rows(p,d['answers'],d['judgments'],'test')
    results={};contrasts=[];gates={}
    for rep in range(3):
        rows=[r for r in all_rows if r['task']['replication']==rep];require(len(rows)==192,'Test budget')
        indexes=[np.asarray([i for i,r in enumerate(rows) if (r['task']['family'],r['task']['level'])==c]) for c in CELLS]
        require(all(len(ids)==COUNTS['test'] for ids in indexes),'Test strata')
        rng=np.random.default_rng(QUESTION_SEED+90000+rep)
        draws=np.concatenate([rng.choice(ids,(CRITERIA['bootstrapResamples'],len(ids)),replace=True) for ids in indexes],axis=1)
        scores={};within={};formats={};accuracy={}
        for producer in ARMS:
            y=np.asarray([int(grade(r['answers'][producer],r['task'])) for r in rows],dtype=float)
            predicted=probabilities(rows,d['bundles'][rep],producer)
            accuracy[producer]=float(y.mean());scores[producer]={name:metrics(y,ps) for name,ps in predicted.items()}
            within[producer]={f'{cell[0]}/{cell[1]}':{name:metrics(y[ids],np.asarray(ps)[ids]) for name,ps in predicted.items()} for cell,ids in zip(CELLS,indexes)}
            own=[r['judgments']['measured'][producer] for r in rows]
            formats[producer]=dict(topCodeFraction=sum(x['topIsCode'] for x in own)/len(own),meanCodeMass=math.fsum(x['candidateMass'] for x in own)/len(own))
            if producer in CRITERIA['primaryProducers']:
                loss=(np.asarray(predicted['measured'])-y)**2
                for name in CRITERIA['comparators']:
                    delta=(np.asarray(predicted[name])-y)**2-loss
                    sampled=delta[draws].mean(axis=1);tail=CRITERIA['familyAlpha']/(2*CRITERIA['primaryComparisons'])
                    bounds=np.quantile(sampled,[tail,1-tail]).tolist();gain=float(delta.mean())
                    contrasts.append(dict(replication=rep,producer=producer,comparison=name+'-measured',brierGain=gain,
                        interval95=np.quantile(sampled,[.025,.975]).tolist(),familyInterval=bounds,
                        passed=bool(gain>=CRITERIA['minimumBrierGain'] and bounds[0]>0)))
        matrix=summarize([dict(id=r['task']['id'],correct={a:int(grade(r['answers'][a],r['task'])) for a in ARMS},judgments=r['judgments']) for r in rows])
        enough=all(min(scores[a]['measured']['correct'],192-scores[a]['measured']['correct'])>=CRITERIA['minimumEachClass'] for a in CRITERIA['primaryProducers'])
        format_pass=all(formats[a]['topCodeFraction']>=CRITERIA['minimumNativeTopCodeFraction'] and formats[a]['meanCodeMass']>=CRITERIA['minimumNativeMeanCodeMass'] for a in CRITERIA['primaryProducers'])
        retained=accuracy['measured']-accuracy['base']>=CRITERIA['minimumAnswerAccuracyChange']
        gates[str(rep)]=dict(enoughBothClasses=enough,nativeCodeFormatPassed=format_pass,answerAccuracyRetained=retained,
                            allPrimaryPassed=all(c['passed'] for c in contrasts if c['replication']==rep))
        results[str(rep)]=dict(answerAccuracy=accuracy,scores=scores,withinCell=within,nativeMeasuredFormat=formats,
                              crossed=matrix,matchedQuality=matched_quality(rows),baselineHash=digest(d['bundles'][rep]))
    require(len(contrasts)==24,'Primary family size')
    return dict(schema='menia-native-answer-confidence-summary-v1',origin=d['header']['origin'],complete=True,
        planHash=d['header']['planHash'],sourceHash=d['header']['sourceHash'],journalChainEnd=d['chainEnd'],
        recordedCalls=d['recorded'],generations=len(d['answers']),judgments=len(d['judgments']),trainingSteps=len(d['steps']),
        baselineFits=len(d['bundles']),fitChecked=True,replications=results,contrasts=contrasts,gates=gates,
        nativeConfidenceCriterion=all(all(g.values()) for g in gates.values()),
        costs=dict(answerTokens=sum(a['metrics']['outputTokens'] for a in d['answers'].values()),
                   answerSeconds=math.fsum(a['seconds'] for a in d['answers'].values()),
                   judgmentSeconds=math.fsum(a['seconds'] for a in d['judgments'].values()),
                   answerTokenLimits=sum(a['metrics']['reachedTokenLimit'] for a in d['answers'].values()),
                   trainingInputTokens=sum(sum(s['inputTokens']) for s in d['steps']),
                   trainingSeconds=math.fsum(c['trainingSeconds'] for c in d['checkpoints'].values())),
        criteria=CRITERIA,scope='Native post-answer textual assessment. Approximate stratified question bootstrap on fixed tasks and one sampled response per producer. No privileged access, native action, consciousness or novelty conclusion.')
