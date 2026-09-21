"""Post-hoc diagnostic of Colab23; never replaces its native primary criterion.

Decompose AUROC pairs by task category. Fit 27 external monotone logistic
calibrators on calibration only, with one fixed regularization and no search.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from research.answer_confidence_analysis import metrics,probabilities
from research.answer_confidence_crossed import ARMS
from research.answer_confidence_journal import read_journal
from research.answer_confidence_study import evaluation_rows
from research.audit_composition_diagnostic import compare
from research.cross_model_prediction import CELLS,digest,grade
from research.iphone_coupling_report import require

FLOOR=1e-12
GRADIENT_TOLERANCE=1e-9
MAX_ITERATIONS=100


def validate(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    require(y.ndim==1 and y.shape==p.shape and len(y)>0,'Nonempty matching vectors required')
    require(np.isfinite(p).all() and np.all((p>=0)&(p<=1)) and np.all((y==0)|(y==1)),'Invalid truth or probability')
    return y,p


def auc_partition(y,p,categories):
    y,p=validate(y,p);require(len(categories)==len(y),'Category count')
    parts={name:dict(pairs=0,twiceConcordance=0) for name in ('within','between')}
    for i in np.flatnonzero(y==1):
        for j in np.flatnonzero(y==0):
            part=parts['within' if categories[i]==categories[j] else 'between']
            part['pairs']+=1
            part['twiceConcordance']+=2*int(p[i]>p[j])+int(p[i]==p[j])
    total=sum(v['pairs'] for v in parts.values());score=sum(v['twiceConcordance'] for v in parts.values())
    for v in parts.values():
        v['auc']=v['twiceConcordance']/(2*v['pairs']) if v['pairs'] else None
        v['pairFraction']=v['pairs']/total if total else None
        v['contributionToGlobalAUC']=v['twiceConcordance']/(2*total) if total else None
    return dict(pairs=total,auc=score/(2*total) if total else None,**parts)


def log_odds(p):
    p=np.asarray(p,dtype=float)
    require(np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Invalid probability')
    clipped=np.clip(p,FLOOR,1-FLOOR)
    return np.log(clipped)-np.log1p(-clipped)


def sigmoid(z):
    return np.exp(-np.logaddexp(0.,-np.asarray(z,dtype=float)))


def objective(theta,x,y,penalty):
    a,b=theta;z=a+b*x;p=sigmoid(z)
    value=float(np.mean(np.logaddexp(0.,z)-y*z)+.5*penalty*b*b)
    design=np.column_stack((np.ones(len(x)),x))
    grad=design.T@(p-y)/len(x)+np.asarray([0.,penalty*b])
    hessian=design.T@((p*(1-p))[:,None]*design)/len(x)+np.diag([0.,penalty])
    return value,grad,hessian


def fit_monotone(y,p):
    y,p=validate(y,p)
    require(0<y.sum()<len(y),'Both calibration classes required')
    odds=log_odds(p);mean=float(odds.mean());scale=float(odds.std())
    if scale<1e-8:scale=1.
    x=(odds-mean)/scale;penalty=1/len(y)
    theta=np.asarray([math.log(float(y.mean())/(1-float(y.mean()))),0.])
    for iteration in range(MAX_ITERATIONS):
        value,grad,hessian=objective(theta,x,y,penalty)
        residual=max(abs(grad[0]),abs(grad[1]) if theta[1]>0 else max(0.,-grad[1]))
        if residual<=GRADIENT_TOLERANCE:break
        direction=np.linalg.solve(hessian,grad);step=1.
        for attempt in range(60):
            candidate=theta-step*direction;candidate[1]=max(0.,candidate[1])
            change=candidate-theta
            if objective(candidate,x,y,penalty)[0]<=value+1e-4*float(grad@change):
                theta=candidate;break
            step*=.5
        else:raise ValueError('Calibration line search failed')
    else:raise ValueError('Calibration did not converge')
    return dict(intercept=float(theta[0]),slope=float(theta[1]),mean=mean,scale=scale,
                penalty=penalty,probabilityFloor=FLOOR,iterations=iteration,kktResidual=float(residual),
                objective=value,calibrationCount=len(y),boundary=bool(theta[1]==0))


def predict_calibrated(fit,p):
    require(fit['probabilityFloor']==FLOOR and fit['slope']>=0 and fit['scale']>0,'Unsupported calibrator')
    return sigmoid(fit['intercept']+fit['slope']*(log_odds(p)-fit['mean'])/fit['scale'])


def fit_rows(rows,replication,producer,judge):
    require(producer in ARMS and judge in ARMS,'Unknown arm')
    selected=[r for r in rows if r['task']['split']=='calibration' and r['task']['replication']==replication]
    require(len(selected)==96 and len({r['task']['id'] for r in selected})==96,'Calibration count')
    require(all(sum((r['task']['family'],r['task']['level'])==c for r in selected)==16 for c in CELLS),'Calibration strata')
    values=[dict(task=r['task']['id'],correct=int(grade(r['answers'][producer],r['task'])),
                 probability=r['judgments'][judge][producer]['conditionalCorrect']) for r in selected]
    fit=fit_monotone([r['correct'] for r in values],[r['probability'] for r in values])
    return dict(**fit,replication=replication,producer=producer,judge=judge,fitDataHash=digest(values),
                calibrationIds=[r['task'] for r in values])


def run(journal):
    root=Path(__file__).resolve().parents[1];public=root/'artifacts/native-answer-confidence-pilot'
    receipt=json.loads((public/'receipt.json').read_text(encoding='utf-8'))
    verification=json.loads((public/'verification.json').read_text(encoding='utf-8'))
    original=json.loads((public/'summary.json').read_text(encoding='utf-8'))
    require(verification['verified'] and verification['weightsChecked'] and len(verification['weights'])==9,'Parent audit required')
    raw=Path(journal).read_bytes();identity=hashlib.sha256(raw).hexdigest()
    require(identity==receipt['journalSHA256']==verification['journalSHA256'],'Different parent journal')
    data=read_journal(journal,check_fit=False)
    require(data['complete'] and original['complete'] and data['recorded']==10368,'Incomplete parent')
    plan=data['header']['plan'];calibration=evaluation_rows(plan,data['answers'],data['judgments'],'calibration')
    test=evaluation_rows(plan,data['answers'],data['judgments'],'test')
    require(not {r['task']['question'] for r in calibration}&{r['task']['question'] for r in test},'Shared questions')
    # Fit every calibrator before consulting test labels in this analysis.
    fits={f'r{rep}/{producer}/{judge}':fit_rows(calibration,rep,producer,judge)
          for rep in range(3) for producer in ARMS for judge in ARMS}
    results={};max_difference=0.;max_parent_difference=0.
    for rep in range(3):
        rows=[r for r in test if r['task']['replication']==rep]
        categories=[f"{r['task']['family']}/{r['task']['level']}" for r in rows]
        for producer in ARMS:
            truth=[int(grade(r['answers'][producer],r['task'])) for r in rows]
            ps=probabilities(rows,data['bundles'][rep],producer);judges={}
            for judge,values in ps.items():
                partition=auc_partition(truth,values,categories)
                metrics_raw=metrics(truth,values)
                old=original['replications'][str(rep)]['scores'][producer][judge]
                max_parent_difference=max(max_parent_difference,compare(metrics_raw,old))
                if partition['auc'] is not None:
                    delta=abs(partition['auc']-old['auc']);max_difference=max(max_difference,delta)
                    require(delta<1e-12,'Pair decomposition differs from parent AUC')
                item=dict(raw=metrics_raw,aucPartition=partition)
                if judge in ARMS:
                    fit=fits[f'r{rep}/{producer}/{judge}'];pred=predict_calibrated(fit,values)
                    item.update(calibrated=metrics(truth,pred),calibratedPartition=auc_partition(truth,pred,categories),
                                calibratedProbabilities=pred.tolist(),rawProbabilities=list(values))
                judges[judge]=item
            results[f'r{rep}/{producer}']=dict(n=len(rows),correct=sum(truth),judges=judges)
    return dict(schema='menia-confidence-calibration-diagnostic-v1',origin='audited_colab23_reanalysis',
                exploratory=True,newLLMCalls=0,weightUpdates=0,sourceHash=digest(Path(__file__).read_text(encoding='utf-8')),
                parentJournalSHA256=identity,parentPrimaryCriterion=original['nativeConfidenceCriterion'],
                nativePrimaryCriterionUnchanged=True,calibrators=fits,results=results,
                maxAUCReconstructionDifference=max_difference,maxParentMetricDifference=max_parent_difference,fitUsesTestLabels=False,
                method=dict(probabilityFloor=FLOOR,penalty='1 / calibration count, on standardized-logit slope only',
                            slopeConstraint='nonnegative',gradientTolerance=GRADIENT_TOLERANCE,maximumIterations=MAX_ITERATIONS,
                            trials=27,selection='No hyperparameter or model selection; all arms and producers reported'),
                scope='Post-hoc descriptive diagnostic after parent outcomes were seen. External recalibration, not a native gain, new confirmation, privileged access, action test or consciousness evidence. AUC pairs are dependent and are not additional observations.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('journal',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=run(args.journal);text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():require(args.output.read_text(encoding='utf-8')==text,'Existing different diagnostic')
    else:args.output.write_text(text,encoding='utf-8')
    print(json.dumps(dict(exploratory=True,newLLMCalls=0,calibrators=len(report['calibrators']),maxAUCReconstructionDifference=report['maxAUCReconstructionDifference'],output=str(args.output))))
