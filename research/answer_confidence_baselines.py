"""Post-answer controls fitted only on fresh calibration for each producer.

External baselines, not the native confidence signal or an action controller.
The complete collection/analysis protocol is still under preparation.
"""
import math
import re

from research.activation_monitor import ridge_fit,predict
from research.answer_confidence_crossed import ARMS
from research.answer_confidence_plan import COUNTS
from research.cross_model_prediction import CELLS,digest,grade
from research.iphone_coupling_report import require
from research.natural_error_journal import validate_result

RIDGE_ALPHA=1.


def features(task,result):
    require(result['status']=='ok','Failed answer cannot be omitted')
    trace=result['trace'];c=trace['preAnswer']
    # Reuse the trace's full-vocabulary and token arithmetic checks, without
    # requiring intermediate activation capture from the older experiment.
    validate_result(result,dict(preAnswer=c))
    require(type(c['vocabularySize']) is int and c['vocabularySize']>=2,'Vocabulary')
    for key in ('maxProbability','topTwoMargin','normalizedEntropy'):
        require(type(c[key]) in (int,float) and math.isfinite(c[key]) and 0<=c[key]<=1+1e-10,'Invalid confidence feature')
    require(c['topTwoMargin']<=c['maxProbability'],'Probability margin')
    cell=(task['family'],task['level']);require(cell in CELLS,'Unknown category')
    return [float(cell==x) for x in CELLS]+[
        c['maxProbability'],c['topTwoMargin'],c['normalizedEntropy'],
        trace['completion']['meanLogProbability'],math.log1p(result['metrics']['outputTokens']),
        float(re.fullmatch(r'-?(?:0|[1-9][0-9]*)',result['text'].strip()) is not None)]


def fit(rows,replication):
    selected=[r for r in rows if r['task']['replication']==replication and r['task']['split']=='calibration']
    require(len(selected)==COUNTS['calibration']*len(CELLS),'Incomplete calibration')
    require(len({r['task']['id'] for r in selected})==len(selected),'Duplicate calibration question')
    for cell in CELLS:
        require(sum((r['task']['family'],r['task']['level'])==cell for r in selected)==COUNTS['calibration'],'Calibration strata')
    models={};beta={}
    for producer in ARMS:
        xs=[features(r['task'],r['answers'][producer]) for r in selected]
        ys=[int(grade(r['answers'][producer],r['task'])) for r in selected]
        models[producer]=ridge_fit(xs,ys,RIDGE_ALPHA)
        beta[producer]={}
        for family,level in CELLS:
            ids=[i for i,r in enumerate(selected) if (r['task']['family'],r['task']['level'])==(family,level)]
            beta[producer][f'{family}/{level}']=(sum(ys[i] for i in ids)+1)/(len(ids)+2)
    return dict(schema='menia-answer-confidence-baselines-v1',replication=replication,fitDataHash=digest(selected),
                calibrationQuestions=len(selected),models=models,beta=beta,
                scope='Fresh calibration labels for each current answer producer; fixed ridge alpha, no test selection.')


def forecast(bundle,task,result,producer):
    require(task['replication']==bundle['replication'] and producer in ARMS,'Mismatched baseline')
    xs=features(task,result)
    return dict(betaCell=bundle['beta'][producer][f"{task['family']}/{task['level']}"],
                outputConfidence=float(predict(bundle['models'][producer],xs)))
