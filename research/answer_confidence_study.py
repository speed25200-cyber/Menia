"""Colab23: fixed native post-answer confidence and matched current responses."""
import copy
from pathlib import Path

from research.answer_confidence_data import SEED,input_messages
from research.answer_confidence_plan import make_plan as prepared_plan,TRAINING
from research.answer_confidence_crossed import ARMS
from research.cross_model_prediction import digest,SOLVE
from research.iphone_coupling_report import require
from research.natural_error_journal import SOURCE_FILES as NATURAL_FILES
from research.value_action_learning import FILES as ACTION_FILES

FILES=tuple(sorted(set(NATURAL_FILES+ACTION_FILES+(
    'answer_confidence_data.py','answer_confidence_plan.py','answer_confidence_crossed.py',
    'answer_confidence_baselines.py','answer_confidence_gpu.py','answer_confidence_study.py',
    'answer_confidence_journal.py','answer_confidence_learning_gpu.py','answer_confidence_analysis.py',
    'audit_answer_confidence.py'))))
CRITERIA=dict(primaryProducers=['base','measured'],comparators=['base','shuffled','betaCell','outputConfidence'],
              minimumBrierGain=.005,bootstrapResamples=10000,familyAlpha=.05,primaryComparisons=24,
              minimumEachClass=20,minimumNativeTopCodeFraction=.95,minimumNativeMeanCodeMass=.5,
              minimumAnswerAccuracyChange=-.02)
JUDGE_SETTINGS=dict(maxInputTokens=1792,temperature=1.,codes=['0','1'],sampling=False,calibration='none')


def source_hash():
    return digest({n:Path(__file__).with_name(n).read_text(encoding='utf-8') for n in FILES})


def make_plan():
    p=copy.deepcopy(prepared_plan());p.update(schema='menia-native-answer-confidence-plan-v1',
        status='Fixed before collection',criteria=CRITERIA,judgeSettings=JUDGE_SETTINGS)
    units=[dict(key=f'r{rep}-{arm}',replication=rep,arm=arm,initializationSeed=SEED+1001+rep)
           for rep in range(3) for arm in ARMS[1:]]
    calls=[]
    for split in ('calibration','test'):
        for rep in range(3):
            tasks=[t for t in p['tasks'] if t['replication']==rep and t['split']==split]
            for producer in ARMS:
                for task in tasks:
                    calls.append(dict(id=len(calls),kind='answer',task=task['id'],replication=rep,
                                      split=split,producer=producer,modelArm=producer,seed=task['seed']))
            for judge in ARMS:
                for producer in ARMS:
                    for task in tasks:
                        calls.append(dict(id=len(calls),kind='judge',task=task['id'],replication=rep,
                                          split=split,producer=producer,judge=judge,modelArm=judge))
    p.update(trainingUnits=units,calls=calls,plannedCalls=len(calls),calibrationCalls=sum(c['split']=='calibration' for c in calls))
    require(len(calls)==10368 and p['calibrationCalls']==3456,'Collection budget')
    return p


def request_for(plan,index,answers,checkpoints):
    c=plan['calls'][index];t=plan['tasks'][c['task']]
    if c['kind']=='answer':
        messages=[dict(role='system',content=SOLVE),dict(role='user',content=t['question'])]
        settings=plan['generation']
    else:
        answer=answers[(t['id'],c['producer'])]
        require(answer['status']=='ok','Judge requires an actual successful generation')
        messages=input_messages(t['question'],answer['text']);settings=plan['judgeSettings']
    key=f'r{c["replication"]}-{c["modelArm"]}'
    return dict(event='request',call=c,messages=messages,settings=settings,
                adapterHash=None if c['modelArm']=='base' else checkpoints[key]['sha256'])


def evaluation_rows(plan,answers,judgments,split):
    rows=[]
    for t in plan['tasks']:
        if t['split']!=split:continue
        rows.append(dict(task=t,answers={p:answers[(t['id'],p)] for p in ARMS},
                         judgments={g:{p:judgments[(t['id'],g,p)]['scores'] for p in ARMS} for g in ARMS}))
    return rows
