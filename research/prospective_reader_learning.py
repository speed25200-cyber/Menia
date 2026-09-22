"""Frozen matched learning plan: actual retained state versus public-token replay."""
import hashlib
import json
from pathlib import Path

from research import prospective_pair_discovery as parent
from research import retained_reader_control as reader_control
from research.prospective_learning_data import partition
from research.cross_model_prediction import digest

ROOT=Path(__file__).resolve().parents[1]
PREPARATION=ROOT/'artifacts/prospective-reader-learning-preparation/design.json'
FILES=tuple(dict.fromkeys(reader_control.FILES+('prospective_learning_data.py','prospective_reader_learning.py',
    'prospective_reader_learning_gpu.py','prospective_reader_learning_audit.py','prospective_reader_learning_metrics.py',
    'audit_generation_numerics.py','audit_prospective_binding_discovery.py')))
ARMS=('base','state','text')
TRAIN_CONDITIONS=('actual','values_permuted','joint_permuted')


def source_hash():
    return digest({n:Path(__file__).with_name(n).read_text(encoding='utf-8') for n in FILES})


def plan():
    old=parent.plan(); data=partition()
    return dict(schema='menia-prospective-reader-learning-plan-v1',model=old['model'],settings=old['settings'],
        maxContextTokens=2048,cases=data['cases'],partitionHash=digest(data),conditions=old['conditions'],
        trainConditions=list(TRAIN_CONDITIONS),arms=list(ARMS),
        forecastMappings=old['forecastMappings'],taskMapping=old['taskMapping'],
        adapter=dict(rank=32,scale=1.0,seed=2026092070),
        optimizer=dict(type='AdamW',lr=5e-5,betas=[.9,.999],eps=1e-8,weight_decay=0,epochs=4,batch=16,clipNorm=1.0),
        exportCacheCases=[1000,1032],
        counts=dict(states=48,trainTasks=768,trainingExamplesPerArm=3584,updatesPerArm=224,
            trainForecasts=1024,trainCoding=256,reservedForecasts=3840,reservedCoding=960,reservedTasks=640),
        inputViews=dict(base='condition-specific producer cache',state='condition-specific producer cache',
            text='same-schedule replay of public raw tokens, identical across hidden conditions'),
        primary=dict(split='reserved',condition='values_permuted',mappings=[0,1],comparators=['base','text'],
            metric='mean squared conditional-code error; invalid native report assigned loss 1',
            minimumMeanGain=.02,minimumNativeFraction=.95,minimumSuppliedCodingFraction=.95,
            minimumPositiveTasks=16,minimumNegativeTasks=16,minimumNativeTaskFraction=1.0,
            discrimination='Within-table AUROC on mixed-outcome tables; any invalid forecast makes that table score zero',
            discriminationComparator='text',minimumMixedTables=8,minimumStateAUROC=.70,minimumAUROCGain=.10,
            bootstrapUnit='table',resamples=20000,seed=2026092072,familyAlpha=.05,comparisons=6,
            interval='two-sided percentile Bonferroni; four loss and two within-table AUROC comparisons'),
        ordering='Training states and task labels; identical initialization and four fixed epochs per arm; training diagnostics; reserved states; all reserved coding and forecasts; freeze; only then reserved tasks.',
        failurePolicy='Single attempt; preserve technical failures. Keep behavioral failures, invalid outputs and every table. No early stopping, best checkpoint, retries or prompt selection.',
        scope='One optimization seed, disjoint assignments of eight fixed keys in one synthetic task. Tests functional prospective prediction from hidden state; no decision utility, subjective-consciousness or architectural-novelty criterion.')


def cases(split,p=None):
    p=plan() if p is None else p
    return [c for c in p['cases'] if c['split']==split]


def training_examples(epoch,p=None):
    p=plan() if p is None else p; assert 0<=epoch<p['optimizer']['epochs']
    rows=[]
    for case in cases('train',p):
        for condition in TRAIN_CONDITIONS:
            for target in range(8):
                rows.append(dict(kind='forecast',case=case['id'],target=target,condition=condition,
                    mapping=(case['id']+target+epoch)%2))
        for verdict in (0,1):
            for mapping in (0,1):
                rows.append(dict(kind='coding',case=case['id'],verdict=verdict,mapping=mapping,condition='actual'))
    return sorted(rows,key=lambda r:digest(['reader-learning-order-v1',epoch,r]))


def calls(kind,split,arm=None,p=None):
    p=plan() if p is None else p
    for case in cases(split,p):
        conditions=p['conditions'] if split=='reserved' else (
            list(TRAIN_CONDITIONS) if kind=='task' else ['values_permuted'] if kind=='forecast' else ['actual'])
        for target in range(2 if kind=='coding' else 8):
            mappings=[None] if kind=='task' else [(case['id']+target)%2,1-(case['id']+target)%2]
            for mapping in mappings:
                shift=(case['id']+target+(mapping or 0))%len(conditions)
                for condition in conditions[shift:]+conditions[:shift]:
                    call=dict(case=case['id'],split=split,condition=condition)
                    call['verdict' if kind=='coding' else 'target']=target
                    if mapping is not None:call['mapping']=mapping
                    if arm is not None:call['arm']=arm
                    yield call


def operations(p=None):
    p=plan() if p is None else p
    yield from (('state',dict(case=c['id'],split='train')) for c in cases('train',p))
    yield from (('task',call) for call in calls('task','train',p=p))
    yield 'weights',dict(arm='initial')
    for arm in ('state','text'):
        yield 'reset',dict(arm=arm)
        for epoch in range(p['optimizer']['epochs']):
            rows=training_examples(epoch,p); assert len(rows)%p['optimizer']['batch']==0
            for batch in range(len(rows)//p['optimizer']['batch']):
                yield 'step',dict(arm=arm,epoch=epoch,batch=batch)
        yield 'weights',dict(arm=arm)
        for kind in ('coding','forecast'):
            yield from ((kind,call) for call in calls(kind,'train',arm,p))
    yield from (('state',dict(case=c['id'],split='reserved')) for c in cases('reserved',p))
    for kind in ('coding','forecast'):
        for arm in ARMS:
            yield from ((kind,call) for call in calls(kind,'reserved',arm,p))
    yield 'forecasts_complete',dict(count=p['counts']['reservedForecasts'])
    yield from (('task',call) for call in calls('task','reserved',p=p))


def memory_condition(arm,condition):
    assert arm in ARMS
    return 'same_schedule' if arm=='text' else condition


def branch_name(kind,call):
    return parent.branch_name(kind,call)


def target_label(kind,call,tasks,case_map):
    if kind=='coding':return call['verdict']
    d=tasks[call['case'],call['target'],call['condition']]['decoded']
    bit=case_map[call['case']]['values'][call['target']]
    return int(d['validNativeResponse'] and d['decision']==('one' if bit else 'zero'))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();assert args.prepare
    p=plan();data=dict(plan=p,planHash=digest(p),sourceHash=source_hash())
    PREPARATION.parent.mkdir(parents=True,exist_ok=True)
    with PREPARATION.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:data[k] for k in ('planHash','sourceHash')}))
