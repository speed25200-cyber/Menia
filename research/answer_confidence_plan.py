"""Matched confidence training schedule and genuinely fresh Menia questions.

Preparatory plan, not an executable/frozen scientific experiment yet. No model
call, held-out answer, calibration fit, or performance-dependent selection.
"""
from collections import Counter
import json
from pathlib import Path
import random

from research.answer_confidence_crossed import ARMS
from research.answer_confidence_data import SEED, input_messages
from research.cross_model_prediction import CELLS, MODELS, SETTINGS, digest
from research.iphone_coupling_report import require
from research.natural_error_questions import exclusions, make_plan as parent_questions

TRAINING = dict(rank=8, scale=1., learningRate=1e-4, betas=[.9,.999],
                epsilon=1e-8, weightDecay=0., clipNorm=1., epochs=2,
                batchSize=8, updatesPerAdapter=144, maxTrainingTokens=1792,
                loss='Code 0/1 then EOS only; response tokens are context')
COUNTS = dict(calibration=16, test=32)
QUESTION_SEED = SEED+1


def validate_training(examples, report):
    require(set(examples)=={'measured','shuffled'}, 'Training arms')
    for arm, rows in examples.items():
        require(len(rows)==1728 and digest(rows)==report['exampleHashes'][arm], 'Prepared data identity')
        require(len({r['sourceId'] for r in rows})==1728, 'Repeated source')
        groups=Counter((r['replication'],r['family'],r['level']) for r in rows)
        require(groups==Counter({(r,*cell):96 for r in range(3) for cell in CELLS}), 'Training strata')
        for row in rows:
            require(row['split']=='train' and row['target'] in ('0','1'), 'Training split or target')
            messages=row['messages']
            require(messages==input_messages(messages[1]['content'],messages[2]['content']), 'Unexpected input')
    a,b=examples['measured'],examples['shuffled']
    require(digest([r['messages'] for r in a])==report['inputHash'], 'Input identity')
    require(len({r['messages'][1]['content'] for r in a})==1728, 'Repeated training question')
    for x,y in zip(a,b):
        require({k:v for k,v in x.items() if k!='target'}=={k:v for k,v in y.items() if k!='target'}, 'Unmatched arms')
    for rep in range(3):
        for family,level in CELLS:
            counts=[Counter(r['target'] for r in examples[arm] if (r['replication'],r['family'],r['level'])==(rep,family,level)) for arm in ('measured','shuffled')]
            require(counts[0]==counts[1], 'Shuffled label counts changed')


def load_training(directory):
    directory=Path(directory)
    report=json.loads((Path(__file__).parents[1]/'artifacts/answer-confidence-preparation/report.json').read_text(encoding='utf-8'))
    examples={arm:[json.loads(line) for line in (directory/(arm+'.jsonl')).read_text(encoding='utf-8').splitlines()] for arm in ('measured','shuffled')}
    validate_training(examples,report)
    return examples,report


def training_batches(examples, replication, arm):
    require(type(replication) is int and 0<=replication<3 and arm in ('measured','shuffled'), 'Training unit')
    rows=sorted((r for r in examples[arm] if r['replication']==replication),key=lambda r:r['sourceId'])
    require(len(rows)==576 and len({r['sourceId'] for r in rows})==576, 'Training unit size')
    batches=[]
    for epoch in range(TRAINING['epochs']):
        order=list(range(len(rows)))
        random.Random(int(digest([SEED,'training-order',replication,epoch])[:16],16)).shuffle(order)
        batches.extend([[rows[i] for i in order[start:start+8]] for start in range(0,len(rows),8)])
    require(len(batches)==144 and all(len(b)==8 for b in batches), 'Training budget')
    return batches


def make_plan():
    old=parent_questions()
    excluded=exclusions() | {t['question'] for t in old['tasks']}
    seen=set(excluded);rng=random.Random(QUESTION_SEED);tasks=[]
    for rep in range(3):
        for split,count in COUNTS.items():
            for block in range(count):
                cells=list(CELLS);rng.shuffle(cells)
                for family,level in cells:
                    while True:
                        letters=''.join(rng.choice('ABCD') for _ in range(level)) if family=='countA' else ''
                        operands=[rng.randrange(10,100) for _ in range(level)] if family=='alternatingSum' else []
                        question=(f'Combien de lettres A contient cette chaîne : {letters} ?' if letters else
                                  'Calcule '+''.join(('' if i==0 else ' + ' if i%2==0 else ' - ')+str(n) for i,n in enumerate(operands))+'.')
                        if question not in seen:seen.add(question);break
                    index=len(tasks)
                    tasks.append(dict(id=index,replication=rep,split=split,block=block,family=family,level=level,
                                      question=question,letters=letters,operands=operands,
                                      seed=int(digest([QUESTION_SEED,rep,index,'answer'])[:8],16)))
    return dict(schema='menia-answer-confidence-plan-draft-v1',status='Preparation only; collector and analysis criteria not yet frozen',
                model=MODELS['A'],training=TRAINING,generation=SETTINGS,questionSeed=QUESTION_SEED,
                arms=list(ARMS),replications=3,countsPerCell=COUNTS,tasks=tasks,
                excludedQuestions=len(excluded),excludedQuestionsHash=digest(sorted(excluded)),
                parentQuestionPlanHash=digest(old),trainingUpdates=864,trainedAdapters=6,initialAdapters=3,
                plannedGenerations=len(tasks)*3,plannedJudgments=len(tasks)*9,
                scope='Textual post-answer correctness. Fresh relative to listed Menia tasks, not pretraining. No privileged access or consciousness criterion.')
