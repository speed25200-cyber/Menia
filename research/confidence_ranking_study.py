"""Colab24: prospective within-category ranking, at fixed adapter capacity."""
import copy
from pathlib import Path

from research.confidence_ranking import ARMS, CONFIG, SEED, rng_for
from research.answer_confidence_study import FILES as PARENT_FILES, JUDGE_SETTINGS, request_for
from research.answer_confidence_plan import make_plan as confidence_questions
from research.natural_error_questions import exclusions, make_plan as natural_questions
from research.cross_model_prediction import CELLS, MODELS, SETTINGS, digest

MODEL_ARMS = ('base', *ARMS)
TEST_PER_CELL = 64
CRITERIA = dict(primaryProducers=['base', 'rank'], comparators=['ce', 'neutral', 'constant'],
                minimumAucGain=.05, minimumGainOverConstant=.10,
                bootstrapResamples=10000, familyAlpha=.05, primaryComparisons=18,
                minimumEachClass=20, minimumCellsWithFiveEach=3,
                minimumNativeTopCodeFraction=.95, minimumNativeMeanCodeMass=.5,
                minimumAnswerAccuracyChange=-.02)
FILES = tuple(sorted(set(PARENT_FILES + (
    'confidence_ranking.py', 'confidence_ranking_gpu.py', 'confidence_ranking_study.py',
    'confidence_ranking_journal.py', 'confidence_ranking_analysis.py',
    'confidence_ranking_learning_gpu.py'))))


def source_hash():
    return digest({n: Path(__file__).with_name(n).read_text(encoding='utf-8') for n in FILES})


def make_plan():
    excluded = exclusions() | {t['question'] for p in (natural_questions(), confidence_questions()) for t in p['tasks']}
    seen = set(excluded); rng = rng_for('fresh-test'); tasks = []
    for rep in range(3):
        for block in range(TEST_PER_CELL):
            cells = list(CELLS); rng.shuffle(cells)
            for family, level in cells:
                while True:
                    letters = ''.join(rng.choice('ABCD') for _ in range(level)) if family == 'countA' else ''
                    operands = [rng.randrange(10, 100) for _ in range(level)] if family == 'alternatingSum' else []
                    question = (f'Combien de lettres A contient cette chaîne : {letters} ?' if letters else
                                'Calcule '+''.join(('' if i == 0 else ' + ' if i % 2 == 0 else ' - ')+str(n)
                                                  for i, n in enumerate(operands))+'.')
                    if question not in seen: seen.add(question); break
                index = len(tasks)
                tasks.append(dict(id=index, replication=rep, split='test', block=block,
                                  family=family, level=level, letters=letters, operands=operands,
                                  question=question, seed=int(digest([SEED, 'answer', index])[:8], 16)))
    calls = []
    for rep in range(3):
        subset = [t for t in tasks if t['replication'] == rep]
        for producer in MODEL_ARMS:
            for task in subset:
                calls.append(dict(id=len(calls), kind='answer', task=task['id'], replication=rep,
                                  split='test', producer=producer, modelArm=producer, seed=task['seed']))
        for judge in MODEL_ARMS:
            for producer in MODEL_ARMS:
                for task in subset:
                    calls.append(dict(id=len(calls), kind='judge', task=task['id'], replication=rep,
                                      split='test', producer=producer, judge=judge, modelArm=judge))
    return dict(schema='menia-confidence-ranking-plan-v1', status='Fixed before collection',
                model=copy.deepcopy(MODELS['A']), training=copy.deepcopy(CONFIG), seed=SEED,
                generation=copy.deepcopy(SETTINGS), judgeSettings=copy.deepcopy(JUDGE_SETTINGS),
                criteria=copy.deepcopy(CRITERIA), arms=list(MODEL_ARMS), replications=3,
                testPerCell=TEST_PER_CELL, tasks=tasks, calls=calls, plannedCalls=len(calls),
                plannedGenerations=len(tasks)*4, plannedJudgments=len(tasks)*16,
                trainingUnits=[dict(key=f'r{r}-{a}', replication=r, arm=a, initializationSeed=SEED+1001+r)
                               for r in range(3) for a in ARMS],
                excludedQuestions=len(excluded), excludedQuestionsHash=digest(sorted(excluded)),
                trainedAdapters=9, initialAdapters=3, trainingUpdates=1296,
                calibration='None: uncalibrated native ranking is primary; Brier is descriptive',
                scope='Post-answer textual assessment. No original-generation state, '
                      'capacity comparison, privileged access, native action or consciousness criterion.')


def evaluation_rows(plan, answers, judgments, replication):
    return [dict(task=t, answers={a: answers[(t['id'], a)] for a in MODEL_ARMS},
                 judgments={g: {a: judgments[(t['id'], g, a)]['scores'] for a in MODEL_ARMS} for g in MODEL_ARMS})
            for t in plan['tasks'] if t['replication'] == replication]
