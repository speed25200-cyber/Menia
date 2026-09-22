"""Fresh question partitions for a draft prospective error-prediction experiment.

No LLM calls, outcome inspection, monitor fitting, or finalized decision rule.
"""
import random

from research.activation_monitor import make_plan as activation_plan
from research.cross_model_prediction import CELLS, MODELS, SETTINGS, digest, make_plan as cross_plan
from research.output_confidence_validation import QUESTIONS as TECHNICAL_QUESTIONS
from research.perturbation_monitor import make_plan as perturbation_plan
from research.replay_controller import make_plan as replay_plan

SEED = 202609197
REPLICATIONS = 3
COUNTS = dict(train=96, validation=32, test=64)


def exclusions():
    return set(TECHNICAL_QUESTIONS) | {
        t['question'] for p in (cross_plan(), activation_plan(), perturbation_plan(), replay_plan())
        for t in p['tasks']}


def make_plan():
    excluded = exclusions()
    seen = set(excluded)
    rng = random.Random(SEED)
    tasks = []
    for rep in range(REPLICATIONS):
        for split, count in COUNTS.items():
            for block in range(count):
                cells = list(CELLS)
                rng.shuffle(cells)
                for family, level in cells:
                    while True:
                        letters = ''.join(rng.choice('ABCD') for _ in range(level)) if family == 'countA' else ''
                        operands = [rng.randrange(10, 100) for _ in range(level)] if family == 'alternatingSum' else []
                        question = (f'Combien de lettres A contient cette chaîne : {letters} ?' if letters else
                                    'Calcule ' + ''.join(('' if i == 0 else ' + ' if i % 2 == 0 else ' - ') + str(n)
                                                        for i, n in enumerate(operands)) + '.')
                        if question not in seen:
                            seen.add(question)
                            break
                    index = len(tasks)
                    tasks.append(dict(id=index, replication=rep, split=split, block=block, family=family,
                                      level=level, letters=letters, operands=operands, question=question,
                                      seed=int(digest([SEED, rep, index, 'answer'])[:8], 16)))
    return dict(schema='menia-natural-error-questions-draft-v1', seed=SEED,
                model=MODELS['A'], generationSettings=SETTINGS, replications=REPLICATIONS,
                countsPerCell=COUNTS, cells=[list(c) for c in CELLS], tasks=tasks,
                excludedQuestions=len(excluded), excludedQuestionsHash=digest(sorted(excluded)),
                plannedGenerations=len(tasks),
                status='Question partitions only. Fitting, journal, contrasts and execution protocol are not yet frozen.',
                scope='New relative to previous Menia numerical tasks, not a claim of absence from pretraining. Same pretrained model across repetitions; no weight changes.')


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plan = make_plan()
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(questions=len(plan['tasks']), excluded=plan['excludedQuestions'], hash=digest(plan))))
