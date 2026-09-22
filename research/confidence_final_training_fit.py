"""Post-Colab24 descriptive evaluation of its final weights on training inputs.

No optimization, new answers, held-out inputs, calibrator or success criterion.
This distinguishes final-checkpoint fit from online pre-update observations;
it does not by itself identify why a model underfits or generalizes poorly.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from research.answer_confidence_plan import load_training
from research.answer_confidence_journal import validate_judgment
from research.answer_confidence_analysis import metrics
from research.audit_confidence_ranking import explicit_auc, independent_metrics
from research.confidence_ranking_analysis import weighted_within_auc
from research.confidence_ranking_study import FILES as PARENT_FILES
from research.cross_model_prediction import CELLS, MODELS, digest
from research.iphone_coupling_report import require, strict_json
from research.natural_error_journal import close_tree

ROOT = Path(__file__).resolve().parents[1]
TRAINING_HASH = '0694f42765731e3ab2031da214855468112c0bdfefe6ddd8019c769efcc403a8'
PARENT_SOURCE_HASH = '12489c817466d98892b43e249fdd7a25d360522770af820c0bfefefda7d87129'
ARMS = ('ce', 'rank', 'neutral')
FILES = tuple(sorted(set(PARENT_FILES + ('confidence_final_training_fit.py', 'confidence_final_training_fit_gpu.py',
                                       'audit_confidence_ranking.py'))))


def source_hash():
    return digest({name: Path(__file__).with_name(name).read_text(encoding='utf-8') for name in FILES})


def training_rows():
    data, _ = load_training(ROOT/'artifacts/answer-confidence-training-data')
    require(digest(data['measured']) == TRAINING_HASH, 'Frozen training examples changed')
    return data['measured']


def make_plan():
    freeze = json.loads((ROOT/'artifacts/confidence-ranking-pilot/training-freeze.json').read_text(encoding='utf-8'))
    require(freeze['sourceHash'] == PARENT_SOURCE_HASH, 'Parent source mismatch')
    units = []
    for rep in range(3):
        for arm in ARMS:
            key = f'r{rep}-{arm}'; name = f'confidence-ranking-20260920-v1.{key}.safetensors'
            units.append(dict(key=key, replication=rep, arm=arm, checkpoint=name, **freeze['checkpointFiles'][name]))
    return dict(schema='menia-final-training-fit-plan-v1', model=MODELS['A'], trainingHash=TRAINING_HASH,
        units=units, examplesPerUnit=576, plannedCalls=5184, maxInputTokens=1792,
        rank=8, scale=1., newWeightUpdates=0, newGenerations=0, heldOutInputs=0,
        selection='All original measured training examples of each adapter; no outcome-based subsampling',
        statistics='Descriptive Brier, within-category AUROC, binary-label agreement and vocabulary code mass; no significance threshold',
        scope='Final training-set fit, decided after Colab24 test interpretation. Not new generalization, '
              'a causal capacity comparison, privileged self-access or consciousness evidence.')


def calls(rows, plan):
    require(digest(rows) == TRAINING_HASH, 'Training inputs changed')
    result = []
    for unit in plan['units']:
        subset = sorted((r for r in rows if r['replication'] == unit['replication']), key=lambda r:r['sourceId'])
        require(len(subset) == plan['examplesPerUnit'], 'Unit example count')
        for row in subset:
            result.append(dict(id=len(result), key=unit['key'], sourceId=row['sourceId']))
    require(len(result) == plan['plannedCalls'], 'Call count')
    return result


def request_for(call, rows_by_id):
    row = rows_by_id[call['sourceId']]
    return dict(event='request', call=call, messages=row['messages'])


def read_journal(path):
    previous = '0'*64; events = []
    for i, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines()):
        envelope = strict_json(line)
        require(set(envelope) == {'sequence','previous','payload','sha256'} and
                envelope['sequence'] == i and envelope['previous'] == previous, 'Journal order')
        require(envelope['sha256'] == digest({k:envelope[k] for k in ('sequence','previous','payload')}), 'Journal hash')
        previous = envelope['sha256']; events.append(envelope['payload'])
    require(bool(events), 'Empty journal'); header = events[0]; plan = make_plan()
    require(header['event'] == 'header' and header['plan'] == plan and header['planHash'] == digest(plan), 'Fixed plan changed')
    require(header['sourceHash'] == source_hash() and header['origin'] in ('transformers_gpu','synthetic_fixture'), 'Source or origin')
    rows = training_rows(); by_id = {r['sourceId']:r for r in rows}; schedule = calls(rows, plan)
    require(header['callPlanHash'] == digest(schedule), 'Call schedule changed')
    require(header['metadata']['model'] == plan['model'] and header['metadata']['newWeightUpdates'] == 0, 'Model metadata')
    active = None; pending = None; completed = []; judgments = []; failed = False; final = False; count = 0
    for event in events[1:]:
        require(not failed and not final, 'Events after terminal event')
        kind = event['event']
        if kind == 'failure':
            failed = True
        elif kind == 'unit_start':
            require(active is None and pending is None and len(completed) < 9, 'Invalid unit start')
            active = plan['units'][len(completed)]; count = 0
            require(event['key'] == active['key'] and event['sha256'] == active['sha256'], 'Checkpoint changed')
        elif kind == 'request':
            require(active is not None and pending is None and count < 576 and len(judgments) < len(schedule), 'Unexpected request')
            call = schedule[len(judgments)]
            require(call['key'] == active['key'] and event == request_for(call, by_id), 'Request changed')
            pending = event
        elif kind == 'judgment':
            require(pending is not None, 'Judgment without request')
            validate_judgment(event, pending, header['metadata'])
            judgments.append(dict(call=pending['call'], scores=event['scores'], seconds=event['seconds']))
            pending = None; count += 1
        elif kind == 'unit_complete':
            require(active is not None and pending is None and count == event['calls'] == 576, 'Incomplete unit')
            require(event['key'] == active['key'] and event['sha256'] == active['sha256'] and
                    event['actualAdapterMatchesFile'] is True and event['baseParameterVersionsUnchanged'] is True, 'Changed model state')
            completed.append(event); active = None
        elif kind == 'complete':
            require(active is None and pending is None and len(completed) == 9 and
                    len(judgments) == event['calls'] == 5184, 'Incomplete diagnostic')
            final = True
        else:
            raise ValueError('Unknown event '+kind)
    return dict(header=header, rows=by_id, judgments=judgments, units=completed, complete=final and not failed,
                pending=pending, active=active, failed=failed, chainEnd=previous)


def summarize(data):
    require(data['complete'], 'Complete diagnostic required')
    groups = {}; max_difference = 0.
    for unit in data['header']['plan']['units']:
        observations = [o for o in data['judgments'] if o['call']['key'] == unit['key']]
        rows = [data['rows'][o['call']['sourceId']] for o in observations]
        y = [int(r['target']) for r in rows]; ps = [o['scores']['conditionalCorrect'] for o in observations]
        cells = np.asarray([CELLS.index((r['family'],r['level'])) for r in rows])
        score = metrics(y, ps); reference = independent_metrics(y, ps)
        close_tree(score, reference, 'Separate descriptive arithmetic')
        weights = np.ones((1, len(rows)))
        auc, pairs = weighted_within_auc(y, ps, cells, weights)
        checked, counted = explicit_auc(y, ps, cells, weights)
        require(np.array_equal(pairs, counted) and np.allclose(auc, checked, rtol=0., atol=1e-12, equal_nan=True), 'Separate pairwise AUROC')
        if np.isfinite(auc[0]): max_difference = max(max_difference, abs(float(auc[0]-checked[0])))
        for field in ('brier','auc','meanProbability'):
            if score[field] is not None: max_difference = max(max_difference, abs(score[field]-reference[field]))
        groups[unit['key']] = dict(**score, withinAuc=float(auc[0]) if np.isfinite(auc[0]) else None,
            withinPairs=int(pairs[0]), binaryAgreement=float(np.mean([(p >= .5) == bool(label) for p,label in zip(ps,y)])),
            topCodeFraction=float(np.mean([o['scores']['topIsCode'] for o in observations])),
            meanCodeMass=float(np.mean([o['scores']['candidateMass'] for o in observations])),
            cells={f'{family}/{level}':metrics([v for v,c in zip(y,cells) if c == i], [v for v,c in zip(ps,cells) if c == i])
                   for i,(family,level) in enumerate(CELLS)})
    return dict(schema='menia-final-training-fit-summary-v1', origin=data['header']['origin'], complete=True,
        planHash=data['header']['planHash'], sourceHash=data['header']['sourceHash'], chainEnd=data['chainEnd'],
        recordedCalls=len(data['judgments']), units=groups, separateArithmeticMaxDifference=max_difference,
        seconds=sum(o['seconds'] for o in data['judgments']), scope=data['header']['plan']['scope'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--journal', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.prepare:
        plan = make_plan(); rows = training_rows()
        report = dict(plan=plan, planHash=digest(plan), sourceHash=source_hash(), callPlanHash=digest(calls(rows,plan)))
    else:
        require(args.journal is not None, 'Journal required'); report = summarize(read_journal(args.journal))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('planHash','sourceHash')}, ensure_ascii=False))
