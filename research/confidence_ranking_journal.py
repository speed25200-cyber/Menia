"""Strict reconstruction of paired training and frozen crossed evaluation."""
import math
from pathlib import Path

from research import confidence_ranking_study as study
from research.confidence_ranking import paired_batches, preparation_report, CONFIG
from research.answer_confidence_plan import load_training
from research.answer_confidence_journal import validate_answer, validate_judgment
from research.cross_model_prediction import digest
from research.iphone_coupling_report import require, strict_json


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_step(event, batch, arm):
    require(event['dataHash'] == digest(batch), 'Paired examples changed')
    require(len(event['losses']) == 8 and all(finite_nonnegative(v) for v in event['losses']), 'CE losses')
    require(len(event['pairLosses']) == len(event['scoreDifferences']) == 4 and
            all(finite_nonnegative(v) for v in event['pairLosses']) and
            all(type(v) in (int, float) and math.isfinite(v) for v in event['scoreDifferences']), 'Pair losses')
    require(event['eligible'] == [p['eligible'] for p in batch], 'Pair eligibility')
    for pair, delta, loss in zip(batch, event['scoreDifferences'], event['pairLosses']):
        softplus = lambda x: max(x, 0.)+math.log1p(math.exp(-abs(x)))
        if arm == 'ce' or not pair['eligible']:
            expected = 0.
        elif arm == 'rank':
            expected = softplus(-(int(pair['left']['target'])-int(pair['right']['target']))*delta)
        else:
            expected = .5*(softplus(delta)+softplus(-delta))
        require(math.isclose(loss, expected, abs_tol=1e-6, rel_tol=1e-6), 'Pair objective arithmetic')
    expected = math.fsum(event['losses'])/8+CONFIG['pairWeight']*math.fsum(event['pairLosses'])/4
    require(finite_nonnegative(event['objective']) and math.isclose(event['objective'], expected, abs_tol=1e-8, rel_tol=1e-8), 'Total objective')
    require(len(event['inputTokens']) == 8 and all(type(n) is int and 1 < n <= CONFIG['maxTrainingTokens'] for n in event['inputTokens']), 'Input lengths')
    require(finite_nonnegative(event['gradientNorm']) and finite_nonnegative(event['seconds']), 'Step metrics')


def read_journal(path):
    events = []; previous = '0'*64
    for index, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines()):
        row = strict_json(line)
        require(set(row) == {'sequence', 'previous', 'payload', 'sha256'} and
                row['sequence'] == index and row['previous'] == previous, 'Journal order')
        require(row['sha256'] == digest({k: row[k] for k in ('sequence', 'previous', 'payload')}), 'Journal hash')
        previous = row['sha256']; events.append(row['payload'])
    require(bool(events), 'Empty journal')
    h = events[0]; p = study.make_plan()
    require(h['event'] == 'header' and h['plan'] == p and h['planHash'] == digest(p), 'Fixed plan changed')
    require(h['sourceHash'] == study.source_hash() and h['origin'] in ('transformers_gpu', 'synthetic_fixture'), 'Source or origin')
    data, parent_report = load_training(Path(__file__).parents[1]/'artifacts/answer-confidence-training-data')
    rows = data['measured']
    require(h['trainingData'] == rows and h['trainingReport'] == preparation_report(rows) and
            h['parentTrainingReportHash'] == digest(parent_report), 'Training data or coverage changed')
    answers = {}; judgments = {}; starts = {}; checkpoints = {}; steps = []
    active = None; pending = None; batch_cache = None; step = 0; index = 0; failed = False
    for event in events[1:]:
        require(not failed, 'Events after failure')
        kind = event['event']
        if kind == 'failure': failed = True; continue
        if kind == 'training_start':
            require(index == 0 and active is None and pending is None and len(checkpoints) < 9, 'Training phase')
            active = p['trainingUnits'][len(checkpoints)]; key = active['key']
            require(event['key'] == key and type(event['initializationHash']) is str and len(event['initializationHash']) == 64 and
                    type(event['trainableParameters']) is int and event['trainableParameters'] > 0, 'Training initialization')
            reference = starts.get(f'r{active["replication"]}-ce')
            if reference:
                require(event['initializationHash'] == reference['initializationHash'] and
                        event['trainableParameters'] == reference['trainableParameters'], 'Unmatched initial weights or capacity')
            starts[key] = event; step = 0; batch_cache = paired_batches(rows, active['replication'])
        elif kind == 'training_step':
            require(active is not None and event['key'] == active['key'] and event['step'] == step+1 <= 144, 'Training step')
            validate_step(event, batch_cache[step], active['arm'])
            reference_key = f'r{active["replication"]}-ce'
            if active['arm'] != 'ce':
                reference = next(s for s in steps if s['key'] == reference_key and s['step'] == event['step'])
                require(event['inputTokens'] == reference['inputTokens'], 'Unmatched token exposure')
            steps.append(event); step += 1
        elif kind == 'training_complete':
            require(active is not None and event['key'] == active['key'] and step == event['steps'] == 144, 'Incomplete training')
            require(event['initializationHash'] == starts[event['key']]['initializationHash'], 'Changed initialization')
            require(len(event['sha256']) == 64 and event['checkpoint'] == Path(path).with_suffix('.'+event['key']+'.safetensors').name and
                    finite_nonnegative(event['trainingSeconds']), 'Checkpoint identity or duration')
            checkpoints[event['key']] = event; active = None; batch_cache = None
        elif kind == 'request':
            require(len(checkpoints) == 9 and active is None and pending is None and index < p['plannedCalls'], 'Unfrozen or extra call')
            pending = study.request_for(p, index, answers, checkpoints)
            require(event == pending, 'Request/context changed')
        elif kind in ('answer', 'judgment'):
            require(pending is not None, 'Unpaired result'); c = pending['call']
            if c['kind'] == 'answer':
                validate_answer(event, pending)
                require(finite_nonnegative(event['seconds']), 'Answer duration')
                answers[(c['task'], c['producer'])] = event
            else:
                validate_judgment(event, pending, h['metadata'])
                judgments[(c['task'], c['judge'], c['producer'])] = event
            pending = None; index += 1
        else:
            raise ValueError('Unknown event '+kind)
    return dict(header=h, answers=answers, judgments=judgments, starts=starts, checkpoints=checkpoints,
                steps=steps, recorded=index, pending=pending, activeTraining=active, failed=failed,
                complete=index == p['plannedCalls'] and pending is None and active is None and not failed,
                chainEnd=previous)
