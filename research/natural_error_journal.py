"""Durable prospective collection: freeze readouts, then record before sampling.

The journal establishes the collector's event order and detects corruption.
It is not a trusted external timestamp or a consciousness test.
"""
import copy
import math
import os
from pathlib import Path
import time

from research.activation_monitor import messages
from research.cross_model_prediction import append, canonical, digest, SETTINGS
from research.iphone_coupling_report import require, strict_json
from research.natural_error_questions import make_plan as question_plan
from research.natural_error_readouts import ALPHAS, FORECASTS, fit_bundle, forecast
from research.prospective_confidence_features import validate_capture

RESAMPLES = 20000
FAMILY_ALPHA = .05
PRIMARY = ('finalControl', 'shuffledMiddle', 'donorMiddle')
MIN_CLASS = 20
MIN_GAIN = .005
SOURCE_FILES = (
    'natural_error_journal.py', 'natural_error_analysis.py', 'natural_error_gpu.py',
    'natural_error_questions.py', 'natural_error_readouts.py',
    'prospective_confidence_features.py', 'joint_prediction_capture.py',
    'output_confidence_trace.py', 'output_confidence_validation.py',
    'activation_monitor.py', 'activation_monitor_gpu.py', 'cross_model_prediction.py',
    'cross_model_gpu.py', 'iphone_coupling_report.py', 'iphone_capability_learning_report.py',
    'perturbation_monitor.py', 'replay_controller.py')


def source_hash():
    return digest({name: Path(__file__).with_name(name).read_text(encoding='utf-8') for name in SOURCE_FILES})


def make_plan():
    questions = question_plan()
    return dict(schema='menia-natural-error-plan-v1', questionPlan=questions,
        questionPlanHash=digest(questions), plannedGenerations=questions['plannedGenerations'],
        forecasts=list(FORECASTS), alphas=list(ALPHAS), primaryComparators=list(PRIMARY),
        bootstrapResamples=RESAMPLES, familyAlpha=FAMILY_ALPHA,
        primaryComparisons=questions['replications']*len(PRIMARY),
        minimumSuccessesAndErrorsPerRepetition=MIN_CLASS, minimumBrierGain=MIN_GAIN,
        directThreshold=.8, assumedVerificationLoss=.2,
        interpretation='External readout, conditioned on question. No native self-report, causal action, consciousness or novelty criterion.')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close_tree(actual, expected, label):
    if type(expected) is dict:
        require(type(actual) is dict and set(actual) == set(expected), label+' fields differ')
        for k in expected:
            close_tree(actual[k], expected[k], label+'/'+k)
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), label+' length differs')
        for i, (a, b) in enumerate(zip(actual, expected)):
            close_tree(a, b, label+'/'+str(i))
    elif type(expected) is float:
        require(finite(actual) and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8), label+' differs')
    else:
        require(type(actual) is type(expected) and actual == expected, label+' differs')


def validate_result(event, capture):
    require(type(event['text']) is str and finite(event['seconds']) and event['seconds'] >= 0, 'Result text/duration')
    require(event['status'] in ('ok', 'error', 'interrupted'), 'Result status')
    if event['status'] != 'ok':
        require(event['text'] == '' and type(event['errorType']) is str, 'Failed result')
        return
    require(capture is not None, 'Result precedes pre-token capture')
    t, m = event['trace'], event['metrics']
    require(t['schema'] == 'menia-output-confidence-trace-v1', 'Trace schema')
    require(t['preAnswer'] == capture['preAnswer'], 'Pre-answer trace differs from committed capture')
    require(t['logitSource'] == 'lm_head output, before generation processors and sampling' and
            t['distribution'] == 'raw full vocabulary; temperature 1; no top-k/top-p filtering', 'Trace source')
    c = t['completion']
    ids, logp, entropy = c['tokenIds'], c['tokenLogProbabilities'], c['entropyNats']
    require(type(ids) is list and 1 <= len(ids) <= SETTINGS['max_new_tokens'], 'Token count')
    require(type(logp) is list and type(entropy) is list and type(m['outputTokens']) is int and
            len(ids) == len(logp) == len(entropy) == m['outputTokens'], 'Trace lengths')
    n = capture['preAnswer']['vocabularySize']
    require(all(type(i) is int and 0 <= i < n for i in ids), 'Token identifier')
    require(all(finite(p) and p <= 0 for p in logp), 'Token log probability')
    require(all(finite(e) and 0 <= e <= math.log(n)+1e-10 for e in entropy), 'Token entropy')
    require(c['includesStopTokens'] is True, 'Stop-token accounting')
    require(finite(c['sumLogProbability']) and finite(c['meanLogProbability']) and
            math.isclose(c['sumLogProbability'], math.fsum(logp), abs_tol=1e-10) and
            math.isclose(c['meanLogProbability'], math.fsum(logp)/len(ids), abs_tol=1e-10), 'Sequence arithmetic')
    require(type(m['inputTokens']) is int and 1 <= m['inputTokens'] <= SETTINGS['max_input_tokens'], 'Input length')
    require(m['reachedTokenLimit'] == (len(ids) == SETTINGS['max_new_tokens']), 'Token-limit flag')


class Writer:
    def __init__(self, path, *, sequence=0, previous='0'*64):
        self.path, self.sequence, self.previous = Path(path), sequence, previous

    def write(self, payload, *, create=False):
        envelope = dict(sequence=self.sequence, previous=self.previous, payload=copy.deepcopy(payload))
        value = dict(envelope, sha256=digest(envelope))
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('x', encoding='utf-8', newline='\n') as f:
                f.write(canonical(value)+'\n')
                f.flush()
                os.fsync(f.fileno())
        else:
            append(self.path, value)
        self.sequence += 1
        self.previous = value['sha256']


def read_journal(path, *, check_fit=True):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    require(bool(lines), 'Empty journal')
    previous, events = '0'*64, []
    for i, line in enumerate(lines):
        e = strict_json(line)
        require(set(e) == {'sequence', 'previous', 'payload', 'sha256'}, 'Journal envelope')
        require(e['sequence'] == i and e['previous'] == previous and
                e['sha256'] == digest({k: e[k] for k in ('sequence', 'previous', 'payload')}), 'Journal chain mismatch')
        previous = e['sha256']
        events.append(e['payload'])
    header = events[0]
    plan = make_plan()
    require(header['event'] == 'header' and header['plan'] == plan and header['planHash'] == digest(plan), 'Fixed plan mismatch')
    require(header['sourceHash'] == source_hash(), 'Scientific sources differ')
    require(header['origin'] in ('transformers_gpu', 'synthetic_fixture'), 'Unknown origin')
    tasks = plan['questionPlan']['tasks']
    rows, bundles, pending, failed = [], {}, None, False
    for event in events[1:]:
        require(not failed, 'Events after terminal error')
        kind = event.get('event')
        if kind == 'fit':
            require(pending is None and len(rows) < len(tasks), 'Fit ordering')
            task = tasks[len(rows)]
            rep = task['replication']
            require(task['split'] == 'test' and rep not in bundles and event['replication'] == rep, 'Fit timing/repetition')
            bundle = event['bundle']
            require(event['bundleHash'] == digest(bundle), 'Bundle hash mismatch')
            used = [r for r in rows if r['task']['replication'] == rep and r['task']['split'] in ('train', 'validation')]
            require(bundle['fitDataHash'] == digest(used), 'Fitting records differ')
            if check_fit:
                close_tree(bundle, fit_bundle(rows, rep), 'Refit')
            bundles[rep] = bundle
        elif kind == 'request':
            require(pending is None and len(rows) < len(tasks), 'Request ordering')
            task = tasks[len(rows)]
            require(event == dict(event='request', task=task, messages=messages(task)), 'Request differs from plan')
            require(task['split'] != 'test' or task['replication'] in bundles, 'Test before frozen fit')
            pending = dict(task=task)
        elif kind == 'capture':
            require(pending is not None and 'capture' not in pending and event['id'] == pending['task']['id'], 'Capture ordering')
            validate_capture(event['capture'])
            require(event['captureHash'] == digest(event['capture']), 'Capture hash mismatch')
            task = pending['task']
            expected = forecast(bundles[task['replication']], rows, task, event['capture']) if task['split'] == 'test' else None
            if expected is not None:
                require(type(event['predictions']) is dict and set(event['predictions']) == set(FORECASTS) and
                        all(finite(x) and 0 <= x <= 1 for x in event['predictions'].values()), 'Forecast range/fields')
            close_tree(event['predictions'], expected, 'Prospective forecast')
            pending.update(capture=event['capture'], predictions=event['predictions'])
        elif kind == 'result':
            require(pending is not None and event['id'] == pending['task']['id'], 'Result ordering')
            validate_result(event, pending.get('capture'))
            rows.append(dict(pending, result=event))
            pending = None
            failed = event['status'] != 'ok'
        else:
            raise ValueError('Unknown journal event')
    return dict(header=header, rows=rows, pending=pending, bundles=bundles, failed=failed,
                sequence=len(lines), previous=previous, fitChecked=bool(check_fit and bundles))


def collect(path, backend, *, resume=False, limit=None):
    require(limit is None or type(limit) is int and limit >= 0, 'Invalid collection limit')
    path = Path(path)
    if resume:
        recovered = read_journal(path)
        header, rows, bundles = recovered['header'], recovered['rows'], recovered['bundles']
        require(header['origin'] == backend.origin and header['metadata'] == backend.metadata, 'Resume environment differs')
        require(not recovered['failed'] and recovered['pending'] is None, 'Interrupted/failed attempt is retained; no automatic replacement')
        writer = Writer(path, sequence=recovered['sequence'], previous=recovered['previous'])
    else:
        plan = make_plan()
        header = dict(event='header', plan=plan, planHash=digest(plan), sourceHash=source_hash(),
                      origin=backend.origin, metadata=backend.metadata)
        writer = Writer(path)
        writer.write(header, create=True)
        rows, bundles = [], {}
    initial = len(rows)
    tasks = header['plan']['questionPlan']['tasks']
    for task in tasks[initial:]:
        if limit is not None and len(rows)-initial >= limit:
            break
        rep = task['replication']
        if task['split'] == 'test' and rep not in bundles:
            bundle = fit_bundle(rows, rep)
            writer.write(dict(event='fit', replication=rep, bundle=bundle, bundleHash=digest(bundle)))
            bundles[rep] = bundle
        writer.write(dict(event='request', task=task, messages=messages(task)))
        row = dict(task=task)
        def capture(value):
            require('capture' not in row, 'Capture must be emitted once before sampling')
            value = copy.deepcopy(value)
            validate_capture(value)
            predictions = forecast(bundles[rep], rows, task, value) if task['split'] == 'test' else None
            writer.write(dict(event='capture', id=task['id'], capture=value,
                              captureHash=digest(value), predictions=predictions))
            row.update(capture=value, predictions=predictions)
        started = time.perf_counter()
        try:
            text, metrics, trace = backend.generate(task, capture)
            event = dict(event='result', id=task['id'], status='ok', text=text,
                         seconds=time.perf_counter()-started, metrics=metrics, trace=trace)
            validate_result(event, row.get('capture'))
        except (Exception, KeyboardInterrupt) as exc:
            writer.write(dict(event='result', id=task['id'], status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                              text='', seconds=time.perf_counter()-started, errorType=type(exc).__name__))
            raise
        writer.write(event)
        rows.append(dict(row, result=event))
        print(f"{len(rows)}/{len(tasks)} r{rep} {task['split']} recorded", flush=True)
    return len(rows)
