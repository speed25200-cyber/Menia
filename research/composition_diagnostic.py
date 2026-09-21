"""Prospective diagnosis of public wording and external decision thresholds; never trains."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random

import numpy as np

from research import state_composition as previous
from research.iphone_coupling_report import require, strict_json

SEED = 202609193
MODEL = previous.MODEL
CONFIG = dict(blocksPerReplication=24, replications=3, resamples=2000,
              maxInputTokens=512, layer=17, strength=1., rank=8)
ARMS = ('base', 'parent', 'composed', 'grammar')
FORMATS = ('trained', 'paraphrase')
STREAMS = (('hidden', 'monitor'), ('visible', 'marker_first'), ('visible', 'marker_second'))
ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / 'artifacts/composition-diagnostic-pilot/frozen-thresholds.json'
COMPOSITION_SHA256 = 'a8dc7d8b018f34ea7c07fe80b08a3775e9ea0b7e132f05b64357c4ff122b8a7c'
PARENT_SHA256 = previous.previous.PARENT_JOURNAL_SHA256
digest = previous.previous.parent.digest
interval = previous.previous.parent.interval
score = previous.score
prompt = previous.prompt


def checkpoints():
    report = strict_json((ROOT / 'artifacts/state-composition-pilot/first-audit-summary.json').read_text(encoding='utf-8'))
    require(report['complete'] and report['origin'] == 'transformers_gpu', 'Incomplete source run')
    parents = previous.previous.checkpoints()
    return {f'r{r}-{a}': parents[f'r{r}-strong'] if a == 'parent' else report['trained'][f'r{r}-{a}']
            for r, a in itertools.product(range(CONFIG['replications']), ARMS[1:])}


def fit_threshold(scores):
    """Each row is absent, first-present, second-present; equal weight per class."""
    s = np.asarray(scores, dtype=float)
    require(s.ndim == 2 and s.shape[1] == 3 and len(s) > 0 and np.isfinite(s).all(), 'Invalid calibration data')
    unique = np.unique(s)
    candidates = np.unique(np.r_[unique[0]-1., (unique[:-1]+unique[1:])/2., unique[-1]+1., 0.])
    def accuracy(t):
        return float((.5*(s[:, 0] <= t) + .25*(s[:, 1] > t) + .25*(s[:, 2] > t)).mean())
    threshold = min(candidates, key=lambda t: (-accuracy(t), abs(t), t))
    return dict(threshold=float(threshold), trainingBalancedAccuracy=accuracy(threshold),
                blocks=len(s), rows=int(s.size), scoresSHA256=digest(s.tolist()))


def make_calibration(journal):
    """Select only composition TRAIN rows. Prior test outcomes motivated this new diagnostic."""
    require(hashlib.sha256(Path(journal).read_bytes()).hexdigest() == COMPOSITION_SHA256, 'Source journal mismatch')
    h, rows, trained, pending, _ = previous.read_journal(journal)
    require(pending is None and len(rows) == len(h['plan']['evaluation']), 'Incomplete source')
    require(all(trained[k] == v for k, v in checkpoints().items() if not k.endswith('-parent')), 'Checkpoint mismatch')
    indexed = {r['request']['id']: r for r in rows}
    fits = {}
    for rep, mapping in itertools.product(range(CONFIG['replications']), range(2)):
        values, ids = [], []
        for b in h['plan']['blocks']:
            if b['replication'] != rep or b['split'] != 'train':
                continue
            group = []
            for pos in range(3):
                key = f"{rep}/composed/hidden/{b['id']}/trained/monitor/{mapping}/{pos}"
                row = indexed[key]
                group.append(score(row['result'], row['request'], b, h['metadata']['choiceTokenIds'])['presenceScore'])
                ids.append(key)
            values.append(group)
        fits[f'{rep}/{mapping}'] = dict(fit_threshold(values), sourceRequestIds=ids)
    return dict(schema='menia-composition-thresholds-v1', sourceJournalSHA256=COMPOSITION_SHA256,
                checkpoints=checkpoints(), selection='train/composed/hidden/trained/monitor; positions 0,1,2 only',
                rule='presenceScore > threshold; maximize class-balanced accuracy; ties: smallest absolute threshold then smallest value',
                role='External numerical readout, not the native model response; no test-row fitting', fits=fits)


def calibration():
    result = strict_json(CALIBRATION.read_text(encoding='utf-8'))
    require(result['schema'] == 'menia-composition-thresholds-v1' and result['sourceJournalSHA256'] == COMPOSITION_SHA256,
            'Calibration lineage mismatch')
    require(result['checkpoints'] == checkpoints(), 'Calibration checkpoints mismatch')
    require(set(result['fits']) == {f'{r}/{m}' for r in range(CONFIG['replications']) for m in range(2)}, 'Missing thresholds')
    for fit in result['fits'].values():
        require(type(fit['threshold']) in (int, float) and np.isfinite(fit['threshold']), 'Invalid threshold')
    return result


def source_hash():
    return digest(dict(previous=previous.source_hash(), calibration=calibration(),
                       files={n: Path(__file__).with_name(n).read_text(encoding='utf-8')
                              for n in ('composition_diagnostic.py', 'composition_diagnostic_gpu.py')}))


def old_blocks():
    p = previous.previous
    return (previous.plan()['blocks'] + p.plan()['blocks'] + p.parent.plan()['blocks'] +
            p.parent.previous.plan()['blocks'] + p.parent.previous.previous.plan()['blocks'] +
            p.parent.previous.previous.previous_plan()['blocks'])


def plan():
    require(CONFIG['blocksPerReplication'] >= 2 and CONFIG['blocksPerReplication'] % 2 == 0, 'Balanced blocks required')
    rng = random.Random(SEED)
    old = old_blocks()
    used = {s for b in old for s in b['sentences']}; noises = {b['noiseSeed'] for b in old}
    pool = [f'{s} {v} {o} {p}.' for s, v, o, p in itertools.product(
        ('Le marin','La voisine','Le peintre','La libraire','Le jardinier','La musicienne','Le facteur','La guide'),
        ('observe','dessine','photographie','déplace','cherche','retrouve','examine','transporte'),
        ('une boîte','un panier','une chaise','un carnet','une lampe','un tableau','une valise','un vase'),
        ('près du port','dans le jardin','devant la maison','dans la cour','près de la fenêtre',"dans l'atelier",'sur la terrasse','près de la porte'))]
    pool = [s for s in pool if s not in used]; rng.shuffle(pool)
    blocks, evaluation = [], []
    for rep in range(CONFIG['replications']):
        group = []
        for i in range(CONFIG['blocksPerReplication']):
            noise = rng.randrange(2**31)
            while noise in noises: noise = rng.randrange(2**31)
            noises.add(noise)
            b = dict(id=f'r{rep}-diagnostic-{i:03d}', replication=rep, marker=i % 2,
                     noiseSeed=noise, sentences=[pool.pop(), pool.pop()])
            blocks.append(b); group.append(b)
        for arm in ARMS:
            requests = []
            for b, form, (family, task), mapping, pos in itertools.product(group, FORMATS, STREAMS, range(2), range(4)):
                requests.append(dict(id=f"{rep}/{arm}/{family}/{b['id']}/{form}/{task}/{mapping}/{pos}",
                    replication=rep, arm=arm, family=family, block=b['id'], format=form, task=task, mapping=mapping, position=pos))
            rng.shuffle(requests); evaluation.extend(requests)
    return dict(schema='menia-composition-diagnostic-plan-v1', seed=SEED, model=MODEL, config=CONFIG,
                parentJournalSHA256=PARENT_SHA256, compositionJournalSHA256=COMPOSITION_SHA256,
                checkpoints=checkpoints(), calibration=calibration(), blocks=blocks, evaluation=evaluation)


def read_journal(path):
    events = [strict_json(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    require(events and events[0].get('event') == 'header', 'Missing header')
    h = events[0]; fixed = plan()
    require(h['plan'] == fixed and h['planHash'] == digest(fixed) and h['sourceHash'] == source_hash(), 'Plan/source mismatch')
    require(h['metadata']['origin'] in ('transformers_gpu', 'synthetic_fixture'), 'Invalid origin')
    choices = h['metadata']['choiceTokenIds']
    require(len(choices) == 4 and len(set(choices)) == 4 and all(type(x) is int and x >= 0 for x in choices), 'Invalid choices')
    blocks = {b['id']: b for b in fixed['blocks']}; pending = None; rows = []
    for e in events[1:]:
        if e['event'] == 'request':
            require(pending is None and len(rows) < len(fixed['evaluation']), 'Unexpected request')
            req = fixed['evaluation'][len(rows)]; b = blocks[req['block']]
            require(e['request'] == req and e['promptHash'] == digest(prompt(b, req['family'], req['task'], req['position'], req['format'], req['mapping'])[0]), 'Input mismatch')
            require(e['adapterHash'] == fixed['checkpoints'].get(f"r{req['replication']}-{req['arm']}"), 'Adapter mismatch')
            require(type(e['inputTokens']) is int and 0 < e['inputTokens'] <= CONFIG['maxInputTokens'], 'Input length')
            pending = e
        elif e['event'] == 'result':
            require(pending is not None and e['id'] == pending['request']['id'], 'Result order')
            req = pending['request']; score(e, req, blocks[req['block']], choices)
            rows.append(dict(request=req, result=e, inputTokens=pending['inputTokens'])); pending = None
        elif e['event'] == 'interrupted_request':
            require(pending is not None and e['id'] == pending['request']['id'], 'Unexpected interruption'); pending = None
        elif e['event'] == 'error': require(type(e['errorType']) is str, 'Invalid error')
        else: raise ValueError('Unexpected event; this diagnostic never trains')
    return h, rows, pending, events


def analyze(path):
    h, rows, pending, events = read_journal(path)
    complete = len(rows) == len(h['plan']['evaluation']) and pending is None
    tables, contrasts, shams = {}, [], 0
    if complete:
        keyed = {r['request']['id']: r for r in rows}
        for rep in range(CONFIG['replications']):
            bs = [b for b in h['plan']['blocks'] if b['replication'] == rep]; n = len(bs)
            rng = np.random.default_rng(SEED+900+rep)
            # Stratify by public marker; all arms, mappings and forms share the same block draws.
            strata = [np.array([i for i, b in enumerate(bs) if b['marker'] == y]) for y in (0, 1)]
            draws = np.concatenate([rng.choice(s, (CONFIG['resamples'], len(s))) for s in strata], axis=1)
            native = {}; calibrated = {}; zero = {}
            def contrast(kind, values, **fields):
                contrasts.append(dict(replication=rep, kind=kind, difference=float(values.mean()),
                    interval95=interval(values[draws].mean(1)), **fields))
            for arm, form, (family, task), mapping in itertools.product(ARMS, FORMATS, STREAMS, range(2)):
                measures, scores, native_blocks = [], [], []
                for b in bs:
                    prefix = f"{rep}/{arm}/{family}/{b['id']}/{form}/{task}/{mapping}/"
                    a, c = (keyed[prefix+str(pos)]['result'] for pos in (0, 3))
                    require(all(a[k] == c[k] for k in ('choiceLogits', 'choiceMass', 'rawTokenId', 'rawChoice')), 'Sham differs')
                    shams += 1; group = []
                    for pos in range(3):
                        r = keyed[prefix+str(pos)]
                        group.append(score(r['result'], r['request'], b, h['metadata']['choiceTokenIds']))
                    measures.extend(group); scores.append([m['presenceScore'] for m in group])
                    weights = (.5, .25, .25) if task == 'monitor' else (1/3, 1/3, 1/3)
                    native_blocks.append(sum(w*m['firstTokenCorrect'] for w, m in zip(weights, group)))
                s = np.array(scores); native[(arm, form, task, mapping)] = np.array(native_blocks)
                acc = native[(arm, form, task, mapping)]
                table = {k: float(np.mean([m[k] for m in measures])) for k in
                         ('correct', 'firstTokenCorrect', 'firstTokenIsOption', 'choiceMass', 'fourDigitMass', 'brier')}
                table.update(n=3*n, blocks=n, nativeBalancedAccuracy=float(acc.mean()),
                             nativeBalancedAccuracyInterval95=interval(acc[draws].mean(1)))
                if task == 'monitor':
                    w = previous.previous.parent.wins(s)
                    table.update(presenceAUROC=previous.previous.parent.auroc(w),
                                 presenceAUROCInterval95=interval(previous.previous.parent.auroc(w, draws)))
                    if arm == 'composed':
                        t = h['plan']['calibration']['fits'][f'{rep}/{mapping}']['threshold']
                        for label, threshold, dest in (('calibrated', t, calibrated), ('zeroThreshold', 0., zero)):
                            values = .5*(s[:, 0] <= threshold) + .25*(s[:, 1] > threshold) + .25*(s[:, 2] > threshold)
                            dest[(form, mapping)] = values
                            table[label+'BalancedAccuracy'] = float(values.mean())
                            table[label+'Interval95'] = interval(values[draws].mean(1))
                        table['externalThreshold'] = t
                tables[f'{rep}/{arm}/{family}/{form}/{task}/{mapping}'] = table
            for arm, task, mapping in itertools.product(ARMS, ('marker_first', 'marker_second'), range(2)):
                penalty = native[(arm, 'paraphrase', task, mapping)] - native[(arm, 'trained', task, mapping)]
                contrast('publicWordingChange', penalty, arm=arm, task=task, mapping=mapping)
                if arm != 'base':
                    base = native[('base', 'paraphrase', task, mapping)] - native[('base', 'trained', task, mapping)]
                    contrast('publicWordingInteractionVsBase', penalty-base, arm=arm, task=task, mapping=mapping)
            for form, mapping in itertools.product(FORMATS, range(2)):
                for reference, values in (('nativeFirstToken', native[('composed', form, 'monitor', mapping)]),
                                          ('zeroThreshold', zero[(form, mapping)])):
                    contrast('externalThresholdGain', calibrated[(form, mapping)]-values,
                             arm='composed', format=form, mapping=mapping, reference=reference)
    return dict(schema='menia-composition-diagnostic-report-v1', origin=h['metadata']['origin'], complete=complete,
        planHash=h['planHash'], sourceHash=h['sourceHash'], checkpoints=h['plan']['checkpoints'],
        calibration=h['plan']['calibration'], recorded=len(rows), planned=len(h['plan']['evaluation']),
        tables=tables, contrasts=contrasts, shamPairs=shams, trainingUpdates=0,
        errors=sum(e['event'] == 'error' for e in events), interruptedRequests=sum(e['event'] == 'interrupted_request' for e in events),
        evaluationSeconds=sum(r['result']['seconds'] for r in rows),
        limitations='Diagnostic chosen after experiment 12. Fresh sentences, reused vocabulary and wording. External thresholds are not native model improvements. No training, global success gate, mechanistic identification or consciousness inference.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('journal', type=Path)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--freeze-thresholds', action='store_true')
    a = p.parse_args(); result = make_calibration(a.journal) if a.freeze_thresholds else analyze(a.journal)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
