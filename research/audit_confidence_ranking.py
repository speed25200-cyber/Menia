"""Second arithmetic implementation and actual adapter checks for Colab24.

The collector/reader and fixed recipe remain shared: this is a numerical audit,
not an independent experiment or an external scientific replication.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from research.confidence_ranking_journal import read_journal
from research.confidence_ranking_study import CRITERIA, MODEL_ARMS, SEED
from research.cross_model_prediction import CELLS


def explicit_auc(y, probabilities, cells, multiplicities):
    """Sum weighted positive-negative comparisons, without sorting/ranking.

    Matrix products weight the *original questions*. A multiplicity >1 creates
    repeated observations, not new independent success/error pairs.
    """
    y = np.asarray(y, dtype=int); ps = np.asarray(probabilities, dtype=float)
    cells = np.asarray(cells); weights = np.asarray(multiplicities, dtype=float)
    assert y.ndim == ps.ndim == cells.ndim == 1 and y.shape == ps.shape == cells.shape
    assert len(y) and np.isin(y, (0, 1)).all() and np.isfinite(ps).all()
    assert weights.ndim == 2 and weights.shape[1] == len(y) and np.isfinite(weights).all() and (weights >= 0).all()
    numerator = np.zeros(len(weights)); denominator = np.zeros(len(weights))
    for cell in sorted(set(cells.tolist())):
        yes = np.flatnonzero((cells == cell) & (y == 1))
        no = np.flatnonzero((cells == cell) & (y == 0))
        if not len(yes) or not len(no): continue
        comparisons = (ps[yes, None] > ps[no]).astype(float)+.5*(ps[yes, None] == ps[no])
        good, bad = weights[:, yes], weights[:, no]
        numerator += np.sum((good @ comparisons)*bad, axis=1)
        denominator += good.sum(axis=1)*bad.sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0), denominator


def independent_metrics(y, ps):
    n = len(y); assert n == len(ps) and n > 0
    positives = [p for p, label in zip(ps, y) if label]
    negatives = [p for p, label in zip(ps, y) if not label]
    auc = (math.fsum(float(a > b)+.5*float(a == b) for a in positives for b in negatives)/
           (len(positives)*len(negatives))) if positives and negatives else None
    bins = []
    for k in range(10):
        ids = [i for i, p in enumerate(ps) if min(int(p*10), 9) == k]
        bins.append(dict(lower=k/10, upper=(k+1)/10, n=len(ids),
                         meanProbability=math.fsum(ps[i] for i in ids)/len(ids) if ids else None,
                         accuracy=sum(y[i] for i in ids)/len(ids) if ids else None))
    return dict(n=n, correct=sum(y), brier=math.fsum((p-label)**2 for p, label in zip(ps, y))/n,
                auc=auc, meanProbability=math.fsum(ps)/n, reliability=bins)


def verify_weights(data, path):
    import torch
    from safetensors.torch import load_file
    path = Path(path); checked = {}; initial_hashes = []
    for rep in range(3):
        initial_path = path.with_suffix(f'.r{rep}-initial.safetensors')
        initial = load_file(str(initial_path)); initial_sha = hashlib.sha256(initial_path.read_bytes()).hexdigest()
        assert initial and all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in initial.values())
        modules = {key.rsplit('.', 1)[0] for key in initial}
        assert set(initial) == {name+'.'+factor for name in modules for factor in ('a', 'b')}
        assert all(name.endswith(('.q_proj', '.v_proj')) for name in modules)
        for name in modules:
            a, b = initial[name+'.a'], initial[name+'.b']
            assert a.ndim == b.ndim == 2 and a.shape[0] == b.shape[1] == 8
            assert bool(torch.count_nonzero(a)) and not bool(torch.count_nonzero(b))
        count = sum(v.numel() for v in initial.values()); initial_hashes.append(initial_sha)
        checked[initial_path.name] = dict(sha256=initial_sha, tensors=len(initial), parameters=count,
                                         bytes=initial_path.stat().st_size, initial=True)
        for arm in ('ce', 'rank', 'neutral'):
            key = f'r{rep}-{arm}'; start = data['starts'][key]; checkpoint = data['checkpoints'][key]
            assert initial_sha == start['initializationHash'] == checkpoint['initializationHash']
            assert count == start['trainableParameters']
            file = path.with_suffix('.'+key+'.safetensors')
            sha = hashlib.sha256(file.read_bytes()).hexdigest(); assert sha == checkpoint['sha256']
            final = load_file(str(file)); assert set(final) == set(initial)
            assert all(v.shape == initial[k].shape and v.dtype == initial[k].dtype and bool(torch.isfinite(v).all())
                       for k, v in final.items())
            assert any(not torch.equal(v, initial[k]) for k, v in final.items())
            checked[file.name] = dict(sha256=sha, tensors=len(final), parameters=count,
                                      bytes=file.stat().st_size, initial=False)
    assert len(set(initial_hashes)) == 3 and len(checked) == 12
    return checked


def verify(path, report, *, check_weights=True):
    data = read_journal(path); assert data['complete'] and report['complete']
    assert report['schema'] == 'menia-confidence-ranking-summary-v1'
    assert report['origin'] == data['header']['origin']
    assert report['planHash'] == data['header']['planHash'] and report['sourceHash'] == data['header']['sourceHash']
    assert report['chainEnd'] == data['chainEnd']
    assert report['recordedCalls'] == data['recorded'] == 23040
    assert report['trainingSteps'] == len(data['steps']) == 1296
    assert len(data['answers']) == 4608 and len(data['judgments']) == 18432
    assert len(report['contrasts']) == 18
    differences = []; bootstrap_compared = 0
    def compare(actual, expected, name):
        if isinstance(expected, dict):
            assert type(actual) is dict and set(actual) == set(expected), name
            for k in expected: compare(actual[k], expected[k], name+'/'+k)
        elif isinstance(expected, list):
            assert type(actual) is list and len(actual) == len(expected), name
            for i, (a, b) in enumerate(zip(actual, expected)): compare(a, b, name+'/'+str(i))
        elif type(expected) is float:
            assert type(actual) in (float, int) and math.isfinite(actual) and math.isfinite(expected), name
            error = abs(actual-expected); assert error <= 1e-10, (name, actual, expected)
            differences.append(error)
        else:
            assert type(actual) is type(expected) and actual == expected, (name, actual, expected)
    expected_repetitions = {}; expected_contrasts = []; expected_gates = {}
    for rep in range(3):
        tasks = [t for t in data['header']['plan']['tasks'] if t['replication'] == rep]
        n = len(tasks); assert n == 384
        cells = np.array([CELLS.index((t['family'], t['level'])) for t in tasks])
        # Regenerate the prespecified draws; do not call the analysis sampler.
        draws = np.zeros((CRITERIA['bootstrapResamples']+1, n), dtype=np.int64); draws[0] = 1
        rng = np.random.default_rng(SEED+8000+rep)
        for cell in range(6):
            ids = np.flatnonzero(cells == cell); assert len(ids) == 64
            draws[1:, ids] = rng.multinomial(64, [1/64]*64, size=CRITERIA['bootstrapResamples'])
        scores = {}; accuracy = {}; coverage = {}; formats = {}
        for producer in MODEL_ARMS:
            truth = []
            for task in tasks:
                text = data['answers'][(task['id'], producer)]['text'].strip()
                solution = (sum(char == 'A' for char in task['letters']) if task['family'] == 'countA' else
                            sum(value if index % 2 == 0 else -value for index, value in enumerate(task['operands'])))
                truth.append(int(re.fullmatch(r'-?(?:0|[1-9][0-9]*)', text) is not None and int(text) == solution))
            accuracy[producer] = sum(truth)/n; coverage[producer] = []
            for cell, (family, level) in enumerate(CELLS):
                ids = [i for i in range(n) if cells[i] == cell]; good = sum(truth[i] for i in ids)
                coverage[producer].append(dict(family=family, level=level, n=len(ids), correct=good,
                                               incorrect=len(ids)-good, pairs=good*(len(ids)-good)))
            result = {}; aucs = {}
            for judge in MODEL_ARMS:
                probabilities = [data['judgments'][(t['id'], judge, producer)]['scores']['conditionalCorrect'] for t in tasks]
                auc, pairs = explicit_auc(truth, probabilities, cells, draws); aucs[judge] = auc
                bootstrap_compared += CRITERIA['bootstrapResamples']
                result[judge] = dict(independent_metrics(truth, probabilities),
                    withinAuc=float(auc[0]) if np.isfinite(auc[0]) else None, withinPairs=int(pairs[0]),
                    undefinedBootstrapSamples=int(np.sum(~np.isfinite(auc[1:]))), withinCell={})
                for cell, (family, level) in enumerate(CELLS):
                    ids = [i for i in range(n) if cells[i] == cell]
                    result[judge]['withinCell'][f'{family}/{level}'] = independent_metrics(
                        [truth[i] for i in ids], [probabilities[i] for i in ids])
            own = [data['judgments'][(t['id'], 'rank', producer)]['scores'] for t in tasks]
            formats[producer] = dict(topCodeFraction=sum(s['topIsCode'] for s in own)/n,
                                    meanCodeMass=math.fsum(s['candidateMass'] for s in own)/n)
            scores[producer] = result
            if producer not in CRITERIA['primaryProducers']: continue
            for comparator in CRITERIA['comparators']:
                delta = aucs['rank']-(.5 if comparator == 'constant' else aucs[comparator])
                finite = bool(np.isfinite(delta).all()); tail = .05/36
                bounds = np.quantile(delta[1:], [tail, 1-tail]).tolist() if finite else None
                ordinary = np.quantile(delta[1:], [.025, .975]).tolist() if finite else None
                gain = float(delta[0]) if np.isfinite(delta[0]) else None
                minimum = .1 if comparator == 'constant' else .05
                expected_contrasts.append(dict(replication=rep, producer=producer, comparator=comparator,
                    withinAucGain=gain, minimumGain=minimum, familyInterval=bounds, interval95=ordinary,
                    passed=bool(finite and gain >= minimum and bounds[0] > 0)))
        enough = all(sum(min(c['correct'], c['incorrect']) >= 5 for c in coverage[a]) >= 3 and
                     min(sum(c['correct'] for c in coverage[a]), sum(c['incorrect'] for c in coverage[a])) >= 20
                     for a in ('base', 'rank'))
        native = all(formats[a]['topCodeFraction'] >= .95 and formats[a]['meanCodeMass'] >= .5 for a in ('base', 'rank'))
        expected_gates[str(rep)] = dict(coveragePassed=enough, nativeFormatPassed=native,
            answerAccuracyPointGate=accuracy['rank']-accuracy['base'] >= -.02,
            allPrimaryPassed=all(c['passed'] for c in expected_contrasts if c['replication'] == rep))
        expected_repetitions[str(rep)] = dict(answerAccuracy=accuracy, scores=scores, coverage=coverage, nativeFormat=formats)
    compare(report['replications'], expected_repetitions, 'replications')
    compare(report['contrasts'], expected_contrasts, 'contrasts')
    compare(report['gates'], expected_gates, 'gates')
    compare(report['rankingCriterion'], all(all(g.values()) for g in expected_gates.values()), 'criterion')
    weights = verify_weights(data, path) if check_weights else {}
    return dict(schema='menia-confidence-ranking-core-audit-v1', origin=report['origin'], verified=True, weightsChecked=check_weights,
                weights=weights, journalSHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                planHash=report['planHash'], sourceHash=report['sourceHash'],
                auditSourceSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                maxAbsoluteDifference=max(differences, default=0.), comparisonsChecked=18,
                bootstrapAucsRecomputed=bootstrap_compared,
                scope='Separate grading, pair-matrix AUROC, Brier, reliability, bootstrap intervals and gate arithmetic. '
                      'Shared journal reader, fixed plan and NumPy random generator. Not an external replication; '
                      'adapter file checks do not independently attest the unchanged base weights.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path); parser.add_argument('summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.journal, json.loads(args.summary.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(verified=report['verified'], weights=len(report['weights']),
                         maxAbsoluteDifference=report['maxAbsoluteDifference'])))
