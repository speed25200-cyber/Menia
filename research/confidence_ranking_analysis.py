"""Within-cell AUROC with paired whole-question stratified bootstrap.

Pairs define a statistic, not independent observations. Bootstrap multiplicities
are applied to original questions and shared by all judges of a producer.
"""
import numpy as np

from research.confidence_ranking_study import CRITERIA, MODEL_ARMS, SEED, evaluation_rows
from research.answer_confidence_analysis import metrics
from research.cross_model_prediction import CELLS, grade
from research.iphone_coupling_report import require


def weighted_within_auc(y, scores, cells, weights):
    y = np.asarray(y); scores = np.asarray(scores, dtype=float)
    cells = np.asarray(cells); weights = np.asarray(weights, dtype=float)
    require(y.ndim == scores.ndim == cells.ndim == 1 and y.shape == scores.shape == cells.shape,
            'Ranking dimensions')
    require(len(y) > 0 and np.all((y == 0) | (y == 1)) and np.isfinite(scores).all(), 'Ranking values')
    require(weights.ndim == 2 and weights.shape[1] == len(y) and
            np.isfinite(weights).all() and np.all(weights >= 0), 'Question multiplicities')
    numerator = np.zeros(len(weights)); denominator = np.zeros(len(weights))
    for cell in np.unique(cells):
        ids = np.flatnonzero(cells == cell)
        order = ids[np.argsort(scores[ids], kind='stable')]
        labels = y[order]; w = weights[:, order]
        good = w*labels; bad = w*(1-labels)
        denominator += good.sum(axis=1)*bad.sum(axis=1)
        # Aggregate equal scores before cumulative sums: exact half-credit ties.
        starts = np.r_[0, 1+np.flatnonzero(np.diff(scores[order]) != 0)]
        good_groups = np.add.reduceat(good, starts, axis=1)
        bad_groups = np.add.reduceat(bad, starts, axis=1)
        lower = np.cumsum(bad_groups, axis=1)-bad_groups
        numerator += np.sum(good_groups*(lower+.5*bad_groups), axis=1)
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0), denominator


def question_weights(cells, resamples, seed):
    cells = np.asarray(cells); rng = np.random.default_rng(seed)
    result = np.zeros((resamples, len(cells)), dtype=np.int64)
    for cell in np.unique(cells):
        ids = np.flatnonzero(cells == cell)
        result[:, ids] = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=resamples)
    return result


def analyze(path):
    from research.confidence_ranking_journal import read_journal
    d = read_journal(path); require(d['complete'], 'Complete reconstructed trial required')
    p = d['header']['plan']; results = {}; contrasts = []; gates = {}
    for rep in range(3):
        rows = evaluation_rows(p, d['answers'], d['judgments'], rep)
        cells = np.asarray([CELLS.index((r['task']['family'], r['task']['level'])) for r in rows])
        w = question_weights(cells, CRITERIA['bootstrapResamples'], SEED+8000+rep)
        all_weights = np.concatenate((np.ones((1, len(rows))), w), axis=0)
        result = {}; accuracy = {}; coverage = {}; formats = {}
        for producer in MODEL_ARMS:
            y = np.asarray([int(grade(r['answers'][producer], r['task'])) for r in rows])
            accuracy[producer] = float(y.mean()); score_rows = {}; auc_draws = {}
            coverage[producer] = []
            for i, (family, level) in enumerate(CELLS):
                idx = cells == i; good = int(y[idx].sum()); n = int(idx.sum())
                coverage[producer].append(dict(family=family, level=level, n=n, correct=good,
                                               incorrect=n-good, pairs=good*(n-good)))
            for judge in MODEL_ARMS:
                ps = np.asarray([r['judgments'][judge][producer]['conditionalCorrect'] for r in rows])
                auc, pairs = weighted_within_auc(y, ps, cells, all_weights); auc_draws[judge] = auc
                score_rows[judge] = dict(metrics(y, ps), withinAuc=float(auc[0]) if np.isfinite(auc[0]) else None,
                    withinPairs=int(pairs[0]), undefinedBootstrapSamples=int(np.sum(~np.isfinite(auc[1:]))),
                    withinCell={f'{family}/{level}': metrics(y[cells == i], ps[cells == i])
                                for i, (family, level) in enumerate(CELLS)})
            own = [r['judgments']['rank'][producer] for r in rows]
            formats[producer] = dict(topCodeFraction=float(np.mean([x['topIsCode'] for x in own])),
                                     meanCodeMass=float(np.mean([x['candidateMass'] for x in own])))
            if producer in CRITERIA['primaryProducers']:
                for comparator in CRITERIA['comparators']:
                    reference = np.full_like(auc_draws['rank'], .5) if comparator == 'constant' else auc_draws[comparator]
                    delta = auc_draws['rank']-reference
                    finite = bool(np.isfinite(delta).all())
                    tail = CRITERIA['familyAlpha']/(2*CRITERIA['primaryComparisons'])
                    bounds = np.quantile(delta[1:], [tail, 1-tail]).tolist() if finite else None
                    gain = float(delta[0]) if np.isfinite(delta[0]) else None
                    minimum = CRITERIA['minimumGainOverConstant'] if comparator == 'constant' else CRITERIA['minimumAucGain']
                    contrasts.append(dict(replication=rep, producer=producer, comparator=comparator,
                        withinAucGain=gain, minimumGain=minimum, familyInterval=bounds,
                        interval95=np.quantile(delta[1:], [.025, .975]).tolist() if finite else None,
                        passed=bool(finite and gain >= minimum and bounds[0] > 0)))
            result[producer] = score_rows
        enough = all(sum(min(c['correct'], c['incorrect']) >= 5 for c in coverage[a]) >= CRITERIA['minimumCellsWithFiveEach']
                     and min(sum(c['correct'] for c in coverage[a]), sum(c['incorrect'] for c in coverage[a])) >= CRITERIA['minimumEachClass']
                     for a in CRITERIA['primaryProducers'])
        format_ok = all(formats[a]['topCodeFraction'] >= CRITERIA['minimumNativeTopCodeFraction'] and
                        formats[a]['meanCodeMass'] >= CRITERIA['minimumNativeMeanCodeMass'] for a in CRITERIA['primaryProducers'])
        gates[str(rep)] = dict(coveragePassed=enough, nativeFormatPassed=format_ok,
            answerAccuracyPointGate=accuracy['rank']-accuracy['base'] >= CRITERIA['minimumAnswerAccuracyChange'],
            allPrimaryPassed=all(c['passed'] for c in contrasts if c['replication'] == rep))
        results[str(rep)] = dict(answerAccuracy=accuracy, scores=result, coverage=coverage, nativeFormat=formats)
    require(len(contrasts) == 18, 'Primary family')
    return dict(schema='menia-confidence-ranking-summary-v1', complete=True, origin=d['header']['origin'],
                planHash=d['header']['planHash'], sourceHash=d['header']['sourceHash'], chainEnd=d['chainEnd'],
                recordedCalls=d['recorded'], trainingSteps=len(d['steps']), replications=results,
                contrasts=contrasts, gates=gates, rankingCriterion=all(all(g.values()) for g in gates.values()),
                scope='Uncalibrated within-category textual correctness ranking. Approximate question bootstrap '
                      'conditional on task categories and one sampled response; point accuracy gate is not '
                      'a noninferiority confidence bound. No causal action or consciousness claim.')
