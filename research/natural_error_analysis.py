"""Fixed prospective Brier comparisons; outcome scores never enter the collector."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from research.cross_model_prediction import CELLS, grade, digest
from research.iphone_coupling_report import require
from research.natural_error_questions import SEED
from research.natural_error_readouts import FORECASTS
from research.natural_error_journal import read_journal


def metrics(rows, name):
    if not rows:
        return dict(n=0, correct=0, brier=None, auc=None, meanProbability=None,
                    direct=0, wrongDirect=0, assumedPointLoss=None, reliability=[])
    y = np.asarray([grade(r['result'], r['task']) for r in rows], dtype=float)
    p = np.asarray([r['predictions'][name] for r in rows])
    good, bad = p[y == 1], p[y == 0]
    auc = float(np.mean((good[:, None] > bad) + .5*(good[:, None] == bad))) if len(good) and len(bad) else None
    bins = np.minimum((p*10).astype(int), 9)
    reliability = []
    for i in range(10):
        idx = bins == i
        reliability.append(dict(lower=i/10, upper=(i+1)/10, n=int(idx.sum()),
            meanProbability=float(p[idx].mean()) if idx.any() else None,
            accuracy=float(y[idx].mean()) if idx.any() else None))
    return dict(n=len(rows), correct=int(y.sum()), brier=float(np.mean((p-y)**2)),
        meanProbability=float(p.mean()), auc=auc, direct=int((p >= .8).sum()),
        wrongDirect=int(((p >= .8) & (y == 0)).sum()),
        assumedPointLoss=float(np.mean(np.where(p >= .8, 1-y, .2))), reliability=reliability)


def analyze(path, *, check_fit=True):
    data = read_journal(path, check_fit=check_fit)
    header, rows = data['header'], data['rows']
    plan = header['plan']
    results, contrasts, gates = {}, [], {}
    primary_count = plan['primaryComparisons']
    tail = plan['familyAlpha']/(2*primary_count)
    for rep in range(plan['questionPlan']['replications']):
        test = [r for r in rows if r['task']['replication'] == rep and r['task']['split'] == 'test' and r['result']['status'] == 'ok']
        expected = plan['questionPlan']['countsPerCell']['test']*len(CELLS)
        scores = {name: metrics(test, name) for name in FORECASTS}
        results[str(rep)] = dict(recordedTest=len(test), plannedTest=expected, scores=scores,
            withinCell={f'{c[0]}/{c[1]}': {name: metrics([r for r in test if (r['task']['family'], r['task']['level']) == c], name)
                                         for name in FORECASTS} for c in CELLS},
            selection=data['bundles'][rep]['selection'] if rep in data['bundles'] else None)
        if len(test) != expected:
            gates[str(rep)] = dict(complete=False, enoughBothClasses=False, allPrimaryPassed=False)
            continue
        y = np.asarray([grade(r['result'], r['task']) for r in test], dtype=float)
        groups = [np.asarray([i for i, r in enumerate(test) if (r['task']['family'], r['task']['level']) == c]) for c in CELLS]
        require(all(len(g) == plan['questionPlan']['countsPerCell']['test'] for g in groups), 'Test cell counts')
        rng = np.random.default_rng(SEED+401+rep)
        draws = np.concatenate([rng.choice(g, (plan['bootstrapResamples'], len(g)), replace=True) for g in groups], axis=1)
        own = (np.asarray([r['predictions']['internal'] for r in test])-y)**2
        for name in FORECASTS:
            if name == 'internal':
                continue
            delta = (np.asarray([r['predictions'][name] for r in test])-y)**2-own
            samples = delta[draws].mean(1)
            interval = np.quantile(samples, [tail, 1-tail]).tolist()
            primary = name in plan['primaryComparators']
            contrasts.append(dict(replication=rep, comparison=name+'-internal', primary=primary,
                otherMinusInternalBrier=float(delta.mean()), interval95=np.quantile(samples, [.025, .975]).tolist(),
                primaryFamilyInterval=interval if primary else None,
                primaryPassed=bool(delta.mean() >= plan['minimumBrierGain'] and interval[0] > 0) if primary else None))
        enough = min(int(y.sum()), int(len(y)-y.sum())) >= plan['minimumSuccessesAndErrorsPerRepetition']
        gates[str(rep)] = dict(complete=True, enoughBothClasses=enough,
            allPrimaryPassed=all(c['primaryPassed'] for c in contrasts if c['replication'] == rep and c['primary']))
    complete = len(rows) == plan['plannedGenerations'] and data['pending'] is None and not data['failed']
    criterion = bool(complete and data['fitChecked'] and all(g['enoughBothClasses'] and g['allPrimaryPassed'] for g in gates.values()))
    return dict(schema='menia-natural-error-report-v1', origin=header['origin'], planHash=header['planHash'],
        sourceHash=header['sourceHash'], journalChainEnd=data['previous'], complete=complete,
        recorded=len(rows), planned=plan['plannedGenerations'], pendingRequest=data['pending'] is not None,
        statuses=dict(Counter(r['result']['status'] for r in rows)), frozenFits=len(data['bundles']),
        fitChecked=data['fitChecked'], bundleHashes={str(k):digest(v) for k,v in data['bundles'].items()},
        replications=results, contrasts=contrasts, prerequisites=gates,
        replicatedIncrementalReadoutCriterion=criterion,
        criterionScope='Nine predeclared Brier gains >=0.005, each lower Bonferroni bootstrap bound >0, at least20 successes and20 errors per repetition. Approximate conditional bootstrap, not an exact guarantee.',
        familyIntervalCoverage=1-plan['familyAlpha']/primary_count,
        limitations='Same pretrained weights, three disjoint reader fits. External question-conditioned forecasts. Sampling variance from one answer per question. No native use, out-of-domain guarantee, subjective experience or novelty inference. Decision losses assume an ideal verification cost; no action executed.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.journal)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('complete','recorded','planned','statuses','frozenFits','fitChecked')}))
