"""Separate arithmetic for Colab17; shares the frozen plan, reader and fit check."""
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean

import numpy as np

from research.audit_composition_diagnostic import compare
from research.cross_model_prediction import digest
from research.iphone_coupling_report import require, strict_json
from research.natural_error_journal import read_journal


def correct(row):
    task, result = row['task'], row['result']
    require(result['status'] == 'ok', 'Technical errors cannot be behavioral labels')
    text = result['text'].strip()
    target = sum(letter == 'A' for letter in task['letters']) if task['family'] == 'countA' else sum(
        number if i % 2 == 0 else -number for i, number in enumerate(task['operands']))
    return int(re.fullmatch(r'-?(?:0|[1-9][0-9]*)', text) is not None and int(text) == target)


def scores(rows, name):
    y = [correct(row) for row in rows]
    p = [row['predictions'][name] for row in rows]
    good = [v for v, label in zip(p, y) if label]
    bad = sorted(v for v, label in zip(p, y) if not label)
    wins = math.fsum(bisect_left(bad, v) + .5*(bisect_right(bad, v)-bisect_left(bad, v)) for v in good)
    auc = wins/(len(good)*len(bad)) if good and bad else None
    reliability = []
    for i in range(10):
        indexes = [j for j,v in enumerate(p) if min(math.floor(v*10), 9) == i]
        reliability.append(dict(lower=i/10, upper=(i+1)/10, n=len(indexes),
            meanProbability=fmean(p[j] for j in indexes) if indexes else None,
            accuracy=fmean(y[j] for j in indexes) if indexes else None))
    return dict(n=len(rows), correct=sum(y), brier=fmean((v-label)**2 for v,label in zip(p,y)),
        meanProbability=fmean(p), auc=auc, direct=sum(v >= .8 for v in p),
        wrongDirect=sum(v >= .8 and label == 0 for v,label in zip(p,y)),
        assumedPointLoss=fmean(1-label if v >= .8 else .2 for v,label in zip(p,y)), reliability=reliability)


def quantiles(values, probabilities):
    """Linear interpolation of sorted order statistics, not the primary quantile call."""
    ordered = np.sort(values)
    result = []
    for p in probabilities:
        position = (len(ordered)-1)*p
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        result.append(float(ordered[lower]+(position-lower)*(ordered[upper]-ordered[lower])))
    return result


def recompute(path):
    data = read_journal(path)
    plan, rows = data['header']['plan'], data['rows']
    require(data['pending'] is None and not data['failed'] and len(rows) == plan['plannedGenerations'], 'Incomplete collection')
    cells = [tuple(c) for c in plan['questionPlan']['cells']]
    names = plan['forecasts']
    reports, contrasts, gates = {}, [], {}
    tail = plan['familyAlpha']/(2*plan['primaryComparisons'])
    for rep in range(plan['questionPlan']['replications']):
        selected = [r for r in rows if r['task']['replication'] == rep and r['task']['split'] == 'test']
        require(len(selected) == len(cells)*plan['questionPlan']['countsPerCell']['test'], 'Test partition')
        labels = [correct(r) for r in selected]
        reports[str(rep)] = dict(recordedTest=len(selected), plannedTest=len(selected),
            scores={name:scores(selected,name) for name in names},
            withinCell={f'{c[0]}/{c[1]}': {name:scores([r for r in selected if (r['task']['family'],r['task']['level']) == c],name)
                                        for name in names} for c in cells},
            selection=data['bundles'][rep]['selection'])
        groups = [[i for i,r in enumerate(selected) if (r['task']['family'],r['task']['level']) == c] for c in cells]
        rng = np.random.default_rng(plan['questionPlan']['seed']+401+rep)
        # Draw positions, then map them to row indices separately within each stratum.
        draws = np.concatenate([np.asarray(g)[rng.integers(0,len(g),size=(plan['bootstrapResamples'],len(g)))] for g in groups],axis=1)
        own = [(r['predictions']['internal']-y)**2 for r,y in zip(selected,labels)]
        for name in names:
            if name == 'internal':
                continue
            changes = [(r['predictions'][name]-y)**2-loss for r,y,loss in zip(selected,labels,own)]
            boot = np.asarray(changes)[draws].sum(axis=1)/len(selected)
            ci = quantiles(boot,[tail,1-tail])
            mean = fmean(changes)
            primary = name in plan['primaryComparators']
            contrasts.append(dict(replication=rep, comparison=name+'-internal', primary=primary,
                otherMinusInternalBrier=mean, interval95=quantiles(boot,[.025,.975]),
                primaryFamilyInterval=ci if primary else None,
                primaryPassed=bool(mean >= plan['minimumBrierGain'] and ci[0] > 0) if primary else None))
        gates[str(rep)] = dict(complete=True,
            enoughBothClasses=min(sum(labels),len(labels)-sum(labels)) >= plan['minimumSuccessesAndErrorsPerRepetition'],
            allPrimaryPassed=all(c['primaryPassed'] for c in contrasts if c['replication'] == rep and c['primary']))
    return data, dict(complete=True, recorded=len(rows), planned=plan['plannedGenerations'], pendingRequest=False,
        statuses=dict(Counter(r['result']['status'] for r in rows)), frozenFits=len(data['bundles']), fitChecked=data['fitChecked'],
        bundleHashes={str(k):digest(v) for k,v in data['bundles'].items()},
        replications=reports, contrasts=contrasts, prerequisites=gates,
        replicatedIncrementalReadoutCriterion=bool(data['fitChecked'] and all(g['enoughBothClasses'] and g['allPrimaryPassed'] for g in gates.values())),
        familyIntervalCoverage=1-plan['familyAlpha']/plan['primaryComparisons'], journalChainEnd=data['previous'])


def verify(path, report):
    require(report['schema'] == 'menia-natural-error-report-v1' and report['complete'], 'Incomplete or wrong report')
    data, calculated = recompute(path)
    discrepancy = compare(calculated, {k:report[k] for k in calculated})
    header = data['header']
    require(all(report[k] == header[k] for k in ('origin','planHash','sourceHash')), 'Summary provenance')
    return dict(schema='menia-natural-error-verification-v1', verified=True, origin=header['origin'],
        journalSHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), planHash=header['planHash'], sourceHash=header['sourceHash'],
        recorded=calculated['recorded'], frozenFits=calculated['frozenFits'],
        overallTables=len(calculated['replications'])*len(header['plan']['forecasts']),
        withinCellTables=sum(len(r['withinCell'])*len(header['plan']['forecasts']) for r in calculated['replications'].values()),
        contrasts=len(calculated['contrasts']), primaryContrasts=sum(c['primary'] for c in calculated['contrasts']),
        maxAbsoluteDifference=discrepancy,
        scope='Separate exact-answer grading, scalar Brier and reliability, rank-count AUROC, paired stratified bootstrap, interpolated quantiles and gates. Shared plan, strict reader, fit/forecast reconstruction and numerical libraries; not an external replication.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal',type=Path)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = verify(args.journal,strict_json(args.summary.read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
