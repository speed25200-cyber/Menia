"""Independent arithmetic for diagnostic 13; shares only the strict reader and frozen plan."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import fmean

import numpy as np

from research import composition_diagnostic as s
from research.iphone_coupling_report import require, strict_json


def ci(values): return np.quantile(values, [.025, .975]).tolist()


def recompute(journal):
    h, rows, pending, events = s.read_journal(journal)
    require(pending is None and len(rows) == len(h['plan']['evaluation']), 'Incomplete diagnostic')
    indexed = {r['request']['id']: r['result'] for r in rows}
    choices = h['metadata']['choiceTokenIds']; tables = {}; contrasts = []; shams = 0
    for rep in range(s.CONFIG['replications']):
        blocks = [b for b in h['plan']['blocks'] if b['replication'] == rep]
        rng = np.random.default_rng(s.SEED+900+rep)
        draws = np.concatenate([rng.choice([i for i,b in enumerate(blocks) if b['marker'] == label],
                    (s.CONFIG['resamples'], sum(b['marker'] == label for b in blocks))) for label in (0,1)], axis=1)
        natives, externals, zeros = {}, {}, {}
        def add(kind, values, **fields):
            contrasts.append(dict(replication=rep, kind=kind, difference=fmean(values),
                                   interval95=ci(np.array(values)[draws].mean(axis=1)), **fields))
        for arm, form, (family, task), mapping in itertools.product(s.ARMS, s.FORMATS, s.STREAMS, (0,1)):
            measurements = []; presence_scores = []; block_acc = []
            for b in blocks:
                prefix = f"{rep}/{arm}/{family}/{b['id']}/{form}/{task}/{mapping}/"
                a, sham = (indexed[prefix+str(p)] for p in (0,3))
                require(all(a[k] == sham[k] for k in ('choiceLogits','choiceMass','rawTokenId','rawChoice')), 'Independent sham failure')
                shams += 1; scores = []; acc = []
                for position in (0,1,2):
                    row = indexed[prefix+str(position)]
                    truth = int(position != 0) if task == 'monitor' else int(b['marker'] == int(task == 'marker_second'))
                    target = truth ^ mapping
                    z = row['choiceLogits']; pair = [math.exp(x-max(z[:2])) for x in z[:2]]
                    p = [x/sum(pair) for x in pair]
                    all_exp = [math.exp(x-max(z)) for x in z]
                    prediction = int(z[1] > z[0])
                    native_correct = int(row['rawTokenId'] == choices[target])
                    acc.append(native_correct)
                    scores.append((z[1]-z[0]) * (-1 if mapping else 1))
                    measurements.append(dict(correct=int(prediction == target), firstTokenCorrect=native_correct,
                        firstTokenIsOption=int(row['rawTokenId'] in choices[:2]),
                        choiceMass=row['choiceMass']*sum(all_exp[:2])/sum(all_exp), fourDigitMass=row['choiceMass'],
                        brier=sum((p[i]-int(i == target))**2 for i in (0,1))))
                presence_scores.append(scores)
                block_acc.append((acc[0]+fmean(acc[1:]))/2 if task == 'monitor' else fmean(acc))
            natives[(arm,form,task,mapping)] = np.array(block_acc)
            table = {k: fmean(m[k] for m in measurements) for k in measurements[0]}
            table.update(n=3*len(blocks), blocks=len(blocks), nativeBalancedAccuracy=fmean(block_acc),
                         nativeBalancedAccuracyInterval95=ci(np.array(block_acc)[draws].mean(axis=1)))
            if task == 'monitor':
                # Row = absent source block, column = present source block, both present positions kept together.
                wins = np.array([[sum(float(v > absent[0]) + .5*float(v == absent[0]) for v in present[1:])
                                  for present in presence_scores] for absent in presence_scores])
                table.update(presenceAUROC=float(wins.mean()/2),
                             presenceAUROCInterval95=ci(wins[draws[:,:,None],draws[:,None,:]].mean(axis=(1,2))/2))
                if arm == 'composed':
                    threshold = h['plan']['calibration']['fits'][f'{rep}/{mapping}']['threshold']
                    for label, t, target_dict in (('calibrated',threshold,externals),('zeroThreshold',0.,zeros)):
                        accuracy = np.array([(float(v[0] <= t)+(float(v[1] > t)+float(v[2] > t))/2)/2 for v in presence_scores])
                        target_dict[(form,mapping)] = accuracy
                        table[label+'BalancedAccuracy'] = fmean(accuracy)
                        table[label+'Interval95'] = ci(accuracy[draws].mean(axis=1))
                    table['externalThreshold'] = threshold
            tables[f'{rep}/{arm}/{family}/{form}/{task}/{mapping}'] = table
        for arm, task, mapping in itertools.product(s.ARMS, ('marker_first','marker_second'), (0,1)):
            delta = natives[(arm,'paraphrase',task,mapping)]-natives[(arm,'trained',task,mapping)]
            add('publicWordingChange',delta,arm=arm,task=task,mapping=mapping)
            if arm != 'base':
                base_delta = natives[('base','paraphrase',task,mapping)]-natives[('base','trained',task,mapping)]
                add('publicWordingInteractionVsBase',delta-base_delta,arm=arm,task=task,mapping=mapping)
        for form, mapping in itertools.product(s.FORMATS, (0,1)):
            for ref, values in (('nativeFirstToken',natives[('composed',form,'monitor',mapping)]),('zeroThreshold',zeros[(form,mapping)])):
                add('externalThresholdGain',externals[(form,mapping)]-values,arm='composed',format=form,mapping=mapping,reference=ref)
    return dict(tables=tables, contrasts=contrasts, shamPairs=shams, trainingUpdates=0,
                recorded=len(rows), planned=len(h['plan']['evaluation']),
                errors=sum(e['event'] == 'error' for e in events),
                interruptedRequests=sum(e['event'] == 'interrupted_request' for e in events),
                evaluationSeconds=math.fsum(r['result']['seconds'] for r in rows))


def compare(expected, actual, path='root'):
    if isinstance(expected, dict):
        require(isinstance(actual,dict) and set(actual) == set(expected), 'Different fields: '+path)
        return max((compare(v,actual[k],path+'/'+k) for k,v in expected.items()), default=0.)
    if isinstance(expected, list):
        require(isinstance(actual,list) and len(actual) == len(expected), 'Different lengths: '+path)
        return max((compare(a,b,path+f'/{i}') for i,(a,b) in enumerate(zip(expected,actual))), default=0.)
    if type(expected) in (int,float):
        require(type(actual) in (int,float) and math.isfinite(actual), 'Nonfinite metric: '+path)
        diff = abs(expected-actual)
        require(diff <= 1e-10*max(1.,abs(expected)), 'Independent calculation mismatch: '+path)
        return diff
    require(actual == expected, 'Value mismatch: '+path); return 0.


def verify(journal, report):
    require(report['complete'], 'Summary is incomplete')
    recalculated = recompute(journal)
    difference = compare(recalculated,{k:report[k] for k in recalculated})
    h, _, _, _ = s.read_journal(journal)
    require(report['planHash'] == h['planHash'] and report['sourceHash'] == h['sourceHash'], 'Summary provenance mismatch')
    require(report['origin'] == h['metadata']['origin'] and report['calibration'] == h['plan']['calibration']
            and report['checkpoints'] == h['plan']['checkpoints'], 'Summary lineage mismatch')
    return dict(schema='menia-composition-diagnostic-verification-v1', verified=True,
        origin=report['origin'], journalSHA256=hashlib.sha256(Path(journal).read_bytes()).hexdigest(),
        planHash=h['planHash'], sourceHash=h['sourceHash'], tables=len(recalculated['tables']),
        contrasts=len(recalculated['contrasts']), shamPairs=recalculated['shamPairs'], maxAbsoluteDifference=difference,
        scope='Independent targets, probabilities, AUROC, class-balanced decisions, paired bootstrap and contrasts; shared strict reader and plan. Not an external replication.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('journal',type=Path)
    p.add_argument('--summary',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a = p.parse_args(); result = verify(a.journal,strict_json(a.summary.read_text(encoding='utf-8')))
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
