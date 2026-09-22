"""Independent arithmetic for experiment 16, sharing its frozen plan and strict reader."""
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import fmean

import numpy as np

from research import cross_task_interchange as study
from research.audit_composition_diagnostic import compare
from research.iphone_coupling_report import require, strict_json


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def signature(out):
    return tuple(out[k] for k in ('rawTokenId', 'rawChoice', 'choiceLogits', 'choiceMass'))


def recompute(journal):
    header, completed, pending, events = study.read_journal(journal)
    design = header['plan']
    require(pending is None and len(completed) == len(design['groups']), 'Incomplete experiment')
    digits = header['metadata']['choiceTokenIds']
    pair_by_id = {p['id']: p for p in design['pairs']}
    cells = {}
    shams = copies = crossed = 0
    for group in completed:
        pair = pair_by_id[group['group']['pair']]
        cells.setdefault((pair['replication'], pair['split'], group['group']['arm']), []).append((pair, group['outputs']))
        intact = {(o['request']['task'], o['request']['role'], o['request']['state'], o['request']['code']): o
                  for o in group['outputs'] if o['request']['kind'] == 'intact'}
        for out in group['outputs']:
            r = out['request']
            if r['kind'] == 'sham':
                original = intact[(r['task'], 'recipient', r['recipientState'], r['recipientCode'])]
                require(signature(out) == signature(original), 'Independent self-sham failure')
                shams += 1
            elif r['kind'] == 'transfer':
                dt = r.get('donorTask', r['task'])
                crossed += int(dt != r['task'])
                if r['site'] == 35:
                    original = intact[(dt, 'donor', r['donorState'], r['donorCode'])]
                    require(signature(out) == signature(original), 'Independent final-copy failure')
                    copies += 1

    tables, baselines, contrasts = {}, {}, []
    for rep, split, arm in itertools.product(range(design['config']['replications']), ('test', 'lexical'), ('prefix', 'base')):
        selected = cells[(rep, split, arm)]
        count = len(selected)
        rng = np.random.default_rng(design['seed'] + 10 * rep + int(split == 'lexical'))
        strata = [[i for i, (p, _) in enumerate(selected) if (p['donor']['marker'], p['recipient']['marker']) == (d, r)]
                  for d in (0, 1) for r in (0, 1)]
        require(all(strata), 'Missing bootstrap stratum')
        samples = np.concatenate([rng.choice(s, (design['config']['resamples'], len(s))) for s in strata], axis=1)
        for task, code in itertools.product(('monitor', 'marker_first'), (0, 1)):
            pair_scores = []
            for pair, outputs in selected:
                correct = []
                for out in outputs:
                    r = out['request']
                    if r['kind'] != 'intact' or r['task'] != task or r['code'] != code:
                        continue
                    value = r['state'] if task == 'monitor' else 1 - pair[r['role']]['marker']
                    correct.append(float(out['rawTokenId'] == digits[(value + code) % 2]))
                require(len(correct) == 4, 'Incomplete intact factorial')
                pair_scores.append(fmean(correct))
            values = np.asarray(pair_scores)
            baselines[f'{rep}/{split}/{arm}/{task}/{code}'] = dict(
                accuracy=fmean(values), pairs=count, interval95=interval(values[samples].mean(1)))

        for dt, rt, site in itertools.product(('monitor', 'marker_first'), ('monitor', 'marker_first'), design['sites']):
            vectors = []
            for pair, outputs in selected:
                originals = {(o['request']['task'], o['request']['role'], o['request']['state'], o['request']['code']): o
                             for o in outputs if o['request']['kind'] == 'intact'}
                observations = []
                for out in outputs:
                    r = out['request']
                    if r['kind'] != 'transfer' or r['task'] != rt or r.get('donorTask', r['task']) != dt or r['site'] != site:
                        continue
                    # Reconstruct task-indexed facts independently of the original
                    # hypothesis and analysis functions. Changing the receiver's
                    # question selects a different fact, not a different code.
                    donor_facts = {'monitor': r['donorState'], 'marker_first': 1 - pair['donor']['marker']}
                    recipient_facts = {'monitor': r['recipientState'], 'marker_first': 1 - pair['recipient']['marker']}
                    bits = [donor_facts[rt], donor_facts[dt], donor_facts[dt], recipient_facts[rt]]
                    codes = [r['recipientCode'], r['recipientCode'], r['donorCode'], r['recipientCode']]
                    names = ('recipientTaskContent', 'donorBoolean', 'donorAnswer', 'recipientUnchanged')
                    labels = {name: (bit + code) % 2 for name, bit, code in zip(names, bits, codes)}
                    logits = out['choiceLogits']
                    maximum = max(logits)
                    log_sum = math.log(math.fsum(math.exp(v - maximum) for v in logits)) + maximum
                    raw = out['rawTokenId']
                    donor = originals[(dt, 'donor', r['donorState'], r['donorCode'])]
                    recipient = originals[(rt, 'recipient', r['recipientState'], r['recipientCode'])]
                    row = {name: float(raw == digits[label]) for name, label in labels.items()}
                    row.update({name + 'CrossEntropy': -math.log(out['choiceMass']) + log_sum - logits[label]
                                for name, label in labels.items()})
                    row.update(observedDonorCopy=float(raw == donor['rawTokenId']),
                               observedRecipientRetention=float(raw == recipient['rawTokenId']),
                               optionOutput=float(raw in digits[:2]), patchNorm=out['patch']['displacementNorm'])
                    observations.append(row)
                require(len(observations) == 16, 'Incomplete transfer factorial')
                vectors.append({k: fmean(o[k] for o in observations) for k in observations[0]})
            table = {}
            for metric in vectors[0]:
                values = np.asarray([v[metric] for v in vectors])
                table[metric] = dict(mean=fmean(values), interval95=interval(values[samples].mean(1)))
            tables[f'{rep}/{split}/{arm}/{dt}-to-{rt}/{site}'] = dict(table, pairs=count, calls=16 * count)
            for rival in ('donorBoolean', 'donorAnswer', 'recipientUnchanged'):
                delta = np.asarray([v['recipientTaskContent'] - v[rival] for v in vectors])
                contrasts.append(dict(replication=rep, split=split, arm=arm, donorTask=dt, recipientTask=rt, site=site,
                    comparison='recipientTaskContent-' + rival,
                    primary=split == 'test' and arm == 'prefix' and site == 23 and dt != rt,
                    difference=fmean(delta), interval95=interval(delta[samples].mean(1))))

    prerequisites = {}
    for rep in range(design['config']['replications']):
        initial = {f'{task}/{code}': baselines[f'{rep}/test/prefix/{task}/{code}']['accuracy'] >= design['config']['baselineGate']
                   for task in ('monitor', 'marker_first') for code in (0, 1)}
        within = {task: tables[f'{rep}/test/prefix/{task}-to-{task}/23']['recipientTaskContent']['mean'] >= design['config']['withinTaskGate']
                  for task in ('monitor', 'marker_first')}
        prerequisites[str(rep)] = dict(intactChecks=initial, withinTaskChecks=within,
            allPassed=all(initial.values()) and all(within.values()),
            interpretation='Necessary competence and within-task transfer controls; never filters any trial')
    return dict(tables=tables, baseline=baselines, contrasts=contrasts, prerequisites=prerequisites,
                groups=len(completed), plannedGroups=len(design['groups']),
                recorded=sum(len(g['outputs']) for g in completed), planned=design['plannedForwards'],
                selfShams=shams, finalLayerCopies=copies, crossTaskTransfers=crossed,
                errors=sum(e['event'] == 'error' for e in events), restarts=sum(e['event'] == 'group_restart' for e in events))


def verify(journal, report):
    require(report['complete'] and report['schema'] == 'menia-cross-task-interchange-report-v1', 'Incomplete or wrong summary')
    computed = recompute(journal)
    discrepancy = compare(computed, {k: report[k] for k in computed})
    header, _, _, _ = study.read_journal(journal)
    require(all(report[k] == header[k] for k in ('planHash', 'sourceHash'))
            and report['origin'] == header['metadata']['origin'], 'Provenance changed')
    return dict(schema='menia-cross-task-interchange-verification-v1', verified=True, origin=report['origin'],
        journalSHA256=hashlib.sha256(Path(journal).read_bytes()).hexdigest(),
        planHash=header['planHash'], sourceHash=header['sourceHash'],
        tables=len(computed['tables']), baselineTables=len(computed['baseline']),
        contrasts=len(computed['contrasts']), primaryContrasts=sum(c['primary'] for c in computed['contrasts']),
        selfShams=computed['selfShams'], finalLayerCopies=computed['finalLayerCopies'], crossTaskTransfers=computed['crossTaskTransfers'],
        maxAbsoluteDifference=discrepancy,
        scope='Independent four-target coding, full-vocabulary losses, actual-token copy/retention, paired stratified bootstrap, contrasts and both prerequisites. Shared plan, strict reader and numerical libraries; not an external replication or consciousness test.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.journal, strict_json(args.summary.read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
