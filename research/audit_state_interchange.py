"""Independent arithmetic for experiment 15; shares its frozen plan and strict reader."""
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import fmean

import numpy as np

from research import state_interchange as study
from research.audit_composition_diagnostic import compare
from research.iphone_coupling_report import require, strict_json


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def output_signature(out):
    return tuple(out[k] for k in ('rawTokenId', 'rawChoice', 'choiceLogits', 'choiceMass'))


def recompute(journal):
    header, completed, pending, events = study.read_journal(journal)
    design = header['plan']
    require(pending is None and len(completed) == len(design['groups']), 'Incomplete experiment')
    digits = header['metadata']['choiceTokenIds']
    pair_by_id = {p['id']: p for p in design['pairs']}
    cells = {}
    shams = final_copies = 0
    for group in completed:
        pair = pair_by_id[group['group']['pair']]
        cell = (pair['replication'], pair['split'], group['group']['arm'])
        cells.setdefault(cell, []).append((pair, group['outputs']))
        intact = {(o['request']['task'], o['request']['role'], o['request']['state'],
                   o['request']['code']): o for o in group['outputs'] if o['request']['kind'] == 'intact'}
        for out in group['outputs']:
            request = out['request']
            if request['kind'] == 'sham':
                original = intact[(request['task'], 'recipient', request['recipientState'], request['recipientCode'])]
                require(output_signature(out) == output_signature(original), 'Independent self-sham failure')
                shams += 1
            elif request['kind'] == 'transfer' and request['site'] == 35:
                original = intact[(request['task'], 'donor', request['donorState'], request['donorCode'])]
                require(output_signature(out) == output_signature(original), 'Independent final-copy failure')
                final_copies += 1

    tables, baselines, contrasts = {}, {}, []
    for rep, split, arm, task in itertools.product(
            range(design['config']['replications']), ('test', 'lexical'), ('prefix', 'base'), ('monitor', 'marker_first')):
        selected = cells[(rep, split, arm)]
        count = len(selected)
        # The seed and the four strata are protocol inputs. Target coding, per-pair
        # metrics and contrasts below do not call the original analysis functions.
        rng = np.random.default_rng(design['seed'] + 10 * rep + int(split == 'lexical'))
        strata = [[i for i, (p, _) in enumerate(selected)
                   if (p['donor']['marker'], p['recipient']['marker']) == (d, r)]
                  for d in (0, 1) for r in (0, 1)]
        require(all(strata), 'Missing bootstrap stratum')
        samples = np.concatenate([rng.choice(s, size=(design['config']['resamples'], len(s)))
                                  for s in strata], axis=1)
        for code in (0, 1):
            pair_scores = []
            for pair, outputs in selected:
                correct = []
                for out in outputs:
                    r = out['request']
                    if r['kind'] != 'intact' or r['task'] != task or r['code'] != code:
                        continue
                    value = r['state'] if task == 'monitor' else 1 - pair[r['role']]['marker']
                    target = (value + code) % 2
                    correct.append(int(out['rawTokenId'] == digits[target]))
                require(len(correct) == 4, 'Incomplete intact factorial')
                pair_scores.append(fmean(correct))
            values = np.asarray(pair_scores)
            baselines[f'{rep}/{split}/{arm}/{task}/{code}'] = dict(
                accuracy=fmean(values), pairs=count, interval95=interval(values[samples].mean(axis=1)))

        for layer in design['sites']:
            vectors = []
            for pair, outputs in selected:
                originals = {(o['request']['role'], o['request']['state'], o['request']['code']): o
                             for o in outputs if o['request']['kind'] == 'intact' and o['request']['task'] == task}
                measurements = []
                for out in outputs:
                    r = out['request']
                    if r['kind'] != 'transfer' or r['task'] != task or r['site'] != layer:
                        continue
                    d = r['donorState'] if task == 'monitor' else 1 - pair['donor']['marker']
                    q = r['recipientState'] if task == 'monitor' else 1 - pair['recipient']['marker']
                    targets = {'stateTransfer': (d + r['recipientCode']) % 2,
                               'donorAnswer': (d + r['donorCode']) % 2,
                               'recipientUnchanged': (q + r['recipientCode']) % 2}
                    scores = out['choiceLogits']
                    maximum = max(scores)
                    log_partition = maximum + math.log(math.fsum(math.exp(x - maximum) for x in scores))
                    raw = out['rawTokenId']
                    donor = originals[('donor', r['donorState'], r['donorCode'])]
                    recipient = originals[('recipient', r['recipientState'], r['recipientCode'])]
                    row = {name: float(raw == digits[label]) for name, label in targets.items()}
                    row.update({name + 'CrossEntropy': log_partition - scores[label] - math.log(out['choiceMass'])
                                for name, label in targets.items()})
                    row.update(observedDonorCopy=float(raw == donor['rawTokenId']),
                               observedRecipientRetention=float(raw == recipient['rawTokenId']),
                               optionOutput=float(raw in digits[:2]), patchNorm=out['patch']['displacementNorm'])
                    measurements.append(row)
                require(len(measurements) == 16, 'Incomplete transfer factorial')
                vectors.append({k: fmean(x[k] for x in measurements) for k in measurements[0]})

            key = f'{rep}/{split}/{arm}/{task}/{layer}'
            table = {}
            for metric in vectors[0]:
                values = np.asarray([v[metric] for v in vectors])
                table[metric] = dict(mean=fmean(values), interval95=interval(values[samples].mean(axis=1)))
            tables[key] = dict(table, pairs=count, calls=count * 16)
            for rival in ('donorAnswer', 'recipientUnchanged'):
                differences = np.asarray([v['stateTransfer'] - v[rival] for v in vectors])
                contrasts.append(dict(
                    replication=rep, split=split, arm=arm, task=task, site=layer,
                    comparison='stateTransfer-' + rival,
                    primary=(split, arm, task, layer) == ('test', 'prefix', 'monitor', 23),
                    difference=fmean(differences), interval95=interval(differences[samples].mean(axis=1))))

    prerequisites = {}
    for rep in range(design['config']['replications']):
        tests = {f'{task}/{code}': baselines[f'{rep}/test/prefix/{task}/{code}']['accuracy'] >= design['config']['baselineGate']
                 for task in ('monitor', 'marker_first') for code in (0, 1)}
        prerequisites[str(rep)] = dict(
            intactChecks=tests, allPassed=all(tests.values()),
            interpretation='Necessary native-task competence for primary mechanistic interpretation; never filters trials')
    return dict(tables=tables, baseline=baselines, contrasts=contrasts, prerequisites=prerequisites,
                groups=len(completed), plannedGroups=len(design['groups']),
                recorded=sum(len(g['outputs']) for g in completed), planned=design['plannedForwards'],
                selfShams=shams, finalLayerCopies=final_copies,
                errors=sum(e['event'] == 'error' for e in events),
                restarts=sum(e['event'] == 'group_restart' for e in events))


def verify(journal, report):
    require(report['complete'], 'Incomplete summary')
    computed = recompute(journal)
    discrepancy = compare(computed, {k: report[k] for k in computed})
    header, _, _, _ = study.read_journal(journal)
    require(all(report[k] == header[k] for k in ('planHash', 'sourceHash'))
            and report['origin'] == header['metadata']['origin'], 'Provenance changed')
    return dict(
        schema='menia-state-interchange-verification-v1', verified=True, origin=report['origin'],
        journalSHA256=hashlib.sha256(Path(journal).read_bytes()).hexdigest(),
        planHash=header['planHash'], sourceHash=header['sourceHash'],
        tables=len(computed['tables']), baselineTables=len(computed['baseline']),
        contrasts=len(computed['contrasts']), primaryContrasts=sum(c['primary'] for c in computed['contrasts']),
        selfShams=computed['selfShams'], finalLayerCopies=computed['finalLayerCopies'],
        maxAbsoluteDifference=discrepancy,
        scope='Independent semantic targets, full-vocabulary losses, actual-token copy/retention, option rate, patch norms, stratified paired bootstrap, contrasts and baseline prerequisites. Shared strict reader, frozen plan and numerical libraries; not an external replication or a consciousness test.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checked = verify(args.journal, strict_json(args.summary.read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(checked, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
