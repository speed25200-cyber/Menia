"""Experiment 16: recipient-relevant content versus a donor-task binary answer."""
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np

from research import state_interchange as previous
from research.cross_task_hypotheses import predictions
from research.iphone_coupling_report import require, strict_json

SEED = 202609196
MODEL = previous.MODEL
SITES = previous.SITES
ARMS = previous.ARMS
TASKS = previous.TASKS
CONFIG = dict(replications=3, testPairs=16, lexicalPairs=16, resamples=2000,
              rank=8, layer=17, strength=1., maxInputTokens=512,
              baselineGate=.90, withinTaskGate=.90)
PARENT_SHA256 = previous.PARENT_SHA256
INTERCHANGE_SHA256 = 'f0a225287d58bc3bb3639a7c4051d24e1a43b1e7394ff2b2449d3921a3e16ad5'
digest = previous.digest
checkpoints = previous.checkpoints


def source_hash():
    return digest(dict(previous=previous.source_hash(), files={name: Path(__file__).with_name(name).read_text(encoding='utf-8')
        for name in ('cross_task_interchange.py', 'cross_task_interchange_gpu.py', 'cross_task_hypotheses.py')}))


def plan():
    rng = random.Random(SEED)
    p = previous.previous
    old = p.plan()['blocks'] + p.diagnostic.plan()['blocks'] + p.previous.plan()['blocks']
    for study in (p.previous.previous, p.previous.previous.parent,
                  p.previous.previous.parent.previous, p.previous.previous.parent.previous.previous):
        old += study.plan()['blocks']
    old += p.previous.previous.parent.previous.previous.previous_plan()['blocks']
    old += [pair[role] for pair in previous.plan()['pairs'] for role in ('donor', 'recipient')]
    used = {sentence for block in old for sentence in block['sentences']}
    noises = {block['noiseSeed'] for block in old}
    pools = {split: [f'{a} {b} {c} {d}.' for a, b, c, d in itertools.product(*lexicon)
                    if f'{a} {b} {c} {d}.' not in used]
             for split, lexicon in (('test', p.SAME_LEXICON), ('lexical', p.LEXICON))}
    for pool in pools.values():
        rng.shuffle(pool)
    pairs = []
    for rep in range(CONFIG['replications']):
        for split in ('test', 'lexical'):
            for index in range(CONFIG[split + 'Pairs']):
                # All 16 combinations of the two public markers and the two
                # possible perturbation positions, separately in each split/rep.
                dm, rm, dp, rp = (index >> k & 1 for k in (3, 2, 1, 0))
                pair = dict(id=f'r{rep}-{split}-{index:03d}', replication=rep, split=split, index=index)
                for role, marker, position in (('donor', dm, dp + 1), ('recipient', rm, rp + 1)):
                    noise = rng.randrange(2**31)
                    while noise in noises:
                        noise = rng.randrange(2**31)
                    noises.add(noise)
                    pair[role] = dict(id=pair['id'] + '-' + role, marker=marker, positivePosition=position,
                                      sentences=[pools[split].pop(), pools[split].pop()], noiseSeed=noise)
                pairs.append(pair)
    groups = [dict(id=arm + '/' + pair['id'], arm=arm, pair=pair['id']) for arm in ARMS for pair in pairs]
    return dict(schema='menia-cross-task-interchange-plan-v1', seed=SEED, config=CONFIG,
                model=MODEL, sites=list(SITES), parentJournalSHA256=PARENT_SHA256,
                priorInterchangeJournalSHA256=INTERCHANGE_SHA256, checkpoints=checkpoints(),
                pairs=pairs, groups=groups, plannedForwards=len(groups) * 232,
                alignment='none; exact full vector at the last prompt token; both task directions')


def requests():
    # The first 136 calls exactly replicate experiment 15 on each new pair.
    calls = previous.requests()
    for recipient_task in TASKS:
        donor_task = TASKS[1 - TASKS.index(recipient_task)]
        for site, ds, rs, dc, rc in itertools.product(SITES, range(2), range(2), range(2), range(2)):
            calls.append(dict(id=len(calls), kind='transfer', task=recipient_task, donorTask=donor_task,
                              site=site, donorState=ds, recipientState=rs, donorCode=dc, recipientCode=rc))
    return calls


def donor_task(request):
    return request.get('donorTask', request['task'])


def semantic_predictions(pair, request):
    return predictions(donor_task(request), request['task'], request['donorState'], request['recipientState'],
                       int(pair['donor']['marker'] == 0), int(pair['recipient']['marker'] == 0),
                       request['donorCode'], request['recipientCode'])


def validate_group(pair, outputs, choices):
    require(len(outputs) == 232, 'Wrong cross-task group size')
    previous.validate_group(pair, outputs[:136], choices)
    intact = {previous.intact_key(o['request']): o for o in outputs[:136] if o['request']['kind'] == 'intact'}
    for task, role, code in itertools.product(TASKS, ('donor', 'recipient'), range(2)):
        a, b = (intact[(task, role, state, code)] for state in (0, 1))
        require(a['promptHash'] == b['promptHash'] and a['inputTokens'] == b['inputTokens'], 'Hidden state leaked into prompt')
    transfers = {}
    for expected, out in zip(requests()[136:], outputs[136:]):
        require(out['request'] == expected, 'Cross-task request order changed')
        scores = out['choiceLogits']
        require(len(scores) == 4 and all(np.isfinite(scores)), 'Invalid cross-task logits')
        require(0 < out['choiceMass'] <= 1.000001 and np.isfinite(out['seconds']) and out['seconds'] >= 0, 'Invalid output metrics')
        raw = out['rawTokenId']
        require(type(raw) is int and raw >= 0 and out['rawChoice'] == (choices.index(raw) if raw in choices else -1), 'Invalid raw choice')
        state, code, site = expected['recipientState'], expected['recipientCode'], expected['site']
        recipient = intact[(expected['task'], 'recipient', state, code)]
        donor = intact[(expected['donorTask'], 'donor', expected['donorState'], expected['donorCode'])]
        require(out['promptHash'] == recipient['promptHash'] and out['inputTokens'] == recipient['inputTokens'], 'Recipient prompt changed')
        trace = out['intervention']
        require(trace['applications'] == int(state != 0) and trace['changed'] == bool(state), 'Wrong physical intervention')
        require(np.isfinite(trace['normRelativeError']) and 0 <= trace['normRelativeError'] <= .01, 'Invalid rotation norm')
        patch = out['patch']
        require(patch['applications'] == 1 and patch['site'] == site and patch['tokenIndex'] == out['inputTokens'] - 1, 'Invalid patch')
        require(patch['recipientHash'] == recipient['activationHashes'][str(site)], 'Wrong recipient activation')
        require(patch['donorHash'] == patch['patchedHash'] == donor['activationHashes'][str(site)], 'Wrong donor task/state/code activation')
        require(all(np.isfinite(patch[k]) and patch[k] >= 0 for k in ('recipientNorm', 'donorNorm', 'displacementNorm')), 'Invalid patch norm')
        if site == 35:
            require(previous.same_output(out, donor), 'Cross-task last-layer copy failed')
        transfers[(expected['task'], site, expected['donorState'], state, expected['donorCode'], code)] = out
    for task, rs, dc, rc in itertools.product(TASKS, range(2), range(2), range(2)):
        require(previous.same_output(transfers[(task, 17, 0, rs, dc, rc)], transfers[(task, 17, 1, rs, dc, rc)]),
                'Cross-task layer-17 donor-state control failed')


def read_journal(path):
    events = [strict_json(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    require(events and events[0]['event'] == 'header', 'Missing header')
    header = events[0]
    fixed = plan()
    require(header['plan'] == fixed and header['planHash'] == digest(fixed) and header['sourceHash'] == source_hash(), 'Plan/source mismatch')
    require(header['metadata']['origin'] in ('transformers_gpu', 'synthetic_fixture'), 'Unknown origin')
    choices = header['metadata']['choiceTokenIds']
    require(len(choices) == len(set(choices)) == 4 and all(type(x) is int and x >= 0 for x in choices), 'Invalid digit IDs')
    pairs = {p['id']: p for p in fixed['pairs']}
    done, pending = [], None
    for event in events[1:]:
        kind = event['event']
        if kind == 'group_start':
            require(pending is None and len(done) < len(fixed['groups']) and event['group'] == fixed['groups'][len(done)], 'Wrong start')
            pending = event['group']
        elif kind == 'group_restart':
            require(pending is not None and event['id'] == pending['id'], 'Wrong restart')
            pending = None
        elif kind == 'group_complete':
            require(pending is not None and event['id'] == pending['id'], 'Wrong completion')
            validate_group(pairs[pending['pair']], event['outputs'], choices)
            done.append(dict(group=pending, outputs=event['outputs']))
            pending = None
        elif kind == 'error':
            require(pending is not None, 'Error without running group')
        else:
            raise ValueError('Unknown event')
    return header, done, pending, events


def analyze(path):
    header, done, pending, events = read_journal(path)
    fixed = header['plan']
    pairs = {p['id']: p for p in fixed['pairs']}
    choices = header['metadata']['choiceTokenIds']
    complete = pending is None and len(done) == len(fixed['groups'])
    tables, baseline, contrasts, prerequisites = {}, {}, [], {}
    if complete:
        for rep, split, arm in itertools.product(range(3), ('test', 'lexical'), ARMS):
            groups = [g for g in done if g['group']['arm'] == arm and pairs[g['group']['pair']]['replication'] == rep
                      and pairs[g['group']['pair']]['split'] == split]
            rng = np.random.default_rng(SEED + rep * 10 + ('test', 'lexical').index(split))
            n = len(groups)
            strata = [[i for i, g in enumerate(groups) if (pairs[g['group']['pair']]['donor']['marker'],
                       pairs[g['group']['pair']]['recipient']['marker']) == (dm, rm)] for dm, rm in itertools.product(range(2), repeat=2)]
            draws = np.concatenate([rng.choice(ids, (CONFIG['resamples'], len(ids))) for ids in strata], axis=1)
            for task, code in itertools.product(TASKS, range(2)):
                values = []
                for g in groups:
                    pair = pairs[g['group']['pair']]
                    part = []
                    for out in g['outputs']:
                        r = out['request']
                        if r['kind'] == 'intact' and r['task'] == task and r['code'] == code:
                            truth = r['state'] if task == 'monitor' else int(pair[r['role']]['marker'] == 0)
                            part.append(int(out['rawTokenId'] == choices[truth ^ code]))
                    values.append(np.mean(part))
                baseline[f'{rep}/{split}/{arm}/{task}/{code}'] = dict(accuracy=float(np.mean(values)), pairs=n,
                    interval95=np.quantile(np.asarray(values)[draws].mean(1), [.025, .975]).tolist())
            for dt, rt, site in itertools.product(TASKS, TASKS, SITES):
                metrics = []
                for g in groups:
                    pair = pairs[g['group']['pair']]
                    intact = {previous.intact_key(o['request']): o for o in g['outputs'] if o['request']['kind'] == 'intact'}
                    part = []
                    for out in g['outputs']:
                        r = out['request']
                        if r['kind'] != 'transfer' or r['task'] != rt or donor_task(r) != dt or r['site'] != site:
                            continue
                        predicted = semantic_predictions(pair, r)
                        donor = intact[(dt, 'donor', r['donorState'], r['donorCode'])]
                        recipient = intact[(rt, 'recipient', r['recipientState'], r['recipientCode'])]
                        scores = out['choiceLogits']
                        log_z = max(scores) + math.log(sum(math.exp(x - max(scores)) for x in scores))
                        raw = out['rawTokenId']
                        part.append(dict(**{k: int(raw == choices[v]) for k, v in predicted.items()},
                            **{k + 'CrossEntropy': log_z - scores[v] - math.log(out['choiceMass']) for k, v in predicted.items()},
                            observedDonorCopy=int(raw == donor['rawTokenId']), observedRecipientRetention=int(raw == recipient['rawTokenId']),
                            optionOutput=int(raw in choices[:2]), patchNorm=out['patch']['displacementNorm']))
                    require(len(part) == 16, 'Incomplete transfer factorial')
                    metrics.append({k: float(np.mean([p[k] for p in part])) for k in part[0]})
                key = f'{rep}/{split}/{arm}/{dt}-to-{rt}/{site}'
                tables[key] = {k: dict(mean=float(np.mean([m[k] for m in metrics])),
                    interval95=np.quantile(np.asarray([m[k] for m in metrics])[draws].mean(1), [.025, .975]).tolist()) for k in metrics[0]}
                tables[key].update(pairs=n, calls=n * 16)
                for rival in ('donorBoolean', 'donorAnswer', 'recipientUnchanged'):
                    delta = np.asarray([m['recipientTaskContent'] - m[rival] for m in metrics])
                    contrasts.append(dict(replication=rep, split=split, arm=arm, donorTask=dt, recipientTask=rt, site=site,
                        comparison='recipientTaskContent-' + rival, primary=split == 'test' and arm == 'prefix' and dt != rt and site == 23,
                        difference=float(delta.mean()), interval95=np.quantile(delta[draws].mean(1), [.025, .975]).tolist()))
        for rep in range(3):
            initial = {f'{t}/{c}': baseline[f'{rep}/test/prefix/{t}/{c}']['accuracy'] >= CONFIG['baselineGate'] for t in TASKS for c in (0, 1)}
            within = {t: tables[f'{rep}/test/prefix/{t}-to-{t}/23']['recipientTaskContent']['mean'] >= CONFIG['withinTaskGate'] for t in TASKS}
            prerequisites[str(rep)] = dict(intactChecks=initial, withinTaskChecks=within, allPassed=all(initial.values()) and all(within.values()),
                interpretation='Necessary competence and within-task transfer controls; never filters any trial')
    return dict(schema='menia-cross-task-interchange-report-v1', origin=header['metadata']['origin'],
        planHash=header['planHash'], sourceHash=header['sourceHash'], complete=complete,
        groups=len(done), plannedGroups=len(fixed['groups']), recorded=sum(len(g['outputs']) for g in done), planned=fixed['plannedForwards'],
        selfShams=sum(o['request']['kind'] == 'sham' for g in done for o in g['outputs']),
        finalLayerCopies=sum(o['request']['kind'] == 'transfer' and o['request']['site'] == 35 for g in done for o in g['outputs']),
        crossTaskTransfers=sum('donorTask' in o['request'] for g in done for o in g['outputs']),
        errors=sum(e['event'] == 'error' for e in events), restarts=sum(e['event'] == 'group_restart' for e in events),
        tables=tables, baseline=baseline, contrasts=contrasts, prerequisites=prerequisites,
        limitations='Post-15 cross-task pilot, exact whole vectors and no learned alignment. Both questions already present before capture. Canonical wording, no claim of necessity, consciousness or novelty.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.journal), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
