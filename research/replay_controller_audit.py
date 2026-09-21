"""Portable, read-only audit of completed replay exports; no new model calls.

The collection code stays pinned. Its strict fit-event hash comparison also
hashes a newly solved floating-point fit, which can differ across BLAS builds.
Here the stored hash is checked against the stored bundle, and the refit is
compared numerically. No checksum or numerical refit check is omitted.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from research import replay_controller as pilot
from research.activation_monitor import features
from research.cross_model_prediction import digest
from research.iphone_coupling_report import require, strict_json
from research.perturbation_monitor import close_tree


def compare(a, b, location='root'):
    """Strict structure/counts/booleans, tolerant floating-point values."""
    if isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), f'Fields differ: {location}')
        return max((compare(v, b[k], f'{location}.{k}') for k, v in a.items()), default=0.)
    if isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), f'Length differs: {location}')
        return max((compare(x, y, f'{location}[{i}]') for i, (x, y) in enumerate(zip(a,b))), default=0.)
    if type(a) is float:
        require(type(b) in (int, float) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10),
                f'Numerical value differs: {location}')
        return abs(a-b)
    require(type(a) is type(b) and a == b, f'Value differs: {location}')
    return 0.


def portable_read(path):
    # This branch still verifies stored bundle/calibration hashes, predictions,
    # decisions, update tables, source/plan hashes and every event's ordering.
    data = pilot.read_journal(path, check_fit=False)
    _, rows, _, bundle, _, _, _ = data
    require(bundle is not None, "No fitted monitor to audit")
    refit = pilot.fit_monitors(rows)
    close_tree(refit, bundle)
    return data, refit


def reference(task):
    """Independent reference from the public question, not stored operands."""
    question = task['question']
    if task['family'] == 'countA':
        letters = question.split(': ', 1)[1].removesuffix(' ?')
        return sum(1 for letter in letters if letter == 'A')
    tokens = re.findall(r'[+-]?\d+', question.replace(' ', ''))
    return sum(map(int, tokens))


def correct(row):
    # Reuse the fixed parser deliberately; independently reconstruct the target.
    return pilot.integer_answer(row['result']['text']) == reference(row['task'])


def choose(policy, probabilities):
    if policy['kind'] != 'threshold':
        return policy['kind']
    return 'direct' if probabilities[policy['source']] >= policy['threshold'] else policy['below']


def units(row, action):
    # Exact twentieths avoid accumulating floating-point .2 / .35 losses.
    return {'direct': 0 if correct(row) else 20, 'verify': 4, 'abstain': 7}[action]


def route(rows, arm):
    actions = [r['actions'][arm] for r in rows]
    return dict(n=len(rows), meanPointLoss=sum(units(r, a) for r, a in zip(rows, actions))/(20*len(rows)),
                finalCorrect=sum(a == 'verify' or (a == 'direct' and correct(r)) for r, a in zip(rows, actions)),
                actionCounts={a: actions.count(a) for a in ('direct', 'verify', 'abstain')})


def audit(path):
    path = Path(path)
    data, refit = portable_read(path)
    header, rows, pending, bundle, policies, updates, failures = data
    require(len(rows) == len(header['plan']['tasks']) and pending is None, "Incomplete experiment")
    supplied = strict_json(path.with_suffix('.summary.json').read_text(encoding='utf-8'))
    exported = strict_json(path.with_suffix('.policy.json').read_text(encoding='utf-8'))
    events = [strict_json(line) for line in path.read_text(encoding='utf-8').splitlines()]
    # The tool is assumed exact by units(); establish that for every actual call.
    require(all(r['result']['verification']['text'] == str(reference(r['task'])) for r in rows),
            "Independent tool reference mismatch")
    phases = {}
    for phase in pilot.COUNTS:
        group = [r for r in rows if r['task']['phase'] == phase]
        phases[phase] = dict(n=len(group), numericCorrect=sum(map(correct, group)),
            strictCorrect=sum(r['result']['text'].strip() == str(reference(r['task'])) for r in group),
            unparseable=sum(pilot.integer_answer(r['result']['text']) is None for r in group),
            routes={a: route(group, a) for a in pilot.ARMS} if phase != 'calibration' else None)
    test = [r for r in rows if r['task']['phase'] == 'test']
    blocks = sorted({r['task']['block'] for r in test})
    draws = np.random.default_rng(pilot.SEED+90).integers(0, len(blocks), size=(2000, len(blocks)))
    contrasts = {}
    for arm in ('public', 'shuffled', 'fixedBeta', 'fixedInternal', 'alwaysVerify'):
        deltas = []
        for block in blocks:
            group = [r for r in test if r['task']['block'] == block]
            deltas.append(sum(units(r, r['actions']['learned'])-units(r, r['actions'][arm]) for r in group)/(20*len(group)))
        values = np.array(deltas)
        contrasts[arm] = dict(learnedMinusOther=float(values.mean()),
            descriptiveBlockBootstrap95=np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist())
    report = dict(schema='menia-replay-report-v1', origin=header['origin'], planHash=header['planHash'],
        complete=True, recordedResults=len(rows), plannedResults=len(header['plan']['tasks']),
        pendingRequest=False, failures=failures, updates=updates,
        frozenPolicies={k: p.payload() for k, p in policies.items()}, byPhase=phases, contrasts=contrasts,
        actualCosts=dict(completedGenerationCalls=len(rows), failedOrInterruptedAttempts=len(failures),
            successfulGenerationSeconds=math.fsum(r['result']['seconds'] for r in rows),
            selectedVerificationCalls=sum(r['result']['verification']['role'] == 'selected' for r in rows),
            additionalAuditVerificationCalls=sum(r['result']['verification']['role'] == 'coverageAudit' for r in rows),
            totalVerificationSeconds=math.fsum(r['result']['verification']['seconds'] for r in rows)),
        interpretation='Primary learned routes executed; alternative routes replayed on identical candidates. Fixed point costs, not currency or latency. Fixed monitors; three policy updates. No consciousness inference.')
    summary_error = compare(report, supplied)
    compare(exported, dict(schema='menia-replay-controller-v1', model=header['plan']['model'], costs=pilot.COSTS,
        sourceHash=header['sourceHash'], planHash=header['planHash'], featureSchema='activation-monitor-projections-v1',
        monitor=bundle, policies=report['frozenPolicies'],
        scope='Experimental routing for this task distribution; not an iPhone model adapter'))

    # Independent SVD formulation of ridge; feature extraction remains shared.
    calibration = [r for r in rows if r['task']['phase'] == 'calibration']
    target = np.array(list(map(correct, calibration)), dtype=float)
    shuffled = target.copy()
    rng = np.random.default_rng(pilot.SEED+1)
    for family, level in pilot.CELLS:
        ids = [i for i, r in enumerate(calibration) if (r['task']['family'], r['task']['level']) == (family, level)]
        shuffled[ids] = rng.permutation(target[ids])
    svd_errors = {}
    for name, fitted in bundle['models'].items():
        x = np.array([features(r['task'], r['state'], internal=name != 'inputOnly') for r in calibration])
        scale = x.std(0); scale[scale < 1e-8] = 1.
        z = (x-x.mean(0))/scale
        y = shuffled if name == 'shuffledLabels' else target
        u, s, vt = np.linalg.svd(z, full_matrices=False)
        weights = vt.T @ ((s/(s*s+len(z))) * (u.T @ (y-y.mean())))
        error = float(np.max(np.abs(weights-np.array(fitted['weights']))))
        require(error < 1e-8, "Independent SVD fit mismatch")
        svd_errors[name] = error

    # Recompute all candidate scores from integer losses; verify first minimum.
    trajectory = []
    candidate_scores = 0
    for event in (e for e in events if e['event'] == 'update'):
        round_number = int(event['phase'][-1])
        history = [r for r in rows if r['task']['phase'] in [f'round{i}' for i in range(1, round_number+1)]]
        for arm, table in event['tables'].items():
            losses = []
            for entry in table['scores']:
                actions = [choose(entry['policy'], r['predictions']) for r in history]
                total = sum(units(r, a) for r, a in zip(history, actions))
                losses.append(total)
                compare(entry['meanPointLoss'], total/(20*len(history)))
                require(entry['actionCounts'] == {a: actions.count(a) for a in ('direct','verify','abstain')},
                        "Candidate action counts mismatch")
                candidate_scores += 1
            require(table['scores'][losses.index(min(losses))]['policy'] == event['policies'][arm],
                    "Independent selected policy mismatch")
        trajectory.append(dict(phase=event['phase'], policies=event['policies'],
            losses={a:{k:v for k,v in t.items() if k != 'scores'} for a,t in event['tables'].items()}))
    by_cell = {}
    for family, level in pilot.CELLS:
        group = [r for r in test if (r['task']['family'], r['task']['level']) == (family, level)]
        key = f'{family}:{level}'
        by_cell[key] = dict(n=len(group), rawCorrect=sum(map(correct, group)),
            calibrationProbability=bundle['beta'][key], learned=route(group, 'learned'))
    return dict(schema='menia-replay-audit-v1', journalSHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
        codeRevision=header['metadata'].get('codeCommit'), sourceHash=header['sourceHash'], planHash=header['planHash'],
        validatedSummary=supplied, verification=dict(
            storedFitHashVerified=True, calibrationHashVerified=True, sourceAndPlanVerified=True,
            numericRefitVerified=True, refitHashEqualsStored=digest(refit) == digest(bundle),
            maxRefitWeightDifference=max(float(np.max(np.abs(np.array(refit['models'][name]['weights'])-model['weights'])))
                for name, model in bundle['models'].items()), svdMaxWeightDifferences=svd_errors,
            independentCandidateScores=candidate_scores, independentlyRegradedResults=len(rows),
            testRoutesChecked=len(test)*len(pilot.ARMS), summaryVerified=True, exportedPolicyVerified=True,
            maxSummaryNumericDifference=summary_error,
            inputTokenRange=[min(r['result']['metrics'].get('inputTokens',0) for r in rows),
                             max(r['result']['metrics'].get('inputTokens',0) for r in rows)],
            maxOutputTokens=max(r['result']['metrics'].get('outputTokens',0) for r in rows)),
        diagnosticAfterReceipt=dict(policyTrajectory=trajectory, testByCell=by_cell,
            learnedActionsIdenticalToPublic=all(r['actions']['learned']==r['actions']['public'] for r in rows if r['actions']),
            testLearnedActionsIdenticalToFixedBeta=all(r['actions']['learned']==r['actions']['fixedBeta'] for r in test)),
        scope='Read-only consistency and numerical audit. Shared fixed parser/features; no independent hardware attestation, new Qwen inference, or consciousness test.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    protected = [args.journal, args.journal.with_suffix('.summary.json'), args.journal.with_suffix('.policy.json')]
    require(args.output.resolve() not in [p.resolve() for p in protected], 'Cannot overwrite a received export')
    result = audit(args.journal)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result['verification'], ensure_ascii=False))
