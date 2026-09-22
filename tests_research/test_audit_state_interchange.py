import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import audit_state_interchange as audit
from research import state_interchange as study
from tests_research.test_state_interchange import fixture_group


def make_fixture(path, mixed=False):
    design = study.plan()
    pairs = {p['id']: p for p in design['pairs']}
    digits = [3, 4, 5, 6]
    events = [dict(event='header', plan=design, planHash=study.digest(design),
                   sourceHash=study.source_hash(), metadata=dict(origin='synthetic_fixture', choiceTokenIds=digits))]
    for i, group in enumerate(design['groups']):
        outputs = fixture_group(pairs[group['pair']], digits)
        if mixed:
            originals = {}
            for out in outputs:
                r = out['request']
                # Deliberately wrong intact answers, ties, large logits and unknown
                # vocabulary tokens. Copies still obey physical control invariants.
                if r['kind'] == 'intact':
                    index = (i + r['id']) % 5
                    scores = [-1000. + ((i + r['id'] + j) % 3) for j in range(4)]
                    if index < 4:
                        scores[index] = max(scores) + 2
                    out.update(rawTokenId=digits[index] if index < 4 else 99,
                               rawChoice=index if index < 4 else -1,
                               choiceLogits=scores,
                               choiceMass=.01 * (1 + i % 10))
                    originals[(r['task'], r['role'], r['state'], r['code'])] = out
                elif r['kind'] == 'sham' or r['site'] in (17, 35):
                    if r['kind'] == 'transfer' and r['site'] == 35:
                        key = (r['task'], 'donor', r['donorState'], r['donorCode'])
                    else:
                        key = (r['task'], 'recipient', r['recipientState'], r['recipientCode'])
                    for k in ('rawTokenId', 'rawChoice', 'choiceLogits', 'choiceMass'):
                        out[k] = copy.deepcopy(originals[key][k])
                else:
                    index = (i + r['id']) % 5
                    scores = [0.] * 4
                    if index < 4:
                        scores[index] = math.log(2)
                    out.update(rawTokenId=digits[index] if index < 4 else 99,
                               rawChoice=index if index < 4 else -1,
                               choiceLogits=scores, choiceMass=.6)
                    out['patch']['displacementNorm'] = .1 * (i + r['id'])
        events.extend([dict(event='group_start', group=group), dict(event='group_complete', id=group['id'], outputs=outputs)])
    path.write_text('\n'.join(json.dumps(e) for e in events) + '\n', encoding='utf-8')
    return events


class IndependentInterchangeAuditTests(unittest.TestCase):
    def test_known_mechanisms_and_public_state_coding(self):
        with patch.dict(study.CONFIG, resamples=20), tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.jsonl'
            make_fixture(path)
            report = study.analyze(path)
            computed = audit.recompute(path)
            checked = audit.verify(path, report)
            self.assertEqual(checked['tables'], 72)
            self.assertEqual(checked['baselineTables'], 48)
            self.assertEqual(checked['contrasts'], 144)
            self.assertEqual(checked['primaryContrasts'], 6)
            self.assertEqual(checked['selfShams'], 3456)
            self.assertEqual(checked['finalLayerCopies'], 4608)
            for task in ('monitor', 'marker_first'):
                for site, mechanism in ((17, 'recipientUnchanged'), (23, 'stateTransfer'), (35, 'donorAnswer')):
                    table = computed['tables'][f'0/test/prefix/{task}/{site}']
                    self.assertEqual(table[mechanism]['mean'], 1.)
                    p_correct = .8 * math.exp(2) / (math.exp(2) + 3)
                    self.assertAlmostEqual(table[mechanism + 'CrossEntropy']['mean'], -math.log(p_correct))
            for c in computed['contrasts']:
                if c['primary']:
                    self.assertEqual(c['difference'], .5)
                    self.assertEqual(c['interval95'], [.5, .5])
            self.assertTrue(all(p['allPassed'] for p in computed['prerequisites'].values()))

    def test_wrong_answers_outside_options_bootstrap_and_tampered_report(self):
        with patch.dict(study.CONFIG, resamples=30), tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mixed.jsonl'
            make_fixture(path, mixed=True)
            report = study.analyze(path)
            checked = audit.verify(path, report)
            self.assertLess(checked['maxAbsoluteDifference'], 1e-10)
            self.assertTrue(all(not x['allPassed'] for x in report['prerequisites'].values()))
            table = report['tables']['0/test/prefix/monitor/23']
            self.assertLess(table['optionOutput']['mean'], .5)
            self.assertNotEqual(table['stateTransfer']['interval95'][0], table['stateTransfer']['interval95'][1])
            self.assertNotEqual(table['donorAnswer']['mean'], table['observedDonorCopy']['mean'])
            altered = copy.deepcopy(report)
            altered['contrasts'][2]['difference'] += .01
            with self.assertRaisesRegex(ValueError, 'calculation mismatch'):
                audit.verify(path, altered)
            altered = copy.deepcopy(report)
            altered['prerequisites']['0']['allPassed'] = True
            with self.assertRaises(ValueError):
                audit.verify(path, altered)
            altered = copy.deepcopy(report)
            altered['complete'] = False
            with self.assertRaisesRegex(ValueError, 'Incomplete summary'):
                audit.verify(path, altered)


if __name__ == '__main__':
    unittest.main()
