import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import audit_cross_task_interchange as audit
from research import cross_task_interchange as study
from tests_research.test_cross_task_interchange import fixture_group


def fixture(path, mixed=False):
    plan = study.plan()
    pairs = {p['id']: p for p in plan['pairs']}
    digits = [3, 4, 5, 6]
    events = [dict(event='header', plan=plan, planHash=study.digest(plan), sourceHash=study.source_hash(),
                   metadata=dict(origin='synthetic_fixture', choiceTokenIds=digits))]
    modes = ('recipientTaskContent', 'donorBoolean', 'recipientUnchanged')
    for i, group in enumerate(plan['groups']):
        pair = pairs[group['pair']]
        mechanism = modes[pair['replication']] if group['arm'] == 'prefix' else 'donorAnswer'
        outputs = fixture_group(pair, digits, mechanism)
        if mixed:
            originals = {}
            for out in outputs:
                r = out['request']
                if r['kind'] == 'intact' or (r['kind'] == 'transfer' and r['site'] == 23):
                    index = (i + r['id']) % 5
                    scores = [-1000. + (i + j) % 3 for j in range(4)]
                    if index < 4:
                        scores[index] = max(scores) + 2
                    out.update(choiceLogits=scores, choiceMass=.02 * (1 + i % 10),
                               rawTokenId=digits[index] if index < 4 else 99, rawChoice=index if index < 4 else -1)
                    if r['kind'] == 'intact':
                        originals[(r['task'], r['role'], r['state'], r['code'])] = out
                    else:
                        out['patch']['displacementNorm'] = (i + r['id']) / 100
                else:
                    if r['kind'] == 'transfer' and r['site'] == 35:
                        original = originals[(r.get('donorTask', r['task']), 'donor', r['donorState'], r['donorCode'])]
                    else:
                        original = originals[(r['task'], 'recipient', r['recipientState'], r['recipientCode'])]
                    for key in ('rawTokenId', 'rawChoice', 'choiceLogits', 'choiceMass'):
                        out[key] = copy.deepcopy(original[key])
        events.extend([dict(event='group_start', group=group), dict(event='group_complete', id=group['id'], outputs=outputs)])
    path.write_text('\n'.join(json.dumps(e) for e in events) + '\n', encoding='utf-8')


class IndependentCrossTaskAuditTests(unittest.TestCase):
    def test_four_known_mechanisms_both_directions_and_full_vocabulary_loss(self):
        with patch.dict(study.CONFIG, resamples=20), tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'known.jsonl'
            fixture(path)
            report = study.analyze(path)
            checked = audit.verify(path, report)
            expected = dict(tables=144, baselineTables=48, contrasts=432, primaryContrasts=18,
                            selfShams=4608, finalLayerCopies=12288, crossTaskTransfers=18432)
            for key, value in expected.items():
                self.assertEqual(checked[key], value)
            perfect_loss = -math.log(.8 * math.exp(2) / (math.exp(2) + 3))
            for dt, rt in (('monitor', 'marker_first'), ('marker_first', 'monitor')):
                for rep, mechanism in enumerate(('recipientTaskContent', 'donorBoolean', 'recipientUnchanged')):
                    table = report['tables'][f'{rep}/test/prefix/{dt}-to-{rt}/23']
                    self.assertEqual(table[mechanism]['mean'], 1.)
                    self.assertAlmostEqual(table[mechanism + 'CrossEntropy']['mean'], perfect_loss)
                    if mechanism != 'recipientTaskContent':
                        self.assertEqual(table['recipientTaskContent']['mean'], .5)
                self.assertEqual(report['tables'][f'0/test/base/{dt}-to-{rt}/23']['donorAnswer']['mean'], 1.)
            self.assertTrue(all(p['allPassed'] for p in report['prerequisites'].values()))

    def test_failed_prerequisites_unknown_tokens_intervals_and_tampering(self):
        with patch.dict(study.CONFIG, resamples=30), tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mixed.jsonl'
            fixture(path, mixed=True)
            report = study.analyze(path)
            checked = audit.verify(path, report)
            self.assertLess(checked['maxAbsoluteDifference'], 1e-10)
            self.assertEqual(report['recorded'], 44544)
            self.assertTrue(all(not v['allPassed'] for v in report['prerequisites'].values()))
            self.assertTrue(all(not passed for v in report['prerequisites'].values() for passed in v['withinTaskChecks'].values()))
            t = report['tables']['0/test/prefix/monitor-to-marker_first/23']
            self.assertLess(t['optionOutput']['mean'], .5)
            self.assertNotEqual(t['recipientTaskContent']['interval95'][0], t['recipientTaskContent']['interval95'][1])
            self.assertNotEqual(t['donorAnswer']['mean'], t['observedDonorCopy']['mean'])
            altered = copy.deepcopy(report)
            altered['contrasts'][0]['difference'] += .01
            with self.assertRaisesRegex(ValueError, 'calculation mismatch'):
                audit.verify(path, altered)
            altered = copy.deepcopy(report)
            altered['prerequisites']['0']['withinTaskChecks']['monitor'] = True
            with self.assertRaises(ValueError):
                audit.verify(path, altered)
            altered = copy.deepcopy(report)
            altered['complete'] = False
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                audit.verify(path, altered)


if __name__ == '__main__':
    unittest.main()
