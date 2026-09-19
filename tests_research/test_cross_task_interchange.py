import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import cross_task_interchange as study
from tests_research.test_state_interchange import fixture_group as within_group


def fixture_group(pair, digits, mechanism):
    outputs = within_group(pair, digits)
    intact = {study.previous.intact_key(o['request']): o for o in outputs if o['request']['kind'] == 'intact'}
    for r in study.requests()[136:]:
        recipient = intact[(r['task'], 'recipient', r['recipientState'], r['recipientCode'])]
        donor = intact[(r['donorTask'], 'donor', r['donorState'], r['donorCode'])]
        site = r['site']
        digit = recipient['rawChoice'] if site == 17 else donor['rawChoice'] if site == 35 else study.semantic_predictions(pair, r)[mechanism]
        z = [0.] * 4
        z[digit] = 2.
        rh, dh = recipient['activationHashes'][str(site)], donor['activationHashes'][str(site)]
        outputs.append(dict(request=r, choiceLogits=z, choiceMass=.8, rawChoice=digit, rawTokenId=digits[digit], seconds=.001,
                            inputTokens=recipient['inputTokens'], promptHash=recipient['promptHash'],
                            intervention=copy.deepcopy(recipient['intervention']),
                            patch=dict(site=site, tokenIndex=recipient['inputTokens'] - 1, applications=1,
                                       recipientHash=rh, donorHash=dh, patchedHash=dh,
                                       recipientNorm=1., donorNorm=1., displacementNorm=float(rh != dh))))
    return outputs


class CrossTaskInterchangeTests(unittest.TestCase):
    def test_balanced_fresh_design_and_all_known_mechanisms(self):
        with patch.dict(study.CONFIG, resamples=20), tempfile.TemporaryDirectory() as directory:
            plan = study.plan()
            pairs = {p['id']: p for p in plan['pairs']}
            digits = [3, 4, 5, 6]
            self.assertEqual(plan['plannedForwards'], 44544)
            for rep, split in itertools.product(range(3), ('test', 'lexical')):
                selected = [p for p in plan['pairs'] if p['replication'] == rep and p['split'] == split]
                combinations = [(p['donor']['marker'], p['recipient']['marker'], p['donor']['positivePosition'],
                                 p['recipient']['positivePosition']) for p in selected]
                self.assertEqual(len(combinations), 16)
                self.assertEqual(set(combinations), set(itertools.product((0, 1), (0, 1), (1, 2), (1, 2))))
            sentences = [s for p in plan['pairs'] for role in ('donor', 'recipient') for s in p[role]['sentences']]
            old = {s for p in study.previous.plan()['pairs'] for role in ('donor', 'recipient') for s in p[role]['sentences']}
            self.assertEqual(len(sentences), len(set(sentences)))
            self.assertFalse(old & set(sentences))
            events = [dict(event='header', plan=plan, planHash=study.digest(plan), sourceHash=study.source_hash(),
                           metadata=dict(origin='synthetic_fixture', choiceTokenIds=digits))]
            modes = ('recipientTaskContent', 'donorBoolean', 'recipientUnchanged')
            for group in plan['groups']:
                pair = pairs[group['pair']]
                mode = modes[pair['replication']] if group['arm'] == 'prefix' else 'donorAnswer'
                outputs = fixture_group(pair, digits, mode)
                events.extend([dict(event='group_start', group=group), dict(event='group_complete', id=group['id'], outputs=outputs)])
            path = Path(directory) / 'synthetic.jsonl'
            path.write_text('\n'.join(json.dumps(e) for e in events) + '\n', encoding='utf-8')
            result = study.analyze(path)
            self.assertTrue(result['complete'])
            for k, v in dict(recorded=44544, selfShams=4608, finalLayerCopies=12288, crossTaskTransfers=18432).items():
                self.assertEqual(result[k], v)
            self.assertEqual(len(result['tables']), 144)
            self.assertEqual(len(result['baseline']), 48)
            self.assertEqual(len(result['contrasts']), 432)
            primary = [c for c in result['contrasts'] if c['primary']]
            self.assertEqual(len(primary), 18)
            for c in primary:
                rival = c['comparison'].split('-')[1]
                expected = .5 if c['replication'] == 0 else -.5 if rival == modes[c['replication']] else 0.
                self.assertEqual(c['difference'], expected)
                self.assertEqual(c['interval95'], [expected, expected])
            self.assertTrue(all(p['allPassed'] for p in result['prerequisites'].values()))
            for dt, rt in (('monitor', 'marker_first'), ('marker_first', 'monitor')):
                self.assertEqual(result['tables'][f'0/test/base/{dt}-to-{rt}/23']['donorAnswer']['mean'], 1.)

    def test_wrong_donor_task_and_cross_copy_and_partial_group_rejected(self):
        plan = study.plan()
        pair = plan['pairs'][0]
        digits = [3, 4, 5, 6]
        outputs = fixture_group(pair, digits, 'recipientTaskContent')
        study.validate_group(pair, outputs, digits)
        bad = copy.deepcopy(outputs)
        bad[136]['patch']['donorHash'] = outputs[0]['activationHashes']['17']
        with self.assertRaisesRegex(ValueError, 'Wrong donor task'):
            study.validate_group(pair, bad, digits)
        bad = copy.deepcopy(outputs)
        bad[168]['choiceMass'] = .7
        with self.assertRaisesRegex(ValueError, 'Cross-task last-layer copy failed'):
            study.validate_group(pair, bad, digits)
        with self.assertRaisesRegex(ValueError, 'Wrong cross-task group size'):
            study.validate_group(pair, outputs[:-1], digits)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'partial.jsonl'
            events = [dict(event='header', plan=plan, planHash=study.digest(plan), sourceHash=study.source_hash(),
                           metadata=dict(origin='synthetic_fixture', choiceTokenIds=digits)),
                      dict(event='group_start', group=plan['groups'][0])]
            path.write_text('\n'.join(json.dumps(e) for e in events) + '\n', encoding='utf-8')
            self.assertFalse(study.analyze(path)['complete'])


if __name__ == '__main__':
    unittest.main()
