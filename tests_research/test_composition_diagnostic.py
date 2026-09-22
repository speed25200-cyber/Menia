import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research import composition_diagnostic as s
from tests_research.test_state_composition import trace, write, fixture as source_fixture


def calibration_fixture(frozen):
    return dict(schema='menia-composition-thresholds-v1', sourceJournalSHA256=s.COMPOSITION_SHA256,
                checkpoints=frozen, fits={f'{r}/{m}': dict(threshold=4.)
                for r in range(s.CONFIG['replications']) for m in range(2)})


def fixture(path):
    plan = s.plan(); blocks = {b['id']: b for b in plan['blocks']}
    events = [dict(event='header', plan=plan, planHash=s.digest(plan), sourceHash=s.source_hash(),
                   metadata=dict(origin='synthetic_fixture', choiceTokenIds=[10,11,12,13]))]
    for req in plan['evaluation']:
        b = blocks[req['block']]; pos = req['position']; form = req['format']; mapping = req['mapping']
        y = s.previous.bit(b, req['task'], pos)
        yes = s.previous.codes(form, mapping)[1]
        # Perfect ranking but a deliberately wrong zero boundary for hidden decisions.
        oriented = (6. if y else 2.) if req['task'] == 'monitor' else (5. if y else -5.)
        if req['task'] == 'marker_second' and form == 'paraphrase' and req['arm'] != 'base':
            oriented = -oriented
        logits = [0., 0., -20., -20.]; logits[yes] = oriented
        raw = max(range(4), key=lambda i: logits[i])
        events.append(dict(event='request', request=req,
            promptHash=s.digest(s.prompt(b, req['family'], req['task'], pos, form, mapping)[0]),
            adapterHash=plan['checkpoints'].get(f"r{req['replication']}-{req['arm']}"), inputTokens=110))
        events.append(dict(event='result', id=req['id'], choiceLogits=logits, rawChoice=raw,
            rawTokenId=10+raw, choiceMass=.99, seconds=.01, intervention=trace(req['family'], pos)))
    write(path, events); return events


class CompositionDiagnosticTests(unittest.TestCase):
    def test_design_is_balanced_disjoint_and_deterministic(self):
        p = s.plan(); self.assertEqual(p, s.plan())
        self.assertEqual(len(p['evaluation']), 13824)
        self.assertEqual(len({r['id'] for r in p['evaluation']}), 13824)
        old = s.old_blocks()
        self.assertEqual(len({x for b in p['blocks'] for x in b['sentences']}), 144)
        self.assertFalse({x for b in p['blocks'] for x in b['sentences']} & {x for b in old for x in b['sentences']})
        self.assertFalse({b['noiseSeed'] for b in p['blocks']} & {b['noiseSeed'] for b in old})
        self.assertEqual(len({b['noiseSeed'] for b in p['blocks']}), 72)
        for r in range(3):
            self.assertEqual(sum(b['replication'] == r and b['marker'] == 0 for b in p['blocks']), 12)

    def test_threshold_is_class_balanced_and_deterministic(self):
        result = s.fit_threshold([[2., 6., 6.], [2., 6., 6.]])
        self.assertEqual(result['threshold'], 4.); self.assertEqual(result['trainingBalancedAccuracy'], 1.)
        tied = s.fit_threshold([[1., 1., 1.]])
        self.assertEqual(tied['trainingBalancedAccuracy'], .5)
        self.assertEqual(tied['threshold'], 0.)
        with self.assertRaises(ValueError): s.fit_threshold([[np.nan, 0., 1.]])

    def test_fit_ignores_source_test_results_and_shams(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(s.previous.CONFIG, dict(trainBlocks=2, testBlocks=2, epochs=1, replications=1)), patch.dict(s.CONFIG, dict(replications=1)):
            p = Path(d)/'source.jsonl'; source_fixture(p)
            h, rows, trained, pending, events = s.previous.read_journal(p)
            frozen = dict(trained); frozen['r0-parent'] = h['parentCheckpoints']['r0-strong']
            with patch.object(s, 'checkpoints', return_value={k:v for k,v in frozen.items() if not k.endswith('-shuffled')}), patch.object(s, 'COMPOSITION_SHA256', hashlib.sha256(p.read_bytes()).hexdigest()):
                first = s.make_calibration(p)
                altered = copy.deepcopy(rows)
                for row in altered:
                    req = row['request']
                    if '-test-' in req['block'] or req['position'] == 3:
                        row['result']['choiceLogits'] = [999., -999., -999., -999.]
                with patch.object(s.previous, 'read_journal', return_value=(h, altered, trained, pending, events)):
                    self.assertEqual(first, s.make_calibration(p))
                for fit in first['fits'].values():
                    self.assertEqual(fit['rows'], 6)
                    self.assertTrue(all('-train-' in key and not key.endswith('/3') for key in fit['sourceRequestIds']))

    def test_reports_distinguish_readout_gain_from_language_damage(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(s.CONFIG, dict(blocksPerReplication=4, replications=1, resamples=30)), patch.object(s, 'calibration', side_effect=lambda: calibration_fixture(s.checkpoints())):
            p = Path(d)/'f.jsonl'; fixture(p); report = s.analyze(p)
            self.assertTrue(report['complete']); self.assertEqual(report['recorded'], 768)
            self.assertEqual(report['shamPairs'], 192)
            self.assertEqual(len(report['tables']), 48); self.assertEqual(len(report['contrasts']), 36)
            for mapping in (0, 1):
                t = report['tables'][f'0/composed/hidden/trained/monitor/{mapping}']
                self.assertEqual(t['presenceAUROC'], 1.); self.assertEqual(t['nativeBalancedAccuracy'], .5)
                self.assertEqual(t['calibratedBalancedAccuracy'], 1.)
            gains = [c for c in report['contrasts'] if c['kind'] == 'externalThresholdGain']
            self.assertTrue(all(c['difference'] == .5 and c['interval95'] == [.5, .5] for c in gains))
            interactions = [c for c in report['contrasts'] if c['kind'] == 'publicWordingInteractionVsBase' and c['task'] == 'marker_second']
            self.assertTrue(all(c['difference'] == -1. for c in interactions))

    def test_rejects_tampered_weights_shams_and_training(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(s.CONFIG, dict(blocksPerReplication=2, replications=1, resamples=10)), patch.object(s, 'calibration', side_effect=lambda: calibration_fixture(s.checkpoints())):
            p = Path(d)/'f.jsonl'; events = fixture(p)
            altered = copy.deepcopy(events); altered[1]['adapterHash'] = 'wrong'; write(p, altered)
            with self.assertRaisesRegex(ValueError, 'Adapter'): s.read_journal(p)
            altered = copy.deepcopy(events)
            next(e for e in altered if e['event'] == 'result' and e['id'].endswith('/3'))['choiceMass'] = .8
            write(p, altered)
            with self.assertRaisesRegex(ValueError, 'Sham'): s.analyze(p)
            write(p, events+[dict(event='training_start')])
            with self.assertRaisesRegex(ValueError, 'never trains'): s.read_journal(p)


if __name__ == '__main__': unittest.main()
