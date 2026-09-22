"""Synthetic positive controls, temporal corruption and fail-closed collection."""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import natural_error_journal as journal
from research import natural_error_analysis as analysis
from research import natural_error_questions as questions
from research.cross_model_prediction import reference, digest, canonical


def synthetic_output(task, capture):
    success = task['block'] % 2 == 0
    state = {name: [0.]*128 for name in ('input', 'middle', 'final')}
    state['middle'] = [float(success)]*128
    before = dict(vocabularySize=4, topTokenId=0, maxProbability=.25,
                  topTwoMargin=0., entropyNats=math.log(4), normalizedEntropy=1.)
    value = dict(state=state, preAnswer=before)
    capture(value)
    text = str(reference(task)+(0 if success else 1))
    trace = dict(schema='menia-output-confidence-trace-v1', preAnswer=before,
        logitSource='lm_head output, before generation processors and sampling',
        distribution='raw full vocabulary; temperature 1; no top-k/top-p filtering',
        completion=dict(tokenIds=[0], tokenLogProbabilities=[-math.log(4)], entropyNats=[math.log(4)],
                        includesStopTokens=True, sumLogProbability=-math.log(4), meanLogProbability=-math.log(4)))
    return text, dict(inputTokens=8, outputTokens=1, reachedTokenLimit=False), trace


class FixtureBackend:
    origin = 'synthetic_fixture'
    metadata = {'fixture': 'middle state predicts its future synthetic answer'}
    def __init__(self, path):
        self.path = path
        self.before_return = []
    def generate(self, task, capture):
        def commit(value):
            capture(value)
            last = json.loads(self.path.read_text(encoding='utf-8').splitlines()[-1])['payload']
            assert last['event'] == 'capture' and last['id'] == task['id']
            assert last['predictions'] is not None if task['split'] == 'test' else last['predictions'] is None
            self.before_return.append(task['id'])
        return synthetic_output(task, commit)


def rewrite(path, events):
    previous = '0'*64
    lines = []
    for i, payload in enumerate(events):
        e = dict(sequence=i, previous=previous, payload=payload)
        previous = digest(e)
        lines.append(canonical(dict(e, sha256=previous)))
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


class NaturalErrorJournalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.context = contextlib.ExitStack()
        cls.context.enter_context(patch.dict(questions.COUNTS, train=4, validation=2, test=4))
        cls.context.enter_context(patch.object(journal, 'RESAMPLES', 2000))
        cls.context.enter_context(patch.object(journal, 'MIN_CLASS', 2))
        cls.path = cls.root/'fixture.jsonl'
        cls.backend = FixtureBackend(cls.path)
        with contextlib.redirect_stdout(io.StringIO()):
            journal.collect(cls.path, cls.backend, limit=5)
            journal.collect(cls.path, cls.backend, resume=True)
        cls.events = [json.loads(s)['payload'] for s in cls.path.read_text(encoding='utf-8').splitlines()]

    @classmethod
    def tearDownClass(cls):
        cls.context.close()
        cls.temp.cleanup()

    def test_complete_three_disjoint_fits_and_forecasts_are_committed(self):
        parsed = journal.read_journal(self.path)
        self.assertEqual(len(parsed['rows']), 180)
        self.assertEqual(len(parsed['bundles']), 3)
        self.assertEqual(self.backend.before_return, list(range(180)))
        report = analysis.analyze(self.path)
        self.assertTrue(report['complete'] and report['fitChecked'])
        self.assertEqual(report['origin'], 'synthetic_fixture')
        self.assertEqual(sum(c['primary'] for c in report['contrasts']), 9)
        for rep in report['replications'].values():
            self.assertLess(rep['scores']['internal']['brier'], .001)
            self.assertGreater(rep['scores']['finalControl']['brier'], .1)
        self.assertTrue(report['replicatedIncrementalReadoutCriterion'])
        self.assertAlmostEqual(report['familyIntervalCoverage'], 1-.05/9)
        unchecked=analysis.analyze(self.path,check_fit=False)
        self.assertFalse(unchecked['fitChecked'] or unchecked['replicatedIncrementalReadoutCriterion'])

    def test_rehashed_temporal_and_forecast_corruptions_are_rejected(self):
        capture_index = next(i for i,e in enumerate(self.events) if e['event']=='capture' and e['predictions'] is not None)
        for mutation, pattern in [('result_first','pre-token'), ('forecast','forecast'), ('trace','trace'), ('fit','Refit')]:
            with self.subTest(mutation=mutation):
                events = copy.deepcopy(self.events)
                if mutation == 'result_first':
                    events[capture_index],events[capture_index+1]=events[capture_index+1],events[capture_index]
                elif mutation == 'forecast':
                    events[capture_index]['predictions']['internal'] = 1-events[capture_index]['predictions']['internal']
                elif mutation == 'trace':
                    events[capture_index+1]['trace']['preAnswer']['topTokenId'] = 2
                else:
                    fit = next(e for e in events if e['event']=='fit')
                    fit['bundle']['models']['internal']['weights'][0] += .5
                    fit['bundleHash'] = digest(fit['bundle'])
                target = self.root/(mutation+'.jsonl')
                rewrite(target,events)
                with self.assertRaisesRegex(ValueError, pattern):
                    journal.read_journal(target)

    def test_chain_and_pending_error_prevent_silent_replacement(self):
        target = self.root/'chain.jsonl'
        lines=self.path.read_text(encoding='utf-8').splitlines()
        e=json.loads(lines[2]);e['payload']['capture']['state']['input'][0] += 1
        lines[2]=canonical(e)
        target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'chain'):
            journal.read_journal(target)
        pending=self.root/'pending.jsonl'
        rewrite(pending,self.events[:3])
        self.assertIsNotNone(journal.read_journal(pending)['pending'])
        with self.assertRaisesRegex(ValueError, 'retained'):
            journal.collect(pending,FixtureBackend(pending),resume=True)
        failed=self.root/'failed.jsonl'
        backend=FixtureBackend(failed)
        def fail(task,capture):
            synthetic_output(task,capture)
            raise RuntimeError('Simulated generation failure')
        backend.generate=fail
        with self.assertRaisesRegex(RuntimeError,'Simulated'):
            journal.collect(failed,backend)
        read=journal.read_journal(failed)
        self.assertTrue(read['failed'])
        self.assertEqual(read['rows'][0]['result']['status'],'error')
        with self.assertRaisesRegex(ValueError,'retained'):
            journal.collect(failed,FixtureBackend(failed),resume=True)
        report=analysis.analyze(failed)
        self.assertFalse(report['complete'] or report['replicatedIncrementalReadoutCriterion'])

    def test_incomplete_collection_never_satisfies_the_criterion(self):
        target=self.root/'partial.jsonl'
        rewrite(target,self.events[:10])
        report=analysis.analyze(target)
        self.assertFalse(report['complete'] or report['replicatedIncrementalReadoutCriterion'])
        self.assertEqual(report['contrasts'],[])


if __name__=='__main__':
    unittest.main()
