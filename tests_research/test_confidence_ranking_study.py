import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research import confidence_ranking_study as study
from research.confidence_ranking import paired_batches, preparation_report
from research.confidence_ranking_journal import read_journal, validate_step
from research.confidence_ranking_analysis import weighted_within_auc, question_weights, analyze
from research.answer_confidence_plan import load_training, make_plan as previous_questions
from research.natural_error_questions import exclusions, make_plan as original_questions
from research.cross_model_prediction import CELLS, digest, canonical, reference
from tests_research.test_answer_confidence_baselines import fixture as baseline_fixture

ROOT = Path(__file__).resolve().parents[1]


def fixture_events(path):
    """Oracle-created synthetic responses for software validation only."""
    p = study.make_plan(); data, parent = load_training(ROOT/'artifacts/answer-confidence-training-data')
    rows = data['measured']; answers = {}; checkpoints = {}
    yield dict(event='header', plan=p, planHash=digest(p), sourceHash=study.source_hash(), origin='synthetic_fixture',
               trainingData=rows, trainingReport=preparation_report(rows), parentTrainingReportHash=digest(parent),
               metadata=dict(confidenceTokenIds=[15, 16], vocabularySize=100))
    for unit in p['trainingUnits']:
        key = unit['key']; initial = digest(['initial', unit['replication']]); arm = unit['arm']
        yield dict(event='training_start', key=key, initializationHash=initial, trainableParameters=8)
        for step, batch in enumerate(paired_batches(rows, unit['replication']), 1):
            pair_losses = [math.log(2) if pair['eligible'] and arm != 'ce' else 0. for pair in batch]
            yield dict(event='training_step', key=key, step=step, dataHash=digest(batch), losses=[.1]*8,
                       pairLosses=pair_losses, scoreDifferences=[0.]*4, eligible=[pair['eligible'] for pair in batch],
                       inputTokens=[32]*8, gradientNorm=1., seconds=.1, objective=.1+sum(pair_losses)/4)
        event = dict(event='training_complete', key=key, steps=144, initializationHash=initial,
                     sha256=digest(['trained', key]), checkpoint=Path(path).with_suffix('.'+key+'.safetensors').name,
                     trainingSeconds=14.4)
        checkpoints[key] = event; yield event
    template = baseline_fixture()[0]['answers']['base']
    for c in p['calls']:
        request = study.request_for(p, c['id'], answers, checkpoints); yield request
        task = p['tasks'][c['task']]; good = task['block'] % 3 != 0
        if c['kind'] == 'answer':
            event = copy.deepcopy(template)
            event.update(event='answer', id=c['id'], errorType=None, text=str(reference(task)+int(not good)))
            event['metrics']['effectiveGeneration'] = copy.deepcopy(p['generation'])
            answers[(task['id'], c['producer'])] = event
        else:
            prob = (.9 if good else .1) if c['judge'] == 'rank' else .5
            event = dict(event='judgment', id=c['id'], status='ok', seconds=.01,
                scores=dict(conditionalCorrect=prob, candidateMass=.9, inputHash=digest(request['messages']), inputTokens=32,
                    codeTokenIds=[15, 16], topTokenId=16 if prob >= .5 else 15, topIsCode=True, sampled=False,
                    source='Recomputed full text prefix, raw lm_head logits at temperature 1'))
        yield event


def write_fixture(path, events):
    # Production journals fsync every event; tests write once, with the same
    # envelope format. Synthetic origin must never be reported as measurement.
    previous = '0'*64
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for i, event in enumerate(events):
            value = dict(sequence=i, previous=previous, payload=event)
            previous = digest(value)
            stream.write(canonical(dict(value, sha256=previous))+'\n')


class RankingStatisticsTests(unittest.TestCase):
    def test_weighted_auc_matches_explicit_question_duplication_with_ties(self):
        y = np.array([1, 0, 1, 0, 0, 1]); p = np.array([.4, .4, .7, .1, .8, .3]); c = np.array([0, 0, 0, 1, 1, 1])
        weights = np.array([[1, 1, 1, 1, 1, 1], [2, 3, 1, 0, 2, 4]])
        auc, denominators = weighted_within_auc(y, p, c, weights)
        for k, w in enumerate(weights):
            ids = np.repeat(np.arange(len(y)), w); values = []
            for i in ids:
                for j in ids:
                    if y[i] == 1 and y[j] == 0 and c[i] == c[j]:
                        values.append(float(p[i] > p[j])+.5*float(p[i] == p[j]))
            self.assertAlmostEqual(auc[k], float(np.mean(values)))
            self.assertEqual(denominators[k], len(values))
        sampled = question_weights(c, 100, 44)
        self.assertTrue(np.all(sampled[:, c == 0].sum(axis=1) == 3))
        self.assertTrue(np.all(sampled[:, c == 1].sum(axis=1) == 3))
        self.assertTrue(np.array_equal(sampled, question_weights(c, 100, 44)))

    def test_between_category_success_does_not_pass_within_category_ranking(self):
        y = [1, 1, 1, 0, 1, 0, 0, 0]; p = [.9]*4+[.1]*4; cells = [0]*4+[1]*4
        auc, _ = weighted_within_auc(y, p, cells, np.ones((1, 8)))
        self.assertEqual(auc[0], .5)
        missing, pairs = weighted_within_auc([1, 1, 0, 0], [.9, .8, .2, .1], [0, 0, 1, 1], np.ones((1, 4)))
        self.assertTrue(np.isnan(missing[0])); self.assertEqual(pairs[0], 0)

    def test_fresh_complete_plan_and_pair_objective_validation(self):
        p = study.make_plan(); tasks = p['tasks']; questions = {t['question'] for t in tasks}
        old = exclusions() | {t['question'] for q in (original_questions(), previous_questions()) for t in q['tasks']}
        self.assertEqual(len(questions), 1152); self.assertFalse(questions & old)
        self.assertEqual((p['plannedGenerations'], p['plannedJudgments'], p['plannedCalls']), (4608, 18432, 23040))
        self.assertEqual(len(p['trainingUnits']), 9); self.assertEqual(p, study.make_plan())
        for rep in range(3):
            for cell in CELLS:
                self.assertEqual(sum((t['replication'], t['family'], t['level']) == (rep, *cell) for t in tasks), 64)
        data, _ = load_training(ROOT/'artifacts/answer-confidence-training-data')
        batch = paired_batches(data['measured'], 0)[0]
        event = next(e for e in fixture_events(Path('trial.jsonl')) if e['event'] == 'training_step')
        validate_step(event, batch, 'ce')
        event['pairLosses'][0] = .1
        with self.assertRaises(ValueError): validate_step(event, batch, 'ce')


class RankingJournalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(); cls.path = Path(cls.directory.name)/'trial.jsonl'
        write_fixture(cls.path, fixture_events(cls.path))
        cls.report = analyze(cls.path)

    @classmethod
    def tearDownClass(cls): cls.directory.cleanup()

    def test_complete_crossed_synthetic_trial_and_expected_primary_gains(self):
        r = self.report
        self.assertEqual(r['origin'], 'synthetic_fixture'); self.assertTrue(r['rankingCriterion'])
        self.assertEqual((r['recordedCalls'], r['trainingSteps']), (23040, 1296))
        self.assertEqual(len(r['contrasts']), 18)
        for contrast in r['contrasts']:
            self.assertEqual(contrast['withinAucGain'], .5)
            self.assertEqual(contrast['familyInterval'], [.5, .5])
        for rep in r['replications'].values():
            for producer in study.MODEL_ARMS:
                self.assertEqual(rep['scores'][producer]['rank']['withinAuc'], 1.)
                self.assertAlmostEqual(rep['scores'][producer]['rank']['brier'], .01)

    def test_rehashed_changed_targets_initialization_or_context_rejected(self):
        for mutation in ('target', 'initial', 'context'):
            def changed():
                mutated = False
                for e in fixture_events(self.path):
                    if not mutated:
                        if mutation == 'target' and e['event'] == 'header':
                            e['trainingData'][0]['target'] = '1' if e['trainingData'][0]['target'] == '0' else '0'; mutated = True
                        if mutation == 'initial' and e['event'] == 'training_start' and e['key'] == 'r0-rank':
                            e['initializationHash'] = 'f'*64; mutated = True
                        if mutation == 'context' and e['event'] == 'judgment':
                            e['scores']['inputHash'] = 'f'*64; mutated = True
                    yield e
                    if mutated: break
            path = Path(self.directory.name)/mutation/'trial.jsonl'; write_fixture(path, changed())
            with self.assertRaises(ValueError): read_journal(path)

    def test_evaluation_before_all_adapters_frozen_rejected(self):
        def early():
            for e in fixture_events(self.path):
                if e['event'] == 'training_complete' and e['key'] == 'r2-neutral': continue
                yield e
                if e['event'] == 'request': break
        path = Path(self.directory.name)/'early'/'trial.jsonl'; write_fixture(path, early())
        with self.assertRaises(ValueError): read_journal(path)


if __name__ == '__main__':
    unittest.main()
