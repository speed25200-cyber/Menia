import copy
import json
from pathlib import Path
import tempfile
import unittest

from research import confidence_final_training_fit as study
from research.cross_model_prediction import canonical, digest


class FinalTrainingFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = study.make_plan(); cls.rows = study.training_rows()
        cls.by_id = {r['sourceId']:r for r in cls.rows}; cls.calls = study.calls(cls.rows, cls.plan)

    def events(self):
        yield dict(event='header',plan=self.plan,planHash=digest(self.plan),sourceHash=study.source_hash(),
                   callPlanHash=digest(self.calls),origin='synthetic_fixture',
                   metadata=dict(model=self.plan['model'],newWeightUpdates=0,confidenceTokenIds=[1,2],vocabularySize=4))
        for unit in self.plan['units']:
            yield dict(event='unit_start',key=unit['key'],sha256=unit['sha256'])
            for call in self.calls:
                if call['key'] != unit['key']: continue
                request = study.request_for(call, self.by_id); row = self.by_id[call['sourceId']]
                positive = row['target'] == '1'
                yield request
                yield dict(event='judgment',id=call['id'],status='ok',seconds=.01,
                    scores=dict(conditionalCorrect=.8 if positive else .2,candidateMass=1.,
                        inputHash=digest(request['messages']),inputTokens=20,codeTokenIds=[1,2],
                        topTokenId=2 if positive else 1,topIsCode=True,sampled=False,
                        source='Recomputed full text prefix, raw lm_head logits at temperature 1'))
            yield dict(event='unit_complete',key=unit['key'],calls=576,sha256=unit['sha256'],
                       actualAdapterMatchesFile=True,baseParameterVersionsUnchanged=True)
        yield dict(event='complete',calls=5184)

    def write(self, path, events):
        previous = '0'*64
        with path.open('w',encoding='utf-8') as stream:
            for i,event in enumerate(events):
                envelope = dict(sequence=i,previous=previous,payload=event)
                envelope['sha256'] = digest(envelope); previous = envelope['sha256']
                stream.write(canonical(envelope)+'\n')

    def test_all_and_only_each_adapters_training_rows_are_requested_without_labels(self):
        self.assertEqual(len(self.calls),5184)
        for unit in self.plan['units']:
            selected = [self.by_id[c['sourceId']] for c in self.calls if c['key'] == unit['key']]
            self.assertEqual(len(selected),576)
            self.assertEqual(len({r['sourceId'] for r in selected}),576)
            self.assertTrue(all(r['split'] == 'train' and r['replication'] == unit['replication'] for r in selected))
        for call in self.calls:
            request = study.request_for(call,self.by_id)
            self.assertEqual(request['messages'],self.by_id[call['sourceId']]['messages'])
            self.assertNotIn('target',request)
        changed = copy.deepcopy(self.rows); changed[0]['split'] = 'test'
        with self.assertRaises(ValueError): study.calls(changed,self.plan)

    def test_complete_synthetic_fit_and_two_arithmetic_paths_preserve_single_class_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.jsonl'; self.write(path,self.events())
            data = study.read_journal(path); report = study.summarize(data)
        self.assertTrue(data['complete']); self.assertEqual(report['origin'],'synthetic_fixture')
        self.assertEqual(report['recordedCalls'],5184)
        for group in report['units'].values():
            self.assertAlmostEqual(group['brier'],.04,delta=1e-12)
            self.assertEqual(group['withinAuc'],1.)
            self.assertEqual(group['binaryAgreement'],1.)
        self.assertIsNone(report['units']['r1-rank']['cells']['alternatingSum/8']['auc'])
        self.assertLess(report['separateArithmeticMaxDifference'],1e-12)

    def test_rehashed_wrong_inputs_weights_duplicate_or_premature_complete_are_rejected(self):
        prefix = list(self.events())[:4]
        mutations = []
        wrong = copy.deepcopy(prefix); wrong[2]['messages'][1]['content'] += ' altered'; mutations.append(wrong)
        wrong = copy.deepcopy(prefix); wrong[1]['sha256'] = '0'*64; mutations.append(wrong)
        mutations.append(prefix+[copy.deepcopy(prefix[-1])])
        mutations.append(prefix+[dict(event='complete',calls=5184)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.jsonl'
            for events in mutations:
                self.write(path,events)
                with self.assertRaises(ValueError): study.read_journal(path)
            self.write(path,prefix)
            partial = study.read_journal(path); self.assertFalse(partial['complete'])
            with self.assertRaises(ValueError): study.summarize(partial)


if __name__ == '__main__': unittest.main()
