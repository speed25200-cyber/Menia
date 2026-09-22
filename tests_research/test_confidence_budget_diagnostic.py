import copy
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from research import confidence_budget_diagnostic as study
from research.confidence_ranking import paired_batches
from research.cross_model_prediction import canonical,digest


class ConfidenceBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = study.make_plan(); cls.rows = study.load_rows()
        cls.by_id = {r['sourceId']:r for r in cls.rows}; cls.calls = study.calls(cls.rows,cls.plan)

    def test_extended_schedule_preserves_original_epochs_and_every_training_example(self):
        rows = study.fit.training_rows()
        for rep in range(3):
            original = study.epoch_batches(rows,rep,0)+study.epoch_batches(rows,rep,1)
            self.assertEqual(original,paired_batches(rows,rep))
            expected = Counter(r['sourceId'] for r in rows if r['replication'] == rep)
            for epoch in range(2,8):
                batches = study.epoch_batches(rows,rep,epoch)
                found = Counter(r['sourceId'] for b in batches for p in b for r in (p['left'],p['right']))
                self.assertEqual(found,expected)
                self.assertEqual(len(batches),72)
                self.assertTrue(all((p['left']['family'],p['left']['level']) ==
                                    (p['right']['family'],p['right']['level']) for b in batches for p in b))
            self.assertEqual(len(study.training_batches(rep)),432)
        changed = copy.deepcopy(rows); changed[0]['messages'][1]['content'] += ' altered'
        with self.assertRaises(ValueError): study.epoch_batches(changed,0,2)

    def test_evaluation_is_complete_and_contains_neither_label_nor_baseline_score(self):
        self.assertEqual(len(self.calls),8640)
        for unit in self.plan['units']:
            ids = [c['sourceId'] for c in self.calls if c['key'] == unit['key']]
            self.assertEqual(len(set(ids)),960)
            self.assertEqual(Counter(self.by_id[i]['phase'] for i in ids),Counter(train=576,seen24base=384))
        for call in self.calls:
            request = study.request_for(call,self.by_id)
            self.assertEqual(set(request),{'event','call','messages'})
            self.assertEqual(request['messages'],self.by_id[call['sourceId']]['messages'])

    def events(self,path):
        yield dict(event='header',plan=self.plan,planHash=digest(self.plan),sourceHash=study.source_hash(),
            callPlanHash=digest(self.calls),origin='synthetic_fixture',metadata=dict(model=self.plan['model'],
            newLLMBaseWeightUpdates=0,confidenceTokenIds=[15,16],vocabularySize=151936))
        for unit in self.plan['units']:
            if unit == self.plan['units'][3]:
                yield dict(event='baseline_complete',maxAbsoluteDifference=0.)
                for rep,initial in enumerate(self.plan['initial']):
                    yield dict(event='training_start',replication=rep,initialHash=initial['sha256'],optimizerReset=True,trainableParameters=2949120)
                    for step,batch in enumerate(study.training_batches(rep),1):
                        yield dict(event='training_step',replication=rep,step=step,dataHash=digest(batch),losses=[0.]*8,
                            pairLosses=[0.]*4,scoreDifferences=[0.]*4,eligible=[p['eligible'] for p in batch],
                            inputTokens=[20]*8,gradientNorm=0.,seconds=.1,objective=0.)
                        if step in (144,432):
                            key = f'r{rep}-e{2+step//72}'
                            yield dict(event='checkpoint',key=key,step=step,checkpoint=path.with_suffix('.'+key+'.safetensors').name,
                                sha256='a'*64,bytes=1,baseParameterVersionsUnchanged=True)
                    yield dict(event='training_complete',replication=rep,steps=432)
            yield dict(event='unit_start',key=unit['key'],sha256=self.plan['initial'][unit['replication']]['sha256'] if unit['epoch']==2 else 'a'*64)
            for call in self.calls:
                if call['key'] != unit['key']: continue
                row = self.by_id[call['sourceId']]; request = study.request_for(call,self.by_id)
                yield request
                scores = copy.deepcopy(row['baseline'])
                if unit['epoch'] != 2:
                    scores.update(conditionalCorrect=.8 if row['target']=='1' else .2,candidateMass=1.,
                        topTokenId=16 if row['target']=='1' else 15,topIsCode=True)
                yield dict(event='judgment',id=call['id'],status='ok',seconds=.01,scores=scores)
            yield dict(event='unit_complete',key=unit['key'],calls=960,actualAdapterMatchesFile=True,baseParameterVersionsUnchanged=True)
        yield dict(event='complete',calls=8640,steps=1296)

    def write(self,path,events):
        previous = '0'*64
        with path.open('w',encoding='utf-8') as stream:
            for i,event in enumerate(events):
                envelope = dict(sequence=i,previous=previous,payload=event)
                envelope['sha256'] = digest(envelope); previous = envelope['sha256']
                stream.write(canonical(envelope)+'\n')

    def test_full_synthetic_run_is_split_by_checkpoint_and_data_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.jsonl'; self.write(path,self.events(path))
            data = study.read_journal(path); report = study.summarize(data)
        self.assertTrue(data['complete']); self.assertEqual(report['recordedCalls'],8640)
        self.assertEqual(report['newAdapterUpdates'],1296); self.assertEqual(len(report['checkpoints']),6)
        self.assertEqual(len(report['units']),18); self.assertEqual(report['baselineMaxAbsoluteDifference'],0.)
        for key,group in report['units'].items():
            if '-e2/' not in key:
                self.assertAlmostEqual(group['brier'],.04,delta=1e-12)
                self.assertEqual(group['withinAuc'],1.)
            self.assertEqual(group['n'],576 if key.endswith('/train') else 384)
        self.assertIsNone(report['units']['r1-e8/train']['cells']['alternatingSum/8']['auc'])

    def test_rehashed_changed_baseline_inputs_and_early_training_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.jsonl'
            import itertools
            prefix = list(itertools.islice(self.events(path),4))
            wrong = copy.deepcopy(prefix); p = wrong[-1]['scores']['conditionalCorrect']
            wrong[-1]['scores']['conditionalCorrect'] = 1. if p<.5 else 0.
            self.write(path,wrong)
            with self.assertRaises(ValueError): study.read_journal(path)
            wrong = copy.deepcopy(prefix); wrong[2]['messages'][1]['content'] += ' label=1'
            self.write(path,wrong)
            with self.assertRaises(ValueError): study.read_journal(path)
            self.write(path,prefix+[dict(event='baseline_complete',maxAbsoluteDifference=0.)])
            with self.assertRaises(ValueError): study.read_journal(path)
            self.write(path,prefix+[dict(event='complete',calls=8640,steps=1296)])
            with self.assertRaises(ValueError): study.read_journal(path)
            self.write(path,prefix); data = study.read_journal(path)
            self.assertFalse(data['complete'])
            with self.assertRaises(ValueError): study.summarize(data)

    def test_rehashed_changed_training_batch_and_unfrozen_assessment_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.jsonl'; events = list(self.events(path))
            step_index = next(i for i,e in enumerate(events) if e['event'] == 'training_step')
            wrong = copy.deepcopy(events[:step_index+1]); wrong[-1]['dataHash'] = '0'*64
            self.write(path,wrong)
            with self.assertRaises(ValueError): study.read_journal(path)
            snapshot_index = next(i for i,e in enumerate(events) if e['event'] == 'checkpoint')
            wrong = copy.deepcopy(events[:snapshot_index+1]); wrong[-1]['step'] = 143
            self.write(path,wrong)
            with self.assertRaises(ValueError): study.read_journal(path)
            start_index = next(i for i,e in enumerate(events) if e['event'] == 'training_start')
            self.write(path,events[:start_index]+[dict(event='unit_start',key='r0-e4',sha256='a'*64)])
            with self.assertRaises(ValueError): study.read_journal(path)


if __name__ == '__main__': unittest.main()
