import copy
import json
from pathlib import Path
import tempfile
import unittest

from research import value_action_learning as s
from research.value_action_learning_journal import read_journal
from research.audit_value_action_learning import verify
from research.natural_error_journal import Writer


class ValueActionLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.path=Path(cls.tmp.name)/'fixture.jsonl';p=s.make_plan();w=Writer(cls.path)
        w.write(dict(event='header',plan=p,planHash=s.digest(p),sourceHash=s.source_hash(),origin='synthetic_fixture',metadata={}),create=True)
        checkpoints={}
        for u in p['trainingUnits']:
            w.write(dict(event='training_start',key=u['key'],initializationHash='0'*64,trainableParameters=128))
            for step in range(1,65):
                w.write(dict(event='training_step',key=u['key'],step=step,dataHash=s.digest(s.training_batch(p,u,step)),losses=[1.]*8,gradientNorm=1.,inputTokens=[150]*8,seconds=.01))
            event=dict(event='training_complete',key=u['key'],steps=64,initializationHash='0'*64,sha256='1'*64,checkpoint=cls.path.with_suffix('.'+u['key']+'.safetensors').name,trainingSeconds=1.)
            w.write(event);checkpoints[u['key']]=event
        for c in p['calls']:
            req=s.request_for(p,c['id'],checkpoints);w.write(req)
            text=s.target(p['testCases'][c['case']],c) if c['arm']=='linked' else s.parent.symbols(c)['direct']
            if c['arm']=='shuffled' and c['symbols']=='letters' and c['case']==0:text='not a code'
            w.write(dict(event='result',id=c['id'],engine='llm',status='ok',text=text,seconds=.01,
                         metrics=dict(inputTokens=150,outputTokens=2,reachedTokenLimit=False,effectiveGeneration=p['settings']),errorType=None))
        cls.report=s.analyze(cls.path)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_new_cases_and_heldout_values_and_forms(self):
        p=s.make_plan();self.assertEqual((len(p['trainingCases']),len(p['testCases']),p['planned']),(32,24,9216))
        old=s.parent.make_plan();old_cases=old['trainingCases']+old['testCases']+s.risk_plan()['cases']
        old_pairs={(x['pPercent'],x['costCents']) for x in old_cases}
        for field in ('pPercent','costCents'):
            self.assertFalse({c[field] for c in p['trainingCases']} & {c[field] for c in p['testCases']})
        self.assertFalse(old_pairs & {(c['pPercent'],c['costCents']) for c in p['testCases']+p['trainingCases']})
        self.assertEqual(sum(s.optimal(c)=='direct' for c in p['testCases']),12)
        self.assertFalse(any(c['pPercent']+c['costCents']==100 for c in p['testCases']+p['trainingCases']))
        self.assertEqual({c['wording'] for c in p['calls'] if c['task']=='choice'},{1,2,3})

    def test_matched_examples_shared_choices_and_shuffled_label_counts(self):
        p=s.make_plan();changes=0
        for rep in range(3):
            for step in range(1,65):
                batches={a:s.training_batch(p,dict(replication=rep,arm=a),step) for a in s.ARMS[1:]}
                self.assertEqual({len(v) for v in batches.values()},{8})
                self.assertEqual(sum(x['task']=='choice' for x in batches['choice']),8)
                self.assertEqual(sum(x['task']=='choice' for x in batches['linked']),4)
                true_aux=[];shuffled_aux=[]
                for baseline,linked,shuffled in zip(batches['choice'],batches['linked'],batches['shuffled']):
                    self.assertEqual(linked['messages'],shuffled['messages'])
                    if linked['task']=='choice':self.assertEqual(baseline,linked);self.assertEqual(linked,shuffled)
                    else:
                        true_aux.append(linked['target']);shuffled_aux.append(shuffled['target'])
                        changes+=linked['target']!=shuffled['target']
                self.assertEqual(sorted(true_aux),['1','1','2','2']);self.assertEqual(sorted(true_aux),sorted(shuffled_aux))
        self.assertGreater(changes,0)

    def test_complete_fixture_and_independent_recompute(self):
        self.assertTrue(self.report['complete']);self.assertTrue(self.report['robustLearningCriterion'])
        self.assertEqual(len(self.report['groups']),384)
        self.assertTrue(verify(self.path,self.report,check_weights=False)['verified'])
        altered=copy.deepcopy(self.report);altered['contrasts'][0]['accuracy']['linked']-=.01
        with self.assertRaises(ValueError):verify(self.path,altered,check_weights=False)

    def test_rehashed_auxiliary_training_change_rejected(self):
        rows=[json.loads(l) for l in self.path.read_text(encoding='utf-8').splitlines()]
        row=next(r for r in rows if r['payload']['event']=='training_step' and r['payload']['key']=='r0-linked')
        row['payload']['dataHash']='f'*64;previous='0'*64
        for row in rows:
            row['previous']=previous;row['sha256']=s.digest({k:row[k] for k in ('sequence','previous','payload')});previous=row['sha256']
        path=Path(self.tmp.name)/'changed.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        with self.assertRaises(ValueError):read_journal(path)


if __name__=='__main__':unittest.main()
