import copy
import json
from pathlib import Path
import tempfile
import unittest

from research import action_binding as s
from research.action_binding_journal import read_journal
from research.audit_action_binding import verify
from research.natural_error_journal import Writer


class ActionBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.path=Path(cls.tmp.name)/'fixture.jsonl';p=s.make_plan();w=Writer(cls.path)
        w.write(dict(event='header',plan=p,planHash=s.digest(p),sourceHash=s.source_hash(),origin='synthetic_fixture',metadata={}),create=True)
        checkpoints={}
        for u in p['trainingUnits']:
            w.write(dict(event='training_start',key=u['key'],initializationHash='0'*64,trainableParameters=128))
            for step in range(1,65):w.write(dict(event='training_step',key=u['key'],step=step,dataHash=s.digest(s.training_batch(p,u,step)),losses=[1.]*8,gradientNorm=1.,inputTokens=[120]*8))
            event=dict(event='training_complete',key=u['key'],steps=64,initializationHash='0'*64,sha256='1'*64,checkpoint=cls.path.with_suffix('.'+u['key']+'.safetensors').name)
            w.write(event);checkpoints[u['key']]=event
        for c in p['calls']:
            req=s.request_for(p,c['id'],checkpoints);w.write(req)
            # A heterogeneous fixture: perfect permutation-trained policy,
            # constant direct baseline, invalid fixed arm on one unseen symbol.
            text=s.target(p['testCases'][c['case']],c) if c['arm']=='permuted' else s.symbols(c)['direct']
            if c['arm']=='fixed' and c['symbols']=='letters' and c['case']==0:text='extra text'
            w.write(dict(event='result',id=c['id'],engine='llm',status='ok',text=text,seconds=.01,metrics=dict(inputTokens=120,outputTokens=2,reachedTokenLimit=False,effectiveGeneration=s.DECISION_SETTINGS),errorType=None))
        cls.report=s.analyze(cls.path)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_no_train_test_value_overlap_and_matched_exposures(self):
        p=s.make_plan();self.assertEqual((len(p['trainingCases']),len(p['testCases']),p['planned']),(32,16,6912))
        for field in ('pPercent','costCents'):self.assertFalse({c[field] for c in p['trainingCases']} & {c[field] for c in p['testCases']})
        for rep in range(3):
            for step in range(1,65):
                fixed,permuted=[s.training_batch(p,dict(replication=rep,arm=a),step) for a in ('fixed','permuted')]
                self.assertEqual(len(fixed),len(permuted));self.assertEqual(len({s.digest(x) for x in fixed}),2);self.assertEqual(len({s.digest(x) for x in permuted}),8)
                self.assertEqual(sum(x['target']=='1' for x in permuted),4)
        self.assertEqual(sum(c['pPercent']+c['costCents']>100 for c in p['testCases']),8)

    def test_symbols_and_order_follow_semantics_without_exposing_targets(self):
        p=s.make_plan()
        for c in p['calls'][:768]:
            case=p['testCases'][c['case']];m=s.messages(case,c);body=m[1]['content'];codes=s.symbols(c)
            first=body.splitlines()[0]
            self.assertIn(codes['direct' if c['order']==0 else 'verify'],first)
            self.assertIn(s.target(case,c),('1','2') if c['symbols']=='digits' else ('A','B'))
            if c['symbols']=='letters':self.assertNotIn('1 ou 2',m[0]['content'])
            self.assertNotIn('optimal',body)

    def test_complete_fixture_is_reconstructed_and_separate_tally_rejects_edits(self):
        r=self.report;self.assertTrue(r['complete']);self.assertTrue(r['robustLearningCriterion']);self.assertEqual(len(r['groups']),432)
        self.assertTrue(verify(self.path,r,check_weights=False)['verified'])
        changed=copy.deepcopy(r);changed['contrasts'][0]['accuracy']['permuted']-=.01
        with self.assertRaises(ValueError):verify(self.path,changed,check_weights=False)

    def test_rehashed_training_or_test_leak_is_rejected(self):
        rows=[json.loads(l) for l in self.path.read_text(encoding='utf-8').splitlines()]
        row=next(x for x in rows if x['payload']['event']=='training_step');row['payload']['dataHash']='f'*64
        previous='0'*64
        for row in rows:
            row['previous']=previous;row['sha256']=s.digest({k:row[k] for k in ('sequence','previous','payload')});previous=row['sha256']
        path=Path(self.tmp.name)/'altered.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        with self.assertRaises(ValueError):read_journal(path)


if __name__=='__main__':unittest.main()
