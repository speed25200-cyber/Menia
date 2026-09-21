import copy
import unittest
from collections import Counter,defaultdict

from research import prospective_reader_learning as study
from research.prospective_reader_learning_audit import validate_order
from research.prospective_reader_learning_metrics import interval,penalized_loss,table_auc,summarize


class LearningProtocolTests(unittest.TestCase):
    def test_reserved_states_and_outcomes_are_after_training_and_forecasts(self):
        p=study.plan();order=list(study.operations(p))
        last_update=max(i for i,(k,c) in enumerate(order) if k=='step')
        first_reserved=min(i for i,(k,c) in enumerate(order) if c.get('split')=='reserved')
        barrier=next(i for i,(k,c) in enumerate(order) if k=='forecasts_complete')
        self.assertLess(last_update,first_reserved)
        self.assertTrue(all(i<barrier for i,(k,c) in enumerate(order) if k=='forecast' and c['split']=='reserved'))
        self.assertTrue(all(i>barrier for i,(k,c) in enumerate(order) if k=='task' and c['split']=='reserved'))
        self.assertEqual(Counter(k for k,c in order),dict(state=48,task=1408,weights=3,reset=2,
            step=448,coding=1216,forecast=4864,forecasts_complete=1))

    def test_matched_training_never_uses_reserved_and_balances_code_inversions(self):
        p=study.plan();train={c['id'] for c in study.cases('train',p)};counts=defaultdict(Counter)
        epochs=[study.training_examples(e,p) for e in range(4)]
        self.assertEqual(epochs,[study.training_examples(e,p) for e in range(4)])
        for rows in epochs:
            self.assertEqual(len(rows),896)
            self.assertTrue(all(r['case'] in train for r in rows))
            for r in rows:
                if r['kind']=='forecast':counts[r['case'],r['target'],r['condition']][r['mapping']]+=1
        self.assertTrue(all(v=={0:2,1:2} for v in counts.values()))
        for condition in p['conditions']:
            self.assertEqual(study.memory_condition('text',condition),'same_schedule')
            self.assertEqual(study.memory_condition('state',condition),condition)

    def test_supervision_is_measured_task_outcome_and_invalid_is_failure(self):
        c={'case':1000,'target':0,'condition':'values_permuted','mapping':0}
        cases={1000:{'values':[1]}}
        tasks={(1000,0,'values_permuted'):{'decoded':dict(validNativeResponse=True,decision='one')}}
        self.assertEqual(study.target_label('forecast',c,tasks,cases),1)
        tasks[1000,0,'values_permuted']['decoded']['validNativeResponse']=False
        self.assertEqual(study.target_label('forecast',c,tasks,cases),0)
        self.assertEqual(study.target_label('coding',dict(verdict=1),{},{}),1)
        with self.assertRaises(KeyError):study.target_label('forecast',dict(c,case=1032),tasks,cases)

    def test_order_audit_rejects_early_outcome_duplicate_and_terminal_append(self):
        p=study.plan();first_kind,first_call=next(study.operations(p))
        prefix=[dict(event='header'),dict(event='request',kind=first_kind,call=first_call),
            dict(event=first_kind,call=first_call,result={},seconds=0)]
        self.assertEqual(validate_order(prefix,p)['operations'],1)
        failed=prefix+[dict(event='failure',error='preserved')]
        self.assertIsNotNone(validate_order(failed,p)['failure'])
        for bad in (prefix+prefix[1:],failed+prefix[1:],
                    [dict(event='header'),dict(event='request',kind='task',call=dict(case=1032,split='reserved'))],
                    prefix+[dict(event='complete',counts=p['counts'],producerUnchanged=True,peakGPUBytes=1)]):
            with self.assertRaises(ValueError):validate_order(bad,p)


class LearningMetricTests(unittest.TestCase):
    def test_invalid_report_cannot_score_as_good_probabilistic_prediction(self):
        self.assertEqual(penalized_loss(1,1,False),1)
        self.assertEqual(penalized_loss(1,1,True),0)
        rows=[dict(label=0,probability=.1,valid=True),dict(label=1,probability=.9,valid=True)]
        self.assertEqual(table_auc(rows),1)
        rows[0]['valid']=False;self.assertEqual(table_auc(rows),0)
        self.assertIsNone(table_auc([rows[1]]))

    def test_uniform_confidence_reduction_does_not_count_as_discrimination(self):
        for confidence in (.99,.75,.5):
            rows=[dict(label=i,probability=confidence,valid=True) for i in (0,0,1,1,1,1,1,1)]
            self.assertEqual(table_auc(rows),.5)

    def test_cluster_interval_is_deterministic_and_uses_correct_family(self):
        cfg=study.plan()['primary'];ci=interval([.25]*16,cfg)
        self.assertEqual(ci,interval([.25]*16,cfg))
        self.assertEqual((ci['low'],ci['high'],ci['mean']),(.25,.25,.25))
        self.assertAlmostEqual(ci['coverage'],1-.05/6)
        mixed=interval([-1,1],cfg)
        self.assertEqual((mixed['low'],mixed['high']),(-1,1))

    @staticmethod
    def fixture(uniform=False):
        p=study.plan();events=[];states={};cm={c['id']:c for c in p['cases']}
        for case in p['cases']:states[case['id']]=dict(bindingMask=dict(selectedRows=[0,1]))
        for split in ('train','reserved'):
            for call in study.calls('task',split,p=p):
                bit=cm[call['case']]['values'][call['target']]
                fails=call['condition']=='values_permuted' and call['target']<2
                events.append(dict(event='task',call=call,result=dict(decoded=dict(validNativeResponse=True,
                    decision='one' if bit^fails else 'zero'))))
            for arm in (('state','text') if split=='train' else study.ARMS):
                for kind in ('forecast','coding'):
                    for call in study.calls(kind,split,arm,p):
                        label=call['verdict'] if kind=='coding' else int(not(call['condition']=='values_permuted' and call['target']<2))
                        probability=(.99 if label else .01) if kind=='coding' or arm=='state' and not uniform else .75
                        decoded=dict(validNativeResponse=True,conditionalPositive=probability,
                            decision='correct' if probability>.5 else 'incorrect',candidateMass=1.)
                        events.append(dict(event=kind,call=call,result=dict(decoded=decoded)))
        for arm in ('state','text'):
            for epoch in range(4):
                events.append(dict(event='step',call=dict(arm=arm,epoch=epoch),result=dict(meanLoss=.1,examples=[0]*16)))
        return dict(complete=True,failure=None,header=dict(plan=p,planHash=study.digest(p),sourceHash='fixture'),
            events=events,states=states,chainEnd='fixture')

    def test_summary_requires_rank_improvement_not_only_score_improvement(self):
        good=summarize(self.fixture())
        self.assertTrue(good['allFunctionalCriteriaPassed'])
        self.assertEqual(len(good['primaryLossComparisons']),4)
        self.assertEqual(len(good['primaryDiscriminationComparisons']),2)
        self.assertTrue(all(x['successes']==96 and x['failures']==32 for x in good['gates']))
        uniform=summarize(self.fixture(uniform=True))
        self.assertFalse(uniform['allFunctionalCriteriaPassed'])
        self.assertTrue(all(not x['passed'] for x in uniform['primaryDiscriminationComparisons']))
        self.assertEqual(good['constantDiagnostic']['trainingVOnlyPrevalence'],.75)


if __name__=='__main__':unittest.main()
