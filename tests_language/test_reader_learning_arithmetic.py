import copy
import math
import unittest

from research.audit_reader_learning_arithmetic import branch,calculate,compare_tree,ranks_auc,bootstrap
from research.prospective_reader_learning_metrics import summarize,interval
from tests_language.test_prospective_reader_learning import LearningMetricTests


class SeparateArithmeticTests(unittest.TestCase):
    def fixture(self,uniform=False):
        data=LearningMetricTests.fixture(uniform);p=data['header']['plan'];p['primary']['resamples']=2000
        tokens={'0':[15],'1':[16],'A':[32],'B':[33]};events=[dict(event='header',plan=p,origin='synthetic-test')]
        events.extend(dict(event='state',result=dict(data['states'][c['id']],case=c)) for c in p['cases'])
        for e in data['events']:
            if e['event'] in ('forecast','coding','task'):
                d=e['result']['decoded'];task=e['event']=='task'
                positive=d['decision']==('one' if task else 'correct')
                probability=(.9 if positive else .1) if task else d['conditionalPositive']
                config=p['taskMapping'] if task else p['forecastMappings'][e['call']['mapping']]
                ids=[tokens[c][0] for c in config['codes']]
                d.update(candidateTokenIds=ids,tokenIds=[ids[int(positive)],151645],eosTokenIds=[151645],
                    candidateLogits=[0.,math.log(probability/(1-probability))],conditionalPositive=probability,candidateMass=1.)
            events.append(e)
        return data,events,tokens

    def test_separate_reconstruction_matches_both_positive_and_negative_fixtures(self):
        for uniform in (False,True):
            data,events,tokens=self.fixture(uniform)
            first=summarize(data);second=calculate(events,tokens)
            self.assertEqual(second['allFunctionalCriteriaPassed'],not uniform)
            differences=compare_tree(second,{k:first[k] for k in second})
            self.assertLess(max(differences),1e-12)
            corrupt=copy.deepcopy(second);corrupt['primaryLossComparisons'][0]['interval']['mean']+=.01
            with self.assertRaises(AssertionError):compare_tree(corrupt,{k:first[k] for k in corrupt})

    def test_native_tokens_override_untrustworthy_validity_and_meaning_fields(self):
        tokens={'A':[32],'B':[33]};cfg=dict(codes=['B','A'])
        d=dict(candidateTokenIds=[33,32],tokenIds=[33,151645],eosTokenIds=[151645],candidateLogits=[1.,0.],
            conditionalPositive=1/(1+math.exp(1)),candidateMass=.9,validNativeResponse=False,decision='correct')
        result=branch(d,cfg,tokens)
        self.assertTrue(result['valid']);self.assertEqual(result['semanticBit'],0)
        d['tokenIds']=[33,999];self.assertFalse(branch(d,cfg,tokens)['valid'])
        d['candidateTokenIds']=[32,33]
        with self.assertRaises(AssertionError):branch(d,cfg,tokens)

    def test_midrank_auc_handles_ties_and_invalid_outputs(self):
        rows=[dict(label=y,probability=p,valid=True) for y,p in ((0,.2),(1,.2),(0,.3),(1,.8))]
        self.assertEqual(ranks_auc(rows),.625)
        rows[-1]['valid']=False;self.assertEqual(ranks_auc(rows),0.)
        self.assertIsNone(ranks_auc([dict(label=1,probability=.5,valid=True)]))

    def test_loop_bootstrap_matches_vectorized_fixed_table_draws(self):
        cfg=dict(seed=2026092072,resamples=20000,familyAlpha=.05,comparisons=6)
        values=[-.2,.15,.16,.20,.01,.25,.3,.05]
        self.assertLessEqual(max(compare_tree(bootstrap(values,cfg),interval(values,cfg))),1e-12)


if __name__=='__main__':unittest.main()
