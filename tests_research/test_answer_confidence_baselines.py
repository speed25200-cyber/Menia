import copy
import math
import unittest

from research.answer_confidence_baselines import fit,forecast,features
from research.answer_confidence_crossed import ARMS
from research.cross_model_prediction import CELLS


def fixture():
    rows=[]
    for family,level in CELLS:
        for i in range(16):
            task=dict(id=len(rows),replication=0,split='calibration',family=family,level=level,letters='A',operands=[1,0])
            answers={}
            for producer in ARMS:
                pre=dict(vocabularySize=100,topTokenId=1,maxProbability=.3,topTwoMargin=.1,entropyNats=2.,normalizedEntropy=2/math.log(100))
                trace=dict(schema='menia-output-confidence-trace-v1',preAnswer=pre,
                           logitSource='lm_head output, before generation processors and sampling',
                           distribution='raw full vocabulary; temperature 1; no top-k/top-p filtering',
                           completion=dict(tokenIds=[1,2],tokenLogProbabilities=[-.3,-.2],entropyNats=[2.,2.],includesStopTokens=True,sumLogProbability=-.5,meanLogProbability=-.25))
                answers[producer]=dict(status='ok',text='1' if producer=='measured' or i%2 else '9',seconds=.1,
                                       metrics=dict(inputTokens=10,outputTokens=2,reachedTokenLimit=False),trace=trace)
            rows.append(dict(task=task,answers=answers))
    return rows


class ConfidenceBaselineTests(unittest.TestCase):
    def test_current_producer_calibration_and_heldout_invariance(self):
        rows=fixture();a=fit(rows,0)
        for cell in a['beta']['base']:
            self.assertEqual(a['beta']['base'][cell],.5)
            self.assertEqual(a['beta']['measured'][cell],17/18)
        # The future test is deliberately not even a readable result: fitting
        # must not access it, rather than merely refrain from using its label.
        self.assertEqual(a,fit(rows+[dict(task=dict(replication=0,split='test'),answers=None)],0))
        row=rows[0];task=dict(row['task'],split='test');result=row['answers']['base']
        p=forecast(a,task,result,'base');changed=dict(task,letters='AAAA',operands=[99,1])
        self.assertEqual(features(task,result),features(changed,result))
        self.assertEqual(p,forecast(a,changed,result,'base'))

    def test_missing_failed_duplicate_and_invalid_confidence_rejected(self):
        rows=fixture()
        with self.assertRaises(ValueError):fit(rows[:-1],0)
        bad=copy.deepcopy(rows);bad[0]['answers']['base']['status']='error'
        with self.assertRaises(ValueError):fit(bad,0)
        bad=copy.deepcopy(rows);bad[1]['task']['id']=bad[0]['task']['id']
        with self.assertRaises(ValueError):fit(bad,0)
        bad=copy.deepcopy(rows);bad[0]['answers']['base']['trace']['preAnswer']['maxProbability']=float('nan')
        with self.assertRaises(ValueError):fit(bad,0)


if __name__=='__main__':unittest.main()
