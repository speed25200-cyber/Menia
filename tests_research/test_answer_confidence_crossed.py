import copy
import unittest

from research.answer_confidence_crossed import ARMS,summarize


def fixture():
    return [dict(id=i,correct={p:int(i%2==1 or p=='measured') for p in ARMS},
                 judgments={g:{p:dict(conditionalCorrect=.8,candidateMass=.9,inputHash=str(i)*63+str(ARMS.index(p))) for p in ARMS} for g in ARMS}) for i in range(4)]


class CrossedConfidenceTests(unittest.TestCase):
    def test_apparent_gain_entirely_due_to_changed_answers(self):
        r=summarize(fixture());c=r['changes']['measured']
        self.assertAlmostEqual(c['coupledBrierGain'],.3)
        self.assertEqual(c['judgeGainOnBaseAnswers'],0.)
        self.assertEqual(c['judgeGainOnCurrentAnswers'],0.)
        self.assertAlmostEqual(c['answerChangeUnderCurrentJudge'],.3)

    def test_same_answer_improvement_is_not_a_producer_effect(self):
        rows=fixture()
        for row in rows:
            row['correct']={p:row['correct']['base'] for p in ARMS}
            for p in ARMS:row['judgments']['measured'][p]['conditionalCorrect']=.9 if row['correct'][p] else .1
        c=summarize(rows)['changes']['measured']
        self.assertAlmostEqual(c['coupledBrierGain'],.33)
        self.assertAlmostEqual(c['judgeGainOnBaseAnswers'],.33)
        self.assertEqual(c['answerChangeUnderCurrentJudge'],0.)

    def test_unmatched_incomplete_and_invalid_records_rejected(self):
        cases=[]
        a=fixture();a[0]['judgments']['measured']['base']['inputHash']='x'*64;cases.append(a)
        a=fixture();del a[0]['judgments']['base']['shuffled'];cases.append(a)
        a=fixture();a[0]['judgments']['base']['base']['conditionalCorrect']=float('nan');cases.append(a)
        a=fixture();a.append(copy.deepcopy(a[0]));cases.append(a)
        for rows in cases:
            with self.assertRaises(ValueError):summarize(rows)


if __name__=='__main__':unittest.main()
