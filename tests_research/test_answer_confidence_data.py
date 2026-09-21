import copy
import math
import unittest
from collections import Counter,defaultdict

from research.answer_confidence_data import build_examples, input_messages, confidence_from_logits


def row(index, text, split='train', answer=1):
    return dict(task=dict(id=index,replication=index%3,split=split,family='countA',level=8,
                         question=f'Compter A dans {index} : '+('A'*answer)+'B'*(8-answer),letters='A'*answer+'B'*(8-answer)),
                result=dict(status='ok',text=text))


class AnswerConfidenceDataTests(unittest.TestCase):
    def test_inputs_independent_of_oracle_and_heldout_answers(self):
        a=row(0,'1');b=copy.deepcopy(a);b['task']['letters']='AABBBBBB'
        ea=build_examples([a])['measured'][0];eb=build_examples([b])['measured'][0]
        self.assertEqual(ea['messages'],eb['messages']);self.assertEqual((ea['target'],eb['target']),('1','0'))
        # Deliberately inconsistent oracle above: it changes ONLY the target,
        # never the input. Real inputs come from the audited parent plan.
        with_heldout=[a,row(1,'SECRET_TEST_VALUE','test'),row(2,'SECRET_VALIDATION_VALUE','validation')]
        self.assertEqual(build_examples([a]),build_examples(with_heldout))
        with_heldout[1]['result']['text']='OTHER_TEST_VALUE'
        self.assertEqual(build_examples([a]),build_examples(with_heldout))

    def test_shuffle_preserves_each_category_and_shared_inputs(self):
        rows=[row(i,'1' if i%2 else '999') for i in range(48)]
        d=build_examples(rows);counts=defaultdict(lambda:[Counter(),Counter()]);changes=0
        for a,b in zip(d['measured'],d['shuffled']):
            self.assertEqual(a['messages'],b['messages']);self.assertEqual(a['sourceId'],b['sourceId'])
            key=(a['replication'],a['family'],a['level'])
            counts[key][0][a['target']]+=1;counts[key][1][b['target']]+=1;changes+=a['target']!=b['target']
        self.assertTrue(all(a==b for a,b in counts.values()));self.assertGreater(changes,0)

    def test_invalid_format_retained_and_technical_failure_rejected(self):
        for text in ('1.0','01','La réponse est 1',''):
            self.assertEqual(build_examples([row(0,text)])['measured'][0]['target'],'0')
        broken=row(0,'');broken['result']['status']='error'
        with self.assertRaises(ValueError):build_examples([broken])
        with self.assertRaises(ValueError):build_examples([row(0,'1'),row(0,'1')])

    def test_pair_probability_does_not_hide_low_vocabulary_mass(self):
        a=confidence_from_logits([0,math.log(3),0],0,1)
        b=confidence_from_logits([0,math.log(3),1000],0,1)
        self.assertAlmostEqual(a['conditionalCorrect'],.75)
        self.assertAlmostEqual(a['candidateMass'],.8)
        self.assertEqual(a['conditionalCorrect'],b['conditionalCorrect'])
        self.assertEqual(b['candidateMass'],0.)
        self.assertAlmostEqual(confidence_from_logits([1000,1000],0,1)['conditionalCorrect'],.5)
        for values,ids in [([0,float('nan')],(0,1)),([0,1],(1,1)),([0,1],(0,3))]:
            with self.assertRaises(ValueError):confidence_from_logits(values,*ids)


if __name__=='__main__':unittest.main()
