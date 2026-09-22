import copy
from collections import Counter
import unittest

from research.answer_confidence_data import build_examples
from research.answer_confidence_plan import make_plan,training_batches,validate_training,COUNTS
from research.cross_model_prediction import CELLS,digest
from research.natural_error_questions import exclusions,make_plan as parent_questions


def prepared_fixture():
    rows=[]
    for rep in range(3):
        for family,level in CELLS:
            for i in range(96):
                task=dict(id=len(rows),replication=rep,split='train',family=family,level=level,
                          question=f'question {len(rows)}',letters='A',operands=[1,0])
                rows.append(dict(task=task,result=dict(status='ok',text='1' if i%3==0 else '9')))
    examples=build_examples(rows)
    report=dict(exampleHashes={a:digest(r) for a,r in examples.items()},inputHash=digest([r['messages'] for r in examples['measured']]))
    return examples,report


class ConfidencePlanTests(unittest.TestCase):
    def test_fresh_questions_and_complete_crossed_budget(self):
        plan=make_plan();tasks=plan['tasks'];questions={t['question'] for t in tasks}
        old={t['question'] for t in parent_questions()['tasks']}|exclusions()
        self.assertEqual(len(questions),864);self.assertFalse(questions & old)
        counts=Counter((t['replication'],t['split'],t['family'],t['level']) for t in tasks)
        self.assertEqual(counts,Counter({(r,s,*c):n for r in range(3) for s,n in COUNTS.items() for c in CELLS}))
        self.assertEqual((plan['plannedGenerations'],plan['plannedJudgments']),(2592,7776))
        self.assertEqual(plan,make_plan())

    def test_training_visits_every_example_twice_with_matched_inputs(self):
        examples,report=prepared_fixture();validate_training(examples,report)
        for rep in range(3):
            a,b=(training_batches(examples,rep,arm) for arm in ('measured','shuffled'))
            self.assertEqual(len(a),144)
            for epoch in range(2):
                counts=Counter(r['sourceId'] for batch in a[72*epoch:72*(epoch+1)] for r in batch)
                self.assertEqual(len(counts),576);self.assertEqual(set(counts.values()),{1})
            self.assertEqual([[r['messages'] for r in batch] for batch in a],[[r['messages'] for r in batch] for batch in b])

    def test_changed_dataset_and_rehashed_bad_partition_rejected(self):
        examples,report=prepared_fixture();bad=copy.deepcopy(examples)
        bad['measured'][0]['target']='1' if bad['measured'][0]['target']=='0' else '0'
        with self.assertRaises(ValueError):validate_training(bad,report)
        bad=copy.deepcopy(examples);bad['measured'][0]['split']='test'
        changed=copy.deepcopy(report);changed['exampleHashes']['measured']=digest(bad['measured'])
        with self.assertRaises(ValueError):validate_training(bad,changed)


if __name__=='__main__':unittest.main()
