from collections import Counter
import unittest

from research import natural_error_questions as study


class NaturalErrorQuestionTests(unittest.TestCase):
    def test_disjoint_from_previous_runs_technical_cases_and_every_partition(self):
        plan = study.make_plan()
        questions = [t['question'] for t in plan['tasks']]
        self.assertEqual(len(questions), 3456)
        self.assertEqual(len(set(questions)), len(questions))
        excluded = study.exclusions()
        self.assertFalse(set(questions) & excluded)
        self.assertEqual(plan['excludedQuestions'], len(excluded))
        self.assertEqual(plan['excludedQuestionsHash'], study.digest(sorted(excluded)))
        self.assertEqual(len({t['seed'] for t in plan['tasks']}), len(questions))
        self.assertEqual([t['id'] for t in plan['tasks']], list(range(len(questions))))

    def test_balanced_training_validation_test_and_repeatable_plan(self):
        plan = study.make_plan()
        self.assertEqual(plan, study.make_plan())
        counts = Counter((t['replication'], t['split'], t['family'], t['level']) for t in plan['tasks'])
        for rep in range(3):
            phase_order = {'train': 0, 'validation': 1, 'test': 2}
            rows = [t for t in plan['tasks'] if t['replication'] == rep]
            self.assertEqual([phase_order[t['split']] for t in rows], sorted(phase_order[t['split']] for t in rows))
            for split, expected in study.COUNTS.items():
                for family, level in study.CELLS:
                    self.assertEqual(counts[rep, split, family, level], expected)
        self.assertEqual(plan['model'], study.MODELS['A'])
        self.assertNotIn('answer', plan['tasks'][0])
        self.assertNotIn('correct', plan['tasks'][0])


if __name__ == '__main__':
    unittest.main()
