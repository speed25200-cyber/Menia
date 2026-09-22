import itertools
import unittest

from research.cross_task_hypotheses import predictions


class CrossTaskHypothesisTests(unittest.TestCase):
    def test_all_four_predictions_are_separable_in_each_direction(self):
        for dt, rt in (('monitor', 'marker_first'), ('marker_first', 'monitor')):
            rows = [predictions(dt, rt, *bits) for bits in itertools.product((0, 1), repeat=6)]
            for a, b in itertools.combinations(rows[0], 2):
                self.assertEqual(sum(row[a] != row[b] for row in rows), 32, (dt, a, b))
        # Hidden donor bit differs from public donor bit, while both codes differ.
        self.assertEqual(predictions('monitor', 'marker_first', 1, 0, 0, 1, 0, 1),
                         dict(recipientTaskContent=1, donorBoolean=0, donorAnswer=1, recipientUnchanged=0))
        self.assertEqual(predictions('marker_first', 'monitor', 1, 0, 0, 1, 0, 1),
                         dict(recipientTaskContent=0, donorBoolean=1, donorAnswer=0, recipientUnchanged=1))

    def test_same_task_cannot_separate_content_from_donor_boolean(self):
        for task in ('monitor', 'marker_first'):
            for bits in itertools.product((0, 1), repeat=6):
                row = predictions(task, task, *bits)
                self.assertEqual(row['recipientTaskContent'], row['donorBoolean'])
        with self.assertRaises(ValueError):
            predictions('unknown', 'monitor', 0, 0, 0, 0, 0, 0)
        with self.assertRaises(ValueError):
            predictions('monitor', 'monitor', True, 0, 0, 0, 0, 0)
