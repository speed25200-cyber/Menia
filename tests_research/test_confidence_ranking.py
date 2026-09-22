from collections import Counter
import copy
import unittest

from research.confidence_ranking import paired_batches, preparation_report
from research.cross_model_prediction import CELLS, digest
from tests_research.test_answer_confidence_plan import prepared_fixture


class ConfidenceRankingScheduleTests(unittest.TestCase):
    def test_each_example_once_per_epoch_within_cell_and_no_reweighting(self):
        rows = prepared_fixture()[0]['measured']; before = digest(rows)
        for rep in range(3):
            batches = paired_batches(rows, rep)
            for epoch in range(2):
                section = batches[72*epoch:72*(epoch+1)]
                ids = Counter(r['sourceId'] for b in section for p in b for r in (p['left'], p['right']))
                expected = Counter(r['sourceId'] for r in rows if r['replication'] == rep)
                self.assertEqual(ids, expected)
                self.assertTrue(all(len(b) == 4 for b in section))
                for b in section:
                    for p in b:
                        a, z = p['left'], p['right']
                        self.assertEqual((a['family'], a['level']), (z['family'], z['level']))
                        self.assertEqual(p['eligible'], a['target'] != z['target'])
                self.assertEqual(sum(p['eligible'] for b in section for p in b), 6*32)
            self.assertEqual(batches, paired_batches(list(reversed(rows)), rep))
        self.assertEqual(digest(rows), before)

    def test_zero_and_one_success_do_not_get_oversampled_or_dropped(self):
        rows = prepared_fixture()[0]['measured']
        for cell, positives in zip(CELLS[:2], (0, 1)):
            group = [r for r in rows if r['replication'] == 0 and (r['family'], r['level']) == cell]
            for index, row in enumerate(group):
                row['target'] = str(int(index < positives))
        report = preparation_report(rows)
        for cell, positives in zip(CELLS[:2], (0, 1)):
            g = report['groups'][f'r0/{cell[0]}/{cell[1]}']
            self.assertEqual(g['eligiblePairsPerEpoch'], positives)
            self.assertTrue(g['rareClass'])
        self.assertEqual(sum(len(b) for b in paired_batches(rows, 0)), 576)

    def test_heldout_duplicate_or_invalid_partition_rejected(self):
        rows = prepared_fixture()[0]['measured']
        for key, value in (('split', 'test'), ('replication', 3), ('target', '2')):
            bad = copy.deepcopy(rows); bad[0][key] = value
            with self.assertRaises(ValueError): paired_batches(bad, 0)
        bad = copy.deepcopy(rows); bad[0] = copy.deepcopy(bad[1])
        with self.assertRaises(ValueError): paired_batches(bad, 0)


if __name__ == '__main__':
    unittest.main()
