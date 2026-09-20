import unittest

from research.prospective_learning_data import partition


class ProspectiveLearningDataTests(unittest.TestCase):
    def test_no_table_reuse_or_cross_split_overlap(self):
        p=partition(); cases=p['cases']
        self.assertEqual(p,partition())
        self.assertEqual(len(cases),48)
        self.assertEqual(len({c['tableHash'] for c in cases}),48)
        self.assertFalse({c['tableHash'] for c in cases}&set(p['exclusions']))
        self.assertEqual(sum(c['split']=='train' for c in cases),32)
        self.assertEqual(sum(c['split']=='reserved' for c in cases),16)
        self.assertTrue(all(sum(c['values'])==4 for c in cases))
        self.assertTrue(all('outcome' not in c and 'correct' not in c for c in cases))


if __name__=='__main__': unittest.main()
