import unittest
from research.interchange_hypotheses import factorial,predictions


class InterchangeHypothesisTests(unittest.TestCase):
    def test_opposite_only_design_confounds_copy_with_no_change(self):
        rows=[r for r in factorial() if r['donorState']!=r['recipientState'] and r['donorCode']!=r['recipientCode']]
        self.assertEqual(len(rows),4)
        for r in rows:
            p=r['predictions']
            self.assertEqual(p['donorAnswer'],p['recipientUnchanged'])
            self.assertNotEqual(p['stateTransfer'],p['recipientUnchanged'])

    def test_complete_factorial_separates_each_pair(self):
        rows=factorial();self.assertEqual(len(rows),16)
        for a,b in (('stateTransfer','donorAnswer'),('stateTransfer','recipientUnchanged'),
                    ('donorAnswer','recipientUnchanged')):
            self.assertEqual(sum(r['predictions'][a]!=r['predictions'][b] for r in rows),8)
        with self.assertRaises(ValueError):predictions(True,0,0,0)


if __name__=='__main__':unittest.main()
