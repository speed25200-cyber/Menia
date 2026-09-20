import unittest

import torch

from research.audit_confidence_budget import weight_difference


class BudgetWeightAuditTests(unittest.TestCase):
    def test_displacement_matches_known_euclidean_distance(self):
        old = dict(a=torch.tensor([[1.,2.]]),b=torch.tensor([[3.,4.]]))
        new = dict(a=torch.tensor([[4.,6.]]),b=old['b'].clone())
        self.assertEqual(weight_difference(old,new),dict(changedTensors=1,maximumAbsoluteDifference=4.,l2=5.))
        self.assertEqual(weight_difference(old,old),dict(changedTensors=0,maximumAbsoluteDifference=0.,l2=0.))

    def test_keys_shapes_precision_and_nonfinite_tensors_are_rejected(self):
        old = dict(a=torch.tensor([[1.,2.]]))
        for new in (dict(b=old['a']),dict(a=old['a'].flatten()),dict(a=old['a'].double()),
                    dict(a=torch.tensor([[float('nan'),2.]])),dict(a=torch.tensor([[float('inf'),2.]]))):
            with self.subTest(new=new):
                with self.assertRaises(ValueError): weight_difference(old,new)


if __name__ == '__main__': unittest.main()
