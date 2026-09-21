import math
import unittest
from research.optimization_loss_diagnostic import final_cross_entropy


class LossReconstructionTests(unittest.TestCase):
    def test_recovers_full_vocabulary_mass_not_pair_conditioned_loss(self):
        # Probabilities .1, .2, .3, .1, .3; only the first four logits are retained.
        r=dict(choiceLogits=[math.log(x) for x in (.1,.2,.3,.1)],choiceMass=.7)
        for i,p in enumerate((.1,.2,.3,.1)):
            self.assertAlmostEqual(final_cross_entropy(r,i),-math.log(p))
        with self.assertRaises(ValueError):final_cross_entropy(dict(r,choiceMass=0.),0)


if __name__=='__main__':unittest.main()
