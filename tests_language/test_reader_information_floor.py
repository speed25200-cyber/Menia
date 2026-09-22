import math
import unittest

from research.reader_information_floor import information_floor


class InformationFloorTests(unittest.TestCase):
    def test_hidden_contradiction_has_exact_variance_entropy_and_error_floor(self):
        rows=[('same-input',1),('same-input',1),('same-input',0)]
        r=information_floor(rows)
        self.assertAlmostEqual(r['minimumConditionalBrier'],2/9)
        self.assertAlmostEqual(r['minimumConditionalCodeCrossEntropyNats'],-(2/3*math.log(2/3)+1/3*math.log(1/3)))
        self.assertAlmostEqual(r['codeEOSCrossEntropyLowerBoundNats'],r['minimumConditionalCodeCrossEntropyNats']/2)
        self.assertEqual(r['minimumNativeErrors'],1)
        self.assertAlmostEqual(r['maximumNativeAccuracy'],2/3)
        self.assertEqual(r['conflictingInputs'],1)

    def test_distinguishable_states_remove_only_the_information_floor(self):
        hidden=information_floor([(('prompt','cache1'),1),(('prompt','cache2'),0)])
        self.assertEqual(hidden['minimumConditionalBrier'],0)
        self.assertEqual(hidden['minimumNativeErrors'],0)
        self.assertEqual(hidden['distinctInputs'],2)
        for label in (0,1):
            pure=information_floor([('same-input',label)]*10)
            self.assertEqual(pure['minimumConditionalCodeCrossEntropyNats'],0)

    def test_repetition_and_code_inversion_preserve_normalized_floor(self):
        rows=[('x',0),('x',1),('x',1),('y',0)]
        original=information_floor(rows);repeated=information_floor(rows*4)
        inverted=information_floor([(i,1-y) for i,y in rows])
        for key in ('minimumConditionalBrier','minimumConditionalCodeCrossEntropyNats','maximumNativeAccuracy'):
            self.assertAlmostEqual(original[key],repeated[key]);self.assertAlmostEqual(original[key],inverted[key])
        self.assertEqual(repeated['minimumNativeErrors'],original['minimumNativeErrors']*4)

    def test_empirical_probability_minimizes_loss_without_claiming_generalization(self):
        labels=[0,0,1,1,1];r=information_floor([('same',y) for y in labels])
        for i in range(101):
            probability=i/100
            loss=sum((probability-y)**2 for y in labels)/len(labels)
            self.assertGreaterEqual(loss+1e-14,r['minimumConditionalBrier'])
        self.assertAlmostEqual(sum((.6-y)**2 for y in labels)/len(labels),r['minimumConditionalBrier'])


if __name__=='__main__':unittest.main()
