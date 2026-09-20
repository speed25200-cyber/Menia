import math
import unittest

from research.confidence_ranking_training_diagnostic import margin_metrics


class TrainingMarginTests(unittest.TestCase):
    def test_ties_orientation_and_stable_extreme_losses(self):
        metrics = margin_metrics([-1000., -2., 0., 2., 1000.])
        self.assertEqual((metrics['correctlyOrdered'], metrics['tied'], metrics['reversed']), (2, 1, 2))
        self.assertEqual(metrics['orderingCredit'], .5)
        self.assertEqual(metrics['meanSignedMargin'], 0.)
        self.assertEqual(metrics['meanAbsoluteMargin'], 400.8)
        reference = (1000.+math.log(1+math.exp(2))+math.log(2)+math.log(1+math.exp(-2)))/5
        self.assertAlmostEqual(metrics['meanLogisticRankingLoss'], reference, places=12)
        self.assertEqual(margin_metrics([1., 2.])['orderingCredit'], 1.)
        self.assertEqual(margin_metrics([-1., -2.])['orderingCredit'], 0.)

    def test_missing_pairs_are_undefined_not_zero_accuracy(self):
        metrics = margin_metrics([])
        self.assertEqual(metrics['eligiblePairs'], 0)
        for field in ('orderingCredit', 'meanSignedMargin', 'meanAbsoluteMargin', 'meanLogisticRankingLoss'):
            self.assertIsNone(metrics[field])
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError): margin_metrics([bad])


if __name__ == '__main__': unittest.main()
