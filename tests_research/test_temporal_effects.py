import unittest
import numpy as np
from research.temporal_effects import dataset, features, fit_ridge, outcomes, predict, world_parameters


class TemporalEffectsTests(unittest.TestCase):
    def test_histories_are_aligned_without_future_samples(self):
        p = world_parameters(0, 'affine')
        history, _, mask = dataset(p, 'intervened')
        np.testing.assert_array_equal(history[1:, 1:11], history[:-1, :10])
        np.testing.assert_array_equal(history[1:, 12:22], history[:-1, 11:21])
        np.testing.assert_array_equal(history[:512, :11], history[:512, 11:])
        self.assertFalse(np.array_equal(history[512:, 0], history[512:, 11]))
        self.assertTrue((mask.sum(axis=1) <= 1).all())

    def test_predictor_cannot_access_out_of_window_action(self):
        p = world_parameters(0, 'beyond_window')
        first, _, _ = dataset(p, 'intervened', split='probe', size=8)
        second = first.copy()
        first[:, 10], second[:, 10] = 1, -1
        np.testing.assert_array_equal(features(first), features(second))
        self.assertGreater(np.linalg.norm(outcomes(first, p)-outcomes(second, p)), 0)

    def test_ridge_ignores_unobserved_targets(self):
        p = world_parameters(0, 'affine')
        h, y, mask = dataset(p, 'intervened')
        altered = y.copy()
        altered[~mask] = 1e12
        np.testing.assert_array_equal(fit_ridge(features(h), y, mask)['linear'],
                                      fit_ridge(features(h), altered, mask)['linear'])

    def test_independent_actions_recover_unknown_linear_delay(self):
        p = world_parameters(3, 'affine')
        h, y, mask = dataset(p, 'intervened')
        weights = fit_ridge(features(h), y, mask)
        first, _, _ = dataset(p, 'intervened', split='probe', size=32)
        second = first.copy()
        first[:, p['delay']], second[:, p['delay']] = 1, -1
        estimated = predict(weights, features(first))-predict(weights, features(second))
        truth = outcomes(first, p)-outcomes(second, p)
        self.assertLess(np.mean((estimated-truth)**2), .003)

    def test_quadratic_effect_requires_zero_reference(self):
        p = world_parameters(0, 'interaction')
        h = np.zeros((1, 22))
        positive, negative = h.copy(), h.copy()
        positive[:, p['delay']], negative[:, p['delay']] = 1, -1
        np.testing.assert_array_equal(outcomes(positive, p), outcomes(negative, p))
        self.assertGreater(np.linalg.norm(outcomes(positive, p)-outcomes(h, p)), 0)


if __name__ == '__main__':
    unittest.main()
