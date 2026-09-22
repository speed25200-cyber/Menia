import math
import unittest
import numpy as np
from research.causal_binding import CausalBinding, logit, window_probability
from research.audit_ownership_inference import binomial_nll, parse_counts, DELAYS, NOISE


class CausalBindingTests(unittest.TestCase):
    def test_response_probability_matches_independent_density_integration(self):
        model = CausalBinding(.73, 348.)
        sigma, stimulus = 120., 150.
        # Independently integrate the latent separate-source delay instead of
        # using the closed-form Gaussian Bayes factor from the implementation.
        latent = np.linspace(-8*348., 8*348., 20001)
        source_density = np.exp(-.5*(latent/348.)**2)/(348.*math.sqrt(2*math.pi))
        measurement_density = np.exp(-.5*((stimulus-latent)/sigma)**2)/(sigma*math.sqrt(2*math.pi))
        separate = np.trapezoid(source_density*measurement_density, latent)
        common = math.exp(-.5*(stimulus/sigma)**2)/(sigma*math.sqrt(2*math.pi))
        posterior = .73*common/(.73*common+.27*separate)
        self.assertAlmostEqual(model.posterior(stimulus, sigma), posterior, places=12)
        # Integrate the measurement density over the independently located
        # positive log posterior region; bisection does not use the k formula.
        lo, hi = 0., 5000.
        for _ in range(70):
            middle = (lo+hi)/2
            if model.posterior(middle, sigma) > .5:
                lo = middle
            else:
                hi = middle
        x = np.linspace(-hi, hi, 20001)
        density = np.exp(-.5*((x-stimulus)/sigma)**2)/(sigma*math.sqrt(2*math.pi))
        integral = np.trapezoid(density, x)
        self.assertAlmostEqual(model.report_probability(stimulus, sigma), integral, places=8)

    def test_prior_threshold_invariance_does_not_preserve_internal_posterior(self):
        a, b = CausalBinding(.9), CausalBinding(.5)
        for sigma in (50., 150., 500.):
            for stimulus in (-1000., -150., 0., 150., 1000.):
                self.assertAlmostEqual(a.report_probability(stimulus, sigma, .07),
                                       b.report_probability(stimulus, sigma, .07, -logit(.9)), places=14)
        self.assertGreater(abs(a.posterior(300, 150)-b.posterior(300, 150)), .2)

    def test_extremes_symmetry_and_invalid_inputs(self):
        model = CausalBinding(1e-12)
        self.assertEqual(model.report_probability(0., 150., .2), .1)
        self.assertEqual(model.report_probability(10000., 150., 1.), .5)
        self.assertGreater(window_probability(12., 1., 1.), 0.)
        self.assertEqual(window_probability(-12., 1., 1.), window_probability(12., 1., 1.))
        with self.assertRaises(ValueError):
            CausalBinding(1.)
        with self.assertRaises(ValueError):
            model.posterior(0., 0.)

    def test_likelihood_conserves_yes_and_no_counts_including_zero_boundaries(self):
        self.assertEqual(binomial_nll([0, 12], [0, 1]), 0.)
        self.assertTrue(math.isinf(binomial_nll([1], [0])))
        self.assertAlmostEqual(binomial_nll([0, 12, 6], [.5]*3), 36*math.log(2))
        with self.assertRaises(ValueError):
            binomial_nll([13], [.5])

    def test_counts_join_by_id_and_preserve_distinct_delay_grids(self):
        headers = ["Participant"]+[f"N{noise}_{delay}" for noise in NOISE for delay in DELAYS["ownership"]]
        rows = [headers, ["S2"]+[5]*21, ["S1"]+[2]*21, [None]*22]
        data = parse_counts(rows, "ownership")
        self.assertEqual(data[1].sum(), 42)
        self.assertEqual(data[2].sum(), 105)
        with self.assertRaises(ValueError):
            parse_counts(rows, "synchrony")
        with self.assertRaises(ValueError):
            parse_counts(rows+[rows[1]], "ownership")
        fractional = [headers, ["S1"]+[3.6]*21]
        with self.assertRaises(ValueError):
            parse_counts(fractional, "ownership")
        audited = parse_counts(fractional, "ownership", allow_noninteger=True)
        self.assertEqual(audited[1][0, 0], 3.6)
        with self.assertRaises(ValueError):
            binomial_nll(audited[1], np.full((3, 7), .5))


if __name__ == "__main__":
    unittest.main()
