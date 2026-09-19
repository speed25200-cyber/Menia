"""Independent checks for the conditional null distribution and multiplicity."""

from fractions import Fraction
from itertools import product
import unittest

from research.audit_felt_self import benjamini_hochberg, exact_signed_rank


def brute_force_p(differences):
    magnitudes = [abs(d) for d in differences if d != 0]
    signed = [d for d in differences if d != 0]
    # Pairwise midrank formula, independent of sorting/grouping in the audit.
    weights = [1 + 2 * sum(y < x for y in magnitudes) + sum(y == x for y in magnitudes)
               for x in magnitudes]
    observed = abs(sum(w * (1 if d > 0 else -1) for w, d in zip(weights, signed)))
    more_extreme = sum(abs(sum(w * s for w, s in zip(weights, signs))) >= observed
                       for signs in product((-1, 1), repeat=len(weights)))
    return Fraction(more_extreme, 2 ** len(weights))


class FeltSelfStatisticsTests(unittest.TestCase):
    def test_exact_distribution_against_exhaustive_signs(self):
        for n in range(5):
            for values in product((-2, -1, 0, 1, 2), repeat=n):
                actual = Fraction(exact_signed_rank(values)["p_exact"])
                self.assertEqual(actual, brute_force_p(values), values)

    def test_all_positive_ties_and_sign_reversal(self):
        self.assertEqual(Fraction(exact_signed_rank([3] * 20)["p_exact"]), Fraction(1, 2 ** 19))
        values = [-1, 1, 2, 2, -3, 0, 5]
        self.assertEqual(exact_signed_rank(values)["p_exact"],
                         exact_signed_rank([-v for v in values])["p_exact"])

    def test_zero_removal_and_scale_invariance(self):
        values = [-1, 2, -3, 4]
        self.assertEqual(exact_signed_rank(values), exact_signed_rank([10 * v for v in values] + [0, 0]))
        self.assertEqual(exact_signed_rank([0, 0])["p"], 1)

    def test_bh_known_values_permutation_and_ties(self):
        values = [0.9, 0.001, 0.04, 0.01, 0.2, 0.02]
        expected = [0.9, 0.006, 0.06, 0.03, 0.24, 0.04]
        for a, b in zip(benjamini_hochberg(values), expected):
            self.assertAlmostEqual(a, b)
        self.assertEqual(benjamini_hochberg([0.01, 0.01, 0.01]), [0.01, 0.01, 0.01])
        self.assertEqual(benjamini_hochberg([1, 1]), [1, 1])


if __name__ == "__main__":
    unittest.main()
