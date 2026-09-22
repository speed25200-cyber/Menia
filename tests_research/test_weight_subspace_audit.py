"""Independent dense references and known counterexamples for factor geometry."""
import unittest

import numpy as np

from research.weight_subspace_audit import compare, compare_states, describe, inner, pair_modules, spectrum, synthetic_audit


class WeightSubspaceTests(unittest.TestCase):
    def test_low_rank_algebra_matches_dense_updates_including_different_ranks(self):
        rng = np.random.default_rng(41)
        for m, n, r, t in ((7, 9, 3, 2), (2, 3, 5, 4), (1, 5, 2, 1)):
            a, b = rng.normal(size=(r, n)), rng.normal(size=(m, r))
            c, d = rng.normal(size=(t, n)), rng.normal(size=(m, t))
            x, y = b @ a, d @ c
            actual = spectrum(a, b)
            reference = np.linalg.svd(x, compute_uv=False)
            np.testing.assert_allclose(actual, reference[:len(actual)], atol=1e-12)
            np.testing.assert_allclose(reference[len(actual):], 0, atol=1e-12)
            self.assertAlmostEqual(inner(a, b, c, d), float(np.sum(x * y)), places=10)
            self.assertAlmostEqual(compare(a, b, c, d)["updateCosine"], float(np.sum(x*y) / np.linalg.norm(x) / np.linalg.norm(y)), places=12)

    def test_nonorthogonal_gauge_does_not_change_update_geometry(self):
        rng = np.random.default_rng(7)
        a, b = rng.normal(size=(3, 8)), rng.normal(size=(6, 3))
        q = np.array([[2., 1., 0.], [0., .5, 1.], [0., 0., 3.]])
        c, d = q @ a, b @ np.linalg.inv(q)
        result = compare(a, b, c, d)
        self.assertAlmostEqual(result["updateCosine"], 1., places=12)
        np.testing.assert_allclose(spectrum(a, b), spectrum(c, d), rtol=1e-12)
        self.assertLess(result["parameterizationDependentFactorCosines"]["a"], .99)

    def test_same_spectrum_can_have_opposite_effective_update(self):
        a = np.eye(3)
        result = compare(a, a, a, -a)
        self.assertEqual(result["left"]["singularValues"], result["right"]["singularValues"])
        self.assertEqual(result["updateCosine"], -1.)

    def test_zero_updates_are_undefined_not_perfect_alignment(self):
        result = compare(np.eye(2), np.zeros((3, 2)), np.eye(2), np.ones((3, 2)))
        self.assertIsNone(result["updateCosine"])
        self.assertIsNone(result["left"]["topOneEnergyFraction"])
        self.assertEqual(result["left"]["rankFor95PercentEnergy"], 0)

    def test_global_cosine_uses_updates_not_mean_of_module_cosines(self):
        a = np.eye(2)
        left = {"q.a": a, "q.b": 3*a, "v.a": a, "v.b": a}
        right = {"q.a": a, "q.b": 3*a, "v.a": a, "v.b": -a}
        result = compare_states(left, right)
        self.assertAlmostEqual(result["globalUpdateCosine"], .8)
        self.assertFalse(result["behaviorMeasured"])
        self.assertFalse(result["provenanceVerified"])

    def test_invalid_checkpoint_or_shapes_are_rejected(self):
        eye = np.eye(2)
        for state in ({}, {"q.a": eye}, {"q.weight": eye}, {".a": eye, ".b": eye},
                      {"q.a": eye, "q.b": np.ones((3, 4))},
                      {"q.a": eye, "q.b": np.full((2, 2), np.nan)}):
            with self.assertRaises(ValueError):
                pair_modules(state)
        for a in (np.ones((0, 2)), np.ones((2,)), np.array([["bad"]]), np.eye(2) * 1j):
            with self.assertRaises(ValueError):
                describe(a, eye)
        with self.assertRaises(ValueError):
            compare(eye, eye, np.ones((2, 3)), eye)
        with self.assertRaises(ValueError):
            compare_states({"q.a": eye, "q.b": eye}, {"v.a": eye, "v.b": eye})

    def test_counterexamples_have_analytically_known_results(self):
        report = synthetic_audit()
        common = report["commonComponent"]
        self.assertAlmostEqual(common["uncenteredTopOneEnergyFraction"], 100/101)
        self.assertAlmostEqual(common["centeredTopOneEnergyFraction"], 1/8)
        self.assertEqual(common["fullTaskAccuracy"], 1.)
        self.assertEqual(common["projectedTaskAccuracy"], .5)
        self.assertEqual(report["gaugeChange"]["parameterizationDependentFactorCosines"], {"a": 0., "b": 0.})
        self.assertAlmostEqual(report["gaugeChange"]["updateCosine"], 1., places=12)
        self.assertIsNone(report["sharedInitialization"]["updateCosine"])
        self.assertFalse(report["paperReplication"])


if __name__ == "__main__":
    unittest.main()
