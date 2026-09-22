import unittest
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from research.audit_presence_causality import index_rows, mediation, sensitivity, equivalent_models, bootstrap, raw_numeric_cell


class PresenceCausalityTests(unittest.TestCase):
    def fixture(self):
        # Balanced orthogonal factorial noise gives exact population moments.
        x, u, v = np.array(np.meshgrid([0., 1.], [-1., 1.], [-1., 1.])).reshape(3, -1)
        m = 1+2*x+u
        y = 4-.5*x+.7*m+v
        return x, m, y

    def test_known_paths_and_sensitivity_recover_a_confounded_structural_effect(self):
        x, m, y = self.fixture()
        # Add shared noise: true b=.7; OLS b=1.1, rho=.4/sqrt(1+.4**2).
        u = m-1-2*x
        result = mediation(x, m, y+.4*u)
        self.assertAlmostEqual(result["b"], 1.1)
        self.assertAlmostEqual(result["total"], .9)
        effect = sensitivity(result["vm"], result["vy"], result["covariance"],
                             result["a"], .4/np.sqrt(1.16))
        self.assertAlmostEqual(effect, 1.4)
        with self.assertRaises(ValueError):
            sensitivity(1., 1., 0., 1., 1.)

    def test_reverse_factorization_has_same_gaussian_moments_and_different_intervention(self):
        result = equivalent_models(mediation(*self.fixture()))
        np.testing.assert_allclose(result["forward_covariance"], result["reverse_covariance"], atol=1e-12)
        self.assertLess(result["max_reconstruction_error"], 1e-12)
        self.assertAlmostEqual(result["effect_do_m_on_y_forward"], .7)
        self.assertEqual(result["effect_do_m_on_y_reverse"], 0)

    def test_row_keys_do_not_depend_on_order_and_duplicates_fail(self):
        rows = [("Participant", "value"), (3, 7), (None, None), (1, 8)]
        reordered = [rows[0], rows[3], rows[1]]
        self.assertEqual(index_rows(rows), index_rows(reordered))
        with self.assertRaises(ValueError):
            index_rows(rows+[(3, 9)])

    def test_numeric_source_read_ignores_format_but_rejects_formula_and_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.xlsx"
            for cell, valid in (( '<c r="D23" s="44"><v>112.06100000000001</v></c>', True),
                                ('<c r="D23"><f>A1+1</f><v>112</v></c>', False),
                                ('<c r="D23" t="s"><v>112</v></c>', False)):
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("xl/worksheets/sheet7.xml", '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row>'+cell+'</row></sheetData></worksheet>')
                if valid:
                    self.assertEqual(raw_numeric_cell(path, "xl/worksheets/sheet7.xml", "D23"), 112.06100000000001)
                else:
                    with self.assertRaises(ValueError):
                        raw_numeric_cell(path, "xl/worksheets/sheet7.xml", "D23")

    def test_bootstrap_reproducibility_and_baseline_covariate(self):
        rng = np.random.default_rng(4)
        x = np.repeat([0., 1.], [19, 21])
        baseline = rng.normal(size=40)
        m = 2*x+.3*baseline+rng.normal(size=40)
        y = .7*m+.4*baseline+rng.normal(size=40)
        a = bootstrap(x, m, y, baseline, 7, 80)
        b = bootstrap(x, m, y, baseline, 7, 80)
        self.assertEqual(a, b)
        self.assertEqual(a["stratum_sizes"], [19, 21])
        self.assertEqual(len(mediation(x, m, y, baseline)["mean_coefficients_m"]), 3)
        with self.assertRaises(ValueError):
            mediation(x, m, y, baseline[:-1])


if __name__ == "__main__":
    unittest.main()
