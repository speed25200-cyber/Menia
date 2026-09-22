import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from scipy.io import savemat
from scipy.special import expit

from research.human_reality_bridge import (design, evaluate, fit_logistic,
                                           read_subject, scale_audit)


class HumanBridgeTests(unittest.TestCase):
    def test_parser_mapping_exclusions_and_incomplete_timing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"fixture.zip"
            with zipfile.ZipFile(path, "w") as z:
                def add(name, values):
                    buffer = io.BytesIO()
                    savemat(buffer, values)
                    z.writestr(name, buffer.getvalue())
                add("RMs_S02.mat", {"responseMappings": [1, 2, 1, 2]})
                for run in range(1, 5):
                    response = np.zeros((8, 12, 4))
                    response[:, :, 0] = np.tile([1., 2., 3., 4.], (8, 3))
                    response[0, 0, 0] = np.nan
                    response[:, :, 2] = np.tile([0., 1.], (8, 6))
                    nblocks = 7 if run == 1 else 8
                    times = np.tile([1., 2., 3.], (nblocks, 12, 1))
                    add(f"MT_S02_run{run}.mat", {
                        "R": response, "trials": np.zeros((8, 12)),
                        "C": [1, 0, 1, 1], "blocks": [1, 1, 1, 1],
                        "miniblocks": [1, 2, 1, 2, 1, 2, 1, 2],
                        "T": {"presTimes": times, "starttime": 0.}})
            rows, excluded = read_subject(path, 2)
            self.assertEqual(len(rows), 272)  # 384 - 96 check failures - 4 NaNs - 12 untimed
            self.assertEqual(rows[rows[:, 0] == 1][0, 4], 2)
            self.assertEqual(rows[rows[:, 0] == 2][0, 4], 3)  # reversed key mapping
            self.assertEqual(sum(e["excluded_union"] for e in excluded), 112)
            self.assertEqual(sum(e["trailing_rows_without_timing"] for e in excluded), 12)
            self.assertEqual(sum(e["reality_screen_first_retained"] for e in excluded), 272)

    def test_penalized_fit_satisfies_independent_stationarity_check(self):
        rng = np.random.default_rng(71)
        x = np.column_stack((np.ones(400), rng.normal(size=(400, 4))))
        y = (rng.random(400) < expit(x @ np.array([.2, .8, -.4, 1., -.2]))).astype(float)
        beta = fit_logistic(x, y)
        direction = np.array([1., -.2, .4, -.5, .8])
        def loss(b):
            p = expit(x @ b)
            return -(y*np.log(p)+(1-y)*np.log1p(-p)).sum()+.5*np.sum(b[1:]**2)
        finite_difference = (loss(beta+1e-5*direction)-loss(beta-1e-5*direction))/2e-5
        self.assertLess(abs(finite_difference), 1e-5)
        self.assertLess(loss(beta), loss(np.zeros(5)))

    def test_each_fit_excludes_entire_held_out_run(self):
        rows = np.array([[r, p, c, int((r+p+c) % 2), 1+p+c]
                         for r in range(1, 5) for p in (0, 1) for c in (0, 1)], dtype=float)
        calls = []
        def fitting(x, y):
            calls.append((x.copy(), y.copy()))
            return np.zeros(x.shape[1])
        with patch("research.human_reality_bridge.fit_logistic", side_effect=fitting):
            result = evaluate(2, rows, [])
        self.assertEqual(len(calls), 12)
        for model_index, model in enumerate(("condition", "vividness", "interaction")):
            for run in range(1, 5):
                x, y = calls[model_index*4+run-1]
                selected = rows[:, 0] != run
                np.testing.assert_array_equal(x, design(rows[selected], model))
                np.testing.assert_array_equal(y, rows[selected, 3])
        self.assertAlmostEqual(result["scores"]["interaction"]["logloss"], np.log(2))

    def test_scale_transform_preserves_binary_and_ordinal_probabilities(self):
        rows = np.array([[1, p, c, 0, 2] for p in (0, 1) for c in (0, 1)], dtype=float)
        report = scale_audit(rows)
        self.assertEqual(len(report["checks"]), 4)
        self.assertLess(max(r["maximum_probability_difference"] for r in report["checks"]), 1e-12)


if __name__ == "__main__":
    unittest.main()
