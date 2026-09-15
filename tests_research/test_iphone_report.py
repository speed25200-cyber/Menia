import copy
import unittest
from research.iphone_report import audit


def fixture():
    probes = []
    for i in range(5):
        probes.append(dict(id=str(i), modelID="fixture", question=f"Calcule {i} + 2. Réponds uniquement par l’entier obtenu, sans explication.",
                           answer=str(i + 2), expected=i + 2, correct=True, predictedSuccess=(i + 1)/(i + 2)))
    return dict(schema="menia-iphone-capability-v1", model=dict(fingerprint="fixture"), probes=probes,
                summary=dict(observations=5, successes=5, predictedSuccess=6/7,
                             brierScore=sum(1/(i+2)**2 for i in range(5))/5))


class IPhoneReportTests(unittest.TestCase):
    def test_independent_score_and_beta_quantiles(self):
        result = audit(fixture())
        self.assertAlmostEqual(result["prequentialBrier"], 1769/18000)
        lo, hi = result["betaEqualTail95UnderStationaryIID"]
        self.assertAlmostEqual(lo**6, .025)
        self.assertAlmostEqual(hi**6, .975)
        self.assertFalse(result["latencyOrTokensPerSecondMeasurable"])

    def test_tampered_reference_forecast_score_and_id_rejected(self):
        original = fixture()
        for section, key, value in [(0, "expected", 7), (1, "predictedSuccess", .99),
                                    (2, "correct", False), (3, "id", "0"), (4, "answer", "the answer is 6")]:
            changed = copy.deepcopy(original)
            changed["probes"][section][key] = value
            with self.assertRaises(ValueError): audit(changed)
        original["summary"]["brierScore"] = 0
        with self.assertRaises(ValueError): audit(original)

    def test_wrong_model_summary_does_not_pool_other_models(self):
        data = fixture()
        data["model"]["fingerprint"] = "other-model"
        with self.assertRaises(ValueError): audit(data)

    def test_non_finite_forecast_rejected(self):
        data = fixture()
        data["probes"][0]["predictedSuccess"] = float("nan")
        with self.assertRaises(ValueError): audit(data)
