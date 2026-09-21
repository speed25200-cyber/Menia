import copy
import json
import unittest

from research.iphone_learning_replication import analyze, supplement, within_family
from tests_research.test_iphone_capability_learning_report import fixture, truncate


def collection():
    base = fixture()
    base["audits"][0]["created"] = 1000.
    combined = copy.deepcopy(base)
    for n in range(1, 5):
        a = copy.deepcopy(base["audits"][0])
        a.update(id=f"PRIVATE_REPLICA_{n}", created=1000.+n)
        combined["audits"].append(a)
    return combined, base


class LearningReplicationTests(unittest.TestCase):
    def test_first_three_attempts_selected_by_creation_not_export_order(self):
        c, b = collection()
        c["audits"].reverse()
        r = analyze(c, b)
        self.assertEqual([x["role"] for x in r["runs"]],
                         ["initial_exploratory"]+["prespecified_replication"]*3+["additional_exploratory"])
        self.assertTrue(r["allThreeCompleteAndCompatible"])
        self.assertFalse(r["baselineUsedInReplicationMean"])
        self.assertEqual(r["equalRunComparison"][0]["runsWithLowerBrier"], 0)
        self.assertNotIn("PRIVATE_", json.dumps(r))
        self.assertNotIn('"created"', json.dumps(r))

    def test_interruption_occupies_slot_and_fourth_cannot_replace_it(self):
        c, b = collection()
        interrupted = truncate(dict(schema=c["schema"], audits=[c["audits"][1]]), 27)
        c["audits"][1] = interrupted["audits"][0]
        r = analyze(c, b)
        self.assertEqual(r["receivedFollowupSlots"], 3)
        self.assertFalse(r["allThreeCompleteAndCompatible"])
        self.assertIsNone(r["equalRunComparison"])
        self.assertEqual(r["runs"][1]["supplementary"]["evaluated"], 0)
        self.assertEqual(r["runs"][4]["role"], "additional_exploratory")

    def test_initial_run_alone_is_not_a_replication(self):
        c, b = collection()
        r = analyze(b, b)
        self.assertEqual(r["missingFollowupSlots"], 3)
        self.assertIsNone(r["equalRunComparison"])

    def test_changed_or_missing_original_and_ambiguous_order_rejected(self):
        for kind in ("changed", "missing", "time"):
            c, b = collection()
            if kind == "changed": c["audits"][0]["appVersion"] = "different"
            if kind == "missing": c["audits"].pop(0)
            if kind == "time": c["audits"][1]["created"] = c["audits"][0]["created"]
            with self.assertRaises(ValueError, msg=kind):
                analyze(c, b)

    def test_configuration_deviation_is_retained_without_confirmatory_mean(self):
        c, b = collection(); c["audits"][2]["model"]["fingerprint"] = "different"
        r = analyze(c, b)
        self.assertFalse(r["runs"][2]["sameModelAppAndSettings"])
        self.assertIsNone(r["equalRunComparison"])
        self.assertEqual(len(r["runs"]), 5)

    def test_frequency_estimates_use_calibration_not_heldout_outcomes(self):
        a = fixture()["audits"][0]
        r = supplement(a)
        self.assertEqual(r["calibrationRates"], dict(addition=1., multiplication=0., countA=1., alternatingSum=1.))
        self.assertAlmostEqual(r["empiricalFamily"]["systemBrier"], .5)
        self.assertEqual(r["empiricalFamily"]["directErrors"], 9)

    def test_auc_requires_within_family_variation_and_preserves_ties(self):
        rows = [dict(family="addition", correct=y, forecasts={c:p for c in ("relevant", "absent", "otherFamily")})
                for y,p in ((True,.9),(False,.1),(True,.1))]
        r = within_family(rows, "addition")
        self.assertTrue(r["mixedOutcomes"])
        self.assertEqual(r["auc"]["relevant"], .75)
        self.assertIsNone(within_family(rows[:1], "addition")["auc"]["relevant"])
        rows[0]["forecasts"]["relevant"] = None
        self.assertIsNone(within_family(rows, "addition")["auc"]["relevant"])


if __name__ == "__main__":
    unittest.main()
