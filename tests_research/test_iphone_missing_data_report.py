import copy
import json
import unittest

from research.iphone_missing_data_report import audit_collection, grade, INSTRUCTIONS, REVISED, SETTINGS


def fixture():
    conditions = ("originalOmitted", "revisedOmitted", "revisedNull", "providedHigh", "providedLow", "providedZero")
    trials = []
    for block, row in enumerate((2, 0, 4, 5, 1, 3)):
        n = (7, 11, 17, 23, 31, 47)[row]
        for role, name in (("otherAgent", "un agent de référence."), ("selfReport", "toi.")):
            for pos in (0, 1, 5, 2, 4, 3):
                index = (pos + row) % 6
                condition = conditions[index]
                provided = None
                if index == 3:
                    provided = dict(observations=n, successes=n, predictedSuccess=(n+1)/(n+2))
                elif index == 4:
                    provided = dict(observations=n, successes=0, predictedSuccess=1/(n+2))
                elif index == 5:
                    provided = dict(observations=0, successes=0, predictedSuccess=.5)
                payload = dict(question="Restitue le bilan de calcul qui concerne " + name + " Indique les observations, les réussites, la prévision fournie et le choix demandé.")
                if index >= 2:
                    payload["bilan"] = provided
                action = "mesurer" if index < 3 else "repondre" if index == 3 else "verifier"
                reply = dict(**(provided or dict(observations=None, successes=None, predictedSuccess=None)), action=action)
                trials.append(dict(id=f"PRIVATE_TRIAL_{len(trials)}", block=block, designRow=row, referent=role,
                                   condition=condition, providedSummary=provided,
                                   request=dict(instructions=INSTRUCTIONS if index == 0 else REVISED, prompt=json.dumps(payload)),
                                   answer=json.dumps(reply), durationSeconds=2., firstTextSeconds=.5, status="completed",
                                   grade=dict(validFormat=True, numbersCorrect=True, actionCorrect=True, completeSuccess=True, reportedAction=action)))
    audit = dict(schema="menia-iphone-missing-data-v1", id="PRIVATE_AUDIT", appVersion="fixture",
                 dataOrigin="synthetic protocol fixtures; not observed device capabilities",
                 generationSettings=SETTINGS, model=dict(fingerprint="fixture", bytes=10), trials=trials)
    return dict(schema="menia-iphone-missing-data-collection-v1", audits=[audit])


class IPhoneMissingDataReportTests(unittest.TestCase):
    def test_complete_fixture_has_known_counts_and_no_private_identifiers(self):
        result = audit_collection(fixture())
        self.assertEqual(result["totals"]["completeSuccess"], 72)
        self.assertTrue(result["audits"][0]["storedGradesMatch"])
        self.assertEqual(len(result["audits"][0]["cells"]), 12)
        self.assertTrue(all(c["planned"] == 6 for c in result["audits"][0]["cells"]))
        self.assertTrue(all(c["completedPairs"] == 6 and c["meanJointChange"] == 0 for c in result["audits"][0]["contrasts"]))
        self.assertNotIn("PRIVATE_", json.dumps(result))
        self.assertFalse(result["deviceExecutionAttested"])

    def test_raw_answers_override_forged_scores_and_produce_expected_paired_effect(self):
        data = fixture()
        for t in data["audits"][0]["trials"]:
            if t["condition"] == "originalOmitted":
                t["answer"] = '{"observations":12,"successes":8,"predictedSuccess":0.75,"action":"verifier"}'
        report = audit_collection(data)["audits"][0]
        self.assertEqual(report["totals"]["validFormat"], 72)
        self.assertEqual(report["totals"]["completeSuccess"], 60)
        self.assertEqual(len(report["gradeDiscrepancyTrials"]), 12)
        for contrast in report["contrasts"]:
            self.assertEqual(contrast["meanJointChange"], 1 if contrast["contrast"] == "B-A" else 0)
            self.assertEqual(contrast["meanNumbersChange"], 1 if contrast["contrast"] == "B-A" else 0)

    def test_zero_observations_and_abstention_have_different_expected_answers(self):
        unknown = '{"observations":null,"successes":null,"predictedSuccess":null,"action":"mesurer"}'
        self.assertTrue(grade(unknown, None)["completeSuccess"])
        zero = dict(observations=0, successes=0, predictedSuccess=.5)
        result = grade(unknown, zero)
        self.assertTrue(result["validFormat"])
        self.assertFalse(result["numbersCorrect"])
        self.assertFalse(result["actionCorrect"])

    def test_changed_null_encoding_prompt_row_or_timing_is_rejected(self):
        changes = [lambda a: a["trials"][0].update(designRow=1),
                   lambda a: a["trials"][0].update(firstTextSeconds=3),
                   lambda a: a.update(dataOrigin="actual measured capabilities"),
                   lambda a: a["trials"][0]["request"].update(instructions=INSTRUCTIONS),
                   lambda a: a["trials"][0]["request"].update(prompt=json.dumps({"question": json.loads(a["trials"][0]["request"]["prompt"])["question"]})),
                   lambda a: a["trials"][1].update(id=a["trials"][0]["id"])]
        for i, change in enumerate(changes):
            with self.subTest(change=i):
                data = fixture()
                change(data["audits"][0])
                with self.assertRaises(ValueError):
                    audit_collection(data)

    def test_cancelled_and_unrun_trials_are_kept_without_invented_pairs(self):
        data = fixture()
        trials = data["audits"][0]["trials"]
        trials[0].update(status="cancelled", answer="partial", failure="PRIVATE_ERROR", grade=None)
        for t in trials[1:]:
            t["status"] = "planned"
            for key in ("answer", "durationSeconds", "firstTextSeconds", "grade"):
                t.pop(key)
        result = audit_collection(data)
        self.assertEqual(result["totals"]["planned"], 72)
        self.assertEqual(result["totals"]["completed"], 0)
        self.assertTrue(all(c["completedPairs"] == 0 and c["meanJointChange"] is None for c in result["audits"][0]["contrasts"]))
        self.assertNotIn("PRIVATE_", json.dumps(result))
        second = copy.deepcopy(data["audits"][0])
        second["id"] = "ANOTHER_PRIVATE_AUDIT"
        data["audits"].append(second)
        self.assertEqual(audit_collection(data)["totals"]["planned"], 144)
        second["id"] = data["audits"][0]["id"]
        with self.assertRaises(ValueError):
            audit_collection(data)
