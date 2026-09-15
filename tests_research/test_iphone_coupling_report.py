import copy
import itertools
import json
import unittest

from research.iphone_coupling_report import (
    CONDITIONS, INSTRUCTIONS, REFERENTS, SETTINGS, audit_collection, grade,
)


def fixture():
    measured = dict(observations=5, successes=5, predictedSuccess=6/7)
    fake = dict(observations=5, successes=0, predictedSuccess=1/7)
    probes = [dict(id=f"private-probe-{i}", modelID="fixture", question=f"Calcule {i} + 2. Réponds uniquement par l’entier obtenu, sans explication.",
                   answer=str(i + 2), expected=i + 2, correct=True, predictedSuccess=(i + 1)/(i + 2)) for i in range(5)]
    trials = []
    for block, order in enumerate(itertools.permutations(CONDITIONS)):
        for role in REFERENTS:
            for condition in order:
                provided = {"measured": measured, "absent": None, "fictional": fake}[condition]
                action = {"measured": "repondre", "absent": "mesurer", "fictional": "verifier"}[condition]
                answer = json.dumps(dict(**(provided or dict(observations=None, successes=None, predictedSuccess=None)), action=action))
                name = "toi." if role == "selfReport" else "un agent de référence."
                payload = dict(question="Restitue le bilan de calcul qui concerne " + name + " Indique les observations, les réussites, la prévision fournie et le choix demandé.")
                if provided is not None:
                    payload["bilan"] = provided
                stored = dict(validFormat=True, matchesProvidedSummary=True, matchesDecisionRule=True, reportedAction=action)
                if provided is not None:
                    stored["matchesMeasuredSummary"] = condition == "measured"
                trials.append(dict(id=f"private-trial-{len(trials)}", block=block, condition=condition,
                                   referent=role, providedSummary=copy.deepcopy(provided),
                                   request=dict(instructions=INSTRUCTIONS, prompt=json.dumps(payload)),
                                   status="completed", answer=answer, grade=stored,
                                   durationSeconds=2., firstTextSeconds=.5))
    audit = dict(schema="menia-iphone-coupling-v1", id="private-audit", appVersion="fixture",
                 model=dict(fingerprint="fixture", bytes=10), generationSettings=SETTINGS,
                 measuredSummary=measured, sourceProbes=probes, sourceProbeIDs=[p["id"] for p in probes], trials=trials)
    return dict(schema="menia-iphone-coupling-collection-v1", audits=[audit])


class IPhoneCouplingReportTests(unittest.TestCase):
    def test_known_counts_separate_context_fidelity_from_measurement_agreement(self):
        report = audit_collection(fixture())
        result = report["audits"][0]
        self.assertTrue(result["storedGradesMatch"])
        self.assertEqual(result["totals"]["matchesProvidedSummary"], 36)
        self.assertEqual(result["totals"]["agreesWithMeasuredSummary"], 12)
        self.assertEqual(result["totals"]["disagreesWithMeasuredSummary"], 12)
        self.assertEqual(result["totals"]["measuredAgreementUnavailable"], 12)
        self.assertTrue(all(c["planned"] == 6 for c in result["cells"]))
        self.assertFalse(report["deviceExecutionAttested"])
        self.assertNotIn("private-", json.dumps(report))

    def test_invented_absent_numbers_count_as_failure_despite_valid_json_and_forged_grade(self):
        data = fixture()
        t = next(t for t in data["audits"][0]["trials"] if t["condition"] == "absent")
        t["answer"] = '{"observations":9,"successes":4,"predictedSuccess":0.7,"action":"verifier"}'
        result = audit_collection(data)["audits"][0]
        self.assertFalse(result["storedGradesMatch"])
        self.assertEqual(result["totals"]["validFormat"], 36)
        self.assertEqual(result["totals"]["matchesProvidedSummary"], 35)
        self.assertEqual(result["totals"]["matchesDecisionRule"], 35)
        self.assertEqual(result["totals"]["disagreesWithMeasuredSummary"], 13)

    def test_strict_format_and_rounding_contract(self):
        measured = dict(observations=5, successes=5, predictedSuccess=6/7)
        valid = dict(observations=5.0, successes=5, predictedSuccess=.857, action="repondre")
        self.assertTrue(grade(json.dumps(valid), measured, measured)["matchesProvidedSummary"])
        invalid = ["```json\n" + json.dumps(valid) + "\n```", "[]", "{}",
                   '{"observations":5,"observations":6,"successes":5,"predictedSuccess":0.857,"action":"repondre"}']
        for key, value in [("observations", True), ("successes", -1), ("observations", 1.5),
                           ("predictedSuccess", float("nan")), ("predictedSuccess", 1.1),
                           ("action", "execute"), ("extra", 0)]:
            changed = dict(valid)
            changed[key] = value
            invalid.append(json.dumps(changed))
        for answer in invalid:
            with self.subTest(answer=answer):
                result = grade(answer, measured, measured)
                self.assertFalse(result["validFormat"])
                self.assertFalse(result["matchesProvidedSummary"])
        valid["predictedSuccess"] = .855
        result = grade(json.dumps(valid), measured, measured)
        self.assertTrue(result["validFormat"])
        self.assertFalse(result["matchesProvidedSummary"])
        self.assertTrue(result["matchesDecisionRule"])
        # The v1 format does not itself enforce successes <= observations.
        valid.update(observations=1, successes=5)
        self.assertTrue(grade(json.dumps(valid), measured, measured)["validFormat"])

    def test_plan_source_prompt_status_and_timing_tampering_rejected(self):
        changes = [lambda a: a["trials"].pop(),
                   lambda a: a["trials"][1].update(id=a["trials"][0]["id"]),
                   lambda a: a["trials"][0].update(block=2),
                   lambda a: a["sourceProbes"][0].update(expected=99),
                   lambda a: a["sourceProbes"][0].update(correct=False),
                   lambda a: a["sourceProbes"][0].update(modelID="another"),
                   lambda a: a["measuredSummary"].update(predictedSuccess=.9),
                   lambda a: a["trials"][2]["providedSummary"].update(successes=False),
                   lambda a: a["trials"][0]["request"].update(instructions="Changed instruction"),
                   lambda a: a["trials"][0]["request"].update(prompt='{"question":"altered"}'),
                   lambda a: a.update(generationSettings="temperature=0"),
                   lambda a: a["trials"][0].update(status="cancelled", grade=None),
                   lambda a: a["trials"][0].update(firstTextSeconds=3),
                   lambda a: a["trials"][0].update(durationSeconds=float("nan"))]
        for i, change in enumerate(changes):
            with self.subTest(change=i):
                data = fixture()
                change(data["audits"][0])
                with self.assertRaises(ValueError):
                    audit_collection(data)

    def test_partial_and_failed_runs_remain_in_planned_denominator(self):
        data = fixture()
        trials = data["audits"][0]["trials"]
        trials[1].update(status="error", failure="private-runtime-error", grade=None, answer="partial")
        for t in trials[2:]:
            t["status"] = "planned"
            for key in ("answer", "grade", "durationSeconds", "firstTextSeconds"):
                t.pop(key)
        result = audit_collection(data)
        self.assertEqual(result["totals"]["planned"], 36)
        self.assertEqual(result["totals"]["completed"], 1)
        self.assertEqual(result["totals"]["validFormat"], 1)
        self.assertEqual(result["totals"]["statuses"], dict(completed=1, error=1, planned=34))
        self.assertNotIn("private-", json.dumps(result))

    def test_all_runs_kept_and_duplicates_rejected(self):
        data = fixture()
        another = copy.deepcopy(data["audits"][0])
        another["id"] = "second-private-audit"
        data["audits"].append(another)
        self.assertEqual(audit_collection(data)["totals"]["planned"], 72)
        another["id"] = data["audits"][0]["id"]
        with self.assertRaises(ValueError):
            audit_collection(data)

    def test_forecast_history_limit_and_manifest_mismatch_are_reported(self):
        data = fixture()
        # Prior observations could have left the global retained window.
        data["audits"][0]["sourceProbes"][0]["predictedSuccess"] = .7
        manifest = dict(files=[dict(name="config.json", sha256="0" * 64, bytes=10)])
        result = audit_collection(data, manifest)["audits"][0]["sourceSummary"]
        self.assertFalse(result["forecastsMatchAvailablePrefix"])
        self.assertFalse(result["completeForecastHistoryAttested"])
        self.assertFalse(result["reportedIdentityMatchesPinnedManifest"])
