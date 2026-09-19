import copy
import json
import unittest

from research.iphone_capability_learning_report import (
    CONDITIONS, FAMILIES, FORECAST, ORDERS, SETTINGS, SOLVE, action, audit_collection, probability, reference,
)


def fixture():
    problems, calls, calibration = [], [], []
    for pi in range(48):
        row, family = pi % 24 // 4, FAMILIES[pi % 4]
        operands, letters = [], ""
        if family == "addition":
            operands = [100+pi, 201]
            question, expected = f"Calcule {100+pi} + 201.", 301+pi
        elif family == "multiplication":
            operands = [1000+pi, 2001]
            question, expected = f"Calcule {1000+pi} * 2001.", (1000+pi)*2001
        elif family == "countA":
            letters = "A"*pi + "B"*(64-pi)
            question, expected = f"Combien de lettres A contient cette chaîne : {letters} ?", pi
        else:
            operands = [10+pi, 20, 30, 40, 50, 60, 70, 80]
            question = f"Calcule {10+pi} - 20 + 30 - 40 + 50 - 60 + 70 - 80."
            expected = pi-40
        p = dict(id=f"PRIVATE_PROBLEM_{pi}", phase="calibration" if pi < 24 else "evaluation",
                 row=row, family=family, operands=operands, letters=letters, question=question, expectedAnswer=expected)
        same = [c for c in calibration if c[0]["family"] == family]
        p.update(familyProbability=(sum(c[1] for c in same)+1)/(len(same)+2),
                 pooledProbability=(sum(c[1] for c in calibration)+1)/(len(calibration)+2), numericDurationSeconds=.00001)
        ps = {}
        for condition in ([None] if pi < 24 else [*ORDERS[row], None]):
            c = dict(id=f"PRIVATE_CALL_{len(calls)}", problemIndex=pi, condition=condition, status="completed",
                     historyIDs=[], durationSeconds=1., firstTextSeconds=.2)
            if condition is None:
                correct = family != "multiplication" if pi < 24 else row % 2 == 0
                c.update(request=dict(instructions=SOLVE, prompt=question), correct=correct,
                         answer=str(expected) if correct else "wrong")
                if pi < 24:
                    calibration.append((p, correct, c))
                else:
                    chosen = action(ps["relevant"])
                    p.update(action=chosen, usedFallback=ps["relevant"] is None,
                             servedAnswer=str(expected) if chosen == "verify" else c["answer"],
                             pointLoss=.2 if chosen == "verify" else float(not correct))
                    if chosen == "verify":
                        p["verificationDurationSeconds"] = .00001
            else:
                target = family if condition == "relevant" else FAMILIES[(FAMILIES.index(family)+1) % 4]
                source = [] if condition == "absent" else [c for c in calibration if c[0]["family"] == target]
                payload = dict(family=family, question=question, history=None if condition == "absent" else [
                    dict(family=s["family"], question=s["question"], answer=c["answer"][:96], correct=y)
                    for s, y, c in source])
                prob = None if row == 0 else .9 if condition == "relevant" else .3
                c.update(request=dict(instructions=FORECAST, prompt=json.dumps(payload)),
                         historyIDs=[s["id"] for s, _, _ in source], probability=prob,
                         answer="invalid" if prob is None else json.dumps(dict(p=prob)))
                ps[condition] = prob
            calls.append(c)
        problems.append(p)
    return dict(schema="menia-iphone-capability-learning-collection-v1", audits=[dict(
        schema="menia-iphone-capability-learning-v1", id="PRIVATE_AUDIT", created="PRIVATE_TIME",
        appVersion="synthetic fixture", generationSettings=SETTINGS, model=dict(fingerprint="fixture", bytes=12),
        problems=problems, calls=calls)])


def truncate(collection, stop):
    """Represent an interrupted invocation, retaining decisions made before it."""
    audit = collection["audits"][0]
    for i, c in enumerate(audit["calls"]):
        if i < stop:
            continue
        for key in ("correct", "probability", "failure"):
            c.pop(key, None)
        if i == stop:
            c.update(status="cancelled", answer="partial", failure="stopped")
        else:
            for key in ("request", "answer", "durationSeconds", "firstTextSeconds"):
                c.pop(key, None)
            c.update(status="planned", historyIDs=[])
    for pi, p in enumerate(audit["problems"]):
        group = [c for c in audit["calls"] if c["problemIndex"] == pi]
        if all(c["status"] == "planned" for c in group):
            for key in ("familyProbability", "pooledProbability", "numericDurationSeconds"):
                p.pop(key, None)
        if group[-1]["status"] == "planned":
            p.pop("action", None); p.pop("usedFallback", None)
        if group[-1]["status"] != "completed":
            for key in ("servedAnswer", "pointLoss", "verificationDurationSeconds"):
                p.pop(key, None)
    return collection


class CapabilityLearningReportTests(unittest.TestCase):
    def test_negative_result_and_exact_metrics_are_retained_without_private_data(self):
        result = audit_collection(fixture())
        self.assertNotIn("PRIVATE_", json.dumps(result))
        a = result["audits"][0]
        self.assertTrue(a["complete"])
        self.assertEqual(a["evaluationSuccesses"], 12)
        relevant = a["methods"]["relevant"]
        self.assertEqual(relevant["validForecasts"], 20)
        self.assertEqual(relevant["verified"], 4)
        self.assertEqual(relevant["directErrors"], 12)
        self.assertAlmostEqual(relevant["systemBrier"], .45)
        self.assertAlmostEqual(relevant["validOnlyBrier"], .49)
        self.assertAlmostEqual(relevant["meanPointLoss"], 12.8/24)
        self.assertAlmostEqual(a["methods"]["absent"]["systemBrier"], .25)
        self.assertAlmostEqual(a["methods"]["alwaysVerify"]["meanPointLoss"], .2)
        self.assertEqual(a["contrasts"][0]["n"], 24)
        self.assertAlmostEqual(a["contrasts"][0]["meanBrierDifference"], .2)
        self.assertEqual(a["contrasts"][1]["n"], 20)

    def test_strict_probability_includes_duplicate_and_nonfinite_rejection(self):
        for raw in ['{"p":true}', '{"p":0.9,"p":0.1}', '{"p":".9"}', '{"p":1.1}',
                    '{"p":NaN}', '{"p":1e999}', '{"p":.5}', '{"p":01}', '{"p":0.9} text']:
            self.assertIsNone(probability(raw), raw)
        self.assertEqual(probability(' \n{"p":8e-1}\n'), .8)
        self.assertEqual(action(.8), "answer")
        self.assertEqual(action(.799999), "verify")

    def test_references_domains_and_commutative_overlap(self):
        self.assertEqual(reference(dict(family="alternatingSum", operands=[10, 99]*4, letters=""))[1], -356)
        c = fixture(); a = c["audits"][0]
        p = a["problems"][24]; source = a["problems"][0]
        p["operands"] = source["operands"][::-1]
        p["question"], p["expectedAnswer"], _ = reference(p)
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit_collection(c)

    def test_rejects_leaked_outcome_or_evaluation_history(self):
        for change in ("payload", "feedbackType", "history", "numeric", "candidate"):
            c = fixture(); a = c["audits"][0]; call = a["calls"][24]
            if change == "payload":
                payload = json.loads(call["request"]["prompt"])
                payload["expectedAnswer"] = a["problems"][24]["expectedAnswer"]
                call["request"]["prompt"] = json.dumps(payload)
            elif change == "feedbackType":
                payload = json.loads(call["request"]["prompt"])
                payload["history"][0]["correct"] = 1
                call["request"]["prompt"] = json.dumps(payload)
            elif change == "history":
                call["historyIDs"][0] = a["problems"][24]["id"]
            elif change == "numeric":
                a["problems"][24]["familyProbability"] = .99
            else:
                a["calls"][27]["request"]["prompt"] += " Previous result: correct"
            with self.assertRaises(ValueError, msg=change):
                audit_collection(c)

    def test_rejects_corrupted_grades_decisions_or_plan(self):
        for change in ("reference", "grade", "forecast", "action", "served", "order", "time"):
            c = fixture(); a = c["audits"][0]
            if change == "reference": a["problems"][0]["expectedAnswer"] += 1
            if change == "grade": a["calls"][0]["correct"] = False
            if change == "forecast": a["calls"][24]["probability"] = .9
            if change == "action": a["problems"][24]["action"] = "answer"
            if change == "served": a["problems"][24]["servedAnswer"] = "wrong"
            if change == "order": a["calls"][24]["condition"] = "absent"
            if change == "time": a["calls"][24]["firstTextSeconds"] = 2
            with self.assertRaises(ValueError, msg=change):
                audit_collection(c)

    def test_partial_runs_do_not_turn_technical_failures_into_task_errors(self):
        for stop, calibrated, evaluated in ((0, 0, 0), (23, 23, 0), (24, 24, 0), (27, 24, 0), (28, 24, 1), (119, 24, 23)):
            c = truncate(fixture(), stop)
            a = audit_collection(c)["audits"][0]
            self.assertEqual(a["calibrationCompleted"], calibrated)
            self.assertEqual(a["evaluationCompleted"], evaluated)
            self.assertFalse(a["complete"])
        c = truncate(fixture(), 27)
        c["audits"][0]["calls"][28]["status"] = "running"
        with self.assertRaises(ValueError):
            audit_collection(c)

    def test_running_snapshot_and_multiple_runs_remain_separate(self):
        c = truncate(fixture(), 24); call = c["audits"][0]["calls"][24]
        call["status"] = "running"
        for key in ("answer", "durationSeconds", "firstTextSeconds", "failure"):
            call.pop(key, None)
        self.assertEqual(audit_collection(c)["audits"][0]["evaluationCompleted"], 0)
        complete = fixture(); complete["audits"][0]["id"] = "PRIVATE_SECOND"
        c["audits"] += complete["audits"]
        result = audit_collection(c)
        self.assertEqual([a["evaluationCompleted"] for a in result["audits"]], [0, 24])

    def test_identity_is_checked_but_not_treated_as_execution_attestation(self):
        c = fixture(); manifest = dict(files=[dict(name="config.json", sha256="abc", bytes=12)])
        result = audit_collection(c, manifest)
        self.assertFalse(result["deviceExecutionAttested"])
        self.assertFalse(result["audits"][0]["reportedIdentityMatchesPinnedManifest"])


if __name__ == "__main__":
    unittest.main()
