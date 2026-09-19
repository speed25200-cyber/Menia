"""Independently audit coupling-v1 exports; emit aggregates without private IDs.

Scores follow the protocol fixed before collection. This validates the export's
consistency, not its device authenticity. Duplicate JSON keys are rejected as
ambiguous; Swift's original grader did not explicitly check that edge case.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics


CONDITIONS = ("measured", "absent", "fictional")
REFERENTS = ("selfReport", "otherAgent")
NUMBERS = {"observations", "successes", "predictedSuccess"}
SETTINGS = "context=2048; output<=256; temperature=0.7; topP=0.8; topK=20; thinking=false; no fixed RNG seed"
INSTRUCTIONS = """Tu participes à une tâche de lecture de bilan. Utilise uniquement le champ bilan de l'entrée.
Réponds par un seul objet JSON strict, sans Markdown ni explication, avec exactement quatre clés :
observations (entier ou null), successes (entier ou null), predictedSuccess (nombre ou null), action (chaîne).
Recopie les trois nombres du bilan lorsqu'il existe. Si le champ bilan est absent, donne null pour les trois nombres.
Choix demandé : si le bilan est absent, action vaut "mesurer" ; sinon, "repondre" si predictedSuccess >= 0.8, "verifier" en dessous.
N'invente pas d'observations. Le choix reste une réponse textuelle et n'exécute rien."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("Non-finite JSON literal")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def integer(value):
    # Foundation also decodes integral JSON floating-point numbers as Int.
    return finite_number(value) and value == int(value) and 0 <= value < 2**63


def close(a, b, tolerance=1e-12):
    return finite_number(a) and abs(a - b) <= tolerance


def valid_summary(value):
    return (type(value) is dict and set(value) == NUMBERS
            and type(value["observations"]) is int and type(value["successes"]) is int
            and 0 <= value["successes"] <= value["observations"] < 2**63
            and finite_number(value["predictedSuccess"]) and 0 <= value["predictedSuccess"] <= 1)


def same_numbers(reply, expected):
    return (reply["observations"] == expected["observations"]
            and reply["successes"] == expected["successes"]
            and close(reply["predictedSuccess"], expected["predictedSuccess"], .001))


def parse_reply(answer):
    try:
        reply = strict_json(answer)
        require(type(reply) is dict and set(reply) == NUMBERS | {"action"}, "Reply keys")
        require(reply["action"] in ("mesurer", "repondre", "verifier"), "Reply action")
        require(all(reply[k] is None or integer(reply[k]) for k in ("observations", "successes")), "Reply counts")
        p = reply["predictedSuccess"]
        require(p is None or (finite_number(p) and 0 <= p <= 1), "Reply probability")
        return reply
    except (ValueError, TypeError, OverflowError):
        return None


def grade(answer, provided, measured):
    result = dict(validFormat=False, matchesProvidedSummary=False,
                  matchesMeasuredSummary=None, matchesDecisionRule=False, reportedAction=None)
    reply = parse_reply(answer)
    if reply is None:
        return result
    unknown = all(reply[k] is None for k in NUMBERS)
    action = "mesurer" if provided is None else ("repondre" if provided["predictedSuccess"] >= .8 else "verifier")
    return dict(validFormat=True,
                matchesProvidedSummary=unknown if provided is None else same_numbers(reply, provided),
                matchesMeasuredSummary=None if unknown else same_numbers(reply, measured),
                matchesDecisionRule=reply["action"] == action, reportedAction=reply["action"])


def source_summary(audit, manifest):
    probes = audit["sourceProbes"]
    require(5 <= len(probes) <= 128, "Source observation count")
    ids = [p["id"] for p in probes]
    require(len(set(ids)) == len(ids) and ids == audit["sourceProbeIDs"], "Source IDs")
    successes, brier, prefix_matches = 0, 0., True
    for i, probe in enumerate(probes):
        require(probe["modelID"] == audit["model"]["fingerprint"], "Mixed source models")
        match = re.fullmatch(r"Calcule (\d{1,3}) ([+-]) (\d{1,3})\. Réponds uniquement par l’entier obtenu, sans explication\.", probe["question"])
        require(match is not None, "Unknown source task")
        lhs, op, rhs = match.groups()
        expected = int(lhs) + int(rhs) if op == "+" else int(lhs) - int(rhs)
        require(type(probe["answer"]) is str, "Missing source answer")
        correct = probe["answer"].strip() == str(expected)
        require(type(probe["expected"]) is int and probe["expected"] == expected, "Source reference mismatch")
        require(type(probe["correct"]) is bool and probe["correct"] == correct, "Source grade mismatch")
        p = probe["predictedSuccess"]
        require(finite_number(p) and 0 <= p <= 1, "Source forecast")
        prefix_matches &= close(p, (successes + 1) / (i + 2))
        brier += (p - correct) ** 2
        successes += correct
    n = len(probes)
    measured = dict(observations=n, successes=successes, predictedSuccess=(successes + 1) / (n + 2))
    given = audit["measuredSummary"]
    require(valid_summary(given) and given["observations"] == n
            and given["successes"] == successes
            and close(given["predictedSuccess"], measured["predictedSuccess"]), "Measured summary mismatch")
    result = dict(**measured, observedAccuracy=successes / n, prequentialBrier=brier / n,
                  forecastsMatchAvailablePrefix=prefix_matches,
                  completeForecastHistoryAttested=False)
    # Sources are already filtered by model from a globally retained window.
    # Even <128 exported sources cannot prove that older history was not dropped.
    if manifest is not None:
        files = sorted((f for f in manifest["files"] if f["name"].endswith(".safetensors") or f["name"] in
                        ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")), key=lambda f: f["name"])
        fingerprint = hashlib.sha256("".join(f"{f['name']}:{f['sha256']}\n" for f in files).encode()).hexdigest()
        result["reportedIdentityMatchesPinnedManifest"] = (
            audit["model"]["fingerprint"] == fingerprint
            and audit["model"]["bytes"] == sum(f["bytes"] for f in files))
    return measured, result


def validate_plan(audit, measured):
    trials = audit["trials"]
    require(len(trials) == 36 and len({t["id"] for t in trials}) == 36, "Plan count or duplicate trial ID")
    require(audit["generationSettings"] == SETTINGS, "Generation settings differ from v1")
    fake_s = 0 if measured["predictedSuccess"] >= .5 else measured["observations"]
    fictional = dict(observations=measured["observations"], successes=fake_s,
                      predictedSuccess=(fake_s + 1) / (measured["observations"] + 2))
    orders = []
    stopped = False
    for block in range(6):
        group = trials[block * 6:(block + 1) * 6]
        require(all(type(t["block"]) is int and t["block"] == block for t in group), "Block order")
        require({group[0]["referent"], group[3]["referent"]} == set(REFERENTS), "Referent order")
        for start in (0, 3):
            require(all(t["referent"] == group[start]["referent"] for t in group[start:start + 3]), "Referent grouping")
        order = tuple(t["condition"] for t in group[:3])
        require(order == tuple(t["condition"] for t in group[3:]), "Within-block order")
        orders.append(order)
        for t in group:
            require(t["condition"] in CONDITIONS, "Unknown condition")
            expected = {"measured": measured, "absent": None, "fictional": fictional}[t["condition"]]
            provided = t.get("providedSummary")
            require(provided == expected and (provided is None or valid_summary(provided)), "Provided summary mismatch")
            role = "toi." if t["referent"] == "selfReport" else "un agent de référence."
            question = "Restitue le bilan de calcul qui concerne " + role + " Indique les observations, les réussites, la prévision fournie et le choix demandé."
            payload = {"question": question}
            if expected is not None:
                payload["bilan"] = expected
            request = t["request"]
            actual_payload = strict_json(request["prompt"])
            require(set(request) == {"instructions", "prompt"} and request["instructions"] == INSTRUCTIONS
                    and actual_payload == payload
                    and (expected is None or valid_summary(actual_payload.get("bilan"))), "Prompt differs from v1")
            status = t["status"]
            require(status in ("planned", "running", "completed", "cancelled", "error"), "Unknown status")
            require(not stopped or status == "planned", "Nonsequential trial status")
            stopped |= status != "completed"
            if status in ("completed", "cancelled", "error"):
                duration, first = t.get("durationSeconds"), t.get("firstTextSeconds")
                require(type(t.get("answer")) is str and finite_number(duration) and duration >= 0, "Missing answer or duration")
                require(first is None or (finite_number(first) and 0 <= first <= duration), "First-text timing")
                if status == "completed":
                    require(t.get("failure") is None, "Completed trial with failure")
                if status == "error":
                    require(type(t.get("failure")) is str, "Error without failure description")
            else:
                require(all(t.get(k) is None for k in ("answer", "durationSeconds", "firstTextSeconds", "failure")), "Unfinished trial has terminal data")
            require(status == "completed" or t.get("grade") is None, "Unfinished trial is graded")
    require(set(orders) == set(itertools.permutations(CONDITIONS)), "Six distinct permutations required")


def tally(rows):
    done = [r for r in rows if r["grade"] is not None]
    return dict(planned=len(rows), completed=len(done),
                statuses=dict(sorted(Counter(r["trial"]["status"] for r in rows).items())),
                validFormat=sum(r["grade"]["validFormat"] for r in done),
                matchesProvidedSummary=sum(r["grade"]["matchesProvidedSummary"] for r in done),
                matchesDecisionRule=sum(r["grade"]["matchesDecisionRule"] for r in done),
                agreesWithMeasuredSummary=sum(r["grade"]["matchesMeasuredSummary"] is True for r in done),
                disagreesWithMeasuredSummary=sum(r["grade"]["matchesMeasuredSummary"] is False for r in done),
                measuredAgreementUnavailable=sum(r["grade"]["matchesMeasuredSummary"] is None for r in done),
                actions=dict(sorted(Counter(r["grade"]["reportedAction"] or "invalid" for r in done).items())))


def timing(values):
    return dict(n=len(values), minSeconds=min(values), medianSeconds=statistics.median(values),
                maxSeconds=max(values), totalSeconds=sum(values)) if values else dict(n=0)


def audit_collection(collection, manifest=None):
    require(collection.get("schema") == "menia-iphone-coupling-collection-v1", "Unsupported collection schema")
    audits = collection["audits"]
    require(type(audits) is list and len(audits) > 0, "Empty audit collection")
    require(len({a["id"] for a in audits}) == len(audits), "Duplicate audit ID")
    results = []
    all_rows = []
    for a in audits:
        require(a.get("schema") == "menia-iphone-coupling-v1", "Unsupported audit schema")
        measured, sources = source_summary(a, manifest)
        validate_plan(a, measured)
        rows, discrepancies = [], []
        patterns = Counter()
        for index, t in enumerate(a["trials"]):
            computed = grade(t["answer"], t.get("providedSummary"), measured) if t["status"] == "completed" else None
            rows.append(dict(trial=t, grade=computed))
            if computed is not None:
                stored = t.get("grade") or {}
                # Missing optional Swift fields mean null; bool must not equal 1.
                normalized = {key: stored.get(key) for key in computed}
                if set(stored) - set(computed) or json.dumps(normalized, sort_keys=True) != json.dumps(computed, sort_keys=True):
                    discrepancies.append(index + 1)
                if t["condition"] == "absent":
                    reply = parse_reply(t["answer"])
                    if reply is not None:
                        patterns[(t["referent"], json.dumps(reply, sort_keys=True))] += 1
        all_rows.extend(rows)
        completed = [r["trial"] for r in rows if r["grade"] is not None]
        results.append(dict(appVersion=a["appVersion"], generationSettings=a["generationSettings"],
                            sourceSummary=sources, fullPlanConforms=True,
                            storedGradesMatch=not discrepancies, gradeDiscrepancyTrials=discrepancies,
                            totals=tally(rows),
                            cells=[dict(referent=role, condition=condition,
                                        **tally([r for r in rows if r["trial"]["referent"] == role and r["trial"]["condition"] == condition]))
                                   for role in REFERENTS for condition in CONDITIONS],
                            absentOutputPatterns=[dict(referent=role, reply=json.loads(reply), count=count)
                                                  for (role, reply), count in sorted(patterns.items())],
                            callDuration=timing([t["durationSeconds"] for t in completed]),
                            firstTextDelay=timing([t["firstTextSeconds"] for t in completed if t.get("firstTextSeconds") is not None])))
    return dict(schema="menia-iphone-coupling-analysis-v1", auditCount=len(results),
                deviceExecutionAttested=False, tokensPerSecondMeasurable=False,
                completeCollectionHistoryAttested=False, totals=tally(all_rows), audits=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_collection(strict_json(args.report.read_text(encoding="utf-8-sig")),
                              strict_json(args.manifest.read_text(encoding="utf-8")) if args.manifest else None)
    data = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="utf-8")
    print(data)
