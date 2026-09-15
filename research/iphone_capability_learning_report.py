"""Independent reconstruction of the prospective iPhone pilot; aggregate output only.

An internally consistent export does not attest device execution or complete run
history. Software fixtures are not LLM results. All runs are reported separately.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import re

from research.iphone_coupling_report import SETTINGS, close, finite_number, require, strict_json, timing

FAMILIES = ("addition", "multiplication", "countA", "alternatingSum")
CONDITIONS = ("relevant", "absent", "otherFamily")
ORDERS = tuple(itertools.permutations(CONDITIONS))
SOLVE = "Réponds uniquement par l'entier demandé, sans explication ni autre texte."
FORECAST = "Prédis la probabilité que tu répondes correctement à la question lors d'un appel séparé, sans outil, sans historique et avec les mêmes paramètres. La réussite exige uniquement l'entier exact. history contient des essais précédents de ce même modèle, ou null si aucun historique n'est fourni. Chaque réponse passée est limitée à ses 96 premiers caractères ; correct évalue la réponse complète. Utilise les informations pertinentes. Ne donne pas la solution. Réponds uniquement par un objet JSON avec une seule clé p, un nombre entre 0 et 1 : {\"p\":0.5}."
NUMBER = r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'


def probability(raw):
    if not isinstance(raw, str):
        return None
    match = re.fullmatch(r'\s*\{\s*"p"\s*:\s*(' + NUMBER + r')\s*\}\s*', raw)
    if match:
        p = float(match[1])
        if finite_number(p) and 0 <= p <= 1:
            return p
    return None


def action(p):
    return "answer" if p is not None and p >= .8 else "verify"


def loss(p, correct):
    return .2 if action(p) == "verify" else float(not correct)


def reference(p):
    family, operands, letters = p["family"], p["operands"], p["letters"]
    require(family in FAMILIES and type(operands) is list and all(type(n) is int for n in operands)
            and type(letters) is str, "Invalid task types")
    if family in ("addition", "multiplication"):
        low, high, symbol = (100, 999, "+") if family == "addition" else (1000, 9999, "*")
        require(len(operands) == 2 and all(low <= n <= high for n in operands) and letters == "", "Operand domain")
        answer = sum(operands) if family == "addition" else operands[0] * operands[1]
        return f"Calcule {operands[0]} {symbol} {operands[1]}.", answer, (family, *sorted(operands))
    if family == "countA":
        require(operands == [] and len(letters) == 64 and set(letters) <= set("ABCD"), "Letter task domain")
        return f"Combien de lettres A contient cette chaîne : {letters} ?", letters.count("A"), (family, letters)
    require(len(operands) == 8 and all(10 <= n <= 99 for n in operands) and letters == "", "Alternating domain")
    question = "Calcule " + "".join(("" if i == 0 else " + " if i % 2 == 0 else " - ") + str(n)
                                      for i, n in enumerate(operands)) + "."
    return question, sum(operands[::2]) - sum(operands[1::2]), (family, *operands)


def validate(audit):
    require(audit.get("schema") == "menia-iphone-capability-learning-v1", "Unsupported audit schema")
    require(audit["generationSettings"] == SETTINGS, "Generation settings differ")
    problems, calls = audit["problems"], audit["calls"]
    require(len(problems) == 48 and len(calls) == 120, "Plan size")
    require(len({p["id"] for p in problems}) == 48 and len({c["id"] for c in calls}) == 120, "Duplicate ID")
    seen, call_index, stopped = set(), 0, False
    calibration, evaluated = [], []
    for pi, problem in enumerate(problems):
        phase = "calibration" if pi < 24 else "evaluation"
        row = pi % 24 // 4
        require(problem["phase"] == phase and type(problem["row"]) is int and problem["row"] == row, "Phase/order mismatch")
        if pi % 4 == 0:
            require({p["family"] for p in problems[pi:pi+4]} == set(FAMILIES), "Unbalanced family block")
        question, expected, key = reference(problem)
        require(key not in seen, "Calibration/evaluation overlap or duplicate task")
        seen.add(key)
        require(problem["question"] == question and type(problem["expectedAnswer"]) is int
                and problem["expectedAnswer"] == expected, "Question/reference mismatch")
        same = [c for c in calibration if c[0]["family"] == problem["family"]]
        family_p = (sum(c[1] for c in same)+1)/(len(same)+2)
        pooled_p = (sum(c[1] for c in calibration)+1)/(len(calibration)+2)
        conditions = [None] if phase == "calibration" else [*ORDERS[row], None]
        group = calls[call_index:call_index+len(conditions)]
        started = any(c["status"] != "planned" for c in group)
        numeric_keys = ("familyProbability", "pooledProbability", "numericDurationSeconds")
        if started:
            require(close(problem.get("familyProbability"), family_p) and close(problem.get("pooledProbability"), pooled_p), "Forecast contains future data or wrong estimate")
            require(finite_number(problem.get("numericDurationSeconds")) and problem["numericDurationSeconds"] >= 0, "Numeric duration")
        else:
            require(all(problem.get(k) is None for k in numeric_keys), "Future numeric forecast")
        forecasts, forecast_times = {}, {}
        solution = None
        for call, condition in zip(group, conditions):
            require(type(call["problemIndex"]) is int and call["problemIndex"] == pi and call.get("condition") == condition, "Call plan mismatch")
            status = call["status"]
            require(status in ("planned", "running", "completed", "error", "cancelled"), "Unknown status")
            require(not stopped or status == "planned", "Call executed after unfinished predecessor")
            stopped |= status != "completed"
            source = []
            if condition is not None:
                target = problem["family"] if condition == "relevant" else FAMILIES[(FAMILIES.index(problem["family"])+1) % 4]
                source = [] if condition == "absent" else [c for c in calibration if c[0]["family"] == target]
                payload = dict(family=problem["family"], question=question, history=None if condition == "absent" else [
                    dict(family=p["family"], question=p["question"], answer=c["answer"][:96], correct=correct)
                    for p, correct, c in source])
            if status == "planned":
                require(call.get("request") is None and call["historyIDs"] == [], "Future request or history")
            else:
                request = call["request"]
                require(set(request) == {"instructions", "prompt"}, "Unexpected request fields")
                require(call["historyIDs"] == [p["id"] for p, _, _ in source], "History provenance mismatch")
                if condition is None:
                    require(request == dict(instructions=SOLVE, prompt=question), "Candidate sees extra context")
                else:
                    require(len(source) == (0 if condition == "absent" else 6), "Incomplete calibration history")
                    actual = strict_json(request["prompt"])
                    require(request["instructions"] == FORECAST and
                            json.dumps(actual, sort_keys=True) == json.dumps(payload, sort_keys=True),
                            "Forecast request mismatch or leaked outcome")
            terminal = status in ("completed", "error", "cancelled")
            if terminal:
                require(type(call.get("answer")) is str, "Missing raw answer")
                duration = call.get("durationSeconds")
                first = call.get("firstTextSeconds")
                require(finite_number(duration) and duration >= 0 and (first is None or finite_number(first) and 0 <= first <= duration), "Invalid inference duration")
                require(status != "error" or type(call.get("failure")) is str, "Missing technical failure")
                require(status != "completed" or call.get("failure") is None, "Completed call with failure")
            else:
                require(all(call.get(k) is None for k in ("answer", "durationSeconds", "firstTextSeconds", "failure")), "Nonterminal call with terminal data")
            if status == "completed" and condition is not None:
                p = probability(call["answer"])
                require(call.get("correct") is None and (call.get("probability") is None if p is None else close(call.get("probability"), p)), "Stored forecast mismatch")
                forecasts[condition] = p; forecast_times[condition] = call["durationSeconds"]
            elif status == "completed":
                correct = call["answer"].strip() == str(expected)
                require(type(call.get("correct")) is bool and call["correct"] == correct and call.get("probability") is None, "Stored candidate grade mismatch")
                solution = (call, correct)
            else:
                require(call.get("correct") is None and call.get("probability") is None, "Unfinished call graded")
            call_index += 1
        decision_started = phase == "evaluation" and group[-1]["status"] != "planned"
        if decision_started:
            require(len(forecasts) == 3, "Decision before forecasts")
            chosen = action(forecasts["relevant"])
            require(problem.get("action") == chosen and type(problem.get("usedFallback")) is bool
                    and problem["usedFallback"] == (forecasts["relevant"] is None), "Decision mismatch")
        else:
            require(problem.get("action") is None and problem.get("usedFallback") is None, "Premature decision")
        if solution is not None and phase == "evaluation":
            candidate, correct = solution
            verified = chosen == "verify"
            require(problem.get("servedAnswer") == (str(expected) if verified else candidate["answer"]), "Served answer mismatch")
            require(close(problem.get("pointLoss"), loss(forecasts["relevant"], correct)), "Executed point loss mismatch")
            duration = problem.get("verificationDurationSeconds")
            require(finite_number(duration) and duration >= 0 if verified else duration is None, "Verification duration mismatch")
            evaluated.append(dict(family=problem["family"], correct=correct, forecasts={**forecasts,
                "betaFamily": family_p, "betaPooled": pooled_p, "prior": .5}, forecastTimes=forecast_times,
                numericDuration=problem["numericDurationSeconds"], verificationDuration=duration))
        else:
            require(all(problem.get(k) is None for k in ("servedAnswer", "pointLoss", "verificationDurationSeconds")), "Unfinished or calibration task has served evaluation output")
        if solution is not None and phase == "calibration":
            calibration.append((problem, solution[1], solution[0]))
    return calibration, evaluated


def mean(values):
    return sum(values)/len(values) if values else None


def metrics(rows, method):
    # None is a format failure; p=.5 is the declared system fallback, not an LLM report.
    pairs = [(r["forecasts"].get(method), r["correct"]) for r in rows]
    if method == "alwaysAnswer":
        pairs = [(1., y) for _, y in pairs]
    if method == "alwaysVerify":
        pairs = [(0., y) for _, y in pairs]
    direct = [(p, y) for p, y in pairs if action(p) == "answer"]
    result = dict(n=len(pairs), validForecasts=sum(p is not None for p, _ in pairs),
                  invalidForecasts=sum(p is None for p, _ in pairs),
                  verified=sum(action(p) == "verify" for p, _ in pairs),
                  directAnswers=len(direct), directErrors=sum(not y for _, y in direct),
                  directErrorRate=mean([int(not y) for _, y in direct]),
                  meanPointLoss=mean([loss(p, y) for p, y in pairs]))
    if method not in ("alwaysAnswer", "alwaysVerify"):
        result.update(systemBrier=mean([((.5 if p is None else p) - y)**2 for p, y in pairs]),
                      validOnlyBrier=mean([(p-y)**2 for p, y in pairs if p is not None]))
    else:
        result.pop("validForecasts"); result.pop("invalidForecasts")
    if method in CONDITIONS:
        result["extraForecastDuration"] = timing([r["forecastTimes"][method] for r in rows])
    return result


def contrasts(rows):
    result = []
    for other in ("absent", "otherFamily", "betaFamily"):
        for common_only in (False, True):
            subset = [r for r in rows if not common_only or all(r["forecasts"][c] is not None for c in CONDITIONS)]
            brier, points = [], []
            for r in subset:
                a, b = r["forecasts"]["relevant"], r["forecasts"][other]
                y = r["correct"]
                brier.append(((.5 if a is None else a)-y)**2 - ((.5 if b is None else b)-y)**2)
                points.append(loss(a, y)-loss(b, y))
            result.append(dict(contrast="relevant-"+other, commonValidOnly=common_only, n=len(subset),
                               meanBrierDifference=mean(brier), meanPointLossDifference=mean(points)))
    return result


def audit_collection(collection, manifest=None):
    require(collection.get("schema") == "menia-iphone-capability-learning-collection-v1", "Unsupported collection")
    audits = collection["audits"]
    require(type(audits) is list and len(audits) > 0 and len({a["id"] for a in audits}) == len(audits), "Empty or duplicate audits")
    results = []
    for audit in audits:
        calibration, rows = validate(audit)
        methods = (*CONDITIONS, "betaFamily", "betaPooled", "prior", "alwaysAnswer", "alwaysVerify")
        complete_calls = [c for c in audit["calls"] if c["status"] == "completed"]
        result = dict(appVersion=audit["appVersion"], generationSettings=audit["generationSettings"],
                      fullPlanAndStoredScoresConform=True, calibrationCompleted=len(calibration),
                      evaluationCompleted=len(rows), complete=len(rows) == 24,
                      statuses=dict(Counter(c["status"] for c in audit["calls"])),
                      calibration=[dict(family=family, n=len(part := [c for c in calibration if c[0]["family"] == family]),
                                        successes=sum(c[1] for c in part), betaNext=(sum(c[1] for c in part)+1)/(len(part)+2),
                                        prequentialBrier=mean([(p["familyProbability"]-y)**2 for p, y, _ in part])) for family in FAMILIES],
                      evaluationSuccesses=sum(r["correct"] for r in rows),
                      methods={m: metrics(rows, m) for m in methods},
                      families={f: {m: metrics([r for r in rows if r["family"] == f], m) for m in methods} for f in FAMILIES},
                      contrasts=contrasts(rows),
                      allCompletedCallsDuration=timing([c["durationSeconds"] for c in complete_calls]),
                      firstTextDelay=timing([c["firstTextSeconds"] for c in complete_calls if c.get("firstTextSeconds") is not None]),
                      numericForecastDuration=timing([r["numericDuration"] for r in rows]),
                      executedVerificationDuration=timing([r["verificationDuration"] for r in rows if r["verificationDuration"] is not None]))
        if manifest is not None:
            files = sorted((f for f in manifest["files"] if f["name"].endswith(".safetensors") or f["name"] in
                            ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")), key=lambda f: f["name"])
            fingerprint = hashlib.sha256("".join(f"{f['name']}:{f['sha256']}\n" for f in files).encode()).hexdigest()
            result["reportedIdentityMatchesPinnedManifest"] = (audit["model"]["fingerprint"] == fingerprint
                and audit["model"]["bytes"] == sum(f["bytes"] for f in files))
        results.append(result)
    return dict(schema="menia-iphone-capability-learning-analysis-v1", auditCount=len(results),
                deviceExecutionAttested=False, completeCollectionHistoryAttested=False,
                lossUnits="declared points: wrong direct=1, verify=0.2, correct direct=0",
                executedPolicy="relevant forecast with fixed threshold and invalid-format verification fallback",
                otherPolicies="paired counterfactual scores on the same candidate and deterministic verifier",
                invalidProbabilityForSystemBrier=.5, audits=results)


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
