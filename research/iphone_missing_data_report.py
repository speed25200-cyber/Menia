"""Independent, aggregate-only analysis of the 72-response missing-data follow-up."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from research.iphone_coupling_report import (
    INSTRUCTIONS, REFERENTS, SETTINGS, finite_number, grade as coupling_grade,
    parse_reply, require, strict_json, timing, valid_summary,
)

CONDITIONS = ("originalOmitted", "revisedOmitted", "revisedNull", "providedHigh", "providedLow", "providedZero")
COUNTS = (7, 11, 17, 23, 31, 47)
BASE = (0, 1, 5, 2, 4, 3)
ORIGIN = "synthetic protocol fixtures; not observed device capabilities"
REVISED = INSTRUCTIONS.replace(
    "Si le champ bilan est absent, donne null pour les trois nombres.",
    "Si le champ bilan est absent ou vaut null, donne null pour les trois nombres."
).replace('si le bilan est absent, action vaut "mesurer"',
          'si le bilan est absent ou vaut null, action vaut "mesurer"')


def expected_input(condition, role, row):
    n = COUNTS[row]
    summary = None
    if condition == "providedHigh":
        summary = dict(observations=n, successes=n, predictedSuccess=(n + 1)/(n + 2))
    elif condition == "providedLow":
        summary = dict(observations=n, successes=0, predictedSuccess=1/(n + 2))
    elif condition == "providedZero":
        summary = dict(observations=0, successes=0, predictedSuccess=.5)
    name = "toi." if role == "selfReport" else "un agent de référence."
    payload = dict(question="Restitue le bilan de calcul qui concerne " + name + " Indique les observations, les réussites, la prévision fournie et le choix demandé.")
    if summary is not None or condition == "revisedNull":
        payload["bilan"] = summary
    return summary, INSTRUCTIONS if condition == "originalOmitted" else REVISED, payload


def grade(answer, provided):
    auxiliary = provided or dict(observations=0, successes=0, predictedSuccess=.5)
    score = coupling_grade(answer, provided, auxiliary)
    return dict(validFormat=score["validFormat"], numbersCorrect=score["matchesProvidedSummary"],
                actionCorrect=score["matchesDecisionRule"],
                completeSuccess=score["validFormat"] and score["matchesProvidedSummary"] and score["matchesDecisionRule"],
                reportedAction=score["reportedAction"])


def validate(audit):
    require(audit.get("schema") == "menia-iphone-missing-data-v1", "Unsupported audit schema")
    require(audit.get("dataOrigin") == ORIGIN, "Unrecognized data origin")
    require(audit["generationSettings"] == SETTINGS, "Generation settings differ")
    trials = audit["trials"]
    require(len(trials) == 72 and len({t["id"] for t in trials}) == 72, "Trial count or duplicate ID")
    seen_rows = set()
    stopped = False
    for block in range(6):
        group = trials[block * 12:(block + 1) * 12]
        row = group[0]["designRow"]
        require(type(row) is int and 0 <= row < 6 and row not in seen_rows, "Invalid design row")
        seen_rows.add(row)
        require({group[0]["referent"], group[6]["referent"]} == set(REFERENTS), "Referent order")
        for start in (0, 6):
            for pos, t in enumerate(group[start:start + 6]):
                condition = CONDITIONS[(BASE[pos] + row) % 6]
                role = group[start]["referent"]
                require(type(t["block"]) is int and t["block"] == block and type(t["designRow"]) is int
                        and t["designRow"] == row and t["referent"] == role and t["condition"] == condition, "Plan mismatch")
                summary, instructions, payload = expected_input(condition, role, row)
                given = t.get("providedSummary")
                require(given == summary and (given is None or valid_summary(given)), "Provided summary mismatch")
                request = t["request"]
                actual = strict_json(request["prompt"])
                require(set(request) == {"instructions", "prompt"} and request["instructions"] == instructions
                        and actual == payload and (summary is None or valid_summary(actual.get("bilan"))), "Prompt mismatch")
                status = t["status"]
                require(status in ("completed", "planned", "running", "cancelled", "error"), "Unknown status")
                require(not stopped or status == "planned", "Nonsequential status")
                stopped |= status != "completed"
                if status in ("completed", "cancelled", "error"):
                    duration, first = t.get("durationSeconds"), t.get("firstTextSeconds")
                    require(type(t.get("answer")) is str and finite_number(duration) and duration >= 0, "Missing answer or duration")
                    require(first is None or (finite_number(first) and 0 <= first <= duration), "First-text timing")
                    require(status != "completed" or t.get("failure") is None, "Completed trial with failure")
                    require(status != "error" or type(t.get("failure")) is str, "Missing error description")
                else:
                    require(all(t.get(k) is None for k in ("answer", "durationSeconds", "firstTextSeconds", "failure")), "Unfinished trial has terminal data")
                require(status == "completed" or t.get("grade") is None, "Unfinished trial is graded")


def tally(rows):
    scores = [g for _, g in rows if g is not None]
    return dict(planned=len(rows), completed=len(scores),
                statuses=dict(sorted(Counter(t["status"] for t, _ in rows).items())),
                **{k: sum(g[k] for g in scores) for k in ("validFormat", "numbersCorrect", "actionCorrect", "completeSuccess")},
                actions=dict(sorted(Counter(g["reportedAction"] or "invalid" for g in scores).items())))


def contrasts(rows):
    by_key = {(t["referent"], t["block"], t["condition"]): g for t, g in rows}
    result = []
    for role in REFERENTS:
        for name, before, after in (("B-A", "originalOmitted", "revisedOmitted"), ("C-B", "revisedOmitted", "revisedNull")):
            pairs = []
            for block in range(6):
                a, b = by_key[role, block, before], by_key[role, block, after]
                if a is not None and b is not None:
                    pairs.append(dict(block=block, numbersChange=int(b["numbersCorrect"]) - int(a["numbersCorrect"]),
                                      jointChange=int(b["completeSuccess"]) - int(a["completeSuccess"])))
            result.append(dict(referent=role, contrast=name, plannedPairs=6, completedPairs=len(pairs),
                               meanNumbersChange=sum(p["numbersChange"] for p in pairs)/len(pairs) if pairs else None,
                               meanJointChange=sum(p["jointChange"] for p in pairs)/len(pairs) if pairs else None,
                               pairs=pairs))
    return result


def audit_collection(collection, manifest=None):
    require(collection.get("schema") == "menia-iphone-missing-data-collection-v1", "Unsupported collection schema")
    audits = collection["audits"]
    require(type(audits) is list and len(audits) > 0, "Empty collection")
    require(len({a["id"] for a in audits}) == len(audits), "Duplicate audit ID")
    results, all_rows = [], []
    for audit in audits:
        validate(audit)
        rows, discrepancies = [], []
        absent_patterns = Counter()
        for index, trial in enumerate(audit["trials"]):
            computed = grade(trial["answer"], trial.get("providedSummary")) if trial["status"] == "completed" else None
            rows.append((trial, computed))
            if computed is not None:
                stored = trial.get("grade") or {}
                normalized = {k: stored.get(k) for k in computed}
                if set(stored) - set(computed) or json.dumps(normalized, sort_keys=True) != json.dumps(computed, sort_keys=True):
                    discrepancies.append(index + 1)
                reply = parse_reply(trial["answer"])
                if trial["condition"] in CONDITIONS[:3] and reply is not None:
                    absent_patterns[(trial["referent"], trial["condition"], json.dumps(reply, sort_keys=True))] += 1
        all_rows.extend(rows)
        done = [t for t, g in rows if g is not None]
        result = dict(appVersion=audit["appVersion"], generationSettings=audit["generationSettings"], fullPlanConforms=True,
                      storedGradesMatch=not discrepancies, gradeDiscrepancyTrials=discrepancies,
                      totals=tally(rows), cells=[dict(referent=role, condition=condition,
                                                    **tally([(t, g) for t, g in rows if t["referent"] == role and t["condition"] == condition]))
                                               for role in REFERENTS for condition in CONDITIONS],
                      contrasts=contrasts(rows),
                      absentOutputPatterns=[dict(referent=role, condition=condition, reply=json.loads(reply), count=count)
                                            for (role, condition, reply), count in sorted(absent_patterns.items())],
                      callDuration=timing([t["durationSeconds"] for t in done]),
                      firstTextDelay=timing([t["firstTextSeconds"] for t in done if t.get("firstTextSeconds") is not None]))
        if manifest is not None:
            files = sorted((f for f in manifest["files"] if f["name"].endswith(".safetensors") or f["name"] in
                            ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")), key=lambda f: f["name"])
            fingerprint = hashlib.sha256("".join(f"{f['name']}:{f['sha256']}\n" for f in files).encode()).hexdigest()
            result["reportedIdentityMatchesPinnedManifest"] = (audit["model"]["fingerprint"] == fingerprint
                and audit["model"]["bytes"] == sum(f["bytes"] for f in files))
        results.append(result)
    return dict(schema="menia-iphone-missing-data-analysis-v1", auditCount=len(results),
                deviceExecutionAttested=False, completeCollectionHistoryAttested=False,
                tokensPerSecondMeasurable=False, totals=tally(all_rows), audits=results)


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
