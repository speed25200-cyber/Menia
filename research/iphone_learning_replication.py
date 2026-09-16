"""Follow-up to the fixed learning pilot. Never replaces its original evaluator.

The empirical-rate comparator was exploratory on the first run, and specified
before the three follow-up attempts. Preserve every attempt; never choose by score.
Raw IDs and dates are used for matching/order only and never emitted.
"""
import argparse
import json
from pathlib import Path

from research.iphone_capability_learning_report import (
    CONDITIONS, FAMILIES, audit_collection, mean, metrics, validate,
)
from research.iphone_coupling_report import finite_number, require, strict_json


def within_family(rows, family):
    subset = [r for r in rows if r["family"] == family]
    good = [r for r in subset if r["correct"]]
    bad = [r for r in subset if not r["correct"]]
    def auc(condition):
        pairs = [(g["forecasts"][condition], b["forecasts"][condition]) for g in good for b in bad]
        if not pairs or any(a is None or b is None for a, b in pairs):
            return None
        return mean([float(a > b) + .5 * (a == b) for a, b in pairs])
    return dict(family=family, n=len(subset), successes=len(good), failures=len(bad),
                mixedOutcomes=bool(good and bad), pairCount=len(good)*len(bad),
                auc={m: auc(m) for m in CONDITIONS},
                aucScope="Within this family and run; null if no outcome contrast or invalid forecast")


def supplement(audit):
    calibration, rows = validate(audit)
    rates = {}
    for family in FAMILIES:
        same = [y for p, y, _ in calibration if p["family"] == family]
        rates[family] = sum(same)/len(same) if same else None
    for row in rows:
        require(rates[row["family"]] is not None, "Evaluation without calibration")
        row["forecasts"]["empiricalFamily"] = rates[row["family"]]
    comparisons = []
    for method in ("absent", "otherFamily", "betaFamily", "empiricalFamily"):
        a, b = metrics(rows, "relevant"), metrics(rows, method)
        comparisons.append(dict(comparator=method, n=len(rows),
            brierDifference=None if not rows else a["systemBrier"]-b["systemBrier"],
            lossDifference=None if not rows else a["meanPointLoss"]-b["meanPointLoss"]))
    return dict(calibrationRates=rates, evaluated=len(rows), complete=len(rows) == 24,
                empiricalFamily=metrics(rows, "empiricalFamily"),
                relevantMinusComparator=comparisons,
                withinFamily=[within_family(rows, f) for f in FAMILIES])


def analyze(collection, baseline_collection, manifest=None):
    require(len(baseline_collection.get("audits", [])) == 1, "Provide the original single-run collection as baseline")
    original = baseline_collection["audits"][0]
    audit_collection(baseline_collection, manifest)
    primary = audit_collection(collection, manifest)
    audits = collection["audits"]
    require(all(finite_number(a.get("created")) for a in audits), "Missing numeric creation date")
    require(len({a["created"] for a in audits}) == len(audits), "Ambiguous creation order")
    matches = [a for a in audits if a["id"] == original["id"]]
    require(len(matches) == 1, "Original pilot missing; export the entire collection")
    require(json.dumps(matches[0], sort_keys=True) == json.dumps(original, sort_keys=True), "Original pilot changed")
    ordered = sorted(enumerate(audits), key=lambda item: item[1]["created"])
    after = [(i, a) for i, a in ordered if a["created"] > original["created"]]
    slots = {a["id"]: slot+1 for slot, (_, a) in enumerate(after[:3])}
    records = []
    for i, a in ordered:
        role = ("initial_exploratory" if a["id"] == original["id"] else
                "prespecified_replication" if a["id"] in slots else "additional_exploratory")
        compatible = (a["model"] == original["model"] and a["appVersion"] == original["appVersion"]
                      and a["generationSettings"] == original["generationSettings"])
        records.append(dict(role=role, replicationSlot=slots.get(a["id"]),
                            sameModelAppAndSettings=compatible,
                            primary=primary["audits"][i], supplementary=supplement(a)))
    selected = [r for r in records if r["role"] == "prespecified_replication"]
    complete = len(selected) == 3 and all(r["sameModelAppAndSettings"] and r["supplementary"]["complete"] for r in selected)
    run_means = None
    if complete:
        run_means = []
        for index, method in enumerate(("absent", "otherFamily", "betaFamily", "empiricalFamily")):
            cells = [r["supplementary"]["relevantMinusComparator"][index] for r in selected]
            run_means.append(dict(comparator=method, runs=3,
                equalRunMeanBrierDifference=mean([c["brierDifference"] for c in cells]),
                equalRunMeanLossDifference=mean([c["lossDifference"] for c in cells]),
                runsWithLowerBrier=sum(c["brierDifference"] < 0 for c in cells),
                runsWithLowerLoss=sum(c["lossDifference"] < 0 for c in cells)))
    return dict(schema="menia-iphone-learning-replication-v1", requestedFollowupAttempts=3,
                receivedFollowupSlots=len(selected), missingFollowupSlots=3-len(selected),
                allThreeCompleteAndCompatible=complete, equalRunComparison=run_means,
                baselineUsedInReplicationMean=False, runs=records,
                inference="Descriptive local replication; no significance test or consciousness inference",
                historyCompletenessAttested=False, deviceExecutionAttested=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(strict_json(args.collection.read_text(encoding="utf-8-sig")),
                     strict_json(args.baseline.read_text(encoding="utf-8-sig")),
                     strict_json(args.manifest.read_text(encoding="utf-8")) if args.manifest else None)
    data = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="utf-8")
    print(data)
