"""Independent consistency audit of an exported iPhone arithmetic report.

This checks reported data, not device authenticity or subjective experience.
No private export is copied into the repository by this script.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def audit(report, manifest=None):
    if report.get("schema") != "menia-iphone-capability-v1":
        raise ValueError("Unsupported report schema")
    probes = report["probes"]
    if len(probes) > 128 or len({p["id"] for p in probes}) != len(probes):
        raise ValueError("Invalid retention or duplicate probe IDs")
    counts = {}
    for p in probes:
        match = re.fullmatch(r"Calcule (\d{1,3}) ([+-]) (\d{1,3})\. Réponds uniquement par l’entier obtenu, sans explication\.", p["question"])
        if not match:
            raise ValueError("Unknown task contract")
        a, op, b = match.groups()
        expected = int(a) + int(b) if op == "+" else int(a) - int(b)
        correct = p["answer"].strip() == str(expected)
        if type(p["expected"]) is not int or p["expected"] != expected or type(p["correct"]) is not bool or p["correct"] != correct:
            raise ValueError("Reference or strict grade mismatch")
        prediction = p["predictedSuccess"]
        if type(prediction) not in (int, float) or not math.isfinite(prediction) or not 0 <= prediction <= 1:
            raise ValueError("Invalid forecast")
        n, s = counts.get(p["modelID"], (0, 0))
        # A full 128-record window may have discarded the forecasting history.
        if len(probes) < 128 and not math.isclose(prediction, (s + 1) / (n + 2), abs_tol=1e-12):
            raise ValueError("Prequential forecast mismatch")
        counts[p["modelID"]] = n + 1, s + int(correct)
    model = report["model"]
    samples = [p for p in probes if p["modelID"] == model["fingerprint"]]
    n = len(samples)
    s = sum(p["correct"] for p in samples)
    mean = (s + 1) / (n + 2)
    brier = sum((p["predictedSuccess"] - p["correct"]) ** 2 for p in samples) / n if n else None
    summary = report["summary"]
    for key, value in [("observations", n), ("successes", s), ("predictedSuccess", mean), ("brierScore", brier)]:
        given = summary.get(key)
        if value is None:
            matches = given is None
        else:
            matches = type(given) in (int, float) and math.isclose(given, value, abs_tol=1e-12)
        if not matches:
            raise ValueError(f"Summary mismatch: {key}")
    result = {"consistent": True, "observations": n, "successes": s,
              "observedAccuracy": s / n if n else None, "betaPosterior": [s + 1, n - s + 1],
              "nextPrediction": mean, "prequentialBrier": brier,
              "forecastHistoryFullyCheckable": len(probes) < 128,
              "deviceExecutionAttested": False,
              "latencyOrTokensPerSecondMeasurable": False}
    # Closed-form Beta(n+1,1) quantiles for an all-success sample.
    if n and s == n:
        result["betaEqualTail95UnderStationaryIID"] = [0.025 ** (1 / (n + 1)), 0.975 ** (1 / (n + 1))]
    if manifest is not None:
        files = sorted((f for f in manifest["files"] if f["name"].endswith(".safetensors") or f["name"] in
                       ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")), key=lambda f: f["name"])
        fingerprint = hashlib.sha256("".join(f"{f['name']}:{f['sha256']}\n" for f in files).encode()).hexdigest()
        result["reportedIdentityMatchesPinnedManifest"] = model["fingerprint"] == fingerprint and model["bytes"] == sum(f["bytes"] for f in files)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.report.read_text(encoding="utf-8-sig")),
                   json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None)
    data = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="utf-8")
    print(data)
