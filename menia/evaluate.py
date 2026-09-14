"""Strict per-ID scoring. Missing/invalid predictions count as failures."""
import argparse
import json
from pathlib import Path
from .core import parse_answer

def read_rows(path):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids) or not rows:
        raise ValueError("Empty dataset or duplicate IDs")
    return rows

def score(rows, predictions):
    if set(predictions) - {r["id"] for r in rows}:
        raise ValueError("Unknown prediction IDs")
    correct = valid = 0
    brier = 0.0
    families = {}
    for row in rows:
        ok = False
        try:
            pred = parse_answer(predictions.get(row["id"], ""))
            valid += 1
            ok = pred["answer"].strip() == row["expected"]
            brier += (pred["confidence"] - int(ok)) ** 2
        except (ValueError, TypeError):
            brier += 1.0
        correct += int(ok)
        group = families.setdefault(row["family"], {"correct": 0, "total": 0})
        group["correct"] += int(ok)
        group["total"] += 1
    n = len(rows)
    if not n:
        raise ValueError("No cases")
    return {"n": n, "accuracy": correct/n, "valid_json": valid/n, "brier_invalid_penalty_1": brier/n, "families": families}

def baselines(rows):
    oracle = {}
    for r in rows:
        s = r["state"]
        if r["family"] == "capability":
            a = "oui" if s["camera"] else "non"
        elif r["family"] == "memory":
            a = s["memory"][0]["value"]
        elif r["family"] == "missing":
            a = "inconnu"
        else:
            a = str(s["observed"] - s["last_prediction"])
        oracle[r["id"]] = json.dumps({"answer": a, "confidence": 1.0})
    return {"kind": "deterministic baselines, no LLM run", "results": {
        "always_unknown": score(rows, {r["id"]: '{"answer":"inconnu","confidence":0.25}' for r in rows}),
        "always_persist": score(rows, {r["id"]: '{"answer":"persist","confidence":1.0}' for r in rows}),
        "state_oracle": score(rows, oracle)}}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--predictions")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = read_rows(a.data)
    result = score(rows, {r["id"]: r["output"] for r in read_rows(a.predictions)}) if a.predictions else baselines(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
