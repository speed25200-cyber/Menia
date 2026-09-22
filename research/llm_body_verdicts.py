"""Verdicts of the latent-body and adjusted-body protocols, recomputed from the rows alone.

Committed before any result of the Mac was read. The thresholds are those of
docs/LATENT_BODY_PROTOCOL.md (L1-L5) and docs/ADJUSTED_BODY_PROTOCOL.md (A1-A5);
this module only fixes the readings the protocols leave implicit:

- "après au moins un mouvement" (L5, A3): a move was played earlier in the life;
  reading the mark alone does not count.
- A2: the confidence when wrong is taken on the same rows as the accuracy,
  jeu R, commandes nouvelles; the value over every wrong prediction of jeu R is
  reported beside it, not used.
- L3: jeu M, rows at steps 12 to 23 (every row after the imposed change).
- L5: the three V models of artifacts/own-action-channel pooled, each model
  reported; the first 48 lives of their jeu R are the lives Qwen was asked on.
- Validity is read first; an invalid model's criteria are reported as None.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from .llm_latent_body import transformer_rows
from .origin_env import N_MOVE
from .own_action_experiment import CHANGE_STEP

VALIDITY_MASS = 0.50


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def with_earlier_move(rows):
    """Mark each row with whether a move was played earlier in its life (rows are the move steps)."""
    first = {}
    for r in rows:
        first[r["life"]] = min(first.get(r["life"], r["step"]), r["step"])
    return [dict(r, after_move=r["step"] > first[r["life"]]) for r in rows]


def accuracy(rows):
    return float(np.mean([r["correct"] for r in rows])) if rows else None


def confidence_when_wrong(rows):
    wrong = [r["confidence"] for r in rows if not r["correct"]]
    return float(np.mean(wrong)) if wrong else None


def at_least(value, threshold):
    return value is not None and value >= threshold


def at_most(value, threshold):
    return value is not None and value <= threshold


def digit_mass(rows):
    return float(np.mean([r["digit_mass"] for r in rows])) if rows else None


def cells(rows_r, rows_m):
    """The measured cells shared by both protocols, on one model's rows."""
    rows_r = with_earlier_move(rows_r)
    new = [r for r in rows_r if r["category"] == "nouveau"]
    changed = [r for r in rows_m if r["step"] >= CHANGE_STEP]
    late = [r for r in rows_m if r["step"] >= 16]
    return {
        "digit_mass": digit_mass(rows_r + rows_m),
        "R_seen": accuracy([r for r in rows_r if r["category"] == "vu"]),
        "R_new": accuracy(new),
        "R_new_after_move": accuracy([r for r in new if r["after_move"]]),
        "R_confidence_when_wrong": confidence_when_wrong(rows_r),
        "R_new_confidence_when_wrong": confidence_when_wrong(new),
        "M_12_23_seen_after": accuracy([r for r in changed if r["category"] == "vu_apres"]),
        "M_12_23_seen_before_only": accuracy([r for r in changed if r["category"] == "vu_avant_seulement"]),
        "M_12_23_new": accuracy([r for r in changed if r["category"] == "nouveau"]),
        "M_16_23": accuracy(late),
        "M_before_change": accuracy([r for r in rows_m if r["step"] < CHANGE_STEP]),
        "counts": {"R": len(rows_r), "M": len(rows_m), "R_seen": sum(r["category"] == "vu" for r in rows_r),
                   "R_new": len(new), "R_new_after_move": sum(r["after_move"] for r in new),
                   "M_12_23_seen_after": sum(r["category"] == "vu_apres" for r in changed),
                   "M_12_23_seen_before_only": sum(r["category"] == "vu_avant_seulement" for r in changed),
                   "M_16_23": len(late)},
    }


def transformer_l5(root, episodes=48):
    """L5: regime V of the own-action channel on the first lives of its jeu R, commands new after a move."""
    from .text_atelier import TextModel
    root = Path(root)
    per_model, pooled = {}, []
    for model_path in sorted(root.glob("model-V-*.npz")):
        tag = model_path.stem.replace("model-", "")
        lives = read_rows(root / f"lives-{tag}-R.jsonl")[:episodes]
        rows = transformer_rows(TextModel.load(model_path), lives)
        for r in rows:
            r["after_move"] = any(a < N_MOVE for a in lives[r["life"]]["actions"][:r["step"]])
        new = [r for r in rows if r["category"] == "nouveau" and r["after_move"]]
        per_model[tag] = {"accuracy": accuracy(new), "rows": len(new),
                          "accuracy_not_before_any": accuracy([r for r in rows if r["category"] == "nouveau" and not r["before_any"]])}
        pooled += new
    return {"accuracy": accuracy(pooled), "rows": len(pooled), "models": per_model}


def latent_verdicts(run, l5=None):
    run = Path(run)
    c = cells(read_rows(run / "rows-R.jsonl"), read_rows(run / "rows-M.jsonl"))
    valid = at_least(c["digit_mass"], VALIDITY_MASS)
    out = {"valid": valid, "cells": c, "L5_values": l5}
    if valid:
        out["L1"] = at_least(c["R_seen"], 0.70)
        out["L2"] = at_most(c["R_new"], 0.40)
        out["L3"] = at_least(c["M_12_23_seen_after"], 0.60) and at_most(c["M_12_23_seen_before_only"], 0.40)
        out["L4"] = at_least(c["R_confidence_when_wrong"], 0.50)
        out["global"] = out["L1"] and out["L2"] and out["L3"]
    else:
        out.update({k: None for k in ("L1", "L2", "L3", "L4", "global")})
    out["L5"] = at_least(l5["accuracy"], 0.90) if l5 else None
    return out


VAL_LOSS = re.compile(r"Iter (\d+): Val loss ([0-9.]+)")


def validation_losses(log_path):
    path = Path(log_path)
    if not path.exists():
        return []
    return [(int(i), float(v)) for i, v in VAL_LOSS.findall(path.read_text())]


def lora_verdicts(run):
    run = Path(run)
    models, valid = {}, {}
    for label in ("base", "F", "VM"):
        if (run / label / "rows-R.jsonl").exists():
            models[label] = cells(read_rows(run / label / "rows-R.jsonl"), read_rows(run / label / "rows-M.jsonl"))
            valid[label] = at_least(models[label]["digit_mass"], VALIDITY_MASS)
    losses = {label: validation_losses(run / f"lora-{label}.log") for label in ("F", "VM")}
    out = {"valid": valid, "cells": models, "validation_losses": losses}
    final = {k: v[-1][1] for k, v in losses.items() if v}
    out["budget_sufficient"] = final["VM"] < final["F"] if len(final) == 2 else None

    def cell(label, key):
        return models[label][key] if valid.get(label) else None

    def verdict(*labels):
        return all(valid.get(label) for label in labels)

    out["A1"] = at_most(cell("base", "R_new"), 0.40) if verdict("base") else None
    out["A2"] = (at_most(cell("F", "R_new"), 0.40) and at_least(cell("F", "R_new_confidence_when_wrong"), 0.60)) if verdict("F") else None
    out["A2_confidence_when_wrong_all_R"] = cell("F", "R_confidence_when_wrong")
    out["A3"] = at_least(cell("VM", "R_new_after_move"), 0.80) if verdict("VM") else None
    out["A4"] = (at_least(cell("VM", "M_16_23"), 0.70) and at_most(cell("F", "M_16_23"), 0.40)) if verdict("VM", "F") else None
    a5 = {}
    for label in ("base", "F", "VM"):
        if label == "F" and out["A2"]:
            continue
        a5[label] = at_least(cell(label, "R_seen"), 0.60) if valid.get(label) else None
    out["A5"] = all(a5.values()) if a5 and None not in a5.values() else None
    out["A5_models"] = a5
    out["global"] = bool(out["A1"] and out["A2"] and out["A3"]) if None not in (out["A1"], out["A2"], out["A3"]) else None
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent", default=None, help="run directory of menia-latent-mac (rows-R.jsonl, rows-M.jsonl)")
    parser.add_argument("--lora", default=None, help="run directory of menia-lora-mac (base/, F/, VM/, lora-*.log)")
    parser.add_argument("--own-action-root", default="artifacts/own-action-channel")
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--output", default=None)
    parser.add_argument("--check", default=None, help="published verdicts to compare with; exit 1 if the recomputation differs")
    a = parser.parse_args(argv)
    out = {}
    if a.latent:
        out["latent"] = latent_verdicts(a.latent, transformer_l5(a.own_action_root, a.episodes))
    if a.lora:
        out["lora"] = lora_verdicts(a.lora)
    out = json.loads(json.dumps(out))
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if a.output:
        Path(a.output).write_text(text + "\n")
    print(text)
    if a.check:
        published = json.loads(Path(a.check).read_text())
        differing = [k for k in out if published.get(k) != out[k]]
        print("differs:", differing or "none")
        if differing:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
