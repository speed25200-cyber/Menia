"""Independent recomputation of the own-action-channel reports and evaluation of predictions P1-P5.

Everything is recomputed from the logged token sequences and the saved weights. Hidden values
are only read from the logs, and the logs are checked by replaying each life from its seed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .origin_env import LIFE, N_MOVE
from .origin_neural import decide, HIT_THRESHOLD
from .text_atelier import (TextModel, TextLife, TextImagination, replay_tokens, displacement_table,
                           displacement_summary, inquiry_summary, confidence_by_steps, STABLE_REGIMES)
from .own_action_experiment import SETS, CHANGE_STEP, sha256

TOL = 1e-6


def read_lives(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def integrity(lives):
    """Every logged life must replay exactly from its seed and actions."""
    for life in lives:
        tokens, d0, d1 = replay_tokens(life)
        if tokens != life["tokens"] or d0 != life["d"] or ("d_final" in life and d1 != life["d_final"]):
            return False
        if "change_step" in life and life["d_final"] == life["d"]:
            return False
    return True


def replay_decisions(model, lives, count):
    """Recompute the P-soi decisions of the first `count` lives with the same rule and random stream."""
    imagination = TextImagination(model)
    seed = lives[0]["seed"]
    rng = np.random.default_rng(seed * 31 + 7)
    for life in lives[:count]:
        replay = TextLife("T", seed * 1000003 + life["index"], forced_change_step=life.get("change_step"))
        for t in range(LIFE):
            hit, expected_distance, _ = imagination.move_predictions(replay.tokens, replay.obs)
            gains = np.zeros((4, 3))
            if hit.max() < HIT_THRESHOLD:
                gains = imagination.inspection_gains(replay.tokens, replay.obs)
            action = decide("self", gains, hit, expected_distance, rng)
            if action != life["actions"][t]:
                return False
            if max(abs(float(g) - float(v)) for g, v in zip(gains[:, 0], life["gains"][t])) > 1e-4:
                return False
            replay.step(action)
    return True


def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= TOL


def compare(recomputed, reported):
    """Flat comparison of every numeric leaf of the recomputed summaries with the report."""
    problems = []

    def walk(x, y, path):
        if isinstance(x, dict):
            for k in x:
                if k not in y:
                    problems.append(path + "/" + k + " missing")
                else:
                    walk(x[k], y[k], path + "/" + k)
        elif isinstance(x, (int, float)) or x is None:
            if not close(x, y):
                problems.append(f"{path}: {x} vs {y}")

    walk(recomputed, reported, "")
    return problems


def pooled(models, regime, setname):
    """Rows and lives of one regime pooled over its seeds."""
    rows, lives = [], []
    for m in models:
        if m["regime"] == regime:
            if setname in m["rows"]:
                rows += m["rows"][setname]
            lives += m["lives"][setname]
    return rows, lives


REGIMES = STABLE_REGIMES


def criteria(models):
    """Validity then P1-P5 on the pooled measures; directions per seed where the protocol says so."""
    models = [m for m in models if m["regime"] in REGIMES]
    by_regime = {r: [m for m in models if m["regime"] == r] for r in REGIMES}
    complete = all(len(by_regime[r]) == 3 for r in REGIMES)
    values = {}
    for r in REGIMES:
        rowsR, livesR = pooled(models, r, "R")
        rowsM, livesM = pooled(models, r, "M")
        _, livesC = pooled(models, r, "C")
        values[r] = {
            "R": displacement_summary(rowsR) if rowsR else None,
            "M_displacement": displacement_summary(rowsM) if rowsM else None,
            "M_inquiry": inquiry_summary(livesM, CHANGE_STEP) if livesM else None,
            "C_inquiry": inquiry_summary(livesC, CHANGE_STEP) if livesC else None,
            "MC_inquiry": inquiry_summary(livesM + livesC, CHANGE_STEP) if livesM or livesC else None,
            "per_seed": {str(m["seed"]): {
                "accuracy_after_first": m["summaries"]["R"]["displacement"]["accuracy_after_first"],
                "mark_share_MC": inquiry_summary(m["lives"]["M"] + m["lives"]["C"], CHANGE_STEP)["mark_share"]}
                for m in by_regime[r]},
        }
    out = {"complete": complete, "values": values}
    if not complete:
        out.update({k: False for k in ("validity", "P1", "P2", "P3", "P4", "P5", "global")})
        return out

    def acc(r):
        return values[r]["R"]["accuracy_after_first"]

    def share(r):
        return values[r]["MC_inquiry"]["mark_share"] or 0.0

    def per_seed(r, key):
        return {s: (v[key] or 0.0) for s, v in values[r]["per_seed"].items()}

    out["validity"] = all((values[r]["R"]["accuracy_after_first_fixed_body"] or 0.0) >= 0.90 for r in ("F", "V"))
    out["P1"] = (acc("V") >= 0.90 and acc("VE") >= 0.90 and acc("F") <= 0.40
                 and all(per_seed("V", "accuracy_after_first")[s] > per_seed("F", "accuracy_after_first")[s] for s in per_seed("V", "accuracy_after_first")))
    conf_wrong_F = values["F"]["R"]["confidence_when_wrong_after_first"]
    conf_before_V = values["V"]["R"]["confidence_before_any"]
    out["P2"] = conf_wrong_F is not None and conf_wrong_F >= 0.60 and conf_before_V is not None and conf_before_V <= 0.40
    out["P3"] = (all(values[r]["MC_inquiry"]["inspections_per_life"] >= 1.0 and share(r) >= 0.50 for r in ("V", "VE"))
                 and values["F"]["MC_inquiry"]["inspections_per_life"] <= 0.30
                 and all(per_seed("V", "mark_share_MC")[s] > per_seed("F", "mark_share_MC")[s] for s in per_seed("V", "mark_share_MC")))
    out["P4"] = (all(values[r]["M_inquiry"]["lives_reading_mark_after_step"] >= 0.50 for r in ("V", "VE"))
                 and all(values[r]["C_inquiry"]["lives_reading_mark_after_step"] <= 0.10 for r in ("V", "VE"))
                 and values["F"]["M_inquiry"]["lives_reading_mark_after_step"] <= 0.10
                 and all((values[r]["M_displacement"]["accuracy_steps_16_23"] or 0.0) >= 0.80 for r in ("V", "VE"))
                 and (values["F"]["M_displacement"]["accuracy_steps_16_23"] or 0.0) <= 0.40)
    out["P5"] = abs(acc("V") - acc("VE")) <= 0.05 and abs(share("V") - share("VE")) <= 0.15
    out["global"] = out["validity"] and out["P1"] and out["P3"] and out["P4"]
    return out


def mutable_criteria(models):
    """Q1-Q5 of docs/OWN_ACTION_MUTABLE_PROTOCOL.md: regime VM against regime V."""
    by = {r: [m for m in models if m["regime"] == r] for r in ("V", "VM")}
    complete = all(len(by[r]) == 3 for r in by) and {m["seed"] for m in by["V"]} == {m["seed"] for m in by["VM"]}
    values = {}
    for r in by:
        rowsR, _ = pooled(models, r, "R")
        rowsM, livesM = pooled(models, r, "M")
        _, livesC = pooled(models, r, "C")
        values[r] = {
            "R": displacement_summary(rowsR) if rowsR else None,
            "M_displacement": displacement_summary(rowsM) if rowsM else None,
            "M_inquiry": inquiry_summary(livesM, CHANGE_STEP) if livesM else None,
            "C_inquiry": inquiry_summary(livesC, CHANGE_STEP) if livesC else None,
            "MC_inquiry": inquiry_summary(livesM + livesC, CHANGE_STEP) if livesM or livesC else None,
            "confidence_9_11": confidence_by_steps(rowsM, (9, 10, 11)) if rowsM else None,
            "confidence_13_15": confidence_by_steps(rowsM, (13, 14, 15)) if rowsM else None,
            "per_seed": {str(m["seed"]): {
                "accuracy_steps_16_23": m["summaries"]["M"]["displacement"]["accuracy_steps_16_23"],
                "lives_reading_mark_after_step": m["summaries"]["M"]["inquiry"]["lives_reading_mark_after_step"]}
                for m in by[r]},
        }
    out = {"complete": complete, "values": values}
    if not complete:
        out.update({k: False for k in ("Q1", "Q2", "Q3", "Q4", "Q5", "global")})
        return out

    def seeds(r, key):
        return {s: (v[key] or 0.0) for s, v in values[r]["per_seed"].items()}
    vm, v = values["VM"], values["V"]
    out["Q1"] = (vm["R"]["accuracy_after_first"] or 0.0) >= 0.90
    out["Q2"] = ((vm["M_displacement"]["accuracy_steps_16_23"] or 0.0) >= 0.80
                 and all(seeds("VM", "accuracy_steps_16_23")[s] > seeds("V", "accuracy_steps_16_23")[s] for s in seeds("VM", "accuracy_steps_16_23")))
    out["Q3"] = (vm["M_inquiry"]["lives_reading_mark_after_step"] >= 0.50
                 and vm["M_inquiry"]["lives_reading_mark_after_step"] - vm["C_inquiry"]["lives_reading_mark_after_step"] >= 0.30
                 and all(seeds("VM", "lives_reading_mark_after_step")[s] > seeds("V", "lives_reading_mark_after_step")[s] for s in seeds("VM", "lives_reading_mark_after_step")))
    out["Q4"] = vm["MC_inquiry"]["inspections_per_life"] >= 1.0 and (vm["MC_inquiry"]["mark_share"] or 0.0) >= 0.50
    drop_vm = (vm["confidence_9_11"] or 0.0) - (vm["confidence_13_15"] or 0.0)
    drop_v = (v["confidence_9_11"] or 0.0) - (v["confidence_13_15"] or 0.0)
    out["Q5"] = drop_vm >= 0.20 and drop_v <= 0.05
    out["confidence_drop"] = {"VM": drop_vm, "V": drop_v}
    out["global"] = out["Q1"] and out["Q2"] and out["Q3"]
    return out


def load_models(root, replay_lives=0, log=print):
    root = Path(root)
    models, problems = [], []
    for report_path in sorted(root.glob("report-*.json")):
        report = json.loads(report_path.read_text())
        tag = f"{report['regime']}-{report['seed']}"
        model_path = root / f"model-{tag}.npz"
        if sha256(model_path) != report["files"]["model"]:
            problems.append(f"{tag}: model hash")
        model = TextModel.load(model_path)
        entry = {"regime": report["regime"], "seed": report["seed"], "lives": {}, "rows": {}, "summaries": {}, "report": report}
        for name in SETS:
            path = root / f"lives-{tag}-{name}.jsonl"
            if sha256(path) != report["files"][f"lives_{name}"]:
                problems.append(f"{tag}: lives {name} hash")
            lives = read_lives(path)
            if not integrity(lives):
                problems.append(f"{tag}: replay integrity {name}")
            entry["lives"][name] = lives
            summary = {"inquiry": inquiry_summary(lives, CHANGE_STEP)}
            if name in ("R", "M"):
                entry["rows"][name] = displacement_table(model, lives)
                summary["displacement"] = displacement_summary(entry["rows"][name])
            entry["summaries"][name] = summary
            if name in ("M", "C") and replay_lives and not replay_decisions(model, lives, replay_lives):
                problems.append(f"{tag}: decision replay {name}")
        problems += [f"{tag}{p}" for p in compare(entry["summaries"], report["sets"])]
        models.append(entry)
        log(f"audited {tag}")
    return models, problems


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/own-action-channel")
    parser.add_argument("--replay-lives", type=int, default=0, help="P-soi lives per set to replay decision by decision")
    parser.add_argument("--check", action="store_true", help="exit 1 if any recomputation disagrees with the reports")
    parser.add_argument("--output", default=None)
    parser.add_argument("--mutable-root", default=None, help="root of the VM regime; evaluates Q1-Q5 against regime V of --root")
    args = parser.parse_args(argv)
    models, problems = load_models(args.root, args.replay_lives)
    verdict = criteria(models)
    out = {"problems": problems, "models": [f"{m['regime']}-{m['seed']}" for m in models], "criteria": verdict}
    if args.mutable_root:
        mutable, more = load_models(args.mutable_root, args.replay_lives)
        problems += more
        out["mutable_models"] = [f"{m['regime']}-{m['seed']}" for m in mutable]
        out["mutable_criteria"] = mutable_criteria([m for m in models if m["regime"] == "V"] + mutable)
        print(json.dumps({k: v for k, v in out["mutable_criteria"].items() if k != "values"}, indent=1))
    path = Path(args.output or Path(args.mutable_root or args.root) / "verification.json")
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in verdict.items() if k != "values"}, indent=1))
    print("problems:", problems or "none")
    if args.check and problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
