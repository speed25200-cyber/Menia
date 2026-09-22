"""Run the pre-registered mutable-body plan: docs/MUTABLE_BODY_PROTOCOL.md."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from .origin_env import LIFE, N_MOVE
from .origin_neural import (WorldModel, train_world_model, run_lives, fit_probe, probe_predict)
from .origin_rl import QNetwork, greedy_lives, feature_size

SEEDS = (17, 29, 43)
TEST_SEED = 910001
CONTROL_SEED = 910002
CHANGE_STEP = 12
REGIMES = {"S": (None, 0.0), "MS": ("self", 0.5), "MW": ("world", 0.5), "MS10": ("self", 0.1)}


def mark_reads_after(lives, step=CHANGE_STEP):
    return float(np.mean([sum(1 for a in life["actions"][step:] if a == N_MOVE) for life in lives]))


def hits_ratio(lives, step=CHANGE_STEP):
    before = sum(sum(life["rewards"][:step]) for life in lives)
    after = sum(sum(life["rewards"][step:]) for life in lives)
    return {"hits_before_per_life": before / len(lives), "hits_after_per_life": after / len(lives),
            "ratio": (after / before) if before else None}


def entropy_jump(lives, step=CHANGE_STEP):
    """Mean motion entropy over steps step+1..step+3 minus steps step-3..step-1 (indices)."""
    after = np.mean([np.mean(life["motion_entropy"][step + 1:step + 4]) for life in lives])
    before = np.mean([np.mean(life["motion_entropy"][step - 3:step]) for life in lives])
    return {"before": float(before), "after": float(after), "jump": float(after - before)}


def update_probe(control_states, control_lives, change_states, change_lives, first=16):
    """Linear readout of D fitted on control lives (all steps), applied to the new body after the change."""
    d_control = np.repeat([life["d"] for life in control_lives], LIFE)
    W = fit_probe(control_states.reshape(-1, control_states.shape[-1]), d_control)
    pred = probe_predict(W, change_states[:, first:].reshape(-1, change_states.shape[-1])).reshape(len(change_lives), LIFE - first)
    new = np.array([life["d_final"] for life in change_lives])[:, None]
    old = np.array([life["d"] for life in change_lives])[:, None]
    return {"new_body_accuracy": float((pred == new).mean()), "old_body_accuracy": float((pred == old).mean()),
            "by_step_new": [float(v) for v in (pred == new).mean(axis=0)],
            "control_accuracy_last_step": float((probe_predict(W, control_states[:, -1]) == np.array([l["d"] for l in control_lives])).mean())}


def write_jsonl(path, lives):
    with open(path, "w") as f:
        for life in lives:
            f.write(json.dumps(life, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--regimes", nargs="+", default=["S", "MS", "MW"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--stable-models", default="artifacts/origin-inquiry")
    parser.add_argument("--policies", default="artifacts/learned-inquiry")
    parser.add_argument("--updates", type=int, default=8000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lives", type=int, default=300)
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    for regime in args.regimes:
        if (root / f"report-mutable-{regime}.json").exists():
            raise ValueError(f"Regime {regime} already has results in {root}")
    start = time.perf_counter()
    for regime in args.regimes:
        mutate, probability = REGIMES[regime]
        report = {"status": "executed CPU experiment", "protocol": "docs/MUTABLE_BODY_PROTOCOL.md", "regime": regime,
                  "mutate": mutate, "mutation_probability": probability, "python": platform.python_version(),
                  "numpy": np.__version__, "arguments": vars(args), "test_seed": TEST_SEED, "control_seed": CONTROL_SEED,
                  "change_step": CHANGE_STEP, "runs": []}
        for seed in args.seeds:
            t0 = time.perf_counter()
            if regime == "S":
                weights = Path(args.stable_models) / f"model-T-{seed}.json"
                model, log = WorldModel.load(weights), None
            else:
                model, log = train_world_model("T", seed, updates=args.updates, hidden=args.hidden, mutate=mutate,
                                               mutation_probability=probability)
                weights = root / f"model-{regime}-{seed}.json"
                model.save(weights, {"regime": regime, "mutate": mutate, "mutation_probability": probability, "seed": seed,
                                     "updates": args.updates, "task": "next-observation prediction only; no D or E label"})
                model = WorldModel.load(weights)
            run = {"regime": regime, "seed": seed, "weights": str(weights),
                   "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "training": log, "lives": {}}
            data = {}
            for name, policy, seed_base, change in (("change-self", "self", TEST_SEED, CHANGE_STEP), ("change-none", "none", TEST_SEED, CHANGE_STEP),
                                                    ("control-self", "self", CONTROL_SEED, None), ("control-none", "none", CONTROL_SEED, None)):
                lives, states = run_lives(model, "T", policy, seed_base, args.lives, keep_states=True,
                                          forced_change_step=change, record_entropy=True)
                path = root / f"lives-{regime}-{seed}-{name}.jsonl"
                write_jsonl(path, lives)
                data[name] = (lives, states)
                run["lives"][name] = {"log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "mark_reads_after_change_step": mark_reads_after(lives), "hits": hits_ratio(lives),
                                      "entropy": entropy_jump(lives)}
            run["M1_update"] = update_probe(data["control-none"][1], data["control-none"][0], data["change-none"][1], data["change-none"][0])
            run["M2_detection"] = run["lives"]["change-none"]["entropy"]
            run["M3_mark_reads_after_change"] = run["lives"]["change-self"]["mark_reads_after_change_step"]
            run["M3_control_mark_reads_after_step12"] = run["lives"]["control-self"]["mark_reads_after_change_step"]
            run["M4_recovery"] = run["lives"]["change-self"]["hits"]
            # Exploratory: learned prudence policy trained on the stable regime, evaluated on change lives.
            policy_path = Path(args.policies) / f"policy-T-{seed}-prudence-3.json"
            if policy_path.exists() and QNetwork.load(policy_path).p["W1"].shape[0] == feature_size(model.hidden):
                q = QNetwork.load(policy_path)
                lives = greedy_lives(model, q, "T", TEST_SEED, args.lives, forced_change_step=CHANGE_STEP)
                path = root / f"lives-{regime}-{seed}-change-learned.jsonl"
                write_jsonl(path, lives)
                run["lives"]["change-learned"] = {"log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                                  "mark_reads_after_change_step": mark_reads_after(lives), "hits": hits_ratio(lives),
                                                  "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest()}
            run["seconds"] = time.perf_counter() - t0
            report["runs"].append(run)
            print(json.dumps({"regime": regime, "seed": seed, "M1": round(run["M1_update"]["new_body_accuracy"], 3),
                              "M2": round(run["M2_detection"]["jump"], 3), "M3": round(run["M3_mark_reads_after_change"], 3),
                              "M3_control": round(run["M3_control_mark_reads_after_step12"], 3),
                              "M4": run["M4_recovery"]["ratio"] and round(run["M4_recovery"]["ratio"], 3),
                              "learned": run["lives"].get("change-learned", {}).get("mark_reads_after_change_step")}), flush=True)
        report["wall_seconds"] = time.perf_counter() - start
        report["source_sha256"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                                   ("research/origin_env.py", "research/origin_neural.py", "research/mutable_experiment.py")}
        (root / f"report-mutable-{regime}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
