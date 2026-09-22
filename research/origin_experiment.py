"""Run the pre-registered origin-inquiry plan: docs/ORIGIN_INQUIRY_PROTOCOL.md."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from .origin_env import Atelier, CONDITIONS, N_MOVE, N_INSPECT, LIFE
from .origin_neural import (WorldModel, train_world_model, run_lives, inspection_shares, hits_per_life,
                            probe_analysis, intervention_analysis, decide, POLICIES)
from .origin_rational import OracleBayes

SEEDS = (17, 29, 43)
TEST_SEED = 910001
CHECKPOINT_SEED = 777001
CHECKPOINTS = (25, 100, 300, 700, 1500, 4000, 8000)


def oracle_lives(condition, policy, seed, count):
    rng = np.random.default_rng(seed * 31 + 7)
    lives = []
    for n in range(count):
        env = Atelier(condition, seed * 1000003 + n)
        obs = env.reset()
        oracle = OracleBayes(condition)
        oracle.observe_shown(obs["d_shown"])
        record = {"d": env.d, "e": env.e, "actions": [], "rewards": []}
        for t in range(LIFE):
            gains = np.array([list(oracle.eig(k, obs["p"], obs["s"])) + [oracle.cue_entropy(k)] for k in range(N_INSPECT)])
            hit = oracle.hit_probabilities(obs["p"], obs["g"])
            dist = oracle.expected_distance(obs["p"], obs["g"])
            action = decide(policy, gains, hit, dist, rng)
            p_prev, s_prev = obs["p"], obs["s"]
            obs, reward, _ = env.step(action)
            if action < N_MOVE:
                oracle.observe_move(p_prev, action, obs["p"])
            else:
                oracle.observe_cue(action - N_MOVE, obs["cue_value"])
            oracle.observe_sky(s_prev, obs["s"])
            record["actions"].append(int(action))
            record["rewards"].append(int(reward))
        lives.append(record)
    return lives


def random_inspector(condition, seed, count):
    rng = np.random.default_rng(seed)
    lives = []
    for n in range(count):
        env = Atelier(condition, seed * 1000003 + n)
        env.reset()
        record = {"d": env.d, "e": env.e, "actions": [], "rewards": []}
        for t in range(LIFE):
            action = int(rng.integers(N_MOVE + N_INSPECT)) if rng.random() < 0.5 else int(rng.integers(N_MOVE))
            _, reward, _ = env.step(action)
            record["actions"].append(action)
            record["rewards"].append(int(reward))
        lives.append(record)
    return lives


def summarize(lives):
    out = inspection_shares(lives)
    out["hits_per_life"] = hits_per_life(lives)
    return out


def write_jsonl(path, lives):
    with open(path, "w") as f:
        for life in lives:
            f.write(json.dumps(life, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--updates", type=int, default=8000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--test-lives", type=int, default=300)
    parser.add_argument("--checkpoint-lives", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    tag = "-".join(args.conditions)
    for condition in args.conditions:
        if (root / f"report-{condition}.json").exists() or any(root.glob(f"model-{condition}-*.json")):
            raise ValueError(f"Condition {condition} already has results in {root}; use a new destination")
    checkpoints = tuple(c for c in CHECKPOINTS if c <= args.updates) + ((args.updates,) if args.updates not in CHECKPOINTS else ())
    start = time.perf_counter()
    report = {"status": "executed CPU experiment", "protocol": "docs/ORIGIN_INQUIRY_PROTOCOL.md",
              "python": platform.python_version(), "numpy": np.__version__,
              "arguments": vars(args), "test_seed": TEST_SEED, "checkpoint_seed": CHECKPOINT_SEED,
              "checkpoints": list(checkpoints), "runs": [], "references": {}}
    for condition in args.conditions:
        for seed in args.seeds:
            t0 = time.perf_counter()
            trajectory = []

            def on_checkpoint(n, model, _condition=condition, _trajectory=trajectory):
                lives = run_lives(model, _condition, "self", CHECKPOINT_SEED, args.checkpoint_lives)
                _trajectory.append({"update": n, **summarize(lives)})
                print(json.dumps({"condition": _condition, "seed": seed, "checkpoint": n, **summarize(lives)}), flush=True)

            model, log = train_world_model(condition, seed, updates=args.updates, hidden=args.hidden, lr=args.lr,
                                           on_checkpoint=on_checkpoint, checkpoints=checkpoints)
            weights = root / f"model-{condition}-{seed}.json"
            model.save(weights, {"condition": condition, "seed": seed, "updates": args.updates,
                                 "task": "next-observation prediction only; no D or E label",
                                 "subjective_consciousness": "not tested"})
            model = WorldModel.load(weights)
            run = {"condition": condition, "seed": seed, "parameters": model.parameter_count,
                   "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                   "training": log, "trajectory": trajectory, "policies": {}}
            for policy in POLICIES:
                keep = policy == "self"
                result = run_lives(model, condition, policy, TEST_SEED, args.test_lives, keep_states=keep)
                lives, states = result if keep else (result, None)
                path = root / f"lives-{condition}-{seed}-{policy}.jsonl"
                write_jsonl(path, lives)
                entry = summarize(lives)
                entry["log_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                if keep:
                    entry["probe"] = probe_analysis(states, lives)
                    entry["intervention"] = intervention_analysis(model, states, lives)
                run["policies"][policy] = entry
            run["train_seconds"] = time.perf_counter() - t0
            report["runs"].append(run)
            print(json.dumps({"condition": condition, "seed": seed, "policies": {
                k: {kk: vv for kk, vv in v.items() if kk in ("share", "per_life", "hits_per_life")} for k, v in run["policies"].items()}}), flush=True)
        report["references"][condition] = {
            "oracle": {policy: summarize(oracle_lives(condition, policy, TEST_SEED, args.test_lives)) for policy in POLICIES},
            "random_inspector": summarize(random_inspector(condition, TEST_SEED, args.test_lives))}
        print(json.dumps({"condition": condition, "references": report["references"][condition]}), flush=True)
    report["wall_seconds"] = time.perf_counter() - start
    report["source_sha256"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                               ("research/origin_env.py", "research/origin_neural.py", "research/origin_rational.py",
                                "research/origin_experiment.py")}
    report["limitations"] = ["synthetic sixteen-state world", "few-thousand-parameter agent", "myopic decision rule",
                             "not a language model", "not a consciousness test", "inquiry is inspection allocation only"]
    (root / f"report-{tag}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
