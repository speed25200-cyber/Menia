"""Run the pre-registered learned-inquiry plan: docs/LEARNED_INQUIRY_PROTOCOL.md."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from .origin_neural import WorldModel
from .origin_rl import train_policy, greedy_lives, summarize

SEEDS = (17, 29, 43)
TEST_SEED = 910001
ARMS_T = (("prudence", 0.0), ("prudence", 0.3), ("prudence", 1.0), ("prudence", 3.0), ("prudence", 10.0),
          ("information", 3.0), ("surprise", 3.0))
ARMS_CONTROL = (("prudence", 0.0), ("prudence", 3.0), ("prudence", 10.0))


def arm_name(kind, beta):
    return f"{kind}-{beta:g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--world-models", default="artifacts/origin-inquiry")
    parser.add_argument("--conditions", nargs="+", default=["T", "C1", "C3"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--test-lives", type=int, default=300)
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    tag = "-".join(args.conditions)
    for condition in args.conditions:
        if (root / f"report-rl-{condition}.json").exists():
            raise ValueError(f"Condition {condition} already has results in {root}")
    start = time.perf_counter()
    report = {"status": "executed CPU experiment", "protocol": "docs/LEARNED_INQUIRY_PROTOCOL.md",
              "python": platform.python_version(), "numpy": np.__version__, "arguments": vars(args),
              "test_seed": TEST_SEED, "runs": []}
    for condition in args.conditions:
        arms = ARMS_T if condition == "T" else ARMS_CONTROL
        for seed in args.seeds:
            weights = Path(args.world_models) / f"model-{condition}-{seed}.json"
            model = WorldModel.load(weights)
            world_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
            for kind, beta in arms:
                t0 = time.perf_counter()
                q, log, trajectory = train_policy(model, condition, seed, kind, beta, episodes=args.episodes)
                name = f"{condition}-{seed}-{arm_name(kind, beta)}"
                q_path = root / f"policy-{name}.json"
                q.save(q_path, {"condition": condition, "seed": seed, "reward": kind, "beta": beta,
                                "world_model_sha256": world_sha, "episodes": args.episodes})
                lives = greedy_lives(model, q, condition, TEST_SEED, args.test_lives)
                log_path = root / f"lives-{name}.jsonl"
                with open(log_path, "w") as f:
                    for life in lives:
                        f.write(json.dumps(life, separators=(",", ":")) + "\n")
                entry = {"condition": condition, "seed": seed, "reward": kind, "beta": beta,
                         "world_model_sha256": world_sha, "policy_sha256": hashlib.sha256(q_path.read_bytes()).hexdigest(),
                         "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
                         "evaluation": summarize(lives), "trajectory": trajectory, "training": log,
                         "train_seconds": time.perf_counter() - t0}
                report["runs"].append(entry)
                print(json.dumps({"run": name, "evaluation": entry["evaluation"], "seconds": round(entry["train_seconds"])}), flush=True)
    report["wall_seconds"] = time.perf_counter() - start
    report["source_sha256"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                               ("research/origin_env.py", "research/origin_neural.py", "research/origin_rl.py",
                                "research/origin_rl_experiment.py")}
    (root / f"report-rl-{tag}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
