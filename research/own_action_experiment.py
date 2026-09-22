"""Run the pre-registered own-action-channel plan: docs/OWN_ACTION_CHANNEL_PROTOCOL.md."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from .text_atelier import (REGIMES, TextModel, train_text_model, run_text_lives, displacement_table,
                           displacement_summary, inquiry_summary)

SEEDS = (17, 29, 43)
RANDOM_SEED = 920001
CHANGE_SEED = 920002
CONTROL_SEED = 920003
CHANGE_STEP = 12
SETS = {"R": ("random", RANDOM_SEED, None), "M": ("self", CHANGE_SEED, CHANGE_STEP), "C": ("self", CONTROL_SEED, None)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_lives(path, lives):
    Path(path).write_text("".join(json.dumps(life) + "\n" for life in lives))


def summarize(model, lives_by_set):
    out = {}
    for name, lives in lives_by_set.items():
        entry = {"inquiry": inquiry_summary(lives, after_step=CHANGE_STEP)}
        if name in ("R", "M"):
            entry["displacement"] = displacement_summary(displacement_table(model, lives))
        out[name] = entry
    return out


def run_one(root, regime, seed, updates, lives, log=print):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tag = f"{regime}-{seed}"
    started = time.time()
    model, history = train_text_model(regime, seed, updates=updates,
                                      on_log=lambda h: log(f"[{tag}] update {h['update']} loss {h['loss']} grad {h['grad_norm']}"))
    trained = time.time()
    model_path = root / f"model-{tag}.npz"
    model.save(model_path, {"regime": regime, "updates": updates, "protocol": "docs/OWN_ACTION_CHANNEL_PROTOCOL.md"})
    lives_by_set, files, timing = {}, {"model": sha256(model_path)}, {"training_seconds": round(trained - started, 1)}
    for name, (policy, test_seed, change) in SETS.items():
        t0 = time.time()
        played = run_text_lives(model, policy, test_seed, lives, forced_change_step=change,
                                on_life=lambda n, life: log(f"[{tag}] set {name} life {n}") if n % 50 == 0 else None)
        path = root / f"lives-{tag}-{name}.jsonl"
        write_lives(path, played)
        lives_by_set[name] = played
        files[f"lives_{name}"] = sha256(path)
        timing[f"set_{name}_seconds"] = round(time.time() - t0, 1)
    report = {"regime": regime, "seed": seed, "updates": updates, "lives_per_set": lives, "parameters": model.parameter_count(),
              "history": history, "sets": summarize(model, lives_by_set), "files": files, "timing": timing,
              "machine": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__},
              "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (root / f"report-{tag}.json").write_text(json.dumps(report, indent=1))
    log(f"[{tag}] done in {round(time.time() - started, 1)} s")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/own-action-channel")
    parser.add_argument("--regime", choices=REGIMES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--updates", type=int, default=6000)
    parser.add_argument("--lives", type=int, default=200)
    parser.add_argument("--smoke", action="store_true", help="tiny budget, seed 1, crash check only; never read")
    args = parser.parse_args(argv)
    if args.smoke:
        for regime in REGIMES:
            run_one(args.root, regime, 1, updates=3, lives=2)
        return
    regimes = [args.regime] if args.regime else list(REGIMES)
    seeds = [args.seed] if args.seed else list(SEEDS)
    for regime in regimes:
        for seed in seeds:
            run_one(args.root, regime, seed, args.updates, args.lives)


if __name__ == "__main__":
    main()
