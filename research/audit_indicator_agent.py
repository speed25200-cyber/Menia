"""Independent audit of the indicator agent: hashes, replay of logged lives, recomputation of the criteria.

It reloads the learned parameters, replays lives of the full agent from their seeds and checks
every action, spotlight intent, workspace writer, return and faint count against the log, can
re-run every variant and compare the summary digests (--full), recomputes the quality space
from the weights, and recomputes the fourteen verdicts from the per-seed reports.
"""
import argparse
import hashlib
import json
from pathlib import Path
from .indicator_agent import Params, VARIANTS
from .indicator_experiment import SETS, run_life, life_seed, evaluate, criteria, quality_space


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def replay(params, seed, set_index, name, record):
    base, mode = SETS[name]
    life = run_life(params, "agent", life_seed(base, record["life"]), mode, seed * 1_000_000 + set_index * 10_000 + record["life"])
    steps = life["steps"]
    return ([r["action"] for r, _ in steps] == record["actions"]
            and [r["next_intent"] for r, _ in steps] == record["intents"]
            and ["".join(m[0] for m in r["writers"]) for r, _ in steps] == record["writers"]
            and round(float(sum(life["rewards"])), 6) == record["return"] and life["faints"] == record["faints"])


def audit(root, replay_lives=2, full=False, log=print):
    root = Path(root)
    problems, reports = [], []
    for report_path in sorted(root.glob("report-*.json")):
        report = json.loads(report_path.read_text())
        seed = report["seed"]
        for name, digest in report["files"].items():
            if sha256(root / name) != digest:
                problems.append(f"{seed}: hash of {name}")
        params = Params.load(root / f"params-{seed}.json")
        space = quality_space(params.hue_code)
        if space != report["quality_space"]:
            problems.append(f"{seed}: quality space differs")
        for set_index, name in enumerate(SETS):
            records = read_jsonl(root / f"lives-{seed}-{name}.jsonl")
            for record in records[:replay_lives]:
                if not replay(params, seed, set_index, name, record):
                    problems.append(f"{seed}: replay of set {name} life {record['life']}")
        if full:
            evaluation, _ = evaluate(params, seed, lives=report["settings"]["test_lives"], log=lambda m: None)
            for variant in VARIANTS:
                for name in SETS:
                    if evaluation[variant][name] != report["evaluation"][variant][name]:
                        problems.append(f"{seed}: evaluation of {variant} on {name}")
        reports.append(report)
        log(f"audited seed {seed}")
    verdict, values = criteria(reports)
    published = json.loads((root / "criteria.json").read_text())
    if published["criteria"] != verdict:
        problems.append("criteria differ from criteria.json")
    return {"problems": problems, "criteria": verdict, "values": values, "replay_lives": replay_lives, "full": full}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/indicator-agent")
    parser.add_argument("--replay-lives", type=int, default=2)
    parser.add_argument("--full", action="store_true", help="re-run every variant on every test life")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", default=None)
    a = parser.parse_args(argv)
    out = audit(a.root, a.replay_lives, a.full)
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out["criteria"].items()}, indent=1))
    print("problems:", out["problems"] or "none")
    if a.check and out["problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
