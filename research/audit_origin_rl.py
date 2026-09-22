"""Independent recomputation of the learned-inquiry report and evaluation of criteria A-F."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .origin_neural import WorldModel
from .origin_rl import QNetwork, greedy_lives, summarize


def read_lives(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def criteria(runs):
    def select(condition, kind, beta):
        return [r for r in runs if r["condition"] == condition and r["reward"] == kind and abs(r["beta"] - beta) < 1e-9]
    def per_life(r): return r["evaluation"]["per_life"]
    def share(r, k): return r["evaluation"]["share"][k]
    out = {}
    def mark_reads(r): return per_life(r) * share(r, 0)
    a = [r for r in runs if r["reward"] == "prudence" and r["beta"] == 0]
    out["A_no_intrinsic_no_inquiry"] = bool(a) and all(per_life(r) < 0.50 for r in a)
    b_ok = True
    for beta in (3.0, 10.0):
        for r in select("T", "prudence", beta):
            base = [x for x in select("T", "prudence", 0.0) if x["seed"] == r["seed"]]
            b_ok &= per_life(r) >= 1.0 and share(r, 0) >= 0.75 and bool(base) and mark_reads(r) >= 5 * max(mark_reads(base[0]), 1e-9)
    out["B_dose_effect_T"] = b_ok and len(select("T", "prudence", 3.0) + select("T", "prudence", 10.0)) == 6
    monotone = True
    for seed in sorted({r["seed"] for r in runs if r["condition"] == "T"}):
        series = [per_life(r) for beta in (0.0, 0.3, 1.0, 3.0, 10.0) for r in select("T", "prudence", beta) if r["seed"] == seed]
        monotone &= len(series) == 5 and all(series[i + 1] >= series[i] - 0.25 for i in range(len(series) - 1))
    out["B_monotone_T"] = monotone
    c = [r for r in runs if r["condition"] == "C1"]
    out["C_origin_given_no_inquiry"] = bool(c) and all(per_life(r) < 0.50 for r in c)
    d = select("C3", "prudence", 3.0) + select("C3", "prudence", 10.0)
    out["D_no_trace_no_inquiry"] = len(d) == 6 and all(per_life(r) < 1.0 and share(r, 0) < 0.50 for r in d)
    e = select("T", "information", 3.0)
    out["E_information_bonus_insufficient"] = len(e) == 3 and all(per_life(r) < 0.50 for r in e)
    f = select("T", "surprise", 3.0)
    out["F_surprise_captured_by_noise"] = len(f) == 3 and all(share(r, 2) + share(r, 3) > 0.5 and r["evaluation"]["hits_per_life"] < 8 for r in f)
    out["global"] = all(out[k] for k in ("A_no_intrinsic_no_inquiry", "B_dose_effect_T", "B_monotone_T",
                                         "C_origin_given_no_inquiry", "D_no_trace_no_inquiry",
                                         "E_information_bonus_insufficient", "F_surprise_captured_by_noise"))
    out["transition_T_prudence_1"] = [(r["seed"], per_life(r), share(r, 0)) for r in select("T", "prudence", 1.0)]
    out["C3_moves_per_life_by_beta"] = {f"{beta:g}": [r["evaluation"]["moves_per_life"] for r in select("C3", "prudence", beta)] for beta in (0.0, 3.0, 10.0)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/learned-inquiry")
    parser.add_argument("--world-models", default="artifacts/origin-inquiry")
    parser.add_argument("--replay-lives", type=int, default=30)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    parts = sorted(root.glob("report-rl-*.json"))
    if not parts:
        raise FileNotFoundError("No report-rl-*.json in " + str(root))
    runs = []
    for part in parts:
        runs.extend(json.loads(part.read_text())["runs"])
    replayed = 0
    for r in runs:
        name = f"{r['condition']}-{r['seed']}-{r['reward']}-{r['beta']:g}"
        weights = Path(args.world_models) / f"model-{r['condition']}-{r['seed']}.json"
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == r["world_model_sha256"], weights
        q_path, log_path = root / f"policy-{name}.json", root / f"lives-{name}.jsonl"
        assert hashlib.sha256(q_path.read_bytes()).hexdigest() == r["policy_sha256"], q_path
        assert hashlib.sha256(log_path.read_bytes()).hexdigest() == r["log_sha256"], log_path
        lives = read_lives(log_path)
        recomputed = summarize(lives)
        for key in ("inspections", "per_life", "hits_per_life", "moves_per_life"):
            np.testing.assert_allclose(recomputed[key], r["evaluation"][key], atol=1e-12)
        np.testing.assert_allclose(recomputed["share"], r["evaluation"]["share"], atol=1e-12)
        model, q = WorldModel.load(weights), QNetwork.load(q_path)
        again = greedy_lives(model, q, r["condition"], 910001, args.replay_lives)
        assert [l["actions"] for l in again] == [l["actions"] for l in lives[:args.replay_lives]], name
        replayed += 1
    verdict = criteria(runs)
    print(json.dumps(verdict, indent=2))
    print(f"Learned-inquiry audit reproduced: hashes, summaries and greedy replays of {args.replay_lives} lives for {replayed} runs.")
    if args.check:
        (root / "verification.json").write_text(json.dumps({"criteria": verdict, "runs_replayed": replayed}, indent=2) + "\n")


if __name__ == "__main__":
    main()
