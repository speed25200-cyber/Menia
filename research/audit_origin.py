"""Independent recomputation of the origin-inquiry report from logs and weights.

Recomputes inspection shares and hits from the raw logs, replays hidden states
from the saved weights, refits the probe, reruns the intervention test, and
evaluates the pre-registered criteria. Shares the code for the model step;
it is not an external replication.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .origin_env import N_MOVE, N_INSPECT
from .origin_neural import WorldModel, replay_states, probe_analysis, intervention_analysis


def read_lives(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def shares_from_logs(lives):
    counts = np.zeros(N_INSPECT)
    hits = 0
    for life in lives:
        hits += sum(life["rewards"])
        for a in life["actions"]:
            if a >= N_MOVE:
                counts[a - N_MOVE] += 1
    total = counts.sum()
    return (counts / total if total else np.zeros(N_INSPECT)), int(total), hits / len(lives)


def criteria(report):
    by = {}
    for run in report["runs"]:
        by.setdefault(run["condition"], []).append(run)
    out = {}
    T = by.get("T", [])
    def share(run, policy, k):
        return run["policies"][policy]["share"][k]
    out["M1_T_self"] = all(share(r, "self", 0) > 0.5 and share(r, "self", 0) - (share(r, "self", 2) + share(r, "self", 3)) >= 0.3 for r in T) and bool(T)
    out["M1_T_world"] = all(share(r, "world", 1) > 0.5 for r in T) and bool(T)
    out["M1_C1"] = all(share(r, "self", 0) <= share(r, "self", 2) + share(r, "self", 3) + 0.1 for r in by.get("C1", [])) and bool(by.get("C1"))
    out["M1_C3"] = all(share(r, "self", 0) <= share(r, "self", 2) + share(r, "self", 3) + 0.1 for r in by.get("C3", [])) and bool(by.get("C3"))
    out["M2_T_accuracy"] = all(r["policies"]["self"]["probe"]["final_accuracy"] >= 0.9 for r in T) and bool(T)
    out["M2_T_stability"] = all(r["policies"]["self"]["probe"]["stability"] >= 0.9 for r in T) and bool(T)
    out["M3_intervention"] = all((r["policies"]["self"]["intervention"]["rate"] or 0) >= 0.8 for r in T) and bool(T)
    c1_hits = {r["seed"]: r["policies"]["self"]["hits_per_life"] for r in by.get("C1", [])}
    out["M3_reward"] = all(r["policies"]["self"]["hits_per_life"] >= 0.8 * c1_hits.get(r["seed"], float("inf")) for r in T) and bool(T) and bool(c1_hits)
    out["global"] = all(out[k] for k in ("M1_T_self", "M1_T_world", "M1_C1", "M1_C3", "M2_T_accuracy", "M2_T_stability", "M3_intervention", "M3_reward"))
    e1 = [share(r, "all", 0) - share(r, "all", 1) for r in T]
    out["E1_curiosity_only_difference_self_minus_world"] = e1
    out["E2_surprise_noise_share"] = [share(r, "surprise", 2) + share(r, "surprise", 3) for r in T]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/origin-inquiry")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    parts = sorted(root.glob("report-*.json"))
    if not parts:
        raise FileNotFoundError("No report-*.json in " + str(root))
    report = {"parts": [p.name for p in parts], "runs": [], "references": {}}
    for part in parts:
        value = json.loads(part.read_text())
        report["runs"].extend(value["runs"])
        report["references"].update(value["references"])
    worst = 0.0
    for run in report["runs"]:
        weights = root / f"model-{run['condition']}-{run['seed']}.json"
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == run["checkpoint_sha256"], weights
        model = WorldModel.load(weights)
        for policy, entry in run["policies"].items():
            path = root / f"lives-{run['condition']}-{run['seed']}-{policy}.jsonl"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["log_sha256"], path
            lives = read_lives(path)
            shares, total, hits = shares_from_logs(lives)
            np.testing.assert_allclose(shares, entry["share"], atol=1e-12)
            assert total == entry["inspections"]
            np.testing.assert_allclose(hits, entry["hits_per_life"], atol=1e-12)
            if policy == "self":
                states = np.asarray([replay_states(model, life) for life in lives])
                probe = probe_analysis(states, lives)
                for key in ("final_accuracy", "stability"):
                    worst = max(worst, abs(probe[key] - entry["probe"][key]))
                    np.testing.assert_allclose(probe[key], entry["probe"][key], atol=1e-9)
                intervention = intervention_analysis(model, states, lives)
                assert intervention["followed_donor_body"] == entry["intervention"]["followed_donor_body"]
                assert intervention["tested"] == entry["intervention"]["tested"]
    verdict = criteria(report)
    print(json.dumps(verdict, indent=2))
    print(f"Origin-inquiry audit reproduced: hashes, shares, hits, probe and intervention (max probe deviation {worst:.2e}).")
    if args.check:
        (root / "verification.json").write_text(json.dumps({"criteria": verdict, "max_probe_deviation": worst}, indent=2) + "\n")


if __name__ == "__main__":
    main()
