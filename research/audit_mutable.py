"""Independent recomputation of the mutable-body report and evaluation of criteria M1-M4."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .origin_neural import WorldModel, replay_states
from .mutable_experiment import mark_reads_after, hits_ratio, entropy_jump, update_probe


def read_lives(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def criteria(runs):
    by = {}
    for r in runs:
        by.setdefault(r["regime"], []).append(r)
    out = {}
    def m1(regime): return [r["M1_update"]["new_body_accuracy"] for r in by.get(regime, [])]
    def m3(regime): return [r["M3_mark_reads_after_change"] for r in by.get(regime, [])]
    def m3c(regime): return [r["M3_control_mark_reads_after_step12"] for r in by.get(regime, [])]
    def m4(regime): return [r["M4_recovery"]["ratio"] for r in by.get(regime, [])]
    def m2(regime): return [r["M2_detection"]["jump"] for r in by.get(regime, [])]
    out["values"] = {regime: {"M1": m1(regime), "M2": m2(regime), "M3": m3(regime), "M3_control": m3c(regime), "M4": m4(regime)} for regime in by}
    out["M1_MS_updates"] = len(m1("MS")) == 3 and all(v >= 0.9 for v in m1("MS"))
    out["M1_S_does_not_update"] = len(m1("S")) == 3 and all(v < 0.9 for v in m1("S"))
    out["M1_MW_does_not_update"] = len(m1("MW")) == 3 and all(v < 0.9 for v in m1("MW"))
    out["M2_MS_detects"] = len(m2("MS")) == 3 and all(v >= 0.3 for v in m2("MS"))
    out["M3_MS_returns_to_mark"] = len(m3("MS")) == 3 and all(v >= 0.5 for v in m3("MS"))
    out["M3_S_MW_do_not"] = len(m3("S")) == 3 and len(m3("MW")) == 3 and all(v <= 0.1 for v in m3("S") + m3("MW"))
    out["M3_control_quiet"] = all(v <= 0.1 for regime in ("S", "MS", "MW") for v in m3c(regime)) and all(len(m3c(r)) == 3 for r in ("S", "MS", "MW"))
    out["M4_MS_recovers"] = len(m4("MS")) == 3 and all(v is not None and v >= 0.8 for v in m4("MS"))
    out["M4_S_does_not"] = len(m4("S")) == 3 and all(v is not None and v < 0.8 for v in m4("S"))
    out["global"] = all(out[k] for k in ("M1_MS_updates", "M1_S_does_not_update", "M1_MW_does_not_update", "M3_MS_returns_to_mark",
                                         "M3_S_MW_do_not", "M3_control_quiet", "M4_MS_recovers", "M4_S_does_not"))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/mutable-body")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    parts = sorted(root.glob("report-mutable-*.json"))
    if not parts:
        raise FileNotFoundError("No report-mutable-*.json in " + str(root))
    runs = []
    for part in parts:
        runs.extend(json.loads(part.read_text())["runs"])
    for r in runs:
        weights = Path(r["weights"])
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == r["weights_sha256"], weights
        model = WorldModel.load(weights)
        data = {}
        for name, entry in r["lives"].items():
            path = root / f"lives-{r['regime']}-{r['seed']}-{name}.jsonl"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["log_sha256"], path
            lives = read_lives(path)
            np.testing.assert_allclose(mark_reads_after(lives), entry["mark_reads_after_change_step"], atol=1e-12)
            hits = hits_ratio(lives)
            for key in ("hits_before_per_life", "hits_after_per_life"):
                np.testing.assert_allclose(hits[key], entry["hits"][key], atol=1e-12)
            if "entropy" in entry:
                np.testing.assert_allclose(entropy_jump(lives)["jump"], entry["entropy"]["jump"], atol=1e-9)
            if name in ("control-none", "change-none"):
                data[name] = (lives, np.asarray([replay_states(model, life) for life in lives]))
        probe = update_probe(data["control-none"][1], data["control-none"][0], data["change-none"][1], data["change-none"][0])
        np.testing.assert_allclose(probe["new_body_accuracy"], r["M1_update"]["new_body_accuracy"], atol=1e-9)
    verdict = criteria(runs)
    print(json.dumps(verdict, indent=2))
    print(f"Mutable-body audit reproduced for {len(runs)} runs: hashes, mark reads, hits, entropy jumps, replayed states and probes.")
    if args.check:
        (root / "verification.json").write_text(json.dumps({"criteria": verdict}, indent=2) + "\n")


if __name__ == "__main__":
    main()
