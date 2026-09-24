"""Third reading of HOT-3 and HOT-4 on new agents (docs/INDICATOR_THIRD_READING_PROTOCOL.md).

Criteria chosen knowing where the earlier readings failed, and said so: HOT-3 keeps only the belief part of the
second reading (Pos during sensor faults, agent against constant gain), dropping the return part; HOT-4 keeps the
structural parts of version 1 and judges the choice between a band object and another good object (values at least
0.2 apart, both hues read), agent against random code. The verdicts of the earlier readings are not replaced.
Lives are replayed exactly as in the evaluation.
"""
import argparse
import json
from pathlib import Path
from .indicator_agent import Params
from .indicator_experiment import SETS, run_life, life_seed, best_square, criteria
from .indicator_second_reading import rate, above
from .sense_atelier import value, in_band

GAP = 0.2


def counts(params, seed, lives, version=6):
    out = {"fault_read": {"agent": [0, 0], "constant_gain": [0, 0]},
           "band_vs_good": {"agent": [0, 0], "random_code": [0, 0]}}
    for set_index, (name, (base, mode)) in enumerate(SETS.items()):
        if name not in ("R", "H"):
            continue
        for variant in {"R": ("agent", "constant_gain"), "H": ("agent", "random_code")}[name]:
            for i in range(lives):
                life = run_life(params, variant, life_seed(base, i), mode, seed * 1_000_000 + set_index * 10_000 + i,
                                version=version)
                for rec, truth in life["steps"]:
                    if name == "R" and rec.get("surprise") is not None and truth["t"] >= 8 and truth["fault"]:
                        out["fault_read"][variant][0] += rec["pos_belief"] == truth["p"]
                        out["fault_read"][variant][1] += 1
                    if name == "H" and isinstance(rec["goal"], int):
                        objects, vals = truth["objects"], rec["vis_values"]
                        known = [x for x in objects if vals[x] is not None]
                        band = [x for x in known if in_band(objects[x]) and value(objects[x]) > 0]
                        good = [x for x in known if not in_band(objects[x]) and value(objects[x]) > 0]
                        if any(abs(value(objects[b]) - value(objects[g])) >= GAP for b in band for g in good):
                            out["band_vs_good"][variant][0] += rec["goal"] == best_square(objects)
                            out["band_vs_good"][variant][1] += 1
    return out


def third_reading(reports, per_seed):
    primary, values = criteria(reports)
    pooled = lambda key, v: sum(c[key][v][0] for c in per_seed) / max(1, sum(c[key][v][1] for c in per_seed))
    fault_gap = pooled("fault_read", "agent") - pooled("fault_read", "constant_gain")
    t3 = bool(primary["valid"] and fault_gap >= 0.05
              and all(above(rate(c["fault_read"]["agent"]), rate(c["fault_read"]["constant_gain"])) for c in per_seed))
    h4 = values["HOT-4"]
    choice, choice_random = pooled("band_vs_good", "agent"), pooled("band_vs_good", "random_code")
    structure = (h4["spearman"] >= 0.85 and h4["active_fraction"] <= 0.4 and h4["band_error"] is not None
                 and h4["band_error"] <= 0.2 and h4["band_error_random"] is not None and h4["band_error_random"] >= 0.4)
    t4 = bool(structure and choice >= 0.7 and choice - choice_random >= 0.2
              and all(above(rate(c["band_vs_good"]["agent"]), rate(c["band_vs_good"]["random_code"])) for c in per_seed))
    return {"valid": primary["valid"], "T3": t3, "T4": t4,
            "values": {"T3": {"pos_fault_agent": pooled("fault_read", "agent"),
                              "pos_fault_constant_gain": pooled("fault_read", "constant_gain"), "gap": fault_gap},
                       "T4": {"spearman": h4["spearman"], "active_fraction": h4["active_fraction"],
                              "band_error": h4["band_error"], "band_error_random": h4["band_error_random"],
                              "band_vs_good_agent": choice, "band_vs_good_random": choice_random,
                              "gap": choice - choice_random}},
            "per_seed": [{"seed": r["seed"], **c} for r, c in zip(reports, per_seed)]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/indicator-agent-v6-third")
    parser.add_argument("--check", action="store_true", help="recompute and compare with third-reading.json")
    a = parser.parse_args(argv)
    root = Path(a.root)
    reports = [json.loads(p.read_text()) for p in sorted(root.glob("report-*.json"))]
    per_seed = []
    for report in reports:
        params = Params.load(root / f"params-{report['seed']}.json")
        per_seed.append(counts(params, report["seed"], report["settings"]["test_lives"], report["settings"]["version"]))
        print(f"replayed seed {report['seed']}", flush=True)
    result = json.loads(json.dumps(third_reading(reports, per_seed)))
    output = root / "third-reading.json"
    if a.check:
        differs = json.loads(output.read_text()) != result
        print("differs:", "yes" if differs else "none")
        if differs:
            raise SystemExit(1)
        return
    output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("valid", "T3", "T4")}, indent=1), json.dumps(result["values"], indent=1))


if __name__ == "__main__":
    main()
