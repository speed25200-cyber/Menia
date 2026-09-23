"""Second reading of the version 5 indicator agent (docs/INDICATOR_AGENT_SECOND_READING_PROTOCOL.md).

The primary reading is the fourteen criteria of version 1, computed by indicator_experiment.criteria. The
second reading replaces four usage tests, thresholds unchanged: RPT-1 by the return lost without
recurrence, GWT-4 by attention to Intero conditioned on the lowest need, the Pos part of HOT-3 on the read
steps during a sensor fault, and the choice part of HOT-4 on conflicts involving a band object. The agents
and their test lives are those of the run; the lives needed for the new measures are replayed from their
seeds with the same code.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .indicator_agent import Params
from .indicator_experiment import SETS, run_life, life_seed, best_square, criteria
from .sense_atelier import value, in_band

NAMES = ["RPT-1", "RPT-2", "GWT-1", "GWT-2", "GWT-3", "GWT-4", "HOT-1", "HOT-2", "HOT-3", "HOT-4", "AST-1", "PP-1",
         "AE-1", "AE-2"]


def refined_counts(params, seed, lives):
    """Counts for the four refined measures, from lives replayed exactly as in the evaluation."""
    out = {"intero_low": [0, 0], "intero_high": [0, 0], "fault_read": {"agent": [0, 0], "constant_gain": [0, 0]},
           "band_conflict": {"agent": [0, 0], "random_code": [0, 0]}}
    per_set = {}
    for set_index, (name, (base, mode)) in enumerate(SETS.items()):
        variants = {"R": ("agent", "constant_gain"), "M": ("agent",), "H": ("agent", "random_code")}[name]
        for variant in variants:
            low, high = [0, 0], [0, 0]
            for i in range(lives):
                life = run_life(params, variant, life_seed(base, i), mode, seed * 1_000_000 + set_index * 10_000 + i, version=5)
                for rec, truth in life["steps"]:
                    if variant == "agent" and name in ("R", "M") and len(rec["writers"]) == 1:
                        need = min(truth["energy"], truth["satiety"])
                        if need < 0.35:
                            low[0] += "intero" in rec["writers"]
                            low[1] += 1
                        elif need > 0.7:
                            high[0] += "intero" in rec["writers"]
                            high[1] += 1
                    if name == "R" and rec.get("surprise") is not None and truth["t"] >= 8 and truth["fault"]:
                        out["fault_read"][variant][0] += rec["pos_belief"] == truth["p"]
                        out["fault_read"][variant][1] += 1
                    if name == "H" and isinstance(rec["goal"], int):
                        objects, vals = truth["objects"], rec["vis_values"]
                        known = [x for x in objects if vals[x] is not None]
                        if any(in_band(objects[x]) and value(objects[x]) > 0.3 for x in known) and \
                                any(value(objects[x]) < -0.3 for x in known):
                            out["band_conflict"][variant][0] += rec["goal"] == best_square(objects)
                            out["band_conflict"][variant][1] += 1
            if variant == "agent" and name in ("R", "M"):
                per_set[name] = {"low": low, "high": high}
    out["intero_by_set"] = per_set
    return out


def rate(pair):
    return pair[0] / pair[1] if pair[1] else None


def above(a, b):
    return a is not None and b is not None and a > b


def second_reading(reports, refined):
    primary, values = criteria(reports)
    out = dict(primary)
    seconds = {}
    ret = lambda r, v, s="R": r["evaluation"][v][s]["return"]
    delta = values["delta"]["pooled"]
    # RPT-1: presence unchanged, usage by the return lost without recurrence.
    memory = values["RPT-1"]["memory_sign"]
    loss = float(np.mean([ret(r, "agent") - ret(r, "no_recurrence") for r in reports]))
    seconds["RPT-1"] = {"memory_sign": memory, "return_loss": loss, "threshold": 0.1 * delta}
    out["RPT-1"] = bool(primary["valid"] and memory is not None and memory >= 0.85 and loss >= 0.1 * delta
                        and all(ret(r, "agent") > ret(r, "no_recurrence") for r in reports))
    # GWT-4: attention to Intero by the lowest need; the two other parts unchanged.
    low = float(np.mean([np.mean([rate(c["intero_by_set"][s]["low"]) or 0 for s in ("R", "M")]) for c in refined]))
    high = float(np.mean([np.mean([rate(c["intero_by_set"][s]["high"]) or 0 for s in ("R", "M")]) for c in refined]))
    g4 = values["GWT-4"]
    seconds["GWT-4"] = {"intero_low_need": low, "intero_high_need": high, "body_writes_after_change": g4["body_writes_after_change"],
                        "round_robin_gap": g4["round_robin_gap"]}
    out["GWT-4"] = bool(primary["valid"] and low - high >= 0.2 and g4["body_writes_after_change"] is not None
                        and g4["body_writes_after_change"] >= 0.6
                        and g4["round_robin_gap"] >= 0.2 * delta
                        and all(ret(r, "agent") > ret(r, "round_robin") for r in reports))
    # HOT-3: Pos during sensor faults; the return part unchanged.
    pooled = lambda v: sum(c["fault_read"][v][0] for c in refined) / max(1, sum(c["fault_read"][v][1] for c in refined))
    fault_gap = pooled("agent") - pooled("constant_gain")
    per_seed = all(above(rate(c["fault_read"]["agent"]), rate(c["fault_read"]["constant_gain"])) for c in refined)
    h3 = values["HOT-3"]
    seconds["HOT-3"] = {"pos_fault_gap": fault_gap, "return_gap": h3["return_gap"], "threshold_return": 0.1 * delta}
    out["HOT-3"] = bool(primary["valid"] and fault_gap >= 0.05 and per_seed and h3["return_gap"] >= 0.1 * delta
                        and all(ret(r, "agent") > ret(r, "constant_gain") for r in reports))
    # HOT-4: choice on conflicts involving a band object; the other parts unchanged.
    pooled = lambda v: sum(c["band_conflict"][v][0] for c in refined) / max(1, sum(c["band_conflict"][v][1] for c in refined))
    choice, choice_random = pooled("agent"), pooled("random_code")
    h4 = values["HOT-4"]
    seconds["HOT-4"] = {"spearman": h4["spearman"], "active_fraction": h4["active_fraction"], "band_error": h4["band_error"],
                        "band_error_random": h4["band_error_random"], "conflict_choice": choice,
                        "conflict_choice_random": choice_random,
                        "counts": [sum(c["band_conflict"][v][1] for c in refined) for v in ("agent", "random_code")]}
    out["HOT-4"] = bool(h4["spearman"] >= 0.85 and h4["active_fraction"] <= 0.4 and h4["band_error"] is not None
                        and h4["band_error"] <= 0.2 and h4["band_error_random"] is not None and h4["band_error_random"] >= 0.4
                        and choice >= 0.8 and choice_random <= 0.65
                        and all(above(rate(c["band_conflict"]["agent"]), rate(c["band_conflict"]["random_code"])) for c in refined))
    out["passed"] = sum(1 for n in NAMES if out[n])
    out["global"] = all(out[n] for n in NAMES)
    return {"primary": primary, "second": out, "second_values": seconds}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/indicator-agent-v5-second")
    parser.add_argument("--output", default=None)
    parser.add_argument("--check", action="store_true", help="recompute and compare with second-reading.json")
    a = parser.parse_args(argv)
    root = Path(a.root)
    reports = [json.loads(p.read_text()) for p in sorted(root.glob("report-*.json"))]
    refined = []
    for report in reports:
        params = Params.load(root / f"params-{report['seed']}.json")
        refined.append(refined_counts(params, report["seed"], report["settings"]["test_lives"]))
        print(f"replayed seed {report['seed']}", flush=True)
    result = second_reading(reports, refined)
    result = json.loads(json.dumps(result))
    output = Path(a.output) if a.output else root / "second-reading.json"
    if a.check:
        published = json.loads(output.read_text())
        differs = published != result
        print("differs:", "yes" if differs else "none")
        if differs:
            raise SystemExit(1)
        return
    output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"primary": result["primary"], "second": result["second"]}, indent=1))


if __name__ == "__main__":
    main()
