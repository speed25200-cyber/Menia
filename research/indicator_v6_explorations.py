"""Exploratory analyses after version 6 (declared in docs/INDICATOR_AGENT_V6_EXPLORATIONS.md).

On the published agent of the first development run of version 6 (seed 23), without retraining; no
confirmatory seed is used. Two questions left open by version 6:
- HOT-4: would a choice test between a band object and another good object (values at least 0.2 or 0.3
  apart, both hues read) separate the quality space from a random code better than the conflicts with a bad
  object? Set H, 200 lives, agent, random code and no bottleneck.
- HOT-3: does a world with more sensor faults (0.2 instead of 0.08 per step) make the monitor matter to the
  return? Set R, 150 lives, agent, constant gain, no bottleneck and random attention.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from . import sense_atelier as sa
from .indicator_agent import Params
from .indicator_agent_v6 import AgentV6
from .indicator_experiment import SETS, run_life, life_seed, best_square

DEV = 23


def band_choices(params, lives):
    base, mode = SETS["H"]
    out = {}
    for variant in ("agent", "random_code", "unlimited"):
        res = Counter()
        for i in range(lives):
            life = run_life(params, variant, life_seed(base, i), mode, DEV * 1_000_000 + 2 * 10_000 + i, version=6)
            for rec, truth in life["steps"]:
                goal, objects, vals = rec["goal"], truth["objects"], rec["vis_values"]
                if not isinstance(goal, int):
                    continue
                known = [x for x in objects if vals[x] is not None]
                correct = goal == best_square(objects)
                band = [x for x in known if sa.in_band(objects[x]) and sa.value(objects[x]) > 0]
                good = [x for x in known if not sa.in_band(objects[x]) and sa.value(objects[x]) > 0]
                for gap in (0.2, 0.3):
                    if any(abs(sa.value(objects[b]) - sa.value(objects[g])) >= gap for b in band for g in good):
                        res[f"n_good_{gap}"] += 1
                        res[f"ok_good_{gap}"] += correct
                if any(sa.value(objects[b]) > 0.3 for b in band) and any(sa.value(objects[x]) < -0.3 for x in known):
                    res["n_bad"] += 1
                    res["ok_bad"] += correct
        out[variant] = {"band_vs_good_0.2": round(res["ok_good_0.2"] / res["n_good_0.2"], 4), "n_0.2": res["n_good_0.2"],
                        "band_vs_good_0.3": round(res["ok_good_0.3"] / res["n_good_0.3"], 4), "n_0.3": res["n_good_0.3"],
                        "band_vs_bad": round(res["ok_bad"] / res["n_bad"], 4), "n_bad": res["n_bad"]}
    return out


def faults(params, lives, rate):
    saved = sa.FAULT_ENTER
    sa.FAULT_ENTER = rate
    try:
        out = {}
        for variant in ("agent", "constant_gain", "unlimited", "random"):
            returns, low, charge, read, right = [], 0, 0, 0, 0
            for i in range(lives):
                env = sa.SenseAtelier(life_seed(SETS["R"][0], i), "fixed", n_objects=3, charger_moves=True)
                agent = AgentV6(params, variant, seed=DEV * 1_000_000 + i)
                obs, truth = env.reset()
                total = 0.0
                for _ in range(sa.LIFE):
                    action, intent, rec = agent.step(obs)
                    if truth["energy"] < 0.3:
                        low += 1
                        charge += rec["goal"] == "charger"
                    if rec.get("surprise") is not None and truth["t"] >= 8:
                        read += 1
                        right += rec["pos_belief"] == truth["p"]
                    obs, truth = env.step(action, intent)
                    total += obs["reward"]
                returns.append(total)
            out[variant] = {"return": round(float(np.mean(returns)), 4), "pos_read": round(right / read, 4),
                            "recharge_when_low": round(charge / max(1, low), 4)}
        delta = out["unlimited"]["return"] - out["random"]["return"]
        out["delta"] = round(delta, 4)
        out["return_gap_in_delta"] = round((out["agent"]["return"] - out["constant_gain"]["return"]) / delta, 4)
        out["pos_read_gap"] = round(out["agent"]["pos_read"] - out["constant_gain"]["pos_read"], 4)
        return out
    finally:
        sa.FAULT_ENTER = saved


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="artifacts/indicator-agent-v6/dev-params-23.json")
    parser.add_argument("--output", default="artifacts/indicator-agent-v6/explorations.json")
    a = parser.parse_args(argv)
    params = Params.load(a.params)
    out = {"agent": a.params, "band_choices_200_lives": band_choices(params, 200),
           "faults_150_lives": {str(rate): faults(params, 150, rate) for rate in (0.08, 0.2)}}
    Path(a.output).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
