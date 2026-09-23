"""Exploratory, declared (docs/INDICATOR_AGENT_V5_EXPLORATIONS.md): on the version 5 development agent (seed 19),
three refined readings, 150 lives of set R: RPT-1 on memory-demanding choices (the best object last read at least
three steps ago), GWT-4 conditioned on the lowest need, the Pos part of HOT-3 on the read steps during a sensor fault.
Output: artifacts/indicator-agent-v5/refined-dev.json."""
import json
from pathlib import Path
import numpy as np
from research.indicator_agent import Params
from research.indicator_experiment import run_life, life_seed, best_square
P = Params.load("artifacts/indicator-agent-v5/dev-params-19.json")
N = 150
def rpt1(variant):
    n = k = 0
    for i in range(N):
        life = run_life(P, variant, life_seed(930001, i), "fixed", 19 * 1_000_000 + i, version=5)
        last = {}
        for rec, truth in life["steps"]:
            t, objs = truth["t"], truth["objects"]
            if rec.get("bound") is not None and rec["bound"] == truth["spot"]:
                pass
            goal = rec["goal"]
            if isinstance(goal, int) and objs:
                best = best_square(objs)
                seen = last.get(best)
                if seen is not None and t - seen >= 3:
                    n += 1; k += goal == best
            if truth["spot"] is not None and truth["spot"] in objs:
                last[truth["spot"]] = t  # the object under the spotlight at this step was read
            for x in list(last):
                if x not in objs:
                    del last[x]
    return k / max(n, 1), n
def gwt4():
    low = high = nl = nh = 0
    for i in range(N):
        life = run_life(P, "agent", life_seed(930001, i), "fixed", 19 * 1_000_000 + i, version=5)
        for rec, truth in life["steps"]:
            m = min(truth["energy"], truth["satiety"])
            w = "intero" in rec["writers"]
            if m < 0.35: nl += 1; low += w
            elif m > 0.7: nh += 1; high += w
    return low / nl, high / nh, nl, nh
def hot3(variant):
    n = k = 0
    for i in range(N):
        life = run_life(P, variant, life_seed(930001, i), "fixed", 19 * 1_000_000 + i, version=5)
        for rec, truth in life["steps"]:
            if rec.get("surprise") is not None and truth["t"] >= 8 and truth["fault"]:
                n += 1; k += rec["pos_belief"] == truth["p"]
    return k / n, n
if __name__ == "__main__":
    a, na = rpt1("agent"); b, nb = rpt1("no_recurrence")
    lo, hi, nl, nh = gwt4()
    x, nx = hot3("agent"); y, ny = hot3("constant_gain")
    out = {"note": "exploratory, declared; version 5 development agent, seed 19, 150 lives of set R",
           "rpt1_memory_demanding_choice": {"agent": [a, na], "no_recurrence": [b, nb], "drop": a - b},
           "gwt4_lowest_need": {"low": [lo, nl], "high": [hi, nh], "gap": lo - hi},
           "hot3_pos_during_faults": {"agent": [x, nx], "constant_gain": [y, ny], "gap": x - y}}
    Path("artifacts/indicator-agent-v5/refined-dev.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
