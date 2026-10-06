"""The need that lasts (docs/TINY_LOOP_DURATION_PROTOCOL.md): DUR1 to DUR3 of test 35.

The ten loops of test 34 (research/tiny_loop_reliable.py, seeds 20 to 29, final checkpoints at 24 000 updates), without
new training. Grafts of their transmitted state at j read at t = j + g for g = 1 to 6, three turns t per recipient and
per g, on 128 new graft lives (stream 83; draw stream 84). The verdicts are numpy only; the measures need torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_relief as RL
from . import need_world as W
from . import tiny_loop as TL
from . import tiny_survival as TS
from .need_carried_content import rule_effect
from .need_relief_graft_event import same_history

SEEDS = tuple(range(20, 30))
GAPS, PER_GAP = (1, 2, 3, 4, 5, 6), 3
LIVES_STREAM, DRAW_STREAM = 83, 84
KINDS = ("a", "a0", "b")
GAP_MAIN, GAP_FAR = 4, 6
DUR1, DUR1_SEED, DUR1_SEEDS, HALF = 0.10, 0.05, 8, 0.5
MIN_GRAFTS, TOLERANCE = 100, 1e-6
SOURCE = "artifacts/tiny-loop-reliable/checkpoints"
ROOT = "artifacts/tiny-loop-duration"


def lives():
    return RL.writer_lives(LIVES_STREAM, TL.LIVES)


def draw(held):
    """Recipients in order; g from 1 to 6; up to PER_GAP turns t; per t one donor of each kind, alive at j = t - g with
    the same written action at j: (a) other needs at j and a rule effect not zero, (a0) other needs and a rule effect
    zero, (b) the same needs at j. No cap."""
    rng = np.random.default_rng([W.SEED, DRAW_STREAM, 0])
    at = [{x["t"]: x for x in W.decisions(life)} for life in held]
    out = []
    for r, life in enumerate(held):
        for g in GAPS:
            turns = [t for t in sorted(at[r]) if t - g >= 1 and t - g in at[r]]
            for k in sorted(rng.permutation(len(turns))[:PER_GAP]):
                t = turns[k]
                j = t - g
                mine = at[r][j]
                for kind in KINDS:
                    donors = []
                    for d in range(len(held)):
                        x = at[d].get(j)
                        if d == r or x is None or x["action"] != mine["action"]:
                            continue
                        if ((x["E"], x["N"]) == (mine["E"], mine["N"])) != (kind == "b"):
                            continue
                        e = rule_effect(life, held[d], j, t)
                        if e is None or (kind == "a" and e == 0) or (kind == "a0" and e != 0):
                            continue
                        donors.append((d, e))
                    if donors:
                        d, e = donors[int(rng.integers(len(donors)))]
                        out.append({"index": len(out), "kind": kind, "r": r, "d": d, "j": j, "t": t, "g": g,
                                    "e": int(e)})
    return out


def per_gap(rows, grafts, held):
    """For one seed: per g, m (the mean over (a) of the change of P(R) times e), |change| per kind, |change| of the (b)
    grafts that have another history, and m at equal event at turn j."""
    got = {i: p1 - p0 for i, p0, p1 in rows}
    if sorted(got) != [g["index"] for g in grafts]:
        raise ValueError("the rows do not match the draw")
    at = [{x["t"]: x for x in W.decisions(life)} for life in held]
    out = {}
    for g in GAPS:
        sel = [x for x in grafts if x["g"] == g]
        delta = np.array([got[x["index"]] for x in sel])
        e = np.array([x["e"] for x in sel], float)
        kind = np.array([x["kind"] for x in sel])
        alike = np.array([x["kind"] == "b" and same_history(held, x["r"], x["d"], x["j"]) for x in sel])
        same = np.array([at[x["r"]][x["j"]]["event"] == at[x["d"]][x["j"]]["event"] for x in sel])
        a, b = kind == "a", kind == "b"
        out[g] = {"m": float((delta[a] * e[a]).mean()), **{f"abs_{k}": float(np.abs(delta[kind == k]).mean()) for k in KINDS},
                  "abs_b_other_history": float(np.abs(delta[b & ~alike]).mean()),
                  "m_same_event": float((delta[a & same] * e[a & same]).mean())}
    return out


def verdicts(runs, grafts, setup, held):
    """runs: {seed: {"grafts": [[index, p without, p with]]}}; grafts: the draw; setup: the checks."""
    seeds = sorted(runs, key=int)
    by = {s: per_gap(runs[s]["grafts"], grafts, held) for s in seeds}
    m = {g: [by[s][g]["m"] for s in seeds] for g in GAPS}
    main = TS.interval(m[GAP_MAIN])
    far = TS.interval(m[GAP_FAR])
    abs_a = float(np.mean([by[s][GAP_MAIN]["abs_a"] for s in seeds]))
    abs_b = float(np.mean([by[s][GAP_MAIN]["abs_b"] for s in seeds]))
    need_not_history = TS.interval([by[s][GAP_MAIN]["abs_a"] - by[s][GAP_MAIN]["abs_b"] for s in seeds])
    counts = {g: {k: sum(x["g"] == g and x["kind"] == k for x in grafts) for k in KINDS} for g in GAPS}
    out = {"DUR1": bool(main["mean"] >= DUR1 and main["low"] > 0 and sum(v > DUR1_SEED for v in m[GAP_MAIN]) >= DUR1_SEEDS),
           "DUR2": bool(abs_b <= HALF * abs_a and need_not_history["low"] > 0),
           "DUR3": bool(far["low"] > 0)}
    out["valid"] = bool(counts[GAP_MAIN]["a"] >= MIN_GRAFTS and counts[GAP_MAIN]["b"] >= MIN_GRAFTS
                        and counts[GAP_FAR]["a"] >= MIN_GRAFTS and setup["self_graft_gap"] <= TOLERANCE
                        and len(seeds) == len(SEEDS) and all(setup["updates"][s] == 24000 for s in seeds))
    out["global"] = bool(out["valid"] and out["DUR1"] and out["DUR2"] and out["DUR3"])
    first = float(np.mean(m[1]))
    values = {"m": {g: TS.interval(m[g]) for g in GAPS},
              "m_ratio_to_g1": {g: float(np.mean(m[g]) / first) for g in GAPS},
              "seeds_above_at_4": int(sum(v > DUR1_SEED for v in m[GAP_MAIN])),
              "abs_a_4": abs_a, "abs_b_4": abs_b, "need_not_history_4": need_not_history, "grafts": counts,
              "abs": {g: {k: float(np.mean([by[s][g][f"abs_{k}"] for s in seeds])) for k in KINDS} for g in GAPS},
              "abs_b_other_history": {g: float(np.mean([by[s][g]["abs_b_other_history"] for s in seeds])) for g in GAPS},
              "m_same_event": {g: TS.interval([by[s][g]["m_same_event"] for s in seeds]) for g in GAPS},
              "by_seed": by}
    return {"verdicts": out, "values": values}


# ----------------------------------------------------------------------------------------------------- torch part

def load(seed, source=SOURCE):
    """The loop of test 34 for this seed, at its final checkpoint (no training)."""
    import torch
    from . import tiny_loop_pilot as TLP
    from . import tiny_loop_reliable as REL
    checkpoint = Path(source) / f"loop-{seed}.pt"
    update = torch.load(checkpoint, weights_only=False)["update"]
    if update != REL.CONFIG["updates"]:
        raise ValueError(f"{checkpoint} is at {update} updates")
    net, _ = TLP.train("loop", seed, REL.CONFIG, lambda m: None, checkpoint=checkpoint)
    net.eval()
    return net, update


def self_graft(net, held):
    """The recipient's own state grafted back changes nothing (4 cases)."""
    reader, gaps = TL.Reader(net, "loop"), []
    for life in [l for l in held if len(W.decisions(l)) >= 6][:4]:
        ds = W.decisions(life)
        j, t = ds[1]["t"], ds[5]["t"]
        own, states = reader.read(life)
        p, _ = reader.read(life, upto=t, graft=(j, states[j]))
        gaps.append(abs(p[t] - own[t]))
    return float(max(gaps))


def run(a):
    import torch
    torch.set_num_threads(1)
    root = Path(a.out)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    held = lives()
    grafts = draw(held)
    setup = {"self_graft_gap": 0.0, "updates": {}}
    for seed in a.seeds:
        net, update = load(seed, a.source)
        setup["updates"][str(seed)] = update
        setup["self_graft_gap"] = max(setup["self_graft_gap"], self_graft(net, held))
        path = root / "runs" / f"loop-{seed}.json"
        if not path.exists():
            rows = TL.measure_grafts(TL.Reader(net, "loop"), held, grafts)
            path.write_text(json.dumps({"seed": seed, "update": update, "grafts": rows}) + "\n")
            print(f"loop seed {seed}: {len(rows)} grafts measured", flush=True)
    if list(a.seeds) == list(SEEDS):
        (root / "setup.json").write_text(json.dumps(setup, indent=1) + "\n")


def gather(root):
    out = Path(root)
    runs = {str(json.loads(p.read_text())["seed"]): json.loads(p.read_text()) for p in sorted((out / "runs").glob("*.json"))}
    setup = json.loads((out / "setup.json").read_text())
    held = lives()
    result = verdicts(runs, draw(held), setup, held)
    return json.loads(json.dumps(dict(result, setup=setup)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    r.add_argument("--source", default=SOURCE)
    sub.add_parser("write")
    sub.add_parser("verdicts")
    for s in sub.choices.values():
        s.add_argument("--out", default=ROOT)
    a = parser.parse_args(argv)
    if a.command == "run":
        run(a)
        return
    result = gather(a.out)
    path = Path(a.out) / "verdicts.json"
    if a.command == "write":
        path.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result["verdicts"]))
    else:
        differs = result != json.loads(path.read_text())
        print("differs:", "yes" if differs else "none")
        if differs:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
