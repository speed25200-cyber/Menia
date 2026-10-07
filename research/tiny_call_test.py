"""One state, two contents (docs/TINY_CALL_PROTOCOL.md): CALL1 to CALL4 of test 36.

The world, agent and training of the pilot (research/tiny_call.py) with the setting it chose (C2: 48 000 updates), for
the loop and the cut alike; new seeds 30 to 39; survival on 256 new worlds (stream 88); 128 new graft lives (stream
97; draw stream 98) with five kinds of grafts between lives that took the same action and call at j: (h) same E and N,
other H, and the rule "call if H <= 3" would change its call at t; (h0) the same without that change; (n) same E and
H, other N, and the rule "needs" would change its action at t; (n0) the same without that change; (b) same E, N, H.
Published without threshold besides: the survival and the grafts measured on the way at 24 000 updates (the setting
C1), and the lowest point of each learning curve after 2 000 updates. The verdicts are numpy only; training and
measures need torch.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from . import need_memory as NM
from . import need_world as W
from . import tiny_call as C
from . import tiny_loop_pilot2 as Q
from . import tiny_survival as TS

CONFIG = C.CONFIG["C2"]
MID = C.CONFIG["C1"]["updates"]
SEEDS = tuple(range(30, 40))
ARMS = ("loop", "cut")
WORLDS_STREAM, LIVES_STREAM, DRAW_STREAM = 88, 97, 98
KINDS = ("h", "h0", "n", "n0", "b")
CALL1, CALL2, CALL3, SEED_LEVEL, SEEDS_NEEDED, HALF = 0.10, 0.10, 0.10, 0.05, 8, 0.5
MIN_B, MIN_GRAFTS, TOLERANCE = 0.20, 100, 1e-6
LOW_FROM = 20
ROOT = "artifacts/tiny-call-test"
JOBS = [(arm, seed) for seed in SEEDS for arm in ARMS]


def lives():
    return C.lives(LIVES_STREAM)


def draw(held):
    """Recipients in order; gaps 1 and 2; every turn t with a decision at j = t - g; per t one donor of each kind (see
    the module docstring), alive at j with the same action and call at j and alive in the replay until t. Each graft
    keeps e_call (rule "call if H <= 3") and e_action (rule "needs") at t, and e, the one its kind is about."""
    rng = np.random.default_rng([W.SEED, DRAW_STREAM, 0])
    at = [{x["t"]: x for x in W.decisions(life)} for life in held]
    out = []
    for r, life in enumerate(held):
        for g in C.GAPS:
            turns = [t for t in sorted(at[r]) if t - g >= 1 and t - g in at[r]]
            for k in sorted(rng.permutation(len(turns))):
                t = turns[k]
                j = t - g
                mine, now = at[r][j], at[r][t]
                for kind in KINDS:
                    donors = []
                    for d in range(len(held)):
                        x = at[d].get(j)
                        if d == r or x is None or (x["action"], x["call"]) != (mine["action"], mine["call"]):
                            continue
                        same = {v: x[v] == mine[v] for v in ("E", "N", "H")}
                        if kind in ("h", "h0") and not (same["E"] and same["N"] and not same["H"]):
                            continue
                        if kind in ("n", "n0") and not (same["E"] and same["H"] and not same["N"]):
                            continue
                        if kind == "b" and not all(same.values()):
                            continue
                        needs = C.replay(life, j, (x["E"], x["N"], x["H"]), t)
                        if needs is None:
                            continue
                        e_call = C.call_rule(needs[2]) - C.call_rule(now["H"])
                        e_action = int(NM.rule(needs[0], needs[1], now["last"]) == 0) - int(now["rule_action"] == 0)
                        if (kind == "h" and e_call == 0) or (kind == "h0" and e_call != 0):
                            continue
                        if (kind == "n" and e_action == 0) or (kind == "n0" and e_action != 0):
                            continue
                        donors.append((d, e_call, e_action))
                    if donors:
                        d, e_call, e_action = donors[int(rng.integers(len(donors)))]
                        e = e_call if kind in ("h", "h0") else e_action if kind in ("n", "n0") else 0
                        out.append({"index": len(out), "kind": kind, "r": r, "d": d, "j": j, "t": t, "g": g, "e": e,
                                    "e_call": e_call, "e_action": e_action})
    return out


def with_rule_action(held):
    """The lives with, at each decision, the action of the rule "needs" (for e_action)."""
    for life in held:
        for x in W.decisions(life):
            x["rule_action"] = NM.rule(x["E"], x["N"], x["last"])
    return held


def per_seed(rows, grafts):
    """For one loop: per kind, the mean |change| of P(R) and of P(call); m_call over (h) and m_action over (n); per
    gap, m_call and m_action."""
    by = {g["index"]: g for g in grafts}
    if sorted(r[0] for r in rows) != sorted(by):
        raise ValueError("the rows do not match the draw")
    d = {i: (r1 - r0, c1 - c0) for i, r0, r1, c0, c1 in rows}
    out = {}
    for kind in KINDS:
        sel = [i for i in d if by[i]["kind"] == kind]
        out[f"abs_action_{kind}"] = float(np.mean([abs(d[i][0]) for i in sel]))
        out[f"abs_call_{kind}"] = float(np.mean([abs(d[i][1]) for i in sel]))
    out["m_call"] = float(np.mean([d[i][1] * by[i]["e"] for i in d if by[i]["kind"] == "h"]))
    out["m_action"] = float(np.mean([d[i][0] * by[i]["e"] for i in d if by[i]["kind"] == "n"]))
    for g in C.GAPS:
        out[f"m_call_g{g}"] = float(np.mean([d[i][1] * by[i]["e"] for i in d if by[i]["kind"] == "h" and by[i]["g"] == g]))
        out[f"m_action_g{g}"] = float(np.mean([d[i][0] * by[i]["e"] for i in d
                                               if by[i]["kind"] == "n" and by[i]["g"] == g]))
    return out


def verdicts(runs, grafts, setup, mid):
    """runs: {arm: {seed: {"survival", "agrees_call_rule", "curve", "grafts" (loop)}}}; grafts: the draw; setup: the
    checks; mid: {arm: {seed: {"survival", "grafts" (loop)}}} at 24 000 updates."""
    seeds = sorted(runs["loop"], key=int)
    by = {s: per_seed(runs["loop"][s]["grafts"], grafts) for s in seeds}
    gain = TS.interval([runs["loop"][s]["survival"] - runs["cut"][s]["survival"] for s in seeds])
    m_call = [by[s]["m_call"] for s in seeds]
    m_action = [by[s]["m_action"] for s in seeds]
    call = TS.interval(m_call)
    action = TS.interval(m_action)
    mean = lambda key: float(np.mean([by[s][key] for s in seeds]))
    h_diff = TS.interval([by[s]["abs_call_h"] - by[s]["abs_action_h"] for s in seeds])
    n_diff = TS.interval([by[s]["abs_action_n"] - by[s]["abs_call_n"] for s in seeds])
    counts = {k: sum(g["kind"] == k for g in grafts) for k in KINDS}
    survival = {a: float(np.mean([runs[a][s]["survival"] for s in seeds])) for a in ARMS}
    out = {"CALL1": bool(gain["mean"] >= CALL1 and gain["low"] > 0),
           "CALL2": bool(call["mean"] >= CALL2 and call["low"] > 0 and sum(v > SEED_LEVEL for v in m_call) >= SEEDS_NEEDED),
           "CALL3": bool(action["mean"] >= CALL3 and action["low"] > 0
                         and sum(v > SEED_LEVEL for v in m_action) >= SEEDS_NEEDED),
           "CALL4": bool(mean("abs_action_h") <= HALF * mean("abs_call_h") and h_diff["low"] > 0
                         and mean("abs_call_n") <= HALF * mean("abs_action_n") and n_diff["low"] > 0)}
    out["valid"] = bool(survival["cut"] >= MIN_B and setup["cut_gap"] <= TOLERANCE and setup["self_graft_gap"] <= TOLERANCE
                        and counts["h"] >= MIN_GRAFTS and counts["n"] >= MIN_GRAFTS
                        and all(len(runs[a]) == len(SEEDS) for a in ARMS))
    out["global"] = bool(out["valid"] and all(out[k] for k in ("CALL1", "CALL2", "CALL3", "CALL4")))
    mid_seeds = sorted(mid["loop"], key=int)
    mid_by = {s: per_seed(mid["loop"][s]["grafts"], grafts) for s in mid_seeds}
    values = {"survival": survival, "survival_gain": gain, "m_call": call, "m_action": action,
              "seeds_above": {"call": int(sum(v > SEED_LEVEL for v in m_call)),
                              "action": int(sum(v > SEED_LEVEL for v in m_action))},
              "abs": {k: {"action": mean(f"abs_action_{k}"), "call": mean(f"abs_call_{k}")} for k in KINDS},
              "call_minus_action_h": h_diff, "action_minus_call_n": n_diff, "grafts": counts,
              "by_gap": {f"g{g}": {"m_call": mean(f"m_call_g{g}"), "m_action": mean(f"m_action_g{g}")} for g in C.GAPS},
              "agrees_call_rule": {a: float(np.mean([runs[a][s]["agrees_call_rule"] for s in seeds])) for a in ARMS},
              "lowest_after_2000": {a: {s: float(min(runs[a][s]["curve"][LOW_FROM:])) for s in seeds} for a in ARMS},
              "by_seed": {s: dict(by[s], survival={a: runs[a][s]["survival"] for a in ARMS}) for s in seeds},
              "at_24000": {"survival": {a: float(np.mean([mid[a][s]["survival"] for s in mid_seeds])) for a in ARMS},
                           "m_call": TS.interval([mid_by[s]["m_call"] for s in mid_seeds]),
                           "m_action": TS.interval([mid_by[s]["m_action"] for s in mid_seeds]),
                           "by_seed": {s: {"m_call": mid_by[s]["m_call"], "m_action": mid_by[s]["m_action"],
                                           "survival": {a: mid[a][s]["survival"] for a in ARMS}} for s in mid_seeds}}}
    return {"verdicts": out, "values": values}


# ----------------------------------------------------------------------------------------------------- torch part

def measure(net, arm, held, grafts):
    survived, agrees = C.survival(net, arm, stream=WORLDS_STREAM)
    row = {"survival": float(np.mean(survived)), "survived": survived, "agrees_call_rule": agrees}
    if arm == "loop":
        row["grafts"] = C.measure_grafts(C.Reader(net, arm), held, grafts)
    return row


def job(root, arm, seed, held, grafts, log):
    import torch
    checkpoint = Path(root) / "checkpoints" / f"{arm}-{seed}.pt"
    mid = Path(root) / "mid" / f"{arm}-{seed}.json"
    if not mid.exists():
        if checkpoint.exists() and torch.load(checkpoint, weights_only=False)["update"] > MID:
            raise ValueError(f"{checkpoint} is past {MID} updates without its measure at {MID}")
        net, _ = C.train(arm, seed, {"updates": MID}, log, checkpoint=checkpoint)
        net.eval()
        row = dict(arm=arm, seed=seed, updates=MID, **measure(net, arm, held, grafts))
        mid.write_text(json.dumps(row) + "\n")
        log(f"{arm} seed {seed} at {MID}: survival {row['survival']:.3f}")
    net, curve = C.train(arm, seed, CONFIG, log, checkpoint=checkpoint)
    net.eval()
    row = dict(arm=arm, seed=seed, updates=CONFIG["updates"], curve=curve, **measure(net, arm, held, grafts))
    (Path(root) / "runs" / f"{arm}-{seed}.json").write_text(json.dumps(row) + "\n")
    log(f"{arm} seed {seed}: survival {row['survival']:.3f}, calls following the rule {row['agrees_call_rule']:.3f}")


def setup(root, held, log):
    """On seed 30, from the final checkpoints: the cut (under "cut", changing an earlier event changes nothing later;
    4 cases) and the self graft (the loop's own state grafted back changes nothing; 4 cases)."""
    import torch
    nets = {}
    for arm in ARMS:
        checkpoint = Path(root) / "checkpoints" / f"{arm}-{SEEDS[0]}.pt"
        if torch.load(checkpoint, weights_only=False)["update"] != CONFIG["updates"]:
            raise ValueError(f"{checkpoint} is not at {CONFIG['updates']} updates")
        nets[arm], _ = C.train(arm, SEEDS[0], CONFIG, log, checkpoint=checkpoint)
        nets[arm].eval()
    cut, selfs = C.Reader(nets["cut"], "cut"), C.Reader(nets["loop"], "loop")
    gaps_cut, gaps_self = [], []
    for life in [l for l in held if len(W.decisions(l)) >= 6][:4]:
        ds = W.decisions(life)
        changed = json.loads(json.dumps(life))
        changed["turns"][1]["event"] = (life["turns"][1]["event"] + 1) % len(W.EVENTS)
        a, b = cut.read(life), cut.read(changed)
        gaps_cut.append(max(max(abs(a[k][x["t"]] - b[k][x["t"]]) for x in ds[2:]) for k in (0, 1)))
        j, t = ds[2]["t"], ds[4]["t"]
        own = selfs.read(life)
        p = selfs.read(life, upto=t, graft=(j, own[2][j]))
        gaps_self.append(max(abs(p[0][t] - own[0][t]), abs(p[1][t] - own[1][t])))
    (Path(root) / "setup.json").write_text(json.dumps({"cut_gap": float(max(gaps_cut)),
                                                       "self_graft_gap": float(max(gaps_self)), "threads": 1},
                                                      indent=1) + "\n")
    log("setup written")


def run(a):
    import torch
    torch.set_num_threads(1)
    log = lambda m: print(m, flush=True)
    root = Path(a.out)
    for d in ("runs", "mid", "checkpoints", "claims"):
        (root / d).mkdir(parents=True, exist_ok=True)
    held = with_rule_action(lives())
    grafts = draw(held)
    while True:
        left = [(arm, s) for arm, s in JOBS if not (root / "runs" / f"{arm}-{s}.json").exists()]
        if not (root / "setup.json").exists():
            left.append(("setup", SEEDS[0]))
        if not left:
            return
        for name, seed in left:
            if name == "setup" and any(not (root / "runs" / f"{arm}-{seed}.json").exists() for arm in ARMS):
                continue
            if Q.claim(root, name, seed):
                done = root / ("setup.json" if name == "setup" else f"runs/{name}-{seed}.json")
                if done.exists():
                    (root / "claims" / f"{name}-{seed}").unlink()
                    break
                if name == "setup":
                    setup(root, held, log)
                else:
                    job(root, name, seed, held, grafts, log)
                (root / "claims" / f"{name}-{seed}").unlink()
                break
        else:
            if all(Q.taken(root, n, s) for n, s in left):
                return
            time.sleep(60)


def gather(root):
    out = Path(root)
    runs, mid = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    for name, store in (("runs", runs), ("mid", mid)):
        for path in sorted((out / name).glob("*.json")):
            r = json.loads(path.read_text())
            store[r["arm"]][str(r["seed"])] = r
    setup_checks = json.loads((out / "setup.json").read_text())
    result = verdicts(runs, draw(with_rule_action(lives())), setup_checks, mid)
    return json.loads(json.dumps(dict(result, setup=setup_checks)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
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
