"""The wager: does the loop know when it does not know? (docs/TINY_WAGER_PROTOCOL.md): BET1 to BET6 of test 37.

The world, agent and training of the pilot (research/tiny_wager.py) with the setting it chose (P1: 48 000 updates), for
the loop and the cut alike; new seeds 40 to 49; reward, survival and wagers on 256 new worlds (stream 136); 128 new
graft lives (stream 137; draw stream 138) with three kinds of grafts between lives that took the same action at j: (u)
the rule's action at t is the same and its wager changes ("close" when the means of E and N at t differ by at most 0.5);
(k) its action changes and its wager does not; (b) the same belief from another life. The verdicts are numpy only;
training and measures need torch.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from . import need_world as W
from . import tiny_loop_pilot2 as Q
from . import tiny_survival as TS
from . import tiny_wager as X

CONFIG = X.CONFIG["P1"]
SEEDS = tuple(range(40, 50))
ARMS = ("loop", "cut")
WORLDS_STREAM, LIVES_STREAM, DRAW_STREAM = 136, 137, 138
BET1, BET2, BET3, SEED_LEVEL, SEEDS_NEEDED, HALF = 1.0, 0.10, 0.10, 0.05, 8, 0.5
MIN_GRAFTS, MIN_CLOSE, TOLERANCE = 100, 50, 1e-6
LOW_FROM = 20
ROOT = "artifacts/tiny-wager-test"
JOBS = [(arm, seed) for seed in SEEDS for arm in ARMS]


def lives():
    return X.lives(LIVES_STREAM)


def draw(held):
    return X.draw(held, stream=DRAW_STREAM)


def exact_auc(decisions):
    """The AUC of P(wager) for served against not served within groups of equal rule probability (rounded to 1e-4),
    averaged with the weights of the groups that have both outcomes; and the share of decisions in those groups."""
    groups = defaultdict(list)
    for d in decisions:
        groups[round(d["p_rule"], 4)].append(d)
    aucs, weights = [], []
    for v in groups.values():
        a = X.auc([d["p_wager"] for d in v], [d["served"] for d in v])
        if a is not None:
            aucs.append(a)
            weights.append(len(v))
    return {"auc_exact": float(np.average(aucs, weights=weights)) if aucs else None,
            "share_exact": float(sum(weights) / len(decisions))}


def per_seed(rows, grafts):
    """For one loop: the effects of research/tiny_wager.py, and m_wager and m_action per gap."""
    out = X.effects(rows, grafts)
    by = {g["index"]: g for g in grafts}
    for g in X.GAPS:
        u = [r for r in rows if by[r[0]]["kind"] == "u" and by[r[0]]["g"] == g]
        k = [r for r in rows if by[r[0]]["kind"] == "k" and by[r[0]]["g"] == g]
        out[f"m_wager_g{g}"] = float(np.mean([(r[4] - r[3]) * by[r[0]]["e"] for r in u]))
        out[f"m_action_g{g}"] = float(np.mean([(r[2] - r[1]) * by[r[0]]["e"] for r in k]))
    k = [r for r in rows if by[r[0]]["kind"] == "k"]
    out["signed_wager_k"] = float(np.mean([r[4] - r[3] for r in k]))
    return out


def average(xs):
    """The mean of the values that exist (a run without any wager has no win rate)."""
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def verdicts(runs, grafts, setup):
    """runs: {arm: {seed: {"reward", "survival", "wager_rate", "win_rate", "decisions", "curve", "draw",
    "grafts" (loop)}}}; grafts: the draw; setup: the checks."""
    seeds = sorted(runs["loop"], key=int)
    by = {s: per_seed(runs["loop"][s]["grafts"], grafts) for s in seeds}
    vals = lambda key: [by[s][key] for s in seeds]
    mean = lambda key: float(np.mean(vals(key)))
    gain = TS.interval([runs["loop"][s]["reward"] - runs["cut"][s]["reward"] for s in seeds])
    m = TS.interval(vals("m_wager"))
    close = TS.interval(vals("m_wager_close"))
    up, down = TS.interval(vals("m_wager_up")), TS.interval(vals("m_wager_down"))
    u_diff = TS.interval([by[s]["abs_wager_u"] - by[s]["abs_action_u"] for s in seeds])
    u_half = TS.interval([HALF * by[s]["abs_wager_u"] - by[s]["abs_action_u"] for s in seeds])
    b_diff = TS.interval([by[s]["abs_wager_u"] - by[s]["abs_wager_b"] for s in seeds])
    counts = {k: sum(g["kind"] == k for g in grafts) for k in X.KINDS}
    counts["u_close"] = sum(bool(g.get("close")) for g in grafts)
    fingerprint = X.fingerprint(grafts)
    out = {"BET1": bool(gain["mean"] >= BET1 and gain["low"] > 0),
           "BET2": bool(m["mean"] >= BET2 and m["low"] > 0 and sum(v > SEED_LEVEL for v in vals("m_wager")) >= SEEDS_NEEDED),
           "BET3": bool(close["mean"] >= BET3 and close["low"] > 0),
           "BET4": bool(up["low"] > 0 and down["low"] > 0),
           "BET5": bool(mean("abs_action_u") <= HALF * mean("abs_wager_u") and u_diff["low"] > 0),
           "BET6": bool(mean("abs_wager_b") <= HALF * mean("abs_wager_u") and b_diff["low"] > 0)}
    out["valid"] = bool(setup["cut_gap"] <= TOLERANCE and setup["self_graft_gap"] <= TOLERANCE
                        and all(counts[k] >= MIN_GRAFTS for k in X.KINDS) and counts["u_close"] >= MIN_CLOSE
                        and all(len(runs[a]) == len(SEEDS) for a in ARMS)
                        and all(runs[a][s]["draw"] == fingerprint for a in ARMS for s in runs[a]))
    out["global"] = bool(out["valid"] and all(out[f"BET{i}"] for i in range(1, 7)))
    second = {a: {s: dict(X.second_order(runs[a][s]["decisions"]), **exact_auc(runs[a][s]["decisions"])) for s in seeds}
              for a in ARMS}
    values = {"reward": {a: float(np.mean([runs[a][s]["reward"] for s in seeds])) for a in ARMS},
              "survival": {a: float(np.mean([runs[a][s]["survival"] for s in seeds])) for a in ARMS},
              "wager_rate": {a: float(np.mean([runs[a][s]["wager_rate"] for s in seeds])) for a in ARMS},
              "win_rate": {a: average([runs[a][s]["win_rate"] for s in seeds]) for a in ARMS},
              "reward_gain": gain, "m_wager": m, "m_wager_close": close, "m_wager_up": up, "m_wager_down": down,
              "seeds_above": int(sum(v > SEED_LEVEL for v in vals("m_wager"))),
              "wager_minus_action_u": u_diff, "half_wager_minus_action_u": u_half, "wager_u_minus_b": b_diff,
              "abs": {k: {"action": mean(f"abs_action_{k}"), "wager": mean(f"abs_wager_{k}")} for k in X.KINDS},
              "k_side": {"m_action": TS.interval(vals("m_action")), "signed_wager_k": mean("signed_wager_k")},
              "signed_wager_b": mean("signed_wager_b"), "grafts": counts,
              "by_gap": {f"g{g}": {"m_wager": mean(f"m_wager_g{g}"), "m_action": mean(f"m_action_g{g}")} for g in X.GAPS},
              "second_order": {a: {k: average([second[a][s][k] for s in seeds]) for k in ("auc", "auc_within",
                                                                                          "auc_exact")}
                               for a in ARMS},
              "lowest_after_2000": {a: {s: float(min(runs[a][s]["curve"][LOW_FROM:])) for s in seeds} for a in ARMS},
              "by_seed": {s: dict(by[s], reward={a: runs[a][s]["reward"] for a in ARMS},
                                  survival={a: runs[a][s]["survival"] for a in ARMS},
                                  second_order={a: second[a][s] for a in ARMS}) for s in seeds}}
    return {"verdicts": out, "values": values}


# ----------------------------------------------------------------------------------------------------- torch part

def job(root, arm, seed, held, grafts, log):
    net, curve = X.train(arm, seed, CONFIG, log, checkpoint=Path(root) / "checkpoints" / f"{arm}-{seed}.pt")
    net.eval()
    x = X.evaluate(net, arm, stream=WORLDS_STREAM)
    row = {"arm": arm, "seed": seed, "updates": CONFIG["updates"], "survival": float(np.mean(x["survived"])),
           "survived": x["survived"], "reward": x["reward"], "wager_rate": x["wager_rate"], "win_rate": x["win_rate"],
           "decisions": x["decisions"], "curve": curve, "draw": X.fingerprint(grafts)}
    text = f"{arm} seed {seed}: survival {row['survival']:.3f}, reward {row['reward']:.2f}"
    if arm == "loop":
        row["grafts"] = X.measure_grafts(X.Reader(net, arm), held, grafts)
    log(text)
    W.write_atomic(Path(root) / "runs" / f"{arm}-{seed}.json", (json.dumps(row) + "\n").encode())


def setup(root, held, log):
    """On seed 40, from the final checkpoints: the cut (under "cut", changing an earlier event changes nothing later;
    4 cases) and the self graft (the loop's own state grafted back changes nothing; 4 cases)."""
    import torch
    nets = {}
    for arm in ARMS:
        checkpoint = Path(root) / "checkpoints" / f"{arm}-{SEEDS[0]}.pt"
        if torch.load(checkpoint, weights_only=False)["update"] != CONFIG["updates"]:
            raise ValueError(f"{checkpoint} is not at {CONFIG['updates']} updates")
        nets[arm], _ = X.train(arm, SEEDS[0], CONFIG, log, checkpoint=checkpoint)
        nets[arm].eval()
    cut, selfs = X.Reader(nets["cut"], "cut"), X.Reader(nets["loop"], "loop")
    gaps_cut, gaps_self = [], []
    for life in [l for l in held if len(W.decisions(l)) >= 6][:4]:
        ds = W.decisions(life)
        changed = {"turns": [dict(x) for x in life["turns"]], "survived": life["survived"]}
        changed["turns"][1]["event"] = (life["turns"][1]["event"] + 3) % len(W.EVENTS)
        changed["turns"][1]["hidden"] = False
        a, b = cut.read(life), cut.read(changed)
        gaps_cut.append(max(max(abs(a[k][x["t"]] - b[k][x["t"]]) for x in ds[2:]) for k in (0, 1)))
        j, t = ds[2]["t"], ds[4]["t"]
        own = selfs.read(life)
        p = selfs.read(life, upto=t, graft=(j, own[2][j]))
        gaps_self.append(max(abs(p[0][t] - own[0][t]), abs(p[1][t] - own[1][t])))
    W.write_atomic(Path(root) / "setup.json", (json.dumps({"cut_gap": float(max(gaps_cut)),
                                                          "self_graft_gap": float(max(gaps_self)), "threads": 1},
                                                         indent=1) + "\n").encode())
    log("setup written")


def run(a):
    import torch
    torch.set_num_threads(1)
    log = lambda m: print(m, flush=True)
    root = Path(a.out)
    for d in ("runs", "checkpoints", "claims"):
        (root / d).mkdir(parents=True, exist_ok=True)
    held = lives()
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
    runs = {a: {} for a in ARMS}
    for path in sorted((out / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        runs[r["arm"]][str(r["seed"])] = r
    setup_checks = json.loads((out / "setup.json").read_text())
    result = verdicts(runs, draw(lives()), setup_checks)
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
