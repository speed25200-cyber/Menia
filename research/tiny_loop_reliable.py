"""The reliable loop (docs/TINY_LOOP_RELIABLE_PROTOCOL.md): LOOP1 to LOOP4 of test 33, test 34.

Test 33 again (research/tiny_loop_gate.py) with the setting the second pilot chose (R3 of research/tiny_loop_pilot2.py:
24 000 updates), for the loop and the cut alike, new seeds 20 to 29 and new graft lives (stream 81; draw stream 82).
Published without threshold besides those of test 33: the lowest point of each learning curve after 2 000 updates, and
the survival and the grafts measured on the way at 12 000 updates (the setting of test 33). Several workers share the
jobs (one per arm and seed, then the checks of seed 20). The verdicts are numpy only; training and measures need torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_relief as RL
from . import tiny_loop as TL
from . import tiny_loop_gate as TG
from . import tiny_loop_pilot as TLP
from . import tiny_loop_pilot2 as Q
from . import tiny_survival as TS

SETTING = "R3"
CONFIG = Q.CONFIG[SETTING]
SEEDS = tuple(range(20, 30))
ARMS = TL.ARMS
LIVES_STREAM, DRAW_STREAM = 81, 82
LOW_FROM = 20  # curve points are means over 100 updates: from 2 000 updates on
MID = 12000
ROOT = "artifacts/tiny-loop-reliable"
JOBS = [(arm, seed) for seed in SEEDS for arm in ARMS]


def lives():
    return RL.writer_lives(LIVES_STREAM, TL.LIVES)


def draw(held):
    return TL.draw(held, DRAW_STREAM)


def lows(runs):
    """Published without threshold: per arm and seed, the lowest point of the learning curve after 2 000 updates."""
    return {arm: {s: float(min(runs[arm][s]["curve"][LOW_FROM:])) for s in sorted(runs[arm], key=int)} for arm in runs}


def at_mid(mid, grafts):
    """Published without threshold: at 12 000 updates, the survival of A and B, m per seed and the seeds with m > 0.05."""
    seeds = sorted(mid["loop"], key=int)
    m = {s: Q.carried(mid["loop"][s]["grafts"], grafts) for s in seeds}
    return {"survival": {a: float(np.mean([mid[a][s]["survival"] for s in seeds])) for a in mid},
            "survival_gain": TS.interval([mid["loop"][s]["survival"] - mid["cut"][s]["survival"] for s in seeds]),
            "m": m, "m_mean": TS.interval(list(m.values())),
            "seeds_above": int(sum(v > TL.LOOP3_SEED for v in m.values()))}


def verdicts(runs, grafts, setup, event_rule, held, mid):
    result = TG.verdicts(runs, grafts, setup, event_rule, held)
    result["values"]["lowest_after_2000"] = lows(runs)
    result["values"]["at_12000"] = at_mid(mid, grafts)
    return result


# ----------------------------------------------------------------------------------------------------- torch part

def job(root, arm, seed, held, grafts, log):
    import torch
    checkpoint = Path(root) / "checkpoints" / f"{arm}-{seed}.pt"
    mid = Path(root) / "mid" / f"{arm}-{seed}.json"
    if not mid.exists():
        if checkpoint.exists() and torch.load(checkpoint, weights_only=False)["update"] > MID:
            raise ValueError(f"{checkpoint} is past {MID} updates without its measure at {MID}")
        net, _ = TLP.train(arm, seed, dict(CONFIG, updates=MID), log, checkpoint=checkpoint)
        net.eval()
        survived, agrees = TL.survival(net, arm)
        row = {"arm": arm, "seed": seed, "updates": MID, "survival": float(np.mean(survived)), "agrees": agrees,
               "grafts": TL.measure_grafts(TL.Reader(net, arm), held, grafts)}
        mid.write_text(json.dumps(row) + "\n")
        log(f"{arm} seed {seed} at {MID}: survival {row['survival']:.3f}"
            + (f", m {Q.carried(row['grafts'], grafts):+.3f}" if arm == "loop" else ""))
    net, curve = TLP.train(arm, seed, CONFIG, log, checkpoint=checkpoint)
    net.eval()
    survived, agrees = TL.survival(net, arm)
    reader = TL.Reader(net, arm)
    untrained = TLP.model(seed, CONFIG["gate"])
    untrained.eval()
    row = {"arm": arm, "seed": seed, "setting": SETTING, "survival": float(np.mean(survived)), "survived": survived,
           "agrees": agrees, "curve": curve, "probe": TL.probe(reader, held),
           "probe_untrained": TL.probe(TL.Reader(untrained, arm), held), "grafts": TL.measure_grafts(reader, held, grafts)}
    if arm == "loop":
        row["gate_mean"] = TG.gate_mean(net, held)
    (Path(root) / "runs" / f"{arm}-{seed}.json").write_text(json.dumps(row) + "\n")
    log(f"{arm} seed {seed}: survival {row['survival']:.3f}, agrees with the needs rule {agrees:.3f}")


def setup(root, held, log):
    """The checks of test 32 on seed 20, once both of its runs are done (their final checkpoints, without training)."""
    nets = {}
    for arm in ARMS:
        net, _ = TLP.train(arm, SEEDS[0], CONFIG, log, checkpoint=Path(root) / "checkpoints" / f"{arm}-{SEEDS[0]}.pt")
        net.eval()
        nets[arm] = net
    (Path(root) / "setup.json").write_text(json.dumps(TL.checks(nets["loop"], nets["cut"], held), indent=1) + "\n")
    log("setup written")


def run(a):
    import time
    import torch
    torch.set_num_threads(1)
    log = lambda m: print(m, flush=True)
    root = Path(a.out)
    for d in ("runs", "mid", "checkpoints", "claims"):
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
    runs = {arm: {} for arm in ARMS}
    for path in sorted((out / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        runs[r["arm"]][str(r["seed"])] = r
    mid = {arm: {} for arm in ARMS}
    for path in sorted((out / "mid").glob("*.json")):
        r = json.loads(path.read_text())
        mid[r["arm"]][str(r["seed"])] = r
    setup = json.loads((out / "setup.json").read_text())
    held = lives()
    result = verdicts(runs, draw(held), setup, TS.event_rule_survival(), held, mid)
    return json.loads(json.dumps(dict(result, setup=setup)))


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
