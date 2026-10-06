"""The second pilot before test 34 (docs/TINY_LOOP_PILOT2_PLAN.md): which way of learning makes the gated loop of test 33
carry the needs in its state on every seed? Four settings (R1: test 33 as is; R2: R1 with the norm of the gradient
bounded to 1; R3: R1 with 24 000 updates; R4: R2 with 24 000 updates), seeds 100 to 103, the cut control once per seed
with R1. Per setting and seed: the survival on the 256 pilot worlds (stream 73) and m, the mean over the (a) grafts of
the change of P(R) times the rule effect e, on 128 pilot graft lives (stream 79; draw stream 80). R1 on seeds 100 to 102
is taken from the checkpoints of the first pilot (its L4); R3 and R4 go on from R1 and R2 at 12 000 updates. Several
workers share the jobs. The choice is numpy only.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
from . import need_relief as RL
from . import tiny_loop as TL
from . import tiny_loop_pilot as TLP

SETTINGS = ("R1", "R2", "R3", "R4")
BASE = TLP.CONFIG["L4"]
CONFIG = {"R1": dict(BASE), "R2": dict(BASE, clip=1.0), "R3": dict(BASE, updates=24000),
          "R4": dict(BASE, updates=24000, clip=1.0)}
FROM = {"R3": "R1", "R4": "R2"}
SEEDS = (100, 101, 102, 103)
FIRST_SEEDS = (100, 101, 102)
LIVES_STREAM, DRAW_STREAM = 79, 80
NEEDED, CARRIED, TIE = 0.04, 0.05, 0.01
ROOT = "artifacts/tiny-loop-pilot2"
FIRST = "artifacts/tiny-loop-pilot"
JOBS = ([(s, seed) for s in ("R1", "R2") for seed in SEEDS] + [(s, seed) for s in ("R3", "R4") for seed in SEEDS]
        + [("cut", seed) for seed in SEEDS])


def lives():
    return RL.writer_lives(LIVES_STREAM, TL.LIVES)


def draw(held):
    return TL.draw(held, DRAW_STREAM)


def carried(rows, grafts):
    """m: the mean over the (a) grafts of (P(R) with the graft - P(R) without) times the rule effect e."""
    by = {g["index"]: g for g in grafts}
    if sorted(r[0] for r in rows) != sorted(by):
        raise ValueError("the rows do not match the draw")
    return float(np.mean([(p1 - p0) * by[i]["e"] for i, p0, p1 in rows if by[i]["kind"] == "a"]))


def choose(loops, cuts):
    """loops: {setting: {seed: {"survival", "m"}}}; cuts: {seed: survival}. A setting is reliable if, on each of the
    four seeds, m > CARRIED and its survival exceeds the cut of the same seed by at least NEEDED. The reliable setting
    with the largest mean m; at a tie within TIE, the simplest (the first in SETTINGS); None if none is reliable."""
    table = {}
    for s in SETTINGS:
        if s not in loops:
            continue
        seeds = sorted(loops[s], key=int)
        m = {k: loops[s][k]["m"] for k in seeds}
        gain = {k: loops[s][k]["survival"] - cuts[k] for k in seeds}
        table[s] = {"m_mean": float(np.mean(list(m.values()))), "gain_mean": float(np.mean(list(gain.values()))),
                    "m": m, "gain": gain,
                    "reliable": bool(sorted(seeds) == sorted(str(k) for k in SEEDS)
                                     and all(v > CARRIED for v in m.values())
                                     and all(v >= NEEDED for v in gain.values()))}
    ok = [s for s in table if table[s]["reliable"]]
    if not ok:
        return None, table
    best = max(table[s]["m_mean"] for s in ok)
    return next(s for s in SETTINGS if s in ok and table[s]["m_mean"] >= best - TIE), table


def summary(root, first=FIRST):
    grafts = draw(lives())
    loops, cuts, first_pilot = {}, {}, {}
    for path in sorted((Path(root) / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        seed = str(r["seed"])
        if r["name"] == "cut":
            cuts[seed] = r["survival"]
            continue
        loops.setdefault(r["name"], {})[seed] = {"survival": r["survival"], "m": carried(r["grafts"], grafts)}
        if r["name"] == "R1" and r["seed"] in FIRST_SEEDS:
            published = json.loads((Path(first) / f"L4-{seed}.json").read_text())["survival"]
            first_pilot[seed] = {"published": published, "again": r["survival"]}
    chosen, table = choose(loops, cuts)
    kinds = [g["kind"] for g in grafts]
    return {"chosen": chosen, "table": table, "cuts": cuts,
            "grafts": {k: kinds.count(k) for k in TL.COUNTS},
            "first_pilot_again": dict(first_pilot, same=all(v["published"] == v["again"] for v in first_pilot.values()))}


# ----------------------------------------------------------------------------------------------------- torch part

def ready(root, name, seed, first=FIRST):
    """Whether the job can start; R1 on seeds 100 to 102 starts from the first pilot's L4 checkpoint, R3 and R4 from the
    checkpoint of R1 and R2 at 12 000 updates once that job is done (copied once, before training)."""
    import torch
    target = Path(root) / "checkpoints" / f"{name}-{seed}.pt"
    if target.exists():
        return True
    if name == "R1" and seed in FIRST_SEEDS:
        source = Path(first) / "checkpoints" / f"L4-{seed}.pt"
    elif name in FROM:
        if not (Path(root) / "runs" / f"{FROM[name]}-{seed}.json").exists():
            return False
        source = Path(root) / "checkpoints" / f"{FROM[name]}-{seed}.pt"
    else:
        return True
    if torch.load(source, weights_only=False)["update"] != BASE["updates"]:
        raise ValueError(f"{source} is not at {BASE['updates']} updates")
    shutil.copyfile(source, target)
    return True


def taken(root, name, seed):
    """Whether a living process holds the job (the claim of a process that died is removed)."""
    path = Path(root) / "claims" / f"{name}-{seed}"
    try:
        os.kill(int(path.read_text()), 0)
        return True
    except FileNotFoundError:
        return False
    except (ProcessLookupError, ValueError):
        path.unlink(missing_ok=True)
        return False


def claim(root, name, seed):
    """Take the job for this process, unless another living process holds it."""
    path = Path(root) / "claims" / f"{name}-{seed}"
    if taken(root, name, seed):
        return False
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def job(root, name, seed, held, grafts, log):
    arm = "cut" if name == "cut" else "loop"
    config = CONFIG["R1" if name == "cut" else name]
    net, curve = TLP.train(arm, seed, config, log, checkpoint=Path(root) / "checkpoints" / f"{name}-{seed}.pt")
    net.eval()
    row = {"name": name, "arm": arm, "seed": seed, "config": config, "survival": TLP.measure(net, arm), "curve": curve}
    if arm == "loop":
        row["grafts"] = TL.measure_grafts(TL.Reader(net, arm), held, grafts)
        log(f"{name} seed {seed}: survival {row['survival']:.3f}, m {carried(row['grafts'], grafts):+.3f}")
    else:
        log(f"{name} seed {seed}: survival {row['survival']:.3f}")
    (Path(root) / "runs" / f"{name}-{seed}.json").write_text(json.dumps(row) + "\n")


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
        left = [(n, s) for n, s in JOBS if not (root / "runs" / f"{n}-{s}.json").exists()]
        if not left:
            return
        for name, seed in left:
            if ready(root, name, seed) and claim(root, name, seed):
                job(root, name, seed, held, grafts, log)
                (root / "claims" / f"{name}-{seed}").unlink()
                break
        else:
            if all(taken(root, n, s) for n, s in left):
                return
            time.sleep(60)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
    sub.add_parser("choose")
    sub.add_parser("check")
    for s in sub.choices.values():
        s.add_argument("--out", default=ROOT)
    a = parser.parse_args(argv)
    if a.command == "run":
        run(a)
        return
    result = summary(a.out)
    path = Path(a.out) / "choice.json"
    if a.command == "choose":
        path.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({"chosen": result["chosen"], "first_pilot_again": result["first_pilot_again"]["same"]}))
    else:
        differs = json.loads(json.dumps(result)) != json.loads(path.read_text())
        print("differs:", "yes" if differs else "none")
        if differs:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
