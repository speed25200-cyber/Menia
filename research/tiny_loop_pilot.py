"""The pilot before test 33 (docs/TINY_LOOP_PILOT_PLAN.md): which way of learning makes the loop of test 32 use its
state? Four settings for the loop (L1: 12 000 updates; L2: learning rate 1e-3; L3: a gated state; L4: gated state and
12 000 updates), the cut control once per seed with the setting of test 32, seeds 100 to 102, measured on 256 pilot
worlds (stream 73) apart from the 256 worlds of measure. World, reward, network and actor-critic are those of
research/tiny_loop.py. The choice is numpy only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import tiny_loop as TL

SETTINGS = ("L1", "L2", "L3", "L4")
CONFIG = {"L1": {"updates": 12000, "lr": 3e-4, "gate": False},
          "L2": {"updates": 4000, "lr": 1e-3, "gate": False},
          "L3": {"updates": 4000, "lr": 3e-4, "gate": True},
          "L4": {"updates": 12000, "lr": 3e-4, "gate": True}}
CUT = {"updates": TL.UPDATES, "lr": TL.LR, "gate": False}
SEEDS = (100, 101, 102)
PILOT_WORLDS = 73
NEEDED, COLLAPSE, TIE = 0.04, 0.05, 0.005
ROOT = "artifacts/tiny-loop-pilot"


def choose(loops, cuts):
    """loops: {setting: {seed: survival}}; cuts: {seed: survival}. The setting with the largest mean gain over the cut
    of the same seed, if it reaches NEEDED and no seed falls below its cut by more than COLLAPSE; at a tie within TIE,
    the simplest (the first in SETTINGS); None if none qualifies."""
    gains, fit = {}, {}
    for s in SETTINGS:
        if s not in loops:
            continue
        diffs = [loops[s][k] - cuts[k] for k in loops[s]]
        gains[s] = float(np.mean(diffs))
        fit[s] = gains[s] >= NEEDED and min(diffs) >= -COLLAPSE
    ok = [s for s in gains if fit[s]]
    if not ok:
        return None, gains
    best = max(gains[s] for s in ok)
    return next(s for s in SETTINGS if s in ok and gains[s] >= best - TIE), gains


# ----------------------------------------------------------------------------------------------------- torch part

def model(seed, gate):
    """The network of test 32; with gate, the next state is (1 - z) * s + z * n, n the normalized output of "Choix"
    (the next state of test 32) and z = sigmoid(W h + b), h that output (b starts at 0)."""
    import torch
    from torch import nn
    base = TL.model(seed)

    class Pilot(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            if gate:
                self.gate = nn.Linear(TL.DIM, TL.DIM)
                nn.init.zeros_(self.gate.bias)

        def initial(self, n):
            return self.base.initial(n)

        def step(self, state, last, event):
            b = self.base
            n = state.shape[0]
            choice = torch.full((n,), TL.CHOICE, dtype=torch.long)
            x = torch.stack([state, b.tok(last), b.tok(event), b.tok(choice)], 1) + b.pos.weight[None]
            h = b.layers(x)[:, -1]
            new = nn.functional.layer_norm(h, (TL.DIM,))
            if gate:
                z = torch.sigmoid(self.gate(h))
                new = (1 - z) * state + z * new
            return b.out(h), b.value_head(h)[:, 0], new
    return Pilot()


def train(arm, seed, config, log, checkpoint=None):
    """The training of research/tiny_loop.py with the given number of updates, learning rate and gate (and, if the
    config has "clip", the norm of the gradient bounded to it); kept every 250 updates and taken back exactly after a
    stop."""
    import torch
    net = model(seed, config["gate"])
    opt = torch.optim.AdamW(net.parameters(), lr=config["lr"])
    curve, window, start = [], [], 0
    if checkpoint is not None and Path(checkpoint).exists():
        state = torch.load(checkpoint, weights_only=False)
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        curve, window, start = state["curve"], state["window"], state["update"]
        log(f"  {checkpoint}: resumed at update {start}")
    for update in range(start, config["updates"]):
        worlds = [np.random.default_rng([W.SEED, TL.WORLD_STREAM, seed, update, i]) for i in range(TL.LIVES_PER_UPDATE)]
        rng = np.random.default_rng([W.SEED, TL.ACTION_STREAM, seed, update])
        decisions, logps, entropies, values, length = TL.episodes(
            net, arm, worlds, lambda p, last: (rng.random(len(p)) >= p).astype(int), grad=True)
        life = np.array([d[0] for d in decisions])
        turn = np.array([d[1] for d in decisions])
        lived = (length[life] - turn).astype(float)
        ret = torch.tensor((1 - TL.GAMMA ** lived) / (1 - TL.GAMMA), dtype=torch.float32)
        chosen, entropy, v = torch.cat(logps), torch.cat(entropies), torch.cat(values)
        with torch.no_grad():
            advantage = ret - v
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        loss = -(advantage * chosen).mean() + TL.VALUE * ((v - ret) ** 2).mean() - TL.ENTROPY * entropy.mean()
        opt.zero_grad()
        loss.backward()
        if config.get("clip"):
            torch.nn.utils.clip_grad_norm_(net.parameters(), config["clip"])
        opt.step()
        window.append(float((length == W.TURNS).mean()))
        if (update + 1) % 100 == 0:
            curve.append(round(float(np.mean(window)), 4))
            window = []
        if checkpoint is not None and (update + 1) % 250 == 0:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "curve": curve, "window": window,
                        "update": update + 1}, checkpoint)
    return net, curve


def measure(net, arm):
    worlds = [W.world_rng(PILOT_WORLDS, i) for i in range(256)]
    greedy = lambda p, last: np.where(p > 0.5, 0, np.where(p < 0.5, 1, 1 - last))
    _, _, _, _, length = TL.episodes(net, arm, worlds, greedy)
    return float((length == W.TURNS).mean())


def run(a):
    import torch
    torch.set_num_threads(1)
    log = lambda m: print(m, flush=True)
    out = Path(a.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    jobs = [("cut", "cut", CUT)] if a.cut else []
    jobs += [(s, "loop", CONFIG[s]) for s in a.settings]
    for seed in a.seeds:
        for name, arm, config in jobs:
            path = out / f"{name}-{seed}.json"
            if path.exists():
                continue
            net, curve = train(arm, seed, config, log, checkpoint=out / "checkpoints" / f"{name}-{seed}.pt")
            net.eval()
            survival = measure(net, arm)
            path.write_text(json.dumps({"name": name, "arm": arm, "seed": seed, "config": config,
                                        "survival": survival, "curve": curve}) + "\n")
            log(f"{name} seed {seed}: survival on the pilot worlds {survival:.3f}")


def summary(root):
    loops, cuts = {}, {}
    for path in sorted(Path(root).glob("*.json")):
        if path.name == "choice.json":
            continue
        r = json.loads(path.read_text())
        if r["name"] == "cut":
            cuts[str(r["seed"])] = r["survival"]
        else:
            loops.setdefault(r["name"], {})[str(r["seed"])] = r["survival"]
    chosen, gains = choose(loops, cuts)
    return {"chosen": chosen, "gains": gains, "loops": loops, "cuts": cuts}


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--settings", nargs="*", choices=SETTINGS, default=list(SETTINGS))
    r.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    r.add_argument("--cut", action="store_true", help="also learn the cut control (setting of test 32)")
    sub.add_parser("choose")
    for s in sub.choices.values():
        s.add_argument("--out", default=ROOT)
    a = parser.parse_args(argv)
    if a.command == "run":
        run(a)
        return
    result = summary(a.out)
    (Path(a.out) / "choice.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"chosen": result["chosen"], "gains": result["gains"]}))


if __name__ == "__main__":
    main()
