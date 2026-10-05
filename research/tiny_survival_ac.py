"""Only surviving, with a learning that works (docs/TINY_SURVIVAL_AC_PROTOCOL.md): SURV1, SURV2 of test 31.

Test 30 (research/tiny_survival.py: world, tokens, masks, model, reward of 1 per turn lived, ten seeds, measure on the
256 worlds of test 22) with the learning chosen by the pilot on the control alone (research/tiny_survival_pilot.py,
setting P2): an actor-critic with a value head at "Choix", returns discounted by 0.9, normalized advantage, value loss
0.5, entropy 0.01, AdamW 3e-4, 4000 updates of 64 lives. The verdicts are those of test 30, numpy only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_memory as NM
from . import tiny_survival as TS
from . import tiny_survival_pilot as SP

SETTING = "P2"
ARMS = TS.ARMS
SEEDS = TS.SEEDS
ROOT = "artifacts/tiny-survival-ac"


def verdicts(runs, setup, event_rule):
    return TS.verdicts(runs, setup, event_rule)


# ----------------------------------------------------------------------------------------------------- torch part

def train(arm, seed, log, checkpoint=None):
    """The setting P2 of the pilot, for any mask. With a checkpoint path, the model, the optimizer and the curve are
    kept every 250 updates and taken back after a stop: each update draws its worlds and actions from its own streams,
    so the run resumes exactly."""
    import torch
    c = SP.CONFIG[SETTING]
    net = SP.model(seed)
    seen = torch.tensor(TS.mask(arm))
    opt = torch.optim.AdamW(net.parameters(), lr=c["lr"])
    curve, window, start = [], [], 0
    if checkpoint is not None and Path(checkpoint).exists():
        state = torch.load(checkpoint, weights_only=False)
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        curve, window, start = state["curve"], state["window"], state["update"]
        log(f"  {arm} seed {seed}: resumed at update {start}")
    g = c["gamma"]
    for update in range(start, c["updates"]):
        worlds = [np.random.default_rng([W.SEED, TS.WORLD_STREAM, seed, update, i]) for i in range(TS.LIVES_PER_UPDATE)]
        rng = np.random.default_rng([W.SEED, TS.ACTION_STREAM, seed, update])
        ids, decisions, length = TS.episodes(net, arm, worlds, lambda p, last: (rng.random(len(p)) >= p).astype(int))
        life = torch.tensor([d[0] for d in decisions])
        turn = np.array([d[1] for d in decisions])
        position = torch.tensor([d[2] for d in decisions])
        action = torch.tensor([d[3] for d in decisions])
        lived = (length[life.numpy()] - turn).astype(float)
        ret = torch.tensor((1 - g ** lived) / (1 - g), dtype=torch.float32)
        logits, value = net.both(ids, seen)
        logp = torch.log_softmax(logits[life, position][:, [TS.R, TS.M]], -1)
        chosen = logp.gather(1, action[:, None])[:, 0]
        v = value[life, position]
        with torch.no_grad():
            advantage = ret - v
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        loss = -(advantage * chosen).mean() + 0.5 * ((v - ret) ** 2).mean() \
            - TS.ENTROPY * (-(logp.exp() * logp).sum(-1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        window.append(float((length == W.TURNS).mean()))
        if (update + 1) % 100 == 0:
            curve.append(round(float(np.mean(window)), 4))
            window = []
        if checkpoint is not None and (update + 1) % 250 == 0:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "curve": curve, "window": window,
                        "update": update + 1}, checkpoint)
    log(f"  {arm} seed {seed}: training survival {curve[-1] if curve else None}")
    return net, curve


def run(a):
    import torch
    torch.set_num_threads(a.threads)
    log = lambda m: print(m, flush=True)
    root = Path(a.out)
    out = root / "runs"
    out.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    for seed in a.seeds:
        paths = {arm: out / f"{arm}-{seed}.json" for arm in a.arms}
        nets = {}
        for arm in a.arms:
            if paths[arm].exists() and not (seed == 0 and arm != "free" and not (root / "setup.json").exists()):
                continue
            net, curve = train(arm, seed, log, checkpoint=root / "checkpoints" / f"{arm}-{seed}.pt")
            nets[arm] = net
            survived, agrees = TS.measure(net, arm)
            paths[arm].write_text(json.dumps({"arm": arm, "seed": seed, "survival": float(np.mean(survived)),
                                              "survived": survived, "agrees": agrees, "curve": curve,
                                              "setting": SETTING}) + "\n")
            log(f"{arm} seed {seed}: survival {np.mean(survived):.3f}, agrees with the needs rule {agrees:.3f}")
        if seed == 0 and "route" in nets and "actions" in nets and not (root / "setup.json").exists():
            (root / "setup.json").write_text(json.dumps(TS.checks(nets), indent=1) + "\n")


def gather(root):
    out = Path(root)
    runs = {arm: {} for arm in ARMS}
    for path in sorted((out / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        runs[r["arm"]][str(r["seed"])] = r
    setup = json.loads((out / "setup.json").read_text())
    return json.loads(json.dumps(dict(verdicts(runs, setup, TS.event_rule_survival()), setup=setup)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    r.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
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
