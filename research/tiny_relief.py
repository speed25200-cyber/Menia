"""Without pre-training: do small transformers carry their needs? (docs/TINY_RELIEF_PROTOCOL.md): TINY1, TINY2 of test 29.

The lives and the only target of test 26 (research/need_relief.py: lives written by the event rule with 30 % random
actions; the relief level of each action), written as 5 simple tokens per turn (event, "Choix", action, "Soulagement",
level) at fixed positions, learned from scratch by a 2-layer transformer under the masks of test 22: "route" (a turn
sees its own tokens and the carried tokens "Choix" and action of past turns), "actions" (past actions only; an action
token sees no event), "free" (causal, published without threshold). Ten seeds per mask. Measures: the accuracy of the
predicted level on held-out lives against the exact actions-only Bayes observer, and the survival when choosing the
action whose predicted relief is larger. The verdicts are numpy only; training and measures need torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_relief as RL

BOS, CHOICE, R, M, RELIEF, END = 0, 7, 8, 9, 10, 15
EVENT0, LEVEL0 = 1, 11  # event k -> 1 + k, level l -> 11 + l
VOCAB, PER_TURN = 16, 5
LENGTH = 1 + PER_TURN * W.TURNS
ARMS = ("route", "actions", "free")
SEEDS = tuple(range(10))
STREAM = 70
STEPS, BATCH, LR = 3000, 32, 1e-3
DIM, HEADS, LAYERS = 64, 4, 2
TINY1, TINY1_EACH, TINY1_COUNT = 0.10, 0.05, 8
TINY2 = 0.08
CEILING_SLACK, MASK_TOLERANCE = 0.02, 1e-6
T_975_DF9 = 2.2621571627409915  # Student, 97.5 %, 9 degrees of freedom
ROOT = "artifacts/tiny-relief"


def tokens(life):
    """BOS, then per turn: event, CHOICE, action, RELIEF, level; a turn where the life goes out: event, END."""
    out = [BOS]
    for x in life["turns"]:
        if x.get("dead"):
            out += [EVENT0 + x["event"], END]
        else:
            out += [EVENT0 + x["event"], CHOICE, R + x["action"], RELIEF, LEVEL0 + x["level"]]
    return out


def mask(arm, length=LENGTH):
    """mask[q, k] = True if position q may see position k."""
    pos = np.arange(length)
    turn = np.where(pos == 0, -1, (pos - 1) // PER_TURN)
    offset = np.where(pos == 0, -1, (pos - 1) % PER_TURN)
    q, k = pos[:, None], pos[None, :]
    causal = k <= q
    if arm == "free":
        return causal
    bos = (k == 0)
    same = (turn[None, :] == turn[:, None]) & causal
    past = turn[None, :] < turn[:, None]
    if arm == "route":
        carried = (offset == 1) | (offset == 2)
        return bos | same | (past & carried[None, :])
    if arm == "actions":
        action = offset == 2
        out = bos | same | (past & action[None, :])
        mine = action[:, None]  # an action token sees BOS, the past actions and itself only
        return np.where(mine, bos | (k == q) | (past & action[None, :]), out)
    raise ValueError(arm)


def ceiling_accuracy(held):
    right = [float(np.argmax(p) == x["level"]) for life in held for x, p in zip(W.decisions(life), RL.ceiling(life))]
    return float(np.mean(right))


def verdicts(runs, setup, ceiling):
    """runs: {arm: {seed: {"accuracy", "survival"}}}."""
    seeds = sorted(runs["route"], key=int)
    d = np.array([runs["route"][s]["accuracy"] - ceiling for s in seeds])
    g = np.array([runs["route"][s]["survival"] - runs["actions"][s]["survival"] for s in seeds])
    interval = lambda x: {"mean": float(x.mean()),
                          "low": float(x.mean() - T_975_DF9 * x.std(ddof=1) / np.sqrt(len(x))),
                          "high": float(x.mean() + T_975_DF9 * x.std(ddof=1) / np.sqrt(len(x)))}
    beyond, gain = interval(d), interval(g)
    out = {"TINY1": bool(beyond["mean"] >= TINY1 and beyond["low"] > 0 and int((d >= TINY1_EACH).sum()) >= TINY1_COUNT),
           "TINY2": bool(gain["mean"] >= TINY2 and gain["low"] > 0)}
    accuracy = {a: float(np.mean([runs[a][s]["accuracy"] for s in seeds])) for a in runs}
    out["valid"] = bool(setup["mask_gap"]["route"] <= MASK_TOLERANCE and setup["mask_gap"]["actions"] <= MASK_TOLERANCE
                        and accuracy["actions"] <= ceiling + CEILING_SLACK and len(seeds) == len(SEEDS))
    out["global"] = bool(out["valid"] and out["TINY1"] and out["TINY2"])
    values = {"ceiling": ceiling, "accuracy": accuracy, "beyond_ceiling": beyond, "seeds_beyond_0.05": int((d >= TINY1_EACH).sum()),
              "survival": {a: float(np.mean([runs[a][s]["survival"] for s in seeds])) for a in runs},
              "survival_gain": gain,
              "by_seed": {s: {a: {k: runs[a][s][k] for k in ("accuracy", "survival")} for a in runs} for s in seeds}}
    return {"verdicts": out, "values": values}


# ----------------------------------------------------------------------------------------------------- torch part

def model(seed):
    import torch
    from torch import nn
    torch.manual_seed(int(np.random.default_rng([W.SEED, STREAM, seed]).integers(2 ** 31)))

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(VOCAB, DIM)
            self.pos = nn.Embedding(LENGTH, DIM)
            layer = nn.TransformerEncoderLayer(DIM, HEADS, 4 * DIM, dropout=0.0, batch_first=True, norm_first=True)
            self.layers = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
            self.out = nn.Linear(DIM, VOCAB)

        def forward(self, ids, seen):
            x = self.tok(ids) + self.pos(torch.arange(ids.shape[1]))[None]
            return self.out(self.layers(x, mask=~seen[:ids.shape[1], :ids.shape[1]]))
    return Tiny()


def batch(lives):
    ids = np.full((len(lives), LENGTH), END, np.int64)
    target = np.zeros((len(lives), LENGTH), bool)
    for i, life in enumerate(lives):
        t = tokens(life)
        ids[i, :len(t)] = t
        for j in range(len(t)):
            if t[j] == RELIEF:
                target[i, j] = True
    return ids, target


def levels(logits):
    import torch
    return torch.softmax(logits[..., LEVEL0:LEVEL0 + len(RL.LEVELS)], -1)


def train(arm, seed, lives, log):
    import torch
    net = model(seed)
    seen = torch.tensor(mask(arm))
    ids, target = (torch.tensor(x) for x in batch(lives))
    labels = torch.roll(ids, -1, 1) - LEVEL0
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    rng = np.random.default_rng([W.SEED, STREAM, seed, 1])
    order, losses = [], []
    for step in range(STEPS):
        if len(order) < BATCH:
            order = list(rng.permutation(len(lives))) + order
        pick = torch.tensor([order.pop() for _ in range(BATCH)])
        logits = net(ids[pick], seen)
        t = target[pick]
        loss = torch.nn.functional.cross_entropy(logits[t][:, LEVEL0:LEVEL0 + len(RL.LEVELS)], labels[pick][t])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            losses.append(round(float(loss.detach()), 5))
    log(f"  {arm} seed {seed}: loss {losses[-1]}")
    return net, losses


def accuracy(net, arm, held):
    import torch
    seen = torch.tensor(mask(arm))
    ids, target = (torch.tensor(x) for x in batch(held))
    with torch.no_grad():
        p = levels(net(ids, seen))[target]
    truth = (torch.roll(ids, -1, 1) - LEVEL0)[target]
    return float((p.argmax(-1) == truth).float().mean()), p.argmax(-1).tolist()


def live(net, arm, index):
    """A life of the 256 worlds of test 22, the action chosen by the predicted relief (research.need_relief.readout)."""
    import torch
    seen = torch.tensor(mask(arm))
    worlds = W.world_rng(RL.LIFE_STREAM, index)
    e = n = W.MAX
    last, seq = 1, [BOS]
    for t in range(1, W.TURNS + 1):
        k = int(worlds.choice(len(W.EVENTS), p=W.PROBS))
        e, n = e + W.EVENTS[k][2], n + W.EVENTS[k][3]
        if e <= 0 or n <= 0:
            return {"survived": False, "length": t - 1}
        both = torch.tensor([seq + [EVENT0 + k, CHOICE, R + a, RELIEF] for a in (0, 1)])
        with torch.no_grad():
            p = levels(net(both, seen))[:, -1].numpy()
        a = RL.readout(p[0], p[1], last)
        lev = RL.level(e if a == 0 else n)
        seq += [EVENT0 + k, CHOICE, R + a, RELIEF, LEVEL0 + lev]
        last = a
        e, n = W.after(e, n, a)
    return {"survived": True, "length": W.TURNS}


def checks(nets, held):
    """Seed 0: under "actions", no prediction of a later turn moves when a past event changes; under "route", the first
    layer's output at later turns' tokens does not move."""
    import torch
    out = {}
    for arm in ("route", "actions"):
        net, seen = nets[arm], torch.tensor(mask(arm))
        gaps = []
        for life in [l for l in held if len(W.decisions(l)) >= 6][:4]:
            ids = torch.tensor([tokens(life)])
            changed = ids.clone()
            changed[0, 1] = EVENT0 + (life["turns"][0]["event"] + 1) % len(W.EVENTS)  # the event of turn 1
            later = slice(1 + PER_TURN, ids.shape[1])
            with torch.no_grad():
                if arm == "actions":
                    a, b = net(ids, seen), net(changed, seen)
                else:
                    first = lambda x: net.layers.layers[0](net.tok(x) + net.pos(torch.arange(x.shape[1]))[None],
                                                          src_mask=~seen[:x.shape[1], :x.shape[1]])
                    a, b = first(ids), first(changed)
            gaps.append(float((a[0, later] - b[0, later]).abs().max()))
        out[arm] = max(gaps)
    return {"mask_gap": out}


def run(a):
    import torch
    torch.set_num_threads(a.threads)
    log = lambda m: print(m, flush=True)
    root = Path(a.out)
    out = root / "runs"
    out.mkdir(parents=True, exist_ok=True)
    lives = RL.writer_lives(RL.TRAIN_STREAM, RL.TRAIN_LIVES)
    held = RL.writer_lives(RL.HELD_STREAM, RL.HELD_LIVES)
    for seed in a.seeds:
        paths = {arm: out / f"{arm}-{seed}.json" for arm in ARMS}
        if all(p.exists() for p in paths.values()) and (seed != 0 or (root / "setup.json").exists()):
            continue
        nets = {}
        for arm in ARMS:  # the three masks of a seed are trained together (seed 0's checks use its nets)
            net, losses = train(arm, seed, lives, log)
            nets[arm] = net
            acc, predicted = accuracy(net, arm, held)
            survived = [live(net, arm, i)["survived"] for i in range(RL.LIVES)]
            paths[arm].write_text(json.dumps({"arm": arm, "seed": seed, "accuracy": acc,
                                              "survival": float(np.mean(survived)), "survived": survived,
                                              "predicted": predicted, "losses": losses}) + "\n")
            log(f"{arm} seed {seed}: accuracy {acc:.3f}, survival {np.mean(survived):.3f}")
        if seed == 0:
            (root / "setup.json").write_text(json.dumps(checks(nets, held), indent=1) + "\n")


def gather(root):
    out = Path(root)
    runs = {arm: {} for arm in ARMS}
    for path in sorted((out / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        runs[r["arm"]][str(r["seed"])] = r
    ceiling = ceiling_accuracy(RL.writer_lives(RL.HELD_STREAM, RL.HELD_LIVES))
    setup = json.loads((out / "setup.json").read_text())
    return json.loads(json.dumps(dict(verdicts(runs, setup, ceiling), setup=setup)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
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
