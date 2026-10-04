"""Only surviving: memory through the state without any target (docs/TINY_SURVIVAL_PROTOCOL.md): SURV1, SURV2 of test 30.

The world of the programme written as 3 simple tokens per turn (event, "Choix", action) at fixed positions; a 2-layer
transformer learned from scratch (as in research/tiny_relief.py) chooses at "Choix" and learns only from its survival
(1 per turn lived after a decision; REINFORCE with the mean return of the batch at the same turn as baseline, entropy
bonus 0.01), under the masks of test 22: "route", "actions", "free" (published without threshold). Ten seeds per mask.
Measure: the survival on the 256 worlds of test 22, the most probable action taken. The verdicts are numpy only;
training and measures need torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_memory as NM
from . import need_rules as NR

BOS, EVENT0, CHOICE, R, M, END = 0, 1, 7, 8, 9, 10
VOCAB, PER_TURN = 11, 3
LENGTH = 1 + PER_TURN * W.TURNS
ARMS = ("route", "actions", "free")
SEEDS = tuple(range(10))
INIT_STREAM, WORLD_STREAM, ACTION_STREAM = 70, 71, 72
UPDATES, LIVES_PER_UPDATE, LR, ENTROPY = 2000, 64, 1e-3, 0.01
DIM, HEADS, LAYERS = 64, 4, 2
SURV1, SURV2 = 0.08, 0.05
MIN_B, MASK_TOLERANCE = 0.60, 1e-6
T_975_DF9 = 2.2621571627409915
ROOT = "artifacts/tiny-survival"


def mask(arm, length=LENGTH):
    """mask[q, k] = True if position q may see position k (the masks of research/tiny_relief.py, 3 tokens a turn)."""
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
        return np.where(action[:, None], bos | (k == q) | (past & action[None, :]), out)
    raise ValueError(arm)


def event_rule_survival():
    """The event rule on the 256 worlds of test 22 (research.need_rules)."""
    out = []
    for i in range(NM.LIVES):
        worlds = W.world_rng(NM.LIFE_STREAM, i)
        e = n = W.MAX
        last, alive = 1, True
        for t in range(1, W.TURNS + 1):
            k = int(worlds.choice(len(W.EVENTS), p=W.PROBS))
            e, n = e + W.EVENTS[k][2], n + W.EVENTS[k][3]
            if e <= 0 or n <= 0:
                alive = False
                break
            last = NR.event(e, n, k, last)
            e, n = W.after(e, n, last)
        out.append(alive)
    return float(np.mean(out))


def interval(x):
    x = np.asarray(x, float)
    half = T_975_DF9 * x.std(ddof=1) / np.sqrt(len(x))
    return {"mean": float(x.mean()), "low": float(x.mean() - half), "high": float(x.mean() + half)}


def verdicts(runs, setup, event_rule):
    seeds = sorted(runs["route"], key=int)
    gain = interval([runs["route"][s]["survival"] - runs["actions"][s]["survival"] for s in seeds])
    beyond = interval([runs["route"][s]["survival"] - event_rule for s in seeds])
    survival = {a: float(np.mean([runs[a][s]["survival"] for s in seeds])) for a in runs}
    out = {"SURV1": bool(gain["mean"] >= SURV1 and gain["low"] > 0),
           "SURV2": bool(beyond["mean"] >= SURV2 and beyond["low"] > 0)}
    out["valid"] = bool(setup["mask_gap"]["route"] <= MASK_TOLERANCE and setup["mask_gap"]["actions"] <= MASK_TOLERANCE
                        and survival["actions"] >= MIN_B and len(seeds) == len(SEEDS))
    out["global"] = bool(out["valid"] and out["SURV1"] and out["SURV2"])
    values = {"event_rule": event_rule, "survival": survival, "survival_gain": gain, "beyond_event_rule": beyond,
              "agrees_with_needs_rule": {a: float(np.mean([runs[a][s]["agrees"] for s in seeds])) for a in runs},
              "by_seed": {s: {a: runs[a][s]["survival"] for a in runs} for s in seeds}}
    return {"verdicts": out, "values": values}


# ----------------------------------------------------------------------------------------------------- torch part

def model(seed):
    import torch
    from torch import nn
    torch.manual_seed(int(np.random.default_rng([W.SEED, INIT_STREAM, seed]).integers(2 ** 31)))

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


def episodes(net, arm, worlds, pick):
    """Lives in the given worlds (one generator each), all at once; pick(p_R, last) -> actions (numpy). Returns the
    token array, the decisions (life, turn, position, action, needs, last) and the lengths."""
    import torch
    seen = torch.tensor(mask(arm))
    n_lives = len(worlds)
    ids = torch.full((n_lives, LENGTH), END, dtype=torch.long)
    ids[:, 0] = BOS
    e, n = np.full(n_lives, W.MAX), np.full(n_lives, W.MAX)
    alive, last = np.ones(n_lives, bool), np.ones(n_lives, int)
    length = np.zeros(n_lives, int)
    decisions = []
    for t in range(1, W.TURNS + 1):
        base = 1 + PER_TURN * (t - 1)
        for i in np.flatnonzero(alive):
            k = int(worlds[i].choice(len(W.EVENTS), p=W.PROBS))
            e[i], n[i] = e[i] + W.EVENTS[k][2], n[i] + W.EVENTS[k][3]
            ids[i, base] = EVENT0 + k
            if e[i] <= 0 or n[i] <= 0:
                ids[i, base + 1] = END
                alive[i] = False
            else:
                ids[i, base + 1] = CHOICE
        live = np.flatnonzero(alive)
        if len(live) == 0:
            break
        with torch.no_grad():
            logits = net(ids[live, :base + 2], seen)[:, -1, [R, M]]
        p_r = torch.softmax(logits, -1)[:, 0].numpy()
        actions = pick(p_r, last[live])
        for i, a, p in zip(live, actions, p_r):
            decisions.append((int(i), t, base + 1, int(a), int(e[i]), int(n[i]), int(last[i])))
            ids[i, base + 2] = R + int(a)
            e[i], n[i] = W.after(e[i], n[i], int(a))
            last[i] = int(a)
            length[i] = t
    return ids, decisions, length


def train(arm, seed, log):
    import torch
    net = model(seed)
    seen = torch.tensor(mask(arm))
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    curve, window = [], []
    for update in range(UPDATES):
        worlds = [np.random.default_rng([W.SEED, WORLD_STREAM, seed, update, i]) for i in range(LIVES_PER_UPDATE)]
        rng = np.random.default_rng([W.SEED, ACTION_STREAM, seed, update])
        ids, decisions, length = episodes(net, arm, worlds, lambda p, last: (rng.random(len(p)) >= p).astype(int))
        life = np.array([d[0] for d in decisions])
        turn = np.array([d[1] for d in decisions])
        position = torch.tensor([d[2] for d in decisions])
        action = torch.tensor([d[3] for d in decisions])
        ret = (length[life] - turn).astype(float)
        baseline = np.array([ret[turn == t].mean() for t in turn])
        advantage = torch.tensor(ret - baseline, dtype=torch.float32)
        logits = net(ids, seen)[torch.tensor(life), position][:, [R, M]]
        logp = torch.log_softmax(logits, -1)
        chosen = logp.gather(1, action[:, None])[:, 0]
        entropy = -(logp.exp() * logp).sum(-1)
        loss = -(advantage * chosen).mean() - ENTROPY * entropy.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        window.append(float((length == W.TURNS).mean()))
        if (update + 1) % 100 == 0:
            curve.append(round(float(np.mean(window)), 4))
            window = []
    log(f"  {arm} seed {seed}: training survival {curve[-1] if curve else None}")
    return net, curve


def measure(net, arm):
    worlds = [W.world_rng(NM.LIFE_STREAM, i) for i in range(NM.LIVES)]
    greedy = lambda p, last: np.where(p > 0.5, 0, np.where(p < 0.5, 1, 1 - last))
    _, decisions, length = episodes(net, arm, worlds, greedy)
    survived = (length == W.TURNS).tolist()
    agrees = float(np.mean([a == NM.rule(e, n, last) for _, _, _, a, e, n, last in decisions]))
    return survived, agrees


def tokens(life):
    out = [BOS]
    for x in life["turns"]:
        out += [EVENT0 + x["event"], END] if x.get("dead") else [EVENT0 + x["event"], CHOICE, R + x["action"]]
    return out


def checks(nets):
    """Seed 0, on lives of the teacher: under "actions", no output of a later turn moves when the event of turn 1
    changes; under "route", the first layer's output at later turns' tokens does not move."""
    import torch
    held = [l for l in NM.teacher_lives(NM.HELD_STREAM, 16) if len(W.decisions(l)) >= 6][:4]
    out = {}
    for arm in ("route", "actions"):
        net, seen = nets[arm], torch.tensor(mask(arm))
        gaps = []
        for life in held:
            ids = torch.tensor([tokens(life)])
            changed = ids.clone()
            changed[0, 1] = EVENT0 + (life["turns"][0]["event"] + 1) % len(W.EVENTS)
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
    for seed in a.seeds:
        paths = {arm: out / f"{arm}-{seed}.json" for arm in ARMS}
        if all(p.exists() for p in paths.values()) and (seed != 0 or (root / "setup.json").exists()):
            continue
        nets = {}
        for arm in ARMS:
            net, curve = train(arm, seed, log)
            nets[arm] = net
            survived, agrees = measure(net, arm)
            paths[arm].write_text(json.dumps({"arm": arm, "seed": seed, "survival": float(np.mean(survived)),
                                              "survived": survived, "agrees": agrees, "curve": curve}) + "\n")
            log(f"{arm} seed {seed}: survival {np.mean(survived):.3f}, agrees with the needs rule {agrees:.3f}")
        if seed == 0:
            (root / "setup.json").write_text(json.dumps(checks(nets), indent=1) + "\n")


def gather(root):
    out = Path(root)
    runs = {arm: {} for arm in ARMS}
    for path in sorted((out / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        runs[r["arm"]][str(r["seed"])] = r
    setup = json.loads((out / "setup.json").read_text())
    return json.loads(json.dumps(dict(verdicts(runs, setup, event_rule_survival()), setup=setup)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=2)
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
