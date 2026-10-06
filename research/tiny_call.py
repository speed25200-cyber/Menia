"""Saying a need that the action does not serve (docs/TINY_CALL_PILOT_PLAN.md): the "chaleur" world, the gated loop
with two outputs, and the pilot before test 36.

The world of the needs with a third need, the warmth H (1 to 8, 8 at the start), lowered by some events ("tu cours" -1,
"il fait froid" -2, "orage" -3) and never by the action: each turn, after the event, the agent chooses an action (R:
E + 3, M: N + 3) and a call; a call brings H + 6 from a partner and costs E - 1. Reward 1 per turn lived, nothing
else. The agent is the gated loop of test 34 with one more input (its last call) and one more output (the call), read
at the same "Choix" token; the cut control has its state replaced by the learned initial vector at every turn. The
grafts of the pilot change only H (same E and N, same action and call at j) and read the call at t = j + 1 or j + 2.
The choice of the pilot is numpy only; training and measures need torch.
"""
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
from . import need_memory as NM
from . import need_world as W
from . import tiny_loop as TL
from . import tiny_loop_pilot2 as Q
from . import tiny_survival as TS

DH = (0, -1, -2, 0, 0, -3)  # effect of each event of research/need_world.py on H
GIFT, COST, CALL_LOW = 6, 1, 3
NONE, R, M, EVENT0, CHOICE, NO, YES = 0, 1, 2, 3, 9, 10, 11
VOCAB, POSITIONS = 12, 5
INIT_STREAM, WORLD_STREAM, ACTION_STREAM = 94, 95, 96
PILOT_WORLDS, LIVES_STREAM, DRAW_STREAM = 85, 86, 87
LIVES, GAPS, PER_GAP, EPS = 128, (1, 2), W.TURNS, 0.3  # every turn t with a decision at t - g
LIVES_PER_UPDATE, LR, GAMMA, VALUE, ENTROPY = 64, 3e-4, 0.9, 0.5, 0.01
CONFIG = {"C1": {"updates": 24000}, "C2": {"updates": 48000}}
SETTINGS = ("C1", "C2")
SEEDS = (100, 101, 102)
NEEDED, CARRIED, TIE = 0.05, 0.05, 0.01
ROOT = "artifacts/tiny-call-pilot"
JOBS = [("C1", s) for s in SEEDS] + [("cut", s) for s in SEEDS] + [("C2", s) for s in SEEDS]


# ----------------------------------------------------------------------------------------------------- the world

def event(e, n, h, k):
    return e + W.EVENTS[k][2], n + W.EVENTS[k][3], h + DH[k]


def decide(e, n, h, action, call):
    """After the event: the action, then the call. Returns the needs; E may reach 0 by the cost of the call."""
    e, n = W.after(e, n, action)
    if call:
        h, e = min(W.MAX, h + GIFT), e - COST
    return e, n, h


def call_rule(h):
    return int(h <= CALL_LOW)


def writer_life(index, stream=LIVES_STREAM):
    """A life written by a rule: the action of the rule "needs" (on E and N) and the call of "call if H <= 3", each
    replaced at random with probability EPS (stream [SEED, stream, 1, index]). Each decision keeps E, N, H after the event, the action,
    the call and the last action and call."""
    worlds = W.world_rng(stream, index)
    noise = np.random.default_rng([W.SEED, stream, 1, index])
    e = n = h = W.MAX
    last, last_call, turns = 1, 0, []
    for t in range(1, W.TURNS + 1):
        k = int(worlds.choice(len(W.EVENTS), p=W.PROBS))
        e, n, h = event(e, n, h, k)
        if min(e, n, h) <= 0:
            turns.append({"t": t, "event": k, "E": e, "N": n, "H": h, "dead": True})
            return {"turns": turns, "survived": False}
        a = NM.rule(e, n, last) if noise.random() >= EPS else int(noise.random() < 0.5)
        c = call_rule(h) if noise.random() >= EPS else int(noise.random() < 0.5)
        turns.append({"t": t, "event": k, "E": e, "N": n, "H": h, "action": a, "call": c, "last": last,
                      "last_call": last_call})
        e, n, h = decide(e, n, h, a, c)
        last, last_call = a, c
        if e <= 0:
            return {"turns": turns, "survived": False}
    return {"turns": turns, "survived": True}


def lives(stream=LIVES_STREAM):
    return [writer_life(i, stream) for i in range(LIVES)]


def replay(life, j, needs, t):
    """The needs at t in this life if they had been `needs` (E, N, H after the event) at j: the life's actions and
    calls from j to t - 1 and its events from j + 1 to t. None if the life would have gone out before t."""
    at = {x["t"]: x for x in W.decisions(life)}
    e, n, h = needs
    for u in range(j, t):
        e, n, h = decide(e, n, h, at[u]["action"], at[u]["call"])
        if e <= 0:
            return None
        e, n, h = event(e, n, h, at[u + 1]["event"])
        if min(e, n, h) <= 0:
            return None
    return e, n, h


def draw(held, stream=DRAW_STREAM):
    """The grafts of the pilot: recipients in order; per gap up to PER_GAP turns t; per t one donor of each kind, alive
    at j = t - g with the same action and call at j and the same E and N but another H: (h) the rule "call if H <= 3"
    would change its call at t with the donor's needs at j (e_call = +1 if it would call where it did not, -1 for the
    reverse), (h0) it would not."""
    rng = np.random.default_rng([W.SEED, stream, 0])
    at = [{x["t"]: x for x in W.decisions(life)} for life in held]
    out = []
    for r, life in enumerate(held):
        for g in GAPS:
            turns = [t for t in sorted(at[r]) if t - g >= 1 and t - g in at[r]]
            for k in sorted(rng.permutation(len(turns))[:PER_GAP]):
                t = turns[k]
                j = t - g
                mine = at[r][j]
                for kind in ("h", "h0"):
                    donors = []
                    for d in range(len(held)):
                        x = at[d].get(j)
                        if d == r or x is None or (x["action"], x["call"]) != (mine["action"], mine["call"]):
                            continue
                        if (x["E"], x["N"]) != (mine["E"], mine["N"]) or x["H"] == mine["H"]:
                            continue
                        needs = replay(life, j, (x["E"], x["N"], x["H"]), t)
                        if needs is None:
                            continue
                        e = call_rule(needs[2]) - call_rule(at[r][t]["H"])
                        if (kind == "h") != (e != 0):
                            continue
                        donors.append((d, e))
                    if donors:
                        d, e = donors[int(rng.integers(len(donors)))]
                        out.append({"index": len(out), "kind": kind, "r": r, "d": d, "j": j, "t": t, "g": g,
                                    "e": int(e)})
    return out


def call_effect(rows, grafts):
    """m_call: the mean over the (h) grafts of (P(call) with the graft - without) times e_call; and the mean |change of
    P(R)| over the same grafts (an H-only graft should hardly move the action)."""
    by = {g["index"]: g for g in grafts}
    if sorted(r[0] for r in rows) != sorted(by):
        raise ValueError("the rows do not match the draw")
    h = [r for r in rows if by[r[0]]["kind"] == "h"]
    return {"m_call": float(np.mean([(c1 - c0) * by[i]["e"] for i, _, _, c0, c1 in h])),
            "abs_action": float(np.mean([abs(r1 - r0) for _, r0, r1, _, _ in h])),
            "abs_call_h0": float(np.mean([abs(c1 - c0) for i, _, _, c0, c1 in rows if by[i]["kind"] == "h0"]))}


def choose(loops, cuts):
    """loops: {setting: {seed: {"survival", "m_call"}}}; cuts: {seed: survival}. A setting is reliable if, on each of
    the three seeds, its survival exceeds the cut of the same seed by at least NEEDED and m_call > CARRIED. The reliable
    setting with the largest mean m_call; within TIE, C1; None if none is reliable."""
    table = {}
    for s in SETTINGS:
        if s not in loops:
            continue
        seeds = sorted(loops[s], key=int)
        m = {k: loops[s][k]["m_call"] for k in seeds}
        gain = {k: loops[s][k]["survival"] - cuts[k] for k in seeds}
        table[s] = {"m_mean": float(np.mean(list(m.values()))), "gain_mean": float(np.mean(list(gain.values()))),
                    "m": m, "gain": gain,
                    "reliable": bool(sorted(seeds) == sorted(str(k) for k in SEEDS)
                                     and all(v > CARRIED for v in m.values()) and all(v >= NEEDED for v in gain.values()))}
    ok = [s for s in table if table[s]["reliable"]]
    if not ok:
        return None, table
    best = max(table[s]["m_mean"] for s in ok)
    return next(s for s in SETTINGS if s in ok and table[s]["m_mean"] >= best - TIE), table


def summary(root):
    grafts = draw(lives())
    loops, cuts = {}, {}
    for path in sorted((Path(root) / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        seed = str(r["seed"])
        if r["name"] == "cut":
            cuts[seed] = r["survival"]
        else:
            loops.setdefault(r["name"], {})[seed] = dict(survival=r["survival"], **call_effect(r["grafts"], grafts))
    chosen, table = choose({s: {k: {"survival": v["survival"], "m_call": v["m_call"]} for k, v in x.items()}
                            for s, x in loops.items()}, cuts)
    kinds = [g["kind"] for g in grafts]
    return {"chosen": chosen, "table": table, "loops": loops, "cuts": cuts,
            "grafts": {k: kinds.count(k) for k in ("h", "h0")}}


# ----------------------------------------------------------------------------------------------------- torch part

def model(seed):
    """The gated loop of test 34 with one more input (the last call) and one more output (the call)."""
    import torch
    from torch import nn
    torch.manual_seed(int(np.random.default_rng([W.SEED, INIT_STREAM, seed]).integers(2 ** 31)))

    class Caller(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(VOCAB, TL.DIM)
            self.pos = nn.Embedding(POSITIONS, TL.DIM)
            layer = nn.TransformerEncoderLayer(TL.DIM, TL.HEADS, 4 * TL.DIM, dropout=0.0, batch_first=True,
                                               norm_first=True)
            self.layers = nn.TransformerEncoder(layer, TL.LAYERS, enable_nested_tensor=False)
            self.out = nn.Linear(TL.DIM, 2)
            self.call = nn.Linear(TL.DIM, 2)
            self.value_head = nn.Linear(TL.DIM, 1)
            self.start = nn.Parameter(torch.randn(TL.DIM) * 0.02)
            self.gate = nn.Linear(TL.DIM, TL.DIM)
            nn.init.zeros_(self.gate.bias)

        def initial(self, n):
            return self.start[None].expand(n, TL.DIM)

        def step(self, state, last, call, event):
            """One turn for a batch: the state (n, DIM) and the tokens of the last action, the last call and the event
            (n,). Returns the logits of R and M, the logits of no call and call, the value and the next state."""
            n = state.shape[0]
            choice = torch.full((n,), CHOICE, dtype=torch.long)
            x = torch.stack([state, self.tok(last), self.tok(call), self.tok(event), self.tok(choice)], 1)
            h = self.layers(x + self.pos.weight[None])[:, -1]
            z = torch.sigmoid(self.gate(h))
            new = (1 - z) * state + z * nn.functional.layer_norm(h, (TL.DIM,))
            return self.out(h), self.call(h), self.value_head(h)[:, 0], new
    return Caller()


def episodes(net, arm, worlds, pick, grad=False):
    """Lives in the given worlds, all at once; pick(p_R, p_call, last) -> (actions, calls) (numpy). Returns the
    decisions (life, turn, action, call, E, N, H, last), the log probabilities of the chosen pair, the entropies (sum of
    both outputs), the values and the lengths (turns lived; a life that dies by the cost of its call at t has lived
    t - 1)."""
    import torch
    size = len(worlds)
    e, n, h = np.full(size, W.MAX), np.full(size, W.MAX), np.full(size, W.MAX)
    alive, last = np.ones(size, bool), np.ones(size, int)
    length = np.zeros(size, int)
    token_last, token_call, events = np.full(size, NONE), np.full(size, NONE), np.zeros(size, int)
    state = net.initial(size)
    decisions, logps, entropies, values = [], [], [], []
    with torch.set_grad_enabled(grad):
        for t in range(1, W.TURNS + 1):
            for i in np.flatnonzero(alive):
                k = int(worlds[i].choice(len(W.EVENTS), p=W.PROBS))
                e[i], n[i], h[i] = event(e[i], n[i], h[i], k)
                events[i] = EVENT0 + k
                if min(e[i], n[i], h[i]) <= 0:
                    alive[i] = False
            live = np.flatnonzero(alive)
            if len(live) == 0:
                break
            la, lc, value, nxt = net.step(state if arm == "loop" else net.initial(size), torch.tensor(token_last),
                                          torch.tensor(token_call), torch.tensor(events))
            state = nxt
            pa, pc = torch.log_softmax(la[live], -1), torch.log_softmax(lc[live], -1)
            actions, calls = pick(pa[:, 0].exp().detach().numpy(), pc[:, 1].exp().detach().numpy(), last[live])
            ta, tc = torch.tensor(actions), torch.tensor(calls)
            logps.append(pa.gather(1, ta[:, None])[:, 0] + pc.gather(1, tc[:, None])[:, 0])
            entropies.append(-(pa.exp() * pa).sum(-1) - (pc.exp() * pc).sum(-1))
            values.append(value[live])
            for i, a, c in zip(live, actions, calls):
                decisions.append((int(i), t, int(a), int(c), int(e[i]), int(n[i]), int(h[i]), int(last[i])))
                e[i], n[i], h[i] = decide(e[i], n[i], h[i], int(a), int(c))
                last[i], token_last[i], token_call[i] = int(a), R + int(a), YES if c else NO
                if e[i] <= 0:
                    alive[i] = False
                    length[i] = t - 1
                else:
                    length[i] = t
    return decisions, logps, entropies, values, length


def train(arm, seed, config, log, checkpoint=None):
    """The actor-critic of test 34 on the pair (action, call); kept every 250 updates and taken back exactly after a
    stop (each update draws its worlds and choices from its own streams)."""
    import torch
    net = model(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    curve, window, start = [], [], 0
    if checkpoint is not None and Path(checkpoint).exists():
        state = torch.load(checkpoint, weights_only=False)
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        curve, window, start = state["curve"], state["window"], state["update"]
        log(f"  {checkpoint}: resumed at update {start}")
    for update in range(start, config["updates"]):
        worlds = [np.random.default_rng([W.SEED, WORLD_STREAM, seed, update, i]) for i in range(LIVES_PER_UPDATE)]
        rng = np.random.default_rng([W.SEED, ACTION_STREAM, seed, update])

        def pick(p_r, p_call, last):
            return (rng.random(len(p_r)) >= p_r).astype(int), (rng.random(len(p_call)) < p_call).astype(int)
        decisions, logps, entropies, values, length = episodes(net, arm, worlds, pick, grad=True)
        life = np.array([d[0] for d in decisions])
        turn = np.array([d[1] for d in decisions])
        lived = (length[life] - turn).astype(float)
        ret = torch.tensor((1 - GAMMA ** lived) / (1 - GAMMA), dtype=torch.float32)
        chosen, entropy, v = torch.cat(logps), torch.cat(entropies), torch.cat(values)
        with torch.no_grad():
            advantage = ret - v
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        loss = -(advantage * chosen).mean() + VALUE * ((v - ret) ** 2).mean() - ENTROPY * entropy.mean()
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
    return net, curve


def greedy(p_r, p_call, last):
    return np.where(p_r > 0.5, 0, np.where(p_r < 0.5, 1, 1 - last)), (p_call > 0.5).astype(int)


def survival(net, arm, stream=PILOT_WORLDS, count=256):
    """Survival on the given worlds with the most probable action and call; and the share of calls that follow the
    rule "call if H <= 3"."""
    worlds = [W.world_rng(stream, i) for i in range(count)]
    decisions, _, _, _, length = episodes(net, arm, worlds, greedy)
    agrees = float(np.mean([c == call_rule(h) for _, _, _, c, _, _, h, _ in decisions]))
    return (length == W.TURNS).tolist(), agrees


class Reader:
    """The agent reading a written life: P(R) and P(call) at each decision, and the state passed by each decision,
    with or without a grafted state."""

    def __init__(self, net, arm):
        self.net, self.arm = net, arm

    def read(self, life, upto=None, graft=None):
        import torch
        state, last, last_call = self.net.initial(1).detach(), NONE, NONE
        p_r, p_call, states = {}, {}, {}
        with torch.no_grad():
            for x in W.decisions(life):
                if upto is not None and x["t"] > upto:
                    break
                la, lc, _, nxt = self.net.step(state if self.arm == "loop" else self.net.initial(1),
                                               torch.tensor([last]), torch.tensor([last_call]),
                                               torch.tensor([EVENT0 + x["event"]]))
                p_r[x["t"]] = float(torch.softmax(la, -1)[0, 0])
                p_call[x["t"]] = float(torch.softmax(lc, -1)[0, 1])
                state = graft[1] if graft is not None and x["t"] == graft[0] else nxt
                states[x["t"]] = state
                last, last_call = R + x["action"], YES if x["call"] else NO
        return p_r, p_call, states


def measure_grafts(reader, held, grafts):
    """Per graft: [index, P(R) without, P(R) with, P(call) without, P(call) with] at t."""
    own = [reader.read(life) for life in held]
    rows = []
    for g in grafts:
        p_r, p_call, _ = reader.read(held[g["r"]], upto=g["t"], graft=(g["j"], own[g["d"]][2][g["j"]]))
        rows.append([g["index"], round(own[g["r"]][0][g["t"]], 7), round(p_r[g["t"]], 7),
                     round(own[g["r"]][1][g["t"]], 7), round(p_call[g["t"]], 7)])
    return rows


def ready(root, name, seed):
    """C2 starts from the checkpoint of C1 at 24 000 updates, once that job is done (copied once)."""
    import torch
    target = Path(root) / "checkpoints" / f"{name}-{seed}.pt"
    if target.exists() or name != "C2":
        return True
    if not (Path(root) / "runs" / f"C1-{seed}.json").exists():
        return False
    source = Path(root) / "checkpoints" / f"C1-{seed}.pt"
    if torch.load(source, weights_only=False)["update"] != CONFIG["C1"]["updates"]:
        raise ValueError(f"{source} is not at {CONFIG['C1']['updates']} updates")
    shutil.copyfile(source, target)
    return True


def job(root, name, seed, held, grafts, log):
    arm = "cut" if name == "cut" else "loop"
    config = CONFIG["C1" if name == "cut" else name]
    net, curve = train(arm, seed, config, log, checkpoint=Path(root) / "checkpoints" / f"{name}-{seed}.pt")
    net.eval()
    survived, agrees = survival(net, arm)
    row = {"name": name, "arm": arm, "seed": seed, "config": config, "survival": float(np.mean(survived)),
           "survived": survived, "agrees_call_rule": agrees, "curve": curve}
    if arm == "loop":
        row["grafts"] = measure_grafts(Reader(net, arm), held, grafts)
        x = call_effect(row["grafts"], grafts)
        log(f"{name} seed {seed}: survival {row['survival']:.3f}, m_call {x['m_call']:+.3f}, |dP(R)| {x['abs_action']:.3f}")
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
            if ready(root, name, seed) and Q.claim(root, name, seed):
                if (root / "runs" / f"{name}-{seed}.json").exists():
                    (root / "claims" / f"{name}-{seed}").unlink()
                    break
                job(root, name, seed, held, grafts, log)
                (root / "claims" / f"{name}-{seed}").unlink()
                break
        else:
            if all(Q.taken(root, n, s) for n, s in left):
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
    result = json.loads(json.dumps(summary(a.out)))
    path = Path(a.out) / "choice.json"
    if a.command == "choose":
        path.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({"chosen": result["chosen"]}))
    else:
        differs = result != json.loads(path.read_text())
        print("differs:", "yes" if differs else "none")
        if differs:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
