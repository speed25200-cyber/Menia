"""The wager: does the loop know when it does not know? (docs/TINY_WAGER_PILOT_PLAN.md): the "pari" world, the gated
loop with a second output (the wager), and the pilot before test 37.

The world of the needs where "tu cours" and "il fait froid" are each hidden with probability HIDE: the agent then sees
the same sign "?". Each turn, after the event, the agent chooses an action (R: E + 3, M: N + 3) and a wager; a wager
wins WIN if the action serves a lowest need (R if E <= N, M if N <= E) and loses LOSS otherwise. Reward 1 per turn
lived plus the wagers; the wager does not change the world and the agent never sees its rewards. The rule is the
exact belief over (E, N) given what the agent saw, did and that it is alive: its action is the one most likely to serve
a lowest need, and it wagers when that probability exceeds LOSS / (WIN + LOSS). The grafts change the belief carried
from j: (u) the rule's action at t is the same and its wager changes, (k) its action changes and its wager does not,
(b) the same belief from another life. The choice of the pilot is numpy only; training and measures need torch.
"""
import argparse
import json
import shutil
import time
from pathlib import Path
import numpy as np
from . import need_world as W
from . import tiny_loop as TL
from . import tiny_loop_pilot2 as Q

HIDE, WIN, LOSS = 0.5, 0.5, 2.0
SURE = LOSS / (WIN + LOSS)  # the rule wagers above this probability of serving a lowest need
STRONG = (1, 2)  # "tu cours", "il fait froid"
NONE, R, M, EVENT0, HIDDEN, CHOICE = 0, 1, 2, 3, 9, 10
VOCAB, POSITIONS = 11, 4
INIT_STREAM, WORLD_STREAM, ACTION_STREAM = 130, 131, 132
PILOT_WORLDS, LIVES_STREAM, DRAW_STREAM = 133, 134, 135
LIVES, GAPS, EPS = 128, (1, 2), 0.3
KINDS = ("u", "k", "b")
LIVES_PER_UPDATE, LR, GAMMA, VALUE, ENTROPY = 64, 3e-4, 0.9, 0.5, 0.01
CONFIG = {"P1": {"updates": 48000}, "P2": {"updates": 96000}}
SETTINGS = ("P1", "P2")
SEEDS = (110, 111, 112)
GAIN, CARRIED, TIE = 1.0, 0.05, 0.01
ROOT = "artifacts/tiny-wager-pilot"
JOBS = [("P1", s) for s in SEEDS] + [("cut", s) for s in SEEDS] + [("P2", s) for s in SEEDS]


# ----------------------------------------------------------------------------------------------------- the world

def draw_event(rng):
    """The event of a turn and whether it is hidden (only "tu cours" and "il fait froid" can be)."""
    k = int(rng.choice(len(W.EVENTS), p=W.PROBS))
    return k, bool(k in STRONG and rng.random() < HIDE)


def token(k, hidden):
    return HIDDEN if hidden else EVENT0 + k


def served(e, n, action):
    return bool(e <= n) if action == 0 else bool(n <= e)


def wager_reward(e, n, action, wager):
    return 0.0 if not wager else (WIN if served(e, n, action) else -LOSS)


# The belief: a (MAX + 1) x (MAX + 1) array over (E, N), index 0 standing for "out"; flat vectors of length SIZE.
SIZE = (W.MAX + 1) ** 2
_E, _N = np.divmod(np.arange(SIZE), W.MAX + 1)
LIVE = (_E > 0) & (_N > 0)


def _move(f):
    """The matrix sending each live (E, N) to f(E, N); out stays out."""
    m = np.zeros((SIZE, SIZE))
    for i in range(SIZE):
        e, n = f(_E[i], _N[i]) if LIVE[i] else (_E[i], _N[i])
        m[i, max(0, e) * (W.MAX + 1) + max(0, n)] = 1
    return m


EVENT_MOVE = [_move(lambda e, n, k=k: (e + W.EVENTS[k][2], n + W.EVENTS[k][3])) for k in range(len(W.EVENTS))]
HIDDEN_MOVE = sum(EVENT_MOVE[k] for k in STRONG) / len(STRONG)
ACTION_MOVE = [_move(lambda e, n, a=a: W.after(e, n, a)) for a in (0, 1)]
SERVES = [(_E <= _N) & LIVE, (_N <= _E) & LIVE]


def start():
    b = np.zeros(SIZE)
    b[W.MAX * (W.MAX + 1) + W.MAX] = 1
    return b


def observe(b, k, hidden):
    """The belief after the event of the turn (seen, or "?"), knowing the agent is still in the world; None if it
    cannot be."""
    b = b @ (HIDDEN_MOVE if hidden else EVENT_MOVE[k]) * LIVE
    s = b.sum()
    return b / s if s > 1e-12 else None


def act(b, action):
    return b @ ACTION_MOVE[action]


def rule(b, last):
    """The rule's action (the one most likely to serve a lowest need; equal: the other than the last), the probability
    that it serves one and its wager."""
    pr, pm = float(b[SERVES[0]].sum()), float(b[SERVES[1]].sum())
    a = 0 if pr > pm + 1e-12 else 1 if pm > pr + 1e-12 else 1 - last
    p = pr if a == 0 else pm
    return a, p, int(p > SURE)


def writer_life(index, stream=LIVES_STREAM):
    """A life written by the rule's action, replaced at random with probability EPS (stream [SEED, stream, 1, index]).
    Each decision keeps the event, whether it was hidden, E and N after the event, the action, the last action and
    the rule's belief (not saved)."""
    worlds = W.world_rng(stream, index)
    noise = np.random.default_rng([W.SEED, stream, 1, index])
    e = n = W.MAX
    b, last, turns = start(), 1, []
    for t in range(1, W.TURNS + 1):
        k, hidden = draw_event(worlds)
        e, n = e + W.EVENTS[k][2], n + W.EVENTS[k][3]
        if min(e, n) <= 0:
            turns.append({"t": t, "event": k, "hidden": hidden, "E": e, "N": n, "dead": True})
            return {"turns": turns, "survived": False}
        b = observe(b, k, hidden)
        a = rule(b, last)[0] if noise.random() >= EPS else int(noise.random() < 0.5)
        turns.append({"t": t, "event": k, "hidden": hidden, "E": e, "N": n, "action": a, "last": last, "belief": b})
        e, n = W.after(e, n, a)
        b, last = act(b, a), a
    return {"turns": turns, "survived": True}


def lives(stream=LIVES_STREAM):
    return [writer_life(i, stream) for i in range(LIVES)]


def replay(life, j, b, t):
    """The rule's belief at t in this life if it had been b at j (after the event of j): the life's actions from j to
    t - 1 and what it saw from j + 1 to t. None if that belief cannot be."""
    at = {x["t"]: x for x in W.decisions(life)}
    for u in range(j, t):
        b = observe(act(b, at[u]["action"]), at[u + 1]["event"], at[u + 1]["hidden"])
        if b is None:
            return None
    return b


def seen(life, j):
    """What the agent saw and did up to its decision at j: the event tokens and the actions."""
    return [(token(x["event"], x["hidden"]), x["action"]) for x in W.decisions(life) if x["t"] <= j]


def draw(held, stream=DRAW_STREAM):
    """The grafts: recipients in order; per gap and per turn t with a decision at j = t - g, one donor of each kind
    (drawn at random among the candidates), alive at j with the same action at j: (u) the rule's action at t is the
    same with the donor's belief at j and its wager changes (e = +1 if it would wager with the graft and not
    without, -1 for the reverse); (k) its action changes (e = +1 towards R, -1 towards M) and its wager does not;
    (b) the donor's belief at j is the recipient's (e = 0; same_history if they saw and did the same up to j)."""
    rng = np.random.default_rng([W.SEED, stream, 0])
    at = [{x["t"]: x for x in W.decisions(life)} for life in held]
    out = []
    for r, life in enumerate(held):
        for g in GAPS:
            for t in sorted(at[r]):
                j = t - g
                if j not in at[r]:
                    continue
                mine = at[r][j]
                a0, _, w0 = rule(at[r][t]["belief"], at[r][t]["last"])
                found = {kind: [] for kind in KINDS}
                for d in range(len(held)):
                    x = at[d].get(j)
                    if d == r or x is None or x["action"] != mine["action"]:
                        continue
                    if np.abs(x["belief"] - mine["belief"]).max() <= 1e-12:
                        found["b"].append((d, 0))
                        continue
                    b = replay(life, j, x["belief"], t)
                    if b is None:
                        continue
                    a1, _, w1 = rule(b, at[r][t]["last"])
                    if a1 == a0 and w1 != w0:
                        found["u"].append((d, w1 - w0))
                    elif a1 != a0 and w1 == w0:
                        found["k"].append((d, int(a1 == 0) - int(a0 == 0)))
                for kind in KINDS:
                    if found[kind]:
                        d, e = found[kind][int(rng.integers(len(found[kind])))]
                        x = {"index": len(out), "kind": kind, "r": r, "d": d, "j": j, "t": t, "g": g, "e": int(e)}
                        if kind == "b":
                            x["same_history"] = seen(held[d], j) == seen(life, j)
                        out.append(x)
    return out


def effects(rows, grafts):
    """rows: per graft [index, P(R) without, with, P(wager) without, with] at t. m_wager: the mean over the (u) grafts
    of the change of P(wager) times e; m_action: over the (k) grafts, the change of P(R) times e; and the mean |change|
    of each output over each kind."""
    by = {g["index"]: g for g in grafts}
    if sorted(r[0] for r in rows) != sorted(by):
        raise ValueError("the rows do not match the draw")
    out = {}
    for kind in KINDS:
        rs = [r for r in rows if by[r[0]]["kind"] == kind]
        out[f"abs_action_{kind}"] = float(np.mean([abs(r[2] - r[1]) for r in rs])) if rs else None
        out[f"abs_wager_{kind}"] = float(np.mean([abs(r[4] - r[3]) for r in rs])) if rs else None
    u = [r for r in rows if by[r[0]]["kind"] == "u"]
    k = [r for r in rows if by[r[0]]["kind"] == "k"]
    out["m_wager"] = float(np.mean([(r[4] - r[3]) * by[r[0]]["e"] for r in u]))
    out["m_action"] = float(np.mean([(r[2] - r[1]) * by[r[0]]["e"] for r in k]))
    return out


def auc(scores, labels):
    """The area under the ROC curve of scores for labels 1 against 0 (ties count half); None without both."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def second_order(decisions):
    """decisions: per decision of the pilot lives {"p_wager", "served", "p_rule"} (p_rule: the rule's probability
    that the agent's own action serves a lowest need, from what the agent saw and did). The AUC of P(wager) for served
    against not served, over all decisions and within slices of p_rule of 0.1 (mean weighted by the decisions of the
    slices that have both)."""
    p = np.array([d["p_wager"] for d in decisions])
    s = np.array([d["served"] for d in decisions], bool)
    q = np.array([d["p_rule"] for d in decisions])
    slices, weights = [], []
    for lo in np.arange(0, 1, 0.1):
        inside = (q >= lo) & ((q < lo + 0.1) if lo < 0.85 else (q <= 1.0))
        a = auc(p[inside], s[inside])
        if a is not None:
            slices.append(a)
            weights.append(int(inside.sum()))
    return {"auc": auc(p, s), "auc_within": float(np.average(slices, weights=weights)) if slices else None,
            "served": float(s.mean()), "decisions": len(decisions)}


def choose(loops, cuts):
    """loops: {setting: {seed: {"reward", "m_wager"}}}; cuts: {seed: reward}. A setting is reliable if, on each of the
    three seeds, its mean reward per life exceeds the cut of the same seed by at least GAIN and m_wager > CARRIED. The
    reliable setting with the largest mean m_wager; within TIE, P1; None if none is reliable."""
    table = {}
    for s in SETTINGS:
        if s not in loops:
            continue
        seeds = sorted(loops[s], key=int)
        m = {k: loops[s][k]["m_wager"] for k in seeds}
        gain = {k: loops[s][k]["reward"] - cuts[k] for k in seeds}
        table[s] = {"m_mean": float(np.mean(list(m.values()))), "gain_mean": float(np.mean(list(gain.values()))),
                    "m": m, "gain": gain,
                    "reliable": bool(sorted(seeds) == sorted(str(k) for k in SEEDS)
                                     and all(v > CARRIED for v in m.values()) and all(v >= GAIN for v in gain.values()))}
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
            cuts[seed] = r["reward"]
        else:
            loops.setdefault(r["name"], {})[seed] = dict(survival=r["survival"], reward=r["reward"],
                                                         wager_rate=r["wager_rate"], win_rate=r["win_rate"],
                                                         **effects(r["grafts"], grafts),
                                                         **second_order(r["decisions"]))
    chosen, table = choose({s: {k: {"reward": v["reward"], "m_wager": v["m_wager"]} for k, v in x.items()}
                            for s, x in loops.items()}, cuts)
    kinds = [g["kind"] for g in grafts]
    return {"chosen": chosen, "table": table, "loops": loops, "cuts": cuts,
            "grafts": {k: kinds.count(k) for k in KINDS},
            "b_same_history": sum(bool(g.get("same_history")) for g in grafts)}


# ----------------------------------------------------------------------------------------------------- torch part

def model(seed):
    """The gated loop of test 34 with one more output (the wager), read at the same "Choix" token."""
    import torch
    from torch import nn
    torch.manual_seed(int(np.random.default_rng([W.SEED, INIT_STREAM, seed]).integers(2 ** 31)))

    class Wagerer(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(VOCAB, TL.DIM)
            self.pos = nn.Embedding(POSITIONS, TL.DIM)
            layer = nn.TransformerEncoderLayer(TL.DIM, TL.HEADS, 4 * TL.DIM, dropout=0.0, batch_first=True,
                                               norm_first=True)
            self.layers = nn.TransformerEncoder(layer, TL.LAYERS, enable_nested_tensor=False)
            self.out = nn.Linear(TL.DIM, 2)
            self.wager = nn.Linear(TL.DIM, 2)
            self.value_head = nn.Linear(TL.DIM, 1)
            self.start = nn.Parameter(torch.randn(TL.DIM) * 0.02)
            self.gate = nn.Linear(TL.DIM, TL.DIM)
            nn.init.zeros_(self.gate.bias)

        def initial(self, n):
            return self.start[None].expand(n, TL.DIM)

        def step(self, state, last, event):
            """One turn for a batch: the state (n, DIM) and the tokens of the last action and of the event or "?"
            (n,). Returns the logits of R and M, the logits of no wager and wager, the value and the next state."""
            n = state.shape[0]
            choice = torch.full((n,), CHOICE, dtype=torch.long)
            x = torch.stack([state, self.tok(last), self.tok(event), self.tok(choice)], 1)
            h = self.layers(x + self.pos.weight[None])[:, -1]
            z = torch.sigmoid(self.gate(h))
            new = (1 - z) * state + z * nn.functional.layer_norm(h, (TL.DIM,))
            return self.out(h), self.wager(h), self.value_head(h)[:, 0], new
    return Wagerer()


def episodes(net, arm, worlds, pick, grad=False):
    """Lives in the given worlds, all at once; pick(p_R, p_wager, last) -> (actions, wagers) (numpy). Returns the
    decisions (life, turn, action, wager, E, N, last, event, hidden, P(wager)), the log probabilities of the chosen
    pair, the entropies (sum of both outputs), the values, the lengths (turns lived) and the wager rewards per
    decision."""
    import torch
    size = len(worlds)
    e, n = np.full(size, W.MAX), np.full(size, W.MAX)
    alive, last = np.ones(size, bool), np.ones(size, int)
    length = np.zeros(size, int)
    token_last, events, hidden = np.full(size, NONE), np.zeros(size, int), np.zeros(size, bool)
    state = net.initial(size)
    decisions, logps, entropies, values, wagered = [], [], [], [], []
    with torch.set_grad_enabled(grad):
        for t in range(1, W.TURNS + 1):
            for i in np.flatnonzero(alive):
                k, hidden[i] = draw_event(worlds[i])
                e[i], n[i] = e[i] + W.EVENTS[k][2], n[i] + W.EVENTS[k][3]
                events[i] = k
                if min(e[i], n[i]) <= 0:
                    alive[i] = False
            live = np.flatnonzero(alive)
            if len(live) == 0:
                break
            tokens = np.where(hidden, HIDDEN, EVENT0 + events)
            la, lw, value, nxt = net.step(state if arm == "loop" else net.initial(size), torch.tensor(token_last),
                                          torch.tensor(tokens))
            state = nxt
            pa, pw = torch.log_softmax(la[live], -1), torch.log_softmax(lw[live], -1)
            p_wager = pw[:, 1].exp().detach().numpy()
            actions, wagers = pick(pa[:, 0].exp().detach().numpy(), p_wager, last[live])
            ta, tw = torch.tensor(actions), torch.tensor(wagers)
            logps.append(pa.gather(1, ta[:, None])[:, 0] + pw.gather(1, tw[:, None])[:, 0])
            entropies.append(-(pa.exp() * pa).sum(-1) - (pw.exp() * pw).sum(-1))
            values.append(value[live])
            for i, a, w, p in zip(live, actions, wagers, p_wager):
                decisions.append((int(i), t, int(a), int(w), int(e[i]), int(n[i]), int(last[i]), int(events[i]),
                                  bool(hidden[i]), float(p)))
                wagered.append(wager_reward(e[i], n[i], int(a), int(w)))
                e[i], n[i] = W.after(e[i], n[i], int(a))
                last[i], token_last[i] = int(a), R + int(a)
                length[i] = t
    return decisions, logps, entropies, values, length, np.array(wagered)


def returns(decisions, length, wagered):
    """The discounted return of each decision: 1 for each later turn lived (as in test 34) plus the wagers of this
    decision and the later ones."""
    out = np.zeros(len(decisions))
    by_life = {}
    for i, d in enumerate(decisions):
        by_life.setdefault(d[0], []).append(i)
    for life, idx in by_life.items():
        g = 0.0
        for i in sorted(idx, key=lambda i: -decisions[i][1]):
            g = (1.0 if decisions[i][1] < length[life] else 0.0) + wagered[i] + GAMMA * g
            out[i] = g
    return out


def train(arm, seed, config, log, checkpoint=None):
    """The actor-critic of test 34 on the pair (action, wager); kept every 250 updates and taken back exactly after a
    stop (each update draws its worlds and choices from its own streams)."""
    import torch
    net = model(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    curve, window, start_at = [], [], 0
    if checkpoint is not None and Path(checkpoint).exists():
        state = torch.load(checkpoint, weights_only=False)
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        curve, window, start_at = state["curve"], state["window"], state["update"]
        log(f"  {checkpoint}: resumed at update {start_at}")
    for update in range(start_at, config["updates"]):
        worlds = [np.random.default_rng([W.SEED, WORLD_STREAM, seed, update, i]) for i in range(LIVES_PER_UPDATE)]
        rng = np.random.default_rng([W.SEED, ACTION_STREAM, seed, update])

        def pick(p_r, p_wager, last):
            return (rng.random(len(p_r)) >= p_r).astype(int), (rng.random(len(p_wager)) < p_wager).astype(int)
        decisions, logps, entropies, values, length, wagered = episodes(net, arm, worlds, pick, grad=True)
        ret = torch.tensor(returns(decisions, length, wagered), dtype=torch.float32)
        chosen, entropy, v = torch.cat(logps), torch.cat(entropies), torch.cat(values)
        with torch.no_grad():
            advantage = ret - v
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        loss = -(advantage * chosen).mean() + VALUE * ((v - ret) ** 2).mean() - ENTROPY * entropy.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        window.append(float(length.sum() + wagered.sum()) / LIVES_PER_UPDATE)
        if (update + 1) % 100 == 0:
            curve.append(round(float(np.mean(window)), 3))
            window = []
        if checkpoint is not None and (update + 1) % 250 == 0:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "curve": curve, "window": window,
                        "update": update + 1}, checkpoint)
    return net, curve


def greedy(p_r, p_wager, last):
    return np.where(p_r > 0.5, 0, np.where(p_r < 0.5, 1, 1 - last)), (p_wager > 0.5).astype(int)


def evaluate(net, arm, stream=PILOT_WORLDS, count=256):
    """On the given worlds with the most probable action and wager: survival per life, the mean reward per life (turns
    lived plus wagers), the share of decisions with a wager, the share of wagers won, and per decision P(wager),
    whether the action served a lowest need and the rule's probability that it would (from what the agent saw and
    did)."""
    worlds = [W.world_rng(stream, i) for i in range(count)]
    decisions, _, _, _, length, wagered = episodes(net, arm, worlds, greedy)
    belief = {i: start() for i in range(count)}
    rows = []
    for life, t, a, w, e, n, last, k, hidden, p in sorted(decisions, key=lambda d: (d[0], d[1])):
        belief[life] = observe(belief[life], k, hidden)
        b = belief[life]
        rows.append({"t": t, "life": life, "p_wager": round(p, 6), "served": served(e, n, a),
                     "p_rule": round(float(b[SERVES[a]].sum()), 6), "wager": w})
        belief[life] = act(b, a)
    bets = [d[3] for d in decisions]
    won = [served(d[4], d[5], d[2]) for d in decisions if d[3]]
    return {"survived": (length == W.TURNS).tolist(), "reward": float((length.sum() + wagered.sum()) / count),
            "wager_rate": float(np.mean(bets)), "win_rate": float(np.mean(won)) if won else None, "decisions": rows}


class Reader:
    """The agent reading a written life: P(R) and P(wager) at each decision, and the state passed by each decision,
    with or without a grafted state."""

    def __init__(self, net, arm):
        self.net, self.arm = net, arm

    def read(self, life, upto=None, graft=None):
        import torch
        state, last = self.net.initial(1).detach(), NONE
        p_r, p_wager, states = {}, {}, {}
        with torch.no_grad():
            for x in W.decisions(life):
                if upto is not None and x["t"] > upto:
                    break
                la, lw, _, nxt = self.net.step(state if self.arm == "loop" else self.net.initial(1),
                                               torch.tensor([last]), torch.tensor([token(x["event"], x["hidden"])]))
                p_r[x["t"]] = float(torch.softmax(la, -1)[0, 0])
                p_wager[x["t"]] = float(torch.softmax(lw, -1)[0, 1])
                state = graft[1] if graft is not None and x["t"] == graft[0] else nxt
                states[x["t"]] = state
                last = R + x["action"]
        return p_r, p_wager, states


def measure_grafts(reader, held, grafts):
    """Per graft: [index, P(R) without, P(R) with, P(wager) without, P(wager) with] at t."""
    own = [reader.read(life) for life in held]
    rows = []
    for g in grafts:
        p_r, p_wager, _ = reader.read(held[g["r"]], upto=g["t"], graft=(g["j"], own[g["d"]][2][g["j"]]))
        rows.append([g["index"], round(own[g["r"]][0][g["t"]], 7), round(p_r[g["t"]], 7),
                     round(own[g["r"]][1][g["t"]], 7), round(p_wager[g["t"]], 7)])
    return rows


def ready(root, name, seed):
    """P2 starts from the checkpoint of P1 at 48 000 updates, once that job is done (copied once)."""
    import torch
    target = Path(root) / "checkpoints" / f"{name}-{seed}.pt"
    if target.exists() or name != "P2":
        return True
    if not (Path(root) / "runs" / f"P1-{seed}.json").exists():
        return False
    source = Path(root) / "checkpoints" / f"P1-{seed}.pt"
    if torch.load(source, weights_only=False)["update"] != CONFIG["P1"]["updates"]:
        raise ValueError(f"{source} is not at {CONFIG['P1']['updates']} updates")
    shutil.copyfile(source, target)
    return True


def job(root, name, seed, held, grafts, log):
    arm = "cut" if name == "cut" else "loop"
    config = CONFIG["P1" if name == "cut" else name]
    net, curve = train(arm, seed, config, log, checkpoint=Path(root) / "checkpoints" / f"{name}-{seed}.pt")
    net.eval()
    x = evaluate(net, arm)
    row = {"name": name, "arm": arm, "seed": seed, "config": config, "survival": float(np.mean(x["survived"])),
           "survived": x["survived"], "reward": x["reward"], "wager_rate": x["wager_rate"], "win_rate": x["win_rate"],
           "decisions": x["decisions"], "curve": curve}
    text = (f"{name} seed {seed}: survival {row['survival']:.3f}, reward {row['reward']:.2f}, "
            f"wagers {row['wager_rate']:.3f}, won {row['win_rate']}")
    if arm == "loop":
        row["grafts"] = measure_grafts(Reader(net, arm), held, grafts)
        y = effects(row["grafts"], grafts)
        text += f", m_wager {y['m_wager']:+.3f}, |dP(R)| (u) {y['abs_action_u']:.3f}"
    log(text)
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
