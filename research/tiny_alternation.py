"""The alternation: two causes, one workspace, an internal time (docs/TINY_ALTERNATION_PILOT_PLAN.md): the pilot before
test 42.

The world of the wager (research/tiny_wager.py), unchanged. The gated loop of test 37 computes h and the state carried
to the next turn as before; the outputs read a workspace instead of h: 16 binary units drawn (Bernoulli) at each of
MICRO micro-steps of a turn, from a GRU micro-state fed by h and by the units of the previous micro-step. The action is
read on the units at each micro-step (probabilities averaged: a vote); the wager is read on a separate reader, a GRU fed
only by the units, micro-step after micro-step, at its last micro-step. Arms: S8 (8 micro-steps, drawn units), S1 (one
micro-step), D8 (8 micro-steps, units not drawn: z is the probability itself) and the cut (S8 with the state replaced
by the learned initial vector at each turn).

At each decision of the pilot worlds taken at a "?", the run keeps the 8 workspaces, the templates (the units' firing
frequencies when the "?" is replaced by "tu cours" (A) or by "il fait froid" (B), 16 independent draws) and P(wager)
under imposed workspaces (unanimity A, unanimity B, sharing A B A B..., sharing B A B A...). The measures and the
choice are numpy only, recomputed from the kept runs; training and runs need torch.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from . import need_world as W
from . import tiny_loop as TL
from . import tiny_loop_pilot2 as Q
from . import tiny_wager as X

UNITS, MICRO, TEMPLATE_RUNS = 16, 8, 16
ARMS = {"S8": (8, True), "S1": (1, True), "D8": (8, False), "cut": (8, True)}
SEEDS = (120, 121, 122)
UPDATES = 48000
INIT_STREAM, WORLD_STREAM, ACTION_STREAM, UNIT_STREAM = 140, 141, 142, 143
PILOT_WORLDS, EVAL_UNITS, TEMPLATE_UNITS, CONTROL_STREAM = 144, 145, 146, 147
PREP_POSTERIOR, PREP_SAMPLER, PREP_DRAWS = 951, 952, 953
GAIN_CUT, GAIN_ONE = 1.0, 0.5
CLIP = 0.01
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)
ROOT = "artifacts/tiny-alternation-pilot"
JOBS = [(arm, s) for s in SEEDS for arm in ("S8", "S1", "D8", "cut")]


# ----------------------------------------------------------------------------------------------- model-free landmarks

def posterior_landmarks(count=1500):
    """At the decisions of the rule taken at a "?" (stream PREP_POSTERIOR): the rule's probability that the hidden
    event was "tu cours" rather than "il fait froid", and whether the rule's action would differ under the two."""
    p_a, differ = [], []
    for index in range(count):
        worlds = W.world_rng(PREP_POSTERIOR, index)
        e = n = W.MAX
        b, last = X.start(), 1
        for t in range(1, W.TURNS + 1):
            k, hidden = X.draw_event(worlds)
            before = b
            e, n = e + W.EVENTS[k][2], n + W.EVENTS[k][3]
            if min(e, n) <= 0:
                break
            b = X.observe(b, k, hidden)
            a = X.rule(b, last)[0]
            if hidden:
                ba, bb = before @ X.EVENT_MOVE[1] * X.LIVE, before @ X.EVENT_MOVE[2] * X.LIVE
                p_a.append(float(ba.sum() / (ba.sum() + bb.sum())))
                differ.append(bool(ba.sum() > 1e-12 and bb.sum() > 1e-12
                                   and X.rule(ba / ba.sum(), last)[0] != X.rule(bb / bb.sum(), last)[0]))
            e, n = W.after(e, n, a)
            b, last = X.act(b, a), a
    p_a, differ = np.array(p_a), np.array(differ)
    ambiguous = (p_a > 0.2) & (p_a < 0.8)
    return {"decisions": int(len(p_a)), "quantiles": [round(float(q), 3) for q in np.quantile(p_a, [0.05, 0.5, 0.95])],
            "ambiguous": round(float(ambiguous.mean()), 3), "differ": round(float(differ.mean()), 3)}


def sampler_life(index, k):
    """A life (stream PREP_SAMPLER) of the rule's belief approximated by k samples of (E, N) at each decision (draws
    [SEED, PREP_DRAWS, k, life]; k None: the exact belief): the action that serves a lowest need in the most samples
    (equal: the other than the last), a wager when the share of samples where it serves exceeds SURE."""
    worlds = W.world_rng(PREP_SAMPLER, index)
    rng = np.random.default_rng([W.SEED, PREP_DRAWS, k or 0, index])
    e = n = W.MAX
    b, last, reward = X.start(), 1, 0.0
    for t in range(1, W.TURNS + 1):
        event, hidden = X.draw_event(worlds)
        e, n = e + W.EVENTS[event][2], n + W.EVENTS[event][3]
        if min(e, n) <= 0:
            return reward, False
        b = X.observe(b, event, hidden)
        if k is None:
            a, _, w = X.rule(b, last)
        else:
            s = rng.choice(X.SIZE, size=k, p=b)
            serve = [float(X.SERVES[0][s].mean()), float(X.SERVES[1][s].mean())]
            a = 0 if serve[0] > serve[1] else 1 if serve[1] > serve[0] else 1 - last
            w = int(serve[a] > X.SURE + 1e-12)
        reward += 1.0 + X.wager_reward(e, n, a, w)
        e, n = W.after(e, n, a)
        b, last = X.act(b, a), a
    return reward, True


def landmarks(count=1500):
    out = {"posterior": posterior_landmarks(count), "samplers": {}}
    for k in (None, 64, 16, 8, 4, 2, 1):
        res = [sampler_life(i, k) for i in range(count)]
        out["samplers"]["exact" if k is None else str(k)] = {
            "survival": round(float(np.mean([r[1] for r in res])), 3),
            "reward": round(float(np.mean([r[0] for r in res])), 2)}
    return out


# ----------------------------------------------------------------------------------------------------- the measures

def unpack(masks):
    """The workspaces kept as integers (one bit per unit) back to a (micro, UNITS) array of 0 and 1."""
    return ((np.asarray(masks, np.int64)[:, None] >> np.arange(UNITS)) & 1).astype(float)


def loglik(z, p):
    """Bernoulli log likelihood of each row of z (k, units) under the firing probabilities p (units)."""
    return (z * np.log(p) + (1 - z) * np.log(1 - p)).sum(-1)


def states(z, a, b):
    """1 (closer to A) or 2 (closer to B) at each micro-step, by the sign of the log likelihood ratio (0 counts as B)."""
    return np.where(loglik(z, a) - loglik(z, b) > 0, 1, 2)


def leaning(f, a, b):
    """Distance to B minus distance to A: positive when f looks like A."""
    return float(np.linalg.norm(f - b) - np.linalg.norm(f - a))


HALVES = (np.arange(UNITS) % 2 == 0, np.arange(UNITS) % 2 == 1)


def split_half(z, a, b):
    """The state inferred on one half of the units, read on the other half: leaning of the mean of the other half
    over state-1 micro-steps minus over state-2 micro-steps, averaged over the two directions that have both states;
    None if neither has."""
    values = []
    for infer, read in (HALVES, HALVES[::-1]):
        s = states(z[:, infer], a[infer], b[infer])
        if (s == 1).any() and (s == 2).any():
            values.append(leaning(z[s == 1][:, read].mean(0), a[read], b[read])
                          - leaning(z[s == 2][:, read].mean(0), a[read], b[read]))
    return float(np.mean(values)) if values else None


def two_minus_one(z, a, b):
    """Per micro-step, the log likelihood of z under the equal mixture of A and B minus under the best single
    template w A + (1 - w) B (w on WEIGHTS, the best for this decision), on each half, averaged over the halves."""
    out = []
    for h in HALVES:
        zh, ah, bh = z[:, h], a[h], b[h]
        two = np.logaddexp(loglik(zh, ah), loglik(zh, bh)).sum() - len(zh) * np.log(2)
        one = max(loglik(zh, w * ah + (1 - w) * bh).sum() for w in WEIGHTS)
        out.append((two - one) / len(zh))
    return float(np.mean(out))


def dwell(s):
    """The mean length of the runs of equal states."""
    changes = np.flatnonzero(np.diff(s) != 0)
    return len(s) / (len(changes) + 1)


def controls(cases, rng):
    """The control models with the same templates and micro-steps: "average" (each unit drawn with (A + B) / 2) and
    "uncoordinated" (each unit alone switches between A and B, starting at random, with probability 1 / D per
    micro-step, D the mean dwell measured in the agent's workspaces, at least 1)."""
    d = max(1.0, float(np.mean([dwell(states(c["z"], c["a"], c["b"])) for c in cases])))
    average, uncoordinated = [], []
    for c in cases:
        a, b, k = c["a"], c["b"], len(c["z"])
        average.append(dict(c, z=(rng.random((k, UNITS)) < (a + b) / 2).astype(float)))
        own = np.zeros((k, UNITS), bool)
        own[0] = rng.random(UNITS) < 0.5
        for i in range(1, k):
            flip = rng.random(UNITS) < 1 / d
            own[i] = np.where(flip, ~own[i - 1], own[i - 1])
        p = np.where(own, a, b)
        uncoordinated.append(dict(c, z=(rng.random((k, UNITS)) < p).astype(float)))
    return {"dwell": d, "average": average, "uncoordinated": uncoordinated}


def describe(cases):
    """Alternation, split-half and two-against-one over the decisions at a "?"."""
    s = [states(c["z"], c["a"], c["b"]) for c in cases]
    halves = [split_half(c["z"], c["a"], c["b"]) for c in cases]
    halves = [h for h in halves if h is not None]
    tmo = [two_minus_one(c["z"], c["a"], c["b"]) for c in cases]
    return {"both_states": float(np.mean([(x == 1).any() and (x == 2).any() for x in s])),
            "changes": float(np.mean([np.count_nonzero(np.diff(x)) for x in s])),
            "share_state1": float(np.mean([np.mean(x == 1) for x in s])),
            "split_half": float(np.mean(halves)) if halves else None, "split_half_cases": len(halves),
            "two_minus_one": float(np.mean(tmo)), "two_better": float(np.mean([v > 0 for v in tmo]))}


def ranks(x):
    """Ranks with ties at their average rank."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        same = x == v
        r[same] = r[same].mean()
    return r


def rank_correlation(x, y):
    """Spearman's correlation; None with fewer than 3 values or without variation."""
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def wager_reading(cases):
    """Within slices of the rule's probability of 0.1 (the probability that the agent's own action serves a lowest
    need), the rank correlation of P(wager) with the agreement |share of state-1 micro-steps - 0.5|, averaged with the
    weights of the slices where it exists; and the intervention: P(wager) under unanimity minus under sharing (the two
    unanimities against the two sharings)."""
    q = np.array([c["p_rule"] for c in cases])
    index = np.minimum((q * 10 + 1e-9).astype(int), 9)
    agreement = np.array([abs(np.mean(states(c["z"], c["a"], c["b"]) == 1) - 0.5) for c in cases])
    p = np.array([c["p_wager"] for c in cases])
    values, weights = [], []
    for i in range(10):
        inside = index == i
        r = rank_correlation(agreement[inside], p[inside])
        if r is not None:
            values.append(r)
            weights.append(int(inside.sum()))
    forced = np.array([np.mean(c["forced"][:2]) - np.mean(c["forced"][2:]) for c in cases])
    return {"agreement_mean": float(agreement.mean()),
            "rank_within": float(np.average(values, weights=weights)) if values else None,
            "forced_unanimity_minus_sharing": float(forced.mean()),
            "forced_positive": float(np.mean(forced > 0))}


def gains(cases):
    """The mean distance of z to the middle of the templates, over the micro-steps in the state of the hypothesis
    whose rule action is the agent's action, and over the others (decisions where the two hypotheses' rule actions
    differ and both states appear)."""
    mine, other = [], []
    for c in cases:
        if c["hypothesis"] is None:
            continue
        s = states(c["z"], c["a"], c["b"])
        if not ((s == 1).any() and (s == 2).any()):
            continue
        d = np.linalg.norm(c["z"] - (c["a"] + c["b"]) / 2, axis=1)
        mine.append(d[s == c["hypothesis"]].mean())
        other.append(d[s != c["hypothesis"]].mean())
    return {"chosen": float(np.mean(mine)) if mine else None, "other": float(np.mean(other)) if other else None,
            "ratio": float(np.mean(mine) / np.mean(other)) if mine and np.mean(other) > 0 else None,
            "cases": len(mine)}


def cases_of(row):
    """The decisions at a "?" of a run, as arrays."""
    out = []
    for c in row["cases"]:
        z = unpack(c["z"]) if row["sample"] else np.asarray(c["z"], float)
        out.append({"z": z, "a": np.clip(np.asarray(c["a"]), CLIP, 1 - CLIP),
                    "b": np.clip(np.asarray(c["b"]), CLIP, 1 - CLIP), "p_wager": c["p_wager"],
                    "p_rule": c["p_rule"], "forced": c["forced"], "hypothesis": c["hypothesis"]})
    return out


def measures(row):
    """All the published measures of one run with micro-steps (none for S1 and the cut beyond the reward)."""
    cases = cases_of(row)
    if not cases or row["micro"] == 1:
        return {}
    out = {"agent": describe(cases), "reading": wager_reading(cases), "gains": gains(cases)}
    if row["sample"]:
        made = controls(cases, np.random.default_rng([W.SEED, CONTROL_STREAM, row["seed"]]))
        out["controls"] = {"dwell": made["dwell"], "average": describe(made["average"]),
                           "uncoordinated": describe(made["uncoordinated"])}
    return out


def choose(rewards):
    """rewards: {arm: {seed: reward}}. S8 is reliable if, on each of the three seeds, its mean reward per life exceeds
    the cut of the same seed by at least GAIN_CUT and S1 of the same seed by at least GAIN_ONE."""
    seeds = [str(s) for s in SEEDS]
    if not all(s in rewards.get(a, {}) for a in ("S8", "S1", "cut") for s in seeds):
        return {"reliable": False, "complete": False}
    cut = {s: rewards["S8"][s] - rewards["cut"][s] for s in seeds}
    one = {s: rewards["S8"][s] - rewards["S1"][s] for s in seeds}
    return {"complete": True, "gain_cut": cut, "gain_one": one,
            "reliable": bool(all(v >= GAIN_CUT for v in cut.values()) and all(v >= GAIN_ONE for v in one.values()))}


def summary(root):
    runs = {}
    for path in sorted((Path(root) / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        runs.setdefault(r["arm"], {})[str(r["seed"])] = r
    rewards = {a: {s: r["reward"] for s, r in v.items()} for a, v in runs.items()}
    table = {a: {s: {"survival": r["survival"], "reward": r["reward"], "wager_rate": r["wager_rate"],
                     "win_rate": r["win_rate"], **measures(r)} for s, r in v.items()} for a, v in runs.items()}
    return {"choice": choose(rewards), "runs": table}


# ----------------------------------------------------------------------------------------------------- torch part

def generator(*key):
    import torch
    return torch.Generator().manual_seed(int(np.random.default_rng([W.SEED, *key]).integers(2 ** 62)))


def model(seed, micro, sample):
    import torch
    from torch import nn
    torch.manual_seed(int(np.random.default_rng([W.SEED, INIT_STREAM, seed]).integers(2 ** 31)))

    class Alternator(nn.Module):
        def __init__(self):
            super().__init__()
            self.micro, self.sample = micro, sample
            self.tok = nn.Embedding(X.VOCAB, TL.DIM)
            self.pos = nn.Embedding(X.POSITIONS, TL.DIM)
            layer = nn.TransformerEncoderLayer(TL.DIM, TL.HEADS, 4 * TL.DIM, dropout=0.0, batch_first=True,
                                               norm_first=True)
            self.layers = nn.TransformerEncoder(layer, TL.LAYERS, enable_nested_tensor=False)
            self.value_head = nn.Linear(TL.DIM, 1)
            self.start = nn.Parameter(torch.randn(TL.DIM) * 0.02)
            self.gate = nn.Linear(TL.DIM, TL.DIM)
            nn.init.zeros_(self.gate.bias)
            self.cell = nn.GRUCell(TL.DIM + UNITS, TL.DIM)
            self.units = nn.Linear(TL.DIM, UNITS)
            self.out = nn.Linear(UNITS, 2)
            self.reader = nn.GRUCell(UNITS, TL.DIM)
            self.wager = nn.Linear(TL.DIM, 2)

        def initial(self, n):
            return self.start[None].expand(n, TL.DIM)

        def front(self, state, last, event):
            """As in test 37: h, the value and the next state, from the state and the tokens of the last action and
            of the event or "?"."""
            n = state.shape[0]
            choice = torch.full((n,), X.CHOICE, dtype=torch.long)
            x = torch.stack([state, self.tok(last), self.tok(event), self.tok(choice)], 1)
            h = self.layers(x + self.pos.weight[None])[:, -1]
            z = torch.sigmoid(self.gate(h))
            return h, self.value_head(h)[:, 0], (1 - z) * state + z * nn.functional.layer_norm(h, (TL.DIM,))

        def workspace(self, h, noise=None, force=None):
            """The micro-steps of a turn: P(action) (the mean over micro-steps), P(wager) (the reader at its last
            micro-step) and the workspaces (n, micro, UNITS). noise: uniform draws (n, micro, UNITS) for the drawn
            units; force: imposed workspaces (n, micro, UNITS)."""
            n = h.shape[0]
            m = h.new_zeros(n, TL.DIM)
            r = h.new_zeros(n, TL.DIM)
            z = h.new_zeros(n, UNITS)
            p_action, zs = 0, []
            for k in range(self.micro):
                m = self.cell(torch.cat([h, z], 1), m)
                p = torch.sigmoid(self.units(m))
                if force is not None:
                    z = force[:, k]
                elif self.sample:
                    z = (noise[:, k] < p).float() + p - p.detach()
                else:
                    z = p
                p_action = p_action + torch.softmax(self.out(z), -1)
                r = self.reader(z, r)
                zs.append(z)
            return p_action / self.micro, torch.softmax(self.wager(r), -1), torch.stack(zs, 1)

    return Alternator()


def episodes(net, arm, worlds, pick, noise, grad=False, keep=False):
    """Lives in the given worlds, all at once, as in research/tiny_wager.py; pick(p_R, p_wager, last) -> (actions,
    wagers); noise(): uniform draws (len(worlds), micro, UNITS) for one turn. keep: also the state, the last token and
    the workspaces at each decision taken at a "?"."""
    import torch
    size = len(worlds)
    e, n = np.full(size, W.MAX), np.full(size, W.MAX)
    alive, last = np.ones(size, bool), np.ones(size, int)
    length = np.zeros(size, int)
    token_last, events, hidden = np.full(size, X.NONE), np.zeros(size, int), np.zeros(size, bool)
    state = net.initial(size)
    decisions, logps, entropies, values, wagered, kept = [], [], [], [], [], {}
    with torch.set_grad_enabled(grad):
        for t in range(1, W.TURNS + 1):
            for i in np.flatnonzero(alive):
                k, hidden[i] = X.draw_event(worlds[i])
                e[i], n[i] = e[i] + W.EVENTS[k][2], n[i] + W.EVENTS[k][3]
                events[i] = k
                if min(e[i], n[i]) <= 0:
                    alive[i] = False
            draws = noise()
            live = np.flatnonzero(alive)
            if len(live) == 0:
                break
            tokens = np.where(hidden, X.HIDDEN, X.EVENT0 + events)
            before = state if arm != "cut" else net.initial(size)
            h, value, nxt = net.front(before, torch.tensor(token_last), torch.tensor(tokens))
            pa, pw, zs = net.workspace(h, noise=draws)
            state = nxt
            la, lw = torch.log(pa[live] + 1e-12), torch.log(pw[live] + 1e-12)
            p_wager = pw[live, 1].detach().numpy()
            actions, wagers = pick(pa[live, 0].detach().numpy(), p_wager, last[live])
            ta, tw = torch.tensor(actions), torch.tensor(wagers)
            logps.append(la.gather(1, ta[:, None])[:, 0] + lw.gather(1, tw[:, None])[:, 0])
            entropies.append(-(la.exp() * la).sum(-1) - (lw.exp() * lw).sum(-1))
            values.append(value[live])
            for i, a, w, p in zip(live, actions, wagers, p_wager):
                decisions.append((int(i), t, int(a), int(w), int(e[i]), int(n[i]), int(last[i]), int(events[i]),
                                  bool(hidden[i]), float(p)))
                if keep and hidden[i]:
                    kept[(int(i), t)] = (before[i].detach().clone(), int(token_last[i]), zs[i].detach().numpy())
                wagered.append(X.wager_reward(e[i], n[i], int(a), int(w)))
                e[i], n[i] = W.after(e[i], n[i], int(a))
                last[i], token_last[i] = int(a), X.R + int(a)
                length[i] = t
    return decisions, logps, entropies, values, length, np.array(wagered), kept


def train(arm, seed, log, checkpoint=None):
    """The actor-critic of test 37 on the pair (action, wager), with the units drawn from their own stream; kept every
    250 updates and taken back exactly after a stop."""
    import torch
    import os
    micro, sample = ARMS[arm]
    net = model(seed, micro, sample)
    opt = torch.optim.AdamW(net.parameters(), lr=X.LR)
    curve, window, start_at = [], [], 0
    if checkpoint is not None and Path(checkpoint).exists():
        state = torch.load(checkpoint, weights_only=False)
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        curve, window, start_at = state["curve"], state["window"], state["update"]
        log(f"  {checkpoint}: resumed at update {start_at}")
    for update in range(start_at, UPDATES):
        worlds = [np.random.default_rng([W.SEED, WORLD_STREAM, seed, update, i]) for i in range(X.LIVES_PER_UPDATE)]
        rng = np.random.default_rng([W.SEED, ACTION_STREAM, seed, update])
        g = generator(UNIT_STREAM, seed, update)

        def pick(p_r, p_wager, last):
            return (rng.random(len(p_r)) >= p_r).astype(int), (rng.random(len(p_wager)) < p_wager).astype(int)

        def noise():
            return torch.rand((X.LIVES_PER_UPDATE, micro, UNITS), generator=g)
        decisions, logps, entropies, values, length, wagered, _ = episodes(net, arm, worlds, pick, noise, grad=True)
        ret = torch.tensor(X.returns(decisions, length, wagered), dtype=torch.float32)
        chosen, entropy, v = torch.cat(logps), torch.cat(entropies), torch.cat(values)
        with torch.no_grad():
            advantage = ret - v
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        loss = -(advantage * chosen).mean() + X.VALUE * ((v - ret) ** 2).mean() - X.ENTROPY * entropy.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        window.append(float(length.sum() + wagered.sum()) / X.LIVES_PER_UPDATE)
        if (update + 1) % 100 == 0:
            curve.append(round(float(np.mean(window)), 3))
            window = []
        if checkpoint is not None and (update + 1) % 250 == 0:
            tmp = Path(str(checkpoint) + ".tmp")
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "curve": curve, "window": window,
                        "update": update + 1}, tmp)
            os.replace(tmp, checkpoint)
    return net, curve


def pack(z):
    """Drawn workspaces (micro, UNITS) of 0 and 1 to integers, one bit per unit."""
    return [int(v) for v in (np.rint(z).astype(np.int64) << np.arange(UNITS)).sum(1)]


def evaluate(net, arm, count=256):
    """On the pilot worlds with the most probable action and wager (units drawn from stream EVAL_UNITS): survival,
    reward, wager rate, win rate and, at each decision taken at a "?", the workspaces, the templates, P(wager), the
    rule's probability that the agent's action serves a lowest need, P(wager) under the imposed workspaces, and the
    hypothesis (1: A, 2: B) whose rule action is the agent's action when the two differ."""
    import torch
    micro, sample = ARMS[arm]
    worlds = [W.world_rng(PILOT_WORLDS, i) for i in range(count)]
    g = generator(EVAL_UNITS)
    decisions, _, _, _, length, wagered, kept = episodes(
        net, arm, worlds, X.greedy, lambda: torch.rand((count, micro, UNITS), generator=g), keep=micro > 1)
    belief = {i: X.start() for i in range(count)}
    cases = []
    with torch.no_grad():
        for life, t, a, w, e, n, last, k, hidden, p in sorted(decisions, key=lambda d: (d[0], d[1])):
            before = belief[life]
            belief[life] = X.observe(before, k, hidden)
            b = belief[life]
            if (life, t) in kept:
                state, token_last, z = kept[(life, t)]
                templates = []
                for event in X.STRONG:
                    h, _, _ = net.front(state[None].expand(TEMPLATE_RUNS, TL.DIM),
                                        torch.full((TEMPLATE_RUNS,), token_last), torch.full((TEMPLATE_RUNS,),
                                                                                             X.EVENT0 + event))
                    gt = generator(TEMPLATE_UNITS, life, t, event)
                    _, _, zs = net.workspace(h, noise=torch.rand((TEMPLATE_RUNS, micro, UNITS), generator=gt))
                    templates.append(zs.reshape(-1, UNITS).mean(0).numpy())
                modes = [torch.tensor((x > 0.5).astype(np.float32)) for x in templates]
                forced_z = [torch.stack([modes[0]] * micro), torch.stack([modes[1]] * micro)] + [
                    torch.stack([modes[(i + j) % 2] for i in range(micro)]) for j in (0, 1)]
                h, _, _ = net.front(state[None], torch.tensor([token_last]), torch.tensor([X.HIDDEN]))
                forced = [float(net.workspace(h, force=f[None])[1][0, 1]) for f in forced_z]
                after = [before @ X.EVENT_MOVE[c] * X.LIVE for c in X.STRONG]
                rules = [X.rule(x / x.sum(), last)[0] if x.sum() > 1e-12 else None for x in after]
                hypothesis = (1 if rules[0] == a else 2) if None not in rules and rules[0] != rules[1] else None
                cases.append({"life": life, "t": t, "z": pack(z) if sample else np.round(z, 6).tolist(),
                              "a": np.round(templates[0], 6).tolist(), "b": np.round(templates[1], 6).tolist(),
                              "p_wager": p, "p_rule": round(float(b[X.SERVES[a]].sum()), 12),
                              "forced": [round(v, 7) for v in forced], "hypothesis": hypothesis})
            belief[life] = X.act(b, a)
    bets = [d[3] for d in decisions]
    won = [X.served(d[4], d[5], d[2]) for d in decisions if d[3]]
    return {"survived": (length == W.TURNS).tolist(), "reward": float((length.sum() + wagered.sum()) / count),
            "wager_rate": float(np.mean(bets)), "win_rate": float(np.mean(won)) if won else None, "cases": cases}


def job(root, arm, seed, log):
    net, curve = train(arm, seed, log, checkpoint=Path(root) / "checkpoints" / f"{arm}-{seed}.pt")
    net.eval()
    x = evaluate(net, arm)
    micro, sample = ARMS[arm]
    row = {"arm": arm, "seed": seed, "updates": UPDATES, "micro": micro, "sample": sample,
           "survival": float(np.mean(x["survived"])), "survived": x["survived"], "reward": x["reward"],
           "wager_rate": x["wager_rate"], "win_rate": x["win_rate"], "cases": x["cases"], "curve": curve}
    m = measures(row)
    text = f"{arm} seed {seed}: survival {row['survival']:.3f}, reward {row['reward']:.2f}"
    if m:
        text += (f", both states {m['agent']['both_states']:.3f}, split half {m['agent']['split_half']}, "
                 f"forced {m['reading']['forced_unanimity_minus_sharing']:+.3f}")
    log(text)
    W.write_atomic(Path(root) / "runs" / f"{arm}-{seed}.json", (json.dumps(row) + "\n").encode())


def run(a):
    import torch
    torch.set_num_threads(1)
    log = lambda m: print(m, flush=True)
    root = Path(a.out)
    for d in ("runs", "checkpoints", "claims"):
        (root / d).mkdir(parents=True, exist_ok=True)
    while True:
        left = [(n, s) for n, s in JOBS if not (root / "runs" / f"{n}-{s}.json").exists()]
        if not left:
            return
        for name, seed in left:
            if Q.claim(root, name, seed):
                if (root / "runs" / f"{name}-{seed}.json").exists():
                    (root / "claims" / f"{name}-{seed}").unlink()
                    break
                job(root, name, seed, log)
                (root / "claims" / f"{name}-{seed}").unlink()
                break
        else:
            if all(Q.taken(root, n, s) for n, s in left):
                return
            time.sleep(60)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("landmarks")
    sub.add_parser("run")
    sub.add_parser("choose")
    sub.add_parser("check")
    for s in sub.choices.values():
        s.add_argument("--out", default=ROOT)
    a = parser.parse_args(argv)
    if a.command == "landmarks":
        print(json.dumps(landmarks(), indent=1))
        return
    if a.command == "run":
        run(a)
        return
    result = json.loads(json.dumps(summary(a.out)))
    path = Path(a.out) / "choice.json"
    if a.command == "choose":
        path.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result["choice"]))
    else:
        differs = result != json.loads(path.read_text())
        print("differs:", "yes" if differs else "none")
        if differs:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
