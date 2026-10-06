"""The loop: a state that only passes from one turn to the next (docs/TINY_LOOP_PROTOCOL.md): LOOP1 to LOOP4 of test 32.

At each turn a small transformer learned from scratch (2 layers, dimension 64, 4 heads) reads 4 positions: the state
s_t (a vector), the last action, the event of the turn, "Choix". Out of "Choix" come the choice (R or M), the value and
the next state s_{t+1} (that output, normalized). A turn never sees past turns: only s_t passes from one turn to the
next. The control "cut" is the same network whose state is replaced at every turn by the learned initial one. Both
learn only from their survival with the actor-critic of test 31 (setting P2), the gradient going through the state
over the whole life. Measures: the survival on the 256 worlds of test 22, and the graft of s_{j+1} from a donor life
(same written action at j) into a recipient life written by the event rule, read at t = j + g. The verdicts are numpy
only; training and measures need torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_memory as NM
from . import need_relief as RL
from . import tiny_survival as TS
from .need_carried_content import rule_effect

ARMS = ("loop", "cut")
SEEDS = tuple(range(10))
NONE, R, M, EVENT0, CHOICE = 0, 1, 2, 3, 9
VOCAB, POSITIONS = 10, 4
DIM, HEADS, LAYERS = 64, 4, 2
INIT_STREAM, WORLD_STREAM, ACTION_STREAM, LIVES_STREAM, DRAW_STREAM = 76, 74, 75, 77, 78
UPDATES, LIVES_PER_UPDATE, LR, GAMMA, VALUE, ENTROPY = 4000, 64, 3e-4, 0.9, 0.5, 0.01
LIVES, GAPS, PER_GAP = 128, (1, 2), 2
COUNTS = {"a": 400, "a0": 200, "b": 200}
LOOP1, LOOP2, LOOP3, LOOP3_SEED, LOOP3_SEEDS, HALF = 0.08, 0.05, 0.10, 0.05, 8, 0.5
MIN_B, MIN_GRAFTS, TOLERANCE = 0.60, 100, 1e-6
ROOT = "artifacts/tiny-loop"


def lives():
    return RL.writer_lives(LIVES_STREAM, LIVES)


def draw(held, stream=DRAW_STREAM):
    """The grafts of the protocol, without any model: recipients in order; per gap up to PER_GAP turns t; per t one
    donor of each kind, alive at j with the same written action at j: (a) other needs after the event of j and a rule
    effect not zero, (a0) other needs and a rule effect zero, (b) the same needs."""
    rng = np.random.default_rng([W.SEED, stream, 0])
    at = [{x["t"]: x for x in W.decisions(life)} for life in held]
    out, counts = [], {kind: 0 for kind in COUNTS}
    for r, life in enumerate(held):
        for g in GAPS:
            turns = [t for t in sorted(at[r]) if t - g >= 1 and t - g in at[r]]
            for k in sorted(rng.permutation(len(turns))[:PER_GAP]):
                t = turns[k]
                j = t - g
                mine = at[r][j]
                for kind in ("a", "a0", "b"):
                    donors = []
                    for d in range(len(held)):
                        x = at[d].get(j)
                        if d == r or x is None or x["action"] != mine["action"]:
                            continue
                        if ((x["E"], x["N"]) == (mine["E"], mine["N"])) != (kind == "b"):
                            continue
                        e = rule_effect(life, held[d], j, t)
                        if e is None or (kind == "a" and e == 0) or (kind == "a0" and e != 0):
                            continue
                        donors.append((d, e))
                    if not donors:
                        continue
                    d, e = donors[int(rng.integers(len(donors)))]
                    if counts[kind] < COUNTS[kind]:
                        out.append({"index": len(out), "kind": kind, "r": r, "d": d, "j": j, "t": t, "g": g,
                                    "e": int(e)})
                        counts[kind] += 1
        if all(counts[k] >= COUNTS[k] for k in COUNTS):
            break
    return out


def graft_measures(rows):
    """rows: the grafts with "p": [P(R) without, with the graft]. Per kind and gap, for one seed and one agent."""
    delta = np.array([r["p"][1] - r["p"][0] for r in rows])
    e = np.array([r["e"] for r in rows], float)
    kind = np.array([r["kind"] for r in rows])
    gap = np.array([r["g"] for r in rows])
    a, a0, b = kind == "a", kind == "a0", kind == "b"
    out = {"align": float((delta[a] * e[a]).mean()), "abs_a": float(np.abs(delta[a]).mean()),
           "abs_a0": float(np.abs(delta[a0]).mean()), "abs_b": float(np.abs(delta[b]).mean())}
    for g in GAPS:
        out[f"align_g{g}"] = float((delta[a & (gap == g)] * e[a & (gap == g)]).mean())
        out[f"abs_b_g{g}"] = float(np.abs(delta[b & (gap == g)]).mean())
    return out


def verdicts(runs, grafts, setup, event_rule):
    """runs: {arm: {seed: {"survival", "agrees", "probe", "grafts": [[index, p without, p with]]}}}; grafts: the
    draw; setup: the checks (seed 0)."""
    seeds = sorted(runs["loop"], key=int)
    gain = TS.interval([runs["loop"][s]["survival"] - runs["cut"][s]["survival"] for s in seeds])
    beyond = TS.interval([runs["loop"][s]["survival"] - event_rule for s in seeds])
    survival = {a: float(np.mean([runs[a][s]["survival"] for s in seeds])) for a in runs}
    measured = {}
    for arm in runs:
        measured[arm] = {}
        for s in seeds:
            got = {i: (p0, p1) for i, p0, p1 in runs[arm][s]["grafts"]}
            if sorted(got) != [g["index"] for g in grafts]:
                raise ValueError(f"the grafts of {arm} seed {s} are not those the protocol draws")
            measured[arm][s] = graft_measures([dict(g, p=got[g["index"]]) for g in grafts])
    m = [measured["loop"][s]["align"] for s in seeds]
    align = TS.interval(m)
    abs_a = float(np.mean([measured["loop"][s]["abs_a"] for s in seeds]))
    abs_b = float(np.mean([measured["loop"][s]["abs_b"] for s in seeds]))
    need_not_history = TS.interval([measured["loop"][s]["abs_a"] - measured["loop"][s]["abs_b"] for s in seeds])
    out = {"LOOP1": bool(gain["mean"] >= LOOP1 and gain["low"] > 0),
           "LOOP2": bool(beyond["mean"] >= LOOP2 and beyond["low"] > 0),
           "LOOP3": bool(align["mean"] >= LOOP3 and align["low"] > 0
                         and sum(x > LOOP3_SEED for x in m) >= LOOP3_SEEDS),
           "LOOP4": bool(abs_b <= HALF * abs_a and need_not_history["low"] > 0)}
    kinds = {k: sum(g["kind"] == k for g in grafts) for k in COUNTS}
    out["valid"] = bool(survival["cut"] >= MIN_B and setup["cut_gap"] <= TOLERANCE
                        and setup["self_graft_gap"] <= TOLERANCE and kinds["a"] >= MIN_GRAFTS
                        and kinds["b"] >= MIN_GRAFTS and all(len(runs[a]) == len(SEEDS) for a in ARMS))
    out["global"] = bool(out["valid"] and all(out[k] for k in ("LOOP1", "LOOP2", "LOOP3", "LOOP4")))
    values = {"event_rule": event_rule, "survival": survival, "survival_gain": gain, "beyond_event_rule": beyond,
              "align": align, "align_seeds_above": int(sum(x > LOOP3_SEED for x in m)),
              "abs_a": abs_a, "abs_b": abs_b, "need_not_history": need_not_history, "grafts": kinds,
              "agrees_with_needs_rule": {a: float(np.mean([runs[a][s]["agrees"] for s in seeds])) for a in runs},
              "by_seed": {s: {a: {"survival": runs[a][s]["survival"], **measured[a][s], "probe": runs[a][s]["probe"]}
                              for a in runs} for s in seeds}}
    return {"verdicts": out, "values": values}


# ----------------------------------------------------------------------------------------------------- torch part

def model(seed):
    import torch
    from torch import nn
    torch.manual_seed(int(np.random.default_rng([W.SEED, INIT_STREAM, seed]).integers(2 ** 31)))

    class Loop(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(VOCAB, DIM)
            self.pos = nn.Embedding(POSITIONS, DIM)
            layer = nn.TransformerEncoderLayer(DIM, HEADS, 4 * DIM, dropout=0.0, batch_first=True, norm_first=True)
            self.layers = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
            self.out = nn.Linear(DIM, 2)
            self.value_head = nn.Linear(DIM, 1)
            self.start = nn.Parameter(torch.randn(DIM) * 0.02)

        def initial(self, n):
            return self.start[None].expand(n, DIM)

        def step(self, state, last, event):
            """One turn for a batch: the state (n, DIM), the last action token and the event token (n,). Returns the
            logits of R and M, the value and the next state."""
            n = state.shape[0]
            choice = torch.full((n,), CHOICE, dtype=torch.long)
            x = torch.stack([state, self.tok(last), self.tok(event), self.tok(choice)], 1) + self.pos.weight[None]
            h = self.layers(x)[:, -1]
            return self.out(h), self.value_head(h)[:, 0], nn.functional.layer_norm(h, (DIM,))
    return Loop()


def episodes(net, arm, worlds, pick, grad=False):
    """Lives in the given worlds (one generator each), all at once; pick(p_R, last) -> actions (numpy). Every life is
    stepped every turn (the dead ones are ignored). Returns the decisions (life, turn, action, E, N, last), their log
    probabilities of the chosen action, entropies and values (tensors), and the lengths."""
    import torch
    n_lives = len(worlds)
    e, n = np.full(n_lives, W.MAX), np.full(n_lives, W.MAX)
    alive, last = np.ones(n_lives, bool), np.ones(n_lives, int)
    length = np.zeros(n_lives, int)
    token_last = np.full(n_lives, NONE)
    events = np.zeros(n_lives, int)
    state = net.initial(n_lives)
    decisions, logps, entropies, values = [], [], [], []
    with torch.set_grad_enabled(grad):
        for t in range(1, W.TURNS + 1):
            for i in np.flatnonzero(alive):
                k = int(worlds[i].choice(len(W.EVENTS), p=W.PROBS))
                e[i], n[i] = e[i] + W.EVENTS[k][2], n[i] + W.EVENTS[k][3]
                events[i] = EVENT0 + k
                if e[i] <= 0 or n[i] <= 0:
                    alive[i] = False
            live = np.flatnonzero(alive)
            if len(live) == 0:
                break
            logits, value, nxt = net.step(state if arm == "loop" else net.initial(n_lives), torch.tensor(token_last),
                                          torch.tensor(events))
            state = nxt
            logp = torch.log_softmax(logits[live], -1)
            p_r = logp[:, 0].exp().detach().numpy()
            actions = pick(p_r, last[live])
            chosen = torch.tensor(actions)
            logps.append(logp.gather(1, chosen[:, None])[:, 0])
            entropies.append(-(logp.exp() * logp).sum(-1))
            values.append(value[live])
            for i, a in zip(live, actions):
                decisions.append((int(i), t, int(a), int(e[i]), int(n[i]), int(last[i])))
                e[i], n[i] = W.after(e[i], n[i], int(a))
                last[i] = int(a)
                token_last[i] = R + int(a)
                length[i] = t
    return decisions, logps, entropies, values, length


def train(arm, seed, log, checkpoint=None):
    """The setting P2 of the pilot (test 31) for the loop; with a checkpoint path, everything is kept every 250
    updates and taken back after a stop (each update draws its worlds and actions from its own streams)."""
    import torch
    net = model(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    curve, window, start = [], [], 0
    if checkpoint is not None and Path(checkpoint).exists():
        state = torch.load(checkpoint, weights_only=False)
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        curve, window, start = state["curve"], state["window"], state["update"]
        log(f"  {arm} seed {seed}: resumed at update {start}")
    for update in range(start, UPDATES):
        worlds = [np.random.default_rng([W.SEED, WORLD_STREAM, seed, update, i]) for i in range(LIVES_PER_UPDATE)]
        rng = np.random.default_rng([W.SEED, ACTION_STREAM, seed, update])
        decisions, logps, entropies, values, length = episodes(
            net, arm, worlds, lambda p, last: (rng.random(len(p)) >= p).astype(int), grad=True)
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
    log(f"  {arm} seed {seed}: training survival {curve[-1] if curve else None}")
    return net, curve


def survival(net, arm):
    worlds = [W.world_rng(NM.LIFE_STREAM, i) for i in range(NM.LIVES)]
    greedy = lambda p, last: np.where(p > 0.5, 0, np.where(p < 0.5, 1, 1 - last))
    decisions, _, _, _, length = episodes(net, arm, worlds, greedy)
    survived = (length == W.TURNS).tolist()
    agrees = float(np.mean([a == NM.rule(e, n, last) for _, _, a, e, n, last in decisions]))
    return survived, agrees


class Reader:
    """The agent reading a written life: its states and P(R) at each decision, with or without a grafted state."""

    def __init__(self, net, arm):
        import torch
        self.net, self.arm, self.torch = net, arm, torch

    def _step(self, state, last, event):
        torch = self.torch
        with torch.no_grad():
            logits, _, nxt = self.net.step(state if self.arm == "loop" else self.net.initial(1),
                                           torch.tensor([last]), torch.tensor([event]))
        return float(torch.softmax(logits, -1)[0, 0]), nxt

    def read(self, life, upto=None, graft=None):
        """P(R) at each decision of the life up to turn upto, and the state passed by each decision; graft = (j,
        state): the state passed by decision j is replaced."""
        state, last = self.net.initial(1).detach(), NONE
        p, states = {}, {}
        for x in W.decisions(life):
            if upto is not None and x["t"] > upto:
                break
            p[x["t"]], state = self._step(state, last, EVENT0 + x["event"])
            if graft is not None and x["t"] == graft[0]:
                state = graft[1]
            states[x["t"]] = state
            last = R + x["action"]
        return p, states


def measure_grafts(reader, held, grafts):
    own = [reader.read(life) for life in held]
    rows = []
    for g in grafts:
        donor_state = own[g["d"]][1][g["j"]]
        p, _ = reader.read(held[g["r"]], upto=g["t"], graft=(g["j"], donor_state))
        rows.append([g["index"], round(own[g["r"]][0][g["t"]], 7), round(p[g["t"]], 7)])
    return rows


def probe(reader, held, folds=4, ridge=1.0):
    """Published without threshold: how much of E and N (after the event of the turn) a linear reader finds in the
    state passed by each decision, cross-validated by lives."""
    xs, ys, ls = [], [], []
    for i, life in enumerate(held):
        _, states = reader.read(life)
        for x in W.decisions(life):
            xs.append(states[x["t"]][0].numpy())
            ys.append([x["E"], x["N"]])
            ls.append(i)
    X, Y, L = np.array(xs, float), np.array(ys, float), np.array(ls)
    X = np.hstack([X, np.ones((len(X), 1))])
    pred = np.zeros_like(Y)
    for f in range(folds):
        test = L % folds == f
        A = X[~test]
        w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ Y[~test])
        pred[test] = X[test] @ w
    r2 = 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    return {"E": float(r2[0]), "N": float(r2[1])}


def checks(net_loop, net_cut, held):
    """Seed 0. The cut: under "cut", changing the event of an earlier turn does not change P(R) later (4 cases); the
    self graft: the recipient's own state grafted back changes nothing (4 cases)."""
    cut, selfs = Reader(net_cut, "cut"), Reader(net_loop, "loop")
    gaps_cut, gaps_self = [], []
    for life in [l for l in held if len(W.decisions(l)) >= 6][:4]:
        ds = W.decisions(life)
        changed = json.loads(json.dumps(life))
        changed["turns"][1]["event"] = (life["turns"][1]["event"] + 1) % len(W.EVENTS)
        a, _ = cut.read(life)
        b, _ = cut.read(changed)
        gaps_cut.append(max(abs(a[x["t"]] - b[x["t"]]) for x in ds[2:]))
        j, t = ds[2]["t"], ds[4]["t"]
        own, states = selfs.read(life)
        p, _ = selfs.read(life, upto=t, graft=(j, states[j]))
        gaps_self.append(abs(p[t] - own[t]))
    return {"cut_gap": float(max(gaps_cut)), "self_graft_gap": float(max(gaps_self)), "threads": 1}


def run(a):
    import torch
    torch.set_num_threads(1)
    log = lambda m: print(m, flush=True)
    root = Path(a.out)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    held = lives()
    grafts = draw(held)
    for seed in a.seeds:
        nets = {}
        for arm in a.arms:
            path = root / "runs" / f"{arm}-{seed}.json"
            if path.exists() and not (seed == 0 and not (root / "setup.json").exists()):
                continue
            net, curve = train(arm, seed, log, checkpoint=root / "checkpoints" / f"{arm}-{seed}.pt")
            net.eval()
            nets[arm] = net
            survived, agrees = survival(net, arm)
            reader = Reader(net, arm)
            path.write_text(json.dumps({"arm": arm, "seed": seed, "survival": float(np.mean(survived)),
                                        "survived": survived, "agrees": agrees, "curve": curve,
                                        "probe": probe(reader, held),
                                        "grafts": measure_grafts(reader, held, grafts)}) + "\n")
            log(f"{arm} seed {seed}: survival {np.mean(survived):.3f}, agrees with the needs rule {agrees:.3f}")
        if seed == 0 and set(nets) == set(ARMS) and not (root / "setup.json").exists():
            (root / "setup.json").write_text(json.dumps(checks(nets["loop"], nets["cut"], held), indent=1) + "\n")


def gather(root):
    out = Path(root)
    runs = {arm: {} for arm in ARMS}
    for path in sorted((out / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        runs[r["arm"]][str(r["seed"])] = r
    setup = json.loads((out / "setup.json").read_text())
    result = verdicts(runs, draw(lives()), setup, TS.event_rule_survival())
    return json.loads(json.dumps(dict(result, setup=setup)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
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
