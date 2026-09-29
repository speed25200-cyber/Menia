"""The need that decides, where it is gathered (docs/LLM_NEED_CAUSAL_PROTOCOL.md).

The final agent of the first need test, replayed in torch on CPU (research/need_torch.py), lives new worlds. Causal
directions of energy and food at block 12 are measured on the last four tokens of the turn line ("." "Cho" "ix" ":")
from pairs of lives identical but for one past event; they are then injected into the turn line, and removed from it
during whole lives. The pair selection, the fit and the verdicts are numpy only; living needs torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_verdicts as V

BLOCK = 12
TAIL = 4  # the last tokens of a turn line: "." " Cho" "ix" " :"
UNITS = 4
PAIRS_PER_DECISION = 3
MAX_PAIRS = 1500
MIN_CALME_COURS = 300
MASSIVE = 10
REPLICA_DECISIONS = 60
REPLICA_TOLERANCE = 0.02
STREAMS = {"direction": 10, "test": 11, "pairs": 12, "random": 13}
# swaps of equal token length: (from event, to event) -> change of E and N at the swapped turn
SWAPS = {(0, 1): (-2, 0), (0, 5): (-1, -1), (3, 2): (-1, -2)}


def replay(life, j, event, t):
    """Levels (E, N) at decision t, after its event, when the event of turn j is replaced, the choices being the same;
    None if the agent would have gone out on the way (amendment 2 of the protocol)."""
    ds = {x["t"]: x for x in W.decisions(life)}
    first = ds[j]
    old = W.EVENTS[first["event"]]
    e = first["E"] - old[2] + W.EVENTS[event][2]
    n = first["N"] - old[3] + W.EVENTS[event][3]
    for x in range(j, t):
        if e <= 0 or n <= 0:
            return None
        e, n = W.after(e, n, ds[x]["action"])
        nxt = W.EVENTS[ds[x + 1]["event"]]
        e, n = e + nxt[2], n + nxt[3]
    return (e, n) if e > 0 and n > 0 else None


def candidate_pairs(life, turn):
    """Past events j whose swap (at equal token length) changes the needs at the decision, the agent staying alive:
    the changes (dE, dN) are those of the exact replay of the life with the same choices."""
    out = []
    for past in W.decisions(life):
        j = past["t"]
        if j >= turn["t"]:
            break
        for (a, b) in SWAPS:
            if past["event"] != a:
                continue
            levels = replay(life, j, b, turn["t"])
            if levels is None:
                continue
            de, dn = levels[0] - turn["E"], levels[1] - turn["N"]
            if de or dn:
                out.append({"j": j, "swap": [a, b], "dE": de, "dN": dn})
    return out


def choose_pairs(lives):
    rng = np.random.default_rng([W.SEED, STREAMS["pairs"]])
    pairs = []
    for index, life in enumerate(lives):
        for turn in W.decisions(life):
            cands = candidate_pairs(life, turn)
            for k in rng.permutation(len(cands))[:PAIRS_PER_DECISION]:
                pairs.append(dict(cands[k], life=index, t=turn["t"]))
    order = rng.permutation(len(pairs))
    return [pairs[i] for i in order[:MAX_PAIRS]]


def swapped(life, pair):
    return [dict(x, event=pair["swap"][1]) if x["t"] == pair["j"] else x for x in life["turns"]]


def decision_text(turns, t):
    turn = [x for x in turns if x["t"] == t][0]
    return W.life_text(turns, upto=t) + W.choice_line(t, turn["event"])


def fit(pairs, diffs):
    """Per tail token, least squares without constant of the activation change on (dE, dN)."""
    x = np.array([[p["dE"], p["dN"]] for p in pairs], float)
    d_e, d_n = [], []
    for k in range(diffs.shape[1]):
        coef, *_ = np.linalg.lstsq(x, diffs[:, k, :].astype(np.float64), rcond=None)
        d_e.append(coef[0])
        d_n.append(coef[1])
    return np.array(d_e), np.array(d_n)


def massive_dims(states):
    """Dimensions whose mean |activation| over the direction decisions exceeds MASSIVE times the median dimension's,
    on any tail token (amendment 1 of the protocol)."""
    m = np.abs(states).mean(axis=0)
    return sorted({int(d) for k in range(m.shape[0]) for d in np.where(m[k] > MASSIVE * np.median(m[k]))[0]})


def planes(d_e, d_n):
    return np.stack([np.linalg.qr(np.stack([e, n], axis=1))[0] for e, n in zip(d_e, d_n)])


def random_like(rng, vectors, count, excluded=()):
    """count sets of random vectors, one per tail token, each with the norm of the given vector of that token, zero
    on the excluded dimensions."""
    out = []
    for _ in range(count):
        g = rng.standard_normal(vectors.shape)
        g[:, list(excluded)] = 0
        out.append(g / np.linalg.norm(g, axis=1, keepdims=True) * np.linalg.norm(vectors, axis=1, keepdims=True))
    return out


def verdicts(intact, lesion, lesion_random):
    passes = lambda d: d["mean"] >= V.EFFECT and d["low"] > 0
    within = lambda x, ref: x is not None and abs(x) <= V.SHARE * ref["mean"]
    ctx = V.contexts(intact)
    ia_e, ia_n = V.effect(ctx, "E", "p_R"), V.effect(ctx, "N", "p_R", -1.0)
    rand_e, rand_n = V.random_effect(ctx, "randE", "p_R"), V.random_effect(ctx, "randN", "p_R", -1.0)
    drop, drop_random = V.paired_survival(intact, lesion), V.paired_survival(intact, lesion_random)
    n_ctx = sum(len(c) for c in ctx)
    out = {"IA2": bool(passes(ia_e) and passes(ia_n) and within(rand_e, ia_e) and within(rand_n, ia_n)),
           "LS2": bool(passes(drop) and within(drop_random["mean"], drop)), "contexts": n_ctx}
    out["global"] = bool(out["IA2"] and out["LS2"])
    values = {"energy_low_raises_R": ia_e, "food_low_raises_M": ia_n, "random_R": rand_e, "random_M": rand_n,
              "lesion_drop": drop, "random_lesion_drop": drop_random,
              "survival": {n: float(np.mean([l["survived"] for l in lives])) for n, lives in
                           (("intact", intact), ("lesion", lesion), ("lesion_random", lesion_random))},
              "turns_need_at_most_2": {n: W.low_turns(lives) for n, lives in
                                       (("intact", intact), ("lesion", lesion), ("lesion_random", lesion_random))}}
    return {"verdicts": out, "values": values}


# ----------------------------------------------------------------------------------------------------- torch part

class TorchAgent:
    """start/decide/commit over a key-value cache, with a hook on block BLOCK acting on the last TAIL tokens of the
    piece being run: add (TAIL x D), or replace the component in per-token planes (TAIL x D x 2) by fixed means."""

    def __init__(self, adapter, tail=TAIL):
        import torch
        from transformers import DynamicCache
        from .need_torch import Agent
        self.torch, self.Cache = torch, DynamicCache
        self.base = Agent(adapter)
        self.model, self.enc = self.base.model, self.base.enc
        self.ids_R, self.ids_M = self.base.R, self.base.M
        self.add = self.project = None
        self.capture = False
        self.captured = None
        self.tail = tail
        self.calls = self.resets = 0
        self.model.model.layers[BLOCK].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if self.add is None and self.project is None and not self.capture:
            return out
        h = h.clone()
        tail = h[0, -self.tail:]
        if self.project is not None:
            u, mean = self.project  # (TAIL, D, 2), (TAIL, 2)
            coord = self.torch.einsum("td,tdk->tk", tail, u)
            tail = tail - self.torch.einsum("tk,tdk->td", coord - mean, u)
        if self.add is not None:
            tail = tail + self.add
        h[0, -self.tail:] = tail
        if self.capture:
            self.captured = tail.detach().numpy().copy()
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def _run(self, ids, hooked=True):
        self.calls += 1
        saved = (self.add, self.project, self.capture)
        if not hooked:
            self.add = self.project = None
            self.capture = False
        with self.torch.no_grad():
            logits = self.model(self.torch.tensor([ids]), past_key_values=self.cache, use_cache=True).logits[0, -1]
        self.add, self.project, self.capture = saved
        p = self.torch.softmax(logits.float(), -1)
        return float(p[self.ids_R]), float(p[self.ids_M])

    def _new(self, text):
        full = self.enc(text)
        if full[:len(self.ids)] != self.ids:
            self.resets += 1
            self.cache, self.ids = self.Cache(), []
        return full, full[len(self.ids):]

    def start(self, header):
        self.cache, self.ids, self.text, self.pending = self.Cache(), [], header, ""
        full, new = self._new(header)
        self._run(new, hooked=False)
        self.ids = full

    def _flush(self):
        if self.pending:
            full, new = self._new(self.text + self.pending)
            self._run(new, hooked=False)
            self.text, self.ids, self.pending = self.text + self.pending, full, ""

    def decide(self, line):
        full, new = self._new(self.text + self.pending + line)
        r, m = self._run(new)
        self.text, self.ids, self.pending = self.text + self.pending + line, full, ""
        return r / max(r + m, 1e-12), r + m

    def commit(self, action):
        self.pending += " " + W.ACTIONS[action] + "\n"

    def evaluate(self, line, add):
        """P(R) after the turn line with `add` on its tail; the cache is cropped back."""
        self._flush()
        full, new = self._new(self.text + line)
        self.add = add
        r, m = self._run(new)
        self.add = None
        self.cache.crop(-len(new))  # a negative value removes that many tokens
        return r / max(r + m, 1e-12)


def replica_check(agent, mac_lives):
    """Mean |P(R) torch - P(R) Mac| over the first REPLICA_DECISIONS decisions at turns 3, 10 and 20."""
    gaps = []
    for life in mac_lives:
        for turn in W.decisions(life):
            if turn["t"] in (3, 10, 20) and len(gaps) < REPLICA_DECISIONS:
                p, _, _ = agent.base.p_r(decision_text(life["turns"], turn["t"]))
                gaps.append(abs(p - turn["p_R"]))
    return {"decisions": len(gaps), "mean_gap": float(np.mean(gaps))}


def tail_activations(agent, text):
    """Block-BLOCK outputs of the last TAIL tokens of a full text, without the cache."""
    agent.cache, agent.ids = agent.Cache(), []
    agent.capture = True
    agent._run(agent.enc(text))
    agent.capture = False
    return agent.captured


def run(a):
    import torch
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    agent = TorchAgent(a.adapter)
    log = lambda m: print(m, flush=True)
    replica = replica_check(agent, W.read_jsonl(a.mac_lives))
    log(json.dumps({"replica": replica}))

    class Recorder:
        def __init__(self):
            self.states = []

        def start(self, header):
            agent.start(header)

        def decide(self, line):
            agent.capture = True
            p = agent.decide(line)
            agent.capture = False
            self.states.append(agent.captured.astype(np.float16))
            return p

        def commit(self, action):
            agent.commit(action)

    rec = Recorder()
    lives = [W.play(rec, W.world_rng(STREAMS["direction"], i), W.choice_rng(STREAMS["direction"], i))
             for i in range(a.direction_lives)]
    W.write_jsonl(out / "lives-direction.jsonl.gz", lives)
    log(f"direction lives: survival {np.mean([l['survived'] for l in lives]):.3f}")
    pairs = choose_pairs(lives)
    real_cache, diffs = {}, []
    for k, pair in enumerate(pairs):
        key = (pair["life"], pair["t"])
        if key not in real_cache:
            real_cache[key] = tail_activations(agent, decision_text(lives[pair["life"]]["turns"], pair["t"]))
        cf = tail_activations(agent, decision_text(swapped(lives[pair["life"]], pair), pair["t"]))
        diffs.append((cf - real_cache[key]).astype(np.float32))
        if (k + 1) % 250 == 0:
            log(f"  {k + 1} pairs")
    diffs = np.array(diffs)
    np.savez_compressed(out / "pairs.npz", diffs=diffs.astype(np.float16))
    W.write_jsonl(out / "pairs.jsonl", pairs)
    states = np.array(rec.states, np.float32)
    massive = massive_dims(states)
    kept = diffs.astype(np.float16).astype(np.float32)
    kept[:, :, massive] = 0
    d_e, d_n = fit(pairs, kept)
    u = planes(d_e, d_n)
    mean = np.einsum("ntd,tdk->ntk", states, u).mean(axis=0)
    rng = np.random.default_rng([W.SEED, STREAMS["random"]])
    rand_e, rand_n = random_like(rng, UNITS * d_e, 3, massive), random_like(rng, UNITS * d_n, 3, massive)
    gaussian = [rng.standard_normal((d_e.shape[1], 2)) for _ in range(TAIL)]
    for g in gaussian:
        g[massive] = 0
    rand_plane = np.stack([np.linalg.qr(g)[0] for g in gaussian])
    rand_mean = np.einsum("ntd,tdk->ntk", states, rand_plane).mean(axis=0)
    direction = {"block": BLOCK, "massive_dims": massive, "pairs": len(pairs), "pairs_by_swap": {str(s): sum(p["swap"] == list(s) for p in pairs)
                 for s in SWAPS}, "norm_d_E": np.linalg.norm(d_e, axis=1).tolist(),
                 "norm_d_N": np.linalg.norm(d_n, axis=1).tolist(), "replica": replica}
    (out / "direction.json").write_text(json.dumps(direction, indent=1) + "\n")
    log(json.dumps(direction))
    T = lambda x: torch.tensor(np.asarray(x, np.float32))
    conditions = {"none": None, "E": T(-UNITS * d_e), "N": T(-UNITS * d_n)}
    conditions.update({f"randE{i}": T(v) for i, v in enumerate(rand_e)})
    conditions.update({f"randN{i}": T(v) for i, v in enumerate(rand_n)})

    def at_decision(ag, t, k, e, n):
        if e >= W.HIGH and n >= W.HIGH:
            return {"injection": {name: {"p_R": round(agent.evaluate(W.choice_line(t, k), v), 6)}
                                  for name, v in conditions.items()}}
        return None

    results = {}
    for name, project in (("intact", None), ("lesion", (T(u), T(mean))), ("lesion_random", (T(rand_plane), T(rand_mean)))):
        agent.project = project
        lives = [W.play(agent, W.world_rng(STREAMS["test"], i), W.choice_rng(STREAMS["test"], i),
                        at_decision if name == "intact" else None) for i in range(a.test_lives)]
        agent.project = None
        W.write_jsonl(out / f"lives-test-{name}.jsonl.gz", lives)
        results[name] = lives
        log(f"{name}: survival {np.mean([l['survived'] for l in lives]):.3f}, calls {agent.calls}, resets {agent.resets}")
    result = verdicts(results["intact"], results["lesion"], results["lesion_random"])
    result["valid"] = bool(replica["mean_gap"] <= REPLICA_TOLERANCE and result["verdicts"]["contexts"] >= 100
                           and direction["pairs_by_swap"][str((0, 1))] >= MIN_CALME_COURS)
    (out / "verdicts.json").write_text(json.dumps(json.loads(json.dumps(result)), indent=1) + "\n")
    log(json.dumps(result["verdicts"]))


def check(a):
    out = Path(a.out)
    lives = {n: W.read_jsonl(out / f"lives-test-{n}.jsonl.gz") for n in ("intact", "lesion", "lesion_random")}
    result = verdicts(lives["intact"], lives["lesion"], lives["lesion_random"])
    direction = json.loads((out / "direction.json").read_text())
    result["valid"] = bool(direction["replica"]["mean_gap"] <= REPLICA_TOLERANCE
                           and result["verdicts"]["contexts"] >= 100
                           and direction["pairs_by_swap"][str((0, 1))] >= MIN_CALME_COURS)
    pairs = W.read_jsonl(out / "pairs.jsonl")
    if choose_pairs(W.read_jsonl(out / "lives-direction.jsonl.gz")) != pairs:
        raise SystemExit("the pairs differ from those drawn from the direction lives")
    published = json.loads((out / "verdicts.json").read_text())
    differs = json.loads(json.dumps(result)) != published
    print("differs:", "yes" if differs else "none")
    if differs:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--adapter", default="artifacts/llm-need/final/report/adapters-final")
    r.add_argument("--mac-lives", default="artifacts/llm-need/test-final/test/lives-test-final.jsonl.gz")
    r.add_argument("--direction-lives", type=int, default=128)
    r.add_argument("--test-lives", type=int, default=256)
    r.add_argument("--threads", type=int, default=4)
    c = sub.add_parser("verdicts")
    for s in (r, c):
        s.add_argument("--out", default="artifacts/llm-need/causal")
    a = parser.parse_args(argv)
    {"run": run, "verdicts": check}[a.command](a)


if __name__ == "__main__":
    main()
