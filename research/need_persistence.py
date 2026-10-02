"""Is the state of a turn taken up at the next turn? (docs/LLM_NEED_PERSISTENCE_PROTOCOL.md): verdicts PERS1, PERS2.

No learning and no new life: the recorded test lives of three agents are read again in torch on CPU. A context is a
decision at turn t + 1 whose turn t is a decision too, both needs of t + 1 at 3 or more. The text of the life up to
"Choix :" of turn t + 1 (and of turn t + 2 when it exists) is read once per condition; the condition acts only on the
output of the agent's block on the three tokens " Cho" "ix" " :" of turn t: none, the lesion of the plane (d_E, d_N)
(coordinates replaced by their means, research/need_necessity.py), the lesion of the random plane, the push -4 d_E and
three random pushes of the same norm per token. The text before turn t's three tokens is the same for every condition
and no condition reaches it: it is run once, kept in a key-value cache, and the rest of the text is run as one batch of
the seven conditions (checked against reading the whole text at once). Resumable. The verdicts are numpy only;
reading needs torch.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_verdicts as V
from . import need_causal as C

K = 3  # " Cho" "ix" " :"
UNITS = 4
EFFECT = 0.03
CONTEXTS = 600
LATER = 300  # contexts read at t + 2 too, published without threshold
MIN_CONTEXTS = 300
LEVEL = 3
REPLICA_TOLERANCE = 0.02
CACHE_TOLERANCE = 1e-4  # the cached batch against the whole text read at once, one condition at a time
CACHE_CHECKS = 4
STREAM = 33
RANDOM = 3
CONDITIONS = ("none", "lesion", "lesion_random", "push") + tuple(f"random{i}" for i in range(RANDOM))
ROOT = "artifacts/llm-need/persistence"
AGENTS = {
    "first": {"adapter": "artifacts/llm-need/final/report/adapters-final",
              "directions": "artifacts/llm-need/reader/test", "lesion": "artifacts/llm-need/reader/necessity",
              "lives": "artifacts/llm-need/reader/test/lives-test.jsonl.gz"},
    "second": {"adapter": "artifacts/llm-need/r1/final/report/adapters-final",
               "directions": "artifacts/llm-need/r1/reader/test", "lesion": "artifacts/llm-need/r1/necessity",
               "lives": "artifacts/llm-need/r1/reader/test/lives-test.jsonl.gz"},
    "two": {"adapter": "artifacts/llm-need/two/need-5-8/adapters-need-8",
            "directions": "artifacts/llm-need/locate-two/slice-8/measure/two/act",
            "lesion": "artifacts/llm-need/locate-two/slice-8/measure/two/lesion",
            "lives": "artifacts/llm-need/locate-two/slice-8/measure/two/act/lives-test.jsonl.gz"},
}


def contexts(lives, limit=CONTEXTS, later=LATER):
    """The first `limit` contexts in the order of the lives: (life, t) with t and t + 1 decisions and both needs of
    t + 1 at LEVEL or more; the first `later` of them whose t + 2 is a decision are read at t + 2 too."""
    out, read_later = [], 0
    for index, life in enumerate(lives):
        ds = {x["t"]: x for x in W.decisions(life)}
        for t in sorted(ds):
            nxt = ds.get(t + 1)
            if nxt is None or nxt["E"] < LEVEL or nxt["N"] < LEVEL:
                continue
            two = t + 2 in ds and read_later < later
            read_later += two
            out.append({"life": index, "t": t, "later": bool(two)})
            if len(out) == limit:
                return out
    return out


def boot(values, lives, draws=V.DRAWS, seed=0):
    """Mean of per-context values and its 95% interval when lives are resampled."""
    values = np.asarray(values, float)
    ids, inverse = np.unique(np.asarray(lives), return_inverse=True)
    sums = np.bincount(inverse, values, len(ids))
    counts = np.bincount(inverse, minlength=len(ids)).astype(float)
    idx = np.random.default_rng(seed).integers(len(ids), size=(draws, len(ids)))
    b = sums[idx].sum(1) / counts[idx].sum(1)
    return {"mean": float(values.mean()), "low": float(np.quantile(b, 0.025)), "high": float(np.quantile(b, 0.975)),
            "contexts": len(values)}


def effects(rows, key):
    """The effects of the conditions on P(R) at the turn read (key "p_R": t + 1, "later": t + 2)."""
    rows = [r for r in rows if r.get(key)]
    life = [r["life"] for r in rows]
    p = {c: np.array([r[key][c] for r in rows]) for c in CONDITIONS}
    return {"lesion": boot(np.abs(p["lesion"] - p["none"]), life),
            "lesion_random": boot(np.abs(p["lesion_random"] - p["none"]), life),
            "push": boot(p["push"] - p["none"], life),
            "random": boot(np.mean([np.abs(p[f"random{i}"] - p["none"]) for i in range(RANDOM)], axis=0), life)}


def agent_verdicts(rows):
    """rows: one per context, {"life", "t", "p_R": {condition: P(R) at t + 1}, "mass", "recorded",
    "later": {condition: P(R) at t + 2} or None, "same_turn": the push at t + 1 itself (recorded) or None}."""
    passes = lambda d: d["mean"] >= EFFECT and d["low"] > 0
    within = lambda x, ref: abs(x["mean"]) <= V.SHARE * ref["mean"]
    now = effects(rows, "p_R")
    gap = float(np.mean([abs(r["p_R"]["none"] - r["recorded"]) for r in rows]))
    mass = float(np.mean([r["mass"] for r in rows]))
    out = {"PERS1": bool(passes(now["lesion"]) and within(now["lesion_random"], now["lesion"])),
           "PERS2": bool(passes(now["push"]) and within(now["random"], now["push"])),
           "valid": bool(gap <= REPLICA_TOLERANCE and len(rows) >= MIN_CONTEXTS and mass >= 0.5)}
    same = [r for r in rows if r.get("same_turn") is not None]
    values = {"t_plus_1": now, "execution_gap": gap, "mass": mass, "contexts": len(rows),
              "t_plus_2": effects(rows, "later") if any(r.get("later") for r in rows) else None,
              "same_turn": {"contexts": len(same),
                            "push_at_t_plus_1": float(np.mean([r["same_turn"] for r in same])) if same else None,
                            "push_at_t": float(np.mean([r["p_R"]["push"] - r["p_R"]["none"] for r in same]))
                            if same else None}}
    return {"verdicts": out, "values": values}


def verdicts(measured):
    """measured: {agent: its rows}. Global: PERS1 and PERS2 (valid) for the first agent and one of the two others."""
    per = {n: agent_verdicts(measured[n]) for n in AGENTS if n in measured}
    out = {f"{k}_{n}": r["verdicts"][k] for n, r in per.items() for k in ("PERS1", "PERS2", "valid")}
    ok = lambda n: n in per and all(per[n]["verdicts"].values())
    out["global"] = bool(len(per) == len(AGENTS) and ok("first") and (ok("second") or ok("two")))
    return {"verdicts": out, "values": {n: r["values"] for n, r in per.items()}}


def random_pushes(d_e, massive):
    """RANDOM pushes, the i-th from the stream [SEED, STREAM, i], each of the norm of UNITS d_E per token, zero on the
    massive dimensions."""
    return [C.random_like(np.random.default_rng([W.SEED, STREAM, i]), UNITS * d_e, 1, massive)[0]
            for i in range(RANDOM)]


def text_upto(life, t):
    """The text of the life up to "Choix :" of turn t."""
    turn = [x for x in life["turns"] if x["t"] == t][0]
    return W.life_text(life["turns"], upto=t) + W.choice_line(t, turn["event"], turn.get("other"))


# ----------------------------------------------------------------------------------------------------- torch part

class Reader:
    """The acting agent merged (research/need_torch.py), with a hook on its block acting, for each sequence of a
    batch, on positions `where` of the piece being run."""

    def __init__(self, adapter, block):
        import torch
        from transformers import DynamicCache
        from .need_torch import Agent
        self.torch, self.Cache = torch, DynamicCache
        base = Agent(adapter)
        self.model, self.enc, self.R, self.M = base.model, base.enc, base.R, base.M
        self.choice = self.enc(" Choix :")
        if len(self.choice) != K:
            raise RuntimeError("\" Choix :\" is not three tokens")
        self.where = self.ops = None
        self.cache, self.ids = None, []
        self.model.model.layers[block].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if self.where is None:
            return out
        torch = self.torch
        h = (out[0] if isinstance(out, tuple) else out).clone()
        a, b = self.where
        for i, op in enumerate(self.ops):
            if op is None:
                continue
            kind, value = op
            tail = h[i, a:b]
            if kind == "plane":
                u, mean = value
                tail = tail - torch.einsum("tk,tdk->td", torch.einsum("td,tdk->tk", tail, u) - mean, u)
            else:
                tail = tail + value
            h[i, a:b] = tail
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def _forward(self, ids, cache, ops, where):
        torch = self.torch
        self.where, self.ops = where, ops
        try:
            with torch.no_grad():
                logits = self.model(torch.tensor([ids] * len(ops)), past_key_values=cache, use_cache=True).logits
        finally:
            self.where = self.ops = None
        return logits

    def probabilities(self, logits, reads):
        p = self.torch.softmax(logits[:, reads].float(), -1)
        r, m = p[..., self.R], p[..., self.M]
        return (r / (r + m)).numpy(), (r + m).numpy()

    def prefix(self, ids):
        """The cache of the life up to ids (no hook), extended from the one kept when it is its beginning."""
        if self.cache is None or ids[:len(self.ids)] != self.ids or len(ids) < len(self.ids):
            self.cache, self.ids = self.Cache(), []
        if len(ids) > len(self.ids):
            self._forward(ids[len(self.ids):], self.cache, [None], None)
            self.ids = list(ids)
        return self.cache

    def read(self, ids, start, reads, ops):
        """P(R) and mass R+M at the positions `reads` of ids, once per op (as one batch), each op acting on the K
        tokens from `start`; the text before `start` is taken from the cache."""
        cache = copy.deepcopy(self.prefix(ids[:start]))
        cache.batch_repeat_interleave(len(ops))
        logits = self._forward(ids[start:], cache, ops, (0, K))
        return self.probabilities(logits, [r - start for r in reads])

    def read_whole(self, ids, start, reads, op):
        """The same read, the whole text at once, without the cache (the check of the cached batch)."""
        logits = self._forward(ids, self.Cache(), [op], (start, start + K))
        return self.probabilities(logits, reads)


def positions(reader, life, t, later):
    """The tokens of the text up to "Choix :" of t + 1 (or t + 2), the start of turn t's " Choix :" in them and the
    positions where P(R) is read at t + 1 (and t + 2); each piece checked to be cut as in the whole text."""
    texts = [text_upto(life, t), text_upto(life, t + 1)] + ([text_upto(life, t + 2)] if later else [])
    ids = reader.enc(texts[-1])
    ends = []
    for text in texts:
        piece = reader.enc(text)
        if ids[:len(piece)] != piece:
            raise RuntimeError(f"the text up to turn t is cut differently (t = {t})")
        ends.append(len(piece))
    if ids[ends[0] - K:ends[0]] != reader.choice:
        raise RuntimeError("the tokens of turn t are not those of \" Choix :\"")
    return ids, ends[0] - K, [e - 1 for e in ends[1:]]


def ops_of(setup):
    T = lambda x: setup["torch"].tensor(np.asarray(x, np.float32))
    ops = {"none": None, "lesion": ("plane", (T(setup["planes"]), T(setup["means"]))),
           "lesion_random": ("plane", (T(setup["random_planes"]), T(setup["random_means"]))),
           "push": ("add", T(-UNITS * setup["d_E"]))}
    ops.update({f"random{i}": ("add", T(v)) for i, v in enumerate(setup["random"])})
    return [ops[c] for c in CONDITIONS]


def measure(a):
    import torch
    torch.set_num_threads(a.threads)
    log = lambda m: print(m, flush=True)
    spec = AGENTS[a.agent]
    out = Path(a.out) / a.agent
    out.mkdir(parents=True, exist_ok=True)
    direction = json.loads((Path(spec["directions"]) / "direction.json").read_text())
    saved = np.load(Path(spec["lesion"]) / "lesion.npz")
    setup = {k: saved[k] for k in ("planes", "means", "random_planes", "random_means")}
    setup["d_E"] = np.array(direction["d_E"])
    setup["random"] = random_pushes(setup["d_E"], direction["massive_dims"])
    setup["torch"] = torch
    ops = ops_of(setup)
    reader = Reader(spec["adapter"], direction["block"])
    lives = W.read_jsonl(spec["lives"])
    chosen = contexts(lives)
    info_file = out / "setup.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    if "cache_gap" not in info:
        gaps = []
        for c in chosen[:CACHE_CHECKS]:
            ids, start, reads = positions(reader, lives[c["life"]], c["t"], c["later"])
            batch = reader.read(ids, start, reads, ops)[0]
            for i, op in enumerate(ops):
                gaps += list(np.abs(reader.read_whole(ids, start, reads, op)[0][0] - batch[i]))
        info = {"agent": a.agent, "block": direction["block"], "massive_dims": direction["massive_dims"],
                "contexts": len(chosen), "later": sum(c["later"] for c in chosen), "cache_gap": float(max(gaps))}
        info_file.write_text(json.dumps(info, indent=1) + "\n")
    log(json.dumps(info))
    if info["cache_gap"] > CACHE_TOLERANCE:
        raise SystemExit("the cached batch differs from reading the whole text")
    final, partial = out / "reads.jsonl.gz", out / "partial-reads.jsonl.gz"
    if final.exists():
        log("already measured")
        return
    rows = W.read_jsonl(partial) if partial.exists() else []
    if rows:
        log(f"resumed after {len(rows)} contexts")
    for k, c in enumerate(chosen[len(rows):], start=len(rows)):
        life = lives[c["life"]]
        ids, start, reads = positions(reader, life, c["t"], c["later"])
        p, mass = reader.read(ids, start, reads, ops)
        nxt = [x for x in life["turns"] if x["t"] == c["t"] + 1][0]
        injection = nxt.get("extra", {}).get("injection")
        rows.append({"life": c["life"], "t": c["t"], "recorded": nxt["p_R"], "mass": round(float(mass[0, 0]), 6),
                     "p_R": {n: round(float(p[i, 0]), 6) for i, n in enumerate(CONDITIONS)},
                     "later": {n: round(float(p[i, 1]), 6) for i, n in enumerate(CONDITIONS)} if c["later"] else None,
                     "same_turn": round(injection["E"]["p_R"] - injection["none"]["p_R"], 6) if injection else None})
        if (k + 1) % 20 == 0:
            W.write_jsonl(partial, rows)
        if (k + 1) % 100 == 0:
            log(f"  {k + 1} contexts")
    W.write_jsonl(final, rows)
    if partial.exists():
        partial.unlink()
    log(json.dumps(agent_verdicts(rows)["verdicts"]))


def gather(root):
    measured = {n: W.read_jsonl(Path(root) / n / "reads.jsonl.gz") for n in AGENTS
                if (Path(root) / n / "reads.jsonl.gz").exists()}
    result = verdicts(measured)
    for n in result["values"]:
        result["values"][n]["setup"] = json.loads((Path(root) / n / "setup.json").read_text())
    return json.loads(json.dumps(result))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--agent", choices=sorted(AGENTS), required=True)
    r.add_argument("--threads", type=int, default=4)
    sub.add_parser("write")
    sub.add_parser("verdicts")
    for s in sub.choices.values():
        s.add_argument("--out", default=ROOT, help="the reads go to <out>/<agent>")
    a = parser.parse_args(argv)
    if a.command == "run":
        measure(a)
        return
    result = gather(a.out)
    path = Path(a.out) / "verdicts.json"
    if a.command == "write":
        path.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result["verdicts"]))
    else:
        for n in result["values"]:  # the contexts are those the protocol picks from the recorded lives
            rows = W.read_jsonl(Path(a.out) / n / "reads.jsonl.gz")
            picked = [{"life": c["life"], "t": c["t"], "later": c["later"]}
                      for c in contexts(W.read_jsonl(AGENTS[n]["lives"]))]
            if [{"life": r["life"], "t": r["t"], "later": r["later"] is not None} for r in rows] != picked:
                raise SystemExit(f"{n}: the reads are not those of the protocol's contexts")
        differs = result != json.loads(path.read_text())
        print("differs:", "yes" if differs else "none")
        if differs:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
