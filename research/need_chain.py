"""Does the need pass from one carried state to the next? (docs/LLM_NEED_CHAIN_PROTOCOL.md): verdict CHAIN of test 24.

The agent A of test 22 (research/need_memory_no_leak.py, after 4000 iterations) under its mask; the graft of test 23
(research/need_carried_content.py): the keys and values of turn i's carried tokens, in every layer, from a donor life
with other needs at i (and a rule effect at t not zero). P(R) at t = i + h is read four ways: own (no graft); total (the
graft, the later turns computed with it); direct (the total, with the carried tokens of turns i + 1 to t - 1, the
mediators, given back their own keys and values); mediators only (own, with the mediators given their keys and values
of the total). The chain is the total minus the direct. The draw and the verdicts are numpy only; the reads need torch.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_memory as NM
from . import need_memory_no_leak as NL22
from .need_carried_content import rule_effect, _boot_diff
from .need_persistence import boot

STREAM = 51
GAPS = (2, 3)
PER_GAP = 3  # turns t drawn per recipient life and gap
TOTAL = 0.10  # the prior condition on the total effect
HALF = 0.5  # CHAIN
MIN_COUNT = 150  # grafts per gap
EXTEND_TOLERANCE, SELF_TOLERANCE = 1e-4, 1e-6
ROOT = "artifacts/llm-need/chain"
DRAWS = 10_000
READS = ("own", "total", "direct", "mediators")


def draw(held):
    """The grafts of the protocol: recipients in order; per gap h up to PER_GAP turns t with the recipient alive from
    i = t - h to t; per t one donor alive at i with the same written action at i, other needs at i and a rule effect
    at t not zero."""
    rng = np.random.default_rng([W.SEED, STREAM, 0])
    at = [{x["t"]: x for x in W.decisions(life)} for life in held]
    out = []
    for r, life in enumerate(held):
        for h in GAPS:
            turns = [t for t in sorted(at[r]) if t - h >= 1 and all(k in at[r] for k in range(t - h, t + 1))]
            for k in sorted(rng.permutation(len(turns))[:PER_GAP]):
                t = turns[k]
                i = t - h
                mine = at[r][i]
                donors = []
                for d in range(len(held)):
                    x = at[d].get(i)
                    if d == r or x is None or x["action"] != mine["action"] or (x["E"], x["N"]) == (mine["E"], mine["N"]):
                        continue
                    effect = rule_effect(life, held[d], i, t)
                    if effect is None or effect == 0:
                        continue
                    donors.append((d, effect))
                if donors:
                    d, effect = donors[int(rng.integers(len(donors)))]
                    out.append({"r": r, "d": d, "i": i, "t": t, "h": h, "rule_effect": effect})
    return out


def _ratio(x, y, lives, draws=DRAWS, seed=0):
    """mean(x) / mean(y) and its 95% interval, the recipient lives resampled."""
    ids, inverse = np.unique(np.asarray(lives), return_inverse=True)
    sx, sy = np.bincount(inverse, x, len(ids)), np.bincount(inverse, y, len(ids))
    idx = np.random.default_rng(seed).integers(len(ids), size=(draws, len(ids)))
    b = sx[idx].sum(1) / sy[idx].sum(1)
    return {"mean": float(np.mean(x) / np.mean(y)), "low": float(np.quantile(b, 0.025)),
            "high": float(np.quantile(b, 0.975))}


def measures(rows, select=None):
    rows = [r for r in rows if select is None or select(r)]
    p = {k: np.array([r["p_R"][k] for r in rows]) for k in READS}
    effect = np.array([r["rule_effect"] for r in rows])
    life = np.array([r["r"] for r in rows])
    total, direct = (p["total"] - p["own"]) * effect, (p["direct"] - p["own"]) * effect
    chain, mediators = (p["total"] - p["direct"]) * effect, (p["mediators"] - p["own"]) * effect
    out = {"total": boot(total, life), "direct": boot(direct, life), "chain": boot(chain, life),
           "mediators": boot(mediators, life), "count": len(rows)}
    out["chain_share"] = _ratio(chain, total, life)
    out["chain_over_half_total"] = _boot_diff(chain, life, HALF * total, life)
    return out


def verdicts(rows, setup):
    m = measures(rows)
    m["by_gap"] = {str(h): measures(rows, lambda r, h=h: r["h"] == h) for h in GAPS}
    prior = bool(m["total"]["mean"] >= TOTAL and m["total"]["low"] > 0)
    out = {"prior": prior,
           "CHAIN": bool(prior and m["chain"]["mean"] >= HALF * m["total"]["mean"] and m["chain"]["low"] > 0)}
    out["valid"] = bool(setup["extend_gap"] <= EXTEND_TOLERANCE and setup["self_graft_gap"] <= SELF_TOLERANCE
                        and setup["positions_same"] and all(m["by_gap"][str(h)]["count"] >= MIN_COUNT for h in GAPS))
    out["global"] = bool(out["valid"] and out["CHAIN"])
    return {"verdicts": out, "values": m}


# ----------------------------------------------------------------------------------------------------- torch part

class Chain:
    """A's grafter of test 23, with the cache carried on from turn i to turn t - 1 and the mediators swapped."""

    def __init__(self):
        from .need_carried_content import Grafter
        self.g = Grafter("route")
        self.agent, self.torch = self.g.agent, self.g.torch

    def carried(self, turns, t):
        """Per turn k < t, the positions of its carried tokens in the decision text at t; the end of each turn."""
        from . import need_lora as NL
        _, turn_of, carried = NL.carry_layout(NL22.decision_text(turns, t), self.agent.tok)
        positions, ends = {}, {}
        for p, (k, c) in enumerate(zip(turn_of, carried)):
            if k >= 0 and k + 1 < t:
                ends[k + 1] = p + 1
                if c:
                    positions.setdefault(k + 1, []).append(p)
        return positions, ends

    def extend(self, cache, ids, mask, start, stop):
        torch = self.torch
        with torch.no_grad():
            self.agent.model(torch.tensor([ids[start:stop]]), past_key_values=cache, use_cache=True,
                             attention_mask=torch.tensor(mask[start:stop, :stop])[None, None])

    @staticmethod
    def take(cache, positions):
        return [(layer.keys[:, :, positions, :].clone(), layer.values[:, :, positions, :].clone()) for layer in cache.layers]

    @staticmethod
    def put(cache, positions, kv):
        for layer, (k, v) in zip(cache.layers, kv):
            layer.keys[:, :, positions, :] = k
            layer.values[:, :, positions, :] = v

    def read(self, recipient, donor, i, t):
        """P(R) at t: own, total, direct, mediators only."""
        turns = recipient["turns"]
        positions, ends = self.carried(turns, t)
        mediators = [p for k in range(i + 1, t) for p in positions[k]]
        own_cache, ids, mask, end, _ = self.g.prefix(turns, t - 1, t)
        own_kv = self.take(own_cache, mediators)
        spare = copy.deepcopy(own_cache)
        own = self.g.finish(own_cache, ids, mask, end)
        cache, ids2, mask2, end_i, at_i = self.g.prefix(turns, i, t)
        dcache, _, _, dend, dpositions = self.g.prefix(donor["turns"], i)
        if at_i != positions[i] or dpositions != at_i or dend != end_i or ids2 != ids or ends[t - 1] != end:
            raise RuntimeError("the carried tokens are not at the same positions")
        self.put(cache, at_i, self.take(dcache, at_i))
        self.extend(cache, ids, mask, end_i, end)
        total_kv = self.take(cache, mediators)
        direct_cache = copy.deepcopy(cache)
        total = self.g.finish(cache, ids, mask, end)
        self.put(direct_cache, mediators, own_kv)
        direct = self.g.finish(direct_cache, ids, mask, end)
        self.put(spare, mediators, total_kv)
        only = self.g.finish(spare, ids, mask, end)
        return {"own": own, "total": total, "direct": direct, "mediators": only}


def checks(chain, held, log):
    """Prefix to i, carried on to t - 1, then finished, against the whole text; the self graft against own."""
    extend, selfs = [], []
    for life in held[:4]:
        ds = W.decisions(life)
        i, t = ds[3]["t"], ds[6]["t"]
        cache, ids, mask, end_i, _ = chain.g.prefix(life["turns"], i, t)
        _, ends = chain.carried(life["turns"], t)
        chain.extend(cache, ids, mask, end_i, ends[t - 1])
        p = chain.g.finish(cache, ids, mask, ends[t - 1])
        extend.append(abs(p - chain.agent.whole(NL22.decision_text(life["turns"], t))[0]))
        read = chain.read(life, life, i, t)
        selfs.append(max(abs(read[k] - read["own"]) for k in READS))
    out = {"extend_gap": float(max(extend)), "self_graft_gap": float(max(selfs)), "positions_same": True}
    log(json.dumps(out))
    return out


def measure(a):
    import torch
    torch.set_num_threads(a.threads)
    log = lambda m: print(m, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    held = NM.teacher_lives(NM.HELD_STREAM, NM.HELD_LIVES)
    grafts = draw(held)
    path, partial = out / "grafts.jsonl.gz", out / "partial-grafts.jsonl.gz"
    if not path.exists():
        chain = Chain()
        setup = out / "setup.json"
        if not setup.exists():
            setup.write_text(json.dumps(checks(chain, held, log), indent=1) + "\n")
        rows = W.read_jsonl(partial) if partial.exists() else []
        for k in range(len(rows), len(grafts)):
            g = grafts[k]
            read = chain.read(held[g["r"]], held[g["d"]], g["i"], g["t"])
            rows.append(dict(g, p_R={key: round(read[key], 6) for key in READS}))
            if (k + 1) % 50 == 0:
                W.write_jsonl(partial, rows)
                log(f"  {k + 1} grafts")
        W.write_jsonl(path, rows)
        if partial.exists():
            partial.unlink()
        log("grafts done")
    log(json.dumps(gather(a.out)["verdicts"]))


def gather(root):
    out = Path(root)
    grafts = draw(NM.teacher_lives(NM.HELD_STREAM, NM.HELD_LIVES))
    measured = W.read_jsonl(out / "grafts.jsonl.gz")
    fields = ("r", "d", "i", "t", "h", "rule_effect")
    if [{k: m[k] for k in fields} for m in measured] != [{k: g[k] for k in fields} for g in grafts]:
        raise SystemExit("the grafts are not those the protocol draws")
    setup = json.loads((out / "setup.json").read_text())
    return json.loads(json.dumps(dict(verdicts(measured, setup), setup=setup)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=4)
    sub.add_parser("write")
    sub.add_parser("verdicts")
    for s in sub.choices.values():
        s.add_argument("--out", default=ROOT)
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
        differs = result != json.loads(path.read_text())
        print("differs:", "yes" if differs else "none")
        if differs:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
