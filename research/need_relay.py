"""The relay step by step (docs/LLM_NEED_RELAY_PROTOCOL.md): verdict RELAY of test 27.

The agent A of test 22 and the graft of tests 23 to 25 (research/need_chain.py), at h = t - i = 3 on 128 new teacher
lives. Three versions of the carried state of turn i + 2: complete (turn i grafted, turn i + 1 computed with it), through
i alone (turn i grafted, turn i + 1 given back its values without graft before turn i + 2 is computed), through i + 1
alone (turn i without graft, turn i + 1 taken from the grafted read). Each is put in the read without graft, turns i and
i + 1 left without graft, and P(R) is read at t. RELAY: the state of turn i + 2 gets the graft mostly through turn i + 1.
One thread per process; the grafts may be shared among processes. The draw and the verdicts are numpy only.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_memory as NM
from .need_carried_content import rule_effect, _boot_diff
from .need_persistence import boot

LIVES_STREAM, LIVES = 61, 128
STREAM = 62
GAP = 3
PER_LIFE = 3
PRIOR = 0.05  # the prior condition on the complete effect
HALF = 0.5
MIN_COUNT = 150
EXTEND_TOLERANCE, SELF_TOLERANCE = 1e-4, 1e-6
READS = ("own", "complete", "through_i", "through_next", "total")
ROOT = "artifacts/llm-need/relay"


def lives():
    return NM.teacher_lives(LIVES_STREAM, LIVES)


def draw(held):
    """Recipients in order; up to PER_LIFE turns t with the recipient alive from i = t - 3 to t; per t one donor alive
    at i with the same written action at i, other needs at i and a rule effect at t not zero; each with its index."""
    rng = np.random.default_rng([W.SEED, STREAM, 0])
    at = [{x["t"]: x for x in W.decisions(life)} for life in held]
    out = []
    for r, life in enumerate(held):
        turns = [t for t in sorted(at[r]) if t - GAP >= 1 and all(k in at[r] for k in range(t - GAP, t + 1))]
        for k in sorted(rng.permutation(len(turns))[:PER_LIFE]):
            t = turns[k]
            i = t - GAP
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
                out.append({"index": len(out), "r": r, "d": d, "i": i, "t": t, "rule_effect": effect})
    return out


def verdicts(rows, setup):
    p = {k: np.array([r["p_R"][k] for r in rows]) for k in READS}
    e = np.array([r["rule_effect"] for r in rows])
    life = np.array([r["r"] for r in rows])
    eff = {k: (p[k] - p["own"]) * e for k in READS if k != "own"}
    m = {k: boot(v, life) for k, v in eff.items()}
    m["non_additive"] = boot(eff["complete"] - eff["through_i"] - eff["through_next"], life)
    m["next_over_half_complete"] = _boot_diff(eff["through_next"], life, HALF * eff["complete"], life)
    m["next_over_i"] = _boot_diff(eff["through_next"], life, eff["through_i"], life)
    m["count"], m["recipient_lives"] = len(rows), len(set(life.tolist()))
    prior = bool(m["complete"]["mean"] >= PRIOR and m["complete"]["low"] > 0)
    out = {"prior": prior,
           "RELAY": bool(prior and m["next_over_half_complete"]["low"] > 0 and m["next_over_i"]["low"] > 0)}
    out["valid"] = bool(setup["extend_gap"] <= EXTEND_TOLERANCE and setup["self_graft_gap"] <= SELF_TOLERANCE
                        and setup["positions_same"] and len(rows) >= MIN_COUNT)
    out["global"] = bool(out["valid"] and out["RELAY"])
    return {"verdicts": out, "values": m}


# ----------------------------------------------------------------------------------------------------- torch part

class Relay:
    def __init__(self):
        from .need_chain import Chain
        self.c = Chain()
        self.g = self.c.g

    def read(self, recipient, donor, i, t):
        c, g = self.c, self.g
        turns = recipient["turns"]
        positions, ends = c.carried(turns, t)
        p1, p2 = positions[i + 1], positions[i + 2]
        base, ids, mask, end, _ = g.prefix(turns, t - 1, t)
        if ends[t - 1] != end:
            raise RuntimeError("the turns are not where they should be")
        s1_own = c.take(base, p1)
        own = g.finish(copy.deepcopy(base), ids, mask, end)
        dcache, _, _, dend, dpositions = g.prefix(donor["turns"], i)
        cache, ids2, _, end_i, at_i = g.prefix(turns, i, t)
        if at_i != positions[i] or dpositions != at_i or dend != end_i or ids2 != ids:
            raise RuntimeError("the carried tokens are not at the same positions")
        c.put(cache, at_i, c.take(dcache, at_i))
        c.extend(cache, ids, mask, end_i, ends[i + 1])
        s1_graft = c.take(cache, p1)
        through_i = copy.deepcopy(cache)
        c.put(through_i, p1, s1_own)
        c.extend(through_i, ids, mask, ends[i + 1], ends[i + 2])
        s2_through_i = c.take(through_i, p2)
        c.extend(cache, ids, mask, ends[i + 1], ends[i + 2])
        s2_complete = c.take(cache, p2)
        total = g.finish(cache, ids, mask, end)
        nxt, _, _, end_1, _ = g.prefix(turns, i + 1, t)
        c.put(nxt, p1, s1_graft)
        c.extend(nxt, ids, mask, end_1, ends[i + 2])
        s2_through_next = c.take(nxt, p2)
        out = {"own": own, "total": total}
        for key, state in (("complete", s2_complete), ("through_i", s2_through_i), ("through_next", s2_through_next)):
            cc = copy.deepcopy(base)
            c.put(cc, p2, state)
            out[key] = g.finish(cc, ids, mask, end)
        return out


def checks(relay, held, log):
    from . import need_memory_no_leak as NL22
    extend, selfs = [], []
    for life in held[:4]:
        ds = W.decisions(life)
        i, t = ds[3]["t"], ds[3]["t"] + GAP
        cache, ids, mask, end_i, _ = relay.g.prefix(life["turns"], i, t)
        _, ends = relay.c.carried(life["turns"], t)
        relay.c.extend(cache, ids, mask, end_i, ends[t - 1])
        p = relay.g.finish(cache, ids, mask, ends[t - 1])
        extend.append(abs(p - relay.c.agent.whole(NL22.decision_text(life["turns"], t))[0]))
        read = relay.read(life, life, i, t)
        selfs.append(max(abs(read[k] - read["own"]) for k in READS))
    out = {"extend_gap": float(max(extend)), "self_graft_gap": float(max(selfs)), "positions_same": True, "threads": 1}
    log(json.dumps(out))
    return out


def measure(a):
    import torch
    torch.set_num_threads(1)
    log = lambda m: print(m, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    held = lives()
    grafts = [g for g in draw(held) if g["index"] % a.shards == a.shard]
    path, partial = out / f"grafts-{a.shard}-of-{a.shards}.jsonl.gz", out / f"partial-{a.shard}-of-{a.shards}.jsonl.gz"
    if path.exists():
        return
    relay = Relay()
    setup = out / "setup.json"
    if a.shard == 0 and not setup.exists():
        setup.write_text(json.dumps(checks(relay, held, log), indent=1) + "\n")
    rows = W.read_jsonl(partial) if partial.exists() else []
    for g in grafts[len(rows):]:
        read = relay.read(held[g["r"]], held[g["d"]], g["i"], g["t"])
        rows.append(dict(g, p_R={key: round(read[key], 6) for key in READS}))
        if len(rows) % 25 == 0:
            W.write_jsonl(partial, rows)
            log(f"  shard {a.shard}: {len(rows)} of {len(grafts)} grafts")
    W.write_jsonl(path, rows)
    if partial.exists():
        partial.unlink()
    log(f"shard {a.shard}: grafts done")


def gather(root):
    out = Path(root)
    grafts = draw(lives())
    measured = sorted((r for p in sorted(out.glob("grafts-*-of-*.jsonl.gz")) for r in W.read_jsonl(p)),
                      key=lambda r: r["index"])
    fields = ("index", "r", "d", "i", "t", "rule_effect")
    if [{k: m[k] for k in fields} for m in measured] != [{k: g[k] for k in fields} for g in grafts]:
        raise SystemExit("the grafts are not those the protocol draws")
    setup = json.loads((out / "setup.json").read_text())
    return json.loads(json.dumps(dict(verdicts(measured, setup), setup=setup)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--shards", type=int, default=1)
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
