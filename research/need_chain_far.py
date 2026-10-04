"""At a distance, does the need pass from state to state? (docs/LLM_NEED_CHAIN_FAR_PROTOCOL.md): verdicts FAR1, FAR2
of test 25.

The reads of test 24 (research/need_chain.py: own, total, direct, mediators only) on 128 new teacher lives, at h = t - i
of 3, 4 or 5. FAR1: the chain (total - direct) is clearly above half the total; FAR2: the mediators-only effect is
clearly above half the total. One thread per process; the grafts may be shared among processes (--shard k --shards n),
each graft being computed alone. The draw and the verdicts are numpy only; the reads need torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_memory as NM
from .need_carried_content import rule_effect, _boot_diff
from .need_chain import READS, _ratio, checks
from .need_persistence import boot

LIVES_STREAM, LIVES = 52, 128
STREAM = 53
GAPS = (3, 4, 5)
PER_GAP = 3
TOTAL = 0.10  # the prior condition on the total effect
HALF = 0.5
MIN_COUNT = 150  # grafts per gap
EXTEND_TOLERANCE, SELF_TOLERANCE = 1e-4, 1e-6
ROOT = "artifacts/llm-need/chain-far"


def lives():
    return NM.teacher_lives(LIVES_STREAM, LIVES)


def draw(held):
    """The grafts of the protocol, as in research.need_chain.draw with this test's stream and gaps; each with its
    index in the draw."""
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
                    out.append({"index": len(out), "r": r, "d": d, "i": i, "t": t, "h": h, "rule_effect": effect})
    return out


def measures(rows, select=None):
    rows = [r for r in rows if select is None or select(r)]
    p = {k: np.array([r["p_R"][k] for r in rows]) for k in READS}
    effect = np.array([r["rule_effect"] for r in rows])
    life = np.array([r["r"] for r in rows])
    total, direct = (p["total"] - p["own"]) * effect, (p["direct"] - p["own"]) * effect
    chain, mediators = total - direct, (p["mediators"] - p["own"]) * effect
    out = {"total": boot(total, life), "direct": boot(direct, life), "chain": boot(chain, life),
           "mediators": boot(mediators, life), "non_additive": boot(total - direct - mediators, life),
           "count": len(rows), "recipient_lives": len(set(life.tolist()))}
    out["chain_share"] = _ratio(chain, total, life)
    out["mediators_share"] = _ratio(mediators, total, life)
    out["chain_over_half_total"] = _boot_diff(chain, life, HALF * total, life)
    out["mediators_over_half_total"] = _boot_diff(mediators, life, HALF * total, life)
    return out


def verdicts(rows, setup):
    m = measures(rows)
    m["by_gap"] = {str(h): measures(rows, lambda r, h=h: r["h"] == h) for h in GAPS}
    prior = bool(m["total"]["mean"] >= TOTAL and m["total"]["low"] > 0)
    out = {"prior": prior,
           "FAR1": bool(prior and m["chain_over_half_total"]["low"] > 0),
           "FAR2": bool(prior and m["mediators_over_half_total"]["low"] > 0)}
    out["valid"] = bool(setup["extend_gap"] <= EXTEND_TOLERANCE and setup["self_graft_gap"] <= SELF_TOLERANCE
                        and setup["positions_same"] and all(m["by_gap"][str(h)]["count"] >= MIN_COUNT for h in GAPS))
    out["global"] = bool(out["valid"] and out["FAR1"] and out["FAR2"])
    return {"verdicts": out, "values": m}


# ----------------------------------------------------------------------------------------------------- torch part

def measure(a):
    import torch
    from .need_chain import Chain
    torch.set_num_threads(1)
    log = lambda m: print(m, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    held = lives()
    grafts = [g for g in draw(held) if g["index"] % a.shards == a.shard]
    path, partial = out / f"grafts-{a.shard}-of-{a.shards}.jsonl.gz", out / f"partial-{a.shard}-of-{a.shards}.jsonl.gz"
    if path.exists():
        return
    chain = Chain()
    setup = out / "setup.json"
    if a.shard == 0 and not setup.exists():
        setup.write_text(json.dumps(dict(checks(chain, held, log), threads=1), indent=1) + "\n")
    rows = W.read_jsonl(partial) if partial.exists() else []
    for g in grafts[len(rows):]:
        read = chain.read(held[g["r"]], held[g["d"]], g["i"], g["t"])
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
    fields = ("index", "r", "d", "i", "t", "h", "rule_effect")
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
