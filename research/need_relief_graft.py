"""Without a teacher, what does the state carry? (docs/LLM_NEED_RELIEF_GRAFT_PROTOCOL.md): verdicts SELF1, SELF2 of
test 28.

The graft of test 23 (research/need_carried_content.py) on the agents of test 26 (research/need_relief.py, after 4000
iterations): the keys and values of turn j's carried tokens, in every layer, from a donor life with the same written
action at j; the recipient's predicted relief is read at t = j + g for its written action, as an expected level (0 to
3). The grafts (a) have other needs at j and a relief level at t that the donor's needs would change; (b) the same needs
at j. Resumable, shared among one-thread processes; the draw and the verdicts are numpy only, the reads need torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_relief as RL
from .need_carried_content import replay, _boot_diff
from .need_persistence import boot

LIVES_STREAM, LIVES = 63, 128
STREAM = 64
GAPS = (1, 2)
PER_GAP = 2
COUNTS = {"a": 400, "a0": 200, "b": 200}
MIN_COUNTS = {"a": 100, "b": 100}
ALIGN = 0.15  # SELF1, in relief levels
HALF = 0.5  # SELF2
NO_GRAFT_TOLERANCE, SELF_TOLERANCE = 1e-4, 1e-6
ARMS = RL.ARMS
ROOT = "artifacts/llm-need/relief-graft"


def lives():
    return RL.writer_lives(LIVES_STREAM, LIVES)


def level_effect(recipient, donor, j, t):
    """The true relief level at t of the recipient's written action when its needs at j are the donor's (its actions
    and events kept), minus its own; None if it would go out before t."""
    xd = next(x for x in W.decisions(donor) if x["t"] == j)
    xt = next(x for x in W.decisions(recipient) if x["t"] == t)
    needs = replay(recipient, j, xd["E"], xd["N"], t)
    if needs is None:
        return None
    return RL.level(needs[0] if xt["action"] == 0 else needs[1]) - xt["level"]


def draw(held):
    rng = np.random.default_rng([W.SEED, STREAM, 0])
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
                        e = level_effect(life, held[d], j, t)
                        if e is None or (kind == "a" and e == 0) or (kind == "a0" and e != 0):
                            continue
                        donors.append((d, e))
                    if not donors:
                        continue
                    d, e = donors[int(rng.integers(len(donors)))]
                    if counts[kind] < COUNTS[kind]:
                        out.append({"index": len(out), "kind": kind, "r": r, "d": d, "j": j, "t": t, "g": g,
                                    "level_effect": int(e)})
                        counts[kind] += 1
        if all(counts[k] >= COUNTS[k] for k in COUNTS):
            break
    return out


def measures(rows, arm):
    delta = np.array([RL.expected(r[arm][1]) - RL.expected(r[arm][0]) for r in rows])
    e = np.array([r["level_effect"] for r in rows], float)
    kind = np.array([r["kind"] for r in rows])
    life = np.array([r["r"] for r in rows])
    gap = np.array([r["g"] for r in rows])
    a, a0, b = kind == "a", kind == "a0", kind == "b"
    return {"align": boot(delta[a] * np.sign(e[a]), life[a]),
            "abs_a": float(np.abs(delta[a]).mean()), "abs_a0": float(np.abs(delta[a0]).mean()),
            "abs_b": float(np.abs(delta[b]).mean()),
            "a_over_b": _boot_diff(np.abs(delta[a]), life[a], np.abs(delta[b]), life[b]),
            "slope": float((delta[a] * e[a]).sum() / (e[a] ** 2).sum()),
            "counts": {"a": int(a.sum()), "a0": int(a0.sum()), "b": int(b.sum())},
            "recipient_lives": {k: len(set(life[m].tolist())) for k, m in (("a", a), ("a0", a0), ("b", b))},
            "by_gap": {str(g): {"align": float((delta[a & (gap == g)] * np.sign(e[a & (gap == g)])).mean()),
                                "abs_b": float(np.abs(delta[b & (gap == g)]).mean())} for g in GAPS}}


def verdicts(rows, setup):
    m = {arm: measures(rows, arm) for arm in ARMS}
    a = m["route"]
    out = {"SELF1": bool(a["align"]["mean"] >= ALIGN and a["align"]["low"] > 0),
           "SELF2": bool(a["abs_b"] <= HALF * a["abs_a"] and a["a_over_b"]["low"] > 0)}
    s = setup["route"]
    out["valid"] = bool(s["no_graft_gap"] <= NO_GRAFT_TOLERANCE and s["self_graft_gap"] <= SELF_TOLERANCE
                        and s["positions_same"] and a["counts"]["a"] >= MIN_COUNTS["a"]
                        and a["counts"]["b"] >= MIN_COUNTS["b"])
    out["global"] = bool(out["valid"] and out["SELF1"] and out["SELF2"])
    return {"verdicts": out, "values": m}


# ----------------------------------------------------------------------------------------------------- torch part

class Grafter:
    """A reader of test 26 whose cache can be grafted at turn j's carried tokens."""

    def __init__(self, arm, root=RL.ROOT):
        self.reader = RL.Reader(RL.folder(root, arm, "final") / f"adapters-{arm}", arm)
        self.agent, self.torch = self.reader.agent, self.reader.torch

    def prefix(self, turns, j, t=None):
        """The cache through turn j; the ids and mask of the text up to " Soulagement :" at t for the written action
        (or of the text through turn j); the end of turn j; the positions of turn j's carried tokens."""
        from . import need_lora as NL
        if t is None:
            text = RL.life_text(turns, upto=j + 1)
        else:
            text = RL.prompt_text(turns, t, next(x for x in turns if x["t"] == t)["action"])
        ids, mask = self.agent.layout(text)
        _, turn_of, carried = NL.carry_layout(text, self.agent.tok)
        end = max(i for i, k in enumerate(turn_of) if k == j - 1) + 1
        if t is None and end != len(ids):
            raise RuntimeError("the text through turn j does not end with turn j")
        cache = self.agent.Cache()
        self.reader._run(cache, ids, mask, 0, end)
        return cache, ids, mask, end, [i for i, (k, c) in enumerate(zip(turn_of, carried)) if k == j - 1 and c]

    def finish(self, cache, ids, mask, end):
        return self.reader._levels(self.reader._run(cache, ids, mask, end, len(ids)))[0]

    def without(self, recipient, j, t):
        cache, ids, mask, end, _ = self.prefix(recipient["turns"], j, t)
        return self.finish(cache, ids, mask, end)

    def graft(self, recipient, donor, j, t):
        own = self.without(recipient, j, t)
        cache, ids, mask, end, positions = self.prefix(recipient["turns"], j, t)
        dcache, _, _, dend, dpositions = self.prefix(donor["turns"], j)
        if positions != dpositions or dend != end:
            raise RuntimeError("the carried tokens of turn j are not at the same positions")
        for layer, dlayer in zip(cache.layers, dcache.layers):
            layer.keys[:, :, positions, :] = dlayer.keys[:, :, positions, :]
            layer.values[:, :, positions, :] = dlayer.values[:, :, positions, :]
        return own, self.finish(cache, ids, mask, end)


def checks(grafter, held, log):
    gaps, selfs = [], []
    for life in [l for l in held if len(W.decisions(l)) >= 6][:4]:
        ds = W.decisions(life)
        j, t = ds[3]["t"], ds[5]["t"]
        own = grafter.without(life, j, t)
        whole = grafter.reader.whole(RL.prompt_text(life["turns"], t, ds[5]["action"]))[0]
        gaps.append(float(np.max(np.abs(np.array(own) - np.array(whole)))))
        a, b = grafter.graft(life, life, j, t)
        selfs.append(float(np.max(np.abs(np.array(a) - np.array(b)))))
    out = {"no_graft_gap": float(max(gaps)), "self_graft_gap": float(max(selfs)), "positions_same": True,
           "threads": 1}
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
    for arm in a.arms:
        path = out / f"grafts-{arm}-{a.shard}-of-{a.shards}.jsonl.gz"
        partial = out / f"partial-{arm}-{a.shard}-of-{a.shards}.jsonl.gz"
        if path.exists():
            continue
        grafter = Grafter(arm, a.root)
        setup = out / f"setup-{arm}.json"
        if a.shard == 0 and not setup.exists():
            setup.write_text(json.dumps(checks(grafter, held, log), indent=1) + "\n")
        rows = W.read_jsonl(partial) if partial.exists() else []
        for g in grafts[len(rows):]:
            own, grafted = grafter.graft(held[g["r"]], held[g["d"]], g["j"], g["t"])
            rows.append(dict(g, levels=[np.round(own, 6).tolist(), np.round(grafted, 6).tolist()]))
            if len(rows) % 25 == 0:
                W.write_jsonl(partial, rows)
                log(f"  {arm} shard {a.shard}: {len(rows)} of {len(grafts)} grafts")
        W.write_jsonl(path, rows)
        if partial.exists():
            partial.unlink()
        log(f"{arm} shard {a.shard}: grafts done")


def gather(root):
    out = Path(root)
    grafts = draw(lives())
    rows = [dict(g) for g in grafts]
    fields = ("index", "kind", "r", "d", "j", "t", "g", "level_effect")
    for arm in ARMS:
        measured = sorted((r for p in sorted(out.glob(f"grafts-{arm}-*-of-*.jsonl.gz")) for r in W.read_jsonl(p)),
                          key=lambda r: r["index"])
        if [{k: m[k] for k in fields} for m in measured] != [{k: g[k] for k in fields} for g in grafts]:
            raise SystemExit(f"the grafts of {arm} are not those the protocol draws")
        for row, m in zip(rows, measured):
            row[arm] = m["levels"]
    setup = {arm: json.loads((out / f"setup-{arm}.json").read_text()) for arm in ARMS}
    return json.loads(json.dumps(dict(verdicts(rows, setup), setup=setup)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--shards", type=int, default=1)
    r.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    r.add_argument("--root", default=RL.ROOT, help="where the builds of test 26 are")
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
