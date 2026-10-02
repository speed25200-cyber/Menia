"""Does one agent's need push another agent? (docs/LLM_NEED_TRANSFER_PROTOCOL.md): verdicts TR1, TR2.

No learning and no new life, in torch on CPU. The first and the second agent (same language model, learned apart)
both gather their need at block 12 on the three "Choix :" tokens. In each agent's own test lives, at the first 600
decisions with both needs high, P(R) is read under: none, the agent's own push -4 d_E, the other agent's -4 d_E
brought per token to the norm of the own push, and three random pushes of that norm. The text before the three
tokens is kept in a cache and the conditions run as one batch (research/need_persistence.py). Resumable. The
verdicts are numpy only; reading needs torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_verdicts as V
from . import need_causal as C
from . import need_persistence as P

K = 3
UNITS = 4
CONTEXTS = 600
MIN_CONTEXTS = 300
REPLICA_TOLERANCE = 0.02
CACHE_TOLERANCE = 1e-4
CACHE_CHECKS = 4
STREAM = 37
RANDOM = 3
THRESHOLD = {"second": 0.02, "first": 0.05}  # TR1: the first's direction in the second; TR2: the reverse
CONDITIONS = ("none", "own", "other") + tuple(f"random{i}" for i in range(RANDOM))
ROOT = "artifacts/llm-need/transfer"
AGENTS = {
    "first": {"adapter": "artifacts/llm-need/final/report/adapters-final",
              "directions": "artifacts/llm-need/reader/test",
              "lives": "artifacts/llm-need/reader/test/lives-test.jsonl.gz", "other": "second"},
    "second": {"adapter": "artifacts/llm-need/r1/final/report/adapters-final",
               "directions": "artifacts/llm-need/r1/reader/test",
               "lives": "artifacts/llm-need/r1/reader/test/lives-test.jsonl.gz", "other": "first"},
}


def contexts(lives, limit=CONTEXTS):
    """The first `limit` decisions, in the order of the lives, with both needs at W.HIGH or more."""
    out = [(i, t["t"]) for i, life in enumerate(lives) for t in W.decisions(life)
           if t["E"] >= W.HIGH and t["N"] >= W.HIGH]
    return out[:limit]


def direction(name):
    d = json.loads((Path(AGENTS[name]["directions"]) / "direction.json").read_text())
    return np.array(d["d_E"]), d["massive_dims"], d["block"]


def pushes(own, other, massive):
    """The own push -UNITS own, the other's -UNITS other at the own push's norm per token, and RANDOM random pushes
    of that norm (stream [SEED, STREAM, i])."""
    norm = np.linalg.norm(UNITS * own, axis=1, keepdims=True)
    foreign = -other / np.linalg.norm(other, axis=1, keepdims=True) * norm
    rand = [C.random_like(np.random.default_rng([W.SEED, STREAM, i]), UNITS * own, 1, massive)[0]
            for i in range(RANDOM)]
    return [-UNITS * own, foreign] + rand


def cosines():
    a, b = direction("first")[0], direction("second")[0]
    return ((a * b).sum(axis=1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)).tolist()


def agent_values(rows):
    life = [r["life"] for r in rows]
    p = {c: np.array([r["p_R"][c] for r in rows]) for c in CONDITIONS}
    own, other = P.boot(p["own"] - p["none"], life), P.boot(p["other"] - p["none"], life)
    rand = P.boot(np.mean([np.abs(p[f"random{i}"] - p["none"]) for i in range(RANDOM)], axis=0), life)
    return {"own": own, "other": other, "random": rand,
            "other_over_own": other["mean"] / own["mean"] if own["mean"] else None,
            "execution_gap": float(np.mean([abs(r["p_R"]["none"] - r["recorded"]) for r in rows])),
            "mass": float(np.mean([r["mass"] for r in rows])), "contexts": len(rows)}


def verdicts(measured, setups):
    """measured: {agent: rows}; setups: {agent: {"cache_gap"}}."""
    values = {n: agent_values(measured[n]) for n in AGENTS if n in measured}
    passes = lambda v, th: (v["other"]["mean"] >= th and v["other"]["low"] > 0
                            and v["random"]["mean"] <= V.SHARE * v["other"]["mean"])
    valid = lambda n: bool(values[n]["execution_gap"] <= REPLICA_TOLERANCE and values[n]["contexts"] >= MIN_CONTEXTS
                           and values[n]["mass"] >= 0.5 and setups[n]["cache_gap"] <= CACHE_TOLERANCE)
    out = {}
    if "second" in values:
        out["TR1"] = bool(passes(values["second"], THRESHOLD["second"]))
        out["valid_second"] = valid("second")
    if "first" in values:
        out["TR2"] = bool(passes(values["first"], THRESHOLD["first"]))
        out["valid_first"] = valid("first")
    out["global"] = bool(len(values) == 2 and all(out.values()))
    return {"verdicts": out, "values": dict(values, cosine_d_E=cosines())}


# ----------------------------------------------------------------------------------------------------- torch part

def measure(a):
    import torch
    torch.set_num_threads(a.threads)
    log = lambda m: print(m, flush=True)
    spec = AGENTS[a.agent]
    out = Path(a.out) / a.agent
    out.mkdir(parents=True, exist_ok=True)
    own, massive, block = direction(a.agent)
    other = direction(spec["other"])[0]
    ops = [None] + [("add", torch.tensor(np.asarray(v, np.float32))) for v in pushes(own, other, massive)]
    reader = P.Reader(spec["adapter"], block)
    lives = W.read_jsonl(spec["lives"])
    chosen = contexts(lives)

    def locate(index, t):
        ids = reader.enc(P.text_upto(lives[index], t))
        if ids[-K:] != reader.choice:
            raise RuntimeError("the read does not end on \" Choix :\"")
        return ids, len(ids) - K, [len(ids) - 1]
    setup_file = out / "setup.json"
    setup = json.loads(setup_file.read_text()) if setup_file.exists() else {}
    if "cache_gap" not in setup:
        gaps = []
        for index, t in chosen[:CACHE_CHECKS]:
            ids, start, reads = locate(index, t)
            batch = reader.read(ids, start, reads, ops)[0]
            for i, op in enumerate(ops):
                gaps.append(abs(reader.read_whole(ids, start, reads, op)[0][0, 0] - batch[i, 0]))
        setup = {"agent": a.agent, "block": block, "contexts": len(chosen), "cache_gap": float(max(gaps))}
        setup_file.write_text(json.dumps(setup, indent=1) + "\n")
    log(json.dumps(setup))
    if setup["cache_gap"] > CACHE_TOLERANCE:
        raise SystemExit("the cached batch differs from reading the whole text")
    final, partial = out / "reads.jsonl.gz", out / "partial-reads.jsonl.gz"
    if final.exists():
        log("already measured")
        return
    rows = W.read_jsonl(partial) if partial.exists() else []
    for k, (index, t) in enumerate(chosen[len(rows):], start=len(rows)):
        ids, start, reads = locate(index, t)
        p, mass = reader.read(ids, start, reads, ops)
        turn = [x for x in lives[index]["turns"] if x["t"] == t][0]
        rows.append({"life": index, "t": t, "recorded": turn["p_R"], "mass": round(float(mass[0, 0]), 6),
                     "p_R": {c: round(float(p[i, 0]), 6) for i, c in enumerate(CONDITIONS)}})
        if (k + 1) % 20 == 0:
            W.write_jsonl(partial, rows)
        if (k + 1) % 100 == 0:
            log(f"  {k + 1} contexts")
    W.write_jsonl(final, rows)
    if partial.exists():
        partial.unlink()
    log(json.dumps(agent_values(rows)))


def gather(root):
    measured = {n: W.read_jsonl(Path(root) / n / "reads.jsonl.gz") for n in AGENTS
                if (Path(root) / n / "reads.jsonl.gz").exists()}
    setups = {n: json.loads((Path(root) / n / "setup.json").read_text()) for n in measured}
    for n, rows in measured.items():
        if [(r["life"], r["t"]) for r in rows] != contexts(W.read_jsonl(AGENTS[n]["lives"])):
            raise SystemExit(f"{n}: the reads are not those of the protocol's contexts")
    return json.loads(json.dumps(dict(verdicts(measured, setups), setups=setups)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--agent", choices=sorted(AGENTS), required=True)
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
