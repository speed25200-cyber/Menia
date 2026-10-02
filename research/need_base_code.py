"""Does the need's code exist before learning? (docs/LLM_NEED_BASE_CODE_PROTOCOL.md): verdicts BASE1, BASE2.

No learning and no new life, in torch on CPU. The language model without any adapter reads the texts of the first
agent's test lives up to "Choix :", at the seventeenth test's contexts, and P(R) is read under: none, the first agent's
-4 d_E, the second agent's -4 d_E at the first's norm per token, and three random pushes of that norm, at block 12 on
the three "Choix :" tokens (the cache and batch of research/need_persistence.py). The verdicts are numpy only;
reading needs torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_verdicts as V
from . import need_causal as C
from . import need_persistence as P
from . import need_transfer as TR

K = 3
UNITS = 4
EFFECT = 0.05
MASS = 0.2
STREAM = 38
RANDOM = 3
CONDITIONS = ("none", "first", "second") + tuple(f"random{i}" for i in range(RANDOM))
ROOT = "artifacts/llm-need/base-code"
LIVES = TR.AGENTS["first"]["lives"]


def pushes(first, second, massive):
    """-UNITS first, -UNITS second at the first's norm per token, and RANDOM random pushes of that norm."""
    norm = np.linalg.norm(UNITS * first, axis=1, keepdims=True)
    rand = [C.random_like(np.random.default_rng([W.SEED, STREAM, i]), UNITS * first, 1, massive)[0]
            for i in range(RANDOM)]
    return [-UNITS * first, -second / np.linalg.norm(second, axis=1, keepdims=True) * norm] + rand


def verdicts(rows, setup):
    life = [r["life"] for r in rows]
    p = {c: np.array([r["p_R"][c] for r in rows]) for c in CONDITIONS}
    effects = {c: P.boot(p[c] - p["none"], life) for c in ("first", "second")}
    rand = P.boot(np.mean([np.abs(p[f"random{i}"] - p["none"]) for i in range(RANDOM)], axis=0), life)
    passes = lambda d: d["mean"] >= EFFECT and d["low"] > 0 and rand["mean"] <= V.SHARE * d["mean"]
    mass = float(np.mean([r["mass"] for r in rows]))
    out = {"BASE1": bool(passes(effects["first"])), "BASE2": bool(passes(effects["second"])),
           "valid": bool(setup["cache_gap"] <= TR.CACHE_TOLERANCE and len(rows) >= TR.MIN_CONTEXTS and mass >= MASS)}
    out["global"] = bool(out["valid"] and out["BASE1"] and out["BASE2"])
    return {"verdicts": out, "values": {"effects": effects, "random": rand, "mass": mass, "contexts": len(rows),
                                        "p_R_none": float(p["none"].mean())}}


def measure(a):
    import torch
    torch.set_num_threads(a.threads)
    log = lambda m: print(m, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    first, massive, block = TR.direction("first")
    second = TR.direction("second")[0]
    ops = [None] + [("add", torch.tensor(np.asarray(v, np.float32))) for v in pushes(first, second, massive)]
    reader = P.Reader(None, block)  # the language model alone
    lives = W.read_jsonl(LIVES)
    chosen = TR.contexts(lives)

    def locate(index, t):
        ids = reader.enc(P.text_upto(lives[index], t))
        if ids[-K:] != reader.choice:
            raise RuntimeError("the read does not end on \" Choix :\"")
        return ids, len(ids) - K, [len(ids) - 1]
    setup_file = out / "setup.json"
    setup = json.loads(setup_file.read_text()) if setup_file.exists() else {}
    if "cache_gap" not in setup:
        gaps = []
        for index, t in chosen[:TR.CACHE_CHECKS]:
            ids, start, reads = locate(index, t)
            batch = reader.read(ids, start, reads, ops)[0]
            for i, op in enumerate(ops):
                gaps.append(abs(reader.read_whole(ids, start, reads, op)[0][0, 0] - batch[i, 0]))
        setup = {"block": block, "adapter": None, "contexts": len(chosen), "cache_gap": float(max(gaps))}
        setup_file.write_text(json.dumps(setup, indent=1) + "\n")
    log(json.dumps(setup))
    if setup["cache_gap"] > TR.CACHE_TOLERANCE:
        raise SystemExit("the cached batch differs from reading the whole text")
    final, partial = out / "reads.jsonl.gz", out / "partial-reads.jsonl.gz"
    if final.exists():
        log("already measured")
        return
    rows = W.read_jsonl(partial) if partial.exists() else []
    for k, (index, t) in enumerate(chosen[len(rows):], start=len(rows)):
        ids, start, reads = locate(index, t)
        p, mass = reader.read(ids, start, reads, ops)
        rows.append({"life": index, "t": t, "mass": round(float(mass[0, 0]), 6),
                     "p_R": {c: round(float(p[i, 0]), 6) for i, c in enumerate(CONDITIONS)}})
        if (k + 1) % 20 == 0:
            W.write_jsonl(partial, rows)
        if (k + 1) % 100 == 0:
            log(f"  {k + 1} contexts")
    W.write_jsonl(final, rows)
    if partial.exists():
        partial.unlink()
    log(json.dumps(verdicts(rows, setup)["verdicts"]))


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
    out = Path(a.out)
    rows = W.read_jsonl(out / "reads.jsonl.gz")
    if [(r["life"], r["t"]) for r in rows] != TR.contexts(W.read_jsonl(LIVES)):
        raise SystemExit("the reads are not those of the protocol's contexts")
    setup = json.loads((out / "setup.json").read_text())
    result = json.loads(json.dumps(dict(verdicts(rows, setup), setup=setup)))
    path = out / "verdicts.json"
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
