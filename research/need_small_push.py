"""A small push: does the code exist before learning, and does learning make it stable?
(docs/LLM_NEED_SMALL_PUSH_PROTOCOL.md): verdicts SMALL1, SMALL2, STAB.

No learning and no new life, in torch on CPU. The language model without adapter and the first agent read the texts of
the first agent's test lives up to "Choix :" at the contexts of the seventeenth and eighteenth tests; P(R) is read
under: none, -1 d_E of the first agent, -1 d_E of the second at the first's norm per token, and three random pushes
of that norm, at block 12 on the three "Choix :" tokens (the cache and batch of research/need_persistence.py). The
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
from . import need_transfer as TR

K = 3
UNITS = 1
EFFECT = 0.05
MASS = {"base": 0.2, "agent": 0.5}
STREAM = 39
RANDOM = 3
CONDITIONS = ("none", "first", "second") + tuple(f"random{i}" for i in range(RANDOM))
MODELS = {"base": None, "agent": TR.AGENTS["first"]["adapter"]}
ROOT = "artifacts/llm-need/small-push"
LIVES = TR.AGENTS["first"]["lives"]


def pushes(first, second, massive):
    norm = np.linalg.norm(UNITS * first, axis=1, keepdims=True)
    rand = [C.random_like(np.random.default_rng([W.SEED, STREAM, i]), UNITS * first, 1, massive)[0]
            for i in range(RANDOM)]
    return [-UNITS * first, -second / np.linalg.norm(second, axis=1, keepdims=True) * norm] + rand


def model_values(rows):
    life = [r["life"] for r in rows]
    p = {c: np.array([r["p_R"][c] for r in rows]) for c in CONDITIONS}
    return {"first": P.boot(p["first"] - p["none"], life), "second": P.boot(p["second"] - p["none"], life),
            "random": P.boot(np.mean([np.abs(p[f"random{i}"] - p["none"]) for i in range(RANDOM)], axis=0), life),
            "above_0_9": {c: float((p[c] > 0.9).mean()) for c in CONDITIONS},
            "p_R_none": float(p["none"].mean()), "mass": float(np.mean([r["mass"] for r in rows])),
            "contexts": len(rows)}


def verdicts(measured, setups):
    """measured: {"base": rows, "agent": rows}; setups: {model: {"cache_gap"}}."""
    v = {m: model_values(measured[m]) for m in MODELS if m in measured}
    out = {}
    if "base" in v:
        b = v["base"]
        passes = lambda d: d["mean"] >= EFFECT and d["low"] > 0 and b["random"]["mean"] <= V.SHARE * d["mean"]
        out["SMALL1"], out["SMALL2"] = bool(passes(b["first"])), bool(passes(b["second"]))
    if len(v) == 2:
        a, b = v["agent"]["random"], v["base"]["random"]
        out["STAB"] = bool(a["mean"] <= V.SHARE * b["mean"] and a["high"] < b["low"])
    out["valid"] = bool(len(v) == 2 and all(setups[m]["cache_gap"] <= TR.CACHE_TOLERANCE
                                            and v[m]["contexts"] >= TR.MIN_CONTEXTS and v[m]["mass"] >= MASS[m]
                                            for m in v))
    out["global"] = bool(out["valid"] and out.get("SMALL1") and out.get("SMALL2") and out.get("STAB"))
    return {"verdicts": out, "values": v}


# ----------------------------------------------------------------------------------------------------- torch part

def measure(a):
    import torch
    torch.set_num_threads(a.threads)
    log = lambda m: print(m, flush=True)
    out = Path(a.out) / a.model
    out.mkdir(parents=True, exist_ok=True)
    first, massive, block = TR.direction("first")
    second = TR.direction("second")[0]
    ops = [None] + [("add", torch.tensor(np.asarray(x, np.float32))) for x in pushes(first, second, massive)]
    reader = P.Reader(MODELS[a.model], block)
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
        setup = {"model": a.model, "adapter": MODELS[a.model], "block": block, "contexts": len(chosen),
                 "cache_gap": float(max(gaps))}
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
    log(f"{a.model} done")


def gather(root):
    measured = {m: W.read_jsonl(Path(root) / m / "reads.jsonl.gz") for m in MODELS
                if (Path(root) / m / "reads.jsonl.gz").exists()}
    setups = {m: json.loads((Path(root) / m / "setup.json").read_text()) for m in measured}
    expected = TR.contexts(W.read_jsonl(LIVES))
    for m, rows in measured.items():
        if [(r["life"], r["t"]) for r in rows] != expected:
            raise SystemExit(f"{m}: the reads are not those of the protocol's contexts")
    return json.loads(json.dumps(dict(verdicts(measured, setups), setups=setups)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", choices=sorted(MODELS), required=True)
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
