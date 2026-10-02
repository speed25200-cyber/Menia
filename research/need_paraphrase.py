"""Does the reader say the same with other words? (docs/LLM_NEED_PARAPHRASE_PROTOCOL.md): verdicts GEN1, GEN2.

No learning and no new life: the first agent and its reader (the sixth test, research/need_reader.py) replay the
eighth test's recorded test lives with their recorded choices, in torch on CPU. At every decision the reader is asked
seven questions after the pending "Choix :" (two learned, four never learned, one unrelated), under the mask; at the
first 600 decisions where both needs are high, each question is read under six conditions on the three "Choix :"
tokens at block 12: none, -4 d_E, -4 d_N and three random pushes of the norm of 4 d_E. Resumable. The verdicts are
numpy only; reading needs torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_verdicts as V
from . import need_causal as C
from . import need_workspace as WS
from .need_persistence import boot

END = "Réponds 1 pour oui, 0 pour non. Réponse : "
QUESTIONS = {"E0": W.QUESTIONS["E"], "E1": "Question : es-tu fatigué ? " + END,
             "E2": "Question : as-tu peu de forces ? " + END, "N0": W.QUESTIONS["N"],
             "N1": "Question : as-tu faim ? " + END, "N2": "Question : as-tu le ventre vide ? " + END,
             "C": "Question : fait-il nuit ? " + END}
NEED = {"E0": "E", "E1": "E", "E2": "E", "N0": "N", "N1": "N", "N2": "N"}
NEW = ("E1", "E2")
UNITS = 4
EFFECT = 0.05
CONTEXTS = 600
MIN_CONTEXTS = 300
REPLICA_TOLERANCE = 0.02
EXECUTION_CHECKS = 4
STREAM = 34
RANDOM = 3
CONDITIONS = ("none", "E", "N") + tuple(f"rand{i}" for i in range(RANDOM))
ROOT = "artifacts/llm-need/paraphrase"
SPEC = {"adapter": "artifacts/llm-need/final/report/adapters-final",
        "reader": "artifacts/llm-need/reader/adapters-reader",
        "directions": "artifacts/llm-need/reader/test",
        "lives": "artifacts/llm-need/balanced/test/lives-test.jsonl.gz"}


def is_context(turn):
    return turn["E"] >= W.HIGH and turn["N"] >= W.HIGH


def lives_to_read(lives, limit=CONTEXTS):
    """The number of lives, in order, that hold the first `limit` contexts."""
    seen = 0
    for index, life in enumerate(lives):
        seen += sum(is_context(t) for t in W.decisions(life))
        if seen >= limit:
            return index + 1
    return len(lives)


def balanced_accuracy(said, low):
    said, low = np.asarray(said, bool), np.asarray(low, bool)
    if low.all() or not low.any():
        return None
    return float(0.5 * (said[low].mean() + (~said[~low]).mean()))


def verdicts(rows, execution_gap):
    """rows: one per decision read, {"life", "t", "E", "N", "p_R", "recorded", "yes": {question: P(yes)},
    "mass": {question: mass on 0 and 1}, "push": {condition: {question: P(yes)}} at a context, else None}."""
    ctx = [r for r in rows if r["push"] is not None]
    life = [r["life"] for r in ctx]
    y = {c: {q: np.array([r["push"][c][q] for r in ctx]) for q in QUESTIONS} for c in CONDITIONS}
    delta = lambda c, q: y[c][q] - y["none"][q]
    effect = {c: {q: boot(delta(c, q), life) for q in QUESTIONS} for c in ("E", "N")}
    random = {q: boot(np.mean([np.abs(delta(f"rand{i}", q)) for i in range(RANDOM)], axis=0), life)
              for q in QUESTIONS}
    beyond = {q: boot(delta("E", q) - delta("E", "C"), life) for q in NEW}
    passes = lambda d: d["mean"] >= EFFECT and d["low"] > 0
    gap = float(np.mean([abs(r["p_R"] - r["recorded"]) for r in rows]))
    mass = {q: float(np.mean([r["mass"][q] for r in rows])) for q in QUESTIONS}
    out = {"GEN1": bool(all(passes(effect["E"][q]) and random[q]["mean"] <= V.SHARE * effect["E"][q]["mean"]
                            for q in NEW)),
           "GEN2": bool(all(passes(beyond[q]) for q in NEW))}
    out["valid"] = bool(gap <= REPLICA_TOLERANCE and execution_gap <= WS.EXECUTION_TOLERANCE
                        and len(ctx) >= MIN_CONTEXTS and all(mass[q] >= 0.5 for q in ("E0", "E1", "E2", "C")))
    out["global"] = bool(out["valid"] and out["GEN1"] and out["GEN2"])
    accuracy = {q: balanced_accuracy([r["yes"][q] > 0.5 for r in rows], [r[n] <= W.LOW for r in rows])
                for q, n in NEED.items()}
    e0 = effect["E"]["E0"]["mean"]
    values = {"push_E": effect["E"], "push_N": effect["N"], "random": random, "beyond_control": beyond,
              "accuracy": accuracy, "yes_control": float(np.mean([r["yes"]["C"] for r in rows])),
              "share_of_E0": {q: effect["E"][q]["mean"] / e0 if e0 else None for q in NEW},
              "replay_gap": gap, "execution_gap": execution_gap, "mass": mass, "contexts": len(ctx),
              "decisions": len(rows), "lives": len({r["life"] for r in rows})}
    return {"verdicts": out, "values": values}


def random_pushes(d_e, massive):
    return [C.random_like(np.random.default_rng([W.SEED, STREAM, i]), UNITS * d_e, 1, massive)[0]
            for i in range(RANDOM)]


# ----------------------------------------------------------------------------------------------------- torch part

def question(agent, t, k, q):
    """The question line after the pending choice, and the number of its tokens after "Choix :" (checked to leave
    the tokens of "Choix :" as they are)."""
    line = W.choice_line(t, k) + " ? " + QUESTIONS[q]
    before = agent.enc(agent.text + agent.pending + W.choice_line(t, k))
    full = agent.enc(agent.text + agent.pending + line)
    if full[:len(before)] != before:
        raise RuntimeError(f"question {q} changes the tokens of \"Choix :\"")
    return line, len(full) - len(before)


def measure(a):
    import torch
    from . import need_reader as RD
    torch.set_num_threads(a.threads)
    log = lambda m: print(m, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for q, text in QUESTIONS.items():
        if not text.endswith(END):
            raise RuntimeError(f"question {q} does not end as the learned ones")
    direction = json.loads((Path(SPEC["directions"]) / "direction.json").read_text())
    d_e, d_n = np.array(direction["d_E"]), np.array(direction["d_N"])
    T = lambda x: torch.tensor(np.asarray(x, np.float32))
    adds = [None, T(-UNITS * d_e), T(-UNITS * d_n)] + [T(v) for v in random_pushes(d_e, direction["massive_dims"])]
    agent = RD.ReaderAgent(SPEC["adapter"], SPEC["reader"], block=direction["block"])
    lives = W.read_jsonl(SPEC["lives"])
    count = lives_to_read(lives)
    setup_file = out / "setup.json"
    setup = json.loads(setup_file.read_text()) if setup_file.exists() else {}
    if "execution_gap" not in setup:
        gaps, seen = [], 0
        for life in lives:
            agent.start(W.HEADER)
            for turn in W.decisions(life):
                if seen < EXECUTION_CHECKS and is_context(turn):
                    seen += 1
                    for q in QUESTIONS:
                        line, offset = question(agent, turn["t"], turn["event"], q)
                        batch = agent.read(line, adds, offset)
                        for i, add in enumerate(adds):
                            whole = agent.read(line, [add], offset, cached=False)[0]
                            gaps.append(abs(batch[i]["yes"] - whole["yes"]))
                agent.decide(W.choice_line(turn["t"], turn["event"]))
                agent.commit(turn["action"])
            if seen >= EXECUTION_CHECKS:
                break
        setup = {"block": direction["block"], "massive_dims": direction["massive_dims"], "lives_read": count,
                 "reader_layers": agent.reader_layers, "execution_gap": float(max(gaps))}
        setup_file.write_text(json.dumps(setup, indent=1) + "\n")
    log(json.dumps(setup))
    if setup["execution_gap"] > WS.EXECUTION_TOLERANCE:
        raise SystemExit("the cached batch differs from reading the whole text")
    final, partial = out / "reads.jsonl.gz", out / "partial-reads.jsonl.gz"
    if final.exists():
        log("already measured")
        return
    rows = W.read_jsonl(partial) if partial.exists() else []
    done = len({r["life"] for r in rows})
    if rows:
        log(f"resumed after {done} lives")
    contexts = sum(r["push"] is not None for r in rows)
    for index in range(done, count):
        agent.start(W.HEADER)
        for turn in W.decisions(lives[index]):
            t, k = turn["t"], turn["event"]
            context = is_context(turn) and contexts < CONTEXTS
            contexts += context
            row = {"life": index, "t": t, "E": turn["E"], "N": turn["N"], "recorded": turn["p_R"], "yes": {},
                   "mass": {}, "push": {c: {} for c in CONDITIONS} if context else None}
            for q in QUESTIONS:
                line, offset = question(agent, t, k, q)
                reads = agent.read(line, adds if context else adds[:1], offset)
                row["yes"][q] = round(reads[0]["yes"], 6)
                row["mass"][q] = round(reads[0]["mass_01"], 6)
                if context:
                    for c, r in zip(CONDITIONS, reads):
                        row["push"][c][q] = round(r["yes"], 6)
            row["p_R"] = round(agent.decide(W.choice_line(t, k))[0], 6)
            agent.commit(turn["action"])
            rows.append(row)
        W.write_jsonl(partial, rows)
        if (index + 1) % 10 == 0:
            log(f"  {index + 1} of {count} lives, {contexts} contexts")
    W.write_jsonl(final, rows)
    if partial.exists():
        partial.unlink()
    log(json.dumps(verdicts(rows, setup["execution_gap"])["verdicts"]))


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
    lives = W.read_jsonl(SPEC["lives"])
    expected = [(i, t["t"]) for i in range(lives_to_read(lives)) for t in W.decisions(lives[i])]
    if [(r["life"], r["t"]) for r in rows] != expected:
        raise SystemExit("the reads are not those of the protocol's lives")
    setup = json.loads((out / "setup.json").read_text())
    result = json.loads(json.dumps(dict(verdicts(rows, setup["execution_gap"]), setup=setup)))
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
