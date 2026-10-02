"""Does the reader say the need, or what the agent is about to do? (docs/LLM_NEED_INTENTION_PROTOCOL.md): INT0, NEED,
INTENT.

No learning and no new life, in torch on CPU. The direction of the decision ĝ: the gradient of the log-odds of P(R),
read at the end of "Choix :", with respect to the block-12 output on the three "Choix :" tokens, at the first 300
decisions with both needs high of the sixth test's direction lives (each gradient normalised per token, averaged,
normalised again). The need without the decision d_E⊥: d_E minus its projection on ĝ, brought back to the norm of
d_E, per token. Then the fourteenth test's lives and contexts (research/need_paraphrase.py) are read under seven
conditions: none, -4 d_E, -4 d_E⊥ (the need alone), +4 |d_E| ĝ (the decision alone) and three random pushes; P(R) and
P(yes) to three questions. Resumable. The verdicts are numpy only; reading needs torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_verdicts as V
from . import need_causal as C
from . import need_workspace as WS
from . import need_paraphrase as PA
from .need_persistence import boot

UNITS = 4
GRADIENTS = 300
MIN_GRADIENTS = 200
DECISION_EFFECT = 0.10
EFFECT = 0.05
BEYOND = 0.03
QUESTIONS = ("E0", "E1", "C")
CONDITIONS = ("none", "E", "need", "decision") + tuple(f"rand{i}" for i in range(PA.RANDOM))
ROOT = "artifacts/llm-need/intention"
SPEC = dict(PA.SPEC, direction_lives="artifacts/llm-need/reader/test/lives-direction.jsonl.gz")
READING = {(True, False): "le lecteur lit le besoin, pas la décision",
           (False, True): "le lecteur lit ce que l'agent va faire, pas pourquoi",
           (True, True): "il lit les deux", (False, False): "ni l'un ni l'autre, isolément"}


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def directions(gradients, d_e, massive):
    """ĝ from the per-decision gradients (n x K x D), and d_E⊥ at the norm of d_E, per token."""
    g = np.array(gradients, np.float64)
    g[:, :, massive] = 0
    g_hat = unit(unit(g).mean(axis=0))
    d_perp = d_e - (d_e * g_hat).sum(axis=1, keepdims=True) * g_hat
    d_perp = unit(d_perp) * np.linalg.norm(d_e, axis=1, keepdims=True)
    agreement = (unit(g) * g_hat[None]).sum(axis=2).mean(axis=0)  # mean cosine of each gradient with ĝ, per token
    return g_hat, d_perp, agreement


def verdicts(rows, setup):
    """rows: one per context, {"life", "t", "push": {condition: {"p_R", "mass_RM", question: P(yes),
    "mass_" + question}}}; setup: the directions' and execution checks' summary."""
    life = [r["life"] for r in rows]
    get = lambda c, k: np.array([r["push"][c][k] for r in rows])
    delta = lambda c, k: get(c, k) - get("none", k)
    effects = {c: {k: boot(delta(c, k), life) for k in ("p_R",) + QUESTIONS} for c in CONDITIONS[1:]}
    passes = lambda d, th: d["mean"] >= th and d["low"] > 0
    out = {"INT0": bool(passes(effects["decision"]["p_R"], DECISION_EFFECT)
                        and abs(effects["need"]["p_R"]["mean"]) <= V.SHARE * abs(effects["E"]["p_R"]["mean"]))}
    beyond = boot(delta("need", "E0") - delta("need", "C"), life)
    out["NEED"] = bool(passes(effects["need"]["E0"], EFFECT) and passes(beyond, BEYOND))
    out["INTENT"] = bool(passes(effects["decision"]["E0"], EFFECT))
    mass = {k: float(get("none", "mass_" + k).mean()) for k in ("RM", "E0", "C")}
    out["valid"] = bool(setup["replay_gap"] <= PA.REPLICA_TOLERANCE
                        and setup["execution_gap"] <= WS.EXECUTION_TOLERANCE and len(rows) >= PA.MIN_CONTEXTS
                        and setup["gradient_decisions"] >= MIN_GRADIENTS and all(m >= 0.5 for m in mass.values()))
    reading = READING[(out["NEED"], out["INTENT"])] if out["INT0"] else "les directions ne séparent pas besoin et décision"
    values = {"effects": effects, "need_beyond_control": beyond, "mass": mass, "contexts": len(rows),
              "random_abs": {k: float(np.mean([np.abs(delta(f"rand{i}", k)).mean() for i in range(PA.RANDOM)]))
                             for k in ("p_R",) + QUESTIONS}}
    return {"verdicts": out, "reading": reading, "values": values}


# ----------------------------------------------------------------------------------------------------- torch part

def gradient(agent, text):
    """The log-odds of P(R) at the end of text (a decision), its gradient with respect to the block's output on the
    last K tokens (K x D), and P(R); the whole text run at once, without the cache."""
    torch = agent.torch
    store = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        leaf = h.detach().requires_grad_(True)
        store["h"] = leaf
        return (leaf,) + tuple(out[1:]) if isinstance(out, tuple) else leaf
    handle = agent.model.model.layers[agent.block].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            logits = agent.model(torch.tensor([agent.enc(text)])).logits[0, -1].float()
            lp = torch.log_softmax(logits, -1)
            odds = lp[agent.ids_R] - lp[agent.ids_M]
            odds.backward()
    finally:
        handle.remove()
    p = torch.softmax(logits.detach(), -1)
    r, m = float(p[agent.ids_R]), float(p[agent.ids_M])
    return store["h"].grad[0, -WS.K:].numpy().copy(), r / (r + m)


def measure(a):
    import torch
    from . import need_reader as RD
    torch.set_num_threads(a.threads)
    log = lambda m: print(m, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    direction = json.loads((Path(SPEC["directions"]) / "direction.json").read_text())
    d_e, massive = np.array(direction["d_E"]), direction["massive_dims"]
    agent = RD.ReaderAgent(SPEC["adapter"], SPEC["reader"], block=direction["block"])
    for p in agent.model.parameters():
        p.requires_grad_(False)
    arrays, setup_file = out / "intention.npz", out / "setup.json"
    setup = json.loads(setup_file.read_text()) if setup_file.exists() else {}
    if not arrays.exists():
        partial = out / "partial-gradients.npz"
        saved = np.load(partial) if partial.exists() else None
        grads = list(saved["gradients"]) if saved is not None else []
        gaps = list(saved["gaps"]) if saved is not None else []
        picked = [(life, t) for life in W.read_jsonl(SPEC["direction_lives"]) for t in W.decisions(life)
                  if t["E"] >= W.HIGH and t["N"] >= W.HIGH][:GRADIENTS]
        for life, turn in picked[len(grads):]:
            g, p = gradient(agent, C.decision_text(life["turns"], turn["t"]))
            grads.append(g)
            gaps.append(abs(p - turn["p_R"]))
            if len(grads) % 25 == 0:
                W.save_npz(partial, gradients=np.array(grads, np.float32), gaps=np.array(gaps))
                log(f"  {len(grads)} gradients")
        g_hat, d_perp, agreement = directions(grads, d_e, massive)
        W.save_npz(arrays, g_hat=g_hat, d_perp=d_perp)
        setup = {"gradient_decisions": len(grads), "gradient_replay_gap": float(np.mean(gaps)),
                 "agreement": agreement.tolist(), "cos_d_E_g": (unit(d_e) * g_hat).sum(axis=1).tolist()}
        setup_file.write_text(json.dumps(setup, indent=1) + "\n")
        if partial.exists():
            partial.unlink()
    saved = np.load(arrays)
    g_hat, d_perp = saved["g_hat"], saved["d_perp"]
    T = lambda x: torch.tensor(np.asarray(x, np.float32))
    adds = [None, T(-UNITS * d_e), T(-UNITS * d_perp), T(UNITS * np.linalg.norm(d_e, axis=1, keepdims=True) * g_hat)]
    adds += [T(v) for v in PA.random_pushes(d_e, massive)]
    lives = W.read_jsonl(SPEC["lives"])
    count = PA.lives_to_read(lives)
    if "execution_gap" not in setup:
        gaps, seen = [], 0
        for life in lives:
            agent.start(W.HEADER)
            for turn in W.decisions(life):
                if seen < PA.EXECUTION_CHECKS and PA.is_context(turn):
                    seen += 1
                    for q in QUESTIONS:
                        line, offset = PA.question(agent, turn["t"], turn["event"], PA.QUESTIONS[q])
                        batch = agent.read(line, adds, offset)
                        for i, add in enumerate(adds):
                            whole = agent.read(line, [add], offset, cached=False)[0]
                            gaps += [abs(batch[i][k] - whole[k]) for k in ("yes", "p_R")]
                agent.decide(W.choice_line(turn["t"], turn["event"]))
                agent.commit(turn["action"])
            if seen >= PA.EXECUTION_CHECKS:
                break
        setup["execution_gap"] = float(max(gaps))
        setup_file.write_text(json.dumps(setup, indent=1) + "\n")
    log(json.dumps(setup))
    if setup["execution_gap"] > WS.EXECUTION_TOLERANCE:
        raise SystemExit("the cached batch differs from reading the whole text")
    final, partial = out / "reads.jsonl.gz", out / "partial-reads.jsonl.gz"
    if final.exists():
        log("already measured")
        return
    rows = W.read_jsonl(partial) if partial.exists() else []
    progress = W.read_jsonl(out / "partial-replay.jsonl.gz") if (out / "partial-replay.jsonl.gz").exists() else []
    done = len(progress)
    contexts = len(rows)
    for index in range(done, count):
        agent.start(W.HEADER)
        gaps = []
        for turn in W.decisions(lives[index]):
            t, k = turn["t"], turn["event"]
            if PA.is_context(turn) and contexts < PA.CONTEXTS:
                contexts += 1
                row = {"life": index, "t": t, "push": {c: {} for c in CONDITIONS}}
                for q in QUESTIONS:
                    line, offset = PA.question(agent, t, k, PA.QUESTIONS[q])
                    for c, r in zip(CONDITIONS, agent.read(line, adds, offset)):
                        row["push"][c][q] = round(r["yes"], 6)
                        row["push"][c]["mass_" + q] = round(r["mass_01"], 6)
                        if q == QUESTIONS[0]:
                            row["push"][c]["p_R"] = round(r["p_R"], 6)
                            row["push"][c]["mass_RM"] = round(r["mass_RM"], 6)
                rows.append(row)
            gaps.append(abs(agent.decide(W.choice_line(t, k))[0] - turn["p_R"]))
            agent.commit(turn["action"])
        progress.append({"life": index, "replay_gap": float(np.mean(gaps)), "decisions": len(gaps)})
        W.write_jsonl(partial, rows)
        W.write_jsonl(out / "partial-replay.jsonl.gz", progress)
        if (index + 1) % 10 == 0:
            log(f"  {index + 1} of {count} lives, {contexts} contexts")
    setup["replay_gap"] = float(np.average([p["replay_gap"] for p in progress],
                                           weights=[p["decisions"] for p in progress]))
    setup_file.write_text(json.dumps(setup, indent=1) + "\n")
    W.write_jsonl(final, rows)
    for f in (partial, out / "partial-replay.jsonl.gz"):
        if f.exists():
            f.unlink()
    result = verdicts(rows, setup)
    log(json.dumps(result["verdicts"]) + " " + result["reading"])


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
    expected = [(i, t["t"]) for i in range(PA.lives_to_read(lives)) for t in W.decisions(lives[i])
                if PA.is_context(t)][:PA.CONTEXTS]
    if [(r["life"], r["t"]) for r in rows] != expected:
        raise SystemExit("the reads are not those of the protocol's contexts")
    setup = json.loads((out / "setup.json").read_text())
    saved = np.load(out / "intention.npz")
    direction = json.loads((Path(SPEC["directions"]) / "direction.json").read_text())
    d_e = np.array(direction["d_E"])
    expected_perp = unit(d_e - (d_e * saved["g_hat"]).sum(axis=1, keepdims=True) * saved["g_hat"])
    if not np.allclose(unit(saved["d_perp"]), expected_perp, atol=1e-6):
        raise SystemExit("d_E⊥ is not d_E without its projection on ĝ")
    result = json.loads(json.dumps(dict(verdicts(rows, setup), setup=setup)))
    path = out / "verdicts.json"
    if a.command == "write":
        path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps(result["verdicts"]), result["reading"])
    else:
        differs = result != json.loads(path.read_text())
        print("differs:", "yes" if differs else "none")
        if differs:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
