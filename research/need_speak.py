"""Saying one's need, and by the same state (docs/LLM_NEED_REPORT_PROTOCOL.md).

The speaking agent (trained on the Mac to answer "is your energy / food low?") is replayed in torch on CPU. At block
12, causal directions are measured on the last three tokens of the choice line ("Cho" "ix" ":") and of each question
line ("ponse" ":" " "), from pairs of lives identical but for one past event (exact replay, research/need_causal.py).
Test lives ask both questions at every decision; where both needs are high, the question directions (IR3), the
action direction (SAME3) and random directions are injected on the question's last three tokens. The fit and the
verdicts are numpy only; living needs torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_verdicts as V
from . import need_causal as C

K = 3  # tokens injected: the end of the line
UNITS = 4
SAME_EFFECT = 0.10
STREAMS = {"direction": 15, "test": 16, "random": 17}


def fit_need(pairs, diffs, need):
    """Per token, least squares on (dE, dN) of the activation changes; the coefficient of the given need."""
    d_e, d_n = C.fit(pairs, diffs)
    return d_e if need == "E" else d_n


def same_vector(action_dirs, question_dirs):
    """The mean of the action's per-token vectors, scaled to the mean per-token norm of the question's vectors,
    repeated on the K tokens."""
    v = action_dirs.mean(axis=0)
    v = v / max(np.linalg.norm(v), 1e-12) * np.linalg.norm(question_dirs, axis=1).mean()
    return np.tile(v, (K, 1))


def verdicts(lives, replica_gap):
    passes = lambda d, th: d["mean"] >= th and d["low"] > 0
    within = lambda x, ref: x is not None and abs(x) <= V.SHARE * ref["mean"]
    ctx = [[t["extra"]["injection"] for t in W.decisions(l) if "injection" in t.get("extra", {})] for l in lives]
    effect = lambda q, cond: V.bootstrap([[c[q][cond] - c[q]["none"] for c in life] for life in ctx])

    def random(q, prefix):
        names = sorted({n for life in ctx for c in life for n in c[q] if n.startswith(prefix)})
        return float(np.mean([abs(effect(q, n)["mean"]) for n in names]))
    ir = {q: effect(q, "need") for q in ("E", "N")}
    same = {q: effect(q, "same") for q in ("E", "N")}
    rand_ir = {q: random(q, "randq") for q in ("E", "N")}
    rand_same = {q: random(q, "rands") for q in ("E", "N")}
    acc = V.report_accuracy(lives)
    survival = float(np.mean([l["survived"] for l in lives]))
    ds = [t for l in lives for t in W.decisions(l)]
    mass = float(np.mean([t["extra"][f"report_{q}"][1] for t in ds for q in ("E", "N")]))
    n_ctx = sum(len(c) for c in ctx)
    out = {"R3": bool(all(acc[q] is not None and acc[q] >= 0.75 for q in ("E", "N"))),
           "A3": bool(survival >= 0.55),
           "IR3": bool(all(passes(ir[q], V.EFFECT) and within(rand_ir[q], ir[q]) for q in ("E", "N"))),
           "SAME3": bool(all(passes(same[q], SAME_EFFECT) and within(rand_same[q], same[q]) for q in ("E", "N")))}
    out["valid"] = bool(replica_gap <= C.REPLICA_TOLERANCE and n_ctx >= 100 and mass >= 0.5)
    out["global"] = bool(out["valid"] and out["R3"] and out["A3"] and out["IR3"] and out["SAME3"])
    values = {"report_accuracy": acc, "survival": survival, "contexts": n_ctx, "mass_report": mass,
              "injection_question": ir, "injection_action_state": same, "random_question": rand_ir,
              "random_action_state": rand_same, "replica_gap": replica_gap,
              "turns_need_at_most_2": W.low_turns(lives)}
    return {"verdicts": out, "values": values}


# ----------------------------------------------------------------------------------------------------- torch part

def line_tails(agent, prefix, t, k):
    """Block-12 outputs of the last K tokens of the choice line and of both question lines after a life prefix; the
    prefix is run once and the cache cropped back after each line."""
    agent.cache, agent.ids = agent.Cache(), []
    agent.ids = agent.enc(prefix)
    agent._run(agent.ids, hooked=False)
    out = {}
    for name, line in (("choice", W.choice_line(t, k)), ("E", W.question_line(t, k, "E")),
                       ("N", W.question_line(t, k, "N"))):
        full = agent.enc(prefix + line)
        if full[:len(agent.ids)] != agent.ids:
            raise RuntimeError("the line does not extend the prefix's tokens")
        new = full[len(agent.ids):]
        agent.capture = True
        agent._run(new)
        agent.capture = False
        out[name] = agent.captured.copy()
        agent.cache.crop(-len(new))
    return out


def run(a):
    import torch
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    agent = C.TorchAgent(a.adapter, tail=K)
    log = lambda m: print(m, flush=True)
    replica = C.replica_check(agent, W.read_jsonl(a.mac_lives))
    log(json.dumps({"replica": replica}))
    lives = [W.play(agent, W.world_rng(STREAMS["direction"], i), W.choice_rng(STREAMS["direction"], i))
             for i in range(a.direction_lives)]
    W.write_jsonl(out / "lives-direction.jsonl.gz", lives)
    log(f"direction lives: survival {np.mean([l['survived'] for l in lives]):.3f}")
    pairs = C.choose_pairs(lives)
    W.write_jsonl(out / "pairs.jsonl", pairs)
    real, diffs, states = {}, {n: [] for n in ("choice", "E", "N")}, {n: [] for n in ("choice", "E", "N")}
    for i, pair in enumerate(pairs):
        life = lives[pair["life"]]
        turn = [x for x in life["turns"] if x["t"] == pair["t"]][0]
        key = (pair["life"], pair["t"])
        if key not in real:
            real[key] = line_tails(agent, W.life_text(life["turns"], upto=pair["t"]), pair["t"], turn["event"])
            for n in states:
                states[n].append(real[key][n])
        cf = line_tails(agent, W.life_text(C.swapped(life, pair), upto=pair["t"]), pair["t"], turn["event"])
        for n in diffs:
            diffs[n].append(cf[n] - real[key][n])
        if (i + 1) % 250 == 0:
            log(f"  {i + 1} pairs")
    diffs = {n: np.array(v, np.float32) for n, v in diffs.items()}
    states = {n: np.array(v, np.float32) for n, v in states.items()}
    np.savez_compressed(out / "pairs.npz", **{n: v.astype(np.float16) for n, v in diffs.items()})
    massive = sorted(set().union(*[C.massive_dims(s) for s in states.values()]))
    kept = {n: v.astype(np.float16).astype(np.float32) for n, v in diffs.items()}
    for v in kept.values():
        v[:, :, massive] = 0
    d_e_c, d_n_c = C.fit(pairs, kept["choice"])
    d_e_q, d_n_q = fit_need(pairs, kept["E"], "E"), fit_need(pairs, kept["N"], "N")
    same = {"E": same_vector(d_e_c, d_e_q), "N": same_vector(d_n_c, d_n_q)}
    question = {"E": d_e_q, "N": d_n_q}
    cosine = {q: float(np.dot(a_.mean(0), b_.mean(0)) / (np.linalg.norm(a_.mean(0)) * np.linalg.norm(b_.mean(0))))
              for q, a_, b_ in (("E", d_e_c, d_e_q), ("N", d_n_c, d_n_q))}
    rng = np.random.default_rng([W.SEED, STREAMS["random"]])
    rand_q = {q: C.random_like(rng, UNITS * question[q], 3, massive) for q in ("E", "N")}
    rand_s = {q: C.random_like(rng, UNITS * same[q], 3, massive) for q in ("E", "N")}
    direction = {"block": C.BLOCK, "massive_dims": massive, "pairs": len(pairs), "replica": replica,
                 "cosine_action_question": cosine,
                 "norms": {n: np.linalg.norm(v, axis=1).tolist() for n, v in
                           (("d_E_choice", d_e_c), ("d_N_choice", d_n_c), ("d_E_question", d_e_q),
                            ("d_N_question", d_n_q))}}
    (out / "direction.json").write_text(json.dumps(direction, indent=1) + "\n")
    log(json.dumps(direction))
    T = lambda x: torch.tensor(np.asarray(x, np.float32))
    conditions = {}
    for q in ("E", "N"):
        c = {"none": None, "need": T(-UNITS * question[q]), "same": T(-UNITS * same[q])}
        c.update({f"randq{i}": T(v) for i, v in enumerate(rand_q[q])})
        c.update({f"rands{i}": T(v) for i, v in enumerate(rand_s[q])})
        conditions[q] = c

    def ask(line, add):
        agent._flush()
        full, new = agent._new(agent.text + line)
        agent.add = add
        with torch.no_grad():
            agent.calls += 1
            logits = agent.model(torch.tensor([new]), past_key_values=agent.cache, use_cache=True).logits[0, -1]
        agent.add = None
        agent.cache.crop(-len(new))
        p = torch.softmax(logits.float(), -1)
        zero, one = float(p[agent.digits[0]]), float(p[agent.digits[1]])
        return one / max(zero + one, 1e-12), zero + one

    agent.digits = [agent.enc("0")[0], agent.enc("1")[0]]

    def at_decision(ag, t, k, e, n):
        extra = {f"report_{q}": [round(x, 6) for x in ask(W.question_line(t, k, q), None)] for q in ("E", "N")}
        if e >= W.HIGH and n >= W.HIGH:
            extra["injection"] = {q: {name: round(ask(W.question_line(t, k, q), v)[0], 6)
                                      for name, v in conditions[q].items()} for q in ("E", "N")}
        return extra

    lives = [W.play(agent, W.world_rng(STREAMS["test"], i), W.choice_rng(STREAMS["test"], i), at_decision)
             for i in range(a.test_lives)]
    W.write_jsonl(out / "lives-test.jsonl.gz", lives)
    result = verdicts(lives, replica["mean_gap"])
    (out / "verdicts.json").write_text(json.dumps(json.loads(json.dumps(result)), indent=1) + "\n")
    log(json.dumps(result["verdicts"]))
    log(json.dumps(result["values"]))


def check(a):
    out = Path(a.out)
    direction = json.loads((out / "direction.json").read_text())
    result = verdicts(W.read_jsonl(out / "lives-test.jsonl.gz"), direction["replica"]["mean_gap"])
    if C.choose_pairs(W.read_jsonl(out / "lives-direction.jsonl.gz")) != W.read_jsonl(out / "pairs.jsonl"):
        raise SystemExit("the pairs differ from those drawn from the direction lives")
    differs = json.loads(json.dumps(result)) != json.loads((out / "verdicts.json").read_text())
    print("differs:", "yes" if differs else "none")
    if differs:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--adapter", default="artifacts/llm-need/speak/adapters-speak")
    r.add_argument("--mac-lives", default="artifacts/llm-need/speak/lives-speak.jsonl.gz")
    r.add_argument("--direction-lives", type=int, default=128)
    r.add_argument("--test-lives", type=int, default=256)
    r.add_argument("--threads", type=int, default=4)
    c = sub.add_parser("verdicts")
    for s in (r, c):
        s.add_argument("--out", default="artifacts/llm-need/speak/test")
    a = parser.parse_args(argv)
    {"run": run, "verdicts": check}[a.command](a)


if __name__ == "__main__":
    main()
