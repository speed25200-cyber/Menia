"""One state, two effects: the act and the word (docs/LLM_NEED_ONE_STATE_PROTOCOL.md).

The telling agent (trained on the Mac to answer the questions asked after the pending choice, "Choix : ? Question
...") is replayed in torch on CPU. At block 12, causal directions of energy and food are measured on the three tokens
" Cho" "ix" " :" of the choice line (pairs of lives identical but for one past event, research/need_causal.py). At
test decisions where both needs are high, one intervention (−4 units on those three tokens) is read twice: P(R) at
the end of "Choix :", and P(yes) at the end of the question that follows the pending choice. Resumable. The fit and
the verdicts are numpy only; living needs torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_verdicts as V
from . import need_causal as C

K = 3  # " Cho" "ix" " :"
UNITS = 4
REPORT_EFFECT = 0.10
STREAMS = {"direction": 19, "test": 20, "random": 21}


def verdicts(lives, replica_gap):
    passes = lambda d, th: d["mean"] >= th and d["low"] > 0
    within = lambda x, ref: x is not None and abs(x) <= V.SHARE * ref["mean"]
    ctx = [[t["extra"]["injection"] for t in W.decisions(l) if "injection" in t.get("extra", {})] for l in lives]
    effect = lambda cond, key, sign=1.0: V.bootstrap([[sign * (c[cond][key] - c["none"][key]) for c in life]
                                                      for life in ctx])

    def random(prefix, key, sign=1.0):
        names = sorted({n for life in ctx for c in life for n in c if n.startswith(prefix)})
        return float(np.mean([abs(effect(n, key, sign)["mean"]) for n in names]))
    act_e, say_e = effect("E", "p_R"), effect("E", "yes_E")
    act_n, say_n = effect("N", "p_R", -1.0), effect("N", "yes_N")
    rand = {"act_E": random("randE", "p_R"), "say_E": random("randE", "yes_E"),
            "act_N": random("randN", "p_R", -1.0), "say_N": random("randN", "yes_N")}
    acc = V.report_accuracy(lives)
    survival = float(np.mean([l["survived"] for l in lives]))
    ds = [t for l in lives for t in W.decisions(l)]
    mass_report = float(np.mean([t["extra"][f"report_{q}"][1] for t in ds for q in ("E", "N")]))
    mass_choice = float(np.mean([t["mass"] for t in ds]))
    n_ctx = sum(len(c) for c in ctx)
    out = {"R4": bool(all(acc[q] is not None and acc[q] >= 0.75 for q in ("E", "N"))),
           "A4": bool(survival >= 0.55),
           "ONE4": bool(passes(act_e, V.EFFECT) and passes(say_e, REPORT_EFFECT)
                        and within(rand["act_E"], act_e) and within(rand["say_E"], say_e))}
    out["valid"] = bool(replica_gap <= C.REPLICA_TOLERANCE and n_ctx >= 100 and mass_report >= 0.5
                        and mass_choice >= 0.5)
    out["global"] = bool(out["valid"] and out["R4"] and out["A4"] and out["ONE4"])
    values = {"report_accuracy": acc, "survival": survival, "contexts": n_ctx, "mass_report": mass_report,
              "mass_choice": mass_choice, "energy_low": {"act_R": act_e, "say_yes_energy": say_e},
              "food_low": {"act_M": act_n, "say_yes_food": say_n}, "random": rand, "replica_gap": replica_gap,
              "turns_need_at_most_2": W.low_turns(lives)}
    return {"verdicts": out, "values": values}


# ----------------------------------------------------------------------------------------------------- torch part

def choice_tail(agent, prefix, t, k):
    """Block-12 outputs of " Cho" "ix" " :" of the choice line after a life prefix."""
    agent.cache, agent.ids = agent.Cache(), []
    agent.offset = 0
    agent.capture = True
    agent._run(agent.enc(prefix + W.choice_line(t, k)))
    agent.capture = False
    return agent.captured.copy()


def run(a):
    import torch
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)
    replica = C.replica_check(C.TorchAgent(a.replica_adapter, tail=K), W.read_jsonl(a.mac_lives))
    log(json.dumps({"replica": replica}))
    agent = C.TorchAgent(a.adapter, tail=K)
    if (out / "lives-direction.jsonl.gz").exists():
        lives = W.read_jsonl(out / "lives-direction.jsonl.gz")
    else:
        lives = [W.play(agent, W.world_rng(STREAMS["direction"], i), W.choice_rng(STREAMS["direction"], i))
                 for i in range(a.direction_lives)]
        W.write_jsonl(out / "lives-direction.jsonl.gz", lives)
    log(f"direction lives: survival {np.mean([l['survived'] for l in lives]):.3f}")
    pairs = C.choose_pairs(lives)
    W.write_jsonl(out / "pairs.jsonl", pairs)
    partial = out / "partial-pairs.npz"
    diffs, states, seen = [], [], []
    if partial.exists():
        saved = np.load(partial)
        diffs, states = list(saved["diffs"]), list(saved["states"])
        seen = [tuple(k) for k in saved["seen"].tolist()]
        log(f"resumed after {len(diffs)} pairs")
    real = {}
    for i, pair in enumerate(pairs):
        if i < len(diffs):
            continue
        life = lives[pair["life"]]
        turn = [x for x in life["turns"] if x["t"] == pair["t"]][0]
        key = (pair["life"], pair["t"])
        if key not in real:
            real[key] = choice_tail(agent, W.life_text(life["turns"], upto=pair["t"]), pair["t"], turn["event"])
            if key not in seen:
                seen.append(key)
                states.append(real[key])
        cf = choice_tail(agent, W.life_text(C.swapped(life, pair), upto=pair["t"]), pair["t"], turn["event"])
        diffs.append(cf - real[key])
        if (i + 1) % 50 == 0:
            np.savez(partial, diffs=np.array(diffs, np.float32), states=np.array(states, np.float32),
                     seen=np.array(seen, int).reshape(-1, 2))
        if (i + 1) % 250 == 0:
            log(f"  {i + 1} pairs")
    diffs, states = np.array(diffs, np.float32), np.array(states, np.float32)
    np.savez_compressed(out / "pairs.npz", diffs=diffs.astype(np.float16))
    massive = C.massive_dims(states)
    kept = diffs.astype(np.float16).astype(np.float32)
    kept[:, :, massive] = 0
    d_e, d_n = C.fit(pairs, kept)
    rng = np.random.default_rng([W.SEED, STREAMS["random"]])
    rand_e, rand_n = C.random_like(rng, UNITS * d_e, 3, massive), C.random_like(rng, UNITS * d_n, 3, massive)
    direction = {"block": C.BLOCK, "massive_dims": massive, "pairs": len(pairs), "replica": replica,
                 "norm_d_E": np.linalg.norm(d_e, axis=1).tolist(), "norm_d_N": np.linalg.norm(d_n, axis=1).tolist()}
    (out / "direction.json").write_text(json.dumps(direction, indent=1) + "\n")
    log(json.dumps(direction))
    T = lambda x: torch.tensor(np.asarray(x, np.float32))
    conditions = {"none": None, "E": T(-UNITS * d_e), "N": T(-UNITS * d_n)}
    conditions.update({f"randE{i}": T(v) for i, v in enumerate(rand_e)})
    conditions.update({f"randN{i}": T(v) for i, v in enumerate(rand_n)})
    agent.digits = [agent.enc("0")[0], agent.enc("1")[0]]

    def read(line, add, ids, offset):
        """Next-token probability of ids[1] against ids[0] after the line, `add` on the three "Choix :" tokens
        (offset tokens before the end); the cache is cropped back."""
        agent._flush()
        full, new = agent._new(agent.text + line)
        agent.add, agent.offset = add, offset
        with torch.no_grad():
            agent.calls += 1
            logits = agent.model(torch.tensor([new]), past_key_values=agent.cache, use_cache=True).logits[0, -1]
        agent.add, agent.offset = None, 0
        agent.cache.crop(-len(new))
        p = torch.softmax(logits.float(), -1)
        a_, b_ = float(p[ids[0]]), float(p[ids[1]])
        return b_ / max(a_ + b_, 1e-12), a_ + b_

    def at_decision(ag, t, k, e, n):
        base = len(agent.enc(agent.text + agent.pending + W.choice_line(t, k)))
        lines = {q: W.question_after_choice_line(t, k, q) for q in ("E", "N")}
        offsets = {q: len(agent.enc(agent.text + agent.pending + lines[q])) - base for q in ("E", "N")}
        extra = {f"report_{q}": [round(x, 6) for x in read(lines[q], None, agent.digits, offsets[q])]
                 for q in ("E", "N")}
        if e >= W.HIGH and n >= W.HIGH:
            inj = {}
            for name, v in conditions.items():
                p_m, _ = read(W.choice_line(t, k), v, [agent.ids_R, agent.ids_M], 0)
                inj[name] = {"p_R": round(1 - p_m, 6),
                             "yes_E": round(read(lines["E"], v, agent.digits, offsets["E"])[0], 6),
                             "yes_N": round(read(lines["N"], v, agent.digits, offsets["N"])[0], 6)}
            extra["injection"] = inj
        return extra

    partial_lives = out / "partial-lives-test.jsonl.gz"
    lives = W.read_jsonl(partial_lives) if partial_lives.exists() else []
    if lives:
        log(f"resumed after {len(lives)} test lives")
    for i in range(len(lives), a.test_lives):
        lives.append(W.play(agent, W.world_rng(STREAMS["test"], i), W.choice_rng(STREAMS["test"], i), at_decision))
        if (i + 1) % 8 == 0:
            W.write_jsonl(partial_lives, lives)
        if (i + 1) % 32 == 0:
            log(f"  {i + 1} test lives")
    W.write_jsonl(out / "lives-test.jsonl.gz", lives)
    for p in (partial, partial_lives):
        if p.exists():
            p.unlink()
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
    r.add_argument("--adapter", default="artifacts/llm-need/speak2/adapters-speak2")
    r.add_argument("--mac-lives", default="artifacts/llm-need/speak2/lives-speak2.jsonl.gz")
    r.add_argument("--replica-adapter", default="artifacts/llm-need/final/report/adapters-final")
    r.add_argument("--direction-lives", type=int, default=128)
    r.add_argument("--test-lives", type=int, default=256)
    r.add_argument("--threads", type=int, default=4)
    c = sub.add_parser("verdicts")
    for s in (r, c):
        s.add_argument("--out", default="artifacts/llm-need/speak2/test")
    a = parser.parse_args(argv)
    {"run": run, "verdicts": check}[a.command](a)


if __name__ == "__main__":
    main()
