"""Memory that the choices require (docs/LLM_NEED_MEMORY_PROTOCOL.md): verdicts MEM4, MEM5, MEM3.

The teacher is the rule "needs" (research/need_rules.py): R if E < N, M if N < E, at equality the other action than the
last. Its lives (20 % of the written actions drawn at random) are the documents of three new adapters learned on the Mac
(research/need_mlx.py memory) on the fused final agent: A "route" under the memory mask of the carry test, B "actions"
(the past seen only through its action tokens, which see no event), C "free" (no mask). Measures in torch on CPU:
choices on 128 held-out teacher lives against the Bayes ceiling of an observer of the actions and of the event of the
turn; survival (256 new lives each); pairs "calme" -> "tu cours" at a past turn j, t - j >= 4. Resumable. The teacher,
the ceiling and the verdicts are numpy only; living needs torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_verdicts as V
from . import need_causal as C
from . import need_rules as NR
from .need_persistence import boot

EPS = 0.2  # share of the teacher's written actions drawn at random
TRAIN_STREAM, TRAIN_LIVES = 47, 4096
VALID_STREAM, VALID_LIVES = 48, 32
HELD_STREAM, HELD_LIVES = 45, 128
LIFE_STREAM, LIVES = 44, 256
PAIR_STREAM = 46
PAIRS, MIN_PAIRS, PER_LIFE, MIN_GAP = 300, 150, 3, 4
SWAP = [0, 1]  # "calme" -> "tu cours": energy -2 at the swapped turn
REPLICA_TURNS = (3, 10, 20)
BEYOND = 0.05  # MEM4: accuracy of A over the Bayes ceiling
SHARE = 0.5  # MEM5: share of the rule's effect
SURVIVAL_GAIN = 0.08  # MEM3: survival of A over B
CEILING_SLACK = 0.02  # validity: B may not beat the ceiling by more
REPLICA_TOLERANCE = 0.02
CACHE_TOLERANCE = 1e-4
MASK_TOLERANCE = 1e-5
ARMS = ("route", "actions", "free")
ROOT = "artifacts/llm-need/memory"
FINAL = "artifacts/llm-need/final/report/adapters-final"


def rule(e, n, last):
    return NR.needs(e, n, None, last)


def teacher_life(stream, index):
    """A life of the rule "needs" in the worlds of the programme; with probability EPS the written action is R or M
    at random (stream [SEED, stream, 1, index]). Each decision keeps the rule's action and the last written action."""
    worlds = W.world_rng(stream, index)
    noise = np.random.default_rng([W.SEED, stream, 1, index])
    e = n = W.MAX
    last, turns = 1, []
    for t in range(1, W.TURNS + 1):
        k = int(worlds.choice(len(W.EVENTS), p=W.PROBS))
        e, n = e + W.EVENTS[k][2], n + W.EVENTS[k][3]
        if e <= 0 or n <= 0:
            turns.append({"t": t, "event": k, "E": e, "N": n, "dead": True})
            return {"turns": turns, "survived": False, "length": t - 1}
        r = rule(e, n, last)
        a = r if noise.random() >= EPS else int(noise.random() < 0.5)
        turns.append({"t": t, "event": k, "E": e, "N": n, "action": a, "rule": r, "last": last})
        last = a
        e, n = W.after(e, n, a)
    return {"turns": turns, "survived": True, "length": W.TURNS}


def teacher_lives(stream, count):
    return [teacher_life(stream, i) for i in range(count)]


def documents(lives, arm):
    """One document per life; targets: the decisions whose written action is the rule's (weight 1), the others 0. The
    arm's mask is given by "carry" (none for "free"). A life with no such decision has no target and is left out
    (amendment 1 of the protocol)."""
    out = []
    for i, life in enumerate(lives):
        d = {"text": W.life_text(life["turns"]), "weights": [int(x["action"] == x["rule"]) for x in W.decisions(life)],
             "life": i}
        if arm != "free":
            d["carry"] = arm
        if sum(d["weights"]):
            out.append(d)
    return out


def replica_decisions(lives):
    """The texts up to "Choix :" of turns 3, 10 and 20 of the validation lives."""
    return [{"life": i, "t": x["t"], "text": W.life_text(life["turns"], upto=x["t"]) + W.choice_line(x["t"], x["event"])}
            for i, life in enumerate(lives) for x in W.decisions(life) if x["t"] in REPLICA_TURNS]


# ------------------------------------------------------------------------------------------------- the Bayes ceiling

def _shift(b, k):
    de, dn = W.EVENTS[k][2], W.EVENTS[k][3]
    out = np.zeros_like(b)
    out[1:W.MAX + 1 + de, 1:W.MAX + 1 + dn] = b[1 - de:W.MAX + 1, 1 - dn:W.MAX + 1]
    return out


def _act(b, action):
    out = np.zeros_like(b)
    for e in range(1, W.MAX + 1):
        for n in range(1, W.MAX + 1):
            if b[e, n]:
                out[W.after(e, n, action)] += b[e, n]
    return out


_LEVELS = range(1, W.MAX + 1)
RULE_R = {last: np.array([[float(e > 0 and n > 0 and rule(e, n, last) == 0) for n in range(W.MAX + 1)]
                          for e in range(W.MAX + 1)]) for last in (0, 1)}


def ceiling(life, eps=EPS):
    """At each decision, P(the rule chooses R | all past written actions, the event of the turn, alive), the past
    events unknown: an exact filter over (E, N) knowing the rule and the share eps of random actions."""
    b = np.zeros((W.MAX + 1, W.MAX + 1))
    b[W.MAX, W.MAX] = 1.0
    out = []
    for x in W.decisions(life):
        now = _shift(b, x["event"])
        out.append(float((now * RULE_R[x["last"]]).sum() / now.sum()))
        mix = sum(p * _shift(b, j) for j, p in enumerate(W.PROBS))
        mix /= mix.sum()
        same = RULE_R[x["last"]] if x["action"] == 0 else (1.0 - RULE_R[x["last"]])
        post = mix * ((1 - eps) * same + eps / 2)
        b = _act(post / post.sum(), x["action"])
    return out


def correct(p, action):
    """1 if choosing R when p > 0.5 (M when p < 0.5) is the rule's action; 1/2 at p = 0.5."""
    return 0.5 if p == 0.5 else float((p > 0.5) == (action == 0))


# ---------------------------------------------------------------------------------------------------------- pairs

def choose_pairs(lives, limit=PAIRS):
    """In the order of the lives, up to PER_LIFE pairs per life drawn from [SEED, PAIR_STREAM, 0] among its decisions'
    candidates: a past "calme" at j turned into "tu cours", t - j >= MIN_GAP, the needs at t changed, alive all along
    (research.need_causal.candidate_pairs, exact replay with the same written actions)."""
    rng = np.random.default_rng([W.SEED, PAIR_STREAM, 0])
    out = []
    for index, life in enumerate(lives):
        cands = [dict(c, life=index, t=turn["t"]) for turn in W.decisions(life) for c in C.candidate_pairs(life, turn)
                 if c["swap"] == SWAP and turn["t"] - c["j"] >= MIN_GAP]
        for k in sorted(rng.permutation(len(cands))[:PER_LIFE]):
            out.append(cands[k])
            if len(out) == limit:
                return out
    return out


def rule_effect(life, pair):
    x = next(d for d in W.decisions(life) if d["t"] == pair["t"])
    changed = rule(x["E"] + pair["dE"], x["N"] + pair["dN"], x["last"]) == 0
    return float(changed) - float(x["rule"] == 0)


# ------------------------------------------------------------------------------------------------------- verdicts

def verdicts(held, choices, survival, rows, setup):
    """held: the 128 held-out teacher lives; choices: one row per decision of the held lives, {"life", "t", arm: P(R)};
    survival: {arm: lives}; rows: one per pair, {"life", "t", "j", arm: [P(R) real, P(R) changed]}; setup: the checks."""
    rule_of = {(i, x["t"]): x for i, life in enumerate(held) for x in W.decisions(life)}
    bayes = {(i, x["t"]): p for i, life in enumerate(held) for x, p in zip(W.decisions(life), ceiling(life))}
    keys = [(c["life"], c["t"]) for c in choices]
    if sorted(keys) != sorted(rule_of):
        raise ValueError("the choices are not those of every decision of the held-out lives")
    life = [k[0] for k in keys]
    act = [rule_of[k]["rule"] for k in keys]
    right = {a: np.array([correct(c[a], r) for c, r in zip(choices, act)]) for a in ARMS}
    right["bayes"] = np.array([correct(bayes[k], r) for k, r in zip(keys, act)])
    memory = np.array([NR.event(0, 0, rule_of[k]["event"], rule_of[k]["last"]) != rule_of[k]["rule"] for k in keys])
    beyond = boot(right["route"] - right["bayes"], life)
    pair_life = [r["life"] for r in rows]
    effect = {a: np.array([r[a][1] - r[a][0] for r in rows]) for a in ARMS}
    ideal = np.array([rule_effect(held[r["life"]], r) for r in rows])
    carried = boot(effect["route"], pair_life)
    gain = V.paired_survival(survival["route"], survival["actions"])
    passes = lambda d, th: d["mean"] >= th and d["low"] > 0
    mass = float(np.mean([t["mass"] for l in survival["route"] for t in W.decisions(l)]))
    accuracy = {a: float(right[a].mean()) for a in right}
    out = {"MEM4": bool(passes(beyond, BEYOND)),
           "MEM5": bool(passes(carried, SHARE * float(ideal.mean()))),
           "MEM3": bool(passes(gain, SURVIVAL_GAIN))}
    out["valid"] = bool(all(setup["replica"][a]["mean_gap"] <= REPLICA_TOLERANCE for a in ARMS)
                        and all(setup["cache_gap"][a] <= CACHE_TOLERANCE for a in ARMS)
                        and setup["mask_gap"]["route"] <= MASK_TOLERANCE
                        and setup["mask_gap"]["actions"] <= MASK_TOLERANCE
                        and len(rows) >= MIN_PAIRS and mass >= 0.5
                        and accuracy["actions"] <= accuracy["bayes"] + CEILING_SLACK)
    out["global"] = bool(out["valid"] and out["MEM4"] and out["MEM5"])
    by_gap = {}
    for g in sorted({r["t"] - r["j"] for r in rows}):
        m = np.array([r["t"] - r["j"] == g for r in rows])
        by_gap[str(g)] = {"pairs": int(m.sum()), "rule": float(ideal[m].mean()),
                          **{a: float(effect[a][m].mean()) for a in ARMS}}
    values = {"accuracy": accuracy, "beyond_ceiling": beyond,
              "accuracy_where_memory_matters": {a: float(right[a][memory].mean()) for a in right},
              "decisions": len(keys), "memory_decisions": int(memory.sum()),
              "effect": {a: boot(effect[a], pair_life) for a in ARMS}, "rule_effect": float(ideal.mean()),
              "by_gap": by_gap, "pairs": len(rows), "pair_lives": len(set(pair_life)),
              "survival": {a: float(np.mean([l["survived"] for l in survival[a]])) for a in ARMS},
              "survival_gain": gain, "survival_free_over_actions": V.paired_survival(survival["free"],
                                                                                       survival["actions"]),
              "mass_route": mass, "turns_need_at_most_2": {a: W.low_turns(survival[a]) for a in ARMS}}
    return {"verdicts": out, "values": values}


# ----------------------------------------------------------------------------------------------------- torch part
# One agent at a time (each was learned in its own Mac build, fetched into root/<arm>): its choices on the held-out
# lives, its 256 lives, its checks and its pairs, each in its own file; the verdicts once the three are measured.

def agent_of(root, arm):
    from .need_carry import MaskedAgent
    return MaskedAgent(Path(root) / arm / f"adapters-{arm}", kind=arm)


def read_held(agent, life):
    """P(R) and mass at each decision of a held-out teacher life, its written actions committed."""
    out = []
    agent.start(W.HEADER)
    for x in W.decisions(life):
        p, mass = agent.decide(W.choice_line(x["t"], x["event"]))
        out.append((x["t"], round(float(p), 6), round(float(mass), 6)))
        agent.commit(x["action"])
    return out


def changed_first_event(agent, life):
    """The life with the event of turn 1 replaced by another of the same token length, and four later decisions."""
    turns = life["turns"]
    first = turns[0]
    enc = lambda s: agent.tok.encode(s, add_special_tokens=False)
    other = next(k for k in range(len(W.EVENTS)) if k != first["event"]
                 and len(enc(W.event_line(1, k))) == len(enc(W.event_line(1, first["event"]))))
    return turns, [dict(first, event=other)] + turns[1:], [x["t"] for x in W.decisions(life)[2:6]]


def checks(agent, arm, held, root, log):
    """The replica against the Mac; the cache against the whole text; the mask: under "route", block 0 of later turns
    does not see a past event; under "actions", P(R) at later decisions does not change with a past event."""
    from . import need_lora as NL
    rows = json.loads((Path(root) / arm / f"{arm}-replica.json").read_text())
    gaps = [abs(agent.whole(r["text"])[0] - r["p_R"]) for r in rows]
    out = {"replica": {"decisions": len(gaps), "mean_gap": float(np.mean(gaps)), "max_gap": float(np.max(gaps))}}
    life, gaps = held[0], []
    agent.start(W.HEADER)
    for x in W.decisions(life)[:4]:
        p = agent.decide(W.choice_line(x["t"], x["event"]))[0]
        gaps.append(abs(p - agent.whole(agent.text)[0]))
        agent.commit(x["action"])
    out["cache_gap"] = float(max(gaps))
    turns, changed, later = changed_first_event(agent, held[0])
    if arm == "actions":
        out["mask_gap"] = float(max(abs(agent.whole(C.decision_text(turns, t))[0]
                                        - agent.whole(C.decision_text(changed, t))[0]) for t in later))
    elif arm == "route":
        outs = []
        for ts in (turns, changed):
            caught = {}
            agent.whole(C.decision_text(ts, later[-1]),
                        hook=lambda m, i, o: caught.update(h=(o[0] if isinstance(o, tuple) else o)[0].clone()))
            outs.append(caught["h"])
        _, turn_of, _ = NL.carry_layout(C.decision_text(turns, later[-1]), agent.tok)
        rest = [j for j, k in enumerate(turn_of) if k >= 1]
        out["mask_gap"] = float((outs[0][rest] - outs[1][rest]).abs().max())
    log(json.dumps({arm: out}))
    return out


def measure_arm(a, arm, log):
    out = Path(a.out) / "test"
    out.mkdir(parents=True, exist_ok=True)
    agent = agent_of(a.out, arm)
    held = teacher_lives(HELD_STREAM, HELD_LIVES)
    choices_file = out / f"choices-{arm}.jsonl.gz"
    if not choices_file.exists():
        rows = [{"life": i, "t": t, "p_R": p, "mass": m} for i, life in enumerate(held) for t, p, m in read_held(agent, life)]
        W.write_jsonl(choices_file, rows)
        log(f"  {arm}: held-out choices read")
    final, partial = out / f"lives-{arm}.jsonl.gz", out / f"partial-lives-{arm}.jsonl.gz"
    if not final.exists():
        done = W.read_jsonl(partial) if partial.exists() else []
        for i in range(len(done), LIVES):
            done.append(W.play(agent, W.world_rng(LIFE_STREAM, i), W.choice_rng(LIFE_STREAM, i)))
            if (i + 1) % 8 == 0:
                W.write_jsonl(partial, done)
            if (i + 1) % 32 == 0:
                log(f"  {arm}: {i + 1} lives")
        W.write_jsonl(final, done)
        if partial.exists():
            partial.unlink()
        log(f"{arm}: survival {np.mean([l['survived'] for l in done]):.3f}")
    setup_file = out / f"setup-{arm}.json"
    if not setup_file.exists():
        setup_file.write_text(json.dumps(checks(agent, arm, held, a.out, log), indent=1) + "\n")
    pairs_file = out / f"pairs-{arm}.jsonl.gz"
    if not pairs_file.exists():
        rows = []
        for k, pair in enumerate(choose_pairs(held)):
            life = held[pair["life"]]
            real, changed = C.decision_text(life["turns"], pair["t"]), C.decision_text(C.swapped(life, pair), pair["t"])
            enc = lambda s: agent.tok.encode(s, add_special_tokens=False)
            if len(enc(real)) != len(enc(changed)):
                raise RuntimeError("a swap changes the number of tokens")
            rows.append(dict(pair, p_R=[round(agent.whole(x)[0], 6) for x in (real, changed)]))
            if (k + 1) % 50 == 0:
                log(f"  {arm}: {k + 1} pairs")
        W.write_jsonl(pairs_file, rows)


def measure(a):
    import torch
    torch.set_num_threads(a.threads)
    log = lambda m: print(m, flush=True)
    for arm in a.arms:
        measure_arm(a, arm, log)
    if all((Path(a.out) / "test" / f"pairs-{arm}.jsonl.gz").exists() for arm in ARMS):
        log(json.dumps(gather(a.out)["verdicts"]))


def gather(root):
    """The three agents' measures put together, the pairs checked against those the protocol draws."""
    out = Path(root) / "test"
    held = teacher_lives(HELD_STREAM, HELD_LIVES)
    pairs = choose_pairs(held)
    fields = ("life", "t", "j", "swap", "dE", "dN")
    choices, rows, setup, lives = {}, [dict(p) for p in pairs], {"replica": {}, "cache_gap": {}, "mask_gap": {}}, {}
    for arm in ARMS:
        for c in W.read_jsonl(out / f"choices-{arm}.jsonl.gz"):
            choices.setdefault((c["life"], c["t"]), {"life": c["life"], "t": c["t"]})[arm] = c["p_R"]
        measured = W.read_jsonl(out / f"pairs-{arm}.jsonl.gz")
        if [{k: r[k] for k in fields} for r in measured] != [{k: p[k] for k in fields} for p in pairs]:
            raise SystemExit("the pairs are not those the protocol draws from the held-out lives")
        for row, r in zip(rows, measured):
            row[arm] = r["p_R"]
        lives[arm] = W.read_jsonl(out / f"lives-{arm}.jsonl.gz")
        checked = json.loads((out / f"setup-{arm}.json").read_text())
        setup["replica"][arm], setup["cache_gap"][arm] = checked["replica"], checked["cache_gap"]
        if "mask_gap" in checked:
            setup["mask_gap"][arm] = checked["mask_gap"]
    result = verdicts(held, [choices[k] for k in sorted(choices)], lives, rows, setup)
    return json.loads(json.dumps(dict(result, setup=setup)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    sub.add_parser("write")
    sub.add_parser("verdicts")
    for s in sub.choices.values():
        s.add_argument("--out", default=ROOT)
    a = parser.parse_args(argv)
    if a.command == "run":
        measure(a)
        return
    result = gather(a.out)
    path = Path(a.out) / "test" / "verdicts.json"
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
