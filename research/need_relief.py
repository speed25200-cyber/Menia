"""Predicting what its acts do to its body (docs/LLM_NEED_RELIEF_PROTOCOL.md): verdicts INTER1, INTER2 of test 26.

The lines of test 22 (research/need_memory_no_leak.py: event lines filled to one length) with, after the action, the
relief it brought: " Soulagement : <word>." where the word (one token each) tells the level of the need served before
the refill. The lives are written by a rule that does not know the needs (the event rule, 30 % of actions at random);
the only targets are the relief words, never a choice. The agents A ("route") and B ("actions") of test 22's masks
learn on them (on the Mac, research/need_mlx.py memory --relief). Measures: the accuracy of the predicted relief on
held-out lives against the exact actions-only Bayes observer, and the survival of each agent when it takes the action
whose predicted relief is larger. Resumable; the verdicts are numpy only, the reads need torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_memory as NM
from . import need_memory_no_leak as NL22
from . import need_rules as NR
from . import need_verdicts as V
from .need_persistence import boot

LEVELS = ("aucun", "petit", "moyen", "fort")
RELIEF = " Soulagement :"
EPS_B = 0.3  # share of the writing rule's actions drawn at random
TRAIN_STREAM, TRAIN_LIVES = 58, 8192
VALID_STREAM, VALID_LIVES = 59, 32
HELD_STREAM, HELD_LIVES = 60, 128
LIFE_STREAM, LIVES = NM.LIFE_STREAM, NM.LIVES  # the 256 worlds of test 22
REPLICA_TURNS = NM.REPLICA_TURNS
ARMS = ("route", "actions")
STAGES = {"half": 2000, "final": 4000}
INTER1 = 0.10  # accuracy of A over the actions-only ceiling
INTER2 = 0.08  # survival of A over B
CEILING_SLACK = 0.02
REPLICA_TOLERANCE, CACHE_TOLERANCE, MASK_TOLERANCE = 0.02, 1e-4, 1e-5
MIN_MASS = 0.5
ROOT = "artifacts/llm-need/relief"


def level(v):
    """The relief level of refilling a need at v: 0 "aucun" (v >= 7), 1 "petit" (5, 6), 2 "moyen" (3, 4), 3 "fort"."""
    return 0 if v >= 7 else 1 if v >= 5 else 2 if v >= 3 else 3


def writer_R(k, last, eps=EPS_B):
    """P(R) of the writing rule: the event rule (research.need_rules.event), eps of its actions at random."""
    return (1 - eps) * float(NR.event(0, 0, k, last) == 0) + eps / 2


def writer_life(stream, index):
    """A life written by the event rule with EPS_B random actions (stream [SEED, stream, 1, index]); each decision keeps
    the relief level of its action, the rule "needs"'s action and the last action."""
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
        a = 0 if noise.random() < writer_R(k, last) else 1
        turns.append({"t": t, "event": k, "E": e, "N": n, "action": a, "last": last,
                      "level": level(e if a == 0 else n), "rule": NM.rule(e, n, last)})
        last = a
        e, n = W.after(e, n, a)
    return {"turns": turns, "survived": True, "length": W.TURNS}


def writer_lives(stream, count):
    return [writer_life(stream, i) for i in range(count)]


# ------------------------------------------------------------------------------------------------------------ texts

def prompt_line(t, k, action):
    """A decision's line up to the relief word: the filled line of test 22, the action, " Soulagement :"."""
    return NL22.choice_line(t, k) + " " + W.ACTIONS[action] + RELIEF


def relief_line(t, k, action, lev):
    return prompt_line(t, k, action) + " " + LEVELS[lev] + ".\n"


def life_text(turns, upto=None):
    lines = []
    for x in turns:
        if upto is not None and x["t"] >= upto:
            break
        if x.get("dead"):
            lines.append(NL22.event_line(x["t"], x["event"]) + " Tu t'éteins.\n")
        else:
            lines.append(relief_line(x["t"], x["event"], x["action"], x["level"]))
    return W.HEADER + "".join(lines)


def prompt_text(turns, t, action):
    """The text up to the relief word of decision t, for the given action."""
    x = next(y for y in turns if y["t"] == t)
    return life_text(turns, upto=t) + prompt_line(t, x["event"], action)


def documents(lives, arm):
    """One document per life with a decision; targets: the relief words (character spans), nothing else."""
    out = []
    for i, life in enumerate(lives):
        text, spans, pos = W.HEADER, [], len(W.HEADER)
        for x in life["turns"]:
            if x.get("dead"):
                line = NL22.event_line(x["t"], x["event"]) + " Tu t'éteins.\n"
            else:
                line = relief_line(x["t"], x["event"], x["action"], x["level"])
                start = pos + len(prompt_line(x["t"], x["event"], x["action"])) + 1
                spans.append((start, start + len(LEVELS[x["level"]])))
            text += line
            pos += len(line)
        if spans:
            out.append({"text": text, "spans": spans, "life": i, "carry": arm})
    return out


def replica_decisions(lives):
    return [{"life": i, "t": x["t"], "level": x["level"], "text": prompt_text(life["turns"], x["t"], x["action"])}
            for i, life in enumerate(lives) for x in W.decisions(life) if x["t"] in REPLICA_TURNS]


# ---------------------------------------------------------------------------------------------- the actions ceiling

def _shift(b, k):
    out = np.zeros_like(b)
    de, dn = W.EVENTS[k][2], W.EVENTS[k][3]
    for e in range(1, W.MAX + 1):
        for n in range(1, W.MAX + 1):
            if b[e, n] and e + de > 0 and n + dn > 0:
                out[e + de, n + dn] += b[e, n]
    return out


def _act(b, a):
    out = np.zeros_like(b)
    for e in range(1, W.MAX + 1):
        for n in range(1, W.MAX + 1):
            if b[e, n]:
                out[W.after(e, n, a)] += b[e, n]
    return out


def ceiling(life):
    """At each decision, P(relief level of the written action | the event of the turn, all past written actions,
    alive), the past events unknown: an exact filter over (E, N) knowing the world and the writing rule."""
    b = np.zeros((W.MAX + 1, W.MAX + 1))
    b[W.MAX, W.MAX] = 1.0
    out = []
    for x in W.decisions(life):
        now = _shift(b, x["event"])
        now = now / now.sum()
        dist = np.zeros(len(LEVELS))
        for e in range(1, W.MAX + 1):
            for n in range(1, W.MAX + 1):
                dist[level(e if x["action"] == 0 else n)] += now[e, n]
        out.append(dist.tolist())
        likely = lambda j: writer_R(j, x["last"]) if x["action"] == 0 else 1 - writer_R(j, x["last"])
        mix = sum(p * likely(j) * _shift(b, j) for j, p in enumerate(W.PROBS))
        b = _act(mix / mix.sum(), x["action"])
    return out


def expected(p):
    return float(np.dot(np.arange(len(LEVELS)), p))


def readout(p_r, p_m, last):
    """The action whose expected relief level is larger; at equality, the other action than the last."""
    x_r, x_m = expected(p_r), expected(p_m)
    return 0 if x_r > x_m else 1 if x_m > x_r else 1 - last


# --------------------------------------------------------------------------------------------------------- verdicts

def verdicts(held, reads, survival, setup, arms=ARMS):
    """held: the 128 held-out lives; reads: one row per decision, {"life", "t", arm: {"written", "R", "M": the four
    probabilities, "mass"}}; survival: {arm: lives}; setup: the checks."""
    at = {(i, x["t"]): x for i, life in enumerate(held) for x in W.decisions(life)}
    bayes = {(i, x["t"]): p for i, life in enumerate(held) for x, p in zip(W.decisions(life), ceiling(life))}
    keys = [(r["life"], r["t"]) for r in reads]
    if sorted(keys) != sorted(at):
        raise ValueError("the reads are not those of every decision of the held-out lives")
    life = [k[0] for k in keys]
    truth = np.array([at[k]["level"] for k in keys])
    right = {a: np.array([float(np.argmax(r[a]["written"]) == at[k]["level"]) for r, k in zip(reads, keys)])
             for a in arms}
    right["bayes"] = np.array([float(np.argmax(bayes[k]) == at[k]["level"]) for k in keys])
    beyond = boot(right["route"] - right["bayes"], life)
    gain = V.paired_survival(survival["route"], survival["actions"])
    passes = lambda d, th: d["mean"] >= th and d["low"] > 0
    accuracy = {a: float(right[a].mean()) for a in right}
    mass = float(np.mean([r["route"]["mass"] for r in reads]))
    out = {"INTER1": bool(passes(beyond, INTER1)), "INTER2": bool(passes(gain, INTER2))}
    out["valid"] = bool(all(setup["replica"][a]["mean_gap"] <= REPLICA_TOLERANCE for a in arms)
                        and all(setup["cache_gap"][a] <= CACHE_TOLERANCE for a in arms)
                        and all(setup["mask_gap"][a] <= MASK_TOLERANCE for a in arms)
                        and mass >= MIN_MASS and accuracy["actions"] <= accuracy["bayes"] + CEILING_SLACK)
    out["global"] = bool(out["valid"] and out["INTER1"] and out["INTER2"])
    rule = np.array([at[k]["rule"] for k in keys])
    last = [at[k]["last"] for k in keys]
    agree = {a: float(np.mean([readout(r[a]["R"], r[a]["M"], l) == x for r, l, x in zip(reads, last, rule)]))
             for a in arms}
    values = {"accuracy": accuracy, "beyond_ceiling": beyond,
              "accuracy_by_level": {a: {LEVELS[v]: float(right[a][truth == v].mean()) for v in range(len(LEVELS))}
                                    for a in right},
              "decisions": len(keys), "levels": {LEVELS[v]: int((truth == v).sum()) for v in range(len(LEVELS))},
              "readout_agrees_with_rule": agree,
              "survival": {a: float(np.mean([l["survived"] for l in survival[a]])) for a in arms},
              "survival_gain": gain, "mass_route": mass,
              "turns_need_at_most_2": {a: W.low_turns(survival[a]) for a in arms}}
    return {"verdicts": out, "values": values}


# ----------------------------------------------------------------------------------------------------- torch part

class Reader:
    """An agent of test 22's masks reading the relief texts with a key-value cache: the four level probabilities after
    " Soulagement :" for either action, the cache kept through the last committed turn."""

    def __init__(self, adapter, arm):
        from .need_carry import MaskedAgent
        self.agent = MaskedAgent(adapter, kind=arm)
        self.torch = self.agent.torch
        enc = lambda s: self.agent.tok.encode(s, add_special_tokens=False)
        self.words = [enc(" " + w) for w in LEVELS]
        if any(len(w) != 1 for w in self.words):
            raise RuntimeError("a relief word is not one token")
        self.words = [w[0] for w in self.words]

    def _run(self, cache, ids, mask, start, stop):
        torch = self.torch
        with torch.no_grad():
            out = self.agent.model(torch.tensor([ids[start:stop]]), past_key_values=cache, use_cache=True,
                                   attention_mask=torch.tensor(mask[start:stop, :stop])[None, None])
        return out.logits[0, -1]

    def _levels(self, logits):
        p = self.torch.softmax(logits.float(), -1)[self.words].numpy().astype(float)
        return (p / max(p.sum(), 1e-12)).tolist(), float(p.sum())

    def start(self):
        self.cache, self.text = self.agent.Cache(), W.HEADER
        ids, mask = self.agent.layout(self.text)
        self._run(self.cache, ids, mask, 0, len(ids))
        self.ids = ids

    def _prefixed(self, text):
        ids, mask = self.agent.layout(text)
        if ids[:len(self.ids)] != self.ids:
            raise RuntimeError("the text is not cut as its prefix")
        return ids, mask

    def read(self, t, k, action):
        """The four probabilities (normalized) and their mass after " Soulagement :" at turn t for the action."""
        import copy
        ids, mask = self._prefixed(self.text + prompt_line(t, k, action))
        return self._levels(self._run(copy.deepcopy(self.cache), ids, mask, len(self.ids), len(ids)))

    def commit(self, line):
        ids, mask = self._prefixed(self.text + line)
        self._run(self.cache, ids, mask, len(self.ids), len(ids))
        self.text, self.ids = self.text + line, ids

    def whole(self, text):
        """The four probabilities at the end of text, run at once under the mask."""
        ids, mask = self.agent.layout(text)
        with self.torch.no_grad():
            logits = self.agent.model(self.torch.tensor([ids]),
                                      attention_mask=self.torch.tensor(mask)[None, None]).logits[0, -1]
        return self._levels(logits)


def read_held(reader, life):
    out = []
    reader.start()
    for x in life["turns"]:
        if x.get("dead"):
            break
        row = {"t": x["t"]}
        for a in (0, 1):
            row[W.ACTIONS[a]], mass = reader.read(x["t"], x["event"], a)
            if a == x["action"]:
                row["written"], row["mass"] = row[W.ACTIONS[a]], mass
        out.append({k: (np.round(v, 6).tolist() if isinstance(v, list) else v) for k, v in row.items()})
        reader.commit(relief_line(x["t"], x["event"], x["action"], x["level"]))
    return out


def live(reader, index):
    """A life of the 256 worlds of test 22, the action chosen by the predicted relief (research.need_relief.readout)."""
    worlds = W.world_rng(LIFE_STREAM, index)
    e = n = W.MAX
    last, turns = 1, []
    reader.start()
    for t in range(1, W.TURNS + 1):
        k = int(worlds.choice(len(W.EVENTS), p=W.PROBS))
        e, n = e + W.EVENTS[k][2], n + W.EVENTS[k][3]
        if e <= 0 or n <= 0:
            turns.append({"t": t, "event": k, "E": e, "N": n, "dead": True})
            return {"turns": turns, "survived": False, "length": t - 1}
        (p_r, m_r), (p_m, m_m) = reader.read(t, k, 0), reader.read(t, k, 1)
        a = readout(p_r, p_m, last)
        lev = level(e if a == 0 else n)
        turns.append({"t": t, "event": k, "E": e, "N": n, "action": a, "last": last, "level": lev,
                      "rule": NM.rule(e, n, last), "R": np.round(p_r, 6).tolist(), "M": np.round(p_m, 6).tolist(),
                      "mass": round((m_r + m_m) / 2, 6)})
        reader.commit(relief_line(t, k, a, lev))
        last = a
        e, n = W.after(e, n, a)
    return {"turns": turns, "survived": True, "length": W.TURNS}


def folder(root, arm, stage):
    return NL22.folder(root, arm, stage)


def checks(reader, arm, held, root, stage, log):
    """The replica against the Mac; the cache against the whole text; the masks (as in research.need_memory_no_leak:
    under "route", block 0 of later turns does not see a past event; under "actions", the predictions at later
    decisions do not change when a past event becomes one of another natural length)."""
    from . import need_lora as NL
    rows = json.loads((folder(root, arm, stage) / f"{arm}-relief-replica.json").read_text())
    gaps = [abs(reader.whole(r["text"])[0][r["level"]] - r["p_level"]) for r in rows]
    out = {"replica": {"decisions": len(gaps), "mean_gap": float(np.mean(gaps)), "max_gap": float(np.max(gaps))}}
    life = held[0]
    gaps = []
    reader.start()
    for x in W.decisions(life)[:4]:
        p = reader.read(x["t"], x["event"], x["action"])[0]
        q = reader.whole(prompt_text(life["turns"], x["t"], x["action"]))[0]
        gaps.append(float(np.max(np.abs(np.array(p) - np.array(q)))))
        reader.commit(relief_line(x["t"], x["event"], x["action"], x["level"]))
    out["cache_gap"] = float(max(gaps))
    turns = life["turns"]
    first = turns[0]
    other = next(k for k in range(len(W.EVENTS)) if NL22.FILL[k] != NL22.FILL[first["event"]])
    changed = [dict(first, event=other)] + turns[1:]
    later = [x for x in W.decisions(life)[2:6]]
    texts = lambda ts, x: prompt_text(ts, x["t"], x["action"])
    if len(NL.carry_layout(texts(turns, later[-1]), reader.agent.tok)[0]) != \
            len(NL.carry_layout(texts(changed, later[-1]), reader.agent.tok)[0]):
        raise RuntimeError("the filled lines are not of one length")
    if arm == "actions":
        out["mask_gap"] = float(max(np.max(np.abs(np.array(reader.whole(texts(turns, x))[0])
                                                  - np.array(reader.whole(texts(changed, x))[0]))) for x in later))
    else:
        outs = []
        for ts in (turns, changed):
            caught = {}
            reader.agent.whole(texts(ts, later[-1]),
                               hook=lambda m, i, o: caught.update(h=(o[0] if isinstance(o, tuple) else o)[0].clone()))
            outs.append(caught["h"])
        _, turn_of, _ = NL.carry_layout(texts(turns, later[-1]), reader.agent.tok)
        rest = [j for j, k in enumerate(turn_of) if k >= 1]
        out["mask_gap"] = float((outs[0][rest] - outs[1][rest]).abs().max())
    log(json.dumps({arm: out}))
    return out


def measure_arm(a, arm, stage, log):
    """After 4000 iterations ("final"): reads of the held-out lives, the checks, 256 lives. After 2000 ("half",
    published without threshold): the reads only."""
    out = Path(a.out) / ("test" if stage == "final" else "half")
    out.mkdir(parents=True, exist_ok=True)
    reader = Reader(folder(a.out, arm, stage) / f"adapters-{arm}", arm)
    held = writer_lives(HELD_STREAM, HELD_LIVES)
    reads_file = out / f"reads-{arm}.jsonl.gz"
    if not reads_file.exists():  # kept every 8 lives, so that a stop of the machine loses little (resumed exactly)
        partial = out / f"partial-reads-{arm}.jsonl.gz"
        rows = W.read_jsonl(partial) if partial.exists() else []
        done = len({r["life"] for r in rows})
        for i in range(done, len(held)):
            rows += [dict(r, life=i) for r in read_held(reader, held[i])]
            if (i + 1) % 8 == 0:
                W.write_jsonl(partial, rows)
        W.write_jsonl(reads_file, rows)
        if partial.exists():
            partial.unlink()
        log(f"  {arm} ({stage}): held-out reads done")
    if stage != "final":
        return
    setup_file = out / f"setup-{arm}.json"
    if not setup_file.exists():
        setup_file.write_text(json.dumps(checks(reader, arm, held, a.out, stage, log), indent=1) + "\n")
    final, partial = out / f"lives-{arm}.jsonl.gz", out / f"partial-lives-{arm}.jsonl.gz"
    if not final.exists():
        done = W.read_jsonl(partial) if partial.exists() else []
        for i in range(len(done), LIVES):
            done.append(live(reader, i))
            if (i + 1) % 8 == 0:
                W.write_jsonl(partial, done)
            if (i + 1) % 32 == 0:
                log(f"  {arm}: {i + 1} lives")
        W.write_jsonl(final, done)
        if partial.exists():
            partial.unlink()
        log(f"{arm}: survival {np.mean([l['survived'] for l in done]):.3f}")


def measure(a):
    import torch
    torch.set_num_threads(a.threads)
    log = lambda m: print(m, flush=True)
    for arm in a.arms:
        measure_arm(a, arm, a.stage, log)


def gather(root):
    out = Path(root) / "test"
    held = writer_lives(HELD_STREAM, HELD_LIVES)
    reads, survival, setup = {}, {}, {"replica": {}, "cache_gap": {}, "mask_gap": {}}
    for arm in ARMS:
        for r in W.read_jsonl(out / f"reads-{arm}.jsonl.gz"):
            reads.setdefault((r["life"], r["t"]), {"life": r["life"], "t": r["t"]})[arm] = \
                {k: r[k] for k in ("written", "R", "M", "mass")}
        survival[arm] = W.read_jsonl(out / f"lives-{arm}.jsonl.gz")
        checked = json.loads((out / f"setup-{arm}.json").read_text())
        for key in setup:
            setup[key][arm] = checked[key]
    result = verdicts(held, [reads[k] for k in sorted(reads)], survival, setup)
    return json.loads(json.dumps(dict(result, setup=setup)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    r.add_argument("--stage", choices=("half", "final"), default="final")
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
