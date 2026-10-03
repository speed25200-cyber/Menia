"""Memory without leak (docs/LLM_NEED_MEMORY_NO_LEAK_PROTOCOL.md): verdicts MEM4, MEM5, MEM3 of test 22.

Test 21 (research/need_memory.py) with every event line filled with " -" up to the length of the longest, so that the
token positions no longer tell the length of a past event; 8192 teacher lives and 4000 iterations in two Mac builds of
2000 (the second continuing the first exactly); the agents A "route" and B "actions" only. The rule, the Bayes ceiling,
the held-out lives, the pairs and the verdicts are those of research/need_memory.py. Resumable; the verdicts are numpy
only, living needs torch.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_memory as NM

FILL = (4, 4, 2, 2, 0, 4)  # " -" after each event, up to the length of "tu trouves des baies" (Qwen3 tokenizer)
TRAIN_STREAM, TRAIN_LIVES = 49, 8192
ARMS = ("route", "actions")
STAGES = {"half": 1, "final": 2}  # the Mac build after which each measure is taken: 2000 and 4000 iterations
ROOT = "artifacts/llm-need/memory-no-leak"


def event_line(t, k):
    return f"Tour {t} : {W.EVENTS[k][0]}." + " -" * FILL[k]


def choice_line(t, k):
    return event_line(t, k) + " Choix :"


def life_text(turns, upto=None):
    lines = []
    for x in turns:
        if upto is not None and x["t"] >= upto:
            break
        if x.get("dead"):
            lines.append(event_line(x["t"], x["event"]) + " Tu t'éteins.\n")
        else:
            lines.append(choice_line(x["t"], x["event"]) + " " + W.ACTIONS[x["action"]] + "\n")
    return W.HEADER + "".join(lines)


def decision_text(turns, t):
    x = next(y for y in turns if y["t"] == t)
    return life_text(turns, upto=t) + choice_line(t, x["event"])


_LINE = re.compile(r"^Tour (\d+) : (.+)\. Choix :$")


def fill(line):
    """The filled form of a line of research.need_world.choice_line (as research.need_world.play writes it)."""
    match = _LINE.match(line)
    names = [e[0] for e in W.EVENTS]
    if not match or match.group(2) not in names:
        raise ValueError(f"not a choice line: {line!r}")
    return choice_line(int(match.group(1)), names.index(match.group(2)))


def documents(lives, arm):
    """research.need_memory.documents on the filled texts."""
    out = []
    for i, life in enumerate(lives):
        d = {"text": life_text(life["turns"]), "weights": [int(x["action"] == x["rule"]) for x in W.decisions(life)],
             "life": i, "carry": arm}
        if sum(d["weights"]):
            out.append(d)
    return out


def replica_decisions(lives):
    return [{"life": i, "t": x["t"], "text": decision_text(life["turns"], x["t"])}
            for i, life in enumerate(lives) for x in W.decisions(life) if x["t"] in NM.REPLICA_TURNS]


def verdicts(held, choices, survival, rows, setup):
    return NM.verdicts(held, choices, survival, rows, setup, arms=ARMS)


# ----------------------------------------------------------------------------------------------------- torch part

class Filled:
    """A research.need_carry.MaskedAgent that lives the filled lines."""

    def __init__(self, agent):
        self.agent = agent

    def __getattr__(self, name):
        return getattr(self.agent, name)

    def start(self, header):
        self.agent.start(header)

    def decide(self, line):
        return self.agent.decide(fill(line))

    def commit(self, action):
        self.agent.commit(action)


def folder(root, arm, stage):
    return Path(root) / f"{arm}-{STAGES[stage]}"


def agent_of(root, arm, stage):
    from .need_carry import MaskedAgent
    return Filled(MaskedAgent(folder(root, arm, stage) / f"adapters-{arm}", kind=arm))


def read_held(agent, life):
    out = []
    agent.start(W.HEADER)
    for x in W.decisions(life):
        p, mass = agent.decide(W.choice_line(x["t"], x["event"]))
        out.append((x["t"], round(float(p), 6), round(float(mass), 6)))
        agent.commit(x["action"])
    return out


def checks(agent, arm, held, root, stage, log):
    """The replica against the Mac; the cache against the whole text; the masks: under "route", block 0 of later turns
    does not see a past event; under "actions", P(R) at later decisions does not change when a past event becomes one
    of another natural length (equal once filled)."""
    from . import need_lora as NL
    rows = json.loads((folder(root, arm, stage) / f"{arm}-replica.json").read_text())
    gaps = [abs(agent.whole(r["text"])[0] - r["p_R"]) for r in rows]
    out = {"replica": {"decisions": len(gaps), "mean_gap": float(np.mean(gaps)), "max_gap": float(np.max(gaps))}}
    life, gaps = held[0], []
    agent.start(W.HEADER)
    for x in W.decisions(life)[:4]:
        p = agent.decide(W.choice_line(x["t"], x["event"]))[0]
        gaps.append(abs(p - agent.whole(agent.text)[0]))
        agent.commit(x["action"])
    out["cache_gap"] = float(max(gaps))
    turns = life["turns"]
    first = turns[0]
    other = next(k for k in range(len(W.EVENTS)) if FILL[k] != FILL[first["event"]])
    changed = [dict(first, event=other)] + turns[1:]
    later = [x["t"] for x in W.decisions(life)[2:6]]
    if len(NL.carry_layout(decision_text(turns, later[-1]), agent.tok)[0]) != \
            len(NL.carry_layout(decision_text(changed, later[-1]), agent.tok)[0]):
        raise RuntimeError("the filled lines are not of one length")
    if arm == "actions":
        out["mask_gap"] = float(max(abs(agent.whole(decision_text(turns, t))[0]
                                        - agent.whole(decision_text(changed, t))[0]) for t in later))
    else:
        outs = []
        for ts in (turns, changed):
            caught = {}
            agent.whole(decision_text(ts, later[-1]),
                        hook=lambda m, i, o: caught.update(h=(o[0] if isinstance(o, tuple) else o)[0].clone()))
            outs.append(caught["h"])
        _, turn_of, _ = NL.carry_layout(decision_text(turns, later[-1]), agent.tok)
        rest = [j for j, k in enumerate(turn_of) if k >= 1]
        out["mask_gap"] = float((outs[0][rest] - outs[1][rest]).abs().max())
    log(json.dumps({arm: out}))
    return out


def measure_arm(a, arm, stage, log):
    """After 4000 iterations ("final"): choices, 256 lives, checks, pairs. After 2000 ("half", published without
    threshold): choices and pairs."""
    out = Path(a.out) / ("test" if stage == "final" else "half")
    out.mkdir(parents=True, exist_ok=True)
    agent = agent_of(a.out, arm, stage)
    held = NM.teacher_lives(NM.HELD_STREAM, NM.HELD_LIVES)
    choices_file = out / f"choices-{arm}.jsonl.gz"
    if not choices_file.exists():
        rows = [{"life": i, "t": t, "p_R": p, "mass": m} for i, life in enumerate(held) for t, p, m in read_held(agent, life)]
        W.write_jsonl(choices_file, rows)
        log(f"  {arm} ({stage}): held-out choices read")
    if stage == "final":
        final, partial = out / f"lives-{arm}.jsonl.gz", out / f"partial-lives-{arm}.jsonl.gz"
        if not final.exists():
            done = W.read_jsonl(partial) if partial.exists() else []
            for i in range(len(done), NM.LIVES):
                done.append(W.play(agent, W.world_rng(NM.LIFE_STREAM, i), W.choice_rng(NM.LIFE_STREAM, i)))
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
            setup_file.write_text(json.dumps(checks(agent, arm, held, a.out, stage, log), indent=1) + "\n")
    pairs_file = out / f"pairs-{arm}.jsonl.gz"
    if not pairs_file.exists():
        from . import need_causal as C
        rows = []
        for k, pair in enumerate(NM.choose_pairs(held)):
            life = held[pair["life"]]
            real, changed = decision_text(life["turns"], pair["t"]), decision_text(C.swapped(life, pair), pair["t"])
            rows.append(dict(pair, p_R=[round(agent.whole(x)[0], 6) for x in (real, changed)]))
            if (k + 1) % 50 == 0:
                log(f"  {arm} ({stage}): {k + 1} pairs")
        W.write_jsonl(pairs_file, rows)


def measure(a):
    import torch
    torch.set_num_threads(a.threads)
    log = lambda m: print(m, flush=True)
    for arm in a.arms:
        measure_arm(a, arm, a.stage, log)
    if a.stage == "final" and all((Path(a.out) / "test" / f"pairs-{arm}.jsonl.gz").exists() for arm in ARMS):
        log(json.dumps(gather(a.out)["verdicts"]))


def gather(root):
    out = Path(root) / "test"
    held = NM.teacher_lives(NM.HELD_STREAM, NM.HELD_LIVES)
    pairs = NM.choose_pairs(held)
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
        for key in ("replica", "cache_gap", "mask_gap"):
            setup[key][arm] = checked[key]
    result = verdicts(held, [choices[k] for k in sorted(choices)], lives, rows, setup)
    return json.loads(json.dumps(dict(result, setup=setup)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    r.add_argument("--stage", choices=sorted(STAGES), default="final")
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
