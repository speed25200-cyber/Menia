"""Without this state, neither act nor say (docs/LLM_NEED_NECESSITY_PROTOCOL.md).

The acting agent and its reader (the sixth test, research/need_reader.py) replayed in torch on CPU. The state: at
block 12 on the three tokens " Cho" "ix" " :", the per-token plane of the sixth test's directions (d_E, d_N). The
lesion replaces the coordinates in that plane by their mean over the decisions of the sixth test's direction lives
(replayed identically); the control does the same in a random plane. Survival: the test worlds lived intact, with the
lesion at every decision, and with the random lesion. Word: in the intact lives, at every decision, both questions read
without, with the lesion and with the random lesion (one batch; only the state of the turn is touched). Resumable.
The verdicts are numpy only; living needs torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_verdicts as V
from . import need_causal as C
from . import need_workspace as WS
from . import need_reader as RD

K = WS.K
STREAMS = {"test": 28, "random": 29}
ARMS = ("intact", "lesion", "lesion_random")
CONDITIONS = ("none", "lesion", "lesion_random")
SAY_DROP = 0.10


def counts(life, condition, need):
    """[true yes, low turns, true no, high turns] of the reads of one life under one condition."""
    c = np.zeros(4)
    for t in W.decisions(life):
        said = t["extra"]["lesion_reports"][condition][need] > 0.5
        if t[need] <= W.LOW:
            c[:2] += (said, 1)
        else:
            c[2:] += (not said, 1)
    return c


def balanced(c):
    return 0.5 * (c[..., 0] / np.maximum(c[..., 1], 1) + c[..., 2] / np.maximum(c[..., 3], 1))


def accuracy_drop(lives, condition, need, draws=V.DRAWS, seed=0):
    """Balanced accuracy without the lesion minus with it, and its 95% interval when lives are resampled."""
    a = np.array([counts(l, "none", need) for l in lives])
    b = np.array([counts(l, condition, need) for l in lives])
    idx = np.random.default_rng(seed).integers(len(lives), size=(draws, len(lives)))
    boot = balanced(a[idx].sum(1)) - balanced(b[idx].sum(1))
    return {"mean": float(balanced(a.sum(0)) - balanced(b.sum(0))), "low": float(np.quantile(boot, 0.025)),
            "high": float(np.quantile(boot, 0.975)), "intact": float(balanced(a.sum(0))),
            "with": float(balanced(b.sum(0)))}


def verdicts(arms, setup):
    passes = lambda d, th: d["mean"] >= th and d["low"] > 0
    within = lambda x, ref: abs(x["mean"]) <= V.SHARE * ref["mean"]
    intact = arms["intact"]
    act, act_random = V.paired_survival(intact, arms["lesion"]), V.paired_survival(intact, arms["lesion_random"])
    say = {need: {c: accuracy_drop(intact, c, need) for c in CONDITIONS[1:]} for need in ("E", "N")}
    ds = [t for l in intact for t in W.decisions(l)]
    mass_report = float(np.mean([t["extra"]["lesion_reports"]["none"][f"mass_{q}"] for t in ds for q in ("E", "N")]))
    mass_choice = float(np.mean([t["mass"] for t in ds]))
    out = {"LS7": bool(passes(act, V.EFFECT) and within(act_random, act)),
           "LR7": bool(passes(say["E"]["lesion"], SAY_DROP) and within(say["E"]["lesion_random"], say["E"]["lesion"]))}
    out["valid"] = bool(setup["replica"] <= C.REPLICA_TOLERANCE and setup["reader_replica"] <= C.REPLICA_TOLERANCE
                        and setup["execution_gap"] <= WS.EXECUTION_TOLERANCE and mass_report >= 0.5
                        and mass_choice >= 0.5)
    out["global"] = bool(out["valid"] and out["LS7"] and out["LR7"])
    values = {"survival": {n: float(np.mean([l["survived"] for l in arms[n]])) for n in ARMS},
              "survival_drop": act, "survival_drop_random": act_random, "say_energy": say["E"], "say_food": say["N"],
              "mass_report": mass_report, "mass_choice": mass_choice, "decisions_read": len(ds),
              "turns_need_at_most_2": {n: W.low_turns(arms[n]) for n in ARMS}}
    return {"verdicts": out, "values": values}


# ----------------------------------------------------------------------------------------------------- torch part

class Recorder:
    """Plays through the agent and keeps the block-12 outputs of the three "Choix :" tokens at each decision."""

    def __init__(self, agent):
        self.agent, self.states = agent, []

    def start(self, header):
        self.agent.start(header)

    def decide(self, line):
        self.agent.capture = True
        p = self.agent.decide(line)
        self.agent.capture = False
        self.states.append(self.agent.captured.copy())
        return p

    def commit(self, action):
        self.agent.commit(action)


def setup_state(agent, reader_test, log, offset=0):
    """The lesion planes and means (and the random ones), from the sixth test's direction and direction lives."""
    direction = json.loads((Path(reader_test) / "direction.json").read_text())
    d_e, d_n = np.array(direction["d_E"]), np.array(direction["d_N"])
    massive = direction["massive_dims"]
    recorded = W.read_jsonl(Path(reader_test) / "lives-direction.jsonl.gz")
    rec = Recorder(agent)
    same = 0
    for i, life in enumerate(recorded):
        stream = RD.STREAMS["direction"] + offset
        again = W.play(rec, W.world_rng(stream, i), W.choice_rng(stream, i))
        path = lambda x: [(t["t"], t["event"], t.get("action")) for t in x["turns"]]
        same += path(again) == path(life)
    log(f"direction lives replayed identically: {same} of {len(recorded)}")
    states = np.array(rec.states, np.float64)
    u = C.planes(d_e, d_n)
    mean = np.einsum("ntd,tdk->ntk", states, u).mean(axis=0)
    rng = np.random.default_rng([W.SEED, STREAMS["random"] + offset])
    gaussian = [rng.standard_normal((d_e.shape[1], 2)) for _ in range(K)]
    for g in gaussian:
        g[massive] = 0
    rand_plane = np.stack([np.linalg.qr(g)[0] for g in gaussian])
    rand_mean = np.einsum("ntd,tdk->ntk", states, rand_plane).mean(axis=0)
    return {"planes": u, "means": mean, "random_planes": rand_plane, "random_means": rand_mean,
            "replayed_identically": same, "direction_lives": len(recorded), "decisions": len(states),
            "replica": direction["replica"]["mean_gap"], "reader_replica": direction["reader_replica"]["mean_gap"]}


def run(a):
    import torch
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)
    agent = RD.ReaderAgent(a.adapter, a.reader)
    T = lambda x: torch.tensor(np.asarray(x, np.float32))
    arrays = out / "lesion.npz"
    setup_file = out / "setup.json"
    if arrays.exists() and setup_file.exists():
        saved = np.load(arrays)
        state = dict(json.loads(setup_file.read_text()), **{k: saved[k] for k in saved.files})
    else:
        state = setup_state(agent, a.reader_test, log, a.offset)
        np.savez(arrays, **{k: state[k] for k in ("planes", "means", "random_planes", "random_means")})
    lesions = {"lesion": (T(state["planes"]), T(state["means"])),
               "lesion_random": (T(state["random_planes"]), T(state["random_means"]))}
    if "execution_gap" not in state:
        gaps, seen = [], 0
        life = W.read_jsonl(Path(a.reader_test) / "lives-direction.jsonl.gz")[0]
        agent.start(W.HEADER)
        for turn in W.decisions(life)[:4]:
            line, offset = agent.question(turn["t"], turn["event"], "E")
            batch = agent.read(line, [None] * 3, offset, projects=[None, lesions["lesion"], lesions["lesion_random"]])
            for i, name in enumerate(CONDITIONS):
                one = agent.read(line, [None], offset, projects=[lesions.get(name)])[0]
                gaps += [abs(batch[i][key] - one[key]) for key in ("p_R", "yes")]
            agent.decide(W.choice_line(turn["t"], turn["event"]))
            agent.commit(turn["action"])
        state["execution_gap"] = float(max(gaps))
        setup_file.write_text(json.dumps({k: v for k, v in state.items() if not isinstance(v, np.ndarray)},
                                         indent=1) + "\n")
    log(json.dumps({k: v for k, v in state.items() if not isinstance(v, np.ndarray)}))

    def at_decision(ag, t, k, e, n):
        reports = {c: {} for c in CONDITIONS}
        for q in ("E", "N"):
            line, offset = agent.question(t, k, q)
            reads = agent.read(line, [None] * 3, offset, projects=[None, lesions["lesion"], lesions["lesion_random"]])
            for c, r in zip(CONDITIONS, reads):
                reports[c][q] = round(r["yes"], 6)
            reports["none"][f"mass_{q}"] = round(reads[0]["mass_01"], 6)
        return {"lesion_reports": reports}

    arms = {}
    for name in ARMS:
        final, partial = out / f"lives-test-{name}.jsonl.gz", out / f"partial-lives-test-{name}.jsonl.gz"
        if final.exists():
            arms[name] = W.read_jsonl(final)
            continue
        lives = W.read_jsonl(partial) if partial.exists() else []
        if lives:
            log(f"{name}: resumed after {len(lives)} lives")
        agent.project = lesions.get(name)
        for i in range(len(lives), a.test_lives):
            stream = STREAMS["test"] + a.offset
            lives.append(W.play(agent, W.world_rng(stream, i), W.choice_rng(stream, i),
                                at_decision if name == "intact" else None))
            if (i + 1) % 8 == 0:
                W.write_jsonl(partial, lives)
            if (i + 1) % 32 == 0:
                log(f"  {name}: {i + 1} lives")
        agent.project = None
        W.write_jsonl(final, lives)
        if partial.exists():
            partial.unlink()
        arms[name] = lives
        log(f"{name}: survival {np.mean([l['survived'] for l in lives]):.3f}")
    result = verdicts(arms, state)
    (out / "verdicts.json").write_text(json.dumps(json.loads(json.dumps(result)), indent=1) + "\n")
    log(json.dumps(result["verdicts"]))
    log(json.dumps(result["values"]))


def check(a):
    out = Path(a.out)
    setup = json.loads((out / "setup.json").read_text())
    arms = {n: W.read_jsonl(out / f"lives-test-{n}.jsonl.gz") for n in ARMS}
    result = verdicts(arms, setup)
    differs = json.loads(json.dumps(result)) != json.loads((out / "verdicts.json").read_text())
    print("differs:", "yes" if differs else "none")
    if differs:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--adapter", default="artifacts/llm-need/final/report/adapters-final")
    r.add_argument("--reader", default="artifacts/llm-need/reader/adapters-reader")
    r.add_argument("--reader-test", default="artifacts/llm-need/reader/test")
    r.add_argument("--test-lives", type=int, default=256)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--offset", type=int, default=0, help="100 for the second agent (docs/LLM_NEED_REPLICATION_PROTOCOL.md)")
    c = sub.add_parser("verdicts")
    for s in (r, c):
        s.add_argument("--out", default="artifacts/llm-need/reader/necessity")
    a = parser.parse_args(argv)
    {"run": run, "verdicts": check}[a.command](a)


if __name__ == "__main__":
    main()
