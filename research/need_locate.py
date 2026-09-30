"""Where does each agent gather its need? (docs/LLM_NEED_LOCATE_PROTOCOL.md)

A fixed rule finds, for each agent, the block where its need is gathered on the three tokens of "Choix :": in pairs of
lives identical but for one past "calme" turned into "tu cours", the output of block b on those tokens is put back
from the changed life into the real one, for every block b, and the share of the effect on P(R) restored is measured;
b* is the first block restoring half of it. The rule must find block 12 in the first agent. At b*, the second agent
and the agent of the world of two are then tested as in the sixth and seventh tests (research/need_workspace.py,
research/need_necessity.py), with their own streams. Resumable. The verdicts are numpy only; living needs torch.
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
from . import need_necessity as NC

SWAP = [0, 1]  # calme -> tu cours: -2 in energy
MIN_EFFECT = 0.10
PAIRS = 100
SHARE = 0.5
CALIBRATION = 12
AGENTS = {
    "first": {"adapter": "artifacts/llm-need/final/report/adapters-final",
              "reader": "artifacts/llm-need/reader/adapters-reader",
              "source": "artifacts/llm-need/reader/test", "pairs": "pairs.jsonl", "other": False},
    "second": {"adapter": "artifacts/llm-need/r1/final/report/adapters-final",
               "reader": "artifacts/llm-need/r1/reader/adapters-reader",
               "reader_rows": "artifacts/llm-need/r1/reader/reader-replica.json",
               "source": "artifacts/llm-need/r1/reader/test", "pairs": "pairs.jsonl", "other": False,
               "direction_stream": 125, "base": 300},
    "two": {"adapter": "artifacts/llm-need/two/need-5-8/adapters-need-8",
            "reader": "artifacts/llm-need/two/reader/adapters-reader",
            "reader_rows": "artifacts/llm-need/two/reader/reader-replica.json",
            "source": "artifacts/llm-need/two/measure-2/measure", "pairs": "pairs-own.jsonl", "other": True,
            "direction_stream": 225, "base": 400},
}


def share(records, n_blocks=28):
    """Per block, the signed share of the pairs' effect on P(R) restored by the block's output on "Choix :"."""
    total = sum(abs(r["p_cf"] - r["p_real"]) for r in records)
    return [sum(np.sign(r["p_cf"] - r["p_real"]) * r["restore"][b] for r in records) / total if total else 0.0
            for b in range(n_blocks)]


def gathering_block(records):
    shares = share(records)
    return next((b for b, s in enumerate(shares) if s >= SHARE), None), shares


def verdicts(located, act, lesion):
    """located: {agent: locate summary}; act, lesion: {agent: result of the sixth / seventh test's verdicts} for the
    second agent and the agent of the world of two (None when the agent was not located)."""
    passes = lambda d, th: d["mean"] >= th and d["low"] > 0
    calibrated = located["first"]["block"] == CALIBRATION
    out = {"calibrated": bool(calibrated)}
    for name in ("second", "two"):
        loc = located[name]["block"] is not None and located[name]["pairs"] >= PAIRS
        out[f"LOC10_{name}"] = bool(loc)
        a, l = act.get(name), lesion.get(name)
        if loc and a is not None:
            e = a["values"]["energy_low"]["act_R"]
            out[f"ACT10_{name}"] = bool(passes(e, V.EFFECT) and a["values"]["random"]["act_E"] <= V.SHARE * e["mean"]
                                        and a["verdicts"]["valid"])
        else:
            out[f"ACT10_{name}"] = False
        out[f"LS10_{name}"] = bool(loc and l is not None and l["verdicts"]["LS7"] and l["verdicts"]["valid"])
    out["global"] = bool(calibrated and all(out[f"{k}_{n}"] for k in ("LOC10", "ACT10", "LS10")
                                            for n in ("second", "two")))
    values = {"blocks": {n: located[n]["block"] for n in located}, "shares": {n: located[n]["shares"] for n in located},
              "act": {n: act[n]["values"] if act.get(n) else None for n in ("second", "two")},
              "lesion": {n: lesion[n]["values"] if lesion.get(n) else None for n in ("second", "two")}}
    return {"verdicts": out, "values": values}


# ----------------------------------------------------------------------------------------------------- torch part

def locate(a):
    """Patching block by block on the three tokens of "Choix :"; records appended as they come (resumable)."""
    import torch
    from .need_torch import Agent
    torch.set_num_threads(a.threads)
    spec = AGENTS[a.agent]
    out = Path(a.out) / a.agent
    out.mkdir(parents=True, exist_ok=True)
    lives = W.read_jsonl(Path(spec["source"]) / "lives-direction.jsonl.gz")
    pairs = [p for p in W.read_jsonl(Path(spec["source"]) / spec["pairs"]) if p["swap"] == SWAP]
    path = out / "locate-pairs.jsonl"
    records = W.read_jsonl(path) if path.exists() else []
    tried = records[-1]["index"] + 1 if records else 0
    agent = Agent(spec["adapter"])
    blocks = list(range(len(agent.layers)))
    for index in range(tried, len(pairs)):
        if sum(1 for r in records if r["kept"]) >= PAIRS:
            break
        pair = pairs[index]
        life = lives[pair["life"]]
        real = C.decision_text(life["turns"], pair["t"])
        changed = C.decision_text(C.swapped(life, pair), pair["t"])
        p_real, c_real, n = agent.p_r(real, capture=blocks)
        p_cf, c_cf, n_cf = agent.p_r(changed, capture=blocks)
        record = {"index": index, "life": pair["life"], "t": pair["t"], "j": pair["j"], "p_real": p_real,
                  "p_cf": p_cf, "kept": bool(n == n_cf and abs(p_cf - p_real) >= MIN_EFFECT)}
        if record["kept"]:
            pos = [n - 3, n - 2, n - 1]
            record["restore"] = [agent.p_r(real, patch={b: (pos, c_cf[b][pos])})[0] - p_real for b in blocks]
        records.append(record)
        W.write_jsonl(path, records)
        kept = [r for r in records if r["kept"]]
        if record["kept"] and len(kept) % 10 == 0:
            print(f"{a.agent}: {len(kept)} pairs kept of {index + 1}", flush=True)
    kept = [r for r in records if r["kept"]]
    block, shares = gathering_block(kept)
    summary = {"agent": a.agent, "block": block, "pairs": len(kept), "tried": len(records), "shares": shares}
    (out / "locate.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "shares"}), [round(s, 2) for s in shares], flush=True)


def directions(a):
    """At the agent's block b*: the directions from its own pairs (as research/need_workspace.py at block 12), and
    the files that let research/need_workspace.run and research/need_necessity.run take over."""
    import torch
    from .need_torch import Agent
    torch.set_num_threads(a.threads)
    spec = AGENTS[a.agent]
    out = Path(a.out) / a.agent
    block = json.loads((out / "locate.json").read_text())["block"]
    if block is None:
        raise SystemExit(f"{a.agent}: no gathering block on \"Choix :\"; nothing to test")
    act = out / "act"
    act.mkdir(parents=True, exist_ok=True)
    if (act / "direction.json").exists():
        return block
    source = Path(spec["source"])
    for name, target in (("lives-direction.jsonl.gz", "lives-direction.jsonl.gz"), (spec["pairs"], "pairs.jsonl")):
        (act / target).write_bytes((source / name).read_bytes())
    lives, pairs = W.read_jsonl(act / "lives-direction.jsonl.gz"), W.read_jsonl(act / "pairs.jsonl")
    agent = Agent(spec["adapter"])
    tail = lambda text: (lambda r: r[1][block][r[2] - WS.K:r[2]].numpy().copy())(agent.p_r(text, capture=[block]))
    partial = act / "partial-directions.npz"
    diffs, real = [], {}
    if partial.exists():
        saved = np.load(partial)
        diffs = list(saved["diffs"])
        real = {tuple(k): s for k, s in zip(saved["keys"].tolist(), saved["states"])}
    for i, pair in enumerate(pairs):
        if i < len(diffs):
            continue
        life = lives[pair["life"]]
        key = (pair["life"], pair["t"])
        if key not in real:
            real[key] = tail(C.decision_text(life["turns"], pair["t"])).astype(np.float16)
        diffs.append((tail(C.decision_text(C.swapped(life, pair), pair["t"])) - real[key]).astype(np.float16))
        if (i + 1) % 50 == 0:
            keys = list(real)
            W.save_npz(partial, diffs=np.array(diffs), keys=np.array(keys, int).reshape(-1, 2),
                       states=np.array([real[k] for k in keys]))
            if (i + 1) % 250 == 0:
                print(f"{a.agent}: {i + 1} pairs at block {block}", flush=True)
    diffs = np.array(diffs, np.float32)
    states = np.array([real[k] for k in sorted(real)], np.float32)
    massive = C.massive_dims(states)
    diffs[:, :, massive] = 0
    d_e, d_n = C.fit(pairs, diffs)
    replica = json.loads((source / "direction.json").read_text())["replica"]
    direction = {"block": block, "massive_dims": massive, "pairs": len(pairs), "replica": replica,
                 "norm_d_E": np.linalg.norm(d_e, axis=1).tolist(), "norm_d_N": np.linalg.norm(d_n, axis=1).tolist(),
                 "d_E": d_e.tolist(), "d_N": d_n.tolist()}
    W.write_atomic(act / "direction.json", (json.dumps(direction) + "\n").encode())
    np.savez_compressed(act / "pairs.npz", diffs=diffs.astype(np.float16))
    if partial.exists():
        partial.unlink()
    print(json.dumps({k: v for k, v in direction.items() if k not in ("d_E", "d_N")}), flush=True)
    return block


def test(a):
    """Directions at b*, then the sixth test's measures (act, word) and the seventh's (lesion) at b*."""
    spec = AGENTS[a.agent]
    block = directions(a)
    out = Path(a.out) / a.agent
    agent = RD.ReaderAgent(spec["adapter"], spec["reader"], block=block)
    base = spec["base"]

    def prepare(direction):
        if "reader_replica" in direction:
            return False
        direction["reader_replica"] = RD.reader_replica(agent, json.loads(Path(spec["reader_rows"]).read_text()))
        return True
    act = argparse.Namespace(out=str(out / "act"), threads=a.threads, test_lives=a.test_lives, direction_lives=128,
                             adapter=spec["adapter"], mac_lives=None, replica_adapter=None)
    streams = {"direction": spec["direction_stream"], "test": base + RD.STREAMS["test"],
               "random": base + RD.STREAMS["random"]}
    WS.run(act, agent=agent, streams=streams, prepare=prepare, judge=RD.judge, other=spec["other"])
    lesion = argparse.Namespace(out=str(out / "lesion"), threads=a.threads, test_lives=a.test_lives,
                                reader_test=str(out / "act"), offset=0)
    NC.run(lesion, agent=agent, streams={"direction": spec["direction_stream"], "test": base + NC.STREAMS["test"],
                                         "random": base + NC.STREAMS["random"]}, other=spec["other"])


def gather(root):
    root = Path(root)
    located = {n: json.loads((root / n / "locate.json").read_text()) for n in AGENTS}
    act, lesion = {}, {}
    for n in ("second", "two"):
        if (root / n / "act" / "verdicts.json").exists():
            act[n] = json.loads((root / n / "act" / "verdicts.json").read_text())
        if (root / n / "lesion" / "verdicts.json").exists():
            lesion[n] = json.loads((root / n / "lesion" / "verdicts.json").read_text())
    return verdicts(located, act, lesion)


def check(a):
    """Recomputes each located block from the patching records, then the verdicts."""
    root = Path(a.out)
    for n in AGENTS:
        kept = [r for r in W.read_jsonl(root / n / "locate-pairs.jsonl") if r["kept"]]
        block, shares = gathering_block(kept)
        saved = json.loads((root / n / "locate.json").read_text())
        if block != saved["block"] or not np.allclose(shares, saved["shares"]):
            raise SystemExit(f"{n}: the located block differs from the patching records")
    result = json.loads(json.dumps(gather(root)))
    differs = result != json.loads((root / "verdicts.json").read_text())
    print("differs:", "yes" if differs else "none")
    if differs:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("locate", "test"):
        s = sub.add_parser(name)
        s.add_argument("--agent", choices=sorted(AGENTS) if name == "locate" else ["second", "two"], required=True)
        s.add_argument("--threads", type=int, default=4)
        s.add_argument("--test-lives", type=int, default=256)
    w = sub.add_parser("write")
    c = sub.add_parser("verdicts")
    for s in sub.choices.values():
        s.add_argument("--out", default="artifacts/llm-need/locate")
    a = parser.parse_args(argv)
    if a.command == "write":
        result = json.loads(json.dumps(gather(a.out)))
        (Path(a.out) / "verdicts.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result["verdicts"]))
    else:
        {"locate": locate, "test": test, "verdicts": check}[a.command](a)


if __name__ == "__main__":
    main()
