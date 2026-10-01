"""A reader that learns enough (docs/LLM_NEED_LONG_READER_PROTOCOL.md): verdicts R11, SAY11.

The readers of the second agent and of the agent of the world of two learned again on the same documents, 2,000
iterations instead of 600 (on the Mac, in two slices of 1,000 continued exactly: research.need_lora.skipping and
keeping_moments). Each is measured as in the sixth test (research/need_workspace.py, research/need_reader.py), at
the block where its agent gathers its need, with the directions the localization test fitted there
(research/need_locate.py) and new test lives. Resumable. The verdicts are numpy only; living needs torch.
"""
import argparse
import json
import re
from pathlib import Path
from . import need_world as W
from . import need_verdicts as V
from . import need_workspace as WS
from . import need_reader as RD
from . import need_one as O

ACCURACY = 0.75
ROOT = "artifacts/llm-need/long"
AGENTS = {
    "second": {"adapter": "artifacts/llm-need/r1/final/report/adapters-final", "reader": f"{ROOT}/second/slice-2",
               "first_reader": "artifacts/llm-need/r1/reader",
               "directions": "artifacts/llm-need/locate/second/act", "base": 500, "other": False},
    "two": {"adapter": "artifacts/llm-need/two/need-5-8/adapters-need-8", "reader": f"{ROOT}/two/slice-2",
            "first_reader": "artifacts/llm-need/two/reader",
            "directions": "artifacts/llm-need/locate-two/slice-3/measure/two/act", "base": 600, "other": True},
}


def agent_verdicts(reader):
    """reader: the sixth test's result (research.need_reader.verdicts) on one agent's long reader."""
    v = reader["values"]
    say = v["energy_low"]["say_yes_energy"]
    accuracy = v["report_accuracy"]["E"]
    out = {"R11": bool(accuracy is not None and accuracy >= ACCURACY),
           "SAY11": bool(say["mean"] >= O.REPORT_EFFECT and say["low"] > 0
                         and abs(v["random"]["say_E"]) <= V.SHARE * say["mean"]),
           "valid": bool(reader["verdicts"]["valid"])}
    return {"verdicts": out, "values": v}


def verdicts(readers):
    """readers: {agent: the sixth test's result on its long reader}, for "second" and "two"."""
    per = {n: agent_verdicts(readers[n]) for n in AGENTS if n in readers}
    out = {f"{k}_{n}": r["verdicts"][k] for n, r in per.items() for k in ("R11", "SAY11", "valid")}
    out["global"] = bool(len(per) == len(AGENTS) and all(all(r["verdicts"].values()) for r in per.values()))
    return {"verdicts": out, "values": {n: r["values"] for n, r in per.items()}}


def measure(a):
    """The sixth test's measures for one agent's long reader, at its block, its test lives of stream base + 26."""
    spec = AGENTS[a.agent]
    out = Path(a.out) / a.agent / "test"
    RD.reuse_direction(Path(spec["directions"]), out)
    block = json.loads((out / "direction.json").read_text())["block"]
    reader = Path(spec["reader"])
    agent = RD.ReaderAgent(spec["adapter"], reader / "adapters-reader", block=block)

    def prepare(direction):
        if "reader_replica" in direction:
            return False
        direction["reader_replica"] = RD.reader_replica(agent, json.loads((reader / "reader-replica.json").read_text()))
        direction["reader_layers"] = agent.reader_layers
        return True
    run = argparse.Namespace(out=str(out), threads=a.threads, test_lives=a.test_lives, direction_lives=128,
                             adapter=spec["adapter"], mac_lives=None, replica_adapter=None)
    streams = {k: spec["base"] + v for k, v in RD.STREAMS.items()}
    WS.run(run, agent=agent, streams=streams, prepare=prepare, judge=RD.judge, other=spec["other"])


def train_losses(log):
    """{iteration: train loss} of an mlx-lm log (every 10 iterations, three decimals)."""
    found = re.findall(r"Iter (\d+): Train loss ([0-9.]+)", Path(log).read_text())
    return {int(i): float(x) for i, x in found}


def same_start(first, long):
    """The first slice of a long reader against its first reader (600 iterations, same seed and batches): the train
    losses of the iterations both logged, and the largest gap (docs/LLM_NEED_LONG_READER_PROTOCOL.md)."""
    logs = [Path(first) / "reader.log", Path(long) / "reader.log"]
    if not all(x.exists() for x in logs):
        return {"iterations": 0, "max_gap": None}
    a, b = (train_losses(x) for x in logs)
    common = sorted(set(a) & set(b))
    return {"iterations": len(common), "max_gap": max((abs(a[i] - b[i]) for i in common), default=None)}


def gather(root):
    readers = {}
    for n in AGENTS:
        out = Path(root) / n / "test"
        if (out / "verdicts.json").exists():
            readers[n] = RD.judge(W.read_jsonl(out / "lives-test.jsonl.gz"),
                                  json.loads((out / "direction.json").read_text()))
    result = verdicts(readers)
    for n, spec in AGENTS.items():
        if n in result["values"]:
            result["values"][n]["same_start"] = same_start(spec["first_reader"], Path(spec["reader"]).parent / "slice-1")
    return json.loads(json.dumps(result))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--agent", choices=sorted(AGENTS), required=True)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--test-lives", type=int, default=256)
    sub.add_parser("write")
    sub.add_parser("verdicts")
    for s in sub.choices.values():
        s.add_argument("--out", default=ROOT, help="the measures go to <out>/<agent>/test")
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
        for n in AGENTS:  # each agent's measures are checked as the sixth test's are
            if (Path(a.out) / n / "test" / "verdicts.json").exists():
                WS.check(argparse.Namespace(out=str(Path(a.out) / n / "test")), judge=RD.judge)
        differs = result != json.loads(path.read_text())
        print("differs:", "yes" if differs else "none")
        if differs:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
