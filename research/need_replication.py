"""The replication on a second agent learned from scratch (docs/LLM_NEED_REPLICATION_PROTOCOL.md).

The measures are those of the sixth and seventh tests (research/need_reader.py, research/need_necessity.py) run with
the second agent, its reader and the streams 100 + s; this module draws the replication's verdicts from their outputs.
Numpy only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_reader as RD
from . import need_necessity as NC


def verdicts(reader, necessity):
    """reader: the sixth test's result on the second agent; necessity: the seventh's."""
    rv, nv = reader["values"], necessity["values"]
    out = {"RR": bool(rv["report_accuracy"]["E"] is not None and rv["report_accuracy"]["E"] >= 0.75),
           "RA": bool(rv["survival"] >= 0.55),
           "RONE": bool(reader["verdicts"]["ONE6"]),
           "RLS": bool(necessity["verdicts"]["LS7"]),
           "RLR": bool(necessity["verdicts"]["LR7"])}
    out["valid"] = bool(reader["verdicts"]["valid"] and necessity["verdicts"]["valid"])
    out["global"] = bool(out["valid"] and all(out[k] for k in ("RR", "RA", "RONE", "RLS", "RLR")))
    values = {"report_accuracy": rv["report_accuracy"], "survival": rv["survival"],
              "energy_low": rv["energy_low"], "food_low": rv["food_low"], "random": rv["random"],
              "survival_by_arm": nv["survival"], "survival_drop": nv["survival_drop"],
              "survival_drop_random": nv["survival_drop_random"], "say_energy": nv["say_energy"],
              "say_food": nv["say_food"]}
    return {"verdicts": out, "values": values}


def compute(root):
    root = Path(root)
    test, necessity = root / "reader" / "test", root / "necessity"
    direction = json.loads((test / "direction.json").read_text())
    reader = RD.judge(W.read_jsonl(test / "lives-test.jsonl.gz"), direction)
    setup = json.loads((necessity / "setup.json").read_text())
    arms = {n: W.read_jsonl(necessity / f"lives-test-{n}.jsonl.gz") for n in NC.ARMS}
    return verdicts(reader, NC.verdicts(arms, setup))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "verdicts"))
    parser.add_argument("--root", default="artifacts/llm-need/r1")
    a = parser.parse_args(argv)
    result = json.loads(json.dumps(compute(a.root)))
    path = Path(a.root) / "replication-verdicts.json"
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
