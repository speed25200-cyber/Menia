"""A reader without shortcut (docs/LLM_NEED_BALANCED_READER_PROTOCOL.md): verdicts R8, SAY8, ONE8.

The measures are those of the sixth test (research/need_reader.py), with the reader learned on documents balanced
within each event, the directions of the sixth test and the test lives of stream 30. Numpy only.
"""
import argparse
import json
from pathlib import Path
from . import need_world as W
from . import need_verdicts as V
from . import need_reader as RD
from . import need_one as O


def verdicts(reader):
    """reader: the sixth test's result (research.need_reader.verdicts) on the new reader."""
    v = reader["values"]
    say_n = v["food_low"]["say_yes_food"]
    out = {"R8": bool(reader["verdicts"]["R6"]),
           "SAY8": bool(say_n["mean"] >= O.REPORT_EFFECT and say_n["low"] > 0
                        and abs(v["random"]["say_N"]) <= V.SHARE * say_n["mean"]),
           "ONE8": bool(reader["verdicts"]["ONE6"]),
           "valid": bool(reader["verdicts"]["valid"])}
    out["global"] = bool(out["valid"] and out["R8"] and out["SAY8"] and out["ONE8"])
    return {"verdicts": out, "values": v}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "verdicts"))
    parser.add_argument("--out", default="artifacts/llm-need/balanced/test")
    a = parser.parse_args(argv)
    out = Path(a.out)
    direction = json.loads((out / "direction.json").read_text())
    result = json.loads(json.dumps(verdicts(RD.judge(W.read_jsonl(out / "lives-test.jsonl.gz"), direction))))
    path = out / "balanced-verdicts.json"
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
