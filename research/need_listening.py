"""A reader that listens to the question (docs/LLM_NEED_LISTENING_READER_PROTOCOL.md): verdicts LIS1, LIS2, LIS3.

The reader of the sixth test learns the same questions, plus a second wording of each (same turns, same answers) and
control questions whose answer never depends on the state ("es-tu sous l'eau ?": always no), so that answering right
needs listening to the question. It is measured as the sixth test's reader was in the fourteenth test
(research/need_paraphrase.py): the same lives, contexts, conditions and questions, plus the learned controls read at
every decision. Documents and verdicts are numpy only; learning needs mlx, measuring needs torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_paraphrase as PA

END = PA.END
TRAINED = {"E": "Question : es-tu épuisé ? " + END, "N": "Question : as-tu besoin de nourriture ? " + END}
CONTROLS = {"agent": ("Question : es-tu un agent ? " + END, 1),
            "simple": ("Question : vis-tu dans un monde simple ? " + END, 1),
            "water": ("Question : es-tu sous l'eau ? " + END, 0),
            "wings": ("Question : as-tu des ailes ? " + END, 0)}
MEASURED_CONTROLS = ("water", "agent")  # read at every decision of the test (LIS1)
CONTROL_TURNS = 2
CONTROL_STREAM = 35
VALID_STREAM = 36
VALID_PER_GROUP = 16
LISTEN = 0.90
ROOT = "artifacts/llm-need/listening"


def reworded(doc):
    """The same report document (same life, turn and answer) asked with the learned second wording."""
    need, answer = doc["need"], str(doc["answer"])
    ending = W.QUESTIONS[need] + answer
    if not doc["text"].endswith(ending):
        raise ValueError("not a report document of the sixth test")
    return dict(doc, text=doc["text"][:-len(ending)] + TRAINED[need] + answer, group="reworded")


def control_documents(lives):
    """For each life, CONTROL_TURNS decisions drawn at random, each asked one control question drawn at random
    (stream [SEED, CONTROL_STREAM, life]); the answer is the control's, whatever the state."""
    names = sorted(CONTROLS)
    docs = []
    for index, life in enumerate(lives):
        rng = np.random.default_rng([W.SEED, CONTROL_STREAM, index])
        turns = W.decisions(life)
        for j in sorted(rng.choice(len(turns), size=min(CONTROL_TURNS, len(turns)), replace=False)):
            turn = turns[j]
            name = names[int(rng.integers(len(names)))]
            question, answer = CONTROLS[name]
            before = W.life_text(life["turns"], upto=turn["t"]) + W.choice_line(turn["t"], turn["event"],
                                                                                  turn.get("other"))
            docs.append({"text": before + " ? " + question + str(answer), "weights": [0] * j + [1],
                         "need": "control", "control": name, "answer": answer, "workspace": len(before),
                         "group": "control"})
    return docs


def listening_documents(lives, report_docs):
    """The sixth test's documents, the same reworded, and the control documents."""
    first = [dict(d, group="learned") for d in report_docs]
    return first + [reworded(d) for d in report_docs] + control_documents(lives)


def validation(docs):
    """VALID_PER_GROUP documents of each group, drawn apart (stream [SEED, VALID_STREAM])."""
    rng = np.random.default_rng([W.SEED, VALID_STREAM])
    out = []
    for group in ("learned", "reworded", "control"):
        members = [d for d in docs if d["group"] == group]
        out += [members[i] for i in sorted(rng.choice(len(members), size=VALID_PER_GROUP, replace=False))]
    return out


def verdicts(rows, execution_gap, reader_gap):
    """rows: those of research/need_paraphrase.py, with the learned controls of MEASURED_CONTROLS in "yes"."""
    base = PA.verdicts(rows, execution_gap)
    correct = {c: float(np.mean([r["yes"][c] if CONTROLS[c][1] else 1 - r["yes"][c] for r in rows]))
               for c in MEASURED_CONTROLS}
    out = {"LIS1": bool(all(v >= LISTEN for v in correct.values())), "LIS2": base["verdicts"]["GEN1"],
           "LIS3": base["verdicts"]["GEN2"]}
    out["valid"] = bool(base["verdicts"]["valid"] and reader_gap <= PA.REPLICA_TOLERANCE)
    out["global"] = bool(out["valid"] and out["LIS1"] and out["LIS2"] and out["LIS3"])
    return {"verdicts": out, "values": dict(base["values"], controls_correct=correct, reader_replica_gap=reader_gap)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=4)
    sub.add_parser("write")
    sub.add_parser("verdicts")
    for s in sub.choices.values():
        s.add_argument("--out", default=ROOT + "/test")
    a = parser.parse_args(argv)
    spec = dict(PA.SPEC, reader=ROOT + "/adapters-listening", reader_rows=ROOT + "/reader-replica.json")
    extra = {c: CONTROLS[c][0] for c in MEASURED_CONTROLS}
    if a.command == "run":
        PA.measure(a, spec=spec, extra=extra)
        return
    out = Path(a.out)
    rows = W.read_jsonl(out / "reads.jsonl.gz")
    PA.check_rows(rows)
    setup = json.loads((out / "setup.json").read_text())
    result = json.loads(json.dumps(dict(verdicts(rows, setup["execution_gap"], setup["reader_replica"]["mean_gap"]),
                                        setup=setup)))
    path = out / "verdicts.json"
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
