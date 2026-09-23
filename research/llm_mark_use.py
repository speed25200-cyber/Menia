"""Does the adjusted LLM read the mark of its body? Exploratory, declared in docs/LLM_INQUIRY_RESULTS.md (VMW).

On the first move of each evaluation life of the adjusted-body cells, the probability given to the true landing
square, split by whether the mark of place 1 was read before that move. Chance is 0.25.
"""
import argparse
import json
from pathlib import Path
from .origin_env import N_MOVE

RUNS = {"VM": "artifacts/llm-lora-mac/run-4-vm/VM", "VMI": "artifacts/llm-lora-mac/run-5-vmi/VMI",
        "VMW": "artifacts/llm-lora-mac/run-7-vmw/VMW"}


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def first_moves(root):
    out = {"read": [], "not_read": []}
    for s in ("R", "M"):
        lives = {life["index"]: life for life in read(f"{root}/lives-{s}.jsonl")}
        for row in read(f"{root}/rows-{s}.jsonl"):
            before = lives[row["life"]]["actions"][:row["step"]]
            if any(a < N_MOVE for a in before):
                continue
            out["read" if N_MOVE in before else "not_read"].append(row["probs"][row["truth"]])
    return {k: {"moves": len(v), "p_true": round(sum(v) / len(v), 4) if v else None} for k, v in out.items()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/llm-lora-mac/mark-use.json")
    parser.add_argument("--check", action="store_true")
    a = parser.parse_args(argv)
    result = {label: first_moves(root) for label, root in RUNS.items()}
    text = json.dumps(result, indent=1) + "\n"
    print(text)
    if a.check:
        if Path(a.output).read_text() != text:
            raise SystemExit("mark-use differs from the published file")
        return
    Path(a.output).write_text(text)


if __name__ == "__main__":
    main()
