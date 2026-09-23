"""Verdicts of the inquiry test (docs/LLM_INQUIRY_PROTOCOL.md), recomputed from the published rows of each model.

The summaries are recomputed from rows.jsonl with llm_inquiry.summarize and compared with the published
summary.json; the verdicts are llm_inquiry.verdicts of those summaries. --check compares with a published file.
"""
import argparse
import json
from pathlib import Path
from .llm_inquiry import summarize, verdicts


def load(root):
    """The summary recomputed from the rows (JSON turned the places into strings), and whether it is the published one."""
    rows = [json.loads(line) for line in Path(root, "rows.jsonl").read_text().splitlines() if line.strip()]
    for row in rows:
        row["gains"] = {int(k): v for k, v in row["gains"].items()}
    summary = summarize(rows)
    return summary, json.loads(json.dumps(summary)) == json.loads(Path(root, "summary.json").read_text())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--vm", required=True, help="directory with the rows and summary of the VM model")
    parser.add_argument("--f", required=True, help="directory with the rows and summary of the F model")
    parser.add_argument("--output", default=None)
    parser.add_argument("--check", default=None, help="published verdicts to compare with; exit 1 if they differ")
    a = parser.parse_args(argv)
    (vm, vm_same), (f, f_same) = load(a.vm), load(a.f)
    result = json.loads(json.dumps({"verdicts": verdicts({"VM": vm, "F": f}), "summaries_match_published": vm_same and f_same,
                                    "VM": vm, "F": f}))
    print(json.dumps(result["verdicts"], indent=1), "\nsummaries match:", result["summaries_match_published"])
    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=1) + "\n")
    if a.check:
        differs = json.loads(Path(a.check).read_text()) != result or not result["summaries_match_published"]
        print("differs:", "yes" if differs else "none")
        if differs:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
