"""Does the adjusted language model read the mark of its body? Protocol: docs/LLM_MARK_READING_PROTOCOL.md.

At the start of a life, after one inspection of place L showing symbol S, the model is asked where command C will
take it from square p. Only its next-token distribution over the four possible landing squares is read
(renormalized), and the probability it gives to the square that S would imply if S told the body. For fixed p and
C the four symbols imply the four squares, so a model blind to the symbol gives exactly 0.25 on average.
P1: mean over place 1, the mark; P0: mean over places 2 to 4.

score: one model's rows; verdicts: L0 to L3, recomputed from the rows of every model and the inquiry test.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from .origin_env import RING, motor_delta
from .llm_atelier import COMMANDS, MARKS, inspect_line
from .llm_lora_body import completion_prompt
from .llm_latent_body import landing_positions
from .llm_inquiry import Cached, ScriptedDigits, verdicts as inquiry_verdicts
from .llm_inquiry_verdicts import load as load_inquiry

PLACES = (1, 2, 3, 4)
POSITIONS = (0, 2, 4, 6)
PUBLISHED = ("F", "VM", "VMI", "VMW")
VALIDITY_MASS = 0.5


def items():
    return [{"place": place, "symbol": s, "command": c, "p": p}
            for place in PLACES for s in range(len(MARKS)) for c in range(len(COMMANDS)) for p in POSITIONS]


def prompt(item):
    p = item["p"]
    return completion_prompt([inspect_line(0, item["place"], item["symbol"], p, (p + 4) % RING)], p, item["command"], 1)


def score(digits):
    rows = []
    for item in items():
        probs, mass = digits(prompt(item))
        landings = landing_positions(item["p"])
        over = np.array([probs[x] for x in landings])
        over = over / max(over.sum(), 1e-12)
        implied = (item["p"] + motor_delta(item["symbol"], item["command"])) % RING
        rows.append(dict(item, implied=implied, p_implied=round(float(over[landings.index(implied)]), 6),
                         best=landings[int(np.argmax(over))], digit_mass=round(float(mass), 6)))
    return rows


def summarize(rows):
    mark = [r for r in rows if r["place"] == 1]
    other = [r for r in rows if r["place"] != 1]
    return {"points": len(rows), "digit_mass": float(np.mean([r["digit_mass"] for r in rows])),
            "P1": float(np.mean([r["p_implied"] for r in mark])), "P0": float(np.mean([r["p_implied"] for r in other])),
            "accuracy_1": float(np.mean([r["best"] == r["implied"] for r in mark])),
            "accuracy_0": float(np.mean([r["best"] == r["implied"] for r in other]))}


def verdicts(reading, inquiry=None):
    """reading: label -> summary of the reading test; inquiry: {"VM": VML's inquiry summary, "F": F's}."""
    out = {f"valid_{label}": s["digit_mass"] >= VALIDITY_MASS for label, s in reading.items()}
    judged = [k for k in PUBLISHED if out.get(f"valid_{k}")]
    out["L0"] = all(reading[k]["P1"] <= 0.35 for k in judged) if judged else None
    vml = reading.get("VML")
    out["L1"] = (vml["P1"] >= 0.6 and vml["P1"] - vml["P0"] >= 0.3) if vml and out["valid_VML"] else None
    found = inquiry_verdicts(inquiry) if inquiry else {}
    out["valid_inquiry_VML"], out["valid_inquiry_F"] = found.get("valid_VM"), found.get("valid_F")
    out["L2"], out["L3"] = found.get("I1"), found.get("I2")
    out["global"] = out["L1"] is True and out["L2"] is True and out["L3"] is True
    return out


def read_rows(root):
    return [json.loads(line) for line in Path(root, "rows.jsonl").read_text().splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--out", required=True)
    sc.add_argument("--backend", choices=["mlx", "scripted"], default="scripted")
    sc.add_argument("--model", default="Qwen/Qwen3-0.6B")
    sc.add_argument("--adapter", default=None)
    sc.add_argument("--label", required=True)
    sc.add_argument("--scripted", choices=["mark", "fixed"], default="mark")
    ve = sub.add_parser("verdicts")
    ve.add_argument("--reading", nargs="+", required=True, help="LABEL=directory of a reading test")
    ve.add_argument("--inquiry-vml", default=None)
    ve.add_argument("--inquiry-f", default=None)
    ve.add_argument("--output", default=None)
    ve.add_argument("--check", default=None)
    a = parser.parse_args(argv)
    if a.command == "score":
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        receipt = {"schema": "menia-llm-mark-reading-receipt-v1", "protocol": "docs/LLM_MARK_READING_PROTOCOL.md",
                   "label": a.label, "backend": a.backend, "model": a.model, "adapter": a.adapter,
                   "started": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        if a.backend == "mlx":
            from .llm_latent_body import MLXScorer
            digits = Cached(MLXScorer(a.model, adapter_path=a.adapter, chat=False))
        else:
            digits = Cached(ScriptedDigits(a.scripted))
        rows = score(digits)
        (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary = summarize(rows)
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        receipt.update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(), model_calls=digits.calls,
                       rows_sha256=hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest())
        (out / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
        print(a.label, json.dumps(summary))
        return
    reading, same = {}, True
    for spec in a.reading:
        label, root = spec.split("=", 1)
        reading[label] = summarize(read_rows(root))
        same &= json.loads(json.dumps(reading[label])) == json.loads(Path(root, "summary.json").read_text())
    inquiry = None
    if a.inquiry_vml and a.inquiry_f:
        (vml, vml_same), (f, f_same) = load_inquiry(a.inquiry_vml), load_inquiry(a.inquiry_f)
        inquiry, same = {"VM": vml, "F": f}, same and vml_same and f_same
    result = json.loads(json.dumps({"verdicts": verdicts(reading, inquiry), "summaries_match_published": same,
                                    "reading": reading, "inquiry": inquiry}))
    print(json.dumps(result["verdicts"], indent=1), "\nsummaries match:", same)
    if a.output:
        Path(a.output).write_text(json.dumps(result, indent=1) + "\n")
    if a.check:
        differs = json.loads(Path(a.check).read_text()) != result or not same
        print("differs:", "yes" if differs else "none")
        if differs:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
