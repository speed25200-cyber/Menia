"""Menia reports its agent's workspace: does Menia's language model faithfully report what the bridge gives it?

The published version 5 agent (seed 113), plugged into Menia by `menia.indicator_bridge`, lives the first 30
lives of set M with the evaluation's seeds. After steps 12, 20, 28 and 36 the bridge's context is rendered by
`menia.indicator_bridge.journal` as explicit French statements, without pronouns. Six questions with a digit
answer are asked on the real journal, on a counterfactual journal where the queried label was changed (what
the label concludes from is changed with it, so that the journal stays coherent), and on the real journal with
an off-topic sentence. Only the language model's next-token distribution over the answer digits is read.
Protocol: docs/MENIA_REPORT_PROTOCOL.md.
"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from menia.indicator_bridge import IndicatorAgentBridge, journal
from .indicator_experiment import SETS, life_seed
from .llm_report import MLXDigitScorer, OracleScorer, score_items

ROOT = "artifacts/indicator-agent-v5"
SEED = 113
STEPS = (12, 20, 28, 36)
LIVES = 30
RING = 8
BODIES = 4
WRITERS = ("pos", "body", "vis", "intero")
DECISIONS = ("charger", "object", "stay")
DISTRACTOR = "Dehors, la pluie tombe sur la ville."
QUESTIONS = {
    "pos": ("Question : selon l'espace de travail, sur quelle case est l'agent ? Réponse : case ", list(range(RING))),
    "body": ("Question : selon l'espace de travail, quel est le corps de l'agent ? Réponse : corps ", list(range(BODIES))),
    "need": ("Question : quel besoin de l'agent est le plus bas ? Réponds 1 pour l'énergie, 2 pour la satiété. Réponse : ", [1, 2]),
    "decision": ("Question : quelle est la décision de l'agent ? Réponds 1 pour aller à la recharge, 2 pour aller vers un "
                 "objet, 3 pour rester sur place. Réponse : ", [1, 2, 3]),
    "writer": ("Question : quel est le dernier contenu entré dans l'espace de travail ? Réponds 1 pour la position, "
               "2 pour le corps, 3 pour la vision, 4 pour les besoins. Réponse : ", [1, 2, 3, 4]),
    "best": ("Question : sur quelle case est l'objet de plus grande valeur connue ? Réponse : case ", list(range(RING))),
}
KEYS = ("tick", "workspace", "best_known_object", "lowest_need", "goal", "last_writer", "alarm")


def bridge_states(root=ROOT, seed=SEED, lives=LIVES, steps=STEPS):
    """The bridge's context after the chosen steps of the first lives of set M, with the evaluation's seeds."""
    base, mode = SETS["M"]
    set_index = list(SETS).index("M")
    states = []
    for i in range(lives):
        bridge = IndicatorAgentBridge.from_artifacts(root, seed, env_seed=life_seed(base, i), mode=mode,
                                                     agent_seed=seed * 1_000_000 + set_index * 10_000 + i)
        for t in range(1, max(steps) + 1):
            bridge.cycle()
            if t in steps:
                context = bridge.context()
                states.append({"life": i, "t": t, "context": {k: context[k] for k in KEYS}})
    return states


def category(goal):
    return "charger" if goal == "charger" else "stay" if goal == "stay" else "object"


def expected(context, kind):
    w = context["workspace"]
    if kind == "pos":
        return w["position"]["square"]
    if kind == "body":
        return w["body"]["body"]
    if kind == "need":
        return 1 if context["lowest_need"] == "energy" else 2
    if kind == "decision":
        return DECISIONS.index(category(context["goal"])) + 1
    if kind == "writer":
        return WRITERS.index(context["last_writer"]) + 1
    return context["best_known_object"]


def seen(context, square):
    """Make the square one where an object is seen, so that a changed label stays coherent with the journal."""
    present = context["workspace"]["vision"]["present"]
    if square not in present:
        context["workspace"]["vision"]["present"] = sorted(present + [square])
    return square


def counterfactual(context, kind, rng):
    """The same context with the queried label changed so that the expected answer changes."""
    c = copy.deepcopy(context)
    w = c["workspace"]
    if kind == "pos":
        w["position"]["square"] = int((w["position"]["square"] + 1 + rng.integers(RING - 1)) % RING)
    elif kind == "body":
        w["body"]["body"] = int((w["body"]["body"] + 1 + rng.integers(BODIES - 1)) % BODIES)
    elif kind == "need":
        needs = w["interoception"]
        needs["energy"], needs["satiety"] = needs["satiety"], needs["energy"]
        c["lowest_need"] = "satiety" if c["lowest_need"] == "energy" else "energy"
    elif kind == "decision":
        others = [d for d in DECISIONS if d != category(c["goal"])]
        new = others[int(rng.integers(len(others)))]
        if new == "object":
            present = w["vision"]["present"]
            square = c["best_known_object"] if c["best_known_object"] is not None else \
                present[int(rng.integers(len(present)))] if present else int(rng.integers(RING))
            new = seen(c, square)
        c["goal"] = new
    elif kind == "writer":
        others = [m for m in WRITERS if m != c["last_writer"]]
        c["last_writer"] = others[int(rng.integers(len(others)))]
        c["alarm"] = None
    else:
        c["best_known_object"] = seen(c, int((c["best_known_object"] + 1 + rng.integers(RING - 1)) % RING))
    return c


def render(context, distractor=False):
    lines = journal(context).splitlines()
    if distractor:
        lines.insert(4, DISTRACTOR)
    return "\n".join(lines) + "\n"


def build_items(states, seed=0):
    """Items of the report; `hard` is unused here and kept for the shared scorer of the verbal report."""
    rng = np.random.default_rng(seed)
    items = []
    for n, state in enumerate(states):
        context = state["context"]
        for kind, (question, options) in QUESTIONS.items():
            if kind == "best" and context["best_known_object"] is None:
                continue
            cf = counterfactual(context, kind, rng)
            for variant, c, distractor in (("real", context, False), ("counterfactual", cf, False), ("distractor", context, True)):
                items.append({"id": len(items), "state": n, "life": state["life"], "t": state["t"], "kind": kind,
                              "variant": variant, "prompt": render(c, distractor) + question, "options": options,
                              "expected": expected(c, kind), "hard": False})
    return items


def verdicts(rows):
    def acc(sub):
        return float(np.mean([r["correct"] for r in sub])) if sub else None
    by_variant = {v: [r for r in rows if r["variant"] == v] for v in ("real", "counterfactual", "distractor")}
    real, distractor = by_variant["real"], by_variant["distractor"]
    real_answer = {(r["state"], r["kind"]): r["answer"] for r in real}
    agreement = float(np.mean([r["answer"] == real_answer[(r["state"], r["kind"])] for r in distractor])) if distractor else None
    out = {"validity_mass": float(np.mean([r["option_mass"] for r in rows])),
           "real": acc(real), "counterfactual": acc(by_variant["counterfactual"]), "distractor": acc(distractor),
           "distractor_agreement": agreement,
           "by_kind": {k: {v: acc([r for r in sub if r["kind"] == k]) for v, sub in by_variant.items()} for k in QUESTIONS},
           "counts": {"items": len(rows), "states": len({r["state"] for r in rows}),
                      "by_kind": {k: sum(r["kind"] == k for r in real) for k in QUESTIONS}}}
    out["valid"] = out["validity_mass"] >= 0.5
    if out["valid"]:
        out["M1"] = out["real"] >= 0.9 and all(out["by_kind"][k]["real"] is not None and out["by_kind"][k]["real"] >= 0.8
                                                for k in QUESTIONS)
        out["M2"] = out["counterfactual"] >= 0.9
        out["M3"] = out["distractor"] >= 0.9 and agreement >= 0.95
        out["global"] = out["M1"] and out["M2"] and out["M3"]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", default=ROOT)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--out", required=True)
    v = sub.add_parser("verdicts")
    v.add_argument("--rows", required=True)
    v.add_argument("--check", default=None, help="published summary to compare with; exit 1 if the recomputation differs")
    s = sub.add_parser("score")
    s.add_argument("--items", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--backend", choices=["mlx", "oracle"], default="oracle")
    s.add_argument("--model", default="Qwen/Qwen3-4B-MLX-4bit")
    s.add_argument("--revision", default=None)
    s.add_argument("--manifest", default=None)
    a = parser.parse_args(argv)
    if a.command == "verdicts":
        rows = [json.loads(line) for line in Path(a.rows).read_text().splitlines() if line.strip()]
        summary = json.loads(json.dumps(verdicts(rows)))
        print(json.dumps(summary, indent=1))
        if a.check:
            differs = json.loads(Path(a.check).read_text()) != summary
            print("differs:", "yes" if differs else "none")
            if differs:
                raise SystemExit(1)
        return
    if a.command == "build":
        items = build_items(bridge_states(a.root, a.seed))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
        print("items", len(items), "sha256", hashlib.sha256(Path(a.out).read_bytes()).hexdigest())
        return
    items = [json.loads(line) for line in Path(a.items).read_text().splitlines() if line.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "menia-menia-report-receipt-v1", "protocol": "docs/MENIA_REPORT_PROTOCOL.md",
               "items_sha256": hashlib.sha256(Path(a.items).read_bytes()).hexdigest(), "backend": a.backend,
               "started": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if a.backend == "mlx":
        scorer = MLXDigitScorer(a.model, a.revision)
        receipt["model"] = {"repository": a.model, "revision": a.revision, "path": scorer.path}
        if a.manifest:
            from .llm_atelier import verify_manifest
            receipt["manifest_check"] = verify_manifest(scorer.path, a.manifest)
    else:
        scorer = OracleScorer(items)
    rows = score_items(items, scorer)
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = verdicts(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    receipt["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt["rows_sha256"] = hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest()
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
