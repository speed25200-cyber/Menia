"""The verbal report: can a language model faithfully report the indicator agent's inner states?

The published agent (version 1, seed 17) lives the lives of set M; at fixed steps its inner state is
written as a short journal: believed square, believed body, where the attention schema says the
spotlight landed and where the agent aimed, the monitor's trust in the last position reading, the
two needs, the last module written into the workspace, the current goal. A language model reads the
journal and a question, in raw completion, and only its next-token distribution over the answer
digits is read. Each question is asked on the real journal, on a counterfactual journal where the
queried state was changed, and on the real journal with an irrelevant sentence added.
Protocol: docs/LLM_REPORT_PROTOCOL.md.
"""
import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .indicator_agent import Params, MODULES
from .indicator_experiment import run_life, life_seed, SETS

STEPS = (12, 20, 28, 36)
LIVES = 30
MODULE_NAMES = {"pos": "Pos", "body": "Corps", "vis": "Vision", "intero": "Intéro"}
DISTRACTOR = "Dehors, il pleut sur la ville."
QUESTIONS = {
    "pos": ("Question : sur quelle case l'agent croit-il être ? Réponse : case ", list(range(8))),
    "body": ("Question : quel corps l'agent croit-il avoir ? Réponse : corps ", list(range(4))),
    "spot": ("Question : selon son schéma d'attention, sur quelle case le projecteur de l'agent a-t-il réellement atterri ? "
             "Réponse : case ", list(range(8))),
    "trust": ("Question : l'agent doit-il se fier à sa dernière lecture de position ? Réponds 1 pour oui, 0 pour non. "
              "Réponse : ", [0, 1]),
    "need": ("Question : quel besoin de l'agent est le plus bas ? Réponds 1 pour l'énergie, 2 pour la satiété. Réponse : ", [1, 2]),
    "module": ("Question : quel module a écrit en dernier dans l'espace de travail ? Réponds 1 pour Pos, 2 pour Corps, "
               "3 pour Vision, 4 pour Intéro. Réponse : ", [1, 2, 3, 4]),
}


def goal_text(goal):
    if goal == "charger":
        return "aller se recharger"
    if goal == "stay":
        return "rester sur place"
    return f"aller vers l'objet de la case {goal}"


def journal(state, distractor=False):
    lines = [f"Journal intérieur de l'agent, pas {state['t']}.",
             f"Position crue : case {state['pos']} (confiance {state['pos_conf']:.2f}).",
             f"Corps cru : corps {state['body']} (confiance {state['body_conf']:.2f}).",
             f"Projecteur d'attention : l'agent visait la case {state['intent']} ; son schéma d'attention estime qu'il a "
             f"atterri sur la case {state['spot']}.",
             f"Dernière lecture de position : case {state['read']} ; fiabilité estimée par le moniteur : {state['trust']:.2f}.",
             f"Énergie : {state['energy']:.2f}. Satiété : {state['satiety']:.2f}.",
             f"Dernier module écrit dans l'espace de travail : {MODULE_NAMES[state['module']]}.",
             f"But courant : {goal_text(state['goal'])}."]
    if distractor:
        lines.insert(4, DISTRACTOR)
    return "\n".join(lines) + "\n"


def expected(state, kind):
    if kind == "pos":
        return state["pos"]
    if kind == "body":
        return state["body"]
    if kind == "spot":
        return state["spot"]
    if kind == "trust":
        return int(state["trust"] >= 0.5)
    if kind == "need":
        return 1 if state["energy"] < state["satiety"] else 2
    return MODULES.index(state["module"]) + 1


def counterfactual(state, kind, rng):
    """The same journal with the queried state changed so that the expected answer changes."""
    s = dict(state)
    if kind == "pos":
        s["pos"] = int((s["pos"] + 1 + rng.integers(7)) % 8)
    elif kind == "body":
        s["body"] = int((s["body"] + 1 + rng.integers(3)) % 4)
    elif kind == "spot":
        s["spot"] = int((s["spot"] + 1 + rng.integers(7)) % 8)
    elif kind == "trust":
        s["trust"] = round(float(1.0 - s["trust"]), 2) if abs(s["trust"] - 0.5) > 0.05 else 0.9
    elif kind == "need":
        s["energy"], s["satiety"] = s["satiety"], s["energy"]
    else:
        others = [m for m in MODULES if m != s["module"]]
        s["module"] = others[int(rng.integers(len(others)))]
    return s


def agent_states(root, seed, lives=LIVES, steps=STEPS):
    """Replay the published agent on set M and take its inner state at the chosen steps (reading present)."""
    params = Params.load(Path(root) / f"params-{seed}.json")
    base, mode = SETS["M"]
    states = []
    for i in range(lives):
        life = run_life(params, "agent", life_seed(base, i), mode, seed * 1_000_000 + 1 * 10_000 + i)
        env = life["env"]
        for rec, truth in life["steps"]:
            t = truth["t"]
            if t not in steps or rec.get("reliability") is None or rec.get("schema_estimate") is None:
                continue
            read = int(env.glitch_square[t]) if env.glitch[t] else int(truth["p"])
            states.append({"life": i, "t": t, "pos": rec["pos_belief"], "pos_conf": rec["pos_confidence"],
                           "body": rec["body_belief"], "body_conf": rec["body_confidence"], "intent": rec["intent"],
                           "spot": rec["schema_estimate"], "read": read, "trust": rec["reliability"],
                           "energy": rec["energy_estimate"], "satiety": rec["satiety_estimate"],
                           "module": rec["writers"][0], "goal": rec["goal"], "captured_estimate": rec["schema_estimate"] != rec["intent"]})
    return states


def build_items(states, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for n, state in enumerate(states):
        state = dict(state)
        for kind, (question, options) in QUESTIONS.items():
            cf = counterfactual(state, kind, rng)
            for variant, s, distractor in (("real", state, False), ("counterfactual", cf, False), ("distractor", state, True)):
                items.append({"id": len(items), "state": n, "kind": kind, "variant": variant,
                              "prompt": journal(s, distractor) + question, "options": options, "expected": expected(s, kind),
                              "hard": bool(kind == "spot" and state["captured_estimate"])})
    return items


def score_items(items, scorer, log=print):
    rows = []
    started = time.time()
    for item in items:
        probs, mass = scorer(item["prompt"])
        over = np.array([probs[o] for o in item["options"]])
        answer = item["options"][int(np.argmax(over))]
        rows.append({"id": item["id"], "state": item["state"], "kind": item["kind"], "variant": item["variant"],
                     "hard": item["hard"], "expected": item["expected"], "answer": int(answer),
                     "correct": bool(answer == item["expected"]), "option_mass": round(float(over.sum() * mass), 6)})
        if len(rows) % 200 == 0:
            log(f"scored {len(rows)} items in {round(time.time() - started, 1)} s")
    return rows


def verdicts(rows):
    def acc(sub):
        return float(np.mean([r["correct"] for r in sub])) if sub else None
    real = [r for r in rows if r["variant"] == "real"]
    by_kind = {k: acc([r for r in real if r["kind"] == k]) for k in QUESTIONS}
    real_answer = {(r["state"], r["kind"]): r["answer"] for r in real}
    distractor = [r for r in rows if r["variant"] == "distractor"]
    agreement = float(np.mean([r["answer"] == real_answer[(r["state"], r["kind"])] for r in distractor])) if distractor else None
    out = {"validity_mass": float(np.mean([r["option_mass"] for r in rows])), "real": acc(real), "real_by_kind": by_kind,
           "counterfactual": acc([r for r in rows if r["variant"] == "counterfactual"]), "distractor": acc(distractor),
           "distractor_agreement": agreement, "hard_spot": acc([r for r in real if r["hard"]]),
           "counts": {"items": len(rows), "hard_spot": sum(r["hard"] for r in real)}}
    out["valid"] = out["validity_mass"] >= 0.5
    if out["valid"]:
        out["R1"] = out["real"] >= 0.9 and all(v is not None and v >= 0.8 for v in by_kind.values())
        out["R2"] = out["counterfactual"] >= 0.9
        out["R3"] = out["distractor"] >= 0.9 and agreement >= 0.95
        out["R4"] = out["hard_spot"] is not None and out["hard_spot"] >= 0.8
        out["global"] = out["R1"] and out["R2"] and out["R3"]
    return out


class MLXDigitScorer:
    """Next-token distribution over the ten digits, raw completion, mlx-lm on Apple silicon."""

    def __init__(self, model_id, revision=None):
        from .llm_latent_body import MLXScorer
        self.inner = MLXScorer(model_id, revision, chat=False)
        self.path = self.inner.path
        tok = self.inner.inner
        self.ids = []
        for d in range(10):
            ids = tok.encode(str(d), add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError("digit is not a single token")
            self.ids.append(ids[0])
        self.mx = self.inner.mx

    def __call__(self, prompt):
        ids = self.inner.inner.encode(prompt, add_special_tokens=False)
        logits = self.inner.model(self.mx.array([ids]))[0, -1]
        probs = self.mx.softmax(logits.astype(self.mx.float32), axis=-1)
        digits = np.array([float(probs[i].item()) for i in self.ids])
        mass = float(digits.sum())
        return digits / max(mass, 1e-12), mass


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", default="artifacts/indicator-agent")
    b.add_argument("--seed", type=int, default=17)
    b.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--items", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--backend", choices=["mlx", "oracle"], default="oracle")
    s.add_argument("--model", default="Qwen/Qwen3-4B-MLX-4bit")
    s.add_argument("--revision", default=None)
    s.add_argument("--manifest", default=None)
    a = parser.parse_args(argv)
    if a.command == "build":
        items = build_items(agent_states(a.root, a.seed))
        Path(a.out).write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
        print("items", len(items), "sha256", hashlib.sha256(Path(a.out).read_bytes()).hexdigest())
        return
    items = [json.loads(line) for line in Path(a.items).read_text().splitlines() if line.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "menia-llm-report-receipt-v1", "protocol": "docs/LLM_REPORT_PROTOCOL.md",
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


class OracleScorer:
    """Test double: knows the expected answer of each prompt."""

    def __init__(self, items):
        self.answers = {i["prompt"]: i["expected"] for i in items}

    def __call__(self, prompt):
        probs = np.full(10, 0.001)
        probs[self.answers[prompt]] = 1.0
        return probs / probs.sum(), 0.9


if __name__ == "__main__":
    main()
