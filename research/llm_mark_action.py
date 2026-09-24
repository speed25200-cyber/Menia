"""Does what the adjusted language model knows of itself earn it points? Protocol: docs/LLM_MARK_ACTION_PROTOCOL.md.

The model acts in the world of VML (the body redrawn after each move): each move is preceded by an inspection of
place 1, the mark, and the command is the one the model's own next-token distributions make best. When the target
can be reached in one move, the best command is the one most likely to land on it; otherwise, the one whose landing
is expected closest to it. Ties are broken at random. Ablation ("lesion"): the inspection line shows a random
symbol instead of the mark, so the same model acts without the information on its body.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from .origin_env import Atelier, LIFE, N_MOVE, RING, ring_distance
from .llm_atelier import inspect_line, move_line
from .llm_lora_body import completion_prompt
from .llm_latent_body import landing_positions
from .llm_inquiry import Cached, ScriptedDigits

SEED = 930001
CONDITIONS = ("intact", "lesion")


class PartialReader(ScriptedDigits):
    """Test double: reads the mark like ScriptedDigits("mark") for commands A to C, knows nothing for D."""

    def __init__(self):
        super().__init__("mark")

    def __call__(self, prompt):
        if self.TAIL.search(prompt).group(1) == "D":
            probs = np.full(10, 0.001)
            for x in landing_positions(int(self.TAIL.search(prompt).group(2))):
                probs[x] = 1.0
            return probs / probs.sum(), 1.0
        return super().__call__(prompt)


def choose(digits, history, p, g, t, rng):
    """The command the model's own predictions make best, and its predicted probability of landing on the target."""
    landings = landing_positions(p)
    reachable = g in landings
    values, hits = [], []
    for command in range(N_MOVE):
        probs, _ = digits(completion_prompt(history, p, command, t))
        over = np.array([probs[x] for x in landings])
        over = over / max(over.sum(), 1e-12)
        hits.append(float(over[landings.index(g)]) if reachable else 0.0)
        values.append(hits[-1] if reachable else -float(sum(q * ring_distance(x, g) for q, x in zip(over, landings))))
    best = [c for c in range(N_MOVE) if values[c] >= max(values) - 1e-9]
    command = int(best[int(rng.integers(len(best)))])
    return command, hits[command]


def play(digits, index, condition, seed=SEED):
    """One life of twelve inspections of the mark, each followed by a move; returns the points and the moves."""
    env = Atelier("T", seed * 1000003 + index)
    obs = env.reset()
    bodies = np.random.default_rng([seed, index, 1])
    lesion = np.random.default_rng([seed, index, 2])
    ties = np.random.default_rng([seed, index, 3])
    history, moves, points = [], [], 0
    for pair in range(LIFE // 2):
        if pair:
            env.d = int(bodies.integers(4))
        t = 2 * pair
        obs, _, _ = env.step(N_MOVE)
        shown = obs["cue_value"] if condition == "intact" else int(lesion.integers(4))
        history.append(inspect_line(t, 1, shown, obs["p"], obs["g"]))
        p, g = obs["p"], obs["g"]
        command, predicted = choose(digits, history, p, g, t + 1, ties)
        obs, reward, _ = env.step(command)
        history.append(move_line(t + 1, command, p, obs["p"], obs["g"], reward))
        points += reward
        moves.append({"reachable": g in landing_positions(p), "command": command, "predicted_hit": round(predicted, 6),
                      "hit": bool(reward), "body": env.d, "shown": shown})
    return points, moves


def run(digits, label, lives, conditions=CONDITIONS, seed=SEED):
    rows = []
    for condition in conditions:
        for index in range(lives):
            points, moves = play(digits, index, condition, seed)
            rows.append({"label": label, "condition": condition, "life": index, "points": points, "moves": moves})
    return rows


def summarize(rows):
    out = {}
    for condition in sorted({r["condition"] for r in rows}):
        rs = [r for r in rows if r["condition"] == condition]
        reachable = [m for r in rs for m in r["moves"] if m["reachable"]]
        out[condition] = {"lives": len(rs), "points_per_life": float(np.mean([r["points"] for r in rs])),
                          "hit_when_reachable": float(np.mean([m["hit"] for m in reachable])) if reachable else None,
                          "predicted_hit_when_reachable": float(np.mean([m["predicted_hit"] for m in reachable]))
                          if reachable else None}
    return out


def paired_difference(rows_a, rows_b, draws=10000, seed=0):
    """Mean difference of points per life between two runs on the same lives, and its bootstrap 95% interval."""
    a = {r["life"]: r["points"] for r in rows_a}
    b = {r["life"]: r["points"] for r in rows_b}
    diffs = np.array([a[k] - b[k] for k in sorted(a)], float)
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(len(diffs), size=(draws, len(diffs)))].mean(axis=1)
    return {"mean": float(diffs.mean()), "low": float(np.quantile(means, 0.025)), "high": float(np.quantile(means, 0.975))}


THRESHOLD = 2.5  # points per life, half of what a perfect reader gains from the mark (docs/LLM_MARK_ACTION_PROTOCOL.md)


def read_rows(root):
    return [json.loads(line) for line in Path(root, "rows.jsonl").read_text().splitlines() if line.strip()]


def action_verdicts(code, control):
    """U1: the reader earns points from its mark (intact minus lesion); U2: from its reading (reader minus a model of
    the same world that does not read, both intact). Each: mean paired difference >= THRESHOLD and 95% low bound > 0."""
    part = lambda rows, c: [r for r in rows if r["condition"] == c]
    u1 = paired_difference(part(code, "intact"), part(code, "lesion"))
    u2 = paired_difference(part(code, "intact"), part(control, "intact"))
    passes = lambda d: d["mean"] >= THRESHOLD and d["low"] > 0
    return {"U1": passes(u1), "U2": passes(u2), "global": passes(u1) and passes(u2),
            "intact_minus_lesion": u1, "reader_minus_control": u2}


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["verdicts"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("--code", required=True)
        parser.add_argument("--control", required=True)
        parser.add_argument("--output", default=None)
        parser.add_argument("--check", default=None)
        a = parser.parse_args(argv[1:])
        code, control = read_rows(a.code), read_rows(a.control)
        same = all(json.loads(json.dumps(summarize(rows))) == json.loads(Path(root, "summary.json").read_text())
                   for rows, root in ((code, a.code), (control, a.control)))
        result = json.loads(json.dumps({"verdicts": action_verdicts(code, control), "summaries_match_published": same,
                                        "code": summarize(code), "control": summarize(control)}))
        print(json.dumps(result["verdicts"], indent=1), "\nsummaries match:", same)
        if a.output:
            Path(a.output).write_text(json.dumps(result, indent=1) + "\n")
        if a.check:
            differs = json.loads(Path(a.check).read_text()) != result or not same
            print("differs:", "yes" if differs else "none")
            if differs:
                raise SystemExit(1)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", choices=["mlx", "scripted"], default="scripted")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--scripted", choices=["mark", "fixed", "mark3"], default="mark")
    parser.add_argument("--lives", type=int, default=48)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    a = parser.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "menia-llm-mark-action-receipt-v1", "protocol": "docs/LLM_MARK_ACTION_PROTOCOL.md",
               "label": a.label, "backend": a.backend, "model": a.model, "adapter": a.adapter, "seed": SEED,
               "started": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if a.backend == "mlx":
        from .llm_latent_body import MLXScorer
        digits = Cached(MLXScorer(a.model, adapter_path=a.adapter, chat=False))
    else:
        digits = Cached(PartialReader() if a.scripted == "mark3" else ScriptedDigits(a.scripted))
    rows = run(digits, a.label, a.lives, a.conditions)
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    receipt.update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(), model_calls=digits.calls,
                   rows_sha256=hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest())
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(a.label, json.dumps(summary))


if __name__ == "__main__":
    main()
