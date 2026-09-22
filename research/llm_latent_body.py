"""Is the body latent in a language model's predictions?

The model is asked, at each move of a random-action life written in the Atelier
prompt, on which square it will land if it gives the command about to be
issued. Only its next-token distribution over the eight squares is read; no
text is generated and no weight is trained. The lives are the same as the
micro-transformer's test sets R and M. Protocol: docs/LATENT_BODY_PROTOCOL.md.
"""
import argparse
import datetime
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
from .origin_env import Atelier, LIFE, RING, N_MOVE, DELTAS, motor_delta
from .llm_atelier import COMMANDS, build_prompt, move_line, inspect_line, place_rotation, machine_info, verify_manifest
from .own_action_experiment import RANDOM_SEED, CHANGE_SEED, CHANGE_STEP
from .text_atelier import run_text_lives

SETS = {"R": (RANDOM_SEED, None), "M": (CHANGE_SEED, CHANGE_STEP)}
INSTRUCTION = "On te posera une question sur ton prochain déplacement. Réponds par un seul chiffre de 0 à 7."
QUESTION = "Question : tu es sur la case {p}. Si tu donnes maintenant la commande {command}, sur quelle case arriveras-tu ?"
CONDITION = "T-implicit"


def question_prompt(history, p, command):
    return build_prompt(CONDITION, history, place_rotation(0), instruction=INSTRUCTION,
                        question=QUESTION.format(p=p, command=COMMANDS[command]))


def landing_positions(p):
    return [(p + d) % RING for d in DELTAS]


def category(command, seen_before, seen_after, changed):
    if changed:
        if command in seen_after:
            return "vu_apres"
        return "vu_avant_seulement" if command in seen_before else "nouveau"
    return "vu" if command in seen_before else "nouveau"


def score_life(scorer, life, on_row=None):
    """Replay a logged random-action life, ask the model at every move step, return the rows."""
    env = Atelier("T", life["seed"] * 1000003 + life["index"], forced_change_step=life.get("change_step"))
    obs = env.reset()
    if obs["p"] != life["tokens"][1] - 1:
        raise ValueError("life does not replay")
    history, rows = [], []
    seen_before, seen_after = set(), set()
    change = life.get("change_step")
    for t, action in enumerate(life["actions"]):
        p_before, g = obs["p"], obs["g"]
        changed = change is not None and t >= change
        if action < N_MOVE:
            prompt = question_prompt(history, p_before, action)
            started = time.perf_counter()
            probs, digit_mass = scorer(prompt)
            seconds = time.perf_counter() - started
            valid = landing_positions(p_before)
            over_valid = np.array([probs[x] for x in valid])
            over_valid = over_valid / max(over_valid.sum(), 1e-12)
            obs, reward, _ = env.step(action)
            truth = obs["p"]
            predicted = valid[int(np.argmax(over_valid))]
            rows.append({"life": life["index"], "step": t, "command": int(action), "p": int(p_before), "truth": int(truth),
                         "predicted": int(predicted), "correct": bool(predicted == truth), "probs": [round(float(v), 6) for v in probs],
                         "valid_mass": round(float(sum(probs[x] for x in valid)), 6), "digit_mass": round(float(digit_mass), 6),
                         "confidence": round(float(over_valid.max()), 6),
                         "category": category(action, seen_before, seen_after, changed), "changed": bool(changed),
                         "seconds": round(seconds, 4), "prompt_characters": len(prompt)})
            (seen_after if changed else seen_before).add(action)
            history.append(move_line(t, action, p_before, obs["p"], obs["g"], reward))
            if on_row:
                on_row(rows[-1])
        else:
            k = action - N_MOVE
            obs, _, _ = env.step(action)
            history.append(inspect_line(t, k + 1, obs["cue_value"], obs["p"], obs["g"]))
    if life.get("d_final") is not None and env.d != life["d_final"]:
        raise ValueError("life does not replay")
    return rows


def summarize(rows):
    def acc(sub):
        return float(np.mean([r["correct"] for r in sub])) if sub else None
    out = {"rows": len(rows), "accuracy": acc(rows),
           "valid_mass_mean": float(np.mean([r["valid_mass"] for r in rows])) if rows else None,
           "digit_mass_mean": float(np.mean([r["digit_mass"] for r in rows])) if rows else None,
           "by_category": {}, "confidence_when_wrong": None, "confidence_when_right": None}
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        out["by_category"][cat] = {"rows": len(sub), "accuracy": acc(sub)}
    wrong = [r for r in rows if not r["correct"]]
    right = [r for r in rows if r["correct"]]
    if wrong:
        out["confidence_when_wrong"] = float(np.mean([r["confidence"] for r in wrong]))
    if right:
        out["confidence_when_right"] = float(np.mean([r["confidence"] for r in right]))
    late = [r for r in rows if r["step"] >= 16]
    out["steps_16_23"] = {"rows": len(late), "accuracy": acc(late),
                          "by_category": {cat: acc([r for r in late if r["category"] == cat]) for cat in sorted({r["category"] for r in late})}}
    return out


def run_sets(scorer, out, episodes, sets=SETS, log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (seed, change) in sets.items():
        lives = run_text_lives(None, "random", seed, episodes, forced_change_step=change)
        rows = []
        started = time.time()
        for life in lives:
            rows += score_life(scorer, life)
            log(f"[{name}] life {life['index']} rows {len(rows)} {round(time.time() - started, 1)} s")
        (out / f"lives-{name}.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lives))
        (out / f"rows-{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary[name] = summarize(rows)
        summary[name]["seconds"] = round(time.time() - started, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


# ---------------------------------------------------------------- scorers: callable(prompt) -> (probs over 8 squares, digit mass)

class ScriptedScorer:
    """Test doubles. 'copier' repeats the last seen effect of the command; 'fixed' assumes body 0; 'uniform'."""
    LINE = re.compile(r"commande ([ABCD]), de la case (\d) à la case (\d)")
    ASK = re.compile(r"tu es sur la case (\d)\. Si tu donnes maintenant la commande ([ABCD])")

    def __init__(self, kind):
        self.kind = kind

    def __call__(self, prompt):
        p, command = self.ASK.search(prompt).groups()
        p, command = int(p), COMMANDS.index(command)
        probs = np.full(RING, 1.0 / RING)
        if self.kind == "copier":
            deltas = {COMMANDS.index(c): (int(b) - int(a)) % RING for c, a, b in self.LINE.findall(prompt)}
            if command in deltas:
                probs = np.full(RING, 0.01)
                probs[(p + deltas[command]) % RING] = 1.0
        elif self.kind == "fixed":
            probs = np.full(RING, 0.01)
            probs[(p + motor_delta(0, command)) % RING] = 1.0
        probs = probs / probs.sum()
        return probs, 1.0


class MLXScorer:
    """Apple-silicon scorer through mlx-lm: the next-token distribution over the digits 0..7. Untested off macOS."""

    def __init__(self, model_id="Qwen/Qwen3-4B-MLX-4bit", revision=None, local_path=None):
        from mlx_lm import load
        import mlx.core as mx
        self.mx = mx
        path = local_path or model_id
        if local_path is None and revision:
            from huggingface_hub import snapshot_download
            path = snapshot_download(model_id, revision=revision)
        self.path = str(path)
        self.model, self.tokenizer = load(self.path)
        self.inner = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        self.digit_ids = []
        for d in range(RING):
            ids = self.inner.encode(str(d), add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"digit {d} is not a single token")
            self.digit_ids.append(ids[0])

    def __call__(self, prompt):
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        ids = self.inner.encode(text, add_special_tokens=False)
        logits = self.model(self.mx.array([ids]))[0, -1]
        probs = self.mx.softmax(logits.astype(self.mx.float32), axis=-1)
        digits = np.array([float(probs[i].item()) for i in self.digit_ids])
        mass = float(digits.sum())
        return digits / max(mass, 1e-12), mass


def write_receipt(out, extra):
    out = Path(out)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.jsonl"))}
    receipt = {"schema": "menia-latent-body-receipt-v1", "machine": machine_info(), "files_sha256": files, **extra}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", choices=["mlx", "scripted"], default="scripted")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-MLX-4bit")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--scripted", choices=["copier", "fixed", "uniform"], default="uniform")
    a = parser.parse_args(argv)
    extra = {"backend": a.backend, "episodes_per_set": a.episodes, "sets": list(SETS),
             "started": datetime.datetime.now(datetime.timezone.utc).isoformat(), "protocol": "docs/LATENT_BODY_PROTOCOL.md"}
    if a.backend == "mlx":
        scorer = MLXScorer(a.model, a.revision)
        extra["model"] = {"repository": a.model, "revision": a.revision, "path": scorer.path}
        if a.manifest:
            extra["manifest_check"] = verify_manifest(scorer.path, a.manifest)
        try:
            import mlx_lm
            extra["mlx_lm_version"] = getattr(mlx_lm, "__version__", None)
        except ImportError:
            pass
    else:
        scorer = ScriptedScorer(a.scripted)
        extra["scripted"] = a.scripted
    summary = run_sets(scorer, a.out, a.episodes)
    extra["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_receipt(a.out, extra)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
