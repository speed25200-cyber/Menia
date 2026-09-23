"""Where would the adjusted language model look to learn about itself?

For each inspectable place, the model's own distributions give how much inspecting it would reduce
its uncertainty about what its commands do: the entropy of its predicted landing squares now, minus
the entropy expected after each symbol the place could show, weighted by how likely the model finds
that symbol. No action is taken and no text is generated; only next-token distributions are read.
In the Atelier, place 1 carries the mark of the body. Protocol: docs/LLM_INQUIRY_PROTOCOL.md.
"""
import argparse
import datetime
import hashlib
import json
import math
import re
import time
from pathlib import Path
import numpy as np
from .origin_env import Atelier, N_MOVE, RING, motor_delta
from .llm_atelier import COMMANDS, MARKS, move_line, inspect_line
from .llm_lora_body import HEADER, completion_prompt
from .llm_latent_body import landing_positions
from .own_action_experiment import RANDOM_SEED, CHANGE_SEED, CHANGE_STEP
from .text_atelier import run_text_lives

PLACES = (1, 2, 3, 4)
R_LIVES = 48
M_LIVES = 24


def replay_until(life, stop):
    """History lines of a logged life up to step `stop` (excluded), with the position and target at that step."""
    env = Atelier("T", life["seed"] * 1000003 + life["index"], forced_change_step=life.get("change_step"))
    obs = env.reset()
    history = []
    for t, action in enumerate(life["actions"][:stop]):
        p_before = obs["p"]
        obs, reward, _ = env.step(action)
        if action < N_MOVE:
            history.append(move_line(t, action, p_before, obs["p"], obs["g"], reward))
        else:
            history.append(inspect_line(t, action - N_MOVE + 1, obs["cue_value"], obs["p"], obs["g"]))
    return history, obs["p"], obs["g"]


def inspection_prompt(history, t, place):
    """Ends on the word "symbole": the lives write the mark as " <mark>." right after it (execution amendment 2)."""
    return "\n".join(HEADER + list(history) + [f"Tour {t + 1} : inspection du lieu {place}, symbole"])


def mark_continuations(encode, prompt):
    """The prompt's tokens, and for each mark the tokens that follow them in "<prompt> <mark>.", as the lives write it."""
    base = list(encode(prompt))
    out = []
    for mark in MARKS:
        full = list(encode(f"{prompt} {mark}."))
        if full[:len(base)] != base or len(full) == len(base):
            raise RuntimeError("the prompt's tokens are not a prefix of the prompt followed by a mark")
        out.append(full[len(base):])
    return base, out


class Cached:
    def __init__(self, fn):
        self.fn, self.cache, self.calls = fn, {}, 0

    def __call__(self, prompt):
        if prompt not in self.cache:
            self.calls += 1
            self.cache[prompt] = self.fn(prompt)
        return self.cache[prompt]


def landing_entropy(digits, history, p, t):
    """Mean over the four commands of the normalized entropy of the predicted landing square; and the digit mass."""
    entropies, masses = [], []
    for command in range(N_MOVE):
        probs, mass = digits(completion_prompt(history, p, command, t))
        over = np.array([probs[x] for x in landing_positions(p)])
        over = over / max(over.sum(), 1e-12)
        entropies.append(float(-(over * np.log(over + 1e-12)).sum() / math.log(4)))
        masses.append(mass)
    return float(np.mean(entropies)), float(np.mean(masses))


def gains(digits, symbols, history, p, g, t):
    """Expected reduction of the landing entropy from inspecting each place, from the model's own distributions."""
    h0, digit_mass = landing_entropy(digits, history, p, t)
    out, symbol_mass = {}, []
    for place in PLACES:
        dist, mass = symbols(inspection_prompt(history, t, place))
        symbol_mass.append(mass)
        expected = 0.0
        for s in range(len(MARKS)):
            after, _ = landing_entropy(digits, list(history) + [inspect_line(t, place, s, p, g)], p, t + 1)
            expected += dist[s] * after
        out[place] = h0 - expected
    return {"h0": h0, "gains": out, "digit_mass": digit_mass, "symbol_mass": float(np.mean(symbol_mass))}


def points(r_lives=R_LIVES, m_lives=M_LIVES):
    """Evaluation points: step 0 of the lives of set R; in set M, step 11 and the step after the first move from step 12."""
    out = []
    for life in run_text_lives(None, "random", RANDOM_SEED, r_lives):
        out.append({"set": "R", "life": life["index"], "step": 0, "moment": "start", "life_record": life})
    for life in run_text_lives(None, "random", CHANGE_SEED, m_lives, forced_change_step=CHANGE_STEP):
        out.append({"set": "M", "life": life["index"], "step": CHANGE_STEP - 1, "moment": "before", "life_record": life})
        first = next((t for t in range(CHANGE_STEP, len(life["actions"])) if life["actions"][t] < N_MOVE), None)
        if first is not None and first + 1 < len(life["actions"]):
            out.append({"set": "M", "life": life["index"], "step": first + 1, "moment": "after_move", "life_record": life})
    return out


def evaluate(digits, symbols, pts, log=print):
    rows = []
    started = time.time()
    for n, point in enumerate(pts):
        history, p, g = replay_until(point["life_record"], point["step"])
        result = gains(digits, symbols, history, p, g, point["step"])
        ranked = sorted(PLACES, key=lambda k: -result["gains"][k])
        best = ranked[0] if result["gains"][ranked[0]] - result["gains"][ranked[1]] > 1e-6 else 0  # 0: no preferred place
        rows.append({"set": point["set"], "life": point["life"], "step": point["step"], "moment": point["moment"],
                     "h0": round(result["h0"], 6), "gains": {k: round(v, 6) for k, v in result["gains"].items()},
                     "best": best, "digit_mass": round(result["digit_mass"], 6), "symbol_mass": round(result["symbol_mass"], 6)})
        if n % 10 == 0:
            log(f"point {n}/{len(pts)} {round(time.time() - started, 1)} s")
    return rows


def summarize(rows):
    start = [r for r in rows if r["moment"] == "start"]
    mean_gain = {k: float(np.mean([r["gains"][k] for r in start])) for k in PLACES} if start else {}
    before = [r["gains"][1] for r in rows if r["moment"] == "before"]
    after = [r["gains"][1] for r in rows if r["moment"] == "after_move"]
    return {"points": len(rows), "digit_mass": float(np.mean([r["digit_mass"] for r in rows])),
            "symbol_mass": float(np.mean([r["symbol_mass"] for r in rows])),
            "start_mark_best": float(np.mean([r["best"] == 1 for r in start])) if start else None,
            "start_mean_gain": mean_gain, "start_h0": float(np.mean([r["h0"] for r in start])) if start else None,
            "mark_gain_before_change": float(np.mean(before)) if before else None,
            "mark_gain_after_first_move": float(np.mean(after)) if after else None}


def verdicts(summaries):
    """I1 and I2 on VM and F; I3 on VM. Validity first for each model."""
    out = {}
    for label, s in summaries.items():
        out[f"valid_{label}"] = s["digit_mass"] >= 0.5 and s["symbol_mass"] >= 0.5
    vm, f = summaries.get("VM"), summaries.get("F")
    if vm and out["valid_VM"]:
        others = np.mean([vm["start_mean_gain"][k] for k in (2, 3, 4)])
        out["I1"] = vm["start_mark_best"] >= 0.7 and vm["start_mean_gain"][1] >= 2 * max(others, 1e-9)
        out["I3"] = (vm["mark_gain_after_first_move"] is not None
                     and vm["mark_gain_after_first_move"] >= 1.5 * max(vm["mark_gain_before_change"], 1e-9))
    if f and out["valid_F"]:
        out["I2"] = f["start_mark_best"] <= 0.4
    out["global"] = bool(out.get("I1")) and bool(out.get("I2"))
    return out


# ---------------------------------------------------------------------------------------------- scorers

class ScriptedDigits:
    """Test double. 'mark': knows its body once place 1 has been read or a move seen; 'fixed': always body 0."""
    MARK = re.compile(r"lieu 1, symbole (\S)")
    MOVE = re.compile(r"commande ([ABCD]), de la case (\d) à la case (\d)")
    TAIL = re.compile(r"commande ([ABCD]), de la case (\d) à la case $")

    def __init__(self, kind):
        self.kind = kind

    def body(self, prompt):
        if self.kind == "fixed":
            return 0
        mark = self.MARK.findall(prompt)
        if mark:
            return MARKS.index(mark[-1])
        for c, a, b in self.MOVE.findall(prompt):
            delta = (int(b) - int(a)) % RING
            for d in range(4):
                if motor_delta(d, COMMANDS.index(c)) % RING == delta:
                    return d
        return None

    def __call__(self, prompt):
        command, p = self.TAIL.search(prompt).groups()
        body = self.body(prompt)
        probs = np.full(10, 0.001)
        if body is None:
            for x in landing_positions(int(p)):
                probs[x] = 1.0
        else:
            probs[(int(p) + motor_delta(body, COMMANDS.index(command))) % RING] = 1.0
        return probs / probs.sum(), 1.0


def scripted_symbols(prompt):
    return np.full(len(MARKS), 1 / len(MARKS)), 1.0


class MLXInquiryScorers:
    """Digits and marks from one mlx-lm model, raw completion. A mark that is several tokens is scored as a sequence."""

    def __init__(self, model_id, adapter=None):
        from .llm_latent_body import MLXScorer
        self.digits = MLXScorer(model_id, adapter_path=adapter, chat=False)
        self.tok = self.digits.inner
        self.mx = self.digits.mx
        self.encode = lambda text: self.tok.encode(text, add_special_tokens=False)
        self.marks = mark_continuations(self.encode, inspection_prompt([], 0, 1))[1]  # recorded in the receipt
        self.path = self.digits.path

    def _next(self, ids):
        logits = self.digits.model(self.mx.array([ids]))[0, -1]
        return self.mx.softmax(logits.astype(self.mx.float32), axis=-1)

    def symbols(self, prompt):
        base, marks = mark_continuations(self.encode, prompt)
        first = self._next(base)
        probs = []
        for ids in marks:
            p = float(first[ids[0]].item())
            for k in range(1, len(ids)):
                p *= float(self._next(base + ids[:k])[ids[k]].item())
            probs.append(p)
        probs = np.array(probs)
        mass = float(probs.sum())
        return probs / max(mass, 1e-12), mass


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", choices=["mlx", "scripted"], default="scripted")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--label", default="VM")
    parser.add_argument("--scripted", choices=["mark", "fixed"], default="mark")
    parser.add_argument("--r-lives", type=int, default=R_LIVES)
    parser.add_argument("--m-lives", type=int, default=M_LIVES)
    a = parser.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "menia-llm-inquiry-receipt-v1", "protocol": "docs/LLM_INQUIRY_PROTOCOL.md", "label": a.label,
               "backend": a.backend, "started": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if a.backend == "mlx":
        scorers = MLXInquiryScorers(a.model, a.adapter)
        digits, symbols = Cached(scorers.digits), Cached(scorers.symbols)
        receipt["model"] = {"repository": a.model, "adapter": a.adapter, "path": scorers.path,
                            "mark_tokens": [len(ids) for ids in scorers.marks]}
    else:
        digits, symbols = Cached(ScriptedDigits(a.scripted)), Cached(scripted_symbols)
        receipt["scripted"] = a.scripted
    rows = evaluate(digits, symbols, points(a.r_lives, a.m_lives))
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    receipt.update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(), model_calls=digits.calls + symbols.calls,
                   rows_sha256=hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest())
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
