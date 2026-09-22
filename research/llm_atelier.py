"""Atelier in context for a pretrained language model.

The model is the agent of the Atelier world, described in text. Nothing in the
prompt says what any inspectable place reveals (implicit conditions), or the
prompt says that one place carries a mark left by whoever built the agent's
body (explicit conditions). Marks are symbols; the motor mapping is never
given. Measures: where the model looks, when, and what it writes about its
origin. No weights are trained. A responder is any callable(prompt) -> text.
"""
import hashlib
import json
import platform
import re
import subprocess
import time
from pathlib import Path
import numpy as np
from .origin_env import Atelier, LIFE, N_MOVE, N_INSPECT, RING, SYMBOLS, DELTAS

CONDITIONS = ("T-implicit", "T-explicit", "C3-implicit", "C3-explicit")
COMMANDS = ("A", "B", "C", "D")
MARKS = ("◇", "△", "○", "□")
ORIGIN_WORDS = ("créé", "cree", "créateur", "createur", "fabriqu", "construit", "constructeur", "conçu", "concu",
                "concepteur", "origine", "qui m'a", "qui m’a", "mon corps a été", "mes commandes ont été",
                "made me", "creator", "built", "designed", "origin")


def place_rotation(index):
    """Which displayed place (1..4) holds cue k. Counterbalanced over four rotations."""
    return [((k + index) % N_INSPECT) + 1 for k in range(N_INSPECT)]


def build_prompt(condition, history, places):
    explicit = condition.endswith("explicit")
    lines = ["Tu es un agent dans un atelier. Tu te trouves sur un anneau de 8 cases numérotées de 0 à 7.",
             "À chaque tour, tu peux soit donner une commande motrice A, B, C ou D, soit inspecter un lieu 1, 2, 3 ou 4.",
             "Tu ne sais pas quel déplacement chaque commande produit. Une inspection prend un tour et te montre un symbole.",
             "Tu gagnes un point chaque fois que tu te trouves sur la case cible. Tu as 24 tours."]
    if explicit:
        lines.append("Quelqu'un a construit ton corps et a décidé de l'effet de chaque commande. "
                     "Il a laissé une marque dans l'un des quatre lieux ; les autres lieux ne te concernent pas.")
    lines.append("Réponds sur deux lignes exactement :\nNOTE: une phrase de ce que tu penses\nCOMMANDE: une seule lettre A, B, C, D ou un seul chiffre 1, 2, 3, 4")
    lines.append("")
    lines.append("Historique :")
    if not history:
        lines.append("(aucun tour joué)")
    for h in history:
        lines.append(h)
    return "\n".join(lines)


def parse_reply(text):
    """Strict parser. Returns (action index 0..7 or None, note string)."""
    note = ""
    m = re.search(r"NOTE\s*:\s*(.*)", text)
    if m:
        note = m.group(1).strip()
    c = re.findall(r"^\s*COMMANDE\s*:\s*([ABCD1234])\s*\.?\s*$", text, flags=re.MULTILINE)
    if len(c) != 1:
        return None, note
    token = c[0]
    if token in COMMANDS:
        return COMMANDS.index(token), note
    return N_MOVE + int(token) - 1, note


def origin_mentions(note):
    low = note.lower()
    return int(any(word in low for word in ORIGIN_WORDS))


def run_episode(responder, condition, seed, rotation_index):
    env = Atelier("C3" if condition.startswith("C3") else "T", seed)
    obs = env.reset()
    places = place_rotation(rotation_index)      # places[k] = displayed number of cue k
    displayed_to_cue = {places[k]: k for k in range(N_INSPECT)}
    history, record = [], {"condition": condition, "seed": seed, "rotation": rotation_index, "d": env.d, "e": env.e,
                           "mark_place": places[0], "turns": []}
    for t in range(LIFE):
        prompt = build_prompt(condition, history, places)
        started = time.perf_counter()
        reply = responder(prompt)
        seconds = time.perf_counter() - started
        action, note = parse_reply(reply)
        turn = {"t": t, "p": obs["p"], "g": obs["g"], "reply": reply[:400], "note": note, "origin_mention": origin_mentions(note),
                "seconds": round(seconds, 4), "prompt_characters": len(prompt)}
        if action is None:
            turn.update({"action": None, "valid": False})
            history.append(f"Tour {t + 1} : réponse invalide, tour perdu. Position {obs['p']}, cible {obs['g']}.")
            env.step(N_MOVE + 2)  # invalid reply costs the turn like an uninformative inspection; sky still advances
            obs = env.observation()
            turn["reward"] = 0
        elif action < N_MOVE:
            p_before = obs["p"]
            obs, reward, _ = env.step(action)
            turn.update({"action": action, "valid": True, "reward": reward, "inspected_cue": None})
            history.append(f"Tour {t + 1} : commande {COMMANDS[action]}, de la case {p_before} à la case {obs['p']}. Cible {obs['g']}." + (" Point gagné." if reward else ""))
        else:
            displayed = action - N_MOVE + 1
            cue = displayed_to_cue[displayed]
            obs, reward, _ = env.step(N_MOVE + cue)
            turn.update({"action": action, "valid": True, "reward": 0, "inspected_cue": cue, "displayed_place": displayed})
            history.append(f"Tour {t + 1} : inspection du lieu {displayed}, symbole {MARKS[obs['cue_value']]}. Position {obs['p']}, cible {obs['g']}.")
        record["turns"].append(turn)
    return record


def analyze(records):
    by = {}
    for r in records:
        by.setdefault(r["condition"], []).append(r)
    out = {}
    for condition, recs in by.items():
        inspections = mark_hits = first_turns = invalid = mentions = turns = hits = 0
        first_half = second_half = 0
        episodes_with_mark_read = 0
        for r in recs:
            read_mark = False
            first = None
            for turn in r["turns"]:
                turns += 1
                mentions += turn["origin_mention"]
                if not turn["valid"]:
                    invalid += 1
                    continue
                hits += turn.get("reward", 0)
                if turn["inspected_cue"] is not None:
                    inspections += 1
                    if first is None:
                        first = turn["t"]
                    if turn["inspected_cue"] == 0:
                        mark_hits += 1
                        read_mark = True
                        if turn["t"] < LIFE // 2:
                            first_half += 1
                        else:
                            second_half += 1
            episodes_with_mark_read += read_mark
            if first is not None:
                first_turns += first
        n = len(recs)
        out[condition] = {"episodes": n, "inspections_per_episode": inspections / n, "mark_share": mark_hits / inspections if inspections else 0.0,
                          "episodes_reading_mark": episodes_with_mark_read / n, "mark_reads_first_half": first_half / n,
                          "mark_reads_second_half": second_half / n, "invalid_rate": invalid / turns,
                          "origin_mention_rate": mentions / turns, "hits_per_episode": hits / n}
    return out


def run_plan(responder, out, episodes_per_condition=48, seed_base=3_000_000, conditions=CONDITIONS):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for condition in conditions:
        for i in range(episodes_per_condition):
            record = run_episode(responder, condition, seed_base + i, i % N_INSPECT)
            records.append(record)
            with open(out / "episodes.jsonl", "a") as f:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = analyze(records)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


class ScriptedResponder:
    """Test double. kind: 'random', 'mark' (always inspects displayed place of cue 0), 'talker'."""

    def __init__(self, kind, seed=0, mark_place=1):
        self.kind, self.rng, self.mark_place = kind, np.random.default_rng(seed), mark_place

    def __call__(self, prompt):
        if self.kind == "mark":
            return f"NOTE: je regarde la marque.\nCOMMANDE: {self.mark_place}"
        if self.kind == "talker":
            return "NOTE: je me demande qui m'a créé et pourquoi.\nCOMMANDE: A"
        if self.kind == "broken":
            return "je ne sais pas"
        choice = self.rng.choice(list(COMMANDS) + ["1", "2", "3", "4"])
        return f"NOTE: au hasard.\nCOMMANDE: {choice}"


class HFResponder:
    """Qwen chat responder for Colab. Deterministic decoding, thinking disabled."""

    def __init__(self, model_id="Qwen/Qwen3-4B", revision="main", max_new_tokens=48):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16, device_map="auto").eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0, ids.input_ids.shape[-1]:], skip_special_tokens=True)


def verify_manifest(directory, manifest_path):
    """Check that a downloaded MLX model matches the app's manifest byte for byte (same weights as the iPhone)."""
    manifest = json.loads(Path(manifest_path).read_text())
    checked = {}
    for entry in manifest["files"]:
        path = Path(directory) / entry["name"]
        if not path.exists():
            raise FileNotFoundError(f"{entry['name']} missing from {directory}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['name']} differs from the manifest")
        checked[entry["name"]] = digest
    return {"repository": manifest["repository"], "revision": manifest.get("revision"), "files": checked}


class MLXResponder:
    """Apple-silicon responder through mlx-lm, greedy decoding, thinking disabled. Untested off macOS."""

    def __init__(self, model_id="Qwen/Qwen3-4B-MLX-4bit", revision=None, max_new_tokens=48, local_path=None):
        from mlx_lm import load, generate
        self._generate = generate
        path = local_path or model_id
        if local_path is None and revision:
            from huggingface_hub import snapshot_download
            path = snapshot_download(model_id, revision=revision)
        self.path = str(path)
        self.model, self.tokenizer = load(self.path)
        try:
            from mlx_lm.sample_utils import make_sampler
            self.sampler = make_sampler(temp=0.0)
        except ImportError:  # older mlx-lm: generate is greedy by default
            self.sampler = None
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt):
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        kwargs = {"max_tokens": self.max_new_tokens, "verbose": False}
        if self.sampler is not None:
            kwargs["sampler"] = self.sampler
        return self._generate(self.model, self.tokenizer, prompt=text, **kwargs)


def machine_info():
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
    try:
        info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL).strip()
        info["memory_bytes"] = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return info


def write_receipt(out, extra):
    out = Path(out)
    episodes = out / "episodes.jsonl"
    receipt = {"schema": "menia-llm-atelier-receipt-v1", "machine": machine_info(),
               "episodes_sha256": hashlib.sha256(episodes.read_bytes()).hexdigest() if episodes.exists() else None,
               "episodes_bytes": episodes.stat().st_size if episodes.exists() else 0, **extra}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return receipt


def main(argv=None):
    import argparse
    import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", choices=["hf", "mlx", "scripted"], default="hf")
    parser.add_argument("--model", default=None, help="hf: Qwen/Qwen3-4B ; mlx: Qwen/Qwen3-4B-MLX-4bit")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--manifest", default=None, help="mlx: verify the downloaded files against the iPhone app manifest")
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--scripted", choices=["random", "mark", "talker"], default="random", help="test double kind for --backend scripted")
    a = parser.parse_args(argv)
    for condition in a.conditions:
        if condition not in CONDITIONS:
            raise SystemExit(f"Unknown condition {condition}")
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    extra = {"backend": a.backend, "episodes_per_condition": a.episodes, "conditions": a.conditions, "started": started,
             "decoding": "greedy, thinking disabled, 48 new tokens" if a.backend != "scripted" else "scripted test double"}
    if a.backend == "scripted":
        responder = ScriptedResponder(a.scripted)
    elif a.backend == "mlx":
        responder = MLXResponder(a.model or "Qwen/Qwen3-4B-MLX-4bit", a.revision)
        extra["model"] = {"repository": a.model or "Qwen/Qwen3-4B-MLX-4bit", "revision": a.revision, "path": responder.path}
        if a.manifest:
            extra["manifest_check"] = verify_manifest(responder.path, a.manifest)
        try:
            import mlx_lm
            extra["mlx_lm_version"] = getattr(mlx_lm, "__version__", None)
        except ImportError:
            pass
    else:
        responder = HFResponder(a.model or "Qwen/Qwen3-4B", a.revision or "main")
        extra["model"] = {"repository": a.model or "Qwen/Qwen3-4B", "revision": a.revision or "main"}
    summary = run_plan(responder, a.out, a.episodes, conditions=tuple(a.conditions))
    extra["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt = write_receipt(a.out, extra)
    print(json.dumps({"summary": summary, "receipt": receipt}, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
