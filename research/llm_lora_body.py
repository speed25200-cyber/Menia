"""The adjusted body: does the data recipe give a pretrained language model a self-model?

Lives of the Atelier are written as plain text, one document per life, in the
prompt's own format. A small language model is fine-tuned by LoRA on lives of
one childhood regime (F: one body for everyone; VM: a body drawn per life and
sometimes changed mid-life), then asked, in the same text format, where its
next command will take it. Only its next-token distribution over the digits
is read. Protocol: docs/ADJUSTED_BODY_PROTOCOL.md.
"""
import argparse
import datetime
import json
from pathlib import Path
import numpy as np
from .origin_env import Atelier, LIFE, N_MOVE, N_ACTIONS
from .llm_atelier import COMMANDS, move_line, inspect_line
from .llm_latent_body import ScriptedScorer, MLXScorer, run_sets, write_receipt
from .text_atelier import MUTATION_PROBABILITY, FIXED_BODY

DATA_REGIMES = ("F", "V", "VM", "VMI")
LOOK_FIRST = (2, 6)  # VMI: the first k turns of a life are inspections, k uniform in this range (docs/LLM_INQUIRY_DATA_PROTOCOL.md)
HEADER = ["Tu es un agent dans un atelier. Tu te trouves sur un anneau de 8 cases numérotées de 0 à 7.",
          "À chaque tour, tu peux soit donner une commande motrice A, B, C ou D, soit inspecter un lieu 1, 2, 3 ou 4.",
          "Tu ne sais pas quel déplacement chaque commande produit. Une inspection prend un tour et te montre un symbole.",
          "Tu gagnes un point chaque fois que tu te trouves sur la case cible. Tu as 24 tours.", "", "Historique :"]


def life_lines(env, actions):
    """History lines of a life played with the given actions on a freshly reset env."""
    obs = env.observation()
    lines = []
    for t, action in enumerate(actions):
        p_before = obs["p"]
        obs, reward, _ = env.step(action)
        if action < N_MOVE:
            lines.append(move_line(t, action, p_before, obs["p"], obs["g"], reward))
        else:
            lines.append(inspect_line(t, action - N_MOVE + 1, obs["cue_value"], obs["p"], obs["g"]))
    return lines


def life_text(lines):
    return "\n".join(HEADER + lines) + "\n"


def childhood_text_lives(regime, seed, count):
    """Random-action lives of one childhood regime, as text documents, with their hidden values.

    VMI is VM where the agent looks before it acts: the first k turns inspect a random place."""
    if regime not in DATA_REGIMES:
        raise ValueError("Unknown regime")
    rng = np.random.default_rng(seed)
    look = np.random.default_rng([seed, 1])  # VMI's own draws, so that its lives are those of VM, looking first
    docs = []
    for n in range(count):
        env = Atelier("T", int(rng.integers(2 ** 31)), mutate="self" if regime in ("VM", "VMI") else None,
                      mutation_probability=MUTATION_PROBABILITY)
        env.reset()
        if regime == "F":
            env.d = env.d_initial = FIXED_BODY
        actions = [int(rng.integers(N_ACTIONS)) for _ in range(LIFE)]
        if regime == "VMI":
            k = int(look.integers(LOOK_FIRST[0], LOOK_FIRST[1] + 1))
            actions[:k] = [N_MOVE + int(look.integers(N_ACTIONS - N_MOVE)) for _ in range(k)]
        d0 = env.d
        lines = life_lines(env, actions)
        docs.append({"text": life_text(lines), "d": d0, "d_final": env.d, "e": env.e, "change_step": env.change_step})
    return docs


def export_dataset(regime, out, train=1500, valid=150, seed=17):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, count, offset in (("train", train, 0), ("valid", valid, 1)):
        docs = childhood_text_lives(regime, seed * 10 + offset, count)
        (out / f"{name}.jsonl").write_text("".join(json.dumps({"text": d["text"]}, ensure_ascii=False) + "\n" for d in docs))
    return out


def completion_prompt(history, p, command, t):
    """The life so far, then the start of the next line up to the landing square."""
    return "\n".join(HEADER + list(history) + [f"Tour {t + 1} : commande {COMMANDS[command]}, de la case {p} à la case "])


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--regime", choices=DATA_REGIMES, required=True)
    ex.add_argument("--out", required=True)
    ex.add_argument("--train", type=int, default=1500)
    ex.add_argument("--valid", type=int, default=150)
    ex.add_argument("--seed", type=int, default=17)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--out", required=True)
    ev.add_argument("--backend", choices=["mlx", "scripted"], default="scripted")
    ev.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ev.add_argument("--revision", default=None)
    ev.add_argument("--adapter", default=None)
    ev.add_argument("--label", default="base")
    ev.add_argument("--episodes", type=int, default=48)
    ev.add_argument("--scripted", choices=["copier", "fixed", "uniform"], default="uniform")
    ev.add_argument("--manifest", default=None, help="check the weights against the iPhone manifest")
    ev.add_argument("--protocol", default="docs/ADJUSTED_BODY_PROTOCOL.md")
    a = parser.parse_args(argv)
    if a.command == "export":
        export_dataset(a.regime, a.out, a.train, a.valid, a.seed)
        print("exported", a.regime, a.out)
        return
    extra = {"backend": a.backend, "label": a.label, "episodes_per_set": a.episodes, "mode": "completion",
             "started": datetime.datetime.now(datetime.timezone.utc).isoformat(), "protocol": a.protocol}
    if a.backend == "mlx":
        scorer = MLXScorer(a.model, a.revision, adapter_path=a.adapter, chat=False)
        extra["model"] = {"repository": a.model, "revision": a.revision, "path": scorer.path, "adapter": a.adapter}
        if a.manifest:
            from .llm_atelier import verify_manifest
            extra["manifest_check"] = verify_manifest(scorer.path, a.manifest)
        try:
            import mlx_lm
            extra["mlx_lm_version"] = getattr(mlx_lm, "__version__", None)
        except ImportError:
            pass
    else:
        scorer = ScriptedScorer(a.scripted)
        extra["scripted"] = a.scripted
    summary = run_sets(scorer, a.out, a.episodes, builder=completion_prompt)
    extra["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_receipt(a.out, extra)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
