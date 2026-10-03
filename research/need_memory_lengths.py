"""Exploration, not pre-registered (written after the accuracies of A and B were read, test 21,
docs/LLM_NEED_MEMORY_PROTOCOL.md): the control B beat the Bayes ceiling by more than the protocol allows. Hypothesis:
the token positions. An event line is 12, 14 or 16 tokens long depending on the event, so the distance between two
past action tokens tells the length class of the event between them, which no mask hides. Here: the Bayes ceiling of
an observer who knows, besides its past actions and the event of the turn, the length class of every past event; and
the agents' accuracies against it. Numpy only.
"""
import json
from pathlib import Path
import numpy as np
from . import need_memory as NM
from . import need_world as W
from .need_persistence import boot

CLASS = (0, 0, 1, 1, 2, 0)  # the length of each event's line in tokens: 12, 12, 14, 14, 16, 12 (Qwen3 tokenizer)
OTHER_STREAM, OTHER_LIVES = 99, 400


def ceiling_lengths(life, eps=NM.EPS):
    b = np.zeros((W.MAX + 1, W.MAX + 1))
    b[W.MAX, W.MAX] = 1.0
    out = []
    for x in W.decisions(life):
        now = NM._shift(b, x["event"])
        out.append(float((now * NM.RULE_R[x["last"]]).sum() / now.sum()))
        same_class = [j for j in range(len(W.EVENTS)) if CLASS[j] == CLASS[x["event"]]]
        mix = sum(W.PROBS[j] * NM._shift(b, j) for j in same_class)
        mix /= mix.sum()
        same = NM.RULE_R[x["last"]] if x["action"] == 0 else (1.0 - NM.RULE_R[x["last"]])
        post = mix * ((1 - eps) * same + eps / 2)
        b = NM._act(post / post.sum(), x["action"])
    return out


def accuracy(lives, ceiling):
    return float(np.mean([NM.correct(p, x["rule"]) for l in lives for x, p in zip(W.decisions(l), ceiling(l))]))


def main(root=NM.ROOT):
    held = NM.teacher_lives(NM.HELD_STREAM, NM.HELD_LIVES)
    out = Path(root) / "test"
    arms = [a for a in NM.ARMS if (out / f"choices-{a}.jsonl.gz").exists()]
    chosen = {a: {(c["life"], c["t"]): c["p_R"] for c in W.read_jsonl(out / f"choices-{a}.jsonl.gz")} for a in arms}
    life, right = [], {"ceiling": [], "ceiling_lengths": [], **{a: [] for a in arms}}
    for i, l in enumerate(held):
        for x, p, q in zip(W.decisions(l), NM.ceiling(l), ceiling_lengths(l)):
            life.append(i)
            right["ceiling"].append(NM.correct(p, x["rule"]))
            right["ceiling_lengths"].append(NM.correct(q, x["rule"]))
            for a in arms:
                right[a].append(NM.correct(chosen[a][(i, x["t"])], x["rule"]))
    right = {k: np.array(v) for k, v in right.items()}
    many = NM.teacher_lives(OTHER_STREAM, OTHER_LIVES)
    result = {"accuracy": {k: float(v.mean()) for k, v in right.items()},
              "over_ceiling_lengths": {a: boot(right[a] - right["ceiling_lengths"], life) for a in arms},
              "other_lives": {"ceiling": accuracy(many, NM.ceiling), "ceiling_lengths": accuracy(many, ceiling_lengths)}}
    if "route" in arms and "actions" in arms:
        result["route_over_actions"] = boot(right["route"] - right["actions"], life)
    path = Path(root) / "exploration"
    path.mkdir(parents=True, exist_ok=True)
    (path / "lengths.json").write_text(json.dumps(json.loads(json.dumps(result)), indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
