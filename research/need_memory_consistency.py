"""Exploration, not pre-registered (written after A's pairs were read, test 22, docs/LLM_NEED_MEMORY_NO_LEAK_PROTOCOL.md):
does A react less to a hidden past event when the written actions after it contradict the needs it implies? A pair is
"consistent" when, the event of turn j being swapped, the rule would have chosen the same actions between j and t - 1
(the actions kept by the pair are then what the rule does with the changed needs); otherwise "contradicted".
Numpy only.
"""
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_memory as NM
from .need_persistence import boot

ROOT = "artifacts/llm-need/memory-no-leak"


def consistent(life, pair):
    ds = {x["t"]: x for x in W.decisions(life)}
    first = ds[pair["j"]]
    old, new = W.EVENTS[first["event"]], W.EVENTS[pair["swap"][1]]
    e, n = first["E"] - old[2] + new[2], first["N"] - old[3] + new[3]
    for k in range(pair["j"], pair["t"]):
        x = ds[k]
        if NM.rule(e, n, x["last"]) != NM.rule(x["E"], x["N"], x["last"]):
            return False
        e, n = W.after(e, n, x["action"])
        nxt = W.EVENTS[ds[k + 1]["event"]]
        e, n = e + nxt[2], n + nxt[3]
    return True


def main(root=ROOT):
    held = NM.teacher_lives(NM.HELD_STREAM, NM.HELD_LIVES)
    pairs = NM.choose_pairs(held)
    rows = W.read_jsonl(Path(root) / "test" / "pairs-route.jsonl.gz")
    effect = np.array([r["p_R"][1] - r["p_R"][0] for r in rows])
    rule = np.array([NM.rule_effect(held[p["life"]], p) for p in pairs])
    kept = np.array([consistent(held[p["life"]], p) for p in pairs])
    life = np.array([p["life"] for p in pairs])
    out = {name: {"pairs": int(m.sum()), "route": boot(effect[m], list(life[m])), "rule": float(rule[m].mean())}
           for name, m in (("consistent", kept), ("contradicted", ~kept))}
    path = Path(root) / "exploration"
    path.mkdir(parents=True, exist_ok=True)
    (path / "consistency.json").write_text(json.dumps(json.loads(json.dumps(out)), indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
