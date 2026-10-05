"""Exploration of test 28, after reading (not pre-registered): does the grafted state move the predicted relief by the
needs of the donor, or by the event of turn j? The grafts of each kind are split by whether donor and recipient had the
same event at turn j; |ΔL| is given for each part, for A and for the control B. Numpy only."""
import json
from pathlib import Path
import numpy as np
from . import need_relief as RL
from . import need_relief_graft as G
from . import need_world as W

OUT = Path(G.ROOT) / "exploration-event.json"


def split(root=G.ROOT):
    held = G.lives()
    grafts = G.draw(held)
    at = [{x["t"]: x for x in W.decisions(life)} for life in held]
    out = {}
    for arm in G.ARMS:
        measured = sorted((r for p in sorted(Path(root).glob(f"grafts-{arm}-*-of-*.jsonl.gz")) for r in W.read_jsonl(p)),
                          key=lambda r: r["index"])
        out[arm] = {}
        for kind in ("a", "a0", "b"):
            parts = {"same_event": [], "other_event": []}
            for g, m in zip(grafts, measured):
                if g["kind"] != kind:
                    continue
                same = at[g["r"]][g["j"]]["event"] == at[g["d"]][g["j"]]["event"]
                parts["same_event" if same else "other_event"].append(
                    abs(RL.expected(m["levels"][1]) - RL.expected(m["levels"][0])))
            out[arm][kind] = {k: {"count": len(v), "abs_delta": float(np.mean(v)) if v else None}
                              for k, v in parts.items()}
    return out


if __name__ == "__main__":
    result = split()
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
