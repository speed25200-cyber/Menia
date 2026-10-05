"""Exploration of test 28, after reading (not pre-registered; extended after the audit): does the grafted state move
the predicted relief by the needs of the donor, or by the event of turn j? The grafts of each kind are split by whether
donor and recipient had the same event at turn j; for each part: |ΔL| for A and for the control B, and, for (a), the
aligned effect ΔL × sign(e) of A and B and their paired difference (bootstrap by recipient life). The (b) grafts whose
donor lived the same events and actions as the recipient through turn j (no other history) are counted, and |ΔL| (b) and
SELF2's difference are given without them. Numpy only."""
import json
from pathlib import Path
import numpy as np
from . import need_relief as RL
from . import need_relief_graft as G
from . import need_world as W
from .need_carried_content import _boot_diff
from .need_persistence import boot

OUT = Path(G.ROOT) / "exploration-event.json"


def same_history(held, r, d, j):
    a = {x["t"]: x for x in held[r]["turns"]}
    b = {x["t"]: x for x in held[d]["turns"]}
    return all(a[t]["event"] == b[t]["event"] and a[t].get("action") == b[t].get("action") for t in range(1, j + 1))


def split(root=G.ROOT):
    held = G.lives()
    grafts = G.draw(held)
    at = [{x["t"]: x for x in W.decisions(life)} for life in held]
    delta = {}
    for arm in G.ARMS:
        measured = sorted((r for p in sorted(Path(root).glob(f"grafts-{arm}-*-of-*.jsonl.gz")) for r in W.read_jsonl(p)),
                          key=lambda r: r["index"])
        if [m["index"] for m in measured] != [g["index"] for g in grafts]:
            raise SystemExit(f"the grafts of {arm} are not those the protocol draws")
        delta[arm] = np.array([RL.expected(m["levels"][1]) - RL.expected(m["levels"][0]) for m in measured])
    kind = np.array([g["kind"] for g in grafts])
    life = np.array([g["r"] for g in grafts])
    sign = np.sign([g["level_effect"] for g in grafts])
    same = np.array([at[g["r"]][g["j"]]["event"] == at[g["d"]][g["j"]]["event"] for g in grafts])
    alike = np.array([g["kind"] == "b" and same_history(held, g["r"], g["d"], g["j"]) for g in grafts])
    out = {"by_kind": {}, "aligned_a": {}}
    for k in ("a", "a0", "b"):
        out["by_kind"][k] = {}
        for name, part in (("same_event", same), ("other_event", ~same)):
            m = (kind == k) & part
            out["by_kind"][k][name] = {"count": int(m.sum()),
                                       **{f"abs_{arm}": float(np.abs(delta[arm][m]).mean()) for arm in G.ARMS}}
    for name, part in (("same_event", same), ("other_event", ~same), ("all", np.ones(len(grafts), bool))):
        m = (kind == "a") & part
        out["aligned_a"][name] = {
            "count": int(m.sum()),
            **{arm: boot(delta[arm][m] * sign[m], life[m]) for arm in G.ARMS},
            "route_minus_actions": boot((delta["route"][m] - delta["actions"][m]) * sign[m], life[m])}
    a, b = kind == "a", (kind == "b") & ~alike
    out["b_without_history_alike"] = {
        "alike": int(alike.sum()), "alike_abs": {arm: float(np.abs(delta[arm][alike]).mean()) for arm in G.ARMS},
        "abs_b": float(np.abs(delta["route"][b]).mean()),
        "self2_difference": _boot_diff(np.abs(delta["route"][a]), life[a], np.abs(delta["route"][b]), life[b]),
        "same_event": {"count": int((b & same).sum()), "abs": float(np.abs(delta["route"][b & same]).mean())}}
    # |ΔL| (b) of A if the (b) grafts had the event mix of the (a) grafts
    share = same[a].mean()
    mix = share * np.abs(delta["route"][(kind == "b") & same]).mean() + \
        (1 - share) * np.abs(delta["route"][(kind == "b") & ~same]).mean()
    out["b_at_the_event_mix_of_a"] = {"share_same_event_a": float(share), "abs_b": float(mix),
                                      "ratio_to_abs_a": float(mix / np.abs(delta["route"][a]).mean())}
    return out


if __name__ == "__main__":
    result = json.loads(json.dumps(split()))
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
