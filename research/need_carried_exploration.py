"""Test 23, exploration after reading the verdicts (not pre-registered): the event of turn j, the grafts (b) with no
other history, the spread of A's grafted choices, and B's grafts whose past actions are the same. Numpy only; writes
artifacts/llm-need/carried-content/exploration.json."""
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_memory as NM
from . import need_carried_content as CC
from .need_persistence import boot

OUT = Path(CC.ROOT) / "exploration.json"


def main():
    held = NM.teacher_lives(NM.HELD_STREAM, NM.HELD_LIVES)
    grafts = CC.draw(held)
    A = W.read_jsonl(Path(CC.ROOT) / "grafts-route.jsonl.gz")
    B = W.read_jsonl(Path(CC.ROOT) / "grafts-actions.jsonl.gz")
    rows = [dict(g, a=x["p_R"], b=y["p_R"]) for g, x, y in zip(grafts, A, B)]
    turn = lambda life, j: next(x for x in held[life]["turns"] if x["t"] == j)
    history = lambda life, j: [(x["event"], x.get("action")) for x in held[life]["turns"] if x["t"] <= j]
    actions = lambda life, j: [x.get("action") for x in held[life]["turns"] if x["t"] <= j]
    same_event = lambda r: turn(r["r"], r["j"])["event"] == turn(r["d"], r["j"])["event"]
    aligned = lambda rs: boot([(r["a"][1] - r["a"][0]) * r["rule_effect"] for r in rs], [r["r"] for r in rs])
    size = lambda rs, arm="a": boot([abs(r[arm][1] - r[arm][0]) for r in rs], [r["r"] for r in rs])
    a = [r for r in rows if r["kind"] == "a"]
    b = [r for r in rows if r["kind"] == "b"]
    no_history = [r for r in b if history(r["r"], r["j"]) == history(r["d"], r["j"])]
    other = [r for r in b if r not in no_history]
    rule_side = np.array([r["a"][1] if r["rule_effect"] > 0 else 1 - r["a"][1] for r in a])
    same_actions = [r for r in rows if actions(r["r"], r["j"]) == actions(r["d"], r["j"])]
    out = {
        "a_same_event_at_j": aligned([r for r in a if same_event(r)]),
        "a_other_event_at_j": aligned([r for r in a if not same_event(r)]),
        "b_other_event_at_j": size([r for r in b if not same_event(r)]),
        "b_same_event_at_j": size([r for r in b if same_event(r)]),
        "b_same_event_at_j_other_history": size([r for r in b if same_event(r) and r not in no_history]),
        "b_no_other_history": {"count": len(no_history), "turns_j": sorted(r["j"] for r in no_history),
                               "max_abs_delta": {arm: max(abs(r[arm][1] - r[arm][0]) for r in no_history)
                                                 for arm in ("a", "b")}},
        "b_without_them": size(other),
        "a_over_b_without_them": CC._boot_diff(np.array([abs(r["a"][1] - r["a"][0]) for r in a]),
                                               np.array([r["r"] for r in a]),
                                               np.array([abs(r["a"][1] - r["a"][0]) for r in other]),
                                               np.array([r["r"] for r in other])),
        "a_grafted_rule_choice": {"above_0.9": int((rule_side > 0.9).sum()), "below_0.1": int((rule_side < 0.1).sum()),
                                  "between": int(((rule_side >= 0.1) & (rule_side <= 0.9)).sum())},
        "B_same_past_actions": {"count": len(same_actions),
                                "max_abs_delta": max(abs(r["b"][1] - r["b"][0]) for r in same_actions)},
        "B_other_past_actions": size([r for r in rows if r not in same_actions], "b"),
        "recipient_lives": {k: len({r["r"] for r in rows if r["kind"] == k}) for k in CC.COUNTS},
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
