"""The loop that uses its state (docs/TINY_LOOP_GATE_PROTOCOL.md): LOOP1 to LOOP4 of test 33.

Test 32 again, with the setting L4 of the pilot (research/tiny_loop_pilot.py) for the loop and the cut alike: a gated
state (s' = (1 - z) s + z n, n the normalized output of "Choix", z = sigmoid(W h + b)) and 12 000 updates. New seeds 10
to 19. Measures, draw and verdicts are those of test 32 (research/tiny_loop.py); published without threshold besides:
the probe of the untrained network, the grafts at equal event at turn j, LOOP4 without the (b) grafts that have no
other history, and the mean gate. The verdicts are numpy only; training and measures need torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import tiny_loop as TL
from . import tiny_loop_pilot as TLP
from . import tiny_survival as TS
from .need_relief_graft_event import same_history

SETTING = "L4"
CONFIG = TLP.CONFIG[SETTING]
SEEDS = tuple(range(10, 20))
ARMS = TL.ARMS
ROOT = "artifacts/tiny-loop-gate"


def extras(runs, grafts, held):
    """Published without threshold: per seed and on average, the grafts at equal event at turn j, and LOOP4 without the
    (b) grafts that have no other history."""
    at = [{x["t"]: x for x in W.decisions(life)} for life in held]
    kind = np.array([g["kind"] for g in grafts])
    e = np.array([g["e"] for g in grafts], float)
    same = np.array([at[g["r"]][g["j"]]["event"] == at[g["d"]][g["j"]]["event"] for g in grafts])
    alike = np.array([g["kind"] == "b" and same_history(held, g["r"], g["d"], g["j"]) for g in grafts])
    out = {"same_event_share": {k: float(same[kind == k].mean()) for k in ("a", "b")},
           "b_without_other_history": int(alike.sum()), "by_seed": {}}
    seeds = sorted(runs["loop"], key=int)
    for s in seeds:
        got = {i: p1 - p0 for i, p0, p1 in runs["loop"][s]["grafts"]}
        delta = np.array([got[g["index"]] for g in grafts])
        a, b = kind == "a", kind == "b"
        out["by_seed"][s] = {"same_event_abs_a": float(np.abs(delta[a & same]).mean()),
                             "same_event_abs_b": float(np.abs(delta[b & same]).mean()),
                             "same_event_align": float((delta[a & same] * e[a & same]).mean()),
                             "abs_a": float(np.abs(delta[a]).mean()),
                             "abs_b_other_history": float(np.abs(delta[b & ~alike]).mean())}
    per = out["by_seed"]
    out["same_event_align"] = TS.interval([per[s]["same_event_align"] for s in seeds])
    out["same_event_abs"] = {k: float(np.mean([per[s][f"same_event_abs_{k}"] for s in seeds])) for k in ("a", "b")}
    abs_a = float(np.mean([per[s]["abs_a"] for s in seeds]))
    abs_b = float(np.mean([per[s]["abs_b_other_history"] for s in seeds]))
    out["loop4_without_alike_b"] = {
        "abs_a": abs_a, "abs_b": abs_b, "half_holds": bool(abs_b <= TL.HALF * abs_a),
        "difference": TS.interval([per[s]["abs_a"] - per[s]["abs_b_other_history"] for s in seeds])}
    out["probe_untrained"] = {k: float(np.mean([runs["loop"][s]["probe_untrained"][k] for s in seeds])) for k in ("E", "N")}
    out["gate_mean"] = float(np.mean([runs["loop"][s]["gate_mean"] for s in seeds]))
    return out


def verdicts(runs, grafts, setup, event_rule, held):
    result = TL.verdicts(runs, grafts, setup, event_rule)
    result["values"]["extras"] = extras(runs, grafts, held)
    return result


# ----------------------------------------------------------------------------------------------------- torch part

def gate_mean(net, held):
    """The mean of z over the decisions of the 128 lives read by the loop (how much of the new state enters)."""
    import torch
    base, zs = net.base, []
    with torch.no_grad():
        for life in held:
            state, last = net.initial(1), TL.NONE
            for x in W.decisions(life):
                choice = torch.tensor([TL.CHOICE])
                inp = torch.stack([state, base.tok(torch.tensor([last])), base.tok(torch.tensor([TL.EVENT0 + x["event"]])),
                                   base.tok(choice)], 1) + base.pos.weight[None]
                h = base.layers(inp)[:, -1]
                zs.append(float(torch.sigmoid(net.gate(h)).mean()))
                _, _, state = net.step(state, torch.tensor([last]), torch.tensor([TL.EVENT0 + x["event"]]))
                last = TL.R + x["action"]
    return float(np.mean(zs))


def run(a):
    import torch
    torch.set_num_threads(1)
    log = lambda m: print(m, flush=True)
    root = Path(a.out)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    held = TL.lives()
    grafts = TL.draw(held)
    for seed in a.seeds:
        nets = {}
        for arm in a.arms:
            path = root / "runs" / f"{arm}-{seed}.json"
            if path.exists() and not (seed == SEEDS[0] and not (root / "setup.json").exists()):
                continue
            net, curve = TLP.train(arm, seed, CONFIG, log, checkpoint=root / "checkpoints" / f"{arm}-{seed}.pt")
            net.eval()
            nets[arm] = net
            survived, agrees = TL.survival(net, arm)
            reader = TL.Reader(net, arm)
            untrained = TLP.model(seed, CONFIG["gate"])
            untrained.eval()
            row = {"arm": arm, "seed": seed, "setting": SETTING, "survival": float(np.mean(survived)),
                   "survived": survived, "agrees": agrees, "curve": curve, "probe": TL.probe(reader, held),
                   "probe_untrained": TL.probe(TL.Reader(untrained, arm), held),
                   "grafts": TL.measure_grafts(reader, held, grafts)}
            if arm == "loop":
                row["gate_mean"] = gate_mean(net, held)
            path.write_text(json.dumps(row) + "\n")
            log(f"{arm} seed {seed}: survival {np.mean(survived):.3f}, agrees with the needs rule {agrees:.3f}")
        if seed == SEEDS[0] and set(nets) == set(ARMS) and not (root / "setup.json").exists():
            (root / "setup.json").write_text(json.dumps(TL.checks(nets["loop"], nets["cut"], held), indent=1) + "\n")


def gather(root):
    out = Path(root)
    runs = {arm: {} for arm in ARMS}
    for path in sorted((out / "runs").glob("*.json")):
        r = json.loads(path.read_text())
        runs[r["arm"]][str(r["seed"])] = r
    setup = json.loads((out / "setup.json").read_text())
    held = TL.lives()
    result = verdicts(runs, TL.draw(held), setup, TS.event_rule_survival(), held)
    return json.loads(json.dumps(dict(result, setup=setup)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    r.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    sub.add_parser("write")
    sub.add_parser("verdicts")
    for s in sub.choices.values():
        s.add_argument("--out", default=ROOT)
    a = parser.parse_args(argv)
    if a.command == "run":
        run(a)
        return
    result = gather(a.out)
    path = Path(a.out) / "verdicts.json"
    if a.command == "write":
        path.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result["verdicts"]))
    else:
        differs = result != json.loads(path.read_text())
        print("differs:", "yes" if differs else "none")
        if differs:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
