"""Exploration of test 32, after the audit (not pre-registered): what does the probe of the transmitted state measure?
1. The same probe on the networks before any learning (loop and cut, seeds 0 to 9).
2. Linear decoders of E and N (after the event of the turn) from the visible past only, as references: the event and
   the last action (what the cut sees), then with k earlier turns (their events and actions).
Same ridge (1) and 4 folds by life as research.tiny_loop.probe, on the 128 lives of the grafts."""
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import tiny_loop as TL

OUT = Path(TL.ROOT) / "exploration-probe.json"


def r2(X, Y, lives, folds=4, ridge=1.0):
    X = np.hstack([X, np.ones((len(X), 1))])
    pred = np.zeros_like(Y)
    for f in range(folds):
        test = lives % folds == f
        A = X[~test]
        w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ Y[~test])
        pred[test] = X[test] @ w
    r = 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    return {"E": float(r[0]), "N": float(r[1])}


def references(held, depths=(0, 1, 2, 3)):
    rows = []
    for i, life in enumerate(held):
        ds = W.decisions(life)
        for q, x in enumerate(ds):
            events = [ds[q - h]["event"] if q - h >= 0 else len(W.EVENTS) for h in range(max(depths) + 1)]
            actions = [ds[q - h]["action"] if q - h >= 0 else 2 for h in range(1, max(depths) + 2)]
            rows.append((i, events, actions, x["E"], x["N"]))
    lives = np.array([r[0] for r in rows])
    Y = np.array([[r[3], r[4]] for r in rows], float)
    onehot = lambda v, n: np.eye(n)[v]
    out = {}
    for k in depths:
        X = np.array([np.concatenate([np.concatenate([onehot(r[1][h], len(W.EVENTS) + 1), onehot(r[2][h], 3)])
                                      for h in range(k + 1)]) for r in rows])
        out[f"event_and_last_action_plus_{k}_earlier_turns"] = r2(X, Y, lives)
    return out


def untrained(held):
    import torch
    torch.set_num_threads(1)
    out = {}
    for seed in TL.SEEDS:
        out[str(seed)] = {}
        for arm in TL.ARMS:
            net = TL.model(seed)
            net.eval()
            out[str(seed)][arm] = TL.probe(TL.Reader(net, arm), held)
    return out


def main():
    held = TL.lives()
    result = {"references": references(held), "untrained": untrained(held)}
    loops = [result["untrained"][str(s)]["loop"] for s in TL.SEEDS]
    result["untrained_loop_mean"] = {k: float(np.mean([x[k] for x in loops])) for k in ("E", "N")}
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
