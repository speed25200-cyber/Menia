"""What does the gathered state carry: the need itself, or only the decision? (exploratory, 1 October 2026)

After test 11 failed for the second agent, an exploratory question, written before looking: at the block where each
agent gathers its need, on the three tokens « Cho » « ix » « : », can its energy level E be read linearly, or only
the balance E − N that decides between R and M? Each agent's 128 direction lives (already recorded) are replayed
through a key-value cache, the block output on those tokens is captured at every decision (massive dimensions
excluded), and ridge regressions, cross-validated by life, read E, N, E − N, E + N, "E low" (E ≤ 3) and P(R).
Nothing here is a pre-registered verdict. Capturing needs torch; the analysis is numpy only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W

AGENTS = {
    "first": {"adapter": "artifacts/llm-need/final/report/adapters-final", "lives": "artifacts/llm-need/reader/test"},
    "second": {"adapter": "artifacts/llm-need/r1/final/report/adapters-final",
               "lives": "artifacts/llm-need/locate/second/act"},
    "two": {"adapter": "artifacts/llm-need/two/need-5-8/adapters-need-8",
            "lives": "artifacts/llm-need/locate-two/slice-5/measure/two/act"},
}
TOKENS = 3  # " Cho" "ix" " :"
FOLDS = 4
ALPHAS = (1e1, 1e2, 1e3, 1e4, 1e5)


def capture(name, out, threads=2, replay_tolerance=1e-3):
    """Block-b outputs on the three tokens of every decision of the agent's direction lives, b its gathering block."""
    import torch
    from .need_causal import TorchAgent
    torch.set_num_threads(threads)
    spec = AGENTS[name]
    direction = json.loads((Path(spec["lives"]) / "direction.json").read_text())
    agent = TorchAgent(spec["adapter"], tail=TOKENS, block=direction["block"])
    states, rows, gaps = [], [], []
    for i, life in enumerate(W.read_jsonl(Path(spec["lives"]) / "lives-direction.jsonl.gz")):
        agent.start(W.HEADER)
        for turn in W.decisions(life):
            agent.capture = True
            p, _ = agent.decide(W.choice_line(turn["t"], turn["event"], turn.get("other")))
            agent.capture = False
            gaps.append(abs(p - turn["p_R"]))
            states.append(agent.captured.astype(np.float32))
            rows.append([i, turn["E"], turn["N"], turn["action"], turn["p_R"]])
            agent.commit(turn["action"])
    gap = float(np.mean(gaps))
    if gap > replay_tolerance:
        raise RuntimeError(f"replay differs from the recorded lives: mean |ΔP(R)| = {gap}")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", states=np.array(states, np.float16), rows=np.array(rows, float),
                        massive=np.array(direction["massive_dims"], int), block=direction["block"], replay_gap=gap)
    return gap


def ridge_cv(x, y, groups, alphas=ALPHAS, folds=FOLDS):
    """Out-of-fold predictions of a ridge regression (alpha chosen inside each training fold by a nested split),
    folds by life so that no life is in both a training and a test set."""
    lives = np.unique(groups)
    rng = np.random.default_rng([W.SEED, 7, 1])
    fold_of = dict(zip(rng.permutation(lives), np.arange(len(lives)) % folds))
    fold = np.array([fold_of[g] for g in groups])
    pred = np.zeros(len(y))
    for f in range(folds):
        train, test = fold != f, fold == f
        inner = fold[train] == (f + 1) % folds  # a held-out part of the training folds picks alpha
        best = min(alphas, key=lambda a: np.mean((fit_predict(x[train][~inner], y[train][~inner], x[train][inner], a)
                                                  - y[train][inner]) ** 2))
        pred[test] = fit_predict(x[train], y[train], x[test], best)
    return pred


def fit_predict(x, y, x_new, alpha):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xs, xn = (x - mu) / sd, (x_new - mu) / sd
    ym = y.mean()
    k = xs @ xs.T  # dual form: n is a few thousand, the features 3 x 1 024
    w = np.linalg.solve(k + alpha * np.eye(len(k)), y - ym)
    return (xn @ xs.T) @ w + ym


def r2(y, pred):
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def balanced(y, pred):
    """Balanced accuracy of pred > 0.5 against the boolean y."""
    return float(np.mean([np.mean((pred[y == c] > 0.5) == c) for c in (True, False) if np.any(y == c)]))


def analyze(path):
    data = np.load(path)
    rows, states = data["rows"], data["states"].astype(np.float32)
    keep = np.setdiff1d(np.arange(states.shape[2]), data["massive"])
    x = states[:, :, keep].reshape(len(states), -1)
    groups, e, n, p_r = rows[:, 0].astype(int), rows[:, 1], rows[:, 2], rows[:, 4]
    targets = {"E": e, "N": n, "E-N": e - n, "E+N": e + n, "P(R)": p_r}
    out = {"decisions": int(len(rows)), "lives": int(len(np.unique(groups))), "block": int(data["block"]),
           "replay_gap": float(data["replay_gap"]), "r2": {}}
    preds = {}
    for name, y in targets.items():
        preds[name] = ridge_cv(x, y, groups)
        out["r2"][name] = round(r2(y, preds[name]), 4)
    low = (e <= W.LOW).astype(float)
    out["E_low_balanced_accuracy"] = round(balanced(low.astype(bool), ridge_cv(x, low, groups)), 4)
    # the same question read from the predicted level (a rare class: a 0/1 regression rarely crosses 0.5)
    out["E_low_from_E_balanced_accuracy"] = round(balanced(low.astype(bool), (preds["E"] <= W.LOW + 0.5) * 1.0), 4)
    out["E_low_share"] = round(float(low.mean()), 4)
    # E once the balance is known: what the state says of E beyond E - N (the residual of E on E - N)
    slope = np.polyfit(e - n, e, 1)
    resid = e - np.polyval(slope, e - n)
    out["r2"]["E | E-N"] = round(r2(resid, ridge_cv(x, resid, groups)), 4)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--agent", choices=sorted(AGENTS), required=True)
    c.add_argument("--threads", type=int, default=2)
    a_ = sub.add_parser("analyze")
    for s in (c, a_):
        s.add_argument("--states", required=True, help="folder of the captured states (large; kept out of git)")
    a_.add_argument("--out", default="artifacts/llm-need/probe/probe.json")
    a = parser.parse_args(argv)
    if a.command == "capture":
        print(json.dumps({"agent": a.agent, "replay_gap": capture(a.agent, Path(a.states), a.threads)}))
        return
    result = {n: analyze(Path(a.states) / f"{n}.npz") for n in AGENTS if (Path(a.states) / f"{n}.npz").exists()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
