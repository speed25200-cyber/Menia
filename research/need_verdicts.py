"""Directions of the need and verdicts of docs/LLM_NEED_PROTOCOL.md, from the published lives (numpy only).

fit_directions: for each candidate block, ridge regressions (alpha 100, standardized activations, 5 folds by lives)
predict E and N from the residual stream at each decision; the block with the best mean R^2 is kept; there, a
linear regression of the activation on E, N, the event, the previous choice and the turn gives b_E and b_N, the
change of activation per unit of each need, the rest being equal. The verdicts S, IA, IR and LS are recomputed from
the lives of the test, with bootstrap intervals by life (10 000 draws).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W

ALPHA = 100.0
FOLDS = 5
DRAWS = 10000
EFFECT = 0.15
SHARE = 1 / 3


def states(lives):
    """E, N, event, previous choice and turn of every decision, in the order the decisions were made."""
    rows = []
    for index, life in enumerate(lives):
        previous = None
        for t in W.decisions(life):
            rows.append({"life": index, "E": t["E"], "N": t["N"], "event": t["event"], "previous": previous, "t": t["t"]})
            previous = t["action"]
    return rows


def shallow_features(rows):
    x = np.zeros((len(rows), len(W.EVENTS) + 3))
    for i, r in enumerate(rows):
        x[i, r["event"]] = 1
        if r["previous"] is not None:
            x[i, len(W.EVENTS) + r["previous"]] = 1
        x[i, -1] = r["t"] / W.TURNS
    return x


def cv_r2(x, y, groups, alpha=ALPHA, folds=FOLDS):
    """Cross-validated R^2 of a ridge regression, folds by life, features standardized on the training folds."""
    fold = np.array(groups) % folds
    pred = np.zeros(len(y))
    for f in range(folds):
        tr, te = fold != f, fold == f
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
        a, b = (x[tr] - mu) / sd, (x[te] - mu) / sd
        ym = y[tr].mean()
        if a.shape[1] > a.shape[0]:
            w = a.T @ np.linalg.solve(a @ a.T + alpha * np.eye(a.shape[0]), y[tr] - ym)
        else:
            w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (y[tr] - ym))
        pred[te] = b @ w + ym
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def need_r2(lives, acts):
    rows = states(lives)
    groups = [r["life"] for r in rows]
    out = {}
    for need in ("E", "N"):
        y = np.array([r[need] for r in rows], float)
        out[need] = cv_r2(np.asarray(acts, np.float64), y, groups)
        out[f"{need}_shallow"] = cv_r2(shallow_features(rows), y, groups)
    return out


def fit_directions(lives, acts_by_layer):
    rows = states(lives)
    by_layer = {int(j): need_r2(lives, a) for j, a in acts_by_layer.items()}
    layer = max(by_layer, key=lambda j: (by_layer[j]["E"] + by_layer[j]["N"]) / 2)
    acts = np.asarray(acts_by_layer[layer], np.float64)
    design = np.column_stack([np.ones(len(rows)), [r["E"] for r in rows], [r["N"] for r in rows],
                              shallow_features(rows)[:, 1:]])  # event 0 is the baseline of the event dummies
    coef, *_ = np.linalg.lstsq(design, acts, rcond=None)
    b_e, b_n = coef[1], coef[2]
    u, _ = np.linalg.qr(np.stack([b_e, b_n], axis=1))
    return {"layer": layer, "r2_by_layer": {str(j): v for j, v in by_layer.items()},
            "r2_shallow": {"E": by_layer[layer]["E_shallow"], "N": by_layer[layer]["N_shallow"]},
            "b_E": b_e.tolist(), "b_N": b_n.tolist(), "U": u.tolist(), "mean": (acts @ u).mean(axis=0).tolist(),
            "norm_b_E": float(np.linalg.norm(b_e)), "norm_b_N": float(np.linalg.norm(b_n)), "decisions": len(rows)}


def load_direction(folder):
    d = json.loads((Path(folder) / "direction.json").read_text())
    return {**d, "b_E": np.array(d["b_E"]), "b_N": np.array(d["b_N"]), "U": np.array(d["U"]),
            "mean": np.array(d["mean"])}


def bootstrap(values_by_life, draws=DRAWS, seed=0):
    """Mean over all values and the 95% interval of the mean when lives are resampled."""
    lives = [np.asarray(v, float) for v in values_by_life if len(v)]
    total = np.concatenate(lives)
    sums, counts = np.array([v.sum() for v in lives]), np.array([len(v) for v in lives])
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(lives), size=(draws, len(lives)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    return {"mean": float(total.mean()), "low": float(np.quantile(means, 0.025)),
            "high": float(np.quantile(means, 0.975)), "n": int(len(total))}


def paired_survival(a, b):
    return bootstrap([[float(x["survived"]) - float(y["survived"])] for x, y in zip(a, b)])


def contexts(lives):
    return [[t["extra"]["injection"] for t in W.decisions(life) if "injection" in t.get("extra", {})] for life in lives]


def effect(ctx, condition, key, sign=1.0):
    return bootstrap([[sign * (c[condition][key] - c["none"][key]) for c in life] for life in ctx])


def random_effect(ctx, prefix, key, sign=1.0):
    names = sorted({n for life in ctx for c in life for n in c if n.startswith(prefix)})
    return float(np.mean([abs(effect(ctx, n, key, sign)["mean"]) for n in names])) if names else None


def report_accuracy(lives):
    out = {}
    for need in ("E", "N"):
        pairs = [(t["extra"][f"report_{need}"][0] > 0.5, t[need] <= W.LOW) for life in lives for t in W.decisions(life)
                 if f"report_{need}" in t.get("extra", {})]
        rates = [np.mean([p == truth for p, truth in pairs if truth == c]) for c in (True, False)
                 if any(truth == c for _, truth in pairs)]
        out[need] = float(np.mean(rates)) if rates else None
    return out


def verdicts(final, control, base, lesion, lesion_random):
    passes = lambda d: d["mean"] >= EFFECT and d["low"] > 0
    s_control, s_base = paired_survival(final, control), paired_survival(final, base)
    ctx = contexts(final)
    n_ctx = sum(len(c) for c in ctx)
    ia_e, ia_n = effect(ctx, "E", "p_R"), effect(ctx, "N", "p_R", -1.0)
    rand_ia_e, rand_ia_n = random_effect(ctx, "randE", "p_R"), random_effect(ctx, "randN", "p_R", -1.0)
    ir_e, ir_n = effect(ctx, "E", "report_E"), effect(ctx, "N", "report_N")
    cross_e, cross_n = effect(ctx, "E", "report_N"), effect(ctx, "N", "report_E")
    rand_ir_e, rand_ir_n = random_effect(ctx, "randE", "report_E"), random_effect(ctx, "randN", "report_N")
    drop, drop_random = paired_survival(final, lesion), paired_survival(final, lesion_random)
    within = lambda x, ref: x is not None and abs(x) <= SHARE * ref["mean"]
    ds = [t for life in final for t in W.decisions(life)]
    mass_choice = float(np.mean([t["mass"] for t in ds]))
    mass_report = float(np.mean([t["extra"][f"report_{n}"][1] for t in ds for n in ("E", "N")
                                 if f"report_{n}" in t.get("extra", {})]))
    valid = mass_choice >= 0.5 and mass_report >= 0.5 and n_ctx >= 100
    out = {"valid": bool(valid),
           "S": bool(passes(s_control) and passes(s_base)),
           "IA": bool(passes(ia_e) and passes(ia_n) and within(rand_ia_e, ia_e) and within(rand_ia_n, ia_n)),
           "IR": bool(passes(ir_e) and passes(ir_n) and within(cross_e["mean"], ir_e) and within(cross_n["mean"], ir_n)
                      and within(rand_ir_e, ir_e) and within(rand_ir_n, ir_n)),
           "LS": bool(passes(drop) and within(drop_random["mean"], drop))}
    out["global"] = bool(valid and out["S"] and out["IA"] and out["IR"] and out["LS"])
    survival = {name: float(np.mean([l["survived"] for l in lives])) for name, lives in
                (("final", final), ("control", control), ("base", base), ("lesion", lesion),
                 ("lesion_random", lesion_random))}
    values = {"survival": survival, "final_minus_control": s_control, "final_minus_base": s_base,
              "contexts": n_ctx, "mass_choice": mass_choice, "mass_report": mass_report,
              "IA": {"energy_low_raises_R": ia_e, "food_low_raises_M": ia_n, "random_R": rand_ia_e, "random_M": rand_ia_n},
              "IR": {"energy_low_raises_yes_E": ir_e, "food_low_raises_yes_N": ir_n, "energy_low_on_yes_N": cross_e,
                     "food_low_on_yes_E": cross_n, "random_E": rand_ir_e, "random_N": rand_ir_n},
              "LS": {"lesion_drop": drop, "random_lesion_drop": drop_random},
              "report_accuracy": report_accuracy(final),
              "turns_need_at_most_2": {name: W.low_turns(lives) for name, lives in
                                       (("final", final), ("control", control), ("base", base), ("lesion", lesion),
                                        ("lesion_random", lesion_random))}}
    return {"verdicts": out, "values": values}


read_jsonl = W.read_jsonl


def curves(root):
    """Survival, share of choices serving the lower need (needs at least 2 apart) and kept choices, per arm and
    round, recomputed from the lives of the learning rounds."""
    out = {}
    for path in sorted(Path(root).glob("*/lives-*-*.jsonl.gz")):
        arm, k = path.name[len("lives-"):-len(".jsonl.gz")].rsplit("-", 1)
        if arm not in W.ARMS:
            continue
        lives = read_jsonl(path)
        ds = [t for l in lives for t in W.decisions(l) if abs(t["E"] - t["N"]) >= 2]
        out.setdefault(arm, {})[k] = {
            "survival": float(np.mean([l["survived"] for l in lives])),
            "serves_lower_need": float(np.mean([(t["action"] == 0) == (t["E"] < t["N"]) for t in ds])),
            "kept": int(sum(map(sum, [l["kept"] for l in lives]))),
            "turns_need_at_most_2": W.low_turns(lives)}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/llm-need")
    parser.add_argument("--test", default="final/test")
    parser.add_argument("--direction", default="final/direction")
    parser.add_argument("--check", action="store_true", help="recompute and compare with verdicts.json")
    a = parser.parse_args(argv)
    root = Path(a.root)
    test = root / a.test
    lives = {n: read_jsonl(test / f"lives-test-{n}.jsonl.gz") for n in ("final", "control", "base", "lesion", "lesion_random")}
    result = verdicts(**lives)
    folder = root / a.direction
    published = json.loads((folder / "direction.json").read_text())
    acts = np.load(folder / "direction-activations.npz")["activations"].astype(np.float32)
    refit = fit_directions(read_jsonl(folder / "lives-direction.jsonl.gz"), {published["layer"]: acts})
    result["direction"] = {"layer": published["layer"], "r2_by_layer": published["r2_by_layer"],
                           "r2_shallow": published["r2_shallow"],
                           "refit_from_float16_matches": bool(
                               np.allclose(refit["b_E"], published["b_E"], rtol=0.05, atol=1e-3 * published["norm_b_E"])
                               and np.allclose(refit["b_N"], published["b_N"], rtol=0.05, atol=1e-3 * published["norm_b_N"]))}
    result["curves"] = curves(root)
    result = json.loads(json.dumps(result))
    output = root / "verdicts.json"
    if a.check:
        differs = json.loads(output.read_text()) != result
        print("differs:", "yes" if differs else "none")
        if differs:
            raise SystemExit(1)
        return
    output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["verdicts"], indent=1))


if __name__ == "__main__":
    main()
