"""Pre-answer readouts on disjoint tasks; no claim of native introspection.

NumPy only. Frozen ridge readouts are selected on validation Brier, then each
test forecast is durably recorded before the first answer token is sampled.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from research.cross_model_prediction import (
    CELLS, MODELS, SETTINGS, SOLVE, append, canonical, digest, grade, make_plan as old_plan,
)
from research.iphone_coupling_report import require, strict_json

SEED = 202609151
COUNTS = {"train": 64, "validation": 16, "test": 32}
ALPHAS = (.001, .01, .1, 1.)
PROJECTIONS = {"input": 8031, "middle": 8032, "final": 8033}
DIM = 128
NAMES = ("betaCell", "inputOnly", "internal", "shuffledLabels", "donorState")


def cell(task):
    return (task["family"], task["level"])


def make_plan():
    rng = random.Random(SEED)
    seen = {t["question"] for t in old_plan()["tasks"]}
    tasks = []
    for split, count in COUNTS.items():
        for block in range(count):
            cells = list(CELLS)
            rng.shuffle(cells)
            for family, level in cells:
                while True:
                    letters = "".join(rng.choice("ABCD") for _ in range(level)) if family == "countA" else ""
                    operands = [rng.randrange(10, 100) for _ in range(level)] if family == "alternatingSum" else []
                    question = (f"Combien de lettres A contient cette chaîne : {letters} ?" if letters else
                        "Calcule " + "".join(("" if i == 0 else " + " if i % 2 == 0 else " - ") + str(n)
                                               for i, n in enumerate(operands)) + ".")
                    if question not in seen:
                        seen.add(question)
                        break
                index = len(tasks)
                tasks.append(dict(id=index, split=split, family=family, level=level,
                    letters=letters, operands=operands, question=question,
                    seed=int(digest([SEED, index, "answer"])[:8], 16)))
    return dict(schema="menia-activation-plan-v1", seed=SEED, counts=COUNTS,
                model=MODELS["A"], settings=SETTINGS, projections=PROJECTIONS,
                projectionDimension=DIM, alphas=list(ALPHAS), tasks=tasks)


def messages(task):
    return [dict(role="system", content=SOLVE), dict(role="user", content=task["question"])]


def public_features(task):
    """Fixed signed character 1/2/3-grams in four positional bins; no outcome."""
    x = np.zeros(256 + len(CELLS), dtype=np.float64)
    x[256 + CELLS.index(cell(task))] = 1.
    text = task["question"]
    for size in (1, 2, 3):
        for i in range(len(text) - size + 1):
            key = f"{min(3, 4*i//len(text))}:{text[i:i+size]}".encode("utf-8")
            h = hashlib.sha256(key).digest()
            x[int.from_bytes(h[:4], "little") % 256] += 1 if h[4] % 2 else -1
    x[:256] /= max(np.linalg.norm(x[:256]), 1.)
    return x


def validate_state(state):
    require(set(state) == set(PROJECTIONS), "State fields differ")
    for key in PROJECTIONS:
        v = state[key]
        require(type(v) is list and len(v) == DIM, "State dimension differs")
        require(all(type(x) in (float, int) and np.isfinite(x) for x in v), "Non-finite state")


def features(task, state, *, internal):
    validate_state(state)
    parts = [public_features(task), state["input"]]
    if internal:
        parts += [state["middle"], state["final"]]
    return np.concatenate(parts)


def ridge_fit(x, y, alpha):
    """min mean squared residual + alpha ||w||²; unpenalized intercept."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    require(x.ndim == 2 and y.shape == (len(x),) and len(x) > 0, "Training dimensions")
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1.
    z, intercept = (x-mean)/scale, float(y.mean())
    penalty = len(x)*alpha
    # Equivalent primal/dual solutions; bound the matrix size by sample count.
    if z.shape[1] > len(z):
        w = z.T @ np.linalg.solve(z @ z.T + penalty*np.eye(len(z)), y-intercept)
    else:
        w = np.linalg.solve(z.T @ z + penalty*np.eye(z.shape[1]), z.T @ (y-intercept))
    return dict(mean=mean.tolist(), scale=scale.tolist(), weights=w.tolist(), intercept=intercept, alpha=alpha)


def predict(model, x):
    z = (np.asarray(x)-model["mean"])/model["scale"]
    return np.clip(z @ np.asarray(model["weights"]) + model["intercept"], 0., 1.)


def usable(rows, split):
    return [r for r in rows if r["task"]["split"] == split and r["result"]["status"] == "ok"]


def fit_bundle(rows):
    """Explicitly ignore all test rows, including their states and labels."""
    train, val = usable(rows, "train"), usable(rows, "validation")
    for group, minimum in ((train, 32), (val, 8)):
        require(all(sum(cell(r["task"]) == c for r in group) >= minimum for c in CELLS),
                "Too few usable training/validation rows; keep incomplete attempt")
    y = np.array([grade(r["result"], r["task"]) for r in train], dtype=float)
    vy = np.array([grade(r["result"], r["task"]) for r in val], dtype=float)
    shuffled = y.copy()
    rng = np.random.default_rng(SEED + 9)
    for c in CELLS:
        ids = [i for i, r in enumerate(train) if cell(r["task"]) == c]
        shuffled[ids] = rng.permutation(y[ids])
    models, candidates = {}, {}
    for name in ("inputOnly", "internal", "shuffledLabels"):
        internal = name != "inputOnly"
        x = np.array([features(r["task"], r["state"], internal=internal) for r in train])
        vx = np.array([features(r["task"], r["state"], internal=internal) for r in val])
        best, losses = None, []
        for alpha in ALPHAS:
            model = ridge_fit(x, shuffled if name == "shuffledLabels" else y, alpha)
            error = float(np.mean((predict(model, vx)-vy)**2))
            losses.append(dict(alpha=alpha, validationBrier=error))
            if best is None or error < best[0]:
                best = error, model
        models[name], candidates[name] = best[1], losses
    beta = []
    for c in CELLS:
        ids = [i for i, r in enumerate(train) if cell(r["task"]) == c]
        beta.append(float((y[ids].sum()+1)/(len(ids)+2)))
    return dict(models=models, selection=candidates, beta=beta,
                trainCount=len(train), validationCount=len(val),
                fitDataHash=digest(train+val))


def forecast(bundle, rows, task, state):
    result = {"betaCell": bundle["beta"][CELLS.index(cell(task))]}
    for name, model in bundle["models"].items():
        result[name] = float(predict(model, features(task, state, internal=name != "inputOnly")))
    donors = [r for r in usable(rows, "train") if cell(r["task"]) == cell(task)]
    donor = donors[int(digest([SEED, task["id"], "donor"])[:8], 16) % len(donors)]
    replaced = dict(state, middle=donor["state"]["middle"], final=donor["state"]["final"])
    result["donorState"] = float(predict(bundle["models"]["internal"], features(task, replaced, internal=True)))
    return result


def read_journal(path):
    events = [strict_json(s) for s in Path(path).read_text(encoding="utf-8").splitlines()]
    require(bool(events) and events[0].get("event") == "header", "Missing header")
    header, rows, pending, bundle = events[0], [], None, None
    plan = make_plan()
    require(canonical(header["plan"]) == canonical(plan) and header["planHash"] == digest(plan), "Fixed plan mismatch")
    require(header["origin"] in ("transformers_gpu", "synthetic_fixture"), "Unknown origin")
    for event in events[1:]:
        kind = event.get("event")
        if kind == "fit":
            require(pending is None and bundle is None and len(rows) == 480, "Fit must precede test requests")
            bundle = event["bundle"]
            require(event["bundleHash"] == digest(bundle), "Fit checksum mismatch")
            require(bundle["fitDataHash"] == digest(usable(rows, "train")+usable(rows, "validation")), "Fit data mismatch")
        elif kind == "request":
            require(pending is None and len(rows) < len(plan["tasks"]), "Unexpected request")
            task = plan["tasks"][len(rows)]
            require(event == dict(event="request", task=task, messages=messages(task)), "Request/plan mismatch")
            require(task["split"] != "test" or bundle is not None, "Test without frozen monitor")
            pending = dict(task=task)
        elif kind == "state":
            require(pending is not None and "state" not in pending and event["id"] == len(rows), "State ordering")
            validate_state(event["state"])
            expected = forecast(bundle, rows, pending["task"], event["state"]) if bundle else None
            recorded = event["predictions"]
            if expected is None:
                require(recorded is None, "Premature forecast")
            else:
                require(type(recorded) is dict and set(recorded) == set(expected), "Forecast keys differ")
                require(all(type(recorded[k]) in (int, float) and np.isfinite(recorded[k]) and
                            0 <= recorded[k] <= 1 and abs(recorded[k]-expected[k]) < 1e-10 for k in expected),
                        "Pre-answer forecast mismatch")
            pending.update(state=event["state"], predictions=event["predictions"])
        elif kind == "result":
            require(pending is not None and event["id"] == len(rows), "Result ordering")
            require(event["status"] in ("ok", "error", "interrupted"), "Result status")
            require(type(event["text"]) is str, "Raw text missing")
            require(type(event["seconds"]) in (int, float) and np.isfinite(event["seconds"]) and event["seconds"] >= 0, "Duration")
            require("state" in pending if event["status"] == "ok" else event["text"] == "", "State/outcome mismatch")
            rows.append(dict(pending, result=event))
            pending = None
        else:
            raise ValueError("Unknown event")
    return header, rows, pending, bundle


def collect(path, backend, *, resume=False, limit=None):
    path = Path(path)
    if resume:
        header, rows, pending, bundle = read_journal(path)
        require(header["metadata"] == backend.metadata and header["origin"] == backend.origin, "Resume environment differs")
        if pending:
            event = dict(event="result", id=len(rows), status="interrupted", text="", seconds=0., errorType="UnrecordedResult")
            append(path, event)
            rows.append(dict(pending, result=event))
    else:
        path.parent.mkdir(exist_ok=True, parents=True)
        plan = make_plan()
        header = dict(event="header", plan=plan, planHash=digest(plan), origin=backend.origin, metadata=backend.metadata)
        with path.open("x", encoding="utf-8") as f:
            f.write(canonical(header)+"\n")
            f.flush()
            import os
            os.fsync(f.fileno())
        rows, bundle = [], None
    start = len(rows)
    for task in header["plan"]["tasks"][start:]:
        if limit is not None and len(rows)-start >= limit:
            break
        if task["split"] == "test" and bundle is None:
            bundle = fit_bundle(rows)
            append(path, dict(event="fit", bundle=bundle, bundleHash=digest(bundle)))
        append(path, dict(event="request", task=task, messages=messages(task)))
        row = dict(task=task)
        def capture(state):
            require("state" not in row, "Capture must occur once before the first token")
            validate_state(state)
            predictions = forecast(bundle, rows, task, state) if bundle else None
            append(path, dict(event="state", id=task["id"], state=state, predictions=predictions))
            row.update(state=state, predictions=predictions)
        started = time.perf_counter()
        try:
            text, metrics = backend.generate(task, capture)
            require("state" in row and type(text) is str, "Missing pre-answer capture")
            event = dict(event="result", id=task["id"], status="ok", text=text,
                         seconds=time.perf_counter()-started, metrics=metrics)
        except (Exception, KeyboardInterrupt) as exc:
            event = dict(event="result", id=task["id"], status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                         text="", seconds=time.perf_counter()-started, errorType=type(exc).__name__)
            append(path, event)
            raise
        append(path, event)
        rows.append(dict(row, result=event))
        print(f"{len(rows)}/672 {task['split']} ok", flush=True)
    return len(rows)


def metrics(rows, name):
    if not rows:
        return dict(n=0, brier=None, auc=None, direct=0, wrongDirect=0, pointLoss=None)
    p = np.array([r["predictions"][name] for r in rows])
    y = np.array([grade(r["result"], r["task"]) for r in rows], dtype=float)
    good, bad = p[y == 1], p[y == 0]
    auc = float(np.mean((good[:, None] > bad) + .5*(good[:, None] == bad))) if len(good) and len(bad) else None
    return dict(n=len(rows), correct=int(y.sum()), meanProbability=float(p.mean()),
                brier=float(np.mean((p-y)**2)), auc=auc, direct=int(sum(p >= .8)),
                wrongDirect=int(sum((p >= .8) & (y == 0))),
                pointLoss=float(np.mean(np.where(p >= .8, 1-y, .2))))


def analyze(path, *, check_fit=True):
    header, rows, pending, bundle = read_journal(path)
    if bundle and check_fit:
        refit = fit_bundle(rows)
        require(refit["fitDataHash"] == bundle["fitDataHash"], "Training records mismatch")
        require(set(refit) == set(bundle) and set(refit["models"]) == set(bundle["models"]), "Fit fields differ")
        require(refit["trainCount"] == bundle["trainCount"] and refit["validationCount"] == bundle["validationCount"], "Fit counts differ")
        require(refit["beta"] == bundle["beta"], "Beta reference mismatch")
        for name in bundle["models"]:
            require(refit["models"][name]["alpha"] == bundle["models"][name]["alpha"], "Selection mismatch")
            selected = bundle["selection"][name]
            require([x["alpha"] for x in selected] == list(ALPHAS), "Selection grid differs")
            require(np.allclose([x["validationBrier"] for x in selected],
                                [x["validationBrier"] for x in refit["selection"][name]],
                                atol=1e-8, rtol=1e-8), "Validation losses mismatch")
            for key in ("weights", "mean", "scale", "intercept"):
                require(np.allclose(refit["models"][name][key], bundle["models"][name][key], atol=1e-8, rtol=1e-8), "Refitted monitor mismatch")
    test = usable(rows, "test")
    scores = {name: metrics(test, name) for name in NAMES}
    contrasts = None
    if len(test) == 192:
        y = np.array([grade(r["result"], r["task"]) for r in test], dtype=float)
        own = (np.array([r["predictions"]["internal"] for r in test])-y)**2
        contrasts = {}
        rng = np.random.default_rng(SEED+20)
        groups = [np.array([i for i, r in enumerate(test) if cell(r["task"]) == c]) for c in CELLS]
        draws = np.concatenate([rng.choice(ids, (2000, len(ids)), replace=True) for ids in groups], axis=1)
        for name in ("inputOnly", "betaCell", "shuffledLabels", "donorState"):
            delta = (np.array([r["predictions"][name] for r in test])-y)**2 - own
            contrasts[name] = dict(otherMinusInternalBrier=float(delta.mean()),
                descriptivePairedBootstrap95=np.quantile(delta[draws].mean(1), [.025, .975]).tolist())
    return dict(schema="menia-activation-report-v1", origin=header["origin"], planHash=header["planHash"],
        model=header["plan"]["model"], settings=header["plan"]["settings"],
        recordedResults=len(rows), plannedResults=672, pendingRequest=pending is not None,
        statuses=dict(Counter(r["result"]["status"] for r in rows)),
        complete=len(rows) == 672 and all(r["result"]["status"] == "ok" for r in rows),
        fitChecked=bool(bundle and check_fit), selection=bundle["selection"] if bundle else None,
        usedForFit={split: len(usable(rows, split)) for split in ("train", "validation")},
        availableTestTargets=len(test), missingTestTargets=192-len(test),
        scores=scores, contrasts=contrasts,
        withinCell=[dict(family=c[0], level=c[1], scores={name: metrics([r for r in test if cell(r["task"]) == c], name) for name in NAMES}) for c in CELLS],
        interpretation="Frozen added monitor, not native LLM self-report. Conditional descriptive uncertainty on one fitted monitor. No consciousness or privileged-access inference.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.journal)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
