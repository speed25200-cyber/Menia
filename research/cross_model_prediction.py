"""Prospective Colab pilot: two fixed models, two name mappings, six task cells.

Pure-Python plan, append-only collection and reconstruction. The GPU backend is
separate. Predictions never receive target evaluation outputs or answer keys.
This is a behavioral pilot, not a measure or implementation of consciousness.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re

from research.iphone_capability_learning_report import probability, loss
from research.iphone_coupling_report import require, strict_json, finite_number

MODELS = {
    "A": {"id": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c"},
    "B": {"id": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218"},
}
SETTINGS = dict(temperature=.7, top_p=.8, top_k=20, min_p=0., do_sample=True,
                max_new_tokens=256, max_input_tokens=1792, enable_thinking=False,
                dtype="bfloat16", attention="sdpa", quantization="none",
                repetition_penalty=1., renormalize_logits=True, use_cache=True)
CELLS = (("countA", 8), ("countA", 24), ("countA", 64),
         ("alternatingSum", 2), ("alternatingSum", 4), ("alternatingSum", 8))
VIEWS = ("normal", "swapped")
SEED = 20260915
SOLVE = "Réponds uniquement par l'entier demandé, sans explication ni autre texte."
FORECAST = ('Prédis la probabilité que le modèle cible réponde correctement à la question dans un appel séparé, '
            'sans outil et sans historique, avec les paramètres indiqués. La réussite exige uniquement l\'entier exact. '
            'Le nom de la cible est un code arbitraire. history contient ses essais de calibration dans cette même '
            'famille et au même niveau ; answer est limité à 96 caractères, correct évalue la réponse entière. '
            'Une erreur technique a correct=null. Ne donne pas la solution. Réponds uniquement par un objet JSON '
            'avec une seule clé p, un nombre entre 0 et 1 : {"p":0.5}.')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def reference(task):
    if task["family"] == "countA":
        return task["letters"].count("A")
    return sum(task["operands"][::2]) - sum(task["operands"][1::2])


def make_plan(seed=SEED):
    require(type(seed) is int and 0 <= seed < 2**32, "Invalid plan seed")
    rng = random.Random(seed)
    tasks, calls, seen = [], [], set()
    for phase, count in (("calibration", 4), ("evaluation", 6)):
        for block in range(count):
            cells = list(CELLS)
            rng.shuffle(cells)
            for family, level in cells:
                while True:
                    letters = "".join(rng.choice("ABCD") for _ in range(level)) if family == "countA" else ""
                    operands = [rng.randrange(10, 100) for _ in range(level)] if family == "alternatingSum" else []
                    key = (family, letters, *operands)
                    if key not in seen:
                        seen.add(key)
                        break
                question = (f"Combien de lettres A contient cette chaîne : {letters} ?" if family == "countA" else
                            "Calcule " + "".join(("" if i == 0 else " + " if i % 2 == 0 else " - ") + str(n)
                                                  for i, n in enumerate(operands)) + ".")
                task = dict(id=len(tasks), phase=phase, block=block, family=family,
                            level=level, letters=letters, operands=operands, question=question)
                tasks.append(task)
                if phase == "evaluation":
                    forecasts = list(itertools.product(MODELS, MODELS, VIEWS))
                    rng.shuffle(forecasts)
                    for predictor, target, view in forecasts:
                        calls.append(dict(task=task["id"], kind="forecast", model=predictor, target=target, view=view))
                targets = list(MODELS)
                rng.shuffle(targets)
                calls.extend(dict(task=task["id"], kind="solve", model=t, target=t, view=None) for t in targets)
    for i, call in enumerate(calls):
        call["id"] = i
        # Paired name mappings use the same RNG seed; solve and forecast streams differ.
        call["seed"] = int(digest([seed, call["task"], call["kind"], call["model"], call["target"]])[:8], 16)
    return json.loads(canonical(dict(schema="menia-cross-model-plan-v1", seed=seed,
                                    models=MODELS, settings=SETTINGS, tasks=tasks, calls=calls)))


def grade(result, task):
    if result["status"] != "ok":
        return None
    raw = result["text"].strip()
    return bool(re.fullmatch(r"-?(?:0|[1-9][0-9]*)", raw) and int(raw) == reference(task))


def history_for(plan, results, task, target):
    history = []
    for call, result in zip(plan["calls"], results):
        previous = plan["tasks"][call["task"]]
        if (call["kind"] == "solve" and call["target"] == target and previous["phase"] == "calibration"
                and previous["family"] == task["family"] and previous["level"] == task["level"]):
            history.append(dict(question=previous["question"], answer=result.get("text", "")[:96],
                                status=result["status"], correct=grade(result, previous)))
    return history


def request_for(plan, index, results):
    call = plan["calls"][index]
    task = plan["tasks"][call["task"]]
    if call["kind"] == "solve":
        messages = [dict(role="system", content=SOLVE), dict(role="user", content=task["question"])]
    else:
        names = dict(A="Quartz", B="Jade") if call["view"] == "normal" else dict(A="Jade", B="Quartz")
        history = history_for(plan, results, task, call["target"])
        require(len(history) == 4, "Calibration must precede every forecast")
        payload = dict(target=dict(name=names[call["target"]], **plan["models"][call["target"]]),
                       generation=plan["settings"], family=task["family"], level=task["level"],
                       question=task["question"], history=history)
        messages = [dict(role="system", content=FORECAST), dict(role="user", content=canonical(payload))]
    return dict(event="request", call=call, messages=messages)


def append(path, event):
    # Each intent and result is durable before the next call; existing lines are never overwritten.
    with Path(path).open("a", encoding="utf-8", newline="\n") as f:
        f.write(canonical(event)+"\n")
        f.flush()
        os.fsync(f.fileno())


def read_journal(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    require(bool(lines), "Empty journal")
    events = [strict_json(line) for line in lines]
    header = events[0]
    require(header.get("event") == "header", "Missing header")
    plan = header["plan"]
    require(canonical(plan) == canonical(make_plan(plan["seed"])), "Plan differs from fixed protocol")
    require(header["planHash"] == digest(plan), "Wrong plan digest")
    require(header["origin"] in ("transformers_gpu", "synthetic_fixture"), "Unknown collection origin")
    results, pending = [], None
    for event in events[1:]:
        require(len(results) < len(plan["calls"]), "Extra call beyond plan")
        if pending is None:
            require(canonical(event) == canonical(request_for(plan, len(results), results)), "Request/order/history mismatch")
            pending = event
        else:
            require(event.get("event") == "result" and type(event.get("callId")) is int
                    and event["callId"] == len(results), "Result without matching request")
            require(event.get("status") in ("ok", "error", "interrupted"), "Invalid result status")
            require(type(event.get("text")) is str, "Missing raw output")
            require(finite_number(event.get("seconds")) and event["seconds"] >= 0, "Invalid elapsed time")
            if event["status"] != "ok":
                require(event["text"] == "", "Failed call cannot provide an answer")
            results.append(event)
            pending = None
    return header, results, pending


def collect(path, backend, *, seed=SEED, resume=False, limit=None):
    """backend.generate receives only a request, never the plan or answer keys."""
    import time
    path = Path(path)
    require(limit is None or type(limit) is int and limit >= 0, "Invalid call limit")
    if resume:
        header, results, pending = read_journal(path)
        require(canonical(header["metadata"]) == canonical(backend.metadata) and header["origin"] == backend.origin,
                "Resume requires identical recorded backend/environment")
        require(seed == header["plan"]["seed"], "Resume seed differs")
        if pending:
            append(path, dict(event="result", callId=len(results), status="interrupted", text="", seconds=0.,
                              errorType="UnrecordedResult", elapsedKnown=False))
            header, results, pending = read_journal(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        plan = make_plan(seed)
        header = dict(event="header", plan=plan, planHash=digest(plan), origin=backend.origin,
                      metadata=backend.metadata)
        with path.open("x", encoding="utf-8", newline="\n") as f:
            f.write(canonical(header)+"\n")
            f.flush()
            os.fsync(f.fileno())
        results = []
    plan, start = header["plan"], len(results)
    stop = len(plan["calls"]) if limit is None else min(len(plan["calls"]), start+limit)
    for index in range(start, stop):
        request = request_for(plan, index, results)
        append(path, request)
        started = time.perf_counter()
        try:
            text, metrics = backend.generate(request)
            require(type(text) is str and type(metrics) is dict, "Backend response types")
            event = dict(event="result", callId=index, status="ok", text=text,
                         seconds=time.perf_counter()-started, metrics=metrics)
        except (Exception, KeyboardInterrupt) as exc:
            event = dict(event="result", callId=index,
                         status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                         text="", seconds=time.perf_counter()-started, errorType=type(exc).__name__)
            append(path, event)
            results.append(event)
            # Technical failure stops the run. Explicit resume retains it and advances.
            raise
        append(path, event)
        results.append(event)
        print(f"{index+1}/{len(plan['calls'])} {request['call']['kind']} {event['status']}", flush=True)
    return analyze(path)


def mean(values):
    return sum(values)/len(values) if values else None


def score(pairs):
    valid = [(p, y) for p, y in pairs if p is not None]
    good = [p for p, y in valid if y]
    bad = [p for p, y in valid if not y]
    auc = mean([float(g > b)+.5*(g == b) for g in good for b in bad]) if len(valid) == len(pairs) else None
    return dict(n=len(pairs), valid=len(valid), invalid=len(pairs)-len(valid),
                successes=sum(y for _, y in pairs),
                systemBrier=mean([((.5 if p is None else p)-y)**2 for p, y in pairs]),
                validBrier=mean([(p-y)**2 for p, y in valid]),
                counterfactualPointLoss=mean([loss(p, y) for p, y in pairs]), auc=auc)


def analyze(path):
    header, results, pending = read_journal(path)
    plan = header["plan"]
    solutions, predictions = {}, {}
    for call, result in zip(plan["calls"], results):
        task = plan["tasks"][call["task"]]
        if call["kind"] == "solve":
            solutions[task["id"], call["target"]] = grade(result, task)
        else:
            predictions[task["id"], call["model"], call["target"], call["view"]] = (
                probability(result["text"]) if result["status"] == "ok" else None)
    evaluation = [t for t in plan["tasks"] if t["phase"] == "evaluation"]
    views = {}
    for view in VIEWS:
        matrix = {}
        common = {}
        for predictor, target in itertools.product(MODELS, repeat=2):
            pairs, by_cell = [], {cell: [] for cell in CELLS}
            for task in evaluation:
                y = solutions.get((task["id"], target))
                if y is None:
                    continue
                p = predictions.get((task["id"], predictor, target, view))
                pairs.append((p, y))
                by_cell[task["family"], task["level"]].append((p, y))
            matrix[f"{predictor}->{target}"] = dict(**score(pairs),
                missingTargets=len(evaluation)-len(pairs),
                withinCell=[dict(family=f, level=l, **score(by_cell[f,l])) for f,l in CELLS])
        gains = {}
        for target in MODELS:
            other = "B" if target == "A" else "A"
            own, external = matrix[f"{target}->{target}"], matrix[f"{other}->{target}"]
            gains[target] = (external["systemBrier"]-own["systemBrier"] if own["n"] else None)
            paired = []
            for task in evaluation:
                y = solutions.get((task["id"], target))
                a = predictions.get((task["id"], target, target, view))
                b = predictions.get((task["id"], other, target, view))
                if y is not None and a is not None and b is not None:
                    paired.append((b-y)**2-(a-y)**2)
            common[target] = dict(n=len(paired), otherMinusSelfBrier=mean(paired))
        full = all(cell["n"] == 36 for cell in matrix.values())
        views[view] = dict(matrix=matrix, otherMinusSelfByTarget=gains, commonValid=common,
                           diagonalAdvantage=mean(list(gains.values())) if full else None)
    baselines = {}
    for target in MODELS:
        pairs = {name: [] for name in ("empiricalCell", "betaCell", "betaPooled", "constantHalf")}
        cal = [solutions[t["id"], target] for t in plan["tasks"] if t["phase"] == "calibration"
               and solutions.get((t["id"], target)) is not None]
        for task in evaluation:
            y = solutions.get((task["id"], target))
            if y is None:
                continue
            history = history_for(plan, results, task, target)
            values = [h["correct"] for h in history if h["correct"] is not None]
            estimates = dict(empiricalCell=mean(values), betaCell=(sum(values)+1)/(len(values)+2),
                             betaPooled=(sum(cal)+1)/(len(cal)+2), constantHalf=.5)
            for name, p in estimates.items():
                pairs[name].append((p, y))
        baselines[target] = {name: score(p) for name, p in pairs.items()}
    name_effect = {}
    for predictor, target in itertools.product(MODELS, repeat=2):
        paired = [(predictions.get((t["id"], predictor, target, "normal")),
                   predictions.get((t["id"], predictor, target, "swapped"))) for t in evaluation]
        differences = [abs(a-b) for a,b in paired if a is not None and b is not None]
        name_effect[f"{predictor}->{target}"] = dict(validPairs=len(differences), meanAbsoluteProbabilityChange=mean(differences))
    return dict(schema="menia-cross-model-report-v1", origin=header["origin"], planHash=header["planHash"],
                seed=plan["seed"], models=plan["models"], settings=plan["settings"],
                plannedCalls=len(plan["calls"]), recordedResults=len(results), pendingRequest=pending is not None,
                statuses=dict(Counter(r["status"] for r in results)),
                finished=len(results) == len(plan["calls"]),
                complete=len(results) == len(plan["calls"]) and all(r["status"] == "ok" for r in results),
                views=views, baselines=baselines, nameEffect=name_effect, executionAttested=False,
                interpretation="Descriptive behavioral pilot. Name views share target outcomes and are not independent replications. No consciousness inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.journal)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
