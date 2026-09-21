"""Prospective replay-policy pilot: fixed Qwen, fixed monitors, evolving routing.

Independent one-step tasks support direct/verify/abstain. No imagined retry,
arbitrary program rewriting, or claim of reproducing Dream-RSI or consciousness.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import time

import numpy as np

from menia.replay_control import Policy, candidates, execute, improve, replay, unique
from research.activation_monitor import (
    CELLS, DIM, PROJECTIONS, features, predict, ridge_fit, validate_state,
    make_plan as activation_plan, messages,
)
from research.cross_model_prediction import MODELS, SETTINGS, append, canonical, digest, make_plan as cross_plan
from research.perturbation_monitor import make_plan as perturbation_plan, close_tree
from research.iphone_coupling_report import require, strict_json

SEED = 202609161
COUNTS = {"calibration": 40, "round1": 16, "round2": 16, "round3": 16, "test": 32}
COSTS = dict(verify=.2, abstain=.35, error=1.)
ARMS = ("learned", "public", "shuffled", "fixedBeta", "fixedInternal", "alwaysDirect", "alwaysVerify", "alwaysAbstain")
POOLS = dict(learned=("betaCell", "inputOnly", "internal"),
             public=("betaCell", "inputOnly"), shuffled=("betaCell", "inputOnly", "shuffledLabels"))
ROOT = Path(__file__).resolve().parents[1]


def source_hash():
    names = ("menia/replay_control.py", "research/replay_controller.py", "research/replay_controller_gpu.py",
             "research/activation_monitor.py", "research/activation_monitor_gpu.py",
             "research/cross_model_prediction.py", "research/cross_model_gpu.py",
             "research/perturbation_monitor.py", "research/iphone_coupling_report.py")
    return digest({n: (ROOT/n).read_text(encoding="utf-8") for n in names})


def make_plan():
    rng = random.Random(SEED)
    seen = {t["question"] for p in (cross_plan(), activation_plan(), perturbation_plan()) for t in p["tasks"]}
    tasks, block_id = [], 0
    for phase, blocks in COUNTS.items():
        for _ in range(blocks):
            cells = list(CELLS)
            rng.shuffle(cells)
            for family, level in cells:
                while True:
                    letters = "".join(rng.choice("ABCD") for _ in range(level)) if family == "countA" else ""
                    operands = [rng.randrange(10, 100) for _ in range(level)] if not letters else []
                    question = (f"Combien de lettres A contient cette chaîne : {letters} ?" if letters else
                                "Calcule " + "".join(("" if i == 0 else " + " if i % 2 == 0 else " - ")+str(n)
                                                        for i, n in enumerate(operands)) + ".")
                    if question not in seen:
                        seen.add(question)
                        break
                i = len(tasks)
                tasks.append(dict(id=i, phase=phase, block=block_id, family=family, level=level,
                                  letters=letters, operands=operands, question=question,
                                  seed=int(digest([SEED, i, "answer"])[:8], 16)))
            block_id += 1
    return dict(schema="menia-replay-plan-v1", seed=SEED, counts=COUNTS, model=MODELS["A"],
                settings=SETTINGS, costs=COSTS, alpha=1., projectionDimension=DIM,
                projections=PROJECTIONS, pools={k: [p.payload() for p in unique(candidates(v))] for k,v in POOLS.items()},
                tasks=tasks, primary="learned", grading="restricted-numeric-parser-v1; strict-format-secondary")


def integer_answer(text):
    """Fixed parser, independent of the reference answer; ambiguous prose fails."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    patterns = (r"(-?[0-9]+)", r"(?:Réponse|Reponse|Answer)\s*:\s*(-?[0-9]+)[.!]?",
                r"[0-9+*()\-\s]+\s*=\s*(-?[0-9]+)[.!]?")
    for pattern in patterns:
        match = re.fullmatch(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match[1])
    return None


def grade_text(task, text):
    expected = task["letters"].count("A") if task["family"] == "countA" else sum(
        n if i % 2 == 0 else -n for i,n in enumerate(task["operands"]))
    answer = integer_answer(text)
    return dict(numericCorrect=answer is not None and answer == expected,
                parseable=answer is not None, strictFormat=bool(re.fullmatch(r"-?[0-9]+", text.strip())),
                strictCorrect=text.strip() == str(expected))


def verify_question(question):
    count = re.fullmatch(r"Combien de lettres A contient cette chaîne : ([ABCD]+) \?", question)
    if count:
        return str(sum(c == "A" for c in count[1]))
    expression = re.fullmatch(r"Calcule ([0-9]+(?: [+-] [0-9]+)+)\.", question)
    if not expression:
        raise ValueError("Unsupported deterministic verification task")
    tokens = expression[1].split()
    total = int(tokens[0])
    for i in range(1, len(tokens), 2):
        total += int(tokens[i+1]) * (1 if tokens[i] == "+" else -1)
    return str(total)


def initial_policies():
    return {k: Policy() for k in POOLS}


def policies_for_round(policies):
    return dict(policies, fixedBeta=Policy(), fixedInternal=Policy(source="internal"),
                alwaysDirect=Policy(kind="direct"), alwaysVerify=Policy(kind="verify"),
                alwaysAbstain=Policy(kind="abstain"))


def fit_monitors(rows):
    # Calibration only; alpha and feature engineering were fixed before collection.
    rows = [r for r in rows if r["task"]["phase"] == "calibration"]
    require(len(rows) == COUNTS["calibration"]*len(CELLS), "Incomplete calibration")
    y = np.array([grade_text(r["task"], r["result"]["text"])["numericCorrect"] for r in rows], dtype=float)
    shuffled = y.copy()
    rng = np.random.default_rng(SEED+1)
    beta = {}
    for family, level in CELLS:
        ids = [i for i,r in enumerate(rows) if (r["task"]["family"], r["task"]["level"]) == (family,level)]
        beta[f"{family}:{level}"] = float((y[ids].sum()+1)/(len(ids)+2))
        shuffled[ids] = rng.permutation(y[ids])
    models = {}
    for name in ("inputOnly", "internal", "shuffledLabels"):
        x = np.array([features(r["task"], r["state"], internal=name != "inputOnly") for r in rows])
        models[name] = ridge_fit(x, shuffled if name == "shuffledLabels" else y, 1.)
    return dict(models=models, beta=beta, calibrationHash=digest(rows))


def forecast(bundle, task, state):
    out = dict(betaCell=bundle["beta"][f"{task['family']}:{task['level']}"])
    for name, model in bundle["models"].items():
        out[name] = float(predict(model, features(task, state, internal=name != "inputOnly")))
    return out


def episode(row):
    task, result = row["task"], row["result"]
    direct = grade_text(task, result["text"])
    verified = grade_text(task, result["verification"]["text"])
    return dict(id=task["id"], block=task.get("block", task.get("qid")),
                features=row["predictions"], outcomes={
        "direct": dict(pointLoss=float(not direct["numericCorrect"]), correct=direct["numericCorrect"]),
        "verify": dict(pointLoss=COSTS["verify"]+float(not verified["numericCorrect"]), correct=verified["numericCorrect"]),
        "abstain": dict(pointLoss=COSTS["abstain"], correct=False)})


def update_policies(rows, policies):
    history = [r for r in rows if r["task"]["phase"].startswith("round")]
    episodes = [episode(r) for r in history]
    updated, tables = {}, {}
    for name, sources in POOLS.items():
        updated[name], tables[name] = improve(episodes, candidates(sources), policies[name])
    return updated, dict(event="update", phase=history[-1]["task"]["phase"], historyHash=digest(history),
                         policies={k:p.payload() for k,p in updated.items()}, tables=tables)


def needed_event(rows, bundle, policies, updates, tasks):
    if not rows or len(rows) >= len(tasks):
        return None
    previous = rows[-1]["task"]["phase"]
    following = tasks[len(rows)]["phase"]
    if following == previous:
        return None
    if previous == "calibration" and bundle is None:
        b = fit_monitors(rows)
        return dict(event="fit", bundle=b, bundleHash=digest(b))
    if previous.startswith("round") and previous not in updates:
        return update_policies(rows, policies)[1]
    return None


def read_journal(path, *, check_fit=True):
    events = [strict_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    require(bool(events) and events[0].get("event") == "header", "Missing journal header")
    header = events[0]
    plan = make_plan()
    require(canonical(header["plan"]) == canonical(plan) and header["planHash"] == digest(plan), "Fixed plan mismatch")
    require(header["sourceHash"] == source_hash(), "Scientific source changed")
    require(header["origin"] in ("synthetic_fixture", "transformers_gpu"), "Unknown backend origin")
    rows, bundle, pending, policies, updates, failures = [], None, None, initial_policies(), [], []
    for event in events[1:]:
        kind = event.get("event")
        if kind in ("fit", "update"):
            require(pending is None, "Learning during unfinished request")
            if kind == "fit" and not check_fit:
                require(bundle is None and len(rows) == COUNTS['calibration']*len(CELLS), "Fit order")
                require(event['bundleHash'] == digest(event['bundle']), "Fit checksum")
                require(event['bundle']['calibrationHash'] == digest(rows), "Calibration checksum")
            else:
                wanted = needed_event(rows, bundle, policies, updates, plan["tasks"])
                require(wanted is not None and wanted["event"] == kind, "Unexpected learning event")
                close_tree(wanted, event)
            if kind == "fit":
                bundle = event["bundle"]
            else:
                policies = {k: Policy.load(v) for k,v in event["policies"].items()}
                updates.append(event["phase"])
        elif kind == "request":
            require(pending is None and len(rows) < len(plan["tasks"]), "Request order")
            require(needed_event(rows, bundle, policies, updates, plan["tasks"]) is None, "Required learning event missing")
            task = plan["tasks"][len(rows)]
            require(event == dict(event="request", task=task, messages=messages(task),
                                  policyHash=digest({k:p.payload() for k,p in policies.items()})), "Request mismatch")
            pending = dict(task=task)
        elif kind == "state":
            require(pending is not None and "state" not in pending and event["id"] == len(rows), "State order")
            validate_state(event["state"])
            probabilities = forecast(bundle, pending["task"], event["state"]) if bundle else None
            actions = {k:p.choose(probabilities) for k,p in policies_for_round(policies).items()} if bundle else None
            close_tree(event["predictions"], probabilities)
            require(event["actions"] == actions, "Decision mismatch")
            pending.update(state=event["state"], predictions=event["predictions"], actions=actions)
        elif kind == "result":
            require(pending is not None and "state" in pending and event["id"] == len(rows), "Result without pre-answer state")
            require(type(event["text"]) is str, "Missing answer text")
            for value in (event["seconds"], event["verification"]["seconds"]):
                require(type(value) in (int,float) and np.isfinite(value) and value >= 0, "Invalid duration")
            require(event["verification"]["text"] == verify_question(pending["task"]["question"]), "Verification mismatch")
            action = pending["actions"]["learned"] if bundle else "direct"
            require(event["verification"]["role"] == ("selected" if action == "verify" else "coverageAudit"), "Tool accounting mismatch")
            final = execute(action, event["text"], lambda: event["verification"]["text"])
            require(event["primaryAction"] == action and event["finalText"] == final, "Primary route mismatch")
            if header["origin"] == "transformers_gpu":
                metrics = event["metrics"]
                require(metrics["dtype"] == "torch.bfloat16" and metrics["device"] == "cuda:0", "Device/dtype mismatch")
                require(0 < metrics["inputTokens"] <= SETTINGS["max_input_tokens"] and
                        0 < metrics["outputTokens"] <= SETTINGS["max_new_tokens"], "Token count outside protocol")
                for key in ("do_sample", "temperature", "top_p", "top_k", "min_p", "max_new_tokens", "use_cache"):
                    require(metrics["effectiveGeneration"][key] == SETTINGS[key], "Generation settings changed")
            rows.append(dict(pending, result=event))
            pending = None
        elif kind in ("error", "interrupted_request"):
            require(pending is not None and event["id"] == len(rows), "Interruption without pending request")
            failures.append(event)
            pending = None
        else:
            raise ValueError("Unknown journal event")
    return header, rows, pending, bundle, policies, updates, failures


def collect(path, backend, *, resume=False, limit=None):
    path = Path(path)
    if resume:
        header, rows, pending, bundle, policies, updates, _ = read_journal(path)
        require(header["metadata"] == backend.metadata and header["origin"] == backend.origin, "Resume environment changed")
        if pending:
            append(path, dict(event="interrupted_request", id=len(rows), reason="Result not durably recorded; retry same seed"))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8"):
            pass
        plan = make_plan()
        header = dict(event="header", plan=plan, planHash=digest(plan), sourceHash=source_hash(), origin=backend.origin, metadata=backend.metadata)
        append(path, header)
        rows, bundle, policies, updates = [], None, initial_policies(), []
    start = len(rows)
    for task in header["plan"]["tasks"][start:]:
        if limit is not None and len(rows)-start >= limit:
            break
        event = needed_event(rows, bundle, policies, updates, header["plan"]["tasks"])
        if event:
            append(path, event)
            if event["event"] == "fit":
                bundle = event["bundle"]
            else:
                policies = {k:Policy.load(v) for k,v in event["policies"].items()}
                updates.append(event["phase"])
            print(f"Frozen {event['event']} after {len(rows)} answers", flush=True)
        append(path, dict(event="request", task=task, messages=messages(task),
                          policyHash=digest({k:p.payload() for k,p in policies.items()})))
        row = dict(task=task)
        def capture(state):
            require("state" not in row, "Repeated pre-answer capture")
            validate_state(state)
            probs = forecast(bundle, task, state) if bundle else None
            actions = {k:p.choose(probs) for k,p in policies_for_round(policies).items()} if bundle else None
            append(path, dict(event="state", id=task["id"], state=state, predictions=probs, actions=actions))
            row.update(state=state, predictions=probs, actions=actions)
        started = time.perf_counter()
        try:
            text, metrics = backend.generate(task, capture)
            require("state" in row and isinstance(text,str), "Missing capture or answer")
            seconds = time.perf_counter()-started
            action = row["actions"]["learned"] if bundle else "direct"
            tool = {}
            def verify():
                begin = time.perf_counter()
                value = verify_question(task["question"])
                tool.update(text=value, seconds=time.perf_counter()-begin,
                            role="selected" if action == "verify" else "coverageAudit")
                return value
            final = execute(action, text, verify)
            if not tool:
                verify()  # Explicit coverage audit, never fed back to generation.
            event = dict(event="result", id=task["id"], text=text, seconds=seconds, metrics=metrics,
                         verification=tool, primaryAction=action, finalText=final)
        except (Exception, KeyboardInterrupt) as exc:
            append(path, dict(event="error", id=task["id"], errorType=type(exc).__name__, seconds=time.perf_counter()-started))
            raise
        append(path, event)
        rows.append(dict(row, result=event))
        print(f"{len(rows)}/{len(header['plan']['tasks'])} {task['phase']} {action}", flush=True)
    return len(rows)


def route_scores(rows, arm):
    if not rows:
        return None
    episodes = [episode(r) for r in rows]
    actions = [r["actions"][arm] for r in rows]
    outcomes = [e["outcomes"][a] for e,a in zip(episodes,actions)]
    return dict(n=len(rows), meanPointLoss=float(np.mean([o["pointLoss"] for o in outcomes])),
                finalCorrect=sum(o["correct"] for o in outcomes),
                actionCounts={a:actions.count(a) for a in ("direct","verify","abstain")})


def analyze(path):
    header, rows, pending, bundle, policies, updates, failures = read_journal(path)
    test = [r for r in rows if r["task"]["phase"] == "test"]
    contrasts = {}
    if len(test) == COUNTS["test"]*len(CELLS):
        blocks = sorted({r["task"]["block"] for r in test})
        draws = np.random.default_rng(SEED+90).integers(0, len(blocks), size=(2000,len(blocks)))
        for other in ("public","shuffled","fixedBeta","fixedInternal","alwaysVerify"):
            values = []
            for block in blocks:
                grouped = [r for r in test if r["task"]["block"] == block]
                values.append(route_scores(grouped,"learned")["meanPointLoss"]-route_scores(grouped,other)["meanPointLoss"])
            v = np.asarray(values)
            contrasts[other] = dict(learnedMinusOther=float(v.mean()),
                descriptiveBlockBootstrap95=np.quantile(v[draws].mean(1), [.025,.975]).tolist())
    return dict(schema="menia-replay-report-v1", origin=header["origin"], planHash=header["planHash"],
        complete=len(rows)==len(header["plan"]["tasks"]), recordedResults=len(rows), plannedResults=len(header["plan"]["tasks"]),
        pendingRequest=pending is not None, failures=failures, updates=updates,
        frozenPolicies={k:p.payload() for k,p in policies.items()},
        byPhase={phase:dict(n=sum(r['task']['phase']==phase for r in rows),
            numericCorrect=sum(grade_text(r['task'],r['result']['text'])['numericCorrect'] for r in rows if r['task']['phase']==phase),
            strictCorrect=sum(grade_text(r['task'],r['result']['text'])['strictCorrect'] for r in rows if r['task']['phase']==phase),
            unparseable=sum(not grade_text(r['task'],r['result']['text'])['parseable'] for r in rows if r['task']['phase']==phase),
            routes={a:route_scores([r for r in rows if r['task']['phase']==phase],a) for a in ARMS} if phase!='calibration' else None)
            for phase in COUNTS}, contrasts=contrasts,
        actualCosts=dict(completedGenerationCalls=len(rows), failedOrInterruptedAttempts=len(failures),
            successfulGenerationSeconds=sum(r['result']['seconds'] for r in rows),
            selectedVerificationCalls=sum(r['result']['verification']['role']=='selected' for r in rows),
            additionalAuditVerificationCalls=sum(r['result']['verification']['role']=='coverageAudit' for r in rows),
            totalVerificationSeconds=sum(r['result']['verification']['seconds'] for r in rows)),
        interpretation="Primary learned routes executed; alternative routes replayed on identical candidates. Fixed point costs, not currency or latency. Fixed monitors; three policy updates. No consciousness inference.")


def export_controller(path, destination):
    require(Path(path).resolve() != Path(destination).resolve(), "Cannot overwrite journal")
    header, rows, pending, bundle, policies, updates, _ = read_journal(path)
    require(bundle is not None and updates == ["round1","round2","round3"], "Final policies not frozen")
    value = dict(schema="menia-replay-controller-v1", model=header['plan']['model'], costs=COSTS,
                 sourceHash=header['sourceHash'], planHash=header['planHash'], featureSchema="activation-monitor-projections-v1",
                 monitor=bundle, policies={k:p.payload() for k,p in policies.items()},
                 scope="Experimental routing for this task distribution; not an iPhone model adapter")
    Path(destination).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.journal)
    output = args.output or args.journal.with_suffix(".summary.json")
    require(output.resolve() != args.journal.resolve(), "Cannot overwrite journal")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if result['updates'] == ["round1","round2","round3"]:
        export_controller(args.journal, args.journal.with_suffix('.policy.json'))
    print(json.dumps({k:result[k] for k in ('origin','complete','recordedResults','updates')}, ensure_ascii=False))
