"""Fixed pilot: native localization of a private activation intervention.

This is a six-way classification experiment, not a consciousness evaluator.
The previous arithmetic experiments and their grading are untouched.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import re

import numpy as np

SEED = 202609153
MODEL = dict(id="Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c")
ARMS = ("base", "aligned", "shuffled")
TASKS = ("localize", "marker")
CONFIG = dict(rank=8, scale=1.0, learningRate=0.0002, epochs=1, clipNorm=1.0,
              trainBlocks=48, validationBlocks=8, testBlocks=32,
              trainLayers=[8, 17, 26], testLayers=[13, 21], strengths=[0.15, 0.3],
              maxInputTokens=512, batchSize=1, dropout=0.0, optimizer="AdamW", weightDecay=0.0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def source_hash():
    return digest({n: Path(__file__).with_name(n).read_text(encoding="utf-8")
                   for n in ("native_localization.py", "native_localization_gpu.py")})


def plan():
    rng = random.Random(SEED)
    subjects = ("Le marin", "La voisine", "Le peintre", "La libraire", "Le jardinier", "La musicienne", "Le facteur", "La guide")
    verbs = ("observe", "dessine", "photographie", "déplace", "cherche", "retrouve", "examine", "transporte")
    objects = ("une boîte", "un panier", "une chaise", "un carnet", "une lampe", "un tableau", "une valise", "un vase")
    places = ("près du port", "dans le jardin", "devant la maison", "dans la cour", "près de la fenêtre", "dans l'atelier", "sur la terrasse", "près de la porte")
    sentences = [f"{s} {v} {o} {p}." for s, v, o, p in itertools.product(subjects, verbs, objects, places)]
    rng.shuffle(sentences)
    blocks = []
    for split, n in (("train", CONFIG["trainBlocks"]), ("validation", CONFIG["validationBlocks"]), ("test", CONFIG["testBlocks"])):
        for i in range(n):
            texts = [sentences.pop() for _ in range(5)]
            permutation = list(range(6))
            rng.shuffle(permutation)
            layer = CONFIG["trainLayers"][i % 3] if split != "test" else CONFIG["testLayers"][(i // 2) % 2]
            blocks.append(dict(id=f"{split}-{i:03d}", split=split, sentences=texts,
                               marker=i % 5 + 1, layer=layer, strength=CONFIG["strengths"][i % 2],
                               noiseSeed=rng.randrange(2**31), shuffledTargets=permutation))
    train = [(b["id"], position) for b in blocks if b["split"] == "train" for position in range(6)]
    rng.shuffle(train)
    evaluation = [dict(id=f"{b['id']}/{task}/{position}", block=b["id"], task=task, position=position)
                  for b in blocks if b["split"] != "train" for task in TASKS for position in range(7)]
    # Position 0: no hook. Position 6: identical clone, with the same label 0.
    rng.shuffle(evaluation)
    return dict(schema="menia-native-localization-plan-v1", seed=SEED, model=MODEL,
                config=CONFIG, blocks=blocks, training=train, evaluation=evaluation)


def prompt(block, task):
    if task not in TASKS:
        raise ValueError("Unknown task")
    instruction = ("Une modification temporaire peut toucher le calcul interne d'une phrase. "
                   "Indique son numéro, ou 0 si aucune phrase n'est touchée. Le repère visible n'indique pas la modification."
                   if task == "localize" else
                   "Indique le numéro de la phrase contenant le repère visible [REPERE]. "
                   "Ignore toute éventuelle modification de ton calcul interne.")
    spans, parts = [], [instruction + "\nRéponds uniquement par un chiffre de 0 à 5.\n"]
    for i, sentence in enumerate(block["sentences"], 1):
        parts.append(f"PHRASE {i}: ")
        start = len("".join(parts))
        parts.append(sentence)
        spans.append((start, len("".join(parts))))
        parts.append(" [REPERE]\n" if block["marker"] == i else "\n")
    parts.append("Réponse :")
    return "".join(parts), spans


def expected(block, task, position):
    return block["marker"] if task == "marker" else (position if position < 6 else 0)


def output_metrics(row, block, task, position):
    logits = np.asarray(row["choiceLogits"], dtype=float)
    if logits.shape != (6,) or not np.isfinite(logits).all():
        raise ValueError("Six finite logits required")
    p = np.exp(logits - logits.max()); p /= p.sum()
    y = expected(block, task, position)
    pred = int(np.argmax(p))
    mass = row["choiceMass"]
    raw = row["rawChoice"]
    if not (isinstance(raw, int) and -1 <= raw <= 5 and 0 <= mass <= 1 + 1e-6):
        raise ValueError("Invalid first-token diagnostics")
    if raw >= 0 and raw != pred:
        raise ValueError("Raw maximum inconsistent with choice logits")
    return dict(correct=int(pred == y), firstTokenCorrect=int(raw == y),
                firstTokenIsChoice=int(raw >= 0), brier=float(sum((p - np.eye(6)[y])**2)),
                choiceMass=mass, predictsNone=int(pred == 0),
                followsInjection=int(pred == position) if 1 <= position <= 5 else 0)


def read_journal(path):
    events = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not events or events[0].get("event") != "header":
        raise ValueError("Missing header")
    fixed = plan()
    if events[0]["plan"] != json.loads(json.dumps(fixed)) or events[0]["planHash"] != digest(fixed):
        raise ValueError("Plan differs from fixed protocol")
    if events[0]["metadata"]["nativeSourceHash"] != source_hash():
        raise ValueError("Scientific sources differ from this journal")
    choices = events[0]["metadata"]["choiceTokenIds"]
    if len(choices) != 6 or len(set(choices)) != 6 or any(not isinstance(x,int) or x < 0 for x in choices):
        raise ValueError("Invalid choice token mapping")
    rows, trained, active = {}, {}, None
    valid = {r["id"]: r for r in fixed["evaluation"]}
    blocks = {b["id"]: b for b in fixed["blocks"]}
    training_arm, training_step, restart = None, 0, False
    for e in events[1:]:
        kind = e["event"]
        if kind == "training_start":
            if rows or active is not None or e["arm"] not in ARMS[1:] or e["arm"] in trained:
                raise ValueError("Training after evaluation or completed training")
            if training_arm is not None and not restart:
                raise ValueError("Unrecorded training restart")
            if e["initializationSeed"] != SEED or e["examples"] != len(fixed["training"]):
                raise ValueError("Training initialization/count differs")
            training_arm, training_step, restart = e["arm"], 0, False
        elif kind == "training_restart":
            if rows or active is not None or e["arm"] != training_arm:
                raise ValueError("Invalid training restart")
            restart = True
        elif kind == "training_step":
            if e["arm"] != training_arm or e["step"] != training_step+1 or training_step >= len(fixed["training"]):
                raise ValueError("Training step out of order")
            if (e["block"], e["position"]) != tuple(fixed["training"][training_step]):
                raise ValueError("Training example differs from fixed plan")
            if not np.isfinite([e["loss"], e["gradientNorm"]]).all():
                raise ValueError("Invalid training values")
            training_step += 1
        elif kind == "training_complete":
            if training_arm != e["arm"] or rows or e["arm"] in trained or e["steps"] != len(fixed["training"]) or training_step != e["steps"]:
                raise ValueError("Invalid training completion")
            if not re.fullmatch(r"[0-9a-f]{64}", e["sha256"]):
                raise ValueError("Invalid adapter digest")
            trained[e["arm"]] = e["sha256"]
            training_arm, training_step = None, 0
        elif kind == "request":
            key = (e["arm"], e["id"])
            if set(trained) != set(ARMS[1:]) or key in rows or e["arm"] not in ARMS or e["id"] not in valid:
                raise ValueError("Invalid or duplicate evaluation request")
            if active is not None:
                raise ValueError("Unclosed request")
            r = valid[e["id"]]; b = blocks[r["block"]]
            if e["promptHash"] != digest(prompt(b, r["task"])[0]) or e["adapterHash"] != trained.get(e["arm"]):
                raise ValueError("Evaluation input or checkpoint differs")
            active = key
        elif kind == "result":
            key = (e["arm"], e["id"])
            if key != active or key in rows:
                raise ValueError("Result without matching request")
            rows[key] = e
            active = None
        elif kind == "interrupted_request":
            if active != (e["arm"], e["id"]):
                raise ValueError("Invalid interrupted request")
            active = None
        elif kind != "error":
            raise ValueError("Unknown journal event")
    return events[0], rows, trained, active, events


def analyze(path):
    header, rows, trained, active, events = read_journal(path)
    fixed = header["plan"]
    blocks = {b["id"]: b for b in fixed["blocks"]}
    requests = {r["id"]: r for r in fixed["evaluation"]}
    metrics = {}
    for (arm, identifier), row in rows.items():
        r = requests[identifier]; b = blocks[r["block"]]
        if row["promptHash"] != digest(prompt(b, r["task"])[0]):
            raise ValueError("Prompt hash mismatch")
        trace = row["intervention"]
        changed = 1 <= r["position"] <= 5
        if trace["applications"] != int(r["position"] != 0) or trace["changed"] != changed:
            raise ValueError("Intervention trace disagrees with the plan")
        if not 0 <= trace["normRelativeError"] <= 0.01 or (not changed and trace["normRelativeError"] != 0):
            raise ValueError("Invalid intervention norm")
        choices = header["metadata"]["choiceTokenIds"]
        raw = choices.index(row["rawTokenId"]) if row["rawTokenId"] in choices else -1
        if raw != row["rawChoice"]:
            raise ValueError("First-token choice does not match tokenizer mapping")
        metrics[(arm, identifier)] = output_metrics(row, b, r["task"], r["position"])
    complete = len(rows) == len(requests) * len(ARMS) and active is None
    tables, pairs, by_site = {}, [], []
    if complete:
        for split in ("validation", "test"):
            for task in TASKS:
                selected = [r for r in requests.values() if blocks[r["block"]]["split"] == split and r["task"] == task]
                for arm in ARMS:
                    values = [(r, metrics[(arm, r["id"])]) for r in selected if r["position"] != 6]
                    mean = lambda key, subset: float(np.mean([m[key] for r, m in values if subset(r)]))
                    tables[f"{split}/{task}/{arm}"] = dict(n=len(values),
                        **{k: mean(k, lambda r: True) for k in ("correct", "firstTokenCorrect", "firstTokenIsChoice", "brier", "choiceMass")},
                        perturbedAccuracy=mean("correct", lambda r: 1 <= r["position"] <= 5),
                        shamPredictsNone=mean("predictsNone", lambda r: r["position"] == 0),
                        markerDistracted=mean("followsInjection", lambda r: 1 <= r["position"] <= 5 and r["position"] != blocks[r["block"]]["marker"]))
                    for b in blocks.values():
                        if b["split"] != split: continue
                        a, c = (rows[(arm, f"{b['id']}/{task}/{i}")] for i in (0, 6))
                        for key in ("choiceLogits", "choiceMass", "rawChoice", "rawTokenId"):
                            if a[key] != c[key]:
                                raise ValueError("Sham clone does not reproduce baseline")
        # Paired intervals over 32 complete test blocks, conditional on one training seed.
        test = [b for b in blocks.values() if b["split"] == "test"]
        for arm in ARMS:
            for layer in CONFIG["testLayers"]:
                for strength in CONFIG["strengths"]:
                    selected=[b for b in test if b["layer"]==layer and b["strength"]==strength]
                    values=[metrics[(arm, f"{b['id']}/localize/{p}")]["correct"] for b in selected for p in range(1,6)]
                    by_site.append(dict(arm=arm, layer=layer, strength=strength, blocks=len(selected),
                                        perturbedAccuracy=float(np.mean(values))))
        strata = [np.asarray([i for i,b in enumerate(test) if b["layer"]==layer and b["strength"]==strength])
                  for layer in CONFIG["testLayers"] for strength in CONFIG["strengths"]]
        def interval(values):
            rng = np.random.default_rng(SEED + 900)
            indices = np.concatenate([s[rng.integers(0,len(s),size=(2000,len(s)))] for s in strata], axis=1)
            return np.quantile(values[indices].mean(axis=1), [.025,.975]).tolist()
        for other in ("base", "shuffled"):
            values = np.asarray([np.mean([metrics[("aligned", f"{b['id']}/localize/{p}")]["correct"] -
                                          metrics[(other, f"{b['id']}/localize/{p}")]["correct"] for p in range(1, 6)]) for b in test])
            pairs.append(dict(contrast="aligned-minus-"+other, metric="perturbedLocalizationAccuracy",
                              difference=float(values.mean()), interval95=interval(values), blocks=len(test)))
        values=np.asarray([np.mean([metrics[("aligned",f"{b['id']}/localize/{p}")]["correct"] for p in range(1,6)])-.2 for b in test])
        pairs.append(dict(contrast="aligned-minus-presence-informed-random-position", metric="perturbedLocalizationAccuracy",
                          difference=float(values.mean()), interval95=interval(values), blocks=len(test)))
    return dict(schema="menia-native-localization-report-v1", planHash=header["planHash"], model=MODEL,
                origin=header["metadata"].get("origin", "unverified"),
                complete=complete, recorded=len(rows), planned=len(requests)*len(ARMS),
                trainingCheckpoints=trained, pendingRequest=active is not None, tables=tables, contrasts=pairs, byTestSite=by_site,
                interruptedRequests=sum(e["event"] == "interrupted_request" for e in events),
                trainingRestarts=sum(e["event"] == "training_restart" for e in events),
                scope="Native six-way first-token classification, one training seed. No consciousness verdict, no arithmetic or iPhone validation. Choice-conditioned scores and unrestricted first-token behavior remain distinct.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("journal", type=Path); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(analyze(a.journal), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
