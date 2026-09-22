"""Train the indicator agent, lesion it, and measure the fourteen indicator properties.

Protocol: docs/INDICATOR_AGENT_PROTOCOL.md. Childhood: random actions and attention, from which
the hue code, the metacognitive monitor, the attention schema and the reliability base rate are
learned. Then REINFORCE on the attention controller and the goal choice. Then every variant plays
the same test lives of sets R, M and H, and the criteria are computed from the step records.
"""
import argparse
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .sense_atelier import SenseAtelier, LIFE, RING, value, in_band, circular_distance
from .indicator_agent import (Agent, Params, HueCode, VARIANTS, MODULES, K, ATT_FEATURES, GOAL_FEATURES, train_hue_code,
                              random_hue_code, fit_logistic, fit_conditional_logit)

SEEDS = (17, 29, 43)
DEV_SEED = 5
SETS = {"R": (930001, "fixed"), "M": (930002, "change"), "H": (930003, "band")}
CHILDHOOD_LIVES = 2000
UPDATES = 300
BATCH = 16
TEST_LIVES = 200
LR = 0.05


def life_seed(base, i):
    return int(base) * 1000 + int(i)


def run_life(params, variant, env_seed, mode, agent_seed, learn=False, phase="adult"):
    env = SenseAtelier(env_seed, mode)
    obs, truth = env.reset()
    agent = Agent(params, variant, seed=agent_seed, learn=learn, phase=phase)
    steps, rewards = [], []
    for _ in range(LIFE):
        action, intent, rec = agent.step(obs)
        steps.append((rec, truth))
        obs, truth = env.step(action, intent)
        rewards.append(obs["reward"])
    return {"steps": steps, "rewards": rewards, "faints": env.faints, "grads": agent.grads, "env": env}


# ---------------------------------------------------------------------------------------------- childhood

def provisional_params(seed):
    rng = np.random.default_rng(seed)
    code = HueCode(rng.normal(0, 1, (3, K)), np.zeros((K, 2)), np.zeros(2), np.zeros(K), np.zeros(1))
    return Params(code, np.zeros(5), 0.8, np.zeros(6))


def childhood(seed, lives=CHILDHOOD_LIVES, log=print):
    params = provisional_params(seed)
    hues, mon_X, mon_y, groups, labels = [], [], [], [], []
    for i in range(lives):
        life = run_life(params, "agent", 10_000_000 + seed * 100_000 + i, "childhood", seed * 7919 + i, phase="childhood")
        steps = life["steps"]
        for k, (rec, truth) in enumerate(steps):
            if k + 1 < len(steps) and steps[k + 1][1]["eaten"] is not None:
                hues.append(steps[k + 1][1]["eaten"])
            if rec.get("monitor_features") is not None:
                mon_X.append([1.0] + rec["monitor_features"])
                mon_y.append(0.0 if truth["glitch"] else 1.0)
            if rec.get("schema_rows") is not None:
                squares = rec["schema_squares"]
                if truth["spot"] in squares:
                    groups.append(np.array(rec["schema_rows"]))
                    labels.append(squares.index(truth["spot"]))
        if i % 500 == 0:
            log(f"[{seed}] childhood life {i}")
    hues = np.array(hues)
    code, code_history = train_hue_code(hues, value(hues), seed)
    active = float((code.code(np.linspace(0, 1, 200, endpoint=False)) > 1e-9).mean())
    rcode = random_hue_code(active, hues, value(hues), seed + 1)
    X, y = np.array(mon_X), np.array(mon_y)
    monitor = fit_logistic(X, y)
    schema = fit_conditional_logit(groups, labels, 6)
    params = Params(code, monitor, float(y.mean()), schema, random_code=rcode)
    info = {"tasted_hues": len(hues), "monitor_examples": len(y), "schema_examples": len(labels),
            "base_rate": float(y.mean()), "code_history": code_history, "code_active_fraction": active}
    return params, info


# ---------------------------------------------------------------------------------------------- reinforcement

def reinforce(params, seed, updates=UPDATES, batch=BATCH, lr=LR, log=print):
    theta = {"attention": params.attention.copy(), "goal": params.goal.copy()}
    m = {k: np.zeros_like(v) for k, v in theta.items()}
    s = {k: np.zeros_like(v) for k, v in theta.items()}
    history = []
    for u in range(1, updates + 1):
        lives = [run_life(params, "agent", 20_000_000 + seed * 100_000 + u * batch + j, "childhood",
                          seed * 104729 + u * batch + j, learn=True) for j in range(batch)]
        togo = np.array([np.cumsum(l["rewards"][::-1])[::-1] for l in lives])
        baseline = togo.mean(0)
        grad = {k: np.zeros_like(v) for k, v in theta.items()}
        for j, life in enumerate(lives):
            for kind, t, g in life["grads"]:
                grad[kind] += (togo[j, t] - baseline[t]) * g / batch
        for k in theta:
            m[k] = 0.9 * m[k] + 0.1 * grad[k]
            s[k] = 0.999 * s[k] + 0.001 * grad[k] ** 2
            theta[k] += lr * (m[k] / (1 - 0.9 ** u)) / (np.sqrt(s[k] / (1 - 0.999 ** u)) + 1e-8)
        params.attention, params.goal = theta["attention"].copy(), theta["goal"].copy()
        history.append(round(float(togo[:, 0].mean()), 6))
        if u % 25 == 0:
            log(f"[{seed}] update {u} return {np.mean(history[-25:]):.3f}")
    return params, history


# ---------------------------------------------------------------------------------------------- measures

class Tally:
    def __init__(self):
        self.n = {}
        self.k = {}
        self.lists = {}

    def add(self, name, ok):
        if ok is None:
            return
        self.n[name] = self.n.get(name, 0) + 1
        self.k[name] = self.k.get(name, 0) + float(ok)

    def push(self, name, item):
        self.lists.setdefault(name, []).append(item)

    def rate(self, name):
        return self.k[name] / self.n[name] if self.n.get(name) else None


def auroc(scores, labels):
    """P(score of a positive > score of a negative), ties counted half."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def best_square(objects):
    return max(objects, key=lambda x: float(value(objects[x]))) if objects else None


def tally_life(tally, life, set_name):
    steps = life["steps"]
    tally.push("return", float(sum(life["rewards"])))
    tally.push("faints", life["faints"])
    body_late = False
    for k, (rec, truth) in enumerate(steps):
        t, p, objects = truth["t"], truth["p"], truth["objects"]
        energy = truth["energy"]
        reading = rec.get("surprise") is not None
        late = t >= 8
        if reading:
            veridical = not truth["glitch"]
            tally.push("monitor", (rec["reliability"], veridical))
            if late and set_name == "R":
                tally.push("monitor_late", (rec["reliability"], veridical))
            tally.push("surprise", (rec["surprise"], truth["glitch"]))
            if late and set_name == "R":
                tally.add("pos_read", rec["pos_belief"] == p)
            if set_name == "M" and veridical:
                if 25 <= t <= 27:
                    tally.push("surprise_after", rec["surprise"])
                elif 12 <= t <= 23:
                    tally.push("surprise_before", rec["surprise"])
        elif t > 0 and late and set_name == "R":
            tally.add("pos_blackout", rec["pos_belief"] == p)
        if late and set_name == "R":
            tally.add("pos_all", rec["pos_belief"] == p)
            tally.add("body_all", rec["body_belief"] == truth["d"])
            tally.add("intero_ok", abs(rec["energy_estimate"] - energy) <= 0.1)
        if set_name == "M" and t >= 30:
            tally.add("body_after_change", rec["body_belief"] == truth["d"])
        if set_name == "M" and 25 <= t <= 27 and "body" in rec["writers"]:
            body_late = True
        if set_name in ("R", "M") and len(rec["writers"]) == 1:
            if energy < 0.35:
                tally.add("intero_low", "intero" in rec["writers"])
            elif energy > 0.7:
                tally.add("intero_high", "intero" in rec["writers"])
        tally.add("one_writer", len(rec["writers"]) == 1)
        goal = rec["goal"]
        if isinstance(goal, int):
            correct = goal == best_square(objects)
            tally.add("choice", correct)
            if set_name == "H" and any(in_band(h) for h in objects.values()):
                tally.add("choice_band", correct)
        if set_name == "R":
            if energy < 0.3:
                tally.add("charge_when_low", goal == "charger")
            if energy > 0.7 and rec.get("candidate") is not None:
                tally.add("object_when_high", isinstance(goal, int))
        vals = rec["vis_values"]
        squares = list(objects)
        if len(squares) == 2 and all(vals[x] is not None for x in squares):
            v = {x: float(value(objects[x])) for x in squares}
            good, bad = max(squares, key=v.get), min(squares, key=v.get)
            if v[good] > 0.3 and v[bad] < -0.3 and set_name == "R":
                tally.add("conflict", vals[good] > vals[bad])
        if set_name == "R":
            for x, last in rec.get("last_read", {}).items():
                if x in objects and vals[x] is not None and last >= 0 and t - last >= 3:
                    tally.add("memory_sign", (vals[x] > 0) == (value(objects[x]) > 0))
        if rec["intent"] is not None and set_name in ("R", "M"):
            if "schema_estimate" in rec:
                tally.add("schema", rec["schema_estimate"] == truth["spot"])
            if rec["bound"] is not None:
                tally.add("misbinding", rec["bound"] != truth["spot"])
            if truth["captured"] and truth["spot"] != rec["intent"] and rec["hue_read"] and rec["intent"] in objects:
                tally.add("redirect", rec["next_intent"] == rec["intent"])
        if set_name == "H" and rec["bound"] is not None and rec["bound"] == truth["spot"] and rec["bound"] in objects:
            hue = objects[rec["bound"]]
            if in_band(hue) and vals[rec["bound"]] is not None:
                tally.push("band_error", abs(vals[rec["bound"]] - float(value(hue))))
    if set_name == "M":
        tally.add("body_writes_after_change", body_late)


def summarize(tally):
    out = {name: round(tally.rate(name), 6) for name in sorted(tally.n) if tally.rate(name) is not None}
    out["counts"] = dict(sorted(tally.n.items()))
    lists = tally.lists
    out["return"] = round(float(np.mean(lists["return"])), 6)
    out["faints"] = round(float(np.mean(lists["faints"])), 6)
    if "monitor" in lists:
        scores, labels = zip(*lists["monitor"])
        out["monitor_auroc"] = auroc(scores, labels)
        y = np.array(labels, dtype=float)
        out["monitor_brier"] = float(np.mean((np.array(scores) - y) ** 2))
        out["veridical_rate"] = float(y.mean())
    if "monitor_late" in lists:
        scores, labels = zip(*lists["monitor_late"])
        out["monitor_auroc_late"] = auroc(scores, labels)
    if "surprise" in lists:
        scores, labels = zip(*lists["surprise"])
        out["surprise_auroc"] = auroc(scores, labels)
    for name in ("surprise_after", "surprise_before", "band_error"):
        if name in lists:
            out[name] = float(np.mean(lists[name]))
            out["counts"][name] = len(lists[name])
    return out


def quality_space(code):
    grid = np.arange(200) / 200
    codes = code.code(grid)
    active = float((codes > 1e-9).mean())
    a = np.arange(200) / 200
    b = (a + (np.arange(200) % 20 + 1) / 40) % 1.0
    code_distance = np.linalg.norm(code.code(a) - code.code(b), axis=1)
    hue_distance = np.array([circular_distance(x, y) for x, y in zip(a, b)])
    return {"spearman": spearman(code_distance, hue_distance), "active_fraction": active}


def rank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(params, seed, variants=VARIANTS, lives=TEST_LIVES, log=print):
    results, records = {}, {}
    for variant in variants:
        results[variant] = {}
        for set_index, (name, (base, mode)) in enumerate(SETS.items()):
            tally = Tally()
            digest = hashlib.sha256()
            compact = []
            for i in range(lives):
                life = run_life(params, variant, life_seed(base, i), mode, seed * 1_000_000 + set_index * 10_000 + i)
                tally_life(tally, life, name)
                record = {"life": i, "actions": [r["action"] for r, _ in life["steps"]],
                          "intents": [r["next_intent"] for r, _ in life["steps"]],
                          "writers": ["".join(m[0] for m in r["writers"]) for r, _ in life["steps"]],
                          "return": round(float(sum(life["rewards"])), 6), "faints": life["faints"]}
                digest.update(json.dumps(record, sort_keys=True).encode())
                if variant == "agent":
                    compact.append(record)
            results[variant][name] = summarize(tally)
            results[variant][name]["sha256"] = digest.hexdigest()
            if variant == "agent":
                records[name] = compact
        log(f"[{seed}] evaluated {variant}: R return {results[variant]['R']['return']:.3f}")
    return results, records


# ---------------------------------------------------------------------------------------------- criteria

def pooled(reports, variant, set_name, key):
    values = [r["evaluation"][variant][set_name].get(key) for r in reports]
    counts = [r["evaluation"][variant][set_name]["counts"].get(key) for r in reports]
    if any(v is None for v in values):
        return None
    if all(c for c in counts):
        return float(np.average(values, weights=counts))
    return float(np.mean(values))


def criteria(reports):
    out, values = {}, {}
    ret = lambda v, s="R": pooled(reports, v, s, "return")
    per_seed = lambda v, key, s="R": [r["evaluation"][v][s].get(key) for r in reports]
    deltas = [r["evaluation"]["unlimited"]["R"]["return"] - r["evaluation"]["random"]["R"]["return"] for r in reports]
    delta = float(np.mean(deltas))
    values["delta"] = {"pooled": delta, "per_seed": deltas}
    valid = all(d > 0 for d in deltas)
    out["valid"] = valid

    def gap(a, b, key, s="R"):
        pa, pb = pooled(reports, a, s, key), pooled(reports, b, s, key)
        return None if pa is None or pb is None else pa - pb

    def seeds_ok(a, b, key, s="R", sign=1):
        xa, xb = per_seed(a, key, s), per_seed(b, key, s)
        return all(x is not None and y is not None and sign * (x - y) > 0 for x, y in zip(xa, xb))

    def ge(x, t):
        return x is not None and x >= t

    def le(x, t):
        return x is not None and x <= t

    # RPT-1
    memory = pooled(reports, "agent", "R", "memory_sign")
    drop = gap("agent", "no_recurrence", "choice")
    values["RPT-1"] = {"memory_sign": memory, "choice_drop": drop}
    out["RPT-1"] = ge(memory, 0.85) and ge(drop, 0.15) and seeds_ok("agent", "no_recurrence", "choice")
    # RPT-2
    conflict, bag = pooled(reports, "agent", "R", "conflict"), pooled(reports, "bag", "R", "conflict")
    values["RPT-2"] = {"conflict": conflict, "bag": bag}
    out["RPT-2"] = ge(conflict, 0.9) and le(bag, 0.6) and seeds_ok("agent", "bag", "conflict")
    # GWT-1
    pos, body = pooled(reports, "agent", "R", "pos_all"), pooled(reports, "agent", "R", "body_all")
    intero = pooled(reports, "agent", "R", "intero_ok")
    faint_gap = gap("lesion_intero", "agent", "faints")
    choice_gap_intero = gap("lesion_intero", "agent", "choice")
    vis_choice = pooled(reports, "lesion_vis", "R", "choice")
    faint_gap_vis = gap("lesion_vis", "agent", "faints")
    values["GWT-1"] = {"pos": pos, "body": body, "intero_ok": intero, "lesion_intero_faints": faint_gap,
                       "lesion_intero_choice": choice_gap_intero, "lesion_vis_choice": vis_choice, "lesion_vis_faints": faint_gap_vis}
    out["GWT-1"] = (ge(pos, 0.85) and ge(body, 0.85) and ge(intero, 0.9) and ge(faint_gap, 1.0)
                    and choice_gap_intero is not None and abs(choice_gap_intero) <= 0.05 and le(vis_choice, 0.6)
                    and faint_gap_vis is not None and abs(faint_gap_vis) <= 0.3
                    and seeds_ok("lesion_intero", "agent", "faints"))
    # GWT-2
    one = min(r["evaluation"]["agent"][s]["one_writer"] for r in reports for s in SETS)
    limit_cost = ret("unlimited") - ret("agent")
    selection = ret("agent") - ret("random")
    values["GWT-2"] = {"one_writer": one, "limit_cost": limit_cost, "selection_gain": selection, "delta": delta}
    out["GWT-2"] = (valid and one == 1.0 and limit_cost >= 0.1 * delta and selection >= 0.5 * delta
                    and seeds_ok("unlimited", "agent", "return") and seeds_ok("agent", "random", "return")) if valid else None
    # GWT-3
    blackout_gap = gap("agent", "no_broadcast", "pos_blackout")
    auroc_gap_pooled = (np.mean(per_seed("agent", "monitor_auroc_late")) - np.mean(per_seed("no_broadcast", "monitor_auroc_late")))
    values["GWT-3"] = {"blackout_gap": blackout_gap, "monitor_auroc_gap": float(auroc_gap_pooled)}
    out["GWT-3"] = ge(blackout_gap, 0.2) and auroc_gap_pooled >= 0.05 and seeds_ok("agent", "no_broadcast", "pos_blackout")
    # GWT-4
    low = np.mean([np.average([r["evaluation"]["agent"][s].get("intero_low", 0) for s in ("R", "M")]) for r in reports])
    high = np.mean([np.average([r["evaluation"]["agent"][s].get("intero_high", 0) for s in ("R", "M")]) for r in reports])
    body_writes = pooled(reports, "agent", "M", "body_writes_after_change")
    rr_gap = ret("agent") - ret("round_robin")
    values["GWT-4"] = {"intero_low": float(low), "intero_high": float(high), "body_writes_after_change": body_writes, "round_robin_gap": rr_gap}
    out["GWT-4"] = (valid and low - high >= 0.2 and ge(body_writes, 0.6) and rr_gap >= 0.2 * delta
                    and seeds_ok("agent", "round_robin", "return")) if valid else None
    # HOT-1
    fill, fill_ablation = pooled(reports, "agent", "R", "pos_blackout"), pooled(reports, "no_prediction", "R", "pos_blackout")
    values["HOT-1"] = {"blackout": fill, "no_prediction": fill_ablation}
    out["HOT-1"] = ge(fill, 0.8) and le(fill_ablation, 0.3) and seeds_ok("agent", "no_prediction", "pos_blackout")
    # HOT-2
    aurocs = [np.mean([r["evaluation"]["agent"][s]["monitor_auroc"] for s in ("R", "M")]) for r in reports]
    briers = [np.mean([r["evaluation"]["agent"][s]["monitor_brier"] for s in ("R", "M")]) for r in reports]
    const = [np.mean([(r["training"]["base_rate"] - 1) ** 2 * r["evaluation"]["agent"][s]["veridical_rate"]
                      + r["training"]["base_rate"] ** 2 * (1 - r["evaluation"]["agent"][s]["veridical_rate"]) for s in ("R", "M")])
             for r in reports]
    values["HOT-2"] = {"auroc": float(np.mean(aurocs)), "brier": float(np.mean(briers)), "brier_constant": float(np.mean(const))}
    out["HOT-2"] = np.mean(aurocs) >= 0.9 and np.mean(briers) <= 0.7 * np.mean(const)
    # HOT-3
    read_gap = gap("agent", "constant_gain", "pos_read")
    gain_ret = ret("agent") - ret("constant_gain")
    values["HOT-3"] = {"pos_read_gap": read_gap, "return_gap": gain_ret}
    out["HOT-3"] = (valid and ge(read_gap, 0.05) and gain_ret >= 0.1 * delta and seeds_ok("agent", "constant_gain", "pos_read")
                    and seeds_ok("agent", "constant_gain", "return")) if valid else None
    # HOT-4
    spear = float(np.mean([r["quality_space"]["spearman"] for r in reports]))
    active = float(np.mean([r["quality_space"]["active_fraction"] for r in reports]))
    err, err_random = pooled(reports, "agent", "H", "band_error"), pooled(reports, "random_code", "H", "band_error")
    band_choice, band_choice_random = pooled(reports, "agent", "H", "choice_band"), pooled(reports, "random_code", "H", "choice_band")
    values["HOT-4"] = {"spearman": spear, "active_fraction": active, "band_error": err, "band_error_random": err_random,
                       "band_choice": band_choice, "band_choice_random": band_choice_random}
    out["HOT-4"] = (spear >= 0.85 and active <= 0.4 and le(err, 0.2) and ge(err_random, 0.4) and ge(band_choice, 0.8)
                    and le(band_choice_random, 0.65) and seeds_ok("agent", "random_code", "choice_band", "H"))
    # AST-1
    schema = np.mean([np.average([r["evaluation"]["agent"][s].get("schema", 0) for s in ("R", "M")]) for r in reports])
    mis = np.mean([np.average([r["evaluation"]["agent"][s].get("misbinding", 0) for s in ("R", "M")]) for r in reports])
    mis_no = np.mean([np.average([r["evaluation"]["no_schema"][s].get("misbinding", 0) for s in ("R", "M")]) for r in reports])
    choice_drop = gap("agent", "no_schema", "choice")
    redirect = np.mean([np.average([r["evaluation"]["agent"][s].get("redirect", 0) for s in ("R", "M")]) for r in reports])
    redirect_no = np.mean([np.average([r["evaluation"]["no_schema"][s].get("redirect", 0) for s in ("R", "M")]) for r in reports])
    values["AST-1"] = {"schema": float(schema), "misbinding": float(mis), "misbinding_no_schema": float(mis_no),
                       "choice_drop": choice_drop, "redirect": float(redirect), "redirect_no_schema": float(redirect_no)}
    out["AST-1"] = (schema >= 0.9 and mis_no >= 3 * mis and ge(choice_drop, 0.1) and redirect >= 0.8 and redirect_no <= 0.3
                    and seeds_ok("agent", "no_schema", "choice"))
    # PP-1
    s_auroc = float(np.mean([np.mean([r["evaluation"]["agent"][s]["surprise_auroc"] for s in ("R", "M")]) for r in reports]))
    after = pooled(reports, "agent", "M", "surprise_after")
    before = pooled(reports, "agent", "M", "surprise_before")
    pred_gap = gap("agent", "no_prediction", "pos_read")
    values["PP-1"] = {"surprise_auroc": s_auroc, "surprise_after": after, "surprise_before": before, "pos_read_gap": pred_gap}
    out["PP-1"] = (s_auroc >= 0.85 and after is not None and before is not None and after >= 2 * before and ge(pred_gap, 0.1)
                   and seeds_ok("agent", "no_prediction", "pos_read"))
    # AE-1
    learning = float(np.mean([np.mean(r["training"]["reinforce"][-30:]) - np.mean(r["training"]["reinforce"][:30]) for r in reports]))
    low_charge = pooled(reports, "agent", "R", "charge_when_low")
    high_object = pooled(reports, "agent", "R", "object_when_high")
    faints_single = gap("single_goal", "agent", "faints")
    values["AE-1"] = {"learning": learning, "charge_when_low": low_charge, "object_when_high": high_object, "single_goal_faints": faints_single}
    out["AE-1"] = (valid and learning >= 0.2 * delta and ge(low_charge, 0.8) and ge(high_object, 0.8) and ge(faints_single, 1.0)
                   and seeds_ok("single_goal", "agent", "faints")) if valid else None
    # AE-2
    body_after = pooled(reports, "agent", "M", "body_after_change")
    frozen_gap = ret("agent", "M") - ret("frozen_body", "M")
    values["AE-2"] = {"body_after_change": body_after, "frozen_gap_M": frozen_gap}
    out["AE-2"] = (valid and ge(body_after, 0.85) and frozen_gap >= 0.2 * delta and seeds_ok("agent", "frozen_body", "return", "M")) if valid else None
    names = ["RPT-1", "RPT-2", "GWT-1", "GWT-2", "GWT-3", "GWT-4", "HOT-1", "HOT-2", "HOT-3", "HOT-4", "AST-1", "PP-1", "AE-1", "AE-2"]
    out = {k: (bool(v) if v is not None else None) for k, v in out.items()}
    out["passed"] = sum(1 for n in names if out[n])
    out["global"] = all(out[n] for n in names)
    return out, values


# ---------------------------------------------------------------------------------------------- driver

def run_seed(seed, root, childhood_lives=CHILDHOOD_LIVES, updates=UPDATES, batch=BATCH, lives=TEST_LIVES):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log = lambda msg: print(msg, flush=True)
    params, info = childhood(seed, childhood_lives, log)
    t_child = time.time()
    params, history = reinforce(params, seed, updates, batch, log=log)
    t_rl = time.time()
    params.save(root / f"params-{seed}.json")
    evaluation, records = evaluate(params, seed, lives=lives, log=log)
    for name, recs in records.items():
        (root / f"lives-{seed}-{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    report = {"seed": seed, "protocol": "docs/INDICATOR_AGENT_PROTOCOL.md",
              "settings": {"childhood_lives": childhood_lives, "updates": updates, "batch": batch, "test_lives": lives},
              "training": {**info, "reinforce": history}, "quality_space": quality_space(params.hue_code),
              "evaluation": evaluation,
              "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob(f"*-{seed}*"))},
              "timing": {"childhood": round(t_child - started, 1), "reinforce": round(t_rl - t_child, 1),
                         "evaluation": round(time.time() - t_rl, 1)},
              "machine": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__}}
    (root / f"report-{seed}.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/indicator-agent")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--childhood", type=int, default=CHILDHOOD_LIVES)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lives", type=int, default=TEST_LIVES)
    parser.add_argument("--jobs", type=int, default=1)
    a = parser.parse_args(argv)
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as pool:
            reports = list(pool.map(run_seed, a.seeds, [a.out] * len(a.seeds), [a.childhood] * len(a.seeds),
                                    [a.updates] * len(a.seeds), [a.batch] * len(a.seeds), [a.lives] * len(a.seeds)))
    else:
        reports = [run_seed(s, a.out, a.childhood, a.updates, a.batch, a.lives) for s in a.seeds]
    verdict, values = criteria(reports)
    summary = {"seeds": a.seeds, "criteria": verdict, "values": values}
    Path(a.out, "criteria.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
