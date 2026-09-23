"""Exploratory diagnostics of version 5 that motivate version 6 (declared in docs/INDICATOR_AGENT_V6_PROTOCOL.md).

All on the published agent of the second development run of version 5 (seed 19), without retraining, on 200
test lives per set; no confirmatory seed is used. The prototypes below are not version 6: they are grafted on
the version 5 agent to size the effects before the protocol was written.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
from . import indicator_agent_v5 as v5
from .indicator_agent import Params, DIST, EPISTEMIC
from .indicator_experiment import SETS, life_seed, best_square
from .sense_atelier import SenseAtelier, RING, N_ACTIONS, N_MOVE, STAY, in_band, motor_delta, value

DEV = 19
KEEP = 0.02
BIND = 0.8
AVOID = 4.0
PRIOR_VALUE = -0.265  # mean value of a hue drawn uniformly on the circle


class ConsequenceMonitor(v5.AgentV5):
    """Prototype: the broadcast position is judged by the recharge or meal it predicts."""

    acted_on = None
    meta_alarm = False

    def _pos(self, obs, rec):
        super()._pos(obs, rec)
        self.meta_alarm = False
        rec["violation"] = None
        acted, self.acted_on = self.acted_on, None
        if acted is None or self._is("constant_gain"):
            return
        square, charger, present = acted
        missed = None
        if square == charger and obs["charger"] == charger and obs["energy"] < 0.8:
            missed = "charge"
        elif present and obs["presence"][square] and square not in obs["onsets"] and obs["reward"] == 0:
            missed = "food"
        if missed:
            self.b[square] *= KEEP
            self.b /= self.b.sum()
            self.meta_alarm = True
            rec["violation"] = missed

    def _alarm(self, contents):
        return "pos" if self.meta_alarm else super()._alarm(contents)

    def _policy(self, obs, rec):
        action = super()._policy(obs, rec)
        x = int(np.argmax(self.W["pos"]))
        self.acted_on = (x, obs["charger"], bool(self.W["vis"]["presence"][x]))
        return action


class RevisionOnly(ConsequenceMonitor):
    """Prototype: the same revision of the Pos belief, without giving the revised content an alarm's priority."""

    def _alarm(self, contents):
        return v5.AgentV5._alarm(self, contents)


class GatedBinding(v5.AgentV5):
    """Prototype: a hue is bound only when the attention schema is confident of where the spotlight landed."""

    def _schema(self, obs, rec):
        estimate = super()._schema(obs, rec)
        confidence = rec.get("schema_confidence")
        if confidence is not None and confidence < BIND:
            rec["unbound"] = True
            return None
        return estimate


class Both(ConsequenceMonitor, GatedBinding):
    """Prototype: both changes, the revised position taking the workspace before any other alarm."""


class BothAlarmLast(Both):
    """Prototype: both changes, the revised position taking the workspace only when no other alarm does."""

    def _alarm(self, contents):
        alarm = v5.AgentV5._alarm(self, contents)
        return "pos" if alarm is None and self.meta_alarm else alarm


class BothRevisionOnly(RevisionOnly, GatedBinding):
    """Prototype: both changes, the revised position competing for the workspace like any content."""


def plan_avoiding(W, target):
    """Prototype route planner: expected distance plus AVOID times the expected loss of landing on another object."""
    b, beta, vis = W["pos"], W["body"], W["vis"]
    bad = np.zeros(RING)
    for x in range(RING):
        if vis["presence"][x] and x != target:
            v = vis["values"][x]
            bad[x] = max(0.0, -(PRIOR_VALUE if v is None else v))
    uncertainty = float(-(beta * np.log(beta + 1e-12)).sum() / math.log(4))
    costs = []
    for a in range(N_ACTIONS):
        if a == STAY:
            costs.append(float(b @ DIST[:, target]))
            continue
        total = 0.0
        for d in range(4):
            land = np.roll(b, motor_delta(d, a))
            total += beta[d] * (float(land @ DIST[:, target]) + AVOID * float(land @ bad))
        costs.append(total - EPISTEMIC * uncertainty)
    order = [STAY] + list(range(N_MOVE))
    return min(order, key=lambda a: (round(costs[a], 9), order.index(a)))


def lives(params, cls, variant, set_name, n):
    base, mode = SETS[set_name]
    set_index = list(SETS).index(set_name)
    for i in range(n):
        env = SenseAtelier(life_seed(base, i), mode, n_objects=3, charger_moves=True)
        agent = cls(params, variant, seed=DEV * 1_000_000 + set_index * 10_000 + i)
        obs, truth = env.reset()
        steps, total = [], 0.0
        for _ in range(48):
            action, intent, rec = agent.step(obs)
            steps.append((rec, truth))
            obs, truth = env.step(action, intent)
            total += obs["reward"]
            steps[-1] = (rec, steps[-1][1], truth)
        yield steps, total


def recharge_failures(params, cls, n):
    """Set R, true energy below 0.3: rate of the charger goal, and why not."""
    cats, total, ok = Counter(), 0, 0
    for steps, _ in lives(params, cls, "agent", "R", n):
        for rec, truth, _ in steps:
            if truth["energy"] >= 0.3:
                continue
            total += 1
            if rec["goal"] == "charger":
                ok += 1
            elif "charger" not in rec["plan_values"]:
                cats["believed on the charger" + (", truly" if truth["p"] == truth["charger"] else ", falsely")] += 1
            elif truth["satiety"] < truth["energy"]:
                cats["satiety truly lower"] += 1
            elif rec["plan_needs"][1] < rec["plan_needs"][0]:
                cats["satiety believed lower"] += 1
            elif isinstance(rec["goal"], int):
                cats["close object first"] += 1
            else:
                cats["other: " + str(rec["goal"])] += 1
    return {"recharge_when_low": round(ok / total, 4), "low_steps": total, "failures": dict(cats.most_common())}


def band_conflicts(params, cls, variant, n):
    """Set H, conflicts involving a band object (text of version 1): correct choice, misbinding, causes of errors."""
    causes, k, m, reads, wrong_binding = Counter(), 0, 0, 0, 0
    for steps, _ in lives(params, cls, variant, "H", n):
        written = {}
        for rec, truth, _ in steps:
            objects, vals, goal = truth["objects"], rec["vis_values"], rec["goal"]
            if rec["bound"] is not None:
                reads += 1
                wrong_binding += rec["bound"] != truth["spot"]
                written[rec["bound"]] = (truth["spot"], objects.get(truth["spot"]))
            if not isinstance(goal, int):
                continue
            known = [x for x in objects if vals[x] is not None]
            if not (any(in_band(objects[x]) and value(objects[x]) > 0.3 for x in known)
                    and any(value(objects[x]) < -0.3 for x in known)):
                continue
            m += 1
            best = best_square(objects)
            if goal == best:
                k += 1
                continue
            if goal not in objects:
                causes["goal gone"] += 1
            elif value(objects[goal]) < -0.3:
                causes["negative object chosen"] += 1
            elif vals[best] is None or vals[goal] is None or vals[goal] <= vals[best]:
                causes["other positive, map not at fault"] += 1
            elif float(value(objects[best])) - vals[best] <= 0.3:
                causes["other positive, close values"] += 1
            else:
                w = written.get(best)
                causes["best underestimated: " + ("hue of another object" if w and w[0] != best
                                                  else "hue of an older object" if w and w[1] != objects[best]
                                                  else "value error")] += 1
    return {"choice": round(k / m, 4), "conflict_steps": m, "misbinding": round(wrong_binding / reads, 4),
            "errors": dict(causes.most_common())}


def returns(params, cls, variants, n):
    """Set R: return, bad objects eaten per life, charger goal when energy is low, Pos on read steps."""
    out = {}
    for variant in variants:
        R, bad, low, ok, read, right = [], 0, 0, 0, 0, 0
        for steps, total in lives(params, cls, variant, "R", n):
            R.append(total)
            for rec, truth, after in steps:
                bad += after["eaten"] is not None and value(after["eaten"]) <= 0
                if truth["energy"] < 0.3:
                    low += 1
                    ok += rec["goal"] == "charger"
                if rec.get("surprise") is not None and truth["t"] >= 8:
                    read += 1
                    right += rec["pos_belief"] == truth["p"]
        out[variant] = {"return": round(float(np.mean(R)), 4), "bad_eaten_per_life": round(bad / n, 3),
                        "recharge_when_low": round(ok / max(1, low), 4), "pos_read": round(right / read, 4)}
    if "unlimited" in out and "random" in out:
        delta = out["unlimited"]["return"] - out["random"]["return"]
        out["delta"] = round(delta, 4)
        out["agent_minus_constant_gain"] = round(out["agent"]["return"] - out["constant_gain"]["return"], 4)
        out["agent_minus_constant_gain_in_delta"] = round((out["agent"]["return"] - out["constant_gain"]["return"]) / delta, 4)
    return out


def body_writes(params, cls, n):
    """Set M: share of lives where Corps writes into the workspace at steps 25 to 27 (GWT-4, presence)."""
    base, mode = SETS["M"]
    hits = 0
    for i in range(n):
        env = SenseAtelier(life_seed(base, i), mode, n_objects=3, charger_moves=True)
        agent = cls(params, "agent", seed=DEV * 1_000_000 + 10_000 + i)
        obs, truth = env.reset()
        hit = False
        for _ in range(28):
            action, intent, rec = agent.step(obs)
            hit = hit or (25 <= truth["t"] <= 27 and "body" in rec["writers"])
            obs, truth = env.step(action, intent)
        hits += hit
    return round(hits / n, 4)


def priority(params, n, n_body):
    """How the revised position should reach the workspace: recharge, body writes, return, gap with the constant gain."""
    out = {}
    for name, cls in (("version 5", v5.AgentV5), ("monitor, alarm first", ConsequenceMonitor), ("monitor, no alarm", RevisionOnly),
                      ("binding only", GatedBinding), ("both, alarm first", Both), ("both, alarm last", BothAlarmLast),
                      ("both, no alarm", BothRevisionOnly)):
        r = returns(params, cls, ("agent", "constant_gain"), n)
        out[name] = {"recharge_when_low": r["agent"]["recharge_when_low"], "body_writes_25_27": body_writes(params, cls, n),
                     "return": r["agent"]["return"], "gap_with_constant_gain": round(r["agent"]["return"] - r["constant_gain"]["return"], 4)}
        if name in ("version 5", "binding only", "both, no alarm"):
            out[name][f"body_writes_25_27_{n_body}_lives"] = body_writes(params, cls, n_body)
    return out


def band_value_error(params):
    code = params.hue_code
    thetas = np.linspace(0.15, 0.25, 11)
    est = code.value(code.code(thetas)).ravel()
    return round(float(np.max(np.abs(est - value(thetas)))), 4)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="artifacts/indicator-agent-v5/dev-params-19.json")
    parser.add_argument("--lives", type=int, default=200)
    parser.add_argument("--body-lives", type=int, default=800)
    parser.add_argument("--output", default="artifacts/indicator-agent-v5/diagnostics-v6.json")
    a = parser.parse_args(argv)
    params, n = Params.load(a.params), a.lives
    four = ("agent", "constant_gain", "unlimited", "random")
    out = {"agent": a.params, "lives_per_set": n,
           "recharge": recharge_failures(params, v5.AgentV5, n),
           "band_value_max_error": band_value_error(params),
           "band_conflicts": {v: band_conflicts(params, v5.AgentV5, v, n) for v in ("agent", "random_code")},
           "returns": returns(params, v5.AgentV5, four, n),
           "prototype_consequence_monitor": {"recharge": recharge_failures(params, ConsequenceMonitor, n),
                                             "returns": returns(params, ConsequenceMonitor, ("agent", "constant_gain"), n)},
           "prototype_gated_binding": {v: band_conflicts(params, GatedBinding, v, n) for v in ("agent", "random_code")}}
    planner = v5.plan
    v5.plan = plan_avoiding
    try:
        out["prototype_avoiding_planner"] = returns(params, v5.AgentV5, four, n)
    finally:
        v5.plan = planner
    out["alarm_priority"] = priority(params, n, a.body_lives)
    out["returns"]["agent_minus_constant_gain_prototype_monitor"] = \
        out["prototype_consequence_monitor"]["returns"]["agent"]["return"] - out["prototype_consequence_monitor"]["returns"]["constant_gain"]["return"]
    Path(a.output).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
