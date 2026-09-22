"""Version 2 of the indicator agent: only the arbitration layer changes.

The attention controller writes into the workspace the module whose current content would change
the agent's decision (goal and action); otherwise the one with the largest salience + age / 8.
Nothing is learned in it. The goals compete through three linear value functions, Q(recharge),
Q(object) and Q(stay), fitted by Monte-Carlo on discounted returns (amendment 1: staying was first
fixed at 0, which made every goal with a negative return lose to it). Modules, monitor,
attention schema, hue code and planner are those of version 1. Protocol:
docs/INDICATOR_AGENT_V2_PROTOCOL.md.
"""
import math
import numpy as np
from .sense_atelier import RING, N_MOVE, STAY, N_ACTIONS, motor_delta
from .indicator_agent import Agent, MODULES, DIST, EPISTEMIC

Q_FEATURES = 8
HORIZON = 16
GAMMA = 0.9


def goal_features(W, charger):
    """Needs, distances and best known good object, read from a workspace."""
    vis = W["vis"]
    known = [x for x in range(RING) if vis["presence"][x] and vis["values"][x] is not None]
    best = max(known, key=lambda x: vis["values"][x]) if known else None
    best_value = vis["values"][best] if best is not None else 0.0
    candidate = best if best is not None and best_value > 0 else None
    b = W["pos"]
    e, f = float(W["intero"][0]), float(W["intero"][1])
    phi = np.array([1.0, e, f, (1 - e) ** 2, (1 - f) ** 2, float(b @ DIST[:, charger]) / 4,
                    best_value if candidate is not None else 0.0,
                    float(b @ DIST[:, candidate]) / 4 if candidate is not None else 0.0])
    return phi, candidate, best


def plan(W, target):
    """The version 1 planner, on a given workspace: expected distance minus the epistemic value of moving."""
    b, beta = W["pos"], W["body"]
    uncertainty = float(-(beta * np.log(beta + 1e-12)).sum() / math.log(4))
    costs = []
    for a in range(N_ACTIONS):
        if a == STAY:
            costs.append(float(b @ DIST[:, target]))
            continue
        total = 0.0
        for d in range(4):
            total += beta[d] * float(np.roll(b, motor_delta(d, a)) @ DIST[:, target])
        costs.append(total - EPISTEMIC * uncertainty)
    order = [STAY] + list(range(N_MOVE))
    return min(order, key=lambda a: (round(costs[a], 9), order.index(a)))


class AgentV2(Agent):
    def __init__(self, params, variant="agent", seed=0, learn=False, phase="adult", epsilon=0.0):
        super().__init__(params, variant, seed, learn, phase)
        self.epsilon = epsilon
        self.samples = []

    # -- values -------------------------------------------------------------------------------------------------

    def _values(self, phi, candidate):
        q = self.P.q if self.P.q is not None else np.zeros((3, Q_FEATURES))
        q_charge = float(phi @ q[0])
        if self.variant == "single_goal":
            q_charge = -math.inf
        q_food = float(phi @ q[1]) if candidate is not None else -math.inf
        return q_charge, q_food, float(phi @ q[2])

    def _greedy(self, phi, candidate):
        q_charge, q_food, q_stay = self._values(phi, candidate)
        options = [(q_food, 0, candidate), (q_charge, 1, "charger"), (q_stay, 2, "stay")]
        return max((o for o in options if o[0] > -math.inf), key=lambda o: (round(o[0], 9), -o[1]))[2]

    def _decision(self, W, charger):
        phi, candidate, _ = goal_features(W, charger)
        goal = self._greedy(phi, candidate)
        if goal == "stay":
            return goal, STAY
        return goal, plan(W, charger if goal == "charger" else goal)

    # -- attention by decision relevance -----------------------------------------------------------------------

    def _attend(self, contents, obs, rec):
        if self.variant in ("unlimited", "random", "round_robin") or self.phase == "childhood":
            return super()._attend(contents, obs, rec)
        salience = self._salience(contents)
        base = self._decision(self.W, obs["charger"])
        changes = {}
        for m in MODULES:
            trial = dict(self.W)
            trial[m] = contents[m]
            changes[m] = self._decision(trial, obs["charger"]) != base
        chosen = [max(MODULES, key=lambda m: (changes[m], round(salience[m] + self.age[m] / 8, 9)))]
        for m in MODULES:
            self.age[m] = min(self.age[m] + 1, 8)
        for m in chosen:
            content = contents[m]
            self.W[m] = {"values": list(content["values"]), "presence": list(content["presence"])} if m == "vis" else content.copy()
            self.age[m] = 0
        rec["writers"] = chosen
        if "vis" in chosen or "intero" in chosen:
            self.decide = True
        rec["changes"] = [m for m in MODULES if changes[m]]
        rec["salience"] = {m: round(v, 6) for m, v in salience.items()}

    # -- goals by value -------------------------------------------------------------------------------------------

    def _policy(self, obs, rec):
        charger = obs["charger"]
        phi, candidate, best = goal_features(self.W, charger)
        believed = int(np.argmax(self.W["pos"]))
        current = self.goal
        if isinstance(current, int) and (not self.W["vis"]["presence"][current] or believed == current):
            self.decide = True
        if current == "charger" and believed == charger:
            self.decide = True
        if current in ("stay", "random"):
            self.decide = True
        decided = self.decide
        q_charge, q_food, q_stay = self._values(phi, candidate)
        if self.phase == "childhood":
            goal = "random"
        elif not self.decide:
            goal = current
        else:
            if self.learn and self.rng.random() < self.epsilon:
                options = ["charger", "stay"] + ([candidate] if candidate is not None else [])
                if self.variant == "single_goal":
                    options.remove("charger")
                goal = options[int(self.rng.integers(len(options)))]
            else:
                goal = self._greedy(phi, candidate)
            if self.learn:
                self.samples.append((obs["t"], 0 if goal == "charger" else 2 if goal == "stay" else 1, phi))
        self.decide = False
        self.goal = goal
        if goal == "random":
            action = int(self.rng.integers(N_ACTIONS))
        elif goal == "stay":
            action = STAY
        else:
            action = plan(self.W, charger if goal == "charger" else goal)
        rec.update(goal=goal, q_charge=None if q_charge == -math.inf else round(q_charge, 6),
                   q_food=None if q_food == -math.inf else round(q_food, 6), q_stay=round(q_stay, 6), best=best,
                   candidate=candidate, decided=decided)
        return action


def discounted_returns(rewards, horizon=HORIZON, gamma=GAMMA):
    rewards = np.asarray(rewards, dtype=float)
    out = np.zeros(len(rewards))
    for t in range(len(rewards)):
        window = rewards[t:t + horizon]
        out[t] = float((window * gamma ** np.arange(len(window))).sum())
    return out


def fit_values(X, y, l2=1e-2):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    return np.linalg.solve(X.T @ X + l2 * np.eye(X.shape[1]), X.T @ y)
