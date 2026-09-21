"""Finite sensor-maintenance POMDP; no claim about subjective experience.

Two independent planners: recursion on scalar beliefs and conditional-cost
alpha vectors. State order is bad, good; actions answer, abstain, maintain.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np


ACTIONS = ("answer", "abstain", "maintain")


@dataclass(frozen=True)
class Model:
    good_accuracy: float = .95
    bad_accuracy: float = .55
    diagnostic_accuracy: float = .85
    wear: float = .15
    restoration: float = .95
    abstain_cost: float = .28
    maintenance_cost: float = .55
    effective_maintenance: bool = True

    def transition(self, action):
        if action == 2 and self.effective_maintenance:
            return np.array([[1-self.restoration, self.restoration]]*2)
        return np.array([[1., 0.], [self.wear, 1-self.wear]])

    def costs(self):
        return np.array([[1-self.bad_accuracy, 1-self.good_accuracy],
                         [self.abstain_cost]*2, [self.maintenance_cost]*2])


def update_belief(prior, diagnostic, accuracy, use_diagnostic=True):
    """Bayes update on a noisy observation of the sensor's own state."""
    if not use_diagnostic:
        return float(prior)
    good_likelihood = accuracy if diagnostic else 1-accuracy
    bad_likelihood = 1-good_likelihood
    evidence = prior*good_likelihood + (1-prior)*bad_likelihood
    if evidence <= 0:
        raise ValueError("Impossible diagnostic under the supplied model")
    return float(prior*good_likelihood/evidence)


def recursive_costs(model, belief, horizon, use_diagnostic):
    """Bellman reference, after the current diagnostic, before the action."""
    if horizon < 1:
        raise ValueError("Positive planning horizon required")
    calls = 0

    @lru_cache(None)
    def value(q, depth):
        nonlocal calls
        calls += 1
        if depth == 0:
            return 0.
        return min(action_costs(q, depth))

    def future(prior, depth):
        if not use_diagnostic:
            return value(prior, depth)
        d = model.diagnostic_accuracy
        p_one = prior*d+(1-prior)*(1-d)
        total = 0.
        for z, probability in ((0, 1-p_one), (1, p_one)):
            if probability > 0:
                total += probability*value(update_belief(prior, z, d), depth)
        return total

    def action_costs(q, depth):
        immediate = ((1-q)*(1-model.bad_accuracy)+q*(1-model.good_accuracy),
                     model.abstain_cost, model.maintenance_cost)
        work_prior = q*(1-model.wear)
        work_future = future(work_prior, depth-1)
        maintenance_prior = model.restoration if model.effective_maintenance else work_prior
        maintenance_future = (future(maintenance_prior, depth-1)
                              if model.effective_maintenance else work_future)
        return (immediate[0]+work_future, immediate[1]+work_future,
                immediate[2]+maintenance_future)

    result = np.array(action_costs(float(belief), horizon))
    return result, calls


def lower_envelope(vectors):
    """Prune affine cost functions on 0 <= P(good) <= 1.

    Retain positive-width optimal segments and endpoints; isolated ties need
    not survive. Root actions are pruned separately so action tie-breaking is
    independent of which future policy represents the same value.
    """
    lines = sorted(((float(v[1]-v[0]), float(v[0]), np.array(v)) for v in vectors),
                   key=lambda line: (-line[0], line[1]))
    unique = []
    for line in lines:
        if unique and abs(line[0]-unique[-1][0]) <= 1e-13:
            if line[1] < unique[-1][1]:
                unique[-1] = line
        else:
            unique.append(line)
    hull, starts = [], []
    for slope, intercept, vector in unique:
        start = -float("inf")
        while hull:
            previous_slope, previous_intercept, _ = hull[-1]
            start = (intercept-previous_intercept)/(previous_slope-slope)
            if start > starts[-1]:
                break
            hull.pop()
            starts.pop()
        if not hull:
            start = -float("inf")
        hull.append((slope, intercept, vector))
        starts.append(start)
    ends = starts[1:]+[float("inf")]
    return np.stack([line[2] for line, start, end in zip(hull, starts, ends)
                     if end >= 0 and start <= 1])


class CompiledPlanner:
    """Ordinary finite-horizon POMDP solver, compiled before evaluation."""

    def __init__(self, model, horizon, use_diagnostic):
        if horizon < 1:
            raise ValueError("Positive planning horizon required")
        self.model = model
        self.use_diagnostic = use_diagnostic
        self.by_depth = {}
        self.compilation = []
        previous = np.zeros((1, 2))
        for depth in range(1, horizon+1):
            if use_diagnostic:
                d = model.diagnostic_accuracy
                zero = np.array([d, 1-d])
                one = np.array([1-d, d])
                contingent = np.stack([zero*a+one*b for a, b in product(previous, repeat=2)])
            else:
                contingent = previous
            actions = []
            for action, immediate in enumerate(model.costs()):
                candidates = immediate+contingent@model.transition(action).T
                actions.append(lower_envelope(candidates))
            self.by_depth[depth] = actions
            previous = lower_envelope(np.concatenate(actions))
            self.compilation.append({"depth": depth, "candidates": int(3*len(contingent)),
                                     "retained_by_action": [len(a) for a in actions],
                                     "retained_for_future": len(previous)})

    def evaluate(self, belief, horizon):
        b = np.array([1-belief, belief])
        vectors = self.by_depth[horizon]
        costs = np.array([np.min(v@b) for v in vectors])
        # Consistent priority for numerically tied costs, no random tie noise.
        choice = int(np.flatnonzero(costs <= np.min(costs)+1e-12)[0])
        return choice, costs, sum(len(v) for v in vectors)


class CapacityAgent:
    """Infer a simulated sensor state and act using the compiled policy.

    Only observe() consumes data. decide() never receives a target, true sensor
    state, realised cost, or future random draw. Model parameters are supplied.
    """

    def __init__(self, planner):
        self.planner = planner
        self.belief = .5
        self.reading = None

    def observe(self, diagnostic, reading):
        self.reading = int(reading)
        self.belief = update_belief(self.belief, diagnostic,
                                   self.planner.model.diagnostic_accuracy,
                                   self.planner.use_diagnostic)

    def decide(self, remaining):
        if self.reading is None:
            raise ValueError("Observe the sensor before deciding")
        model = self.planner.model
        horizon = min(remaining, len(self.planner.by_depth))
        action, costs, operations = self.planner.evaluate(self.belief, horizon)
        b = np.array([1-self.belief, self.belief])
        decision = {"belief_good": float(self.belief), "horizon": horizon,
                    "predicted_sensor_accuracy": float(b@np.array([model.bad_accuracy, model.good_accuracy])),
                    "immediate_expected_costs": (model.costs()@b).tolist(),
                    "planned_expected_costs": costs.tolist(),
                    "predicted_next_good": [float(b@model.transition(a)[:, 1]) for a in range(3)],
                    "action": ACTIONS[action], "response": self.reading if action == 0 else None,
                    "online_dot_products": operations}
        self.belief = decision["predicted_next_good"][action]
        self.reading = None
        return action, decision
