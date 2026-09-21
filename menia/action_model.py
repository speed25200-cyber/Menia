"""Online categorical action-effect learning, independent of a language model.

Effects are measured displacements. No action-to-direction mapping is provided.
The recent sample window adapts to changes; forecast errors are scored before
the observed transition is added. This is statistical learning, not a neural AST.
"""
from collections import Counter, deque
import math


class ActionModel:
    def __init__(self, actions, *, window=16, prior=0.1):
        if not actions or len(set(actions)) != len(actions) or not all(isinstance(a, str) and a for a in actions):
            raise ValueError("Distinct action names required")
        if type(window) is not int or window < 1 or not math.isfinite(prior) or prior <= 0:
            raise ValueError("Invalid learning parameters")
        self.actions = tuple(actions)
        self.window = window
        self.prior = prior
        self.effects = tuple((x, y) for x in (-1, 0, 1) for y in (-1, 0, 1))
        self.samples = {a: deque(maxlen=window) for a in actions}
        self.errors = []
        self.learning_enabled = True

    def distribution(self, action):
        counts = Counter(self.samples[action])
        total = len(self.samples[action]) + self.prior*len(self.effects)
        return [(effect, (counts[effect]+self.prior)/total) for effect in self.effects]

    def forecast(self, action):
        distribution = self.distribution(action)
        effect, confidence = max(distribution, key=lambda item: item[1])
        return {"action": action, "effect": list(effect), "confidence": confidence,
                "samples": len(self.samples[action]),
                "distribution": [{"effect": list(e), "p": p} for e, p in distribution]}

    def learn(self, action, effect, forecast):
        effect = tuple(effect)
        if effect not in self.effects or forecast["action"] != action:
            raise ValueError("Unsupported transition or mismatched forecast")
        probabilities = {tuple(row["effect"]): row["p"] for row in forecast["distribution"]}
        if set(probabilities) != set(self.effects) or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1):
            raise ValueError("Invalid precommitted distribution")
        correct = list(effect) == forecast["effect"]
        assessment = {"action": action, "effect": list(effect), "correct": correct,
                      "confidence": forecast["confidence"],
                      "brier": sum((p-(candidate == effect))**2 for candidate, p in probabilities.items()),
                      "surprisal": -math.log(max(probabilities[effect], 1e-12))}
        previous = next((row for row in reversed(self.errors) if row["action"] == action), None)
        changed = (len(self.samples[action]) >= 3 and not correct and previous is not None
                   and not previous["correct"] and previous["effect"] == list(effect))
        assessment["regime_reset"] = changed and self.learning_enabled
        if changed and self.learning_enabled:
            self.samples[action].clear()
            self.samples[action].append(effect)
        self.errors.append(assessment)
        if self.learning_enabled:
            self.samples[action].append(effect)
        return assessment

    def expected_distance(self, action, position, target):
        return sum(p*(abs(position[0]+effect[0]-target[0])+abs(position[1]+effect[1]-target[1]))
                   for effect, p in self.distribution(action))

    def state(self):
        return {"actions": list(self.actions), "window": self.window, "prior": self.prior,
                "samples": {a: [list(e) for e in values] for a, values in self.samples.items()},
                "errors": self.errors.copy(), "learning_enabled": self.learning_enabled}

    @classmethod
    def from_state(cls, state):
        model = cls(state["actions"], window=state["window"], prior=state["prior"])
        for action in model.actions:
            values = state["samples"][action]
            if len(values) > model.window:
                raise ValueError("Too many stored transitions")
            for value in values:
                effect = tuple(value)
                if effect not in model.effects:
                    raise ValueError("Invalid stored effect")
                model.samples[action].append(effect)
        model.errors = list(state["errors"])
        model.learning_enabled = state["learning_enabled"]
        return model
