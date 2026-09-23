"""Version 4 of the indicator agent: homeostatic priority.

Two alarms take the workspace before the version 2 controller: a need read by Intero below 0.35
that the workspace does not reflect, and a Body belief that has moved away from the workspace's by
more than 0.5 in total variation. The goal values of version 3 are fitted on an internal reward,
the world's reward minus the drive D = (1 - e)^2 + (1 - f)^2 felt through the Intero estimates
(homeostatic reinforcement learning, Keramati and Gutkin 2014). Staying is a commitment like the
other goals, and goals of exactly equal value are drawn at random. Everything else is version 3.
Protocol: docs/INDICATOR_AGENT_V4_PROTOCOL.md.
"""
import math
import numpy as np
from .sense_atelier import STAY, N_ACTIONS
from .indicator_agent import MODULES, total_variation
from .indicator_agent_v2 import AgentV3, plan

ALARM = 0.35
ALARM_STALE = 0.1
BODY_ALARM = 0.5


def drive(energy, satiety):
    return (1.0 - energy) ** 2 + (1.0 - satiety) ** 2


def internal_rewards(life):
    """r_t - D_{t+1}, the drive read from the Intero estimates of the next step (the last step keeps its own)."""
    recs = [rec for rec, _ in life["steps"]]
    out = []
    for t, reward in enumerate(life["rewards"]):
        after = recs[min(t + 1, len(recs) - 1)]
        out.append(float(reward) - drive(after["energy_estimate"], after["satiety_estimate"]))
    return out


class AgentV4(AgentV3):
    """Version 3 with alarms that take the workspace, drive-based values, a committed stay and random ties."""

    def _alarm(self, contents):
        sensed, held = contents["intero"], self.W["intero"]
        for i in range(2):
            if sensed[i] < ALARM and (held[i] >= ALARM or held[i] - sensed[i] > ALARM_STALE):
                return "intero"
        if total_variation(contents["body"], self.W["body"]) > BODY_ALARM:
            return "body"
        return None

    def _attend(self, contents, obs, rec):
        if any(self._is(v) for v in ("unlimited", "random", "round_robin")) or self.phase == "childhood":
            return super()._attend(contents, obs, rec)
        alarm = self._alarm(contents)
        rec["alarm"] = alarm
        if alarm is None:
            return super()._attend(contents, obs, rec)
        salience = self._salience(contents)
        for m in MODULES:
            self.age[m] = min(self.age[m] + 1, 8)
        self.W[alarm] = contents[alarm].copy()
        self.age[alarm] = 0
        rec["writers"] = [alarm]
        if alarm == "intero":
            self.decide = True
        rec["changes"] = []
        rec["salience"] = {m: round(v, 6) for m, v in salience.items()}

    def _greedy_tied(self, phi, candidate):
        """The greedy goal; goals of exactly equal value are drawn at random."""
        q_charge, q_food, q_stay = self._values(phi, candidate)
        options = [(q, g) for q, g in ((q_food, candidate), (q_charge, "charger"), (q_stay, "stay")) if q > -math.inf]
        top = max(round(q, 9) for q, _ in options)
        tied = [g for q, g in options if round(q, 9) == top]
        return tied[0] if len(tied) == 1 else tied[int(self.rng.integers(len(tied)))]

    def _policy(self, obs, rec):
        charger = obs["charger"]
        phi, candidate, best = self._features(self.W, charger)
        believed = int(np.argmax(self.W["pos"]))
        current = self.goal
        if isinstance(current, int) and (not self.W["vis"]["presence"][current] or believed == current):
            self.decide = True
        if current == "charger" and believed == charger:
            self.decide = True
        if current == "random":
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
                if self._is("single_goal"):
                    options.remove("charger")
                goal = options[int(self.rng.integers(len(options)))]
            else:
                goal = self._greedy_tied(phi, candidate)
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
