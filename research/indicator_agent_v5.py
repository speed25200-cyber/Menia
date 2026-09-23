"""Version 5 of the indicator agent: allostatic arbitration.

The agent learns from its own records a model of its needs: how much its energy and satiety fall
per step, the energy held on the charger, the satiety and the reward an object brings as a function
of the value the agent lends it, and the steps needed to reach a target at a given expected
distance. At every step it simulates with this model, for each goal followed by each second goal,
its needs and rewards over 16 steps, and pursues the first goal of the best sequence (regulation by
anticipation, Sterling 2012). The model starts empty: nothing is foreseen, and the agent stays.
While learning, exploration goals are held until reached. In the workspace, content that would
change the goal wins over content that would only change the action; the alarms of version 4 keep
their priority. Everything else is version 4. Protocol: docs/INDICATOR_AGENT_V5_PROTOCOL.md.

Amendment 1, after the first development run: the charger leaves for a free square once used (world
flag charger_moves); the model learns the energy on arriving at the charger and, if it ever sees
it, the change of energy while staying on it, instead of assuming that the charger holds energy.
"""
import numpy as np
from .sense_atelier import STAY, N_ACTIONS
from .indicator_agent import Agent, MODULES, DIST
from .indicator_agent_v2 import goal_features, plan, HORIZON, GAMMA
from .indicator_agent_v4 import AgentV4, drive

EXPLORE = 0.05
TIE = 1e-9


class NeedModel:
    """What the agent has learned about its needs. Empty: no decay and no effect, so every goal looks alike."""

    def __init__(self, decay=(0.0, 0.0), charge=None, food=(0.0, 0.0), reward=(0.0, 0.0), travel=(1.0, 0.0), hold=None):
        self.decay = np.asarray(decay, dtype=float)
        self.charge = None if charge is None else float(charge)  # energy on arriving at the charger
        self.hold = None if hold is None else float(hold)  # change of energy per step while staying on it, if ever seen
        self.food = np.asarray(food, dtype=float)  # satiety gain = food[0] * value + food[1]
        self.reward = np.asarray(reward, dtype=float)  # reward on arrival = reward[0] * value + reward[1]
        self.travel = np.asarray(travel, dtype=float)  # steps = travel[0] + travel[1] * expected distance

    def to_json(self):
        return {"decay": self.decay.tolist(), "charge": self.charge, "food": self.food.tolist(),
                "reward": self.reward.tolist(), "travel": self.travel.tolist(), "hold": self.hold}

    @classmethod
    def from_json(cls, value):
        return cls(**value) if value else cls()

    def steps(self, distance):
        return max(1, int(round(float(self.travel[0] + self.travel[1] * distance))))


def simulate(model, e, f, legs, on_charger, horizon=HORIZON, gamma=GAMMA):
    """Discounted predicted rewards minus the drive over the horizon, for legs (kind, steps, value), then staying."""
    total, disc, s, here = 0.0, 1.0, 0, on_charger
    for kind, steps, v in list(legs) + [("stay", horizon, 0.0)]:
        if kind != "stay":
            here = False
        for i in range(steps):
            if s >= horizon:
                return total
            if here and model.hold is not None:
                e = float(np.clip(e + model.hold, 0.0, 1.0))
            else:
                e = max(e - model.decay[0], 0.0)
            f = max(f - model.decay[1], 0.0)
            reward = 0.0
            if i == steps - 1 and kind == "charger":
                here = True
                if model.charge is not None:
                    e = model.charge
            if i == steps - 1 and kind == "food":
                reward = float(model.reward[0] * v + model.reward[1])
                f = min(1.0, f + max(0.0, float(model.food[0] * v + model.food[1])))
            total += disc * (reward - drive(e, f))
            disc *= gamma
            s += 1
    return total


def _line(x, y, l2=1e-3):
    """Least squares y = a x + b, lightly regularised; zeros without enough data."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 5:
        return [0.0, 0.0]
    A = np.stack([x, np.ones_like(x)], axis=1)
    return np.linalg.solve(A.T @ A + l2 * np.eye(2), A.T @ y).tolist()


def need_observations(life):
    """What one life teaches about the needs, from the agent's own records only."""
    recs = [rec for rec, _ in life["steps"]]
    rewards = life["rewards"]
    out = {"decay_e": [], "decay_f": [], "charge": [], "hold": [], "food": [], "travel": []}
    for t in range(len(recs) - 1):
        a, b = recs[t], recs[t + 1]
        charger = a["charger"]
        de = a["energy_estimate"] - b["energy_estimate"]
        df = a["satiety_estimate"] - b["satiety_estimate"]
        if b["pos_belief"] != charger and de >= 0:
            out["decay_e"].append(de)
        if df >= 0:
            out["decay_f"].append(df)
        if b["pos_belief"] == charger and a["pos_belief"] != charger:
            out["charge"].append(b["energy_estimate"])  # arriving on the charger of the step before
        if a["pos_belief"] == charger and b["pos_belief"] == charger and b["charger"] == charger:
            out["hold"].append(-de)  # staying on a charger that stayed
        g = a["goal"]
        if isinstance(g, int) and b["pos_belief"] == g and not b["presence"][g] and a.get("goal_value") is not None:
            out["food"].append((a["goal_value"], a["satiety_estimate"], b["satiety_estimate"], float(rewards[t])))
    t = 0
    while t < len(recs):
        g = recs[t]["goal"]
        if (g == "charger" or isinstance(g, int)) and recs[t].get("goal_distance") is not None:
            target = recs[t]["charger"] if g == "charger" else g
            u = t
            while u + 1 < len(recs) and recs[u + 1]["pos_belief"] != target and recs[u + 1]["goal"] == g:
                u += 1
            if u + 1 < len(recs) and recs[u + 1]["pos_belief"] == target:
                out["travel"].append((recs[t]["goal_distance"], u + 1 - t))
            t = u + 1
        else:
            t += 1
    return out


def fit_model(obs):
    """The need model from pooled observations; an effect never observed stays empty."""
    decay = [float(np.mean(obs["decay_e"])) if obs["decay_e"] else 0.0,
             float(np.mean(obs["decay_f"])) if obs["decay_f"] else 0.0]
    charge = float(np.mean(obs["charge"])) if obs["charge"] else None
    hold = float(np.mean(obs["hold"])) if obs["hold"] else None
    food_rows = [(v, b - a + decay[1]) for v, a, b, _ in obs["food"] if a + v < 0.95]
    food = _line([r[0] for r in food_rows], [r[1] for r in food_rows])
    reward = _line([r[0] for r in obs["food"]], [r[3] for r in obs["food"]])
    travel = _line([d for d, _ in obs["travel"]], [n for _, n in obs["travel"]]) if len(obs["travel"]) >= 5 else [1.0, 0.0]
    return NeedModel(decay, charge, food, reward, travel, hold)


class AgentV5(AgentV4):
    """Version 4 with a learned need model, goals chosen by anticipation at every step, and goal-first access."""

    def __init__(self, params, variant="agent", seed=0, learn=False, phase="adult", epsilon=0.0):
        super().__init__(params, variant, seed, learn, phase, epsilon)
        self.model = NeedModel.from_json(params.need)
        self.intero_age = 0
        self.explore_goal = None

    # -- anticipation ---------------------------------------------------------------------------------------------

    def _anticipate(self, W, charger, age):
        """Best first goal by simulated needs over the horizon; equal values keep the agent where it is."""
        m = self.model
        _, candidate, best = goal_features(W, charger)
        e0 = float(np.clip(W["intero"][0] - age * m.decay[0], 0.0, 1.0))
        f0 = float(np.clip(W["intero"][1] - age * m.decay[1], 0.0, 1.0))
        b = W["pos"]
        on_charger = int(np.argmax(b)) == charger
        charging = not self._is("single_goal")
        d_c = float(b @ DIST[:, charger])
        value = W["vis"]["values"][candidate] if candidate is not None else None
        d_o = float(b @ DIST[:, candidate]) if candidate is not None else None
        legs = {"stay": ("stay", 1, 0.0)}
        if charging and not on_charger:
            legs["charger"] = ("charger", m.steps(d_c), 0.0)
        if candidate is not None:
            legs[candidate] = ("food", m.steps(d_o), value)
        values = {}
        for first, leg in legs.items():
            seconds = [None]
            if first == "stay":
                seconds += [legs[g] for g in legs if g != "stay"]
            if first == "charger" and candidate is not None:
                seconds.append(("food", m.steps(DIST[charger, candidate]), value))
            if first == candidate and charging:
                seconds.append(("charger", m.steps(DIST[candidate, charger]), 0.0))
            values[first] = max(simulate(m, e0, f0, [leg] + ([second] if second else []), on_charger) for second in seconds)
        top = max(values.values())
        tied = [g for g in values if values[g] >= top - TIE]
        goal = "stay" if "stay" in tied else tied[0]
        return goal, values, candidate, best, (e0, f0)

    def _decision(self, W, charger, age=0):
        goal = self._anticipate(W, charger, age)[0]
        return goal, STAY if goal == "stay" else plan(W, charger if goal == "charger" else goal)

    # -- workspace: alarms, then the goal first -------------------------------------------------------------------

    def _attend(self, contents, obs, rec):
        if any(self._is(v) for v in ("unlimited", "random", "round_robin")) or self.phase == "childhood":
            Agent._attend(self, contents, obs, rec)
        else:
            alarm = self._alarm(contents)
            rec["alarm"] = alarm
            salience = self._salience(contents)
            if alarm is not None:
                chosen = alarm
                rec["changes"] = []
            else:
                later = self.intero_age + 1
                base = self._decision(self.W, obs["charger"], later)
                goal_change, any_change = {}, {}
                for m in MODULES:
                    trial = dict(self.W)
                    trial[m] = contents[m]
                    decision = self._decision(trial, obs["charger"], 0 if m == "intero" else later)
                    goal_change[m] = decision[0] != base[0]
                    any_change[m] = decision != base
                chosen = max(MODULES, key=lambda m: (goal_change[m], any_change[m], round(salience[m] + self.age[m] / 8, 9)))
                rec["changes"] = [m for m in MODULES if any_change[m]]
                rec["goal_changes"] = [m for m in MODULES if goal_change[m]]
            for m in MODULES:
                self.age[m] = min(self.age[m] + 1, 8)
            content = contents[chosen]
            self.W[chosen] = {"values": list(content["values"]), "presence": list(content["presence"])} if chosen == "vis" \
                else content.copy()
            self.age[chosen] = 0
            rec["writers"] = [chosen]
            rec["salience"] = {m: round(v, 6) for m, v in salience.items()}
        self.intero_age = 0 if "intero" in rec["writers"] else self.intero_age + 1

    # -- goals ----------------------------------------------------------------------------------------------------

    def _policy(self, obs, rec):
        charger = obs["charger"]
        believed = int(np.argmax(self.W["pos"]))
        goal, values, candidate, best, needs = self._anticipate(self.W, charger, self.intero_age)
        exploring = False
        if self.phase == "childhood":
            goal = "random"
        elif self.learn:
            g = self.explore_goal
            if g is not None and (g == "stay" or (g == "charger" and believed == charger)
                                  or (isinstance(g, int) and (not self.W["vis"]["presence"][g] or believed == g))):
                self.explore_goal = None
            if self.explore_goal is None and self.rng.random() < EXPLORE:
                options = [o for o in values]
                self.explore_goal = options[int(self.rng.integers(len(options)))]
            if self.explore_goal is not None:
                goal, exploring = self.explore_goal, True
        self.goal = goal
        if goal == "random":
            action = int(self.rng.integers(N_ACTIONS))
        elif goal == "stay":
            action = STAY
        else:
            action = plan(self.W, charger if goal == "charger" else goal)
        target = charger if goal == "charger" else goal if isinstance(goal, int) else None
        rec.update(goal=goal, candidate=candidate, best=best, decided=not exploring, exploring=exploring, charger=charger,
                   plan_values={str(k): round(v, 6) for k, v in values.items()}, plan_needs=[round(x, 6) for x in needs],
                   goal_value=(None if not isinstance(goal, int) else self.W["vis"]["values"][goal]),
                   goal_distance=(None if target is None else round(float(self.W["pos"] @ DIST[:, target]), 6)),
                   presence=list(self.presence))
        return action
